//! Is a local Ollama server reachable? JEV gets its language models from Ollama, so the setup
//! window and the in-app Doctor show this. The probe is a plain `GET /api/version` over a TCP
//! socket with short timeouts, only ever to a loopback address (127.0.0.1, ::1 or localhost):
//! a non-loopback `OLLAMA_HOST` is ignored, so the shell never makes a network request.

use std::io::{Read, Write};
use std::net::{SocketAddr, TcpStream, ToSocketAddrs};
use std::time::Duration;

use serde::Serialize;

pub const DEFAULT_ADDR: &str = "127.0.0.1:11434";

#[derive(Debug, Clone, Default, Serialize, PartialEq, Eq)]
#[serde(rename_all = "camelCase")]
pub struct OllamaStatus {
    pub reachable: bool,
    /// e.g. "http://127.0.0.1:11434"
    pub url: String,
    pub version: Option<String>,
    pub error: Option<String>,
}

/// The `host:port` to probe: `OLLAMA_HOST` when it names a loopback host, else the default.
/// Accepts the forms Ollama itself accepts: `host`, `host:port`, `http://host:port`, `:port`.
pub fn loopback_addr(ollama_host: Option<&str>) -> String {
    let Some(raw) = ollama_host.map(str::trim).filter(|s| !s.is_empty()) else { return DEFAULT_ADDR.into() };
    let s = raw.strip_prefix("http://").or_else(|| raw.strip_prefix("https://")).unwrap_or(raw);
    let s = s.split('/').next().unwrap_or("");
    let (host, port) = match s.rsplit_once(':') {
        // "[::1]:11434" or "host:port" (an IPv6 literal without brackets has several colons)
        Some((h, p)) if !h.is_empty() && (h.starts_with('[') || !h.contains(':')) => (h, p),
        Some(("", p)) => ("127.0.0.1", p),
        _ => (s, "11434"),
    };
    let port: u16 = match port.parse() {
        Ok(p) if p > 0 => p,
        _ => return DEFAULT_ADDR.into(),
    };
    match host {
        "127.0.0.1" | "localhost" => format!("127.0.0.1:{port}"),
        "[::1]" | "::1" => format!("[::1]:{port}"),
        // 0.0.0.0 means "listen everywhere"; locally it is reachable on loopback
        "0.0.0.0" => format!("127.0.0.1:{port}"),
        _ => DEFAULT_ADDR.into(),
    }
}

/// Parse an HTTP/1.x response to `/api/version`: `Some(version)` for a 200 with `{"version": ...}`.
pub fn parse_version_response(resp: &str) -> Result<Option<String>, String> {
    let status_line = resp.lines().next().unwrap_or("");
    let code = status_line.split_whitespace().nth(1).unwrap_or("");
    if !status_line.starts_with("HTTP/1.") {
        return Err("not an HTTP response".into());
    }
    if code != "200" {
        return Err(format!("HTTP {code}"));
    }
    let body = resp.split_once("\r\n\r\n").or_else(|| resp.split_once("\n\n")).map(|(_, b)| b).unwrap_or("");
    let v: serde_json::Value =
        serde_json::from_str(body.trim()).map_err(|_| "unexpected reply (not Ollama?)".to_string())?;
    Ok(v.get("version").and_then(|x| x.as_str()).map(|s| s.chars().take(40).collect()))
}

pub fn probe(addr: &str, timeout: Duration) -> OllamaStatus {
    let url = format!("http://{addr}");
    let fail = |e: String| OllamaStatus { reachable: false, url: url.clone(), version: None, error: Some(e) };
    let sock: SocketAddr = match addr.to_socket_addrs().ok().and_then(|mut a| a.next()) {
        Some(a) if a.ip().is_loopback() => a,
        _ => return fail("not a loopback address".into()),
    };
    let mut s = match TcpStream::connect_timeout(&sock, timeout) {
        Ok(s) => s,
        Err(e) => return fail(format!("{e}")),
    };
    let _ = s.set_read_timeout(Some(timeout));
    let _ = s.set_write_timeout(Some(timeout));
    let req =
        format!("GET /api/version HTTP/1.0\r\nHost: {addr}\r\nAccept: application/json\r\nConnection: close\r\n\r\n");
    if let Err(e) = s.write_all(req.as_bytes()) {
        return fail(format!("{e}"));
    }
    let mut buf = Vec::with_capacity(1024);
    let _ = s.take(16 * 1024).read_to_end(&mut buf);
    match parse_version_response(&String::from_utf8_lossy(&buf)) {
        Ok(version) => OllamaStatus { reachable: true, url, version, error: None },
        Err(e) => fail(e),
    }
}

/// Probe the address from the environment (`OLLAMA_HOST`), with 400 ms timeouts.
pub fn probe_default() -> OllamaStatus {
    let host = std::env::var("OLLAMA_HOST").ok();
    probe(&loopback_addr(host.as_deref()), Duration::from_millis(400))
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::net::TcpListener;

    #[test]
    fn ollama_host_is_loopback_only() {
        assert_eq!(loopback_addr(None), "127.0.0.1:11434");
        assert_eq!(loopback_addr(Some("")), "127.0.0.1:11434");
        assert_eq!(loopback_addr(Some("127.0.0.1:11500")), "127.0.0.1:11500");
        assert_eq!(loopback_addr(Some("http://localhost:8080/")), "127.0.0.1:8080");
        assert_eq!(loopback_addr(Some("0.0.0.0")), "127.0.0.1:11434");
        assert_eq!(loopback_addr(Some(":9999")), "127.0.0.1:9999");
        assert_eq!(loopback_addr(Some("[::1]:7000")), "[::1]:7000");
        // never anything off this machine
        assert_eq!(loopback_addr(Some("example.com:11434")), DEFAULT_ADDR);
        assert_eq!(loopback_addr(Some("http://10.0.0.5:11434")), DEFAULT_ADDR);
        assert_eq!(loopback_addr(Some("localhost:notaport")), DEFAULT_ADDR);
    }

    #[test]
    fn parses_version_replies() {
        let ok = "HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"version\":\"0.12.6\"}";
        assert_eq!(parse_version_response(ok), Ok(Some("0.12.6".into())));
        assert!(parse_version_response("HTTP/1.1 404 Not Found\r\n\r\n").unwrap_err().contains("404"));
        assert!(parse_version_response("SSH-2.0-OpenSSH").is_err());
        assert!(parse_version_response("HTTP/1.1 200 OK\r\n\r\n<html>").is_err());
    }

    #[test]
    fn probes_a_fake_server_and_a_closed_port() {
        let l = TcpListener::bind("127.0.0.1:0").unwrap();
        let addr = l.local_addr().unwrap().to_string();
        let t = std::thread::spawn(move || {
            let (mut c, _) = l.accept().unwrap();
            let mut b = [0u8; 512];
            let _ = c.read(&mut b);
            c.write_all(b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n\r\n{\"version\":\"9.9.9\"}").unwrap();
        });
        let st = probe(&addr, Duration::from_millis(800));
        t.join().unwrap();
        assert!(st.reachable, "{st:?}");
        assert_eq!(st.version.as_deref(), Some("9.9.9"));

        let closed = TcpListener::bind("127.0.0.1:0").unwrap();
        let a2 = closed.local_addr().unwrap().to_string();
        drop(closed);
        let st = probe(&a2, Duration::from_millis(300));
        assert!(!st.reachable && st.error.is_some());
        assert!(!probe("93.184.216.34:80", Duration::from_millis(50)).reachable);
        // refused before connecting
    }
}
