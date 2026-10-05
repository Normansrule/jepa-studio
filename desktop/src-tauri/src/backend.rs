//! The local Python backend: session token, the allowed-folders file, spawning
//! `python -m jepa_studio serve`, waiting for its `{"ready": true, "port": N}` line, and killing it.

use std::fs;
use std::io::{BufRead, BufReader, Write};
use std::path::{Path, PathBuf};
use std::process::{Child, ChildStdin, Command, Stdio};
use std::sync::mpsc;
use std::time::Duration;

use base64::Engine;

use crate::setup::{quiet, Logger};

/// How long the backend may take to import torch and start listening. The first start after an
/// install is the slow one (bytecode compilation, antivirus scans of the CUDA DLLs on Windows).
pub const READY_TIMEOUT: Duration = Duration::from_secs(180);

/// 32 random bytes from the OS CSPRNG, base64url without padding (43 characters).
pub fn generate_token() -> Result<String, String> {
    let mut buf = [0u8; 32];
    getrandom::fill(&mut buf).map_err(|e| format!("OS random number generator failed: {e}"))?;
    Ok(base64::engine::general_purpose::URL_SAFE_NO_PAD.encode(buf))
}

/// Parse one stdout line from the backend.
/// * `Ok(Some(port))` for `{"ready": true, "port": N}` with 0 < N < 65536
/// * `Ok(None)` for anything else (warnings, blank lines), which the caller just logs
/// * `Err` if the backend announced its own token: it did not receive ours, so the web view could
///   never authenticate, and the line must not be forwarded anywhere.
pub fn parse_ready_line(line: &str) -> Result<Option<u16>, String> {
    let v: serde_json::Value = match serde_json::from_str(line.trim()) {
        Ok(v) => v,
        Err(_) => return Ok(None),
    };
    if v.get("ready") != Some(&serde_json::Value::Bool(true)) {
        return Ok(None);
    }
    if v.get("token").is_some() {
        return Err("the backend generated its own token (JEPA_STUDIO_TOKEN was not passed through)".into());
    }
    match v.get("port").and_then(|p| p.as_u64()) {
        Some(p) if p > 0 && p < 65536 => Ok(Some(p as u16)),
        _ => Err(format!("the backend reported an invalid port: {}", line.trim())),
    }
}

/// Create the allowed-folders file if missing (grants persist across launches).
pub fn ensure_allowed_file(path: &Path) -> std::io::Result<()> {
    if let Some(d) = path.parent() {
        fs::create_dir_all(d)?;
    }
    fs::OpenOptions::new().create(true).append(true).open(path)?;
    Ok(())
}

/// Append a user-picked folder to the allowed file: canonicalized (symlinks and `..` resolved, no
/// `\\?\` prefix on Windows), must be an existing directory, stored once. Paths that could not be
/// represented as a single line exactly as the backend reads it (newlines, surrounding
/// whitespace, invalid UTF-8) are refused rather than altered.
pub fn append_allowed(file: &Path, picked: &Path) -> Result<PathBuf, String> {
    let canon = dunce::canonicalize(picked).map_err(|e| format!("cannot open {}: {e}", picked.display()))?;
    if !canon.is_dir() {
        return Err(format!("{} is not a folder", canon.display()));
    }
    let s = canon.to_str().ok_or("the folder path is not valid Unicode")?;
    if s.contains(['\n', '\r', '\0']) || s.trim() != s {
        return Err("the folder name contains line breaks or leading/trailing spaces; rename it and try again".into());
    }
    let existing = fs::read_to_string(file).unwrap_or_default();
    let already = existing.lines().any(|l| {
        let l = l.trim();
        !l.is_empty() && (l == s || dunce::canonicalize(l).map(|p| p == canon).unwrap_or(false))
    });
    if !already {
        let mut f = fs::OpenOptions::new().create(true).append(true).open(file).map_err(|e| e.to_string())?;
        let sep = if existing.is_empty() || existing.ends_with('\n') { "" } else { "\n" };
        // One write call per grant so the backend (which re-reads the file per request) never sees
        // half a line.
        f.write_all(format!("{sep}{s}\n").as_bytes()).map_err(|e| e.to_string())?;
    }
    Ok(canon)
}

/// Python run with `-c`: a daemon thread blocks on stdin and hard-exits when it reaches EOF. The
/// shell holds the other end of that pipe, so the backend dies with the shell even if the shell
/// is killed or crashes and never gets to run its own cleanup.
const WATCHDOG: &str = "import os, sys, threading
def _watch():
    try:
        while sys.stdin.buffer.read(1):
            pass
    finally:
        os._exit(0)
threading.Thread(target=_watch, daemon=True).start()
from jepa_studio.cli import main
main(sys.argv[1:])
";

pub struct BackendSpec<'a> {
    pub python: &'a Path,
    /// `true` for the private venv: scrub PYTHONPATH/PYTHONHOME so nothing shadows the install.
    /// `false` for the JEPA_STUDIO_PYTHON developer override, which relies on them.
    pub isolated: bool,
    pub workspace: &'a Path,
    pub allowed_file: &'a Path,
    /// Bundled docs/ folder (README.md sits next to it) that the read-only assistant searches.
    pub docs_dir: Option<&'a Path>,
    pub token: &'a str,
    pub log: &'a Logger,
    pub stderr_log: &'a Path,
}

pub struct Backend {
    child: Child,
    stdin: Option<ChildStdin>,
    pub port: u16,
}

impl Backend {
    pub fn spawn(spec: &BackendSpec) -> Result<Backend, String> {
        fs::create_dir_all(spec.workspace).map_err(|e| e.to_string())?;
        let err_file = fs::OpenOptions::new()
            .create(true)
            .append(true)
            .open(spec.stderr_log)
            .map_err(|e| format!("cannot open {}: {e}", spec.stderr_log.display()))?;
        let mut cmd = quiet(Command::new(spec.python));
        cmd.arg("-c")
            .arg(WATCHDOG)
            .args(["serve", "--root"])
            .arg(spec.workspace)
            .args(["--port", "0"])
            // Secrets and grants go through the environment, never argv (visible in `ps`).
            .env("JEPA_STUDIO_TOKEN", spec.token)
            .env("JEPA_STUDIO_ALLOWED_FILE", spec.allowed_file)
            .env_remove("JEPA_STUDIO_ALLOWED_DIRS")
            .env("PYTHONUNBUFFERED", "1")
            // The allowed file is UTF-8; make Python read it as such on Windows too.
            .env("PYTHONUTF8", "1")
            .current_dir(spec.workspace)
            .stdin(Stdio::piped())
            .stdout(Stdio::piped())
            .stderr(Stdio::from(err_file));
        if let Some(d) = spec.docs_dir {
            cmd.env("JEPA_STUDIO_DOCS_DIR", d);
        }
        if spec.isolated {
            cmd.env_remove("PYTHONPATH").env_remove("PYTHONHOME").env("PYTHONNOUSERSITE", "1");
        }
        spec.log.line(&format!(
            "starting backend: {} -c <watchdog> serve --root {} --port 0",
            spec.python.display(),
            spec.workspace.display()
        ));
        let mut child = cmd.spawn().map_err(|e| format!("could not start {}: {e}", spec.python.display()))?;
        let stdin = child.stdin.take();
        let stdout = child.stdout.take().expect("piped stdout");

        let (tx, rx) = mpsc::channel::<Result<u16, String>>();
        let out_log = spec.stderr_log.to_path_buf();
        std::thread::spawn(move || {
            let mut sent = false;
            let mut log = fs::OpenOptions::new().create(true).append(true).open(&out_log).ok();
            for line in BufReader::new(stdout).lines().map_while(Result::ok) {
                if !sent {
                    match parse_ready_line(&line) {
                        Ok(Some(port)) => {
                            let _ = tx.send(Ok(port));
                            sent = true;
                            continue;
                        }
                        Ok(None) => {}
                        Err(e) => {
                            let _ = tx.send(Err(e));
                            sent = true;
                            continue; // never log a line that may carry a token
                        }
                    }
                }
                if let Some(f) = log.as_mut() {
                    let _ = writeln!(f, "[stdout] {line}");
                }
            }
            if !sent {
                let _ = tx.send(Err("the backend exited before it was ready".into()));
            }
        });

        let mut b = Backend { child, stdin, port: 0 };
        match rx.recv_timeout(READY_TIMEOUT) {
            Ok(Ok(port)) => {
                b.port = port;
                spec.log.line(&format!("backend ready on 127.0.0.1:{port} (pid {})", b.child.id()));
                Ok(b)
            }
            Ok(Err(e)) => {
                b.kill();
                Err(e)
            }
            Err(_) => {
                b.kill();
                Err(format!("the backend did not report ready within {} s", READY_TIMEOUT.as_secs()))
            }
        }
    }

    /// `Some(description)` once the process has exited.
    pub fn try_wait(&mut self) -> Option<String> {
        match self.child.try_wait() {
            Ok(Some(status)) => Some(status.to_string()),
            Ok(None) => None,
            Err(e) => Some(format!("wait failed: {e}")),
        }
    }

    pub fn kill(&mut self) {
        self.stdin.take(); // closing the pipe alone makes the watchdog exit
        let _ = self.child.kill();
        let _ = self.child.wait();
    }
}

impl Drop for Backend {
    fn drop(&mut self) {
        self.kill();
    }
}

/// Last `n` lines of a log file, for error messages.
pub fn log_tail(path: &Path, n: usize) -> String {
    let s = fs::read_to_string(path).unwrap_or_default();
    let lines: Vec<&str> = s.lines().collect();
    lines[lines.len().saturating_sub(n)..].join("\n")
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tmpdir(tag: &str) -> PathBuf {
        let d = std::env::temp_dir().join(format!("jepa-desktop-test-{tag}-{}", std::process::id()));
        let _ = fs::remove_dir_all(&d);
        fs::create_dir_all(&d).unwrap();
        d
    }

    #[test]
    fn token_is_43_url_safe_chars_and_random() {
        let a = generate_token().unwrap();
        let b = generate_token().unwrap();
        assert_eq!(a.len(), 43);
        assert!(a.chars().all(|c| c.is_ascii_alphanumeric() || c == '-' || c == '_'), "{a}");
        assert_ne!(a, b);
        let raw = base64::engine::general_purpose::URL_SAFE_NO_PAD.decode(&a).unwrap();
        assert_eq!(raw.len(), 32);
    }

    #[test]
    fn ready_line_parsing() {
        assert_eq!(parse_ready_line(r#"{"ready": true, "port": 43817}"#), Ok(Some(43817)));
        assert_eq!(parse_ready_line("  {\"port\":1,\"ready\":true}\r\n"), Ok(Some(1)));
        assert_eq!(parse_ready_line("UserWarning: something"), Ok(None));
        assert_eq!(parse_ready_line(""), Ok(None));
        assert_eq!(parse_ready_line(r#"{"ready": false, "port": 5}"#), Ok(None));
        assert_eq!(parse_ready_line(r#"{"ready": "true", "port": 5}"#), Ok(None));
        assert!(parse_ready_line(r#"{"ready": true, "port": 0}"#).is_err());
        assert!(parse_ready_line(r#"{"ready": true, "port": 70000}"#).is_err());
        assert!(parse_ready_line(r#"{"ready": true}"#).is_err());
        assert!(parse_ready_line(r#"{"ready": true, "port": 5, "token": "x"}"#).is_err());
    }

    #[test]
    fn allowed_file_append_is_canonical_and_deduped() {
        let d = tmpdir("allow");
        let file = d.join("allowed-folders.txt");
        ensure_allowed_file(&file).unwrap();
        let data = d.join("data");
        fs::create_dir_all(data.join("sub")).unwrap();

        let p1 = append_allowed(&file, &data).unwrap();
        assert_eq!(p1, dunce::canonicalize(&data).unwrap());
        // Same folder through `..` and a trailing component: stored once.
        let p2 = append_allowed(&file, &data.join("sub").join("..")).unwrap();
        assert_eq!(p1, p2);
        let lines: Vec<String> = fs::read_to_string(&file).unwrap().lines().map(String::from).collect();
        assert_eq!(lines, vec![p1.to_str().unwrap().to_string()]);

        // A second folder is appended on its own line.
        let p3 = append_allowed(&file, &data.join("sub")).unwrap();
        let lines: Vec<String> = fs::read_to_string(&file).unwrap().lines().map(String::from).collect();
        assert_eq!(lines.len(), 2);
        assert_eq!(lines[1], p3.to_str().unwrap());

        // Missing folders and files are refused.
        assert!(append_allowed(&file, &d.join("nope")).is_err());
        fs::write(d.join("f.txt"), "x").unwrap();
        assert!(append_allowed(&file, &d.join("f.txt")).is_err());
        let _ = fs::remove_dir_all(&d);
    }

    #[cfg(unix)]
    #[test]
    fn allowed_file_refuses_line_injection_and_resolves_symlinks() {
        let d = tmpdir("inject");
        let file = d.join("allowed-folders.txt");
        let evil = d.join("a\n/");
        fs::create_dir_all(&evil).unwrap();
        assert!(append_allowed(&file, &evil).is_err());
        let spaced = d.join(" padded ");
        fs::create_dir_all(&spaced).unwrap();
        assert!(append_allowed(&file, &spaced).is_err());
        assert_eq!(fs::read_to_string(&file).unwrap_or_default(), "");

        let real = d.join("real");
        fs::create_dir_all(&real).unwrap();
        std::os::unix::fs::symlink(&real, d.join("link")).unwrap();
        let p = append_allowed(&file, &d.join("link")).unwrap();
        assert_eq!(p, fs::canonicalize(&real).unwrap());
        append_allowed(&file, &real).unwrap();
        assert_eq!(fs::read_to_string(&file).unwrap().lines().count(), 1);
        let _ = fs::remove_dir_all(&d);
    }

    #[test]
    fn allowed_file_appends_after_unterminated_line() {
        let d = tmpdir("unterminated");
        let file = d.join("allowed-folders.txt");
        let other = d.join("other");
        fs::create_dir_all(&other).unwrap();
        fs::write(&file, "/some/manual/entry").unwrap();
        let p = append_allowed(&file, &other).unwrap();
        let s = fs::read_to_string(&file).unwrap();
        assert_eq!(s, format!("/some/manual/entry\n{}\n", p.to_str().unwrap()));
        let _ = fs::remove_dir_all(&d);
    }
}
