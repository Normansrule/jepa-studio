//! The site's HTML carries the *web* tier's CSP in a `<meta http-equiv>` tag (`connect-src 'self'`),
//! which would block the desktop tier's calls to the local backend, because a meta policy and a
//! header policy are both enforced. For HTML served to the desktop webview we therefore drop the
//! meta tag and let Tauri's header policy (tauri.conf.json `app.security.csp`) govern, with its
//! `http://127.0.0.1:*` source narrowed to the one port this session's backend listens on.

/// Remove every `<meta http-equiv="Content-Security-Policy" ...>` tag (case-insensitive, any
/// quoting). Returns `None` when there is nothing to remove.
pub fn strip_meta_csp(html: &str) -> Option<String> {
    let lower = html.to_ascii_lowercase();
    let mut out = String::with_capacity(html.len());
    let mut last = 0;
    let mut pos = 0;
    let mut changed = false;
    while let Some(i) = lower[pos..].find("<meta") {
        let start = pos + i;
        let Some(j) = lower[start..].find('>') else { break };
        let end = start + j + 1;
        let tag = &lower[start..end];
        let compact: String = tag.chars().filter(|c| !c.is_whitespace() && *c != '"' && *c != '\'').collect();
        if compact.contains("http-equiv=content-security-policy") {
            out.push_str(&html[last..start]);
            last = end;
            changed = true;
        }
        pos = end;
    }
    if !changed {
        return None;
    }
    out.push_str(&html[last..]);
    Some(out)
}

/// Replace the `http://127.0.0.1:*` wildcard with the backend's exact port, or remove it while no
/// backend is running (`port == 0`).
pub fn pin_backend_port(csp: &str, port: u16) -> String {
    if port == 0 {
        csp.replace(" http://127.0.0.1:*", "").replace("http://127.0.0.1:*", "")
    } else {
        csp.replace("http://127.0.0.1:*", &format!("http://127.0.0.1:{port}"))
    }
}

/// JEV's artifacts (videos, audio) are fetched with the session token and played from `blob:`
/// URLs. Make sure the policy allows that: add `blob:` to an existing `media-src`, or add
/// `media-src 'self' blob:` when the policy has none (so a stale config cannot silently break
/// playback, and nothing wider than `blob:` is ever added).
pub fn ensure_media_blob(csp: &str) -> String {
    let mut parts: Vec<String> = csp.split(';').map(|p| p.trim().to_string()).filter(|p| !p.is_empty()).collect();
    match parts.iter_mut().find(|p| p.split_whitespace().next() == Some("media-src")) {
        Some(p) => {
            if !p.split_whitespace().any(|t| t == "blob:") {
                p.push_str(" blob:");
            }
        }
        None => parts.push("media-src 'self' blob:".into()),
    }
    parts.join("; ")
}

#[cfg(test)]
mod tests {
    use super::*;

    const SITE_HEAD: &str = r#"<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'self'; connect-src 'self'; object-src 'none'">
<meta name="referrer" content="no-referrer">
<title>jepa-studio</title>"#;

    #[test]
    fn strips_only_the_csp_meta() {
        let out = strip_meta_csp(SITE_HEAD).unwrap();
        assert!(!out.to_lowercase().contains("content-security-policy"));
        assert!(out.contains(r#"<meta charset="utf-8">"#));
        assert!(out.contains(r#"<meta name="referrer" content="no-referrer">"#));
        assert!(out.contains("<title>jepa-studio</title>"));
    }

    #[test]
    fn handles_case_quoting_and_multiple_tags() {
        let html = "<META HTTP-EQUIV=Content-Security-Policy CONTENT=\"x\"><p>a</p><meta http-equiv = 'content-security-policy' content='y' />b";
        assert_eq!(strip_meta_csp(html).unwrap(), "<p>a</p>b");
    }

    #[test]
    fn leaves_other_pages_alone() {
        assert_eq!(strip_meta_csp("<html><head><meta charset=utf-8></head></html>"), None);
        assert_eq!(strip_meta_csp("no tags at all"), None);
        assert_eq!(strip_meta_csp("<meta unterminated"), None);
    }

    #[test]
    fn media_blob_is_allowed_and_nothing_else_widens() {
        let base = "default-src 'self'; img-src 'self' data: blob:; connect-src 'self' http://127.0.0.1:*";
        let out = ensure_media_blob(base);
        assert_eq!(out, "default-src 'self'; img-src 'self' data: blob:; connect-src 'self' http://127.0.0.1:*; media-src 'self' blob:");
        assert_eq!(ensure_media_blob("media-src 'self'"), "media-src 'self' blob:");
        assert_eq!(ensure_media_blob("media-src 'self' blob:;"), "media-src 'self' blob:");
        // pinning still works afterwards, and the media rule survives it
        let pinned = pin_backend_port(&out, 40001);
        assert!(pinned.contains("connect-src 'self' http://127.0.0.1:40001"));
        assert!(pinned.contains("media-src 'self' blob:"));
        assert!(!pinned.contains("127.0.0.1:*"));
    }

    #[test]
    fn the_shipped_policy_allows_blob_media_and_pins() {
        let conf: serde_json::Value = serde_json::from_str(include_str!("../tauri.conf.json")).unwrap();
        let csp = conf["app"]["security"]["csp"].as_str().unwrap();
        assert!(csp.contains("media-src 'self' blob:"), "{csp}");
        assert_eq!(ensure_media_blob(csp), csp.trim_end_matches(';').to_string());
        let pinned = pin_backend_port(csp, 51000);
        assert!(pinned.contains("http://127.0.0.1:51000") && !pinned.contains("127.0.0.1:*"));
        assert!(!csp.contains("unsafe-inline") && !csp.contains("unsafe-eval"));
    }

    #[test]
    fn pins_the_port() {
        let csp = "default-src 'self'; connect-src 'self' http://127.0.0.1:* ipc: http://ipc.localhost";
        assert_eq!(
            pin_backend_port(csp, 51234),
            "default-src 'self'; connect-src 'self' http://127.0.0.1:51234 ipc: http://ipc.localhost"
        );
        assert_eq!(pin_backend_port(csp, 0), "default-src 'self'; connect-src 'self' ipc: http://ipc.localhost");
    }
}
