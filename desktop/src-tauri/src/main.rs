//! jepa-studio desktop shell.
//!
//! 1. Shows a small setup window (served from the `jepa-setup:` scheme, not from site/).
//! 2. Finds or creates the private Python environment (setup.rs).
//! 3. Starts `python -m jepa_studio serve` on 127.0.0.1 with a per-session token (backend.rs).
//! 4. Opens site/app.html with `window.__JEPA_DESKTOP__ = {port, token}` injected before any page
//!    script runs, and the page CSP narrowed to that one backend port (csp.rs).
//! 5. Kills the backend when the main window goes away or the app exits.
//!
//! The web view gets exactly two commands: `pick_data_folder` and `backend_status`.

#![cfg_attr(not(debug_assertions), windows_subsystem = "windows")]

mod backend;
mod csp;
mod setup;

use std::borrow::Cow;
use std::path::PathBuf;
use std::sync::atomic::{AtomicU16, Ordering};
use std::sync::{Arc, Mutex};
use std::time::Duration;

use serde::Serialize;
use tauri::http::{header, Response};
use tauri::path::BaseDirectory;
use tauri::webview::NewWindowResponse;
use tauri::{
    AppHandle, Emitter, Manager, RunEvent, State, Url, WebviewUrl, WebviewWindow, WebviewWindowBuilder, WindowEvent,
};
use tauri_plugin_dialog::{DialogExt, MessageDialogKind};

use backend::{Backend, BackendSpec};

const MAIN: &str = "main";
const SETUP: &str = "setup";
const PROGRESS_EVENT: &str = "setup://progress";

// ------------------------------------------------------------------ shared state

/// What `backend_status` returns and what the setup window renders (camelCase JSON).
#[derive(Debug, Clone, Default, Serialize)]
#[serde(rename_all = "camelCase")]
struct Status {
    /// "setup" | "starting" | "ready" | "error"
    phase: String,
    message: String,
    progress: Option<f64>,
    line: Option<String>,
    error: Option<String>,
    port: Option<u16>,
    accelerator: Option<String>,
    torch_index: Option<String>,
    python: Option<String>,
    log_dir: Option<String>,
}

#[derive(Default)]
struct Shared {
    status: Mutex<Status>,
    backend: Mutex<Option<Backend>>,
    allowed_file: Mutex<Option<PathBuf>>,
}

type SharedState = Arc<Shared>;

fn report(app: &AppHandle, f: impl FnOnce(&mut Status)) {
    let st = app.state::<SharedState>();
    let snapshot = {
        let mut s = st.status.lock().unwrap();
        f(&mut s);
        s.clone()
    };
    let _ = app.emit_to(SETUP, PROGRESS_EVENT, snapshot);
}

fn kill_backend(app: &AppHandle) {
    if let Some(st) = app.try_state::<SharedState>() {
        if let Some(mut b) = st.backend.lock().unwrap().take() {
            b.kill();
        }
    }
}

// ------------------------------------------------------------------ commands

/// Opens the native folder picker, records the canonical folder in the allowed-folders file the
/// backend reads, and returns it (or `null` if the user cancelled). JS cannot name a path here:
/// the only way a folder becomes readable by the backend is the user picking it.
#[tauri::command]
async fn pick_data_folder(window: WebviewWindow, state: State<'_, SharedState>) -> Result<Option<String>, String> {
    if window.label() != MAIN {
        return Err("not allowed from this window".into());
    }
    let allowed = state.allowed_file.lock().unwrap().clone().ok_or("the backend is not running yet")?;
    let (tx, mut rx) = tauri::async_runtime::channel(1);
    window.dialog().file().set_title("Choose a data folder for jepa-studio").set_parent(&window).pick_folder(
        move |p| {
            let _ = tx.try_send(p);
        },
    );
    let Some(picked) = rx.recv().await.flatten() else { return Ok(None) };
    let path = picked.into_path().map_err(|e| e.to_string())?;
    let canon = backend::append_allowed(&allowed, &path)?;
    Ok(Some(canon.to_string_lossy().into_owned()))
}

/// Setup/backend state: phase, message, progress, port, accelerator, torch index, python, log dir.
#[tauri::command]
fn backend_status(state: State<'_, SharedState>) -> Status {
    state.status.lock().unwrap().clone()
}

// ------------------------------------------------------------------ setup window (jepa-setup:)

const SPLASH_HTML: &str = include_str!("../splash/index.html");
const SPLASH_CSS: &str = include_str!("../splash/splash.css");
const SPLASH_JS: &str = include_str!("../splash/splash.js");
const SPLASH_CSP: &str = "default-src 'none'; script-src 'self'; style-src 'self'; img-src 'self'; \
    connect-src ipc: http://ipc.localhost; object-src 'none'; base-uri 'none'; form-action 'none'; frame-ancestors 'none'";

fn splash_protocol(path: &str) -> Response<Cow<'static, [u8]>> {
    let (body, mime) = match path {
        "/" | "/index.html" => (SPLASH_HTML, "text/html; charset=utf-8"),
        "/splash.css" => (SPLASH_CSS, "text/css; charset=utf-8"),
        "/splash.js" => (SPLASH_JS, "text/javascript; charset=utf-8"),
        _ => {
            return Response::builder().status(404).body(Cow::Borrowed(&b""[..])).unwrap();
        }
    };
    Response::builder()
        .header(header::CONTENT_TYPE, mime)
        .header("Content-Security-Policy", SPLASH_CSP)
        .header("X-Content-Type-Options", "nosniff")
        .body(Cow::Borrowed(body.as_bytes()))
        .unwrap()
}

fn splash_url() -> Url {
    // Custom schemes are served as http://<scheme>.localhost on Windows (WebView2).
    let s = if cfg!(windows) { "http://jepa-setup.localhost/index.html" } else { "jepa-setup://localhost/index.html" };
    Url::parse(s).unwrap()
}

fn is_splash_url(u: &Url) -> bool {
    (u.scheme() == "jepa-setup" && u.host_str() == Some("localhost"))
        || (matches!(u.scheme(), "http" | "https") && u.host_str() == Some("jepa-setup.localhost"))
}

/// The bundled site: tauri://localhost (Linux, macOS) or http(s)://tauri.localhost (Windows).
fn is_app_url(u: &Url) -> bool {
    (u.scheme() == "tauri" && u.host_str() == Some("localhost"))
        || (matches!(u.scheme(), "http" | "https") && u.host_str() == Some("tauri.localhost"))
}

fn open_setup_window(app: &AppHandle) -> tauri::Result<WebviewWindow> {
    WebviewWindowBuilder::new(app, SETUP, WebviewUrl::CustomProtocol(splash_url()))
        // On Linux all windows share one web context, and the `tauri:` protocol handler (with its
        // resource hook) is registered by the *first* webview, which is this one. So this window
        // carries the same hook as the main window.
        .on_web_resource_request(rewrite_site_csp)
        .title("jepa-studio setup")
        .inner_size(560.0, 380.0)
        .resizable(false)
        .center()
        .on_navigation(is_splash_url)
        .on_new_window(|_, _| NewWindowResponse::Deny)
        .build()
}

// ------------------------------------------------------------------ main window

/// Port of this session's backend; 0 until it is ready.
static BACKEND_PORT: AtomicU16 = AtomicU16::new(0);

/// Resource hook for the bundled site (`tauri:` protocol): pin the header CSP's backend source to
/// this session's port and drop the web tier's `<meta>` CSP from HTML (see csp.rs).
fn rewrite_site_csp(_req: tauri::http::Request<Vec<u8>>, resp: &mut Response<Cow<'static, [u8]>>) {
    // Only HTML responses carry the CSP header.
    let Some(value) = resp.headers().get("Content-Security-Policy").and_then(|v| v.to_str().ok()) else { return };
    let pinned = csp::pin_backend_port(value, BACKEND_PORT.load(Ordering::SeqCst));
    if let Ok(v) = header::HeaderValue::from_str(&pinned) {
        resp.headers_mut().insert("Content-Security-Policy", v);
    }
    let stripped = std::str::from_utf8(resp.body()).ok().and_then(csp::strip_meta_csp);
    if let Some(html) = stripped {
        *resp.body_mut() = Cow::Owned(html.into_bytes());
        resp.headers_mut().remove(header::CONTENT_LENGTH);
    }
}

fn init_script(port: u16, token: &str) -> String {
    let desktop = serde_json::json!({ "port": port, "token": token });
    // Only on the bundled site's origin, read-only, non-enumerable, frozen.
    format!(
        "(function () {{\n\
           var l = window.location;\n\
           if (!(l.protocol === 'tauri:' || l.hostname === 'tauri.localhost')) return;\n\
           Object.defineProperty(window, '__JEPA_DESKTOP__', {{ value: Object.freeze({desktop}), writable: false, configurable: false, enumerable: false }});\n\
         }})();"
    )
}

fn open_main_window(app: &AppHandle, port: u16, token: &str) -> tauri::Result<WebviewWindow> {
    #[allow(unused_mut)]
    let mut init = init_script(port, token);
    #[cfg(debug_assertions)]
    if smoke_mode() {
        init.push_str(SMOKE_INIT);
    }
    #[allow(unused_mut)]
    let mut b = WebviewWindowBuilder::new(app, MAIN, WebviewUrl::App("app.html".into()))
        .title("jepa-studio")
        .inner_size(1320.0, 860.0)
        .min_inner_size(960.0, 640.0)
        .center()
        .initialization_script(init)
        .on_web_resource_request(rewrite_site_csp)
        // No external pages in the app window (the token lives here); there is no shell plugin
        // to open them elsewhere either.
        .on_navigation(is_app_url)
        .on_new_window(|_, _| NewWindowResponse::Deny);
    #[cfg(debug_assertions)]
    if smoke_mode() {
        b = b
            .on_page_load(|w, p| {
                if matches!(p.event(), tauri::webview::PageLoadEvent::Finished) {
                    let _ = w.eval(SMOKE_PROBE);
                }
            })
            .on_document_title_changed(|w, title| {
                if let Some(rest) = title.strip_prefix("SMOKE-") {
                    eprintln!("[smoke] {rest}");
                    let code = if rest.starts_with("OK") { 0 } else { 1 };
                    w.app_handle().exit(code);
                }
            });
    }
    b.build()
}

// Debug builds only: JEPA_STUDIO_SMOKE=1 loads the real site module, calls the backend through it,
// prints the outcome on stderr and exits. Used to verify the shell headlessly (xvfb-run).
#[cfg(debug_assertions)]
fn smoke_mode() -> bool {
    std::env::var("JEPA_STUDIO_SMOKE").map(|v| v == "1").unwrap_or(false)
}
#[cfg(debug_assertions)]
const SMOKE_INIT: &str = "\n;window.__smokeErrors = [];\
    window.addEventListener('error', function (e) { window.__smokeErrors.push(String(e.message)); });\
    window.addEventListener('securitypolicyviolation', function (e) { window.__smokeErrors.push('CSP ' + e.violatedDirective + ' ' + e.blockedURI + ' policy=' + e.originalPolicy); });";
#[cfg(debug_assertions)]
const SMOKE_PROBE: &str = "setTimeout(async function () {\
  try {\
    const m = await import('./js/backend.js');\
    const tier = m.detectTier();\
    const b = m.makeBackend();\
    const health = await b.req('GET', '/api/health');\
    let noToken = null;\
    try { noToken = (await fetch(b.base + '/api/health')).status; } catch (e) { noToken = 'fetch error: ' + e; }\
    const d = window.__JEPA_DESKTOP__;\
    document.title = 'SMOKE-OK ' + JSON.stringify({ tier: tier, health: health, noTokenStatus: noToken,\
      frozen: Object.isFrozen(d), writable: Object.getOwnPropertyDescriptor(window, '__JEPA_DESKTOP__').writable,\
      tauriGlobal: typeof window.__TAURI__, appTitle: document.querySelector('h1, .wordmark') ? 'ok' : 'missing',\
      errors: window.__smokeErrors });\
  } catch (e) {\
    document.title = 'SMOKE-FAIL ' + e + ' ' + JSON.stringify(window.__smokeErrors || []);\
  }\
}, 1500);";

// ------------------------------------------------------------------ boot sequence

fn boot(app: &AppHandle) -> Result<(), String> {
    let data_dir = app.path().app_local_data_dir().map_err(|e| format!("no app data folder: {e}"))?;
    let logs = data_dir.join("logs");
    let log =
        setup::Logger::open(&logs.join("setup.log")).map_err(|e| format!("cannot write {}: {e}", logs.display()))?;
    log.line(&format!("--- jepa-studio {} starting", env!("CARGO_PKG_VERSION")));
    report(app, |s| {
        s.phase = "setup".into();
        s.message = "Checking the Python environment".into();
        s.log_dir = Some(logs.display().to_string());
    });

    // (a) interpreter: developer override, or the private venv (created on first launch)
    let (python, isolated) = match std::env::var_os("JEPA_STUDIO_PYTHON").filter(|v| !v.is_empty()) {
        Some(p) => {
            let p = PathBuf::from(p);
            log.line(&format!("JEPA_STUDIO_PYTHON override: {}", p.display()));
            report(app, |s| {
                s.accelerator = Some("unknown (JEPA_STUDIO_PYTHON override)".into());
                s.python = Some(p.display().to_string());
            });
            (p, false)
        }
        None => {
            let bundled_pkg = app
                .path()
                .resolve("python/jepa_studio", BaseDirectory::Resource)
                .map_err(|e| format!("bundled jepa_studio package not found: {e}"))?;
            let paths = setup::SetupPaths { data_dir: data_dir.clone(), bundled_pkg };
            let app2 = app.clone();
            let progress = move |f: Option<f64>, head: &str, line: &str| {
                report(&app2, |s| {
                    if let Some(f) = f {
                        s.progress = Some(f);
                    }
                    s.message = head.to_string();
                    s.line = Some(line.chars().take(300).collect());
                });
            };
            let rec = setup::ensure_env(&paths, env!("CARGO_PKG_VERSION"), &log, &progress)?;
            report(app, |s| {
                s.accelerator = Some(rec.accelerator.clone());
                s.torch_index = Some(rec.torch_index.clone());
                s.python = Some(rec.venv_python.display().to_string());
            });
            (rec.venv_python, true)
        }
    };

    // (b) token, allowed-folders file, backend process
    report(app, |s| {
        s.phase = "starting".into();
        s.message = "Starting the local backend".into();
        s.progress = Some(0.96);
        s.line = None;
    });
    let token = backend::generate_token()?;
    let allowed = data_dir.join("allowed-folders.txt");
    backend::ensure_allowed_file(&allowed).map_err(|e| format!("cannot create {}: {e}", allowed.display()))?;
    let backend_log = logs.join("backend.log");
    // Bundled docs for the assistant; absent in a dev run, where the backend finds ./docs itself.
    let docs_dir = app.path().resolve("python/docs", BaseDirectory::Resource).ok().filter(|d| d.is_dir());
    let spec = BackendSpec {
        python: &python,
        isolated,
        workspace: &data_dir.join("workspace"),
        allowed_file: &allowed,
        docs_dir: docs_dir.as_deref(),
        token: &token,
        log: &log,
        stderr_log: &backend_log,
    };
    let b = Backend::spawn(&spec).map_err(|e| {
        format!("{e}\n\nLast lines of {}:\n{}", backend_log.display(), backend::log_tail(&backend_log, 15))
    })?;
    let port = b.port;
    BACKEND_PORT.store(port, Ordering::SeqCst);
    {
        let st = app.state::<SharedState>();
        *st.backend.lock().unwrap() = Some(b);
        *st.allowed_file.lock().unwrap() = Some(allowed);
    }
    report(app, |s| {
        s.phase = "ready".into();
        s.message = "Ready".into();
        s.progress = Some(1.0);
        s.port = Some(port);
    });

    // (c) main window with the injected desktop object; then drop the setup window
    open_main_window(app, port, &token).map_err(|e| format!("could not open the main window: {e}"))?;
    if let Some(w) = app.get_webview_window(SETUP) {
        let _ = w.destroy();
    }
    watch_backend(app.clone(), backend_log);
    Ok(())
}

/// If the backend dies while the app is open, say so instead of leaving the UI with failing calls.
fn watch_backend(app: AppHandle, backend_log: PathBuf) {
    std::thread::spawn(move || loop {
        std::thread::sleep(Duration::from_secs(2));
        let st = app.state::<SharedState>();
        let exited = {
            let mut g = st.backend.lock().unwrap();
            match g.as_mut() {
                None => return, // shutting down
                Some(b) => b.try_wait(),
            }
        };
        if let Some(code) = exited {
            let msg = format!(
                "The local backend stopped unexpectedly ({code}). Restart jepa-studio.\n\nLast lines of {}:\n{}",
                backend_log.display(),
                backend::log_tail(&backend_log, 12)
            );
            st.backend.lock().unwrap().take();
            report(&app, |s| {
                s.phase = "error".into();
                s.message = "The backend stopped".into();
                s.error = Some(msg.clone());
                s.port = None;
            });
            app.dialog().message(msg).title("jepa-studio").kind(MessageDialogKind::Error).show(|_| {});
            return;
        }
    });
}

fn fail(app: &AppHandle, err: String) {
    eprintln!("jepa-studio: {err}");
    report(app, |s| {
        s.phase = "error".into();
        s.message = "Setup failed".into();
        s.error = Some(err.clone());
    });
    if let Some(w) = app.get_webview_window(SETUP) {
        let _ = w.show();
        let _ = w.set_focus();
    } else {
        app.dialog().message(err).title("jepa-studio").kind(MessageDialogKind::Error).show(|_| {});
    }
}

fn main() {
    let shared: SharedState = Arc::new(Shared::default());
    let app = tauri::Builder::default()
        .plugin(tauri_plugin_dialog::init())
        .manage(shared)
        .register_uri_scheme_protocol("jepa-setup", |_ctx, req| splash_protocol(req.uri().path()))
        .invoke_handler(tauri::generate_handler![pick_data_folder, backend_status])
        .setup(|app| {
            let handle = app.handle().clone();
            if let Err(e) = open_setup_window(&handle) {
                eprintln!("jepa-studio: could not open the setup window: {e}");
            }
            std::thread::spawn(move || {
                if let Err(e) = boot(&handle) {
                    kill_backend(&handle);
                    fail(&handle, e);
                }
            });
            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the jepa-studio shell");

    if let Ok(d) = app.path().app_local_data_dir() {
        eprintln!("jepa-studio: data folder {}", d.display());
    }

    app.run(|handle, event| match event {
        RunEvent::WindowEvent { label, event: WindowEvent::Destroyed, .. } if label == MAIN => {
            // The main window is gone (closed or its web process crashed): stop the backend and quit.
            kill_backend(handle);
            handle.exit(0);
        }
        RunEvent::Exit => kill_backend(handle),
        _ => {}
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn init_script_is_frozen_and_origin_guarded() {
        let s = init_script(43817, "abc_DEF-123");
        assert!(s.contains(r#"{"port":43817,"token":"abc_DEF-123"}"#), "{s}");
        assert!(s.contains("Object.freeze("));
        assert!(s.contains("writable: false"));
        assert!(s.contains("l.protocol === 'tauri:'"));
    }

    #[test]
    fn url_allow_lists() {
        let ok = [
            "tauri://localhost/app.html",
            "http://tauri.localhost/app.html#train",
            "https://tauri.localhost/index.html",
        ];
        for u in ok {
            assert!(is_app_url(&Url::parse(u).unwrap()), "{u}");
        }
        let bad = [
            "https://example.com/",
            "http://127.0.0.1:5000/",
            "tauri://evil/",
            "http://tauri.localhost.evil.com/",
            "file:///etc/passwd",
        ];
        for u in bad {
            assert!(!is_app_url(&Url::parse(u).unwrap()), "{u}");
        }
        assert!(is_splash_url(&splash_url()));
        assert!(!is_splash_url(&Url::parse("https://example.com").unwrap()));
    }

    #[test]
    fn splash_protocol_serves_only_its_files() {
        let r = splash_protocol("/index.html");
        assert_eq!(r.status(), 200);
        assert!(r.headers().get("Content-Security-Policy").unwrap().to_str().unwrap().contains("default-src 'none'"));
        assert_eq!(splash_protocol("/splash.js").status(), 200);
        assert_eq!(splash_protocol("/../../etc/passwd").status(), 404);
        assert_eq!(splash_protocol("/app.html").status(), 404);
    }
}
