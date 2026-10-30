"""Headless-browser tests of the JEV and Science tabs (site/js/tabs/jev.js, science.js).

Desktop tier, simulated: the test injects what the Tauri shell injects
(`window.__JEPA_DESKTOP__ = {port, token}`), serves site/ from that origin and answers every
`/api/*` request with an in-test fake of the JEV backend (docs/jev/api.md). The fake streams a
scripted turn:

    tokens -> tool_call (sandbox) -> progress -> tool_result with an image artifact
           -> tool_call (approve) -> approval_request for run_shell
           -> [the user approves in the dialog] -> tool_result with an mp4 artifact -> tokens -> done

and the test checks the UI states, the approval dialog's keyboard behaviour (focus starts on
Deny, Tab order, Enter approves, Esc denies), the "N waiting" badges, that the image artifact is
shown from a blob: URL and the video artifact as a <video>, the Approvals history and the
Doctor. Web tier: the JEV tab shows the desktop-required panel. Science: the canvas is present
and either renders (WebGL via SwiftShader) or shows its fallback. No console errors or CSP
violations anywhere; no horizontal scroll at 375 px.

Also writes assets/screens/tab-jev.png, jev-approval.png and tab-science.png.
Skipped when Playwright or its Chromium build is not installed.
"""
from __future__ import annotations

import functools
import http.server
import io
import json
import mimetypes
import os
import socketserver
import threading
import time
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
SHOTS = ROOT / "assets" / "screens"
sync_api = pytest.importorskip("playwright.sync_api")

PORT = 47213                     # never listened on: every request to it is answered by page.route
TOKEN = "t" * 43
CTYPES = {".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
          ".woff2": "font/woff2", ".svg": "image/svg+xml", ".css": "text/css", ".html": "text/html"}


def _png(w=320, h=180) -> bytes:
    from PIL import Image, ImageDraw
    im = Image.new("RGB", (w, h), (5, 7, 12))
    d = ImageDraw.Draw(im)
    d.ellipse((w // 2 - 70, h // 2 - 22, w // 2 + 70, h // 2 + 22), outline=(255, 190, 90), width=6)
    d.ellipse((w // 2 - 30, h // 2 - 30, w // 2 + 30, h // 2 + 30), fill=(0, 0, 0), outline=(255, 230, 180), width=2)
    b = io.BytesIO()
    im.save(b, "PNG")
    return b.getvalue()


IMG = {"id": "art-img", "path": "black_hole.png", "kind": "image", "mime": "image/png", "title": "Black hole, spin 0.9",
       "url": "/api/jev/artifacts/art-img", "bytes": 0}
VID = {"id": "art-vid", "path": "orbit.mp4", "kind": "video", "mime": "video/mp4", "title": "Orbit, 10 s",
       "url": "/api/jev/artifacts/art-vid", "bytes": 4096}
CODE = {"id": "art-code", "path": "render.py", "kind": "code", "mime": "text/x-python", "title": "render.py",
        "url": "/api/jev/artifacts/art-code", "bytes": 120}


class FakeJev:
    """The JEV routes, enough to drive one scripted turn. Thread-safe enough for page.route."""

    def __init__(self):
        self.png = _png()
        IMG["bytes"] = len(self.png)
        self.lock = threading.Lock()
        self.decisions: list[tuple[str, str, object]] = []
        self.pending: list[dict] = []
        self.history: list[dict] = [
            {"id": "old1", "created": time.time() - 3600, "tool": "run_python", "args": {"code": "print(1)"}, "reason": "",
             "risk": "sandbox", "status": "auto"},
        ]
        self.conversations: list[dict] = []
        self.chats: list[dict] = []
        self.calls: list[tuple[str, str, object]] = []        # (method, path?query, body) for the new routes
        self.models = [
            {"name": "qwen2.5:7b", "size_bytes": 4_683_087_332, "family": "qwen2", "parameter_size": "7.6B",
             "quantization": "Q4_K_M", "digest": "sha256:845dbda0ea48ed749caafd9e6037047aa19acfcfd82e704d7ca97d631a0b697e"},
            {"name": "llama3.2:3b", "size_bytes": 2_019_393_189, "family": "llama", "parameter_size": "3.2B",
             "quantization": "Q4_K_M", "digest": "sha256:a80c4f17acd55265feec403c7aef86be0c25983ab279d83f3bcd3abbcb5b8b72"}]
        self.imported: list[dict] = []
        self.settings = {
            "ollama_url": "http://127.0.0.1:11434", "allow_remote_ollama": False, "allow_cloud_models": False,
            "default_model": "qwen2.5:7b", "workspace_root": "/home/me/JEV/jev-data/workspaces",
            "limits": {"cpu_seconds": 120, "memory_mb": 2048, "wall_seconds": 300, "max_output_mb": 200, "max_file_mb": 500},
            "network_allowed": False, "approval_rules": {}, "approval_timeout_s": 1800, "max_tool_rounds": 12,
            "sandbox": {"isolation": "auto", "hide_home": True}}
        self.rules = [
            {"rule": "read", "default": "run", "action": "run", "covers": "tools with no side effects", "locked": False},
            {"rule": "shell", "default": "ask", "action": "deny", "covers": "running shell commands", "locked": False},
            {"rule": "model_download", "default": "ask", "action": "ask", "covers": "downloading a model (ollama pull)", "locked": False},
            {"rule": "path_escape", "default": "deny", "action": "deny", "covers": "workspace-path arguments escaping", "locked": True}]
        self.actions: dict[str, object] = {}       # approval id -> fn() run on approve, returns a job id or None
        self.jobs: dict[str, list[dict]] = {}
        self.released = 0
        self.approved = False
        self.script = [
            {"event": "start", "model": "qwen2.5:7b"},
            {"event": "token", "text": "Let me render "},
            {"event": "token", "text": "that for you."},
            {"event": "tool_call", "call_id": "k1", "name": "render_black_hole",
             "args": {"spin": 0.9, "width": 640, "height": 360, "code": "import jev\nprint('render')  # demo"},
             "risk": "sandbox", "decision": "run"},
            {"event": "progress", "call_id": "k1", "fraction": 0.5, "message": "frame 60 of 120"},
            {"event": "tool_result", "call_id": "k1", "ok": True, "content": "Rendered a 640x360 frame.", "artifacts": [IMG, CODE]},
            {"event": "tool_call", "call_id": "k2", "name": "run_shell", "args": {"command": "ls ~/Videos"},
             "risk": "approve", "decision": "ask"},
            {"event": "approval_request", "approval_id": "a1", "call_id": "k2", "name": "run_shell",
             "args": {"command": "ls ~/Videos"}, "reason": "run_shell runs outside the sandbox with your permissions."},
            # --- gated: nothing after this is released until a1 is decided
            {"event": "tool_result", "call_id": "k2", "ok": True, "content": "clip.mp4\n", "artifacts": [VID]},
            {"event": "token", "text": "Here is the black hole **and** a video of the orbit."},
            {"event": "done", "text": "Here is the black hole **and** a video of the orbit."},
        ]
        self.gate = next(i for i, e in enumerate(self.script) if e["event"] == "approval_request") + 1

    # -------------------------------------------------------------- dispatch
    def handle(self, method: str, path: str, query: dict, body: dict | None):
        with self.lock:
            return self._handle(method, path, query, body)

    def _handle(self, method, path, query, body):
        if path == "/api/health":
            return 200, {"ok": True}
        if path.startswith("/api/hardware"):
            return 200, {"card": {"backend": "cpu", "device": "cpu", "logical_cores": 2, "ram_gb": 8, "torch": "test"},
                         "decisions": []}
        if path == "/api/jev/status":
            return 200, {"version": "0.4.0", "ollama": {"url": "http://127.0.0.1:11434", "reachable": True, "version": "0.12.6"},
                         "default_model": "qwen2.5:7b", "models": ["qwen2.5:7b", "llama3.2:3b"], "tools": [],
                         "workspace_root": "/tmp/jev", "data_dir": "/home/me/JEV/jev-data",
                         "sandbox": {"network_isolation": "enforced", "method": "bwrap", "limits": {},
                                     "filesystem": "read-only outside the workspace", "rlimits": "cpu, data, fsize, nproc, nofile",
                                     "process_tree_kill": "session + descendant walk"},
                         "gpu": {"name": "Test GPU", "memory_gb": 8}, "approval_rules": self.rules,
                         "default_model": self.settings["default_model"]}
        if path == "/api/jev/models" and method == "GET":
            return 200, {"models": self.models, "imported": self.imported}
        new = self._new_routes(method, path, query, body)
        if new is not None:
            return new
        if path == "/api/jev/conversations" and method == "GET":
            return 200, {"conversations": self.conversations}
        if path == "/api/jev/conversations" and method == "POST":
            return 200, {"id": "c1"}
        if path.startswith("/api/jev/conversations/") and path.endswith("/files"):
            return 200, {"path": f"uploads/{body.get('name')}", "name": body.get("name"), "bytes": len(body.get("data_base64", "")) * 3 // 4}
        if path.startswith("/api/jev/conversations/") and method == "GET":
            return 200, {"id": "c1", "title": "Black hole render", "artifacts": [IMG, VID], "messages": [
                {"role": "user", "content": "Render a black hole"},
                {"role": "assistant", "content": "", "tool_calls": json.dumps(
                    [{"id": "k1", "function": {"name": "render_black_hole", "arguments": {"spin": 0.9}}}])},
                {"role": "tool", "content": "Rendered a 640x360 frame.", "tool_name": "render_black_hole"},
                {"role": "assistant", "content": "Here it is, with `spin = 0.9`."}]}
        if path == "/api/jev/chat":
            self.chats.append(body)
            self.conversations = [{"id": "c1", "title": "Black hole render", "created": time.time(), "updated": time.time(), "model": "qwen2.5:7b"}]
            return 200, {"conversation_id": "c1", "turn_id": "t1"}
        if path == "/api/jev/turns/t1/events":
            since = int(query.get("since", ["0"])[0])
            limit = len(self.script) if self.approved else self.gate
            self.released = min(limit, max(self.released, since) + 2)     # two events per poll: streaming
            status = "done" if self.released >= len(self.script) else "running"
            return 200, {"events": self.script[since:self.released], "next": self.released, "status": status}
        if path == "/api/jev/approvals" and method == "GET":
            if query.get("status") == ["pending"]:
                return 200, {"approvals": self.pending}
            return 200, {"approvals": self.history + self.pending}
        if path.startswith("/api/jev/approvals/") and method == "POST":
            aid = path.rsplit("/", 1)[1]
            self.decisions.append((aid, body.get("decision"), body.get("remember")))
            if aid == "a1" and body.get("decision") == "approve":
                self.approved = True
            self.pending = [p for p in self.pending if p["id"] != aid]
            status = "approved" if body.get("decision") == "approve" else "denied"
            if aid in self.actions:
                fn = self.actions.pop(aid)
                jid = fn() if status == "approved" else None
                self.history.append({"id": aid, "created": time.time(), "tool": "x", "args": {}, "status": status})
                return 200, {"ok": True, "status": status, **({"job_id": jid} if jid else {})}
            self.history.append({"id": aid, "created": time.time(), "tool": "run_shell" if aid == "a1" else "pull_model",
                                 "args": {"command": "ls ~/Videos"} if aid == "a1" else {"name": "llama3.1:8b"}, "status": status,
                                 "remember": body.get("remember")})
            return 200, {"ok": True, "status": status}
        if path == "/api/jev/artifacts/art-img":
            return 200, (self.png, "image/png")
        if path == "/api/jev/artifacts/art-vid":
            return 200, (b"\x00\x00\x00\x18ftypmp42" + b"\x00" * 4000, "video/mp4")
        if path == "/api/jev/artifacts/art-code":
            return 200, (b"import numpy as np\n\n# a comment\ndef spin(a=0.9):\n    return np.sqrt(1 - a * a)  # horizon\n", "text/x-python")
        if path.startswith("/api/jev/artifacts"):
            return 200, {"artifacts": [IMG, VID]}
        return 404, {"error": f"no fake route for {method} {path}"}


    def _gate(self, aid, tool, args, fn):
        """Like JevApi._gate: a pending approval (also in the pending list) whose approval runs fn."""
        self.actions[aid] = fn
        self.pending.append({"id": aid, "created": time.time(), "kind": tool, "tool": tool, "args": args,
                             "reason": f"{tool} asks first", "risk": "approve", "status": "pending"})
        return 200, {"approval_id": aid, "status": "pending"}

    def _new_routes(self, method, path, query, body):
        """Models, settings, export / delete all, model jobs (docs/jev/api.md)."""
        q = "&".join(f"{k}={v[0]}" for k, v in sorted(query.items()))
        if path.startswith("/api/jev/models") or path.startswith("/api/jev/settings") or path.startswith("/api/jev/data") \
                or path == "/api/jev/export" or (path == "/api/jev/approvals" and method == "GET"):
            self.calls.append((method, path + (f"?{q}" if q else ""), body))
        aid = f"m{len(self.actions) + len(self.calls)}"
        if path == "/api/jev/settings" and method == "GET":
            return 200, {**self.settings, "rules": self.rules, "paths": {
                "data_dir": "/home/me/JEV/jev-data", "settings": "/home/me/JEV/jev-data/settings.json",
                "db": "/home/me/JEV/jev-data/jev.db", "workspaces": self.settings["workspace_root"],
                "exports": "/home/me/JEV/jev-data/exports"}}
        if path == "/api/jev/settings" and method == "POST":
            bad = [k for k in body if k not in ("default_model", "limits", "network_allowed", "ollama_url")]
            if bad:
                return 400, {"error": f"these settings can only be changed by editing settings.json: {bad}"}
            if "ollama_url" in body and not urlparse(body["ollama_url"]).hostname in ("127.0.0.1", "localhost", "::1"):
                return 400, {"error": "ollama_url is not on this machine"}
            for k, v in body.items():
                self.settings[k] = {**self.settings[k], **v} if k == "limits" else v
            return 200, dict(self.settings)
        if path == "/api/jev/models/pull" and method == "POST":
            name = body["name"]

            def pull():
                self.jobs["jp"] = [{"event": "start"}, {"event": "progress", "fraction": 0.5, "message": "pulling 4.7 GB"},
                                   {"event": "done", "text": f"Pulled {name}."}]
                self.models.append({"name": name, "size_bytes": 4_700_000_000, "family": "qwen2", "parameter_size": "7.6B",
                                    "quantization": "Q4_K_M", "digest": "sha256:" + "ab" * 32})
                return "jp"
            return self._gate(aid, "model_pull", {"name": name}, pull)
        if path == "/api/jev/models/import" and method == "POST":
            name, gguf = body["name"], body["gguf_path"]
            sha = "9f" * 32

            def imp():
                self.jobs["ji"] = [{"event": "progress", "fraction": 0.0, "message": "computing SHA-256"},
                                   {"event": "done", "text": f"Imported as {name} (sha256 {sha}).", "sha256": sha}]
                self.models.append({"name": f"{name}:latest", "size_bytes": 1_000_000, "family": "llama", "quantization": "Q4_K_M",
                                    "digest": "sha256:" + "cd" * 32})
                self.imported.append({"name": name, "path": gguf, "sha256": sha, "bytes": 1_000_000, "created": time.time()})
                return "ji"
            return self._gate(aid, "model_import", {"name": name, "gguf_path": gguf, "bytes": 1_000_000}, imp)
        if path.startswith("/api/jev/models/") and method == "DELETE":
            from urllib.parse import unquote
            name = unquote(path.split("/api/jev/models/", 1)[1])

            def rm():
                self.jobs["jd"] = [{"event": "done", "text": f"Deleted {name} from Ollama."}]
                self.models = [m for m in self.models if m["name"] != name]
                return "jd"
            return self._gate(aid, "model_delete", {"name": name}, rm)
        if path.startswith("/api/jev/jobs/") and path.endswith("/events"):
            jid = path.split("/")[4]
            since = int(query.get("since", ["0"])[0])
            ev = self.jobs.get(jid, [])
            return 200, {"events": ev[since:since + 1], "next": since + 1, "status": "done" if since + 1 >= len(ev) else "running"}
        if path == "/api/jev/export":
            return 200, (b"PK\x05\x06" + b"\0" * 18, "application/zip")
        if path == "/api/jev/data/delete" and method == "POST":
            def wipe():
                self.conversations = []
                self.history = []
                return None
            return self._gate(aid, "data_delete", {"keep_audit_log": body.get("keep_audit_log"), "conversations": 0,
                                                   "data_dir": "/home/me/JEV/jev-data"}, wipe)
        return None


def _route(fake: FakeJev):
    def handler(route):
        req = route.request
        u = urlparse(req.url)
        if u.path.startswith("/api/"):
            if req.headers.get("x-jepa-token") != TOKEN:
                return route.fulfill(status=401, body='{"error":"bad token"}', content_type="application/json")
            body = None
            if req.post_data:
                try:
                    body = json.loads(req.post_data)
                except ValueError:
                    body = None
            status, payload = fake.handle(req.method, u.path, parse_qs(u.query), body)
            if isinstance(payload, tuple):
                return route.fulfill(status=status, body=payload[0], content_type=payload[1])
            return route.fulfill(status=status, body=json.dumps(payload), content_type="application/json")
        f = (SITE / (u.path.lstrip("/") or "index.html")).resolve()
        if SITE.resolve() not in f.parents or not f.is_file():
            return route.fulfill(status=404, body="not found")
        ctype = CTYPES.get(f.suffix) or mimetypes.guess_type(f.name)[0] or "application/octet-stream"
        return route.fulfill(status=200, body=f.read_bytes(), content_type=ctype)
    return handler


class Console:
    """Fails on console errors, page errors and CSP violations."""

    def __init__(self, page):
        self.errors: list[str] = []
        self.csp: list[str] = []
        self.missing_optional: list[str] = []
        page.on("console", self.on_console)
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        page.on("response", self.on_response)

    def on_response(self, resp):
        if resp.status < 400:
            return
        path = urlparse(resp.url).path
        # the science modules are written separately; until they exist the tab shows its fallback
        if path.startswith("/js/science/") and not (SITE / path.lstrip("/")).is_file():
            self.missing_optional.append(path)
        else:
            self.errors.append(f"HTTP {resp.status} {resp.url}")

    def on_console(self, msg):
        if "Content Security Policy" in msg.text or "Content-Security-Policy" in msg.text:
            self.csp.append(msg.text)
        if msg.type == "error":
            if "Failed to load resource" in msg.text and self.missing_optional:
                return
            self.errors.append(msg.text)

    def check(self):
        assert not self.csp, f"CSP violations: {self.csp}"
        assert not self.errors, f"console errors: {self.errors}"


@pytest.fixture(scope="module")
def browser():
    try:
        pw = sync_api.sync_playwright().start()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"playwright unavailable: {e}")
    try:
        b = pw.chromium.launch(args=["--enable-unsafe-swiftshader", "--use-angle=swiftshader", "--ignore-gpu-blocklist"])
    except Exception as e:  # pragma: no cover
        pw.stop()
        pytest.skip(f"chromium not installed for playwright: {e}")
    yield b
    b.close()
    pw.stop()


@pytest.fixture(scope="module")
def web_server():
    class H(http.server.SimpleHTTPRequestHandler):
        extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, **CTYPES}

        def log_message(self, *a):
            pass
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(H, directory=str(SITE)))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


def wait_until(page, fn, timeout=30.0, what="the page"):
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        page.wait_for_timeout(100)
    raise AssertionError(f"timed out waiting for {what}")


def desktop_page(browser, fake, *, width=1280, height=860, scheme="light", hash_="", gguf=None):
    ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme, reduced_motion="reduce",
                              device_scale_factor=1, accept_downloads=True)
    ctx.add_init_script("Object.defineProperty(window, '__JEPA_DESKTOP__', {value: Object.freeze(%s)});"
                        % json.dumps({"port": PORT, "token": TOKEN}))
    if gguf is not None:
        # the shell's withGlobalTauri API, stubbed: pick_gguf_file answers with a path chosen in the
        # "native dialog" (the page never supplies one); every invoke is recorded
        ctx.add_init_script("""window.__invoked = [];
            window.__TAURI__ = {core: {invoke: async (cmd, args) => {
              window.__invoked.push([cmd, args === undefined ? null : args]);
              if (cmd === 'pick_gguf_file') return %s;
              return null; }}};""" % json.dumps(gguf))
    page = ctx.new_page()
    con = Console(page)
    page.route("**/*", _route(fake))
    page.goto(f"http://127.0.0.1:{PORT}/app.html{hash_}")
    wait_until(page, lambda: page.get_attribute("html", "data-ready") == "true", what="app ready")
    return ctx, page, con


def review_shot(page, name, **kw):
    """Extra screenshots for a human review (dark mode, phone width), only when asked for."""
    d = os.environ.get("JEV_UI_REVIEW_DIR")
    if d:
        Path(d).mkdir(parents=True, exist_ok=True)
        page.screenshot(path=str(Path(d) / name), **kw)


def no_hscroll(page, width):
    w = page.evaluate("() => document.documentElement.scrollWidth")
    assert w <= width, f"horizontal scroll: {w}px at {width}px"


def clear_toasts(page):
    page.evaluate("() => document.querySelectorAll('.toast').forEach((t) => t.remove())")


# ------------------------------------------------------------------------------------------ tests

def test_jev_scripted_turn_with_approval(browser):
    SHOTS.mkdir(parents=True, exist_ok=True)
    fake = FakeJev()
    ctx, page, con = desktop_page(browser, fake)
    # the desktop tier opens on JEV, with the JEPA stepper hidden
    assert page.get_attribute("#tab-jev", "aria-selected") == "true"
    assert page.is_visible("#jev-desk") and not page.is_visible("#jev-web")
    assert not page.is_visible("#stepper-bar")
    wait_until(page, lambda: "Ollama running 0.12.6" in (page.text_content("#st-ollama") or ""), what="status strip")
    assert "Sandbox offline" in page.text_content("#st-net")
    assert "Test GPU" in page.text_content("#st-gpu")
    assert page.get_attribute("#st-ollama", "data-status") == "good"
    opts = page.locator("#jev-model option").all_text_contents()
    assert opts[0].startswith("qwen2.5:7b (4.7 GB, 7.6B)"), opts
    assert page.input_value("#jev-model") == "qwen2.5:7b"
    # starters fill the box; send is disabled until there is text
    assert page.locator("#jev-starter-list button").count() == 6
    assert page.is_disabled("#jev-send")
    page.click("#jev-starter-list button >> text=Render a spinning black hole")
    assert "black hole" in page.input_value("#jev-input")
    assert not page.is_disabled("#jev-send")
    # the log is a polite live region
    assert page.get_attribute("#jev-log", "role") == "log" and page.get_attribute("#jev-log", "aria-live") == "polite"
    page.focus("#jev-input")
    page.keyboard.press("Enter")
    wait_until(page, lambda: page.locator("#jev-log .msg-user").count() == 1, what="user message")
    assert fake.chats and fake.chats[0]["model"] == "qwen2.5:7b" and "black hole" in fake.chats[0]["message"]
    # streaming: the log is busy and the streamed text is hidden from screen readers
    wait_until(page, lambda: page.locator(".tool-card[data-call=k1]").count() == 1, what="first tool card")
    assert page.get_attribute("#jev-log", "aria-busy") == "true"
    assert page.get_attribute("#jev-log .seg-text.stream", "aria-hidden") == "true"
    card = page.locator(".tool-card[data-call=k1]")
    assert card.locator(".risk").get_attribute("data-risk") == "sandbox"
    assert card.locator(".risk .risk-word").inner_text() == "Sandbox"
    wait_until(page, lambda: card.get_attribute("data-state") == "done", what="k1 done")
    assert not card.locator(".tc-progress").is_visible()            # progress is replaced by the result
    assert "Rendered a 640x360 frame." in card.locator(".tc-out").inner_text()
    # the image artifact: in the card and in the gallery, from a blob: URL
    wait_until(page, lambda: (page.get_attribute("#jev-gallery .art[data-kind=image] img", "src") or "").startswith("blob:"),
               what="gallery image from a blob URL")
    assert page.evaluate("() => document.querySelector('#jev-gallery .art[data-kind=image] img').naturalWidth") == 320
    wait_until(page, lambda: page.locator("#jev-gallery .art[data-kind=code] .tk-kw").count() > 0, what="highlighted code artifact")

    # ---- the approval request: a modal dialog, focus on Deny, badges show 1 waiting
    wait_until(page, lambda: page.evaluate("() => document.getElementById('jev-approval').open"), what="approval dialog")
    assert page.evaluate("() => document.activeElement.id") == "appr-deny"
    assert page.text_content("#appr-title") == "Allow JEV to run a command on your computer?"
    assert page.text_content("#appr-exact") == "ls ~/Videos"
    assert "outside the sandbox" in page.text_content("#appr-why")
    assert page.locator("#appr-risk .risk[data-risk=approve] .risk-word").inner_text() == "Asks first"
    assert page.text_content("#jev-waiting") == "1 waiting" and page.text_content("#jev-tab-count") == "1"
    assert page.get_attribute("#tab-jev", "aria-label") == "JEV, 1 approval waiting"
    assert page.locator(".tool-card[data-call=k2]").get_attribute("data-state") == "waiting"
    page.screenshot(path=str(SHOTS / "jev-approval.png"))
    # keyboard: Tab order Deny -> session -> once; Enter on "Approve once" approves
    page.keyboard.press("Tab")
    assert page.evaluate("() => document.activeElement.id") == "appr-session"
    page.keyboard.press("Tab")
    assert page.evaluate("() => document.activeElement.id") == "appr-once"
    page.keyboard.press("Enter")
    wait_until(page, lambda: fake.decisions, what="the decision POST")
    assert fake.decisions[0] == ("a1", "approve", None)
    wait_until(page, lambda: not page.evaluate("() => document.getElementById('jev-approval').open"), what="dialog closed")
    assert not page.is_visible("#jev-waiting") and not page.is_visible("#jev-tab-count")

    # ---- the turn finishes: final answer rendered, log not busy, one summary announced
    wait_until(page, lambda: page.locator("#jev-log .msg-jev .msg-state[data-status=good]").count() == 1, what="turn done")
    assert page.get_attribute("#jev-log", "aria-busy") == "false"
    assert page.locator("#jev-log .seg-text.stream").count() == 0
    assert page.locator("#jev-log .msg-jev strong", has_text="and").count() == 1          # **and** rendered as bold
    summary = page.locator("#jev-log .sr-summary").all_text_contents()
    assert len(summary) == 1 and summary[0].startswith("JEV answered after 2 tool calls: Let me render that for you.")
    assert page.locator(".tool-card[data-call=k2]").get_attribute("data-state") == "done"
    # arguments are collapsible, code shown highlighted with line numbers
    assert not card.locator(".tc-args").evaluate("(d) => d.open")
    card.locator(".tc-args summary").click()
    assert card.locator(".tc-args pre.code[data-lang=python] .ln").count() == 2
    assert card.locator(".tc-args pre.code[data-lang=json]").count() == 1
    # the video artifact is a <video controls> on a blob: URL; downloads are anchors with download attrs
    wait_until(page, lambda: (page.get_attribute("#jev-gallery video", "src") or "").startswith("blob:"), what="video blob")
    assert page.get_attribute("#jev-gallery video", "controls") is not None
    assert page.get_attribute("#jev-gallery .art[data-kind=video] a.art-dl", "download") == "orbit.mp4"
    # lightbox: opens from the image, Esc closes it and focus returns
    page.click("#jev-gallery .art[data-kind=image] .art-open")
    assert page.evaluate("() => document.getElementById('jev-lightbox').open")
    assert page.get_attribute("#jev-lightbox img", "src").startswith("blob:")
    page.keyboard.press("Escape")
    assert not page.evaluate("() => document.getElementById('jev-lightbox').open")
    assert page.evaluate("() => document.activeElement.classList.contains('art-open')")
    # the conversation now shows in the sidebar
    wait_until(page, lambda: page.locator("#jev-conv-list .conv[aria-current=true]").count() == 1, what="conversation listed")
    clear_toasts(page)
    page.evaluate("() => window.scrollTo(0, 0)")
    page.screenshot(path=str(SHOTS / "tab-jev.png"))

    # ---- a second request arrives from the pending list (a model download); Esc denies it
    with fake.lock:
        fake.pending.append({"id": "a2", "created": time.time(), "tool": "pull_model", "args": {"name": "llama3.1:8b"},
                             "reason": "Downloading a model uses the internet.", "risk": "approve", "status": "pending"})
    wait_until(page, lambda: page.evaluate("() => document.getElementById('jev-approval').open"), timeout=10, what="2nd dialog")
    assert page.text_content("#appr-title") == "Allow JEV to download a model?"
    assert page.text_content("#appr-exact") == "llama3.1:8b"
    page.keyboard.press("Escape")
    wait_until(page, lambda: len(fake.decisions) == 2, what="deny POST")
    assert fake.decisions[1] == ("a2", "deny", None)
    wait_until(page, lambda: not page.evaluate("() => document.getElementById('jev-approval').open"), what="dialog closed")

    # ---- Approvals panel: history with words and icons
    page.click("#jev-approvals-btn")
    assert page.get_attribute("#jev-approvals-btn", "aria-expanded") == "true"
    wait_until(page, lambda: page.locator("#jev-history tr").count() >= 3, what="history rows")
    rows = page.locator("#jev-history").inner_text()
    assert "Denied" in rows and "Approved" in rows and "Ran automatically" in rows
    assert "Nothing is waiting" in page.text_content("#jev-pending")
    page.click("[data-close-panel=jev-approvals]")
    assert not page.is_visible("#jev-approvals")
    # ---- Doctor
    page.click("#jev-doctor-btn")
    assert page.is_visible("#jev-doctor")
    assert "Ollama is running (0.12.6)" in page.text_content("#doctor-checks")
    assert "Sandboxed code has no network" in page.text_content("#doctor-checks")
    con.check()
    ctx.close()


def test_jev_doctor_when_ollama_is_down_and_phone_width(browser):
    fake = FakeJev()
    orig = fake._handle

    def down(method, path, query, body):
        status, payload = orig(method, path, query, body)
        if path == "/api/jev/status":
            payload = {**payload, "ollama": {"url": "http://127.0.0.1:11434", "reachable": False, "error": "connection refused"},
                       "models": [], "sandbox": {"network_isolation": "not available", "method": "none"}}
        if path == "/api/jev/models":
            payload = {"models": []}
        return status, payload
    fake._handle = down
    ctx, page, con = desktop_page(browser, fake, width=375, height=740)
    wait_until(page, lambda: page.get_attribute("#st-ollama", "data-status") == "bad", what="ollama down")
    assert page.text_content("#st-ollama").strip() == "Ollama not reachable"
    assert page.is_disabled("#jev-send")
    page.fill("#jev-input", "hello")
    page.dispatch_event("#jev-input", "input")
    assert page.is_disabled("#jev-send") and "Ollama is not running" in page.text_content("#jev-send-why")
    page.click("#jev-doctor-btn")
    checks = page.text_content("#doctor-checks")
    assert "Ollama is not running" in checks and "Sandboxed code could reach the network" in checks
    # per-OS commands with copy buttons
    page.check("input[name=doc-os][value=linux]", force=True)
    assert "curl -fsSL https://ollama.com/install.sh | sh" in page.text_content("#doctor-checks")
    page.check("input[name=doc-os][value=windows]", force=True)
    assert "winget install Ollama.Ollama" in page.text_content("#doctor-checks")
    assert page.locator("#doctor-checks .copy-btn").count() >= 2
    # phones: the conversation list is a drawer (a modal dialog), Esc closes it
    assert not page.is_visible("#jev-conv-list") and page.is_visible("#jev-convs-toggle")
    page.click("#jev-convs-toggle")
    assert page.evaluate("() => document.getElementById('jev-convs-dialog').open")
    page.keyboard.press("Escape")
    assert not page.evaluate("() => document.getElementById('jev-convs-dialog').open")
    no_hscroll(page, 375)
    small = json.loads(page.evaluate("""() => JSON.stringify([...document.querySelectorAll(
        '#panel-jev :is(.btn, .starter, select, .conv, .seg label)')]
        .filter((n) => n.offsetParent !== null)
        .map((n) => [n.id || n.className, Math.round(n.getBoundingClientRect().height)])
        .filter(([, h]) => h < 44))"""))
    assert not small, f"touch targets under 44 px: {small}"
    review_shot(page, "jev-375-doctor.png", full_page=True)
    con.check()
    ctx.close()


def test_attachments_upload_to_proposed_route(browser, tmp_path):
    fake = FakeJev()
    ctx, page, con = desktop_page(browser, fake)
    f = tmp_path / "notes.txt"
    f.write_text("hello from a file\n")
    big = tmp_path / "big.bin"
    big.write_bytes(b"\0" * (2 * 1024 * 1024))
    page.set_input_files("#jev-files", [str(f), str(big)])
    wait_until(page, lambda: page.locator("#jev-attach .chip[data-state=ready]").count() == 1, what="upload")
    chips = page.locator("#jev-attach").inner_text()
    assert "notes.txt" in chips and "over 1.5 MB" in chips
    page.fill("#jev-input", "summarize the file")
    page.dispatch_event("#jev-input", "input")
    page.click("#jev-send")
    wait_until(page, lambda: fake.chats, what="chat POST")
    assert fake.chats[0]["attachments"] == ["uploads/notes.txt"] and fake.chats[0]["conversation_id"] == "c1"
    wait_until(page, lambda: page.locator("#jev-log .msg-attach li").count() == 1, what="attachment listed")
    con.check()
    ctx.close()


def test_web_tier_shows_desktop_panel(browser, web_server):
    ctx = browser.new_context(viewport={"width": 1280, "height": 800}, reduced_motion="reduce")
    page = ctx.new_page()
    con = Console(page)
    page.goto(f"{web_server}/app.html#jev")
    wait_until(page, lambda: page.get_attribute("html", "data-ready") == "true")
    assert page.is_visible("#jev-web") and not page.is_visible("#jev-desk")
    assert page.text_content("#jev-web h2") == "JEV runs on your computer"
    assert page.get_attribute("#jev-download", "href") == "https://github.com/Normansrule/jev-jepa-studio/releases/latest"
    assert page.locator("#jev-web .risk").count() == 3
    page.click("#jev-goto-science")
    assert page.get_attribute("#tab-science", "aria-selected") == "true"
    # the web tier never polls a backend: nothing but static files was requested
    con.check()
    ctx.close()
    # phone width: web JEV and Science panels fit
    ctx = browser.new_context(viewport={"width": 375, "height": 740}, reduced_motion="reduce")
    page = ctx.new_page()
    con = Console(page)
    for tab in ("jev", "science"):
        page.goto(f"{web_server}/app.html#{tab}")
        wait_until(page, lambda: page.get_attribute("html", "data-ready") == "true")
        page.wait_for_timeout(300)
        no_hscroll(page, 375)
    con.check()
    ctx.close()


def test_science_tab_renders_or_falls_back(browser, web_server):
    SHOTS.mkdir(parents=True, exist_ok=True)
    ctx = browser.new_context(viewport={"width": 1280, "height": 800}, reduced_motion="reduce", device_scale_factor=1)
    page = ctx.new_page()
    con = Console(page)
    page.goto(f"{web_server}/app.html#science")
    wait_until(page, lambda: page.get_attribute("html", "data-ready") == "true")
    assert page.is_visible("#sci-canvas") or page.is_visible("#sci-fallback")
    wait_until(page, lambda: page.get_attribute("#sci-stage", "data-state") in ("live", "fallback"), timeout=30,
               what="science view live or fallback")
    state = page.get_attribute("#sci-stage", "data-state")
    if state == "fallback":
        assert page.text_content("#sci-fallback-text")
    else:
        assert page.locator("#sci-canvas").count() == 1
        page.wait_for_timeout(1500)          # a few frames
    # every slider has its number box; desktop-only render controls are hidden on the web
    assert page.locator("#bh-spin-num").count() == 1
    assert not page.is_visible("#bh-render") and page.is_visible("#sci-web-note")
    # simulations: switch view and kind
    page.check("input[name=sci-view][value=sims]", force=True)
    assert page.is_visible("#sci-sims") and not page.is_visible("#sci-bh")
    page.select_option("#sim-kind", "double_pendulum")
    assert page.is_visible("#sim-theta1") and page.text_content("#sci-stage-title") == "Double pendulum"
    wait_until(page, lambda: page.get_attribute("#sci-stage", "data-state") in ("live", "fallback"), timeout=30)
    page.check("input[name=sci-view][value=blackhole]", force=True)
    wait_until(page, lambda: page.get_attribute("#sci-stage", "data-state") in ("live", "fallback"), timeout=30)
    page.wait_for_timeout(800)
    clear_toasts(page)
    page.screenshot(path=str(SHOTS / "tab-science.png"))
    con.check()
    ctx.close()


def test_science_render_on_computer(browser):
    """Desktop tier: "Render high-quality video on this computer" posts the settings and shows
    the job's progress and its MP4 in the tab's gallery."""
    fake = FakeJev()
    orig = fake._handle
    posted = []

    def jobs(method, path, query, body):
        if path == "/api/jev/science/blackhole":
            posted.append(body)
            return 200, {"job_id": "j1"}
        if path == "/api/jev/jobs/j1/events":
            ev = [{"event": "progress", "fraction": 0.4, "message": "frame 96 of 240"},
                  {"event": "artifact", "artifact": VID}, {"event": "done", "text": ""}]
            since = int(query.get("since", ["0"])[0])
            return 200, {"events": ev[since:since + 1], "next": since + 1, "status": "done" if since + 1 >= 3 else "running"}
        return orig(method, path, query, body)
    fake._handle = jobs
    ctx, page, con = desktop_page(browser, fake, hash_="#science")
    assert page.is_visible("#bh-render")
    page.click("#bh-render")
    wait_until(page, lambda: posted, what="render POST")
    p = posted[0]
    assert p["spin"] == 0.9 and p["inclination_deg"] == 80 and p["distance_M"] == 25 and p["width"] == 1280
    assert p["disk"] == {"enabled": True, "doppler": True, "beaming": True} and p["seconds"] == 10
    wait_until(page, lambda: (page.get_attribute("#sci-gallery video", "src") or "").startswith("blob:"), what="render video")
    wait_until(page, lambda: "Finished" in (page.text_content("#sci-job-status") or ""), what="job finished")
    assert not page.is_disabled("#bh-render")
    con.check()
    ctx.close()


def test_open_a_saved_conversation(browser):
    fake = FakeJev()
    fake.conversations = [{"id": "c1", "title": "Black hole render", "created": time.time(), "updated": time.time(),
                           "model": "qwen2.5:7b"}]
    ctx, page, con = desktop_page(browser, fake)
    wait_until(page, lambda: page.locator("#jev-conv-list .conv").count() == 1, what="conversation list")
    page.click("#jev-conv-list .conv")
    wait_until(page, lambda: page.locator("#jev-log .msg").count() == 3, what="history rendered")
    assert page.text_content("#jev-conv-title") == "Black hole render"
    assert page.get_attribute("#jev-conv-list .conv", "aria-current") == "true"
    assert "Rendered a 640x360 frame." in page.locator("#jev-log .tool-card .tc-out").inner_text()
    assert page.locator("#jev-log .msg-jev code", has_text="spin = 0.9").count() == 1
    assert not page.is_visible("#jev-starters")
    wait_until(page, lambda: page.locator("#jev-gallery .art").count() == 2, what="gallery from the conversation")
    # New conversation clears the log and the gallery
    page.click("#jev-new")
    assert page.locator("#jev-log .msg").count() == 0 and page.is_visible("#jev-starters")
    assert page.text_content("#jev-conv-title") == "New conversation"
    assert page.evaluate("() => document.activeElement.id") == "jev-input"
    con.check()
    ctx.close()


# ------------------------------------------------------------------ Models, Settings, audit log

def dialog_open(page):
    return page.evaluate("() => document.getElementById('jev-approval').open")


def approve_in_dialog(page, title):
    wait_until(page, lambda: dialog_open(page), what=f"approval dialog: {title}")
    assert page.text_content("#appr-title") == title
    page.click("#appr-once")
    wait_until(page, lambda: not dialog_open(page), what="dialog closed")


def test_models_panel_list_default_delete_pull_import(browser):
    fake = FakeJev()
    gguf = "/home/me/models/My-Model.Q4_K_M.gguf"
    ctx, page, con = desktop_page(browser, fake, gguf=gguf)
    wait_until(page, lambda: page.input_value("#jev-model") == "qwen2.5:7b", what="models loaded")
    # the composer's button and the strip's button open the same panel
    page.click("#jev-pull-btn")
    assert page.is_visible("#jev-models")
    assert page.get_attribute("#jev-pull-btn", "aria-expanded") == "true" == page.get_attribute("#jev-models-btn", "aria-expanded")
    rows = page.locator("#jev-model-list .model-row")
    wait_until(page, lambda: rows.count() == 2, what="two installed models")
    q = page.locator(".model-row[data-model='qwen2.5:7b']").inner_text()
    assert "4.7 GB" in q and "7.6B" in q and "Q4_K_M" in q and "qwen2" in q and "digest 845dbda0ea48" in q and "Default" in q
    assert page.is_disabled(".model-row[data-model='qwen2.5:7b'] button >> text=Default")
    # examples sized by the reported GPU (8 GB): the 6-11 GB tier first, then the small one
    ex = page.text_content("#jev-pull-examples")
    assert "your GPU has 8 GB" in ex and "Examples" in ex
    tiers = page.locator("#jev-pull-examples .example-list").evaluate_all("(l) => l.map((u) => u.getAttribute('aria-label'))")
    assert tiers == ["6 to 11 GB of GPU memory", "Small: CPU only or under 6 GB of GPU memory"]
    # make llama3.2:3b the default
    page.click(".model-row[data-model='llama3.2:3b'] button >> text=Make default")
    wait_until(page, lambda: ("POST", "/api/jev/settings", {"default_model": "llama3.2:3b"}) in fake.calls, what="default POST")
    wait_until(page, lambda: "Default" in page.locator(".model-row[data-model='llama3.2:3b'] .tag").all_text_contents(), what="tag moved")
    wait_until(page, lambda: page.input_value("#jev-model") == "llama3.2:3b", what="composer uses the new default")
    clear_toasts(page)
    # delete asks; approving runs the job and the row goes away
    page.click(".model-row[data-model='llama3.2:3b'] button >> text=Delete…")
    wait_until(page, lambda: dialog_open(page), what="delete dialog")
    assert page.text_content("#appr-exact") == "llama3.2:3b"
    assert "DELETE" in [c[0] for c in fake.calls] and ("DELETE", "/api/jev/models/llama3.2%3A3b", None) in fake.calls
    approve_in_dialog(page, "Allow JEV to delete a model?")
    wait_until(page, lambda: rows.count() == 1, what="model removed from the list")
    wait_until(page, lambda: "Deleted llama3.2:3b" in page.text_content("#jev-models-status"), what="delete status")
    # download: an example fills the name; the request asks; progress then the new row
    page.click("#jev-pull-examples .example >> text=qwen2.5-coder:7b")
    assert page.input_value("#jev-pull-name") == "qwen2.5-coder:7b"
    page.click("#jev-pull-go")
    wait_until(page, lambda: dialog_open(page), what="pull dialog")
    assert page.text_content("#appr-exact") == "qwen2.5-coder:7b"
    approve_in_dialog(page, "Allow JEV to download a model?")
    wait_until(page, lambda: "Pulled qwen2.5-coder:7b." in page.text_content("#jev-pull-status"), what="pull done")
    wait_until(page, lambda: page.locator(".model-row[data-model='qwen2.5-coder:7b']").count() == 1, what="pulled model listed")
    page.fill("#jev-pull-name", "../../etc")
    page.click("#jev-pull-go")
    assert "Type a model name" in page.text_content("#jev-pull-status")
    # import: the native dialog (stubbed) returns the path; the page only forwards it
    page.click("#jev-gguf-pick")
    wait_until(page, lambda: page.is_visible("#jev-gguf-form"), what="import form")
    assert page.evaluate("() => window.__invoked.filter(([c]) => c === 'pick_gguf_file')") == [["pick_gguf_file", None]]
    assert page.text_content("#jev-gguf-path") == gguf and page.input_value("#jev-gguf-name") == "my-model.q4_k_m"
    page.fill("#jev-gguf-name", "bad name!")
    page.click("#jev-gguf-go")
    assert page.is_visible("#jev-gguf-name-error") and page.get_attribute("#jev-gguf-name", "aria-invalid") == "true"
    assert not any(c[1] == "/api/jev/models/import" for c in fake.calls)
    page.fill("#jev-gguf-name", "my-model")
    page.click("#jev-gguf-go")
    wait_until(page, lambda: dialog_open(page), what="import dialog")
    assert ("POST", "/api/jev/models/import", {"gguf_path": gguf, "name": "my-model"}) in fake.calls
    assert page.text_content("#appr-exact") == f"{gguf}  as  my-model"
    approve_in_dialog(page, "Allow JEV to import a model file?")
    wait_until(page, lambda: page.is_visible("#jev-gguf-sha"), what="recorded SHA-256")
    assert page.text_content("#jev-gguf-sha-value") == "9f" * 32
    wait_until(page, lambda: "imported file SHA-256 " + "9f" * 32 in page.locator(".model-row[data-model='my-model:latest']").inner_text(),
               what="SHA-256 in the list")
    review_shot(page, "jev-models-1280.png", full_page=True)
    con.check()
    ctx.close()


def test_models_panel_without_shell_and_cpu_only(browser):
    """No __TAURI__ (or no GPU): import explains the terminal route; examples are the small tier."""
    fake = FakeJev()
    orig = fake._handle

    def cpu(method, path, query, body):
        status, payload = orig(method, path, query, body)
        if path == "/api/jev/status":
            payload = {**payload, "gpu": {"cuda": False, "mps": False, "devices": []}}
        return status, payload
    fake._handle = cpu
    ctx, page, con = desktop_page(browser, fake)
    page.click("#jev-models-btn")
    wait_until(page, lambda: page.locator("#jev-model-list .model-row").count() == 2, what="models")
    assert page.is_disabled("#jev-gguf-pick") and "jev models import" in page.text_content("#jev-gguf-pick-why")
    assert "no GPU was found" in page.text_content("#jev-pull-examples")
    assert page.locator("#jev-pull-examples .example-list").count() == 1
    assert "OLLAMA_MODELS" in page.text_content("#jev-models") and "~/.ollama/models" in page.text_content("#jev-models")
    page.click("[data-close-panel=jev-models]")
    assert not page.is_visible("#jev-models") and page.get_attribute("#jev-models-btn", "aria-expanded") == "false"
    con.check()
    ctx.close()


def test_settings_panel_validation_save_data_and_rules(browser):
    fake = FakeJev()
    ctx, page, con = desktop_page(browser, fake)
    page.click("#jev-settings-btn")
    wait_until(page, lambda: page.input_value("#set-ollama-url") == "http://127.0.0.1:11434", what="settings loaded")
    assert page.input_value("#set-wall-seconds") == "300" and page.input_value("#set-memory-mb") == "2048"
    assert page.input_value("#set-default-model") == "qwen2.5:7b"
    assert page.is_disabled("#set-save")                                  # nothing changed yet
    # limits: out of range -> message next to the field, Save disabled
    page.fill("#set-wall-seconds", "0")
    assert page.is_visible("#set-wall-seconds-error") and "1 to 86,400" in page.text_content("#set-wall-seconds-error")
    assert page.get_attribute("#set-wall-seconds", "aria-invalid") == "true" and page.is_disabled("#set-save")
    page.fill("#set-memory-mb", "10")
    assert "64 to 1,048,576" in page.text_content("#set-memory-mb-error")
    page.fill("#set-memory-mb", "2048")
    page.fill("#set-wall-seconds", "600")
    assert not page.is_visible("#set-wall-seconds-error") and not page.is_disabled("#set-save")
    # a remote Ollama: refused unless "allow remote" is ticked, then a warning, and still blocked
    # because allow_remote_ollama is off in the file (the app cannot turn it on)
    page.fill("#set-ollama-url", "http://10.0.0.5:11434")
    assert "not this computer" in page.text_content("#set-ollama-url-error") and page.is_disabled("#set-save")
    assert not page.is_visible("#set-remote-warn")
    page.check("#set-allow-remote", force=True)
    assert page.is_visible("#set-remote-warn") and "leave this computer" in page.text_content("#set-remote-warn")
    assert "allow_remote_ollama" in page.text_content("#set-ollama-url-error") and page.is_disabled("#set-save")
    assert "/home/me/JEV/jev-data/settings.json" in page.text_content("#set-ollama-url-error")
    page.fill("#set-ollama-url", "not a url")
    assert "full address" in page.text_content("#set-ollama-url-error")
    page.fill("#set-ollama-url", "http://localhost:11434/")
    page.uncheck("#set-allow-remote", force=True)
    assert not page.is_visible("#set-ollama-url-error")
    # network: the switch explains it; a warning appears when on
    assert "still ask every time" in page.text_content("#set-network-hint")
    page.check("#set-network", force=True)
    assert page.is_visible("#set-network-warn")
    review_shot(page, "jev-settings-1280.png", full_page=True)
    page.click("#set-save")
    wait_until(page, lambda: any(c[:2] == ("POST", "/api/jev/settings") for c in fake.calls), what="settings POST")
    sent = [c[2] for c in fake.calls if c[:2] == ("POST", "/api/jev/settings")][-1]
    assert sent == {"ollama_url": "http://localhost:11434", "limits": {"wall_seconds": 600}, "network_allowed": True}
    wait_until(page, lambda: "Saved" in page.text_content("#set-status"), what="saved")
    assert page.is_disabled("#set-save")
    # what the sandbox enforces, where the data is, the read-only rules table
    caps = page.text_content("#set-caps")
    assert "Network isolation" in caps and "enforced" in caps and "read-only outside the workspace" in caps
    paths = page.locator("#set-paths code").all_text_contents()
    assert "/home/me/JEV/jev-data/jev.db" in paths and "/home/me/JEV/jev-data/settings.json" in paths
    assert page.locator("#set-paths .copy-btn").count() == len(paths)
    assert page.locator("#set-rules tr").count() == 4
    assert page.locator("#set-rules tr[data-rule=path_escape] .tag").inner_text() == "Locked"
    shell = page.locator("#set-rules tr[data-rule=shell]").inner_text()
    assert "Refused" in shell and "default: Asks first" in shell
    assert "allow_remote_ollama" in page.text_content("#set-file-only")
    assert page.locator("#jev-settings input:not([type=checkbox]), #jev-settings select").evaluate_all(
        "(l) => l.every((n) => n.id && document.querySelector(`label[for='${n.id}']`))")
    # export everything: a zip download, no navigation
    with page.expect_download() as dl:
        page.click("#set-export")
    assert dl.value.suggested_filename.startswith("jev-data-") and dl.value.suggested_filename.endswith(".zip")
    # delete everything: asks; Esc denies; then approve with the audit log kept
    page.click("#set-delete")
    wait_until(page, lambda: dialog_open(page), what="delete-all dialog")
    assert page.text_content("#appr-title") == "Allow JEV to delete all of its data?"
    assert "and the audit log" in page.text_content("#appr-what")
    page.keyboard.press("Escape")
    wait_until(page, lambda: "Nothing was deleted" in page.text_content("#set-data-status"), what="denied")
    page.check("#set-keep-audit", force=True)
    page.click("#set-delete")
    assert ("POST", "/api/jev/data/delete", {"keep_audit_log": True}) in fake.calls
    wait_until(page, lambda: dialog_open(page), what="delete-all dialog again")
    assert "and the audit log" not in page.text_content("#appr-what")
    approve_in_dialog(page, "Allow JEV to delete all of its data?")
    wait_until(page, lambda: "was deleted" in page.text_content("#set-data-status"), what="deleted")
    con.check()
    ctx.close()


def test_audit_log_filters_and_csv(browser, tmp_path):
    fake = FakeJev()
    day = 86400
    now = time.time()
    fake.history = [
        {"id": "h1", "created": now - 3 * day, "tool": "run_python", "args": {"code": "print(1)"}, "status": "auto_run", "decided_by": "policy"},
        {"id": "h2", "created": now - 2 * day, "tool": "run_shell", "args": {"command": "=HYPERLINK(\"x\")"}, "status": "approved", "decided_by": "user"},
        {"id": "h3", "created": now - 1 * day, "tool": "model_pull", "args": {"name": "llama3.1:8b"}, "status": "denied", "decided_by": "user"},
        {"id": "h4", "created": now, "tool": "write_file", "args": {"path": "../x"}, "status": "auto_deny", "decided_by": "policy"},
        {"id": "h5", "created": now, "tool": "run_shell", "args": {"command": "ls"}, "status": "expired", "decided_by": "timeout"},
    ]
    ctx, page, con = desktop_page(browser, fake)
    page.click("#jev-approvals-btn")
    body = page.locator("#jev-history tr")
    wait_until(page, lambda: body.count() == 5, what="five audit rows")
    assert ("GET", "/api/jev/approvals?limit=5000", None) in fake.calls
    assert page.text_content("#audit-count") == "5 of 5 decisions"
    words = page.locator("#jev-history .appr-status").all_text_contents()
    assert "Ran automatically" in words and "Refused by policy" in words and any(w.startswith("Expired") for w in words)
    assert page.locator("#audit-tool option").all_text_contents() == ["All tools", "model_pull", "run_python", "run_shell", "write_file"]
    page.select_option("#audit-decision", "denied")
    assert body.count() == 1 and "llama3.1:8b" in body.first.inner_text()
    page.select_option("#audit-decision", "")
    page.select_option("#audit-tool", "run_shell")
    assert body.count() == 2 and page.text_content("#audit-count") == "2 of 5 decisions"
    page.select_option("#audit-tool", "")
    days = page.locator("#jev-history tr").evaluate_all("(l) => l.map((r) => r.dataset.day)")
    page.fill("#audit-from", sorted(days)[2])                  # the last three days
    page.dispatch_event("#audit-from", "change")
    assert body.count() == 3
    page.fill("#audit-to", sorted(days)[2])
    page.dispatch_event("#audit-to", "change")
    assert body.count() == 1 and "model_pull" in body.first.inner_text()
    page.select_option("#audit-decision", "approved")
    assert "No decisions match" in body.first.inner_text() and page.is_disabled("#audit-csv")
    page.click("#audit-reset")
    assert body.count() == 5
    page.select_option("#audit-tool", "run_shell")
    with page.expect_download() as dl:
        page.click("#audit-csv")
    out = tmp_path / "audit.csv"
    dl.value.save_as(out)
    import csv
    rows = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert dl.value.suggested_filename.startswith("jev-audit-") and [r["id"] for r in rows] == ["h5", "h2"]
    assert rows[1]["status"] == "approved" and rows[1]["created"].endswith("Z")
    assert rows[1]["args"].startswith("'{") or rows[1]["args"].startswith("{")
    assert '"command"' in rows[1]["args"] and "HYPERLINK" in rows[1]["args"]
    assert all(not r[k].startswith(("=", "+", "-", "@")) for r in rows for k in r)     # no spreadsheet formulas
    review_shot(page, "jev-audit-1280.png", full_page=True)
    con.check()
    ctx.close()


@pytest.mark.parametrize("scheme", ["light", "dark"])
def test_new_panels_fit_a_phone(browser, scheme):
    fake = FakeJev()
    ctx, page, con = desktop_page(browser, fake, width=375, height=740, scheme=scheme, gguf="/home/me/m.gguf")
    for btn, panel in (("#jev-models-btn", "#jev-models"), ("#jev-settings-btn", "#jev-settings"), ("#jev-approvals-btn", "#jev-approvals")):
        page.click(btn)
        assert page.is_visible(panel)
    wait_until(page, lambda: page.locator("#jev-model-list .model-row").count() == 2, what="models")
    wait_until(page, lambda: page.locator("#set-rules tr").count() == 4, what="rules")
    page.click("#jev-gguf-pick")
    wait_until(page, lambda: page.is_visible("#jev-gguf-form"), what="import form")
    page.fill("#set-ollama-url", "http://10.0.0.5:11434")
    page.check("#set-allow-remote", force=True)
    page.wait_for_timeout(200)
    no_hscroll(page, 375)
    small = json.loads(page.evaluate("""() => JSON.stringify([...document.querySelectorAll(
        '#jev-models :is(.btn, .example, select, input), #jev-settings :is(.btn, select, input:not([type=checkbox])), #jev-approvals :is(.btn, select, input)')]
        .filter((n) => n.offsetParent !== null)
        .map((n) => [n.id || n.className, Math.round(n.getBoundingClientRect().height)])
        .filter(([, h]) => h < 44))"""))
    assert not small, f"touch targets under 44 px: {small}"
    clear_toasts(page)
    for panel in ("jev-models", "jev-settings", "jev-approvals"):
        review_shot(page.locator(f"#{panel}"), f"jev-375-{panel}-{scheme}.png")
    con.check()
    ctx.close()
