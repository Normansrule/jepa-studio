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
                         "workspace_root": "/tmp/jev", "sandbox": {"network_isolation": "enforced", "method": "bwrap", "limits": {}},
                         "gpu": {"name": "Test GPU", "memory_gb": 8}}
        if path == "/api/jev/models":
            return 200, {"models": [
                {"name": "qwen2.5:7b", "size_bytes": 4_683_087_332, "family": "qwen2", "parameter_size": "7.6B", "quantization": "Q4_K_M"},
                {"name": "llama3.2:3b", "size_bytes": 2_019_393_189, "family": "llama", "parameter_size": "3.2B", "quantization": "Q4_K_M"}]}
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


def desktop_page(browser, fake, *, width=1280, height=860, scheme="light", hash_=""):
    ctx = browser.new_context(viewport={"width": width, "height": height}, color_scheme=scheme, reduced_motion="reduce",
                              device_scale_factor=1)
    ctx.add_init_script("Object.defineProperty(window, '__JEPA_DESKTOP__', {value: Object.freeze(%s)});"
                        % json.dumps({"port": PORT, "token": TOKEN}))
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
