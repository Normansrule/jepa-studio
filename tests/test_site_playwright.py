"""Headless-browser test of the static site (site/) with Python Playwright + Chromium.

Serves site/ with a stock http.server on a free port, opens app.html and:
  * fails on any console error, page error or CSP violation
  * clicks every tab
  * runs ~30 demo training steps and checks that the loss went down and the isotropy panel updated
  * runs the evaluation
  * writes README screenshots to assets/screens/ (1280x800, light theme; plus landing-dark.png)
  * input features (v0.3): slider <-> number sync and inline schema validation that blocks
    Start, presets and "Custom", help disclosures (keyboard), drag-and-drop and paste of images
    and CSV, the config editor (validate / apply / confirm-before-replace), the progress stepper,
    preferences across a reload, and the app with a localStorage that throws

Skipped when Playwright or its Chromium build is not installed.
"""
from __future__ import annotations

import functools
import http.server
import json
import socketserver
import threading
import time
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
SHOTS = ROOT / "assets" / "screens"

sync_api = pytest.importorskip("playwright.sync_api")


class _Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map,
                      ".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
                      ".woff2": "font/woff2", ".svg": "image/svg+xml"}

    def log_message(self, *args):  # noqa: D401 - keep pytest output clean
        pass


@pytest.fixture(scope="module")
def server():
    handler = functools.partial(_Handler, directory=str(SITE))
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), handler)
    srv.daemon_threads = True
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture(scope="module")
def browser():
    try:
        pw = sync_api.sync_playwright().start()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"playwright unavailable: {e}")
    try:
        b = pw.chromium.launch(args=["--enable-unsafe-webgpu", "--enable-features=Vulkan"])
    except Exception as e:  # pragma: no cover
        pw.stop()
        pytest.skip(f"chromium not installed for playwright: {e}")
    yield b
    b.close()
    pw.stop()


# Requests the site may legitimately 404 on: the world-model files are produced by a separate
# export step and the World tab shows a "model files not found" message without them.
OPTIONAL_404 = ("/models/world-",)


class Console:
    def __init__(self, page):
        self.errors: list[str] = []
        self.csp: list[str] = []
        self.failed_optional: list[str] = []
        page.on("console", self.on_console)
        page.on("pageerror", lambda e: self.errors.append(f"pageerror: {e}"))
        page.on("response", self.on_response)
        page.on("requestfailed", lambda r: self.errors.append(f"request failed: {r.url} {r.failure}"))

    def on_response(self, resp):
        if resp.status >= 400:
            if any(p in resp.url for p in OPTIONAL_404):
                self.failed_optional.append(resp.url)
            else:
                self.errors.append(f"HTTP {resp.status} {resp.url}")

    def on_console(self, msg):
        text = msg.text
        if "Content Security Policy" in text or "Content-Security-Policy" in text:
            self.csp.append(text)
        if msg.type == "error":
            if "Failed to load resource" in text and self.failed_optional:
                return  # the optional world-model 404 already recorded above
            self.errors.append(text)

    def check(self):
        assert not self.csp, f"CSP violations: {self.csp}"
        assert not self.errors, f"console errors: {self.errors}"


def new_page(browser, color_scheme="light"):
    ctx = browser.new_context(viewport={"width": 1280, "height": 800}, color_scheme=color_scheme,
                              reduced_motion="reduce", device_scale_factor=1)
    page = ctx.new_page()
    return ctx, page


def wait_until(page, fn, timeout=120.0, interval=0.25):
    """Poll a Python predicate (the page's CSP forbids string evaluation, so no wait_for_function)."""
    t0 = time.time()
    while time.time() - t0 < timeout:
        v = fn()
        if v:
            return v
        page.wait_for_timeout(int(interval * 1000))
    raise AssertionError("timed out waiting for the page")


def ready(page):
    wait_until(page, lambda: page.get_attribute("html", "data-ready") == "true", timeout=30)


def clear_toasts(page):
    """Notices are fixed to the corner of the viewport; keep them out of README screenshots."""
    page.evaluate("() => document.querySelectorAll('.toast').forEach((t) => t.remove())")


def shot(page, name, **kw):
    clear_toasts(page)
    page.screenshot(path=str(SHOTS / name), **kw)


def test_landing_page(server, browser):
    SHOTS.mkdir(parents=True, exist_ok=True)
    for scheme in ("light", "dark"):
        ctx, page = new_page(browser, scheme)
        con = Console(page)
        page.goto(f"{server}/index.html")
        ready(page)
        wait_until(page, lambda: "Gaussian batch averages" in (page.text_content("#hero-ep") or ""), timeout=10)
        assert page.locator("#cap-body tr").count() >= 8
        page.screenshot(path=str(SHOTS / ("landing.png" if scheme == "light" else "landing-dark.png")))
        con.check()
        ctx.close()


def test_app_end_to_end(server, browser):
    SHOTS.mkdir(parents=True, exist_ok=True)
    ctx, page = new_page(browser)
    con = Console(page)
    page.goto(f"{server}/app.html")
    ready(page)
    assert page.text_content("#tier-label") == "Web demo"

    # every tab is reachable by click and by keyboard
    for tab in ["data", "train", "inspect", "evaluate", "world", "explain", "export"]:
        page.click(f"#tab-{tab}")
        assert page.get_attribute(f"#tab-{tab}", "aria-selected") == "true"
        assert page.is_visible(f"#panel-{tab}")
    page.focus("#tab-export")
    page.keyboard.press("Home")
    assert page.get_attribute("#tab-data", "aria-selected") == "true"
    page.keyboard.press("ArrowRight")
    assert page.get_attribute("#tab-train", "aria-selected") == "true"

    # Data tab screenshot (views of one input), once the hardware card has named the engine
    page.click("#tab-data")
    wait_until(page, lambda: (page.text_content("#backend-label") or "") not in ("", "Checking…"), timeout=30)
    page.wait_for_timeout(300)
    shot(page, "tab-data.png")

    # ---- train ~30 steps: fixed batch 32 so the step count is deterministic, auto-probe tested below
    page.click("#tab-train")
    page.select_option("#train-batch", "32")
    page.fill("#train-steps", "30")
    page.dispatch_event("#train-steps", "input")
    page.dispatch_event("#train-steps", "change")
    wait_until(page, lambda: (page.text_content("#backend-label") or "") not in ("", "Checking…"), timeout=30)
    t0 = time.time()
    page.click("#train-start")
    wait_until(page, lambda: page.text_content("#train-status") in ("Finished", "Error") or
               (page.text_content("#train-status") or "").startswith("Error"), timeout=300)
    elapsed = time.time() - t0
    assert page.text_content("#train-status") == "Finished", page.text_content("#train-status")
    steps = json.loads(page.evaluate("() => JSON.stringify(window.jepaApp.run.events.filter(e => e.event === 'step'))"))
    assert steps[-1]["step"] == 30
    losses = [s["loss"] for s in steps]
    head, tail = sum(losses[:5]) / 5, sum(losses[-5:]) / 5
    assert tail < head, f"loss did not decrease: first {losses[:5]} last {losses[-5:]}"
    # isotropy panel was filled from training snapshots
    assert page.locator("#train-iso dd").count() == 4
    assert page.get_attribute("#train-collapse", "data-status") in ("good", "warn", "bad")
    sps = [s["samples_per_s"] for s in steps if s["step"] > 2]
    info = {"backend": page.text_content("#backend-label"), "mean_samples_per_s": sum(sps) / len(sps),
            "seconds_for_30_steps": elapsed, "loss_first5": head, "loss_last5": tail,
            "webgpu_api": page.evaluate("() => !!navigator.gpu"),
            "gpu": json.loads(page.evaluate("() => JSON.stringify(window.jepaApp.hardware && window.jepaApp.hardware.card.gpu)"))}
    (ROOT / "assets" / "screens" / "headless-benchmark.json").write_text(json.dumps(info, indent=2) + "\n")
    print("\nheadless demo training:", json.dumps(info))
    page.evaluate("() => window.scrollTo(0, 0)")
    shot(page, "tab-train.png")

    # hardware card with a real timing probe
    page.click("#hw-probe")
    wait_until(page, lambda: page.text_content("#hw-probe") == "Run timing probe" and "Picked batch" in (page.text_content("#hw-kv") or ""), timeout=120)
    clear_toasts(page)
    page.locator("#hardware-card").screenshot(path=str(SHOTS / "hardware-card.png"))

    # ---- inspect: refresh, isotropy + collapse verdict, click a point, neighbours, animation
    page.click("#tab-inspect")
    page.click("#inspect-refresh")
    wait_until(page, lambda: page.get_attribute("#inspect-verdict", "data-status") in ("good", "warn", "bad"), timeout=60)
    assert page.locator("#inspect-metrics dd").count() == 4
    page.click("#inspect-thumbs button >> nth=3")
    wait_until(page, lambda: page.locator("#nn-row figure").count() == 9, timeout=30)
    wait_until(page, lambda: page.locator("#saliency canvas").count() == 2, timeout=30)
    assert not page.is_disabled("#anim-slider")
    page.evaluate("() => window.scrollTo(0, 0)")
    shot(page, "tab-inspect.png")

    # ---- evaluate
    page.click("#tab-evaluate")
    page.fill("#eval-frac", "0.1")
    page.click("#eval-run")
    wait_until(page, lambda: (page.text_content("#eval-status") or "").startswith(("Done", "Evaluation failed")), timeout=300)
    assert (page.text_content("#eval-status") or "").startswith("Done"), page.text_content("#eval-status")
    assert page.locator("#eval-table tbody tr").count() == 3
    shot(page, "tab-evaluate.png")

    # ---- world model (works with or without the exported model files)
    page.click("#tab-world")
    wait_until(page, lambda: (page.text_content("#world-status") or "") not in ("", "Loading world models…"), timeout=30)
    if not page.is_disabled("#world-run"):
        page.fill("#world-steps", "12")
        page.dispatch_event("#world-steps", "input")
        page.select_option("#world-samples", "100")
        page.click("#world-run")
        wait_until(page, lambda: page.get_attribute("#world-badge", "data-status") != "idle", timeout=180)
        page.click("#world-random")
        assert page.locator("#world-compare tbody tr").count() == 2
    else:
        assert "not found" in page.text_content("#world-status")
    shot(page, "tab-world.png")

    # ---- explain: off by default, extractive answers, log shown as text
    page.click("#tab-explain")
    assert not page.is_checked("#assistant-toggle")
    page.check("#assistant-toggle")
    page.fill("#assistant-q", "why does SIGReg prevent collapse")
    page.click("#assistant-ask")
    wait_until(page, lambda: page.locator("#assistant-answers > *").count() > 0, timeout=20)
    assert page.locator("#assistant-answers script").count() == 0
    shot(page, "tab-explain.png")

    # ---- export: all downloads work and the bundle is a valid zip
    page.click("#tab-export")
    for btn, suffix in [("#exp-csv", ".csv"), ("#exp-npy", ".npy"), ("#exp-weights", ".safetensors"),
                        ("#exp-report-json", ".json"), ("#exp-report-html", ".html"), ("#exp-bundle", ".zip"),
                        ("#cfg-save", ".json")]:
        with page.expect_download(timeout=60000) as dl:
            page.click(btn)
        path = dl.value.path()
        assert dl.value.suggested_filename.endswith(suffix)
        if suffix == ".zip":
            import zipfile
            with zipfile.ZipFile(path) as z:
                assert z.testzip() is None
                assert sorted(z.namelist()) == ["README.txt", "config.json", "embeddings.npy", "report.json", "weights.safetensors"]
                cfg = json.loads(z.read("config.json"))
                from jepa_studio.config import load_schema, validate
                assert validate(cfg, load_schema()) == []
                assert "jepa-studio train --config config.json" in z.read("README.txt").decode()
    # open a broken config -> errors shown, config unchanged
    bad = ROOT / "assets" / "screens" / ".bad-config.json"
    bad.write_text(json.dumps({"version": 1, "seed": -3, "model": {"arch": "resnet-9000"}, "typo": 1}))
    page.set_input_files("#cfg-open", str(bad))
    wait_until(page, lambda: page.locator("#cfg-result .errors li").count() >= 3, timeout=10)
    bad.unlink()
    shot(page, "tab-export.png")

    con.check()
    ctx.close()


def test_local_images_and_phone_width(server, browser, tmp_path):
    """Local image files are decoded in the page, oversized/non-image files are rejected with a
    reason, and the app has no horizontal scroll at phone width."""
    from PIL import Image
    files = []
    for i in range(20):
        p = tmp_path / f"img_{i:02d}.png"
        Image.new("RGB", (48 + i, 40), (10 * i, 255 - 10 * i, 128)).save(p)
        files.append(str(p))
    junk = tmp_path / "notes.png"
    junk.write_bytes(b"this is not a png")
    files.append(str(junk))
    ctx = browser.new_context(viewport={"width": 375, "height": 740}, reduced_motion="reduce")
    page = ctx.new_page()
    con = Console(page)
    page.goto(f"{server}/app.html")
    ready(page)
    page.check("input[name=dataset][value=images]", force=True)
    page.set_input_files("#image-files", files)
    wait_until(page, lambda: "loaded locally" in (page.text_content("#data-status") or ""), timeout=60)
    status = page.text_content("#data-status")
    assert "20 images" in status and "notes.png" in status
    assert page.locator("#data-thumbs button").count() == 20
    for tab in ["data", "train", "inspect", "evaluate", "world", "explain", "export"]:
        page.click(f"#tab-{tab}")
        width = page.evaluate("() => document.documentElement.scrollWidth")
        assert width <= 375, f"horizontal scroll on {tab}: {width}px"
        # touch targets: every visible control in the panel and the shared chrome is >= 44 px tall
        small = json.loads(page.evaluate("""() => JSON.stringify([...document.querySelectorAll(
            '[role=tabpanel]:not([hidden]) :is(.btn, .info-btn, .link-btn, select, input[type=number], .seg label, .switch), #stepper button, [role=tab]')]
            .filter((n) => n.offsetParent !== null)
            .map((n) => [n.id || n.className || n.tagName, Math.round(n.getBoundingClientRect().height)])
            .filter(([, h]) => h < 44))"""))
        assert not small, f"touch targets under 44 px on {tab}: {small}"
    page.goto(f"{server}/index.html")
    ready(page)
    assert page.evaluate("() => document.documentElement.scrollWidth") <= 375
    con.check()
    ctx.close()


def test_auto_batch_probe_and_pause(server, browser):
    """The default 'auto' batch size runs the timing probe; pause and stop work."""
    ctx, page = new_page(browser)
    con = Console(page)
    page.goto(f"{server}/app.html#train")
    ready(page)
    page.fill("#train-steps", "200")
    page.dispatch_event("#train-steps", "change")
    page.click("#train-start")
    wait_until(page, lambda: page.text_content("#train-status") == "Training", timeout=120)
    page.click("#train-pause")
    wait_until(page, lambda: page.text_content("#train-status") == "Paused", timeout=30)
    page.click("#train-pause")
    wait_until(page, lambda: page.text_content("#train-status") == "Training", timeout=30)
    page.click("#train-stop")
    wait_until(page, lambda: page.text_content("#train-status") == "Stopped", timeout=30)
    assert "Picked batch" in page.text_content("#hw-kv")
    con.check()
    ctx.close()


def test_slow_first_hardware_card_does_not_hide_probe(server, browser):
    """Regression (seen on a 6-core machine): the first hardware card resolved after the probe
    started by "Start", overwrote it, and "Picked batch" vanished. Delay that first card on
    purpose: only the newest hardware result may be shown."""
    ctx, page = new_page(browser)
    ctx.add_init_script("""
      (() => {
        const orig = Worker.prototype.postMessage;
        let first = true;
        Worker.prototype.postMessage = function (msg, ...rest) {
          if (first && msg && msg.type === 'init') {
            first = false;
            setTimeout(() => orig.call(this, msg, ...rest), 4000);   // the slow first card
            return;
          }
          return orig.call(this, msg, ...rest);
        };
      })();
    """)
    con = Console(page)
    page.goto(f"{server}/app.html#train")
    ready(page)
    page.fill("#train-steps", "200")
    page.dispatch_event("#train-steps", "change")
    page.click("#train-start")
    wait_until(page, lambda: page.text_content("#train-status") == "Training", timeout=120)
    assert "Picked batch" in page.text_content("#hw-kv")
    page.wait_for_timeout(5000)                      # the delayed first card has arrived by now
    assert "Picked batch" in page.text_content("#hw-kv")
    page.click("#train-stop")
    wait_until(page, lambda: page.text_content("#train-status") == "Stopped", timeout=30)
    con.check()
    ctx.close()



# ---------------------------------------------------------------------------- v0.3 input features

def app_page(browser, server, hash_="", **kw):
    ctx, page = new_page(browser, **kw)
    con = Console(page)
    page.goto(f"{server}/app.html{hash_}")
    ready(page)
    return ctx, page, con


def cfg(page, expr):
    return json.loads(page.evaluate(f"() => JSON.stringify(window.jepaApp.config.{expr})"))


def visible_errors(page, field_id):
    """Text of the inline message shown for a field (empty when valid)."""
    return page.evaluate("""(id) => { const n = document.getElementById(id);
        const box = n.closest('.field') || n.parentElement;
        const e = box.querySelector('.field-error:not([hidden])'); return e ? e.textContent : ''; }""", field_id)


def test_slider_number_sync_and_inline_validation(server, browser):
    ctx, page, con = app_page(browser, server, "#train")
    # every range slider has a number input with the same min/max/step
    pairs = json.loads(page.evaluate("""() => JSON.stringify([...document.querySelectorAll('input[type=range]')].map((r) => {
        const n = document.getElementById(r.id + '-num');
        return [r.id, !!n, n && ['min', 'max', 'step'].every((a) => r.getAttribute(a) === n.getAttribute(a))]; }))"""))
    assert pairs and all(has and same for _, has, same in pairs), pairs
    # slider -> number
    page.fill("#train-steps", "500")
    page.dispatch_event("#train-steps", "input")
    assert page.input_value("#train-steps-num") == "500"
    # number -> slider and config (keyboard: type, then Enter)
    page.fill("#train-steps-num", "750")
    page.press("#train-steps-num", "Enter")
    assert page.input_value("#train-steps") == "750"
    assert cfg(page, "train.max_steps") == 750
    # the number is the precise value even between slider positions (step 10)
    page.fill("#train-steps-num", "35")
    page.press("#train-steps-num", "Tab")
    assert cfg(page, "train.max_steps") == 35
    assert page.input_value("#train-steps-num") == "35"
    # arrow keys on the number input step it
    page.fill("#train-steps-num", "300")
    page.press("#train-steps-num", "ArrowUp")
    assert page.input_value("#train-steps-num") == "310"
    page.press("#train-steps-num", "Enter")
    # out of the web demo's range: inline message, aria-invalid, Start disabled with the reason
    page.fill("#train-steps-num", "5")
    page.press("#train-steps-num", "Tab")
    assert page.get_attribute("#train-steps-num", "aria-invalid") == "true"
    assert "20 to 2000 steps" in visible_errors(page, "train-steps")
    err_id = page.evaluate("() => document.querySelector('#train-steps-num').getAttribute('aria-describedby')")
    assert page.evaluate("(ids) => ids.split(' ').some((i) => document.getElementById(i)?.textContent.includes('steps'))", err_id)
    assert page.is_disabled("#train-start")
    assert "Steps" in page.text_content("#train-start-why") and page.is_visible("#train-start-why")
    assert cfg(page, "train.max_steps") == 310            # nothing was applied or clamped
    # a schema violation shows the schema rule and key
    page.fill("#train-lambda-num", "1.5")
    page.press("#train-lambda-num", "Tab")
    msg = visible_errors(page, "train-lambda")
    assert "from 0 to 1" in msg and "objective.lambda" in msg, msg
    # a Data-tab field blocks Start too, and the reason names its tab
    page.click("#tab-data")
    page.fill("#data-seed", "-3")
    page.press("#data-seed", "Tab")
    assert "whole number from 0 to 4294967295" in visible_errors(page, "data-seed")
    assert cfg(page, "seed") == 0
    page.click("#tab-train")
    why = page.text_content("#train-start-why")
    assert "Seed (Data tab)" in why and "λ" in why, why
    # fixing everything enables Start again
    page.fill("#train-steps-num", "40")
    page.press("#train-steps-num", "Tab")
    page.fill("#train-lambda-num", "0.1")
    page.press("#train-lambda-num", "Tab")
    page.click("#tab-data")
    page.fill("#data-seed", "4")
    page.press("#data-seed", "Tab")
    assert cfg(page, "seed") == 4
    page.click("#tab-train")
    assert not page.is_disabled("#train-start")
    assert not page.is_visible("#train-start-why")
    assert page.get_attribute("#train-steps-num", "aria-invalid") is None
    # reset to defaults (from the config module)
    page.click("[data-reset=train]")
    assert page.input_value("#train-steps-num") == "300" and page.input_value("#train-lambda-num") == "0.05"
    assert cfg(page, "train.max_steps") == 300 and cfg(page, "objective.lambda") == 0.05
    page.click("#tab-data")
    page.click("[data-reset=data-shapes]")
    assert page.input_value("#data-seed") == "0" and cfg(page, "seed") == 0
    con.check()
    ctx.close()


def test_presets_and_custom(server, browser):
    ctx, page, con = app_page(browser, server, "#train")
    assert page.is_checked("input[name=preset][value=balanced]")
    assert not page.is_visible("#preset-state")
    page.check("input[name=preset][value=quick]", force=True)
    assert page.input_value("#train-steps-num") == "100"
    assert page.input_value("#train-batch") == "32" and page.input_value("#train-slices") == "16"
    assert cfg(page, "train.max_steps") == 100 and cfg(page, "objective.num_slices") == 16
    # any change after picking a preset reads "Custom"
    page.fill("#train-lambda-num", "0.2")
    page.press("#train-lambda-num", "Tab")
    assert page.is_visible("#preset-state") and page.text_content("#preset-state") == "Custom"
    assert page.locator("input[name=preset]:checked").count() == 0
    page.check("input[name=preset][value=thorough]", force=True)
    assert page.input_value("#train-steps-num") == "1000" and page.input_value("#train-lambda-num") == "0.05"
    assert not page.is_visible("#preset-state")
    page.select_option("#train-slices", "64")
    assert page.text_content("#preset-state") == "Custom"
    page.click("[data-reset=train]")
    assert page.is_checked("input[name=preset][value=balanced]")
    con.check()
    ctx.close()


def test_help_disclosure_keyboard(server, browser):
    ctx, page, con = app_page(browser, server, "#train")
    buttons = page.locator(".info-btn")
    assert buttons.count() >= 15
    # every info button controls a hidden panel with a docs link on GitHub
    links = json.loads(page.evaluate("""() => JSON.stringify([...document.querySelectorAll('.info-btn')].map((b) => {
        const p = document.getElementById(b.getAttribute('aria-controls'));
        return [b.getAttribute('aria-expanded'), !!p && p.hidden, p && p.querySelector('a').href]; }))"""))
    for expanded, hidden, href in links:
        assert expanded == "false" and hidden
        assert href.startswith("https://github.com/Normansrule/jepa-studio/blob/main/docs/") and "#" in href
    btn = ".info-btn[data-help-for='train-lambda']"
    page.focus(btn)
    page.keyboard.press("Enter")
    assert page.get_attribute(btn, "aria-expanded") == "true"
    assert page.is_visible("#help-train-lambda")
    assert "(1 − λ)" in page.text_content("#help-train-lambda")
    assert page.get_attribute("#help-train-lambda a", "href").endswith("equations.md#1-the-lejepa-objective")
    page.keyboard.press("Escape")
    assert page.get_attribute(btn, "aria-expanded") == "false"
    assert not page.is_visible("#help-train-lambda")
    assert page.evaluate("() => document.activeElement.dataset.helpFor") == "train-lambda"
    # from the link inside the panel, Esc also closes and returns focus to the button
    page.keyboard.press("Space")
    assert page.is_visible("#help-train-lambda")
    page.keyboard.press("Tab")
    assert page.evaluate("() => document.activeElement.tagName") == "A"
    page.keyboard.press("Escape")
    assert not page.is_visible("#help-train-lambda")
    assert page.evaluate("() => document.activeElement.dataset.helpFor") == "train-lambda"
    # opening another one closes the first; a click elsewhere closes it
    page.click(btn)
    page.click(".info-btn[data-help-for='train-steps']")
    assert not page.is_visible("#help-train-lambda") and page.is_visible("#help-train-steps")
    page.click("#train-collapse")
    assert not page.is_visible("#help-train-steps")
    con.check()
    ctx.close()


# Builds N PNG files on a canvas inside the page and drops / pastes them like a real drag.
DROP_JS = """async ({n, mode, csv}) => {
  const files = [];
  for (let i = 0; i < n; i++) {
    const c = document.createElement('canvas'); c.width = 40 + i; c.height = 36;
    const g = c.getContext('2d'); g.fillStyle = `rgb(${10 * i % 255}, ${200 - 5 * i}, 120)`; g.fillRect(0, 0, c.width, c.height);
    g.fillStyle = '#fff'; g.fillRect(4 + i % 20, 6, 10, 12);
    const blob = await new Promise((r) => c.toBlob(r, 'image/png'));
    files.push(new File([blob], `drop_${i}.png`, {type: 'image/png'}));
  }
  if (csv) {
    const rows = ['a,b,c'];
    for (let i = 0; i < 400; i++) rows.push(`${Math.sin(i / 9).toFixed(4)},${Math.cos(i / 7).toFixed(4)},${(i % 13) / 13}`);
    files.push(new File([rows.join('\\n')], 'signal.csv', {type: 'text/csv'}));
  }
  const dt = new DataTransfer();
  for (const f of files) dt.items.add(f);
  const zone = document.getElementById('dropzone');
  if (mode === 'paste') {
    zone.focus();
    const ev = new Event('paste', {bubbles: true, cancelable: true});
    Object.defineProperty(ev, 'clipboardData', {value: dt});
    zone.dispatchEvent(ev);
    return {over: null};
  }
  zone.dispatchEvent(new DragEvent('dragenter', {bubbles: true, cancelable: true, dataTransfer: dt}));
  zone.dispatchEvent(new DragEvent('dragover', {bubbles: true, cancelable: true, dataTransfer: dt}));
  const over = zone.classList.contains('is-over');
  zone.dispatchEvent(new DragEvent('drop', {bubbles: true, cancelable: true, dataTransfer: dt}));
  return {over};
}"""


def test_drop_and_paste_images_and_csv(server, browser):
    ctx, page, con = app_page(browser, server)
    status = lambda: page.text_content("#data-status") or ""  # noqa: E731
    assert page.get_attribute("#dropzone", "tabindex") == "0" and page.get_attribute("#dropzone", "role") == "button"
    # paste one image from Shapes mode: switches to My images and asks for more
    page.evaluate(DROP_JS, {"n": 1, "mode": "paste", "csv": False})
    wait_until(page, lambda: "so far" in status(), timeout=30)
    assert page.is_checked("input[name=dataset][value=images]")
    assert "at least 16" in status() and "15 more" in status()
    # pasting more adds to the set
    page.evaluate(DROP_JS, {"n": 15, "mode": "paste", "csv": False})
    wait_until(page, lambda: "16 images loaded locally" in status(), timeout=60)
    assert page.locator("#data-thumbs button").count() == 16
    # a drop replaces the set, with the same loader and limits as the file buttons
    r = page.evaluate(DROP_JS, {"n": 20, "mode": "drop", "csv": False})
    assert r["over"] is True
    wait_until(page, lambda: "20 images loaded locally" in status(), timeout=60)
    assert not page.evaluate("() => document.getElementById('dropzone').classList.contains('is-over')")
    assert json.loads(page.evaluate("() => JSON.stringify(window.jepaApp.data.n)")) == 20
    assert page.locator("#data-thumbs button").count() == 20
    # a CSV dropped switches to time series and previews it
    page.evaluate(DROP_JS, {"n": 0, "mode": "drop", "csv": True})
    wait_until(page, lambda: "numeric columns" in status(), timeout=30)
    assert page.is_checked("input[name=dataset][value=series]")
    assert "400 rows × 3 numeric columns from signal.csv" in status()
    assert page.is_visible("#series-panel")
    assert "Drop a CSV" in page.text_content("#dropzone-title")
    con.check()
    ctx.close()


def test_config_editor_round_trip(server, browser, tmp_path):
    ctx, page, con = app_page(browser, server, "#export")
    text = page.input_value("#cfg-text")
    assert json.loads(text) == json.loads(page.evaluate("() => JSON.stringify(window.jepaApp.config)"))
    assert page.get_attribute("#cfg-state", "data-state") == "clean"
    # a valid edit: Validate, then Apply moves the Train controls
    c = json.loads(text)
    c["train"]["max_steps"] = 123
    c["objective"]["lambda"] = 0.2
    page.fill("#cfg-text", json.dumps(c, indent=2))
    assert page.get_attribute("#cfg-state", "data-state") == "dirty"
    assert page.is_disabled("#cfg-save") and "Apply or revert" in page.text_content("#cfg-save-why")
    page.click("#cfg-validate")
    assert page.locator("#cfg-result .notice.good").count() == 1
    assert cfg(page, "train.max_steps") != 123                # validating changes nothing
    page.click("#cfg-apply")
    assert cfg(page, "train.max_steps") == 123 and cfg(page, "objective.lambda") == 0.2
    assert page.input_value("#train-steps-num") == "123" and page.input_value("#train-lambda-num") == "0.2"
    assert page.get_attribute("#cfg-state", "data-state") == "clean" and not page.is_disabled("#cfg-save")
    # an invalid edit: schema errors listed inline with their paths; nothing applied
    c["objective"]["lambda"] = 2
    c["typo"] = 1
    page.fill("#cfg-text", json.dumps(c, indent=2))
    page.click("#cfg-apply")
    errs = [t.strip() for t in page.locator("#cfg-result .errors li").all_text_contents()]
    assert any(e.startswith("$.objective.lambda: 2 > maximum 1") for e in errs), errs
    assert any("unknown key 'typo'" in e for e in errs), errs
    assert page.get_attribute("#cfg-text", "aria-invalid") == "true"
    assert "cfg-result" in page.get_attribute("#cfg-text", "aria-describedby")
    assert cfg(page, "objective.lambda") == 0.2
    # an error's path selects the key in the editor
    page.locator("#cfg-result .err-loc", has_text="objective.lambda").click()
    assert page.evaluate("() => { const t = document.getElementById('cfg-text'); return t.value.slice(t.selectionStart, t.selectionEnd); }") == '"lambda"'
    # a JSON syntax error says where
    page.fill("#cfg-text", '{\n  "version": 1,,\n}')
    page.click("#cfg-validate")
    assert "line 2" in page.text_content("#cfg-result")
    # confirm before an opened file replaces unapplied edits
    good = tmp_path / "seven.json"
    good.write_text(json.dumps({"version": 1, "name": "seven", "seed": 7}))
    page.set_input_files("#cfg-open", str(good))
    page.wait_for_selector("#cfg-confirm:not([hidden])")
    assert "seven.json" in page.text_content("#cfg-confirm-text")
    page.click("#cfg-confirm-no")
    assert page.input_value("#cfg-text") == '{\n  "version": 1,,\n}' and cfg(page, "seed") == 0
    page.set_input_files("#cfg-open", str(good))
    page.wait_for_selector("#cfg-confirm:not([hidden])")
    page.click("#cfg-confirm-yes")
    wait_until(page, lambda: cfg(page, "seed") == 7, timeout=10)
    assert cfg(page, "name") == "seven" and page.get_attribute("#cfg-state", "data-state") == "clean"
    page.click("#tab-data")
    assert page.input_value("#data-seed") == "7"
    page.click("#tab-export")
    # revert discards edits; Download saves a config the desktop validator accepts
    page.fill("#cfg-text", "{}")
    page.click("#cfg-revert")
    assert json.loads(page.input_value("#cfg-text"))["seed"] == 7
    with page.expect_download() as dl:
        page.click("#cfg-save")
    from jepa_studio.config import load_schema, validate
    assert validate(json.loads(Path(dl.value.path()).read_text()), load_schema()) == []
    con.check()
    ctx.close()


def test_stepper_empty_states_and_next_step(server, browser):
    ctx, page, con = app_page(browser, server)
    state = lambda k: page.get_attribute(f"#stepper li[data-step={k}]", "data-state")  # noqa: E731
    assert state("data") == "done" and state("trained") == "current" and state("inspected") == "upcoming"
    assert page.get_attribute("#stepper li[data-step=trained] button", "aria-current") == "step"
    # empty states and disabled buttons that say why, before any run
    for tab, sel in (("inspect", "#inspect-empty"), ("evaluate", "#evaluate-empty"), ("export", "#export-empty")):
        page.click(f"#tab-{tab}")
        assert page.is_visible(sel)
    page.click("#tab-evaluate")
    assert page.is_disabled("#eval-run") and "Needs a trained model" in page.text_content("#eval-run-why")
    page.click("#tab-export")
    assert page.is_disabled("#exp-csv") and page.is_visible("#exp-emb-why")
    page.click("#export-empty [data-goto=train]")
    assert page.get_attribute("#tab-train", "aria-selected") == "true"
    # a short run: Quick look preset, 20 steps
    page.check("input[name=preset][value=quick]", force=True)
    page.fill("#train-steps-num", "20")
    page.press("#train-steps-num", "Enter")
    page.click("#train-start")
    wait_until(page, lambda: page.text_content("#train-status") in ("Finished", "Error") or
               (page.text_content("#train-status") or "").startswith(("Error", "Could not")), timeout=300)
    assert page.text_content("#train-status") == "Finished"
    assert page.is_visible("#train-next") and "Run finished" in page.text_content("#train-next")
    assert page.locator(".toast", has_text="Run finished").count() == 1
    assert state("trained") == "done" and state("inspected") == "current"
    assert "inspect" in page.text_content("#stepper-hint").lower()
    page.click("#train-next-go")
    assert page.get_attribute("#tab-inspect", "aria-selected") == "true"
    assert not page.is_visible("#inspect-empty")
    wait_until(page, lambda: state("inspected") == "done", timeout=60)
    assert state("evaluated") == "current"
    page.click("#stepper li[data-step=evaluated] button")
    assert page.get_attribute("#tab-evaluate", "aria-selected") == "true"
    assert not page.is_disabled("#eval-run") and not page.is_visible("#eval-run-why")
    con.check()
    ctx.close()


def test_preferences_survive_reload(server, browser):
    ctx, page, con = app_page(browser, server, "#train")
    page.check("input[name=preset][value=thorough]", force=True)
    page.click("#tab-explain")
    page.click("#theme-toggle")
    assert page.get_attribute("html", "data-theme") == "dark"
    page.goto(f"{server}/app.html")
    ready(page)
    assert page.get_attribute("html", "data-theme") == "dark"
    assert page.get_attribute("#tab-explain", "aria-selected") == "true"
    assert page.is_checked("input[name=preset][value=thorough]")
    assert page.input_value("#train-steps-num") == "1000" and cfg(page, "train.max_steps") == 1000
    # an explicit hash still wins over the remembered tab
    page.goto(f"{server}/app.html#data")
    ready(page)
    assert page.get_attribute("#tab-data", "aria-selected") == "true"
    con.check()
    ctx.close()


def test_app_works_when_storage_throws(server, browser):
    """Private windows, blocked site data and sandboxed frames make localStorage throw."""
    ctx, page = new_page(browser)
    ctx.add_init_script("""
      (() => {
        const deny = () => { throw new DOMException('storage is disabled', 'SecurityError'); };
        for (const m of ['getItem', 'setItem', 'removeItem', 'clear', 'key']) Storage.prototype[m] = deny;
        Object.defineProperty(window, 'localStorage', { get: deny, configurable: true });
        Object.defineProperty(window, 'sessionStorage', { get: deny, configurable: true });
      })();
    """)
    con = Console(page)
    page.goto(f"{server}/index.html")
    ready(page)
    page.click("#theme-toggle")
    page.goto(f"{server}/app.html#train")
    ready(page)
    page.click("#theme-toggle")
    assert page.get_attribute("html", "data-theme") in ("dark", "light")
    page.check("input[name=preset][value=quick]", force=True)
    page.fill("#train-steps-num", "20")
    page.press("#train-steps-num", "Enter")
    page.click("#tab-explain")
    page.click("#tab-train")
    page.click("#train-start")
    wait_until(page, lambda: page.text_content("#train-status") in ("Finished", "Error") or
               (page.text_content("#train-status") or "").startswith(("Error", "Could not")), timeout=300)
    assert page.text_content("#train-status") == "Finished"
    con.check()
    ctx.close()
