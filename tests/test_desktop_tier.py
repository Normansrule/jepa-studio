"""The front end in the desktop tier, against the real local backend (`jepa_studio.server`).

The Tauri window is replaced by Playwright: the test injects what the shell injects
(`window.__JEPA_DESKTOP__ = {port, token}` and a `__TAURI__.core.invoke` whose
`pick_data_folder` returns a folder the backend was told it may read), serves site/ from the
backend's own origin (so the page's CSP `connect-src 'self'` covers the API, as the shell's
port-specific CSP does), and checks:

  * the tier is detected and the desktop-only controls appear
  * "Use a folder on this computer…" previews real samples and views made by the backend
  * a short PyTorch run starts from the page and finishes
  * the Inspect tab draws backend saliency for it, and [CLS] attention for a vit-tiny run
  * a folder the backend was NOT granted is refused

Skipped when Playwright or its Chromium build is not installed.
"""
from __future__ import annotations

import io
import json
import mimetypes
import secrets
import threading
import time
from pathlib import Path
from urllib.parse import urlparse

import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
sync_api = pytest.importorskip("playwright.sync_api")


def _image_folder(root: Path) -> Path:
    from PIL import Image
    d = root / "pets"
    for cls, colour in (("cats", (200, 60, 40)), ("dogs", (40, 90, 200))):
        (d / cls).mkdir(parents=True)
        for i in range(16):
            im = Image.new("RGB", (40, 36), colour)
            im.paste((255 - 8 * i, 8 * i, 120), (4 + i, 4, 14 + i, 20))
            im.save(d / cls / f"{cls}_{i:02d}.png")
    return d


@pytest.fixture(scope="module")
def backend(tmp_path_factory):
    from jepa_studio.server import serve
    tmp = tmp_path_factory.mktemp("desk")
    data = _image_folder(tmp)
    token = secrets.token_urlsafe(32)
    ready = io.StringIO()
    t = threading.Thread(target=serve, kwargs=dict(root=str(tmp / "workspace"), port=0, token=token,
                                                   allowed_dirs=[str(data)], ready_file=ready), daemon=True)
    t.start()
    for _ in range(100):
        if ready.getvalue():
            break
        time.sleep(0.05)
    port = json.loads(ready.getvalue().splitlines()[0])["port"]
    yield {"port": port, "token": token, "data": data, "outside": tmp}


@pytest.fixture(scope="module")
def browser():
    try:
        pw = sync_api.sync_playwright().start()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"playwright unavailable: {e}")
    try:
        b = pw.chromium.launch()
    except Exception as e:  # pragma: no cover
        pw.stop()
        pytest.skip(f"chromium not installed for playwright: {e}")
    yield b
    b.close()
    pw.stop()


def _serve_site(route):
    path = urlparse(route.request.url).path
    if path.startswith("/api/"):
        return route.continue_()
    f = (SITE / (path.lstrip("/") or "index.html")).resolve()
    if SITE.resolve() not in f.parents or not f.is_file():
        return route.fulfill(status=404, body="not found")
    ctype = {".js": "text/javascript", ".mjs": "text/javascript", ".json": "application/json",
             ".woff2": "font/woff2", ".svg": "image/svg+xml"}.get(f.suffix) or mimetypes.guess_type(f.name)[0]
    route.fulfill(status=200, body=f.read_bytes(), content_type=ctype or "application/octet-stream")


def _page(browser, backend, picked: Path):
    ctx = browser.new_context(viewport={"width": 1280, "height": 800}, reduced_motion="reduce")
    ctx.add_init_script(
        "Object.defineProperty(window, '__JEPA_DESKTOP__', {value: Object.freeze(%s)});"
        "window.__TAURI__ = {core: {invoke: async (cmd) => {"
        "  if (cmd === 'pick_data_folder') return %s;"
        "  throw new Error('unknown command ' + cmd); }}};"
        % (json.dumps({"port": backend["port"], "token": backend["token"]}), json.dumps(str(picked))))
    page = ctx.new_page()
    errors: list[str] = []
    page.on("pageerror", lambda e: errors.append(str(e)))
    page.on("console", lambda m: errors.append(m.text) if m.type == "error" and "Failed to load resource" not in m.text
            else None)
    page.route("**/*", _serve_site)
    page.goto(f"http://127.0.0.1:{backend['port']}/app.html")
    t0 = time.time()
    while page.get_attribute("html", "data-ready") != "true":
        assert time.time() - t0 < 30, "app did not become ready"
        page.wait_for_timeout(200)
    return ctx, page, errors


def _wait(page, fn, timeout=180):
    t0 = time.time()
    while time.time() - t0 < timeout:
        if fn():
            return
        page.wait_for_timeout(250)
    raise AssertionError("timed out")


def test_desktop_folder_preview_and_training(browser, backend):
    ctx, page, errors = _page(browser, backend, backend["data"])
    assert page.text_content("#tier-label") == "Desktop"
    page.check("input[name=dataset][value=images]", force=True)
    assert page.is_visible("#desktop-folder-images")
    page.click("#desktop-folder-images")
    _wait(page, lambda: "Using" in (page.text_content("#data-status") or "") or
          "not used" in (page.text_content("#data-status") or ""), timeout=60)
    status = page.text_content("#data-status")
    assert "Using" in status and "32 items" in status and "cats" in status, status
    assert page.locator("#data-thumbs img").count() == 6
    assert page.locator("#view-globals img").count() >= 1 and page.locator("#view-locals img").count() >= 1
    cfg = json.loads(page.evaluate("() => JSON.stringify(window.jepaApp.config.data)"))
    assert cfg["kind"] == "images" and cfg["path"] == str(backend["data"])

    # a short real PyTorch run started from the page
    page.click("#tab-train")
    page.select_option("#train-batch", "32")
    page.fill("#train-steps", "20")
    page.dispatch_event("#train-steps", "change")
    page.click("#train-start")
    _wait(page, lambda: (page.text_content("#train-status") or "") in ("Finished", "Stopped") or
          (page.text_content("#train-status") or "").startswith("Error"), timeout=300)
    assert page.text_content("#train-status") == "Finished", page.text_content("#train-status")

    # Inspect: saliency for a desktop-folder sample, computed by the backend (/saliency) and drawn
    # over the PNG of the exact input it explained
    page.click("#tab-inspect")
    _wait(page, lambda: page.get_attribute("#inspect-verdict", "data-status") in ("good", "warn", "bad"), timeout=120)
    page.click("#inspect-thumbs button >> nth=2")
    _wait(page, lambda: page.locator("#saliency canvas").count() == 2 or page.locator("#saliency p").count() > 0,
          timeout=60)
    assert page.locator("#saliency canvas").count() == 2, page.text_content("#saliency")
    assert page.evaluate("() => document.querySelector('#saliency canvas[role=img]').width") == 32
    assert page.locator("#saliency canvas[aria-label^='ViT attention']").count() == 0

    # a vit-tiny run on the same folder adds the [CLS] attention map as a third figure
    page.evaluate("""async () => {
      const c = JSON.parse(JSON.stringify(window.jepaApp.config));
      Object.assign(c.model, {arch: 'vit-tiny', embed_dim: 32, proj_dim: 8, proj_hidden: 32, depth: 2, heads: 4, patch_size: 8});
      Object.assign(c.train, {max_steps: 2, batch_size: 16});
      c.name = 'vit-inspect';
      await window.jepaApp.backend.startRun(c);
    }""")
    _wait(page, lambda: page.evaluate("() => window.jepaApp.backend.log.some((e) => e.event === 'end' || e.event === 'error')"),
          timeout=180)
    page.click("#inspect-thumbs button >> nth=5")
    _wait(page, lambda: page.locator("#saliency canvas").count() == 3 or page.locator("#saliency p").count() > 0,
          timeout=60)
    assert page.locator("#saliency canvas[aria-label^='ViT attention']").count() == 1, page.text_content("#saliency")
    assert not errors, errors
    ctx.close()


def test_desktop_refuses_folder_outside_grant(browser, backend):
    ctx, page, _ = _page(browser, backend, backend["outside"])
    page.check("input[name=dataset][value=images]", force=True)
    page.click("#desktop-folder-images")
    _wait(page, lambda: "not used" in (page.text_content("#data-status") or ""), timeout=60)
    assert "outside" in page.text_content("#data-status") or "not allowed" in page.text_content("#data-status") \
        or "grant" in page.text_content("#data-status"), page.text_content("#data-status")
    cfg = json.loads(page.evaluate("() => JSON.stringify(window.jepaApp.config.data)"))
    assert cfg["path"] is None and cfg["kind"] == "synthetic-shapes"
    ctx.close()
