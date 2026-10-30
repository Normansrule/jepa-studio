"""Browser side of the science features (site/js/science/).

* Node: the DOM-free physics and colour code (tests/js/science_sims.mjs) against the same bounds as the
  Python simulations, and parity with jev.science (colour table, Page-Thorne flux, shadow size).
* Chromium (Python Playwright, SwiftShader WebGL): site/ is served as-is and a throwaway harness page is
  injected with page.route (no committed test page). It mounts the WebGL2 black hole and the 2-D
  simulations under the same Content-Security-Policy as the app and checks the pixels: black shadow at the
  centre with the analytic radius, the lensed disk arching over it, Doppler/beaming asymmetry, the
  Interstellar mode's symmetry, snapshots, and no console errors or CSP violations.

Skipped when node / Playwright / Chromium are not installed.
"""
from __future__ import annotations

import functools
import http.server
import json
import math
import shutil
import socketserver
import subprocess
import threading
import time
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parent.parent
SITE = ROOT / "site"
NODE = shutil.which("node")

CSP = ("default-src 'self'; script-src 'self'; style-src 'self'; img-src 'self' data: blob:; font-src 'self'; "
       "connect-src 'self'; worker-src 'self'; object-src 'none'; base-uri 'none'; form-action 'none'")


# ----------------------------------------------------------------------------------------------
# Node: physics + parity with Python
# ----------------------------------------------------------------------------------------------


@pytest.fixture(scope="module")
def node_out():
    if NODE is None:
        pytest.skip("node is not installed")
    r = subprocess.run([NODE, str(ROOT / "tests" / "js" / "science_sims.mjs")], capture_output=True, text=True,
                       timeout=240)
    assert r.returncode == 0, r.stderr
    return json.loads(r.stdout)


def test_js_physics_bounds(node_out):
    o = node_out
    assert o["figure8_energy_drift"] < 1e-5 and o["figure8_return_error"] < 1e-3
    assert o["cluster_energy_drift"] < 1e-3
    assert o["pendulum_energy_drift"] < 1e-6 and o["pendulum_separation"] > 1e-5      # grew from 1e-9
    assert o["cfl_guard"] is True and o["wave_courant"] == pytest.approx(0.5)
    assert o["wave_energy_variation"] < 1e-10
    assert o["wave_front_radius"] == pytest.approx(o["wave_front_expected"], abs=0.02)
    assert o["kepler_position_error"] < 1e-3
    assert o["kepler_apoapsis"] == pytest.approx([-1.6, 0.0], abs=1e-9)
    assert all(abs(v) < 60 for v in o["lorenz_state"])


def test_js_matches_python_physics(node_out):
    from jev.science import color, kerr
    for T, xy in node_out["chromaticity"].items():
        assert xy == pytest.approx(list(color.xy_chromaticity([float(T)])[0]), abs=2e-4)
    # the shader's luminance column is log10(Y / Y_6500): ~0 at 6500 K
    assert node_out["table_6500"][3] == pytest.approx(0.0, abs=0.02)
    py = kerr.page_thorne_flux(np.array([7.0, 10.0, 20.0, 50.0]), 0.0)
    js = np.array(node_out["fluxPT"])
    assert js / py == pytest.approx(np.full(4, 8 * math.pi / 3), rel=1e-6)   # same profile, no 3/(8 pi)
    assert node_out["shadow_30"] == pytest.approx(math.asin(3 * math.sqrt(3) * math.sqrt(1 - 2 / 30) / 30), rel=1e-12)


# ----------------------------------------------------------------------------------------------
# Chromium
# ----------------------------------------------------------------------------------------------

HARNESS_HTML = f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="{CSP}">
<title>science harness</title></head><body>
<canvas id="bh" width="192" height="108"></canvas>
<canvas id="bh2" width="192" height="108"></canvas>
<canvas id="s1" width="160" height="90"></canvas><canvas id="s2" width="160" height="90"></canvas>
<canvas id="s3" width="160" height="90"></canvas><canvas id="s4" width="160" height="90"></canvas>
<pre id="out"></pre>
<script type="module" src="/__science_harness.js"></script></body></html>"""

HARNESS_JS = r"""
import { mountBlackHole, shadowAngle, capabilities } from '/js/science/blackhole.js';
import { mountSim } from '/js/science/sims.js';

function pixels(canvas) {
  const c = document.createElement('canvas');
  c.width = canvas.width; c.height = canvas.height;
  const g = c.getContext('2d');
  g.drawImage(canvas, 0, 0);
  const d = g.getImageData(0, 0, c.width, c.height).data;
  const L = [];
  for (let y = 0; y < c.height; y++) {
    const row = [];
    for (let x = 0; x < c.width; x++) { const i = 4 * (y * c.width + x); row.push(0.2126 * d[i] + 0.7152 * d[i + 1] + 0.0722 * d[i + 2]); }
    L.push(row);
  }
  return L;
}
const nextFrame = () => new Promise((r) => requestAnimationFrame(() => requestAnimationFrame(r)));

async function run() {
  const out = { capabilities };
  const bh = document.getElementById('bh');
  // 1. shadow against the grid sky, disk off: exact size, black centre
  const h = mountBlackHole(bh, { distance: 30, inclination: 60, fov: 50, disk: false, background: 'grid', quality: 'high', reducedMotion: true });
  h.render(); await nextFrame();
  out.grid = pixels(bh);
  out.shadowAngle = shadowAngle(30);
  // 2. disk on, physical (Doppler + beaming), no stars: asymmetric
  h.setParams({ disk: true, background: 'none', inclination: 80, doppler: true, beaming: true });
  h.render(); await nextFrame();
  out.physical = pixels(bh);
  // 3. Interstellar mode: symmetric
  h.setParams({ interstellar: true });
  h.render(); await nextFrame();
  out.interstellar = pixels(bh);
  const blob = await h.snapshot();
  out.snapshot = { type: blob.type, size: blob.size };
  out.params = h.getParams();
  // keyboard orbit + zoom change the view
  bh.focus();
  bh.dispatchEvent(new KeyboardEvent('keydown', { key: 'ArrowRight', bubbles: true }));
  bh.dispatchEvent(new KeyboardEvent('keydown', { key: '+', bubbles: true }));
  out.afterKeys = h.getParams();
  h.destroy();
  // 4. low quality (half-resolution framebuffer + blit), stars, animated disk
  const h2 = mountBlackHole(document.getElementById('bh2'), { quality: 'low', reducedMotion: false, autoRotate: true });
  await nextFrame(); await nextFrame();
  out.low = pixels(document.getElementById('bh2'));
  h2.destroy();
  // 5. simulations
  out.sims = {};
  const kinds = [['s1', 'nbody', { bodies: 3 }], ['s2', 'pendulum', {}], ['s3', 'wave', {}], ['s4', 'lorenz', {}]];
  const handles = kinds.map(([id, k, o]) => [k, mountSim(document.getElementById(id), k, { ...o, reducedMotion: false })]);
  for (let i = 0; i < 20; i++) await nextFrame();
  for (const [k, s] of handles) {
    const L = pixels(document.getElementById(kinds.find((x) => x[1] === k)[0]));
    const flat = L.flat();
    const bg = flat[0];
    out.sims[k] = { varied: flat.filter((v) => Math.abs(v - bg) > 20).length, metrics: s.metrics(), running: s.running };
    s.destroy();
  }
  // reduced motion: not running, but drawn
  const rm = mountSim(document.getElementById('s1'), 'double_pendulum', { reducedMotion: true });
  out.reduced = { running: rm.running, t: rm.metrics().t };
  rm.destroy();
  return out;
}
run().then((o) => { document.getElementById('out').textContent = JSON.stringify(o); document.documentElement.dataset.done = '1'; })
  .catch((e) => { document.getElementById('out').textContent = JSON.stringify({ error: String(e && e.stack || e) }); document.documentElement.dataset.done = '1'; });
"""


class _Handler(http.server.SimpleHTTPRequestHandler):
    extensions_map = {**http.server.SimpleHTTPRequestHandler.extensions_map, ".js": "text/javascript",
                      ".mjs": "text/javascript", ".json": "application/json"}

    def log_message(self, *args):
        pass


@pytest.fixture(scope="module")
def server():
    srv = socketserver.ThreadingTCPServer(("127.0.0.1", 0), functools.partial(_Handler, directory=str(SITE)))
    srv.daemon_threads = True
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield f"http://127.0.0.1:{srv.server_address[1]}"
    srv.shutdown()
    srv.server_close()


@pytest.fixture(scope="module")
def harness(server):
    sync_api = pytest.importorskip("playwright.sync_api")
    try:
        pw = sync_api.sync_playwright().start()
    except Exception as e:  # pragma: no cover
        pytest.skip(f"playwright unavailable: {e}")
    try:
        browser = pw.chromium.launch(args=["--use-angle=swiftshader", "--enable-unsafe-swiftshader",
                                           "--ignore-gpu-blocklist"])
    except Exception as e:  # pragma: no cover
        pw.stop()
        pytest.skip(f"chromium not installed for playwright: {e}")
    ctx = browser.new_context(viewport={"width": 800, "height": 600}, device_scale_factor=1)
    page = ctx.new_page()
    errors, csp = [], []

    def on_console(msg):
        if "Content Security Policy" in msg.text or "Content-Security-Policy" in msg.text:
            csp.append(msg.text)
        if msg.type == "error":
            errors.append(msg.text)
    page.on("console", on_console)
    page.on("pageerror", lambda e: errors.append(f"pageerror: {e}"))
    page.route("**/__science_harness.html", lambda route: route.fulfill(
        status=200, content_type="text/html; charset=utf-8", body=HARNESS_HTML, headers={"Content-Security-Policy": CSP}))
    page.route("**/__science_harness.js", lambda route: route.fulfill(
        status=200, content_type="text/javascript", body=HARNESS_JS))
    page.goto(f"{server}/__science_harness.html")
    t0 = time.time()
    while page.get_attribute("html", "data-done") != "1":
        if time.time() - t0 > 180:
            raise AssertionError("harness timed out")
        page.wait_for_timeout(250)
    data = json.loads(page.text_content("#out"))
    yield {"data": data, "errors": errors, "csp": csp}
    browser.close()
    pw.stop()


def test_no_console_errors_or_csp_violations(harness):
    assert "error" not in harness["data"], harness["data"].get("error")
    assert not harness["csp"], harness["csp"]
    assert not harness["errors"], harness["errors"]


def test_webgl_shadow_size_and_black_centre(harness):
    d = harness["data"]
    L = np.array(d["grid"])                    # rows from the top
    h, w = L.shape
    assert L[h // 2 - 1: h // 2 + 1, w // 2 - 1: w // 2 + 1].max() < 8     # the shadow is black
    row = L[h // 2]
    dark = np.nonzero(row < 4)[0]
    half = (dark.max() - dark.min() + 1) / 2
    tan_half = math.tan(math.radians(50) / 2)
    ang = math.atan(half / (w / 2) * tan_half)
    px = math.atan(tan_half / (w / 2))
    assert ang == pytest.approx(d["shadowAngle"], abs=2 * px)
    assert row[: int(dark.min()) - 2].mean() > 25                            # lensed sky outside


def test_webgl_disk_arches_over_the_shadow_and_beaming(harness):
    d = harness["data"]
    L = np.array(d["physical"])
    h, w = L.shape
    assert L[h // 2, w // 2] < 8
    col = L[:, w // 2]
    top = int(col[: h // 2].argmax())
    assert col[top] > 100                      # the far side of the disk, lensed over the top
    # approaching (left) side brighter with Doppler beaming
    assert L[:, : w // 2].sum() > 1.3 * L[:, w // 2:].sum()
    Li = np.array(d["interstellar"])
    left, right = Li[:, : w // 2].sum(), Li[:, w // 2:][:, ::-1].sum()
    assert abs(left - right) / (left + right) < 0.03
    assert np.abs(Li - Li[:, ::-1]).mean() < 3


def test_webgl_api_snapshot_controls_and_low_quality(harness):
    d = harness["data"]
    assert d["capabilities"]["spin"] is False
    assert d["snapshot"]["type"] == "image/png" and d["snapshot"]["size"] > 500
    assert d["afterKeys"]["azimuth"] == pytest.approx(d["params"]["azimuth"] + 5)
    assert d["afterKeys"]["distance"] < d["params"]["distance"]
    low = np.array(d["low"])
    assert low.max() > 100 and low[low.shape[0] // 2, low.shape[1] // 2] < 10


def test_canvas_simulations_draw_and_report(harness):
    sims = harness["data"]["sims"]
    for k in ("nbody", "pendulum", "wave", "lorenz"):
        assert sims[k]["varied"] > 30, (k, sims[k])
        assert sims[k]["running"] is True
    assert sims["nbody"]["metrics"]["energy_rel_error"] < 1e-4
    assert sims["pendulum"]["metrics"]["energy_rel_error"] < 1e-5
    assert sims["wave"]["metrics"]["cfl"] <= 1
    rm = harness["data"]["reduced"]
    assert rm["running"] is False and rm["t"] > 1        # jumped ahead instead of animating
