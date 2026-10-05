"""Generate the README visuals in assets/ from the real runs, checkpoints and code in this repo.

    PYTHONPATH=. python3 scripts/make_visuals.py            # everything
    PYTHONPATH=. python3 scripts/make_visuals.py sigreg hero # a subset
    PYTHONPATH=. python3 scripts/make_visuals.py --measure   # also re-run the throughput benchmark

Outputs (all in assets/):
    architecture.svg     animated LeJEPA data flow (diagram; loss form copied from jepa_studio/losses.py)
    sigreg.gif           proj_hist records of runs/shapes-desktop/events.jsonl (+ one computed random-init panel)
    jepa-vs-pixels.svg   pixel reconstruction vs JEPA (diagram)
    throughput.svg       auto-tuning stages, samples/s (assets/throughput-benchmark.json) + web backend probe
                         (assets/screens/headless-benchmark.json)
    web-vs-desktop.svg   capability table, checked against site/js and jepa_studio
    hero.gif             dataset -> loss curves -> embedding PCA -> planning bot

Nothing here invents numbers: every plotted value is read from a run file, a checkpoint, or is
computed by the repo's own code with fixed seeds. Where a figure interpolates between real
snapshots (hero, embedding panel) the figure says so. Needs matplotlib, Pillow, numpy, torch;
gifsicle (optional) for smaller GIFs.
"""
from __future__ import annotations

import base64
import io
import json
import math
import shutil
import subprocess
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
ASSETS = ROOT / "assets"
RUN = ROOT / "runs" / "shapes-desktop"

# ----------------------------------------------------------------------------- tokens (docs/STYLE.md)
LIGHT = dict(view="#00897b", encoder="#b86e00", predictor="#5e3fc0", pred="#3c8fe8", sigreg="#d42f5f",
             bg="#f5f6f8", surface="#ffffff", surface2="#eef0f4", ink="#172033", ink2="#4a5468",
             ink3="#636c7e", line="#d9dde5", line_strong="#aeb6c4", good="#0b8a0b", warn="#a86f00",
             bad="#c43333", cat=["#2a78d6", "#eb6834", "#1baf7a", "#4a3aa7", "#e87ba4"])
DARK = dict(view="#129e8e", encoder="#c4850a", predictor="#8a5ce0", pred="#4196f0", sigreg="#e84a66",
            bg="#121826", surface="#1b2437", surface2="#232e45", ink="#e8ecf4", ink2="#b4bccb",
            ink3="#97a1b4", line="#2e3a54", line_strong="#4a587a", good="#35c235", warn="#fab219",
            bad="#f06a6a", cat=["#3987e5", "#d95926", "#199e70", "#9085e9", "#d55181"])
SANS = "-apple-system, BlinkMacSystemFont, 'Segoe UI', Helvetica, Arial, sans-serif"
SERIF = "'Latin Modern Roman', 'CMU Serif', 'Computer Modern', Georgia, 'Times New Roman', serif"
MONO = "ui-monospace, SFMono-Regular, Menlo, Consolas, monospace"


def css_vars(t: dict) -> str:
    keys = ["view", "encoder", "predictor", "pred", "sigreg", "bg", "surface", "surface2", "ink", "ink2",
            "ink3", "line", "line_strong", "good", "warn", "bad"]
    name = {"ink2": "ink-2", "ink3": "ink-3", "surface2": "surface-2", "line_strong": "line-strong"}
    return "".join(f"--{name.get(k, k)}:{t[k]};" for k in keys)


def theme_css(extra: str = "") -> str:
    """Light tokens by default, dark tokens under prefers-color-scheme: dark."""
    return (f":root{{{css_vars(LIGHT)}}}\n"
            f"@media (prefers-color-scheme: dark){{:root{{{css_vars(DARK)}}}}}\n" + extra)


def esc(s: str) -> str:
    return s.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def load_cfg() -> dict:
    return json.loads((RUN / "config.json").read_text())


def load_events(path: Path = RUN / "events.jsonl") -> list[dict]:
    return [json.loads(x) for x in path.read_text().splitlines() if x.strip()]


def png_data_uri(arr: np.ndarray, scale: int = 1) -> str:
    from PIL import Image
    a = (np.clip(arr, 0, 1) * 255 + 0.5).astype(np.uint8)
    im = Image.fromarray(a)
    if scale != 1:
        im = im.resize((a.shape[1] * scale, a.shape[0] * scale), Image.Resampling.NEAREST)
    buf = io.BytesIO()
    im.save(buf, format="PNG", optimize=True)
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def optimise_gif(path: Path, lossy: int = 0) -> None:
    if shutil.which("gifsicle"):
        args = ["gifsicle", "-O3", "--batch", str(path)]
        if lossy:
            args.insert(2, f"--lossy={lossy}")
        subprocess.run(args, check=True)


# ----------------------------------------------------------------------------- real data helpers

def shapes_images(n: int, seed: int = 0, size: int = 32):
    from jepa_studio.data import synthetic
    imgs, labels = [], []
    for i in range(n):
        im, c = synthetic.render(seed, i, size)
        imgs.append(im)
        labels.append(c)
    return np.stack(imgs), np.array(labels)


def build_model(cfg: dict, weights: Path | None, init_seed: int | None = None):
    """weights=None -> random init. init_seed=cfg seed reproduces the training run's step-0 weights
    (Trainer.fit seeds torch with cfg['seed'] and builds the model before touching torch's RNG)."""
    import torch
    from jepa_studio.models import build_encoder
    from jepa_studio.weights import load_state_dict
    torch.manual_seed(cfg["seed"] if init_seed is None else init_seed)
    m = build_encoder(cfg)
    if weights is not None:
        load_state_dict(m, weights)
    m.eval()
    return m


def fixed_probe_batch(cfg: dict):
    """The exact batch Trainer.fit uses for proj_hist: global view 0 of items 0..255, drawn from a
    fresh ImageViews(seed) (nothing consumes the view RNG before it when batch_size is fixed)."""
    import torch
    from jepa_studio.data.loaders import _hwc_to_chw
    from jepa_studio.data.views import ImageViews
    imgs, _ = shapes_images(256, cfg["seed"], max(cfg["data"]["image_size"], 32))
    views = ImageViews(cfg["data"], cfg["seed"])
    batch = []
    for im in imgs:
        g, _ = views(_hwc_to_chw(im))
        batch.append(g[0])
    return torch.stack(batch)


def probe_hist(model, batch, bins: int = 24) -> list:
    """Same computation as Trainer.projection_histograms (directions: seed 1234, 3 slices)."""
    import torch
    from jepa_studio.losses import random_directions
    dirs = random_directions(model.projector.net[-1].out_features, 3, torch.Generator().manual_seed(1234))
    with torch.no_grad():
        _, proj = model(batch)
    p = proj.float() @ dirs
    edges = torch.linspace(-4, 4, bins + 1)
    out = []
    for j in range(p.shape[1]):
        h = torch.histc(p[:, j].clamp(-4, 4), bins=bins, min=-4, max=4)
        out.append([float(x) for x in (h / h.sum() / (edges[1] - edges[0]))])
    return out, p.numpy()




def example_views(cfg: dict, index: int = 4, seed: int = 11):
    """One real Shapes image and the real multi-crop views the training pipeline makes of it."""
    from jepa_studio.data.loaders import _hwc_to_chw
    from jepa_studio.data.views import ImageViews
    from jepa_studio.data import synthetic
    im, _ = synthetic.render(cfg["seed"], index, 32)
    g, loc = ImageViews(cfg["data"], seed)(_hwc_to_chw(im))
    chw = lambda t: t.permute(1, 2, 0).numpy()  # noqa: E731
    return im, [chw(x) for x in g], [chw(x) for x in loc]


def svg_open(w: int, h: int, title: str, desc: str, extra_css: str = "") -> str:
    base = (f"svg{{font-family:{SANS};}}"
            ".bg{fill:var(--surface);stroke:var(--line);stroke-width:1}"
            ".t{fill:var(--ink)} .t2{fill:var(--ink-2)} .t3{fill:var(--ink-3)}"
            f".m{{font-family:{SERIF};fill:var(--ink)}} .mono{{font-family:{MONO};}}"
            ".h{font-weight:600}")
    return (f'<svg xmlns="http://www.w3.org/2000/svg" xmlns:xlink="http://www.w3.org/1999/xlink" '
            f'viewBox="0 0 {w} {h}" width="{w}" height="{h}" role="img" aria-labelledby="t d">'
            f'<title id="t">{esc(title)}</title><desc id="d">{esc(desc)}</desc>'
            f'<style>{theme_css(base + extra_css)}</style>'
            f'<rect class="bg" x="0.5" y="0.5" width="{w - 1}" height="{h - 1}" rx="6"/>')


def text(x, y, s, cls="t", size=13, anchor="start", weight=None, raw=False, extra=""):
    w = f' font-weight="{weight}"' if weight else ""
    body = s if raw else esc(s)
    return f'<text x="{x}" y="{y}" class="{cls}" font-size="{size}" text-anchor="{anchor}"{w}{extra}>{body}</text>'


def sub(s: str) -> str:
    return f'<tspan baseline-shift="sub" font-size="75%">{s}</tspan>'


def sup(s: str) -> str:
    return f'<tspan baseline-shift="super" font-size="70%">{s}</tspan>'


def it(s: str) -> str:
    return f'<tspan font-style="italic">{s}</tspan>'


# ----------------------------------------------------------------------------- 1. architecture.svg

def make_architecture() -> Path:
    cfg = load_cfg()
    lam, M = cfg["objective"]["lambda"], cfg["objective"]["num_slices"]
    ng, nl = cfg["data"]["n_global"], cfg["data"]["n_local"]
    gs, ls = cfg["data"]["global_scale"], cfg["data"]["local_scale"]
    K, E = cfg["model"]["proj_dim"], cfg["model"]["embed_dim"]
    im, gv, lv = example_views(cfg)
    W, H = 1000, 540
    css = (".box{fill:var(--surface);stroke-width:2}"
           ".v{stroke:var(--view)} .e{stroke:var(--encoder)} .p{stroke:var(--predictor)}"
           ".lp{stroke:var(--pred)} .ls{stroke:var(--sigreg)} .tot{stroke:var(--ink)}"
           ".fv{fill:var(--view)} .fe{fill:var(--encoder)} .fp{fill:var(--predictor)}"
           ".flp{fill:var(--pred)} .fls{fill:var(--sigreg)} .fink{fill:var(--ink)}"
           ".wire{fill:none;stroke-width:2;stroke-linejoin:round;stroke-linecap:round}"
           ".back{fill:none;stroke:var(--ink-3);stroke-width:1.5;stroke-dasharray:5 4}"
           ".tag{font-size:12px;font-weight:600}"
           ".cv{fill:var(--view)} .ce{fill:var(--encoder)} .cp{fill:var(--predictor)}"
           ".clp{fill:var(--pred)} .cls{fill:var(--sigreg)}"
           ".thumb{stroke:var(--line-strong);stroke-width:1;fill:none}"
           ".dot{stroke:var(--surface);stroke-width:1}"
           "@media (prefers-reduced-motion: reduce){.dot{display:none}}")
    o = [svg_open(W, H, "LeJEPA training step",
                  "An unlabeled image becomes 2 global and 4 local views; one shared encoder and projector "
                  "map every view to an embedding z; the loss is (1 - lambda) times the prediction term "
                  "(every view's z toward the mean of the global views) plus lambda times SIGReg "
                  "(Epps-Pulley test of random 1-D projections of z against N(0,1)).", css)]
    o.append('<defs>'
             + "".join(f'<marker id="a{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" '
                       f'orient="auto-start-reverse"><path d="M0,0 L10,5 L0,10 z" class="{f}"/></marker>'
                       for c, f in [("v", "fv"), ("e", "fe"), ("p", "fp"), ("lp", "flp"), ("ls", "fls"),
                                    ("k", "t3")])
             + '</defs>')
    o.append(text(20, 30, "One LeJEPA training step", size=16, weight=600))
    o.append(text(W - 20, 30, "jepa_studio/losses.py · lejepa_loss", "t3 mono", 12, "end"))

    # ---- row 1
    y0, y1 = 52, 236
    # input
    o.append(text(20, y0 + 14, "Unlabeled image", "t2", 12))
    o.append(f'<image x="20" y="{y0 + 26}" width="112" height="112" href="{png_data_uri(im, 4)}" '
             f'style="image-rendering:pixelated"/><rect class="thumb" x="20" y="{y0 + 26}" width="112" height="112"/>')
    o.append(text(20, y0 + 156, "32×32 RGB, no label", "t3", 11))
    # views panel (teal)
    vx, vw = 168, 214
    o.append(f'<rect class="box v" x="{vx}" y="{y0}" width="{vw}" height="{y1 - y0}" rx="6"/>')
    o.append(text(vx + 12, y0 + 20, "Views", "cv tag") + text(vx + 58, y0 + 20, "(augmentations)", "t3", 11))
    o.append(text(vx + 12, y0 + 40, f"{ng} global · 32×32 · {int(gs[0]*100)}–{int(gs[1]*100)}% area", "t2", 11))
    for i, g in enumerate(gv):
        x = vx + 12 + i * 70
        o.append(f'<image x="{x}" y="{y0 + 48}" width="64" height="64" href="{png_data_uri(g, 2)}" '
                 f'style="image-rendering:pixelated"/><rect class="thumb" x="{x}" y="{y0 + 48}" width="64" height="64"/>')
    o.append(text(vx + 12, y0 + 134, f"{nl} local · 16×16 · {int(ls[0]*100)}–{int(ls[1]*100)}% area", "t2", 11))
    for i, g in enumerate(lv):
        x = vx + 12 + i * 48
        o.append(f'<image x="{x}" y="{y0 + 142}" width="40" height="40" href="{png_data_uri(g, 2)}" '
                 f'style="image-rendering:pixelated"/><rect class="thumb" x="{x}" y="{y0 + 142}" width="40" height="40"/>')
    # encoder (amber)
    ex, ew = 420, 168
    o.append(f'<rect class="box e" x="{ex}" y="{y0 + 30}" width="{ew}" height="124" rx="6"/>')
    o.append(text(ex + 12, y0 + 52, "Shared encoder", "ce tag"))
    o.append(text(ex + 12, y0 + 74, f"f{sub('θ')}", "m", 17, raw=True, extra=' font-style="italic"'))
    o.append(text(ex + 12, y0 + 98, "ConvNet-tiny backbone", "t2", 12))
    o.append(text(ex + 12, y0 + 116, f"→ embedding ∈ ℝ{sup(str(E))}", "t2", 12, raw=True))
    o.append(text(ex + 12, y0 + 140, "same weights, all 6 views", "t3", 11))
    # projector (violet)
    px, pw = 626, 160
    o.append(f'<rect class="box p" x="{px}" y="{y0 + 30}" width="{pw}" height="124" rx="6"/>')
    o.append(text(px + 12, y0 + 52, "Projector", "cp tag"))
    o.append(text(px + 12, y0 + 74, f"g{sub('φ')}", "m", 17, raw=True, extra=' font-style="italic"'))
    o.append(text(px + 12, y0 + 98, "MLP with BatchNorm", "t2", 12))
    o.append(text(px + 12, y0 + 116, f"{E} → 256 → 256 → {K}", "t2", 12))
    o.append(text(px + 12, y0 + 140, "used only for the loss", "t3", 11))
    # z
    zx, zw = 824, 156
    o.append(f'<rect class="box p" x="{zx}" y="{y0 + 30}" width="{zw}" height="124" rx="6" stroke-dasharray="4 3"/>')
    o.append(text(zx + 12, y0 + 52, "Embeddings", "cp tag"))
    o.append(text(zx + 12, y0 + 76, f"{it('z')}{sub(it('v,n'))} ∈ ℝ{sup(str(K))}", "m", 17, raw=True))
    o.append(text(zx + 12, y0 + 98, f"{ng + nl} views × {it('N')} images", "t2", 12, raw=True))
    o.append(text(zx + 12, y0 + 116, f"({it('N')} = {cfg['train']['batch_size']} per batch)", "t2", 12, raw=True))
    o.append(text(zx + 12, y0 + 140, "predictor = identity", "t3", 11))

    ym = y0 + 92
    wires = [  # (class, marker, path, dur)
        ("v", "v", f"M134,{ym} L{vx - 4},{ym}", 1.2),
        ("v", "v", f"M{vx + vw},{ym} L{ex - 4},{ym}", 1.2),
        ("e", "e", f"M{ex + ew},{ym} L{px - 4},{ym}", 1.2),
        ("p", "p", f"M{px + pw},{ym} L{zx - 4},{ym}", 1.2),
    ]
    # ---- row 2
    r2 = 300
    tx, tw = 20, 330
    lpx, lpw = 380, 300
    lsx, lsw = 700, 280
    zc = zx + zw / 2
    wires += [
        ("lp", "lp", f"M{zc - 40},{y0 + 154} L{zc - 40},{r2 - 30} L{lpx + lpw / 2},{r2 - 30} L{lpx + lpw / 2},{r2 - 4}", 2.0),
        ("ls", "ls", f"M{zc + 20},{y0 + 154} L{zc + 20},{r2 - 4}", 1.2),
        ("lp", "lp", f"M{lpx},{r2 + 70} L{tx + tw + 4},{r2 + 70}", 1.0),
        ("ls", "ls", f"M{lsx + lsw / 2},{r2 + 196} L{lsx + lsw / 2},{r2 + 214} L{tx + tw / 2},{r2 + 214} "
                     f"L{tx + tw / 2},{r2 + 168}", 2.4),
    ]
    # L_pred box (blue)
    o.append(f'<rect class="box lp" x="{lpx}" y="{r2}" width="{lpw}" height="150" rx="6"/>')
    o.append(text(lpx + 12, r2 + 22, "Prediction term", "clp tag"))
    o.append(text(lpx + 12, r2 + 50, f"{it('μ')}{sub(it('n'))} = mean of the {ng} global-view {it('z')}{sub(it('g,n'))}",
                  "m", 15, raw=True))
    o.append(text(lpx + 12, r2 + 82, f"{it('L')}{sub('pred')} = mean{sub(it('v,n,k'))} ({it('μ')}{sub(it('n,k'))} − {it('z')}{sub(it('v,n,k'))}){sup('2')}",
                  "m", 16, raw=True))
    o.append(text(lpx + 12, r2 + 110, "every view, global and local, predicts", "t2", 12))
    o.append(text(lpx + 12, r2 + 127, "the centre of the global views", "t2", 12))
    # SIGReg box (magenta-red) with a histogram glyph
    o.append(f'<rect class="box ls" x="{lsx}" y="{r2}" width="{lsw}" height="196" rx="6"/>')
    o.append(text(lsx + 12, r2 + 22, "SIGReg", "cls tag") + text(lsx + 64, r2 + 22, "(anti-collapse)", "t3", 11))
    o.append(text(lsx + 12, r2 + 44, f"{M} random unit directions {it('a')}{sub(it('m'))}", "t2", 12, raw=True))
    o.append(text(lsx + 12, r2 + 61, "(redrawn every step)", "t3", 11))
    gx, gy, gw, gh = lsx + 176, r2 + 30, 86, 44
    xs = np.linspace(-3, 3, 13)
    heights = np.exp(-xs ** 2 / 2)
    bw = gw / len(xs)
    for i, hv in enumerate(heights):
        bh = hv * gh * 0.9
        o.append(f'<rect class="cls" opacity="0.55" x="{gx + i * bw + 0.5:.1f}" y="{gy + gh - bh:.1f}" width="{bw - 1:.1f}" height="{bh:.1f}"/>')
    curve = " ".join(f"{gx + (u + 3.25) / 6.5 * gw:.1f},{gy + gh - math.exp(-u * u / 2) * gh * 0.95:.1f}"
                     for u in np.linspace(-3.25, 3.25, 40))
    o.append(f'<polyline points="{curve}" fill="none" stroke="var(--ink)" stroke-width="1.3" stroke-dasharray="3 2"/>')
    o.append(text(gx + gw / 2, gy + gh + 13, "a projection vs N(0, 1)", "t3", 10, "middle"))
    o.append(text(lsx + 12, r2 + 112, f"EP = {it('N')} ∫ |{it('φ̂')}({it('t')}) − {it('e')}{sup('−t²/2')}|{sup('2')} {it('e')}{sup('−t²/2')} {it('dt')}",
                  "m", 14, raw=True))
    o.append(text(lsx + 12, r2 + 130, "Epps–Pulley test of each projection", "t3", 11))
    o.append(text(lsx + 12, r2 + 162, f"{it('L')}{sub('SIGReg')} = mean{sub(it('v'))} mean{sub(it('m'))} EP({it('z')}{sub(it('v'))}{it('a')}{sub(it('m'))})",
                  "m", 15, raw=True))
    o.append(text(lsx + 12, r2 + 184, "pushes each view's batch toward N(0, I)", "t2", 12))
    # total (ink) with coloured underlines
    o.append(f'<rect class="box tot" x="{tx}" y="{r2 + 20}" width="{tw}" height="148" rx="6"/>')
    o.append(text(tx + 12, r2 + 42, "Total loss", "t tag"))
    o.append(text(tx + 12, r2 + 80, f"{it('L')} = (1 − {it('λ')}) {it('L')}{sub('pred')} + {it('λ')} {it('L')}{sub('SIGReg')}",
                  "m", 19, raw=True))
    o.append(f'<line x1="{tx + 90}" x2="{tx + 142}" y1="{r2 + 97}" y2="{r2 + 97}" stroke="var(--pred)" stroke-width="3"/>')
    o.append(f'<line x1="{tx + 184}" x2="{tx + 260}" y1="{r2 + 97}" y2="{r2 + 97}" stroke="var(--sigreg)" stroke-width="3"/>')
    o.append(text(tx + 12, r2 + 120, f"{it('λ')} = {lam} in runs/shapes-desktop", "t2", 12, raw=True))
    o.append(text(tx + 12, r2 + 138, "no decoder, no EMA teacher, no stop-gradient", "t3", 11))
    o.append(text(tx + 12, r2 + 155, "one hyperparameter trades the two terms", "t3", 11))
    # backprop
    o.append(f'<path class="back" d="M{tx + 60},{r2 + 20} L{tx + 60},{y1 + 22} L{ex + ew / 2},{y1 + 22} L{ex + ew / 2},{y0 + 158}" marker-end="url(#ak)"/>')
    o.append(text(tx + 72, y1 + 16, f"backprop updates {it('θ')} and {it('φ')}", "t3", 11, raw=True))

    for i, (c, mk, d, dur) in enumerate(wires):
        o.append(f'<path class="wire {c}" d="{d}" marker-end="url(#a{mk})"/>')
    for i, (c, mk, d, dur) in enumerate(wires):
        for k in range(3):
            o.append(f'<circle class="dot f{c}" r="3.6"><animateMotion dur="{dur}s" repeatCount="indefinite" '
                     f'begin="-{dur * k / 3:.2f}s" path="{d}"/></circle>')
    o.append(text(W - 20, H - 12, "Colours: teal views · amber encoder · violet projector/embeddings · blue L_pred · magenta-red SIGReg",
                  "t3", 11, "end"))
    o.append("</svg>")
    out = ASSETS / "architecture.svg"
    out.write_text("\n".join(o))
    return out


# ----------------------------------------------------------------------------- SVG check (Playwright)

def render_svg(svg: Path, out_prefix: Path, scale: float = 1.0) -> list[Path]:
    """Screenshot an SVG as GitHub shows it (<img>) in light and dark colour schemes."""
    import os
    os.environ.setdefault("PLAYWRIGHT_BROWSERS_PATH", "/opt/pw-browsers")
    from playwright.sync_api import sync_playwright
    outs = []
    with sync_playwright() as p:
        b = p.chromium.launch()
        for scheme, bg in (("light", "#ffffff"), ("dark", "#0d1117")):
            pg = b.new_page(color_scheme=scheme, device_scale_factor=scale)
            pg.set_content(f'<html><body style="margin:16px;background:{bg}">'
                           f'<img src="data:image/svg+xml;base64,{base64.b64encode(svg.read_bytes()).decode()}"></body></html>')
            pg.wait_for_timeout(700)
            o = out_prefix.with_name(f"{out_prefix.name}-{scheme}.png")
            pg.screenshot(path=str(o), full_page=True)
            outs.append(o)
        b.close()
    return outs


# ----------------------------------------------------------------------------- 3. jepa-vs-pixels.svg

def make_jepa_vs_pixels() -> Path:
    cfg = load_cfg()
    K = cfg["model"]["proj_dim"]
    S = cfg["data"]["image_size"]
    npx = S * S * cfg["data"]["channels"]
    im, gv, lv = example_views(cfg)
    masked = im.copy()
    rng = np.random.default_rng(3)
    blocks = rng.choice(16, size=10, replace=False)  # mask 10 of 16 8x8 patches (illustration)
    for b in blocks:
        r, c = divmod(int(b), 4)
        masked[r * 8:(r + 1) * 8, c * 8:(c + 1) * 8] = 0.55
    W, H = 1000, 468
    css = (".box{fill:var(--surface);stroke-width:2}"
           ".v{stroke:var(--view)} .e{stroke:var(--encoder)} .p{stroke:var(--predictor)} .lp{stroke:var(--pred)}"
           ".n{stroke:var(--ink-3)} .ghost{fill:none;stroke:var(--ink-3);stroke-width:1.5;stroke-dasharray:5 4}"
           ".tag{font-size:12px;font-weight:600} .cv{fill:var(--view)} .ce{fill:var(--encoder)}"
           ".cp{fill:var(--predictor)} .clp{fill:var(--pred)} .cls{fill:var(--sigreg)}"
           ".wire{fill:none;stroke-width:2} .thumb{stroke:var(--line-strong);stroke-width:1;fill:none}"
           ".panel{fill:var(--surface-2);stroke:none}")
    o = [svg_open(W, H, "Pixel reconstruction vs JEPA",
                  f"Top: a reconstruction model encodes a masked image and a decoder must output all {npx} pixel "
                  f"values; the loss is computed in pixel space. Bottom: a JEPA encodes views and compares "
                  f"{K}-number embeddings; the loss is computed in representation space and there is no decoder.", css)]
    o.append('<defs>' + "".join(
        f'<marker id="m{c}" viewBox="0 0 10 10" refX="9" refY="5" markerWidth="7" markerHeight="7" orient="auto">'
        f'<path d="M0,0 L10,5 L0,10 z" fill="var(--{v})"/></marker>'
        for c, v in [("v", "view"), ("e", "encoder"), ("p", "predictor"), ("n", "ink-3")]) + '</defs>')
    o.append(text(20, 30, "Where the loss lives", size=16, weight=600))

    def thumb(x, y, arr, size, scale):
        return (f'<image x="{x}" y="{y}" width="{size}" height="{size}" href="{png_data_uri(arr, scale)}" '
                f'style="image-rendering:pixelated"/><rect class="thumb" x="{x}" y="{y}" width="{size}" height="{size}"/>')

    def arrow(x1, x2, y, c):
        return f'<path class="wire" stroke="var(--{ {"v": "view", "e": "encoder", "p": "predictor", "n": "ink-3"}[c]})" d="M{x1},{y} L{x2 - 3},{y}" marker-end="url(#m{c})"/>'

    for row, y in (("pix", 48), ("jepa", 250)):
        o.append(f'<rect class="panel" x="12" y="{y}" width="{W - 24}" height="186" rx="6"/>')
        yc = y + 110
        if row == "pix":
            o.append(text(28, y + 26, "Pixel reconstruction", "t tag", 14)
                     + text(176, y + 26, "(e.g. masked autoencoders (MAE); not part of this repo, shown for contrast)", "t3", 11))
            o.append(thumb(28, yc - 40, im, 80, 2) + text(68, yc + 58, "image", "t3", 11, "middle"))
            o.append(arrow(112, 136, yc, "n"))
            o.append(thumb(138, yc - 40, masked, 80, 2) + text(178, yc + 58, "masked input", "t3", 11, "middle"))
            o.append(arrow(222, 250, yc, "n"))
            o.append(f'<rect class="box e" x="252" y="{yc - 44}" width="140" height="88" rx="6"/>')
            o.append(text(264, yc - 22, "Encoder", "ce tag") + text(264, yc, "features of the", "t2", 12)
                     + text(264, yc + 17, "visible patches", "t2", 12))
            o.append(arrow(392, 424, yc, "e"))
            o.append(f'<rect class="box n" x="426" y="{yc - 44}" width="150" height="88" rx="6"/>')
            o.append(text(438, yc - 22, "Decoder", "t tag") + text(438, yc, "must paint back", "t2", 12)
                     + text(438, yc + 17, "every pixel", "t2", 12))
            o.append(arrow(576, 606, yc, "n"))
            o.append(f'<rect class="box n" x="608" y="{yc - 44}" width="132" height="88" rx="6"/>')
            o.append(text(620, yc - 20, f"{it('x̂')} ∈ ℝ{sup(f'{npx:,}')}", "m", 17, raw=True)
                     + text(620, yc + 2, f"{S}×{S}×3 pixel", "t2", 12) + text(620, yc + 19, "values", "t2", 12))
            o.append(arrow(740, 770, yc, "n"))
            o.append(f'<rect class="box n" x="772" y="{yc - 60}" width="200" height="120" rx="6"/>')
            o.append(text(784, yc - 38, "Loss in pixel space", "t tag"))
            o.append(text(784, yc - 10, f"‖{it('x̂')} − {it('x')}‖{sup('2')} over {npx:,}", "m", 16, raw=True))
            o.append(text(784, yc + 9, "numbers per image", "t2", 12))
            o.append(text(784, yc + 30, "noise, texture and exact", "t3", 11))
            o.append(text(784, yc + 45, "colour all cost loss", "t3", 11))
        else:
            o.append(text(28, y + 26, "Joint-Embedding Predictive Architecture (JEPA)", "t tag", 14)
                     + text(332, y + 26, "(this repo: LeJEPA)", "t3", 11))
            o.append(thumb(28, yc - 40, im, 80, 2) + text(68, yc + 58, "image", "t3", 11, "middle"))
            o.append(arrow(112, 136, yc, "v"))
            o.append(f'<rect class="box v" x="138" y="{yc - 50}" width="92" height="100" rx="6"/>')
            o.append(thumb(146, yc - 42, gv[0], 36, 1) + thumb(186, yc - 42, gv[1], 36, 1))
            o.append(thumb(146, yc + 2, lv[1], 36, 2) + thumb(186, yc + 2, lv[3], 36, 2))
            o.append(text(184, yc + 66, "views", "cv", 11, "middle"))
            o.append(arrow(230, 250, yc, "v"))
            o.append(f'<rect class="box e" x="252" y="{yc - 44}" width="140" height="88" rx="6"/>')
            o.append(text(264, yc - 22, "Encoder", "ce tag") + text(264, yc, "+ projector,", "t2", 12)
                     + text(264, yc + 17, "shared by all views", "t2", 12))
            o.append(arrow(392, 424, yc, "p"))
            o.append(f'<rect class="ghost" x="426" y="{yc - 44}" width="150" height="88" rx="6"/>')
            o.append(text(501, yc - 4, "no decoder", "t3", 13, "middle") + text(501, yc + 14, "nothing is drawn back", "t3", 11, "middle"))
            o.append(arrow(576, 606, yc, "p"))
            o.append(f'<rect class="box p" x="608" y="{yc - 44}" width="132" height="88" rx="6"/>')
            o.append(text(620, yc - 20, f"{it('z')} ∈ ℝ{sup(str(K))}", "m", 17, raw=True)
                     + text(620, yc + 2, "per view: an", "t2", 12) + text(620, yc + 19, "embedding", "t2", 12))
            o.append(f'<path class="wire" stroke="var(--pred)" d="M740,{yc} L767,{yc}" marker-end="url(#mp)"/>')
            o.append(f'<rect class="box lp" x="772" y="{yc - 60}" width="200" height="120" rx="6"/>')
            o.append(text(784, yc - 38, "Loss in representation space", "clp tag"))
            o.append(text(784, yc - 10, f"({it('μ')} − {it('z')}){sup('2')} over {K} numbers", "m", 16, raw=True))
            o.append(text(784, yc + 9, "per view (predict the centre)", "t2", 12))
            o.append(text(784, yc + 30, "+ SIGReg keeps z spread out,", "cls", 11))
            o.append(text(784, yc + 45, "so it cannot collapse", "cls", 11))
    o.append(text(W - 20, H - 10, f"{npx:,} = {S}×{S}×3 values of one Shapes image; {K} = projector output size in runs/shapes-desktop",
                  "t3", 11, "end"))
    o.append("</svg>")
    out = ASSETS / "jepa-vs-pixels.svg"
    out.write_text("\n".join(o))
    return out


# ----------------------------------------------------------------------------- 4. throughput.svg

BENCH_JSON = ASSETS / "throughput-benchmark.json"


def measure_throughput(repeats: int = 3, steps: int = 12) -> Path:
    """Re-run the real auto-tuning benchmark (slow-ish: ~1 CPU-minute per repeat)."""
    import os
    import tempfile
    import time
    from jepa_studio import hardware as hw
    from jepa_studio.train import benchmark_optimizations
    cfg = load_cfg()
    reps = []
    cwd = os.getcwd()
    with tempfile.TemporaryDirectory() as tmp:
        os.chdir(tmp)  # benchmark_optimizations writes (and removes) ./runs/_bench
        try:
            for _ in range(repeats):
                la = os.getloadavg()[0]
                t = time.time()
                reps.append({"measured_unix": round(t), "load_avg_1min_before": round(la, 2),
                             "stages": benchmark_optimizations(cfg, steps=steps, log=lambda *_: None)})
            card = hw.probe(tmp, measure_disk=False)
        finally:
            os.chdir(cwd)
    out = {"about": "Real measurements for assets/throughput.svg. Produced by scripts/make_visuals.py --measure, "
                    "which runs jepa_studio.train.benchmark_optimizations on runs/shapes-desktop/config.json "
                    f"({steps} timed steps per stage, 2 warm-up steps) {repeats} times in a row.",
           "config": "runs/shapes-desktop/config.json", "steps_per_stage": steps,
           "card": {k: getattr(card, k) for k in ("cpu_model", "cpu_cores_physical", "cpu_cores_logical",
                                                  "ram_total_gb", "accelerator", "torch", "os")},
           "repeats": reps}
    BENCH_JSON.write_text(json.dumps(out, indent=2) + "\n")
    return BENCH_JSON


def make_throughput() -> Path:
    b = json.loads(BENCH_JSON.read_text())
    web = json.loads((ASSETS / "screens" / "headless-benchmark.json").read_text())
    stages = [s["stage"] for s in b["repeats"][0]["stages"]]
    vals = np.array([[s["samples_per_s"] for s in r["stages"]] for r in b["repeats"]])  # (R, S)
    med = np.median(vals, 0)
    card = b["card"]
    cpu = card["cpu_model"].replace("(R)", "").replace("  ", " ")
    pretty = {  # plain-language stage names; order = the order the tuner applies them
        "baseline": "Baseline: fp32, batch 32, no loader workers",
        "+ auto batch size": "+ batch size {bs}",
        "+ workers": "+ 1 data-loader worker process",
        "+ precision": "+ precision {v}", "+ channels_last": "+ channels-last memory format",
        "+ pin_memory": "+ pinned host memory",
    }

    def label(st, s):
        for k, v in pretty.items():
            if st.startswith(k):
                return v.format(bs=s["batch"], v=st.split("=")[-1].strip())
        return st

    labels = [label(st, s) for st, s in zip(stages, b["repeats"][0]["stages"])]
    W = 780
    L, R = 262, 700            # plot x-range
    top = 104
    bh, gap = 28, 14
    xmax = 250.0
    X = lambda v: L + (R - L) * v / xmax  # noqa: E731
    ny = len(stages)
    yb = top + ny * (bh + gap)
    top2 = yb + 128
    ms = [("JavaScript on the CPU", web["gpu"]["speed"]["cpuMs"], "kept"),
          (f"WebGPU ({web['gpu']['adapter']['architecture']} software adapter)", web["gpu"]["speed"]["gpuMs"], "rejected")]
    ms.sort(key=lambda r: r[1])
    xmax2 = 600.0
    X2 = lambda v: L + (R - L) * v / xmax2  # noqa: E731
    yb2 = top2 + len(ms) * (bh + gap)
    H = yb2 + 80
    css = (".bar{fill:var(--ink-2)} .bar2{fill:var(--ink-3)} .grid{stroke:var(--line);stroke-width:1}"
           ".axis{stroke:var(--line-strong);stroke-width:1} .rep{fill:var(--surface);stroke:var(--ink);stroke-width:1.5}"
           ".num{font-variant-numeric:tabular-nums}")
    o = [svg_open(W, H, "Training throughput by auto-tuning stage",
                  "Desktop: samples per second of a LeJEPA training step as each automatic setting is switched on "
                  f"({', '.join(f'{l}: {m:.1f}' for l, m in zip(labels, med))}; median of {len(vals)} repeats). "
                  f"Web: first-layer time, JavaScript {ms[0][1]:.1f} ms vs WebGPU {ms[1][1]:.1f} ms.", css)]
    o.append(text(20, 30, "Auto-tuning on a 2-core CPU: no measurable gain", size=16, weight=600))
    o.append(text(20, 50, f"Machine (hardware card): {cpu}, {card['cpu_cores_physical']} cores, "
                          f"{card['ram_total_gb']:.1f} GB RAM, no GPU, PyTorch {card['torch'].split('+')[0]}, Linux", "t2", 12))
    o.append(text(20, 80, "Desktop: samples/s of one LeJEPA training step (ConvNet-tiny, 6 views), as each setting is switched on",
                  "t", 12, weight=600))
    for v in range(0, int(xmax) + 1, 50):
        o.append(f'<line class="grid" x1="{X(v):.1f}" x2="{X(v):.1f}" y1="{top - 6}" y2="{yb - gap + 6}"/>')
        o.append(text(X(v), yb + 8, str(v), "t3 num", 11, "middle"))
    o.append(text(R, yb + 24, "samples per second (higher is better)", "t3", 11, "end"))
    for i, (lab, m) in enumerate(zip(labels, med)):
        y = top + i * (bh + gap)
        o.append(text(L - 12, y + bh / 2 + 4, lab, "t", 12, "end"))
        o.append(f'<path class="bar" d="M{L},{y} H{X(m) - 3:.1f} q3,0 3,3 V{y + bh - 3} q0,3 -3,3 H{L} Z"/>')
        for v in vals[:, i]:
            o.append(f'<circle class="rep" cx="{X(v):.1f}" cy="{y + bh / 2}" r="3.5"/>')
        o.append(text(X(vals[:, i].max()) + 10, y + bh / 2 + 4, f"{m:.0f}", "t num", 12, weight=600))
    o.append(f'<line class="axis" x1="{L}" x2="{L}" y1="{top - 6}" y2="{yb - gap + 6}"/>')
    lo, hi = vals.min(), vals.max()
    o.append(text(20, yb + 44, f"Bars: median of {len(vals)} runs of {b['steps_per_stage']} timed steps; rings: each run. "
                               f"Every stage lands in the same {lo:.0f}–{hi:.0f} samples/s band: on this machine the", "t2", 12))
    o.append(text(20, yb + 61, "tuner's choices (batch 128, 1 loader worker, fp32 kept) are speed-neutral within "
                               "run-to-run noise.", "t2", 12))
    # web panel
    o.append(text(20, top2 - 24, "Web demo: time for the first layer at start-up (headless Chromium), lower is better",
                  "t", 12, weight=600))
    for v in range(0, int(xmax2) + 1, 100):
        o.append(f'<line class="grid" x1="{X2(v):.1f}" x2="{X2(v):.1f}" y1="{top2 - 6}" y2="{yb2 - gap + 6}"/>')
        o.append(text(X2(v), yb2 + 8, str(v), "t3 num", 11, "middle"))
    o.append(text(R, yb2 + 24, "milliseconds", "t3", 11, "end"))
    for i, (lab, v, verdict) in enumerate(ms):
        y = top2 + i * (bh + gap)
        o.append(text(L - 12, y + bh / 2 + 4, lab, "t", 12, "end"))
        o.append(f'<path class="bar2" d="M{L},{y} H{X2(v) - 3:.1f} q3,0 3,3 V{y + bh - 3} q0,3 -3,3 H{L} Z"/>')
        lab2 = f"{v:.1f} ms · {verdict}"
        if X2(v) + 12 + 7 * len(lab2) > W - 10:
            o.append(text(X2(v) - 10, y + bh / 2 + 4, lab2, "num", 12, "end", weight=600, extra=' fill="var(--surface)"'))
        else:
            o.append(text(X2(v) + 10, y + bh / 2 + 4, lab2, "t num", 12, weight=600))
    o.append(f'<line class="axis" x1="{L}" x2="{L}" y1="{top2 - 6}" y2="{yb2 - gap + 6}"/>')
    ratio = ms[1][1] / ms[0][1]
    o.append(text(20, yb2 + 44, f"The app times both paths and keeps the faster one: the software WebGPU adapter was {ratio:.1f}× slower, "
                                f"so the demo trained with", "t2", 12))
    o.append(text(20, yb2 + 61, f"JavaScript at {web['mean_samples_per_s']:.1f} samples/s. Data: assets/throughput-benchmark.json, "
                                "assets/screens/headless-benchmark.json.", "t2", 12))
    o.append("</svg>")
    out = ASSETS / "throughput.svg"
    out.write_text("\n".join(o))
    return out


# ----------------------------------------------------------------------------- 5. web-vs-desktop.svg

# Every row was checked against the code on 2026-09-30:
#   web      docs/web-demo.md, site/js/engine/gpu.js (navigator.gpu, correctness + timing check),
#            site/js/engine/trainer.js (mlp-tiny, probeBatch, baselines), site/js/tabs/*.js,
#            site/js/world/*.js + site/models/world-*.json. No .wasm module exists: the config enum and
#            the Train tab accept "wasm", but the CPU path is plain JavaScript -> WebAssembly is "planned".
#   desktop  jepa_studio/hardware.py (cuda / rocm via torch.version.hip / mps / cpu), train.py
#            (checkpoints, resume, benchmark_optimizations), models.py (video-convnet, series-conv, ViT),
#            server.py routes, export.py (bundle + manifest), cli.py (sign-weights, verify-weights).
CAPABILITIES = [
    ("Compute backend",
     ("part", "WebGPU for the first layer when it is", "correct and faster; else JavaScript (CPU)"),
     ("yes", "PyTorch on NVIDIA CUDA, AMD ROCm,", "Apple Metal Performance Shaders (MPS) or CPU")),
    ("WebAssembly (WASM) kernels",
     ("plan", "the config accepts \"wasm\", but the CPU", "path is plain JavaScript today"),
     ("na", "not needed: native PyTorch", "")),
    ("Pretraining",
     ("yes", "tiny: mlp-tiny on 16×16 inputs,", "a few minutes in the tab"),
     ("yes", "full: ConvNet-tiny/small, ViT-tiny,", "checkpoints, pause and resume")),
    ("Your own data",
     ("part", "up to 2,000 images, read in the page", "(never uploaded)"),
     ("yes", "image folders of any size, granted", "one at a time in a native folder dialog")),
    ("Video and time series",
     ("part", "CSV preview and views only", ""),
     ("yes", "video-convnet and series-conv encoders", "")),
    ("Inference (embeddings)",
     ("part", "small encoders only: train or load", "mlp-tiny .safetensors in the browser"),
     ("yes", "any trained encoder", "")),
    ("Embedding explorer",
     ("yes", "PCA 2-D/3-D, neighbours, isotropy and", "collapse check, saliency"),
     ("yes", "same front end, served by the", "local PyTorch backend")),
    ("Evaluation",
     ("yes", "linear probe, k-NN, random-init and", "supervised baselines (small model)"),
     ("yes", "same protocol on the full model", "")),
    ("Planning bot (world model)",
     ("yes", "pretrained two-room and push-block models;", "cross-entropy method (CEM) planning"),
     ("yes", "train your own world models, then plan", "")),
    ("Auto-tuning",
     ("part", "batch-size timing probe;", "WebGPU vs JavaScript timing"),
     ("yes", "device, precision, channels-last, workers,", "threads, batch size; jepa-studio bench")),
    ("Export",
     ("yes", "embeddings (CSV, .npy), .safetensors,", "run report, zip"),
     ("yes", "plus reproducibility bundle with SHA-256", "manifest (Ed25519 signing via sign-weights)")),
]


def make_web_vs_desktop() -> Path:
    W = 1000
    c0, c1, c2 = 20, 250, 620        # column x
    row_h, head = 50, 96
    H = head + row_h * len(CAPABILITIES) + 56
    css = (".rule{stroke:var(--line);stroke-width:1} .hdr{font-weight:600}"
           ".yes{stroke:var(--good);fill:none} .part{stroke:var(--warn);fill:none} .plan{stroke:var(--ink-3);fill:none}"
           ".na{stroke:var(--ink-3);fill:none} .wyes{fill:var(--good)} .wpart{fill:var(--warn)}"
           ".wplan{fill:var(--ink-3)} .wna{fill:var(--ink-3)} .band{fill:var(--surface-2)}")
    o = [svg_open(W, H, "Web demo vs desktop app",
                  "Capability table: " + "; ".join(f"{r[0]}: web {r[1][0]}, desktop {r[2][0]}" for r in CAPABILITIES), css)]
    o.append(text(20, 30, "What runs where", size=16, weight=600))
    o.append(text(20, 50, "Checked against site/js, docs/web-demo.md and jepa_studio/ (2026-09-30). Every mark has a word; "
                          "“planned” means not implemented yet.", "t2", 12))
    o.append(text(c1, head - 14, "Web demo (in the browser tab)", "t hdr", 13))
    o.append(text(c2, head - 14, "Desktop app and CLI (PyTorch)", "t hdr", 13))
    o.append(f'<line class="rule" x1="12" x2="{W - 12}" y1="{head - 4}" y2="{head - 4}"/>')
    icon = {"yes": '<path class="yes" stroke-width="2.4" stroke-linecap="round" stroke-linejoin="round" d="M{x},{y} l4,4 l8,-9"/>',
            "part": '<path class="part" stroke-width="2.4" stroke-linecap="round" d="M{x},{y} h12"/>',
            "plan": '<circle class="plan" stroke-width="1.8" stroke-dasharray="2.5 2" cx="{cx}" cy="{y}" r="6"/>',
            "na": '<path class="na" stroke-width="1.8" stroke-linecap="round" d="M{x},{y} h8"/>'}
    word = {"yes": "Yes", "part": "Partly", "plan": "Planned", "na": "n/a"}
    for i, (name, web, desk) in enumerate(CAPABILITIES):
        y = head + i * row_h
        if i % 2 == 0:
            o.append(f'<rect class="band" x="12" y="{y}" width="{W - 24}" height="{row_h}"/>')
        o.append(text(c0 + 8, y + 29, name, "t hdr", 13))
        for cx, (kind, l1, l2) in ((c1, web), (c2, desk)):
            yy = y + 20
            o.append(icon[kind].format(x=cx, y=yy, cx=cx + 6))
            o.append(text(cx + 20, yy + 4, word[kind], f"w{kind}", 12, weight=700))
            o.append(text(cx + 78, yy + 4, l1, "t2", 12))
            if l2:
                o.append(text(cx + 78, yy + 21, l2, "t2", 12))
    yb = head + row_h * len(CAPABILITIES)
    o.append(f'<line class="rule" x1="12" x2="{W - 12}" y1="{yb}" y2="{yb}"/>')
    o.append(text(20, yb + 24, "GPU paths are chosen by jepa_studio/hardware.py; the runs shipped in runs/ were trained on a 2-core CPU. "
                               "The web tier never uploads your files", "t3", 11))
    o.append(text(20, yb + 40, "(Content Security Policy allows only its own origin); the desktop backend binds to 127.0.0.1 only.",
                  "t3", 11))
    o.append("</svg>")
    out = ASSETS / "web-vs-desktop.svg"
    out.write_text("\n".join(o))
    return out


# ----------------------------------------------------------------------------- GIF helpers

def fig_to_rgb(fig) -> np.ndarray:
    fig.canvas.draw()
    a = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    return a


def save_gif(frames: list[np.ndarray], durations_ms: list[int], path: Path, colors: int = 128, lossy: int = 0) -> Path:
    """Shared adaptive palette (built from a sample of frames), consecutive-duplicate merge, gifsicle -O3."""
    from PIL import Image
    merged, durs = [], []
    for f, d in zip(frames, durations_ms):
        if merged and np.array_equal(merged[-1], f):
            durs[-1] += d
        else:
            merged.append(f)
            durs.append(d)
    step = max(1, len(merged) // 12)
    sample = np.concatenate([m for m in merged[::step]], axis=0)
    pal = Image.fromarray(sample).quantize(colors=colors, method=Image.Quantize.MEDIANCUT, dither=Image.Dither.NONE)
    ims = [Image.fromarray(m).quantize(palette=pal, dither=Image.Dither.NONE) for m in merged]
    ims[0].save(path, save_all=True, append_images=ims[1:], duration=durs, loop=0, optimize=True, disposal=1)
    optimise_gif(path, lossy)
    return path


def mpl_setup():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.edgecolor": LIGHT["line_strong"],
                         "axes.labelcolor": LIGHT["ink3"], "xtick.color": LIGHT["ink3"], "ytick.color": LIGHT["ink3"],
                         "axes.linewidth": 0.8, "text.color": LIGHT["ink"], "figure.facecolor": LIGHT["surface"],
                         "axes.facecolor": LIGHT["surface"], "savefig.facecolor": LIGHT["surface"]})
    return plt


# ----------------------------------------------------------------------------- 2. sigreg.gif

def make_sigreg() -> Path:
    import torch
    torch.set_num_threads(1)
    from jepa_studio.losses import ep_expected_under_null
    plt = mpl_setup()
    cfg = load_cfg()
    ev = load_events()
    steps = [e for e in ev if e.get("event") == "step"]
    total = next(e["total_steps"] for e in ev if e.get("event") == "start")
    recs = [e for e in steps if "proj_hist" in e]
    # extra first panel, computed for real: step-0 weights on the same fixed batch and directions
    h0, _ = probe_hist(build_model(cfg, None), fixed_probe_batch(cfg))
    snaps = [dict(step=0, hist=h0, sigreg=None)] + [dict(step=e["step"], hist=e["proj_hist"], sigreg=e["sigreg"]) for e in recs]
    floor = ep_expected_under_null(cfg["objective"]["t_max"], cfg["objective"]["knots"])
    edges = np.linspace(-4, 4, 25)
    ctr = (edges[:-1] + edges[1:]) / 2
    xs = np.linspace(-4, 4, 200)
    gauss = np.exp(-xs ** 2 / 2) / math.sqrt(2 * math.pi)
    ymax = 0.62
    sx = np.array([e["step"] for e in steps])
    sy = np.array([e["sigreg"] for e in steps])
    frames, durs = [], []
    for k, sn in enumerate(snaps):
        fig = plt.figure(figsize=(7.2, 4.3), dpi=100)
        fig.text(0.03, 0.935, "SIGReg pulls random 1-D projections of z toward N(0, 1)", fontsize=13, weight="bold",
                 color=LIGHT["ink"])
        fig.text(0.03, 0.885, "runs/shapes-desktop · 3 fixed random directions · 256 fixed images · histograms "
                              "logged every 50 steps", fontsize=8.5, color=LIGHT["ink3"])
        if sn["step"] == 0:
            head = "Before training: random init (computed for this figure, not in the log)"
        else:
            head = f"Step {sn['step']} of {total}" + (" (first logged step)" if sn["step"] == 1 else "")
        fig.text(0.03, 0.805, head, fontsize=11, color=LIGHT["ink"], weight="bold")
        for j in range(3):
            ax = fig.add_axes([0.06 + j * 0.315, 0.35, 0.28, 0.36])
            h = np.array(sn["hist"][j])
            ax.bar(ctr, np.minimum(h, ymax), width=(edges[1] - edges[0]) * 0.9, color=LIGHT["sigreg"], alpha=0.85,
                   linewidth=0)
            for c, v in zip(ctr, h):
                if v > ymax:
                    ax.text(c + 0.28, ymax * 0.965, f"{v:.2f}↑", ha="left", va="top", fontsize=8, color=LIGHT["ink"],
                            weight="bold")
            ax.plot(xs, gauss, "--", color=LIGHT["ink"], lw=1.3)
            if j == 0:
                ax.text(1.35, 0.33, "N(0, 1)", fontsize=8.5, color=LIGHT["ink"])
            ax.set_xlim(-4, 4)
            ax.set_ylim(0, ymax)
            ax.set_xticks([-4, -2, 0, 2, 4])
            ax.set_yticks([0, 0.2, 0.4, 0.6] if j == 0 else [])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(axis="y", color=LIGHT["line"], lw=0.8)
            ax.set_axisbelow(True)
            ax.set_title(f"direction a{j + 1}", fontsize=9, color=LIGHT["ink2"], loc="left")
            ax.tick_params(labelsize=8)
        fig.text(0.06, 0.262, "density of the projected values (axis tops out at 0.62; a taller bar has its height printed next to it)",
                 fontsize=7.5, color=LIGHT["ink3"])
        # SIGReg trace
        ax = fig.add_axes([0.06, 0.07, 0.6, 0.16])
        ax.plot(sx, sy, color=LIGHT["sigreg"], lw=1.6, alpha=0.35)
        m = sx <= max(sn["step"], 0)
        ax.plot(sx[m], sy[m], color=LIGHT["sigreg"], lw=2)
        ax.axhline(floor, ls="--", color=LIGHT["ink2"], lw=1)
        if sn["sigreg"] is not None:
            ax.plot([sn["step"]], [sn["sigreg"]], "o", color=LIGHT["sigreg"], ms=6, mec=LIGHT["surface"], mew=1.5)
        ax.set_yscale("log")
        ax.set_xlim(0, total)
        ax.set_ylim(0.8, 60)
        ax.set_yticks([1, 10])
        ax.set_yticklabels(["1", "10"])
        ax.tick_params(labelsize=7.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        ax.set_xlabel("training step", fontsize=7.5, labelpad=1)
        val = "not logged (untrained)" if sn["sigreg"] is None else f"{sn['sigreg']:.2f}"
        fig.text(0.70, 0.19, "SIGReg on the training batch", fontsize=8.5, color=LIGHT["ink2"])
        fig.text(0.70, 0.145, "(256 slices, 6 views, log scale; dashed:", fontsize=7.5, color=LIGHT["ink3"])
        fig.text(0.70, 0.115, f"a Gaussian batch scores ≈ {floor:.2f})", fontsize=7.5, color=LIGHT["ink3"])
        fig.text(0.70, 0.045, val, fontsize=15 if sn["sigreg"] is not None else 9.5, color=LIGHT["ink"], weight="bold")
        frames.append(fig_to_rgb(fig))
        plt.close(fig)
        durs.append(1800 if k == 0 else 1300 if k == 1 else 2600 if k == len(snaps) - 1 else 520)
    return save_gif(frames, durs, ASSETS / "sigreg.gif", colors=64)


# ----------------------------------------------------------------------------- embedding snapshots (hero)

def embedding_snapshots(n: int = 800):
    """Backbone embeddings (the features Evaluate/Inspect use) of the first n un-augmented Shapes images
    for: random init (seed-0 weights), ckpt step 600, ckpt step 768. Each snapshot -> 2-D PCA, then
    Procrustes-aligned (rotation/reflection + scale) to the final snapshot so the axes are comparable."""
    import torch
    from jepa_studio.data.loaders import _hwc_to_chw
    from jepa_studio.evaluate import embed, pca
    cfg = load_cfg()
    imgs, labels = shapes_images(n, cfg["seed"], 32)
    x = torch.stack([_hwc_to_chw(im) for im in imgs])
    snaps = []
    for name, w in (("random init", None), ("step 600", RUN / "ckpt" / "step-0000600" / "model.safetensors"),
                    ("step 768", RUN / "ckpt" / "step-0000768" / "model.safetensors")):
        e, _ = embed(build_model(cfg, w), x)
        e = (e - e.mean(0)) / (e.std(0) + 1e-6)          # standardized, as the probe / explorer do
        xy, var = pca(e, 2)
        snaps.append(dict(name=name, xy=xy.numpy().astype(np.float64), var=var.numpy()))
    ref = snaps[-1]["xy"]
    ref = ref / np.sqrt((ref ** 2).sum(1).mean())
    for s in snaps:
        a = s["xy"] / np.sqrt((s["xy"] ** 2).sum(1).mean())
        u, _, vt = np.linalg.svd(a.T @ ref)
        s["aligned"] = a @ (u @ vt)
    return snaps, labels, imgs


def world_episode(env_name: str = "two-room"):
    """Re-run the planning bot from runs/world-<env>/world.safetensors on the first planning-eval
    episode that CEM solved (the same selection build_site_models uses for assets/world-<env>.gif)."""
    import torch
    from jepa_studio.weights import load_state_dict
    from jepa_studio.world import (EPISODES_BY_ENV, BANK_SIZE, CEMConfig, WorldHParams, WorldModel, build_bank,
                                   collect_dataset, make_env, plan_and_execute)
    torch.set_num_threads(1)
    d = ROOT / "runs" / f"world-{env_name}"
    m = json.loads((d / "metrics.json").read_text())
    hp = dict(m["hparams"])
    hp["enc_channels"] = tuple(hp["enc_channels"])
    model = WorldModel(WorldHParams(**hp))
    load_state_dict(model, d / "world.safetensors")
    model.eval()
    env = make_env(env_name)
    seed = int(m.get("seed", 0))
    data = collect_dataset(env, EPISODES_BY_ENV.get(env_name, 400), m["config_world"]["episode_len"], seed)
    bank = build_bank(model, env, data.states, BANK_SIZE, seed=seed)
    pl = m["planning"]
    plan = CEMConfig(**m["plan_config"])
    solved = [s for s, ok in zip(pl["seeds"], pl["cem_success"]) if ok]
    start = solved[0]
    if env_name == "two-room":
        # the most typical doorway crossing: start and goal in different rooms (wall at x = 0.5),
        # episode length closest to the mean length of solved episodes
        cross = [s for s in solved if (env.reset(s)[0] - 0.5) * (env.goal_state(s)[0] - 0.5) < 0]
        lens = {s: plan_and_execute(env, model, s, None, 40, plan, None, seed=s, record=False)["steps"] for s in cross}
        if lens:
            start = min(lens, key=lambda s: (abs(lens[s] - pl["cem_mean_steps_success"]), s))
    r = plan_and_execute(env, model, start, None, 40, plan, bank, seed=start)
    r["start_seed"], r["metrics"] = start, pl
    return r


# ----------------------------------------------------------------------------- 6. hero.gif

def _hero_frame(plt, seg: int, title: str, subtitle: str):
    fig = plt.figure(figsize=(7.2, 4.05), dpi=100)
    fig.patches.append(plt.Rectangle((0.028, 0.892), 0.05, 0.07, transform=fig.transFigure, color=LIGHT["ink"],
                                     zorder=0))
    fig.text(0.053, 0.927, f"{seg}/4", ha="center", va="center", fontsize=10, color=LIGHT["surface"], weight="bold")
    fig.text(0.095, 0.927, title, va="center", fontsize=14, weight="bold", color=LIGHT["ink"])
    fig.text(0.095, 0.858, subtitle, va="center", fontsize=8.3, color=LIGHT["ink3"])
    for i in range(4):  # progress strip
        fig.patches.append(plt.Rectangle((0.028 + i * 0.237, 0.018), 0.227, 0.008, transform=fig.transFigure,
                                         color=LIGHT["ink"] if i + 1 == seg else LIGHT["line"], zorder=0))
    fig.text(0.972, 0.045, "jepa-studio", ha="right", va="bottom", fontsize=8, color=LIGHT["ink3"])
    return fig


def _clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def _upscale(a: np.ndarray, k: int) -> np.ndarray:
    return a.repeat(k, 0).repeat(k, 1)


def make_hero() -> Path:
    import torch
    torch.set_num_threads(1)
    plt = mpl_setup()
    cfg = load_cfg()
    ev = load_events()
    frames, durs = [], []

    def push(fig, ms):
        frames.append(fig_to_rgb(fig))
        durs.append(ms)
        plt.close(fig)

    # (a) unlabeled images ------------------------------------------------------------------
    info = next(e for e in ev if e.get("event") == "data")
    imgs, _ = shapes_images(60, cfg["seed"], 32)
    cols, rows, tile, gap = 12, 5, 44, 6
    Wg, Hg = cols * (tile + gap) - gap, rows * (tile + gap) - gap
    for k in range(1, 11):
        fig = _hero_frame(plt, 1, "Start with unlabeled images",
                          f"Shapes dataset (jepa_studio/data/synthetic.py): {info['count']:,} procedurally drawn "
                          "32×32 images, no labels used.")
        canvas = np.full((Hg, Wg, 3), 1.0)
        for i in range(min(len(imgs), k * 6)):
            r, c = divmod(i, cols)
            y, x = r * (tile + gap), c * (tile + gap)
            up = np.asarray(__import__("PIL.Image", fromlist=["Image"]).fromarray(
                (imgs[i] * 255).astype(np.uint8)).resize((tile, tile), 0)) / 255.0
            canvas[y:y + tile, x:x + tile] = up
        ax = fig.add_axes([(720 - Wg) / 2 / 720, 0.12, Wg / 720, Hg / 405])
        ax.imshow(canvas, interpolation="nearest")
        _clean(ax)
        push(fig, 1000 if k == 10 else 80)

    # (b) the real loss curves --------------------------------------------------------------
    st = [e for e in ev if e.get("event") == "step"]
    start = next(e for e in ev if e.get("event") == "start")
    end = next(e for e in ev if e.get("event") == "end")
    minutes = (end["t"] - start["t"]) / 60
    x = np.array([e["step"] for e in st])
    series = [("Total loss  $L$", np.array([e["loss"] for e in st]), LIGHT["ink"], False),
              (r"Prediction term  $L_\mathrm{pred}$", np.array([e["pred"] for e in st]), LIGHT["pred"], False),
              (r"SIGReg  $L_\mathrm{SIGReg}$  (log scale)", np.array([e["sigreg"] for e in st]), LIGHT["sigreg"], True)]
    n_fr = 18
    for k in range(1, n_fr + 1):
        upto = x[-1] * k / n_fr
        fig = _hero_frame(plt, 2, "Pretrain with LeJEPA: no labels, no decoder",
                          f"Real training log (runs/shapes-desktop/events.jsonl): {start['total_steps']} steps, batch "
                          f"{start['batch']}, {start['device'].upper()}, {minutes:.0f} min. λ = {cfg['objective']['lambda']}.")
        for j, (name, y, col, logy) in enumerate(series):
            ax = fig.add_axes([0.07 + j * 0.315, 0.2, 0.26, 0.52])
            m = x <= upto
            ax.plot(x[m], y[m], color=col, lw=2, solid_joinstyle="round")
            if m.sum():
                ax.plot([x[m][-1]], [y[m][-1]], "o", color=col, ms=5)
                ax.text(0.97, 0.93, f"{y[m][-1]:.3g}", transform=ax.transAxes, ha="right", va="top", fontsize=12,
                        weight="bold", color=LIGHT["ink"])
            ax.set_xlim(0, x[-1])
            if logy:
                ax.set_yscale("log")
                ax.set_ylim(1, 50)
                ax.set_yticks([1, 10])
                ax.set_yticklabels(["1", "10"])
            else:
                ax.set_ylim(0, y.max() * 1.05)
            ax.set_title(name, fontsize=9.5, loc="left", color=LIGHT["ink"])
            for sp in ("top", "right"):
                ax.spines[sp].set_visible(False)
            ax.grid(axis="y", color=LIGHT["line"], lw=0.7)
            ax.set_axisbelow(True)
            ax.tick_params(labelsize=7.5)
            ax.set_xticks([0, 250, 500, 750])
            ax.set_xlabel("step", fontsize=7.5, labelpad=1)
        push(fig, 1300 if k == n_fr else 70)

    # (c) embedding cloud -------------------------------------------------------------------
    from jepa_studio.data.synthetic import CLASSES
    snaps, labels, _ = embedding_snapshots(800)
    evj = json.loads((RUN / "eval.json").read_text())["methods"]
    lp = {"random init": evj["random-init encoder"]["linear_probe"], "step 768": evj["LeJEPA (this run)"]["linear_probe"]}
    P = [s["aligned"] for s in snaps]
    lim = max(np.abs(p).max() for p in P) * 1.05
    ease = lambda t: 0.5 - 0.5 * math.cos(math.pi * t)  # noqa: E731
    plan = [(0, 0, 0.0, 900)]
    plan += [(0, 1, (i + 1) / 10, 60) for i in range(10)] + [(1, 1, 0.0, 450)]
    plan += [(1, 2, (i + 1) / 6, 60) for i in range(6)] + [(2, 2, 0.0, 1400)]
    for a, b, t, ms in plan:
        xy = P[a] if a == b else (1 - ease(t)) * P[a] + ease(t) * P[b]
        state = snaps[a]["name"] if a == b else (snaps[b]["name"] if t >= 1 else f"{snaps[a]['name']} → {snaps[b]['name']}")
        exact = a == b or t >= 1
        fig = _hero_frame(plt, 3, "Embeddings spread out; classes start to separate",
                          "Backbone embeddings of 800 images, 2-D PCA. Colour = shape class, which training never saw.")
        ax = fig.add_axes([0.03, 0.08, 0.44, 0.74])
        for c in range(10):
            m = labels == c
            col = LIGHT["cat"][c % 5]
            if c < 5:
                ax.scatter(*xy[m].T, s=7, color=col, linewidths=0, alpha=0.9)
            else:
                ax.scatter(*xy[m].T, s=9, facecolors="none", edgecolors=col, linewidths=0.9, alpha=0.9)
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_aspect("equal")
        ax.set_facecolor(LIGHT["surface2"])
        _clean(ax)
        tx = 0.5
        fig.text(tx, 0.76, state, fontsize=13, weight="bold", color=LIGHT["ink"])
        fig.text(tx, 0.715, "real snapshot" if exact else "interpolated between real snapshots", fontsize=8,
                 color=LIGHT["ink3"])
        snap = snaps[b] if (a != b and t >= 1) else snaps[a]
        if exact:
            fig.text(tx, 0.64, f"share of total variance on the 1st principal axis: {snap['var'][0] * 100:.0f}%",
                     fontsize=8.5, color=LIGHT["ink2"])
            if snap["name"] in lp:
                fig.text(tx, 0.595, f"linear probe accuracy (eval.json): {lp[snap['name']] * 100:.0f}%  (chance 10%)",
                         fontsize=8.5, color=LIGHT["ink2"])
        # legend: 5 hues x filled / ring
        for c in range(10):
            r_, k_ = divmod(c, 2)
            lx, ly = tx + k_ * 0.2, 0.47 - r_ * 0.05
            col = LIGHT["cat"][c % 5]
            fig.lines.append(plt.Line2D([lx + 0.006], [ly + 0.012], transform=fig.transFigure, marker="o", ms=5,
                                        color=col, mfc=col if c < 5 else "none", mew=1, ls=""))
            fig.text(lx + 0.02, ly, CLASSES[c], fontsize=8, color=LIGHT["ink2"])
        fig.text(tx, 0.165, "Motion between snapshots is interpolated. Snapshots: random init,", fontsize=7,
                 color=LIGHT["ink3"])
        fig.text(tx, 0.13, "checkpoints step 600 and 768 (PCA, Procrustes-aligned).", fontsize=7, color=LIGHT["ink3"])
        push(fig, ms)

    # (d) planning bot ----------------------------------------------------------------------
    r = world_episode("two-room")
    pl = r["metrics"]
    n_ok, n = sum(pl["cem_success"]), len(pl["cem_success"])
    n_rnd = sum(pl["random_success"])
    goal = r["goal_frame"].transpose(1, 2, 0)
    T = len(r["frames_real"])
    ld = r["latent_dist"]
    for t in range(T):
        fig = _hero_frame(plt, 4, "Plan by imagining in a learned world model",
                          "runs/world-two-room: each step imagines 300 action sequences × 5 steps in latent space, "
                          "acts, re-plans.")
        ax = fig.add_axes([0.03, 0.24, 0.25, 0.25 * 720 / 405 * 0.9])
        ax.imshow(np.clip(r["frames_real"][t].transpose(1, 2, 0), 0, 1), interpolation="nearest")
        _clean(ax)
        ax.set_title(f"real, t = {t}", fontsize=9, loc="left", color=LIGHT["ink"])
        ax = fig.add_axes([0.30, 0.24, 0.25, 0.25 * 720 / 405 * 0.9])
        ax.imshow(np.clip(goal, 0, 1), interpolation="nearest")
        _clean(ax)
        ax.set_title("goal", fontsize=9, loc="left", color=LIGHT["ink"])
        k = min(t, len(r["retrieved_frames"]) - 1)
        fig.text(0.585, 0.715, "imagined next 5 steps", fontsize=9, color=LIGHT["predictor"], weight="bold")
        fig.text(0.585, 0.675, "(each shown as the nearest real frame: no decoder)", fontsize=7, color=LIGHT["ink3"])
        for h, fr in enumerate(r["retrieved_frames"][k]):
            ax = fig.add_axes([0.585 + h * 0.078, 0.475, 0.07, 0.07 * 720 / 405])
            ax.imshow(np.clip(fr.transpose(1, 2, 0), 0, 1), interpolation="nearest")
            _clean(ax)
            for sp in ax.spines.values():
                sp.set_visible(True)
                sp.set_color(LIGHT["predictor"])
                sp.set_linewidth(1.2)
            ax.set_title(f"+{h + 1}", fontsize=7.5, color=LIGHT["predictor"], pad=2)
        ax = fig.add_axes([0.6, 0.14, 0.36, 0.2])
        ax.plot(range(len(ld)), ld, color=LIGHT["predictor"], lw=1.5, alpha=0.35)
        ax.plot(range(min(t + 1, len(ld))), ld[:t + 1], color=LIGHT["predictor"], lw=2, marker="o", ms=3)
        ax.set_xlim(-0.3, max(len(ld) - 0.7, 1))
        ax.set_ylim(0, max(ld) * 1.1)
        ax.set_title("latent distance to goal  ‖z_t − z_goal‖²", fontsize=7.5, loc="left", color=LIGHT["ink2"])
        ax.tick_params(labelsize=7)
        ax.set_xlabel("step", fontsize=7, labelpad=0)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
        last = t == T - 1
        if last and r["success"]:
            fig.text(0.03, 0.105, f"✓ Reached the goal in {r['steps']} steps", fontsize=10.5, weight="bold",
                     color=LIGHT["good"])
        else:
            fig.text(0.03, 0.105, "planning…", fontsize=10.5, color=LIGHT["ink2"])
        fig.text(0.03, 0.065, f"Eval episode {r['start_seed']}. Over {n} eval episodes: planner {n_ok}/{n} solved, "
                              f"random actions {n_rnd}/{n}.", fontsize=7.5, color=LIGHT["ink3"])
        push(fig, 1600 if last else 270)
    out = ASSETS / "hero.gif"
    save_gif(frames, durs, out, colors=160)
    if out.stat().st_size > 3_000_000:
        save_gif(frames, durs, out, colors=96, lossy=40)
    return out


# ----------------------------------------------------------------------------- CLI

MAKERS = {"architecture": make_architecture, "sigreg": make_sigreg, "jepa-vs-pixels": make_jepa_vs_pixels,
          "throughput": make_throughput, "web-vs-desktop": make_web_vs_desktop, "hero": make_hero}


def main(argv: list[str]) -> None:
    import argparse
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("targets", nargs="*", help=f"any of: {', '.join(MAKERS)} (default: all)")
    ap.add_argument("--measure", action="store_true", help="re-run the throughput benchmark first (~3 CPU-minutes)")
    ap.add_argument("--check", metavar="DIR", help="also screenshot each SVG in light and dark mode into DIR")
    a = ap.parse_args(argv)
    unknown = [t for t in a.targets if t not in MAKERS]
    if unknown:
        ap.error(f"unknown target(s) {unknown}; choose from {list(MAKERS)}")
    if a.measure:
        print("measured", measure_throughput())
    for name in a.targets or list(MAKERS):
        out = MAKERS[name]()
        print(f"{out.relative_to(ROOT)}  {out.stat().st_size / 1024:.1f} KB")
        if a.check and out.suffix == ".svg":
            for p in render_svg(out, Path(a.check) / out.stem):
                print("   ", p)


if __name__ == "__main__":
    main(sys.argv[1:])
