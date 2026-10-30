"""Images without big models: charts (matplotlib), box-and-arrow diagrams from a JSON spec,
title cards, contact sheets and simple edits (resize, crop, annotate). Also the shared themes
and text-layout helpers used by slides and video.
"""
from __future__ import annotations

import io
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
from PIL import Image, ImageDraw

from .common import Limits, MediaError, font, inside, open_image_safely


# ----------------------------------------------------------------------------- themes

@dataclass(frozen=True)
class Theme:
    name: str
    bg: str
    panel: str
    fg: str
    muted: str
    accent: str
    accent2: str
    code_bg: str
    grid: str
    palette: tuple[str, ...]
    code: dict


THEMES = {
    "dark": Theme("dark", bg="#0e1117", panel="#171c28", fg="#eef1f7", muted="#9aa4b8", accent="#6ea8fe",
                  accent2="#f0b35a", code_bg="#131823", grid="#2a3142",
                  palette=("#6ea8fe", "#f0b35a", "#5fd0a5", "#e77d9a", "#b69cf2", "#7fd6e8"),
                  code={"keyword": "#c792ea", "string": "#a5d6a7", "comment": "#7f8aa3", "number": "#f0b35a",
                        "function": "#82aaff", "class": "#ffcb6b", "builtin": "#89ddff", "operator": "#d9dfeb",
                        "decorator": "#f78c6c", "text": "#e6e9f0"}),
    "light": Theme("light", bg="#fbfbfd", panel="#eef1f6", fg="#161a23", muted="#5b6475", accent="#2f62d6",
                   accent2="#c77a12", code_bg="#f3f5f9", grid="#d9dee8",
                   palette=("#2f62d6", "#c77a12", "#1f9d72", "#c4456a", "#7a55c9", "#2391a8"),
                   code={"keyword": "#8b2fc9", "string": "#2e7d32", "comment": "#7a8291", "number": "#b45309",
                         "function": "#1d4ed8", "class": "#a15c00", "builtin": "#0e7490", "operator": "#2b2f38",
                         "decorator": "#c2410c", "text": "#1f2430"}),
}


def theme(name: str | Theme | None) -> Theme:
    if isinstance(name, Theme):
        return name
    t = THEMES.get((name or "dark").lower())
    if t is None:
        raise MediaError(f"unknown theme {name!r}; use one of {', '.join(THEMES)}")
    return t


def rgb(color: str) -> tuple[int, int, int]:
    c = color.lstrip("#")
    if len(c) == 3:
        c = "".join(ch * 2 for ch in c)
    if len(c) != 6:
        raise MediaError(f"invalid colour {color!r} (use #rrggbb)")
    return int(c[0:2], 16), int(c[2:4], 16), int(c[4:6], 16)


# ----------------------------------------------------------------------------- text layout

def text_width(draw: ImageDraw.ImageDraw, text: str, f) -> float:
    return draw.textlength(text, font=f)


def wrap(draw: ImageDraw.ImageDraw, text: str, f, max_w: float) -> list[str]:
    """Greedy word wrap by pixel width; very long words are broken by characters."""
    lines: list[str] = []
    for para in str(text).split("\n"):
        words = para.split(" ")
        cur = ""
        for w in words:
            trial = (cur + " " + w).strip() if cur else w
            if text_width(draw, trial, f) <= max_w:
                cur = trial
                continue
            if cur:
                lines.append(cur)
            while text_width(draw, w, f) > max_w and len(w) > 1:
                k = len(w)
                while k > 1 and text_width(draw, w[:k], f) > max_w:
                    k -= 1
                lines.append(w[:k])
                w = w[k:]
            cur = w
        lines.append(cur)
    return lines


def line_height(f) -> int:
    try:
        a, d = f.getmetrics()
        return int((a + d) * 1.18)
    except AttributeError:
        return int(getattr(f, "size", 12) * 1.3)


def fit_image(im: Image.Image, size: tuple[int, int], mode: str = "fit", bg: str = "#000000") -> Image.Image:
    """Scale `im` into `size`: fit (letterbox, whole image visible) or cover (fill and crop)."""
    W, H = size
    w, h = im.size
    if mode == "stretch":
        return im.convert("RGB").resize((W, H), Image.Resampling.LANCZOS)
    s = (max if mode == "cover" else min)(W / w, H / h)
    nw, nh = max(1, round(w * s)), max(1, round(h * s))
    r = im.resize((nw, nh), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (W, H), rgb(bg))
    if r.mode == "RGBA":
        canvas.paste(r, ((W - nw) // 2, (H - nh) // 2), r)
    else:
        canvas.paste(r.convert("RGB"), ((W - nw) // 2, (H - nh) // 2))
    return canvas


# ----------------------------------------------------------------------------- charts

MAX_POINTS = 200_000


def _require_matplotlib() -> None:
    try:
        import matplotlib  # noqa: F401
    except ImportError as e:
        raise MediaError("charts and equations need matplotlib: `pip install matplotlib` "
                         "(or `pip install 'jepa-studio[media]'`)") from e


def _figure(size: tuple[int, int], th: Theme):
    _require_matplotlib()
    from matplotlib.figure import Figure

    fig = Figure(figsize=(size[0] / 100, size[1] / 100), dpi=100, facecolor=th.bg)
    return fig


def _to_image(fig, size: tuple[int, int], transparent: bool = False) -> Image.Image:
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    FigureCanvasAgg(fig)
    buf = io.BytesIO()
    fig.savefig(buf, format="png", dpi=100, facecolor=fig.get_facecolor(), transparent=transparent)
    buf.seek(0)
    im = Image.open(buf)
    im.load()
    if im.size != size:
        im = im.resize(size, Image.Resampling.LANCZOS)
    return im.convert("RGBA" if transparent else "RGB")


def chart(spec: dict, size: tuple[int, int] = (1600, 900), th: str | Theme = "dark",
          text_scale: float | None = None) -> Image.Image:
    """spec: {kind: line|bar|scatter, title?, x?: [...], series: [{name?, values: [...]} or
    {name?, x: [...], y: [...]} for scatter], xlabel?, ylabel?, log_y?: bool}
    text_scale: size of text and lines relative to a 900 px tall chart (default: from `size`)."""
    t = theme(th)
    kind = spec.get("kind", "line")
    if kind not in ("line", "bar", "scatter"):
        raise MediaError("chart kind must be line, bar or scatter")
    series = spec.get("series") or []
    if not series:
        raise MediaError("chart needs at least one series")
    total = sum(len(s.get("values") or s.get("y") or []) for s in series)
    if total > MAX_POINTS:
        raise MediaError(f"chart has {total} points; the cap is {MAX_POINTS}")
    fig = _figure(size, t)
    scale = text_scale or size[1] / 900
    fs = 16 * scale                                  # points; 1 pt = 100/72 px at dpi 100
    fs_px = fs * 100 / 72
    lm = min(0.3, fs_px * 4.8 / size[0] + 0.01)
    bm = min(0.35, fs_px * (3.8 if spec.get("xlabel") else 2.4) / size[1])
    top = 1 - fs_px * (3.2 if spec.get("title") else 1.0) / size[1]
    ax = fig.add_axes((lm, bm, 0.97 - lm, max(0.2, top - bm)))
    ax.set_facecolor(t.bg)
    for s in ax.spines.values():
        s.set_color(t.grid)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(colors=t.muted, labelsize=fs * 0.9)
    ax.grid(True, axis="y" if kind == "bar" else "both", color=t.grid, linewidth=0.8 * scale, alpha=0.8)
    ax.set_axisbelow(True)
    x = spec.get("x")
    nb = len(series)
    for i, s in enumerate(series):
        c = s.get("color") or t.palette[i % len(t.palette)]
        name = s.get("name")
        if kind == "scatter":
            xs, ys = np.asarray(s.get("x", []), float), np.asarray(s.get("y", []), float)
            if len(xs) != len(ys):
                raise MediaError("scatter series need x and y of the same length")
            ax.scatter(xs, ys, s=36 * scale ** 2, color=c, alpha=0.85, label=name, edgecolors="none")
            continue
        vals = np.asarray(s.get("values", []), float)
        xs = list(range(len(vals))) if x is None else x
        if len(xs) != len(vals):
            raise MediaError(f"series {name or i}: {len(vals)} values but {len(xs)} x labels")
        if kind == "line":
            ax.plot(range(len(vals)) if x is not None and isinstance(xs[0], str) else xs, vals, color=c,
                    linewidth=3.2 * scale, label=name, marker="o" if len(vals) <= 24 else None,
                    markersize=6 * scale)
        else:
            w = 0.8 / nb
            pos = np.arange(len(vals)) - 0.4 + w * (i + 0.5)
            ax.bar(pos, vals, width=w * 0.92, color=c, label=name)
    if x is not None and kind in ("bar", "line") and (kind == "bar" or isinstance(x[0], str)):
        step = max(1, len(x) // 12)
        ax.set_xticks(range(0, len(x), step))
        ax.set_xticklabels([str(v) for v in x][::step])
    if spec.get("log_y"):
        ax.set_yscale("log")
    if spec.get("xlabel"):
        ax.set_xlabel(spec["xlabel"], color=t.muted, fontsize=fs)
    if spec.get("ylabel"):
        ax.set_ylabel(spec["ylabel"], color=t.muted, fontsize=fs)
    if spec.get("title"):
        fig.text(lm, 1 - fs_px * 1.4 / size[1], spec["title"], color=t.fg, fontsize=fs * 1.5, fontweight="bold", va="center")
    if any(s.get("name") for s in series):
        leg = ax.legend(frameon=False, fontsize=fs * 0.95, loc="best")
        for txt in leg.get_texts():
            txt.set_color(t.fg)
    return _to_image(fig, size)


def equation(tex: str, height_px: int, color: str, max_width: int | None = None) -> Image.Image:
    """Render a mathtext expression (a LaTeX subset, no TeX install needed) to a transparent image."""
    _require_matplotlib()
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    tex = tex.strip()
    if len(tex) > 600:
        raise MediaError("equation longer than 600 characters")
    if not (tex.startswith("$") and tex.endswith("$")):
        tex = f"${tex}$"
    fig = Figure(figsize=(1, 1), dpi=100)
    FigureCanvasAgg(fig)
    fig.text(0, 0, tex, fontsize=height_px * 0.72, color=color)
    buf = io.BytesIO()
    try:
        fig.savefig(buf, format="png", dpi=100, transparent=True, bbox_inches="tight", pad_inches=0.06)
    except ValueError as e:
        raise MediaError(f"cannot render equation: {str(e).splitlines()[0][:200]}") from e
    buf.seek(0)
    im = Image.open(buf).convert("RGBA")
    if max_width and im.width > max_width:
        im = im.resize((max_width, max(1, round(im.height * max_width / im.width))), Image.Resampling.LANCZOS)
    return im


# ----------------------------------------------------------------------------- diagrams

def _layout(nodes: list[dict], edges: list[dict]) -> dict[str, tuple[float, float]]:
    """Left-to-right layered layout for nodes without x/y (longest path from sources)."""
    ids = [n["id"] for n in nodes]
    layer = {i: 0 for i in ids}
    for _ in range(len(ids)):
        changed = False
        for e in edges:
            a, b = e.get("from"), e.get("to")
            if a in layer and b in layer and layer[b] < layer[a] + 1 and layer[a] + 1 < len(ids):
                layer[b] = layer[a] + 1
                changed = True
        if not changed:
            break
    cols: dict[int, list[str]] = {}
    for i in ids:
        cols.setdefault(layer[i], []).append(i)
    ncol = max(cols) + 1
    pos = {}
    for c, members in cols.items():
        for r, i in enumerate(members):
            pos[i] = ((c + 0.5) / ncol, (r + 0.5) / len(members))
    return pos


def _box_edge(cx, cy, hw, hh, tx, ty, shape):
    """Point where the ray from the box centre toward (tx, ty) leaves the box/ellipse."""
    dx, dy = tx - cx, ty - cy
    if dx == 0 and dy == 0:
        return cx, cy
    if shape == "ellipse":
        k = 1 / math.sqrt((dx / hw) ** 2 + (dy / hh) ** 2)
    else:
        k = min(hw / abs(dx) if dx else 1e9, hh / abs(dy) if dy else 1e9)
    return cx + dx * k, cy + dy * k


def _arrow(d: ImageDraw.ImageDraw, p0, p1, color, width, head, dashed=False):
    x0, y0 = p0
    x1, y1 = p1
    ang = math.atan2(y1 - y0, x1 - x0)
    bx, by = x1 - head * 0.9 * math.cos(ang), y1 - head * 0.9 * math.sin(ang)
    if dashed:
        L = math.hypot(bx - x0, by - y0)
        n = max(1, int(L // (width * 5)))
        for k in range(0, n, 2):
            a, b = k / n, min(1, (k + 1) / n)
            d.line([(x0 + (bx - x0) * a, y0 + (by - y0) * a), (x0 + (bx - x0) * b, y0 + (by - y0) * b)],
                   fill=color, width=width)
    else:
        d.line([(x0, y0), (bx, by)], fill=color, width=width)
    left = (x1 - head * math.cos(ang - 0.42), y1 - head * math.sin(ang - 0.42))
    right = (x1 - head * math.cos(ang + 0.42), y1 - head * math.sin(ang + 0.42))
    d.polygon([(x1, y1), left, right], fill=color)


def diagram(spec: dict, size: tuple[int, int] = (1600, 900), th: str | Theme = "dark",
            canvas: Image.Image | None = None, box: tuple[int, int, int, int] | None = None) -> Image.Image:
    """spec: {title?, nodes: [{id, label, x?, y?, w?, h?, shape?: box|round|ellipse, color?}],
    edges: [{from, to, label?, dashed?}]}. x, y are centres in 0..1 of the drawing area; nodes
    without them are laid out left to right. w, h are fractions of the area (default sized to the grid)."""
    t = theme(th)
    nodes = spec.get("nodes") or []
    edges = spec.get("edges") or []
    if not nodes:
        raise MediaError("diagram needs at least one node")
    if len(nodes) > 60 or len(edges) > 200:
        raise MediaError("diagram too large (max 60 nodes, 200 edges)")
    seen = set()
    for n in nodes:
        if "id" not in n:
            raise MediaError("every diagram node needs an id")
        if n["id"] in seen:
            raise MediaError(f"duplicate node id {n['id']!r}")
        seen.add(n["id"])
    for e in edges:
        if e.get("from") not in seen or e.get("to") not in seen:
            raise MediaError(f"edge {e.get('from')}->{e.get('to')} refers to an unknown node")
    im = canvas if canvas is not None else Image.new("RGB", size, rgb(t.bg))
    W, H = im.size
    d = ImageDraw.Draw(im)
    u = H / 1080
    if box is None:
        mx, my = int(W * 0.06), int(H * 0.08)
        top = my
        if spec.get("title"):
            tf = font("bold", 58 * u)
            d.text((mx, my), spec["title"], font=tf, fill=t.fg)
            top = my + line_height(tf) + int(24 * u)
        box = (mx, top, W - mx, H - my)
    x0, y0, x1, y1 = box
    aw, ah = x1 - x0, y1 - y0
    auto = _layout([n for n in nodes], edges)
    ncol = len({round(v[0], 4) for v in auto.values()})
    nrow = max(sum(1 for v in auto.values() if round(v[0], 4) == c) for c in {round(v[0], 4) for v in auto.values()})
    dw, dh = min(0.22, 0.72 / ncol), min(0.24, 0.6 / nrow)
    geo = {}
    for i, n in enumerate(nodes):
        ax_, ay_ = auto[n["id"]]
        cx = x0 + aw * float(n.get("x", ax_))
        cy = y0 + ah * float(n.get("y", ay_))
        hw = aw * float(n.get("w", dw)) / 2
        hh = ah * float(n.get("h", dh)) / 2
        geo[n["id"]] = (cx, cy, hw, hh, n.get("shape", "round"))
    lw = max(2, int(4 * u))
    for e in edges:
        ax0, ay0, ahw, ahh, ash = geo[e["from"]]
        bx0, by0, bhw, bhh, bsh = geo[e["to"]]
        p0 = _box_edge(ax0, ay0, ahw, ahh, bx0, by0, ash)
        p1 = _box_edge(bx0, by0, bhw, bhh, ax0, ay0, bsh)
        col = e.get("color") or t.muted
        _arrow(d, p0, p1, col, lw, 22 * u, dashed=bool(e.get("dashed")))
        if e.get("label"):
            lf = font("sans", 26 * u)
            mx_, my_ = (p0[0] + p1[0]) / 2, (p0[1] + p1[1]) / 2
            tw = text_width(d, e["label"], lf)
            pad = 6 * u
            d.rounded_rectangle((mx_ - tw / 2 - pad, my_ - 18 * u, mx_ + tw / 2 + pad, my_ + 18 * u),
                                radius=int(8 * u), fill=rgb(t.bg))
            d.text((mx_, my_), e["label"], font=lf, fill=t.muted, anchor="mm")
    for i, n in enumerate(nodes):
        cx, cy, hw, hh, shape = geo[n["id"]]
        col = n.get("color") or t.palette[i % len(t.palette)]
        rect = (cx - hw, cy - hh, cx + hw, cy + hh)
        fill = _mix(rgb(t.panel), rgb(col), 0.18)
        if shape == "ellipse":
            d.ellipse(rect, fill=fill, outline=rgb(col), width=lw)
        else:
            d.rounded_rectangle(rect, radius=int((18 if shape == "round" else 4) * u), fill=fill,
                                outline=rgb(col), width=lw)
        label = str(n.get("label", n["id"]))
        size_px = 34 * u
        while True:
            f = font("bold", size_px)
            lines = wrap(d, label, f, hw * 2 * 0.86)
            lh = line_height(f)
            if (len(lines) * lh <= hh * 2 * 0.86 and all(text_width(d, ln, f) <= hw * 2 * 0.9 for ln in lines)) \
                    or size_px < 12 * u:
                break
            size_px *= 0.9
        ty = cy - len(lines) * lh / 2 + lh / 2
        for ln in lines:
            d.text((cx, ty), ln, font=f, fill=t.fg, anchor="mm")
            ty += lh
    return im


def _mix(a, b, k):
    return tuple(int(round(a[i] * (1 - k) + b[i] * k)) for i in range(3))


# ----------------------------------------------------------------------------- cards and sheets

def title_card(title: str, subtitle: str = "", size: tuple[int, int] = (1920, 1080),
               th: str | Theme = "dark", kicker: str = "") -> Image.Image:
    t = theme(th)
    W, H = size
    u = H / 1080
    im = Image.new("RGB", size, rgb(t.bg))
    d = ImageDraw.Draw(im)
    # a soft accent glow in one corner keeps cards from looking flat
    glow = Image.new("L", (W // 4, H // 4), 0)
    gd = ImageDraw.Draw(glow)
    gd.ellipse((W // 4 * 0.55, -H // 4 * 0.6, W // 4 * 1.45, H // 4 * 0.75), fill=42)
    from PIL import ImageFilter

    glow = glow.filter(ImageFilter.GaussianBlur(W / 4 * 0.08)).resize(size, Image.Resampling.BICUBIC)
    im = Image.composite(Image.new("RGB", size, rgb(t.accent)), im, glow)
    d = ImageDraw.Draw(im)
    mx = int(W * 0.08)
    maxw = W - 2 * mx
    size_px = 96 * u
    while True:
        tf = font("bold", size_px)
        lines = wrap(d, title, tf, maxw)
        if len(lines) <= 3 or size_px < 40 * u:
            break
        size_px *= 0.9
    sf = font("sans", 40 * u)
    sub = wrap(d, subtitle, sf, maxw) if subtitle else []
    kf = font("bold", 26 * u)
    total = len(lines) * line_height(tf) + (len(sub) * line_height(sf) + 30 * u if sub else 0) + \
        (line_height(kf) + 18 * u if kicker else 0)
    y = (H - total) / 2
    if kicker:
        d.text((mx, y), kicker.upper(), font=kf, fill=t.accent)
        y += line_height(kf) + 18 * u
    for ln in lines:
        d.text((mx, y), ln, font=tf, fill=t.fg)
        y += line_height(tf)
    d.rectangle((mx, y + 8 * u, mx + 120 * u, y + 16 * u), fill=t.accent)
    y += 30 * u
    for ln in sub:
        d.text((mx, y + 10 * u), ln, font=sf, fill=t.muted)
        y += line_height(sf)
    return im


def contact_sheet(images: Iterable[Image.Image], cols: int = 4, thumb_w: int = 320,
                  labels: list[str] | None = None, th: str | Theme = "dark") -> Image.Image:
    t = theme(th)
    ims = list(images)
    if not ims:
        raise MediaError("contact sheet needs at least one image")
    cols = max(1, min(cols, len(ims)))
    thumb_h = int(round(thumb_w * 9 / 16))
    pad = max(6, thumb_w // 24)
    lf = font("sans", max(12, thumb_w // 18))
    lab_h = line_height(lf) if labels else 0
    rows = math.ceil(len(ims) / cols)
    W = cols * thumb_w + (cols + 1) * pad
    H = rows * (thumb_h + lab_h) + (rows + 1) * pad
    sheet = Image.new("RGB", (W, H), rgb(t.bg))
    d = ImageDraw.Draw(sheet)
    for i, im in enumerate(ims):
        r, c = divmod(i, cols)
        x = pad + c * (thumb_w + pad)
        y = pad + r * (thumb_h + lab_h + pad)
        sheet.paste(fit_image(im, (thumb_w, thumb_h), "fit", t.panel), (x, y))
        if labels and i < len(labels):
            d.text((x, y + thumb_h + 2), str(labels[i])[:60], font=lf, fill=t.muted)
    return sheet


def _xy(v: list[float], size: tuple[int, int]) -> tuple[float, ...]:
    vals = [float(a) for a in v]
    if all(0.0 <= a <= 1.0 for a in vals):
        return tuple(a * (size[i % 2]) for i, a in enumerate(vals))
    return tuple(vals)


def edit_image(im: Image.Image, ops: list[dict], th: str | Theme = "dark") -> Image.Image:
    """Apply ops in order. Coordinates are pixels, or fractions when every value is within 0..1.
      {op: resize, width, height?, mode?: fit|cover|stretch}   {op: crop, box: [x0, y0, x1, y1]}
      {op: text, text, xy: [x, y], size?: px, color?}             {op: rect, box, color?, width?}
      {op: arrow, from: [x, y], to: [x, y], color?, width?}"""
    t = theme(th)
    im = im.convert("RGB")
    if len(ops) > 100:
        raise MediaError("at most 100 edit operations")
    for op in ops:
        kind = op.get("op")
        if kind == "resize":
            w = int(op["width"])
            h = int(op.get("height") or round(im.height * w / im.width))
            if not (1 <= w <= 8192 and 1 <= h <= 8192):
                raise MediaError("resize must stay within 1..8192 px per side")
            im = fit_image(im, (w, h), op.get("mode", "fit"), t.bg)
        elif kind == "crop":
            x0, y0, x1, y1 = _xy(op["box"], im.size)
            box = (max(0, int(x0)), max(0, int(y0)), min(im.width, int(x1)), min(im.height, int(y1)))
            if box[2] <= box[0] or box[3] <= box[1]:
                raise MediaError(f"empty crop box {op['box']}")
            im = im.crop(box)
        elif kind == "text":
            d = ImageDraw.Draw(im)
            f = font("bold", int(op.get("size", max(14, im.height // 20))))
            x, y = _xy(op.get("xy", [0.05, 0.05]), im.size)
            col = op.get("color", t.fg)
            d.text((x, y), str(op["text"])[:500], font=f, fill=col, stroke_width=max(1, f.size // 14),
                   stroke_fill=t.bg)
        elif kind == "rect":
            d = ImageDraw.Draw(im)
            x0, y0, x1, y1 = _xy(op["box"], im.size)
            d.rectangle((x0, y0, x1, y1), outline=op.get("color", t.accent2), width=int(op.get("width", 4)))
        elif kind == "arrow":
            d = ImageDraw.Draw(im)
            p0 = _xy(op["from"], im.size)
            p1 = _xy(op["to"], im.size)
            w = int(op.get("width", max(3, im.height // 160)))
            _arrow(d, p0, p1, op.get("color", t.accent2), w, w * 5)
        else:
            raise MediaError(f"unknown image op {kind!r} (resize, crop, text, rect, arrow)")
    return im


def load_image(path: str | Path, workspace: str | Path, limits: Limits | None = None) -> Image.Image:
    return open_image_safely(inside(path, workspace), limits, root=Path(workspace).resolve())


def save_png(im: Image.Image, out: Path, meta: dict[str, Any] | None = None, root: Path | None = None) -> Path:
    """Save a PNG under `root` (the workspace or a scratch folder) without following links."""
    from PIL.PngImagePlugin import PngInfo

    from jev import wsfs
    info = PngInfo()
    for k, v in (meta or {}).items():
        info.add_text(str(k), str(v)[:2000])
    if root is None:
        try:
            root = wsfs.root_of(out)
        except wsfs.UnsafePath:
            root = Path(out).parent
    wsfs.save_image(im, root, out, format="PNG", pnginfo=info, optimize=False)
    return out
