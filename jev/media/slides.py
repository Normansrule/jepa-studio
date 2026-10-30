"""Clean 16:9 slides rendered with Pillow (no browser, no office suite).

Slide spec (one dict per slide; `type` picks the layout):
    {type: "title",    title, subtitle?, kicker?}
    {type: "bullets",  title, bullets: [str | {text, level}]}
    {type: "code",     title?, code, language?, caption?}
    {type: "image",    title?, path (workspace-relative), fit?: fit|cover, caption?}
    {type: "equation", title?, equations: [latex, ...] | equation: latex, caption?}
    {type: "chart",    title?, chart: {see jev.media.images.chart}}
    {type: "diagram",  title?, diagram: {see jev.media.images.diagram}}
Common optional keys: footer (small text bottom-left), number (shown bottom-right),
safe_bottom (fraction of the height kept free at the bottom, e.g. for burned-in captions).

Sizes are relative to the frame height, so 1920x1080 and the 1280x720 preview look identical.
Fonts: DejaVu/Liberation/system fonts if found, else the DejaVu copies inside matplotlib, else
Pillow's built-in font (see jev.media.common.font_file).
"""
from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw

from jev import wsfs

from .common import Limits, MediaError, font, inside, open_image_safely
from .images import Theme, chart, diagram, equation, fit_image, line_height, rgb, theme, wrap

FULL = (1920, 1080)
PREVIEW = (1280, 720)
SLIDE_TYPES = ("title", "bullets", "code", "image", "equation", "chart", "diagram")
MAX_CODE_LINES = 60


class _Canvas:
    def __init__(self, size, th: Theme):
        self.W, self.H = size
        self.u = self.H / 1080
        self.t = th
        self.im = Image.new("RGB", size, rgb(th.bg))
        self.d = ImageDraw.Draw(self.im)
        self.mx = int(self.W * 0.065)
        self.my = int(self.H * 0.075)
        self.safe = 0                    # extra bottom band kept free (burned-in video captions)

    def px(self, v: float) -> int:
        return max(1, int(round(v * self.u)))

    def title(self, text: str) -> int:
        """Draw the slide title; return the y where the body starts."""
        if not text:
            return self.my
        size = 64
        while True:
            f = font("bold", self.px(size))
            lines = wrap(self.d, text, f, self.W - 2 * self.mx)
            if len(lines) <= 2 or size <= 40:
                break
            size -= 4
        y = self.my
        for ln in lines[:2]:
            self.d.text((self.mx, y), ln, font=f, fill=self.t.fg)
            y += line_height(f)
        y += self.px(10)
        self.d.rounded_rectangle((self.mx, y, self.mx + self.px(96), y + self.px(7)), radius=self.px(3),
                                 fill=self.t.accent)
        return y + self.px(44)

    def footer(self, spec: dict) -> None:
        f = font("sans", self.px(24))
        y = self.H - self.my * 0.55
        if spec.get("footer"):
            self.d.text((self.mx, y), str(spec["footer"])[:120], font=f, fill=self.t.muted, anchor="lm")
        if spec.get("number") is not None:
            self.d.text((self.W - self.mx, y), str(spec["number"]), font=f, fill=self.t.muted, anchor="rm")

    def body_box(self, top: int, caption: bool = False) -> tuple[int, int, int, int]:
        bottom = self.H - self.my - self.safe - (self.px(64) if caption else 0)
        return self.mx, top, self.W - self.mx, bottom

    def caption(self, text: str) -> None:
        if not text:
            return
        f = font("sans", self.px(30))
        lines = wrap(self.d, text, f, self.W - 2 * self.mx)[:2]
        y = self.H - self.my - self.safe - line_height(f) * len(lines) + self.px(10)
        for ln in lines:
            self.d.text((self.W / 2, y), ln, font=f, fill=self.t.muted, anchor="ma")
            y += line_height(f)


def _title_slide(c: _Canvas, s: dict) -> None:
    from .images import title_card

    c.im.paste(title_card(s.get("title", ""), s.get("subtitle", ""), (c.W, c.H), c.t, s.get("kicker", "")))
    c.d = ImageDraw.Draw(c.im)


def _bullets(c: _Canvas, s: dict) -> None:
    top = c.title(s.get("title", ""))
    items = []
    for b in s.get("bullets") or []:
        if isinstance(b, dict):
            items.append((str(b.get("text", "")), max(0, min(2, int(b.get("level", 0))))))
        else:
            items.append((str(b), 0))
    if not items:
        raise MediaError("a bullets slide needs at least one bullet")
    x0, y0, x1, y1 = c.body_box(top)
    size = 54.0 if len(items) <= 4 else 44.0
    while True:
        f = font("sans", c.px(size))
        lh = line_height(f)
        gap = lh * 0.45
        layout = []
        h = 0.0
        for text, lvl in items:
            indent = c.px(size * 1.1) + lvl * c.px(size * 1.3)
            lines = wrap(c.d, text, f, (x1 - x0) - indent)
            layout.append((lines, lvl, indent))
            h += len(lines) * lh + gap
        if h - gap <= (y1 - y0) or size <= 26:
            break
        size -= 2
    y = y0
    for lines, lvl, indent in layout:
        r = max(4, int(lh * (0.11 if lvl == 0 else 0.08)))
        cx = x0 + indent - c.px(size * 0.75)
        cy = y + lh * 0.5
        col = c.t.accent if lvl == 0 else c.t.muted
        c.d.ellipse((cx - r, cy - r, cx + r, cy + r), fill=col)
        for ln in lines:
            if y + lh > y1 + lh * 0.2:
                break
            c.d.text((x0 + indent, y), ln, font=f, fill=c.t.fg if lvl == 0 else c.t.muted)
            y += lh
        y += gap


def _code_tokens(code: str, language: str | None, th: Theme) -> list[list[tuple[str, str]]]:
    """Lines of (text, colour). Uses pygments when installed, else plain text."""
    lines: list[list[tuple[str, str]]] = [[]]
    try:
        from pygments import lex
        from pygments.lexers import TextLexer, get_lexer_by_name, guess_lexer
        from pygments.token import Comment, Keyword, Name, Number, Operator, String

        try:
            lexer = get_lexer_by_name(language) if language else guess_lexer(code)
        except Exception:  # noqa: BLE001 - unknown language name
            lexer = TextLexer()
        cmap = th.code

        def colour(tok) -> str:
            if tok in Comment:
                return cmap["comment"]
            if tok in String:
                return cmap["string"]
            if tok in Number:
                return cmap["number"]
            if tok in Keyword:
                return cmap["keyword"]
            if tok in Name.Decorator:
                return cmap["decorator"]
            if tok in Name.Function:
                return cmap["function"]
            if tok in Name.Class:
                return cmap["class"]
            if tok in Name.Builtin:
                return cmap["builtin"]
            if tok in Operator:
                return cmap["operator"]
            return cmap["text"]

        for tok, val in lex(code, lexer):
            parts = val.split("\n")
            for k, part in enumerate(parts):
                if k:
                    lines.append([])
                if part:
                    lines[-1].append((part, colour(tok)))
    except ImportError:
        lines = [[(ln, th.code["text"])] for ln in code.split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return lines


def _code(c: _Canvas, s: dict) -> None:
    code = str(s.get("code", "")).expandtabs(4).rstrip("\n")
    if not code:
        raise MediaError("a code slide needs code")
    tok_lines = _code_tokens(code, s.get("language"), c.t)
    if len(tok_lines) > MAX_CODE_LINES:
        tok_lines = tok_lines[:MAX_CODE_LINES - 1] + [[("...", c.t.muted)]]
    top = c.title(s.get("title", ""))
    x0, y0, x1, y1 = c.body_box(top, caption=bool(s.get("caption")))
    pad = c.px(36)
    longest = max((sum(len(t) for t, _ in ln) for ln in tok_lines), default=1)
    size = 40.0
    while size > 14:
        f = font("mono", c.px(size))
        cw = c.d.textlength("M", font=f)
        lh = int(line_height(f) * 0.95)
        if cw * longest <= (x1 - x0) - 2 * pad - c.px(70) and lh * len(tok_lines) <= (y1 - y0) - 2 * pad:
            break
        size -= 1
    f = font("mono", c.px(size))
    cw = c.d.textlength("M", font=f)
    lh = int(line_height(f) * 0.95)
    panel_h = min(y1 - y0, lh * len(tok_lines) + 2 * pad)
    c.d.rounded_rectangle((x0, y0, x1, y0 + panel_h), radius=c.px(18), fill=c.t.code_bg, outline=c.t.grid,
                          width=c.px(2))
    gutter = c.d.textlength(str(len(tok_lines)), font=f) + c.px(28)
    y = y0 + pad
    nf = font("mono", c.px(size * 0.8))
    for i, ln in enumerate(tok_lines, 1):
        c.d.text((x0 + pad + gutter - c.px(20), y + lh * 0.1), str(i), font=nf, fill=c.t.muted, anchor="ra")
        x = x0 + pad + gutter
        for text, col in ln:
            if x > x1 - pad:
                break
            c.d.text((x, y), text, font=f, fill=col)
            x += c.d.textlength(text, font=f)
        y += lh
    c.caption(s.get("caption", ""))


def _image(c: _Canvas, s: dict, workspace: Path | None, limits: Limits) -> None:
    if not s.get("path"):
        raise MediaError("an image slide needs a path")
    if workspace is None:
        raise MediaError("image slides need a workspace to read from")
    src = open_image_safely(inside(s["path"], workspace), limits, root=Path(workspace).resolve())
    mode = s.get("fit", "fit")
    if mode == "cover" and not s.get("title"):
        c.im.paste(fit_image(src, (c.W, c.H), "cover", c.t.bg))
        c.d = ImageDraw.Draw(c.im)
        if s.get("caption"):
            band = Image.new("RGBA", (c.W, c.px(130)), rgb(c.t.bg) + (190,))
            c.im.paste(band, (0, c.H - c.px(130)), band)
            c.d = ImageDraw.Draw(c.im)
            c.caption(s["caption"])
        return
    top = c.title(s.get("title", ""))
    x0, y0, x1, y1 = c.body_box(top, caption=bool(s.get("caption")))
    fitted = fit_image(src, (x1 - x0, y1 - y0), mode, c.t.bg)
    c.im.paste(fitted, (x0, y0))
    c.caption(s.get("caption", ""))


def _equation(c: _Canvas, s: dict) -> None:
    eqs = s.get("equations") or ([s["equation"]] if s.get("equation") else [])
    if not eqs:
        raise MediaError("an equation slide needs equation or equations")
    top = c.title(s.get("title", ""))
    x0, y0, x1, y1 = c.body_box(top, caption=bool(s.get("caption")))
    h_each = min(c.px(110), (y1 - y0) / (len(eqs) * 1.6))
    rendered = [equation(e, int(h_each), c.t.fg, max_width=x1 - x0) for e in eqs[:6]]
    total = sum(r.height for r in rendered) + c.px(40) * (len(rendered) - 1)
    y = y0 + max(0, ((y1 - y0) - total) / 2)
    for r in rendered:
        c.im.paste(r, (int((c.W - r.width) / 2), int(y)), r)
        y += r.height + c.px(40)
    c.d = ImageDraw.Draw(c.im)
    c.caption(s.get("caption", ""))


def _chart(c: _Canvas, s: dict) -> None:
    top = c.title(s.get("title", ""))
    x0, y0, x1, y1 = c.body_box(top, caption=bool(s.get("caption")))
    spec = dict(s.get("chart") or {})
    spec.pop("title", None) if s.get("title") else None
    im = chart(spec, (x1 - x0, y1 - y0), c.t, text_scale=c.u * 1.45)
    c.im.paste(im, (x0, y0))
    c.caption(s.get("caption", ""))


def _diagram(c: _Canvas, s: dict) -> None:
    top = c.title(s.get("title", ""))
    x0, y0, x1, y1 = c.body_box(top, caption=bool(s.get("caption")))
    spec = dict(s.get("diagram") or {})
    spec.pop("title", None)
    diagram(spec, th=c.t, canvas=c.im, box=(x0, y0, x1, y1))
    c.d = ImageDraw.Draw(c.im)
    c.caption(s.get("caption", ""))


def render_slide(spec: dict, size: tuple[int, int] = FULL, th: str | Theme = "dark",
                 workspace: str | Path | None = None, limits: Limits | None = None) -> Image.Image:
    kind = spec.get("type", "bullets")
    if kind not in SLIDE_TYPES:
        raise MediaError(f"unknown slide type {kind!r}; use one of {', '.join(SLIDE_TYPES)}")
    W, H = size
    if not (160 <= W <= 7680 and 90 <= H <= 4320):
        raise MediaError(f"slide size {W}x{H} out of range")
    c = _Canvas(size, theme(th))
    c.safe = int(H * max(0.0, min(0.3, float(spec.get("safe_bottom", 0) or 0))))
    ws = Path(workspace).resolve() if workspace else None
    if kind == "title":
        _title_slide(c, spec)
    elif kind == "bullets":
        _bullets(c, spec)
    elif kind == "code":
        _code(c, spec)
    elif kind == "image":
        _image(c, spec, ws, limits or Limits())
    elif kind == "equation":
        _equation(c, spec)
    elif kind == "chart":
        _chart(c, spec)
    else:
        _diagram(c, spec)
    c.footer(spec)
    return c.im


def render_deck(slides: list[dict], out_dir: str | Path, workspace: str | Path, *, size: tuple[int, int] = FULL,
                th: str | Theme = "dark", pdf: bool = True, number: bool = True,
                progress=None, cancelled=None) -> dict:
    """Render every slide to PNG (slide_01.png, ...) and optionally one PDF of the deck."""
    if not slides:
        raise MediaError("no slides given")
    if len(slides) > 200:
        raise MediaError("at most 200 slides per deck")
    ws = Path(workspace).resolve()
    out = inside(out_dir, ws)
    wsfs.makedirs(ws, out)
    pngs, ims = [], []
    for i, s in enumerate(slides, 1):
        if cancelled and cancelled():
            raise MediaError("cancelled")
        spec = dict(s)
        if number and spec.get("type") != "title" and "number" not in spec:
            spec["number"] = f"{i} / {len(slides)}"
        im = render_slide(spec, size, th, ws)
        p = out / f"slide_{i:02d}.png"
        wsfs.save_image(im, ws, p, format="PNG")
        pngs.append(p)
        if pdf:
            ims.append(im)
        if progress:
            progress(i / len(slides), f"slide {i}/{len(slides)}")
    pdf_path = None
    if pdf and ims:
        pdf_path = out / "slides.pdf"
        with wsfs.atomic_writer(ws, pdf_path) as fh:
            ims[0].save(fh, format="PDF", save_all=True, append_images=ims[1:], resolution=144.0)
    return {"pngs": pngs, "pdf": pdf_path, "size": size}
