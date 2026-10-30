"""Slides and procedural images render at the right size, in both themes, with font fallbacks."""
import numpy as np
import pytest
from PIL import Image

from jev.media import common, images as I, slides as S
from jev.media.common import MediaError

SPECS = [
    {"type": "title", "title": "What is a JEPA?", "subtitle": "Joint-Embedding Predictive Architectures"},
    {"type": "bullets", "title": "Bullets", "bullets": ["one", {"text": "two, indented", "level": 1},
                                                         "a much longer bullet " * 8]},
    {"type": "code", "title": "Code", "language": "python", "code": "def f(x):\n    return x * 2  # double\n"},
    {"type": "equation", "title": "Equation", "equations": [r"E = mc^2", r"\int_0^1 x\,dx = \frac{1}{2}"]},
    {"type": "chart", "title": "Chart", "chart": {"kind": "bar", "x": ["a", "b"], "series": [{"values": [1, 2]}]}},
    {"type": "diagram", "title": "Diagram", "diagram": {"nodes": [{"id": "a", "label": "A"}, {"id": "b", "label": "B"}],
                                                        "edges": [{"from": "a", "to": "b", "label": "f"}]}},
]


@pytest.mark.parametrize("size", [S.FULL, S.PREVIEW])
@pytest.mark.parametrize("spec", SPECS, ids=[s["type"] for s in SPECS])
def test_slide_sizes(spec, size):
    im = S.render_slide(spec, size, "dark")
    assert im.size == size and im.mode == "RGB"
    arr = np.asarray(im)
    assert arr.std() > 3                      # something was drawn, not a blank frame


def test_themes_differ_and_light_background():
    dark = np.asarray(S.render_slide(SPECS[1], S.PREVIEW, "dark"))
    light = np.asarray(S.render_slide(SPECS[1], S.PREVIEW, "light"))
    assert dark[5, 5].mean() < 60 and light[5, 5].mean() > 200
    with pytest.raises(MediaError):
        S.render_slide(SPECS[1], S.PREVIEW, "neon")


def test_image_slide_fit_and_cover(tmp_path):
    Image.new("RGB", (400, 100), (250, 0, 0)).save(tmp_path / "wide.png")
    fit = np.asarray(S.render_slide({"type": "image", "path": "wide.png", "fit": "fit"}, S.PREVIEW, "dark", tmp_path))
    cover = np.asarray(S.render_slide({"type": "image", "path": "wide.png", "fit": "cover"}, S.PREVIEW, "dark",
                                      tmp_path))
    assert (cover[..., 0] > 200).mean() > 0.95          # cover fills the frame
    assert 0.05 < (fit[..., 0] > 200).mean() < 0.6       # fit letterboxes it
    with pytest.raises(MediaError):
        S.render_slide({"type": "image", "path": "../outside.png"}, S.PREVIEW, "dark", tmp_path)


def test_code_slide_highlighting_changes_colours():
    pytest.importorskip("pygments")
    toks = S._code_tokens("def f():\n    return 1  # c", "python", I.theme("dark"))
    colours = {c for line in toks for _, c in line}
    assert len(colours) >= 3


def test_bad_specs_rejected():
    with pytest.raises(MediaError):
        S.render_slide({"type": "nope"})
    with pytest.raises(MediaError):
        S.render_slide({"type": "bullets", "title": "x", "bullets": []})
    with pytest.raises(MediaError):
        S.render_slide({"type": "equation", "equation": r"\frac{"}, S.PREVIEW)


def test_font_fallback_to_pillow_default(monkeypatch):
    monkeypatch.setattr(common, "_FONT_CANDIDATES", {})
    import matplotlib

    monkeypatch.setattr(matplotlib, "get_data_path", lambda: "/nonexistent")
    common.font_file.cache_clear()
    common.font.cache_clear()
    try:
        assert common.font_file("sans") is None
        im = S.render_slide(SPECS[1], S.PREVIEW)
        assert im.size == S.PREVIEW
    finally:
        common.font_file.cache_clear()
        common.font.cache_clear()


def test_deck_png_and_pdf(tmp_path):
    r = S.render_deck(SPECS[:3], "deck", tmp_path, size=S.PREVIEW)
    assert len(r["pngs"]) == 3 and r["pdf"].stat().st_size > 1000
    assert Image.open(r["pngs"][0]).size == S.PREVIEW


def test_images_chart_diagram_card_sheet_edit():
    c = I.chart({"kind": "line", "series": [{"name": "a", "values": [1, 3, 2]}], "title": "T"}, (800, 450))
    assert c.size == (800, 450)
    sc = I.chart({"kind": "scatter", "series": [{"x": [1, 2], "y": [3, 4]}]}, (640, 360), "light")
    assert sc.size == (640, 360)
    d = I.diagram({"nodes": [{"id": "x"}, {"id": "y", "shape": "ellipse"}], "edges": [{"from": "x", "to": "y"}]},
                  (800, 450))
    assert d.size == (800, 450)
    with pytest.raises(MediaError):
        I.diagram({"nodes": [{"id": "x"}], "edges": [{"from": "x", "to": "missing"}]})
    t = I.title_card("Hello", "world", (1280, 720))
    assert t.size == (1280, 720)
    sheet = I.contact_sheet([t, d, c], cols=2, thumb_w=160, labels=["a", "b", "c"])
    assert sheet.width == 2 * 160 + 3 * (160 // 24)
    e = I.edit_image(t, [{"op": "resize", "width": 640}, {"op": "crop", "box": [0, 0, 0.5, 0.5]},
                         {"op": "text", "text": "hi", "xy": [0.1, 0.1]}, {"op": "arrow", "from": [0, 0], "to": [0.5, 0.5]},
                         {"op": "rect", "box": [0.1, 0.1, 0.9, 0.9]}])
    assert e.size == (320, 180)
    with pytest.raises(MediaError):
        I.edit_image(t, [{"op": "explode"}])
