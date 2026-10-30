"""Footage editing: shot boundaries on a synthesized 3-segment video, search, highlights, cuts."""
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

from jev.media import ffmpeg as F
from jev.media import footage as FT
from jev.media.common import Limits, MediaError

FPS = 25
SEG = 30                         # frames per segment -> cuts at frames 30 and 60
SEGMENTS = [((200, 40, 40), "circle"), ((30, 60, 200), "square"), ((40, 170, 70), "triangle")]
RUN = Path(__file__).resolve().parent.parent / "runs" / "shapes-desktop"


def draw(colour, shape, k, size=(160, 120)):
    W, H = size
    im = Image.new("RGB", size, colour)
    d = ImageDraw.Draw(im)
    x, y = 20 + 2 * k, 40
    if shape == "circle":
        d.ellipse((x, y, x + 40, y + 40), fill=(255, 255, 255))
    elif shape == "square":
        d.rectangle((x, y, x + 40, y + 40), fill=(255, 230, 0))
    else:
        d.polygon([(x, y + 40), (x + 20, y), (x + 40, y + 40)], fill=(0, 0, 0))
    return im


def synth(path: Path):
    import imageio.v2 as imageio

    w = imageio.get_writer(path, fps=FPS, codec="libx264", quality=8, macro_block_size=1, pixelformat="yuv420p",
                           ffmpeg_log_level="error")
    for colour, shape in SEGMENTS:
        for k in range(SEG):
            w.append_data(np.asarray(draw(colour, shape, k)))
    w.close()
    return path


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    d = tmp_path_factory.mktemp("footage")
    synth(d / "clip.mp4")
    return d


def boundaries(a):
    return [s.start_frame for s in a.shots[1:]]


def test_shot_boundaries_histogram(ws):
    a = FT.analyze("clip.mp4", ws, cache=False)
    assert a.method == "histogram" and len(a.shots) == 3
    for got, want in zip(boundaries(a), [SEG, 2 * SEG]):
        assert abs(got - want) <= 2
    assert np.allclose(np.linalg.norm(a.features, axis=1), 1, atol=1e-4)
    # second call is served from the cache (npz, no pickle), kept in backend-owned storage outside the workspace
    from jev import wsfs
    b = FT.analyze("clip.mp4", ws)
    c = FT.analyze("clip.mp4", ws)
    assert boundaries(b) == boundaries(c) == boundaries(a)
    assert list(wsfs.cache_dir("media-cache").glob("*.npz"))
    assert not (ws / ".jev_media_cache").exists()


def test_subsampled_analysis_still_finds_cuts(ws):
    a = FT.analyze("clip.mp4", ws, sample_fps=10, cache=False)
    for got, want in zip(boundaries(a), [SEG, 2 * SEG]):
        assert abs(got - want) <= FPS / 10 + 2          # one sample period of slack


@pytest.mark.skipif(not (RUN / "encoder.safetensors").exists(), reason="sample run not present")
def test_shot_boundaries_jepa_encoder(ws):
    a = FT.analyze("clip.mp4", ws, encoder_run=RUN, cache=False)
    assert a.method == "jepa:shapes-desktop"
    assert np.allclose(np.linalg.norm(a.features, axis=1), 1, atol=1e-4)
    assert len(a.shots) == 3
    for got, want in zip(boundaries(a), [SEG, 2 * SEG]):
        assert abs(got - want) <= 2


def test_search_finds_right_segment(ws):
    a = FT.analyze("clip.mp4", ws)
    draw(*SEGMENTS[1], 5).save(ws / "query.png")
    hits = FT.search(a, FT.query_from_image(a, "query.png", ws), top_k=3)
    assert hits[0]["index"] == 1 and hits[0]["similarity"] > hits[1]["similarity"]
    hits = FT.search(a, FT.query_from_time(a, 2.9), top_k=3)
    assert hits[0]["index"] == 2


def test_highlight_reel_within_budget(ws):
    a = FT.analyze("clip.mp4", ws)
    budget = 2.5
    cuts = FT.select_highlights(a, budget, min_clip_s=0.5, max_clip_s=1.0)
    assert sum(c.duration for c in cuts) <= budget + 1e-6
    assert len({c.label for c in cuts}) == 3                  # diversity: every shot represented
    assert [c.start for c in cuts] == sorted(c.start for c in cuts)
    r = FT.render_cuts("clip.mp4", cuts, "reel.mp4", ws)
    assert r["duration_s"] <= budget + 0.1
    assert abs(r["duration_s"] - r["expected_s"]) < 0.1
    edl = r["edl"].read_text()
    assert edl.startswith("TITLE:") and edl.count("FROM CLIP NAME: clip.mp4") == len(cuts)
    assert r["cuts_json"].exists()


def test_cut_modes(ws):
    r = FT.render_cuts("clip.mp4", [FT.Cut(0.0, 1.0, "a")], "copy.mp4", ws)
    assert r["mode"] == "copy"
    r = FT.render_cuts("clip.mp4", [FT.Cut(0.5, 1.5, "a"), FT.Cut(2.5, 3.0, "b")], "re.mp4", ws)
    assert r["mode"] == "reencode" and abs(r["duration_s"] - 1.5) < 0.1
    with pytest.raises(MediaError, match="stream copy"):
        FT.render_cuts("clip.mp4", [FT.Cut(0.5, 1.5)], "x.mp4", ws, mode="copy")
    r = FT.render_cuts("clip.mp4", [FT.Cut(0.0, 1.0, "first")], "titled.mp4", ws, title="Trip", captions=True)
    assert r["mode"] == "reencode" and abs(r["duration_s"] - 3.5) < 0.15 and r["srt"].exists()
    with pytest.raises(MediaError):
        FT.render_cuts("clip.mp4", [FT.Cut(10.0, 12.0)], "y.mp4", ws)


def test_cuts_keep_audio(tmp_path):
    clip = tmp_path / "av.mp4"
    F.run(["-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=3", "-f", "lavfi", "-i",
           "sine=frequency=330:duration=3", *F.h264_args(30, "ultrafast"), "-c:a", "aac", str(clip)],
          tmp_path, outputs=[clip])
    r = FT.render_cuts("av.mp4", [FT.Cut(0.4, 1.2), FT.Cut(2.0, 2.6)], "out.mp4", tmp_path, title="T")
    info = F.probe(r["video"], tmp_path)
    assert info.has_audio and abs(info.duration_s - (2.5 + 0.8 + 0.6)) < 0.2


def test_footage_refusals(ws, tmp_path):
    with pytest.raises(MediaError, match="outside"):
        FT.analyze("../elsewhere.mp4", ws)
    with pytest.raises(MediaError, match="longer than"):
        FT.analyze("clip.mp4", ws, limits=Limits(max_footage_seconds=1.0), cache=False)
    (tmp_path / "bad_run").mkdir()
    with pytest.raises(MediaError, match="not a jepa_studio run"):
        FT.analyze("clip.mp4", ws, encoder_run=tmp_path / "bad_run", cache=False)
