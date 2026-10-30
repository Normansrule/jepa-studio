"""Explainer assembly end to end (2 short scenes, small frames so it runs fast on a CPU)."""
import json

import numpy as np
import pytest
from PIL import Image

from jev.media import ffmpeg as F
from jev.media import video as V
from jev.media.common import MediaError

SIZE = (480, 270)


def scenes(extra_visual=None):
    return [
        {"narration": "A JEPA predicts embeddings. It never reconstructs pixels.",
         "visual": {"kind": "slide", "slide": {"type": "title", "title": "What is a JEPA?"}}},
        {"narration": "Here is a picture.", "visual": extra_visual or {"kind": "image", "path": "pic.png"},
         "min_seconds": 2.0},
    ]


@pytest.fixture
def ws(tmp_path):
    Image.fromarray((np.indices((180, 320)).sum(0) % 255).astype(np.uint8)).convert("RGB").save(tmp_path / "pic.png")
    return tmp_path


def _streams(path, ws):
    info = F.probe(path, ws)
    return info


def check_output(r, ws, expect_backend=None):
    info = _streams(r["video"], ws)
    sb = json.loads(r["storyboard"].read_text())
    durs = [s["duration"] for s in sb["scenes"]]
    expected = sum(durs) - (len(durs) - 1) * sb["crossfade_s"]
    assert abs(info.duration_s - expected) < 0.15, (info.duration_s, expected)
    assert abs(r["duration_s"] - expected) < 2e-3          # storyboard rounds to ms
    assert info.has_audio and info.audio_codec == "aac"
    assert info.video_codec == "h264" and (info.width, info.height) == SIZE
    cues = V.parse_srt(r["srt"].read_text())
    assert cues, "captions were written"
    starts = [c.start for c in cues]
    assert starts == sorted(starts) and all(c.end > c.start for c in cues)
    assert all(a.end <= b.start + 1e-3 for a, b in zip(cues, cues[1:]))
    for s in sb["scenes"]:                      # every caption sits inside its scene
        for c in s["captions"]:
            assert s["start"] - 1e-3 <= c["start"] and c["end"] <= s["start"] + s["duration"] + 1e-3
    if expect_backend:
        assert sb["tts_backends"] == [expect_backend]
    return info, sb


def test_explainer_end_to_end(ws):
    r = V.make_explainer(scenes(), "out/ex.mp4", ws, size=SIZE, fps=12, preset="ultrafast")
    info, sb = check_output(r, ws)
    assert r["caption_mode"] == ("libass" if F.has_filter("subtitles") else "pillow")
    assert sb["scenes"][1]["duration"] >= 2.0
    assert not (ws / "out" / ".ex.work").exists()
    # faststart: the moov atom comes before mdat
    head = r["video"].read_bytes()[:4096]
    assert b"moov" in head


def test_pillow_caption_fallback_silent_voice_and_music(ws, monkeypatch):
    monkeypatch.setattr(F, "has_filter", lambda name: False)
    music = ws / "music.wav"
    F.run(["-f", "lavfi", "-i", "sine=frequency=220:duration=1.5", str(music)], ws, outputs=[music])
    r = V.make_explainer(scenes(), "ex2.mp4", ws, size=SIZE, fps=12, preset="ultrafast", tts_backend="silent",
                         music="music.wav", ken_burns=False)
    info, sb = check_output(r, ws, expect_backend="silent")
    assert r["caption_mode"] == "pillow" and r["silent_narration"] and sb["music"]["path"] == "music.wav"


def test_clip_visual_and_srt_only(ws):
    clip = ws / "clip.mp4"
    F.run(["-f", "lavfi", "-i", "testsrc2=size=320x180:rate=12:duration=3", *F.h264_args(30, "ultrafast"),
           str(clip)], ws, outputs=[clip])
    r = V.make_explainer(scenes({"kind": "video", "path": "clip.mp4", "start": 0.5, "end": 2.0}), "ex3.mp4", ws,
                         size=SIZE, fps=12, preset="ultrafast", captions="srt")
    check_output(r, ws)
    assert r["caption_mode"] == "srt-only"


def test_refusals(ws):
    with pytest.raises(MediaError, match="outside"):
        V.make_explainer(scenes({"kind": "image", "path": "../../etc/passwd"}), "x.mp4", ws, size=SIZE)
    with pytest.raises(MediaError):
        V.make_explainer([], "x.mp4", ws)
    with pytest.raises(MediaError):
        V.make_explainer(scenes(), "x.mp4", ws, captions="loud")
    with pytest.raises(MediaError, match="outside"):
        V.make_explainer(scenes(), "../x.mp4", ws)


def test_caption_chunks_and_estimate():
    text = "This is a sentence. " * 12 + "And a final, much longer sentence that keeps going and going for a while."
    chunks = V.split_caption_chunks(text)
    assert all(len(c) <= 2 * V.CAPTION_LINE_CHARS + 10 for c in chunks)
    assert " ".join(chunks).split() == text.split()
    cues = V.make_cues(text, 1.0, 10.0)
    assert cues[0].start == 1.0 and abs(cues[-1].end - 11.0) < 1e-6
    small = V.estimate_render_seconds(scenes(), (1280, 720))
    big = V.estimate_render_seconds(scenes() * 40, (1920, 1080))
    assert 0 < small < 120 and big > 600
