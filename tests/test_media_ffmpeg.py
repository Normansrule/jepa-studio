"""FFmpeg wrapper: locating, argument-list runs, workspace confinement, caps before decoding,
progress, timeouts and cancellation."""
import threading

import numpy as np
import pytest

from jev.media import ffmpeg as F
from jev.media.common import Limits, MediaError


def make_clip(ws, name="c.mp4", seconds=2.0, size=(160, 96), fps=12, audio=True):
    args = ["-f", "lavfi", "-i", f"testsrc2=size={size[0]}x{size[1]}:rate={fps}:duration={seconds}"]
    if audio:
        args += ["-f", "lavfi", "-i", f"sine=frequency=440:duration={seconds}", "-c:a", "aac"]
    out = ws / name
    F.run(args + [*F.h264_args(28, "ultrafast"), str(out)], ws, outputs=[out])
    return out


def test_find_ffmpeg_and_imageio_fallback(monkeypatch, tmp_path):
    assert F.find_ffmpeg()
    pytest.importorskip("imageio_ffmpeg")
    F.find_ffmpeg.cache_clear()
    monkeypatch.setenv("PATH", str(tmp_path))
    try:
        exe = F.find_ffmpeg()
        assert "imageio_ffmpeg" in exe
    finally:
        F.find_ffmpeg.cache_clear()


def test_probe_and_banner_fallback(tmp_path, monkeypatch):
    clip = make_clip(tmp_path)
    info = F.probe(clip, tmp_path)
    assert (info.width, info.height) == (160, 96) and abs(info.fps - 12) < 0.01
    assert abs(info.duration_s - 2.0) < 0.15 and info.has_audio and info.video_codec == "h264"
    monkeypatch.setattr(F, "find_ffprobe", lambda: None)
    info2 = F.probe(clip, tmp_path)
    assert info2.source == "ffmpeg -i"
    assert (info2.width, info2.height) == (160, 96) and abs(info2.duration_s - info.duration_s) < 0.05
    assert info2.has_audio and info2.audio_rate == info.audio_rate


def test_caps_checked_before_decoding(tmp_path):
    clip = make_clip(tmp_path, seconds=2.0)
    with pytest.raises(MediaError, match="longer than"):
        F.probe(clip, tmp_path, Limits(max_footage_seconds=1.0))
    with pytest.raises(MediaError, match="megapixel"):
        F.probe(clip, tmp_path, Limits(max_pixels=100 * 50))
    with pytest.raises(MediaError, match="larger than"):
        F.probe(clip, tmp_path, Limits(max_bytes=10))
    with pytest.raises(MediaError, match="clip"):
        F.probe(clip, tmp_path, Limits(max_clip_seconds=1.0), kind="clip")
    (tmp_path / "junk.mp4").write_bytes(b"not a video at all")
    with pytest.raises(MediaError):
        F.probe("junk.mp4", tmp_path)


def test_workspace_confinement(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    outside = make_clip(tmp_path, "outside.mp4", audio=False)
    with pytest.raises(MediaError, match="outside the workspace"):
        F.probe(outside, ws)
    with pytest.raises(MediaError, match="outside the workspace"):
        F.run(["-i", str(outside), str(ws / "o.mp4")], ws, outputs=[ws / "o.mp4"])
    with pytest.raises(MediaError, match="outside the workspace"):
        F.run(["-i", "../outside.mp4", str(ws / "o.mp4")], ws, outputs=[ws / "o.mp4"])
    with pytest.raises(MediaError, match="outside the workspace"):
        F.run(["-f", "lavfi", "-i", "testsrc", "-t", "1", str(tmp_path / "o.mp4")], ws, outputs=[tmp_path / "o.mp4"])
    for bad in ("concat:a.mp4|b.mp4", "http://example.com/x.mp4", "subfile:x.mp4"):
        with pytest.raises(MediaError, match="URL or protocol"):
            F.run(["-i", bad, str(ws / "o.mp4")], ws, outputs=[ws / "o.mp4"])
    with pytest.raises(MediaError, match="open files"):
        F.run(["-f", "lavfi", "-i", "movie=/etc/passwd", str(ws / "o.mp4")], ws, outputs=[ws / "o.mp4"])
    with pytest.raises(MediaError, match="open files"):
        F.run(["-f", "lavfi", "-i", "testsrc", "-vf", "drawtext=textfile=/etc/passwd", str(ws / "o.mp4")], ws,
              outputs=[ws / "o.mp4"])
    with pytest.raises(MediaError, match="outside the workspace"):
        F.run(["-f", "lavfi", "-i", "testsrc", "-vf", "subtitles=../../x.srt", str(ws / "o.mp4")], ws,
              outputs=[ws / "o.mp4"])


def test_progress_reported(tmp_path):
    seen = []
    out = tmp_path / "p.mp4"
    F.run(["-f", "lavfi", "-i", "testsrc2=size=160x96:rate=12:duration=3", *F.h264_args(30, "ultrafast"), str(out)],
          tmp_path, outputs=[out], duration_s=3.0, progress=lambda f, m: seen.append(f))
    assert seen and seen[-1] == 1.0 and all(0 <= f <= 1 for f in seen)


def test_cancel_and_timeout_remove_partial_output(tmp_path):
    out = tmp_path / "long.mp4"
    ev = threading.Event()
    threading.Timer(0.4, ev.set).start()
    args = ["-re", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=10:duration=60", *F.h264_args(30, "ultrafast"),
            str(out)]
    with pytest.raises(MediaError, match="cancelled"):
        F.run(args, tmp_path, outputs=[out], cancelled=ev.is_set)
    assert not out.exists()
    with pytest.raises(MediaError, match="timed out"):
        F.run(args, tmp_path, outputs=[out], timeout=0.5)
    assert not out.exists()


def test_decode_and_encode_frames(tmp_path):
    frames = [np.full((64, 96, 3), i * 20, np.uint8) for i in range(10)]
    out = F.encode_frames(frames, tmp_path / "e.mp4", tmp_path, size=(96, 64), fps=10)
    info = F.probe(out, tmp_path)
    assert (info.width, info.height) == (96, 64) and abs(info.duration_s - 1.0) < 0.05
    got = list(F.decode_frames(out, tmp_path, (48, 32)))
    assert len(got) == 10 and got[0].shape == (32, 48, 3)
    assert got[-1].mean() > got[0].mean() + 100
    with pytest.raises(MediaError):
        F.encode_frames([np.zeros((10, 10, 3), np.uint8)], tmp_path / "bad.mp4", tmp_path, size=(96, 64), fps=10)
    assert not (tmp_path / "bad.mp4").exists()
    kfs = F.keyframe_times(out, tmp_path)
    assert kfs and kfs[0] == 0.0


# ------------------------------------------------------------------ review H2: demuxers, links, sandbox
def _weak_sandbox():
    from jev.sandbox import Sandbox
    return Sandbox({"sandbox": {"isolation": "none"}})        # what Linux without bubblewrap, macOS, Windows get


def test_h2_sniff_container_refuses_playlists_and_scripts(tmp_path):
    clip = make_clip(tmp_path)
    assert F.sniff_container(clip)[0] == "mov"
    for name, body in (("c.mp4", "ffconcat version 1.0\nfile /etc/passwd\n"),
                       ("p.mp4", "#EXTM3U\n#EXTINF:1,\nfile:///etc/passwd\n"),
                       ("s.mp4", "v=0\no=- 0 0 IN IP4 127.0.0.1\n"), ("x.wav", "data:,hello")):
        (tmp_path / name).write_text(body)
        with pytest.raises(MediaError, match="not a recognised media file"):
            F.sniff_container(tmp_path / name)
        with pytest.raises(MediaError, match="not a recognised media file"):
            F.probe(name, tmp_path)


def test_h2_callers_cannot_pick_dangerous_demuxers(tmp_path):
    clip = make_clip(tmp_path, audio=False)
    out = tmp_path / "o.mp4"
    for fmt in ("hls", "concat", "image2", "data", "ffmetadata"):
        target = clip if fmt != "concat" else tmp_path / "l.txt"
        if fmt == "concat":
            target.write_text(f"file '{clip}'\n")             # absolute path: not a plain name next to the list
        with pytest.raises(MediaError):
            F.run(["-f", fmt, "-i", str(target), "-c", "copy", str(out)], tmp_path, outputs=[out])
    (tmp_path / "l.txt").write_text("file 'c.mp4'\nfile 'c.mp4'\n")     # JEV's own lists still work
    F.run(["-f", "concat", "-safe", "1", "-i", "l.txt", "-c", "copy", str(out)], tmp_path, outputs=[out], cwd=tmp_path)
    assert F.probe(out, tmp_path).duration_s > 3


def test_h2_weak_sandbox_refuses_links_fifos_and_hard_links(tmp_path):
    import os
    clip = make_clip(tmp_path, audio=False)
    os.symlink("c.mp4", tmp_path / "link.mp4")
    os.link(clip, tmp_path / "hard.mp4")
    os.mkfifo(tmp_path / "fifo.mp4")
    with F.use_sandbox(_weak_sandbox()):
        for name, why in (("link.mp4", "symbolic link"), ("hard.mp4", "hard link"), ("fifo.mp4", "regular file")):
            with pytest.raises(MediaError, match=why):
                F.probe(name, tmp_path)
            with pytest.raises(MediaError, match=why):
                list(F.decode_frames(name, tmp_path, (32, 32)))
    from jev.sandbox import Sandbox
    strong = Sandbox({"sandbox": {"isolation": "auto"}})
    if strong.capabilities()["fs_isolated"]:
        with F.use_sandbox(strong):                    # a link inside the workspace is fine under bubblewrap
            assert F.probe("link.mp4", tmp_path).width == 160
        with pytest.raises(MediaError, match="regular file"):
            F.probe("fifo.mp4", tmp_path)


def test_h2_media_pipeline_works_without_bwrap(tmp_path):
    """Simulates a machine without bubblewrap: FFmpeg still runs (in the weak sandbox) end to end."""
    with F.use_sandbox(_weak_sandbox()):
        clip = make_clip(tmp_path, seconds=1.0)
        info = F.probe(clip, tmp_path)
        assert info.has_video and info.has_audio
        frames = list(F.decode_frames(clip, tmp_path, (32, 16)))
        assert len(frames) == 12
        assert F.keyframe_times(clip, tmp_path)[0] == 0.0
        out = F.encode_frames(frames, tmp_path / "e.mp4", tmp_path, size=(32, 16), fps=12)
        assert F.probe(out, tmp_path).width == 32
