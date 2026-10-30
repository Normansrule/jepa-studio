"""Narrated explainer videos.

    script -> scenes [{narration, visual, min_seconds}]
           -> speech per scene (jev.media.tts) -> scene length = max(min_seconds, speech + padding)
           -> one video segment per scene (slide or still with optional Ken Burns pan/zoom, or a clip)
           -> crossfades (FFmpeg xfade) -> narration track (+ optional background music you provide,
              ducked under the voice) -> captions: SRT always, burned in with libass when FFmpeg has
              it, else drawn with Pillow and overlaid
           -> H.264 MP4 (yuv420p, +faststart) + captions .srt + storyboard .json

A visual is one of:
    {kind: "slide", slide: {...jev.media.slides spec...}}     (or just a slide spec with a `type`)
    {kind: "image", path, fit?: fit|cover}                     a still in the workspace
    {kind: "video", path, start?, end?}                        a clip in the workspace (<= 60 s; its audio is not used)
    {kind: "render", path}                                     a science render (image or video by extension)
Paths are workspace-relative.
"""
from __future__ import annotations

import json
import math
import re
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from . import ffmpeg as F
from . import tts
from jev import wsfs

from .common import Limits, MediaError, even, font, font_file, inside, open_image_safely, out_inside
from .images import fit_image, line_height, rgb, theme, wrap
from .slides import render_slide

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXT = {".mp4", ".mov", ".webm", ".mkv", ".avi", ".m4v", ".gif"}
AUDIO_RATE = 48000
MAX_SCENES = 80
CAPTION_LINE_CHARS = 42


@dataclass
class Cue:
    start: float
    end: float
    text: str


@dataclass
class SceneResult:
    index: int
    start: float
    duration: float
    narration: str
    visual: dict
    speech_start: float
    speech_seconds: float
    tts_backend: str
    silent: bool
    cues: list[Cue] = field(default_factory=list)


# ----------------------------------------------------------------------------- captions

def _fmt_srt(t: float) -> str:
    ms = int(round(max(0.0, t) * 1000))
    h, ms = divmod(ms, 3_600_000)
    m, ms = divmod(ms, 60_000)
    s, ms = divmod(ms, 1000)
    return f"{h:02d}:{m:02d}:{s:02d},{ms:03d}"


def split_caption_chunks(text: str, max_chars: int = 2 * CAPTION_LINE_CHARS) -> list[str]:
    """Sentence-aware chunks of at most ~max_chars (two caption lines)."""
    sentences = [s.strip() for s in re.split(r"(?<=[.!?;:])\s+", text.strip()) if s.strip()]
    chunks: list[str] = []
    for s in sentences:
        words, cur = s.split(), ""
        for w in words:
            if cur and len(cur) + 1 + len(w) > max_chars:
                chunks.append(cur)
                cur = w
            else:
                cur = f"{cur} {w}".strip()
        if cur:
            chunks.append(cur)
    # merge a tiny tail into the previous chunk when it fits
    out: list[str] = []
    for c in chunks:
        if out and len(c) < 14 and len(out[-1]) + 1 + len(c) <= max_chars + 10:
            out[-1] = f"{out[-1]} {c}"
        else:
            out.append(c)
    return out


def caption_lines(chunk: str, width: int = CAPTION_LINE_CHARS) -> list[str]:
    """One line if it fits, else two lines split at the space nearest the middle (balanced)."""
    chunk = " ".join(chunk.split())
    if len(chunk) <= width:
        return [chunk]
    spaces = [i for i, ch in enumerate(chunk) if ch == " "]
    if not spaces:
        return [chunk]
    mid = len(chunk) / 2
    cut = min(spaces, key=lambda i: (max(i, len(chunk) - i - 1) > width, abs(i - mid)))
    return [chunk[:cut], chunk[cut + 1:]]


def make_cues(text: str, start: float, seconds: float) -> list[Cue]:
    """Time each chunk in proportion to its length across the speech interval."""
    chunks = split_caption_chunks(text)
    if not chunks:
        return []
    weights = np.array([len(c) + 6 for c in chunks], float)
    edges = start + seconds * np.concatenate([[0.0], np.cumsum(weights) / weights.sum()])
    return [Cue(float(edges[i]), float(edges[i + 1]), chunks[i]) for i in range(len(chunks))]


def write_srt(cues: list[Cue], path: Path, root: Path | None = None) -> Path:
    blocks = []
    for i, c in enumerate(cues, 1):
        blocks.append(f"{i}\n{_fmt_srt(c.start)} --> {_fmt_srt(c.end)}\n" + "\n".join(caption_lines(c.text)) + "\n")
    wsfs.write_text(root if root is not None else Path(path).parent, path, "\n".join(blocks))
    return path


def parse_srt(text: str) -> list[Cue]:
    cues = []
    for block in re.split(r"\n\s*\n", text.strip()):
        m = re.search(r"(\d+):(\d+):(\d+),(\d+)\s*-->\s*(\d+):(\d+):(\d+),(\d+)", block)
        if not m:
            continue
        g = [int(x) for x in m.groups()]
        a = g[0] * 3600 + g[1] * 60 + g[2] + g[3] / 1000
        b = g[4] * 3600 + g[5] * 60 + g[6] + g[7] / 1000
        body = block[m.end():].strip()
        cues.append(Cue(a, b, body))
    return cues


def caption_image(text: str, size: tuple[int, int], th="dark") -> Image.Image:
    """A caption box (RGBA) for the Pillow burn-in path, sized for frame `size`."""
    t = theme(th)
    W, H = size
    f = font("bold", max(12, int(H * 0.042)))
    probe = ImageDraw.Draw(Image.new("RGBA", (1, 1)))
    lines = wrap(probe, text, f, W * 0.8)[:3]
    lh = line_height(f)
    tw = max(probe.textlength(ln, font=f) for ln in lines)
    pad_x, pad_y = int(H * 0.022), int(H * 0.012)
    bw, bh = int(tw + 2 * pad_x), int(lh * len(lines) + 2 * pad_y)
    im = Image.new("RGBA", (bw, bh), (0, 0, 0, 0))
    d = ImageDraw.Draw(im)
    d.rounded_rectangle((0, 0, bw - 1, bh - 1), radius=int(H * 0.012), fill=(8, 10, 16, 175))
    y = pad_y
    for ln in lines:
        d.text((bw / 2, y), ln, font=f, fill=rgb("#ffffff") + (255,), anchor="ma")
        y += lh
    _ = t
    return im


# ----------------------------------------------------------------------------- visuals

def _normalize_visual(v: dict | None) -> dict:
    if not v:
        raise MediaError("every scene needs a visual")
    if "kind" not in v:
        if "type" in v:
            return {"kind": "slide", "slide": v}
        if "slide" in v:
            return {"kind": "slide", "slide": v["slide"]}
        raise MediaError("visual needs kind: slide | image | video | render")
    kind = v["kind"]
    if kind == "render":
        ext = Path(str(v.get("path", ""))).suffix.lower()
        kind = "video" if ext in VIDEO_EXT - {".gif"} else "image"
        v = {**v, "kind": kind, "from_render": True}
    if kind not in ("slide", "image", "video"):
        raise MediaError(f"unknown visual kind {kind!r}")
    if kind in ("image", "video") and not v.get("path"):
        raise MediaError(f"{kind} visual needs a workspace path")
    return v


def _kb_frames(still: Image.Image, size: tuple[int, int], n: int, zoom: float, seed: int):
    """Ken Burns: a slow zoom with a gentle pan, sub-pixel smooth (Pillow box resize)."""
    W, H = size
    rng = np.random.default_rng(seed)
    direction = rng.choice([-1, 1], size=2) * rng.uniform(0.3, 1.0, size=2)
    zoom_in = bool(rng.integers(0, 2))
    src = still.convert("RGB")
    for k in range(n):
        p = k / max(1, n - 1)
        e = p * p * (3 - 2 * p)                               # smoothstep
        z = 1 + zoom * (e if zoom_in else 1 - e)
        cw, ch = W / z, H / z
        cx = W / 2 + direction[0] * (W - cw) / 2 * (2 * e - 1)
        cy = H / 2 + direction[1] * (H - ch) / 2 * (2 * e - 1)
        box = (cx - cw / 2, cy - ch / 2, cx + cw / 2, cy + ch / 2)
        yield np.asarray(src.resize((W, H), Image.Resampling.BICUBIC, box=box))


def _still_segment(png: Path, seg: Path, ws: Path, size, fps, dur, *, kb: bool, kb_zoom: float, seed: int,
                   limits: Limits, cancelled) -> None:
    n = max(1, int(round(dur * fps)))
    if kb:
        with wsfs.open_file(wsfs.root_of(png, ws), png, "rb") as fh:
            im = Image.open(fh)
            im.load()
        F.encode_frames(_kb_frames(im, size, n, kb_zoom, seed), seg, ws, size=size, fps=fps, total_frames=n,
                        crf=14, preset="veryfast", cancelled=cancelled, timeout=limits.ffmpeg_timeout_s,
                        label="ken burns")
        return
    F.run(["-loop", "1", "-framerate", f"{fps:g}", "-i", str(png), "-frames:v", str(n),
           "-vf", f"scale={size[0]}:{size[1]},format=yuv420p", *F.h264_args(14, "veryfast"),
           "-tune", "stillimage", "-r", f"{fps:g}", str(seg)],
          ws, outputs=[seg], cancelled=cancelled, timeout=limits.ffmpeg_timeout_s, label="still segment")


def _clip_segment(src: Path, seg: Path, ws: Path, size, fps, dur, start: float, end: float | None,
                  bg: str, limits: Limits, cancelled) -> dict:
    info = F.probe(src, ws, limits, kind="clip")
    if not info.has_video:
        raise MediaError(f"{src.name} has no video stream")
    start = max(0.0, float(start or 0))
    end = min(info.duration_s, float(end)) if end else info.duration_s
    if end <= start:
        raise MediaError(f"{src.name}: empty range {start}..{end}")
    W, H = size
    n = max(1, int(round(dur * fps)))
    c = bg.lstrip("#")
    vf = (f"scale={W}:{H}:force_original_aspect_ratio=decrease,pad={W}:{H}:(ow-iw)/2:(oh-ih)/2:color=0x{c},"
          f"setsar=1,fps={fps:g},tpad=stop_mode=clone:stop_duration={dur:.3f},format=yuv420p")
    F.run(["-ss", f"{start:.3f}", "-t", f"{end - start:.3f}", "-i", str(src), "-an", "-vf", vf,
           "-frames:v", str(n), *F.h264_args(14, "veryfast"), str(seg)],
          ws, outputs=[seg], cancelled=cancelled, timeout=limits.ffmpeg_timeout_s, label="clip segment")
    return {"source_seconds": end - start, "probe": info.to_json()}


# ----------------------------------------------------------------------------- audio

def _to_pcm(src: Path, dst: Path, ws: Path, limits: Limits, seconds: float | None = None,
            loop: bool = False) -> np.ndarray:
    args = (["-stream_loop", "-1"] if loop else []) + ["-i", str(src), "-vn", "-ac", "1", "-ar", str(AUDIO_RATE)]
    if seconds:
        args += ["-t", f"{seconds:.3f}"]
    F.run(args + ["-c:a", "pcm_s16le", str(dst)], ws, outputs=[dst], timeout=limits.ffmpeg_timeout_s,
          label="audio convert")
    with wsfs.open_file(wsfs.root_of(dst, ws), dst, "rb") as fh, wave.open(fh, "rb") as w:
        return np.frombuffer(w.readframes(w.getnframes()), np.int16).astype(np.float32) / 32768.0


def _write_wav(path: Path, x: np.ndarray) -> None:
    pcm = (np.clip(x, -1, 1) * 32767).astype(np.int16)
    with wsfs.atomic_writer(wsfs.root_of(path), path) as fh, wave.open(fh, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(AUDIO_RATE)
        w.writeframes(pcm.tobytes())


def _music_bed(music: np.ndarray, voice: np.ndarray, volume: float) -> np.ndarray:
    """Music at `volume`, ducked to 35% of that while the narration is speaking, 1 s fades."""
    n = len(voice)
    m = np.resize(music, n) if len(music) else np.zeros(n, np.float32)
    win = int(0.25 * AUDIO_RATE)
    env = np.abs(voice)
    if win > 1 and n > win:
        env = np.convolve(env, np.ones(win) / win, mode="same")
    speaking = env > 0.01
    gain = np.where(speaking, 0.35, 1.0).astype(np.float32)
    sm = int(0.3 * AUDIO_RATE)
    if sm > 1 and n > sm:
        gain = np.convolve(gain, np.ones(sm) / sm, mode="same").astype(np.float32)
    fade = min(n // 2, AUDIO_RATE)
    ramp = np.ones(n, np.float32)
    if fade > 0:
        ramp[:fade] = np.linspace(0, 1, fade)
        ramp[-fade:] = np.linspace(1, 0, fade)
    return m * volume * gain * ramp


# ----------------------------------------------------------------------------- estimate

def estimate_render_seconds(scenes: list[dict], size=(1920, 1080), fps: float = 30, ken_burns: str | bool = "auto",
                            tts_backend: str = "auto") -> float:
    """Rough wall-clock estimate on a 2-core laptop CPU (deliberately pessimistic)."""
    px = size[0] * size[1] / (1920 * 1080)
    total = 0.0
    secs = 0.0
    for s in scenes:
        narr = str(s.get("narration", ""))
        d = max(float(s.get("min_seconds", 0) or 0), tts.estimate_seconds(narr) + 1.0 if narr else 2.0)
        secs += d
        frames = d * fps
        v = s.get("visual") or {}
        kind = v.get("kind", "slide") if isinstance(v, dict) else "slide"
        kb = ken_burns is True or (ken_burns == "auto" and kind in ("image", "render"))
        total += 1.0 + 0.4 + len(narr) * 0.003                  # slide render + speech
        total += frames * px * (0.035 if kb else 0.004)          # segment
    frames_all = secs * fps
    total += frames_all * px * 0.075                              # final encode with crossfades/captions
    return total


# ----------------------------------------------------------------------------- assembly

def make_explainer(scenes: list[dict], out: str | Path, workspace: str | Path, *, size=(1920, 1080),
                   fps: float = 30, theme_name: str = "dark", tts_backend: str = "auto", voice: str | None = None,
                   rate: float = 1.0, ken_burns: str | bool = "auto", ken_burns_zoom: float = 0.08,
                   crossfade: float = 0.5, captions: str = "burn", music: str | None = None,
                   music_volume: float = 0.12, crf: int = 20, preset: str = "medium", seed: int = 0,
                   settings: dict | None = None, limits: Limits | None = None, keep_work: bool = False,
                   progress: Callable[[float, str], None] | None = None,
                   cancelled: Callable[[], bool] | None = None) -> dict:
    """Build an explainer MP4 from scenes. Returns paths, timing, the speech backends used and
    the caption method. See the module docstring for the scene format. Intermediate files are made
    in backend-owned scratch, never in the workspace."""
    with wsfs.ensure_scratch():
        return _make_explainer(scenes, out, workspace, size=size, fps=fps, theme_name=theme_name,
                               tts_backend=tts_backend, voice=voice, rate=rate, ken_burns=ken_burns,
                               ken_burns_zoom=ken_burns_zoom, crossfade=crossfade, captions=captions, music=music,
                               music_volume=music_volume, crf=crf, preset=preset, seed=seed, settings=settings,
                               limits=limits, keep_work=keep_work, progress=progress, cancelled=cancelled)


def _make_explainer(scenes, out, workspace, *, size, fps, theme_name, tts_backend, voice, rate, ken_burns,
                    ken_burns_zoom, crossfade, captions, music, music_volume, crf, preset, seed, settings, limits,
                    keep_work, progress, cancelled) -> dict:
    t_start = time.monotonic()
    ws = Path(workspace).resolve()
    limits = limits or Limits.from_settings(settings)
    out_mp4 = out_inside(out, ws)
    if out_mp4.suffix.lower() != ".mp4":
        out_mp4 = out_mp4.with_suffix(".mp4")
    wsfs.makedirs(ws, out_mp4.parent)
    if not scenes:
        raise MediaError("an explainer needs at least one scene")
    if len(scenes) > MAX_SCENES:
        raise MediaError(f"at most {MAX_SCENES} scenes")
    W, H = even(size[0]), even(size[1])
    size = (W, H)
    if W * H > limits.max_output_pixels:
        raise MediaError(f"{W}x{H} is above the output cap")
    if not (5 <= fps <= 60):
        raise MediaError("fps must be within 5..60")
    if captions not in ("burn", "srt", "none"):
        raise MediaError("captions must be burn, srt or none")
    th = theme(theme_name)
    prog = progress or (lambda f, m: None)
    is_cancelled = cancelled or (lambda: False)
    work = wsfs.new_scratch("explainer-")
    n = len(scenes)
    visuals = [_normalize_visual(s.get("visual")) for s in scenes]
    for v in visuals:
        if v["kind"] in ("image", "video"):
            inside(v["path"], ws)                       # refuse escapes before any work is done

    # 1. speech
    speech = []
    for i, s in enumerate(scenes):
        if is_cancelled():
            raise MediaError("cancelled")
        narr = str(s.get("narration", "") or "").strip()
        if narr:
            r = tts.synthesize(narr, work / f"voice_{i:02d}.wav", work, backend=tts_backend, voice=voice, rate=rate,
                               settings=settings)
        else:
            r = None
        speech.append(r)
        prog(0.25 * (i + 1) / n, f"speech {i + 1}/{n}")

    # 2. timing: scene i spans [start_i, start_i + dur_i]; neighbours overlap by `c` (the crossfade)
    c = max(0.0, float(crossfade)) if n > 1 else 0.0
    lead, tail = 0.25 + c * 0.6, 0.45 + c
    durs = []
    for i, s in enumerate(scenes):
        sp = speech[i].duration_s if speech[i] else 0.0
        d = max(float(s.get("min_seconds", 0) or 0), (lead + sp + tail) if sp else 0.0)
        if visuals[i]["kind"] == "video" and d == 0:
            st, en = float(visuals[i].get("start") or 0), visuals[i].get("end")
            d = (float(en) - st) if en else 0.0
            if d == 0:
                d = F.probe(visuals[i]["path"], ws, limits, kind="clip").duration_s - st
        durs.append(max(d, 1.0))
    if n > 1:
        c = min(c, min(durs) / 3)
    starts = [sum(durs[:i]) - i * c for i in range(n)]
    total = starts[-1] + durs[-1]
    if total > limits.max_output_seconds:
        raise MediaError(f"the video would be {total:.0f} s long; the cap is {limits.max_output_seconds:g} s")

    # 3. one segment per scene
    results: list[SceneResult] = []
    segs = []
    for i, v in enumerate(visuals):
        if is_cancelled():
            raise MediaError("cancelled")
        seg = work / f"seg_{i:02d}.mp4"
        extra: dict = {}
        if v["kind"] == "slide":
            png = work / f"slide_{i:02d}.png"
            spec = dict(v["slide"])
            if captions == "burn" and spec.get("type") != "title":
                spec.setdefault("safe_bottom", 0.1)        # keep slide content clear of the captions
            wsfs.save_image(render_slide(spec, size, th, ws, limits), work, png)
            kb = ken_burns is True
            _still_segment(png, seg, ws, size, fps, durs[i], kb=kb, kb_zoom=ken_burns_zoom * 0.5, seed=seed + i,
                           limits=limits, cancelled=is_cancelled)
        elif v["kind"] == "image":
            src = open_image_safely(inside(v["path"], ws), limits, root=ws)
            png = work / f"still_{i:02d}.png"
            wsfs.save_image(fit_image(src, size, v.get("fit", "fit"), th.bg), work, png)
            kb = ken_burns is True or ken_burns == "auto"
            _still_segment(png, seg, ws, size, fps, durs[i], kb=kb, kb_zoom=ken_burns_zoom, seed=seed + i,
                           limits=limits, cancelled=is_cancelled)
        else:
            extra = _clip_segment(inside(v["path"], ws), seg, ws, size, fps, durs[i], v.get("start", 0),
                                  v.get("end"), th.bg, limits, is_cancelled)
        segs.append(seg)
        sp = speech[i]
        sp_start = starts[i] + lead
        sres = SceneResult(i, starts[i], durs[i], str(scenes[i].get("narration", "") or ""),
                           {**v, **({"clip": extra} if extra else {})}, sp_start if sp else starts[i],
                           sp.duration_s if sp else 0.0, sp.backend if sp else "none", bool(sp and sp.silent))
        if sp:
            sres.cues = make_cues(sres.narration, sp_start, sp.duration_s)
        results.append(sres)
        prog(0.25 + 0.4 * (i + 1) / n, f"scene {i + 1}/{n}")

    # 4. audio: narration placed on the timeline (+ ducked music)
    nsamp = int(math.ceil(total * AUDIO_RATE))
    voice_track = np.zeros(nsamp, np.float32)
    for i, sp in enumerate(speech):
        if not sp:
            continue
        x = _to_pcm(sp.path, work / f"voice48_{i:02d}.wav", ws, limits)
        a = int(round(results[i].speech_start * AUDIO_RATE))
        b = min(nsamp, a + len(x))
        voice_track[a:b] += x[: b - a]
    mix = voice_track.copy()
    music_info = None
    if music:
        mp = inside(music, ws)
        F.probe(mp, ws, limits, kind="audio")
        m = _to_pcm(mp, work / "music48.wav", ws, limits, seconds=total + 0.5, loop=True)
        mix += _music_bed(m, voice_track, float(music_volume))
        music_info = {"path": mp.relative_to(ws).as_posix(), "volume": music_volume}
    peak = float(np.max(np.abs(mix))) if len(mix) else 0.0
    if peak > 0.98:
        mix *= 0.98 / peak
    audio = work / "audio.wav"
    _write_wav(audio, mix)

    # 5. captions
    cues = [cu for r in results for cu in r.cues]
    srt = out_mp4.with_suffix(".srt")
    write_srt(cues, srt, root=ws)
    write_srt(cues, work / "captions.srt", root=work)
    caption_mode = "none"
    if captions == "burn" and cues:
        caption_mode = "libass" if F.has_filter("subtitles") else "pillow"
    elif captions == "srt":
        caption_mode = "srt-only"

    # 6. final pass: crossfades + captions + audio -> H.264/AAC MP4
    args: list[str] = []
    for seg in segs:
        args += ["-i", seg.name]
    args += ["-i", audio.name]
    audio_idx = len(segs)
    graph = [f"[{i}:v]settb=AVTB,setpts=PTS-STARTPTS,fps={fps:g},format=yuv420p[v{i}]" for i in range(n)]
    last = "v0"
    for i in range(1, n):
        graph.append(f"[{last}][v{i}]xfade=transition=fade:duration={c:.3f}:offset={starts[i]:.3f}[x{i}]")
        last = f"x{i}"
    if caption_mode == "libass":
        fam = "DejaVu Sans" if "DejaVu" in (font_file("bold") or "") else "Arial"
        fs = 14 if H >= 700 else 13
        style = (f"FontName={fam},FontSize={fs},Bold=1,PrimaryColour=&H00FFFFFF,OutlineColour=&H50100A08,"
                 "BackColour=&H50100A08,BorderStyle=3,Outline=2,Shadow=0,MarginV=16,Alignment=2")
        graph.append(f"[{last}]subtitles=captions.srt:force_style='{style}'[vc]")
        last = "vc"
    elif caption_mode == "pillow":
        for k, cu in enumerate(cues):
            png = work / f"cap_{k:03d}.png"
            wsfs.save_image(caption_image(cu.text, size, th), work, png)
            args += ["-i", png.name]
            idx = audio_idx + 1 + k
            margin = int(H * 0.05)
            graph.append(f"[{last}][{idx}:v]overlay=x=(W-w)/2:y=H-h-{margin}:"
                         f"enable='between(t,{cu.start:.3f},{cu.end:.3f})'[c{k}]")
            last = f"c{k}"
    graph.append(f"[{last}]format=yuv420p[vout]")
    args += ["-filter_complex", ";".join(graph), "-map", "[vout]", "-map", f"{audio_idx}:a",
             *F.h264_args(crf, preset), "-r", f"{fps:g}", "-c:a", "aac", "-b:a", "160k", "-ar", str(AUDIO_RATE),
             "-t", f"{total:.3f}", "-movflags", "+faststart", str(out_mp4)]
    F.run(args, ws, outputs=[out_mp4], cwd=work, duration_s=total, timeout=limits.ffmpeg_timeout_s,
          cancelled=is_cancelled, progress=lambda f, m: prog(0.65 + 0.33 * f, "encoding"), label="final encode")

    backends = sorted({r.tts_backend for r in results if r.tts_backend != "none"})
    silent = any(r.silent for r in results)
    storyboard = {
        "format": "jev-storyboard/v1", "video": out_mp4.name, "captions": srt.name,
        "size": [W, H], "fps": fps, "duration_s": round(total, 3), "crossfade_s": round(c, 3),
        "theme": th.name, "caption_mode": caption_mode, "tts_backends": backends, "silent_narration": silent,
        "music": music_info, "render_seconds": round(time.monotonic() - t_start, 2),
        "scenes": [{"index": r.index, "start": round(r.start, 3), "duration": round(r.duration, 3),
                    "narration": r.narration, "visual": r.visual, "speech_start": round(r.speech_start, 3),
                    "speech_seconds": round(r.speech_seconds, 3), "tts_backend": r.tts_backend,
                    "captions": [{"start": round(cu.start, 3), "end": round(cu.end, 3), "text": cu.text}
                                 for cu in r.cues]} for r in results],
    }
    sb = out_mp4.with_suffix(".storyboard.json")
    wsfs.write_text(ws, sb, json.dumps(storyboard, indent=2, default=str))
    if not keep_work:
        wsfs.drop_scratch(work)
    prog(1.0, "done")
    return {"video": out_mp4, "srt": srt, "storyboard": sb, "duration_s": total, "caption_mode": caption_mode,
            "tts_backends": backends, "silent_narration": silent, "size": size, "fps": fps,
            "render_seconds": storyboard["render_seconds"],
            "notes": sorted({nt for sp in speech if sp for nt in sp.notes})}
