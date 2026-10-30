"""Editing your own footage: shots, search, highlights, cuts.

    analyze()          probe (caps checked before decoding) -> sample frames -> one feature vector
                       per sampled frame: a trained JEPA encoder from a jepa_studio run
                       (load_run_weights, hash-checked; embeddings L2-normalised) or a colour
                       histogram fallback (global 8x8x8 RGB + 2x2 grid of 4x4x4, Hellinger-normalised)
                       + motion energy (mean absolute frame difference)
    detect_shots()     cosine distance between consecutive samples; a cut is a local peak above an
                       adaptive threshold (rolling median + k x MAD, with an absolute floor)
    search()           "find scenes like this image / this moment": cosine similarity per sample,
                       best sample per shot, ranked
    select_highlights() a reel within a time budget: farthest-point sampling over shot embeddings
                       (diversity) weighted by motion energy
    render_cuts()      cut list -> FFmpeg trims (stream copy when every cut starts on a keyframe and
                       nothing is drawn on top, else re-encode) -> concat, optional title card and
                       per-clip captions
    export_cut_list()  JSON and a CMX3600-style EDL that NLEs (Resolve, Premiere) import

Limits: hard cuts are found reliably; slow dissolves and fades may be missed or found late; the
histogram fallback sees colour and layout, not meaning (a trained encoder does better on
semantically similar but differently coloured shots). Search takes an image or a moment of the
video, not text (there is no local text-image model here).
"""
from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image

from . import ffmpeg as F
from .common import Limits, MediaError, even, inside, open_image_safely

ANALYSIS_WIDTH = 160
CACHE_VERSION = 2


@dataclass
class Shot:
    index: int
    start: float
    end: float
    start_frame: int
    end_frame: int
    score: float               # cut strength at the start boundary (0 for the first shot)

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_json(self) -> dict:
        return {"index": self.index, "start": round(self.start, 3), "end": round(self.end, 3),
                "start_frame": self.start_frame, "end_frame": self.end_frame,
                "duration": round(self.duration, 3), "score": round(self.score, 4)}


@dataclass
class Analysis:
    path: Path
    info: F.MediaInfo
    times: np.ndarray            # (N,) seconds of each sampled frame
    features: np.ndarray         # (N, D) L2-normalised
    motion: np.ndarray           # (N,) mean absolute difference to the previous sample, 0..1
    method: str                  # "histogram" | "jepa:<run>"
    sample_fps: float
    analysis_size: tuple[int, int]
    shots: list[Shot] = field(default_factory=list)
    distances: np.ndarray | None = None

    @property
    def fps(self) -> float:
        return self.info.fps or 25.0

    def to_json(self) -> dict:
        return {"path": self.path.name, "duration_s": self.info.duration_s, "fps": self.fps,
                "samples": int(len(self.times)), "sample_fps": round(self.sample_fps, 3), "method": self.method,
                "shots": [s.to_json() for s in self.shots]}


# ----------------------------------------------------------------------------- features

def histogram_features(frames: np.ndarray) -> np.ndarray:
    """(N, H, W, 3) uint8 -> (N, 768) float32: global 8^3 RGB histogram + a 2x2 grid of 4^3
    histograms (layout), square-rooted (Hellinger) and L2-normalised."""
    frames = np.asarray(frames, np.uint8)
    n, h, w, _ = frames.shape
    q8 = (frames >> 5).astype(np.int32)
    idx8 = (q8[..., 0] * 64 + q8[..., 1] * 8 + q8[..., 2]).reshape(n, -1)
    off = (np.arange(n) * 512)[:, None]
    g = np.bincount((idx8 + off).ravel(), minlength=n * 512).reshape(n, 512).astype(np.float32)
    g /= h * w
    q4 = (frames >> 6).astype(np.int32)
    idx4 = q4[..., 0] * 16 + q4[..., 1] * 4 + q4[..., 2]
    cells = []
    for ys in (slice(0, h // 2), slice(h // 2, h)):
        for xs in (slice(0, w // 2), slice(w // 2, w)):
            c = idx4[:, ys, xs].reshape(n, -1)
            o = (np.arange(n) * 64)[:, None]
            hist = np.bincount((c + o).ravel(), minlength=n * 64).reshape(n, 64).astype(np.float32)
            cells.append(hist / max(1, c.shape[1]))
    feat = np.sqrt(np.concatenate([g] + [0.5 * c for c in cells], axis=1))
    return (feat / (np.linalg.norm(feat, axis=1, keepdims=True) + 1e-8)).astype(np.float32)


class JEPAFeaturizer:
    """Frame embeddings from a trained jepa_studio run (<run>/config.json + encoder.safetensors).
    The weights are checked against the hash the run recorded (jepa_studio.weights.load_run_weights)."""

    def __init__(self, run_dir: str | Path):
        import torch

        from jepa_studio.models import build_encoder
        from jepa_studio.weights import WeightError, load_run_weights

        d = Path(run_dir)
        cfg_path = d / "config.json"
        if not cfg_path.is_file():
            raise MediaError(f"{d.name}: not a jepa_studio run (no config.json)")
        cfg = json.loads(cfg_path.read_text())
        arch = cfg.get("model", {}).get("arch")
        if arch == "series-conv":
            raise MediaError(f"{d.name} is a time-series encoder; footage needs an image or video encoder")
        if int(cfg.get("data", {}).get("channels", 3)) != 3:
            raise MediaError(f"{d.name} expects {cfg['data']['channels']} channels; footage is RGB")
        enc = build_encoder(cfg)
        try:
            self.verified = load_run_weights(enc, d)
        except (WeightError, FileNotFoundError) as e:
            raise MediaError(f"{d.name}: cannot load encoder weights ({e})") from e
        enc.eval()
        bb = enc.backbone
        self.net = bb.frame if arch == "video-convnet" else bb
        self.size = int(cfg.get("data", {}).get("image_size", 32))
        self.name = d.name
        self.torch = torch

    def __call__(self, frames: np.ndarray, batch: int = 256) -> np.ndarray:
        torch = self.torch
        import torch.nn.functional as TF

        out = []
        with torch.no_grad():
            for i in range(0, len(frames), batch):
                x = torch.from_numpy(np.ascontiguousarray(frames[i:i + batch])).float().div_(255.0)
                x = x.permute(0, 3, 1, 2)
                x = TF.interpolate(x, size=(self.size, self.size), mode="bilinear", align_corners=False,
                                   antialias=True)
                e = self.net(x)
                out.append(TF.normalize(e, dim=1).cpu().numpy())
        return np.concatenate(out).astype(np.float32)


def motion_energy(frames: np.ndarray) -> np.ndarray:
    g = np.asarray(frames, np.float32).mean(axis=3)
    m = np.zeros(len(g), np.float32)
    if len(g) > 1:
        m[1:] = np.abs(np.diff(g, axis=0)).mean(axis=(1, 2)) / 255.0
    return m


def featurize_image(im: Image.Image, analysis: Analysis, featurizer=None) -> np.ndarray:
    w, h = analysis.analysis_size
    arr = np.asarray(im.convert("RGB").resize((w, h), Image.Resampling.BOX))[None]
    return (featurizer(arr) if featurizer is not None else histogram_features(arr))[0]


# ----------------------------------------------------------------------------- analysis

def _cache_key(p: Path, method: str, sample_fps: float | None, width: int) -> str:
    st = p.stat()
    raw = f"{CACHE_VERSION}|{p}|{st.st_size}|{st.st_mtime_ns}|{method}|{sample_fps}|{width}"
    return hashlib.sha256(raw.encode()).hexdigest()[:24]


def analyze(path: str | Path, workspace: str | Path, *, encoder_run: str | Path | None = None,
            sample_fps: float | None = None, limits: Limits | None = None, cache: bool = True,
            sensitivity: float = 6.0, min_shot_s: float = 0.4,
            progress: Callable[[float, str], None] | None = None,
            cancelled: Callable[[], bool] | None = None) -> Analysis:
    ws = Path(workspace).resolve()
    limits = limits or Limits()
    p = inside(path, ws)
    info = F.probe(p, ws, limits, kind="footage")
    if not info.has_video:
        raise MediaError(f"{p.name} has no video stream")
    fps = info.fps or 25.0
    if sample_fps is None:
        sample_fps = fps if info.duration_s * fps <= limits.max_sample_frames else limits.max_sample_frames / info.duration_s
    sample_fps = float(min(fps, max(0.1, sample_fps)))
    w = min(ANALYSIS_WIDTH, even(info.width))
    h = even(w * info.height / info.width)
    featurizer = JEPAFeaturizer(encoder_run) if encoder_run else None
    method = f"jepa:{featurizer.name}" if featurizer else "histogram"
    cache_file = None
    if cache:
        cdir = ws / ".jev_media_cache"
        cdir.mkdir(exist_ok=True)
        cache_file = cdir / (_cache_key(p, method + (str(Path(encoder_run).resolve()) if encoder_run else ""),
                                        sample_fps, w) + ".npz")
        if cache_file.is_file():
            try:
                z = np.load(cache_file, allow_pickle=False)
                a = Analysis(p, info, z["times"], z["features"], z["motion"], method, sample_fps, (w, h))
                a.shots, a.distances = detect_shots(a, sensitivity=sensitivity, min_shot_s=min_shot_s)
                return a
            except Exception:  # noqa: BLE001 - stale or damaged cache: recompute
                pass
    native = abs(sample_fps - fps) < 1e-3
    expected = int(math.ceil(info.duration_s * sample_fps)) + 2
    frames = []
    for k, fr in enumerate(F.decode_frames(p, ws, (w, h), fps=None if native else sample_fps,
                                           max_frames=limits.max_sample_frames, cancelled=cancelled,
                                           timeout=limits.ffmpeg_timeout_s)):
        frames.append(fr)
        if progress and k % 50 == 0:
            progress(0.7 * min(1.0, k / expected), "sampling frames")
    arr = np.stack(frames)
    times = np.arange(len(arr), dtype=np.float64) / sample_fps
    feats = featurizer(arr) if featurizer else histogram_features(arr)
    if progress:
        progress(0.9, "features")
    motion = motion_energy(arr)
    if cache_file is not None:
        np.savez(cache_file, times=times, features=feats, motion=motion)
    a = Analysis(p, info, times, feats, motion, method, sample_fps, (w, h))
    a.shots, a.distances = detect_shots(a, sensitivity=sensitivity, min_shot_s=min_shot_s)
    if progress:
        progress(1.0, "analysed")
    return a


def detect_shots(a: Analysis, *, sensitivity: float = 6.0, min_shot_s: float = 0.4,
                 floor: float | None = None) -> tuple[list[Shot], np.ndarray]:
    """Cuts = local maxima of the consecutive cosine distance above max(floor, median + k*MAD)
    in a ~2 s window around each sample."""
    f = a.features
    n = len(f)
    d = np.zeros(n, np.float32)
    if n > 1:
        d[1:] = 1.0 - np.clip(np.sum(f[1:] * f[:-1], axis=1), -1, 1)
    if floor is None:
        floor = 0.06 if a.method == "histogram" else 0.03
    half = max(3, int(round(a.sample_fps)))
    m = max(1, int(round(min_shot_s * a.sample_fps)))
    cuts = []
    for t in range(1, n):
        lo, hi = max(1, t - half), min(n, t + half + 1)
        win = np.concatenate([d[lo:t], d[t + 1:hi]])
        if len(win) == 0:
            continue
        med = float(np.median(win))
        mad = float(np.median(np.abs(win - med)))
        thr = max(floor, med + sensitivity * max(mad, 0.004))
        if d[t] <= thr:
            continue
        if d[t] < d[max(1, t - m):min(n, t + m + 1)].max():
            continue
        if cuts and t - cuts[-1] < m:
            continue
        cuts.append(t)
    fps = a.fps
    bounds = [0] + cuts
    shots = []
    for i, b in enumerate(bounds):
        start = float(a.times[b])
        end = float(a.times[bounds[i + 1]]) if i + 1 < len(bounds) else float(a.info.duration_s)
        shots.append(Shot(i, start, end, int(round(start * fps)), int(round(end * fps)) - 1,
                          float(d[b]) if b else 0.0))
    return shots, d


def shot_embeddings(a: Analysis) -> np.ndarray:
    out = []
    for s in a.shots:
        sel = (a.times >= s.start) & (a.times < s.end)
        v = a.features[sel].mean(0) if sel.any() else a.features[np.argmin(np.abs(a.times - s.start))]
        out.append(v / (np.linalg.norm(v) + 1e-8))
    return np.stack(out)


# ----------------------------------------------------------------------------- search

def search(a: Analysis, query: np.ndarray, top_k: int = 5) -> list[dict]:
    q = np.asarray(query, np.float32)
    q = q / (np.linalg.norm(q) + 1e-8)
    sims = a.features @ q
    ranked = []
    for s in a.shots:
        sel = np.where((a.times >= s.start) & (a.times < s.end))[0]
        if len(sel) == 0:
            continue
        j = sel[np.argmax(sims[sel])]
        ranked.append({**s.to_json(), "best_time": round(float(a.times[j]), 3), "similarity": round(float(sims[j]), 4)})
    ranked.sort(key=lambda r: -r["similarity"])
    return ranked[:max(1, top_k)]


def query_from_image(a: Analysis, path: str | Path, workspace: str | Path, encoder_run=None,
                     limits: Limits | None = None) -> np.ndarray:
    im = open_image_safely(inside(path, workspace), limits)
    fz = JEPAFeaturizer(encoder_run) if encoder_run else None
    return featurize_image(im, a, fz)


def query_from_time(a: Analysis, t: float) -> np.ndarray:
    return a.features[int(np.argmin(np.abs(a.times - float(t))))]


# ----------------------------------------------------------------------------- highlights

@dataclass
class Cut:
    start: float
    end: float
    label: str = ""
    reason: str = ""
    score: float = 0.0

    @property
    def duration(self) -> float:
        return self.end - self.start

    def to_json(self) -> dict:
        return {"start": round(self.start, 3), "end": round(self.end, 3), "duration": round(self.duration, 3),
                "label": self.label, "reason": self.reason, "score": round(self.score, 4)}


def select_highlights(a: Analysis, budget_s: float, *, min_clip_s: float = 1.0, max_clip_s: float = 4.0,
                      motion_weight: float = 0.5) -> list[Cut]:
    """Pick clips totalling <= budget_s: start from the highest-motion candidate, then repeatedly
    take the candidate farthest (cosine) from everything chosen, weighted by its motion energy."""
    if budget_s <= 0:
        raise MediaError("budget must be positive")
    min_clip_s = min(min_clip_s, budget_s)
    cands = []
    for s in a.shots:
        if s.duration < min(min_clip_s, 0.5):
            continue
        k = max(1, int(math.ceil(s.duration / max_clip_s)))
        step = s.duration / k
        for j in range(k):
            st = s.start + j * step
            en = min(s.end, st + min(max_clip_s, step))
            sel = (a.times >= st) & (a.times < en)
            if not sel.any():
                continue
            emb = a.features[sel].mean(0)
            mot = float(a.motion[sel].mean())
            cands.append({"shot": s.index, "start": st, "end": en, "emb": emb / (np.linalg.norm(emb) + 1e-8),
                          "motion": mot})
    if not cands:
        raise MediaError("no usable shots (all shorter than the minimum clip length)")
    mot = np.array([c["motion"] for c in cands])
    mnorm = (mot - mot.min()) / (mot.max() - mot.min() + 1e-9) if len(mot) > 1 else np.ones(1)
    E = np.stack([c["emb"] for c in cands])
    chosen: list[int] = []
    cuts: list[Cut] = []
    total = 0.0
    used_shots: set[int] = set()
    first = int(np.argmax(mnorm))
    while True:
        if not chosen:
            pick, why, val = first, "most motion", float(mnorm[first])
        else:
            dist = 1.0 - (E @ E[chosen].T).max(axis=1)
            val_all = dist * ((1 - motion_weight) + motion_weight * mnorm)
            val_all[chosen] = -np.inf
            # prefer shots not used yet
            for i, c in enumerate(cands):
                if c["shot"] in used_shots and np.isfinite(val_all[i]):
                    val_all[i] *= 0.5
            pick = int(np.argmax(val_all))
            if not np.isfinite(val_all[pick]):
                break
            val = float(val_all[pick])
            why = f"different from chosen clips (d={float(dist[pick]):.2f}), motion {float(mnorm[pick]):.2f}"
        c = cands[pick]
        length = c["end"] - c["start"]
        remaining = budget_s - total
        if remaining < min_clip_s - 1e-6:
            break
        if length > remaining:
            length = remaining
        chosen.append(pick)
        used_shots.add(c["shot"])
        cuts.append(Cut(c["start"], c["start"] + length, f"shot {c['shot'] + 1}", why, val))
        total += length
        if len(chosen) == len(cands):
            break
    cuts.sort(key=lambda x: x.start)
    merged: list[Cut] = []
    for c in cuts:                      # adjacent windows of the same shot become one clip
        if merged and merged[-1].label == c.label and abs(merged[-1].end - c.start) < 1e-6:
            merged[-1] = Cut(merged[-1].start, c.end, c.label, merged[-1].reason, max(merged[-1].score, c.score))
        else:
            merged.append(c)
    return merged


# ----------------------------------------------------------------------------- rendering cuts

def _tc(t: float, fps: int) -> str:
    f = int(round(t * fps))
    h, f = divmod(f, 3600 * fps)
    m, f = divmod(f, 60 * fps)
    s, f = divmod(f, fps)
    return f"{h:02d}:{m:02d}:{s:02d}:{f:02d}"


def export_cut_list(cuts: list[Cut], source: Path, json_out: Path, edl_out: Path | None, fps: float,
                    title: str = "JEV cut list") -> dict:
    rec = 0.0
    items = []
    for c in cuts:
        items.append({**c.to_json(), "record_start": round(rec, 3), "record_end": round(rec + c.duration, 3)})
        rec += c.duration
    doc = {"format": "jev-cutlist/v1", "source": source.name, "fps": fps, "duration_s": round(rec, 3), "cuts": items}
    json_out.write_text(json.dumps(doc, indent=2), encoding="utf-8")
    if edl_out is not None:
        ifps = max(1, int(round(fps)))
        lines = [f"TITLE: {title[:60]}", "FCM: NON-DROP FRAME", ""]
        rec = 3600.0                                 # record timecode starts at 01:00:00:00
        for i, c in enumerate(cuts, 1):
            lines.append(f"{i:03d}  AX       V     C        {_tc(c.start, ifps)} {_tc(c.end, ifps)} "
                         f"{_tc(rec, ifps)} {_tc(rec + c.duration, ifps)}")
            lines.append(f"* FROM CLIP NAME: {source.name}")
            if c.label:
                lines.append(f"* COMMENT: {c.label}")
            lines.append("")
            rec += c.duration
        edl_out.write_text("\n".join(lines), encoding="utf-8")
    return doc


def _on_keyframe(t: float, kfs: list[float], fps: float) -> bool:
    tol = 0.5 / (fps or 25.0)
    return any(abs(k - t) <= tol for k in kfs)


def render_cuts(src: str | Path, cuts: list[Cut], out: str | Path, workspace: str | Path, *, mode: str = "auto",
                title: str | None = None, captions: bool = False, limits: Limits | None = None, crf: int = 20,
                theme_name: str = "dark", progress: Callable[[float, str], None] | None = None,
                cancelled: Callable[[], bool] | None = None) -> dict:
    """Trim `cuts` from `src` and join them into `out` (MP4). mode: auto | copy | reencode.
    Stream copy is lossless and fast but only exact when every cut starts on a keyframe and
    nothing is drawn on top; auto falls back to re-encoding (frame-accurate) otherwise."""
    from .images import title_card
    from .video import Cue, caption_image, write_srt

    ws = Path(workspace).resolve()
    limits = limits or Limits()
    sp = inside(src, ws)
    o = inside(out, ws)
    if o.suffix.lower() != ".mp4":
        o = o.with_suffix(".mp4")
    o.parent.mkdir(parents=True, exist_ok=True)
    if mode not in ("auto", "copy", "reencode"):
        raise MediaError("mode must be auto, copy or reencode")
    if not cuts:
        raise MediaError("empty cut list")
    if len(cuts) > 500:
        raise MediaError("at most 500 cuts")
    info = F.probe(sp, ws, limits, kind="footage")
    if not info.has_video:
        raise MediaError(f"{sp.name} has no video stream")
    clean = []
    for c in cuts:
        st, en = max(0.0, float(c.start)), min(info.duration_s, float(c.end))
        if en - st < 0.04:
            raise MediaError(f"cut {c.start:.2f}-{c.end:.2f} s is empty or outside the {info.duration_s:.2f} s source")
        clean.append(Cut(st, en, c.label, c.reason, c.score))
    cuts = clean
    total = sum(c.duration for c in cuts)
    if total > limits.max_output_seconds:
        raise MediaError(f"the cut list is {total:.0f} s; the cap is {limits.max_output_seconds:g} s")
    drawn = bool(title) or captions
    used = mode
    notes = []
    if mode in ("auto", "copy"):
        kfs = F.keyframe_times(sp, ws)
        aligned = all(_on_keyframe(c.start, kfs, info.fps) for c in cuts)
        if aligned and not drawn:
            used = "copy"
        elif mode == "copy":
            why = "a title or captions are drawn" if drawn else "some cuts do not start on a keyframe"
            raise MediaError(f"stream copy is not exact here ({why}); use mode=reencode or auto")
        else:
            used = "reencode"
            notes.append("re-encoded: " + ("title/captions drawn" if drawn else "cuts not on keyframes"))
    work = o.parent / f".{o.stem}.work"
    work.mkdir(parents=True, exist_ok=True)
    prog = progress or (lambda f, m: None)
    parts = []
    W, H = even(info.width), even(info.height)
    fps = info.fps or 25.0
    has_audio = info.has_audio
    if title and used == "reencode":
        png = work / "title.png"
        title_card(title, "", (W, H), theme_name).save(png)
        part = work / "part_title.mp4"
        args = ["-loop", "1", "-framerate", f"{fps:g}", "-i", str(png)]
        if has_audio:
            args += ["-f", "lavfi", "-i", "anullsrc=r=48000:cl=stereo"]
        args += ["-t", "2.5", "-vf", f"scale={W}:{H},setsar=1,format=yuv420p", *F.h264_args(crf, "veryfast"),
                 "-r", f"{fps:g}"]
        args += (["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-shortest"] if has_audio else [])
        F.run(args + [str(part)], ws, outputs=[part], timeout=limits.ffmpeg_timeout_s, cancelled=cancelled,
              label="title card")
        parts.append(part)
    for i, c in enumerate(cuts):
        part = work / f"part_{i:03d}.mp4"
        args = ["-ss", f"{c.start:.3f}", "-t", f"{c.duration:.3f}", "-i", str(sp)]
        if used == "copy":
            # starts are keyframe-exact; the end is cut by frame count (with B-frame sources the
            # last frame or two can differ from a re-encode) and audio by time
            nframes = max(1, int(round(c.duration * fps)))
            args = ["-ss", f"{c.start:.3f}", "-i", str(sp), "-map", "0:v:0"] + \
                (["-map", "0:a:0"] if has_audio else []) + \
                ["-c", "copy", "-frames:v", str(nframes), "-t", f"{nframes / fps:.4f}", "-avoid_negative_ts",
                 "make_zero"]
        else:
            vf = f"scale={W}:{H},setsar=1,fps={fps:g},format=yuv420p"
            if captions and c.label:
                cap = work / f"cap_{i:03d}.png"
                caption_image(c.label, (W, H), theme_name).save(cap)
                args += ["-i", str(cap), "-filter_complex",
                         f"[0:v]{vf}[b];[b][1:v]overlay=x=(W-w)/2:y=H-h-{int(H * 0.05)},format=yuv420p[v]",
                         "-map", "[v]"] + (["-map", "0:a:0"] if has_audio else [])
            else:
                args += ["-vf", vf, "-map", "0:v:0"] + (["-map", "0:a:0"] if has_audio else [])
            nframes = max(1, int(round(c.duration * fps)))
            args += [*F.h264_args(crf, "veryfast"), "-r", f"{fps:g}", "-frames:v", str(nframes)]
            args += (["-c:a", "aac", "-b:a", "160k", "-ar", "48000", "-ac", "2", "-t", f"{nframes / fps:.4f}"]
                     if has_audio else ["-an"])
        F.run(args + [str(part)], ws, outputs=[part], timeout=limits.ffmpeg_timeout_s, cancelled=cancelled,
              label=f"cut {i + 1}")
        parts.append(part)
        prog(0.9 * (i + 1) / len(cuts), f"cut {i + 1}/{len(cuts)}")
    lst = work / "list.txt"
    lst.write_text("".join(f"file '{p.name}'\n" for p in parts), encoding="utf-8")
    F.run(["-f", "concat", "-safe", "1", "-i", lst.name, "-c", "copy", "-movflags", "+faststart", str(o)],
          ws, outputs=[o], cwd=work, timeout=limits.ffmpeg_timeout_s, cancelled=cancelled, label="concat")
    title_s = 2.5 if (title and used == "reencode") else 0.0
    rec = title_s
    cues = []
    for c in cuts:
        if c.label:
            cues.append(Cue(rec, rec + c.duration, c.label))
        rec += c.duration
    srt = None
    if cues:
        srt = write_srt(cues, o.with_suffix(".srt"))
    doc = export_cut_list(cuts, sp, o.with_suffix(".cuts.json"), o.with_suffix(".edl"), fps,
                          title=f"JEV {o.stem}")
    for p in work.iterdir():
        p.unlink(missing_ok=True)
    work.rmdir()
    out_info = F.probe(o, ws, limits, kind="footage", check=False)
    prog(1.0, "done")
    return {"video": o, "mode": used, "duration_s": out_info.duration_s, "expected_s": title_s + total,
            "cuts_json": o.with_suffix(".cuts.json"), "edl": o.with_suffix(".edl"), "srt": srt,
            "cut_list": doc, "notes": notes}


def frame_at(path: Path, workspace: Path, t: float, width: int = 320, info: F.MediaInfo | None = None) -> Image.Image:
    info = info or F.probe(path, workspace, kind="footage")
    w = min(width, even(info.width))
    h = even(w * info.height / info.width)
    for fr in F.decode_frames(path, workspace, (w, h), start=max(0.0, t), max_frames=1):
        return Image.fromarray(fr)
    raise MediaError(f"no frame at {t:.2f} s")
