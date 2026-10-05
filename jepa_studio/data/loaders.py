"""Loading user data safely: images, short video clips and time series from local folders.

Security (docs/SECURITY_MODEL.md, "malicious datasets"):
  * extension allowlist; files are opened by path only after resolving inside the chosen root
    (symlinks pointing outside the root are skipped)
  * per-file size cap and a pixel-count cap (Pillow raises DecompressionBombError above it)
  * video: duration and frame-size caps checked from the container metadata BEFORE any frame
    is decoded (unreadable metadata -> refused); then, while decoding, a frame-count cap
    (decoding stops there) and a re-check of every decoded frame's size
  * time series: CSV parsed as numbers only (no eval, no pickle); .npy loaded with
    allow_pickle=False
  * nothing here touches the network
"""
from __future__ import annotations

import csv
import os
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import synthetic
from .views import ImageViews, SeriesViews

IMAGE_EXT = {".png", ".jpg", ".jpeg", ".webp", ".bmp", ".gif"}
VIDEO_EXT = {".mp4", ".webm", ".mov", ".gif", ".avi", ".mkv"}
SERIES_EXT = {".csv", ".npy", ".tsv"}
MAX_FILE_BYTES = 64 * 1024 * 1024          # 64 MB per image / series file
MAX_VIDEO_BYTES = 512 * 1024 * 1024        # 512 MB per clip
MAX_PIXELS = 40_000_000                    # ~6300 x 6300; above this is treated as a bomb
MAX_FRAMES = 512                           # frames decoded per clip, at most
MAX_CLIP_SECONDS = 60.0                    # longer clips are refused before decoding
MAX_CLIP_PIXELS = 4096 * 4096              # per video frame (4K/DCI fits, 8K does not)
MAX_SERIES_VALUES = 50_000_000

Image.MAX_IMAGE_PIXELS = MAX_PIXELS


class UnsafeInput(ValueError):
    pass


def safe_listing(root: str | Path, exts: set[str], max_items: int | None = None) -> list[Path]:
    root = Path(root).expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"not a folder: {root}")
    out: list[Path] = []
    for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
        dirnames.sort()
        for f in sorted(filenames):
            p = Path(dirpath) / f
            if p.suffix.lower() not in exts:
                continue
            rp = p.resolve()
            if root not in rp.parents:
                continue  # symlink escaping the chosen folder
            out.append(rp)
            if max_items and len(out) >= max_items:
                return out
    return out


def load_image(path: Path, max_side: int) -> np.ndarray:
    if path.stat().st_size > MAX_FILE_BYTES:
        raise UnsafeInput(f"{path.name}: larger than {MAX_FILE_BYTES >> 20} MB")
    with Image.open(path) as im:
        w, h = im.size
        if w * h > MAX_PIXELS:
            raise UnsafeInput(f"{path.name}: {w}x{h} exceeds pixel cap")
        im.draft("RGB", (max_side, max_side))  # JPEG: decode at reduced size
        im = im.convert("RGB")
        im.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
        return np.asarray(im, dtype=np.float32) / 255.0


def clip_metadata(path: Path) -> dict:
    """Frame size, duration and frame count of a clip, read from the container header only
    (no pixel data is decoded). Raises UnsafeInput when the header cannot be read.

    * GIF: Pillow opens the file lazily; the frame count comes from walking the frame headers.
      GIF stores a delay per frame and reading every delay means seeking through the frames,
      so the duration is estimated as frame count x the first frame's delay.
    * everything else: FFmpeg (imageio-ffmpeg) probes the container: size, fps, duration.
    """
    if path.suffix.lower() == ".gif":
        try:
            with Image.open(path) as im:
                w, h = im.size
                n = int(getattr(im, "n_frames", 1))
                delay_ms = float(im.info.get("duration") or 0)
        except Exception as e:  # noqa: BLE001 - any parse error means "unreadable"
            raise UnsafeInput(f"{path.name}: cannot read GIF header ({type(e).__name__}: {e})") from e
        return {"width": w, "height": h, "frames": n, "duration_s": n * delay_ms / 1000.0,
                "duration_estimated": True}
    import imageio.v3 as iio

    try:
        with iio.imopen(path, "r", plugin="FFMPEG") as f:
            meta = f.metadata()
    except Exception as e:  # noqa: BLE001
        raise UnsafeInput(f"{path.name}: cannot read clip metadata ({type(e).__name__}: "
                          f"{str(e).splitlines()[0][:120] if str(e) else ''})") from e
    sizes = [tuple(meta[k]) for k in ("size", "source_size") if meta.get(k)]
    if not sizes:
        raise UnsafeInput(f"{path.name}: clip metadata has no frame size")
    w, h = max(sizes, key=lambda s: s[0] * s[1])
    fps = float(meta.get("fps") or 0)
    nframes = meta.get("nframes")
    dur = meta.get("duration")
    if not dur and fps > 0 and isinstance(nframes, (int, float)) and np.isfinite(nframes):
        dur = nframes / fps
    if dur is None or not np.isfinite(float(dur)):
        raise UnsafeInput(f"{path.name}: clip duration unknown (refused: the duration cap cannot be checked)")
    return {"width": int(w), "height": int(h), "fps": fps, "duration_s": float(dur),
            "frames": nframes if isinstance(nframes, int) else None, "duration_estimated": False}


def check_clip_metadata(path: Path, meta: dict) -> None:
    if meta["width"] <= 0 or meta["height"] <= 0:
        raise UnsafeInput(f"{path.name}: invalid frame size {meta['width']}x{meta['height']}")
    if meta["width"] * meta["height"] > MAX_CLIP_PIXELS:
        raise UnsafeInput(f"{path.name}: frame size {meta['width']}x{meta['height']} exceeds the "
                          f"{MAX_CLIP_PIXELS / 1e6:.1f} megapixel cap for video")
    if meta["duration_s"] > MAX_CLIP_SECONDS:
        raise UnsafeInput(f"{path.name}: clip is {meta['duration_s']:.1f} s long, above the "
                          f"{MAX_CLIP_SECONDS:g} s cap for clips")


def load_clip(path: Path, frames: int, max_side: int) -> np.ndarray:
    """Evenly spaced frames of a short clip, shape (T, H, W, 3) in [0,1].

    Caps, in order: file size (MAX_VIDEO_BYTES); then, from the container header and before any
    frame is decoded, duration (MAX_CLIP_SECONDS) and frame size (MAX_CLIP_PIXELS); then, while
    decoding, at most MAX_FRAMES frames and every decoded frame re-checked against MAX_CLIP_PIXELS
    (a header can lie about the size)."""
    import imageio.v3 as iio

    if path.stat().st_size > MAX_VIDEO_BYTES:
        raise UnsafeInput(f"{path.name}: clip larger than {MAX_VIDEO_BYTES >> 20} MB")
    check_clip_metadata(path, clip_metadata(path))
    got = []
    plugin = None if path.suffix.lower() == ".gif" else "FFMPEG"
    try:
        for i, fr in enumerate(iio.imiter(path, plugin=plugin)):
            if i >= MAX_FRAMES:
                break
            fr = np.asarray(fr)
            if fr.ndim == 2:
                fr = np.stack([fr] * 3, -1)
            if fr.shape[0] * fr.shape[1] > MAX_CLIP_PIXELS:
                raise UnsafeInput(f"{path.name}: decoded frame {fr.shape[1]}x{fr.shape[0]} exceeds pixel cap")
            im = Image.fromarray(fr[..., :3].astype(np.uint8))
            im.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
            got.append(np.asarray(im, dtype=np.float32) / 255.0)
    except UnsafeInput:
        raise
    except Exception as e:  # noqa: BLE001 - corrupt stream after a valid header
        raise UnsafeInput(f"{path.name}: decoding failed ({type(e).__name__})") from e
    if not got:
        raise UnsafeInput(f"{path.name}: no frames decoded")
    idx = np.linspace(0, len(got) - 1, frames).round().astype(int)
    return np.stack([got[i] for i in idx])


def load_series(path: Path) -> np.ndarray:
    """Numeric table -> (channels, length) float32. Header row allowed; non-numeric columns dropped."""
    if path.stat().st_size > MAX_FILE_BYTES:
        raise UnsafeInput(f"{path.name}: larger than {MAX_FILE_BYTES >> 20} MB")
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=False)
        if arr.size > MAX_SERIES_VALUES:
            raise UnsafeInput("series too large")
        arr = np.asarray(arr, dtype=np.float32)
        return arr[None] if arr.ndim == 1 else arr.T
    delim = "\t" if path.suffix.lower() == ".tsv" else ","
    rows: list[list[float]] = []
    with open(path, newline="", encoding="utf-8", errors="replace") as f:
        for r in csv.reader(f, delimiter=delim):
            vals = []
            for x in r:
                try:
                    vals.append(float(x))
                except ValueError:
                    vals.append(float("nan"))
            rows.append(vals)
            if len(rows) * max(1, len(vals)) > MAX_SERIES_VALUES:
                raise UnsafeInput("series too large")
    width = max(len(r) for r in rows)
    arr = np.array([r + [np.nan] * (width - len(r)) for r in rows], dtype=np.float32)
    keep = [c for c in range(width) if np.isfinite(arr[1:, c]).mean() > 0.9]
    arr = arr[:, keep]
    arr = arr[np.isfinite(arr).all(1)]
    if arr.size == 0:
        raise UnsafeInput(f"{path.name}: no numeric columns")
    return arr.T


# --------------------------------------------------------------------------- datasets

def _labels_from_folders(paths: list[Path], root: Path) -> tuple[np.ndarray, list[str]]:
    """Optional labels = immediate subfolder name (only used by Evaluate)."""
    names = [p.parent.name if p.parent != root else "_" for p in paths]
    classes = sorted(set(names))
    idx = {c: i for i, c in enumerate(classes)}
    return np.array([idx[n] for n in names], dtype=np.int64), classes


class ViewDataset(Dataset):
    """Returns (list_of_global_views, list_of_local_views, label, index)."""

    def __init__(self, items, labels, views, to_tensor):
        self.items, self.labels, self.views, self.to_tensor = items, labels, views, to_tensor

    def __len__(self):
        return len(self.items)

    def raw(self, i: int) -> torch.Tensor:
        return self.to_tensor(self.items[i])

    def __getitem__(self, i):
        g, loc = self.views(self.raw(i))
        return g, loc, int(self.labels[i]), i


def collate_views(batch):
    g = [torch.stack([b[0][v] for b in batch]) for v in range(len(batch[0][0]))]
    loc = [torch.stack([b[1][v] for b in batch]) for v in range(len(batch[0][1]))]
    y = torch.tensor([b[2] for b in batch])
    idx = torch.tensor([b[3] for b in batch])
    return g, loc, y, idx


def _thwc_to_tchw(a: np.ndarray) -> torch.Tensor:
    """(T, H, W, C) clip -> (T, C, H, W) tensor. A module-level function (not a lambda) so the
    dataset pickles for spawn/forkserver DataLoader workers."""
    return torch.from_numpy(np.ascontiguousarray(a)).permute(0, 3, 1, 2).contiguous()


def _hwc_to_chw(a: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1).contiguous()


def _series_windows(arrs: list[np.ndarray], win: int, max_items: int):
    items = []
    for a in arrs:
        a = (a - a.mean(1, keepdims=True)) / (a.std(1, keepdims=True) + 1e-6)
        step = max(1, win // 2)
        for s in range(0, a.shape[1] - win + 1, step):
            items.append(a[:, s:s + win].astype(np.float32))
            if len(items) >= max_items:
                return items
    return items


def build_dataset(cfg: dict, seed_offset: int = 0) -> tuple[ViewDataset, dict]:
    """Returns the dataset and an info dict (counts, classes, notes) for the Data tab."""
    d = cfg["data"]
    kind = d["kind"]
    n_max = d.get("max_items", 4096)
    seed = cfg["seed"] + seed_offset
    info: dict = {"kind": kind}
    if kind == "synthetic-shapes":
        size = max(d["image_size"], 32)
        imgs, labels = synthetic.make_dataset(n_max, cfg["seed"], size)
        items = list(imgs)
        views = ImageViews(d, seed)
        tt = _hwc_to_chw
        info.update(classes=synthetic.CLASSES, count=len(items))
    elif kind == "images":
        root = Path(d["path"]).expanduser().resolve()
        paths = safe_listing(root, IMAGE_EXT, n_max)
        items, keep, skipped = [], [], []
        for p in paths:
            try:
                items.append(load_image(p, max(2 * d["image_size"], 64)))
                keep.append(p)
            except Exception as e:  # noqa: BLE001 - report and continue
                skipped.append(f"{p.name}: {e}")
        labels, classes = _labels_from_folders(keep, root)
        views = ImageViews(d, seed)
        tt = _hwc_to_chw
        info.update(count=len(items), classes=classes, skipped=skipped[:50])
    elif kind == "video":
        root = Path(d["path"]).expanduser().resolve()
        paths = safe_listing(root, VIDEO_EXT, n_max)
        items, keep, skipped = [], [], []
        for p in paths:
            try:
                items.append(load_clip(p, d.get("clip_frames", 8), max(2 * d["image_size"], 64)))
                keep.append(p)
            except Exception as e:  # noqa: BLE001
                skipped.append(f"{p.name}: {e}")
        labels, classes = _labels_from_folders(keep, root)
        views = ImageViews(d, seed)
        tt = _thwc_to_tchw
        info.update(count=len(items), classes=classes, skipped=skipped[:50])
    elif kind == "timeseries":
        root = Path(d["path"]).expanduser().resolve()
        paths = safe_listing(root, SERIES_EXT, 10_000) if root.is_dir() else [root]
        arrs, skipped = [], []
        for p in paths:
            try:
                arrs.append(load_series(p))
            except Exception as e:  # noqa: BLE001 - one bad file is skipped, not fatal
                skipped.append(f"{p.name}: {e}")
        if not arrs:
            raise ValueError(f"no readable series files in {root}" + (f" ({skipped[0]})" if skipped else ""))
        ch = min(a.shape[0] for a in arrs)
        arrs = [a[:ch] for a in arrs]
        items = _series_windows(arrs, d.get("series_window", 128), n_max)
        labels = np.zeros(len(items), dtype=np.int64)
        views = SeriesViews(d, seed)
        tt = torch.from_numpy
        info.update(count=len(items), channels=ch, classes=["_"], skipped=skipped[:50])
    else:
        raise ValueError(kind)
    if not items:
        raise UnsafeInput("no usable items found")
    return ViewDataset(items, labels, views, tt), info


def build_eval_arrays(ds: ViewDataset, n: int | None = None, size: int | None = None):
    """Un-augmented inputs (center view) + labels for probes / k-NN / explorer."""
    import torch.nn.functional as F

    idx = range(len(ds) if n is None else min(n, len(ds)))
    xs = []
    for i in idx:
        x = ds.raw(i).float()
        if size is not None and x.dim() >= 3 and x.shape[-1] != size:
            flat = x.reshape(-1, *x.shape[-3:])
            flat = F.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
            x = flat.reshape(*x.shape[:-3], *flat.shape[-3:])
        xs.append(x)
    return torch.stack(xs), torch.as_tensor(ds.labels[: len(xs)])
