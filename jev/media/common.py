"""Shared pieces for jev.media: the error type, workspace checks, limits and font lookup."""
from __future__ import annotations

import os
import sys
from dataclasses import dataclass, fields
from functools import lru_cache
from pathlib import Path

from PIL import Image, ImageFont

from jepa_studio.data import loaders as L
from jev.tools.base import ToolError


class MediaError(ToolError):
    """An expected media failure (bad input, cap exceeded, missing optional dependency). The
    message is shown to the user and should say what to do next."""


# ----------------------------------------------------------------------------- workspace

def inside(path: str | Path, workspace: str | Path) -> Path:
    """Resolve `path` and refuse it unless it lies inside `workspace` (symlinks resolved)."""
    ws = Path(workspace).resolve()
    p = Path(path)
    p = (p if p.is_absolute() else ws / p).resolve()
    if p != ws and ws not in p.parents:
        raise MediaError(f"{path} is outside the workspace ({ws}); media tools only read and write there")
    return p


def out_path(workspace: str | Path, relative: str | Path, suffix: str) -> Path:
    """An output path inside the workspace with the given suffix; parent folders are created."""
    p = inside(relative, workspace)
    if p.suffix.lower() != suffix:
        p = p.with_suffix(suffix)
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


# ----------------------------------------------------------------------------- limits

@dataclass
class Limits:
    """Caps checked from container metadata BEFORE any frame is decoded.

    Frame size and file size match jepa_studio.data.loaders (the training loaders). Duration has a
    separate, larger cap for footage editing (a 60 s training-clip cap would make editing useless);
    clips inserted into explainers keep the loaders' 60 s cap. Sampled frames are capped too."""
    max_bytes: int = L.MAX_VIDEO_BYTES                 # 512 MB per input file
    max_pixels: int = L.MAX_CLIP_PIXELS                # 4096 x 4096 per frame
    max_image_pixels: int = L.MAX_PIXELS               # stills (Pillow bomb cap)
    max_image_bytes: int = L.MAX_FILE_BYTES            # 64 MB per still
    max_footage_seconds: float = 30 * 60.0             # footage you edit
    max_clip_seconds: float = L.MAX_CLIP_SECONDS       # clips inserted into an explainer
    max_sample_frames: int = 3000                      # frames sampled for analysis
    max_output_seconds: float = 30 * 60.0              # rendered videos
    max_output_pixels: int = 3840 * 2160               # rendered frame size (4K)
    ffmpeg_timeout_s: float = 3600.0                   # one FFmpeg call

    @classmethod
    def from_settings(cls, settings: dict | None) -> "Limits":
        media = (settings or {}).get("media", {}) or {}
        given = media.get("limits", {}) or {}
        names = {f.name for f in fields(cls)}
        return cls(**{k: type(getattr(cls, k))(v) for k, v in given.items() if k in names})


# ----------------------------------------------------------------------------- fonts

_FONT_CANDIDATES: dict[str, list[str]] = {
    "sans": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSans.ttf",
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/Library/Fonts/Arial.ttf",
        "/System/Library/Fonts/Helvetica.ttc",
        "C:/Windows/Fonts/segoeui.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ],
    "bold": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/TTF/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Bold.ttf",
        "/usr/share/fonts/truetype/freefont/FreeSansBold.ttf",
        "/System/Library/Fonts/Supplemental/Arial Bold.ttf",
        "/Library/Fonts/Arial Bold.ttf",
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
    ],
    "mono": [
        "/usr/share/fonts/truetype/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/TTF/DejaVuSansMono.ttf",
        "/usr/share/fonts/dejavu/DejaVuSansMono.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationMono-Regular.ttf",
        "/usr/share/fonts/liberation-mono/LiberationMono-Regular.ttf",
        "/System/Library/Fonts/Menlo.ttc",
        "/System/Library/Fonts/Monaco.ttf",
        "C:/Windows/Fonts/consola.ttf",
        "C:/Windows/Fonts/cour.ttf",
    ],
}
_MPL_FONT = {"sans": "DejaVuSans.ttf", "bold": "DejaVuSans-Bold.ttf", "mono": "DejaVuSansMono.ttf"}


def _windows_fonts_dir() -> list[str]:
    windir = os.environ.get("WINDIR")
    return [str(Path(windir) / "Fonts")] if windir and sys.platform == "win32" else []


@lru_cache(maxsize=None)
def font_file(role: str) -> str | None:
    """Path of a TrueType font for `role` (sans | bold | mono), or None.

    Order: common system locations (Linux DejaVu/Liberation/FreeFont, macOS Arial/Helvetica/Menlo,
    Windows Segoe UI/Arial/Consolas), then the DejaVu fonts matplotlib ships with (present wherever
    matplotlib is installed, on every OS). None means Pillow's built-in font is used."""
    cands = list(_FONT_CANDIDATES.get(role, []))
    for d in _windows_fonts_dir():
        cands += [str(Path(d) / Path(c).name) for c in cands if c.startswith("C:/")]
    for c in cands:
        if Path(c).is_file():
            return c
    try:
        import matplotlib

        p = Path(matplotlib.get_data_path()) / "fonts" / "ttf" / _MPL_FONT.get(role, "DejaVuSans.ttf")
        if p.is_file():
            return str(p)
    except Exception:  # noqa: BLE001 - matplotlib optional here
        pass
    return None


@lru_cache(maxsize=256)
def font(role: str, size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    size = max(6, int(size))
    f = font_file(role)
    if f:
        try:
            return ImageFont.truetype(f, size)
        except OSError:
            pass
    try:
        return ImageFont.load_default(size=size)      # Pillow >= 10.1: scalable built-in font
    except TypeError:                                 # older Pillow: fixed bitmap font
        return ImageFont.load_default()


def open_image_safely(path: Path, limits: Limits | None = None) -> Image.Image:
    """Open a still with the same caps as the training loaders (file size, pixel count)."""
    limits = limits or Limits()
    if path.stat().st_size > limits.max_image_bytes:
        raise MediaError(f"{path.name}: larger than {limits.max_image_bytes >> 20} MB")
    try:
        im = Image.open(path)
        w, h = im.size
        if w * h > limits.max_image_pixels:
            raise MediaError(f"{path.name}: {w}x{h} exceeds the {limits.max_image_pixels / 1e6:.0f} MP cap")
        im.load()
    except MediaError:
        raise
    except Exception as e:  # noqa: BLE001 - Pillow raises many types for bad files
        raise MediaError(f"{path.name}: cannot read image ({type(e).__name__}: {e})") from e
    return im.convert("RGBA") if im.mode in ("RGBA", "LA", "P") else im.convert("RGB")


def even(n: float) -> int:
    """Nearest even integer >= 2 (yuv420p needs even frame sizes)."""
    return max(2, int(round(n / 2.0)) * 2)
