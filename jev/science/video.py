"""Small helpers shared by the renderers: video writers and a dark plotting style for summary charts."""
from __future__ import annotations

from pathlib import Path

# Dark-mode tokens from docs/STYLE.md (videos and charts are shown on dark backgrounds)
BG = (18, 24, 38)            # --bg dark #121826
SURFACE = (27, 36, 55)       # --surface dark #1b2437
INK = (232, 236, 244)        # --ink dark
INK3 = (151, 161, 180)       # --ink-3 dark
LINE = (46, 58, 84)          # --line dark
CAT = [(57, 135, 229), (217, 89, 38), (25, 158, 112), (144, 133, 233), (213, 81, 129)]   # --cat-1..5 dark


def hexc(rgb):
    return "#%02x%02x%02x" % tuple(rgb)


def open_writer(path: Path, fps: float, fmt: str | None = None):
    """imageio writer for .mp4 (H.264 via imageio-ffmpeg, yuv420p) or .gif."""
    import imageio.v2 as imageio
    fmt = fmt or Path(path).suffix.lstrip(".")
    if fmt == "gif":
        return imageio.get_writer(path, mode="I", duration=1000.0 / fps, loop=0)  # wsfs-ok: scratch path (published via wsfs)
    try:
        import imageio_ffmpeg  # noqa: F401
    except ImportError as e:   # pragma: no cover - depends on the install
        raise RuntimeError("MP4 output needs imageio-ffmpeg: pip install imageio-ffmpeg "
                           "(or ask for format='gif')") from e
    return imageio.get_writer(path, fps=fps, codec="libx264", quality=8, pixelformat="yuv420p",  # wsfs-ok: scratch path
                              macro_block_size=2, ffmpeg_log_level="error")


def chart_style():
    """Matplotlib rcParams following docs/STYLE.md chart rules (dark surface, 2 px lines, quiet grid)."""
    return {"figure.facecolor": hexc(SURFACE), "axes.facecolor": hexc(SURFACE), "savefig.facecolor": hexc(SURFACE),
            "axes.edgecolor": hexc(LINE), "axes.labelcolor": hexc(INK3), "axes.titlecolor": hexc(INK),
            "xtick.color": hexc(INK3), "ytick.color": hexc(INK3), "grid.color": hexc(LINE), "axes.grid": True,
            "grid.linewidth": 1.0, "lines.linewidth": 2.0, "lines.solid_joinstyle": "round", "font.size": 10,
            "axes.titlesize": 11, "legend.frameon": False, "legend.labelcolor": hexc(INK), "text.color": hexc(INK)}
