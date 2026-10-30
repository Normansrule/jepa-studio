"""Running FFmpeg safely.

Rules every caller gets for free:
  * FFmpeg and ffprobe run inside the same sandbox as model-written code (jev.sandbox): no
    network, only the workspace writable, jev-data and your home folder hidden (with bubblewrap),
    resource limits, the whole process tree killed on timeout or cancel. The media tools pass
    their conversation's sandbox with `use_sandbox()`; without one a default sandbox is used.
  * FFmpeg is found on PATH, else the binary bundled with imageio-ffmpeg (`pip install imageio-ffmpeg`).
  * Commands are argument lists, never a shell string.
  * Every `-i` input that is a file must resolve inside the workspace and be a regular file; it is
    passed as `file:<abs>` with `-protocol_whitelist file`, an explicit `-f <demuxer>` chosen from
    the file's first bytes (sniff_container) and `-format_whitelist` of plain media containers, so
    a playlist or script disguised as a video (ffconcat, HLS/m3u8, image2 patterns, data, subfile,
    ...) is refused before FFmpeg opens it: such demuxers could open other files, including files
    outside the workspace. Only JEV's own concat lists (validated entry by entry) use `concat`.
    Where the sandbox has no file-system isolation, symlinks and hard-linked files are refused too.
    Filter graphs that open files themselves (movie=, amovie=, textfile=) are refused, and a
    `subtitles=` file must be inside the workspace.
  * Outputs must be inside the workspace (or a pipe).
  * Each call has a timeout and is cancellable (kill of the sandboxed tree); progress comes from
    `-progress pipe:1`.
  * probe() checks duration, frame size and file size from the container metadata BEFORE any
    frame is decoded (the same caps as jepa_studio.data.loaders; unknown duration is refused).
"""
from __future__ import annotations

import contextlib
import contextvars
import json
import os
import re
import shutil
import stat
import subprocess
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from functools import lru_cache
from pathlib import Path
from typing import Callable, Iterable, Iterator

import numpy as np

from jev.sandbox import Limits as RunLimits
from jev.sandbox import Mounts, Sandbox, SandboxError, Spawned, sandbox_for

from .common import Limits, MediaError, inside

ProgressFn = Callable[[float, str], None]
CancelFn = Callable[[], bool]

_FILE_OPENING_FILTERS = re.compile(r"(^|[\[,;:=\s'])(a?movie|textfile|filename|fontfile|file|f)\s*=", re.I)
_FILE_OPENING_NAMES = re.compile(r"(^|[\[,;\s])(ladspa|lv2|frei0r|frei0r_src|a?sendcmd|a?zmq|lut3d|lut1d|haldclutsrc|"
                                 r"dnn_\w+|sofalizer|arnndn|vidstab\w+|ass)\s*(=|\[|,|;|$)", re.I)
_SUBTITLE_FILE = re.compile(r"subtitles=(?:filename=|f=)?'?([^:,;\]']+)")
_PROTOCOL = re.compile(r"^[A-Za-z][A-Za-z0-9+.\-]{1,}:(?![\\/])|://")

# Demuxers JEV lets FFmpeg use on workspace files: plain containers that never open other files.
SAFE_DEMUXERS = ("mov", "matroska", "avi", "wav", "mp3", "ogg", "flac", "aac", "mpegts", "mpeg", "flv", "gif",
                 "image2", "png_pipe", "jpeg_pipe", "webp_pipe", "bmp_pipe")
FORMAT_WHITELIST = ",".join(SAFE_DEMUXERS)
_CONCAT_LINE = re.compile(r"^file '([A-Za-z0-9._-]{1,200})'$")

# ----------------------------------------------------------------------------- sandbox

_SANDBOX: contextvars.ContextVar = contextvars.ContextVar("jev_media_sandbox", default=None)


@contextlib.contextmanager
def use_sandbox(sb: Sandbox | None):
    """Run every FFmpeg / ffprobe call made inside this block with `sb` (the conversation's sandbox)."""
    tok = _SANDBOX.set(sb)
    try:
        yield sb
    finally:
        _SANDBOX.reset(tok)


def current_sandbox() -> Sandbox:
    sb = _SANDBOX.get()
    return sb if sb is not None else sandbox_for({})


def _weak(sb: Sandbox) -> bool:
    return not sb.capabilities().get("fs_isolated")


def _bin_dirs() -> list[Path]:
    out = []
    for exe in (find_ffmpeg(), find_ffprobe()):
        if exe:
            d = Path(exe).resolve().parent
            if d not in out:
                out.append(d)
    return out


def _run_limits(sb: Sandbox, timeout: float | None) -> RunLimits:
    base = sb.limits
    t = float(timeout or 3600.0)
    return RunLimits(cpu_seconds=int(t * max(1, os.cpu_count() or 1)) + 60, memory_mb=max(base.memory_mb, 4096),
                     wall_seconds=int(t) + 60, max_output_mb=max(base.max_output_mb, 8192),
                     max_file_mb=base.max_file_mb, max_workspace_mb=base.max_workspace_mb)


def _spawn(cmd: list[str], ws: Path, *, cwd: Path | None = None, stdin: bool = False, text: bool = False,
           timeout: float | None = 3600.0) -> Spawned:
    sb = current_sandbox()
    try:
        return sb.spawn(cmd, ws, _run_limits(sb, timeout), network=False, cwd=cwd, stdin=stdin, text=text,
                        mounts=Mounts(ro=_bin_dirs()), honor_network_setting=False)
    except SandboxError as e:
        raise MediaError(f"FFmpeg was not started: {e}") from e


def _capture(cmd: list[str], ws: Path, timeout: float = 60.0, out_cap: int = 64 << 20) -> tuple[int, str, str]:
    """Run a short FFmpeg / ffprobe command in the sandbox and return (returncode, stdout, stderr)."""
    sp = _spawn(cmd, ws, timeout=timeout)
    bufs = {"out": bytearray(), "err": bytearray()}

    def reader(stream, key, cap):
        try:
            while chunk := stream.read(65536):
                if len(bufs[key]) < cap:
                    bufs[key] += chunk[: cap - len(bufs[key])]
        except (OSError, ValueError):
            pass
    ts = [threading.Thread(target=reader, args=(sp.proc.stdout, "out", out_cap), daemon=True),
          threading.Thread(target=reader, args=(sp.proc.stderr, "err", 1 << 20), daemon=True)]
    for t in ts:
        t.start()
    try:
        sp.proc.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        sp.kill()
        sp.proc.wait()
        raise MediaError(f"{Path(cmd[0]).name} timed out after {timeout:.0f} s") from None
    finally:
        sp.finish()
        for t in ts:
            t.join(timeout=2)
    return sp.proc.returncode, bufs["out"].decode("utf-8", "replace"), bufs["err"].decode("utf-8", "replace")


# ----------------------------------------------------------------------------- inputs

def sniff_container(path: str | Path) -> tuple[str, list[str]]:
    """The demuxer for a workspace media file, from its first bytes (never FFmpeg's own guess).
    Returns (demuxer, extra input options). Anything that is not a plain media container (text
    playlists such as ffconcat or m3u8, SDP, scripts, unknown data) raises MediaError."""
    p = Path(path)
    try:
        with open(p, "rb") as f:
            head = f.read(4096)
    except OSError as e:
        raise MediaError(f"{p.name}: cannot read ({e.strerror})") from e
    ext = p.suffix.lower()
    if len(head) >= 12 and head[4:8] in (b"ftyp", b"moov", b"mdat", b"wide", b"free", b"skip", b"pnot"):
        return "mov", []
    if head[:4] == b"\x1aE\xdf\xa3":
        return "matroska", []
    if head[:4] == b"RIFF" and len(head) >= 12:
        kind = head[8:12]
        if kind == b"WAVE":
            return "wav", []
        if kind == b"AVI ":
            return "avi", []
        if kind == b"WEBP":
            return "webp_pipe", []
    if head[:4] == b"OggS":
        return "ogg", []
    if head[:4] == b"fLaC":
        return "flac", []
    if head[:3] == b"FLV":
        return "flv", []
    if head[:6] in (b"GIF87a", b"GIF89a"):
        return "gif", []
    if head[:8] == b"\x89PNG\r\n\x1a\n":
        return ("image2", ["-pattern_type", "none"]) if ext == ".png" else ("png_pipe", [])
    if head[:3] == b"\xff\xd8\xff":
        return ("image2", ["-pattern_type", "none"]) if ext in (".jpg", ".jpeg") else ("jpeg_pipe", [])
    if head[:2] == b"BM" and ext == ".bmp":
        return "bmp_pipe", []
    if head[:4] == b"\x00\x00\x01\xba":
        return "mpeg", []
    if len(head) > 376 and head[0] == 0x47 and head[188] == 0x47 and head[376] == 0x47:
        return "mpegts", []
    if head[:3] == b"ID3":
        return "mp3", []
    if len(head) >= 2 and head[0] == 0xFF and head[1] & 0xE0 == 0xE0:
        return ("aac", []) if head[1] & 0x06 == 0 else ("mp3", [])
    raise MediaError(f"{p.name} is not a recognised media file (refused: JEV only reads common audio, video and "
                     "image containers, never playlists or scripts such as ffconcat, m3u8 or sdp)")


def check_input_file(raw: str | Path, ws: Path, weak: bool | None = None) -> Path:
    """Resolve an input inside the workspace and refuse anything but a plain regular file. On a weak
    sandbox (no file-system isolation) symlinks and hard-linked files are refused as well, because
    FFmpeg would then run with your user's full view of the disk."""
    ws = Path(ws).resolve()
    rawp = Path(raw)
    rawp = rawp if rawp.is_absolute() else ws / rawp
    weak = _weak(current_sandbox()) if weak is None else weak
    p = inside(rawp, ws)
    if weak and rawp.is_symlink():
        raise MediaError(f"{rawp.name} is a symbolic link; media tools do not follow links on this computer's "
                         "sandbox (it has no file-system isolation)")
    try:
        st = os.lstat(p)
    except OSError as e:
        raise MediaError(f"input not found: {rawp.name}") from e
    if not stat.S_ISREG(st.st_mode):
        raise MediaError(f"{rawp.name} is not a regular file (FIFO, socket, device or folder); refused")
    if weak and st.st_nlink > 1:
        raise MediaError(f"{rawp.name} is a hard link to another file; media tools refuse hard links on this "
                         "computer's sandbox (it has no file-system isolation)")
    return p


def input_args(path: str | Path, ws: Path, weak: bool | None = None) -> tuple[list[str], Path]:
    """FFmpeg input options + the checked path for one workspace media file."""
    p = check_input_file(path, ws, weak)
    fmt, extra = sniff_container(p)
    return ["-protocol_whitelist", "file", "-format_whitelist", FORMAT_WHITELIST, "-f", fmt, *extra], p


def _check_concat_list(lst: Path, ws: Path, weak: bool) -> None:
    """JEV's own concat lists: `file '<plain name>'` lines naming regular media files next to the list."""
    try:
        text = lst.read_text(encoding="utf-8")
    except (OSError, UnicodeDecodeError) as e:
        raise MediaError(f"{lst.name}: unreadable concat list") from e
    names = []
    for line in text.splitlines():
        if not line.strip():
            continue
        m = _CONCAT_LINE.match(line.strip())
        if not m or m.group(1).startswith("."):
            raise MediaError(f"{lst.name}: concat lists may only name plain files next to the list")
        names.append(m.group(1))
    if not names:
        raise MediaError(f"{lst.name}: empty concat list")
    for n in names:
        p = check_input_file(lst.parent / n, ws, weak)
        fmt, _ = sniff_container(p)
        if fmt not in ("mov", "matroska", "mpegts", "wav"):
            raise MediaError(f"{n}: concat parts must be video or audio files")


# ----------------------------------------------------------------------------- locating

@lru_cache(maxsize=1)
def find_ffmpeg() -> str:
    exe = shutil.which("ffmpeg")
    if exe:
        return exe
    try:
        import imageio_ffmpeg

        exe = imageio_ffmpeg.get_ffmpeg_exe()
        if exe and Path(exe).is_file():
            return exe
    except Exception:  # noqa: BLE001 - not installed or no bundled binary for this platform
        pass
    raise MediaError("FFmpeg not found. Install it (Ubuntu: `sudo apt install ffmpeg`, macOS: "
                     "`brew install ffmpeg`, Windows: `winget install Gyan.FFmpeg`) or "
                     "`pip install imageio-ffmpeg`, which bundles a copy.")


@lru_cache(maxsize=1)
def find_ffprobe() -> str | None:
    """ffprobe next to ffmpeg or on PATH; None is fine (probe() parses `ffmpeg -i` instead)."""
    exe = shutil.which("ffprobe")
    if exe:
        return exe
    try:
        ff = Path(find_ffmpeg())
    except MediaError:
        return None
    for name in ("ffprobe", "ffprobe.exe"):
        cand = ff.with_name(name)
        if cand.is_file():
            return str(cand)
    return None


@lru_cache(maxsize=1)
def _filters() -> frozenset[str]:
    out = subprocess.run([find_ffmpeg(), "-hide_banner", "-filters"], capture_output=True, text=True,
                         timeout=30).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 3 and re.fullmatch(r"[TSC.|]{2,3}", parts[0]):
            names.add(parts[1])
    return frozenset(names)


def has_filter(name: str) -> bool:
    try:
        return name in _filters()
    except Exception:  # noqa: BLE001
        return False


@lru_cache(maxsize=1)
def _encoders() -> frozenset[str]:
    out = subprocess.run([find_ffmpeg(), "-hide_banner", "-encoders"], capture_output=True, text=True,
                         timeout=30).stdout
    names = set()
    for line in out.splitlines():
        parts = line.split()
        if len(parts) >= 2 and re.fullmatch(r"[VASFXBD.]{6}", parts[0]):
            names.add(parts[1])
    return frozenset(names)


def h264_encoder() -> str:
    enc = _encoders()
    for name in ("libx264", "libopenh264"):
        if name in enc:
            return name
    raise MediaError("this FFmpeg build has no H.264 encoder (libx264). Install a full FFmpeg build "
                     "or `pip install imageio-ffmpeg`, whose bundled FFmpeg includes libx264.")


def h264_args(crf: int = 20, preset: str = "medium") -> list[str]:
    enc = h264_encoder()
    if enc == "libx264":
        return ["-c:v", "libx264", "-preset", preset, "-crf", str(crf), "-pix_fmt", "yuv420p"]
    return ["-c:v", enc, "-b:v", "4M", "-pix_fmt", "yuv420p"]


# ----------------------------------------------------------------------------- running

@dataclass
class RunResult:
    returncode: int
    seconds: float
    stderr_tail: str = ""
    progress: dict = field(default_factory=dict)


def _check_filters(args: list[str], workspace: Path, cwd: Path | None) -> None:
    for i, a in enumerate(args):
        if i and args[i - 1] in ("-vf", "-af", "-filter_complex", "-filter:v", "-filter:a", "-lavfi") \
                or (i >= 3 and args[i - 1] == "-i" and args[i - 3:i - 1] == ["-f", "lavfi"]):
            if _FILE_OPENING_FILTERS.search(a) or _FILE_OPENING_NAMES.search(a):
                raise MediaError("filter graphs that open files or load plugins (movie=, textfile=, file=, "
                                 "ladspa, sendcmd, ...) are not allowed")
            for m in _SUBTITLE_FILE.finditer(a):
                inside((cwd or workspace) / m.group(1), workspace)


def _prepare_args(args: list[str], workspace: Path, outputs: Iterable[str | Path], cwd: Path | None) -> list[str]:
    ws = Path(workspace).resolve()
    _check_filters(args, ws, cwd)
    weak = _weak(current_sandbox())
    out: list[str] = []
    i = 0
    while i < len(args):
        a = args[i]
        if a == "-i" and i + 1 < len(args):
            src = str(args[i + 1])
            start = max((k for k, x in enumerate(out) if x == "-i"), default=-1) + 1   # this input's options
            fopts = [out[k + 1] for k in range(start, len(out) - 1) if out[k] == "-f"]
            fmt = fopts[-1] if fopts else None
            if fmt == "lavfi" or (src in ("pipe:0", "pipe:", "-") and fmt is not None):
                out += [a, src]
                i += 2
                continue
            if src in ("pipe:0", "pipe:", "-"):
                raise MediaError("a pipe input needs an explicit raw format (-f rawvideo ...)")
            if _PROTOCOL.search(src) and not re.match(r"^[A-Za-z]:[\\/]", src):
                raise MediaError(f"input {src!r} looks like a URL or protocol; only workspace files are read")
            raw = Path(cwd) / src if cwd and not Path(src).is_absolute() else Path(src)
            inside(raw, ws)
            if fmt == "concat":
                p = check_input_file(raw, ws, weak)
                _check_concat_list(p, ws, weak)
                if "-safe" not in out:
                    out += ["-safe", "1"]
                out += ["-protocol_whitelist", "file", "-format_whitelist", "concat," + FORMAT_WHITELIST,
                        a, "file:" + str(p)]
            elif fmt is not None:
                raise MediaError(f"-f {fmt} is not allowed for workspace files (JEV picks the demuxer itself)")
            else:
                opts, p = input_args(raw, ws, weak)
                out += [*opts, a, "file:" + str(p)]
            i += 2
            continue
        out.append(a)
        i += 1
    for o in outputs:
        if str(o).startswith("pipe:"):
            continue
        inside(o, ws)
    return out


def run(args: list[str], workspace: str | Path, *, outputs: Iterable[str | Path] = (),
        duration_s: float | None = None, progress: ProgressFn | None = None,
        cancelled: CancelFn | None = None, timeout: float | None = 3600.0,
        cwd: str | Path | None = None, label: str = "ffmpeg") -> RunResult:
    """Run `ffmpeg <args>` in the sandbox (the binary and global flags are added here). `outputs`
    lists every output file so it can be checked against the workspace. Raises MediaError on
    failure, timeout or cancellation (the partial output is removed)."""
    ws = Path(workspace).resolve()
    cwd_p = inside(cwd, ws) if cwd else None
    outputs = list(outputs)
    prepared = _prepare_args([str(a) for a in args], ws, outputs, cwd_p)
    cmd = [find_ffmpeg(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error",
           "-progress", "pipe:1", "-nostats", *prepared]
    t0 = time.monotonic()
    sp = _spawn(cmd, ws, cwd=cwd_p, text=True, timeout=timeout)
    proc = sp.proc
    tail: deque[str] = deque(maxlen=40)
    state: dict = {}

    def read_stdout():
        assert proc.stdout is not None
        for line in proc.stdout:
            k, _, v = line.strip().partition("=")
            state[k] = v
            if k in ("out_time_us", "out_time_ms") and duration_s and progress:
                try:
                    progress(min(1.0, int(v) / 1e6 / duration_s), label)
                except ValueError:
                    pass

    def read_stderr():
        assert proc.stderr is not None
        for line in proc.stderr:
            tail.append(line.rstrip())

    threads = [threading.Thread(target=read_stdout, daemon=True), threading.Thread(target=read_stderr, daemon=True)]
    for t in threads:
        t.start()
    reason = None
    while proc.poll() is None:
        if cancelled and cancelled():
            reason = "cancelled"
        elif timeout and time.monotonic() - t0 > timeout:
            reason = f"timed out after {timeout:.0f} s"
        if reason:
            sp.kill()
            break
        time.sleep(0.05)
    proc.wait()
    over = sp.finish()
    for t in threads:
        t.join(timeout=2)
    res = RunResult(proc.returncode, time.monotonic() - t0, "\n".join(tail), dict(state))
    if reason or proc.returncode != 0 or over:
        for o in outputs:
            if not str(o).startswith("pipe:"):
                Path(o).unlink(missing_ok=True)
        if reason:
            raise MediaError(f"{label} {reason}")
        if over:
            raise MediaError(f"{label}: {over}")
        raise MediaError(f"{label} failed (exit {proc.returncode}): {res.stderr_tail[-600:] or 'no message'}")
    if progress:
        progress(1.0, label)
    return res


# ----------------------------------------------------------------------------- probing

@dataclass
class MediaInfo:
    path: str
    size_bytes: int
    duration_s: float
    width: int = 0
    height: int = 0
    fps: float = 0.0
    frames: int | None = None
    video_codec: str | None = None
    audio_codec: str | None = None
    audio_rate: int | None = None
    audio_channels: int | None = None
    container: str | None = None
    source: str = "ffprobe"

    @property
    def has_video(self) -> bool:
        return self.video_codec is not None

    @property
    def has_audio(self) -> bool:
        return self.audio_codec is not None

    def to_json(self) -> dict:
        d = dict(self.__dict__)
        d.update(has_video=self.has_video, has_audio=self.has_audio)
        return d


def _rate(s: str | None) -> float:
    if not s:
        return 0.0
    if "/" in s:
        a, b = s.split("/", 1)
        try:
            return float(a) / float(b) if float(b) else 0.0
        except ValueError:
            return 0.0
    try:
        return float(s)
    except ValueError:
        return 0.0


def _probe_ffprobe(exe: str, p: Path, ws: Path | None = None) -> MediaInfo:
    ws = Path(ws or p.parent).resolve()
    opts, p = input_args(p, ws)
    cmd = [exe, "-v", "error", *opts, "-print_format", "json", "-show_format", "-show_streams", "file:" + str(p)]
    rc, out, err = _capture(cmd, ws, timeout=60)
    if rc != 0:
        raise MediaError(f"{p.name}: cannot read media metadata ({err.strip()[:200]})")
    try:
        meta = json.loads(out or "{}")
    except json.JSONDecodeError as e:
        raise MediaError(f"{p.name}: unreadable metadata") from e
    fmt = meta.get("format", {})
    info = MediaInfo(path=str(p), size_bytes=p.stat().st_size, duration_s=float("nan"),
                     container=fmt.get("format_name"))
    durs = []
    if fmt.get("duration") not in (None, "N/A"):
        durs.append(float(fmt["duration"]))
    for s in meta.get("streams", []):
        if s.get("codec_type") == "video" and info.video_codec is None and \
                not (s.get("disposition") or {}).get("attached_pic"):
            info.video_codec = s.get("codec_name")
            info.width, info.height = int(s.get("width") or 0), int(s.get("height") or 0)
            info.fps = _rate(s.get("avg_frame_rate")) or _rate(s.get("r_frame_rate"))
            if s.get("nb_frames", "").isdigit():
                info.frames = int(s["nb_frames"])
            if s.get("duration") not in (None, "N/A"):
                durs.append(float(s["duration"]))
        elif s.get("codec_type") == "audio" and info.audio_codec is None:
            info.audio_codec = s.get("codec_name")
            info.audio_rate = int(s.get("sample_rate") or 0) or None
            info.audio_channels = int(s.get("channels") or 0) or None
    if durs:
        info.duration_s = max(durs)
    return info


_DUR_RE = re.compile(r"Duration:\s*(\d+):(\d+):(\d+(?:\.\d+)?)")
_VID_RE = re.compile(r"Stream #\d+:\d+.*?: Video: (\w+).*?, (\d{2,5})x(\d{2,5})")
_FPS_RE = re.compile(r"([\d.]+) (?:fps|tbr)")
_AUD_RE = re.compile(r"Stream #\d+:\d+.*?: Audio: (\w+).*?, (\d+) Hz, (mono|stereo|[\d.]+)")


def parse_ffmpeg_banner(text: str, p: Path) -> MediaInfo:
    """Metadata from the stderr of `ffmpeg -i <file>` (used when ffprobe is absent)."""
    info = MediaInfo(path=str(p), size_bytes=p.stat().st_size if p.exists() else 0,
                     duration_s=float("nan"), source="ffmpeg -i")
    m = _DUR_RE.search(text)
    if m:
        info.duration_s = int(m.group(1)) * 3600 + int(m.group(2)) * 60 + float(m.group(3))
    for line in text.splitlines():
        if info.video_codec is None and (v := _VID_RE.search(line)) and "attached pic" not in line:
            info.video_codec, info.width, info.height = v.group(1), int(v.group(2)), int(v.group(3))
            f = _FPS_RE.search(line)
            info.fps = float(f.group(1)) if f else 0.0
        elif info.audio_codec is None and (a := _AUD_RE.search(line)):
            info.audio_codec, info.audio_rate = a.group(1), int(a.group(2))
            ch = a.group(3)
            info.audio_channels = {"mono": 1, "stereo": 2}.get(ch) or int(float(ch))
    return info


def probe(path: str | Path, workspace: str | Path, limits: Limits | None = None, *,
          kind: str = "footage", check: bool = True) -> MediaInfo:
    """Read container metadata (no frames decoded) and enforce the caps.

    kind: "footage" (max_footage_seconds) | "clip" (max_clip_seconds, the training-loader cap)
          | "audio" (max_output_seconds; no frame-size check)."""
    limits = limits or Limits()
    ws = Path(workspace).resolve()
    inside(path, ws)
    rawp = Path(path) if Path(path).is_absolute() else ws / path
    if not rawp.exists():
        raise MediaError(f"not found: {Path(path).name}")
    p = check_input_file(rawp, ws)
    size = p.stat().st_size
    if check and size > limits.max_bytes:
        raise MediaError(f"{p.name}: {size / 2**20:.0f} MB is larger than the {limits.max_bytes >> 20} MB cap")
    exe = find_ffprobe()
    if exe:
        info = _probe_ffprobe(exe, p, ws)
    else:
        opts, p = input_args(p, ws)
        _, _, err = _capture([find_ffmpeg(), "-hide_banner", "-nostdin", *opts, "-i", "file:" + str(p)], ws,
                             timeout=60)
        info = parse_ffmpeg_banner(err, p)
    if check:
        check_limits(info, limits, kind)
    return info


def check_limits(info: MediaInfo, limits: Limits, kind: str = "footage") -> None:
    name = Path(info.path).name
    if not info.has_video and not info.has_audio:
        raise MediaError(f"{name}: no audio or video stream found")
    if not np.isfinite(info.duration_s) or info.duration_s <= 0:
        raise MediaError(f"{name}: duration unknown (refused: the duration cap cannot be checked)")
    cap = {"footage": limits.max_footage_seconds, "clip": limits.max_clip_seconds,
           "audio": limits.max_output_seconds}.get(kind, limits.max_footage_seconds)
    if info.duration_s > cap:
        raise MediaError(f"{name}: {info.duration_s:.1f} s is longer than the {cap:g} s cap for {kind}")
    if info.has_video:
        if info.width <= 0 or info.height <= 0:
            raise MediaError(f"{name}: invalid frame size {info.width}x{info.height}")
        if info.width * info.height > limits.max_pixels:
            raise MediaError(f"{name}: frame size {info.width}x{info.height} exceeds the "
                             f"{limits.max_pixels / 1e6:.1f} megapixel cap for video")


def keyframe_times(path: str | Path, workspace: str | Path, timeout: float = 300) -> list[float]:
    """Presentation times of the video keyframes (for stream-copy cuts)."""
    ws = Path(workspace).resolve()
    opts, p = input_args(path, ws)
    exe = find_ffprobe()
    if exe:
        _, stdout, _ = _capture([exe, "-v", "error", *opts, "-select_streams", "v:0",
                                 "-show_entries", "packet=pts_time,flags", "-of", "csv=p=0", "file:" + str(p)],
                                ws, timeout=timeout)
        out = []
        for line in stdout.splitlines():
            t, _, flags = line.partition(",")
            if "K" in flags:
                try:
                    out.append(float(t))
                except ValueError:
                    pass
        return sorted(out)
    _, _, err = _capture([find_ffmpeg(), "-hide_banner", "-nostdin", "-skip_frame", "nokey", *opts,
                          "-i", "file:" + str(p), "-an", "-vf", "showinfo", "-f", "null", "-"], ws, timeout=timeout)
    return sorted(float(x) for x in re.findall(r"pts_time:([\d.]+)", err))


# ----------------------------------------------------------------------------- frame pipes

def decode_frames(path: str | Path, workspace: str | Path, size: tuple[int, int], *, fps: float | None = None,
                  start: float = 0.0, duration: float | None = None, max_frames: int = 3000,
                  cancelled: CancelFn | None = None, timeout: float = 3600.0) -> Iterator[np.ndarray]:
    """Yield RGB uint8 frames (H, W, 3) scaled to `size` = (W, H), optionally resampled to `fps`.
    Call probe() first; this only decodes. Stops after max_frames."""
    ws = Path(workspace).resolve()
    opts, p = input_args(path, ws)
    w, h = size
    vf = ([f"fps={fps:.6f}"] if fps else []) + [f"scale={w}:{h}:flags=area"]
    cmd = [find_ffmpeg(), "-hide_banner", "-nostdin", "-loglevel", "error"]
    if start:
        cmd += ["-ss", f"{start:.3f}"]
    cmd += [*opts, "-i", "file:" + str(p)]
    if duration:
        cmd += ["-t", f"{duration:.3f}"]
    cmd += ["-an", "-vf", ",".join(vf), "-frames:v", str(int(max_frames)), "-f", "rawvideo", "-pix_fmt", "rgb24",
            "pipe:1"]
    sp = _spawn(cmd, ws, timeout=timeout)
    proc = sp.proc
    tail: deque[bytes] = deque(maxlen=20)
    t_err = threading.Thread(target=lambda: [tail.append(x) for x in proc.stderr], daemon=True)  # type: ignore[union-attr]
    t_err.start()
    n = w * h * 3
    t0 = time.monotonic()
    count = 0
    try:
        assert proc.stdout is not None
        while count < max_frames:
            if cancelled and cancelled():
                raise MediaError("decoding cancelled")
            if time.monotonic() - t0 > timeout:
                raise MediaError(f"decoding timed out after {timeout:.0f} s")
            buf = proc.stdout.read(n)
            if len(buf) < n:
                break
            count += 1
            yield np.frombuffer(buf, np.uint8).reshape(h, w, 3)
    finally:
        if proc.poll() is None:
            proc.stdout.close()           # unblocks an FFmpeg stuck writing a full pipe
            sp.kill()
        proc.wait()
        sp.finish()
        t_err.join(timeout=2)
    if count == 0:
        msg = b"".join(tail).decode("utf-8", "replace").strip()
        raise MediaError(f"{p.name}: no frames decoded{(' (' + msg[-300:] + ')') if msg else ''}")


def encode_frames(frames: Iterable[np.ndarray], out: str | Path, workspace: str | Path, *, size: tuple[int, int],
                  fps: float, total_frames: int | None = None, crf: int = 18, preset: str = "veryfast",
                  progress: ProgressFn | None = None, cancelled: CancelFn | None = None,
                  timeout: float = 3600.0, label: str = "encoding") -> Path:
    """Encode RGB uint8 frames (H, W, 3) to an H.264 MP4 (yuv420p, faststart), no audio."""
    ws = Path(workspace).resolve()
    o = inside(out, ws)
    w, h = size
    cmd = [find_ffmpeg(), "-hide_banner", "-nostdin", "-y", "-loglevel", "error", "-f", "rawvideo",
           "-pix_fmt", "rgb24", "-s", f"{w}x{h}", "-r", f"{fps:g}", "-i", "pipe:0", *h264_args(crf, preset),
           "-movflags", "+faststart", str(o)]
    sp = _spawn(cmd, ws, stdin=True, timeout=timeout)
    proc = sp.proc
    tail: deque[bytes] = deque(maxlen=20)
    t_out = threading.Thread(target=lambda: [None for _ in proc.stdout], daemon=True)  # type: ignore[union-attr]
    t_err = threading.Thread(target=lambda: [tail.append(x) for x in proc.stderr], daemon=True)  # type: ignore[union-attr]
    t_out.start()
    t_err.start()
    t0 = time.monotonic()
    i = 0
    err = None
    try:
        assert proc.stdin is not None
        for fr in frames:
            if cancelled and cancelled():
                err = f"{label} cancelled"
                break
            if time.monotonic() - t0 > timeout:
                err = f"{label} timed out after {timeout:.0f} s"
                break
            fr = np.ascontiguousarray(fr, dtype=np.uint8)
            if fr.shape != (h, w, 3):
                err = f"{label}: frame {i} has shape {fr.shape}, expected {(h, w, 3)}"
                break
            proc.stdin.write(fr.tobytes())
            i += 1
            if progress and total_frames:
                progress(i / total_frames, label)
        proc.stdin.close()
    except BrokenPipeError:
        pass
    if err:
        sp.kill()
    try:
        proc.wait(timeout=max(30.0, timeout))
    except subprocess.TimeoutExpired:
        sp.kill()
        proc.wait()
    over = sp.finish()
    t_err.join(timeout=2)
    err = err or over
    if err or proc.returncode != 0:
        o.unlink(missing_ok=True)
        raise MediaError(err or f"{label} failed: {b''.join(tail).decode('utf-8', 'replace')[-400:]}")
    if i == 0:
        o.unlink(missing_ok=True)
        raise MediaError(f"{label}: no frames")
    return o
