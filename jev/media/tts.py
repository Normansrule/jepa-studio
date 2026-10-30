"""Local text-to-speech. Nothing leaves the machine and nothing is downloaded without approval.

Backends, tried in this order ("auto"):
  1. piper     neural voices; the `piper` CLI or the `piper-tts` Python package, plus a voice
               model (.onnx + .onnx.json) you downloaded into the voices folder (see download_voice)
  2. espeak-ng formant synthesis; robotic but clear, tiny, on every Linux distro
  3. espeak    the older espeak, same interface
  4. say       macOS built-in voices
  5. sapi      Windows built-in voices through Windows PowerShell + System.Speech
  6. silent    no engine found: a silent track with the estimated speaking time; the narration
               is still shown as captions, and the result says so (silent=True)

Every result is a WAV file plus the backend name and the measured duration.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import urllib.parse
import urllib.request
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np

from .common import MediaError, inside

BACKENDS = ("piper", "espeak-ng", "espeak", "say", "sapi", "silent")
MAX_TEXT_CHARS = 20_000
SILENT_RATE = 22050
WORDS_PER_SECOND = 2.6              # used to size the silent fallback (about 155 words per minute)

# Piper voices are published at huggingface.co/rhasspy/piper-voices. The revision is pinned so a
# voice name always means the same files; voices.json at that revision lists each file's size and
# MD5, which download_voice checks before recording a SHA-256 in the local manifest.
PIPER_REPO = "https://huggingface.co/rhasspy/piper-voices/resolve/v1.0.0"
PIPER_HOSTS = ("huggingface.co", "hf.co")       # and their subdomains (the LFS/CDN redirects)
MAX_VOICE_BYTES = 300 * 1024 * 1024
RECOMMENDED_VOICES = {
    "en_US-lessac-medium": "US English, clear neutral voice (about 63 MB)",
    "en_US-amy-medium": "US English, female (about 63 MB)",
    "en_US-ryan-high": "US English, male, higher quality (about 120 MB)",
    "en_GB-alan-medium": "British English, male (about 63 MB)",
}
_VOICE_RE = re.compile(r"^([a-z]{2,3})_([A-Z]{2})-([A-Za-z0-9_]+)-(x_low|low|medium|high)$")
MANIFEST = "voices-manifest.json"

# Python-package detection is a module attribute so tests can switch it off.
PIPER_PYTHON = importlib.util.find_spec("piper") is not None


@dataclass
class TTSResult:
    path: Path
    backend: str
    duration_s: float
    sample_rate: int
    silent: bool = False
    voice: str | None = None
    notes: list[str] = field(default_factory=list)

    def to_json(self) -> dict:
        return {"path": str(self.path), "backend": self.backend, "duration_s": round(self.duration_s, 3),
                "sample_rate": self.sample_rate, "silent": self.silent, "voice": self.voice, "notes": self.notes}


# ----------------------------------------------------------------------------- voices folder

def voices_dir(settings: dict | None = None) -> Path:
    media = (settings or {}).get("media", {}) or {}
    if media.get("voices_dir"):
        return Path(media["voices_dir"]).expanduser()
    if os.environ.get("JEV_VOICES_DIR"):
        return Path(os.environ["JEV_VOICES_DIR"]).expanduser()
    home = Path(os.environ.get("JEV_HOME", Path.home() / ".jev")).expanduser()
    return home / "voices"


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(1 << 20):
            h.update(b)
    return h.hexdigest()


def _manifest(vdir: Path) -> dict:
    try:
        return json.loads((vdir / MANIFEST).read_text())
    except (OSError, json.JSONDecodeError):
        return {}


def list_voices(settings: dict | None = None) -> list[dict]:
    vdir = voices_dir(settings)
    if not vdir.is_dir():
        return []
    man = _manifest(vdir)
    out = []
    for onnx in sorted(vdir.glob("*.onnx")):
        out.append({"name": onnx.stem, "path": str(onnx), "config": (onnx.parent / (onnx.name + ".json")).is_file(),
                    "bytes": onnx.stat().st_size, "recorded_sha256": man.get(onnx.name, {}).get("sha256")})
    return out


def find_voice(voice: str | None, settings: dict | None = None, verify: bool = True) -> Path | None:
    """The .onnx file for `voice` (a name like en_US-lessac-medium; None = first voice found).
    A recorded SHA-256 (written by download_voice) is checked; a mismatch raises."""
    vdir = voices_dir(settings)
    if not vdir.is_dir():
        return None
    if voice:
        if not re.fullmatch(r"[A-Za-z0-9_.\-]+", voice):
            raise MediaError(f"invalid voice name {voice!r}")
        cand = vdir / (voice if voice.endswith(".onnx") else voice + ".onnx")
        cands = [cand] if cand.is_file() else []
    else:
        cands = sorted(vdir.glob("*.onnx"))
    for c in cands:
        if not (c.parent / (c.name + ".json")).is_file():
            continue
        rec = _manifest(vdir).get(c.name, {}).get("sha256")
        if verify and rec and _sha256(c) != rec:
            raise MediaError(f"voice {c.name}: SHA-256 does not match the one recorded at download; "
                             "delete it and download it again")
        return c
    return None


# ----------------------------------------------------------------------------- backends

def _which(name: str) -> str | None:
    return shutil.which(name, path=os.environ.get("PATH", ""))


def _powershell() -> str | None:
    if sys.platform != "win32":
        return None
    return _which("powershell") or _which("powershell.exe")


def available_backends(settings: dict | None = None) -> list[dict]:
    """What can speak on this machine, in fallback order."""
    out = []
    piper_exe = _which("piper")
    try:
        voice = find_voice(None, settings, verify=False)
    except MediaError:
        voice = None
    piper_ok = bool((piper_exe or PIPER_PYTHON) and voice)
    detail = ("piper CLI" if piper_exe else "piper-tts package" if PIPER_PYTHON else "not installed")
    detail += f"; voice: {voice.stem if voice else 'none in ' + str(voices_dir(settings))}"
    out.append({"name": "piper", "available": piper_ok, "detail": detail})
    for name in ("espeak-ng", "espeak", "say"):
        exe = _which(name)
        if name == "say" and sys.platform != "darwin":
            exe = None
        out.append({"name": name, "available": bool(exe), "detail": exe or "not found"})
    ps = _powershell()
    out.append({"name": "sapi", "available": bool(ps), "detail": ps or "Windows only"})
    out.append({"name": "silent", "available": True, "detail": "silent track + captions"})
    return out


def wav_duration(path: Path) -> tuple[float, int]:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate()), w.getframerate()
    except (wave.Error, EOFError, OSError):
        from . import ffmpeg as F

        info = F.probe(path, path.parent, kind="audio", check=False)
        return float(info.duration_s), int(info.audio_rate or 0)


def write_silence(path: Path, seconds: float, rate: int = SILENT_RATE) -> None:
    n = max(1, int(round(seconds * rate)))
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(np.zeros(n, np.int16).tobytes())


def estimate_seconds(text: str) -> float:
    words = len(re.findall(r"\w+", text))
    return max(1.2, words / WORDS_PER_SECOND + 0.4)


def clean_text(text: str) -> str:
    text = re.sub(r"[\x00-\x08\x0b-\x1f\x7f]", " ", text or "")
    text = re.sub(r"\s+", " ", text).strip()
    if len(text) > MAX_TEXT_CHARS:
        raise MediaError(f"text is {len(text)} characters; split it into parts under {MAX_TEXT_CHARS}")
    return text


def _run(cmd: list[str], timeout: float, stdin: str | None = None) -> None:
    r = subprocess.run(cmd, input=stdin, capture_output=True, text=True, timeout=timeout,
                       encoding="utf-8", errors="replace")
    if r.returncode != 0:
        raise MediaError(f"{Path(cmd[0]).name} failed: {(r.stderr or r.stdout).strip()[-300:]}")


def _piper(text: str, out: Path, voice: Path, timeout: float, rate: float) -> None:
    exe = _which("piper")
    length_scale = f"{1.0 / max(0.5, min(2.0, rate)):.3f}"
    if exe:
        _run([exe, "-m", str(voice), "-f", str(out), "--length_scale", length_scale], timeout, stdin=text + "\n")
        return
    from piper import PiperVoice  # type: ignore[import-not-found]

    pv = PiperVoice.load(str(voice))
    with wave.open(str(out), "wb") as wf:
        if hasattr(pv, "synthesize_wav"):          # piper-tts >= 1.3
            pv.synthesize_wav(text, wf)
        else:                                      # piper-tts 1.2
            pv.synthesize(text, wf, length_scale=float(length_scale))


def _espeak(exe: str, text_file: Path, out: Path, voice: str | None, timeout: float, rate: float) -> None:
    cmd = [exe, "-s", str(int(165 * rate)), "-w", str(out), "-f", str(text_file)]
    if voice and re.fullmatch(r"[A-Za-z0-9_+\-]+", voice):
        cmd[1:1] = ["-v", voice]
    _run(cmd, timeout)


def _say(text_file: Path, out: Path, voice: str | None, timeout: float, rate: float) -> None:
    cmd = ["say", "-r", str(int(180 * rate)), "-o", str(out), "--file-format=WAVE", "--data-format=LEI16@22050",
           "-f", str(text_file)]
    if voice and re.fullmatch(r"[A-Za-z0-9 _\-()]+", voice):
        cmd[1:1] = ["-v", voice]
    _run(cmd, timeout)


_SAPI_PS1 = r"""param([string]$TextPath, [string]$WavPath, [int]$Rate)
Add-Type -AssemblyName System.Speech
$s = New-Object System.Speech.Synthesis.SpeechSynthesizer
$s.Rate = $Rate
$s.SetOutputToWaveFile($WavPath)
$s.Speak([IO.File]::ReadAllText($TextPath, [Text.Encoding]::UTF8))
$s.Dispose()
"""


def _sapi(ps: str, text_file: Path, out: Path, tmp: Path, timeout: float, rate: float) -> None:
    script = tmp / "jev_sapi.ps1"
    script.write_text(_SAPI_PS1, encoding="utf-8")
    sapi_rate = str(int(max(-10, min(10, round((rate - 1.0) * 10)))))
    _run([ps, "-NoProfile", "-NonInteractive", "-ExecutionPolicy", "Bypass", "-File", str(script),
          str(text_file), str(out), sapi_rate], timeout)


def synthesize(text: str, out_wav: str | Path, workspace: str | Path, *, backend: str = "auto",
               voice: str | None = None, rate: float = 1.0, settings: dict | None = None,
               timeout: float = 600.0) -> TTSResult:
    """Speak `text` into `out_wav` (inside the workspace) with the first backend that works.

    backend: "auto" or one of BACKENDS (a named backend that is unavailable falls through to the
    rest of the chain, and the notes say so). voice: a Piper voice name, an espeak voice
    (e.g. "en-us"), or a macOS voice name, depending on the backend used."""
    if backend != "auto" and backend not in BACKENDS:
        raise MediaError(f"unknown speech backend {backend!r}; choose one of {', '.join(BACKENDS)}")
    ws = Path(workspace).resolve()
    out = inside(out_wav, ws)
    if out.suffix.lower() != ".wav":
        out = out.with_suffix(".wav")
    out.parent.mkdir(parents=True, exist_ok=True)
    text = clean_text(text)
    notes: list[str] = []
    if not text:
        write_silence(out, 0.5)
        return TTSResult(out, "silent", 0.5, SILENT_RATE, True, None, ["empty text"])
    tmp = out.parent / ".jev_tts"
    tmp.mkdir(exist_ok=True)
    text_file = tmp / (out.stem + ".txt")
    text_file.write_text(text, encoding="utf-8")
    order = list(BACKENDS) if backend == "auto" else [backend] + [b for b in BACKENDS if b != backend]
    try:
        for b in order:
            try:
                if b == "piper":
                    if not (_which("piper") or PIPER_PYTHON):
                        notes.append("piper: not installed")
                        continue
                    v = find_voice(voice if voice and _VOICE_RE.match(voice) else None, settings)
                    if v is None:
                        notes.append(f"piper: no voice model in {voices_dir(settings)} (see download_voice)")
                        continue
                    _piper(text, out, v, timeout, rate)
                    used_voice = v.stem
                elif b in ("espeak-ng", "espeak"):
                    exe = _which(b)
                    if not exe:
                        notes.append(f"{b}: not found")
                        continue
                    ev = voice if voice and not _VOICE_RE.match(voice) else None
                    _espeak(exe, text_file, out, ev, timeout, rate)
                    used_voice = ev or "default"
                elif b == "say":
                    if sys.platform != "darwin" or not _which("say"):
                        continue
                    _say(text_file, out, voice, timeout, rate)
                    used_voice = voice or "system default"
                elif b == "sapi":
                    ps = _powershell()
                    if not ps:
                        continue
                    _sapi(ps, text_file, out, tmp, timeout, rate)
                    used_voice = "system default"
                else:
                    secs = estimate_seconds(text) / max(0.5, rate)
                    write_silence(out, secs)
                    notes.append("no speech engine available: wrote a silent track of the estimated speaking "
                                 "time; the narration is carried by captions")
                    return TTSResult(out, "silent", secs, SILENT_RATE, True, None, notes)
                if not out.is_file() or out.stat().st_size < 64:
                    notes.append(f"{b}: produced no audio")
                    continue
                dur, sr = wav_duration(out)
                if dur <= 0:
                    notes.append(f"{b}: produced empty audio")
                    continue
                return TTSResult(out, b, dur, sr, False, used_voice, notes)
            except (MediaError, OSError, subprocess.TimeoutExpired, ImportError) as e:
                notes.append(f"{b}: {str(e)[:200]}")
                continue
    finally:
        text_file.unlink(missing_ok=True)
    raise MediaError("speech synthesis failed: " + "; ".join(notes))  # unreachable: silent always works


# ----------------------------------------------------------------------------- voice download

def voice_urls(name: str) -> dict[str, str]:
    m = _VOICE_RE.match(name)
    if not m:
        raise MediaError(f"{name!r} is not a Piper voice name like en_US-lessac-medium")
    lang, region, speaker, quality = m.groups()
    base = f"{lang}/{lang}_{region}/{speaker}/{quality}/{name}"
    return {f"{name}.onnx": f"{base}.onnx", f"{name}.onnx.json": f"{base}.onnx.json"}


def _host_ok(url: str) -> bool:
    u = urllib.parse.urlparse(url)
    host = (u.hostname or "").lower()
    return u.scheme == "https" and any(host == h or host.endswith("." + h) for h in PIPER_HOSTS)


def _fetch_https(url: str, max_bytes: int, on_chunk: Callable[[int], None] | None = None) -> bytes:
    if not _host_ok(url):
        raise MediaError(f"refusing to download from {url}: only https://huggingface.co is allowed for voices")
    req = urllib.request.Request(url, headers={"User-Agent": "jev-media/1"})
    with urllib.request.urlopen(req, timeout=60) as r:  # noqa: S310 - scheme and host checked above
        if not _host_ok(r.geturl()):
            raise MediaError(f"download redirected to an unexpected host: {r.geturl()}")
        chunks, total = [], 0
        while b := r.read(1 << 20):
            total += len(b)
            if total > max_bytes:
                raise MediaError(f"download larger than {max_bytes >> 20} MB; stopped")
            chunks.append(b)
            if on_chunk:
                on_chunk(total)
        return b"".join(chunks)


def download_voice(name: str, settings: dict | None = None, *, fetch: Callable[..., bytes] | None = None,
                   progress: Callable[[float, str], None] | None = None) -> dict:
    """Download a Piper voice (.onnx + .onnx.json) into the voices folder. Approval-gated by the
    tool layer. Each file's size and MD5 must match voices.json from the same pinned revision;
    the SHA-256 of what was saved is recorded in voices-manifest.json and checked on every use."""
    fetch = fetch or _fetch_https
    files = voice_urls(name)
    vdir = voices_dir(settings)
    vdir.mkdir(parents=True, exist_ok=True)
    index = json.loads(fetch(f"{PIPER_REPO}/voices.json", 20 << 20).decode("utf-8"))
    entry = index.get(name)
    if not entry:
        raise MediaError(f"voice {name!r} is not in the Piper voices index (v1.0.0)")
    listed = entry.get("files", {})
    man = _manifest(vdir)
    saved = {}
    for i, (fname, rel) in enumerate(files.items()):
        meta = listed.get(rel)
        if not meta:
            raise MediaError(f"{rel} is not listed in voices.json; refusing an unlisted file")
        if progress:
            progress(i / len(files), f"downloading {fname}")
        data = fetch(f"{PIPER_REPO}/{rel}", MAX_VOICE_BYTES)
        if "size_bytes" in meta and len(data) != int(meta["size_bytes"]):
            raise MediaError(f"{fname}: size {len(data)} does not match voices.json ({meta['size_bytes']})")
        if "md5_digest" in meta and hashlib.md5(data).hexdigest() != meta["md5_digest"]:  # noqa: S324 - integrity vs index
            raise MediaError(f"{fname}: MD5 does not match voices.json; not saved")
        tmp = vdir / (".part-" + fname)
        tmp.write_bytes(data)
        os.replace(tmp, vdir / fname)
        sha = hashlib.sha256(data).hexdigest()
        man[fname] = {"sha256": sha, "bytes": len(data), "source": f"{PIPER_REPO}/{rel}"}
        saved[fname] = sha
    (vdir / MANIFEST).write_text(json.dumps(man, indent=2, sort_keys=True))
    if progress:
        progress(1.0, "voice ready")
    return {"voice": name, "folder": str(vdir), "files": saved}
