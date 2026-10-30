"""Speech fallback chain (Piper -> espeak-ng -> espeak -> say -> SAPI -> silent), simulated with PATH."""
import hashlib
import json
import shutil
import sys
import wave

import numpy as np
import pytest

from jev.media import tts as T
from jev.media.common import MediaError

FAKE_WAV = """#!{py}
import sys, wave, math, struct
args = sys.argv[1:]
out = None
for flag in ("-w", "-f", "-o", "--output_file"):
    if flag in args and args.index(flag) + 1 < len(args):
        cand = args[args.index(flag) + 1]
        if cand.endswith(".wav"):
            out = cand
if out is None:                       # positional output (the SAPI PowerShell script)
    out = next(a for a in args if a.endswith(".wav"))
if "{name}" == "piper":
    sys.stdin.read()
with wave.open(out, "wb") as w:
    w.setnchannels(1); w.setsampwidth(2); w.setframerate(16000)
    w.writeframes(b"".join(struct.pack("<h", int(8000 * math.sin(i / 8))) for i in range({n})))
open(out + ".called", "w").write(" ".join(args))
"""


def fake_bin(d, name, seconds=1.0):
    d.mkdir(exist_ok=True)
    p = d / name
    p.write_text(FAKE_WAV.format(py=sys.executable, name=name, n=int(16000 * seconds)))
    p.chmod(0o755)
    return p


@pytest.fixture
def isolated(tmp_path, monkeypatch):
    """Empty PATH + no piper Python package + an empty voices folder."""
    bin_dir = tmp_path / "bin"
    bin_dir.mkdir()
    monkeypatch.setenv("PATH", str(bin_dir))
    monkeypatch.setattr(T, "PIPER_PYTHON", False)
    monkeypatch.setattr(T.sys, "platform", "linux")
    ws = tmp_path / "ws"
    ws.mkdir()
    settings = {"media": {"voices_dir": str(tmp_path / "voices")}}
    return bin_dir, ws, settings


def test_silent_fallback(isolated):
    _, ws, settings = isolated
    r = T.synthesize("one two three four five six seven", "a.wav", ws, settings=settings)
    assert r.backend == "silent" and r.silent
    assert abs(r.duration_s - T.estimate_seconds("one two three four five six seven")) < 0.01
    with wave.open(str(r.path)) as w:
        assert w.getnframes() > 0
    assert any("no speech engine" in n for n in r.notes)
    assert {b["name"]: b["available"] for b in T.available_backends(settings)}["espeak-ng"] is False


def test_fake_espeak_ng_used(isolated):
    bin_dir, ws, settings = isolated
    fake_bin(bin_dir, "espeak-ng", 1.5)
    r = T.synthesize("hello there", "b.wav", ws, settings=settings)
    assert r.backend == "espeak-ng" and not r.silent
    assert abs(r.duration_s - 1.5) < 0.01
    called = (ws / "b.wav.called").read_text()
    assert "-f" in called and "hello" not in called      # text goes through a file, never argv


def test_espeak_legacy_then_say_on_macos(isolated, monkeypatch):
    bin_dir, ws, settings = isolated
    fake_bin(bin_dir, "say", 0.5)
    monkeypatch.setattr(T.sys, "platform", "darwin")
    r = T.synthesize("hi", "c.wav", ws, settings=settings)
    assert r.backend == "say"
    fake_bin(bin_dir, "espeak", 0.7)
    assert T.synthesize("hi", "d.wav", ws, settings=settings).backend == "espeak"


def test_sapi_on_windows(isolated, monkeypatch):
    bin_dir, ws, settings = isolated
    ps = fake_bin(bin_dir, "powershell", 0.6)
    monkeypatch.setattr(T, "_powershell", lambda: str(ps))   # shutil.which follows PATHEXT rules on win32
    r = T.synthesize("hello windows", "e.wav", ws, settings=settings)
    assert r.backend == "sapi"
    called = (ws / "e.wav.called").read_text()
    assert "-File" in called and "hello" not in called


def _voice(settings, name="en_US-test-medium", record_sha=True, bad=False):
    vdir = T.voices_dir(settings)
    vdir.mkdir(parents=True, exist_ok=True)
    onnx = vdir / f"{name}.onnx"
    onnx.write_bytes(b"fake onnx model")
    (vdir / f"{name}.onnx.json").write_text("{}")
    if record_sha:
        sha = hashlib.sha256(b"different" if bad else b"fake onnx model").hexdigest()
        (vdir / T.MANIFEST).write_text(json.dumps({onnx.name: {"sha256": sha}}))
    return onnx


def test_piper_preferred_when_voice_installed(isolated):
    bin_dir, ws, settings = isolated
    fake_bin(bin_dir, "espeak-ng")
    fake_bin(bin_dir, "piper", 2.0)
    r = T.synthesize("hello", "f.wav", ws, settings=settings)
    assert r.backend == "espeak-ng" and any("no voice model" in n for n in r.notes)
    _voice(settings)
    r = T.synthesize("hello", "g.wav", ws, settings=settings)
    assert r.backend == "piper" and r.voice == "en_US-test-medium"
    assert "-m" in (ws / "g.wav.called").read_text()


def test_piper_voice_checksum_mismatch_falls_back(isolated):
    bin_dir, ws, settings = isolated
    fake_bin(bin_dir, "espeak-ng")
    fake_bin(bin_dir, "piper")
    _voice(settings, bad=True)
    r = T.synthesize("hello", "h.wav", ws, settings=settings)
    assert r.backend == "espeak-ng"
    assert any("SHA-256" in n for n in r.notes)


def test_output_must_stay_in_workspace(isolated):
    _, ws, settings = isolated
    with pytest.raises(MediaError):
        T.synthesize("x", "../escape.wav", ws, settings=settings)
    with pytest.raises(MediaError):
        T.synthesize("x", "a.wav", ws, backend="festival", settings=settings)


@pytest.mark.skipif(not shutil.which("espeak-ng"), reason="espeak-ng not installed on this machine")
def test_real_espeak_ng(tmp_path):
    r = T.synthesize("JEV speaks with espeak.", "real.wav", tmp_path, backend="espeak-ng")
    assert r.backend == "espeak-ng" and 0.8 < r.duration_s < 5
    with wave.open(str(r.path)) as w:
        x = np.frombuffer(w.readframes(w.getnframes()), np.int16)
    assert np.abs(x).mean() > 100              # actual speech, not silence


def test_download_voice_checks_index(tmp_path):
    settings = {"media": {"voices_dir": str(tmp_path / "voices")}}
    name = "en_US-lessac-medium"
    files = T.voice_urls(name)
    blobs = {files[f"{name}.onnx"]: b"model-bytes", files[f"{name}.onnx.json"]: b"{}"}
    index = {name: {"files": {rel: {"size_bytes": len(b), "md5_digest": hashlib.md5(b).hexdigest()}
                              for rel, b in blobs.items()}}}
    seen = []

    def fetch(url, max_bytes):
        seen.append(url)
        assert url.startswith(T.PIPER_REPO)
        if url.endswith("voices.json"):
            return json.dumps(index).encode()
        return blobs[url[len(T.PIPER_REPO) + 1:]]

    r = T.download_voice(name, settings, fetch=fetch)
    vdir = tmp_path / "voices"
    assert (vdir / f"{name}.onnx").read_bytes() == b"model-bytes"
    man = json.loads((vdir / T.MANIFEST).read_text())
    assert man[f"{name}.onnx"]["sha256"] == hashlib.sha256(b"model-bytes").hexdigest() == r["files"][f"{name}.onnx"]
    assert T.find_voice(name, settings) == vdir / f"{name}.onnx"

    blobs[files[f"{name}.onnx"]] = b"tampered!!!"          # same size, wrong MD5
    with pytest.raises(MediaError, match="MD5"):
        T.download_voice(name, {"media": {"voices_dir": str(tmp_path / "v2")}}, fetch=fetch)
    assert not (tmp_path / "v2" / f"{name}.onnx").exists()
    with pytest.raises(MediaError):
        T.download_voice("not a voice", settings, fetch=fetch)


def test_download_hosts_restricted():
    assert T._host_ok("https://huggingface.co/rhasspy/x")
    assert T._host_ok("https://cdn-lfs.huggingface.co/x")
    assert not T._host_ok("http://huggingface.co/x")
    assert not T._host_ok("https://huggingface.co.evil.com/x")
    with pytest.raises(MediaError):
        T._fetch_https("https://example.com/voice.onnx", 10)
