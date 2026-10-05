"""Weights I/O with a safety-first policy.

  * Only safetensors (tensors + JSON header, no code) are written or read. Pickle-based
    checkpoints (.pt/.pth/.ckpt/.bin) are refused; `import_legacy` converts one with
    torch.load(weights_only=True) only when the user explicitly asks.
  * Every downloaded weight file must match the SHA-256 in a manifest, and the manifest
    must carry a valid Ed25519 signature from a key the app trusts (weights/trusted_keys.json).
  * Writes are atomic (temp file + os.replace), so a crash never leaves a half checkpoint.
"""
from __future__ import annotations

import base64
import hashlib
import json
import re
import os
import tempfile
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file

REFUSED_SUFFIXES = {".pt", ".pth", ".ckpt", ".bin", ".pkl", ".pickle", ".joblib"}


class WeightError(RuntimeError):
    pass


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


def atomic_write_bytes(path: str | Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-")
    with os.fdopen(fd, "wb") as f:
        f.write(data)
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, path)


def save_tensors(tensors: dict[str, torch.Tensor], path: str | Path, metadata: dict | None = None) -> str:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=path.parent, prefix=".tmp-", suffix=".safetensors")
    os.close(fd)
    clean = {k: v.detach().contiguous().cpu() for k, v in tensors.items()}
    save_file(clean, tmp, metadata={k: str(v) for k, v in (metadata or {}).items()})
    os.replace(tmp, path)
    return sha256_file(path)


def load_tensors(path: str | Path, expected_sha256: str | None = None) -> dict[str, torch.Tensor]:
    path = Path(path)
    if path.suffix.lower() in REFUSED_SUFFIXES:
        raise WeightError(f"{path.name}: pickle-based formats are refused. Convert it once with "
                          "`jepa-studio import-legacy` (weights-only) if you trust its source.")
    if path.suffix.lower() != ".safetensors":
        raise WeightError(f"{path.name}: only .safetensors weights are loaded")
    if expected_sha256 is not None:
        got = sha256_file(path)
        if got != expected_sha256.lower():
            raise WeightError(f"{path.name}: SHA-256 mismatch (expected {expected_sha256[:12]}..., got {got[:12]}...)")
    return load_file(str(path))


def save_state_dict(model: torch.nn.Module, path: str | Path, metadata: dict | None = None) -> str:
    return save_tensors(model.state_dict(), path, metadata)


def load_state_dict(model: torch.nn.Module, path: str | Path, expected_sha256: str | None = None,
                    strict: bool = True) -> None:
    model.load_state_dict(load_tensors(path, expected_sha256), strict=strict)


# ------------------------------------------------------------- run weights, verified on load

RUN_WEIGHTS = "encoder.safetensors"
RUN_WEIGHTS_HASH = "encoder.sha256"   # `sha256sum` format, written next to the weights


def write_run_hash(run_dir: str | Path, sha: str) -> None:
    """Record the SHA-256 of a run's final weights (also checkable with `sha256sum -c`)."""
    atomic_write_bytes(Path(run_dir) / RUN_WEIGHTS_HASH, f"{sha}  {RUN_WEIGHTS}\n".encode())


def recorded_run_hash(run_dir: str | Path) -> str | None:
    """The SHA-256 the run recorded for its weights: encoder.sha256, else the `end` event of
    events.jsonl (runs made before encoder.sha256 existed). None if the run recorded nothing."""
    d = Path(run_dir)
    f = d / RUN_WEIGHTS_HASH
    if f.exists():
        h = f.read_text().split()[0].strip().lower() if f.read_text().strip() else ""
        if not re.fullmatch(r"[0-9a-f]{64}", h):
            raise WeightError(f"{f}: not a SHA-256 line")
        return h
    ev = d / "events.jsonl"
    if ev.exists():
        sha = None
        for line in ev.read_text().splitlines():
            if '"encoder_sha256"' in line:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if rec.get("event") == "end" and isinstance(rec.get("encoder_sha256"), str):
                    sha = rec["encoder_sha256"].lower()
        return sha
    return None


def load_run_weights(model: torch.nn.Module, run_dir: str | Path, require_record: bool = False) -> bool:
    """Load <run>/encoder.safetensors, checking it against the hash the run recorded.

    Returns True when a recorded hash was checked. A mismatch raises WeightError (the file is
    corrupted or not the one this run produced). A run with no record loads unverified (False)
    unless require_record. This is integrity, not authenticity: whoever can edit the weights can
    edit the record too. For weights from someone else, use signed manifests (sign-weights).
    """
    d = Path(run_dir)
    sha = recorded_run_hash(d)
    if sha is None and require_record:
        raise WeightError(f"{d}: no recorded SHA-256 for {RUN_WEIGHTS} (encoder.sha256 or an end event)")
    load_state_dict(model, d / RUN_WEIGHTS, expected_sha256=sha)
    return sha is not None


# ------------------------------------------------------------- optimizer state (no pickle)

def optimizer_to_tensors(opt: torch.optim.Optimizer) -> tuple[dict[str, torch.Tensor], dict]:
    sd = opt.state_dict()
    tensors, scalars = {}, {"param_groups": sd["param_groups"], "state": {}}
    for pid, st in sd["state"].items():
        scalars["state"][str(pid)] = {}
        for k, v in st.items():
            if torch.is_tensor(v):
                tensors[f"{pid}.{k}"] = v
            else:
                scalars["state"][str(pid)][k] = v
    return tensors, scalars


def optimizer_from_tensors(opt: torch.optim.Optimizer, tensors: dict, scalars: dict) -> None:
    state: dict = {}
    for pid, st in scalars["state"].items():
        state[int(pid)] = dict(st)
    for key, v in tensors.items():
        pid, k = key.split(".", 1)
        state.setdefault(int(pid), {})[k] = v
    opt.load_state_dict({"state": state, "param_groups": scalars["param_groups"]})


def import_legacy(src: str | Path, dst: str | Path) -> str:
    """Explicit, opt-in conversion of a legacy checkpoint using weights-only unpickling."""
    obj = torch.load(str(src), map_location="cpu", weights_only=True)
    if isinstance(obj, dict) and "state_dict" in obj:
        obj = obj["state_dict"]
    if not isinstance(obj, dict) or not all(torch.is_tensor(v) for v in obj.values()):
        raise WeightError("legacy file does not contain a flat tensor dict")
    return save_tensors(obj, dst, {"converted_from": Path(src).name})


# ------------------------------------------------------------- signed manifests

def _canonical(manifest: dict) -> bytes:
    body = {k: v for k, v in manifest.items() if k != "signature"}
    return json.dumps(body, sort_keys=True, separators=(",", ":")).encode()


def generate_keypair() -> tuple[str, str]:
    from cryptography.hazmat.primitives import serialization
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sk = Ed25519PrivateKey.generate()
    raw_sk = sk.private_bytes(serialization.Encoding.Raw, serialization.PrivateFormat.Raw,
                              serialization.NoEncryption())
    raw_pk = sk.public_key().public_bytes(serialization.Encoding.Raw, serialization.PublicFormat.Raw)
    return base64.b64encode(raw_sk).decode(), base64.b64encode(raw_pk).decode()


def sign_manifest(manifest: dict, private_key_b64: str) -> dict:
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey

    sk = Ed25519PrivateKey.from_private_bytes(base64.b64decode(private_key_b64))
    out = dict(manifest)
    out["signature"] = base64.b64encode(sk.sign(_canonical(manifest))).decode()
    return out


def verify_manifest(manifest: dict, trusted_public_keys_b64: list[str]) -> bool:
    from cryptography.exceptions import InvalidSignature
    from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PublicKey

    sig = manifest.get("signature")
    if not sig:
        return False
    for pk in trusted_public_keys_b64:
        try:
            Ed25519PublicKey.from_public_bytes(base64.b64decode(pk)).verify(base64.b64decode(sig), _canonical(manifest))
            return True
        except (InvalidSignature, ValueError):
            continue
    return False


def build_manifest(files: list[str | Path], base: str | Path, extra: dict | None = None) -> dict:
    base = Path(base)
    entries = []
    for f in files:
        f = Path(f)
        entries.append({"path": f.relative_to(base).as_posix(), "sha256": sha256_file(f), "bytes": f.stat().st_size})
    return {"format": "jepa-studio-manifest/v1", "files": entries, **(extra or {})}


def verify_files(manifest: dict, base: str | Path, trusted_keys: list[str]) -> list[str]:
    """Return problems (empty list = everything verified)."""
    problems = []
    if not verify_manifest(manifest, trusted_keys):
        problems.append("manifest signature is missing or not from a trusted key")
    base = Path(base).resolve()
    for e in manifest.get("files", []):
        p = (base / e["path"]).resolve()
        if base not in p.parents:
            problems.append(f"{e['path']}: path escapes the weights folder")
            continue
        if not p.exists():
            problems.append(f"{e['path']}: missing")
            continue
        if "bytes" in e and p.stat().st_size != int(e["bytes"]):
            problems.append(f"{e['path']}: size {p.stat().st_size} != {e['bytes']} in the manifest")
        if sha256_file(p) != e["sha256"]:
            problems.append(f"{e['path']}: SHA-256 mismatch")
    return problems
