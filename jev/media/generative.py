"""Optional local generative models through Hugging Face `diffusers` (install `jepa-studio[gen]`).

Written against the documented diffusers pipeline APIs (AutoPipelineForText2Image / FluxPipeline /
LTXPipeline / WanPipeline `from_pretrained(..., torch_dtype=, use_safetensors=True)` and
`pipe(prompt=..., generator=..., num_inference_steps=..., guidance_scale=...)` returning `.images`
or `.frames[0]`), checked against diffusers 0.42; LTXPipeline needs >= 0.32 and WanPipeline >= 0.33.

Strict rules:
  * weights come only from a local folder you chose (settings media.models_dir / media.model_dirs)
    or from download_model, which the tool layer always asks you to approve
  * safetensors only: a single pickle checkpoint (.bin/.ckpt/.pt/.pth/...) is refused with the same
    policy as jepa_studio.weights, a component that has only pickle weights is refused, and loading
    passes use_safetensors=True so stray pickle files are never read; custom pipeline code
    (a model_index.json class outside the allowlist) is refused
  * loading is offline (HF_HUB_OFFLINE=1, local_files_only=True)
  * GPU/VRAM detection with "needs ~N GB VRAM" messages; text-to-video on CPU is refused
  * deterministic: a fixed seed gives the same output on the same hardware and versions

A test or embedding app can inject a pipeline with set_pipeline_factory() (see the tests).
"""
from __future__ import annotations

import hashlib
import inspect
import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

import numpy as np
from PIL import Image

from jepa_studio.weights import REFUSED_SUFFIXES

from . import ffmpeg as F
from .common import MediaError, even, inside
from .images import save_png

MANIFEST = "jev-model.json"
ALLOWED_PIPELINES = {"StableDiffusionXLPipeline", "StableDiffusionPipeline", "StableDiffusion3Pipeline",
                     "FluxPipeline", "LTXPipeline", "WanPipeline"}


@dataclass(frozen=True)
class ModelSpec:
    name: str
    kind: str                       # image | video
    repo: str
    pipeline: str                   # diffusers class used to load
    class_names: tuple[str, ...]    # model_index.json _class_name values it accepts
    vram_gb: float                  # approx. peak VRAM fully on the GPU (fp16/bf16)
    vram_gb_offload: float          # approx. with enable_model_cpu_offload()
    download_gb: float
    defaults: dict = field(default_factory=dict)
    variant: str | None = None
    cpu_ok: bool = False            # usable (slowly) without a GPU
    license: str = ""
    size_multiple: int = 8
    frames_rule: tuple[int, int] | None = None   # num_frames = k * a + b


CATALOG: dict[str, ModelSpec] = {
    "sdxl-turbo": ModelSpec(
        "sdxl-turbo", "image", "stabilityai/sdxl-turbo", "AutoPipelineForText2Image", ("StableDiffusionXLPipeline",),
        vram_gb=8, vram_gb_offload=5, download_gb=7, variant="fp16", cpu_ok=True,
        defaults={"steps": 1, "guidance": 0.0, "width": 512, "height": 512},
        license="Stability AI Community License (read it before commercial use)"),
    "flux-schnell": ModelSpec(
        "flux-schnell", "image", "black-forest-labs/FLUX.1-schnell", "FluxPipeline", ("FluxPipeline",),
        vram_gb=33, vram_gb_offload=24, download_gb=34, size_multiple=16,
        defaults={"steps": 4, "guidance": 0.0, "width": 1024, "height": 768, "max_sequence_length": 256},
        license="Apache-2.0"),
    "ltx-video": ModelSpec(
        "ltx-video", "video", "Lightricks/LTX-Video", "LTXPipeline", ("LTXPipeline",),
        vram_gb=18, vram_gb_offload=10, download_gb=28, size_multiple=32, frames_rule=(8, 1),
        defaults={"steps": 40, "guidance": 3.0, "width": 704, "height": 480, "frames": 97, "fps": 24,
                  "decode_timestep": 0.03, "decode_noise_scale": 0.025},
        license="LTX-Video open weights license (see the model card)"),
    "wan2.1-t2v-1.3b": ModelSpec(
        "wan2.1-t2v-1.3b", "video", "Wan-AI/Wan2.1-T2V-1.3B-Diffusers", "WanPipeline", ("WanPipeline",),
        vram_gb=14, vram_gb_offload=8.2, download_gb=30, size_multiple=16, frames_rule=(4, 1),
        defaults={"steps": 30, "guidance": 5.0, "width": 832, "height": 480, "frames": 81, "fps": 16},
        license="Apache-2.0"),
}

# --------------------------------------------------------------------------- injection hook

PipelineFactory = Callable[[Path, ModelSpec, dict], Any]
_FACTORY: PipelineFactory | None = None


def set_pipeline_factory(factory: PipelineFactory | None) -> None:
    """Replace how pipelines are built (tests inject a tiny fake; None restores diffusers)."""
    global _FACTORY
    _FACTORY = factory


# --------------------------------------------------------------------------- hardware

def detect_gpu() -> dict:
    """{backend: cuda|mps|none, name, vram_gb, free_gb}. MPS reports unified memory."""
    try:
        import torch
    except ImportError:
        return {"backend": "none", "name": "no torch", "vram_gb": 0.0, "free_gb": 0.0}
    if torch.cuda.is_available():
        i = torch.cuda.current_device()
        props = torch.cuda.get_device_properties(i)
        try:
            free, total = torch.cuda.mem_get_info(i)
        except Exception:  # noqa: BLE001
            free, total = props.total_memory, props.total_memory
        return {"backend": "cuda", "name": props.name, "vram_gb": round(total / 2**30, 1),
                "free_gb": round(free / 2**30, 1)}
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        try:
            import psutil

            total = psutil.virtual_memory().total
        except Exception:  # noqa: BLE001
            total = 0
        gb = round(total * 0.7 / 2**30, 1)          # macOS lets the GPU use most of unified memory
        return {"backend": "mps", "name": "Apple GPU (unified memory)", "vram_gb": gb, "free_gb": gb}
    return {"backend": "none", "name": "no GPU", "vram_gb": 0.0, "free_gb": 0.0}


def plan_device(spec: ModelSpec, gpu: dict, *, allow_cpu: bool = True) -> dict:
    """Decide device/dtype/offload, or raise a MediaError that says what is needed."""
    need = f"{spec.name} needs ~{spec.vram_gb:g} GB of GPU memory (~{spec.vram_gb_offload:g} GB with CPU offload)"
    if gpu["backend"] == "none":
        if spec.kind == "video":
            raise MediaError(
                f"text-to-video needs a GPU: {need}, and none was found. On a CPU a few seconds of video would "
                "take many hours (dozens of denoising steps through a multi-billion-parameter transformer over "
                "every frame), so JEV refuses rather than freeze your machine. Explainer videos and footage "
                "editing work without a GPU.")
        if not (spec.cpu_ok and allow_cpu):
            raise MediaError(f"{need}; no GPU was found and this model is too large to run on the CPU. "
                             "sdxl-turbo works on a CPU (slowly, about a minute per 512x512 image).")
        return {"device": "cpu", "dtype": "float32", "offload": False,
                "note": "running on the CPU: expect about a minute per image"}
    have = float(gpu.get("vram_gb") or 0)
    dtype = "float16" if spec.name.startswith("sdxl") else "bfloat16"
    if gpu["backend"] == "mps":
        dtype = "float16" if dtype == "bfloat16" else dtype
    if have >= spec.vram_gb:
        return {"device": gpu["backend"], "dtype": dtype, "offload": False, "note": ""}
    if have >= spec.vram_gb_offload:
        return {"device": gpu["backend"], "dtype": dtype, "offload": True,
                "note": f"{have:g} GB GPU: using model CPU offload (slower, fits ~{spec.vram_gb_offload:g} GB)"}
    raise MediaError(f"{need}; this GPU ({gpu.get('name')}) has {have:g} GB. Choose a smaller model "
                     f"({', '.join(n for n, s in CATALOG.items() if s.kind == spec.kind and s.vram_gb_offload <= have) or 'none fits'}).")


# --------------------------------------------------------------------------- model folders

def models_dir(settings: dict | None = None) -> Path:
    media = (settings or {}).get("media", {}) or {}
    if media.get("models_dir"):
        return Path(media["models_dir"]).expanduser()
    if os.environ.get("JEV_MODELS_DIR"):
        return Path(os.environ["JEV_MODELS_DIR"]).expanduser()
    return Path(os.environ.get("JEV_HOME", Path.home() / ".jev")).expanduser() / "models"


def resolve_model(model: str, settings: dict | None = None) -> tuple[Path, ModelSpec | None]:
    """A catalog name or a folder name under the models folder; settings media.model_dirs can map
    a name to any folder you chose. Arbitrary paths are not accepted from the model."""
    media = (settings or {}).get("media", {}) or {}
    mapped = (media.get("model_dirs") or {}).get(model)
    if mapped:
        d = Path(mapped).expanduser()
    else:
        if not model or any(ch in model for ch in "/\\") or model in (".", ".."):
            raise MediaError(f"model must be a catalog name ({', '.join(CATALOG)}) or a folder name in "
                             f"{models_dir(settings)}, not a path")
        d = models_dir(settings) / model
    return d, CATALOG.get(model)


def _sha256(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as f:
        while b := f.read(1 << 22):
            h.update(b)
    return h.hexdigest()


def inspect_model_dir(path: Path, kind: str | None = None, verify: bool = False) -> dict:
    """Check a diffusers folder before anything is loaded. Returns {class_name, components,
    ignored_pickles, verified}. Raises MediaError for pickle-only weights, unknown pipeline
    classes, or (verify=True) files that do not match jev-model.json."""
    path = Path(path).expanduser()
    if path.is_file() or path.suffix.lower() in REFUSED_SUFFIXES:
        if path.suffix.lower() in REFUSED_SUFFIXES:
            raise MediaError(f"{path.name}: pickle-based checkpoints are refused (they can run code when "
                             "loaded). Use a diffusers folder with .safetensors weights.")
        raise MediaError(f"{path.name}: expected a diffusers model folder (with model_index.json), not a file")
    if not path.is_dir():
        raise MediaError(f"model folder not found: {path}. Download it first (download_generative_model) or set "
                         "media.model_dirs in the settings to a folder you already have.")
    idx = path / "model_index.json"
    if not idx.is_file() or idx.stat().st_size > 1 << 20:
        raise MediaError(f"{path.name}: no model_index.json (not a diffusers model folder)")
    try:
        index = json.loads(idx.read_text())
    except json.JSONDecodeError as e:
        raise MediaError(f"{path.name}/model_index.json is not valid JSON") from e
    cls = index.get("_class_name")
    if cls not in ALLOWED_PIPELINES:
        raise MediaError(f"{path.name}: pipeline class {cls!r} is not on the allowlist "
                         f"({', '.join(sorted(ALLOWED_PIPELINES))}); custom pipeline code is never run")
    components = {}
    for name, val in index.items():
        if name.startswith("_") or not isinstance(val, list) or len(val) != 2 or val[0] is None:
            continue
        sub = path / name
        files = [p for p in sub.rglob("*") if p.is_file()] if sub.is_dir() else []
        st = [p for p in files if p.suffix == ".safetensors"]
        pk = [p for p in files if p.suffix.lower() in REFUSED_SUFFIXES]
        if pk and not st:
            raise MediaError(f"{path.name}/{name}: only pickle weights ({pk[0].name}); refused. Download the "
                             "safetensors version of this model.")
        components[name] = {"library": val[0], "class": val[1], "safetensors": len(st)}
    ignored = sorted(p.relative_to(path).as_posix() for p in path.rglob("*")
                     if p.is_file() and p.suffix.lower() in REFUSED_SUFFIXES)
    verified = False
    man = path / MANIFEST
    if verify and man.is_file():
        rec = json.loads(man.read_text()).get("files", {})
        for rel, meta in rec.items():
            p = (path / rel).resolve()
            if path.resolve() not in p.parents or not p.is_file():
                raise MediaError(f"{path.name}: {rel} listed in {MANIFEST} is missing")
            if _sha256(p) != meta.get("sha256"):
                raise MediaError(f"{path.name}: {rel} does not match the SHA-256 recorded at download")
        verified = True
    return {"class_name": cls, "components": components, "ignored_pickles": ignored, "verified": verified}


def _spec_for(path: Path, spec: ModelSpec | None, cls: str, kind: str) -> ModelSpec:
    if spec is None:
        spec = next((s for s in CATALOG.values() if cls in s.class_names), None)
    if spec is None:
        raise MediaError(f"no settings known for a {cls} folder; add it to the catalog first")
    if spec.kind != kind:
        raise MediaError(f"{spec.name} is a text-to-{spec.kind} model, not text-to-{kind}")
    return spec


def _diffusers_factory(path: Path, spec: ModelSpec, plan: dict):
    try:
        import diffusers
        import torch
    except ImportError as e:
        raise MediaError("local generative models need diffusers: `pip install 'jepa-studio[gen]'` "
                         "(diffusers, transformers, accelerate, sentencepiece, ftfy)") from e
    os.environ["HF_HUB_OFFLINE"] = "1"
    cls = getattr(diffusers, spec.pipeline, None)
    if cls is None:
        raise MediaError(f"this diffusers ({diffusers.__version__}) has no {spec.pipeline}; upgrade diffusers")
    kwargs: dict = {"torch_dtype": getattr(torch, plan["dtype"]), "use_safetensors": True, "local_files_only": True}
    if spec.variant and any(path.rglob(f"*.{spec.variant}.safetensors")):
        kwargs["variant"] = spec.variant
    pipe = cls.from_pretrained(str(path), **kwargs)
    if plan["offload"]:
        pipe.enable_model_cpu_offload()
    else:
        pipe = pipe.to(plan["device"])
    return pipe


def _call(pipe, kwargs: dict, steps: int, progress, cancelled):
    """Call the pipeline with only the arguments it accepts, plus a progress/cancel callback."""
    try:
        params = inspect.signature(pipe.__call__).parameters
        accepts_any = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in params.values())
        kw = {k: v for k, v in kwargs.items() if accepts_any or k in params}
    except (TypeError, ValueError):
        params, kw = {}, dict(kwargs)
    if "callback_on_step_end" in params:
        def cb(p, step, timestep, cb_kwargs):
            if progress:
                progress(0.05 + 0.85 * (step + 1) / max(1, steps), f"step {step + 1}/{steps}")
            if cancelled and cancelled():
                p._interrupt = True
            return cb_kwargs
        kw["callback_on_step_end"] = cb
    out = pipe(**kw)
    if cancelled and cancelled():
        raise MediaError("generation cancelled")
    return out


def _check_size(spec: ModelSpec, width: int, height: int) -> tuple[int, int]:
    m = spec.size_multiple
    w, h = int(round(width / m) * m), int(round(height / m) * m)
    if not (256 <= w <= 2048 and 256 <= h <= 2048):
        raise MediaError("width and height must be within 256..2048")
    return w, h


def _generator(seed: int, plan: dict):
    import torch

    dev = "cpu" if plan["offload"] or plan["device"] in ("cpu", "mps") else plan["device"]
    try:
        return torch.Generator(device=dev).manual_seed(int(seed))
    except RuntimeError:            # device generator unavailable: a CPU generator is still deterministic
        return torch.Generator(device="cpu").manual_seed(int(seed))


def _prepare(model: str, kind: str, settings: dict | None, gpu: dict | None, allow_cpu: bool, verify: bool):
    path, spec = resolve_model(model, settings)
    info = inspect_model_dir(path, kind, verify=verify)
    spec = _spec_for(path, spec, info["class_name"], kind)
    plan = plan_device(spec, gpu or detect_gpu(), allow_cpu=allow_cpu)
    return path, spec, info, plan


def estimate_seconds(spec: ModelSpec, plan: dict, steps: int, frames: int = 1) -> float:
    per_step = {"image": 0.25, "video": 2.5}[spec.kind] * (6 if plan["offload"] else 1)
    if plan["device"] == "cpu":
        per_step = 40.0
    return 30 + per_step * steps * (max(1, frames) / 50 if spec.kind == "video" else 1)


def generate_image(model: str, prompt: str, out: str | Path, workspace: str | Path, *, seed: int = 0,
                   steps: int | None = None, guidance: float | None = None, width: int | None = None,
                   height: int | None = None, negative_prompt: str | None = None, settings: dict | None = None,
                   gpu: dict | None = None, allow_cpu: bool = True, verify: bool = True,
                   progress=None, cancelled=None) -> dict:
    ws = Path(workspace).resolve()
    o = inside(out, ws).with_suffix(".png")
    if not prompt or len(prompt) > 2000:
        raise MediaError("prompt must be 1..2000 characters")
    path, spec, info, plan = _prepare(model, "image", settings, gpu, allow_cpu, verify)
    d = spec.defaults
    steps = int(steps or d.get("steps", 20))
    if not 1 <= steps <= 100:
        raise MediaError("steps must be within 1..100")
    w, h = _check_size(spec, width or d.get("width", 1024), height or d.get("height", 1024))
    g = d.get("guidance", 5.0) if guidance is None else float(guidance)
    t0 = time.monotonic()
    pipe = (_FACTORY or _diffusers_factory)(path, spec, plan)
    kw = {"prompt": prompt, "num_inference_steps": steps, "guidance_scale": g, "width": w, "height": h,
          "generator": _generator(seed, plan)}
    if negative_prompt:
        kw["negative_prompt"] = negative_prompt
    if "max_sequence_length" in d:
        kw["max_sequence_length"] = d["max_sequence_length"]
    res = _call(pipe, kw, steps, progress, cancelled)
    im = res.images[0]
    if not isinstance(im, Image.Image):
        im = Image.fromarray(_to_uint8(np.asarray(im)))
    meta = {"model": spec.name, "repo": spec.repo, "prompt": prompt, "negative_prompt": negative_prompt or "",
            "seed": seed, "steps": steps, "guidance": g, "width": w, "height": h, "device": plan["device"],
            "dtype": plan["dtype"], "offload": plan["offload"], "license": spec.license,
            "seconds": round(time.monotonic() - t0, 2), "generator": "jev.media.generative"}
    save_png(im, o, {"jev": json.dumps(meta)})
    o.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return {"image": o, "meta": meta, "plan": plan, "ignored_pickles": info["ignored_pickles"]}


def _to_uint8(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a)
    if a.dtype != np.uint8:
        a = np.clip(a * (255.0 if a.max() <= 1.0 + 1e-6 else 1.0), 0, 255).round().astype(np.uint8)
    if a.ndim == 3 and a.shape[0] in (1, 3) and a.shape[-1] not in (1, 3):
        a = np.transpose(a, (1, 2, 0))
    if a.ndim == 2:
        a = np.stack([a] * 3, -1)
    return a[..., :3]


def generate_video(model: str, prompt: str, out: str | Path, workspace: str | Path, *, seed: int = 0,
                   steps: int | None = None, guidance: float | None = None, width: int | None = None,
                   height: int | None = None, frames: int | None = None, fps: int | None = None,
                   negative_prompt: str | None = None, settings: dict | None = None, gpu: dict | None = None,
                   verify: bool = True, progress=None, cancelled=None) -> dict:
    ws = Path(workspace).resolve()
    o = inside(out, ws).with_suffix(".mp4")
    if not prompt or len(prompt) > 2000:
        raise MediaError("prompt must be 1..2000 characters")
    path, spec, info, plan = _prepare(model, "video", settings, gpu, False, verify)
    d = spec.defaults
    steps = int(steps or d.get("steps", 30))
    if not 1 <= steps <= 100:
        raise MediaError("steps must be within 1..100")
    w, h = _check_size(spec, width or d.get("width", 832), height or d.get("height", 480))
    n = int(frames or d.get("frames", 49))
    if spec.frames_rule:
        a, b = spec.frames_rule
        n = max(a + b, int(round((n - b) / a)) * a + b)
    if n > 257:
        raise MediaError("at most 257 frames per clip")
    fps = int(fps or d.get("fps", 16))
    g = d.get("guidance", 5.0) if guidance is None else float(guidance)
    t0 = time.monotonic()
    pipe = (_FACTORY or _diffusers_factory)(path, spec, plan)
    kw = {"prompt": prompt, "num_inference_steps": steps, "guidance_scale": g, "width": w, "height": h,
          "num_frames": n, "generator": _generator(seed, plan)}
    for k in ("decode_timestep", "decode_noise_scale"):
        if k in d:
            kw[k] = d[k]
    if negative_prompt:
        kw["negative_prompt"] = negative_prompt
    res = _call(pipe, kw, steps, progress, cancelled)
    vid = res.frames[0]
    arrs = [_to_uint8(np.asarray(f)) for f in vid]
    H, W = arrs[0].shape[:2]
    size = (even(W), even(H))
    arrs = [a[:size[1], :size[0]] for a in arrs]
    F.encode_frames(arrs, o, ws, size=size, fps=fps, total_frames=len(arrs), crf=18, preset="medium",
                    label="video encode")
    meta = {"model": spec.name, "repo": spec.repo, "prompt": prompt, "negative_prompt": negative_prompt or "",
            "seed": seed, "steps": steps, "guidance": g, "width": size[0], "height": size[1], "frames": len(arrs),
            "fps": fps, "device": plan["device"], "dtype": plan["dtype"], "offload": plan["offload"],
            "license": spec.license, "seconds": round(time.monotonic() - t0, 2), "generator": "jev.media.generative"}
    o.with_suffix(".json").write_text(json.dumps(meta, indent=2))
    return {"video": o, "meta": meta, "plan": plan, "ignored_pickles": info["ignored_pickles"]}


# --------------------------------------------------------------------------- download (approval-gated)

WEIGHT_IGNORE = ["*.bin", "*.ckpt", "*.pt", "*.pth", "*.pkl", "*.pickle", "*.joblib", "*.msgpack", "*.h5",
                 "*.onnx", "*.onnx_data", "*.gguf", "*.mlmodel*", "*.png", "*.jpg", "*.gif", "*.mp4"]


def download_patterns(spec: ModelSpec) -> tuple[list[str], list[str]]:
    weights = [f"*.{spec.variant}.safetensors", f"*/*.{spec.variant}.safetensors"] if spec.variant else \
        ["*.safetensors", "*/*.safetensors"]
    allow = ["model_index.json", "*.json", "*/*.json", "*.txt", "*/*.txt", "*/*.model", "*/spiece.model",
             "*/*.tiktoken", *weights]
    return allow, list(WEIGHT_IGNORE)


def download_model(name: str, settings: dict | None = None, *, revision: str = "main",
                   snapshot: Callable[..., str] | None = None, progress=None) -> dict:
    """Download a catalog model (safetensors + configs only) into the models folder and record the
    SHA-256 of every file in jev-model.json. Always approval-gated by the tool layer."""
    spec = CATALOG.get(name)
    if spec is None:
        raise MediaError(f"unknown model {name!r}; downloadable: {', '.join(CATALOG)}")
    if snapshot is None:
        try:
            from huggingface_hub import snapshot_download as snapshot
        except ImportError as e:
            raise MediaError("downloading models needs huggingface_hub: `pip install 'jepa-studio[gen]'`") from e
    dest = models_dir(settings) / name
    dest.mkdir(parents=True, exist_ok=True)
    allow, ignore = download_patterns(spec)
    if progress:
        progress(0.01, f"downloading {spec.repo} (~{spec.download_gb:g} GB)")
    snapshot(repo_id=spec.repo, revision=revision, local_dir=str(dest), allow_patterns=allow, ignore_patterns=ignore)
    for p in dest.rglob("*"):                       # belt and braces: never keep a pickle
        if p.is_file() and p.suffix.lower() in REFUSED_SUFFIXES:
            p.unlink()
    info = inspect_model_dir(dest, spec.kind)
    files = {}
    for p in sorted(dest.rglob("*")):
        if p.is_file() and p.name != MANIFEST and ".cache" not in p.parts:
            files[p.relative_to(dest).as_posix()] = {"sha256": _sha256(p), "bytes": p.stat().st_size}
    (dest / MANIFEST).write_text(json.dumps({"format": "jev-model/v1", "name": name, "repo": spec.repo,
                                             "revision": revision, "license": spec.license, "files": files},
                                            indent=2))
    if progress:
        progress(1.0, "model ready")
    return {"model": name, "folder": str(dest), "files": len(files), "class_name": info["class_name"],
            "bytes": sum(f["bytes"] for f in files.values())}
