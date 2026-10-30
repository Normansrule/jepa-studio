"""Generative adapters with a tiny fake pipeline injected through set_pipeline_factory (no
weights are downloaded here), plus the refusal paths: pickles, custom code, VRAM, CPU video."""
import json
import types

import numpy as np
import pytest
import torch
from PIL import Image
from safetensors.torch import save_file

from jev.media import ffmpeg as F
from jev.media import generative as G
from jev.media.common import MediaError

NO_GPU = {"backend": "none", "name": "no GPU", "vram_gb": 0.0, "free_gb": 0.0}
BIG_GPU = {"backend": "cuda", "name": "Fake 24 GB", "vram_gb": 24.0, "free_gb": 24.0}


def make_model(root, name, cls, components=("unet",), weight_ext=".safetensors"):
    d = root / name
    index = {"_class_name": cls, "_diffusers_version": "0.42.1", "scheduler": ["diffusers", "EulerDiscreteScheduler"]}
    (d / "scheduler").mkdir(parents=True)
    (d / "scheduler" / "scheduler_config.json").write_text("{}")
    for c in components:
        index[c] = ["diffusers", "SomeModel"]
        (d / c).mkdir()
        (d / c / "config.json").write_text("{}")
        if weight_ext == ".safetensors":
            save_file({"w": torch.zeros(2)}, str(d / c / "diffusion_pytorch_model.safetensors"))
        else:
            (d / c / f"diffusion_pytorch_model{weight_ext}").write_bytes(b"\x80\x04pickle")
    (d / "model_index.json").write_text(json.dumps(index))
    return d


class FakePipe:
    """Mimics a diffusers pipeline call: returns .images or .frames depending on the kind."""

    def __init__(self, kind):
        self.kind = kind
        self.calls = []
        self._interrupt = False

    def __call__(self, prompt, num_inference_steps, guidance_scale, width, height, generator, num_frames=None,
                 negative_prompt=None, callback_on_step_end=None, **kw):
        self.calls.append(dict(prompt=prompt, steps=num_inference_steps, guidance=guidance_scale, width=width,
                               height=height, num_frames=num_frames, extra=kw))
        for i in range(num_inference_steps):
            if callback_on_step_end:
                callback_on_step_end(self, i, 1000 - i, {})
        noise = torch.rand(3, generator=generator).numpy()
        if self.kind == "image":
            arr = np.ones((height, width, 3), np.float32) * noise
            return types.SimpleNamespace(images=[Image.fromarray((arr * 255).astype(np.uint8))])
        frames = [np.ones((height, width, 3), np.float32) * noise * (k / num_frames) for k in range(num_frames)]
        return types.SimpleNamespace(frames=[frames])


@pytest.fixture
def setup(tmp_path):
    models = tmp_path / "models"
    ws = tmp_path / "ws"
    ws.mkdir()
    settings = {"media": {"models_dir": str(models)}}
    made = {}

    def factory(path, spec, plan):
        made["plan"] = plan
        made["pipe"] = FakePipe(spec.kind)
        return made["pipe"]

    G.set_pipeline_factory(factory)
    yield models, ws, settings, made
    G.set_pipeline_factory(None)


def test_text_to_image_fake_pipeline_deterministic(setup):
    models, ws, settings, made = setup
    make_model(models, "sdxl-turbo", "StableDiffusionXLPipeline")
    progress = []
    r1 = G.generate_image("sdxl-turbo", "a red fox", "a.png", ws, seed=7, settings=settings, gpu=NO_GPU,
                          progress=lambda f, m: progress.append(f))
    r2 = G.generate_image("sdxl-turbo", "a red fox", "b.png", ws, seed=7, settings=settings, gpu=NO_GPU)
    r3 = G.generate_image("sdxl-turbo", "a red fox", "c.png", ws, seed=8, settings=settings, gpu=NO_GPU)
    a, b, c = (np.asarray(Image.open(r["image"])) for r in (r1, r2, r3))
    assert a.shape == (512, 512, 3) and np.array_equal(a, b) and not np.array_equal(a, c)
    assert made["plan"]["device"] == "cpu" and progress
    call = made["pipe"].calls[-1]
    assert call["steps"] == 1 and call["guidance"] == 0.0            # SDXL-Turbo defaults
    meta = json.loads(Image.open(r1["image"]).text["jev"])
    assert meta["seed"] == 7 and meta["model"] == "sdxl-turbo"
    assert json.loads(r1["image"].with_suffix(".json").read_text())["prompt"] == "a red fox"


def test_text_to_video_fake_pipeline(setup):
    models, ws, settings, made = setup
    make_model(models, "ltx-video", "LTXPipeline", components=("transformer", "vae"))
    r = G.generate_video("ltx-video", "waves at sunset", "v.mp4", ws, seed=1, frames=20, fps=10, steps=2,
                         width=256, height=256, settings=settings, gpu=BIG_GPU)
    call = made["pipe"].calls[-1]
    assert (call["num_frames"] - 1) % 8 == 0                          # LTX frame rule applied (8k + 1)
    assert call["extra"]["decode_timestep"] == 0.03
    info = F.probe(r["video"], ws)
    assert info.video_codec == "h264" and (info.width, info.height) == (256, 256)
    assert abs(info.duration_s - call["num_frames"] / 10) < 0.15
    assert made["plan"] == {"device": "cuda", "dtype": "bfloat16", "offload": False, "note": ""}


def test_cancel_interrupts(setup):
    models, ws, settings, made = setup
    make_model(models, "sdxl-turbo", "StableDiffusionXLPipeline")
    with pytest.raises(MediaError, match="cancelled"):
        G.generate_image("sdxl-turbo", "x", "a.png", ws, settings=settings, gpu=NO_GPU, cancelled=lambda: True)
    assert made["pipe"]._interrupt is True


def test_pickle_weights_refused(setup, tmp_path):
    models, ws, settings, _ = setup
    make_model(models, "sdxl-turbo", "StableDiffusionXLPipeline", weight_ext=".bin")
    with pytest.raises(MediaError, match="pickle"):
        G.generate_image("sdxl-turbo", "x", "a.png", ws, settings=settings, gpu=NO_GPU)
    ckpt = tmp_path / "model.ckpt"
    ckpt.write_bytes(b"x")
    with pytest.raises(MediaError, match="pickle"):
        G.inspect_model_dir(ckpt)
    # a stray .bin next to safetensors is ignored (use_safetensors=True) and reported
    d = make_model(tmp_path / "other", "m", "StableDiffusionXLPipeline")
    (d / "unet" / "diffusion_pytorch_model.bin").write_bytes(b"x")
    info = G.inspect_model_dir(d)
    assert info["ignored_pickles"] == ["unet/diffusion_pytorch_model.bin"]


def test_custom_pipeline_and_paths_refused(setup, tmp_path):
    models, ws, settings, _ = setup
    make_model(models, "weird", "TotallyCustomPipeline")
    with pytest.raises(MediaError, match="allowlist"):
        G.generate_image("weird", "x", "a.png", ws, settings=settings, gpu=NO_GPU)
    for bad in ("../etc", "/abs/path", "a/b"):
        with pytest.raises(MediaError, match="not a path"):
            G.resolve_model(bad, settings)
    with pytest.raises(MediaError, match="not found"):
        G.generate_image("sdxl-turbo", "x", "a.png", ws, settings=settings, gpu=NO_GPU)
    make_model(models, "sdxl-turbo", "StableDiffusionXLPipeline")
    with pytest.raises(MediaError, match="outside"):
        G.generate_image("sdxl-turbo", "x", "../a.png", ws, settings=settings, gpu=NO_GPU)


def test_manifest_verification(setup):
    models, ws, settings, _ = setup
    d = make_model(models, "sdxl-turbo", "StableDiffusionXLPipeline")
    (d / G.MANIFEST).write_text(json.dumps({"files": {"unet/diffusion_pytorch_model.safetensors": {"sha256": "0" * 64}}}))
    with pytest.raises(MediaError, match="SHA-256"):
        G.generate_image("sdxl-turbo", "x", "a.png", ws, settings=settings, gpu=NO_GPU)


def test_vram_messages_and_cpu_video_refused():
    flux, ltx, sdxl = G.CATALOG["flux-schnell"], G.CATALOG["ltx-video"], G.CATALOG["sdxl-turbo"]
    with pytest.raises(MediaError, match=r"needs ~33 GB.*has 8 GB"):
        G.plan_device(flux, {"backend": "cuda", "name": "8G", "vram_gb": 8.0})
    plan = G.plan_device(flux, {"backend": "cuda", "name": "4090", "vram_gb": 24.0})
    assert plan["offload"] is True
    with pytest.raises(MediaError, match="needs a GPU"):
        G.plan_device(ltx, NO_GPU)
    with pytest.raises(MediaError, match="too large to run on the CPU"):
        G.plan_device(flux, NO_GPU)
    assert G.plan_device(sdxl, NO_GPU)["device"] == "cpu"
    assert G.detect_gpu()["backend"] in ("none", "cuda", "mps")


def test_cpu_video_refused_end_to_end(setup):
    models, ws, settings, made = setup
    make_model(models, "wan2.1-t2v-1.3b", "WanPipeline", components=("transformer",))
    with pytest.raises(MediaError, match="needs a GPU"):
        G.generate_video("wan2.1-t2v-1.3b", "x", "v.mp4", ws, settings=settings, gpu=NO_GPU)
    assert "pipe" not in made                               # refused before anything was loaded


def test_download_patterns_and_manifest(tmp_path):
    settings = {"media": {"models_dir": str(tmp_path / "models")}}
    seen = {}

    def snapshot(repo_id, revision, local_dir, allow_patterns, ignore_patterns):
        seen.update(repo=repo_id, allow=allow_patterns, ignore=ignore_patterns)
        d = make_model(tmp_path / "dl", "x", "StableDiffusionXLPipeline")
        import shutil

        shutil.copytree(d, local_dir, dirs_exist_ok=True)
        (tmp_path / "models" / "sdxl-turbo" / "unet" / "sneaky.bin").write_bytes(b"x")
        return local_dir

    r = G.download_model("sdxl-turbo", settings, snapshot=snapshot)
    assert seen["repo"] == "stabilityai/sdxl-turbo"
    assert "*.bin" in seen["ignore"] and "*.ckpt" in seen["ignore"]
    assert not any(p.endswith((".bin", ".ckpt", ".pt", ".pth")) for p in seen["allow"])
    assert any("fp16.safetensors" in p for p in seen["allow"])
    d = tmp_path / "models" / "sdxl-turbo"
    assert not (d / "unet" / "sneaky.bin").exists()
    man = json.loads((d / G.MANIFEST).read_text())
    assert "unet/diffusion_pytorch_model.safetensors" in man["files"] and r["files"] == len(man["files"])
    assert G.inspect_model_dir(d, verify=True)["verified"]
    with pytest.raises(MediaError, match="unknown model"):
        G.download_model("some-random-model", settings, snapshot=snapshot)
