"""Python <-> JavaScript parity for the world envs and exported world models.

Runs `node tests/js/world_parity.mjs <cases.json> [<model.json>]` and compares:
  * env reset / step / goal / render   max abs diff < 1e-5
  * encoder + predictor outputs        max abs diff < 1e-4 (vs PyTorch with the rounded weights)
  * nearest-bank retrieval             same index (or an exact tie)
  * CEM timing                         300 samples x H 5 x 10 iters < 100 ms (reported; hard
                                       limit 1 s so slow CI machines do not flake)
"""
from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import numpy as np
import pytest
import torch

from jepa_studio.world import CEMConfig, WorldHParams, build_bank, make_env
from jepa_studio.world.export import export_json
from jepa_studio.world.model import WorldModel, collect_dataset

ROOT = Path(__file__).resolve().parent.parent
SCRIPT = ROOT / "tests" / "js" / "world_parity.mjs"
NODE = shutil.which("node")
pytestmark = pytest.mark.skipif(NODE is None, reason="node not installed")


def _env_cases() -> list[dict]:
    rng = np.random.default_rng(123)
    cases = []
    for name in ["two-room", "push-block"]:
        for seed in [0, 1, 42, 2**31 - 5]:
            acts = rng.uniform(-1.3, 1.3, (25, 2))            # includes out-of-range (clipped) actions
            cases.append({"env": name, "seed": seed, "actions": acts.tolist()})
    # deterministic wall / push interactions
    cases.append({"env": "two-room", "seed": 3, "actions": [[1.0, 0.0]] * 12 + [[0.0, 1.0]] * 8 + [[-1.0, 0.0]] * 8})
    cases.append({"env": "push-block", "seed": 5, "actions": [[1.0, 0.0]] * 10 + [[0.0, -1.0]] * 10 + [[-1.0, 1.0]] * 6})
    return cases


def _run(tmp_path: Path, cases: dict, model_json: Path | None = None) -> dict:
    cpath = tmp_path / "cases.json"
    cpath.write_text(json.dumps(cases))
    args = [NODE, str(SCRIPT), str(cpath)] + ([str(model_json)] if model_json else [])
    out = subprocess.run(args, capture_output=True, text=True, timeout=300, cwd=ROOT)
    assert out.returncode == 0, out.stderr
    return json.loads(out.stdout)


def test_env_parity(tmp_path):
    cases = _env_cases()
    js = _run(tmp_path, {"env_cases": cases})
    for c, j in zip(cases, js["env"]):
        env = make_env(c["env"])
        s = env.reset(c["seed"])
        states, frames = [s], [env.render(s)]
        for a in c["actions"]:
            s = env.step(s, a)
            states.append(s)
            frames.append(env.render(s))
        np.testing.assert_allclose(np.array(j["states"]), np.stack(states), atol=1e-12, rtol=0)
        np.testing.assert_allclose(np.array(j["goal"]), env.goal_state(c["seed"]), atol=1e-12, rtol=0)
        diff = np.abs(np.array(j["frames"], dtype=np.float32).reshape(-1, 3, 32, 32) - np.stack(frames)).max()
        assert diff < 1e-5, f"{c['env']} seed {c['seed']}: render diff {diff}"
        assert j["goal_success"]


def _model_cases(model: WorldModel, env_name: str) -> tuple[dict, np.ndarray, np.ndarray, np.ndarray]:
    env = make_env(env_name)
    rng = np.random.default_rng(0)
    imgs = np.stack([env.render(env.reset(s)) for s in range(6)] + [env.render(env.goal_state(0))])
    z = rng.normal(0, 1, (7, model.latent_dim)).astype(np.float32)
    a = rng.uniform(-1, 1, (7, 2)).astype(np.float32)
    return {"images": imgs.reshape(len(imgs), -1).tolist(), "z": z.tolist(), "a": a.tolist()}, imgs, z, a


def _check_model(tmp_path: Path, model: WorldModel, json_path: Path, env_name: str) -> dict:
    mc, imgs, z, a = _model_cases(model, env_name)
    js = _run(tmp_path, {"env_cases": [], "model_cases": mc}, json_path)["model"]
    doc = json.loads(json_path.read_text())
    with torch.no_grad():
        lat = model.encode(torch.from_numpy(imgs)).numpy()
        pred = model.predict(torch.from_numpy(z), torch.from_numpy(a)).numpy()
    dl = np.abs(np.array(js["latents"]) - lat).max()
    dp = np.abs(np.array(js["preds"]) - pred).max()
    assert dl < 1e-4, f"encoder parity {dl}"
    assert dp < 1e-4, f"predictor parity {dp}"
    bank = np.array(doc["bank"]["latents"], dtype=np.float32).reshape(doc["bank"]["size"], -1)
    d2 = ((lat[:, None] - bank[None]) ** 2).sum(-1)
    for i, k in enumerate(js["nearest"]):
        assert d2[i, k] <= d2[i].min() + 1e-4
    assert js["timing_ms"] < 1000
    return js


def test_model_parity_fresh_export(tmp_path):
    torch.manual_seed(0)
    env_name = "push-block"
    model = WorldModel(WorldHParams(latent_dim=16, enc_channels=(8, 16, 16), enc_hidden=64, pred_hidden=64))
    data = collect_dataset(env_name, 8, 6, seed=0)
    model.encoder.set_input_stats(data.obs.reshape(-1, 3, 32, 32))
    model.eval()
    bank = build_bank(model, make_env(env_name), data.states, 40)
    path = tmp_path / "m.json"
    export_json(model, env_name, bank, CEMConfig(), {}, path)   # rounds `model` in place
    js = _check_model(tmp_path, model, path, env_name)
    # stored bank latents are exactly what the (rounded) model computes from the stored states
    doc = json.loads(path.read_text())
    env = make_env(env_name)
    with torch.no_grad():
        re = model.encode(torch.from_numpy(np.stack([env.render(np.array(s)) for s in doc["bank"]["states"]]))).numpy()
    assert np.abs(re - np.array(doc["bank"]["latents"]).reshape(re.shape)).max() < 1e-5
    assert len(js["plan"]["costHistory"]) == 10


@pytest.mark.parametrize("env_name", ["two-room", "push-block"])
def test_shipped_model_parity(tmp_path, env_name):
    """The models shipped in site/models/ match their safetensors-free JSON in JS."""
    path = ROOT / "site" / "models" / f"world-{env_name}.json"
    if not path.exists():
        pytest.skip("shipped model not built yet")
    doc = json.loads(path.read_text())
    assert path.stat().st_size < 3_000_000
    hp = WorldHParams(**{k: tuple(v) if isinstance(v, list) else v
                         for k, v in doc["metrics"]["hparams"].items()})
    model = WorldModel(hp)
    _load_from_json(model, doc)
    js = _check_model(tmp_path, model, path, env_name)
    print(f"{env_name}: JS CEM step {js['timing_ms']:.1f} ms")


def _load_from_json(model: WorldModel, doc: dict) -> None:
    """Rebuild PyTorch weights from the JSON layers (lets the parity test run without the
    training output directory)."""
    enc = model.encoder
    L = doc["encoder"]["layers"]
    with torch.no_grad():
        enc.mean_image.copy_(torch.tensor(L[0]["mean"]).reshape(enc.mean_image.shape))
        enc.pixel_std.fill_(L[0]["std"])
        convs = [x for x in L if x["type"] == "conv2d"]
        for m, x in zip(enc.convs, convs):
            m.weight.copy_(torch.tensor(x["W"]).reshape(m.weight.shape))
            m.bias.copy_(torch.tensor(x["b"]))
        lins = [x for x in L if x["type"] == "linear"]
        for m, x in zip([enc.fc1, enc.fc2], lins):
            m.weight.copy_(torch.tensor(x["W"]).reshape(m.weight.shape))
            m.bias.copy_(torch.tensor(x["b"]))
        plins = [x for x in doc["predictor"]["layers"] if x["type"] == "linear"]
        pm = [m for m in model.predictor.net if isinstance(m, torch.nn.Linear)]
        for m, x in zip(pm, plins):
            m.weight.copy_(torch.tensor(x["W"]).reshape(m.weight.shape))
            m.bias.copy_(torch.tensor(x["b"]))
    model.eval()
