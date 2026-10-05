"""World model + planning bot: a JEPA world model trained from pixels (LeWorldModel recipe:
next-latent prediction + SIGReg, end-to-end, no decoder / EMA / stop-gradient) and a CEM
planner in latent space.

    envs.py     TwoRoom, PushBlock (pure step, 32x32 RGB render, exploration policies)
    model.py    WorldModel, collect_dataset, train_world_model
    planner.py  CEM, receding-horizon MPC, nearest-real-frame retrieval, success evaluation
    export.py   browser JSON (site/models/world-<env>.json)

This module adds the two entry points the CLI / server call:
    run_world(cfg, out_dir, log)  train + evaluate + export (safetensors, metrics.json, JSON model)
    make_rollout_gif(...)         real rollout next to retrieved "imagined" frames, animated GIF
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import asdict, replace
from pathlib import Path
from typing import Callable

import numpy as np

from ..weights import atomic_write_bytes, save_state_dict
from .envs import ENVS, PushBlock, TwoRoom, WorldEnv, make_env
from .export import export_json, round_model_
from .model import HPARAMS_BY_ENV, WorldHParams, WorldModel, collect_dataset, train, train_world_model
from .planner import CEMConfig, LatentBank, build_bank, evaluate_planner, plan_and_execute

__all__ = ["ENVS", "TwoRoom", "PushBlock", "WorldEnv", "make_env", "WorldModel", "WorldHParams",
           "collect_dataset", "train_world_model", "CEMConfig", "plan_and_execute", "evaluate_planner",
           "build_site_models", "EPISODES_BY_ENV",
           "build_bank", "LatentBank", "export_json", "run_world", "make_rollout_gif", "HPARAMS_BY_ENV",
           "PLAN_BY_ENV"]

# Data per environment for the shipped models (push-block needs more pushes in the data).
EPISODES_BY_ENV: dict[str, int] = {"two-room": 400, "push-block": 800}
# Planner settings (LeWM eval defaults: horizon 5, 300 samples, 30 elites); cost "sum".
PLAN_BY_ENV: dict[str, dict] = {"two-room": {}, "push-block": {}}
BANK_SIZE = 1024
EVAL_EPISODES = 20
EVAL_MAX_STEPS = 40


def run_world(cfg: dict, out_dir: str | Path, log: Callable[[str], None] = print,
              hp: WorldHParams | None = None, json_path: str | Path | None = None,
              eval_episodes: int = EVAL_EPISODES) -> dict:
    """Train, evaluate and export one world model.

    Writes into out_dir:
        world.safetensors   model weights (jepa_studio.weights.save_state_dict; never pickle)
        metrics.json        training + prediction + planning metrics
        events.jsonl        training event stream (data/start/step/end records)
        world-<env>.json    browser model (or at ``json_path`` if given)
    Returns the metrics dict.
    """
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    w = cfg["world"]
    env = make_env(w["env"])
    hp = hp or HPARAMS_BY_ENV.get(w["env"], WorldHParams())
    res = train(cfg, log=log, events_path=out / "events.jsonl", hp=hp)
    model, metrics = res.model, res.metrics
    # Round weights to the 6 significant digits the browser JSON stores BEFORE evaluating, so
    # the planning metrics, world.safetensors and world-<env>.json all describe the same model.
    round_model_(model)
    plan = replace(CEMConfig.from_world_cfg(w), **PLAN_BY_ENV.get(w["env"], {}))

    t0 = time.time()
    metrics["planning"] = evaluate_planner(env, model, plan, eval_episodes, EVAL_MAX_STEPS, log=log)
    metrics["planning"]["seconds"] = round(time.time() - t0, 1)
    metrics["plan_config"] = asdict(plan)
    metrics["config_world"] = dict(w)
    metrics["seed"] = int(cfg.get("seed", 0))

    bank = build_bank(model, env, res.data.states, BANK_SIZE, seed=int(cfg.get("seed", 0)))
    sha = save_state_dict(model, out / "world.safetensors", {"env": w["env"], "format": "jepa-studio-world/v1"})
    metrics["weights_sha256"] = sha
    summary = {k: v for k, v in metrics.items() if k != "history"}
    doc = export_json(model, w["env"], bank, plan, summary, json_path or out / f"world-{w['env']}.json")
    atomic_write_bytes(out / "metrics.json", (json.dumps(metrics, indent=2) + "\n").encode())
    for f in (out / "world.safetensors", out / "metrics.json", Path(json_path or out / f"world-{w['env']}.json")):
        os.chmod(f, 0o644)       # atomic writes create 0600 files; these are meant to be served/shared
    metrics["_model"] = model        # handy for callers in-process; stripped before JSON
    metrics["_bank"] = bank
    metrics["_json_bytes"] = len(json.dumps(doc, separators=(",", ":")))
    return metrics


def build_site_models(root: str | Path = ".", seed: int = 0, log: Callable[[str], None] = print,
                      envs: tuple[str, ...] = ("two-room", "push-block")) -> dict:
    """Reproduce the shipped artifacts: site/models/world-<env>.json, site/models/world-metrics.json,
    assets/world-<env>.gif, and runs/world-<env>/ (safetensors + metrics + events).

        PYTHONPATH=. python3 -c "from jepa_studio.world import build_site_models; build_site_models()"
    """
    from ..config import DEFAULT_CONFIG, deep_merge

    root = Path(root)
    summary: dict = {"format": "jepa-studio-world-metrics/v1", "seed": seed, "envs": {}}
    for name in envs:
        cfg = deep_merge(DEFAULT_CONFIG, {"seed": seed, "world": {"env": name,
                                                                   "episodes": EPISODES_BY_ENV.get(name, 400)}})
        m = run_world(cfg, root / "runs" / f"world-{name}", log=log,
                      json_path=root / "site" / "models" / f"world-{name}.json")
        env = make_env(name)
        plan = CEMConfig(**m["plan_config"])
        gif = root / "assets" / f"world-{name}.gif"
        # showcase = the first planning-eval episode that CEM solved (the success RATE above is
        # the honest number; the GIF just illustrates what a solved episode looks like)
        pl = m["planning"]
        solved = [s for s, ok in zip(pl["seeds"], pl["cem_success"]) if ok]
        gif_seed = solved[0] if solved else pl["seeds"][0]
        g = make_rollout_gif(env, m["_model"], m["_bank"], gif, gif_seed, plan=plan)
        os.chmod(gif, 0o644)
        pv = m["prediction_val"]
        summary["envs"][name] = {
            "train_seconds": m["train_seconds"], "data_seconds": m["data_seconds"],
            "planning_eval_seconds": m["planning"]["seconds"],
            "params": m["params"], "hparams": m["hparams"], "config_world": m["config_world"],
            "prediction": {"horizon": pv["horizon"], "model_mse": pv["model_mse"], "copy_mse": pv["copy_mse"],
                           "ratio": pv["ratio"],
                           "note": "open-loop k-step latent MSE on held-out episodes; copy = predict z_t for z_{t+k}"},
            "latent": m["latent_val"],
            "planning": {k: v for k, v in m["planning"].items()},
            "gif": {"path": gif.relative_to(root).as_posix(), "start_seed": gif_seed,
                    "selection": "first CEM-solved planning-eval episode" if solved else "first eval episode",
                    "success": g["success"],
                    "steps": g["steps"], "bytes": gif.stat().st_size},
            "json_bytes": m["_json_bytes"], "weights_sha256": m["weights_sha256"],
        }
    mpath = root / "site" / "models" / "world-metrics.json"
    atomic_write_bytes(mpath, (json.dumps(summary, indent=2) + "\n").encode())
    os.chmod(mpath, 0o644)
    return summary


# ----------------------------------------------------------------------------- GIF

_FONT = {  # 3x5 pixel font for the label strip (only the glyphs we need)
    "A": "010101111101101", "B": "110101110101110", "C": "011100100100011", "D": "110101101101110",
    "E": "111100110100111", "G": "011100101101011", "H": "101101111101101", "I": "111010010010111",
    "L": "100100100100111", "M": "101111111101101", "N": "110101101101101", "O": "010101101101010",
    "P": "110101110100100", "R": "110101110101101", "S": "011100010001110", "T": "111010010010010",
    "U": "101101101101111", "V": "101101101101010", "W": "101101111111101", "X": "101101010101101",
    "Y": "101101010010010", "K": "101101110101101", "F": "111100110100100", "Q": "010101101110011",
    "0": "111101101101111", "1": "010110010010111", "2": "110001010100111", "3": "110001010001110",
    "4": "101101111001001", "5": "111100110001110", "6": "011100111101111", "7": "111001010010010",
    "8": "111101111101111", "9": "111101111001110", " ": "000000000000000", "-": "000000111000000",
    "+": "000010111010000", ":": "000010000010000", "/": "001001010100100", ".": "000000000000010",
    "(": "010100100100010", ")": "010001001001010", "=": "000111000111000",
}


def _text(img: np.ndarray, x: int, y: int, s: str, scale: int, color) -> None:
    for ch in s.upper():
        g = _FONT.get(ch, _FONT[" "])
        for i in range(5):
            for j in range(3):
                if g[i * 3 + j] == "1":
                    img[y + i * scale:y + (i + 1) * scale, x + j * scale:x + (j + 1) * scale] = color
        x += 4 * scale


def _to_u8(frame: np.ndarray, scale: int) -> np.ndarray:
    """(3,S,S) float -> (S*scale, S*scale, 3) uint8, nearest-neighbour upscale."""
    hwc = (np.clip(frame.transpose(1, 2, 0), 0, 1) * 255 + 0.5).astype(np.uint8)
    return hwc.repeat(scale, 0).repeat(scale, 1)


def make_rollout_gif(env: WorldEnv, model: WorldModel, bank: LatentBank, path: str | Path,
                     start_seed: int, goal_seed: int | None = None, max_steps: int = 40,
                     plan: CEMConfig | None = None, scale: int = 5, fps: int = 4) -> dict:
    """Animated GIF: [real frame | goal | retrieved frames for imagined steps 1..H].

    The right-hand panels are NOT generated images: a JEPA has no decoder. Each shows the
    real training frame whose encoding is nearest to the imagined latent (retrieval), and
    the label strip says so. Returns the plan_and_execute result.
    """
    import imageio.v2 as imageio

    plan = plan or CEMConfig()
    r = plan_and_execute(env, model, start_seed, goal_seed, max_steps, plan, bank, seed=start_seed)
    S = env.spec.image_size * scale
    H = plan.horizon
    gap = 4
    small = max(1, scale * 3 // 5)                 # retrieved thumbnails are smaller
    sS = env.spec.image_size * small
    width = 2 * S + 3 * gap + H * (sS + gap) + gap
    label_h = 7 * 2 + 6
    height = label_h + S + label_h + gap
    goal = _to_u8(r["goal_frame"], scale)
    frames = []
    n = len(r["frames_real"])
    for t in range(n):
        canvas = np.full((height, width, 3), 255, np.uint8)
        _text(canvas, gap, 4, f"REAL T={t}", 2, (40, 40, 40))
        _text(canvas, 2 * gap + S, 4, "GOAL", 2, (40, 40, 40))
        x0 = 3 * gap + 2 * S
        _text(canvas, x0, 4, "IMAGINED: NEAREST REAL FRAME", 2, (150, 40, 40))
        canvas[label_h:label_h + S, gap:gap + S] = _to_u8(r["frames_real"][t], scale)
        canvas[label_h:label_h + S, 2 * gap + S:2 * gap + 2 * S] = goal
        k = min(t, len(r.get("retrieved_frames", [])) - 1)
        if k >= 0:
            for h, fr in enumerate(r["retrieved_frames"][k]):
                xx = x0 + h * (sS + gap)
                canvas[label_h:label_h + sS, xx:xx + sS] = _to_u8(fr, small)
                _text(canvas, xx, label_h + sS + 3, f"+{h + 1}", 2, (150, 40, 40))
        status = "SUCCESS" if (t == n - 1 and r["success"]) else ("FAILED" if t == n - 1 else "PLANNING")
        _text(canvas, gap, label_h + S + 5, f"{env.spec.name}  CEM  {status}", 2,
              (30, 120, 50) if status == "SUCCESS" else (40, 40, 40))
        _text(canvas, x0, label_h + sS + 20, "RETRIEVAL NOT GENERATION", 2, (120, 120, 120))
        frames.append(canvas)
    frames += [frames[-1]] * 4                    # hold the last frame
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    imageio.mimsave(str(path), frames, duration=1000.0 / fps, loop=0)  # imageio v2.28+: milliseconds per frame
    return r


# ----------------------------------------------------------------------------- desktop API helpers

def load_world_run(run_dir: str | Path) -> dict:
    """Reload a finished world run (weights are safetensors; hparams come from metrics.json).
    The retrieval bank is rebuilt from a small, deterministic exploration dataset."""
    from ..weights import load_state_dict
    from .model import collect_dataset

    d = Path(run_dir)
    cfg = json.loads((d / "config.json").read_text())
    metrics = json.loads((d / "metrics.json").read_text())
    hpd = dict(metrics["hparams"])
    hpd["enc_channels"] = tuple(hpd["enc_channels"])
    hp = WorldHParams(**hpd)
    model = WorldModel(hp)
    # checked against the SHA-256 that run_world recorded in metrics.json
    load_state_dict(model, d / "world.safetensors", expected_sha256=metrics.get("weights_sha256"))
    model.eval()
    env = make_env(cfg["world"]["env"])
    data = collect_dataset(env, 64, cfg["world"]["episode_len"], seed=int(cfg.get("seed", 0)) + 7)
    bank = build_bank(model, env, data.states, BANK_SIZE, seed=int(cfg.get("seed", 0)))
    plan = CEMConfig(**metrics["plan_config"])
    return {"env": env, "model": model, "bank": bank, "plan": plan}


def plan_for_api(w: dict, seed: int, max_steps: int = EVAL_MAX_STEPS) -> dict:
    """Run one receding-horizon episode and return JSON-able frames (PNG data URLs)."""
    import base64
    import io

    from PIL import Image

    def png(frame: np.ndarray) -> str:
        arr = (np.clip(frame, 0, 1).transpose(1, 2, 0) * 255).round().astype(np.uint8)
        buf = io.BytesIO()
        Image.fromarray(arr).resize((128, 128), Image.Resampling.NEAREST).save(buf, format="PNG")
        return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()

    r = plan_and_execute(w["env"], w["model"], 10_000 + seed, None, max_steps, w["plan"], w["bank"], seed=seed)
    return {"frames_real": [png(f) for f in r["frames_real"]],
            "frames_imagined": [[png(f) for f in fr] for fr in r.get("retrieved_frames", [])],
            "imagined_note": "nearest real frames retrieved for each imagined latent (JEPAs predict embeddings, not pixels)",
            "goal": png(r["goal_frame"]), "costs": r["costs"], "latent_dist": r["latent_dist"],
            "success": r["success"], "steps": r["steps"], "final_distance": r["final_distance"]}
