"""Export a trained world model as plain JSON the browser runs without ONNX or WebGL.

Format "jepa-studio-world/v1" (site/models/world-<env>.json):

{
  "format": "jepa-studio-world/v1",
  "env": "two-room" | "push-block",
  "latent_dim": d, "action_dim": 2, "image_size": 32,
  "encoder":   {"layers": [ ...layer... ]},    input: Float32Array(3*32*32), CHW, values in [0,1]
  "predictor": {"layers": [ ...layer... ], "residual": true, "input": "concat(z, a)"},
  "bank":      {"size": B, "latents": [B*d floats, row-major], "states": [[...state...], ...]},
  "plan_defaults": {"horizon", "samples", "elites", "iters", "init_std", "min_std", "cost"},
  "metrics":   {...}
}

Layer types (applied in order; tensors are flat Float32Arrays, CHW for images):
  {"type":"normalize", "mean":[C*H*W], "std": s}                 x <- (x - mean) / s
  {"type":"conv2d", "in", "out", "k", "stride", "pad", "in_h", "in_w", "out_h", "out_w",
   "W":[out*in*k*k] (PyTorch order out,in,kh,kw), "b":[out]}      zero padding, cross-correlation
  {"type":"flatten"}                                              no-op on a CHW flat array
  {"type":"linear", "in", "out", "W":[out*in] row-major, "b":[out]}   y = W x + b
  {"type":"act", "fn":"relu"|"gelu"|"tanh"}
  {"type":"layernorm", "dim", "weight", "bias", "eps"}            (not used by the default model)
Predictor: ẑ_{t+1} = z_t + layers([z_t, a_t]).

Floats are rounded to 6 significant digits; tests/test_world_parity.py checks the JS output
against PyTorch *with the rounded weights*, so rounding does not affect parity.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
from torch import nn

from ..weights import atomic_write_bytes
from .envs import make_env
from .model import WorldModel
from .planner import CEMConfig, LatentBank

FORMAT = "jepa-studio-world/v1"


def _r(x) -> list:
    """Flatten and round to 6 significant digits (float32 carries ~7)."""
    a = np.asarray(x, dtype=np.float64).ravel()
    return [float(f"{v:.6g}") for v in a]


def encoder_layers(model: WorldModel) -> list[dict]:
    enc = model.encoder
    S = model.image_size
    layers: list[dict] = [{"type": "normalize", "mean": _r(enc.mean_image), "std": float(f"{float(enc.pixel_std):.6g}")}]
    h = S
    for conv in enc.convs:
        k, st, pd = conv.kernel_size[0], conv.stride[0], conv.padding[0]
        oh = (h + 2 * pd - k) // st + 1
        layers.append({"type": "conv2d", "in": conv.in_channels, "out": conv.out_channels, "k": k,
                       "stride": st, "pad": pd, "in_h": h, "in_w": h, "out_h": oh, "out_w": oh,
                       "W": _r(conv.weight.detach()), "b": _r(conv.bias.detach())})
        layers.append({"type": "act", "fn": "relu"})
        h = oh
    layers.append({"type": "flatten"})
    layers += [_linear(enc.fc1), {"type": "act", "fn": "relu"}, _linear(enc.fc2)]
    return layers


def _linear(m: nn.Linear) -> dict:
    return {"type": "linear", "in": m.in_features, "out": m.out_features,
            "W": _r(m.weight.detach()), "b": _r(m.bias.detach())}


def _module_layers(seq: nn.Sequential) -> list[dict]:
    out = []
    for m in seq:
        if isinstance(m, nn.Linear):
            out.append(_linear(m))
        elif isinstance(m, nn.ReLU):
            out.append({"type": "act", "fn": "relu"})
        elif isinstance(m, nn.GELU):
            out.append({"type": "act", "fn": "gelu"})
        elif isinstance(m, nn.Tanh):
            out.append({"type": "act", "fn": "tanh"})
        elif isinstance(m, nn.LayerNorm):
            out.append({"type": "layernorm", "dim": m.normalized_shape[0], "weight": _r(m.weight.detach()),
                        "bias": _r(m.bias.detach()), "eps": m.eps})
        else:
            raise TypeError(f"cannot export layer {type(m).__name__}")
    return out


def round_model_(model: WorldModel) -> WorldModel:
    """Round every parameter/buffer in place to the 6 significant digits that the JSON stores,
    so the PyTorch model and the exported JSON compute exactly the same function."""
    with torch.no_grad():
        for t in list(model.parameters()) + list(model.buffers()):
            t.copy_(torch.tensor(np.array(_r(t), dtype=np.float32).reshape(t.shape)))
    return model


def export_json(model: WorldModel, env_name: str, bank: LatentBank, plan: CEMConfig, metrics: dict,
                path: str | Path) -> dict:
    """Write the browser JSON (see module docstring). Returns the dict written.

    Rounds the model's weights in place first (round_model_), rounds the bank states, and
    re-renders + re-encodes them with the rounded model so the stored bank latents are exactly
    what the browser computes from the stored states.
    """
    round_model_(model)
    states = np.array([[float(f"{v:.6g}") for v in st] for st in bank.states], dtype=np.float64)
    env = make_env(env_name)
    with torch.no_grad():
        frames = torch.from_numpy(np.stack([env.render(st) for st in states]))
        latents = model.encode(frames).numpy()
    doc = {
        "format": FORMAT,
        "env": env_name,
        "latent_dim": model.latent_dim,
        "action_dim": model.action_dim,
        "image_size": model.image_size,
        "encoder": {"layers": encoder_layers(model)},
        "predictor": {"layers": _module_layers(model.predictor.net), "residual": True, "input": "concat(z, a)"},
        "bank": {"size": int(len(states)), "latents": _r(latents), "states": states.tolist()},
        "plan_defaults": {"horizon": plan.horizon, "samples": plan.samples, "elites": plan.elites,
                          "iters": plan.iters, "init_std": plan.init_std, "min_std": plan.min_std,
                          "cost": plan.cost},
        "metrics": metrics,
    }
    data = json.dumps(doc, separators=(",", ":")).encode()
    atomic_write_bytes(path, data)
    return doc


def load_json_model(path: str | Path) -> dict:
    doc = json.loads(Path(path).read_text())
    if doc.get("format") != FORMAT:
        raise ValueError(f"{path}: not a {FORMAT} file")
    return doc
