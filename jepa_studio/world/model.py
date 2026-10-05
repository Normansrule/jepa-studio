"""JEPA world model trained end-to-end from pixels (the LeWorldModel recipe, re-implemented).

Idea (Maes et al., "LeWorldModel", arXiv:2603.19312): encode each frame o_t into a latent
z_t = f(o_t), predict the next latent from the current one and the action,
ẑ_{t+1} = g(z_t, a_t), and train f and g *jointly* with only two terms:

    L = (1 - λ) · mean_{t,k} || ẑ_{t+k} - z_{t+k} ||²     (latent prediction, open-loop k steps)
      +      λ  · SIGReg({z})                              (latents -> isotropic Gaussian N(0, I))

There is no decoder, no reconstruction loss, no EMA target network and no stop-gradient:
gradients flow through the targets z_{t+k} as well. The prediction term alone would be
minimised by a constant encoder (collapse); SIGReg (jepa_studio.losses.SIGReg) forbids that
by requiring every 1-D projection of the batch of latents to look standard-normal, which
pins the latent covariance near the identity. `collapse_report` checks this explicitly.

Architecture (chosen so the browser can run it from plain JSON, see world/export.py):
    encoder  : 3 strided 4x4 convs (32 -> 16 -> 8 -> 4 px, ReLU) -> flatten -> MLP -> z ∈ R^d
               no BatchNorm (the batch statistics would make export and single-frame
               inference awkward); the last layer is linear so z is unconstrained.
    predictor: MLP on concat(z_t, a_t), residual:  ẑ_{t+1} = z_t + MLP([z_t, a_t])
               ReLU activations, kept small because the browser planner calls it
               samples × horizon × iterations = 15 000 times per replanning step.

The multi-step ("open-loop") term makes the model good at exactly what the planner does:
rolling the predictor forward several steps without seeing new frames.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from ..losses import SIGReg
from .envs import WorldEnv, make_env


# ----------------------------------------------------------------------------- hyperparameters

@dataclass
class WorldHParams:
    """Model/optimisation settings that are not part of the shared run config (the config
    schema's `world` block only holds env/data/planning keys)."""

    latent_dim: int = 32
    enc_channels: tuple[int, int, int] = (16, 32, 32)
    enc_hidden: int = 128
    pred_hidden: int = 64
    pred_layers: int = 2          # hidden layers in the predictor MLP
    lam: float = 0.1              # SIGReg weight λ in (1-λ)·pred + λ·SIGReg
    num_slices: int = 64          # SIGReg random directions per step
    window: int = 4               # frames per training window -> window-1 open-loop steps
    batch: int = 64               # windows per step (-> batch*window frames through SIGReg)
    steps: int = 3000
    lr: float = 2e-3
    weight_decay: float = 1e-4
    warmup: int = 100
    threads: int = 2
    val_frac: float = 0.1


# Model/optimiser settings per environment, tuned on 2 CPU cores (see world-metrics.json).
# latent_dim 16 keeps one browser CEM step (300 samples x H5 x 10 iters) under ~100 ms;
# λ = 0.2 gave the best planning success for both envs in a small sweep (0.05/0.1/0.2/0.3);
# 6-8-frame windows = 5-7 step open-loop prediction loss, which matches how CEM uses the model.
HPARAMS_BY_ENV: dict[str, WorldHParams] = {
    "two-room": WorldHParams(latent_dim=16, lam=0.2, window=6, batch=48, steps=3000),
    "push-block": WorldHParams(latent_dim=16, lam=0.2, window=8, batch=32, steps=3000),
}

# ----------------------------------------------------------------------------- data

@dataclass
class WorldDataset:
    env: str
    obs: np.ndarray       # (E, T, 3, S, S) float32 in [0, 1]
    actions: np.ndarray   # (E, T-1, 2) float32 in [-1, 1]
    states: np.ndarray    # (E, T, state_dim) float64


def collect_dataset(env: WorldEnv | str, episodes: int, episode_len: int, seed: int) -> WorldDataset:
    """Roll out the env's scripted exploration policy.

    Returns obs (E,T,3,32,32), actions (E,T-1,2), states (E,T,state_dim). Episode e starts
    from ``env.reset(seed_e)`` where seed_e is drawn from a numpy Generator seeded by ``seed``.
    """
    env = make_env(env) if isinstance(env, str) else env
    rng = np.random.default_rng(seed)
    S = env.spec.image_size
    obs = np.empty((episodes, episode_len, 3, S, S), dtype=np.float32)
    acts = np.empty((episodes, episode_len - 1, 2), dtype=np.float32)
    states = np.empty((episodes, episode_len, env.spec.state_dim), dtype=np.float64)
    for e in range(episodes):
        s = env.reset(int(rng.integers(0, 2**31 - 1)))
        pol = env.policy(rng)
        for t in range(episode_len):
            states[e, t] = s
            obs[e, t] = env.render(s)
            if t < episode_len - 1:
                a = np.asarray(pol.act(s), dtype=np.float32)
                acts[e, t] = a
                s = env.step(s, a)
    return WorldDataset(env.spec.name, obs, acts, states)


# ----------------------------------------------------------------------------- model

class ConvEncoder(nn.Module):
    """32x32x3 -> R^d. Three stride-2 4x4 convs (padding 1) halve the resolution each time.

    Input standardisation x' = (x - mean_image) / pixel_std uses FIXED statistics of the
    training frames (buffers, exported to JSON). This matters a lot here: the frames are
    ~95% constant background, so without it every frame maps to almost the same latent at
    initialisation (spread ~5e-4). A constant batch is a stationary point of SIGReg (its
    gradient is proportional to the spread of the projections), and the prediction term
    pulls the same way, so training stays collapsed. With standardised inputs and Kaiming
    init the initial latent spread is O(1) and SIGReg keeps it there.
    """

    def __init__(self, latent_dim: int = 32, channels=(16, 32, 32), hidden: int = 128, image_size: int = 32):
        super().__init__()
        c1, c2, c3 = channels
        self.register_buffer("mean_image", torch.zeros(3, image_size, image_size))
        self.register_buffer("pixel_std", torch.ones(()))
        self.convs = nn.ModuleList([
            nn.Conv2d(3, c1, 4, 2, 1), nn.Conv2d(c1, c2, 4, 2, 1), nn.Conv2d(c2, c3, 4, 2, 1)])
        side = image_size // 8
        self.fc1 = nn.Linear(c3 * side * side, hidden)
        self.fc2 = nn.Linear(hidden, latent_dim)
        for m in [*self.convs, self.fc1]:
            nn.init.kaiming_normal_(m.weight, nonlinearity="relu")
            nn.init.zeros_(m.bias)
        nn.init.normal_(self.fc2.weight, std=1.0 / math.sqrt(hidden))
        nn.init.zeros_(self.fc2.bias)

    @torch.no_grad()
    def set_input_stats(self, frames: np.ndarray) -> None:
        """frames (N, 3, S, S): store the mean image and the scalar std of (frame - mean)."""
        mu = frames.mean(0, dtype=np.float64)
        sd = float(np.sqrt(((frames - mu) ** 2).mean(dtype=np.float64)))
        self.mean_image.copy_(torch.from_numpy(mu.astype(np.float32)))
        self.pixel_std.fill_(max(sd, 1e-6))

    def forward(self, x: Tensor) -> Tensor:
        x = (x - self.mean_image) / self.pixel_std
        for conv in self.convs:
            x = F.relu(conv(x))
        x = x.flatten(1)                    # (N, C*H*W) in C-major order (matches JS export)
        return self.fc2(F.relu(self.fc1(x)))


class Predictor(nn.Module):
    """ẑ_{t+1} = z_t + MLP([z_t, a_t]); MLP = (Linear, ReLU) × layers, Linear."""

    def __init__(self, latent_dim: int, action_dim: int = 2, hidden: int = 64, layers: int = 2):
        super().__init__()
        mods: list[nn.Module] = []
        d = latent_dim + action_dim
        for _ in range(layers):
            mods += [nn.Linear(d, hidden), nn.ReLU()]
            d = hidden
        mods.append(nn.Linear(d, latent_dim))
        self.net = nn.Sequential(*mods)
        # start near the identity map ("the world does not change"), which is the copy baseline
        nn.init.zeros_(self.net[-1].weight)
        nn.init.zeros_(self.net[-1].bias)

    def forward(self, z: Tensor, a: Tensor) -> Tensor:
        return z + self.net(torch.cat([z, a], dim=-1))


class WorldModel(nn.Module):
    def __init__(self, hp: WorldHParams | None = None, action_dim: int = 2, image_size: int = 32):
        super().__init__()
        self.hp = hp or WorldHParams()
        self.latent_dim = self.hp.latent_dim
        self.action_dim = action_dim
        self.image_size = image_size
        self.encoder = ConvEncoder(self.hp.latent_dim, self.hp.enc_channels, self.hp.enc_hidden, image_size)
        self.predictor = Predictor(self.hp.latent_dim, action_dim, self.hp.pred_hidden, self.hp.pred_layers)

    def encode(self, obs: Tensor) -> Tensor:
        """obs (..., 3, S, S) -> z (..., d)."""
        lead = obs.shape[:-3]
        return self.encoder(obs.reshape(-1, *obs.shape[-3:])).reshape(*lead, self.latent_dim)

    def predict(self, z: Tensor, a: Tensor) -> Tensor:
        return self.predictor(z, a)

    def rollout(self, z0: Tensor, actions: Tensor) -> Tensor:
        """Open-loop imagination. z0 (N, d), actions (N, H, 2) -> (N, H, d) = ẑ_1..ẑ_H."""
        out, z = [], z0
        for h in range(actions.shape[1]):
            z = self.predictor(z, actions[:, h])
            out.append(z)
        return torch.stack(out, dim=1)


# ----------------------------------------------------------------------------- loss

def world_loss(model: WorldModel, obs: Tensor, actions: Tensor, sigreg: SIGReg, lam: float
               ) -> tuple[Tensor, Tensor, Tensor]:
    """obs (B, W, 3, S, S), actions (B, W-1, 2).

    z_k = f(o_k) for every frame (gradients flow into ALL of them, incl. the targets).
    ẑ_k = g^k(z_0, a_0..a_{k-1}) is the open-loop rollout from the first frame.
        pred   = mean_{b, k=1..W-1, dim} (ẑ_k - z_k)²
        sigreg = SIGReg over all B·W latents
        loss   = (1-λ) pred + λ sigreg
    """
    B, W = obs.shape[:2]
    z = model.encode(obs)                           # (B, W, d)
    zhat = model.rollout(z[:, 0], actions)          # (B, W-1, d)
    pred = (zhat - z[:, 1:]).pow(2).mean()
    sig = sigreg(z.reshape(B * W, -1))
    return (1 - lam) * pred + lam * sig, pred, sig


def _windows(n_ep: int, T: int, W: int) -> np.ndarray:
    """All (episode, start) pairs of length-W windows."""
    e, s = np.meshgrid(np.arange(n_ep), np.arange(T - W + 1), indexing="ij")
    return np.stack([e.ravel(), s.ravel()], 1)


# ----------------------------------------------------------------------------- evaluation

@torch.no_grad()
def encode_all(model: WorldModel, obs: np.ndarray, batch: int = 1024) -> np.ndarray:
    flat = torch.from_numpy(obs.reshape(-1, *obs.shape[-3:]))
    zs = [model.encode(flat[i:i + batch]) for i in range(0, len(flat), batch)]
    return torch.cat(zs).numpy().reshape(*obs.shape[:-3], model.latent_dim)


@torch.no_grad()
def prediction_report(model: WorldModel, obs: np.ndarray, actions: np.ndarray, horizon: int) -> dict:
    """Open-loop k-step latent prediction error on held-out episodes vs "copy last latent".

        err_model(k) = mean || g^k(z_t, a_t..a_{t+k-1}) - z_{t+k} ||²   (per latent dim)
        err_copy(k)  = mean || z_t - z_{t+k} ||²
    ratio < 1 means the model uses the action to anticipate the future.
    """
    model.eval()
    z = torch.from_numpy(encode_all(model, obs))        # (E, T, d)
    a = torch.from_numpy(actions)
    E, T, d = z.shape
    H = min(horizon, T - 1)
    starts = torch.arange(0, T - H)
    z0 = z[:, starts].reshape(-1, d)
    acts = torch.stack([a[:, s:s + H] for s in starts.tolist()], 1).reshape(-1, H, 2)
    tgt = torch.stack([z[:, s + 1:s + 1 + H] for s in starts.tolist()], 1).reshape(-1, H, d)
    roll = model.rollout(z0, acts)
    err_m = (roll - tgt).pow(2).mean(dim=(0, 2))
    err_c = (z0[:, None] - tgt).pow(2).mean(dim=(0, 2))
    return {"horizon": H,
            "model_mse": [round(float(v), 5) for v in err_m],
            "copy_mse": [round(float(v), 5) for v in err_c],
            "ratio": [round(float(m / max(c, 1e-12)), 4) for m, c in zip(err_m, err_c)]}


@torch.no_grad()
def collapse_report(model: WorldModel, obs: np.ndarray, states: np.ndarray) -> dict:
    """Is the latent space alive? std per dim, effective rank, and how linearly decodable the
    true (hidden) env state is from z (ridge R²; ~1 means the encoder found the positions)."""
    z = encode_all(model, obs).reshape(-1, model.latent_dim).astype(np.float64)
    s = states.reshape(-1, states.shape[-1])
    zc = z - z.mean(0)
    cov = zc.T @ zc / len(zc)
    ev = np.clip(np.linalg.eigvalsh(cov), 1e-12, None)
    p = ev / ev.sum()
    eff_rank = float(np.exp(-(p * np.log(p)).sum()))
    X = np.concatenate([zc, np.ones((len(zc), 1))], 1)
    n = len(X) // 2   # fit on half, score on the other half
    w = np.linalg.solve(X[:n].T @ X[:n] + 1e-3 * np.eye(X.shape[1]), X[:n].T @ s[:n])
    res = s[n:] - X[n:] @ w
    r2 = 1 - res.var(0) / np.maximum(s[n:].var(0), 1e-12)
    std = np.sqrt(np.diag(cov))
    return {"latent_std_mean": round(float(std.mean()), 4), "latent_std_min": round(float(std.min()), 4),
            "effective_rank": round(eff_rank, 2), "state_probe_r2": [round(float(v), 4) for v in r2],
            "collapsed": bool(std.mean() < 0.05 or eff_rank < 1.5)}


# ----------------------------------------------------------------------------- training

@dataclass
class TrainResult:
    model: WorldModel
    metrics: dict
    data: WorldDataset
    val_idx: np.ndarray = field(default_factory=lambda: np.zeros(0, dtype=int))


def _emit(path: Path | None, event: str, **data) -> None:
    if path is None:
        return
    with open(path, "a", encoding="utf-8") as f:
        f.write(json.dumps({"t": round(time.time(), 3), "event": event, **data}) + "\n")


def train(cfg: dict, log: Callable[[str], None] = print, events_path: str | Path | None = None,
          hp: WorldHParams | None = None, data: WorldDataset | None = None) -> TrainResult:
    """Full training run (see ``train_world_model`` for the public wrapper)."""
    hp = hp or WorldHParams()
    w = cfg["world"]
    seed = int(cfg.get("seed", 0))
    events_path = Path(events_path) if events_path else None
    torch.set_num_threads(hp.threads)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    t0 = time.time()
    if data is None:
        log(f"[world] collecting {w['episodes']} episodes x {w['episode_len']} steps of {w['env']}")
        data = collect_dataset(w["env"], w["episodes"], w["episode_len"], seed)
    E, T = data.obs.shape[:2]
    n_val = max(1, int(round(E * hp.val_frac)))
    perm = rng.permutation(E)
    val_idx, tr_idx = np.sort(perm[:n_val]), np.sort(perm[n_val:])
    t_data = time.time() - t0
    _emit(events_path, "data", env=data.env, episodes=int(E), episode_len=int(T), seconds=round(t_data, 2))

    W = min(hp.window, T)
    win = _windows(len(tr_idx), T, W)
    obs_t = torch.from_numpy(data.obs[tr_idx])
    act_t = torch.from_numpy(data.actions[tr_idx])
    model = WorldModel(hp, image_size=data.obs.shape[-1])
    sub = data.obs[tr_idx].reshape(-1, *data.obs.shape[-3:])
    model.encoder.set_input_stats(sub[rng.permutation(len(sub))[:4096]])
    sigreg = SIGReg(num_slices=hp.num_slices, seed=seed)
    opt = torch.optim.AdamW(model.parameters(), lr=hp.lr, weight_decay=hp.weight_decay)
    n_params = sum(p.numel() for p in model.parameters())
    log(f"[world] model: {n_params} params, latent {hp.latent_dim}, λ={hp.lam}, {hp.steps} steps")
    _emit(events_path, "start", total=hp.steps, params=int(n_params), hparams=asdict(hp))

    history = []
    t1 = time.time()
    model.train()
    ar = torch.arange(W)
    for step in range(hp.steps):
        # linear warmup then cosine decay to 1% of the base learning rate
        if step < hp.warmup:
            lr = hp.lr * (step + 1) / hp.warmup
        else:
            p = (step - hp.warmup) / max(1, hp.steps - hp.warmup)
            lr = hp.lr * (0.01 + 0.99 * 0.5 * (1 + math.cos(math.pi * p)))
        for g in opt.param_groups:
            g["lr"] = lr
        pick = torch.from_numpy(win[rng.integers(0, len(win), hp.batch)])
        ep, st = pick[:, 0], pick[:, 1]
        idx_t = st[:, None] + ar[None]                              # (B, W)
        o = obs_t[ep[:, None], idx_t]                               # (B, W, 3, S, S)
        a = act_t[ep[:, None], idx_t[:, :-1]]                       # (B, W-1, 2)
        loss, pred, sig = world_loss(model, o, a, sigreg, hp.lam)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if step % 50 == 0 or step == hp.steps - 1:
            el = time.time() - t1
            rec = {"step": step, "epoch": round(step * hp.batch / max(1, len(win)), 3),
                   "loss": round(loss.item(), 5), "pred": round(pred.item(), 5),
                   "sigreg": round(sig.item(), 5), "lr": round(lr, 6),
                   "samples_per_s": round((step + 1) * hp.batch * W / max(el, 1e-6), 1)}
            history.append(rec)
            _emit(events_path, "step", **rec)
            if step % 500 == 0 or step == hp.steps - 1:
                log(f"[world] step {step:5d} loss {rec['loss']:.4f} pred {rec['pred']:.4f} sigreg {rec['sigreg']:.4f}")
    train_s = time.time() - t1
    model.eval()

    metrics = {
        "env": data.env, "params": int(n_params), "hparams": asdict(hp),
        "episodes": int(E), "episode_len": int(T), "train_seconds": round(train_s, 1),
        "data_seconds": round(t_data, 1),
        "final": history[-1], "history": history,
    }
    metrics["prediction_val"] = prediction_report(model, data.obs[val_idx], data.actions[val_idx], w["horizon"])
    metrics["latent_val"] = collapse_report(model, data.obs[val_idx], data.states[val_idx])
    _emit(events_path, "end", train_seconds=metrics["train_seconds"],
          prediction_val=metrics["prediction_val"], latent_val=metrics["latent_val"])
    return TrainResult(model, metrics, data, val_idx)


def train_world_model(cfg: dict, log: Callable[[str], None] = print,
                      events_path: str | Path | None = None,
                      hp: WorldHParams | None = None) -> tuple[WorldModel, dict]:
    """Collect data for cfg["world"]["env"], train the JEPA world model, evaluate it.

    Returns (model, metrics). metrics["prediction_val"] compares open-loop prediction with the
    copy-last-latent baseline; metrics["latent_val"] reports collapse diagnostics.
    """
    hp = hp or HPARAMS_BY_ENV.get(cfg["world"]["env"], WorldHParams())
    res = train(cfg, log, events_path, hp)
    return res.model, res.metrics
