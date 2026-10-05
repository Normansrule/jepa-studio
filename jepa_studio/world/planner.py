"""Planning in latent space with the cross-entropy method (CEM), receding horizon (MPC).

Given the current frame o_t and a goal frame o_g, encode z_t = f(o_t), z_g = f(o_g) and search
for an action sequence a_{0..H-1} ∈ [-1, 1]^{H×2} whose imagined latent trajectory
ẑ_{h+1} = g(ẑ_h, a_h), ẑ_0 = z_t, ends near z_g.

Cost (default "sum", the one reported in site/models/world-metrics.json):

    C(a) = Σ_{h=1..H} || ẑ_h - z_g ||²

Why the sum rather than the final step only ("final": C = ||ẑ_H - z_g||²): under receding
horizon we only execute the first action, so what matters is that the *first* step is a good
one. The sum rewards getting close early and staying close (it penalises overshooting and
"arrive at the last moment" plans that the final-step cost is indifferent to), and it gives
a useful signal when the goal is farther than H steps away (every step of progress counts,
not only the endpoint). Both are available; `evaluate_planner` can compare them.

CEM (one replanning step):
    μ ∈ R^{H×2} (warm-started from the previous plan shifted by one step), σ = σ_0
    repeat `iters` times:
        a^(i) = clip(μ + σ ⊙ ε^(i), -1, 1),  ε^(i) ~ N(0, I),  i = 1..N   (sample 0 = μ itself)
        E     = the `elites` samples with the lowest C(a^(i))
        μ ← mean(E),   σ ← std(E) + σ_min       (σ_min keeps the search from freezing)
    return the best sample seen over all rounds (its first action is executed; μ warm-starts the next step)

Visualising imagination: a JEPA has no decoder, so imagined latents cannot be turned back
into pixels. Instead we show *retrieval*: for each imagined latent, the nearest latent in a
bank of encoded REAL frames from the training data, and that real frame. It is labelled as
retrieval everywhere it is shown.
"""
from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np
import torch

from .envs import WorldEnv
from .model import WorldModel


@dataclass
class CEMConfig:
    horizon: int = 5
    samples: int = 300
    elites: int = 30
    iters: int = 10
    init_std: float = 1.0
    min_std: float = 0.05
    cost: str = "sum"            # "sum" | "final"

    @classmethod
    def from_world_cfg(cls, w: dict, **kw) -> "CEMConfig":
        return cls(horizon=w["horizon"], samples=w["cem_samples"], elites=w["cem_elites"],
                   iters=w["cem_iters"], **kw)


def trajectory_cost(zs: torch.Tensor, z_goal: torch.Tensor, kind: str = "sum") -> torch.Tensor:
    """zs (N, H, d) imagined latents, z_goal (d,) -> (N,) costs."""
    d2 = (zs - z_goal).pow(2).sum(-1)            # (N, H)
    if kind == "sum":
        return d2.sum(-1)
    if kind == "final":
        return d2[:, -1]
    raise ValueError(f"unknown cost {kind!r}")


@torch.no_grad()
def plan_cem(model: WorldModel, z0: torch.Tensor, z_goal: torch.Tensor, cfg: CEMConfig,
             gen: torch.Generator, init_mean: torch.Tensor | None = None) -> dict:
    """One CEM search. z0, z_goal: (d,). Returns actions (H,2), imagined (H,d), cost history."""
    H, A = cfg.horizon, model.action_dim
    mu = torch.zeros(H, A) if init_mean is None else init_mean.clone()
    std = torch.full((H, A), cfg.init_std)
    z0b = z0.expand(cfg.samples, -1)
    hist = []
    best_a, best_c = mu.clone(), float("inf")
    for _ in range(cfg.iters):
        eps = torch.randn(cfg.samples, H, A, generator=gen)
        eps[0] = 0.0                                     # always evaluate the current mean
        acts = (mu + std * eps).clamp(-1.0, 1.0)
        zs = model.rollout(z0b, acts)
        c = trajectory_cost(zs, z_goal, cfg.cost)
        top = torch.topk(c, cfg.elites, largest=False).indices
        elite = acts[top]
        mu = elite.mean(0)
        std = elite.std(0, unbiased=False) + cfg.min_std
        hist.append(float(c[top].mean()))
        if float(c[top[0]]) < best_c:
            best_c, best_a = float(c[top[0]]), acts[top[0]].clone()
    imagined = model.rollout(z0[None], best_a[None])[0]
    return {"actions": best_a, "mean": mu, "imagined": imagined, "cost_history": hist, "cost": best_c}


# ----------------------------------------------------------------------------- retrieval bank

@dataclass
class LatentBank:
    """Encoded real frames used to *retrieve* (not generate) pictures for imagined latents."""
    latents: np.ndarray    # (B, d) float32
    states: np.ndarray     # (B, state_dim) env states; frames are re-rendered from these

    def nearest(self, z: np.ndarray) -> np.ndarray:
        """z (..., d) -> index (...) of the nearest bank latent (squared Euclidean)."""
        z = np.asarray(z, dtype=np.float32)
        flat = z.reshape(-1, z.shape[-1])
        d2 = (flat ** 2).sum(1, keepdims=True) - 2 * flat @ self.latents.T + (self.latents ** 2).sum(1)[None]
        return d2.argmin(1).reshape(z.shape[:-1])


@torch.no_grad()
def build_bank(model: WorldModel, env: WorldEnv, states: np.ndarray, size: int, seed: int = 0) -> LatentBank:
    """Pick `size` distinct training states (uniformly at random), render, encode."""
    flat = states.reshape(-1, states.shape[-1])
    idx = np.random.default_rng(seed).choice(len(flat), size=min(size, len(flat)), replace=False)
    st = flat[np.sort(idx)]
    frames = torch.from_numpy(np.stack([env.render(s) for s in st]))
    z = model.encode(frames).numpy().astype(np.float32)
    return LatentBank(z, st)


# ----------------------------------------------------------------------------- MPC loop

@torch.no_grad()
def plan_and_execute(env: WorldEnv, model: WorldModel, start_seed: int, goal_seed: int | None = None,
                     max_steps: int = 40, cfg: CEMConfig | None = None, bank: LatentBank | None = None,
                     seed: int = 0, record: bool = True) -> dict:
    """Receding-horizon control: plan H steps with CEM, execute the first action in the REAL
    env, re-encode the new real frame, replan (warm-starting from the shifted plan).

    Returns
        states          (T+1, state_dim) real trajectory
        frames_real     (T+1, 3, S, S)   rendered real frames           (if record)
        goal_state, goal_frame
        imagined        list over replanning steps of (H, d) imagined latents
        retrieved       list over replanning steps of (H,) bank indices  (if bank given)
        retrieved_frames list of (H, 3, S, S) re-rendered bank frames   (RETRIEVAL, not generation)
        costs           cost of the best sample per replanning step
        latent_dist     ||z_t - z_goal||² of the real frame per step
        success, steps, plan_ms
    """
    cfg = cfg or CEMConfig()
    goal_seed = start_seed if goal_seed is None else goal_seed
    gen = torch.Generator().manual_seed(seed)
    state = env.reset(start_seed)
    goal = env.goal_state(goal_seed)
    z_goal = model.encode(torch.from_numpy(env.render(goal))[None])[0]
    states, frames = [state], [env.render(state)] if record else []
    imagined, retrieved, costs, ldist, plan_ms = [], [], [], [], []
    mean = None
    success = env.success(state, goal)
    t = 0
    while not success and t < max_steps:
        frame = env.render(state)
        z = model.encode(torch.from_numpy(frame)[None])[0]
        ldist.append(float((z - z_goal).pow(2).sum()))
        t0 = time.perf_counter()
        out = plan_cem(model, z, z_goal, cfg, gen, mean)
        plan_ms.append((time.perf_counter() - t0) * 1000)
        a = out["actions"][0].numpy()
        mean = torch.cat([out["mean"][1:], torch.zeros(1, model.action_dim)])   # shift warm start
        costs.append(out["cost"])
        if record:
            imagined.append(out["imagined"].numpy())
            if bank is not None:
                retrieved.append(bank.nearest(out["imagined"].numpy()))
        state = env.step(state, a)
        states.append(state)
        if record:
            frames.append(env.render(state))
        success = env.success(state, goal)
        t += 1
    res = {"states": np.stack(states), "goal_state": goal, "success": bool(success), "steps": t,
           "costs": costs, "latent_dist": ldist, "plan_ms": float(np.mean(plan_ms)) if plan_ms else 0.0,
           "final_distance": env.distance(state, goal)}
    if record:
        res.update(frames_real=np.stack(frames), goal_frame=env.render(goal), imagined=imagined,
                   retrieved=retrieved)
        if bank is not None:
            res["retrieved_frames"] = [np.stack([env.render(bank.states[i]) for i in r]) for r in retrieved]
    return res


def random_policy_episode(env: WorldEnv, start_seed: int, goal_seed: int | None, max_steps: int,
                          rng: np.random.Generator) -> dict:
    """Baseline: uniform random actions in [-1, 1]^2 (success is checked every step)."""
    goal_seed = start_seed if goal_seed is None else goal_seed
    s, g = env.reset(start_seed), env.goal_state(goal_seed)
    for t in range(max_steps):
        if env.success(s, g):
            return {"success": True, "steps": t, "final_distance": env.distance(s, g)}
        s = env.step(s, rng.uniform(-1, 1, 2))
    return {"success": bool(env.success(s, g)), "steps": max_steps, "final_distance": env.distance(s, g)}


def evaluate_planner(env: WorldEnv, model: WorldModel, cfg: CEMConfig, episodes: int = 20,
                     max_steps: int = 40, first_seed: int = 10_000, log=print) -> dict:
    """CEM-MPC vs random actions on the same `episodes` start/goal pairs (seeds first_seed+i,
    which are disjoint from the data-collection seeds in practice)."""
    rng = np.random.default_rng(first_seed)
    cem, rnd, init_d = [], [], []
    for i in range(episodes):
        sd = first_seed + i
        init_d.append(env.distance(env.reset(sd), env.goal_state(sd)))
        r = plan_and_execute(env, model, sd, sd, max_steps, cfg, None, seed=sd, record=False)
        cem.append(r)
        rnd.append(random_policy_episode(env, sd, sd, max_steps, rng))
    out = {
        "episodes": episodes, "max_steps": max_steps, "cost": cfg.cost,
        "cem": {"horizon": cfg.horizon, "samples": cfg.samples, "elites": cfg.elites, "iters": cfg.iters},
        "cem_success_rate": float(np.mean([r["success"] for r in cem])),
        "random_success_rate": float(np.mean([r["success"] for r in rnd])),
        "cem_mean_final_distance": round(float(np.mean([r["final_distance"] for r in cem])), 4),
        "random_mean_final_distance": round(float(np.mean([r["final_distance"] for r in rnd])), 4),
        "mean_initial_distance": round(float(np.mean(init_d)), 4),
        "cem_mean_steps_success": round(float(np.mean([r["steps"] for r in cem if r["success"]] or [0])), 2),
        "python_plan_ms": round(float(np.mean([r["plan_ms"] for r in cem])), 1),
        "seeds": [first_seed + i for i in range(episodes)],
        "cem_success": [bool(r["success"]) for r in cem],
        "random_success": [bool(r["success"]) for r in rnd],
    }
    log(f"[world] {env.spec.name}: CEM success {out['cem_success_rate']:.2f} vs random "
        f"{out['random_success_rate']:.2f} (final dist {out['cem_mean_final_distance']} vs "
        f"{out['random_mean_final_distance']}, start {out['mean_initial_distance']})")
    return out
