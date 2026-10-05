"""World model + planner tests (small and fast: tiny data, a few training steps)."""
from __future__ import annotations

import copy
import json

import numpy as np
import pytest
import torch

from jepa_studio.config import DEFAULT_CONFIG
from jepa_studio.losses import SIGReg
from jepa_studio.world import (CEMConfig, PushBlock, TwoRoom, WorldHParams, WorldModel, build_bank,
                               collect_dataset, make_env)
from jepa_studio.world.envs import AGENT_R, Mulberry32, rollout
from jepa_studio.world.export import FORMAT, export_json, load_json_model
from jepa_studio.world.model import train, world_loss
from jepa_studio.world.planner import plan_and_execute, plan_cem, random_policy_episode, trajectory_cost

ENV_NAMES = ["two-room", "push-block"]


# ----------------------------------------------------------------------------- environments

def test_mulberry32_known_values():
    r = Mulberry32(0)
    vals = [r.next() for _ in range(3)]
    # reference values from the canonical JS implementation (mulberry32(0))
    assert vals == pytest.approx([0.26642920868471265, 0.0003297457005828619, 0.2232720274478197], abs=0)


@pytest.mark.parametrize("name", ENV_NAMES)
def test_step_is_pure_and_deterministic(name):
    env = make_env(name)
    s = env.reset(7)
    s_copy = s.copy()
    a = np.array([0.3, -0.8])
    n1 = env.step(s, a)
    n2 = env.step(s, a)
    np.testing.assert_array_equal(s, s_copy)          # input not mutated
    np.testing.assert_array_equal(n1, n2)             # same output
    assert n1 is not s
    np.testing.assert_array_equal(env.reset(7), env.reset(7))
    np.testing.assert_array_equal(env.goal_state(7), env.goal_state(7))
    assert not np.array_equal(env.reset(7), env.reset(8))


@pytest.mark.parametrize("name", ENV_NAMES)
def test_render_shape_range(name):
    env = make_env(name)
    for seed in range(5):
        img = env.render(env.reset(seed))
        assert img.shape == (3, 32, 32) and img.dtype == np.float32
        assert img.min() >= 0.0 and img.max() <= 1.0
        # the agent is visible: some pixel is close to the agent color
        assert (np.abs(img - np.array([0.90, 0.25, 0.20], np.float32)[:, None, None]).max(0) < 0.05).any()


def test_two_room_wall_blocks_except_at_door():
    env = TwoRoom()
    # far from the door: pushing right forever never crosses the wall
    s = np.array([0.2, 0.15])
    for _ in range(30):
        s = env.step(s, [1.0, 0.0])
    assert s[0] < env.WALL_X - env.WALL_HALF - AGENT_R + 1e-6
    assert s[0] > 0.4                                   # but it did reach the wall face
    # sliding along the wall face vertically works
    s2 = env.step(s, [0.0, 1.0])
    assert s2[1] > s[1] and s2[0] == s[0]
    # at the door height the agent passes through
    s = np.array([0.2, env.DOOR_Y])
    for _ in range(10):
        s = env.step(s, [1.0, 0.0])
    assert s[0] > env.WALL_X + env.WALL_HALF + AGENT_R
    # inside the door corridor it cannot move up into the wall
    s = np.array([env.WALL_X, env.DOOR_Y])
    for _ in range(10):
        s = env.step(s, [0.0, -1.0])
    assert abs(s[1] - env.DOOR_Y) < env.DOOR_HALF - AGENT_R
    assert env.free(*s)


def test_two_room_never_inside_wall_random_actions():
    env = TwoRoom()
    rng = np.random.default_rng(0)
    for seed in range(20):
        s = env.reset(seed)
        for _ in range(60):
            s = env.step(s, rng.uniform(-1, 1, 2))
            assert env.free(s[0], s[1]) or np.isclose(abs(s[0] - env.WALL_X), env.WALL_HALF + AGENT_R, atol=1e-6)


def test_push_moves_block():
    env = PushBlock()
    r, h = env.AGENT_R, env.BLOCK_HALF
    s = np.array([0.5 - h - r - 0.01, 0.5, 0.5, 0.5])
    s1 = rollout(env, s, [[1.0, 0.0]] * 3)[-1]
    assert s1[2] > 0.5 + 0.1                       # block moved right
    assert s1[3] == pytest.approx(0.5)             # not sideways
    assert s1[2] - s1[0] >= h + r - 1e-6           # no interpenetration
    # pushing into the border: block clamped, agent stops behind it
    s = np.array([0.5, 0.5, 0.8, 0.5])
    for _ in range(20):
        s = env.step(s, [1.0, 0.0])
    assert s[2] == pytest.approx(1 - h)
    assert s[2] - s[0] >= h + r - 1e-6
    # moving away never drags the block
    s2 = env.step(s, [-1.0, 0.0])
    assert s2[2] == s[2]


def test_goal_reachability_and_success():
    for env in (TwoRoom(), PushBlock()):
        g = env.goal_state(3)
        assert env.success(g, g)
        assert not env.success(env.reset(3), g)


# ----------------------------------------------------------------------------- model

def _tiny_cfg(name="two-room"):
    cfg = copy.deepcopy(DEFAULT_CONFIG)
    cfg["world"].update(env=name, episodes=12, episode_len=10)
    return cfg


def test_collect_dataset_shapes():
    d = collect_dataset("push-block", 3, 6, seed=0)
    assert d.obs.shape == (3, 6, 3, 32, 32) and d.obs.dtype == np.float32
    assert d.actions.shape == (3, 5, 2) and np.abs(d.actions).max() <= 1
    assert d.states.shape == (3, 6, 4)
    env = PushBlock()
    np.testing.assert_allclose(env.step(d.states[1, 2], d.actions[1, 2]), d.states[1, 3], atol=1e-6)
    d2 = collect_dataset("push-block", 3, 6, seed=0)
    np.testing.assert_array_equal(d.obs, d2.obs)


def test_train_no_nan_and_sigreg_decreases():
    torch.manual_seed(0)
    hp = WorldHParams(latent_dim=16, enc_channels=(8, 8, 8), enc_hidden=32, pred_hidden=32, steps=120,
                      batch=16, window=3, warmup=10, lr=3e-3, num_slices=32)
    res = train(_tiny_cfg(), log=lambda s: None, hp=hp)
    hist = res.metrics["history"]
    assert all(np.isfinite([h["loss"], h["pred"], h["sigreg"]]).all() for h in hist)
    assert hist[-1]["sigreg"] < hist[0]["sigreg"]
    assert not res.metrics["latent_val"]["collapsed"]
    for p in res.model.parameters():
        assert torch.isfinite(p).all()


def test_gradients_flow_through_targets():
    """LeWM is end-to-end: the target latents z_{t+1} receive gradient too (no stop-grad)."""
    m = WorldModel(WorldHParams(latent_dim=8, enc_channels=(4, 4, 4), enc_hidden=16, pred_hidden=16))
    d = collect_dataset("two-room", 2, 3, seed=1)
    obs = torch.from_numpy(d.obs).requires_grad_(True)
    loss, _, _ = world_loss(m, obs, torch.from_numpy(d.actions), SIGReg(16), lam=0.0)
    loss.backward()
    assert obs.grad[:, 1:].abs().sum() > 0          # frames used only as targets


# ----------------------------------------------------------------------------- planner

class _StateModel:
    """Identity 'encoder' + TRUE dynamics as predictor: isolates the CEM planner."""
    action_dim = 2

    def __init__(self, env):
        self.env = env

    def rollout(self, z0, acts):
        N, H, _ = acts.shape
        out = np.empty((N, H, z0.shape[-1]))
        z = z0.numpy().astype(np.float64)
        a = acts.numpy()
        for i in range(N):
            s = z[i]
            for h in range(H):
                s = self.env.step(s, a[i, h])
                out[i, h] = s
        return torch.from_numpy(out).float()


def test_trajectory_cost_kinds():
    zs = torch.tensor([[[1.0, 0.0], [0.0, 0.0]]])
    g = torch.zeros(2)
    assert trajectory_cost(zs, g, "sum").item() == 1.0
    assert trajectory_cost(zs, g, "final").item() == 0.0


def test_cem_beats_random_with_true_dynamics():
    env = TwoRoom()
    model = _StateModel(env)
    cfg = CEMConfig(horizon=5, samples=64, elites=8, iters=4)
    rng = np.random.default_rng(0)
    cem_ok, rnd_ok = 0, 0
    for seed in range(6):
        s, g = env.reset(seed), env.goal_state(seed)
        gen = torch.Generator().manual_seed(seed)
        mean = None
        for _ in range(25):
            if env.success(s, g):
                break
            out = plan_cem(model, torch.tensor(s).float(), torch.tensor(g).float(), cfg, gen, mean)
            mean = torch.cat([out["mean"][1:], torch.zeros(1, 2)])
            s = env.step(s, out["actions"][0].numpy())
        cem_ok += env.success(s, g)
        rnd_ok += random_policy_episode(env, seed, seed, 25, rng)["success"]
    assert cem_ok >= 5 and cem_ok > rnd_ok


def test_plan_and_execute_with_tiny_model_and_export(tmp_path):
    hp = WorldHParams(latent_dim=8, enc_channels=(4, 8, 8), enc_hidden=32, pred_hidden=16, steps=20,
                      batch=8, window=3, warmup=2, num_slices=16)
    res = train(_tiny_cfg("push-block"), log=lambda s: None, hp=hp)
    env = make_env("push-block")
    bank = build_bank(res.model, env, res.data.states, 50)
    cfg = CEMConfig(horizon=3, samples=16, elites=4, iters=2)
    r = plan_and_execute(env, res.model, 5, 5, max_steps=3, cfg=cfg, bank=bank)
    T = r["steps"]
    assert r["frames_real"].shape == (T + 1, 3, 32, 32)
    assert len(r["imagined"]) == T and r["imagined"][0].shape == (3, 8)
    assert len(r["retrieved_frames"]) == T and r["retrieved_frames"][0].shape == (3, 3, 32, 32)
    assert len(r["costs"]) == T and isinstance(r["success"], bool)

    path = tmp_path / "world-push-block.json"
    export_json(res.model, "push-block", bank, cfg, {"note": "test"}, path)
    doc = load_json_model(path)
    assert doc["format"] == FORMAT and doc["env"] == "push-block"
    assert doc["latent_dim"] == 8 and doc["action_dim"] == 2 and doc["image_size"] == 32
    assert len(doc["bank"]["latents"]) == 50 * 8 and len(doc["bank"]["states"]) == 50
    kinds = [L["type"] for L in doc["encoder"]["layers"]]
    assert kinds[0] == "normalize" and "conv2d" in kinds and kinds[-1] == "linear"
    lin = [L for L in doc["predictor"]["layers"] if L["type"] == "linear"]
    assert lin[0]["in"] == 10 and lin[-1]["out"] == 8
    assert len(lin[0]["W"]) == lin[0]["in"] * lin[0]["out"]
    assert doc["plan_defaults"]["horizon"] == 3
    json.dumps(doc)  # round-trips


def test_run_world_writes_outputs(tmp_path):
    from jepa_studio.weights import load_tensors
    from jepa_studio.world import run_world

    hp = WorldHParams(latent_dim=8, enc_channels=(4, 4, 4), enc_hidden=16, pred_hidden=16, steps=10,
                      batch=8, window=3, warmup=2, num_slices=16)
    cfg = _tiny_cfg("two-room")
    cfg["world"].update(cem_samples=16, cem_elites=4, cem_iters=2, horizon=3)
    m = run_world(cfg, tmp_path, log=lambda s: None, hp=hp, eval_episodes=2)
    assert (tmp_path / "world.safetensors").exists()
    assert load_tensors(tmp_path / "world.safetensors")
    met = json.loads((tmp_path / "metrics.json").read_text())
    assert "planning" in met and "prediction_val" in met
    assert (tmp_path / "world-two-room.json").exists()
    assert not list(tmp_path.glob("*.pt")) and not list(tmp_path.glob("*.pkl"))
    assert 0.0 <= m["planning"]["cem_success_rate"] <= 1.0
