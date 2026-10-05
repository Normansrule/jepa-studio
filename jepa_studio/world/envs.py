"""Two tiny deterministic 2-D environments rendered as 32x32 RGB images.

Both environments are *pure*: ``step(state, action)`` never mutates anything and always
returns the same next state for the same inputs, so the world model, the planner and the
browser port (site/js/world/env.js) can all replay trajectories exactly.

Coordinates live in the unit square [0, 1]^2 with y pointing DOWN (row 0 of the image is
y = 0). Pixel (row i, column j) is sampled at its center ((j + 0.5)/S, (i + 0.5)/S).

Rendering is deliberately simple so JavaScript reproduces it bit-for-bit (all arithmetic is
float64, then rounded once to float32):

    smoothstep(e0, e1, x) = t^2 (3 - 2t),   t = clamp((x - e0)/(e1 - e0), 0, 1)
    disc coverage         = 1 - smoothstep(r - aa, r + aa, |p - c|)       aa = half a pixel
    box coverage          = (1 - smoothstep(h - aa, h + aa, |px - cx|))
                          * (1 - smoothstep(h - aa, h + aa, |py - cy|))
    wall                  = hard inside test (|px - cx| < hw and |py - cy| >= door half)
    compositing           = img <- img * (1 - cov) + color * cov   (painter's order)

Randomness for reset()/goal_state() comes from mulberry32 (a 32-bit PRNG that is trivial to
port), so ``reset(seed)`` gives the same start in Python and in the browser. The exploration
policies used to collect training data use numpy's Generator (Python only).

Actions are continuous in [-1, 1]^2 and are interpreted as a velocity scaled by MAX_SPEED.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

IMAGE_SIZE = 32
MAX_SPEED = 0.08          # arena units per step at |action| = 1
AGENT_R = 0.04            # agent disc radius (also its collision radius)
EPS = 1e-9                # contact offset so an agent resting on a wall face is "free"
AA = 0.5 / IMAGE_SIZE     # anti-aliasing half-width (half a pixel)

# colors (RGB in [0, 1])
BG = (0.93, 0.93, 0.90)
WALL = (0.25, 0.25, 0.30)
AGENT = (0.90, 0.25, 0.20)
BLOCK = (0.20, 0.45, 0.85)

_M32 = 0xFFFFFFFF


# ----------------------------------------------------------------------------- PRNG

class Mulberry32:
    """mulberry32 PRNG, identical to the JS one-liner (uint32 arithmetic emulated with masks)."""

    def __init__(self, seed: int):
        self.a = int(seed) & _M32

    @staticmethod
    def _imul(x: int, y: int) -> int:
        return (x * y) & _M32

    def next(self) -> float:
        """Uniform float in [0, 1) with 32 bits of randomness."""
        self.a = (self.a + 0x6D2B79F5) & _M32
        t = self.a
        t = self._imul(t ^ (t >> 15), t | 1)
        t = ((t + self._imul(t ^ (t >> 7), t | 61)) & _M32) ^ t
        return ((t ^ (t >> 14)) & _M32) / 4294967296.0

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.next()


# ----------------------------------------------------------------------------- drawing helpers

def _pixel_centers(size: int = IMAGE_SIZE) -> tuple[np.ndarray, np.ndarray]:
    c = (np.arange(size, dtype=np.float64) + 0.5) / size
    px = np.broadcast_to(c[None, :], (size, size))   # x varies along columns
    py = np.broadcast_to(c[:, None], (size, size))   # y varies along rows
    return px, py


def smoothstep(e0: float, e1: float, x: np.ndarray) -> np.ndarray:
    t = np.clip((x - e0) / (e1 - e0), 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


def _blend(img: np.ndarray, cov: np.ndarray, color: tuple[float, float, float]) -> None:
    for ch in range(3):
        img[ch] = img[ch] * (1.0 - cov) + color[ch] * cov


def _disc(img: np.ndarray, cx: float, cy: float, r: float, color) -> None:
    px, py = _pixel_centers(img.shape[-1])
    dx = px - cx
    dy = py - cy
    d = np.sqrt(dx * dx + dy * dy)
    _blend(img, 1.0 - smoothstep(r - AA, r + AA, d), color)


def _box(img: np.ndarray, cx: float, cy: float, h: float, color) -> None:
    px, py = _pixel_centers(img.shape[-1])
    cov = (1.0 - smoothstep(h - AA, h + AA, np.abs(px - cx))) * (1.0 - smoothstep(h - AA, h + AA, np.abs(py - cy)))
    _blend(img, cov, color)


def _background(size: int = IMAGE_SIZE) -> np.ndarray:
    img = np.empty((3, size, size), dtype=np.float64)
    for ch in range(3):
        img[ch] = BG[ch]
    return img


def _clamp(v: float, lo: float, hi: float) -> float:
    return lo if v < lo else hi if v > hi else v


def _clip_action(a) -> tuple[float, float]:
    return _clamp(float(a[0]), -1.0, 1.0), _clamp(float(a[1]), -1.0, 1.0)


# ----------------------------------------------------------------------------- base

@dataclass(frozen=True)
class EnvSpec:
    name: str
    state_dim: int
    action_dim: int = 2
    image_size: int = IMAGE_SIZE


class WorldEnv:
    """Interface shared by both environments (all methods are pure)."""

    spec: EnvSpec
    success_radius: float

    def reset(self, seed: int) -> np.ndarray:
        raise NotImplementedError

    def goal_state(self, seed: int) -> np.ndarray:
        raise NotImplementedError

    def step(self, state: np.ndarray, action) -> np.ndarray:
        raise NotImplementedError

    def render(self, state: np.ndarray) -> np.ndarray:
        raise NotImplementedError

    def success(self, state: np.ndarray, goal: np.ndarray) -> bool:
        raise NotImplementedError

    def distance(self, state: np.ndarray, goal: np.ndarray) -> float:
        raise NotImplementedError

    def policy(self, rng: np.random.Generator) -> "ExplorationPolicy":
        raise NotImplementedError


class ExplorationPolicy:
    """Stateful data-collection policy: call ``act(state)`` once per step."""

    def act(self, state: np.ndarray) -> np.ndarray:  # pragma: no cover - interface
        raise NotImplementedError


class SmoothRandomWalk(ExplorationPolicy):
    """Ornstein-Uhlenbeck-like random walk: a_t = rho a_{t-1} + sqrt(1 - rho^2) * noise."""

    def __init__(self, rng: np.random.Generator, rho: float = 0.8, scale: float = 0.9):
        self.rng, self.rho, self.scale = rng, rho, scale
        self.a = rng.uniform(-1, 1, 2)

    def act(self, state: np.ndarray) -> np.ndarray:
        noise = self.rng.normal(0.0, self.scale, 2)
        self.a = self.rho * self.a + np.sqrt(1 - self.rho ** 2) * noise
        return np.clip(self.a, -1, 1)


def _toward(pos: np.ndarray, target: np.ndarray, rng: np.random.Generator, noise: float = 0.25) -> np.ndarray:
    d = target - pos
    a = d / max(MAX_SPEED, float(np.abs(d).max()))       # full speed until within one step
    return np.clip(a + rng.normal(0.0, noise, 2), -1, 1)


# ----------------------------------------------------------------------------- Two-Room

class TwoRoom(WorldEnv):
    """Agent dot in a square arena split by a vertical wall at x = 0.5 with a door gap.

    state = [x, y] (agent center). Collision is resolved one axis at a time (x first, then y);
    an axis that would enter the wall (or leave the arena) is stopped at the contact point.
    The agent fits through the door iff |y - DOOR_Y| < DOOR_HALF - AGENT_R.
    """

    spec = EnvSpec("two-room", state_dim=2)
    WALL_X = 0.5
    WALL_HALF = 0.03
    DOOR_Y = 0.5
    DOOR_HALF = 0.12
    success_radius = 0.08

    # --- geometry
    def in_door(self, y: float) -> bool:
        return abs(y - self.DOOR_Y) < self.DOOR_HALF - AGENT_R

    def in_wall_band(self, x: float) -> bool:
        return abs(x - self.WALL_X) < self.WALL_HALF + AGENT_R

    def free(self, x: float, y: float) -> bool:
        r = AGENT_R
        if x < r or x > 1 - r or y < r or y > 1 - r:
            return False
        return not (self.in_wall_band(x) and not self.in_door(y))

    # --- sampling (mulberry32 so JS reproduces it)
    def _sample_room_point(self, rng: Mulberry32) -> tuple[float, float]:
        margin = self.WALL_HALF + AGENT_R + 0.02
        for _ in range(1000):
            x = rng.uniform(AGENT_R, 1 - AGENT_R)
            y = rng.uniform(AGENT_R, 1 - AGENT_R)
            if abs(x - self.WALL_X) >= margin:
                return x, y
        return 0.25, 0.5  # unreachable in practice

    def _reset_with(self, rng: Mulberry32) -> np.ndarray:
        return np.array(self._sample_room_point(rng), dtype=np.float64)

    def reset(self, seed: int) -> np.ndarray:
        return self._reset_with(Mulberry32(seed))

    def goal_state(self, seed: int) -> np.ndarray:
        """Goal = agent position 0.2..0.6 away from the start of the same seed (either room)."""
        rng = Mulberry32(seed)
        s = self._reset_with(rng)
        g = (0.75, 0.5)
        for _ in range(1000):
            g = self._sample_room_point(rng)
            d = float(np.hypot(g[0] - s[0], g[1] - s[1]))
            if 0.2 <= d <= 0.6:
                break
        return np.array(g, dtype=np.float64)

    # --- dynamics
    def step(self, state: np.ndarray, action) -> np.ndarray:
        ax, ay = _clip_action(action)
        x, y = float(state[0]), float(state[1])
        r = AGENT_R
        band = self.WALL_HALF + r
        # x phase: move, clamp to arena, stop at the wall face unless inside the door corridor
        nx = _clamp(x + ax * MAX_SPEED, r, 1 - r)
        if self.in_wall_band(nx) and not self.in_door(y):
            nx = self.WALL_X - band - EPS if x < self.WALL_X else self.WALL_X + band + EPS
        # y phase: move, clamp to arena; inside the wall band y must stay in the door corridor
        ny = _clamp(y + ay * MAX_SPEED, r, 1 - r)
        if self.in_wall_band(nx) and not self.in_door(ny):
            lim = self.DOOR_HALF - r - EPS
            ny = _clamp(ny, self.DOOR_Y - lim, self.DOOR_Y + lim)
        return np.array([nx, ny], dtype=np.float64)

    # --- rendering
    def render(self, state: np.ndarray) -> np.ndarray:
        img = _background()
        px, py = _pixel_centers()
        wall = (np.abs(px - self.WALL_X) < self.WALL_HALF) & (np.abs(py - self.DOOR_Y) >= self.DOOR_HALF)
        _blend(img, wall.astype(np.float64), WALL)
        _disc(img, float(state[0]), float(state[1]), AGENT_R, AGENT)
        return img.astype(np.float32)

    def distance(self, state: np.ndarray, goal: np.ndarray) -> float:
        return float(np.hypot(state[0] - goal[0], state[1] - goal[1]))

    def success(self, state: np.ndarray, goal: np.ndarray) -> bool:
        return self.distance(state, goal) < self.success_radius

    def policy(self, rng: np.random.Generator) -> ExplorationPolicy:
        return TwoRoomExplorer(self, rng)


class TwoRoomExplorer(ExplorationPolicy):
    """Smoothed random walk; in ~half of the episodes it first heads for the door and crosses
    it, so door crossings (the only interesting dynamics) are well represented in the data."""

    def __init__(self, env: TwoRoom, rng: np.random.Generator):
        self.env, self.rng = env, rng
        self.walk = SmoothRandomWalk(rng)
        self.cross = rng.random() < 0.5
        self.phase = 0  # 0: go to door mouth, 1: go through, 2: random walk

    def act(self, state: np.ndarray) -> np.ndarray:
        if not self.cross or self.phase == 2:
            return self.walk.act(state)
        e = self.env
        side = -1.0 if state[0] < e.WALL_X else 1.0
        if self.phase == 0:
            mouth = np.array([e.WALL_X + side * (e.WALL_HALF + AGENT_R + 0.03), e.DOOR_Y])
            if np.abs(state - mouth).max() < 0.04:
                self.phase = 1
                self.side = side
            return _toward(state, mouth, self.rng, 0.15)
        target = np.array([e.WALL_X - self.side * 0.2, e.DOOR_Y + self.rng.normal(0, 0.03)])
        if (state[0] - e.WALL_X) * self.side < -0.1:
            self.phase = 2
        return _toward(state, target, self.rng, 0.1)


# ----------------------------------------------------------------------------- Push-Block

class PushBlock(WorldEnv):
    """Agent dot pushes a square block in an open arena.

    state = [ax, ay, bx, by]. Kinematic push: the agent moves (clamped to the arena); if its
    collision square (half-size AGENT_R) then overlaps the block (half-size BLOCK_HALF), the
    block is displaced along the axis of *least* penetration, away from the agent, by exactly
    the penetration depth. The block is clamped to the arena; if it cannot move (against the
    border), the agent is pushed back out of it instead. Goal = block position (agent anywhere).
    """

    spec = EnvSpec("push-block", state_dim=4)
    AGENT_R = 0.06        # bigger than in two-room so the agent is as visible as the block
    BLOCK_HALF = 0.08
    success_radius = 0.06

    def _reset_with(self, rng: Mulberry32) -> np.ndarray:
        h = self.BLOCK_HALF
        bx = rng.uniform(0.2, 0.8)
        by = rng.uniform(0.2, 0.8)
        ax, ay = 0.5, 0.5
        for _ in range(1000):
            ax = rng.uniform(self.AGENT_R, 1 - self.AGENT_R)
            ay = rng.uniform(self.AGENT_R, 1 - self.AGENT_R)
            gap = max(abs(ax - bx), abs(ay - by))
            if h + self.AGENT_R + 0.03 <= gap and gap <= 0.35:
                break
        return np.array([ax, ay, bx, by], dtype=np.float64)

    def reset(self, seed: int) -> np.ndarray:
        return self._reset_with(Mulberry32(seed))

    def goal_state(self, seed: int) -> np.ndarray:
        """Goal = block displaced 0.1..0.2 along one axis; agent resting behind it (a state the
        agent reaches by pushing, which is what the goal image shows)."""
        rng = Mulberry32(seed)
        s = self._reset_with(rng)
        h = self.BLOCK_HALF
        axis = 0 if rng.next() < 0.5 else 1
        sign = -1.0 if rng.next() < 0.5 else 1.0
        dist = rng.uniform(0.1, 0.2)
        b = [s[2], s[3]]
        b[axis] = _clamp(b[axis] + sign * dist, h + 0.02, 1 - h - 0.02)
        a = [b[0], b[1]]
        a[axis] = _clamp(b[axis] - sign * (h + self.AGENT_R + 0.01), self.AGENT_R, 1 - self.AGENT_R)
        return np.array([a[0], a[1], b[0], b[1]], dtype=np.float64)

    def step(self, state: np.ndarray, action) -> np.ndarray:
        vx, vy = _clip_action(action)
        r, h = self.AGENT_R, self.BLOCK_HALF
        ax = _clamp(float(state[0]) + vx * MAX_SPEED, r, 1 - r)
        ay = _clamp(float(state[1]) + vy * MAX_SPEED, r, 1 - r)
        bx, by = float(state[2]), float(state[3])
        dx, dy = bx - ax, by - ay
        ox = (h + r) - abs(dx)
        oy = (h + r) - abs(dy)
        if ox > 0 and oy > 0:
            if ox < oy:  # push along x
                sx = 1.0 if dx >= 0 else -1.0
                bx = _clamp(bx + sx * ox, h, 1 - h)
                if abs(bx - ax) < h + r:           # block stuck at the border: agent yields
                    ax = bx - sx * (h + r + EPS)
            else:        # push along y
                sy = 1.0 if dy >= 0 else -1.0
                by = _clamp(by + sy * oy, h, 1 - h)
                if abs(by - ay) < h + r:
                    ay = by - sy * (h + r + EPS)
        return np.array([ax, ay, bx, by], dtype=np.float64)

    def render(self, state: np.ndarray) -> np.ndarray:
        img = _background()
        _box(img, float(state[2]), float(state[3]), self.BLOCK_HALF, BLOCK)
        _disc(img, float(state[0]), float(state[1]), self.AGENT_R, AGENT)
        return img.astype(np.float32)

    def distance(self, state: np.ndarray, goal: np.ndarray) -> float:
        return float(np.hypot(state[2] - goal[2], state[3] - goal[3]))

    def success(self, state: np.ndarray, goal: np.ndarray) -> bool:
        return self.distance(state, goal) < self.success_radius

    def policy(self, rng: np.random.Generator) -> ExplorationPolicy:
        return PushExplorer(self, rng)


class PushExplorer(ExplorationPolicy):
    """In ~75% of episodes: pick an axis-aligned push direction, walk to the spot behind the
    block, then push for a few steps (with noise), then pick another push; otherwise a smoothed
    random walk. This makes block motion common in the data."""

    def __init__(self, env: PushBlock, rng: np.random.Generator):
        self.env, self.rng = env, rng
        self.walk = SmoothRandomWalk(rng)
        self.pusher = rng.random() < 0.75
        self._new_push()

    def _new_push(self) -> None:
        self.dir = np.zeros(2)
        self.dir[self.rng.integers(2)] = self.rng.choice([-1.0, 1.0])
        self.phase, self.left = 0, int(self.rng.integers(3, 8))

    def act(self, state: np.ndarray) -> np.ndarray:
        if not self.pusher:
            return self.walk.act(state)
        e, h = self.env, self.env.BLOCK_HALF
        a, b = state[:2], state[2:]
        if self.phase == 0:
            behind = b - self.dir * (h + e.AGENT_R + 0.03)
            # walk around the block: if the direct path crosses it, go sideways first
            if np.abs(a - b).max() < h + e.AGENT_R + 0.02 and np.dot(a - b, self.dir) > -h:
                perp = np.array([self.dir[1], self.dir[0]])
                return _toward(a, a + perp * np.sign(np.dot(a - b, perp) + 1e-9) * 0.1 - self.dir * 0.1, self.rng, 0.1)
            if np.abs(a - behind).max() < 0.04:
                self.phase = 1
            return _toward(a, behind, self.rng, 0.15)
        self.left -= 1
        if self.left <= 0:
            self._new_push()
        return np.clip(self.dir + self.rng.normal(0, 0.2, 2), -1, 1)


ENVS: dict[str, type[WorldEnv]] = {"two-room": TwoRoom, "push-block": PushBlock}


def make_env(name: str) -> WorldEnv:
    try:
        return ENVS[name]()
    except KeyError:
        raise ValueError(f"unknown world env {name!r}; choose one of {sorted(ENVS)}") from None


def rollout(env: WorldEnv, state: np.ndarray, actions) -> np.ndarray:
    """States s_0..s_T produced by applying actions a_0..a_{T-1} from ``state``."""
    out = [np.asarray(state, dtype=np.float64)]
    for a in actions:
        out.append(env.step(out[-1], a))
    return np.stack(out)
