// Exact port of jepa_studio/world/envs.py (reset, goal, step, render, success).
// All arithmetic is float64 (JS numbers), rendered images are rounded once to float32,
// exactly like the Python version. Coordinates: unit square, y points down,
// pixel (row i, col j) is sampled at its center ((j + 0.5) / S, (i + 0.5) / S).
//
// API
//   makeEnv(name)                 -> env object ("two-room" | "push-block")
//   env.reset(seed)               -> Float64Array state
//   env.goal(seed)                -> Float64Array goal state
//   env.step(state, action)       -> NEW Float64Array state (pure; action is [ax, ay] in [-1,1])
//   env.render(state, out?)       -> Float32Array(3*32*32), CHW, values in [0,1]
//   env.success(state, goal)      -> boolean
//   env.distance(state, goal)     -> number
//   env.name, env.stateDim, env.actionDim, env.imageSize, env.successRadius
//   toRGBA(img, size?)            -> Uint8ClampedArray(size*size*4) for ImageData
//   mulberry32(seed)              -> () => float in [0, 1)

export const IMAGE_SIZE = 32;
export const MAX_SPEED = 0.08;
const AGENT_R_TWO_ROOM = 0.04;
const EPS = 1e-9;
const AA = 0.5 / IMAGE_SIZE;

const BG = [0.93, 0.93, 0.90];
const WALL = [0.25, 0.25, 0.30];
const AGENT = [0.90, 0.25, 0.20];
const BLOCK = [0.20, 0.45, 0.85];

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6D2B79F5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t = ((t + Math.imul(t ^ (t >>> 7), t | 61)) >>> 0) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

const uniform = (rng, lo, hi) => lo + (hi - lo) * rng();
const clamp = (v, lo, hi) => (v < lo ? lo : v > hi ? hi : v);
const clipAction = (a) => [clamp(+a[0], -1, 1), clamp(+a[1], -1, 1)];

function smoothstep(e0, e1, x) {
  let t = (x - e0) / (e1 - e0);
  t = t < 0 ? 0 : t > 1 ? 1 : t;
  return t * t * (3 - 2 * t);
}

// ------------------------------------------------------------------ drawing (float64 buffer)
function background(S) {
  const img = new Float64Array(3 * S * S);
  for (let c = 0; c < 3; c++) img.fill(BG[c], c * S * S, (c + 1) * S * S);
  return img;
}

function blendAt(img, S, idx, cov, color) {
  if (cov === 0) return;
  for (let c = 0; c < 3; c++) {
    const k = c * S * S + idx;
    img[k] = img[k] * (1 - cov) + color[c] * cov;
  }
}

function disc(img, S, cx, cy, r, color) {
  for (let i = 0; i < S; i++) {
    const py = (i + 0.5) / S;
    for (let j = 0; j < S; j++) {
      const px = (j + 0.5) / S;
      const dx = px - cx, dy = py - cy;
      const d = Math.sqrt(dx * dx + dy * dy);
      blendAt(img, S, i * S + j, 1 - smoothstep(r - AA, r + AA, d), color);
    }
  }
}

function box(img, S, cx, cy, h, color) {
  for (let i = 0; i < S; i++) {
    const py = (i + 0.5) / S;
    const cy1 = 1 - smoothstep(h - AA, h + AA, Math.abs(py - cy));
    for (let j = 0; j < S; j++) {
      const px = (j + 0.5) / S;
      const cov = (1 - smoothstep(h - AA, h + AA, Math.abs(px - cx))) * cy1;
      blendAt(img, S, i * S + j, cov, color);
    }
  }
}

function toF32(img, out) {
  const o = out || new Float32Array(img.length);
  for (let k = 0; k < img.length; k++) o[k] = img[k];   // Float32Array store = round-to-nearest
  return o;
}

// ------------------------------------------------------------------ Two-Room
class TwoRoom {
  constructor() {
    this.name = "two-room"; this.stateDim = 2; this.actionDim = 2; this.imageSize = IMAGE_SIZE;
    this.WALL_X = 0.5; this.WALL_HALF = 0.03; this.DOOR_Y = 0.5; this.DOOR_HALF = 0.12;
    this.AGENT_R = AGENT_R_TWO_ROOM; this.successRadius = 0.08;
  }
  inDoor(y) { return Math.abs(y - this.DOOR_Y) < this.DOOR_HALF - this.AGENT_R; }
  inWallBand(x) { return Math.abs(x - this.WALL_X) < this.WALL_HALF + this.AGENT_R; }
  _sampleRoomPoint(rng) {
    const r = this.AGENT_R, margin = this.WALL_HALF + r + 0.02;
    for (let n = 0; n < 1000; n++) {
      const x = uniform(rng, r, 1 - r), y = uniform(rng, r, 1 - r);
      if (Math.abs(x - this.WALL_X) >= margin) return [x, y];
    }
    return [0.25, 0.5];
  }
  _resetWith(rng) { return Float64Array.from(this._sampleRoomPoint(rng)); }
  reset(seed) { return this._resetWith(mulberry32(seed)); }
  goal(seed) {
    const rng = mulberry32(seed);
    const s = this._resetWith(rng);
    let g = [0.75, 0.5];
    for (let n = 0; n < 1000; n++) {
      g = this._sampleRoomPoint(rng);
      const d = Math.hypot(g[0] - s[0], g[1] - s[1]);
      if (d >= 0.2 && d <= 0.6) break;
    }
    return Float64Array.from(g);
  }
  step(state, action) {
    const [ax, ay] = clipAction(action);
    const x = state[0], y = state[1], r = this.AGENT_R, band = this.WALL_HALF + r;
    let nx = clamp(x + ax * MAX_SPEED, r, 1 - r);
    if (this.inWallBand(nx) && !this.inDoor(y)) nx = x < this.WALL_X ? this.WALL_X - band - EPS : this.WALL_X + band + EPS;
    let ny = clamp(y + ay * MAX_SPEED, r, 1 - r);
    if (this.inWallBand(nx) && !this.inDoor(ny)) {
      const lim = this.DOOR_HALF - r - EPS;
      ny = clamp(ny, this.DOOR_Y - lim, this.DOOR_Y + lim);
    }
    return Float64Array.of(nx, ny);
  }
  render(state, out) {
    const S = IMAGE_SIZE, img = background(S);
    for (let i = 0; i < S; i++) {
      const py = (i + 0.5) / S;
      for (let j = 0; j < S; j++) {
        const px = (j + 0.5) / S;
        if (Math.abs(px - this.WALL_X) < this.WALL_HALF && Math.abs(py - this.DOOR_Y) >= this.DOOR_HALF) blendAt(img, S, i * S + j, 1, WALL);
      }
    }
    disc(img, S, state[0], state[1], this.AGENT_R, AGENT);
    return toF32(img, out);
  }
  distance(s, g) { return Math.hypot(s[0] - g[0], s[1] - g[1]); }
  success(s, g) { return this.distance(s, g) < this.successRadius; }
}

// ------------------------------------------------------------------ Push-Block
class PushBlock {
  constructor() {
    this.name = "push-block"; this.stateDim = 4; this.actionDim = 2; this.imageSize = IMAGE_SIZE;
    this.AGENT_R = 0.06; this.BLOCK_HALF = 0.08; this.successRadius = 0.06;
  }
  _resetWith(rng) {
    const h = this.BLOCK_HALF, r = this.AGENT_R;
    const bx = uniform(rng, 0.2, 0.8), by = uniform(rng, 0.2, 0.8);
    let ax = 0.5, ay = 0.5;
    for (let n = 0; n < 1000; n++) {
      ax = uniform(rng, r, 1 - r); ay = uniform(rng, r, 1 - r);
      const gap = Math.max(Math.abs(ax - bx), Math.abs(ay - by));
      if (h + r + 0.03 <= gap && gap <= 0.35) break;
    }
    return Float64Array.of(ax, ay, bx, by);
  }
  reset(seed) { return this._resetWith(mulberry32(seed)); }
  goal(seed) {
    const rng = mulberry32(seed);
    const s = this._resetWith(rng);
    const h = this.BLOCK_HALF, r = this.AGENT_R;
    const axis = rng() < 0.5 ? 0 : 1;
    const sign = rng() < 0.5 ? -1 : 1;
    const dist = uniform(rng, 0.1, 0.2);
    const b = [s[2], s[3]];
    b[axis] = clamp(b[axis] + sign * dist, h + 0.02, 1 - h - 0.02);
    const a = [b[0], b[1]];
    a[axis] = clamp(b[axis] - sign * (h + r + 0.01), r, 1 - r);
    return Float64Array.of(a[0], a[1], b[0], b[1]);
  }
  step(state, action) {
    const [vx, vy] = clipAction(action);
    const r = this.AGENT_R, h = this.BLOCK_HALF;
    let ax = clamp(state[0] + vx * MAX_SPEED, r, 1 - r);
    let ay = clamp(state[1] + vy * MAX_SPEED, r, 1 - r);
    let bx = state[2], by = state[3];
    const dx = bx - ax, dy = by - ay;
    const ox = (h + r) - Math.abs(dx), oy = (h + r) - Math.abs(dy);
    if (ox > 0 && oy > 0) {
      if (ox < oy) {
        const sx = dx >= 0 ? 1 : -1;
        bx = clamp(bx + sx * ox, h, 1 - h);
        if (Math.abs(bx - ax) < h + r) ax = bx - sx * (h + r + EPS);
      } else {
        const sy = dy >= 0 ? 1 : -1;
        by = clamp(by + sy * oy, h, 1 - h);
        if (Math.abs(by - ay) < h + r) ay = by - sy * (h + r + EPS);
      }
    }
    return Float64Array.of(ax, ay, bx, by);
  }
  render(state, out) {
    const S = IMAGE_SIZE, img = background(S);
    box(img, S, state[2], state[3], this.BLOCK_HALF, BLOCK);
    disc(img, S, state[0], state[1], this.AGENT_R, AGENT);
    return toF32(img, out);
  }
  distance(s, g) { return Math.hypot(s[2] - g[2], s[3] - g[3]); }
  success(s, g) { return this.distance(s, g) < this.successRadius; }
}

const ENVS = { "two-room": TwoRoom, "push-block": PushBlock };

export function makeEnv(name) {
  const C = ENVS[name];
  if (!C) throw new Error(`unknown world env ${name}`);
  return new C();
}

export const ENV_NAMES = Object.keys(ENVS);

/** CHW float image in [0,1] -> RGBA bytes (for ctx.putImageData). */
export function toRGBA(img, size = IMAGE_SIZE) {
  const n = size * size, out = new Uint8ClampedArray(n * 4);
  for (let k = 0; k < n; k++) {
    out[4 * k] = Math.round(img[k] * 255);
    out[4 * k + 1] = Math.round(img[n + k] * 255);
    out[4 * k + 2] = Math.round(img[2 * n + k] * 255);
    out[4 * k + 3] = 255;
  }
  return out;
}
