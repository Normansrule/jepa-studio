// Live physics simulations on a 2-D canvas: gravitational N-body, chaotic double pendulum, 2-D waves
// (plus a Kepler orbit and the Lorenz attractor). The numerical methods mirror jev/science/sims.py:
//   N-body: velocity Verlet (kick-drift-kick leapfrog), Plummer softening for clusters
//   double pendulum: RK4; a twin starts 1e-5 rad away to show chaotic divergence
//   waves: leapfrog finite differences, 5-point Laplacian, CFL number checked (<= 1), absorbing edges
// The physics functions are exported and DOM-free so tests/js/science_sims.mjs can check them in Node.
//
// API: mountSim(canvas, kind, options) -> {setParams, getParams, start, pause, reset, step, metrics,
// snapshot, destroy, running}. kind: 'nbody' | 'double_pendulum' (alias 'pendulum') | 'wave' | 'orbit' |
// 'lorenz'. With reducedMotion (default: the prefers-reduced-motion query) nothing animates on its own:
// the view jumps to a later state with full trails; start() still plays on request.

// ------------------------------------------------------------------ N-body
export function createNBody(n = 3, seed = 1) {
  if (n <= 3) {
    const x = [[0.97000436, -0.24308753], [-0.97000436, 0.24308753], [0, 0]];
    const v3 = [-0.93240737, -0.86473146];
    return { x, v: [[-v3[0] / 2, -v3[1] / 2], [-v3[0] / 2, -v3[1] / 2], [...v3]], m: [1, 1, 1], G: 1, eps: 0, view: 1.4, t: 0, preset: 'figure8' };
  }
  const rnd = mulberry32(seed);
  const m = new Array(n).fill(1 / n);
  const x = [], v = [];
  for (let i = 0; i < n; i++) {
    const u = 0.05 + 0.95 * rnd();
    const r = Math.min(2, 0.3 / Math.sqrt(u ** (-2 / 3) - 1));
    const a = 2 * Math.PI * rnd();
    x.push([r * Math.cos(a), r * Math.sin(a)]);
    v.push([gauss(rnd), gauss(rnd)]);
  }
  center(x, m); center(v, m);
  const s = { x, v, m, G: 1, eps: 0.05, view: 1.6, t: 0, preset: 'cluster' };
  // virialise: 2K = -W
  const K = kinetic(s), W = nbodyEnergy({ ...s, v: v.map(() => [0, 0]) });
  const f = Math.sqrt(-W / (2 * K));
  for (const vi of v) { vi[0] *= f; vi[1] *= f; }
  return s;
}

function center(a, m) {
  let cx = 0, cy = 0, M = 0;
  for (let i = 0; i < a.length; i++) { cx += m[i] * a[i][0]; cy += m[i] * a[i][1]; M += m[i]; }
  for (const p of a) { p[0] -= cx / M; p[1] -= cy / M; }
}

function kinetic(s) {
  let k = 0;
  for (let i = 0; i < s.m.length; i++) k += 0.5 * s.m[i] * (s.v[i][0] ** 2 + s.v[i][1] ** 2);
  return k;
}

export function nbodyAccel(s) {
  const n = s.m.length, a = s.x.map(() => [0, 0]), e2 = s.eps * s.eps;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = s.x[j][0] - s.x[i][0], dy = s.x[j][1] - s.x[i][1];
      const r2 = dx * dx + dy * dy + e2, inv = s.G / (r2 * Math.sqrt(r2));
      a[i][0] += s.m[j] * dx * inv; a[i][1] += s.m[j] * dy * inv;
      a[j][0] -= s.m[i] * dx * inv; a[j][1] -= s.m[i] * dy * inv;
    }
  }
  return a;
}

export function nbodyEnergy(s) {
  let pe = 0;
  const n = s.m.length, e2 = s.eps * s.eps;
  for (let i = 0; i < n; i++) {
    for (let j = i + 1; j < n; j++) {
      const dx = s.x[j][0] - s.x[i][0], dy = s.x[j][1] - s.x[i][1];
      pe -= (s.G * s.m[i] * s.m[j]) / Math.sqrt(dx * dx + dy * dy + e2);
    }
  }
  return kinetic(s) + pe;
}

/** One velocity-Verlet step (kick-drift-kick). Keeps the acceleration cached in s.a. */
export function nbodyStep(s, dt) {
  if (!s.a) s.a = nbodyAccel(s);
  for (let i = 0; i < s.m.length; i++) {
    s.v[i][0] += 0.5 * dt * s.a[i][0]; s.v[i][1] += 0.5 * dt * s.a[i][1];
    s.x[i][0] += dt * s.v[i][0]; s.x[i][1] += dt * s.v[i][1];
  }
  s.a = nbodyAccel(s);
  for (let i = 0; i < s.m.length; i++) { s.v[i][0] += 0.5 * dt * s.a[i][0]; s.v[i][1] += 0.5 * dt * s.a[i][1]; }
  s.t += dt;
}

// ------------------------------------------------------------------ double pendulum
export function pendulumRHS(y, m1 = 1, m2 = 1, l1 = 1, l2 = 1, g = 9.81) {
  const [t1, t2, w1, w2] = y;
  const d = t2 - t1, c = Math.cos(d), s = Math.sin(d);
  const den1 = (m1 + m2) * l1 - m2 * l1 * c * c;
  const a1 = (m2 * l1 * w1 * w1 * s * c + m2 * g * Math.sin(t2) * c + m2 * l2 * w2 * w2 * s - (m1 + m2) * g * Math.sin(t1)) / den1;
  const den2 = (l2 / l1) * den1;
  const a2 = (-m2 * l2 * w2 * w2 * s * c + (m1 + m2) * (g * Math.sin(t1) * c - l1 * w1 * w1 * s - g * Math.sin(t2))) / den2;
  return [w1, w2, a1, a2];
}

export function pendulumEnergy(y, m1 = 1, m2 = 1, l1 = 1, l2 = 1, g = 9.81) {
  const [t1, t2, w1, w2] = y;
  const T = 0.5 * (m1 + m2) * l1 * l1 * w1 * w1 + 0.5 * m2 * l2 * l2 * w2 * w2 + m2 * l1 * l2 * w1 * w2 * Math.cos(t1 - t2);
  const U = -(m1 + m2) * g * l1 * Math.cos(t1) - m2 * g * l2 * Math.cos(t2);
  return T + U;
}

export function rk4(f, y, dt) {
  const add = (a, b, k) => a.map((v, i) => v + k * b[i]);
  const k1 = f(y), k2 = f(add(y, k1, dt / 2)), k3 = f(add(y, k2, dt / 2)), k4 = f(add(y, k3, dt));
  return y.map((v, i) => v + (dt / 6) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]));
}

// ------------------------------------------------------------------ waves
export function courant(c, dt, dx, dy = dx) { return c * dt * Math.sqrt(1 / (dx * dx) + 1 / (dy * dy)); }

/**
 * 2-D wave equation u_tt = c^2 lap(u) on an nx x ny grid (cell size dx), leapfrog in time.
 * options: c, cfl (target Courant number, default 0.5; > 1 throws), damping (per step, 0..0.2),
 * sponge (absorbing border cells), sources [{i, j, freq}] (sinusoidal point sources).
 */
export function createWave(nx, ny, opts = {}) {
  const c = opts.c ?? 1, dx = opts.dx ?? 1 / (Math.max(nx, ny) - 1);
  const cfl = opts.cfl ?? 0.5;
  const dt = opts.dt ?? (cfl * dx) / (c * Math.SQRT2);
  const C = courant(c, dt, dx);
  if (C > 1 + 1e-12) throw new RangeError(`unstable: CFL number ${C.toFixed(3)} > 1 (use dt <= ${(dx / (c * Math.SQRT2)).toExponential(3)})`);
  const n = nx * ny;
  const sponge = new Float32Array(n);
  const k = opts.sponge ?? 0;
  if (k > 0) {
    for (let j = 0; j < ny; j++) {
      for (let i = 0; i < nx; i++) {
        const d = Math.min(i, j, nx - 1 - i, ny - 1 - j);
        if (d < k) sponge[j * nx + i] = 0.25 * (1 - d / k) ** 2;
      }
    }
  }
  return { nx, ny, dx, dt, c, C, t: 0, u: new Float64Array(n), prev: new Float64Array(n), next: new Float64Array(n),
    sponge, damping: opts.damping ?? 0, sources: opts.sources ?? [] };
}

export function waveStep(w) {
  const { nx, ny, u, prev, next, sponge } = w;
  const r2 = (w.c * w.dt / w.dx) ** 2;
  for (let j = 1; j < ny - 1; j++) {
    for (let i = 1; i < nx - 1; i++) {
      const p = j * nx + i;
      const lap = u[p - 1] + u[p + 1] + u[p - nx] + u[p + nx] - 4 * u[p];
      const g = w.damping + sponge[p];
      next[p] = (2 * u[p] - (1 - g) * prev[p] + r2 * lap) / (1 + g);
    }
  }
  w.t += w.dt;
  for (const s of w.sources) next[s.j * nx + s.i] = Math.sin(2 * Math.PI * s.freq * w.t) * Math.min(1, w.t * s.freq);
  w.prev = u; w.u = next; w.next = prev;
}

/** Discrete energy the undamped leapfrog conserves exactly with fixed (u = 0) edges. */
export function waveEnergy(w) {
  const { nx, ny, u, prev, dt, c, dx } = w;
  let kin = 0, pot = 0;
  for (let p = 0; p < nx * ny; p++) { const v = (u[p] - prev[p]) / dt; kin += v * v; }
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx - 1; i++) { const p = j * nx + i; pot += (u[p + 1] - u[p]) * (prev[p + 1] - prev[p]); }
  }
  for (let j = 0; j < ny - 1; j++) {
    for (let i = 0; i < nx; i++) { const p = j * nx + i; pot += (u[p + nx] - u[p]) * (prev[p + nx] - prev[p]); }
  }
  return 0.5 * dx * dx * (kin + (c * c * pot) / (dx * dx));
}

/** A Gaussian bump with zero velocity (second-order start). */
export function wavePulse(w, ci, cj, width) {
  const { nx, ny } = w;
  for (let j = 0; j < ny; j++) {
    for (let i = 0; i < nx; i++) {
      const v = (i > 0 && j > 0 && i < nx - 1 && j < ny - 1) ? Math.exp(-((i - ci) ** 2 + (j - cj) ** 2) / (2 * width * width)) : 0;
      w.u[j * nx + i] += v; w.prev[j * nx + i] += v;
    }
  }
}

// ------------------------------------------------------------------ Kepler and Lorenz
export function keplerPosition(t, a = 1, e = 0.5) {
  const M = t / Math.sqrt(a ** 3);
  let E = M + e * Math.sin(M);
  for (let k = 0; k < 30; k++) E -= (E - e * Math.sin(E) - M) / (1 - e * Math.cos(E));
  return [a * (Math.cos(E) - e), a * Math.sqrt(1 - e * e) * Math.sin(E)];
}

export function lorenzRHS([x, y, z], sigma = 10, rho = 28, beta = 8 / 3) {
  return [sigma * (y - x), x * (rho - z) - y, x * y - beta * z];
}

// ------------------------------------------------------------------ helpers
function mulberry32(a) {
  return () => {
    a |= 0; a = (a + 0x6d2b79f5) | 0;
    let t = Math.imul(a ^ (a >>> 15), 1 | a);
    t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}
function gauss(rnd) { return Math.sqrt(-2 * Math.log(1 - rnd())) * Math.cos(2 * Math.PI * rnd()); }

const FALLBACK = { '--surface': '#1b2437', '--surface-2': '#232e45', '--ink': '#e8ecf4', '--ink-2': '#b4bccb', '--ink-3': '#97a1b4',
  '--line': '#2e3a54', '--cat-1': '#3987e5', '--cat-2': '#d95926', '--cat-3': '#199e70', '--cat-4': '#9085e9', '--cat-5': '#d55181',
  '--status-warn': '#fab219' };
function palette() {
  const out = {};
  const css = typeof getComputedStyle === 'function' ? getComputedStyle(document.documentElement) : null;
  for (const [k, v] of Object.entries(FALLBACK)) out[k] = (css && css.getPropertyValue(k).trim()) || v;
  return out;
}
function hexToRgb(h) {
  const m = /^#?([0-9a-f]{6})$/i.exec(h.trim());
  if (!m) return [128, 128, 128];
  const n = parseInt(m[1], 16);
  return [(n >> 16) & 255, (n >> 8) & 255, n & 255];
}

// ------------------------------------------------------------------ mounting
const KINDS = ['nbody', 'double_pendulum', 'wave', 'orbit', 'lorenz'];
const ALIAS = { pendulum: 'double_pendulum', 'n-body': 'nbody' };
const SIM_DEFAULTS = {
  nbody: { bodies: 3, speed: 1 },
  double_pendulum: { theta1: 120, theta2: -20, massRatio: 1, speed: 1 },
  wave: { speed: 1, damping: 0.002, freq: 1.5 },
  orbit: { e: 0.6, gr: false, speed: 1 },
  lorenz: { speed: 1 },
};

/**
 * Mount a live simulation. options (per kind, all optional):
 *   nbody: bodies (3 = figure-eight orbit; more = star cluster), speed
 *   double_pendulum: theta1, theta2 (degrees), massRatio (m2/m1), speed
 *   wave: speed (wave speed c), damping (0..0.05), freq (source frequency); click or tap adds a drop
 *   orbit: e (eccentricity), speed;  lorenz: speed
 *   reducedMotion, autoplay (default: !reducedMotion), onMetrics(metrics) (about once a second)
 */
export function mountSim(canvas, kind, options = {}) {
  kind = ALIAS[kind] || kind;
  if (!KINDS.includes(kind)) throw new Error(`unknown simulation "${kind}"`);
  const ctx = canvas.getContext('2d');
  if (!ctx) throw new Error('2-D canvas is not available');
  const mq = typeof matchMedia === 'function' ? matchMedia('(prefers-reduced-motion: reduce)') : null;
  let params = { ...SIM_DEFAULTS[kind], reducedMotion: mq ? mq.matches : false, ...options };
  let state = null, raf = 0, running = false, destroyed = false, last = 0, frames = 0;
  let colors = palette(), colorAge = 0;
  const trails = [];
  let E0 = 0, lastMetrics = {};
  const off = document.createElement('canvas');

  function init() {
    trails.length = 0;
    if (kind === 'nbody') {
      state = createNBody(Math.max(2, Math.min(300, Math.round(params.bodies))), 7);
      E0 = nbodyEnergy(state);
    } else if (kind === 'double_pendulum') {
      const t1 = (params.theta1 * Math.PI) / 180, t2 = (params.theta2 * Math.PI) / 180;
      state = { a: [t1, t2, 0, 0], b: [t1 + 1e-5, t2, 0, 0], t: 0, m2: Math.max(0.05, params.massRatio) };
      E0 = pendulumEnergy(state.a, 1, state.m2);
    } else if (kind === 'wave') {
      const aspect = (canvas.width || 16) / (canvas.height || 9);
      const ny = 100, nx = Math.max(40, Math.round(ny * aspect));
      state = createWave(nx, ny, { c: 1, cfl: 0.5, damping: params.damping, sponge: 10,
        sources: [{ i: Math.round(nx * 0.3), j: Math.round(ny * 0.42), freq: 0 }, { i: Math.round(nx * 0.3), j: Math.round(ny * 0.58), freq: 0 }] });
      setWaveFreq();
      off.width = nx; off.height = ny;
      state.img = off.getContext('2d').createImageData(nx, ny);
    } else if (kind === 'orbit') {
      const e = Math.min(0.95, Math.max(0, params.e));
      state = { x: [1 - e, 0], v: [0, Math.sqrt((1 + e) / (1 - e))], t: 0, e };
      E0 = 0.5 * (state.v[0] ** 2 + state.v[1] ** 2) - 1 / Math.hypot(...state.x);
    } else {
      state = { a: [1, 1, 20], b: [1 + 1e-6, 1, 20], t: 0 };
    }
  }
  function setWaveFreq() {
    // frequency in cycles per unit time; wave speed c scales the time step via the CFL rule
    for (const s of state.sources) s.freq = params.freq * 4;
    state.damping = Math.max(0, Math.min(0.05, params.damping));
  }

  function advance(seconds) {
    const sp = Math.max(0, params.speed ?? 1);
    if (kind === 'nbody') {
      const dt = state.preset === 'figure8' ? 2e-3 : 2e-3;
      const n = Math.min(400, Math.ceil((seconds * sp * 0.8) / dt));
      for (let k = 0; k < n; k++) nbodyStep(state, dt);
      pushTrail(state.x.map((p) => [...p]));
    } else if (kind === 'double_pendulum') {
      const dt = 1 / 1200, n = Math.min(600, Math.ceil((seconds * sp) / dt));
      const f = (y) => pendulumRHS(y, 1, state.m2);
      for (let k = 0; k < n; k++) { state.a = rk4(f, state.a, dt); state.b = rk4(f, state.b, dt); }
      state.t += n * dt;
      pushTrail([tip(state.a), tip(state.b)]);
    } else if (kind === 'wave') {
      const n = Math.min(12, Math.max(1, Math.round(seconds * 60 * sp * Math.max(0.2, params.speed) * 1.5)));
      for (let k = 0; k < n; k++) waveStep(state);
    } else if (kind === 'orbit') {
      const dt = 2e-3, n = Math.min(2000, Math.ceil((seconds * sp * 2) / dt));
      for (let k = 0; k < n; k++) {
        const acc = (x) => { const r = Math.hypot(...x); return [-x[0] / r ** 3, -x[1] / r ** 3]; };
        let a = acc(state.x);
        state.v = [state.v[0] + 0.5 * dt * a[0], state.v[1] + 0.5 * dt * a[1]];
        state.x = [state.x[0] + dt * state.v[0], state.x[1] + dt * state.v[1]];
        a = acc(state.x);
        state.v = [state.v[0] + 0.5 * dt * a[0], state.v[1] + 0.5 * dt * a[1]];
        state.t += dt;
      }
      pushTrail([[...state.x]]);
    } else {
      const dt = 0.004, n = Math.min(500, Math.ceil((seconds * sp * 0.6) / dt));
      for (let k = 0; k < n; k++) { state.a = rk4(lorenzRHS, state.a, dt); state.b = rk4(lorenzRHS, state.b, dt); }
      state.t += n * dt;
      pushTrail([[...state.a], [...state.b]]);
    }
  }
  const tip = (y) => [Math.sin(y[0]) + Math.sin(y[1]), -Math.cos(y[0]) - Math.cos(y[1])];
  function pushTrail(p) { trails.push(p); const max = kind === 'lorenz' ? 900 : kind === 'nbody' && state.m.length > 10 ? 40 : 300; while (trails.length > max) trails.shift(); }

  function metrics() {
    const m = { kind, t: state.t };
    if (kind === 'nbody') { m.energy_rel_error = Math.abs((nbodyEnergy(state) - E0) / E0); m.bodies = state.m.length; m.integrator = 'velocity Verlet'; }
    if (kind === 'double_pendulum') {
      m.energy_rel_error = Math.abs((pendulumEnergy(state.a, 1, state.m2) - E0) / E0);
      m.separation = Math.hypot(state.a[0] - state.b[0], state.a[1] - state.b[1], state.a[2] - state.b[2], state.a[3] - state.b[3]);
      m.integrator = 'RK4';
    }
    if (kind === 'wave') { m.cfl = state.C; m.scheme = 'leapfrog'; }
    if (kind === 'orbit') {
      m.energy_rel_error = Math.abs((0.5 * (state.v[0] ** 2 + state.v[1] ** 2) - 1 / Math.hypot(...state.x) - E0) / E0);
      const k = keplerPosition(state.t, 1, state.e);
      m.position_error = Math.hypot(state.x[0] - k[0], state.x[1] - k[1]);
    }
    if (kind === 'lorenz') m.separation = Math.hypot(state.a[0] - state.b[0], state.a[1] - state.b[1], state.a[2] - state.b[2]);
    lastMetrics = m;
    return m;
  }

  // ------------------------------------------------------------------ drawing
  function draw() {
    if (++colorAge > 60) { colors = palette(); colorAge = 0; }
    const W = canvas.width, H = canvas.height;
    ctx.fillStyle = colors['--surface'];
    ctx.fillRect(0, 0, W, H);
    const cat = ['--cat-1', '--cat-2', '--cat-3', '--cat-4', '--cat-5'].map((k) => colors[k]);
    const S = Math.min(W, H) / 2;
    const toPx = (p, view, cx = 0, cy = 0) => [W / 2 + ((p[0] - cx) / view) * S, H / 2 - ((p[1] - cy) / view) * S];
    const lw = Math.max(1, Math.round(Math.min(W, H) / 300));
    ctx.lineJoin = 'round'; ctx.lineCap = 'round';
    if (kind === 'wave') {
      const { u, nx, ny, img } = state;
      const d = img.data;
      const lo = hexToRgb(colors['--cat-1']), hi = hexToRgb(colors['--cat-2']), base = hexToRgb(colors['--surface']);
      for (let p = 0; p < nx * ny; p++) {
        const v = Math.max(-1, Math.min(1, u[p] * 1.2));
        const c = v < 0 ? lo : hi, f = Math.abs(v);
        const q = ((ny - 1 - Math.floor(p / nx)) * nx + (p % nx)) * 4;
        d[q] = base[0] + (c[0] - base[0]) * f; d[q + 1] = base[1] + (c[1] - base[1]) * f; d[q + 2] = base[2] + (c[2] - base[2]) * f; d[q + 3] = 255;
      }
      off.getContext('2d').putImageData(img, 0, 0);
      ctx.imageSmoothingEnabled = true;
      ctx.drawImage(off, 0, 0, W, H);
      ctx.fillStyle = colors['--ink'];
      for (const s of state.sources) { const [x, y] = [(s.i / (nx - 1)) * W, H - (s.j / (ny - 1)) * H]; ctx.beginPath(); ctx.arc(x, y, 3 * lw, 0, 2 * Math.PI); ctx.fill(); }
    } else if (kind === 'nbody') {
      const view = state.view;
      const nb = state.m.length;
      for (let b = 0; b < nb; b++) {
        ctx.strokeStyle = cat[b % 5]; ctx.lineWidth = nb > 10 ? 1 : 2 * lw; ctx.globalAlpha = 0.55;
        ctx.beginPath();
        trails.forEach((tr, k) => { const [x, y] = toPx(tr[b], view); if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
        ctx.stroke();
        ctx.globalAlpha = 1;
        const [x, y] = toPx(state.x[b], view);
        ctx.fillStyle = cat[b % 5];
        ctx.beginPath(); ctx.arc(x, y, (nb > 10 ? 2.5 : 6) * lw, 0, 2 * Math.PI); ctx.fill();
      }
    } else if (kind === 'double_pendulum') {
      const view = 2.3;
      [state.a, state.b].forEach((y, k) => {
        ctx.strokeStyle = cat[k]; ctx.globalAlpha = 0.5; ctx.lineWidth = 2 * lw;
        ctx.beginPath();
        trails.forEach((tr, i) => { const [x, yy] = toPx(tr[k], view); if (i) ctx.lineTo(x, yy); else ctx.moveTo(x, yy); });
        ctx.stroke();
        ctx.globalAlpha = 1;
        const p1 = [Math.sin(y[0]), -Math.cos(y[0])], p2 = tip(y);
        const pts = [toPx([0, 0], view), toPx(p1, view), toPx(p2, view)];
        ctx.lineWidth = 3 * lw;
        ctx.beginPath(); ctx.moveTo(...pts[0]); ctx.lineTo(...pts[1]); ctx.lineTo(...pts[2]); ctx.stroke();
        ctx.fillStyle = cat[k];
        for (const q of pts.slice(1)) { ctx.beginPath(); ctx.arc(q[0], q[1], 7 * lw, 0, 2 * Math.PI); ctx.fill(); }
      });
    } else if (kind === 'orbit') {
      const e = state.e, view = (1 + e) * 1.15, cx = -e;
      ctx.strokeStyle = colors['--ink-3']; ctx.setLineDash([4 * lw, 4 * lw]); ctx.lineWidth = lw;
      ctx.beginPath();
      for (let k = 0; k <= 200; k++) { const E = (2 * Math.PI * k) / 200; const [x, y] = toPx([Math.cos(E) - e, Math.sqrt(1 - e * e) * Math.sin(E)], view, cx); if (k) ctx.lineTo(x, y); else ctx.moveTo(x, y); }
      ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = colors['--status-warn'];
      const [sx, sy] = toPx([0, 0], view, cx); ctx.beginPath(); ctx.arc(sx, sy, 7 * lw, 0, 2 * Math.PI); ctx.fill();
      const [bx, by] = toPx(state.x, view, cx);
      ctx.fillStyle = cat[0]; ctx.beginPath(); ctx.arc(bx, by, 5 * lw, 0, 2 * Math.PI); ctx.fill();
    } else {
      const view = 30;
      [0, 1].forEach((k) => {
        ctx.strokeStyle = cat[k]; ctx.lineWidth = lw; ctx.globalAlpha = 0.8;
        ctx.beginPath();
        trails.forEach((tr, i) => { const [x, y] = toPx([tr[k][0], tr[k][2] - 25], view); if (i) ctx.lineTo(x, y); else ctx.moveTo(x, y); });
        ctx.stroke();
      });
      ctx.globalAlpha = 1;
    }
    // read-out
    const m = metrics();
    const txt = [`t = ${m.t.toFixed(2)}`];
    if (m.energy_rel_error !== undefined) txt.push(`energy error ${m.energy_rel_error.toExponential(1)}`);
    if (m.separation !== undefined) txt.push(`twin separation ${m.separation.toExponential(1)}`);
    if (m.cfl !== undefined) txt.push(`CFL ${m.cfl.toFixed(2)}`);
    ctx.fillStyle = colors['--ink-3'];
    ctx.font = `${Math.max(11, Math.round(H / 32))}px system-ui, sans-serif`;
    ctx.textBaseline = 'bottom';
    ctx.fillText(txt.join('   '), 10, H - 8);
    if (typeof params.onMetrics === 'function' && frames % 60 === 0) params.onMetrics(m);
  }

  function frame(t) {
    raf = 0;
    if (destroyed || !running) return;
    const dt = last ? Math.min(0.05, (t - last) / 1000) : 1 / 60;
    last = t;
    advance(dt);
    draw();
    frames++;
    raf = requestAnimationFrame(frame);
  }
  function start() { if (destroyed || running) return; running = true; last = 0; raf = requestAnimationFrame(frame); }
  function pause() { running = false; if (raf) cancelAnimationFrame(raf); raf = 0; }
  function step(n = 1) { for (let k = 0; k < n; k++) advance(1 / 60); draw(); }
  function still() {
    // reduced motion: show a later state with its trails instead of an animation
    step(kind === 'wave' ? 90 : kind === 'lorenz' ? 400 : 240);
  }
  function reset() {
    const was = running;
    pause();
    init();
    if (was || (!params.reducedMotion && params.autoplay !== false)) start(); else still();
  }
  function setParams(p = {}) {
    const old = params;
    params = { ...params, ...p };
    const restart = (kind === 'nbody' && Math.round(old.bodies) !== Math.round(params.bodies))
      || (kind === 'double_pendulum' && (old.theta1 !== params.theta1 || old.theta2 !== params.theta2 || old.massRatio !== params.massRatio))
      || (kind === 'orbit' && old.e !== params.e);
    if (restart) reset();
    else { if (kind === 'wave') setWaveFreq(); if (!running) draw(); }
  }
  const onPointer = (e) => {
    if (kind !== 'wave') return;
    const r = canvas.getBoundingClientRect();
    const i = Math.round(((e.clientX - r.left) / r.width) * (state.nx - 1));
    const j = Math.round((1 - (e.clientY - r.top) / r.height) * (state.ny - 1));
    wavePulse(state, i, j, 2.5);
    if (!running) draw();
  };
  canvas.addEventListener('pointerdown', onPointer);
  function destroy() { destroyed = true; pause(); canvas.removeEventListener('pointerdown', onPointer); }

  init();
  if (!params.reducedMotion && params.autoplay !== false) start(); else still();
  return {
    setParams, getParams: () => ({ ...params }), start, pause, reset, step, destroy,
    metrics: () => lastMetrics,
    snapshot: () => new Promise((resolve, reject) => canvas.toBlob((b) => (b ? resolve(b) : reject(new Error('no image'))), 'image/png')),
    get running() { return running; },
    kind,
  };
}
