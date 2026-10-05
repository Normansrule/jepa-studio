// Browser runtime for exported JEPA world models (format "jepa-studio-world/v1", written by
// jepa_studio/world/export.py) + latent-space CEM planner. ES module, no dependencies.
//
// API
//   await loadWorldModel(url)                 -> model (parsed + typed arrays; also prepareWorldModel(json))
//   encode(model, image: Float32Array(3*S*S)) -> Float32Array(d)          z = f(o)
//   predict(model, Z, A, n)                   -> Float32Array(n*d)        batched ẑ' = z + MLP([z, a])
//   rollout(model, z0, actions, H)            -> Float32Array(H*d)        one open-loop imagination
//   planCEM(model, z0, zGoal, opts, rng)      -> {actions: Float32Array(H*2), mean, imaginedLatents: Float32Array(H*d),
//                                                costHistory: number[], cost}
//   shiftPlan(mean, actionDim)                -> warm start for the next MPC step (drop first action, pad 0)
//   nearestBank(model, z)                     -> bank index (use model.bank.states[i] + env.render to show it;
//                                                this is RETRIEVAL of a real frame, not generation)
//   latentDistance(a, b)                      -> ||a - b||²
//   mulberry32(seed), gaussian(rng)           seeded PRNG + standard normal (Box-Muller)
//
// Cost (same as Python, jepa_studio/world/planner.py): C(a) = Σ_{h=1..H} ||ẑ_h - z_goal||²
// ("final" = only h = H). CEM update: μ ← mean(elites), σ ← std(elites) + minStd.

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

/** Standard normal via Box-Muller; caches the second value on the rng function. */
export function gaussian(rng) {
  if (rng._spare !== undefined) {
    const v = rng._spare;
    rng._spare = undefined;
    return v;
  }
  let u = 0;
  while (u <= 1e-12) u = rng();
  const v = rng();
  const r = Math.sqrt(-2 * Math.log(u));
  rng._spare = r * Math.sin(2 * Math.PI * v);
  return r * Math.cos(2 * Math.PI * v);
}

// ------------------------------------------------------------------ loading

const F32 = (a) => (a instanceof Float32Array ? a : Float32Array.from(a));

export function prepareWorldModel(doc) {
  if (doc.format !== "jepa-studio-world/v1") throw new Error(`unsupported world model format ${doc.format}`);
  const prep = (layers) => layers.map((L) => {
    const o = { ...L };
    for (const k of ["W", "b", "mean", "weight", "bias"]) if (L[k] !== undefined) o[k] = F32(L[k]);
    return o;
  });
  const d = doc.latent_dim;
  const bankLat = F32(doc.bank.latents);
  const bankNorm = new Float32Array(doc.bank.size);
  for (let i = 0; i < doc.bank.size; i++) {
    let s = 0;
    for (let k = 0; k < d; k++) s += bankLat[i * d + k] * bankLat[i * d + k];
    bankNorm[i] = s;
  }
  return {
    ...doc,
    encoder: { layers: prep(doc.encoder.layers) },
    predictor: { ...doc.predictor, layers: prep(doc.predictor.layers) },
    bank: { ...doc.bank, latents: bankLat, norms: bankNorm },
    _scratch: {},
  };
}

export async function loadWorldModel(url) {
  const res = await fetch(url);
  if (!res.ok) throw new Error(`failed to load ${url}: ${res.status}`);
  return prepareWorldModel(await res.json());
}

// ------------------------------------------------------------------ layers

function conv2d(L, x) {
  const { in: C, out: O, k: K, stride: st, pad, in_h: H, in_w: W, out_h: OH, out_w: OW } = L;
  const y = new Float32Array(O * OH * OW);
  const w = L.W, b = L.b;
  for (let o = 0; o < O; o++) {
    for (let oy = 0; oy < OH; oy++) {
      for (let ox = 0; ox < OW; ox++) {
        let s = b[o];
        const iy0 = oy * st - pad, ix0 = ox * st - pad;
        for (let c = 0; c < C; c++) {
          const wBase = ((o * C + c) * K) * K, xBase = c * H * W;
          for (let ky = 0; ky < K; ky++) {
            const iy = iy0 + ky;
            if (iy < 0 || iy >= H) continue;
            const wRow = wBase + ky * K, xRow = xBase + iy * W;
            for (let kx = 0; kx < K; kx++) {
              const ix = ix0 + kx;
              if (ix < 0 || ix >= W) continue;
              s += w[wRow + kx] * x[xRow + ix];
            }
          }
        }
        y[(o * OH + oy) * OW + ox] = s;
      }
    }
  }
  return y;
}

/** Batched linear: X (n × in) -> Y (n × out), W row-major (out × in).
 *  Register-blocked 4 rows × 2 outputs: every loaded weight is used 4 times and every loaded
 *  input twice, ~3x faster than the naive triple loop in V8. Each output is still accumulated
 *  as b + Σ_i w_i x_i from i = 0 upward (float64 accumulators, rounded once on store). */
function linear(L, X, n, Y) {
  const I = L.in, O = L.out, w = L.W, b = L.b;
  const y = Y || new Float32Array(n * O);
  const O2 = O - (O % 2);
  let r = 0;
  for (; r + 4 <= n; r += 4) {
    const x0 = r * I, x1 = x0 + I, x2 = x1 + I, x3 = x2 + I;
    const y0 = r * O, y1 = y0 + O, y2 = y1 + O, y3 = y2 + O;
    let o = 0;
    for (; o < O2; o += 2) {
      const w0 = o * I, w1 = w0 + I;
      let a0 = b[o], a1 = a0, a2 = a0, a3 = a0;
      let c0 = b[o + 1], c1 = c0, c2 = c0, c3 = c0;
      for (let i = 0; i < I; i++) {
        const u = w[w0 + i], v = w[w1 + i];
        const p0 = X[x0 + i], p1 = X[x1 + i], p2 = X[x2 + i], p3 = X[x3 + i];
        a0 += u * p0; a1 += u * p1; a2 += u * p2; a3 += u * p3;
        c0 += v * p0; c1 += v * p1; c2 += v * p2; c3 += v * p3;
      }
      y[y0 + o] = a0; y[y1 + o] = a1; y[y2 + o] = a2; y[y3 + o] = a3;
      y[y0 + o + 1] = c0; y[y1 + o + 1] = c1; y[y2 + o + 1] = c2; y[y3 + o + 1] = c3;
    }
    for (; o < O; o++) {
      const w0 = o * I;
      let a0 = b[o], a1 = a0, a2 = a0, a3 = a0;
      for (let i = 0; i < I; i++) {
        const u = w[w0 + i];
        a0 += u * X[x0 + i]; a1 += u * X[x1 + i]; a2 += u * X[x2 + i]; a3 += u * X[x3 + i];
      }
      y[y0 + o] = a0; y[y1 + o] = a1; y[y2 + o] = a2; y[y3 + o] = a3;
    }
  }
  for (; r < n; r++) {
    const xo = r * I, yo = r * O;
    for (let o = 0; o < O; o++) {
      let s = b[o];
      const wo = o * I;
      for (let i = 0; i < I; i++) s += w[wo + i] * X[xo + i];
      y[yo + o] = s;
    }
  }
  return y;
}

const SQRT1_2 = Math.SQRT1_2;
function erf(x) { // Abramowitz-Stegun 7.1.26 (|err| < 1.5e-7), only used by "gelu" layers
  const s = x < 0 ? -1 : 1, a = Math.abs(x), t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

function act(fn, x) {
  if (fn === "relu") { for (let i = 0; i < x.length; i++) if (x[i] < 0) x[i] = 0; }
  else if (fn === "tanh") { for (let i = 0; i < x.length; i++) x[i] = Math.tanh(x[i]); }
  else if (fn === "gelu") { for (let i = 0; i < x.length; i++) x[i] = 0.5 * x[i] * (1 + erf(x[i] * SQRT1_2)); }
  else throw new Error(`unknown activation ${fn}`);
  return x;
}

function layernorm(L, x, n) {
  const D = L.dim;
  for (let r = 0; r < n; r++) {
    let m = 0, v = 0;
    for (let i = 0; i < D; i++) m += x[r * D + i];
    m /= D;
    for (let i = 0; i < D; i++) { const t = x[r * D + i] - m; v += t * t; }
    const inv = 1 / Math.sqrt(v / D + L.eps);
    for (let i = 0; i < D; i++) x[r * D + i] = (x[r * D + i] - m) * inv * L.weight[i] + L.bias[i];
  }
  return x;
}

// ------------------------------------------------------------------ encoder / predictor

/** z = f(o). image: Float32Array(3*S*S) CHW in [0,1]. */
export function encode(model, image) {
  let x = image;
  for (const L of model.encoder.layers) {
    switch (L.type) {
      case "normalize": {
        const y = new Float32Array(x.length);
        for (let i = 0; i < x.length; i++) y[i] = (x[i] - L.mean[i]) / L.std;
        x = y; break;
      }
      case "conv2d": x = conv2d(L, x); break;
      case "flatten": break;
      case "linear": x = linear(L, x, 1); break;
      case "act": x = act(L.fn, x === image ? Float32Array.from(x) : x); break;
      case "layernorm": x = layernorm(L, Float32Array.from(x), 1); break;
      default: throw new Error(`unknown encoder layer ${L.type}`);
    }
  }
  return x;
}

function scratch(model, key, size) {
  const s = model._scratch || (model._scratch = {});
  if (!s[key] || s[key].length < size) s[key] = new Float32Array(size);
  return s[key];
}

/**
 * Batched predictor. Z: Float32Array(n*d), A: Float32Array(n*actionDim).
 * Returns Float32Array(n*d) (a fresh array unless `out` is given).
 */
export function predict(model, Z, A, n, out) {
  const d = model.latent_dim, ad = model.action_dim, din = d + ad;
  let x = scratch(model, "pin", n * din);
  for (let r = 0; r < n; r++) {
    for (let k = 0; k < d; k++) x[r * din + k] = Z[r * d + k];
    for (let k = 0; k < ad; k++) x[r * din + d + k] = A[r * ad + k];
  }
  let li = 0, width = din;
  for (const L of model.predictor.layers) {
    if (L.type === "linear") {
      x = linear(L, x, n, scratch(model, `p${li++ % 2}`, n * L.out));
      width = L.out;
    } else if (L.type === "act") {
      act(L.fn, x.subarray(0, n * width));   // scratch buffers may be longer than n*width
    } else if (L.type === "layernorm") {
      layernorm(L, x, n);
    } else throw new Error(`unknown predictor layer ${L.type}`);
  }
  const y = out || new Float32Array(n * d);
  if (model.predictor.residual) for (let i = 0; i < n * d; i++) y[i] = Z[i] + x[i];
  else for (let i = 0; i < n * d; i++) y[i] = x[i];
  return y;
}

/** Open-loop imagination: z0 (d), actions (H*actionDim) -> Float32Array(H*d) = ẑ_1..ẑ_H. */
export function rollout(model, z0, actions, H) {
  const d = model.latent_dim, ad = model.action_dim;
  const out = new Float32Array(H * d);
  let z = Float32Array.from(z0);
  for (let h = 0; h < H; h++) {
    z = predict(model, z, actions.subarray(h * ad, (h + 1) * ad), 1);
    out.set(z, h * d);
  }
  return out;
}

export function latentDistance(a, b) {
  let s = 0;
  for (let i = 0; i < a.length; i++) { const t = a[i] - b[i]; s += t * t; }
  return s;
}

/** Index of the nearest bank latent (squared Euclidean). Retrieval, not generation. */
export function nearestBank(model, z) {
  const { latents, norms, size } = model.bank, d = model.latent_dim;
  let zz = 0;
  for (let k = 0; k < d; k++) zz += z[k] * z[k];
  let best = 0, bestD = Infinity;
  for (let i = 0; i < size; i++) {
    let dot = 0;
    const o = i * d;
    for (let k = 0; k < d; k++) dot += latents[o + k] * z[k];
    const dist = zz - 2 * dot + norms[i];
    if (dist < bestD) { bestD = dist; best = i; }
  }
  return best;
}

// ------------------------------------------------------------------ CEM

/**
 * Cross-entropy method in latent space.
 * opts: {horizon, samples, elites, iters, initStd=1, minStd=0.05, cost="sum"|"final", initMean?: Float32Array(H*ad)}
 *       (missing keys fall back to model.plan_defaults)
 * rng:  () => uniform [0,1)  (use mulberry32(seed) for reproducibility)
 */
export function planCEM(model, z0, zGoal, opts = {}, rng = mulberry32(0)) {
  const pd = model.plan_defaults || {};
  const H = opts.horizon ?? pd.horizon ?? 5;
  const N = opts.samples ?? pd.samples ?? 300;
  const E = Math.min(opts.elites ?? pd.elites ?? 30, N);
  const iters = opts.iters ?? pd.iters ?? 10;
  const initStd = opts.initStd ?? pd.init_std ?? 1.0;
  const minStd = opts.minStd ?? pd.min_std ?? 0.05;
  const costKind = opts.cost ?? pd.cost ?? "sum";
  const d = model.latent_dim, ad = model.action_dim, P = H * ad;

  const mu = opts.initMean ? Float32Array.from(opts.initMean) : new Float32Array(P);
  const sd = new Float32Array(P).fill(initStd);
  const acts = new Float32Array(N * P);          // sample-major: acts[i*P + h*ad + k]
  const Z = new Float32Array(N * d), Znext = new Float32Array(N * d), At = new Float32Array(N * ad);
  const cost = new Float64Array(N);
  const order = new Int32Array(N);
  const costHistory = [];
  let bestCost = Infinity;
  const bestA = Float32Array.from(mu);

  for (let it = 0; it < iters; it++) {
    for (let i = 0; i < N; i++) {
      for (let p = 0; p < P; p++) {
        const e = i === 0 ? 0 : gaussian(rng);  // sample 0 = the current mean
        let v = mu[p] + sd[p] * e;
        acts[i * P + p] = v < -1 ? -1 : v > 1 ? 1 : v;
      }
      for (let k = 0; k < d; k++) Z[i * d + k] = z0[k];
      cost[i] = 0;
    }
    let cur = Z, nxt = Znext;
    for (let h = 0; h < H; h++) {
      for (let i = 0; i < N; i++) for (let k = 0; k < ad; k++) At[i * ad + k] = acts[i * P + h * ad + k];
      predict(model, cur, At, N, nxt);
      if (costKind === "sum" || h === H - 1) {
        for (let i = 0; i < N; i++) {
          let s = 0;
          for (let k = 0; k < d; k++) { const t = nxt[i * d + k] - zGoal[k]; s += t * t; }
          cost[i] += s;
        }
      }
      const tmp = cur; cur = nxt; nxt = tmp;
    }
    for (let i = 0; i < N; i++) order[i] = i;
    order.sort((a, b) => cost[a] - cost[b]);
    // μ ← mean(elites), σ ← std(elites) + minStd
    let eliteMean = 0;
    for (let p = 0; p < P; p++) {
      let m = 0;
      for (let j = 0; j < E; j++) m += acts[order[j] * P + p];
      m /= E;
      let v = 0;
      for (let j = 0; j < E; j++) { const t = acts[order[j] * P + p] - m; v += t * t; }
      mu[p] = m;
      sd[p] = Math.sqrt(v / E) + minStd;
    }
    for (let j = 0; j < E; j++) eliteMean += cost[order[j]];
    costHistory.push(eliteMean / E);
    if (cost[order[0]] < bestCost) {
      bestCost = cost[order[0]];
      bestA.set(acts.subarray(order[0] * P, order[0] * P + P));
    }
  }
  return { actions: bestA, mean: mu, imaginedLatents: rollout(model, z0, bestA, H), costHistory, cost: bestCost };
}

/** Warm start for the next receding-horizon step: drop the executed action, append zeros. */
export function shiftPlan(mean, actionDim = 2) {
  const out = new Float32Array(mean.length);
  out.set(mean.subarray(actionDim));
  return out;
}
