// LeJEPA objective and its hand-derived gradient, matching jepa_studio/losses.py.
//
//   centers  c_n      = (1/Vg) sum_g z_{g,n}
//   L_pred            = mean_{v,n,k} (c_{n,k} - z_{v,n,k})^2
//   EP(x_1..x_N)      = N * sum_j W_j [ (C_j - phi_j)^2 + S_j^2 ]
//                       C_j = mean_n cos(t_j x_n), S_j = mean_n sin(t_j x_n), W_j = quad_j * exp(-t_j^2/2)
//   L_sigreg          = (1/V) sum_v (1/M) sum_m EP(Z_v a_{v,m})      (fresh directions per view)
//   L                 = (1 - lambda) L_pred + lambda L_sigreg
//
// Gradient of EP w.r.t. one projected value x_n (the N and the 1/N inside C_j, S_j cancel):
//   dEP/dx_n = sum_j W_j * 2 [ (C_j - phi_j)(-t_j sin(t_j x_n)) + S_j t_j cos(t_j x_n) ]
// then dL_sigreg/dz_{v,n,k} = (1/(V M)) sum_m dEP_m/dx_{n,m} * a_{v,k,m}.
//
// The knots are equally spaced from 0 (t_j = j * dt), so cos/sin(t_j x) come from one cos and
// one sin per sample via the Chebyshev recurrence c_{j+1} = 2 cos(dt x) c_j - c_{j-1}.
import { gaussian, mulberry32, mixSeed } from './rng.js';

/** Same quadrature as losses.trapezoid_half_line + EppsPulley buffers. */
export function epConstants(tMax = 3.0, knots = 17) {
  if (knots < 2) throw new Error('knots must be >= 2');
  const dt = tMax / (knots - 1);
  const t = new Float64Array(knots), phi = new Float64Array(knots), weights = new Float64Array(knots);
  for (let j = 0; j < knots; j++) {
    t[j] = tMax * j / (knots - 1);
    const w = (j === 0 || j === knots - 1) ? dt : 2 * dt;
    phi[j] = Math.exp(-0.5 * t[j] * t[j]);
    weights[j] = w * phi[j];
  }
  return { t, phi, weights, dt, knots, tMax };
}

/** E[EP] under N(0,1) for this quadrature (losses.ep_expected_under_null). */
export function epExpectedUnderNull(tMax = 3.0, knots = 17) {
  const { t, weights } = epConstants(tMax, knots); // weights already include exp(-t^2/2)
  let s = 0;
  for (let j = 0; j < knots; j++) s += (1 - Math.exp(-t[j] * t[j])) * weights[j];
  return s;
}

/**
 * EP statistic of x (length N) and, optionally, its gradient written into gx (length N).
 * cs: scratch Float64Array(2*knots) for C_j and S_j.
 */
export function epStat(x, N, ep, gx) {
  const { dt, knots, phi, weights, t } = ep;
  const C = new Float64Array(knots), S = new Float64Array(knots);
  for (let n = 0; n < N; n++) {
    const a = dt * x[n];
    const c1 = Math.cos(a), s1 = Math.sin(a), two = 2 * c1;
    let cPrev = 1, sPrev = 0, cCur = c1, sCur = s1;
    C[0] += 1;
    if (knots > 1) { C[1] += c1; S[1] += s1; }
    for (let j = 2; j < knots; j++) {
      const cN = two * cCur - cPrev, sN = two * sCur - sPrev;
      C[j] += cN; S[j] += sN;
      cPrev = cCur; sPrev = sCur; cCur = cN; sCur = sN;
    }
  }
  let stat = 0;
  const coefC = new Float64Array(knots), coefS = new Float64Array(knots);
  for (let j = 0; j < knots; j++) {
    C[j] /= N; S[j] /= N;
    const dc = C[j] - phi[j];
    stat += weights[j] * (dc * dc + S[j] * S[j]);
    coefC[j] = -2 * weights[j] * dc * t[j];   // multiplies sin(t_j x)
    coefS[j] = 2 * weights[j] * S[j] * t[j];  // multiplies cos(t_j x)
  }
  stat *= N;
  if (gx) {
    for (let n = 0; n < N; n++) {
      const a = dt * x[n];
      const c1 = Math.cos(a), s1 = Math.sin(a), two = 2 * c1;
      let g = 0;
      if (knots > 1) g += coefC[1] * s1 + coefS[1] * c1;
      let cPrev = 1, sPrev = 0, cCur = c1, sCur = s1;
      for (let j = 2; j < knots; j++) {
        const cN = two * cCur - cPrev, sN = two * sCur - sPrev;
        g += coefC[j] * sN + coefS[j] * cN;
        cPrev = cCur; sPrev = sCur; cCur = cN; sCur = sN;
      }
      gx[n] = g;
    }
  }
  return stat;
}

/** M unit directions in R^K as a (K, M) row-major matrix (columns are directions), like losses.random_directions. */
export function randomDirections(K, M, seed) {
  const g = gaussian(mulberry32(seed));
  const a = new Float64Array(K * M);
  for (let i = 0; i < K * M; i++) a[i] = g();
  for (let m = 0; m < M; m++) {
    let s = 0;
    for (let k = 0; k < K; k++) s += a[k * M + m] ** 2;
    const inv = 1 / Math.max(Math.sqrt(s), 1e-12);
    for (let k = 0; k < K; k++) a[k * M + m] *= inv;
  }
  return a;
}

/** Directions for every view of one training step (fresh per view, like SIGReg called per view). */
export function stepDirections(V, K, M, seed, step) {
  const out = [];
  for (let v = 0; v < V; v++) out.push(randomDirections(K, M, mixSeed(mixSeed(seed, step), v + 1)));
  return out;
}

/** SIGReg of one (N, K) block with directions a (K, M); gradient accumulated into gz (scaled). */
export function sigregBlock(z, zOff, N, K, a, M, ep, gz, scale) {
  const x = new Float64Array(N), gx = gz ? new Float64Array(N) : null;
  let total = 0;
  for (let m = 0; m < M; m++) {
    for (let n = 0; n < N; n++) {
      let s = 0;
      const zo = zOff + n * K;
      for (let k = 0; k < K; k++) s += z[zo + k] * a[k * M + m];
      x[n] = s;
    }
    total += epStat(x, N, ep, gx);
    if (gz) {
      for (let n = 0; n < N; n++) {
        const g = gx[n] * scale;
        const zo = zOff + n * K;
        for (let k = 0; k < K; k++) gz[zo + k] += g * a[k * M + m];
      }
    }
  }
  return total / M;
}

/**
 * Full LeJEPA loss for projections Z laid out (V, N, K), first Vg views global.
 * dirs: array of V direction matrices (K, M). Returns {loss, pred, sigreg, dZ?}.
 */
export function lejepaLoss(Z, V, Vg, N, K, dirs, M, lam = 0.05, opts = {}) {
  if (!(lam >= 0 && lam <= 1)) throw new Error('lambda must be in [0, 1]');
  const ep = opts.ep || epConstants(opts.tMax ?? 3.0, opts.knots ?? 17);
  const wantGrad = opts.grad !== false;
  const NK = N * K, T = V * NK;
  const c = new Float64Array(NK);
  for (let g = 0; g < Vg; g++) for (let i = 0; i < NK; i++) c[i] += Z[g * NK + i];
  for (let i = 0; i < NK; i++) c[i] /= Vg;
  let pred = 0;
  const S = new Float64Array(NK);
  for (let v = 0; v < V; v++) {
    for (let i = 0; i < NK; i++) {
      const d = c[i] - Z[v * NK + i];
      pred += d * d;
      S[i] += d;
    }
  }
  pred /= T;
  const dZ = wantGrad ? new (opts.Arr || Float64Array)(T) : null;
  const gSig = wantGrad ? new Float64Array(T) : null;
  let sig = 0;
  for (let v = 0; v < V; v++) sig += sigregBlock(Z, v * NK, N, K, dirs[v], M, ep, gSig, 1 / (V * M));
  sig /= V;
  if (wantGrad) {
    const a = (1 - lam) * 2 / T;
    for (let v = 0; v < V; v++) {
      for (let i = 0; i < NK; i++) {
        let g = -a * (c[i] - Z[v * NK + i]);
        if (v < Vg) g += a * S[i] / Vg;
        dZ[v * NK + i] = g + lam * gSig[v * NK + i];
      }
    }
  }
  return { loss: (1 - lam) * pred + lam * sig, pred, sigreg: sig, dZ };
}

/** SIGReg statistic alone for an (N, K) matrix (used by the Inspect tab). */
export function sigregValue(Z, N, K, M, seed, ep = epConstants()) {
  return sigregBlock(Z, 0, N, K, randomDirections(K, M, seed), M, ep, null, 0);
}
