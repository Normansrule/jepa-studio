// Evaluation and diagnostics, ported from jepa_studio/evaluate.py:
//   knnAccuracy       = knn_accuracy (cosine, majority vote, nearest-neighbour tie-break)
//   isotropyReport    = isotropy_report (eigen spectrum, effective rank, ratio, SIGReg)
//   collapseCheck     = collapse_check (same thresholds and messages)
//   linearProbe       = linear_probe (standardized features, full-batch AdamW on softmax regression)
// plus a Jacobi eigen-solver for symmetric matrices (PCA) and cosine nearest neighbours.
import { normalizeRows } from './tensor.js';
import { sigregValue, epConstants } from './lejepa.js';
import { gaussian, mulberry32, permutation } from './rng.js';

const round = (x, d) => { const f = 10 ** d; return Math.round(x * f) / f; };

/** Cyclic Jacobi for a symmetric (d, d) matrix. Returns {values (desc), vectors (d, d) columns}. */
export function symmetricEigen(A, d, { sweeps = 60, tol = 1e-12 } = {}) {
  const a = Float64Array.from(A);
  const v = new Float64Array(d * d);
  for (let i = 0; i < d; i++) v[i * d + i] = 1;
  for (let s = 0; s < sweeps; s++) {
    let off = 0;
    for (let p = 0; p < d; p++) for (let q = p + 1; q < d; q++) off += a[p * d + q] ** 2;
    let diag = 0;
    for (let p = 0; p < d; p++) diag += a[p * d + p] ** 2;
    if (off <= tol * tol * Math.max(diag, 1e-300)) break;
    for (let p = 0; p < d; p++) {
      for (let q = p + 1; q < d; q++) {
        const apq = a[p * d + q];
        if (Math.abs(apq) < 1e-300) continue;
        const app = a[p * d + p], aqq = a[q * d + q];
        const theta = (aqq - app) / (2 * apq);
        const t = Math.sign(theta || 1) / (Math.abs(theta) + Math.sqrt(theta * theta + 1));
        const c = 1 / Math.sqrt(t * t + 1), sn = t * c;
        for (let k = 0; k < d; k++) {
          const akp = a[k * d + p], akq = a[k * d + q];
          a[k * d + p] = c * akp - sn * akq;
          a[k * d + q] = sn * akp + c * akq;
        }
        for (let k = 0; k < d; k++) {
          const apk = a[p * d + k], aqk = a[q * d + k];
          a[p * d + k] = c * apk - sn * aqk;
          a[q * d + k] = sn * apk + c * aqk;
        }
        for (let k = 0; k < d; k++) {
          const vkp = v[k * d + p], vkq = v[k * d + q];
          v[k * d + p] = c * vkp - sn * vkq;
          v[k * d + q] = sn * vkp + c * vkq;
        }
      }
    }
  }
  const order = [...Array(d).keys()].sort((i, j) => a[j * d + j] - a[i * d + i]);
  const values = order.map((i) => a[i * d + i]);
  const vectors = new Float64Array(d * d);
  order.forEach((src, dst) => { for (let k = 0; k < d; k++) vectors[k * d + dst] = v[k * d + src]; });
  return { values, vectors };
}

/** Sample covariance (unbiased) of an (n, d) matrix and its column means. */
export function covariance(Z, n, d) {
  const mu = new Float64Array(d);
  for (let r = 0; r < n; r++) for (let k = 0; k < d; k++) mu[k] += Z[r * d + k];
  for (let k = 0; k < d; k++) mu[k] /= n;
  const C = new Float64Array(d * d);
  const row = new Float64Array(d);
  for (let r = 0; r < n; r++) {
    for (let k = 0; k < d; k++) row[k] = Z[r * d + k] - mu[k];
    for (let i = 0; i < d; i++) {
      const ri = row[i];
      for (let j = i; j < d; j++) C[i * d + j] += ri * row[j];
    }
  }
  const den = Math.max(1, n - 1);
  for (let i = 0; i < d; i++) for (let j = i; j < d; j++) { C[i * d + j] /= den; C[j * d + i] = C[i * d + j]; }
  return { C, mu };
}

/** PCA to `dims` components: {coords (n, dims), varRatio}. */
export function pca(Z, n, d, dims = 3) {
  const { C, mu } = covariance(Z, n, d);
  const { values, vectors } = symmetricEigen(C, d);
  const total = values.reduce((s, x) => s + Math.max(0, x), 0) || 1;
  const coords = new Float32Array(n * dims);
  for (let r = 0; r < n; r++) {
    for (let c = 0; c < dims; c++) {
      let s = 0;
      for (let k = 0; k < d; k++) s += (Z[r * d + k] - mu[k]) * vectors[k * d + c];
      coords[r * dims + c] = s;
    }
  }
  return { coords, varRatio: values.slice(0, dims).map((x) => Math.max(0, x) / total) };
}

/** isotropy_report(z, num_slices, seed). SIGReg directions come from the JS PRNG, so that one
 *  number differs from the desktop value by sampling noise (same expectation). */
export function isotropyReport(Z, n, d, numSlices = 256, seed = 0) {
  const { C, mu } = covariance(Z, n, d);
  const ev = symmetricEigen(C, d).values.map((x) => Math.max(0, x));
  const sum = Math.max(ev.reduce((s, x) => s + x, 0), 1e-12);
  const p = ev.map((x) => x / sum);
  const h = -p.reduce((s, x) => s + x * Math.log(x + 1e-12), 0);
  let meanNorm = 0;
  for (let r = 0; r < n; r++) {
    let s = 0;
    for (let k = 0; k < d; k++) s += Z[r * d + k] ** 2;
    meanNorm += Math.sqrt(s);
  }
  let meanStd = 0;
  for (let k = 0; k < d; k++) meanStd += Math.sqrt(C[k * d + k]);
  return {
    dim: d, samples: n,
    eigenvalues: ev.map((x) => round(x, 6)),
    effective_rank: round(Math.exp(h), 3),
    isotropy_ratio: round(ev[d - 1] / Math.max(ev[0], 1e-12), 5),
    top_eigen_share: round(p[0], 4),
    mean_norm: round(meanNorm / n, 4),
    mean_abs_mean: round(mu.reduce((s, x) => s + Math.abs(x), 0) / d, 4),
    mean_std: round(meanStd / d, 4),
    sigreg: round(sigregValue(Z, n, d, numSlices, seed, epConstants()), 4),
  };
}

/** collapse_check: plain-language verdict, same thresholds as the desktop app. */
export function collapseCheck(rep) {
  const k = rep.dim;
  if (rep.mean_std < 1e-3) return { status: 'complete-collapse', message: 'Every input maps to (almost) the same point.' };
  if (rep.effective_rank < Math.max(1.5, 0.1 * k) || rep.top_eigen_share > 0.9) {
    return { status: 'dimensional-collapse', message: `Embeddings use about ${rep.effective_rank.toFixed(1)} of ${k} directions.` };
  }
  if (rep.isotropy_ratio < 0.01) return { status: 'anisotropic', message: 'Some directions carry far less variance than others.' };
  return { status: 'healthy', message: `Spread over ~${rep.effective_rank.toFixed(1)} of ${k} directions.` };
}

/** Indices of the top-k values of arr (descending). */
function topk(arr, k) {
  const idx = [...arr.keys()];
  idx.sort((i, j) => arr[j] - arr[i] || i - j);
  return idx.slice(0, k);
}

/** knn_accuracy: cosine k-NN, majority vote, ties go to the most similar neighbour's class. */
export function knnAccuracy(xtr, ytr, ntr, xte, yte, nte, d, k = 20) {
  const a = normalizeRows(xtr, ntr, d), b = normalizeRows(xte, nte, d);
  k = Math.min(k, ntr);
  let c = 0;
  for (const y of ytr) c = Math.max(c, y + 1);
  for (const y of yte) c = Math.max(c, y + 1);
  let correct = 0;
  const sims = new Float64Array(ntr);
  for (let q = 0; q < nte; q++) {
    for (let r = 0; r < ntr; r++) {
      let s = 0;
      for (let j = 0; j < d; j++) s += b[q * d + j] * a[r * d + j];
      sims[r] = s;
    }
    const nn = topk(sims, k);
    const counts = new Float64Array(c);
    for (const i of nn) counts[ytr[i]] += 1;
    counts[ytr[nn[0]]] += 1e-3;
    let best = 0;
    for (let j = 1; j < c; j++) if (counts[j] > counts[best]) best = j;
    if (best === yte[q]) correct++;
  }
  return { accuracy: correct / nte, k };
}

/** nearest_neighbors: the k most cosine-similar rows to row `query` (excluding itself). */
export function nearestNeighbors(Z, n, d, query, k = 8) {
  const e = normalizeRows(Z, n, d);
  const s = new Float64Array(n);
  for (let r = 0; r < n; r++) {
    let t = 0;
    for (let j = 0; j < d; j++) t += e[r * d + j] * e[query * d + j];
    s[r] = t;
  }
  s[query] = -2;
  return topk(s, k).map((i) => [i, round(s[i], 4)]);
}

/** Standardize with train statistics (unbiased std + 1e-6), like evaluate.standardize. */
export function standardize(train, ntr, d, ...others) {
  const mu = new Float64Array(d), sd = new Float64Array(d);
  for (let r = 0; r < ntr; r++) for (let k = 0; k < d; k++) mu[k] += train[r * d + k];
  for (let k = 0; k < d; k++) mu[k] /= ntr;
  for (let r = 0; r < ntr; r++) for (let k = 0; k < d; k++) sd[k] += (train[r * d + k] - mu[k]) ** 2;
  for (let k = 0; k < d; k++) sd[k] = Math.sqrt(sd[k] / Math.max(1, ntr - 1)) + 1e-6;
  const f = (X) => { const o = new Float64Array(X.length); for (let i = 0; i < X.length; i++) o[i] = (X[i] - mu[i % d]) / sd[i % d]; return o; };
  return [f(train), ...others.map(f)];
}

/** linear_probe: multinomial logistic regression, full-batch AdamW, standardized features. */
export function linearProbe(xtr, ytr, ntr, xte, yte, nte, d, { epochs = 100, lr = 1e-2, wd = 1e-6, seed = 0 } = {}) {
  const [Xtr, Xte] = standardize(xtr, ntr, d, xte);
  let c = 0;
  for (const y of ytr) c = Math.max(c, y + 1);
  for (const y of yte) c = Math.max(c, y + 1);
  const g = gaussian(mulberry32(seed));
  const W = new Float64Array(c * d).map(() => g() * 0.01), b = new Float64Array(c);
  const mW = new Float64Array(c * d), vW = new Float64Array(c * d), mb = new Float64Array(c), vb = new Float64Array(c);
  const logits = new Float64Array(c);
  let loss = 0;
  const b1 = 0.9, b2 = 0.999, eps = 1e-8;
  for (let t = 1; t <= epochs; t++) {
    const gW = new Float64Array(c * d), gb = new Float64Array(c);
    loss = 0;
    for (let r = 0; r < ntr; r++) {
      let mx = -Infinity;
      for (let j = 0; j < c; j++) {
        let s = b[j];
        for (let k = 0; k < d; k++) s += W[j * d + k] * Xtr[r * d + k];
        logits[j] = s; if (s > mx) mx = s;
      }
      let z = 0;
      for (let j = 0; j < c; j++) { logits[j] = Math.exp(logits[j] - mx); z += logits[j]; }
      loss += -Math.log(logits[ytr[r]] / z);
      for (let j = 0; j < c; j++) {
        const gj = (logits[j] / z - (j === ytr[r] ? 1 : 0)) / ntr;
        gb[j] += gj;
        for (let k = 0; k < d; k++) gW[j * d + k] += gj * Xtr[r * d + k];
      }
    }
    loss /= ntr;
    const bc1 = 1 - b1 ** t, bc2 = 1 - b2 ** t;
    const upd = (P, G, M, Vv) => {
      for (let i = 0; i < P.length; i++) {
        P[i] *= 1 - lr * wd;
        M[i] = b1 * M[i] + (1 - b1) * G[i];
        Vv[i] = b2 * Vv[i] + (1 - b2) * G[i] * G[i];
        P[i] -= (lr / bc1) * M[i] / (Math.sqrt(Vv[i]) / Math.sqrt(bc2) + eps);
      }
    };
    upd(W, gW, mW, vW); upd(b, gb, mb, vb);
  }
  const acc = (X, y, n) => {
    let ok = 0;
    for (let r = 0; r < n; r++) {
      let best = 0, bv = -Infinity;
      for (let j = 0; j < c; j++) {
        let s = b[j];
        for (let k = 0; k < d; k++) s += W[j * d + k] * X[r * d + k];
        if (s > bv) { bv = s; best = j; }
      }
      if (best === y[r]) ok++;
    }
    return ok / n;
  };
  return { accuracy: acc(Xte, yte, nte), train_accuracy: acc(Xtr, ytr, ntr), final_loss: loss };
}

/** split_indices analogue: held-out test (20%) and a labeled subset of the rest. */
export function splitIndices(n, labeledFraction, seed = 0, testFraction = 0.2) {
  const perm = permutation(n, mulberry32((seed ^ 0x5eed) >>> 0));
  const nTest = Math.max(1, Math.floor(n * testFraction));
  const test = perm.slice(0, nTest), pool = perm.slice(nTest);
  const nLab = Math.min(pool.length, Math.max(10, Math.floor(pool.length * labeledFraction)));
  return { labeled: pool.slice(0, nLab), test };
}

/** Rows `idx` of an (n, d) matrix. */
export function gatherRows(Z, d, idx) {
  const out = new Float32Array(idx.length * d);
  idx.forEach((r, i) => out.set(Z.subarray(r * d, r * d + d), i * d));
  return out;
}
