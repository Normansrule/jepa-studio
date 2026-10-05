// Dense math on flat typed arrays (row-major). Everything here is written so the same
// code runs on Float32Array (training) and Float64Array (gradient checks in Node).
//
// Weight layout follows PyTorch nn.Linear: W is (out, in), so y = x W^T + b. That keeps the
// exported .safetensors tensors loadable by the desktop models without transposes.

/** Y[n, o] = b[o] + sum_i X[n, i] * W[o, i].  X (n, din), W (dout, din), Y (n, dout).
 *  Register-blocked 4 rows x 2 outputs (about 3x faster than the naive loop in V8). */
export function linearForward(X, n, din, W, b, dout, Y) {
  let r = 0;
  for (; r + 3 < n; r += 4) {
    const x0 = r * din, x1 = x0 + din, x2 = x1 + din, x3 = x2 + din;
    const y0 = r * dout, y1 = y0 + dout, y2 = y1 + dout, y3 = y2 + dout;
    let o = 0;
    for (; o + 1 < dout; o += 2) {
      const w0 = o * din, w1 = w0 + din;
      let a00 = 0, a01 = 0, a10 = 0, a11 = 0, a20 = 0, a21 = 0, a30 = 0, a31 = 0;
      for (let i = 0; i < din; i++) {
        const u = W[w0 + i], v = W[w1 + i];
        const p = X[x0 + i], q = X[x1 + i], s = X[x2 + i], t = X[x3 + i];
        a00 += p * u; a01 += p * v; a10 += q * u; a11 += q * v;
        a20 += s * u; a21 += s * v; a30 += t * u; a31 += t * v;
      }
      const b0 = b ? b[o] : 0, b1 = b ? b[o + 1] : 0;
      Y[y0 + o] = a00 + b0; Y[y0 + o + 1] = a01 + b1;
      Y[y1 + o] = a10 + b0; Y[y1 + o + 1] = a11 + b1;
      Y[y2 + o] = a20 + b0; Y[y2 + o + 1] = a21 + b1;
      Y[y3 + o] = a30 + b0; Y[y3 + o + 1] = a31 + b1;
    }
    for (; o < dout; o++) {
      const w0 = o * din;
      let a0 = 0, a1 = 0, a2 = 0, a3 = 0;
      for (let i = 0; i < din; i++) { const u = W[w0 + i]; a0 += X[x0 + i] * u; a1 += X[x1 + i] * u; a2 += X[x2 + i] * u; a3 += X[x3 + i] * u; }
      const bb = b ? b[o] : 0;
      Y[y0 + o] = a0 + bb; Y[y1 + o] = a1 + bb; Y[y2 + o] = a2 + bb; Y[y3 + o] = a3 + bb;
    }
  }
  for (; r < n; r++) {
    const xo = r * din, yo = r * dout;
    for (let o = 0; o < dout; o++) {
      const wo = o * din;
      let s = 0;
      for (let i = 0; i < din; i++) s += X[xo + i] * W[wo + i];
      Y[yo + o] = s + (b ? b[o] : 0);
    }
  }
  return Y;
}

/**
 * Backward of linearForward. Accumulates into dW (dout, din) and db (dout) (+=), and writes
 * dX (n, din) when given (overwrites).
 *   dW[o, i] += sum_r dY[r, o] X[r, i]      (blocked over 4 rows of the batch)
 *   dX[r, i]  = sum_o dY[r, o] W[o, i]
 */
export function linearBackward(dY, X, n, din, W, dout, dW, db, dX) {
  if (db) for (let r = 0; r < n; r++) for (let o = 0; o < dout; o++) db[o] += dY[r * dout + o];
  if (dW) {
    let r = 0;
    for (; r + 3 < n; r += 4) {
      const x0 = r * din, x1 = x0 + din, x2 = x1 + din, x3 = x2 + din;
      for (let o = 0; o < dout; o++) {
        const g0 = dY[r * dout + o], g1 = dY[(r + 1) * dout + o], g2 = dY[(r + 2) * dout + o], g3 = dY[(r + 3) * dout + o];
        if (g0 === 0 && g1 === 0 && g2 === 0 && g3 === 0) continue;
        const wo = o * din;
        for (let i = 0; i < din; i++) dW[wo + i] += g0 * X[x0 + i] + g1 * X[x1 + i] + g2 * X[x2 + i] + g3 * X[x3 + i];
      }
    }
    for (; r < n; r++) {
      const xo = r * din;
      for (let o = 0; o < dout; o++) {
        const g = dY[r * dout + o];
        if (g === 0) continue;
        const wo = o * din;
        for (let i = 0; i < din; i++) dW[wo + i] += g * X[xo + i];
      }
    }
  }
  if (dX) {
    dX.fill(0);
    for (let r = 0; r < n; r++) {
      const xo = r * din;
      let o = 0;
      for (; o + 3 < dout; o += 4) {
        const g0 = dY[r * dout + o], g1 = dY[r * dout + o + 1], g2 = dY[r * dout + o + 2], g3 = dY[r * dout + o + 3];
        const w0 = o * din, w1 = w0 + din, w2 = w1 + din, w3 = w2 + din;
        for (let i = 0; i < din; i++) dX[xo + i] += g0 * W[w0 + i] + g1 * W[w1 + i] + g2 * W[w2 + i] + g3 * W[w3 + i];
      }
      for (; o < dout; o++) {
        const g = dY[r * dout + o];
        if (g === 0) continue;
        const wo = o * din;
        for (let i = 0; i < din; i++) dX[xo + i] += g * W[wo + i];
      }
    }
  }
}

// --- GELU, exact (erf) form, as PyTorch nn.GELU() default -------------------------------
// erf via Abramowitz & Stegun 7.1.26 (|error| < 1.5e-7), so the web model and the desktop
// MLPTiny compute the same function within float32 noise.
const INV_SQRT2 = 0.7071067811865476;
const INV_SQRT_2PI = 0.3989422804014327;

export function erf(x) {
  const s = x < 0 ? -1 : 1;
  const a = Math.abs(x);
  const t = 1 / (1 + 0.3275911 * a);
  const y = 1 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * Math.exp(-a * a);
  return s * y;
}

/** out[i] = x * Phi(x). */
export function geluForward(X, out) {
  for (let i = 0; i < X.length; i++) {
    const x = X[i];
    out[i] = 0.5 * x * (1 + erf(x * INV_SQRT2));
  }
  return out;
}

/** dX[i] = dY[i] * (Phi(x) + x * phi(x)). */
export function geluBackward(dY, X, dX) {
  for (let i = 0; i < X.length; i++) {
    const x = X[i];
    const cdf = 0.5 * (1 + erf(x * INV_SQRT2));
    const pdf = INV_SQRT_2PI * Math.exp(-0.5 * x * x);
    dX[i] = dY[i] * (cdf + x * pdf);
  }
  return dX;
}

export function relu(X, out) { for (let i = 0; i < X.length; i++) out[i] = X[i] > 0 ? X[i] : 0; return out; }

/** Mean of each column of a (n, d) matrix. */
export function colMean(Z, n, d) {
  const m = new Float64Array(d);
  for (let r = 0; r < n; r++) for (let k = 0; k < d; k++) m[k] += Z[r * d + k];
  for (let k = 0; k < d; k++) m[k] /= n;
  return m;
}

/** Row-wise L2 normalization (returns a new Float64Array). */
export function normalizeRows(Z, n, d) {
  const out = new Float64Array(n * d);
  for (let r = 0; r < n; r++) {
    let s = 0;
    for (let k = 0; k < d; k++) s += Z[r * d + k] * Z[r * d + k];
    const inv = 1 / Math.max(Math.sqrt(s), 1e-12);
    for (let k = 0; k < d; k++) out[r * d + k] = Z[r * d + k] * inv;
  }
  return out;
}
