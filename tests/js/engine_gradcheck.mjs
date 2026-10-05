// Finite-difference check of the hand-written LeJEPA backward pass (float64 throughout).
// 1) dL/dZ of lejepaLoss vs central differences on Z
// 2) dL/dW through the whole mlp-tiny (small random input) vs central differences
// Exit code 1 if any relative error >= 1e-3. Run: node tests/js/engine_gradcheck.mjs
import { lejepaLoss, stepDirections, epConstants, epStat } from '../../site/js/engine/lejepa.js';
import { initParams, zeroGrads, forward, backward } from '../../site/js/engine/mlp.js';
import { mulberry32, gaussian } from '../../site/js/engine/rng.js';

const TOL = 1e-3;
let worst = 0;
let failed = false;

function relErr(a, b) { return Math.abs(a - b) / Math.max(1e-6, Math.abs(a) + Math.abs(b)); }
function report(name, errs) {
  const m = Math.max(...errs);
  worst = Math.max(worst, m);
  const ok = m < TOL;
  if (!ok) failed = true;
  console.log(`${ok ? 'PASS' : 'FAIL'} ${name}: max relative error ${m.toExponential(2)} over ${errs.length} coords`);
}

const g = gaussian(mulberry32(7));

// ---- 0. EP statistic gradient in isolation
{
  const ep = epConstants(3.0, 17);
  const N = 20;
  const x = new Float64Array(N).map(() => g() * 1.3 + 0.2);
  const gx = new Float64Array(N);
  epStat(x, N, ep, gx);
  const errs = [];
  const h = 1e-5;
  for (let n = 0; n < N; n++) {
    const o = x[n];
    x[n] = o + h; const lp = epStat(x, N, ep, null);
    x[n] = o - h; const lm = epStat(x, N, ep, null);
    x[n] = o;
    errs.push(relErr((lp - lm) / (2 * h), gx[n]));
  }
  report('Epps-Pulley dEP/dx', errs);
}

// ---- 1. loss gradient w.r.t. projections
for (const lam of [0.05, 0.5, 1.0, 0.0]) {
  const V = 6, Vg = 2, N = 12, K = 5, M = 7;
  const Z = new Float64Array(V * N * K).map(() => g());
  const dirs = stepDirections(V, K, M, 3, 11);
  const out = lejepaLoss(Z, V, Vg, N, K, dirs, M, lam);
  const errs = [];
  const h = 1e-5;
  for (let i = 0; i < Z.length; i += 3) {
    const o = Z[i];
    Z[i] = o + h; const lp = lejepaLoss(Z, V, Vg, N, K, dirs, M, lam, { grad: false }).loss;
    Z[i] = o - h; const lm = lejepaLoss(Z, V, Vg, N, K, dirs, M, lam, { grad: false }).loss;
    Z[i] = o;
    const fd = (lp - lm) / (2 * h);
    if (Math.abs(fd) + Math.abs(out.dZ[i]) > 1e-9) errs.push(relErr(fd, out.dZ[i]));
  }
  report(`lejepa dL/dZ (lambda=${lam})`, errs);
}

// ---- 2. end-to-end through the MLP (smaller layer sizes, same code path)
{
  const spec = {
    side: 4, channels: 3, embedLayer: 1,
    layers: [
      { name: 'backbone.net.0', din: 48, dout: 20, act: 'gelu' },
      { name: 'backbone.net.2', din: 20, dout: 10, act: null },
      { name: 'projector.net.0', din: 10, dout: 10, act: 'gelu' },
      { name: 'projector.net.2', din: 10, dout: 4, act: null },
    ],
  };
  const params = initParams(5, spec, Float64Array);
  const V = 4, Vg = 2, N = 10, K = 4, M = 6;
  const X = new Float64Array(V * N * 48).map(() => g());
  const dirs = stepDirections(V, K, M, 1, 2);
  const lossOf = () => {
    const c = forward(params, X, V * N);
    return lejepaLoss(c.out, V, Vg, N, K, dirs, M, 0.3, { grad: false }).loss;
  };
  const cache = forward(params, X, V * N);
  const out = lejepaLoss(cache.out, V, Vg, N, K, dirs, M, 0.3);
  const grads = zeroGrads(params);
  backward(params, cache, out.dZ, grads);
  const h = 1e-6;
  for (let li = 0; li < params.length; li++) {
    for (const key of ['W', 'b']) {
      const P = params[li][key], G = grads[li][key];
      const errs = [];
      const stride = Math.max(1, Math.floor(P.length / 25));
      for (let i = 0; i < P.length; i += stride) {
        const o = P[i];
        P[i] = o + h; const lp = lossOf();
        P[i] = o - h; const lm = lossOf();
        P[i] = o;
        const fd = (lp - lm) / (2 * h);
        if (Math.abs(fd) + Math.abs(G[i]) > 1e-9) errs.push(relErr(fd, G[i]));
      }
      report(`mlp ${params[li].name}.${key === 'W' ? 'weight' : 'bias'}`, errs);
    }
  }
}

console.log(`worst relative error ${worst.toExponential(2)} (tolerance ${TOL})`);
process.exit(failed ? 1 : 0);
