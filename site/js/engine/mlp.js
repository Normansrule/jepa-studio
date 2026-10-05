// "mlp-tiny", the demo-tier encoder that trains in the browser.
//
//   backbone:  x (3*16*16 = 768, CHW flattened) -> Linear(768, 256) -> GELU -> Linear(256, 64) = emb
//   projector: emb -> Linear(64, 64) -> GELU -> Linear(64, 16) = proj   (the loss lives here)
//
// GELU is the exact erf form (PyTorch nn.GELU() default), not the tanh approximation.
// Difference from the desktop encoder: the desktop Projector is 3 Linear layers with
// BatchNorm1d; the web projector is 2 Linear layers and has no BatchNorm (per-sample only).
// Parameter names mirror PyTorch (backbone.net.0.weight ...) so exported safetensors map 1:1
// onto jepa_studio.models.MLPTiny's backbone.
import { linearForward, linearBackward, geluForward, geluBackward } from './tensor.js';
import { gaussian, mulberry32 } from './rng.js';

export const MLP_SPEC = {
  side: 16, channels: 3,
  layers: [
    { name: 'backbone.net.0', din: 768, dout: 256, act: 'gelu' },
    { name: 'backbone.net.2', din: 256, dout: 64, act: null },   // -> embedding
    { name: 'projector.net.0', din: 64, dout: 64, act: 'gelu' },
    { name: 'projector.net.2', din: 64, dout: 16, act: null },   // -> projection
  ],
  embedLayer: 1,
};

export function inputDim(spec = MLP_SPEC) { return spec.layers[0].din; }
export function embedDim(spec = MLP_SPEC) { return spec.layers[spec.embedLayer].dout; }
export function projDim(spec = MLP_SPEC) { return spec.layers[spec.layers.length - 1].dout; }

/** PyTorch-style init: W, b ~ U(-1/sqrt(din), 1/sqrt(din)). */
export function initParams(seed, spec = MLP_SPEC, Arr = Float32Array) {
  const rnd = mulberry32(seed >>> 0);
  const params = [];
  for (const L of spec.layers) {
    const bound = 1 / Math.sqrt(L.din);
    const W = new Arr(L.dout * L.din), b = new Arr(L.dout);
    for (let i = 0; i < W.length; i++) W[i] = (rnd() * 2 - 1) * bound;
    for (let i = 0; i < b.length; i++) b[i] = (rnd() * 2 - 1) * bound;
    params.push({ name: L.name, W, b, din: L.din, dout: L.dout, act: L.act });
  }
  return params;
}

export function zeroGrads(params) {
  return params.map((p) => ({ W: new p.W.constructor(p.W.length), b: new p.b.constructor(p.b.length) }));
}

export function cloneParams(params) {
  return params.map((p) => ({ ...p, W: p.W.slice(), b: p.b.slice() }));
}

/** Forward through layers [from, to). Returns cache for backward. */
export function forward(params, X, n, from = 0, to = params.length, matmul = null) {
  const Arr = X.constructor;
  const cache = { n, from, to, inputs: [], pre: [] };
  let h = X;
  for (let li = from; li < to; li++) {
    const p = params[li];
    const y = new Arr(n * p.dout);
    if (matmul) matmul(h, n, p.din, p.W, p.b, p.dout, y); else linearForward(h, n, p.din, p.W, p.b, p.dout, y);
    cache.inputs.push(h);
    cache.pre.push(y);
    h = p.act === 'gelu' ? geluForward(y, new Arr(y.length)) : y;
  }
  cache.out = h;
  return cache;
}

/** Backward; accumulates into grads. Returns dX (input gradient) if wantInput. */
export function backward(params, cache, dOut, grads, wantInput = false) {
  let d = dOut;
  const Arr = dOut.constructor;
  for (let li = cache.to - 1; li >= cache.from; li--) {
    const k = li - cache.from;
    const p = params[li];
    if (p.act === 'gelu') d = geluBackward(d, cache.pre[k], new Arr(d.length));
    const needDX = li > cache.from || wantInput;
    const dX = needDX ? new Arr(cache.n * p.din) : null;
    linearBackward(d, cache.inputs[k], cache.n, p.din, p.W, p.dout, grads ? grads[li].W : null, grads ? grads[li].b : null, dX);
    d = dX;
  }
  return d;
}

/** Encode inputs (n, 768) -> {emb (n, 64), proj (n, 16)}. */
export function encode(params, X, n, spec = MLP_SPEC, matmul = null) {
  const c1 = forward(params, X, n, 0, spec.embedLayer + 1, matmul);
  const c2 = forward(params, c1.out, n, spec.embedLayer + 1, params.length, matmul);
  return { emb: c1.out, proj: c2.out };
}

/** Saliency: |d ||emb|| / d x|, max over channels, as a (side*side) map. Analytic backprop. */
export function saliency(params, x, spec = MLP_SPEC) {
  const X = Float64Array.from(x);
  const p64 = params.map((p) => ({ ...p, W: Float64Array.from(p.W), b: Float64Array.from(p.b) }));
  const cache = forward(p64, X, 1, 0, spec.embedLayer + 1);
  const e = cache.out;
  let nrm = 0;
  for (let i = 0; i < e.length; i++) nrm += e[i] * e[i];
  nrm = Math.sqrt(Math.max(nrm, 1e-24));
  const dE = new Float64Array(e.length);
  for (let i = 0; i < e.length; i++) dE[i] = e[i] / nrm;
  const dX = backward(p64, cache, dE, null, true);
  const s = spec.side, hw = s * s;
  const out = new Float32Array(hw);
  for (let i = 0; i < hw; i++) {
    let m = 0;
    for (let c = 0; c < spec.channels; c++) m = Math.max(m, Math.abs(dX[c * hw + i]));
    out[i] = m;
  }
  return out;
}

/** Supervised head for the baseline: logits = emb W^T + b. */
export function initHead(seed, din, classes) {
  const g = gaussian(mulberry32(seed));
  const W = new Float32Array(classes * din), b = new Float32Array(classes);
  for (let i = 0; i < W.length; i++) W[i] = g() * 0.01;
  return { name: 'head', W, b, din, dout: classes, act: null };
}

export function paramCount(params) {
  return params.reduce((s, p) => s + p.W.length + p.b.length, 0);
}
