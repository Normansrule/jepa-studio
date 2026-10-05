// Demo-tier LeJEPA trainer: data + views + mlp-tiny + loss + AdamW. No DOM, no worker APIs,
// so the same class runs inside site/js/engine/worker.js and in Node tests.
import { makeDataset, CLASSES } from '../shapes.js';
import { mulberry32, mixSeed, permutation } from './rng.js';
import { makeImage, makeViews, toSide, flipH, colorJitter } from './views.js';
import { MLP_SPEC, initParams, zeroGrads, forward, backward, encode, saliency, initHead, paramCount } from './mlp.js';
import { lejepaLoss, stepDirections, randomDirections, epConstants } from './lejepa.js';
import { geluForward, geluBackward, linearForward, linearBackward } from './tensor.js';
import { AdamW, lrAt } from './adamw.js';
import {
  isotropyReport, collapseCheck, pca, knnAccuracy, linearProbe, splitIndices, gatherRows, nearestNeighbors,
} from './evaluate.js';

const SIDE = MLP_SPEC.side;
const IN = MLP_SPEC.layers[0].din;
const EMB = MLP_SPEC.layers[MLP_SPEC.embedLayer].dout;
const PROJ = MLP_SPEC.layers[MLP_SPEC.layers.length - 1].dout;

export const WEB_TIER = {
  arch: 'mlp-tiny',
  gelu: 'exact (erf)',
  projector: 'Linear(64,64) + GELU + Linear(64,16), no BatchNorm',
  desktopProjector: 'Linear-BN-GELU x2 + Linear, with BatchNorm',
};

export class Trainer {
  /**
   * cfg: shared run config. data: {kind: 'synthetic-shapes'} or
   * {kind: 'images', images: Float32Array (n*3*S*S CHW), n, size, labels: Int32Array|null, classes: string[]|null}
   */
  constructor(cfg, data = { kind: 'synthetic-shapes' }, gpu = null) {
    this.cfg = cfg;
    this.gpu = gpu;
    this.seed = cfg.seed >>> 0;
    if (data.kind === 'images') {
      this.ds = { images: data.images, labels: data.labels, n: data.n, size: data.size, channels: 3 };
      this.classes = data.classes || null;
    } else {
      const n = Math.min(cfg.data.max_items ?? 2048, 4096);
      this.ds = makeDataset(n, this.seed, 32);
      this.classes = CLASSES;
    }
    this.ep = epConstants(cfg.objective.t_max, cfg.objective.knots);
    this.params = initParams(mixSeed(this.seed, 1));
    this.opt = new AdamW(this.params, { lr: cfg.train.lr, weightDecay: cfg.train.weight_decay ?? 0.05 });
    this.step = 0;
    this.rng = mulberry32(mixSeed(this.seed, 2));
    this.batch = typeof cfg.train.batch_size === 'number' ? cfg.train.batch_size : 32;
    const perEpoch = Math.max(1, Math.floor(this.ds.n / this.batch));
    this.total = cfg.train.max_steps || perEpoch * cfg.train.epochs;
    this.seen = 0;
    this.epoch = 0;
    this.order = permutation(this.ds.n, this.rng);
    this.cursor = 0;
    // fixed probe batch + 3 fixed directions for the "projections turn Gaussian" animation
    this.probeIdx = Array.from({ length: Math.min(256, this.ds.n) }, (_, i) => i);
    this.probeDirs = randomDirections(PROJ, 3, 1234);
    this.evalN = Math.min(this.ds.n, 1024);
    this.inputStats = this.computeInputStats();
  }

  /** Per-channel mean/std of the (16x16) inputs. Inputs are standardized with these before the
   *  first layer; weightsTensors() folds them back into backbone.net.0 so exported weights take
   *  raw [0, 1] pixels like jepa_studio.models.MLPTiny. */
  computeInputStats() {
    const n = Math.min(this.ds.n, 512), hw = SIDE * SIDE;
    const mean = [0, 0, 0], sq = [0, 0, 0];
    for (let i = 0; i < n; i++) {
      const x = toSide(this.imageAt(i), SIDE);
      for (let c = 0; c < 3; c++) for (let j = 0; j < hw; j++) { const v = x[c * hw + j]; mean[c] += v; sq[c] += v * v; }
    }
    const cnt = n * hw;
    const std = mean.map((m, c) => Math.sqrt(Math.max(sq[c] / cnt - (m / cnt) ** 2, 1e-6)));
    return { mean: mean.map((m) => m / cnt), std };
  }

  normalize(x) {
    const hw = SIDE * SIDE, { mean, std } = this.inputStats;
    for (let c = 0; c < 3; c++) { const m = mean[c], s = 1 / std[c]; for (let j = 0; j < hw; j++) x[c * hw + j] = (x[c * hw + j] - m) * s; }
    return x;
  }

  /** Model input for one image view: resample to 16x16, then standardize. */
  input(img) { return this.normalize(Float32Array.from(toSide(img, SIDE))); }

  imageAt(i) {
    const S = this.ds.size, per = 3 * S * S;
    return makeImage(this.ds.images.subarray(i * per, (i + 1) * per), 3, S, S);
  }

  /** Un-augmented input rows (n, 768) for dataset indices. */
  plainInputs(idx) {
    const X = new Float32Array(idx.length * IN);
    idx.forEach((i, r) => X.set(this.input(this.imageAt(i)), r * IN));
    return X;
  }

  nextIndices(n) {
    const out = [];
    for (let k = 0; k < n; k++) {
      if (this.cursor >= this.order.length) {
        this.order = permutation(this.ds.n, this.rng);
        this.cursor = 0;
        this.epoch += 1;
      }
      out.push(this.order[this.cursor++]);
    }
    return out;
  }

  /** Views for a batch, laid out (V, N, 768), global views first. */
  batchInputs(idx) {
    const d = this.cfg.data;
    const Vg = d.n_global, V = d.n_global + d.n_local, N = idx.length;
    const X = new Float32Array(V * N * IN);
    idx.forEach((i, n) => {
      const { globals, locals } = makeViews(this.imageAt(i), d, this.rng);
      [...globals, ...locals].forEach((v, vi) => X.set(this.input(v), (vi * N + n) * IN));
    });
    return { X, V, Vg, N };
  }

  async firstLayer(X, n) {
    const L = this.params[0];
    if (this.gpu) return this.gpu.linear(X, n, L.din, L.W, L.b, L.dout);
    return linearForward(X, n, L.din, L.W, L.b, L.dout, new Float32Array(n * L.dout));
  }

  /** One optimisation step (apply=false: forward/backward only, used by the batch-size probe). */
  async trainStep(N = this.batch, apply = true) {
    const idx = this.nextIndices(N);
    const { X, V, Vg } = this.batchInputs(idx);
    const rows = V * N;
    const pre0 = await this.firstLayer(X, rows);
    const h0 = geluForward(pre0, new Float32Array(pre0.length));
    const cache = forward(this.params, h0, rows, 1, this.params.length);
    const o = this.cfg.objective;
    const dirs = stepDirections(V, PROJ, o.num_slices, this.seed, this.step);
    const out = lejepaLoss(cache.out, V, Vg, N, PROJ, dirs, o.num_slices, o.lambda, { ep: this.ep, Arr: Float32Array });
    const grads = zeroGrads(this.params);
    const dH0 = backward(this.params, cache, out.dZ, grads, true);
    const dPre0 = geluBackward(dH0, pre0, new Float32Array(pre0.length));
    const L = this.params[0];
    if (this.gpu) {
      await this.gpu.gradW(dPre0, X, rows, L.din, L.dout, grads[0].W);
      for (let r = 0; r < rows; r++) for (let j = 0; j < L.dout; j++) grads[0].b[j] += dPre0[r * L.dout + j];
    } else {
      linearBackward(dPre0, X, rows, L.din, L.W, L.dout, grads[0].W, grads[0].b, null);
    }
    if (!Number.isFinite(out.loss)) throw new Error(`loss became ${out.loss} at step ${this.step + 1}`);
    let lr = null;
    if (apply) {
      const t = this.cfg.train;
      lr = lrAt(this.step, this.total, t.lr, t.warmup_frac ?? 0.1, t.final_lr_ratio ?? 1e-3);
      this.opt.lr = lr;
      this.opt.step(grads);
      this.step += 1;
      this.seen += N;
    }
    return { loss: out.loss, pred: out.pred, sigreg: out.sigreg, lr };
  }

  /** Double the batch while one step takes < budgetMs (the browser's auto batch size).
   *  Starts at 32: below that the Epps-Pulley statistic of each slice is too noisy to be a
   *  useful training signal, so a slow machine gets 32 and simply trains slower. */
  async probeBatch(budgetMs = 150, lo = 32, hi = 256) {
    const now = () => (globalThis.performance ? performance.now() : Date.now());
    const trials = [];
    let b = lo, pick = lo;
    await this.trainStep(lo, false); // warm up JIT
    while (b <= hi) {
      const t0 = now();
      await this.trainStep(b, false);
      const ms = now() - t0;
      trials.push({ batch: b, ms: Math.round(ms) });
      if (ms >= budgetMs) break;
      pick = b;
      b *= 2;
    }
    this.cursor = 0;
    this.batch = pick;
    const perEpoch = Math.max(1, Math.floor(this.ds.n / this.batch));
    this.total = this.cfg.train.max_steps || perEpoch * this.cfg.train.epochs;
    return { batch: pick, trials, budgetMs };
  }

  /** Projector-output histograms along 3 fixed directions (24 bins over [-4, 4], density). */
  projectionHistograms(bins = 24) {
    const X = this.plainInputs(this.probeIdx);
    const { proj } = encode(this.params, X, this.probeIdx.length);
    const n = this.probeIdx.length, w = 8 / bins;
    const out = [];
    for (let j = 0; j < 3; j++) {
      const h = new Array(bins).fill(0);
      for (let r = 0; r < n; r++) {
        let s = 0;
        for (let k = 0; k < PROJ; k++) s += proj[r * PROJ + k] * this.probeDirs[k * 3 + j];
        const c = Math.max(-4, Math.min(4, s));
        h[Math.min(bins - 1, Math.floor((c + 4) / w))] += 1;
      }
      out.push(h.map((x) => Math.round((x / n / w) * 1e4) / 1e4));
    }
    return out;
  }

  /** Backbone embeddings and projections of the first `n` un-augmented items. */
  embedAll(n = this.ds.n, params = this.params) {
    const idx = Array.from({ length: n }, (_, i) => i);
    const emb = new Float32Array(n * EMB), proj = new Float32Array(n * PROJ);
    for (let s = 0; s < n; s += 256) {
      const part = idx.slice(s, s + 256);
      const r = encode(params, this.plainInputs(part), part.length);
      emb.set(r.emb, s * EMB);
      proj.set(r.proj, s * PROJ);
    }
    return { emb, proj, n };
  }

  isotropy(n = Math.min(512, this.ds.n)) {
    const { proj } = this.embedAll(n);
    const rep = isotropyReport(proj, n, PROJ, this.cfg.objective.num_slices, 0);
    return { report: rep, collapse: collapseCheck(rep) };
  }

  /** Everything the Inspect tab needs (like GET /api/runs/<id>/embeddings). */
  inspect(n = this.evalN, space = 'embedding') {
    const { emb, proj } = this.embedAll(n);
    const Z = space === 'projection' ? proj : emb;
    const d = space === 'projection' ? PROJ : EMB;
    const p = pca(Z, n, d, 3);
    const iso = isotropyReport(proj, n, PROJ, this.cfg.objective.num_slices, 0);
    return {
      pca3: p.coords, var: p.varRatio, labels: this.ds.labels ? Array.from(this.ds.labels.subarray(0, n)) : null,
      classes: this.classes, isotropy: iso, collapse: collapseCheck(iso), n, space, dim: d,
      embIsotropy: isotropyReport(emb, n, EMB, 64, 0),
    };
  }

  neighbors(i, k = 8, space = 'embedding') {
    const n = this.evalN;
    const { emb, proj } = this.embedAll(n);
    const Z = space === 'projection' ? proj : emb;
    return { query: i, neighbors: nearestNeighbors(Z, n, space === 'projection' ? PROJ : EMB, i, k) };
  }

  saliencyMap(i) {
    const m = saliency(this.params, this.input(this.imageAt(i)));
    // chain rule through the standardization: d/dx_raw = d/dx_norm / std_c (max over channels
    // was already taken, so use the smallest std as a conservative common factor)
    const f = 1 / Math.min(...this.inputStats.std);
    for (let j = 0; j < m.length; j++) m[j] *= f;
    return { index: i, map: m, side: SIDE };
  }

  /** Train the same MLP backbone + linear head with cross-entropy on the labeled subset. */
  trainSupervised(labIdx, epochs = 30, batch = 64, lr = 1e-3) {
    const rng = mulberry32(mixSeed(this.seed, 77));
    const params = initParams(mixSeed(this.seed, 999)).slice(0, MLP_SPEC.embedLayer + 1);
    let c = 0;
    for (const i of labIdx) c = Math.max(c, this.ds.labels[i] + 1);
    const head = initHead(mixSeed(this.seed, 5), EMB, Math.max(c, this.classes ? this.classes.length : c));
    const all = [...params, head];
    const opt = new AdamW(all, { lr, weightDecay: 0.05 });
    for (let e = 0; e < epochs; e++) {
      const perm = permutation(labIdx.length, rng);
      for (let s = 0; s < labIdx.length; s += batch) {
        const part = Array.from(perm.subarray(s, s + batch), (p) => labIdx[p]);
        if (part.length < 2) continue;
        const X = new Float32Array(part.length * IN);
        part.forEach((i, r) => {
          let v = this.imageAt(i);
          if (rng() < 0.5) v = flipH(v);
          v = colorJitter(v, 0.2, rng);
          X.set(this.input(v), r * IN);
        });
        const cache = forward(all, X, part.length);
        const logits = cache.out, C = head.dout, dL = new Float32Array(logits.length);
        for (let r = 0; r < part.length; r++) {
          let mx = -Infinity;
          for (let j = 0; j < C; j++) mx = Math.max(mx, logits[r * C + j]);
          let z = 0;
          for (let j = 0; j < C; j++) z += Math.exp(logits[r * C + j] - mx);
          for (let j = 0; j < C; j++) {
            const p = Math.exp(logits[r * C + j] - mx) / z;
            dL[r * C + j] = (p - (j === this.ds.labels[part[r]] ? 1 : 0)) / part.length;
          }
        }
        const grads = zeroGrads(all);
        backward(all, cache, dL, grads);
        opt.step(grads);
      }
    }
    // pad with an untouched projector so embedAll() works on the same code path
    return [...params, ...this.params.slice(MLP_SPEC.embedLayer + 1)];
  }

  /** Linear probe + k-NN for this run and baselines (like POST /api/runs/<id>/evaluate). */
  evaluate(labeledFraction = this.cfg.data.labeled_fraction ?? 0.1, onProgress = () => {}) {
    if (!this.ds.labels) throw new Error('This dataset has no labels, so there is nothing to score against.');
    const ev = this.cfg.eval || {};
    const n = this.ds.n;
    const { labeled, test } = splitIndices(n, labeledFraction, this.seed);
    const lab = Array.from(labeled), tst = Array.from(test);
    const ytr = lab.map((i) => this.ds.labels[i]), yte = tst.map((i) => this.ds.labels[i]);
    const out = { n_labeled: lab.length, n_test: tst.length, methods: {} };
    const score = (name, params) => {
      onProgress(name);
      const { emb, proj } = this.embedAll(n, params);
      const xtr = gatherRows(emb, EMB, lab), xte = gatherRows(emb, EMB, tst);
      const lp = linearProbe(xtr, ytr, lab.length, xte, yte, tst.length, EMB, { epochs: ev.probe_epochs ?? 100, seed: this.seed });
      const kn = knnAccuracy(xtr, ytr, lab.length, xte, yte, tst.length, EMB, ev.knn_k ?? 20);
      const iso = isotropyReport(proj.subarray(0, Math.min(n, 1024) * PROJ), Math.min(n, 1024), PROJ, 256, 0);
      out.methods[name] = { linear_probe: lp.accuracy, knn: kn.accuracy, effective_rank: iso.effective_rank, sigreg: iso.sigreg, collapse: collapseCheck(iso).status };
    };
    score('LeJEPA (this run)', this.params);
    const baselines = ev.baselines || ['random-init', 'supervised'];
    if (baselines.includes('random-init')) score('random-init encoder', initParams(mixSeed(this.seed, 999)));
    if (baselines.includes('supervised')) {
      onProgress('training supervised baseline');
      score(`supervised (${lab.length} labels)`, this.trainSupervised(lab));
    }
    const nc = this.classes ? this.classes.length : Math.max(...this.ds.labels) + 1;
    out.chance = 1 / Math.max(1, nc);
    out.labeled_fraction = labeledFraction;
    return out;
  }

  /** Exported tensors, with the input standardization folded into the first layer:
   *  W'[o, c*hw+j] = W[o, c*hw+j] / std_c,  b'[o] = b[o] - sum_{c,j} W[o, c*hw+j] mean_c / std_c. */
  weightsTensors() {
    const t = [];
    const hw = SIDE * SIDE, { mean, std } = this.inputStats;
    this.params.forEach((p, li) => {
      let W = p.W, b = p.b;
      if (li === 0) {
        W = new Float32Array(p.W.length); b = Float32Array.from(p.b);
        for (let o = 0; o < p.dout; o++) for (let i = 0; i < p.din; i++) {
          const c = Math.floor(i / hw);
          W[o * p.din + i] = p.W[o * p.din + i] / std[c];
          b[o] -= p.W[o * p.din + i] * mean[c] / std[c];
        }
      }
      t.push({ name: `${p.name}.weight`, data: W, shape: [p.dout, p.din] });
      t.push({ name: `${p.name}.bias`, data: b, shape: [p.dout] });
    });
    return t;
  }

  /** Load raw-pixel weights (our export or a desktop MLPTiny backbone); un-folds the input stats. */
  loadWeights(tensors) {
    for (const p of this.params) {
      const W = tensors[`${p.name}.weight`], b = tensors[`${p.name}.bias`];
      if (!W || !b || W.data.length !== p.W.length || b.data.length !== p.b.length) throw new Error(`weights file does not match mlp-tiny (${p.name})`);
    }
    const hw = SIDE * SIDE, { mean, std } = this.inputStats;
    this.params.forEach((p, li) => {
      const W = tensors[`${p.name}.weight`].data, b = tensors[`${p.name}.bias`].data;
      if (li !== 0) { p.W.set(W); p.b.set(b); return; }
      for (let o = 0; o < p.dout; o++) {
        let bb = b[o];
        for (let i = 0; i < p.din; i++) {
          const c = Math.floor(i / hw);
          p.W[o * p.din + i] = W[o * p.din + i] * std[c];
          bb += W[o * p.din + i] * mean[c];
        }
        p.b[o] = bb;
      }
    });
  }

  summary() {
    return { params: paramCount(this.params), n: this.ds.n, batch: this.batch, total: this.total, embed_dim: EMB, proj_dim: PROJ };
  }
}
