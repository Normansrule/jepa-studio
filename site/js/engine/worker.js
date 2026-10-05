// Web Worker hosting the demo-tier trainer, so training never blocks the page.
// Messages in:  {id, type, ...}  types: init, probe, start, control, inspect, neighbors,
//               saliency, evaluate, embeddings, weights, loadWeights, isotropy
// Messages out: {id, ok, result} replies, and {type: 'event', event} records in the same shape
//               as the desktop events.jsonl (event: start | step | paused | resumed | stopped | end | error).
import { Trainer } from './trainer.js';
import { GpuMatmul } from './gpu.js';

let trainer = null;
let gpu = null;
let gpuInfo = { available: false, reason: 'not checked' };
let gpuChecked = false;
let gpuCached = null; // the accepted GpuMatmul, kept across on/off toggles
let state = 'idle'; // idle | running | paused | stopping
let runSeq = 0;

const post = (msg, transfer) => self.postMessage(msg, transfer || []);
const emit = (event) => post({ type: 'event', event: { t: Date.now() / 1000, ...event } });

async function initGpu(want) {
  if (!want) { gpu = null; return { available: false, reason: 'turned off in the Train tab' }; }
  if (gpuChecked) { gpu = gpuCached; return gpuInfo; }
  gpuChecked = true;
  if (!self.navigator || !self.navigator.gpu) { gpuInfo = { available: false, reason: 'navigator.gpu is not available in this browser' }; return gpuInfo; }
  try {
    const g = await GpuMatmul.create(self.navigator.gpu);
    if (!g) { gpu = null; gpuInfo = { available: false, reason: 'navigator.gpu returned no adapter' }; }
    else if (g.rejected) { gpu = null; gpuInfo = { available: false, adapter: g.info, reason: g.reason, speed: g.speed }; }
    else { gpu = g; gpuInfo = { available: true, ...g.info, selfTestError: g.selfTestError, speed: g.speed }; }
    gpuCached = gpu;
  } catch (e) {
    gpu = null;
    gpuInfo = { available: false, reason: String(e && e.message || e) };
  }
  return gpuInfo;
}

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

async function run(myRun) {
  const cfg = trainer.cfg;
  const logEvery = Math.max(1, cfg.train.log_every || 1);
  let tLast = performance.now(), seenLast = trainer.seen;
  emit({ event: 'start', total: trainer.total, batch: trainer.batch, backend: gpu ? 'WebGPU' : 'CPU (JavaScript)', ...trainer.summary() });
  try {
    while (trainer.step < trainer.total && myRun === runSeq) {
      if (state === 'paused') {
        emit({ event: 'paused', step: trainer.step });
        while (state === 'paused' && myRun === runSeq) await sleep(100);
        if (state === 'running') emit({ event: 'resumed', step: trainer.step });
      }
      if (state === 'stopping' || myRun !== runSeq) break;
      const r = await trainer.trainStep();
      const s = trainer.step;
      if (s % logEvery === 0 || s === 1 || s === trainer.total) {
        const now = performance.now();
        const sps = (trainer.seen - seenLast) / Math.max(1e-6, (now - tLast) / 1000);
        tLast = now; seenLast = trainer.seen;
        const rec = {
          event: 'step', step: s, epoch: trainer.epoch, total: trainer.total, loss: r.loss, pred: r.pred, sigreg: r.sigreg,
          lr: r.lr, samples_per_s: Math.round(sps * 10) / 10,
        };
        if (self.performance && performance.memory) rec.heap_mb = Math.round(performance.memory.usedJSHeapSize / 1048576);
        const diagEvery = Math.max(logEvery * 5, Math.ceil(trainer.total / 40));
        if (s === 1 || s % diagEvery === 0 || s === trainer.total) {
          rec.proj_hist = trainer.projectionHistograms();
          const iso = trainer.isotropy(256);
          rec.isotropy = iso.report;
          rec.collapse = iso.collapse;
        }
        emit(rec);
      }
      // yield so control messages (pause/stop) are processed promptly
      if (s % 2 === 0) await sleep(0);
    }
    const stopped = state === 'stopping' || myRun !== runSeq;
    if (myRun === runSeq) {
      emit({ event: stopped ? 'stopped' : 'end', status: stopped ? 'stopped' : 'done', steps: trainer.step });
      state = 'idle';
    }
  } catch (e) {
    emit({ event: 'error', message: String(e && e.message || e), step: trainer.step });
    state = 'idle';
  }
}

const handlers = {
  async init({ useGpu = true }) { return { gpu: await initGpu(useGpu) }; },

  async probe({ cfg, data, useGpu = true, budgetMs = 150 }) {
    await initGpu(useGpu);
    const t = new Trainer(cfg, data || { kind: 'synthetic-shapes' }, gpu);
    const res = await t.probeBatch(budgetMs);
    return { ...res, backend: gpu ? 'WebGPU' : 'CPU (JavaScript)' };
  },

  async start({ cfg, data, useGpu = true, batch = null }) {
    await initGpu(useGpu);
    runSeq += 1;
    state = 'running';
    trainer = new Trainer(cfg, data || { kind: 'synthetic-shapes' }, gpu);
    if (batch) {
      trainer.batch = batch;
      const perEpoch = Math.max(1, Math.floor(trainer.ds.n / batch));
      trainer.total = cfg.train.max_steps || perEpoch * cfg.train.epochs;
    }
    run(runSeq);
    return { ...trainer.summary(), backend: gpu ? 'WebGPU' : 'CPU (JavaScript)', gpu: gpuInfo };
  },

  async control({ command }) {
    if (!trainer) return { ok: false };
    if (command === 'pause' && state === 'running') state = 'paused';
    else if (command === 'resume' && state === 'paused') state = 'running';
    else if (command === 'stop' && (state === 'running' || state === 'paused')) state = 'stopping';
    return { ok: true, state };
  },

  need() { if (!trainer) throw new Error('No model yet. Start a training run first.'); },

  async inspect({ n, space }) { handlers.need(); return trainer.inspect(n, space); },
  async isotropy() { handlers.need(); return trainer.isotropy(); },
  async neighbors({ i, k, space }) { handlers.need(); return trainer.neighbors(i, k, space); },
  async saliency({ i }) { handlers.need(); return trainer.saliencyMap(i); },
  async evaluate({ labeledFraction }) {
    handlers.need();
    return trainer.evaluate(labeledFraction, (stage) => post({ type: 'progress', stage }));
  },
  async embeddings({ n }) {
    handlers.need();
    const { emb, proj } = trainer.embedAll(n || trainer.ds.n);
    return { emb, proj, n: n || trainer.ds.n, embDim: 64, projDim: 16, labels: trainer.ds.labels ? Array.from(trainer.ds.labels.subarray(0, n || trainer.ds.n)) : null, classes: trainer.classes };
  },
  async weights() {
    handlers.need();
    return { tensors: trainer.weightsTensors(), step: trainer.step, summary: trainer.summary(), inputStats: trainer.inputStats };
  },
  async loadWeights({ tensors, cfg, data }) {
    if (!trainer) trainer = new Trainer(cfg, data || { kind: 'synthetic-shapes' }, gpu);
    trainer.loadWeights(tensors);
    return { ok: true };
  },
  async images({ indices }) {
    handlers.need();
    return indices.map((i) => ({ i, data: trainer.imageAt(i).data.slice(), size: trainer.ds.size, label: trainer.ds.labels ? trainer.ds.labels[i] : null }));
  },
};

self.onmessage = async (e) => {
  const { id, type, ...args } = e.data || {};
  const h = handlers[type];
  if (!h || type === 'need') { post({ id, ok: false, error: `unknown message ${type}` }); return; }
  try {
    const result = await h(args);
    post({ id, ok: true, result });
  } catch (err) {
    post({ id, ok: false, error: String(err && err.message || err) });
  }
};

post({ type: 'ready' });
