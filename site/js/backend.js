// One interface, two tiers:
//   WebBackend      in-browser demo engine (site/js/engine/worker.js), always available
//   DesktopBackend  the local Python server (docs/api.md), used when the Tauri shell injects
//                   window.__JEPA_DESKTOP__ = {port, token}
// Methods: hardware(), preview(cfg), startRun(cfg), events(since), control(cmd), evaluate(),
//          embeddings(), neighbors(i, k), saliency(i), assistant(q), exportBundle()
// The desktop URL is built here and only here; the web build's CSP (connect-src 'self') would
// block it anyway, so a static-site visitor can never reach a localhost port by accident.

export function detectTier() {
  const d = globalThis.__JEPA_DESKTOP__;
  if (d && Number.isInteger(d.port) && d.port > 0 && d.port < 65536 && typeof d.token === 'string' && d.token.length >= 16) return 'desktop';
  return 'web';
}

// ------------------------------------------------------------------ web tier

export class WebBackend {
  constructor() {
    this.tier = 'web';
    this.worker = new Worker(new URL('./engine/worker.js', import.meta.url), { type: 'module', name: 'jepa-engine' });
    this.seq = 0;
    this.pending = new Map();
    this.log = [];            // events, same records as events.jsonl
    this.listeners = new Set();
    this.progress = new Set();
    this.ready = new Promise((res) => { this.onReady = res; });
    this.worker.onmessage = (e) => {
      const m = e.data;
      if (m.type === 'ready') { this.onReady(); return; }
      if (m.type === 'event') { this.log.push(m.event); for (const f of this.listeners) f(m.event); return; }
      if (m.type === 'progress') { for (const f of this.progress) f(m.stage); return; }
      const p = this.pending.get(m.id);
      if (!p) return;
      this.pending.delete(m.id);
      if (m.ok) p.resolve(m.result); else p.reject(new Error(m.error));
    };
    this.worker.onerror = (e) => {
      const err = new Error(e.message || 'the training worker failed to load');
      for (const p of this.pending.values()) p.reject(err);
      this.pending.clear();
      for (const f of this.listeners) f({ event: 'error', message: err.message });
    };
    this.useGpu = true;
    this.data = { kind: 'synthetic-shapes' };
  }

  call(type, args = {}, transfer = []) {
    const id = ++this.seq;
    return new Promise((resolve, reject) => {
      this.pending.set(id, { resolve, reject });
      this.worker.postMessage({ id, type, ...args }, transfer);
    });
  }

  onEvent(f) { this.listeners.add(f); return () => this.listeners.delete(f); }
  onProgress(f) { this.progress.add(f); return () => this.progress.delete(f); }

  /** Browser hardware card: cores, memory, WebGPU adapter, and a timed batch-size probe. */
  async hardware({ probe = false, cfg = null } = {}) {
    await this.ready;
    const nav = globalThis.navigator || {};
    const card = {
      tier: 'web',
      user_agent: nav.userAgent || '',
      logical_cores: nav.hardwareConcurrency ?? null,
      device_memory_gb: nav.deviceMemory ?? null,
      webgpu_api: !!nav.gpu,
      js_heap_limit_mb: globalThis.performance && performance.memory ? Math.round(performance.memory.jsHeapSizeLimit / 1048576) : null,
      cross_origin_isolated: !!globalThis.crossOriginIsolated,
    };
    const { gpu } = await this.call('init', { useGpu: this.useGpu });
    card.gpu = gpu;
    card.backend = gpu.available ? 'WebGPU' : 'CPU (JavaScript)';
    let decisions = [];
    if (probe && cfg) {
      const p = await this.call('probe', { cfg, data: this.dataForWorker(), useGpu: this.useGpu });
      card.probe = p;
      decisions.push({ name: 'batch_size', value: p.batch, why: `doubled from 32 while one step stayed under ${p.budgetMs} ms; 32 is the floor because SIGReg needs enough samples per batch (${p.trials.map((t) => `${t.batch}: ${t.ms} ms`).join(', ')})` });
    }
    decisions.push({ name: 'backend', value: card.backend, why: gpu.available ? `WebGPU kernels passed a self-test against JavaScript and were faster (${gpu.speed.gpuMs.toFixed(1)} vs ${gpu.speed.cpuMs.toFixed(1)} ms)` : `not using WebGPU: ${gpu.reason}` });
    return { card, decisions };
  }

  dataForWorker() {
    if (this.data.kind === 'images') return { kind: 'images', images: this.data.images, n: this.data.n, size: this.data.size, labels: this.data.labels, classes: this.data.classes };
    return { kind: 'synthetic-shapes' };
  }

  setData(data) { this.data = data; }

  /** Views are computed on the main thread in the web tier (see tabs/data.js); nothing to do here. */
  async preview() { return { info: { tier: 'web' } }; }

  async startRun(cfg, { batch = null } = {}) {
    await this.ready;
    this.log = [];
    return this.call('start', { cfg, data: this.dataForWorker(), useGpu: this.useGpu, batch });
  }

  async events(since = 0) { return { events: this.log.slice(since), next: this.log.length }; }
  control(command) { return this.call('control', { command }); }
  evaluate(labeledFraction) { return this.call('evaluate', { labeledFraction }); }
  inspect(n, space) { return this.call('inspect', { n, space }); }
  embeddings(n) { return this.call('embeddings', { n }); }
  neighbors(i, k = 8, space = 'embedding') { return this.call('neighbors', { i, k, space }); }
  saliency(i) { return this.call('saliency', { i }); }
  images(indices) { return this.call('images', { indices }); }
  weights() { return this.call('weights'); }
  loadWeights(tensors, cfg) { return this.call('loadWeights', { tensors, cfg, data: this.dataForWorker() }); }
  /** Web tier assistant is retrieval-only and runs on the main thread (tabs/explain.js). */
  async assistant() { throw new Error('The web assistant runs locally; use tabs/explain.js'); }
  /** Web tier bundles are assembled on the main thread (tabs/export.js). */
  async exportBundle() { return null; }
}

// ------------------------------------------------------------------ desktop tier

export class DesktopBackend {
  constructor({ port, token }) {
    this.tier = 'desktop';
    this.base = `http://127.0.0.1:${Number(port)}`;
    this.token = token;
    this.runId = null;
    this.cursor = 0;
    this.log = [];
    this.listeners = new Set();
    this.progress = new Set();
    this.polling = null;
    this.data = { kind: 'synthetic-shapes' };
  }

  async req(method, path, body) {
    const res = await fetch(this.base + path, {
      method,
      headers: { 'X-JEPA-Token': this.token, ...(body ? { 'Content-Type': 'application/json' } : {}) },
      body: body ? JSON.stringify(body) : undefined,
      cache: 'no-store',
      credentials: 'omit',
      referrerPolicy: 'no-referrer',
    });
    const j = await res.json().catch(() => ({ error: `HTTP ${res.status}` }));
    if (!res.ok) throw new Error(j.error || `HTTP ${res.status}`);
    return j;
  }

  onEvent(f) { this.listeners.add(f); return () => this.listeners.delete(f); }
  onProgress(f) { this.progress.add(f); return () => this.progress.delete(f); }
  setData(data) { this.data = data; }

  async hardware() { const r = await this.req('GET', '/api/hardware?disk=1'); return { card: { tier: 'desktop', ...r.card }, decisions: r.decisions || [] }; }
  validate(config) { return this.req('POST', '/api/config/validate', { config }); }
  preview(config, n = 1) { return this.req('POST', '/api/data/preview', { config, n }); }

  async startRun(config) {
    const { run_id: id } = await this.req('POST', '/api/runs', { config });
    this.runId = id;
    this.cursor = 0;
    this.log = [];
    this.poll();
    return { run_id: id, backend: 'desktop (PyTorch)' };
  }

  poll() {
    clearTimeout(this.polling);
    const tick = async () => {
      try {
        const { events, next } = await this.events(this.cursor);
        this.cursor = next;
        for (const ev of events) { this.log.push(ev); for (const f of this.listeners) f(ev); }
        if (events.some((e) => e.event === 'end' || e.event === 'stopped')) return;
      } catch (e) {
        for (const f of this.listeners) f({ event: 'error', message: String(e.message || e) });
        return;
      }
      this.polling = setTimeout(tick, 1000);
    };
    tick();
  }

  events(since = 0) { return this.req('GET', `/api/runs/${encodeURIComponent(this.runId)}/events?since=${Number(since)}`); }
  control(command) { return this.req('POST', `/api/runs/${encodeURIComponent(this.runId)}/control`, { command }); }
  evaluate() { return this.req('POST', `/api/runs/${encodeURIComponent(this.runId)}/evaluate`); }
  embeddings(n = 1000) { return this.req('GET', `/api/runs/${encodeURIComponent(this.runId)}/embeddings?n=${Number(n)}`); }
  async inspect(n = 1000) {
    const r = await this.embeddings(n);
    return { pca3: Float32Array.from(r.pca3.flat()), var: r.var, labels: r.labels, classes: r.classes, isotropy: r.isotropy, collapse: r.collapse, n: r.labels ? r.labels.length : r.pca3.length, thumbs: r.thumbs, space: 'embedding' };
  }
  neighbors(i, k = 8) { return this.req('GET', `/api/runs/${encodeURIComponent(this.runId)}/neighbors?i=${Number(i)}&k=${Number(k)}`); }
  assistant(question) { return this.req('POST', '/api/assistant', { question: String(question).slice(0, 2000), run_id: this.runId || undefined }); }
  exportBundle() { return this.req('POST', `/api/runs/${encodeURIComponent(this.runId)}/export`); }
  worldPlan(worldRunId, seed) { return this.req('POST', `/api/world/${encodeURIComponent(worldRunId)}/plan`, { seed: Number(seed) }); }
  /** Same shape as the web engine's {index, map (flat side*side), side}, plus `image` (PNG data URL of
   *  the sample the map was computed on) and, for vit-tiny runs, `attention` {map, side, grid}. */
  async saliency(i) {
    const r = await this.req('GET', `/api/runs/${encodeURIComponent(this.runId)}/saliency?i=${Number(i)}`);
    const out = { index: r.index, map: Float32Array.from(r.map.flat()), side: r.side, kind: r.kind, image: r.image };
    if (r.attention) out.attention = { map: Float32Array.from(r.attention.map.flat()), side: r.attention.side, grid: r.attention.grid, kind: r.attention.kind };
    return out;
  }
}

export function makeBackend() {
  return detectTier() === 'desktop' ? new DesktopBackend(globalThis.__JEPA_DESKTOP__) : new WebBackend();
}
