// Train tab: presets, start/pause/stop the run, live charts, hardware card with batch-size probe.
import { $, el, clear, gate, setStatus, collapseStatus, COLLAPSE_TEXT, metrics } from '../ui/dom.js';
import { LineChart, fmt } from '../ui/charts.js';
import { epExpectedUnderNull } from '../engine/lejepa.js';
import { field, valueOf, setValue, onValidity, invalidFor, fixMessage } from '../ui/fields.js';
import { getPref, setPref } from '../ui/prefs.js';
import { notify } from '../ui/toast.js';
import { webDefaultConfig } from '../config.js';

let app, charts, state = 'idle', total = 300, snapshots = [];
const defaults = webDefaultConfig();

// Presets set steps, batch size and SIGReg directions, and put λ and the learning rate back to
// their defaults. The desktop tier trains with PyTorch on the local machine, so its presets are
// larger. Which preset is shown is derived from the values, so any edit reads "Custom".
export const PRESETS = {
  web: {
    quick: { steps: 100, batch: '32', slices: 16, desc: '100 steps, batch 32, 16 SIGReg directions: the fastest way to watch the loss fall.' },
    balanced: { steps: 300, batch: 'auto', slices: 64, desc: '300 steps, batch size from the timing probe, 64 directions. The default.' },
    thorough: { steps: 1000, batch: '128', slices: 256, desc: '1000 steps, batch 128, 256 directions: several minutes in a browser, a healthier embedding.' },
  },
  desktop: {
    quick: { steps: 300, batch: '64', slices: 64, desc: '300 steps, batch 64, 64 SIGReg directions: a quick check of a folder.' },
    balanced: { steps: 2000, batch: 'auto', slices: 256, desc: '2000 steps, batch size from the memory probe, 256 directions.' },
    thorough: { steps: 10000, batch: 'auto', slices: 1024, desc: '10 000 steps, automatic batch size, 1024 directions: a real pretraining run.' },
  },
};
const CUSTOM_DESC = 'Your own settings: pick a preset to go back to one.';

function presets() { return PRESETS[app.tier] || PRESETS.web; }

function matchPreset() {
  const steps = valueOf('train-steps'), lam = valueOf('train-lambda');
  const lr = Number($('#train-lr').value), slices = Number($('#train-slices').value), batch = $('#train-batch').value;
  for (const [name, p] of Object.entries(presets())) {
    if (steps === p.steps && batch === p.batch && slices === p.slices && lam === defaults.objective.lambda && Math.abs(lr - defaults.train.lr) < 1e-12) return name;
  }
  return 'custom';
}

function showPreset() {
  const name = matchPreset();
  for (const r of document.querySelectorAll('input[name="preset"]')) r.checked = r.value === name;
  $('#preset-state').hidden = name !== 'custom';
  $('#preset-seg').dataset.preset = name;
  $('#preset-desc').textContent = name === 'custom' ? CUSTOM_DESC : presets()[name].desc;
  return name;
}

function applyPreset(name) {
  const p = presets()[name];
  if (!p) return;
  setValue('train-steps', p.steps);
  setValue('train-batch', p.batch);
  setValue('train-slices', p.slices);
  setValue('train-lambda', defaults.objective.lambda);
  setValue('train-lr', defaults.train.lr);
  applyControls();
}

function applyControls() {
  const c = app.config;
  const steps = valueOf('train-steps'), lam = valueOf('train-lambda');
  if (steps !== null) c.train.max_steps = steps;
  if (lam !== null) c.objective.lambda = lam;
  c.train.lr = Number($('#train-lr').value);
  c.objective.num_slices = Number($('#train-slices').value);
  const b = $('#train-batch').value;
  c.train.batch_size = b === 'auto' ? 'auto' : Number(b);
  c.model.arch = 'mlp-tiny';
  c.model.embed_dim = 64;
  c.model.proj_dim = 16;
  c.model.proj_hidden = 64;
  c.hardware.device = $('#train-gpu').checked ? 'auto' : 'wasm';
  app.backend.useGpu = $('#train-gpu').checked;
  if (state === 'idle' && steps !== null) $('#train-step').textContent = `0 / ${steps}`;
  showPreset();
  app.emit('config');
}

function syncControls() {
  const c = app.config;
  setValue('train-steps', c.train.max_steps ?? defaults.train.max_steps);
  setValue('train-lambda', c.objective.lambda);
  const lr = [...$('#train-lr').options].find((o) => Math.abs(Number(o.value) - c.train.lr) < 1e-12);
  setValue('train-lr', lr ? lr.value : String(c.train.lr));
  setValue('train-slices', String(c.objective.num_slices));
  setValue('train-batch', String(c.train.batch_size));
  showPreset();
}

/** Start is blocked while a run is active and while any run setting (Data or Train) is invalid. */
function updateStart() {
  const busy = state === 'running' || state === 'paused' || state === 'probing';
  if (busy) { gate($('#train-start'), ''); $('#train-start').disabled = true; return; }
  gate($('#train-start'), fixMessage(invalidFor('run'), 'start', 'Train'));
}

function setState(s) {
  state = s;
  const running = s === 'running' || s === 'paused';
  updateStart();
  $('#train-pause').disabled = !running;
  $('#train-stop').disabled = !running;
  $('#train-pause').textContent = s === 'paused' ? 'Resume' : 'Pause';
  $('#train-dot').dataset.on = String(running);
  const txt = { idle: 'Not started', probing: 'Timing the browser…', running: 'Training', paused: 'Paused', done: 'Finished', stopped: 'Stopped', error: 'Error' };
  $('#train-status').textContent = txt[s] || s;
}

async function start() {
  if (invalidFor('run').length) { updateStart(); return; }
  applyControls();
  $('#train-next').hidden = true;
  for (const c of Object.values(charts)) c.reset();
  snapshots = [];
  app.emit('snapshots', snapshots);
  let batch = null;
  if (app.config.train.batch_size === 'auto' && app.backend.tier === 'web') {
    setState('probing');
    try {
      const hw = await requestHardware({ probe: true, cfg: app.config });
      batch = hw.card.probe.batch;
    } catch (e) {
      setState('error');
      $('#train-status').textContent = `Probe failed: ${e.message}`;
      notify('bad', 'Timing probe failed', e.message);
      return;
    }
  }
  try {
    const summary = await app.backend.startRun(app.config, { batch });
    total = summary.total || app.config.train.max_steps;
    app.run = { started: new Date().toISOString(), summary, events: [], batch: summary.batch, backend: summary.backend, config: JSON.parse(JSON.stringify(app.config)) };
    app.setBackendLabel(summary.backend || 'desktop');
    $('#train-step').textContent = `0 / ${total}`;
    setState('running');
    app.emit('run-start', app.run);
  } catch (e) {
    setState('error');
    $('#train-status').textContent = `Could not start: ${e.message}`;
    notify('bad', 'Training could not start', e.message);
  }
}

/** Non-modal next step after a run: "Run finished — inspect the embeddings". */
function finished(ev, stopped) {
  const steps = app.run.events.filter((e) => e.event === 'step');
  const last = steps.at(-1);
  const secs = (Date.parse(app.run.finished) - Date.parse(app.run.started)) / 1000;
  const verdict = app.lastIsotropy ? `, ${COLLAPSE_TEXT[app.lastIsotropy.collapse.status] || app.lastIsotropy.collapse.status}` : '';
  const what = `${last ? last.step : 0} steps in ${Number.isFinite(secs) ? secs.toFixed(0) : '?'} s${verdict}.`;
  $('#train-next').querySelector('strong').textContent = stopped ? 'Run stopped' : 'Run finished';
  $('#train-next-text').textContent = `— ${what}`;
  $('#train-next').hidden = false;
  const reason = ev.reason ? ` Reason: ${ev.reason}.` : '';
  // the prompt above already offers the next step on this tab; elsewhere the notice carries it
  const onTrain = app.router && app.router.current() === 'train';
  notify(stopped ? 'info' : 'good', stopped ? 'Run stopped' : 'Run finished', `${what}${reason}`,
    { action: onTrain ? null : { label: 'Inspect the embeddings', run: () => app.goto('inspect') } });
}

function onEvent(ev) {
  if (!app.run) return;
  app.run.events.push(ev);
  if (ev.event === 'step') {
    total = ev.total || total;
    charts.loss.push(ev.step, { loss: ev.loss });
    charts.pred.push(ev.step, { pred: ev.pred });
    charts.sig.push(ev.step, { sig: ev.sigreg, floor: epExpectedUnderNull() });
    charts.lr.push(ev.step, { lr: ev.lr });
    if (ev.step > 1) charts.sps.push(ev.step, { sps: ev.samples_per_s });
    // performance.memory is Chromium-only and not exposed in workers, so sample the page's heap
    const heap = ev.heap_mb ?? (globalThis.performance && performance.memory ? Math.round(performance.memory.usedJSHeapSize / 1048576) : undefined);
    if (heap !== undefined) charts.heap.push(ev.step, { heap });
    $('#v-loss').textContent = fmt(ev.loss, 4);
    $('#v-pred').textContent = fmt(ev.pred, 4);
    $('#v-sig').textContent = fmt(ev.sigreg, 4);
    $('#v-lr').textContent = fmt(ev.lr, 3);
    $('#v-sps').textContent = ev.samples_per_s ? `${Math.round(ev.samples_per_s)}/s` : '–';
    $('#v-heap').textContent = heap !== undefined ? `${heap} MB` : 'n/a';
    $('#train-step').textContent = `${ev.step} / ${total}`;
    $('#train-progress').style.width = `${Math.min(100, (100 * ev.step) / total)}%`;
    if (ev.isotropy) {
      const iso = ev.isotropy;
      setStatus($('#train-collapse'), collapseStatus(ev.collapse.status), `${COLLAPSE_TEXT[ev.collapse.status]}: ${ev.collapse.message}`);
      metrics($('#train-iso'), [
        ['Effective rank', fmt(iso.effective_rank, 3), `of ${iso.dim}`],
        ['Isotropy ratio', fmt(iso.isotropy_ratio, 2), 'λmin / λmax'],
        ['SIGReg (fixed 256 imgs)', fmt(iso.sigreg, 3), '≈1.05 is Gaussian'],
        ['Top eigen share', `${Math.round(iso.top_eigen_share * 100)}%`],
      ]);
      app.lastIsotropy = { report: iso, collapse: ev.collapse, step: ev.step };
      app.emit('isotropy', app.lastIsotropy);
    }
    if (ev.proj_hist) { snapshots.push({ step: ev.step, hist: ev.proj_hist }); app.emit('snapshots', snapshots); }
    app.emit('step', ev);
  } else if (ev.event === 'paused') setState('paused');
  else if (ev.event === 'resumed') setState('running');
  else if (ev.event === 'end' || ev.event === 'stopped') {
    // the desktop backend sends `stopped` then `end` with status "stopped"; keep it "Stopped"
    setState(ev.event === 'end' && ev.status !== 'stopped' ? 'done' : 'stopped');
    if (ev.reason) $('#train-status').textContent = `Stopped: ${ev.reason}`;   // e.g. the memory guard
    app.run.finished = new Date().toISOString();
    if (!app.run.announced) { app.run.announced = true; finished(ev, state === 'stopped'); }
    app.emit('run-end', ev);
  } else if (ev.event === 'error') {
    setState('error');
    $('#train-status').textContent = `Error: ${ev.message}`;
    notify('bad', 'Training error', ev.message);
  }
}

// Hardware requests can finish out of order: the slow first card (WebGPU adapter + timing) may
// resolve after a later probe started by "Start". Only the newest request is shown, so a stale
// card never overwrites the probe result ("Picked batch").
let hwSeq = 0, hwShown = 0;
async function requestHardware(opts) {
  const id = ++hwSeq;
  const hw = await (opts ? app.backend.hardware(opts) : app.backend.hardware());
  if (id > hwShown) {
    hwShown = id;
    app.hardware = hw;
    renderHardware(hw);
  }
  return hw;
}

function hwLoaded() {
  $('#hw-loading').hidden = true;
  $('#hw-kv').hidden = false;
  $('#hardware-card').setAttribute('aria-busy', 'false');
}

export function renderHardware(hw) {
  const { card, decisions } = hw;
  hwLoaded();
  const kv = clear($('#hw-kv'));
  const row = (k, v) => kv.append(el('dt', { text: k }), el('dd', { text: v }));
  if (card.tier === 'desktop') {
    for (const [k, v] of Object.entries(card)) if (typeof v !== 'object') row(k.replace(/_/g, ' '), String(v));
  } else {
    row('Engine', card.backend);
    row('Logical cores', card.logical_cores ?? 'not reported');
    row('Device memory', card.device_memory_gb ? `≥ ${card.device_memory_gb} GB (rounded by the browser)` : 'not reported');
    row('WebGPU API', card.webgpu_api ? 'present' : 'absent');
    const ad = card.gpu && (card.gpu.available ? card.gpu : card.gpu.adapter);
    if (ad) row('WebGPU adapter', [ad.vendor, ad.architecture, ad.description].filter(Boolean).join(' / ') || 'unnamed adapter');
    if (card.gpu && card.gpu.speed) row('First layer, 192×768→256', `WebGPU ${card.gpu.speed.gpuMs.toFixed(1)} ms, JavaScript ${card.gpu.speed.cpuMs.toFixed(1)} ms`);
    if (card.gpu && !card.gpu.available && !card.gpu.speed) row('Why CPU', card.gpu.reason);
    row('JS heap limit', card.js_heap_limit_mb ? `${card.js_heap_limit_mb} MB` : 'not reported');
    if (card.probe) row('Picked batch', `${card.probe.batch} (${card.probe.trials.map((t) => `${t.batch}→${t.ms} ms`).join(', ')})`);
  }
  const dec = clear($('#hw-decisions'));
  const rec = [];
  if (card.tier === 'web') {
    rec.push('Keep this tab in the foreground: browsers throttle background tabs.');
    if (!card.webgpu_api) rec.push('Chrome or Edge 113+ (or Safari 26+) enable WebGPU; it only speeds up the first layer here.');
    if (card.logical_cores && card.logical_cores <= 4) rec.push('Few cores: close other heavy tabs while training.');
  }
  dec.append(el('ul', { class: 'list' }, [...decisions.map((d) => el('li', {}, el('b', { text: `${d.name} = ${d.value}` }), `: ${d.why}`)), ...rec.map((r) => el('li', { text: r }))]));
}

export function init(a) {
  app = a;
  if (app.tier === 'desktop') {
    $('#train-steps').max = '20000';
    $('#train-slices').append(el('option', { text: '1024', attrs: { value: '1024' } }));
    $('#train-batch').append(el('option', { text: '256', attrs: { value: '256' } }));
    $('#train-batch').options[0].textContent = 'Auto (memory probe)';
    const note = $('#tier-note');
    note.className = 'notice';
    clear(note).append(el('strong', { text: 'Desktop tier.' }), ' Training runs in the local Python backend with PyTorch; the hardware card shows the device and settings it picked.');
  }
  const tierName = () => (app.tier === 'desktop' ? 'desktop app' : 'web demo');
  field('train-steps', { path: 'train.max_steps', gate: 'run', section: 'train', def: () => defaults.train.max_steps, commit: applyControls,
    rangeMsg: (lo, hi) => `The ${tierName()} runs ${lo} to ${hi} steps.` });
  field('train-lambda', { path: 'objective.lambda', gate: 'run', section: 'train', def: () => defaults.objective.lambda, commit: applyControls });
  field('train-lr', { section: 'train', def: () => String(defaults.train.lr), commit: applyControls });
  field('train-slices', { section: 'train', def: () => String(defaults.objective.num_slices), commit: applyControls });
  field('train-batch', { section: 'train', def: () => String(defaults.train.batch_size), commit: applyControls });
  field('train-gpu', { section: 'train', def: () => true, commit: applyControls });
  onValidity(updateStart);
  for (const r of document.querySelectorAll('input[name="preset"]')) {
    r.addEventListener('change', () => { if (r.checked) { applyPreset(r.value); setPref('preset', r.value); } });
  }
  $('#train-next-go').addEventListener('click', () => app.goto('inspect'));
  $('#train-next-dismiss').addEventListener('click', () => { $('#train-next').hidden = true; $('#train-start').focus(); });
  const opt = { pad: { l: 48 } };
  charts = {
    loss: new LineChart($('#c-loss'), { ...opt, series: [{ key: 'loss', label: 'loss', color: '--ink' }] }),
    pred: new LineChart($('#c-pred'), { ...opt, series: [{ key: 'pred', label: 'prediction', color: '--flow-loss-pred' }] }),
    sig: new LineChart($('#c-sig'), { ...opt, log: true, series: [{ key: 'sig', label: 'SIGReg', color: '--flow-loss-sigreg' }, { key: 'floor', label: 'Gaussian level', color: '--ink-2', dash: true }] }),
    lr: new LineChart($('#c-lr'), { ...opt, zero: true, series: [{ key: 'lr', label: 'lr', color: '--ink-2' }] }),
    sps: new LineChart($('#c-sps'), { ...opt, zero: true, series: [{ key: 'sps', label: 'samples/s', color: '--flow-encoder' }] }),
    heap: new LineChart($('#c-heap'), { ...opt, series: [{ key: 'heap', label: 'MB', color: '--flow-view' }], empty: globalThis.performance && performance.memory ? 'No data yet' : 'Not reported by this browser' }),
  };
  $('#train-start').addEventListener('click', start);
  $('#train-pause').addEventListener('click', () => app.backend.control(state === 'paused' ? 'resume' : 'pause'));
  $('#train-stop').addEventListener('click', () => app.backend.control('stop'));
  $('#hw-probe').addEventListener('click', async () => {
    $('#hw-probe').disabled = true;
    $('#hw-probe').textContent = 'Timing…';
    try { applyControls(); await requestHardware({ probe: true, cfg: app.config }); } catch (e) { $('#hw-decisions').textContent = `Probe failed: ${e.message}`; notify('bad', 'Timing probe failed', e.message); }
    $('#hw-probe').disabled = false;
    $('#hw-probe').textContent = 'Run timing probe';
  });
  app.backend.onEvent(onEvent);
  // a loaded config only moves the controls; applyControls() would overwrite its model block
  app.on('config-loaded', () => { syncControls(); const st = valueOf('train-steps'); if (state === 'idle' && st !== null) $('#train-step').textContent = `0 / ${st}`; });
  syncControls();
  // the preset this viewer picked last time (a remembered "Custom" keeps the defaults)
  const saved = getPref('preset');
  if (saved && saved !== 'custom' && presets()[saved]) applyPreset(saved);
  applyControls();
  setState('idle');
  requestHardware().then((hw) => { app.setBackendLabel(hw.card.backend || 'desktop'); }).catch((e) => {
    app.setBackendLabel('unavailable');
    hwLoaded();
    $('#hw-decisions').textContent = `The hardware card could not be read: ${e.message}`;
    notify('bad', 'Hardware card unavailable', e.message);
  });
}

export function show() {}
