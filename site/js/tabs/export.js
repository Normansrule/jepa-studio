// Export tab: embeddings (CSV / .npy), weights (.safetensors), run report (HTML / JSON),
// reproducibility bundle (.zip), and the run-config editor: view the config as JSON, edit,
// Validate (schema errors with their paths, inline), Apply, Revert, Download and Open, with a
// confirmation before an opened file replaces unapplied edits.
import { $, el, clear, download, gate } from '../ui/dom.js';
import { notify } from '../ui/toast.js';
import { npyBytes } from '../io/npy.js';
import { safetensorsBytes, parseSafetensors } from '../io/safetensors.js';
import { zipBytes } from '../io/zip.js';
import { matrixToCsv } from '../io/csv.js';
import { validate, loadConfigText, deepClone } from '../config.js';
import { WEB_TIER } from '../engine/trainer.js';

let app;

function status(t) { $('#export-status').textContent = t; }

function needRun() {
  if (!app.run) { status('Train a model first (Train tab); there is nothing to export yet.'); return false; }
  return true;
}

/** Exports that need a model are disabled, with the reason shown, until a run or weights exist. */
function updateGates() {
  const why = app.run ? '' : 'Available after a training run, or after opening saved weights.';
  gate($('#exp-csv'), why, 'exp-emb-why');
  gate($('#exp-npy'), why, 'exp-emb-why');
  gate($('#exp-weights'), why);
  gate($('#exp-bundle'), why);
}

function saved(title, text = '') { status(`${title}${text ? ` ${text}` : ''}`); notify('good', title, text); app.emit('exported'); }

function stamp() { return new Date().toISOString().replace(/[:.]/g, '-').slice(0, 19); }
function runName() { return (app.config.name || 'run').replace(/[^\w.-]+/g, '-').slice(0, 60); }

async function embeddings() {
  const r = await app.backend.embeddings(app.data.n);
  return r;
}

/** The run report: everything needed to understand and reproduce the run. */
export function buildReport(appRef) {
  const a = appRef;
  const steps = (a.run?.events || []).filter((e) => e.event === 'step');
  const last = steps.at(-1) || null;
  const first = steps[0] || null;
  const sps = steps.filter((e) => e.step > 1 && e.samples_per_s).map((e) => e.samples_per_s);
  const nav = globalThis.navigator || {};
  return {
    format: 'jepa-studio-report/v1',
    tool: 'jepa-studio web',
    code_version: a.version,
    tier: a.tier,
    created: new Date().toISOString(),
    run: a.run ? {
      started: a.run.started, finished: a.run.finished || null, backend: a.run.backend, batch_size: a.run.batch,
      steps_done: last ? last.step : 0, total_steps: a.run.summary?.total ?? null, params: a.run.summary?.params ?? null,
    } : null,
    seed: a.config.seed,
    config: a.run ? a.run.config : a.config,
    web_model: { ...WEB_TIER, input: '3x16x16 (views resampled to 16x16), per-channel standardized' },
    data: { kind: a.data.kind, n: a.data.n, classes: a.data.kind === 'images' ? a.data.classes : 'shapes (10)', note: a.data.kind === 'images' ? 'local files; names not included' : 'generated from seed' },
    metrics: last ? {
      first_loss: first.loss, final_loss: last.loss, final_pred: last.pred, final_sigreg: last.sigreg,
      mean_samples_per_s: sps.length ? sps.reduce((s, x) => s + x, 0) / sps.length : null,
      isotropy: a.lastIsotropy ? a.lastIsotropy.report : null, collapse: a.lastIsotropy ? a.lastIsotropy.collapse : null,
    } : null,
    evaluation: a.evaluation || null,
    world: a.worldResult || null,
    hardware: a.hardware ? a.hardware.card : null,
    browser: { user_agent: nav.userAgent || '', language: nav.language || '', platform: nav.platform || '' },
    history: steps.map((e) => ({ step: e.step, loss: e.loss, pred: e.pred, sigreg: e.sigreg, lr: e.lr, samples_per_s: e.samples_per_s })),
  };
}

/** Escape text for HTML element content and double-quoted attributes. */
export function esc(v) {
  return String(v ?? '').replace(/[&<>"']/g, (c) => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c]));
}

const REPORT_CSS = 'body{font:15px/1.5 system-ui,sans-serif;max-width:860px;margin:32px auto;padding:0 16px;color:#172033;background:#fff}h1,h2{font-family:Georgia,serif;font-weight:400}table{border-collapse:collapse;width:100%}td,th{border-bottom:1px solid #d9dde5;padding:4px 8px;text-align:left}pre{background:#eef0f4;padding:12px;overflow:auto;font-size:12px}svg{width:100%;height:auto}@media (prefers-color-scheme:dark){body{background:#121826;color:#e8ecf4}pre{background:#232e45}td,th{border-color:#2e3a54}}';

/** Standalone HTML report: no scripts, every value escaped, its own strict CSP. Built as a
 *  string (never parsed in this page), so nothing in it touches the app's DOM or CSP. */
export function reportHtml(rep) {
  const rows = [];
  const row = (k, v) => rows.push(`<tr><th>${esc(k)}</th><td>${esc(v)}</td></tr>`);
  if (rep.run) {
    row('Backend', rep.run.backend); row('Steps', `${rep.run.steps_done} / ${rep.run.total_steps}`);
    row('Batch size', rep.run.batch_size); row('Parameters', rep.run.params);
    row('Started', rep.run.started); row('Finished', rep.run.finished || '(not finished)');
  }
  if (rep.metrics) {
    row('Loss', `${rep.metrics.first_loss?.toFixed(4)} → ${rep.metrics.final_loss?.toFixed(4)}`);
    row('Prediction / SIGReg (final)', `${rep.metrics.final_pred?.toFixed(4)} / ${rep.metrics.final_sigreg?.toFixed(4)}`);
    row('Throughput', rep.metrics.mean_samples_per_s ? `${rep.metrics.mean_samples_per_s.toFixed(1)} samples/s` : 'n/a');
    if (rep.metrics.collapse) row('Collapse check', `${rep.metrics.collapse.status}: ${rep.metrics.collapse.message}`);
  }
  let chart = '';
  if (rep.history.length > 1) {
    const w = 800, h = 200, xs = rep.history.map((r) => r.step), ys = rep.history.map((r) => r.loss);
    const lo = Math.min(...ys), hi = Math.max(...ys), x0 = xs[0], x1 = xs.at(-1) || 1;
    const pts = rep.history.map((r) => `${(40 + ((r.step - x0) / Math.max(1, x1 - x0)) * (w - 50)).toFixed(1)},${(h - 20 - ((r.loss - lo) / Math.max(1e-9, hi - lo)) * (h - 30)).toFixed(1)}`).join(' ');
    chart = `<h2>Loss</h2><svg viewBox="0 0 ${w} ${h}" role="img" aria-label="${esc(`Total loss, between ${lo.toFixed(3)} and ${hi.toFixed(3)}`)}"><polyline points="${pts}" fill="none" stroke="currentColor" stroke-width="2"/></svg>`;
  }
  let evalHtml = '';
  if (rep.evaluation) {
    const tr = Object.entries(rep.evaluation.methods).map(([n, m]) => `<tr>${[n, `${(m.linear_probe * 100).toFixed(1)}%`, `${(m.knn * 100).toFixed(1)}%`, m.effective_rank, m.collapse].map((v) => `<td>${esc(v)}</td>`).join('')}</tr>`).join('');
    evalHtml = `<h2>Evaluation</h2><table><tr><th>Encoder</th><th>Linear probe</th><th>k-NN</th><th>Effective rank</th><th>Collapse</th></tr>${tr}</table><p>${esc(`${rep.evaluation.n_labeled} labeled, ${rep.evaluation.n_test} held out; chance ${(rep.evaluation.chance * 100).toFixed(0)}%.`)}</p>`;
  }
  return `<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>${esc(`jepa-studio report: ${rep.config.name}`)}</title>
<style>${REPORT_CSS}</style></head><body>
<h1>${esc(`Run report: ${rep.config.name}`)}</h1>
<p>${esc(`Created ${rep.created} by jepa-studio ${rep.code_version} (${rep.tier} tier). Seed ${rep.seed}.`)}</p>
<h2>Summary</h2><table>${rows.join('')}</table>
${chart}
${evalHtml}
<h2>Hardware</h2><pre>${esc(JSON.stringify(rep.hardware, null, 2))}</pre>
<h2>Config</h2><pre>${esc(JSON.stringify(rep.config, null, 2))}</pre>
<h2>Reproduce on desktop</h2><pre>pip install jepa-studio
jepa-studio train --config config.json</pre>
<p>The web demo trains mlp-tiny without BatchNorm in the projector; the desktop run of the same config uses the desktop projector, so numbers are comparable in trend, not identical.</p>
</body></html>
`;
}

function readme(rep) {
  return [
    `jepa-studio reproducibility bundle: ${rep.config.name}`,
    `created ${rep.created} by jepa-studio ${rep.code_version} (web demo tier)`,
    '',
    'Files',
    '  config.json          the run config (schema/config.schema.json, version 1)',
    '  report.json          metrics, evaluation, hardware card, seed, timestamps, browser',
    '  weights.safetensors  mlp-tiny weights (F32). backbone.net.* loads into jepa_studio.models.MLPTiny;',
    '                       input normalization is folded into backbone.net.0, inputs are raw [0,1] pixels',
    '                       resized to 16x16. projector.net.* is the 2-layer web projector (no BatchNorm).',
    '  embeddings.npy       float32 (N, 64) backbone embeddings of the dataset in order',
    '',
    'Reproduce on desktop',
    '  pip install jepa-studio',
    '  jepa-studio train --config config.json',
    '',
    `Seed: ${rep.seed}. Dataset: ${rep.data.kind}${rep.data.kind === 'synthetic-shapes' ? ' (regenerated from the seed, nothing to download)' : ' (your own files, not included)'}.`,
    'The desktop app runs the same config with its own projector (with BatchNorm) and PyTorch kernels,',
    'so losses follow the same trend but are not bit-identical to the browser run.',
    '',
    'Load the weights in Python:',
    '  from safetensors.torch import load_file',
    '  sd = load_file("weights.safetensors")',
    '',
  ].join('\n');
}

async function weightsBytes() {
  const w = await app.backend.weights();
  return safetensorsBytes(w.tensors, {
    format: 'jepa-studio-web/v1', arch: 'mlp-tiny', step: w.step, seed: app.config.seed,
    input: '3x16x16 raw [0,1] pixels (normalization folded into backbone.net.0)',
    projector: 'Linear(64,64)+GELU(erf)+Linear(64,16), no BatchNorm', code_version: app.version,
  });
}

async function wrap(btn, f) {
  const old = btn.textContent;
  btn.disabled = true;
  btn.textContent = 'Preparing…';
  try { await f(); } catch (e) { status(`Export failed: ${e.message}`); notify('bad', 'Export failed', e.message); }
  btn.disabled = false;
  btn.textContent = old;
  updateGates();
}

// ------------------------------------------------------------------ config editor
const editor = { base: '', stale: false, pending: null };

function cfgText() { return JSON.stringify(app.config, null, 2); }
function isDirty() { return $('#cfg-text').value !== editor.base; }

function renderState() {
  const st = $('#cfg-state');
  const dirty = isDirty();
  st.dataset.state = dirty ? 'dirty' : 'clean';
  st.textContent = !dirty ? 'Matches the Data and Train tabs' : editor.stale ? 'Unapplied edits (the app changed since you started)' : 'Unapplied edits';
  gate($('#cfg-save'), dirty ? 'Apply or revert your edits to download the config.' : '');
  $('#cfg-revert').disabled = !dirty;
}

/** Show the app's config, unless the viewer is in the middle of an edit. */
function showConfig({ force = false } = {}) {
  if (!force && isDirty()) { editor.stale = cfgText() !== editor.base; renderState(); return; }
  const t = cfgText();
  $('#cfg-text').value = t;
  editor.base = t;
  editor.stale = false;
  setInvalid(false);
  renderState();
}

function setInvalid(on) { if (on) $('#cfg-text').setAttribute('aria-invalid', 'true'); else $('#cfg-text').removeAttribute('aria-invalid'); }

/** Character range of the key a validation error points at ("$.train.lr: …"), or null. */
export function locate(text, message) {
  const m = /^(\$[^:\s]*):\s*(.*)$/.exec(message);
  if (!m) return null;
  const keys = m[1].replace(/^\$\.?/, '').split(/\.|\[\d+\]/).filter(Boolean);
  const named = /(?:unknown|missing required) key '([^']+)'/.exec(m[2]);
  if (named && /^missing/.test(m[2])) return null;
  if (named) keys.push(named[1]);
  if (!keys.length) return null;
  let i = 0;
  for (const k of keys) {
    const at = text.indexOf(`"${k}"`, i);
    if (at < 0) return null;
    i = at;
  }
  return [i, i + keys.at(-1).length + 2];
}

function select(range) {
  const ta = $('#cfg-text');
  ta.focus();
  ta.setSelectionRange(range[0], range[1]);
  const line = ta.value.slice(0, range[0]).split('\n').length - 1;
  const lh = parseFloat(getComputedStyle(ta).lineHeight) || 18;
  ta.scrollTop = Math.max(0, line * lh - ta.clientHeight / 3);
}

/** JSON.parse error -> "line 3, column 5: …" plus the offset to select. */
function syntaxProblem(e, text) {
  const pos = /position (\d+)/.exec(e.message);
  if (!pos) return { text: e.message, at: null };
  const at = Number(pos[1]);
  const before = text.slice(0, at).split('\n');
  const msg = e.message.replace(/\s*\(line \d+ column \d+\)/, '').replace(/ in JSON at position \d+/, '');
  return { text: `line ${before.length}, column ${before.at(-1).length + 1}: ${msg}`, at };
}

/** Parse + merge onto the defaults + validate, exactly like Open and the desktop loader. */
function checkText(text) {
  if (!app.schema) return { config: null, errors: ['the schema (schema/config.schema.json) could not be loaded, so nothing can be validated'] };
  try { return loadConfigText(text, app.schema); } catch (e) { return { config: null, errors: [], syntax: syntaxProblem(e, text) }; }
}

function showProblems(box, res, lead) {
  setInvalid(true);
  if (res.syntax) {
    box.append(el('p', { class: 'notice bad' }, el('strong', { text: lead }), ` is not valid JSON: ${res.syntax.text}`));
    if (res.syntax.at !== null) box.append(el('p', {}, el('button', { class: 'link-btn', text: 'Show where', attrs: { type: 'button' }, on: { click: () => select([res.syntax.at, res.syntax.at + 1]) } })));
    return;
  }
  const n = res.errors.length;
  box.append(el('p', { class: 'notice bad' }, el('strong', { text: lead }), ` has ${n} problem${n > 1 ? 's' : ''}; nothing was changed:`));
  const text = $('#cfg-text').value;
  box.append(el('ul', { class: 'errors' }, res.errors.slice(0, 30).map((x) => {
    const r = locate(text, x);
    return el('li', {}, r ? el('button', { class: 'err-loc', attrs: { type: 'button', title: 'Select this key in the editor' }, on: { click: () => select(r) } }, el('code', { text: x })) : el('code', { text: x }));
  })));
}

function tierNotes(config) {
  const notes = [];
  if (app.tier !== 'desktop' && config.model.arch !== 'mlp-tiny') notes.push(`model.arch is "${config.model.arch}"; the web demo always trains mlp-tiny (the desktop app will use ${config.model.arch}).`);
  if (app.tier !== 'desktop' && !['synthetic-shapes', 'images'].includes(config.data.kind)) notes.push(`data.kind "${config.data.kind}" needs the desktop app.`);
  return notes;
}

function applyConfig(config) {
  app.config = config;
  app.config.model = { ...config.model };
  app.emit('config-loaded');
  showConfig({ force: true });
}

function validateEditor() {
  const box = clear($('#cfg-result'));
  const res = checkText($('#cfg-text').value);
  if (res.syntax || res.errors.length) { showProblems(box, res, 'The edited config'); return null; }
  setInvalid(false);
  box.append(el('p', { class: 'notice good', text: 'Valid: it passes config.schema.json. Apply sends it to the Data and Train tabs.' }));
  const notes = tierNotes(res.config);
  if (notes.length) box.append(el('ul', { class: 'hint list' }, notes.map((n) => el('li', { text: n }))));
  return res.config;
}

function applyEditor() {
  const box = clear($('#cfg-result'));
  const res = checkText($('#cfg-text').value);
  if (res.syntax || res.errors.length) { showProblems(box, res, 'The edited config'); notify('bad', 'Config not applied', 'Fix the problems listed under the editor.'); return; }
  const notes = tierNotes(res.config);
  applyConfig(res.config);
  box.append(el('p', { class: 'notice good', text: 'Applied. The Data and Train tabs now use this config.' }));
  if (notes.length) box.append(el('ul', { class: 'hint list' }, notes.map((n) => el('li', { text: n }))));
  notify('good', 'Config applied', 'The Data and Train tabs now use it.');
}

/** An opened file goes into the editor; a valid one is applied at once, an invalid one stays
 *  there with its problems listed, so it can be fixed in place. */
function openText(name, text) {
  const box = clear($('#cfg-result'));
  let shown = text;
  try { shown = JSON.stringify(JSON.parse(text), null, 2); } catch { /* keep the raw text so the error position matches */ }
  const res = checkText(text);
  if (res.syntax || res.errors.length) {
    $('#cfg-text').value = res.syntax ? text : shown;
    editor.stale = false;
    renderState();
    showProblems(box, res, name);
    notify('bad', `${name} not applied`, res.syntax ? 'It is not valid JSON.' : `${res.errors.length} problem${res.errors.length > 1 ? 's' : ''}; see the Export tab.`);
    return;
  }
  const notes = tierNotes(res.config);
  applyConfig(res.config);
  box.append(el('p', { class: 'notice good', text: `Opened ${name}. Settings were applied to the Data and Train tabs.` }));
  if (notes.length) box.append(el('ul', { class: 'hint list' }, notes.map((n) => el('li', { text: n }))));
  notify('good', `Opened ${name}`, 'Settings were applied to the Data and Train tabs.');
}

function confirmReplace(name, text) {
  editor.pending = { name, text };
  $('#cfg-confirm-text').textContent = `You have unapplied edits. Replace them with ${name}?`;
  $('#cfg-confirm').hidden = false;
  $('#cfg-confirm-yes').focus();
}

function setupEditor() {
  const ta = $('#cfg-text');
  ta.addEventListener('input', () => { renderState(); if (ta.getAttribute('aria-invalid')) setInvalid(false); });
  ta.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); applyEditor(); }
  });
  $('#cfg-validate').addEventListener('click', validateEditor);
  $('#cfg-apply').addEventListener('click', applyEditor);
  $('#cfg-revert').addEventListener('click', () => {
    showConfig({ force: true });
    clear($('#cfg-result')).append(el('p', { class: 'notice', text: 'Edits discarded; this is the config the app is using.' }));
  });
  $('#cfg-confirm-yes').addEventListener('click', () => {
    const p = editor.pending;
    editor.pending = null;
    $('#cfg-confirm').hidden = true;
    if (p) openText(p.name, p.text);
    ta.focus();
  });
  $('#cfg-confirm-no').addEventListener('click', () => {
    editor.pending = null;
    $('#cfg-confirm').hidden = true;
    clear($('#cfg-result')).append(el('p', { class: 'notice', text: 'Kept your edits; the file was not opened.' }));
    ta.focus();
  });
}

export function init(a) {
  app = a;
  $('#exp-csv').addEventListener('click', (e) => wrap(e.currentTarget, async () => {
    if (!needRun()) return;
    const r = await embeddings();
    download(`${runName()}-embeddings.csv`, matrixToCsv(r.emb, r.n, r.embDim, r.labels, r.classes), 'text/csv');
    saved('Saved embeddings (CSV).', `${r.n} × ${r.embDim} values.`);
  }));
  $('#exp-npy').addEventListener('click', (e) => wrap(e.currentTarget, async () => {
    if (!needRun()) return;
    const r = await embeddings();
    download(`${runName()}-embeddings.npy`, npyBytes(r.emb, [r.n, r.embDim]));
    saved('Saved embeddings (.npy).', `${r.n} × ${r.embDim} float32 values.`);
  }));
  $('#exp-weights').addEventListener('click', (e) => wrap(e.currentTarget, async () => {
    if (!needRun()) return;
    download(`${runName()}-model.safetensors`, await weightsBytes());
    saved('Saved weights (.safetensors).');
  }));
  $('#imp-weights').addEventListener('change', async (e) => {
    const f = e.target.files[0];
    e.target.value = '';
    if (!f) return;
    if (f.size > 64 * 1024 * 1024) { status('That file is larger than 64 MB; mlp-tiny weights are under 1 MB.'); return; }
    try {
      const { tensors } = parseSafetensors(new Uint8Array(await f.arrayBuffer()));
      await app.backend.loadWeights(tensors, app.config);
      if (!app.run) app.run = { started: new Date().toISOString(), events: [], summary: {}, backend: 'loaded weights', config: deepClone(app.config) };
      status(`Loaded weights from ${f.name}. Inspect and Evaluate now use them.`);
      notify('good', 'Weights opened', `${f.name}: Inspect and Evaluate now use them.`);
      app.emit('weights-loaded');
    } catch (err) { status(`Could not load ${f.name}: ${err.message}`); notify('bad', 'Weights not opened', err.message); }
  });
  $('#exp-report-json').addEventListener('click', () => {
    download(`${runName()}-report.json`, JSON.stringify(buildReport(app), null, 2), 'application/json');
    saved('Saved the run report (JSON).');
  });
  $('#exp-report-html').addEventListener('click', () => {
    download(`${runName()}-report.html`, reportHtml(buildReport(app)), 'text/html');
    saved('Saved the run report (HTML).');
  });
  $('#exp-bundle').addEventListener('click', (e) => wrap(e.currentTarget, async () => {
    if (!needRun()) return;
    const rep = buildReport(app);
    const r = await embeddings();
    const files = [
      { name: 'config.json', data: `${JSON.stringify(rep.config, null, 2)}\n` },
      { name: 'report.json', data: `${JSON.stringify(rep, null, 2)}\n` },
      { name: 'weights.safetensors', data: await weightsBytes() },
      { name: 'embeddings.npy', data: npyBytes(r.emb, [r.n, r.embDim]) },
      { name: 'README.txt', data: readme(rep) },
    ];
    const zip = zipBytes(files);
    download(`${runName()}-${stamp()}.zip`, zip, 'application/zip');
    app.lastBundle = zip;
    saved('Saved the reproducibility bundle.', `${(zip.length / 1048576).toFixed(2)} MB zip.`);
  }));
  $('#cfg-save').addEventListener('click', () => {
    const errs = app.schema ? validate(app.config, app.schema) : [];
    const box = clear($('#cfg-result'));
    if (errs.length) { box.append(el('p', { class: 'notice bad', text: 'This config does not pass the schema, so it was not saved:' }), el('ul', { class: 'errors' }, errs.map((x) => el('li', { text: x })))); notify('bad', 'Config not saved', 'It does not pass the schema.'); return; }
    download(`${runName()}.json`, `${JSON.stringify(app.config, null, 2)}\n`, 'application/json');
    box.append(el('p', { class: 'notice good', text: 'Saved. Open it here or run `jepa-studio train --config` on desktop.' }));
    notify('good', 'Saved the config', 'Open it here or with jepa-studio train --config on desktop.');
    app.emit('exported');
  });
  $('#cfg-open').addEventListener('change', async (e) => {
    const f = e.target.files[0];
    e.target.value = '';
    if (!f) return;
    if (f.size > 1_000_000) {
      clear($('#cfg-result')).append(el('p', { class: 'notice bad', text: `${f.name} is larger than 1 MB; refusing to parse.` }));
      notify('bad', `${f.name} not opened`, 'Config files are limited to 1 MB.');
      return;
    }
    const text = await f.text();
    if (isDirty()) confirmReplace(f.name, text); else openText(f.name, text);
  });
  setupEditor();
  app.on('config', showConfig);
  for (const ev of ['run-start', 'weights-loaded']) app.on(ev, updateGates);
  updateGates();
  showConfig();
}

export function show() { showConfig(); }
