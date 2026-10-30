// Science tab (both tiers): a live WebGL black hole and three physics simulations, drawn by the
// modules in site/js/science/ (blackhole.js: mountBlackHole, sims.js: mountSim). They are loaded
// with a dynamic import, so a browser without WebGL, or a build without the modules, gets a
// plain explanation instead of a broken page. In the desktop tier the same settings can be
// rendered as a high-quality video by the local backend (POST /api/jev/science/*), with job
// progress and the resulting files in this tab's gallery.
import { $, $$, el, download } from '../ui/dom.js';
import { field, valueOf } from '../ui/fields.js';
import { getPref, setPref } from '../ui/prefs.js';
import { Gallery } from '../jev/gallery.js';
import { follow } from '../jev/client.js';

let app;
let view = 'blackhole';
let bh = null, sim = null, bhMod = null, simMod = null;
let simKind = 'nbody';
let gallery = null, client = null;
let active = false;

const BH_DEFAULTS = { spin: 0.9, distance: 25, inclination: 80, quality: 'medium', disk: true, doppler: true, beaming: true };
const SIM_DEFAULTS = {
  nbody: { 'sim-bodies': 60, 'sim-nbody-speed': 1 },
  double_pendulum: { 'sim-theta1': 120, 'sim-theta2': -20, 'sim-mass-ratio': 1, 'sim-pend-speed': 1 },
  wave: { 'sim-wave-speed': 1, 'sim-wave-damping': 0.002, 'sim-wave-freq': 1.5 },
};
const KIND_LABEL = { nbody: 'N-body gravity', double_pendulum: 'Double pendulum', wave: 'Waves on a membrane' };
// The backend's simulate route names (docs/jev/api.md) where they differ from the browser module's.
const API_KIND = { double_pendulum: 'pendulum' };

const reduced = () => matchMedia('(prefers-reduced-motion: reduce)').matches;

export function init(a) {
  app = a;
  const num = (id, def, section) => field(id, { section, def: () => def, rangeMsg: (lo, hi) => `Use ${lo} to ${hi}.`, commit: () => push() });
  num('bh-spin', BH_DEFAULTS.spin, 'science-bh');
  num('bh-distance', BH_DEFAULTS.distance, 'science-bh');
  num('bh-incl', BH_DEFAULTS.inclination, 'science-bh');
  for (const [, ids] of Object.entries(SIM_DEFAULTS)) for (const [id, def] of Object.entries(ids)) num(id, def, 'science-sim');
  for (const id of ['bh-spin', 'bh-distance', 'bh-incl', ...Object.values(SIM_DEFAULTS).flatMap(Object.keys)]) {
    $(`#${id}`).addEventListener('input', () => push());
  }
  for (const id of ['bh-disk', 'bh-doppler', 'bh-beaming', 'bh-quality']) $(`#${id}`).addEventListener('change', () => { syncDiskDeps(); push(); });
  $('[data-reset="science-bh"]').addEventListener('click', () => {
    for (const [id, k] of [['bh-disk', 'disk'], ['bh-doppler', 'doppler'], ['bh-beaming', 'beaming']]) $(`#${id}`).checked = BH_DEFAULTS[k];
    $('#bh-quality').value = BH_DEFAULTS.quality;
    syncDiskDeps();
    push();
  });
  $('[data-reset="science-sim"]').addEventListener('click', () => push());

  for (const r of $$('input[name="sci-view"]')) r.addEventListener('change', () => { if (r.checked) setView(r.value); });
  $('#sim-kind').addEventListener('change', (e) => setKind(e.target.value));
  $('#bh-snapshot').addEventListener('click', snapshot);
  $('#sim-reset').addEventListener('click', () => { if (sim && sim.reset) sim.reset(); $('#sci-status').textContent = `${KIND_LABEL[simKind]} restarted.`; });
  $('#sci-retry').addEventListener('click', () => { teardown(); mount(); });

  const savedView = getPref('sci-view');
  if (savedView === 'sims' || savedView === 'blackhole') view = savedView;
  const savedKind = getPref('sci-kind');
  if (SIM_DEFAULTS[savedKind]) simKind = savedKind;
  for (const r of $$('input[name="sci-view"]')) r.checked = r.value === view;
  $('#sim-kind').value = simKind;
  syncGroups();
  syncDiskDeps();

  if (app.tier === 'desktop') {
    for (const n of $$('#panel-science .desktop-only')) n.hidden = false;
    $('#sci-web-note').hidden = true;
    $('#bh-render').addEventListener('click', () => renderOnComputer('blackhole'));
    $('#sim-render').addEventListener('click', () => renderOnComputer('sim'));
  }
  // Pause rendering while another tab is open or the page is hidden.
  document.addEventListener('visibilitychange', () => { if (document.hidden) teardown(); else if (active) mount(); });
}

export function show() {
  active = true;
  mount();
  if (!app._sciHideHooked && app.router) {
    app._sciHideHooked = true;
    // the router has no "hide" callback; watch the panel's hidden attribute instead
    new MutationObserver(() => { if ($('#panel-science').hidden) { active = false; teardown(); } }).observe($('#panel-science'), { attributes: true, attributeFilter: ['hidden'] });
  }
}

function syncGroups() {
  $('#sci-bh').hidden = view !== 'blackhole';
  $('#sci-sims').hidden = view !== 'sims';
  for (const g of $$('[data-sim-kind]')) g.hidden = g.dataset.simKind !== simKind;
  const topic = view === 'blackhole' ? 'blackhole' : simKind;
  for (const g of $$('[data-explain]')) g.hidden = g.dataset.explain !== topic;
  $('#sci-stage-title').textContent = view === 'blackhole' ? 'Black hole' : KIND_LABEL[simKind];
}

function syncDiskDeps() {
  const on = $('#bh-disk').checked;
  for (const id of ['bh-doppler', 'bh-beaming']) $(`#${id}`).disabled = !on;
}

function setView(v) {
  view = v;
  setPref('sci-view', v);
  syncGroups();
  teardown();
  mount();
}

function setKind(k) {
  if (!SIM_DEFAULTS[k]) return;
  simKind = k;
  setPref('sci-kind', k);
  syncGroups();
  teardown();
  mount();
}

function bhParams() {
  return {
    spin: valueOf('bh-spin') ?? BH_DEFAULTS.spin,
    distance: valueOf('bh-distance') ?? BH_DEFAULTS.distance,
    inclination: valueOf('bh-incl') ?? BH_DEFAULTS.inclination,
    disk: $('#bh-disk').checked,
    doppler: $('#bh-doppler').checked,
    beaming: $('#bh-beaming').checked,
    quality: $('#bh-quality').value,
  };
}

function simParams() {
  const p = {};
  for (const id of Object.keys(SIM_DEFAULTS[simKind])) {
    const key = id.replace(/^sim-/, '').replace(/^(nbody|pend|wave)-/, '').replace(/-([a-z])/g, (_, c) => c.toUpperCase());
    p[key] = valueOf(id) ?? SIM_DEFAULTS[simKind][id];
  }
  if (simKind === 'nbody') p.bodies = Math.round(p.bodies);
  return p;
}

function push() {
  try {
    if (view === 'blackhole' && bh) bh.setParams(bhParams());
    if (view === 'sims' && sim) sim.setParams(simParams());
  } catch (e) { console.warn('science setParams', e); }
}

function teardown() {
  for (const h of [bh, sim]) { if (h && h.destroy) { try { h.destroy(); } catch (e) { console.warn(e); } } }
  bh = null; sim = null;
  // a fresh canvas each time: a WebGL context lost by destroy() cannot be reused
  const old = $('#sci-canvas');
  const c = el('canvas', { attrs: { id: 'sci-canvas', role: 'img', 'aria-label': old.getAttribute('aria-label') || '' } });
  old.replaceWith(c);
}

function fallback(reason) {
  const box = $('#sci-fallback');
  box.hidden = false;
  $('#sci-fallback-text').textContent = reason;
  $('#sci-canvas').hidden = true;
  $('#sci-stage').dataset.state = 'fallback';
  $('#bh-snapshot').disabled = true;
}

async function mount() {
  if (!active || document.hidden) return;
  const canvas = $('#sci-canvas');
  canvas.hidden = false;
  $('#sci-fallback').hidden = true;
  $('#bh-snapshot').disabled = false;
  $('#sci-stage').dataset.state = 'loading';
  const rect = canvas.getBoundingClientRect();
  const dpr = Math.min(2, window.devicePixelRatio || 1);
  if (rect.width) { canvas.width = Math.round(rect.width * dpr); canvas.height = Math.round(rect.width * 9 / 16 * dpr); }
  const want = view;
  const opts = { reducedMotion: reduced() };
  try {
    if (want === 'blackhole') {
      bhMod = bhMod || await import('../science/blackhole.js');
      if (view !== want || !active) return;
      canvas.setAttribute('aria-label', 'Rendered black hole: a dark shadow ringed by a lensed accretion disk. Change the view with the controls.');
      bh = bhMod.mountBlackHole(canvas, { ...opts, ...bhParams() });
      describeCaps(bhMod, bh);
    } else {
      simMod = simMod || await import('../science/sims.js');
      if (view !== want || !active) return;
      canvas.setAttribute('aria-label', `${KIND_LABEL[simKind]} simulation, animated.`);
      sim = simMod.mountSim(canvas, simKind, { ...opts, ...simParams() });
    }
    $('#sci-stage').dataset.state = 'live';
    $('#sci-status').textContent = want === 'blackhole' ? 'Black hole view running on your graphics card.' : `${KIND_LABEL[simKind]} running.`;
  } catch (e) {
    console.warn('science view unavailable', e);
    const msg = String((e && e.message) || e);
    const noModule = /Failed to fetch dynamically imported module|Importing a module script failed|error loading dynamically imported module/i.test(msg);
    fallback(noModule
      ? 'The live view is not part of this build yet.'
      : /webgl/i.test(msg) ? `This browser could not start WebGL, which the live view needs (${msg}). Try another browser, or turn on hardware acceleration.`
        : `The live view could not start: ${msg}`);
    $('#sci-status').textContent = 'Live view unavailable.';
  }
}

/** Show what the browser renderer supports (e.g. spin), if the module says. */
function describeCaps(mod, handle) {
  const caps = (handle && handle.capabilities) || mod.capabilities || mod.BLACKHOLE_CAPABILITIES || null;
  const note = $('#bh-spin-note');
  if (caps && caps.spin === false) {
    note.hidden = false;
    note.textContent = app.tier === 'desktop'
      ? 'The live view draws a non-spinning hole; spin is used when you render on this computer.'
      : 'The live view draws a non-spinning hole; the desktop app renders spin.';
  } else if (caps && caps.spin && typeof caps.spin === 'object' && caps.spin.note) {
    note.hidden = false;
    note.textContent = String(caps.spin.note);
  } else {
    note.hidden = true;
  }
}

async function snapshot() {
  const name = view === 'blackhole' ? 'black-hole.png' : `${simKind}.png`;
  try {
    let s = bh && bh.snapshot ? await bh.snapshot() : null;
    if (s && typeof s === 'string') s = await (await fetch(s)).blob();
    else if (s && s instanceof HTMLCanvasElement) s = await new Promise((r) => s.toBlob(r, 'image/png'));
    if (!(s instanceof Blob)) s = await new Promise((r) => $('#sci-canvas').toBlob(r, 'image/png'));
    if (!s) throw new Error('the view returned no image');
    download(name, s, 'image/png');
    app.notify('good', 'Snapshot saved', name);
  } catch (e) {
    app.notify('bad', 'Could not save the snapshot', String(e.message || e));
  }
}

// ------------------------------------------------------------------ desktop: render on this computer
function ensureGallery() {
  if (gallery) return gallery;
  client = app.jev && app.jev.client;
  gallery = new Gallery($('#sci-gallery'), client, { empty: 'Videos and images rendered on this computer appear here.', count: $('#sci-gallery-count') });
  return gallery;
}

async function renderOnComputer(which) {
  if (!app.jev) { app.notify('bad', 'Rendering needs the desktop backend'); return; }
  ensureGallery();
  const btn = which === 'blackhole' ? $('#bh-render') : $('#sim-render');
  const st = $('#sci-job-status');
  const bar = $('#sci-job-bar');
  const [w, h] = $('#sci-res').value.split('x').map(Number);
  const seconds = Number($('#sci-seconds').value) || 8;
  btn.disabled = true;
  st.textContent = 'Starting the render…';
  bar.hidden = false;
  bar.firstChild.style.width = '0%';
  bar.setAttribute('aria-valuenow', '0');
  let r;
  try {
    if (which === 'blackhole') {
      const p = bhParams();
      r = await app.jev.client.renderBlackHole({
        spin: p.spin, inclination_deg: p.inclination, distance_M: p.distance,
        disk: { enabled: p.disk, doppler: p.doppler, beaming: p.beaming },
        width: w, height: h, seconds, frames: Math.round(seconds * 24),
      });
    } else {
      r = await app.jev.client.simulate(API_KIND[simKind] || simKind, { ...simParams(), seconds, width: w, height: h });
    }
  } catch (e) {
    st.textContent = `The render did not start: ${e.message || e}`;
    bar.hidden = true;
    btn.disabled = false;
    return;
  }
  const go = (jobId) => followJob(jobId, btn);
  if (r.job_id) go(r.job_id);
  else if (r.approval_id) {
    st.textContent = 'This render is long, so it needs your approval.';
    app.jev.approvals.add({ approval_id: r.approval_id, name: which === 'blackhole' ? 'render_black_hole' : 'simulate', args: r.args || { width: w, height: h, seconds }, reason: r.reason || 'Long renders use the CPU or GPU for several minutes.', risk: 'sandbox' });
    app.jev.approvals.whenDecided(r.approval_id, (decision, res) => {
      if (decision === 'approve' && res.job_id) go(res.job_id);
      else { st.textContent = decision === 'approve' ? 'Approved; the render runs in the background.' : 'Render cancelled.'; bar.hidden = true; btn.disabled = false; }
    });
  } else {
    st.textContent = 'The backend did not return a job.';
    bar.hidden = true;
    btn.disabled = false;
  }
}

function followJob(jobId, btn) {
  const st = $('#sci-job-status');
  const bar = $('#sci-job-bar');
  const t0 = performance.now();
  follow((since) => client.jobEvents(jobId, since), (ev) => {
    if (ev.event === 'progress') {
      const pct = Math.round((Number(ev.fraction) || 0) * 100);
      bar.firstChild.style.width = `${pct}%`;
      bar.setAttribute('aria-valuenow', String(pct));
      st.textContent = `Rendering: ${pct}%${ev.message ? `, ${ev.message}` : ''}`;
    } else if (ev.event === 'artifact') {
      gallery.add(ev.artifact || ev);
    } else if (ev.event === 'tool_result') {
      for (const a of ev.artifacts || []) gallery.add(a);
      if (!ev.ok) st.textContent = `The render failed: ${ev.content || ''}`;
    } else if (ev.event === 'done') {
      const secs = ((performance.now() - t0) / 1000).toFixed(0);
      st.textContent = `Finished in ${secs} s. The files are below.`;
      app.notify('good', 'Render finished', `${secs} s on this computer`);
    } else if (ev.event === 'error') {
      st.textContent = `The render failed: ${ev.message}`;
    } else if (ev.event === 'cancelled') {
      st.textContent = 'The render was cancelled.';
    }
  }, { fast: 700 }).done.then(() => { bar.hidden = true; btn.disabled = false; });
}

