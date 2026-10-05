// Inspect tab: PCA explorer (2D/3D), isotropy + collapse detector, nearest neighbours,
// saliency (plus ViT attention on desktop vit-tiny runs), and the "projections turn Gaussian" animation from training snapshots.
import { $, $$, el, clear, setStatus, collapseStatus, COLLAPSE_TEXT, metrics } from '../ui/dom.js';
import { BarChart, HistChart, drawCHW, fmt, cssVar } from '../ui/charts.js';
import { Scatter3D, classStyle } from '../ui/scatter3d.js';
import { datasetImage, datasetLabel } from './data.js';
import { epExpectedUnderNull } from '../engine/lejepa.js';
import { field, syncFromRange } from '../ui/fields.js';
import { notify } from '../ui/toast.js';

let app, scatter, eigen, hists, snaps = [], playing = null, space = 'embedding', data = null;

function thumb(i, cls = '') {
  // desktop tier: the backend sends thumbnails of the real samples (PNG data URLs)
  if (data && data.thumbs && data.thumbs[i]) return el('img', { attrs: { src: data.thumbs[i], alt: '', width: '32', height: '32' } });
  if (String(app.data.kind).startsWith('desktop-')) return el('span', { class: 'hint', text: `#${i}` });
  const img = datasetImage(app, i);
  const c = el('canvas', { attrs: { 'aria-hidden': 'true' } });
  drawCHW(c, img.data, img.c, img.h, img.w);
  return c;
}

function renderIsotropy(rep, col) {
  setStatus($('#inspect-verdict'), collapseStatus(col.status), COLLAPSE_TEXT[col.status]);
  $('#inspect-verdict-msg').textContent = col.message;
  metrics($('#inspect-metrics'), [
    ['Effective rank', fmt(rep.effective_rank, 3), `of ${rep.dim}`],
    ['Isotropy ratio', fmt(rep.isotropy_ratio, 2)],
    ['SIGReg', fmt(rep.sigreg, 3), `null ≈ ${epExpectedUnderNull().toFixed(2)}`],
    ['Mean std', fmt(rep.mean_std, 3), 'target 1'],
  ]);
  eigen.set(rep.eigenvalues, 1);
}

async function refresh() {
  const btn = $('#inspect-refresh');
  btn.disabled = true;
  btn.textContent = 'Computing…';
  try {
    data = await app.backend.inspect(Math.min(1024, app.data.n), space);
    app.inspect = data;
    scatter.set(data.pca3, data.n, data.labels);
    scatter.select(-1, []);
    renderIsotropy(data.isotropy, data.collapse);
    $('#pca-var').textContent = `PCA of ${space === 'projection' ? 'projector outputs (16-d)' : 'backbone embeddings (64-d)'}: ${data.var.map((v) => `${(v * 100).toFixed(0)}%`).join(' / ')} of variance`;
    renderLegend();
    if (app.run) app.emit('inspected', data);
  } catch (e) {
    setStatus($('#inspect-verdict'), 'idle', 'No model yet');
    $('#inspect-verdict-msg').textContent = e.message;
    if (app.run) notify('bad', 'Inspect failed', e.message);
  }
  btn.textContent = 'Refresh from model';
  updateRefresh();
}

/** Before any run there is nothing to inspect: the empty state above says so. */
function updateRefresh() { $('#inspect-refresh').disabled = !app.run; }

function renderLegend() {
  const box = clear($('#scatter-legend'));
  if (!data || !data.labels || !data.classes) return;
  data.classes.forEach((name, i) => {
    const st = classStyle(i);
    const sw = el('span', { class: 'swatch', attrs: { 'aria-hidden': 'true' } });
    sw.style.setProperty('--c', `var(${st.color})`);
    if (st.ring) { sw.style.background = 'transparent'; sw.style.border = `2px solid var(${st.color})`; sw.style.borderRadius = '50%'; } else sw.style.borderRadius = '50%';
    const b = el('button', { attrs: { type: 'button', 'aria-pressed': String(scatter.focusClass === null || scatter.focusClass === i) }, on: { click: () => { scatter.focusClass = scatter.focusClass === i ? null : i; scatter.draw(); renderLegend(); } } }, sw, name);
    box.append(b);
  });
}

function labelOf(i) {
  if (String(app.data.kind).startsWith('desktop-')) {
    return data && data.labels && data.classes ? data.classes[data.labels[i]] ?? null : null;
  }
  return datasetLabel(app, i);
}

async function pick(i) {
  if (!data) { await refresh(); if (!data) return; }
  const q = clear($('#nn-query'));
  q.append(`Query: image ${i}`, labelOf(i) ? ` (${labelOf(i)})` : '');
  const row = clear($('#nn-row'));
  row.append(el('figure', {}, thumb(i), el('figcaption', { text: 'query' })));
  try {
    const r = await app.backend.neighbors(i, 8, 'embedding');
    for (const [j, s] of r.neighbors) {
      const lab = labelOf(j);
      row.append(el('figure', {}, thumb(j), el('figcaption', { text: `${lab ? `${lab}, ` : ''}cos ${s.toFixed(2)}` })));
    }
    scatter.select(i, r.neighbors.map((x) => x[0]));
  } catch (e) { row.append(el('p', { class: 'hint', text: e.message })); }
  try {
    const s = await app.backend.saliency(i);
    // desktop tier: the backend sends the exact input it explained; web tier: the in-page dataset
    const img = s.image ? await decodeImage(s.image) : datasetImage(app, i);
    const box = clear($('#saliency'));
    const c1 = el('canvas', { attrs: { 'aria-hidden': 'true' } });
    drawCHW(c1, img.data, 3, img.h, img.w);
    const c2 = el('canvas', { attrs: { role: 'img', 'aria-label': 'Saliency overlay' } });
    drawSaliency(c2, img, s.map, s.side);
    box.append(el('figure', {}, c1, el('figcaption', { text: 'input' })), el('figure', {}, c2, el('figcaption', { text: 'saliency overlay' })));
    box.classList.toggle('has-attention', !!s.attention);
    if (s.attention) {
      const c3 = el('canvas', { attrs: { role: 'img', 'aria-label': 'ViT attention overlay: last block, CLS token to each patch, mean over heads' } });
      drawSaliency(c3, img, s.attention.map, s.attention.side);
      box.append(el('figure', {}, c3, el('figcaption', { text: 'attention (CLS → patches)' })));
    }
  } catch (e) { clear($('#saliency')).append(el('p', { class: 'hint', text: e.message })); }
  $$('#inspect-thumbs button').forEach((b) => b.setAttribute('aria-pressed', String(Number(b.dataset.i) === i)));
}

function drawSaliency(canvas, img, map, side) {
  const S = img.h, hw = S * S;
  canvas.width = S; canvas.height = S;
  const ctx = canvas.getContext('2d');
  const im = ctx.createImageData(S, S);
  // min-max normalize so the overlay shows relative importance, then a gamma to keep the
  // low end transparent (a nearly uniform map would otherwise tint the whole image)
  let mx = -Infinity, mn = Infinity;
  for (const v of map) { mx = Math.max(mx, v); mn = Math.min(mn, v); }
  const span = Math.max(mx - mn, 1e-12);
  const heat = hexToRgb(cssVar('--flow-loss-sigreg'));
  for (let y = 0; y < S; y++) for (let x = 0; x < S; x++) {
    const i = y * S + x;
    const m = ((map[Math.floor((y * side) / S) * side + Math.floor((x * side) / S)] - mn) / span) ** 2;
    const g = 0.299 * img.data[i] + 0.587 * img.data[hw + i] + 0.114 * img.data[2 * hw + i];
    const base = 0.25 + 0.5 * g;
    for (let c = 0; c < 3; c++) im.data[4 * i + c] = Math.round(255 * ((1 - m) * base + m * heat[c]));
    im.data[4 * i + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}

/** PNG data URL -> {data (CHW floats in [0, 1]), c: 3, h, w}, the shape datasetImage() returns. */
async function decodeImage(url) {
  const im = new Image();
  im.decoding = 'async';
  await new Promise((resolve, reject) => {
    im.onload = resolve;
    im.onerror = () => reject(new Error('could not decode the sample image from the backend'));
    im.src = url;
  });
  const h = im.naturalHeight, w = im.naturalWidth, hw = h * w;
  const c = document.createElement('canvas');
  c.width = w; c.height = h;
  const ctx = c.getContext('2d');
  ctx.drawImage(im, 0, 0);
  const px = ctx.getImageData(0, 0, w, h).data;
  const data = new Float32Array(3 * hw);
  for (let i = 0; i < hw; i++) for (let k = 0; k < 3; k++) data[k * hw + i] = px[4 * i + k] / 255;
  return { data, c: 3, h, w };
}

function hexToRgb(h) {
  const m = /^#?([0-9a-f]{2})([0-9a-f]{2})([0-9a-f]{2})$/i.exec(h.trim());
  return m ? [parseInt(m[1], 16) / 255, parseInt(m[2], 16) / 255, parseInt(m[3], 16) / 255] : [1, 0, 0];
}

/** k is the 0-based snapshot index; the slider and its number show k + 1 ("12 of 12"). */
function showSnapshot(k) {
  if (!snaps.length) return;
  const s = snaps[Math.max(0, Math.min(k, snaps.length - 1))];
  s.hist.forEach((h, j) => hists[j].set(h));
  $('#anim-step').textContent = `step ${s.step} (${Math.min(k, snaps.length - 1) + 1} of ${snaps.length} snapshots)`;
}

function renderThumbs() {
  const box = clear($('#inspect-thumbs'));
  const n = Math.min(64, app.data.n);
  for (let i = 0; i < n; i++) {
    const b = el('button', { attrs: { type: 'button', 'data-i': i, 'aria-label': `Image ${i}${labelOf(i) ? `, ${labelOf(i)}` : ''}`, 'aria-pressed': 'false' }, on: { click: () => pick(i) } });
    b.append(thumb(i));
    box.append(b);
  }
}

export function init(a) {
  app = a;
  scatter = new Scatter3D($('#scatter'), { onPick: pick });
  eigen = new BarChart($('#c-eigen'), { color: '--flow-encoder', refLabel: 'N(0, I) target = 1', xLabel: 'eigenvalues, largest first', itemLabel: (i) => `λ${i + 1}`, empty: 'Refresh to compute' });
  hists = [0, 1, 2].map((j) => new HistChart($(`#h-${j}`), { color: '--flow-loss-sigreg' }));
  $('#inspect-refresh').addEventListener('click', refresh);
  $$('input[name="space"]').forEach((r) => r.addEventListener('change', () => { space = r.value; refresh(); }));
  $$('input[name="dims"]').forEach((r) => r.addEventListener('change', () => scatter.setMode(r.value)));
  const slider = $('#anim-slider');
  const setSlider = (k) => { slider.value = String(k + 1); syncFromRange('anim-slider'); showSnapshot(k); };
  field('anim-slider', { integer: true, label: 'Training snapshot', commit: () => showSnapshot(Number(slider.value) - 1),
    rangeMsg: (lo, hi) => `Pick a snapshot from ${lo} to ${hi}.` });
  slider.addEventListener('input', () => showSnapshot(Number(slider.value) - 1));
  $('#anim-play').addEventListener('click', () => {
    if (playing) { clearInterval(playing); playing = null; $('#anim-play').textContent = 'Play'; return; }
    if (!snaps.length) return;
    const reduce = matchMedia('(prefers-reduced-motion: reduce)').matches;
    if (reduce) { setSlider(snaps.length - 1); return; }
    const cur = Number(slider.value) - 1;
    let k = cur >= snaps.length - 1 ? 0 : cur;
    $('#anim-play').textContent = 'Pause';
    playing = setInterval(() => {
      setSlider(k); k++;
      if (k >= snaps.length) { clearInterval(playing); playing = null; $('#anim-play').textContent = 'Play'; }
    }, 350);
  });
  app.on('snapshots', (s) => {
    snaps = s;
    slider.disabled = !snaps.length;
    slider.max = String(Math.max(1, snaps.length));
    if (!playing && snaps.length) setSlider(snaps.length - 1);
    if (!snaps.length) { slider.value = '1'; syncFromRange('anim-slider'); hists.forEach((h) => h.set([])); $('#anim-step').textContent = 'no snapshots yet'; }
  });
  app.on('isotropy', ({ report, collapse }) => { if (!data || space === 'projection') renderIsotropy(report, collapse); else renderIsotropy(report, collapse); });
  app.on('run-end', () => { if (app.router && app.router.current() === 'inspect') refresh(); });
  app.on('run-start', updateRefresh);
  app.on('weights-loaded', updateRefresh);
  updateRefresh();
  app.on('data', () => { data = null; renderThumbs(); scatter.set(null, 0, null); });
  app.on('theme', () => { scatter.draw(); renderLegend(); });
  renderThumbs();
}

export function show() {
  scatter.draw();
  if (!data && app.run) refresh();
}
