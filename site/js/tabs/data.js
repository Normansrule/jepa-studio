// Data tab: built-in Shapes, local images (decoded in the page, never uploaded), or a numeric
// CSV (preview only); augmentation views side by side. Images and CSV files can also be dropped
// (files or whole folders) or pasted onto the drop zone; they go through the same loaders.
import { $, $$, el, clear, gate } from '../ui/dom.js';
import { field, onCommit, valueOf, setValue, onValidity, invalidFor, fixMessage, refreshValidity } from '../ui/fields.js';
import { collectDropped, isImageFile, isCsvFile } from '../ui/dropzone.js';
import { notify } from '../ui/toast.js';
import { webDefaultConfig } from '../config.js';
import { drawCHW, SeriesPlot } from '../ui/charts.js';
import { render, hwcToChw, CLASSES } from '../shapes.js';
import { makeImage, makeViews, seriesView } from '../engine/views.js';
import { mulberry32, mixSeed } from '../engine/rng.js';
import { parseNumericCsv, CSV_LIMITS } from '../io/csv.js';

export const IMAGE_LIMITS = { maxFiles: 2000, maxBytes: 20 * 1024 * 1024, maxPixels: 40_000_000, size: 32 };

let app, current = 0, viewSeed = 1, series = null, seriesPlots = null;
// the image files behind the current "My images" set, so pasted images can be added to it
let imageSet = { files: [], paths: [] };

/** CHW image i of whatever dataset is active. */
export function datasetImage(appRef, i) {
  const d = appRef.data;
  if (String(d.kind).startsWith('desktop-')) throw new Error('samples from a desktop folder live in the backend');
  if (d.kind === 'images') {
    const per = 3 * d.size * d.size;
    return makeImage(d.images.subarray(i * per, (i + 1) * per), 3, d.size, d.size);
  }
  const { img } = render(appRef.config.seed, i, 32);
  return makeImage(hwcToChw(img, 32, 32), 3, 32, 32);
}

export function datasetLabel(appRef, i) {
  const d = appRef.data;
  if (String(d.kind).startsWith('desktop-')) return null;   // labels come from the backend there
  if (d.kind === 'images') return d.labels ? d.classes[d.labels[i]] : null;
  return CLASSES[render(appRef.config.seed, i, 16).label];
}

function canvasFor(img, cls = '') {
  const c = el('canvas', { attrs: { 'aria-hidden': 'true' } });
  drawCHW(c, img.data, img.c, img.h, img.w);
  return c;
}

function renderViews() {
  if (String(app.data.kind).startsWith('desktop-')) {      // new views come from the backend
    app.backend.preview(app.config, 6).then((r) => showDesktopPreview(r.samples)).catch((e) => status(e.message || String(e), 'bad'));
    return;
  }
  const img = datasetImage(app, current);
  const d = { ...app.config.data };
  const rng = mulberry32(mixSeed(viewSeed, current));
  const { globals, locals } = makeViews(img, d, rng);
  const orig = $('#view-original'), g = $('#view-globals'), l = $('#view-locals');
  clear(orig); clear(g); clear(l);
  const label = datasetLabel(app, current);
  orig.append(el('figure', { class: 'view-cell' }, canvasFor(img), el('figcaption', { text: `${img.w}×${img.h}${label ? `, ${label}` : ''}` })));
  globals.forEach((v, i) => g.append(el('figure', { class: 'view-cell global' }, canvasFor(v), el('figcaption', {}, el('span', { class: 'swatch c-view' }), `global ${i + 1}`))));
  locals.forEach((v, i) => l.append(el('figure', { class: 'view-cell local' }, canvasFor(v), el('figcaption', {}, el('span', { class: 'swatch c-view' }), `local ${i + 1}`))));
  $('#views-caption').textContent = `Image ${current}${label ? ` (${label})` : ''}`;
}

function renderThumbs() {
  const box = clear($('#data-thumbs'));
  const n = Math.min(48, app.data.n);
  for (let i = 0; i < n; i++) {
    const b = el('button', { attrs: { type: 'button', 'aria-label': `Image ${i}${datasetLabel(app, i) ? `, ${datasetLabel(app, i)}` : ''}`, 'aria-pressed': String(i === current) }, on: { click: () => { current = i; $$('#data-thumbs button').forEach((x, j) => x.setAttribute('aria-pressed', String(j === i))); renderViews(); } } });
    b.append(canvasFor(datasetImage(app, i)));
    box.append(b);
  }
}

function status(text, kind = '') {
  const s = $('#data-status');
  s.className = `notice ${kind}`;
  clear(s);
  if (Array.isArray(text)) {
    s.append(el('strong', { text: text[0] }));
    if (text[1] && text[1].length) s.append(el('ul', {}, text[1].map((t) => el('li', { text: t }))));
  } else s.textContent = text;
}

function useShapes() {
  const n = valueOf('shapes-n'), seed = valueOf('data-seed');
  if (n === null || seed === null) return;           // invalid input: the message is shown inline
  app.data = { kind: 'synthetic-shapes', n };
  app.config.data.kind = 'synthetic-shapes';
  app.config.data.max_items = n;
  app.config.seed = seed;
  app.config.data.path = null;
  app.backend.setData({ kind: 'synthetic-shapes' });
  const where = app.backend.tier === 'desktop' ? 'generated locally by the backend' : 'generated in your browser';
  status(`Shapes: ${n} images, ${where} (seed ${app.config.seed}).`);
  current = 0;
  renderThumbs();
  renderViews();
  app.emit('data');
}

/** Decode, check and resize local image files. Nothing is uploaded.
 *  paths: optional relative paths (folder drops), used for labels like the folder input's
 *  webkitRelativePath. source: 'files' | 'drop' | 'paste' (only changes the wording). */
async function loadImages(files, { paths = null, source = 'files' } = {}) {
  const list = [...files];
  const pathList = paths ? [...paths] : list.map((f) => f.webkitRelativePath || f.name);
  imageSet = { files: list.slice(), paths: pathList.slice() };
  const rejected = [];
  if (list.length > IMAGE_LIMITS.maxFiles) {
    rejected.push(`${list.length - IMAGE_LIMITS.maxFiles} files over the ${IMAGE_LIMITS.maxFiles}-file limit were skipped`);
    list.length = IMAGE_LIMITS.maxFiles;
    pathList.length = IMAGE_LIMITS.maxFiles;
  }
  const S = IMAGE_LIMITS.size, per = 3 * S * S;
  const kept = [], names = [];
  const canvas = new OffscreenCanvas(S, S);
  const ctx = canvas.getContext('2d', { willReadFrequently: true });
  let done = 0;
  for (const [idx, f] of list.entries()) {
    done++;
    if (done % 25 === 0) status(`Decoding ${done} / ${list.length}…`);
    if (f.size > IMAGE_LIMITS.maxBytes) { rejected.push(`${f.name}: ${(f.size / 1048576).toFixed(1)} MB is over the 20 MB limit`); continue; }
    let bmp;
    try { bmp = await createImageBitmap(f); } catch { rejected.push(`${f.name}: not an image this browser can decode`); continue; }
    if (bmp.width * bmp.height > IMAGE_LIMITS.maxPixels) { rejected.push(`${f.name}: ${((bmp.width * bmp.height) / 1e6).toFixed(0)} MP is over the 40 MP limit`); bmp.close(); continue; }
    // center square crop, then resize to S x S
    const s = Math.min(bmp.width, bmp.height);
    ctx.clearRect(0, 0, S, S);
    ctx.drawImage(bmp, (bmp.width - s) / 2, (bmp.height - s) / 2, s, s, 0, 0, S, S);
    bmp.close();
    const px = ctx.getImageData(0, 0, S, S).data;
    const chw = new Float32Array(per);
    for (let i = 0; i < S * S; i++) for (let c = 0; c < 3; c++) chw[c * S * S + i] = px[4 * i + c] / 255;
    kept.push(chw);
    names.push(pathList[idx] || f.name);
  }
  if (kept.length < 16) {
    if (source === 'paste' && !rejected.length) {
      status([`${kept.length} pasted image${kept.length === 1 ? '' : 's'} so far; at least 16 are needed.`, [`Paste or drop ${16 - kept.length} more to use them.`]], 'warn');
      return;
    }
    status([`Only ${kept.length} usable images; at least 16 are needed.`, rejected.slice(0, 8)], 'bad');
    notify('bad', 'Images not loaded', `Only ${kept.length} usable images; at least 16 are needed.`);
    return;
  }
  const images = new Float32Array(kept.length * per);
  kept.forEach((a, i) => images.set(a, i * per));
  // labels from parent folder names when a folder of folders was chosen
  const folders = names.map((n) => { const parts = n.split('/'); return parts.length > 2 ? parts[parts.length - 2] : null; });
  let labels = null, classes = null;
  if (folders.every((x) => x) && new Set(folders).size >= 2) {
    classes = [...new Set(folders)].sort();
    labels = Int32Array.from(folders, (x) => classes.indexOf(x));
  }
  app.data = { kind: 'images', n: kept.length, size: S, images, labels, classes, names };
  app.config.data.kind = 'images';
  app.config.data.max_items = Math.max(16, kept.length);
  app.backend.setData(app.data);
  status([`${kept.length} images loaded locally${classes ? `, ${classes.length} classes from folder names` : ', no labels'}.`, rejected.length ? [`${rejected.length} rejected:`, ...rejected.slice(0, 6)] : []], rejected.length ? 'warn' : 'good');
  notify(rejected.length ? 'warn' : 'good', `${kept.length} images loaded`, rejected.length ? `${rejected.length} file${rejected.length > 1 ? 's were' : ' was'} rejected; see the Data tab.` : (classes ? `${classes.length} classes from folder names.` : 'No labels: Evaluate needs a folder of class sub-folders.'));
  current = 0;
  renderThumbs();
  renderViews();
  app.emit('data');
}

async function loadCsv(file) {
  const panel = $('#series-panel');
  panel.hidden = false;
  if (file.size > CSV_LIMITS.maxBytes) { status(`${file.name} is ${(file.size / 1048576).toFixed(1)} MB; the limit is 20 MB.`, 'bad'); notify('bad', 'CSV not loaded', 'The file is over the 20 MB limit.'); return; }
  try {
    series = parseNumericCsv(await file.text());
  } catch (e) {
    status(`${file.name} was not loaded: ${e.message}.`, 'bad');
    notify('bad', 'CSV not loaded', e.message);
    return;
  }
  status(`${series.rows} rows × ${series.columns.length} numeric columns from ${file.name} (preview only in the web demo).`, 'good');
  notify('good', 'Time series loaded', `${series.rows} rows × ${series.columns.length} columns.`);
  renderSeries();
}

function renderSeries() {
  if (!series) return;
  const w = valueOf('series-window');
  const win = Math.max(8, Math.min(w ?? app.config.data.series_window ?? 128, series.rows));
  const colors = ['--flow-view', '--flow-encoder', '--flow-predictor'];
  const lines = series.data.slice(0, 3).map((d, i) => ({ data: d, color: colors[i] }));
  const rng = mulberry32(viewSeed);
  const start = Math.floor(rng() * Math.max(1, series.rows - win));
  seriesPlots.full.set(lines, [start, win]);
  const x = series.data[0].subarray(start, start + win);
  const g = seriesView(x, win, [0.8, 1.0], rng, app.config.data.mask_ratio);
  const l = seriesView(x, Math.max(8, Math.floor(win / 2)), [0.3, 0.6], rng, app.config.data.mask_ratio);
  seriesPlots.g.set([{ data: g.data, color: '--flow-view', width: 2 }]);
  seriesPlots.l.set([{ data: l.data, color: '--flow-view', width: 2 }]);
  $('#series-caption').textContent = `${series.columns.slice(0, 3).join(', ')}${series.columns.length > 3 ? ` and ${series.columns.length - 3} more` : ''}; window ${win} from row ${start}`;
}

/** Desktop tier: the native folder dialog (Rust side) grants the backend read access to one folder;
 *  the page never gets file-system access itself. Returns the canonical path or null. */
async function pickFolder() {
  const invoke = globalThis.__TAURI__ && globalThis.__TAURI__.core && globalThis.__TAURI__.core.invoke;
  if (!invoke) throw new Error('the desktop shell is not available');
  return invoke('pick_data_folder');
}

const DESKTOP_KINDS = { images: 'images', video: 'video', series: 'timeseries' };

async function useDesktopFolder(kind) {
  let path;
  try { path = await pickFolder(); } catch (e) { status(`Folder not opened: ${e.message || e}`, 'bad'); return; }
  if (!path) return;                                   // dialog cancelled
  const prev = { kind: app.config.data.kind, path: app.config.data.path, arch: app.config.model.arch };
  app.config.data.kind = DESKTOP_KINDS[kind];
  app.config.data.path = path;
  if (kind === 'video' && !/^video-/.test(app.config.model.arch)) app.config.model.arch = 'video-convnet';
  if (kind === 'series' && !/^series-/.test(app.config.model.arch)) app.config.model.arch = 'series-conv';
  if (kind === 'images' && /^(video|series)-/.test(app.config.model.arch)) app.config.model.arch = 'convnet-small';
  status(`Reading ${path}…`);
  try {
    const r = await app.backend.preview(app.config, 6);
    const info = r.info || {};
    const n = info.count ?? r.samples.length;
    app.data = { kind: `desktop-${DESKTOP_KINDS[kind]}`, n, path, samples: r.samples, info };
    app.backend.setData(app.data);
    showDesktopPreview(r.samples);
    status([`Using ${path} on this computer${Number.isFinite(n) ? ` (${n} items)` : ''}.`, info.classes && info.classes.length ? [`classes from folder names: ${info.classes.slice(0, 12).join(', ')}`] : []], 'good');
    app.emit('config');
    app.emit('data');
  } catch (e) {
    Object.assign(app.config.data, { kind: prev.kind, path: prev.path });
    app.config.model.arch = prev.arch;
    status(`${path} was not used: ${e.message || e}`, 'bad');
  }
}

function showDesktopPreview(samples) {
  const box = clear($('#data-thumbs'));
  const orig = clear($('#view-original')), g = clear($('#view-globals')), l = clear($('#view-locals'));
  const imgEl = (src, alt) => el('img', { attrs: { src, alt, width: '64', height: '64' } });
  samples.forEach((s, i) => { if (s.original) box.append(el('figure', { class: 'view-cell' }, imgEl(s.original, `sample ${i}`))); });
  const s0 = samples[0];
  if (!s0) return;
  if (s0.original) {
    orig.append(el('figure', { class: 'view-cell' }, imgEl(s0.original, 'sample 0'), el('figcaption', { text: 'sample 0' })));
    s0.globals.forEach((v, i) => g.append(el('figure', { class: 'view-cell global' }, imgEl(v, `global view ${i + 1}`), el('figcaption', {}, el('span', { class: 'swatch c-view' }), `global ${i + 1}`))));
    s0.locals.forEach((v, i) => l.append(el('figure', { class: 'view-cell local' }, imgEl(v, `local view ${i + 1}`), el('figcaption', {}, el('span', { class: 'swatch c-view' }), `local ${i + 1}`))));
    $('#views-caption').textContent = 'Sample 0 from the chosen folder (views made by the backend)';
  } else {
    $('#views-caption').textContent = `Series sample 0: ${s0.globals.length} global and ${s0.locals.length} local windows (made by the backend)`;
  }
}

function setKind(kind) {
  const r = $(`input[name="dataset"][value="${kind}"]`);
  if (r && !r.checked) { r.checked = true; switchKind(kind); }
}

function switchKind(kind) {
  const dz = $('#dropzone');
  dz.hidden = kind === 'video';
  $('#dropzone-title').textContent = kind === 'series' ? 'Drop a CSV file here' : kind === 'images' ? 'Drop images or a folder here' : 'Use your own images: drop them here';
  $('#dropzone-hint').textContent = kind === 'series' ? 'Or press Enter to choose one.' : 'A folder works too. Or paste an image, or press Enter to choose files.';
  $('#data-shapes').hidden = kind !== 'shapes';
  $('#data-images').hidden = kind !== 'images';
  $('#data-series').hidden = kind !== 'series';
  $('#data-video').hidden = kind !== 'video';
  $('#series-panel').hidden = kind !== 'series' || !series;
  if (kind === 'shapes') useShapes();
  if (kind === 'images' && app.data.kind !== 'images') status('Choose image files (or a folder) to use your own data.');
  if (kind === 'series') status(series ? 'Time series loaded.' : 'Choose a CSV file of numbers.');
  if (kind === 'video') status('Choose a folder of short video clips.');
  refreshValidity();
}

function currentKind() { const r = $('input[name="dataset"]:checked'); return r ? r.value : 'shapes'; }

/** Files or folders dropped on the zone: CSV -> time series, images -> My images. */
async function onDrop(dt) {
  let got;
  try { got = await collectDropped(dt, { maxFiles: IMAGE_LIMITS.maxFiles + 1000 }); } catch (e) { status(`Nothing was loaded: ${e.message}`, 'bad'); return; }
  const csv = got.files.filter(isCsvFile);
  const imgIdx = got.files.map((f, i) => (isImageFile(f) ? i : -1)).filter((i) => i >= 0);
  if (csv.length && (currentKind() === 'series' || !imgIdx.length)) {
    setKind('series');
    if (csv.length > 1) notify('info', `${csv.length} CSV files dropped`, `Using ${csv[0].name}; the web demo previews one series at a time.`);
    loadCsv(csv[0]);
    return;
  }
  if (!imgIdx.length) {
    status(got.files.length ? 'Nothing usable was dropped: no image or CSV files.' : 'Nothing was dropped.', 'bad');
    notify('bad', 'Nothing loaded', 'Drop image files, a folder of images, or a CSV file.');
    return;
  }
  setKind('images');
  const skipped = got.files.length - imgIdx.length;
  if (skipped) notify('info', `${skipped} non-image file${skipped > 1 ? 's' : ''} ignored`);
  if (got.truncated) notify('warn', 'Folder too large', `Only the first ${got.files.length} files were read.`);
  await loadImages(imgIdx.map((i) => got.files[i]), { paths: imgIdx.map((i) => got.paths[i]), source: 'drop' });
}

/** Pasted images are added to the current set (one screenshot at a time works). */
async function onPaste(files) {
  const imgs = files.filter(isImageFile);
  if (!imgs.length) return false;
  const base = app.data.kind === 'images' || currentKind() === 'images' ? imageSet : { files: [], paths: [] };
  const stamp = Date.now();
  const named = imgs.map((f, i) => new File([f], `pasted-${stamp}-${base.files.length + i + 1}.${(f.type.split('/')[1] || 'png').replace(/[^a-z0-9]/g, '')}`, { type: f.type }));
  setKind('images');
  await loadImages([...base.files, ...named], { paths: [...base.paths, ...named.map((f) => f.name)], source: 'paste' });
  return true;
}

function setupDropzone() {
  const dz = $('#dropzone');
  let depth = 0;
  const hasFiles = (e) => !!e.dataTransfer && [...(e.dataTransfer.types || [])].includes('Files');
  dz.addEventListener('dragenter', (e) => { if (!hasFiles(e)) return; e.preventDefault(); depth++; dz.classList.add('is-over'); });
  dz.addEventListener('dragover', (e) => { if (!hasFiles(e)) return; e.preventDefault(); e.dataTransfer.dropEffect = 'copy'; dz.classList.add('is-over'); });
  dz.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) dz.classList.remove('is-over'); });
  dz.addEventListener('drop', (e) => {
    e.preventDefault();
    depth = 0;
    dz.classList.remove('is-over');
    onDrop(e.dataTransfer);
  });
  // a file dropped next to the zone must not navigate away from the app
  window.addEventListener('dragover', (e) => { if (hasFiles(e) && !dz.contains(e.target)) { e.preventDefault(); e.dataTransfer.dropEffect = 'none'; } });
  window.addEventListener('drop', (e) => { if (hasFiles(e) && !dz.contains(e.target)) e.preventDefault(); });
  // keyboard alternative: Enter / Space open the matching file picker
  dz.addEventListener('keydown', (e) => {
    if (e.key !== 'Enter' && e.key !== ' ') return;
    e.preventDefault();
    (currentKind() === 'series' ? $('#csv-file') : $('#image-files')).click();
  });
  dz.addEventListener('click', () => (currentKind() === 'series' ? $('#csv-file') : $('#image-files')).click());
  // paste: on the focused zone, or anywhere on the Data tab outside a text field
  document.addEventListener('paste', (e) => {
    if ($('#panel-data').hidden || dz.hidden) return;
    const t = e.target;
    if (t !== dz && t && t.closest && t.closest('input, textarea, select, [contenteditable="true"]')) return;
    const files = [...((e.clipboardData && e.clipboardData.files) || [])];
    if (!files.length) {
      if (t === dz) status('The clipboard has no image. Copy an image (not its file name) and paste again.', 'warn');
      return;
    }
    e.preventDefault();
    onPaste(files).then((ok) => { if (!ok) status('Only images can be pasted. Use Choose CSV for time series.', 'warn'); });
  });
}

export function init(a) {
  app = a;
  app.data = { kind: 'synthetic-shapes', n: 2048 };
  const defaults = webDefaultConfig();
  if (app.tier === 'desktop') $('#shapes-n').max = '65536';   // the backend generates Shapes there
  field('shapes-n', { path: 'data.max_items', gate: 'run', section: 'data-shapes', def: () => defaults.data.max_items, commit: useShapes,
    rangeMsg: (lo, hi, tier) => `The ${tier === 'desktop' ? 'desktop app' : 'web demo'} generates ${lo} to ${hi} Shapes images.` });
  field('data-seed', { path: 'seed', gate: 'run', section: 'data-shapes', def: () => defaults.seed, commit: useShapes });
  field('series-window', { path: 'data.series_window', gate: 'run', section: 'data-series', def: () => defaults.data.series_window,
    commit: () => { app.config.data.series_window = valueOf('series-window'); renderSeries(); app.emit('config'); } });
  const augs = [['aug-jitter', 'color_jitter'], ['aug-gray', 'grayscale_p'], ['aug-flip', 'flip_p'], ['aug-mask', 'mask_ratio']];
  const applyAug = () => {
    for (const [id, key] of augs) { const v = valueOf(id); if (v !== null) app.config.data[key] = v; }
    renderViews(); if (series) renderSeries(); app.emit('config');
  };
  for (const [id, key] of augs) {
    field(id, { path: `data.${key}`, gate: 'run', section: 'data-aug', def: () => defaults.data[key], commit: applyAug });
    setValue(id, app.config.data[key]);
  }
  onValidity(() => {
    const bad = invalidFor('run').filter((x) => x.tab === 'Data');
    gate($('#regen-views'), bad.length ? fixMessage(bad, 'make new views', 'Data') : '');
  });
  $$('input[name="dataset"]').forEach((r) => r.addEventListener('change', () => switchKind(r.value)));
  $('#regen-views').addEventListener('click', () => { viewSeed += 1; renderViews(); if (series) renderSeries(); });
  $('#image-files').addEventListener('change', (e) => { if (e.target.files.length) loadImages(e.target.files); e.target.value = ''; });
  $('#image-folder').addEventListener('change', (e) => {
    const imgs = [...e.target.files].filter(isImageFile);
    if (imgs.length) loadImages(imgs); else status('That folder has no image files.', 'bad');
    e.target.value = '';
  });
  setupDropzone();
  $('#csv-file').addEventListener('change', (e) => { if (e.target.files[0]) loadCsv(e.target.files[0]); e.target.value = ''; });
  if (app.backend.tier === 'desktop') {
    $$('.desktop-only').forEach((x) => { x.hidden = false; });
    $('#desktop-folder-images').addEventListener('click', () => useDesktopFolder('images'));
    $('#desktop-folder-video').addEventListener('click', () => useDesktopFolder('video'));
    $('#desktop-folder-series').addEventListener('click', () => useDesktopFolder('series'));
  }
  seriesPlots = {
    full: new SeriesPlot($('#series-canvas')),
    g: new SeriesPlot($('#series-global')),
    l: new SeriesPlot($('#series-local')),
  };
  app.on('config-loaded', () => {
    for (const [id, key] of augs) setValue(id, app.config.data[key]);
    setValue('data-seed', app.config.seed);
    setValue('series-window', app.config.data.series_window ?? defaults.data.series_window);
    // out-of-range values (e.g. 5000 Shapes images in the web demo) are shown with their
    // message and block Start; they are never clamped to fit the slider
    if (app.data.kind === 'synthetic-shapes') { setValue('shapes-n', app.config.data.max_items); useShapes(); }
    if (series) renderSeries();
  });
  // start on Shapes even if the browser restored another radio from a previous visit
  $('input[name="dataset"][value="shapes"]').checked = true;
  switchKind('shapes');
}
