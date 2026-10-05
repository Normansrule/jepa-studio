// Landing page: theme toggle, the hero figure (a real Epps-Pulley statistic on samples that
// morph from a collapsed cluster to N(0, 1)), and the web vs desktop capability table.
import { epConstants, epStat, epExpectedUnderNull } from './engine/lejepa.js';
import { gaussian, mulberry32 } from './engine/rng.js';
import { cssVar } from './ui/charts.js';

const root = document.documentElement;
function safe(fn) { try { return fn(); } catch { return null; } }

function setupTheme(onChange) {
  const btn = document.getElementById('theme-toggle');
  const saved = safe(() => localStorage.getItem('jepa-theme'));
  if (saved === 'light' || saved === 'dark') root.dataset.theme = saved;
  const dark = () => (root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches);
  const label = () => btn.setAttribute('aria-label', dark() ? 'Switch to light theme' : 'Switch to dark theme');
  label();
  btn.addEventListener('click', () => { root.dataset.theme = dark() ? 'light' : 'dark'; safe(() => localStorage.setItem('jepa-theme', root.dataset.theme)); label(); onChange(); });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { label(); onChange(); });
}

// ------------------------------------------------------------------ hero figure
const N = 512;
const g = gaussian(mulberry32(7));
// target: N(0, 1) samples, standardized so the demo lands near the statistic's expected value
const target = (() => {
  const t = Float64Array.from({ length: N }, () => g());
  let m = 0, v = 0;
  for (const x of t) m += x;
  m /= N;
  for (const x of t) v += (x - m) ** 2;
  const sd = Math.sqrt(v / N);
  return t.map((x) => (x - m) / sd);
})();
// "collapsed" start: almost everything in one tight cluster, a few stragglers
const start = Float64Array.from({ length: N }, (_, i) => (i % 9 === 0 ? 1.4 + 0.3 * g() : 0.55 + 0.06 * g()));
const ep = epConstants(3.0, 17);
let frac = 0, raf = null;

function samples(a) {
  const e = a * a * (3 - 2 * a); // smoothstep
  const x = new Float64Array(N);
  for (let i = 0; i < N; i++) x[i] = (1 - e) * start[i] + e * target[i];
  return x;
}

function drawHero() {
  const c = document.getElementById('hero-canvas');
  const dpr = window.devicePixelRatio || 1;
  const w = c.clientWidth, h = c.clientHeight;
  if (!w || !h) return;
  c.width = Math.round(w * dpr); c.height = Math.round(h * dpr);
  const ctx = c.getContext('2d');
  ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
  ctx.clearRect(0, 0, w, h);
  const x = samples(frac);
  const bins = 40, lo = -4, hi = 4, bw = (hi - lo) / bins;
  const hist = new Array(bins).fill(0);
  for (const v of x) { const k = Math.floor((Math.max(lo, Math.min(hi - 1e-9, v)) - lo) / bw); hist[k] += 1; }
  const dens = hist.map((k) => k / N / bw);
  const pad = { l: 8, r: 8, t: 8, b: 22 };
  const ymax = Math.max(0.5, ...dens) * 1.05;
  const X = (v) => pad.l + ((v - lo) / (hi - lo)) * (w - pad.l - pad.r);
  const Y = (v) => h - pad.b - (v / ymax) * (h - pad.t - pad.b);
  ctx.strokeStyle = cssVar('--line'); ctx.lineWidth = 1;
  ctx.beginPath(); ctx.moveTo(pad.l, Y(0) + 0.5); ctx.lineTo(w - pad.r, Y(0) + 0.5); ctx.stroke();
  ctx.fillStyle = cssVar('--flow-loss-sigreg');
  const colW = (w - pad.l - pad.r) / bins;
  dens.forEach((d, i) => {
    if (d <= 0) return;
    const x0 = pad.l + i * colW + 1, top = Y(d), bh = Y(0) - top;
    const r = Math.min(3, (colW - 2) / 2, bh);
    ctx.beginPath(); ctx.moveTo(x0, Y(0)); ctx.lineTo(x0, top + r); ctx.arcTo(x0, top, x0 + r, top, r);
    ctx.lineTo(x0 + colW - 2 - r, top); ctx.arcTo(x0 + colW - 2, top, x0 + colW - 2, top + r, r); ctx.lineTo(x0 + colW - 2, Y(0)); ctx.fill();
  });
  ctx.strokeStyle = cssVar('--ink'); ctx.lineWidth = 2;
  ctx.beginPath();
  for (let i = 0; i <= 160; i++) {
    const v = lo + ((hi - lo) * i) / 160;
    const y = Math.exp(-0.5 * v * v) / Math.sqrt(2 * Math.PI);
    if (i === 0) ctx.moveTo(X(v), Y(y)); else ctx.lineTo(X(v), Y(y));
  }
  ctx.stroke();
  ctx.fillStyle = cssVar('--ink-3'); ctx.font = '12px system-ui, sans-serif'; ctx.textAlign = 'center';
  for (const t of [-3, -2, -1, 0, 1, 2, 3]) ctx.fillText(String(t), X(t), h - 6);
  ctx.textAlign = 'left'; ctx.fillStyle = cssVar('--ink-2');
  ctx.fillText('N(0, 1)', X(1.1), Y(0.26));
  const stat = epStat(x, N, ep, null);
  document.getElementById('hero-ep').textContent = `${stat.toFixed(2)} (a Gaussian batch averages ${epExpectedUnderNull().toFixed(2)})`;
}

function play() {
  cancelAnimationFrame(raf);
  if (matchMedia('(prefers-reduced-motion: reduce)').matches) { frac = 1; drawHero(); return; }
  const t0 = performance.now(), dur = 3200;
  frac = 0;
  const tick = (t) => {
    frac = Math.min(1, Math.max(0, (t - t0 - 500) / dur));
    drawHero();
    if (frac < 1) raf = requestAnimationFrame(tick);
  };
  raf = requestAnimationFrame(tick);
}

// ------------------------------------------------------------------ capability table
const CAPS = [
  ['Pretrain on built-in Shapes', ['yes', 'mlp-tiny, 16×16 inputs, a few minutes'], ['yes', 'ConvNet / ResNet / ViT, full resolution']],
  ['Pretrain on your images', ['part', 'up to 2000 files, resized to 32×32'], ['yes', 'folders of any size, streamed from disk']],
  ['Time series and video', ['part', 'CSV preview and views only'], ['yes', 'series-conv and video-convnet encoders']],
  ['GPU acceleration', ['part', 'WebGPU for the first layer when available'], ['yes', 'CUDA, Apple MPS, mixed precision, torch.compile']],
  ['Inspect: PCA, isotropy, collapse check, neighbours, saliency', ['yes', ''], ['yes', 'plus ViT attention maps']],
  ['Evaluate: linear probe, k-NN, baselines', ['yes', 'same protocol, smaller model'], ['yes', '']],
  ['World-model planning', ['yes', 'pre-trained models, CEM in the browser'], ['yes', 'train your own world models']],
  ['Explain panel', ['part', 'keyword search over the docs'], ['yes', 'local assistant over docs and logs']],
  ['Export: safetensors, .npy, report, bundle', ['yes', ''], ['yes', 'plus checkpoints and resume']],
  ['Your data leaves the device', ['no', 'never'], ['no', 'never; binds to 127.0.0.1 only']],
];
const ICON = { yes: 'M4 12l5 5L20 6', part: 'M5 12h14', no: 'M6 6l12 12M18 6L6 18' };
const WORD = { yes: 'Yes', part: 'Partly', no: 'No' };

function cell(kind, detail, label) {
  const td = document.createElement('td');
  td.dataset.label = label;
  const s = document.createElement('span');
  s.className = kind;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24'); svg.setAttribute('aria-hidden', 'true');
  const p = document.createElementNS(svg.namespaceURI, 'path');
  p.setAttribute('d', ICON[kind]); p.setAttribute('fill', 'none'); p.setAttribute('stroke', 'currentColor'); p.setAttribute('stroke-width', '2.5'); p.setAttribute('stroke-linecap', 'round');
  svg.append(p);
  s.append(svg, document.createTextNode(WORD[kind]));
  td.append(s);
  if (detail) { const d = document.createElement('span'); d.className = 'detail'; d.textContent = detail; td.append(d); }
  return td;
}

function buildTable() {
  const body = document.getElementById('cap-body');
  for (const [name, web, desk] of CAPS) {
    const tr = document.createElement('tr');
    const th = document.createElement('th');
    th.scope = 'row';
    th.textContent = name;
    tr.append(th, cell(web[0], web[1], 'Web demo'), cell(desk[0], desk[1], 'Desktop app'));
    body.append(tr);
  }
}

setupTheme(drawHero);
buildTable();
document.getElementById('hero-replay').addEventListener('click', play);
new ResizeObserver(drawHero).observe(document.getElementById('hero-canvas'));
play();
document.documentElement.dataset.ready = 'true';
