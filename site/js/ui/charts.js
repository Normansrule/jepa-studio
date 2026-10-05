// Tiny canvas chart helpers (no chart library). Rules from docs/STYLE.md:
//   * colors come from CSS custom properties, read at draw time, so theme switches just redraw
//   * 2px lines, recessive grid, one y-axis per chart, text in ink tokens (never series color)
//   * every chart has a hover read-out; >= 2 series always get a legend
export function cssVar(name, el = document.documentElement) {
  return getComputedStyle(el).getPropertyValue(name).trim() || '#888';
}

export function fmt(v, digits = 3) {
  if (v === null || v === undefined || Number.isNaN(v)) return '–';
  const a = Math.abs(v);
  if (a !== 0 && (a < 1e-3 || a >= 1e5)) return v.toExponential(2);
  return Number(v.toPrecision(digits)).toString();
}

/** Tick labels with just enough decimals for the tick spacing (no duplicate labels). */
export function tickFormatter(step) {
  const a = Math.abs(step);
  if (!a || !Number.isFinite(a)) return (v) => fmt(v);
  if (a < 1e-3 || a >= 1e5) return (v) => (v === 0 ? '0' : v.toExponential(1));
  const dec = Math.max(0, Math.min(6, Math.ceil(-Math.log10(a)) + (a / 10 ** Math.floor(Math.log10(a)) < 2 ? 1 : 0)));
  return (v) => v.toFixed(dec);
}

const charts = new Set();
/** Redraw every live chart (theme change). */
export function redrawAll() { for (const c of charts) c.draw(); }

class BaseChart {
  constructor(canvas, opts = {}) {
    this.canvas = canvas;
    this.opts = opts;
    this.ctx = canvas.getContext('2d');
    this.pad = { l: 44, r: 12, t: 10, b: 22, ...(opts.pad || {}) };
    this.tip = document.createElement('div');
    this.tip.className = 'chart-tip';
    this.tip.hidden = true;
    this.tip.setAttribute('role', 'status');
    canvas.parentElement.appendChild(this.tip);
    canvas.addEventListener('pointermove', (e) => this.hover(e));
    canvas.addEventListener('pointerleave', () => { this.hoverX = null; this.tip.hidden = true; this.draw(); });
    this.ro = new ResizeObserver(() => this.draw());
    this.ro.observe(canvas);
    charts.add(this);
  }

  size() {
    const dpr = window.devicePixelRatio || 1;
    const w = Math.max(10, this.canvas.clientWidth), h = Math.max(10, this.canvas.clientHeight);
    if (this.canvas.width !== Math.round(w * dpr) || this.canvas.height !== Math.round(h * dpr)) {
      this.canvas.width = Math.round(w * dpr);
      this.canvas.height = Math.round(h * dpr);
    }
    this.ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { w, h };
  }

  frame(w, h, y0, y1, ticks = 4, fmtY = null, log = false) {
    const { ctx, pad } = this;
    ctx.clearRect(0, 0, w, h);
    if (!fmtY) fmtY = log ? fmt : tickFormatter((y1 - y0) / ticks);
    ctx.font = '11px system-ui, sans-serif';
    ctx.textBaseline = 'middle';
    ctx.textAlign = 'right';
    const grid = cssVar('--line'), ink3 = cssVar('--ink-3');
    ctx.lineWidth = 1;
    for (let i = 0; i <= ticks; i++) {
      const f = i / ticks;
      const v = log ? 10 ** (Math.log10(y0) + f * (Math.log10(y1) - Math.log10(y0))) : y0 + f * (y1 - y0);
      const y = h - pad.b - f * (h - pad.t - pad.b);
      ctx.strokeStyle = grid;
      ctx.beginPath(); ctx.moveTo(pad.l, Math.round(y) + 0.5); ctx.lineTo(w - pad.r, Math.round(y) + 0.5); ctx.stroke();
      ctx.fillStyle = ink3;
      ctx.fillText(fmtY(v, 2), pad.l - 6, y);
    }
  }

  hover() {}
}

/** Multi-series line chart over a shared numeric x (e.g. step). */
export class LineChart extends BaseChart {
  constructor(canvas, opts) {
    super(canvas, opts);
    this.series = opts.series; // [{key, label, color: '--css-var'}]
    this.x = [];
    this.data = Object.fromEntries(this.series.map((s) => [s.key, []]));
    this.hidden = new Set();
  }

  push(x, values) {
    this.x.push(x);
    for (const s of this.series) this.data[s.key].push(values[s.key] ?? null);
    this.scheduleDraw();
  }

  reset() { this.x = []; for (const s of this.series) this.data[s.key] = []; this.draw(); }

  scheduleDraw() {
    if (this.pending) return;
    this.pending = true;
    requestAnimationFrame(() => { this.pending = false; this.draw(); });
  }

  toggle(key) { if (this.hidden.has(key)) this.hidden.delete(key); else this.hidden.add(key); this.draw(); }

  bounds() {
    let lo = Infinity, hi = -Infinity;
    for (const s of this.series) {
      if (this.hidden.has(s.key)) continue;
      for (const v of this.data[s.key]) if (v !== null && Number.isFinite(v)) { lo = Math.min(lo, v); hi = Math.max(hi, v); }
    }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (this.opts.log) { lo = Math.max(lo, 1e-6); hi = Math.max(hi, lo * 1.5); return [lo / 1.2, hi * 1.2]; }
    if (this.opts.zero) lo = Math.min(0, lo);
    if (hi - lo < 1e-12) { hi += 0.5; lo -= 0.5; }
    const m = (hi - lo) * 0.08;
    return [this.opts.zero ? lo : lo - m, hi + m];
  }

  draw() {
    const { w, h } = this.size();
    const { ctx, pad } = this;
    const n = this.x.length;
    if (!n) {
      ctx.clearRect(0, 0, w, h);
      ctx.font = '12px system-ui, sans-serif'; ctx.textBaseline = 'middle';
      ctx.fillStyle = cssVar('--ink-3'); ctx.textAlign = 'center';
      ctx.fillText(this.opts.empty || 'No data yet', w / 2, h / 2);
      return;
    }
    const [y0, y1] = this.bounds();
    this.frame(w, h, y0, y1, 4, this.opts.fmtY || null, this.opts.log);
    const x0 = this.x[0], x1 = Math.max(this.x[n - 1], x0 + 1);
    const X = (x) => pad.l + ((x - x0) / (x1 - x0)) * (w - pad.l - pad.r);
    const Y = this.opts.log
      ? (v) => h - pad.b - ((Math.log10(Math.max(v, 1e-9)) - Math.log10(y0)) / (Math.log10(y1) - Math.log10(y0))) * (h - pad.t - pad.b)
      : (v) => h - pad.b - ((v - y0) / (y1 - y0)) * (h - pad.t - pad.b);
    ctx.fillStyle = cssVar('--ink-3'); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillText(String(x0), pad.l, h - 6);
    ctx.textAlign = 'right';
    ctx.fillText(`${this.opts.xLabel || 'step'} ${this.x[n - 1]}`, w - pad.r, h - 6);
    for (const s of this.series) {
      if (this.hidden.has(s.key)) continue;
      ctx.strokeStyle = cssVar(s.color);
      ctx.lineWidth = s.dash ? 1.5 : 2; ctx.lineJoin = 'round';
      ctx.setLineDash(s.dash ? [5, 4] : []);
      ctx.beginPath();
      let started = false;
      const ys = this.data[s.key];
      for (let i = 0; i < n; i++) {
        const v = ys[i];
        if (v === null || !Number.isFinite(v)) { started = false; continue; }
        if (!started) { ctx.moveTo(X(this.x[i]), Y(v)); started = true; } else ctx.lineTo(X(this.x[i]), Y(v));
      }
      ctx.stroke();
      ctx.setLineDash([]);
    }
    if (this.hoverX !== null && this.hoverX !== undefined) {
      const i = this.hoverX;
      const xx = X(this.x[i]);
      ctx.strokeStyle = cssVar('--line-strong'); ctx.lineWidth = 1;
      ctx.beginPath(); ctx.moveTo(xx + 0.5, pad.t); ctx.lineTo(xx + 0.5, h - pad.b); ctx.stroke();
      for (const s of this.series) {
        const v = this.data[s.key][i];
        if (this.hidden.has(s.key) || v === null || !Number.isFinite(v)) continue;
        ctx.fillStyle = cssVar('--surface'); ctx.strokeStyle = cssVar(s.color); ctx.lineWidth = 2;
        ctx.beginPath(); ctx.arc(xx, Y(v), 4, 0, 2 * Math.PI); ctx.fill(); ctx.stroke();
      }
    }
  }

  hover(e) {
    const n = this.x.length;
    if (!n) return;
    const r = this.canvas.getBoundingClientRect();
    const px = e.clientX - r.left;
    const { pad } = this;
    const x0 = this.x[0], x1 = Math.max(this.x[n - 1], x0 + 1);
    const xv = x0 + ((px - pad.l) / (r.width - pad.l - pad.r)) * (x1 - x0);
    let best = 0;
    for (let i = 1; i < n; i++) if (Math.abs(this.x[i] - xv) < Math.abs(this.x[best] - xv)) best = i;
    this.hoverX = best;
    const lines = [`${this.opts.xLabel || 'step'} ${this.x[best]}`];
    for (const s of this.series) if (!this.hidden.has(s.key)) lines.push(`${s.label}: ${fmt(this.data[s.key][best], 4)}`);
    this.tip.textContent = lines.join('  ·  ');
    this.tip.hidden = false;
    this.tip.style.left = `${Math.min(px + 12, r.width - this.tip.offsetWidth - 4)}px`;
    this.tip.style.top = '0px';
    this.draw();
  }
}

/** Vertical bars (eigen spectrum) with optional horizontal reference line. */
export class BarChart extends BaseChart {
  constructor(canvas, opts) { super(canvas, { ...opts, pad: { l: 44, r: 10, t: 10, b: 22, ...(opts.pad || {}) } }); this.values = []; }
  set(values, ref = null) { this.values = values; this.ref = ref; this.draw(); }
  draw() {
    const { w, h } = this.size();
    const { ctx, pad } = this;
    const v = this.values;
    let hi = Math.max(1e-9, ...v, this.ref || 0) * 1.08;
    this.frame(w, h, 0, hi, 4, this.opts.fmtY || null);
    if (!v.length) {
      ctx.fillStyle = cssVar('--ink-3'); ctx.textAlign = 'center';
      ctx.fillText(this.opts.empty || 'No data yet', (pad.l + w - pad.r) / 2, (pad.t + h - pad.b) / 2);
      return;
    }
    const bw = (w - pad.l - pad.r) / v.length;
    const Y = (x) => h - pad.b - (x / hi) * (h - pad.t - pad.b);
    ctx.fillStyle = cssVar(this.opts.color || '--flow-encoder');
    v.forEach((x, i) => {
      const gap = Math.min(2, bw * 0.2);
      const bx = pad.l + i * bw + gap / 2, bh = Math.max(1, h - pad.b - Y(x));
      roundTop(ctx, bx, h - pad.b - bh, Math.max(1, bw - gap), bh, Math.min(3, (bw - gap) / 2));
      if (this.hoverI === i) { ctx.save(); ctx.globalAlpha = 0.25; ctx.fillStyle = cssVar('--ink'); roundTop(ctx, bx, h - pad.b - bh, Math.max(1, bw - gap), bh, 2); ctx.restore(); ctx.fillStyle = cssVar(this.opts.color || '--flow-encoder'); }
    });
    if (this.ref) {
      ctx.strokeStyle = cssVar('--ink-2'); ctx.setLineDash([4, 3]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(pad.l, Y(this.ref)); ctx.lineTo(w - pad.r, Y(this.ref)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = cssVar('--ink-2'); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom';
      ctx.fillText(this.opts.refLabel || 'reference', w - pad.r, Y(this.ref) - 3);
    }
    ctx.fillStyle = cssVar('--ink-3'); ctx.textAlign = 'left'; ctx.textBaseline = 'alphabetic';
    ctx.fillText(this.opts.xLabel || '', pad.l, h - 6);
  }
  hover(e) {
    if (!this.values.length) return;
    const r = this.canvas.getBoundingClientRect();
    const px = e.clientX - r.left;
    const bw = (r.width - this.pad.l - this.pad.r) / this.values.length;
    const i = Math.floor((px - this.pad.l) / bw);
    if (i < 0 || i >= this.values.length) { this.tip.hidden = true; return; }
    this.hoverI = i;
    this.tip.textContent = `${this.opts.itemLabel ? this.opts.itemLabel(i) : `#${i + 1}`}: ${fmt(this.values[i], 4)}`;
    this.tip.hidden = false;
    this.tip.style.left = `${Math.min(Math.max(0, px - 30), r.width - this.tip.offsetWidth - 4)}px`;
    this.tip.style.top = '0px';
    this.draw();
  }
}

function roundTop(ctx, x, y, w, h, r) {
  r = Math.max(0, Math.min(r, w / 2, h));
  ctx.beginPath();
  ctx.moveTo(x, y + h);
  ctx.lineTo(x, y + r);
  ctx.arcTo(x, y, x + r, y, r);
  ctx.lineTo(x + w - r, y);
  ctx.arcTo(x + w, y, x + w, y + r, r);
  ctx.lineTo(x + w, y + h);
  ctx.closePath();
  ctx.fill();
}

/** Grouped bars: categories x series, values in [0,1], optional chance line. */
export class GroupedBarChart extends BaseChart {
  constructor(canvas, opts) { super(canvas, { ...opts, pad: { l: 40, r: 10, t: 16, b: 44, ...(opts.pad || {}) } }); this.cats = []; }
  set(cats, rows, chance = null) { this.cats = cats; this.rows = rows; this.chance = chance; this.draw(); }
  geometry(w) {
    const { pad } = this;
    const gw = (w - pad.l - pad.r) / Math.max(1, this.cats.length);
    const ns = this.opts.series.length;
    const bw = Math.min(44, (gw * 0.7) / ns);
    return { gw, bw, ns };
  }
  draw() {
    const { w, h } = this.size();
    const { ctx, pad } = this;
    this.frame(w, h, 0, 1, 4, (v) => `${Math.round(v * 100)}%`);
    if (!this.cats.length) {
      ctx.fillStyle = cssVar('--ink-3'); ctx.textAlign = 'center';
      ctx.fillText(this.opts.empty || 'No results yet', (pad.l + w - pad.r) / 2, (pad.t + h - pad.b) / 2);
      return;
    }
    const { gw, bw, ns } = this.geometry(w);
    const Y = (x) => h - pad.b - x * (h - pad.t - pad.b);
    this.cats.forEach((cat, ci) => {
      const cx = pad.l + ci * gw + gw / 2;
      this.opts.series.forEach((s, si) => {
        const v = this.rows[ci][s.key];
        const bx = cx - (ns * bw) / 2 + si * bw + 1;
        ctx.fillStyle = cssVar(s.color);
        roundTop(ctx, bx, Y(v), bw - 2, Math.max(1, Y(0) - Y(v)), 4);
        ctx.fillStyle = cssVar('--ink');
        ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'center'; ctx.textBaseline = 'bottom';
        ctx.fillText(`${Math.round(v * 1000) / 10}`, bx + (bw - 2) / 2, Y(v) - 2);
      });
      ctx.fillStyle = cssVar('--ink-2'); ctx.textAlign = 'center'; ctx.textBaseline = 'top';
      ctx.font = '12px system-ui, sans-serif';
      wrapText(ctx, cat, cx, h - pad.b + 6, gw - 8, 14);
    });
    if (this.chance !== null) {
      ctx.strokeStyle = cssVar('--ink-2'); ctx.setLineDash([5, 4]); ctx.lineWidth = 1.5;
      ctx.beginPath(); ctx.moveTo(pad.l, Y(this.chance)); ctx.lineTo(w - pad.r, Y(this.chance)); ctx.stroke(); ctx.setLineDash([]);
      ctx.fillStyle = cssVar('--ink-2'); ctx.textAlign = 'right'; ctx.textBaseline = 'bottom'; ctx.font = '11px system-ui, sans-serif';
      ctx.fillText(`chance ${Math.round(this.chance * 100)}%`, w - pad.r, Y(this.chance) - 3);
    }
  }
  hover(e) {
    if (!this.cats.length) return;
    const r = this.canvas.getBoundingClientRect();
    const px = e.clientX - r.left;
    const { gw } = this.geometry(r.width);
    const ci = Math.floor((px - this.pad.l) / gw);
    if (ci < 0 || ci >= this.cats.length) { this.tip.hidden = true; return; }
    this.tip.textContent = `${this.cats[ci]}: ` + this.opts.series.map((s) => `${s.label} ${(this.rows[ci][s.key] * 100).toFixed(1)}%`).join(', ');
    this.tip.hidden = false;
    this.tip.style.left = `${Math.min(Math.max(0, px - 60), r.width - this.tip.offsetWidth - 4)}px`;
    this.tip.style.top = '0px';
  }
}

function wrapText(ctx, text, x, y, maxW, lh) {
  const words = String(text).split(' ');
  let line = '', row = 0;
  for (const wd of words) {
    const t = line ? `${line} ${wd}` : wd;
    if (ctx.measureText(t).width > maxW && line) { ctx.fillText(line, x, y + row * lh); line = wd; row++; } else line = t;
  }
  if (line) ctx.fillText(line, x, y + row * lh);
}

/** Histogram (density over [-4, 4]) against the N(0, 1) curve. */
export class HistChart extends BaseChart {
  constructor(canvas, opts = {}) { super(canvas, { ...opts, pad: { l: 30, r: 8, t: 8, b: 20, ...(opts.pad || {}) } }); this.bins = []; }
  set(bins) { this.bins = bins || []; this.draw(); }
  draw() {
    const { w, h } = this.size();
    const { ctx, pad } = this;
    const hi = Math.max(0.45, ...this.bins) * 1.05;
    this.frame(w, h, 0, hi, 2, (v) => v.toFixed(1));
    const X = (x) => pad.l + ((x + 4) / 8) * (w - pad.l - pad.r);
    const Y = (v) => h - pad.b - (v / hi) * (h - pad.t - pad.b);
    ctx.fillStyle = cssVar('--ink-3'); ctx.textAlign = 'center'; ctx.textBaseline = 'alphabetic'; ctx.font = '11px system-ui, sans-serif';
    for (const t of [-4, -2, 0, 2, 4]) ctx.fillText(String(t), X(t), h - 5);
    const nb = this.bins.length;
    if (nb) {
      const bw = (w - pad.l - pad.r) / nb;
      ctx.fillStyle = cssVar(this.opts.color || '--flow-loss-sigreg');
      this.bins.forEach((v, i) => roundTop(ctx, pad.l + i * bw + 1, Y(v), Math.max(1, bw - 2), Math.max(0, Y(0) - Y(v)), 2));
    }
    ctx.strokeStyle = cssVar('--ink'); ctx.lineWidth = 2;
    ctx.beginPath();
    for (let i = 0; i <= 100; i++) {
      const x = -4 + (8 * i) / 100;
      const y = Math.exp(-0.5 * x * x) / Math.sqrt(2 * Math.PI);
      if (i === 0) ctx.moveTo(X(x), Y(y)); else ctx.lineTo(X(x), Y(y));
    }
    ctx.stroke();
  }
}

/** Simple multi-line plot of raw arrays (time-series preview); x = index. */
export class SeriesPlot extends BaseChart {
  constructor(canvas, opts = {}) { super(canvas, opts); this.lines = []; }
  set(lines, highlight = null) { this.lines = lines; this.highlight = highlight; this.draw(); }
  draw() {
    const { w, h } = this.size();
    const { ctx, pad } = this;
    let lo = Infinity, hi = -Infinity, n = 0;
    for (const l of this.lines) { n = Math.max(n, l.data.length); for (const v of l.data) { lo = Math.min(lo, v); hi = Math.max(hi, v); } }
    if (!Number.isFinite(lo)) { lo = 0; hi = 1; }
    if (hi - lo < 1e-12) { hi += 1; lo -= 1; }
    this.frame(w, h, lo, hi, 3, null);
    const X = (i) => pad.l + (i / Math.max(1, n - 1)) * (w - pad.l - pad.r);
    const Y = (v) => h - pad.b - ((v - lo) / (hi - lo)) * (h - pad.t - pad.b);
    if (this.highlight) {
      ctx.fillStyle = cssVar('--selection');
      const [s, len] = this.highlight;
      ctx.fillRect(X(s), pad.t, X(s + len - 1) - X(s), h - pad.t - pad.b);
    }
    for (const l of this.lines) {
      ctx.strokeStyle = cssVar(l.color || '--flow-view'); ctx.lineWidth = l.width || 1.5;
      ctx.beginPath();
      const step = Math.max(1, Math.floor(l.data.length / 1500));
      for (let i = 0; i < l.data.length; i += step) { if (i === 0) ctx.moveTo(X(i), Y(l.data[i])); else ctx.lineTo(X(i), Y(l.data[i])); }
      ctx.stroke();
    }
  }
}

/** Draw a CHW float image (values [0,1]) onto a canvas at its native size. */
export function drawCHW(canvas, data, c, h, w) {
  canvas.width = w; canvas.height = h;
  const ctx = canvas.getContext('2d');
  const im = ctx.createImageData(w, h);
  const hw = h * w;
  for (let i = 0; i < hw; i++) {
    const r = data[i], g = c === 3 ? data[hw + i] : r, b = c === 3 ? data[2 * hw + i] : r;
    im.data[4 * i] = Math.round(Math.min(1, Math.max(0, r)) * 255);
    im.data[4 * i + 1] = Math.round(Math.min(1, Math.max(0, g)) * 255);
    im.data[4 * i + 2] = Math.round(Math.min(1, Math.max(0, b)) * 255);
    im.data[4 * i + 3] = 255;
  }
  ctx.putImageData(im, 0, 0);
}
