// Small 3D scatter on a 2D canvas: orthographic projection with yaw/pitch rotation.
// Drag (pointer) or arrow keys rotate; click picks the nearest point (reported via onPick).
// Also draws a flat 2D PCA plot when mode = '2d'.
// 10 classes are encoded with 5 hues x 2 marker styles (filled dot / ring) so identity never
// rests on hue alone; the legend isolates one class on click.
import { cssVar } from './charts.js';

export const CLASS_HUES = ['--cat-1', '--cat-2', '--cat-3', '--cat-4', '--cat-5'];
export function classStyle(label) {
  if (label === null || label === undefined || label < 0) return { color: '--ink-3', ring: false };
  return { color: CLASS_HUES[label % 5], ring: Math.floor(label / 5) % 2 === 1 };
}

export class Scatter3D {
  constructor(canvas, { onPick = () => {}, describe = (i) => `#${i}` } = {}) {
    this.canvas = canvas;
    this.ctx = canvas.getContext('2d');
    this.onPick = onPick;
    this.describe = describe;
    this.yaw = 0.6; this.pitch = 0.35;
    this.mode = '3d';
    this.pts = null; this.labels = null; this.n = 0;
    this.selected = -1; this.highlight = new Set(); this.focusClass = null;
    this.projected = [];
    let drag = null;
    canvas.addEventListener('pointerdown', (e) => { drag = { x: e.clientX, y: e.clientY, moved: false }; canvas.setPointerCapture(e.pointerId); });
    canvas.addEventListener('pointermove', (e) => {
      if (!drag) return;
      const dx = e.clientX - drag.x, dy = e.clientY - drag.y;
      if (Math.abs(dx) + Math.abs(dy) > 2) drag.moved = true;
      if (this.mode === '3d' && drag.moved) {
        this.yaw += dx * 0.01;
        this.pitch = Math.max(-1.5, Math.min(1.5, this.pitch + dy * 0.01));
        drag.x = e.clientX; drag.y = e.clientY;
        this.draw();
      }
    });
    canvas.addEventListener('pointerup', (e) => {
      if (drag && !drag.moved) this.pick(e);
      drag = null;
    });
    canvas.tabIndex = 0;
    canvas.addEventListener('keydown', (e) => {
      const k = e.key;
      if (this.mode !== '3d') return;
      if (k === 'ArrowLeft') this.yaw -= 0.1; else if (k === 'ArrowRight') this.yaw += 0.1;
      else if (k === 'ArrowUp') this.pitch = Math.max(-1.5, this.pitch - 0.1);
      else if (k === 'ArrowDown') this.pitch = Math.min(1.5, this.pitch + 0.1);
      else return;
      e.preventDefault();
      this.draw();
    });
    new ResizeObserver(() => this.draw()).observe(canvas);
  }

  set(pts, n, labels) {
    this.pts = pts; this.n = n; this.labels = labels;
    let s = 0;
    for (let i = 0; i < n * 3; i++) s = Math.max(s, Math.abs(pts[i]));
    this.scale = s || 1;
    this.draw();
  }

  setMode(m) { this.mode = m; this.draw(); }

  project(i, w, h) {
    const p = this.pts, s = this.scale;
    let x = p[i * 3] / s, y = p[i * 3 + 1] / s, z = p[i * 3 + 2] / s;
    if (this.mode === '3d') {
      const cy = Math.cos(this.yaw), sy = Math.sin(this.yaw), cp = Math.cos(this.pitch), sp = Math.sin(this.pitch);
      const x1 = cy * x + sy * z, z1 = -sy * x + cy * z;
      const y1 = cp * y - sp * z1, z2 = sp * y + cp * z1;
      x = x1; y = y1; z = z2;
    } else z = 0;
    const r = Math.min(w, h) * 0.44;
    return { x: w / 2 + x * r, y: h / 2 - y * r, z };
  }

  draw() {
    const dpr = window.devicePixelRatio || 1;
    const w = this.canvas.clientWidth, h = this.canvas.clientHeight;
    if (!w || !h) return;
    this.canvas.width = Math.round(w * dpr); this.canvas.height = Math.round(h * dpr);
    const ctx = this.ctx;
    ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
    ctx.clearRect(0, 0, w, h);
    this.drawAxes(ctx, w, h);
    if (!this.pts) {
      ctx.fillStyle = cssVar('--ink-3'); ctx.font = '13px system-ui, sans-serif'; ctx.textAlign = 'center';
      ctx.fillText('Train a model, then refresh to see its embeddings', w / 2, h / 2);
      return;
    }
    const order = [];
    this.projected = new Array(this.n);
    for (let i = 0; i < this.n; i++) { this.projected[i] = this.project(i, w, h); order.push(i); }
    order.sort((a, b) => this.projected[a].z - this.projected[b].z);
    const colors = CLASS_HUES.map((c) => cssVar(c));
    const surface = cssVar('--surface-2'), ink3 = cssVar('--ink-3');
    for (const i of order) {
      const p = this.projected[i];
      const lab = this.labels ? this.labels[i] : -1;
      const st = classStyle(lab);
      const dim = this.focusClass !== null && lab !== this.focusClass;
      const depth = this.mode === '3d' ? 0.55 + 0.45 * ((p.z + 1.2) / 2.4) : 1;
      ctx.globalAlpha = dim ? 0.08 : Math.max(0.35, Math.min(1, depth));
      const col = lab >= 0 ? colors[lab % 5] : ink3;
      const r = this.highlight.has(i) ? 5 : 3.2;
      ctx.beginPath(); ctx.arc(p.x, p.y, r + 1, 0, 2 * Math.PI); ctx.fillStyle = surface; ctx.fill();
      ctx.beginPath(); ctx.arc(p.x, p.y, r, 0, 2 * Math.PI);
      if (st.ring) { ctx.strokeStyle = col; ctx.lineWidth = 1.6; ctx.stroke(); } else { ctx.fillStyle = col; ctx.fill(); }
    }
    ctx.globalAlpha = 1;
    if (this.selected >= 0 && this.selected < this.n) {
      const p = this.projected[this.selected];
      ctx.strokeStyle = cssVar('--ink'); ctx.lineWidth = 2;
      ctx.beginPath(); ctx.arc(p.x, p.y, 8, 0, 2 * Math.PI); ctx.stroke();
      for (const j of this.highlight) {
        const q = this.projected[j];
        if (!q) continue;
        ctx.strokeStyle = cssVar('--line-strong'); ctx.lineWidth = 1;
        ctx.beginPath(); ctx.moveTo(p.x, p.y); ctx.lineTo(q.x, q.y); ctx.stroke();
      }
    }
  }

  drawAxes(ctx, w, h) {
    ctx.strokeStyle = cssVar('--line'); ctx.lineWidth = 1;
    ctx.fillStyle = cssVar('--ink-3'); ctx.font = '11px system-ui, sans-serif'; ctx.textAlign = 'left';
    const axes = [[1, 0, 0, 'PC1'], [0, 1, 0, 'PC2'], [0, 0, 1, 'PC3']];
    const saved = this.pts;
    for (const [x, y, z, name] of axes) {
      if (this.mode === '2d' && z) continue;
      this.pts = new Float32Array([x, y, z]);
      const s = this.scale; this.scale = 1;
      const p = this.project(0, w, h);
      this.scale = s;
      ctx.beginPath(); ctx.moveTo(w / 2, h / 2); ctx.lineTo(p.x, p.y); ctx.stroke();
      ctx.fillText(name, p.x + 4, p.y - 4);
    }
    this.pts = saved;
  }

  pick(e) {
    if (!this.pts) return;
    const r = this.canvas.getBoundingClientRect();
    const x = e.clientX - r.left, y = e.clientY - r.top;
    let best = -1, bd = 14 * 14;
    for (let i = 0; i < this.n; i++) {
      const p = this.projected[i];
      if (!p) continue;
      if (this.focusClass !== null && this.labels && this.labels[i] !== this.focusClass) continue;
      const d = (p.x - x) ** 2 + (p.y - y) ** 2;
      if (d < bd) { bd = d; best = i; }
    }
    if (best >= 0) this.onPick(best);
  }

  select(i, neighbors = []) { this.selected = i; this.highlight = new Set(neighbors); this.draw(); }
}
