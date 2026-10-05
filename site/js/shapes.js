// Built-in "Shapes" dataset, an exact port of jepa_studio/data/synthetic.py.
// Image i with seed s matches Python within float rounding (checked by tests/js/shapes_dump.mjs).
// Layout of render(): Float32Array (size*size*3), row-major HWC, values in [0, 1].
import { Mulberry32 } from './engine/rng.js';

export const CLASSES = ['circle', 'square', 'triangle', 'cross', 'ring', 'diamond', 'flower',
  'stripes', 'checker', 'crescent'];

/** (seed * 2654435761 + index * 40503 + 12345) mod 2^32, exact for any 32-bit seed. */
export function itemSeed(seed, index) {
  const v = (BigInt(seed) * 2654435761n + BigInt(index) * 40503n + 12345n) & 0xffffffffn;
  return Number(v);
}

function inside(cls, u, v, r, phase) {
  const ax = Math.abs(u), ay = Math.abs(v);
  const rad = Math.sqrt(u * u + v * v);
  switch (cls) {
    case 0: return rad < r;
    case 1: return Math.max(ax, ay) < r * 0.85;
    case 2: return v < r * 0.6 && v > -r * 0.9 + 1.9 * ax;
    case 3: { const w = r * 0.32; return (ax < w && ay < r) || (ay < w && ax < r); }
    case 4: return rad < r && rad > r * 0.55;
    case 5: return ax + ay < r;
    case 6: { const th = Math.atan2(v, u); return rad < r * (0.6 + 0.4 * Math.cos(5 * th)); }
    case 7: return Math.max(ax, ay) < r && Math.sin(u * 40.0 + phase) > 0;
    case 8: return Math.max(ax, ay) < r && Math.sin(u * 30.0 + phase) * Math.sin(v * 30.0 + phase) > 0;
    case 9: { const du = u - r * 0.45; return rad < r && Math.sqrt(du * du + v * v) > r * 0.8; }
    default: throw new Error(`unknown class ${cls}`);
  }
}

/** One image and its label: {img: Float32Array(size*size*3) HWC, label}. */
export function render(seed, index, size = 32) {
  const rng = new Mulberry32(itemSeed(seed, index));
  const cls = Math.floor(rng.next() * CLASSES.length) % CLASSES.length;
  let bg = [rng.uniform(0.0, 0.45), rng.uniform(0.0, 0.45), rng.uniform(0.0, 0.45)];
  let fg = [rng.uniform(0.55, 1.0), rng.uniform(0.55, 1.0), rng.uniform(0.55, 1.0)];
  if (rng.next() < 0.5) { bg = bg.map((x) => 1.0 - x); fg = fg.map((x) => 1.0 - x); }
  const cx = rng.uniform(-0.18, 0.18), cy = rng.uniform(-0.18, 0.18);
  const r = rng.uniform(0.2, 0.34);
  const ang = rng.uniform(0.0, 2 * Math.PI);
  const phase = rng.uniform(0.0, 2 * Math.PI);
  const noiseAmp = rng.uniform(0.0, 0.08);
  const noiseSeed = Math.floor(rng.next() * 4294967295);

  const ca = Math.cos(ang), sa = Math.sin(ang);
  const nrng = new Mulberry32(noiseSeed);
  const img = new Float32Array(size * size * 3);
  for (let row = 0; row < size; row++) {
    const y = (row + 0.5) / size - 0.5;
    for (let col = 0; col < size; col++) {
      const x = (col + 0.5) / size - 0.5;
      const px = x - cx, py = -(y - cy);
      const u = ca * px + sa * py;
      const v = -sa * px + ca * py;
      const m = inside(cls, u, v, r, phase) ? 1.0 : 0.0;
      const n = nrng.next();
      const o = (row * size + col) * 3;
      for (let c = 0; c < 3; c++) {
        let val = bg[c] * (1 - m) + fg[c] * m;
        val = val + (n - 0.5) * 2 * noiseAmp;
        img[o + c] = val < 0 ? 0 : val > 1 ? 1 : val;
      }
    }
  }
  return { img, label: cls };
}

/** HWC -> CHW (the layout the encoder flattens, matching PyTorch's (C, H, W)). */
export function hwcToChw(hwc, h, w, c = 3) {
  const out = new Float32Array(h * w * c);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) for (let k = 0; k < c; k++) {
    out[k * h * w + y * w + x] = hwc[(y * w + x) * c + k];
  }
  return out;
}

/** n images as one CHW block (n*3*size*size) plus labels. */
export function makeDataset(n, seed = 0, size = 32) {
  const per = 3 * size * size;
  const images = new Float32Array(n * per);
  const labels = new Int32Array(n);
  for (let i = 0; i < n; i++) {
    const { img, label } = render(seed, i, size);
    images.set(hwcToChw(img, size, size), i * per);
    labels[i] = label;
  }
  return { images, labels, n, size, channels: 3 };
}
