// Seeded randomness shared by the web engine. mulberry32 is bit-identical to
// jepa_studio.data.synthetic.Mulberry32 (Python), so seeds mean the same thing on both tiers.

export function mulberry32(seed) {
  let a = seed >>> 0;
  return function next() {
    a = (a + 0x6d2b79f5) >>> 0;
    let t = a;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Class wrapper with the same method names as the Python Mulberry32. */
export class Mulberry32 {
  constructor(seed) { this.next = mulberry32(seed); }
  uniform(lo, hi) { return lo + (hi - lo) * this.next(); }
}

/** Standard normal sampler (Box-Muller) on top of a uniform generator. */
export function gaussian(next) {
  let spare = null;
  return function () {
    if (spare !== null) { const s = spare; spare = null; return s; }
    let u = 0;
    while (u <= 1e-12) u = next();
    const v = next();
    const r = Math.sqrt(-2 * Math.log(u));
    spare = r * Math.sin(2 * Math.PI * v);
    return r * Math.cos(2 * Math.PI * v);
  };
}

/** Mix two 32-bit integers into a new seed (for per-step / per-purpose streams). */
export function mixSeed(a, b) {
  let h = (Math.imul((a >>> 0) ^ 0x9e3779b9, 0x85ebca6b) ^ Math.imul((b >>> 0) + 0x632be5ab, 0xc2b2ae35)) >>> 0;
  h ^= h >>> 16; h = Math.imul(h, 0x7feb352d) >>> 0; h ^= h >>> 15;
  return h >>> 0;
}

/** Fisher-Yates permutation of 0..n-1. */
export function permutation(n, next) {
  const p = new Int32Array(n);
  for (let i = 0; i < n; i++) p[i] = i;
  for (let i = n - 1; i > 0; i--) {
    const j = Math.floor(next() * (i + 1));
    const t = p[i]; p[i] = p[j]; p[j] = t;
  }
  return p;
}
