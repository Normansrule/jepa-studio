// Augmentation "views" on Float32Array images in CHW layout with values in [0, 1],
// following jepa_studio/data/views.py: random resized crop (scale range, aspect 3/4..4/3),
// horizontal flip, colour jitter (p=0.8), grayscale, blur, and an optional block mask.
// Resampling is separable and antialiased (triangle filter widened when downscaling), which is
// what torch's interpolate(mode="bilinear", antialias=True) does.

/** One image object: {data: Float32Array(C*H*W), c, h, w}. */
export function makeImage(data, c, h, w) { return { data, c, h, w }; }

/** Port of views._crop_box (top, left, ch, cw); rng is a () => [0,1) function. */
export function cropBox(h, w, scale, rng, ratio = [3 / 4, 4 / 3]) {
  const area = h * w;
  const uni = (a, b) => a + (b - a) * rng();
  const randint = (a, b) => a + Math.floor(rng() * (b - a + 1));
  for (let i = 0; i < 10; i++) {
    const target = area * uni(scale[0], scale[1]);
    const ar = Math.exp(uni(Math.log(ratio[0]), Math.log(ratio[1])));
    const cw = Math.round(Math.sqrt(target * ar));
    const ch = Math.round(Math.sqrt(target / ar));
    if (cw > 0 && cw <= w && ch > 0 && ch <= h) return [randint(0, h - ch), randint(0, w - cw), ch, cw];
  }
  const s = Math.min(h, w);
  return [Math.floor((h - s) / 2), Math.floor((w - s) / 2), s, s];
}

const FILTER_CACHE = new Map();
function filterWeights(inSize, outSize, start, len) {
  const key = `${inSize},${outSize},${start},${len}`;
  let f = FILTER_CACHE.get(key);
  if (!f) {
    if (FILTER_CACHE.size > 4096) FILTER_CACHE.clear();
    f = computeFilter(inSize, outSize, start, len);
    FILTER_CACHE.set(key, f);
  }
  return f;
}

function computeFilter(inSize, outSize, start, len) {
  // Maps output index o to source coordinates inside [start, start+len).
  const scale = len / outSize;
  const support = Math.max(scale, 1);
  const rows = [];
  for (let o = 0; o < outSize; o++) {
    const center = start + (o + 0.5) * scale - 0.5;
    const lo = Math.max(0, Math.floor(center - support));
    const hi = Math.min(inSize - 1, Math.ceil(center + support));
    const idx = [], wts = [];
    let sum = 0;
    for (let i = lo; i <= hi; i++) {
      const wgt = Math.max(0, 1 - Math.abs((i - center) / support));
      if (wgt > 0) { idx.push(i); wts.push(wgt); sum += wgt; }
    }
    if (!idx.length) { idx.push(Math.min(inSize - 1, Math.max(0, Math.round(center)))); wts.push(1); sum = 1; }
    rows.push({ idx, wts: wts.map((x) => x / sum) });
  }
  return rows;
}

/** Crop box [top, left, ch, cw] of img resampled to (outH, outW). */
export function resizedCrop(img, box, outH, outW = outH) {
  const [top, left, ch, cw] = box;
  const { c, h, w, data } = img;
  const fy = filterWeights(h, outH, top, ch);
  const fx = filterWeights(w, outW, left, cw);
  const tmp = new Float32Array(c * h * outW);
  for (let k = 0; k < c; k++) for (let y = 0; y < h; y++) {
    const row = k * h * w + y * w;
    for (let o = 0; o < outW; o++) {
      const f = fx[o];
      let s = 0;
      for (let j = 0; j < f.idx.length; j++) s += data[row + f.idx[j]] * f.wts[j];
      tmp[k * h * outW + y * outW + o] = s;
    }
  }
  const out = new Float32Array(c * outH * outW);
  for (let k = 0; k < c; k++) for (let o = 0; o < outH; o++) {
    const f = fy[o];
    for (let x = 0; x < outW; x++) {
      let s = 0;
      for (let j = 0; j < f.idx.length; j++) s += tmp[k * h * outW + f.idx[j] * outW + x] * f.wts[j];
      out[k * outH * outW + o * outW + x] = s;
    }
  }
  return makeImage(out, c, outH, outW);
}

export function resize(img, outH, outW = outH) { return resizedCrop(img, [0, 0, img.h, img.w], outH, outW); }

export function flipH(img) {
  const { c, h, w, data } = img;
  const out = new Float32Array(data.length);
  for (let k = 0; k < c; k++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    out[k * h * w + y * w + x] = data[k * h * w + y * w + (w - 1 - x)];
  }
  return makeImage(out, c, h, w);
}

const clamp01 = (x) => (x < 0 ? 0 : x > 1 ? 1 : x);

export function colorJitter(img, strength, rng) {
  if (strength <= 0) return img;
  const uni = (a, b) => a + (b - a) * rng();
  const b = uni(1 - strength, 1 + strength);
  const cf = uni(1 - strength, 1 + strength);
  const s = uni(1 - strength / 2, 1 + strength / 2);
  const { c, h, w } = img;
  const out = Float32Array.from(img.data, (x) => x * b);
  let mean = 0;
  for (const x of out) mean += x;
  mean /= out.length;
  for (let i = 0; i < out.length; i++) out[i] = (out[i] - mean) * cf + mean;
  if (c === 3) {
    const hw = h * w;
    for (let i = 0; i < hw; i++) {
      const gray = 0.299 * out[i] + 0.587 * out[hw + i] + 0.114 * out[2 * hw + i];
      for (let k = 0; k < 3; k++) out[k * hw + i] = (out[k * hw + i] - gray) * s + gray;
    }
  }
  for (let i = 0; i < out.length; i++) out[i] = clamp01(out[i]);
  return makeImage(out, c, h, w);
}

export function toGray(img) {
  if (img.c !== 3) return img;
  const hw = img.h * img.w, d = img.data;
  const out = new Float32Array(d.length);
  for (let i = 0; i < hw; i++) {
    const g = 0.299 * d[i] + 0.587 * d[hw + i] + 0.114 * d[2 * hw + i];
    out[i] = g; out[hw + i] = g; out[2 * hw + i] = g;
  }
  return makeImage(out, 3, img.h, img.w);
}

export function blur(img, sigma) {
  const k = Math.max(3, 2 * Math.round(2 * sigma) + 1);
  const half = Math.floor(k / 2);
  const g = [];
  let sum = 0;
  for (let i = 0; i < k; i++) { const v = Math.exp(-0.5 * ((i - half) / sigma) ** 2); g.push(v); sum += v; }
  for (let i = 0; i < k; i++) g[i] /= sum;
  const { c, h, w, data } = img;
  const refl = (i, n) => { if (n === 1) return 0; while (i < 0 || i >= n) i = i < 0 ? -i : 2 * (n - 1) - i; return i; };
  const tmp = new Float32Array(data.length), out = new Float32Array(data.length);
  for (let ch = 0; ch < c; ch++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = 0; j < k; j++) s += data[ch * h * w + y * w + refl(x + j - half, w)] * g[j];
    tmp[ch * h * w + y * w + x] = s;
  }
  for (let ch = 0; ch < c; ch++) for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    let s = 0;
    for (let j = 0; j < k; j++) s += tmp[ch * h * w + refl(y + j - half, h) * w + x] * g[j];
    out[ch * h * w + y * w + x] = s;
  }
  return makeImage(out, c, h, w);
}

/** Set ~ratio of patch-aligned blocks to 0.5 (views.block_mask). Returns {img, blocks}. */
export function blockMask(img, ratio, rng, patch = 4) {
  if (ratio <= 0) return img;
  const { c, h, w } = img;
  const gh = Math.max(1, Math.floor(h / patch)), gw = Math.max(1, Math.floor(w / patch));
  const n = Math.round(ratio * gh * gw);
  const cells = [...Array(gh * gw).keys()];
  for (let i = cells.length - 1; i > 0; i--) { const j = Math.floor(rng() * (i + 1)); [cells[i], cells[j]] = [cells[j], cells[i]]; }
  const out = Float32Array.from(img.data);
  for (const cell of cells.slice(0, n)) {
    const r = Math.floor(cell / gw), cc = cell % gw;
    for (let k = 0; k < c; k++) for (let y = r * patch; y < Math.min(h, (r + 1) * patch); y++) {
      for (let x = cc * patch; x < Math.min(w, (cc + 1) * patch); x++) out[k * h * w + y * w + x] = 0.5;
    }
  }
  return makeImage(out, c, h, w);
}

/** One augmented view (views.ImageViews.one). */
export function oneView(img, size, scale, d, rng) {
  let v = resizedCrop(img, cropBox(img.h, img.w, scale, rng), size);
  if (rng() < (d.flip_p ?? 0.5)) v = flipH(v);
  if (rng() < 0.8) v = colorJitter(v, d.color_jitter ?? 0.4, rng);
  if (rng() < (d.grayscale_p ?? 0.2)) v = toGray(v);
  if (rng() < (d.blur_p ?? 0.0)) v = blur(v, 0.1 + rng() * (0.9 + size / 64));
  v = blockMask(v, d.mask_ratio ?? 0.0, rng);
  return v;
}

/** {globals: [img], locals: [img]} for one input image. */
export function makeViews(img, d, rng) {
  const globals = [], locals = [];
  for (let i = 0; i < d.n_global; i++) globals.push(oneView(img, d.image_size, d.global_scale || [0.3, 1.0], d, rng));
  for (let i = 0; i < d.n_local; i++) locals.push(oneView(img, d.local_size, d.local_scale || [0.05, 0.3], d, rng));
  return { globals, locals };
}

/** Bring a view to side x side for the MLP: area-average down, or pixel-replicate up
 *  (what F.adaptive_avg_pool2d does in jepa_studio.models.MLPTiny). */
export function toSide(img, side) {
  if (img.h === side && img.w === side) return img.data;
  const { c, h, w, data } = img;
  const out = new Float32Array(c * side * side);
  for (let k = 0; k < c; k++) for (let oy = 0; oy < side; oy++) {
    const y0 = Math.floor(oy * h / side), y1 = Math.max(y0 + 1, Math.ceil((oy + 1) * h / side));
    for (let ox = 0; ox < side; ox++) {
      const x0 = Math.floor(ox * w / side), x1 = Math.max(x0 + 1, Math.ceil((ox + 1) * w / side));
      let s = 0;
      for (let y = y0; y < y1; y++) for (let x = x0; x < x1; x++) s += data[k * h * w + y * w + x];
      out[k * side * side + oy * side + ox] = s / ((y1 - y0) * (x1 - x0));
    }
  }
  return out;
}

/** 1-D views for time series windows (views.SeriesViews): crop + linear resample + scale + noise. */
export function seriesView(x, outLen, frac, rng, maskRatio = 0) {
  const n = x.length;
  const ln = Math.max(4, Math.floor(n * (frac[0] + (frac[1] - frac[0]) * rng())));
  const s = Math.floor(rng() * (n - ln + 1));
  const out = new Float32Array(outLen);
  const amp = 0.8 + 0.4 * rng(), noise = 0.03 * rng();
  for (let i = 0; i < outLen; i++) {
    const src = s + ((i + 0.5) * ln / outLen - 0.5);
    const i0 = Math.max(s, Math.min(s + ln - 1, Math.floor(src)));
    const i1 = Math.min(s + ln - 1, i0 + 1);
    const t = Math.max(0, Math.min(1, src - i0));
    out[i] = (x[i0] * (1 - t) + x[i1] * t) * amp + noise * (rng() * 2 - 1) * 1.7;
  }
  if (maskRatio > 0) {
    const ml = Math.floor(outLen * maskRatio), ms = Math.floor(rng() * (outLen - ml + 1));
    out.fill(0, ms, ms + ml);
  }
  return { data: out, start: s, len: ln };
}
