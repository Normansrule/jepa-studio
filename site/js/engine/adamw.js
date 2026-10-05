// AdamW (decoupled weight decay, Loshchilov & Hutter) as torch.optim.AdamW with default betas,
// and the warmup + cosine schedule of jepa_studio.train.lr_at.

export function lrAt(step, total, base, warmupFrac, finalRatio) {
  const warm = Math.max(1, Math.floor(total * warmupFrac));
  if (step < warm) return base * (step + 1) / warm;
  const p = Math.min(1, (step - warm) / Math.max(1, total - warm));
  return base * (finalRatio + (1 - finalRatio) * 0.5 * (1 + Math.cos(Math.PI * p)));
}

export class AdamW {
  constructor(params, { lr = 5e-4, betas = [0.9, 0.999], eps = 1e-8, weightDecay = 0.05 } = {}) {
    this.params = params;
    this.lr = lr; this.b1 = betas[0]; this.b2 = betas[1]; this.eps = eps; this.wd = weightDecay;
    this.t = 0;
    this.m = params.map((p) => ({ W: new Float32Array(p.W.length), b: new Float32Array(p.b.length) }));
    this.v = params.map((p) => ({ W: new Float32Array(p.W.length), b: new Float32Array(p.b.length) }));
  }

  step(grads) {
    this.t += 1;
    const { b1, b2, eps, lr, wd } = this;
    const bc1 = 1 - b1 ** this.t, bc2 = 1 - b2 ** this.t;
    const stepSize = lr / bc1;
    const sq2 = Math.sqrt(bc2);
    for (let i = 0; i < this.params.length; i++) {
      for (const key of ['W', 'b']) {
        const p = this.params[i][key], g = grads[i][key], m = this.m[i][key], v = this.v[i][key];
        const decay = 1 - lr * wd;
        for (let j = 0; j < p.length; j++) {
          const gj = g[j];
          m[j] = b1 * m[j] + (1 - b1) * gj;
          v[j] = b2 * v[j] + (1 - b2) * gj * gj;
          p[j] = p[j] * decay - stepSize * m[j] / (Math.sqrt(v[j]) / sq2 + eps);
        }
      }
    }
  }
}
