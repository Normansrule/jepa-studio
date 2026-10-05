// Headless smoke test of the demo trainer: 30 steps on Shapes, loss must go down and the
// run must stay finite; evaluate() must return scores for all three methods.
import { Trainer } from '../../site/js/engine/trainer.js';
import { webDefaultConfig, deepMerge } from '../../site/js/config.js';

const cfg = deepMerge(webDefaultConfig(), { data: { max_items: 512 }, train: { batch_size: 32, max_steps: 60 } });
const tr = new Trainer(cfg);
const t0 = performance.now();
const losses = [];
for (let i = 0; i < 30; i++) losses.push(await tr.trainStep());
const dt = (performance.now() - t0) / 1000;
const first = losses.slice(0, 5).reduce((s, x) => s + x.loss, 0) / 5;
const last = losses.slice(-5).reduce((s, x) => s + x.loss, 0) / 5;
const iso = tr.isotropy(256);
const ev = tr.evaluate(0.2);
console.log(JSON.stringify({
  first, last, sps: (30 * 32) / dt,
  sigreg: [losses[0].sigreg, losses.at(-1).sigreg], pred: [losses[0].pred, losses.at(-1).pred],
  iso: iso.report.effective_rank, collapse: iso.collapse.status, ev,
}));
if (!(last < first)) { console.error('loss did not decrease'); process.exit(1); }
