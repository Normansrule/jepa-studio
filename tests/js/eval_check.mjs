// Run the web ports of knn_accuracy / isotropy_report / collapse_check on inputs written by
// tests/test_web_engine.py::test_eval_parity and print the results as JSON.
import { readFileSync } from 'node:fs';
import { knnAccuracy, isotropyReport, collapseCheck } from '../../site/js/engine/evaluate.js';

const inp = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const flat = (m) => Float64Array.from(m.flat());
const out = { knn: [], iso: {}, collapse: {} };
for (const c of inp.knn) {
  const d = c.xtr[0].length;
  out.knn.push(knnAccuracy(flat(c.xtr), c.ytr, c.xtr.length, flat(c.xte), c.yte, c.xte.length, d, c.k));
}
for (const [name, m] of Object.entries(inp.mats)) {
  const rep = isotropyReport(flat(m), m.length, m[0].length, 64, 0);
  out.iso[name] = rep;
  out.collapse[name] = collapseCheck(rep);
}
process.stdout.write(JSON.stringify(out));
