// Compute the LeJEPA loss with the web engine for inputs written by tests/test_web_engine.py.
// Input JSON: {V, Vg, N, K, M, lam, t_max, knots, z: [V*N*K], dirs: [[K*M] per view]}
// Output JSON: {loss, pred, sigreg, grad_norm}
import { readFileSync } from 'node:fs';
import { lejepaLoss, epConstants } from '../../site/js/engine/lejepa.js';

const inp = JSON.parse(readFileSync(process.argv[2], 'utf8'));
const Z = Float32Array.from(inp.z);
const dirs = inp.dirs.map((d) => Float64Array.from(d));
const out = lejepaLoss(Z, inp.V, inp.Vg, inp.N, inp.K, dirs, inp.M, inp.lam, { ep: epConstants(inp.t_max, inp.knots) });
let g = 0;
for (const x of out.dZ) g += x * x;
process.stdout.write(JSON.stringify({ loss: out.loss, pred: out.pred, sigreg: out.sigreg, grad: Array.from(out.dZ) }));
