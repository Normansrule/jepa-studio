// Parity harness: node tests/js/world_parity.mjs <cases.json> [<model.json>]
// cases.json: {env_cases: [{env, seed, actions:[[ax,ay],...]}], model_cases?: {images: [[...3072 floats]], z:[[...]], a:[[...]]}}
// Prints JSON: {env: [{states:[[...]], goal:[...], frames:[[...3072]]}], model?: {latents:[[...]], preds:[[...]],
//               nearest:[i...]}, timing_ms?: number}
import { readFileSync } from "node:fs";
import { makeEnv } from "../../site/js/world/env.js";
import { encode, mulberry32, nearestBank, planCEM, predict, prepareWorldModel } from "../../site/js/world/planner.js";

const cases = JSON.parse(readFileSync(process.argv[2], "utf8"));
const out = { env: [] };

for (const c of cases.env_cases) {
  const env = makeEnv(c.env);
  let s = env.reset(c.seed);
  const states = [Array.from(s)], frames = [Array.from(env.render(s))];
  for (const a of c.actions) {
    s = env.step(s, a);
    states.push(Array.from(s));
    frames.push(Array.from(env.render(s)));
  }
  const goal = env.goal(c.seed);
  out.env.push({ states, frames, goal: Array.from(goal), goal_success: env.success(goal, goal) });
}

if (process.argv[3]) {
  const model = prepareWorldModel(JSON.parse(readFileSync(process.argv[3], "utf8")));
  const mc = cases.model_cases;
  const latents = mc.images.map((im) => Array.from(encode(model, Float32Array.from(im))));
  const d = model.latent_dim, n = mc.z.length;
  const Z = Float32Array.from(mc.z.flat()), A = Float32Array.from(mc.a.flat());
  const P = predict(model, Z, A, n);
  const preds = [];
  for (let i = 0; i < n; i++) preds.push(Array.from(P.subarray(i * d, (i + 1) * d)));
  const nearest = latents.map((z) => nearestBank(model, z));
  // timing of one full CEM replanning step with the exported defaults
  const z0 = Float32Array.from(latents[0]), zg = Float32Array.from(latents[latents.length - 1]);
  planCEM(model, z0, zg, {}, mulberry32(1));   // warm-up (JIT)
  const t0 = performance.now();
  const reps = 5;
  let plan;
  for (let r = 0; r < reps; r++) plan = planCEM(model, z0, zg, {}, mulberry32(r));
  const timing = (performance.now() - t0) / reps;
  out.model = { latents, preds, nearest, timing_ms: timing,
                plan: { actions: Array.from(plan.actions), costHistory: plan.costHistory } };
}

process.stdout.write(JSON.stringify(out));
