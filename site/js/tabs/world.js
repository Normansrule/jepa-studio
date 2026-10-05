// World model tab. Uses the world-model engine written separately:
//   site/js/world/env.js      makeEnv(name) -> {reset, goal, step, render, success, distance}
//   site/js/world/planner.js  loadWorldModel, encode, planCEM, shiftPlan, nearestBank, mulberry32
//   site/models/world-<env>.json  exported "jepa-studio-world/v1" models
// Modules are imported dynamically so the rest of the app still works when they are missing.
import { $, el, clear, gate } from '../ui/dom.js';
import { LineChart, drawCHW, fmt } from '../ui/charts.js';
import { field, onCommit, valueOf, setValue, syncFromRange, onValidity, invalidFor, fixMessage } from '../ui/fields.js';
import { notify } from '../ui/toast.js';
import { DEFAULT_CONFIG } from '../config.js';

let app, envMod = null, planMod = null, models = {}, cem, dist, result = null, randomResult = null, busy = false;
let modelReady = false, codeReady = false;

/** Plan & run needs the world-model files and valid settings; Random needs the env code. */
function updateGates() {
  if (busy) return;
  const fix = fixMessage(invalidFor('world'), 'run a policy', 'World model');
  gate($('#world-run'), fix);
  if (!fix) $('#world-run').disabled = !modelReady;   // the status line says why the model is missing
  $('#world-random').disabled = !!fix || !codeReady;
}

async function loadModules() {
  try {
    [envMod, planMod] = await Promise.all([import('../world/env.js'), import('../world/planner.js')]);
  } catch (e) {
    envMod = planMod = null;
    return `World-model code not found (site/js/world/*.js): ${e.message}`;
  }
  return null;
}

async function getModel(name) {
  if (models[name]) return models[name];
  const m = await planMod.loadWorldModel(new URL(`../../models/world-${name}.json`, import.meta.url).href);
  models[name] = m;
  return m;
}

function frame(canvas, img) { drawCHW(canvas, img, 3, 32, 32); }

function status(t) { $('#world-status').textContent = t; }

const sleep = (ms) => new Promise((r) => setTimeout(r, ms));

/** Receding-horizon CEM: replan every step, execute the first action. */
async function planAndRun() {
  if (busy) return;
  const name = $('#world-env').value;
  let model;
  try { model = await getModel(name); } catch (e) { status(`Model files not found for "${name}" (site/models/world-${name}.json). Train and export one on desktop: ${e.message}`); return; }
  busy = true;
  $('#world-run').disabled = true;
  const env = envMod.makeEnv(name);
  const startSeed = valueOf('world-start') >>> 0, goalSeed = valueOf('world-goal') >>> 0;
  let s = env.reset(startSeed);
  const g = env.goal(goalSeed);
  const zGoal = planMod.encode(model, env.render(g));
  const rng = planMod.mulberry32(startSeed * 7919 + goalSeed);
  const maxSteps = valueOf('world-steps');
  const samples = Number($('#world-samples').value);
  const pd = model.plan_defaults || {};
  const steps = [{ state: s, img: env.render(s), dist: env.distance(s, g) }];
  let warm = null;
  const t0 = performance.now();
  for (let t = 0; t < maxSteps; t++) {
    if (env.success(s, g)) break;
    const z0 = planMod.encode(model, env.render(s));
    const plan = planMod.planCEM(model, z0, zGoal, { samples, elites: Math.max(5, Math.round((samples * (pd.elites || 30)) / (pd.samples || 300))), initMean: warm || undefined }, rng);
    const H = plan.imaginedLatents.length / model.latent_dim;
    const imagined = [];
    for (let h = 0; h < H; h++) {
      const z = plan.imaginedLatents.subarray(h * model.latent_dim, (h + 1) * model.latent_dim);
      const bi = planMod.nearestBank(model, z);
      imagined.push(env.render(Float64Array.from(model.bank.states[bi])));
    }
    const a = [plan.actions[0], plan.actions[1]];
    s = env.step(s, a);
    warm = planMod.shiftPlan(plan.mean, model.action_dim || 2);
    steps.push({ state: s, img: env.render(s), dist: env.distance(s, g), costs: plan.costHistory, imagined, action: a });
    status(`Planning… step ${t + 1} / ${maxSteps}`);
    if (t % 2 === 0) await sleep(0);
  }
  const ok = env.success(s, g);
  result = { env: name, startSeed, goalSeed, steps, success: ok, goalImg: env.render(g), ms: performance.now() - t0 };
  app.worldResult = { env: name, startSeed, goalSeed, steps: steps.length - 1, success: ok, final_distance: steps.at(-1).dist };
  status(`${ok ? 'Reached the goal' : 'Did not reach the goal'} in ${steps.length - 1} steps (${(result.ms / 1000).toFixed(1)} s).`);
  notify(ok ? 'good' : 'info', ok ? 'Goal reached' : 'Goal not reached', `The planner took ${steps.length - 1} steps.`);
  renderResult();
  busy = false;
  updateGates();
}

function runRandom() {
  const name = $('#world-env').value;
  const env = envMod.makeEnv(name);
  const startSeed = valueOf('world-start') >>> 0, goalSeed = valueOf('world-goal') >>> 0;
  let s = env.reset(startSeed);
  const g = env.goal(goalSeed);
  const rng = planMod.mulberry32(startSeed * 31 + goalSeed + 1);
  const maxSteps = valueOf('world-steps');
  const d = [env.distance(s, g)];
  let t = 0;
  for (; t < maxSteps && !env.success(s, g); t++) {
    s = env.step(s, [rng() * 2 - 1, rng() * 2 - 1]);
    d.push(env.distance(s, g));
  }
  randomResult = { dists: d, success: env.success(s, g), steps: t };
  renderCompare();
  renderDist();
}

function renderDist() {
  dist.reset();
  const a = result ? result.steps.map((x) => x.dist) : [];
  const b = randomResult ? randomResult.dists : [];
  const n = Math.max(a.length, b.length);
  for (let i = 0; i < n; i++) dist.push(i, { plan: a[i] ?? null, rand: b[i] ?? null });
}

function renderCompare() {
  const tb = clear($('#world-compare tbody'));
  const row = (name, steps, d, ok) => tb.append(el('tr', {}, el('td', { text: name }), el('td', { class: 'num', text: steps }), el('td', { class: 'num', text: fmt(d, 3) }), el('td', { text: ok ? 'yes' : 'no' })));
  if (result) row('CEM planner (latent space)', result.steps.length - 1, result.steps.at(-1).dist, result.success);
  if (randomResult) row('Random actions', randomResult.steps, randomResult.dists.at(-1), randomResult.success);
}

function showStep(k) {
  if (!result) return;
  const st = result.steps[k];
  frame($('#world-real'), st.img);
  $('#world-real-step').textContent = String(k);
  const nxt = result.steps[Math.min(k + 1, result.steps.length - 1)];
  const im = clear($('#world-film-imag'));
  cem.reset();
  if (nxt.imagined) {
    frame($('#world-imag'), nxt.imagined.at(-1));
    nxt.imagined.forEach((img, h) => { const c = el('canvas', { attrs: { 'aria-label': `Retrieved frame for imagined step ${h + 1}` } }); frame(c, img); im.append(c); });
    nxt.costs.forEach((c, i) => cem.push(i + 1, { cost: c }));
  } else frame($('#world-imag'), result.goalImg);
  [...$('#world-film-real').children].forEach((c, i) => c.setAttribute('aria-current', i === k ? 'true' : 'false'));
}

function renderResult() {
  const b = $('#world-badge');
  b.dataset.status = result.success ? 'good' : 'bad';
  b.textContent = result.success ? 'Goal reached' : 'Goal not reached';
  const film = clear($('#world-film-real'));
  result.steps.forEach((st, i) => { const c = el('canvas', { attrs: { 'aria-label': `Real frame ${i}` } }); frame(c, st.img); c.addEventListener('click', () => { $('#world-scrub').value = i; syncFromRange('world-scrub'); showStep(i); }); film.append(c); });
  frame($('#world-goal-frame'), result.goalImg);
  const scrub = $('#world-scrub');
  scrub.disabled = false;
  scrub.max = String(result.steps.length - 1);
  scrub.value = '0';
  syncFromRange('world-scrub');
  showStep(0);
  renderDist();
  renderCompare();
}

function renderInfo(name) {
  const kv = clear($('#world-info'));
  const m = models[name];
  if (!m) return;
  const row = (k, v) => kv.append(el('dt', { text: k }), el('dd', { text: v }));
  row('Latent size', String(m.latent_dim));
  row('Memory bank', `${m.bank.size} real frames`);
  if (m.plan_defaults) row('Horizon', `${m.plan_defaults.horizon} steps`);
  if (m.metrics) for (const [k, v] of Object.entries(m.metrics).slice(0, 4)) if (typeof v !== 'object') row(k.replace(/_/g, ' '), typeof v === 'number' ? fmt(v, 3) : String(v));
}

async function preload() {
  const err = await loadModules();
  if (err) { status(err); modelReady = codeReady = false; updateGates(); return; }
  codeReady = true;
  const name = $('#world-env').value;
  try {
    await getModel(name);
    status('Model loaded. Choose seeds and press Plan & run.');
    renderInfo(name);
    modelReady = true;
    const env = envMod.makeEnv(name);
    const s0 = valueOf('world-start'), g0 = valueOf('world-goal');
    if (s0 !== null) frame($('#world-real'), env.render(env.reset(s0 >>> 0)));
    if (g0 !== null) frame($('#world-goal-frame'), env.render(env.goal(g0 >>> 0)));
    const ic = $('#world-imag');
    ic.getContext('2d').clearRect(0, 0, ic.width, ic.height);
  } catch (e) {
    status(`Model files not found: site/models/world-${name}.json is missing or unreadable. Rebuild it with \`jepa-studio world --env ${name} --site\` from the repo root.`);
    modelReady = false;
  }
  updateGates();
}

export function init(a) {
  app = a;
  const htmlDefault = (id) => () => Number($(`#${id}`).defaultValue);
  field('world-env', { section: 'world', def: () => DEFAULT_CONFIG.world.env, commit: () => { result = null; randomResult = null; renderCompare(); preload(); } });
  field('world-start', { integer: true, gate: 'world', section: 'world', def: htmlDefault('world-start') });
  field('world-goal', { integer: true, gate: 'world', section: 'world', def: htmlDefault('world-goal') });
  field('world-steps', { integer: true, gate: 'world', section: 'world', def: htmlDefault('world-steps'),
    rangeMsg: (lo, hi) => `The planner runs ${lo} to ${hi} steps here.` });
  field('world-samples', { section: 'world', def: () => String(DEFAULT_CONFIG.world.cem_samples) });
  field('world-scrub', { integer: true, label: 'Step', commit: () => showStep(Number($('#world-scrub').value)) });
  onValidity(updateGates);
  cem = new LineChart($('#c-cem'), { xLabel: 'iteration', series: [{ key: 'cost', label: 'elite cost', color: '--flow-predictor' }], empty: 'Run the planner' });
  dist = new LineChart($('#c-dist'), { zero: true, series: [{ key: 'plan', label: 'planner', color: '--flow-predictor' }, { key: 'rand', label: 'random', color: '--ink-3' }], empty: 'Run a policy' });
  const lg = $('#dist-legend');
  for (const s of dist.series) {
    const sw = el('span', { class: 'swatch' });
    sw.style.setProperty('--c', `var(${s.color})`);
    lg.append(el('button', { attrs: { type: 'button', 'aria-pressed': 'true' }, on: { click: (e) => { dist.toggle(s.key); e.currentTarget.setAttribute('aria-pressed', String(!dist.hidden.has(s.key))); } } }, sw, s.label));
  }
  $('#world-run').addEventListener('click', planAndRun);
  $('#world-random').addEventListener('click', () => { if (envMod) runRandom(); });
  $('#world-scrub').addEventListener('input', () => showStep(Number($('#world-scrub').value)));
  for (const id of ['world-start', 'world-goal']) onCommit(id, () => { if (envMod) preload(); });
  app.on('config-loaded', () => { if (app.config.world && app.config.world.env) { setValue('world-env', app.config.world.env); preload(); } });
  updateGates();
  preload();
}
