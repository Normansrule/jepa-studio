// Evaluate tab: linear probe + k-NN for this run vs random-init and supervised baselines.
import { $, el, clear, gate } from '../ui/dom.js';
import { GroupedBarChart, fmt } from '../ui/charts.js';
import { COLLAPSE_TEXT } from '../ui/dom.js';
import { field, valueOf, setValue, onValidity, invalidFor, fixMessage } from '../ui/fields.js';
import { notify } from '../ui/toast.js';
import { webDefaultConfig } from '../config.js';

let app, chart, busy = false;

/** Why "Run evaluation" cannot run right now ('' = it can). */
function blocker() {
  if (!app.run) return 'Needs a trained model: start a run on the Train tab.';
  if (app.data.kind === 'images' && !app.data.labels) return 'These images have no labels: choose a folder of class sub-folders to evaluate.';
  return fixMessage(invalidFor('eval'), 'evaluate', 'Evaluate');
}

function updateGate() { if (!busy) gate($('#eval-run'), blocker()); }

function render(ev) {
  const names = Object.keys(ev.methods);
  chart.set(names, names.map((n) => ({ probe: ev.methods[n].linear_probe, knn: ev.methods[n].knn })), ev.chance);
  $('#eval-sub').textContent = `${ev.n_labeled} labeled, ${ev.n_test} held out, chance ${(ev.chance * 100).toFixed(0)}%`;
  const tb = clear($('#eval-table tbody'));
  for (const n of names) {
    const m = ev.methods[n];
    tb.append(el('tr', {},
      el('td', { text: n }),
      el('td', { class: 'num', text: `${(m.linear_probe * 100).toFixed(1)}%` }),
      el('td', { class: 'num', text: `${(m.knn * 100).toFixed(1)}%` }),
      el('td', { class: 'num', text: fmt(m.effective_rank, 3) }),
      el('td', { class: 'num', text: fmt(m.sigreg, 3) }),
      el('td', { text: COLLAPSE_TEXT[m.collapse] || m.collapse })));
  }
}

async function run() {
  const btn = $('#eval-run');
  const st = $('#eval-status');
  const why = blocker();
  if (why) { st.textContent = why; updateGate(); return; }
  busy = true;
  btn.disabled = true;
  const frac = valueOf('eval-frac');
  st.textContent = 'Scoring…';
  const off = app.backend.onProgress((s) => { st.textContent = `Scoring: ${s}…`; });
  try {
    const t0 = performance.now();
    const ev = await app.backend.evaluate(frac);
    app.evaluation = { ...ev, at: new Date().toISOString(), step: app.run.events.filter((e) => e.event === 'step').at(-1)?.step ?? null };
    render(ev);
    const secs = ((performance.now() - t0) / 1000).toFixed(1);
    st.textContent = `Done in ${secs} s.`;
    const mine = ev.methods[Object.keys(ev.methods)[0]];
    notify('good', 'Evaluation done', mine ? `Linear probe ${(mine.linear_probe * 100).toFixed(1)}%, k-NN ${(mine.knn * 100).toFixed(1)}% (chance ${(ev.chance * 100).toFixed(0)}%).` : `Done in ${secs} s.`);
    app.emit('evaluation', app.evaluation);
  } catch (e) {
    st.textContent = `Evaluation failed: ${e.message}`;
    notify('bad', 'Evaluation failed', e.message);
  } finally {
    off();
    busy = false;
    updateGate();
  }
}

export function init(a) {
  app = a;
  const out = $('#eval-frac-out');
  const pct = () => { const v = valueOf('eval-frac'); out.textContent = v === null ? '–' : `${Math.round(v * 100)}%`; };
  field('eval-frac', { path: 'data.labeled_fraction', gate: 'eval', section: 'eval', def: () => webDefaultConfig().data.labeled_fraction,
    commit: () => { app.config.data.labeled_fraction = valueOf('eval-frac'); app.emit('config'); } });
  $('#eval-frac').addEventListener('input', pct);
  $('#eval-frac-num').addEventListener('input', pct);
  pct();
  onValidity(() => { pct(); updateGate(); });
  for (const e of ['run-start', 'weights-loaded', 'data']) app.on(e, updateGate);
  updateGate();
  $('#sw-probe').style.setProperty('--c', 'var(--flow-encoder)');
  $('#sw-knn').style.setProperty('--c', 'var(--flow-loss-pred)');
  chart = new GroupedBarChart($('#c-eval'), { series: [{ key: 'probe', label: 'linear probe', color: '--flow-encoder' }, { key: 'knn', label: 'k-NN', color: '--flow-loss-pred' }] });
  $('#eval-run').addEventListener('click', run);
  app.on('config-loaded', () => { setValue('eval-frac', app.config.data.labeled_fraction ?? 0.1); pct(); });
}
