// Guided flow: Data -> Train -> Inspect -> Evaluate -> Export, computed from real app state.
// A step is done when its outcome exists (data chosen, a run finished, embeddings inspected,
// evaluation run, something exported); the first step that is not done is the current one.
import { $, $$ } from './dom.js';

export const STEPS = [
  { key: 'data', tab: 'data', label: 'Data', next: 'Pick a dataset on the Data tab.' },
  { key: 'trained', tab: 'train', label: 'Train', next: 'Next: train a model (a Quick look run takes about a minute).' },
  { key: 'inspected', tab: 'inspect', label: 'Inspect', next: 'Next: inspect the embeddings for collapse.' },
  { key: 'evaluated', tab: 'evaluate', label: 'Evaluate', next: 'Next: score the encoder with a few labels.' },
  { key: 'exported', tab: 'export', label: 'Export', next: 'Next: export embeddings, weights or a bundle.' },
];

export function initStepper(app) {
  app.progress = { data: !!(app.data && app.data.n), trained: false, inspected: false, evaluated: false, exported: false };
  const render = () => {
    const firstOpen = STEPS.findIndex((s) => !app.progress[s.key]);
    STEPS.forEach((s, i) => {
      const li = $(`#stepper li[data-step="${s.key}"]`);
      const state = app.progress[s.key] ? 'done' : i === firstOpen ? 'current' : 'upcoming';
      li.dataset.state = state;
      const btn = li.querySelector('button');
      if (state === 'current') btn.setAttribute('aria-current', 'step'); else btn.removeAttribute('aria-current');
      li.querySelector('.state').textContent = state === 'done' ? ', done' : state === 'current' ? ', next step' : '';
    });
    $('#stepper-hint').textContent = firstOpen === -1 ? 'All steps done. Change a setting and train again to compare.' : STEPS[firstOpen].next;
  };
  const set = (patch) => { Object.assign(app.progress, patch); render(); };
  app.on('data', () => set({ data: true }));
  app.on('run-start', () => set({ trained: false, inspected: false, evaluated: false, exported: false }));
  app.on('run-end', (ev) => { if (!ev || ev.status !== 'error') set({ trained: true }); });
  app.on('weights-loaded', () => set({ trained: true }));
  app.on('inspected', () => set({ inspected: true }));
  app.on('evaluation', () => set({ evaluated: true }));
  app.on('exported', () => set({ exported: true }));
  $$('#stepper button').forEach((b) => b.addEventListener('click', () => app.router.select(b.dataset.goto)));
  render();
}
