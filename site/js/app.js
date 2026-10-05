// App shell: picks the backend (web demo or desktop), owns shared state, routes tabs.
// Tabs live in site/js/tabs/*.js; each exports init(app) and optionally show(app).
// Shared UI: ui/fields.js (validated inputs), ui/help.js (info disclosures), ui/stepper.js
// (progress), ui/toast.js (notices), ui/prefs.js (per-viewer preferences, storage-safe).
import { makeBackend, detectTier } from './backend.js';
import { setupTabs } from './ui/tabs.js';
import { redrawAll } from './ui/charts.js';
import { webDefaultConfig, loadSchema } from './config.js';
import { $$ } from './ui/dom.js';
import { initFields, resetSection } from './ui/fields.js';
import { initHelp } from './ui/help.js';
import { initStepper } from './ui/stepper.js';
import { initToasts, notify } from './ui/toast.js';
import { getPref, setPref, getRaw, setRaw } from './ui/prefs.js';
import * as dataTab from './tabs/data.js';
import * as trainTab from './tabs/train.js';
import * as inspectTab from './tabs/inspect.js';
import * as evalTab from './tabs/evaluate.js';
import * as worldTab from './tabs/world.js';
import * as explainTab from './tabs/explain.js';
import * as exportTab from './tabs/export.js';

export const APP_VERSION = '0.3.0';

const tabs = { data: dataTab, train: trainTab, inspect: inspectTab, evaluate: evalTab, world: worldTab, explain: explainTab, export: exportTab };

function setupTheme() {
  const btn = document.getElementById('theme-toggle');
  const root = document.documentElement;
  const saved = getRaw('jepa-theme');
  if (saved === 'light' || saved === 'dark') root.dataset.theme = saved;
  const isDark = () => (root.dataset.theme ? root.dataset.theme === 'dark' : matchMedia('(prefers-color-scheme: dark)').matches);
  const label = () => btn.setAttribute('aria-label', isDark() ? 'Switch to light theme' : 'Switch to dark theme');
  label();
  btn.addEventListener('click', () => {
    root.dataset.theme = isDark() ? 'light' : 'dark';
    setRaw('jepa-theme', root.dataset.theme);
    label();
    redrawAll();
    app.emit('theme');
  });
  matchMedia('(prefers-color-scheme: dark)').addEventListener('change', () => { label(); redrawAll(); app.emit('theme'); });
}

const app = {
  version: APP_VERSION,
  tier: detectTier(),
  backend: null,
  config: webDefaultConfig(),
  schema: null,
  run: null,          // {started, events: [], summary, backend, hardware}
  hardware: null,
  evaluation: null,
  inspect: null,
  listeners: {},
  on(name, f) { (this.listeners[name] ||= []).push(f); },
  emit(name, payload) { for (const f of this.listeners[name] || []) { try { f(payload); } catch (e) { console.error(e); } } },
  setBackendLabel(text) { document.getElementById('backend-label').textContent = text; },
  notify,
  /** Switch tab from a "Go to…" button: select it, move focus to its panel, scroll to the top. */
  goto(tab) {
    if (!this.router) return;
    this.router.select(tab);
    const panel = document.getElementById(`panel-${tab}`);
    if (panel) panel.focus({ preventScroll: true });
    window.scrollTo({ top: 0 });
  },
};
globalThis.jepaApp = app; // for debugging and the Playwright test

/** Empty states ("No model yet…") show until a run exists or weights are opened. */
function setupEmptyStates() {
  const update = () => { for (const n of $$('[data-empty-until="run"]')) n.hidden = !!app.run; };
  app.on('run-start', update);
  app.on('weights-loaded', update);
  update();
}

async function main() {
  initToasts();
  setupTheme();
  document.getElementById('app-version').textContent = `v${APP_VERSION}`;
  const badge = document.getElementById('tier-badge');
  badge.dataset.tier = app.tier;
  document.getElementById('tier-label').textContent = app.tier === 'desktop' ? 'Desktop' : 'Web demo';
  badge.title = app.tier === 'desktop' ? 'Training runs in the local Python backend (PyTorch).' : 'Training runs in this browser tab (demo tier).';
  app.backend = makeBackend();
  try { app.schema = await loadSchema(); } catch (e) { console.warn('schema not loaded', e); }
  initFields(app);
  for (const t of Object.values(tabs)) t.init(app);
  initHelp(document, app.tier);
  for (const b of $$('[data-reset]')) {
    b.addEventListener('click', () => {
      const n = resetSection(b.dataset.reset);
      const what = b.closest('.section-head')?.querySelector('h4')?.textContent || 'Settings';
      if (n) notify('info', `${what} reset to defaults`);
    });
  }
  // Last tab: the URL hash wins; otherwise reopen the tab this viewer used last.
  const lastTab = getPref('tab');
  if (!location.hash && typeof lastTab === 'string' && tabs[lastTab]) history.replaceState(null, '', `#${lastTab}`);
  const router = setupTabs(document.querySelector('[role="tablist"]'), {
    onChange: (name) => {
      const t = tabs[name];
      if (t && t.show) t.show(app);
      setPref('tab', name);
      requestAnimationFrame(redrawAll);
    },
  });
  app.router = router;
  for (const b of $$('[data-goto]')) if (!b.closest('#stepper')) b.addEventListener('click', () => app.goto(b.dataset.goto));
  initStepper(app);
  setupEmptyStates();
  app.emit('ready');
  document.documentElement.dataset.ready = 'true';
}

main().catch((e) => {
  console.error(e);
  const m = document.getElementById('main');
  const n = document.createElement('div');
  n.className = 'notice bad';
  n.textContent = `The app failed to start: ${e.message}`;
  m.prepend(n);
});
