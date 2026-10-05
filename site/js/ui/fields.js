// Precise, validated inputs.
//
// * pairRange(): every range slider gets a number input with the same min/max/step, synced both
//   ways. The number is the precise value (typing 35 into a slider with step 10 means 35); the
//   slider shows the nearest position.
// * field(): registers a control with its config path (validated with config.js validate()
//   against the shared schema), the slider's own range for this tier, a section for "reset to
//   defaults" and a gate (the button it blocks while invalid). Invalid values get an inline
//   message (aria-invalid + aria-describedby); nothing is clamped or applied silently.
// * onCommit(): tab code runs only for committed, valid values.
import { el, clear } from './dom.js';
import { validate } from '../config.js';

const fields = new Map();
const validityListeners = new Set();
let schema = null, tier = 'web';
let uid = 0;

export function initFields(app) {
  schema = app.schema;
  tier = app.tier;
}

function subschema(path) {
  if (!schema || !path) return null;
  let s = schema;
  for (const k of path.split('.')) {
    if (!s || !s.properties || !(k in s.properties)) return null;
    s = s.properties[k];
  }
  return s;
}

function fmtNum(v) { return Number.isInteger(v) ? String(v) : String(Number(v.toPrecision(6))); }

function describe(sub) {
  const types = [sub.type].flat().filter((t) => t && t !== 'null');
  let s = types.includes('integer') ? 'a whole number' : 'a number';
  const lo = sub.minimum, hi = sub.maximum;
  if (lo !== undefined && hi !== undefined) s += ` from ${fmtNum(lo)} to ${fmtNum(hi)}`;
  else if (lo !== undefined) s += ` of at least ${fmtNum(lo)}`;
  else if (hi !== undefined) s += ` of at most ${fmtNum(hi)}`;
  return s;
}

function labelText(id) {
  const l = document.querySelector(`label[for="${id}"]`);
  return l ? l.textContent.trim() : id;
}

function tabName(node) {
  const p = node.closest('[role="tabpanel"]');
  const t = p && document.getElementById(p.getAttribute('aria-labelledby'));
  return t ? t.firstChild.textContent.trim() : '';
}

/** Create the number input that mirrors a range slider. Returns it. */
export function pairRange(range) {
  const existing = document.getElementById(`${range.id}-num`);
  if (existing) return existing;
  const num = el('input', { class: 'num', attrs: { type: 'number', id: `${range.id}-num`, inputmode: 'decimal' } });
  const copy = () => {
    for (const a of ['min', 'max', 'step']) { if (range.hasAttribute(a)) num.setAttribute(a, range.getAttribute(a)); else num.removeAttribute(a); }
    num.disabled = range.disabled;
  };
  copy();
  num.value = range.value;
  const label = document.querySelector(`label[for="${range.id}"]`);
  if (label) {
    if (!label.id) label.id = `${range.id}-label`;
    num.setAttribute('aria-labelledby', label.id);
  }
  const out = range.parentElement.querySelector('output');
  if (out && !out.dataset.keep) out.remove();
  range.after(num);
  new MutationObserver(() => { copy(); const f = fields.get(range.id); if (f) check(f); }).observe(range, { attributes: true, attributeFilter: ['min', 'max', 'step', 'disabled'] });
  return num;
}

/**
 * Register a control. opts: {path, gate, section, def: () => value, integer, rangeMsg(min, max),
 * label, commit: fn}. Range sliders are paired with a number input automatically.
 */
export function field(id, opts = {}) {
  const node = document.getElementById(id);
  if (!node) throw new Error(`no control #${id}`);
  const kind = node.type === 'range' ? 'range' : node.type === 'number' ? 'number' : node.type === 'checkbox' ? 'checkbox' : 'select';
  const f = { id, el: node, kind, num: null, shown: false, commits: new Set(), label: opts.label || labelText(id), ...opts };
  if (kind === 'range') f.num = pairRange(node);
  fields.set(id, f);
  if (kind === 'range' || kind === 'number') setupNumeric(f);
  else node.addEventListener('change', () => runCommits(f));
  if (opts.commit) f.commits.add(opts.commit);
  return f;
}

function setupNumeric(f) {
  const input = f.num || f.el;
  const sub = subschema(f.path);
  const integer = f.integer ?? (sub ? [sub.type].flat().includes('integer') : false);
  f.integer = integer;
  input.setAttribute('inputmode', integer ? 'numeric' : 'decimal');
  const container = f.el.closest('.field') || f.el.parentElement;
  f.err = el('p', { class: 'field-error', attrs: { id: `${f.id}-err-${++uid}`, hidden: true } });
  container.append(f.err);
  for (const n of [f.el, f.num].filter(Boolean)) {
    const ids = new Set((n.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean));
    ids.add(f.err.id);
    n.setAttribute('aria-describedby', [...ids].join(' '));
  }
  let internal = false;
  if (f.num) {
    // slider moved by the user (or set by code that dispatches events): copy to the number
    f.el.addEventListener('input', () => { f.num.value = f.el.value; check(f, { show: f.shown }); });
    f.el.addEventListener('change', () => {
      if (!internal) { f.num.value = f.el.value; }
      if (check(f, { show: true }) === null) runCommits(f);
    });
    f.num.addEventListener('input', () => {
      const ok = check(f, { show: f.shown }) === null;
      if (ok) { internal = true; f.el.value = f.num.value; internal = false; }
    });
    const commit = () => {
      if (check(f, { show: true }) !== null) return;
      internal = true;
      f.el.value = f.num.value;
      f.el.dispatchEvent(new Event('change'));
      internal = false;
    };
    f.num.addEventListener('change', commit);
    f.num.addEventListener('keydown', (e) => { if (e.key === 'Enter') commit(); });
  } else {
    f.el.addEventListener('input', () => check(f, { show: f.shown }));
    f.el.addEventListener('change', () => { if (check(f, { show: true }) === null) runCommits(f); });
    f.el.addEventListener('keydown', (e) => { if (e.key === 'Enter') f.el.dispatchEvent(new Event('change')); });
  }
}

function runCommits(f) { for (const fn of f.commits) fn(f); }

/** Validate one numeric field. Returns null or {text, path}. `show` renders the message. */
function check(f, { show = false } = {}) {
  if (f.kind !== 'range' && f.kind !== 'number') return null;
  const input = f.num || f.el;
  let problem = null;
  const raw = String(input.value).trim();
  if (input.disabled && f.kind === 'range') problem = null;
  else if (raw === '') problem = { text: input.validity && input.validity.badInput ? 'Enter a number.' : 'Enter a value.' };
  else {
    const v = Number(raw);
    const sub = subschema(f.path);
    const min = f.el.hasAttribute('min') ? Number(f.el.getAttribute('min')) : -Infinity;
    const max = f.el.hasAttribute('max') ? Number(f.el.getAttribute('max')) : Infinity;
    if (!Number.isFinite(v)) problem = { text: 'Enter a number.' };
    else if (sub && validate(v, sub, `$.${f.path}`).length) problem = { text: `Use ${describe(sub)}.`, path: f.path };
    else if (f.integer && !Number.isInteger(v)) problem = { text: 'Use a whole number.' };
    else if (v < min || v > max) {
      problem = { text: f.rangeMsg ? f.rangeMsg(fmtNum(min), fmtNum(max), tier) : `The ${tier === 'desktop' ? 'desktop app' : 'web demo'} accepts ${fmtNum(min)} to ${fmtNum(max)} here.` };
    }
  }
  const wasValid = f.valid !== false;
  f.valid = !problem;
  f.problem = problem;
  if (problem && show) {
    f.shown = true;
    clear(f.err).append(errIcon(), el('span', { text: problem.text }));
    if (problem.path) f.err.append(' ', el('span', { class: 'field-path' }, 'Schema key ', el('code', { text: problem.path })));
    f.err.hidden = false;
    for (const n of [f.el, f.num].filter(Boolean)) n.setAttribute('aria-invalid', 'true');
  } else if (!problem) {
    f.shown = false;
    f.err.hidden = true;
    clear(f.err);
    for (const n of [f.el, f.num].filter(Boolean)) n.removeAttribute('aria-invalid');
  }
  if (wasValid !== f.valid) emitValidity();
  return problem;
}

function errIcon() {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', 'true');
  const c = document.createElementNS(ns, 'circle');
  c.setAttribute('cx', '12'); c.setAttribute('cy', '12'); c.setAttribute('r', '9');
  const p = document.createElementNS(ns, 'path');
  p.setAttribute('d', 'M12 7v6M12 16.5v.01');
  for (const n of [c, p]) { n.setAttribute('fill', 'none'); n.setAttribute('stroke', 'currentColor'); n.setAttribute('stroke-width', '2.2'); n.setAttribute('stroke-linecap', 'round'); }
  svg.append(c, p);
  return svg;
}

function emitValidity() { for (const fn of validityListeners) { try { fn(); } catch (e) { console.error(e); } } }
export function onValidity(fn) { validityListeners.add(fn); }
export function refreshValidity() { for (const f of fields.values()) check(f, { show: f.shown }); emitValidity(); }

/** Run fn(field) whenever the control commits a valid value. */
export function onCommit(id, fn) { fields.get(id).commits.add(fn); }

export function isValid(id) { const f = fields.get(id); return !f || check(f, { show: f.shown }) === null; }

/** The control's value if it is valid, else null. Numbers come from the precise number input. */
export function valueOf(id) {
  const f = fields.get(id);
  if (!f) { const n = document.getElementById(id); return n ? n.value : null; }
  if (f.kind === 'checkbox') return f.el.checked;
  if (f.kind === 'select') return f.el.value;
  if (check(f, { show: f.shown }) !== null) return null;
  return Number((f.num || f.el).value);
}

/** Set a control from code (config loaded, preset, reset). Out-of-range values are shown with
 *  their message, never clamped. `commit` runs the field's commit handlers when valid. */
export function setValue(id, v, { commit = false } = {}) {
  const f = fields.get(id);
  if (!f) return;
  if (f.kind === 'checkbox') f.el.checked = !!v;
  else if (f.kind === 'select') {
    if (![...f.el.options].some((o) => o.value === String(v))) f.el.append(el('option', { text: String(v), attrs: { value: String(v) } }));
    f.el.value = String(v);
  } else {
    if (f.num) f.num.value = String(v);
    f.el.value = String(v);
  }
  const ok = check(f, { show: true }) === null;
  if (commit && ok) runCommits(f);
}

/** Copy a slider's current value to its number input (after code moved the slider). */
export function syncFromRange(id) {
  const f = fields.get(id);
  if (f && f.num) { f.num.value = f.el.value; check(f, { show: f.shown }); }
}

function isActive(f) { return !f.el.closest('[hidden]:not([role="tabpanel"])'); }

/** Invalid, visible fields that block `gate`: [{id, label, tab, text}]. */
export function invalidFor(gate) {
  const out = [];
  for (const f of fields.values()) {
    if (f.gate !== gate || !isActive(f)) continue;
    if (check(f, { show: f.shown }) !== null) out.push({ id: f.id, label: f.label, tab: tabName(f.el), text: f.problem.text });
  }
  return out;
}

/** "Fix Steps (Train) and Seed (Data) to start." */
export function fixMessage(list, verb, here) {
  if (!list.length) return '';
  const names = list.map((x) => (x.tab && x.tab !== here ? `${x.label} (${x.tab} tab)` : x.label));
  const joined = names.length === 1 ? names[0] : `${names.slice(0, -1).join(', ')} and ${names.at(-1)}`;
  return `Fix ${joined} to ${verb}.`;
}

/** Reset every field of a section to its default and apply it (each handler runs once). */
export function resetSection(section) {
  const handlers = new Set();
  const touched = [];
  for (const f of fields.values()) {
    if (f.section !== section || typeof f.def !== 'function') continue;
    setValue(f.id, f.def());
    touched.push(f);
    for (const fn of f.commits) handlers.add([fn, f]);
  }
  const seen = new Set();
  for (const [fn, f] of handlers) { if (!seen.has(fn)) { seen.add(fn); fn(f); } }
  return touched.length;
}

export function getField(id) { return fields.get(id); }
