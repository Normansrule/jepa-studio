// Small notices ("toasts") for completions and errors, in addition to the text each panel
// already shows. No dependencies. Two live regions exist from start-up so screen readers
// announce what is added: role=status (polite) for completions, role=alert for errors.
// Text is always set with textContent.
import { el, clear } from './dom.js';

let host = null, politeRegion = null, alertRegion = null;
const LIFETIME = { good: 5000, info: 5000, warn: 8000, bad: 12000 };   // paused while hovered or focused
const ICON = {
  good: 'M4 12l5 5L20 6',
  info: 'M12 8v.01M12 11v6',
  warn: 'M12 3l10 18H2zM12 10v5M12 18v.01',
  bad: 'M6 6l12 12M18 6L6 18',
};

function icon(kind) {
  const ns = 'http://www.w3.org/2000/svg';
  const svg = document.createElementNS(ns, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', 'true');
  const p = document.createElementNS(ns, 'path');
  p.setAttribute('d', ICON[kind] || ICON.info);
  p.setAttribute('fill', 'none'); p.setAttribute('stroke', 'currentColor'); p.setAttribute('stroke-width', '2.2');
  p.setAttribute('stroke-linecap', 'round'); p.setAttribute('stroke-linejoin', 'round');
  svg.append(p);
  return svg;
}

export function initToasts() {
  if (host) return;
  host = el('div', { class: 'toasts' });
  politeRegion = el('div', { class: 'toast-region', attrs: { role: 'status', 'aria-live': 'polite', 'aria-label': 'Notifications' } });
  alertRegion = el('div', { class: 'toast-region', attrs: { role: 'alert', 'aria-label': 'Errors' } });
  host.append(alertRegion, politeRegion);
  document.body.append(host);
}

/**
 * notify('good' | 'info' | 'warn' | 'bad', title, text?, {action: {label, run}})
 * Returns the toast element. Notices fade after 5-12 s (errors last longest); hovering or
 * focusing one keeps it; the panel text stays either way.
 */
export function notify(kind, title, text = '', { action = null } = {}) {
  initToasts();
  const region = kind === 'bad' ? alertRegion : politeRegion;
  const close = el('button', { class: 'toast-close', attrs: { type: 'button', 'aria-label': 'Dismiss notification' }, text: '×' });
  const body = el('div', { class: 'toast-body' },
    el('p', { class: 'toast-title', text: title }),
    text ? el('p', { class: 'toast-text', text }) : null);
  const t = el('div', { class: 'toast', attrs: { 'data-kind': kind } }, icon(kind), body, close);
  if (action) {
    body.append(el('button', { class: 'btn toast-action', attrs: { type: 'button' }, text: action.label, on: { click: () => { remove(); action.run(); } } }));
  }
  let timer = null;
  function remove() { clearTimeout(timer); t.remove(); }
  close.addEventListener('click', remove);
  region.append(t);
  // keep the stack short: drop the oldest non-error notices first
  const all = [...host.querySelectorAll('.toast')];
  if (all.length > 4) (all.find((x) => x.dataset.kind !== 'bad') || all[0]).remove();
  const life = LIFETIME[kind] ?? 6000;
  if (life) {
    const arm = () => { clearTimeout(timer); timer = setTimeout(remove, life); };
    t.addEventListener('mouseenter', () => clearTimeout(timer));
    t.addEventListener('mouseleave', arm);
    t.addEventListener('focusin', () => clearTimeout(timer));
    t.addEventListener('focusout', arm);
    arm();
  }
  return t;
}

export function clearToasts() { if (host) { clear(politeRegion); clear(alertRegion); } }
