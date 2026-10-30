// Small inline SVG icons and the risk badge. Every badge is icon + words, never colour alone.
import { el } from '../ui/dom.js';

const NS = 'http://www.w3.org/2000/svg';

export const PATHS = {
  read: ['M2.5 12s3.5-6.5 9.5-6.5S21.5 12 21.5 12 18 18.5 12 18.5 2.5 12 2.5 12z', 'circle:12,12,3'],
  sandbox: ['M12 2.8l8.2 4.6v9.2L12 21.2l-8.2-4.6V7.4z', 'M3.8 7.4L12 12l8.2-4.6', 'M12 12v9.2'],
  approve: ['M12 2.8l7.5 3v5.9c0 4.6-3.2 8-7.5 9.5-4.3-1.5-7.5-4.9-7.5-9.5V5.8z', 'M12 8v4.6', 'M12 15.6v.01'],
  check: ['M5 12.5l4.5 4.5L19 7.5'],
  cross: ['M6.5 6.5l11 11M17.5 6.5l-11 11'],
  clock: ['circle:12,12,9', 'M12 7.5V12l3 2'],
  warn: ['M12 3.5l9.5 16.5h-19z', 'M12 10v4.5', 'M12 17.3v.01'],
  dot: ['circle:12,12,4'],
  download: ['M12 4v11', 'M7.5 10.5L12 15l4.5-4.5', 'M5 19.5h14'],
  attach: ['M16.5 7.5l-7.6 7.6a2 2 0 0 0 2.8 2.8l8-8a4 4 0 0 0-5.7-5.7l-8.1 8.1a6 6 0 0 0 8.5 8.5L20 15'],
  send: ['M4 12l16-7.5-6 16-2.5-6.5z', 'M11.5 14l3-3'],
  stop: ['rect:7,7,10,10'],
  close: ['M6.5 6.5l11 11M17.5 6.5l-11 11'],
  plus: ['M12 5v14M5 12h14'],
  list: ['M8 6.5h12M8 12h12M8 17.5h12', 'M4 6.5v.01M4 12v.01M4 17.5v.01'],
  copy: ['rect:8.5,8.5,11,11', 'M15.5 8.5V5.5a1 1 0 0 0-1-1h-9a1 1 0 0 0-1 1v9a1 1 0 0 0 1 1h3'],
  play: ['M8 5.5v13l10.5-6.5z'],
  camera: ['M4 8h3.5l1.5-2.5h6L16.5 8H20v11H4z', 'circle:12,13,3.5'],
  file: ['M6 3.5h8l4 4v13H6z', 'M14 3.5v4h4'],
  doctor: ['M12 4v16M4 12h16'],
  chevron: ['M9 6l6 6-6 6'],
};

export function icon(name, { cls = '' } = {}) {
  const svg = document.createElementNS(NS, 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', 'true');
  svg.setAttribute('focusable', 'false');
  if (cls) svg.setAttribute('class', cls);
  for (const d of PATHS[name] || PATHS.dot) {
    let n;
    if (d.startsWith('circle:')) {
      const [cx, cy, r] = d.slice(7).split(',');
      n = document.createElementNS(NS, 'circle');
      n.setAttribute('cx', cx); n.setAttribute('cy', cy); n.setAttribute('r', r);
    } else if (d.startsWith('rect:')) {
      const [x, y, w, h] = d.slice(5).split(',');
      n = document.createElementNS(NS, 'rect');
      n.setAttribute('x', x); n.setAttribute('y', y); n.setAttribute('width', w); n.setAttribute('height', h); n.setAttribute('rx', '1.5');
    } else {
      n = document.createElementNS(NS, 'path');
      n.setAttribute('d', d);
    }
    n.setAttribute('fill', 'none'); n.setAttribute('stroke', 'currentColor'); n.setAttribute('stroke-width', '2');
    n.setAttribute('stroke-linecap', 'round'); n.setAttribute('stroke-linejoin', 'round');
    svg.append(n);
  }
  return svg;
}

export const RISK = {
  read: { label: 'Read only', explain: 'Reads or computes only; changes nothing on your computer.' },
  sandbox: { label: 'Sandbox', explain: "Writes or runs code only inside this conversation's workspace folder, within time and memory limits." },
  approve: { label: 'Asks first', explain: 'Can reach outside the sandbox (network, your files, installing or deleting), so it never runs without your OK.' },
};

/** <span class="risk" data-risk="sandbox"><svg/>Sandbox</span>, with the explanation as a tooltip.
 *  Screen readers hear "Sandbox risk level" (or the whole explanation with {explain: true}). */
export function riskBadge(risk, { explain = false } = {}) {
  const r = RISK[risk] ? risk : 'approve';
  return el('span', { class: 'risk', attrs: { 'data-risk': r, title: RISK[r].explain } },
    icon(r), el('span', { class: 'risk-word', text: RISK[r].label }),
    el('span', { class: 'visually-hidden', text: explain ? ` risk level: ${RISK[r].explain}` : ' risk level' }));
}

/** Status item: icon + words, data-status good | warn | bad | idle. */
export function statusItem(node, status, text, detail = '') {
  while (node.firstChild) node.removeChild(node.firstChild);
  node.dataset.status = status;
  node.append(icon(status === 'good' ? 'check' : status === 'bad' ? 'cross' : status === 'warn' ? 'warn' : 'clock'), el('span', { text }));
  if (detail) node.append(el('span', { class: 'detail', text: detail }));
  return node;
}
