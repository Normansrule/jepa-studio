// DOM helpers. Everything user- or file-provided goes through textContent (never innerHTML),
// so file names, CSV headers and log lines can never become markup.
export const $ = (sel, root = document) => root.querySelector(sel);
export const $$ = (sel, root = document) => [...root.querySelectorAll(sel)];

/** el('div', {class: 'x', text: 'hi', on: {click: f}, attrs: {...}}, ...children) */
export function el(tag, opts = {}, ...children) {
  const n = document.createElement(tag);
  if (opts.class) n.className = opts.class;
  if (opts.text !== undefined) n.textContent = String(opts.text);
  if (opts.attrs) for (const [k, v] of Object.entries(opts.attrs)) if (v !== null && v !== undefined && v !== false) n.setAttribute(k, v === true ? '' : String(v));
  if (opts.on) for (const [k, f] of Object.entries(opts.on)) n.addEventListener(k, f);
  for (const c of children.flat()) if (c !== null && c !== undefined) n.append(c instanceof Node ? c : document.createTextNode(String(c)));
  return n;
}

export function clear(n) { while (n.firstChild) n.removeChild(n.firstChild); return n; }

/** Trigger a download of bytes/text without any network request. */
export function download(name, data, type = 'application/octet-stream') {
  const blob = data instanceof Blob ? data : new Blob([data], { type });
  const url = URL.createObjectURL(blob);
  const a = el('a', { attrs: { href: url, download: name } });
  document.body.append(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 5000);
}

export function bindRange(input, output, format = (v) => v) {
  const upd = () => { output.textContent = format(Number(input.value)); };
  input.addEventListener('input', upd);
  upd();
  return upd;
}

/**
 * Enable or disable a button and say why it is disabled, in a visible hint next to it
 * (element `#<button id>-why` unless `whyId` names a shared one, linked with aria-describedby).
 * `reason` falsy = enabled.
 */
export function gate(btn, reason, whyId = `${btn && btn.id}-why`) {
  if (!btn) return;
  btn.disabled = !!reason;
  const why = document.getElementById(whyId);
  if (!why) return;
  why.textContent = reason || '';
  why.hidden = !reason;
  const ids = new Set((btn.getAttribute('aria-describedby') || '').split(/\s+/).filter(Boolean));
  if (reason) ids.add(why.id); else ids.delete(why.id);
  if (ids.size) btn.setAttribute('aria-describedby', [...ids].join(' ')); else btn.removeAttribute('aria-describedby');
}

const ICONS = {
  good: 'M4 12l5 5L20 6',
  warn: 'M12 3l10 18H2zM12 10v5M12 18v.01',
  bad: 'M6 6l12 12M18 6L6 18',
  idle: 'M12 7v5l3 3',
};
/** Status pill: icon + text so status never rests on colour alone. */
export function setStatus(node, status, text) {
  clear(node);
  node.dataset.status = status;
  const svg = document.createElementNS('http://www.w3.org/2000/svg', 'svg');
  svg.setAttribute('viewBox', '0 0 24 24');
  svg.setAttribute('aria-hidden', 'true');
  if (status === 'idle') {
    const c = document.createElementNS(svg.namespaceURI, 'circle');
    c.setAttribute('cx', '12'); c.setAttribute('cy', '12'); c.setAttribute('r', '9');
    c.setAttribute('fill', 'none'); c.setAttribute('stroke', 'currentColor'); c.setAttribute('stroke-width', '2');
    svg.append(c);
  }
  const p = document.createElementNS(svg.namespaceURI, 'path');
  p.setAttribute('d', ICONS[status] || ICONS.idle);
  p.setAttribute('fill', 'none'); p.setAttribute('stroke', 'currentColor'); p.setAttribute('stroke-width', '2.2');
  p.setAttribute('stroke-linecap', 'round'); p.setAttribute('stroke-linejoin', 'round');
  svg.append(p);
  node.append(svg, document.createTextNode(text));
}

export function collapseStatus(status) {
  return status === 'healthy' ? 'good' : status === 'anisotropic' ? 'warn' : status ? 'bad' : 'idle';
}

export const COLLAPSE_TEXT = {
  healthy: 'Healthy', anisotropic: 'Anisotropic', 'dimensional-collapse': 'Dimensional collapse', 'complete-collapse': 'Complete collapse',
};

/** Fill a <dl class="metrics"> from [[label, value, small?]]. */
export function metrics(dl, rows) {
  clear(dl);
  for (const [k, v, small] of rows) {
    dl.append(el('div', {}, el('dt', { text: k }), el('dd', {}, String(v), small ? el('small', { text: ` ${small}` }) : null)));
  }
}
