// The approval safety net, front-end half. Anything JEV wants to do outside its sandbox arrives
// as a pending approval (from a turn's approval_request event, or from the pending list for
// actions such as a model download). One focused, modal <dialog> asks about one approval at a
// time: what will happen, why it needs permission, the exact command / path / model, every argument
// in full (scrollable) with the SHA-256 of those exact arguments, and three answers: Approve once,
// Approve for this session (only these exact arguments, in this conversation), Deny. Esc (the
// dialog's cancel) means Deny. Approving sends the SHA-256 back; the backend refuses an approval
// whose hash does not match the pending call.
// A "N waiting" badge stays in the tab bar and the JEV status strip until every request is
// answered, and the Approvals panel lists pending requests plus the audit history.
import { el, clear, $, download } from '../ui/dom.js';
import { icon, riskBadge } from './icons.js';
import { filterAudit, toCsv, dayOf } from './audit.js';

const SHELL = /(^|_)(shell|command|exec|terminal)($|_)|run_shell/;

// Characters that change how text looks without being visible: bidirectional overrides and
// isolates, zero-width characters, other format controls and C0/C1 controls (newline and tab
// excepted). On an approval card they are shown as visible \uXXXX escapes, with a warning.
const HIDDEN = /[\u0000-\u0008\u000b-\u001f\u007f-\u009f\u00ad\u061c\u115f\u1160\u17b4\u17b5\u180e\u200b-\u200f\u202a-\u202e\u2060-\u206f\u3164\ufe00-\ufe0f\ufeff\uffa0\ufff9-\ufffb]/g;

/** {text, found}: `s` with every invisible or direction-changing character written as \uXXXX. */
export function reveal(s) {
  let found = 0;
  const text = String(s ?? '').replace(HIDDEN, (c) => { found += 1; return `\\u${c.charCodeAt(0).toString(16).toUpperCase().padStart(4, '0')}`; });
  return { text, found };
}

/** Structural equality of two JSON values (numbers by value, objects by keys). */
export function jsonEqual(x, y) {
  if (x === y) return true;
  if (typeof x !== typeof y || x === null || y === null || typeof x !== 'object') return false;
  if (Array.isArray(x) !== Array.isArray(y)) return false;
  if (Array.isArray(x)) return x.length === y.length && x.every((v, i) => jsonEqual(v, y[i]));
  const kx = Object.keys(x).sort();
  const ky = Object.keys(y).sort();
  return kx.length === ky.length && kx.every((k, i) => k === ky[i] && jsonEqual(x[k], y[k]));
}

const K256 = [0x428a2f98, 0x71374491, 0xb5c0fbcf, 0xe9b5dba5, 0x3956c25b, 0x59f111f1, 0x923f82a4, 0xab1c5ed5, 0xd807aa98, 0x12835b01,
  0x243185be, 0x550c7dc3, 0x72be5d74, 0x80deb1fe, 0x9bdc06a7, 0xc19bf174, 0xe49b69c1, 0xefbe4786, 0x0fc19dc6, 0x240ca1cc, 0x2de92c6f,
  0x4a7484aa, 0x5cb0a9dc, 0x76f988da, 0x983e5152, 0xa831c66d, 0xb00327c8, 0xbf597fc7, 0xc6e00bf3, 0xd5a79147, 0x06ca6351, 0x14292967,
  0x27b70a85, 0x2e1b2138, 0x4d2c6dfc, 0x53380d13, 0x650a7354, 0x766a0abb, 0x81c2c92e, 0x92722c85, 0xa2bfe8a1, 0xa81a664b, 0xc24b8b70,
  0xc76c51a3, 0xd192e819, 0xd6990624, 0xf40e3585, 0x106aa070, 0x19a4c116, 0x1e376c08, 0x2748774c, 0x34b0bcb5, 0x391c0cb3, 0x4ed8aa4a,
  0x5b9cca4f, 0x682e6ff3, 0x748f82ee, 0x78a5636f, 0x84c87814, 0x8cc70208, 0x90befffa, 0xa4506ceb, 0xbef9a3f7, 0xc67178f2];

/** SHA-256 of bytes, in plain JavaScript (used when Web Crypto is unavailable). */
export function sha256Bytes(bytes) {
  const n = bytes.length;
  const padded = new Uint8Array(((n + 9 + 63) >> 6) << 6);
  padded.set(bytes);
  padded[n] = 0x80;
  const view = new DataView(padded.buffer);
  view.setUint32(padded.length - 8, Math.floor(n / 0x20000000));
  view.setUint32(padded.length - 4, (n << 3) >>> 0);
  const H = [0x6a09e667, 0xbb67ae85, 0x3c6ef372, 0xa54ff53a, 0x510e527f, 0x9b05688c, 0x1f83d9ab, 0x5be0cd19];
  const w = new Uint32Array(64);
  const rotr = (x, k) => (x >>> k) | (x << (32 - k));
  for (let off = 0; off < padded.length; off += 64) {
    for (let i = 0; i < 16; i++) w[i] = view.getUint32(off + i * 4);
    for (let i = 16; i < 64; i++) {
      const s0 = rotr(w[i - 15], 7) ^ rotr(w[i - 15], 18) ^ (w[i - 15] >>> 3);
      const s1 = rotr(w[i - 2], 17) ^ rotr(w[i - 2], 19) ^ (w[i - 2] >>> 10);
      w[i] = (w[i - 16] + s0 + w[i - 7] + s1) >>> 0;
    }
    let [a, b, c, d, e, f, g, h] = H;
    for (let i = 0; i < 64; i++) {
      const t1 = (h + (rotr(e, 6) ^ rotr(e, 11) ^ rotr(e, 25)) + ((e & f) ^ (~e & g)) + K256[i] + w[i]) >>> 0;
      const t2 = ((rotr(a, 2) ^ rotr(a, 13) ^ rotr(a, 22)) + ((a & b) ^ (a & c) ^ (b & c))) >>> 0;
      h = g; g = f; f = e; e = (d + t1) >>> 0; d = c; c = b; b = a; a = (t1 + t2) >>> 0;
    }
    H[0] = (H[0] + a) >>> 0; H[1] = (H[1] + b) >>> 0; H[2] = (H[2] + c) >>> 0; H[3] = (H[3] + d) >>> 0;
    H[4] = (H[4] + e) >>> 0; H[5] = (H[5] + f) >>> 0; H[6] = (H[6] + g) >>> 0; H[7] = (H[7] + h) >>> 0;
  }
  return H.map((x) => x.toString(16).padStart(8, '0')).join('');
}

/** SHA-256 (hex) of a string's UTF-8 bytes: Web Crypto when available, else sha256Bytes. */
export async function sha256Hex(text) {
  const bytes = new TextEncoder().encode(text);
  try {
    if (globalThis.crypto && globalThis.crypto.subtle) {
      const d = await globalThis.crypto.subtle.digest('SHA-256', bytes);
      return [...new Uint8Array(d)].map((b) => b.toString(16).padStart(2, '0')).join('');
    }
  } catch { /* fall back below */ }
  return sha256Bytes(bytes);
}

/** The hash this page sends when approving: computed HERE over the canonical JSON the backend
 *  sent, after checking that this JSON is exactly the arguments the card displays. */
export async function approvalHash(a) {
  if (typeof a.argsCanonical !== 'string') throw new Error('the request did not include its exact arguments, so it cannot be approved here');
  let parsed;
  try { parsed = JSON.parse(a.argsCanonical); } catch { throw new Error('the request’s arguments are not valid JSON'); }
  if (!jsonEqual(parsed, a.args)) throw new Error('the arguments shown differ from the ones that would run');
  const h = await sha256Hex(a.argsCanonical);
  if (a.argsSha256 && h !== a.argsSha256) throw new Error('the SHA-256 computed here does not match the backend’s');
  return h;
}

/** Turn a tool call into plain words: {title, what, exact, exactLabel}. */
export function describe(tool, args = {}) {
  const t = String(tool || 'tool');
  const a = args && typeof args === 'object' ? args : {};
  const s = (v) => (v === undefined || v === null ? '' : typeof v === 'string' ? v : JSON.stringify(v));
  if (SHELL.test(t)) {
    return { title: 'Allow JEV to run a command on your computer?', what: 'JEV wants to run this command outside its sandbox, with your user account’s permissions.', exact: s(a.command ?? a.cmd ?? a.script), exactLabel: 'Command' };
  }
  if (/pull|download_model/.test(t) || (/model/.test(t) && a.name && /pull|download/.test(t))) {
    return { title: 'Allow JEV to download a model?', what: 'This downloads a model from the Ollama library to your computer. Large models take several GB of disk space.', exact: s(a.name ?? a.model), exactLabel: 'Model' };
  }
  if (/import/.test(t) && (a.gguf_path || a.path)) {
    return { title: 'Allow JEV to import a model file?', what: 'This registers a GGUF file from your computer with Ollama under a new name.', exact: `${s(a.gguf_path ?? a.path)}${a.name ? `  as  ${s(a.name)}` : ''}`, exactLabel: 'File and name' };
  }
  if (t === 'data_delete') {
    return { title: 'Allow JEV to delete all of its data?', what: `This deletes every conversation, its messages and workspace files, science renders, exports and imported-model records${a.keep_audit_log ? '' : ', and the audit log'}. settings.json and the models in Ollama stay. It cannot be undone.`, exact: s(a.data_dir || 'jev-data'), exactLabel: 'Data folder' };
  }
  if (/model/.test(t) && /delete|remove/.test(t)) {
    return { title: 'Allow JEV to delete a model?', what: 'This removes the model from Ollama, including its files on disk (unless another model shares them). You can download it again later.', exact: s(a.name ?? a.model), exactLabel: 'Model' };
  }
  if (/delete|remove/.test(t)) {
    const target = a.name ?? a.model ?? a.path ?? a.conversation_id ?? a.id;
    return { title: 'Allow JEV to delete something?', what: /conversation/.test(t) ? 'This deletes the conversation and its workspace folder, including every file JEV made in it. It cannot be undone.' : 'This deletes the item below. It cannot be undone.', exact: s(target), exactLabel: 'What' };
  }
  if (/install|pip/.test(t)) {
    return { title: 'Allow JEV to install a package?', what: 'This downloads and installs software into JEV’s Python environment.', exact: s(a.packages ?? a.package ?? a.name ?? a.requirement), exactLabel: 'Package' };
  }
  if (/fetch|http|web|url|download|search/.test(t)) {
    return { title: 'Allow JEV to use the internet?', what: 'This sends a request from your computer to the address below. The site sees your IP address.', exact: s(a.url ?? a.query ?? a.address), exactLabel: 'Address' };
  }
  if (/write|save|copy|move/.test(t) && (a.path || a.dest)) {
    return { title: 'Allow JEV to write a file outside its workspace?', what: 'This creates or replaces a file in a folder JEV does not normally write to.', exact: s(a.path ?? a.dest), exactLabel: 'Path' };
  }
  if (/read|open|list/.test(t) && a.path) {
    return { title: 'Allow JEV to read a file outside its workspace?', what: 'This lets JEV read a file or folder on your computer and use its contents in the conversation.', exact: s(a.path), exactLabel: 'Path' };
  }
  const key = ['command', 'path', 'url', 'name', 'model', 'code'].find((k) => a[k] !== undefined);
  return { title: `Allow JEV to run ${t}?`, what: 'This tool can reach outside the sandbox, so it needs your OK each time.', exact: key ? s(a[key]) : '', exactLabel: key ? key[0].toUpperCase() + key.slice(1) : '' };
}

const STATUS_WORD = {
  pending: ['clock', 'Waiting'], approved: ['check', 'Approved'], approve: ['check', 'Approved'], allowed: ['check', 'Approved'],
  denied: ['cross', 'Denied'], deny: ['cross', 'Denied'], refused: ['cross', 'Refused by policy'], expired: ['clock', 'Expired'],
  auto: ['check', 'Ran automatically'], auto_run: ['check', 'Ran automatically'], remembered: ['check', 'Approved earlier this session'],
  auto_deny: ['cross', 'Refused by policy'], cancelled: ['cross', 'Cancelled'], failed: ['warn', 'Failed to start'],
};
const HISTORY_ROWS = 500;     // rows drawn at once; the CSV has every match

export class Approvals {
  constructor(app, client, { onDecided = () => {}, canAutoOpen = () => true } = {}) {
    this.app = app;
    this.client = client;
    this.onDecided = onDecided;
    this.canAutoOpen = canAutoOpen;
    this.pending = new Map();          // id -> approval
    this.current = null;
    this.opener = null;
    this.busy = false;
    this.listeners = new Set();
    this.waiters = new Map();          // id -> fn(decision, response), for callers that started the action
    this.dlg = $('#jev-approval');
    this.history = null;
    this.bind();
    this.bindAudit();
  }

  onChange(f) { this.listeners.add(f); }
  whenDecided(id, f) { this.waiters.set(id, f); }
  emit() { for (const f of this.listeners) { try { f(this.pending.size); } catch (e) { console.error(e); } } }

  bind() {
    const d = this.dlg;
    $('#appr-once').addEventListener('click', () => this.answer('approve', null));
    $('#appr-session').addEventListener('click', () => this.answer('approve', 'session'));
    $('#appr-deny').addEventListener('click', () => this.answer('deny', null));
    d.addEventListener('cancel', (e) => { e.preventDefault(); if (!this.busy) this.answer('deny', null); });   // Esc = Deny
    d.addEventListener('close', () => {
      const o = this.opener;
      this.opener = null;
      this.current = null;
      if (o && o.isConnected) o.focus();
    });
  }

  /** Normalize an approval from either source. */
  add(raw) {
    const id = raw.approval_id || raw.id;
    if (!id || this.pending.has(id)) return;
    const a = {
      id, tool: raw.name || raw.tool || 'tool', args: raw.args || {}, argsSha256: raw.args_sha256 || null, argsCanonical: typeof raw.args_canonical === 'string' ? raw.args_canonical : null, reason: raw.reason || '', risk: raw.risk || 'approve',
      conversation_id: raw.conversation_id || null, turn_id: raw.turn_id || null, call_id: raw.call_id || null,
      created: raw.created || Date.now() / 1000,
    };
    this.pending.set(id, a);
    if (a.argsCanonical === null) this.refreshOne(id);   // added from an action's reply: fetch the exact arguments
    this.emit();
    this.renderPanel();
    if (!this.dlg.open && this.canAutoOpen()) this.open();
    else if (!this.dlg.open) {
      this.app.notify('warn', 'JEV is waiting for your approval', describe(a.tool, a.args).title, { action: { label: 'Review', run: () => this.open() } });
    } else this.renderCount();
  }

  /** Replace an approval's arguments with the backend's exact ones (and its canonical JSON). */
  async refreshOne(id) {
    let list;
    try { list = (await this.client.approvals('pending')).approvals || []; } catch { return; }
    const x = list.find((r) => r.id === id);
    const a = this.pending.get(id);
    if (!x || !a) return;
    Object.assign(a, { tool: x.tool || a.tool, args: x.args || {}, argsSha256: x.args_sha256 || null,
      argsCanonical: typeof x.args_canonical === 'string' ? x.args_canonical : null, reason: x.reason || a.reason });
    if (this.current === a && this.dlg.open) this.fill(a);
  }

  /** Reconcile with the backend's pending list (requests decided elsewhere disappear). */
  async sync() {
    let list;
    try { list = (await this.client.approvals('pending')).approvals || []; } catch { return; }
    const ids = new Set(list.map((x) => x.id));
    let changed = false;
    for (const id of [...this.pending.keys()]) {
      if (!ids.has(id) && (!this.current || this.current.id !== id)) { this.pending.delete(id); changed = true; }
    }
    for (const x of list) if (!this.pending.has(x.id)) this.add(x);
    if (changed) { this.emit(); this.renderPanel(); }
  }

  /** Show the dialog for `id` (or the oldest pending request). */
  open(id = null, opener = null) {
    if (!this.pending.size) return;
    const a = (id && this.pending.get(id)) || [...this.pending.values()][0];
    this.current = a;
    if (!this.dlg.open) this.opener = opener || (document.activeElement instanceof HTMLElement ? document.activeElement : null);
    this.fill(a);
    if (!this.dlg.open) this.dlg.showModal();
    $('#appr-deny').focus();
  }

  renderCount() {
    const n = this.pending.size;
    $('#appr-n').textContent = n > 1 ? `${[...this.pending.keys()].indexOf(this.current?.id) + 1} of ${n} requests` : '';
  }

  fill(a) {
    const d = describe(a.tool, a.args);
    $('#appr-title').textContent = d.title;
    $('#appr-what').textContent = d.what;
    $('#appr-why').textContent = a.reason || 'This tool is marked "Asks first": it can reach outside the conversation’s sandbox folder.';
    $('#appr-exact-label').textContent = d.exactLabel || 'Details';
    const exact = $('#appr-exact');
    const shown = reveal(d.exact || JSON.stringify(a.args, null, 2));
    exact.textContent = shown.text;
    $('#appr-tool').textContent = a.tool;
    clear($('#appr-risk')).append(riskBadge(a.risk || 'approve', { explain: true }));
    const all = reveal(JSON.stringify(a.args, null, 2));
    $('#appr-args').textContent = all.text;
    const warn = $('#appr-warn');
    if (warn) {
      const n = shown.found + all.found;
      warn.hidden = !n;
      warn.textContent = n ? 'Careful: this request contains invisible or text-direction characters (shown above as \\uXXXX). Text that looks harmless may not be what would run.' : '';
    }
    const hash = $('#appr-hash');
    if (hash) {
      hash.textContent = 'computing…';
      approvalHash(a).then((h) => { if (this.current === a) hash.textContent = h; })
        .catch((e) => { if (this.current === a) hash.textContent = `cannot verify: ${e.message || e}`; });
    }
    $('#appr-error').hidden = true;
    for (const b of this.dlg.querySelectorAll('.appr-actions .btn')) b.disabled = false;
    this.renderCount();
  }

  async answer(decision, remember) {
    const a = this.current;
    if (!a || this.busy) return;
    this.busy = true;
    for (const b of this.dlg.querySelectorAll('.appr-actions .btn')) b.disabled = true;
    try {
      // the hash is computed here, over what this card shows; the backend refuses a mismatch
      const sha = decision === 'approve' ? await approvalHash(a) : null;
      const r = await this.client.decide(a.id, decision, remember, sha);
      this.pending.delete(a.id);
      this.emit();
      this.renderPanel();
      const what = describe(a.tool, a.args).exact || a.tool;
      if (r && r.ok === false) this.app.notify('warn', r.status === 'failed' ? 'Approved, but it could not start' : `Already ${r.status || 'decided'}`, r.error || what);
      else this.app.notify(decision === 'approve' ? 'good' : 'info', decision === 'approve' ? (remember ? 'Approved for this session' : 'Approved once') : 'Denied', r && r.note ? `${what}. Note: ${r.note}.` : what);
      try { this.onDecided(a, decision, r || {}); } catch (e) { console.error(e); }
      const w = this.waiters.get(a.id);
      if (w) { this.waiters.delete(a.id); try { w(decision, r || {}); } catch (e) { console.error(e); } }
      if (this.pending.size) this.open();
      else this.dlg.close();
    } catch (e) {
      const err = $('#appr-error');
      err.textContent = `Your answer was not saved: ${e.message || e}. Try again.`;
      err.hidden = false;
      for (const b of this.dlg.querySelectorAll('.appr-actions .btn')) b.disabled = false;
    } finally {
      this.busy = false;
    }
  }

  // ------------------------------------------------------------------ Approvals panel
  renderPanel() {
    const box = $('#jev-pending');
    if (!box) return;
    clear(box);
    if (!this.pending.size) { box.append(el('p', { class: 'hint', text: 'Nothing is waiting for you.' })); return; }
    const ul = el('ul', { class: 'appr-list' });
    for (const a of this.pending.values()) {
      const d = describe(a.tool, a.args);
      const review = el('button', { class: 'btn', attrs: { type: 'button' }, text: 'Review' });
      review.addEventListener('click', () => this.open(a.id, review));
      ul.append(el('li', {}, el('div', { class: 'appr-li-text' }, el('strong', { text: d.title }), el('code', { text: d.exact || a.tool })), riskBadge(a.risk), review));
    }
    box.append(ul);
  }

  // ------------------------------------------------------------------ audit log (History)
  bindAudit() {
    for (const id of ['#audit-decision', '#audit-tool', '#audit-from', '#audit-to']) {
      const n = $(id);
      if (n) n.addEventListener('change', () => this.renderHistory({ refetch: false }));
    }
    const csv = $('#audit-csv');
    if (csv) csv.addEventListener('click', () => this.exportCsv());
    const reset = $('#audit-reset');
    if (reset) reset.addEventListener('click', () => {
      for (const id of ['#audit-decision', '#audit-tool', '#audit-from', '#audit-to']) $(id).value = '';
      this.renderHistory({ refetch: false });
    });
  }

  auditFilter() {
    const v = (id) => ($(id) ? $(id).value : '');
    return { decision: v('#audit-decision'), tool: v('#audit-tool'), from: v('#audit-from'), to: v('#audit-to') };
  }

  filtered() {
    return filterAudit(this.history || [], this.auditFilter()).sort((x, y) => (y.created || 0) - (x.created || 0));
  }

  fillToolFilter() {
    const sel = $('#audit-tool');
    if (!sel) return;
    const cur = sel.value;
    clear(sel).append(el('option', { attrs: { value: '' }, text: 'All tools' }));
    for (const t of [...new Set((this.history || []).map((a) => String(a.tool || '')).filter(Boolean))].sort()) {
      sel.append(el('option', { attrs: { value: t }, text: t }));
    }
    sel.value = [...sel.options].some((o) => o.value === cur) ? cur : '';
  }

  async renderHistory({ refetch = true } = {}) {
    const body = $('#jev-history');
    if (!body) return;
    if (refetch || !this.history) {
      try { this.history = (await this.client.approvals(null, 5000)).approvals || []; } catch (e) {
        clear(body).append(el('tr', {}, el('td', { attrs: { colspan: 4 }, class: 'hint', text: `Could not read the audit log: ${e.message || e}` })));
        return;
      }
      this.fillToolFilter();
    }
    const rows = this.filtered();
    const count = $('#audit-count');
    if (count) count.textContent = `${rows.length} of ${this.history.length} decision${this.history.length === 1 ? '' : 's'}${rows.length > HISTORY_ROWS ? `; the newest ${HISTORY_ROWS} are shown, the CSV has all` : ''}`;
    if ($('#audit-csv')) $('#audit-csv').disabled = !rows.length;
    clear(body);
    if (!rows.length) { body.append(el('tr', {}, el('td', { attrs: { colspan: 4 }, class: 'hint', text: this.history.length ? 'No decisions match these filters.' : 'No decisions yet.' }))); return; }
    for (const a of rows.slice(0, HISTORY_ROWS)) {
      const st = String(a.status || 'pending').toLowerCase();
      const [ic, word] = STATUS_WORD[st] || ['dot', st];
      const when = a.created ? new Date(a.created * (a.created < 1e12 ? 1000 : 1)).toLocaleString() : '';
      const d = describe(a.tool, a.args);
      const by = a.decided_by && !['policy', 'user'].includes(a.decided_by) ? ` · ${a.decided_by}` : '';
      body.append(el('tr', { attrs: { 'data-status': st, 'data-day': dayOf(a.created) } },
        el('td', { text: when }),
        el('td', {}, el('code', { text: String(a.tool || '') })),
        el('td', { class: 'appr-exact-cell', text: d.exact || '' }),
        el('td', {}, el('span', { class: 'appr-status', attrs: { 'data-status': st } }, icon(ic), el('span', { text: `${word}${a.remember === 'session' ? ' (session)' : ''}${by}` })))));
    }
  }

  exportCsv() {
    const rows = this.filtered();
    if (!rows.length) return;
    download(`jev-audit-${new Date().toISOString().slice(0, 10)}.csv`, toCsv(rows), 'text/csv;charset=utf-8');
    this.app.notify('good', 'Audit log exported', `${rows.length} decision${rows.length === 1 ? '' : 's'} as CSV.`);
  }
}
