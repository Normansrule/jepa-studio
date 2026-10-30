// Settings panel: the settings the HTTP API may change (default model, Ollama URL, sandbox limits,
// network_allowed), what the sandbox actually enforces on this computer, where JEV keeps its data
// (with Export all and Delete all, which asks), and a read-only view of the approval table.
// Approval rules, allow_remote_ollama and allow_cloud_models can only be changed by editing
// settings.json, on purpose: no web page and no model output can loosen them (jev/settings.py).
import { el, clear, $, download } from '../ui/dom.js';
import { icon } from './icons.js';

// jev/settings.py SCHEMA bounds (the "absurd limits" guard); the backend checks them again.
export const LIMITS = [
  ['cpu_seconds', 'CPU time per code run', 'seconds', 1, 86400],
  ['wall_seconds', 'Wall-clock time per code run', 'seconds', 1, 86400],
  ['memory_mb', 'Memory per code run', 'MB', 64, 1048576],
  ['max_output_mb', 'Largest file a code run may write', 'MB', 1, 1048576],
  ['max_file_mb', 'Files bigger than this ask first', 'MB', 1, 1048576],
];

export function isLoopbackHost(host) {
  const h = String(host || '').replace(/^\[|\]$/g, '').toLowerCase();
  if (h === 'localhost' || h === '::1' || h === '0:0:0:0:0:0:0:1') return true;
  const m = /^(\d{1,3})\.(\d{1,3})\.(\d{1,3})\.(\d{1,3})$/.exec(h);
  return !!(m && Number(m[1]) === 127 && m.slice(2).every((x) => Number(x) <= 255));
}

/** Validate the Ollama URL the way jev/settings.py check_ollama_url does. Returns {error?, remote}. */
export function checkUrl(raw, allowRemote, fileAllowsRemote, settingsPath) {
  let u;
  try { u = new URL(String(raw).trim()); } catch { return { error: 'Type a full address such as http://127.0.0.1:11434.' }; }
  if (!/^https?:$/.test(u.protocol) || !u.hostname) return { error: 'The address must start with http:// or https://.' };
  const remote = !isLoopbackHost(u.hostname);
  if (!remote) return { remote };
  if (!allowRemote) return { remote, error: `${u.hostname} is not this computer. JEV only talks to a local Ollama (127.0.0.1, ::1 or localhost) unless you tick “Allow an Ollama on another machine”.` };
  if (!fileAllowsRemote) return { remote, error: `The backend still refuses it: set "allow_remote_ollama": true in ${settingsPath || 'jev-data/settings.json'} first. That switch cannot be turned on from the app, so no web page can redirect your prompts.` };
  return { remote };
}

export function checkLimit(raw, min, max) {
  const s = String(raw).trim();
  if (!/^\d+$/.test(s)) return 'Use a whole number.';
  const n = Number(s);
  if (n < min || n > max) return `Use a number from ${min.toLocaleString()} to ${max.toLocaleString()}.`;
  return null;
}

const ACTION_WORD = { run: ['check', 'Runs'], ask: ['approve', 'Asks first'], deny: ['cross', 'Refused'] };

function copyButton(text, label) {
  const b = el('button', { class: 'btn copy-btn', attrs: { type: 'button', 'aria-label': `Copy ${label}` } }, icon('copy'), el('span', { text: 'Copy' }));
  b.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(text); b.lastChild.textContent = 'Copied'; } catch { b.lastChild.textContent = 'Select and copy'; }
    setTimeout(() => { b.lastChild.textContent = 'Copy'; }, 1500);
  });
  return b;
}

export class SettingsPanel {
  /** opts.onSaved(settings), opts.onDataDeleted() */
  constructor(app, client, approvals, { onSaved = () => {}, onDataDeleted = () => {} } = {}) {
    Object.assign(this, { app, client, approvals, onSaved, onDataDeleted });
    this.s = null;
    this.status = null;
    this.models = [];
    this.buildLimits();
    this.bind();
  }

  buildLimits() {
    const box = clear($('#set-limits'));
    for (const [key, label, unit, min, max] of LIMITS) {
      const id = `set-${key.replace(/_/g, '-')}`;
      const input = el('input', { attrs: { id, type: 'number', inputmode: 'numeric', min, max, step: 1, 'aria-describedby': `${id}-hint ${id}-error`, 'data-limit': key } });
      box.append(el('div', { class: 'field' },
        el('label', { attrs: { for: id }, text: `${label} (${unit})` }),
        input,
        el('span', { class: 'hint', attrs: { id: `${id}-hint` }, text: `${min.toLocaleString()} to ${max.toLocaleString()}` }),
        el('p', { class: 'field-error', attrs: { id: `${id}-error`, hidden: true } })));
      input.addEventListener('input', () => this.validate());
    }
  }

  bind() {
    $('#set-ollama-url').addEventListener('input', () => this.validate());
    $('#set-allow-remote').addEventListener('change', () => this.validate());
    $('#set-default-model').addEventListener('change', () => this.validate());
    $('#set-network').addEventListener('change', () => this.validate());
    $('#jev-settings-form').addEventListener('submit', (e) => { e.preventDefault(); this.save(); });
    $('#set-revert').addEventListener('click', () => this.fill());
    $('#set-export').addEventListener('click', () => this.exportAll());
    $('#set-delete').addEventListener('click', () => this.deleteAll());
  }

  async load(status, models) {
    this.status = status;
    this.models = models || [];
    try {
      this.s = await this.client.getSettings();
      $('#set-load-error').hidden = true;
    } catch (e) {
      this.s = null;
      const p = $('#set-load-error');
      p.textContent = `Could not read the settings: ${e.message || e}`;
      p.hidden = false;
    }
    this.fill();
    this.renderCaps();
    this.renderPaths();
    this.renderRules();
  }

  fill() {
    const s = this.s;
    $('#set-save').disabled = !s;
    if (!s) return;
    const sel = clear($('#set-default-model'));
    const names = this.models.map((m) => String(m.name));
    if (s.default_model && !names.includes(s.default_model)) names.unshift(s.default_model);
    for (const n of names) sel.append(el('option', { attrs: { value: n }, text: this.models.some((m) => m.name === n) ? n : `${n} (not installed)` }));
    sel.value = s.default_model || '';
    $('#set-ollama-url').value = s.ollama_url || '';
    let remote = false;
    try { remote = !isLoopbackHost(new URL(s.ollama_url).hostname); } catch { /* keep false */ }
    $('#set-allow-remote').checked = remote;
    $('#set-network').checked = !!s.network_allowed;
    for (const [key] of LIMITS) $(`#set-${key.replace(/_/g, '-')}`).value = String((s.limits || {})[key] ?? '');
    const fileOnly = clear($('#set-file-only'));
    const rows = [
      ['allow_remote_ollama', s.allow_remote_ollama ? 'true' : 'false'],
      ['allow_cloud_models', s.allow_cloud_models ? 'true' : 'false'],
      ['approval_timeout_s', `${s.approval_timeout_s} (an unanswered approval counts as Deny after this)`],
      ['max_tool_rounds', String(s.max_tool_rounds)],
      ['sandbox.isolation', String((s.sandbox || {}).isolation || 'auto')],
      ['workspace_root', String(s.workspace_root || '')],
    ];
    for (const [k, v] of rows) fileOnly.append(el('li', {}, el('code', { text: k }), el('span', { text: ` ${v}` })));
    $('#set-status').textContent = '';
    this.validate();
  }

  /** Check every field; show messages next to them. Returns the patch to POST, or null. */
  validate() {
    const s = this.s;
    if (!s) return null;
    let ok = true;
    const patch = {};
    const url = $('#set-ollama-url');
    const allow = $('#set-allow-remote').checked;
    const r = checkUrl(url.value, allow, !!s.allow_remote_ollama, (s.paths || {}).settings);
    const urlErr = $('#set-ollama-url-error');
    url.setAttribute('aria-invalid', String(!!r.error));
    urlErr.hidden = !r.error;
    clear(urlErr);
    if (r.error) { urlErr.append(icon('warn'), el('span', { text: r.error })); ok = false; }
    $('#set-remote-warn').hidden = !allow;
    const cleanUrl = url.value.trim().replace(/\/+$/, '');
    if (cleanUrl !== String(s.ollama_url || '').replace(/\/+$/, '')) patch.ollama_url = cleanUrl;
    const limits = {};
    for (const [key, , , min, max] of LIMITS) {
      const id = `set-${key.replace(/_/g, '-')}`;
      const input = $(`#${id}`);
      const err = checkLimit(input.value, min, max);
      const p = $(`#${id}-error`);
      input.setAttribute('aria-invalid', String(!!err));
      p.hidden = !err;
      clear(p);
      if (err) { p.append(icon('warn'), el('span', { text: err })); ok = false; continue; }
      if (Number(input.value) !== Number((s.limits || {})[key])) limits[key] = Number(input.value);
    }
    if (Object.keys(limits).length) patch.limits = limits;
    const dm = $('#set-default-model').value;
    if (dm && dm !== s.default_model) patch.default_model = dm;
    const net = $('#set-network').checked;
    if (net !== !!s.network_allowed) patch.network_allowed = net;
    $('#set-network-warn').hidden = !net;
    const dirty = Object.keys(patch).length > 0;
    $('#set-save').disabled = !ok || !dirty;
    $('#set-revert').disabled = !dirty && ok;
    $('#set-dirty').textContent = !ok ? 'Fix the fields marked above to save.' : dirty ? 'You have unsaved changes.' : '';
    return ok ? patch : null;
  }

  async save() {
    const patch = this.validate();
    const st = $('#set-status');
    if (!patch || !Object.keys(patch).length) return;
    $('#set-save').disabled = true;
    st.textContent = 'Saving…';
    try {
      const out = await this.client.settings(patch);
      this.s = { ...this.s, ...out };
      this.fill();
      st.textContent = 'Saved. New turns use these settings.';
      this.app.notify('good', 'Settings saved', Object.keys(patch).join(', ').replace(/_/g, ' '));
      this.onSaved(this.s);
    } catch (e) {
      st.textContent = `Not saved: ${e.message || e}`;
      this.validate();
    }
  }

  renderCaps() {
    const ul = clear($('#set-caps'));
    const sb = this.status && this.status.sandbox;
    if (!sb) { ul.append(el('li', { class: 'hint', text: 'Unknown until the backend answers.' })); return; }
    const net = String(sb.network_isolation || 'unknown');
    const fs = String(sb.filesystem || 'unknown');
    const rows = [
      ['Network isolation', net, /^enforced/i.test(net) ? 'good' : 'warn'],
      ['File isolation', fs, /^(enforced|read-only|workspace)/i.test(fs) ? 'good' : 'warn'],
      ['Method', String(sb.method || 'none'), sb.method && sb.method !== 'none' ? 'good' : 'warn'],
      ['Resource limits', String(sb.rlimits || 'unknown'), /not available/i.test(String(sb.rlimits)) ? 'warn' : 'good'],
      ['Stopping runaway code', String(sb.process_tree_kill || 'unknown'), 'good'],
    ];
    for (const [k, v, tone] of rows) {
      ul.append(el('li', { class: 'cap-row', attrs: { 'data-status': tone } }, icon(tone === 'good' ? 'check' : 'warn'), el('span', { class: 'cap-k', text: k }), el('span', { class: 'cap-v', text: v })));
    }
    if (sb.probe) ul.append(el('li', { class: 'hint cap-probe', text: `Probe: ${sb.probe}` }));
  }

  renderPaths() {
    const ul = clear($('#set-paths'));
    const p = (this.s && this.s.paths) || {};
    const rows = [['Data folder', p.data_dir || (this.status && this.status.data_dir)], ['Database (SQLite)', p.db], ['Settings file', p.settings],
      ['Workspaces', p.workspaces || (this.s && this.s.workspace_root)], ['Exports', p.exports]];
    for (const [label, path] of rows) {
      if (!path) continue;
      ul.append(el('li', {}, el('span', { class: 'cmd-label', text: label }), el('div', { class: 'cmd-line' }, el('code', { text: path }), copyButton(path, `${label} path`))));
    }
    if (!ul.childNodes.length) ul.append(el('li', { class: 'hint', text: 'Unknown until the backend answers.' }));
  }

  renderRules() {
    const tb = clear($('#set-rules'));
    const rules = (this.s && this.s.rules) || (this.status && this.status.approval_rules) || [];
    if (!rules.length) { tb.append(el('tr', {}, el('td', { attrs: { colspan: 3 }, class: 'hint', text: 'Unknown until the backend answers.' }))); return; }
    for (const r of rules) {
      const [ic, word] = ACTION_WORD[r.action] || ['dot', r.action];
      const changed = r.action !== r.default;
      tb.append(el('tr', { attrs: { 'data-rule': r.rule } },
        el('td', {}, el('code', { text: r.rule }), r.locked ? el('span', { class: 'tag', text: 'Locked' }) : null),
        el('td', { text: r.covers || '' }),
        el('td', {}, el('span', { class: 'rule-action', attrs: { 'data-action': r.action } }, icon(ic), el('span', { text: word })),
          changed ? el('span', { class: 'hint', text: ` (default: ${(ACTION_WORD[r.default] || [, r.default])[1]})` }) : null)));
    }
  }

  async exportAll() {
    const b = $('#set-export');
    b.disabled = true;
    $('#set-data-status').textContent = 'Preparing the export…';
    try {
      const blob = await this.client.exportAll();
      download(`jev-data-${new Date().toISOString().slice(0, 10)}.zip`, blob, 'application/zip');
      $('#set-data-status').textContent = 'Exported: a zip with every conversation, its files, the audit log and your settings.';
    } catch (e) {
      $('#set-data-status').textContent = `Could not export: ${e.message || e}`;
    }
    b.disabled = false;
  }

  async deleteAll() {
    const keep = $('#set-keep-audit').checked;
    const st = $('#set-data-status');
    let r;
    try { r = await this.client.deleteAllData(keep); } catch (e) { st.textContent = `Could not ask: ${e.message || e}`; return; }
    const done = () => { st.textContent = 'All of JEV’s data was deleted.'; this.onDataDeleted(); };
    if (r.status === 'auto_run') { done(); return; }
    this.approvals.whenDecided(r.approval_id, (decision, res) => {
      if (decision !== 'approve') st.textContent = 'Nothing was deleted.';
      else if (res && res.ok === false) st.textContent = `Not deleted: ${res.error || res.status}`;
      else done();
    });
    this.approvals.add({ approval_id: r.approval_id, name: 'data_delete', args: { keep_audit_log: keep }, reason: 'Deleting all data removes every conversation, workspace file and export, and cannot be undone.', risk: 'approve' });
    st.textContent = 'Waiting for your approval.';
  }
}
