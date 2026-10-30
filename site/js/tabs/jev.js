// JEV tab: the local AI agent. Desktop tier only (it needs the local backend and Ollama); the web
// tier shows an honest "get the desktop app" panel instead. Pieces live in site/js/jev/:
//   client.js     /api/jev/* calls (token header, blob URLs for artifacts)
//   turn.js       one streamed turn: text + tool cards (risk badge, args, progress, result, files)
//   approvals.js  the approval dialog, "N waiting" badge, pending list and audit history
//   gallery.js    per-conversation files (images + lightbox, video, audio, code, downloads)
//   doctor.js     status strip and the Doctor view (fixes, per-OS commands)
import { $, $$, el, clear, download } from '../ui/dom.js';
import { getPref, setPref } from '../ui/prefs.js';
import { collectDropped } from '../ui/dropzone.js';
import { JevClient, follow, fmtBytes, MAX_UPLOAD_BYTES } from '../jev/client.js';
import { TurnView, userMessage, toolCard } from '../jev/turn.js';
import { Approvals } from '../jev/approvals.js';
import { Gallery } from '../jev/gallery.js';
import { renderStrip, renderDoctor, bindDoctorOs } from '../jev/doctor.js';
import { renderAnswer } from '../jev/highlight.js';
import { icon } from '../jev/icons.js';

let app, client, approvals, gallery;
let status = null, statusErr = null, shellStatus = null;
let convId = null, turn = null, follower = null, model = '';
let attachments = [];       // [{name, path, bytes, state: 'uploading'|'ready'|'error', error}]
let pollTimer = null, statusTimer = null;

const STARTERS = [
  ['Explain …', 'Explain '],
  ['Write and run code that …', 'Write and run Python code that '],
  ['Make a narrated video about …', 'Make a short narrated video about '],
  ['Render a spinning black hole', 'Render a spinning black hole seen from just above the disk, as an image and a 10-second video.'],
  ['Simulate a double pendulum', 'Simulate a double pendulum for 20 seconds and show me the trajectory.'],
  ['Find the best moments in my footage', 'Find the best moments in my footage and cut them into a 30-second highlight video.'],
];

export const RELEASES = 'https://github.com/Normansrule/jev-jepa-studio/releases/latest';

export function init(a) {
  app = a;
  if (app.tier !== 'desktop') { initWeb(); return; }
  $('#jev-web').hidden = true;
  $('#jev-desk').hidden = false;
  client = new JevClient(app.backend);
  approvals = new Approvals(app, client, {
    canAutoOpen: () => ['jev', 'science'].includes(app.router ? app.router.current() : 'jev'),
    onDecided: decided,
  });
  approvals.onChange(renderWaiting);
  gallery = new Gallery($('#jev-gallery'), client, { count: $('#jev-gallery-count') });
  app.jev = { client, approvals, available: () => !!status };
  renderWaiting(0);
  buildStarters();
  bindComposer();
  bindPanels();
  bindConversations();
  bindDoctorOs(() => renderDoctor(status, statusErr, shellStatus));
  refreshStatus();
  loadConversations();
  statusTimer = setInterval(() => { if (document.visibilityState === 'visible') refreshStatus({ quiet: true }); }, 15000);
  pollTimer = setInterval(() => { if (status && document.visibilityState === 'visible') approvals.sync(); }, 3000);
}

export function show() {
  if (app.tier !== 'desktop') return;
  if (approvals.pending.size && !$('#jev-approval').open) approvals.open();
}

// ------------------------------------------------------------------ web tier
function initWeb() {
  $('#jev-web').hidden = false;
  $('#jev-desk').hidden = true;
  $('#jev-goto-science').addEventListener('click', () => app.goto('science'));
}

// ------------------------------------------------------------------ status + doctor
async function refreshStatus({ quiet = false } = {}) {
  try {
    status = await client.status();
    statusErr = null;
  } catch (e) {
    status = null;
    statusErr = e;
  }
  if (globalThis.__TAURI__ && __TAURI__.core) {
    try { shellStatus = await __TAURI__.core.invoke('backend_status'); } catch { shellStatus = null; }
  }
  if (status) await refreshModels();
  renderStrip(status, statusErr, model);
  if (!$('#jev-doctor').hidden || !quiet) renderDoctor(status, statusErr, shellStatus);
  const ready = !!(status && status.ollama && status.ollama.reachable && model);
  const why = !status ? 'JEV’s backend is not available. Open Doctor for details.'
    : !(status.ollama && status.ollama.reachable) ? 'Ollama is not running. Open Doctor to start it.'
      : !model ? 'Download or pick a model first.' : '';
  const w = $('#jev-send-why');
  w.textContent = why;
  w.hidden = !why;
  $('#jev-composer').dataset.ready = String(ready);
  updateSend();
}

async function refreshModels() {
  let list = [];
  try { list = (await client.models()).models || []; } catch { list = (status.models || []).map((m) => (typeof m === 'string' ? { name: m } : m)); }
  const sel = $('#jev-model');
  const prev = getPref('jev-model');
  clear(sel);
  if (!list.length) sel.append(el('option', { attrs: { value: '' }, text: 'No models yet' }));
  for (const m of list) {
    const bits = [fmtBytes(Number(m.size_bytes || m.size)), m.parameter_size].filter(Boolean).join(', ');
    sel.append(el('option', { attrs: { value: m.name }, text: bits ? `${m.name} (${bits})` : m.name }));
  }
  const names = list.map((m) => m.name);
  model = names.includes(model) ? model : names.includes(prev) ? prev : names.includes(status.default_model) ? status.default_model : names[0] || '';
  sel.value = model;
  sel.disabled = !list.length;
}

function renderWaiting(n) {
  const b = $('#jev-waiting');
  b.textContent = `${n} waiting`;
  b.hidden = !n;
  const t = $('#jev-tab-count');
  t.hidden = !n;
  t.textContent = String(n);
  $('#tab-jev').setAttribute('aria-label', n ? `JEV, ${n} approval${n === 1 ? '' : 's'} waiting` : 'JEV');
}

function togglePanel(btn, panel, onOpen) {
  const open = panel.hidden;
  panel.hidden = !open;
  btn.setAttribute('aria-expanded', String(open));
  if (open) { if (onOpen) onOpen(); panel.focus({ preventScroll: false }); }
}

function bindPanels() {
  $('#jev-doctor-btn').addEventListener('click', () => togglePanel($('#jev-doctor-btn'), $('#jev-doctor'), () => renderDoctor(status, statusErr, shellStatus)));
  $('#jev-approvals-btn').addEventListener('click', () => togglePanel($('#jev-approvals-btn'), $('#jev-approvals'), () => { approvals.renderPanel(); approvals.renderHistory(); }));
  $('#doctor-recheck').addEventListener('click', async () => {
    $('#doctor-recheck').disabled = true;
    await refreshStatus();
    $('#doctor-recheck').disabled = false;
    app.notify(status && status.ollama && status.ollama.reachable ? 'good' : 'warn', 'Checked again', status ? (status.ollama && status.ollama.reachable ? 'Ollama is running.' : 'Ollama is still not reachable.') : 'The JEV backend did not answer.');
  });
  $('#history-refresh').addEventListener('click', () => approvals.renderHistory());
  for (const b of $$('[data-close-panel]')) b.addEventListener('click', () => {
    const p = document.getElementById(b.dataset.closePanel);
    const opener = document.querySelector(`[aria-controls="${p.id}"]`);
    p.hidden = true;
    if (opener) { opener.setAttribute('aria-expanded', 'false'); opener.focus(); }
  });
  $('#jev-model').addEventListener('change', (e) => { model = e.target.value; setPref('jev-model', model); renderStrip(status, statusErr, model); refreshStatus({ quiet: true }); });
  $('#jev-pull-btn').addEventListener('click', () => {
    const f = $('#jev-pull-form');
    f.hidden = !f.hidden;
    $('#jev-pull-btn').setAttribute('aria-expanded', String(!f.hidden));
    if (!f.hidden) $('#jev-pull-name').focus();
  });
  $('#jev-pull-go').addEventListener('click', pullModel);
  $('#jev-pull-name').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); pullModel(); } });
}

async function pullModel() {
  const name = $('#jev-pull-name').value.trim();
  const st = $('#jev-pull-status');
  if (!/^[\w.\-/:]{1,120}$/.test(name)) { st.textContent = 'Type a model name such as qwen2.5:7b.'; return; }
  st.textContent = `Asking to download ${name}…`;
  try {
    const r = await client.pullModel(name);
    if (r.approval_id) approvals.add({ approval_id: r.approval_id, name: 'pull_model', args: { name }, reason: 'Downloading a model uses the internet and several GB of disk space, so JEV always asks first.', risk: 'approve' });
    st.textContent = `Waiting for your approval to download ${name}.`;
  } catch (e) {
    st.textContent = `Could not request the download: ${e.message || e}`;
  }
}

/** After an approval: a model download continues as a job (job_id) or shows up in the list. */
function decided(a, decision, r) {
  const st = $('#jev-pull-status');
  if (!/pull|download_model/.test(a.tool)) {
    if (/delete/.test(a.tool) && decision === 'approve') setTimeout(loadConversations, 800);
    return;
  }
  const name = (a.args && (a.args.name || a.args.model)) || 'the model';
  if (decision !== 'approve') { st.textContent = `Download of ${name} denied.`; return; }
  if (r && r.job_id) {
    st.textContent = `Downloading ${name}…`;
    const bar = $('#jev-pull-bar');
    bar.hidden = false;
    follow((since) => client.jobEvents(r.job_id, since), (ev) => {
      if (ev.event === 'progress' && (ev.fraction === null || ev.fraction === undefined)) {
        if (ev.message) st.textContent = `Downloading ${name}: ${ev.message}`;
      } else if (ev.event === 'progress') {
        const pct = Math.round((Number(ev.fraction) || 0) * 100);
        bar.firstChild.style.width = `${pct}%`;
        bar.setAttribute('aria-valuenow', String(pct));
        st.textContent = `Downloading ${name}: ${pct}%${ev.message ? `, ${ev.message}` : ''}`;
      } else if (ev.event === 'done' || ev.event === 'tool_result') {
        st.textContent = `${name} is ready.`;
        bar.hidden = true;
        refreshStatus({ quiet: true });
      } else if (ev.event === 'error') {
        st.textContent = `Download failed: ${ev.message}`;
        bar.hidden = true;
      }
    }, { fast: 1000 });
  } else {
    st.textContent = `Downloading ${name}; it appears in the model menu when it finishes.`;
    let tries = 0;
    const t = setInterval(async () => {
      tries += 1;
      await refreshStatus({ quiet: true });
      const names = [...$('#jev-model').options].map((o) => o.value);
      if (names.includes(name) || tries > 200) { clearInterval(t); if (names.includes(name)) st.textContent = `${name} is ready.`; }
    }, 3000);
  }
}

// ------------------------------------------------------------------ conversations
function bindConversations() {
  $('#jev-new').addEventListener('click', newConversation);
  $('#jev-export').addEventListener('click', async () => {
    if (!convId) return;
    try {
      download(`jev-conversation-${convId}.zip`, await client.exportConversation(convId), 'application/zip');
      app.notify('good', 'Conversation exported', 'A zip with the messages, tool calls, approvals and files.');
    } catch (e) { app.notify('bad', 'Could not export the conversation', String(e.message || e)); }
  });
  $('#jev-new-2').addEventListener('click', () => { $('#jev-convs-dialog').close(); newConversation(); });
  const dlg = $('#jev-convs-dialog');
  $('#jev-convs-toggle').addEventListener('click', () => { dlg.showModal(); const cur = dlg.querySelector('[aria-current="true"]') || dlg.querySelector('button'); if (cur) cur.focus(); });
  $('#jev-convs-close').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });
  dlg.addEventListener('close', () => $('#jev-convs-toggle').focus());
}

let convCache = [];
async function loadConversations() {
  try { convCache = (await client.conversations()).conversations || []; } catch { convCache = []; }
  const cur = convCache.find((c) => c.id === convId);
  $('#jev-export').hidden = !convId;
  if (cur && cur.title) $('#jev-conv-title').textContent = cur.title;
  renderConversationLists();
}

function renderConversationLists() {
  for (const ul of [$('#jev-conv-list'), $('#jev-conv-list-2')]) {
    clear(ul);
    if (!convCache.length) { ul.append(el('li', { class: 'hint conv-empty', text: 'No conversations yet.' })); continue; }
    for (const c of [...convCache].sort((x, y) => String(y.updated || '').localeCompare(String(x.updated || '')))) {
      const when = c.updated ? new Date(typeof c.updated === 'number' ? c.updated * (c.updated < 1e12 ? 1000 : 1) : c.updated) : null;
      const open = el('button', { class: 'conv', attrs: { type: 'button', 'aria-current': c.id === convId ? 'true' : null } },
        el('span', { class: 'conv-title', text: c.title || 'Untitled' }),
        el('span', { class: 'hint', text: [when && !Number.isNaN(+when) ? when.toLocaleDateString() : '', c.model || ''].filter(Boolean).join(', ') }));
      open.addEventListener('click', () => { const d = $('#jev-convs-dialog'); if (d.open) d.close(); openConversation(c.id); });
      const del = el('button', { class: 'icon-btn conv-del', attrs: { type: 'button', 'aria-label': `Delete conversation ${c.title || 'Untitled'}` } }, icon('close'));
      del.addEventListener('click', () => deleteConversation(c));
      ul.append(el('li', {}, open, del));
    }
  }
}

async function deleteConversation(c) {
  try {
    const r = await client.deleteConversation(c.id);
    if (r.approval_id) approvals.add({ approval_id: r.approval_id, name: 'delete_conversation', args: { conversation_id: c.id, title: c.title }, reason: 'Deleting removes the conversation and every file in its workspace folder.', risk: 'approve' });
  } catch (e) {
    app.notify('bad', 'Could not delete the conversation', String(e.message || e));
  }
}

function resetLog() {
  if (follower) { follower.stop(); follower = null; }
  turn = null;
  clear($('#jev-log'));
  client.clearBlobs();
  gallery.renderEmpty();
  attachments = [];
  renderAttachments();
  $('#jev-starters').hidden = false;
  $('#jev-log').setAttribute('aria-busy', 'false');
  $('#jev-log').dataset.pinned = 'true';
}

function newConversation() {
  convId = null;
  resetLog();
  $('#jev-conv-title').textContent = 'New conversation';
  $('#jev-export').hidden = true;
  renderConversationLists();
  $('#jev-input').focus();
}

async function openConversation(id) {
  resetLog();
  convId = id;
  renderConversationLists();
  let c;
  try { c = await client.conversation(id); } catch (e) { app.notify('bad', 'Could not open the conversation', String(e.message || e)); return; }
  $('#jev-conv-title').textContent = c.title || 'Untitled';
  $('#jev-export').hidden = false;
  $('#jev-starters').hidden = (c.messages || []).length > 0;
  const log = $('#jev-log');
  // Stored messages (jev/store.py): {role, content, tool_calls (JSON), tool_name}. A tool message
  // answers the oldest tool call still waiting for its result.
  const waiting = [];
  for (const m of c.messages || []) {
    if (m.role === 'user') log.append(userMessage(String(m.content || ''), (m.attachments || []).map((p) => ({ path: p }))));
    else if (m.role === 'assistant') {
      const body = el('div', { class: 'msg-body' });
      if (m.content) body.append(el('div', { class: 'seg-text' }, renderAnswer(m.content)));
      let calls = m.tool_calls;
      if (typeof calls === 'string') { try { calls = JSON.parse(calls); } catch { calls = []; } }
      for (const tc of Array.isArray(calls) ? calls : []) {
        const fn = tc.function || tc;
        let args = fn.arguments;
        if (typeof args === 'string') { try { args = JSON.parse(args); } catch { /* keep the string */ } }
        const card = toolCard({ name: fn.name, risk: tc.risk || 'read', args, state: 'done', client });
        waiting.push(card);
        body.append(card);
      }
      if (body.childNodes.length) log.append(el('article', { class: 'msg msg-jev' }, el('h4', { class: 'who' }, el('span', { text: 'JEV' }), el('span', { class: 'hint who-model', text: m.model || '' })), body));
    } else if (m.role === 'tool') {
      const card = waiting.shift();
      if (card && m.content) {
        const box = card.querySelector('.tc-result');
        box.hidden = false;
        box.append(el('pre', { class: 'tc-out', attrs: { tabindex: '0' }, text: String(m.content).slice(0, 6000) }));
      }
    }
  }
  log.scrollTop = log.scrollHeight;
  let arts = c.artifacts;
  if (!arts) { try { arts = (await client.artifacts(id)).artifacts; } catch { arts = []; } }
  gallery.setAll(arts || []);
}

// ------------------------------------------------------------------ composer
function buildStarters() {
  const box = $('#jev-starter-list');
  for (const [label, text] of STARTERS) {
    const b = el('button', { class: 'starter', attrs: { type: 'button' }, text: label });
    b.addEventListener('click', () => {
      const t = $('#jev-input');
      t.value = text;
      t.focus();
      t.setSelectionRange(text.length, text.length);
      autosize();
      updateSend();
    });
    box.append(el('li', {}, b));
  }
}

function autosize() {
  const t = $('#jev-input');
  t.style.height = 'auto';
  t.style.height = `${Math.min(240, t.scrollHeight + 2)}px`;
}

function busy() { return !!(turn && !turn.ended); }

function updateSend() {
  const ready = $('#jev-composer').dataset.ready === 'true';
  const text = $('#jev-input').value.trim();
  const uploading = attachments.some((a) => a.state === 'uploading');
  $('#jev-send').disabled = !ready || !text || busy() || uploading;
  $('#jev-stop').hidden = !busy();
  $('#jev-send').hidden = busy();
}

function bindComposer() {
  const t = $('#jev-input');
  t.addEventListener('input', () => { autosize(); updateSend(); });
  t.addEventListener('keydown', (e) => {
    if (e.key === 'Enter' && !e.shiftKey && !e.isComposing) { e.preventDefault(); if (!$('#jev-send').disabled) send(); }
  });
  $('#jev-send').addEventListener('click', send);
  $('#jev-stop').addEventListener('click', stop);
  $('#jev-files').addEventListener('change', (e) => { addFiles([...e.target.files]); e.target.value = ''; });
  const zone = $('#jev-composer');
  let depth = 0;
  zone.addEventListener('dragenter', (e) => { if ([...(e.dataTransfer?.types || [])].includes('Files')) { depth++; zone.classList.add('is-over'); e.preventDefault(); } });
  zone.addEventListener('dragover', (e) => { if ([...(e.dataTransfer?.types || [])].includes('Files')) e.preventDefault(); });
  zone.addEventListener('dragleave', () => { depth = Math.max(0, depth - 1); if (!depth) zone.classList.remove('is-over'); });
  zone.addEventListener('drop', async (e) => {
    e.preventDefault();
    depth = 0;
    zone.classList.remove('is-over');
    const { files } = await collectDropped(e.dataTransfer, { maxFiles: 20, maxDepth: 2 });
    addFiles(files);
  });
}

async function ensureConversation() {
  if (convId) return convId;
  const r = await client.createConversation(model || null);
  convId = r.id || r.conversation_id;
  $('#jev-conv-title').textContent = r.title || 'New conversation';
  loadConversations();
  return convId;
}

function renderAttachments() {
  const ul = clear($('#jev-attach'));
  ul.hidden = !attachments.length;
  for (const a of attachments) {
    const rm = el('button', { class: 'chip-x', attrs: { type: 'button', 'aria-label': `Remove ${a.name}` } }, icon('close'));
    rm.addEventListener('click', () => { attachments = attachments.filter((x) => x !== a); renderAttachments(); updateSend(); $('#jev-input').focus(); });
    const state = a.state === 'uploading' ? 'uploading…' : a.state === 'error' ? a.error : fmtBytes(a.bytes);
    ul.append(el('li', { class: 'chip', attrs: { 'data-state': a.state } }, icon(a.state === 'error' ? 'warn' : 'attach'), el('span', { class: 'chip-name', text: a.name }), el('span', { class: 'hint', text: state }), rm));
  }
}

async function addFiles(files) {
  for (const f of files.slice(0, 20)) {
    const a = { name: f.name, bytes: f.size, state: 'uploading', path: null };
    attachments.push(a);
    renderAttachments();
    updateSend();
    if (f.size > MAX_UPLOAD_BYTES) { Object.assign(a, { state: 'error', error: 'over 1.5 MB, not attached' }); renderAttachments(); updateSend(); continue; }
    try {
      const id = await ensureConversation();
      const r = await client.uploadFile(id, f);
      Object.assign(a, { state: 'ready', path: r.path, bytes: r.bytes ?? f.size });
    } catch (e) {
      Object.assign(a, { state: 'error', error: e.status === 404 || e.status === 405 ? 'this backend cannot receive files yet' : `not attached: ${e.message || e}` });
    }
    renderAttachments();
    updateSend();
  }
  $('#attach-status').textContent = `${attachments.filter((x) => x.state === 'ready').length} file(s) attached.`;
}

async function send() {
  const t = $('#jev-input');
  const message = t.value.trim();
  if (!message || busy()) return;
  const ready = attachments.filter((a) => a.state === 'ready');
  $('#jev-starters').hidden = true;
  const log = $('#jev-log');
  log.append(userMessage(message, ready));
  t.value = '';
  autosize();
  turn = new TurnView(log, $('#jev-turn-status'), { client, gallery, approvals }, { model });
  attachments = [];
  renderAttachments();
  updateSend();
  let r;
  try {
    r = await client.chat({ message, conversationId: convId, model, attachments: ready.map((a) => a.path) });
  } catch (e) {
    turn.finish('error', String(e.message || e));
    updateSend();
    return;
  }
  const isNew = convId !== r.conversation_id;
  convId = r.conversation_id;
  if (isNew) loadConversations();
  const myTurn = turn;
  follower = follow((since) => client.turnEvents(r.turn_id, since), (ev) => {
    myTurn.handle(ev);
    if (['done', 'error', 'cancelled'].includes(ev.event)) { updateSend(); if (isNew) loadConversations(); }
  }, { fast: 200, slow: 900, isSlow: () => approvals.pending.size > 0 });
  myTurn.turnId = r.turn_id;
  follower.done.then(() => updateSend());
  updateSend();
}

async function stop() {
  if (!turn || turn.ended || !turn.turnId) return;
  $('#jev-stop').disabled = true;
  try { await client.cancelTurn(turn.turnId); } catch (e) { app.notify('bad', 'Could not stop JEV', String(e.message || e)); }
  $('#jev-stop').disabled = false;
}
