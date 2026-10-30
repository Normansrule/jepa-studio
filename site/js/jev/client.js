// JEV API client: a thin wrapper over the desktop backend's /api/jev/* routes (docs/jev/api.md).
// It reuses DesktopBackend.req / .blob, so every call carries the session token, goes only to
// 127.0.0.1:<port>, and the web tier (no DesktopBackend) never constructs one.
//
// One route is proposed by the UI and not yet in the contract (docs/jev/ui.md, "Proposed routes"):
//   POST /api/jev/conversations                 {title?, model?}          -> {id}
//   POST /api/jev/conversations/<id>/files      {name, mime, data_base64} -> {path, name, bytes}

const enc = encodeURIComponent;
export const MAX_UPLOAD_BYTES = 1.5 * 1024 * 1024;   // base64 of this stays under the 2 MB JSON cap

export class JevClient {
  constructor(backend) {
    this.b = backend;
    this.blobUrls = new Map();     // artifact url -> object URL (revoked on clearBlobs)
  }

  status() { return this.b.req('GET', '/api/jev/status'); }
  models() { return this.b.req('GET', '/api/jev/models'); }
  pullModel(name) { return this.b.req('POST', '/api/jev/models/pull', { name: String(name).slice(0, 200) }); }
  settings(patch) { return this.b.req('POST', '/api/jev/settings', patch); }

  chat({ message, conversationId = null, model = null, attachments = [] }) {
    const body = { message: String(message).slice(0, 20000) };
    if (conversationId) body.conversation_id = conversationId;
    if (model) body.model = model;
    if (attachments.length) body.attachments = attachments.slice(0, 20);
    return this.b.req('POST', '/api/jev/chat', body);
  }
  turnEvents(turnId, since = 0) { return this.b.req('GET', `/api/jev/turns/${enc(turnId)}/events?since=${Number(since) || 0}`); }
  cancelTurn(turnId) { return this.b.req('POST', `/api/jev/turns/${enc(turnId)}/cancel`); }

  conversations() { return this.b.req('GET', '/api/jev/conversations'); }
  conversation(id) { return this.b.req('GET', `/api/jev/conversations/${enc(id)}`); }
  exportConversation(id) { return this.b.blob(`/api/jev/conversations/${enc(id)}/export`); }
  deleteConversation(id) { return this.b.req('DELETE', `/api/jev/conversations/${enc(id)}`); }
  createConversation(model = null) { return this.b.req('POST', '/api/jev/conversations', model ? { model } : {}); }

  async uploadFile(conversationId, file) {
    if (file.size > MAX_UPLOAD_BYTES) throw new Error(`${file.name} is larger than 1.5 MB`);
    const data = await toBase64(file);
    return this.b.req('POST', `/api/jev/conversations/${enc(conversationId)}/files`, {
      name: String(file.name).slice(0, 200), mime: file.type || 'application/octet-stream', data_base64: data,
    });
  }

  approvals(status = null) { return this.b.req('GET', `/api/jev/approvals${status ? `?status=${enc(status)}` : ''}`); }
  decide(id, decision, remember = null) {
    return this.b.req('POST', `/api/jev/approvals/${enc(id)}`, { decision, remember: remember || null });
  }

  artifacts(conversationId) { return this.b.req('GET', `/api/jev/artifacts?conversation_id=${enc(conversationId)}`); }

  renderBlackHole(params) { return this.b.req('POST', '/api/jev/science/blackhole', params); }
  simulate(kind, params) { return this.b.req('POST', '/api/jev/science/simulate', { kind, params }); }
  jobEvents(jobId, since = 0) { return this.b.req('GET', `/api/jev/jobs/${enc(jobId)}/events?since=${Number(since) || 0}`); }

  /** Object URL for an artifact (fetched once with the token header, then cached). */
  async blobUrl(url) {
    if (this.blobUrls.has(url)) return this.blobUrls.get(url);
    const p = this.b.blob(url).then((blob) => URL.createObjectURL(blob));
    this.blobUrls.set(url, p);
    try { return await p; } catch (e) { this.blobUrls.delete(url); throw e; }
  }
  async text(url, max = 400_000) {
    const blob = await this.b.blob(url);
    return (await blob.slice(0, max).text());
  }
  clearBlobs() {
    for (const p of this.blobUrls.values()) Promise.resolve(p).then((u) => URL.revokeObjectURL(u)).catch(() => {});
    this.blobUrls.clear();
  }
}

function toBase64(file) {
  return new Promise((resolve, reject) => {
    const r = new FileReader();
    r.onload = () => resolve(String(r.result).replace(/^data:[^,]*,/, ''));
    r.onerror = () => reject(r.error || new Error('could not read the file'));
    r.readAsDataURL(file);
  });
}

/** Poll an event log ({events, next, status}) until a terminal event or `stop()`. */
export function follow(fetchPage, onEvent, { fast = 250, slow = 1000, isSlow = () => false } = {}) {
  let since = 0, stopped = false, timer = null;
  const TERMINAL = new Set(['done', 'error', 'cancelled', 'end', 'finished']);
  const done = new Promise((resolve) => {
    const tick = async () => {
      if (stopped) { resolve('stopped'); return; }
      let page;
      try {
        page = await fetchPage(since);
      } catch (e) {
        onEvent({ event: 'error', message: String(e.message || e), transport: true });
        resolve('error');
        return;
      }
      since = Number.isFinite(page.next) ? page.next : since + (page.events || []).length;
      let end = null;
      for (const ev of page.events || []) {
        onEvent(ev);
        if (TERMINAL.has(ev.event)) end = ev.event;
      }
      if (!end && page.status && TERMINAL.has(page.status)) end = page.status;
      if (end || stopped) { resolve(end || 'stopped'); return; }
      timer = setTimeout(tick, isSlow() ? slow : fast);
    };
    tick();
  });
  return { done, stop() { stopped = true; clearTimeout(timer); } };
}

export function fmtBytes(n) {
  if (!Number.isFinite(n) || n <= 0) return '';
  const u = ['B', 'KB', 'MB', 'GB', 'TB'];
  let i = 0;
  while (n >= 1000 && i < u.length - 1) { n /= 1000; i++; }
  return `${n >= 10 || i === 0 ? Math.round(n) : n.toFixed(1)} ${u[i]}`;
}
