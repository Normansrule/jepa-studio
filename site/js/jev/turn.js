// One chat turn in the log: your message, then JEV's streamed answer with a card for every tool
// call (name, risk badge, collapsible arguments, progress, result, files).
//
// Screen readers: the log is role=log aria-live=polite. While a turn streams, the log is
// aria-busy and the streaming text is aria-hidden, so tokens are not read one by one; when the
// turn ends the final answer replaces the stream (a new node) and a one-sentence summary is
// appended, which is what gets announced. Progress between tools goes to a separate role=status
// line, one short message per tool, never per token.
import { el, clear } from '../ui/dom.js';
import { icon, riskBadge } from './icons.js';
import { codeBlock, renderAnswer } from './highlight.js';
import { artifactFigure } from './gallery.js';

const STATE_WORD = {
  running: ['clock', 'Running'], waiting: ['warn', 'Waiting for your approval'], done: ['check', 'Done'],
  failed: ['cross', 'Failed'], refused: ['cross', 'Not allowed'], denied: ['cross', 'You denied this'], cancelled: ['cross', 'Cancelled'],
};

export function userMessage(text, attachments = []) {
  const art = el('article', { class: 'msg msg-user' },
    el('h4', { class: 'who', text: 'You' }),
    el('div', { class: 'msg-body' }, el('p', { class: 'msg-plain', text })));
  if (attachments.length) {
    art.append(el('ul', { class: 'msg-attach', attrs: { 'aria-label': 'Attached files' } },
      attachments.map((a) => el('li', {}, icon('attach'), el('span', { text: a.name || a.path })))));
  }
  return art;
}

/** A finished tool card from history (no live updates). */
export function toolCard({ name, risk = 'read', args = null, state = 'done', result = '', artifacts = [], client = null }) {
  const card = new ToolCard({ call_id: '', name, risk, args, decision: 'run' }, client);
  card.setState(state);
  if (result) card.setResult(result, state !== 'failed');
  for (const a of artifacts) card.addArtifact(a);
  return card.node;
}

class ToolCard {
  constructor(ev, client) {
    this.client = client;
    this.name = String(ev.name || 'tool');
    this.stateEl = el('span', { class: 'tc-state' });
    const head = el('header', { class: 'tc-head' },
      el('code', { class: 'tc-name', text: this.name }), riskBadge(ev.risk), this.stateEl);
    this.node = el('section', { class: 'tool-card', attrs: { 'aria-label': `Tool ${this.name}`, 'data-call': ev.call_id || '' } }, head);
    const args = ev.args && typeof ev.args === 'object' ? { ...ev.args } : ev.args;
    if (args && (typeof args !== 'object' || Object.keys(args).length)) {
      const det = el('details', { class: 'tc-args' }, el('summary', {}, el('span', { text: 'Arguments' })));
      if (args && typeof args === 'object' && typeof args.code === 'string') {
        const lang = /py/i.test(args.language || 'python') ? 'python' : /js|javascript|node/i.test(args.language || '') ? 'js' : /sh|bash/i.test(args.language || '') ? 'shell' : 'python';
        det.append(codeBlock(args.code, lang, { label: `${this.name} code` }));
        delete args.code;
      }
      if (args && (typeof args !== 'object' || Object.keys(args).length)) det.append(codeBlock(JSON.stringify(args, null, 2), 'json', { label: `${this.name} arguments` }));
      this.node.append(det);
    }
    this.bar = el('div', { class: 'progress tc-progress', attrs: { role: 'progressbar', 'aria-label': `${this.name} progress`, 'aria-valuemin': '0', 'aria-valuemax': '100', hidden: true } }, el('div'));
    this.msg = el('p', { class: 'hint tc-msg', attrs: { hidden: true } });
    this.result = el('div', { class: 'tc-result', attrs: { hidden: true } });
    this.arts = el('div', { class: 'tc-arts', attrs: { hidden: true } });
    this.node.append(this.bar, this.msg, this.result, this.arts);
    this.setState(ev.decision === 'deny' ? 'refused' : ev.decision === 'ask' ? 'waiting' : 'running');
  }

  setState(s) {
    this.state = s;
    this.node.dataset.state = s;
    const [ic, word] = STATE_WORD[s] || STATE_WORD.running;
    clear(this.stateEl).append(icon(ic), el('span', { text: word }));
    if (s !== 'running') this.bar.hidden = this.bar.hidden || s !== 'done';
  }

  progress(fraction, message) {
    const pct = Math.round(Math.max(0, Math.min(1, Number(fraction) || 0)) * 100);
    this.bar.hidden = false;
    this.bar.firstChild.style.width = `${pct}%`;
    this.bar.setAttribute('aria-valuenow', String(pct));
    if (message) { this.msg.hidden = false; this.msg.textContent = String(message); }
  }

  setResult(content, ok) {
    this.result.hidden = !content;
    clear(this.result);
    if (content) {
      const text = String(content);
      const pre = el('pre', { class: `tc-out${ok ? '' : ' bad'}`, attrs: { tabindex: '0', 'aria-label': `${this.name} result` }, text: text.length > 6000 ? `${text.slice(0, 6000)}\n…` : text });
      this.result.append(pre);
    }
    // progress is for running tools only; the result replaces it
    this.bar.hidden = true;
    this.msg.hidden = true;
  }

  addArtifact(a) {
    if (!this.client || !a || !a.url) return;
    if (this.arts.querySelector(`[data-artifact="${CSS.escape(String(a.id || ''))}"]`) && a.id) return;
    this.arts.hidden = false;
    this.arts.append(artifactFigure(a, this.client, { compact: true }));
  }
}

export class TurnView {
  /** log: the role=log element; status: role=status line; deps: {client, gallery, approvals} */
  constructor(log, status, deps, { model = '' } = {}) {
    this.log = log;
    this.status = status;
    this.deps = deps;
    this.cards = new Map();
    this.segments = [];
    this.seg = null;
    this.ended = false;
    this.t0 = performance.now();
    this.whoModel = el('span', { class: 'hint who-model', text: model });
    this.body = el('div', { class: 'msg-body' });
    this.state = el('p', { class: 'msg-state' }, el('span', { class: 'spinner', attrs: { 'aria-hidden': 'true' } }), el('span', { text: 'Thinking…' }));
    this.node = el('article', { class: 'msg msg-jev', attrs: { 'aria-busy': 'true' } },
      el('h4', { class: 'who' }, el('span', { text: 'JEV' }), this.whoModel), this.body, this.state);
    log.append(this.node);
    log.setAttribute('aria-busy', 'true');
    this.say('JEV is thinking.');
    this.scroll();
  }

  say(text) { if (this.status) this.status.textContent = text; }
  setWorking(text) { clear(this.state).append(el('span', { class: 'spinner', attrs: { 'aria-hidden': 'true' } }), el('span', { text })); }

  /** Keep the newest content in view while the reader is at the bottom of the log; once they
   *  scroll up to read, leave the position alone until they come back down. */
  scroll() {
    const l = this.log;
    if (!l.dataset.pinWatch) {
      l.dataset.pinWatch = '1';
      l.dataset.pinned = 'true';
      l.addEventListener('scroll', () => { l.dataset.pinned = String(l.scrollHeight - l.scrollTop - l.clientHeight < 80); }, { passive: true });
    }
    if (!this.ro && typeof ResizeObserver === 'function') {
      this.ro = new ResizeObserver(() => { if (l.dataset.pinned !== 'false') l.scrollTop = l.scrollHeight; });
      this.ro.observe(this.node);
    }
    if (l.dataset.pinned !== 'false' || !this.scrolledOnce) { l.scrollTop = l.scrollHeight; this.scrolledOnce = true; l.dataset.pinned = 'true'; }
  }

  text(t) {
    if (!this.seg) {
      this.seg = { text: '', node: el('div', { class: 'seg-text stream', attrs: { 'aria-hidden': 'true' } }) };
      this.seg.textNode = document.createTextNode('');
      this.seg.node.append(this.seg.textNode);
      this.segments.push(this.seg);
      this.body.append(this.seg.node);
    }
    this.seg.text += t;
    this.seg.textNode.data = this.seg.text;
  }

  handle(ev) {
    if (this.ended && !['artifact'].includes(ev.event)) return;
    switch (ev.event) {
      case 'start':
        if (ev.model) this.whoModel.textContent = ev.model;
        break;
      case 'token':
        this.text(String(ev.text ?? ''));
        break;
      case 'tool_call': {
        this.seg = null;
        const card = new ToolCard(ev, this.deps.client);
        this.cards.set(ev.call_id, card);
        this.body.append(card.node);
        if (ev.decision === 'deny') this.say(`${ev.name} is not allowed by the safety policy.`);
        else if (ev.decision === 'ask') this.say(`${ev.name} needs your approval.`);
        else this.say(`Running ${ev.name}.`);
        this.setWorking(ev.decision === 'ask' ? `Waiting for your approval: ${ev.name}` : `Running ${ev.name}…`);
        break;
      }
      case 'approval_request': {
        const card = this.cards.get(ev.call_id);
        if (card) card.setState('waiting');
        this.setWorking(`Waiting for your approval: ${ev.name}`);
        this.deps.approvals.add({ ...ev, risk: card ? card.node.querySelector('.risk')?.dataset.risk : 'approve' });
        break;
      }
      case 'progress': {
        const card = this.cards.get(ev.call_id);
        if (card) { if (card.state === 'waiting') card.setState('running'); card.progress(ev.fraction, ev.message); }
        break;
      }
      case 'tool_result': {
        const card = this.cards.get(ev.call_id);
        if (card) {
          const denied = !ev.ok && /denied|not approved|refused/i.test(String(ev.content || ''));
          card.setState(ev.ok ? 'done' : denied ? 'denied' : card.state === 'refused' ? 'refused' : 'failed');
          card.setResult(ev.content, !!ev.ok);
          for (const a of ev.artifacts || []) { card.addArtifact(a); this.deps.gallery.add(a); }
        }
        this.setWorking('Thinking…');
        this.say(card ? `${card.name} ${ev.ok ? 'finished' : 'did not finish'}.` : 'A tool finished.');
        break;
      }
      case 'artifact':
        this.deps.gallery.add(ev.artifact || ev);
        break;
      case 'done':
        this.finish('done', ev.text);
        break;
      case 'error':
        this.finish('error', ev.message);
        break;
      case 'cancelled':
        this.finish('cancelled');
        break;
      default:
        break;
    }
    this.scroll();
  }

  finish(kind, text = '') {
    if (this.ended) return;
    this.ended = true;
    // streamed segments -> rendered answer (new nodes, readable)
    for (const s of this.segments) {
      const out = el('div', { class: 'seg-text' }, renderAnswer(s.text));
      s.node.replaceWith(out);
      s.node = out;
    }
    const streamed = this.segments.map((s) => s.text).join('').trim();
    if (kind === 'done' && text && !streamed) this.body.append(el('div', { class: 'seg-text' }, renderAnswer(text)));
    for (const c of this.cards.values()) if (c.state === 'running' || c.state === 'waiting') c.setState(kind === 'cancelled' ? 'cancelled' : kind === 'error' ? 'failed' : c.state);
    const secs = ((performance.now() - this.t0) / 1000).toFixed(0);
    clear(this.state);
    if (kind === 'done') {
      this.state.append(icon('check'), el('span', { text: `Done in ${secs} s` }));
      this.state.dataset.status = 'good';
    } else if (kind === 'cancelled') {
      this.state.append(icon('cross'), el('span', { text: 'Stopped' }));
      this.state.dataset.status = 'idle';
    } else {
      this.state.append(icon('warn'), el('span', { text: `Something went wrong: ${text || 'unknown error'}` }));
      this.state.dataset.status = 'bad';
    }
    this.node.removeAttribute('aria-busy');
    this.log.setAttribute('aria-busy', 'false');
    const answer = (streamed || text || '').replace(/\s+/g, ' ').trim();
    const nTools = this.cards.size;
    const summary = kind === 'done'
      ? `JEV answered${nTools ? ` after ${nTools} tool call${nTools === 1 ? '' : 's'}` : ''}: ${answer.slice(0, 280)}${answer.length > 280 ? '…' : ''}`
      : kind === 'cancelled' ? 'JEV stopped.' : `JEV hit an error: ${text}`;
    this.log.append(el('p', { class: 'visually-hidden sr-summary', text: summary }));
    this.say(kind === 'done' ? `Done in ${secs} seconds.` : kind === 'cancelled' ? 'Stopped.' : 'Error.');
    this.scroll();
    if (this.ro) setTimeout(() => { this.ro.disconnect(); this.ro = null; }, 4000);   // late image loads still pin
  }
}
