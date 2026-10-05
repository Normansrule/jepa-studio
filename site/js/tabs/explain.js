// Explain tab: optional, OFF by default. Web tier = retrieval only: BM25 over
// site/docs-index.json (built by scripts/build_docs_index.py) plus this run's log lines.
// Answers are extractive passages with their heading and a link, never generated text.
// Everything shown (doc text, log lines, file names) is inserted with textContent, so it can
// never become markup, and nothing in an answer is wired to an action.
import { $, el, clear } from '../ui/dom.js';

let app, index = null, enabled = false;

const STOP = new Set('a an and are as at be by can do does for from how i if in into is it its of on or so that the their then there these this to was what when where which why will with you your'.split(' '));

export function tokenize(text) {
  return String(text).toLowerCase().normalize('NFKD').replace(/[^\p{L}\p{N}]+/gu, ' ').split(' ').filter((t) => t.length > 1 && !STOP.has(t));
}

/** BM25 (k1 = 1.2, b = 0.75) over [{text, ...}] documents. */
export class BM25 {
  constructor(docs, k1 = 1.2, b = 0.75) {
    this.docs = docs; this.k1 = k1; this.b = b;
    this.tf = docs.map((d) => { const m = new Map(); for (const t of tokenize(`${d.heading || ''} ${d.text}`)) m.set(t, (m.get(t) || 0) + 1); return m; });
    this.len = this.tf.map((m) => [...m.values()].reduce((s, x) => s + x, 0));
    this.avg = this.len.reduce((s, x) => s + x, 0) / Math.max(1, docs.length);
    this.df = new Map();
    for (const m of this.tf) for (const t of m.keys()) this.df.set(t, (this.df.get(t) || 0) + 1);
  }
  search(q, k = 5) {
    const terms = [...new Set(tokenize(q))];
    const N = this.docs.length;
    const scores = this.docs.map((_, i) => {
      let s = 0;
      for (const t of terms) {
        const f = this.tf[i].get(t);
        if (!f) continue;
        const idf = Math.log(1 + (N - this.df.get(t) + 0.5) / (this.df.get(t) + 0.5));
        s += idf * (f * (this.k1 + 1)) / (f + this.k1 * (1 - this.b + (this.b * this.len[i]) / this.avg));
      }
      return s;
    });
    return scores.map((s, i) => ({ i, s })).filter((x) => x.s > 0).sort((a, b) => b.s - a.s).slice(0, k).map((x) => ({ ...this.docs[x.i], score: x.s }));
  }
}

/** Short excerpt around the best-matching sentences. */
function excerpt(text, q, max = 420) {
  const terms = new Set(tokenize(q));
  const sents = text.split(/(?<=[.!?])\s+|\n+/);
  let best = 0, bs = -1;
  sents.forEach((s, i) => { const sc = tokenize(s).filter((t) => terms.has(t)).length; if (sc > bs) { bs = sc; best = i; } });
  let out = '';
  for (let i = best; i < sents.length && out.length < max; i++) out += (out ? ' ' : '') + sents[i];
  return out.length > max ? `${out.slice(0, max)}…` : out;
}

/** Append text to `node`, wrapping query terms in <mark> (still text nodes only). */
function highlighted(node, text, q) {
  const terms = new Set(tokenize(q));
  for (const part of text.split(/(\s+)/)) {
    const t = tokenize(part)[0];
    if (t && terms.has(t)) node.append(el('mark', { text: part })); else node.append(document.createTextNode(part));
  }
}

function runLog() {
  const evs = (app.run && app.run.events) || [];
  return evs.slice(-400).map((e) => {
    if (e.event === 'step') return `step ${e.step} loss ${e.loss?.toFixed(4)} pred ${e.pred?.toFixed(4)} sigreg ${e.sigreg?.toFixed(4)} lr ${e.lr?.toExponential(2)} ${Math.round(e.samples_per_s || 0)} samples/s${e.collapse ? ` collapse-check ${e.collapse.status}: ${e.collapse.message}` : ''}`;
    return `${e.event}${e.message ? `: ${e.message}` : ''}${e.status ? ` (${e.status})` : ''}`;
  });
}

async function ask() {
  const q = $('#assistant-q').value.trim().slice(0, 500);
  const box = clear($('#assistant-answers'));
  if (!q) return;
  if (app.backend.tier === 'desktop') {
    try {
      const r = await app.backend.assistant(q);
      const card = el('article', { class: 'answer' }, el('header', {}, el('h4', { text: 'Answer from the desktop assistant' })), el('p', { text: String(r.answer || '') }));
      box.append(card);
      for (const s of r.sources || []) box.append(el('article', { class: 'answer' }, el('header', {}, el('h4', { text: String(s.heading || '') }), el('code', { text: String(s.file || '') }))));
    } catch (e) { box.append(el('p', { class: 'notice bad', text: `Assistant error: ${e.message}` })); }
    return;
  }
  if (!index) {
    try {
      const r = await fetch(new URL('../../docs-index.json', import.meta.url));
      const j = await r.json();
      index = { docs: j.passages, bm: new BM25(j.passages) };
    } catch (e) { box.append(el('p', { class: 'notice bad', text: `Could not load docs-index.json: ${e.message}` })); return; }
  }
  const logLines = runLog();
  const logDocs = logLines.map((text, i) => ({ text, heading: `Run log line ${i + 1}`, file: 'run log', log: true }));
  const docHits = index.bm.search(q, 4);
  const logHits = logDocs.length ? new BM25(logDocs).search(q, 2) : [];
  if (!docHits.length && !logHits.length) { box.append(el('p', { class: 'notice', text: 'No passage matches those words. Try terms like "collapse", "SIGReg", "linear probe" or "batch size".' })); return; }
  for (const h of [...docHits, ...logHits]) {
    const head = el('header', {}, el('h4', { text: h.heading || h.file }), el('span', {}, h.log ? 'from ' : 'from ', el('code', { text: h.file }), ` · score ${h.score.toFixed(2)}`));
    const p = el('p');
    highlighted(p, h.log ? h.text : excerpt(h.text, q), q);
    const card = el('article', { class: 'answer' }, head, p);
    if (!h.log && h.url) card.append(el('a', { text: 'Read the full section', attrs: { href: h.url, rel: 'noopener', target: '_blank' } }));
    box.append(card);
  }
}

function toggle(on) {
  enabled = on;
  $('#assistant-on').hidden = !on;
  $('#assistant-off').hidden = on;
  if (on) {
    $('#assistant-mode').textContent = app.backend.tier === 'desktop' ? 'Desktop tier: questions go to the local backend (POST /api/assistant).' : 'Web tier: keyword search (BM25) over the docs and your run log. Nothing leaves the page.';
    $('#assistant-log').textContent = runLog().slice(-60).join('\n') || 'No run yet.';
  }
}

export function init(a) {
  app = a;
  $('#assistant-toggle').checked = false;
  $('#assistant-toggle').addEventListener('change', (e) => toggle(e.target.checked));
  $('#assistant-ask').addEventListener('click', ask);
  $('#assistant-q').addEventListener('keydown', (e) => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) ask(); });
  app.on('step', () => { if (enabled && app.run.events.length % 20 === 0) $('#assistant-log').textContent = runLog().slice(-60).join('\n'); });
  toggle(false);
}

export function show() { if (enabled) $('#assistant-log').textContent = runLog().slice(-60).join('\n') || 'No run yet.'; }
