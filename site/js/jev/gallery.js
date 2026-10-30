// Artifact gallery: images (with a lightbox), videos and audio (<video>/<audio> controls),
// code and text (highlighted, line numbers), everything else as a download. Bytes come from
// the backend with the token header and are shown through blob: URLs (CSP img-src / media-src
// allow blob:); file names and titles are text only.
import { el, clear } from '../ui/dom.js';
import { icon } from './icons.js';
import { codeBlock, langOf } from './highlight.js';
import { fmtBytes } from './client.js';

const KIND_WORD = { image: 'Image', video: 'Video', audio: 'Audio', code: 'Code', text: 'Text', data: 'Data', model: 'Model', other: 'File' };

export function kindOf(a) {
  const k = String(a.kind || '');
  if (KIND_WORD[k] && k !== 'other') return k;
  const m = String(a.mime || '');
  if (m.startsWith('image/')) return 'image';
  if (m.startsWith('video/')) return 'video';
  if (m.startsWith('audio/')) return 'audio';
  if (/json|csv|text\//.test(m)) return 'text';
  return 'other';
}

export function fileName(a) {
  const p = String(a.path || a.title || 'file');
  return p.split(/[\\/]/).pop() || 'file';
}

let lightbox = null;
function setupLightbox() {
  if (lightbox) return lightbox;
  const dlg = document.getElementById('jev-lightbox');
  if (!dlg) return null;
  const img = dlg.querySelector('img');
  const cap = dlg.querySelector('.lb-title');
  const dl = dlg.querySelector('.lb-download');
  dlg.querySelector('.lb-close').addEventListener('click', () => dlg.close());
  dlg.addEventListener('click', (e) => { if (e.target === dlg) dlg.close(); });   // backdrop
  lightbox = {
    open(src, title, name, opener) {
      img.src = src;
      img.alt = title;
      cap.textContent = title;
      dl.href = src;
      dl.setAttribute('download', name);
      dlg.addEventListener('close', () => { if (opener && opener.isConnected) opener.focus(); }, { once: true });
      dlg.showModal();
    },
  };
  return lightbox;
}

export class Gallery {
  /** container: element that receives figures; client: JevClient; count: optional element for "N files" */
  constructor(container, client, { empty = 'Files JEV makes in this conversation appear here.', count = null, onAdd = null } = {}) {
    this.box = container;
    this.client = client;
    this.empty = empty;
    this.count = count;
    this.onAdd = onAdd;
    this.items = new Map();
    this.renderEmpty();
  }

  renderEmpty() {
    clear(this.box);
    this.items.clear();
    this.box.append(el('p', { class: 'hint gallery-empty', text: this.empty }));
    this.updateCount();
  }

  updateCount() {
    if (this.count) this.count.textContent = this.items.size ? `${this.items.size} file${this.items.size === 1 ? '' : 's'}` : '';
  }

  setAll(list) { this.renderEmpty(); for (const a of list || []) this.add(a); }

  add(a) {
    if (!a || !a.url) return null;
    const key = a.id || a.url;
    if (this.items.has(key)) return this.items.get(key);
    const empty = this.box.querySelector('.gallery-empty');
    if (empty) empty.remove();
    const fig = artifactFigure(a, this.client);
    this.items.set(key, fig);
    this.box.prepend(fig);
    this.updateCount();
    if (this.onAdd) this.onAdd(a, fig);
    return fig;
  }
}

/** One artifact as a <figure>. Also used inline in tool-result cards (compact). */
export function artifactFigure(a, client, { compact = false } = {}) {
  const kind = kindOf(a);
  const title = String(a.title || fileName(a));
  const name = fileName(a);
  const fig = el('figure', { class: `art${compact ? ' compact' : ''}`, attrs: { 'data-kind': kind, 'data-artifact': a.id || '' } });
  const media = el('div', { class: 'art-media' });
  const size = Number(a.bytes || (a.meta && a.meta.bytes));
  const dl = el('a', { class: 'btn art-dl', attrs: { href: '#', download: name, 'aria-label': `Download ${name}` } }, icon('download'), el('span', { text: 'Download' }));
  const status = el('span', { class: 'hint art-status', attrs: { role: 'status' } });
  fig.append(media, el('figcaption', {},
    el('span', { class: 'art-title', text: title }),
    el('span', { class: 'hint', text: [KIND_WORD[kind], fmtBytes(size)].filter(Boolean).join(', ') }),
    status, dl));

  let src = null;
  const load = async () => {
    if (src) return src;
    src = await client.blobUrl(a.url);
    dl.href = src;
    return src;
  };
  const fail = (e) => { status.textContent = `Could not load: ${e.message || e}`; fig.dataset.state = 'error'; };

  dl.addEventListener('click', async (e) => {
    if (src) return;                         // a real blob: href with download attr, let it go
    e.preventDefault();
    try { await load(); dl.click(); } catch (err) { fail(err); }
  });

  if (kind === 'image') {
    const img = el('img', { attrs: { alt: title, decoding: 'async' } });
    const open = el('button', { class: 'art-open', attrs: { type: 'button', 'aria-label': `Open ${title} larger` } }, img);
    media.append(open);
    load().then((u) => { img.src = u; }).catch(fail);
    open.addEventListener('click', async () => {
      try { const u = await load(); const lb = setupLightbox(); if (lb) lb.open(u, title, name, open); } catch (err) { fail(err); }
    });
  } else if (kind === 'video' || kind === 'audio') {
    const m = el(kind, { attrs: { controls: true, preload: 'metadata', 'aria-label': title } });
    if (kind === 'video') m.setAttribute('playsinline', '');
    media.append(m);
    load().then((u) => { m.src = u; }).catch(fail);
  } else if (kind === 'code' || kind === 'text' || kind === 'data') {
    const lang = langOf(name, a.mime);
    const holder = el('div', { class: 'art-code' }, el('p', { class: 'hint', text: 'Loading…' }));
    media.append(holder);
    if (/^(text\/|application\/(json|x-python|javascript|x-sh))/.test(a.mime || 'text/plain') || lang !== 'text' || kind === 'code') {
      client.text(a.url, 200_000).then((t) => {
        clear(holder).append(codeBlock(t, lang, { label: `${name}, ${lang === 'text' ? 'text' : `${lang} code`}` }));
      }).catch((e) => { clear(holder); fail(e); });
    } else {
      clear(holder).append(el('p', { class: 'hint', text: 'Preview not available for this file type.' }));
    }
  } else {
    media.append(el('div', { class: 'art-file' }, icon('file'), el('span', { text: name })));
  }
  return fig;
}
