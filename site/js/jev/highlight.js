// A small in-house highlighter (no library): comments, strings, numbers, keywords and
// function names for Python, JavaScript/TypeScript, JSON and shell. It builds DOM nodes with
// textContent only, so code from the model or a file can never become markup. Each line is its
// own element, and line numbers are drawn by CSS counters (so copying the code copies no numbers).
import { el } from '../ui/dom.js';

const KW = {
  python: 'False None True and as assert async await break class continue def del elif else except finally for from global if import in is lambda nonlocal not or pass raise return try while with yield match case print self',
  js: 'async await break case catch class const continue default delete do else export extends false finally for from function if import in instanceof let new null of return static super switch this throw true try typeof undefined var void while yield',
  shell: 'if then else fi for do done while case esac in function export echo cd sudo set unset source return exit',
  json: 'true false null',
};

export function langOf(name = '', mime = '') {
  const n = String(name).toLowerCase();
  if (/\.py$/.test(n) || /python/.test(mime)) return 'python';
  if (/\.(m?js|cjs|ts|tsx|jsx)$/.test(n) || /javascript|typescript/.test(mime)) return 'js';
  if (/\.json$/.test(n) || /json/.test(mime)) return 'json';
  if (/\.(sh|bash|zsh|ps1)$/.test(n) || /shellscript|x-sh/.test(mime)) return 'shell';
  return 'text';
}

function tokenRe(lang) {
  const comment = lang === 'python' || lang === 'shell' ? '#[^\\n]*' : lang === 'js' ? '\\/\\/[^\\n]*|\\/\\*[\\s\\S]*?\\*\\/' : '(?!)';
  const str = lang === 'python'
    ? '"""[\\s\\S]*?"""|\'\'\'[\\s\\S]*?\'\'\'|"(?:\\\\.|[^"\\\\\\n])*"|\'(?:\\\\.|[^\'\\\\\\n])*\''
    : '`(?:\\\\.|[^`\\\\])*`|"(?:\\\\.|[^"\\\\\\n])*"|\'(?:\\\\.|[^\'\\\\\\n])*\'';
  return new RegExp(`(${comment})|(${str})|(\\b\\d[\\d_]*(?:\\.\\d+)?(?:e[+-]?\\d+)?\\b)|([A-Za-z_$][\\w$]*)(?=\\s*\\()|([A-Za-z_$][\\w$]*)`, 'g');
}

/** Split code into [{cls, text}] tokens; cls is '' for plain text. */
export function tokens(code, lang) {
  const out = [];
  if (lang === 'text') return [{ cls: '', text: code }];
  const kw = new Set((KW[lang] || '').split(' '));
  const re = tokenRe(lang);
  let last = 0, m;
  while ((m = re.exec(code))) {
    if (m.index > last) out.push({ cls: '', text: code.slice(last, m.index) });
    let cls = '';
    if (m[1]) cls = 'tk-com';
    else if (m[2]) cls = 'tk-str';
    else if (m[3]) cls = 'tk-num';
    else if (m[4]) cls = kw.has(m[4]) ? 'tk-kw' : 'tk-fn';
    else if (m[5]) cls = kw.has(m[5]) ? 'tk-kw' : '';
    out.push({ cls, text: m[0] });
    last = re.lastIndex;
    if (m[0] === '') re.lastIndex++;
  }
  if (last < code.length) out.push({ cls: '', text: code.slice(last) });
  return out;
}

/** <pre class="code" data-lang=…><code><span class="ln">…</span>…</code></pre> */
export function codeBlock(code, lang = 'text', { label = '' } = {}) {
  const codeEl = el('code');
  let line = el('span', { class: 'ln' });
  const flush = () => { codeEl.append(line); line = el('span', { class: 'ln' }); };
  for (const t of tokens(String(code).replace(/\r\n?/g, '\n').replace(/\n$/, ''), lang)) {
    const parts = t.text.split('\n');
    parts.forEach((p, i) => {
      if (i > 0) flush();
      if (p) line.append(t.cls ? el('span', { class: t.cls, text: p }) : document.createTextNode(p));
    });
  }
  flush();
  const pre = el('pre', { class: 'code', attrs: { 'data-lang': lang, tabindex: '0' } }, codeEl);
  if (label) pre.setAttribute('aria-label', label);
  return pre;
}

/** Minimal Markdown for answers: paragraphs, fenced code, bullet/numbered lists, `code`, **bold**.
 *  DOM only, no HTML parsing. */
export function renderAnswer(text) {
  const frag = document.createDocumentFragment();
  const src = String(text || '').replace(/\r\n?/g, '\n');
  const blocks = src.split(/(```[^\n]*\n[\s\S]*?(?:```|$))/g);
  for (const b of blocks) {
    if (!b) continue;
    const fence = /^```([^\n]*)\n([\s\S]*?)(?:```)?$/.exec(b);
    if (fence) {
      const lang = { py: 'python', python: 'python', js: 'js', javascript: 'js', ts: 'js', json: 'json', sh: 'shell', bash: 'shell', shell: 'shell' }[fence[1].trim().toLowerCase()] || 'text';
      frag.append(codeBlock(fence[2], lang, { label: `${fence[1].trim() || 'Code'} block` }));
      continue;
    }
    for (const para of b.split(/\n{2,}/)) {
      const p = para.trim();
      if (!p) continue;
      const lines = p.split('\n');
      if (lines.every((l) => /^\s*([-*]|\d+[.)])\s+/.test(l))) {
        const ordered = /^\s*\d/.test(lines[0]);
        const list = el(ordered ? 'ol' : 'ul');
        for (const l of lines) list.append(inline(el('li'), l.replace(/^\s*([-*]|\d+[.)])\s+/, '')));
        frag.append(list);
      } else if (/^#{1,4}\s/.test(p) && lines.length === 1) {
        frag.append(inline(el('p', { class: 'ans-h' }), p.replace(/^#{1,4}\s+/, '')));
      } else {
        frag.append(inline(el('p'), p));
      }
    }
  }
  return frag;
}

function inline(node, text) {
  for (const part of text.split(/(`[^`\n]+`|\*\*[^*\n]+\*\*)/g)) {
    if (!part) continue;
    if (part.startsWith('`') && part.endsWith('`') && part.length > 2) node.append(el('code', { text: part.slice(1, -1) }));
    else if (part.startsWith('**') && part.endsWith('**') && part.length > 4) node.append(el('strong', { text: part.slice(2, -2) }));
    else node.append(document.createTextNode(part));
  }
  return node;
}
