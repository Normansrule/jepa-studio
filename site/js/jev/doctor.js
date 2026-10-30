// Status strip (Ollama, model, sandbox network isolation, GPU) and the Doctor view: each check
// with what it means and, when something is missing, the commands that fix it for the viewer's
// operating system. Everything is text; commands have a Copy button.
import { el, clear, $ } from '../ui/dom.js';
import { icon, statusItem } from './icons.js';
import { getPref, setPref } from '../ui/prefs.js';

export const FALLBACK_MODEL = 'qwen2.5:7b';

export function guessOs() {
  const ua = String((globalThis.navigator && (navigator.userAgentData?.platform || navigator.platform || navigator.userAgent)) || '');
  if (/win/i.test(ua)) return 'windows';
  if (/mac|iphone|ipad/i.test(ua)) return 'macos';
  return 'linux';
}

const COMMANDS = {
  install: {
    macos: [['Install Ollama (Homebrew), or get the app from ollama.com/download', 'brew install ollama']],
    windows: [['Install Ollama (winget), or get the installer from ollama.com/download', 'winget install Ollama.Ollama']],
    linux: [['Install Ollama with its official script (read it first at ollama.com/install.sh)', 'curl -fsSL https://ollama.com/install.sh | sh']],
  },
  start: {
    macos: [['Start it (or open the Ollama app)', 'ollama serve']],
    windows: [['Start it from the Start menu ("Ollama"), or in a terminal', 'ollama serve']],
    linux: [['Start the service installed by the script', 'sudo systemctl start ollama'], ['Or run it in a terminal', 'ollama serve']],
  },
};
const pullCmd = (model) => [[`Download ${model}, a model that can call tools${model === FALLBACK_MODEL ? ' (about 4.7 GB)' : ''}`, `ollama pull ${model}`]];

function isolationOf(sb) {
  if (!sb) return { known: false };
  const n = sb.network_isolation;
  // jev/sandbox.py reports a string: "enforced", "not available", "disabled (network_allowed is on)"
  if (typeof n === 'string' && n) return /^enforced/i.test(n) ? { known: true, on: true, how: sb.method && sb.method !== 'none' ? sb.method : '' } : { known: true, on: false, how: /^not available$/i.test(n) ? '' : n };
  if (n === true || (n && typeof n === 'object' && (n.enforced === true || n.enabled === true))) return { known: true, on: true, how: typeof n === 'object' ? n.method || n.how || '' : '' };
  if (n === false || n === 'unavailable' || n === 'off' || (n && typeof n === 'object' && n.enforced === false)) return { known: true, on: false, how: typeof n === 'object' ? n.reason || n.method || '' : '' };
  return { known: false };
}

export function gpuText(g) {
  if (!g) return 'CPU only';
  if (typeof g === 'string') return g;
  if (Array.isArray(g)) return g.map(gpuText).join(', ') || 'CPU only';
  if (g.available === false) return 'CPU only';
  return [g.name || g.device || g.backend || 'GPU', g.memory_gb ? `${g.memory_gb} GB` : g.vram_gb ? `${g.vram_gb} GB` : ''].filter(Boolean).join(', ');
}

/** Fill the status strip. `s` is /api/jev/status or null (unreachable), err its error. */
export function renderStrip(s, err, model) {
  if (!s) {
    statusItem($('#st-ollama'), 'bad', err && err.status === 404 ? 'JEV is not in this backend' : 'JEV backend not responding');
    statusItem($('#st-model'), 'idle', 'Model: none');
    statusItem($('#st-net'), 'idle', 'Sandbox: unknown');
    statusItem($('#st-gpu'), 'idle', 'GPU: unknown');
    return;
  }
  const o = s.ollama || {};
  statusItem($('#st-ollama'), o.reachable ? 'good' : 'bad', o.reachable ? `Ollama running${o.version ? ` ${o.version}` : ''}` : 'Ollama not reachable');
  const nModels = (s.models || []).length;
  statusItem($('#st-model'), model ? 'good' : nModels ? 'warn' : 'bad', model ? `Model: ${model}` : nModels ? 'Model: pick one' : 'No model yet');
  const iso = isolationOf(s.sandbox);
  statusItem($('#st-net'), !iso.known ? 'idle' : iso.on ? 'good' : 'warn', !iso.known ? 'Sandbox network: unknown' : iso.on ? 'Sandbox offline (network blocked)' : 'Sandbox can reach the network');
  statusItem($('#st-gpu'), 'idle', `GPU: ${gpuText(s.gpu)}`);
  $('#st-gpu').dataset.status = s.gpu && gpuText(s.gpu) !== 'CPU only' ? 'good' : 'idle';
}

function cmdRow([label, cmd]) {
  const code = el('code', { text: cmd });
  const copy = el('button', { class: 'btn copy-btn', attrs: { type: 'button', 'aria-label': `Copy: ${cmd}` } }, icon('copy'), el('span', { text: 'Copy' }));
  copy.addEventListener('click', async () => {
    try { await navigator.clipboard.writeText(cmd); copy.lastChild.textContent = 'Copied'; }
    catch {
      const r = document.createRange(); r.selectNodeContents(code);
      const sel = getSelection(); sel.removeAllRanges(); sel.addRange(r);
      copy.lastChild.textContent = 'Selected: press Ctrl+C';
    }
    setTimeout(() => { copy.lastChild.textContent = 'Copy'; }, 2500);
  });
  return el('li', { class: 'cmd' }, el('span', { class: 'cmd-label', text: label }), el('div', { class: 'cmd-line' }, code, copy));
}

function check(status, title, text, cmds = []) {
  const li = el('li', { class: 'doc-check', attrs: { 'data-status': status } });
  const head = el('p', { class: 'doc-head' });
  statusItem(head, status, title);
  li.append(head);
  if (text) li.append(el('p', { class: 'doc-text', text }));
  if (cmds.length) li.append(el('ul', { class: 'cmds' }, cmds.map(cmdRow)));
  return li;
}

/** Render the Doctor panel body. shell: optional result of the shell's backend_status. */
export function renderDoctor(s, err, shell) {
  const os = getPref('jev-os') || guessOs();
  for (const r of document.querySelectorAll('input[name="doc-os"]')) r.checked = r.value === os;
  const list = clear($('#doctor-checks'));
  const model = (s && s.default_model) || FALLBACK_MODEL;
  if (!s) {
    list.append(check('bad', 'The JEV backend did not answer', err && err.status === 404
      ? 'This backend has no /api/jev routes. Update JEV Studio, or start the backend from a checkout that includes the jev package.'
      : `The local backend did not answer (${err ? err.message : 'no response'}). Restart JEV Studio; if it keeps happening, look at logs/backend.log in the app's data folder.`));
  } else {
    list.append(check('good', `Local backend running${s.version ? ` (JEV ${s.version})` : ''}`, 'JEV and the JEPA tools share one backend on 127.0.0.1, protected by a per-session token.'));
  }
  const o = (s && s.ollama) || {};
  const shellSays = shell && shell.ollama ? shell.ollama : null;
  if (o.reachable) {
    list.append(check('good', `Ollama is running${o.version ? ` (${o.version})` : ''}`, `JEV talks to it at ${o.url || 'http://127.0.0.1:11434'}.`));
  } else {
    const extra = shellSays ? (shellSays.reachable ? ' The app shell can reach it, so the backend may need a restart.' : ' The app shell cannot reach it either.') : '';
    list.append(check('bad', 'Ollama is not running', `JEV gets its language models from Ollama, a free program that runs models on your computer. Install it, start it, then press Check again.${extra}${o.error ? ` (${o.error})` : ''}`,
      [...COMMANDS.install[os], ...COMMANDS.start[os]]));
  }
  const n = s && Array.isArray(s.models) ? s.models.length : 0;
  if (o.reachable && !n) list.append(check('warn', 'No model downloaded yet', 'Download one with the command below, or with "Download a model" next to the model picker (it asks you first).', pullCmd(model)));
  else if (n) list.append(check('good', `${n} model${n === 1 ? '' : 's'} available`, s.default_model ? `Default: ${s.default_model}.` : 'Pick one in the model menu.'));
  const iso = isolationOf(s && s.sandbox);
  if (iso.known && iso.on) list.append(check('good', 'Sandboxed code has no network', `Code JEV runs on its own cannot reach the internet${iso.how ? ` (${iso.how})` : ''}; anything that needs the network asks you first.`));
  else if (iso.known) list.append(check('warn', 'Sandboxed code could reach the network', `This computer does not support blocking the network for JEV's sandbox${iso.how ? ` (${iso.how})` : ''}. Code still runs only inside the conversation folder, with time and memory limits, and network tools still ask first.`));
  else list.append(check('idle', 'Sandbox network isolation unknown', 'The backend did not report whether it can block the network for sandboxed code.'));
  list.append(check(s && s.gpu && gpuText(s.gpu) !== 'CPU only' ? 'good' : 'idle', `GPU: ${gpuText(s && s.gpu)}`, 'Ollama uses the GPU by itself when it can. Without one, choose a small model (3B to 8B parameters) for reasonable speed.'));
}

export function bindDoctorOs(rerender) {
  for (const r of document.querySelectorAll('input[name="doc-os"]')) {
    r.addEventListener('change', () => { if (r.checked) { setPref('jev-os', r.value); rerender(); } });
  }
}
