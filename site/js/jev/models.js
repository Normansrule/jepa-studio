// Models panel: what is installed in Ollama (size, quantization, family, digest), which one is the
// default, Delete (asks), Download by name (asks; examples sized by the GPU the backend reports),
// and Import a GGUF file (desktop shell: the native file dialog grants exactly that one file, then
// the import itself asks). Everything shown is text; model names come from Ollama and are never
// inserted as markup.
import { el, clear, $ } from '../ui/dom.js';
import { follow, fmtBytes } from './client.js';

export const MODEL_NAME = /^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$/;   // jev/api.py MODEL_NAME

// Examples only: names from the Ollama library, approximate download sizes at Ollama's default
// 4-bit quantization (docs/jev/README.md, "Which model fits your machine?").
const EXAMPLES = [
  { min: 0, label: 'Small: CPU only or under 6 GB of GPU memory', models: [
    ['qwen2.5-coder:3b', '1.9 GB', true], ['llama3.2:3b', '2.0 GB', true], ['phi4-mini', '2.5 GB', true], ['gemma3:4b', '3.3 GB', false]] },
  { min: 6, label: '6 to 11 GB of GPU memory', models: [
    ['qwen2.5-coder:7b', '4.7 GB', true], ['llama3.1:8b', '4.9 GB', true], ['qwen3:8b', '5.2 GB', true]] },
  { min: 12, label: '12 to 23 GB of GPU memory', models: [
    ['qwen2.5-coder:14b', '9.0 GB', true], ['qwen3:14b', '9.3 GB', true], ['gpt-oss:20b', '14 GB', true]] },
  { min: 24, label: '24 GB of GPU memory or more', models: [
    ['qwen2.5-coder:32b', '20 GB', true], ['qwen3:32b', '20 GB', true]] },
];

/** GPU memory in GB from /api/jev/status `gpu` ({devices: [{memory_gb}]}, or an older {memory_gb}). */
export function vramGb(g) {
  if (!g || typeof g !== 'object') return null;
  const devs = Array.isArray(g.devices) ? g.devices : [];
  const sizes = devs.map((d) => Number(d && d.memory_gb)).filter((n) => Number.isFinite(n) && n > 0);
  if (Number.isFinite(Number(g.memory_gb)) && Number(g.memory_gb) > 0) sizes.push(Number(g.memory_gb));
  return sizes.length ? Math.max(...sizes) : null;
}

/** The example tiers to show for this machine: the one that fits first, then the next smaller one. */
export function examplesFor(gpu) {
  const v = vramGb(gpu);
  const apple = !!(gpu && gpu.mps);
  const fit = v === null ? (apple ? 1 : 0) : EXAMPLES.reduce((best, t, i) => (v >= t.min ? i : best), 0);
  const tiers = [EXAMPLES[fit]];
  if (fit > 0) tiers.push(EXAMPLES[fit - 1]);
  const why = v !== null ? `your GPU has ${v} GB` : apple ? 'an Apple GPU shares the computer’s memory; 16 GB of RAM or more runs these' : 'no GPU was found, so models run on the CPU';
  return { tiers, why };
}

/** A safe default model name for an imported file: its stem, lowercased, other characters as '-'. */
export function nameFromFile(path) {
  const base = String(path || '').split(/[\\/]/).pop().replace(/\.gguf$/i, '');
  const n = base.toLowerCase().replace(/[^a-z0-9._-]+/g, '-').replace(/^[^a-z0-9]+/, '').slice(0, 80);
  return n || 'my-model';
}

export class ModelsPanel {
  /** opts.onDefault(name) after the default changed; opts.onChanged() after a pull/import/delete finished. */
  constructor(app, client, approvals, { onDefault = () => {}, onChanged = () => {} } = {}) {
    Object.assign(this, { app, client, approvals, onDefault, onChanged });
    this.list = [];
    this.imported = [];
    this.status = null;
    this.error = null;
    this.ggufPath = null;
    this.bind();
  }

  invoke() {
    const t = globalThis.__TAURI__;
    return t && t.core && typeof t.core.invoke === 'function' ? t.core.invoke.bind(t.core) : null;
  }

  bind() {
    $('#jev-pull-go').addEventListener('click', () => this.pull());
    $('#jev-pull-name').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); this.pull(); } });
    $('#jev-models-refresh').addEventListener('click', () => this.onChanged());
    $('#jev-gguf-pick').addEventListener('click', () => this.pickGguf());
    $('#jev-gguf-go').addEventListener('click', () => this.importGguf());
    $('#jev-gguf-name').addEventListener('keydown', (e) => { if (e.key === 'Enter') { e.preventDefault(); this.importGguf(); } });
    if (!this.invoke()) {
      $('#jev-gguf-pick').disabled = true;
      const why = $('#jev-gguf-pick-why');
      why.textContent = 'Choosing a file needs the desktop app’s file dialog. From a terminal: jev models import FILE.gguf --name NAME';
      why.hidden = false;
      $('#jev-gguf-pick').setAttribute('aria-describedby', why.id);
    }
  }

  /** New data from GET /api/jev/models (or an error) and GET /api/jev/status. */
  update({ models = [], imported = [], error = null } = {}, status = null) {
    this.list = models;
    this.imported = imported;
    this.error = error;
    this.status = status;
    this.render();
  }

  render() {
    const ul = clear($('#jev-model-list'));
    const st = $('#jev-models-note');
    const def = this.status && this.status.default_model;
    if (this.error) {
      st.textContent = `Could not list the models: ${this.error}`;
      st.hidden = false;
    } else if (!this.list.length) {
      st.textContent = 'No models are installed yet. Download one below, or import a GGUF file.';
      st.hidden = false;
    } else st.hidden = true;
    const sha = new Map(this.imported.map((m) => [m.name, m]));
    for (const m of this.list) {
      const name = String(m.name || '');
      const isDefault = name === def || `${name}:latest` === def || name === `${def}:latest`;
      const meta = [fmtBytes(Number(m.size_bytes ?? m.size)), m.parameter_size, m.quantization, m.family].filter(Boolean).join(' · ');
      const digest = String(m.digest || '').replace(/^sha256:/, '');
      const head = el('div', { class: 'model-head' }, el('strong', { class: 'model-name', text: name }));
      if (isDefault) head.append(el('span', { class: 'tag', text: 'Default' }));
      if (m.remote) head.append(el('span', { class: 'tag warn', text: 'Runs on ollama.com' }));
      const info = el('div', { class: 'model-info' }, head);
      if (meta) info.append(el('span', { class: 'hint', text: meta }));
      if (digest) info.append(el('span', { class: 'hint mono', attrs: { title: `Ollama digest sha256:${digest}` }, text: `digest ${digest.slice(0, 12)}` }));
      const imp = sha.get(name) || sha.get(name.replace(/:latest$/, ''));
      if (imp && imp.sha256) info.append(el('span', { class: 'hint mono model-sha', text: `imported file SHA-256 ${imp.sha256}` }));
      const actions = el('div', { class: 'btn-row model-actions' });
      const mk = el('button', { class: 'btn', attrs: { type: 'button', disabled: isDefault, 'aria-label': `Make ${name} the default model` }, text: isDefault ? 'Default' : 'Make default' });
      mk.addEventListener('click', () => this.makeDefault(name, mk));
      const del = el('button', { class: 'btn', attrs: { type: 'button', 'aria-label': `Delete ${name}` }, text: 'Delete…' });
      del.addEventListener('click', () => this.remove(name));
      actions.append(mk, del);
      ul.append(el('li', { class: 'model-row', attrs: { 'data-model': name } }, info, actions));
    }
    this.renderExamples();
  }

  renderExamples() {
    const box = clear($('#jev-pull-examples'));
    const installed = new Set(this.list.map((m) => String(m.name)));
    const { tiers, why } = examplesFor(this.status && this.status.gpu);
    box.append(el('p', { class: 'hint', text: `Examples that fit this computer (${why}). Sizes are approximate downloads; any name from ollama.com/library works.` }));
    for (const t of tiers) {
      const ul = el('ul', { class: 'example-list', attrs: { 'aria-label': t.label } });
      for (const [name, size, tools] of t.models) {
        const have = installed.has(name);
        const b = el('button', { class: 'example', attrs: { type: 'button', 'aria-label': `Use ${name}, about ${size}${tools ? '' : ', no native tool calling'}${have ? ', already installed' : ''}` } },
          el('span', { class: 'mono', text: name }),
          el('span', { class: 'hint', text: [size, tools ? '' : 'no tool calls', have ? 'installed' : ''].filter(Boolean).join(' · ') }));
        b.addEventListener('click', () => { const i = $('#jev-pull-name'); i.value = name; i.focus(); });
        ul.append(el('li', {}, b));
      }
      box.append(el('p', { class: 'sub-h', text: t.label }), ul);
    }
  }

  async makeDefault(name, btn) {
    btn.disabled = true;
    try {
      await this.client.settings({ default_model: name });
      if (this.status) this.status.default_model = name;
      this.render();
      this.app.notify('good', 'Default model changed', `New conversations use ${name}.`);
      this.onDefault(name);
    } catch (e) {
      btn.disabled = false;
      this.app.notify('bad', 'Could not change the default model', String(e.message || e));
    }
  }

  /** Start an approval-gated action; follow its job when approved (or right away when the rules say run). */
  async gated(request, { pending, line, bar, label, onDone }) {
    let r;
    try { r = await request(); } catch (e) { line.textContent = `Could not ask: ${e.message || e}`; return; }
    const go = (jobId) => this.followJob(jobId, { line, bar, label, onDone });
    if (r.job_id) { go(r.job_id); return; }
    if (!r.approval_id) { line.textContent = 'The backend did not return an approval.'; return; }
    this.approvals.whenDecided(r.approval_id, (decision, res) => {
      if (decision !== 'approve') { line.textContent = `${label}: you denied it.`; return; }
      if (res && res.ok === false) { line.textContent = `${label} could not start: ${res.error || res.status}`; return; }
      if (res && res.job_id) go(res.job_id);
      else { line.textContent = `${label}: approved.`; this.onChanged(); }
    });
    this.approvals.add({ approval_id: r.approval_id, ...pending, risk: 'approve' });
    line.textContent = `${label}: waiting for your approval.`;
  }

  followJob(jobId, { line, bar, label, onDone = () => {} }) {
    if (bar) { bar.hidden = false; bar.firstChild.style.width = '0%'; bar.setAttribute('aria-valuenow', '0'); }
    line.textContent = `${label}: started.`;
    follow((since) => this.client.jobEvents(jobId, since), (ev) => {
      if (ev.event === 'progress') {
        const f = ev.fraction;
        if (f !== null && f !== undefined && bar) {
          const pct = Math.round(Number(f) * 100);
          bar.firstChild.style.width = `${pct}%`;
          bar.setAttribute('aria-valuenow', String(pct));
          line.textContent = `${label}: ${pct}%${ev.message ? `, ${ev.message}` : ''}`;
        } else if (ev.message) line.textContent = `${label}: ${ev.message}`;
      } else if (ev.event === 'done') {
        if (bar) bar.hidden = true;
        line.textContent = ev.text || `${label}: done.`;
        onDone(ev);
        this.onChanged();
      } else if (ev.event === 'error' || ev.event === 'cancelled') {
        if (bar) bar.hidden = true;
        line.textContent = ev.event === 'cancelled' ? `${label}: cancelled.` : `${label} failed: ${ev.message || 'unknown error'}`;
        this.onChanged();
      }
    }, { fast: 1000 });
  }

  pull() {
    const name = $('#jev-pull-name').value.trim();
    const line = $('#jev-pull-status');
    if (!MODEL_NAME.test(name) || name.includes('..')) { line.textContent = 'Type a model name such as qwen2.5-coder:7b.'; $('#jev-pull-name').focus(); return; }
    line.textContent = `Asking to download ${name}…`;
    this.gated(() => this.client.pullModel(name), {
      pending: { name: 'model_pull', args: { name }, reason: 'Downloading a model uses the internet and several GB of disk space, so JEV always asks first.' },
      line, bar: $('#jev-pull-bar'), label: `Download of ${name}`,
    });
  }

  remove(name) {
    const line = $('#jev-models-status');
    this.gated(() => this.client.deleteModel(name), {
      pending: { name: 'model_delete', args: { name }, reason: 'Deleting a model removes its files from Ollama’s models folder.' },
      line, bar: null, label: `Delete ${name}`,
    });
  }

  async pickGguf() {
    const invoke = this.invoke();
    const line = $('#jev-gguf-status');
    if (!invoke) return;
    let path;
    try { path = await invoke('pick_gguf_file'); } catch (e) { line.textContent = `That file cannot be imported: ${e.message || e}`; return; }
    if (!path) { line.textContent = 'No file chosen.'; return; }
    this.ggufPath = String(path);
    $('#jev-gguf-path').textContent = this.ggufPath;
    $('#jev-gguf-name').value = nameFromFile(this.ggufPath);
    $('#jev-gguf-form').hidden = false;
    $('#jev-gguf-sha').hidden = true;
    line.textContent = 'File chosen. Pick a name, then request the import.';
    $('#jev-gguf-name').focus();
  }

  importGguf() {
    const name = $('#jev-gguf-name').value.trim();
    const line = $('#jev-gguf-status');
    const err = $('#jev-gguf-name-error');
    const ok = MODEL_NAME.test(name) && !name.includes('..');
    $('#jev-gguf-name').setAttribute('aria-invalid', String(!ok));
    err.hidden = ok;
    if (!ok) { err.textContent = 'Use letters, digits and . _ : / @ + - (start with a letter or digit), such as my-model or my-model:q4.'; return; }
    if (!this.ggufPath) { line.textContent = 'Choose a .gguf file first.'; return; }
    const path = this.ggufPath;
    this.gated(() => this.client.importModel(path, name), {
      pending: { name: 'model_import', args: { name, gguf_path: path }, reason: 'Importing copies the file into Ollama’s models folder and registers it under the new name.' },
      line, bar: null, label: `Import of ${name}`,
      onDone: (ev) => {
        if (ev.sha256) {
          $('#jev-gguf-sha-value').textContent = ev.sha256;
          $('#jev-gguf-sha').hidden = false;
        }
      },
    });
  }
}
