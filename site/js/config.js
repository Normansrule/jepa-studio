// Shared run-config format. site/schema/config.schema.json is a verbatim copy of
// schema/config.schema.json (it must stay byte-identical; tests/test_web_engine.py checks it),
// and validate() is a line-by-line port of jepa_studio/config.py:validate, so the web and
// desktop apps accept and reject exactly the same files.

export const DEFAULT_CONFIG = {
  version: 1,
  name: 'shapes-quickstart',
  seed: 0,
  data: {
    kind: 'synthetic-shapes', path: null, max_items: 4096,
    image_size: 32, local_size: 16, channels: 3,
    n_global: 2, n_local: 4,
    global_scale: [0.3, 1.0], local_scale: [0.05, 0.3],
    flip_p: 0.5, color_jitter: 0.4, grayscale_p: 0.2, blur_p: 0.2,
    mask_ratio: 0.0, labeled_fraction: 0.1,
    clip_frames: 8, series_window: 128,
  },
  model: { arch: 'convnet-tiny', embed_dim: 128, proj_dim: 16, proj_hidden: 256, patch_size: 4, depth: 4, heads: 4 },
  objective: { lambda: 0.05, num_slices: 256, t_max: 3.0, knots: 17 },
  train: {
    epochs: 10, max_steps: null, batch_size: 'auto', effective_batch: 256,
    lr: 5e-4, weight_decay: 5e-2, warmup_frac: 0.1, final_lr_ratio: 1e-3,
    checkpoint_every: 200, log_every: 10,
  },
  hardware: {
    mode: 'balanced', device: 'auto', precision: 'auto', channels_last: 'auto',
    compile: false, workers: 'auto', pin_memory: 'auto',
    max_memory_frac: 0.85, max_temp_c: 85, max_disk_gb: 5,
  },
  eval: { knn_k: 20, probe_epochs: 100, baselines: ['random-init', 'supervised'] },
  world: { env: 'two-room', episodes: 400, episode_len: 24, history: 1, horizon: 5, cem_samples: 300, cem_elites: 30, cem_iters: 10 },
};

/** The demo-tier settings the browser actually trains with (still a valid shared config). */
export function webDefaultConfig() {
  const c = deepMerge(DEFAULT_CONFIG, {
    name: 'shapes-web-demo',
    data: { max_items: 2048, n_local: 4, local_size: 16, blur_p: 0.0 },
    model: { arch: 'mlp-tiny', embed_dim: 64, proj_dim: 16, proj_hidden: 64 },
    objective: { num_slices: 64 },
    train: { epochs: 3, max_steps: 300, batch_size: 'auto', lr: 2e-3, log_every: 1 },
    hardware: { device: 'auto' },
  });
  return c;
}

export function deepClone(v) { return v === undefined ? v : JSON.parse(JSON.stringify(v)); }

export function deepMerge(base, override) {
  const out = deepClone(base);
  for (const [k, v] of Object.entries(override)) {
    if (isPlainObject(v) && isPlainObject(out[k])) out[k] = deepMerge(out[k], v);
    else out[k] = deepClone(v);
  }
  return out;
}

function isPlainObject(v) { return v !== null && typeof v === 'object' && !Array.isArray(v); }

// JSON has one number type; Python's json gives int for "3" and float for "3.0". JS loses that
// distinction after JSON.parse, so parseConfigText() records which numbers were written with a
// fraction/exponent and _typeOk consults that set (mirrors isinstance(value, int)).
const FLOAT_LITERALS = new WeakMap();

function typeOk(value, t, holder, key) {
  switch (t) {
    case 'integer': return typeof value === 'number' && Number.isInteger(value) && !isFloatLiteral(holder, key);
    case 'number': return typeof value === 'number' && Number.isFinite(value);
    case 'string': return typeof value === 'string';
    case 'boolean': return typeof value === 'boolean';
    case 'null': return value === null;
    case 'array': return Array.isArray(value);
    case 'object': return isPlainObject(value);
    default: throw new Error(`schema uses unsupported type ${t}`);
  }
}

function isFloatLiteral(holder, key) {
  if (!holder) return false;
  const s = FLOAT_LITERALS.get(holder);
  return !!(s && s.has(key));
}

function pyRepr(v) {
  if (typeof v === 'string') return `'${v}'`;
  if (v === null) return 'None';
  if (v === true) return 'True';
  if (v === false) return 'False';
  return JSON.stringify(v);
}

function pyTypeName(v) {
  if (v === null) return 'NoneType';
  if (Array.isArray(v)) return 'list';
  if (typeof v === 'object') return 'dict';
  if (typeof v === 'boolean') return 'bool';
  if (typeof v === 'number') return Number.isInteger(v) ? 'int' : 'float';
  return 'str';
}

function deepEqual(a, b) { return JSON.stringify(a) === JSON.stringify(b); }

/** Return a list of human-readable problems (empty = valid). Port of config.validate. */
export function validate(value, schema, path = '$', holder = null, key = null) {
  const errs = [];
  if ('oneOf' in schema) {
    const matches = schema.oneOf.filter((s) => validate(value, s, path, holder, key).length === 0);
    if (matches.length !== 1) errs.push(`${path}: must match exactly one of ${JSON.stringify(schema.oneOf)}`);
    return errs;
  }
  if ('const' in schema && !deepEqual(value, schema.const)) return [`${path}: must be ${pyRepr(schema.const)}`];
  if ('type' in schema) {
    const types = Array.isArray(schema.type) ? schema.type : [schema.type];
    if (!types.some((t) => typeOk(value, t, holder, key))) return [`${path}: expected ${types.join('/')}, got ${pyTypeName(value)}`];
  }
  if ('enum' in schema && !schema.enum.some((e) => deepEqual(e, value))) errs.push(`${path}: ${pyRepr(value)} not in ${JSON.stringify(schema.enum)}`);
  if (typeof value === 'number') {
    if ('minimum' in schema && value < schema.minimum) errs.push(`${path}: ${value} < minimum ${schema.minimum}`);
    if ('maximum' in schema && value > schema.maximum) errs.push(`${path}: ${value} > maximum ${schema.maximum}`);
  }
  if (typeof value === 'string' && 'maxLength' in schema && [...value].length > schema.maxLength) errs.push(`${path}: longer than ${schema.maxLength} characters`);
  if (typeof value === 'string' && 'pattern' in schema && !new RegExp(schema.pattern, 'u').test(value)) errs.push(`${path}: does not match ${schema.pattern}`);
  if (Array.isArray(value)) {
    if ('minItems' in schema && value.length < schema.minItems) errs.push(`${path}: needs at least ${schema.minItems} items`);
    if ('maxItems' in schema && value.length > schema.maxItems) errs.push(`${path}: at most ${schema.maxItems} items`);
    if ('items' in schema) value.forEach((v, i) => errs.push(...validate(v, schema.items, `${path}[${i}]`, value, i)));
  }
  if (isPlainObject(value)) {
    const props = schema.properties || {};
    for (const k of schema.required || []) if (!(k in value)) errs.push(`${path}: missing required key '${k}'`);
    for (const [k, v] of Object.entries(value)) {
      if (k in props) errs.push(...validate(v, props[k], `${path}.${k}`, value, k));
      else if (schema.additionalProperties === false) errs.push(`${path}: unknown key '${k}'`);
    }
  }
  return errs;
}

/**
 * JSON.parse that remembers which numbers were written as floats ("1.0", "1e3"), so that
 * "seed": 1.0 is rejected as non-integer exactly as Python would. Uses the reviver's
 * context.source (Node 21+/Chrome 114+); falls back to plain parsing elsewhere.
 */
export function parseConfigText(text) {
  if (text.length > 1_000_000) throw new Error('config file larger than 1 MB; refusing to parse');
  return JSON.parse(text, function reviver(k, v, ctx) {
    if (typeof v === 'number' && ctx && typeof ctx.source === 'string' && /[.eE]/.test(ctx.source)) {
      let s = FLOAT_LITERALS.get(this);
      if (!s) { s = new Set(); FLOAT_LITERALS.set(this, s); }
      s.add(Array.isArray(this) ? Number(k) : k);
    }
    return v;
  });
}

let schemaPromise = null;
/** Fetch the shared schema (same-origin). */
export function loadSchema(url = 'schema/config.schema.json') {
  if (!schemaPromise) schemaPromise = fetch(url).then((r) => { if (!r.ok) throw new Error(`schema ${r.status}`); return r.json(); });
  return schemaPromise;
}

/** Merge a user file onto the defaults, then validate (like config.load_config). */
export function loadConfigText(text, schema) {
  const raw = parseConfigText(text);
  if (!isPlainObject(raw)) return { config: null, errors: ['$: expected object'] };
  const merged = mergeKeepingMarks(DEFAULT_CONFIG, raw);
  const errors = validate(merged, schema);
  return { config: errors.length ? null : deepClone(merged), errors };
}

/** deepMerge that carries the float-literal marks of `override` objects onto the result. */
function mergeKeepingMarks(base, override) {
  const out = deepClone(base);
  const marks = FLOAT_LITERALS.get(override);
  for (const [k, v] of Object.entries(override)) {
    if (isPlainObject(v) && isPlainObject(out[k])) out[k] = mergeKeepingMarks(out[k], v);
    else out[k] = v;
  }
  if (marks) FLOAT_LITERALS.set(out, new Set(marks));
  return out;
}
