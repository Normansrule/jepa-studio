// Per-viewer preferences (last tab, last preset; the theme keeps its own 'jepa-theme' key, which
// the landing page shares). Everything is wrapped in try/catch: storage can be missing, full,
// blocked or throwing (private windows, sandboxed frames, tests), and the app must work anyway.
const KEY = 'jepa-prefs';

function readAll() {
  try {
    const o = JSON.parse(globalThis.localStorage.getItem(KEY) || '{}');
    return o && typeof o === 'object' && !Array.isArray(o) ? o : {};
  } catch { return {}; }
}

export function getPref(name, fallback = null) {
  const o = readAll();
  return Object.prototype.hasOwnProperty.call(o, name) ? o[name] : fallback;
}

export function setPref(name, value) {
  try {
    const o = readAll();
    o[name] = value;
    globalThis.localStorage.setItem(KEY, JSON.stringify(o));
    return true;
  } catch { return false; }
}

/** Raw single-key access for keys shared with other pages (the theme). */
export function getRaw(key) { try { return globalThis.localStorage.getItem(key); } catch { return null; } }
export function setRaw(key, value) { try { globalThis.localStorage.setItem(key, value); return true; } catch { return false; } }
