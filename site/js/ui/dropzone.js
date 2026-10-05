// Turn a drop's DataTransfer into a flat list of File objects with relative paths.
// Folders are walked with DataTransferItem.webkitGetAsEntry (every current browser); when no
// entries are available (synthetic drops, some platforms) the plain file list is used.
// The walk is bounded (files and depth) so a huge folder cannot hang the page.

const IMAGE_EXT = /\.(png|jpe?g|gif|webp|bmp|avif)$/i;
export function isImageFile(f) { return /^image\//.test(f.type || '') || IMAGE_EXT.test(f.name || ''); }
export function isCsvFile(f) { return /\.(csv|txt)$/i.test(f.name || '') || f.type === 'text/csv'; }

function readAllEntries(reader) {
  // readEntries returns at most ~100 entries per call; call until it returns none
  return new Promise((resolve, reject) => {
    const out = [];
    const next = () => reader.readEntries((batch) => { if (!batch.length) resolve(out); else { out.push(...batch); next(); } }, reject);
    next();
  });
}

const fileOf = (entry) => new Promise((resolve, reject) => entry.file(resolve, reject));

/** {files: File[], paths: string[], truncated: boolean}. Paths of entries start with "drop/" so
 *  that dropping class folders (cats/, dogs/) side by side labels them like a folder of
 *  class folders picked with the folder button. */
export async function collectDropped(dt, { maxFiles = 5000, maxDepth = 8 } = {}) {
  const files = [], paths = [];
  let truncated = false;
  // everything on a DataTransfer must be read synchronously, during the drop event
  const items = [...(dt.items || [])].filter((it) => it.kind === 'file');
  const entries = items.map((it) => (typeof it.webkitGetAsEntry === 'function' ? it.webkitGetAsEntry() : null));
  const direct = items.map((it) => it.getAsFile());
  if (!entries.length || entries.some((e) => !e)) {
    for (const f of dt.files || []) { if (files.length >= maxFiles) { truncated = true; break; } files.push(f); paths.push(f.name); }
    return { files, paths, truncated };
  }
  async function walk(entry, depth) {
    if (files.length >= maxFiles) { truncated = true; return; }
    if (entry.isFile) {
      try { files.push(await fileOf(entry)); paths.push(`drop/${entry.fullPath.replace(/^\/+/, '')}`); } catch { /* unreadable: skip */ }
    } else if (entry.isDirectory && depth < maxDepth) {
      const children = await readAllEntries(entry.createReader());
      children.sort((a, b) => a.name.localeCompare(b.name));
      for (const c of children) await walk(c, depth + 1);
    }
  }
  for (const [i, e] of entries.entries()) {
    if (e.isFile && direct[i]) {                    // a plain file: no need to go through the entry
      if (files.length >= maxFiles) { truncated = true; break; }
      files.push(direct[i]);
      paths.push(`drop/${direct[i].name}`);
    } else await walk(e, 0);
  }
  return { files, paths, truncated };
}
