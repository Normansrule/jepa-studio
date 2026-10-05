// CSV helpers. Parsing accepts numbers only (no quoting, no strings) - a time-series file is a
// table of numeric columns with an optional header row. Anything else is rejected with a reason.

export const CSV_LIMITS = { maxBytes: 20 * 1024 * 1024, maxRows: 200000, maxCols: 64 };

/** Parse numeric CSV text -> {columns: string[], data: Float32Array[] (per column), rows}. */
export function parseNumericCsv(text) {
  if (text.length > CSV_LIMITS.maxBytes) throw new Error('file is larger than 20 MB');
  const lines = text.split(/\r?\n/).filter((l) => l.trim().length);
  if (!lines.length) throw new Error('file is empty');
  const sep = lines[0].includes(',') ? ',' : lines[0].includes(';') ? ';' : lines[0].includes('\t') ? '\t' : ',';
  const first = lines[0].split(sep).map((s) => s.trim());
  const hasHeader = first.some((s) => s !== '' && !Number.isFinite(Number(s)));
  const cols = first.length;
  if (cols > CSV_LIMITS.maxCols) throw new Error(`more than ${CSV_LIMITS.maxCols} columns`);
  const columns = hasHeader ? first.map((s, i) => s.slice(0, 40) || `col ${i + 1}`) : first.map((_, i) => `col ${i + 1}`);
  const body = hasHeader ? lines.slice(1) : lines;
  if (body.length > CSV_LIMITS.maxRows) throw new Error(`more than ${CSV_LIMITS.maxRows} rows`);
  if (body.length < 8) throw new Error('needs at least 8 numeric rows');
  const data = columns.map(() => new Float32Array(body.length));
  body.forEach((line, r) => {
    const parts = line.split(sep);
    if (parts.length !== cols) throw new Error(`row ${r + (hasHeader ? 2 : 1)} has ${parts.length} fields, expected ${cols}`);
    for (let c = 0; c < cols; c++) {
      const s = parts[c].trim();
      const v = Number(s);
      if (s === '' || !Number.isFinite(v)) throw new Error(`row ${r + (hasHeader ? 2 : 1)}, column ${c + 1} is not a number`);
      data[c][r] = v;
    }
  });
  return { columns, data, rows: body.length };
}

/** (n, d) matrix -> CSV text with header e0..e{d-1} and optional label column. */
export function matrixToCsv(M, n, d, labels = null, classNames = null) {
  const head = [];
  if (labels) head.push('label');
  for (let k = 0; k < d; k++) head.push(`e${k}`);
  const rows = [head.join(',')];
  for (let r = 0; r < n; r++) {
    const row = [];
    if (labels) row.push(classNames ? classNames[labels[r]] : String(labels[r]));
    for (let k = 0; k < d; k++) row.push(Number(M[r * d + k]).toPrecision(7));
    rows.push(row.join(','));
  }
  return rows.join('\n') + '\n';
}
