// Audit log view (the Approvals panel's History): every decision JEV's policy or you made, from
// GET /api/jev/approvals?limit=5000, filtered in the page by decision, tool and date, and exported
// as CSV. Cells are written with textContent; the CSV neutralises spreadsheet formulas.

// status (jev/policy.py, jev/store.py) -> filter group
export const GROUPS = {
  approved: ['approved', 'approve', 'allowed'],
  denied: ['denied', 'deny'],
  auto: ['auto', 'auto_run', 'remembered'],
  refused: ['auto_deny', 'refused'],
  pending: ['pending'],
  other: ['expired', 'cancelled', 'failed'],
};

const toMs = (t) => (typeof t === 'number' ? t * (t < 1e12 ? 1000 : 1) : Date.parse(t));

/** Local calendar day (YYYY-MM-DD) of a backend timestamp. */
export function dayOf(t) {
  const d = new Date(toMs(t));
  if (Number.isNaN(+d)) return '';
  return `${d.getFullYear()}-${String(d.getMonth() + 1).padStart(2, '0')}-${String(d.getDate()).padStart(2, '0')}`;
}

/** f = {decision: '' | group, tool: '' | name, from: '' | 'YYYY-MM-DD', to: same}; dates inclusive. */
export function filterAudit(list, f = {}) {
  const allow = f.decision ? new Set(GROUPS[f.decision] || [f.decision]) : null;
  return list.filter((a) => {
    const st = String(a.status || 'pending').toLowerCase();
    if (allow && !allow.has(st)) return false;
    if (f.tool && String(a.tool || '') !== f.tool) return false;
    const day = dayOf(a.created);
    if (f.from && (!day || day < f.from)) return false;
    if (f.to && (!day || day > f.to)) return false;
    return true;
  });
}

const COLS = ['id', 'created', 'decided', 'kind', 'tool', 'status', 'decided_by', 'remember', 'risk', 'rules', 'reason', 'args',
  'conversation_id', 'turn_id', 'call_id', 'job_id'];

function cell(v) {
  let s = v === null || v === undefined ? '' : typeof v === 'string' ? v : JSON.stringify(v);
  if (/^[=+\-@\t\r]/.test(s)) s = `'${s}`;          // never a formula when opened in a spreadsheet
  return `"${s.replace(/"/g, '""')}"`;
}

/** RFC 4180 CSV with a header row; times as ISO 8601 (UTC). */
export function toCsv(list) {
  const iso = (t) => { const ms = toMs(t); return Number.isFinite(ms) ? new Date(ms).toISOString() : ''; };
  const lines = [COLS.join(',')];
  for (const a of list) lines.push(COLS.map((k) => cell(k === 'created' || k === 'decided' ? (a[k] ? iso(a[k]) : '') : a[k])).join(','));
  return `${lines.join('\r\n')}\r\n`;
}
