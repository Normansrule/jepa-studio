"""JEV's SQLite store: `<root>/jev-data/jev.db` (WAL mode, file mode 0600, standard library only).

Tables
    conversations  id, title, model, workspace (folder name under workspace_root), created, updated
    turns          id, conversation_id, model, status, created, finished
    messages       conversation_id, turn_id, role (user|assistant|tool), content, tool_calls, tool_name
    tool_calls     call_id, turn_id, name, args, risk, decision, reason, ok, result (first 4000 chars)
    approvals      the audit log: one row for EVERY decision, automatic ones included
                   (status auto_run | auto_deny | remembered | pending | approved | denied | expired |
                   cancelled | failed, decided_by policy | session | settings | user | timeout)
    artifacts      id, conversation_id, workspace, path (relative to that workspace), kind, mime, sha256, bytes
    models         GGUF files you imported: name, path, sha256, bytes

Inspect it with any SQLite tool (`sqlite3 jev-data/jev.db "select * from approvals"`). Export a
conversation with `export_conversation` (JSON + its files, zipped) and delete with
`delete_conversation` (rows and workspace; the audit rows stay, with their arguments redacted).
`export_all` / `delete_all` do the same for everything at once (the app's Settings > Your data).
To remove everything JEV ever stored, including settings.json, delete the jev-data folder.
"""
from __future__ import annotations

import json
import os
import secrets
import shutil
import sqlite3
import threading
import time
import zipfile
from pathlib import Path

MIGRATIONS: list[str] = [
    # v1
    """
    CREATE TABLE conversations (id TEXT PRIMARY KEY, title TEXT, model TEXT, workspace TEXT NOT NULL,
                                created REAL, updated REAL);
    CREATE TABLE turns (id TEXT PRIMARY KEY, conversation_id TEXT, model TEXT, status TEXT, created REAL,
                        finished REAL);
    CREATE TABLE messages (id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT NOT NULL, turn_id TEXT,
                           role TEXT NOT NULL, content TEXT, tool_calls TEXT, tool_name TEXT, created REAL);
    CREATE INDEX messages_conv ON messages(conversation_id, id);
    CREATE TABLE tool_calls (call_id TEXT PRIMARY KEY, conversation_id TEXT, turn_id TEXT, name TEXT, args TEXT,
                             risk TEXT, decision TEXT, reason TEXT, ok INTEGER, result TEXT, started REAL,
                             finished REAL);
    CREATE TABLE approvals (id TEXT PRIMARY KEY, kind TEXT, tool TEXT, args TEXT, reason TEXT, risk TEXT,
                            rules TEXT, conversation_id TEXT, turn_id TEXT, call_id TEXT, status TEXT,
                            decided_by TEXT, created REAL, decided REAL, job_id TEXT);
    CREATE INDEX approvals_status ON approvals(status, created);
    CREATE TABLE artifacts (id TEXT PRIMARY KEY, conversation_id TEXT, turn_id TEXT, call_id TEXT,
                            workspace TEXT NOT NULL, path TEXT NOT NULL, kind TEXT, mime TEXT, title TEXT,
                            sha256 TEXT, bytes INTEGER, meta TEXT, created REAL);
    CREATE INDEX artifacts_conv ON artifacts(conversation_id, created);
    CREATE TABLE models (name TEXT PRIMARY KEY, path TEXT, sha256 TEXT, bytes INTEGER, created REAL);
    """,
    # v2: remember flag on approvals (which approvals were "remember for this session")
    """
    ALTER TABLE approvals ADD COLUMN remember TEXT;
    """,
]
SCHEMA_VERSION = len(MIGRATIONS)


def _now() -> float:
    return time.time()


def _row(r: sqlite3.Row | None) -> dict | None:
    if r is None:
        return None
    d = dict(r)
    for k in ("args", "rules", "tool_calls", "meta"):
        if k in d and isinstance(d[k], str):
            try:
                d[k] = json.loads(d[k])
            except json.JSONDecodeError:
                pass
    return d


class Store:
    def __init__(self, path: str | Path, workspace_root: str | Path):
        self.path = Path(path)
        self.workspace_root = Path(workspace_root).resolve()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            os.close(os.open(self.path, os.O_WRONLY | os.O_CREAT, 0o600))
        self.lock = threading.RLock()
        self.db = sqlite3.connect(self.path, check_same_thread=False, isolation_level=None, timeout=30)
        self.db.row_factory = sqlite3.Row
        self.db.execute("PRAGMA journal_mode=WAL")
        self.db.execute("PRAGMA foreign_keys=ON")
        self.migrate()
        for p in (self.path, Path(str(self.path) + "-wal"), Path(str(self.path) + "-shm")):
            try:
                if p.exists():
                    os.chmod(p, 0o600)
            except OSError:
                pass

    def migrate(self) -> int:
        with self.lock:
            v = self.db.execute("PRAGMA user_version").fetchone()[0]
            if v > SCHEMA_VERSION:
                raise RuntimeError(f"{self.path} was written by a newer JEV (schema {v}); upgrade jepa-studio")
            for i in range(v, SCHEMA_VERSION):
                self.db.execute("BEGIN")
                try:
                    for stmt in MIGRATIONS[i].split(";"):
                        if stmt.strip():
                            self.db.execute(stmt)
                    self.db.execute(f"PRAGMA user_version = {i + 1}")
                    self.db.execute("COMMIT")
                except Exception:
                    self.db.execute("ROLLBACK")
                    raise
            # approvals left pending by a previous backend session can never be answered now
            self.db.execute("UPDATE approvals SET status='expired', decided_by='restart', decided=? "
                            "WHERE status='pending'", (_now(),))
            return SCHEMA_VERSION

    def _x(self, sql: str, params=()) -> sqlite3.Cursor:
        with self.lock:
            return self.db.execute(sql, params)

    def close(self) -> None:
        with self.lock:
            self.db.close()

    # ------------------------------------------------------------------ conversations
    def create_conversation(self, title: str, model: str) -> dict:
        cid = "c-" + secrets.token_hex(6)
        t = _now()
        self._x("INSERT INTO conversations VALUES (?,?,?,?,?,?)", (cid, title[:120], model, cid, t, t))
        (self.workspace_root / cid).mkdir(parents=True, exist_ok=True)
        return self.get_conversation(cid)

    def get_conversation(self, cid: str) -> dict | None:
        return _row(self._x("SELECT * FROM conversations WHERE id=?", (cid,)).fetchone())

    def workspace(self, cid: str) -> Path:
        c = self.get_conversation(cid)
        if c is None:
            raise KeyError(cid)
        p = (self.workspace_root / c["workspace"]).resolve()
        if self.workspace_root not in p.parents:
            raise PermissionError("workspace outside workspace_root")
        p.mkdir(parents=True, exist_ok=True)
        return p

    def touch(self, cid: str, model: str | None = None) -> None:
        if model:
            self._x("UPDATE conversations SET updated=?, model=? WHERE id=?", (_now(), model, cid))
        else:
            self._x("UPDATE conversations SET updated=? WHERE id=?", (_now(), cid))

    def list_conversations(self) -> list[dict]:
        return [_row(r) for r in self._x("SELECT id, title, created, updated, model FROM conversations "
                                         "ORDER BY updated DESC").fetchall()]

    # ------------------------------------------------------------------ turns / messages
    def add_turn(self, tid: str, cid: str, model: str) -> None:
        self._x("INSERT INTO turns VALUES (?,?,?,?,?,?)", (tid, cid, model, "running", _now(), None))

    def finish_turn(self, tid: str, status: str) -> None:
        self._x("UPDATE turns SET status=?, finished=? WHERE id=?", (status, _now(), tid))

    def add_message(self, cid: str, role: str, content: str, turn_id: str | None = None,
                    tool_calls: list | None = None, tool_name: str | None = None) -> int:
        cur = self._x("INSERT INTO messages (conversation_id, turn_id, role, content, tool_calls, tool_name, created) "
                      "VALUES (?,?,?,?,?,?,?)", (cid, turn_id, role, content,
                                                 json.dumps(tool_calls) if tool_calls else None, tool_name, _now()))
        self.touch(cid)
        return cur.lastrowid

    def messages(self, cid: str, limit: int = 200) -> list[dict]:
        rows = self._x("SELECT * FROM (SELECT * FROM messages WHERE conversation_id=? ORDER BY id DESC LIMIT ?) "
                       "ORDER BY id", (cid, limit)).fetchall()
        return [_row(r) for r in rows]

    # ------------------------------------------------------------------ tool calls
    def add_tool_call(self, call_id: str, cid: str, tid: str, name: str, args: dict, risk: str, decision: str,
                      reason: str) -> None:
        self._x("INSERT OR REPLACE INTO tool_calls VALUES (?,?,?,?,?,?,?,?,?,?,?,?)",
                (call_id, cid, tid, name, json.dumps(args, default=str)[:200000], risk, decision, reason, None, None,
                 _now(), None))

    def finish_tool_call(self, call_id: str, ok: bool, result: str) -> None:
        self._x("UPDATE tool_calls SET ok=?, result=?, finished=? WHERE call_id=?",
                (int(ok), result[:4000], _now(), call_id))

    def tool_calls(self, turn_id: str) -> list[dict]:
        return [_row(r) for r in self._x("SELECT * FROM tool_calls WHERE turn_id=? ORDER BY started",
                                         (turn_id,)).fetchall()]

    # ------------------------------------------------------------------ approvals (audit log)
    def add_approval(self, a: dict) -> None:
        self._x("INSERT INTO approvals (id, kind, tool, args, reason, risk, rules, conversation_id, turn_id, call_id, "
                "status, decided_by, created, decided, job_id, remember) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (a["id"], a.get("kind", "tool"), a.get("tool"), json.dumps(a.get("args"), default=str)[:200000],
                 a.get("reason"), a.get("risk"), json.dumps(a.get("rules") or []), a.get("conversation_id"),
                 a.get("turn_id"), a.get("call_id"), a.get("status"), a.get("decided_by"),
                 a.get("created") or _now(), a.get("decided"), a.get("job_id"), a.get("remember")))

    def update_approval(self, aid: str, **fields) -> None:
        allowed = {"status", "decided_by", "decided", "job_id", "reason", "remember"}
        sets = [(k, v) for k, v in fields.items() if k in allowed]
        if sets:
            self._x(f"UPDATE approvals SET {', '.join(k + '=?' for k, _ in sets)} WHERE id=?",
                    [v for _, v in sets] + [aid])

    def get_approval(self, aid: str) -> dict | None:
        return _row(self._x("SELECT * FROM approvals WHERE id=?", (aid,)).fetchone())

    def list_approvals(self, status: str | None = None, limit: int = 500) -> list[dict]:
        if status:
            rows = self._x("SELECT * FROM approvals WHERE status=? ORDER BY created DESC LIMIT ?", (status, limit))
        else:
            rows = self._x("SELECT * FROM approvals ORDER BY created DESC LIMIT ?", (limit,))
        return [_row(r) for r in rows.fetchall()]

    # ------------------------------------------------------------------ artifacts
    def add_artifact(self, workspace: Path, path: Path, kind: str, mime: str, title: str = "", meta: dict | None = None,
                     conversation_id: str | None = None, turn_id: str | None = None, call_id: str | None = None,
                     sha256: str | None = None) -> dict:
        from .llm import sha256_file

        ws = Path(workspace).resolve()
        p = Path(path).resolve()
        if ws not in p.parents or (self.workspace_root not in ws.parents and ws != self.workspace_root):
            raise PermissionError(f"artifact {p} is outside its workspace")
        aid = "f-" + secrets.token_hex(8)
        rel_ws = ws.relative_to(self.workspace_root).as_posix()
        self._x("INSERT INTO artifacts VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
                (aid, conversation_id, turn_id, call_id, rel_ws, p.relative_to(ws).as_posix(), kind, mime,
                 title or p.name, sha256 or sha256_file(p), p.stat().st_size, json.dumps(meta or {}, default=str),
                 _now()))
        return self.get_artifact(aid)

    def get_artifact(self, aid: str) -> dict | None:
        return _row(self._x("SELECT * FROM artifacts WHERE id=?", (aid,)).fetchone())

    def list_artifacts(self, conversation_id: str | None = None) -> list[dict]:
        if conversation_id:
            rows = self._x("SELECT * FROM artifacts WHERE conversation_id=? ORDER BY created", (conversation_id,))
        else:
            rows = self._x("SELECT * FROM artifacts ORDER BY created DESC LIMIT 500")
        return [_row(r) for r in rows.fetchall()]

    def artifact_file(self, aid: str) -> tuple[dict, Path]:
        """The artifact's row and its absolute path, refusing anything outside its workspace."""
        a = self.get_artifact(aid)
        if a is None:
            raise KeyError(aid)
        ws = (self.workspace_root / a["workspace"]).resolve()
        if ws != self.workspace_root and self.workspace_root not in ws.parents:
            raise PermissionError("artifact workspace escapes workspace_root")
        p = (ws / a["path"]).resolve()     # resolve(): a symlink planted later cannot point outside
        if ws not in p.parents:
            raise PermissionError("artifact path escapes its workspace")
        if not p.is_file():
            raise FileNotFoundError(a["path"])
        return a, p

    # ------------------------------------------------------------------ models
    def record_model(self, name: str, path: str, sha256: str, size: int) -> None:
        self._x("INSERT OR REPLACE INTO models VALUES (?,?,?,?,?)", (name, path, sha256, size, _now()))

    def list_models(self) -> list[dict]:
        return [_row(r) for r in self._x("SELECT * FROM models ORDER BY created DESC").fetchall()]

    # ------------------------------------------------------------------ export / delete
    def conversation_json(self, cid: str) -> dict:
        c = self.get_conversation(cid)
        if c is None:
            raise KeyError(cid)
        turns = [_row(r) for r in self._x("SELECT * FROM turns WHERE conversation_id=? ORDER BY created", (cid,))]
        calls = [_row(r) for r in self._x("SELECT * FROM tool_calls WHERE conversation_id=? ORDER BY started", (cid,))]
        appr = [_row(r) for r in self._x("SELECT * FROM approvals WHERE conversation_id=? ORDER BY created", (cid,))]
        return {"format": "jev-conversation/1", "conversation": c, "turns": turns, "messages": self.messages(cid, 100000),
                "tool_calls": calls, "approvals": appr, "artifacts": self.list_artifacts(cid)}

    def export_conversation(self, cid: str, out_dir: str | Path) -> Path:
        """Write <out_dir>/<cid>-<time>.zip with conversation.json and every workspace file."""
        data = self.conversation_json(cid)
        ws = self.workspace(cid)
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"{cid}-{time.strftime('%Y%m%d-%H%M%S')}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("conversation.json", json.dumps(data, indent=2, default=str))
            for dirpath, dirnames, filenames in os.walk(ws):
                dirnames[:] = [d for d in dirnames if d not in (".venv", ".tmp", "node_modules", "__pycache__")]
                for fn in filenames:
                    p = Path(dirpath) / fn
                    if p.is_symlink() or not p.is_file():
                        continue
                    z.write(p, "files/" + p.relative_to(ws).as_posix())
        return out

    def delete_conversation(self, cid: str) -> None:
        c = self.get_conversation(cid)
        if c is None:
            raise KeyError(cid)
        ws = self.workspace(cid)
        with self.lock:
            self.db.execute("BEGIN")
            try:
                for t in ("messages", "tool_calls", "artifacts", "turns"):
                    self.db.execute(f"DELETE FROM {t} WHERE conversation_id=?", (cid,))
                self.db.execute("UPDATE approvals SET args=? WHERE conversation_id=? AND kind='tool'",
                                (json.dumps({"redacted": "conversation deleted"}), cid))
                self.db.execute("DELETE FROM conversations WHERE id=?", (cid,))
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
        if ws.exists() and self.workspace_root in ws.parents:
            shutil.rmtree(ws, ignore_errors=True)

    # ------------------------------------------------------------------ everything at once
    def _workspace_dirs(self) -> list[tuple[str, Path]]:
        """(name, folder) for every conversation workspace and the science job folder, all inside
        workspace_root (anything else is skipped, so a custom workspace_root is never wiped wholesale)."""
        out = []
        names = [r[0] for r in self._x("SELECT workspace FROM conversations").fetchall()] + ["_science"]
        for name in names:
            p = (self.workspace_root / str(name)).resolve()
            if self.workspace_root in p.parents and p.is_dir() and not p.is_symlink():
                out.append((p.relative_to(self.workspace_root).as_posix(), p))
        return out

    def export_all(self, out_dir: str | Path, settings: dict | None = None) -> Path:
        """Write <out_dir>/jev-data-<time>.zip: `jev-data.json` (every conversation with its turns,
        messages, tool calls and artifacts; the whole audit log; imported models; the settings) and
        `workspaces/<folder>/...` (every workspace file, without .venv / caches / symlinks)."""
        convs = [self.conversation_json(c["id"]) for c in self.list_conversations()]
        data = {"format": "jev-export/1", "exported": _now(), "schema_version": SCHEMA_VERSION,
                "conversations": convs,
                "approvals": [_row(r) for r in self._x("SELECT * FROM approvals ORDER BY created").fetchall()],
                "models": self.list_models(), "settings": settings or {}}
        out_dir = Path(out_dir)
        out_dir.mkdir(parents=True, exist_ok=True)
        out = out_dir / f"jev-data-{time.strftime('%Y%m%d-%H%M%S')}.zip"
        with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
            z.writestr("jev-data.json", json.dumps(data, indent=2, default=str))
            for name, ws in self._workspace_dirs():
                for dirpath, dirnames, filenames in os.walk(ws):
                    dirnames[:] = [d for d in dirnames if d not in (".venv", ".tmp", "node_modules", "__pycache__")
                                   and not (Path(dirpath) / d).is_symlink()]
                    for fn in filenames:
                        p = Path(dirpath) / fn
                        if p.is_symlink() or not p.is_file():
                            continue
                        z.write(p, f"workspaces/{name}/" + p.relative_to(ws).as_posix())
        return out

    def delete_all(self, keep_audit_log: bool = False, before: float | None = None) -> dict:
        """Delete every conversation (rows and workspace folders), the science outputs, the imported-model
        records and, unless keep_audit_log, the audit-log rows created before `before` (so the approval
        that asked for this deletion stays as the first row of the new log). Then VACUUM, so the
        deleted rows do not linger in free pages of jev.db. Returns counts."""
        dirs = self._workspace_dirs()
        counts = {}
        with self.lock:
            self.db.execute("BEGIN")
            try:
                for t in ("messages", "tool_calls", "artifacts", "turns", "conversations", "models"):
                    counts[t] = self.db.execute(f"DELETE FROM {t}").rowcount
                if keep_audit_log:
                    self.db.execute("UPDATE approvals SET args=? WHERE kind='tool'",
                                    (json.dumps({"redacted": "all data deleted"}),))
                    counts["approvals"] = 0
                else:
                    counts["approvals"] = self.db.execute(
                        "DELETE FROM approvals WHERE created < ? AND status != 'pending'",
                        (before if before is not None else _now(),)).rowcount
                self.db.execute("COMMIT")
            except Exception:
                self.db.execute("ROLLBACK")
                raise
            try:
                self.db.execute("PRAGMA wal_checkpoint(TRUNCATE)")
                self.db.execute("VACUUM")
            except sqlite3.Error:
                pass
        for _, p in dirs:
            shutil.rmtree(p, ignore_errors=True)
        counts["workspaces"] = len(dirs)
        return counts
