"""SQLite store: migrations, audit rows, artifacts containment, export and delete."""
import json
import os
import sqlite3
import stat
import zipfile

import pytest

from jev.store import MIGRATIONS, SCHEMA_VERSION, Store


def test_fresh_db_mode_wal_and_version(tmp_path):
    s = Store(tmp_path / "d" / "jev.db", tmp_path / "ws")
    assert s.db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    assert s.db.execute("PRAGMA journal_mode").fetchone()[0] == "wal"
    if os.name != "nt":
        assert stat.S_IMODE((tmp_path / "d" / "jev.db").stat().st_mode) == 0o600


def test_migrates_a_v1_database_and_expires_stale_pending(tmp_path):
    p = tmp_path / "jev.db"
    db = sqlite3.connect(p)
    for stmt in MIGRATIONS[0].split(";"):
        if stmt.strip():
            db.execute(stmt)
    db.execute("PRAGMA user_version = 1")
    db.execute("INSERT INTO approvals (id, kind, tool, args, status, created) VALUES "
               "('a-old', 'tool', 'run_shell', '{}', 'pending', 1)")
    db.commit()
    db.close()
    s = Store(p, tmp_path / "ws")
    assert s.db.execute("PRAGMA user_version").fetchone()[0] == SCHEMA_VERSION
    cols = [r[1] for r in s.db.execute("PRAGMA table_info(approvals)")]
    assert "remember" in cols
    a = s.get_approval("a-old")
    assert a["status"] == "expired" and a["decided_by"] == "restart"   # a restart can never answer it
    s.close()
    db = sqlite3.connect(p)
    db.execute(f"PRAGMA user_version = {SCHEMA_VERSION + 5}")
    db.commit()
    db.close()
    with pytest.raises(RuntimeError, match="newer"):
        Store(p, tmp_path / "ws")


def test_conversation_artifacts_export_delete(tmp_path):
    s = Store(tmp_path / "jev.db", tmp_path / "ws")
    c = s.create_conversation("hello", "llama3.1:8b")
    ws = s.workspace(c["id"])
    (ws / "plot.png").write_bytes(b"\x89PNG...")
    (ws / ".venv").mkdir()
    (ws / ".venv" / "big").write_bytes(b"x")
    s.add_turn("t-1", c["id"], "llama3.1:8b")
    s.add_message(c["id"], "user", "draw", "t-1")
    s.add_message(c["id"], "assistant", "", "t-1", tool_calls=[{"name": "make_plot", "arguments": {}}])
    s.add_tool_call("call-1", c["id"], "t-1", "make_plot", {"kind": "line"}, "sandbox", "run", "ok")
    s.finish_tool_call("call-1", True, "Saved")
    s.add_approval({"id": "a-1", "tool": "make_plot", "args": {"kind": "line"}, "status": "auto_run",
                    "conversation_id": c["id"], "turn_id": "t-1", "call_id": "call-1", "decided_by": "policy"})
    art = s.add_artifact(ws, ws / "plot.png", "image", "image/png", conversation_id=c["id"])
    assert art["path"] == "plot.png" and art["workspace"] == c["id"] and len(art["sha256"]) == 64
    row, path = s.artifact_file(art["id"])
    assert path == (ws / "plot.png").resolve()
    with pytest.raises(PermissionError):
        s.add_artifact(ws, tmp_path / "jev.db", "data", "x")
    # tamper with the row: a path escaping the workspace is refused when served
    s.db.execute("UPDATE artifacts SET path='../../jev.db' WHERE id=?", (art["id"],))
    with pytest.raises(PermissionError):
        s.artifact_file(art["id"])
    s.db.execute("UPDATE artifacts SET path='plot.png', workspace='../..' WHERE id=?", (art["id"],))
    with pytest.raises(PermissionError):
        s.artifact_file(art["id"])
    s.db.execute("UPDATE artifacts SET workspace=? WHERE id=?", (c["id"], art["id"]))
    # a symlink swapped in after the fact cannot point outside either
    (ws / "plot.png").unlink()
    os.symlink(tmp_path / "jev.db", ws / "plot.png")
    with pytest.raises(PermissionError):
        s.artifact_file(art["id"])
    (ws / "plot.png").unlink()
    (ws / "plot.png").write_bytes(b"\x89PNG...")

    z = s.export_conversation(c["id"], tmp_path / "exports")
    with zipfile.ZipFile(z) as zf:
        names = zf.namelist()
        data = json.loads(zf.read("conversation.json"))
    assert "files/plot.png" in names and not any(".venv" in n for n in names)
    assert data["conversation"]["id"] == c["id"] and len(data["messages"]) == 2
    assert data["tool_calls"][0]["result"] == "Saved" and data["approvals"][0]["status"] == "auto_run"

    s.delete_conversation(c["id"])
    assert s.get_conversation(c["id"]) is None and not ws.exists()
    assert s.messages(c["id"]) == [] and s.list_artifacts(c["id"]) == []
    a = s.get_approval("a-1")                  # the audit row stays, its arguments are redacted
    assert a["status"] == "auto_run" and a["args"] == {"redacted": "conversation deleted"}


def test_models_table(tmp_path):
    s = Store(tmp_path / "jev.db", tmp_path / "ws")
    s.record_model("mine", "/x/mine.gguf", "ab" * 32, 123)
    assert s.list_models()[0]["sha256"] == "ab" * 32
