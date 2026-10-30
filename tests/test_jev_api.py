"""/api/jev/* through the real jepa_studio server: guard, chat, approvals, artifacts, models, settings."""
import base64
import json
import os
import sys
import threading
import time
import types
import urllib.error
import urllib.request
from pathlib import Path

import pytest

from test_jev_support import FakeOllama, free_port


@pytest.fixture(scope="module")
def srv(tmp_path_factory):
    from jepa_studio import server

    fake = FakeOllama().start()
    old = os.environ.get("JEV_OLLAMA_URL")
    os.environ["JEV_OLLAMA_URL"] = fake.url
    root = tmp_path_factory.mktemp("jevsrv")
    granted = tmp_path_factory.mktemp("granted")
    port = free_port()
    token = "tok-" + base64.urlsafe_b64encode(os.urandom(12)).decode()
    th = threading.Thread(target=server.serve, kwargs=dict(root=str(root), port=port, token=token,
                                                           allowed_dirs=[str(granted)],
                                                           ready_file=open(os.devnull, "w")), daemon=True)
    th.start()
    time.sleep(0.4)
    yield types.SimpleNamespace(port=port, token=token, root=root, fake=fake, granted=granted)
    fake.stop()
    if old is None:
        os.environ.pop("JEV_OLLAMA_URL", None)
    else:
        os.environ["JEV_OLLAMA_URL"] = old


def req(s, path, body=None, method=None, token=True, host=None, origin=None, raw=False):
    r = urllib.request.Request(f"http://127.0.0.1:{s.port}{path}", method=method or ("POST" if body is not None else "GET"),
                               data=json.dumps(body).encode() if body is not None else None)
    if token:
        r.add_header("X-JEPA-Token", s.token if token is True else token)
    if host:
        r.add_header("Host", host)
    if origin:
        r.add_header("Origin", origin)
    if body is not None:
        r.add_header("Content-Type", "application/json")
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            data = resp.read()
            return resp.status, (data, dict(resp.headers)) if raw else json.loads(data)
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def poll(s, path, until=("done", "error", "cancelled"), timeout=60, stop_on=None):
    t0, n, events = time.monotonic(), 0, []
    while time.monotonic() - t0 < timeout:
        code, page = req(s, f"{path}?since={n}")
        assert code == 200, page
        events += page["events"]
        n = page["next"]
        if stop_on and any(e["event"] == stop_on for e in events):
            return events, page["status"]
        if page["status"] in until:
            return events, page["status"]
        time.sleep(0.05)
    raise AssertionError(f"timeout; events {[e['event'] for e in events]}")


def test_guard_applies_to_jev_routes(srv):
    for path in ("/api/jev/status", "/api/jev/conversations", "/api/jev/artifacts/f-x", "/api/jev/approvals"):
        assert req(srv, path, token=False)[0] == 401
        assert req(srv, path, token="wrong")[0] == 401
    assert req(srv, "/api/jev/status", host="evil.example:80")[0] == 421
    assert req(srv, "/api/jev/status", origin="https://evil.example")[0] == 403
    assert req(srv, "/api/jev/chat", body={"message": "hi"}, token=False)[0] == 401
    assert req(srv, "/api/jev/models/llama3.1:8b", method="DELETE", token=False)[0] == 401
    assert req(srv, "/api/jev/nope")[0] == 404


def test_status_models_settings(srv):
    code, st = req(srv, "/api/jev/status")
    assert code == 200, st
    assert st["ollama"]["reachable"] and st["ollama"]["version"] == "0.9.9-fake"
    assert {"qwen2.5-coder:7b", "llama3.1:8b"} <= {m["name"] for m in st["models"]}
    tools = {t["name"]: t for t in st["tools"]}
    assert tools["run_python"]["risk"] == "sandbox" and tools["run_shell"]["risk"] == "approve"
    assert set(tools["calculator"]) == {"name", "description", "risk", "category"}
    assert st["sandbox"]["network_isolation"] in ("enforced", "not available")
    assert st["sandbox"]["limits"]["wall_seconds"] == 300
    assert st["workspace_root"].startswith(str(srv.root))
    assert any(r["rule"] == "shell" and r["action"] == "ask" for r in st["approval_rules"])
    code, m = req(srv, "/api/jev/models")
    assert code == 200 and m["models"][0]["size_bytes"] > 0
    code, out = req(srv, "/api/jev/settings", body={"limits": {"wall_seconds": 120}, "default_model": "llama3.1:8b"})
    assert code == 200 and out["limits"]["wall_seconds"] == 120 and out["default_model"] == "llama3.1:8b"
    assert req(srv, "/api/jev/settings", body={"ollama_url": "http://10.0.0.5:11434"})[0] == 400
    assert req(srv, "/api/jev/settings", body={"approval_rules": {"shell": "run"}})[0] == 400
    assert req(srv, "/api/jev/settings", body={"allow_remote_ollama": True})[0] == 400
    assert req(srv, "/api/jev/settings", body={"limits": {"memory_mb": 1}})[0] == 400
    code, g = req(srv, "/api/jev/settings")
    assert g["limits"]["wall_seconds"] == 120 and g["rules"]
    req(srv, "/api/jev/settings", body={"default_model": "qwen2.5-coder:7b", "limits": {"wall_seconds": 300}})


def test_chat_turn_artifacts_and_downloads(srv):
    srv.fake.script = [
        {"content": "", "tool_calls": [
            {"name": "write_file", "arguments": {"path": "page.html", "content": "<script>alert(1)</script>"}},
            {"name": "run_python", "arguments": {"code": "open('data.bin','wb').write(bytes(range(256)) * 4000)"}}]},
        {"content": "Made two files."}]
    code, r = req(srv, "/api/jev/chat", body={"message": "make files"})
    assert code == 200 and r["conversation_id"].startswith("c-") and r["turn_id"].startswith("t-")
    events, status = poll(srv, f"/api/jev/turns/{r['turn_id']}/events")
    assert status == "done" and events[-1] == {**events[-1], "event": "done", "text": "Made two files."}
    assert all(isinstance(e["i"], int) and "t" in e for e in events)
    arts = {e["path"]: e for e in events if e["event"] == "artifact"}
    assert set(arts) == {"page.html", "data.bin"}
    # binary download: exact bytes, inline, streamed with a length
    code, (data, h) = req(srv, arts["data.bin"]["url"], raw=True)
    assert code == 200 and data == bytes(range(256)) * 4000
    assert h["Content-Type"] == "application/octet-stream" and h["Content-Length"] == str(len(data))
    assert h["Content-Disposition"].startswith('inline; filename="data.bin"')
    assert h["X-Content-Type-Options"] == "nosniff" and "sandbox" in h["Content-Security-Policy"]
    # model-written HTML is never served as HTML
    code, (data, h) = req(srv, arts["page.html"]["url"], raw=True)
    assert h["Content-Type"].startswith("text/plain") and b"<script>" in data
    assert req(srv, arts["data.bin"]["url"], token=False)[0] == 401
    assert req(srv, "/api/jev/artifacts/f-doesnotexist")[0] == 404
    # listings
    code, conv = req(srv, f"/api/jev/conversations/{r['conversation_id']}")
    assert code == 200 and [m["role"] for m in conv["messages"]][0] == "user" and len(conv["artifacts"]) == 2
    code, lst = req(srv, f"/api/jev/artifacts?conversation_id={r['conversation_id']}")
    assert {a["path"] for a in lst["artifacts"]} == {"page.html", "data.bin"}
    assert any(c["id"] == r["conversation_id"] for c in req(srv, "/api/jev/conversations")[1]["conversations"])
    # export
    code, (zipdata, h) = req(srv, f"/api/jev/conversations/{r['conversation_id']}/export", raw=True)
    assert code == 200 and zipdata[:2] == b"PK" and h["Content-Disposition"].startswith("attachment")


def test_artifact_path_containment(srv):
    from jev.api import get_api
    srv.fake.script = [{"content": "", "tool_calls": [{"name": "write_file", "arguments": {"path": "a.txt",
                                                                                          "content": "hi"}}]},
                       {"content": "ok"}]
    _, r = req(srv, "/api/jev/chat", body={"message": "write"})
    events, _ = poll(srv, f"/api/jev/turns/{r['turn_id']}/events")
    aid = [e for e in events if e["event"] == "artifact"][0]["id"]
    store = get_api(srv.root).rt.store
    (Path(srv.root) / "outside-secret.txt").write_text("secret")
    store.db.execute("UPDATE artifacts SET path='../../../outside-secret.txt' WHERE id=?", (aid,))
    code, body = req(srv, f"/api/jev/artifacts/{aid}")
    assert code == 403 and "secret" not in json.dumps(body)


def test_approval_flow_over_http(srv):
    srv.fake.script = [{"content": "", "tool_calls": [{"name": "run_shell", "arguments": {"command": "echo hi-$((2+3))"}}]},
                       {"content": "done shell"}]
    _, r = req(srv, "/api/jev/chat", body={"message": "shell"})
    events, status = poll(srv, f"/api/jev/turns/{r['turn_id']}/events", stop_on="approval_request")
    ar = [e for e in events if e["event"] == "approval_request"][0]
    assert status == "waiting_approval" and ar["name"] == "run_shell" and ar["reason"]
    code, pend = req(srv, "/api/jev/approvals?status=pending")
    p = [a for a in pend["approvals"] if a["id"] == ar["approval_id"]][0]
    assert p["tool"] == "run_shell" and p["turn_id"] == r["turn_id"] and p["args"] == {"command": "echo hi-$((2+3))"}
    assert req(srv, f"/api/jev/approvals/{ar['approval_id']}", body={"decision": "maybe"})[0] == 400
    code, out = req(srv, f"/api/jev/approvals/{ar['approval_id']}", body={"decision": "approve", "remember": None})
    assert code == 200 and out == {"ok": True, "status": "approved"}
    events, status = poll(srv, f"/api/jev/turns/{r['turn_id']}/events")
    res = [e for e in events if e["event"] == "tool_result"][0]
    assert status == "done" and res["ok"] and "hi-5" in res["content"]
    assert req(srv, "/api/jev/approvals/a-unknown", body={"decision": "approve"})[0] == 404
    code, allap = req(srv, "/api/jev/approvals")
    assert any(a["status"] == "auto_run" for a in allap["approvals"])        # automatic decisions are logged too


def test_cancel_turn_over_http(srv):
    srv.fake.script = [{"content": "", "tool_calls": [{"name": "run_python", "arguments": {
        "code": "import time\ntime.sleep(60)"}}]}, {"content": "never"}]
    _, r = req(srv, "/api/jev/chat", body={"message": "sleep"})
    poll(srv, f"/api/jev/turns/{r['turn_id']}/events", stop_on="tool_call")
    time.sleep(0.3)
    assert req(srv, f"/api/jev/turns/{r['turn_id']}/cancel", body={})[1] == {"ok": True}
    events, status = poll(srv, f"/api/jev/turns/{r['turn_id']}/events", timeout=20)
    assert status == "cancelled" and events[-1]["event"] == "cancelled"
    assert req(srv, "/api/jev/turns/t-nope/events")[0] == 404


def test_model_pull_import_delete_need_approval(srv, tmp_path):
    code, out = req(srv, "/api/jev/models/pull", body={"name": "phi3:mini"})
    assert code == 200 and out["status"] == "pending" and "job_id" not in out
    assert "phi3:mini" not in srv.fake.models                  # nothing downloaded before you approve
    code, dec = req(srv, f"/api/jev/approvals/{out['approval_id']}", body={"decision": "approve"})
    assert dec["status"] == "approved" and dec["job_id"].startswith("j-")
    events, status = poll(srv, f"/api/jev/jobs/{dec['job_id']}/events")
    assert status == "done", events
    assert [e["fraction"] for e in events if e["event"] == "progress" and e["fraction"] is not None][-1] == 1.0
    assert "phi3:mini" in srv.fake.models
    assert req(srv, "/api/jev/models/pull", body={"name": "../../etc"})[0] == 400
    # import: only files inside granted folders
    outside = tmp_path / "m.gguf"
    outside.write_bytes(b"GGUF" + b"\0" * 100)
    assert req(srv, "/api/jev/models/import", body={"gguf_path": str(outside), "name": "mine"})[0] == 403
    g = srv.granted / "m.gguf"
    g.write_bytes(b"GGUF" + os.urandom(1000))
    code, out = req(srv, "/api/jev/models/import", body={"gguf_path": str(g), "name": "mine"})
    assert code == 200 and out["status"] == "pending"
    _, dec = req(srv, f"/api/jev/approvals/{out['approval_id']}", body={"decision": "approve"})
    events, status = poll(srv, f"/api/jev/jobs/{dec['job_id']}/events")
    import hashlib
    assert status == "done" and events[-1]["sha256"] == hashlib.sha256(g.read_bytes()).hexdigest()
    _, m = req(srv, "/api/jev/models")
    assert m["imported"][0]["sha256"] == events[-1]["sha256"]
    # delete (name with ':' url-encoded) and deny path
    code, out = req(srv, "/api/jev/models/llama3.1%3A8b", method="DELETE")
    assert code == 200 and out["status"] == "pending"
    _, dec = req(srv, f"/api/jev/approvals/{out['approval_id']}", body={"decision": "deny"})
    assert dec == {"ok": True, "status": "denied"} and "llama3.1:8b" in srv.fake.models
    _, out = req(srv, "/api/jev/models/llama3.1:8b", method="DELETE")
    _, dec = req(srv, f"/api/jev/approvals/{out['approval_id']}", body={"decision": "approve"})
    poll(srv, f"/api/jev/jobs/{dec['job_id']}/events")
    assert "llama3.1:8b" not in srv.fake.models


def test_conversation_files_and_delete(srv):
    srv.fake.script = [{"content": "hello"}]
    _, r = req(srv, "/api/jev/chat", body={"message": "hi"})
    poll(srv, f"/api/jev/turns/{r['turn_id']}/events")
    cid = r["conversation_id"]
    code, up = req(srv, f"/api/jev/conversations/{cid}/files",
                   body={"name": "in/data.csv", "content_base64": base64.b64encode(b"a,b\n1,2\n").decode()})
    assert code == 200 and up["path"] == "in/data.csv" and up["artifact"]["kind"] == "data"
    assert req(srv, f"/api/jev/conversations/{cid}/files", body={"name": "../x.csv", "content_base64": ""})[0] == 400
    assert req(srv, f"/api/jev/conversations/{cid}/files", body={"name": "x.txt", "from_path": "/etc/hostname"})[0] == 403
    (srv.granted / "photo.png").write_bytes(b"\x89PNG fake")
    code, up = req(srv, f"/api/jev/conversations/{cid}/files",
                   body={"name": "photo.png", "from_path": str(srv.granted / "photo.png")})
    assert code == 200 and up["artifact"]["kind"] == "image"
    # attachments must be in the workspace
    srv.fake.script = [{"content": "seen"}]
    code, r2 = req(srv, "/api/jev/chat", body={"message": "look", "conversation_id": cid, "attachments": ["in/data.csv"]})
    assert code == 200
    poll(srv, f"/api/jev/turns/{r2['turn_id']}/events")
    assert "in/data.csv" in srv.fake.chats()[-1]["messages"][-1]["content"]
    assert req(srv, "/api/jev/chat", body={"message": "x", "conversation_id": cid, "attachments": ["/etc/passwd"]})[0] == 403
    # delete asks; approving removes rows and the workspace folder
    from jev.api import get_api
    ws = get_api(srv.root).rt.store.workspace(cid)
    code, out = req(srv, f"/api/jev/conversations/{cid}", method="DELETE")
    assert code == 200 and out["status"] == "pending" and ws.exists()
    req(srv, f"/api/jev/approvals/{out['approval_id']}", body={"decision": "approve"})
    assert req(srv, f"/api/jev/conversations/{cid}")[0] == 404 and not ws.exists()


def test_science_routes(srv, monkeypatch):
    def render(params, out_dir, progress, cancelled):
        progress(0.5, "half")
        p = Path(out_dir) / "bh.png"
        p.write_bytes(b"\x89PNG" + b"0" * 100)
        return [p, Path("/etc/passwd")]          # anything outside out_dir is ignored

    def run(kind, params, out_dir, progress, cancelled):
        p = Path(out_dir) / f"{kind}.mp4"
        p.write_bytes(b"\0\0\0 ftypisom")
        return [p]

    import jev
    fake_pkg = types.ModuleType("jev.science")
    fake_pkg.blackhole = types.SimpleNamespace(render=render)
    fake_pkg.sims = types.SimpleNamespace(run=run)
    monkeypatch.setitem(sys.modules, "jev.science", fake_pkg)
    monkeypatch.setattr(jev, "science", fake_pkg, raising=False)
    code, out = req(srv, "/api/jev/science/blackhole", body={"spin": 0.9, "width": 64, "height": 36})
    assert code == 200, out
    events, status = poll(srv, f"/api/jev/jobs/{out['job_id']}/events")
    assert status == "done" and [e["path"] for e in events if e["event"] == "artifact"] == ["bh.png"]
    assert any(e["event"] == "progress" and e["fraction"] == 0.5 for e in events)
    url = [e for e in events if e["event"] == "artifact"][0]["url"]
    code, (data, h) = req(srv, url, raw=True)
    assert h["Content-Type"] == "image/png" and data.startswith(b"\x89PNG")
    code, out = req(srv, "/api/jev/science/simulate", body={"kind": "pendulum", "params": {}})
    events, status = poll(srv, f"/api/jev/jobs/{out['job_id']}/events")
    art = [e for e in events if e["event"] == "artifact"][0]
    assert art["mime"] == "video/mp4" and art["kind"] == "video"
    assert req(srv, "/api/jev/science/blackhole", body={"width": 100000, "height": 100000})[0] == 400
    assert req(srv, "/api/jev/science/simulate", body={"params": {}})[0] == 400
    # a render estimated over 10 minutes goes through the approval table (gpu_long) first
    fake_pkg.blackhole.estimate_seconds = lambda params: {"seconds": 3600.0}
    code, out = req(srv, "/api/jev/science/blackhole", body={"spin": 0.5, "width": 3840, "height": 2160, "frames": 600})
    assert code == 200 and out["status"] == "pending" and "60 minutes" in out["reason"] and "job_id" not in out
    _, dec = req(srv, f"/api/jev/approvals/{out['approval_id']}", body={"decision": "approve"})
    events, status = poll(srv, f"/api/jev/jobs/{dec['job_id']}/events")
    assert status == "done"
    fake_pkg.blackhole.estimate_seconds = lambda params: {"seconds": 5.0}
    code, out = req(srv, "/api/jev/science/blackhole", body={"spin": 0.5})
    assert out["job_id"] and out["estimate_s"] == 5.0


def test_create_conversation_and_ui_upload_shape(srv):
    code, c = req(srv, "/api/jev/conversations", body={"title": "Empty"})
    assert code == 200 and c["id"].startswith("c-") and c["title"] == "Empty"
    code, up = req(srv, f"/api/jev/conversations/{c['id']}/files",
                   body={"name": "n.txt", "mime": "text/plain", "data_base64": base64.b64encode(b"hello").decode()})
    assert code == 200 and up["path"] == "n.txt" and up["name"] == "n.txt" and up["bytes"] == 5
    assert req(srv, "/api/jev/conversations", body={"model": "bad model!"})[0] == 400


def test_paths_audit_limit_export_and_delete_all(srv):
    """Settings > Your data in the app: where things live, the whole audit log, export everything,
    delete everything (asks first). Runs last: it empties the store the other tests filled."""
    import zipfile
    import io
    from jev.api import get_api
    code, g = req(srv, "/api/jev/settings")
    assert code == 200 and set(g["paths"]) == {"data_dir", "settings", "db", "workspaces", "exports"}
    assert g["paths"]["db"].endswith("jev.db") and g["paths"]["settings"].endswith("settings.json")
    assert g["paths"]["data_dir"].startswith(str(srv.root))
    # audit log with a limit
    code, few = req(srv, "/api/jev/approvals?limit=2")
    assert code == 200 and len(few["approvals"]) == 2
    assert req(srv, "/api/jev/approvals?limit=0")[0] == 400
    assert req(srv, "/api/jev/approvals?limit=abc")[0] == 400
    # a conversation with a file, then export everything
    srv.fake.script = [{"content": "", "tool_calls": [{"name": "write_file", "arguments": {"path": "keep.txt",
                                                                                          "content": "mine"}}]},
                       {"content": "ok"}]
    _, r = req(srv, "/api/jev/chat", body={"message": "write a file"})
    poll(srv, f"/api/jev/turns/{r['turn_id']}/events")
    cid = r["conversation_id"]
    ws = get_api(srv.root).rt.store.workspace(cid)
    code, (zipdata, h) = req(srv, "/api/jev/export", raw=True)
    assert code == 200 and h["Content-Disposition"].startswith("attachment") and h["Content-Type"] == "application/zip"
    z = zipfile.ZipFile(io.BytesIO(zipdata))
    data = json.loads(z.read("jev-data.json"))
    assert data["format"] == "jev-export/1" and data["approvals"] and "rules" not in data["settings"]
    assert any(c["conversation"]["id"] == cid for c in data["conversations"])
    assert z.read(f"workspaces/{ws.name}/keep.txt") == b"mine"
    assert req(srv, "/api/jev/data/delete", body={"keep_audit_log": "yes"})[0] == 400
    # deleting everything asks; deny keeps it all
    code, out = req(srv, "/api/jev/data/delete", body={})
    assert code == 200 and out["status"] == "pending"
    _, pend = req(srv, "/api/jev/approvals?status=pending")
    row = [a for a in pend["approvals"] if a["id"] == out["approval_id"]][0]
    assert row["tool"] == "data_delete" and row["kind"] == "data_delete" and "delete" in row["rules"]
    assert row["args"]["keep_audit_log"] is False and row["args"]["conversations"] >= 1
    req(srv, f"/api/jev/approvals/{out['approval_id']}", body={"decision": "deny"})
    assert req(srv, f"/api/jev/conversations/{cid}")[0] == 200 and ws.exists()
    # approve: conversations, workspaces, exports and the old audit rows are gone; the approval stays
    _, out = req(srv, "/api/jev/data/delete", body={"keep_audit_log": False})
    code, dec = req(srv, f"/api/jev/approvals/{out['approval_id']}", body={"decision": "approve"})
    assert code == 200 and dec == {"ok": True, "status": "approved"}
    assert req(srv, "/api/jev/conversations")[1]["conversations"] == []
    assert not ws.exists() and not (Path(g["paths"]["exports"])).exists()
    _, log = req(srv, "/api/jev/approvals")
    assert [a["id"] for a in log["approvals"]] == [out["approval_id"]] and log["approvals"][0]["status"] == "approved"
    assert req(srv, "/api/jev/models")[1]["imported"] == []
    assert Path(g["paths"]["settings"]).exists()        # settings.json (written by an earlier test) is kept
