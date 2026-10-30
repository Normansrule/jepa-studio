"""Shared helpers for the JEV tests: a fake Ollama HTTP server and a few fixtures.

FakeOllama implements the parts of Ollama's API JEV uses (/api/version, /api/tags, /api/show,
/api/chat streaming NDJSON with scripted tool_calls, /api/pull streaming, /api/blobs, /api/create,
/api/delete). Nothing here needs Ollama, a GPU or the network.
"""
from __future__ import annotations

import hashlib
import json
import socket
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import pytest


def free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


class FakeOllama:
    def __init__(self):
        self.script: list = []          # replies for /api/chat: {"content", "tool_calls": [{"name","arguments"}]}
        self.requests: list[dict] = []
        self.models = {"qwen2.5-coder:7b": 4_700_000_000, "llama3.1:8b": 4_900_000_000}
        self.no_tools: set[str] = set()       # /api/show says no tools, chat with tools -> 400
        self.reject_tools: set[str] = set()   # /api/show says tools, but chat with tools -> 400 anyway
        self.remote: set[str] = set()         # /api/show reports remote_host: an Ollama cloud model under any name
        self.blobs: dict[str, int] = {}
        self.lock = threading.Lock()
        self.port = free_port()
        fake = self

        class H(BaseHTTPRequestHandler):
            protocol_version = "HTTP/1.1"

            def log_message(self, *a):
                pass

            def _json(self, code, obj):
                data = json.dumps(obj).encode()
                self.send_response(code)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(data)))
                self.end_headers()
                self.wfile.write(data)

            def _stream(self, recs):
                self.send_response(200)
                self.send_header("Content-Type", "application/x-ndjson")
                self.send_header("Transfer-Encoding", "chunked")
                self.end_headers()
                for r in recs:
                    line = (json.dumps(r) + "\n").encode()
                    self.wfile.write(f"{len(line):x}\r\n".encode() + line + b"\r\n")
                    self.wfile.flush()
                self.wfile.write(b"0\r\n\r\n")

            def _body(self):
                n = int(self.headers.get("Content-Length") or 0)
                return self.rfile.read(n) if n else b""

            def do_HEAD(self):  # noqa: N802
                d = self.path.rsplit(":", 1)[-1]
                self.send_response(200 if d in fake.blobs else 404)
                self.send_header("Content-Length", "0")
                self.end_headers()

            def do_GET(self):  # noqa: N802
                fake.requests.append({"method": "GET", "path": self.path})
                if self.path == "/api/version":
                    return self._json(200, {"version": "0.9.9-fake"})
                if self.path == "/api/tags":
                    return self._json(200, {"models": [
                        {"name": n, "model": n, "size": s, "digest": "d" * 64, "modified_at": "2026-10-01T00:00:00Z",
                         "details": {"family": "llama", "parameter_size": "7B", "quantization_level": "Q4_K_M"}}
                        for n, s in fake.models.items()]})
                self._json(404, {"error": "not found"})

            def do_DELETE(self):  # noqa: N802
                body = json.loads(self._body() or b"{}")
                fake.requests.append({"method": "DELETE", "path": self.path, "body": body})
                if body.get("model") in fake.models:
                    fake.models.pop(body["model"])
                    return self._json(200, {})
                self._json(404, {"error": f"model '{body.get('model')}' not found"})

            def do_POST(self):  # noqa: N802
                raw = self._body()
                if self.path.startswith("/api/blobs/sha256:"):
                    digest = self.path.rsplit(":", 1)[-1]
                    if hashlib.sha256(raw).hexdigest() != digest:
                        return self._json(400, {"error": "digest mismatch"})
                    fake.blobs[digest] = len(raw)
                    self.send_response(201)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                body = json.loads(raw or b"{}")
                fake.requests.append({"method": "POST", "path": self.path, "body": body})
                if self.path == "/api/show":
                    if body.get("model") not in fake.models:
                        return self._json(404, {"error": f"model '{body.get('model')}' not found"})
                    caps = ["completion"] + ([] if body["model"] in fake.no_tools else ["tools"])
                    info = {"capabilities": caps, "template": "{{ .Prompt }}"}
                    if body["model"] in fake.remote:
                        info.update(remote_host="https://ollama.com:443", remote_model="gpt-oss:120b")
                    return self._json(200, info)
                if self.path == "/api/chat":
                    if body.get("model") not in fake.models:
                        return self._json(404, {"error": f"model \"{body.get('model')}\" not found, try pulling it"})
                    if body.get("tools") and body["model"] in fake.no_tools | fake.reject_tools:
                        return self._json(400, {"error": f"registry.ollama.ai/library/{body['model']} does not "
                                                         "support tools"})
                    with fake.lock:
                        reply = fake.script.pop(0) if fake.script else {"content": "(fake: no script left)"}
                    if callable(reply):
                        reply = reply(body)
                    recs = []
                    words = (reply.get("content") or "").split(" ")
                    for i, w in enumerate(words):
                        if w or i == 0:
                            recs.append({"model": body["model"], "message": {"role": "assistant",
                                                                             "content": ("" if i == 0 else " ") + w},
                                         "done": False})
                    if reply.get("tool_calls"):
                        recs.append({"model": body["model"], "done": False, "message": {
                            "role": "assistant", "content": "",
                            "tool_calls": [{"function": {"name": c["name"], "arguments": c.get("arguments", {})}}
                                           for c in reply["tool_calls"]]}})
                    recs.append({"model": body["model"], "message": {"role": "assistant", "content": ""},
                                 "done": True, "done_reason": "stop", "eval_count": 7})
                    return self._stream(recs)
                if self.path == "/api/pull":
                    name = body.get("model")
                    if name == "does-not-exist":
                        return self._stream([{"status": "pulling manifest"},
                                             {"error": "pull model manifest: file does not exist"}])
                    total = 1000
                    recs = [{"status": "pulling manifest"}] + [
                        {"status": "pulling abc", "digest": "sha256:abc", "total": total, "completed": c}
                        for c in (0, 250, 500, 1000)] + [{"status": "verifying sha256 digest"}, {"status": "success"}]
                    fake.models[name] = total
                    return self._stream(recs)
                if self.path == "/api/create":
                    files = body.get("files") or {}
                    missing = [v for v in files.values() if v.split(":", 1)[-1] not in fake.blobs]
                    if not files or missing:
                        return self._json(400, {"error": "missing blob"})
                    fake.models[body["model"]] = sum(fake.blobs[v.split(":", 1)[-1]] for v in files.values())
                    return self._stream([{"status": "parsing GGUF"}, {"status": "writing manifest"},
                                         {"status": "success"}])
                self._json(404, {"error": "not found"})

        self.httpd = ThreadingHTTPServer(("127.0.0.1", self.port), H)
        self.thread = threading.Thread(target=self.httpd.serve_forever, daemon=True)

    @property
    def url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    def start(self) -> "FakeOllama":
        self.thread.start()
        time.sleep(0.05)
        return self

    def stop(self) -> None:
        self.httpd.shutdown()
        self.httpd.server_close()

    def chats(self) -> list[dict]:
        return [r["body"] for r in self.requests if r["path"] == "/api/chat"]


@pytest.fixture
def fake_ollama():
    f = FakeOllama().start()
    yield f
    f.stop()


@pytest.fixture
def jev_env(fake_ollama, tmp_path, monkeypatch):
    """A backend root wired to the fake Ollama; returns (root, fake)."""
    monkeypatch.setenv("JEV_OLLAMA_URL", fake_ollama.url)
    root = tmp_path / "root"
    root.mkdir()
    return root, fake_ollama


def dead_url() -> str:
    return f"http://127.0.0.1:{free_port()}"


def test_fake_ollama_speaks_ndjson(fake_ollama):
    from jev.llm import Ollama

    fake_ollama.script = [{"content": "hello there"}]
    out = Ollama(fake_ollama.url).chat("llama3.1:8b", [{"role": "user", "content": "hi"}])
    assert out["content"] == "hello there" and out["tool_calls"] == []
