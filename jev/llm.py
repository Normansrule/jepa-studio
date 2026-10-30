"""Local model access: a small Ollama HTTP client (standard library only) and a scripted MockLLM.

JEV never talks to a hosted model. It talks to Ollama on your own machine (127.0.0.1:11434 by
default; any non-loopback address is refused unless you set `allow_remote_ollama` yourself),
with proxies disabled so no corporate or system proxy ever sees a prompt.

Where the model files live (they are ordinary files you can copy, back up or delete):

    Windows   %USERPROFILE%\\.ollama\\models
    macOS     ~/.ollama/models
    Linux     /usr/share/ollama/.ollama/models   (the official install script's systemd service)
              ~/.ollama/models                   (when you run `ollama serve` yourself)
    WSL2      as Linux, inside the WSL file system (\\\\wsl$\\Ubuntu\\usr\\share\\ollama\\...)

To keep them on a drive you choose, set OLLAMA_MODELS before Ollama starts:
    Windows   Settings > System > About > Advanced system settings > Environment Variables,
              add OLLAMA_MODELS=D:\\ollama-models, then quit Ollama from the tray and start it again
    macOS     launchctl setenv OLLAMA_MODELS /Volumes/Models/ollama   (then restart the Ollama app)
    Linux     sudo systemctl edit ollama   ->   [Service]  Environment="OLLAMA_MODELS=/data/ollama"
              (the folder must be writable by the `ollama` user), then sudo systemctl restart ollama
docs/jev/README.md has the full walkthrough.

Importing your own GGUF (`create`): the file's SHA-256 is computed and recorded in JEV's store,
uploaded to Ollama's blob store (POST /api/blobs/sha256:<digest>) and registered as a model
(POST /api/create {"model", "files"}). This is exactly what `ollama create NAME -f Modelfile`
with `FROM /abs/path/model.gguf` does; current Ollama versions no longer accept a Modelfile
string over HTTP.
"""
from __future__ import annotations

import hashlib
import json
import socket
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Callable, Iterator

from .settings import check_ollama_url

NOT_RUNNING = ("Ollama isn't running — start it with `ollama serve` or the Ollama app "
               "(install it from https://ollama.com/download; see docs/jev/README.md).")

TokenCb = Callable[[str], None]
CancelCb = Callable[[], bool]


class OllamaError(RuntimeError):
    """Ollama answered with an error, or the answer was unusable."""


class OllamaUnavailable(OllamaError):
    """Nothing is listening at the configured URL."""


class ToolsUnsupported(OllamaError):
    """The model has no native tool calling (the agent then uses the JSON protocol)."""


class Cancelled(Exception):
    pass


def is_cloud_model(name: str, info: dict | None = None) -> bool:
    """Ollama's "cloud" models run on ollama.com: your prompt would leave this machine."""
    n = name.lower()
    if n.endswith("-cloud") or n.endswith(":cloud") or "-cloud:" in n or ":cloud-" in n:
        return True
    return bool(info and (info.get("remote_host") or info.get("remote_model")))


def _opener() -> urllib.request.OpenerDirector:
    # ProxyHandler({}) disables HTTP(S)_PROXY: a loopback request must never go through a proxy
    return urllib.request.build_opener(urllib.request.ProxyHandler({}))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    from .tools.base import open_regular
    h = hashlib.sha256()
    with open_regular(path, "rb") as f:          # never blocks on a FIFO a sandboxed program left behind
        while b := f.read(chunk):
            h.update(b)
    return h.hexdigest()


class Ollama:
    def __init__(self, url: str = "http://127.0.0.1:11434", allow_remote: bool = False, timeout: float = 10.0):
        self.url = check_ollama_url(url, allow_remote)
        self.timeout = timeout
        self.opener = _opener()
        self._caps: dict[str, bool] = {}

    @classmethod
    def from_settings(cls, s: dict) -> "Ollama":
        return cls(s["ollama_url"], s.get("allow_remote_ollama", False))

    # ------------------------------------------------------------------ transport
    def _open(self, method: str, path: str, body=None, timeout: float | None = None, headers=None,
              raw: bytes | object | None = None, length: int | None = None):
        data = raw if raw is not None else (json.dumps(body).encode() if body is not None else None)
        req = urllib.request.Request(self.url + path, data=data, method=method)
        if body is not None:
            req.add_header("Content-Type", "application/json")
        if length is not None:
            req.add_header("Content-Length", str(length))
        for k, v in (headers or {}).items():
            req.add_header(k, v)
        try:
            return self.opener.open(req, timeout=timeout or self.timeout)
        except urllib.error.HTTPError as e:
            try:
                msg = json.loads(e.read() or b"{}").get("error") or e.reason
            except (ValueError, AttributeError):
                msg = e.reason
            if e.code == 400 and "does not support tools" in str(msg):
                raise ToolsUnsupported(str(msg)) from e
            if e.code == 404 and "not found" in str(msg).lower():
                raise OllamaError(f"{msg} (pull it first: `ollama pull <name>` or Models > Pull)") from e
            raise OllamaError(f"Ollama {method} {path} failed ({e.code}): {msg}") from e
        except (urllib.error.URLError, ConnectionError, socket.timeout, TimeoutError) as e:
            reason = getattr(e, "reason", e)
            if isinstance(reason, (ConnectionRefusedError, ConnectionResetError, socket.gaierror)) or \
                    "refused" in str(reason).lower():
                raise OllamaUnavailable(f"{NOT_RUNNING} (tried {self.url})") from e
            raise OllamaUnavailable(f"Could not reach Ollama at {self.url}: {reason}. {NOT_RUNNING}") from e

    def _json(self, method: str, path: str, body=None, timeout: float | None = None) -> dict:
        with self._open(method, path, body, timeout) as r:
            raw = r.read()
        return json.loads(raw or b"{}")

    def _stream(self, method: str, path: str, body: dict, cancelled: CancelCb | None = None,
                timeout: float = 600.0) -> Iterator[dict]:
        r = self._open(method, path, body, timeout)
        try:
            for line in r:
                if cancelled and cancelled():
                    raise Cancelled()
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError as e:
                    raise OllamaError(f"Ollama sent a line that is not JSON: {line[:200]!r}") from e
                if rec.get("error"):
                    msg = str(rec["error"])
                    if "does not support tools" in msg:
                        raise ToolsUnsupported(msg)
                    raise OllamaError(msg)
                yield rec
        finally:
            r.close()

    # ------------------------------------------------------------------ API
    def version(self) -> str:
        return str(self._json("GET", "/api/version", timeout=3).get("version", "?"))

    def tags(self) -> list[dict]:
        out = []
        for m in self._json("GET", "/api/tags").get("models", []):
            d = m.get("details") or {}
            out.append({"name": m.get("name") or m.get("model"), "size_bytes": m.get("size", 0),
                        "family": d.get("family"), "parameter_size": d.get("parameter_size"),
                        "quantization": d.get("quantization_level"), "digest": m.get("digest"),
                        "modified": m.get("modified_at"), "remote": is_cloud_model(m.get("name") or "", m)})
        return out

    def show(self, name: str) -> dict:
        return self._json("POST", "/api/show", {"model": name})

    def is_remote(self, name: str) -> bool:
        """True when Ollama says this model runs elsewhere (a cloud model, possibly copied under a local-
        looking name): /api/show's remote_host / remote_model, or the /api/tags entry. Unknown models
        are not remote (chat will fail on them anyway)."""
        if is_cloud_model(name):
            return True
        try:
            if is_cloud_model(name, self.show(name)):
                return True
        except OllamaError:
            pass
        try:
            for m in self._json("GET", "/api/tags").get("models", []):
                if (m.get("name") or m.get("model")) == name and is_cloud_model(name, m):
                    return True
        except OllamaError:
            pass
        return False

    def supports_tools(self, name: str) -> bool:
        if name not in self._caps:
            try:
                info = self.show(name)
                caps = info.get("capabilities")
                self._caps[name] = ("tools" in caps) if isinstance(caps, list) else \
                    (".Tools" in str(info.get("template", "")))
            except OllamaError:
                self._caps[name] = True  # unknown: try native first; ToolsUnsupported switches over
        return self._caps[name]

    def mark_no_tools(self, name: str) -> None:
        self._caps[name] = False

    def chat(self, model: str, messages: list[dict], tools: list[dict] | None = None,
             on_token: TokenCb | None = None, cancelled: CancelCb | None = None,
             options: dict | None = None, timeout: float = 900.0) -> dict:
        """Streaming /api/chat. Returns {content, tool_calls: [{id, name, arguments}], stats}."""
        body: dict = {"model": model, "messages": messages, "stream": True}
        if tools:
            body["tools"] = tools
        if options:
            body["options"] = options
        content: list[str] = []
        calls: list[dict] = []
        stats: dict = {}
        for rec in self._stream("POST", "/api/chat", body, cancelled, timeout):
            msg = rec.get("message") or {}
            t = msg.get("content") or ""
            if t:
                content.append(t)
                if on_token:
                    on_token(t)
            for tc in msg.get("tool_calls") or []:
                calls.append(normalize_tool_call(tc, len(calls)))
            if rec.get("done"):
                stats = {k: rec.get(k) for k in ("done_reason", "eval_count", "prompt_eval_count", "total_duration")}
        return {"content": "".join(content), "tool_calls": calls, "stats": stats}

    def pull(self, name: str, on_progress: Callable[[dict], None] | None = None,
             cancelled: CancelCb | None = None) -> dict:
        last: dict = {}
        for rec in self._stream("POST", "/api/pull", {"model": name, "stream": True}, cancelled, timeout=3600):
            last = rec
            if on_progress:
                total, done = rec.get("total"), rec.get("completed")
                frac = (done / total) if total and done is not None else None
                on_progress({"status": rec.get("status", ""), "completed": done, "total": total, "fraction": frac})
        if last.get("status") != "success":
            raise OllamaError(f"pull of {name} did not finish (last status: {last.get('status')!r})")
        return last

    def create_from_gguf(self, name: str, gguf_path: str | Path, on_progress: Callable[[dict], None] | None = None,
                         cancelled: CancelCb | None = None, sha256: str | None = None) -> str:
        """Upload a local GGUF to Ollama's blob store and register it as `name`. Returns its SHA-256."""
        p = Path(gguf_path).resolve()
        with open(p, "rb") as f:  # wsfs-ok: a GGUF file you chose (granted folder / CLI path), not a workspace
            if f.read(4) != b"GGUF":
                raise OllamaError(f"{p.name} is not a GGUF file (missing GGUF magic bytes)")
        digest = sha256 or sha256_file(p)  # wsfs-ok: the GGUF checked above (your file, not a workspace)
        blob = f"/api/blobs/sha256:{digest}"
        try:
            self._open("HEAD", blob, timeout=10).close()
            exists = True
        except OllamaError:
            exists = False
        if not exists:
            size = p.stat().st_size
            with open(p, "rb") as f:  # wsfs-ok: a GGUF file you chose (granted folder / CLI path), not a workspace
                self._open("POST", blob, raw=f, length=size, timeout=3600,
                           headers={"Content-Type": "application/octet-stream"}).close()
        body = {"model": name, "files": {p.name: f"sha256:{digest}"}, "stream": True}
        for rec in self._stream("POST", "/api/create", body, cancelled, timeout=3600):
            if on_progress:
                on_progress({"status": rec.get("status", ""), "fraction": None})
        return digest

    def delete(self, name: str) -> None:
        self._open("DELETE", "/api/delete", {"model": name}).close()


def normalize_tool_call(tc: dict, index: int = 0) -> dict:
    fn = tc.get("function") or tc
    args = fn.get("arguments", {})
    if isinstance(args, str):
        try:
            args = json.loads(args) if args.strip() else {}
        except json.JSONDecodeError:
            args = {"_unparsed": args}
    return {"id": tc.get("id") or f"call_{index}_{int(time.time() * 1000) % 100000}",
            "name": str(fn.get("name", "")), "arguments": args if isinstance(args, dict) else {"_value": args}}


class MockLLM:
    """Scripted stand-in for Ollama (tests, offline demos). `script` is a list of replies, each
    {"content": str, "tool_calls": [{"name", "arguments"}]} or a callable(messages) -> reply.
    Content is streamed word by word through on_token, like a real model."""

    url = "mock://"

    def __init__(self, script: list, tools: bool = True):
        self.script = list(script)
        self.calls: list[dict] = []
        self.tools = tools

    def supports_tools(self, name: str) -> bool:
        return self.tools

    def is_remote(self, name: str) -> bool:
        return is_cloud_model(name)

    def mark_no_tools(self, name: str) -> None:
        self.tools = False

    def chat(self, model, messages, tools=None, on_token=None, cancelled=None, options=None, timeout=0):
        self.calls.append({"model": model, "messages": [dict(m) for m in messages], "tools": tools})
        if not self.script:
            reply = {"content": "(mock: script exhausted)"}
        else:
            reply = self.script.pop(0)
            if callable(reply):
                reply = reply(messages)
        text = reply.get("content", "")
        for i, w in enumerate(text.split(" ")):
            if cancelled and cancelled():
                raise Cancelled()
            if on_token:
                on_token(("" if i == 0 else " ") + w)
        calls = [normalize_tool_call({"function": c}, i) for i, c in enumerate(reply.get("tool_calls") or [])]
        return {"content": text, "tool_calls": calls, "stats": {"done_reason": "stop"}}

    def version(self) -> str:
        return "mock"

    def tags(self) -> list[dict]:
        return [{"name": "mock", "size_bytes": 0, "family": "mock", "parameter_size": "0B", "quantization": None,
                 "digest": None, "modified": None, "remote": False}]
