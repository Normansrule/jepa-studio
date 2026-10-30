"""`/api/jev/*` routes (docs/jev/api.md), mounted by jepa_studio/server.py behind the same guard as
every other route: 127.0.0.1 only, X-JEPA-Token, Host and Origin checks, JSON bodies <= 2 MB.

`JevApi.handle(method, parts, query, body)` returns a JSON-able dict or a FileResponse, or raises
HttpError(status, message). It never executes anything from a request directly: chat messages go
to the agent (whose tools pass through jev.policy), and downloads / imports / deletions only create
approval requests until you approve them.
"""
from __future__ import annotations

import base64
import re
import threading
import shutil
import time
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote

from . import __version__
from .agent import Runtime, artifact_json, register_artifacts
from .llm import OllamaError, OllamaUnavailable, sha256_file
from .policy import GPU_MAX_MINUTES, Decision, canonical_args, check_path, describe_rules
from .settings import SettingsError
from .tools.base import REGISTRY, ToolResult, describe

MODEL_NAME = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]{0,199}$")
# served as text/plain so a model-written page can never run script in the app's origin
ACTIVE_TYPES = {"text/html", "application/xhtml+xml", "text/javascript", "application/javascript",
                "application/x-javascript", "text/xml", "application/xml"}
MAX_PIXELS = 7680 * 4320
MAX_FRAMES = 20000


class HttpError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


@dataclass
class FileResponse:
    path: Path
    mime: str
    filename: str
    disposition: str = "inline"
    chunk: int = 256 * 1024


_GPU: dict = {}


def gpu_info() -> dict:
    if _GPU:
        return dict(_GPU)
    info: dict = {"cuda": False, "mps": False, "devices": []}
    try:
        import torch
        if torch.cuda.is_available():
            info["cuda"] = True
            for i in range(torch.cuda.device_count()):
                p = torch.cuda.get_device_properties(i)
                info["devices"].append({"name": p.name, "memory_gb": round(p.total_memory / 2**30, 1)})
        mps = getattr(torch.backends, "mps", None)
        info["mps"] = bool(mps and mps.is_available())
    except Exception as e:  # noqa: BLE001
        info["error"] = str(e)[:200]
    _GPU.update(info)
    return info


def _limit(raw, default: int = 500, top: int = 10000) -> int:
    if raw in (None, ""):
        return default
    try:
        n = int(raw)
    except (TypeError, ValueError) as e:
        raise HttpError(400, "limit must be a whole number") from e
    if not 1 <= n <= top:
        raise HttpError(400, f"limit must be between 1 and {top}")
    return n


def _model_name(name) -> str:
    if not isinstance(name, str) or not MODEL_NAME.match(name) or ".." in name:
        raise HttpError(400, "model name must look like llama3.1:8b or qwen2.5-coder:7b")
    return name


class JevApi:
    def __init__(self, root: str | Path, studio=None, llm=None, env: dict | None = None):
        self.rt = Runtime(root, studio=studio, llm=llm, env=env)
        self.studio = studio

    # ------------------------------------------------------------------ dispatch
    def handle(self, method: str, parts: list[str], q: dict, body: dict):
        p = [unquote(x) for x in parts]
        try:
            return self._route(method, p, q, body)
        except HttpError:
            raise
        except OllamaUnavailable as e:
            raise HttpError(503, str(e)) from e
        except OllamaError as e:
            raise HttpError(502, str(e)[:500]) from e
        except KeyError as e:
            raise HttpError(404, f"not found: {str(e).strip(chr(39))}") from e
        except PermissionError as e:
            raise HttpError(403, str(e)[:500]) from e
        except FileNotFoundError as e:
            raise HttpError(404, str(e)[:500]) from e
        except (SettingsError, ValueError, TypeError) as e:
            raise HttpError(400, str(e)[:500]) from e
        except RuntimeError as e:
            raise HttpError(409, str(e)[:500]) from e

    def _route(self, method: str, p: list[str], q: dict, body: dict):
        rt = self.rt
        one = lambda k, d=None: (q.get(k) or [d])[0]  # noqa: E731
        key = (method, tuple(p))
        simple = {
            ("GET", ("status",)): self.status,
            ("GET", ("models",)): lambda: {"models": rt.llm().tags(), "imported": rt.store.list_models()},
            ("POST", ("models", "pull")): lambda: self.model_pull(body),
            ("POST", ("models", "import")): lambda: self.model_import(body),
            ("GET", ("settings",)): lambda: {**rt.settings.public(), "rules": describe_rules(rt.settings.resolved()),
                                             "paths": self.paths()},
            ("POST", ("settings",)): lambda: self.update_settings(body),
            ("POST", ("chat",)): lambda: rt.start_turn(body.get("message"), body.get("conversation_id"),
                                                      body.get("model"), body.get("attachments")),
            ("GET", ("conversations",)): lambda: {"conversations": rt.store.list_conversations()},
            ("POST", ("conversations",)): lambda: self.conversation_create(body),
            ("GET", ("approvals",)): lambda: {"approvals": [self._approval_json(a) for a in
                                                            rt.store.list_approvals(one("status"),
                                                                                    _limit(one("limit")))]},
            ("GET", ("artifacts",)): lambda: {"artifacts": [artifact_json(a) for a in
                                                            rt.store.list_artifacts(one("conversation_id"))]},
            ("GET", ("export",)): self.export_all,
            ("POST", ("data", "delete")): lambda: self.data_delete(body),
            ("GET", ("jobs",)): lambda: {"jobs": [j.to_json() for j in list(rt.jobs.jobs.values())[-100:]]},
            ("POST", ("science", "blackhole")): lambda: self.science("blackhole", body),
            ("POST", ("science", "simulate")): lambda: self.science("simulate", body),
        }
        if key in simple:
            return simple[key]()
        if method == "DELETE" and len(p) >= 2 and p[0] == "models":
            return self.model_delete("/".join(p[1:]))
        if len(p) == 3 and p[0] in ("turns", "jobs"):
            jid, action = p[1], p[2]
            if method == "GET" and action == "events":
                j = rt.jobs.get(jid)
                if j is None:
                    raise HttpError(404, f"unknown {p[0][:-1]} {jid} (jobs live only while the backend runs)")
                return j.since(int(one("since", "0") or 0))
            if method == "POST" and action == "cancel":
                if rt.jobs.get(jid) is None:
                    raise HttpError(404, f"unknown {p[0][:-1]} {jid}")
                return {"ok": rt.cancel(jid)}
        if p[:1] == ["conversations"] and len(p) >= 2:
            cid = p[1]
            if len(p) == 2 and method == "GET":
                c = rt.store.get_conversation(cid)
                if c is None:
                    raise HttpError(404, "unknown conversation")
                return {**c, "messages": rt.store.messages(cid, 1000),
                        "artifacts": [artifact_json(a) for a in rt.store.list_artifacts(cid)]}
            if len(p) == 2 and method == "DELETE":
                return self.conversation_delete(cid)
            if len(p) == 3 and p[2] == "export" and method == "GET":
                return self.conversation_export(cid)
            if len(p) == 3 and p[2] == "files" and method == "POST":
                return self.add_file(cid, body)
        if p[:1] == ["approvals"] and len(p) == 2 and method == "POST":
            try:
                # approving needs the SHA-256 the card showed: a stale or forged answer approves nothing
                return rt.broker.decide(p[1], body.get("decision"), body.get("remember"),
                                        args_sha256=body.get("args_sha256"), require_hash=True)
            except KeyError as e:
                raise HttpError(404, "unknown approval") from e
        if p[:1] == ["artifacts"] and len(p) == 2 and method == "GET":
            return self.artifact_file(p[1])
        raise HttpError(404, "not found")

    # ------------------------------------------------------------------ status / settings
    def status(self) -> dict:
        rt = self.rt
        s = rt.settings.resolved()
        llm = rt.llm()
        ollama = {"url": s["ollama_url"], "reachable": False, "version": None}
        models: list = []
        try:
            ollama["version"] = llm.version()
            ollama["reachable"] = True
            models = llm.tags()
        except OllamaError as e:
            ollama["error"] = str(e)
        return {"version": __version__, "ollama": ollama, "default_model": s["default_model"], "models": models,
                "tools": [{k: v for k, v in describe(sp).items() if k != "parameters"} for sp in REGISTRY.values()],
                "workspace_root": s["workspace_root"], "data_dir": s["data_dir"],
                "sandbox": rt.sandbox.capabilities(), "network_allowed": s["network_allowed"], "gpu": gpu_info(),
                "approval_rules": describe_rules(s)}

    def paths(self) -> dict:
        """Where everything lives on disk (shown in the app so you can back it up or inspect it)."""
        s = self.rt.settings.resolved()
        d = Path(s["data_dir"])
        return {"data_dir": str(d), "settings": str(self.rt.settings.path), "db": str(self.rt.store.path),
                "workspaces": s["workspace_root"], "exports": str(d / "exports")}

    def update_settings(self, body: dict) -> dict:
        """Change the API-writable settings; every change is an audit-log row with the values before and after."""
        import secrets as _secrets

        before = self.rt.settings.public()
        out = self.rt.settings.update(body, api=True)
        self.rt.refresh()
        changed = {k: {"before": before.get(k), "after": out.get(k)} for k in body if before.get(k) != out.get(k)}
        self.rt.store.add_approval({"id": "a-" + _secrets.token_hex(6), "kind": "settings", "tool": "settings_update",
                                    "args": {"patch": body, "changed": changed}, "risk": "approve",
                                    "reason": "settings changed through the API: " +
                                              (", ".join(sorted(changed)) or "no effective change"),
                                    "rules": [], "status": "applied", "decided_by": "user", "decided": time.time()})
        return out

    # ------------------------------------------------------------------ approvals for direct actions
    def _gate(self, kind: str, rule: str, tool: str, args: dict, reason: str, start, extra_rules=()) -> dict:
        """Create an approval (or honor the table: 'run' starts now, 'deny' refuses) for a UI action."""
        rt = self.rt
        rt.policy.settings = rt.settings.resolved()
        rules = [rule, *extra_rules]
        actions = [rt.policy.action_for(r) for r in rules]
        if "deny" in actions:
            rt.broker.record_auto(Decision("deny", reason, rules), tool, args, "approve")
            raise HttpError(403, f"refused by approval_rules: {reason}")
        if all(a == "run" for a in actions):
            aid = rt.broker.record_auto(Decision("run", reason, rules, "settings"), tool, args, "approve")
            jid = start()
            rt.store.update_approval(aid, job_id=jid)
            return {"approval_id": aid, "status": "auto_run", **({"job_id": jid} if jid else {})}
        a = rt.broker.request(kind, tool, args, reason, "approve", rules, on_approve=start)
        return {"approval_id": a.id, "status": "pending", "args": args, "args_canonical": canonical_args(args),
                "args_sha256": a.args_sha256}

    def _approval_json(self, row: dict) -> dict:
        live = self.rt.broker.get(row["id"])
        out = {k: row.get(k) for k in ("id", "created", "kind", "tool", "args", "args_sha256", "reason", "risk",
                                       "rules", "conversation_id", "turn_id", "call_id", "status", "decided_by",
                                       "decided", "remember")}
        if live is not None:              # the exact, uncut arguments of a call that is still pending
            out["args"], out["args_sha256"] = live.args, live.args_sha256
            out["args_canonical"] = canonical_args(live.args)
        return out | {"job_id": (live.job_id if live else None) or row.get("job_id")}

    def model_pull(self, body: dict) -> dict:
        name = _model_name(body.get("name"))
        rt = self.rt

        def start():
            def work(job):
                job.emit("start", model=name, action="pull")
                last = {"t": 0.0, "f": -1.0, "s": None}

                def prog(d):   # throttled: a new status, +1 % or 0.5 s since the last event
                    now, f = time.monotonic(), d.get("fraction")
                    if d.get("status") != last["s"] or (f is not None and f - last["f"] >= 0.01) or \
                            now - last["t"] > 0.5:
                        last.update(t=now, f=f if f is not None else last["f"], s=d.get("status"))
                        job.emit("progress", call_id=None, fraction=d.get("fraction"), message=d.get("status", ""),
                                 completed=d.get("completed"), total=d.get("total"))
                rt.llm().pull(name, prog, job.cancelled)
                job.emit("done", text=f"Pulled {name}.")
            return rt.jobs.start("model_pull", work, meta={"model": name}, heavy=True).id
        return self._gate("model_pull", "model_download", "model_pull", {"name": name},
                          f"Download the model {name} from the Ollama registry (uses the network; the files are "
                          "saved where Ollama keeps its models)", start, ("network",))

    def model_import(self, body: dict) -> dict:
        rt = self.rt
        name = _model_name(body.get("name"))
        raw = body.get("gguf_path")
        if not isinstance(raw, str) or not raw:
            raise HttpError(400, "gguf_path is required")
        path = Path(raw).expanduser().resolve()
        if path.suffix.lower() != ".gguf" or not path.is_file():
            raise HttpError(400, f"{raw} is not a .gguf file")
        if self.studio is not None and not any(path == a or a in path.parents for a in self.studio.allowed):
            raise HttpError(403, "the GGUF file is outside the folders you granted to JEPA Studio")
        size = path.stat().st_size
        max_mb = rt.settings.resolved()["limits"]["max_file_mb"]
        extra = ("big_file",) if size > (max_mb << 20) else ()

        def start():
            def work(job):
                job.emit("start", model=name, action="import")
                job.emit("progress", call_id=None, fraction=0.0, message="computing SHA-256")
                digest = sha256_file(path)  # wsfs-ok: a GGUF you chose in a folder granted to JEPA Studio
                job.emit("progress", call_id=None, fraction=0.3, message=f"sha256 {digest[:16]}..., uploading")
                rt.llm().create_from_gguf(name, path, lambda d: job.emit("progress", call_id=None, fraction=None,
                                                                         message=d.get("status", "")),
                                          job.cancelled, sha256=digest)
                rt.store.record_model(name, str(path), digest, size)
                job.emit("done", text=f"Imported {path.name} as {name} (sha256 {digest}).", sha256=digest)
            return rt.jobs.start("model_import", work, meta={"model": name}, heavy=True).id
        return self._gate("model_import", "model_import", "model_import",
                          {"name": name, "gguf_path": str(path), "bytes": size},
                          f"Import {path} ({size / 2**30:.2f} GB) into Ollama as {name}", start, extra)

    def model_delete(self, name: str) -> dict:
        name = _model_name(name)
        rt = self.rt

        def start():
            def work(job):
                job.emit("start", model=name, action="delete")
                rt.llm().delete(name)
                job.emit("done", text=f"Deleted {name} from Ollama.")
            return rt.jobs.start("model_delete", work, meta={"model": name}).id
        return self._gate("model_delete", "model_delete", "model_delete", {"name": name},
                          f"Delete the model {name} from Ollama (its files are removed from disk)", start, ("delete",))

    # ------------------------------------------------------------------ conversations
    def conversation_create(self, body: dict) -> dict:
        s = self.rt.settings.resolved()
        model = body.get("model") or s["default_model"]
        title = body.get("title") or "New conversation"
        if not isinstance(title, str) or not isinstance(model, str):
            raise HttpError(400, "title and model must be text")
        _model_name(model)
        c = self.rt.store.create_conversation(title[:120], model)
        return {"id": c["id"], **c}

    def conversation_delete(self, cid: str) -> dict:
        rt = self.rt
        c = rt.store.get_conversation(cid)
        if c is None:
            raise HttpError(404, "unknown conversation")

        def start():
            tid = rt.active.get(cid)
            if tid:
                rt.cancel(tid)
                rt.jobs.wait(tid, 10)
            rt.store.delete_conversation(cid)
            return None
        n_files = len(rt.store.list_artifacts(cid))
        return self._gate("conversation_delete", "delete", "conversation_delete", {"conversation_id": cid,
                                                                                    "title": c["title"]},
                          f"Delete the conversation \"{c['title']}\", its messages and its workspace folder "
                          f"({n_files} files)", start)

    def export_all(self) -> FileResponse:
        rt = self.rt
        out = rt.store.export_all(Path(rt.settings.resolved()["data_dir"]) / "exports",
                                  {"file": rt.settings.file, "effective": rt.settings.public(), "paths": self.paths()},
                                  prune=True)
        return FileResponse(out, "application/zip", out.name, "attachment")

    def data_delete(self, body: dict) -> dict:
        """Delete all conversations, workspaces, science outputs, exports, imported-model records and (unless
        keep_audit_log) the audit log. Asks first (the `delete` row). settings.json and the models in Ollama
        stay: settings hold your approval rules, which no API may reset."""
        rt = self.rt
        keep = (body or {}).get("keep_audit_log", False)
        if not isinstance(keep, bool):
            raise HttpError(400, "keep_audit_log must be true or false")
        before = time.time()
        n_conv = len(rt.store.list_conversations())

        def start():
            for tid in list(rt.active.values()):
                rt.cancel(tid)
                rt.jobs.wait(tid, 10)
            for j in list(rt.jobs.jobs.values()):
                if j.kind.startswith("science_") and j.status in ("queued", "running", "waiting_approval"):
                    rt.jobs.cancel(j.id)
                    rt.jobs.wait(j.id, 10)
            rt.store.delete_all(keep_audit_log=keep, before=before)
            exports = Path(rt.settings.resolved()["data_dir"]) / "exports"
            if exports.is_dir() and not exports.is_symlink():
                shutil.rmtree(exports, ignore_errors=True)  # wsfs-ok: jev-data/exports, backend-owned
            return None
        what = (f"Delete all of JEV's data: {n_conv} conversation(s) with their messages and workspace files, "
                "science renders, exports, imported-model records" + ("" if keep else " and the audit log") +
                ". settings.json and the models in Ollama are kept. This cannot be undone")
        return self._gate("data_delete", "delete", "data_delete",
                          {"keep_audit_log": keep, "conversations": n_conv,
                           "data_dir": str(Path(rt.settings.resolved()["data_dir"]))}, what, start)

    def conversation_export(self, cid: str) -> FileResponse:
        out = self.rt.store.export_conversation(cid, Path(self.rt.settings.resolved()["data_dir"]) / "exports",
                                                prune=True)
        return FileResponse(out, "application/zip", out.name, "attachment")

    def add_file(self, cid: str, body: dict) -> dict:
        """Put a file into a conversation's workspace: small uploads as base64 (<= ~1.5 MB, body cap) or a
        copy of a file inside a folder you granted to JEPA Studio (`from_path`)."""
        rt = self.rt
        ws = rt.store.workspace(cid)
        name = body.get("name")
        if not isinstance(name, str) or not name or (why := check_path(name, ws)):
            raise HttpError(400, f"name must be a relative path inside the workspace ({name!r}"
                                 f"{': ' + why if isinstance(name, str) and name else ''})")
        from . import wsfs
        from .tools.base import NotRegularFile, open_regular

        dest = (ws / name.replace("\\", "/")).resolve()
        if dest.relative_to(ws).parts[0] in (".jev", ".venv"):
            raise HttpError(400, "that folder is reserved")
        b64 = body.get("content_base64", body.get("data_base64"))
        try:
            if b64 is not None:
                try:
                    data = base64.b64decode(b64, validate=True)
                except (ValueError, TypeError) as e:
                    raise HttpError(400, "content_base64 is not valid base64") from e
                wsfs.write_bytes(ws, dest, data)         # no link followed, a FIFO is replaced, never opened
            elif "from_path" in body:
                src = Path(str(body["from_path"])).expanduser().resolve()
                allowed = self.studio.allowed if self.studio is not None else []
                if not any(src == a or a in src.parents for a in allowed):
                    raise HttpError(403, "from_path is outside the folders you granted to JEPA Studio")
                if not src.is_file():
                    raise HttpError(404, "from_path is not a file")
                if src.stat().st_size > rt.settings.resolved()["limits"]["max_file_mb"] << 20:
                    raise HttpError(413, "file larger than limits.max_file_mb")
                with open_regular(src, "rb") as fi, wsfs.atomic_writer(ws, dest) as fo:
                    shutil.copyfileobj(fi, fo, 1 << 20)
            else:
                raise HttpError(400, "send content_base64 (or data_base64) or from_path")
        except (NotRegularFile, wsfs.UnsafePath) as e:
            raise HttpError(400, str(e)) from e
        from .tools.core import artifact_kind
        k = artifact_kind(dest)
        row = rt.store.add_artifact(ws, dest, k if k != "other" else "data",
                                    __import__("mimetypes").guess_type(dest.name)[0] or "application/octet-stream",
                                    dest.name, {"uploaded": True}, cid)
        return {"artifact": artifact_json(row), "path": row["path"], "name": dest.name, "bytes": row["bytes"]}

    # ------------------------------------------------------------------ artifacts
    def artifact_file(self, aid: str) -> FileResponse:
        try:
            a, path = self.rt.store.artifact_file(aid)
        except KeyError as e:
            raise HttpError(404, "unknown artifact") from e
        mime = (a.get("mime") or "application/octet-stream").split(";")[0].strip().lower()
        if mime in ACTIVE_TYPES:
            mime = "text/plain; charset=utf-8"
        elif mime.startswith("text/"):
            mime += "; charset=utf-8"
        return FileResponse(path, mime, path.name)

    # ------------------------------------------------------------------ science
    def science(self, what: str, body: dict) -> dict:
        rt = self.rt
        try:
            if what == "blackhole":
                from .science import blackhole as mod
            else:
                from .science import sims as mod
        except ImportError as e:
            raise HttpError(501, f"the science module is not installed in this build ({e})") from e
        w, h = body.get("width", 0), body.get("height", 0)
        if isinstance(w, int) and isinstance(h, int) and w * h > MAX_PIXELS:
            raise HttpError(400, f"{w}x{h} is larger than 8K")
        frames = body.get("frames")
        if isinstance(frames, int) and frames > MAX_FRAMES:
            raise HttpError(400, f"at most {MAX_FRAMES} frames")
        if what == "simulate" and not isinstance(body.get("kind"), str):
            raise HttpError(400, "kind is required (nbody, pendulum, wave, ...)")
        root = Path(rt.settings.resolved()["workspace_root"])
        seconds = None
        est_fn = getattr(mod, "estimate_seconds", None)
        if est_fn is not None:
            try:
                est = est_fn(dict(body)) if what == "blackhole" else est_fn(body["kind"], dict(body.get("params") or {}))
                seconds = float(est["seconds"] if isinstance(est, dict) else est)
            except Exception:  # noqa: BLE001 - an estimate is advisory
                seconds = None

        def work(job):
            out = root / "_science" / job.id
            out.mkdir(parents=True, exist_ok=True)  # wsfs-ok: backend-owned workspace_root/_science job folder
            job.emit("start", action=what, kind=body.get("kind"))

            def prog(f, m=""):
                job.emit("progress", call_id=None, fraction=None if f is None else round(float(f), 4), message=m)
            if what == "blackhole":
                paths = mod.render(dict(body), out, prog, job.cancelled)
            else:
                paths = mod.run(body["kind"], dict(body.get("params") or {}), out, prog, job.cancelled)
            if job.cancelled():
                job.emit("cancelled")
                job.status = "cancelled"
                return
            from .tools.core import artifact_kind
            res = ToolResult(True, "", artifacts=[])
            for pth in paths or []:
                pth = Path(pth).resolve()
                if out.resolve() in pth.parents and pth.is_file():
                    import mimetypes
                    from .tools.base import Artifact
                    res.artifacts.append(Artifact(pth, artifact_kind(pth) if artifact_kind(pth) != "other" else "data",
                                                  mimetypes.guess_type(pth.name)[0] or "application/octet-stream",
                                                  pth.name))
            arts = register_artifacts(rt, job, res, out, None, None, None)
            job.emit("done", text=f"{len(arts)} file(s) ready.", artifacts=arts)
        def start():
            return rt.jobs.start(f"science_{what}", work, meta={"kind": body.get("kind")}, heavy=True).id
        if seconds is not None and seconds > GPU_MAX_MINUTES * 60:
            args = {k: body.get(k) for k in ("kind", "width", "height", "frames", "seconds", "spin") if k in body}
            reason = (f"This render is estimated to take about {seconds / 60:.0f} minutes of CPU/GPU time "
                      f"(over {GPU_MAX_MINUTES})")
            out = self._gate("science", "gpu_long", f"science_{what}", args, reason, start)
            return {**out, "reason": reason, "args": args, "estimate_s": round(seconds, 1)}
        return {"job_id": start(), "estimate_s": None if seconds is None else round(seconds, 1)}


_INSTANCES: dict = {}
_INSTANCES_LOCK = threading.Lock()


def get_api(root: str | Path, studio=None) -> JevApi:
    """One JevApi per backend root (created on the first /api/jev request)."""
    key = str(Path(root).resolve())
    with _INSTANCES_LOCK:
        if key not in _INSTANCES:
            _INSTANCES[key] = JevApi(root, studio=studio)
        return _INSTANCES[key]


__all__ = ["JevApi", "HttpError", "FileResponse", "get_api"]
