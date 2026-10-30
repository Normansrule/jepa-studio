"""The safety net: decides for every tool call whether it runs now, asks you first, or is refused.

====================================================================================================
 APPROVAL TABLE  (this is the one place to change what JEV may do on its own)
====================================================================================================
 Actions:  "run"  = runs immediately (still inside the sandbox, within the limits in settings)
           "ask"  = the turn stops and waits for you (Approve / Deny in the app, y/N in `jev chat`);
                    no answer within approval_timeout_s (default 30 min) counts as Deny
           "deny" = refused; the model is told why

 Without editing code you can override any unlocked row in jev-data/settings.json:
     "approval_rules": {"shell": "deny", "network": "run", "tool:run_python": "ask"}
 A "tool:<name>" entry overrides the final decision for that tool (except the locked rows).
 The HTTP API cannot change approval_rules, and no tool can, so neither a web page nor the
 model can loosen them.
"""
from __future__ import annotations

import ast
import json
import re
import secrets
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

RULES: dict[str, tuple[str, str]] = {
    # rule key             action   what it covers
    "read":               ("run",  "tools with no side effects: answering, math, listing/reading workspace files"),
    "sandbox":            ("run",  "writing and running code inside this conversation's workspace, within limits"),
    "approve_tool":       ("ask",  "tools declared risk='approve' (shell, installs, training runs, ...)"),
    # ---- the "important stuff": always asks, even though sandboxed work auto-runs ----------------
    "network":            ("ask",  "network access from code or a tool (unless settings.network_allowed is on)"),
    "outside_workspace":  ("ask",  "code that reads or writes files outside the workspace (absolute paths, ~, ..)"),
    "delete":             ("ask",  "deleting anything: files, folders, conversations"),
    "install":            ("ask",  "installing packages (pip, npm, apt, conda, brew, winget, ...)"),
    "model_download":     ("ask",  "downloading a model (ollama pull)"),
    "model_import":       ("ask",  "importing a GGUF model file into Ollama"),
    "model_delete":       ("ask",  "deleting a model from Ollama"),
    "gpu_long":           ("ask",  "GPU jobs estimated over 10 minutes or over 80% of VRAM"),
    "big_file":           ("ask",  "files larger than limits.max_file_mb"),
    "shell":              ("ask",  "running shell commands (bash, sh, PowerShell, cmd, subprocess, os.system)"),
    "send_data":          ("ask",  "anything that sends data off this machine (uploads, URLs, email, publishing)"),
    "dynamic_code":       ("ask",  "exec/eval/__import__ tricks, only when the OS gives no network isolation"),
    "tool_specific":      ("ask",  "a tool's own reason to ask (its needs_approval hook returned a reason)"),
    # ---- locked: refused whatever settings say ------------------------------------------------------
    "path_escape":        ("deny", "workspace-path arguments escaping via '..', symlinks, absolute, ~ or drive paths"),
    "self_approval":      ("deny", "code or arguments touching JEV's approvals API, session token, settings or audit DB"),
    "absurd_limits":      ("deny", "arguments asking for more time or memory than the configured limits"),
}
LOCKED = {"path_escape", "self_approval", "absurd_limits"}
GPU_MAX_MINUTES = 10           # gpu_long: estimated minutes above this ask
GPU_MAX_VRAM_FRAC = 0.80       # gpu_long: estimated fraction of VRAM above this asks
# never remembered for the session, even if you tick "remember": each one is asked every time
NEVER_REMEMBER = {"delete", "model_download", "model_import", "model_delete", "send_data"}
# ====================================================================================================

SEVERITY = {"run": 0, "ask": 1, "deny": 2}
PATH_KEYS = {"path", "paths", "file", "files", "filename", "dir", "directory", "folder", "src", "dst", "source",
             "target", "out", "output", "out_dir", "attachments", "image", "images", "video", "audio", "cwd"}
LIMIT_KEYS = {"timeout_s": "wall_seconds", "wall_seconds": "wall_seconds", "cpu_seconds": "cpu_seconds",
              "memory_mb": "memory_mb"}
SELF_PATTERNS = [re.compile(p, re.I) for p in (
    r"/api/jev\b", r"X-JEPA-Token", r"JEPA_STUDIO_TOKEN", r"jev-data", r"\bjev\.db\b",
    r"/proc/[^\s'\"]*/environ", r"approval_rules", r"JEPA_STUDIO_ALLOWED")]
TAG = re.compile(r"^\[([a-z_]+)\]\s*(.*)$")


def flag(rule: str, text: str) -> str:
    """Format a needs_approval reason so policy knows which table row it belongs to."""
    return f"[{rule}] {text}"


def gpu_reason(est_minutes: float | None = None, vram_frac: float | None = None) -> str | None:
    """Helper for GPU tools' needs_approval hooks."""
    if est_minutes is not None and est_minutes > GPU_MAX_MINUTES:
        return flag("gpu_long", f"estimated {est_minutes:.0f} min on the GPU (over {GPU_MAX_MINUTES})")
    if vram_frac is not None and vram_frac > GPU_MAX_VRAM_FRAC:
        return flag("gpu_long", f"needs about {vram_frac:.0%} of GPU memory (over {GPU_MAX_VRAM_FRAC:.0%})")
    return None


@dataclass
class Decision:
    action: str                      # run | ask | deny
    reason: str
    rules: list[str] = field(default_factory=list)
    by: str = "policy"               # policy | session | settings

    def to_json(self) -> dict:
        return {"action": self.action, "reason": self.reason, "rules": self.rules, "by": self.by}


# ------------------------------------------------------------------------------------------ paths
def check_path(value: Any, workspace: Path) -> str | None:
    """Reason string if `value` (a model-supplied path) is not safely inside `workspace`."""
    if not isinstance(value, str):
        return None
    if "\x00" in value:
        return "path contains a NUL byte"
    v = value.replace("\\", "/")
    if re.match(r"^[A-Za-z]:", v) or v.startswith("//"):
        return f"{value!r} is a Windows drive or network path outside the workspace"
    if v.startswith("~"):
        return f"{value!r} points into a home folder, outside the workspace"
    ws = workspace.resolve()
    p = Path(v)
    target = p.resolve() if p.is_absolute() else (ws / p).resolve()   # resolve() follows symlinks
    if target != ws and ws not in target.parents:
        how = "a symlink" if not any(part == ".." for part in p.parts) and not p.is_absolute() else \
            ("an absolute path" if p.is_absolute() else "'..'")
        return f"{value!r} escapes the workspace via {how}"
    return None


def path_args(params: dict, args: dict):
    props = (params or {}).get("properties", {})
    for k, v in (args or {}).items():
        schema = props.get(k, {})
        is_path = k in PATH_KEYS or k.endswith(("_path", "_file", "_dir", "_paths", "_files")) or \
            schema.get("format") == "path" or (schema.get("items") or {}).get("format") == "path"
        if not is_path:
            continue
        for item in (v if isinstance(v, list) else [v]):
            if isinstance(item, str):
                yield k, item


def _strings(obj, depth=0):
    if depth > 6:
        return
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for k, v in obj.items():
            yield from _strings(k, depth + 1)
            yield from _strings(v, depth + 1)
    elif isinstance(obj, list):
        for v in obj:
            yield from _strings(v, depth + 1)


# ------------------------------------------------------------------------------------------ code scanners
NET_MODULES = {"socket", "requests", "urllib", "urllib2", "urllib3", "http", "httpx", "aiohttp", "ftplib", "smtplib",
               "poplib", "imaplib", "telnetlib", "paramiko", "websocket", "websockets", "ssl", "xmlrpc",
               "pycurl", "grpc", "boto3", "botocore", "socketserver", "asyncssh", "webbrowser"}
SHELL_MODULES = {"subprocess", "pty", "pexpect", "sh", "plumbum"}
SHELL_CALLS = {"system", "popen", "spawnl", "spawnlp", "spawnv", "spawnvp", "execv", "execvp", "execl", "execlp",
               "startfile", "fork", "forkpty"}
DELETE_CALLS = {"remove", "unlink", "rmdir", "rmtree", "removedirs", "truncate"}
INSTALL_RE = re.compile(r"\b(pip3?|conda|mamba|npm|pnpm|yarn|apt(-get)?|brew|winget|choco|dnf|yum|pacman|gem|cargo)"
                        r"['\"]?[\s,]+['\"]?(install|add|i)\b|ensurepip|get-pip\.py|\bnpx\s", re.I)
ABS_PATH_RE = re.compile(r"^(/(?!/)[A-Za-z0-9._-]+(/|$)|[A-Za-z]:[\\/]|\\\\|~[/\\]|~$)")
DOTDOT_RE = re.compile(r"(^|[\\/])\.\.([\\/]|$)")
URL_RE = re.compile(r"\b(https?|ftp|wss?)://", re.I)
NODE_NET_RE = re.compile(r"require\(\s*['\"](node:)?(http|https|net|dgram|tls|dns|http2)['\"]\s*\)|"
                         r"\bfetch\s*\(|\bWebSocket\b|\bXMLHttpRequest\b|from\s+['\"](node:)?(http|https|net|dgram|tls)['\"]")
NODE_SHELL_RE = re.compile(r"child_process|\bexecSync\b|\bspawnSync\b|process\.binding")
NODE_DELETE_RE = re.compile(r"\b(unlinkSync|unlink|rmSync|rmdirSync|rm|rmdir)\s*\(")
SHELL_NET_RE = re.compile(r"\b(curl|wget|ssh|scp|sftp|rsync|nc|ncat|netcat|telnet|ftp|git\s+(clone|pull|push|fetch)|"
                          r"Invoke-WebRequest|Invoke-RestMethod|iwr|irm|Start-BitsTransfer)\b", re.I)
SHELL_DELETE_RE = re.compile(r"(^|[\s;&|(])(rm|rmdir|del|erase|rd|shred|Remove-Item|unlink)(\s|$)", re.I)


def _self_hits(text: str) -> list[tuple[str, str]]:
    return [("self_approval", f"refers to {m.group(0)!r}") for p in SELF_PATTERNS if (m := p.search(text))][:1]


def _path_literal_hits(strings) -> list[tuple[str, str]]:
    for s in strings:
        s2 = s.strip()
        if len(s2) > 400 or "\n" in s2:
            continue
        if ABS_PATH_RE.match(s2) or DOTDOT_RE.search(s2):
            return [("outside_workspace", f"uses the path {s2[:80]!r}, outside the workspace")]
    return []


def scan_python(code: str, isolated: bool = True) -> list[tuple[str, str]]:
    """(rule, reason) pairs for Python source. A heuristic: the sandbox is the real boundary."""
    hits = _self_hits(code)
    if INSTALL_RE.search(code):
        hits.append(("install", "installs packages"))
    try:
        tree = ast.parse(code)
    except SyntaxError:
        tree = None
    mods, calls, consts = set(), set(), []
    if tree is not None:
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                mods |= {a.name.split(".")[0] for a in node.names}
            elif isinstance(node, ast.ImportFrom) and node.module:
                mods.add(node.module.split(".")[0])
            elif isinstance(node, ast.Call):
                f = node.func
                calls.add(f.attr if isinstance(f, ast.Attribute) else getattr(f, "id", ""))
            elif isinstance(node, ast.Constant) and isinstance(node.value, str):
                consts.append(node.value)
    else:
        mods |= set(re.findall(r"^\s*(?:import|from)\s+([A-Za-z_]\w*)", code, re.M))
        consts = re.findall(r"['\"]([^'\"\n]{1,300})['\"]", code)
    if mods & NET_MODULES or any(URL_RE.search(c) for c in consts):
        hits.append(("network", "uses the network (" + ", ".join(sorted(mods & NET_MODULES)[:4] or ["a URL"]) + ")"))
    if mods & SHELL_MODULES or calls & SHELL_CALLS:
        hits.append(("shell", "starts other programs (" + ", ".join(sorted((mods & SHELL_MODULES) |
                                                                           (calls & SHELL_CALLS))[:4]) + ")"))
    if calls & DELETE_CALLS:
        hits.append(("delete", "deletes files (" + ", ".join(sorted(calls & DELETE_CALLS)[:3]) + ")"))
    hits += _path_literal_hits(consts)
    if not isolated and ({"exec", "eval", "compile", "__import__"} & calls or "importlib" in mods or "ctypes" in mods):
        hits.append(("dynamic_code", "builds code at run time (exec/eval/__import__/importlib/ctypes) and this OS "
                                     "has no network isolation"))
    return hits


def scan_node(code: str, isolated: bool = True) -> list[tuple[str, str]]:
    hits = _self_hits(code)
    if INSTALL_RE.search(code):
        hits.append(("install", "installs packages"))
    if NODE_NET_RE.search(code) or URL_RE.search(code):
        hits.append(("network", "uses the network"))
    if NODE_SHELL_RE.search(code):
        hits.append(("shell", "starts other programs (child_process)"))
    if NODE_DELETE_RE.search(code):
        hits.append(("delete", "deletes files"))
    hits += _path_literal_hits(re.findall(r"['\"`]([^'\"`\n]{1,300})['\"`]", code))
    if not isolated and re.search(r"\beval\s*\(|new\s+Function\s*\(|\bvm\.run", code):
        hits.append(("dynamic_code", "builds code at run time and this OS has no network isolation"))
    return hits


def scan_shell(command: str) -> list[tuple[str, str]]:
    hits = _self_hits(command)
    if INSTALL_RE.search(command):
        hits.append(("install", "installs packages"))
    if SHELL_NET_RE.search(command) or URL_RE.search(command):
        hits.append(("network", "uses the network"))
    if SHELL_DELETE_RE.search(command):
        hits.append(("delete", "deletes files"))
    toks = re.findall(r"[^\s;&|<>()'\"]+", command)
    hits += _path_literal_hits(toks)
    return hits


# ------------------------------------------------------------------------------------------ policy
class Policy:
    def __init__(self, settings: dict | None = None):
        self.settings = settings or {}
        self.remembered: set[tuple] = set()
        self.lock = threading.Lock()

    def action_for(self, rule: str) -> str:
        base = RULES.get(rule, RULES["tool_specific"])[0]
        if rule in LOCKED:
            return base
        if rule == "network" and self.settings.get("network_allowed"):
            base = "run"
        return (self.settings.get("approval_rules") or {}).get(rule, base)

    @staticmethod
    def shape(tool: str, args: dict, rules: list[str]) -> tuple:
        return (tool, tuple(sorted((args or {}).keys())), tuple(sorted(set(rules))))

    def remember(self, tool: str, args: dict, rules: list[str]) -> bool:
        if set(rules) & NEVER_REMEMBER:
            return False
        with self.lock:
            self.remembered.add(self.shape(tool, args, rules))
        return True

    def decide(self, spec, args: dict, ctx) -> Decision:
        """spec: jev.tools.base.ToolSpec; ctx: ToolContext (workspace + settings)."""
        ws = Path(ctx.workspace)
        limits = (self.settings.get("limits") or {})
        hits: list[tuple[str, str]] = []
        for k, v in path_args(spec.parameters, args):
            if r := check_path(v, ws):
                hits.append(("path_escape", f"{k}: {r}"))
        for k, v in (args or {}).items():
            if k in LIMIT_KEYS and isinstance(v, (int, float)) and limits.get(LIMIT_KEYS[k]) is not None \
                    and v > limits[LIMIT_KEYS[k]]:
                hits.append(("absurd_limits", f"{k}={v} is above the limit {limits[LIMIT_KEYS[k]]}"))
        texts = list(_strings(args))
        for t in texts:
            hits += _self_hits(t)
        max_bytes = int(limits.get("max_file_mb", 500)) << 20
        for t in texts:
            if len(t.encode("utf-8", "ignore")) > max_bytes:
                hits.append(("big_file", f"an argument is larger than {limits.get('max_file_mb')} MB"))
        for k, v in (args or {}).items():
            if k in ("url", "urls") or k.endswith("_url"):
                hits.append(("network", f"{k} points at the network"))
        risk_rule = {"read": "read", "sandbox": "sandbox", "approve": "approve_tool"}[spec.risk]
        hits.append((risk_rule, {"read": "read-only tool", "sandbox": "runs inside the workspace sandbox",
                                 "approve_tool": f"{spec.name} always asks first"}[risk_rule]))
        if spec.needs_approval and not any(r == "path_escape" for r, _ in hits):
            try:
                extra = spec.needs_approval(args, ctx)
            except Exception as e:  # noqa: BLE001 - a broken hook must not auto-run
                extra = flag("tool_specific", f"could not check this call ({type(e).__name__}: {e})")
            for line in (extra or "").splitlines():
                if line.strip():
                    m = TAG.match(line.strip())
                    hits.append((m.group(1), m.group(2)) if m and m.group(1) in RULES else ("tool_specific", line))
        # combine: the strictest row wins
        rules = [r for r, _ in hits]
        worst, reasons = "run", []
        for rule, text in hits:
            a = self.action_for(rule)
            if SEVERITY[a] > SEVERITY[worst]:
                worst, reasons = a, [text]
            elif a == worst and SEVERITY[a] > 0:
                reasons.append(text)
        locked_deny = any(r in LOCKED for r in rules)
        by = "policy"
        override = (self.settings.get("approval_rules") or {}).get(f"tool:{spec.name}")
        if override in SEVERITY and not locked_deny:
            worst, by = override, "settings"
            reasons = reasons or [f"approval_rules sets {spec.name} to {override}"]
        if worst == "ask":
            ask_rules = [r for r in rules if self.action_for(r) == "ask"]
            with self.lock:
                if self.shape(spec.name, args, ask_rules) in self.remembered:
                    return Decision("run", "approved earlier this session (remember)", rules, "session")
        reason = "; ".join(dict.fromkeys(reasons)) or hits[-1][1]
        return Decision(worst, reason, rules, by)


# ------------------------------------------------------------------------------------------ approvals
@dataclass
class Approval:
    id: str
    kind: str                       # tool | model_pull | model_import | model_delete | conversation_delete
    tool: str
    args: dict
    reason: str
    risk: str
    rules: list[str]
    conversation_id: str | None = None
    turn_id: str | None = None
    call_id: str | None = None
    created: float = field(default_factory=time.time)
    status: str = "pending"         # pending | approved | denied | expired | cancelled
    remember: str | None = None
    job_id: str | None = None
    on_approve: Callable[[], str | None] | None = None
    event: threading.Event = field(default_factory=threading.Event)

    def to_json(self) -> dict:
        return {"id": self.id, "created": self.created, "kind": self.kind, "tool": self.tool, "args": self.args,
                "reason": self.reason, "risk": self.risk, "rules": self.rules,
                "conversation_id": self.conversation_id, "turn_id": self.turn_id, "call_id": self.call_id,
                "status": self.status, "job_id": self.job_id}


class ApprovalBroker:
    """Pending approvals live here (in memory, this backend session only) and in the store's audit log."""

    def __init__(self, store, policy: Policy):
        self.store = store
        self.policy = policy
        self.pending: dict[str, Approval] = {}
        self.lock = threading.Lock()

    def record_auto(self, decision: Decision, tool: str, args: dict, risk: str, conversation_id=None, turn_id=None,
                    call_id=None) -> str:
        """Audit-log an automatic decision (run / deny / remembered)."""
        aid = "a-" + secrets.token_hex(6)
        status = {"run": "auto_run", "deny": "auto_deny"}.get(decision.action, decision.action)
        if decision.by == "session":
            status = "remembered"
        self.store.add_approval({"id": aid, "kind": "tool", "tool": tool, "args": args, "reason": decision.reason,
                                 "risk": risk, "rules": decision.rules, "conversation_id": conversation_id,
                                 "turn_id": turn_id, "call_id": call_id, "status": status,
                                 "decided_by": decision.by, "decided": time.time()})
        return aid

    def request(self, kind: str, tool: str, args: dict, reason: str, risk: str = "approve", rules=None,
                conversation_id=None, turn_id=None, call_id=None, on_approve=None) -> Approval:
        a = Approval("a-" + secrets.token_hex(6), kind, tool, args, reason, risk, list(rules or []),
                     conversation_id, turn_id, call_id, on_approve=on_approve)
        with self.lock:
            self.pending[a.id] = a
        self.store.add_approval({**a.to_json(), "decided_by": None, "decided": None})
        return a

    def get(self, aid: str) -> Approval | None:
        with self.lock:
            return self.pending.get(aid)

    def decide(self, aid: str, decision: str, remember: str | None = None, by: str = "user") -> dict:
        if decision not in ("approve", "deny"):
            raise ValueError("decision must be 'approve' or 'deny'")
        if remember not in (None, "session"):
            raise ValueError("remember must be 'session' or null")
        with self.lock:
            a = self.pending.get(aid)
            if a is None or a.status != "pending":
                row = self.store.get_approval(aid)
                if row is None:
                    raise KeyError(aid)
                return {"ok": False, "status": row["status"], "job_id": row.get("job_id"),
                        "error": f"approval is already {row['status']}"}
            a.status = "approved" if decision == "approve" else "denied"
            a.remember = remember
        remembered = False
        if a.status == "approved" and remember == "session" and a.kind == "tool":
            ask_rules = [r for r in a.rules if self.policy.action_for(r) == "ask"]
            remembered = self.policy.remember(a.tool, a.args, ask_rules)
        if a.status == "approved" and a.on_approve:
            try:
                a.job_id = a.on_approve()
            except Exception as e:  # noqa: BLE001
                a.status = "failed"
                self.store.update_approval(aid, status="failed", decided_by=by, decided=time.time(),
                                           reason=f"{a.reason} (failed to start: {e})")
                a.event.set()
                return {"ok": False, "status": "failed", "error": str(e)[:300]}
        self.store.update_approval(aid, status=a.status, decided_by=by, decided=time.time(), job_id=a.job_id,
                                   remember=remember if remembered else None)
        a.event.set()
        out = {"ok": True, "status": a.status}
        if a.job_id:
            out["job_id"] = a.job_id
        if remember == "session" and not remembered and a.status == "approved":
            out["note"] = "this kind of action is never remembered; you will be asked each time"
        return out

    def wait(self, aid: str, timeout: float, cancelled: Callable[[], bool] | None = None) -> str:
        a = self.get(aid)
        if a is None:
            return "expired"
        deadline = time.monotonic() + timeout
        while not a.event.wait(0.1):
            if cancelled and cancelled():
                self._close(a, "cancelled", "turn cancelled")
                return "cancelled"
            if time.monotonic() > deadline:
                self._close(a, "expired", "timeout")
                return "expired"
        return a.status

    def _close(self, a: Approval, status: str, by: str) -> None:
        with self.lock:
            if a.status != "pending":
                return
            a.status = status
        self.store.update_approval(a.id, status=status, decided_by=by, decided=time.time())
        a.event.set()

    def cancel_turn(self, turn_id: str) -> None:
        with self.lock:
            items = [a for a in self.pending.values() if a.turn_id == turn_id and a.status == "pending"]
        for a in items:
            self._close(a, "cancelled", "turn cancelled")

    def list(self, status: str | None = None) -> list[dict]:
        return self.store.list_approvals(status)


def describe_rules(settings: dict | None = None) -> list[dict]:
    p = Policy(settings)
    return [{"rule": k, "default": v[0], "action": p.action_for(k), "covers": v[1], "locked": k in LOCKED}
            for k, v in RULES.items()]


def dumps_args(args: Any, limit: int = 20000) -> str:
    s = json.dumps(args, default=str)
    return s if len(s) <= limit else s[:limit] + "...(truncated)"


__all__ = ["RULES", "LOCKED", "Decision", "Policy", "ApprovalBroker", "Approval", "flag", "gpu_reason", "check_path",
           "scan_python", "scan_node", "scan_shell", "describe_rules"]
