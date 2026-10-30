"""JEV settings: one JSON file you own, `<root>/jev-data/settings.json` (mode 0600).

Every value has a safe default, so the file only needs the keys you change. Environment variables
override the file (useful for a one-off session or a locked-down machine):

    JEV_OLLAMA_URL            ollama_url                   (loopback only unless allow_remote_ollama)
    JEV_ALLOW_REMOTE_OLLAMA   allow_remote_ollama          ("1" to allow a non-loopback Ollama)
    JEV_DEFAULT_MODEL         default_model
    JEV_WORKSPACE_ROOT        workspace_root
    JEV_NETWORK_ALLOWED       network_allowed              ("1" lets sandboxed code use the network
                                                            without asking; default off)

Why `jev-data/` and not `jev/`: in a source checkout the backend root is the repository itself,
where `jev/` is this Python package. Keeping user data in its own folder means your
conversations never end up inside the code tree (and are easy to back up or delete).

The HTTP API can change only `default_model`, `limits`, `network_allowed` and `ollama_url`
(POST /api/jev/settings; every change is written to the audit log with the values before and
after). Approval rules (`approval_rules`), `allow_remote_ollama` and `allow_cloud_models` can be
changed only by editing this file, so no web page and no model output can loosen them. The file
is validated on every load: unknown keys, unknown approval-table rows and invalid values are
refused, and `approval_rules` can only tighten the locked rows (jev/policy.py), never loosen
them. Sandboxed code cannot write this file wherever the OS gives JEV a mount namespace
(bubblewrap, unshare); see docs/jev/SAFETY.md.
"""
from __future__ import annotations

import copy
import ipaddress
import json
import os
import re
import stat
import threading
from pathlib import Path
from urllib.parse import urlparse

from jepa_studio.config import deep_merge, validate

DATA_DIRNAME = "jev-data"

DEFAULTS: dict = {
    "ollama_url": "http://127.0.0.1:11434",
    "allow_remote_ollama": False,
    "allow_cloud_models": False,        # Ollama "-cloud" models run on ollama.com, not here
    "default_model": "qwen2.5-coder:7b",
    "workspace_root": None,             # None -> <root>/jev-data/workspaces
    "limits": {
        "cpu_seconds": 120,             # CPU time per code run (RLIMIT_CPU)
        "memory_mb": 2048,              # memory per code run (RLIMIT_DATA on Linux, RLIMIT_AS elsewhere)
        "wall_seconds": 300,            # wall clock per code run; the whole process tree is killed after
        "max_output_mb": 200,           # largest single file a code run may write (RLIMIT_FSIZE)
        "max_file_mb": 500,             # files bigger than this need approval (writing, importing)
        "max_workspace_mb": 2048,       # total size of one conversation's workspace (checked around every run)
    },
    "network_allowed": False,           # True: sandboxed code may use the network without asking
    "approval_rules": {},               # {"<rule>": "run"|"ask"|"deny"} or {"tool:<name>": ...}; see policy.py
    "approval_timeout_s": 1800,         # an unanswered approval is treated as "deny" after this
    "max_tool_rounds": 12,
    "sandbox": {"isolation": "auto", "hide_home": True},  # auto | bwrap | unshare | none
}

SCHEMA: dict = {
    "type": "object",
    "additionalProperties": False,
    "properties": {
        "ollama_url": {"type": "string", "maxLength": 300, "pattern": r"^https?://"},
        "allow_remote_ollama": {"type": "boolean"},
        "allow_cloud_models": {"type": "boolean"},
        "default_model": {"type": "string", "maxLength": 200, "pattern": r"^[A-Za-z0-9][A-Za-z0-9._:/@+-]*$"},
        "workspace_root": {"type": ["string", "null"], "maxLength": 1000},
        "limits": {
            "type": "object",
            "additionalProperties": False,
            "properties": {
                # bounds are the "absurd limits" guard: outside them a settings change is refused
                "cpu_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
                "memory_mb": {"type": "integer", "minimum": 64, "maximum": 1048576},
                "wall_seconds": {"type": "integer", "minimum": 1, "maximum": 86400},
                "max_output_mb": {"type": "integer", "minimum": 1, "maximum": 1048576},
                "max_file_mb": {"type": "integer", "minimum": 1, "maximum": 1048576},
                "max_workspace_mb": {"type": "integer", "minimum": 16, "maximum": 1048576},
            },
        },
        "network_allowed": {"type": "boolean"},
        "approval_rules": {"type": "object"},
        "approval_timeout_s": {"type": "integer", "minimum": 1, "maximum": 86400},
        "max_tool_rounds": {"type": "integer", "minimum": 1, "maximum": 50},
        "sandbox": {
            "type": "object",
            "additionalProperties": False,
            "properties": {"isolation": {"enum": ["auto", "bwrap", "unshare", "sandbox-exec", "none"]},
                           "hide_home": {"type": "boolean"}},
        },
    },
}

API_WRITABLE = ("default_model", "limits", "network_allowed", "ollama_url")
RULE_ACTIONS = ("run", "ask", "deny")


class SettingsError(ValueError):
    pass


def is_loopback_url(url: str) -> bool:
    host = (urlparse(url).hostname or "").strip("[]").lower()
    if host == "localhost":
        return True
    try:
        return ipaddress.ip_address(host).is_loopback
    except ValueError:
        return False


def check_ollama_url(url: str, allow_remote: bool) -> str:
    u = urlparse(url)
    if u.scheme not in ("http", "https") or not u.hostname:
        raise SettingsError(f"ollama_url must look like http://127.0.0.1:11434 (got {url!r})")
    if not allow_remote and not is_loopback_url(url):
        raise SettingsError(
            f"ollama_url {url!r} is not on this machine. JEV only talks to a local Ollama "
            "(127.0.0.1, ::1 or localhost) so your prompts never leave the computer. If you really "
            "run Ollama on another machine you control, set \"allow_remote_ollama\": true in "
            "jev-data/settings.json (or JEV_ALLOW_REMOTE_OLLAMA=1).")
    return url.rstrip("/")


TOOL_RULE = re.compile(r"^tool:[A-Za-z0-9_]{1,64}$")


def _check(cfg: dict) -> None:
    from .policy import RULES

    errs = validate(cfg, SCHEMA)
    for k, v in (cfg.get("approval_rules") or {}).items():
        if v not in RULE_ACTIONS:
            errs.append(f"$.approval_rules.{k}: must be one of {RULE_ACTIONS}")
        if not isinstance(k, str) or not (k in RULES or TOOL_RULE.match(k)):
            errs.append(f"$.approval_rules.{k}: not a row of the approval table (see jev/policy.py) "
                        "or a \"tool:<name>\" entry")
    if errs:
        raise SettingsError("invalid JEV settings: " + "; ".join(errs[:10]))
    check_ollama_url(cfg["ollama_url"], cfg["allow_remote_ollama"])


def private_dir(path: str | Path) -> Path:
    """mkdir -p with mode 0700 (and tighten an existing folder JEV owns to 0700): JEV's data,
    workspaces and exports are readable by you only."""
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True, mode=0o700)  # wsfs-ok: jev-data folders
    if os.name != "nt":
        try:
            if not p.is_symlink() and stat.S_IMODE(p.stat().st_mode) & 0o077:
                os.chmod(p, 0o700)
        except OSError:
            pass
    return p


def _write_private(path: Path, text: str) -> None:
    private_dir(path.parent)
    tmp = path.with_name(path.name + ".tmp")
    fd = os.open(tmp, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)  # wsfs-ok: jev-data/settings.json
    with os.fdopen(fd, "w", encoding="utf-8") as f:
        f.write(text)
    os.replace(tmp, path)  # wsfs-ok: jev-data/settings.json
    try:
        os.chmod(path, 0o600)
    except OSError:
        pass


class Settings:
    """Loads, validates and saves settings.json. `resolved()` returns a fresh dict each call."""

    def __init__(self, root: str | Path, env: dict | None = None):
        self.root = Path(root).resolve()
        self.dir = self.root / DATA_DIRNAME
        self.path = self.dir / "settings.json"
        self.env = os.environ if env is None else env
        self.lock = threading.Lock()
        self.file: dict = {}
        self.load()

    def load(self) -> None:
        if self.path.exists():
            raw = self.path.read_text(encoding="utf-8")  # wsfs-ok: jev-data/settings.json
            if len(raw) > 1_000_000:
                raise SettingsError("settings.json larger than 1 MB; refusing to parse")
            data = json.loads(raw or "{}")
            if not isinstance(data, dict):
                raise SettingsError("settings.json must hold a JSON object")
            self.file = data
        merged = deep_merge(DEFAULTS, self.file)
        _check(self._apply_env(merged))

    def _apply_env(self, cfg: dict) -> dict:
        e = self.env
        if e.get("JEV_ALLOW_REMOTE_OLLAMA"):
            cfg["allow_remote_ollama"] = e["JEV_ALLOW_REMOTE_OLLAMA"] == "1"
        if e.get("JEV_OLLAMA_URL"):
            cfg["ollama_url"] = e["JEV_OLLAMA_URL"]
        if e.get("JEV_DEFAULT_MODEL"):
            cfg["default_model"] = e["JEV_DEFAULT_MODEL"]
        if e.get("JEV_WORKSPACE_ROOT"):
            cfg["workspace_root"] = e["JEV_WORKSPACE_ROOT"]
        if e.get("JEV_NETWORK_ALLOWED"):
            cfg["network_allowed"] = e["JEV_NETWORK_ALLOWED"] == "1"
        return cfg

    def resolved(self) -> dict:
        with self.lock:
            cfg = self._apply_env(deep_merge(DEFAULTS, self.file))
        cfg["ollama_url"] = check_ollama_url(cfg["ollama_url"], cfg["allow_remote_ollama"])
        ws = cfg.get("workspace_root")
        cfg["workspace_root"] = str(Path(ws).expanduser().resolve() if ws else self.dir / "workspaces")
        cfg["data_dir"] = str(self.dir)
        cfg["root"] = str(self.root)
        return cfg

    @property
    def workspace_root(self) -> Path:
        p = Path(self.resolved()["workspace_root"])
        p.mkdir(parents=True, exist_ok=True)  # wsfs-ok: workspace_root (jev-data)
        return p

    def update(self, patch: dict, api: bool = True) -> dict:
        """Merge `patch` into the file. With api=True only API_WRITABLE keys are accepted."""
        if not isinstance(patch, dict):
            raise SettingsError("settings patch must be an object")
        if api:
            bad = [k for k in patch if k not in API_WRITABLE]
            if bad:
                raise SettingsError(f"these settings can only be changed by editing {self.path}: {', '.join(bad)}")
        with self.lock:
            new_file = deep_merge(self.file, patch)
            effective = self._apply_env(deep_merge(DEFAULTS, new_file))
            _check(effective)
            # the file's own URL must be acceptable too (an env override may be hiding it today)
            check_ollama_url(deep_merge(DEFAULTS, new_file)["ollama_url"], effective["allow_remote_ollama"])
            _write_private(self.path, json.dumps(new_file, indent=2) + "\n")
            self.file = new_file
        return self.public()

    def public(self) -> dict:
        """What the UI sees (no secrets exist in settings, but keep the shape stable)."""
        r = self.resolved()
        return {k: copy.deepcopy(r[k]) for k in (
            "ollama_url", "allow_remote_ollama", "allow_cloud_models", "default_model", "workspace_root",
            "limits", "network_allowed", "approval_rules", "approval_timeout_s", "max_tool_rounds", "sandbox")}
