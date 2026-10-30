"""Run model-written code (and the media tools' FFmpeg) as a contained subprocess inside the
conversation's workspace.

What every run gets, on every OS:
  * cwd = the workspace; HOME / USERPROFILE / TMPDIR point inside it
  * a scrubbed environment built from scratch (PATH and a few OS essentials; nothing whose name
    looks like a token, key, password or session, so JEPA_STUDIO_TOKEN is never visible)
  * a wall-clock timeout; on timeout or cancel the whole process tree is killed
  * stdout / stderr captured and capped (STREAM_CAP bytes each, the rest is counted and dropped)
  * no shell: argv lists only (run_shell passes its text to `bash -c` explicitly, after approval)
  * a cap on the workspace's total size (limits.max_workspace_mb): a run does not start while the
    workspace is over it, and a run that pushes it over reports an error

POSIX (Linux, macOS) adds resource limits, set by a tiny launcher that calls setrlimit() and then
starts the program (subprocess's preexec_fn is not safe in a threaded server):
  RLIMIT_CPU (cpu_seconds), memory (RLIMIT_DATA on Linux, which counts heap and anonymous memory but
  not the large address-space reservations V8 and CUDA make; RLIMIT_AS elsewhere; macOS does not
  enforce either), RLIMIT_FSIZE (max_output_mb per file, SIGXFSZ ignored so writes fail with EFBIG),
  RLIMIT_NPROC (your current process count + 256: a fork-bomb brake), RLIMIT_NOFILE (512),
  RLIMIT_CORE (0). On Linux the launcher stays as a "child subreaper": every process the program
  leaves behind (even after double-forking and setsid()) is re-parented to it and killed when the
  program ends or the run is stopped.

Isolation layers, detected by a probe at startup (Sandbox.capabilities() reports what the probe
actually observed, never what we hope):
  Linux  bwrap (bubblewrap): new network namespace (only `lo`, which is not the host's loopback, so
         even the JEPA Studio backend is unreachable), new PID namespace (the whole tree dies with
         it), the file system read-only except the workspace; /tmp, /run, /var/run, your home
         folder, /mnt, /media, jev-data (settings.json, jev.db, other conversations) hidden behind
         read-only or size-capped tmpfs (the Python / Node / FFmpeg installs are re-exposed
         read-only), docker/podman sockets masked, AF_VSOCK sockets refused (seccomp), WSL's
         /init and interop sockets hidden.
         unshare --user --mount --pid --net: a network and PID namespace plus a mount namespace in
         which jev-data, the other workspaces and /run are hidden (when bwrap is missing or
         blocked). The rest of the file system stays writable, so this counts as a weak sandbox.
  macOS  sandbox-exec with a (deny default) profile: no network, writes only in the workspace, no
         programs except the interpreter's own tree (Apple marks sandbox-exec deprecated; the probe
         checks it still works). Until a maintainer has verified the profile on a Mac
         (MACOS_PROFILE_VERIFIED), macOS counts as a weak sandbox.
  Windows  a Job Object (kill-on-close, job memory limit, CPU time, max 64 processes); a run is
         refused if the job cannot be created or the process cannot be put into it. No network or
         file-system isolation: a weak sandbox. docs/jev/SAFETY.md says what that means.

On a weak sandbox (no file-system or no network isolation) the policy's locked `weak_sandbox` row
makes every tool that runs code or FFmpeg ask first (jev/policy.py).
"""
from __future__ import annotations

import json
import os
import platform
import re
import shutil
import signal
import stat
import struct
import subprocess
import sys
import tempfile
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable

STREAM_CAP = 1 << 20            # bytes kept per stream (stdout, stderr)
SKIP_DIRS = {".jev", ".venv", ".tmp", "__pycache__", "node_modules", ".git"}
SECRET_RE = re.compile(r"TOKEN|SECRET|PASSW|KEY|AUTH|CRED|COOKIE|SESSION|PRIVATE", re.I)
PASS_ENV = ("PATH", "SYSTEMROOT", "WINDIR", "COMSPEC", "PATHEXT", "NUMBER_OF_PROCESSORS", "PROCESSOR_ARCHITECTURE",
            "CUDA_VISIBLE_DEVICES", "NVIDIA_VISIBLE_DEVICES", "LD_LIBRARY_PATH", "TZ")
# passed through only when a run has network access (approved, or network_allowed): proxies and CA bundles
NET_ENV = ("HTTP_PROXY", "HTTPS_PROXY", "NO_PROXY", "http_proxy", "https_proxy", "no_proxy", "SSL_CERT_FILE",
           "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "PIP_CERT",
           "NODE_EXTRA_CA_CERTS")
NET_FILE_ENV = ("SSL_CERT_FILE", "SSL_CERT_DIR", "REQUESTS_CA_BUNDLE", "PIP_CERT", "NODE_EXTRA_CA_CERTS")
IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"
IS_LINUX = sys.platform.startswith("linux")
DATA_DIRNAME = "jev-data"
# Flip to True only after a maintainer has run the checks in docs/jev/SAFETY.md ("Verifying the
# macOS profile") on a real Mac. Until then macOS code runs count as a weak sandbox and ask first.
MACOS_PROFILE_VERIFIED = False
# /run entries some systems need inside the sandbox (NixOS keeps its system and its NVIDIA / OpenGL
# driver libraries there); they are recreated as read-only symlinks, everything else in /run is hidden
RUN_KEEP = ("/run/current-system", "/run/booted-system", "/run/opengl-driver", "/run/opengl-driver-32")
# bubblewrap's root is an empty tmpfs holding only these system folders (read-only) plus what a run
# needs; everything else (/var, /opt, /srv, /home, /root, /mnt, /media, /run, custom top-level
# folders) is absent. /nix and /gnu are the NixOS / Guix stores.
ROOT_ALLOW = ("/usr", "/etc", "/bin", "/sbin", "/lib", "/lib32", "/lib64", "/libx32", "/sys", "/nix", "/gnu")


class SandboxError(RuntimeError):
    """The run could not be started safely (and was not started)."""


# Sets rlimits, then (Linux) stays as a child subreaper that kills whatever the program leaves
# behind, or (elsewhere) execs the program. Runs with `python -I -S` so nothing from the workspace
# is imported.
LAUNCHER = r"""
import os, resource as r, signal, sys
cpu, mem, fsize, nproc, nofile, kind = (int(x) for x in sys.argv[1].split(","))
def s(k, soft, hard=None):
    hard = soft if hard is None else hard
    try:
        _, h = r.getrlimit(k)
        if h != r.RLIM_INFINITY:
            soft, hard = min(soft, h), min(hard, h)
        r.setrlimit(k, (soft, hard))
    except (ValueError, OSError):
        pass
s(r.RLIMIT_CPU, cpu, cpu + 2)
s(r.RLIMIT_DATA if kind == 0 else r.RLIMIT_AS, mem)
s(r.RLIMIT_FSIZE, fsize)
if hasattr(r, "RLIMIT_NPROC"):
    s(r.RLIMIT_NPROC, nproc)
s(r.RLIMIT_NOFILE, nofile)
s(r.RLIMIT_CORE, 0)
signal.signal(signal.SIGXFSZ, signal.SIG_IGN)
argv = sys.argv[2:]
reaper = False
if sys.platform.startswith("linux"):
    try:
        import ctypes
        reaper = ctypes.CDLL(None, use_errno=True).prctl(36, 1, 0, 0, 0) == 0   # PR_SET_CHILD_SUBREAPER
    except Exception:
        reaper = False
if not reaper:
    os.execvp(argv[0], argv)
pid = os.fork()
if pid == 0:
    try:
        os.execvp(argv[0], argv)
    except OSError as e:
        sys.stderr.write(f"could not start {argv[0]}: {e}\n")
        os._exit(127)
def fwd(sig, _frame):
    try:
        os.kill(pid, sig)
    except ProcessLookupError:
        pass
for sg in (signal.SIGTERM, signal.SIGINT, signal.SIGHUP):
    signal.signal(sg, fwd)
status = 0
while True:
    try:
        p, st = os.waitpid(-1, 0)
    except ChildProcessError:
        break
    except InterruptedError:
        continue
    if p == pid:
        status = st
        break
me = str(os.getpid())
def children():
    out = []
    for d in os.listdir("/proc"):
        if d.isdigit():
            try:
                with open(f"/proc/{d}/stat", "rb") as f:
                    if f.read().rsplit(b")", 1)[1].split()[1].decode() == me:
                        out.append(int(d))
            except (OSError, IndexError):
                pass
    return out
import time
deadline = time.monotonic() + 5
while time.monotonic() < deadline:
    kids = children()
    if not kids:
        break
    for k in kids:
        try:
            os.kill(k, signal.SIGKILL)
        except ProcessLookupError:
            pass
    while True:
        try:
            if os.waitpid(-1, os.WNOHANG)[0] == 0:
                break
        except ChildProcessError:
            break
    time.sleep(0.01)
if os.WIFSIGNALED(status):
    sig = os.WTERMSIG(status)
    try:
        signal.signal(sig, signal.SIG_DFL)
        os.kill(os.getpid(), sig)
    except (OSError, ValueError):
        pass
    os._exit(128 + sig)
os._exit(os.WEXITSTATUS(status) if os.WIFEXITED(status) else 1)
"""

# unshare mode: runs as PID 1 of the new namespaces, hides jev-data, the other workspaces and /run
# behind read-only tmpfs mounts, binds the workspace (and the package venv) back, then execs.
PRELUDE = r"""
import ctypes, json, os, sys
spec = json.loads(sys.argv[1])
argv = sys.argv[2:]
libc = ctypes.CDLL(None, use_errno=True)
libc.mount.argtypes = [ctypes.c_char_p, ctypes.c_char_p, ctypes.c_char_p, ctypes.c_ulong, ctypes.c_char_p]
RDONLY, NOSUID, NODEV, NOEXEC, REMOUNT, BIND, REC, PRIVATE = 1, 2, 4, 8, 32, 4096, 16384, 1 << 18
def mount(src, dst, fs, flags, data=None):
    b = lambda x: x.encode() if isinstance(x, str) else x
    if libc.mount(b(src), b(dst), b(fs), flags, b(data)) != 0:
        e = ctypes.get_errno()
        raise OSError(e, f"mount {dst}: {os.strerror(e)}")
def kept(path):
    f, out = os.statvfs(path).f_flag, 0
    for st, ms in ((2, NOSUID), (4, NODEV), (8, NOEXEC), (1024, 1024), (2048, 2048), (4096, 1 << 21)):
        if f & st:
            out |= ms
    return out
try:
    mount(None, "/", None, REC | PRIVATE)
    binds = [(p, False) for p in spec["ro"]] + [(p, True) for p in spec["rw"]] + [(spec["ws"], spec["ws_rw"])]
    fds = {}
    for p, _ in binds:
        if os.path.exists(p):
            fds[p] = os.open(p, os.O_PATH | os.O_CLOEXEC)
    hidden = []
    for h in spec["hide"]:
        if os.path.isdir(h) and not os.path.islink(h):
            mount("tmpfs", h, "tmpfs", NOSUID | NODEV, "mode=0755,size=4m")
            hidden.append(h)
    for target, dest in spec["symlinks"]:
        if not os.path.lexists(dest):
            os.symlink(target, dest)
    for p, rw in binds:
        if p not in fds:
            continue
        src = f"/proc/self/fd/{fds[p]}"
        if not os.path.exists(p):
            if os.path.isdir(src):
                os.makedirs(p, exist_ok=True)
            else:
                os.makedirs(os.path.dirname(p), exist_ok=True)
                open(p, "a").close()
        mount(src, p, None, BIND | REC)
        if not rw:
            mount(None, p, None, REMOUNT | BIND | RDONLY | kept(p))
    for h in hidden:
        mount(None, h, None, REMOUNT | BIND | RDONLY | NOSUID | NODEV)
    for sock in spec.get("mask", []):
        try:
            mount("/dev/null", sock, None, BIND)
        except OSError:
            pass
    os.chdir(spec["cwd"])
    # Lock every mount above and give up all capabilities: a second user namespace makes the mounts
    # "locked" (they can no longer be unmounted or remounted from inside), then securebits, the
    # bounding set, the ambient set and capset() leave the program with no capability at all, now
    # or after exec.
    uid, gid = os.getuid(), os.getgid()
    if libc.unshare(0x10000000 | 0x20000) != 0:            # CLONE_NEWUSER | CLONE_NEWNS
        raise OSError(ctypes.get_errno(), "unshare(CLONE_NEWUSER|CLONE_NEWNS)")
    for name, text in (("setgroups", "deny"), ("uid_map", f"{uid} {uid} 1"), ("gid_map", f"{gid} {gid} 1")):
        with open(f"/proc/self/{name}", "w") as fh:
            fh.write(text)
    libc.prctl.argtypes = [ctypes.c_int, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong, ctypes.c_ulong]
    if libc.prctl(28, 0x2F, 0, 0, 0) != 0:                  # PR_SET_SECUREBITS: NOROOT, NO_SETUID_FIXUP (+locks)
        raise OSError(ctypes.get_errno(), "PR_SET_SECUREBITS")
    for cap in range(64):
        libc.prctl(24, cap, 0, 0, 0)                         # PR_CAPBSET_DROP
    libc.prctl(47, 4, 0, 0, 0)                               # PR_CAP_AMBIENT_CLEAR_ALL
    hdr = (ctypes.c_uint32 * 2)(0x20080522, 0)
    data = (ctypes.c_uint32 * 6)()
    if libc.capset(hdr, data) != 0:
        raise OSError(ctypes.get_errno(), "capset")
    if spec.get("seccomp"):
        prog = bytes.fromhex(spec["seccomp"])
        class Fprog(ctypes.Structure):
            _fields_ = [("len", ctypes.c_ushort), ("filter", ctypes.c_char_p)]
        fp = Fprog(len(prog) // 8, prog)
        if libc.prctl(38, 1, 0, 0, 0) != 0 or libc.prctl(22, 2, ctypes.addressof(fp), 0, 0) != 0:
            raise OSError(ctypes.get_errno(), "seccomp filter could not be installed")
except OSError as e:
    sys.stderr.write(f"JEV sandbox setup failed, not running: {e}\n")
    os._exit(126)
os.execvp(argv[0], argv)
"""

PROBE = r"""
import os, socket, sys
ws, outside, protected = sys.argv[1:4]
try:
    ifs = sorted(n for _, n in socket.if_nameindex())
except OSError:
    ifs = ["?"]
s = socket.socket(); s.settimeout(1.5)
try:
    rc = s.connect_ex(("192.0.2.1", 9))   # TEST-NET-1: never routed anywhere real
except OSError as e:
    rc = e.errno or -1
net = "blocked" if rc in (101, 113, 1, 13, 10051, 10065) or ifs == ["lo"] else "open"
open(os.path.join(ws, "probe.txt"), "w").write("x")
for d in (outside, protected):
    try:
        open(os.path.join(d, "jev-escape-probe.txt"), "w").write("x")
    except OSError:
        pass
print("JEVPROBE", ",".join(ifs), net)
"""


@dataclass
class Limits:
    cpu_seconds: int = 120
    memory_mb: int = 2048
    wall_seconds: int = 300
    max_output_mb: int = 200
    max_file_mb: int = 500
    max_workspace_mb: int = 2048
    max_workspace_entries: int = 200_000

    @classmethod
    def from_settings(cls, s: dict | None) -> "Limits":
        d = dict((s or {}).get("limits") or {})
        return cls(**{k: int(v) for k, v in d.items() if k in cls.__dataclass_fields__})


@dataclass
class Mounts:
    """Extra paths for one run. `ro` are made visible read-only (re-exposed when hidden), `rw` are
    writable (only install_package uses this: its venv and pip's cache). `workspace_writable=False`
    turns the workspace itself read-only for the run."""
    ro: list = field(default_factory=list)
    rw: list = field(default_factory=list)
    workspace_writable: bool = True


@dataclass
class RunResult:
    returncode: int | None
    stdout: str
    stderr: str
    seconds: float
    timed_out: bool = False
    cancelled: bool = False
    truncated: bool = False
    limit_hit: str | None = None
    changed_files: list[str] = field(default_factory=list)
    isolation: str = "none"
    refused: str | None = None          # the run was not started (why)
    workspace_over: str | None = None   # the run pushed the workspace over max_workspace_mb

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.cancelled and not self.refused \
            and not self.workspace_over

    def summary(self, max_chars: int = 3500) -> str:
        if self.refused:
            return f"not run: {self.refused}"
        head = f"exit code {self.returncode}" if self.returncode is not None else "killed"
        if self.timed_out:
            head = f"stopped: wall-clock limit reached after {self.seconds:.0f} s"
        elif self.cancelled:
            head = "cancelled"
        elif self.limit_hit:
            head += f" ({self.limit_hit})"
        parts = [head + f", {self.seconds:.1f} s"]
        if self.workspace_over:
            parts.append("ERROR: " + self.workspace_over)
        half = max_chars // 2
        if self.stdout.strip():
            parts.append("stdout:\n" + _clip(self.stdout, half))
        if self.stderr.strip():
            parts.append("stderr:\n" + _clip(self.stderr, half))
        if self.changed_files:
            parts.append("files written: " + ", ".join(self.changed_files[:20]) +
                         (f" (+{len(self.changed_files) - 20} more)" if len(self.changed_files) > 20 else ""))
        return "\n".join(parts)


def _clip(s: str, n: int) -> str:
    if len(s) <= n:
        return s.rstrip()
    return s[: n * 2 // 3].rstrip() + f"\n[... {len(s) - n} characters cut ...]\n" + s[-n // 3:].rstrip()


def scrubbed_env(workspace: Path, extra: dict | None = None, network: bool = False, venv: Path | None = None,
                 tmp: Path | None = None) -> dict:
    env = {}
    for k in PASS_ENV + (NET_ENV if network else ()):
        v = os.environ.get(k)
        if v is not None and not SECRET_RE.search(k):
            env[k] = v
    env.setdefault("PATH", os.defpath)
    if tmp is None:
        from .wsfs import ensure_dir
        tmp = ensure_dir(workspace, ".tmp")       # a link planted at .tmp is removed, never followed
    else:
        try:
            Path(tmp).mkdir(parents=True, exist_ok=True)  # wsfs-ok: backend-owned scratch given by the caller
        except OSError:
            pass
    threads = str(max(1, min(4, os.cpu_count() or 1)))
    env.update({"HOME": str(workspace), "USERPROFILE": str(workspace), "TMPDIR": str(tmp), "TEMP": str(tmp),
                "TMP": str(tmp), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(tmp / "mpl"), "XDG_CACHE_HOME": str(tmp / "cache"),
                "OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads, "NO_COLOR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INPUT": "1", "npm_config_update_notifier": "false"})
    if venv is not None:
        bindir = Path(venv) / ("Scripts" if IS_WIN else "bin")
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = str(bindir) + os.pathsep + env["PATH"]
    for k, v in (extra or {}).items():
        if not SECRET_RE.search(k):
            env[k] = str(v)
    return env


def python_for(workspace: Path | None = None, venv: Path | None = None) -> str:
    """The interpreter for a code run: the conversation's package venv (kept in jev-data, never in
    the workspace) when it exists and its interpreter is still this Python, else this Python."""
    if venv is not None:
        exe = trusted_venv_python(venv)
        if exe is not None:
            return str(exe)
    return sys.executable


def _same_file(a: Path, b: Path) -> bool:
    try:
        return os.path.samefile(a, b)
    except OSError:
        return False


def trusted_venv_python(venv: Path) -> Path | None:
    """`venv`'s interpreter if it is still the trusted base interpreter (POSIX: a symlink resolving
    to this Python; Windows: byte-identical to the venv launcher or base python.exe), else None.
    Anything else (a planted script, a swapped binary) is never executed."""
    venv = Path(venv)
    exe = venv / ("Scripts/python.exe" if IS_WIN else "bin/python")
    try:
        st = exe.lstat()
    except OSError:
        return None
    bases = {Path(sys.executable).resolve()}
    base_exe = getattr(sys, "_base_executable", None)
    if base_exe:
        bases.add(Path(base_exe).resolve())
    try:
        cfg = (venv / "pyvenv.cfg").read_text(encoding="utf-8", errors="replace")  # wsfs-ok: the package venv lives in jev-data
    except OSError:
        return None
    home = next((ln.split("=", 1)[1].strip() for ln in cfg.splitlines() if ln.split("=", 1)[0].strip() == "home"), "")
    homes = {Path(sys.executable).parent.resolve(), *(b.parent for b in bases)}
    if base_exe:
        homes.add(Path(base_exe).parent.resolve())
    if not home or Path(home).resolve() not in homes:
        return None                          # pyvenv.cfg points the interpreter somewhere else
    if not IS_WIN:
        if stat.S_ISLNK(st.st_mode) or stat.S_ISREG(st.st_mode):
            real = exe.resolve()
            if any(_same_file(real, b) for b in bases) or real in bases:
                return exe
        return None
    import hashlib
    import venv as _venv_mod
    want = set()
    for cand in [Path(_venv_mod.__file__).parent / "scripts" / "nt" / "python.exe", *bases]:
        try:
            want.add(hashlib.sha256(cand.read_bytes()).hexdigest())  # wsfs-ok: the Python install
        except OSError:
            pass
    try:
        return exe if hashlib.sha256(exe.read_bytes()).hexdigest() in want else None  # wsfs-ok: the package venv in jev-data
    except OSError:
        return None


def snapshot(workspace: Path, limit: int = 5000) -> dict[str, tuple[int, int]]:
    out: dict[str, tuple[int, int]] = {}
    for dirpath, dirnames, filenames in os.walk(workspace):
        dirnames[:] = [d for d in dirnames if d not in SKIP_DIRS]
        for fn in filenames:
            p = Path(dirpath) / fn
            try:
                st = p.lstat()
            except OSError:
                continue
            out[p.relative_to(workspace).as_posix()] = (st.st_size, st.st_mtime_ns)
            if len(out) >= limit:
                return out
    return out


def workspace_usage(workspace: Path, max_entries: int = 200_000) -> tuple[int, int, bool]:
    """(bytes, entries, complete) for the workspace: bytes on disk of every file (caches and .tmp
    included; allocated blocks on POSIX, so a sparse file counts for what it really occupies) and
    the number of files and folders seen. The walk stops after max_entries (its cost is bounded);
    complete=False then, and callers treat that as over the limit (fail closed, N10)."""
    total, n = 0, 0
    stack = [str(workspace)]
    while stack:
        d = stack.pop()
        try:
            it = os.scandir(d)
        except OSError:
            continue
        with it:
            for e in it:
                n += 1
                if n > max_entries:
                    return total, n, False
                try:
                    st = e.stat(follow_symlinks=False)
                except OSError:
                    continue
                if stat.S_ISDIR(st.st_mode):
                    stack.append(e.path)
                    continue
                total += st.st_blocks * 512 if hasattr(st, "st_blocks") else st.st_size
    return total, n, True


def _user_process_count() -> int:
    try:
        import psutil
        uid = os.getuid()
        return sum(1 for p in psutil.process_iter(["uids"]) if p.info["uids"] and p.info["uids"].real == uid)
    except Exception:  # noqa: BLE001
        return 512


def _sbpl_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


def _under(p: Path, parent: Path) -> bool:
    return p == parent or parent in p.parents


def _home() -> Path | None:
    try:
        h = Path.home().resolve()
    except (RuntimeError, OSError):
        return None
    return h if h != Path("/") else None


# ---------------------------------------------------------------------------------------- helpers
def bwrap_supports_size(version_text: str) -> bool:
    """bubblewrap 0.10.0 added `--size` for --tmpfs."""
    m = re.search(r"(\d+)\.(\d+)(?:\.(\d+))?", version_text or "")
    return bool(m) and (int(m.group(1)), int(m.group(2))) >= (0, 10)


_BWRAP_VERSION: dict = {}


def _bwrap_version() -> str:
    if "v" not in _BWRAP_VERSION:
        try:
            _BWRAP_VERSION["v"] = subprocess.run(["bwrap", "--version"], capture_output=True, text=True,
                                                 timeout=10).stdout.strip()
        except (OSError, subprocess.TimeoutExpired):
            _BWRAP_VERSION["v"] = ""
    return _BWRAP_VERSION["v"]


def userns_restricted_by_apparmor() -> bool:
    """Ubuntu 23.10+ sets kernel.apparmor_restrict_unprivileged_userns=1: unconfined programs (bwrap,
    unshare) can't create user namespaces unless an AppArmor profile grants `userns`."""
    try:
        return Path("/proc/sys/kernel/apparmor_restrict_unprivileged_userns").read_text().strip() == "1"  # wsfs-ok: a fixed kernel sysctl, not a workspace path
    except OSError:
        return False


def wsl_info(proc: str = "/proc", run: str = "/run") -> dict:
    """Is this WSL, and is Windows interop (running .exe files from Linux) switched on?"""
    def read(p):
        try:
            return Path(p).read_text(errors="replace")  # wsfs-ok: /proc files
        except OSError:
            return ""
    rel = (read(f"{proc}/sys/kernel/osrelease") + read(f"{proc}/version")).lower()
    detected = "microsoft" in rel or "wsl" in rel or os.path.isdir(f"{run}/WSL")
    interop = False
    for name in ("WSLInterop", "WSLInterop-late"):
        txt = read(f"{proc}/sys/fs/binfmt_misc/{name}")
        if txt.strip().startswith("enabled"):
            interop = True
    win_path = any(p.startswith("/mnt/") and ("Windows" in p or "/c/" in p) for p in
                   os.environ.get("PATH", "").split(os.pathsep))
    return {"detected": bool(detected), "interop_enabled": bool(detected and interop),
            "windows_path_in_PATH": bool(detected and win_path)}


def seccomp_program(machine: str | None = None) -> bytes | None:
    """A classic-BPF seccomp filter for sandboxed runs:
      * socket(AF_VSOCK, ...) fails with EAFNOSUPPORT (WSL2 and VMs talk to their host over vsock,
        which network namespaces do not isolate);
      * io_uring_setup / io_uring_enter / io_uring_register fail with ENOSYS: io_uring can create
        sockets (IORING_OP_SOCKET) without the socket() syscall, so it would bypass the check
        above (the same three are blocked by Docker's default profile);
      * syscalls from other ABIs (32-bit, x32) fail with ENOSYS so they cannot get around either;
    the rest is allowed. None for machines we have no syscall numbers for."""
    arch = {"x86_64": (0xC000003E, 41, True), "amd64": (0xC000003E, 41, True),
            "aarch64": (0xC00000B7, 198, False), "arm64": (0xC00000B7, 198, False)}.get(
        (machine or platform.machine()).lower())
    if arch is None:
        return None
    audit_arch, nr_socket, x32 = arch
    allow, errno_ = 0x7FFF0000, 0x00050000
    enosys = errno_ | 38
    ld, jeq, jge, ret = 0x20, 0x15, 0x35, 0x06
    prog: list[tuple[int, int, int, int]] = [
        (ld, 0, 0, 4),                          # A = arch
        (jeq, 1, 0, audit_arch),                # native ABI -> continue, else
        (ret, 0, 0, enosys),
        (ld, 0, 0, 0),                          # A = syscall number
    ]
    if x32:
        prog += [(jge, 0, 1, 0x40000000), (ret, 0, 0, enosys)]
    for nr in (425, 426, 427):                  # io_uring_setup, io_uring_enter, io_uring_register (both ABIs)
        prog += [(jeq, 0, 1, nr), (ret, 0, 0, enosys)]
    prog += [
        (jeq, 0, 3, nr_socket),                 # socket()? else allow
        (ld, 0, 0, 16),                         # A = args[0] (domain, low 32 bits)
        (jeq, 0, 1, 40),                        # AF_VSOCK
        (ret, 0, 0, errno_ | 97),               # EAFNOSUPPORT
        (ret, 0, 0, allow),
    ]
    return b"".join(struct.pack("=HBBI", *i) for i in prog)


def listening_unix_sockets(limit: int | None = None) -> list[str]:
    """Paths of the Unix sockets listening on this host (from /proc/net/unix): bubblewrap and
    unshare mask the ones the sandbox could otherwise see (/var/lib/mysql/mysqld.sock, LXD, ...)."""
    out: list[str] = []
    try:
        with open("/proc/net/unix", encoding="utf-8", errors="replace") as f:  # wsfs-ok: kernel table
            next(f, None)
            for line in f:
                cols = line.split(None, 7)
                if len(cols) < 8 or not cols[7].startswith("/"):
                    continue
                if int(cols[3], 16) & 0x10000 == 0:         # __SO_ACCEPTCON: listening
                    continue
                path = cols[7].rstrip("\n")
                try:
                    if stat.S_ISSOCK(os.lstat(path).st_mode) and path not in out:
                        out.append(path)
                except OSError:
                    continue
                if limit is not None and len(out) >= limit:
                    break
    except OSError:
        pass
    return out


def macos_profile(ws: Path, network: bool = False, *, read_only=(), writable=(), hide=(), exec_paths=(),
                  shell: bool = False) -> str:
    """The sandbox-exec (SBPL) profile for one run, built from (deny default); later rules win.

    Reads: the system libraries and frameworks, the interpreter installs in `read_only` and the
    workspace; `hide` (jev-data, other workspaces) stays unreadable. Writes: the workspace and
    `writable` only. Programs: only those under `exec_paths` (the interpreter's own install; for
    run_shell, after approval, also /bin and /usr/bin), and never open, osascript, launchctl or
    the AppleScript/automation tools. Network: none unless `network`. Mach services: only the few
    lookups a Python or Node process needs; LaunchServices, AppleEvents and the pasteboard are
    denied, so `open -a` and `osascript` cannot reach the GUI session even if started.
    Not yet verified on a real Mac: see MACOS_PROFILE_VERIFIED and docs/jev/SAFETY.md."""
    q = _sbpl_quote

    def subpaths(paths):
        return " ".join(f"(subpath {q(str(p))})" for p in paths)
    ws_s = str(ws)
    rules = [
        "(version 1)",
        "(deny default)",
        ";; JEV run profile: deny everything, then allow the minimum (no (import \"system.sb\"))",
        "(allow process-fork)",
        "(allow signal (target same-sandbox))",
        "(allow sysctl-read)",
        "(allow file-read-metadata)",
        "(allow file-read* (literal \"/\") (subpath \"/System\") (subpath \"/usr/lib\") (subpath \"/usr/share\")"
        " (subpath \"/Library/Frameworks\") (subpath \"/private/etc\") (subpath \"/private/var/db/timezone\")"
        " (subpath \"/private/var/db/dyld\") (subpath \"/usr/local\") (subpath \"/opt/homebrew\")"
        " (literal \"/dev/null\") (literal \"/dev/zero\") (literal \"/dev/random\") (literal \"/dev/urandom\")"
        " (literal \"/dev/dtracehelper\") (regex #\"^/dev/fd/\"))",
        "(allow file-read* " + subpaths([ws, *read_only, *writable]) + ")",
        "(allow file-write* (subpath " + q(ws_s) + ")" + (" " + subpaths(writable) if writable else "") +
        " (literal \"/dev/null\") (literal \"/dev/zero\") (literal \"/dev/dtracehelper\") (regex #\"^/dev/fd/\"))",
        "(allow file-ioctl (literal \"/dev/dtracehelper\"))",
        "(allow process-exec* " + subpaths([*exec_paths, *(("/bin", "/usr/bin") if shell else ())]) + ")",
        "(deny process-exec* (literal \"/usr/bin/open\") (literal \"/usr/bin/osascript\")"
        " (literal \"/usr/bin/osacompile\") (literal \"/bin/launchctl\") (literal \"/usr/bin/automator\")"
        " (literal \"/usr/bin/shortcuts\") (literal \"/usr/bin/pbcopy\") (literal \"/usr/bin/pbpaste\"))",
        "(allow mach-lookup (global-name \"com.apple.system.opendirectoryd.libinfo\")"
        " (global-name \"com.apple.system.notification_center\") (global-name \"com.apple.logd\")"
        " (global-name \"com.apple.system.logger\"))",
        "(allow ipc-posix-shm-read* ipc-posix-shm-write-data ipc-posix-shm-write-create ipc-posix-sem)",
        "(deny mach-lookup (global-name \"com.apple.coreservices.launchservicesd\")"
        " (global-name \"com.apple.coreservices.appleevents\") (global-name \"com.apple.pasteboard.1\")"
        " (global-name \"com.apple.windowserver.active\"))",
    ]
    if hide:
        rules.append("(deny file-read* file-write* " + subpaths(hide) + ")")
        rules.append("(allow file-read* file-write* (subpath " + q(ws_s) + ")" +
                     (" " + subpaths(writable) if writable else "") + ")")
        if read_only:
            rules.append("(allow file-read* " + subpaths(read_only) + ")")
    if network:
        rules.append("(allow network-outbound) (allow network-inbound (local ip \"localhost:*\")) (allow system-socket)"
                     " (allow mach-lookup (global-name \"com.apple.dnssd.service\")"
                     " (global-name \"com.apple.trustd\") (global-name \"com.apple.SecurityServer\"))")
    return "\n".join(rules)


@dataclass
class Spawned:
    """A process started by Sandbox.spawn. Call finish() exactly once when done with it."""
    proc: subprocess.Popen
    sandbox: "Sandbox"
    workspace: Path
    limits: Limits
    method: str
    job: object = None
    unit: str | None = None
    check_workspace: bool = True
    token: str = ""
    _finished: bool = False
    _tracked: dict = field(default_factory=dict)
    _next_track: float = 0.0
    _next_size: float = 0.0
    _size_every: float = 1.0
    over: str | None = None

    def kill(self) -> None:
        self.sandbox._kill_tree(self.proc, self.job, self.unit, self)

    def guard(self) -> str | None:
        """Call often while the run is going: remembers every descendant (so a process that kills its
        launcher can still be found) and enforces max_workspace_mb during the run. Returns an error
        text (and kills the run) when the workspace goes over its cap."""
        now = time.monotonic()
        if now >= self._next_track and self.method not in ("bwrap", "unshare"):
            self._next_track = now + 0.1
            self.sandbox._track(self)
        if self.check_workspace and now >= self._next_size:
            t0 = time.monotonic()
            self.over = self.sandbox.workspace_over(self.workspace, self.limits)
            took = time.monotonic() - t0
            self._next_size = time.monotonic() + max(self._size_every, 10 * took)   # at most ~10% of the time
            if self.over:
                self.kill()
                return self.over
        return None

    def finish(self) -> str | None:
        """Kill leftovers, release the job, and return an error text if the workspace is now over
        its size limit (None otherwise)."""
        if self._finished:
            return None
        self._finished = True
        self.kill()
        if self.job:
            self.job.close()
        if self.check_workspace:
            return self.over or self.sandbox.workspace_over(self.workspace, self.limits)
        return self.over


# ---------------------------------------------------------------------------------------- sandbox
class Sandbox:
    def __init__(self, settings: dict | None = None):
        s = settings or {}
        sb = s.get("sandbox") or {}
        self.mode = sb.get("isolation", "auto")
        self.hide_home = bool(sb.get("hide_home", True))
        self.network_allowed = bool(s.get("network_allowed", False))
        self.limits = Limits.from_settings(s)
        self.data_dir = Path(s["data_dir"]).resolve() if s.get("data_dir") else None
        self.workspace_root = Path(s["workspace_root"]).resolve() if s.get("workspace_root") else None
        self._caps: dict | None = None
        self._lock = threading.Lock()
        self._wsl: dict | None = None

    # ------------------------------------------------------------------ detection
    def capabilities(self) -> dict:
        with self._lock:
            if self._caps is None:
                self._caps = self._detect()
            caps = dict(self._caps)
        if self.network_allowed:
            caps["network_isolation"] = "disabled (network_allowed is on)"
        caps["limits"] = self.limits.__dict__.copy()
        return caps

    def _exe(self, name: str) -> str | None:
        """Absolute path of a sandbox helper (bwrap, unshare, ...), looked up once: later PATH changes
        (a tool's own environment) never change which binary wraps the runs."""
        cache = self.__dict__.setdefault("_exes", {})
        if name not in cache:
            cache[name] = shutil.which(name)
        return cache[name]

    def wsl(self) -> dict:
        if self._wsl is None:
            self._wsl = wsl_info() if IS_LINUX else {"detected": False, "interop_enabled": False,
                                                    "windows_path_in_PATH": False}
        return self._wsl

    def _finish_caps(self, base: dict) -> dict:
        reasons = []
        if not base.get("fs_isolated"):
            reasons.append(f"no file-system isolation (method {base['method']})")
        if not base.get("net_isolated"):
            reasons.append(f"no network isolation (method {base['method']})")
        if base["method"] == "sandbox-exec" and not MACOS_PROFILE_VERIFIED:
            reasons.append("the macOS sandbox profile has not been verified on a Mac yet")
        base["weak"] = bool(reasons)
        base["weak_reasons"] = reasons
        w = self.wsl()
        base["wsl"] = dict(w)
        notes = []
        if w.get("interop_enabled"):
            notes.append("WSL interop is on: Windows programs can be started from Linux. JEV hides /init, "
                         "/run/WSL, /mnt and blocks vsock inside bubblewrap, but turn interop off for a stronger "
                         "boundary ([interop] enabled=false in /etc/wsl.conf; see docs/jev/SAFETY.md)")
        if base["method"] != "bwrap" and IS_LINUX and userns_restricted_by_apparmor():
            notes.append("Ubuntu's AppArmor blocks user namespaces for unconfined programs "
                         "(kernel.apparmor_restrict_unprivileged_userns=1), so bubblewrap cannot isolate runs. "
                         "Fix: sudo apt-get install -y bubblewrap apparmor && sudo install -m 644 "
                         "scripts/apparmor/jev-bwrap /etc/apparmor.d/jev-bwrap && sudo apparmor_parser -r "
                         "/etc/apparmor.d/jev-bwrap (docs/jev/SAFETY.md, 'Ubuntu 24.04 and later')")
        if w.get("windows_path_in_PATH"):
            notes.append("Windows folders are on PATH (appendWindowsPath); set appendWindowsPath=false")
        base["isolation_notes"] = notes
        base["isolation"] = "weak" if reasons else ("weaker (WSL interop)" if w.get("interop_enabled") else "strong")
        return base

    def _detect(self) -> dict:
        base = {"os": sys.platform, "method": "none", "network_isolation": "not available",
                "filesystem": "not enforced (code can read and write anything your user can)",
                "rlimits": "not available" if IS_WIN else
                ("cpu, data, fsize, nproc, nofile" if IS_LINUX else
                 "cpu, fsize, nproc, nofile (macOS does not enforce memory rlimits)"),
                "process_tree_kill": "job object" if IS_WIN else
                ("subreaper launcher + session + descendant walk" if IS_LINUX else "session + descendant walk"),
                "windows_job": False, "probe": None, "fs_isolated": False, "net_isolated": False,
                "data_dir_protected": False, "tmpfs_size_limit": False, "cgroup_scope": False}
        if IS_WIN:
            try:
                self._win_job_create(64, 1, 1).close()
                base.update(windows_job=True, method="job-object",
                            rlimits="job object: memory, CPU time, 64 processes, kill on close")
            except Exception as e:  # noqa: BLE001
                base["probe"] = f"job object unavailable: {e}"
            return self._finish_caps(base)
        order = {"auto": ["bwrap", "unshare", "sandbox-exec"], "none": []}.get(self.mode, [self.mode])
        notes = []
        for method in order:
            if method == "sandbox-exec" and not IS_MAC:
                continue
            if method in ("bwrap", "unshare") and not IS_LINUX:
                continue
            if not self._exe(method):
                notes.append(f"{method}: not installed")
                continue
            res = self._probe(method)
            notes.append(f"{method}: {res}")
            if res and res.startswith("JEVPROBE"):
                _, ifs, net, fs, prot = res.split()
                if net != "blocked":
                    continue
                base.update(method=method, network_isolation="enforced", probe="; ".join(notes),
                            net_isolated=True, data_dir_protected=prot == "protected")
                if method == "unshare":
                    base["process_tree_kill"] = "PID namespace (unshare --pid --kill-child)"
                    base["filesystem"] = ("not enforced outside jev-data: jev-data, the other workspaces and /run "
                                          "are hidden, everything else your user can write is writable"
                                          if prot == "protected" else base["filesystem"])
                elif fs == "blocked":
                    base["fs_isolated"] = True
                    if method == "bwrap":
                        base["process_tree_kill"] = "PID namespace (bwrap)"
                        base["tmpfs_size_limit"] = bwrap_supports_size(_bwrap_version())
                        base["filesystem"] = ("read-only outside the workspace; home folder, /mnt, /media, /run, "
                                              "jev-data and other workspaces hidden" if self.hide_home else
                                              "read-only outside the workspace; /run, jev-data and other "
                                              "workspaces hidden")
                    else:
                        base["filesystem"] = "read-only outside the workspace"
                return self._finish_caps(base)
        base["probe"] = "; ".join(notes) or None
        if IS_LINUX and self._systemd_scope_ok():
            base["cgroup_scope"] = True
            base["process_tree_kill"] = "systemd --user scope (cgroup) + subreaper launcher"
        return self._finish_caps(base)

    @staticmethod
    def _bus_env() -> dict:
        """The only variables systemd-run needs to reach the user manager. Runs get them for the
        systemd-run process alone; the sandboxed program itself starts from `env -i` (see run())."""
        return {k: os.environ[k] for k in ("XDG_RUNTIME_DIR", "DBUS_SESSION_BUS_ADDRESS") if os.environ.get(k)}

    def _systemd_scope_ok(self) -> bool:
        """True only if a scope really starts the way runs will start it: same minimal environment,
        child launched through `env -i`. WSL without systemd, CI runners and containers have
        systemd-run but no user bus ("Failed to connect to bus"); those fall back to the subreaper."""
        exe, env_exe = self._exe("systemd-run"), self._exe("env")
        if not exe or not env_exe:
            return False
        bus = self._bus_env()
        if "XDG_RUNTIME_DIR" not in bus:
            return False
        try:
            r = subprocess.run([exe, "--user", "--scope", "--quiet", "--collect", "--", env_exe, "-i", "true"],
                               capture_output=True, timeout=10, env={"PATH": "/usr/bin:/bin", **bus})
            return r.returncode == 0 and b"Failed to connect" not in r.stderr
        except (OSError, subprocess.TimeoutExpired):
            return False

    def _probe(self, method: str) -> str:
        with tempfile.TemporaryDirectory(prefix="jev-probe-") as d:
            data = Path(d) / DATA_DIRNAME
            ws = data / "workspaces" / "ws"
            ws.mkdir(parents=True)  # wsfs-ok: probe temp folder
            outside = Path(d) / "outside"
            outside.mkdir()  # wsfs-ok: probe temp folder
            argv, fds = self._wrap(method, [sys.executable, "-I", "-S", "-c", PROBE, str(ws), str(outside), str(data)],
                                   ws, network=False, mounts=Mounts(), cwd=ws, limits=self.limits, hide_extra=[data])
            try:
                r = subprocess.run(argv, capture_output=True, text=True, timeout=20, cwd=ws, env=scrubbed_env(ws),
                                   pass_fds=fds)
            except (OSError, subprocess.TimeoutExpired) as e:
                return f"failed ({type(e).__name__})"
            finally:
                for fd in fds:
                    os.close(fd)
            out = (r.stdout or "").strip().splitlines()
            if not (out and out[-1].startswith("JEVPROBE") and (ws / "probe.txt").exists()):
                return f"failed (exit {r.returncode}: {(r.stderr or '').strip()[:160]})"
            # the escape checks look from outside: did writes next to the workspace really land?
            fs = "writable" if (outside / "jev-escape-probe.txt").exists() else "blocked"
            prot = "exposed" if (data / "jev-escape-probe.txt").exists() else "protected"
            return f"{out[-1]} {fs} {prot}"

    # ------------------------------------------------------------------ what each run hides
    def _protected(self, ws: Path, extra=()) -> list[Path]:
        """jev-data (settings.json, jev.db, venvs, other workspaces) and the workspace root."""
        out = [Path(p) for p in extra]
        for p in (self.data_dir, self.workspace_root):
            if p is not None:
                out.append(p)
        if ws.parent.parent.name == DATA_DIRNAME:          # the default layout, even without settings
            out.append(ws.parent.parent)
        res = []
        for p in out:
            p = Path(p).resolve()
            if p == Path("/") or p == ws:
                continue
            if p.is_dir() and not p.is_symlink() and p not in res:
                res.append(p)
        return res

    def _keep_paths(self, extra=()) -> list[Path]:
        """Interpreter installs to re-expose read-only when they sit inside a hidden folder. A path
        that is, or contains, your home folder is never re-exposed whole (L5)."""
        home = _home()
        keep = [Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix,
                                            os.path.dirname(os.path.realpath(sys.executable)))]
        node = shutil.which("node")
        if node:
            real = Path(node).resolve()
            prefix = real.parent.parent
            if home is not None and _under(home, prefix):
                keep.append(real)          # the prefix is (or contains) your home: only the binary itself
            else:
                keep.append(prefix)
        # CUDA and other library folders the environment points at, and the interpreter's import path
        for var in ("CUDA_HOME", "CUDA_PATH", "CUDA_ROOT"):
            if os.environ.get(var):
                keep.append(Path(os.environ[var]).resolve())
        for var in ("LD_LIBRARY_PATH",):
            for d in (os.environ.get(var) or "").split(os.pathsep):
                if d:
                    keep.append(Path(d).resolve())
        for d in sys.path:                      # site folders outside the prefix (user / distro site-packages)
            if d and os.path.isdir(d) and Path(d).name in ("site-packages", "dist-packages") and \
                    not (home is not None and _under(Path(d).resolve(), home)):    # user site: -I ignores it
                keep.append(Path(d).resolve())
        for d in ("/opt/cuda", "/usr/local/cuda"):
            if os.path.isdir(d):
                keep.append(Path(d).resolve())
        keep += [Path(p).resolve() for p in extra]
        out = []
        for p in keep:
            if home is not None and _under(home, p) and p.is_dir():
                continue
            if p not in out:
                out.append(p)
        return out

    # ------------------------------------------------------------------ wrappers
    def _wrap(self, method: str, argv: list[str], ws: Path, network: bool, mounts: Mounts | None = None,
              cwd: Path | None = None, limits: Limits | None = None, hide_extra=()) -> tuple[list[str], list[int]]:
        mounts = mounts or Mounts()
        cwd = Path(cwd or ws)
        lim = limits or self.limits
        if method == "bwrap":
            return self._wrap_bwrap(argv, ws, network, mounts, cwd, lim, hide_extra)
        if method == "unshare":
            return self._wrap_unshare(argv, ws, network, mounts, cwd, hide_extra), []
        if method == "sandbox-exec":
            hide = self._protected(ws, hide_extra)
            exec_paths = self._keep_paths([*mounts.ro])
            shell = Path(argv[0]).name.lower() in ("bash", "sh", "zsh", "dash", "pwsh")
            prof = macos_profile(ws, network, read_only=[*self._keep_paths(), *mounts.ro],
                                 writable=[*mounts.rw], hide=hide, exec_paths=exec_paths, shell=shell)
            return [self._exe("sandbox-exec") or "sandbox-exec", "-p", prof] + argv, []
        return argv, []

    def _hidden_dirs(self, ws: Path, hide_extra=()) -> list[Path]:
        hidden: list[Path] = []
        for m in ("/run", "/var/run"):
            if os.path.isdir(m) and not os.path.islink(m):
                hidden.append(Path(m))
        if os.path.isdir("/mnt/wsl") and not os.path.islink("/mnt/wsl"):
            hidden.append(Path("/mnt/wsl"))
        if self.hide_home:
            # your home folder and mounted drives (on WSL, /mnt/c is your Windows drive)
            for m in ("/mnt", "/media", "/run/media"):
                if os.path.isdir(m) and not os.path.islink(m):
                    hidden.append(Path(m))
            home = _home()
            if home is not None and home.is_dir():
                hidden.append(home)
        hidden += self._protected(ws, hide_extra)
        out: list[Path] = []
        for h in sorted(set(hidden), key=lambda p: (len(p.parts), str(p))):
            if h == ws:
                continue
            if any(_under(h, o) for o in out):
                continue                       # already hidden by a parent
            out.append(h)
        return out

    def _bound_roots(self) -> tuple[list[tuple[str, str]], list[Path]]:
        """The allowlisted root: (bwrap ops for the system folders, the real folders they show).
        Top-level symlinks (merged /usr: /bin -> usr/bin) are recreated as symlinks."""
        ops, real = [], []
        for top in ROOT_ALLOW:
            if os.path.islink(top):
                ops.append(("--symlink", os.readlink(top), top))
            elif os.path.isdir(top):
                ops.append(("--ro-bind", top, top))
                real.append(Path(top).resolve())
        if not self.hide_home:                        # you chose to let code read your home folder and drives
            for extra in [_home(), Path("/mnt"), Path("/media")]:
                if extra is not None and extra.is_dir() and not extra.is_symlink():
                    ops.append(("--ro-bind", str(extra), str(extra)))
                    real.append(extra.resolve())
        return ops, real

    def _wrap_bwrap(self, argv, ws, network, mounts, cwd, lim, hide_extra) -> tuple[list[str], list[int]]:
        # The new root is an empty tmpfs (read-only at the end). Only system folders (ROOT_ALLOW) are
        # bound into it, plus what this run needs (the interpreter installs, FFmpeg / speech engines,
        # CUDA folders, the workspace, its scratch). /var, /opt, /srv, /home, /root, /mnt, /media,
        # /run, /tmp and any other top-level folder are simply absent, so Unix sockets there are
        # unreachable whatever network namespace their listener runs in.
        a = [self._exe("bwrap") or "bwrap", "--die-with-parent", "--new-session", "--unshare-pid", "--unshare-ipc",
             "--unshare-uts", "--unshare-cgroup-try"]
        root_ops, bound = self._bound_roots()
        for op in root_ops:
            a += list(op)
        a += ["--dev", "/dev", "--proc", "/proc"]

        def visible(p: Path) -> bool:
            return any(_under(p, r) for r in bound)
        sized = bwrap_supports_size(_bwrap_version())
        tmp_bytes = max(64, int(lim.memory_mb)) << 20
        late_ro = [Path("/dev")]
        # /tmp and /dev/shm: size-capped tmpfs (bwrap >= 0.10) or folders inside the workspace, which
        # count toward max_workspace_mb; never an unbounded RAM-backed tmpfs
        if sized:
            a += ["--size", str(tmp_bytes), "--tmpfs", "/tmp", "--size", str(tmp_bytes), "--tmpfs", "/dev/shm"]
        else:
            from .wsfs import ensure_dir
            ws_tmp = ensure_dir(ws, ".tmp")           # never bind a planted link onto /tmp
            shm = ensure_dir(ws, ".tmp/.shm")
            if _under(ws, Path("/tmp")):
                a += ["--tmpfs", "/tmp"]
                late_ro.append(Path("/tmp"))
            else:
                a += ["--bind", str(ws_tmp), "/tmp"]
            a += ["--bind", str(shm), "/dev/shm"]
        ro = list(self._keep_paths())
        if network:
            # a proxy CA bundle or resolv.conf's target (often under /run) must stay reachable
            for k in NET_FILE_ENV:
                v = os.environ.get(k)
                if v and os.path.exists(v):
                    ro.append(Path(v).resolve())
            if os.path.exists("/etc/resolv.conf"):
                ro.append(Path("/etc/resolv.conf").resolve())
        protected = self._hidden_dirs(ws, hide_extra)
        extra: list[Path] = []
        for p in sorted({p for p in ro if p.exists()}, key=lambda p: (len(p.parts), str(p))):
            if not (visible(p) or any(_under(p, d) for d in extra)):
                extra.append(p)
        for p in extra:
            a += ["--ro-bind", str(p), str(p)]
        bound += extra
        a += ["--dir", "/run"]
        for keep in RUN_KEEP:                          # NixOS: the system profile and the GPU driver libraries
            if os.path.islink(keep):
                a += ["--symlink", os.readlink(keep), keep]
            elif os.path.isdir(keep):
                a += ["--ro-bind", keep, keep]
        for dev in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset",
                    "/dev/dxg", "/dev/dri", "/dev/kfd", *[f"/dev/nvidia{i}" for i in range(8)]):
            if os.path.exists(dev):
                a += ["--dev-bind", dev, dev]
        # folders that must stay hidden though a bound folder contains them (jev-data inside a bound
        # home when hide_home is off, a data folder under /usr, ...); a runtime inside one stays visible
        hidden = [h for h in protected if visible(h) and h != Path("/run")]
        for h in hidden:
            a += ["--tmpfs", str(h)]
        for p in extra:
            if any(_under(p, h) for h in hidden):
                a += ["--ro-bind", str(p), str(p)]
        # defence in depth: sockets listening (in this network namespace) inside a bound folder
        for sock in listening_unix_sockets():
            sp = Path(sock)
            if visible(sp) and not any(_under(sp, h) for h in hidden) and not _under(sp, ws):
                a += ["--ro-bind", "/dev/null", sock]
        for p in mounts.ro:                          # always explicit: they may sit in a hidden or private folder
            p = Path(p).resolve()
            if p.exists():
                a += ["--ro-bind", str(p), str(p)]
        for p in mounts.rw:
            p = Path(p).resolve()
            if p.exists():
                a += ["--bind", str(p), str(p)]
        a += ["--bind" if mounts.workspace_writable else "--ro-bind", str(ws), str(ws)]
        for h in hidden + late_ro:
            a += ["--remount-ro", str(h)]
        a += ["--remount-ro", "/"]                  # the root tmpfs itself: nothing can be created in it
        a += ["--chdir", str(cwd)]
        if not network:
            a += ["--unshare-net"]
        fds: list[int] = []
        prog = seccomp_program()
        if prog:
            r, wfd = os.pipe()
            os.write(wfd, prog)
            os.close(wfd)
            fds.append(r)
            a += ["--seccomp", str(r)]
        return a + ["--"] + argv, fds

    def _wrap_unshare(self, argv, ws, network, mounts, cwd, hide_extra) -> list[str]:
        hide = [str(h) for h in self._protected(ws, hide_extra)]
        for m in ("/run", "/var/run", "/mnt/wsl"):
            if os.path.isdir(m) and not os.path.islink(m):
                hide.append(m)
        links = [[os.readlink(k), k] for k in RUN_KEEP if os.path.islink(k)]
        ro = [str(Path(p).resolve()) for p in mounts.ro]
        if network:
            for k in NET_FILE_ENV:
                v = os.environ.get(k)
                if v and os.path.exists(v):
                    ro.append(str(Path(v).resolve()))
            if os.path.exists("/etc/resolv.conf"):
                ro.append(str(Path("/etc/resolv.conf").resolve()))
        spec = {"hide": sorted(set(hide), key=lambda p: (len(Path(p).parts), p)), "ro": ro,
                "rw": [str(Path(p).resolve()) for p in mounts.rw], "ws": str(ws),
                "ws_rw": bool(mounts.workspace_writable), "cwd": str(cwd), "symlinks": links,
                "seccomp": (seccomp_program() or b"").hex(),
                "mask": [p for p in listening_unix_sockets() if not _under(Path(p), ws)
                         and not any(_under(Path(p), Path(h)) for h in hide)]}
        ns = [self._exe("unshare") or "unshare", "--user", "--map-root-user", "--mount", "--pid", "--fork", "--kill-child", "--mount-proc"]
        if not network:
            ns.append("--net")
        return ns + ["--", sys.executable, "-I", "-S", "-c", PRELUDE, json.dumps(spec)] + argv

    # ------------------------------------------------------------------ workspace size
    def workspace_over(self, ws: Path, limits: Limits | None = None) -> str | None:
        """An error text when the workspace is over max_workspace_mb or holds more than
        max_workspace_entries files and folders (the walk stops there, so a huge tree is refused,
        never measured partially); None otherwise."""
        lim = limits or self.limits
        cap = int(lim.max_workspace_mb) << 20
        used, entries, complete = workspace_usage(Path(ws), int(lim.max_workspace_entries))
        if not complete:
            return (f"workspace entry limit reached: the workspace holds more than {lim.max_workspace_entries} "
                    f"files and folders, over limits.max_workspace_entries = {lim.max_workspace_entries}. Delete "
                    "files (.tmp, caches, generated folders) before running more code")
        if used > cap:
            return (f"workspace size limit reached: the workspace uses {used / 2**20:.0f} MB, over limits."
                    f"max_workspace_mb = {lim.max_workspace_mb} MB. Delete files (large outputs, .tmp, caches) "
                    "before running more code")
        return None

    # ------------------------------------------------------------------ run
    def spawn(self, argv: list[str], workspace: Path, limits: Limits | None = None, network: bool = False, *,
              extra_env: dict | None = None, cwd: Path | None = None, mounts: Mounts | None = None,
              stdin: bool = False, honor_network_setting: bool = True, venv: Path | None = None,
              check_workspace: bool = True, text: bool = False, tmp: Path | None = None) -> Spawned:
        """Start argv inside the sandbox and return at once (for streaming callers such as FFmpeg
        frame pipes). Raises SandboxError when the run cannot be started safely."""
        if isinstance(argv, str):
            raise TypeError("argv must be a list")
        ws = Path(workspace).resolve()
        ws.mkdir(parents=True, exist_ok=True)  # wsfs-ok: the workspace folder itself (resolved, created by the backend)
        lim = limits or self.limits
        if check_workspace and (over := self.workspace_over(ws, lim)):
            raise SandboxError(over)
        run_cwd = Path(cwd).resolve() if cwd else ws
        allowed = [ws, *[Path(p).resolve() for p in ((mounts.rw if mounts else []) + (mounts.ro if mounts else []))]]
        if not any(_under(run_cwd, a) for a in allowed):
            raise SandboxError(f"cwd {run_cwd} is outside the workspace")
        caps = self.capabilities()
        if honor_network_setting:
            network = network or self.network_allowed
        method = caps["method"]
        mounts = mounts or Mounts()
        full = list(argv)
        fds: list[int] = []
        unit = None
        import secrets as _secrets
        token = _secrets.token_hex(12)
        env = scrubbed_env(ws, extra_env, network=network, venv=venv, tmp=tmp)
        env["JEV_RUN_ID"] = token          # lets the end-of-run sweep find descendants that escaped the tree
        if method in ("bwrap", "unshare", "sandbox-exec"):
            full, fds = self._wrap(method, full, ws, network, mounts, run_cwd, lim)
        elif IS_LINUX and caps.get("cgroup_scope"):
            unit = f"jev-run-{_secrets.token_hex(6)}"
            # systemd-run itself needs the user-bus variables; the program must not see them, so it
            # starts from `env -i` with exactly the scrubbed environment (no secrets in it to leak via argv)
            full = ([self._exe("systemd-run") or "systemd-run", "--user", "--scope", "--quiet", "--collect",
                     "--unit", unit, "-p", f"MemoryMax={int(lim.memory_mb)}M", "-p", "TasksMax=512", "--",
                     self._exe("env") or "/usr/bin/env", "-i"]
                    + [f"{k}={v}" for k, v in sorted(env.items()) if "\0" not in v and "=" not in k]
                    + full)
            env = {**env, **self._bus_env()}
        if not IS_WIN:
            nproc = _user_process_count() + 256
            kind = 0 if IS_LINUX else 1
            spec = f"{lim.cpu_seconds},{lim.memory_mb << 20},{lim.max_output_mb << 20},{nproc},512,{kind}"
            full = [sys.executable, "-I", "-S", "-c", LAUNCHER, spec] + full
        job = None
        kw: dict = {}
        if IS_WIN:
            kw["creationflags"] = 0x00000200 | 0x08000000 | 0x00000004   # NEW_PROCESS_GROUP, NO_WINDOW, SUSPENDED
            try:
                job = self._win_job_create(64, lim.memory_mb, lim.cpu_seconds)
            except Exception as e:  # noqa: BLE001
                raise SandboxError(f"refused: the Windows Job Object that limits and kills runs could not be "
                                   f"created ({e})") from e
        else:
            kw["start_new_session"] = True
        try:
            proc = subprocess.Popen(full, cwd=run_cwd if method != "bwrap" else ws, env=env,
                                    stdin=subprocess.PIPE if stdin else subprocess.DEVNULL, stdout=subprocess.PIPE,
                                    stderr=subprocess.PIPE, pass_fds=fds, text=text,
                                    **({"encoding": "utf-8", "errors": "replace"} if text else {}), **kw)
        except OSError as e:
            if job:
                job.close()
            raise SandboxError(f"could not start {argv[0]}: {e}") from e
        finally:
            for fd in fds:
                try:
                    os.close(fd)
                except OSError:
                    pass
        if job:
            try:
                job.adopt(proc)
            except Exception as e:  # noqa: BLE001
                try:
                    proc.kill()
                except OSError:
                    pass
                job.close()
                raise SandboxError(f"refused: the process could not be put into its Job Object ({e})") from e
        sp = Spawned(proc, self, ws, lim, method, job, unit, check_workspace, token)
        sp._next_size = time.monotonic() + 0.5
        return sp

    def run(self, argv: list[str], workspace: Path, limits: Limits | None = None, network: bool = False,
            cancelled: Callable[[], bool] | None = None, extra_env: dict | None = None,
            stdin_text: str | None = None, *, mounts: Mounts | None = None, cwd: Path | None = None,
            venv: Path | None = None, honor_network_setting: bool = True, tmp: Path | None = None) -> RunResult:
        """Run argv (a list, never a shell string) inside `workspace` with limits and isolation."""
        if isinstance(argv, str):
            raise TypeError("argv must be a list")
        ws = Path(workspace).resolve()
        ws.mkdir(parents=True, exist_ok=True)  # wsfs-ok: the workspace folder itself (resolved, created by the backend)
        lim = limits or self.limits
        method = self.capabilities()["method"]
        before = snapshot(ws)
        t0 = time.monotonic()
        try:
            sp = self.spawn(argv, ws, lim, network, extra_env=extra_env, cwd=cwd, mounts=mounts,
                            stdin=stdin_text is not None, honor_network_setting=honor_network_setting, venv=venv,
                            tmp=tmp)
        except SandboxError as e:
            return RunResult(None, "", str(e), 0.0, isolation=method, refused=str(e))
        proc = sp.proc
        bufs = {"out": bytearray(), "err": bytearray()}
        dropped = {"out": 0, "err": 0}

        def reader(stream, key):
            try:
                while chunk := stream.read(65536):
                    room = STREAM_CAP - len(bufs[key])
                    if room > 0:
                        bufs[key] += chunk[:room]
                    dropped[key] += max(0, len(chunk) - max(room, 0))
            except (OSError, ValueError):
                pass
        threads = [threading.Thread(target=reader, args=(proc.stdout, "out"), daemon=True),
                   threading.Thread(target=reader, args=(proc.stderr, "err"), daemon=True)]
        for t in threads:
            t.start()
        if stdin_text is not None:
            try:
                proc.stdin.write(stdin_text.encode())
                proc.stdin.close()
            except OSError:
                pass
        timed_out = was_cancelled = False
        deadline = t0 + lim.wall_seconds
        sp.guard()
        while True:
            try:
                proc.wait(timeout=0.05)
                break
            except subprocess.TimeoutExpired:
                if sp.guard():
                    break                     # over max_workspace_mb: guard() killed the run
                if cancelled and cancelled():
                    was_cancelled = True
                elif time.monotonic() > deadline:
                    timed_out = True
                else:
                    continue
                sp.kill()
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                break
        over = sp.finish()                # also kills stray background children of a finished run
        for t in threads:
            t.join(timeout=2)
        for s in (proc.stdout, proc.stderr):
            try:
                s.close()
            except OSError:
                pass
        seconds = time.monotonic() - t0
        rc = proc.returncode
        limit_hit = None
        if method == "bwrap" and rc is not None and 128 < rc < 160:
            rc = -(rc - 128)       # bwrap reports a child killed by signal N as exit 128+N
        if not IS_WIN and rc is not None and rc < 0:
            sig = -rc
            limit_hit = {signal.SIGXCPU: "CPU time limit reached", signal.SIGKILL: "killed",
                         signal.SIGSEGV: "crashed (segmentation fault)"}.get(sig, f"signal {sig}")
            if sig == signal.SIGKILL and seconds >= lim.cpu_seconds and not timed_out:
                limit_hit = "CPU time limit reached"
        stderr = bufs["err"].decode("utf-8", "replace")
        if "MemoryError" in stderr or "Cannot allocate memory" in stderr or "out of memory" in stderr.lower():
            limit_hit = limit_hit or f"memory limit ({lim.memory_mb} MB) reached"
        if "File too large" in stderr:
            limit_hit = limit_hit or f"file size limit ({lim.max_output_mb} MB) reached"
        if "No space left on device" in stderr:
            limit_hit = limit_hit or "temporary space (/tmp, /dev/shm) is full"
        after = snapshot(ws)
        changed = sorted(k for k, v in after.items() if before.get(k) != v and not k.startswith("probe.txt"))
        out = bufs["out"].decode("utf-8", "replace")
        if dropped["out"]:
            out += f"\n[... {dropped['out']} more bytes of output dropped (cap {STREAM_CAP >> 10} KB) ...]"
        if dropped["err"]:
            stderr += f"\n[... {dropped['err']} more bytes dropped ...]"
        return RunResult(rc, out, stderr, seconds, timed_out, was_cancelled, bool(dropped["out"] or dropped["err"]),
                         limit_hit, changed, method, workspace_over=over)

    @staticmethod
    def _track(sp: "Spawned") -> None:
        try:
            import psutil
            for k in psutil.Process(sp.proc.pid).children(recursive=True):
                sp._tracked.setdefault(k.pid, k.create_time())
        except Exception:  # noqa: BLE001 - the launcher already exited
            pass

    @staticmethod
    def _sweep(sp: "Spawned") -> None:
        """Without a PID namespace or cgroup: kill every process seen in this run's tree (pid +
        start time checked) and every process of this user still carrying the run's JEV_RUN_ID."""
        try:
            import psutil
        except ImportError:
            return
        me = os.getpid()
        for pid, ct in list(sp._tracked.items()):
            try:
                p = psutil.Process(pid)
                if abs(p.create_time() - ct) < 0.01:
                    p.kill()
            except Exception:  # noqa: BLE001
                pass
        if not sp.token:
            return
        mark = sp.token
        uid = os.getuid() if hasattr(os, "getuid") else None
        for p in psutil.process_iter(["pid", "uids"]):
            try:
                if p.pid == me or (uid is not None and p.info["uids"] and p.info["uids"].real != uid):
                    continue
                if p.environ().get("JEV_RUN_ID") == mark:
                    p.kill()
            except Exception:  # noqa: BLE001
                continue

    def _kill_tree(self, proc: subprocess.Popen, job, unit: str | None = None, sp: "Spawned | None" = None) -> None:
        if sp is not None and sp.method not in ("bwrap", "unshare"):
            self._track(sp)
        if job:
            job.terminate()
        if IS_WIN:
            if proc.poll() is None:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
            return
        if unit:
            try:
                subprocess.run(["systemctl", "--user", "kill", "--signal=SIGKILL", f"{unit}.scope"],
                               capture_output=True, timeout=10)
            except (OSError, subprocess.TimeoutExpired):
                pass
        kids = []
        try:
            import psutil
            kids = psutil.Process(proc.pid).children(recursive=True)
        except Exception:  # noqa: BLE001
            pass
        try:
            os.killpg(proc.pid, signal.SIGKILL)
        except (ProcessLookupError, PermissionError):
            pass
        for k in kids:
            try:
                k.kill()
            except Exception:  # noqa: BLE001
                pass
        if sp is not None and sp.method not in ("bwrap", "unshare"):
            self._sweep(sp)

    # ------------------------------------------------------------------ Windows job object
    @staticmethod
    def _win_job_create(max_procs: int, memory_mb: int, cpu_seconds: int):
        return _WinJob(max_procs, memory_mb, cpu_seconds)


_SANDBOXES: dict = {}
_SANDBOXES_LOCK = threading.Lock()


def sandbox_for(settings: dict | None) -> Sandbox:
    """The Sandbox a tool context uses: settings["_sandbox"] when the runtime set one, else one
    shared Sandbox per isolation mode (its probe runs once)."""
    s = settings or {}
    sb = s.get("_sandbox")
    if sb is not None:
        return sb
    key = json.dumps([(s.get("sandbox") or {}).get("isolation", "auto"), (s.get("sandbox") or {}).get("hide_home", True),
                      s.get("data_dir"), s.get("workspace_root"), bool(s.get("network_allowed"))])
    with _SANDBOXES_LOCK:
        if key not in _SANDBOXES:
            _SANDBOXES[key] = Sandbox(s)
        return _SANDBOXES[key]


class _WinJob:
    """Windows Job Object: the child starts suspended, joins the job, then resumes. Closing the job
    handle kills every process in it (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE). If the process cannot be
    assigned to the job it is terminated, never resumed."""

    def __init__(self, max_procs: int, memory_mb: int, cpu_seconds: int):
        import ctypes
        from ctypes import wintypes

        self.ct = ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k32 = k32
        self.ntdll = ctypes.WinDLL("ntdll")

        class IoCounters(ctypes.Structure):
            _fields_ = [(n, ctypes.c_ulonglong) for n in ("r_ops", "w_ops", "o_ops", "r_bytes", "w_bytes", "o_bytes")]

        class Basic(ctypes.Structure):
            _fields_ = [("PerProcessUserTimeLimit", ctypes.c_int64), ("PerJobUserTimeLimit", ctypes.c_int64),
                        ("LimitFlags", wintypes.DWORD), ("MinimumWorkingSetSize", ctypes.c_size_t),
                        ("MaximumWorkingSetSize", ctypes.c_size_t), ("ActiveProcessLimit", wintypes.DWORD),
                        ("Affinity", ctypes.c_size_t), ("PriorityClass", wintypes.DWORD),
                        ("SchedulingClass", wintypes.DWORD)]

        class Extended(ctypes.Structure):
            _fields_ = [("BasicLimitInformation", Basic), ("IoInfo", IoCounters),
                        ("ProcessMemoryLimit", ctypes.c_size_t), ("JobMemoryLimit", ctypes.c_size_t),
                        ("PeakProcessMemoryUsed", ctypes.c_size_t), ("PeakJobMemoryUsed", ctypes.c_size_t)]

        k32.CreateJobObjectW.restype = wintypes.HANDLE
        h = k32.CreateJobObjectW(None, None)
        if not h:
            raise OSError(ctypes.get_last_error(), "CreateJobObjectW failed")
        self.h = h
        info = Extended()
        # KILL_ON_JOB_CLOSE | JOB_MEMORY | ACTIVE_PROCESS | JOB_TIME | DIE_ON_UNHANDLED_EXCEPTION
        info.BasicLimitInformation.LimitFlags = 0x2000 | 0x200 | 0x8 | 0x4 | 0x400
        info.BasicLimitInformation.ActiveProcessLimit = max_procs
        info.BasicLimitInformation.PerJobUserTimeLimit = int(cpu_seconds) * 10_000_000
        info.JobMemoryLimit = int(memory_mb) << 20
        if not k32.SetInformationJobObject(wintypes.HANDLE(h), 9, ctypes.byref(info), ctypes.sizeof(info)):
            err = ctypes.get_last_error()
            k32.CloseHandle(wintypes.HANDLE(h))
            raise OSError(err, "SetInformationJobObject failed")

    def adopt(self, proc: subprocess.Popen) -> None:
        """Put the suspended process into the job, then resume it. If assigning fails the process is
        terminated and OSError raised: it never runs outside the job."""
        handle = int(proc._handle)  # noqa: SLF001 - Popen exposes no public handle
        ok = self.k32.AssignProcessToJobObject(self.ct.c_void_p(self.h), self.ct.c_void_p(handle))
        if not ok:
            err = self.ct.get_last_error() if hasattr(self.ct, "get_last_error") else 0
            self.k32.TerminateProcess(self.ct.c_void_p(handle), 1)
            raise OSError(err, "AssignProcessToJobObject failed")
        status = self.ntdll.NtResumeProcess(self.ct.c_void_p(handle))
        if status not in (0, None):
            self.k32.TerminateProcess(self.ct.c_void_p(handle), 1)
            raise OSError(status, "NtResumeProcess failed")

    def terminate(self) -> None:
        if self.h:
            self.k32.TerminateJobObject(self.ct.c_void_p(self.h), 1)

    def close(self) -> None:
        if self.h:
            self.k32.CloseHandle(self.ct.c_void_p(self.h))
            self.h = None
