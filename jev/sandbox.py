"""Run model-written code as a contained subprocess inside the conversation's workspace.

What every run gets, on every OS:
  * cwd = the workspace; HOME / USERPROFILE / TMPDIR point inside it
  * a scrubbed environment built from scratch (PATH and a few OS essentials; nothing whose name
    looks like a token, key, password or session, so JEPA_STUDIO_TOKEN is never visible)
  * a wall-clock timeout; on timeout or cancel the whole process tree is killed
  * stdout / stderr captured and capped (STREAM_CAP bytes each, the rest is counted and dropped)
  * no shell: argv lists only (run_shell passes its text to `bash -c` explicitly, after approval)

POSIX (Linux, macOS) adds resource limits, set by a tiny launcher that calls setrlimit() and then
exec()s the program (subprocess's preexec_fn is not safe in a threaded server):
  RLIMIT_CPU (cpu_seconds), memory (RLIMIT_DATA on Linux, which counts heap and anonymous memory but
  not the large address-space reservations V8 and CUDA make; RLIMIT_AS elsewhere; macOS does not
  enforce either), RLIMIT_FSIZE (max_output_mb per file, SIGXFSZ ignored so writes fail with EFBIG),
  RLIMIT_NPROC (your current process count + 256: a fork-bomb brake), RLIMIT_NOFILE (512),
  RLIMIT_CORE (0). The process runs in its own session; on timeout every descendant found is killed.

Isolation layers, detected by a probe at startup (Sandbox.capabilities() reports what the probe
actually observed, never what we hope):
  Linux  bwrap (bubblewrap): new network namespace (only `lo`, which is not the host's loopback, so
         even the JEPA Studio backend is unreachable), new PID namespace (the whole tree dies with
         it), the file system read-only except the workspace, /tmp private, your home folder hidden
         (the Python install is re-exposed read-only), docker/podman sockets masked.
         unshare --user --net: network namespace only (when bwrap is missing or blocked).
  macOS  sandbox-exec with a profile denying network and file writes outside the workspace
         (Apple marks sandbox-exec deprecated; the probe checks it still works).
  Windows  a Job Object (kill-on-close, job memory limit, CPU time, max 64 processes). No network
         or file-system isolation: docs/jev/SAFETY.md says what that means.
"""
from __future__ import annotations

import os
import re
import shutil
import signal
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
IS_WIN = os.name == "nt"
IS_MAC = sys.platform == "darwin"

# Sets rlimits then execs argv[2:]. Runs with `python -I -S` so nothing from the workspace is imported.
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
os.execvp(sys.argv[2], sys.argv[2:])
"""

PROBE = r"""
import os, socket, sys
ws = sys.argv[1]
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
try:
    open(os.path.join(os.path.dirname(ws), "jev-escape-probe.txt"), "w").write("x")
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

    @classmethod
    def from_settings(cls, s: dict | None) -> "Limits":
        d = dict((s or {}).get("limits") or {})
        return cls(**{k: int(v) for k, v in d.items() if k in cls.__dataclass_fields__})


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

    @property
    def ok(self) -> bool:
        return self.returncode == 0 and not self.timed_out and not self.cancelled

    def summary(self, max_chars: int = 3500) -> str:
        head = f"exit code {self.returncode}" if self.returncode is not None else "killed"
        if self.timed_out:
            head = f"stopped: wall-clock limit reached after {self.seconds:.0f} s"
        elif self.cancelled:
            head = "cancelled"
        elif self.limit_hit:
            head += f" ({self.limit_hit})"
        parts = [head + f", {self.seconds:.1f} s"]
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


def scrubbed_env(workspace: Path, extra: dict | None = None, network: bool = False) -> dict:
    env = {}
    for k in PASS_ENV + (NET_ENV if network else ()):
        v = os.environ.get(k)
        if v is not None and not SECRET_RE.search(k):
            env[k] = v
    env.setdefault("PATH", os.defpath)
    tmp = workspace / ".tmp"
    tmp.mkdir(parents=True, exist_ok=True)
    threads = str(max(1, min(4, os.cpu_count() or 1)))
    env.update({"HOME": str(workspace), "USERPROFILE": str(workspace), "TMPDIR": str(tmp), "TEMP": str(tmp),
                "TMP": str(tmp), "LANG": "C.UTF-8", "LC_ALL": "C.UTF-8", "PYTHONIOENCODING": "utf-8",
                "PYTHONDONTWRITEBYTECODE": "1", "PYTHONUNBUFFERED": "1", "MPLBACKEND": "Agg",
                "MPLCONFIGDIR": str(tmp / "mpl"), "OMP_NUM_THREADS": threads, "OPENBLAS_NUM_THREADS": threads,
                "MKL_NUM_THREADS": threads, "NO_COLOR": "1", "PIP_DISABLE_PIP_VERSION_CHECK": "1",
                "PIP_NO_INPUT": "1", "npm_config_update_notifier": "false"})
    venv = workspace / ".venv"
    if venv.is_dir():
        bindir = venv / ("Scripts" if IS_WIN else "bin")
        env["VIRTUAL_ENV"] = str(venv)
        env["PATH"] = str(bindir) + os.pathsep + env["PATH"]
    for k, v in (extra or {}).items():
        if not SECRET_RE.search(k):
            env[k] = str(v)
    return env


def python_for(workspace: Path) -> str:
    """The workspace venv's interpreter when install_package made one, else this Python."""
    venv = workspace / ".venv"
    exe = venv / ("Scripts/python.exe" if IS_WIN else "bin/python")
    return str(exe) if exe.exists() else sys.executable


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


def _user_process_count() -> int:
    try:
        import psutil
        uid = os.getuid()
        return sum(1 for p in psutil.process_iter(["uids"]) if p.info["uids"] and p.info["uids"].real == uid)
    except Exception:  # noqa: BLE001
        return 512


def _sbpl_quote(s: str) -> str:
    return '"' + s.replace("\\", "\\\\").replace('"', '\\"') + '"'


class Sandbox:
    def __init__(self, settings: dict | None = None):
        s = settings or {}
        sb = s.get("sandbox") or {}
        self.mode = sb.get("isolation", "auto")
        self.hide_home = bool(sb.get("hide_home", True))
        self.network_allowed = bool(s.get("network_allowed", False))
        self.limits = Limits.from_settings(s)
        self._caps: dict | None = None
        self._lock = threading.Lock()

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

    def _detect(self) -> dict:
        base = {"os": sys.platform, "method": "none", "network_isolation": "not available",
                "filesystem": "not enforced (code can read and write anything your user can)",
                "rlimits": "not available" if IS_WIN else
                ("cpu, data, fsize, nproc, nofile" if sys.platform.startswith("linux") else
                 "cpu, fsize, nproc, nofile (macOS does not enforce memory rlimits)"),
                "process_tree_kill": "job object" if IS_WIN else "session + descendant walk",
                "windows_job": False, "probe": None}
        if IS_WIN:
            try:
                self._win_job_create(64, 1, 1).close()
                base.update(windows_job=True, method="job-object",
                            rlimits="job object: memory, CPU time, 64 processes, kill on close")
            except Exception as e:  # noqa: BLE001
                base["probe"] = f"job object unavailable: {e}"
            return base
        order = {"auto": ["bwrap", "unshare", "sandbox-exec"], "none": []}.get(self.mode, [self.mode])
        notes = []
        for method in order:
            if method == "sandbox-exec" and not IS_MAC:
                continue
            if method in ("bwrap", "unshare") and not sys.platform.startswith("linux"):
                continue
            if not shutil.which(method):
                notes.append(f"{method}: not installed")
                continue
            res = self._probe(method)
            notes.append(f"{method}: {res}")
            if res and res.startswith("JEVPROBE"):
                _, ifs, net, fs = res.split()
                if net != "blocked":
                    continue
                base.update(method=method, network_isolation="enforced", probe="; ".join(notes))
                if fs == "blocked":
                    base["filesystem"] = ("read-only outside the workspace; home folder and /mnt, /media hidden"
                                          if method == "bwrap" and self.hide_home else
                                          "read-only outside the workspace")
                return base
        base["probe"] = "; ".join(notes) or None
        return base

    def _probe(self, method: str) -> str:
        with tempfile.TemporaryDirectory(prefix="jev-probe-") as d:
            ws = Path(d) / "ws"
            ws.mkdir()
            argv = self._wrap(method, [sys.executable, "-I", "-S", "-c", PROBE, str(ws)], ws, network=False)
            try:
                r = subprocess.run(argv, capture_output=True, text=True, timeout=20, cwd=ws,
                                   env=scrubbed_env(ws))
            except (OSError, subprocess.TimeoutExpired) as e:
                return f"failed ({type(e).__name__})"
            out = (r.stdout or "").strip().splitlines()
            if not (out and out[-1].startswith("JEVPROBE") and (ws / "probe.txt").exists()):
                return f"failed (exit {r.returncode}: {(r.stderr or '').strip()[:160]})"
            # the escape check looks from outside: did a write next to the workspace really land?
            fs = "writable" if (Path(d) / "jev-escape-probe.txt").exists() else "blocked"
            return f"{out[-1]} {fs}"

    # ------------------------------------------------------------------ wrappers
    def _wrap(self, method: str, argv: list[str], ws: Path, network: bool) -> list[str]:
        if method == "bwrap":
            a = ["bwrap", "--die-with-parent", "--unshare-pid", "--unshare-ipc", "--unshare-uts",
                 "--unshare-cgroup-try", "--ro-bind", "/", "/", "--dev", "/dev", "--proc", "/proc", "--tmpfs", "/tmp"]
            for dev in ("/dev/nvidiactl", "/dev/nvidia-uvm", "/dev/nvidia-uvm-tools", "/dev/nvidia-modeset",
                        "/dev/dxg", "/dev/dri", "/dev/kfd", *[f"/dev/nvidia{i}" for i in range(8)]):
                if os.path.exists(dev):
                    a += ["--dev-bind", dev, dev]
            for sock in ("/run/docker.sock", "/var/run/docker.sock", "/run/podman/podman.sock"):
                if os.path.exists(sock) and not os.path.islink(sock):
                    a += ["--ro-bind", "/dev/null", sock]
            run_user = f"/run/user/{os.getuid()}"
            if os.path.isdir(run_user):
                a += ["--tmpfs", run_user]
            if self.hide_home:
                # your home folder and mounted drives (on WSL, /mnt/c is your Windows drive) become
                # empty tmpfs; the Python / Node installs are re-exposed read-only, the workspace below
                hidden = [Path(m) for m in ("/mnt", "/media", "/run/media") if os.path.isdir(m) and not os.path.islink(m)]
                home = Path.home().resolve()
                if home != Path("/") and home.is_dir() and not any(h == home or h in home.parents for h in hidden):
                    hidden.append(home)
                for h in hidden:
                    a += ["--tmpfs", str(h)]
                keep = {Path(p).resolve() for p in (sys.prefix, sys.base_prefix, sys.exec_prefix,
                                                    os.path.dirname(os.path.realpath(sys.executable)))}
                node = shutil.which("node")
                if node:
                    keep.add(Path(node).resolve().parent.parent)
                for p in sorted(keep):
                    if p.is_dir() and any(p == h or h in p.parents for h in hidden):
                        a += ["--ro-bind", str(p), str(p)]
            a += ["--bind", str(ws), str(ws), "--chdir", str(ws)]
            if not network:
                a += ["--unshare-net"]
            return a + ["--"] + argv
        if method == "unshare":
            return argv if network else ["unshare", "--user", "--map-root-user", "--net", "--"] + argv
        if method == "sandbox-exec":
            rules = ["(version 1)", "(allow default)"]
            if not network:
                rules.append("(deny network*)")
            rules += ["(deny file-write*)",
                      "(allow file-write* (subpath " + _sbpl_quote(str(ws)) + ") (subpath \"/private/var/folders\")"
                      " (subpath \"/private/tmp\") (literal \"/dev/null\") (literal \"/dev/stdout\")"
                      " (literal \"/dev/stderr\") (literal \"/dev/dtracehelper\") (regex #\"^/dev/tty\"))"]
            return ["sandbox-exec", "-p", " ".join(rules)] + argv
        return argv

    # ------------------------------------------------------------------ run
    def run(self, argv: list[str], workspace: Path, limits: Limits | None = None, network: bool = False,
            cancelled: Callable[[], bool] | None = None, extra_env: dict | None = None,
            stdin_text: str | None = None) -> RunResult:
        """Run argv (a list, never a shell string) inside `workspace` with limits and isolation."""
        if isinstance(argv, str):
            raise TypeError("argv must be a list")
        ws = Path(workspace).resolve()
        ws.mkdir(parents=True, exist_ok=True)
        lim = limits or self.limits
        caps = self.capabilities()
        network = network or self.network_allowed
        method = caps["method"]
        full = list(argv)
        if method in ("bwrap", "unshare", "sandbox-exec"):
            full = self._wrap(method, full, ws, network)
        if not IS_WIN:
            nproc = _user_process_count() + 256
            kind = 0 if sys.platform.startswith("linux") else 1
            spec = f"{lim.cpu_seconds},{lim.memory_mb << 20},{lim.max_output_mb << 20},{nproc},512,{kind}"
            full = [sys.executable, "-I", "-S", "-c", LAUNCHER, spec] + full
        before = snapshot(ws)
        env = scrubbed_env(ws, extra_env, network=network)
        t0 = time.monotonic()
        job = None
        kw: dict = {}
        if IS_WIN:
            kw["creationflags"] = 0x00000200 | 0x08000000      # CREATE_NEW_PROCESS_GROUP | CREATE_NO_WINDOW
            try:
                job = self._win_job_create(64, lim.memory_mb, lim.cpu_seconds)
                kw["creationflags"] |= 0x00000004              # CREATE_SUSPENDED until it is in the job
            except Exception:  # noqa: BLE001
                job = None
        else:
            kw["start_new_session"] = True
        try:
            proc = subprocess.Popen(full, cwd=ws, env=env, stdin=subprocess.PIPE if stdin_text is not None
                                    else subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.PIPE, **kw)
        except OSError as e:
            if job:
                job.close()
            return RunResult(None, "", f"could not start {argv[0]}: {e}", 0.0, isolation=method)
        if job:
            job.adopt(proc)
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
        while True:
            try:
                proc.wait(timeout=0.1)
                break
            except subprocess.TimeoutExpired:
                if cancelled and cancelled():
                    was_cancelled = True
                elif time.monotonic() > deadline:
                    timed_out = True
                else:
                    continue
                self._kill_tree(proc, job)
                try:
                    proc.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    pass
                break
        if not (timed_out or was_cancelled):
            self._kill_tree(proc, job)   # stray background children of a finished run
        for t in threads:
            t.join(timeout=2)
        for s in (proc.stdout, proc.stderr):
            try:
                s.close()
            except OSError:
                pass
        if job:
            job.close()
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
        after = snapshot(ws)
        changed = sorted(k for k, v in after.items() if before.get(k) != v and not k.startswith("probe.txt"))
        out = bufs["out"].decode("utf-8", "replace")
        if dropped["out"]:
            out += f"\n[... {dropped['out']} more bytes of output dropped (cap {STREAM_CAP >> 10} KB) ...]"
        if dropped["err"]:
            stderr += f"\n[... {dropped['err']} more bytes dropped ...]"
        return RunResult(rc, out, stderr, seconds, timed_out, was_cancelled, bool(dropped["out"] or dropped["err"]),
                         limit_hit, changed, method)

    def _kill_tree(self, proc: subprocess.Popen, job) -> None:
        if job:
            job.terminate()
        if IS_WIN:
            if proc.poll() is None:
                subprocess.run(["taskkill", "/T", "/F", "/PID", str(proc.pid)], capture_output=True)
            return
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

    # ------------------------------------------------------------------ Windows job object
    @staticmethod
    def _win_job_create(max_procs: int, memory_mb: int, cpu_seconds: int):
        return _WinJob(max_procs, memory_mb, cpu_seconds)


class _WinJob:
    """Windows Job Object: the child starts suspended, joins the job, then resumes. Closing the job
    handle kills every process in it (JOB_OBJECT_LIMIT_KILL_ON_JOB_CLOSE)."""

    def __init__(self, max_procs: int, memory_mb: int, cpu_seconds: int):
        import ctypes
        from ctypes import wintypes

        self.ct = ctypes
        k32 = ctypes.WinDLL("kernel32", use_last_error=True)
        self.k32 = k32

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
        handle = int(proc._handle)  # noqa: SLF001 - Popen exposes no public handle
        try:
            self.k32.AssignProcessToJobObject(self.ct.c_void_p(self.h), self.ct.c_void_p(handle))
        finally:
            self.ct.WinDLL("ntdll").NtResumeProcess(self.ct.c_void_p(handle))

    def terminate(self) -> None:
        if self.h:
            self.k32.TerminateJobObject(self.ct.c_void_p(self.h), 1)

    def close(self) -> None:
        if self.h:
            self.k32.CloseHandle(self.ct.c_void_p(self.h))
            self.h = None
