"""Regression tests for the third review round of JEV (N9-N11 and the informational items).

Each test turns the reviewer's finding into a check that fails on the code it was reported
against. OS-specific ones skip where the feature is missing.
"""
import json
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
from jev.tools.base import REGISTRY, ToolContext, ToolError, load_all

load_all()
LINUX = sys.platform.startswith("linux")
PY = sys.executable
REPO = Path(__file__).resolve().parent.parent


def _bwrap() -> Sandbox:
    sb = Sandbox({"sandbox": {"isolation": "bwrap"}})
    if sb.capabilities()["method"] != "bwrap":
        pytest.skip("bubblewrap not available here")
    return sb


@pytest.fixture
def var_root():
    if not os.access("/var/tmp", os.W_OK):
        pytest.skip("/var/tmp not writable")
    d = Path(tempfile.mkdtemp(prefix="jev-r3-", dir="/var/tmp"))
    yield d
    shutil.rmtree(d, ignore_errors=True)


def _no_hang(fn, timeout=20.0):
    box = {}

    def go():
        try:
            fn()
            box["exc"] = None
        except BaseException as e:  # noqa: BLE001
            box["exc"] = e
    t = threading.Thread(target=go, daemon=True)
    t.start()
    t.join(timeout)
    assert not t.is_alive(), "the backend blocked on a FIFO"
    return box["exc"]


# ================================================================================================ N9
NS_LISTENER = r'''
import os, socket, sys, time
s = socket.socket(socket.AF_UNIX)
s.bind(sys.argv[1]); s.listen(5)
open(sys.argv[1] + ".ready", "w").write("1")
s.settimeout(30)
try:
    while True:
        c, _ = s.accept(); c.close()
except OSError:
    pass
'''


def test_n9_listener_in_another_network_namespace_is_unreachable(var_root):
    """Reviewer: a listener started under `unshare -n` on /var/tmp/ns.sock is invisible to
    /proc/net/unix of the host namespace, so masking missed it and the bwrap sandbox reached it."""
    if not LINUX:
        pytest.skip("Linux only")
    sb = _bwrap()
    unshare = shutil.which("unshare")
    if not unshare:
        pytest.skip("unshare missing")
    path = str(var_root / "ns.sock")
    cmd = [unshare, "-n", PY, "-I", "-c", NS_LISTENER, path] if os.geteuid() == 0 else \
        [unshare, "-r", "-n", PY, "-I", "-c", NS_LISTENER, path]
    lst = subprocess.Popen(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    try:
        t0 = time.monotonic()
        while not os.path.exists(path + ".ready") and time.monotonic() - t0 < 10:
            if lst.poll() is not None:
                pytest.skip(f"cannot start a listener in a new network namespace: {lst.stderr.read()[:200]}")
            time.sleep(0.05)
        from jev.sandbox import listening_unix_sockets
        assert path not in listening_unix_sockets()          # the host table really does not show it
        c = socket.socket(socket.AF_UNIX)
        c.connect(path)                                       # but it is reachable from the host
        c.close()
        ws = var_root / "ws"
        ws.mkdir()
        code = (f"import socket, os\nc = socket.socket(socket.AF_UNIX)\ntry:\n    c.connect({path!r}); print('CONNECTED')\n"
                "except OSError as e:\n    print('blocked', e.errno)\n"
                "for d in ('/var/lib', '/opt', '/srv', '/var/tmp', '/home'):\n"
                "    print(d, sorted(os.listdir(d)) if os.path.isdir(d) else 'absent')\n"
                "try:\n    open('/jev-root-write', 'w'); print('ROOT WRITABLE')\nexcept OSError as e:\n"
                "    print('root ro', e.errno)\n")
        r = sb.run([PY, "-I", "-c", code], ws)
        assert "CONNECTED" not in r.stdout and "blocked" in r.stdout, (r.stdout, r.stderr)
        assert "ROOT WRITABLE" not in r.stdout
        for d in ("/var/lib", "/srv", "/home"):
            assert f"{d} absent" in r.stdout, r.stdout
        # /opt shows only runtimes the run needs (e.g. a Node install), never the rest of /opt
        runtimes = sorted({p.parts[2] for p in sb._keep_paths() if len(p.parts) > 2 and p.parts[1] == "opt"})
        assert f"/opt {runtimes}" in r.stdout or "/opt absent" in r.stdout, (runtimes, r.stdout)
        # /var/tmp shows only the folder holding this workspace, not the listener's socket folder siblings
        assert f"/var/tmp {[var_root.name]}" in r.stdout
    finally:
        lst.kill()
        lst.wait()


def test_n9_bwrap_root_is_an_allowlist(tmp_path):
    """Structural fix: no `--ro-bind / /`; only system folders and what a run needs are bound."""
    if not LINUX:
        pytest.skip("Linux only")
    sb = _bwrap()
    ws = tmp_path / "ws"
    ws.mkdir()
    argv, fds = sb._wrap("bwrap", ["true"], ws, False)
    for fd in fds:
        os.close(fd)
    pairs = [(argv[i + 1], argv[i + 2]) for i, a in enumerate(argv) if a in ("--ro-bind", "--bind")]
    assert ("/", "/") not in pairs
    for _, dest in pairs:
        assert not any(dest == d or dest.startswith(d + "/") for d in ("/var/lib", "/srv", "/root")), dest


# ================================================================================================ N10
def test_n10_workspace_entry_cap_fails_closed(tmp_path):
    """Reviewer: workspace_usage stopped after 500 000 entries and returned a partial total, so a
    workspace with many files defeated the size cap."""
    from jev.sandbox import workspace_usage
    ws = tmp_path / "ws"
    ws.mkdir()
    for i in range(300):
        (ws / f"f{i}").write_bytes(b"")
    used, entries, complete = workspace_usage(ws, max_entries=200)
    assert not complete and entries >= 200
    sb = Sandbox({"sandbox": {"isolation": "auto"}})
    lim = Limits(max_workspace_entries=200)
    r = sb.run([PY, "-I", "-c", "print('should not run')"], ws, lim)
    assert r.refused and "should not run" not in r.stdout and "max_workspace_entries" in r.summary()
    for p in ws.iterdir():
        p.unlink()
    code = "import os\nfor i in range(400):\n    open(f'g{i}', 'w').close()\nprint('made')\n"
    r = sb.run([PY, "-I", "-c", code], ws, lim)
    assert not r.ok and r.workspace_over and "max_workspace_entries" in r.summary()


# ================================================================================================ N11
def test_n11_fifos_in_a_workspace_encoder_run_never_hang(tmp_path):
    """Reviewer: footage tools load a workspace encoder run with jepa_studio's plain open/read_text;
    a FIFO at <run>/encoder.sha256 or encoder.safetensors hung the turn."""
    cfg_src = REPO / "runs" / "shapes-desktop" / "config.json"
    if not cfg_src.exists() or not shutil.which("ffmpeg"):
        pytest.skip("shipped run or ffmpeg missing")
    data = tmp_path / "root" / "jev-data"
    ws = data / "workspaces" / "c-1"
    ws.mkdir(parents=True)
    subprocess.run(["ffmpeg", "-loglevel", "error", "-y", "-f", "lavfi", "-i", "testsrc2=size=160x96:rate=10",
                    "-t", "2", "-pix_fmt", "yuv420p", str(ws / "a.mp4")], check=True)
    run = ws / "myrun"
    run.mkdir()
    shutil.copy(cfg_src, run / "config.json")
    for name in ("encoder.sha256", "encoder.safetensors"):
        os.mkfifo(run / name)
    settings = {"data_dir": str(data), "workspace_root": str(data / "workspaces")}
    ctx = ToolContext(workspace=ws, settings={**settings, "_sandbox": Sandbox({**settings}), "_approved_rules": []})
    exc = _no_hang(lambda: REGISTRY["find_scenes"].fn(ctx, path="a.mp4", encoder_run="myrun"))
    assert isinstance(exc, ToolError) and "regular file" in str(exc), exc
    # a link to weights outside the workspace is not followed either
    for name in ("encoder.sha256", "encoder.safetensors"):
        (run / name).unlink()
    outside = tmp_path / "outside.safetensors"
    outside.write_bytes(b"x")
    os.symlink(outside, run / "encoder.safetensors")
    exc = _no_hang(lambda: REGISTRY["find_scenes"].fn(ctx, path="a.mp4", encoder_run="myrun"))
    assert isinstance(exc, ToolError), exc


def test_n11_jepa_studio_loaders_refuse_fifos_and_links(tmp_path):
    """jepa_train hands workspace images to jepa_studio's loaders: a FIFO named *.png (or a final
    link) is refused at once instead of hanging the training thread."""
    from jepa_studio.data.loaders import UnsafeInput, load_image, load_series
    os.mkfifo(tmp_path / "a.png")
    os.mkfifo(tmp_path / "s.csv")
    exc = _no_hang(lambda: load_image(tmp_path / "a.png", 32))
    assert isinstance(exc, UnsafeInput) and "regular file" in str(exc), exc
    exc = _no_hang(lambda: load_series(tmp_path / "s.csv"))
    assert isinstance(exc, UnsafeInput), exc
    from PIL import Image
    Image.new("RGB", (8, 8)).save(tmp_path / "real.png")
    os.symlink(tmp_path / "real.png", tmp_path / "link.png")
    assert load_image(tmp_path / "real.png", 32).shape == (8, 8, 3)
    exc = _no_hang(lambda: load_image(tmp_path / "link.png", 32))
    assert isinstance(exc, UnsafeInput), exc


def test_n11_jepa_studio_loaders_open_files_only_through_open_regular():
    """Guard for jepa_studio code reachable from JEV tools with workspace paths: the data loaders
    hand Pillow / NumPy / csv only file objects from open_regular(), never a path."""
    import ast
    src = (REPO / "jepa_studio" / "data" / "loaders.py").read_text()
    bad = []
    for node in ast.walk(ast.parse(src)):
        if not isinstance(node, ast.Call):
            continue
        f = node.func
        name = f.id if isinstance(f, ast.Name) else f"{getattr(f.value, 'id', '?')}.{f.attr}" \
            if isinstance(f, ast.Attribute) else "?"
        if name in ("open", "io.open", "Image.open", "np.load", "numpy.load", "os.open"):
            first = node.args[0] if node.args else None
            if name == "os.open" and node.lineno and "_OPEN_FLAGS" in ast.unparse(node):
                continue
            if isinstance(first, ast.Name) and first.id in ("fh", "f"):
                continue
            bad.append(f"line {node.lineno}: {ast.unparse(node)[:100]}")
    assert not bad, bad


# ================================================================================================ info items
def test_info_huge_ints_and_nan_in_arguments_are_refused(tmp_path):
    """Integers beyond 2^53 cannot be shown exactly by the app, and NaN/Infinity break the JSON
    event stream: tool arguments with either are refused before any decision."""
    from jev.agent import Runtime
    from jev.llm import MockLLM
    script = [{"content": "", "tool_calls": [{"name": "calculator", "arguments": {"expression": "1", "n": 2 ** 60}}]},
              {"content": "", "tool_calls": [{"name": "make_plot", "arguments": {
                  "kind": "line", "series": [{"y": [1.0, float("nan"), float("inf")]}]}}]},
              {"content": "done"}]
    rt = Runtime(tmp_path, llm=MockLLM(script))
    r = rt.start_turn("go")
    job = rt.jobs.wait(r["turn_id"], 30)
    calls = [(e["name"], e["decision"]) for e in job.events if e["event"] == "tool_call"]
    assert calls == [("calculator", "deny"), ("make_plot", "deny")], calls
    reasons = " ".join(e["reason"] for e in job.events if e["event"] == "tool_call")
    assert "2^53" in reasons and "NaN" in reasons
    json.dumps(job.events, allow_nan=False)                 # the event stream stays strict JSON


def test_info_venv_pip_conf_is_removed_before_installs(tmp_path, monkeypatch):
    """A pip.conf inside the venv itself (pip reads it even with --isolated) is removed first."""
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
    try:
        data = tmp_path / "root" / "jev-data"
        ws = data / "workspaces" / "c-1"
        ws.mkdir(parents=True)
        settings = {"data_dir": str(data), "workspace_root": str(data / "workspaces")}
        sb = Sandbox({**settings, "sandbox": {"isolation": "auto"}})
        ctx = ToolContext(workspace=ws, settings={**settings, "_sandbox": sb, "_approved_rules": [],
                                                  "_conversation_id": "c-1"})
        from jev.tools.code import venv_dir
        venv = venv_dir(ctx)
        subprocess.run([PY, "-m", "venv", "--system-site-packages", str(venv)], check=True)
        (venv / "pip.conf").write_text(f"[global]\nindex-url = http://127.0.0.1:{srv.server_address[1]}/simple\n"
                                      f"extra-index-url = http://127.0.0.1:{srv.server_address[1]}/simple\n")
        dead = "http://127.0.0.1:9"
        for k in ("HTTP_PROXY", "HTTPS_PROXY", "http_proxy", "https_proxy"):
            monkeypatch.setenv(k, dead)
        spec = REGISTRY["install_package"]
        d = Policy(settings).decide(spec, {"packages": ["jev-nonexistent-pkg-r3"]}, ctx)
        ctx.settings["_approved_rules"] = list(d.rules)
        r = spec.fn(ctx, packages=["jev-nonexistent-pkg-r3"])
        assert hits == [], hits
        assert not (venv / "pip.conf").exists() and "pip.conf" in r.content
    finally:
        srv.shutdown()
        srv.server_close()
