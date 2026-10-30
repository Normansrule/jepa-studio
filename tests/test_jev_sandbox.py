"""What the sandbox really enforces on this machine (the tests skip what the OS cannot do)."""
import os
import sys
import time
from pathlib import Path

import pytest

from jev.sandbox import STREAM_CAP, Limits, Sandbox

POSIX = os.name == "posix"
LINUX = sys.platform.startswith("linux")


@pytest.fixture(scope="module")
def sb():
    return Sandbox({"sandbox": {"isolation": "auto"}})


@pytest.fixture
def ws(tmp_path):
    w = tmp_path / "ws"
    w.mkdir()
    return w


def py(code):
    return [sys.executable, "-I", "-c", code]


def test_capabilities_report_is_honest(sb):
    caps = sb.capabilities()
    assert caps["network_isolation"] in ("enforced", "not available")
    assert caps["method"] in ("bwrap", "unshare", "sandbox-exec", "job-object", "none")
    if caps["network_isolation"] == "enforced":
        assert "blocked" in caps["probe"]
    assert caps["limits"]["wall_seconds"] == 300
    print("sandbox capabilities:", caps)


def test_runs_in_workspace_with_scrubbed_env(sb, ws, monkeypatch):
    monkeypatch.setenv("JEPA_STUDIO_TOKEN", "super-secret-token")
    monkeypatch.setenv("AWS_SECRET_ACCESS_KEY", "aws-secret")
    monkeypatch.setenv("GITHUB_TOKEN", "ghp_x")
    r = sb.run(py("import os, json; print(json.dumps({'cwd': os.getcwd(), 'env': dict(os.environ)}))"), ws)
    assert r.ok, r.stderr
    import json
    out = json.loads(r.stdout)
    assert os.path.realpath(out["cwd"]) == os.path.realpath(ws)
    blob = json.dumps(out["env"])
    assert "super-secret-token" not in blob and "aws-secret" not in blob and "ghp_x" not in blob
    assert out["env"]["HOME"] == str(ws)


@pytest.mark.parametrize("mode", ["auto", "unshare", "none"])
def test_wall_timeout_kills_the_whole_tree(mode, ws):
    sb = Sandbox({"sandbox": {"isolation": mode}})
    if mode == "unshare" and sb.capabilities()["method"] != "unshare":
        pytest.skip("unshare unavailable")
    # a grandchild that would keep writing a heartbeat file forever, even after starting its own session
    child = ("import os, time\nos.setsid()\n"
             "while True:\n    open('beat.txt', 'w').write(str(time.time())); time.sleep(0.05)")
    code = ("import subprocess, sys, time\n"
            f"p = subprocess.Popen([sys.executable, '-c', {child!r}])\n"
            "open('child.pid', 'w').write(str(p.pid)); time.sleep(60)")
    t0 = time.monotonic()
    r = sb.run(py(code), ws, Limits(wall_seconds=2))
    assert r.timed_out and time.monotonic() - t0 < 10
    assert "wall-clock" in r.summary()
    time.sleep(0.3)
    before = (ws / "beat.txt").read_text()
    time.sleep(0.5)
    assert (ws / "beat.txt").read_text() == before, f"grandchild still running after the timeout ({mode})"


def test_cancel_stops_a_run(sb, ws):
    t0 = time.monotonic()
    r = sb.run(py("import time; time.sleep(60)"), ws, cancelled=lambda: time.monotonic() - t0 > 0.5)
    assert r.cancelled and time.monotonic() - t0 < 10


@pytest.mark.skipif(not POSIX or sys.platform == "darwin", reason="memory rlimits are not enforced on macOS/Windows")
def test_memory_limit_enforced(sb, ws):
    r = sb.run(py("x = bytearray(900 * 1024 * 1024); print(len(x))"), ws, Limits(memory_mb=256))
    assert not r.ok and "MemoryError" in r.stderr
    assert "memory limit" in (r.limit_hit or "")
    r = sb.run(py("x = bytearray(50 * 1024 * 1024); print('fine')"), ws, Limits(memory_mb=512))
    assert r.ok and "fine" in r.stdout


@pytest.mark.skipif(not POSIX, reason="rlimits are POSIX")
def test_cpu_and_file_size_limits(sb, ws):
    r = sb.run(py("while True: pass"), ws, Limits(cpu_seconds=1, wall_seconds=20))
    assert not r.ok and not r.timed_out and "CPU" in (r.limit_hit or ""), (r.returncode, r.limit_hit)
    r = sb.run(py("open('big.bin', 'wb').write(b'x' * (3 << 20))"), ws, Limits(max_output_mb=1))
    assert not r.ok and "File too large" in r.stderr
    assert (ws / "big.bin").stat().st_size <= 1 << 20


def test_output_is_capped(sb, ws):
    r = sb.run(py(f"import sys; sys.stdout.write('y' * {STREAM_CAP * 3})"), ws)
    assert r.truncated and len(r.stdout) < STREAM_CAP + 500
    assert "dropped" in r.stdout
    assert len(r.summary(2000)) < 2600


def test_changed_files_reported(sb, ws):
    r = sb.run(py("open('plot.png','wb').write(b'png'); import os; os.makedirs('out', exist_ok=True); "
                  "open('out/r.csv','w').write('a')"), ws)
    assert r.ok and set(r.changed_files) == {"plot.png", "out/r.csv"}


def test_argv_strings_refused(sb, ws):
    with pytest.raises(TypeError):
        sb.run("echo hi", ws)


def test_network_blocked_when_isolation_enforced(sb, ws):
    if sb.capabilities()["network_isolation"] != "enforced":
        pytest.skip(f"no network isolation on this OS: {sb.capabilities()['probe']}")
    code = ("import socket\ns = socket.socket(); s.settimeout(3)\n"
            "print('rc', s.connect_ex(('1.1.1.1', 80)))\n"
            "print(sorted(n for _, n in socket.if_nameindex()))")
    r = sb.run(py(code), ws)
    assert r.ok, r.stderr
    assert "rc 0" not in r.stdout and "['lo']" in r.stdout
    # the JEPA Studio backend on the host's loopback is unreachable too
    import socket as s
    srv = s.socket()
    srv.bind(("127.0.0.1", 0))
    srv.listen(1)
    port = srv.getsockname()[1]
    r = sb.run(py(f"import socket; s=socket.socket(); s.settimeout(2); print('rc', s.connect_ex(('127.0.0.1', {port})))"),
               ws)
    srv.close()
    assert "rc 0" not in r.stdout


@pytest.mark.skipif(not LINUX, reason="bwrap is Linux only")
def test_bwrap_file_system_is_read_only_outside_workspace(ws, tmp_path):
    sb = Sandbox({"sandbox": {"isolation": "bwrap"}})
    if sb.capabilities()["method"] != "bwrap":
        pytest.skip("bwrap not available here")
    outside = tmp_path / "outside.txt"           # /tmp is a private tmpfs inside: the write never lands
    sb.run(py(f"open({str(outside)!r}, 'w').write('x')"), ws)
    assert not outside.exists()
    if os.path.isdir("/var/tmp") and os.access("/var/tmp", os.W_OK):
        import tempfile
        d = tempfile.mkdtemp(dir="/var/tmp")
        try:
            target = os.path.join(d, "outside.txt")
            r = sb.run(py(f"open({target!r}, 'w').write('x')"), ws)
            assert not r.ok and "Read-only file system" in r.stderr and not os.path.exists(target)
        finally:
            import shutil
            shutil.rmtree(d, ignore_errors=True)
    r = sb.run(py("open('inside.txt', 'w').write('ok'); print('wrote')"), ws)
    assert r.ok and (ws / "inside.txt").read_text() == "ok"


@pytest.mark.skipif(not LINUX, reason="unshare is Linux only")
def test_unshare_network_namespace_reported(ws):
    sb = Sandbox({"sandbox": {"isolation": "unshare"}})
    caps = sb.capabilities()
    print("unshare probe:", caps["probe"])
    if caps["method"] != "unshare":
        pytest.skip(f"unshare --user --net does not work here: {caps['probe']}")
    r = sb.run(py("import socket; print(sorted(n for _, n in socket.if_nameindex()))"), ws)
    assert "['lo']" in r.stdout
    assert caps["filesystem"].startswith("not enforced")      # unshare isolates the network only


def test_network_on_request(ws):
    sb = Sandbox({"sandbox": {"isolation": "auto"}})
    if sb.capabilities()["network_isolation"] != "enforced":
        pytest.skip("no isolation to lift")
    r = sb.run(py("import socket; print(len([n for _, n in socket.if_nameindex()]))"), ws, network=True)
    assert r.ok and int(r.stdout.strip()) >= 1


@pytest.mark.skipif(not LINUX, reason="bwrap is Linux only")
def test_bwrap_hides_home_folder(ws, monkeypatch):
    import shutil
    import tempfile
    sb = Sandbox({"sandbox": {"isolation": "bwrap", "hide_home": True}})
    if sb.capabilities()["method"] != "bwrap" or not os.access("/var/tmp", os.W_OK):
        pytest.skip("bwrap or /var/tmp not available")
    home = tempfile.mkdtemp(dir="/var/tmp")          # not under /tmp, which is private anyway
    try:
        (Path(home) / "id_rsa").write_text("PRIVATE")
        monkeypatch.setenv("HOME", home)
        code = f"import os; print(os.path.exists({os.path.join(home, 'id_rsa')!r}))"
        assert sb.run(py(code), ws).stdout.strip() == "False"
        open_sb = Sandbox({"sandbox": {"isolation": "bwrap", "hide_home": False}})
        assert open_sb.run(py(code), ws).stdout.strip() == "True"
    finally:
        shutil.rmtree(home, ignore_errors=True)
