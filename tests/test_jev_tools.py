"""JEV's own tools (jev/tools/core.py, jev/tools/code.py), called directly."""
import shutil
from pathlib import Path

import pytest

from jev.sandbox import Sandbox
from jev.tools.base import ToolContext, ToolError, load_all

REPO = Path(__file__).resolve().parent.parent
TOOLS = load_all()


@pytest.fixture
def ctx(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    s = {"limits": {"cpu_seconds": 60, "memory_mb": 2048, "wall_seconds": 60, "max_output_mb": 50,
                    "max_file_mb": 50}, "root": str(REPO), "sandbox": {"isolation": "auto"}}
    s["_sandbox"] = Sandbox(s)
    return ToolContext(workspace=ws, settings=s)


def call(_tool, _ctx, **kw):
    return TOOLS[_tool].fn(_ctx, **kw)


def test_every_tool_schema_is_valid_for_ollama():
    for spec in TOOLS.values():
        sch = spec.schema()
        assert sch["type"] == "function" and sch["function"]["parameters"]["type"] == "object"
        assert spec.risk in ("read", "sandbox", "approve") and spec.description
    mine = {"list_files", "read_file", "calculator", "get_time", "write_file", "make_plot", "summarize_file",
            "run_python", "run_node", "run_shell", "install_package", "create_project", "jepa_embed_images",
            "jepa_train"}
    assert mine <= set(TOOLS)
    assert {n for n in mine if TOOLS[n].risk == "approve"} == {"run_shell", "install_package", "jepa_train"}
    assert {n for n in mine if TOOLS[n].risk == "read"} == {"list_files", "read_file", "calculator", "get_time",
                                                           "summarize_file"}


@pytest.mark.parametrize("expr,val", [("2+2", 4), ("2^10", 1024), ("sqrt(16) + pi - pi", 4.0), ("factorial(5)", 120),
                                      ("-3 // 2", -2), ("log(e)", 1.0)])
def test_calculator(ctx, expr, val):
    r = call("calculator", ctx, expression=expr)
    assert r.ok and r.data["value"] == val


@pytest.mark.parametrize("expr", ["__import__('os').system('ls')", "open('x')", "(1).__class__", "9**9**9",
                                  "factorial(100000)", "[x for x in range(10)]", "lambda: 1", "a.b", "'s' * 3",
                                  "1/0", "x" * 600])
def test_calculator_refuses_non_math(ctx, expr):
    with pytest.raises(ToolError):
        call("calculator", ctx, expression=expr)


def test_file_tools(ctx):
    r = call("write_file", ctx, path="notes/a.md", content="# Title\nline two\n")
    assert r.ok and r.artifacts[0].kind == "text" and r.artifacts[0].path.name == "a.md"
    call("write_file", ctx, path="notes/a.md", content="more\n", append=True)
    assert (ctx.workspace / "notes/a.md").read_text().endswith("more\n")
    assert "notes/a.md" in call("list_files", ctx).content
    r = call("read_file", ctx, path="notes/a.md", max_chars=7)
    assert r.content.startswith("# Title") and "offset=" in r.content
    (ctx.workspace / "t.csv").write_text("x,y\n1,2\n3,4\n")
    s = call("summarize_file", ctx, path="t.csv").content
    assert "2 rows" in s and "x: min 1, mean 2, max 3" in s
    (ctx.workspace / "b.bin").write_bytes(b"\x00\x01" * 100)
    assert "binary" in call("read_file", ctx, path="b.bin").content
    for bad in ("../x", "/etc/passwd"):
        with pytest.raises(ToolError):
            call("read_file", ctx, path=bad)
    with pytest.raises(ToolError):
        call("write_file", ctx, path=".jev/x", content="no")
    assert "JEV" not in call("get_time", ctx).content and "UTC" in call("get_time", ctx).content


def test_make_plot(ctx):
    pytest.importorskip("matplotlib")
    r = call("make_plot", ctx, kind="bar", series=[{"name": "a", "x": ["q1", "q2"], "y": [3, 5]},
                                                 {"name": "b", "y": [1, 2]}], title="t", filename="bars.png")
    p = ctx.workspace / "bars.png"
    assert r.ok and p.read_bytes()[:4] == b"\x89PNG" and r.artifacts[0].mime == "image/png"
    with pytest.raises(ToolError):
        call("make_plot", ctx, kind="line", series=[{"x": [1, 2, 3], "y": [1, 2]}])


def test_run_python_uses_workspace_modules_and_reports_files(ctx):
    (ctx.workspace / "helper.py").write_text("def f():\n    return 41 + 1\n")
    r = call("run_python", ctx, code="from helper import f\nprint('answer', f())\nopen('r.txt','w').write('x')")
    assert r.ok, r.content
    assert "answer 42" in r.content and [a.path.name for a in r.artifacts] == ["r.txt"]
    r = call("run_python", ctx, code="raise SystemExit(3)")
    assert not r.ok and "exit code 3" in r.content
    r = call("run_python", ctx, code="import time\ntime.sleep(5)", timeout_s=1)
    assert not r.ok and "wall-clock" in r.content


def test_run_node(ctx):
    if not shutil.which("node"):
        pytest.skip("node not installed")
    r = call("run_node", ctx, code="const fs = require('fs'); fs.writeFileSync('n.json', '{}'); console.log(6*7)")
    assert r.ok and "42" in r.content and r.artifacts[0].path.name == "n.json"
    r = call("run_node", ctx, code="import { readFileSync } from 'fs';\nconsole.log('esm', readFileSync('n.json','utf8'))")
    assert r.ok and "esm {}" in r.content


def test_run_shell_only_after_approval_and_offline(ctx):
    r = call("run_shell", ctx, command="echo $((40+2)) > s.txt && cat s.txt")
    assert r.ok and "42" in r.content
    if ctx.settings["_sandbox"].capabilities()["network_isolation"] == "enforced":
        r = call("run_shell", ctx, command="python3 -c \"import socket;print(socket.socket().connect_ex(('1.1.1.1',80)))\"")
        assert "0\n" not in r.content.split("stdout:")[-1][:3]


def test_install_package_validates_names(ctx):
    for bad in (["--index-url=http://evil"], ["-r", "req.txt"], ["git+https://x/y.git"], ["../local"], ["a; rm -rf /"]):
        with pytest.raises(ToolError, match="not a plain package"):
            call("install_package", ctx, packages=bad)
    reason = TOOLS["install_package"].needs_approval({"packages": ["rich"]}, ctx)
    assert "[install]" in reason and "[network]" in reason


def test_create_project(ctx):
    r = call("create_project", ctx, name="demo", template="python", description="A demo",
             files={"data/readme.txt": "hi"})
    assert r.ok and (ctx.workspace / "demo/main.py").exists() and (ctx.workspace / "demo/data/readme.txt").exists()
    with pytest.raises(ToolError, match="already exists"):
        call("create_project", ctx, name="demo")
    with pytest.raises(ToolError):
        call("create_project", ctx, name="evil", files={"../../x.txt": "no"})
    r = call("run_python", ctx, code="import runpy\nrunpy.run_path('demo/main.py', run_name='__main__')")
    assert "hello from demo" in r.content


def test_jepa_embed_images_with_the_shipped_run(ctx):
    from PIL import Image
    if not (REPO / "runs/shapes-desktop/encoder.safetensors").exists():
        pytest.skip("shipped run not present")
    imgs = ctx.workspace / "imgs"
    imgs.mkdir()
    for i, color in enumerate(["red", "red", "blue", "green"]):
        Image.new("RGB", (40, 40), color).save(imgs / f"{i}_{color}.png")
    r = call("jepa_embed_images", ctx, run_id="shapes-desktop", folder="imgs")
    assert r.ok and "hash-verified" in r.content and "Embedded 4 images" in r.content
    assert "0_red.png ~ imgs/1_red.png" in r.content.replace("imgs/0_red.png", "0_red.png")
    import numpy as np
    e = np.load(ctx.workspace / "embeddings.npy")
    assert e.shape[0] == 4
    with pytest.raises(ToolError):
        call("jepa_embed_images", ctx, run_id="../../etc")
    with pytest.raises(ToolError):
        call("jepa_embed_images", ctx, run_id="no-such-run")
