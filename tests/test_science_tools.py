"""The three JEV science tools: registration, schemas, approval hooks and a real (tiny) run of each."""
from __future__ import annotations

import pytest

from jev.tools.base import REGISTRY, ToolContext, ToolError


@pytest.fixture(scope="module")
def tools():
    import jev.tools.science  # noqa: F401 - registers the tools
    return REGISTRY


def test_registered_with_risk_and_schema(tools):
    for name, risk in (("render_black_hole", "sandbox"), ("simulate_physics", "sandbox"),
                       ("explain_black_hole_physics", "read")):
        spec = tools[name]
        assert spec.risk == risk and spec.category == "science"
        sch = spec.schema()
        assert sch["function"]["parameters"]["type"] == "object"
    assert tools["simulate_physics"].parameters["required"] == ["kind"]


def test_needs_approval_for_huge_or_long_jobs(tools, tmp_path):
    ctx = ToolContext(workspace=tmp_path)
    hook = tools["render_black_hole"].needs_approval
    assert hook({"width": 160, "height": 90}, ctx) is None
    reason = hook({"width": 3840, "height": 2160, "frames": 600, "quality": "final"}, ctx)
    assert reason and "pixel" in reason
    long = hook({"width": 1920, "height": 1080, "frames": 60, "animation": "zoom", "quality": "final"}, ctx)
    assert long and "minutes" in long
    assert tools["simulate_physics"].needs_approval({"kind": "wave", "n": 1024, "t_end": 50}, ctx)
    assert tools["simulate_physics"].needs_approval({"kind": "orbit"}, ctx) is None


def test_render_tool_writes_artifacts(tools, tmp_path):
    ctx = ToolContext(workspace=tmp_path)
    res = tools["render_black_hole"].fn(ctx, width=64, height=36, interstellar=True)
    assert res.ok and "Kerr black hole" in res.content and "Interstellar mode" in res.content
    kinds = [a.kind for a in res.artifacts]
    assert kinds == ["image", "data"]
    assert all(tmp_path.resolve() in a.path.parents for a in res.artifacts)
    with pytest.raises(ToolError):
        tools["render_black_hole"].fn(ctx, spin=2.0)


def test_simulate_tool(tools, tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    ctx = ToolContext(workspace=tmp_path)
    res = tools["simulate_physics"].fn(ctx, kind="orbit", seconds=1.0, fps=5, width=96, height=96)
    assert res.ok and "energy_rel_drift_max" in res.content
    assert res.artifacts[0].kind == "video"
    with pytest.raises(ToolError, match="CFL"):
        tools["simulate_physics"].fn(ctx, kind="wave", courant=1.5, seconds=1.0, fps=5)


def test_explain_tool(tools, tmp_path):
    ctx = ToolContext(workspace=tmp_path)
    res = tools["explain_black_hole_physics"].fn(ctx, topic="interstellar")
    assert "1502.03808" in res.content and "Doppler" in res.content
    assert "Carter" in tools["explain_black_hole_physics"].fn(ctx).content
    with pytest.raises(ToolError):
        tools["explain_black_hole_physics"].fn(ctx, topic="wormholes")
