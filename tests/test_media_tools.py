"""jev.tools.media: registration, risk levels, approval hooks, argument schemas and workspace use."""
import json

import numpy as np
import pytest
from PIL import Image

from jepa_studio.config import validate
from jev.media import ffmpeg as F
from jev.media import generative as G
from jev.tools import media as M  # noqa: F401 - registers the tools
from jev.tools.base import REGISTRY, ToolContext, ToolError

EXPECTED_RISK = {
    "make_explainer_video": "sandbox", "make_slides": "sandbox", "text_to_speech": "sandbox", "probe_media": "read",
    "find_scenes": "sandbox", "search_footage": "sandbox", "make_highlight_reel": "sandbox", "cut_video": "sandbox",
    "make_image": "sandbox", "generate_image": "approve", "generate_video": "approve", "download_voice": "approve",
    "download_generative_model": "approve",
}

SAMPLE_ARGS = {
    "make_explainer_video": {"scenes": [{"narration": "hi", "visual": {"kind": "slide", "slide": {"type": "title",
                                                                                                "title": "T"}}}],
                             "resolution": "720p", "ken_burns": "auto"},
    "make_slides": {"slides": [{"type": "bullets", "title": "x", "bullets": ["a"]}]},
    "text_to_speech": {"text": "hello"},
    "probe_media": {"path": "a.mp4"},
    "find_scenes": {"path": "a.mp4"},
    "search_footage": {"path": "a.mp4", "query_time": 1.5},
    "make_highlight_reel": {"path": "a.mp4", "budget_seconds": 30},
    "cut_video": {"path": "a.mp4", "cuts": [{"start": 0, "end": 1, "label": "x"}]},
    "make_image": {"kind": "chart", "spec": {"series": [{"values": [1, 2]}]}},
    "generate_image": {"model": "sdxl-turbo", "prompt": "a fox"},
    "generate_video": {"model": "ltx-video", "prompt": "waves", "frames": 49},
    "download_voice": {"voice": "en_US-lessac-medium"},
    "download_generative_model": {"model": "flux-schnell"},
}


@pytest.fixture
def ctx(tmp_path):
    ws = tmp_path / "ws"
    ws.mkdir()
    return ToolContext(workspace=ws, log=lambda m: None, settings={"media": {"voices_dir": str(tmp_path / "v"),
                                                                             "models_dir": str(tmp_path / "m")}})


def test_registered_with_expected_risk():
    for name, risk in EXPECTED_RISK.items():
        assert REGISTRY[name].risk == risk, name
        schema = REGISTRY[name].schema()
        assert schema["function"]["name"] == name and schema["function"]["parameters"]["type"] == "object"


def test_sample_args_validate_and_bad_args_do_not():
    for name, args in SAMPLE_ARGS.items():
        assert validate(args, REGISTRY[name].parameters) == [], name
    assert validate({"voice": "../../etc/passwd"}, REGISTRY["download_voice"].parameters)
    assert validate({"model": "anything"}, REGISTRY["download_generative_model"].parameters)
    assert validate({"scenes": []}, REGISTRY["make_explainer_video"].parameters)


def test_long_explainer_needs_approval(ctx):
    hook = REGISTRY["make_explainer_video"].needs_approval
    assert hook(SAMPLE_ARGS["make_explainer_video"], ctx) is None
    long = {"scenes": [{"narration": "word " * 300, "visual": {"kind": "image", "path": "p.png"}}] * 40,
            "resolution": "1080p", "fps": 60}
    reason = hook(long, ctx)
    assert reason and "10 min" in reason


def test_make_slides_and_image_tools(ctx):
    r = REGISTRY["make_slides"].fn(ctx, slides=SAMPLE_ARGS["make_slides"]["slides"] * 2, resolution="720p")
    assert r.ok and len(r.artifacts) == 3                      # 2 PNGs + PDF
    assert Image.open(r.artifacts[0].path).size == (1280, 720)
    for a in r.artifacts:
        assert ctx.workspace.resolve() in a.path.parents
        a.to_json(ctx.workspace)
    r = REGISTRY["make_image"].fn(ctx, kind="diagram", spec={"nodes": [{"id": "a"}, {"id": "b"}],
                                                             "edges": [{"from": "a", "to": "b"}]}, width=640, height=360)
    assert r.ok and Image.open(r.artifacts[0].path).size == (640, 360)
    Image.new("RGB", (100, 80), "red").save(ctx.workspace / "pic.png")
    r = REGISTRY["make_image"].fn(ctx, kind="edit", spec={"path": "pic.png", "ops": [{"op": "resize", "width": 50}]},
                                  output="out/edited.png")
    assert Image.open(ctx.workspace / "out" / "edited.png").size == (50, 40)


def test_paths_confined_to_workspace(ctx):
    with pytest.raises(ToolError):
        REGISTRY["probe_media"].fn(ctx, path="../../etc/passwd")
    with pytest.raises(ToolError):
        REGISTRY["make_slides"].fn(ctx, slides=SAMPLE_ARGS["make_slides"]["slides"], output_dir="../outside")
    with pytest.raises(ToolError):
        REGISTRY["make_image"].fn(ctx, kind="edit", spec={"path": "/etc/hostname", "ops": []})
    with pytest.raises(ToolError):
        REGISTRY["text_to_speech"].fn(ctx, text="hi", output="../x.wav")


def test_speech_probe_and_footage_tools(ctx):
    r = REGISTRY["text_to_speech"].fn(ctx, text="Hello from JEV.")
    assert r.ok and r.artifacts[0].kind == "audio" and r.data["backend"]
    clip = ctx.workspace / "a.mp4"
    F.run(["-f", "lavfi", "-i", "color=c=red:size=160x96:rate=12:duration=1.5", "-f", "lavfi", "-i",
           "color=c=blue:size=160x96:rate=12:duration=1.5", "-filter_complex", "[0:v][1:v]concat=n=2:v=1[v]",
           "-map", "[v]", *F.h264_args(30, "ultrafast"), str(clip)], ctx.workspace, outputs=[clip])
    r = REGISTRY["probe_media"].fn(ctx, path="a.mp4")
    assert r.ok and r.data["within_limits"] and abs(r.data["duration_s"] - 3.0) < 0.15
    r = REGISTRY["find_scenes"].fn(ctx, path="a.mp4")
    assert r.ok and len(r.data["shots"]) == 2 and abs(r.data["shots"][1]["start"] - 1.5) < 0.1
    assert any(a.kind == "image" for a in r.artifacts)
    r = REGISTRY["search_footage"].fn(ctx, path="a.mp4", query_time=2.5, top_k=2)
    assert r.data["hits"][0]["index"] == 1
    r = REGISTRY["make_highlight_reel"].fn(ctx, path="a.mp4", budget_seconds=2.0, min_clip_seconds=0.5,
                                           max_clip_seconds=1.0)
    assert r.ok and r.data["duration_s"] <= 2.1
    r = REGISTRY["cut_video"].fn(ctx, path="a.mp4", cuts=[{"start": 0.2, "end": 1.0, "label": "red"}],
                                 captions=True)
    assert r.ok and r.data["mode"] == "reencode"
    cut_json = next(a.path for a in r.artifacts if a.path.name.endswith(".cuts.json"))
    assert json.loads(cut_json.read_text())["cuts"][0]["label"] == "red"


def test_explainer_tool_small(ctx):
    np.random.seed(0)
    r = REGISTRY["make_explainer_video"].fn(
        ctx, scenes=[{"narration": "One.", "visual": {"kind": "slide", "slide": {"type": "title", "title": "A"}}},
                     {"narration": "Two.", "visual": {"type": "bullets", "title": "B", "bullets": ["x"]}}],
        resolution="480p", fps=12)
    assert r.ok and {a.kind for a in r.artifacts} == {"video", "text", "data"}
    info = F.probe(r.artifacts[0].path, ctx.workspace)
    assert (info.width, info.height) == (854, 480) and info.has_audio


def test_generate_video_refused_without_gpu(ctx, monkeypatch):
    monkeypatch.setattr(G, "detect_gpu", lambda: {"backend": "none", "name": "no GPU", "vram_gb": 0.0})
    d = ctx.settings["media"]["models_dir"]
    from pathlib import Path

    (Path(d) / "ltx-video").mkdir(parents=True)
    (Path(d) / "ltx-video" / "model_index.json").write_text(json.dumps({"_class_name": "LTXPipeline"}))
    with pytest.raises(ToolError, match="needs a GPU"):
        REGISTRY["generate_video"].fn(ctx, model="ltx-video", prompt="waves")
