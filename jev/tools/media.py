"""JEV media tools: explainer videos, slides, speech, footage editing, images, generative models.

Risk (see jev/tools/base.py):
    read     probe_media
    sandbox  make_explainer_video (asks when the render estimate is over 10 minutes), make_slides,
             text_to_speech, find_scenes, search_footage, make_highlight_reel, cut_video, make_image
    approve  generate_image, generate_video (GPU jobs that load large models),
             download_voice, download_generative_model (network downloads)

All paths the model passes are workspace-relative and resolved with ctx.path(); every output is
written inside the workspace and registered with ctx.artifact().
"""
from __future__ import annotations

import json
import re
from pathlib import Path

from jev.media import ffmpeg as F
from jev.media import footage as FT
from jev.media import generative as G
from jev.media import images as I
from jev.media import slides as S
from jev.media import tts as T
from jev.media import video as V
import functools

from jev import wsfs

from jev.media.common import Limits, MediaError, open_image_safely
from jev.sandbox import sandbox_for
from jev.tools.base import ToolContext, ToolResult
from jev.tools.base import tool as _tool

RESOLUTIONS = {"1080p": (1920, 1080), "720p": (1280, 720), "480p": (854, 480)}
# tools that start FFmpeg / ffprobe or a speech engine: they run in the conversation's sandbox and,
# on a weak sandbox, ask first (the locked weak_sandbox row)
RUNS_FFMPEG = {"make_explainer_video", "text_to_speech", "probe_media", "find_scenes", "search_footage",
               "make_highlight_reel", "cut_video", "generate_video"}


def tool(name: str, **kw):
    """jev.tools.base.tool, plus: every FFmpeg call the tool makes runs in its conversation's sandbox."""
    def wrap(fn):
        @functools.wraps(fn)
        def in_sandbox(ctx: ToolContext, *a, **k):
            with F.use_sandbox(sandbox_for(ctx.settings)), wsfs.use_scratch(ctx.scratch()):
                return fn(ctx, *a, **k)
        return _tool(name, runs_code=name in RUNS_FFMPEG, **kw)(in_sandbox)
    return wrap
APPROVAL_SECONDS = 600.0


# ----------------------------------------------------------------------------- helpers

def _limits(ctx: ToolContext) -> Limits:
    return Limits.from_settings(ctx.settings)


def _out(ctx: ToolContext, rel: str | None, default: str, suffix: str) -> Path:
    p = ctx.out_path(rel or default)
    if p.suffix.lower() != suffix:
        p = p.with_suffix(suffix)
    wsfs.makedirs(ctx.workspace, p.parent)
    return p


def _rel(ctx: ToolContext, p: Path) -> str:
    return Path(p).resolve().relative_to(ctx.workspace.resolve()).as_posix()


def _run_dir(ctx: ToolContext, name: str | None) -> Path | None:
    """A jepa_studio run by name (in the runs folder) or a run folder inside the workspace."""
    if not name:
        return None
    if re.fullmatch(r"[A-Za-z0-9_.\-]+", name) and name not in (".", ".."):
        root = Path((ctx.settings.get("media", {}) or {}).get("runs_dir") or ctx.settings.get("runs_dir") or "runs")
        d = root.expanduser() / name
        if (d / "config.json").is_file():
            return d
    d = ctx.path(name)
    if (d / "config.json").is_file():
        return d
    raise MediaError(f"no jepa_studio run named {name!r} (looked in the runs folder and the workspace)")


def _size(resolution: str | None, default: str = "1080p") -> tuple[int, int]:
    r = resolution or default
    if r not in RESOLUTIONS:
        raise MediaError(f"resolution must be one of {', '.join(RESOLUTIONS)}")
    return RESOLUTIONS[r]


_SLIDE_SCHEMA = {"type": "object", "description": "A slide: {type: title|bullets|code|image|equation|chart|diagram, "
                 "title?, subtitle?, bullets?, code?, language?, path?, fit?, equations?, chart?, diagram?, caption?}",
                 "properties": {"type": {"type": "string", "enum": list(S.SLIDE_TYPES)}}, "required": ["type"]}


# ----------------------------------------------------------------------------- explainer

def _explainer_estimate(args: dict, ctx: ToolContext) -> str | None:
    try:
        size = _size(args.get("resolution"))
        est = V.estimate_render_seconds(args.get("scenes") or [], size, float(args.get("fps", 30)),
                                        args.get("ken_burns", "auto"))
    except Exception:  # noqa: BLE001 - malformed args are rejected later by the tool itself
        return None
    if est > APPROVAL_SECONDS:
        return f"estimated render time ~{est / 60:.0f} min (over 10 min) on this machine"
    return None


@tool(
    name="make_explainer_video",
    description=("Make a narrated explainer video (MP4 + SRT captions + storyboard JSON) from scenes. Each scene "
                 "has narration text (spoken with a local voice) and a visual: a slide spec, an image, a video "
                 "clip or a science render from the workspace. Scene length follows the narration."),
    parameters={
        "type": "object",
        "properties": {
            "scenes": {"type": "array", "minItems": 1, "maxItems": V.MAX_SCENES, "items": {
                "type": "object",
                "properties": {
                    "narration": {"type": "string", "maxLength": 4000},
                    "visual": {"type": "object", "description": "{kind: slide, slide: {...}} | {kind: image, path, "
                               "fit?} | {kind: video, path, start?, end?} | {kind: render, path}",
                               "properties": {"kind": {"type": "string", "enum": ["slide", "image", "video", "render"]},
                                              "slide": _SLIDE_SCHEMA, "path": {"type": "string"},
                                              "fit": {"type": "string", "enum": ["fit", "cover"]},
                                              "start": {"type": "number", "minimum": 0},
                                              "end": {"type": "number", "minimum": 0}},
                               "required": ["kind"]},
                    "min_seconds": {"type": "number", "minimum": 0, "maximum": 600},
                },
                "required": ["visual"]}},
            "output": {"type": "string", "description": "workspace-relative .mp4 path"},
            "resolution": {"type": "string", "enum": list(RESOLUTIONS)},
            "fps": {"type": "integer", "minimum": 10, "maximum": 60},
            "theme": {"type": "string", "enum": list(I.THEMES)},
            "voice": {"type": "string", "maxLength": 80},
            "tts_backend": {"type": "string", "enum": ["auto", *T.BACKENDS]},
            "ken_burns": {"type": ["string", "boolean"], "description": "auto (stills only), true or false"},
            "crossfade": {"type": "number", "minimum": 0, "maximum": 2},
            "captions": {"type": "string", "enum": ["burn", "srt", "none"]},
            "music": {"type": "string", "description": "optional workspace audio file you provided"},
            "seed": {"type": "integer"},
        },
        "required": ["scenes"],
    },
    risk="sandbox", category="video", needs_approval=_explainer_estimate)
def make_explainer_video(ctx: ToolContext, scenes: list, output: str | None = None, resolution: str = "1080p",
                         fps: int = 30, theme: str = "dark", voice: str | None = None, tts_backend: str = "auto",
                         ken_burns="auto", crossfade: float = 0.5, captions: str = "burn", music: str | None = None,
                         seed: int = 0) -> ToolResult:
    out = _out(ctx, output, "media/explainer.mp4", ".mp4")
    if isinstance(ken_burns, str) and ken_burns.lower() in ("true", "false"):
        ken_burns = ken_burns.lower() == "true"
    r = V.make_explainer(scenes, out, ctx.workspace, size=_size(resolution), fps=fps, theme_name=theme,
                         tts_backend=tts_backend, voice=voice, ken_burns=ken_burns, crossfade=crossfade,
                         captions=captions, music=music, seed=seed, settings=ctx.settings, limits=_limits(ctx),
                         progress=ctx.progress, cancelled=ctx.cancelled)
    speech = ", ".join(r["tts_backends"]) or "none"
    msg = (f"Rendered {_rel(ctx, r['video'])}: {r['duration_s']:.1f} s, {r['size'][0]}x{r['size'][1]} @ {fps} fps, "
           f"{len(scenes)} scenes. Speech: {speech}. Captions: {r['caption_mode']} (+ {_rel(ctx, r['srt'])}).")
    if r["silent_narration"]:
        msg += " No speech engine was found, so the narration is silent and shown only as captions."
    return ToolResult(True, msg, [ctx.artifact(r["video"], "video", "Explainer video", duration_s=r["duration_s"]),
                                  ctx.artifact(r["srt"], "text", "Captions (SRT)"),
                                  ctx.artifact(r["storyboard"], "data", "Storyboard")],
                      data={k: (str(v) if isinstance(v, Path) else v) for k, v in r.items()})


# ----------------------------------------------------------------------------- slides and images

@tool(
    name="make_slides",
    description="Render 16:9 slides (title, bullets, code with syntax highlighting, image, equation, chart, "
                "diagram) to PNG files and one PDF.",
    parameters={"type": "object", "properties": {
        "slides": {"type": "array", "minItems": 1, "maxItems": 200, "items": _SLIDE_SCHEMA},
        "output_dir": {"type": "string"}, "theme": {"type": "string", "enum": list(I.THEMES)},
        "resolution": {"type": "string", "enum": list(RESOLUTIONS)}, "pdf": {"type": "boolean"}},
        "required": ["slides"]},
    risk="sandbox", category="image")
def make_slides(ctx: ToolContext, slides: list, output_dir: str | None = None, theme: str = "dark",
                resolution: str = "1080p", pdf: bool = True) -> ToolResult:
    out = ctx.path(output_dir or "media/slides")
    r = S.render_deck(slides, out, ctx.workspace, size=_size(resolution), th=theme, pdf=pdf,
                      progress=ctx.progress, cancelled=ctx.cancelled)
    arts = [ctx.artifact(p, "image", f"Slide {i}") for i, p in enumerate(r["pngs"], 1)]
    if r["pdf"]:
        arts.append(ctx.artifact(r["pdf"], "other", "Slides (PDF)"))
    return ToolResult(True, f"Rendered {len(r['pngs'])} slides at {r['size'][0]}x{r['size'][1]} in "
                            f"{_rel(ctx, out)}" + (" plus slides.pdf." if r["pdf"] else "."), arts)


@tool(
    name="make_image",
    description=("Make an image without a generative model: a chart (line/bar/scatter), a box-and-arrow diagram "
                 "from a JSON spec, a title card, a contact sheet of workspace images, or edits of a workspace "
                 "image (resize, crop, text, rectangle, arrow)."),
    parameters={"type": "object", "properties": {
        "kind": {"type": "string", "enum": ["chart", "diagram", "title_card", "contact_sheet", "edit"]},
        "spec": {"type": "object", "description": "chart: {kind, x?, series, xlabel?, ylabel?, title?}; diagram: "
                 "{title?, nodes: [{id, label, x?, y?}], edges: [{from, to, label?}]}; title_card: {title, "
                 "subtitle?, kicker?}; contact_sheet: {images: [paths], labels?, cols?}; edit: {path, ops: [...]}"},
        "output": {"type": "string"}, "theme": {"type": "string", "enum": list(I.THEMES)},
        "width": {"type": "integer", "minimum": 64, "maximum": 4096},
        "height": {"type": "integer", "minimum": 64, "maximum": 4096}},
        "required": ["kind", "spec"]},
    risk="sandbox", category="image")
def make_image(ctx: ToolContext, kind: str, spec: dict, output: str | None = None, theme: str = "dark",
               width: int = 1600, height: int = 900) -> ToolResult:
    size = (int(width), int(height))
    lim = _limits(ctx)
    if kind == "chart":
        im = I.chart(spec, size, theme)
    elif kind == "diagram":
        im = I.diagram(spec, size, theme)
    elif kind == "title_card":
        im = I.title_card(str(spec.get("title", "")), str(spec.get("subtitle", "")), size, theme,
                          str(spec.get("kicker", "")))
    elif kind == "contact_sheet":
        paths = spec.get("images") or []
        if not paths or len(paths) > 200:
            raise MediaError("contact_sheet needs 1..200 image paths")
        ims = [open_image_safely(ctx.path(p), lim, root=ctx.workspace) for p in paths]
        im = I.contact_sheet(ims, int(spec.get("cols", 4)), int(spec.get("thumb_width", 320)), spec.get("labels"),
                             theme)
    elif kind == "edit":
        if not spec.get("path"):
            raise MediaError("edit needs spec.path")
        im = I.edit_image(open_image_safely(ctx.path(spec["path"]), lim, root=ctx.workspace), spec.get("ops") or [],
                          theme)
    else:
        raise MediaError(f"unknown kind {kind!r}")
    out = _out(ctx, output, f"media/{kind}.png", ".png")
    I.save_png(im, out, {"jev": json.dumps({"kind": kind})}, root=ctx.workspace)
    return ToolResult(True, f"Saved {_rel(ctx, out)} ({im.width}x{im.height}).",
                      [ctx.artifact(out, "image", kind.replace("_", " ").title())])


# ----------------------------------------------------------------------------- speech

@tool(
    name="text_to_speech",
    description="Speak text with a local voice into a WAV file (Piper if a voice is installed, else espeak-ng, "
                "macOS say or Windows SAPI; silence plus a note if none exists).",
    parameters={"type": "object", "properties": {
        "text": {"type": "string", "minLength": 1, "maxLength": T.MAX_TEXT_CHARS},
        "output": {"type": "string"}, "voice": {"type": "string", "maxLength": 80},
        "backend": {"type": "string", "enum": ["auto", *T.BACKENDS]},
        "rate": {"type": "number", "minimum": 0.5, "maximum": 2.0}},
        "required": ["text"]},
    risk="sandbox", category="video")
def text_to_speech(ctx: ToolContext, text: str, output: str | None = None, voice: str | None = None,
                   backend: str = "auto", rate: float = 1.0) -> ToolResult:
    out = _out(ctx, output, "media/speech.wav", ".wav")
    r = T.synthesize(text, out, ctx.workspace, backend=backend, voice=voice, rate=rate, settings=ctx.settings)
    msg = f"Spoke {r.duration_s:.1f} s with {r.backend}" + (f" ({r.voice})" if r.voice else "") + \
        f" -> {_rel(ctx, r.path)}."
    if r.silent:
        msg += " No speech engine is installed: the file is silence of the estimated length."
    return ToolResult(True, msg, [ctx.artifact(r.path, "audio", "Speech")], data=r.to_json())


@tool(
    name="download_voice",
    description="Download a Piper neural voice (e.g. en_US-lessac-medium) from huggingface.co/rhasspy/piper-voices "
                "into the local voices folder. Checked against the published index; asks first.",
    parameters={"type": "object", "properties": {"voice": {"type": "string", "pattern": T._VOICE_RE.pattern}},
                "required": ["voice"]},
    risk="approve", category="models")
def download_voice(ctx: ToolContext, voice: str) -> ToolResult:
    r = T.download_voice(voice, ctx.settings, progress=ctx.progress)
    return ToolResult(True, f"Voice {voice} saved in {r['folder']}. Install Piper (`pip install piper-tts`) to use "
                            "it; text_to_speech and explainers will pick it automatically.", data=r)


# ----------------------------------------------------------------------------- probing and footage

@tool(
    name="probe_media",
    description="Read a workspace media file's metadata (duration, size, fps, codecs, audio) without decoding it, "
                "and say whether it is within the editing limits.",
    parameters={"type": "object", "properties": {"path": {"type": "string"}}, "required": ["path"]},
    risk="read", category="video")
def probe_media(ctx: ToolContext, path: str) -> ToolResult:
    p = ctx.path(path)
    lim = _limits(ctx)
    if p.suffix.lower() in V.IMAGE_EXT - {".gif"}:
        im = open_image_safely(p, lim, root=ctx.workspace)
        return ToolResult(True, f"{p.name}: image {im.width}x{im.height}, {p.stat().st_size / 1024:.0f} KB.",
                          data={"kind": "image", "width": im.width, "height": im.height, "bytes": p.stat().st_size})
    info = F.probe(p, ctx.workspace, lim, check=False)
    try:
        F.check_limits(info, lim, "footage")
        ok, why = True, ""
    except MediaError as e:
        ok, why = False, str(e)
    parts = [f"{info.duration_s:.2f} s"]
    if info.has_video:
        parts.append(f"{info.width}x{info.height} @ {info.fps:.3g} fps ({info.video_codec})")
    if info.has_audio:
        parts.append(f"audio {info.audio_codec} {info.audio_rate} Hz x{info.audio_channels}")
    msg = f"{p.name}: " + ", ".join(parts) + "." + ("" if ok else f" Over the limits: {why}")
    return ToolResult(True, msg, data={**info.to_json(), "within_limits": ok})


def _shot_sheet(ctx: ToolContext, a: FT.Analysis, times: list[float], labels: list[str], out: Path) -> Path:
    thumbs = [FT.frame_at(a.path, ctx.workspace, t, 320, a.info) for t in times]
    wsfs.save_image(I.contact_sheet(thumbs, cols=min(4, len(thumbs)), thumb_w=320, labels=labels), ctx.workspace, out)
    return out


_FOOTAGE_COMMON = {"path": {"type": "string", "description": "workspace video"},
                   "encoder_run": {"type": "string", "description": "optional jepa_studio run name for features"}}


@tool(
    name="find_scenes",
    description="Find the shots (scene cuts) in a workspace video. Returns shot times and a contact sheet.",
    parameters={"type": "object", "properties": {
        **_FOOTAGE_COMMON, "sensitivity": {"type": "number", "minimum": 1, "maximum": 20},
        "min_shot_seconds": {"type": "number", "minimum": 0.1, "maximum": 30}}, "required": ["path"]},
    risk="sandbox", category="video")
def find_scenes(ctx: ToolContext, path: str, encoder_run: str | None = None, sensitivity: float = 6.0,
                min_shot_seconds: float = 0.4) -> ToolResult:
    a = FT.analyze(ctx.path(path), ctx.workspace, encoder_run=_run_dir(ctx, encoder_run), limits=_limits(ctx),
                   sensitivity=sensitivity, min_shot_s=min_shot_seconds, progress=ctx.progress,
                   cancelled=ctx.cancelled)
    stem = Path(path).stem
    js = _out(ctx, f"media/{stem}.shots.json", "", ".json")
    wsfs.write_text(ctx.workspace, js, json.dumps(a.to_json(), indent=2))
    arts = [ctx.artifact(js, "data", "Shots")]
    shown = a.shots[:24]
    if shown:
        sheet = _out(ctx, f"media/{stem}.shots.png", "", ".png")
        _shot_sheet(ctx, a, [s.start + min(0.1, s.duration / 2) for s in shown],
                    [f"{s.index + 1}: {s.start:.2f}-{s.end:.2f} s" for s in shown], sheet)
        arts.append(ctx.artifact(sheet, "image", "Shot contact sheet"))
    lines = ", ".join(f"{s.index + 1}) {s.start:.2f}-{s.end:.2f}s" for s in a.shots[:20])
    return ToolResult(True, f"{len(a.shots)} shots in {Path(path).name} ({a.method} features): {lines}"
                      + (" ..." if len(a.shots) > 20 else ""), arts, data=a.to_json())


@tool(
    name="search_footage",
    description="Find the shots in a workspace video that look most like a query image (or like a given moment of "
                "the same video). Image-based search; text queries are not supported.",
    parameters={"type": "object", "properties": {
        **_FOOTAGE_COMMON, "query_image": {"type": "string"}, "query_time": {"type": "number", "minimum": 0},
        "top_k": {"type": "integer", "minimum": 1, "maximum": 20}}, "required": ["path"]},
    risk="sandbox", category="video")
def search_footage(ctx: ToolContext, path: str, query_image: str | None = None, query_time: float | None = None,
                   top_k: int = 5, encoder_run: str | None = None) -> ToolResult:
    if (query_image is None) == (query_time is None):
        raise MediaError("give exactly one of query_image or query_time")
    run = _run_dir(ctx, encoder_run)
    a = FT.analyze(ctx.path(path), ctx.workspace, encoder_run=run, limits=_limits(ctx), progress=ctx.progress,
                   cancelled=ctx.cancelled)
    q = FT.query_from_image(a, ctx.path(query_image), ctx.workspace, run, _limits(ctx)) if query_image \
        else FT.query_from_time(a, float(query_time))
    hits = FT.search(a, q, top_k)
    sheet = _out(ctx, f"media/{Path(path).stem}.search.png", "", ".png")
    _shot_sheet(ctx, a, [h["best_time"] for h in hits],
                [f"#{i + 1} {h['best_time']:.2f}s sim {h['similarity']:.2f}" for i, h in enumerate(hits)], sheet)
    txt = "; ".join(f"#{i + 1} shot {h['index'] + 1} ({h['start']:.2f}-{h['end']:.2f}s, best {h['best_time']:.2f}s, "
                    f"similarity {h['similarity']:.2f})" for i, h in enumerate(hits))
    return ToolResult(True, f"Best matches ({a.method}): {txt}", [ctx.artifact(sheet, "image", "Search results")],
                      data={"hits": hits, "method": a.method})


def _cut_artifacts(ctx: ToolContext, r: dict, title: str) -> list:
    arts = [ctx.artifact(r["video"], "video", title, duration_s=r["duration_s"]),
            ctx.artifact(r["cuts_json"], "data", "Cut list (JSON)"), ctx.artifact(r["edl"], "text", "Cut list (EDL)")]
    if r.get("srt"):
        arts.append(ctx.artifact(r["srt"], "text", "Captions (SRT)"))
    return arts


@tool(
    name="make_highlight_reel",
    description="Make a highlight reel of a workspace video within a time budget: diverse, high-motion moments "
                "chosen automatically, joined into an MP4, plus the cut list (JSON and EDL).",
    parameters={"type": "object", "properties": {
        **_FOOTAGE_COMMON, "budget_seconds": {"type": "number", "minimum": 1, "maximum": 1800},
        "output": {"type": "string"}, "min_clip_seconds": {"type": "number", "minimum": 0.2, "maximum": 30},
        "max_clip_seconds": {"type": "number", "minimum": 0.5, "maximum": 60}, "title": {"type": "string",
                                                                                      "maxLength": 120},
        "captions": {"type": "boolean"}}, "required": ["path", "budget_seconds"]},
    risk="sandbox", category="video")
def make_highlight_reel(ctx: ToolContext, path: str, budget_seconds: float, output: str | None = None,
                        min_clip_seconds: float = 1.0, max_clip_seconds: float = 4.0, title: str | None = None,
                        captions: bool = False, encoder_run: str | None = None) -> ToolResult:
    src = ctx.path(path)
    a = FT.analyze(src, ctx.workspace, encoder_run=_run_dir(ctx, encoder_run), limits=_limits(ctx),
                   progress=lambda f, m: ctx.progress(0.4 * f, m), cancelled=ctx.cancelled)
    cuts = FT.select_highlights(a, float(budget_seconds), min_clip_s=min_clip_seconds, max_clip_s=max_clip_seconds)
    out = _out(ctx, output, f"media/{src.stem}.highlights.mp4", ".mp4")
    r = FT.render_cuts(src, cuts, out, ctx.workspace, title=title, captions=captions, limits=_limits(ctx),
                       progress=lambda f, m: ctx.progress(0.4 + 0.6 * f, m), cancelled=ctx.cancelled)
    return ToolResult(True, f"Highlight reel {_rel(ctx, r['video'])}: {len(cuts)} clips, {r['duration_s']:.1f} s "
                            f"(budget {budget_seconds:g} s, {r['mode']}, {a.method} features).",
                      _cut_artifacts(ctx, r, "Highlight reel"),
                      data={"cuts": [c.to_json() for c in cuts], "mode": r["mode"], "duration_s": r["duration_s"]})


@tool(
    name="cut_video",
    description="Cut and join parts of a workspace video from a cut list [{start, end, label?}] (seconds). Uses "
                "lossless stream copy when cuts start on keyframes, else re-encodes. Optional title card and "
                "per-clip captions. Also writes the cut list as JSON and EDL.",
    parameters={"type": "object", "properties": {
        "path": {"type": "string"},
        "cuts": {"type": "array", "minItems": 1, "maxItems": 500, "items": {"type": "object", "properties": {
            "start": {"type": "number", "minimum": 0}, "end": {"type": "number", "minimum": 0},
            "label": {"type": "string", "maxLength": 200}}, "required": ["start", "end"]}},
        "output": {"type": "string"}, "mode": {"type": "string", "enum": ["auto", "copy", "reencode"]},
        "title": {"type": "string", "maxLength": 120}, "captions": {"type": "boolean"}},
        "required": ["path", "cuts"]},
    risk="sandbox", category="video")
def cut_video(ctx: ToolContext, path: str, cuts: list, output: str | None = None, mode: str = "auto",
              title: str | None = None, captions: bool = False) -> ToolResult:
    src = ctx.path(path)
    cl = [FT.Cut(float(c["start"]), float(c["end"]), str(c.get("label", ""))) for c in cuts]
    out = _out(ctx, output, f"media/{src.stem}.cut.mp4", ".mp4")
    r = FT.render_cuts(src, cl, out, ctx.workspace, mode=mode, title=title, captions=captions, limits=_limits(ctx),
                       progress=ctx.progress, cancelled=ctx.cancelled)
    note = f" ({'; '.join(r['notes'])})" if r["notes"] else ""
    return ToolResult(True, f"Cut {_rel(ctx, r['video'])}: {len(cl)} parts, {r['duration_s']:.2f} s, "
                            f"{r['mode']}{note}.", _cut_artifacts(ctx, r, "Edited video"),
                      data={"mode": r["mode"], "duration_s": r["duration_s"]})


# ----------------------------------------------------------------------------- generative (approval)

_GEN_COMMON = {"model": {"type": "string", "description": "catalog name or a folder name in the models folder"},
               "prompt": {"type": "string", "minLength": 1, "maxLength": 2000},
               "negative_prompt": {"type": "string", "maxLength": 2000}, "seed": {"type": "integer"},
               "steps": {"type": "integer", "minimum": 1, "maximum": 100},
               "guidance": {"type": "number", "minimum": 0, "maximum": 30},
               "width": {"type": "integer", "minimum": 256, "maximum": 2048},
               "height": {"type": "integer", "minimum": 256, "maximum": 2048}, "output": {"type": "string"}}


@tool(
    name="generate_image",
    description=f"Generate an image from text with a local diffusion model ({', '.join(n for n, s in G.CATALOG.items() if s.kind == 'image')}). "
                "For artistic or photographic pictures only: NOT for black holes or physics (use render_black_hole "
                "or simulate_physics: exact, free and no download), charts (make_plot) or diagrams, title cards "
                "and slides (make_image). Needs the model downloaded (see 'Generative models installed' in the "
                "system prompt) and diffusers installed; uses the GPU. Leave steps, guidance and size unset: each "
                "model has tuned defaults. Always asks first.",
    parameters={"type": "object", "properties": dict(_GEN_COMMON), "required": ["model", "prompt"]},
    risk="approve", category="image")
def generate_image(ctx: ToolContext, model: str, prompt: str, negative_prompt: str | None = None, seed: int = 0,
                   steps: int | None = None, guidance: float | None = None, width: int | None = None,
                   height: int | None = None, output: str | None = None) -> ToolResult:
    out = _out(ctx, output, "media/generated.png", ".png")
    r = G.generate_image(model, prompt, out, ctx.workspace, seed=seed, steps=steps, guidance=guidance, width=width,
                         height=height, negative_prompt=negative_prompt, settings=ctx.settings,
                         progress=ctx.progress, cancelled=ctx.cancelled)
    m = r["meta"]
    return ToolResult(True, f"Generated {_rel(ctx, r['image'])} with {m['model']} ({m['width']}x{m['height']}, "
                            f"{m['steps']} steps, seed {seed}) on {m['device']} in {m['seconds']:.0f} s."
                            + (f" Note: {m['adjusted']}." if m.get("adjusted") else ""),
                      [ctx.artifact(r["image"], "image", "Generated image", **m)], data=m)


@tool(
    name="generate_video",
    description=f"Generate a short video clip from text with a local model ({', '.join(n for n, s in G.CATALOG.items() if s.kind == 'video')}). "
                "For artistic or photographic footage only: NOT for black holes or physics (render_black_hole with "
                "frames > 1, simulate_physics) or explainers (make_explainer_video). Needs a GPU (refused on CPU), "
                "the model downloaded and diffusers installed. Leave steps and guidance unset. Always asks first.",
    parameters={"type": "object", "properties": {**_GEN_COMMON, "frames": {"type": "integer", "minimum": 9,
                                                                           "maximum": 257},
                                                 "fps": {"type": "integer", "minimum": 8, "maximum": 30}},
                "required": ["model", "prompt"]},
    risk="approve", category="video")
def generate_video(ctx: ToolContext, model: str, prompt: str, negative_prompt: str | None = None, seed: int = 0,
                   steps: int | None = None, guidance: float | None = None, width: int | None = None,
                   height: int | None = None, frames: int | None = None, fps: int | None = None,
                   output: str | None = None) -> ToolResult:
    out = _out(ctx, output, "media/generated.mp4", ".mp4")
    r = G.generate_video(model, prompt, out, ctx.workspace, seed=seed, steps=steps, guidance=guidance, width=width,
                         height=height, frames=frames, fps=fps, negative_prompt=negative_prompt,
                         settings=ctx.settings, progress=ctx.progress, cancelled=ctx.cancelled)
    m = r["meta"]
    return ToolResult(True, f"Generated {_rel(ctx, r['video'])} with {m['model']}: {m['frames']} frames at "
                            f"{m['fps']} fps, {m['width']}x{m['height']}, seed {seed}, {m['seconds']:.0f} s.",
                      [ctx.artifact(r["video"], "video", "Generated video", **m)], data=m)


@tool(
    name="download_generative_model",
    description="Download a local generative model (safetensors and configs only) into the models folder: "
                + "; ".join(f"{n} (~{s.download_gb:g} GB, needs ~{s.vram_gb_offload:g}+ GB VRAM, {s.license})"
                            for n, s in G.CATALOG.items()) + ". Always asks first.",
    parameters={"type": "object", "properties": {"model": {"type": "string", "enum": list(G.CATALOG)},
                                                 "revision": {"type": "string", "maxLength": 64}},
                "required": ["model"]},
    risk="approve", category="models")
def download_generative_model(ctx: ToolContext, model: str, revision: str = "main") -> ToolResult:
    r = G.download_model(model, ctx.settings, revision=revision, progress=ctx.progress)
    return ToolResult(True, f"{model} saved in {r['folder']} ({r['files']} files, {r['bytes'] / 2**30:.1f} GB, "
                            "SHA-256 recorded in jev-model.json).", data=r)
