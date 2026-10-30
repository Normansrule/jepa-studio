"""The contract every JEV capability implements.

A tool is a plain Python function plus a description the language model sees. The agent loop
(jev.agent) never calls a tool directly: it asks jev.policy first, which decides from the tool's
declared `risk` and the actual arguments whether the call runs now, needs your approval, or is
refused. Tools therefore declare their risk honestly and keep every file they write inside
`ctx.workspace`.

Risk levels (see docs/jev/SAFETY.md):
    "read"     no side effects outside memory (answering, math, listing workspace files)
    "sandbox"  writes or executes, but only inside the conversation's workspace folder and
               within resource limits (code runs, renders, video assembly)
    "approve"  always asks first: network, files outside the workspace, deleting, installing
               packages or models, long GPU jobs, anything publishing or sending data out

Registering a tool:

    from jev.tools.base import tool, ToolContext, ToolResult

    @tool(
        name="render_black_hole",
        description="Render an image of a black hole with a glowing accretion disk.",
        parameters={
            "type": "object",
            "properties": {"spin": {"type": "number", "minimum": 0, "maximum": 0.999}},
            "required": [],
        },
        risk="sandbox",
    )
    def render_black_hole(ctx: ToolContext, spin: float = 0.0) -> ToolResult:
        path = ctx.workspace / "black_hole.png"
        ...
        return ToolResult(ok=True, content="Rendered a 1024x576 frame.",
                          artifacts=[ctx.artifact(path, kind="image")])

`parameters` is JSON Schema (it is passed to the model as a function-calling schema and
validated with jepa_studio.config.validate before the call). Tools must be deterministic for a
fixed seed where they involve randomness, check `ctx.cancelled()` in long loops, and report
progress with `ctx.progress(fraction, message)`.
"""
from __future__ import annotations

import mimetypes
import threading
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Literal

Risk = Literal["read", "sandbox", "approve"]
ArtifactKind = Literal["image", "video", "audio", "code", "text", "data", "model", "other"]

RISKS: tuple[str, ...] = ("read", "sandbox", "approve")


class ToolError(Exception):
    """A tool failed in an expected way (bad input, missing optional dependency). The message is
    shown to the model and the user, so it should say what to do next."""


@dataclass
class Artifact:
    """A file a tool produced, always inside the workspace."""
    path: Path                      # absolute path inside ctx.workspace
    kind: ArtifactKind
    mime: str
    title: str = ""
    meta: dict = field(default_factory=dict)

    def to_json(self, workspace: Path) -> dict:
        return {"path": self.path.relative_to(workspace).as_posix(), "kind": self.kind,
                "mime": self.mime, "title": self.title, "meta": self.meta,
                "bytes": self.path.stat().st_size if self.path.exists() else 0}


@dataclass
class ToolResult:
    ok: bool
    content: str                    # text the model reads next (keep it short; no file dumps)
    artifacts: list[Artifact] = field(default_factory=list)
    data: dict = field(default_factory=dict)   # structured extras for the UI (not sent to the model)


@dataclass
class ToolContext:
    workspace: Path                 # this conversation's sandbox folder (created by the caller)
    log: Callable[[str], None] = print
    _progress: Callable[[float, str], None] | None = None
    _cancel: threading.Event = field(default_factory=threading.Event)
    settings: dict = field(default_factory=dict)   # resolved jev settings (limits, models, paths)

    def progress(self, fraction: float, message: str = "") -> None:
        if self._progress:
            self._progress(max(0.0, min(1.0, float(fraction))), message)

    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def path(self, relative: str) -> Path:
        """Resolve a model-supplied relative path inside the workspace; refuse escapes."""
        p = (self.workspace / relative).resolve()
        ws = self.workspace.resolve()
        if p != ws and ws not in p.parents:
            raise ToolError(f"{relative!r} is outside this conversation's workspace")
        return p

    def artifact(self, path: Path, kind: ArtifactKind, title: str = "", **meta) -> Artifact:
        path = Path(path).resolve()
        if self.workspace.resolve() not in path.parents:
            raise ToolError(f"artifact {path} is outside the workspace")
        mime = mimetypes.guess_type(path.name)[0] or "application/octet-stream"
        return Artifact(path=path, kind=kind, mime=mime, title=title or path.name, meta=meta)


@dataclass
class ToolSpec:
    name: str
    description: str
    parameters: dict
    risk: Risk
    fn: Callable[..., ToolResult]
    # Optional: decide per call whether a "sandbox" tool must still ask (e.g. a render that
    # would take > 10 minutes, or a GPU job). Returns a reason string to ask, or None.
    needs_approval: Callable[[dict, ToolContext], str | None] | None = None
    category: str = "general"       # answer | code | image | video | science | files | models

    def schema(self) -> dict:
        """Function-calling schema in the format Ollama's /api/chat `tools` field expects."""
        return {"type": "function",
                "function": {"name": self.name, "description": self.description,
                             "parameters": self.parameters}}


REGISTRY: dict[str, ToolSpec] = {}


def tool(name: str, description: str, parameters: dict, risk: Risk, category: str = "general",
         needs_approval: Callable[[dict, ToolContext], str | None] | None = None):
    """Decorator that registers a tool. Names are unique; re-registering replaces (for tests)."""
    if risk not in RISKS:
        raise ValueError(f"risk must be one of {RISKS}")
    if parameters.get("type") != "object":
        raise ValueError("tool parameters must be a JSON Schema object")

    def wrap(fn: Callable[..., ToolResult]) -> Callable[..., ToolResult]:
        REGISTRY[name] = ToolSpec(name, description, parameters, risk, fn, needs_approval, category)
        return fn
    return wrap


def load_all() -> dict[str, ToolSpec]:
    """Import every tool module so its @tool decorators run. Optional modules whose heavy
    dependencies are missing still register (they raise ToolError with install advice on use)."""
    import importlib
    for mod in ("jev.tools.core", "jev.tools.code", "jev.tools.science", "jev.tools.media"):
        try:
            importlib.import_module(mod)
        except ModuleNotFoundError as e:
            if e.name and e.name.startswith("jev.tools"):
                continue        # module not written yet
            raise
    return REGISTRY


def describe(spec: ToolSpec) -> dict[str, Any]:
    return {"name": spec.name, "description": spec.description, "risk": spec.risk,
            "category": spec.category, "parameters": spec.parameters}
