"""Code tools: run Python / Node / shell in the sandbox, install packages into a per-conversation
venv, scaffold projects, and use JEPA Studio's own models (embed images with a trained run, start
training).

run_python and run_node auto-run inside the sandbox unless their code trips a row of the approval
table (network, shell, installs, deletes, paths outside the workspace; see jev/policy.py) or the
sandbox is weak (the locked weak_sandbox row); run_shell, install_package and jepa_train always ask.

The package venv of a conversation lives in jev-data/venvs/<workspace>, never in the workspace:
code runs see it read-only (where the OS gives a mount namespace), only an approved
install_package run can write to it (and then the workspace is read-only), and its interpreter is
checked to still be this Python before anything runs it.
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import secrets
import shlex
import shutil
import sys
import threading
from pathlib import Path

from .. import wsfs
from ..policy import flag, scan_node, scan_python, scan_shell
from ..sandbox import DATA_DIRNAME, IS_WIN, Limits, Mounts, Sandbox, python_for, sandbox_for, trusted_venv_python
from .base import ToolContext, ToolError, ToolResult, tool
from .core import artifact_kind

PKG_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,99}(\[[A-Za-z0-9,._-]{1,100}\])?"
                    r"((==|>=|<=|~=|!=|>|<)[A-Za-z0-9.*+!_-]{1,40})?(,(==|>=|<=|~=|!=|>|<)[A-Za-z0-9.*+!_-]{1,40}){0,3}$")
NAME_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,63}$")


def get_sandbox(ctx: ToolContext) -> Sandbox:
    return sandbox_for(ctx.settings)


# ------------------------------------------------------------------------------------- package venv
def venv_dir(ctx: ToolContext) -> Path:
    """jev-data/venvs/<workspace folder name>: outside the workspace, so sandboxed code cannot plant
    an interpreter there."""
    ws = ctx.workspace.resolve()
    data = ctx.settings.get("data_dir")
    if not data and ws.parent.parent.name == DATA_DIRNAME:
        data = ws.parent.parent
    if not data:
        raise ToolError("install_package needs JEV's data folder (jev-data): run it through JEV")
    name = ws.name if NAME_RE.match(ws.name) else "ws-" + hashlib.sha256(str(ws).encode()).hexdigest()[:16]
    return Path(data).resolve() / "venvs" / name


def _venv_for_run(ctx: ToolContext) -> tuple[Path | None, str]:
    """(venv, note) for a code run: the conversation's package venv if it exists and is intact."""
    try:
        v = venv_dir(ctx)
    except ToolError:
        return None, ""
    if not v.exists():
        return None, ""
    if trusted_venv_python(v) is None:
        return None, ("note: the installed-packages venv failed its integrity check and was not used "
                      "(install_package recreates it)\n")
    return v, ""


def pip_argv(ctx: ToolContext, packages: list, build_from_source: bool = False) -> list[str]:
    """The exact pip command install_package runs (shown on the approval card). --isolated: pip
    reads no user configuration and no PIP_* variables (HOME is also an empty folder and
    PIP_CONFIG_FILE is the null device, so a pip.conf, .netrc or .pydistutils.cfg planted in the
    workspace is never read); --only-binary :all: installs wheels only, so no package's own build
    code runs, unless build_from_source; a per-conversation cache."""
    try:
        v = venv_dir(ctx)
    except ToolError:
        v = Path("<jev-data>") / "venvs" / ctx.workspace.name
    exe = v / ("Scripts/python.exe" if IS_WIN else "bin/python")
    argv = [str(exe), "-I", "-m", "pip", "--isolated", "install", "--no-input", "--disable-pip-version-check",
            "--index-url", "https://pypi.org/simple", "--cache-dir", str(v.parent / f"{v.name}.pip-cache")]
    if not build_from_source:
        argv += ["--only-binary", ":all:"]
    cert = os.environ.get("PIP_CERT")
    if cert and os.path.isfile(cert):
        argv += ["--cert", cert]                      # your own CA bundle (PIP_CERT is ignored under --isolated)
    return argv + ["--", *[str(p) for p in packages]]


def _limits(ctx: ToolContext, timeout_s: int | None = None) -> Limits:
    lim = Limits.from_settings(ctx.settings) if ctx.settings.get("limits") else get_sandbox(ctx).limits
    if timeout_s:
        lim = Limits(**{**lim.__dict__, "wall_seconds": min(int(timeout_s), lim.wall_seconds)})
    return lim


def _network_ok(ctx: ToolContext) -> bool:
    """Network for this run only when the decision covered the network (approved by you, remembered
    is impossible for these rows, or set to "run" in your approval_rules)."""
    approved = set(ctx.settings.get("_approved_rules") or [])
    return bool(approved & {"network", "install", "send_data"})


def _isolated(ctx: ToolContext) -> bool:
    return get_sandbox(ctx).capabilities().get("network_isolation") == "enforced"


def _snippet(code: str, suffix: str) -> Path:
    """The program text, in backend-owned scratch (never in the workspace, where a planted link
    could redirect the write); the run sees it read-only. Call inside wsfs.use_scratch()."""
    d = wsfs.new_scratch("run-")
    return wsfs.write_text(d, f"run-{secrets.token_hex(4)}{suffix}", code)


def _result(ctx: ToolContext, res, label: str, note: str = "") -> ToolResult:
    arts = []
    ws = ctx.workspace.resolve()
    for rel in res.changed_files[:20]:
        p = (ws / rel)
        k = artifact_kind(p)
        if p.is_file() and not p.is_symlink():
            try:
                arts.append(ctx.artifact(p, kind=k))
            except ToolError:
                pass
    return ToolResult(ok=res.ok, content=note + f"{label}: " + res.summary(),
                      artifacts=arts, data={"returncode": res.returncode, "timed_out": res.timed_out,
                                            "isolation": res.isolation, "seconds": round(res.seconds, 2)})


def _tags(hits) -> str | None:
    return "\n".join(flag(r, t) for r, t in hits) or None


# ------------------------------------------------------------------------------------- run_python
def _python_needs(args: dict, ctx: ToolContext) -> str | None:
    return _tags(scan_python(str(args.get("code", "")), _isolated(ctx)))


@tool(name="run_python",
      description="Run a Python 3 program in this conversation's sandboxed workspace (cwd = workspace, no network, "
                  "time and memory limits). Print results; save figures/files into the workspace and they are shown "
                  "to the user. numpy, torch, PIL and matplotlib are usually available.",
      parameters={"type": "object", "properties": {
          "code": {"type": "string", "maxLength": 200000},
          "timeout_s": {"type": "integer", "minimum": 1, "maximum": 86400}}, "required": ["code"]},
      risk="sandbox", category="code", needs_approval=_python_needs, runs_code=True)
def run_python(ctx: ToolContext, code: str, timeout_s: int | None = None) -> ToolResult:
    ws = str(ctx.workspace.resolve())
    boot = ("import sys, runpy; sys.path.insert(0, sys.argv[1]); sys.argv = [sys.argv[2]]; "
            "runpy.run_path(sys.argv[0], run_name='__main__')")
    venv, note = _venv_for_run(ctx)
    with wsfs.use_scratch(ctx.scratch()):
        script = _snippet(code, ".py")
        argv = [python_for(venv=venv), "-I", "-c", boot, ws, str(script)]
        res = get_sandbox(ctx).run(argv, ctx.workspace, _limits(ctx, timeout_s), network=_network_ok(ctx),
                                   cancelled=ctx.cancelled, venv=venv,
                                   mounts=Mounts(ro=[script.parent] + ([venv] if venv else [])))
    return _result(ctx, res, "python", note)


# ------------------------------------------------------------------------------------- run_node
def _node_needs(args: dict, ctx: ToolContext) -> str | None:
    return _tags(scan_node(str(args.get("code", "")), _isolated(ctx)))


@tool(name="run_node",
      description="Run a JavaScript program with Node.js in the sandboxed workspace (no network, time and memory "
                  "limits). Use console.log for results.",
      parameters={"type": "object", "properties": {
          "code": {"type": "string", "maxLength": 200000},
          "timeout_s": {"type": "integer", "minimum": 1, "maximum": 86400}}, "required": ["code"]},
      risk="sandbox", category="code", needs_approval=_node_needs, runs_code=True)
def run_node(ctx: ToolContext, code: str, timeout_s: int | None = None) -> ToolResult:
    node = shutil.which("node")
    if not node:
        raise ToolError("Node.js is not installed. Install it from https://nodejs.org (LTS) or use run_python.")
    esm = bool(re.search(r"^\s*(import|export)\s", code, re.M)) and "require(" not in code
    ws = ctx.workspace.resolve()
    with wsfs.use_scratch(ctx.scratch()):
        script = _snippet(code, ".mjs" if esm else ".cjs")
        res = get_sandbox(ctx).run([node, str(script)], ctx.workspace, _limits(ctx, timeout_s),
                                   network=_network_ok(ctx), cancelled=ctx.cancelled,
                                   extra_env={"NODE_PATH": str(ws / "node_modules")},
                                   mounts=Mounts(ro=[script.parent]))
    return _result(ctx, res, "node")


# ------------------------------------------------------------------------------------- run_shell
def _shell_needs(args: dict, ctx: ToolContext) -> str | None:
    return _tags(scan_shell(str(args.get("command", "")))) or flag("shell", "runs a shell command")


@tool(name="run_shell",
      description="Run a shell command in the workspace (bash on Linux/macOS, PowerShell on Windows). ALWAYS asks "
                  "the user first. Prefer run_python for computation.",
      parameters={"type": "object", "properties": {
          "command": {"type": "string", "maxLength": 20000},
          "shell": {"enum": ["auto", "bash", "sh", "powershell", "cmd"]},
          "timeout_s": {"type": "integer", "minimum": 1, "maximum": 86400}}, "required": ["command"]},
      risk="approve", category="code", needs_approval=_shell_needs, runs_code=True)
def run_shell(ctx: ToolContext, command: str, shell: str = "auto", timeout_s: int | None = None) -> ToolResult:
    if shell == "auto":
        shell = "powershell" if IS_WIN else ("bash" if shutil.which("bash") else "sh")
    exe = {"bash": "bash", "sh": "sh", "powershell": "powershell" if IS_WIN else "pwsh", "cmd": "cmd"}[shell]
    path = shutil.which(exe)
    if not path:
        raise ToolError(f"{exe} is not available on this computer")
    if shell == "powershell":
        argv = [path, "-NoProfile", "-NonInteractive", "-Command", command]
    elif shell == "cmd":
        argv = [path, "/d", "/s", "/c", command]
    else:
        argv = [path, "-c", command]   # the one place model text reaches a shell, after approval
    venv, note = _venv_for_run(ctx)
    res = get_sandbox(ctx).run(argv, ctx.workspace, _limits(ctx, timeout_s), network=_network_ok(ctx),
                               cancelled=ctx.cancelled, venv=venv, mounts=Mounts(ro=[venv] if venv else []))
    return _result(ctx, res, shell, note)


# ------------------------------------------------------------------------------------- install_package
def _install_needs(args: dict, ctx: ToolContext) -> str | None:
    pk = [str(p) for p in args.get("packages") or []]
    build = bool(args.get("build_from_source"))
    try:
        where = str(venv_dir(ctx))
    except ToolError:
        where = "jev-data/venvs"
    text = (f"pip install {', '.join(pk)} into this conversation's package venv ({where}); exact command: "
            f"{shlex.join(pip_argv(ctx, pk, build))}")
    text += ("; BUILDS FROM SOURCE: the packages' own build code runs during the install, with network access"
             if build else "; wheels only (no package build code runs during the install)")
    return flag("install", text) + "\n" + \
        flag("network", "downloads from the Python Package Index (pypi.org) only: pip ignores every configuration "
                        "file and variable a program could have planted; only the venv and its pip cache are "
                        "writable during the install, the workspace is read-only")


SITE_PIP_CONFIGS = ("pip.conf", "pip.ini")     # pip's per-environment ("site") configuration files


@tool(name="install_package",
      description="Install Python packages with pip into this conversation's own virtual environment "
                  "(kept in jev-data, outside the workspace; run_python then uses it). ALWAYS asks the user first.",
      parameters={"type": "object", "properties": {
          "packages": {"type": "array", "minItems": 1, "maxItems": 20, "items": {"type": "string", "maxLength": 200}},
          "manager": {"enum": ["pip"]},
          "build_from_source": {"type": "boolean", "description": "allow source distributions (runs their build "
                                                                  "code); default false: wheels only"}},
          "required": ["packages"]},
      risk="approve", category="code", needs_approval=_install_needs, runs_code=True)
def install_package(ctx: ToolContext, packages: list, manager: str = "pip",
                    build_from_source: bool = False) -> ToolResult:
    from ..settings import private_dir

    bad = [p for p in packages if not PKG_RE.match(p)]
    if bad:
        raise ToolError(f"not a plain package name/version: {', '.join(bad)} (options, URLs and paths are refused)")
    sb = get_sandbox(ctx)
    base = _limits(ctx)
    lim = Limits(**{**base.__dict__, "wall_seconds": max(base.wall_seconds, 900),
                    "cpu_seconds": max(base.cpu_seconds, 600)})
    venv = venv_dir(ctx)
    root = private_dir(venv.parent)
    cache = private_dir(root / f"{venv.name}.pip-cache")
    tmp = private_dir(root / f"{venv.name}.tmp")
    home = private_dir(root / f"{venv.name}.home")          # an empty HOME: no planted config is ever read
    for leftover in list(home.iterdir()):
        wsfs.rmtree(home, leftover.name)
    log = []
    if venv.exists() and trusted_venv_python(venv) is None:
        shutil.rmtree(venv, ignore_errors=True)  # wsfs-ok: the venv lives in jev-data, not in a workspace
        log.append("the existing package venv failed its integrity check (its interpreter or pyvenv.cfg had "
                   "changed), so it was deleted and created again")
    if not venv.exists():
        private_dir(venv)
        # the trusted interpreter (this Python, never a file from the workspace) creates the venv
        res = sb.run([sys.executable, "-I", "-m", "venv", "--system-site-packages", str(venv)], ctx.workspace, lim,
                     cancelled=ctx.cancelled, mounts=Mounts(rw=[venv, tmp, home], workspace_writable=False), tmp=tmp,
                     honor_network_setting=False, extra_env={"HOME": str(home), "XDG_CONFIG_HOME": str(home),
                                                             "PIP_CONFIG_FILE": os.devnull})
        if not res.ok:
            return ToolResult(False, "could not create the package venv: " + res.summary())
        log.append(f"created the package venv {venv} (sees this computer's packages; installs go into it)")
    if trusted_venv_python(venv) is None:
        return ToolResult(False, f"refused: {venv} does not hold this Python's interpreter; nothing was installed")
    # pip reads <venv>/pip.conf (pip.ini on Windows) as "site" configuration even with --isolated and
    # PIP_CONFIG_FILE set: a build script from an earlier install could have planted an index there
    for name in SITE_PIP_CONFIGS:
        if os.path.lexists(venv / name):
            wsfs.rmtree(venv, name)
            log.append(f"removed {name} from the package venv (pip would have read its index settings)")
    argv = pip_argv(ctx, packages, build_from_source)
    res = sb.run(argv, ctx.workspace, lim, network=True, cancelled=ctx.cancelled, venv=venv, tmp=tmp,
                 mounts=Mounts(rw=[venv, cache, tmp, home], workspace_writable=False),
                 extra_env={"HOME": str(home), "USERPROFILE": str(home), "XDG_CONFIG_HOME": str(home),
                            "XDG_CACHE_HOME": str(home), "PIP_CONFIG_FILE": os.devnull, "PYTHONNOUSERSITE": "1",
                            "NETRC": os.devnull})
    log.append("$ " + shlex.join(argv))
    log.append(res.summary(2500))
    return ToolResult(res.ok, "\n".join(log), data={"packages": packages, "venv": str(venv)})


# ------------------------------------------------------------------------------------- create_project
TEMPLATES = {
    "python": {"README.md": "# {name}\n\n{description}\n\nRun: `python main.py`\n",
               "main.py": "\"\"\"{description}\"\"\"\n\n\ndef main():\n    print(\"hello from {name}\")\n\n\n"
                          "if __name__ == \"__main__\":\n    main()\n",
               "requirements.txt": "",
               "tests/test_main.py": "from main import main\n\n\ndef test_main(capsys):\n    main()\n"
                                     "    assert \"{name}\" in capsys.readouterr().out\n"},
    "node": {"README.md": "# {name}\n\n{description}\n\nRun: `node index.js`\n",
             "package.json": "{{\n  \"name\": \"{slug}\",\n  \"version\": \"0.1.0\",\n  \"private\": true,\n"
                             "  \"main\": \"index.js\",\n  \"scripts\": {{\"start\": \"node index.js\"}}\n}}\n",
             "index.js": "// {description}\nconsole.log('hello from {name}');\n"},
    "web": {"index.html": "<!doctype html>\n<html lang=\"en\">\n<head>\n<meta charset=\"utf-8\">\n"
                          "<meta name=\"viewport\" content=\"width=device-width, initial-scale=1\">\n"
                          "<title>{name}</title>\n<link rel=\"stylesheet\" href=\"style.css\">\n</head>\n<body>\n"
                          "<h1>{name}</h1>\n<p>{description}</p>\n<script src=\"app.js\"></script>\n</body>\n</html>\n",
            "style.css": "body {{ font-family: system-ui, sans-serif; margin: 2rem; }}\n",
            "app.js": "// {description}\n"},
    "empty": {"README.md": "# {name}\n\n{description}\n"},
}


@tool(name="create_project",
      description="Create a new project folder in the workspace from a template (python, node, web or empty), "
                  "optionally with extra files you provide.",
      parameters={"type": "object", "properties": {
          "name": {"type": "string", "pattern": NAME_RE.pattern},
          "template": {"enum": sorted(TEMPLATES)},
          "description": {"type": "string", "maxLength": 500},
          "files": {"type": "object", "description": "extra files: {relative path: content}"}},
          "required": ["name"]},
      risk="sandbox", category="code")
def create_project(ctx: ToolContext, name: str, template: str = "python", description: str = "",
                   files: dict | None = None) -> ToolResult:
    root = ctx.path(name)
    if root.exists() and any(root.iterdir()):
        raise ToolError(f"{name}/ already exists and is not empty")
    slug = re.sub(r"[^a-z0-9-]+", "-", name.lower()).strip("-") or "project"
    written = []
    for rel, body in TEMPLATES[template].items():
        wsfs.write_text(ctx.workspace, ctx.path(f"{name}/{rel}"),
                        body.format(name=name, slug=slug, description=description or name))
        written.append(rel)
    for rel, body in (files or {}).items():
        if not isinstance(body, str):
            raise ToolError(f"files[{rel!r}] must be text")
        wsfs.write_text(ctx.workspace, ctx.path(f"{name}/{rel}"), body)
        written.append(rel)
    return ToolResult(True, f"Created {name}/ ({template}): " + ", ".join(written),
                      artifacts=[ctx.artifact(ctx.path(f"{name}/{w}"), "code" if artifact_kind(Path(w)) == "code"
                                              else "text") for w in written[:10]])


# ------------------------------------------------------------------------------------- JEPA tie-ins
RUN_ID = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def _run_dir(ctx: ToolContext, run_id: str) -> Path:
    if not RUN_ID.match(run_id) or run_id in (".", ".."):
        raise ToolError("run_id must be a run folder name (see the Runs tab)")
    root = Path(ctx.settings.get("root") or ".").resolve()
    d = root / "runs" / run_id
    if not (d / "config.json").exists():
        raise ToolError(f"no JEPA Studio run named {run_id!r} under {root / 'runs'}")
    return d


def _load_ws_image(ws: Path, path: Path, max_side: int):
    """jepa_studio.data.loaders.load_image, but opened through jev/wsfs.py: no link followed, a FIFO
    named *.png never blocks the backend."""
    import numpy as np
    from PIL import Image

    from jepa_studio.data.loaders import MAX_FILE_BYTES, MAX_PIXELS
    with wsfs.open_file(ws, path, "rb") as fh:
        if os.fstat(fh.fileno()).st_size > MAX_FILE_BYTES:
            raise ToolError(f"{path.name}: larger than {MAX_FILE_BYTES >> 20} MB")
        with Image.open(fh) as im:
            w, h = im.size
            if w * h > MAX_PIXELS:
                raise ToolError(f"{path.name}: {w}x{h} exceeds the pixel cap")
            im.draft("RGB", (max_side, max_side))
            im = im.convert("RGB")
            im.thumbnail((max_side, max_side), Image.Resampling.BICUBIC)
            return np.asarray(im, dtype=np.float32) / 255.0


@tool(name="jepa_embed_images",
      description="Embed the images in a workspace folder with the encoder of a trained JEPA Studio run (read-only use "
                  "of its hash-verified weights). Saves embeddings.npy and nearest-neighbour pairs to the workspace.",
      parameters={"type": "object", "properties": {
          "run_id": {"type": "string", "maxLength": 128},
          "folder": {"type": "string", "maxLength": 500},
          "max_images": {"type": "integer", "minimum": 1, "maximum": 2048}}, "required": ["run_id"]},
      risk="sandbox", category="answer")
def jepa_embed_images(ctx: ToolContext, run_id: str, folder: str = ".", max_images: int = 256) -> ToolResult:
    import numpy as np
    import torch
    import torch.nn.functional as F

    from jepa_studio.config import load_config
    from jepa_studio.data.loaders import IMAGE_EXT, safe_listing
    from jepa_studio.evaluate import embed
    from jepa_studio.models import build_encoder
    from jepa_studio.weights import WeightError, load_run_weights

    d = _run_dir(ctx, run_id)
    cfg = load_config(d / "config.json")  # wsfs-ok: a run in <root>/runs (backend-owned), not the workspace
    if cfg["data"]["kind"] not in ("images", "synthetic-shapes"):
        raise ToolError(f"run {run_id} was trained on {cfg['data']['kind']}, not images")
    src = ctx.path(folder)
    paths = safe_listing(src, IMAGE_EXT, max_images)  # wsfs-ok: listing only; files are read by _load_ws_image
    if not paths:
        raise ToolError(f"no images (png/jpg/webp/bmp/gif) in {folder!r}")
    model = build_encoder(cfg)
    try:
        verified = load_run_weights(model, d)  # wsfs-ok: a run in <root>/runs (backend-owned)
    except WeightError as e:
        raise ToolError(f"weights of {run_id} failed their hash check: {e}") from e
    size = cfg["data"]["image_size"]
    xs, names = [], []
    for i, p in enumerate(paths):
        if ctx.cancelled():
            raise ToolError("cancelled")
        try:
            a = _load_ws_image(ctx.workspace, p, max(2 * size, 64))
        except Exception as e:  # noqa: BLE001
            ctx.log(f"skipped {p.name}: {e}")
            continue
        x = torch.from_numpy(np.ascontiguousarray(a)).permute(2, 0, 1)[None].float()
        xs.append(F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False, antialias=True)[0])
        names.append(p.relative_to(ctx.workspace.resolve()).as_posix())
        ctx.progress((i + 1) / len(paths), p.name)
    e, _ = embed(model.eval(), torch.stack(xs))
    e = F.normalize(e, dim=1)
    out = ctx.path("embeddings.npy")
    with wsfs.atomic_writer(ctx.workspace, out) as fh:
        np.save(fh, e.numpy())
    sim = e @ e.T
    sim.fill_diagonal_(-2)
    nn = sim.argmax(1).tolist()
    pairs = [{"image": names[i], "nearest": names[j], "cosine": round(float(sim[i, j]), 4)} for i, j in enumerate(nn)]
    nn_path = ctx.path("neighbors.json")
    wsfs.write_text(ctx.workspace, nn_path, json.dumps({"run_id": run_id, "weights_verified": verified,
                                                        "pairs": pairs}, indent=2))
    lines = [f"{p['image']} ~ {p['nearest']} ({p['cosine']})" for p in pairs[:10]]
    return ToolResult(True, f"Embedded {len(names)} images with {run_id} ({e.shape[1]}-dim, weights "
                            f"{'hash-verified' if verified else 'UNVERIFIED (no recorded hash)'}). Nearest pairs:\n" +
                      "\n".join(lines), artifacts=[ctx.artifact(out, "data"), ctx.artifact(nn_path, "data")])


@tool(name="jepa_train",
      description="Start a LeJEPA pretraining run in JEPA Studio on a folder of workspace images (or the built-in "
                  "synthetic shapes). Runs in the background; ALWAYS asks the user first.",
      parameters={"type": "object", "properties": {
          "name": {"type": "string", "pattern": r"^[A-Za-z0-9][A-Za-z0-9._-]{0,60}$"},
          "data_folder": {"type": "string", "maxLength": 500},
          "epochs": {"type": "integer", "minimum": 1, "maximum": 500},
          "arch": {"enum": ["convnet-tiny", "convnet-small", "vit-tiny"]}}, "required": ["name"]},
      risk="approve", category="models")
def jepa_train(ctx: ToolContext, name: str, data_folder: str | None = None, epochs: int = 10,
               arch: str = "convnet-tiny") -> ToolResult:
    from jepa_studio.config import load_config
    from jepa_studio.train import Trainer

    ov: dict = {"name": name, "train": {"epochs": epochs}, "model": {"arch": arch}}
    if data_folder:
        ov["data"] = {"kind": "images", "path": str(ctx.path(data_folder))}
    try:
        cfg = load_config(overrides=ov)  # wsfs-ok: no file read (overrides only)
    except ValueError as e:
        raise ToolError(str(e)[:500]) from e
    root = Path(ctx.settings.get("root") or ".").resolve()
    rid = f"{name}-{secrets.token_hex(3)}"
    run_dir = root / "runs" / rid
    # run_dir is <root>/runs; workspace images go through jepa_studio.data.loaders.open_regular
    # (no final link followed, FIFOs/devices refused without blocking) after safe_listing
    tr = Trainer(cfg, run_dir=run_dir, log=lambda *_: None)  # wsfs-ok: see the comment above
    studio = ctx.settings.get("_studio")
    rec = {"run_id": rid, "dir": run_dir, "status": "running", "trainer": tr, "kind": "pretrain", "cfg": cfg}

    def work():
        try:
            rec["status"] = tr.fit(resume=False)["status"]
        except Exception as e:  # noqa: BLE001
            rec["status"] = "failed"
            tr.emit("error", message=str(e)[:500])
    threading.Thread(target=work, daemon=True, name=f"jev-train-{rid}").start()
    if studio is not None:
        with studio.lock:
            studio.runs[rid] = rec
    return ToolResult(True, f"Started training run {rid} ({arch}, {epochs} epochs, data: "
                            f"{data_folder or 'synthetic shapes'}). Follow it in the Train tab.",
                      data={"run_id": rid})

