"""Science tools: an Interstellar-style Kerr black-hole renderer, filmed physics simulations and the
physics notes behind them. The heavy lifting is in jev.science (see docs/jev/science.md).

Risk: rendering and simulating write only into the conversation's workspace ("sandbox"). A render or
simulation estimated to take more than 10 minutes (a quick timing probe), or a very large frames x
pixels product, asks first through `needs_approval`.
"""
from __future__ import annotations

import json
from pathlib import Path

from .base import ToolContext, ToolError, ToolResult, tool

MAX_AUTO_SECONDS = 600.0                 # longer jobs ask first
MAX_AUTO_PIXEL_FRAMES = 1920 * 1080 * 240

_DISK_SCHEMA = {
    "type": "object",
    "description": "Thin accretion disk. Omit for the defaults (inner edge at the ISCO, outer 16 M).",
    "properties": {
        "enabled": {"type": "boolean"},
        "inner": {"type": "number", "minimum": 1, "maximum": 500, "description": "inner radius in M (default ISCO)"},
        "outer": {"type": "number", "minimum": 2, "maximum": 1000, "description": "outer radius in M"},
        "temperature_scale": {"type": "number", "minimum": 800, "maximum": 60000,
                              "description": "peak temperature in kelvin (sets the colour)"},
        "doppler": {"type": "boolean", "description": "colour shift from the gas motion"},
        "redshift": {"type": "boolean", "description": "gravitational redshift"},
        "beaming": {"type": "boolean", "description": "brightness change from the gas motion"},
        "profile": {"type": "string", "enum": ["novikov_thorne", "power_law"]},
        "opacity": {"type": "number", "minimum": 0.05, "maximum": 1},
        "texture": {"type": "boolean"},
    },
}

BLACK_HOLE_PARAMS = {
    "type": "object",
    "properties": {
        "spin": {"type": "number", "minimum": 0, "maximum": 0.999, "description": "a/M; Interstellar used 0.6"},
        "inclination_deg": {"type": "number", "minimum": 0, "maximum": 180,
                            "description": "camera angle from the spin axis; 90 = edge-on, 85 = just above the disk"},
        "distance_M": {"type": "number", "minimum": 3, "maximum": 10000, "description": "camera radius in units of M"},
        "fov_deg": {"type": "number", "minimum": 1, "maximum": 150},
        "width": {"type": "integer", "minimum": 16, "maximum": 3840},
        "height": {"type": "integer", "minimum": 16, "maximum": 2160},
        "disk": _DISK_SCHEMA,
        "background": {"type": "string", "enum": ["stars", "grid", "none"]},
        "interstellar": {"type": "boolean", "description": "the film's look: no Doppler colour shift or beaming"},
        "frames": {"type": "integer", "minimum": 1, "maximum": 1800, "description": "1 = still PNG, more = video"},
        "seconds": {"type": "number", "minimum": 0.1, "maximum": 600},
        "animation": {"type": "string", "enum": ["orbit", "zoom", "inclination"],
                      "description": "orbit: camera circles the hole (traced once, fast); zoom / inclination: "
                                     "re-traced every frame"},
        "format": {"type": "string", "enum": ["mp4", "gif"]},
        "quality": {"type": "string", "enum": ["preview", "final"],
                    "description": "final = 2x2 supersampling and tighter integration tolerance"},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2 ** 31 - 1},
        "device": {"type": "string", "enum": ["auto", "cpu", "cuda"]},
    },
    "required": [],
}


def _black_hole_needs_approval(args: dict, ctx: ToolContext) -> str | None:
    from jev.science import blackhole
    try:
        p = blackhole.normalize(args)
    except ValueError:
        return None                       # the call itself will report the bad argument
    pf = p["width"] * p["height"] * p["frames"] * p["_spp"]
    if pf > MAX_AUTO_PIXEL_FRAMES:
        return (f"This render covers {pf / 1e6:.0f} million pixel-samples ({p['frames']} frames of "
                f"{p['width']}x{p['height']}, {p['_spp']} samples each), which can take a long time.")
    est = blackhole.estimate_seconds(args)["seconds"]
    if est > MAX_AUTO_SECONDS:
        text = f"This render is estimated to take about {est / 60:.0f} minutes on {p['device']}."
        if p["device"] == "cuda":
            from jev.policy import flag
            return flag("gpu_long", text)
        return text
    return None


def _publish(ctx: ToolContext, paths: list, folder: str) -> list[Path]:
    """Renders are made in backend-owned scratch, then copied into the workspace without following
    links (jev/wsfs.py): the renderers' own writers (imageio, matplotlib) never see a workspace path."""
    from jev import wsfs
    dest = ctx.path(folder)
    return [wsfs.publish(ctx.workspace, dest / Path(p).name, p) for p in paths]


@tool(name="render_black_hole",
      description="Render a physically based image or video of a spinning (Kerr) black hole with a glowing "
                  "accretion disk, ray-traced through curved spacetime like the black hole in the film "
                  "Interstellar. Stills are PNG; frames > 1 makes an MP4 (or GIF). Defaults: spin 0.6, "
                  "camera 85 degrees from the axis at 28 M, 640x360 preview.",
      parameters=BLACK_HOLE_PARAMS, risk="sandbox", category="science",
      needs_approval=_black_hole_needs_approval)
def render_black_hole(ctx: ToolContext, **params) -> ToolResult:
    from jev.science import blackhole
    try:
        blackhole.normalize(params)
    except ValueError as e:
        raise ToolError(str(e)) from e
    from jev import wsfs
    with wsfs.use_scratch(ctx.scratch()):
        out = wsfs.new_scratch("render-")
        paths = blackhole.render(params, out, ctx.progress, ctx.cancelled)
        if not paths:
            return ToolResult(False, "The render was cancelled.")
        meta = json.loads(wsfs.read_text(out, paths[-1]))
        paths = _publish(ctx, paths, "science")
    media, meta_path = paths[0], paths[-1]
    pp, phys = meta["params"], meta["physics"]
    kind = "image" if media.suffix == ".png" else "video"
    what = "image" if kind == "image" else f"{pp['frames']}-frame {pp['animation']} video"
    text = (f"Rendered a {pp['width']}x{pp['height']} {what} of a Kerr black hole (a = {pp['spin']:.3g} M, "
            f"camera at {pp['distance_M']:.3g} M, {pp['inclination_deg']:.0f} degrees from the spin axis) in "
            f"{meta['timings']['total_s']:.1f} s on {meta['device']}. Horizon r+ = {phys['r_horizon']:.3f} M, "
            f"ISCO = {phys['r_isco']:.3f} M (disk inner edge {pp['disk']['inner']:.3f} M). "
            + ("Interstellar mode: no Doppler colour shift or beaming." if not pp["disk"]["doppler"]
               and not pp["disk"]["beaming"] else "Doppler shift and beaming on: the approaching (left) side "
               "is brighter and bluer."))
    return ToolResult(True, text, artifacts=[ctx.artifact(media, kind=kind, title="Black hole render"),
                                             ctx.artifact(meta_path, kind="data", title="Render parameters")],
                      data={"meta": meta})


SIM_PARAMS = {
    "type": "object",
    "properties": {
        "kind": {"type": "string", "enum": ["nbody", "double_pendulum", "pendulum", "wave", "orbit", "lorenz"]},
        "preset": {"type": "string", "description": "nbody: figure8 | solar | cluster; wave: pulse | double_slit | "
                                                    "ripple"},
        "seconds": {"type": "number", "minimum": 0.5, "maximum": 120, "description": "video length"},
        "fps": {"type": "integer", "minimum": 5, "maximum": 60},
        "width": {"type": "integer", "minimum": 64, "maximum": 1920},
        "height": {"type": "integer", "minimum": 64, "maximum": 1920},
        "format": {"type": "string", "enum": ["mp4", "gif"]},
        "t_end": {"type": "number", "minimum": 0.01, "maximum": 10000, "description": "simulated time"},
        "dt": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
        "n": {"type": "integer", "minimum": 2, "maximum": 1024, "description": "bodies (cluster) or grid size (wave)"},
        "delta": {"type": "number", "minimum": 0, "maximum": 1, "description": "initial offset of the twin run"},
        "theta1_deg": {"type": "number"}, "theta2_deg": {"type": "number"},
        "courant": {"type": "number", "exclusiveMinimum": 0, "maximum": 2,
                    "description": "wave: CFL number c dt sqrt(1/dx^2+1/dy^2); must be <= 1"},
        "c": {"type": "number", "exclusiveMinimum": 0, "maximum": 10},
        "frequency": {"type": "number", "exclusiveMinimum": 0, "maximum": 100},
        "a": {"type": "number", "exclusiveMinimum": 0, "maximum": 100}, "e": {"type": "number", "minimum": 0,
                                                                              "exclusiveMaximum": 1},
        "orbits": {"type": "number", "exclusiveMinimum": 0, "maximum": 100},
        "integrator": {"type": "string", "enum": ["verlet", "rk4"]},
        "gr": {"type": "boolean", "description": "orbit: add the Schwarzschild perihelion-precession term"},
        "p_over_M": {"type": "number", "minimum": 12.5, "maximum": 1e6},
        "seed": {"type": "integer", "minimum": 0, "maximum": 2 ** 31 - 1},
    },
    "required": ["kind"],
}


def _sim_needs_approval(args: dict, ctx: ToolContext) -> str | None:
    from jev.science import sims
    params = {k: v for k, v in args.items() if k != "kind"}
    try:
        est = sims.estimate_seconds(str(args.get("kind")), params)["seconds"]
    except ValueError:
        return None
    if est > MAX_AUTO_SECONDS:
        return f"This simulation is estimated to take about {est / 60:.0f} minutes."
    return None


@tool(name="simulate_physics",
      description="Run and film a physics simulation, with accuracy numbers: gravitational N-body (figure-eight "
                  "three-body orbit, solar system, star cluster), chaotic double pendulum, 2-D waves (pulse, "
                  "double slit, ripple tank), a Kepler orbit checked against the exact solution (gr=true shows "
                  "relativistic perihelion precession) or the Lorenz attractor. Returns an MP4, a summary chart "
                  "and a metrics JSON (energy error, CFL number, Lyapunov estimate...).",
      parameters=SIM_PARAMS, risk="sandbox", category="science", needs_approval=_sim_needs_approval)
def simulate_physics(ctx: ToolContext, kind: str, **params) -> ToolResult:
    from jev.science import sims
    from jev import wsfs
    with wsfs.use_scratch(ctx.scratch()):
        out = wsfs.new_scratch("sim-")
        try:
            paths = sims.run(kind, params, out, ctx.progress, ctx.cancelled)
        except ValueError as e:
            raise ToolError(str(e)) from e
        if not paths:
            return ToolResult(False, "The simulation was cancelled.")
        metrics = json.loads(wsfs.read_text(out, paths[-1]))
        paths = _publish(ctx, paths, "science")
    keys = [k for k in ("energy_rel_drift_max", "energy_rel_variation_max", "cfl_number", "figure8_return_error",
                        "position_error_max", "period_rel_error", "lyapunov_estimate_per_s", "lyapunov_estimate",
                        "precession_per_orbit_rad", "precession_orbit_equation_rad", "separation_growth")
            if metrics.get(k) is not None]
    nums = ", ".join(f"{k} = {metrics[k]:.3g}" for k in keys)
    arts = [ctx.artifact(paths[0], kind="video", title=f"{metrics['kind']} simulation")]
    if len(paths) == 3:
        arts.append(ctx.artifact(paths[1], kind="image", title="Summary chart"))
    arts.append(ctx.artifact(paths[-1], kind="data", title="Metrics"))
    return ToolResult(True, f"Simulated {metrics['kind']} ({metrics.get('integrator') or metrics.get('scheme')}, "
                            f"{metrics.get('steps')} steps) in {metrics['timings']['total_s']:.1f} s. {nums}.",
                      artifacts=arts, data={"metrics": metrics})


@tool(name="explain_black_hole_physics",
      description="Return JEV's notes on the physics of the black-hole renderer (equations, assumptions and "
                  "references) so the answer quotes them instead of guessing. Topics: overview, geodesics, "
                  "camera, disk, redshift, interstellar, shadow, limits, references, all.",
      parameters={"type": "object", "properties": {
          "topic": {"type": "string", "enum": ["overview", "geodesics", "camera", "disk", "redshift", "interstellar",
                                               "shadow", "limits", "references", "all"]}}, "required": []},
      risk="read", category="science")
def explain_black_hole_physics(ctx: ToolContext, topic: str = "all") -> ToolResult:
    from jev.science import notes
    try:
        text = notes.explain(topic)
    except KeyError as e:
        raise ToolError(f"unknown topic {topic!r}; choose one of {', '.join(notes.TOPICS)}") from e
    return ToolResult(True, text, data={"topic": topic})
