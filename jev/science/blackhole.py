"""Interstellar-style images and animations of a Kerr black hole with a thin accretion disk.

Method (after James, von Tunzelmann, Franklin & Thorne 2015, arXiv:1502.03808, and Luminet 1979):

1. For every pixel (and sub-pixel when supersampling) a ray is traced *backwards* from a camera at rest
   in the locally non-rotating (ZAMO / FIDO) frame, optionally moving along e_phi with speed beta,
   through Kerr spacetime (jev.science.kerr: Mino-time second-order geodesic equations, adaptive
   Dormand-Prince 5(4), vectorized in torch so it runs on a CUDA GPU when one is available).
2. Each crossing of the equatorial plane between r_in and r_out is a disk hit. The disk is a
   geometrically thin, optically thick Novikov-Thorne disk (Page & Thorne 1974 flux, so
   T(r) ~ F(r)^(1/4) with a zero-torque inner edge at the ISCO) or, optionally, T ~ r^(-3/4).
   Rays that cross inside the ISCO or beyond the outer edge continue, so the secondary and
   higher-order images (the disk seen "under" and "over" the shadow) appear by themselves.
3. Frequency shift g = nu_obs/nu_emit = E_cam / (u^t (1 - Omega b)) for Keplerian emitters; it is
   split into a gravitational part (ZAMO to ZAMO) and a Doppler part. The observed spectrum of a
   blackbody at T is a blackbody at gT (I_nu ~ g^3), so colour and visible brightness come from
   the CIE-integrated blackbody table at gT (jev.science.color). Toggles: redshift (gravitational),
   doppler (colour shift from the disk's motion), beaming (brightness change from the motion).
   "Interstellar mode" (doppler = beaming = False) reproduces the film's choice to omit the
   Doppler shifts (James et al. 2015).
4. Escaped rays sample a procedural sky (jev.science.background) in their asymptotic direction;
   captured rays are black. HDR radiance -> optional bloom -> ACES filmic curve -> sRGB.

Public entry point: render(params, out_dir, progress, cancelled) -> [image or video, metadata.json].
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from . import background, color, kerr


class Cancelled(RuntimeError):
    """Raised internally when the caller's cancelled() returns True; render() then removes partial
    files and returns an empty list (the caller checks its own cancel flag)."""


DEFAULTS = {
    "spin": 0.6, "inclination_deg": 85.0, "distance_M": 28.0, "fov_deg": 55.0,
    "width": 640, "height": 360, "background": "stars", "frames": 1, "seconds": 6.0, "fps": None,
    "quality": "preview", "seed": 0, "device": "auto", "camera_beta": 0.0, "animation": "orbit",
    "orbit_degrees": 90.0, "zoom_to_M": None, "inclination_to_deg": None, "time_M_per_s": 30.0,
    "exposure": None, "bloom": None, "format": "mp4", "interstellar": False,
}
DISK_DEFAULTS = {"inner": None, "outer": 16.0, "temperature_scale": 9000.0, "doppler": True,
                 "redshift": True, "beaming": True, "profile": "novikov_thorne", "opacity": 1.0,
                 "texture": True, "enabled": True}
QUALITY = {"preview": {"rtol": 1e-5, "spp": 1, "bloom": 0.06},
           "final": {"rtol": 1e-7, "spp": 4, "bloom": 0.1}}
MAX_PIXELS = 3840 * 2160
MAX_FRAMES = 1800


def normalize(params: dict | None) -> dict:
    """Fill defaults, clamp to physical ranges and validate. Raises ValueError with a readable message."""
    params = params or {}
    unknown = sorted(set(params) - set(DEFAULTS) - {"disk"})
    if unknown:
        params = {k: v for k, v in params.items() if k not in unknown}     # e.g. API envelope fields
    p = {**DEFAULTS, **{k: v for k, v in params.items() if k != "disk" and v is not None}}
    disk_in = params.get("disk") or {}
    if not isinstance(disk_in, dict):
        raise ValueError("disk must be an object")
    disk = {**DISK_DEFAULTS, **{k: v for k, v in disk_in.items() if v is not None and
                                (k in DISK_DEFAULTS or k == "interstellar")}}
    if p.pop("interstellar", False) or disk.pop("interstellar", False):
        disk["doppler"] = disk["beaming"] = False
    disk.pop("interstellar", None)
    a = float(p["spin"])
    if not 0 <= a <= 0.999:
        raise ValueError("spin must be between 0 and 0.999")
    p["spin"] = a
    p["inclination_deg"] = float(np.clip(float(p["inclination_deg"]), 0.5, 179.5))
    rh = kerr.horizon(a)
    p["distance_M"] = float(p["distance_M"])
    if p["distance_M"] < rh + 1.0 or p["distance_M"] > 1e4:
        raise ValueError(f"distance_M must be between {rh + 1:.2f} (just outside the horizon) and 10000")
    p["fov_deg"] = float(np.clip(float(p["fov_deg"]), 1.0, 150.0))
    w, h = int(p["width"]), int(p["height"])
    if w < 16 or h < 16 or w * h > MAX_PIXELS:
        raise ValueError(f"width x height must be at least 16 x 16 and at most {MAX_PIXELS} pixels")
    p["width"], p["height"] = w + (w % 2), h + (h % 2)   # even sizes for H.264
    p["frames"] = int(p["frames"])
    if not 1 <= p["frames"] <= MAX_FRAMES:
        raise ValueError(f"frames must be between 1 and {MAX_FRAMES}")
    p["seconds"] = float(np.clip(float(p["seconds"]), 0.1, 600.0))
    if p["quality"] not in QUALITY:
        raise ValueError("quality must be 'preview' or 'final'")
    if p["background"] not in ("stars", "grid", "none"):
        raise ValueError("background must be 'stars', 'grid' or 'none'")
    if p["animation"] not in ("orbit", "zoom", "inclination"):
        raise ValueError("animation must be 'orbit', 'zoom' or 'inclination'")
    if p["format"] not in ("mp4", "gif"):
        raise ValueError("format must be 'mp4' or 'gif'")
    if p["device"] not in ("auto", "cpu", "cuda"):
        raise ValueError("device must be 'auto', 'cpu' or 'cuda'")
    p["camera_beta"] = float(np.clip(float(p["camera_beta"]), 0.0, 0.99))
    r_isco = kerr.isco(a)
    r_ph = kerr.photon_orbit(a)
    inner = disk["inner"]
    disk["inner"] = r_isco if inner is None else float(inner)
    if disk["inner"] < r_ph * 1.02:
        raise ValueError(f"disk.inner must be above the prograde photon orbit ({r_ph:.3f} M); "
                         f"the ISCO is {r_isco:.3f} M")
    disk["outer"] = float(disk["outer"])
    if disk["outer"] <= disk["inner"]:
        raise ValueError("disk.outer must be larger than disk.inner")
    if disk["profile"] not in ("novikov_thorne", "power_law"):
        raise ValueError("disk.profile must be 'novikov_thorne' or 'power_law'")
    if disk["profile"] == "novikov_thorne" and disk["inner"] < r_isco - 1e-9:
        disk["profile"] = "power_law"   # Novikov-Thorne has zero flux inside the ISCO
    disk["temperature_scale"] = float(np.clip(float(disk["temperature_scale"]), 800.0, 60000.0))
    disk["opacity"] = float(np.clip(float(disk["opacity"]), 0.05, 1.0))
    for k in ("doppler", "redshift", "beaming", "texture", "enabled"):
        disk[k] = bool(disk[k])
    p["disk"] = disk
    q = QUALITY[p["quality"]]
    p["_rtol"], p["_spp"] = q["rtol"], q["spp"]
    p["bloom"] = q["bloom"] if p["bloom"] is None else float(p["bloom"])
    if p["device"] == "auto":
        p["device"] = "cuda" if torch.cuda.is_available() else "cpu"
    elif p["device"] == "cuda" and not torch.cuda.is_available():
        p["device"] = "cpu"
    return p


# ----------------------------------------------------------------------------------------------
# camera rays for an image
# ----------------------------------------------------------------------------------------------


def screen_coords(w: int, h: int, fov_deg: float, spp: int = 1, seed: int = 0):
    """Tangent-plane coordinates (x right, y up) of every (sub)sample, shape [spp, h, w].
    The horizontal field of view is fov_deg. Supersamples are stratified and jittered."""
    tx = math.tan(math.radians(fov_deg) / 2)
    k = int(round(math.sqrt(spp)))
    rng = np.random.default_rng(seed)
    xs, ys = [], []
    for i in range(k):
        for j in range(k):
            jx = (i + rng.random((h, w))) / k if k > 1 else np.full((h, w), 0.5)
            jy = (j + rng.random((h, w))) / k if k > 1 else np.full((h, w), 0.5)
            px = np.arange(w)[None, :] + jx
            py = np.arange(h)[:, None] + jy
            xs.append((px / w * 2 - 1) * tx)
            ys.append((1 - py / h * 2) * tx * h / w)
    return np.stack(xs), np.stack(ys)


def trace_view(p: dict, r_cam: float, incl_deg: float, progress=None, cancelled=None):
    a = p["spin"]
    cam = kerr.Camera(r=r_cam, theta=math.radians(incl_deg), a=a, beta=p["camera_beta"])
    sx, sy = screen_coords(p["width"], p["height"], p["fov_deg"], p["_spp"], p["seed"])
    rays = kerr.camera_rays(cam, sx, sy)
    d = p["disk"]
    r_in, r_out = (d["inner"], d["outer"]) if d["enabled"] else (0.0, -1.0)
    max_hits = 1 if d["opacity"] >= 1 else 3
    res = kerr.trace(cam, rays, r_in, r_out, max_hits=max_hits, rtol=p["_rtol"], device=p["device"],
                     progress=progress, cancelled=cancelled)
    return cam, res, sx.shape


# ----------------------------------------------------------------------------------------------
# shading
# ----------------------------------------------------------------------------------------------


def _vnoise1(x, seed):
    i = np.floor(x)
    f = x - i
    f = f * f * (3 - 2 * f)
    a = background._hash(i.astype(np.int64), seed=seed)
    b = background._hash((i + 1).astype(np.int64), seed=seed)
    return a + (b - a) * f


def disk_texture(r, psi, seed: int = 0):
    """Procedural brightness pattern in the co-rotating frame: fine rings (value noise in ln r)
    modulated by low azimuthal modes, which differential rotation shears into spirals."""
    u = np.log(np.maximum(r, 1e-6))
    ring = 0.0
    amp, freq = 0.5, 9.0
    for o in range(4):
        ring = ring + amp * _vnoise1(u * freq + 17.3 * o, seed + o)
        amp *= 0.55
        freq *= 2.3
    ring = ring / 0.9
    az = 0.0
    rng = np.random.default_rng(seed + 99)
    for m in (2, 3, 5, 8, 13):
        az = az + rng.uniform(0.3, 1.0) / math.sqrt(m) * np.sin(m * psi + rng.uniform(0, 2 * math.pi)
                                                                  + rng.uniform(-3, 3) * u)
    return np.clip(0.5 + 0.5 * ring * (1 + 0.3 * az), 0.05, None)


def disk_temperature(r, p: dict):
    d = p["disk"]
    if d["profile"] == "novikov_thorne":
        a = p["spin"]
        rr = np.linspace(d["inner"], max(d["outer"], d["inner"] + 1), 2048)
        Fmax = float(kerr.page_thorne_flux(rr, a).max())
        return d["temperature_scale"] * (np.maximum(kerr.page_thorne_flux(r, a), 0) / Fmax) ** 0.25
    return d["temperature_scale"] * (np.maximum(r, d["inner"]) / d["inner"]) ** -0.75


def shade(res: kerr.TraceResult, shape, p: dict, cam, phi_cam: float = 0.0, t_cam: float = 0.0):
    """Linear HDR image [h, w, 3] from a trace. phi_cam rotates the camera in azimuth (the trace is
    axisymmetric, so it does not need recomputing) and t_cam is the camera's coordinate time."""
    spp, h, w = shape
    n = spp * h * w
    d = p["disk"]
    a = p["spin"]
    rgb = np.zeros((n, 3))
    trans = np.ones(n)
    for k in range(res.hit_r.shape[0]):
        r = res.hit_r[k]
        m = np.isfinite(r)
        if not m.any():
            continue
        rk = r[m]
        b, E_cam = res.b[m], res.E_cam[m]
        g, g_grav = kerr.redshift(b, rk, a, E_cam, cam.r, cam.theta)
        g_dop = g / g_grav
        T = disk_temperature(rk, p)
        g_col = (g_grav if d["redshift"] else 1.0) * (g_dop if d["doppler"] else 1.0)
        g_lum = (g_grav if d["redshift"] else 1.0) * (g_dop if d["beaming"] else 1.0)
        col = color.blackbody_rgb(T * g_col, luminance=False).clip(0)
        lum = color.bb_luminance(T * g_lum)
        if d["texture"]:
            t_emit = t_cam - res.hit_t[k][m]
            psi = res.hit_phi[k][m] + phi_cam - kerr.kepler_omega(rk, a) * t_emit
            lum = lum * disk_texture(rk, psi, p["seed"])
        # soft fade at the outer edge (and the inner edge for the power-law profile)
        edge = np.clip((d["outer"] - rk) / (0.15 * d["outer"]), 0, 1)
        if d["profile"] == "power_law":
            edge = edge * np.clip((rk - d["inner"]) / (0.1 * d["inner"]), 0, 1)
        alpha = d["opacity"] * edge
        rgb[m] += (trans[m] * alpha * lum)[:, None] * col
        trans[m] *= 1 - alpha
    esc = res.status == kerr.ESCAPED
    if p["background"] != "none" and esc.any():
        pix = math.radians(p["fov_deg"]) / p["width"]
        jac = None
        if p["background"] == "stars":
            # lens-map Jacobian from neighbouring rays, so stars render as magnified points
            D = np.full((n, 3), np.nan)
            D[esc] = background.direction(res.theta_inf[esc], res.phi_inf[esc] + phi_cam)
            jac = background.lens_jacobian(D.reshape(spp, h, w, 3), pix).reshape(n, 2, 2)[esc]
        bg = background.sample(p["background"], res.theta_inf[esc], res.phi_inf[esc] + phi_cam, pix, p["seed"],
                               jac=jac)
        rgb[esc] += trans[esc, None] * bg
    return rgb.reshape(spp, h, w, 3).mean(0)


def tonemap(hdr, p: dict, exposure: float | None = None):
    img = color.bloom(hdr, p["bloom"]) if p["bloom"] > 0 else hdr
    if exposure is None:
        exposure = auto_exposure(hdr)
    return color.to_uint8(img, exposure), exposure


def auto_exposure(hdr):
    lum = hdr @ np.array([0.2126, 0.7152, 0.0722])
    bright = lum[lum > 1e-6]
    if bright.size == 0:
        return 1.0
    return float(1.7 / max(np.percentile(bright, 99.5), 1e-9))


# ----------------------------------------------------------------------------------------------
# entry points
# ----------------------------------------------------------------------------------------------


def _noop(*_a, **_k):
    return None


_BENCH: dict = {}


def _bench_per_element(device: str) -> float:
    """Seconds per ray for one Dormand-Prince step's 7 right-hand-side evaluations, measured once per
    device on 65 536 rays (so it reflects this machine)."""
    if device not in _BENCH:
        n = 1 << 16
        dev = torch.device(device)
        Y = torch.rand(6, n, dtype=torch.float64, device=dev) + torch.tensor(
            [[30.0], [1.0], [0.0], [0.0], [1.0], [0.1]], dtype=torch.float64, device=dev)
        b = torch.rand(n, dtype=torch.float64, device=dev)
        best = float("inf")
        for _ in range(3):
            t0 = time.perf_counter()
            for _ in range(7):
                kerr._rhs(Y, b, b, 0.6)
            if device == "cuda":
                torch.cuda.synchronize()
            best = min(best, time.perf_counter() - t0)
        _BENCH[device] = best / n
    return _BENCH[device]


def estimate_seconds(params: dict) -> dict:
    """Runtime estimate: a 32 x 18 probe trace with the requested camera and tolerance gives the mean
    number of steps per ray; a micro-benchmark gives this machine's cost per ray-step (a full step is
    about 2x its 7 right-hand-side evaluations: error control, events, bookkeeping). Shading costs
    about 1.8 us per sample per frame. Measured on a 2-core CPU it is 0-35 % above the actual time
    (docs/jev/science.md), erring long on purpose since it gates approval. Returns {seconds, rays, traces, mean_steps, per_ray_step_s}."""
    p = normalize(params)
    probe = {**p, "width": 32, "height": 18, "_spp": 1}
    _, res, _ = trace_view(probe, p["distance_M"], p["inclination_deg"])
    steps = max(1.0, res.stats["mean_steps"])
    per_step = 2.0 * _bench_per_element(p["device"])
    rays = p["width"] * p["height"] * p["_spp"]
    traces = 1 if (p["frames"] == 1 or p["animation"] == "orbit") else p["frames"]
    if traces > 1 and p["animation"] == "zoom":
        steps *= 0.8                     # closer cameras need fewer steps per ray
    trace_s = rays * steps * per_step * traces
    shade_s = p["frames"] * rays * 1.8e-6
    return {"seconds": float(trace_s + shade_s + 1.0), "rays": rays, "traces": traces, "mean_steps": steps,
            "per_ray_step_s": per_step}


def render(params: dict, out_dir: Path, progress: Callable[[float, str], None] | None = None,
           cancelled: Callable[[], bool] | None = None) -> list[Path]:
    """Render a still (frames == 1, PNG) or an animation (MP4 or GIF) plus a JSON sidecar with the
    parameters, timings and physical reference numbers. See DEFAULTS / DISK_DEFAULTS for params.
    Invalid parameters raise ValueError; a cancelled render returns [] and leaves no partial files."""
    progress = progress or _noop
    cancelled = cancelled or (lambda: False)
    p = normalize(params)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)  # wsfs-ok: out_dir is backend scratch or _science
    try:
        return _render(p, out_dir, progress, cancelled)
    except Cancelled:
        return []


def _render(p, out_dir, progress, cancelled):
    a = p["spin"]
    stem = f"blackhole_a{a:.3f}_i{p['inclination_deg']:.0f}_r{p['distance_M']:.0f}".replace(".", "p")
    t_start = time.perf_counter()
    timings = {}
    meta = {"params": {k: v for k, v in p.items() if not k.startswith("_")},
            "physics": {"r_horizon": kerr.horizon(a), "r_isco": kerr.isco(a),
                        "r_photon_prograde": kerr.photon_orbit(a, True),
                        "r_photon_retrograde": kerr.photon_orbit(a, False),
                        "shadow_radius_schwarzschild_rad": math.asin(min(1.0, 3 * math.sqrt(3) * math.sqrt(
                            max(0.0, 1 - 2 / p["distance_M"])) / p["distance_M"]))}}

    def check():
        if cancelled():
            raise Cancelled("render cancelled")

    if p["frames"] == 1:
        cam, res, shape = trace_view(p, p["distance_M"], p["inclination_deg"],
                                     progress=lambda f, m: progress(0.9 * f, m), cancelled=cancelled)
        check()
        timings["trace_s"] = time.perf_counter() - t_start
        t1 = time.perf_counter()
        hdr = shade(res, shape, p, cam)
        img, expo = tonemap(hdr, p, p["exposure"])
        timings["shade_s"] = time.perf_counter() - t1
        path = out_dir / f"{stem}.png"
        from PIL import Image
        Image.fromarray(img).save(path, optimize=True)  # wsfs-ok: out_dir is backend scratch or _science
        meta["trace"] = res.stats
        meta["exposure"] = expo
        outputs = [path]
    else:
        path = out_dir / f"{stem}_{p['animation']}.{p['format']}"
        fps = p["fps"] or max(1.0, p["frames"] / p["seconds"])
        frames = []
        expo = p["exposure"]
        res = cam = shape = None
        try:
            writer = _open_writer(path, fps, p["format"])
            try:
                for f in range(p["frames"]):
                    check()
                    s = f / max(1, p["frames"] - 1)
                    t_cam = f / fps * p["time_M_per_s"]
                    phi_cam = 0.0
                    if p["animation"] == "orbit":
                        if res is None:
                            cam, res, shape = trace_view(p, p["distance_M"], p["inclination_deg"],
                                                         progress=lambda fr, m: progress(0.6 * fr, m),
                                                         cancelled=cancelled)
                            timings["trace_s"] = time.perf_counter() - t_start
                        phi_cam = math.radians(p["orbit_degrees"]) * s
                    else:
                        r_cam, inc = p["distance_M"], p["inclination_deg"]
                        if p["animation"] == "zoom":
                            r_to = p["zoom_to_M"] or max(kerr.horizon(a) + 4, p["distance_M"] * 0.5)
                            r_cam = p["distance_M"] * (r_to / p["distance_M"]) ** _ease(s)
                        else:
                            inc_to = p["inclination_to_deg"] if p["inclination_to_deg"] is not None else 60.0
                            inc = p["inclination_deg"] + (float(inc_to) - p["inclination_deg"]) * _ease(s)
                        cam, res, shape = trace_view(p, r_cam, inc, cancelled=cancelled)
                    check()
                    hdr = shade(res, shape, p, cam, phi_cam=phi_cam, t_cam=t_cam)
                    img, expo = tonemap(hdr, p, expo)
                    writer.append_data(img)
                    frames.append(f)
                    base = 0.6 if p["animation"] == "orbit" else 0.0
                    progress(base + (0.99 - base) * (f + 1) / p["frames"], f"frame {f + 1}/{p['frames']}")
            finally:
                writer.close()
        except Cancelled:
            path.unlink(missing_ok=True)  # wsfs-ok: out_dir is backend scratch or _science
            raise
        meta["trace"] = res.stats if res is not None else {}
        meta["exposure"] = expo
        meta["fps"] = fps
        outputs = [path]
    timings["total_s"] = time.perf_counter() - t_start
    meta["timings"] = timings
    meta["device"] = p["device"]
    meta_path = out_dir / f"{path.stem}.json"
    meta_path.write_text(json.dumps(meta, indent=2, default=float))  # wsfs-ok: out_dir is backend scratch or _science
    progress(1.0, "done")
    return outputs + [meta_path]


def _ease(s: float) -> float:
    return 0.5 - 0.5 * math.cos(math.pi * s)


def _open_writer(path: Path, fps: float, fmt: str):
    from .video import open_writer
    return open_writer(path, fps, fmt)
