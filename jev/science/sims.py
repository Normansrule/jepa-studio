"""Physics simulations that JEV can run and film: each produces an MP4 (or GIF), a PNG summary chart and
a JSON file of accuracy metrics (energy error and friends), so the numbers can be checked, not just
watched.

    nbody            gravitational N-body, velocity Verlet (kick-drift-kick leapfrog, symplectic);
                     presets: figure8 (Chenciner & Montgomery 2000), solar, cluster
    double_pendulum  RK4; two runs whose initial angles differ by `delta` show chaotic divergence
                     (alias: pendulum)
    wave             2-D wave equation u_tt = c^2 lap(u), second-order leapfrog finite differences;
                     the CFL number C = c dt sqrt(1/dx^2 + 1/dy^2) must be <= 1 (checked)
                     presets: pulse (reflecting walls, energy conserved), double_slit, ripple
    orbit            Kepler two-body (test particle, G M = 1) vs the analytic solution of Kepler's
                     equation; gr=True adds the Schwarzschild term (perihelion precession 6 pi M / p)
    lorenz           Lorenz (1963) system, RK4, two nearby trajectories and a Lyapunov estimate

Entry point: run(kind, params, out_dir, progress, cancelled) -> [video, chart.png, metrics.json].
"""
from __future__ import annotations

import json
import math
import time
from pathlib import Path
from typing import Callable

import numpy as np
from PIL import Image, ImageDraw

from . import video as V


class Cancelled(RuntimeError):
    pass


COMMON = {"seconds": 8.0, "fps": 30, "width": 640, "height": 480, "format": "mp4", "seed": 0}


def _noop(*_a, **_k):
    return None


def _common(params: dict) -> dict:
    p = {**COMMON, **{k: v for k, v in (params or {}).items() if v is not None}}
    p["seconds"] = float(np.clip(float(p["seconds"]), 0.5, 120))
    p["fps"] = int(np.clip(int(p["fps"]), 5, 60))
    w, h = int(p["width"]), int(p["height"])
    if not (64 <= w <= 1920 and 64 <= h <= 1920):
        raise ValueError("width and height must be between 64 and 1920")
    p["width"], p["height"] = w + w % 2, h + h % 2
    if p["format"] not in ("mp4", "gif"):
        raise ValueError("format must be 'mp4' or 'gif'")
    return p


# ----------------------------------------------------------------------------------------------
# N-body
# ----------------------------------------------------------------------------------------------


def nbody_accel(x, m, G=1.0, eps=0.0):
    d = x[None, :, :] - x[:, None, :]                       # d[i, j] = x_j - x_i
    r2 = (d * d).sum(-1) + eps * eps
    np.fill_diagonal(r2, np.inf)
    return G * (d * (m[None, :, None] / r2[..., None] ** 1.5)).sum(1)


def nbody_energy(x, v, m, G=1.0, eps=0.0):
    ke = 0.5 * (m * (v * v).sum(-1)).sum()
    d = x[None, :, :] - x[:, None, :]
    r = np.sqrt((d * d).sum(-1) + eps * eps)
    iu = np.triu_indices(len(m), 1)
    pe = -G * (m[iu[0]] * m[iu[1]] / r[iu]).sum()
    return ke + pe


def nbody_preset(name: str, n: int = 40, seed: int = 0):
    """(x, v, m, G, eps, dt, t_end, view_radius)."""
    if name == "figure8":
        x = np.array([[0.97000436, -0.24308753], [-0.97000436, 0.24308753], [0.0, 0.0]])
        v3 = np.array([-0.93240737, -0.86473146])
        v = np.array([-v3 / 2, -v3 / 2, v3])
        return x, v, np.ones(3), 1.0, 0.0, 1e-3, 6.32591398 * 2, 1.4
    if name == "solar":
        # Sun + four planets on circular orbits; units AU, years, G = 4 pi^2
        G = 4 * math.pi ** 2
        a = np.array([0.39, 0.72, 1.0, 1.52])
        mp = np.array([1.66e-7, 2.45e-6, 3.0e-6, 3.2e-7])
        rng = np.random.default_rng(seed)
        ph = rng.uniform(0, 2 * math.pi, 4)
        x = np.vstack([[0, 0], np.c_[a * np.cos(ph), a * np.sin(ph)]])
        vc = np.sqrt(G / a)
        v = np.vstack([[0, 0], np.c_[-vc * np.sin(ph), vc * np.cos(ph)]])
        m = np.r_[1.0, mp]
        v[0] = -(m[1:, None] * v[1:]).sum(0) / m[0]     # zero total momentum
        return x, v, m, G, 0.0, 2e-4, 2.0, 1.7
    if name == "cluster":
        rng = np.random.default_rng(seed)
        r = 0.3 / np.sqrt(rng.uniform(0.05, 1, n) ** (-2 / 3) - 1)    # Plummer radii (scale 0.3)
        r = np.minimum(r, 2.0)
        ang = rng.uniform(0, 2 * math.pi, n)
        x = np.c_[r * np.cos(ang), r * np.sin(ang)]
        m = np.full(n, 1.0 / n)
        x -= (m[:, None] * x).sum(0)
        v = rng.normal(size=(n, 2))
        v -= (m[:, None] * v).sum(0)
        eps = 0.05
        # virialise: 2 K = -W
        ke = 0.5 * (m * (v * v).sum(-1)).sum()
        pe = nbody_energy(x, np.zeros_like(v), m, 1.0, eps)
        v *= math.sqrt(-pe / (2 * ke))
        return x, v, m, 1.0, eps, 2e-3, 6.0, 1.6
    raise ValueError("nbody preset must be 'figure8', 'solar' or 'cluster'")


def simulate_nbody(p: dict, cancelled=None):
    x, v, m, G, eps, dt, t_end, view = nbody_preset(p.get("preset", "figure8"), int(p.get("n", 40)),
                                                    int(p.get("seed", 0)))
    dt = float(p.get("dt", dt))
    t_end = float(p.get("t_end", t_end))
    steps = int(round(t_end / dt))
    E0 = nbody_energy(x, v, m, G, eps)
    L0 = (m * (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0])).sum()
    P0 = (m[:, None] * v).sum(0)
    a = nbody_accel(x, m, G, eps)
    rec_every = max(1, steps // 2000)
    xs, ts, Es = [x.copy()], [0.0], [E0]
    for i in range(1, steps + 1):
        v += 0.5 * dt * a                 # kick
        x += dt * v                       # drift
        a = nbody_accel(x, m, G, eps)
        v += 0.5 * dt * a                 # kick
        if i % rec_every == 0 or i == steps:
            xs.append(x.copy())
            ts.append(i * dt)
            Es.append(nbody_energy(x, v, m, G, eps))
            if cancelled and cancelled():
                raise Cancelled("simulation cancelled")
    Es = np.array(Es)
    L1 = (m * (x[:, 0] * v[:, 1] - x[:, 1] * v[:, 0])).sum()
    metrics = {"integrator": "velocity Verlet (KDK leapfrog)", "dt": dt, "steps": steps, "bodies": len(m),
               "energy_initial": float(E0), "energy_rel_drift_max": float(np.max(np.abs(Es - E0)) / abs(E0)),
               "energy_rel_drift_final": float(abs(Es[-1] - E0) / abs(E0)),
               "angular_momentum_abs_drift": float(abs(L1 - L0)),
               "momentum_abs_drift": float(np.linalg.norm((m[:, None] * v).sum(0) - P0))}
    if p.get("preset", "figure8") == "figure8":
        T = 6.32591398
        k = int(round(T / dt))
        if k <= steps:
            # rerun one period exactly (cheap) to measure the return-to-start error
            x1, v1, *_ = nbody_preset("figure8")
            a1 = nbody_accel(x1, m)
            x_start = x1.copy()
            for _ in range(k):
                v1 += 0.5 * dt * a1
                x1 += dt * v1
                a1 = nbody_accel(x1, m)
                v1 += 0.5 * dt * a1
            metrics["figure8_return_error"] = float(np.abs(x1 - x_start).max())
    return {"t": np.array(ts), "x": np.array(xs), "E": Es, "m": m, "view": view, "metrics": metrics}


# ----------------------------------------------------------------------------------------------
# double pendulum
# ----------------------------------------------------------------------------------------------


def pendulum_rhs(s, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    """s = [theta1, theta2, omega1, omega2] (angles from the downward vertical); shape [..., 4]."""
    t1, t2, w1, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    d = t2 - t1
    den1 = (m1 + m2) * l1 - m2 * l1 * np.cos(d) ** 2
    a1 = (m2 * l1 * w1 * w1 * np.sin(d) * np.cos(d) + m2 * g * np.sin(t2) * np.cos(d)
          + m2 * l2 * w2 * w2 * np.sin(d) - (m1 + m2) * g * np.sin(t1)) / den1
    den2 = (l2 / l1) * den1
    a2 = (-m2 * l2 * w2 * w2 * np.sin(d) * np.cos(d)
          + (m1 + m2) * (g * np.sin(t1) * np.cos(d) - l1 * w1 * w1 * np.sin(d) - g * np.sin(t2))) / den2
    return np.stack([w1, w2, a1, a2], -1)


def pendulum_energy(s, m1=1.0, m2=1.0, l1=1.0, l2=1.0, g=9.81):
    t1, t2, w1, w2 = s[..., 0], s[..., 1], s[..., 2], s[..., 3]
    T = 0.5 * (m1 + m2) * l1 ** 2 * w1 ** 2 + 0.5 * m2 * l2 ** 2 * w2 ** 2 + m2 * l1 * l2 * w1 * w2 * np.cos(t1 - t2)
    U = -(m1 + m2) * g * l1 * np.cos(t1) - m2 * g * l2 * np.cos(t2)
    return T + U


def rk4(f, s, dt):
    k1 = f(s)
    k2 = f(s + 0.5 * dt * k1)
    k3 = f(s + 0.5 * dt * k2)
    k4 = f(s + dt * k3)
    return s + dt / 6 * (k1 + 2 * k2 + 2 * k3 + k4)


def simulate_pendulum(p: dict, cancelled=None):
    th1, th2 = math.radians(float(p.get("theta1_deg", 120))), math.radians(float(p.get("theta2_deg", -20)))
    delta = float(p.get("delta", 1e-9))
    dt = float(p.get("dt", 1e-3))
    t_end = float(p.get("t_end", 20.0))
    s = np.array([[th1, th2, 0.0, 0.0], [th1 + delta, th2, 0.0, 0.0]])
    steps = int(round(t_end / dt))
    E0 = pendulum_energy(s)
    rec = max(1, steps // 2000)
    ts, S, E = [0.0], [s.copy()], [E0]
    for i in range(1, steps + 1):
        s = rk4(pendulum_rhs, s, dt)
        if i % rec == 0 or i == steps:
            ts.append(i * dt)
            S.append(s.copy())
            E.append(pendulum_energy(s))
            if cancelled and cancelled():
                raise Cancelled("simulation cancelled")
    S, E, ts = np.array(S), np.array(E), np.array(ts)
    # separation in phase space (angles wrapped)
    dth = np.angle(np.exp(1j * (S[:, 1, :2] - S[:, 0, :2])))
    sep = np.sqrt((dth ** 2).sum(-1) + ((S[:, 1, 2:] - S[:, 0, 2:]) ** 2).sum(-1))
    lam = _lyapunov_fit(ts, sep, lo=max(delta * 10, 1e-12), hi=1e-2)
    metrics = {"integrator": "RK4", "dt": dt, "steps": steps, "delta_initial": delta,
               "energy_rel_drift_max": float(np.max(np.abs(E - E0[None, :]) / np.abs(E0[None, :]))),
               "separation_final": float(sep[-1]), "separation_growth": float(sep[-1] / max(sep[0], 1e-300)),
               "lyapunov_estimate_per_s": lam,
               "time_to_separation_0p1_s": float(ts[np.argmax(sep > 0.1)]) if (sep > 0.1).any() else None}
    return {"t": ts, "S": S, "E": E, "sep": sep, "metrics": metrics}


def _lyapunov_fit(t, sep, lo, hi):
    m = (sep > lo) & (sep < hi)
    if m.sum() < 10:
        return None
    k = np.polyfit(t[m], np.log(sep[m]), 1)[0]
    return float(k)


# ----------------------------------------------------------------------------------------------
# 2-D wave equation
# ----------------------------------------------------------------------------------------------


def courant(c, dt, dx, dy):
    return c * dt * math.sqrt(1 / dx ** 2 + 1 / dy ** 2)


def wave_setup(p: dict):
    n = int(p.get("n", 200))
    if not 32 <= n <= 1024:
        raise ValueError("wave grid n must be between 32 and 1024")
    L = 1.0
    dx = L / (n - 1)
    c = float(p.get("c", 1.0))
    C = float(p.get("courant", 0.5))
    dt = p.get("dt")
    dt = float(dt) if dt is not None else C * dx / (c * math.sqrt(2))
    Cn = courant(c, dt, dx, dx)
    if Cn > 1.0 + 1e-12:
        raise ValueError(f"unstable: CFL number c dt sqrt(1/dx^2 + 1/dy^2) = {Cn:.3f} > 1. "
                         f"Use dt <= {dx / (c * math.sqrt(2)):.3e} for this grid.")
    xs = np.linspace(0, L, n)
    X, Y = np.meshgrid(xs, xs, indexing="xy")
    preset = p.get("preset", "pulse")
    wall = np.zeros((n, n), bool)
    sponge = np.zeros((n, n))
    u0 = np.zeros((n, n))
    sources = []
    if preset == "pulse":
        w = float(p.get("width", 0.03))
        u0 = np.exp(-((X - 0.5) ** 2 + (Y - 0.5) ** 2) / (2 * w * w))
    elif preset in ("double_slit", "ripple"):
        k = int(0.08 * n)      # absorbing layer thickness
        ramp = np.clip(1 - np.minimum.reduce([X, L - X, Y, L - Y]) / (k * dx), 0, 1)
        sponge = 30.0 * ramp ** 2
        if preset == "double_slit":
            j = int(0.35 * n)
            wall[:, j:j + max(2, n // 100)] = True
            gap, sepn = max(2, int(0.03 * n)), int(0.12 * n)
            for cy in (n // 2 - sepn // 2, n // 2 + sepn // 2):
                wall[cy - gap // 2: cy + gap // 2 + 1, :] &= False
            sources.append(("line", int(0.12 * n), float(p.get("frequency", 12.0))))
        else:
            f = float(p.get("frequency", 10.0))
            sources += [("point", (n // 2, int(0.4 * n)), f), ("point", (n // 2, int(0.6 * n)), f)]
    else:
        raise ValueError("wave preset must be 'pulse', 'double_slit' or 'ripple'")
    return {"n": n, "dx": dx, "dt": dt, "c": c, "courant": Cn, "u0": u0, "wall": wall, "sponge": sponge,
            "sources": sources, "preset": preset}


def wave_energy(u_new, u_old, c, dt, dx):
    """Energy the leapfrog scheme conserves exactly (up to round-off) with fixed boundaries:
    E^{n+1/2} = 1/2 sum ((u^{n+1} - u^n) / dt)^2 + c^2/2 sum grad(u^{n+1}) . grad(u^n)   (times dx^2),
    with one-sided differences on the grid edges."""
    ut = (u_new - u_old) / dt
    gx = np.diff(u_new, axis=1) * np.diff(u_old, axis=1)
    gy = np.diff(u_new, axis=0) * np.diff(u_old, axis=0)
    return 0.5 * dx * dx * ((ut * ut).sum() + c * c * (gx.sum() + gy.sum()) / (dx * dx))


def simulate_wave(p: dict, frames: int, cancelled=None):
    w = wave_setup(p)
    n, dx, dt, c = w["n"], w["dx"], w["dt"], w["c"]
    t_end = float(p.get("t_end", 1.2 if w["preset"] == "pulse" else 1.6))
    steps = int(round(t_end / dt))
    u_prev = w["u0"].copy()
    lap = np.zeros((n, n))
    u = u_prev.copy()
    # first step from u_t = 0: u1 = u0 + (c dt)^2 / 2 lap(u0)
    _laplacian(u_prev, dx, lap)
    u = u_prev + 0.5 * (c * dt) ** 2 * lap
    _bc(u, w)
    every = max(1, steps // frames)
    snaps, ts, Es = [], [], []
    damp = w["sponge"] * dt
    for i in range(1, steps + 1):
        _laplacian(u, dx, lap)
        u_new = (2 * u - (1 - damp) * u_prev + (c * dt) ** 2 * lap) / (1 + damp)
        t = (i + 1) * dt
        for kind, where, f in w["sources"]:
            val = math.sin(2 * math.pi * f * t) * min(1.0, t * f / 2)
            if kind == "line":
                u_new[:, where] = val
            else:
                u_new[where] = val * 2.0
        _bc(u_new, w)
        if w["preset"] == "pulse":
            Es.append(wave_energy(u_new, u, c, dt, dx))
        u_prev, u = u, u_new
        if i % every == 0:
            snaps.append(u.copy())
            ts.append(t)
            if cancelled and cancelled():
                raise Cancelled("simulation cancelled")
    metrics = {"scheme": "leapfrog, 5-point Laplacian", "grid": n, "dx": dx, "dt": dt, "c": c,
               "cfl_number": w["courant"], "cfl_limit": 1.0, "steps": steps}
    if Es:
        Es = np.array(Es)
        metrics["energy_rel_variation_max"] = float((Es.max() - Es.min()) / Es.mean())
    return {"snaps": np.array(snaps), "t": np.array(ts), "E": np.array(Es), "wall": w["wall"],
            "metrics": metrics, "preset": w["preset"]}


def _laplacian(u, dx, out):
    out[:] = 0
    out[1:-1, 1:-1] = (u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4 * u[1:-1, 1:-1]) / (dx * dx)
    return out


def _bc(u, w):
    u[0, :] = u[-1, :] = 0
    u[:, 0] = u[:, -1] = 0
    u[w["wall"]] = 0


# ----------------------------------------------------------------------------------------------
# Kepler orbit (and the Schwarzschild correction)
# ----------------------------------------------------------------------------------------------


def kepler_position(t, a=1.0, e=0.5, mu=1.0):
    """Analytic position of a Kepler orbit with periapsis on +x at t = 0 (Newton on Kepler's equation)."""
    n = math.sqrt(mu / a ** 3)
    M = n * np.asarray(t, dtype=float)
    E = M + e * np.sin(M)
    for _ in range(50):
        E = E - (E - e * np.sin(E) - M) / (1 - e * np.cos(E))
    x = a * (np.cos(E) - e)
    y = a * math.sqrt(1 - e * e) * np.sin(E)
    return np.stack([x, y], -1)


def simulate_orbit(p: dict, cancelled=None):
    a, e = float(p.get("a", 1.0)), float(p.get("e", 0.6))
    if not 0 <= e < 1:
        raise ValueError("eccentricity e must be in [0, 1)")
    gr = bool(p.get("gr", False))
    mu = 1.0
    T = 2 * math.pi * math.sqrt(a ** 3 / mu)
    orbits = float(p.get("orbits", 3 if not gr else 6))
    dt = float(p.get("dt", T / float(p.get("steps_per_orbit", 2000))))
    integ = p.get("integrator", "verlet")
    # in GR mode distances are in units of M with p_semi = a (1 - e^2) given by `p_over_M`
    M_over = 0.0
    if gr:
        p_semi = float(p.get("p_over_M", 100.0))
        if p_semi < 12.5:
            raise ValueError("p_over_M must be >= 12.5 (near or inside the ISCO the orbit is not a rosette)")
        M_over = a * (1 - e * e) / p_semi             # the mass in the same length units
    x = np.array([a * (1 - e), 0.0])
    v = np.array([0.0, math.sqrt(mu * (1 + e) / (a * (1 - e)))])
    h = x[0] * v[1] - x[1] * v[0]

    def acc(x):
        r2 = x @ x
        r = math.sqrt(r2)
        f = -mu / (r2 * r)
        if gr:
            f *= 1 + 3 * M_over * h * h / (mu * r2)   # Binet: u'' + u = M/h^2 + 3 M u^2 (c = 1)
        return f * x

    steps = int(round(orbits * T / dt))
    rec = max(1, steps // 3000)
    def energy(x, v):
        r = math.sqrt(x @ x)
        # the extra Schwarzschild term is the potential -mu M h^2 / r^3 (h is conserved: central force)
        return 0.5 * v @ v - mu / r - (mu * M_over * h * h / r ** 3 if gr else 0.0)

    xs, ts, Es = [x.copy()], [0.0], []
    E0 = energy(x, v)
    Es.append(E0)
    peri_t, peri_ang = [], []
    hist = []                                   # last three (r, angle) samples
    a_now = acc(x)
    for i in range(1, steps + 1):
        if integ == "rk4":
            def f(s):
                return np.r_[s[2:], acc(s[:2])]
            s = rk4(f, np.r_[x, v], dt)
            x, v = s[:2], s[2:]
        else:
            v = v + 0.5 * dt * a_now
            x = x + dt * v
            a_now = acc(x)
            v = v + 0.5 * dt * a_now
        r = math.sqrt(x @ x)
        hist.append((r, math.atan2(x[1], x[0])))
        if len(hist) > 3:
            hist.pop(0)
        if len(hist) == 3 and hist[1][0] < hist[0][0] and hist[1][0] < hist[2][0]:
            # periapsis near the middle sample: vertex of the parabola through the three radii
            r0, r1, r2 = (hh[0] for hh in hist)
            d = 0.5 * (r0 - r2) / (r0 - 2 * r1 + r2)
            a0, a1, a2 = np.unwrap([hh[1] for hh in hist])
            ang = a1 + 0.5 * d * (a2 - a0) + 0.5 * d * d * (a2 - 2 * a1 + a0)
            peri_t.append((i - 1 + d) * dt)
            peri_ang.append(ang)
        if i % rec == 0 or i == steps:
            xs.append(x.copy())
            ts.append(i * dt)
            Es.append(energy(x, v))
            if cancelled and cancelled():
                raise Cancelled("simulation cancelled")
    xs, ts, Es = np.array(xs), np.array(ts), np.array(Es)
    metrics = {"integrator": "velocity Verlet" if integ != "rk4" else "RK4", "dt": dt, "steps": steps,
               "period_analytic": T, "energy_rel_drift_max": float(np.max(np.abs(Es - E0)) / abs(E0))}
    if not gr:
        ana = kepler_position(ts, a, e, mu)
        err = np.linalg.norm(xs - ana, axis=-1)
        metrics["position_error_max"] = float(err.max())
        metrics["position_error_per_orbit"] = float(err.max() / orbits)
        if len(peri_t) >= 2:
            metrics["period_measured"] = float(np.mean(np.diff(peri_t)))
            metrics["period_rel_error"] = float(abs(metrics["period_measured"] - T) / T)
        analytic = ana
    else:
        adv = np.diff(np.unwrap(peri_ang)) if len(peri_ang) >= 2 else np.array([])
        p_semi = a * (1 - e * e) / M_over
        metrics.update({"p_over_M": p_semi, "precession_per_orbit_rad": float(adv.mean()) if adv.size else None,
                        "precession_1st_order_rad": 6 * math.pi / p_semi,
                        "precession_orbit_equation_rad": binet_precession(p_semi, e)})
        analytic = None
    return {"t": ts, "x": xs, "E": Es, "analytic": analytic, "metrics": metrics, "a": a, "e": e, "gr": gr}


def binet_precession(p_over_M: float, e: float, dphi: float = 1e-3) -> float:
    """Reference periapsis advance per orbit from the Schwarzschild orbit equation in phi,
    u'' + u = 1/p + 3 M u^2 (units M = 1), integrated with RK4 in phi (independent of the time-domain
    run). 6 pi M / p is its first-order expansion."""
    p = float(p_over_M)
    y = np.array([(1 + e) / p, 0.0])

    def f(y):
        return np.array([y[1], -y[0] + 1 / p + 3 * y[0] ** 2])

    phi, peri = 0.0, []
    while len(peri) < 3 and phi < 40 * math.pi:
        y2 = rk4(f, y, dphi)
        if y[1] > 0 >= y2[1]:                      # du/dphi changes sign at a maximum of u
            peri.append(phi + dphi * y[1] / (y[1] - y2[1]))
        y, phi = y2, phi + dphi
    return float(np.diff(peri).mean() - 2 * math.pi)


# ----------------------------------------------------------------------------------------------
# Lorenz
# ----------------------------------------------------------------------------------------------


def lorenz_rhs(s, sigma=10.0, rho=28.0, beta=8 / 3):
    x, y, z = s[..., 0], s[..., 1], s[..., 2]
    return np.stack([sigma * (y - x), x * (rho - z) - y, x * y - beta * z], -1)


def simulate_lorenz(p: dict, cancelled=None):
    dt = float(p.get("dt", 0.005))
    t_end = float(p.get("t_end", 30.0))
    delta = float(p.get("delta", 1e-8))
    s = np.array([[1.0, 1.0, 20.0], [1.0 + delta, 1.0, 20.0]])
    steps = int(round(t_end / dt))
    S, ts = [s.copy()], [0.0]
    for i in range(1, steps + 1):
        s = rk4(lorenz_rhs, s, dt)
        S.append(s.copy())
        ts.append(i * dt)
        if cancelled and i % 1000 == 0 and cancelled():
            raise Cancelled("simulation cancelled")
    S, ts = np.array(S), np.array(ts)
    sep = np.linalg.norm(S[:, 1] - S[:, 0], axis=-1)
    metrics = {"integrator": "RK4", "dt": dt, "steps": steps, "delta_initial": delta,
               "lyapunov_estimate": _lyapunov_fit(ts, sep, delta * 10, 1.0),
               "lyapunov_reference": 0.9056, "separation_final": float(sep[-1])}
    return {"t": ts, "S": S, "sep": sep, "metrics": metrics}


# ----------------------------------------------------------------------------------------------
# drawing
# ----------------------------------------------------------------------------------------------


def _canvas(p):
    return Image.new("RGB", (p["width"], p["height"]), V.BG)


def _to_px(xy, p, view, center=(0.0, 0.0)):
    s = min(p["width"], p["height"]) / (2 * view)
    return [(p["width"] / 2 + (x - center[0]) * s, p["height"] / 2 - (y - center[1]) * s) for x, y in xy]


def _trail(dr, pts, rgb, width=2):
    n = len(pts)
    for k in range(1, n):
        f = k / n
        col = tuple(int(V.BG[c] + (rgb[c] - V.BG[c]) * f) for c in range(3))
        dr.line([pts[k - 1], pts[k]], fill=col, width=width)


def _font(size):
    from PIL import ImageFont
    try:
        return ImageFont.load_default(size=size)     # FreeType default font (Pillow >= 10.1)
    except TypeError:                                # pragma: no cover - older Pillow
        return ImageFont.load_default()


def _label(dr, p, text):
    size = max(11, p["height"] // 30)
    dr.text((12, p["height"] - size - 10), text, fill=V.INK3, font=_font(size))


def frames_nbody(res, p, nframes):
    X, t = res["x"], res["t"]
    idx = np.linspace(0, len(t) - 1, nframes).astype(int)
    trail = max(2, len(t) // 12)
    m = res["m"]
    rad = 2 + 6 * (m / m.max()) ** (1 / 3)
    for f, i in enumerate(idx):
        im = _canvas(p)
        dr = ImageDraw.Draw(im)
        for b in range(X.shape[1]):
            col = V.CAT[b % len(V.CAT)]
            pts = _to_px(X[max(0, i - trail): i + 1, b], p, res["view"])
            if len(pts) > 1:
                _trail(dr, pts, col, 2 if X.shape[1] < 10 else 1)
            (cx, cy), = _to_px([X[i, b]], p, res["view"])
            r = rad[b]
            dr.ellipse([cx - r, cy - r, cx + r, cy + r], fill=col)
        _label(dr, p, f"t = {t[i]:.2f}   dE/E = {abs(res['E'][i] - res['E'][0]) / abs(res['E'][0]):.1e}")
        yield np.asarray(im)


def frames_pendulum(res, p, nframes):
    S, t = res["S"], res["t"]
    idx = np.linspace(0, len(t) - 1, nframes).astype(int)
    view = 2.3
    trail = max(2, len(t) // 25)
    for i in idx:
        im = _canvas(p)
        dr = ImageDraw.Draw(im)
        for k, col in ((0, V.CAT[0]), (1, V.CAT[1])):
            t1, t2 = S[:, k, 0], S[:, k, 1]
            x1, y1 = np.sin(t1), -np.cos(t1)
            x2, y2 = x1 + np.sin(t2), y1 - np.cos(t2)
            lo = max(0, i - trail)
            _trail(dr, _to_px(np.c_[x2[lo:i + 1], y2[lo:i + 1]], p, view), col, 2)
            pts = _to_px([(0, 0), (x1[i], y1[i]), (x2[i], y2[i])], p, view)
            dr.line(pts, fill=col, width=3)
            for (cx, cy) in pts[1:]:
                dr.ellipse([cx - 7, cy - 7, cx + 7, cy + 7], fill=col)
        (ox, oy), = _to_px([(0, 0)], p, view)
        dr.ellipse([ox - 3, oy - 3, ox + 3, oy + 3], fill=V.INK)
        _label(dr, p, f"t = {t[i]:.2f} s   separation = {res['sep'][i]:.1e}  (start {res['metrics']['delta_initial']:.0e})")
        yield np.asarray(im)


# ColorBrewer RdBu (11 classes, reversed: negative blue, positive red), interpolated to 256 entries
_RDBU = [(5, 48, 97), (33, 102, 172), (67, 147, 195), (146, 197, 222), (209, 229, 240), (247, 247, 247),
         (253, 219, 199), (244, 165, 130), (214, 96, 77), (178, 24, 43), (103, 0, 31)]


def _diverging_lut():
    a = np.array(_RDBU, float)
    x = np.linspace(0, 1, len(a))
    t = np.linspace(0, 1, 256)
    return np.stack([np.interp(t, x, a[:, k]) for k in range(3)], -1).astype(np.uint8)


def frames_wave(res, p, nframes):
    snaps = res["snaps"]
    lut = _diverging_lut()
    idx = np.linspace(0, len(snaps) - 1, nframes).astype(int)
    vmax = max(1e-9, np.percentile(np.abs(snaps[len(snaps) // 3:]), 99.5))
    for i in idx:
        u = snaps[i]
        k = np.clip((u / vmax * 0.5 + 0.5) * 255, 0, 255).astype(np.uint8)
        rgb = lut[k[::-1]]
        rgb[res["wall"][::-1]] = (90, 98, 115)
        im = Image.fromarray(rgb).resize((p["height"], p["height"]), Image.BILINEAR)
        canvas = _canvas(p)
        canvas.paste(im, ((p["width"] - p["height"]) // 2, 0))
        dr = ImageDraw.Draw(canvas)
        _label(dr, p, f"t = {res['t'][i]:.3f}   CFL = {res['metrics']['cfl_number']:.2f}")
        yield np.asarray(canvas)


def frames_orbit(res, p, nframes):
    X, t = res["x"], res["t"]
    idx = np.linspace(0, len(t) - 1, nframes).astype(int)
    a, e = res["a"], res["e"]
    view = a * (1 + e) * 1.15
    center = (-a * e, 0.0)
    E = np.linspace(0, 2 * np.pi, 200)
    ell = np.c_[a * (np.cos(E) - e), a * math.sqrt(1 - e * e) * np.sin(E)]
    for i in idx:
        im = _canvas(p)
        dr = ImageDraw.Draw(im)
        pts = _to_px(ell, p, view, center)
        for k in range(0, len(pts) - 1, 2):
            dr.line([pts[k], pts[k + 1]], fill=V.INK3, width=1)          # dashed analytic ellipse
        _trail(dr, _to_px(X[: i + 1], p, view, center), V.CAT[0], 2)
        (sx, sy), = _to_px([(0, 0)], p, view, center)
        dr.ellipse([sx - 6, sy - 6, sx + 6, sy + 6], fill=(250, 178, 25))
        if res["analytic"] is not None:
            (gx, gy), = _to_px([res["analytic"][i]], p, view, center)
            dr.ellipse([gx - 6, gy - 6, gx + 6, gy + 6], outline=V.INK, width=2)
        (cx, cy), = _to_px([X[i]], p, view, center)
        dr.ellipse([cx - 4, cy - 4, cx + 4, cy + 4], fill=V.CAT[0])
        txt = f"t = {t[i]:.2f}"
        if res["gr"]:
            txt += f"   GR precession, p = {res['metrics']['p_over_M']:.0f} M"
        else:
            txt += "   ring = analytic (Kepler's equation), dot = numerical"
        _label(dr, p, txt)
        yield np.asarray(im)


def frames_lorenz(res, p, nframes):
    S, t = res["S"], res["t"]
    idx = np.linspace(1, len(t) - 1, nframes).astype(int)
    trail = len(t) // 6
    for i in idx:
        im = _canvas(p)
        dr = ImageDraw.Draw(im)
        for k, col in ((0, V.CAT[0]), (1, V.CAT[1])):
            lo = max(0, i - trail)
            seg = S[lo:i + 1:2, k]
            pts = _to_px(np.c_[seg[:, 0], seg[:, 2] - 25], p, 30)
            _trail(dr, pts, col, 2)
        _label(dr, p, f"t = {t[i]:.2f}   separation = {res['sep'][i]:.1e}")
        yield np.asarray(im)


# ----------------------------------------------------------------------------------------------
# summary chart
# ----------------------------------------------------------------------------------------------


def summary_chart(kind, res, path) -> bool:
    """Two-panel PNG (energy error + a kind-specific panel). Needs matplotlib (optional, the `jev`
    extra); returns False without it."""
    try:
        import matplotlib
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError:
        return False
    with matplotlib.rc_context(V.chart_style()):
        fig = Figure(figsize=(9, 3.2), dpi=110)
        FigureCanvasAgg(fig)
        axes = fig.subplots(1, 2)
        c0, c1 = V.hexc(V.CAT[0]), V.hexc(V.CAT[1])
        ax = axes[0]
        if kind in ("nbody", "double_pendulum", "orbit") or (kind == "wave" and len(res.get("E", [])) > 0):
            E = res["E"]
            if E.ndim == 2:
                for k, c, lab in ((0, c0, "run A"), (1, c1, "run B")):
                    ax.semilogy(res["t"], np.abs(E[:, k] - E[0, k]) / abs(E[0, k]) + 1e-17, color=c, label=lab)
                ax.legend()
            else:
                tE = res["t"] if len(E) == len(res["t"]) else np.linspace(0, res["t"][-1] if len(res["t"]) else 1,
                                                                          len(E))
                ax.semilogy(tE, np.abs(E - E[0]) / abs(E[0]) + 1e-17, color=c0)
            ax.set_title("Relative energy error |E - E0| / |E0|")
            ax.set_xlabel("time")
        else:
            ax.set_axis_off()
            ax.text(0.02, 0.5, "\n".join(f"{k}: {v:.4g}" if isinstance(v, float) else f"{k}: {v}"
                                         for k, v in res["metrics"].items()), va="center", fontsize=9)
        ax = axes[1]
        if kind in ("double_pendulum", "lorenz"):
            ax.semilogy(res["t"], res["sep"], color=c0)
            lam = res["metrics"].get("lyapunov_estimate_per_s", res["metrics"].get("lyapunov_estimate"))
            ax.set_title(f"Separation of two nearby runs (fit {lam:.3g} / time)" if lam else
                         "Separation of two nearby runs")
            ax.set_xlabel("time")
        elif kind == "nbody":
            X = res["x"]
            for b in range(X.shape[1]):
                ax.plot(X[:, b, 0], X[:, b, 1], color=V.hexc(V.CAT[b % 5]), lw=1 if X.shape[1] > 5 else 2)
            ax.set_aspect("equal")
            ax.set_title("Trajectories")
        elif kind == "orbit":
            X = res["x"]
            ax.plot(X[:, 0], X[:, 1], color=c0, label="numerical")
            if res["analytic"] is not None:
                ax.plot(res["analytic"][:, 0], res["analytic"][:, 1], color=V.hexc(V.INK), lw=1, ls="--",
                        label="analytic")
                ax.legend()
            ax.set_aspect("equal")
            ax.set_title("Orbit")
        elif kind == "wave":
            u = res["snaps"][-1]
            vmax = np.abs(u).max() + 1e-12
            ax.imshow(u, cmap="RdBu_r", vmin=-vmax, vmax=vmax, origin="lower")
            ax.set_title(f"Final field (CFL {res['metrics']['cfl_number']:.2f})")
            ax.grid(False)
        fig.tight_layout()
        fig.savefig(path)
    return True


# ----------------------------------------------------------------------------------------------
# entry point
# ----------------------------------------------------------------------------------------------

KINDS = ("nbody", "double_pendulum", "wave", "orbit", "lorenz")
ALIASES = {"pendulum": "double_pendulum", "n_body": "nbody", "kepler": "orbit"}


def estimate_seconds(kind: str, params: dict | None = None) -> dict:
    """Rough CPU-time estimate (calibrated on a 2-core laptop-class CPU): simulation steps plus frame
    drawing and encoding. Advisory only: used to decide whether a run needs approval."""
    kind = ALIASES.get(kind, kind)
    if kind not in KINDS:
        raise ValueError(f"unknown simulation {kind!r}")
    p = _common(params or {})
    frames = p["seconds"] * p["fps"]
    px = p["width"] * p["height"] / 307200
    if kind == "nbody":
        preset = p.get("preset", "figure8")
        n = {"figure8": 3, "solar": 5}.get(preset, int(p.get("n", 40)))
        dt = float(p.get("dt", {"figure8": 1e-3, "solar": 2e-4}.get(preset, 2e-3)))
        t_end = float(p.get("t_end", {"figure8": 12.65, "solar": 2.0}.get(preset, 6.0)))
        sim = t_end / dt * (5e-5 + 4e-7 * n * n)
        draw = frames * (0.01 + 0.0025 * n) * px
    elif kind == "double_pendulum":
        sim = float(p.get("t_end", 20.0)) / float(p.get("dt", 1e-3)) * 4e-4
        draw = frames * 0.02 * px
    elif kind == "wave":
        n = int(p.get("n", 200))
        dx = 1.0 / (n - 1)
        dt = float(p.get("dt") or float(p.get("courant", 0.5)) * dx / (float(p.get("c", 1.0)) * math.sqrt(2)))
        sim = float(p.get("t_end", 1.4)) / dt * n * n * 4e-8
        draw = frames * 0.03 * px
    elif kind == "orbit":
        sim = float(p.get("orbits", 4)) * float(p.get("steps_per_orbit", 2000)) * 4e-5
        draw = frames * 0.02 * px
    else:
        sim = float(p.get("t_end", 30.0)) / float(p.get("dt", 0.005)) * 1e-4
        draw = frames * 0.02 * px
    return {"seconds": float(sim + draw + 1.5), "simulate_s": float(sim), "draw_s": float(draw)}


def run(kind: str, params: dict, out_dir: Path, progress: Callable[[float, str], None] | None = None,
        cancelled: Callable[[], bool] | None = None) -> list[Path]:
    """Run a simulation and film it. Returns [video, summary.png (if matplotlib), metrics.json];
    invalid parameters raise ValueError; a cancelled run returns [] and leaves no partial files."""
    try:
        return _run(kind, params, out_dir, progress, cancelled)
    except Cancelled:
        return []


def _run(kind, params, out_dir, progress, cancelled):
    progress = progress or _noop
    kind = ALIASES.get(kind, kind)
    if kind not in KINDS:
        raise ValueError(f"unknown simulation {kind!r}; choose one of {', '.join(KINDS)}")
    p = _common(params)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    nframes = max(2, int(round(p["seconds"] * p["fps"])))
    t0 = time.perf_counter()
    progress(0.02, f"simulating {kind}")
    if kind == "nbody":
        res, frames = simulate_nbody(p, cancelled), frames_nbody
    elif kind == "double_pendulum":
        res, frames = simulate_pendulum(p, cancelled), frames_pendulum
    elif kind == "wave":
        res, frames = simulate_wave(p, nframes, cancelled), frames_wave
    elif kind == "orbit":
        res, frames = simulate_orbit(p, cancelled), frames_orbit
    else:
        res, frames = simulate_lorenz(p, cancelled), frames_lorenz
    t_sim = time.perf_counter() - t0
    progress(0.5, "drawing frames")
    stem = f"{kind}" + (f"_{p['preset']}" if p.get("preset") else "")
    vid = out_dir / f"{stem}.{p['format']}"
    writer = V.open_writer(vid, p["fps"], p["format"])
    try:
        for i, fr in enumerate(frames(res, p, nframes)):
            writer.append_data(fr)
            if i % 10 == 0:
                progress(0.5 + 0.45 * i / nframes, f"frame {i}/{nframes}")
                if cancelled and cancelled():
                    raise Cancelled("simulation cancelled")
    except Cancelled:
        writer.close()
        vid.unlink(missing_ok=True)
        raise
    writer.close()
    chart = out_dir / f"{stem}_summary.png"
    has_chart = summary_chart(kind, res, chart)
    metrics = {"kind": kind, "params": dict(p), **res["metrics"], "summary_chart": has_chart,
               "timings": {"simulate_s": t_sim, "total_s": time.perf_counter() - t0}, "frames": nframes}
    mpath = out_dir / f"{stem}_metrics.json"
    mpath.write_text(json.dumps(metrics, indent=2, default=float))
    progress(1.0, "done")
    return [vid, chart, mpath] if has_chart else [vid, mpath]
