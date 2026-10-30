"""Accuracy checks for jev.science.sims: energy drift bounds, the CFL check, chaos in the double
pendulum and the Lorenz system, Kepler orbits against Kepler's equation, relativistic precession
against the orbit equation, and a quick end-to-end run that writes a video, a chart and metrics."""
from __future__ import annotations

import json
import math

import numpy as np
import pytest

from jev.science import sims


def test_nbody_figure_eight_conserves_and_returns():
    res = sims.simulate_nbody({"preset": "figure8", "t_end": 6.32591398})
    m = res["metrics"]
    assert m["energy_rel_drift_max"] < 1e-5            # symplectic: bounded, no secular drift
    assert m["angular_momentum_abs_drift"] < 1e-12 and m["momentum_abs_drift"] < 1e-12
    assert m["figure8_return_error"] < 1e-3             # periodic orbit closes after one period


def test_nbody_energy_error_shrinks_as_dt_squared():
    e1 = sims.simulate_nbody({"preset": "figure8", "dt": 4e-3, "t_end": 3.0})["metrics"]["energy_rel_drift_max"]
    e2 = sims.simulate_nbody({"preset": "figure8", "dt": 2e-3, "t_end": 3.0})["metrics"]["energy_rel_drift_max"]
    assert 3.0 < e1 / e2 < 5.0                          # second-order method


def test_double_pendulum_energy_and_chaos():
    res = sims.simulate_pendulum({"t_end": 12.0, "dt": 1e-3, "delta": 1e-9})
    m = res["metrics"]
    assert m["energy_rel_drift_max"] < 1e-6             # RK4 with a correct equation of motion
    assert m["separation_growth"] > 1e5                 # nearby starts diverge exponentially
    assert 0.3 < m["lyapunov_estimate_per_s"] < 3.0
    # a small-angle pendulum is not chaotic
    calm = sims.simulate_pendulum({"t_end": 12.0, "theta1_deg": 5, "theta2_deg": 3, "delta": 1e-9})["metrics"]
    assert calm["separation_growth"] < 1e3


def test_wave_cfl_is_enforced_and_energy_conserved():
    with pytest.raises(ValueError, match="CFL"):
        sims.wave_setup({"n": 64, "courant": 1.2})
    dx = 1 / 63
    with pytest.raises(ValueError):
        sims.wave_setup({"n": 64, "dt": 1.01 * dx / math.sqrt(2)})
    assert sims.courant(1.0, dx / math.sqrt(2), dx, dx) == pytest.approx(1.0)
    res = sims.simulate_wave({"preset": "pulse", "n": 96, "t_end": 0.6}, frames=10)
    assert res["metrics"]["cfl_number"] <= 1.0
    assert res["metrics"]["energy_rel_variation_max"] < 1e-10    # leapfrog's discrete energy


def test_wave_front_travels_at_c():
    n, c, t_end = 161, 1.0, 0.3
    res = sims.simulate_wave({"preset": "pulse", "n": n, "c": c, "t_end": t_end, "width": 0.02}, frames=200)
    u, t = res["snaps"][-1], res["t"][-1]
    row = np.abs(u[n // 2, n // 2:])
    r_peak = row.argmax() / (n - 1)
    assert r_peak == pytest.approx(c * t, abs=0.02)


def test_kepler_orbit_matches_kepler_equation():
    res = sims.simulate_orbit({"a": 1.0, "e": 0.6, "orbits": 2.5, "steps_per_orbit": 4000})
    m = res["metrics"]
    assert m["position_error_max"] < 2e-3
    assert m["period_rel_error"] < 1e-4
    assert m["energy_rel_drift_max"] < 1e-4
    rk = sims.simulate_orbit({"a": 1.0, "e": 0.6, "orbits": 2, "steps_per_orbit": 4000, "integrator": "rk4"})
    assert rk["metrics"]["position_error_max"] < 1e-4
    # analytic solution itself: periapsis at t = 0 and apoapsis at T/2
    T = 2 * math.pi
    assert np.allclose(sims.kepler_position(0.0, 1.0, 0.6), [0.4, 0.0])
    assert np.allclose(sims.kepler_position(T / 2, 1.0, 0.6), [-1.6, 0.0], atol=1e-9)


def test_relativistic_precession():
    m = sims.simulate_orbit({"gr": True, "p_over_M": 100.0, "e": 0.6, "orbits": 4})["metrics"]
    assert m["precession_per_orbit_rad"] == pytest.approx(m["precession_orbit_equation_rad"], rel=2e-3)
    assert m["precession_orbit_equation_rad"] == pytest.approx(6 * math.pi / 100, rel=0.1)   # 1st order
    assert m["energy_rel_drift_max"] < 1e-3
    # weak field: the orbit equation tends to 6 pi M / p
    assert sims.binet_precession(1e4, 0.3) == pytest.approx(6 * math.pi / 1e4, rel=2e-3)


def test_lorenz_lyapunov():
    m = sims.simulate_lorenz({"t_end": 25.0})["metrics"]
    assert 0.7 < m["lyapunov_estimate"] < 1.1           # reference 0.906


def test_run_writes_video_chart_and_metrics(tmp_path):
    pytest.importorskip("imageio_ffmpeg")
    paths = sims.run("pendulum", {"t_end": 2.0, "seconds": 1.0, "fps": 8, "width": 160, "height": 120}, tmp_path)
    assert paths[0].suffix == ".mp4" and paths[0].stat().st_size > 1000
    assert paths[-1].name.endswith("_metrics.json")
    m = json.loads(paths[-1].read_text())
    assert m["kind"] == "double_pendulum" and m["frames"] == 8
    if m["summary_chart"]:
        assert paths[1].suffix == ".png"
    with pytest.raises(ValueError):
        sims.run("tachyons", {}, tmp_path)
    assert sims.run("orbit", {"seconds": 1.0, "fps": 5, "width": 96, "height": 96}, tmp_path / "c",
                    cancelled=lambda: True) == []
    assert 0 < sims.estimate_seconds("wave", {"n": 200})["seconds"] < 600
