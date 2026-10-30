"""Validation of the Kerr geodesic tracer and disk physics in jev.science.kerr against closed forms.

Units G = c = M = 1. Every check compares the numerical tracer with an independent analytic result:
shadow size and asymmetry (Bardeen 1973 critical curve), photon orbit, ISCO (Bardeen, Press &
Teukolsky 1972), weak-field deflection 4M/b + 15 pi M^2 / (4 b^2), conservation of E, L, Q along a
geodesic, an independent Schwarzschild orbit-equation integration, the Page-Thorne flux and the sign
of the disk redshift.
"""
from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from jev.science import kerr


def shadow_edge(cam, direction, lo, hi, rounds=3, n=33, rtol=1e-8):
    """Bracket the screen-plane distance (tangent of the angle) along `direction` where rays stop
    being captured; n rays per round, `rounds` refinements."""
    dx, dy = direction
    for _ in range(rounds):
        s = np.linspace(lo, hi, n)
        rays = kerr.camera_rays(cam, s * dx, s * dy)
        cap = kerr.trace(cam, rays, 0.0, -1.0, rtol=rtol, max_steps=20000).status == kerr.CAPTURED
        assert cap[0] and not cap[-1]
        k = int(np.argmax(~cap))
        lo, hi = s[k - 1], s[k]
    return 0.5 * (lo + hi)


def curve_y0_points(a, inc, r):
    """x of the two points where the finite-distance critical curve crosses y = 0 (Theta(theta_obs) = 0,
    i.e. the ends of the allowed photon-orbit range)."""
    cx, cy = kerr.critical_curve(a, inc, r_obs=r, n=400001)
    half = cx[: cx.size // 2]
    return half.min(), half.max()


# ----------------------------------------------------------------------------------------------
# closed-form radii
# ----------------------------------------------------------------------------------------------


def test_horizon_photon_orbit_isco():
    assert kerr.horizon(0.0) == pytest.approx(2.0)
    assert kerr.photon_orbit(0.0) == pytest.approx(3.0)
    assert kerr.photon_orbit(1.0, True) == pytest.approx(1.0, abs=1e-9)
    assert kerr.photon_orbit(1.0, False) == pytest.approx(4.0)
    assert kerr.isco(0.0) == pytest.approx(6.0)
    assert kerr.isco(0.998) == pytest.approx(1.2370, abs=2e-4)
    assert kerr.isco(1.0, False) == pytest.approx(9.0)
    # the ISCO is the minimum of the specific energy of circular orbits, dE/dr = 0
    for a in (0.0, 0.5, 0.9):
        r = np.linspace(kerr.isco(a) * 0.8, kerr.isco(a) * 1.5, 40001)
        _, E, _, _ = kerr.circular_orbit(r, a)
        assert r[np.argmin(E)] == pytest.approx(kerr.isco(a), rel=1e-4)


def test_photon_sphere_is_where_critical_rays_wind():
    """A ray launched just outside the Schwarzschild shadow skims r = 3M before escaping."""
    r_obs = 30.0
    cam = kerr.Camera(r=r_obs, theta=math.pi / 2, a=0.0)
    crit = math.tan(math.asin(3 * math.sqrt(3) * math.sqrt(1 - 2 / r_obs) / r_obs))
    Y, b, q = kerr.integrate_single(cam, crit * (1 + 1e-7), 0.0, sigma_max=5.0, project=True)
    assert Y[0].min() == pytest.approx(3.0, abs=2e-3)
    assert b * b + q == pytest.approx(27.0, rel=1e-5)


# ----------------------------------------------------------------------------------------------
# shadow
# ----------------------------------------------------------------------------------------------


def test_schwarzschild_shadow_radius_finite_distance():
    """sin(alpha_sh) = 3 sqrt(3) M sqrt(1 - 2M/r) / r for a static observer at r (Synge 1966)."""
    for r_obs in (10.0, 40.0):
        cam = kerr.Camera(r=r_obs, theta=math.pi / 2, a=0.0)
        ang = math.atan(shadow_edge(cam, (1, 0), 0.0, 1.0))
        exact = math.asin(3 * math.sqrt(3) * math.sqrt(1 - 2 / r_obs) / r_obs)
        assert ang == pytest.approx(exact, rel=2e-4)
        # and it is round: same radius straight up
        cam_i = kerr.Camera(r=r_obs, theta=math.radians(70), a=0.0)
        ang_up = math.atan(shadow_edge(cam_i, (0, 1), 0.0, 1.0))
        assert ang_up == pytest.approx(exact, rel=2e-4)


def test_kerr_shadow_asymmetry_matches_bardeen_curve():
    """a = 0.99 seen edge-on: the shadow is shifted toward the receding (retrograde, +x) side and
    flattened on the prograde side; both edges agree with the critical curve."""
    a, inc, r_obs = 0.99, 90.0, 1000.0
    cam = kerr.Camera(r=r_obs, theta=math.radians(inc), a=a)
    right = math.atan(shadow_edge(cam, (1, 0), 0.0, 0.02))
    left = -math.atan(shadow_edge(cam, (-1, 0), 0.0, 0.02))
    lo, hi = curve_y0_points(a, inc, r_obs)
    assert right == pytest.approx(hi, rel=2e-4)
    assert left == pytest.approx(lo, rel=2e-4)
    # far-field Bardeen numbers (alpha = -b / sin i): about -2.25 M and +6.98 M
    alpha, _ = kerr.critical_curve(a, inc, None)
    assert left * r_obs == pytest.approx(alpha.min(), rel=3e-3)
    assert right * r_obs == pytest.approx(alpha.max(), rel=3e-3)
    centre = 0.5 * (left + right) * r_obs
    assert centre > 2.0          # displaced by > 2M (a = 0 would give 0)


def test_kerr_shadow_inclined_camera_close_in():
    a, inc, r_obs = 0.9, 60.0, 50.0
    cam = kerr.Camera(r=r_obs, theta=math.radians(inc), a=a)
    right = math.atan(shadow_edge(cam, (1, 0), 0.0, 0.5))
    lo, hi = curve_y0_points(a, inc, r_obs)
    assert right == pytest.approx(hi, rel=2e-4)


# ----------------------------------------------------------------------------------------------
# geodesic accuracy
# ----------------------------------------------------------------------------------------------


def test_weak_field_deflection():
    """Total bending of an equatorial ray with impact parameter b >> M:
    delta = 4M/b + 15 pi M^2 / (4 b^2) + O(M^3/b^3)."""
    r_c = 1e6
    cam = kerr.Camera(r=r_c, theta=math.pi / 2, a=0.0)
    for b in (100.0, 400.0):
        sin_psi = b * math.sqrt(1 - 2 / r_c) / r_c          # angle from the inward radial
        psi = math.asin(sin_psi)
        rays = kerr.camera_rays(cam, np.array([math.tan(psi)]), np.array([0.0]))
        assert rays["b"][0] == pytest.approx(-b, rel=1e-9)   # right of centre = retrograde (L < 0)
        res = kerr.trace(cam, rays, 0.0, -1.0, rtol=1e-10, max_steps=20000)
        assert res.status[0] == kerr.ESCAPED
        delta = res.phi_inf[0] - (math.pi - psi)
        expected = 4 / b + 15 * math.pi / (4 * b * b)
        assert delta == pytest.approx(expected, rel=2e-3), (b, delta, expected)


def _metric(r, th, a):
    s2, c2 = math.sin(th) ** 2, math.cos(th) ** 2
    sig = r * r + a * a * c2
    dl = r * r - 2 * r + a * a
    gtt = -(1 - 2 * r / sig)
    gtp = -2 * a * r * s2 / sig
    gpp = (r * r + a * a + 2 * a * a * r * s2 / sig) * s2
    return sig, dl, gtt, gtp, gpp


def test_conservation_of_E_L_Q_along_a_geodesic():
    """Raw second-order integration (no projection): R, Theta first integrals stay satisfied, and the
    momenta rebuilt from (dt, dphi, dtheta) give back E = 1, L = b and Q = q."""
    a = 0.9
    cam = kerr.Camera(r=25.0, theta=math.radians(70), a=a)
    Y, b, q = kerr.integrate_single(cam, 0.30, 0.12, sigma_max=2.0, rtol=1e-10)
    assert Y.shape[1] > 50 and Y[0].min() < 8          # it went close to the hole
    errs_E, errs_L, errs_Q, errs_R = [], [], [], []
    for r, th, _, _, vr, vth in Y.T:
        sig, dl, gtt, gtp, gpp = _metric(r, th, a)
        s2 = math.sin(th) ** 2
        P = r * r + a * a - a * b
        # forward photon: dt/dlambda, dphi/dlambda from the Carter equations (backward ray -> signs)
        tdot = ((r * r + a * a) * P / dl + a * (b - a * s2)) / sig
        pdot = (a * P / dl + b / s2 - a) / sig
        errs_E.append(abs(-(gtt * tdot + gtp * pdot) - 1.0))
        errs_L.append(abs((gtp * tdot + gpp * pdot) - b))
        Q = vth * vth + math.cos(th) ** 2 * (b * b / s2 - a * a)
        errs_Q.append(abs(Q - q) / (abs(q) + b * b))
        R = float(kerr.radial_potential(r, b, q, a))
        errs_R.append(abs(vr * vr - R) / (r ** 4))
    assert max(errs_E) < 1e-12 and max(errs_L) < 1e-10      # identities of the Carter equations
    assert max(errs_Q) < 1e-7
    assert max(errs_R) < 1e-7


def test_schwarzschild_matches_independent_orbit_equation():
    """a = 0 equatorial rays: compare the asymptotic direction with scipy's integration of
    d^2u/dphi^2 + u = 3 M u^2 (u = 1/r)."""
    scipy_int = pytest.importorskip("scipy.integrate")
    r_c = 20.0
    cam = kerr.Camera(r=r_c, theta=math.pi / 2, a=0.0)
    nx = np.array([0.35, 0.5, 0.8, 1.5, 3.0])               # all outside the shadow (tan 0.26 rad = 0.27)
    rays = kerr.camera_rays(cam, nx, np.zeros_like(nx))
    res = kerr.trace(cam, rays, 0.0, -1.0, rtol=1e-10, max_steps=20000)
    for i, x in enumerate(nx):
        b = abs(rays["b"][i])
        u0 = 1 / r_c
        du0 = math.sqrt(max(0.0, 1 / b ** 2 - u0 * u0 * (1 - 2 * u0)))   # moving inward: u grows

        def f(phi, y):
            return [y[1], -y[0] + 3 * y[0] ** 2]

        def esc(phi, y):
            return y[0]
        esc.terminal, esc.direction = True, -1
        sol = scipy_int.solve_ivp(f, (0, 20), [u0, du0], events=esc, rtol=1e-12, atol=1e-14)
        phi_inf = sol.t_events[0][0]
        assert res.status[i] == kerr.ESCAPED
        # the tracer stops at r = 100 M and extrapolates (straight line + first-order bending tail)
        assert res.phi_inf[i] == pytest.approx(phi_inf, abs=1e-6), (x, res.phi_inf[i], phi_inf)


def test_tracer_runs_in_float64_on_cpu_and_reports_steps():
    cam = kerr.Camera(r=30.0, theta=math.radians(80), a=0.5)
    xs = np.linspace(-0.3, 0.3, 64)
    rays = kerr.camera_rays(cam, xs, 0.05 * np.ones_like(xs))
    res = kerr.trace(cam, rays, kerr.isco(0.5), 15.0)
    assert res.stats["unresolved"] == 0
    assert set(np.unique(res.status)) <= {kerr.ESCAPED, kerr.CAPTURED, kerr.DISK}
    assert (res.steps > 0).all()
    assert torch.get_default_dtype() in (torch.float32, torch.float64)


# ----------------------------------------------------------------------------------------------
# disk physics
# ----------------------------------------------------------------------------------------------


def test_page_thorne_closed_form_matches_quadrature():
    for a in (0.0, 0.6, 0.9):
        r = np.linspace(kerr.isco(a) * 1.01, 50, 200)
        F1 = kerr.page_thorne_flux(r, a)
        F2 = kerr.page_thorne_flux_numeric(r, a)
        assert np.max(np.abs(F1 - F2)) / F1.max() < 5e-4
        assert kerr.page_thorne_flux(kerr.isco(a) * 0.99, a) == 0.0
    # Newtonian limit far out: F -> 3 / (8 pi r^3) (approached slowly: the bracket has ln x terms)
    r = 1e10
    assert float(kerr.page_thorne_flux(r, 0.0)) * 8 * math.pi * r ** 3 / 3 == pytest.approx(1.0, rel=1e-3)


def test_redshift_signs_and_face_on_limit():
    a = 0.6
    cam = kerr.Camera(r=40.0, theta=math.radians(85), a=a)
    xs = np.linspace(-0.4, 0.4, 81)
    rays = kerr.camera_rays(cam, xs, np.full_like(xs, -0.02))     # a row across the near disk
    res = kerr.trace(cam, rays, kerr.isco(a), 20.0)
    m = np.isfinite(res.hit_r[0])
    g, g_grav = kerr.redshift(res.b[m], res.hit_r[0][m], a, res.E_cam[m], cam.r, cam.theta)
    x = xs[m]
    # disk rotates toward +phi: the left side (x < 0, b > 0) approaches the camera -> blueshift
    assert g[x < -0.15].min() > 1.0
    assert g[x > 0.15].max() < 1.0
    assert (g_grav < 1).all()
    # face-on, far away, a = 0: g = sqrt(1 - 3M/r) (transverse Doppler + gravitational), times the
    # static camera's own blueshift 1 / sqrt(1 - 2M/r_cam)
    r_cam = 1e4
    cam0 = kerr.Camera(r=r_cam, theta=1e-6, a=0.0)        # on the axis: every ray has L_z ~ 0
    rays0 = kerr.camera_rays(cam0, np.array([0.0012]), np.array([0.0]))
    res0 = kerr.trace(cam0, rays0, 6.0, 30.0, rtol=1e-9)
    r_e = res0.hit_r[0][0]
    g0, _ = kerr.redshift(res0.b, res0.hit_r[0], 0.0, res0.E_cam, cam0.r, cam0.theta)
    assert abs(res0.b[0]) < 1e-4
    expected = math.sqrt(1 - 3 / r_e) / math.sqrt(1 - 2 / r_cam)
    assert g0[0] == pytest.approx(expected, rel=2e-3)
