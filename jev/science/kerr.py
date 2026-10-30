"""Kerr spacetime and a vectorized null-geodesic tracer (units G = c = M = 1).

Geometry (Boyer-Lindquist coordinates t, r, theta, phi; spin a = J/M in [0, 1)):

    Sigma = r^2 + a^2 cos^2(theta)          Delta = r^2 - 2 r + a^2
    A     = (r^2 + a^2)^2 - a^2 Delta sin^2(theta)
    alpha = sqrt(Sigma Delta / A)           (lapse of the ZAMO / FIDO)
    omega = 2 a r / A                       (frame-dragging angular velocity)
    varpi = sqrt(A / Sigma) sin(theta)      (circumferential radius)

Photons have conserved energy E, axial angular momentum L and Carter constant Q (Carter 1968).
With E = 1, b = L/E and q = Q/E^2, and the Mino time sigma (d sigma = d lambda / Sigma):

    (dr/dsigma)^2     = R(r)      = P^2 - Delta [(b - a)^2 + q],     P = r^2 + a^2 - a b
    (dtheta/dsigma)^2 = Theta(th) = q + a^2 cos^2(th) - b^2 cot^2(th)
    dphi/dsigma       = a P / Delta + b / sin^2(th) - a
    dt/dsigma         = (r^2 + a^2) P / Delta + a (b - a sin^2(th))

We integrate the second-order form d^2r/dsigma^2 = R'(r)/2, d^2theta/dsigma^2 = Theta'(theta)/2,
which has no square roots and passes smoothly through radial and polar turning points (no sign
bookkeeping). The first integrals R and Theta are then *checks* (tests/test_science_kerr.py).
Rays are traced backwards from a camera at rest in the locally non-rotating frame (ZAMO/FIDO),
as in James et al. (2015), with an optional boost along e_phi for a moving camera.

References: Carter, Phys. Rev. 174, 1559 (1968); Bardeen, Press & Teukolsky, ApJ 178, 347 (1972);
Bardeen, in Black Holes (Les Houches 1972), p. 215 (1973); Page & Thorne, ApJ 191, 499 (1974);
James, von Tunzelmann, Franklin & Thorne, Class. Quantum Grav. 32, 065001 (2015), arXiv:1502.03808.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable

import numpy as np
import torch

# ----------------------------------------------------------------------------------------------
# Closed-form geometry
# ----------------------------------------------------------------------------------------------


def horizon(a: float) -> float:
    """Outer event horizon r_+ = M + sqrt(M^2 - a^2)."""
    return 1.0 + math.sqrt(max(0.0, 1.0 - a * a))


def isco(a: float, prograde: bool = True) -> float:
    """Innermost stable circular orbit, Bardeen, Press & Teukolsky (1972) eq. 2.21."""
    z1 = 1 + (1 - a * a) ** (1 / 3) * ((1 + a) ** (1 / 3) + (1 - a) ** (1 / 3))
    z2 = math.sqrt(3 * a * a + z1 * z1)
    s = math.sqrt(max(0.0, (3 - z1) * (3 + z1 + 2 * z2)))
    return 3 + z2 - s if prograde else 3 + z2 + s


def photon_orbit(a: float, prograde: bool = True) -> float:
    """Radius of the circular equatorial photon orbit, r = 2M[1 + cos(2/3 arccos(-+a))]."""
    return 2 * (1 + math.cos(2 / 3 * math.acos(-a if prograde else a)))


def zamo(r, theta, a):
    """(alpha, omega, varpi, Sigma, Delta) of the zero-angular-momentum observer. numpy or floats."""
    s2 = np.sin(theta) ** 2
    sig = r * r + a * a * np.cos(theta) ** 2
    dl = r * r - 2 * r + a * a
    A = (r * r + a * a) ** 2 - a * a * dl * s2
    return np.sqrt(sig * dl / A), 2 * a * r / A, np.sqrt(A / sig) * np.sqrt(s2), sig, dl


def kepler_omega(r, a):
    """Angular velocity dphi/dt of a prograde circular equatorial geodesic."""
    return 1.0 / (r ** 1.5 + a)


def circular_orbit(r, a):
    """(u^t, E, L, Omega) of prograde circular equatorial geodesics (Bardeen et al. 1972, 2.12-2.13).
    Valid for r above the prograde photon orbit; stable for r >= isco(a)."""
    r = np.asarray(r, dtype=float)
    x = np.sqrt(r)
    den = r ** 0.75 * np.sqrt(np.maximum(r ** 1.5 - 3 * x + 2 * a, 1e-300))
    ut = (r ** 1.5 + a) / den
    E = (r ** 1.5 - 2 * x + a) / den
    L = (r * r - 2 * a * x + a * a) / den
    return ut, E, L, 1.0 / (r ** 1.5 + a)


def critical_curve(a: float, inclination_deg: float, r_obs: float | None = None, n: int = 2000):
    """Boundary of the shadow (photons asymptotic to spherical photon orbits), Bardeen (1973).

    For each spherical photon orbit radius r_p the constants are
        b(r_p) = [r_p^2 (3 - r_p) - a^2 (r_p + 1)] / [a (r_p - 1)]
        q(r_p) = r_p^3 [4 a^2 - r_p (r_p - 3)^2] / [a^2 (r_p - 1)^2]
    r_obs=None returns Bardeen's celestial coordinates of a distant observer,
        alpha = -b / sin i,  beta = +-sqrt(q + a^2 cos^2 i - b^2 cot^2 i)   (units of M);
    otherwise returns the exact angles (radians) on the sky of a ZAMO camera at (r_obs, i), in the
    same screen convention as the renderer (x = right = +e_phi, y = up = -e_theta).
    Returns (x, y) arrays tracing the closed curve.
    """
    th = math.radians(inclination_deg)
    if a < 1e-6:
        rp = np.full(n, 3.0)
        b_abs = math.sqrt(27.0)
        ang = np.linspace(0, 2 * np.pi, n)
        b = b_abs * np.cos(ang)
        q = b_abs ** 2 - b ** 2   # a=0: q + b^2 = 27
    else:
        r1, r2 = photon_orbit(a, True), photon_orbit(a, False)
        u = np.linspace(0, np.pi, n)
        rp = r1 + (r2 - r1) * (1 - np.cos(u)) / 2      # cluster near the ends
        b = (rp ** 2 * (3 - rp) - a * a * (rp + 1)) / (a * (rp - 1))
        q = rp ** 3 * (4 * a * a - rp * (rp - 3) ** 2) / (a * a * (rp - 1) ** 2)
    c, s = math.cos(th), math.sin(th)
    theta_term = q + a * a * c * c - b * b * c * c / (s * s)
    ok = theta_term >= 0
    b, q, theta_term = b[ok], q[ok], theta_term[ok]
    if r_obs is None:
        x = -b / s
        y = np.sqrt(theta_term)
        return np.concatenate([x, x[::-1]]), np.concatenate([y, -y[::-1]])
    al, om, vp, sig, dl = zamo(r_obs, th, a)
    P = r_obs ** 2 + a * a - a * b
    R = P * P - dl * ((b - a) ** 2 + q)
    ez = (1 - om * b) / al                       # ZAMO-measured energy for E = 1
    n_phi = -b / (vp * ez)
    n_th = np.sqrt(theta_term) / (math.sqrt(sig) * ez)
    n_r = -np.sqrt(np.maximum(R, 0)) / (math.sqrt(sig * dl) * ez)
    x = np.arctan2(n_phi, -n_r)
    y = np.arctan2(n_th, np.hypot(-n_r, n_phi))
    return np.concatenate([x, x[::-1]]), np.concatenate([y, -y[::-1]])


# ----------------------------------------------------------------------------------------------
# Novikov-Thorne / Page-Thorne thin disk
# ----------------------------------------------------------------------------------------------


def page_thorne_flux(r, a: float):
    """Page & Thorne (1974) flux of a Novikov-Thorne disk with a zero-torque inner edge at the ISCO,
    in units of (accretion rate) / M^2, closed form with x = sqrt(r/M) (their eq. 15n):

        F = 3 / (8 pi) * 1 / (x^4 (x^3 - 3x + 2a)) * [x - x0 - 3/2 a ln(x/x0)
            - sum_i 3 (x_i - a)^2 / (x_i (x_i - x_j)(x_i - x_k)) ln((x - x_i)/(x0 - x_i))]

    with x0 = sqrt(r_isco) and x_i the roots of x^3 - 3x + 2a = 0. Zero for r <= r_isco.
    Agrees with the direct quadrature in page_thorne_flux_numeric (tested)."""
    r = np.asarray(r, dtype=float)
    x = np.sqrt(np.maximum(r, 1e-12))
    x0 = math.sqrt(isco(a))
    ac = math.acos(max(-1.0, min(1.0, a)))
    x1 = 2 * math.cos(ac / 3 - math.pi / 3)
    x2 = 2 * math.cos(ac / 3 + math.pi / 3)
    x3 = -2 * math.cos(ac / 3)
    roots = (x1, x2, x3)
    with np.errstate(divide="ignore", invalid="ignore"):
        br = x - x0 - 1.5 * a * np.log(x / x0)
        for i, xi in enumerate(roots):
            xj, xk = [roots[k] for k in range(3) if k != i]
            br = br - 3 * (xi - a) ** 2 / (xi * (xi - xj) * (xi - xk)) * np.log(np.abs((x - xi) / (x0 - xi)))
        F = 3 / (8 * math.pi) * br / (x ** 4 * (x ** 3 - 3 * x + 2 * a))
    return np.where(r > x0 * x0, np.nan_to_num(F), 0.0)


def page_thorne_flux_numeric(r, a: float, n: int = 20000):
    """Direct quadrature of the Page-Thorne integral (independent check of the closed form):
    F = -Omega_,r / (4 pi sqrt(-g) (E - Omega L)^2) * int_{r_isco}^{r} (E - Omega L) L_,r dr, sqrt(-g) = r."""
    r = np.atleast_1d(np.asarray(r, dtype=float))
    r0 = isco(a)
    grid = np.linspace(r0, max(float(r.max()), r0 + 1e-6), n)
    _, E, L, Om = circular_orbit(grid, a)
    dL = np.gradient(L, grid)
    integrand = (E - Om * L) * dL
    cum = np.concatenate([[0.0], np.cumsum(0.5 * (integrand[1:] + integrand[:-1]) * np.diff(grid))])
    dOm = np.gradient(Om, grid)
    F = -dOm / (4 * math.pi * grid * (E - Om * L) ** 2) * cum
    return np.where(r > r0, np.interp(r, grid, F), 0.0)


# ----------------------------------------------------------------------------------------------
# Camera: pixel directions -> conserved quantities and initial Mino velocities
# ----------------------------------------------------------------------------------------------


@dataclass
class Camera:
    r: float                      # Boyer-Lindquist radius (units of M)
    theta: float                  # polar angle (radians); inclination from the spin axis
    a: float
    beta: float = 0.0             # speed relative to the ZAMO along +e_phi (0 = ZAMO camera)

    @property
    def frame(self):
        return zamo(self.r, self.theta, self.a)


def camera_rays(cam: Camera, nx: np.ndarray, ny: np.ndarray):
    """Map screen-plane coordinates (tangent of the angle right / up of the optical axis) to rays.

    The camera looks at the hole: forward = -e_r, right = +e_phi, up = -e_theta (spin axis up).
    Returns dict of float64 numpy arrays: b, q, vr, vth (Mino velocities of the *backward* ray,
    normalised to E = 1), E_cam (photon energy measured by the camera when E = 1) and n_F (the
    look direction in the ZAMO frame)."""
    nx = np.asarray(nx, dtype=float)
    ny = np.asarray(ny, dtype=float)
    # look direction in the camera frame (components along e_r, e_theta, e_phi)
    norm = np.sqrt(1 + nx * nx + ny * ny)
    n_r, n_th, n_ph = -1 / norm, -ny / norm, nx / norm
    D = np.ones_like(n_r)                       # E_camera / E_ZAMO
    if cam.beta:
        # aberration: the photon propagates along m = -n; the camera moves with speed beta along
        # +e_phi relative to the ZAMO (James et al. 2015, appendix A.1)
        bt = cam.beta
        g = 1 / math.sqrt(1 - bt * bt)
        m_ph, m_r, m_th = -n_ph, -n_r, -n_th
        den = 1 + bt * m_ph
        mF_ph = (m_ph + bt) / den
        mF_r, mF_th = m_r / (g * den), m_th / (g * den)
        n_r, n_th, n_ph = -mF_r, -mF_th, -mF_ph
        D = 1 / (g * den)
    al, om, vp, sig, dl = cam.frame
    # covariant momentum of the incoming photon with ZAMO energy 1
    p_r = -n_r * math.sqrt(sig / dl)
    p_th = -n_th * math.sqrt(sig)
    L = -n_ph * vp
    E = al + om * L
    b = L / E
    s, c = math.sin(cam.theta), math.cos(cam.theta)
    q = (p_th ** 2 + c * c * (L * L / (s * s) - cam.a ** 2 * E * E)) / (E * E)
    vr = -dl * p_r / E          # backward ray: minus the photon's dr/dsigma
    vth = -p_th / E
    return {"b": b, "q": q, "vr": vr, "vth": vth, "E_cam": D / E, "n_F": (n_r, n_th, n_ph)}


def redshift(b, r_emit, a: float, E_cam, r_cam: float, theta_cam: float):
    """g = nu_obs / nu_emit for a photon (E = 1, L = b) from a prograde Keplerian disk element at
    r_emit to the camera (E_cam = photon energy measured by the camera when E = 1):

        g = E_cam / (u^t (1 - Omega b))

    Also returns the gravitational part g_grav = alpha(r_emit) / alpha(r_cam), the lapse ratio
    (the shift between ZAMOs for a photon with zero angular momentum). The rest, g / g_grav, is the
    "Doppler" part: orbital motion of the gas, frame dragging and any camera motion."""
    ut, _, _, Om = circular_orbit(r_emit, a)
    g = E_cam / (ut * (1 - Om * b))
    al_e = zamo(r_emit, np.pi / 2, a)[0]
    al_c = zamo(r_cam, theta_cam, a)[0]
    return g, al_e / al_c * np.ones_like(g)


# ----------------------------------------------------------------------------------------------
# Vectorized Dormand-Prince 5(4) tracer
# ----------------------------------------------------------------------------------------------

_C = (0, 1 / 5, 3 / 10, 4 / 5, 8 / 9, 1, 1)
_A = (
    (),
    (1 / 5,),
    (3 / 40, 9 / 40),
    (44 / 45, -56 / 15, 32 / 9),
    (19372 / 6561, -25360 / 2187, 64448 / 6561, -212 / 729),
    (9017 / 3168, -355 / 33, 46732 / 5247, 49 / 176, -5103 / 18656),
    (35 / 384, 0, 500 / 1113, 125 / 192, -2187 / 6784, 11 / 84),
)
_E = (71 / 57600, 0, -71 / 16695, 71 / 1920, -17253 / 339200, 22 / 525, -1 / 40)

ESCAPED, CAPTURED, UNRESOLVED, DISK = 0, 1, 2, 3


@dataclass
class TraceResult:
    status: np.ndarray            # ESCAPED / CAPTURED / UNRESOLVED / DISK (stopped by an opaque disk)
    theta_inf: np.ndarray         # asymptotic direction of escaped rays (celestial sphere)
    phi_inf: np.ndarray
    hit_r: np.ndarray             # [K, N] equatorial-disk crossings inside [r_in, r_out] (NaN = none)
    hit_phi: np.ndarray
    hit_t: np.ndarray             # coordinate-time delay camera -> crossing
    hit_order: np.ndarray         # how many equatorial crossings preceded this one (0 = primary image)
    b: np.ndarray
    q: np.ndarray
    E_cam: np.ndarray
    steps: np.ndarray
    stats: dict = field(default_factory=dict)


def _rhs(Y, b, q, a):
    r, th, vr, vth = Y[0], Y[1], Y[4], Y[5]
    s = torch.sin(th)
    c = torch.cos(th)
    s = torch.where(s.abs() < 1e-9, torch.full_like(s, 1e-9), s)
    s2 = s * s
    P = r * r + a * a - a * b
    dl = r * r - 2 * r + a * a
    K = (b - a) ** 2 + q
    out = torch.empty_like(Y)
    out[0] = vr
    out[1] = vth
    out[2] = -(a * P / dl + b / s2 - a)                      # backward ray: phi runs the other way
    out[3] = (r * r + a * a) * P / dl + a * (b - a * s2)     # coordinate-time delay accumulates
    out[4] = 2 * r * P - (r - 1) * K                          # R'(r) / 2
    out[5] = -a * a * s * c + b * b * c / (s2 * s)            # Theta'(theta) / 2
    return out


def radial_potential(r, b, q, a):
    """R(r) = (r^2 + a^2 - a b)^2 - Delta [(b - a)^2 + q]  (E = 1)."""
    P = r * r + a * a - a * b
    return P * P - (r * r - 2 * r + a * a) * ((b - a) ** 2 + q)


def polar_potential(th, b, q, a):
    """Theta(theta) = q + a^2 cos^2 theta - b^2 cot^2 theta  (E = 1)."""
    s, c = (torch.sin(th), torch.cos(th)) if torch.is_tensor(th) else (np.sin(th), np.cos(th))
    return q + a * a * c * c - b * b * c * c / (s * s)


def asymptotic_direction(Y, dphi, a, b=None, q=None):
    """Direction (theta, phi) at infinity of rays at large r: continue each ray as a straight line in
    the flat oblate-spheroidal coordinates that Boyer-Lindquist reduce to for M -> 0,
    x = sqrt(r^2 + a^2) sin(th) cos(ph), y = ... sin(ph), z = r cos(th).

    The bending left beyond r is added to first order in M: expanding
    dphi/dr = (l/r^2)(1 - l^2/r^2 + 2 M l^2/r^3)^(-1/2) (l^2 = q + b^2, exact for a = 0) gives a remaining
    sweep asin(l/r) - M l^3/(4 r^4), while the coordinate straight line gives asin(l/r) - M l^3/r^4, so
    the direction is rotated on by 3 M l^3 / (4 r^4) within the orbital plane. What is left is
    O(M^2 l^3/r^5, a M l/r^3): below 1e-6 rad at r = 100 M."""
    r, th, ph, vr, vth = Y[0], Y[1], Y[2], Y[4], Y[5]
    rt = torch.sqrt(r * r + a * a)
    drt = r * vr / rt
    s, c, sp, cp = torch.sin(th), torch.cos(th), torch.sin(ph), torch.cos(ph)
    vx = drt * s * cp + rt * c * vth * cp - rt * s * sp * dphi
    vy = drt * s * sp + rt * c * vth * sp + rt * s * cp * dphi
    vz = vr * c - r * s * vth
    nrm = torch.sqrt(vx * vx + vy * vy + vz * vz)
    vx, vy, vz = vx / nrm, vy / nrm, vz / nrm
    if b is not None:
        l2 = torch.clamp(q + b * b, min=0)
        dl = 0.75 * l2 * torch.sqrt(l2) / r ** 4
        xx, xy, xz = rt * s * cp, rt * s * sp, r * c          # position, then its part normal to v
        dot = xx * vx + xy * vy + xz * vz
        wx, wy, wz = xx - dot * vx, xy - dot * vy, xz - dot * vz
        wn = torch.sqrt(wx * wx + wy * wy + wz * wz) + 1e-300
        cd, sd = torch.cos(dl), torch.sin(dl)
        vx, vy, vz = cd * vx - sd * wx / wn, cd * vy - sd * wy / wn, cd * vz - sd * wz / wn
        nrm = torch.ones_like(vx)
    ph_inf = torch.atan2(vy, vx)
    ph_inf = ph_inf + 2 * math.pi * torch.round((ph - ph_inf) / (2 * math.pi))   # keep the winding count
    return torch.acos(torch.clamp(vz / nrm, -1, 1)), ph_inf


def _hermite(y0, y1, d0, d1, s):
    s2, s3 = s * s, s * s * s
    return ((2 * s3 - 3 * s2 + 1) * y0 + (s3 - 2 * s2 + s) * d0 + (-2 * s3 + 3 * s2) * y1 + (s3 - s2) * d1)


def trace(cam: Camera, rays: dict, r_in: float, r_out: float, max_hits: int = 1, rtol: float = 1e-6,
          max_steps: int = 4000, device: str = "cpu", progress: Callable[[float, str], None] | None = None,
          cancelled: Callable[[], bool] | None = None, chunk: int = 1 << 18, project: bool = True) -> TraceResult:
    """Trace rays backwards from the camera until each is captured (r -> r_+), escapes (r > r_esc,
    moving out) or has recorded `max_hits` disk crossings (opaque disk: max_hits = 1).

    Every crossing of the equatorial plane is located by cubic Hermite interpolation of the
    accepted step (Newton on theta(s) = pi/2); crossings with r_in <= r <= r_out are disk hits, the
    others (inside the ISCO or beyond the outer edge) let the ray continue, which is what produces
    the secondary and higher-order images of the disk.

    project=True rescales vr and vth after each accepted step so that vr^2 = R(r) and vth^2 = Theta,
    except near turning points (where they are small); this removes the slow drift of the second-order
    form on long legs. The raw integrator (project=False) conserves them to ~rtol (tested)."""
    a = float(cam.a)
    n = rays["b"].size
    dt = torch.float64
    dev = torch.device(device)
    rh = horizon(a)
    # inside the prograde photon orbit an ingoing ray has no radial turning point left: it is captured
    r_stop = rh + 0.5 * (photon_orbit(a, True) - rh)
    r_esc = max(1.5 * cam.r, 2.0 * r_out, 100.0)
    out = TraceResult(status=np.full(n, UNRESOLVED, np.int8), theta_inf=np.full(n, np.nan),
                      phi_inf=np.full(n, np.nan), hit_r=np.full((max_hits, n), np.nan),
                      hit_phi=np.full((max_hits, n), np.nan), hit_t=np.full((max_hits, n), np.nan),
                      hit_order=np.full((max_hits, n), -1, np.int16),
                      b=rays["b"].ravel(), q=rays["q"].ravel(), E_cam=rays["E_cam"].ravel(),
                      steps=np.zeros(n, np.int32))
    total_steps = 0
    for start in range(0, n, chunk):
        sl = slice(start, min(n, start + chunk))
        total_steps += _trace_chunk(cam, rays, sl, out, r_in, r_out, max_hits, rtol, max_steps, dev, dt,
                                    r_stop, r_esc, progress, cancelled, n, project)
        if cancelled and cancelled():
            break
    out.stats = {"rays": n, "mean_steps": float(out.steps.mean()) if n else 0.0,
                 "unresolved": int((out.status == UNRESOLVED).sum()), "r_escape": r_esc, "r_stop": r_stop}
    return out


def _trace_chunk(cam, rays, sl, out, r_in, r_out, max_hits, rtol, max_steps, dev, dt, r_stop, r_esc,
                 progress, cancelled, n_total, project=True):
    a = float(cam.a)
    t = lambda x: torch.as_tensor(np.ascontiguousarray(np.asarray(x).ravel()[sl]), dtype=dt, device=dev)  # noqa: E731
    b, q = t(rays["b"]), t(rays["q"])
    m = b.numel()
    Y = torch.zeros((6, m), dtype=dt, device=dev)
    Y[0] = cam.r
    Y[1] = cam.theta
    Y[4], Y[5] = t(rays["vr"]), t(rays["vth"])
    idx = torch.arange(sl.start, sl.start + m, device=dev)
    nhit = torch.zeros(m, dtype=torch.int64, device=dev)
    ncross = torch.zeros(m, dtype=torch.int64, device=dev)
    nstep = torch.zeros(m, dtype=torch.int64, device=dev)
    atol_ang = rtol
    # initial step: limit the relative change of r and the change of theta
    h = 0.02 / (Y[4].abs() / Y[0] + Y[5].abs() + 1e-12)
    k1 = _rhs(Y, b, q, a)
    half_pi = math.pi / 2
    hit_r = torch.full((max_hits, m), float("nan"), dtype=dt, device=dev)
    hit_phi = torch.full_like(hit_r, float("nan"))
    hit_t = torch.full_like(hit_r, float("nan"))
    hit_ord = torch.full((max_hits, m), -1, dtype=torch.int64, device=dev)
    status = torch.full((m,), UNRESOLVED, dtype=torch.int64, device=dev)
    th_inf = torch.full((m,), float("nan"), dtype=dt, device=dev)
    ph_inf = torch.full_like(th_inf, float("nan"))
    steps_out = torch.zeros(m, dtype=torch.int64, device=dev)
    loc = torch.arange(m, device=dev)       # active -> local index
    it = 0
    total = 0
    while loc.numel() and it < max_steps * 3:
        it += 1
        if cancelled and it % 16 == 0 and cancelled():
            break
        ks = [k1]
        for i in range(1, 7):
            Yi = Y.clone()
            for j, aij in enumerate(_A[i]):
                if aij:
                    Yi = Yi + (h * aij) * ks[j]
            ks.append(_rhs(Yi, b, q, a))
        Yn = Yi                     # stage 7 input is the 5th-order solution (FSAL)
        err = sum((h * e) * k for e, k in zip(_E, ks) if e)
        sc_r = rtol * torch.maximum(Y[0].abs(), Yn[0].abs())
        sc_vr = rtol * torch.maximum(torch.maximum(Y[4].abs(), Yn[4].abs()), Y[0] * Y[0] * 1e-3)
        sc_vth = rtol * torch.maximum(torch.maximum(Y[5].abs(), Yn[5].abs()), torch.ones_like(h))
        e = torch.stack([err[0] / sc_r, err[1] / atol_ang, err[2] / (atol_ang * 10), err[4] / sc_vr,
                         err[5] / sc_vth]).abs().amax(0)
        e = torch.nan_to_num(e, nan=1e6, posinf=1e6)
        acc = e <= 1.0
        # step-size update (standard PI-free controller, bounded growth)
        fac = torch.clamp(0.9 * torch.clamp(e, min=1e-10) ** -0.2, 0.2, 5.0)
        h_new = h * fac
        # never let a single step move theta by more than 0.15 rad or r by more than 30 %
        lim = 0.15 / (Yn[5].abs() + 1e-12)
        lim = torch.minimum(lim, 0.3 * Yn[0] / (Yn[4].abs() + 1e-12))
        h_new = torch.minimum(h_new, lim)
        k7 = ks[6]
        nstep += acc.long()
        total += int(acc.sum())
        if bool(acc.any()):
            ai = acc.nonzero().squeeze(1)
            y0, y1, d0, d1, hh = Y[:, ai], Yn[:, ai], ks[0][:, ai], k7[:, ai], h[ai]
            # the equator is cos(theta) = 0; theta may leave [0, pi] when a ray passes over a pole
            c0, c1 = torch.cos(y0[1]), torch.cos(y1[1])
            cross = ((c0 > 0) & (c1 <= 0)) | ((c0 < 0) & (c1 >= 0))
            if bool(cross.any()):
                ci = cross.nonzero().squeeze(1)
                th_star = half_pi + math.pi * torch.floor((torch.maximum(y0[1, ci], y1[1, ci]) - half_pi) / math.pi)
                g0, g1 = y0[1, ci] - th_star, y1[1, ci] - th_star
                D0, D1, H = d0[1, ci] * hh[ci], d1[1, ci] * hh[ci], hh[ci]
                s = torch.clamp(g0 / (g0 - g1), 0, 1)
                for _ in range(4):        # Newton on the Hermite cubic of theta(s) - pi/2
                    s2 = s * s
                    f = (2 * s2 * s - 3 * s2 + 1) * g0 + (s2 * s - 2 * s2 + s) * D0 + (-2 * s2 * s + 3 * s2) * g1 \
                        + (s2 * s - s2) * D1
                    fp = (6 * s2 - 6 * s) * g0 + (3 * s2 - 4 * s + 1) * D0 + (-6 * s2 + 6 * s) * g1 + (3 * s2 - 2 * s) * D1
                    s = torch.clamp(s - f / torch.where(fp.abs() < 1e-30, torch.full_like(fp, 1e-30), fp), 0, 1)
                Y0c, Y1c, d0c, d1c = y0[:, ci], y1[:, ci], d0[:, ci] * H, d1[:, ci] * H
                rc = _hermite(Y0c[0], Y1c[0], d0c[0], d1c[0], s)
                pc = _hermite(Y0c[2], Y1c[2], d0c[2], d1c[2], s)
                tc = _hermite(Y0c[3], Y1c[3], d0c[3], d1c[3], s)
                li = loc[ai[ci]]
                on_disk = (rc >= r_in) & (rc <= r_out)
                if bool(on_disk.any()):
                    di = on_disk.nonzero().squeeze(1)
                    lj = li[di]
                    slot = nhit[lj]
                    okslot = slot < max_hits
                    lj, slot, di = lj[okslot], slot[okslot], di[okslot]
                    hit_r[slot, lj] = rc[di]
                    hit_phi[slot, lj] = pc[di]
                    hit_t[slot, lj] = tc[di]
                    hit_ord[slot, lj] = ncross[lj]
                    nhit[lj] += 1
                ncross[li] += 1
            # commit accepted states
            if project:
                # keep the first integrals exact away from turning points: vr^2 = R(r), vth^2 = Theta(th)
                bb, qq = b[ai], q[ai]
                Rv = radial_potential(y1[0], bb, qq, a)
                Tv = polar_potential(y1[1], bb, qq, a)
                okr = (Rv > 1e-3 * y1[0] ** 4) & (y1[4] ** 2 > 0.25 * Rv)
                okt = (Tv > 1e-3 * (qq.abs() + bb * bb + a * a + 1e-12)) & (y1[5] ** 2 > 0.25 * Tv)
                y1[4] = torch.where(okr, torch.sign(y1[4]) * torch.sqrt(Rv.clamp(min=0)), y1[4])
                y1[5] = torch.where(okt, torch.sign(y1[5]) * torch.sqrt(Tv.clamp(min=0)), y1[5])
                d1 = d1.clone()
                d1[0], d1[1] = y1[4], y1[5]
            Y[:, ai] = y1
            k1[:, ai] = d1
        h = torch.where(acc, h_new, h * torch.clamp(fac, 0.1, 0.9))
        # termination tests (on the active set)
        r, vr = Y[0], Y[4]
        lg = loc
        done_cap = (r < r_stop) & (vr < 0)
        done_esc = (r > r_esc) & (vr > 0)
        done_hit = nhit[lg] >= max_hits
        done_max = nstep >= max_steps
        bad = ~torch.isfinite(Y).all(0)
        done = done_cap | done_esc | done_hit | done_max | bad
        if bool(done_esc.any()):
            ei = done_esc.nonzero().squeeze(1)
            th_e, ph_e = asymptotic_direction(Y[:, ei], k1[2, ei], a, b[ei], q[ei])
            th_inf[lg[ei]] = th_e
            ph_inf[lg[ei]] = ph_e
            status[lg[ei]] = ESCAPED
        if bool(done_cap.any()):
            status[lg[done_cap]] = CAPTURED
        hit_only = done_hit & ~done_esc & ~done_cap
        if bool(hit_only.any()):
            status[lg[hit_only]] = DISK          # opaque disk: nothing behind it is visible
        if bool(done.any()):
            steps_out[lg[done]] = nstep[done]
            keep = ~done
            Y, k1, h, b, q = Y[:, keep], k1[:, keep], h[keep], b[keep], q[keep]
            loc, nstep = loc[keep], nstep[keep]
            if progress and it % 8 == 0:
                fr = (sl.start + (m - loc.numel())) / max(1, n_total)
                progress(min(0.999, fr), "tracing geodesics")
    if loc.numel():
        steps_out[loc] = nstep
    gi = idx.cpu().numpy()
    out.status[gi] = status.cpu().numpy().astype(np.int8)
    out.theta_inf[gi] = th_inf.cpu().numpy()
    out.phi_inf[gi] = ph_inf.cpu().numpy()
    out.hit_r[:, gi] = hit_r.cpu().numpy()
    out.hit_phi[:, gi] = hit_phi.cpu().numpy()
    out.hit_t[:, gi] = hit_t.cpu().numpy()
    out.hit_order[:, gi] = hit_ord.cpu().numpy()
    out.steps[gi] = steps_out.cpu().numpy()
    return total


def integrate_single(cam: Camera, nx: float, ny: float, sigma_max: float = 10.0, rtol: float = 1e-9,
                     max_steps: int = 20000, project: bool = False):
    """Integrate one backward ray and return its sampled states (for the conservation tests).
    project=False is the raw second-order integration. Returns (Y [6, steps], b, q)."""
    rays = camera_rays(cam, np.array([nx]), np.array([ny]))
    a = cam.a
    b = torch.tensor(rays["b"], dtype=torch.float64)
    q = torch.tensor(rays["q"], dtype=torch.float64)
    Y = torch.tensor([[cam.r], [cam.theta], [0.0], [0.0], [float(rays["vr"][0])], [float(rays["vth"][0])]],
                     dtype=torch.float64)
    h = torch.tensor([1e-4 / cam.r])
    rows = [Y[:, 0].clone()]
    sig = 0.0
    rh = horizon(a)
    for _ in range(max_steps):
        ks = [_rhs(Y, b, q, a)]
        for i in range(1, 7):
            Yi = Y.clone()
            for j, aij in enumerate(_A[i]):
                if aij:
                    Yi = Yi + (h * aij) * ks[j]
            ks.append(_rhs(Yi, b, q, a))
        err = sum((h * e) * k for e, k in zip(_E, ks) if e)
        sc = rtol * torch.maximum(Y.abs(), Yi.abs()) + rtol
        e = float((err / sc)[[0, 1, 2, 4, 5]].abs().max())
        if e <= 1:
            Y = Yi
            if project:
                Rv = radial_potential(Y[0], b, q, a)
                Tv = polar_potential(Y[1], b, q, a)
                if Rv > 1e-3 * Y[0, 0] ** 4 and Y[4, 0] ** 2 > 0.25 * Rv:
                    Y[4] = torch.sign(Y[4]) * torch.sqrt(Rv)
                if Tv > 1e-3 * (abs(float(q)) + float(b) ** 2 + a * a + 1e-12) and Y[5, 0] ** 2 > 0.25 * Tv:
                    Y[5] = torch.sign(Y[5]) * torch.sqrt(Tv)
            sig += float(h)
            rows.append(Y[:, 0].clone())
            if Y[0, 0] < rh + 0.05 or Y[0, 0] > 1e3 or sig > sigma_max:
                break
        h = h * min(5.0, max(0.2, 0.9 * max(e, 1e-10) ** -0.2))
    return torch.stack(rows, 1).numpy(), float(rays["b"][0]), float(rays["q"][0])
