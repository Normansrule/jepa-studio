"""Procedural celestial sphere, sampled in the asymptotic direction of each escaped ray.

No textures are shipped: stars are placed by an integer hash on a cube map of the sky (one jittered
star per cell, three layers of cell sizes with a steep brightness distribution, blackbody colours
from 2500 K to 25 000 K), plus a faint band of diffuse light. The checker grid (10 degree squares in
celestial latitude / longitude) makes the lensing map easy to read: the Einstein ring and the
mirrored secondary images show up as distorted squares.
"""
from __future__ import annotations

import numpy as np

from . import color


def _hash(*ints, seed: int = 0):
    """Deterministic 32-bit hash of integer arrays (a PCG-style mix), returned as floats in [0, 1)."""
    h = np.uint32(2166136261 ^ (seed * 16777619 & 0xFFFFFFFF))
    h = np.full(np.broadcast(*ints).shape, h, dtype=np.uint32)
    with np.errstate(over="ignore"):
        for v in ints:
            h = (h ^ np.asarray(v).astype(np.uint32)) * np.uint32(16777619)
            h ^= h >> np.uint32(15)
            h *= np.uint32(2246822519)
            h ^= h >> np.uint32(13)
    return h.astype(np.float64) / 4294967296.0


def direction(theta, phi):
    s = np.sin(theta)
    return np.stack([s * np.cos(phi), s * np.sin(phi), np.cos(theta)], -1)


def _cube(d):
    ax = np.argmax(np.abs(d), -1)
    m = np.take_along_axis(d, ax[..., None], -1)[..., 0]
    face = ax * 2 + (m < 0)
    i1 = (ax + 1) % 3
    i2 = (ax + 2) % 3
    u = np.take_along_axis(d, i1[..., None], -1)[..., 0] / np.abs(m)
    v = np.take_along_axis(d, i2[..., None], -1)[..., 0] / np.abs(m)
    return face, u, v, ax, i1, i2, np.sign(m)


# (cells per cube-face edge, brightness scale, probability a cell holds a star)
LAYERS = ((120, 0.5, 0.35), (40, 2.0, 0.5), (13, 8.0, 0.6))


def tangent_basis(d):
    """Orthonormal (e1, e2) perpendicular to each unit vector d."""
    up = np.where((np.abs(d[..., 2]) > 0.99)[..., None], np.array([1.0, 0, 0]), np.array([0, 0, 1.0]))
    e1 = np.cross(up, d)
    e1 /= np.linalg.norm(e1, axis=-1, keepdims=True)
    return e1, np.cross(d, e1)


def stars(d, pix_angle: float, seed: int = 0, density: float = 1.0, jac=None):
    """Linear-sRGB radiance of the star field in unit directions d [..., 3].

    Stars are point sources. A pixel's footprint on the sky is the Gaussian with covariance
    J J^T s^2, where J [..., 2, 2] is the Jacobian of the lens map (pixel -> sky, in the tangent basis
    of d, radians per pixel; None = unlensed, J = pix_angle * I). A star inside the footprint adds
    flux * exp(-q/2) * pix^2 s^2 / sqrt(det Sigma), i.e. its image stays about one pixel wide and its
    brightness scales with the magnification mu = pix^2 / |det J| (capped near caustics)."""
    face, u, v, ax, i1, i2, sg = _cube(d)
    e1, e2 = tangent_basis(d)
    s = 0.6
    pix2 = (pix_angle * s) ** 2
    if jac is None:
        S11 = np.full(d.shape[:-1], pix2)
        S22 = S11.copy()
        S12 = np.zeros_like(S11)
    else:
        J = jac
        S11 = (J[..., 0, 0] ** 2 + J[..., 0, 1] ** 2) * s * s
        S22 = (J[..., 1, 0] ** 2 + J[..., 1, 1] ** 2) * s * s
        S12 = (J[..., 0, 0] * J[..., 1, 0] + J[..., 0, 1] * J[..., 1, 1]) * s * s
    floor = pix2 * 0.05            # magnification cap ~ 1 / sqrt(0.05^2) = 20 near caustics
    S11, S22 = S11 + floor, S22 + floor
    det = np.maximum(S11 * S22 - S12 * S12, 1e-30)
    norm = pix2 / np.sqrt(det)
    out = np.zeros(d.shape[:-1] + (3,))
    for li, (n, bright, p) in enumerate(LAYERS):
        n = max(8, int(n * density))
        cu = np.clip(np.floor((u + 1) / 2 * n), 0, n - 1).astype(np.int64)
        cv = np.clip(np.floor((v + 1) / 2 * n), 0, n - 1).astype(np.int64)
        h1 = _hash(face, cu, cv, li, seed=seed)
        h2 = _hash(face, cu, cv, li + 7, seed=seed)
        h3 = _hash(face, cu, cv, li + 13, seed=seed)
        h4 = _hash(face, cu, cv, li + 29, seed=seed)
        su = ((cu + 0.2 + 0.6 * h1) / n) * 2 - 1
        sv = ((cv + 0.2 + 0.6 * h2) / n) * 2 - 1
        sd = np.zeros_like(d)
        np.put_along_axis(sd, ax[..., None], sg[..., None], -1)
        np.put_along_axis(sd, i1[..., None], su[..., None], -1)
        np.put_along_axis(sd, i2[..., None], sv[..., None], -1)
        sd /= np.linalg.norm(sd, axis=-1, keepdims=True)
        dv = sd - d
        x1 = np.sum(dv * e1, -1)
        x2 = np.sum(dv * e2, -1)
        q = (S22 * x1 * x1 - 2 * S12 * x1 * x2 + S11 * x2 * x2) / det
        present = h3 < p
        mag = bright * (h4 ** 6 * 30 + 0.15)          # steep: most stars faint, a few bright
        w = present * mag * np.exp(-0.5 * q) * norm
        # where one pixel's footprint covers many cells (strong demagnification next to the shadow) a
        # single sampled star is noise: use the layer's mean surface brightness instead (lensing
        # conserves surface brightness), mean = p <flux> 2 pi s^2 pix^2 / cell area, <h^6> = 1/7
        cell_area = (2.0 / n) ** 2
        mean = p * bright * (30 / 7 + 0.15) * 2 * np.pi * s * s * pix_angle ** 2 / cell_area
        blend = np.clip((np.sqrt(det) / (s * s) / cell_area - 0.3) / 0.7, 0, 1)     # sqrt(det) ~ |det J| s^2
        w = w * (1 - blend) + mean * blend
        temp = 2500 + 22500 * _hash(face, cu, cv, li + 41, seed=seed) ** 2.5
        out += w[..., None] * color.blackbody_rgb(temp, luminance=False).clip(0)
    # faint diffuse band (a stylised galaxy), tilted against the black hole's equator
    nrm = np.array([0.35, -0.25, 0.9])
    nrm /= np.linalg.norm(nrm)
    z = d @ nrm
    band = np.exp(-(z / 0.18) ** 2)
    lon = np.arctan2(d[..., 1], d[..., 0])
    mod = 0.6 + 0.25 * np.sin(3 * lon + 1.3) + 0.15 * np.sin(7 * lon + 2 * z * 10)
    out += (band * mod * 0.02)[..., None] * np.array([1.0, 0.86, 0.72])
    return out


def lens_jacobian(D, pix_angle: float):
    """Jacobian [..., 2, 2] of the pixel -> sky map from a grid of sky directions D [..., h, w, 3]
    (NaN where the ray did not escape), by central differences, in the tangent basis of D. Pixels
    whose neighbours did not escape fall back to the unlensed footprint."""
    def diff(axis):
        fwd = np.roll(D, -1, axis) - D
        bwd = D - np.roll(D, 1, axis)
        g = 0.5 * (fwd + bwd)
        g = np.where(np.isfinite(g), g, np.where(np.isfinite(fwd), fwd, bwd))
        idx = [slice(None)] * D.ndim
        idx[axis] = 0
        g[tuple(idx)] = fwd[tuple(idx)]
        idx[axis] = -1
        g[tuple(idx)] = bwd[tuple(idx)]
        return g
    dx, dy = diff(D.ndim - 2), diff(D.ndim - 3)
    e1, e2 = tangent_basis(np.nan_to_num(D, nan=1.0) / np.linalg.norm(np.nan_to_num(D, nan=1.0), axis=-1,
                                                                      keepdims=True))
    J = np.stack([np.stack([np.sum(dx * e1, -1), np.sum(dy * e1, -1)], -1),
                  np.stack([np.sum(dx * e2, -1), np.sum(dy * e2, -1)], -1)], -2)
    bad = ~np.isfinite(J).all((-1, -2))
    J[bad] = np.eye(2) * pix_angle
    return J


def grid(d, step_deg: float = 10.0):
    """Checker board on the celestial sphere with thin lines at every `step_deg` degrees."""
    th = np.arccos(np.clip(d[..., 2], -1, 1))
    ph = np.arctan2(d[..., 1], d[..., 0])
    s = np.radians(step_deg)
    i = np.floor(th / s).astype(int)
    j = np.floor(ph / s).astype(int)
    chk = (i + j) % 2
    c0 = np.array([0.08, 0.10, 0.16])
    c1 = np.array([0.20, 0.26, 0.40])
    out = np.where(chk[..., None] == 1, c1, c0)
    # coloured octants so mirrored images are recognisable
    hue = np.where((ph > 0)[..., None], np.array([1.0, 0.85, 0.6]), np.array([0.6, 0.85, 1.0]))
    out = out * hue
    ft = np.abs(th / s - np.round(th / s))
    fp = np.abs(ph / s - np.round(ph / s))
    line = np.minimum(ft, fp * np.maximum(np.sin(th), 0.05)) < 0.03
    out = np.where(line[..., None], np.array([0.55, 0.6, 0.7]), out)
    return out * 0.6


def sample(kind: str, theta, phi, pix_angle: float, seed: int = 0, jac=None):
    d = direction(theta, phi)
    if kind == "stars":
        return stars(d, pix_angle, seed, jac=jac)
    if kind == "grid":
        return grid(d)
    return np.zeros(d.shape)
