"""Blackbody colour and tone mapping.

Colour matching uses the multi-lobe Gaussian fit of the CIE 1931 2-degree observer by Wyman, Sloan &
Shirley, "Simple Analytic Approximations to the CIE XYZ Color Matching Functions", JCGT 2(2), 2013
(within the variability of the tabulated functions). A blackbody's XYZ is integrated against Planck's
law once, on a 1 nm grid, into a compact lookup table over log T (BB_TABLE), and converted to linear
sRGB (IEC 61966-2-1 matrix, D65 white). The renderer interpolates the table, it never re-integrates.

Because a blackbody stays a blackbody under a frequency shift g (I_nu / nu^3 is invariant, so
I_nu,obs(nu) = g^3 B_nu(nu / g; T) = B_nu(nu; g T)), the observed colour and visible brightness of a
disk element are exactly those of a blackbody at temperature g T.
"""
from __future__ import annotations

import numpy as np

H_PLANCK, C_LIGHT, K_BOLTZ = 6.62607015e-34, 2.99792458e8, 1.380649e-23

XYZ_TO_SRGB = np.array([[3.2404542, -1.5371385, -0.4985314],
                        [-0.9692660, 1.8760108, 0.0415560],
                        [0.0556434, -0.2040259, 1.0572252]])


def _g(x, mu, s1, s2):
    t = (x - mu) / np.where(x < mu, s1, s2)
    return np.exp(-0.5 * t * t)


def cie_xyz_bar(lam_nm):
    """CIE 1931 colour matching functions, Wyman-Sloan-Shirley multi-lobe fit."""
    lam = np.asarray(lam_nm, dtype=float)
    x = 1.056 * _g(lam, 599.8, 37.9, 31.0) + 0.362 * _g(lam, 442.0, 16.0, 26.7) - 0.065 * _g(lam, 501.1, 20.4, 26.2)
    y = 0.821 * _g(lam, 568.8, 46.9, 40.5) + 0.286 * _g(lam, 530.9, 16.3, 31.1)
    z = 1.217 * _g(lam, 437.0, 11.8, 36.0) + 0.681 * _g(lam, 459.0, 26.0, 13.8)
    return np.stack([x, y, z], -1)


def planck(lam_nm, T):
    """Spectral radiance B_lambda (W sr^-1 m^-3)."""
    lam = np.asarray(lam_nm, dtype=float)[..., None] * 1e-9
    T = np.atleast_1d(np.asarray(T, dtype=float))
    with np.errstate(over="ignore"):
        return 2 * H_PLANCK * C_LIGHT ** 2 / lam ** 5 / np.expm1(H_PLANCK * C_LIGHT / (lam * K_BOLTZ * T))


def blackbody_xyz(T):
    """XYZ (arbitrary but consistent scale) of blackbodies at temperatures T (K)."""
    lam = np.arange(360.0, 831.0, 1.0)
    w = cie_xyz_bar(lam)                     # [L, 3]
    B = planck(lam, T)                       # [L, nT]
    return (B.T @ w) * 1e-9                  # [nT, 3]


def xy_chromaticity(T):
    X = blackbody_xyz(T)
    s = X.sum(-1, keepdims=True)
    return X[:, :2] / s


# Lookup table: log10 T from 500 K to 100 000 K; rows = (r, g, b) linear sRGB with luminance Y = 1, and
# log10 Y (visible luminance) relative to a 6500 K blackbody. Built once at import (about 2 ms).
LOGT_MIN, LOGT_MAX, TABLE_N = np.log10(500.0), np.log10(1e5), 512


def _build_table():
    T = np.logspace(LOGT_MIN, LOGT_MAX, TABLE_N)
    X = blackbody_xyz(T)
    Y = X[:, 1:2]
    rgb = (X / Y) @ XYZ_TO_SRGB.T
    y_ref = blackbody_xyz([6500.0])[0, 1]
    return rgb, np.log10(Y[:, 0] / y_ref)


BB_RGB, BB_LOGY = _build_table()


def blackbody_rgb(T, luminance: bool = True):
    """Linear sRGB of a blackbody at T (K). luminance=True scales by its visible luminance
    (relative to 6500 K), so a hotter element is brighter as well as bluer."""
    T = np.asarray(T, dtype=float)
    u = (np.log10(np.clip(T, 10 ** LOGT_MIN, 10 ** LOGT_MAX)) - LOGT_MIN) / (LOGT_MAX - LOGT_MIN) * (TABLE_N - 1)
    i0 = np.clip(np.floor(u).astype(int), 0, TABLE_N - 2)
    f = (u - i0)[..., None]
    rgb = BB_RGB[i0] * (1 - f) + BB_RGB[i0 + 1] * f
    if luminance:
        ly = BB_LOGY[i0] * (1 - f[..., 0]) + BB_LOGY[i0 + 1] * f[..., 0]
        rgb = rgb * (10.0 ** ly)[..., None]
    return rgb


def bb_luminance(T):
    T = np.asarray(T, dtype=float)
    u = (np.log10(np.clip(T, 10 ** LOGT_MIN, 10 ** LOGT_MAX)) - LOGT_MIN) / (LOGT_MAX - LOGT_MIN) * (TABLE_N - 1)
    return 10.0 ** np.interp(u, np.arange(TABLE_N), BB_LOGY)


def aces(x):
    """Narkowicz's fit of the ACES filmic tone curve (maps HDR >= 0 to [0, 1])."""
    x = np.maximum(x, 0)
    return np.clip((x * (2.51 * x + 0.03)) / (x * (2.43 * x + 0.59) + 0.14), 0, 1)


def srgb_encode(x):
    x = np.clip(x, 0, 1)
    return np.where(x <= 0.0031308, 12.92 * x, 1.055 * np.power(x, 1 / 2.4) - 0.055)


def to_uint8(hdr, exposure: float = 1.0):
    return (srgb_encode(aces(hdr * exposure)) * 255 + 0.5).astype(np.uint8)


def bloom(hdr, strength: float = 0.08, radii=(2.0, 8.0, 24.0)):
    """Soft glow of a camera lens (cosmetic): sum of Gaussian blurs of the image, done with FFTs."""
    if strength <= 0:
        return hdr
    h, w, _ = hdr.shape
    fy = np.fft.fftfreq(h)[:, None]
    fx = np.fft.rfftfreq(w)[None, :]
    F = np.fft.rfft2(hdr, axes=(0, 1))
    out = np.zeros_like(hdr)
    for s in radii:
        s = s * w / 960.0
        k = np.exp(-2 * (np.pi * s) ** 2 * (fx * fx + fy * fy))[..., None]
        out += np.fft.irfft2(F * k, s=(h, w), axes=(0, 1))
    return hdr + strength * out / len(radii)
