"""Built-in synthetic dataset: procedurally drawn shapes (no downloads, fully reproducible).

The same generator exists in JavaScript (site/js/shapes.js). Both use the mulberry32 PRNG
and per-pixel inside-tests, so image i with seed s is identical (to float rounding) in
Python and in the browser; tests/test_web_engine.py checks this with Node
(tests/js/shapes_dump.mjs).

Labels exist only so the Evaluate tab can run a linear probe / k-NN on a small labeled
subset. Pretraining never sees them.
"""
from __future__ import annotations

import math

import numpy as np

CLASSES = ["circle", "square", "triangle", "cross", "ring", "diamond", "flower", "stripes",
           "checker", "crescent"]


class Mulberry32:
    """32-bit PRNG with a tiny state, easy to reproduce exactly in JavaScript."""

    def __init__(self, seed: int):
        self.a = seed & 0xFFFFFFFF

    def next(self) -> float:
        self.a = (self.a + 0x6D2B79F5) & 0xFFFFFFFF
        t = self.a
        t = ((t ^ (t >> 15)) * (t | 1)) & 0xFFFFFFFF
        t ^= (t + (((t ^ (t >> 7)) * (t | 61)) & 0xFFFFFFFF)) & 0xFFFFFFFF
        t &= 0xFFFFFFFF
        return ((t ^ (t >> 14)) & 0xFFFFFFFF) / 4294967296.0

    def uniform(self, lo: float, hi: float) -> float:
        return lo + (hi - lo) * self.next()


def item_seed(seed: int, index: int) -> int:
    """Independent stream per image, so item i does not depend on items before it."""
    return (seed * 2654435761 + index * 40503 + 12345) & 0xFFFFFFFF


def _inside(cls: int, u: np.ndarray, v: np.ndarray, r: float, phase: float) -> np.ndarray:
    """u, v: pixel coordinates in the shape's rotated frame, in units of the canvas."""
    ax, ay = np.abs(u), np.abs(v)
    rad = np.sqrt(u * u + v * v)
    if cls == 0:
        return rad < r
    if cls == 1:
        return np.maximum(ax, ay) < r * 0.85
    if cls == 2:  # triangle: flat edge at v = 0.6 r, apex at v = -0.9 r
        return (v < r * 0.6) & (v > -r * 0.9 + 1.9 * ax)
    if cls == 3:
        w = r * 0.32
        return ((ax < w) & (ay < r)) | ((ay < w) & (ax < r))
    if cls == 4:
        return (rad < r) & (rad > r * 0.55)
    if cls == 5:
        return ax + ay < r
    if cls == 6:
        th = np.arctan2(v, u)
        return rad < r * (0.6 + 0.4 * np.cos(5 * th))
    if cls == 7:
        return (np.maximum(ax, ay) < r) & (np.sin(u * 40.0 + phase) > 0)
    if cls == 8:
        return (np.maximum(ax, ay) < r) & ((np.sin(u * 30.0 + phase) * np.sin(v * 30.0 + phase)) > 0)
    if cls == 9:
        du = u - r * 0.45
        return (rad < r) & (np.sqrt(du * du + v * v) > r * 0.8)
    raise ValueError(cls)


def render(seed: int, index: int, size: int = 32) -> tuple[np.ndarray, int]:
    """One image as float32 array (size, size, 3) in [0, 1], and its class label."""
    rng = Mulberry32(item_seed(seed, index))
    cls = int(rng.next() * len(CLASSES)) % len(CLASSES)
    bg = np.array([rng.uniform(0.0, 0.45) for _ in range(3)], dtype=np.float64)
    fg = np.array([rng.uniform(0.55, 1.0) for _ in range(3)], dtype=np.float64)
    if rng.next() < 0.5:  # sometimes dark-on-light
        bg, fg = 1.0 - bg, 1.0 - fg
    cx, cy = rng.uniform(-0.18, 0.18), rng.uniform(-0.18, 0.18)
    r = rng.uniform(0.2, 0.34)
    ang = rng.uniform(0.0, 2 * math.pi)
    phase = rng.uniform(0.0, 2 * math.pi)
    noise_amp = rng.uniform(0.0, 0.08)
    noise_seed = int(rng.next() * 4294967295)

    coords = (np.arange(size, dtype=np.float64) + 0.5) / size - 0.5
    x, y = np.meshgrid(coords, coords)  # x: columns, y: rows (down)
    px, py = x - cx, -(y - cy)
    ca, sa = math.cos(ang), math.sin(ang)
    u = ca * px + sa * py
    v = -sa * px + ca * py
    mask = _inside(cls, u, v, r, phase).astype(np.float64)

    nrng = Mulberry32(noise_seed)
    noise = np.array([nrng.next() for _ in range(size * size)], dtype=np.float64).reshape(size, size)
    img = bg[None, None, :] * (1 - mask[..., None]) + fg[None, None, :] * mask[..., None]
    img = img + (noise[..., None] - 0.5) * 2 * noise_amp
    return np.clip(img, 0.0, 1.0).astype(np.float32), cls


def make_dataset(n: int, seed: int = 0, size: int = 32) -> tuple[np.ndarray, np.ndarray]:
    imgs = np.empty((n, size, size, 3), dtype=np.float32)
    labels = np.empty(n, dtype=np.int64)
    for i in range(n):
        imgs[i], labels[i] = render(seed, i, size)
    return imgs, labels
