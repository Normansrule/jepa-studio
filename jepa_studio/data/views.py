"""Augmentation 'views' for JEPA pretraining: global and local crops (multi-crop), photometric
jitter, and optional block masks. Global views see 30-100% of the input at full resolution,
local views see 5-30% at low resolution (LeJEPA README defaults, scaled to small images).
"""
from __future__ import annotations

import math
import random

import torch
import torch.nn.functional as F
from torch import Tensor


def _crop_box(h: int, w: int, scale: tuple[float, float], rng: random.Random,
              ratio: tuple[float, float] = (3 / 4, 4 / 3)) -> tuple[int, int, int, int]:
    area = h * w
    for _ in range(10):
        target = area * rng.uniform(*scale)
        logr = (math.log(ratio[0]), math.log(ratio[1]))
        ar = math.exp(rng.uniform(*logr))
        cw = int(round(math.sqrt(target * ar)))
        ch = int(round(math.sqrt(target / ar)))
        if 0 < cw <= w and 0 < ch <= h:
            return rng.randint(0, h - ch), rng.randint(0, w - cw), ch, cw
    s = min(h, w)
    return (h - s) // 2, (w - s) // 2, s, s


def resized_crop(img: Tensor, box: tuple[int, int, int, int], size: int) -> Tensor:
    """img: (..., C, H, W). Bilinear resize of the crop to (size, size)."""
    top, left, ch, cw = box
    crop = img[..., top:top + ch, left:left + cw]
    lead = crop.shape[:-3]
    flat = crop.reshape(-1, *crop.shape[-3:])
    out = F.interpolate(flat, size=(size, size), mode="bilinear", align_corners=False, antialias=True)
    return out.reshape(*lead, *out.shape[-3:])


def color_jitter(img: Tensor, strength: float, rng: random.Random) -> Tensor:
    """Brightness / contrast / saturation jitter with factors in [1-s, 1+s]."""
    if strength <= 0:
        return img
    b = rng.uniform(1 - strength, 1 + strength)
    c = rng.uniform(1 - strength, 1 + strength)
    s = rng.uniform(1 - strength / 2, 1 + strength / 2)
    img = img * b
    mean = img.mean(dim=(-3, -2, -1), keepdim=True)
    img = (img - mean) * c + mean
    if img.shape[-3] == 3:
        gray = (0.299 * img[..., 0:1, :, :] + 0.587 * img[..., 1:2, :, :] + 0.114 * img[..., 2:3, :, :])
        img = (img - gray) * s + gray
    return img.clamp(0, 1)


def to_gray(img: Tensor) -> Tensor:
    if img.shape[-3] != 3:
        return img
    g = 0.299 * img[..., 0:1, :, :] + 0.587 * img[..., 1:2, :, :] + 0.114 * img[..., 2:3, :, :]
    return g.expand_as(img).clone()


def blur(img: Tensor, sigma: float) -> Tensor:
    k = max(3, int(2 * round(2 * sigma) + 1))
    x = torch.arange(k, dtype=img.dtype) - k // 2
    g = torch.exp(-0.5 * (x / sigma) ** 2)
    g = (g / g.sum()).to(img.device)
    c = img.shape[-3]
    flat = img.reshape(-1, c, *img.shape[-2:])
    flat = F.conv2d(F.pad(flat, (k // 2,) * 4, mode="reflect"), g.view(1, 1, 1, k).repeat(c, 1, 1, 1), groups=c)
    flat = F.conv2d(flat, g.view(1, 1, k, 1).repeat(c, 1, 1, 1), groups=c)
    return flat.reshape(img.shape)


def block_mask(img: Tensor, ratio: float, rng: random.Random, patch: int = 4) -> Tensor:
    """Zero out a random set of patch-aligned blocks covering ~ratio of the view."""
    if ratio <= 0:
        return img
    h, w = img.shape[-2:]
    gh, gw = max(1, h // patch), max(1, w // patch)
    n = int(round(ratio * gh * gw))
    idx = rng.sample(range(gh * gw), n)
    out = img.clone()
    for i in idx:
        r, c = divmod(i, gw)
        out[..., r * patch:(r + 1) * patch, c * patch:(c + 1) * patch] = 0.5
    return out


class ImageViews:
    """Callable producing (global_views, local_views) from one (C, H, W) float tensor in [0,1].
    Also works for clips shaped (T, C, H, W): the same crop is used for every frame."""

    def __init__(self, dcfg: dict, seed: int | None = None):
        self.d = dcfg
        self.rng = random.Random(seed)

    def one(self, img: Tensor, size: int, scale) -> Tensor:
        d, rng = self.d, self.rng
        h, w = img.shape[-2:]
        v = resized_crop(img, _crop_box(h, w, tuple(scale), rng), size)
        if rng.random() < d.get("flip_p", 0.5):
            v = v.flip(-1)
        if rng.random() < 0.8:
            v = color_jitter(v, d.get("color_jitter", 0.4), rng)
        if rng.random() < d.get("grayscale_p", 0.2):
            v = to_gray(v)
        if rng.random() < d.get("blur_p", 0.0):
            v = blur(v, rng.uniform(0.1, 1.0 + size / 64))
        v = block_mask(v, d.get("mask_ratio", 0.0), rng)
        return v

    def __call__(self, img: Tensor) -> tuple[list[Tensor], list[Tensor]]:
        d = self.d
        g = [self.one(img, d["image_size"], d.get("global_scale", (0.3, 1.0))) for _ in range(d["n_global"])]
        loc = [self.one(img, d["local_size"], d.get("local_scale", (0.05, 0.3))) for _ in range(d["n_local"])]
        return g, loc


class SeriesViews:
    """Views of a (C, L) window: global = the whole window, local = a 30-60% sub-window
    resampled to L/2, both with amplitude scaling, jitter noise and optional masking."""

    def __init__(self, dcfg: dict, seed: int | None = None):
        self.d = dcfg
        self.rng = random.Random(seed)

    def one(self, x: Tensor, out_len: int, frac: tuple[float, float]) -> Tensor:
        rng = self.rng
        c, n = x.shape
        ln = max(4, int(n * rng.uniform(*frac)))
        s = rng.randint(0, n - ln)
        v = F.interpolate(x[None, :, s:s + ln], size=out_len, mode="linear", align_corners=False)[0]
        v = v * rng.uniform(0.8, 1.2) + torch.randn(v.shape, generator=None) * 0.03 * rng.random()
        m = self.d.get("mask_ratio", 0.0)
        if m > 0:
            ml = int(out_len * m)
            ms = rng.randint(0, out_len - ml)
            v[:, ms:ms + ml] = 0
        return v

    def __call__(self, x: Tensor):
        n = x.shape[-1]
        g = [self.one(x, n, (0.8, 1.0)) for _ in range(self.d["n_global"])]
        loc = [self.one(x, max(8, n // 2), (0.3, 0.6)) for _ in range(self.d["n_local"])]
        return g, loc
