"""Per-image explanation maps for the Inspect tab (desktop tier; docs/api.md, `/saliency`).

saliency(model, x)
    The same quantity the web engine draws (site/js/engine/mlp.js `saliency`):

        S[h, w] = max over channels c of | d ||emb(x)||_2 / d x[c, h, w] |

    where emb is the *backbone* embedding (the projector is not involved) and x is the input
    exactly as the encoder sees it. The map is then divided by its maximum, so it lies in
    [0, 1], the most influential pixel is 1 and a pixel that cannot change the embedding is 0.
    (The web engine scales its map by a constant instead; the Inspect overlay min-max
    normalises whatever it receives, so both tiers draw the same picture.) The model is put in
    eval mode, so BatchNorm uses running statistics and every sample is independent.

vit_attention(model, x)
    vit-tiny only: the last transformer block's attention from the [CLS] token (the token whose
    final state *is* the embedding) to every patch token, averaged over heads, laid out on the
    patch grid and divided by its maximum. Each head's softmax row sums to 1 over all tokens
    (CLS + patches); `attention_weights` returns those full rows. The weights are read with a
    forward hook on the last block's Attention module, which recomputes softmax(q k^T / sqrt(d))
    from the module's own input and weights: the model's outputs, code path and saved weight
    names are untouched.

Both raise MapError (a ValueError, so the server answers 400) for encoders that do not take
images: series-conv (N, C, L) and video-convnet (N, T, C, H, W).
"""
from __future__ import annotations

import math
import threading

import torch
from torch import Tensor

from .models import SeriesConv, VideoConvNet, ViTTiny


class MapError(ValueError):
    """The model or input cannot produce this map."""


def _backbone(model):
    return getattr(model, "backbone", model)


def _image_batch(model, x: Tensor) -> tuple[Tensor, bool]:
    bb = _backbone(model)
    if isinstance(bb, SeriesConv):
        raise MapError("saliency and attention maps are for image encoders; series-conv reads time series")
    if isinstance(bb, VideoConvNet):
        raise MapError("saliency and attention maps are for image encoders; video-convnet reads clips")
    if x.dim() == 3:
        return x.unsqueeze(0), True
    if x.dim() == 4:
        return x, False
    raise MapError(f"expected an image (C, H, W) or a batch (N, C, H, W), got shape {tuple(x.shape)}")


def _unit_max(m: Tensor) -> Tensor:
    """Divide each map (last two dims) by its maximum; an all-zero map stays zero."""
    peak = m.flatten(-2).amax(-1).clamp_min(1e-12)
    return m / peak[..., None, None]


def saliency(model, x: Tensor) -> Tensor:
    """|d ||emb|| / d pixel|, max over channels, scaled to [0, 1]. (H, W) for one image, (N, H, W) for a batch."""
    xb, single = _image_batch(model, x)
    bb = _backbone(model)
    was = model.training
    model.eval()
    try:
        with torch.enable_grad():
            xg = xb.detach().to(next(model.parameters()).device, torch.float32).clone().requires_grad_(True)
            e = bb(xg)
            # sum of per-sample norms: in eval mode samples are independent, so each sample's
            # gradient is its own. clamp avoids the 0/0 of d||e||/de at e = 0 (as the web engine does)
            norm = e.pow(2).sum(dim=1).clamp_min(1e-24).sqrt().sum()
            (g,) = torch.autograd.grad(norm, xg)  # parameter .grad buffers are left alone
    finally:
        model.train(was)
    m = _unit_max(g.detach().abs().amax(dim=1).float().cpu())
    return m[0] if single else m


_hook_lock = threading.Lock()


def attention_weights(model, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
    """Last-block attention of a vit-tiny encoder: (N, heads, 1 + P, 1 + P) and the (gh, gw) patch grid.

    Rows are softmax distributions (each sums to 1). Token 0 is [CLS]; tokens 1.. are patches in
    row-major order over the grid."""
    xb, _ = _image_batch(model, x)
    bb = _backbone(model)
    if not isinstance(bb, ViTTiny):
        raise MapError("attention maps need a vit-tiny encoder")
    attn = bb.blocks[-1].attn
    me = threading.get_ident()
    got: list[Tensor] = []

    def hook(mod, inputs, _output):
        if threading.get_ident() != me:  # another request's forward pass on the same model
            return
        h = inputs[0]
        n, length, d = h.shape
        q, k, _v = mod.qkv(h).view(n, length, 3, mod.heads, d // mod.heads).permute(2, 0, 3, 1, 4)
        got.append(((q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])).softmax(dim=-1).detach())

    was = model.training
    model.eval()
    with _hook_lock:
        handle = attn.register_forward_hook(hook)
    try:
        with torch.no_grad():
            xd = xb.to(next(model.parameters()).device, torch.float32)
            bb(xd)
            gh, gw = bb.embed(xd[:1]).shape[-2:]
    finally:
        handle.remove()
        model.train(was)
    return got[-1].float().cpu(), (int(gh), int(gw))


def vit_attention(model, x: Tensor) -> tuple[Tensor, tuple[int, int]]:
    """[CLS] -> patch attention of the last block, averaged over heads, on the patch grid, scaled to [0, 1].

    Returns ((gh, gw) map for one image or (N, gh, gw) for a batch, (gh, gw))."""
    att, (gh, gw) = attention_weights(model, x)
    m = _unit_max(att[:, :, 0, 1:].mean(dim=1).reshape(-1, gh, gw))
    return (m[0] if x.dim() == 3 else m), (gh, gw)
