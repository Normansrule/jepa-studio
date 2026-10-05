"""Encoders used by jepa-studio. LeJEPA needs no special architecture: any backbone plus a
small projector works (paper Sec. on architectures). These are deliberately small so the
whole pipeline runs on a laptop CPU, and scale up through config (embed_dim, depth...).

Every encoder maps a batch of one view to:
    emb  (N, embed_dim)   backbone features, used for probes, k-NN and export
    proj (N, proj_dim)    projector output, where the LeJEPA loss is applied
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F
from torch import Tensor, nn


def conv_bn(cin: int, cout: int, stride: int = 1, dims: int = 2) -> nn.Sequential:
    conv = nn.Conv2d if dims == 2 else nn.Conv1d
    bn = nn.BatchNorm2d if dims == 2 else nn.BatchNorm1d
    return nn.Sequential(conv(cin, cout, 3, stride, 1, bias=False), bn(cout), nn.GELU())


class ConvNetTiny(nn.Module):
    """3 conv stages + global average pooling. Works for any input size >= 4 px."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 128, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(in_ch, width), conv_bn(width, width * 2, 2),
            conv_bn(width * 2, width * 2), conv_bn(width * 2, width * 4, 2),
            conv_bn(width * 4, embed_dim),
        )
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x).mean(dim=(-2, -1))

    def feature_map(self, x: Tensor) -> Tensor:
        return self.features(x)


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, stride: int):
        super().__init__()
        self.a = conv_bn(cin, cout, stride)
        self.b = nn.Sequential(nn.Conv2d(cout, cout, 3, 1, 1, bias=False), nn.BatchNorm2d(cout))
        self.skip = nn.Identity() if (cin == cout and stride == 1) else nn.Sequential(
            nn.Conv2d(cin, cout, 1, stride, bias=False), nn.BatchNorm2d(cout))

    def forward(self, x: Tensor) -> Tensor:
        return F.gelu(self.b(self.a(x)) + self.skip(x))


class ConvNetSmall(nn.Module):
    """A ResNet-style encoder (4 stages, 2 blocks each at width 32..256 by default)."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 256, width: int = 32):
        super().__init__()
        chans = [width, width * 2, width * 4, width * 8]
        layers: list[nn.Module] = [conv_bn(in_ch, chans[0])]
        cin = chans[0]
        for i, c in enumerate(chans):
            layers += [ResBlock(cin, c, 1 if i == 0 else 2), ResBlock(c, c, 1)]
            cin = c
        layers.append(nn.Conv2d(cin, embed_dim, 1))
        self.features = nn.Sequential(*layers)
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x).mean(dim=(-2, -1))

    def feature_map(self, x: Tensor) -> Tensor:
        return self.features(x)


class Attention(nn.Module):
    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.heads = heads
        self.qkv = nn.Linear(dim, dim * 3)
        self.out = nn.Linear(dim, dim)
        self.last_attn: Tensor | None = None
        self.keep_attn = False

    def forward(self, x: Tensor) -> Tensor:
        n, l, d = x.shape
        q, k, v = self.qkv(x).view(n, l, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        if self.keep_attn:  # explicit path so the Inspect tab can draw attention maps
            att = (q @ k.transpose(-2, -1)) / math.sqrt(q.shape[-1])
            att = att.softmax(dim=-1)
            self.last_attn = att.detach()
            y = att @ v
        else:
            y = F.scaled_dot_product_attention(q, k, v)
        return self.out(y.transpose(1, 2).reshape(n, l, d))


class Block(nn.Module):
    def __init__(self, dim: int, heads: int, mlp_ratio: float = 4.0):
        super().__init__()
        self.n1, self.n2 = nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = Attention(dim, heads)
        self.mlp = nn.Sequential(nn.Linear(dim, int(dim * mlp_ratio)), nn.GELU(),
                                 nn.Linear(int(dim * mlp_ratio), dim))

    def forward(self, x: Tensor) -> Tensor:
        x = x + self.attn(self.n1(x))
        return x + self.mlp(self.n2(x))


class ViTTiny(nn.Module):
    """Vision Transformer with a [CLS] token. Position embeddings are stored for the
    global-view grid and bicubically resized for smaller (local) views."""

    def __init__(self, in_ch: int = 3, image_size: int = 32, patch: int = 4, embed_dim: int = 128,
                 depth: int = 4, heads: int = 4):
        super().__init__()
        if embed_dim % heads:
            raise ValueError("embed_dim must be divisible by heads")
        self.patch = patch
        self.grid = image_size // patch
        self.embed = nn.Conv2d(in_ch, embed_dim, patch, patch)
        self.cls = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos = nn.Parameter(torch.randn(1, self.grid * self.grid, embed_dim) * 0.02)
        self.blocks = nn.ModuleList([Block(embed_dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(embed_dim)
        self.embed_dim = embed_dim

    def _pos(self, gh: int, gw: int) -> Tensor:
        if gh == self.grid and gw == self.grid:
            return self.pos
        p = self.pos.reshape(1, self.grid, self.grid, -1).permute(0, 3, 1, 2)
        p = F.interpolate(p, size=(gh, gw), mode="bicubic", align_corners=False)
        return p.permute(0, 2, 3, 1).reshape(1, gh * gw, -1)

    def tokens(self, x: Tensor) -> Tensor:
        t = self.embed(x)
        gh, gw = t.shape[-2:]
        t = t.flatten(2).transpose(1, 2) + self._pos(gh, gw)
        t = torch.cat([self.cls.expand(t.shape[0], -1, -1), t], dim=1)
        for b in self.blocks:
            t = b(t)
        return self.norm(t)

    def forward(self, x: Tensor) -> Tensor:
        return self.tokens(x)[:, 0]

    def set_keep_attention(self, on: bool) -> None:
        for b in self.blocks:
            b.attn.keep_attn = on


class MLPTiny(nn.Module):
    """Flattened 16x16 input -> 2 hidden layers. The same network trains in the browser."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 64, side: int = 16):
        super().__init__()
        self.side = side
        self.net = nn.Sequential(nn.Linear(in_ch * side * side, 256), nn.GELU(),
                                 nn.Linear(256, embed_dim))
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        x = F.adaptive_avg_pool2d(x, self.side) if x.shape[-1] != self.side else x
        return self.net(x.flatten(1))


class VideoConvNet(nn.Module):
    """Per-frame ConvNetTiny, then mean over time. Input (N, T, C, H, W)."""

    def __init__(self, in_ch: int = 3, embed_dim: int = 128):
        super().__init__()
        self.frame = ConvNetTiny(in_ch, embed_dim)
        self.temporal = nn.Conv1d(embed_dim, embed_dim, 3, padding=1)
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        n, t = x.shape[:2]
        f = self.frame(x.flatten(0, 1)).view(n, t, -1).transpose(1, 2)  # (N, D, T)
        return (f + self.temporal(f)).mean(-1)


class SeriesConv(nn.Module):
    """1-D conv encoder for multichannel time series. Input (N, C, L)."""

    def __init__(self, in_ch: int = 1, embed_dim: int = 128, width: int = 32):
        super().__init__()
        self.features = nn.Sequential(
            conv_bn(in_ch, width, dims=1), conv_bn(width, width * 2, 2, dims=1),
            conv_bn(width * 2, width * 4, 2, dims=1), conv_bn(width * 4, embed_dim, 2, dims=1))
        self.embed_dim = embed_dim

    def forward(self, x: Tensor) -> Tensor:
        return self.features(x).mean(-1)


class Projector(nn.Module):
    """embed -> hidden -> hidden -> proj, with BatchNorm (as in the LeJEPA minimal example)."""

    def __init__(self, cin: int, hidden: int, cout: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(cin, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, hidden), nn.BatchNorm1d(hidden), nn.GELU(),
            nn.Linear(hidden, cout))

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class JEPAEncoder(nn.Module):
    def __init__(self, backbone: nn.Module, proj_hidden: int, proj_dim: int):
        super().__init__()
        self.backbone = backbone
        self.projector = Projector(backbone.embed_dim, proj_hidden, proj_dim)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor]:
        emb = self.backbone(x)
        return emb, self.projector(emb)


def build_encoder(cfg: dict) -> JEPAEncoder:
    m, d = cfg["model"], cfg["data"]
    ch = d.get("channels", 3)
    arch = m["arch"]
    if arch == "convnet-tiny":
        bb = ConvNetTiny(ch, m["embed_dim"])
    elif arch == "convnet-small":
        bb = ConvNetSmall(ch, m["embed_dim"])
    elif arch == "vit-tiny":
        bb = ViTTiny(ch, d["image_size"], m.get("patch_size", 4), m["embed_dim"], m.get("depth", 4),
                     m.get("heads", 4))
    elif arch == "mlp-tiny":
        bb = MLPTiny(ch, m["embed_dim"])
    elif arch == "video-convnet":
        bb = VideoConvNet(ch, m["embed_dim"])
    elif arch == "series-conv":
        bb = SeriesConv(ch, m["embed_dim"])
    else:
        raise ValueError(f"unknown arch {arch}")
    return JEPAEncoder(bb, m.get("proj_hidden", 256), m["proj_dim"])


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())
