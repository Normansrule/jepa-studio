"""LeJEPA objective: prediction term + Sketched Isotropic Gaussian Regularization (SIGReg).

The equations are from Balestriero & LeCun, "LeJEPA: Provable and Scalable Self-Supervised
Learning Without the Heuristics", arXiv:2511.08544 (Definition 2, Algorithm 2).

Provenance, stated plainly: the numerical recipe for the Epps-Pulley integral used here
(integrate the even integrand on the half line [0, t_max] with doubled trapezoid weights,
t_max = 3, 17 knots, scale by N) is the one in the authors' reference implementation
(github.com/rbalestr-lab/lejepa, CC BY-NC 4.0), not the paper's Algorithm 1, which uses the
full line t in [-5, 5]. This module is our own code, but it was written with that repository
at hand and its EppsPulley class closely parallels the reference SIGReg class. If you need a
licence-clean implementation for commercial use, have it reviewed or re-derived independently.
Known deviation from Algorithm 2: slice directions are drawn per view, not shared by all views
of a step (docs/equations.md, section 2).

Notation used everywhere in this repo (see docs/equations.md):
    N  batch size (samples)            K  embedding (projector) dimension
    V  number of views, Vg of them global
    z  embedding of one view, shape (N, K)
    a  a unit-norm random direction in R^K ("slice"), M of them per step
    t  frequency at which characteristic functions are compared (dimensionless)
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
from torch import Tensor, nn


def trapezoid_half_line(t_max: float, knots: int) -> tuple[Tensor, Tensor]:
    """Nodes t_k in [0, t_max] and weights for integrating an EVEN function over [-t_max, t_max].

    Because |phi_hat(t) - phi(t)|^2 is even in t (the ECF of real data satisfies
    phi_hat(-t) = conj(phi_hat(t))), the integral over [-t_max, t_max] equals twice the
    integral over [0, t_max]. Trapezoid weights on [0, t_max] are dt/2 at the ends and dt
    inside; doubling gives dt at the ends and 2*dt inside.
    """
    if knots < 2:
        raise ValueError("knots must be >= 2")
    t = torch.linspace(0.0, t_max, knots, dtype=torch.float64)
    dt = t_max / (knots - 1)
    w = torch.full((knots,), 2.0 * dt, dtype=torch.float64)
    w[0] = dt
    w[-1] = dt
    return t, w


class EppsPulley(nn.Module):
    """Epps-Pulley goodness-of-fit statistic against N(0, 1).

        EP(x_1..x_N) = N * integral |phi_hat(t) - phi(t)|^2 w(t) dt
        phi_hat(t)   = (1/N) sum_n exp(i t x_n)        (empirical characteristic function)
        phi(t)       = exp(-t^2 / 2)                   (standard-normal characteristic function)
        w(t)         = exp(-t^2 / 2)                   (Gaussian weight, sigma = 1)

    Expanding |.|^2 with exp(i t x) = cos(t x) + i sin(t x) and phi real:
        |phi_hat - phi|^2 = (mean cos(t x) - phi(t))^2 + (mean sin(t x))^2
    which is what forward() computes. Each term is bounded (|cos|,|sin| <= 1), so the
    statistic and its gradient are bounded for any input, which is the paper's stability
    argument. Cost is O(N * knots) per slice: linear in batch size.
    """

    def __init__(self, t_max: float = 3.0, knots: int = 17):
        super().__init__()
        t, w = trapezoid_half_line(t_max, knots)
        phi = torch.exp(-0.5 * t * t)
        self.register_buffer("t", t.float(), persistent=False)
        self.register_buffer("phi", phi.float(), persistent=False)
        self.register_buffer("weights", (w * phi).float(), persistent=False)  # quadrature * w(t)

    def per_frequency_error(self, x: Tensor) -> Tensor:
        """x: (..., N, M) projected samples -> (..., M, knots) squared ECF error."""
        t, phi = self.t.to(x.dtype), self.phi.to(x.dtype)
        xt = x.unsqueeze(-1) * t  # (..., N, M, knots)
        cos_mean = torch.cos(xt).mean(dim=-3)
        sin_mean = torch.sin(xt).mean(dim=-3)
        return (cos_mean - phi) ** 2 + sin_mean ** 2

    def forward(self, x: Tensor) -> Tensor:
        """x: (..., N, M) -> statistic per slice, shape (..., M)."""
        n = x.shape[-2]
        return (self.per_frequency_error(x) @ self.weights.to(x.dtype)) * n


def random_directions(dim: int, num_slices: int, generator: torch.Generator | None = None,
                      device=None, dtype=torch.float32) -> Tensor:
    """M directions drawn uniformly on the unit sphere in R^dim, as columns of a (dim, M) matrix."""
    a = torch.randn(dim, num_slices, generator=generator, dtype=torch.float32)
    a = a / a.norm(dim=0, keepdim=True).clamp_min(1e-12)
    return a.to(device=device, dtype=dtype)


class SIGReg(nn.Module):
    """Sketched Isotropic Gaussian Regularization (paper Definition 2).

        SIGReg(Z) = (1/M) sum_m EP({a_m^T z_n}_n),   a_m ~ Uniform(unit sphere)

    By the Cramer-Wold theorem a distribution is determined by all its 1-D projections, so
    driving every projection toward N(0,1) drives Z toward the isotropic Gaussian N(0, I).
    Directions are re-drawn every call ("sketching"), so over training all directions get
    tested even though each step only looks at M of them.
    """

    def __init__(self, num_slices: int = 256, t_max: float = 3.0, knots: int = 17, seed: int = 0):
        super().__init__()
        self.num_slices = num_slices
        self.ep = EppsPulley(t_max, knots)
        self.generator = torch.Generator().manual_seed(seed)
        self.last_directions: Tensor | None = None

    def forward(self, z: Tensor, directions: Tensor | None = None) -> Tensor:
        """z: (N, K) or (V, N, K). Returns the mean statistic over slices (and views)."""
        if directions is None:
            directions = random_directions(z.shape[-1], self.num_slices, self.generator,
                                           device=z.device, dtype=z.dtype)
        self.last_directions = directions
        proj = z @ directions  # (..., N, M)
        if proj.dtype in (torch.float16, torch.bfloat16):
            proj = proj.float()  # cos/sin sums are accuracy-sensitive: never do them in half precision
        return self.ep(proj).mean()


@dataclass
class LeJEPAOutput:
    loss: Tensor
    prediction: Tensor
    sigreg: Tensor


def lejepa_loss(global_emb: Tensor, all_emb: Tensor, sigreg: SIGReg, lam: float = 0.05) -> LeJEPAOutput:
    """LeJEPA objective (paper Algorithm 2).

    global_emb: (Vg, N, K) embeddings of the global views.
    all_emb:    (V,  N, K) embeddings of all views (global + local).

        mu_n      = (1/Vg) sum_{g} z_{g,n}                    "center" of the global views
        L_pred    = mean_{v,n,k} (mu_{n,k} - z_{v,n,k})^2      every view predicts the center
        L_sigreg  = (1/V) sum_v SIGReg(z_v)                    each view's batch -> N(0, I)
        L         = (1 - lambda) L_pred + lambda L_sigreg

    The predictor here is the identity map in embedding space (the paper's default for
    image pretraining); the world model in jepa_studio.world adds a learned predictor.
    """
    if not 0.0 <= lam <= 1.0:
        raise ValueError("lambda must be in [0, 1]")
    # The loss is always computed in at least float32, also under bf16/fp16 autocast: the
    # projections z @ a and the quadrature sums lose too much precision in half floats.
    with torch.autocast(device_type=all_emb.device.type, enabled=False):
        if global_emb.dtype in (torch.float16, torch.bfloat16):
            global_emb, all_emb = global_emb.float(), all_emb.float()
        centers = global_emb.mean(dim=0)
        pred = (centers.unsqueeze(0) - all_emb).pow(2).mean()
        sig = torch.stack([sigreg(v) for v in all_emb]).mean()
    return LeJEPAOutput((1.0 - lam) * pred + lam * sig, pred.detach(), sig.detach())


# ---------------------------------------------------------------- closed forms for tests/docs

def ep_expected_under_null(t_max: float = 3.0, knots: int = 17) -> float:
    """E[EP] when x_1..x_N ~ N(0,1) i.i.d. (any N).

    For one t: E|phi_hat - phi|^2 = Var(e^{itx})/N = (1 - phi(t)^2)/N, so
    E[EP] = integral (1 - exp(-t^2)) exp(-t^2/2) dt over [-t_max, t_max] (the N cancels).
    Uses the same quadrature as EppsPulley so tests compare like with like.
    """
    t, w = trapezoid_half_line(t_max, knots)
    return float(((1 - torch.exp(-t * t)) * torch.exp(-0.5 * t * t) * w).sum())


def ep_expected_under_null_exact() -> float:
    """Exact integral over the whole real line: sqrt(2 pi) - sqrt(2 pi / 3)."""
    return math.sqrt(2 * math.pi) - math.sqrt(2 * math.pi / 3)
