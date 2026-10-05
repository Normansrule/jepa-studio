"""Numerical checks of the LeJEPA objective (docs/equations.md). Every equation in the docs
has at least one test here that checks it against an independent computation."""
import math

import numpy as np
import pytest
import torch

from jepa_studio.losses import (EppsPulley, SIGReg, ep_expected_under_null, ep_expected_under_null_exact,
                                lejepa_loss, random_directions, trapezoid_half_line)


def test_quadrature_integrates_even_function():
    # integral of exp(-t^2/2) over [-3, 3] = sqrt(2 pi) * erf(3/sqrt 2)
    t, w = trapezoid_half_line(3.0, 201)
    got = float((torch.exp(-t * t / 2) * w).sum())
    assert got == pytest.approx(math.sqrt(2 * math.pi) * math.erf(3 / math.sqrt(2)), rel=1e-5)


def test_ep_matches_brute_force_complex_integral():
    torch.manual_seed(0)
    x = torch.randn(300, dtype=torch.float64) * 1.3 + 0.2
    ep = EppsPulley(3.0, 17).double()
    ours = float(ep(x[:, None])[0])
    # brute force: N * trapezoid over [-3, 3] of |ecf - phi|^2 * exp(-t^2/2), complex arithmetic
    t = torch.linspace(-3, 3, 33, dtype=torch.float64)
    ecf = torch.exp(1j * t[None, :] * x[:, None]).mean(0)
    integrand = (ecf - torch.exp(-t * t / 2)).abs() ** 2 * torch.exp(-t * t / 2)
    brute = len(x) * float(torch.trapezoid(integrand, t))
    assert ours == pytest.approx(brute, rel=1e-6)  # buffers are stored in float32


def test_null_expectation_formula():
    # E[EP] under N(0,1) data equals the closed form, whatever N is
    assert ep_expected_under_null(3.0, 17) == pytest.approx(1.0525, abs=1e-3)
    assert ep_expected_under_null_exact() == pytest.approx(math.sqrt(2 * math.pi) - math.sqrt(2 * math.pi / 3))
    g = torch.Generator().manual_seed(1)
    ep = EppsPulley()
    vals = [float(ep(torch.randn(256, 64, generator=g)).mean()) for _ in range(20)]
    assert np.mean(vals) == pytest.approx(ep_expected_under_null(), rel=0.05)


def test_sigreg_orders_distributions():
    torch.manual_seed(0)
    s = SIGReg(512)
    gauss = float(s(torch.randn(2048, 16)))
    shifted = float(s(torch.randn(2048, 16) + 1.0))
    squashed = float(s(torch.rand(2048, 16)))
    collapsed = float(s(torch.zeros(2048, 16)))
    low_rank = float(s(torch.randn(2048, 2) @ torch.randn(2, 16) / 2))
    assert gauss < 1.5
    assert gauss < low_rank < collapsed and gauss < shifted and gauss < squashed


def test_sigreg_gradient_is_bounded_and_correct():
    torch.manual_seed(0)
    z = (torch.randn(64, 8, dtype=torch.float64) * 5).requires_grad_(True)
    a = random_directions(8, 16, torch.Generator().manual_seed(0), dtype=torch.float64)
    s = SIGReg(16).double()
    assert torch.autograd.gradcheck(lambda z: s(z, a), (z,))
    s(z, a).backward()
    # |d EP / d x_n| <= sum_k w_k * 2 * t_k * (|C-phi| + |S|) <= sum_k w_k * 2 * t_k * 3
    bound = float((s.ep.weights.double() * s.ep.t.double()).sum() * 6)
    assert float(z.grad.abs().max()) <= bound


def test_directions_are_unit_and_resampled():
    s = SIGReg(64, seed=3)
    z = torch.randn(32, 10)
    s(z)
    a1 = s.last_directions.clone()
    s(z)
    assert torch.allclose(a1.norm(dim=0), torch.ones(64), atol=1e-6)
    assert not torch.allclose(a1, s.last_directions)


def test_lejepa_loss_terms():
    torch.manual_seed(0)
    g = torch.randn(2, 32, 8)
    loc = torch.randn(3, 32, 8)
    all_ = torch.cat([g, loc])
    sig = SIGReg(32, seed=0)
    out = lejepa_loss(g, all_, sig, lam=0.05)
    center = g.mean(0)
    pred = ((center - all_) ** 2).mean()
    assert float(out.prediction) == pytest.approx(float(pred), rel=1e-6)
    # identical views => zero prediction term
    same = torch.randn(1, 32, 8).expand(4, 32, 8)
    assert float(lejepa_loss(same[:2], same, sig).prediction) == pytest.approx(0.0, abs=1e-12)
    with pytest.raises(ValueError):
        lejepa_loss(g, all_, sig, lam=1.5)


def test_one_hundred_steps_decrease_sigreg():
    """Pure optimisation of free embeddings: SIGReg alone pulls a bad cloud to N(0, I)."""
    torch.manual_seed(0)
    z = (torch.rand(512, 8) * 0.1).requires_grad_(True)
    s = SIGReg(128, seed=0)
    opt = torch.optim.Adam([z], lr=0.05)
    first = None
    for _ in range(150):
        loss = s(z)
        first = first if first is not None else float(loss.detach())
        opt.zero_grad()
        loss.backward()
        opt.step()
    assert float(s(z).detach()) < first / 20
    cov = torch.cov(z.detach().T)
    assert torch.allclose(torch.diag(cov), torch.ones(8), atol=0.35)


def test_loss_is_float32_under_bf16_autocast():
    """Regression: under autocast the projections ran in bf16 (SIGReg came out as an exact bf16
    value). The loss now disables autocast and upcasts half-precision embeddings."""
    from jepa_studio.losses import SIGReg, lejepa_loss, random_directions
    g = torch.Generator().manual_seed(3)
    z = torch.randn(4, 64, 16, generator=g)
    dirs = random_directions(16, 32, g)

    class Fixed(SIGReg):
        def forward(self, zz, directions=None):  # noqa: ARG002
            return super().forward(zz, dirs)

    ref = lejepa_loss(z[:2], z, Fixed(32), 0.05)
    with torch.autocast("cpu", dtype=torch.bfloat16):
        out = lejepa_loss(z[:2], z, Fixed(32), 0.05)
    assert out.loss.dtype == torch.float32
    assert abs(float(out.sigreg) - float(ref.sigreg)) < 1e-5
    # bf16 inputs are upcast too
    outb = lejepa_loss(z[:2].bfloat16(), z.bfloat16(), Fixed(32), 0.05)
    assert outb.loss.dtype == torch.float32
