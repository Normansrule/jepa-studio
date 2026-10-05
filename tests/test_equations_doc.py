"""Recompute every worked-example number printed in docs/equations.md.

Each check does two things: the printed string must appear in the doc, and the value computed
with the repo's functions (or by hand, where the doc derives it by hand) must round to it at
the printed precision. Change the code or the doc without the other and this fails.
"""
from __future__ import annotations

import math
from pathlib import Path

import pytest
import torch
import torch.nn.functional as F

from jepa_studio.config import DEFAULT_CONFIG
from jepa_studio.evaluate import collapse_check, isotropy_report, knn_accuracy, standardize
from jepa_studio.losses import (EppsPulley, SIGReg, ep_expected_under_null, ep_expected_under_null_exact,
                                lejepa_loss, trapezoid_half_line)
from jepa_studio.world.planner import CEMConfig, trajectory_cost

DOC = (Path(__file__).resolve().parent.parent / "docs" / "equations.md").read_text(encoding="utf-8")


def check(value: float, printed: str) -> None:
    """`printed` appears in the doc and `value` rounds to it (half a unit in the last digit)."""
    assert printed in DOC, f"{printed!r} not found in docs/equations.md"
    s = printed.replace("−", "-").replace(",", "").replace(" ", "").replace(" ", "").replace("\\,", "")
    decimals = len(s.split(".")[1]) if "." in s else 0
    tol = 0.5 * 10 ** (-decimals) + 1e-9
    assert abs(float(value) - float(s)) <= tol, f"doc says {printed}, code gives {value}"


def T(x, dtype=torch.float64):
    return torch.tensor(x, dtype=dtype)


# ------------------------------------------------------------------------ section 1: objective

def test_defaults_quoted_in_doc():
    ob, d, m = DEFAULT_CONFIG["objective"], DEFAULT_CONFIG["data"], DEFAULT_CONFIG["model"]
    assert (ob["lambda"], ob["num_slices"], ob["t_max"], ob["knots"]) == (0.05, 256, 3.0, 17)
    assert (d["n_global"], d["image_size"], d["n_local"], d["local_size"], m["proj_dim"]) == (2, 32, 4, 16, 16)
    assert "$\\lambda = 0.05$, $M = 256$, $t_{\\max} = 3.0$, 17 knots" in DOC
    check(d["n_global"] + d["n_local"], "6")


A = T([[1.0, 0.6], [0.0, 0.8]])  # columns a1 = (1, 0), a2 = (0.6, 0.8)
G1 = T([[1.0, 0.0], [-1.0, 0.5]])
G2 = T([[0.6, 0.2], [-0.8, 0.1]])
L1 = T([[0.9, -0.1], [-1.2, 0.4]])


def test_lejepa_worked_example():
    sig = SIGReg(2).double()
    glob, all_v = torch.stack([G1, G2]), torch.stack([G1, G2, L1])
    centers = glob.mean(0)
    assert torch.allclose(centers, T([[0.8, 0.1], [-0.9, 0.3]]))
    sq = (centers - all_v).pow(2).sum(-1)  # per view, per sample
    assert torch.allclose(sq, T([[0.05, 0.05], [0.05, 0.05], [0.05, 0.10]]))
    check(float(sq.sum()), "0.35")
    check(all_v.numel(), "12")
    out = lejepa_loss(glob, all_v, lambda v: sig(v, A), lam=0.05)
    check(out.prediction, "0.029167")
    per_view = [float(sig(v, A)) for v in all_v]
    for v, p in zip(per_view, ["0.3382", "0.1993", "0.3305"]):
        check(v, p)
    check(out.sigreg, "0.2893")
    check(out.loss, "0.042174")
    check(0.95 * 0.029167 + 0.05 * 0.2893, "0.042174")  # the hand arithmetic as printed


# ------------------------------------------------------------------------ section 2: SIGReg

def test_quadrature_numbers():
    t, w = trapezoid_half_line(3.0, 17)
    check(float(t[1] - t[0]), "0.1875")
    check(float(t[2]), "0.375")
    assert float(t[-1]) == 3.0
    assert float(w[0]) == float(w[-1]) == 0.1875 and torch.all(w[1:-1] == 0.375)
    check(float(w.sum()), "6")
    ep = EppsPulley(3.0, 17)
    check(ep.weights[0], "0.1875")
    check(ep.weights[1], "0.3685")
    check(ep.weights[16], "0.002083")
    check(0.375 * math.exp(-0.1875 ** 2 / 2), "0.3685")
    check(0.1875 * math.exp(-4.5), "0.002083")


def test_direction_normalisation():
    g = torch.tensor([3.0, 4.0])
    a = g / g.norm()
    check(a[0], "0.6")
    check(a[1], "0.8")


def test_one_slice_example():
    x = T([-1.0, 0.0, 1.0, 2.0])
    ep = EppsPulley(3.0, 17).double()
    err = ep.per_frequency_error(x[:, None])[0]
    assert float(err[0]) == pytest.approx(0.0, abs=1e-12)
    # t = 0.75 is knot 4, t = 1.5 is knot 8
    assert float(ep.t[4]) == 0.75 and float(ep.t[8]) == 1.5
    for k, t, (c, s, p, e) in [(4, 0.75, ("0.6335", "0.2494", "0.7548", "0.0769")),
                               (8, 1.5, ("0.0379", "0.0353", "0.3247", "0.0835"))]:
        check(torch.cos(t * x).mean(), c)
        check(torch.sin(t * x).mean(), s)
        check(math.exp(-t * t / 2), p)
        check(err[k], e)
        # by hand: (mean cos - phi)^2 + (mean sin)^2
        check((torch.cos(t * x).mean() - math.exp(-t * t / 2)) ** 2 + torch.sin(t * x).mean() ** 2, e)
    check(ep(x[:, None])[0], "0.5582")
    check(EppsPulley()(x.float()[:, None])[0], "0.5582")  # default float32 module agrees
    # the paper's Algorithm 1 rule (full line, t in [-5, 5], 17 points, trapezoid) for comparison
    tt = torch.linspace(-5, 5, 17, dtype=torch.float64)
    dt = 10 / 16
    wq = torch.full((17,), dt, dtype=torch.float64)
    wq[0] = wq[-1] = dt / 2
    xd = x.double()
    errf = (torch.cos(tt[:, None] * xd).mean(1) - torch.exp(-tt ** 2 / 2)) ** 2 + torch.sin(tt[:, None] * xd).mean(1) ** 2
    full = float(len(xd) * (wq * torch.exp(-tt ** 2 / 2) * errf).sum())
    check(full, "0.55856")
    check(float(ep(x[:, None])[0]), "0.55819")


def test_collapse_scales_with_n():
    ep = EppsPulley()
    t = ep.t.double()
    per_sample = float(((1 - torch.exp(-t * t / 2)) ** 2 * ep.weights.double()).sum())
    check(per_sample, "0.40205")
    check(ep(torch.zeros(4, 1))[0], "1.6082")
    check(4 * per_sample, "1.6082")
    check(ep(torch.zeros(128, 1))[0], "51.46")


def test_null_expectation():
    check(ep_expected_under_null(3.0, 17), "1.0525")
    check(ep_expected_under_null_exact(), "1.0594")
    check(math.sqrt(2 * math.pi) - math.sqrt(2 * math.pi / 3), "1.0594")


def test_sigreg_example():
    z = T([[1.0, 0.5], [-1.0, 1.0], [0.5, -1.5], [-0.5, 0.0]])
    proj = z @ A
    assert torch.allclose(proj[:, 0], T([1.0, -1.0, 0.5, -0.5]))
    assert torch.allclose(proj[:, 1], T([1.0, 0.2, -0.9, -0.3]))
    sig = SIGReg(2).double()
    per = sig.ep(proj)
    check(per[0], "0.0619")
    check(per[1], "0.1547")
    check(sig(z, A), "0.1083")


# ------------------------------------------------------------------------ section 3: isotropy

def test_why_isotropic_numbers():
    a = T([0.6, 0.8])
    sigma = torch.diag(T([1.0, 0.01]))
    check(a @ sigma @ a, "0.3664")
    check(1 / 1.9 + 1 / 0.1, "10.53")
    check(1 / 1.0 + 1 / 1.0, "2")


# ------------------------------------------------------------------------ section 4: evaluation

def test_standardize_example():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 9.0]])
    check(x.std(0)[1], "3.6056")
    s = standardize(x)[0]
    check(s[2, 0], "1.0000")
    check(s[2, 1], "1.1094")
    check(s[0, 0], "-1.0000")
    check(s[0, 1], "-0.8321")


def test_knn_example():
    xtr = torch.tensor([[1.0, 0.0], [0.8, 0.6], [0.0, 1.0], [-1.0, 0.0]])
    ytr = torch.tensor([0, 1, 1, 2])
    q = torch.tensor([[1.0, 0.2]])
    sims = (F.normalize(q, dim=1) @ F.normalize(xtr, dim=1).T)[0]
    for v, p in zip(sims, ["0.9806", "0.9021", "0.1961", "−0.9806"]):
        check(v, p)
    assert knn_accuracy(xtr, ytr, q, torch.tensor([1]), k=3)["accuracy"] == 1.0  # majority -> class 1
    assert knn_accuracy(xtr, ytr, q, torch.tensor([0]), k=2)["accuracy"] == 1.0  # tie -> nearest (class 0)
    assert knn_accuracy(xtr, ytr, q, torch.tensor([1]), k=2)["accuracy"] == 0.0


def _hadamard_cloud() -> torch.Tensor:
    h2 = torch.tensor([[1.0, 1.0], [1.0, -1.0]])
    h = torch.kron(torch.kron(h2, h2), h2)  # 8 x 8 Sylvester Hadamard
    scale = torch.tensor([math.sqrt(7 / 4), math.sqrt(7 / 8), math.sqrt(7 / 16), math.sqrt(7 / 16)])
    return h[:, 1:5] * scale


def test_effective_rank_example():
    z = _hadamard_cloud()
    rep = isotropy_report(z, num_slices=64)
    assert rep["eigenvalues"] == pytest.approx([2.0, 1.0, 0.5, 0.5], abs=1e-5)
    p = T([0.5, 0.25, 0.125, 0.125])
    ent = float(-(p * p.log()).sum())
    check(ent, "1.2130")
    check(rep["effective_rank"], "3.364")
    check(math.exp(ent), "3.364")
    check(rep["isotropy_ratio"], "0.25")
    check(rep["top_eigen_share"], "0.5")
    check(rep["mean_std"], "0.9571")


def _report_from_spectrum(lam):
    """isotropy_report-shaped dict for a given spectrum (mean_std irrelevant here: non-zero)."""
    lam = T(lam)
    p = lam / lam.sum()
    return {"dim": len(lam), "mean_std": 1.0, "effective_rank": float((-(p * p.log()).sum()).exp()),
            "top_eigen_share": float(p[0]), "isotropy_ratio": float(lam.min() / lam.max())}


def test_collapse_detector_examples():
    check(max(1.5, 0.1 * 16), "1.6")
    assert collapse_check(isotropy_report(_hadamard_cloud(), num_slices=64))["status"] == "healthy"

    r = _report_from_spectrum([3.8, 0.1, 0.05, 0.05])
    check(r["top_eigen_share"], "0.95")
    check(r["effective_rank"], "1.285")
    assert collapse_check(r)["status"] == "dimensional-collapse"

    r = _report_from_spectrum([1.0, 1.0, 1.0, 0.005])
    check(r["effective_rank"], "3.032")
    check(r["top_eigen_share"], "0.3328")
    check(r["isotropy_ratio"], "0.005")
    assert collapse_check(r)["status"] == "anisotropic"

    same = torch.full((8, 4), 0.3)
    rep = isotropy_report(same, num_slices=16)
    assert rep["mean_std"] == 0.0 and collapse_check(rep)["status"] == "complete-collapse"


# ------------------------------------------------------------------------ section 5: planning

def test_planning_cost_example():
    goal = torch.tensor([0.0, 0.2])
    plans = torch.tensor([[[1.0, 1.0], [0.5, 0.5], [0.0, 0.2]],
                          [[2.0, 2.0], [1.0, 1.0], [0.0, 0.2]]])
    d2 = (plans[0] - goal).pow(2).sum(-1)
    for v, p in zip(d2, ["1.64", "0.34", "0"]):
        check(v, p)
    s, f = trajectory_cost(plans, goal, "sum"), trajectory_cost(plans, goal, "final")
    check(s[0], "1.98")
    check(s[1], "8.88")
    assert float(f[0]) == pytest.approx(0.0, abs=1e-7) and float(f[1]) == pytest.approx(0.0, abs=1e-7)


def test_cem_round_example():
    """One CEM update exactly as plan_cem does it (topk of lowest cost, population std + min_std)."""
    cfg = CEMConfig()
    assert (cfg.horizon, cfg.samples, cfg.elites, cfg.iters, cfg.init_std, cfg.min_std, cfg.cost) == \
        (5, 300, 30, 10, 1.0, 0.05, "sum")
    acts = torch.tensor([-0.9, -0.4, 0.1, 0.3, 0.6, 1.0])
    zs = (0.0 + acts)[:, None, None]  # toy model z1 = z0 + a, H = 1, d = 1
    cost = trajectory_cost(zs, torch.tensor([0.4]), "sum")
    for v, p in zip(cost, ["1.69", "0.64", "0.09", "0.01", "0.04", "0.36"]):
        check(v, p)
    top = torch.topk(cost, 2, largest=False).indices
    elite = acts[top]
    assert sorted(elite.tolist()) == pytest.approx([0.3, 0.6])
    check(elite.mean(), "0.45")
    check(elite.std(unbiased=False), "0.15")
    check(elite.std(unbiased=False) + cfg.min_std, "0.20")
    check(cost[top].mean(), "0.025")
    check(0.45 + 0.2 * 3, "1.05")
    check((torch.tensor(0.45) + 0.2 * 3).clamp(-1.0, 1.0), "1.0")


def test_cem_budget():
    w = DEFAULT_CONFIG["world"]
    n = w["cem_samples"] * w["horizon"] * w["cem_iters"]
    assert n == 15_000
    assert "300 \\times 5 \\times 10 = 15\\,000" in DOC
