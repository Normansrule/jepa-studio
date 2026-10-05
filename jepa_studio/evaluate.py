"""Evaluation and diagnostics: linear probe, k-NN, isotropy metrics and a collapse detector.

Protocol (docs/equations.md#evaluation):
  * features = backbone embeddings of un-augmented inputs (the projector is discarded, as in
    the LeJEPA paper's linear-probe protocol)
  * a labeled subset (data.labeled_fraction of the training pool) trains the probe / serves
    as the k-NN memory; a disjoint held-out split is scored
  * baselines use exactly the same labeled subset: random-init encoder (same architecture,
    no training) and a supervised encoder trained end-to-end with cross-entropy on it
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from torch import Tensor, nn

from .losses import SIGReg


@torch.no_grad()
def embed(model, x: Tensor, batch: int = 256, device=None) -> tuple[Tensor, Tensor]:
    device = device or next(model.parameters()).device
    was = model.training
    model.eval()
    embs, projs = [], []
    for i in range(0, len(x), batch):
        e, p = model(x[i:i + batch].to(device))
        embs.append(e.float().cpu())
        projs.append(p.float().cpu())
    model.train(was)
    return torch.cat(embs), torch.cat(projs)


def standardize(train: Tensor, *others: Tensor):
    mu, sd = train.mean(0, keepdim=True), train.std(0, keepdim=True) + 1e-6
    return [(t - mu) / sd for t in (train, *others)]


def linear_probe(xtr: Tensor, ytr: Tensor, xte: Tensor, yte: Tensor, epochs: int = 100,
                 lr: float = 1e-2, wd: float = 1e-6, seed: int = 0) -> dict:
    """Multinomial logistic regression on standardized features (full batch AdamW)."""
    g = torch.Generator().manual_seed(seed)
    xtr, xte = standardize(xtr, xte)
    c = int(max(ytr.max(), yte.max())) + 1
    lin = nn.Linear(xtr.shape[1], c)
    with torch.no_grad():
        lin.weight.normal_(0, 0.01, generator=g)
        lin.bias.zero_()
    opt = torch.optim.AdamW(lin.parameters(), lr=lr, weight_decay=wd)
    for _ in range(epochs):
        loss = F.cross_entropy(lin(xtr), ytr)
        opt.zero_grad()
        loss.backward()
        opt.step()
    with torch.no_grad():
        acc = (lin(xte).argmax(1) == yte).float().mean().item()
        tr_acc = (lin(xtr).argmax(1) == ytr).float().mean().item()
    return {"accuracy": acc, "train_accuracy": tr_acc, "final_loss": loss.item()}


def knn_accuracy(xtr: Tensor, ytr: Tensor, xte: Tensor, yte: Tensor, k: int = 20) -> dict:
    """Cosine-similarity k-NN with majority vote (ties -> most similar neighbor's class)."""
    a = F.normalize(xtr, dim=1)
    b = F.normalize(xte, dim=1)
    k = min(k, len(a))
    sims, idx = (b @ a.T).topk(k, dim=1)
    votes = ytr[idx]
    c = int(max(ytr.max(), yte.max())) + 1
    counts = torch.zeros(len(b), c).scatter_add_(1, votes, torch.ones_like(votes, dtype=torch.float))
    counts += 1e-3 * F.one_hot(votes[:, 0], c)  # tie-break
    pred = counts.argmax(1)
    return {"accuracy": (pred == yte).float().mean().item(), "k": k}


def nearest_neighbors(emb: Tensor, query: int, k: int = 8) -> list[tuple[int, float]]:
    e = F.normalize(emb, dim=1)
    s = e @ e[query]
    s[query] = -2
    v, i = s.topk(k)
    return [(int(a), round(float(b), 4)) for a, b in zip(i, v)]


def isotropy_report(z: Tensor, num_slices: int = 1024, seed: int = 0) -> dict:
    """How close the embedding cloud is to N(0, I).

      eigen spectrum of Cov(z)          all equal  <=> isotropic
      effective rank = exp(H(p)),  p_i = lambda_i / sum lambda   (Roy & Vetterli, 2007): in [1, K]
      isotropy ratio = lambda_min / lambda_max                   in [0, 1]
      sigreg        = SIGReg statistic of z (what the loss drives down); E[.] under the null
                      is about 1.05 for the default quadrature (see losses.ep_expected_under_null), so values near it mean "Gaussian"
    """
    z = z.float()
    n, k = z.shape
    zc = z - z.mean(0)
    cov = zc.T @ zc / max(1, n - 1)
    ev = torch.linalg.eigvalsh(cov).clamp_min(0).flip(0)
    p = ev / ev.sum().clamp_min(1e-12)
    h = -(p * (p + 1e-12).log()).sum()
    sig = SIGReg(num_slices=num_slices, seed=seed)
    return {
        "dim": k, "samples": n,
        "eigenvalues": [round(float(x), 6) for x in ev],
        "effective_rank": round(float(h.exp()), 3),
        "isotropy_ratio": round(float(ev[-1] / ev[0].clamp_min(1e-12)), 5),
        "top_eigen_share": round(float(p[0]), 4),
        "mean_norm": round(float(z.norm(dim=1).mean()), 4),
        "mean_abs_mean": round(float(z.mean(0).abs().mean()), 4),
        "mean_std": round(float(z.std(0).mean()), 4),
        "sigreg": round(float(sig(z)), 4),
    }


def collapse_check(rep: dict) -> dict:
    """Plain-language verdict used by the Inspect tab's collapse detector."""
    k = rep["dim"]
    if rep["mean_std"] < 1e-3:
        return {"status": "complete-collapse", "message": "Every input maps to (almost) the same point."}
    if rep["effective_rank"] < max(1.5, 0.1 * k) or rep["top_eigen_share"] > 0.9:
        return {"status": "dimensional-collapse",
                "message": f"Embeddings use about {rep['effective_rank']:.1f} of {k} directions."}
    if rep["isotropy_ratio"] < 0.01:
        return {"status": "anisotropic", "message": "Some directions carry far less variance than others."}
    return {"status": "healthy", "message": f"Spread over ~{rep['effective_rank']:.1f} of {k} directions."}


def pca(z: Tensor, dims: int = 3) -> tuple[Tensor, Tensor]:
    zc = z - z.mean(0)
    u, s, vt = torch.linalg.svd(zc, full_matrices=False)
    comps = vt[:dims]
    var = (s ** 2) / (s ** 2).sum()
    return zc @ comps.T, var[:dims]


def train_supervised(build_fn, x: Tensor, y: Tensor, epochs: int = 30, lr: float = 1e-3, batch: int = 64,
                     device="cpu", seed: int = 0, augment=None) -> nn.Module:
    """Supervised baseline: same encoder, linear head, cross-entropy on the labeled subset."""
    torch.manual_seed(seed)
    model = build_fn().to(device)
    c = int(y.max()) + 1
    head = nn.Linear(model.backbone.embed_dim, c).to(device)
    opt = torch.optim.AdamW(list(model.parameters()) + list(head.parameters()), lr=lr, weight_decay=0.05)
    n = len(x)
    for _ in range(epochs):
        perm = torch.randperm(n)
        for i in range(0, n, batch):
            j = perm[i:i + batch]
            if len(j) < 2:
                continue
            xb = x[j]
            if augment is not None:
                xb = augment(xb)
            e, _ = model(xb.to(device))
            loss = F.cross_entropy(head(e), y[j].to(device))
            opt.zero_grad()
            loss.backward()
            opt.step()
    return model


def flip_crop_augment(x: Tensor) -> Tensor:
    """Light augmentation for the supervised baseline (random flip + 1-pixel-jitter crop)."""
    if x.dim() != 4:
        return x
    flip = torch.rand(len(x)) < 0.5
    x = torch.where(flip[:, None, None, None], x.flip(-1), x)
    pad = F.pad(x, (2, 2, 2, 2), mode="replicate")
    dx, dy = np.random.randint(0, 5, 2)
    return pad[..., dy:dy + x.shape[-2], dx:dx + x.shape[-1]]


def split_indices(n: int, labeled_fraction: float, seed: int = 0, test_fraction: float = 0.2):
    g = torch.Generator().manual_seed(seed)
    perm = torch.randperm(n, generator=g)
    n_test = max(1, int(n * test_fraction))
    test, pool = perm[:n_test], perm[n_test:]
    n_lab = max(10, int(len(pool) * labeled_fraction))
    return pool[:n_lab], test


def evaluate_all(cfg: dict, model, x: Tensor, y: Tensor, build_fn, log=print) -> dict:
    """Everything the Evaluate tab shows, as one dict (also written into the run report)."""
    ev = cfg["eval"]
    lab, test = split_indices(len(x), cfg["data"].get("labeled_fraction", 0.1), cfg["seed"])
    device = next(model.parameters()).device
    out: dict = {"n_labeled": len(lab), "n_test": len(test), "methods": {}}

    def score(name, m):
        e, p = embed(m, x, device=device)
        lp = linear_probe(e[lab], y[lab], e[test], y[test], ev.get("probe_epochs", 100), seed=cfg["seed"])
        kn = knn_accuracy(e[lab], y[lab], e[test], y[test], ev.get("knn_k", 20))
        iso = isotropy_report(p)
        out["methods"][name] = {"linear_probe": lp["accuracy"], "knn": kn["accuracy"],
                                "effective_rank": iso["effective_rank"], "sigreg": iso["sigreg"],
                                "collapse": collapse_check(iso)["status"]}
        log(f"  {name:28s} linear {lp['accuracy']:.3f}  kNN {kn['accuracy']:.3f}  eff.rank {iso['effective_rank']:.1f}")
        return e, p

    score("LeJEPA (this run)", model)
    if "random-init" in ev.get("baselines", []):
        torch.manual_seed(cfg["seed"] + 999)
        score("random-init encoder", build_fn().to(device))
    if "supervised" in ev.get("baselines", []):
        sup = train_supervised(build_fn, x[lab], y[lab], device=device, seed=cfg["seed"],
                               augment=flip_crop_augment)
        score(f"supervised ({len(lab)} labels)", sup)
    out["chance"] = 1.0 / max(1, int(y.max()) + 1)
    return out


# Saliency and ViT attention maps live in maps.py (shared by the server and tests).
from .maps import saliency, vit_attention  # noqa: E402,F401  (kept importable from here)
