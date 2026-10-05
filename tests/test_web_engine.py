"""Parity tests for the in-browser engine (site/js) against the Python package.

Each test shells out to a small Node script in tests/js/ (no npm dependencies) and compares
with the Python implementation. Skipped when `node` is not installed.
"""
from __future__ import annotations

import copy
import json
import shutil
import subprocess
import zipfile
from pathlib import Path

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
JS = ROOT / "tests" / "js"
NODE = shutil.which("node")

pytestmark = pytest.mark.skipif(NODE is None, reason="node is not installed")


def run_node(script: str, *args, timeout: int = 120) -> str:
    res = subprocess.run([NODE, str(JS / script), *map(str, args)], capture_output=True, text=True,
                         timeout=timeout, cwd=ROOT)
    assert res.returncode == 0, f"{script} failed:\n{res.stdout}\n{res.stderr}"
    return res.stdout


# ------------------------------------------------------------------ shapes generator

@pytest.mark.parametrize("seed,size", [(0, 32), (7, 32), (123456789, 16), (4294967295, 24)])
def test_shapes_match_python(seed, size):
    from jepa_studio.data.synthetic import render
    idx = [0, 1, 2, 3, 5, 17, 99, 1000]
    items = json.loads(run_node("shapes_dump.mjs", seed, size, *idx))
    labels = set()
    for it in items:
        img, cls = render(seed, it["index"], size)
        js = np.asarray(it["img"], dtype=np.float32).reshape(size, size, 3)
        assert it["label"] == cls
        np.testing.assert_allclose(js, img, atol=1e-5, rtol=0)
        labels.add(cls)
    assert len(labels) > 1


def test_shapes_cover_all_classes():
    from jepa_studio.data.synthetic import render
    idx = list(range(60))
    items = json.loads(run_node("shapes_dump.mjs", 3, 16, *idx))
    got = {it["label"] for it in items}
    assert got == {render(3, i, 16)[1] for i in idx}
    assert len(got) == 10


# ------------------------------------------------------------------ config schema + validator

def test_site_schema_is_byte_identical():
    a = (ROOT / "schema" / "config.schema.json").read_bytes()
    b = (ROOT / "site" / "schema" / "config.schema.json").read_bytes()
    assert a == b, "site/schema/config.schema.json must be a verbatim copy of schema/config.schema.json"


def _config_variants() -> dict[str, dict | list | str]:
    from jepa_studio.config import DEFAULT_CONFIG
    d = copy.deepcopy(DEFAULT_CONFIG)
    v: dict = {"default": d}

    def mod(name, fn):
        c = copy.deepcopy(DEFAULT_CONFIG)
        fn(c)
        v[name] = c

    mod("web-demo", lambda c: (c["model"].update(arch="mlp-tiny", embed_dim=64, proj_dim=16),
                               c["train"].update(max_steps=300, batch_size=64)))
    mod("unknown-top-key", lambda c: c.update(extra=1))
    mod("unknown-nested-key", lambda c: c["data"].update(colour_jitter=0.3))
    mod("missing-required", lambda c: c.pop("objective"))
    mod("missing-nested-required", lambda c: c["objective"].pop("knots"))
    mod("bad-enum", lambda c: c["model"].update(arch="resnet-152"))
    mod("too-small", lambda c: c["data"].update(image_size=4))
    mod("too-big", lambda c: c["objective"].update(**{"lambda": 1.5}))
    mod("wrong-type-str", lambda c: c.update(seed="zero"))
    mod("float-for-int", lambda c: c.update(seed=1.0))
    mod("bool-for-int", lambda c: c["data"].update(n_local=True))
    mod("batch-auto", lambda c: c["train"].update(batch_size="auto"))
    mod("batch-bad-string", lambda c: c["train"].update(batch_size="big"))
    mod("batch-too-small", lambda c: c["train"].update(batch_size=1))
    mod("scale-wrong-len", lambda c: c["data"].update(global_scale=[0.3]))
    mod("scale-item-range", lambda c: c["data"].update(local_scale=[0.0, 0.3]))
    mod("max-steps-null", lambda c: c["train"].update(max_steps=None))
    mod("path-null", lambda c: c["data"].update(path=None))
    mod("path-number", lambda c: c["data"].update(path=3))
    mod("name-too-long", lambda c: c.update(name="x" * 81))
    mod("baselines-bad", lambda c: c["eval"].update(baselines=["random-init", "oracle"]))
    mod("compile-auto", lambda c: c["hardware"].update(compile="auto"))
    mod("compile-str", lambda c: c["hardware"].update(compile="yes"))
    mod("workers-neg", lambda c: c["hardware"].update(workers=-1))
    mod("version-2", lambda c: c.update(version=2))
    mod("int-for-number", lambda c: c["objective"].update(t_max=3))
    mod("world-env-bad", lambda c: c["world"].update(env="maze"))
    mod("seed-max", lambda c: c.update(seed=4294967295))
    mod("seed-over", lambda c: c.update(seed=4294967296))
    mod("name-traversal", lambda c: c.update(name="../../escape"))
    mod("name-absolute", lambda c: c.update(name="/etc/x"))
    mod("name-backslash", lambda c: c.update(name="a\\b"))
    mod("name-ok-dots", lambda c: c.update(name="run.v2_final-1"))
    v["not-an-object"] = [1, 2, 3]
    return v


def test_config_validator_parity(tmp_path):
    from jepa_studio.config import load_schema, validate
    schema = load_schema()
    files = []
    expected = {}
    for name, cfg in _config_variants().items():
        p = tmp_path / f"{name}.json"
        p.write_text(json.dumps(cfg, indent=2))
        files.append(str(p))
        expected[str(p)] = validate(cfg, schema)
    for p in sorted((ROOT / "configs").glob("*.json")):
        files.append(str(p))
        expected[str(p)] = validate(json.loads(p.read_text()), schema)
    got = json.loads(run_node("config_check.mjs", *files))
    for f in files:
        py, js = expected[f], got[f]
        assert bool(py) == bool(js), f"{Path(f).name}: python {py} vs js {js}"
        assert len(py) == len(js), f"{Path(f).name}: python {py} vs js {js}"
        assert [e.split(":")[0] for e in py] == [e.split(":")[0] for e in js], f"{Path(f).name}"
    assert not expected[files[0]], "DEFAULT_CONFIG must be valid"
    assert sum(1 for f in files if expected[f]) >= 15, "expected many broken variants to be rejected"


# ------------------------------------------------------------------ loss

@pytest.mark.parametrize("lam,V,Vg,N,K,M", [(0.05, 6, 2, 32, 16, 64), (0.5, 3, 1, 17, 8, 5), (1.0, 2, 2, 64, 4, 16)])
def test_loss_matches_pytorch(tmp_path, lam, V, Vg, N, K, M):
    from jepa_studio.losses import SIGReg, lejepa_loss, random_directions
    g = torch.Generator().manual_seed(V * 100 + N)
    z = (torch.randn(V, N, K, generator=g) * 1.7 + 0.3).float()
    dirs = [random_directions(K, M, g) for _ in range(V)]
    it = iter(dirs)

    class FixedSIGReg(SIGReg):
        def forward(self, zz, directions=None):  # noqa: ARG002
            return super().forward(zz, next(it))

    sig = FixedSIGReg(num_slices=M)
    zr = z.clone().requires_grad_(True)
    out = lejepa_loss(zr[:Vg], zr, sig, lam)
    out.loss.backward()
    inp = {"V": V, "Vg": Vg, "N": N, "K": K, "M": M, "lam": lam, "t_max": 3.0, "knots": 17,
           "z": z.flatten().tolist(), "dirs": [d.flatten().tolist() for d in dirs]}
    f = tmp_path / "inp.json"
    f.write_text(json.dumps(inp))
    js = json.loads(run_node("loss_eval.mjs", f))
    assert abs(js["loss"] - out.loss.item()) < 1e-4 * max(1.0, abs(out.loss.item()))
    assert abs(js["pred"] - out.prediction.item()) < 1e-4 * max(1.0, abs(out.prediction.item()))
    assert abs(js["sigreg"] - out.sigreg.item()) < 1e-4 * max(1.0, abs(out.sigreg.item()))
    np.testing.assert_allclose(np.array(js["grad"]).reshape(V, N, K), zr.grad.numpy(), atol=1e-5, rtol=1e-3)


def test_gradient_check():
    out = run_node("engine_gradcheck.mjs")
    assert "FAIL" not in out
    assert "worst relative error" in out


# ------------------------------------------------------------------ file writers

def test_writers_load_in_python(tmp_path):
    from safetensors import safe_open
    from safetensors.torch import load_file
    run_node("io_write.mjs", tmp_path)
    exp = json.loads((tmp_path / "expected.json").read_text())

    st = load_file(str(tmp_path / "weights.safetensors"))
    assert set(st) == set(exp["tensors"])
    for name, info in exp["tensors"].items():
        assert list(st[name].shape) == info["shape"]
        assert st[name].dtype == torch.float32
        np.testing.assert_allclose(st[name].flatten()[:4].numpy(), np.float32(info["head"]))
    with safe_open(str(tmp_path / "weights.safetensors"), "pt") as f:
        meta = f.metadata()
    assert meta == {"arch": "mlp-tiny", "step": "12", "tier": "web-demo"}
    # the web backbone loads into the desktop MLPTiny with no renaming
    from jepa_studio.models import MLPTiny
    bb = MLPTiny(3, 64)
    bb.load_state_dict({k[len("backbone."):]: v for k, v in st.items() if k.startswith("backbone.")})

    emb = np.load(tmp_path / "emb.npy", allow_pickle=False)
    assert emb.dtype == np.float32 and emb.shape == (7, 5)
    np.testing.assert_array_equal(emb.flatten(), np.float32(exp["emb"]))
    vec = np.load(tmp_path / "vec.npy", allow_pickle=False)
    assert vec.shape == (3,) and vec.tolist() == [1.5, -2.0, 3.25]

    with zipfile.ZipFile(tmp_path / "bundle.zip") as z:
        assert z.testzip() is None  # verifies every CRC
        names = z.namelist()
        assert names == ["config.json", "README.txt", "weights.safetensors", "data/bin.dat", "empty.txt"]
        assert z.read("README.txt").decode("utf-8") == "héllo — unicode ✓\n"
        assert z.read("weights.safetensors") == (tmp_path / "weights.safetensors").read_bytes()
        assert z.read("data/bin.dat") == bytes((i * 37) & 0xFF for i in range(1000))
        assert all(i.compress_type == zipfile.ZIP_STORED for i in z.infolist())


# ------------------------------------------------------------------ evaluation helpers

def test_eval_parity(tmp_path):
    """knn_accuracy, isotropy_report and collapse_check ported to site/js/engine/evaluate.js."""
    from jepa_studio.evaluate import collapse_check, isotropy_report, knn_accuracy
    g = torch.Generator().manual_seed(0)
    cases = []
    for n, d, c in [(80, 8, 4), (150, 16, 10)]:
        xtr = torch.randn(n, d, generator=g)
        ytr = torch.randint(0, c, (n,), generator=g)
        xte = torch.randn(40, d, generator=g)
        yte = torch.randint(0, c, (40,), generator=g)
        cases.append({"xtr": xtr.tolist(), "ytr": ytr.tolist(), "xte": xte.tolist(), "yte": yte.tolist(), "k": 20})
    # low-rank and collapsed embeddings for the collapse detector
    low = torch.randn(200, 2, generator=g) @ torch.randn(2, 16, generator=g)
    iso = torch.randn(200, 16, generator=g)
    aniso = iso * torch.tensor([1.0] * 15 + [0.05])
    dead = torch.ones(100, 16) + 1e-5 * torch.randn(100, 16, generator=g)
    mats = {"low": low, "iso": iso, "aniso": aniso, "dead": dead}
    f = tmp_path / "eval.json"
    f.write_text(json.dumps({"knn": cases, "mats": {k: v.tolist() for k, v in mats.items()}}))
    js = json.loads(run_node("eval_check.mjs", f))
    for case, got in zip(cases, js["knn"]):
        ref = knn_accuracy(torch.tensor(case["xtr"]), torch.tensor(case["ytr"]), torch.tensor(case["xte"]),
                           torch.tensor(case["yte"]), case["k"])
        assert abs(ref["accuracy"] - got["accuracy"]) < 1e-6  # torch reports a float32 mean
        assert ref["k"] == got["k"]
    for name, m in mats.items():
        rep = isotropy_report(m, num_slices=64)
        got = js["iso"][name]
        np.testing.assert_allclose(got["eigenvalues"], rep["eigenvalues"], rtol=2e-3, atol=2e-5)
        assert abs(got["effective_rank"] - rep["effective_rank"]) < 2e-2
        assert abs(got["top_eigen_share"] - rep["top_eigen_share"]) < 2e-3
        assert abs(got["mean_std"] - rep["mean_std"]) < 2e-3
        assert js["collapse"][name]["status"] == collapse_check(rep)["status"], name


def test_trainer_smoke():
    """30 demo-tier steps in Node: loss decreases, evaluation returns all three methods."""
    out = json.loads(run_node("trainer_smoke.mjs", timeout=300))
    assert out["last"] < out["first"]
    assert set(out["ev"]["methods"]) == {"LeJEPA (this run)", "random-init encoder", "supervised (82 labels)"}
