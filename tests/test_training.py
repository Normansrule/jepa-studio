"""Training loop, hardware decisions, checkpoints/resume, evaluation metrics and export bundle."""
import json

import pytest
import torch

from jepa_studio import hardware as hw
from jepa_studio.config import load_config
from jepa_studio.evaluate import collapse_check, isotropy_report, knn_accuracy, linear_probe
from jepa_studio.models import build_encoder
from jepa_studio.train import Trainer, lr_at


def tiny_cfg(**train):
    return load_config(overrides={
        "name": "t", "data": {"max_items": 64, "n_local": 2},
        "model": {"arch": "convnet-tiny", "embed_dim": 32, "proj_dim": 8, "proj_hidden": 32},
        "objective": {"num_slices": 32},
        "train": {"epochs": 1, "batch_size": 16, "effective_batch": 32, "log_every": 1, "checkpoint_every": 2, **train},
        "hardware": {"mode": "max"}})


def test_recommendations_have_reasons():
    card = hw.probe(".", measure_disk=False)
    ds = hw.recommend(card, tiny_cfg())
    names = {d.setting for d in ds}
    assert {"precision", "channels_last", "compile", "workers", "pin_memory", "torch_threads"} <= names
    assert all(d.reason for d in ds)


def test_lr_schedule_shape():
    lrs = [lr_at(s, 100, 1.0, 0.1, 1e-3) for s in range(100)]
    assert lrs[0] == pytest.approx(0.1) and max(lrs) == pytest.approx(1.0)
    assert lrs[-1] < 0.01 and all(a >= b - 1e-12 for a, b in zip(lrs[10:], lrs[11:]))


def test_train_checkpoint_resume_and_control(tmp_path):
    cfg = tiny_cfg(max_steps=4)
    tr = Trainer(cfg, run_dir=tmp_path / "r", log=lambda *_: None, measure_disk=False)
    res = tr.fit(resume=False)
    assert res["status"] == "done" and res["steps"] == 4
    ev = [json.loads(x) for x in (tmp_path / "r/events.jsonl").read_text().splitlines()]
    steps = [e for e in ev if e["event"] == "step"]
    assert steps and "proj_hist" in steps[0] and len(steps[0]["proj_hist"]) == 3
    decisions = json.loads((tmp_path / "r/hardware.json").read_text())["decisions"]
    assert any(d["setting"] == "grad_accumulation" and d["value"] == 2 for d in decisions)
    # resume continues from the saved step instead of starting over
    cfg2 = tiny_cfg(max_steps=6)
    tr2 = Trainer(cfg2, run_dir=tmp_path / "r", log=lambda *_: None, measure_disk=False)
    res2 = tr2.fit(resume=True)
    assert res2["steps"] == 6
    ev2 = (tmp_path / "r/events.jsonl").read_text().splitlines()
    first_new = [json.loads(x) for x in ev2[len(ev):] if '"event": "step"' in x][0]
    assert first_new["step"] == 5
    # a stop command ends the run with a checkpoint
    (tmp_path / "s").mkdir()
    (tmp_path / "s/control.json").write_text(json.dumps({"command": "stop"}))
    tr3 = Trainer(tiny_cfg(max_steps=50), run_dir=tmp_path / "s", log=lambda *_: None, measure_disk=False)
    assert tr3.fit(resume=False)["status"] == "stopped"
    assert (tmp_path / "s/ckpt/LATEST").exists()


def test_export_bundle_and_verify(tmp_path):
    from jepa_studio.export import export_bundle, verify_bundle
    import zipfile

    tr = Trainer(tiny_cfg(max_steps=2), run_dir=tmp_path / "r", log=lambda *_: None, measure_disk=False)
    tr.fit(resume=False)
    bundle, report = export_bundle(tmp_path / "r", n_embed=32)
    assert report.read_text().startswith("<!doctype html>")
    out = tmp_path / "unz"
    zipfile.ZipFile(bundle).extractall(out)
    assert verify_bundle(out) == []
    (out / "config.json").write_text("{}")
    assert verify_bundle(out)


def test_metrics_on_known_cases():
    torch.manual_seed(0)
    # linearly separable clusters -> probe and kNN near perfect
    y = torch.arange(400) % 4
    x = torch.randn(400, 8) * 0.1 + torch.nn.functional.one_hot(y, 8).float() * 3
    assert linear_probe(x[:300], y[:300], x[300:], y[300:])["accuracy"] > 0.95
    assert knn_accuracy(x[:300], y[:300], x[300:], y[300:], k=5)["accuracy"] > 0.95
    iso = isotropy_report(torch.randn(4000, 16))
    assert iso["effective_rank"] > 15 and collapse_check(iso)["status"] == "healthy"
    col = isotropy_report(torch.randn(4000, 1) @ torch.randn(1, 16))
    assert collapse_check(col)["status"] == "dimensional-collapse"
    assert collapse_check(isotropy_report(torch.ones(100, 16)))["status"] == "complete-collapse"


@pytest.mark.parametrize("arch", ["convnet-tiny", "convnet-small", "vit-tiny", "mlp-tiny"])
def test_every_image_arch_handles_global_and_local_sizes(arch):
    cfg = load_config(overrides={"model": {"arch": arch, "embed_dim": 64, "proj_dim": 8}})
    m = build_encoder(cfg).eval()
    for s in (32, 16):
        e, p = m(torch.rand(3, 3, s, s))
        assert e.shape == (3, 64) and p.shape == (3, 8) and torch.isfinite(p).all()


def test_small_dataset_caps_grad_accumulation(tmp_path):
    """Regression: 32 images, batch 32, effective batch 256 used to accumulate forever (no step ever
    finished) because the micro-batch counter resets each epoch. accum is now capped at the batches
    available per epoch."""
    cfg = tiny_cfg(max_steps=3)
    cfg["data"]["max_items"] = 32
    cfg["train"]["batch_size"] = 32
    cfg["train"]["effective_batch"] = 256
    tr = Trainer(cfg, run_dir=tmp_path / "r", log=lambda *_: None, measure_disk=False)
    res = tr.fit(resume=False)
    assert res["status"] == "done" and res["steps"] == 3
    dec = {d["setting"]: d for d in json.loads((tmp_path / "r/hardware.json").read_text())["decisions"]}
    assert dec["grad_accumulation"]["value"] == 1 and "capped" in dec["grad_accumulation"]["reason"]


# ----------------------------------------------------------------- ONNX export

def _fake_run(tmp_path, arch, **data):
    """A run folder (config.json + encoder.safetensors) without training: random weights with
    non-trivial BatchNorm statistics, so eval-mode export is actually exercised."""
    from jepa_studio.config import save_config
    from jepa_studio.weights import save_state_dict

    cfg = load_config(overrides={"name": arch, "data": data,
                                 "model": {"arch": arch, "embed_dim": 32, "proj_dim": 8, "proj_hidden": 32}})
    torch.manual_seed(0)
    m = build_encoder(cfg)
    if cfg["data"]["kind"] in ("images", "synthetic-shapes"):
        with torch.no_grad():
            m.train()(torch.rand(8, 3, cfg["data"]["image_size"], cfg["data"]["image_size"]) * 2 + 0.5)
    d = tmp_path / arch
    d.mkdir()
    save_config(cfg, d / "config.json")
    save_state_dict(m, d / "encoder.safetensors")
    return d, cfg, m.eval()


@pytest.mark.parametrize("arch", ["convnet-tiny", "convnet-small", "vit-tiny", "mlp-tiny"])
def test_onnx_export_matches_pytorch(tmp_path, arch):
    ort = pytest.importorskip("onnxruntime")
    pytest.importorskip("onnxscript")
    from jepa_studio.export import export_onnx

    d, cfg, m = _fake_run(tmp_path, arch)
    path = export_onnx(d)
    assert path == d / "encoder.onnx"
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in sess.get_inputs()] == ["image"]
    assert [o.name for o in sess.get_outputs()] == ["embedding", "projection"]
    s = cfg["data"]["image_size"]
    for n in (1, 5):  # batch axis is dynamic
        x = torch.rand(n, 3, s, s, generator=torch.Generator().manual_seed(n))
        e, p = sess.run(None, {"image": x.numpy()})
        with torch.no_grad():
            e2, p2 = m(x)
        assert e.shape == (n, 32) and p.shape == (n, 8)
        assert abs(e - e2.numpy()).max() < 1e-4 and abs(p - p2.numpy()).max() < 1e-4


def test_onnx_refuses_non_image_archs(tmp_path):
    from jepa_studio.export import export_onnx

    d, _, _ = _fake_run(tmp_path, "series-conv", kind="timeseries")
    with pytest.raises(ValueError, match="image encoders.*series-conv"):
        export_onnx(d)
    assert not (d / "encoder.onnx").exists()


def test_export_bundle_records_onnx_hash(tmp_path):
    pytest.importorskip("onnxscript")
    import zipfile

    from jepa_studio.export import export_bundle, verify_bundle
    from jepa_studio.weights import sha256_file

    tr = Trainer(tiny_cfg(max_steps=1), run_dir=tmp_path / "r", log=lambda *_: None, measure_disk=False)
    tr.fit(resume=False)
    bundle, _ = export_bundle(tmp_path / "r", n_embed=16, onnx=True)
    sha = sha256_file(tmp_path / "r/encoder.onnx")
    man = json.loads((tmp_path / "r/manifest.json").read_text())
    assert {"path": "encoder.onnx", "sha256": sha}.items() <= next(
        f for f in man["files"] if f["path"] == "encoder.onnx").items()
    assert json.loads((tmp_path / "r/report.json").read_text())["onnx_sha256"] == sha
    out = tmp_path / "unz"
    zipfile.ZipFile(bundle).extractall(out)
    assert (out / "encoder.onnx").exists() and verify_bundle(out) == []


# ----------------------------------------------------------------- live memory guard

def test_memory_guard_stops_with_checkpoint_and_reason(tmp_path):
    """An injected memory reader reports 95% of RAM used from step 3 on; with max_memory_frac 0.85
    the run must checkpoint, emit a 'stopped' event with the reason and end cleanly."""
    calls = []

    def fake_reader(device):
        calls.append(device)
        used = 0.95 if len(calls) >= 3 else 0.10
        return [{"what": "process RAM (RSS)", "used": int(used * 2**34), "total": 2**34}]

    cfg = tiny_cfg(max_steps=20, effective_batch=16, checkpoint_every=100)
    tr = Trainer(cfg, run_dir=tmp_path / "m", log=lambda *_: None, measure_disk=False, memory_reader=fake_reader)
    res = tr.fit(resume=False)
    assert res["status"] == "stopped" and res["steps"] == 3
    assert res["reason"].startswith("memory guard: process RAM (RSS) 15.20 GB")
    ev = [json.loads(x) for x in (tmp_path / "m/events.jsonl").read_text().splitlines()]
    stopped = [e for e in ev if e["event"] == "stopped"]
    assert len(stopped) == 1 and stopped[0]["step"] == 3 and "max_memory_frac=0.85" in stopped[0]["reason"]
    assert ev[-1]["event"] == "end" and ev[-1]["status"] == "stopped"
    assert (tmp_path / "m/ckpt/LATEST").read_text() == "step-0000003"
    assert (tmp_path / "m/encoder.safetensors").exists()


def test_memory_breach_reading():
    assert hw.memory_breach([{"what": "x", "used": 80, "total": 100}], 0.85) is None
    assert "x 0.00 GB" in hw.memory_breach([{"what": "x", "used": 90, "total": 100}], 0.85)
    r = hw.memory_usage(torch.device("cpu"))
    assert r[0]["what"] == "process RAM (RSS)" and 0 < r[0]["used"] < r[0]["total"]


def test_workers_in_threaded_process_use_forkserver(tmp_path):
    """Regression: loader workers were forked from a multi-threaded process (desktop backend,
    pytest with browser threads): Python 3.12 warns that this can deadlock. Training from a
    thread with 1 worker now uses forkserver, finishes, and forks nothing."""
    import threading
    import warnings
    from jepa_studio.train import _worker_context

    assert _worker_context(0) is None
    cfg = tiny_cfg(max_steps=2)
    cfg["hardware"]["workers"] = 1
    out = {}

    def run():
        with warnings.catch_warnings(record=True) as w:
            warnings.simplefilter("always")
            out["res"] = Trainer(cfg, run_dir=tmp_path / "r", log=lambda *_: None, measure_disk=False).fit(resume=False)
            out["warn"] = [str(x.message) for x in w if "fork()" in str(x.message)]
        out["ctx"] = _worker_context(1)

    th = threading.Thread(target=run)
    th.start()
    th.join()
    assert out["res"]["status"] == "done"
    assert out["warn"] == []
    if out["ctx"] is not None:
        assert out["ctx"].get_start_method() == "forkserver"
