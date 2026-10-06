"""Saliency and ViT attention maps (jepa_studio/maps.py) and the server's /saliency endpoint."""
import base64
import io
import json
import os
import threading
import time
import urllib.request

import pytest
import torch

from jepa_studio.config import load_config
from jepa_studio.maps import MapError, attention_weights, saliency, vit_attention
from jepa_studio.models import build_encoder


def _model(arch, **model):
    cfg = load_config(overrides={"data": {"image_size": 16, "max_items": 32},
                                 "model": {"arch": arch, "embed_dim": 32, "proj_dim": 8, "proj_hidden": 32,
                                           "patch_size": 4, "depth": 2, "heads": 4, **model}})
    torch.manual_seed(0)
    return build_encoder(cfg)


def _img(seed=1, side=16):
    return torch.rand(3, side, side, generator=torch.Generator().manual_seed(seed))


@pytest.mark.parametrize("arch", ["convnet-tiny", "vit-tiny", "mlp-tiny"])
def test_saliency_shape_range_determinism(arch):
    m = _model(arch)
    x = _img()
    s = saliency(m, x)
    assert s.shape == (16, 16)
    assert torch.isfinite(s).all() and s.min() >= 0 and s.max() == pytest.approx(1.0)
    # repeatable; not bit-exact across CPU kernels/thread counts (seen on a 6-core AVX-512 machine)
    assert torch.allclose(s, saliency(m, x), atol=1e-6)
    b = saliency(m, torch.stack([x, _img(2)]))
    assert b.shape == (2, 16, 16)
    # a sample's map does not depend on the rest of the batch (one backward pass per image)
    assert torch.allclose(b[0], s, atol=1e-6)
    assert m.training and all(p.grad is None for p in m.parameters())  # mode restored, no grads left


def test_saliency_matches_web_definition():
    """mlp-tiny on 16x16: |d||emb||/dx| max over channels, as site/js/engine/mlp.js computes it."""
    m = _model("mlp-tiny").eval()
    x = _img(3)
    xg = x.clone().requires_grad_(True)
    m.backbone(xg[None]).norm().backward()
    ref = xg.grad.abs().amax(0)
    assert torch.allclose(saliency(m, x), ref / ref.max(), atol=1e-6)


def test_pixel_that_cannot_affect_output_has_zero_saliency():
    m = _model("mlp-tiny")
    with torch.no_grad():  # cut every weight reading pixel (row 5, col 7) in all 3 channels
        w = m.backbone.net[0].weight
        for c in range(3):
            w[:, c * 256 + 5 * 16 + 7] = 0
    s = saliency(m, _img(4))
    assert s[5, 7] == 0 and s.max() == pytest.approx(1.0)


def test_attention_rows_are_distributions_and_map_matches_grid():
    m = _model("vit-tiny")
    x = _img(5)
    with torch.no_grad():
        before = m.eval()(x[None])
    att, grid = attention_weights(m, x)
    assert grid == (4, 4) and att.shape == (1, 4, 17, 17)
    assert torch.allclose(att.sum(-1), torch.ones(1, 4, 17), atol=1e-5)
    a, g = vit_attention(m, x)
    assert g == (4, 4) and a.shape == (4, 4)
    assert a.min() >= 0 and a.max() == pytest.approx(1.0)
    raw = att[0, :, 0, 1:].mean(0).reshape(4, 4)
    assert torch.allclose(a, raw / raw.max(), atol=1e-6)
    with torch.no_grad():  # capturing the weights changed nothing about the model
        after = m(x[None])
    assert all(torch.equal(u, v) for u, v in zip(before, after))
    assert not m.backbone.blocks[-1].attn._forward_hooks
    assert vit_attention(m, torch.stack([x, x]))[0].shape == (2, 4, 4)


def test_non_image_models_are_refused():
    with pytest.raises(MapError, match="series"):
        saliency(_model("series-conv"), torch.rand(1, 3, 64))
    with pytest.raises(MapError, match="clip"):
        saliency(_model("video-convnet"), torch.rand(1, 4, 3, 16, 16))
    with pytest.raises(MapError, match="vit-tiny"):
        vit_attention(_model("convnet-tiny"), _img())


# ----------------------------------------------------------------- server

def _write_run(root, rid, arch):
    """A finished run folder as the trainer leaves it (config.json + encoder.safetensors), at init weights."""
    from jepa_studio.weights import save_state_dict

    cfg = load_config(overrides={"name": rid, "data": {"max_items": 48, "image_size": 32},
                                 "model": {"arch": arch, "embed_dim": 32, "proj_dim": 8, "proj_hidden": 32,
                                           "depth": 2, "heads": 4}})
    d = root / "runs" / rid
    d.mkdir(parents=True)
    (d / "config.json").write_text(json.dumps(cfg))
    torch.manual_seed(0)
    save_state_dict(build_encoder(cfg), d / "encoder.safetensors", {"arch": arch, "step": 0})


@pytest.fixture(scope="module")
def server(tmp_path_factory):
    from jepa_studio import server as srv

    root = tmp_path_factory.mktemp("maps")
    _write_run(root, "vit-run", "vit-tiny")
    _write_run(root, "conv-run", "convnet-tiny")
    token = "t0k3n-" + base64.urlsafe_b64encode(os.urandom(16)).decode()
    ready = io.StringIO()
    threading.Thread(target=srv.serve, kwargs=dict(root=str(root), port=0, token=token, allowed_dirs=[str(root)],
                                                   ready_file=ready), daemon=True).start()
    for _ in range(100):
        if ready.getvalue():
            break
        time.sleep(0.05)
    port = json.loads(ready.getvalue().splitlines()[0])["port"]
    return port, token


def _get(port, token, path):
    r = urllib.request.Request(f"http://127.0.0.1:{port}{path}")
    r.add_header("X-JEPA-Token", token)
    try:
        with urllib.request.urlopen(r, timeout=60) as resp:
            return resp.status, json.loads(resp.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read() or b"{}")


def test_server_saliency_endpoint(server):
    port, token = server
    code, b = _get(port, token, "/api/runs/vit-run/saliency?i=3")
    assert code == 200, b
    assert b["index"] == 3 and b["kind"] == "gradient" and b["side"] == 32
    assert len(b["map"]) == 32 and all(len(r) == 32 for r in b["map"])
    flat = [v for r in b["map"] for v in r]
    assert min(flat) >= 0 and max(flat) == pytest.approx(1.0)
    assert b["image"].startswith("data:image/png;base64,")
    a = b["attention"]
    assert a["grid"] == [8, 8] and a["side"] == 8 and len(a["map"]) == 8 and len(a["map"][0]) == 8

    code, c = _get(port, token, "/api/runs/conv-run/saliency?i=0")
    assert code == 200 and "attention" not in c and c["side"] == 32

    for bad in ("-1", "48", "99999", "x"):
        code, e = _get(port, token, f"/api/runs/conv-run/saliency?i={bad}")
        assert code == 400 and "error" in e, (bad, code, e)
    assert _get(port, "wrong-token", "/api/runs/conv-run/saliency?i=0")[0] == 401
    assert _get(port, token, "/api/runs/..%2Fx/saliency?i=0")[0] == 404
