"""Run report (JSON + self-contained HTML) and reproducibility bundle (zip with manifest).

The bundle contains everything needed to re-run and to check the result:
    config.json  hardware.json  events.jsonl  eval.json (if evaluated)  encoder.safetensors
    embeddings.npy (backbone features of the first 2048 un-augmented inputs, if computable)
    environment.json (python / torch / package versions, OS)   REPRODUCE.md
    encoder.onnx (only with export_bundle(onnx=True) / `jepa-studio export --onnx`)
    manifest.json (SHA-256 of every file above)

ONNX is export only: jepa-studio never loads .onnx files (weights are read from safetensors).
"""
from __future__ import annotations

import html
import json
import platform
import sys
import time
import zipfile
from pathlib import Path

import numpy as np

from . import __version__
from .weights import atomic_write_bytes, build_manifest, sha256_file


def _events(run_dir: Path) -> list[dict]:
    p = run_dir / "events.jsonl"
    return [json.loads(x) for x in p.read_text().splitlines()] if p.exists() else []


def environment() -> dict:
    import importlib.metadata as md

    pk = {}
    for name in ("torch", "torchvision", "numpy", "safetensors", "pillow", "psutil", "cryptography", "imageio"):
        try:
            pk[name] = md.version(name)
        except md.PackageNotFoundError:
            pass
    return {"jepa_studio": __version__, "python": sys.version.split()[0], "platform": platform.platform(),
            "packages": pk}


ONNX_ARCHS = ("convnet-tiny", "convnet-small", "vit-tiny", "mlp-tiny")
ONNX_OPSET = 18


def export_onnx(run_dir: str | Path, out: str | Path | None = None) -> Path:
    """Write the trained encoder + projector of an image run as ONNX.

    Input ``image`` (N, C, S, S) float32 with S = data.image_size; outputs ``embedding`` (N, D) and
    ``projection`` (N, P), as in eval mode (BatchNorm running statistics). The batch axis is
    dynamic; height and width are fixed at the trained size. Uses the torch.export-based exporter
    (needs the ``onnx`` and ``onnxscript`` packages: ``pip install "jepa-studio[onnx]"``).
    """
    import torch

    from .config import load_config
    from .models import build_encoder
    from .weights import load_run_weights

    d = Path(run_dir)
    cfg = load_config(d / "config.json")
    arch = cfg["model"]["arch"]
    if arch not in ONNX_ARCHS:
        raise ValueError(f"ONNX export supports the image encoders ({', '.join(ONNX_ARCHS)}); "
                         f"this run uses '{arch}' ({cfg['data']['kind']} data), which is not exported")
    if not (d / "encoder.safetensors").exists():
        raise FileNotFoundError(f"{d / 'encoder.safetensors'} not found: train the run first")
    try:
        import onnx  # noqa: F401
        import onnxscript  # noqa: F401
    except ImportError as e:
        raise RuntimeError(f"ONNX export needs the optional extra: pip install \"jepa-studio[onnx]\" ({e})") from e
    model = build_encoder(cfg)
    load_run_weights(model, d)
    model = model.float().cpu().eval()
    s, ch = cfg["data"]["image_size"], cfg["data"].get("channels", 3)
    example = torch.rand(2, ch, s, s, generator=torch.Generator().manual_seed(0))
    out = Path(out) if out else d / "encoder.onnx"
    tmp = out.with_name(out.name + ".tmp")
    with torch.no_grad():
        torch.onnx.export(model, (example,), str(tmp), input_names=["image"],
                          output_names=["embedding", "projection"], opset_version=ONNX_OPSET, dynamo=True,
                          dynamic_shapes={"x": {0: torch.export.Dim("batch", min=1, max=65536)}},
                          external_data=False, verbose=False)
    tmp.replace(out)
    return out


def build_report(run_dir: str | Path) -> dict:
    d = Path(run_dir)
    ev = _events(d)
    steps = [e for e in ev if e.get("event") == "step"]
    rep = {
        "format": "jepa-studio-report/v1",
        "generated": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "run": d.name,
        "config": json.loads((d / "config.json").read_text()),
        "hardware": json.loads((d / "hardware.json").read_text()) if (d / "hardware.json").exists() else None,
        "evaluation": json.loads((d / "eval.json").read_text()) if (d / "eval.json").exists() else None,
        "environment": environment(),
        "encoder_sha256": sha256_file(d / "encoder.safetensors") if (d / "encoder.safetensors").exists() else None,
        "onnx_sha256": None,
        "training": {
            "steps": steps[-1]["step"] if steps else 0,
            "final": {k: steps[-1].get(k) for k in ("loss", "pred", "sigreg", "lr")} if steps else None,
            "mean_samples_per_s": round(float(np.mean([s["samples_per_s"] for s in steps[1:]])), 1) if len(steps) > 1 else None,
            "wall_seconds": round(ev[-1]["t"] - ev[0]["t"], 1) if ev else None,
            "curve": [{k: s.get(k) for k in ("step", "loss", "pred", "sigreg", "lr", "samples_per_s")} for s in steps],
        },
    }
    return rep


def _svg_line(points: list[tuple[float, float]], w=560, h=160, color="#3c8fe8", label="") -> str:
    if len(points) < 2:
        return ""
    xs, ys = zip(*points)
    x0, x1, y0, y1 = min(xs), max(xs), min(ys), max(ys)
    if y1 == y0:
        y1 = y0 + 1
    path = " ".join(f"{'M' if i == 0 else 'L'}{40 + (x - x0) / (x1 - x0 or 1) * (w - 50):.1f},"
                    f"{10 + (1 - (y - y0) / (y1 - y0)) * (h - 30):.1f}" for i, (x, y) in enumerate(points))
    return (f'<svg viewBox="0 0 {w} {h}" role="img" aria-label="{html.escape(label)}">'
            f'<path d="{path}" fill="none" stroke="{color}" stroke-width="2"/>'
            f'<text x="40" y="{h - 4}" font-size="11">step {x0:g}–{x1:g}</text>'
            f'<text x="0" y="14" font-size="11">{y1:.3g}</text><text x="0" y="{h - 20}" font-size="11">{y0:.3g}</text>'
            f'</svg>')


def report_html(rep: dict) -> str:
    esc = html.escape
    c = rep["training"]["curve"]
    ev = rep.get("evaluation") or {}
    rows = "".join(f"<tr><td>{esc(k)}</td><td>{v['linear_probe']:.3f}</td><td>{v['knn']:.3f}</td>"
                   f"<td>{esc(str(v['effective_rank']))}</td><td>{esc(v['collapse'])}</td></tr>"
                   for k, v in (ev.get("methods") or {}).items())
    hwj = rep.get("hardware") or {}
    dec = "".join(f"<tr><td>{esc(str(x['setting']))}</td><td>{esc(str(x['value']))}</td><td>{esc(str(x['reason']))}</td></tr>"
                  for x in hwj.get("decisions", []))
    return f"""<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; style-src 'unsafe-inline'; img-src data:">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>Run report: {esc(rep['run'])}</title>
<style>body{{font:15px/1.5 system-ui,sans-serif;max-width:820px;margin:2rem auto;padding:0 16px;color:#172033;background:#f5f6f8}}
table{{border-collapse:collapse;width:100%}}td,th{{border-bottom:1px solid #d5dae3;padding:4px 6px;text-align:left;font-size:13px}}
svg{{width:100%;height:auto;background:#fff;border:1px solid #d5dae3;border-radius:6px}}code,pre{{font:12px ui-monospace,monospace}}
@media (prefers-color-scheme:dark){{body{{background:#121826;color:#e8ecf4}}svg{{background:#1b2437}}svg text{{fill:#e8ecf4}}}}</style></head><body>
<h1>Run report: {esc(rep['run'])}</h1>
<p>Generated {esc(rep['generated'])} by jepa-studio {esc(rep['environment']['jepa_studio'])}. Encoder SHA-256:
<code>{esc(str(rep['encoder_sha256']))}</code>{f"; ONNX SHA-256: <code>{esc(rep['onnx_sha256'])}</code>" if rep.get('onnx_sha256') else ''}</p>
<h2>Training</h2><p>{esc(str(rep['training']['steps']))} steps, mean {esc(str(rep['training']['mean_samples_per_s']))} samples/s,
{esc(str(rep['training']['wall_seconds']))} s wall time.</p>
<h3>Total loss</h3>{_svg_line([(p['step'], p['loss']) for p in c], label='total loss')}
<h3>SIGReg term</h3>{_svg_line([(p['step'], p['sigreg']) for p in c], color='#d42f5f', label='SIGReg')}
<h3>Prediction term</h3>{_svg_line([(p['step'], p['pred']) for p in c], color='#3c8fe8', label='prediction')}
<h2>Evaluation</h2>{'<table><tr><th>encoder</th><th>linear probe</th><th>k-NN</th><th>effective rank</th><th>collapse check</th></tr>' + rows + '</table>' if rows else '<p>Not evaluated.</p>'}
<p>Labeled examples: {esc(str(ev.get('n_labeled', '–')))}, held-out from the probe (seen unlabeled in pretraining): {esc(str(ev.get('n_test', '–')))}, chance: {esc(str(ev.get('chance', '–')))}</p>
<h2>Hardware decisions</h2><table><tr><th>setting</th><th>value</th><th>why</th></tr>{dec}</table>
<h2>Config</h2><pre>{esc(json.dumps(rep['config'], indent=2))}</pre>
<h2>Environment</h2><pre>{esc(json.dumps(rep['environment'], indent=2))}</pre>
</body></html>"""


REPRODUCE = """# Reproduce this run

    pip install jepa-studio=={version}          # or: pip install -e . inside the repo
    jepa-studio verify-bundle .                  # checks every SHA-256 in manifest.json
    jepa-studio train --config config.json --run-dir rerun
    jepa-studio eval --run-dir rerun

Same config + same seed + same package versions (environment.json) reproduce the same
numbers on the same device type. GPUs and different BLAS libraries can differ in the last
digits; loss curves should overlap to plotting precision.
"""


def export_bundle(run_dir: str | Path, out: str | Path | None = None, n_embed: int = 2048,
                  onnx: bool = False) -> tuple[Path, Path]:
    """Write report.json/html and the bundle zip. With onnx=True also (re)write encoder.onnx and
    include it in the manifest, the zip and the report (``onnx_sha256``); without it no .onnx file
    is bundled, even if an older one sits in the run folder."""
    d = Path(run_dir)
    # never bundle weights that don't match what the run recorded (corrupted or swapped file)
    from .weights import WeightError, recorded_run_hash
    rec = recorded_run_hash(d)
    if rec and (d / "encoder.safetensors").exists() and sha256_file(d / "encoder.safetensors") != rec:
        raise WeightError(f"{d / 'encoder.safetensors'} does not match the SHA-256 this run recorded; not exporting")
    if onnx:
        export_onnx(d)
    rep = build_report(d)
    if onnx:
        rep["onnx_sha256"] = sha256_file(d / "encoder.onnx")
    atomic_write_bytes(d / "report.json", json.dumps(rep, indent=2).encode())
    atomic_write_bytes(d / "report.html", report_html(rep).encode())
    atomic_write_bytes(d / "environment.json", json.dumps(rep["environment"], indent=2).encode())
    atomic_write_bytes(d / "REPRODUCE.md", REPRODUCE.format(version=__version__).encode())
    if (d / "encoder.safetensors").exists() and not (d / "embeddings.npy").exists():
        try:
            from .config import load_config
            from .data.loaders import build_dataset, build_eval_arrays
            from .evaluate import embed
            from .models import build_encoder
            from .weights import load_run_weights

            cfg = load_config(d / "config.json")
            m = build_encoder(cfg)
            load_run_weights(m, d)
            ds, _ = build_dataset(cfg)
            size = cfg["data"]["image_size"] if cfg["data"]["kind"] != "timeseries" else None
            x, _ = build_eval_arrays(ds, n=n_embed, size=size)
            e, _ = embed(m, x)
            np.save(d / "embeddings.npy", e.numpy(), allow_pickle=False)
        except Exception as ex:  # noqa: BLE001 - bundle still useful without embeddings
            (d / "embeddings.txt").write_text(f"embeddings not exported: {ex}\n")
    names = ["config.json", "hardware.json", "events.jsonl", "eval.json", "encoder.safetensors", "encoder.sha256",
             "embeddings.npy",
             "environment.json", "report.json", "report.html", "REPRODUCE.md"] + (["encoder.onnx"] if onnx else [])
    files = [d / n for n in names if (d / n).exists()]
    manifest = build_manifest(files, d, {"run": d.name, "created": rep["generated"]})
    atomic_write_bytes(d / "manifest.json", json.dumps(manifest, indent=2).encode())
    out = Path(out) if out else d / f"{d.name}-bundle.zip"
    with zipfile.ZipFile(out, "w", compression=zipfile.ZIP_DEFLATED) as z:
        for f in files + [d / "manifest.json"]:
            z.write(f, f.name)
    return out, d / "report.html"


def verify_bundle(folder: str | Path) -> list[str]:
    """Check an unpacked bundle against its manifest (hashes only; bundles are not signed)."""
    d = Path(folder).resolve()
    man = json.loads((d / "manifest.json").read_text())
    problems = []
    for e in man["files"]:
        p = (d / e["path"]).resolve()
        if d not in p.parents:
            problems.append(f"{e['path']}: path escapes bundle")
        elif not p.exists():
            problems.append(f"{e['path']}: missing")
        elif sha256_file(p) != e["sha256"]:
            problems.append(f"{e['path']}: SHA-256 mismatch")
    return problems
