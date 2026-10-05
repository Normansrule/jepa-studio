"""Command line: `jepa-studio <command>`. Every UI action has a CLI equivalent."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def _cfg(args):
    from .config import load_config

    ov = {}
    if getattr(args, "name", None):
        ov["name"] = args.name
    if getattr(args, "data", None):
        ov.setdefault("data", {})["path"] = args.data
    if getattr(args, "kind", None):
        ov.setdefault("data", {})["kind"] = args.kind
    if getattr(args, "epochs", None):
        ov.setdefault("train", {})["epochs"] = args.epochs
    if getattr(args, "mode", None):
        ov.setdefault("hardware", {})["mode"] = args.mode
    return load_config(args.config, ov)


def cmd_hardware(args):
    from . import hardware as hw
    from .config import DEFAULT_CONFIG

    card = hw.probe(".", measure_disk=not args.no_disk)
    if args.json:
        print(json.dumps({"card": json.loads(card.to_json()),
                          "decisions": [d.__dict__ for d in hw.recommend(card, DEFAULT_CONFIG)]}, indent=2))
        return
    c = card
    print(f"""
  ┌─ hardware card ─────────────────────────────────────────────
  │ OS            {c.os}   Python {c.python}   torch {c.torch}
  │ CPU           {c.cpu_model}  ({c.cpu_cores_physical} cores / {c.cpu_cores_logical} threads)
  │ RAM           {c.ram_available_gb:.1f} GB free of {c.ram_total_gb:.1f} GB
  │ accelerator   {c.accelerator}: {c.device_name}{f'  ({c.device_memory_gb} GB)' if c.device_memory_gb else ''}
  │ precision     bf16 {'yes' if c.bf16 else 'no'} · fp16 {'yes' if c.fp16 else 'no'} · torch.compile {'yes' if c.compile_available else 'no'}
  │ disk          {c.disk_free_gb} GB free · write {c.disk_write_mb_s} MB/s · read {c.disk_read_mb_s} MB/s (cached)
  │ temperatures  {c.temps_c or 'no sensors exposed'}
  └─────────────────────────────────────────────────────────────""")
    print("  recommended (balanced mode):")
    for d in hw.recommend(card, DEFAULT_CONFIG):
        print(f"    {d.setting:14s} {str(d.value):8s} {d.reason}")


def cmd_train(args):
    from .train import Trainer

    cfg = _cfg(args)
    tr = Trainer(cfg, run_dir=args.run_dir, runs_root=args.runs)
    print(f"run folder: {tr.run_dir}")
    res = tr.fit(resume=not args.fresh)
    print(json.dumps(res, indent=2))
    if args.eval:
        _evaluate(tr.run_dir)


def _evaluate(run_dir):
    from .config import load_config
    from .data.loaders import build_dataset, build_eval_arrays
    from .evaluate import evaluate_all
    from .models import build_encoder
    from .weights import load_run_weights

    run_dir = Path(run_dir)
    cfg = load_config(run_dir / "config.json")
    m = build_encoder(cfg)
    if not load_run_weights(m, run_dir):
        print(f"note: {run_dir} recorded no SHA-256 for its weights; loaded unverified")
    ds, _ = build_dataset(cfg)
    size = cfg["data"]["image_size"] if cfg["data"]["kind"] != "timeseries" else None
    x, y = build_eval_arrays(ds, size=size)
    res = evaluate_all(cfg, m, x, y, lambda: build_encoder(cfg))
    (run_dir / "eval.json").write_text(json.dumps(res, indent=2))
    return res


def cmd_eval(args):
    _evaluate(args.run_dir)


def cmd_inspect(args):
    from .config import load_config
    from .data.loaders import build_dataset, build_eval_arrays
    from .evaluate import collapse_check, embed, isotropy_report
    from .models import build_encoder
    from .weights import load_run_weights

    d = Path(args.run_dir)
    cfg = load_config(d / "config.json")
    m = build_encoder(cfg)
    if not load_run_weights(m, d):
        print(f"note: {d} recorded no SHA-256 for its weights; loaded unverified")
    ds, _ = build_dataset(cfg)
    x, _ = build_eval_arrays(ds, n=args.n, size=cfg["data"]["image_size"] if cfg["data"]["kind"] != "timeseries" else None)
    _, p = embed(m, x)
    rep = isotropy_report(p)
    rep["verdict"] = collapse_check(rep)
    rep["eigenvalues"] = rep["eigenvalues"][:16]
    print(json.dumps(rep, indent=2))


def cmd_bench(args):
    from .train import benchmark_optimizations

    res = benchmark_optimizations(_cfg(args), steps=args.steps)
    if args.out:
        Path(args.out).write_text(json.dumps(res, indent=2))


def cmd_world(args):
    from .config import load_config
    from .world import EPISODES_BY_ENV, build_site_models, run_world

    if args.site:
        # rebuild the shipped artifacts for this env: site/models/world-<env>.json,
        # assets/world-<env>.gif and runs/world-<env>/ (same recipe as the published ones)
        build_site_models(".", envs=(args.env,))
        return
    over = {"world": {"env": args.env}}
    if not args.config:
        over["world"]["episodes"] = EPISODES_BY_ENV.get(args.env, 400)  # same data budget as the shipped models
    cfg = load_config(args.config, over)
    out = Path(args.out or f"runs/world-{args.env}")
    m = run_world(cfg, out)
    print(json.dumps({k: v for k, v in m.items() if not k.startswith("_") and k != "history"}, indent=2, default=str)[:4000])


def cmd_serve(args):
    from .server import serve

    serve(args.root, args.port, allowed_dirs=args.allow)


def cmd_export(args):
    from .export import export_bundle

    try:
        b, r = export_bundle(args.run_dir, args.out, onnx=args.onnx)
    except (ValueError, RuntimeError, FileNotFoundError) as e:
        if not args.onnx:
            raise
        sys.exit(f"ONNX export refused: {e}")
    print(f"bundle: {b}\nreport: {r}")
    if args.onnx:
        print(f"onnx:   {Path(args.run_dir) / 'encoder.onnx'} (SHA-256 in manifest.json and report.json)")


def cmd_verify_bundle(args):
    from .export import verify_bundle

    probs = verify_bundle(args.folder)
    print("OK: every file matches manifest.json" if not probs else "\n".join(probs))
    sys.exit(1 if probs else 0)


def cmd_keygen(args):
    from .weights import generate_keypair

    sk, pk = generate_keypair()
    Path(args.private).write_text(sk + "\n")
    Path(args.private).chmod(0o600)
    print(f"private key -> {args.private} (keep it secret, e.g. a CI secret)\npublic key: {pk}")


def cmd_sign(args):
    from .weights import build_manifest, sign_manifest

    base = Path(args.folder)
    files = [p for p in sorted(base.rglob("*")) if p.is_file() and p.name not in ("manifest.json",)
             and p.suffix in (".safetensors", ".json", ".onnx") and not p.name.startswith(".")]
    man = build_manifest(files, base, {"name": args.name})
    sk = Path(args.key).read_text().strip() if args.key else __import__("os").environ["JEPA_STUDIO_SIGNING_KEY"]
    (base / "manifest.json").write_text(json.dumps(sign_manifest(man, sk), indent=2) + "\n")
    print(f"signed manifest for {len(files)} files -> {base / 'manifest.json'}")


def cmd_verify_weights(args):
    from .weights import verify_files

    base = Path(args.folder)
    keys = json.loads(Path(args.keys).read_text())["ed25519"]
    probs = verify_files(json.loads((base / "manifest.json").read_text()), base, keys)
    print("OK: signature valid and every hash matches" if not probs else "\n".join(probs))
    sys.exit(1 if probs else 0)


def cmd_import_legacy(args):
    from .weights import import_legacy

    print(f"WARNING: converting {args.src}. Only do this for files from a source you trust.")
    sha = import_legacy(args.src, args.dst)
    print(f"wrote {args.dst} (sha256 {sha})")


def cmd_ask(args):
    from .assistant import answer

    r = answer(" ".join(args.question), args.run_dir)
    print(r["answer"])
    print("\nsources: " + ", ".join(f"{s['file']} — {s['heading']}" for s in r["sources"]))


def cmd_doctor(args):
    import platform

    ok = True
    print(f"python {platform.python_version()}")
    for mod in ("torch", "numpy", "safetensors", "PIL", "psutil", "cryptography"):
        try:
            m = __import__(mod)
            print(f"  ok   {mod} {getattr(m, '__version__', '')}")
        except ImportError:
            ok = False
            print(f"  MISSING {mod}  ->  pip install jepa-studio")
    try:
        import torch
        print(f"  cuda {torch.cuda.is_available()}  mps {getattr(torch.backends, 'mps', None) and torch.backends.mps.is_available()}")
    except Exception:  # noqa: BLE001
        pass
    sys.exit(0 if ok else 1)


def main(argv=None):
    ap = argparse.ArgumentParser(prog="jepa-studio", description="Pretrain, inspect and use LeJEPA models.")
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("hardware", help="benchmark this machine and show recommended settings")
    p.add_argument("--json", action="store_true")
    p.add_argument("--no-disk", action="store_true")
    p.set_defaults(fn=cmd_hardware)

    def cfg_args(p):
        p.add_argument("--config", help="run config JSON (shared with the web app)")
        p.add_argument("--data", help="folder with your images / clips / series")
        p.add_argument("--kind", choices=["synthetic-shapes", "images", "video", "timeseries"])
        p.add_argument("--name")
        p.add_argument("--epochs", type=int)
        p.add_argument("--mode", choices=["max", "balanced", "quiet"])

    p = sub.add_parser("train", help="LeJEPA pretraining")
    cfg_args(p)
    p.add_argument("--run-dir")
    p.add_argument("--runs", default="runs")
    p.add_argument("--fresh", action="store_true", help="ignore existing checkpoints in --run-dir")
    p.add_argument("--eval", action="store_true", help="evaluate when training ends")
    p.set_defaults(fn=cmd_train)

    p = sub.add_parser("eval", help="linear probe + k-NN + baselines")
    p.add_argument("--run-dir", required=True)
    p.set_defaults(fn=cmd_eval)

    p = sub.add_parser("inspect", help="isotropy and collapse report")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--n", type=int, default=2048)
    p.set_defaults(fn=cmd_inspect)

    p = sub.add_parser("bench", help="measure samples/s as each auto-optimization is switched on")
    cfg_args(p)
    p.add_argument("--steps", type=int, default=8)
    p.add_argument("--out")
    p.set_defaults(fn=cmd_bench)

    p = sub.add_parser("world", help="train a world model and evaluate the planning bot")
    p.add_argument("--env", choices=["two-room", "push-block"], default="two-room")
    p.add_argument("--config")
    p.add_argument("--out")
    p.add_argument("--site", action="store_true",
                   help="rebuild site/models/world-<env>.json, assets/world-<env>.gif and runs/world-<env>/")
    p.set_defaults(fn=cmd_world)

    p = sub.add_parser("serve", help="local backend for the desktop app (127.0.0.1 only)")
    p.add_argument("--root", default=".")
    p.add_argument("--port", type=int, default=0)
    p.add_argument("--allow", action="append", default=[], help="folder the app may read data from")
    p.set_defaults(fn=cmd_serve)

    p = sub.add_parser("export", help="run report + reproducibility bundle")
    p.add_argument("--run-dir", required=True)
    p.add_argument("--out")
    p.add_argument("--onnx", action="store_true",
                   help="also write encoder.onnx (image archs; needs pip install \"jepa-studio[onnx]\")")
    p.set_defaults(fn=cmd_export)

    p = sub.add_parser("verify-bundle", help="check an unpacked bundle against its manifest")
    p.add_argument("folder")
    p.set_defaults(fn=cmd_verify_bundle)

    p = sub.add_parser("keygen", help="create an Ed25519 key pair for signing weight manifests")
    p.add_argument("--private", default="signing-key.txt")
    p.set_defaults(fn=cmd_keygen)

    p = sub.add_parser("sign-weights", help="write a signed manifest.json for a weights folder")
    p.add_argument("folder")
    p.add_argument("--key")
    p.add_argument("--name", default="jepa-studio weights")
    p.set_defaults(fn=cmd_sign)

    p = sub.add_parser("verify-weights", help="check manifest signature + every SHA-256")
    p.add_argument("folder")
    p.add_argument("--keys", default=str(Path(__file__).resolve().parent.parent / "weights" / "trusted_keys.json"))
    p.set_defaults(fn=cmd_verify_weights)

    p = sub.add_parser("import-legacy", help="convert a .pt/.pth to safetensors (weights-only, explicit)")
    p.add_argument("src")
    p.add_argument("dst")
    p.set_defaults(fn=cmd_import_legacy)

    p = sub.add_parser("ask", help="ask the docs (retrieval; optional local model)")
    p.add_argument("question", nargs="+")
    p.add_argument("--run-dir")
    p.set_defaults(fn=cmd_ask)

    p = sub.add_parser("doctor", help="check the installation")
    p.set_defaults(fn=cmd_doctor)

    args = ap.parse_args(argv)
    args.fn(args)


if __name__ == "__main__":
    main()
