"""Local backend for the desktop app (docs/api.md). Standard library HTTP server only.

Security model (docs/SECURITY_MODEL.md, "local privilege boundaries"):
  * binds 127.0.0.1, never a public interface
  * per-session random token required in X-JEPA-Token (constant-time compare)
  * Host header must be 127.0.0.1:<port> or localhost:<port>  (DNS-rebinding defense)
  * Origin, when present, must be the desktop shell's origin (tauri://localhost,
    http(s)://tauri.localhost) or the same host; no wildcard CORS
  * JSON bodies only, capped at 2 MB; data paths must be inside the folders the user granted
    (JEPA_STUDIO_ALLOWED_DIRS, set by the shell from the folder picker)
  * no endpoint runs user-supplied code or reaches the network
"""
from __future__ import annotations

import base64
import hmac
import io
import json
import os
import re
import secrets
import sys
import threading
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import parse_qs, urlparse

import numpy as np
import torch

from . import __version__
from . import hardware as hw
from .config import ConfigError, DEFAULT_CONFIG, deep_merge, load_schema, validate
from .weights import WeightError

MAX_BODY = 2 * 1024 * 1024
RUN_ID = re.compile(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}")  # one folder name under <root>/runs
ALLOWED_ORIGINS = {"tauri://localhost", "http://tauri.localhost", "https://tauri.localhost"}


class ApiError(Exception):
    def __init__(self, status: int, msg: str):
        super().__init__(msg)
        self.status = status


def png_data_url(img: torch.Tensor) -> str:
    """(C,H,W) or (T,C,H,W) float [0,1] -> PNG data URL (first frame for clips)."""
    from PIL import Image

    if img.dim() == 4:
        img = img[0]
    if img.dim() == 2:  # time series window (C, L) -> tiny line plot is done client side
        raise ValueError("not an image")
    arr = (img.clamp(0, 1).permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)
    if arr.shape[-1] == 1:
        arr = arr[..., 0]
    buf = io.BytesIO()
    Image.fromarray(arr).save(buf, format="PNG")
    return "data:image/png;base64," + base64.b64encode(buf.getvalue()).decode()


def _index(q: dict, n: int) -> int:
    """The ?i= sample index, checked against the n samples the run's arrays hold."""
    i = int(q.get("i", ["0"])[0])
    if not 0 <= i < n:
        raise ApiError(400, "index out of range")
    return i


class Studio:
    """Holds runs in memory and on disk (runs/<id>)."""

    def __init__(self, root: Path, allowed_dirs: list[Path], allowed_file: Path | None = None):
        self.root = root
        self._static_allowed = [p.resolve() for p in allowed_dirs]
        self.allowed_file = allowed_file
        self.runs: dict[str, dict] = {}
        self.lock = threading.Lock()
        (root / "runs").mkdir(parents=True, exist_ok=True)

    # -------------------------------------------------------------- config / paths
    @property
    def allowed(self) -> list[Path]:
        """Folders the user granted. The desktop shell appends a line to JEPA_STUDIO_ALLOWED_FILE
        each time the user picks a folder in the native dialog; the web view cannot write it."""
        extra = []
        if self.allowed_file and self.allowed_file.exists():
            extra = [Path(x.strip()).expanduser().resolve() for x in self.allowed_file.read_text().splitlines()
                     if x.strip()]
        return self._static_allowed + extra

    def checked_config(self, body: dict) -> dict:
        cfg = deep_merge(DEFAULT_CONFIG, body.get("config") or {})
        errs = validate(cfg, load_schema())
        if errs:
            raise ApiError(400, "invalid config: " + "; ".join(errs[:10]))
        path = cfg["data"].get("path")
        if cfg["data"]["kind"] != "synthetic-shapes":
            if not path:
                raise ApiError(400, "data.path is required for this data kind")
            p = Path(path).expanduser().resolve()
            if not any(p == a or a in p.parents for a in self.allowed):
                raise ApiError(403, "data.path is outside the folders you granted to jepa-studio")
        return cfg

    def run(self, rid: str) -> dict:
        r = self.runs.get(rid)
        if r is None:
            # run ids are single folder names; anything else (.., separators, drive letters) is refused
            # before it touches the file system
            if not RUN_ID.fullmatch(rid) or rid in (".", ".."):
                raise ApiError(404, "unknown run")
            d = self.root / "runs" / rid
            if not (d / "config.json").exists():
                raise ApiError(404, "unknown run")
            r = {"run_id": rid, "dir": d, "status": "finished", "trainer": None, "kind": "pretrain"}
            self.runs[rid] = r
        return r

    # -------------------------------------------------------------- endpoints
    def hardware(self, q) -> dict:
        card = hw.probe(str(self.root), measure_disk=q.get("disk", ["0"])[0] == "1")
        return {"card": json.loads(card.to_json()),
                "decisions": [d.__dict__ for d in hw.recommend(card, DEFAULT_CONFIG)]}

    def preview(self, body) -> dict:
        from .data.loaders import build_dataset

        cfg = self.checked_config(body)
        cfg["data"]["max_items"] = min(cfg["data"].get("max_items", 64), 64)
        ds, info = build_dataset(cfg)
        n = min(int(body.get("n", 4)), 8, len(ds))
        samples = []
        for i in range(n):
            raw = ds.raw(i)
            if raw.dim() == 2:  # time series
                g, loc = ds.views(raw)
                samples.append({"series": raw.tolist(), "globals": [v.tolist() for v in g],
                                "locals": [v.tolist() for v in loc]})
                continue
            g, loc = ds.views(raw.float())
            samples.append({"original": png_data_url(raw.float()), "globals": [png_data_url(v) for v in g],
                            "locals": [png_data_url(v) for v in loc]})
        return {"info": info, "samples": samples}

    def start_run(self, body) -> dict:
        from .train import Trainer

        cfg = self.checked_config(body)
        rid = f"{cfg['name']}-{secrets.token_hex(3)}"
        d = self.root / "runs" / rid
        if not RUN_ID.fullmatch(rid) or d.resolve().parent != (self.root / "runs").resolve():
            raise ApiError(400, "config.name must be a plain folder name (letters, digits, . _ -)")
        tr = Trainer(cfg, run_dir=d, log=lambda *_: None)
        rec = {"run_id": rid, "dir": d, "status": "running", "trainer": tr, "kind": "pretrain", "cfg": cfg}

        def work():
            try:
                res = tr.fit(resume=False)
                rec["status"] = res["status"]
            except Exception as e:  # noqa: BLE001
                rec["status"] = "failed"
                tr.emit("error", message=str(e)[:500])
        threading.Thread(target=work, daemon=True).start()
        with self.lock:
            self.runs[rid] = rec
        return {"run_id": rid}

    def list_runs(self) -> dict:
        out = []
        for d in sorted((self.root / "runs").glob("*")):
            if not (d / "config.json").exists():
                continue
            r = self.runs.get(d.name, {})
            step = total = None
            ev = d / "events.jsonl"
            if ev.exists():
                for line in ev.read_text().splitlines()[-50:]:
                    rec = json.loads(line)
                    step = rec.get("step", step)
                    total = rec.get("total_steps", total)
            out.append({"run_id": d.name, "name": json.loads((d / "config.json").read_text())["name"],
                        "status": r.get("status", "finished"), "step": step, "total": total})
        return {"runs": out}

    def events(self, rid, q) -> dict:
        r = self.run(rid)
        since = int(q.get("since", ["0"])[0])
        p = r["dir"] / "events.jsonl"
        lines = p.read_text().splitlines() if p.exists() else []
        return {"events": [json.loads(x) for x in lines[since:since + 500]], "next": min(len(lines), since + 500),
                "status": r["status"]}

    def control(self, rid, body) -> dict:
        r = self.run(rid)
        cmd = body.get("command")
        if cmd not in ("pause", "resume", "stop"):
            raise ApiError(400, "command must be pause, resume or stop")
        (r["dir"] / "control.json").write_text(json.dumps({"command": cmd}))
        return {"ok": True}

    def _model_and_data(self, r):
        from .config import load_config
        from .data.loaders import build_dataset, build_eval_arrays
        from .models import build_encoder
        from .weights import load_run_weights

        if r.get("model") is None:
            cfg = load_config(r["dir"] / "config.json")
            model = build_encoder(cfg)
            r["weights_verified"] = load_run_weights(model, r["dir"])
            ds, info = build_dataset(cfg)
            size = cfg["data"]["image_size"] if cfg["data"]["kind"] in ("images", "synthetic-shapes", "video") else None
            x, y = build_eval_arrays(ds, n=min(len(ds), 4096), size=size)
            r.update(model=model.eval(), cfg=cfg, x=x, y=y, dataset=ds, classes=info.get("classes"))
        return r["model"], r["cfg"], r["x"], r["y"]

    def evaluate(self, rid) -> dict:
        from .evaluate import evaluate_all
        from .models import build_encoder

        r = self.run(rid)
        model, cfg, x, y = self._model_and_data(r)
        res = evaluate_all(cfg, model, x, y, lambda: build_encoder(cfg), log=lambda *_: None)
        (r["dir"] / "eval.json").write_text(json.dumps(res, indent=2))
        return res

    def embeddings(self, rid, q) -> dict:
        from .evaluate import collapse_check, embed, isotropy_report, pca

        r = self.run(rid)
        model, cfg, x, y = self._model_and_data(r)
        n = min(int(q.get("n", ["1000"])[0]), len(x), 4000)
        e, p = embed(model, x[:n])
        coords, var = pca(e, 3)
        iso = isotropy_report(p)
        thumbs = [png_data_url(x[i]) for i in range(min(n, 64))] if x.dim() >= 4 else []
        r["emb"] = e
        classes = r.get("classes")
        return {"pca3": coords.tolist(), "var": var.tolist(), "labels": y[:n].tolist(), "classes": classes,
                "isotropy": iso, "collapse": collapse_check(iso), "thumbs": thumbs}

    def neighbors(self, rid, q) -> dict:
        from .evaluate import embed, nearest_neighbors

        r = self.run(rid)
        model, cfg, x, y = self._model_and_data(r)
        if r.get("emb") is None:
            r["emb"], _ = embed(model, x)
        i = _index(q, len(r["emb"]))
        k = min(int(q.get("k", ["8"])[0]), 64)
        return {"query": i, "neighbors": nearest_neighbors(r["emb"], i, k)}

    def saliency(self, rid, q) -> dict:
        """Gradient saliency for one sample, plus [CLS] attention for vit-tiny (jepa_studio/maps.py)."""
        from .maps import saliency, vit_attention
        from .models import ViTTiny

        r = self.run(rid)
        model, cfg, x, _ = self._model_and_data(r)
        if x.dim() != 4:
            raise ApiError(400, f"saliency maps are for image runs; this run's data is {cfg['data']['kind']}")
        i = _index(q, len(x))
        xi = x[i].float()
        m = saliency(model, xi)
        out = {"index": i, "kind": "gradient", "side": int(m.shape[-1]), "map": m.tolist(),
               "image": png_data_url(xi)}
        if isinstance(model.backbone, ViTTiny):
            a, grid = vit_attention(model, xi)
            out["attention"] = {"kind": "cls-attention", "map": a.tolist(), "side": grid[0], "grid": list(grid)}
        return out

    def benchmark(self, body) -> dict:
        from .train import benchmark_optimizations

        cfg = self.checked_config(body)
        return {"stages": benchmark_optimizations(cfg, log=lambda *_: None)}

    def world(self, body) -> dict:
        from .world import run_world

        cfg = self.checked_config(body)
        rid = f"world-{cfg['world']['env']}-{secrets.token_hex(3)}"
        d = self.root / "runs" / rid
        d.mkdir(parents=True)
        (d / "config.json").write_text(json.dumps(cfg, indent=2))
        rec = {"run_id": rid, "dir": d, "status": "running", "kind": "world", "cfg": cfg}

        def work():
            try:
                rec["result"] = run_world(cfg, d, log=lambda *_: None)
                rec["status"] = "done"
            except Exception as e:  # noqa: BLE001
                rec["status"] = "failed"
                rec["error"] = str(e)[:500]
        threading.Thread(target=work, daemon=True).start()
        self.runs[rid] = rec
        return {"run_id": rid}

    def plan(self, rid, body) -> dict:
        from .world import load_world_run, plan_for_api

        r = self.run(rid)
        if r.get("status") == "running":
            raise ApiError(409, "world model is still training")
        seed = int(body.get("seed", 0))
        return plan_for_api(load_world_run(r["dir"]), seed)

    def export(self, rid) -> dict:
        from .export import export_bundle

        r = self.run(rid)
        bundle, report = export_bundle(r["dir"])
        return {"bundle": str(bundle), "report": str(report)}

    def assistant(self, body) -> dict:
        from .assistant import answer

        q = str(body.get("question", ""))[:2000]
        run_dir = self.run(body["run_id"])["dir"] if body.get("run_id") else None
        return answer(q, run_dir)


def make_handler(studio: Studio, token: str, port_ref: list):
    class Handler(BaseHTTPRequestHandler):
        server_version = "jepa-studio"
        sys_version = ""

        def log_message(self, fmt, *args):  # quiet by default; no request bodies in logs
            pass

        def _host_ok(self) -> bool:
            host = self.headers.get("Host", "")
            return host in (f"127.0.0.1:{port_ref[0]}", f"localhost:{port_ref[0]}")

        def _origin(self) -> str | None:
            o = self.headers.get("Origin")
            if o is None:
                return None
            if o in ALLOWED_ORIGINS or o in (f"http://127.0.0.1:{port_ref[0]}", f"http://localhost:{port_ref[0]}"):
                return o
            return ""

        def _send(self, status: int, obj) -> None:
            data = json.dumps(obj).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(data)))
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Content-Type-Options", "nosniff")
            o = self._origin()
            if o:
                self.send_header("Access-Control-Allow-Origin", o)
                self.send_header("Vary", "Origin")
            self.end_headers()
            self.wfile.write(data)

        def do_OPTIONS(self):  # noqa: N802 - CORS preflight from the desktop webview
            o = self._origin()
            if not self._host_ok() or not o:
                self.send_response(403)
                self.end_headers()
                return
            self.send_response(204)
            self.send_header("Access-Control-Allow-Origin", o)
            self.send_header("Access-Control-Allow-Methods", "GET, POST")
            self.send_header("Access-Control-Allow-Headers", "Content-Type, X-JEPA-Token")
            self.send_header("Access-Control-Max-Age", "600")
            self.end_headers()

        def _guard(self) -> None:
            if not self._host_ok():
                raise ApiError(421, "bad Host header")
            if self._origin() == "":
                raise ApiError(403, "origin not allowed")
            got = self.headers.get("X-JEPA-Token", "")
            if not hmac.compare_digest(got.encode(), token.encode()):
                raise ApiError(401, "missing or wrong session token")

        def _body(self) -> dict:
            try:
                n = int(self.headers.get("Content-Length") or 0)
            except ValueError as e:
                raise ApiError(400, "bad Content-Length") from e
            if n < 0:
                raise ApiError(400, "bad Content-Length")
            if n > MAX_BODY:
                raise ApiError(413, "request body too large")
            if n and (self.headers.get("Content-Type") or "").split(";")[0].strip().lower() != "application/json":
                raise ApiError(415, "JSON only")
            raw = self.rfile.read(n) if n else b"{}"
            try:
                body = json.loads(raw or b"{}")
            except json.JSONDecodeError as e:
                raise ApiError(400, f"bad JSON: {e}") from e
            if not isinstance(body, dict):
                raise ApiError(400, "JSON object expected")
            return body

        def _dispatch(self, method: str):
            try:
                self._guard()
                u = urlparse(self.path)
                q = parse_qs(u.query)
                parts = [p for p in u.path.split("/") if p]
                if parts[:1] != ["api"]:
                    raise ApiError(404, "not found")
                p = parts[1:]
                body = self._body() if method == "POST" else {}
                s = studio
                routes = {
                    ("GET", ("health",)): lambda: {"ok": True, "version": __version__},
                    ("GET", ("hardware",)): lambda: s.hardware(q),
                    ("POST", ("config", "validate")): lambda: {"errors": validate(
                        deep_merge(DEFAULT_CONFIG, body.get("config") or {}), load_schema())},
                    ("POST", ("data", "preview")): lambda: s.preview(body),
                    ("POST", ("runs",)): lambda: s.start_run(body),
                    ("GET", ("runs",)): s.list_runs,
                    ("POST", ("benchmark",)): lambda: s.benchmark(body),
                    ("POST", ("world",)): lambda: s.world(body),
                    ("POST", ("assistant",)): lambda: s.assistant(body),
                }
                key = (method, tuple(p))
                if key in routes:
                    return self._send(200, routes[key]())
                if len(p) == 3 and p[0] == "runs":
                    rid, action = p[1], p[2]
                    table = {("GET", "events"): lambda: s.events(rid, q),
                             ("POST", "control"): lambda: s.control(rid, body),
                             ("POST", "evaluate"): lambda: s.evaluate(rid),
                             ("GET", "embeddings"): lambda: s.embeddings(rid, q),
                             ("GET", "neighbors"): lambda: s.neighbors(rid, q),
                             ("GET", "saliency"): lambda: s.saliency(rid, q),
                             ("POST", "export"): lambda: s.export(rid)}
                    if (method, action) in table:
                        return self._send(200, table[(method, action)]())
                if len(p) == 3 and p[0] == "world" and p[2] == "plan" and method == "POST":
                    return self._send(200, s.plan(p[1], body))
                raise ApiError(404, "not found")
            except ApiError as e:
                self._send(e.status, {"error": str(e)})
            except WeightError as e:
                self._send(409, {"error": str(e)[:500]})
            except (ConfigError, ValueError, FileNotFoundError) as e:
                self._send(400, {"error": str(e)[:500]})
            except Exception as e:  # noqa: BLE001
                traceback.print_exc(file=sys.stderr)
                self._send(500, {"error": f"internal error: {type(e).__name__}"})

        def do_GET(self):  # noqa: N802
            self._dispatch("GET")

        def do_POST(self):  # noqa: N802
            self._dispatch("POST")

    return Handler


def serve(root: str = ".", port: int = 0, token: str | None = None, allowed_dirs: list[str] | None = None,
          ready_file=sys.stdout) -> None:
    """Start the backend. Prints one JSON line {"ready": true, "port": N} for the shell."""
    token = token or os.environ.get("JEPA_STUDIO_TOKEN") or secrets.token_urlsafe(32)
    env_dirs = [d for d in os.environ.get("JEPA_STUDIO_ALLOWED_DIRS", "").split(os.pathsep) if d]
    dirs = [Path(d).expanduser() for d in (allowed_dirs or []) + env_dirs]
    af = os.environ.get("JEPA_STUDIO_ALLOWED_FILE")
    studio = Studio(Path(root).resolve(), dirs, Path(af) if af else None)
    port_ref = [port]
    httpd = ThreadingHTTPServer(("127.0.0.1", port), make_handler(studio, token, port_ref))
    port_ref[0] = httpd.server_address[1]
    msg = {"ready": True, "port": port_ref[0]}
    if not os.environ.get("JEPA_STUDIO_TOKEN") and token:
        msg["token_hint"] = "token printed once; pass it as X-JEPA-Token"
        msg["token"] = token
    print(json.dumps(msg), file=ready_file, flush=True)
    try:
        httpd.serve_forever()
    finally:
        httpd.server_close()
