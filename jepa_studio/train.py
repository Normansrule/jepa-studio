"""LeJEPA pretraining loop with hardware-aware settings, safe checkpoints, pause/resume and a
JSON-lines event stream that the UI (and the run report) reads.

Run directory layout:
    runs/<name>-<timestamp>/
        config.json          exact config used (shared format)
        hardware.json        hardware card + every automatic decision and its reason
        events.jsonl         one JSON object per log step (loss terms, lr, throughput, memory...)
        control.json         write {"command": "pause" | "resume" | "stop"} to steer a live run

Memory guard: at every log step the process RSS (and CUDA allocated memory on a GPU) is compared
with hardware.max_memory_frac of total memory; above it the run checkpoints, emits
{"event": "stopped", "reason": "memory guard: ..."} and ends cleanly (status "stopped").
        ckpt/step-XXXXXXX/   model.safetensors, optim.safetensors, state.json   (last 2 kept)
        encoder.safetensors  final weights
"""
from __future__ import annotations

import json
import math
import random
import shutil
import re
import signal
import threading
import time
from pathlib import Path

import numpy as np
import psutil
import torch
from torch.utils.data import DataLoader

from . import hardware as hw
from .config import save_config
from .data.loaders import build_dataset, collate_views
from .losses import SIGReg, lejepa_loss, random_directions
from .models import build_encoder, count_params
from .weights import (WeightError, atomic_write_bytes, load_tensors, optimizer_from_tensors, optimizer_to_tensors,
                      save_state_dict, save_tensors, write_run_hash)


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed % (2**32))
    torch.manual_seed(seed)


def lr_at(step: int, total: int, base: float, warmup_frac: float, final_ratio: float) -> float:
    """Linear warmup, then cosine decay to base * final_ratio."""
    warm = max(1, int(total * warmup_frac))
    if step < warm:
        return base * (step + 1) / warm
    p = min(1.0, (step - warm) / max(1, total - warm))
    return base * (final_ratio + (1 - final_ratio) * 0.5 * (1 + math.cos(math.pi * p)))


def _worker_context(workers: int):
    """Start method for DataLoader workers. Forking a process that already runs other threads
    (the desktop backend's HTTP server, a training thread, a test runner) can deadlock the child,
    and Python 3.12+ warns about it. Then use 'forkserver' (POSIX) instead; macOS and Windows
    already default to 'spawn'. Single-threaded CLI runs keep the fast default."""
    import multiprocessing as mp
    if workers <= 0 or threading.active_count() <= 1:
        return None
    if "forkserver" in mp.get_all_start_methods():
        ctx = mp.get_context("forkserver")
        # the server process imports these once, so each worker starts as a cheap fork of it
        ctx.set_forkserver_preload(["torch", "numpy", "jepa_studio.data.loaders", "jepa_studio.train"])
        return ctx
    return None


def _worker_init(worker_id: int) -> None:
    info = torch.utils.data.get_worker_info()
    ds = info.dataset
    base = torch.initial_seed() % (2**31)
    ds.views.rng.seed(base + worker_id)


class Trainer:
    def __init__(self, cfg: dict, run_dir: str | Path | None = None, runs_root: str | Path = "runs",
                 log=print, measure_disk: bool = True, memory_reader=None):
        self.cfg = cfg
        # Live memory guard (fit): checked every log step; tests inject a fake reader.
        self.memory_reader = memory_reader or hw.memory_usage
        self.stop_reason: str | None = None
        self.log = log
        stamp = time.strftime("%Y%m%d-%H%M%S")
        if run_dir is None and not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,79}", str(cfg["name"])):
            raise ValueError("config name must be a plain folder name (letters, digits, '.', '_', '-')")
        self.run_dir = Path(run_dir) if run_dir else Path(runs_root) / f"{cfg['name']}-{stamp}"
        self.run_dir.mkdir(parents=True, exist_ok=True)
        self.card = hw.probe(str(self.run_dir), measure_disk=measure_disk)
        self.decisions = hw.recommend(self.card, cfg)
        self.settings = hw.decisions_dict(self.decisions)
        self.device = hw.torch_device(self.card, cfg["hardware"].get("device", "auto"))
        torch.set_num_threads(int(self.settings["torch_threads"]))
        self.stop_requested = False
        self.events_path = self.run_dir / "events.jsonl"
        self.control_path = self.run_dir / "control.json"

    # ------------------------------------------------------------------ helpers
    def emit(self, event: str, **data) -> None:
        rec = {"t": round(time.time(), 3), "event": event, **data}
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec) + "\n")

    def read_control(self) -> str | None:
        try:
            cmd = json.loads(self.control_path.read_text()).get("command")
            return cmd
        except Exception:  # noqa: BLE001
            return None

    def _model(self):
        m = build_encoder(self.cfg).to(self.device)
        if self.settings.get("channels_last") and self.cfg["data"]["kind"] in ("images", "synthetic-shapes"):
            m = m.to(memory_format=torch.channels_last)
        return m

    def _views_to_device(self, views):
        cl = self.settings.get("channels_last") and views[0].dim() == 4
        nb = self.settings.get("pin_memory", False)
        out = []
        for v in views:
            v = v.to(self.device, non_blocking=nb)
            if cl:
                v = v.contiguous(memory_format=torch.channels_last)
            out.append(v)
        return out

    def forward_loss(self, model, sigreg, g_views, l_views):
        """Encode global views together and local views together (sizes differ)."""
        vg = len(g_views)
        emb_g, proj_g = model(torch.cat(g_views))
        n = g_views[0].shape[0]
        proj_g = proj_g.view(vg, n, -1)
        if l_views:
            _, proj_l = model(torch.cat(l_views))
            proj_all = torch.cat([proj_g, proj_l.view(len(l_views), n, -1)])
        else:
            proj_all = proj_g
        return lejepa_loss(proj_g.float(), proj_all.float(), sigreg, self.cfg["objective"]["lambda"]), emb_g

    # ------------------------------------------------------------------ auto batch size
    def pick_batch_size(self, ds) -> int:
        tb = self.cfg["train"]["batch_size"]
        if tb != "auto":
            self.decisions.append(hw.Decision("batch_size", tb, "set in config"))
            return int(tb)
        model = self._model()
        sig = SIGReg(self.cfg["objective"]["num_slices"], self.cfg["objective"]["t_max"],
                     self.cfg["objective"]["knots"]).to(self.device)
        g, loc, _, _ = collate_views([ds[i % len(ds)] for i in range(2)])
        prec = self.settings["precision"]

        def step(b):
            gv = self._views_to_device([x[:1].expand(b, *x.shape[1:]).clone() for x in g])
            lv = self._views_to_device([x[:1].expand(b, *x.shape[1:]).clone() for x in loc])
            with hw.autocast_ctx(self.device, prec):
                out, _ = self.forward_loss(model, sig, gv, lv)
            out.loss.backward()
            model.zero_grad(set_to_none=True)

        cap = {"max": 1024, "balanced": 512, "quiet": 128}[self.cfg["hardware"]["mode"]]
        cap = min(cap, len(ds))
        if self.device.type == "cpu":
            cap = min(cap, 256)  # bigger CPU batches rarely raise samples/s
        best, trace = hw.memory_probe_batch(step, 16, cap, self.device,
                                            self.cfg["hardware"].get("max_memory_frac", 0.85))
        self.decisions.append(hw.Decision("batch_size", best,
                                          f"largest power of two that fit under {int(100 * self.cfg['hardware'].get('max_memory_frac', .85))}% memory (probe: {trace})"))
        return best

    # ------------------------------------------------------------------ checkpoints
    def save_checkpoint(self, model, opt, scaler, step, epoch, extra: dict | None = None) -> Path:
        d = self.run_dir / "ckpt" / f"step-{step:07d}"
        d.mkdir(parents=True, exist_ok=True)
        model_sha = save_state_dict(model, d / "model.safetensors")
        ot, os_ = optimizer_to_tensors(opt)
        optim_sha = save_tensors(ot, d / "optim.safetensors")
        state = {"step": step, "epoch": epoch, "optim_scalars": os_,
                 "sha256": {"model.safetensors": model_sha, "optim.safetensors": optim_sha},
                 "scaler": scaler.state_dict() if scaler is not None and scaler.is_enabled() else None,
                 "python_random": None, "torch_rng": torch.get_rng_state().tolist(),
                 "numpy_rng": None, **(extra or {})}
        atomic_write_bytes(d / "state.json", json.dumps(state).encode())
        atomic_write_bytes(self.run_dir / "ckpt" / "LATEST", d.name.encode())
        ckpts = sorted((self.run_dir / "ckpt").glob("step-*"))
        for old in ckpts[:-2]:
            shutil.rmtree(old, ignore_errors=True)
        return d

    def load_checkpoint(self, model, opt, scaler):
        latest = self.run_dir / "ckpt" / "LATEST"
        if not latest.exists():
            return 0, 0
        d = self.run_dir / "ckpt" / latest.read_text().strip()
        if not re.fullmatch(r"step-\d{7}", d.name):
            raise WeightError(f"{latest}: unexpected checkpoint name {d.name!r}")
        st = json.loads((d / "state.json").read_text())
        hashes = st.get("sha256") or {}      # checkpoints written before v0.2 have none
        model.load_state_dict(load_tensors(d / "model.safetensors", hashes.get("model.safetensors")))
        optimizer_from_tensors(opt, load_tensors(d / "optim.safetensors", hashes.get("optim.safetensors")),
                               st["optim_scalars"])
        if st.get("scaler") and scaler is not None:
            scaler.load_state_dict(st["scaler"])
        torch.set_rng_state(torch.tensor(st["torch_rng"], dtype=torch.uint8))
        self.log(f"resumed from {d.name}")
        return st["step"], st["epoch"]

    def check_disk(self) -> None:
        used = sum(f.stat().st_size for f in self.run_dir.rglob("*") if f.is_file()) / 2**30
        if used > self.cfg["hardware"].get("max_disk_gb", 5):
            raise RuntimeError(f"run folder uses {used:.2f} GB, above max_disk_gb; stopping safely")
        if shutil.disk_usage(self.run_dir).free < 1 * 2**30:
            raise RuntimeError("less than 1 GB free on disk; stopping safely")

    # ------------------------------------------------------------------ main loop
    def fit(self, resume: bool = True) -> dict:
        cfg = self.cfg
        seed_everything(cfg["seed"])
        save_config(cfg, self.run_dir / "config.json")
        ds, info = build_dataset(cfg)
        self.emit("data", **{k: v for k, v in info.items() if k != "skipped"}, skipped=len(info.get("skipped", [])))
        batch = self.pick_batch_size(ds)
        eff = cfg["train"].get("effective_batch") or batch
        accum = max(1, round(eff / batch))
        # One optimizer step never spans two epochs: with a small dataset (fewer batches per epoch
        # than the accumulation count) cap accum at the batches available, or no step would ever finish.
        n_batches = len(ds) // batch if len(ds) >= batch else 1
        why = f"effective batch {batch}x{accum}={batch * accum} (target {eff})"
        if accum > n_batches:
            accum = n_batches
            why = (f"effective batch {batch}x{accum}={batch * accum}: target {eff} capped because the dataset "
                   f"has only {n_batches} batch(es) of {batch} per epoch")
        self.decisions.append(hw.Decision("grad_accumulation", accum, why))
        self.settings = hw.decisions_dict(self.decisions)
        atomic_write_bytes(self.run_dir / "hardware.json", json.dumps(
            {"card": json.loads(self.card.to_json()),
             "decisions": [d.__dict__ for d in self.decisions]}, indent=2, default=str).encode())
        for d in self.decisions:
            self.log(f"  [auto] {d.setting} = {d.value}  ({d.reason[:110]})")

        workers = int(self.settings["workers"])
        loader = DataLoader(ds, batch_size=batch, shuffle=True, drop_last=len(ds) >= batch,
                            num_workers=workers, collate_fn=collate_views,
                            multiprocessing_context=_worker_context(workers),
                            pin_memory=bool(self.settings.get("pin_memory")),
                            persistent_workers=workers > 0, worker_init_fn=_worker_init if workers else None,
                            generator=torch.Generator().manual_seed(cfg["seed"]))
        model = self._model()
        if self.settings.get("compile"):
            model = torch.compile(model)
        sigreg = SIGReg(cfg["objective"]["num_slices"], cfg["objective"]["t_max"], cfg["objective"]["knots"],
                        seed=cfg["seed"]).to(self.device)
        opt = torch.optim.AdamW(model.parameters(), lr=cfg["train"]["lr"],
                                weight_decay=cfg["train"].get("weight_decay", 0.05))
        prec = self.settings["precision"]
        scaler = torch.amp.GradScaler(self.device.type, enabled=(prec == "fp16" and self.device.type == "cuda"))
        steps_per_epoch = max(1, len(loader) // accum)
        total = cfg["train"].get("max_steps") or steps_per_epoch * cfg["train"]["epochs"]
        step, epoch0 = self.load_checkpoint(model, opt, scaler) if resume else (0, 0)
        self.emit("start", total_steps=total, batch=batch, accum=accum, params=count_params(model),
                  device=str(self.device), precision=prec)
        self.log(f"training {count_params(model):,} params on {self.device} ({prec}), {total} steps")

        throttle = hw.Throttle(cfg["hardware"]["mode"], cfg["hardware"].get("max_temp_c", 85))
        probe_dirs = random_directions(cfg["model"]["proj_dim"], 3, torch.Generator().manual_seed(1234),
                                       device=self.device)
        fixed_batch = collate_views([ds[i] for i in range(min(256, len(ds)))])[0][0]

        def handle_term(signum, frame):  # noqa: ARG001
            self.stop_requested = True
        # CLI runs (main thread): SIGTERM -> finish the step, checkpoint, exit cleanly. Runs started by
        # the desktop backend live in a worker thread, where Python forbids signal handlers; there the
        # backend's control file (pause/stop) and the periodic checkpoints cover the same need.
        in_main = threading.current_thread() is threading.main_thread()
        old = signal.signal(signal.SIGTERM, handle_term) if in_main and hasattr(signal, "SIGTERM") else None

        log_every = cfg["train"].get("log_every", 10)
        max_mem = cfg["hardware"].get("max_memory_frac", 0.85)
        ck_every = cfg["train"].get("checkpoint_every", 200)
        proc = psutil.Process()
        t_last, seen_last = time.perf_counter(), 0
        seen = 0
        result = {"status": "running"}
        try:
            epoch = epoch0
            while step < total and not self.stop_requested:
                model.train()
                micro = 0
                for g, loc, _, _ in loader:
                    t0 = time.perf_counter()
                    cmd = self.read_control()
                    if cmd in ("pause", "stop"):
                        self.save_checkpoint(model, opt, scaler, step, epoch)
                        self.emit("paused" if cmd == "pause" else "stopped", step=step)
                        if cmd == "stop":
                            self.stop_requested = True
                            break
                        while self.read_control() == "pause":
                            time.sleep(0.5)
                        self.emit("resumed", step=step)
                    for grp in opt.param_groups:
                        grp["lr"] = lr_at(step, total, cfg["train"]["lr"], cfg["train"].get("warmup_frac", 0.1),
                                          cfg["train"].get("final_lr_ratio", 1e-3))
                    gv, lv = self._views_to_device(g), self._views_to_device(loc)
                    with hw.autocast_ctx(self.device, prec):
                        out, _ = self.forward_loss(model, sigreg, gv, lv)
                    scaler.scale(out.loss / accum).backward()
                    micro += 1
                    seen += g[0].shape[0]
                    if micro % accum:
                        continue
                    scaler.step(opt)
                    scaler.update()
                    opt.zero_grad(set_to_none=True)
                    step += 1
                    throttle.after_step(time.perf_counter() - t0)
                    if not math.isfinite(out.loss.item()):
                        raise FloatingPointError(f"loss became {out.loss.item()} at step {step}")
                    if step % log_every == 0 or step == 1 or step == total:
                        now = time.perf_counter()
                        sps = (seen - seen_last) / max(1e-9, now - t_last)
                        t_last, seen_last = now, seen
                        rec = dict(step=step, epoch=epoch, loss=round(out.loss.item(), 6),
                                   pred=round(out.prediction.item(), 6), sigreg=round(out.sigreg.item(), 6),
                                   lr=opt.param_groups[0]["lr"], samples_per_s=round(sps, 1),
                                   cpu_pct=psutil.cpu_percent(None), rss_gb=round(proc.memory_info().rss / 2**30, 3))
                        if self.device.type == "cuda":
                            rec["gpu_mem_gb"] = round(torch.cuda.memory_allocated() / 2**30, 3)
                        if step % (log_every * 5) == 0 or step == 1:
                            rec["proj_hist"] = self.projection_histograms(model, fixed_batch, probe_dirs)
                        self.emit("step", **rec)
                        self.log(f"step {step:6d}/{total} loss {rec['loss']:.4f} pred {rec['pred']:.4f} "
                                 f"sigreg {rec['sigreg']:.4f} lr {rec['lr']:.2e} {sps:7.1f} samples/s")
                        reason = hw.memory_breach(self.memory_reader(self.device), max_mem)
                        if reason:
                            self.save_checkpoint(model, opt, scaler, step, epoch)
                            self.stop_reason = reason
                            self.stop_requested = True
                            self.emit("stopped", step=step, reason=reason)
                            self.log(f"stopping: {reason}")
                            break
                    if step % ck_every == 0:
                        self.save_checkpoint(model, opt, scaler, step, epoch)
                        self.check_disk()
                    if step >= total or self.stop_requested:
                        break
                epoch += 1
            final = model._orig_mod if hasattr(model, "_orig_mod") else model
            sha = save_state_dict(final, self.run_dir / "encoder.safetensors",
                                  {"arch": cfg["model"]["arch"], "step": step})
            write_run_hash(self.run_dir, sha)
            self.save_checkpoint(model, opt, scaler, step, epoch)
            result = {"status": "stopped" if self.stop_requested else "done", "steps": step,
                      "encoder_sha256": sha, "throttle_events": throttle.events}
            if self.stop_reason:
                result["reason"] = self.stop_reason
            self.emit("end", **result)
        finally:
            if old is not None:
                signal.signal(signal.SIGTERM, old)
        self.model = model._orig_mod if hasattr(model, "_orig_mod") else model
        self.dataset = ds
        return result

    @torch.no_grad()
    def projection_histograms(self, model, batch, dirs, bins: int = 24) -> list:
        """1-D projections of the projector output along fixed directions (for the SIGReg
        animation). Histogram over [-4, 4] of standardized counts."""
        was = model.training
        model.eval()
        _, proj = model(self._views_to_device([batch])[0])
        model.train(was)
        p = (proj.float() @ dirs).cpu()
        edges = torch.linspace(-4, 4, bins + 1)
        out = []
        for j in range(p.shape[1]):
            h = torch.histc(p[:, j].clamp(-4, 4), bins=bins, min=-4, max=4)
            out.append([round(float(x), 4) for x in (h / h.sum() / (edges[1] - edges[0]))])
        return out


def benchmark_optimizations(cfg: dict, steps: int = 8, log=print) -> list[dict]:
    """Measure samples/s as each automatic optimization is switched on, one at a time and
    cumulatively, from a plain fp32 / batch-32 / 0-worker baseline. Used for the Train tab's
    before/after chart and the README throughput figure."""
    import copy

    tr = Trainer(cfg, run_dir=Path("runs") / "_bench", log=lambda *_: None, measure_disk=False)
    base = copy.deepcopy(cfg)
    ds, _ = build_dataset(base)
    auto = dict(tr.settings)
    auto_bs = tr.pick_batch_size(ds) if cfg["train"]["batch_size"] == "auto" else cfg["train"]["batch_size"]
    stages = [("baseline (fp32, batch 32, 0 workers)", dict(precision="fp32", channels_last=False, workers=0,
                                                          pin_memory=False), 32)]
    cur = dict(stages[0][1])
    cur_bs = 32
    if auto_bs != 32:
        cur_bs = auto_bs
        stages.append((f"+ auto batch size ({auto_bs})", dict(cur), cur_bs))
    for key in ("precision", "channels_last", "workers", "pin_memory"):
        if auto.get(key) != cur.get(key):
            cur[key] = auto[key]
            stages.append((f"+ {key} = {auto[key]}", dict(cur), cur_bs))
    results = []
    for label, sets, bs in stages:
        tr.settings.update(sets)
        model = tr._model()
        sig = SIGReg(cfg["objective"]["num_slices"]).to(tr.device)
        opt = torch.optim.AdamW(model.parameters(), lr=1e-4)
        dl = DataLoader(ds, batch_size=bs, shuffle=True, drop_last=True, num_workers=int(sets["workers"]),
                        collate_fn=collate_views, pin_memory=bool(sets["pin_memory"]),
                        worker_init_fn=_worker_init if int(sets["workers"]) else None)
        it = iter(dl)
        n, t0 = 0, None
        for i in range(steps + 2):
            try:
                g, loc, _, _ = next(it)
            except StopIteration:
                it = iter(dl)
                g, loc, _, _ = next(it)
            if i == 2:
                t0 = time.perf_counter()  # skip warmup steps
            with hw.autocast_ctx(tr.device, sets["precision"]):
                out, _ = tr.forward_loss(model, sig, tr._views_to_device(g), tr._views_to_device(loc))
            out.loss.backward()
            opt.step()
            opt.zero_grad(set_to_none=True)
            if i >= 2:
                n += g[0].shape[0]
        sps = n / (time.perf_counter() - t0)
        results.append({"stage": label, "samples_per_s": round(sps, 1), "settings": sets, "batch": bs})
        log(f"{label:45s} {sps:8.1f} samples/s")
    shutil.rmtree(Path("runs") / "_bench", ignore_errors=True)
    return results
