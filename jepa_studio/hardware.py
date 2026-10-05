"""Hardware awareness: probe the machine, recommend settings, and measure their effect.

Nothing here changes system settings. Every decision is returned with the reason for it so
the Train tab (and the run report) can show *why* a setting was chosen.
"""
from __future__ import annotations

import json
import os
import platform
import shutil
import subprocess
import tempfile
import time
from dataclasses import asdict, dataclass, field

import psutil
import torch


@dataclass
class HardwareCard:
    os: str
    python: str
    torch: str
    cpu_model: str
    cpu_cores_physical: int
    cpu_cores_logical: int
    ram_total_gb: float
    ram_available_gb: float
    accelerator: str                  # cuda | rocm | mps | cpu
    device_name: str
    device_memory_gb: float | None
    bf16: bool
    fp16: bool
    compile_available: bool
    disk_free_gb: float
    disk_write_mb_s: float | None
    disk_read_mb_s: float | None
    temps_c: dict = field(default_factory=dict)
    notes: list = field(default_factory=list)

    def to_json(self) -> str:
        return json.dumps(asdict(self), indent=2)


def _cpu_model() -> str:
    try:
        if platform.system() == "Linux":
            for line in open("/proc/cpuinfo", encoding="utf-8"):
                if line.startswith("model name"):
                    return line.split(":", 1)[1].strip()
        if platform.system() == "Darwin":
            return subprocess.run(["sysctl", "-n", "machdep.cpu.brand_string"], capture_output=True,
                                  text=True, timeout=3).stdout.strip() or platform.processor()
    except Exception:  # noqa: BLE001
        pass
    return platform.processor() or platform.machine()


def disk_speed(path: str, mb: int = 64) -> tuple[float | None, float | None]:
    """Sequential write+read of a temp file (fsync'd). Small, so it takes ~a second."""
    try:
        buf = os.urandom(1 << 20)
        with tempfile.NamedTemporaryFile(dir=path, delete=False) as f:
            name = f.name
            t0 = time.perf_counter()
            for _ in range(mb):
                f.write(buf)
            f.flush()
            os.fsync(f.fileno())
            w = mb / (time.perf_counter() - t0)
        t0 = time.perf_counter()
        with open(name, "rb", buffering=0) as f:
            while f.read(1 << 20):
                pass
        r = mb / (time.perf_counter() - t0)  # likely page-cached: an upper bound
        os.unlink(name)
        return round(w, 1), round(r, 1)
    except Exception:  # noqa: BLE001
        return None, None


def temperatures() -> dict:
    out: dict = {}
    try:
        for name, entries in (psutil.sensors_temperatures() or {}).items():
            vals = [e.current for e in entries if e.current]
            if vals:
                out[name] = round(max(vals), 1)
    except Exception:  # noqa: BLE001 - not available on every OS
        pass
    if torch.cuda.is_available() and shutil.which("nvidia-smi"):
        try:
            r = subprocess.run(["nvidia-smi", "--query-gpu=temperature.gpu", "--format=csv,noheader,nounits"],
                               capture_output=True, text=True, timeout=3)
            out["gpu"] = float(r.stdout.strip().splitlines()[0])
        except Exception:  # noqa: BLE001
            pass
    return out


def probe(workdir: str = ".", measure_disk: bool = True) -> HardwareCard:
    vm = psutil.virtual_memory()
    notes = []
    acc, name, mem, bf16, fp16 = "cpu", _cpu_model(), None, False, False
    if torch.cuda.is_available():
        props = torch.cuda.get_device_properties(0)
        acc = "rocm" if getattr(torch.version, "hip", None) else "cuda"
        name, mem = props.name, round(props.total_memory / 2**30, 2)
        bf16 = torch.cuda.is_bf16_supported()
        fp16 = True
        if torch.cuda.device_count() > 1:
            notes.append(f"{torch.cuda.device_count()} GPUs found; jepa-studio uses the first one")
    elif getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        acc, name = "mps", "Apple GPU (Metal Performance Shaders)"
        fp16, bf16 = True, False
        mem = round(vm.total / 2**30, 2)
        notes.append("MPS shares system memory with the CPU")
    else:
        try:
            bf16 = torch.backends.mkldnn.is_available()  # oneDNN bf16 path on modern x86/ARM
        except Exception:  # noqa: BLE001
            bf16 = False
    free = shutil.disk_usage(workdir).free / 2**30
    w, r = disk_speed(workdir) if measure_disk else (None, None)
    compile_ok = hasattr(torch, "compile") and platform.system() != "Windows"
    return HardwareCard(
        os=f"{platform.system()} {platform.release()}", python=platform.python_version(),
        torch=torch.__version__, cpu_model=_cpu_model(),
        cpu_cores_physical=psutil.cpu_count(logical=False) or 1,
        cpu_cores_logical=psutil.cpu_count(logical=True) or 1,
        ram_total_gb=round(vm.total / 2**30, 2), ram_available_gb=round(vm.available / 2**30, 2),
        accelerator=acc, device_name=name, device_memory_gb=mem, bf16=bf16, fp16=fp16,
        compile_available=compile_ok, disk_free_gb=round(free, 1), disk_write_mb_s=w, disk_read_mb_s=r,
        temps_c=temperatures(), notes=notes)


def torch_device(card: HardwareCard, requested: str = "auto") -> torch.device:
    if requested not in ("auto", None):
        if requested in ("webgpu", "wasm"):
            requested = "cpu"
        return torch.device(requested)
    return torch.device({"cuda": "cuda", "rocm": "cuda", "mps": "mps"}.get(card.accelerator, "cpu"))


@dataclass
class Decision:
    setting: str
    value: object
    reason: str


def recommend(card: HardwareCard, cfg: dict) -> list[Decision]:
    """Turn a hardware card + config into concrete settings, each with its reason."""
    h = cfg["hardware"]
    mode = h["mode"]
    out: list[Decision] = []
    acc = card.accelerator
    # precision
    prec = h.get("precision", "auto")
    if prec == "auto":
        if acc in ("cuda", "rocm") and card.bf16:
            prec, why = "bf16", "GPU supports bfloat16: half the memory traffic, fp32 range, no loss scaling"
        elif acc in ("cuda", "rocm"):
            prec, why = "fp16", "GPU lacks bf16; fp16 with a gradient scaler"
        elif acc == "mps":
            prec, why = "fp16", "Apple GPU runs fp16 autocast"
        else:
            prec, why = "fp32", "CPU: bf16 autocast is often slower on small convnets, keep fp32 (toggle to test)"
        out.append(Decision("precision", prec, why))
    else:
        out.append(Decision("precision", prec, "set in config"))
    # channels-last
    cl = h.get("channels_last", "auto")
    if cl == "auto":
        cl = acc in ("cuda", "rocm") and cfg["model"]["arch"].startswith("convnet")
        out.append(Decision("channels_last", cl, "NHWC helps tensor-core convolutions on NVIDIA/AMD GPUs"
                            if cl else "no benefit expected on this device/architecture"))
    else:
        out.append(Decision("channels_last", cl, "set in config"))
    # compile
    comp = h.get("compile", False)
    if comp == "auto":
        comp = card.compile_available and acc in ("cuda", "rocm") and mode == "max"
        out.append(Decision("compile", comp, "torch.compile pays off on long GPU runs in Max mode"
                            if comp else "skipped: startup cost outweighs gains for short/CPU runs"))
    else:
        out.append(Decision("compile", comp, "set in config"))
    # workers
    wk = h.get("workers", "auto")
    if wk == "auto":
        cores = card.cpu_cores_physical
        wk = {"max": max(0, cores - 1), "balanced": max(0, cores // 2), "quiet": 0}[mode]
        wk = min(wk, 8)
        if acc == "cpu":
            wk = min(wk, 1)  # compute shares the same cores; extra workers just contend
        out.append(Decision("workers", wk, f"{cores} physical cores, mode={mode}"
                            + ("; CPU training shares cores with loading" if acc == "cpu" else "")))
    else:
        out.append(Decision("workers", wk, "set in config"))
    pm = h.get("pin_memory", "auto")
    if pm == "auto":
        pm = acc in ("cuda", "rocm")
        out.append(Decision("pin_memory", pm, "page-locked host memory speeds host->GPU copies"
                            if pm else "only useful with a discrete GPU"))
    else:
        out.append(Decision("pin_memory", pm, "set in config"))
    threads = {"max": card.cpu_cores_physical, "balanced": max(1, card.cpu_cores_physical // 2 + 1),
               "quiet": max(1, card.cpu_cores_physical // 4)}[mode]
    out.append(Decision("torch_threads", threads, f"intra-op threads for mode={mode}"))
    return out


def decisions_dict(ds: list[Decision]) -> dict:
    return {d.setting: d.value for d in ds}


def autocast_ctx(device: torch.device, precision: str):
    if precision == "fp32":
        return torch.autocast(device_type=device.type, enabled=False)
    dtype = torch.bfloat16 if precision == "bf16" else torch.float16
    return torch.autocast(device_type=device.type, dtype=dtype)


def memory_probe_batch(step_fn, start: int, limit: int, device: torch.device, max_frac: float) -> tuple[int, list]:
    """Largest power-of-two batch whose training step fits in memory (doubling search).

    step_fn(batch_size) runs one forward+backward. On CPU/MPS we use the RAM headroom as the
    budget; on CUDA the allocator's peak. Returns (batch, trace) where trace lists each try.
    """
    trace = []
    best = None
    b = start
    while b <= limit:
        try:
            if device.type == "cuda":
                torch.cuda.empty_cache()
                torch.cuda.reset_peak_memory_stats()
            rss0 = psutil.Process().memory_info().rss
            t0 = time.perf_counter()
            step_fn(b)
            dt = time.perf_counter() - t0
            if device.type == "cuda":
                used = torch.cuda.max_memory_allocated() / torch.cuda.get_device_properties(0).total_memory
            else:
                grow = psutil.Process().memory_info().rss - rss0
                used = (psutil.virtual_memory().total - psutil.virtual_memory().available + max(grow, 0)) \
                    / psutil.virtual_memory().total
            trace.append({"batch": b, "ok": True, "mem_frac": round(used, 3), "step_s": round(dt, 4)})
            if used > max_frac:
                break
            best = b
            b *= 2
        except RuntimeError as e:  # out of memory
            trace.append({"batch": b, "ok": False, "error": str(e).splitlines()[0][:120]})
            if device.type == "cuda":
                torch.cuda.empty_cache()
            break
    return (best or start), trace


def memory_usage(device: torch.device) -> list[dict]:
    """What the live memory guard watches: this process's resident memory (RSS) against total
    RAM, plus PyTorch's allocated CUDA memory against the GPU's total when training on CUDA.
    Returns [{"what", "used", "total"}] in bytes. Replaceable in tests (Trainer(memory_reader=...))."""
    out = [{"what": "process RAM (RSS)", "used": psutil.Process().memory_info().rss,
            "total": psutil.virtual_memory().total}]
    if device.type == "cuda":
        idx = device.index if device.index is not None else torch.cuda.current_device()
        out.append({"what": "CUDA allocated", "used": torch.cuda.memory_allocated(idx),
                    "total": torch.cuda.get_device_properties(idx).total_memory})
    return out


def memory_breach(readings: list[dict], max_frac: float) -> str | None:
    """First reading above max_frac of its total, as a human-readable reason; None if all fit."""
    for r in readings:
        if r["total"] and r["used"] > max_frac * r["total"]:
            return (f"memory guard: {r['what']} {r['used'] / 2**30:.2f} GB is above "
                    f"hardware.max_memory_frac={max_frac:g} of {r['total'] / 2**30:.2f} GB")
    return None


class Throttle:
    """Keeps training polite. Quiet mode adds a duty-cycle sleep; every mode pauses when a
    temperature sensor goes above max_temp_c, and resumes 5 C below it."""

    def __init__(self, mode: str, max_temp_c: float = 85.0, check_every: int = 20):
        self.mode, self.max_temp, self.every = mode, max_temp_c, check_every
        self.i = 0
        self.cooling = False
        self.events: list[str] = []

    def hottest(self) -> float | None:
        t = temperatures()
        return max(t.values()) if t else None

    def after_step(self, step_seconds: float) -> float:
        slept = 0.0
        if self.mode == "quiet":
            time.sleep(step_seconds * 0.5)  # ~66% duty cycle
            slept += step_seconds * 0.5
        self.i += 1
        if self.i % self.every == 0:
            t = self.hottest()
            if t is not None and t > self.max_temp:
                self.events.append(f"paused at {t:.0f} C (limit {self.max_temp:.0f} C)")
                self.cooling = True
                waited = 0
                while t is not None and t > self.max_temp - 5 and waited < 600:
                    time.sleep(5)
                    slept += 5
                    waited += 5
                    t = self.hottest()
                self.events.append("resumed after cooling")
                self.cooling = False
        return slept
