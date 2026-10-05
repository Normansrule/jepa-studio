# Hardware awareness

jepa-studio looks at your machine before it trains, picks settings for it, and writes down
every choice with the reason, so you can see *why* it runs the way it does and override
anything. Nothing here changes system settings. Code: `jepa_studio/hardware.py` and
`jepa_studio/train.py`.

## 1. The hardware card

`hardware.probe()` builds a `HardwareCard`; `jepa-studio hardware` prints it (`--json` for
machine-readable output, `--no-disk` to skip the disk test) together with the settings it would
pick for the default config in balanced mode. The desktop app shows the same card
(`assets/screens/hardware-card.png`).

| Field | How it is measured |
|---|---|
| OS, Python, torch | `platform`, `torch.__version__` |
| CPU model, physical / logical cores | `/proc/cpuinfo` on Linux, `sysctl` on macOS; `psutil.cpu_count` |
| RAM total / available | `psutil.virtual_memory()` |
| accelerator | `cuda`, `rocm` (a HIP build of PyTorch), `mps` (Apple GPU) or `cpu`; with several GPUs the first one is used and a note says so |
| device memory | CUDA/ROCm: the device's total memory; MPS: system RAM (shared, noted) |
| bf16 / fp16 | CUDA: `torch.cuda.is_bf16_supported()`, fp16 always; MPS: fp16 yes, bf16 no; CPU: bf16 = oneDNN available |
| torch.compile | available unless on Windows |
| disk free, write / read MB/s | a 64 MB temp file written with `fsync`, then read back; the read is probably page-cached, so treat it as an upper bound |
| temperatures | `psutil.sensors_temperatures()` (Linux/FreeBSD) and `nvidia-smi` for NVIDIA GPUs; empty when nothing is exposed |

## 2. Automatic decisions

`hardware.recommend(card, cfg)` turns the card and the config into settings. Every setting in
the config's `hardware` block that says `"auto"` is decided here; anything else is kept and
recorded as "set in config".

| Setting | Rule when `"auto"` | Reason recorded |
|---|---|---|
| `precision` | CUDA/ROCm with bf16 → `bf16`; CUDA/ROCm without → `fp16` (with a gradient scaler); MPS → `fp16`; CPU → `fp32` | e.g. "CPU: bf16 autocast is often slower on small convnets, keep fp32 (toggle to test)" |
| `channels_last` | on only for CUDA/ROCm with a `convnet-*` architecture | "NHWC helps tensor-core convolutions on NVIDIA/AMD GPUs" |
| `compile` | only if available, on CUDA/ROCm, in **max** mode (the default config sets `false`, so it is "set in config" unless you choose `"auto"`) | "torch.compile pays off on long GPU runs in Max mode" |
| `workers` | max: cores − 1; balanced: cores ÷ 2; quiet: 0; at most 8; **at most 1 on CPU** (loading competes with compute for the same cores) | "2 physical cores, mode=balanced; CPU training shares cores with loading" |
| `pin_memory` | on for CUDA/ROCm | "page-locked host memory speeds host->GPU copies" |
| `torch_threads` | max: all physical cores; balanced: cores ÷ 2 + 1; quiet: cores ÷ 4 (at least 1). Always decided by the mode. | "intra-op threads for mode=balanced" |
| `batch_size` | memory probe (below) | "largest power of two that fit under 85% memory (probe: ...)" |
| `grad_accumulation` | `round(effective_batch / batch_size)`, at least 1, and at most the batches available per epoch (an optimizer step never spans two epochs) | "effective batch 128x1=128 (target 128)"; small datasets: "... target 256 capped because the dataset has only 1 batch(es) of 32 per epoch" |

**Batch-size probe** (`Trainer.pick_batch_size`, `hardware.memory_probe_batch`). Starting at 16,
run one real forward + backward step on a batch of that size and double, up to a cap: 1024 in
max mode, 512 in balanced, 128 in quiet, never more than the dataset, and at most 256 on CPU
(bigger CPU batches rarely raise samples/s). The memory used is measured as the peak allocated
fraction of GPU memory on CUDA, or (system memory in use + this process's growth) / total on
CPU and MPS. The largest size whose step stayed at or under `hardware.max_memory_frac` (default
0.85) and did not run out of memory wins; if even 16 does not fit, 16 is used. The full trace
(size, memory fraction, step time) goes into the reason.
The same `max_memory_frac` is then watched during training by the memory guard (section 6).

Everything lands in the run folder's `hardware.json` as `{"card": ..., "decisions": [...]}`.
The real example in this repo, `runs/shapes-desktop/hardware.json`, was a 2-core Intel Xeon at
2.80 GHz with 7.84 GB RAM and no GPU: precision `fp32`, `channels_last` off, `workers` 1,
`pin_memory` off, `torch_threads` 2, `batch_size` 128 (set in config), gradient accumulation 1.

## 3. Modes and throttling

| | max | balanced (default) | quiet |
|---|---|---|---|
| data workers (auto) | cores − 1 (≤ 8) | cores ÷ 2 (≤ 8) | 0 |
| torch threads | all physical cores | cores ÷ 2 + 1 | cores ÷ 4 |
| auto batch cap | 1024 | 512 | 128 |
| `torch.compile` when `"auto"` | on CUDA/ROCm | off | off |
| duty cycle | 100% | 100% | sleeps half a step after each step (about 66%) |

On CPU, workers are capped at 1 in every mode. When the process already runs other threads (the desktop backend's
HTTP server and training thread), workers start from a `forkserver` that has preloaded torch
instead of being forked from the threaded process, which can deadlock on Linux (`train._worker_context`).

**Thermal pause** (`hardware.Throttle`, every mode). Every 20 optimizer steps the hottest sensor
is read. Above `hardware.max_temp_c` (default 85 °C) training sleeps in 5 s chunks until the
sensor reads 5 °C below the limit, for at most 10 minutes per pause; "paused at X C" and
"resumed after cooling" are recorded and returned as `throttle_events`. Without sensors (macOS,
Windows and many VMs), this never triggers; see
[SECURITY_MODEL.md](SECURITY_MODEL.md#6-resource-safety).

## 4. Measuring what each optimization buys

`jepa-studio bench` (`train.benchmark_optimizations`, also the Train tab's before/after chart)
starts from a plain baseline (fp32, batch 32, 0 workers, no pinned memory, no channels-last)
and switches on, cumulatively and in this order, each automatic setting that differs from the
baseline: auto batch size, precision, channels-last, workers, pinned memory. Each stage runs 2
warm-up steps and then `--steps` (default 8) timed steps, and reports samples/s. `--out f.json`
saves the table. It works in a temporary `runs/_bench` folder that it deletes afterwards.

![Throughput as each automatic setting is switched on](../assets/throughput.svg)

Numbers depend entirely on the machine. The figure's data is in `assets/throughput-benchmark.json`
(produced by `scripts/make_visuals.py --measure`: this benchmark on `runs/shapes-desktop/config.json`,
12 timed steps per stage, repeated 3 times) on a 2-core Intel Xeon at 2.10 GHz with 7.84 GB RAM and
no GPU. Medians: baseline 202.5 samples/s, + batch size 128: 199.7, + 1 loader worker: 195.3. The
three runs overlap, so on this machine the automatic choices are speed-neutral; that is the
expected outcome on a small CPU, where the rules keep fp32 and at most one worker. No GPU
measurement is in the repo yet.

## 5. Learning-rate schedule

`train.lr_at`: linear warm-up over the first `warmup_frac` of the steps, then cosine decay to
`final_lr_ratio` × the base rate.

$$
\text{lr}(s) = \begin{cases}
\text{lr}_0 \dfrac{s+1}{W} & s < W = \max(1, \lfloor S f_w \rfloor) \\
\text{lr}_0 \Bigl(r + (1-r)\,\tfrac12\bigl(1 + \cos(\pi p)\bigr)\Bigr), & p = \min\Bigl(1, \dfrac{s - W}{\max(1, S - W)}\Bigr)
\end{cases}
$$

with $s$ the optimizer step (from 0), $S$ total optimizer steps, $f_w$ = `warmup_frac` (default
0.1) and $r$ = `final_lr_ratio` (default $10^{-3}$). Optimizer: AdamW,
`train.lr` (default $5\times10^{-4}$), `train.weight_decay` (default 0.05). Total steps are
`train.max_steps` if set, else `epochs × (batches per epoch ÷ grad_accumulation)`.

## 6. Checkpoints, pause, stop and resume

Run folder (`train.py` header):

```
runs/<name>-<timestamp>/
  config.json          exact config used
  hardware.json        card + every decision and its reason
  events.jsonl         one JSON record per event (see docs/api.md)
  control.json         {"command": "pause" | "resume" | "stop"} steers a live run
  ckpt/step-XXXXXXX/   model.safetensors, optim.safetensors, state.json
  ckpt/LATEST          name of the newest checkpoint
  encoder.safetensors  final weights (metadata: arch, step)
```

* **When:** every `train.checkpoint_every` optimizer steps (default 200), on pause, on stop, and
  at the end. Only the newest 2 checkpoints are kept. After each periodic checkpoint the disk
  guard runs (stop if the run folder exceeds `hardware.max_disk_gb` or under 1 GB is free).
* **What:** model and optimizer tensors as safetensors (no pickle), plus `state.json` with the
  step, epoch, optimizer scalars, gradient-scaler state and the torch RNG state. `LATEST` and
  `state.json` are written atomically.
* **Control:** `control.json` is read before every micro-batch. `pause` writes a checkpoint and
  waits (polling every 0.5 s) until the command changes; `stop` writes a checkpoint and ends the
  run with status `stopped`. SIGTERM also stops cleanly. The desktop app's buttons write this
  file through `POST /api/runs/<id>/control`.
* **Resume:** `jepa-studio train --run-dir runs/<name>` resumes from `ckpt/LATEST` by default
  (`--fresh` ignores checkpoints). The step counter continues where it stopped
  (`test_training.py::test_train_checkpoint_resume_and_control`).
* **Not bit-exact:** the Python `random` and NumPy RNG states, the augmentation RNG and the data
  loader's position inside the epoch are not saved, so a resumed run sees a different
  augmentation/data order from an uninterrupted one. Loss curves continue smoothly but will not
  match an uninterrupted run digit for digit.
* A non-finite loss stops the run with an error rather than writing garbage checkpoints.
* **Memory guard:** at every log step (every `train.log_every` steps, and step 1) the run reads
  this process's resident memory (RSS) and, on CUDA, PyTorch's allocated GPU memory
  (`hardware.memory_usage`). If either is above `hardware.max_memory_frac` (default 0.85) of the
  machine's RAM or the GPU's memory, the run writes a checkpoint, emits
  `{"event": "stopped", "step": ..., "reason": "memory guard: process RAM (RSS) 6.81 GB is above
  hardware.max_memory_frac=0.85 of 7.84 GB"}`, saves `encoder.safetensors` and ends with status
  `stopped` (the `end` event and `fit()`'s result carry the same `reason`). Resume with a smaller
  `batch_size` or `max_items`. It counts only this process and runs only at log steps, so it
  cannot catch a spike between checks or memory used by other programs
  ([SECURITY_MODEL.md §6](SECURITY_MODEL.md#6-resource-safety)). Tests inject a fake reader:
  `Trainer(cfg, memory_reader=fn)`, where `fn(device)` returns
  `[{"what": ..., "used": bytes, "total": bytes}]`.

The real run in this repo, `runs/shapes-desktop`, ran 768 steps (12 epochs of 8192 Shapes images
at batch 128) and kept `step-0000600` and `step-0000768`.
