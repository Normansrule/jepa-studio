# Changelog

## 0.3.1 (2026-10-05)

**Fixed**
- Desktop installers failed to build on every platform: the workflow passed `--ci` to
  tauri-action, which forwards extra arguments to cargo (`unexpected argument '--ci'`).
- `tests/test_setup_script.py` failed for non-root users (CI runners, your terminal): the fake
  `sudo` did not understand `sudo -n`.
- The ViT saliency test required bit-identical results; batched CPU attention kernels on some
  machines round differently. It now uses a tolerance.

**Changed**
- Dependabot groups minor and patch updates into one weekly PR per ecosystem.

## 0.3.0 (2026-10-04)

Input and design refinements across the web app (same front end in the desktop app); see
`docs/web-demo.md`, "Entering settings" and "The run config editor".

**New**
- **Precise, validated input.** Every slider has a synced number box (same min/max/step). Every
  numeric control is checked inline with the shared `validate()` + `config.schema.json` and the
  tier's own range; bad values get an accessible message, are never applied or clamped, and
  disable Start with the reason shown. "Reset to defaults" per section.
- **Presets** on the Train tab: Quick look, Balanced, Thorough (larger in the desktop app);
  any later change reads "Custom".
- **Help** buttons on every control label: a short explanation and a link to the matching
  docs section (Esc closes and returns focus).
- **Guided flow.** A progress stepper (Data → Train → Inspect → Evaluate → Export) from real
  app state, a "Run finished — inspect the embeddings" prompt, notices for completions and
  errors (`site/js/ui/toast.js`, role=status / role=alert), empty states, and disabled buttons
  that say why.
- **Drag and drop and paste** of images (files or whole folders) and CSV files onto the Data
  tab's drop zone, through the same loader and limits as the file buttons.
- **Config editor** on the Export tab: edit the run config as JSON, Validate (errors with their
  paths, click to select), Apply, Revert, Download, Open, with a confirmation before an opened
  file replaces unapplied edits.
- **Remembered preferences** (theme, last tab, last preset) in `localStorage`, wrapped so the app
  works when storage throws.
- Landing page: tighter hero copy and a three-step "How it works" row linking into the app.

**Changed**
- Spacing scale and 44 px touch targets on phones; the hardware card shows a loading state.
- The Train tab's tier note describes the desktop backend in the desktop app (it said "Demo
  tier" in both).
- Time-series window length is now saved in the config (`data.series_window`).
- A loaded config whose Shapes count is outside the slider's range is shown and flagged instead
  of being silently clamped.
- **Setup script rewritten for fresh machines** (`scripts/setup_ubuntu.sh`): `--doctor`
  preflight, finds the zip in any Windows/Ubuntu Downloads folder (including "(1)" copies),
  installs missing packages and the GitHub CLI in one step, works with or without conda,
  signs in to GitHub with the `workflow` scope and pushes over HTTPS when no SSH alias is set
  up, asks before publishing when tests fail, logs every run to `~/jepa-studio-setup-*.log`
  and says exactly which step failed.
- Phones: the header stays on one row (the compute backend is shown on the Train tab instead),
  and the World environment names fit their menu.

## 0.2.0 (2026-10-01)

**New**
- **Desktop saliency and attention maps.** `GET /api/runs/<id>/saliency` returns the gradient
  saliency map (same definition as the web tier) and, for `vit-tiny`, the last block's [CLS]
  attention over the patch grid. The Inspect tab shows both in the desktop app (`jepa_studio/maps.py`).
- **ONNX export.** `jepa-studio export --run-dir <run> --onnx` writes `encoder.onnx` (opset 18,
  dynamic batch, outputs `embedding` and `projection`) for the image encoders, checked against
  PyTorch with onnxruntime in the tests. Optional extra: `pip install "jepa-studio[onnx]"`.
- **Verified weights.** Runs record `encoder.sha256`; `eval`, `inspect`, `export`, ONNX export,
  the desktop server and the world-model loader check it and refuse a mismatched file.
  Checkpoints record and check the hashes of their model and optimizer files on resume.
- **Live memory guard.** Training stops cleanly with a checkpoint and a `reason` when the
  process passes `hardware.max_memory_frac` of RAM (or of GPU memory on CUDA); the Train tab
  shows the reason.
- **Video caps before decoding.** Clips over 60 s or with frames over 16.8 megapixels are refused
  from the container header, before any frame is decoded; unreadable clips are skipped with a reason.
- **Desktop Data tab** picks folders through the native dialog (images, video, CSV time series)
  and previews backend-made views.

**Fixed**
- Training never finished a step when a dataset had fewer batches per epoch than the
  gradient-accumulation count (accumulation is now capped at the batches available).
- Training started from the desktop backend crashed (`signal only works in main thread`).
- A config `name` such as `../../x` could place a run folder outside `runs/` (schema pattern +
  server and trainer checks); a negative `Content-Length` bypassed the 2 MB body cap.
- The LeJEPA loss ran in bf16/fp16 under autocast; it is now always at least float32.
- World-model GIFs played with 0 ms frames.
- The first hardware card could finish after the batch-size probe started by Start and
  overwrite it ("Picked batch" vanished; seen on a 6-core machine). Only the newest hardware
  result is shown now.
- DataLoader workers were forked from multi-threaded processes (desktop backend, test runner),
  which can deadlock on Linux; they now start from a preloaded forkserver there.

## 0.1.0 (2026-09-30)

First release: LeJEPA pretraining with SIGReg, hardware auto-tuning, evaluation (linear probe,
k-NN, isotropy, collapse check), world model + CEM planning bot, web demo (GitHub Pages),
Tauri desktop shell, docs and CI.
