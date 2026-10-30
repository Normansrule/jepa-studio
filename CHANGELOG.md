# Changelog

## 0.4.0 (2026-10-30): JEV Studio

The project is now **JEV Studio** (repository `jev-jepa-studio`): JEPA Studio plus **JEV**, a
local AI agent. Python package and command names are unchanged (`jepa-studio`), plus a new `jev`
package and command.

**New: JEV, the local agent** (`jev/`, docs in `docs/jev/`)
- Chat and tool use with your own models through a local Ollama (loopback only unless you
  allow a remote URL). Download models, import GGUF files you already have (SHA-256 recorded),
  delete them; weights stay in folders you control.
- **Sandbox auto-run with an approval safety net.** Code and renders run freely inside each
  conversation's workspace with CPU/memory/time/file limits; on Linux/WSL with bubblewrap the
  sandbox also has no network and cannot see your home folder or Windows drives. Network, files
  outside the workspace, deleting, installing packages or models, shell commands, long GPU jobs
  and anything leaving the machine always ask first. The rules are one editable table
  (`jev/policy.py`); every decision is written to an audit log.
- Local SQLite store (conversations, tool calls, approvals, artifacts) with export and delete.
- Tools: files, calculator, plots, Python/Node in the sandbox, shell (asks), packages (asks),
  JEPA embeddings and training, science renders and simulations, media.
- `jev doctor`, `jev chat`, `jev models list|pull|import|rm`.
- Ubuntu 24.04+: JEV detects the AppArmor user-namespace restriction (which would leave the
  sandbox weak) and `jev doctor` prints the one-step fix, an AppArmor profile for bubblewrap
  (`scripts/apparmor/jev-bwrap`); the setup script offers to install it. On Linux with a user
  systemd, each run also gets its own scope with memory and task limits.

**New: science**
- Interstellar-style black-hole renderer (`jev/science/`): null geodesics in Kerr spacetime with
  conserved-quantity projection, thin Page–Thorne disk, blackbody colours, gravitational and
  Doppler shifts with beaming, an "Interstellar mode" as in James et al. (2015), lensed stars,
  stills and orbit videos. Validated against the analytic shadow, photon sphere, ISCO, Kerr
  critical curve and weak-field deflection (`docs/jev/science.md`).
- Physics simulations: N-body (figure-eight), double pendulum, 2-D wave, Kepler/relativistic
  orbit, Lorenz, with conservation checks.
- Real-time WebGL2 black hole and canvas simulations in the browser (Science tab, both tiers).

**New: media**
- Narrated explainer videos (local voice via Piper/espeak-ng/say/SAPI, slides, burned-in
  captions), footage editing (shot detection, JEPA-embedding scene search, highlight reels,
  cuts), images and charts, optional local diffusion text-to-image/video behind approval.

**New: UI**
- JEV tab (desktop): streaming chat, tool cards with risk badges, approval dialog and audit
  view, artifact gallery, model manager, settings, status and Doctor. Science tab (web +
  desktop). Product renamed in the site and the desktop app (new app id: existing users start
  with a fresh data folder; see `docs/desktop.md`).

**Changed**
- The docs assistant indexes `docs/jev/` too. Setup script: new repo name, keeps an existing
  `~/jepa-studio` clone, and offers to install Ollama (asks in a terminal even with `--yes`) and a
  model sized for your GPU.

## 0.3.1 (2026-10-05)

**Fixed**
- Desktop installers failed to build on every platform: the workflow passed `--ci` to
  tauri-action, which forwards extra arguments to cargo (`unexpected argument '--ci'`).
- `tests/test_setup_script.py` failed for non-root users (CI runners, your terminal): the fake
  `sudo` did not understand `sudo -n`.
- ViT saliency maps changed slightly (up to ~0.5%) depending on which other images were in the
  same batch, because batched CPU attention kernels round differently on some machines (seen
  on CI and a 6-core desktop). Saliency now runs one backward pass per image.
- CI annotates failed tests from a JUnit report (`scripts/annotate_junit.py`), so failures are
  readable on the run page without downloading logs.

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
