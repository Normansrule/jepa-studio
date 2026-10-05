# Security model

jepa-studio reads files you point it at, loads model weights, runs a local HTTP server for the
desktop app and (optionally) a local language model. This page lists what can go wrong, what
the code does about it, and **which file and which test enforce each claim**. Anything we would
like to be true but is not implemented yet is listed under "Not yet enforced" in each section,
so you can judge the gap yourself. To report a vulnerability, see [SECURITY.md](../SECURITY.md).

Tests named below live in `tests/`; run them with `python -m pytest tests/test_security.py -q`
(plus the other files named). "Code only" means the behaviour is in the code but no automated
test checks it yet.

## Assets and attackers

* **What we protect:** your files (only the folders you chose should be readable), your
  machine's resources (memory, disk, temperature), and the integrity of weights and results you
  share or receive.
* **Who we assume might attack:** someone who hands you a weights file, a dataset (images,
  clips, CSV/NPY), a run folder or a reproducibility bundle; a web page open in your browser
  while the desktop backend runs; text inside files and logs that tries to steer the assistant;
  a compromised dependency or CI action.
* **Out of scope:** an attacker who already runs code as your user, and the security of
  PyTorch, Pillow, imageio/FFmpeg, safetensors and the OS themselves (we reduce what reaches
  them, we don't audit them).

## 1. Malicious weights

Pickle-based checkpoints (`.pt`, `.pth`, `.ckpt`, `.bin`) can run arbitrary code when loaded.
jepa-studio never loads them by default.

| Claim | Where | Test |
|---|---|---|
| Only `.safetensors` files are loaded; `.pt .pth .ckpt .bin .pkl .pickle .joblib` are refused with an explanation, and any other suffix is refused too | `weights.load_tensors`, `REFUSED_SUFFIXES` | `test_security.py::test_pickle_formats_refused` (the `.pt` case) |
| An expected SHA-256 can be required when loading; a mismatch raises | `weights.load_tensors(expected_sha256=...)` | `test_security.py::test_hash_mismatch_detected` |
| A weights folder can carry a `manifest.json` with every file's SHA-256 and size, signed with Ed25519; verification fails on an untrusted key, an edited manifest, a swapped file, or a path that escapes the folder | `weights.build_manifest / sign_manifest / verify_files`; CLI `keygen`, `sign-weights`, `verify-weights` | `test_security.py::test_signed_manifest_roundtrip_and_tamper` |
| Every run records the SHA-256 of its final weights (`encoder.sha256`, `sha256sum` format; runs from before v0.2 fall back to the hash in their `end` event). `eval`, `inspect`, `export`, ONNX export and the desktop server check it on load and refuse a mismatch (server: 409); `export` refuses to bundle mismatched weights. World models are checked against `weights_sha256` in their `metrics.json`. Checkpoints store the SHA-256 of `model.safetensors` and `optim.safetensors` in `state.json`, checked on resume, and `ckpt/LATEST` must name a `step-NNNNNNN` folder. This is integrity, not authenticity | `weights.load_run_weights / recorded_run_hash`, `Trainer.load_checkpoint`, `world.load_world_run` | `test_security.py::test_run_weights_verified_on_every_load`, `::test_checkpoint_hashes_checked_on_resume`, `::test_shipped_runs_match_their_recorded_hashes` |
| Optimizer state in checkpoints is stored as safetensors + JSON, not pickle | `weights.optimizer_to_tensors / optimizer_from_tensors` | `test_security.py::test_optimizer_state_roundtrip_without_pickle` |
| Converting a legacy checkpoint is explicit and uses `torch.load(weights_only=True)`; the result must be a flat tensor dict | `weights.import_legacy`, CLI `import-legacy` (prints a warning) | code only |
| Weight and checkpoint writes are atomic (temp file + `os.replace`), so a crash never leaves a half-written file | `weights.save_tensors`, `atomic_write_bytes` | code only (resume path exercised by `test_training.py::test_train_checkpoint_resume_and_control`) |
| Reproducibility bundles carry a SHA-256 manifest; `verify-bundle` detects accidental corruption (a file whose bytes no longer match the manifest) and manifest paths that escape the bundle. The manifest is **not signed**, so it does not stop someone who edits both a file and the manifest, and files not listed in it are ignored; sign weights with `sign-weights` when that matters | `export.export_bundle / verify_bundle` | `test_training.py::test_export_bundle_and_verify` |
| Weights written by the browser (safetensors, `.npy`, zip) load in Python | `site/js/io/*.js` | `test_web_engine.py::test_writers_load_in_python` |
| World models ship to the browser as plain JSON numbers (no code, no ONNX) | `world/export.py` | `test_world_parity.py` |
| ONNX is **export only**: `jepa-studio export --run-dir <run> --onnx` writes `encoder.onnx` (encoder + projector of an image run: `convnet-tiny`, `convnet-small`, `vit-tiny`, `mlp-tiny`; opset 18, dynamic batch axis, outputs `embedding` and `projection`) and records its SHA-256 in the bundle's `manifest.json` and in `report.json` (`onnx_sha256`). Other archs are refused with a message. The file is a protobuf graph of standard ONNX operators: parsing it and opening an onnxruntime session runs no Python code; outputs match PyTorch to 1e-4 | `export.export_onnx`, `export.export_bundle(onnx=True)` | `test_security.py::test_onnx_export_is_plain_protobuf_and_loads_without_python`, `test_training.py::test_onnx_export_matches_pytorch`, `test_onnx_refuses_non_image_archs`, `test_export_bundle_records_onnx_hash` |

**Not yet enforced**

* **ONNX import is not supported.** jepa-studio can export `encoder.onnx` but never loads an
  `.onnx` file: no command, server endpoint or tab reads one, so weights still come only from
  safetensors. `sign-weights` includes `.onnx` files when it builds a manifest, and
  `verify-weights` checks their hashes, but nothing in jepa-studio consumes them afterwards. The
  exported file is checked by the ONNX tests only when `onnx`, `onnxscript` and `onnxruntime`
  are installed (`pip install "jepa-studio[onnx]"`); otherwise those tests are skipped.
* **Signed manifests are opt-in and manual.** Nothing in the load path requires a *signed*
  manifest. Run weights are hash-checked against the run's own record (see the table above),
  which catches corruption and swapped files but not someone who edits both the weights and
  the record. A run with no record at all still loads, with a note that it is unverified. There
  is no download feature, so the `weights.py` docstring's "every downloaded weight file must
  match..." describes the intended policy, not a code path.
* **No trusted keys ship with the repo.** `verify-weights` defaults to `weights/trusted_keys.json`,
  which does not exist; pass `--keys your_keys.json` (format `{"ed25519": ["<base64 key>"]}`).
* The safetensors header itself is parsed by the `safetensors` library; we do not add our own
  size limits on it.

## 2. Malicious datasets

A dataset folder is untrusted input: decompression bombs, symlinks pointing elsewhere, pickled
`.npy` files, enormous CSVs.

| Claim | Where | Test |
|---|---|---|
| Extension allowlist per data kind; files are resolved and anything (e.g. a symlink) resolving outside the chosen folder is skipped; `fake.png.exe` and `script.py` are ignored | `data/loaders.safe_listing`, `IMAGE_EXT`, `VIDEO_EXT`, `SERIES_EXT` | `test_security.py::test_symlink_escape_and_extensions` |
| Images: 64 MB file cap, 40 megapixel cap (Pillow's `MAX_IMAGE_PIXELS` is set to the same value, so a bomb raises before decoding), JPEG decoded at reduced size | `loaders.load_image`, `MAX_FILE_BYTES`, `MAX_PIXELS` | `test_security.py::test_decompression_bomb_rejected` (a 100 Mpx PNG under 200 kB) |
| `.npy` loaded with `allow_pickle=False`; CSV/TSV parsed as numbers only (no `eval`), capped at 50 million values and 64 MB | `loaders.load_series`, `MAX_SERIES_VALUES` | `test_security.py::test_series_loader_no_pickle` |
| Video: 512 MB file cap; then, from the container header and **before any frame is decoded**, a duration cap (60 s) and a frame-size cap (4096 × 4096 pixels); a header that cannot be read (or has no size or duration) is refused. While decoding: at most 512 frames, and every decoded frame is re-checked against the size cap (a header can lie). Malformed clips are skipped with a reason | `loaders.clip_metadata / check_clip_metadata / load_clip`, `MAX_VIDEO_BYTES`, `MAX_CLIP_SECONDS`, `MAX_CLIP_PIXELS`, `MAX_FRAMES` | `test_security.py::test_video_duration_cap_before_decoding`, `test_video_frame_size_cap_before_decoding`, `test_malformed_clips_skipped_with_reason` (mp4 cases need `imageio-ffmpeg`), `test_timeseries_and_video_datasets` |
| At most `data.max_items` items (default 4096) are listed; unreadable files are skipped and reported (first 50) instead of crashing the run | `loaders.build_dataset` | code only |
| Config files over 1 MB are refused; unknown or misspelled keys and out-of-range values are rejected by the schema | `config.load_config`, `schema/config.schema.json` | `test_security.py::test_default_config_valid_and_typos_rejected`, `test_example_configs_valid` |
| Browser tier: files are decoded in the page and never uploaded (Content Security Policy `connect-src 'self'`); limits 2000 images, 20 MB and 40 megapixels per image, CSV 20 MB / 200 000 rows / 64 columns | `site/js/tabs/data.js` `IMAGE_LIMITS`, `site/js/io/csv.js` `CSV_LIMITS`, CSP in `site/*.html` | CSP violations fail `test_site_playwright.py` (runs only when Playwright's Chromium is installed); limits: code only |

**Not yet enforced**

* **Video metadata comes from the parsers we are protecting against.** Size and duration are
  read by FFmpeg (through imageio-ffmpeg) or, for `.gif`, by Pillow; a crafted header can lie,
  which is why decoded frames are re-checked and decoding stops at 512 frames. GIF stores one
  delay per frame and reading them all means seeking through the frames, so a GIF's duration is
  **estimated** as frame count × the first frame's delay; a GIF with a short first delay and long
  later ones passes the duration cap (the frame and size caps still apply). Non-GIF clips are
  always opened with the FFmpeg plugin (`pip install "jepa-studio[video]"`); PyAV is not used.
* **No total-memory cap for a dataset.** All items are decoded into RAM up front
  (`max_items` × downsampled size); a large `max_items` on a small machine can exhaust memory.
* A time-series `data.path` that is a single file (not a folder) skips `safe_listing`; in the
  desktop server it still has to lie inside a granted folder (section 4).

## 3. Supply chain

| Claim | Where | Check |
|---|---|---|
| Every GitHub Action is pinned to a full 40-character commit SHA with the version in a comment | `.github/workflows/*.yml` | review; OpenSSF Scorecard "Pinned-Dependencies" |
| Workflows declare least-privilege `permissions:` (read-only by default, write only where a job needs it) | `.github/workflows/*.yml` | Scorecard "Token-Permissions" |
| Dependabot watches pip, npm (`desktop/`), cargo (`desktop/src-tauri/`) and GitHub Actions | `.github/dependabot.yml` | GitHub |
| Lockfiles for the desktop shell | `desktop/package-lock.json`, `desktop/src-tauri/Cargo.lock` | committed |
| The desktop app installs exact versions (`==`) of the **top-level** Python packages into a private environment. Their own dependencies, and the `pip` upgrade, are not pinned or hash-checked (see "Not yet enforced") | `requirements-desktop.txt`, `desktop/src-tauri/src/setup.rs` (see [desktop.md](desktop.md)) | code only |
| The gitleaks binary used in CI is verified against its published SHA-256 before it runs | `.github/workflows/gitleaks.yml` | CI |
| CodeQL (Python, JavaScript/TypeScript, Actions) and Scorecard run on pushes, pull requests and a schedule | `.github/workflows/codeql.yml`, `scorecard.yml` | CI |
| The web app loads no third-party scripts, styles or fonts (fonts are self-hosted) | `site/*.html` CSP `default-src 'self'` | `test_site_playwright.py` (when browsers are present) |

**Not yet enforced**

* **No hash-locked Python installs.** `requirements-desktop.txt` pins versions but not hashes, and
  pip is not run with `--require-hashes`; `pyproject.toml` has lower bounds only. CI installs
  CPU PyTorch and the test tools by version, not by hash.
* No signed releases, build provenance (SLSA) or Software Bill of Materials (SBOM) yet.
* Dependabot and Scorecard only report; merging updates is a human decision.

## 4. Local privilege boundaries (desktop backend)

`jepa-studio serve` is the desktop app's backend. Any web page in any browser on your machine
can try to send requests to `127.0.0.1`, so the server must refuse everything that is not the
desktop shell.

| Claim | Where | Test |
|---|---|---|
| Binds `127.0.0.1` only (never `0.0.0.0`), port chosen by the OS | `server.serve` | code only |
| Every request needs the per-session token in `X-JEPA-Token` (32 random bytes, constant-time compare) or gets 401 | `server.make_handler._guard` | `test_security.py::test_server_requires_token_and_host` |
| `Host` must be `127.0.0.1:<port>` or `localhost:<port>`, else 421 (DNS-rebinding defense) | `_host_ok` | same test |
| `Origin`, when sent, must be the Tauri shell (`tauri://localhost`, `http(s)://tauri.localhost`) or the server's own origin, else 403; no wildcard CORS | `ALLOWED_ORIGINS`, `_origin` | same test |
| `data.path` must lie inside a folder the user granted (`--allow`, `JEPA_STUDIO_ALLOWED_DIRS`, or lines the shell appends to `JEPA_STUDIO_ALLOWED_FILE` after a native folder-picker), else 403 | `Studio.checked_config`, `Studio.allowed` | `test_security.py::test_server_rejects_paths_outside_grant` |
| Run IDs are single folder names (`[A-Za-z0-9][A-Za-z0-9._-]*`); `..`, separators and drive letters are refused before any file access | `Studio.run`, `RUN_ID` | `test_security.py::test_server_preview_and_unknown_run` |
| A run's config `name` becomes its folder name, so the schema restricts it to letters, digits, `.`, `_`, `-` (no `../`, no absolute paths); the server also checks the new run folder sits directly in `runs/`, and `Trainer` refuses such names from the CLI | `schema/config.schema.json` (`name.pattern`), `Studio.start_run`, `Trainer.__init__` | `test_security.py::test_server_refuses_run_names_that_escape_runs_folder`, `test_web_engine.py::test_config_validator_parity` |
| JSON bodies only (`Content-Type: application/json`, 415 otherwise), capped at 2 MB (413); a missing, non-numeric or negative `Content-Length` is refused (400) | `_body`, `MAX_BODY` | `test_security.py::test_server_rejects_negative_content_length` (the 2 MB cap: code only) |
| No endpoint executes user-supplied code or reaches the network; request bodies are never logged | `server.py` (routes table) | code review |
| The desktop shell restricts what the web view may call: only `tauri-plugin-dialog` is registered, and the main window's capabilities are `allow-pick-data-folder` and `allow-backend-status`; the shell generates the token, passes it to Python in the environment, and appends canonicalized folders picked in the native dialog to the allowed-folders file | `desktop/src-tauri/` (capabilities in `desktop/src-tauri/capabilities/`) | see [desktop.md](desktop.md#security-boundaries); not covered by the Python tests |

**Not yet enforced**

* When `JEPA_STUDIO_TOKEN` is not set, `serve` prints the generated token once on stdout (in
  its ready line) for the shell to read; any process that can read that stdout can use it.
* 500 errors print a traceback to the server's stderr (not to the client).
* Endpoints that train or plan run in background threads with no concurrency limit.

## 5. The assistant and prompt injection

The optional "Explain" assistant answers questions from the docs and your run logs. Logs and
file names are attacker-controllable text (a file named `ignore previous instructions.png`).

| Claim | Where | Test |
|---|---|---|
| Off by default in the UI; retrieval first (BM25), and the default answer is **extractive**: quoted passages with their source | `assistant.answer`, `site/js/tabs/explain.js` | `test_security.py::test_assistant_answers_from_docs_and_has_no_actions` |
| **No tools:** the answer is text plus citations (`{answer, mode, sources}`); nothing in it is wired to an action, so an injected instruction has nothing to act on | `assistant.answer`; the web tab inserts everything with `textContent` | same test (checks the exact key set) |
| Untrusted text (logs, `eval.json`, `hardware.json`) is stripped of control and bidi-override characters, whitespace-collapsed and length-capped (2000/3000 chars); questions are capped at 2000 chars by the server and 500 into the model | `sanitize_untrusted`, `run_passages`, `Studio.assistant` | `test_security.py::test_untrusted_text_is_sanitized` |
| The optional local model (`llama-cpp-python` + `JEPA_STUDIO_LLM=/path/model.gguf`) runs offline, is never installed automatically, sees untrusted passages inside `<untrusted>` tags with a system prompt saying they are data, and its output is sanitized too | `assistant._llm_answer` | code only |

**Not yet enforced**

* The local-model path is not exercised in CI, and the `.gguf` file is not hash-checked before
  llama.cpp parses it.
* Tagging untrusted text reduces, but does not eliminate, the chance that a model repeats an
  injected sentence. The real defence is that the assistant cannot do anything.

## 6. Resource safety

| Claim | Where | Test |
|---|---|---|
| Automatic batch size: doubling search that keeps the probed step under `hardware.max_memory_frac` (default 85%) of RAM (CPU/MPS) or GPU memory (CUDA) | `hardware.memory_probe_batch`, `Trainer.pick_batch_size` | code only |
| Disk: after every checkpoint the run stops safely if its folder exceeds `hardware.max_disk_gb` (default 5 GB) or less than 1 GB is free; only the last 2 checkpoints are kept | `Trainer.check_disk`, `Trainer.save_checkpoint` | code only |
| Temperature: every 20 steps the hottest sensor is read; above `hardware.max_temp_c` (default 85 °C) training sleeps until it is 5 °C cooler (at most 10 minutes per pause), and the pause is recorded | `hardware.Throttle` | code only |
| Quiet mode sleeps half a step after each step (about a two-thirds duty cycle) and uses a quarter of the cores | `Throttle.after_step`, `hardware.recommend` | `test_training.py::test_recommendations_have_reasons` (decisions exist) |
| A non-finite loss stops the run with an error; SIGTERM and the Stop button end it with a checkpoint | `Trainer.fit` | stop path: `test_training.py::test_train_checkpoint_resume_and_control` |
| Live memory guard: at every log step (`train.log_every`, and step 1) this process's RSS is compared with `hardware.max_memory_frac` × total RAM, and on CUDA also PyTorch's allocated GPU memory × the GPU's total; above it the run writes a checkpoint, emits `{"event": "stopped", "reason": "memory guard: ..."}`, saves `encoder.safetensors` and ends with status `stopped` (the reason is also in the `end` event) | `hardware.memory_usage / memory_breach`, `Trainer.fit` | `test_training.py::test_memory_guard_stops_with_checkpoint_and_reason`, `test_memory_breach_reading` |

**Not yet enforced**

* **The memory guard is a check, not a limit.** It runs only every `log_every` steps and counts
  only this process (RSS, plus CUDA memory allocated through PyTorch), so a sudden spike between
  checks, memory used by other programs, the dataset loader's worker processes, MPS memory, or
  the CUDA caching allocator's reserved-but-unallocated memory can still exhaust the machine
  before it triggers. In the desktop app the backend's own memory is part of the same process
  and is counted.
* **Temperature needs sensors.** `psutil.sensors_temperatures()` is only available on Linux and
  FreeBSD, and GPU temperature is read only through `nvidia-smi`. On macOS and Windows (and many
  VMs) no sensor is found and the thermal pause never triggers.
* The disk check runs only when a checkpoint is written (every `checkpoint_every` steps).
