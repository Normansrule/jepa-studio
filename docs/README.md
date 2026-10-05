# jepa-studio documentation

Start with the [project README](../README.md). Then pick what you need:

| Page | What it covers |
|---|---|
| [equations.md](equations.md) | The maths exactly as implemented: the LeJEPA objective, SIGReg (random slices + Epps–Pulley), why an isotropic Gaussian, linear probe, k-NN, effective rank and the collapse detector, latent planning cost and CEM. Every worked example is re-checked by `tests/test_equations_doc.py`. |
| [data.md](data.md) | Loading images, video clips and time series; labels from subfolders; the global/local views; file limits. |
| [hardware.md](hardware.md) | The hardware card, every automatic setting and its rule, modes and thermal pause, the throughput benchmark, learning-rate schedule, checkpoints, pause/stop/resume. |
| [world-model.md](world-model.md) | The two toy environments, the action-conditioned JEPA world model, its training loss, the CEM planner and the measured results. |
| [web-demo.md](web-demo.md) | What trains in the browser tab and how it differs from the desktop app. |
| [desktop.md](desktop.md) | The desktop app (Tauri shell + local Python backend): install, one-click setup, capabilities. |
| [api.md](api.md) | The local HTTP API the desktop front end talks to. |
| [SECURITY_MODEL.md](SECURITY_MODEL.md) | Threats, defences, the file and test that enforce each, and what is not enforced yet. |
| [bibliography.md](bibliography.md) | Every paper we use, verified at its primary source, and what we use it for. |
| [references.md](references.md) | The reference repositories, their licences, and what we borrow (no code from non-commercial ones). |
| [STYLE.md](STYLE.md) | Colours, type and chart rules shared by the app and the figures. |

The in-app **Explain** assistant searches these pages. After editing any of them, rebuild its
index with `python scripts/build_docs_index.py` (CI fails if `site/docs-index.json` is stale).
