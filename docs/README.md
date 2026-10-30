# JEV Studio documentation

Start with the [project README](../README.md). Then pick what you need.

## JEV, the local AI agent

| Page | What it covers |
|---|---|
| [jev/README.md](jev/README.md) | What JEV is, how it is built, installing Ollama and models on Windows, macOS, Linux and WSL. |
| [jev/SAFETY.md](jev/SAFETY.md) | Sandbox auto-run and the approval safety net: what runs freely, what always asks, what each OS enforces, the audit log, running fully offline. |
| [jev/api.md](jev/api.md) | The `/api/jev/*` routes and event shapes. |
| [jev/ui.md](jev/ui.md) | The JEV and Science tabs: chat, tool cards, approvals, gallery, models and settings. |
| [jev/science.md](jev/science.md) | The Interstellar-style black-hole ray tracer (Kerr geodesics, accretion disk, redshift) and the physics simulations, with validation numbers and references. |
| [jev/media.md](jev/media.md) | Narrated explainer videos, footage editing, images and optional local generative models. |

## JEPA models

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
