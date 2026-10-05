# Reference repositories

These repositories were cloned and read while building jepa-studio. The licence column comes
from each repository's own `LICENSE` file (and its README where the README carves out
exceptions), at the commit listed. "Code reuse" says whether their licence would allow copying
code into this MIT project; "what we did" says what actually happened.

**Rule we follow: no code from non-commercial (CC BY-NC) repositories is copied into jepa-studio,
and no weights from them are shipped.** Where we implement the same maths, it is written from the
paper's equations.

| Repository (commit) | Licence (from `LICENSE`) | What we borrow | Code reuse allowed in MIT? | What we did | Weights |
|---|---|---|---|---|---|
| [rbalestr-lab/lejepa](https://github.com/rbalestr-lab/lejepa) (`c293d29`) | **CC BY-NC 4.0** (Attribution-NonCommercial 4.0 International) | The numerical recipe of its SIGReg (integrate on $[0, 3]$ with 17 trapezoid knots, fold $[-t_{\max}, t_{\max}]$ onto the half line, scale by $N$), and the multi-crop defaults (2 global + several local views). The paper's Algorithm 1 uses $t \in [-5, 5]$ instead | **No** (non-commercial) | `losses.py` is our own code, written with this repo at hand; its `EppsPulley` class closely parallels the reference `SIGReg` class. Not a copy, but have it reviewed or re-derived before **commercial** use | None used |
| [lucas-maes/le-wm](https://github.com/lucas-maes/le-wm) (`8edfeb3`) | MIT, © 2026 Lucas Maes | The LeWorldModel recipe (two-term loss, end-to-end from pixels) and the planner's evaluation defaults (horizon 5, 300 samples, 30 elites) | Yes, with the copyright notice | Re-implemented at toy scale in `world/`; no code copied | None used. Their checkpoints are pickle `.ckpt` files on Hugging Face, which jepa-studio refuses to load (only `import-legacy`, explicitly) |
| [MLO-lab/LeVJEPA](https://github.com/MLO-lab/LeVJEPA) (`3ea0dda`) | MIT, © 2026 GalilAI-group, **except** `module.py` (adapted from V-JEPA) which the README says remains **CC BY-NC 4.0** | Context only (LeJEPA on video; paper identification for arXiv:2608.27395) | MIT parts yes; **`module.py` no** | Nothing copied | None used. The released LeVJEPA-VideoMix-Large weights are **CC BY-NC 4.0** per the README |
| [klindtlab/lejepa-identifiability](https://github.com/klindtlab/lejepa-identifiability) (`de7503f`) | MIT, © 2026 David Klindt | Reading on linear identifiability; motivation for the linear state probe in `world/model.collapse_report` | Yes, with notice | Nothing copied | n/a |
| [lucidrains/x-jepa](https://github.com/lucidrains/x-jepa) (`8fecb09`) | MIT, © 2026 Phil Wang | Survey of JEPA variants in small control environments (work in progress upstream) | Yes, with notice | Nothing copied | n/a |
| [AbdelStark/awesome-jepa](https://github.com/AbdelStark/awesome-jepa) (`8eadde1`) | No `LICENSE` file; the README declares **CC0 1.0** (public domain dedication) | Reading list used to find primary sources | Yes | Nothing copied; every paper we cite was checked at its primary source ([bibliography.md](bibliography.md)) | n/a |
| [facebookresearch/ijepa](https://github.com/facebookresearch/ijepa) (`52c1ae9`) | **CC BY-NC 4.0** | The concept (predict in representation space) | **No** (non-commercial) | Nothing copied | None used (the README links pickle `.pth.tar` checkpoints) |
| [facebookresearch/vjepa2](https://github.com/facebookresearch/vjepa2) (`204698b`) | MIT, © Meta Platforms; per the README three files (`src/datasets/utils/video/randaugment.py`, `randerase.py`, `src/datasets/utils/worker_init_fn.py`) are Apache-2.0 (`APACHE-LICENSE`) | The idea of goal-image planning with CEM and model-predictive control in a JEPA latent space | Yes (MIT, and Apache-2.0 for those files, with notices) | Nothing copied; `world/planner.py` is written from the CEM description | None used |
| [tauri-apps/tauri](https://github.com/tauri-apps/tauri) (`f040897`) | Dual **MIT or Apache-2.0** (`LICENSE-MIT`, `LICENSE-APACHE-2.0`, `LICENSE.spdx`) | The desktop shell framework and its capability (permission allowlist) model | Yes | Used as a dependency of `desktop/` (see [desktop.md](desktop.md)); no source copied | n/a |

## Other third-party material in this repository

| Item | Where | Licence |
|---|---|---|
| Latin Modern Roman fonts (WOFF2 subsets) | `site/fonts/` | GUST Font License (text in `site/fonts/GUST-FONT-LICENSE.txt`) |
| mulberry32 PRNG | `data/synthetic.py`, `world/envs.py`, `site/js/engine/rng.js` | Public-domain snippet, re-typed |

Everything else is original to jepa-studio and MIT-licensed ([LICENSE](../LICENSE)).
