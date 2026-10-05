# Bibliography

Every entry was checked against its primary source (the arXiv abstract page, the publisher's
page or the DOI record) on 2026-09-30. "Used for" says what jepa-studio actually takes from it
and where; "cited in code" means the source file names it.

## Methods this repo implements

| Work | Authors | Year | ID | Used for |
|---|---|---|---|---|
| **LeJEPA: Provable and Scalable Self-Supervised Learning Without the Heuristics** | Randall Balestriero, Yann LeCun | 2025 | [arXiv:2511.08544](https://arxiv.org/abs/2511.08544) | The whole pretraining objective: prediction term + Sketched Isotropic Gaussian Regularization (SIGReg), Definition 2 and Algorithm 2, written from the equations in `jepa_studio/losses.py` (cited in code). Also the argument for an isotropic Gaussian target ([equations.md §3](equations.md#3-why-an-isotropic-gaussian)) and the linear-probe protocol that discards the projector (`evaluate.py`). |
| **LeWorldModel: Stable End-to-End Joint-Embedding Predictive Architecture from Pixels** | Lucas Maes, Quentin Le Lidec, Damien Scieur, Yann LeCun, Randall Balestriero | 2026 | [arXiv:2603.19312](https://arxiv.org/abs/2603.19312) | The world-model recipe in `jepa_studio/world/model.py` (cited in code): next-latent prediction + SIGReg trained end-to-end from pixels, no decoder, no EMA target, no stop-gradient; and the planner's evaluation defaults (horizon 5, 300 samples, 30 elites; `world/__init__.py`). |
| **A test for normality based on the empirical characteristic function** | T. W. Epps, Lawrence B. Pulley | 1983 | *Biometrika* 70(3):723–726, [doi:10.1093/biomet/70.3.723](https://doi.org/10.1093/biomet/70.3.723) | The 1-D goodness-of-fit statistic inside SIGReg (`losses.EppsPulley`, cited in code). |
| **Some Theorems on Distribution Functions** | H. Cramér, H. Wold | 1936 | *J. London Math. Soc.* s1-11(4):290–294, [doi:10.1112/jlms/s1-11.4.290](https://doi.org/10.1112/jlms/s1-11.4.290) | The Cramér–Wold theorem: a distribution is determined by its 1-D projections, which is why random slices suffice (`losses.SIGReg` docstring, cited in code by name). |
| **The Effective Rank: a Measure of Effective Dimensionality** | Olivier Roy, Martin Vetterli | 2007 | EUSIPCO 2007, Poznań; [doi:10.5281/zenodo.40328](https://doi.org/10.5281/zenodo.40328) | `effective_rank = exp(entropy of normalised eigenvalues)` in `evaluate.isotropy_report` and `world/model.collapse_report` (cited in code). |
| **The Cross-Entropy Method for Combinatorial and Continuous Optimization** | Reuven Rubinstein | 1999 | *Methodol. Comput. Appl. Probab.* 1(2):127–190, [doi:10.1023/A:1010091220143](https://doi.org/10.1023/A:1010091220143) | The Cross-Entropy Method (CEM) planner in `world/planner.py` and `site/js/world/planner.js` (the method is named in code; the paper is not). |
| **The Probabilistic Relevance Framework: BM25 and Beyond** | Stephen Robertson, Hugo Zaragoza | 2009 | *Found. Trends Inf. Retr.* 3(4):333–389, [doi:10.1561/1500000019](https://doi.org/10.1561/1500000019) | BM25 ranking in the Explain assistant (`assistant.bm25`, `site/js/tabs/explain.js`; method named in code). |
| **Decoupled Weight Decay Regularization** | Ilya Loshchilov, Frank Hutter | 2017 (ICLR 2019) | [arXiv:1711.05101](https://arxiv.org/abs/1711.05101) | AdamW, the optimizer for pretraining, probes and the world model (`torch.optim.AdamW`; `site/js/engine/adamw.js`). |
| **Delving Deep into Rectifiers: Surpassing Human-Level Performance on ImageNet Classification** | Kaiming He, Xiangyu Zhang, Shaoqing Ren, Jian Sun | 2015 | [arXiv:1502.01852](https://arxiv.org/abs/1502.01852) | Kaiming (He) initialisation of the world-model encoder (`nn.init.kaiming_normal_` in `world/model.ConvEncoder`), which keeps the initial latent spread away from collapse. |
| **Gaussian Error Linear Units (GELUs)** | Dan Hendrycks, Kevin Gimpel | 2016 | [arXiv:1606.08415](https://arxiv.org/abs/1606.08415) | The GELU activation in the encoders and projector (`models.py`; exact erf form in the browser engine). |

## Background: the JEPA line

| Work | Authors | Year | ID | Used for |
|---|---|---|---|---|
| **Self-Supervised Learning from Images with a Joint-Embedding Predictive Architecture** (I-JEPA) | Mahmoud Assran, Quentin Duval, Ishan Misra, Piotr Bojanowski, Pascal Vincent, Michael Rabbat, Yann LeCun, Nicolas Ballas | 2023 | [arXiv:2301.08243](https://arxiv.org/abs/2301.08243) | The core idea shown in `assets/jepa-vs-pixels.svg`: predict in representation space, not pixel space. No code or weights used (the reference code is CC BY-NC 4.0). |
| **V-JEPA 2: Self-Supervised Video Models Enable Understanding, Prediction and Planning** | Mido Assran, Adrien Bardes, David Fan, Quentin Garrido, Russell Howes, Mojtaba Komeili, Matthew Muckley, Ammar Rizvi, Claire Roberts, Koustuv Sinha, Artem Zholus, Sergio Arnaud, Abha Gejji, Ada Martin, Francois Robert Hogan, Daniel Dugas, Piotr Bojanowski, Vasil Khalidov, Patrick Labatut, Francisco Massa, Marc Szafraniec, Kapil Krishnakumar, Yong Li, Xiaodong Ma, Sarath Chandar, Franziska Meier, Yann LeCun, Michael Rabbat, Nicolas Ballas | 2025 | [arXiv:2506.09985](https://arxiv.org/abs/2506.09985) | Background for planning with model-predictive control in a JEPA latent space toward an image goal (its repo's planning notebook uses CEM). No code or weights used. |
| **LeVJEPA: Efficient & Scalable Video Pretraining without the Heuristics** | Lukas Kuhn, Lucas Maes, Giuseppe Serra, Quentin Le Lidec, Yann LeCun, Randall Balestriero, Florian Buettner | 2026 | [arXiv:2608.27395](https://arxiv.org/abs/2608.27395) | Context for the `video` data kind: LeJEPA's objective applied to video. Identified from the cloned LeVJEPA README and confirmed on arXiv. jepa-studio's `video-convnet` is a tiny per-frame model, not LeVJEPA's architecture; no code or weights used. |
| **When Does LeJEPA Learn a World Model?** | David Klindt, Yann LeCun, Randall Balestriero | 2026 | [arXiv:2605.26379](https://arxiv.org/abs/2605.26379) | Related reading on linear identifiability of LeJEPA latents. The world model's `state_probe_r2` (ridge R² of the true env state from the latent) measures that kind of linear decodability; the code does not cite the paper. |

## Notes

* Years are the arXiv first-submission year (or the journal year).
* The LeJEPA reference implementation is CC BY-NC 4.0; `losses.py` was written from the paper's
  equations, and no code was copied. See [references.md](references.md) for every reference
  repository, its licence and what we borrow.
* "mulberry32", the 32-bit pseudo-random generator used for bit-identical Python/JavaScript data
  (`data/synthetic.py`, `world/envs.py`, `site/js/engine/rng.js`), is a public-domain snippet
  rather than a paper, so it has no entry here.
