# Equations

Every formula on this page is what the code computes, with the file and function it lives in.
Each section goes: intuition, equation, symbols, a worked example with small numbers, and the
picture in the app or the README that shows it. Every number in a worked example is recomputed
by `tests/test_equations_doc.py` with the repo's own functions (and by hand where that is
clearer), so if the code changes and this page doesn't, the test fails.

Notation shared with `jepa_studio/losses.py`:

| Symbol | Meaning | Shape / range |
|---|---|---|
| $N$ | batch size (samples per view) | integer |
| $K$ | projector (embedding) dimension, `model.proj_dim` | integer, default 16 |
| $V$, $V_g$ | number of views, of which $V_g$ are global | defaults 6 and 2 |
| $z_{v,n}$ | projector output for view $v$ of sample $n$ | $\mathbb{R}^K$ |
| $Z_v$ | the batch of embeddings of view $v$ | $N \times K$ |
| $a_m$ | a random unit direction ("slice") | $\mathbb{R}^K$, $\lVert a_m \rVert = 1$ |
| $M$ | slices per SIGReg call, `objective.num_slices` | default 256 |
| $t$ | frequency where characteristic functions are compared | dimensionless, $[0, 3]$ |

All quantities are dimensionless (embeddings have no physical unit); "shape" is the tensor shape.

---

## 1. The LeJEPA objective

**Intuition.** Show the encoder several augmented views of the same input. Two things should
hold: every view should land close to the average of the *global* views of that input (the
prediction term), and the cloud of embeddings of a batch should look like a standard Gaussian
blob in every direction (SIGReg). The prediction term alone is solved by mapping everything to
one point; SIGReg makes that point-cloud impossible. One number, $\lambda$, trades the two.

**Equation** (`losses.lejepa_loss`, paper Algorithm 2):

$$
\mu_n = \frac{1}{V_g}\sum_{g=1}^{V_g} z_{g,n}
$$

$$
\mathcal{L}_{\text{pred}} = \frac{1}{V N K}\sum_{v=1}^{V}\sum_{n=1}^{N}\sum_{k=1}^{K}\left(\mu_{n,k} - z_{v,n,k}\right)^2
$$

$$
\mathcal{L}_{\text{SIGReg}} = \frac{1}{V}\sum_{v=1}^{V} \operatorname{SIGReg}(Z_v)
$$

$$
\mathcal{L} = (1-\lambda)\,\mathcal{L}_{\text{pred}} + \lambda\,\mathcal{L}_{\text{SIGReg}}
$$

Exactly as implemented:

* $z$ is the **projector** output (`JEPAEncoder.projector`, $K$ = `proj_dim`), not the backbone
  embedding. The projector is thrown away after training; probes use the backbone.
* The sum over $v$ runs over **all** $V$ views, globals included, so a global view is also pulled
  toward the global mean.
* The "predictor" is the identity map in embedding space (the paper's default for image
  pretraining). The world model (section 5) is where a learned predictor appears.
* Each view gets its own SIGReg call, and every call draws **fresh** random directions.
* `Trainer.forward_loss` encodes the global views as one batch and the local views as another
  (their sizes differ), then evaluates the loss in float32 even under mixed precision.
* $\lambda$ must be in $[0, 1]$; anything else raises `ValueError`.

Defaults (`config.DEFAULT_CONFIG`): $\lambda = 0.05$, $M = 256$, $t_{\max} = 3.0$, 17 knots,
2 global views at 32 px, 4 local views at 16 px, $K = 16$. (The LeJEPA reference minimal example
and LeVJEPA use $\lambda = 0.02$; treat $\lambda$ as the knob it is.)

**Symbols.**

| Symbol | Code | Shape | Meaning |
|---|---|---|---|
| $z_{g,n}$ | `global_emb[g, n]` | $V_g \times N \times K$ | global-view embeddings |
| $z_{v,n}$ | `all_emb[v, n]` | $V \times N \times K$ | all views (global first, then local) |
| $\mu_n$ | `centers[n]` | $N \times K$ | per-sample mean of the global views |
| $\lambda$ | `objective.lambda` | scalar in $[0,1]$ | SIGReg weight |
| $\mathcal{L}_{\text{pred}}$, $\mathcal{L}_{\text{SIGReg}}$ | `out.prediction`, `out.sigreg` | scalars | the two logged terms (`pred`, `sigreg` in `events.jsonl`) |

**Worked example.** $V_g = 2$ global views, one local view ($V = 3$), $N = 2$ samples, $K = 2$:

| view | sample 1 | sample 2 |
|---|---|---|
| global 1 | (1.0, 0.0) | (−1.0, 0.5) |
| global 2 | (0.6, 0.2) | (−0.8, 0.1) |
| local 1 | (0.9, −0.1) | (−1.2, 0.4) |

Centres: $\mu_1 = (0.8, 0.1)$, $\mu_2 = (-0.9, 0.3)$. Squared differences summed over the
dimensions: global 1 gives 0.05 + 0.05, global 2 gives 0.05 + 0.05, the local view gives
0.05 + 0.10. Total 0.35 over $V N K = 12$ entries, so $\mathcal{L}_{\text{pred}} = 0.35/12 = 0.029167$.

For SIGReg we fix the two directions of section 2's example, $a_1 = (1, 0)$ and $a_2 = (0.6, 0.8)$,
for every view (the trainer would draw new ones). Per view: 0.3382, 0.1993, 0.3305, mean
$\mathcal{L}_{\text{SIGReg}} = 0.2893$. With $\lambda = 0.05$:

$$
\mathcal{L} = 0.95 \times 0.029167 + 0.05 \times 0.2893 = 0.042174
$$

**Where you see it.** The loss equation on the landing page and the Train tab, with each term
underlined in its colour (`docs/STYLE.md`); the three loss charts in the Train tab
(`assets/screens/tab-train.png`); the pipeline in `assets/architecture.svg`.

---

## 2. SIGReg: random slices + the Epps–Pulley test

**Intuition.** Checking "is this 16-dimensional cloud a standard Gaussian?" directly is hard.
The Cramér–Wold theorem says a distribution is pinned down by all its one-dimensional
shadows. So: shine the cloud onto a few random lines, and on each line ask a 1-D question, "does
this histogram look like $\mathcal{N}(0, 1)$?". New random lines every step ("sketching") means
that over training every direction gets checked. The 1-D question is answered by comparing
**characteristic functions**: the Fourier transform of the sample against the Gaussian's
$e^{-t^2/2}$.

**Equation** (`losses.SIGReg`, `losses.EppsPulley`, `losses.trapezoid_half_line`, paper Definition 2):

Directions, redrawn every call (`random_directions`):

$$
a_m = \frac{g_m}{\lVert g_m \rVert}, \qquad g_m \sim \mathcal{N}(0, I_K), \qquad m = 1..M
$$

Projections of one view's batch: $x_{n,m} = a_m^\top z_n$. For one slice, the Epps–Pulley
statistic against $\mathcal{N}(0,1)$ with a Gaussian window $w(t) = e^{-t^2/2}$:

$$
\hat\varphi(t) = \frac{1}{N}\sum_{n=1}^{N} e^{\,i t x_n}, \qquad \varphi(t) = e^{-t^2/2}
$$

$$
\operatorname{EP}(x_1..x_N) = N \int_{-t_{\max}}^{t_{\max}} \bigl\lvert \hat\varphi(t) - \varphi(t) \bigr\rvert^2\, w(t)\, dt
$$

Because $\varphi$ is real, the squared modulus splits into a cosine and a sine part, which is
literally what `per_frequency_error` computes:

$$
\bigl\lvert \hat\varphi(t) - \varphi(t) \bigr\rvert^2 = \Bigl(\tfrac1N \textstyle\sum_n \cos(t x_n) - e^{-t^2/2}\Bigr)^2 + \Bigl(\tfrac1N \textstyle\sum_n \sin(t x_n)\Bigr)^2
$$

The integrand is even in $t$, so the code integrates over $[0, t_{\max}]$ and doubles
(trapezoid rule, $T$ knots, $\Delta t = t_{\max}/(T-1)$):

$$
t_k = k\,\Delta t,\quad
\omega_k = \begin{cases} \Delta t & k = 0 \text{ or } k = T-1 \\ 2\Delta t & \text{otherwise}\end{cases}
\qquad
\operatorname{EP} \approx N \sum_{k=0}^{T-1} \omega_k\, e^{-t_k^2/2}\, \bigl\lvert \hat\varphi(t_k) - \varphi(t_k) \bigr\rvert^2
$$

The buffer `EppsPulley.weights` holds $\omega_k e^{-t_k^2/2}$ (quadrature weight times window).

**Where these numbers come from.** The paper's Algorithm 1 integrates over the full line with
$t \in [-5, 5]$ (17 points). The half-line fold with $t_{\max} = 3$ and 17 knots used here is the
choice made in the authors' reference implementation (rbalestr-lab/lejepa, CC BY-NC 4.0); see
[references.md](references.md) for what that means for reuse. The two give nearly the same value
(0.55819 vs 0.55856 on the worked example below), because the window $e^{-t^2/2}$ is already
below 0.012 at $t = 3$.
Finally

$$
\operatorname{SIGReg}(Z) = \frac{1}{M}\sum_{m=1}^{M} \operatorname{EP}\bigl(\{a_m^\top z_n\}_{n=1}^{N}\bigr)
$$

Implementation details that matter: `lejepa_loss` turns autocast off and upcasts
half-precision embeddings, so the projections $z^\top a$ and the cos/sin sums always run in at
least float32 (`test_losses.py::test_loss_is_float32_under_bf16_autocast`) (the trainer also casts the projector
outputs to float32 before calling the loss); every term is bounded ($\lvert\cos\rvert, \lvert\sin\rvert \le 1$),
so the loss and its gradient are bounded for any input; the cost is $O(N \cdot M \cdot T)$, linear
in the batch size.

**What the numbers mean.** If the projections really are i.i.d. $\mathcal{N}(0,1)$, then for each
$t$, $\mathbb{E}\lvert\hat\varphi - \varphi\rvert^2 = (1 - e^{-t^2})/N$, the $N$ cancels, and

$$
\mathbb{E}[\operatorname{EP}] = \int (1 - e^{-t^2})\, e^{-t^2/2}\, dt
$$

which is $\sqrt{2\pi} - \sqrt{2\pi/3} = 1.0594$ over the whole line and **1.0525** with the
default 17-knot quadrature on $[-3, 3]$ (`ep_expected_under_null`). So a SIGReg value near 1.05
means "looks Gaussian"; a collapsed batch scores in proportion to $N$ (below).

**Symbols.**

| Symbol | Code | Shape | Meaning |
|---|---|---|---|
| $a_m$ | column $m$ of `directions` | $K \times M$ | unit directions, redrawn each call |
| $x_{n,m}$ | `proj = z @ directions` | $N \times M$ | 1-D projections |
| $t_k$ | `EppsPulley.t` | $T$ = `knots` = 17 | frequencies in $[0, t_{\max}]$ |
| $\omega_k e^{-t_k^2/2}$ | `EppsPulley.weights` | $T$ | quadrature weight × Gaussian window |
| $\varphi(t_k)$ | `EppsPulley.phi` | $T$ | $\mathcal{N}(0,1)$ characteristic function |
| $t_{\max}$ | `objective.t_max` | scalar, default 3.0 | integration limit |

**Worked example.**

*Quadrature.* $t_{\max} = 3$, $T = 17$: $\Delta t = 0.1875$, nodes $0, 0.1875, 0.375, \dots, 3$.
Weights $\omega = (0.1875, 0.375, 0.375, \dots, 0.375, 0.1875)$; they sum to $6 = 2 t_{\max}$,
the length of $[-3, 3]$, as they should. Times the window: `weights[0]` = 0.1875,
`weights[1]` = $0.375\, e^{-0.1875^2/2}$ = 0.3685, `weights[16]` = $0.1875\, e^{-4.5}$ = 0.002083.

*Directions.* A raw Gaussian draw $g = (3, 4)$ normalises to $a = (0.6, 0.8)$.

*One slice.* Projections $x = (-1, 0, 1, 2)$, $N = 4$. At the knot $t = 0.75$:
mean $\cos(0.75 x)$ = 0.6335, mean $\sin(0.75 x)$ = 0.2494, $\varphi(0.75)$ = 0.7548, so the
squared error is $(0.6335 - 0.7548)^2 + 0.2494^2$ = 0.0769. At $t = 1.5$: 0.0379, 0.0353, 0.3247,
error 0.0835. At $t = 0$ the error is always 0. Summing all 17 knots with their weights and
multiplying by $N$: $\operatorname{EP} = 0.5582$.

*Collapse.* If all four values are 0, $\hat\varphi(t) = 1$ and each knot contributes
$(1 - e^{-t^2/2})^2$: $\operatorname{EP} = 4 \times 0.40205 = 1.6082$. The same collapsed batch at
$N = 128$ scores 51.46: the penalty for collapse grows with $N$, while the Gaussian baseline
stays at about 1.05.

*SIGReg.* $K = 2$, $N = 4$, $Z$ rows $(1, 0.5), (-1, 1), (0.5, -1.5), (-0.5, 0)$, directions
$a_1 = (1, 0)$, $a_2 = (0.6, 0.8)$. Projections on $a_1$: $(1, -1, 0.5, -0.5)$, EP = 0.0619;
on $a_2$: $(1, 0.2, -0.9, -0.3)$, EP = 0.1547. $\operatorname{SIGReg}(Z) = 0.1083$.

**Where you see it.** `assets/sigreg.gif` (a batch's 1-D projections turning Gaussian while the
statistic falls); the hero figure on the landing page (a real Epps–Pulley value on live samples);
the "projections turn Gaussian" player in the Inspect tab (`assets/screens/tab-inspect.png`),
fed by the `proj_hist` histograms in `events.jsonl`.

---

## 3. Why an isotropic Gaussian?

**Intuition.** After pretraining, someone will fit a simple model (a linear probe, a k-NN) on
your embeddings for a task you don't know yet. The LeJEPA paper asks which embedding
distribution makes that downstream fit best in the *worst case* over tasks, and its answer is
the isotropic Gaussian $\mathcal{N}(0, I)$. SIGReg is simply the most direct way to push the
embeddings there.

**The argument, in plain language** (our summary; the paper has the exact theorems and their
assumptions, so read it before quoting any of this as a theorem):

1. *Linear probes dislike stretched clouds.* For least squares on features with covariance
   $\Sigma$, the estimator's variance scales with $\operatorname{tr}(\Sigma^{-1})$. At a fixed total
   variance $\operatorname{tr}(\Sigma)$, that trace is smallest when all eigenvalues are equal.
   Thin directions (tiny eigenvalues) blow it up. *Simplification:* this is the ordinary
   least-squares textbook case; the paper treats ridge-regularised probes and bias as well.
2. *Nonlinear probes (k-NN, kernels) also do best with isotropy,* and among isotropic choices the
   paper argues for the Gaussian shape. *Simplification:* we are not reproducing that proof here.
3. *Collapse is the extreme of anisotropy.* A collapsed or low-rank cloud has zero variance in
   some direction, so it is as far from $\mathcal{N}(0, I)$ as it gets, and SIGReg rules it out
   without an EMA teacher, stop-gradient or negative pairs.
4. *Why 1-D tests suffice* (Cramér–Wold): $Z \sim \mathcal{N}(0, I_K)$ if and only if
   $a^\top Z \sim \mathcal{N}(0, 1)$ for every unit vector $a$.

**Equation** (projected variance, the quantity each slice checks):

$$
\operatorname{Var}(a^\top z) = a^\top \Sigma\, a \quad\text{should be } 1 \text{ for every unit } a
\iff \Sigma = I \ \text{(for the second moment)}
$$

**Symbols.** $\Sigma$: covariance of the embeddings ($K \times K$); $a$: unit direction ($K$);
$\lambda_i$: eigenvalues of $\Sigma$.

**Worked example.** A cloud with $\Sigma = \operatorname{diag}(1, 0.01)$ looks fine along the first
axis, but along $a = (0.6, 0.8)$ its variance is $0.36 \times 1 + 0.64 \times 0.01 = 0.3664$, far
from 1, so that slice's Epps–Pulley statistic is large. For the probe argument: two 2-D spectra
with the same total variance 2, $(1.9, 0.1)$ and $(1, 1)$, give
$\operatorname{tr}(\Sigma^{-1}) = 1/1.9 + 1/0.1 = 10.53$ versus $2$: about five times the estimator
variance for the stretched cloud.

**Where you see it.** `assets/jepa-vs-pixels.svg` (predicting embeddings instead of pixels);
the eigenvalue bars and collapse verdict in the Inspect tab (`assets/screens/tab-inspect.png`).

---

## 4. Evaluation: linear probe, k-NN, effective rank, collapse detector

All in `jepa_studio/evaluate.py`. Protocol (`evaluate_all`): 20% of the items are held out
(`split_indices`, `test_fraction = 0.2`); a labeled subset of `data.labeled_fraction` (default
10%, at least 10 items) of the rest trains the probe and serves as k-NN memory. Probes and k-NN
use **backbone embeddings of un-augmented inputs**; the isotropy numbers and the collapse verdict
use the **projector outputs** (where SIGReg acts). Labels are never used in pretraining.

### 4a. Linear probe

**Intuition.** Freeze the encoder and ask whether a straight line (hyperplane) in embedding space
separates the classes.

**Equation** (`standardize`, `linear_probe`):

$$
\tilde e = \frac{e - \bar e_{\text{train}}}{s_{\text{train}} + 10^{-6}}, \qquad
p(y = c \mid e) = \operatorname{softmax}_c(W \tilde e + b)
$$

Trained with full-batch AdamW (learning rate $10^{-2}$, weight decay $10^{-6}$) on cross-entropy
for `eval.probe_epochs` (default 100) epochs, $W$ initialised from $\mathcal{N}(0, 0.01^2)$, $b = 0$.
Score: accuracy on the held-out split, $\frac{1}{n_{\text{test}}}\sum \mathbb{1}[\arg\max_c = y]$.
Mean and standard deviation (unbiased, $n-1$) come from the labeled training features only.

**Symbols.** $e$: backbone embedding (`embed_dim`, default 128); $W$: $C \times$ `embed_dim`;
$C$: number of classes.

**Worked example.** Train features $(1, 2), (3, 4), (5, 9)$: mean $(3, 5)$, std $(2, 3.6056)$,
so $(5, 9)$ becomes $(1.0000, 1.1094)$ and $(1, 2)$ becomes $(-1.0000, -0.8321)$.

### 4b. k-nearest-neighbor (kNN) accuracy

**Intuition.** Label a test point by asking its closest labeled neighbours to vote.

**Equation** (`knn_accuracy`):

$$
s_j = \frac{q^\top x_j}{\lVert q \rVert \lVert x_j \rVert}, \qquad
\hat y = \arg\max_c \Bigl(\#\{j \in \text{top-}k(s) : y_j = c\} + 10^{-3}\,\mathbb{1}[c = y_{j^*}]\Bigr)
$$

where $j^*$ is the single most similar neighbour: ties go to its class. $k = \min(k, n_{\text{train}})$,
default `eval.knn_k` = 20.

**Worked example.** Memory $(1, 0)$ class 0, $(0.8, 0.6)$ class 1, $(0, 1)$ class 1, $(-1, 0)$
class 2; query $q = (1, 0.2)$. Cosine similarities: 0.9806, 0.9021, 0.1961, −0.9806.
With $k = 3$ the votes are class 1: 2, class 0: 1, so $\hat y = 1$. With $k = 2$ it is a 1–1 tie,
broken toward the most similar neighbour, so $\hat y = 0$.

### 4c. Effective rank and isotropy

**Intuition.** Count how many directions the embedding cloud really uses, and how lopsided it is.

**Equation** (`isotropy_report`, effective rank after Roy & Vetterli, 2007):

$$
\Sigma = \frac{1}{n-1} Z_c^\top Z_c,\quad \lambda_1 \ge \dots \ge \lambda_K \ge 0,\quad
p_i = \frac{\lambda_i}{\sum_j \lambda_j}
$$

$$
\operatorname{erank} = \exp\Bigl(-\sum_i p_i \ln p_i\Bigr) \in [1, K], \qquad
\text{isotropy ratio} = \frac{\lambda_K}{\lambda_1} \in [0, 1], \qquad
\text{top share} = p_1
$$

The report also carries `mean_std` (mean per-dimension standard deviation), `mean_norm`,
`mean_abs_mean` and the SIGReg statistic of the whole set (1024 slices). That SIGReg value is
$N$ times a per-sample deviation plus about 1.05 (section 2), so it is only comparable between
sets of the same size.

**Worked example.** $Z$ is 8 × 4 with columns $\pm\sqrt{7/4}$, $\pm\sqrt{7/8}$, $\pm\sqrt{7/16}$,
$\pm\sqrt{7/16}$ in the sign patterns of four columns of an 8 × 8 Hadamard matrix (orthogonal,
zero mean), so $\Sigma = \operatorname{diag}(2, 1, 0.5, 0.5)$ exactly. Then $p = (0.5, 0.25, 0.125, 0.125)$,
entropy 1.2130, effective rank 3.364 of 4, isotropy ratio 0.25, top share 0.5, mean std 0.9571.

### 4d. Collapse detector

**Intuition.** Turn those numbers into a plain-language verdict, checked in this order.

**Rules** (`collapse_check`, first match wins):

| Verdict | Condition |
|---|---|
| complete collapse | `mean_std` < 0.001 |
| dimensional collapse | effective rank < max(1.5, 0.1 K) **or** top share > 0.9 |
| anisotropic | isotropy ratio < 0.01 |
| healthy | none of the above |

For the default $K = 16$ the effective-rank threshold is max(1.5, 1.6) = 1.6.

**Worked example.** The 8 × 4 cloud above: 3.364 ≥ 1.5, top share 0.5, ratio 0.25, so
**healthy**. A spectrum $(3.8, 0.1, 0.05, 0.05)$: top share 0.95 > 0.9 (effective rank 1.285),
so **dimensional collapse**. A spectrum $(1, 1, 1, 0.005)$: effective rank 3.032, top share 0.3328,
but ratio 0.005 < 0.01, so **anisotropic**. Eight identical rows: `mean_std` = 0, so **complete
collapse**.

**Where you see it.** The Inspect tab's verdict badge and eigenvalue bars
(`assets/screens/tab-inspect.png`); the Evaluate tab's table (`assets/screens/tab-evaluate.png`);
`jepa-studio inspect` and `jepa-studio eval` print the same dicts.

---

## 5. Planning in a latent world model: cost and CEM

**Intuition.** The world model imagines: from the current frame's embedding $z_t$ and a candidate
action sequence it predicts the next embeddings without seeing new frames. A good plan is one
whose imagined embeddings get close to the goal frame's embedding and stay close. The
Cross-Entropy Method (CEM) searches for it by sampling many action sequences, keeping the best
few, refitting a Gaussian to them and repeating. Only the first action is executed, then the bot
looks again and replans (receding horizon, model-predictive control).

**Equation** (`world/planner.py`: `trajectory_cost`, `plan_cem`, `plan_and_execute`):

Imagination (`WorldModel.rollout`, predictor $g$ with a residual connection):

$$
\hat z_0 = z_t = f(o_t),\qquad \hat z_{h+1} = \hat z_h + \operatorname{MLP}([\hat z_h, a_h]),\qquad h = 0..H-1
$$

Cost, with $z_g = f(o_{\text{goal}})$:

$$
C_{\text{sum}}(a) = \sum_{h=1}^{H} \lVert \hat z_h - z_g \rVert^2 \quad\text{(default)},\qquad
C_{\text{final}}(a) = \lVert \hat z_H - z_g \rVert^2
$$

CEM, one replanning step, `iters` rounds:

$$
a^{(i)} = \operatorname{clip}\bigl(\mu + \sigma \odot \varepsilon^{(i)},\, -1,\, 1\bigr),\quad
\varepsilon^{(i)} \sim \mathcal{N}(0, I),\ \varepsilon^{(0)} = 0,\quad i = 0..S-1
$$

$$
E = \text{the } n_e \text{ samples with lowest } C,\qquad
\mu \leftarrow \operatorname{mean}(E),\qquad
\sigma \leftarrow \operatorname{std}(E) + \sigma_{\min}
$$

`std` is the population standard deviation (`unbiased=False`). Sample 0 is always the current
mean. The plan returned is the lowest-cost sample seen in any round; its first action is
executed. The next step warm-starts $\mu$ from the previous $\mu$ shifted by one step (zeros
appended); $\sigma$ restarts at $\sigma_0$.

**Symbols.**

| Symbol | Code | Default | Meaning |
|---|---|---|---|
| $H$ | `horizon` (`world.horizon`) | 5 | steps imagined |
| $S$ | `samples` (`world.cem_samples`) | 300 | action sequences per round |
| $n_e$ | `elites` (`world.cem_elites`) | 30 | kept per round |
| iters | `iters` (`world.cem_iters`) | 10 | rounds per replanning step |
| $\sigma_0$, $\sigma_{\min}$ | `init_std`, `min_std` | 1.0, 0.05 | initial and floor standard deviation |
| $a_h$ | `acts` | $S \times H \times 2$, in $[-1, 1]$ | velocity actions |
| $\hat z_h$, $z_g$ | `zs`, `z_goal` | $d$ = 16 for the shipped models | imagined and goal latents |

**Worked example.**

*Cost.* $d = 2$, goal $z_g = (0, 0.2)$, $H = 3$. Plan A imagines $(1, 1), (0.5, 0.5), (0, 0.2)$:
squared distances 1.64, 0.34, 0, so $C_{\text{sum}}$ = 1.98 and $C_{\text{final}}$ = 0. Plan B
imagines $(2, 2), (1, 1), (0, 0.2)$: $C_{\text{sum}}$ = 8.88, $C_{\text{final}}$ = 0 again. The final
cost cannot tell them apart; the sum prefers A, which gets close sooner. Under receding horizon
only the first action runs, so that is the behaviour we want.

*One CEM round.* One action dimension, $H = 1$, a toy model $\hat z_1 = z_0 + a$ with $z_0 = 0$
and goal 0.4, so $C = (a - 0.4)^2$. Six samples $a = (-0.9, -0.4, 0.1, 0.3, 0.6, 1.0)$ cost
$(1.69, 0.64, 0.09, 0.01, 0.04, 0.36)$. Keep $n_e = 2$ elites: 0.3 and 0.6. New $\mu = 0.45$,
new $\sigma = 0.15 + 0.05 = 0.20$; the logged elite-mean cost is 0.025. A draw $\varepsilon = 3$
from the new Gaussian gives $0.45 + 0.2 \times 3 = 1.05$, clipped to 1.0.

*Budget.* With the defaults, one replanning step calls the predictor
$S \times H \times \text{iters} = 300 \times 5 \times 10 = 15\,000$ times (batched).

**Where you see it.** `assets/world-two-room.gif`, `assets/world-push-block.gif` (real frames,
goal, and the *retrieved* nearest real frames for the imagined latents: a JEPA has no decoder);
the World model tab's cost-per-iteration and distance charts (`assets/screens/tab-world.png`).
Training of the world model itself is in [world-model.md](world-model.md).
