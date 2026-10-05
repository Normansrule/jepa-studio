# World model and planning bot

A JEPA world model learns, from pixels and actions only, to predict the *embedding* of the next
frame. A planner then searches for actions whose imagined embeddings reach the embedding of a
goal image. Code: `jepa_studio/world/` (Python) and `site/js/world/` (the browser port).
Recipe: LeWorldModel (Maes et al., arXiv:2603.19312), re-implemented at toy scale.

```
jepa-studio world --env two-room            # train + evaluate + export, into runs/world-two-room/
jepa-studio world --env push-block --out runs/world-push-block
```

## 1. Environments (`world/envs.py`)

Two tiny, deterministic 2-D worlds rendered as 32 × 32 RGB images. Both are *pure*
(`step(state, action)` never mutates anything), and `reset(seed)` uses the mulberry32 generator,
so Python and the browser replay the same trajectories (`tests/test_world_parity.py`).

| | two-room | push-block |
|---|---|---|
| picture | a red agent dot; a vertical wall at x = 0.5 with a door around y = 0.5 | a red agent dot and a blue square block in an open arena |
| state | agent (x, y) | agent (x, y), block (x, y) |
| action | velocity in $[-1, 1]^2$, × 0.08 arena units per step | same |
| dynamics | the wall blocks the agent except through the door (agent radius 0.04, door half-height 0.12) | a kinematic push: overlapping the block moves it along the axis of least penetration; a block stuck at the border pushes the agent back |
| goal | a point in either room, 0.2–0.6 from the start | the block moved 0.1–0.2 along one axis, agent resting behind it |
| success | agent within 0.08 of the goal | **block** within 0.06 of its goal position (agent anywhere) |
| exploration data | smoothed random walk; in about half the episodes it first heads through the door | in about 75% of episodes, walk behind the block and push for a few steps, repeatedly; otherwise a random walk |

## 2. Model (`world/model.py`)

$$
z_t = f(o_t), \qquad \hat z_{t+1} = g(z_t, a_t) = z_t + \operatorname{MLP}([z_t, a_t])
$$

* **Encoder $f$:** fixed input standardisation (the mean training image and one scalar pixel
  standard deviation, stored as buffers), three stride-2 4 × 4 convolutions (32 → 16 → 8 → 4 px,
  16/32/32 channels, ReLU), then Linear → ReLU → Linear to $z \in \mathbb{R}^{d}$. No BatchNorm, so a
  single frame can be encoded on its own (and in the browser). Kaiming initialisation. Without the
  input standardisation, frames that are ~95% background map to nearly the same latent at
  initialisation and training stays collapsed (see the `ConvEncoder` docstring).
* **Predictor $g$:** residual MLP on the concatenation of latent and action, 2 hidden layers of 64
  ReLU units. The last layer starts at zero, so an untrained predictor is the "nothing changes"
  (copy) baseline.
* The shipped models: $d = 16$, 99 568 parameters each.

## 3. Training (the two-term loss)

For a window of $W$ consecutive frames and the $W-1$ actions between them (`world_loss`):

$$
\mathcal{L} = (1-\lambda)\,\underbrace{\operatorname{mean}_{b,k,j}\bigl(\hat z_k - z_k\bigr)^2}_{\text{open-loop prediction}}
+ \lambda\,\underbrace{\operatorname{SIGReg}\bigl(\{z\}_{B \cdot W}\bigr)}_{\text{latents} \to \mathcal{N}(0, I)}
$$

where $\hat z_k = g^k(z_0, a_0..a_{k-1})$ is rolled out from the first frame only (open loop,
$k = 1..W-1$), exactly how the planner uses the model. There is no decoder, no reconstruction
loss, no EMA target network and no stop-gradient: gradients flow into the targets $z_k$ too
(`tests/test_world.py::test_gradients_flow_through_targets`). The prediction term alone is
minimised by a constant encoder; SIGReg (the same `losses.SIGReg` as pretraining) forbids it.

Settings for the shipped models (`HPARAMS_BY_ENV`; not part of the shared config schema):

| | two-room | push-block |
|---|---|---|
| episodes × length | 400 × 24 | 800 × 24 |
| window $W$ (open-loop steps) | 6 (5) | 8 (7) |
| batch (windows) | 48 | 32 |
| $\lambda$, SIGReg slices | 0.2, 64 | 0.2, 64 |
| steps, learning rate | 3000, 2e-3 (100 warm-up, cosine to 1%) | same |
| optimizer | AdamW, weight decay 1e-4, gradient clipping at norm 1 | same |
| threads | 2 | 2 |

The `world` block of the run config holds only the environment, data and planner keys
(`env`, `episodes`, `episode_len`, `horizon`, `cem_samples`, `cem_elites`, `cem_iters`); the
default `episodes` is 400. Without `--config`, `jepa-studio world --env <env>` uses the same
episode budget as the shipped models (`EPISODES_BY_ENV`: two-room 400, push-block 800). To
rebuild the shipped artifacts exactly, run

```
jepa-studio world --env two-room --site
jepa-studio world --env push-block --site
# or both at once:
python -c "from jepa_studio.world import build_site_models; build_site_models()"
```

which retrains both, rewrites `site/models/world-*.json`, `site/models/world-metrics.json`,
`runs/world-*/` and the two GIFs in `assets/`. 10% of the episodes are held out for the
prediction and collapse metrics.

## 4. Planner (`world/planner.py`)

The Cross-Entropy Method (CEM) with receding horizon: plan $H$ = 5 steps with 300 samples, 30
elites and 10 rounds, execute only the first action in the **real** environment, re-encode the
new real frame, replan (warm-started from the previous plan shifted by one step). An episode
ends at success or after 40 steps. The cost is the summed squared distance of the imagined
latents to the goal latent. Every formula, with a worked example, is in
[equations.md §5](equations.md#5-planning-in-a-latent-world-model-cost-and-cem).

**Imagination is shown by retrieval, not generation.** A JEPA has no decoder, so imagined
latents cannot be turned into pixels. The GIFs and the World tab show, for each imagined
latent, the real training frame whose embedding is nearest (a bank of 1024 encoded real
frames), and label it "retrieval, not generation".

## 5. Results (from `runs/world-*/metrics.json`)

Both models trained on 2 CPU threads. Planning was scored on 20 episodes (start/goal seeds
10000–10019, disjoint from the data-collection seeds in practice), 40 steps max, against uniform
random actions on the same episodes.

| | two-room | push-block |
|---|---|---|
| training time | 133.8 s | 104.2 s |
| final loss / prediction / SIGReg (last logged step) | 0.754 / 0.441 / 2.006 | 0.891 / 0.673 / 1.763 |
| held-out open-loop latent MSE, 1 → 5 steps | 0.519 → 0.527 | 0.704 → 0.683 |
| "copy the current latent" baseline, 1 → 5 steps | 0.974 → 1.566 | 1.210 → 1.696 |
| model ÷ copy, 1 → 5 steps | 0.533 → 0.336 | 0.582 → 0.403 |
| latent effective rank (of 16), collapsed? | 15.56, no | 15.81, no |
| linear probe R² of the true state from $z$ | agent x 0.785, y 0.581 | agent x 0.737, y 0.784; block x 0.190, y 0.376 |
| **CEM success rate** | **0.80** (16 / 20) | **0.25** (5 / 20) |
| random-action success rate | 0.15 | 0.00 |
| mean final distance: CEM / random / at start | 0.144 / 0.380 / 0.428 | 0.189 / 0.158 / 0.151 |
| mean steps to success (CEM) | 8.94 | 3.8 |
| mean planning time per replanning step (Python) | 11.5 ms | 11.8 ms |

What this does and doesn't show:

* **two-room works:** the planner reaches the goal in 16 of 20 episodes, versus 3 of 20 for random
  actions, and ends much closer on average.
* **push-block is weak.** CEM solves 5 of 20 episodes (random: 0), but its *mean* final block
  distance (0.189) is worse than random (0.158) and worse than not moving at all (0.151): when it
  fails, it often pushes the block the wrong way. The latent encodes the block's position poorly
  (R² 0.19 / 0.38), which is the likely bottleneck. Treat it as an honest baseline to improve on.
* 20 episodes is a small sample; one seed, one training run per environment. These are toy
  environments at toy scale, meant to show the mechanism, not to compare with published numbers.
* The GIFs (`assets/world-two-room.gif`, `assets/world-push-block.gif`) show the **first
  CEM-solved evaluation episode** for each environment, chosen on purpose to illustrate a
  success. The success rates above are the representative numbers.

## 6. In the browser

`world/export.py` writes each model as plain JSON (`format: jepa-studio-world/v1`: layer
weights rounded to 6 significant digits, the retrieval bank, the planner defaults and the
metrics). The web app's World model tab loads `site/models/world-<env>.json` and runs the
encoder, predictor and CEM in JavaScript: no ONNX, no WebGL. The weights are rounded *before*
evaluation, so the metrics above, `world.safetensors` and the JSON describe the same model, and
`tests/test_world_parity.py` checks the JavaScript encoder, predictor and environments against
PyTorch. The web tier can plan with the shipped models; training a new world model needs the
command line or the desktop app.
