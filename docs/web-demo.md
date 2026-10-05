# The web demo tier

The website (`site/`) trains a real LeJEPA model inside the browser tab. It is a demo: small
enough to finish in a few minutes on a laptop CPU, faithful enough that the loss, the collapse
detector and the evaluation behave like the desktop app. For real models, use the desktop app.

## What trains in the browser

The encoder is `mlp-tiny`: each view is resampled to 16×16 RGB, standardized per channel, and
flattened to 768 numbers.

* backbone: Linear(768 → 256), GELU, Linear(256 → 64). The 64-d output is the embedding used
  by Inspect, Evaluate and Export.
* projector: Linear(64 → 64), GELU, Linear(64 → 16). The LeJEPA loss is applied to this 16-d output.

GELU is the exact erf form, like PyTorch's default. Differences from the desktop app: the web
projector has two layers and no BatchNorm (the desktop projector has three layers with
BatchNorm), and SIGReg uses 64 random directions per step by default instead of 256.

## The loss, exactly

For a batch of N images with V views each (Vg of them global), and projections z:

* centres: the mean projection of the global views of each image.
* prediction term L_pred: the mean over views, images and dimensions of (centre − z)².
* SIGReg: for each view, draw M random unit directions, project the batch onto each, and
  compute the Epps–Pulley statistic of the N projected values against N(0, 1); average over
  directions and views.
* total: L = (1 − λ) L_pred + λ SIGReg, with λ = 0.05 by default.

The Epps–Pulley statistic compares the empirical characteristic function of the projected
values with exp(−t²/2) at 17 frequencies t in [0, 3], weighted by exp(−t²/2), trapezoid rule,
times N. Under a perfect Gaussian its expected value is about 1.05, so SIGReg values near 1
mean "Gaussian". A collapsed embedding gives values in the tens or hundreds.

The JavaScript loss matches `jepa_studio.losses.lejepa_loss` to 1e-4 on the same inputs and
directions (tests/test_web_engine.py), and its hand-written gradient is checked against finite
differences (tests/js/engine_gradcheck.mjs).

## Why SIGReg prevents collapse

The prediction term alone is minimized by mapping every input to the same vector: all views
then agree perfectly. SIGReg makes that impossible, because a batch of identical vectors
projects to a single spike, which is as far from N(0, 1) as a distribution can be. Requiring
every random 1-D projection to look like N(0, 1) forces the embeddings to spread out with equal
variance in every direction (an isotropic Gaussian), by the Cramér–Wold theorem.

## Reading the collapse detector

The Inspect tab computes the covariance of the projector outputs of up to 1024 images.

* effective rank: exp of the entropy of the normalized eigenvalues, from 1 (one direction) to 16.
* isotropy ratio: smallest eigenvalue divided by the largest.
* complete collapse: the mean standard deviation is below 0.001.
* dimensional collapse: effective rank below max(1.5, 10% of the dimension), or one direction
  holds more than 90% of the variance.
* anisotropic: isotropy ratio below 0.01.
* healthy: none of the above.

A short demo run usually reports "anisotropic" or "dimensional collapse" in its first few
dozen steps and moves toward healthy as SIGReg does its work. Raising λ speeds this up at the
cost of the prediction term.

## Batch size and the timing probe

With batch size "auto", the app times one training step at batch 32, 64, 128, 256 and keeps
the largest batch whose step stays under 150 ms. 32 is the floor: with fewer samples the
Epps–Pulley statistic of each direction is too noisy to be a useful signal.

## WebGPU

When the browser exposes WebGPU, the two largest products (the first layer's forward pass and
its weight gradient) run as compute shaders. The app checks the shaders against the JavaScript
result and times both paths at start-up; it only uses WebGPU when it is correct and faster.
The header badge says which backend is active: "WebGPU" or "CPU (JavaScript)". Software
adapters (for example SwiftShader in headless browsers) are usually slower than JavaScript and
are rejected.

## Linear probe and k-NN

Evaluate freezes the encoder and uses its 64-d embeddings of un-augmented images. 20% of the
images are held out; a labeled subset (10% of the rest by default) trains the classifiers.

* linear probe: softmax regression on standardized features, full-batch AdamW for 100 epochs.
* k-NN: cosine similarity, k = 20, majority vote; ties go to the class of the single most
  similar neighbour.
* random-init baseline: the same MLP with untrained weights.
* supervised baseline: the same MLP trained end-to-end with cross-entropy on the labeled subset.

## Your files

Images and CSV files are read with the File API and decoded in the page. Nothing is uploaded:
the page's Content Security Policy only allows requests to its own origin. Limits: 2000 images,
20 MB and 40 megapixels per image, 20 MB per CSV.

Three ways in, all through the same loader and the same limits:

* the **Choose images…**, **Choose a folder…** and **Choose CSV…** buttons (the keyboard route);
* **drag and drop** onto the drop zone at the top of the Data tab: image files, a whole folder
  (walked with `webkitGetAsEntry`, up to 8 levels deep), or a CSV file. Dropping class folders
  side by side (`cats/`, `dogs/`) labels the images like a folder of class folders; a CSV switches
  to Time series. A drop replaces the current image set;
* **paste** (Ctrl+V / ⌘V) with the drop zone focused, or anywhere on the Data tab outside a text
  field. Pasted images are *added* to the current set, so you can paste screenshots one at a time;
  the status line counts them until there are the 16 that training needs.

The drop zone is a focusable button: Enter or Space opens the matching file picker.

## Entering settings

* **Precise values.** Every slider has a number box with the same minimum, maximum and step,
  synced both ways. The number is the value used: typing 35 steps into a slider that moves in
  tens trains 35 steps (the slider shows the nearest position). Arrow keys step the box; Enter
  or leaving the box applies it.
* **Inline validation against the shared schema.** Each control knows its config key
  (`train.max_steps`, `objective.lambda`, `seed`…) and is checked with the same `validate()` and
  `config.schema.json` the desktop app uses, then against the range this tier offers (the web
  demo runs 20–2000 steps). A bad value gets a message under the field (`aria-invalid`,
  `aria-describedby`), is never applied or clamped, and Start stays disabled with the reason
  shown next to it ("Fix Steps and Seed (Data tab) to start.").
* **Reset to defaults** next to each section heading restores the values of
  `webDefaultConfig()` in `site/js/config.js`.
* **Presets** on the Train tab set steps, batch size and SIGReg directions and put λ and the
  learning rate back to their defaults. Changing any of those afterwards shows "Custom".

  | Preset | Web demo | Desktop app |
  |---|---|---|
  | Quick look | 100 steps, batch 32, 16 directions | 300 steps, batch 64, 64 directions |
  | Balanced (default) | 300 steps, auto batch, 64 directions | 2000 steps, auto batch, 256 directions |
  | Thorough | 1000 steps, batch 128, 256 directions | 10 000 steps, auto batch, 1024 directions |

* **Help.** The ⓘ button next to a label opens a short explanation with a link to the matching
  section of these docs. Esc closes it and returns focus to the button.

## The run config editor

The Export tab shows the current run config as formatted JSON. Edit it, then:

* **Validate** parses it, merges it onto the defaults and checks it against the schema, exactly
  like opening a file (and like `jepa_studio.config.load_config`). Problems are listed under the
  editor with their paths (`$.objective.lambda: 2 > maximum 1`); clicking one selects that key.
  A JSON syntax error says which line and column.
* **Apply** (or Ctrl+Enter) sends a valid config to the Data and Train tabs.
* **Revert** discards the edits; **Download** saves the applied config (disabled while there
  are unapplied edits, and it says so); **Open config…** loads a file into the editor and
  applies it when valid. If you have unapplied edits, it asks before replacing them.

The file format is the desktop app's, so a config saved here opens there unchanged.

## Guidance and preferences

* A progress strip under the tabs (Data → Train → Inspect → Evaluate → Export) marks each step
  done from what has actually happened: data chosen, a run finished, embeddings inspected, an
  evaluation run, something exported. A new run starts the later steps over.
* When a run ends, the Train tab offers the next step ("Inspect the embeddings"). Completions and
  errors also appear as short notices in the corner (announced to screen readers: errors as
  alerts, the rest politely); the panels keep their own text too.
* Inspect, Evaluate and Export say what to do first when there is no model yet, and disabled
  buttons say why they are disabled.
* The theme, the last tab and the last preset are remembered per browser in `localStorage`.
  When storage is unavailable (private windows, blocked site data) the app works the same and
  simply forgets them.
