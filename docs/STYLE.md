# Visual style

One visual language for the web app (`site/`), the desktop app (same front end) and every
diagram in `docs/`. The source of truth for the values is `site/css/app.css` (`:root` and its
dark-mode overrides); this file explains them. If you change a token, change it in both places.

## Data-flow colours

These five colours always mean the same stage of a JEPA, in charts, pipeline strips, diagram
boxes and equation underlines. Use them only for these meanings: never as a generic "series 3".

| Token | Light | Dark | Meaning |
|---|---|---|---|
| `--flow-view` | `#00897b` teal | `#129e8e` | views / data: crops, augmentations, input batches |
| `--flow-encoder` | `#b86e00` amber | `#c4850a` | encoder (backbone) and its embeddings |
| `--flow-predictor` | `#5e3fc0` violet | `#8a5ce0` | predictor / projector, world-model imagination, planner |
| `--flow-loss-pred` | `#3c8fe8` blue | `#4196f0` | prediction term L_pred |
| `--flow-loss-sigreg` | `#d42f5f` magenta-red | `#e84a66` | SIGReg term L_SIGReg and its histograms |

The two sets were checked with a colour-vision validator (OKLab distance, deutan / protan /
tritan simulation) as a five-colour categorical palette, all pairs:

* light on `#ffffff`: every colour passes lightness, chroma and 3:1 contrast; worst normal-vision
  distance 16.9 (floor 15); worst colour-blind distance 6.9 (magenta vs teal, deutan), which is
  in the 6-8 band where a secondary cue is required.
* dark on `#1b2437`: passes lightness, chroma, contrast and the normal-vision floor (15.3);
  worst colour-blind distance 7.6 (magenta vs amber, deutan), same rule.

So the secondary cue is mandatory: a flow colour is never the only carrier of meaning. Every
use has a text label next to it (the pipeline strip names each stage, chart titles name the
term, diagram boxes are labelled, equation terms are the symbols themselves).

## Surfaces and ink

| Token | Light | Dark | Use |
|---|---|---|---|
| `--bg` | `#f5f6f8` | `#121826` | page background |
| `--surface` | `#ffffff` | `#1b2437` | panels, chart plot areas |
| `--surface-2` | `#eef0f4` | `#232e45` | wells: scatter plot, image wells, code |
| `--ink` | `#172033` | `#e8ecf4` | primary text, the total-loss line |
| `--ink-2` | `#4a5468` | `#b4bccb` | secondary text, reference lines |
| `--ink-3` | `#636c7e` | `#97a1b4` | captions, axis labels, "random" baselines |
| `--line` | `#d9dde5` | `#2e3a54` | panel borders, gridlines |
| `--line-strong` | `#aeb6c4` | `#4a587a` | input borders, crosshair, arrows |
| `--focus` | `#2567c9` | `#7fb4ff` | focus ring, links |

Dark mode is its own set of steps (chosen and validated against the dark surface), not an
inversion. It follows `prefers-color-scheme` and the header toggle; the toggle wins both ways
(`:root[data-theme="light"|"dark"]`).

## Status colours

`--status-good` `#0b8a0b` / `#35c235`, `--status-warn` `#a86f00` / `#fab219`,
`--status-bad` `#c43333` / `#f06a6a`. Used only for the collapse detector, the world-model
success badge and the capability table, always with an icon and a word (Healthy, Anisotropic,
Dimensional collapse...). Never used for a data series.

## Class colours (embedding explorer)

Ten Shapes classes are drawn with five hues x two marker styles (filled dot for classes 0-4,
ring for 5-9), so identity never depends on hue alone and the legend isolates a class on click.

| Token | Light | Dark |
|---|---|---|
| `--cat-1` | `#2a78d6` | `#3987e5` |
| `--cat-2` | `#eb6834` | `#d95926` |
| `--cat-3` | `#1baf7a` | `#199e70` |
| `--cat-4` | `#4a3aa7` | `#9085e9` |
| `--cat-5` | `#e87ba4` | `#d55181` |

(Adjacent-pair colour-blind distance >= 9.2 in both modes. In light mode three hues sit below
3:1 on the plot well, so the labelled legend and the neighbour captions are the required relief.)

## Typography

* **Display and numbers**: Latin Modern Roman (`--serif`), the Computer Modern face of the
  papers this project implements. Self-hosted WOFF2 subsets in `site/fonts/` (GUST Font
  License, text in `site/fonts/GUST-FONT-LICENSE.txt`). Used for headings, metric values and
  the loss equation, which is set as real maths (italic variables, subscripts).
* **Interface text**: the system UI stack (`--sans`), for controls, captions and body copy.
* **Code and logs**: the system monospace stack (`--mono`).
* Scale (rem): 0.8125 / 0.9375 (body) / 1.125 / 1.5 / 2 / 3. Body line-height 1.5; headings 1.15.
* Sentence case everywhere. No all-caps labels, no letter-spaced eyebrows.
* Figures use tabular numerals (`font-variant-numeric: tabular-nums`) so live values don't jitter.

## The signature element

The loss equation `L = (1 − λ) L_pred + λ L_SIGReg` appears on the landing page and the Train
tab with each term underlined in its flow colour. Charts of those terms reuse the same colours,
so the equation doubles as the legend. Keep it the single bold element on a page.

## Chart rules

1. One y-axis per chart. Two measures of different scale get two charts (loss terms are
   separate small charts, not a dual axis).
2. Lines 2 px, round joins. Bars have 2-4 px rounded data ends anchored at the baseline, with a
   2 px gap between neighbours. Gridlines 1 px `--line`; axis labels 11 px `--ink-3`.
3. Text on charts uses ink tokens, never the series colour.
4. Every chart has a hover read-out (crosshair + tooltip on lines, per-bar tooltip on bars).
5. With 2+ series a legend is always present; reference lines (chance, N(0, 1), the Gaussian
   floor of SIGReg) are dashed or drawn in `--ink` and labelled in place.
6. Colours are read from CSS custom properties at draw time; theme changes redraw.
7. Animation (the hero histogram, the SIGReg snapshot player) respects
   `prefers-reduced-motion`: it jumps to the final state.

## Layout

* Controls in a left column (sticky on wide screens), visuals on the right; single column
  below 900 px; 16 px side gutter; no horizontal page scroll down to 320 px.
* Panels: 1 px `--line` border, 6 px radius, no shadows. Inputs and buttons: 4 px radius.
* Structure encodes information: the only numbered or stepped elements are the pipeline strip
  (views -> encoder -> projector -> losses), the progress stepper under the tabs (Data -> Train ->
  Inspect -> Evaluate -> Export) and the landing page's "How it works" row, because each is an
  actual sequence. Stepper numerals are set in the serif face; a finished step becomes a filled
  `--ink` disc with a check, the current step a 2 px `--ink` ring, later steps `--line-strong`.
* Spacing comes from one scale: `--sp-1` … `--sp-7` = 4, 8, 12, 16, 24, 32, 48 px.
* Hit targets: `--tap` is 32 px, and 44 px below 620 px wide or on coarse pointers. Buttons,
  tabs, selects, number boxes, segmented options, info buttons and the stepper use it.
* Every focusable element shows the 2 px `--focus` ring on `:focus-visible`.

## Inputs, help and notices

* A slider is always paired with a number box (`.num`, right-aligned tabular figures).
* An invalid field gets a `--status-bad` border and a message under it with an icon and the
  words ("Use a whole number from 0 to 4294967295."), never colour alone.
* A disabled button that the user might expect to work has a hint under it (`.why`: a small
  `--status-warn` dot plus the reason in `--ink-2`).
* Help disclosures are inline, under their label: `--surface-2` with a 2 px `--focus` rule.
* Notices (toasts) sit bottom right (full width on phones): `--surface`, 1 px `--line-strong`
  border, a 3 px left rule in the status colour and the matching icon. No shadows.
* The drop zone is a dashed `--line-strong` well on `--surface-2`; while a file is dragged over
  it turns solid `--focus` on `--selection`.

## Content Security Policy constraints on UI code

Every page ships `default-src 'self'; script-src 'self'; style-src 'self'; ...`. So: no inline
`<script>`, no `on*=` attributes, no `style="..."` in HTML strings, no `eval`/`new Function`.
Setting `element.style.x` from JavaScript is allowed. Untrusted text (file names, CSV headers,
logs, doc passages) is always inserted with `textContent`.
