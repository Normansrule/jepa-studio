# Your data: loading, views and limits

jepa-studio pretrains on **unlabeled** folders of images, short video clips or time series.
Labels are optional and only used by the Evaluate tab. Code: `jepa_studio/data/loaders.py`,
`views.py`, `synthetic.py`; config keys live in the `data` block (`schema/config.schema.json`).

```
jepa-studio train --kind images     --data ~/photos   --name photos
jepa-studio train --kind video      --data ~/clips    --name clips  --config my-video.json
jepa-studio train --kind timeseries --data ~/sensors  --name sensors --config my-series.json
```

The command line has no `--arch` flag: pick the encoder in a config file. A minimal `my-video.json`
(everything else comes from the defaults in `config.py`):

```json
{"model": {"arch": "video-convnet"}, "data": {"clip_frames": 8}}
```

and a minimal `my-series.json` for files with 3 numeric columns:

```json
{"model": {"arch": "series-conv"}, "data": {"channels": 3, "series_window": 128}}
```

## 1. Data kinds

| `data.kind` | Files | One item becomes | Encoder to pick (`model.arch`) |
|---|---|---|---|
| `synthetic-shapes` | none: generated | a 32 × 32 RGB image of one of 10 shapes | `convnet-tiny` (default), `convnet-small`, `vit-tiny`, `mlp-tiny` |
| `images` | `.png .jpg .jpeg .webp .bmp .gif` | an RGB image | same as above |
| `video` | `.mp4 .webm .mov .gif .avi .mkv` | `clip_frames` evenly spaced frames (default 8) | `video-convnet` |
| `timeseries` | `.csv .tsv .npy` | a window of `series_window` samples (default 128) × channels | `series-conv` |

**Labels.** For `images` and `video`, the name of an item's immediate subfolder is its class
(`photos/cats/1.jpg` → `cats`); files directly in the chosen folder get the class `_`. Only
Evaluate uses them. With a single class the probe and k-NN accuracies are meaningless (every
guess is right), so organise a small labeled subset into subfolders if you want those numbers.
Time series have no labels.

### Images

Each file is opened with Pillow, JPEGs are decoded at reduced size (`draft`), converted to RGB and
shrunk (keeping aspect ratio) so the longest side is at most max(2 × `image_size`, 64) px.

### Video

Frames are decoded with imageio: Pillow for `.gif`, FFmpeg for every other format
(`pip install -e ".[video]"` adds `imageio-ffmpeg`, which is required for them). Before any
frame is decoded, the clip's size and duration are read from its header: clips longer than
60 s or with frames larger than 4096 × 4096 pixels are refused, and so are files whose header
cannot be read (they are skipped and listed with the reason). A GIF's duration is estimated as
frame count × the first frame's delay. Then up to 512 frames are decoded (each re-checked
against the size cap); `clip_frames` evenly spaced frames are kept, each shrunk like an image. Every view of a clip uses the **same crop for all frames**. `.gif` counts
as an image for `images` and as a clip for `video`.

### Time series

* **CSV / TSV:** a header row is allowed; every cell is parsed as a number (non-numbers become
  NaN). Columns that are more than 90% numeric below the first row are kept, then any row with a
  non-numeric value is dropped. Each remaining column is one channel.
* **NPY:** a 1-D array is one channel; a 2-D array is read as (time, channels).
* Each file is standardised per channel (zero mean, unit variance over the whole file), cut into
  windows of `series_window` samples with 50% overlap, and all files are truncated to the smallest
  channel count among them.
* Set `data.channels` to the number of channels your files have (the Data tab and
  `POST /api/data/preview` report it as `info.channels`); the encoder is built with that many
  input channels.

### Synthetic Shapes

Ten classes (circle, square, triangle, cross, ring, diamond, flower, stripes, checker, crescent)
with random colours, position, size, rotation and noise, drawn per pixel from a mulberry32 seed.
Image *i* with seed *s* is the same in Python and in the browser
(`tests/test_web_engine.py::test_shapes_match_python`). No download, fully reproducible.

## 2. Views (augmentations)

LeJEPA learns by making different views of the same item agree (and SIGReg keeps them spread
out). `ImageViews` produces `n_global` global and `n_local` local views per item:

| Step | Global views | Local views | Config key (default) |
|---|---|---|---|
| random crop, aspect ratio 3/4–4/3 | 30–100% of the area | 5–30% of the area | `global_scale` [0.3, 1.0], `local_scale` [0.05, 0.3] |
| resize (bilinear, antialiased) | `image_size` (32) | `local_size` (16) | |
| horizontal flip | with probability 0.5 | same | `flip_p` |
| colour jitter (applied with probability 0.8) | brightness and contrast × [1 − s, 1 + s], saturation × [1 − s/2, 1 + s/2] | same | `color_jitter` s = 0.4 |
| grayscale | probability 0.2 | same | `grayscale_p` |
| Gaussian blur, σ ∈ [0.1, 1 + size/64] | probability 0.2 | same | `blur_p` |
| block mask (4 px blocks filled with 0.5) | `mask_ratio` of the view (off) | same | `mask_ratio` 0.0 |
| how many | 2 | 4 | `n_global`, `n_local` |

`SeriesViews` for time series: a global view is a random 80–100% sub-window resampled to the full
window length; a local view is a 30–60% sub-window resampled to max(8, length ÷ 2). Both are
scaled by a random factor in [0.8, 1.2], get small Gaussian noise (standard deviation up to 0.03),
and optionally a zeroed span of `mask_ratio` of the length.

The Data tab shows an item next to its views (`assets/screens/tab-data.png`).

**Evaluation inputs are not augmented:** probes, k-NN and the embedding explorer use each item
resized to `image_size` (antialiased bilinear) with no crop or jitter (`build_eval_arrays`).

## 3. Limits and safety

A dataset folder is treated as untrusted input. What is enforced, and by which test, is in
[SECURITY_MODEL.md §2](SECURITY_MODEL.md#2-malicious-datasets); the numbers:

| Limit | Value | Where |
|---|---|---|
| items per dataset | `data.max_items` (default 4096) | `build_dataset` |
| files scanned for time series | 10 000 | `build_dataset` |
| image / series file size | 64 MB | `MAX_FILE_BYTES` |
| image size | 40 megapixels (Pillow's bomb check uses the same cap) | `MAX_PIXELS` |
| video file size | 512 MB | `MAX_VIDEO_BYTES` |
| clip duration (from the header, before decoding) | 60 s | `MAX_CLIP_SECONDS` |
| video frame size (header before decoding, then every decoded frame) | 4096 × 4096 pixels | `MAX_CLIP_PIXELS` |
| frames decoded per clip | 512 | `MAX_FRAMES` |
| numbers per series file | 50 million | `MAX_SERIES_VALUES` |
| `.npy` pickles | refused (`allow_pickle=False`) | `load_series` |
| symlinks leaving the folder | skipped | `safe_listing` |
| unreadable files | skipped and listed (first 50) in the run's `data` event | `build_dataset` |

Everything is decoded into memory before training starts, so `max_items` × item size has to fit
in RAM; there is no separate total-memory cap. In the desktop app, `data.path` must also be inside
a folder you granted through the folder picker.

**In the browser** (web demo), files are decoded in the page and never uploaded: up to 2000
images, 20 MB and 40 megapixels each; CSV up to 20 MB, 200 000 rows and 64 columns. The web demo
previews time-series views but trains on images only; see [web-demo.md](web-demo.md).
