# JEV media: explainers, your footage, images, generative video

Everything here runs on your machine. FFmpeg does the encoding, speech comes from a local engine,
slides and diagrams are drawn with Pillow and matplotlib, and the optional diffusion models load
from a folder you chose. Nothing is uploaded, and nothing is downloaded unless you approve it.

Code: `jev/media/` (library) and `jev/tools/media.py` (the tools the agent calls).
Demo: [`assets/media/demo_explainer.mp4`](../../assets/media/demo_explainer.mp4), built by
`python -m jev.media.demo`.

## What each feature does

| Feature | Module | What you get |
|---|---|---|
| Narrated explainer | `video.py` | scenes `{narration, visual, min_seconds}` → speech per scene → scene length = max(min_seconds, speech + padding) → one segment per scene (slide, still with optional Ken Burns pan and zoom, or a clip) → crossfades → narration track (plus optional background music you provide, ducked under the voice) → captions → H.264 MP4 (yuv420p, `+faststart`) + `.srt` + `.storyboard.json` |
| Slides | `slides.py` | 16:9 PNG slides (1920×1080, or the 1280×720 preview) and one PDF: title, bullets, code (syntax-highlighted when `pygments` is installed), image (fit or cover), equation (matplotlib mathtext, no TeX install), chart, diagram. Dark and light themes; all sizes scale with the frame height |
| Speech | `tts.py` | WAV + which engine spoke + measured duration (see the fallback chain below) |
| Footage: shots | `footage.py` | sampled frames → features → cuts where the frame-to-frame cosine distance peaks above an adaptive threshold (rolling median + k·MAD, with a floor) |
| Footage: search | `footage.py` | "find shots like this image" or "like this moment": cosine similarity per sampled frame, best frame per shot, ranked |
| Footage: highlights | `footage.py` | a reel inside a time budget: farthest-point sampling over clip embeddings (diversity) weighted by motion energy, joined in time order |
| Footage: cuts | `footage.py` | cut list → FFmpeg trims → concat, optional title card and per-clip captions; the cut list is also written as JSON and as a CMX3600-style EDL that DaVinci Resolve or Premiere can import |
| Images | `images.py` | charts (line, bar, scatter), box-and-arrow diagrams from JSON, title cards, contact sheets, resize, crop and annotate |
| Generative | `generative.py` | text-to-image and text-to-video through `diffusers`, with VRAM checks, safetensors only and fixed seeds |

**Features for footage.** By default each sampled frame is described by a colour histogram
(a global 8×8×8 RGB histogram plus a 2×2 grid of 4×4×4 histograms, square-rooted and
L2-normalised). If you pass `encoder_run` (the name of a jepa_studio run, e.g. one you trained on
your own clips), JEV loads that encoder with `load_run_weights`, so the weights are checked against
the hash the run recorded. It then uses the L2-normalised embeddings instead. A trained encoder can
match shots that look alike in content but differ in colour. The histogram cannot.

**Captions.** An `.srt` is always written, with captions timed to each scene's speech in proportion
to the text length. They are burned in with FFmpeg's `subtitles` filter when FFmpeg was built with
libass (the Ubuntu package and the imageio-ffmpeg binary both are). Without libass, JEV draws each
caption with Pillow and overlays it at its time. While captions are burned in, slides keep a band
at the bottom free so the captions never cover slide content.

## Tools and approvals

| Tool | Risk | Why |
|---|---|---|
| `probe_media` | read | reads container metadata only |
| `make_slides`, `make_image`, `text_to_speech` | sandbox | write only inside the conversation workspace |
| `make_explainer_video` | sandbox, **asks** if the render estimate is over 10 minutes | long CPU jobs should not start silently. The estimate is deliberately pessimistic (frame count × pixels × measured per-frame costs) |
| `find_scenes`, `search_footage`, `make_highlight_reel`, `cut_video` | sandbox | read your footage from the workspace and write results and a feature cache (`.jev_media_cache/*.npz`, no pickle) there |
| `generate_image`, `generate_video` | **approve** | they load multi-GB models and run long GPU jobs |
| `download_voice`, `download_generative_model` | **approve** | network downloads into folders outside the workspace |

Every path the model passes is workspace-relative and goes through `ctx.path()`. Every output is
registered with `ctx.artifact()`.

## Local dependencies

| | Ubuntu / Debian | macOS | Windows |
|---|---|---|---|
| FFmpeg (required) | `sudo apt install ffmpeg` | `brew install ffmpeg` | `winget install Gyan.FFmpeg` |
| …or the bundled copy | `pip install imageio-ffmpeg` (the Linux build has libx264 and libass, which was checked. There is no ffprobe, so JEV reads metadata from `ffmpeg -i` instead) | same (without libass, captions are drawn with Pillow) | same |
| espeak-ng (fallback voice) | `sudo apt install espeak-ng` | `brew install espeak-ng` | the `.msi` from github.com/espeak-ng/espeak-ng/releases |
| Built-in voice | none | `say` (always present) | SAPI through Windows PowerShell 5.1 (always present) |
| Piper (neural voice, best quality) | `pip install piper-tts` + a voice | same | same |
| Fonts | DejaVu or Liberation if installed, else the DejaVu copies inside matplotlib | Arial or Helvetica or Menlo, else matplotlib's DejaVu | Segoe UI / Arial / Consolas, else matplotlib's DejaVu |
| Code highlighting (optional) | `pip install pygments` | same | same |
| Charts and equations | `pip install matplotlib` | same | same |
| Generative models (optional) | `pip install 'jepa-studio[gen]'` + an NVIDIA GPU | Apple Silicon (MPS) works for images, slowly | NVIDIA GPU |

If no font file is found at all, Pillow's built-in font is used. It is readable but plain.

### Speech fallback chain

`piper` (CLI or the `piper-tts` package, **and** a voice in the voices folder) → `espeak-ng` →
`espeak` → `say` (macOS) → `sapi` (Windows) → **silent**. In the silent case JEV writes a silent
track of the estimated speaking time (about 155 words per minute). Captions then carry the narration,
and the tool result says so in plain words. Text reaches every engine through a file or stdin, never
on the command line.

### Piper voices: where to get them, and checksums

Voices are published at `https://huggingface.co/rhasspy/piper-voices` (you can listen to samples at
`https://rhasspy.github.io/piper-samples/`). Each voice is two files, `<name>.onnx` and
`<name>.onnx.json`, e.g. `en_US-lessac-medium`.

* **With JEV (asks first):** `download_voice` fetches both files from the pinned revision `v1.0.0`
  over HTTPS (only `huggingface.co` and its CDN subdomains are allowed). It checks each file's size
  and MD5 against `voices.json` from the same revision and only saves files that match. It then
  records each file's SHA-256 in `voices-manifest.json`, which is checked every time the voice is
  used, so a corrupted or swapped file is refused.
* **By hand:** put both files in the voices folder (`~/.jev/voices`; change it with
  `media.voices_dir` or `JEV_VOICES_DIR`). To pin them, add
  `{"<name>.onnx": {"sha256": "<sha256sum output>"}}` to `voices-manifest.json`.
* Recommended: `en_US-lessac-medium` (clear, neutral), `en_US-amy-medium`, `en_US-ryan-high`,
  `en_GB-alan-medium`.

Licence note: `piper-tts` 1.3 and later (the OHF-Voice/piper1-gpl project) is GPL-3.0. JEV only
calls it as an optional external program or package. Each voice has its own licence on its model
card.

## Generative models

Install `pip install 'jepa-studio[gen]'` (diffusers, transformers, accelerate, sentencepiece, ftfy,
huggingface_hub). The adapters are written against the documented pipeline APIs and checked against
diffusers 0.42: `from_pretrained(..., torch_dtype=, use_safetensors=True, local_files_only=True)`,
then `pipe(prompt=..., generator=..., num_inference_steps=..., guidance_scale=...)` returning
`.images` or `.frames[0]`. `LTXPipeline` needs diffusers ≥ 0.32 and `WanPipeline` needs ≥ 0.33.

| Model | Kind | Download | VRAM on GPU | VRAM with CPU offload | Defaults | Licence |
|---|---|---|---|---|---|---|
| `sdxl-turbo` | image | ~7 GB (fp16) | ~8 GB | ~5 GB; **also runs on CPU** (about a minute per 512² image) | 1 step, guidance 0, 512×512 | Stability AI Community Licence |
| `flux-schnell` | image | ~34 GB | ~33 GB | ~24 GB | 4 steps, guidance 0, 1024×768 | Apache-2.0 |
| `ltx-video` | video | ~28 GB | ~18 GB | ~10 GB | 97 frames (8k+1) at 24 fps, 704×480, 40 steps | LTX-Video open weights licence |
| `wan2.1-t2v-1.3b` | video | ~30 GB | ~14 GB | ~8.2 GB | 81 frames (4k+1) at 16 fps, 832×480, 30 steps | Apache-2.0 |

VRAM figures are approximate peaks at fp16/bf16 and depend on resolution, frame count and diffusers
version. JEV picks "fully on GPU" when the card has the first figure and model CPU offload when it
has the second. Otherwise it refuses and names the model and both figures: "needs ~N GB of GPU
memory (~M GB with CPU offload); this GPU has X GB".

Rules:

* **Weights only from folders you chose:** `<models_dir>/<name>` (`media.models_dir`,
  `JEV_MODELS_DIR`, default `~/.jev/models`), or a name mapped to any folder with
  `media.model_dirs`. The model can only pass a catalog or folder name, never a path.
* **Safetensors only.** A pickle checkpoint (`.bin/.ckpt/.pt/.pth/.pkl/...`, the same list as
  `jepa_studio.weights`) is refused, and so is a component that has only pickle weights. Stray
  pickles next to safetensors are reported and never read (`use_safetensors=True`). A
  `model_index.json` whose pipeline class is not on the allowlist (SDXL/SD/SD3, FLUX, LTX, Wan) is
  refused, so custom pipeline code is never run.
* **Offline loading:** `HF_HUB_OFFLINE=1` and `local_files_only=True`.
* **Downloads (approval):** `download_generative_model` uses `huggingface_hub.snapshot_download`
  with allow-patterns for JSON, text, tokenizer files and `*.safetensors` (the fp16 variant for
  SDXL-Turbo), and ignore-patterns for every pickle and other format. It deletes any pickle that
  arrives anyway, then records each file's SHA-256 in `jev-model.json`. Those hashes are verified
  before each load.
* **No text-to-video on CPU.** A few seconds of video means dozens of denoising steps through a
  multi-billion-parameter transformer over every frame. On a CPU that takes many hours, so JEV
  refuses with that explanation. Image generation on CPU is allowed only for `sdxl-turbo`.
* **Deterministic:** a `torch.Generator` seeded with `seed`. The same seed, model, versions and
  hardware give the same output. Every output gets a `.json` sidecar (and PNG text metadata) with
  the prompt, seed, steps, model, device and licence.

## Settings

| Key | Default | Meaning |
|---|---|---|
| `media.voices_dir` | `~/.jev/voices` | Piper voices (or `JEV_VOICES_DIR`, `JEV_HOME`) |
| `media.models_dir` | `~/.jev/models` | diffusers model folders (or `JEV_MODELS_DIR`) |
| `media.model_dirs` | `{}` | `{name: folder}` for models you already have elsewhere |
| `media.runs_dir` / `runs_dir` | `runs` | where `encoder_run` names are looked up |
| `media.limits.*` | see below | caps |

## Limits (checked from metadata before any frame is decoded)

| Cap | Default | Note |
|---|---|---|
| input file size | 512 MB | same as the training loaders |
| frame size | 4096×4096 | same as the training loaders |
| clip inside an explainer | 60 s | same as the training loaders |
| footage you edit | 30 min | longer than the 60 s training cap on purpose |
| frames sampled for analysis | 3000 | long footage is sampled more sparsely (shot times then have one sample period of slack) |
| still images | 64 MB, 40 MP | same as the training loaders |
| output length / size | 30 min, 4K | |
| one FFmpeg call | 1 h | timeout, then terminate and kill |

A file whose duration cannot be read is refused, because the duration cap cannot be checked.

## How FFmpeg is run

FFmpeg is found on PATH, else the imageio-ffmpeg binary is used. It always runs from an argument
list, never a shell, with `-nostdin`. Every file input must resolve inside the workspace and is
passed as `file:<absolute path>` with `-protocol_whitelist file`. Inputs that look like URLs or
protocols (`http://`, `concat:`, `subfile:`) are refused. Filter graphs that open files or load
plugins (`movie=`, `amovie=`, `textfile=`, `file=`, `ladspa`, `sendcmd`, `lut3d`, ...) are refused,
and a `subtitles=` file must be inside the workspace. Progress comes from `-progress pipe:1`, and
cancelling or timing out terminates the process and removes the partial output.

## Honest limits

* espeak-ng sounds robotic. It is clear but not natural. Install Piper and a voice for natural
  narration.
* Shot detection finds hard cuts reliably. Slow dissolves and fades may be missed or placed late.
  Search takes an image or a moment of the video, not a text query, because no local text-image
  model ships here.
* With stream copy (lossless, fast), the start of each cut is keyframe-exact and the end is cut by
  frame count. With sources that use B-frames, the last frame or two can differ from a re-encode.
  `mode=reencode` is frame-accurate. `auto` copies only when every cut starts on a keyframe and
  nothing is drawn on top.
* A clip's own audio is not used inside explainers (the narration track is). Cut lists keep the
  source audio.
* The render estimate is a rough CPU figure. Generative model timings vary a lot between GPUs.
* The generative adapters are tested here with an injected fake pipeline, because no weights can be
  downloaded on the CI machine. The real diffusers path follows the documented APIs, but it has not
  been run end to end on a GPU in this repo's tests.

## Formats

* **Storyboard** (`*.storyboard.json`, `jev-storyboard/v1`): size, fps, total duration, crossfade,
  caption mode, the speech engine(s) used, music, render time, and per scene the start, duration,
  narration, visual, speech start and length, and caption cues.
* **Cut list** (`*.cuts.json`, `jev-cutlist/v1`): source, fps, cuts `{start, end, label, reason,
  score, record_start, record_end}`. The **EDL** (`*.edl`) holds the same cuts in CMX3600 form,
  with the record timecode starting at 01:00:00:00.

## Demo asset

`assets/media/demo_explainer.mp4`: "What is a JEPA?" in 4 scenes (title, diagram, bullets,
equation), 1280×720 at 24 fps, about 18.8 s and about 0.5 MB, H.264 + AAC, narrated with espeak-ng
and with captions burned in through libass. `demo_explainer.srt` holds the captions and
`demo_explainer_poster.png` is a still of the diagram slide. Rebuild it with
`python -m jev.media.demo` (`--tts piper` once a voice is installed).
