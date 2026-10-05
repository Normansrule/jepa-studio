# Desktop backend API (localhost only)

The desktop app runs the same front end as the website. When it detects the desktop shell
(`window.__JEPA_DESKTOP__ = {port, token}` injected by the Tauri shell), the front end talks to
the local Python backend (`jepa-studio serve`) instead of the in-browser demo engine.

Security rules (docs/SECURITY_MODEL.md):
* binds `127.0.0.1` only, random port, never `0.0.0.0`
* every request must carry `X-JEPA-Token: <per-session random token>` (32 bytes, urlsafe base64)
* the `Host` header must be `127.0.0.1:<port>` or `localhost:<port>` (DNS-rebinding defense)
* an `Origin` header, when present, must be the desktop shell (`tauri://localhost`,
  `http(s)://tauri.localhost`) or the server's own origin; no wildcard CORS
* request bodies are JSON, capped at 2 MB; paths must lie inside folders the user picked
* no endpoint executes code, downloads anything, or reaches the network

All responses are JSON. Errors: `{"error": "message"}` with a 4xx/5xx status.

| Method | Path | Body / query | Returns |
|---|---|---|---|
| GET | `/api/health` | | `{ok, version}` |
| GET | `/api/hardware` | `?disk=1` | `{card, decisions}` (decisions for the default config) |
| POST | `/api/config/validate` | `{config}` | `{errors: [...]}` |
| POST | `/api/data/preview` | `{config, n}` | `{info, samples: [{original, globals:[png dataURL], locals:[...]}]}` |
| POST | `/api/runs` | `{config}` | `{run_id}` (training starts in a background thread) |
| GET | `/api/runs` | | `{runs: [{run_id, name, status, step, total}]}` |
| GET | `/api/runs/<id>/events` | `?since=<line>` | `{events: [...], next: <line>, status}` (up to 500 events.jsonl records) |
| POST | `/api/runs/<id>/control` | `{command: pause/resume/stop}` | `{ok}` |
| POST | `/api/runs/<id>/evaluate` | | evaluation dict (see `evaluate.evaluate_all`) |
| GET | `/api/runs/<id>/embeddings` | `?n=1000` | `{pca3, var, labels, classes, isotropy, collapse, thumbs}` |
| GET | `/api/runs/<id>/neighbors` | `?i=<index>&k=8` | `{query, neighbors: [[index, cosine]]}` |
| GET | `/api/runs/<id>/saliency` | `?i=<index>` | `{index, kind: "gradient", side, map: [[...]], image, attention?: {kind: "cls-attention", side, grid: [gh, gw], map: [[...]]}}` (image runs only; see below) |
| POST | `/api/benchmark` | `{config}` | `{stages: [{stage, samples_per_s, settings, batch}]}` |
| POST | `/api/world` | `{config}` | `{run_id}` world-model training |
| POST | `/api/world/<id>/plan` | `{seed}` | `{frames_real, frames_imagined, imagined_note, goal, costs, latent_dist, success, steps, final_distance}` (imagined frames are retrieved nearest real frames) |
| POST | `/api/runs/<id>/export` | | `{bundle: path, report: path}` |
| POST | `/api/assistant` | `{question, run_id?}` | `{answer, mode, sources: [{file, heading, score, trusted}]}` |

`/saliency` explains one sample `i` of the run's un-augmented evaluation inputs (the same
indices as `/embeddings` and `/neighbors`; out of range is a 400). `map` is `side`×`side`
(the input's pixel grid) and holds, per pixel, the largest over colour channels of
|∂‖emb‖₂ / ∂pixel|, where `emb` is the backbone embedding: the same quantity the web engine
draws, divided by its maximum so it lies in [0, 1]. `image` is that input as a PNG data URL, so
the overlay is drawn on exactly what was explained. For `vit-tiny` runs, `attention.map` is the
last block's attention from the [CLS] token to each patch, averaged over heads, on the
`grid` of patches and divided by its maximum. Runs on video or time series answer 400
(`jepa_studio/maps.py` has the definitions).

`events.jsonl` records have `event` in `data | start | step | paused | resumed | stopped | end | error`;
`stopped` and `end` may carry `reason` (for example `"memory guard: ..."` when the live memory
guard stopped the run); `end.status` is `done` or `stopped`;
`step` records carry `step, epoch, loss, pred, sigreg, lr, samples_per_s, cpu_pct, rss_gb`,
optional `gpu_mem_gb`, and every 5th log step `proj_hist` (3 histograms of 24 bins over [-4, 4]).
