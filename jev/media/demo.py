"""Build the README demo: a short narrated explainer, "What is a JEPA?", rendered entirely with
the local pipeline (slides -> local speech -> crossfades -> captions -> H.264).

    python -m jev.media.demo [--out assets/media] [--resolution 720p] [--tts auto]

Writes demo_explainer.mp4, demo_explainer.srt and demo_explainer_poster.png into --out.
"""
from __future__ import annotations

import argparse
import json
import shutil
import tempfile
from pathlib import Path

from .slides import render_slide
from .video import make_explainer

FOOTER = "JEV - made locally with jev.media"

SCENES = [
    {"narration": "What is a JEPA?",
     "visual": {"kind": "slide", "slide": {"type": "title", "kicker": "JEV explainer", "title": "What is a JEPA?",
                                           "subtitle": "Joint-Embedding Predictive Architecture"}}},
    {"narration": "An encoder embeds two views of the same data. A predictor maps one embedding onto the other.",
     "visual": {"kind": "slide", "slide": {
         "type": "diagram", "title": "Two views, one embedding space", "footer": FOOTER,
         "diagram": {"nodes": [
             {"id": "x1", "label": "View 1", "x": 0.1, "y": 0.25}, {"id": "x2", "label": "View 2", "x": 0.1, "y": 0.75},
             {"id": "e1", "label": "Encoder f", "x": 0.38, "y": 0.25}, {"id": "e2", "label": "Encoder f", "x": 0.38,
                                                                          "y": 0.75},
             {"id": "p", "label": "Predictor g", "x": 0.66, "y": 0.25},
             {"id": "l", "label": "match in embedding space", "x": 0.9, "y": 0.5, "w": 0.18, "h": 0.3,
              "shape": "ellipse"}],
             "edges": [{"from": "x1", "to": "e1"}, {"from": "x2", "to": "e2"}, {"from": "e1", "to": "p", "label": "z1"},
                       {"from": "p", "to": "l"}, {"from": "e2", "to": "l", "label": "z2", "dashed": True}]}}}},
    {"narration": "It predicts embeddings, not pixels, so it learns what matters.",
     "visual": {"kind": "slide", "slide": {
         "type": "bullets", "title": "Predict meaning, not pixels", "footer": FOOTER,
         "bullets": ["No pixel reconstruction", "No labels needed", "Features transfer to new tasks"]}}},
    {"narration": "LeJEPA adds SIGReg, which keeps the embeddings from collapsing.",
     "visual": {"kind": "slide", "slide": {
         "type": "equation", "title": "LeJEPA objective", "footer": FOOTER,
         "equations": [r"\mathcal{L} = (1-\lambda)\,\|g(z_1) - z_2\|^2 + \lambda\,\mathrm{SIGReg}(z)"],
         "caption": "SIGReg pushes embeddings toward an isotropic Gaussian"}}},
]


def main(argv: list[str] | None = None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--out", default="assets/media")
    ap.add_argument("--resolution", default="720p", choices=["720p", "1080p"])
    ap.add_argument("--tts", default="auto")
    ap.add_argument("--crf", type=int, default=27)
    args = ap.parse_args(argv)
    size = (1280, 720) if args.resolution == "720p" else (1920, 1080)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)  # wsfs-ok: demo CLI writes to the folder you name
    with tempfile.TemporaryDirectory() as tmp:
        ws = Path(tmp)
        r = make_explainer(SCENES, "demo_explainer.mp4", ws, size=size, fps=24, crossfade=0.4, crf=args.crf,
                           preset="slow", tts_backend=args.tts, rate=1.05)
        shutil.copy(r["video"], out / "demo_explainer.mp4")  # wsfs-ok: demo CLI output folder
        shutil.copy(r["srt"], out / "demo_explainer.srt")  # wsfs-ok: demo CLI output folder
        sb = json.loads(r["storyboard"].read_text())  # wsfs-ok: demo CLI output folder
    render_slide(SCENES[1]["visual"]["slide"], (960, 540)).save(out / "demo_explainer_poster.png", optimize=True)  # wsfs-ok: demo CLI output folder
    summary = {"duration_s": round(r["duration_s"], 2), "tts": r["tts_backends"], "captions": r["caption_mode"],
               "bytes": (out / "demo_explainer.mp4").stat().st_size, "size": list(size),
               "render_seconds": sb["render_seconds"]}
    print(json.dumps(summary))
    return summary


if __name__ == "__main__":
    main()
