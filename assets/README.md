# Target images

- `train/` — the 3 fitness images the model is scored on every generation.
  A file named `hero.*` becomes the hero (filmstrip/progression) image.
- `holdout/` — 2 held-out images scored only on accepted generations, for the
  generalization curve. The model never sees them.

Any `.jpg/.png/...` works: images are converted to grayscale, center-cropped
square, resized to the working resolution, and contrast-stretched at load time.
Missing slots are padded with deterministic synthetic targets, so the pipeline
runs with an empty assets directory too.

`make_targets.py` regenerates the procedurally-drawn `holdout/spiral.png`.
