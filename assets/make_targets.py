"""Generate the procedurally-drawn second held-out target.

    uv run python assets/make_targets.py

Writes holdout/spiral.png — procedurally drawn → guaranteed novel pixels
(nothing the model could have memorized). Replace with your own photo whenever
you like.
"""
from __future__ import annotations

import math
from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

S = 1024
HERE = Path(__file__).resolve().parent


def spiral() -> Image.Image:
    img = Image.new("L", (S, S), 236)
    d = ImageDraw.Draw(img)
    turns, a = 3.6, 118
    for t in range(0, int(turns * 360), 2):
        th = math.radians(t)
        rr = a * th / (2 * math.pi)
        x = 512 + rr * math.cos(th)
        y = 512 + rr * math.sin(th)
        w = 30 - 14 * t / (turns * 360)
        d.ellipse((x - w, y - w, x + w, y + w), fill=52)
    return img.filter(ImageFilter.GaussianBlur(3))


def main() -> None:
    (HERE / "holdout").mkdir(exist_ok=True)
    spiral().save(HERE / "holdout" / "spiral.png")
    print("wrote holdout/spiral.png")


if __name__ == "__main__":
    main()
