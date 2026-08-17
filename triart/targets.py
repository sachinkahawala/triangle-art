"""Target image loading and deterministic synthetic targets.

A target is a (work, work) float64 luminance field in [0, 1] (1 = white,
0 = black) — the same space as the canvas. Real images: grayscale →
center-crop → resize → gentle percentile contrast stretch. Synthetic targets
(fixed seed) let the whole pipeline run before the user drops in photos.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tiff"}


def _finalize(lum: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(lum, [1.0, 99.0])
    if hi - lo < 1e-6:
        hi = lo + 1e-6
    return np.clip((lum - lo) / (hi - lo), 0.0, 1.0).astype(np.float64)


def load_image_target(path: str | Path, size: int) -> np.ndarray:
    img = Image.open(path)
    img = ImageOps.exif_transpose(img).convert("L")
    w, h = img.size
    s = min(w, h)
    img = img.crop(((w - s) // 2, (h - s) // 2, (w + s) // 2, (h + s) // 2))
    img = img.resize((size, size), Image.LANCZOS)
    return _finalize(np.asarray(img, dtype=np.float64) / 255.0)


def synthetic_target(kind: str, size: int) -> np.ndarray:
    """Deterministic synthetic targets: 'blobs', 'rings', 'wedge', 'grid'.

    Built as darkness fields, returned as luminance (white background)."""
    c = (size - 1) / 2.0
    yy, xx = np.mgrid[0:size, 0:size].astype(np.float64)
    dy, dx = yy - c, xx - c
    rr = np.sqrt(dy * dy + dx * dx) / c
    if kind == "blobs":
        rng = np.random.default_rng(7)
        img = np.zeros((size, size))
        for _ in range(6):
            py, px = rng.uniform(0.2, 0.8, 2) * size
            s = rng.uniform(0.06, 0.16) * size
            a = rng.uniform(0.5, 1.0)
            img += a * np.exp(-((yy - py) ** 2 + (xx - px) ** 2) / (2 * s * s))
    elif kind == "rings":
        img = (0.5 + 0.5 * np.sin(rr * 6.0 * np.pi)) * np.exp(-rr * 1.2)
    elif kind == "wedge":
        theta = np.arctan2(dy, dx)
        img = (np.cos(theta * 3.0) * 0.5 + 0.5) * np.clip(1.0 - rr * 0.6, 0, 1)
    elif kind == "grid":
        img = (np.sin(yy * 2 * np.pi / (size / 5))
               * np.sin(xx * 2 * np.pi / (size / 5)) * 0.5 + 0.5)
        img *= np.clip(1.0 - rr * 0.4, 0, 1)
    else:
        raise ValueError(f"unknown synthetic target kind: {kind}")
    return _finalize(1.0 - np.clip(img, 0.0, 1.0))


@dataclass
class TargetSet:
    train: list[tuple[str, np.ndarray]]    # (name, luminance); train[0] = hero
    holdout: list[tuple[str, np.ndarray]]

    @property
    def hero(self) -> tuple[str, np.ndarray]:
        return self.train[0]


def _images_in(directory: Path) -> list[Path]:
    if not directory.is_dir():
        return []
    return sorted(p for p in directory.iterdir() if p.suffix.lower() in IMAGE_EXTS)


def load_target_set(assets_dir: str | Path, size: int) -> TargetSet:
    """Load 3 fitness targets from assets/train and 2 held-out targets from
    assets/holdout; pad with synthetics when short.

    A file named hero.* in assets/train becomes the hero (filmstrip) image;
    otherwise the alphabetically-first image does.
    """
    assets_dir = Path(assets_dir)
    train = [(p.stem, load_image_target(p, size))
             for p in _images_in(assets_dir / "train")]
    train.sort(key=lambda t: (not t[0].lower().startswith("hero"), t[0]))
    holdout = [(p.stem, load_image_target(p, size))
               for p in _images_in(assets_dir / "holdout")]

    synth_kinds = ["blobs", "rings", "wedge"]
    while len(train) < 3:
        kind = synth_kinds[len(train) % len(synth_kinds)]
        train.append((f"synth_{kind}", synthetic_target(kind, size)))
    hold_kinds = ["grid", "rings"]
    while len(holdout) < 2:
        kind = hold_kinds[len(holdout) % len(hold_kinds)]
        holdout.append((f"synth_{kind}", synthetic_target(kind, size)))
    return TargetSet(train=train[:3], holdout=holdout[:2])
