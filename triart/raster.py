"""Triangle rasterization and the budget-limited per-slot evaluator.

All pixel semantics are luminance: 0 = black, 1 = white. The canvas starts
all-white and each painted triangle composites a constant gray g at fixed
opacity alpha:  canvas[mask] = canvas[mask] * (1 - alpha) + g * alpha.
"""
from __future__ import annotations

import numpy as np
from PIL import Image, ImageDraw


class BudgetExceeded(Exception):
    """evaluate() was called more than K times within one triangle slot."""


def triangle_mask(pts: np.ndarray, size: int) -> np.ndarray:
    """Boolean (size, size) mask of the filled triangle. pts is (3, 2) of
    (x, y) vertex coordinates; out-of-canvas parts are clipped away."""
    img = Image.new("L", (size, size), 0)
    ImageDraw.Draw(img).polygon(
        [(float(x), float(y)) for x, y in pts], fill=255)
    return np.asarray(img, dtype=bool)


def optimal_gray(target: np.ndarray, canvas: np.ndarray, mask: np.ndarray,
                 alpha: float) -> float:
    """Least-squares optimal g for painting `mask` now, clipped to [0, 1].

    Minimizes sum((c*(1-a) + g*a - t)^2) over the mask:
        g* = mean(t - c*(1-a)) / a
    """
    t = target[mask]
    base = canvas[mask] * (1.0 - alpha)
    g = float(np.mean(t - base)) / alpha
    return float(np.clip(g, 0.0, 1.0))


def paint_delta(target: np.ndarray, canvas: np.ndarray, mask: np.ndarray,
                g: float, alpha: float) -> float:
    """Change in total SSE if the triangle were painted now (negative = better)."""
    t = target[mask]
    c = canvas[mask]
    new = c * (1.0 - alpha) + g * alpha
    old_err = c - t
    new_err = new - t
    return float(new_err @ new_err - old_err @ old_err)


class SlotEvaluator:
    """The `evaluate` callable handed to the heuristic for one triangle slot.

    Counts every call (including degenerate proposals) and raises
    BudgetExceeded on call K+1. Holds a reference to the harness's live canvas;
    the canvas does not change during a slot, so all K evaluations see the same
    state.
    """

    def __init__(self, target: np.ndarray, canvas: np.ndarray,
                 alpha: float, k_max: int):
        self._target = target
        self._canvas = canvas
        self._alpha = alpha
        self._k_max = k_max
        self.calls = 0

    def __call__(self, pts, g: float | None = None):
        self.calls += 1
        if self.calls > self._k_max:
            raise BudgetExceeded(
                f"evaluate() called more than K={self._k_max} times in one slot")
        try:
            pts = np.asarray(pts, dtype=np.float64)
        except (TypeError, ValueError):
            return None
        if pts.shape != (3, 2) or not np.all(np.isfinite(pts)):
            return None
        if g is not None:
            try:
                g = float(g)
            except (TypeError, ValueError):
                return None
            if not np.isfinite(g) or not 0.0 <= g <= 1.0:
                return None
        mask = triangle_mask(pts, self._target.shape[0])
        if not mask.any():
            return None
        if g is None:
            g = optimal_gray(self._target, self._canvas, mask, self._alpha)
        delta = paint_delta(self._target, self._canvas, mask, g, self._alpha)
        return (delta, pts.copy(), g, mask)
