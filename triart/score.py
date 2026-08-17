"""Scoring: plain RMSE over the full image. Lower is better.

Blank-canvas RMSE (all-white vs target) is the worst-case fitness assigned to
crashed/rule-breaking candidates and a useful per-image reference point.
"""
from __future__ import annotations

import numpy as np


def rmse(canvas: np.ndarray, target: np.ndarray) -> float:
    d = canvas - target
    return float(np.sqrt(np.mean(d * d)))


def blank_rmse(target: np.ndarray) -> float:
    return rmse(np.ones_like(target), target)
