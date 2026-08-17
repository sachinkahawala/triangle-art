"""Fixed evaluation harness: runs an evolved propose_triangle function for the
triangle budget and maintains the authoritative canvas.

Anti-cheat: the heuristic returns a (delta_sse, pts, g, mask) tuple, but the
harness trusts only pts and g — it re-rasterizes the triangle, recomputes the
true delta itself, and paints only if that true delta is negative. Scores come
from pixels, not claims.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from .config import TaskCfg
from .raster import SlotEvaluator, paint_delta, triangle_mask


class HeuristicError(Exception):
    """The evolved function broke a rule (bad return value, wrong type...)."""


class HeuristicTimeout(Exception):
    """The evolved function was too slow to finish within its deadline."""


def snapshot_slots(n_tri: int) -> tuple[int, ...]:
    """Canvas snapshot points: n/8, n/4, n/2, n — {10, 20, 40, 80} at n=80."""
    return tuple(sorted({max(1, n_tri // 8), max(1, n_tri // 4),
                         max(1, n_tri // 2), n_tri}))


@dataclass
class RunResult:
    canvas: np.ndarray
    placed: int
    skipped: int
    evals_used: int
    snapshots: dict[int, np.ndarray] = field(default_factory=dict)


def _unpack_result(result) -> tuple[np.ndarray, float]:
    """Validate the heuristic's return value; extract (pts, g)."""
    if not isinstance(result, (tuple, list)) or len(result) != 4:
        raise HeuristicError(
            f"propose_triangle must return the 4-tuple from evaluate() or None, "
            f"got {type(result).__name__}")
    _, pts, g, _ = result
    try:
        pts = np.asarray(pts, dtype=np.float64)
        g = float(g)
    except (TypeError, ValueError) as e:
        raise HeuristicError(f"unusable pts/g in returned tuple: {e}") from e
    if pts.shape != (3, 2) or not np.all(np.isfinite(pts)):
        raise HeuristicError(f"returned pts must be a finite (3, 2) array")
    if not np.isfinite(g) or not 0.0 <= g <= 1.0:
        raise HeuristicError(f"returned g={g!r} must be a float in [0, 1]")
    return pts, g


def run_heuristic(
    fn,
    target: np.ndarray,
    task: TaskCfg,
    n_tri: int | None = None,
    deadline: float | None = None,
    want_snapshots: bool = False,
) -> RunResult:
    """Run fn for n_tri triangle slots on a fresh all-white canvas."""
    size = task.work
    n = n_tri if n_tri is not None else task.n_tri
    alpha = task.alpha
    snaps_at = set(snapshot_slots(n)) if want_snapshots else set()

    canvas = np.ones((size, size), dtype=np.float64)
    target_ro = target.view()
    target_ro.setflags(write=False)
    rng = np.random.default_rng(task.seed)

    placed = skipped = evals_used = 0
    snapshots: dict[int, np.ndarray] = {}

    for t in range(n):
        if deadline is not None and time.monotonic() > deadline:
            raise HeuristicTimeout(
                f"exceeded time limit after {t}/{n} triangle slots")
        evaluator = SlotEvaluator(target, canvas, alpha, task.k_evals)
        canvas_copy = canvas.copy()
        canvas_copy.setflags(write=False)
        result = fn(target_ro, canvas_copy, t, n, evaluator, rng)
        evals_used += evaluator.calls

        if result is None:
            skipped += 1
        else:
            pts, g = _unpack_result(result)
            mask = triangle_mask(pts, size)
            if mask.any() and paint_delta(target, canvas, mask, g, alpha) < 0:
                canvas[mask] = canvas[mask] * (1.0 - alpha) + g * alpha
                placed += 1
            else:
                skipped += 1

        if (t + 1) in snaps_at:
            snapshots[t + 1] = canvas.copy()

    return RunResult(canvas=canvas, placed=placed, skipped=skipped,
                     evals_used=evals_used, snapshots=snapshots)
