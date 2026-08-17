"""Invariants of the frozen harness: compositing math, optimal gray, budget
enforcement, anti-cheat re-verification, determinism, snapshots."""
import numpy as np
import pytest

from triart.config import TaskCfg
from triart.harness import (HeuristicError, RunResult, run_heuristic,
                            snapshot_slots)
from triart.raster import (BudgetExceeded, SlotEvaluator, optimal_gray,
                           paint_delta, triangle_mask)
from triart.score import blank_rmse, rmse
from triart.targets import synthetic_target

TASK = TaskCfg(work=64, n_tri=16, k_evals=10, alpha=0.65, seed=0)
BIG = np.array([[2.0, 2.0], [60.0, 2.0], [30.0, 60.0]])


def target():
    return synthetic_target("blobs", TASK.work)


def test_composite_math():
    tgt = np.zeros((64, 64))  # all black target
    canvas = np.ones((64, 64))
    ev = SlotEvaluator(tgt, canvas, TASK.alpha, TASK.k_evals)
    delta, pts, g, mask = ev(BIG, 0.2)
    assert mask.any() and not mask.all()
    # painting by hand must reproduce the evaluator's delta exactly
    new = canvas.copy()
    new[mask] = new[mask] * (1 - TASK.alpha) + g * TASK.alpha
    assert np.allclose(new[mask], 1 * (1 - 0.65) + 0.2 * 0.65)
    true_delta = np.sum((new - tgt) ** 2) - np.sum((canvas - tgt) ** 2)
    assert delta == pytest.approx(true_delta)
    assert delta < 0  # darkening toward a black target always helps


def test_optimal_gray_is_optimal():
    tgt = target()
    canvas = np.ones_like(tgt)
    mask = triangle_mask(BIG, TASK.work)
    g_star = optimal_gray(tgt, canvas, mask, TASK.alpha)
    d_star = paint_delta(tgt, canvas, mask, g_star, TASK.alpha)
    for g in np.linspace(0, 1, 21):
        assert d_star <= paint_delta(tgt, canvas, mask, float(g), TASK.alpha) + 1e-9


def test_evaluator_matches_optional_g():
    tgt = target()
    canvas = np.ones_like(tgt)
    ev = SlotEvaluator(tgt, canvas, TASK.alpha, TASK.k_evals)
    d_auto, _, g_auto, _ = ev(BIG)          # g=None -> optimal
    d_forced, _, g_forced, _ = ev(BIG, g_auto)
    assert g_forced == pytest.approx(g_auto)
    assert d_forced == pytest.approx(d_auto)


def test_budget_enforced():
    tgt = target()
    ev = SlotEvaluator(tgt, np.ones_like(tgt), TASK.alpha, k_max=5)
    for _ in range(5):
        ev(BIG)
    with pytest.raises(BudgetExceeded):
        ev(BIG)
    assert ev.calls == 6  # the raising call still counted


def test_invalid_proposals_return_none_and_cost_a_call():
    tgt = target()
    ev = SlotEvaluator(tgt, np.ones_like(tgt), TASK.alpha, TASK.k_evals)
    off_canvas = np.array([[-50.0, -50.0], [-10.0, -50.0], [-30.0, -10.0]])
    assert ev(off_canvas) is None
    assert ev(np.zeros((2, 2))) is None            # wrong shape
    assert ev(BIG, 1.5) is None                    # g out of range
    assert ev(BIG, float("nan")) is None
    assert ev.calls == 4


def test_anticheat_fabricated_delta_not_painted():
    tgt = np.ones((64, 64))  # all-white target: any paint makes things worse

    def cheater(target_, canvas, t, n_total, evaluate, rng):
        # claims a huge improvement, but painting g=0 on white worsens SSE
        mask = triangle_mask(BIG, 64)
        return (-1e9, BIG, 0.0, mask)

    res = run_heuristic(cheater, tgt, TASK)
    assert res.placed == 0
    assert np.array_equal(res.canvas, np.ones((64, 64)))


def test_bad_return_type_raises_rule_error():
    def bad(target_, canvas, t, n_total, evaluate, rng):
        return (1.0, 2.0)

    with pytest.raises(HeuristicError):
        run_heuristic(bad, target(), TASK)


def test_baseline_like_run_improves_and_is_deterministic():
    tgt = target()

    def baseline(target_, canvas, t, n_total, evaluate, rng):
        H, W = target_.shape
        best = None
        for _ in range(TASK.k_evals):
            pts = rng.integers(0, W, size=(3, 2))
            r = evaluate(pts, rng.random())
            if r is not None and (best is None or r[0] < best[0]):
                best = r
        return best

    res1 = run_heuristic(baseline, tgt, TASK, want_snapshots=True)
    res2 = run_heuristic(baseline, tgt, TASK, want_snapshots=True)
    assert np.array_equal(res1.canvas, res2.canvas)
    assert rmse(res1.canvas, tgt) < blank_rmse(tgt)
    assert res1.placed + res1.skipped == TASK.n_tri
    assert res1.evals_used <= TASK.n_tri * TASK.k_evals
    assert set(res1.snapshots) == set(snapshot_slots(TASK.n_tri))


def test_snapshot_slots_spec_values():
    assert snapshot_slots(80) == (10, 20, 40, 80)


def test_canvas_passed_to_heuristic_is_readonly_copy():
    tgt = target()
    seen = {}

    def prober(target_, canvas, t, n_total, evaluate, rng):
        seen["writable"] = canvas.flags.writeable
        with pytest.raises((ValueError, RuntimeError)):
            canvas[0, 0] = 0.0
        return None

    res = run_heuristic(prober, tgt, TASK, n_tri=1)
    assert seen["writable"] is False
    assert res.skipped == 1
