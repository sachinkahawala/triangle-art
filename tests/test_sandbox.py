"""Sandbox rules: AST validation, structured error reporting, end-to-end
baseline evaluation in a subprocess."""
import numpy as np

from triart.baseline import BASELINE_SOURCE
from triart.config import TaskCfg
from triart.sandbox import evaluate_candidate, validate_source
from triart.score import blank_rmse
from triart.targets import synthetic_target

TASK = TaskCfg(work=64, n_tri=12, k_evals=10, alpha=0.65, seed=0)


def targets():
    return [("blobs", synthetic_target("blobs", TASK.work))]


def test_validate_baseline_ok():
    assert validate_source(BASELINE_SOURCE) is None


def test_validate_rejects_bad_sources():
    assert validate_source("x = ][") is not None                      # syntax
    assert validate_source("def foo():\n    return None\n") is not None  # name
    assert "import" in validate_source(
        "import os\ndef propose_triangle(*a):\n    return None\n")
    assert "import" in validate_source(
        "import math\ndef propose_triangle(*a):\n    return None\n")  # numpy only
    assert "dunder" in validate_source(
        "def propose_triangle(*a):\n    return a[0].__class__\n")


def test_baseline_evaluates_and_beats_blank():
    tgts = targets()
    res = evaluate_candidate(BASELINE_SOURCE, tgts, TASK,
                             n_tri=TASK.n_tri, per_run_timeout_s=30)
    assert res.ok, res.error
    assert res.mean_rmse < blank_rmse(tgts[0][1])
    assert res.placed["blobs"] <= TASK.n_tri
    assert res.evals["blobs"] <= TASK.n_tri * TASK.k_evals


def test_want_render_returns_canvas_and_snapshots():
    res = evaluate_candidate(BASELINE_SOURCE, targets(), TASK,
                             n_tri=TASK.n_tri, per_run_timeout_s=30,
                             want_render=True)
    assert res.ok
    assert res.renders["blobs"].shape == (TASK.work, TASK.work)
    assert len(res.snapshots) == 4


def test_crash_reported():
    src = "def propose_triangle(target, canvas, t, n, evaluate, rng):\n    return 1/0\n"
    res = evaluate_candidate(src, targets(), TASK,
                             n_tri=2, per_run_timeout_s=30)
    assert not res.ok and res.error_type == "crash"
    assert "ZeroDivision" in res.error


def test_budget_bust_reported():
    src = (
        "def propose_triangle(target, canvas, t, n, evaluate, rng):\n"
        "    H, W = target.shape\n"
        "    for _ in range(K + 5):\n"
        "        evaluate(rng.integers(0, W, size=(3, 2)))\n"
        "    return None\n"
    )
    res = evaluate_candidate(src, targets(), TASK,
                             n_tri=2, per_run_timeout_s=30)
    assert not res.ok and res.error_type == "budget"


def test_bad_return_reported_as_rule():
    src = "def propose_triangle(target, canvas, t, n, evaluate, rng):\n    return 42\n"
    res = evaluate_candidate(src, targets(), TASK,
                             n_tri=2, per_run_timeout_s=30)
    assert not res.ok and res.error_type == "rule"


def test_determinism_across_processes():
    r1 = evaluate_candidate(BASELINE_SOURCE, targets(), TASK,
                            n_tri=TASK.n_tri, per_run_timeout_s=30)
    r2 = evaluate_candidate(BASELINE_SOURCE, targets(), TASK,
                            n_tri=TASK.n_tri, per_run_timeout_s=30)
    assert r1.ok and r2.ok
    assert r1.scores == r2.scores
