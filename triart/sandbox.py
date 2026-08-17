"""Sandboxed candidate evaluation.

Each candidate program is validated (AST allowlist), exec'd in a restricted
namespace, and run in a spawned subprocess with a hard wall-clock timeout.
Crashes, timeouts, budget busts, and rule violations come back as structured
errors (score None, fitness = worst case) whose text is fed to the model in
the next generation.

This is crash isolation for a local, non-adversarial experiment — not a
security boundary.
"""
from __future__ import annotations

import ast
import builtins as _builtins
import multiprocessing as mp
import time
import traceback
from dataclasses import dataclass, field
from queue import Empty

import numpy as np

from .config import TaskCfg
from .harness import HeuristicError, HeuristicTimeout, run_heuristic
from .raster import BudgetExceeded
from .score import rmse

ALLOWED_IMPORTS = {"numpy"}

_SAFE_BUILTIN_NAMES = [
    "abs", "min", "max", "sum", "len", "range", "enumerate", "zip", "map",
    "filter", "sorted", "reversed", "int", "float", "bool", "str", "list",
    "dict", "set", "tuple", "frozenset", "round", "divmod", "pow", "any",
    "all", "isinstance", "issubclass", "print", "repr", "hash", "iter",
    "next", "slice", "object", "getattr", "hasattr", "callable",
    "Exception", "BaseException", "ValueError", "TypeError", "IndexError",
    "KeyError", "ZeroDivisionError", "ArithmeticError", "RuntimeError",
    "StopIteration", "OverflowError", "FloatingPointError", "NotImplementedError",
    "True", "False", "None",
]

_ALLOWED_TOP_LEVEL = (
    ast.Import, ast.ImportFrom, ast.FunctionDef, ast.Assign, ast.AnnAssign,
    ast.AugAssign, ast.Expr, ast.If, ast.ClassDef, ast.For, ast.While, ast.Try,
)


def validate_source(src: str) -> str | None:
    """Return an error string if the source breaks the rules, else None."""
    try:
        tree = ast.parse(src)
    except SyntaxError as e:
        return f"SyntaxError: {e}"
    func_names = set()
    for node in tree.body:
        if not isinstance(node, _ALLOWED_TOP_LEVEL):
            return f"top-level {type(node).__name__} statements are not allowed"
        if isinstance(node, ast.FunctionDef):
            func_names.add(node.name)
    if "propose_triangle" not in func_names:
        return "program must define a function named propose_triangle"
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.split(".")[0] not in ALLOWED_IMPORTS:
                    return f"import of '{alias.name}' is not allowed (only numpy)"
        elif isinstance(node, ast.ImportFrom):
            if (node.module or "").split(".")[0] not in ALLOWED_IMPORTS:
                return f"import from '{node.module}' is not allowed (only numpy)"
        elif isinstance(node, ast.Attribute) and node.attr.startswith("__"):
            return f"dunder attribute access ('{node.attr}') is not allowed"
    return None


def _guarded_import(name, *args, **kwargs):
    if name.split(".")[0] in ALLOWED_IMPORTS:
        return __import__(name, *args, **kwargs)
    raise ImportError(f"import of '{name}' is not allowed (only numpy)")


def build_namespace(task: TaskCfg) -> dict:
    safe = {name: getattr(_builtins, name) for name in _SAFE_BUILTIN_NAMES
            if hasattr(_builtins, name)}
    safe["__import__"] = _guarded_import
    return {
        "__builtins__": safe,
        "np": np,
        "A": task.alpha,
        "K": task.k_evals,
        "BudgetExceeded": BudgetExceeded,
    }


@dataclass
class EvalResult:
    ok: bool
    scores: dict[str, float] = field(default_factory=dict)  # name -> RMSE
    mean_rmse: float = float("inf")
    placed: dict[str, int] = field(default_factory=dict)
    evals: dict[str, int] = field(default_factory=dict)
    error_type: str | None = None    # syntax | crash | timeout | budget | rule | no_result
    error: str | None = None
    wall_s: float = 0.0
    renders: dict[str, np.ndarray] = field(default_factory=dict)
    snapshots: dict[int, np.ndarray] = field(default_factory=dict)  # hero only


def _worker(src: str, task_dict: dict,
            named_targets: list[tuple[str, np.ndarray]], n_tri: int,
            per_run_timeout_s: float, want_render: bool, queue: mp.Queue) -> None:
    try:
        task = TaskCfg(**task_dict)
        scores: dict[str, float] = {}
        placed: dict[str, int] = {}
        evals: dict[str, int] = {}
        renders: dict[str, np.ndarray] = {}
        snapshots: dict[int, np.ndarray] = {}
        for i, (name, target) in enumerate(named_targets):
            # Fresh namespace per image: module-level caches in the candidate
            # must not leak information between targets. The harness's seeded
            # rng is the only sanctioned randomness; np.random is seeded too so
            # candidates that misuse it stay comparable across generations.
            np.random.seed(0)
            ns = build_namespace(task)
            exec(compile(src, "<candidate>", "exec"), ns)
            fn = ns["propose_triangle"]
            deadline = time.monotonic() + per_run_timeout_s
            res = run_heuristic(fn, target, task, n_tri=n_tri,
                                deadline=deadline,
                                want_snapshots=want_render and i == 0)
            scores[name] = rmse(res.canvas, target)
            placed[name] = res.placed
            evals[name] = res.evals_used
            if want_render:
                renders[name] = res.canvas
                if i == 0:
                    snapshots = res.snapshots
        queue.put({"ok": True, "scores": scores, "placed": placed,
                   "evals": evals, "renders": renders, "snapshots": snapshots})
    except HeuristicTimeout as e:
        queue.put({"ok": False, "error_type": "timeout", "error": str(e)})
    except BudgetExceeded as e:
        queue.put({"ok": False, "error_type": "budget", "error": str(e)})
    except HeuristicError as e:
        queue.put({"ok": False, "error_type": "rule", "error": str(e)})
    except BaseException:
        queue.put({"ok": False, "error_type": "crash", "error": _crash_summary()})


def _crash_summary() -> str:
    """The candidate's own frames plus the final exception message — the only
    parts of a traceback the model can act on when it reads the feedback."""
    lines = traceback.format_exc().splitlines()
    frames = [l.strip() for l in lines if '"<candidate>"' in l]
    exc = lines[-1].strip() if lines else "unknown error"
    return "; ".join(frames[-2:] + [exc])[:400]


def evaluate_candidate(
    src: str,
    named_targets: list[tuple[str, np.ndarray]],
    task: TaskCfg,
    n_tri: int,
    per_run_timeout_s: float,
    want_render: bool = False,
) -> EvalResult:
    t0 = time.monotonic()
    err = validate_source(src)
    if err is not None:
        return EvalResult(ok=False, error_type="syntax", error=err,
                          wall_s=time.monotonic() - t0)

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    proc = ctx.Process(
        target=_worker,
        args=(src, task.__dict__, named_targets, n_tri,
              per_run_timeout_s, want_render, queue),
    )
    proc.start()
    # Read the result BEFORE joining: with large payloads (renders) the child
    # blocks on the queue's feeder thread until we read, so join-first deadlocks.
    # Grace on top of the in-process deadline: the worker should time itself
    # out cleanly; the hard kill is a backstop for pathological candidates.
    hard_limit = per_run_timeout_s * len(named_targets) + 30.0
    deadline = time.monotonic() + hard_limit
    payload = None
    while time.monotonic() < deadline:
        try:
            payload = queue.get(timeout=0.25)
            break
        except Empty:
            if not proc.is_alive():
                try:
                    payload = queue.get(timeout=0.5)
                except Empty:
                    pass
                break
    if payload is None and proc.is_alive():
        proc.terminate()
        proc.join(5.0)
        if proc.is_alive():
            proc.kill()
            proc.join()
        return EvalResult(ok=False, error_type="timeout",
                          error=f"hard-killed after {hard_limit:.0f}s wall clock",
                          wall_s=time.monotonic() - t0)
    proc.join(5.0)
    if payload is None:
        return EvalResult(ok=False, error_type="no_result",
                          error=f"worker died without result (exit code {proc.exitcode})",
                          wall_s=time.monotonic() - t0)
    wall = time.monotonic() - t0
    if not payload["ok"]:
        return EvalResult(ok=False, error_type=payload["error_type"],
                          error=payload["error"], wall_s=wall)
    scores = payload["scores"]
    return EvalResult(ok=True, scores=scores,
                      mean_rmse=float(np.mean(list(scores.values()))),
                      placed=payload["placed"], evals=payload["evals"],
                      wall_s=wall, renders=payload["renders"],
                      snapshots=payload["snapshots"])
