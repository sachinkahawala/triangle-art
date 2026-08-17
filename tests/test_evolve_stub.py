"""Offline end-to-end check of the evolution loop machinery with a stubbed LLM
(no Ollama needed): gen 0 + one generation, state persistence, resume, and the
report generator over the resulting run directory."""
import json
import sys
from dataclasses import dataclass
from pathlib import Path

from triart.config import Cfg, EvolveCfg, ModelCfg, TaskCfg
from triart.evolve import Run
from triart.targets import load_target_set

IMPROVED = """\
Rationale: use the harness's optimal gray instead of a random one — every try
lands at the least-squares best gray for its triangle, so each accepted paint
removes more error for the same K budget.

```python
# Optimal-gray baseline: random vertices, harness-optimal gray.
def propose_triangle(target, canvas, t, n_total, evaluate, rng):
    H, W = target.shape
    best = None
    for _ in range(K):
        pts = rng.integers(0, W, size=(3, 2))
        r = evaluate(pts)
        if r is not None and (best is None or r[0] < best[0]):
            best = r
    return best
```
"""


@dataclass
class _Gen:
    text: str
    code: str | None
    wall_s: float = 0.1
    completion_tokens: int = 100
    tok_s: float = 100.0


class StubLLM:
    model = "stub"

    def generate(self, messages, temperature):
        from triart.llm import extract_code
        return _Gen(text=IMPROVED, code=extract_code(IMPROVED))


def small_cfg(tmp_path: Path) -> Cfg:
    return Cfg(
        task=TaskCfg(work=64, n_tri=12, k_evals=10, alpha=0.65, seed=0),
        evolve=EvolveCfg(generations=1, candidates_per_gen=2, population=4,
                         run_timeout_s=30, screen_n_tri=4, screen_timeout_s=15),
        model=ModelCfg(vision_feedback="off",
                       temperatures=[0.6, 0.7]),
        assets_dir=tmp_path / "assets",   # empty -> synthetic targets
        runs_dir=tmp_path / "runs",
    )


def test_loop_report_and_resume(tmp_path):
    cfg = small_cfg(tmp_path)
    targets = load_target_set(cfg.assets_dir, cfg.task.work)
    assert len(targets.train) == 3 and len(targets.holdout) == 2

    run_dir = tmp_path / "runs" / "stub"
    run_dir.mkdir(parents=True)
    run = Run(cfg, run_dir, targets)
    run.run_gen0()
    assert run.best["score"] < 1.0
    assert (run_dir / "gen_000" / "best_hero.png").exists()

    run.run_generation(StubLLM())
    assert run.gen == 1
    state = json.loads((run_dir / "state.json").read_text())
    rows = [json.loads(l) for l in
            (run_dir / "log.jsonl").read_text().splitlines()]
    assert len(rows) == 2
    # optimal gray strictly beats random gray -> the stub must be accepted
    assert rows[1]["accepted"], rows[1]
    assert state["best"]["score"] < state["baseline_entry"]["score"]
    assert len(state["population"]) == 2  # baseline + one deduped improvement
    assert state["best"]["rationale"].startswith("Rationale: use the harness")
    snaps = sorted((run_dir / "gen_001").glob("hero_s*.png"))
    assert len(snaps) == 4

    # resume restores state
    run2 = Run(cfg, run_dir, targets)
    run2.load_state()
    assert run2.gen == 1
    assert run2.best["score"] == run.best["score"]
    assert run2.best_hero_render is not None

    # report generator works off the run directory alone
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "report"))
    import make_report
    data = make_report.load_run(run_dir)
    rpt = run_dir / "report"
    rpt.mkdir()
    make_report.make_filmstrip(run_dir, data["rows"], rpt / "filmstrip.png")
    make_report.make_fitness_curve(data["rows"], rpt / "fitness.png")
    make_report.make_progression(run_dir, data["state"], rpt / "progression.png")
    make_report.make_gif(run_dir, data["rows"], rpt / "evolution.gif")
    diffs = make_report.make_diffs(run_dir, data["rows"], rpt / "diffs.md")
    rats = make_report.make_rationales(data["rows"], rpt / "rationales.md")
    make_report.make_html(run_dir, data["state"], data["rows"], rpt, diffs, rats)
    for f in ("filmstrip.png", "fitness.png", "progression.png",
              "evolution.gif", "diffs.md", "rationales.md", "report.html"):
        assert (rpt / f).exists(), f
