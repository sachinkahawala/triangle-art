"""The evolution loop: population-based search with stagnation escalation.

Each generation the local model sees the 2-3 best programs (with fitness), one
weaker program for contrast, a short score history, and error feedback from
failed siblings, then proposes k candidates (one per temperature). Candidates
are screened cheaply, then scored on all training images in a sandbox.
Selection keeps the top-`population` programs by mean RMSE; on sustained
stagnation one candidate restarts from the naive baseline to keep diversity.

Usage:
    uv run python -m triart.evolve [--smoke] [--name NAME] [--resume runs/NAME]
                                   [--generations N] [--model NAME]
"""
from __future__ import annotations

import argparse
import datetime as dt
import json
import time
from dataclasses import asdict
from pathlib import Path

import numpy as np
from PIL import Image

from .baseline import BASELINE_SOURCE
from .config import Cfg, load_config
from .llm import LLM, extract_rationale
from .prompts import (RETRY_NUDGE, generation_prompt, render_to_png_b64,
                      system_prompt)
from .sandbox import EvalResult, evaluate_candidate
from .targets import TargetSet, load_target_set


def _save_canvas_png(canvas: np.ndarray, path: Path) -> None:
    img = Image.fromarray((canvas * 255).clip(0, 255).astype(np.uint8))
    img.save(path)


def _atomic_write(path: Path, text: str) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(text)
    tmp.replace(path)


def _entry(gen: int, code: str, res: EvalResult, rationale: str = "") -> dict:
    return {"gen": gen, "score": res.mean_rmse, "per_image": res.scores,
            "placed": res.placed, "code": code, "rationale": rationale}


class Run:
    def __init__(self, cfg: Cfg, run_dir: Path, targets: TargetSet):
        self.cfg = cfg
        self.dir = run_dir
        self.targets = targets
        self.state_path = run_dir / "state.json"
        self.log_path = run_dir / "log.jsonl"
        # mutable state (persisted in state.json)
        self.gen = 0
        self.stagnation = 0
        self.best: dict = {}           # {code, score, per_image, placed, holdout,
                                       #  gen_found, rationale}
        self.population: list[dict] = []  # entries, ascending mean RMSE
        self.baseline_entry: dict = {}    # gen-0 entry, kept for restart prompts
        self.history_rows: list[dict] = []
        self.sibling_feedback: list[str] = []
        self.best_hero_render: np.ndarray | None = None

    # ---------- persistence ----------

    def save_state(self) -> None:
        state = {
            "gen": self.gen,
            "stagnation": self.stagnation,
            "best": self.best,
            "population": self.population,
            "baseline_entry": self.baseline_entry,
            "history_rows": self.history_rows,
            "sibling_feedback": self.sibling_feedback,
            "config": {
                "task": asdict(self.cfg.task),
                "evolve": asdict(self.cfg.evolve),
                "model": asdict(self.cfg.model),
                "train_images": [n for n, _ in self.targets.train],
                "holdout_images": [n for n, _ in self.targets.holdout],
            },
        }
        _atomic_write(self.state_path, json.dumps(state, indent=1))

    def load_state(self) -> None:
        state = json.loads(self.state_path.read_text())
        self.gen = state["gen"]
        self.stagnation = state["stagnation"]
        self.best = state["best"]
        self.population = state["population"]
        self.baseline_entry = state["baseline_entry"]
        self.history_rows = state["history_rows"]
        self.sibling_feedback = state.get("sibling_feedback", [])
        hero_png = self.dir / f"gen_{self.best['gen_found']:03d}" / "best_hero.png"
        if hero_png.exists():
            img = np.asarray(Image.open(hero_png), dtype=np.float64) / 255.0
            self.best_hero_render = img

    def log(self, row: dict) -> None:
        with self.log_path.open("a") as f:
            f.write(json.dumps(row) + "\n")

    # ---------- evaluation helpers ----------

    def screen(self, code: str) -> EvalResult:
        e = self.cfg.evolve
        return evaluate_candidate(
            code, [self.targets.hero], self.cfg.task,
            n_tri=e.screen_n_tri, per_run_timeout_s=e.screen_timeout_s,
        )

    def full_eval(self, code: str, holdout: bool = False) -> EvalResult:
        e = self.cfg.evolve
        named = self.targets.holdout if holdout else self.targets.train
        return evaluate_candidate(
            code, named, self.cfg.task,
            n_tri=self.cfg.task.n_tri,
            per_run_timeout_s=e.run_timeout_s,
            want_render=not holdout,
        )

    # ---------- population ----------

    def merge_population(self, entries: list[dict]) -> None:
        seen: set[str] = set()
        merged: list[dict] = []
        for e in sorted(self.population + entries, key=lambda e: e["score"]):
            if e["code"] not in seen:
                seen.add(e["code"])
                merged.append(e)
        self.population = merged[: self.cfg.evolve.population]

    # ---------- generation steps ----------

    def run_gen0(self) -> None:
        gen_dir = self.dir / "gen_000"
        gen_dir.mkdir(parents=True, exist_ok=True)
        (gen_dir / "cand_0.py").write_text(BASELINE_SOURCE)
        res = self.full_eval(BASELINE_SOURCE)
        if not res.ok:
            raise RuntimeError(f"baseline failed evaluation: {res.error}")
        hold = self.full_eval(BASELINE_SOURCE, holdout=True)
        self.baseline_entry = _entry(0, BASELINE_SOURCE, res,
                                     "naive random-search baseline")
        self.best = dict(self.baseline_entry,
                         holdout=hold.scores if hold.ok else None, gen_found=0)
        self.best_hero_render = res.renders[self.targets.hero[0]]
        self._save_renders(res, gen_dir)
        (gen_dir / "scores.json").write_text(json.dumps(
            {"mean_rmse": res.mean_rmse, "per_image": res.scores,
             "placed": res.placed, "evals": res.evals,
             "holdout": self.best["holdout"]}, indent=1))
        self.population = [self.baseline_entry]
        self.history_rows.append(
            {"gen": 0, "best_score": res.mean_rmse, "accepted": True})
        self.log({"gen": 0, "mode": "baseline", "accepted": True,
                  "best_score_after": res.mean_rmse, "per_image": res.scores,
                  "placed": res.placed, "evals": res.evals,
                  "holdout": self.best["holdout"],
                  "rationale": self.baseline_entry["rationale"]})
        self.save_state()
        print(f"gen   0 | baseline mean RMSE {res.mean_rmse:.4f} "
              f"| per-image {fmt_scores(res.scores)} "
              f"| placed {fmt_placed(res.placed)}")

    def _save_renders(self, res: EvalResult, gen_dir: Path) -> None:
        """One PNG per training image on accepted generations; the hero's is
        canonically named best_hero.png plus its progression snapshots
        (the report generator keys on these names)."""
        hero_name = self.targets.hero[0]
        for name, canvas in res.renders.items():
            fname = "best_hero.png" if name == hero_name else f"best_{name}.png"
            _save_canvas_png(canvas, gen_dir / fname)
        for slot, canvas in res.snapshots.items():
            _save_canvas_png(canvas, gen_dir / f"hero_s{slot:03d}.png")

    def _prompt_messages(self, mode: str, vision_b64: str | None) -> list[dict]:
        cfg = self.cfg
        sys_msg = {"role": "system",
                   "content": system_prompt(cfg.task, cfg.evolve.run_timeout_s)}
        if mode == "restart":
            parents, weaker = [self.baseline_entry], None
        else:
            parents = self.population[:3]
            weaker = self.population[-1] if len(self.population) >= 4 else None
        return [sys_msg] + generation_prompt(
            parents, weaker, self.history_rows, self.sibling_feedback,
            mode, vision_b64)

    def run_generation(self, llm: LLM) -> None:
        cfg, e = self.cfg, self.cfg.evolve
        gen = self.gen + 1
        gen_dir = self.dir / f"gen_{gen:03d}"
        gen_dir.mkdir(parents=True, exist_ok=True)
        t_gen0 = time.monotonic()

        mode = "bold" if self.stagnation >= e.stagnation_escalate else "normal"
        vision_mode = cfg.model.vision_feedback
        use_vision = (vision_mode == "always"
                      or (vision_mode == "stagnation" and self.stagnation >= 1))
        vision_b64 = None
        if use_vision and self.best_hero_render is not None:
            vision_b64 = render_to_png_b64(self.best_hero_render,
                                           self.targets.hero[1])

        messages = self._prompt_messages(mode, vision_b64)
        (gen_dir / "prompt.txt").write_text(
            json.dumps([{k: v for k, v in m.items() if k != "images"}
                        for m in messages], indent=1))

        cand_rows: list[dict] = []
        feedback: list[str] = []
        new_entries: list[dict] = []
        best_cand: tuple[float, str, EvalResult, str] | None = None

        temps = cfg.model.temperatures[:e.candidates_per_gen]
        for i, temp in enumerate(temps):
            # On a plateau, the hottest candidate restarts from the baseline
            # instead of mutating the incumbents (diversity injection).
            cand_mode = ("restart" if mode == "bold" and i == len(temps) - 1
                         else mode)
            cand_messages = (self._prompt_messages("restart", None)
                             if cand_mode == "restart" else messages)
            row: dict = {"idx": i, "temp": temp, "mode": cand_mode}
            g = llm.generate(cand_messages, temperature=temp)
            if g.code is None:
                g = llm.generate(
                    cand_messages + [{"role": "assistant", "content": g.text},
                                     {"role": "user", "content": RETRY_NUDGE}],
                    temperature=temp)
            row.update({"llm_s": round(g.wall_s, 1), "tokens": g.completion_tokens,
                        "tok_s": round(g.tok_s, 1) if g.tok_s else None})
            (gen_dir / f"reply_{i}.txt").write_text(g.text)
            if g.code is None:
                row.update({"error_type": "no_code", "score": None})
                feedback.append(f"attempt {i}: reply contained no usable python block")
                cand_rows.append(row)
                continue
            (gen_dir / f"cand_{i}.py").write_text(g.code)
            rationale = extract_rationale(g.text)
            row["rationale"] = rationale
            idea = rationale[:120] or "(no stated idea)"

            scr = self.screen(g.code)
            if not scr.ok:
                row.update({"error_type": scr.error_type, "score": None,
                            "error_head": (scr.error or "")[:300]})
                feedback.append(
                    f'idea "{idea}" -> {scr.error_type}: {(scr.error or "")[:200]}')
                cand_rows.append(row)
                continue

            res = self.full_eval(g.code)
            if not res.ok:
                row.update({"error_type": res.error_type, "score": None,
                            "error_head": (res.error or "")[:300]})
                feedback.append(
                    f'idea "{idea}" -> {res.error_type}: {(res.error or "")[:200]}')
                cand_rows.append(row)
                continue

            row.update({"score": round(res.mean_rmse, 5),
                        "per_image": {k: round(v, 5) for k, v in res.scores.items()},
                        "placed": res.placed, "evals": res.evals,
                        "eval_s": round(res.wall_s, 1)})
            cand_rows.append(row)
            new_entries.append(_entry(gen, g.code, res, rationale))
            if best_cand is None or res.mean_rmse < best_cand[0]:
                best_cand = (res.mean_rmse, g.code, res, rationale)
            if res.mean_rmse >= self.best["score"]:
                feedback.append(
                    f'idea "{idea}" -> scored {res.mean_rmse:.4f} '
                    f'(best {self.best["score"]:.4f}), not an improvement')

        accepted = (best_cand is not None
                    and best_cand[0] < self.best["score"] - 1e-9)
        holdout_scores = None
        if accepted:
            score, code, res, rationale = best_cand
            hold = self.full_eval(code, holdout=True)
            holdout_scores = hold.scores if hold.ok else None
            self.best = {"code": code, "score": score, "per_image": res.scores,
                         "placed": res.placed, "holdout": holdout_scores,
                         "gen_found": gen, "rationale": rationale}
            self.best_hero_render = res.renders[self.targets.hero[0]]
            self._save_renders(res, gen_dir)
            self.stagnation = 0
        else:
            self.stagnation += 1
        self.merge_population(new_entries)

        self.sibling_feedback = feedback[:4]
        self.gen = gen
        self.history_rows.append(
            {"gen": gen, "best_score": self.best["score"], "accepted": accepted})
        (gen_dir / "scores.json").write_text(json.dumps(
            {"candidates": cand_rows, "accepted": accepted,
             "best_score_after": self.best["score"], "holdout": holdout_scores},
            indent=1))
        self.log({"gen": gen, "mode": mode, "vision": use_vision,
                  "candidates": cand_rows, "accepted": accepted,
                  "best_score_after": self.best["score"],
                  "holdout": holdout_scores,
                  "population": [round(p["score"], 5) for p in self.population],
                  "gen_wall_s": round(time.monotonic() - t_gen0, 1)})
        self.save_state()

        mark = "ACCEPTED" if accepted else f"kept best (stagnation {self.stagnation})"
        cand_summary = ", ".join(
            f"{c['score']:.4f}" if c.get("score") is not None
            else (c.get("error_type") or "?") for c in cand_rows)
        print(f"gen {gen:3d} | best {self.best['score']:.4f} | {mark:>26} "
              f"| mode {mode}{'+vision' if use_vision else ''} | cands [{cand_summary}] "
              f"| {time.monotonic() - t_gen0:5.0f}s")


def fmt_scores(scores: dict[str, float]) -> str:
    return ", ".join(f"{k} {v:.4f}" for k, v in scores.items())


def fmt_placed(placed: dict[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in placed.items())


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--config", default="config.yaml")
    ap.add_argument("--smoke", action="store_true")
    ap.add_argument("--name", default=None, help="run directory name")
    ap.add_argument("--resume", default=None, help="existing run directory")
    ap.add_argument("--generations", type=int, default=None)
    ap.add_argument("--model", default=None)
    args = ap.parse_args()

    cfg = load_config(args.config, smoke=args.smoke)
    if args.generations is not None:
        cfg.evolve.generations = args.generations
    if args.model is not None:
        cfg.model.name = args.model

    targets = load_target_set(cfg.assets_dir, cfg.task.work)
    print("train:", [n for n, _ in targets.train],
          "| holdout:", [n for n, _ in targets.holdout])

    if args.resume:
        run_dir = Path(args.resume)
        run = Run(cfg, run_dir, targets)
        run.load_state()
        print(f"resuming {run_dir} at gen {run.gen}, best {run.best['score']:.4f}")
    else:
        name = args.name or dt.datetime.now().strftime("%Y%m%d-%H%M%S")
        if cfg.smoke:
            name += "-smoke"
        run_dir = cfg.runs_dir / name
        run_dir.mkdir(parents=True, exist_ok=False)
        run = Run(cfg, run_dir, targets)

    llm = LLM(cfg.model)
    print(f"model: {llm.model} | run dir: {run.dir}")

    try:
        if run.gen == 0 and not run.best:
            run.run_gen0()
        while run.gen < cfg.evolve.generations:
            run.run_generation(llm)
    except KeyboardInterrupt:
        print(f"\ninterrupted — state saved; resume with:\n"
              f"  uv run python -m triart.evolve --resume {run.dir}"
              + (" --smoke" if cfg.smoke else ""))
        return
    print(f"\ndone: best mean RMSE {run.best['score']:.4f} "
          f"(found gen {run.best['gen_found']}) | run dir: {run.dir}")


if __name__ == "__main__":
    main()
