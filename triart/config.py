"""Configuration loading. All experiment knobs live in config.yaml."""
from __future__ import annotations

import copy
from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass
class TaskCfg:
    work: int = 220          # working resolution (square)
    n_tri: int = 80          # hard triangle budget
    k_evals: int = 45        # max evaluate() calls per slot
    alpha: float = 0.65      # paint opacity
    seed: int = 0            # per-run RNG seed


@dataclass
class EvolveCfg:
    generations: int = 40
    candidates_per_gen: int = 4
    population: int = 6
    run_timeout_s: float = 120.0
    screen_n_tri: int = 20
    screen_timeout_s: float = 30.0
    stagnation_escalate: int = 3
    seed: int = 0


@dataclass
class ModelCfg:
    name: str = "qwen3.8:27b"
    fallbacks: list[str] = field(default_factory=list)
    endpoint: str = "http://localhost:11434/v1"
    max_tokens: int = 2200
    temperatures: list[float] = field(default_factory=lambda: [0.6, 0.7, 0.85, 1.0])
    thinking: bool = False
    vision_feedback: str = "stagnation"  # off | stagnation | always


@dataclass
class Cfg:
    task: TaskCfg
    evolve: EvolveCfg
    model: ModelCfg
    assets_dir: Path
    runs_dir: Path
    smoke: bool = False


def load_config(path: str | Path = "config.yaml", smoke: bool = False) -> Cfg:
    raw = yaml.safe_load(Path(path).read_text())
    raw = copy.deepcopy(raw)
    if smoke:
        for section, overrides in (raw.get("smoke") or {}).items():
            raw.setdefault(section, {}).update(overrides)
    root = Path(path).resolve().parent
    paths = raw.get("paths", {})
    return Cfg(
        task=TaskCfg(**raw.get("task", {})),
        evolve=EvolveCfg(**raw.get("evolve", {})),
        model=ModelCfg(**raw.get("model", {})),
        assets_dir=root / paths.get("assets", "assets"),
        runs_dir=root / paths.get("runs", "runs"),
        smoke=smoke,
    )
