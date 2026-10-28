"""Experiment plans (``configs/experiments/*.yaml``): the runs of an experiment and their state.

A plan lists named entries, each with ``--model`` files, optional overrides and seeds; a
``common`` list of overrides applies to every entry. Runs are ordered seed by seed (all entries
with their first seed, then the second, ...), so an interrupted plan still has balanced results.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import DictConfig, OmegaConf

from connectome_driver.utils.config import compose_config


@dataclass
class Run:
    exp_name: str
    seed: int
    model: list[str]
    overrides: list[str] = field(default_factory=list)

    def dir(self, run_dir: Path) -> Path:
        return run_dir / self.exp_name / f"seed{self.seed}"

    def status(self, run_dir: Path) -> str:
        """``done`` (trained and evaluated), ``trained``, ``partial`` (started) or ``pending``."""
        d = self.dir(run_dir)
        if (d / "DONE").exists():
            return "done" if (d / "eval.json").exists() else "trained"
        return "partial" if d.exists() else "pending"

    def config(self, base: str | Path = "configs/train.yaml") -> DictConfig:
        return compose_config(base, self.model, [*self.overrides, f"exp_name={self.exp_name}", f"seed={self.seed}"])


def load_plan(path: str | Path) -> list[Run]:
    plan = OmegaConf.load(path)
    common = list(plan.get("common", []))
    entries = {name: OmegaConf.to_container(e) for name, e in plan.runs.items()}
    max_seeds = max(len(e["seeds"]) for e in entries.values())
    runs = []
    for i in range(max_seeds):  # seed by seed
        for name, e in entries.items():
            if i < len(e["seeds"]):
                runs.append(Run(name, e["seeds"][i], e["model"], common + e.get("overrides", [])))
    return runs
