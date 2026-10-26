"""Run an experiment plan: every (variant, seed) pair, training then evaluation.

Finished runs (``DONE`` marker from train.py and ``eval.json``) are skipped, so the command
can be re-run after an interruption or split across machines with ``--only``. Runs are
ordered seed by seed. Each run logs to ``<run_dir>/<exp_name>/seed<k>.log``.

Usage:
    python scripts/run_experiments.py configs/experiments/main.yaml --status
    python scripts/run_experiments.py configs/experiments/main.yaml --only connectome nocore
    python scripts/run_experiments.py configs/experiments/main.yaml --parallel 2

Wall-clock times of concurrent runs include contention, so they are not used as a
cost metric; per-step time and memory come from scripts/profile_core.py.
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path

from omegaconf import OmegaConf

from connectome_driver.utils.config import load_config


@dataclass
class Run:
    exp_name: str
    seed: int
    model: list[str]
    overrides: list[str] = field(default_factory=list)

    def dir(self, run_dir: Path) -> Path:
        return run_dir / self.exp_name / f"seed{self.seed}"

    def status(self, run_dir: Path) -> str:
        d = self.dir(run_dir)
        if (d / "DONE").exists():
            return "done" if (d / "eval.json").exists() else "trained"
        return "partial" if d.exists() else "pending"


def load_plan(path: Path) -> list[Run]:
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


def execute(run: Run, run_dir: Path, episodes: int) -> str:
    d = run.dir(run_dir)
    log = run_dir / run.exp_name / f"seed{run.seed}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    python = [sys.executable, "-W", "ignore", "-u"]
    start = time.time()
    if run.status(run_dir) != "trained":
        cmd = [*python, "scripts/train.py"]
        for m in run.model:
            cmd += ["--model", m]
        cmd += [f"exp_name={run.exp_name}", f"seed={run.seed}", f"run_dir={run_dir.as_posix()}", *run.overrides]
        with open(log, "w") as f:
            if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode != 0:
                return f"FAILED  {run.exp_name} seed{run.seed} (see {log})"
    with open(log, "a") as f:
        cmd = [*python, "scripts/evaluate.py", str(d), "--episodes", str(episodes)]
        if subprocess.run(cmd, stdout=f, stderr=subprocess.STDOUT, check=False).returncode != 0:
            return f"EVAL FAILED  {run.exp_name} seed{run.seed} (see {log})"
    return f"done    {run.exp_name} seed{run.seed} in {(time.time() - start) / 3600:.2f} h"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("plan", type=Path)
    parser.add_argument("--only", nargs="*", default=None, help="experiment names to run")
    parser.add_argument("--seeds", nargs="*", type=int, default=None, help="seeds to run")
    parser.add_argument("--parallel", type=int, default=1, help="runs at the same time")
    parser.add_argument("--episodes", type=int, default=20, help="evaluation episodes per run")
    parser.add_argument("--status", action="store_true", help="only print the state of each run")
    args = parser.parse_args()

    run_dir = Path(load_config("configs/train.yaml").run_dir)
    runs = [
        r for r in load_plan(args.plan)
        if (args.only is None or r.exp_name in args.only) and (args.seeds is None or r.seed in args.seeds)
    ]
    for r in runs:
        print(f"{r.status(run_dir):8} {r.exp_name} seed{r.seed}")
    if args.status:
        return

    todo = [r for r in runs if r.status(run_dir) != "done"]
    print(f"\n{len(todo)} runs to do, {args.parallel} at a time\n", flush=True)
    with ThreadPoolExecutor(max_workers=args.parallel) as pool:
        for message in pool.map(lambda r: execute(r, run_dir, args.episodes), todo):
            print(time.strftime("%H:%M ") + message, flush=True)


if __name__ == "__main__":
    main()
