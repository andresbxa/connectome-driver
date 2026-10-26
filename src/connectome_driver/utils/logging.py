"""Run outputs: TensorBoard scalars, a CSV of every finished episode, config and checkpoint.

Layout: ``<run_dir>/<exp_name>/seed<seed>/`` with ``config.yaml``, ``episodes.csv``,
``iterations.csv``, TensorBoard event files and ``policy.pt``.
"""

from __future__ import annotations

import csv
import time
from pathlib import Path

import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.tensorboard import SummaryWriter


class RunLogger:
    def __init__(self, cfg: DictConfig, print_every: int = 10):
        self.dir = Path(cfg.run_dir) / cfg.exp_name / f"seed{cfg.seed}"
        self.dir.mkdir(parents=True, exist_ok=True)
        OmegaConf.save(cfg, self.dir / "config.yaml")
        self.tb = SummaryWriter(str(self.dir))
        self._episodes = open(self.dir / "episodes.csv", "w", newline="")  # noqa: SIM115 (kept open for the run)
        self._ep_writer = csv.writer(self._episodes)
        self._ep_writer.writerow(["step", "return", "length_frames", "wall_time"])
        self._iterations = None
        self.recent_returns: list[float] = []
        self.print_every = print_every
        self._iteration = 0
        self._start = time.time()

    def log_episode(self, step: int, ret: float, length: int) -> None:
        self._ep_writer.writerow([step, f"{ret:.2f}", length, f"{time.time() - self._start:.1f}"])
        self.tb.add_scalar("charts/episodic_return", ret, step)
        self.tb.add_scalar("charts/episodic_length_frames", length, step)
        self.recent_returns = (self.recent_returns + [ret])[-20:]

    def log_iteration(self, step: int, stats: dict[str, float]) -> None:
        for k, v in stats.items():
            self.tb.add_scalar(f"train/{k}", v, step)
        if self._iterations is None:
            self._iterations = open(self.dir / "iterations.csv", "w", newline="")  # noqa: SIM115
            self._it_writer = csv.DictWriter(self._iterations, ["step", *stats])
            self._it_writer.writeheader()
        self._it_writer.writerow({"step": step, **{k: f"{v:.6g}" for k, v in stats.items()}})
        self._iteration += 1
        if self._iteration % self.print_every == 0:
            self._episodes.flush()
            self._iterations.flush()
            mean_ret = sum(self.recent_returns) / max(len(self.recent_returns), 1)
            print(
                f"step {step:>9,}  return(last 20) {mean_ret:8.1f}  "
                f"sps {stats['steps_per_second']:6.0f}  kl {stats['approx_kl']:.4f}  "
                f"ent {stats['entropy']:.3f}  ev {stats['explained_variance']:.2f}",
                flush=True,
            )

    def save_checkpoint(self, policy: torch.nn.Module, name: str = "policy.pt") -> Path:
        path = self.dir / name
        torch.save(policy.state_dict(), path)
        return path

    def close(self) -> None:
        self.tb.close()
        self._episodes.close()
        if self._iterations is not None:
            self._iterations.close()
