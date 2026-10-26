"""Evaluate a trained policy on fresh CarRacing tracks.

Plays ``--episodes`` full episodes on track seeds never used in training (10_000 + i) and
writes the returns to ``<run>/eval.json``. Returns are the standard, unclipped CarRacing score.

Usage:
    python scripts/evaluate.py runs/mlp_baseline/seed1 [--episodes 20] [--greedy]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from omegaconf import OmegaConf

from connectome_driver.envs.car_racing import make_env, observation_shape
from connectome_driver.models.policy import build_policy

EVAL_SEED_OFFSET = 10_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_dir", type=Path)
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--greedy", action="store_true", help="take the most likely action")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.run_dir / "config.yaml")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    env = make_env(cfg.env)
    policy = build_policy(cfg.model, observation_shape(cfg.env), env.action_space.n).to(device)
    policy.load_state_dict(torch.load(args.run_dir / "policy.pt", map_location=device))
    policy.eval()

    returns = []
    for i in range(args.episodes):
        obs, _ = env.reset(seed=EVAL_SEED_OFFSET + i)
        while True:
            with torch.no_grad():
                logits, _ = policy(torch.as_tensor(obs, device=device)[None])
            action = logits.argmax(-1) if args.greedy else torch.distributions.Categorical(logits=logits).sample()
            obs, _, terminated, truncated, info = env.step(int(action))
            if terminated or truncated:
                returns.append(float(info["episode"]["r"]))
                print(f"episode {i:>3}  return {returns[-1]:7.1f}", flush=True)
                break
    env.close()

    result = {
        "episodes": args.episodes,
        "policy": "greedy" if args.greedy else "sampled",
        "mean": float(np.mean(returns)),
        "std": float(np.std(returns)),
        "min": float(np.min(returns)),
        "max": float(np.max(returns)),
        "returns": returns,
    }
    name = "eval_greedy.json" if args.greedy else "eval.json"
    (args.run_dir / name).write_text(json.dumps(result, indent=2))
    print(f"\nmean {result['mean']:.1f} +- {result['std']:.1f}  ({result['policy']}, {args.episodes} episodes)")


if __name__ == "__main__":
    main()
