"""Train a PPO policy on CarRacing.

Usage:
    python scripts/train.py [--model configs/model/mlp.yaml] [key=value ...]
    python scripts/train.py exp_name=mlp_baseline seed=2 ppo.total_steps=2_000_000
"""

from __future__ import annotations

import argparse

import torch
from omegaconf import OmegaConf

from connectome_driver.envs.car_racing import make_vector_env, observation_shape
from connectome_driver.models.policy import build_policy
from connectome_driver.rl.ppo import PPO
from connectome_driver.utils.config import load_config
from connectome_driver.utils.logging import RunLogger
from connectome_driver.utils.seeding import seed_everything


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--config", default="configs/train.yaml")
    parser.add_argument("--model", default=None, help="YAML merged into the `model` section")
    parser.add_argument("overrides", nargs="*")
    args = parser.parse_args()

    cfg = load_config(args.config)
    if args.model:
        cfg.model = OmegaConf.merge(cfg.model, OmegaConf.load(args.model))
    cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist(args.overrides))

    seed_everything(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")
    envs = make_vector_env(cfg.env, cfg.ppo.num_envs)
    policy = build_policy(cfg.model, observation_shape(cfg.env), envs.single_action_space.n).to(device)

    logger = RunLogger(cfg)
    counts = policy.parameter_counts()
    print(f"run dir: {logger.dir}\nparameters: {counts} total {sum(counts.values()):,}")
    for name, n in counts.items():
        logger.tb.add_scalar(f"params/{name}", n, 0)

    trainer = PPO(cfg.ppo, policy, envs, device, logger)
    try:
        trainer.train(cfg.seed)
    finally:
        if cfg.log.checkpoint:
            print(f"checkpoint: {logger.save_checkpoint(policy)}")
        logger.close()
        envs.close()


if __name__ == "__main__":
    main()
