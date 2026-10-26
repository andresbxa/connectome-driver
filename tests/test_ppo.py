import gymnasium as gym
import numpy as np
import pytest
import torch
from conftest import CONFIG
from omegaconf import OmegaConf

from connectome_driver.envs.car_racing import make_vector_env, observation_shape
from connectome_driver.models.policy import build_policy
from connectome_driver.rl.ppo import PPO, compute_gae
from connectome_driver.utils.config import load_config
from connectome_driver.utils.logging import RunLogger

TRAIN_CONFIG = CONFIG.parent / "train.yaml"


def naive_gae(rewards, values, ended, last_value, gamma, lam):
    """Per-environment reference: discounted sum of TD errors until the episode ends."""
    T, N = rewards.shape
    values_ext = np.vstack([values, last_value[None]])
    adv = np.zeros((T, N))
    for n in range(N):
        for t in range(T):
            total, discount = 0.0, 1.0
            for k in range(t, T):
                not_ended = 1.0 - ended[k, n]
                delta = rewards[k, n] + gamma * values_ext[k + 1, n] * not_ended - values[k, n]
                total += discount * delta
                if ended[k, n]:
                    break
                discount *= gamma * lam
            adv[t, n] = total
    return adv


def test_gae_matches_reference():
    rng = np.random.default_rng(0)
    T, N = 12, 3
    rewards, values = rng.normal(size=(T, N)), rng.normal(size=(T, N))
    ended = (rng.random((T, N)) < 0.2).astype(float)
    last_value = rng.normal(size=N)
    adv, ret = compute_gae(*(torch.tensor(x) for x in (rewards, values, ended, last_value)), 0.99, 0.95)
    np.testing.assert_allclose(adv.numpy(), naive_gae(rewards, values, ended, last_value, 0.99, 0.95), rtol=1e-10)
    np.testing.assert_allclose(ret.numpy(), adv.numpy() + values)


def small_cfg(tmp_path, **overrides):
    cfg = load_config(TRAIN_CONFIG)
    cfg.run_dir = str(tmp_path)
    cfg.model.readout_neurons = 1305  # no graph in these tests
    return OmegaConf.merge(cfg, OmegaConf.create(overrides))


def test_policy_shapes(tmp_path):
    cfg = small_cfg(tmp_path)
    policy = build_policy(cfg.model, observation_shape(cfg.env), 5)
    obs = torch.randint(0, 256, (3, *observation_shape(cfg.env)), dtype=torch.uint8)
    action, logprob, entropy, value = policy.act(obs)
    assert action.shape == logprob.shape == entropy.shape == value.shape == (3,)
    assert policy.core(policy.encoder(obs)).shape == (3, cfg.model.readout_neurons, cfg.model.node_dim)


def test_ppo_runs_one_iteration_on_car_racing(tmp_path):
    cfg = small_cfg(
        tmp_path,
        device="cpu",
        ppo={"num_envs": 2, "rollout_steps": 8, "num_minibatches": 2, "update_epochs": 1, "total_steps": 16},
    )
    envs = make_vector_env(cfg.env, cfg.ppo.num_envs, asynchronous=False)
    policy = build_policy(cfg.model, observation_shape(cfg.env), 5)
    logger = RunLogger(cfg)
    PPO(cfg.ppo, policy, envs, torch.device("cpu"), logger).train(seed=0)
    logger.close()
    envs.close()
    assert (logger.dir / "iterations.csv").read_text().count("\n") == 2  # header + 1 iteration


def cartpole():
    # Actor and critic share their trunk, so returns must be on a small scale (as CarRacing's
    # clipped rewards are); otherwise the value loss dominates the clipped gradient norm.
    env = gym.wrappers.RecordEpisodeStatistics(gym.make("CartPole-v1"))
    return gym.wrappers.TransformReward(env, lambda r: 0.1 * r)


@pytest.mark.slow
def test_ppo_learns_cartpole(tmp_path):
    """The PPO implementation alone, on a task it must solve quickly."""
    cfg = small_cfg(
        tmp_path,
        device="cpu",
        model={"encoder": {"type": "flat", "feature_dim": 64}, "readout_neurons": 16, "node_dim": 4,
               "decoder": {"hidden_dim": 64}, "core": {"type": "mlp", "hidden_dims": [64]}},
        ppo={"num_envs": 4, "rollout_steps": 128, "total_steps": 100_000, "clip_coef": 0.2},
    )
    torch.manual_seed(0)
    envs = gym.vector.SyncVectorEnv(
        [cartpole for _ in range(4)],
        autoreset_mode=gym.vector.AutoresetMode.SAME_STEP,
    )
    policy = build_policy(cfg.model, envs.single_observation_space.shape, 2)
    logger = RunLogger(cfg)
    PPO(cfg.ppo, policy, envs, torch.device("cpu"), logger).train(seed=0)
    logger.close()
    envs.close()
    assert np.mean(logger.recent_returns) >= 195  # the classic "solved" bar for CartPole
