import gymnasium as gym
import numpy as np
from gymnasium.envs.box2d.car_racing import CarRacing
from omegaconf import OmegaConf

from connectome_driver.envs.car_racing import FastCarRacing, FrameSkip, make_env

ENV_CFG = OmegaConf.create(
    {"frame_skip": 4, "frame_stack": 4, "resolution": 84, "reward_clip": 1.0,
     "lap_complete_percent": 0.95}
)


def test_skipping_renders_does_not_change_dynamics():
    """Fast env + frame skip == original env stepped frame by frame, rendering every frame."""
    rng = np.random.default_rng(0)
    actions = rng.integers(0, 5, size=60)

    ref = CarRacing(continuous=False)
    ref.reset(seed=3)
    ref_rewards = []
    for a in actions:
        total = 0.0
        for _ in range(4):
            obs_ref, r, *_ = ref.step(int(a))
            total += r
        ref_rewards.append(total)

    fast = FrameSkip(FastCarRacing(continuous=False), 4)
    fast.reset(seed=3)
    fast_rewards = []
    for a in actions:
        obs_fast, r, *_ = fast.step(int(a))
        fast_rewards.append(r)

    np.testing.assert_allclose(fast_rewards, ref_rewards)
    np.testing.assert_array_equal(obs_fast, obs_ref)
    np.testing.assert_allclose(fast.unwrapped.car.hull.position, ref.car.hull.position)


class CountingEnv(gym.Env):
    observation_space = gym.spaces.Box(0, 1, (1,))
    action_space = gym.spaces.Discrete(2)

    def __init__(self, end_at: int):
        self.end_at, self.t = end_at, 0

    def reset(self, *, seed=None, options=None):
        self.t = 0
        return np.zeros(1, np.float32), {}

    def step(self, action):
        self.t += 1
        return np.full(1, self.t, np.float32), 1.0, self.t >= self.end_at, False, {}


def test_frame_skip_sums_rewards_and_stops_at_episode_end():
    env = FrameSkip(CountingEnv(end_at=6), 4)
    env.reset()
    obs, reward, terminated, *_ = env.step(0)
    assert (obs[0], reward, terminated) == (4, 4.0, False)
    obs, reward, terminated, *_ = env.step(0)
    assert (obs[0], reward, terminated) == (6, 2.0, True)


def test_preprocessed_observation():
    env = make_env(ENV_CFG)
    obs, _ = env.reset(seed=0)
    assert obs.shape == (4, 84, 84) and obs.dtype == np.uint8
    assert env.action_space.n == 5
    _, reward, *_ = env.step(3)
    assert -1.0 <= reward <= 1.0
    env.close()
