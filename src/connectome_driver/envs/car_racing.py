"""CarRacing with discrete actions, frame skip and pixel preprocessing.

Wrapper stack, innermost first:

    FastCarRacing          physics + rendering (renders only the frames the agent sees)
    TimeLimit              1000 frames, as in the registered CarRacing-v3
    RecordEpisodeStatistics  true episode return/length in frames, before any reward clipping
    FrameSkip(4)           repeat each action for 4 frames and sum the rewards
    GrayscaleObservation   96x96x3 -> 96x96
    ResizeObservation      -> 84x84
    FrameStackObservation  -> 4x84x84 (uint8)
    ClipReward             training reward in [-1, 1]

The reported episode return is the standard CarRacing score (sum of unclipped rewards over
all frames), so it is comparable with other work. Agent steps are 4 frames each.
"""

from __future__ import annotations

import gymnasium as gym
from gymnasium.envs.box2d.car_racing import CarRacing
from gymnasium.wrappers import (
    ClipReward,
    FrameStackObservation,
    GrayscaleObservation,
    RecordEpisodeStatistics,
    ResizeObservation,
    TimeLimit,
)
from omegaconf import DictConfig

MAX_EPISODE_FRAMES = 1000


class FastCarRacing(CarRacing):
    """CarRacing that can skip rendering frames whose pixels nobody looks at.

    Rendering the road polygons in Python is ~90% of the cost of a step. The rendered frame
    only becomes the observation; rewards and terminations come from the physics. So during
    frame skip the intermediate frames are simulated without rendering, which leaves the
    dynamics untouched (``tests/test_env.py`` checks this against the original env).
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.skip_render = False

    def _render(self, mode: str):
        if self.skip_render and mode == "state_pixels":
            return self.state  # stale frame, discarded by FrameSkip
        return super()._render(mode)


class FrameSkip(gym.Wrapper):
    """Repeat the action for ``skip`` frames, summing rewards; stop early if the episode ends."""

    def __init__(self, env: gym.Env, skip: int):
        super().__init__(env)
        if skip < 1:
            raise ValueError("skip must be >= 1")
        self.skip = skip

    def _set_skip_render(self, value: bool) -> None:
        base = self.env.unwrapped
        if isinstance(base, FastCarRacing):
            base.skip_render = value

    def step(self, action):
        total = 0.0
        for i in range(self.skip):
            # Only the last frame of the block is rendered. If the episode ends earlier, the
            # observation is irrelevant to learning (the next one comes from reset).
            self._set_skip_render(i < self.skip - 1)
            obs, reward, terminated, truncated, info = self.env.step(action)
            total += float(reward)
            if terminated or truncated:
                break
        self._set_skip_render(False)
        return obs, total, terminated, truncated, info


def make_env(cfg: DictConfig) -> gym.Env:
    """A single preprocessed CarRacing environment (discrete actions)."""
    env = FastCarRacing(continuous=False, lap_complete_percent=cfg.lap_complete_percent)
    env = TimeLimit(env, max_episode_steps=MAX_EPISODE_FRAMES)
    env = RecordEpisodeStatistics(env)
    env = FrameSkip(env, cfg.frame_skip)
    env = GrayscaleObservation(env, keep_dim=False)
    env = ResizeObservation(env, (cfg.resolution, cfg.resolution))
    env = FrameStackObservation(env, cfg.frame_stack)
    if cfg.reward_clip:
        env = ClipReward(env, -cfg.reward_clip, cfg.reward_clip)
    return env


def make_vector_env(cfg: DictConfig, num_envs: int, asynchronous: bool = True) -> gym.vector.VectorEnv:
    """``num_envs`` copies in subprocesses, auto-resetting in the same step an episode ends.

    With ``SAME_STEP`` autoreset, the observation returned at the end of an episode is already
    the first one of the next episode, and the true final observation is in
    ``info["final_obs"]`` (used to bootstrap the value of truncated episodes).
    """
    fns = [lambda: make_env(cfg) for _ in range(num_envs)]
    kwargs = {"autoreset_mode": gym.vector.AutoresetMode.SAME_STEP}
    if asynchronous:
        return gym.vector.AsyncVectorEnv(fns, **kwargs)
    return gym.vector.SyncVectorEnv(fns, **kwargs)


def observation_shape(cfg: DictConfig) -> tuple[int, int, int]:
    return (cfg.frame_stack, cfg.resolution, cfg.resolution)
