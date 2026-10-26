"""Observation encoders: observation -> feature vector z.

The encoder is shared by every variant and deliberately small, so that a policy cannot
drive well on the encoder alone (the no-core ablation tests exactly that).
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from connectome_driver.models.layers import layer_init


class CNNEncoder(nn.Module):
    """Stacked grayscale frames (uint8) -> z. Three conv layers and one linear projection."""

    def __init__(self, obs_shape: Sequence[int], channels: Sequence[int], feature_dim: int):
        super().__init__()
        c1, c2, c3 = channels
        self.conv = nn.Sequential(
            layer_init(nn.Conv2d(obs_shape[0], c1, kernel_size=8, stride=4)),
            nn.ReLU(),
            layer_init(nn.Conv2d(c1, c2, kernel_size=4, stride=2)),
            nn.ReLU(),
            layer_init(nn.Conv2d(c2, c3, kernel_size=3, stride=1)),
            nn.ReLU(),
            nn.Flatten(),
        )
        with torch.no_grad():
            flat = self.conv(torch.zeros(1, *obs_shape)).shape[1]
        self.proj = nn.Sequential(layer_init(nn.Linear(flat, feature_dim)), nn.ReLU())
        self.feature_dim = feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.proj(self.conv(obs.float() / 255.0))


class FlatEncoder(nn.Module):
    """Vector observation -> z. Used only to sanity-check PPO on simple control tasks."""

    def __init__(self, obs_dim: int, feature_dim: int):
        super().__init__()
        self.net = nn.Sequential(layer_init(nn.Linear(obs_dim, feature_dim)), nn.Tanh())
        self.feature_dim = feature_dim

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs.float())
