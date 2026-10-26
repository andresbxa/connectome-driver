"""Actor-critic policy: encoder -> core -> decoder, shared by actor and critic."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from omegaconf import DictConfig
from torch import nn
from torch.distributions import Categorical

from connectome_driver.models.cores import build_core
from connectome_driver.models.decoder import Decoder
from connectome_driver.models.encoder import CNNEncoder, FlatEncoder
from connectome_driver.models.layers import count_parameters


class ActorCritic(nn.Module):
    def __init__(self, encoder: nn.Module, core: nn.Module, decoder: Decoder):
        super().__init__()
        self.encoder, self.core, self.decoder = encoder, core, decoder

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.decoder(self.core(self.encoder(obs)))

    def get_value(self, obs: torch.Tensor) -> torch.Tensor:
        return self(obs)[1]

    def act(self, obs: torch.Tensor, action: torch.Tensor | None = None):
        """Sample (or score) actions. Returns action, log-prob, entropy and value."""
        logits, value = self(obs)
        dist = Categorical(logits=logits)
        if action is None:
            action = dist.sample()
        return action, dist.log_prob(action), dist.entropy(), value

    def parameter_counts(self) -> dict[str, int]:
        return {
            name: count_parameters(part)
            for name, part in (("encoder", self.encoder), ("core", self.core), ("decoder", self.decoder))
        }


def build_policy(cfg: DictConfig, obs_shape: Sequence[int], num_actions: int) -> ActorCritic:
    enc_cfg = cfg.encoder
    if enc_cfg.type == "cnn":
        encoder = CNNEncoder(obs_shape, enc_cfg.channels, enc_cfg.feature_dim)
    elif enc_cfg.type == "flat":
        encoder = FlatEncoder(obs_shape[0], enc_cfg.feature_dim)
    else:
        raise ValueError(f"unknown encoder type {enc_cfg.type!r}")
    core = build_core(cfg.core, encoder.feature_dim, cfg.readout_neurons, cfg.node_dim)
    decoder = Decoder(cfg.readout_neurons, cfg.node_dim, cfg.decoder.hidden_dim, num_actions)
    return ActorCritic(encoder, core, decoder)
