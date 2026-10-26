"""Decoder: readout-neuron states -> action logits and state value."""

from __future__ import annotations

import torch
from torch import nn

from connectome_driver.models.layers import layer_init


class Decoder(nn.Module):
    """Flatten the (R, d) readout states, one hidden layer, then policy and value heads.

    Flattening keeps the identity of each readout neuron: the decoder can learn that a given
    descending neuron means "steer left", as the fly's motor system does.
    """

    def __init__(self, readout_neurons: int, node_dim: int, hidden_dim: int, num_actions: int):
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Flatten(),
            layer_init(nn.Linear(readout_neurons * node_dim, hidden_dim)),
            nn.ReLU(),
        )
        self.policy = layer_init(nn.Linear(hidden_dim, num_actions), gain=0.01)
        self.value = layer_init(nn.Linear(hidden_dim, 1), gain=1.0)

    def forward(self, readout: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(readout)
        return self.policy(h), self.value(h).squeeze(-1)
