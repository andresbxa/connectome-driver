"""MLP core: z -> readout-neuron states, with no graph structure at all."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from connectome_driver.models.layers import layer_init


class MLPCore(nn.Module):
    """Fully connected layers from z to ``readout_neurons x node_dim`` states.

    Its output has the same shape as the readout of the graph cores, so the decoder that
    follows is identical for every variant.
    """

    def __init__(
        self, in_dim: int, hidden_dims: Sequence[int], readout_neurons: int, node_dim: int
    ):
        super().__init__()
        layers: list[nn.Module] = []
        prev = in_dim
        for h in hidden_dims:
            layers += [layer_init(nn.Linear(prev, h)), nn.Tanh()]
            prev = h
        layers += [layer_init(nn.Linear(prev, readout_neurons * node_dim)), nn.Tanh()]
        self.net = nn.Sequential(*layers)
        self.readout_shape = (readout_neurons, node_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.net(z).view(z.shape[0], *self.readout_shape)
