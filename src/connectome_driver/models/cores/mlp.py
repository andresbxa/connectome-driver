"""Cores without graph structure: an MLP, and a single linear map (the no-core ablation)."""

from __future__ import annotations

import itertools
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


def mlp_param_count(in_dim: int, out_dim: int, hidden_dims: Sequence[int]) -> int:
    dims = [in_dim, *hidden_dims, out_dim]
    return sum(a * b + b for a, b in itertools.pairwise(dims))


def matched_hidden_width(in_dim: int, out_dim: int, depth: int, target: int) -> int:
    """Width of ``depth`` equal hidden layers whose MLP has the parameter count closest to target."""
    best, best_gap = 1, None
    for width in range(1, 4097):
        gap = abs(mlp_param_count(in_dim, out_dim, [width] * depth) - target)
        if best_gap is None or gap < best_gap:
            best, best_gap = width, gap
        elif gap > best_gap:  # the count grows with width: past the optimum
            break
    return best


class LinearCore(nn.Module):
    """The no-core ablation: one linear map from z to the readout shape, no nonlinearity.

    It exists only so the decoder keeps exactly the same input shape. Composed with the
    decoder's first linear layer it is just a reparametrized linear layer, so the policy is
    effectively encoder -> decoder.
    """

    def __init__(self, in_dim: int, readout_neurons: int, node_dim: int):
        super().__init__()
        self.proj = layer_init(nn.Linear(in_dim, readout_neurons * node_dim), gain=1.0)
        self.readout_shape = (readout_neurons, node_dim)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        return self.proj(z).view(z.shape[0], *self.readout_shape)
