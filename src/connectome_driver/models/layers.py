"""Small building blocks shared by the model parts."""

from __future__ import annotations

import math

from torch import nn


def layer_init(layer: nn.Module, gain: float = math.sqrt(2), bias: float = 0.0) -> nn.Module:
    """Orthogonal weights and constant bias, the usual initialization for PPO networks."""
    nn.init.orthogonal_(layer.weight, gain)
    nn.init.constant_(layer.bias, bias)
    return layer


def count_parameters(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)
