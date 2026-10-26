"""Cores map the encoder features z to the states of the readout neurons, (B, R, d).

This is the only part of the policy that differs between the experimental variants.
"""

from __future__ import annotations

from omegaconf import DictConfig
from torch import nn

from connectome_driver.models.cores.mlp import MLPCore


def build_core(cfg: DictConfig, feature_dim: int, readout_neurons: int, node_dim: int) -> nn.Module:
    if cfg.type == "mlp":
        return MLPCore(feature_dim, cfg.hidden_dims, readout_neurons, node_dim)
    raise ValueError(f"unknown core type {cfg.type!r}")
