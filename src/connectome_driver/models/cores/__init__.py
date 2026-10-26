"""Cores map the encoder features z to the states of the readout neurons, (B, R, d).

This is the only part of the policy that differs between the experimental variants.
"""

from __future__ import annotations

from omegaconf import DictConfig
from torch import nn

from connectome_driver.data.graphs import GraphSpec
from connectome_driver.models.cores.graph import GraphCore
from connectome_driver.models.cores.mlp import MLPCore


def build_core(
    cfg: DictConfig,
    feature_dim: int,
    readout_neurons: int,
    node_dim: int,
    graph: GraphSpec | None = None,
) -> nn.Module:
    if cfg.type == "mlp":
        return MLPCore(feature_dim, cfg.hidden_dims, readout_neurons, node_dim)
    if cfg.type == "graph":
        if graph is None:
            raise ValueError("a graph core needs a graph")
        g = graph.graph
        return GraphCore(
            num_nodes=g.num_nodes,
            src=g.src,
            dst=g.dst,
            syn_count=g.syn_count,
            node_sign=g.neurons["sign"].to_numpy(),
            input_index=graph.input_index,
            readout_index=graph.readout_index,
            feature_dim=feature_dim,
            node_dim=node_dim,
            steps=cfg.steps,
            signed=cfg.signed,
            dtype=cfg.dtype,
            checkpoint_steps=cfg.checkpoint,
        )
    raise ValueError(f"unknown core type {cfg.type!r}")
