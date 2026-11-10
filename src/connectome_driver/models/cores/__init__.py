"""Cores map the encoder features z to the states of the readout neurons, (B, R, d).

This is the only part of the policy that differs between the experimental variants:

- ``graph``: message passing over the connectome or a null model of it (``GraphCore``);
- ``mlp``: fully connected layers; ``hidden_dims: match_graph`` sizes them to the same number
  of parameters as the graph core on the experiment's graph;
- ``none``: no core, a single linear map from z to the readout shape (the ablation).
"""

from __future__ import annotations

from omegaconf import DictConfig
from torch import nn

from connectome_driver.data.graphs import GraphSpec
from connectome_driver.models.cores.graph import GraphCore, graph_core_param_count
from connectome_driver.models.cores.mlp import LinearCore, MLPCore, matched_hidden_width


def build_core(
    cfg: DictConfig,
    feature_dim: int,
    readout_neurons: int,
    node_dim: int,
    graph: GraphSpec | None = None,
) -> nn.Module:
    if cfg.type == "mlp":
        hidden = cfg.hidden_dims
        if hidden == "match_graph":
            if graph is None:
                raise ValueError("hidden_dims: match_graph needs the experiment graph")
            target = graph_core_param_count(
                graph.graph.num_nodes, len(graph.input_index), feature_dim, node_dim
            )
            width = matched_hidden_width(feature_dim, readout_neurons * node_dim, cfg.depth, target)
            hidden = [width] * cfg.depth
        return MLPCore(feature_dim, hidden, readout_neurons, node_dim)
    if cfg.type == "none":
        return LinearCore(feature_dim, readout_neurons, node_dim)
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
            normalize=cfg.get("normalize", "l2"),
            leak_init=cfg.get("leak_init", 0.9),
            dtype=cfg.dtype,
            checkpoint_steps=cfg.checkpoint,
            sparse_backend=cfg.get("sparse_backend", "torch"),
        )
    raise ValueError(f"unknown core type {cfg.type!r}")
