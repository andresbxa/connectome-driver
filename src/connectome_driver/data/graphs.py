"""The graph a policy runs on: which variant, whole brain or path subgraph, and its key indices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

from connectome_driver.data.graph import Connectome, load_connectome
from connectome_driver.data.subgraph import induced_subgraph, path_subgraph_mask


@dataclass
class GraphSpec:
    graph: Connectome
    input_index: np.ndarray  # visual afferents: where the encoder drive enters
    readout_index: np.ndarray  # descending neurons: what the decoder reads


def build_graph(cfg: DictConfig) -> GraphSpec:
    """Load the processed connectome, optionally restrict it, and apply the graph variant.

    The subgraph is cut from the real connectome *before* any null model is applied, so every
    variant has exactly the same neurons, inputs and readouts; only the edges differ.
    """
    graph = load_connectome(cfg.processed_dir)
    if cfg.subgraph_k is not None:
        n = graph.neurons
        mask = path_subgraph_mask(
            graph,
            np.flatnonzero(n["is_visual_afferent"].to_numpy()),
            np.flatnonzero(n["is_readout"].to_numpy()),
            cfg.subgraph_k,
        )
        graph = induced_subgraph(graph, mask)
    if cfg.variant != "connectome":
        raise ValueError(f"graph variant {cfg.variant!r} is not implemented yet (Phase 4)")
    return spec_from_graph(graph)


def spec_from_graph(graph: Connectome) -> GraphSpec:
    n = graph.neurons
    return GraphSpec(
        graph=graph,
        input_index=np.flatnonzero(n["is_visual_afferent"].to_numpy()),
        readout_index=np.flatnonzero(n["is_readout"].to_numpy()),
    )
