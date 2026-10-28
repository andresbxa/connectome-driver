"""The graph a policy runs on: which variant, whole brain or path subgraph, and its key indices."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from omegaconf import DictConfig

from connectome_driver.data.graph import Connectome, load_connectome
from connectome_driver.data.subgraph import induced_subgraph, path_subgraph_mask

VARIANTS = ("connectome", "rewired", "random", "block")


@dataclass
class GraphSpec:
    graph: Connectome
    input_index: np.ndarray  # visual afferents: where the encoder drive enters
    readout_index: np.ndarray  # descending neurons: what the decoder reads


def build_graph(cfg: DictConfig, seed: int = 0) -> GraphSpec:
    """Load the processed connectome, optionally restrict it, and apply the graph variant.

    The subgraph is cut from the real connectome *before* any null model is applied, so every
    variant has exactly the same neurons, inputs and readouts; only the edges differ. Null
    models are sampled with ``cfg.variant_seed`` if set, otherwise with ``seed`` (the run's
    seed), so each training seed sees its own sample of the null ensemble.
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
    variant_seed = cfg.get("variant_seed") if cfg.get("variant_seed") is not None else seed
    return spec_from_graph(apply_variant(graph, cfg.variant, variant_seed, cfg))


def apply_variant(graph: Connectome, variant: str, seed: int, cfg: DictConfig) -> Connectome:
    if variant == "connectome":
        return graph
    if variant not in VARIANTS:
        raise ValueError(f"unknown graph variant {variant!r}; expected one of {VARIANTS}")
    try:
        from connectome_driver.data import nulls
    except ImportError as err:  # specified in docs/tasks/null-models.md
        raise NotImplementedError(f"graph variant {variant!r} needs data/nulls.py") from err
    swaps = cfg.get("swaps_per_edge", 10.0)
    if variant == "rewired":
        return nulls.rewire_degree_preserving(graph, seed=seed, swaps_per_edge=swaps)
    if variant == "random":
        return nulls.random_graph(graph, seed=seed)
    blocks = nulls.neuron_blocks(graph, by=cfg.get("blocks", "super_class"))
    return nulls.rewire_within_blocks(graph, blocks, seed=seed, swaps_per_edge=swaps)


def spec_from_graph(graph: Connectome) -> GraphSpec:
    n = graph.neurons
    return GraphSpec(
        graph=graph,
        input_index=np.flatnonzero(n["is_visual_afferent"].to_numpy()),
        readout_index=np.flatnonzero(n["is_readout"].to_numpy()),
    )
