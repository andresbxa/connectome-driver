"""Descriptive statistics of a connectome graph, used as sanity checks and to choose T and k."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
import pandas as pd

from connectome_driver.data.graph import Connectome
from connectome_driver.data.partition import ROLE_NAMES
from connectome_driver.data.subgraph import bfs_distances, induced_subgraph, path_subgraph_mask

# Node partition reported by FlyGM (arXiv:2602.17997), used only as a reference point.
FLYGM_COUNTS = {"afferent": 19_262, "intrinsic": 118_496, "efferent": 1_488}


def _summary(x: np.ndarray) -> dict[str, float]:
    q = np.percentile(x, [50, 90, 99]) if len(x) else [np.nan] * 3
    return {
        "mean": float(np.mean(x)) if len(x) else np.nan,
        "median": float(q[0]),
        "p90": float(q[1]),
        "p99": float(q[2]),
        "max": float(np.max(x)) if len(x) else np.nan,
    }


def role_counts(neurons: pd.DataFrame) -> dict[str, int]:
    counts = np.bincount(neurons["role"].to_numpy(), minlength=len(ROLE_NAMES))
    return {name: int(c) for name, c in zip(ROLE_NAMES, counts)}


def super_class_table(neurons: pd.DataFrame) -> pd.DataFrame:
    """Neuron count per super_class with its assigned role (and Codex ``flow`` if present)."""
    cols = ["super_class", "role"] + (["flow"] if "flow" in neurons.columns else [])
    table = neurons[cols].fillna("<NA>").value_counts().reset_index(name="neurons")
    table["role"] = table["role"].map(lambda r: ROLE_NAMES[r])
    return table.sort_values(["role", "neurons"], ascending=[True, False], ignore_index=True)


def degree_stats(graph: Connectome) -> dict[str, dict[str, float] | int]:
    n = graph.num_nodes
    out_deg = np.bincount(graph.src, minlength=n)
    in_deg = np.bincount(graph.dst, minlength=n)
    return {
        "in_degree": _summary(in_deg),
        "out_degree": _summary(out_deg),
        "isolated_nodes": int(np.sum((in_deg == 0) & (out_deg == 0))),
        "no_inputs": int(np.sum(in_deg == 0)),
        "no_outputs": int(np.sum(out_deg == 0)),
    }


def sign_stats(graph: Connectome) -> dict[str, float]:
    sign = graph.neurons["sign"].to_numpy()
    edge_sign = sign[graph.src]
    return {
        "inhibitory_neuron_fraction": float(np.mean(sign < 0)),
        "inhibitory_edge_fraction": float(np.mean(edge_sign < 0)),
        "inhibitory_synapse_fraction": float(
            graph.syn_count[edge_sign < 0].sum() / graph.syn_count.sum()
        ),
    }


def reachability(
    graph: Connectome, sources: np.ndarray, targets: np.ndarray, hop_budgets: Sequence[int]
) -> dict:
    """How far the targets are from the nearest source, and how many fit in each hop budget."""
    dist = bfs_distances(graph.out_adjacency(), sources, max(hop_budgets))
    d_t = dist[targets]
    reached = d_t[d_t >= 0]
    return {
        "targets": len(targets),
        "hop_histogram": {int(h): int(c) for h, c in zip(*np.unique(reached, return_counts=True))},
        "reachable_fraction": {
            int(b): float(np.mean((d_t >= 0) & (d_t <= b))) for b in hop_budgets
        },
    }


def subgraph_sizes(
    graph: Connectome, sources: np.ndarray, targets: np.ndarray, ks: Sequence[int]
) -> list[dict]:
    rows = []
    for k in ks:
        mask = path_subgraph_mask(graph, sources, targets, k)
        sub = induced_subgraph(graph, mask)
        rows.append(
            {
                "k": int(k),
                "nodes": sub.num_nodes,
                "edges": sub.num_edges,
                "sources_kept": int(mask[sources].sum()),
                "targets_kept": int(mask[targets].sum()),
            }
        )
    return rows
