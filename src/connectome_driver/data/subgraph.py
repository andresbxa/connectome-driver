"""Hop distances and the path subgraph between two neuron sets.

The path subgraph keeps every neuron that lies on some directed path of length <= k from a
source set (visual afferents) to a target set (descending neurons). It is the neural
territory the core can actually use within k message-passing steps, which makes it a cheap
stand-in for the whole brain while iterating.
"""

from __future__ import annotations

import numpy as np
import scipy.sparse as sp

from connectome_driver.data.graph import Connectome

UNREACHED = -1


def bfs_distances(adj: sp.csr_matrix, sources: np.ndarray, max_hops: int) -> np.ndarray:
    """Hop distance from the nearest source to every node (``UNREACHED`` beyond max_hops).

    ``adj`` has rows = from, columns = to. Pass ``adj.T.tocsr()`` to walk edges backwards.
    """
    n = adj.shape[0]
    dist = np.full(n, UNREACHED, dtype=np.int32)
    frontier = np.unique(np.asarray(sources, dtype=np.int64))
    dist[frontier] = 0
    for hop in range(1, max_hops + 1):
        if frontier.size == 0:
            break
        neighbours = np.unique(adj[frontier].indices)
        frontier = neighbours[dist[neighbours] == UNREACHED]
        dist[frontier] = hop
    return dist


def path_subgraph_mask(
    graph: Connectome, sources: np.ndarray, targets: np.ndarray, k: int
) -> np.ndarray:
    """Nodes on a directed source -> target path of at most ``k`` hops."""
    adj = graph.out_adjacency()
    forward = bfs_distances(adj, sources, k)
    backward = bfs_distances(adj.T.tocsr(), targets, k)
    return (forward >= 0) & (backward >= 0) & (forward + backward <= k)


def induced_subgraph(graph: Connectome, mask: np.ndarray) -> Connectome:
    """Keep the nodes in ``mask`` and the edges between them, reindexed to 0..n-1.

    The new neuron table gains a ``full_index`` column pointing back to the full graph.
    """
    new_index = np.full(graph.num_nodes, -1, dtype=np.int64)
    new_index[mask] = np.arange(int(mask.sum()))
    keep = mask[graph.src] & mask[graph.dst]
    neurons = graph.neurons[mask].copy()
    neurons.insert(0, "full_index", np.flatnonzero(mask))
    return Connectome(
        neurons=neurons.reset_index(drop=True),
        src=new_index[graph.src[keep]],
        dst=new_index[graph.dst[keep]],
        syn_count=graph.syn_count[keep],
        neuropil=graph.neuropil[keep],
        neuropil_names=list(graph.neuropil_names),
    )
