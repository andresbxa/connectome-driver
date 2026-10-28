"""Null-model graphs: the same neurons with the edges rewired or redrawn.

Every null returns a new ``Connectome`` with the *same* neuron table (so roles, signs, inputs
and readouts are unchanged) and new edge arrays. Signs are a property of the presynaptic neuron,
so an edge that moves takes the sign of its new source automatically (Dale's law).

- ``rewire_degree_preserving``: directed double edge swaps ``a->b, c->d  =>  a->d, c->b``. Each
  edge keeps its source, synapse count and neuropil and only its target changes, so every
  neuron keeps its in-degree, its out-degree and the total synapses it sends.
- ``rewire_within_blocks``: the same, but only between edges joining the same pair of blocks,
  which also keeps the number of edges between every pair of blocks.
- ``random_graph``: E distinct ordered pairs drawn uniformly, with the real synapse counts
  shuffled over them.

Swaps are vectorized: in each round edges are paired at random (within their block pair), all
proposals are checked at once and the valid ones are applied together. A proposal is rejected
if it would create a self-loop or an edge that already exists, or if one of its new edges
collides with another proposal of the same round.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from connectome_driver.data.graph import Connectome

MAX_IDLE_ROUNDS = 50  # stop if this many consecutive rounds accept nothing (saturated graph)


def _contains(sorted_keys: np.ndarray, query: np.ndarray) -> np.ndarray:
    idx = np.minimum(np.searchsorted(sorted_keys, query), len(sorted_keys) - 1)
    return sorted_keys[idx] == query


def _double_edge_swaps(
    src: np.ndarray,
    dst: np.ndarray,
    num_nodes: int,
    num_swaps: int,
    rng: np.random.Generator,
    groups: np.ndarray | None = None,
) -> tuple[np.ndarray, dict]:
    """New targets after ``num_swaps`` accepted swaps between edges of the same group."""
    original, dst = dst, dst.copy()
    num_edges = len(src)
    groups = np.zeros(num_edges, np.int64) if groups is None else groups
    keys = np.sort(src * num_nodes + dst)
    accepted = proposed = rounds = idle = 0
    while accepted < num_swaps and idle < MAX_IDLE_ROUNDS and num_edges > 1:
        rounds += 1
        # Random order within each group; consecutive edges of the same group form a proposal,
        # so an edge is in at most one proposal per round. The random offset changes which
        # edge is left out of odd-sized groups.
        order = np.lexsort((rng.random(num_edges), groups))[int(rng.integers(2)):]
        pairs = len(order) // 2
        i, j = order[0 : 2 * pairs : 2], order[1 : 2 * pairs : 2]
        same = groups[i] == groups[j]
        i, j = i[same], j[same]
        proposed += len(i)

        new_i, new_j = src[i] * num_nodes + dst[j], src[j] * num_nodes + dst[i]
        ok = (src[i] != dst[j]) & (src[j] != dst[i]) & (dst[i] != dst[j])
        ok &= ~_contains(keys, new_i) & ~_contains(keys, new_j)
        candidates = np.flatnonzero(ok)
        new_keys = np.concatenate([new_i[candidates], new_j[candidates]])
        _, inverse, counts = np.unique(new_keys, return_inverse=True, return_counts=True)
        duplicated = counts[inverse] > 1
        clash = duplicated[: len(candidates)] | duplicated[len(candidates) :]
        winners = candidates[~clash][: num_swaps - accepted]
        if len(winners) == 0:
            idle += 1
            continue
        idle = 0
        a, b = i[winners], j[winners]
        dst[a], dst[b] = dst[b], dst[a].copy()
        accepted += len(winners)
        keys = np.sort(src * num_nodes + dst)

    report = {
        "swaps_requested": int(num_swaps),
        "swaps_accepted": int(accepted),
        "proposals": int(proposed),
        "acceptance_rate": accepted / proposed if proposed else 0.0,
        "rounds": rounds,
        "rewired_fraction": float(np.mean(dst != original)) if num_edges else 0.0,
    }
    return dst, report


def _with_edges(graph: Connectome, src, dst, syn_count, neuropil) -> Connectome:
    return Connectome(graph.neurons, src, dst, syn_count, neuropil, list(graph.neuropil_names))


def rewire_degree_preserving(
    graph: Connectome, seed: int, swaps_per_edge: float = 10.0, report: dict | None = None
) -> Connectome:
    """Degree- and out-strength-preserving null (directed double edge swap).

    ``report``, if given, is filled with the acceptance rate and the fraction of edges whose
    target changed.
    """
    num_swaps = round(swaps_per_edge * graph.num_edges)
    dst, info = _double_edge_swaps(graph.src, graph.dst, graph.num_nodes, num_swaps, np.random.default_rng(seed))
    if report is not None:
        report.update(info)
    return _with_edges(graph, graph.src.copy(), dst, graph.syn_count.copy(), graph.neuropil.copy())


def rewire_within_blocks(
    graph: Connectome,
    blocks: np.ndarray,
    seed: int,
    swaps_per_edge: float = 10.0,
    report: dict | None = None,
) -> Connectome:
    """As ``rewire_degree_preserving``, swapping only edges with the same (source block,
    target block) pair, so the number of edges between every pair of blocks is kept."""
    if len(blocks) != graph.num_nodes:
        raise ValueError(f"blocks has {len(blocks)} entries for {graph.num_nodes} neurons")
    codes = np.unique(blocks, return_inverse=True)[1].astype(np.int64)
    groups = codes[graph.src] * (codes.max() + 1) + codes[graph.dst]
    num_swaps = round(swaps_per_edge * graph.num_edges)
    rng = np.random.default_rng(seed)
    dst, info = _double_edge_swaps(graph.src, graph.dst, graph.num_nodes, num_swaps, rng, groups)
    if report is not None:
        report.update(info, block_pairs=len(np.unique(groups)))
    return _with_edges(graph, graph.src.copy(), dst, graph.syn_count.copy(), graph.neuropil.copy())


def random_graph(graph: Connectome, seed: int, report: dict | None = None) -> Connectome:
    """Same N and E; edges drawn uniformly among ordered pairs ``i != j`` without duplicates.

    The synapse counts (with their neuropils) are a random permutation of the real ones, so the
    weight distribution is kept and each edge's sign comes from its new presynaptic neuron.
    """
    rng = np.random.default_rng(seed)
    n, e = graph.num_nodes, graph.num_edges
    pairs = n * (n - 1)
    if e > pairs:
        raise ValueError(f"{e} edges do not fit in a simple directed graph on {n} nodes")
    keys = np.empty(0, np.int64)
    while len(keys) < e:  # rejection of duplicates; a few rounds for sparse graphs
        draw = rng.integers(0, pairs, size=int(1.05 * (e - len(keys))) + 16)
        keys = np.unique(np.concatenate([keys, draw]))
    keys = np.sort(rng.choice(keys, size=e, replace=False))  # uniform subset of the distinct draws
    src = keys // (n - 1)
    rest = keys % (n - 1)
    dst = rest + (rest >= src)  # skip the diagonal
    perm = rng.permutation(e)
    if report is not None:
        real = np.sort(graph.src * n + graph.dst)
        report.update(new_edge_fraction=float(1 - np.mean(_contains(real, keys))))
    return _with_edges(graph, src.astype(np.int64), dst.astype(np.int64), graph.syn_count[perm], graph.neuropil[perm])


def neuron_blocks(graph: Connectome, by: str) -> np.ndarray:
    """Block of every neuron, for ``rewire_within_blocks``.

    - ``"super_class"``: one block per value of ``neurons["super_class"]`` (missing: -1);
    - ``"neuropil"``: the neuron's dominant output neuropil, weighted by synapse count; neurons
      without outgoing edges get block -1.
    """
    if by == "super_class":
        codes, _ = pd.factorize(graph.neurons["super_class"])
        return codes.astype(np.int64)
    if by == "neuropil":
        n = graph.num_nodes
        k = max(len(graph.neuropil_names), int(graph.neuropil.max(initial=-1)) + 1)
        flat = graph.src * k + graph.neuropil.astype(np.int64)
        weight = np.bincount(flat, weights=graph.syn_count, minlength=n * k).reshape(n, k)
        blocks = weight.argmax(axis=1).astype(np.int64)
        blocks[weight.sum(axis=1) == 0] = -1
        return blocks
    raise ValueError(f"unknown block definition {by!r}; expected 'super_class' or 'neuropil'")
