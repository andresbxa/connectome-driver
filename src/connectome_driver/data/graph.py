"""The connectome as a directed graph: edge list on disk, sparse CSR adjacency in memory.

Nodes are indexed 0..N-1 in the order of the neuron table (sorted by root_id). Edges are kept
as raw synapse counts; edge weights are derived only when the adjacency is built, so the null
models (rewired / random graphs) can reuse the same weighting code.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.sparse as sp
import torch


@dataclass
class Connectome:
    neurons: pd.DataFrame  # one row per node, row i is node i
    src: np.ndarray  # (E,) int64, presynaptic node
    dst: np.ndarray  # (E,) int64, postsynaptic node
    syn_count: np.ndarray  # (E,) int32, synapses summed over neuropils
    neuropil: np.ndarray  # (E,) int16, neuropil holding most of the pair's synapses
    neuropil_names: list[str] = field(default_factory=list)

    @property
    def num_nodes(self) -> int:
        return len(self.neurons)

    @property
    def num_edges(self) -> int:
        return len(self.src)

    def out_adjacency(self) -> sp.csr_matrix:
        """Unweighted adjacency with rows = presynaptic, columns = postsynaptic."""
        ones = np.ones(self.num_edges, dtype=np.int8)
        n = self.num_nodes
        return sp.csr_matrix((ones, (self.src, self.dst)), shape=(n, n))


def aggregate_edges(
    connections: pd.DataFrame,
    root_ids: np.ndarray,
    min_syn_count: int,
    drop_autapses: bool,
) -> tuple[dict[str, np.ndarray], list[str], dict[str, int]]:
    """Collapse per-neuropil rows into one edge per (pre, post) neuron pair.

    Returns the edge arrays, the neuropil vocabulary and a report of what was dropped.
    ``root_ids`` must be sorted; it defines the node index of every neuron.
    """
    report = {"rows_in": len(connections)}

    pre = np.searchsorted(root_ids, connections["pre_root_id"].to_numpy())
    post = np.searchsorted(root_ids, connections["post_root_id"].to_numpy())
    pre = np.minimum(pre, len(root_ids) - 1)
    post = np.minimum(post, len(root_ids) - 1)
    known = (root_ids[pre] == connections["pre_root_id"].to_numpy()) & (
        root_ids[post] == connections["post_root_id"].to_numpy()
    )
    report["rows_unknown_neuron"] = int((~known).sum())

    neuropil = connections["neuropil"].astype("category")
    neuropil_names = [str(c) for c in neuropil.cat.categories]
    rows = pd.DataFrame(
        {
            "src": pre[known],
            "dst": post[known],
            "syn": connections["syn_count"].to_numpy()[known],
            "np": neuropil.cat.codes.to_numpy()[known],
        }
    )

    # Dominant neuropil of a pair: the row with the most synapses (first one on ties).
    rows = rows.sort_values(["src", "dst", "syn"], ascending=[True, True, False], kind="stable")
    grouped = rows.groupby(["src", "dst"], sort=True)
    edges = grouped.agg(syn=("syn", "sum"), np=("np", "first")).reset_index()
    report["pairs"] = len(edges)

    keep = edges["syn"].to_numpy() >= min_syn_count
    report["pairs_below_threshold"] = int((~keep).sum())
    autapse = edges["src"].to_numpy() == edges["dst"].to_numpy()
    report["autapses"] = int((autapse & keep).sum())
    if drop_autapses:
        keep &= ~autapse
    edges = edges[keep]
    report["edges"] = len(edges)

    arrays = {
        "src": edges["src"].to_numpy(np.int64),
        "dst": edges["dst"].to_numpy(np.int64),
        "syn_count": edges["syn"].to_numpy(np.int32),
        "neuropil": edges["np"].to_numpy(np.int16),
    }
    return arrays, neuropil_names, report


def build_connectome(
    neurons: pd.DataFrame,
    connections: pd.DataFrame,
    min_syn_count: int = 5,
    drop_autapses: bool = True,
) -> tuple[Connectome, dict[str, int]]:
    """Graph over the neurons of ``neurons`` (which must be sorted by root_id)."""
    root_ids = neurons["root_id"].to_numpy()
    if not np.all(root_ids[1:] > root_ids[:-1]):
        raise ValueError("neuron table must be sorted by unique root_id")
    arrays, names, report = aggregate_edges(connections, root_ids, min_syn_count, drop_autapses)
    return Connectome(neurons.reset_index(drop=True), neuropil_names=names, **arrays), report


# --------------------------------------------------------------------------- persistence

NEURONS_FILE = "neurons.parquet"
EDGES_FILE = "edges.npz"
META_FILE = "meta.json"


def save_connectome(graph: Connectome, out_dir: str | Path, meta: dict | None = None) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    graph.neurons.to_parquet(out / NEURONS_FILE, index=False)
    np.savez_compressed(
        out / EDGES_FILE,
        src=graph.src,
        dst=graph.dst,
        syn_count=graph.syn_count,
        neuropil=graph.neuropil,
        neuropil_names=np.array(graph.neuropil_names, dtype=str),
    )
    meta = dict(meta or {}, num_nodes=graph.num_nodes, num_edges=graph.num_edges)
    (out / META_FILE).write_text(json.dumps(meta, indent=2))


def load_connectome(processed_dir: str | Path) -> Connectome:
    d = Path(processed_dir)
    neurons = pd.read_parquet(d / NEURONS_FILE)
    with np.load(d / EDGES_FILE) as z:
        return Connectome(
            neurons=neurons,
            src=z["src"],
            dst=z["dst"],
            syn_count=z["syn_count"],
            neuropil=z["neuropil"],
            neuropil_names=[str(s) for s in z["neuropil_names"]],
        )


# --------------------------------------------------------------------------- adjacency


def edge_weights(
    syn_count: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    num_nodes: int,
    node_sign: np.ndarray | None = None,
    normalize: str | None = "l2",
) -> np.ndarray:
    """Weight of each edge: ``log(1 + syn_count)``, optionally signed and normalized.

    Normalization is per postsynaptic neuron, over its incoming weights:

    - ``"l2"``: squared weights sum to 1. For independent inputs of equal variance the summed
      input keeps that variance, so a signal neither explodes nor fades from hop to hop.
    - ``"l1"``: weights sum to 1, i.e. a weighted *average* of the inputs. A signal carried by
      a few of many inputs is diluted at every hop (about 10x per hop in the connectome).
    - ``None``: raw log synapse counts.

    The sign comes from the presynaptic neuron (``node_sign[src]``) so all outgoing edges of a
    neuron share it.
    """
    w = np.log1p(syn_count.astype(np.float64))
    if normalize == "l1":
        w = w / np.bincount(dst, weights=w, minlength=num_nodes)[dst]
    elif normalize == "l2":
        w = w / np.sqrt(np.bincount(dst, weights=w**2, minlength=num_nodes))[dst]
    elif normalize is not None:
        raise ValueError(f"unknown normalization {normalize!r}")
    if node_sign is not None:
        w = w * node_sign[src]
    return w.astype(np.float32)


def to_torch_csr(
    src: np.ndarray, dst: np.ndarray, weight: np.ndarray, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sparse adjacency ``A`` (rows = postsynaptic) and its transpose, both CSR.

    ``A @ H`` sums the weighted states of each node's presynaptic partners. ``A.T`` is
    precomputed (not obtained by transposing at runtime) for the custom SpMM backward.
    """

    def csr(rows: np.ndarray, cols: np.ndarray) -> torch.Tensor:
        m = sp.csr_matrix((weight, (rows, cols)), shape=(num_nodes, num_nodes))
        m.sort_indices()
        return torch.sparse_csr_tensor(
            torch.from_numpy(m.indptr.astype(np.int64)),
            torch.from_numpy(m.indices.astype(np.int64)),
            torch.from_numpy(m.data.astype(np.float32)),
            size=(num_nodes, num_nodes),
            check_invariants=True,  # one-off cost at build time, never in the training loop
        )

    return csr(dst, src), csr(src, dst)
