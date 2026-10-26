import numpy as np
import pandas as pd

from connectome_driver.data.graph import Connectome, load_connectome
from connectome_driver.data.preprocess import preprocess, run
from connectome_driver.data.stats import reachability
from connectome_driver.data.subgraph import (
    UNREACHED,
    bfs_distances,
    induced_subgraph,
    path_subgraph_mask,
)


def chain(n: int) -> Connectome:
    """0 -> 1 -> ... -> n-1, plus a side branch 1 -> n."""
    src = np.array(list(range(n - 1)) + [1])
    dst = np.array(list(range(1, n)) + [n])
    return Connectome(
        neurons=pd.DataFrame({"root_id": np.arange(n + 1)}),
        src=src,
        dst=dst,
        syn_count=np.full(len(src), 5, dtype=np.int32),
        neuropil=np.zeros(len(src), dtype=np.int16),
    )


def test_bfs_distances_respect_direction_and_budget():
    g = chain(5)
    adj = g.out_adjacency()
    assert bfs_distances(adj, np.array([0]), 10).tolist() == [0, 1, 2, 3, 4, 2]
    assert bfs_distances(adj, np.array([0]), 2).tolist() == [0, 1, 2, -1, -1, 2]
    backward = bfs_distances(adj.T.tocsr(), np.array([4]), 10)
    assert backward.tolist() == [4, 3, 2, 1, 0, UNREACHED]


def test_path_subgraph_excludes_dead_ends():
    g = chain(5)
    mask = path_subgraph_mask(g, np.array([0]), np.array([4]), k=4)
    assert mask.tolist() == [True, True, True, True, True, False]
    assert not path_subgraph_mask(g, np.array([0]), np.array([4]), k=3).any()


def test_induced_subgraph_reindexes():
    g = chain(5)
    mask = np.array([False, True, True, True, False, True])
    sub = induced_subgraph(g, mask)
    assert sub.neurons["full_index"].tolist() == [1, 2, 3, 5]
    assert sorted(zip(sub.src.tolist(), sub.dst.tolist())) == [(0, 1), (0, 3), (1, 2)]


def test_preprocess_end_to_end(cfg):
    report = run(cfg)
    assert report["visual_afferents"] == 2
    assert report["readout"] == 2

    g = load_connectome(cfg.processed_dir)
    n = g.neurons.set_index("root_id")
    assert n.loc[202, "sign"] == -1  # glutamate is inhibitory by default
    assert n.loc[302, "sign"] == 1  # unknown transmitter
    assert n.loc[203, "sign"] == -1  # GABA, taken from its outgoing connections
    assert n.loc[[201, 203, 302], "nt_source"].tolist() == ["neuron", "edges", "none"]
    assert n["role"].tolist() == [0, 0, 0, 1, 1, 1, 2, 2, 2, 0]

    sources = np.flatnonzero(g.neurons["is_visual_afferent"])
    targets = np.flatnonzero(g.neurons["is_readout"])
    reach = reachability(g, sources, targets, [2, 3])
    assert reach["hop_histogram"] == {3: 1}  # 301 at 3 hops, 302 unreachable from vision
    assert reach["reachable_fraction"] == {2: 0.0, 3: 0.5}

    mask = path_subgraph_mask(g, sources, targets, k=3)
    assert sorted(g.neurons["root_id"][mask]) == [101, 102, 201, 202, 301]


def test_glutamate_override(cfg):
    cfg.signs.glut = 1
    g, _ = preprocess(cfg)
    assert g.neurons.set_index("root_id").loc[202, "sign"] == 1
