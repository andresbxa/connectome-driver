import numpy as np
import pandas as pd
import pytest
import torch
from conftest import EXPECTED_EDGES

from connectome_driver.data import codex
from connectome_driver.data.graph import (
    build_connectome,
    edge_weights,
    load_connectome,
    save_connectome,
    to_torch_csr,
)


@pytest.fixture
def graph(raw_dir):
    neurons = codex.read_neuron_table(raw_dir / "classification.csv.gz", raw_dir / "neurons.csv.gz")
    conns = codex.read_connections(raw_dir / "connections.csv.gz")
    g, report = build_connectome(neurons, conns, min_syn_count=5, drop_autapses=True)
    return g, report


def edge_dict(g):
    ids = g.neurons["root_id"].to_numpy()
    return {(ids[s], ids[d]): int(c) for s, d, c in zip(g.src, g.dst, g.syn_count)}


def test_neuron_table_keeps_missing_neurotransmitter(raw_dir):
    neurons = codex.read_neuron_table(raw_dir / "classification.csv.gz", raw_dir / "neurons.csv.gz")
    assert neurons["root_id"].is_monotonic_increasing
    assert pd.isna(neurons.set_index("root_id").loc[302, "nt_type"])


def test_missing_column_fails_loudly(tmp_path):
    pd.DataFrame({"pre_root_id": [1], "post_root_id": [2]}).to_csv(tmp_path / "c.csv", index=False)
    with pytest.raises(ValueError, match="missing columns"):
        codex.read_connections(tmp_path / "c.csv")


def test_edges_are_aggregated_filtered_and_reported(graph):
    g, report = graph
    assert edge_dict(g) == EXPECTED_EDGES
    assert report["rows_unknown_neuron"] == 1
    assert report["pairs_below_threshold"] == 1
    assert report["autapses"] == 1
    assert report["edges"] == len(EXPECTED_EDGES)


def test_dominant_neuropil(graph):
    g, _ = graph
    ids = g.neurons["root_id"].to_numpy()
    i = np.flatnonzero((ids[g.src] == 101) & (ids[g.dst] == 201))[0]
    assert g.neuropil_names[g.neuropil[i]] == "LO_R"


def test_save_load_roundtrip(graph, tmp_path):
    g, _ = graph
    save_connectome(g, tmp_path, {"note": "test"})
    h = load_connectome(tmp_path)
    assert edge_dict(h) == edge_dict(g)
    assert h.neuropil_names == g.neuropil_names
    pd.testing.assert_frame_equal(h.neurons, g.neurons)


def test_weights_are_log_scaled_normalized_and_follow_dale():
    src = np.array([0, 1, 2, 0])
    dst = np.array([3, 3, 3, 1])
    syn = np.array([5, 10, 20, 7])
    sign = np.array([1, -1, 1, 1], dtype=np.int8)

    raw = edge_weights(syn, src, dst, 4, normalize=False)
    np.testing.assert_allclose(raw, np.log1p(syn), rtol=1e-6)

    w = edge_weights(syn, src, dst, 4, node_sign=sign, normalize=True)
    in_abs = np.bincount(dst, weights=np.abs(w), minlength=4)
    np.testing.assert_allclose(in_abs[[1, 3]], 1.0, rtol=1e-6)
    np.testing.assert_array_equal(np.sign(w), sign[src])


def test_torch_csr_matches_dense():
    rng = np.random.default_rng(0)
    n, e = 30, 120
    keys = rng.choice(n * n, size=e, replace=False)
    src, dst = keys // n, keys % n
    w = rng.normal(size=e).astype(np.float32)

    a, a_t = to_torch_csr(src, dst, w, n)
    dense = np.zeros((n, n), dtype=np.float32)
    dense[dst, src] = w  # row = postsynaptic

    h = torch.randn(n, 5)
    torch.testing.assert_close(a @ h, torch.from_numpy(dense) @ h)
    torch.testing.assert_close(a_t.to_dense(), torch.from_numpy(dense.T))
