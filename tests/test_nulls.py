import numpy as np
import pandas as pd
import pytest
from test_graph_core import toy_spec  # noqa: F401  (fixture)

from connectome_driver.data.graph import Connectome
from connectome_driver.data.nulls import (
    neuron_blocks,
    random_graph,
    rewire_degree_preserving,
    rewire_within_blocks,
)


def random_connectome(n: int = 300, e: int = 3000, seed: int = 0) -> Connectome:
    """A random simple directed graph with a skewed degree sequence and 4 super_classes."""
    rng = np.random.default_rng(seed)
    p = rng.pareto(1.5, n) + 1
    keys = set()
    while len(keys) < e:
        a, b = rng.choice(n, 2, p=p / p.sum()), rng.integers(n)
        if a[0] != b:
            keys.add((int(a[0]), int(b)))
    src, dst = np.array(sorted(keys)).T
    neurons = pd.DataFrame({"super_class": rng.choice(["optic", "central", "visual", "descending"], n)})
    neurons["sign"] = rng.choice([-1, 1], n)
    return Connectome(
        neurons, src.astype(np.int64), dst.astype(np.int64),
        rng.integers(5, 200, e).astype(np.int32), rng.integers(0, 6, e).astype(np.int16),
        [f"np{i}" for i in range(6)],
    )


def assert_simple(g: Connectome) -> None:
    assert not np.any(g.src == g.dst), "self-loop"
    keys = g.src * g.num_nodes + g.dst
    assert len(np.unique(keys)) == g.num_edges, "duplicate edge"


def degrees(g: Connectome) -> tuple[np.ndarray, np.ndarray]:
    return np.bincount(g.src, minlength=g.num_nodes), np.bincount(g.dst, minlength=g.num_nodes)


def edge_set(g: Connectome) -> set[tuple[int, int]]:
    return set(zip(g.src.tolist(), g.dst.tolist(), strict=True))


@pytest.fixture
def graph() -> Connectome:
    return random_connectome()


def test_rewire_keeps_degrees_and_out_strength(graph):
    report = {}
    null = rewire_degree_preserving(graph, seed=1, swaps_per_edge=10, report=report)
    assert_simple(null)
    assert null.num_edges == graph.num_edges
    for a, b in zip(degrees(null), degrees(graph), strict=True):
        np.testing.assert_array_equal(a, b)
    np.testing.assert_array_equal(np.sort(null.syn_count), np.sort(graph.syn_count))
    strength = lambda g: np.bincount(g.src, weights=g.syn_count, minlength=g.num_nodes)
    np.testing.assert_array_equal(strength(null), strength(graph))
    assert report["swaps_accepted"] == report["swaps_requested"] == 10 * graph.num_edges
    assert report["rewired_fraction"] > 0.9  # it actually mixes


def test_block_rewire_keeps_degrees_and_block_matrix(graph):
    blocks = neuron_blocks(graph, by="super_class")
    null = rewire_within_blocks(graph, blocks, seed=1, swaps_per_edge=10)
    assert_simple(null)
    for a, b in zip(degrees(null), degrees(graph), strict=True):
        np.testing.assert_array_equal(a, b)
    k = blocks.max() + 1
    block_matrix = lambda g: np.bincount(blocks[g.src] * k + blocks[g.dst], minlength=k * k)
    np.testing.assert_array_equal(block_matrix(null), block_matrix(graph))
    assert len(edge_set(null) - edge_set(graph)) > 0.5 * graph.num_edges


def test_random_graph_keeps_size_and_weights(graph):
    null = random_graph(graph, seed=1)
    assert_simple(null)
    assert (null.num_nodes, null.num_edges) == (graph.num_nodes, graph.num_edges)
    np.testing.assert_array_equal(np.sort(null.syn_count), np.sort(graph.syn_count))
    np.testing.assert_array_equal(np.sort(null.neuropil), np.sort(graph.neuropil))
    # Synapse counts travel with their neuropil (the pairs are a permutation of the real pairs).
    pairs = lambda g: sorted(zip(g.syn_count.tolist(), g.neuropil.tolist(), strict=True))
    assert pairs(null) == pairs(graph)


@pytest.mark.parametrize("nodes, edges", [(3, 6), (20, 80)])
def test_random_graph_reports_actual_new_edge_fraction(nodes, edges):
    graph = random_connectome(n=nodes, e=edges)
    report = {}
    null = random_graph(graph, seed=1, report=report)
    expected = len(edge_set(null) - edge_set(graph)) / graph.num_edges
    assert report["new_edge_fraction"] == pytest.approx(expected)


@pytest.mark.parametrize("make", [
    lambda g, s: rewire_degree_preserving(g, seed=s, swaps_per_edge=2),
    lambda g, s: random_graph(g, seed=s),
    lambda g, s: rewire_within_blocks(g, neuron_blocks(g, "super_class"), seed=s, swaps_per_edge=2),
])
def test_nulls_are_deterministic_and_keep_neurons(graph, make):
    a, b, c = make(graph, 0), make(graph, 0), make(graph, 1)
    assert edge_set(a) == edge_set(b)
    np.testing.assert_array_equal(a.syn_count, b.syn_count)
    assert edge_set(a) != edge_set(c)
    assert a.neurons is graph.neurons  # signs, roles and readouts unchanged
    assert edge_set(graph) == edge_set(random_connectome())  # input not modified


def test_nulls_on_the_toy_brain(toy_spec):  # noqa: F811
    g = toy_spec(None).graph
    for null in (rewire_degree_preserving(g, seed=0), random_graph(g, seed=0),
                 rewire_within_blocks(g, neuron_blocks(g, "super_class"), seed=0)):
        assert_simple(null)
        assert null.num_edges == g.num_edges
        assert null.neurons is g.neurons


def test_rewire_stops_when_no_swap_is_possible():
    # A directed 3-cycle: every swap creates a self-loop, so nothing can be accepted.
    g = Connectome(pd.DataFrame({"super_class": ["a"] * 3}), np.array([0, 1, 2]), np.array([1, 2, 0]),
                   np.full(3, 5, np.int32), np.zeros(3, np.int16), ["np0"])
    report = {}
    null = rewire_degree_preserving(g, seed=0, report=report)
    assert report["swaps_accepted"] == 0
    assert edge_set(null) == edge_set(g)


def test_neuropil_blocks_follow_dominant_output():
    g = Connectome(
        pd.DataFrame({"super_class": ["a"] * 4}),
        np.array([0, 0, 1, 1]), np.array([1, 2, 2, 3]),
        np.array([10, 30, 50, 5], np.int32), np.array([2, 1, 0, 0], np.int16), ["np0", "np1", "np2"],
    )
    np.testing.assert_array_equal(neuron_blocks(g, "neuropil"), [1, 0, -1, -1])
    np.testing.assert_array_equal(neuron_blocks(g, "super_class"), [0, 0, 0, 0])
    with pytest.raises(ValueError, match="unknown block"):
        neuron_blocks(g, "lobe")
