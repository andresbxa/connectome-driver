import importlib.util

import pytest
import torch
from omegaconf import OmegaConf
from test_graph_core import make_core, toy_spec  # noqa: F401  (fixture)

from connectome_driver.data.graphs import apply_variant
from connectome_driver.models.cores.graph import graph_core_param_count
from connectome_driver.models.cores.mlp import LinearCore, matched_hidden_width, mlp_param_count
from connectome_driver.models.layers import count_parameters
from connectome_driver.models.policy import build_policy

NULLS_AVAILABLE = importlib.util.find_spec("connectome_driver.data.nulls") is not None


def test_graph_core_param_count_matches_module(toy_spec):  # noqa: F811
    spec = toy_spec(None)
    core = make_core(spec, steps=2)
    expected = graph_core_param_count(spec.graph.num_nodes, len(spec.input_index), 6, 3)
    assert count_parameters(core) == expected


def test_matched_mlp_on_k3_subgraph_is_within_one_percent():
    # Sizes of the k=3 experiment graph: 13,261 neurons, 3,628 photoreceptors, 1,014 readouts.
    target = graph_core_param_count(13_261, 3_628, 64, 8)
    assert target == 338_417  # what the Phase 3 run reported
    width = matched_hidden_width(64, 1_014 * 8, depth=2, target=target)
    count = mlp_param_count(64, 1_014 * 8, [width, width])
    assert abs(count - target) / target < 0.01


def test_matched_mlp_policy_builds_from_the_graph(toy_spec):  # noqa: F811
    spec = toy_spec(None)
    cfg = OmegaConf.create({
        "encoder": {"type": "flat", "feature_dim": 6}, "readout_neurons": "auto", "node_dim": 3,
        "decoder": {"hidden_dim": 8}, "core": {"type": "mlp", "hidden_dims": "match_graph", "depth": 2},
    })
    policy = build_policy(cfg, (4,), 5, spec)
    width = matched_hidden_width(6, 2 * 3, 2, graph_core_param_count(10, 2, 6, 3))
    assert [m.out_features for m in policy.core.net if isinstance(m, torch.nn.Linear)] == [width, width, 6]


def test_no_core_is_a_single_affine_map():
    core = LinearCore(in_dim=5, readout_neurons=4, node_dim=3)
    z1, z2 = torch.randn(2, 5), torch.randn(2, 5)
    out = core(0.3 * z1 + 0.7 * z2)
    assert out.shape == (2, 4, 3)
    torch.testing.assert_close(out, 0.3 * core(z1) + 0.7 * core(z2))
    assert count_parameters(core) == 5 * 12 + 12


def test_unknown_graph_variant_is_rejected(toy_spec):  # noqa: F811
    with pytest.raises(ValueError, match="unknown graph variant"):
        apply_variant(toy_spec(None).graph, "scrambled", 0, OmegaConf.create({}))


@pytest.mark.skipif(NULLS_AVAILABLE, reason="null models are implemented")
def test_null_variants_explain_what_is_missing(toy_spec):  # noqa: F811
    with pytest.raises(NotImplementedError, match="nulls.py"):
        apply_variant(toy_spec(None).graph, "rewired", 0, OmegaConf.create({}))


@pytest.mark.skipif(not NULLS_AVAILABLE, reason="null models not merged yet")
@pytest.mark.parametrize("variant", ["rewired", "random", "block"])
def test_null_variants_keep_neurons_and_edge_count(toy_spec, variant):  # noqa: F811
    graph = toy_spec(None).graph
    null = apply_variant(graph, variant, 0, OmegaConf.create({"swaps_per_edge": 2}))
    assert null.num_edges == graph.num_edges
    assert null.neurons is graph.neurons or null.neurons.equals(graph.neurons)
