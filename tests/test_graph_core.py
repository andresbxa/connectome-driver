import gymnasium as gym
import numpy as np
import pytest
import torch
from omegaconf import OmegaConf
from test_ppo import small_cfg

from connectome_driver.data.graphs import build_graph
from connectome_driver.data.preprocess import run
from connectome_driver.models.cores.graph import GraphCore, sparse_matmul
from connectome_driver.models.policy import build_policy
from connectome_driver.rl.ppo import PPO
from connectome_driver.utils.logging import RunLogger


def test_sparse_matmul_gradient_matches_dense():
    rng = np.random.default_rng(0)
    n = 12
    keys = rng.choice(n * n, size=40, replace=False)
    rows, cols = keys // n, keys % n
    dense = torch.zeros(n, n, dtype=torch.float64)
    dense[rows, cols] = torch.from_numpy(rng.normal(size=40))
    a, a_t = dense.to_sparse_csr(), dense.T.contiguous().to_sparse_csr()

    x = torch.randn(n, 5, dtype=torch.float64, requires_grad=True)
    torch.autograd.gradcheck(lambda x: sparse_matmul(a, a_t, x), (x,))
    torch.testing.assert_close(sparse_matmul(a, a_t, x), dense @ x)


@pytest.fixture
def toy_spec(cfg):
    run(cfg)  # preprocess the toy brain into cfg.processed_dir
    return lambda k: build_graph(
        OmegaConf.create({"processed_dir": cfg.processed_dir, "subgraph_k": k, "variant": "connectome"})
    )


def make_core(spec, steps, **kw) -> GraphCore:
    g = spec.graph
    torch.manual_seed(0)
    return GraphCore(
        g.num_nodes, g.src, g.dst, g.syn_count, g.neurons["sign"].to_numpy(),
        spec.input_index, spec.readout_index, feature_dim=6, node_dim=3, steps=steps, **kw,
    )


def test_graph_spec_whole_brain_and_subgraph(toy_spec):
    full = toy_spec(None)
    assert full.graph.num_nodes == 10
    assert len(full.input_index) == 2 and len(full.readout_index) == 2
    sub = toy_spec(3)
    assert sorted(sub.graph.neurons["root_id"]) == [101, 102, 201, 202, 301]
    assert sub.graph.neurons["root_id"].iloc[sub.readout_index].tolist() == [301]


def readout_grad_wrt_z(core: GraphCore) -> torch.Tensor:
    z = torch.randn(4, 6, requires_grad=True)
    out = core(z)  # (B, R, d)
    assert out.shape == (4, len(core.readout_index), 3)
    grads = []
    for r in range(out.shape[1]):
        (g,) = torch.autograd.grad(out[:, r].sum(), z, retain_graph=True)
        grads.append(g.abs().sum().item())
    return torch.tensor(grads)


def test_signal_needs_as_many_steps_as_hops(toy_spec):
    """Descending 301 is 3 hops from the photoreceptors and 302 is unreachable from them."""
    spec = toy_spec(None)
    ids = spec.graph.neurons["root_id"].iloc[spec.readout_index].tolist()
    assert ids == [301, 302]
    assert readout_grad_wrt_z(make_core(spec, steps=2)).tolist() == [0.0, 0.0]
    g3 = readout_grad_wrt_z(make_core(spec, steps=3))
    assert g3[0] > 0 and g3[1] == 0


def test_checkpointing_does_not_change_results(toy_spec):
    spec = toy_spec(None)
    z = torch.randn(4, 6)
    outs, grads = [], []
    for ckpt in (False, True):
        core = make_core(spec, steps=4, checkpoint_steps=ckpt)
        out = core(z)
        out.square().sum().backward()
        outs.append(out.detach())
        grads.append(torch.cat([p.grad.flatten() for p in core.parameters()]))
    torch.testing.assert_close(outs[0], outs[1])
    torch.testing.assert_close(grads[0], grads[1])


@pytest.mark.skipif(not torch.cuda.is_available(), reason="bf16 sparse products need CUDA")
def test_bfloat16_core_runs_and_returns_float32(toy_spec):
    core = make_core(toy_spec(None), steps=3, dtype="bfloat16").cuda()
    out = core(torch.randn(2, 6, device="cuda"))
    assert out.dtype == torch.float32 and torch.isfinite(out).all()


def test_micro_batching_gives_the_same_update(tmp_path):
    """One PPO update with and without micro-batches must give identical parameters."""
    results = []
    for micro in (None, 3):
        cfg = small_cfg(
            tmp_path, device="cpu",
            model={"encoder": {"type": "flat", "feature_dim": 8}, "readout_neurons": 4,
                   "node_dim": 2, "decoder": {"hidden_dim": 8}, "core": {"type": "mlp", "hidden_dims": [8]}},
            ppo={"num_envs": 2, "rollout_steps": 6, "num_minibatches": 2, "update_epochs": 1,
                 "total_steps": 12, "micro_batch_size": micro},
        )
        envs = gym.vector.SyncVectorEnv([lambda: gym.make("CartPole-v1")] * 2)
        torch.manual_seed(0)
        policy = build_policy(cfg.model, (4,), 2)
        ppo = PPO(cfg.ppo, policy, envs, torch.device("cpu"), RunLogger(cfg))
        g = torch.Generator().manual_seed(1)
        b = ppo.buffer
        b.obs.copy_(torch.randn(b.obs.shape, generator=g))
        b.actions.copy_(torch.randint(0, 2, b.actions.shape, generator=g))
        b.logprobs.fill_(np.log(0.5))
        b.values.copy_(torch.randn(b.values.shape, generator=g))
        adv, ret = torch.randn(6, 2, generator=g), torch.randn(6, 2, generator=g)
        torch.manual_seed(2)  # same minibatch permutation
        ppo.update(adv, ret)
        results.append(torch.cat([p.detach().flatten() for p in policy.parameters()]))
        envs.close()
    torch.testing.assert_close(results[0], results[1])
