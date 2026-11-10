"""Graph core: message passing over a fixed directed graph (the connectome or a null model).

Every neuron i holds a state h_i in R^d. At each environment step the state restarts from zero,
except the visual afferents, which start at tanh(x_i); then the core runs T steps of leaky rate
dynamics

    m   = A h                                   # weighted sum over presynaptic partners
    u   = m W_msg + h W_self + b_i + x_i        # x_i: encoder drive, visual afferents only
    h  <- (1 - a) h + a tanh(u)                 # a = sigmoid(leak) in (0, 1)

and returns the states of the readout (descending) neurons. ``A`` is the fixed sparse
adjacency (rows = postsynaptic): weights log(1 + synapses), normalized per postsynaptic neuron
(by default to unit L2 norm, which preserves signal variance across hops; see
``edge_weights``) and optionally signed by the presynaptic neuron's transmitter. All learnable parameters are
shared across neurons except the per-neuron bias ``b_i`` and the per-afferent input weights,
so the computation is shaped by the wiring, not by a neuron-specific network.

The encoder drive is injected at every step (not only at t = 0) so it is not diluted before
reaching the descending neurons. The batch is folded into the columns of the dense operand,
``(N, B * d)``, so each step is a single sparse-dense product whatever the batch size.
"""

from __future__ import annotations

import math

import numpy as np
import torch
from torch import nn
from torch.utils.checkpoint import checkpoint

from connectome_driver.data.graph import edge_weights, to_torch_csr


class SparseMatmul(torch.autograd.Function):
    """``A @ X`` with A sparse and constant. Backward is ``A.T @ grad`` with a precomputed A.T.

    Relying on autograd through CSR tensors is fragile (and would also try to track A); here A
    gets no gradient, nothing is saved for backward except the transposed matrix, and both
    directions are plain CSR x dense products.
    """

    @staticmethod
    def forward(ctx, a: torch.Tensor, a_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        ctx.a_t = a_t
        return a @ x

    @staticmethod
    def backward(ctx, grad: torch.Tensor):
        return None, None, ctx.a_t @ grad


def sparse_matmul(a: torch.Tensor, a_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
    return SparseMatmul.apply(a, a_t, x)


def graph_core_param_count(num_nodes: int, num_inputs: int, feature_dim: int, node_dim: int) -> int:
    """Learnable parameters of a ``GraphCore``; it does not depend on T or on the edges."""
    return (
        num_inputs * feature_dim  # input_proj
        + node_dim  # input_lift
        + 2 * node_dim * node_dim  # w_msg, w_self
        + num_nodes * node_dim  # bias
        + 1  # leak
    )


class GraphCore(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        src: np.ndarray,
        dst: np.ndarray,
        syn_count: np.ndarray,
        node_sign: np.ndarray,
        input_index: np.ndarray,
        readout_index: np.ndarray,
        feature_dim: int,
        node_dim: int,
        steps: int,
        signed: bool = True,
        normalize: str | None = "l2",
        leak_init: float = 0.9,
        dtype: str = "float32",
        checkpoint_steps: bool = False,
        sparse_backend: str = "torch",
    ):
        super().__init__()
        if sparse_backend not in ("torch", "rocm_row"):
            raise ValueError(f"unknown sparse backend {sparse_backend!r}")
        self.sparse_backend = sparse_backend
        self._rocm_matmul = None
        self.num_nodes, self.node_dim, self.steps = num_nodes, node_dim, steps
        self.compute_dtype = getattr(torch, dtype)
        self.checkpoint_steps = checkpoint_steps

        weight = edge_weights(syn_count, src, dst, num_nodes, node_sign if signed else None, normalize)
        a, a_t = to_torch_csr(src, dst, weight, num_nodes)
        # Sparse matrices are plain attributes, moved by `_apply` below (buffers cannot be CSR
        # tensors in every PyTorch version).
        self.a, self.a_t = a.to(self.compute_dtype), a_t.to(self.compute_dtype)
        self.register_buffer("input_index", torch.as_tensor(input_index, dtype=torch.long))
        self.register_buffer("readout_index", torch.as_tensor(readout_index, dtype=torch.long))

        n_in = len(input_index)
        # Each visual afferent gets a scalar drive (its own projection of z), lifted to R^d by a
        # shared vector: a photoreceptor carries one signal, not a feature vector.
        self.input_proj = nn.Parameter(torch.randn(n_in, feature_dim) / math.sqrt(feature_dim))
        self.input_lift = nn.Parameter(torch.ones(node_dim))
        self.w_msg = nn.Parameter(torch.empty(node_dim, node_dim))
        self.w_self = nn.Parameter(torch.empty(node_dim, node_dim))
        nn.init.orthogonal_(self.w_msg)
        nn.init.orthogonal_(self.w_self, gain=0.5)
        self.bias = nn.Parameter(torch.zeros(num_nodes, node_dim))
        # a = sigmoid(leak): fraction of the state replaced at each step (1 = no memory).
        self.leak = nn.Parameter(torch.tensor(math.log(leak_init / (1 - leak_init))))

    def _apply(self, fn, recurse=True):
        if self._rocm_matmul is not None:
            self._rocm_matmul.close()
            self._rocm_matmul = None
        super()._apply(fn, recurse)
        # Follow device moves only; the sparse matrices keep the compute dtype.
        self.a = fn(self.a).to(self.compute_dtype)
        self.a_t = fn(self.a_t).to(self.compute_dtype)
        return self

    def _step(self, h: torch.Tensor, drive: torch.Tensor) -> torch.Tensor:
        n, b, d = h.shape
        if self.sparse_backend == "rocm_row":
            if self._rocm_matmul is None:
                from connectome_driver.models.cores.rocm_sparse import RocmSparseMatmul

                self._rocm_matmul = RocmSparseMatmul()
            product = self._rocm_matmul(self.a, self.a_t, h.reshape(n, b * d))
        else:
            product = sparse_matmul(self.a, self.a_t, h.reshape(n, b * d))
        m = product.view(n, b, d)
        dt = self.compute_dtype
        u = m @ self.w_msg.to(dt) + h @ self.w_self.to(dt) + self.bias.to(dt)[:, None, :]
        u = u.index_add(0, self.input_index, drive)
        a = torch.sigmoid(self.leak).to(dt)
        return (1 - a) * h + a * torch.tanh(u)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        b = z.shape[0]
        dt = self.compute_dtype
        # (n_in, B) scalar drive per afferent -> (n_in, B, d)
        drive = ((self.input_proj @ z.T)[..., None] * self.input_lift).to(dt)
        # The afferents start already driven, so step t moves the signal t hops: with T steps it
        # reaches exactly the neurons within T hops of the photoreceptors.
        h = torch.zeros(self.num_nodes, b, self.node_dim, device=z.device, dtype=dt)
        h = h.index_copy(0, self.input_index, torch.tanh(drive))
        for _ in range(self.steps):
            if self.checkpoint_steps and torch.is_grad_enabled():
                h = checkpoint(self._step, h, drive, use_reentrant=False)
            else:
                h = self._step(h, drive)
        return h[self.readout_index].transpose(0, 1).float()  # (B, R, d)
