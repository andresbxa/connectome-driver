# Opt-in Windows ROCm execution

The default graph backend remains PyTorch CSR. `model.core.sparse_backend=rocm_row`
selects a bfloat16 CSR implementation using the official rocSPARSE library installed
with the Windows ROCm wheels. It uses row-major operands and the deterministic CSR
row-split algorithm for both `A @ X` and `A.T @ grad`.

This backend changes execution and floating-point reduction ordering, not the graph,
learnable parameters, node dynamics, loss or PPO minibatch. It is not bitwise equivalent
to every PyTorch backend. Compare outputs and individual gradients before using a new
GPU/software combination; passing numerical tests alone does not establish equal
learning curves. Linux ROCm and other dtypes are not enabled by this backend.

## Local setup

Install a compatible ROCm PyTorch wheel separately. The initial measured environment was
RX 7600 / gfx1102, Windows 11, PyTorch 2.14.0+rocm10.1.0. The wheel's
`_rocm_sdk_libraries/bin/rocsparse.dll` supplies the native API; no compiler or third-party
sparse package is required. CPU CI does not load this DLL.

Generate a local TunableOp cache for the actual graph size and micro-batch:

```powershell
python scripts/tune_rocm.py --nodes 139255 --micro-batch 8 --out .venv/rocm-gemm.csv
$env:CONNECTOME_TUNABLEOP_CACHE = (Resolve-Path .venv/rocm-gemm.csv).Path
python scripts/run_experiments.py configs/experiments/whole_brain_rocm.yaml --status
```

The tuner measures the very tall reduction that produces the shared graph-weight
gradient. Caches are local machine artifacts: do not commit or copy solution indices
between GPU/library versions. PyTorch validates cached results against its runtime.
Training/evaluation replay the cache with further tuning disabled. Each run writes
`runtime.json` with the GPU, versions, backend and cache SHA-256; evaluation writes its
own runtime record. Keep that record alongside `config.yaml` and the ordinary outputs.
An existing compatible cache is reused by the tuner. `log.print_every=1` prints and
flushes progress at every PPO iteration, which is useful for long whole-brain runs.

## Validate, then run

- Run `ruff check src scripts tests` and `pytest`. GPU tests require ROCm; CPU tests
  verify the default path and skip the native GPU tests.
- Compare the complete policy's outputs and **every** parameter gradient against the
  default backend, including the shared message/self matrices. Test all graph variants
  and seeds in the intended sweep.
- Run a short, separately named PPO execution check and reload its checkpoint. Measure
  complete iterations after warmup rather than extrapolating from one kernel.
- Use the same backend, cache and micro-batch for every run of a comparison. Keep the
  common reference machine for published timing/memory comparisons.

Once validated:

```powershell
python scripts/run_experiments.py configs/experiments/whole_brain_rocm.yaml
```

This plan retains the original six graph/seed pairs and uses micro-batches of 8 inside
the original 256-sample PPO minibatch. The normal `whole_brain.yaml` plan is unchanged.
Choose a fresh output location or preserve old partial runs before restarting; the
runner cannot resume an interrupted optimizer/RNG state.

## Implementation details

Descriptors/workspaces are owned by each graph core and cached by adjacency, width and
CUDA stream. Each call returns independent storage: a later message-passing step must
not overwrite tensors retained by autograd. The implementation synchronizes before
releasing temporary operands, serializes descriptor updates and checks int32 index
bounds. Moving a core to another device clears its native cache. Adjacency gradients
are deliberately unsupported, matching the fixed graph model.

References: [rocSPARSE ordering and algorithms](https://rocm.docs.amd.com/projects/rocSPARSE/en/latest/reference/generic.html),
[PyTorch TunableOp](https://docs.pytorch.org/docs/main/cuda.tunable.html),
[PyTorch ROCm operand preparation](https://github.com/pytorch/pytorch/blob/v2.14.0/aten/src/ATen/native/sparse/cuda/SparseBlasImpl.cpp).

## Initial validation evidence

[Local validation](../results/rocm_validation.json) records all six full-graph/seed
pairs, with initial and nonzero neuron biases, and a separate 2,048-step PPO execution
check including checkpoint reload and one evaluation episode. The post-warmup iteration
ran at about 22.7 agent steps/s, suggesting about 12.3 training hours per million steps
on the measured RX 7600 setup. This is a short-run extrapolation, not a completed
training result or a guarantee of equal learning curves.
