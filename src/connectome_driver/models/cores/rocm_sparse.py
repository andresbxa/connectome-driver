"""Opt-in CSR products for the validated Windows ROCm wheels.

rocSPARSE's row-split algorithm uses row-major operands directly. The ordinary
PyTorch path remains the default. This backend supports fixed bfloat16 adjacency
matrices; it never computes edge-weight gradients. See docs/rocm.md.
"""

from __future__ import annotations

import ctypes as ct
import sys
import threading
import weakref
from pathlib import Path

import torch

_PTR, _INT, _I64, _SIZE = ct.c_void_p, ct.c_int, ct.c_int64, ct.c_size_t


def _check(status: int) -> None:
    if status:
        raise RuntimeError(f"rocSPARSE failed with status {status}")


class _API:
    def __init__(self):
        if sys.platform != "win32":
            raise RuntimeError("rocm_row currently supports the validated Windows ROCm wheels")
        path = Path(torch.__file__).parents[1] / "_rocm_sdk_libraries/bin/rocsparse.dll"
        if not path.is_file():
            raise RuntimeError(f"The ROCm wheel's rocSPARSE library was not found: {path}")
        self.library = ct.CDLL(str(path))
        signatures = {
            "create_handle": ("rocsparse_create_handle", [ct.POINTER(_PTR)]),
            "destroy_handle": ("rocsparse_destroy_handle", [_PTR]),
            "set_stream": ("rocsparse_set_stream", [_PTR, _PTR]),
            "create_csr": ("rocsparse_create_csr_descr",
                           [ct.POINTER(_PTR), _I64, _I64, _I64, _PTR, _PTR, _PTR,
                            _INT, _INT, _INT, _INT]),
            "destroy_csr": ("rocsparse_destroy_spmat_descr", [_PTR]),
            "create_dense": ("rocsparse_create_dnmat_descr",
                             [ct.POINTER(_PTR), _I64, _I64, _I64, _PTR, _INT, _INT]),
            "destroy_dense": ("rocsparse_destroy_dnmat_descr", [_PTR]),
            "set_values": ("rocsparse_dnmat_set_values", [_PTR, _PTR]),
            "spmm": ("rocsparse_spmm", [_PTR, _INT, _INT, _PTR, _PTR, _PTR, _PTR,
                                         _PTR, _INT, _INT, _INT, ct.POINTER(_SIZE), _PTR]),
        }
        for name, (symbol, arguments) in signatures.items():
            function = getattr(self.library, symbol)
            function.argtypes, function.restype = arguments, _INT
            setattr(self, name, function)


class _Operation:
    def __init__(self, api: _API, handle, matrix: torch.Tensor, x: torch.Tensor):
        self.api, self.handle, self.matrix = api, handle, matrix
        # Dimensions and nnz are checked before narrowing indices to signed int32.
        self.crow, self.cols = matrix.crow_indices().int(), matrix.col_indices().int()
        self.values = matrix.values()
        self.output = torch.empty((matrix.shape[0], x.shape[1]), device=x.device, dtype=x.dtype)
        self.a, self.b, self.c = _PTR(), _PTR(), _PTR()
        self.alpha, self.beta, self.size = ct.c_float(1), ct.c_float(0), _SIZE()
        self.buffer = None
        try:
            # Public rocSPARSE enums: i32=2, base_zero=0, bf16=168, order_row=0.
            _check(api.create_csr(ct.byref(self.a), *matrix.shape, matrix._nnz(),
                                  _PTR(self.crow.data_ptr()), _PTR(self.cols.data_ptr()),
                                  _PTR(self.values.data_ptr()), 2, 2, 0, 168))
            _check(api.create_dense(ct.byref(self.b), *x.shape, x.shape[1],
                                    _PTR(x.data_ptr()), 168, 0))
            _check(api.create_dense(ct.byref(self.c), *self.output.shape, self.output.shape[1],
                                    _PTR(self.output.data_ptr()), 168, 0))
            self.stage(1)  # buffer size
            self.buffer = torch.empty(max(1, self.size.value), device=x.device, dtype=torch.uint8)
            self.stage(2)  # preprocess once for this matrix/shape/stream
        except BaseException:
            self.close()
            raise

    def stage(self, stage: int) -> None:
        # operation_none=111, compute_f32=151, csr_row_split=4.
        _check(self.api.spmm(self.handle, 111, 111, ct.byref(self.alpha), self.a, self.b,
                             ct.byref(self.beta), self.c, 151, 4, stage, ct.byref(self.size),
                             _PTR(self.buffer.data_ptr()) if self.buffer is not None else _PTR()))

    def run(self, x: torch.Tensor) -> torch.Tensor:
        operand = x.detach().contiguous()
        _check(self.api.set_values(self.b, _PTR(operand.data_ptr())))
        self.stage(3)
        # Earlier message-passing steps must retain their own output for autograd.
        result = self.output.clone()
        # Keep operand, workspace and descriptors alive until the library is finished.
        torch.cuda.synchronize(x.device)
        return result

    def close(self) -> None:
        for name, destroy in (("c", self.api.destroy_dense), ("b", self.api.destroy_dense),
                              ("a", self.api.destroy_csr)):
            descriptor = getattr(self, name)
            if descriptor.value:
                destroy(descriptor)
                setattr(self, name, _PTR())


class _Resources:
    def __init__(self):
        self.api = None
        self.device = None
        self.handles = {}
        self.operations = {}
        self.lock = threading.RLock()


def _dispose(resources: _Resources) -> None:
    with resources.lock:
        if resources.api is None:
            return
        try:
            torch.cuda.synchronize(resources.device)
        finally:
            for operation in resources.operations.values():
                operation.close()
            resources.operations.clear()
            for handle in resources.handles.values():
                resources.api.destroy_handle(handle)
            resources.handles.clear()
            resources.api = None
            resources.device = None


class _RowProduct(torch.autograd.Function):
    @staticmethod
    def forward(ctx, backend, a, a_t, x):
        ctx.backend, ctx.a_t = backend, a_t
        return backend.product(a, x)

    @staticmethod
    def backward(ctx, grad):
        return None, None, None, ctx.backend.product(ctx.a_t, grad)


class RocmSparseMatmul:
    """Own the descriptors for one core; no DLL or GPU work occurs at construction."""

    def __init__(self):
        self._resources = _Resources()
        self._finalizer = weakref.finalize(self, _dispose, self._resources)

    def __call__(self, a: torch.Tensor, a_t: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if a_t.shape != (a.shape[1], a.shape[0]):
            raise ValueError("The supplied transpose has the wrong shape")
        if a.requires_grad or a_t.requires_grad:
            raise ValueError("rocm_row requires constant adjacency matrices")
        return _RowProduct.apply(self, a, a_t, x)

    def product(self, matrix: torch.Tensor, x: torch.Tensor) -> torch.Tensor:
        if torch.version.hip is None or not x.is_cuda:
            raise RuntimeError("rocm_row requires a ROCm GPU; CPU fallback is not supported")
        if matrix.layout != torch.sparse_csr or matrix.ndim != 2 or x.ndim != 2:
            raise ValueError("rocm_row requires a CSR matrix and a two-dimensional dense operand")
        if matrix.device != x.device or matrix.shape[1] != x.shape[0]:
            raise ValueError("Sparse and dense operands must have matching devices and dimensions")
        if matrix.dtype != torch.bfloat16 or x.dtype != torch.bfloat16:
            raise ValueError("rocm_row currently supports bfloat16 operands only")
        if max(*matrix.shape, matrix._nnz()) > torch.iinfo(torch.int32).max:
            raise ValueError("rocm_row's int32 indices cannot represent this matrix")
        if matrix._nnz() == 0 or x.shape[1] == 0:
            return x.new_zeros((matrix.shape[0], x.shape[1]))
        resources = self._resources
        # Serializes descriptor updates if host threads share a CUDA stream.
        with resources.lock, torch.cuda.device(x.device):
            if resources.api is None:
                resources.api = _API()
                resources.device = x.device
            if resources.device != x.device:
                raise ValueError("Close the backend before moving it to another GPU")
            stream = torch.cuda.current_stream(x.device).cuda_stream
            if stream not in resources.handles:
                handle = _PTR()
                _check(resources.api.create_handle(ct.byref(handle)))
                try:
                    _check(resources.api.set_stream(handle, _PTR(stream)))
                except BaseException:
                    resources.api.destroy_handle(handle)
                    raise
                resources.handles[stream] = handle
            key = (id(matrix), x.shape[1], stream)
            if key not in resources.operations:
                resources.operations[key] = _Operation(
                    resources.api, resources.handles[stream], matrix, x.detach().contiguous()
                )
            return resources.operations[key].run(x)

    def close(self) -> None:
        """Release cached native resources; safe to call repeatedly or reuse afterwards."""
        _dispose(self._resources)
