import pytest
import torch

from connectome_driver.models.cores.rocm_sparse import RocmSparseMatmul


def test_rocm_backend_requires_rocm_without_loading_dll():
    backend = RocmSparseMatmul()
    dense = torch.eye(3).bfloat16()
    a = dense.to_sparse_csr()
    with pytest.raises(RuntimeError, match="requires a ROCm GPU"):
        backend(a, a, torch.ones(3, 2, dtype=torch.bfloat16))
    backend.close()
    backend.close()


@pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm")
def test_row_product_rectangular_gradients_and_independent_outputs():
    backend = RocmSparseMatmul()
    dense = torch.tensor([[1., 0., -0.5], [0., 2., 0.], [0., 0., 0.], [0.25, 0., 1.]])
    a = dense.bfloat16().to_sparse_csr().cuda()
    at = dense.T.contiguous().bfloat16().to_sparse_csr().cuda()
    # Non-contiguous input and nonconstant upstream gradient exercise both directions.
    x = torch.arange(15, dtype=torch.float32).reshape(5, 3).T.bfloat16().cuda().detach()
    x.requires_grad_()
    grad = torch.linspace(-1, 1, 20).reshape(4, 5).bfloat16().cuda()
    try:
        first = backend(a, at, x)
        saved = first.detach().clone()
        second = backend(a, at, x + 1)
        torch.testing.assert_close(first, saved, rtol=0, atol=0)
        assert not torch.equal(first, second)
        first.backward(grad)
        expected = (dense @ x.detach().cpu().float()).bfloat16()
        expected_grad = (dense.T @ grad.cpu().float()).bfloat16()
        torch.testing.assert_close(first.cpu(), expected, rtol=0.02, atol=0.02)
        torch.testing.assert_close(x.grad.cpu(), expected_grad, rtol=0.02, atol=0.02)
        backend.close()
        # A closed cache can be rebuilt without stale descriptors or buffers.
        torch.testing.assert_close(backend(a, at, x).cpu(), expected, rtol=0.02, atol=0.02)
    finally:
        backend.close()


@pytest.mark.skipif(not torch.version.hip or not torch.cuda.is_available(), reason="requires ROCm")
def test_empty_row_product_has_zero_gradient():
    backend = RocmSparseMatmul()
    a = torch.zeros(3, 3, dtype=torch.bfloat16, device="cuda").to_sparse_csr()
    x = torch.randn(3, 4, device="cuda", dtype=torch.bfloat16, requires_grad=True)
    out = backend(a, a, x)
    out.sum().backward()
    assert torch.count_nonzero(out) == 0
    assert torch.count_nonzero(x.grad) == 0
