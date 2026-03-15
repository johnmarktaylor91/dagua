"""Tests for CUDA and numpy CSR construction paths."""

from __future__ import annotations

import pytest
import torch

from dagua.utils import _build_csr


def _reference_csr(
    src: torch.Tensor, tgt: torch.Tensor, num_nodes: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Build CSR via naive Python for ground-truth comparison.

    Parameters
    ----------
    src, tgt : torch.Tensor
        Edge endpoint tensors.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(offsets, targets)`` with targets sorted within each node's range.
    """
    neighbors: dict[int, list[int]] = {i: [] for i in range(num_nodes)}
    for s, t in zip(src.tolist(), tgt.tolist()):
        neighbors[s].append(t)
    offsets = [0]
    targets = []
    for i in range(num_nodes):
        sorted_nbrs = sorted(neighbors[i])
        targets.extend(sorted_nbrs)
        offsets.append(len(targets))
    return (
        torch.tensor(offsets, dtype=torch.long),
        torch.tensor(targets, dtype=torch.long),
    )


def _assert_csr_equivalent(
    offsets_a: torch.Tensor,
    targets_a: torch.Tensor,
    offsets_b: torch.Tensor,
    targets_b: torch.Tensor,
    num_nodes: int,
) -> None:
    """Assert two CSR representations describe the same adjacency.

    Order within each node's neighbor list may differ, so we compare sets.

    Parameters
    ----------
    offsets_a, targets_a, offsets_b, targets_b : torch.Tensor
        Two CSR representations to compare.
    num_nodes : int
        Number of nodes.
    """
    assert torch.equal(offsets_a, offsets_b), "CSR offsets differ"
    for i in range(num_nodes):
        s, e = offsets_a[i].item(), offsets_a[i + 1].item()
        set_a = set(targets_a[s:e].tolist())
        set_b = set(targets_b[s:e].tolist())
        assert set_a == set_b, f"Node {i}: {set_a} != {set_b}"


@pytest.fixture
def small_graph() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Small random graph for CSR tests.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, int]
        ``(src, tgt, num_nodes)``
    """
    torch.manual_seed(42)
    num_nodes = 1000
    num_edges = 5000
    src = torch.randint(0, num_nodes, (num_edges,))
    tgt = torch.randint(0, num_nodes, (num_edges,))
    return src, tgt, num_nodes


@pytest.fixture
def medium_graph() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Medium graph to test scaling.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, int]
        ``(src, tgt, num_nodes)``
    """
    torch.manual_seed(123)
    num_nodes = 10000
    num_edges = 50000
    src = torch.randint(0, num_nodes, (num_edges,))
    tgt = torch.randint(0, num_nodes, (num_edges,))
    return src, tgt, num_nodes


def test_build_csr_correctness(small_graph: tuple) -> None:
    """_build_csr should produce correct adjacency for a small graph."""
    src, tgt, n = small_graph
    offsets, targets = _build_csr(src, tgt, n, "cpu")
    ref_offsets, ref_targets = _reference_csr(src, tgt, n)
    _assert_csr_equivalent(offsets, targets, ref_offsets, ref_targets, n)


def test_build_csr_empty_graph() -> None:
    """CSR of an empty graph should have zero-length targets."""
    src = torch.zeros(0, dtype=torch.long)
    tgt = torch.zeros(0, dtype=torch.long)
    offsets, targets = _build_csr(src, tgt, 5, "cpu")
    assert offsets.shape == (6,)
    assert targets.shape == (0,)
    assert (offsets == 0).all()


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_csr_matches_cpu(medium_graph: tuple) -> None:
    """CUDA CSR kernel should produce same adjacency as CPU path."""
    src, tgt, n = medium_graph
    cpu_offsets, cpu_targets = _build_csr(src, tgt, n, "cpu")

    try:
        from dagua.layout.cuda_kernels import build_csr_cuda, is_available

        if not is_available():
            pytest.skip("CUDA kernel compilation failed")
        cuda_offsets, cuda_targets = build_csr_cuda(src.cuda(), tgt.cuda(), n)
        _assert_csr_equivalent(
            cpu_offsets,
            cpu_targets,
            cuda_offsets.cpu(),
            cuda_targets.cpu(),
            n,
        )
    except ImportError:
        pytest.skip("CUDA kernels not available")


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_cuda_csr_int32_edges() -> None:
    """CUDA CSR should handle int32 edge tensors."""
    torch.manual_seed(99)
    n, e = 500, 2000
    src = torch.randint(0, n, (e,), dtype=torch.int32)
    tgt = torch.randint(0, n, (e,), dtype=torch.int32)

    try:
        from dagua.layout.cuda_kernels import build_csr_cuda, is_available

        if not is_available():
            pytest.skip("CUDA kernel compilation failed")
        offsets, targets = build_csr_cuda(src.cuda(), tgt.cuda(), n)
        assert offsets.shape == (n + 1,)
        assert targets.shape == (e,)
        assert targets.dtype == torch.int32
    except ImportError:
        pytest.skip("CUDA kernels not available")


def test_numpy_csr_matches_reference(small_graph: tuple) -> None:
    """Numpy CSR path should produce correct adjacency."""
    from dagua.utils import _build_csr_numpy

    src, tgt, n = small_graph
    np_offsets, np_targets = _build_csr_numpy(src, tgt, n)
    ref_offsets, ref_targets = _reference_csr(src, tgt, n)
    _assert_csr_equivalent(np_offsets, np_targets, ref_offsets, ref_targets, n)
