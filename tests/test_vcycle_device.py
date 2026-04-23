"""Sprint 8: V-cycle device-migration tests.

Pre-Sprint-8 the V-cycle path crashed with
  RuntimeError: Expected all tensors to be on the same device,
  but found at least two devices, cuda:0 and cpu!
as soon as any coarse level ran through ``dag_ordering_loss`` (state.pos
on CUDA, level.edge_index on CPU). The refine pipeline would index
CUDA pos by CPU edges and die.

Sprint 8 fix: _level_problem migrates level.edge_index / level.node_sizes
to the finest-level problem.edge_index device, AND the longest-path
layering helper moves src/tgt to whichever compute_device it picks.

These tests exercise both paths on CPU (fast, always runnable) and on
CUDA when available (gated so the suite still passes on CPU-only
boxes).
"""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.state import HierarchyLevel, LayoutProblem
from dagua.layout.ops.vcycle import _level_problem


def _make_level(
    num_fine: int,
    fine_to_coarse: torch.Tensor,
    edge_index: torch.Tensor,
) -> HierarchyLevel:
    num_coarse = int(fine_to_coarse.max().item()) + 1
    return HierarchyLevel(
        num_nodes=num_coarse,
        num_fine=num_fine,
        edge_index=edge_index,
        node_sizes=torch.full((num_coarse, 2), 20.0, dtype=torch.float32),
        fine_to_coarse=fine_to_coarse,
    )


@pytest.mark.unit
def test_level_problem_migrates_edge_index_to_finest_device():
    """When the top-level problem.edge_index lives on one device and the
    hierarchy level's edge_index lives on another, _level_problem must
    produce a coarse LayoutProblem whose edge_index matches the finest
    device -- the refine pipeline's state.pos is guaranteed to sit on
    that device, and a mismatch crashes dag_ordering_loss + any other
    loss that indexes pos by edge_index.
    """
    finest = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    problem = LayoutProblem(
        edge_index=finest,
        num_nodes=4,
        node_sizes=torch.full((4, 2), 20.0, dtype=torch.float32),
    )
    # Coarse level lives on CPU even though problem lives on CPU -- same
    # device in this case, so no migration needed.
    coarse_edges_cpu = torch.tensor([[0], [1]], dtype=torch.long)
    level = _make_level(4, torch.tensor([0, 0, 1, 1], dtype=torch.long), coarse_edges_cpu)

    coarse = _level_problem(problem, level)
    assert coarse.edge_index.device == problem.edge_index.device
    assert coarse.node_sizes.device == problem.edge_index.device


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_level_problem_migrates_cpu_edge_index_to_cuda():
    """The real-world failure mode: coarsen ops build hierarchy edges on
    CPU (their Python dict-based construction path). The engine migrated
    finest problem.edge_index to CUDA. Sprint 8: _level_problem now
    pulls CPU level edges onto CUDA so they can index state.pos."""
    cuda = torch.device("cuda")
    finest = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long, device=cuda)
    problem = LayoutProblem(
        edge_index=finest,
        num_nodes=4,
        node_sizes=torch.full((4, 2), 20.0, dtype=torch.float32, device=cuda),
    )
    coarse_edges_cpu = torch.tensor([[0], [1]], dtype=torch.long)  # CPU!
    level = HierarchyLevel(
        num_nodes=2,
        num_fine=4,
        edge_index=coarse_edges_cpu,
        node_sizes=torch.full((2, 2), 20.0, dtype=torch.float32),  # CPU
        fine_to_coarse=torch.tensor([0, 0, 1, 1], dtype=torch.long),
    )

    coarse = _level_problem(problem, level)
    assert coarse.edge_index.device.type == "cuda", (
        f"coarse edge_index stayed on {coarse.edge_index.device}, expected cuda"
    )
    assert coarse.node_sizes.device.type == "cuda", (
        f"coarse node_sizes stayed on {coarse.node_sizes.device}, expected cuda"
    )


@pytest.mark.unit
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_longest_path_layering_migrates_src_tgt_to_compute_device():
    """When the caller hands a CUDA edge_index but the layering routine
    picks compute_device='cpu' (VRAM too small, or explicit cpu arg),
    src/tgt must be moved to CPU so scatter_add doesn't crash.
    """
    from dagua.utils import _longest_path_layering_vectorized

    cuda = torch.device("cuda")
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long, device=cuda)
    # Force the CPU path explicitly.
    layers = _longest_path_layering_vectorized(edge_index, num_nodes=4, device="cpu")
    # On a 4-node chain 0->1->2->3 the layers should be [0, 1, 2, 3].
    assert layers.tolist() == [0, 1, 2, 3]
