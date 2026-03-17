"""Tests for subset-resident GPU execution helpers and engine selection."""

from __future__ import annotations

from typing import Optional

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.constraints import (
    edge_attraction_loss,
    edge_length_variance_loss,
    edge_straightness_loss,
    overlap_avoidance_loss,
    repulsion_loss,
    spacing_consistency_loss,
)
from dagua.layout.engine import EdgeBatchContext, SampledNodeContext, _resolve_execution_mode
from dagua.layout.layers import LayerIndex, build_layer_index
from dagua.layout.multilevel import _apply_large_final_level_execution_overrides
from dagua.layout.subset_gpu import (
    EdgeAccessPattern,
    GlobalAccessPattern,
    SampledAccessPattern,
    SubsetGPUExecutor,
    SubsetGPULossTerm,
)


def _make_executor(
    node_sizes: torch.Tensor,
    layer_index: Optional[LayerIndex],
    edge_ref: list[Optional[torch.Tensor]],
    edge_ctx_ref: list[Optional[EdgeBatchContext]],
    sampled_ref: list[Optional[SampledNodeContext]],
) -> SubsetGPUExecutor:
    """Build a subset executor for CPU-backed tests.

    Parameters
    ----------
    node_sizes : torch.Tensor
        Node-size tensor shaped ``[N, 2]``.
    layer_index : LayerIndex, optional
        Layer structure for global loss tests.
    edge_ref : list[torch.Tensor | None]
        Mutable edge reference shared with the loss lambdas.
    edge_ctx_ref : list[EdgeBatchContext | None]
        Mutable edge-context reference shared with the loss lambdas.
    sampled_ref : list[SampledNodeContext | None]
        Mutable sampled-context reference shared with the loss lambdas.

    Returns
    -------
    SubsetGPUExecutor
        Executor configured to run subset logic on CPU for deterministic tests.
    """
    return SubsetGPUExecutor(
        node_sizes=node_sizes,
        layer_index=layer_index,
        execution_device="cpu",
        batch_edges_ref=edge_ref,
        edge_ctx_ref=edge_ctx_ref,
        sampled_ctx_ref=sampled_ref,
    )


def _sampled_context(
    num_nodes: int,
    sample_width: int = 16,
    active_count: int = 200,
) -> SampledNodeContext:
    """Create a deterministic sampled-node context for tests.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    sample_width : int, default=16
        Number of sampled indices per active node.
    active_count : int, default=200
        Number of active nodes included in the context.

    Returns
    -------
    SampledNodeContext
        Sampled-node context with fixed active and sampled indices.
    """
    active_idx = torch.arange(0, min(num_nodes, active_count), dtype=torch.long)
    sampled = torch.arange(active_idx.numel() * sample_width, dtype=torch.long)
    sampled = sampled.reshape(active_idx.numel(), sample_width) % num_nodes
    return SampledNodeContext(active_idx=active_idx, sampled=sampled)


def test_edge_access_pattern_remaps_indices_correctly() -> None:
    """EdgeAccessPattern should remap global edge endpoints into local indices."""
    edge_batch = torch.tensor([[10, 4, 10, 7], [4, 7, 1, 1]], dtype=torch.long)

    pattern = EdgeAccessPattern(edge_batch)
    unique_nodes = pattern.get_indices()
    prepared = pattern.remap_to_local(unique_nodes)

    expected_nodes = torch.tensor([1, 4, 7, 10], dtype=torch.long)
    expected_edges = torch.tensor([[3, 1, 3, 2], [1, 2, 0, 0]], dtype=torch.long)

    torch.testing.assert_close(unique_nodes, expected_nodes)
    assert prepared.local_edges is not None
    torch.testing.assert_close(
        prepared.local_edges,
        expected_edges.to(dtype=prepared.local_edges.dtype),
    )
    torch.testing.assert_close(unique_nodes[prepared.local_edges], edge_batch)


def test_subset_gpu_gradient_matches_standard_path() -> None:
    """Subset executor should match the direct mixed-loss gradient."""
    torch.manual_seed(7)
    num_nodes = 1_000
    pos = torch.randn(num_nodes, 2, dtype=torch.float32)
    node_sizes = torch.full((num_nodes, 2), 10.0, dtype=torch.float32)
    edge_index = torch.stack(
        [
            torch.arange(0, num_nodes - 1, dtype=torch.long),
            torch.arange(1, num_nodes, dtype=torch.long),
        ]
    )
    layer_index = build_layer_index(torch.arange(num_nodes, dtype=torch.long) % 8)
    sampled_ctx = _sampled_context(num_nodes)

    edge_ref: list[Optional[torch.Tensor]] = [edge_index]
    edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
    sampled_ref: list[Optional[SampledNodeContext]] = [sampled_ctx]

    def edge_loss(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the current edge-attraction term."""
        del ns, li
        assert edge_ref[0] is not None
        return edge_attraction_loss(p, edge_ref[0], x_bias=1.2)

    def overlap_loss(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the current sampled overlap term."""
        del li
        return overlap_avoidance_loss(p, ns, sampled_ctx=sampled_ref[0])

    def spacing_loss(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the current spacing-consistency term."""
        assert li is not None
        return spacing_consistency_loss(p, ns, li, target_gap=18.0)

    direct_pos = pos.clone().requires_grad_(True)
    direct_loss = (
        1.7 * edge_loss(direct_pos, node_sizes, layer_index)
        + 0.4 * overlap_loss(direct_pos, node_sizes, layer_index)
        + 0.2 * spacing_loss(direct_pos, node_sizes, layer_index)
    )
    direct_grad = torch.autograd.grad(direct_loss, direct_pos)[0]

    executor = _make_executor(node_sizes, layer_index, edge_ref, edge_ctx_ref, sampled_ref)
    exec_pos = pos.clone().requires_grad_(True)
    grad_buffer = executor.compute_step(
        exec_pos,
        [
            SubsetGPULossTerm(
                name="edge",
                weight=1.7,
                loss_fn=edge_loss,
                access_pattern=EdgeAccessPattern(edge_index),
            ),
            SubsetGPULossTerm(
                name="overlap",
                weight=0.4,
                loss_fn=overlap_loss,
                access_pattern=SampledAccessPattern(sampled_ctx),
            ),
            SubsetGPULossTerm(
                name="spacing",
                weight=0.2,
                loss_fn=spacing_loss,
                access_pattern=GlobalAccessPattern(),
            ),
        ],
    )

    torch.testing.assert_close(grad_buffer, direct_grad, atol=1e-5, rtol=1e-5)


def test_subset_gpu_edge_loss_matches_standard() -> None:
    """Edge-only subset execution should match direct attraction gradients."""
    pos = torch.tensor(
        [[0.0, 0.0], [20.0, 50.0], [40.0, 30.0], [80.0, 90.0]],
        dtype=torch.float32,
    )
    node_sizes = torch.full((4, 2), 10.0, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_ref: list[Optional[torch.Tensor]] = [edge_index]
    edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
    sampled_ref: list[Optional[SampledNodeContext]] = [None]

    def loss_fn(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the edge-attraction loss for the active edge ref."""
        del ns, li
        assert edge_ref[0] is not None
        return edge_attraction_loss(p, edge_ref[0], x_bias=1.0)

    direct_pos = pos.clone().requires_grad_(True)
    direct_grad = torch.autograd.grad(2.0 * loss_fn(direct_pos, node_sizes, None), direct_pos)[0]

    executor = _make_executor(node_sizes, None, edge_ref, edge_ctx_ref, sampled_ref)
    exec_pos = pos.clone().requires_grad_(True)
    grad_buffer = executor.compute_step(
        exec_pos,
        [
            SubsetGPULossTerm(
                name="edge",
                weight=2.0,
                loss_fn=loss_fn,
                access_pattern=EdgeAccessPattern(edge_index),
            )
        ],
    )

    torch.testing.assert_close(grad_buffer, direct_grad, atol=1e-5, rtol=1e-5)


def test_subset_gpu_shared_edge_remap_matches_per_term_remap() -> None:
    """Shared edge remapping should preserve the multi-term edge gradient."""
    torch.manual_seed(19)
    num_nodes = 256
    pos = torch.randn(num_nodes, 2, dtype=torch.float32)
    node_sizes = torch.full((num_nodes, 2), 8.0, dtype=torch.float32)
    edge_index = torch.stack(
        [
            torch.arange(0, num_nodes - 1, dtype=torch.long),
            torch.arange(1, num_nodes, dtype=torch.long),
        ]
    )

    def _run_executor(*edge_batches: torch.Tensor) -> torch.Tensor:
        """Execute the same edge losses against the provided edge batches."""
        edge_ref: list[Optional[torch.Tensor]] = [edge_index]
        edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
        sampled_ref: list[Optional[SampledNodeContext]] = [None]

        def attraction_loss(
            p: torch.Tensor,
            ns: torch.Tensor,
            li: Optional[LayerIndex],
        ) -> torch.Tensor:
            """Return the active edge-attraction loss."""
            del ns, li
            assert edge_ref[0] is not None
            return edge_attraction_loss(p, edge_ref[0], x_bias=1.1, edge_ctx=edge_ctx_ref[0])

        def straightness_loss(
            p: torch.Tensor,
            ns: torch.Tensor,
            li: Optional[LayerIndex],
        ) -> torch.Tensor:
            """Return the active edge-straightness loss."""
            del ns, li
            assert edge_ref[0] is not None
            return edge_straightness_loss(p, edge_ref[0], edge_ctx=edge_ctx_ref[0])

        def variance_loss(
            p: torch.Tensor,
            ns: torch.Tensor,
            li: Optional[LayerIndex],
        ) -> torch.Tensor:
            """Return the active edge-length-variance loss."""
            del ns, li
            assert edge_ref[0] is not None
            return edge_length_variance_loss(p, edge_ref[0], edge_ctx=edge_ctx_ref[0])

        executor = _make_executor(node_sizes, None, edge_ref, edge_ctx_ref, sampled_ref)
        return executor.compute_step(
            pos.clone().requires_grad_(True),
            [
                SubsetGPULossTerm(
                    name="attract",
                    weight=1.3,
                    loss_fn=attraction_loss,
                    access_pattern=EdgeAccessPattern(edge_batches[0]),
                ),
                SubsetGPULossTerm(
                    name="straight",
                    weight=0.4,
                    loss_fn=straightness_loss,
                    access_pattern=EdgeAccessPattern(edge_batches[1]),
                ),
                SubsetGPULossTerm(
                    name="variance",
                    weight=0.2,
                    loss_fn=variance_loss,
                    access_pattern=EdgeAccessPattern(edge_batches[2]),
                ),
            ],
        )

    direct_pos = pos.clone().requires_grad_(True)
    direct_loss = (
        1.3 * edge_attraction_loss(direct_pos, edge_index, x_bias=1.1)
        + 0.4 * edge_straightness_loss(direct_pos, edge_index)
        + 0.2 * edge_length_variance_loss(direct_pos, edge_index)
    )
    direct_grad = torch.autograd.grad(direct_loss, direct_pos)[0]

    shared_grad = _run_executor(edge_index, edge_index, edge_index)
    per_term_grad = _run_executor(edge_index.clone(), edge_index.clone(), edge_index.clone())

    torch.testing.assert_close(shared_grad, direct_grad, atol=1e-5, rtol=1e-5)
    torch.testing.assert_close(shared_grad, per_term_grad, atol=1e-5, rtol=1e-5)


def test_subset_gpu_shared_remap_falls_back_for_distinct_edge_batches() -> None:
    """Distinct edge batch objects should skip shared remap and still succeed."""
    pos = torch.tensor(
        [[0.0, 0.0], [25.0, 15.0], [50.0, 10.0], [80.0, 35.0]],
        dtype=torch.float32,
    )
    node_sizes = torch.full((4, 2), 10.0, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    edge_ref: list[Optional[torch.Tensor]] = [edge_index]
    edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
    sampled_ref: list[Optional[SampledNodeContext]] = [None]

    def attraction_loss(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the active edge-attraction loss."""
        del ns, li
        assert edge_ref[0] is not None
        return edge_attraction_loss(p, edge_ref[0], x_bias=1.0, edge_ctx=edge_ctx_ref[0])

    def straightness_loss(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the active edge-straightness loss."""
        del ns, li
        assert edge_ref[0] is not None
        return edge_straightness_loss(p, edge_ref[0], edge_ctx=edge_ctx_ref[0])

    direct_pos = pos.clone().requires_grad_(True)
    direct_grad = torch.autograd.grad(
        0.8 * edge_attraction_loss(direct_pos, edge_index, x_bias=1.0)
        + 0.3 * edge_straightness_loss(direct_pos, edge_index),
        direct_pos,
    )[0]

    executor = _make_executor(node_sizes, None, edge_ref, edge_ctx_ref, sampled_ref)
    grad_buffer = executor.compute_step(
        pos.clone().requires_grad_(True),
        [
            SubsetGPULossTerm(
                name="attract",
                weight=0.8,
                loss_fn=attraction_loss,
                access_pattern=EdgeAccessPattern(edge_index.clone()),
            ),
            SubsetGPULossTerm(
                name="straight",
                weight=0.3,
                loss_fn=straightness_loss,
                access_pattern=EdgeAccessPattern(edge_index.clone()),
            ),
        ],
    )

    torch.testing.assert_close(grad_buffer, direct_grad, atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_subset_gpu_shared_remap_logs_cpu_fallback_on_low_vram(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Low-VRAM shared remap should fall back to per-term execution cleanly."""
    pos = torch.randn(128, 2, dtype=torch.float32)
    node_sizes = torch.full((128, 2), 8.0, dtype=torch.float32)
    edge_index = torch.stack(
        [
            torch.arange(0, 127, dtype=torch.long),
            torch.arange(1, 128, dtype=torch.long),
        ]
    )
    edge_ref: list[Optional[torch.Tensor]] = [edge_index]
    edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
    sampled_ref: list[Optional[SampledNodeContext]] = [None]

    def attraction_loss(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the active edge-attraction loss."""
        del ns, li
        assert edge_ref[0] is not None
        return edge_attraction_loss(p, edge_ref[0], x_bias=1.1, edge_ctx=edge_ctx_ref[0])

    def straightness_loss(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the active edge-straightness loss."""
        del ns, li
        assert edge_ref[0] is not None
        return edge_straightness_loss(p, edge_ref[0], edge_ctx=edge_ctx_ref[0])

    direct_pos = pos.clone().to("cuda").requires_grad_(True)
    direct_grad = torch.autograd.grad(
        0.8 * edge_attraction_loss(direct_pos, edge_index.to("cuda"), x_bias=1.1)
        + 0.3 * edge_straightness_loss(direct_pos, edge_index.to("cuda")),
        direct_pos,
    )[0].cpu()

    executor = SubsetGPUExecutor(
        node_sizes=node_sizes,
        layer_index=None,
        execution_device="cuda",
        batch_edges_ref=edge_ref,
        edge_ctx_ref=edge_ctx_ref,
        sampled_ctx_ref=sampled_ref,
        verbose=True,
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1, 2))

    grad_buffer = executor.compute_step(
        pos.clone().requires_grad_(True),
        [
            SubsetGPULossTerm(
                name="attract",
                weight=0.8,
                loss_fn=attraction_loss,
                access_pattern=EdgeAccessPattern(edge_index),
            ),
            SubsetGPULossTerm(
                name="straight",
                weight=0.3,
                loss_fn=straightness_loss,
                access_pattern=EdgeAccessPattern(edge_index),
            ),
        ],
    )

    torch.testing.assert_close(grad_buffer, direct_grad, atol=1e-5, rtol=1e-5)
    assert "[dagua]   Shared edge remap: CPU fallback (need " in capsys.readouterr().out


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
def test_subset_gpu_shared_remap_logs_oom_cpu_fallback(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Shared remap OOM should fall back to per-term execution cleanly."""
    subset_gpu_module = __import__("dagua.layout.subset_gpu", fromlist=["unused"])
    pos = torch.randn(96, 2, dtype=torch.float32)
    node_sizes = torch.full((96, 2), 8.0, dtype=torch.float32)
    edge_index = torch.stack(
        [
            torch.arange(0, 95, dtype=torch.long),
            torch.arange(1, 96, dtype=torch.long),
        ]
    )
    edge_ref: list[Optional[torch.Tensor]] = [edge_index]
    edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
    sampled_ref: list[Optional[SampledNodeContext]] = [None]

    def attraction_loss(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the active edge-attraction loss."""
        del ns, li
        assert edge_ref[0] is not None
        return edge_attraction_loss(p, edge_ref[0], x_bias=1.0, edge_ctx=edge_ctx_ref[0])

    def variance_loss(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the active edge-length-variance loss."""
        del ns, li
        assert edge_ref[0] is not None
        return edge_length_variance_loss(p, edge_ref[0], edge_ctx=edge_ctx_ref[0])

    direct_pos = pos.clone().to("cuda").requires_grad_(True)
    direct_grad = torch.autograd.grad(
        0.9 * edge_attraction_loss(direct_pos, edge_index.to("cuda"), x_bias=1.0)
        + 0.2 * edge_length_variance_loss(direct_pos, edge_index.to("cuda")),
        direct_pos,
    )[0].cpu()

    original_build_local_edge_context = subset_gpu_module._build_local_edge_context
    raised = {"done": False}

    def _raise_oom_once(
        local_pos: torch.Tensor,
        local_edge_index: torch.Tensor,
    ) -> EdgeBatchContext:
        """Raise one shared-remap OOM before delegating to the real builder."""
        if not raised["done"]:
            raised["done"] = True
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return original_build_local_edge_context(local_pos, local_edge_index)

    executor = SubsetGPUExecutor(
        node_sizes=node_sizes,
        layer_index=None,
        execution_device="cuda",
        batch_edges_ref=edge_ref,
        edge_ctx_ref=edge_ctx_ref,
        sampled_ctx_ref=sampled_ref,
        verbose=True,
    )
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (10_000_000_000, 20_000_000_000))
    monkeypatch.setattr(subset_gpu_module, "_build_local_edge_context", _raise_oom_once)

    grad_buffer = executor.compute_step(
        pos.clone().requires_grad_(True),
        [
            SubsetGPULossTerm(
                name="attract",
                weight=0.9,
                loss_fn=attraction_loss,
                access_pattern=EdgeAccessPattern(edge_index),
            ),
            SubsetGPULossTerm(
                name="variance",
                weight=0.2,
                loss_fn=variance_loss,
                access_pattern=EdgeAccessPattern(edge_index),
            ),
        ],
    )

    torch.testing.assert_close(grad_buffer, direct_grad, atol=1e-5, rtol=1e-5)
    out = capsys.readouterr().out
    assert "[dagua]   Shared edge remap: CUDA (" in out
    assert "[dagua]   Shared edge remap: OOM, CPU fallback" in out


def test_subset_gpu_sampled_repulsion_matches_standard() -> None:
    """Sampled repulsion should match direct gradients after local remapping."""
    torch.manual_seed(11)
    num_nodes = 1_000
    pos = torch.randn(num_nodes, 2, dtype=torch.float32)
    node_sizes = torch.rand(num_nodes, 2, dtype=torch.float32) + 1.0
    sampled_ctx = _sampled_context(num_nodes, sample_width=24)
    edge_ref: list[Optional[torch.Tensor]] = [None]
    edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
    sampled_ref: list[Optional[SampledNodeContext]] = [sampled_ctx]

    def loss_fn(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return sampled repulsion for the active sampled context."""
        del li
        return repulsion_loss(
            p,
            num_nodes=num_nodes,
            threshold=0,
            sample_k=24,
            layer_index=None,
            node_sizes=ns,
            rvs_threshold=1,
            rvs_nn_k=8,
            sampled_ctx=sampled_ref[0],
        )

    direct_pos = pos.clone().requires_grad_(True)
    direct_grad = torch.autograd.grad(0.7 * loss_fn(direct_pos, node_sizes, None), direct_pos)[0]

    executor = _make_executor(node_sizes, None, edge_ref, edge_ctx_ref, sampled_ref)
    exec_pos = pos.clone().requires_grad_(True)
    grad_buffer = executor.compute_step(
        exec_pos,
        [
            SubsetGPULossTerm(
                name="repel",
                weight=0.7,
                loss_fn=loss_fn,
                access_pattern=SampledAccessPattern(sampled_ctx),
            )
        ],
    )

    torch.testing.assert_close(grad_buffer, direct_grad, atol=1e-5, rtol=1e-5)


def test_subset_gpu_sampled_overlap_matches_standard_with_small_gather() -> None:
    """Sampled overlap should stay on the sampled branch even for small gathered subsets."""
    torch.manual_seed(13)
    num_nodes = 1_000
    pos = torch.randn(num_nodes, 2, dtype=torch.float32)
    node_sizes = torch.rand(num_nodes, 2, dtype=torch.float32) + 1.0
    sampled_ctx = _sampled_context(num_nodes, sample_width=4, active_count=50)
    edge_ref: list[Optional[torch.Tensor]] = [None]
    edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
    sampled_ref: list[Optional[SampledNodeContext]] = [sampled_ctx]

    def loss_fn(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return sampled overlap for the active sampled context."""
        del li
        return overlap_avoidance_loss(p, ns, sampled_ctx=sampled_ref[0])

    direct_pos = pos.clone().requires_grad_(True)
    direct_grad = torch.autograd.grad(0.9 * loss_fn(direct_pos, node_sizes, None), direct_pos)[0]

    executor = _make_executor(node_sizes, None, edge_ref, edge_ctx_ref, sampled_ref)
    exec_pos = pos.clone().requires_grad_(True)
    grad_buffer = executor.compute_step(
        exec_pos,
        [
            SubsetGPULossTerm(
                name="overlap",
                weight=0.9,
                loss_fn=loss_fn,
                access_pattern=SampledAccessPattern(sampled_ctx),
            )
        ],
    )

    max_abs_diff = (grad_buffer - direct_grad).abs().max().item()
    assert torch.allclose(grad_buffer, direct_grad, atol=1e-4), max_abs_diff


def test_global_spacing_loss_runs_on_cpu() -> None:
    """GlobalAccessPattern should match direct spacing gradients."""
    pos = torch.tensor(
        [[0.0, 0.0], [15.0, 0.0], [45.0, 50.0], [55.0, 50.0]],
        dtype=torch.float32,
    )
    node_sizes = torch.full((4, 2), 8.0, dtype=torch.float32)
    layer_index = build_layer_index(torch.tensor([0, 0, 1, 1], dtype=torch.long))
    edge_ref: list[Optional[torch.Tensor]] = [None]
    edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
    sampled_ref: list[Optional[SampledNodeContext]] = [None]

    def loss_fn(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return spacing-consistency loss."""
        assert li is not None
        return spacing_consistency_loss(p, ns, li, target_gap=20.0)

    direct_pos = pos.clone().requires_grad_(True)
    direct_grad = torch.autograd.grad(
        0.5 * loss_fn(direct_pos, node_sizes, layer_index), direct_pos
    )[0]

    executor = _make_executor(node_sizes, layer_index, edge_ref, edge_ctx_ref, sampled_ref)
    exec_pos = pos.clone().requires_grad_(True)
    grad_buffer = executor.compute_step(
        exec_pos,
        [
            SubsetGPULossTerm(
                name="spacing",
                weight=0.5,
                loss_fn=loss_fn,
                access_pattern=GlobalAccessPattern(),
            )
        ],
    )

    torch.testing.assert_close(grad_buffer, direct_grad, atol=1e-5, rtol=1e-5)


def test_subset_gpu_empty_edge_batch_produces_zero_gradient() -> None:
    """Empty edge batches should skip cleanly and leave the gradient buffer at zero."""
    pos = torch.randn(6, 2, dtype=torch.float32)
    node_sizes = torch.full((6, 2), 10.0, dtype=torch.float32)
    empty_edges = torch.zeros((2, 0), dtype=torch.long)
    edge_ref: list[Optional[torch.Tensor]] = [empty_edges]
    edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
    sampled_ref: list[Optional[SampledNodeContext]] = [None]

    def loss_fn(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the edge-attraction loss for the active edge batch."""
        del ns, li
        assert edge_ref[0] is not None
        return edge_attraction_loss(p, edge_ref[0], x_bias=1.0)

    executor = _make_executor(node_sizes, None, edge_ref, edge_ctx_ref, sampled_ref)
    grad_buffer = executor.compute_step(
        pos.clone().requires_grad_(True),
        [
            SubsetGPULossTerm(
                name="edge",
                weight=1.0,
                loss_fn=loss_fn,
                access_pattern=EdgeAccessPattern(empty_edges),
            )
        ],
    )

    torch.testing.assert_close(grad_buffer, torch.zeros_like(grad_buffer))


def test_compute_step_updates_optimizer_state() -> None:
    """Executor gradients should drive a normal optimizer step."""
    pos = torch.tensor([[0.0, 0.0], [20.0, 30.0], [60.0, 90.0]], dtype=torch.float32)
    node_sizes = torch.full((3, 2), 10.0, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_ref: list[Optional[torch.Tensor]] = [edge_index]
    edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
    sampled_ref: list[Optional[SampledNodeContext]] = [None]

    def loss_fn(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the edge-attraction loss for one optimizer step."""
        del ns, li
        assert edge_ref[0] is not None
        return edge_attraction_loss(p, edge_ref[0], x_bias=1.0)

    executor = _make_executor(node_sizes, None, edge_ref, edge_ctx_ref, sampled_ref)
    pos_param = pos.clone().requires_grad_(True)
    optimizer = torch.optim.SGD([pos_param], lr=0.1)
    before = pos_param.detach().clone()

    optimizer.zero_grad(set_to_none=True)
    pos_param.grad = executor.compute_step(
        pos_param,
        [
            SubsetGPULossTerm(
                name="edge",
                weight=1.0,
                loss_fn=loss_fn,
                access_pattern=EdgeAccessPattern(edge_index),
            )
        ],
    )
    optimizer.step()

    assert not torch.allclose(before, pos_param.detach())


def test_subset_gpu_amortized_skip_returns_zero_gradient() -> None:
    """A detached zero-valued amortized loss should produce zero subset gradients."""
    pos = torch.randn(4, 2, dtype=torch.float32)
    node_sizes = torch.full((4, 2), 5.0, dtype=torch.float32)
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    edge_ref: list[Optional[torch.Tensor]] = [edge_index]
    edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
    sampled_ref: list[Optional[SampledNodeContext]] = [None]

    def loss_fn(
        p: torch.Tensor,
        ns: torch.Tensor,
        li: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return a leaf zero tensor that is not connected to ``p``."""
        del ns, li
        return torch.zeros(1, device=p.device, requires_grad=True)

    executor = _make_executor(node_sizes, None, edge_ref, edge_ctx_ref, sampled_ref)
    grad_buffer = executor.compute_step(
        pos.clone().requires_grad_(True),
        [
            SubsetGPULossTerm(
                name="amortized_zero",
                weight=1.0,
                loss_fn=loss_fn,
                access_pattern=EdgeAccessPattern(edge_index),
            )
        ],
    )

    torch.testing.assert_close(grad_buffer, torch.zeros_like(grad_buffer))


def test_execution_mode_auto_selects_subset_gpu(monkeypatch: pytest.MonkeyPatch) -> None:
    """Large CUDA graphs should auto-select subset_gpu execution."""
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    config = LayoutConfig(device="cuda", execution_mode="auto", subset_gpu_threshold=10_000_000)

    assert _resolve_execution_mode(config, "cuda", 5_000_000) == "standard"
    assert _resolve_execution_mode(config, "cuda", 10_000_001) == "subset_gpu"
    assert _resolve_execution_mode(config, "cuda", 50_000_000) == "subset_gpu"


def test_large_final_level_override_selects_subset_gpu() -> None:
    """The 200M+ refinement override should force subset_gpu mode."""
    refine_config = LayoutConfig(
        device="cuda",
        execution_mode="standard",
        per_loss_backward="on",
        hybrid_device="on",
        gradient_checkpointing="on",
        edge_batch_size=0,
    )

    overridden = _apply_large_final_level_execution_overrides(
        refine_config,
        fine_n=200_000_000,
        force_cpu=False,
        level_edge_batch=1_000_000,
    )

    assert overridden.execution_mode == "subset_gpu"
    assert overridden.edge_batch_size == 1_000_000
    assert overridden.per_loss_backward == "off"
    assert overridden.hybrid_device == "off"
    assert overridden.gradient_checkpointing == "off"
