"""CUDA activation, fallback, and OOM recovery tests for layout stages."""

from __future__ import annotations

import importlib
from typing import Optional

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.engine import (
    EdgeBatchContext,
    SampledNodeContext,
    _auto_edge_batch_size,
    _auto_sampled_node_cap,
    _layout_inner,
    _resolve_execution_mode,
)
from dagua.layout.layers import LayerIndex, build_layer_index
from dagua.layout.losses import (
    edge_attraction_loss,
    edge_length_variance_loss,
    edge_straightness_loss,
)
from dagua.layout.multilevel import _auto_cpu_edge_batch_size, coarsen_once
from dagua.layout.ops.pipelines.dagua_native import _should_lattice_uniform_centered_slots
from dagua.layout.projection import project_overlaps
from dagua.layout.subset_gpu import EdgeAccessPattern, SubsetGPUExecutor, SubsetGPULossTerm
from dagua.metrics import count_overlaps
from dagua.utils import longest_path_layering

CUDA_REQUIRED = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


def _make_layered_dag(
    num_layers: int,
    layer_width: int,
) -> tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]:
    """Build a deterministic layered DAG for CUDA path tests.

    Parameters
    ----------
    num_layers : int
        Number of layers in the DAG.
    layer_width : int
        Number of nodes per layer.

    Returns
    -------
    tuple[torch.Tensor, int, torch.Tensor, torch.Tensor]
        Edge index shaped ``[2, E]``, node count, node sizes shaped ``[N, 2]``,
        and layer assignments shaped ``[N]``.
    """
    num_nodes = num_layers * layer_width
    src_parts: list[torch.Tensor] = []
    tgt_parts: list[torch.Tensor] = []
    for layer in range(num_layers - 1):
        current = torch.arange(layer * layer_width, (layer + 1) * layer_width, dtype=torch.long)
        nxt = torch.arange(
            (layer + 1) * layer_width,
            (layer + 2) * layer_width,
            dtype=torch.long,
        )
        src_parts.append(current.repeat_interleave(2))
        tgt_parts.append(torch.stack([nxt, nxt.roll(shifts=-1)]).transpose(0, 1).reshape(-1))
    edge_index = torch.stack([torch.cat(src_parts), torch.cat(tgt_parts)])
    node_sizes = torch.full((num_nodes, 2), 10.0, dtype=torch.float32)
    layers = torch.arange(num_nodes, dtype=torch.long) // layer_width
    return edge_index, num_nodes, node_sizes, layers


def _make_projection_case(
    num_layers: int,
    layer_width: int,
) -> tuple[torch.Tensor, torch.Tensor, LayerIndex]:
    """Build a layered overlap case that exercises the sweep projector.

    Parameters
    ----------
    num_layers : int
        Number of layers to create.
    layer_width : int
        Nodes per layer.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, LayerIndex]
        Initial positions shaped ``[N, 2]``, node sizes shaped ``[N, 2]``, and
        the CPU layer index for the case.
    """
    xs = torch.arange(layer_width, dtype=torch.float32) * 8.0
    ys = torch.arange(num_layers, dtype=torch.float32) * 40.0
    pos = torch.stack(
        [
            xs.repeat(num_layers),
            ys.repeat_interleave(layer_width),
        ],
        dim=1,
    )
    node_sizes = torch.full((num_layers * layer_width, 2), 12.0, dtype=torch.float32)
    layer_assignments = torch.arange(num_layers * layer_width, dtype=torch.long) // layer_width
    return pos, node_sizes, build_layer_index(layer_assignments)


def _make_disconnected_polish_graph() -> DaguaGraph:
    """Build a small disconnected graph that exercises component tiling polish.

    Returns
    -------
    DaguaGraph
        Graph with two non-trivial components and computed node sizes.
    """
    graph = DaguaGraph()
    for node_id in range(5):
        graph.add_node(str(node_id))
    graph.add_edge("0", "1")
    graph.add_edge("2", "3")
    graph.add_edge("3", "4")
    graph.compute_node_sizes()
    return graph


def _make_cuda_lattice_gate_case() -> tuple[torch.Tensor, torch.Tensor]:
    """Create CUDA edge and LP-position tensors for the lattice polish gate.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Edge index shaped ``[2, E]`` and LP-like positions shaped ``[N, 2]``.
    """
    layer_width = 4
    num_layers = 5
    edges: list[tuple[int, int]] = []
    for layer in range(num_layers - 1):
        base = layer * layer_width
        nxt = (layer + 1) * layer_width
        for offset in range(layer_width):
            edges.append((base + offset, nxt + offset))
            if offset + 1 < layer_width:
                edges.append((base + offset, nxt + offset + 1))
    edge_index = torch.tensor(edges, dtype=torch.long, device="cuda").t().contiguous()
    positions = torch.zeros((layer_width * num_layers, 2), dtype=torch.float32, device="cuda")
    for layer in range(num_layers):
        start = layer * layer_width
        positions[start : start + layer_width, 0] = torch.arange(
            layer_width,
            dtype=torch.float32,
            device="cuda",
        )
        positions[start : start + layer_width, 1] = float(layer * 25)
    return edge_index, positions


def _make_shared_edge_executor(
    execution_device: str,
) -> tuple[
    SubsetGPUExecutor,
    torch.Tensor,
    list[Optional[torch.Tensor]],
    list[Optional[EdgeBatchContext]],
]:
    """Create a multi-term subset executor for shared edge-remap tests.

    Parameters
    ----------
    execution_device : str
        Device used by the subset executor.

    Returns
    -------
    tuple[SubsetGPUExecutor, torch.Tensor, list[torch.Tensor | None], list[EdgeBatchContext | None]]
        Configured executor, CPU position tensor shaped ``[N, 2]``, mutable
        edge reference, and mutable edge-context reference.
    """
    edge_index, num_nodes, node_sizes, _layers = _make_layered_dag(8, 32)
    pos = torch.randn(num_nodes, 2, dtype=torch.float32)
    edge_ref: list[Optional[torch.Tensor]] = [edge_index]
    edge_ctx_ref: list[Optional[EdgeBatchContext]] = [None]
    sampled_ref: list[Optional[SampledNodeContext]] = [None]
    executor = SubsetGPUExecutor(
        node_sizes=node_sizes,
        layer_index=None,
        execution_device=execution_device,
        batch_edges_ref=edge_ref,
        edge_ctx_ref=edge_ctx_ref,
        sampled_ctx_ref=sampled_ref,
        verbose=True,
    )
    return executor, pos, edge_ref, edge_ctx_ref


def _shared_edge_loss_terms(
    edge_ref: list[Optional[torch.Tensor]],
    edge_ctx_ref: list[Optional[EdgeBatchContext]],
    edge_index: torch.Tensor,
) -> list[SubsetGPULossTerm]:
    """Build edge-only loss terms that can share one remapped subset.

    Parameters
    ----------
    edge_ref : list[torch.Tensor | None]
        Mutable edge reference used by the loss closures.
    edge_ctx_ref : list[EdgeBatchContext | None]
        Mutable edge-context reference used by the loss closures.
    edge_index : torch.Tensor
        Global edge batch shaped ``[2, E]``.

    Returns
    -------
    list[SubsetGPULossTerm]
        Three edge-based loss terms that all reference the same edge object.
    """

    def attraction_loss(
        pos: torch.Tensor,
        node_sizes: torch.Tensor,
        layer_index: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the active edge-attraction loss."""
        del node_sizes, layer_index
        assert edge_ref[0] is not None
        return edge_attraction_loss(pos, edge_ref[0], x_bias=1.1, edge_ctx=edge_ctx_ref[0])

    def straightness_loss(
        pos: torch.Tensor,
        node_sizes: torch.Tensor,
        layer_index: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the active edge-straightness loss."""
        del node_sizes, layer_index
        assert edge_ref[0] is not None
        return edge_straightness_loss(pos, edge_ref[0], edge_ctx=edge_ctx_ref[0])

    def variance_loss(
        pos: torch.Tensor,
        node_sizes: torch.Tensor,
        layer_index: Optional[LayerIndex],
    ) -> torch.Tensor:
        """Return the active edge-length variance loss."""
        del node_sizes, layer_index
        assert edge_ref[0] is not None
        return edge_length_variance_loss(pos, edge_ref[0], edge_ctx=edge_ctx_ref[0])

    return [
        SubsetGPULossTerm(
            name="attract",
            weight=1.2,
            loss_fn=attraction_loss,
            access_pattern=EdgeAccessPattern(edge_index),
        ),
        SubsetGPULossTerm(
            name="straight",
            weight=0.4,
            loss_fn=straightness_loss,
            access_pattern=EdgeAccessPattern(edge_index),
        ),
        SubsetGPULossTerm(
            name="variance",
            weight=0.2,
            loss_fn=variance_loss,
            access_pattern=EdgeAccessPattern(edge_index),
        ),
    ]


def _assert_valid_coarsening(fine_to_coarse: torch.Tensor, num_nodes: int) -> None:
    """Assert that a fine-to-coarse assignment is well formed.

    Parameters
    ----------
    fine_to_coarse : torch.Tensor
        Fine-to-coarse assignment shaped ``[N]``.
    num_nodes : int
        Expected fine node count.

    Returns
    -------
    None
    """
    assert fine_to_coarse.shape == (num_nodes,)
    assert fine_to_coarse.min().item() == 0
    assert fine_to_coarse.max().item() < num_nodes


def _layer_sorted_x(pos: torch.Tensor, layers: torch.Tensor) -> torch.Tensor:
    """Return x coordinates sorted independently within each layer.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor shaped ``[N, 2]``.
    layers : torch.Tensor
        Layer assignments shaped ``[N]``.

    Returns
    -------
    torch.Tensor
        Concatenated x coordinates with a stable per-layer sort.
    """
    sorted_parts: list[torch.Tensor] = []
    num_layers = int(layers.max().item()) + 1 if layers.numel() > 0 else 0
    for layer_idx in range(num_layers):
        layer_mask = layers == layer_idx
        sorted_parts.append(pos[layer_mask, 0].sort().values)
    return torch.cat(sorted_parts) if sorted_parts else torch.zeros(0, dtype=pos.dtype)


@CUDA_REQUIRED
def test_gpu_coarsening_activates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """GPU streaming coarsening should emit a CUDA activation log when eligible."""
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    edge_index, num_nodes, node_sizes, layers = _make_layered_dag(24, 50)
    monkeypatch.setattr(multilevel_module, "_STREAMING_THRESHOLD", 1_000)

    result = coarsen_once(edge_index, num_nodes, node_sizes, layers, device="cpu")
    captured = capsys.readouterr()

    assert "Coarsen assignment" in captured.out
    assert "CUDA" in captured.out
    _assert_valid_coarsening(result.fine_to_coarse.cpu(), num_nodes)


@CUDA_REQUIRED
def test_gpu_layer_index_build_activates(capsys: pytest.CaptureFixture[str]) -> None:
    """LayerIndex should emit a CUDA activation log when the sort fits in VRAM."""
    layer_assignments = torch.arange(10_000, dtype=torch.long) % 50

    index = build_layer_index(layer_assignments, device="cpu", verbose=True)
    captured = capsys.readouterr()

    assert "LayerIndex" in captured.out
    assert "CUDA" in captured.out
    assert index.sorted_nodes.shape == (10_000,)
    assert index.layer_offsets.shape == (51,)
    assert index.sorted_nodes.device.type == "cpu"


@CUDA_REQUIRED
def test_gpu_layering_activates(capsys: pytest.CaptureFixture[str]) -> None:
    """CUDA longest-path layering should activate and match the CPU assignment."""
    edge_index, num_nodes, _node_sizes, _layers = _make_layered_dag(100, 50)

    cpu_layers = torch.as_tensor(longest_path_layering(edge_index, num_nodes, device="cpu"))
    gpu_layers = torch.as_tensor(
        longest_path_layering(edge_index, num_nodes, device="cuda", verbose=True)
    )
    captured = capsys.readouterr()

    assert "GPU layering" in captured.out
    assert "CUDA" in captured.out
    torch.testing.assert_close(gpu_layers, cpu_layers)


@CUDA_REQUIRED
def test_subset_gpu_mode_activates(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Subset-GPU execution should activate once the node threshold is crossed."""
    engine_module = importlib.import_module("dagua.layout.engine")

    edge_index, num_nodes, node_sizes, _layers = _make_layered_dag(20, 10)
    monkeypatch.setattr(engine_module, "SUBSET_GPU_REQUIRED_THRESHOLD", 100)

    pos = _layout_inner(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        config=LayoutConfig(
            steps=1,
            device="cuda",
            verbose=True,
            w_repel=0.0,
            w_overlap=0.0,
            w_crossing=0.0,
            w_spacing=0.0,
            w_fanout=0.0,
        ),
        device="cuda",
        skip_classification=True,
    )
    captured = capsys.readouterr()

    assert pos.shape == (num_nodes, 2)
    assert torch.isfinite(pos).all()
    assert "Execution mode: subset_gpu" in captured.out
    assert "strategy=[" in captured.out
    assert "subset_gpu" in captured.out


@CUDA_REQUIRED
def test_shared_edge_remap_activates(capsys: pytest.CaptureFixture[str]) -> None:
    """Shared edge remapping should emit a CUDA activation log and produce gradients."""
    executor, pos, edge_ref, edge_ctx_ref = _make_shared_edge_executor("cuda")
    edge_index = edge_ref[0]
    assert edge_index is not None

    grad_buffer = executor.compute_step(
        pos.clone().requires_grad_(True),
        _shared_edge_loss_terms(edge_ref, edge_ctx_ref, edge_index),
        verbose=True,
    )
    captured = capsys.readouterr()

    assert "Shared edge remap" in captured.out
    assert "CUDA" in captured.out
    assert grad_buffer.shape == pos.shape
    assert torch.isfinite(grad_buffer).all()


@CUDA_REQUIRED
def test_gpu_projection_activates(capsys: pytest.CaptureFixture[str]) -> None:
    """Layered overlap projection should emit a CUDA activation log on CUDA tensors."""
    pos, node_sizes, layer_index = _make_projection_case(6, 100)
    layer_index_cuda = build_layer_index(layer_index.node_to_layer.to("cuda"), device="cuda")

    projected = project_overlaps(
        pos.to("cuda"),
        node_sizes.to("cuda"),
        iterations=6,
        layer_index=layer_index_cuda,
    )
    captured = capsys.readouterr()

    assert "Overlap projection" in captured.out
    assert "CUDA" in captured.out
    assert torch.isfinite(projected).all()
    assert count_overlaps(projected.cpu(), node_sizes) <= count_overlaps(pos, node_sizes)


@CUDA_REQUIRED
def test_dynamic_edge_batch_uses_vram(capsys: pytest.CaptureFixture[str]) -> None:
    """The auto edge-batch helper should log a VRAM-derived CUDA batch size."""
    batch = _auto_edge_batch_size(verbose=True)
    captured = capsys.readouterr()

    assert "Auto" in captured.out
    assert "edge batch" in captured.out.lower()
    assert 1_000_000 <= batch <= 50_000_000


def test_coarsening_falls_back_on_low_vram(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Streaming coarsening should stay on CPU when mocked free VRAM is tiny."""
    layers_module = importlib.import_module("dagua.layout.layers")
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    edge_index, num_nodes, node_sizes, layers = _make_layered_dag(24, 50)
    monkeypatch.setattr(multilevel_module, "_STREAMING_THRESHOLD", 1_000)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_024, 1_024))

    def _fail_if_called(node_to_layer: torch.Tensor, output_device: str) -> torch.Tensor:
        """Fail the test if the CUDA LayerIndex sort is still attempted."""
        del node_to_layer, output_device
        raise AssertionError("CUDA LayerIndex sort should not run under low VRAM")

    monkeypatch.setattr(layers_module, "_cuda_layer_argsort", _fail_if_called)

    result = coarsen_once(edge_index, num_nodes, node_sizes, layers, device="cpu")
    captured = capsys.readouterr()

    assert "CPU fallback" in captured.out
    _assert_valid_coarsening(result.fine_to_coarse.cpu(), num_nodes)


def test_layer_index_falls_back_on_low_vram(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """LayerIndex should log CPU fallback when mocked free VRAM is insufficient."""
    layers_module = importlib.import_module("dagua.layout.layers")

    layer_assignments = torch.arange(10_000, dtype=torch.long) % 50
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_024, 1_024))

    def _fail_if_called(node_to_layer: torch.Tensor, output_device: str) -> torch.Tensor:
        """Fail the test if the CUDA argsort still launches."""
        del node_to_layer, output_device
        raise AssertionError("CUDA argsort should not launch under low VRAM")

    monkeypatch.setattr(layers_module, "_cuda_layer_argsort", _fail_if_called)

    index = build_layer_index(layer_assignments, device="cpu", verbose=True)
    captured = capsys.readouterr()

    assert "LayerIndex" in captured.out
    assert "CPU fallback" in captured.out
    assert index.sorted_nodes.shape == (10_000,)


def test_layering_falls_back_on_low_vram(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CUDA layering should log CPU fallback when the mocked VRAM budget is tiny."""
    edge_index, num_nodes, _node_sizes, _layers = _make_layered_dag(100, 20)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_024, 1_024))
    monkeypatch.setattr(torch.cuda, "memory_allocated", lambda: 0)
    monkeypatch.setattr(torch.cuda, "memory_reserved", lambda: 0)

    gpu_layers = torch.as_tensor(
        longest_path_layering(edge_index, num_nodes, device="cuda", verbose=True)
    )
    cpu_layers = torch.as_tensor(longest_path_layering(edge_index, num_nodes, device="cpu"))
    captured = capsys.readouterr()

    assert "GPU layering" in captured.out
    assert "CPU fallback" in captured.out
    torch.testing.assert_close(gpu_layers, cpu_layers)


@CUDA_REQUIRED
def test_shared_edge_remap_falls_back_on_low_vram(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Shared remap should fall back when its mocked CUDA VRAM budget is too small."""
    executor, pos, edge_ref, edge_ctx_ref = _make_shared_edge_executor("cuda")
    edge_index = edge_ref[0]
    assert edge_index is not None
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_024, 1_024))

    grad_buffer = executor.compute_step(
        pos.clone().requires_grad_(True),
        _shared_edge_loss_terms(edge_ref, edge_ctx_ref, edge_index),
        verbose=True,
    )
    captured = capsys.readouterr()

    assert "Shared edge remap" in captured.out
    assert "CPU fallback" in captured.out
    assert torch.isfinite(grad_buffer).all()


@CUDA_REQUIRED
def test_projection_falls_back_on_low_vram(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Layered projection should log CPU fallback when mocked VRAM is insufficient."""
    projection_module = importlib.import_module("dagua.layout.projection")

    pos, node_sizes, layer_index = _make_projection_case(6, 100)
    layer_index_cuda = build_layer_index(layer_index.node_to_layer.to("cuda"), device="cuda")
    monkeypatch.setattr(projection_module, "_vram_fits", lambda _needed_bytes: False)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (1_024, 1_024))

    projected = project_overlaps(
        pos.to("cuda"),
        node_sizes.to("cuda"),
        iterations=4,
        layer_index=layer_index_cuda,
    )
    captured = capsys.readouterr()

    assert "Overlap projection" in captured.out
    assert "CPU fallback" in captured.out
    assert projected.device.type == "cuda"
    assert torch.isfinite(projected).all()


def test_all_stages_fall_back_when_no_cuda(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Requesting CUDA on a no-CUDA runtime should still produce valid CPU results."""
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    edge_index, num_nodes, node_sizes, layers = _make_layered_dag(8, 16)
    monkeypatch.setattr(torch.cuda, "is_available", lambda: False)
    monkeypatch.setattr(multilevel_module, "_STREAMING_THRESHOLD", 100)

    layer_index = build_layer_index(layers, device="cpu", verbose=True)
    layering = torch.as_tensor(
        longest_path_layering(edge_index, num_nodes, device="cuda", verbose=True)
    )
    coarsened = coarsen_once(edge_index, num_nodes, node_sizes, layers, device="cpu")
    projected = project_overlaps(
        torch.zeros(num_nodes, 2, dtype=torch.float32),
        node_sizes,
        iterations=2,
        layer_index=layer_index,
    )
    layout_pos = _layout_inner(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        config=LayoutConfig(
            steps=1,
            device="cuda",
            verbose=True,
            w_repel=0.0,
            w_overlap=0.0,
            w_crossing=0.0,
            w_spacing=0.0,
            w_fanout=0.0,
        ),
        device="cuda",
        skip_classification=True,
    )
    captured = capsys.readouterr()

    assert layer_index.sorted_nodes.device.type == "cpu"
    assert layering.shape == (num_nodes,)
    _assert_valid_coarsening(coarsened.fine_to_coarse.cpu(), num_nodes)
    assert projected.device.type == "cpu"
    assert torch.isfinite(layout_pos).all()
    assert "CPU fallback" in captured.out or "no CUDA" in captured.out


@CUDA_REQUIRED
def test_coarsening_recovers_from_oom(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Streaming coarsening should catch CUDA OOM and finish on CPU."""
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    edge_index, num_nodes, node_sizes, layers = _make_layered_dag(24, 50)
    monkeypatch.setattr(multilevel_module, "_STREAMING_THRESHOLD", 1_000)
    original_sort = multilevel_module._stable_argsort_on_device
    state = {"raised": False}

    def _oom_once(values: torch.Tensor, compute_device: str) -> torch.Tensor:
        """Raise a CUDA OOM on the first CUDA sort attempt only."""
        if compute_device == "cuda" and not state["raised"]:
            state["raised"] = True
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return original_sort(values, compute_device)

    monkeypatch.setattr(multilevel_module, "_stable_argsort_on_device", _oom_once)

    result = coarsen_once(edge_index, num_nodes, node_sizes, layers, device="cpu")
    captured = capsys.readouterr()

    assert "OOM" in captured.out
    assert "CPU" in captured.out
    _assert_valid_coarsening(result.fine_to_coarse.cpu(), num_nodes)


@CUDA_REQUIRED
def test_layering_recovers_from_oom(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """CUDA layering should catch a mocked OOM and return the CPU result."""
    utils_module = importlib.import_module("dagua.utils")

    edge_index, num_nodes, _node_sizes, _layers = _make_layered_dag(100, 20)
    cpu_layers = torch.as_tensor(longest_path_layering(edge_index, num_nodes, device="cpu"))
    monkeypatch.setattr(
        utils_module,
        "_gpu_longest_path_layering",
        lambda _edge_index, _num_nodes: (_ for _ in ()).throw(
            torch.cuda.OutOfMemoryError("CUDA out of memory")
        ),
    )

    gpu_layers = torch.as_tensor(
        longest_path_layering(edge_index, num_nodes, device="cuda", verbose=True)
    )
    captured = capsys.readouterr()

    assert "OOM" in captured.out
    assert "CPU fallback" in captured.out
    torch.testing.assert_close(gpu_layers, cpu_layers)


@CUDA_REQUIRED
def test_shared_edge_remap_recovers_from_oom(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    """Shared edge remap should catch OOM and continue via per-term remapping."""
    subset_gpu_module = importlib.import_module("dagua.layout.subset_gpu")

    executor, pos, edge_ref, edge_ctx_ref = _make_shared_edge_executor("cuda")
    edge_index = edge_ref[0]
    assert edge_index is not None
    original_build = subset_gpu_module._build_local_edge_context
    state = {"raised": False}

    def _oom_once(pos_local: torch.Tensor, local_edges: torch.Tensor) -> EdgeBatchContext:
        """Raise CUDA OOM on the first shared edge-context build only."""
        if not state["raised"]:
            state["raised"] = True
            raise torch.cuda.OutOfMemoryError("CUDA out of memory")
        return original_build(pos_local, local_edges)

    monkeypatch.setattr(subset_gpu_module, "_build_local_edge_context", _oom_once)

    grad_buffer = executor.compute_step(
        pos.clone().requires_grad_(True),
        _shared_edge_loss_terms(edge_ref, edge_ctx_ref, edge_index),
        verbose=True,
    )
    captured = capsys.readouterr()

    assert "Shared edge remap" in captured.out
    assert "OOM" in captured.out
    assert torch.isfinite(grad_buffer).all()


def test_gpu_coarsening_matches_cpu_exactly(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Simulated CUDA streaming coarsening should match the CPU assignment exactly."""
    layers_module = importlib.import_module("dagua.layout.layers")
    multilevel_module = importlib.import_module("dagua.layout.multilevel")

    edge_index, num_nodes, node_sizes, layers = _make_layered_dag(20, 30)
    layer_counts = torch.bincount(layers, minlength=20)
    layer_offsets = torch.zeros(21, dtype=torch.long)
    layer_offsets[1:] = layer_counts.cumsum(0)
    min_neighbor = torch.arange(num_nodes, dtype=torch.long) % 17

    monkeypatch.setattr(
        multilevel_module,
        "_build_streaming_min_neighbor",
        lambda edge_index, num_nodes, index_dtype, gpu_device: (
            min_neighbor.to(dtype=index_dtype),
            False,
            0.0,
        ),
    )
    monkeypatch.setattr(
        multilevel_module,
        "_deduplicate_streaming_coarse_edges",
        lambda edge_index, fine_to_coarse, num_coarse_nodes, output_device, gpu_device: (
            torch.zeros((2, 0), dtype=torch.long, device=output_device),
            False,
            0.0,
        ),
    )
    cpu_result = multilevel_module._coarsen_once_streaming(
        edge_index=edge_index,
        N=num_nodes,
        node_sizes=node_sizes,
        layers=layers,
        num_layers=20,
        layer_counts=layer_counts,
        layer_offsets=layer_offsets,
        device="cpu",
    )

    monkeypatch.setattr(torch.cuda, "is_available", lambda: True)
    monkeypatch.setattr(torch.cuda, "mem_get_info", lambda: (10_000_000_000, 20_000_000_000))
    monkeypatch.setattr(
        layers_module,
        "_cuda_layer_argsort",
        lambda node_to_layer, output_device: node_to_layer.argsort(stable=True).to(output_device),
    )
    monkeypatch.setattr(
        multilevel_module,
        "_stable_argsort_on_device",
        lambda values, compute_device: values.argsort(stable=True),
    )
    gpu_result = multilevel_module._coarsen_once_streaming(
        edge_index=edge_index,
        N=num_nodes,
        node_sizes=node_sizes,
        layers=layers,
        num_layers=20,
        layer_counts=layer_counts,
        layer_offsets=layer_offsets,
        device="cpu",
    )

    torch.testing.assert_close(cpu_result.fine_to_coarse.cpu(), gpu_result.fine_to_coarse.cpu())


@CUDA_REQUIRED
def test_cuda_component_tiling_polish_keeps_edges_on_device() -> None:
    """Component tiling polish should use CUDA graph tensors end to end."""
    graph = _make_disconnected_polish_graph()
    pos = layout(graph, LayoutConfig(seed=42, device="cuda", steps=1))

    assert pos.device.type == "cuda"
    assert pos.shape == (graph.num_nodes, 2)
    assert bool(torch.isfinite(pos).all().item())


@CUDA_REQUIRED
def test_cuda_lattice_uniform_slots_gate_keeps_degree_on_device() -> None:
    """The lattice polish gate should not mix CPU degree tensors with CUDA edges."""
    edge_index, positions = _make_cuda_lattice_gate_case()

    assert _should_lattice_uniform_centered_slots(edge_index, positions.shape[0], positions)


@CUDA_REQUIRED
def test_gpu_layering_matches_cpu_exactly() -> None:
    """CUDA longest-path layering should match the CPU layering exactly."""
    edge_index, num_nodes, _node_sizes, _layers = _make_layered_dag(40, 50)
    cpu_layers = torch.as_tensor(longest_path_layering(edge_index, num_nodes, device="cpu"))
    gpu_layers = torch.as_tensor(longest_path_layering(edge_index, num_nodes, device="cuda"))

    torch.testing.assert_close(gpu_layers, cpu_layers)


@CUDA_REQUIRED
def test_gpu_projection_matches_cpu_closely() -> None:
    """CUDA layered projection should stay numerically close to the CPU sweep."""
    pos, node_sizes, layer_index = _make_projection_case(6, 100)
    cpu_pos = project_overlaps(
        pos.clone(),
        node_sizes.clone(),
        iterations=6,
        layer_index=layer_index,
    )
    gpu_pos = project_overlaps(
        pos.clone().to("cuda"),
        node_sizes.clone().to("cuda"),
        iterations=6,
        layer_index=build_layer_index(layer_index.node_to_layer.to("cuda"), device="cuda"),
    ).cpu()

    torch.testing.assert_close(gpu_pos, cpu_pos, atol=0.1, rtol=0.0)


@CUDA_REQUIRED
def test_end_to_end_layout_quality_unchanged() -> None:
    """A small end-to-end CUDA layout should stay close to the CPU solve."""
    edge_index, num_nodes, node_sizes, layers = _make_layered_dag(25, 20)
    config = LayoutConfig(
        steps=2,
        verbose=False,
        w_repel=0.0,
        w_overlap=0.0,
        w_crossing=0.0,
        w_spacing=0.0,
        w_fanout=0.0,
    )

    cpu_pos = _layout_inner(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        config=config,
        device="cpu",
        skip_classification=True,
    )
    gpu_pos = _layout_inner(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        config=config,
        device="cuda",
        skip_classification=True,
    ).cpu()

    cpu_loss = float(edge_attraction_loss(cpu_pos, edge_index, x_bias=1.0).item())
    gpu_loss = float(edge_attraction_loss(gpu_pos, edge_index, x_bias=1.0).item())

    assert torch.allclose(
        _layer_sorted_x(gpu_pos, layers),
        _layer_sorted_x(cpu_pos, layers),
        atol=100.0,
        rtol=0.0,
    )
    assert gpu_loss <= cpu_loss * 1.25 + 1e-3


def test_execution_mode_selection_thresholds() -> None:
    """Execution-mode threshold logic should select subset-GPU only when expected."""
    config = LayoutConfig(device="cuda")

    assert _resolve_execution_mode(config, "cuda", 1_000_000) == "standard"
    assert _resolve_execution_mode(config, "cuda", 50_000_001) == "subset_gpu"
    assert _resolve_execution_mode(config, "cpu", 200_000_000) == "standard"


def test_dynamic_allocation_functions() -> None:
    """Auto-sizing helpers should always stay within their documented bounds."""
    edge_batch = _auto_edge_batch_size()
    cpu_batch = _auto_cpu_edge_batch_size()

    assert 1_000_000 <= edge_batch <= 50_000_000
    assert 500_000 <= cpu_batch <= 20_000_000

    sampled_cap = _auto_sampled_node_cap()
    assert 10_000 <= sampled_cap <= 2_000_000
