"""Regression tests for dagua_native CUDA preparation fallback."""

from __future__ import annotations

from typing import Optional

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.ops.pipelines import dagua_native as dagua_native_module


def _make_fourteen_node_graph() -> DaguaGraph:
    """Return a small DAG matching the benchmark OOM failure scale.

    Returns
    -------
    DaguaGraph
        Fourteen-node graph with fifteen edges.
    """
    edge_index = torch.tensor(
        [
            [0, 0, 1, 1, 2, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
            [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 12, 13],
        ],
        dtype=torch.long,
    )
    graph = DaguaGraph.from_edge_index(edge_index, num_nodes=14)
    graph.node_sizes = torch.full((14, 2), 20.0, dtype=torch.float32)
    return graph


def test_dagua_native_fourteen_node_cpu_layout_returns_positions() -> None:
    """Native layout should produce finite positions for tiny graphs on CPU."""
    graph = _make_fourteen_node_graph()

    pos = layout(graph, LayoutConfig(algorithm="dagua_native", device="cpu", seed=42, steps=2))

    assert pos.device.type == "cpu"
    assert pos.shape == (14, 2)
    assert bool(torch.isfinite(pos).all().item())


def test_native_tensor_preparation_falls_back_to_cpu_on_cuda_oom(monkeypatch) -> None:
    """CUDA preparation OOM should fall back before native layout work starts."""
    graph = _make_fourteen_node_graph()
    calls: list[torch.device] = []
    original_normalize = dagua_native_module.normalize_node_sizes

    def _oom_once_then_normalize(
        node_sizes: Optional[torch.Tensor],
        device: torch.device,
    ) -> torch.Tensor:
        """Simulate a CUDA context/preallocation OOM during tensor preparation.

        Parameters
        ----------
        node_sizes : torch.Tensor, optional
            Node sizes with shape ``[N, 2]``.
        device : torch.device
            Requested tensor preparation device.

        Returns
        -------
        torch.Tensor
            Normalized node sizes on ``device`` after the simulated CUDA
            failure has forced a CPU retry.
        """
        calls.append(device)
        if device.type == "cuda":
            raise RuntimeError("CUDA driver error: out of memory")
        return original_normalize(node_sizes=node_sizes, device=device)

    monkeypatch.setattr(dagua_native_module, "normalize_node_sizes", _oom_once_then_normalize)

    (
        effective_device,
        prepared_node_sizes,
        prepared_edge_index,
        prepared_init_pos,
        prepared_edge_weights,
        prepared_layer_assignments,
    ) = dagua_native_module._prepare_native_tensors_for_device(
        edge_index=graph.edge_index,
        node_sizes=graph.node_sizes,
        init_pos=None,
        edge_weights=None,
        layer_assignments=None,
        target_device=torch.device("cuda"),
    )

    assert [device.type for device in calls] == ["cuda", "cpu"]
    assert effective_device.type == "cpu"
    assert prepared_node_sizes.shape == (14, 2)
    assert prepared_edge_index.shape == (2, 15)
    assert prepared_init_pos is None
    assert prepared_edge_weights is None
    assert prepared_layer_assignments is None
