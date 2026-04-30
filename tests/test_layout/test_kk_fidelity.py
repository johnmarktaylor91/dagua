"""Regression tests for Kamada-Kawai NetworkX fidelity adapter semantics."""

from __future__ import annotations

import importlib
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest
import torch

from dagua.eval.competitors.classic_competitor import ClassicKK
from dagua.eval.competitors.networkx_competitor import NetworkXKamadaKawai
from dagua.graph import DaguaGraph
from dagua.layout.ops.distance import KamadaKawaiAllPairsShortestPaths
from dagua.layout.ops.pipelines.kk import layout_kk_pipeline
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _small_graph() -> DaguaGraph:
    """Create a minimal graph for KK adapter tests.

    Returns
    -------
    DaguaGraph
        Two-node directed graph with computed node sizes.
    """
    graph = DaguaGraph.from_edge_list([("a", "b")])
    graph.compute_node_sizes()
    return graph


def _kk_distance_matrix(edge_weights: torch.Tensor) -> torch.Tensor:
    """Compute KK directed distances for duplicate weighted edges.

    Parameters
    ----------
    edge_weights : torch.Tensor
        Duplicate edge weights with shape ``[2]``.

    Returns
    -------
    torch.Tensor
        KK all-pairs distance matrix with shape ``[2, 2]``.
    """
    edge_index = torch.tensor([[0, 0], [1, 1]], dtype=torch.long)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=2,
        edge_weights=edge_weights,
    )
    state = KamadaKawaiAllPairsShortestPaths().apply(
        problem=problem,
        state=SolveState(),
        ctx=RuntimeContext(),
    )
    assert state.distance_matrix is not None
    return state.distance_matrix


def test_classic_kk_layout_uses_networkx_fidelity_defaults(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The direct KK competitor should call the pipeline with reference defaults."""
    kk_pipeline = importlib.import_module("dagua.layout.ops.pipelines.kk")

    observed: dict[str, Any] = {}

    def fake_layout_kk_pipeline(*args: Any, **kwargs: Any) -> torch.Tensor:
        """Capture KK adapter arguments and return a valid position tensor.

        Parameters
        ----------
        *args : Any
            Positional pipeline arguments.
        **kwargs : Any
            Keyword pipeline arguments.

        Returns
        -------
        torch.Tensor
            Zero positions with shape ``[2, 2]``.
        """
        del args
        observed.update(kwargs)
        return torch.zeros((2, 2), dtype=torch.float32)

    monkeypatch.setattr(kk_pipeline, "layout_kk_pipeline", fake_layout_kk_pipeline)

    result = ClassicKK().layout(_small_graph(), seed=42)

    assert result.error is None
    assert observed["steps"] is None
    assert observed["orient_to_direction"] is False


def test_classic_kk_variant_defaults_use_networkx_fidelity_path(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Variant dispatch should not reintroduce capped iterations or orientation."""
    kk_pipeline = importlib.import_module("dagua.layout.ops.pipelines.kk")

    observed: dict[str, Any] = {}

    def fake_layout_kk_pipeline(*args: Any, **kwargs: Any) -> torch.Tensor:
        """Capture variant-dispatch arguments and return positions.

        Parameters
        ----------
        *args : Any
            Positional pipeline arguments.
        **kwargs : Any
            Keyword pipeline arguments.

        Returns
        -------
        torch.Tensor
            Zero positions with shape ``[2, 2]``.
        """
        del args
        observed.update(kwargs)
        return torch.zeros((2, 2), dtype=torch.float32)

    monkeypatch.setattr(kk_pipeline, "layout_kk_pipeline", fake_layout_kk_pipeline)

    result = ClassicKK().layout_with_variant(_small_graph(), seed=42, variant_params={})

    assert result.error is None
    assert observed["steps"] is None
    assert observed["orient_to_direction"] is False


def test_kk_weighted_duplicate_edges_follow_networkx_last_write() -> None:
    """KK weighted duplicate edge collapse should match ``nx.DiGraph`` insertion."""
    first_last = _kk_distance_matrix(torch.tensor([10.0, 1.0], dtype=torch.float64))
    second_last = _kk_distance_matrix(torch.tensor([1.0, 10.0], dtype=torch.float64))

    assert float(first_last[0, 1].item()) == pytest.approx(1.0)
    assert float(second_last[0, 1].item()) == pytest.approx(10.0)


def test_networkx_kk_adapter_uses_last_duplicate_and_unit_scale(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The KK reference adapter should keep literal NetworkX scale and duplicates."""
    networkx = pytest.importorskip("networkx")
    graph = DaguaGraph.from_edge_list([("a", "b"), ("a", "b")])
    graph.compute_node_sizes()
    graph.edge_weights = torch.tensor([10.0, 1.0], dtype=torch.float32)

    observed: dict[str, float] = {}

    def fake_kamada_kawai_layout(nx_graph: Any, **kwargs: Any) -> dict[int, tuple[float, float]]:
        """Capture the copied edge weight and return unscaled coordinates.

        Parameters
        ----------
        nx_graph : Any
            NetworkX graph passed by the adapter.
        **kwargs : Any
            Layout keyword arguments.

        Returns
        -------
        dict[int, tuple[float, float]]
            Minimal deterministic position mapping.
        """
        del kwargs
        observed["weight"] = float(nx_graph[0][1]["weight"])
        return {0: (2.0, 3.0), 1: (-1.0, 4.0)}

    monkeypatch.setattr(networkx, "kamada_kawai_layout", fake_kamada_kawai_layout)

    result = NetworkXKamadaKawai().layout(graph)

    assert result.error is None
    assert observed["weight"] == pytest.approx(1.0)
    assert result.pos is not None
    assert result.pos.dtype == torch.float32
    assert result.pos[0].tolist() == pytest.approx([2.0, 3.0])


def test_networkx_kk_adapter_supports_float64_variant(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The NetworkX side should offer the same dtype audit mode as Dagua KK."""
    networkx = pytest.importorskip("networkx")

    def fake_kamada_kawai_layout(nx_graph: Any, **kwargs: Any) -> dict[int, tuple[float, float]]:
        """Return deterministic positions for dtype conversion tests.

        Parameters
        ----------
        nx_graph : Any
            NetworkX graph passed by the adapter.
        **kwargs : Any
            Layout keyword arguments.

        Returns
        -------
        dict[int, tuple[float, float]]
            Minimal deterministic position mapping.
        """
        del nx_graph, kwargs
        return {0: (0.25, -0.5), 1: (0.75, 1.5)}

    monkeypatch.setattr(networkx, "kamada_kawai_layout", fake_kamada_kawai_layout)

    result = NetworkXKamadaKawai().layout_with_variant(
        _small_graph(),
        variant_params={"output_dtype": "float64"},
    )

    assert result.error is None
    assert result.pos is not None
    assert result.pos.dtype == torch.float64
    assert result.pos[1].tolist() == pytest.approx([0.75, 1.5])


def test_layout_kk_pipeline_can_preserve_float64() -> None:
    """The Dagua KK pipeline should keep float64 only when explicitly requested."""
    pytest.importorskip("scipy")
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)

    default = layout_kk_pipeline(edge_index=edge_index, num_nodes=3, steps=5)
    precise = layout_kk_pipeline(
        edge_index=edge_index,
        num_nodes=3,
        steps=5,
        preserve_float64=True,
    )

    assert default.dtype == torch.float32
    assert precise.dtype == torch.float64
    torch.testing.assert_close(precise.to(dtype=torch.float32), default)


def test_layout_kk_pipeline_rejects_negative_weights() -> None:
    """Negative weighted path lengths are invalid for KK Dijkstra distances."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)

    with pytest.raises(ValueError, match="nonnegative"):
        layout_kk_pipeline(
            edge_index=edge_index,
            num_nodes=2,
            edge_weights=torch.tensor([-1.0], dtype=torch.float32),
        )


def test_kk_distance_op_rejects_negative_weights() -> None:
    """The raw KK distance op should also guard Dijkstra's nonnegative contract."""
    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        edge_weights=torch.tensor([-1.0], dtype=torch.float32),
    )

    with pytest.raises(ValueError, match="nonnegative"):
        KamadaKawaiAllPairsShortestPaths().apply(
            problem=problem,
            state=SolveState(),
            ctx=RuntimeContext(),
        )


def test_layout_kk_pipeline_forwards_capped_iterations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit capped KK variants should still forward SciPy ``maxiter``."""
    scipy = pytest.importorskip("scipy")
    captured_kwargs: dict[str, Any] = {}

    def fake_minimize(
        objective: Any,
        initial_vector: np.ndarray,
        **kwargs: Any,
    ) -> SimpleNamespace:
        """Capture SciPy optimizer kwargs without running L-BFGS-B.

        Parameters
        ----------
        objective : Any
            Objective callable passed to SciPy.
        initial_vector : numpy.ndarray
            Initial flattened coordinates.
        **kwargs : Any
            Optimizer keyword arguments.

        Returns
        -------
        types.SimpleNamespace
            Minimal optimizer result exposing ``x``.
        """
        del objective
        captured_kwargs.update(kwargs)
        return SimpleNamespace(x=np.asarray(initial_vector, dtype=np.float64), fun=0.0)

    monkeypatch.setattr(scipy.optimize, "minimize", fake_minimize)

    result = layout_kk_pipeline(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        steps=7,
    )

    assert result.shape == (2, 2)
    assert captured_kwargs["options"] == {"maxiter": 7}
