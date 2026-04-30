"""Regression tests for Kamada-Kawai NetworkX fidelity adapter semantics."""

from __future__ import annotations

import importlib
from typing import Any

import pytest
import torch

from dagua.eval.competitors.classic_competitor import ClassicKK
from dagua.graph import DaguaGraph
from dagua.layout.ops.distance import KamadaKawaiAllPairsShortestPaths
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
