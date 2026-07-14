"""Conformance tests for public Graphviz fdp/dot algorithm names."""

from __future__ import annotations

from typing import Optional

import pytest
import torch

import dagua
from dagua.eval.benchmark import DEFAULT_COMPETITOR_ORDER
from dagua.eval.competitors import get_competitor
from dagua.eval.competitors.classic_competitor import Dot, Fdp
from dagua.eval.competitors.graphviz_competitor import GraphvizDot, GraphvizFdp
from dagua.eval.equivalence_metrics import anisotropic_procrustes
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY
from dagua.layout.ops.pipelines.dot import layout_dot_pipeline
from dagua.layout.ops.pipelines.fdp import layout_fdp_pipeline
from dagua.layout.ops.pipelines.fmmm import layout_fmmm_pipeline
from dagua.layout.ops.pipelines.sugiyama import layout_sugiyama_pipeline


def _graph(edges: list[tuple[str, str]]) -> dagua.DaguaGraph:
    """Build a graph with computed node sizes for Graphviz-fidelity tests.

    Parameters
    ----------
    edges : list[tuple[str, str]]
        Edge list expressed with stable node labels.

    Returns
    -------
    dagua.DaguaGraph
        Graph with ``node_sizes`` populated.
    """
    graph = dagua.DaguaGraph.from_edge_list(edges)
    graph.compute_node_sizes()
    return graph


def _assert_successful_result(result_name: str, position: Optional[torch.Tensor]) -> torch.Tensor:
    """Return a result tensor or fail with an actionable message.

    Parameters
    ----------
    result_name : str
        Name of the competitor result being checked.
    position : torch.Tensor or None
        Position tensor returned by a competitor.

    Returns
    -------
    torch.Tensor
        Non-``None`` position tensor.
    """
    assert position is not None, f"{result_name} did not return positions"
    assert torch.isfinite(position).all(), f"{result_name} returned non-finite positions"
    return position


def _anisotropic_residual(left: torch.Tensor, right: torch.Tensor) -> float:
    """Compute the project-standard anisotropic Procrustes residual.

    Parameters
    ----------
    left : torch.Tensor
        First coordinate tensor with shape ``[N, 2]``.
    right : torch.Tensor
        Reference coordinate tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Free-aspect aligned residual.
    """
    return float(anisotropic_procrustes(left, right)["anisotropic_rmsd"])


def test_fdp_and_dot_pipeline_registry_entries() -> None:
    """The public pipeline registry exposes ``fdp`` and ``dot`` names.

    Returns
    -------
    None
        The test asserts registry metadata.
    """
    assert PIPELINE_REGISTRY["fdp"] == (
        "dagua.layout.ops.pipelines.fdp",
        "layout_fdp_pipeline",
    )
    assert PIPELINE_REGISTRY["dot"] == (
        "dagua.layout.ops.pipelines.dot",
        "layout_dot_pipeline",
    )


def test_public_fdp_and_dot_layout_dispatch() -> None:
    """``LayoutConfig(algorithm=...)`` dispatches both public names.

    Returns
    -------
    None
        The test asserts finite public layout coordinates.
    """
    graph = _graph([("a", "b"), ("a", "c")])

    fdp_pos = dagua.layout(graph, dagua.LayoutConfig(algorithm="fdp", seed=42))
    dot_pos = dagua.layout(graph, dagua.LayoutConfig(algorithm="dot", seed=42))

    assert fdp_pos.shape == (3, 2)
    assert dot_pos.shape == (3, 2)
    assert torch.isfinite(fdp_pos).all()
    assert torch.isfinite(dot_pos).all()


def test_fdp_and_dot_match_internal_fidelity_branches_exactly() -> None:
    """Public wrappers are exact aliases for the internal fidelity branches.

    Returns
    -------
    None
        The test asserts coordinate equality.
    """
    graph = _graph([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])

    public_fdp = layout_fdp_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        seed=42,
    )
    internal_fdp = layout_fmmm_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        seed=42,
        fidelity_mode="graphviz_fdp",
    )
    public_dot = layout_dot_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        seed=42,
    )
    internal_dot = layout_sugiyama_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        seed=42,
        fidelity_mode="graphviz",
    )

    assert torch.equal(public_fdp, internal_fdp)
    assert torch.equal(public_dot, internal_dot)


def test_fdp_and_dot_are_selectable_benchmark_competitors() -> None:
    """The benchmark registry exposes direct Dagua ``fdp`` and ``dot`` engines.

    Returns
    -------
    None
        The test asserts competitor discovery and default ordering.
    """
    fdp = get_competitor("fdp")
    dot = get_competitor("dot")

    assert isinstance(fdp, Fdp)
    assert isinstance(dot, Dot)
    assert "fdp" in DEFAULT_COMPETITOR_ORDER
    assert "dot" in DEFAULT_COMPETITOR_ORDER


@pytest.mark.parametrize(
    ("case_name", "edges", "max_residual"),
    [
        ("path3", [("a", "b"), ("b", "c")], 0.04),
        ("fork", [("a", "b"), ("a", "c")], 0.05),
        ("diamond", [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")], 0.07),
    ],
)
def test_fdp_conformance_against_graphviz_reference(
    case_name: str,
    edges: list[tuple[str, str]],
    max_residual: float,
) -> None:
    """Public FDP remains within the established Graphviz-FDP fidelity envelope.

    Parameters
    ----------
    case_name : str
        Human-readable small-graph case name.
    edges : list[tuple[str, str]]
        Graph edge list.
    max_residual : float
        Maximum allowed anisotropic Procrustes residual.

    Returns
    -------
    None
        The test asserts the residual threshold.
    """
    graph = _graph(edges)
    actual = Fdp().layout(graph, seed=42)
    reference = GraphvizFdp().layout(graph, seed=42)

    actual_pos = _assert_successful_result(actual.name, actual.pos)
    reference_pos = _assert_successful_result(reference.name, reference.pos)
    residual = _anisotropic_residual(actual_pos, reference_pos)

    assert residual < max_residual, f"{case_name}: anisotropic residual={residual:.6g}"


@pytest.mark.parametrize(
    ("case_name", "edges"),
    [
        ("path3", [("a", "b"), ("b", "c")]),
        ("fork", [("a", "b"), ("a", "c")]),
        ("diamond", [("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]),
    ],
)
def test_dot_conformance_against_graphviz_reference(
    case_name: str,
    edges: list[tuple[str, str]],
) -> None:
    """Public DOT matches Graphviz DOT up to project-standard free-aspect alignment.

    Parameters
    ----------
    case_name : str
        Human-readable small-graph case name.
    edges : list[tuple[str, str]]
        Graph edge list.

    Returns
    -------
    None
        The test asserts the anisotropic residual.
    """
    graph = _graph(edges)
    actual = Dot().layout(graph)
    reference = GraphvizDot().layout(graph)

    actual_pos = _assert_successful_result(actual.name, actual.pos)
    reference_pos = _assert_successful_result(reference.name, reference.pos)
    residual = _anisotropic_residual(actual_pos, reference_pos)

    assert residual < 1.0e-12, f"{case_name}: anisotropic residual={residual:.6g}"
