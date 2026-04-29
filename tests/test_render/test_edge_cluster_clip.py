"""Cluster-boundary edge clipping tests."""

from __future__ import annotations

from typing import Any, Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pytest
import torch

from dagua.config import LayoutConfig
from dagua.edges import BezierCurve, route_edges
from dagua.graph import DaguaGraph
from dagua.render import mpl as mpl_renderer
from dagua.render.edges import DaguaEdgeCollection
from dagua.styles import ClusterStyle


def _prepared_collection(
    graph: DaguaGraph,
    positions: torch.Tensor,
    *,
    cluster_aware: bool = True,
) -> Tuple[DaguaEdgeCollection, Dict[str, Tuple[float, float, float, float]]]:
    """Build a prepared edge collection for a fixed-position graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph with clusters and one or more edges.
    positions : torch.Tensor
        Node center positions with shape ``[N, 2]``.
    cluster_aware : bool, default=True
        Whether to pass cluster clipping data into the edge collection.

    Returns
    -------
    tuple[DaguaEdgeCollection, dict[str, tuple[float, float, float, float]]]
        Prepared edge collection and render cluster bboxes.
    """
    graph.compute_node_sizes()
    pos = positions.detach().cpu().numpy()
    sizes = graph.node_sizes.detach().cpu().numpy()
    curves = route_edges(positions, graph.edge_index, graph.node_sizes, graph.direction, graph)
    fig, ax = plt.subplots()
    ax.set_xlim(-50.0, 150.0)
    ax.set_ylim(-80.0, 80.0)
    membership: Dict[int, List[str]] = {}
    bboxes: Dict[str, Tuple[float, float, float, float]] = {}
    if cluster_aware:
        membership, bboxes = mpl_renderer._render_cluster_edge_clip_data(ax, graph, pos, sizes)
    collection = mpl_renderer._build_custom_edge_collection(
        ax,
        graph,
        curves,
        positions=pos,
        cluster_membership=membership,
        cluster_bboxes=bboxes,
    )
    plt.close(fig)
    return collection, bboxes


def _external_to_cluster_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Return a fixed graph with one external-to-internal cluster edge.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and node positions with shape ``[2, 2]``.
    """
    graph = DaguaGraph(direction="LR")
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge("A", "B")
    graph.add_cluster("C", ["B"], style=ClusterStyle(padding=12.0, label_position="top-center"))
    positions = torch.tensor([[0.0, 0.0], [100.0, 0.0]], dtype=torch.float32)
    return graph, positions


def test_external_to_internal_edge_body_terminates_at_cluster_perimeter() -> None:
    """External-to-internal edge bodies should stop at the cluster bbox edge."""
    graph, positions = _external_to_cluster_graph()

    collection, bboxes = _prepared_collection(graph, positions)

    body_curve = collection.prepared_edges[0].body_curve
    assert body_curve is not None
    assert body_curve.p1[0] == pytest.approx(bboxes["C"][0], abs=0.75)
    assert body_curve.p1[1] == pytest.approx(0.0, abs=0.75)
    assert collection.prepared_edges[0].head_result is not None
    assert collection.prepared_edges[0].edge.curve.p1[0] == pytest.approx(
        body_curve.p1[0],
        abs=0.75,
    )


def test_nested_cluster_edge_clips_at_outer_perimeter() -> None:
    """Nested external-to-internal edges should clip at the outer cluster."""
    graph = DaguaGraph(direction="LR")
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge("A", "B")
    graph.add_cluster(
        "outer",
        ["B"],
        style=ClusterStyle(padding=22.0, label_position="top-center"),
    )
    graph.add_cluster(
        "inner",
        ["B"],
        parent="outer",
        style=ClusterStyle(padding=8.0, label_position="top-center"),
    )
    positions = torch.tensor([[0.0, 0.0], [100.0, 0.0]], dtype=torch.float32)

    collection, bboxes = _prepared_collection(graph, positions)

    body_curve = collection.prepared_edges[0].body_curve
    assert body_curve is not None
    assert body_curve.p1[0] == pytest.approx(bboxes["outer"][0], abs=0.75)
    assert body_curve.p1[0] < bboxes["inner"][0]


def test_cross_cluster_edge_body_keeps_visible_span_between_perimeters() -> None:
    """Edges between sibling clusters should keep the body between perimeters."""
    graph = DaguaGraph(direction="LR")
    graph.add_node("A")
    graph.add_node("B")
    graph.add_edge("A", "B")
    graph.add_cluster(
        "source",
        ["A"],
        style=ClusterStyle(padding=12.0, label_position="top-center"),
    )
    graph.add_cluster(
        "target",
        ["B"],
        style=ClusterStyle(padding=12.0, label_position="top-center"),
    )
    positions = torch.tensor([[0.0, 0.0], [120.0, 0.0]], dtype=torch.float32)

    collection, bboxes = _prepared_collection(graph, positions)

    body_curve = collection.prepared_edges[0].body_curve
    assert body_curve is not None
    assert body_curve.p0[0] == pytest.approx(bboxes["source"][2], abs=0.75)
    assert body_curve.p1[0] == pytest.approx(bboxes["target"][0], abs=0.75)
    assert body_curve.p1[0] - body_curve.p0[0] > 40.0


def test_bypass_edge_body_splits_around_foreign_cluster() -> None:
    """Edges that pass through an unrelated cluster should render with a gap."""
    graph = DaguaGraph(direction="LR")
    graph.add_node("A")
    graph.add_node("B")
    graph.add_node("C")
    graph.add_edge("A", "C")
    graph.add_cluster(
        "foreign",
        ["B"],
        style=ClusterStyle(padding=12.0, label_position="top-center"),
    )
    positions = torch.tensor([[0.0, 0.0], [60.0, 0.0], [120.0, 0.0]], dtype=torch.float32)

    collection, bboxes = _prepared_collection(graph, positions)

    body_edges = [
        prepared
        for prepared in collection.prepared_edges
        if prepared.edge.arrowhead == "none" and prepared.edge.tail_arrow == "none"
    ]
    assert len(body_edges) >= 2
    assert body_edges[0].body_curve is not None
    assert body_edges[-1].body_curve is not None
    assert body_edges[0].body_curve.p1[0] == pytest.approx(bboxes["foreign"][0], abs=1.0)
    assert body_edges[-1].body_curve.p0[0] == pytest.approx(bboxes["foreign"][2], abs=1.0)


def test_cluster_aware_false_preserves_unclipped_edge_body() -> None:
    """Disabled cluster awareness should preserve legacy full edge bodies."""
    graph, positions = _external_to_cluster_graph()

    collection, bboxes = _prepared_collection(graph, positions, cluster_aware=False)

    body_curve = collection.prepared_edges[0].body_curve
    assert body_curve is not None
    assert body_curve.p1[0] > bboxes.get("C", (0.0, 0.0, 0.0, 0.0))[0]
    assert collection.prepared_edges[0].edge.body_curve is None


def test_render_respects_cluster_aware_false_gate(monkeypatch: pytest.MonkeyPatch) -> None:
    """Render should not pass clipping bboxes when ``cluster_aware`` is false."""
    graph, positions = _external_to_cluster_graph()
    observed: List[Dict[str, Any]] = []

    def record_draw_edges(
        ax: Any,
        graph: Any,
        curves: List[BezierCurve],
        positions: Any = None,
        svg_hover_map: Any = None,
        cluster_membership: Any = None,
        cluster_bboxes: Any = None,
    ) -> None:
        """Record render-time clipping inputs instead of drawing edges.

        Parameters
        ----------
        ax : Any
            Matplotlib axes.
        graph : Any
            Graph being rendered.
        curves : list[BezierCurve]
            Routed edge curves.
        positions : Any, default=None
            Render positions.
        svg_hover_map : Any, default=None
            SVG hover map.
        cluster_membership : Any, default=None
            Render-time cluster membership.
        cluster_bboxes : Any, default=None
            Render-time cluster bboxes.

        Returns
        -------
        None
            Appends observed arguments.
        """
        del ax, graph, curves, positions, svg_hover_map
        observed.append(
            {
                "cluster_membership": cluster_membership,
                "cluster_bboxes": cluster_bboxes,
            }
        )
        return None

    monkeypatch.setattr(mpl_renderer, "_draw_edges", record_draw_edges)

    fig, _ = mpl_renderer.render(
        graph,
        positions,
        config=LayoutConfig(cluster_aware=False),
        show=False,
    )
    plt.close(fig)

    assert observed
    assert observed[0]["cluster_membership"] == {}
    assert observed[0]["cluster_bboxes"] == {}
