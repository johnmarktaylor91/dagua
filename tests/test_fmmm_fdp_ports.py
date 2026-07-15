"""Regression tests for Graphviz fdp compound-edge port handling."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.cluster_geometry import ClusterTree
from dagua.layout.ops.pipelines.fmmm import (
    _FDP_COMPOUND_EDGE_ATTACHMENTS_KEY,
    _fdp_compound_obstacle_list,
    _fdp_compute_compound_edge_attachments,
    _fdp_deepest_cluster_by_node,
    _fdp_expand_box,
    _fdp_node_boxes,
    _fdp_obstacle_vertices,
    _FdpCompoundEdgeAttachmentOp,
    _graphviz_fdp_component_layout,
    _graphviz_fdp_edge_lists,
    _graphviz_fdp_explicit_size_points,
    _graphviz_fdp_prism_average_edge_length,
    _graphviz_fdp_prism_delaunay_edges,
    _graphviz_fdp_prism_half_size_lists_in_inches,
    _graphviz_fdp_prism_overlap,
    _graphviz_fdp_prism_overlap_edges,
    _graphviz_fdp_xlayout_edges,
    build_fmmm_pipeline,
    layout_fmmm_pipeline,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState


def _nested_port_tree() -> ClusterTree:
    """Build a nested cluster tree used by fdp object-list tests.

    Returns
    -------
    ClusterTree
        Tree with ``outer`` containing ``inner`` and root sibling ``sibling``.
    """
    return ClusterTree.from_flat_membership(
        clusters={
            "outer": [0, 1, 2],
            "inner": [1, 2],
            "sibling": [4],
        },
        cluster_parents={
            "outer": None,
            "inner": "outer",
            "sibling": None,
        },
    )


def test_fdp_make_cluster_obstacle_additive_matches_graphviz_golden() -> None:
    """Graphviz ``makeClustObs`` additive expansion should be bit-stable.

    Returns
    -------
    None
        The assertion checks the literal lower-left, upper-left, upper-right,
        lower-right vertex order from ``clusteredges.c``.
    """
    box = _fdp_expand_box(
        key=("cluster", "c"),
        bounds=(10.0, 20.0, 30.0, 50.0),
        expand=(2.0, 3.0),
        do_add=True,
    )

    assert _fdp_obstacle_vertices(box) == (
        (8.0, 17.0),
        (8.0, 53.0),
        (32.0, 53.0),
        (32.0, 17.0),
    )


def test_fdp_make_cluster_obstacle_multiplicative_matches_graphviz_golden() -> None:
    """Graphviz ``makeClustObs`` multiplicative expansion should match C math.

    Returns
    -------
    None
        The assertion validates the branch where ``pm->doAdd`` is false.
    """
    box = _fdp_expand_box(
        key=("cluster", "c"),
        bounds=(10.0, 20.0, 30.0, 50.0),
        expand=(1.5, 2.0),
        do_add=False,
    )

    assert _fdp_obstacle_vertices(box) == (
        (5.0, 5.0),
        (5.0, 65.0),
        (35.0, 65.0),
        (35.0, 5.0),
    )


def test_fdp_object_list_nested_to_sibling_matches_graphviz_walk() -> None:
    """Ported ``objectList`` should exclude endpoint containers by level.

    Returns
    -------
    None
        The expected keys are a golden vector from the C ``raiseLevel`` and
        sibling-walk logic for an edge from ``inner`` to ``sibling``.
    """
    tree = _nested_port_tree()
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
        ],
        dtype=torch.float32,
    )
    node_boxes = _fdp_node_boxes(pos, None, expand=(0.0, 0.0), do_add=True)
    cluster_boxes = {
        "outer": _fdp_expand_box(("cluster", "outer"), (-1.0, -1.0, 21.0, 1.0), (0.0, 0.0), True),
        "inner": _fdp_expand_box(("cluster", "inner"), (9.0, -1.0, 21.0, 1.0), (0.0, 0.0), True),
        "sibling": _fdp_expand_box(
            ("cluster", "sibling"),
            (39.0, -1.0, 41.0, 1.0),
            (0.0, 0.0),
            True,
        ),
    }
    node_parent = _fdp_deepest_cluster_by_node(tree, 5)

    obstacles = _fdp_compound_obstacle_list(
        source=1,
        target=4,
        tree=tree,
        node_parent=node_parent,
        node_boxes=node_boxes,
        cluster_boxes=cluster_boxes,
    )

    assert [obstacle.key for obstacle in obstacles] == [
        ("node", 2),
        ("node", 0),
        ("node", 3),
    ]


def test_fdp_attachment_points_clip_to_crossed_cluster_boundaries() -> None:
    """Inter-cluster edges should attach at source and target cluster boxes.

    Returns
    -------
    None
        The assertion checks deterministic boundary intersections for sibling
        clusters on a horizontal edge.
    """
    problem = LayoutProblem(
        edge_index=torch.tensor([[1], [4]], dtype=torch.long),
        num_nodes=5,
        node_sizes=torch.zeros((5, 2), dtype=torch.float32),
        clusters={
            "outer": [0, 1, 2],
            "inner": [1, 2],
            "sibling": [4],
        },
        cluster_parents={
            "outer": None,
            "inner": "outer",
            "sibling": None,
        },
    )
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [10.0, 0.0],
            [20.0, 0.0],
            [30.0, 0.0],
            [40.0, 0.0],
        ],
        dtype=torch.float32,
    )

    attachments, _, _ = _fdp_compute_compound_edge_attachments(problem, pos)

    assert len(attachments) == 1
    attachment = attachments[0]
    assert attachment.tail_cluster == "inner"
    assert attachment.head_cluster == "sibling"
    assert attachment.tail_point == pytest.approx((55.0, 0.0))
    assert attachment.head_point == pytest.approx((5.0, 0.0))
    assert attachment.polyline[0] == pytest.approx((55.0, 0.0))
    assert attachment.polyline[1] == pytest.approx((5.0, 0.0))


def test_fdp_compound_attachment_op_is_fidelity_only() -> None:
    """The fdp attachment op should only be present for ``fidelity_mode``.

    Returns
    -------
    None
        The assertion protects default behavior while checking the fidelity
        pipeline records metadata in ``SolveState.extras``.
    """
    default_pipeline = build_fmmm_pipeline(steps=0)
    fidelity_pipeline = build_fmmm_pipeline(steps=0, fidelity_mode=True)

    assert [op.name for op in default_pipeline.ops].count(_FdpCompoundEdgeAttachmentOp.name) == 0
    assert [op.name for op in fidelity_pipeline.ops].count(_FdpCompoundEdgeAttachmentOp.name) == 1

    problem = LayoutProblem(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        num_nodes=2,
        clusters={"left": [0], "right": [1]},
        cluster_parents={"left": None, "right": None},
    )
    state = SolveState(pos=torch.tensor([[0.0, 0.0], [10.0, 0.0]], dtype=torch.float32))

    result = _FdpCompoundEdgeAttachmentOp().apply(problem, state, RuntimeContext())

    assert _FDP_COMPOUND_EDGE_ATTACHMENTS_KEY in result.extras
    assert result.extras[_FDP_COMPOUND_EDGE_ATTACHMENTS_KEY][0].tail_point == pytest.approx(
        (35.0, 0.0)
    )


def test_fdp_prism_delaunay_edges_cover_triangle_neighbors() -> None:
    """SciPy Delaunay should provide the PRISM proximity graph primitive.

    Returns
    -------
    None
        The assertion checks the expected complete triangle neighbor set.
    """
    edges = _graphviz_fdp_prism_delaunay_edges(
        x_positions=[0.0, 1.0, 0.0],
        y_positions=[0.0, 0.0, 1.0],
    )

    assert edges == {(0, 1), (0, 2), (1, 2)}


def test_fdp_prism_overlap_stage_reduces_compact_component_overlaps() -> None:
    """Graphviz FDP PRISM should expand a compact component before packing.

    Returns
    -------
    None
        The assertion verifies that the named overlap-removal stage moves a
        compact component toward the Graphviz no-overlap target.
    """
    positions = torch.tensor(
        [
            [0.0, 0.0],
            [0.1, 0.0],
            [0.0, 0.1],
            [0.1, 0.1],
        ],
        dtype=torch.float64,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 0],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((4, 2), 72.0, dtype=torch.float64)
    half_widths = [0.5 + (4.0 / 72.0)] * 4
    half_heights = [0.5 + (4.0 / 72.0)] * 4

    before = len(
        _graphviz_fdp_prism_overlap_edges(
            x_positions=[float(value) for value in positions[:, 0].tolist()],
            y_positions=[float(value) for value in positions[:, 1].tolist()],
            half_widths=half_widths,
            half_heights=half_heights,
        )
    )
    adjusted = _graphviz_fdp_prism_overlap(
        positions=positions,
        edge_index=edge_index,
        node_sizes=node_sizes,
    )
    after = len(
        _graphviz_fdp_prism_overlap_edges(
            x_positions=[float(value) for value in adjusted[:, 0].tolist()],
            y_positions=[float(value) for value in adjusted[:, 1].tolist()],
            half_widths=half_widths,
            half_heights=half_heights,
        )
    )

    assert before == 6
    assert after < before


def test_fdp_prism_preserves_graphviz_705_scaling_arithmetic() -> None:
    """Pin PRISM padding and Graphviz 7.0.5's edge-length indexing bug.

    Returns
    -------
    None
        Half-sizes and the symmetric sparse-edge average match Graphviz.
    """
    half_widths, half_heights = _graphviz_fdp_prism_half_size_lists_in_inches(
        torch.tensor([[44.0, 34.0], [44.0, 34.0]], dtype=torch.float64),
        2,
    )
    average = _graphviz_fdp_prism_average_edge_length(
        [0.0, 3.0],
        [10.0, 4.0],
        torch.tensor([[0], [1]], dtype=torch.long),
    )

    assert half_widths == [44.0 / 72.0 / 2.0 + 4.0 / 72.0] * 2
    assert half_heights == [34.0 / 72.0 / 2.0 + 4.0 / 72.0] * 2
    assert average == pytest.approx((58.0**0.5 + 5.0) / 2.0)


def test_fdp_component_skips_prism_after_xlayout_resolves_overlaps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Match Graphviz's early return after successful native x-layout.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Fixture used to reject an unexpected PRISM fallback call.

    Returns
    -------
    None
        The five-cycle fixture reaches zero overlaps in native x-layout.
    """

    def reject_prism(**_kwargs: object) -> torch.Tensor:
        """Fail if the PRISM fallback runs.

        Parameters
        ----------
        **_kwargs : object
            Ignored PRISM call arguments.

        Returns
        -------
        torch.Tensor
            This function never returns.

        Raises
        ------
        AssertionError
            Always, because Graphviz skips PRISM for this fixture.
        """
        raise AssertionError("PRISM must not run after x-layout reaches zero overlaps.")

    monkeypatch.setitem(
        _graphviz_fdp_component_layout.__globals__,
        "_graphviz_fdp_prism_overlap",
        reject_prism,
    )
    edge_index = torch.tensor(
        [[0, 1, 2, 3, 4], [1, 2, 3, 4, 0]],
        dtype=torch.long,
    )
    node_sizes = torch.tensor([[79.5363, 34.0]] * 5, dtype=torch.float64)

    positions = _graphviz_fdp_component_layout(
        edge_index=edge_index,
        num_nodes=5,
        node_sizes=node_sizes,
        seed=42,
        flip_y=False,
    )

    assert positions.shape == (5, 2)


def test_fdp_explicit_sizes_preserve_subdefault_fixed_height() -> None:
    """Match Graphviz's fixed-size point quantization without default floors.

    Returns
    -------
    None
        The measured cycle label box becomes Graphviz's emitted 80x34 points.
    """
    assert _graphviz_fdp_explicit_size_points(79.5363) == 80.0
    assert _graphviz_fdp_explicit_size_points(34.0) == 34.0
    assert _graphviz_fdp_explicit_size_points(78.2906) == 78.0


def test_fdp_xlayout_orients_derived_edges_by_node_sequence() -> None:
    """Match Graphviz's canonical orientation without reordering edges.

    Returns
    -------
    None
        Reversed input edges point from lower to higher local node indices.
    """
    edges = torch.tensor([[3, 0, 4], [1, 2, 2]], dtype=torch.long)

    oriented = _graphviz_fdp_xlayout_edges(edges)

    assert torch.equal(oriented, torch.tensor([[1, 0, 2], [3, 2, 4]], dtype=torch.long))


def test_fdp_outgoing_edges_follow_graphviz_target_order() -> None:
    """Match cgraph's target-node ordering within each outgoing edge list.

    Returns
    -------
    None
        Input insertion order does not override Graphviz's node sequence.
    """
    edges = torch.tensor([[0, 0, 0], [4, 1, 3]], dtype=torch.long)

    outgoing, records = _graphviz_fdp_edge_lists(edges, 5, None)

    assert [records[edge_id][1] for edge_id in outgoing[0]] == [1, 3, 4]


def test_fdp_disconnected_cycles_use_node_polyomino_pack_offsets() -> None:
    """Pin Graphviz's flat FDP node-polyomino component translations.

    Returns
    -------
    None
        Four identical cycles retain the captured relative pack offsets.
    """
    sources: list[int] = []
    targets: list[int] = []
    for component_start in range(0, 20, 5):
        for offset in range(5):
            sources.append(component_start + offset)
            targets.append(component_start + ((offset + 1) % 5))
    edge_index = torch.tensor([sources, targets], dtype=torch.long)
    node_sizes = torch.tensor([[79.5363, 34.0]] * 20, dtype=torch.float64)

    positions = layout_fmmm_pipeline(
        edge_index=edge_index,
        num_nodes=20,
        node_sizes=node_sizes,
        steps=200,
        seed=42,
        fidelity_mode="graphviz_fdp",
    ).to(dtype=torch.float64)
    relative_offsets = positions[[5, 10, 15]] - positions[0]

    assert torch.allclose(
        relative_offsets,
        torch.tensor(
            [[190.0, -95.0], [228.0, 76.0], [76.0, -228.0]],
            dtype=torch.float64,
        ),
        atol=0.01,
        rtol=0.0,
    )
