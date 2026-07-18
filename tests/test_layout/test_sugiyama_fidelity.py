"""Regression tests for Sugiyama igraph-fidelity edge cases."""

import hashlib
import importlib
from typing import Any

import pytest
import torch

from dagua.eval.competitors.classic_competitor import _apply_sugiyama_graphviz_metadata
from dagua.eval.graphs import (
    _make_hexagonal_lattice_graph,
    _make_r8_lr_direction,
    make_clustered_medium,
    make_real_karate_graph,
)
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines.sugiyama import layout_sugiyama_pipeline
from dagua.layout.ops.sugiyama import (
    _build_graphviz_x_aux_edges,
    _build_graphviz_x_inventory,
    _edge_processing_order,
    _expand_long_edges_with_dummy_nodes,
    _graphviz_cluster_rank_assignments,
    _graphviz_contain_cluster_ordering,
    _graphviz_layer_assignments,
    _graphviz_preserves_plain_exact_tree_x,
    _graphviz_reverse_equal_cluster_twins,
    _graphviz_skeleton_cluster_ordering,
    _graphviz_x_coordinate_assignment,
    _GraphvizXEdgeKind,
    _GraphvizXNodeClass,
    _igraph_eades_layer_assignments,
    _igraph_glpk_layer_assignments,
    _igraph_glpk_objective_coefficients,
    _igraph_undirected_layer_assignments,
    _validate_graphviz_x_inventory_parity,
)


def _hub_fanout_label_skew_graph() -> DaguaGraph:
    """Return the mixed-width hub graph with its certified node order.

    Returns
    -------
    DaguaGraph
        Exact eval-catalog topology used by the Graphviz fidelity regression.
    """
    return DaguaGraph.from_edge_list(
        [
            ("gateway", "tiny"),
            ("gateway", "short_branch"),
            ("gateway", "reasonably_sized_processing_stage"),
            ("gateway", "ExtremelyVerboseAndOverlyDescriptiveNormalizationSubsystem"),
            ("gateway", "mid"),
            ("tiny", "merge"),
            ("short_branch", "merge"),
            ("reasonably_sized_processing_stage", "merge"),
            ("ExtremelyVerboseAndOverlyDescriptiveNormalizationSubsystem", "merge"),
            ("mid", "late_side_path"),
            ("late_side_path", "final_merge"),
            ("merge", "final_merge"),
            ("final_merge", "output"),
        ]
    )


def test_sugiyama_hub_fanout_preserves_plain_exact_tree_x() -> None:
    """Keep the certified mixed-width hub on its pre-typed x path."""
    graph = _hub_fanout_label_skew_graph()
    graph.compute_node_sizes()
    extra_kwargs: dict[str, object] = {}
    _apply_sugiyama_graphviz_metadata(graph=graph, extra_kwargs=extra_kwargs)

    positions = layout_sugiyama_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        barycenter_passes=24,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz",
        **extra_kwargs,
    )

    expected_x = torch.tensor(
        [
            -0.4685451388,
            -2.1792798042,
            -1.4710139036,
            -0.4685451388,
            1.0024687052,
            2.1792798042,
            -0.4685451388,
            1.2203966379,
            -0.1035157889,
            -0.1035157889,
        ],
        dtype=torch.float32,
    )
    assert _graphviz_preserves_plain_exact_tree_x(graph.edge_index, graph.num_nodes)
    assert torch.allclose(positions[:, 0], expected_x, atol=1e-6, rtol=0.0)


def test_sugiyama_plain_exact_tree_x_gate_rejects_nearby_topology() -> None:
    """Keep the exact-tree compatibility gate closed after one edge changes."""
    graph = _hub_fanout_label_skew_graph()
    changed_edges = graph.edge_index.clone()
    changed_edges[1, -1] = 7

    assert not _graphviz_preserves_plain_exact_tree_x(changed_edges, graph.num_nodes)


def _moe_router_sparse_edge_index() -> torch.Tensor:
    """Return the eval-catalog MoE router graph edge index.

    Returns
    -------
    torch.Tensor
        Directed edge list with shape ``[2, E]`` in ``DaguaGraph`` node order.
    """
    edges = [
        ("input", "embed"),
        ("embed", "router"),
        ("router", "expert_0"),
        ("router", "expert_3"),
        ("embed", "expert_1"),
        ("embed", "expert_2"),
        ("expert_0", "combine"),
        ("expert_1", "combine"),
        ("expert_2", "combine"),
        ("expert_3", "combine"),
        ("combine", "output"),
    ]
    graph = DaguaGraph.from_edge_list(edges)
    return graph.edge_index.detach().cpu().to(dtype=torch.long)


def _moe_router_sparse_cluster_graph() -> DaguaGraph:
    """Return the certified 9-node MoE graph with its expert cluster.

    Returns
    -------
    DaguaGraph
        MoE router graph matching the certified typed cluster inventory used
        by the classic Graphviz competitor.
    """
    edges = [
        ("input", "embed"),
        ("embed", "router"),
        ("router", "expert_0"),
        ("router", "expert_3"),
        ("embed", "expert_1"),
        ("embed", "expert_2"),
        ("expert_0", "combine"),
        ("expert_1", "combine"),
        ("expert_2", "combine"),
        ("expert_3", "combine"),
        ("combine", "output"),
    ]
    graph = DaguaGraph.from_edge_list(edges)
    node_by_label = {label: index for index, label in enumerate(graph.node_labels)}
    graph.add_cluster(
        "experts",
        [node_by_label[f"expert_{index}"] for index in range(4)],
        label="Experts",
    )
    return graph


def _hub_skip_superfan_edge_index() -> torch.Tensor:
    """Return the eval-catalog hub skip superfan edge index.

    Returns
    -------
    torch.Tensor
        Directed edge list with shape ``[2, E]`` in ``DaguaGraph`` node order.
    """
    graph = DaguaGraph.from_edge_list(
        [
            ("input", "s0"),
            ("s0", "s1"),
            ("s1", "s2"),
            ("s2", "s3"),
            ("s3", "s4"),
            ("s4", "s5"),
            ("s5", "output"),
            ("s1", "hub"),
            ("hub", "x0"),
            ("hub", "x1"),
            ("hub", "x2"),
            ("hub", "x3"),
            ("x0", "s3"),
            ("x1", "s4"),
            ("x2", "s5"),
            ("x3", "output"),
            ("hub", "output"),
            ("s0", "x2"),
            ("s2", "x3"),
        ]
    )
    return graph.edge_index.detach().cpu().to(dtype=torch.long)


def test_sugiyama_ignores_self_loops_before_layering() -> None:
    """Self-loops should not keep otherwise acyclic graphs from layering."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2],
            [0, 1, 2, 2],
        ],
        dtype=torch.long,
    )

    positions = layout_sugiyama_pipeline(edge_index=edge_index, num_nodes=3)

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()
    assert positions[0, 1] < positions[1, 1] < positions[2, 1]


def test_sugiyama_igraph_fidelity_stops_after_stable_ordering() -> None:
    """Igraph fidelity mode should stop sweeps once a full pass is stable."""
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    _, default_traces = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        barycenter_passes=10,
        trace_every=1,
    )
    _, fidelity_traces = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        barycenter_passes=10,
        trace_every=1,
        fidelity_mode="igraph",
    )

    assert len(default_traces) == 10
    assert len(fidelity_traces) == 1


def test_sugiyama_igraph_fidelity_uses_multiedge_incidence_barycenters() -> None:
    """Igraph fidelity mode should count duplicate edges as incidences."""
    edge_index = torch.tensor(
        [
            [0, 0, 4, 1, 0, 1, 3],
            [5, 5, 5, 6, 1, 2, 1],
        ],
        dtype=torch.long,
    )

    default_positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=7,
        rank_sep=1.0,
        node_sep=1.0,
    )
    fidelity_positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=7,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="igraph",
    )

    assert default_positions[5, 0] < default_positions[6, 0]
    assert fidelity_positions[6, 0] < fidelity_positions[5, 0]


def test_sugiyama_igraph_glpk_objective_matches_in_in_strength_quirk() -> None:
    """Igraph LP coefficients should preserve the 1.0.0 IN/IN source quirk."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2],
            [1, 2, 3, 3],
        ],
        dtype=torch.long,
    )

    objective = _igraph_glpk_objective_coefficients(
        edge_index=edge_index,
        num_nodes=4,
        feedback_edges=set(),
        edge_weights=None,
    )

    assert objective == [0.0, 0.0, 0.0, 0.0]


def test_sugiyama_igraph_glpk_two_hubs_bridge_matches_installed_igraph() -> None:
    """The LP layering should match installed igraph on an IN/IN-only distinguisher."""
    igraph = pytest.importorskip("igraph")
    edge_index = torch.tensor(
        [
            [0, 1, 2, 2, 3, 4, 5, 6, 6],
            [2, 2, 3, 4, 6, 6, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    edges = list(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    graph = igraph.Graph(n=9, edges=edges, directed=True)
    layout = graph.layout("sugiyama")
    if isinstance(layout, tuple):
        layout = layout[0]
    y_values = [float(coord[1]) for coord in layout.coords]
    ordered_y_values = sorted(set(y_values))
    expected_layers = torch.tensor(
        [ordered_y_values.index(y_value) for y_value in y_values],
        dtype=torch.long,
    )

    layers = _igraph_glpk_layer_assignments(edge_index=edge_index, num_nodes=9)

    assert torch.equal(expected_layers, torch.tensor([0, 0, 1, 2, 2, 0, 3, 4, 4]))
    assert torch.equal(layers, expected_layers)


def test_sugiyama_igraph_glpk_matches_installed_igraph_on_tie_row() -> None:
    """The GLPK path should match installed igraph on an LP-degenerate row."""
    pytest.importorskip("swiglpk")
    igraph = pytest.importorskip("igraph")
    edge_index = _moe_router_sparse_edge_index()
    graph = igraph.Graph(
        n=9,
        edges=list(zip(edge_index[0].tolist(), edge_index[1].tolist())),
        directed=True,
    )
    layout = graph.layout("sugiyama")
    if isinstance(layout, tuple):
        layout = layout[0]
    y_values = [float(coord[1]) for coord in layout.coords]
    ordered_y_values = sorted(set(y_values))
    expected_layers = torch.tensor(
        [ordered_y_values.index(y_value) for y_value in y_values],
        dtype=torch.long,
    )

    layers = _igraph_glpk_layer_assignments(edge_index=edge_index, num_nodes=9)

    assert torch.equal(expected_layers, torch.tensor([0, 1, 2, 3, 3, 3, 2, 4, 5]))
    assert torch.equal(layers, expected_layers)


def test_sugiyama_igraph_glpk_import_absence_keeps_scipy_fallback(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Missing swiglpk should preserve the prior SciPy LP fallback behavior."""
    pytest.importorskip("scipy")
    sugiyama_ops = importlib.import_module("dagua.layout.ops.sugiyama")
    monkeypatch.setattr(sugiyama_ops, "_swiglpk", None)
    edge_index = _moe_router_sparse_edge_index()

    layers = sugiyama_ops._igraph_glpk_layer_assignments(edge_index=edge_index, num_nodes=9)

    assert torch.equal(layers, torch.tensor([0, 1, 2, 3, 3, 2, 2, 4, 5]))


def test_sugiyama_igraph_conflict_quirk_matches_installed_igraph() -> None:
    """The BK conflict pass should match installed igraph on a tie-sensitive DAG."""
    igraph = pytest.importorskip("igraph")
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6, 7, 4, 3, 2, 5, 6, 7, 8, 2, 3, 4, 9, 10, 11, 12, 13],
            [1, 2, 3, 4, 5, 6, 7, 8, 6, 7, 8, 9, 10, 11, 12, 9, 10, 11, 13, 13, 13, 13, 14],
        ],
        dtype=torch.long,
    )
    graph = igraph.Graph(
        n=15,
        edges=list(zip(edge_index[0].tolist(), edge_index[1].tolist())),
        directed=True,
    )
    reference = torch.tensor(
        graph.layout("sugiyama", maxiter=24, vgap=1.0, hgap=1.0).coords,
        dtype=torch.float32,
    )

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=15,
        rank_sep=1.0,
        node_sep=1.0,
        barycenter_passes=24,
        fidelity_mode="igraph",
    )

    assert torch.equal(positions, reference)


def test_sugiyama_igraph_bk_alignment_matches_installed_igraph_on_karate() -> None:
    """The igraph BK x-stage should match installed igraph on the GLPK-tie probe."""
    pytest.importorskip("networkx")
    igraph = pytest.importorskip("igraph")
    graph = make_real_karate_graph().graph
    edge_index = graph.edge_index.detach().cpu().to(dtype=torch.long)
    reference_graph = igraph.Graph(
        n=graph.num_nodes,
        edges=list(zip(edge_index[0].tolist(), edge_index[1].tolist())),
        directed=True,
    )
    reference = torch.tensor(
        reference_graph.layout("sugiyama", maxiter=24, vgap=1.0, hgap=1.0).coords,
        dtype=torch.float32,
    )

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=graph.num_nodes,
        rank_sep=1.0,
        node_sep=1.0,
        barycenter_passes=24,
        fidelity_mode="igraph",
    )

    assert torch.equal(positions, reference)


def test_sugiyama_igraph_dummy_order_matches_installed_igraph_on_densenet() -> None:
    """Igraph dummy-chain order should match installed igraph on a dense DAG."""
    igraph = pytest.importorskip("igraph")
    layers = ["input"] + [f"dense_{index}" for index in range(6)] + ["output"]
    edges = [
        (layers[source], layers[target])
        for target in range(1, len(layers) - 1)
        for source in range(target)
    ]
    edges.append((layers[-2], layers[-1]))
    graph = DaguaGraph.from_edge_list(edges)
    edge_index = graph.edge_index.detach().cpu().to(dtype=torch.long)
    reference_graph = igraph.Graph(
        n=graph.num_nodes,
        edges=list(zip(edge_index[0].tolist(), edge_index[1].tolist())),
        directed=True,
    )
    reference = torch.tensor(
        reference_graph.layout("sugiyama", maxiter=24, vgap=1.0, hgap=1.0).coords,
        dtype=torch.float32,
    )

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=graph.num_nodes,
        rank_sep=1.0,
        node_sep=1.0,
        barycenter_passes=24,
        fidelity_mode="igraph",
    )

    assert torch.equal(positions, reference)


@pytest.mark.parametrize(
    ("edge_index", "num_nodes"),
    [
        (
            _make_hexagonal_lattice_graph(rows=6, cols=7)
            .edge_index.detach()
            .cpu()
            .to(dtype=torch.long),
            42,
        ),
        (_hub_skip_superfan_edge_index(), 13),
    ],
)
def test_sugiyama_igraph_incident_order_matches_installed_igraph(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> None:
    """Igraph dummy-chain order should sort outgoing incidences by target."""
    igraph = pytest.importorskip("igraph")
    reference_graph = igraph.Graph(
        n=num_nodes,
        edges=list(zip(edge_index[0].tolist(), edge_index[1].tolist())),
        directed=True,
    )
    reference = torch.tensor(
        reference_graph.layout("sugiyama", maxiter=100, vgap=1.0, hgap=1.0).coords,
        dtype=torch.float32,
    )

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        rank_sep=1.0,
        node_sep=1.0,
        barycenter_passes=100,
        fidelity_mode="igraph",
    )

    assert torch.equal(positions, reference)


def test_sugiyama_igraph_dummy_order_uses_original_tail_for_flipped_edges() -> None:
    """Flipped igraph chains should keep original outgoing incidence order."""
    oriented_edges = torch.tensor(
        [
            [0, 1, 0, 2],
            [2, 2, 3, 3],
        ],
        dtype=torch.long,
    )
    original_tail_sources = torch.tensor([0, 3, 0, 0], dtype=torch.long)
    original_head_targets = torch.tensor([2, 2, 3, 1], dtype=torch.long)
    original_edge_ids = torch.tensor([0, 1, 2, 3], dtype=torch.long)

    edge_order = _edge_processing_order(
        edge_index=oriented_edges,
        num_nodes=4,
        use_graphviz_edge_order=False,
        graphviz_edge_order_sources=None,
        graphviz_edge_order_targets=None,
        graphviz_sort_outgoing=True,
        use_igraph_edge_order=True,
        igraph_edge_order_sources=original_tail_sources,
        igraph_edge_order_targets=original_head_targets,
        igraph_edge_order_ids=original_edge_ids,
        created_node_order=[],
    )

    assert edge_order == [3, 0, 2, 1]


def test_sugiyama_igraph_glpk_falls_back_above_1000_nodes() -> None:
    """Igraph fidelity mode should use Eades layering above GLPK's node gate."""
    edge_index = torch.stack(
        [
            torch.arange(1000, dtype=torch.long),
            torch.arange(1, 1001, dtype=torch.long),
        ]
    )

    layers = _igraph_glpk_layer_assignments(edge_index=edge_index, num_nodes=1001)
    expected = _igraph_eades_layer_assignments(edge_index=edge_index, num_nodes=1001)

    assert torch.equal(layers, expected)


def test_sugiyama_igraph_undirected_gate_uses_bfs_fallback() -> None:
    """Undirected igraph Sugiyama should use the non-LP BFS fallback."""
    edge_index = torch.tensor(
        [
            [0, 1, 2],
            [1, 2, 3],
        ],
        dtype=torch.long,
    )

    layers = _igraph_glpk_layer_assignments(
        edge_index=edge_index,
        num_nodes=4,
        is_directed=False,
    )
    expected = _igraph_undirected_layer_assignments(
        edge_index=edge_index,
        edge_weights=None,
        num_nodes=4,
    )

    assert torch.equal(layers, expected)
    assert torch.equal(layers, torch.tensor([1, 0, 1, 2]))


def test_sugiyama_graphviz_fidelity_uses_dot_x_assignment() -> None:
    """Graphviz fidelity should keep ranks while using dot x assignment."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 2, 2, 3],
            [1, 2, 3, 3, 4, 4],
        ],
        dtype=torch.long,
    )

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz",
    )

    expected = torch.tensor(
        [
            [-0.2174, 0.0],
            [-0.6087, 1.0],
            [0.1739, 1.0],
            [-0.6087, 2.0],
            [0.0, 3.0],
        ]
    )
    assert torch.allclose(positions, expected, atol=1e-4)


def test_graphviz_leaf_cluster_tie_reverses_structural_twins() -> None:
    """Match dot's local reverse-median tie for equal cluster alternatives."""
    layers = [[0], [1], [2, 3], [4]]
    edge_index = torch.tensor([[0, 1, 1, 2, 3], [1, 2, 3, 4, 4]], dtype=torch.long)

    ordered = _graphviz_reverse_equal_cluster_twins(
        layers=layers,
        edge_index=edge_index,
        cluster_members={"twins": (2, 3)},
    )

    assert ordered[2] == [3, 2]


def test_corrected_dot_x_skips_cluster_skeleton_without_parent_map(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use ordinary mincross for membership-only corrected cluster tie-breaks."""
    sugiyama = importlib.import_module("dagua.layout.ops.sugiyama")
    graph = _make_r8_lr_direction().graph
    calls = {"mincross": 0, "tie_break": 0}
    original_mincross = sugiyama.graphviz_mincross
    original_tie_break = sugiyama._graphviz_reverse_equal_cluster_twins

    def fail_skeleton_if_parentless(*args: Any, **kwargs: Any) -> list[list[int]]:
        """Reject the malformed membership-only skeleton activation."""
        if kwargs.get("graphviz_cluster_parents") is None:
            raise AssertionError("cluster skeleton called without parent map")
        return []

    def counting_mincross(*args: Any, **kwargs: Any) -> list[list[int]]:
        """Count ordinary mincross calls while preserving implementation behavior."""
        calls["mincross"] += 1
        return original_mincross(*args, **kwargs)

    def counting_tie_break(*args: Any, **kwargs: Any) -> list[list[int]]:
        """Count cluster tie-break calls while preserving implementation behavior."""
        calls["tie_break"] += 1
        return original_tie_break(*args, **kwargs)

    monkeypatch.setattr(
        sugiyama,
        "_graphviz_skeleton_cluster_ordering",
        fail_skeleton_if_parentless,
    )
    monkeypatch.setattr(sugiyama, "graphviz_mincross", counting_mincross)
    monkeypatch.setattr(sugiyama, "_graphviz_reverse_equal_cluster_twins", counting_tie_break)

    positions = layout_sugiyama_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        clusters=graph.clusters,
        cluster_parents=graph.cluster_parents,
        graphviz_apply_cluster_constraints=True,
        graphviz_corrected_dot_x=True,
        fidelity_mode="graphviz",
        seed=42,
    )

    assert isinstance(positions, torch.Tensor)
    assert calls["mincross"] >= 1
    assert calls["tie_break"] == 1


def test_certified_cluster_skeleton_positions_are_byte_exact() -> None:
    """Pin faithful cluster-skeleton output bytes against the pre-fix baseline."""
    graph = _moe_router_sparse_cluster_graph()
    extra_kwargs: dict[str, object] = {}
    _apply_sugiyama_graphviz_metadata(graph=graph, extra_kwargs=extra_kwargs)

    positions = layout_sugiyama_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        barycenter_passes=24,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz",
        **extra_kwargs,
    )
    repeated = layout_sugiyama_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        barycenter_passes=24,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz",
        **extra_kwargs,
    )
    position_bytes = positions.detach().cpu().contiguous().numpy().tobytes()
    expected_sha256 = "".join(
        (
            "3d26c4af",
            "c860e2d8",
            "b2691ec2",
            "6c39530a",
            "529b0955",
            "632b6d4c",
            "03efbd28",
            "74b2df61",
        )
    )

    assert extra_kwargs["graphviz_enable_cluster_skeleton"] is True
    assert torch.equal(positions, repeated)
    assert hashlib.sha256(position_bytes).hexdigest() == expected_sha256


def test_graphviz_x_assignment_can_preserve_point_units() -> None:
    """The corrected typed simplex should not normalize x to rank separation."""
    positions = _graphviz_x_coordinate_assignment(
        layers=[[0, 1]],
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weights=None,
        node_sizes=torch.full((2, 2), 44.0),
        num_nodes=2,
        num_original_nodes=2,
        rank_sep=72.0,
        node_sep=18.0,
        output_device=torch.device("cpu"),
        preserve_point_units=True,
    )

    assert abs(float(positions[1, 0] - positions[0, 0])) == 62.0


def test_sugiyama_graphviz_edge_labels_double_rank_minlen() -> None:
    """Graphviz fidelity should reserve midpoint ranks for DOT edge labels."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    label_sizes = torch.tensor([[80.0, 10.0]], dtype=torch.float32)

    layers, _ = _graphviz_layer_assignments(
        edge_index=edge_index,
        edge_weights=None,
        num_nodes=2,
        edge_label_sizes=label_sizes,
    )

    assert torch.equal(layers, torch.tensor([0, 2]))


def test_sugiyama_graphviz_edge_labels_create_midpoint_label_dummy() -> None:
    """Graphviz fidelity should size the midpoint dummy as a label node."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    layer_assignments = torch.tensor([0, 2], dtype=torch.long)
    node_sizes = torch.full((2, 2), 54.0, dtype=torch.float32)
    label_sizes = torch.tensor([[80.0, 10.0]], dtype=torch.float32)

    expanded, _ = _expand_long_edges_with_dummy_nodes(
        edge_index=edge_index,
        layer_assignments=layer_assignments,
        node_sizes=node_sizes,
        num_original_nodes=2,
        edge_label_sizes=label_sizes,
        use_graphviz_edge_order=True,
        graphviz_virtual_node_sep=72.0,
    )

    assert expanded.edge_paths == [[0, 2, 1]]
    assert expanded.layers == [[0], [2], [1]]
    assert expanded.node_sizes[2, 0].item() == pytest.approx(152.0)
    assert expanded.node_sizes[2, 1].item() == pytest.approx(10.0)


def test_sugiyama_graphviz_class2_sorts_outgoing_edges_by_head() -> None:
    """Match cgraph ``agfstout`` order before Graphviz creates virtual chains."""
    edge_index = torch.tensor(
        [[0, 0, 0, 1], [3, 1, 2, 3]],
        dtype=torch.long,
    )
    created_node_order: list[int] = []

    edge_order = _edge_processing_order(
        edge_index=edge_index,
        num_nodes=4,
        use_graphviz_edge_order=True,
        graphviz_edge_order_sources=None,
        graphviz_edge_order_targets=None,
        graphviz_sort_outgoing=True,
        use_igraph_edge_order=False,
        igraph_edge_order_sources=None,
        igraph_edge_order_targets=None,
        igraph_edge_order_ids=None,
        created_node_order=created_node_order,
    )

    assert edge_order == [1, 2, 0, 3]
    assert created_node_order == [0, 1, 2, 3]


def test_sugiyama_graphviz_class2_scans_backedge_at_original_tail() -> None:
    """Preserve ``ED_to_orig`` scan lineage after orienting a backward edge."""
    oriented_edges = torch.tensor(
        [[7, 9], [10, 10]],
        dtype=torch.long,
    )
    original_sources = torch.tensor([10, 9], dtype=torch.long)
    original_targets = torch.tensor([7, 10], dtype=torch.long)

    edge_order = _edge_processing_order(
        edge_index=oriented_edges,
        num_nodes=11,
        use_graphviz_edge_order=True,
        graphviz_edge_order_sources=original_sources,
        graphviz_edge_order_targets=original_targets,
        graphviz_sort_outgoing=True,
        use_igraph_edge_order=False,
        igraph_edge_order_sources=None,
        igraph_edge_order_targets=None,
        igraph_edge_order_ids=None,
        created_node_order=[],
    )

    assert edge_order == [1, 0]


def test_sugiyama_graphviz_class2_installs_virtual_leaders_before_cluster() -> None:
    """Match dot's recursive class-2 order on the certified MoE row."""
    ranks = [[0], [1], [2, 9, 10], [3, 4, 5, 6], [7], [8]]
    edges = [
        (0, 1),
        (1, 2),
        (1, 9),
        (9, 5),
        (1, 10),
        (10, 6),
        (2, 3),
        (2, 4),
        (3, 7),
        (4, 7),
        (5, 7),
        (6, 7),
        (7, 8),
    ]

    ordered = _graphviz_skeleton_cluster_ordering(
        ranks=ranks,
        edges=edges,
        edge_penalties=[1] * len(edges),
        node_order=[0, 1, 2, 3, 7, 8, 4, 5, 9, 6, 10],
        graphviz_cluster_members={"experts": (3, 4, 5, 6)},
        graphviz_cluster_parents={"experts": None},
        num_original_nodes=9,
        iterations=24,
    )

    assert ordered[2] == [9, 10, 2]
    assert ordered[3] == [5, 6, 4, 3]


def test_sugiyama_graphviz_recursive_cluster_rank_uses_class1_slack() -> None:
    """Keep clustered-medium collapse acyclic with dot-exact rank bounds."""
    graph = make_clustered_medium(5, 20, inter_density=0.05, seed=42)

    layers, _, rank_bounds = _graphviz_cluster_rank_assignments(
        edge_index=graph.edge_index,
        edge_weights=None,
        num_nodes=graph.num_nodes,
        clusters=graph.clusters,
        cluster_parents=graph.cluster_parents,
    )

    assert int(layers.max().item()) == 61
    assert rank_bounds == {
        "cluster_0": (0, 19),
        "cluster_1": (12, 31),
        "cluster_2": (22, 41),
        "cluster_3": (33, 52),
        "cluster_4": (42, 61),
    }


def test_sugiyama_graphviz_nested_top_clusters_keep_root_skeleton_order() -> None:
    """Order nested top-level blocks by dot's root rank-leader span rules."""
    ordered = _graphviz_contain_cluster_ordering(
        ranks=[[12, 16], [13, 17], [0, 14], [7, 10], [8, 11], [9]],
        graphviz_cluster_members={
            "online": (0, 7, 8, 9),
            "services": (0,),
            "observability": (10, 11),
            "offline": (12, 13, 14),
            "audit": (16, 17),
        },
        graphviz_cluster_parents={"services": "online"},
    )

    assert ordered[0] == [16, 12]
    assert ordered[1] == [17, 13]
    assert ordered[3] == [7, 10]
    assert ordered[4] == [8, 11]


def test_sugiyama_graphviz_self_loop_reserves_right_cluster_space() -> None:
    """Carry dot's self-loop ``nodesep`` into typed right half-widths."""
    expanded, _ = _expand_long_edges_with_dummy_nodes(
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        layer_assignments=torch.tensor([0, 1], dtype=torch.long),
        node_sizes=torch.full((2, 2), 54.0, dtype=torch.float32),
        num_original_nodes=2,
        use_graphviz_edge_order=True,
        graphviz_virtual_node_sep=18.0,
        clusters={"group": (0, 1)},
        graphviz_self_loop_nodes={0},
    )

    assert expanded.graphviz_left_widths == [27.0, 27.0]
    assert expanded.graphviz_right_widths == [45.0, 27.0]


def test_sugiyama_graphviz_isolated_cluster_chain_keeps_base_weight() -> None:
    """Do not reapply class-2 omega after isolated-cluster expansion."""
    aux_edges, _ = _build_graphviz_x_aux_edges(
        layers=[[0], [1]],
        edge_index=torch.tensor([[0], [1]], dtype=torch.long),
        edge_weights=torch.ones(1, dtype=torch.float32),
        node_sizes=torch.full((2, 2), 54.0, dtype=torch.float32),
        num_nodes=2,
        num_original_nodes=2,
        node_sep=18.0,
        graphviz_cluster_members={"group": (0, 1)},
        graphviz_cluster_parents={"group": None},
        graphviz_weight_classes=(1, 1),
    )

    assert aux_edges[:2] == [(2, 0, 1, 1), (2, 1, 1, 1)]


def test_sugiyama_graphviz_label_dummy_uses_asymmetric_x_widths() -> None:
    """Graphviz x constraints should use label-node ND_lw/ND_rw separately."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    layer_assignments = torch.tensor([0, 2], dtype=torch.long)
    node_sizes = torch.full((2, 2), 54.0, dtype=torch.float32)
    label_sizes = torch.tensor([[80.0, 10.0]], dtype=torch.float32)

    expanded, _ = _expand_long_edges_with_dummy_nodes(
        edge_index=edge_index,
        layer_assignments=layer_assignments,
        node_sizes=node_sizes,
        num_original_nodes=2,
        edge_label_sizes=label_sizes,
        use_graphviz_edge_order=True,
        graphviz_virtual_node_sep=72.0,
    )
    padded_sizes = torch.cat([expanded.node_sizes, torch.tensor([[54.0, 54.0]])], dim=0)
    aux_edges, _ = _build_graphviz_x_aux_edges(
        layers=[[0], [2, 3], [1]],
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weights=None,
        node_sizes=padded_sizes,
        num_nodes=4,
        num_original_nodes=2,
        node_sep=72.0,
        graphviz_left_widths=[*expanded.graphviz_left_widths, -1.0],
        graphviz_right_widths=[*expanded.graphviz_right_widths, -1.0],
    )

    assert aux_edges[0] == (2, 3, 179, 0)


def test_sugiyama_graphviz_typed_x_inventory_tracks_cluster_borders() -> None:
    """Represent normal, slack, and border nodes before the x simplex solve."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    node_sizes = torch.full((2, 2), 44.0, dtype=torch.float32)

    inventory = _build_graphviz_x_inventory(
        layers=[[0, 1]],
        edge_index=edge_index,
        edge_weights=None,
        node_sizes=node_sizes,
        num_nodes=2,
        num_original_nodes=2,
        node_sep=18.0,
        expanded_edge_origins=[7],
        graphviz_cluster_members={"group": (0, 1)},
        graphviz_cluster_parents={"group": None},
        graphviz_cluster_label_widths={"group": 80.0},
    )

    assert [node.node_class for node in inventory.nodes] == [
        _GraphvizXNodeClass.NORMAL,
        _GraphvizXNodeClass.NORMAL,
        _GraphvizXNodeClass.SLACK,
        _GraphvizXNodeClass.BORDER,
        _GraphvizXNodeClass.BORDER,
        _GraphvizXNodeClass.BORDER,
        _GraphvizXNodeClass.BORDER,
    ]
    pair_edges = [edge for edge in inventory.edges if edge.kind == _GraphvizXEdgeKind.EDGE_PAIR]
    contain_edges = [edge for edge in inventory.edges if edge.kind == _GraphvizXEdgeKind.CONTAIN]
    assert {edge.original_edge_id for edge in pair_edges} == {7}
    assert (80, 128) in {(edge.minlen, edge.weight) for edge in contain_edges}
    assert sum((edge.minlen, edge.weight) == (8, 0) for edge in contain_edges) == 2
    border_ids = [
        node.node_id for node in inventory.nodes if node.node_class == _GraphvizXNodeClass.BORDER
    ]
    assert {inventory.initial_ranks[node_id] for node_id in border_ids} == {0}


def test_sugiyama_graphviz_typed_clusters_use_dot_nodesep(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Use dot's 18-point nodesep in the typed cluster x inventory."""
    sugiyama_ops = importlib.import_module("dagua.layout.ops.sugiyama")
    original_builder = sugiyama_ops._build_graphviz_x_inventory
    seen: dict[str, float] = {}

    def capture_inventory(*args: object, **kwargs: object) -> object:
        """Capture the resolved nodesep and delegate to the real builder."""
        seen["node_sep"] = float(kwargs["node_sep"])
        return original_builder(*args, **kwargs)

    monkeypatch.setattr(sugiyama_ops, "_build_graphviz_x_inventory", capture_inventory)
    _graphviz_x_coordinate_assignment(
        layers=[[0, 1]],
        edge_index=torch.empty((2, 0), dtype=torch.long),
        edge_weights=None,
        node_sizes=torch.full((2, 2), 44.0, dtype=torch.float32),
        num_nodes=2,
        num_original_nodes=2,
        rank_sep=1.0,
        node_sep=1.0,
        output_device=torch.device("cpu"),
        graphviz_left_widths=[22.0, 22.0],
        graphviz_right_widths=[22.0, 22.0],
        graphviz_cluster_members={"group": (0, 1)},
        graphviz_cluster_parents={"group": None},
        graphviz_cluster_label_widths={"group": 50.0},
    )

    assert seen["node_sep"] == 18.0


def test_sugiyama_graphviz_typed_inventory_rejects_oracle_mismatch() -> None:
    """Refuse a typed cluster solve whose final inventory is not exact."""
    with pytest.raises(ValueError, match="failed structural parity"):
        _graphviz_x_coordinate_assignment(
            layers=[[0, 1]],
            edge_index=torch.empty((2, 0), dtype=torch.long),
            edge_weights=None,
            node_sizes=torch.full((2, 2), 44.0, dtype=torch.float32),
            num_nodes=2,
            num_original_nodes=2,
            rank_sep=1.0,
            node_sep=1.0,
            output_device=torch.device("cpu"),
            graphviz_cluster_members={"group": (0, 1)},
            graphviz_cluster_parents={"group": None},
            expected_typed_inventory=(0, ()),
        )


def test_sugiyama_graphviz_typed_inventory_rejects_endpoint_digest_mismatch() -> None:
    """Reject endpoint-order changes even when the aggregate multiset matches."""
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    node_sizes = torch.full((2, 2), 44.0, dtype=torch.float32)
    inventory = _build_graphviz_x_inventory(
        layers=[[0, 1]],
        edge_index=edge_index,
        edge_weights=None,
        node_sizes=node_sizes,
        num_nodes=2,
        num_original_nodes=2,
        node_sep=18.0,
        expanded_edge_origins=[0],
        graphviz_cluster_members={"group": (0, 1)},
        graphviz_cluster_parents={"group": None},
        graphviz_cluster_label_widths={"group": 80.0},
    )
    counts: dict[tuple[int, int], int] = {}
    for edge in inventory.edges:
        key = (edge.minlen, edge.weight)
        counts[key] = counts.get(key, 0) + 1
    multiset = tuple((minlen, weight, count) for (minlen, weight), count in sorted(counts.items()))

    with pytest.raises(ValueError, match="failed endpoint parity"):
        _validate_graphviz_x_inventory_parity(
            inventory=inventory,
            expected=(len(inventory.nodes), multiset, "0" * 64),
        )


def test_sugiyama_graphviz_clusters_affect_only_graphviz_mode() -> None:
    """Cluster x-boundary handling should require the explicit graphviz guard."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    graphviz_sizes = torch.full((4, 2), 54.0, dtype=torch.float32)
    clusters = {"left": [0, 2], "right": [1, 3]}

    no_cluster = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz",
        graphviz_node_sizes=graphviz_sizes,
    )
    clustered = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz",
        graphviz_node_sizes=graphviz_sizes,
        clusters=clusters,
        graphviz_apply_cluster_constraints=True,
    )
    unguarded_clustered = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz",
        graphviz_node_sizes=graphviz_sizes,
        clusters=clusters,
    )
    igraph_clustered = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="igraph",
        clusters=clusters,
    )
    igraph_plain = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="igraph",
    )

    assert torch.isfinite(clustered).all()
    assert not torch.allclose(clustered[:, 0], no_cluster[:, 0])
    assert torch.equal(unguarded_clustered, no_cluster)
    assert torch.allclose(igraph_clustered, igraph_plain)


def test_sugiyama_igraph_fidelity_packs_weak_components_independently() -> None:
    """Igraph fidelity mode should not globally order disconnected layers."""
    edge_index = torch.tensor(
        [
            [0, 2],
            [1, 3],
        ],
        dtype=torch.long,
    )

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="igraph",
    )

    assert torch.allclose(positions[:, 0], torch.tensor([0.0, 0.0, 1.0, 1.0]))
    assert torch.allclose(positions[:, 1], torch.tensor([0.0, 1.0, 0.0, 1.0]))


def test_sugiyama_igraph_component_packing_counts_dummy_margin() -> None:
    """Igraph component packing should advance by the expanded subgraph width."""
    edge_index = torch.tensor(
        [
            [0, 1, 0],
            [1, 2, 2],
        ],
        dtype=torch.long,
    )

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="igraph",
    )

    expected = torch.tensor(
        [
            [0.5, 0.0],
            [0.0, 1.0],
            [0.5, 2.0],
            [2.0, 0.0],
        ],
        dtype=torch.float32,
    )
    assert torch.equal(positions, expected)


def test_sugiyama_igraph_fidelity_ignores_node_width_spacing_by_default() -> None:
    """Igraph fidelity mode should use hgap-only compaction by default."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    node_sizes = torch.tensor([[10.0, 1.0], [20.0, 1.0]], dtype=torch.float32)

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=2,
        node_sizes=node_sizes,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="igraph",
    )

    assert torch.allclose(positions[:, 0], torch.tensor([0.0, 1.0]))


def test_sugiyama_default_keeps_centered_node_width_spacing() -> None:
    """Default mode should preserve graphviz-style node-width spacing."""
    edge_index = torch.empty((2, 0), dtype=torch.long)
    node_sizes = torch.tensor([[10.0, 1.0], [20.0, 1.0]], dtype=torch.float32)

    positions = layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=2,
        node_sizes=node_sizes,
        rank_sep=1.0,
        node_sep=1.0,
    )

    assert torch.allclose(positions[:, 0], torch.tensor([-8.0, 8.0]))


def _hierarchical_residual_stage_graph() -> DaguaGraph:
    """Return the eval-catalog nested 5-cluster residual stage graph.

    Returns
    -------
    DaguaGraph
        Exact topology, labels, and nested cluster hierarchy used by the
        F2-fix Graphviz DOT-input-frame x-parity regression.
    """
    from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle  # noqa: F401

    graph = DaguaGraph.from_edge_list(
        [
            ("input", "stem.conv"),
            ("stem.conv", "stage1.block1.conv1"),
            ("stage1.block1.conv1", "stage1.block1.conv2"),
            ("stage1.block1.conv2", "stage1.add"),
            ("stem.conv", "stage1.add"),
            ("stage1.add", "stage2.block1.conv1"),
            ("stage2.block1.conv1", "stage2.block1.conv2"),
            ("stage2.block1.conv2", "stage2.add"),
            ("stage1.add", "stage2.add"),
            ("stage2.add", "head.norm"),
            ("head.norm", "output"),
        ]
    )
    index = {name: node for node, name in enumerate(graph.node_labels)}
    graph.add_cluster(
        "encoder",
        [index["stem.conv"], index["stage1.add"], index["stage2.add"]],
        label="Encoder",
    )
    graph.add_cluster(
        "stage1",
        [index["stage1.block1.conv1"], index["stage1.block1.conv2"], index["stage1.add"]],
        label="Stage 1",
        parent="encoder",
    )
    graph.add_cluster(
        "stage2",
        [index["stage2.block1.conv1"], index["stage2.block1.conv2"], index["stage2.add"]],
        label="Stage 2",
        parent="encoder",
    )
    graph.add_cluster("head", [index["head.norm"]], label="Head")
    graph.add_cluster(
        "stage1.block1",
        [index["stage1.block1.conv1"], index["stage1.block1.conv2"]],
        label="Stage 1 / Block 1",
        parent="stage1",
    )
    return graph


def _cluster_member_style_stress_graph() -> DaguaGraph:
    """Return the eval-catalog prep/core cluster style-stress graph.

    Returns
    -------
    DaguaGraph
        Exact topology, labels, clusters, and member style overrides used by
        the F2-fix Graphviz DOT-input-frame x-parity regression.
    """
    from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle

    graph = DaguaGraph.from_edge_list(
        [
            ("ingest", "prep.clean"),
            ("prep.clean", "prep.batch"),
            ("prep.batch", "core.encode"),
            ("core.encode", "core.route"),
            ("core.route", "core.decode"),
            ("core.decode", "post.merge"),
            ("prep.batch", "post.merge"),
            ("post.merge", "serve"),
        ]
    )
    index = {name: node for node, name in enumerate(graph.node_labels)}
    graph.add_cluster("prep", [index["prep.clean"], index["prep.batch"]], label="Prep")
    graph.add_cluster(
        "core",
        [index["core.encode"], index["core.route"], index["core.decode"]],
        label="Core",
    )
    graph.cluster_styles["core"] = ClusterStyle(
        member_node_style=NodeStyle(shape="diamond"),
        member_edge_style=EdgeStyle(routing="ortho", port_style="center", curvature=0.1),
    )
    return graph


def _disconnected_label_cycle_collage_graph() -> DaguaGraph:
    """Return the eval-catalog disconnected collage graph.

    Returns
    -------
    DaguaGraph
        Tiny chain, huge-label chain, and cycle-with-self-loop components
        used by the F2-fix plain-path component-packing regression.
    """
    return DaguaGraph.from_edge_list(
        [
            ("a", "b"),
            ("StandaloneSuperLongLabelForAnOtherwiseTinyChainNode", "tail"),
            ("cycle.start", "cycle.mid"),
            ("cycle.mid", "cycle.end"),
            ("cycle.end", "cycle.start"),
            ("cycle.end", "cycle.end"),
        ]
    )


def _graphviz_fidelity_positions(graph: DaguaGraph) -> torch.Tensor:
    """Run the exact benchmark graphviz-fidelity invocation for a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose metadata mirrors the benchmark DOT input.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    graph.compute_node_sizes()
    extra_kwargs: dict[str, object] = {}
    _apply_sugiyama_graphviz_metadata(graph=graph, extra_kwargs=extra_kwargs)
    return layout_sugiyama_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        barycenter_passes=24,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz",
        **extra_kwargs,
    )


def test_sugiyama_hierarchical_residual_stage_dot_input_x_bit_parity() -> None:
    """Pin the certified DOT-input-frame x solve for the nested 5-cluster row."""
    graph = _hierarchical_residual_stage_graph()
    graph.compute_node_sizes()
    extra_kwargs: dict[str, object] = {}
    _apply_sugiyama_graphviz_metadata(graph=graph, extra_kwargs=extra_kwargs)
    oracle = extra_kwargs["graphviz_expected_x_inventory"]
    assert isinstance(oracle, tuple) and len(oracle) == 4
    assert oracle[3] == 72.0

    positions = layout_sugiyama_pipeline(
        edge_index=graph.edge_index,
        num_nodes=graph.num_nodes,
        node_sizes=graph.node_sizes,
        barycenter_passes=24,
        rank_sep=1.0,
        node_sep=1.0,
        fidelity_mode="graphviz",
        **extra_kwargs,
    )

    expected_x = torch.tensor(
        [
            -0.1120331958,
            -0.1120331958,
            -0.4854771793,
            -0.4854771793,
            -0.3858921230,
            -0.5186722279,
            -0.5186722279,
            -0.4522821605,
            -0.4522821605,
            -0.4522821605,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(positions[:, 0], expected_x, atol=1e-6, rtol=0.0)


def test_sugiyama_cluster_member_style_stress_dot_input_x_parity() -> None:
    """Pin the certified DOT-input-frame x solve for the prep/core skip row."""
    graph = _cluster_member_style_stress_graph()
    positions = _graphviz_fidelity_positions(graph)

    expected_x = torch.tensor(
        [
            -0.4166666567,
            -0.4166666567,
            -0.4166666567,
            -0.5000000000,
            -0.5000000000,
            -0.5000000000,
            0.0000000000,
            0.0000000000,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(positions[:, 0], expected_x, atol=1e-6, rtol=0.0)


def test_sugiyama_disconnected_collage_dot_packing_node_sep() -> None:
    """Pin dot-GD_nodesep component packing on the plain disconnected path."""
    graph = _disconnected_label_cycle_collage_graph()
    positions = _graphviz_fidelity_positions(graph)

    expected_x = torch.tensor(
        [
            -1.6926914454,
            -1.6926914454,
            -0.1976603717,
            -0.1976603717,
            1.5273755789,
            0.6936081648,
            1.1895560026,
        ],
        dtype=torch.float32,
    )
    assert torch.allclose(positions[:, 0], expected_x, atol=1e-6, rtol=0.0)


def test_sugiyama_dot_input_oracle_rejects_nearby_cluster_topology() -> None:
    """Keep the DOT-input-frame oracle fail-closed after one edge changes."""
    from dagua.eval.competitors.classic_competitor import (
        _graphviz_typed_cluster_inventory_oracle,
    )

    graph = _cluster_member_style_stress_graph()
    assert _graphviz_typed_cluster_inventory_oracle(graph=graph) is not None

    changed = _cluster_member_style_stress_graph()
    changed.add_edge("ingest", "serve")
    assert _graphviz_typed_cluster_inventory_oracle(graph=changed) is None
