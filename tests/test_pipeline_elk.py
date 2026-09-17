"""Pipeline pins and reference checks for the ELK Layered-style pipeline."""

from __future__ import annotations

from pathlib import Path

import torch

from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.engine import layout
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.elk import (
    build_elk_pipeline,
    layout_elk_layered_bk_pipeline,
    layout_elk_layered_ns_pipeline,
    layout_elk_lp_pipeline,
    layout_elk_pipeline,
)
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import get_op_class


def _diamond_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fixed diamond topology and box sizes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Edge index ``[2, 4]`` and node sizes ``[4, 2]``.
    """
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
    node_sizes = torch.tensor([[64.1085968, 34.0]] * 4, dtype=torch.float64)
    return edge_index, node_sizes


def _compound_inputs() -> tuple[torch.Tensor, torch.Tensor, dict[str, list[int]]]:
    """Return a small clustered graph with one cross-hierarchy edge.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, dict[str, list[int]]]
        Edge index ``[2, 3]``, fixed node sizes ``[4, 2]``, and one cluster.
    """
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_sizes = torch.tensor(
        [[40.0, 30.0], [40.0, 30.0], [40.0, 30.0], [40.0, 30.0]],
        dtype=torch.float64,
    )
    return edge_index, node_sizes, {"alpha": [0, 1, 2]}


def test_elk_pipeline_and_ops_are_registered() -> None:
    """Register ELK algorithms and composable ops.

    Returns
    -------
    None
        Registry lookups must resolve the ELK entrypoints and op classes.
    """
    assert PIPELINE_REGISTRY["elk"] == ("dagua.layout.ops.pipelines.elk", "layout_elk_pipeline")
    assert PIPELINE_REGISTRY["elk_lp"] == (
        "dagua.layout.ops.pipelines.elk",
        "layout_elk_lp_pipeline",
    )
    assert get_pipeline_function("ELK") is layout_elk_pipeline
    assert get_pipeline_function("elk_layered_ns") is layout_elk_layered_ns_pipeline
    assert get_pipeline_function("elk_layered_bk") is layout_elk_layered_bk_pipeline
    assert get_pipeline_function("elk_lp") is layout_elk_lp_pipeline
    assert get_op_class("elk_prepare_graph").__name__ == "ElkPrepareGraph"
    assert get_op_class("elk_recursive_compound").__name__ == "ElkRecursiveCompound"
    assert get_op_class("elk_place_nodes").__name__ == "ElkPlaceNodes"


def test_elk_pipeline_has_stage_composition() -> None:
    """Pin the ELK pipeline as explicit composable operations.

    Returns
    -------
    None
        Operation sequence must remain phase-structured.
    """
    pipeline = build_elk_pipeline()
    assert [operation.name for operation in pipeline.ops] == [
        "elk_prepare_graph",
        "elk_break_cycles",
        "elk_assign_layers",
        "elk_minimize_crossings",
        "elk_place_nodes",
    ]


def test_elk_diamond_stage_and_position_pins() -> None:
    """Pin layers, order, and top-left positions on a diamond.

    Returns
    -------
    None
        Stage metadata and coordinates must stay deterministic.
    """
    edge_index, node_sizes = _diamond_inputs()
    final_state = build_elk_pipeline().apply(
        LayoutProblem(edge_index=edge_index, num_nodes=4, node_sizes=node_sizes),
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert final_state.extras["elk_layers"] == [[0], [2, 1], [3]]
    assert final_state.extras["elk_order"] == {0: 0, 2: 0, 1: 1, 3: 0}
    torch.testing.assert_close(
        final_state.pos,
        torch.tensor(
            [
                [22.684766133333333, 12.0],
                [116.1085968, 106.0],
                [12.0, 106.0],
                [22.684766133333333, 200.0],
            ],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=1.0e-7,
    )


def test_elk_variant_position_pins() -> None:
    """Pin direction, spacing, and named variant outputs.

    Returns
    -------
    None
        Public variant options must produce stable coordinates.
    """
    edge_index, node_sizes = _diamond_inputs()
    expected = {
        "UP": [
            [22.684766133333333, 200.0],
            [116.1085968, 106.0],
            [12.0, 106.0],
            [22.684766133333333, 12.0],
        ],
        "RIGHT": [
            [12.0, 22.684766133333333],
            [106.0, 116.1085968],
            [106.0, 12.0],
            [200.0, 22.684766133333333],
        ],
        "spacing": [
            [22.684766133333333, 12.0],
            [76.1085968, 126.0],
            [12.0, 126.0],
            [22.684766133333333, 240.0],
        ],
        "lp": [
            [22.684766133333333, 12.0],
            [116.1085968, 106.0],
            [12.0, 106.0],
            [22.684766133333333, 200.0],
        ],
    }
    outputs = {
        "UP": layout_elk_pipeline(edge_index, 4, node_sizes, direction="UP"),
        "RIGHT": layout_elk_pipeline(edge_index, 4, node_sizes, direction="RIGHT"),
        "spacing": layout_elk_pipeline(
            edge_index,
            4,
            node_sizes,
            node_node_spacing=0.0,
            between_layers_spacing=80.0,
        ),
        "lp": layout_elk_lp_pipeline(edge_index, 4, node_sizes),
    }
    for name, positions in outputs.items():
        torch.testing.assert_close(
            positions,
            torch.tensor(expected[name], dtype=torch.float64),
            rtol=0.0,
            atol=1.0e-7,
        )


def test_elk_empty_clusters_preserve_flat_positions() -> None:
    """Keep the non-compound ELK caller path byte-identical.

    Returns
    -------
    None
        Passing an empty cluster mapping must not engage the recursion wrapper.
    """
    edge_index, node_sizes = _diamond_inputs()

    flat = layout_elk_pipeline(edge_index, 4, node_sizes, seed=9, random_seed=9)
    empty_cluster = layout_elk_pipeline(
        edge_index,
        4,
        node_sizes,
        seed=9,
        random_seed=9,
        clusters={},
    )

    assert torch.equal(flat, empty_cluster)


def test_elk_compound_is_byte_deterministic() -> None:
    """Pin deterministic wrapper behavior for flat and compound rows.

    Returns
    -------
    None
        Repeated calls with identical inputs must produce identical tensors.
    """
    edge_index, node_sizes, clusters = _compound_inputs()

    compound_a = layout_elk_pipeline(edge_index, 4, node_sizes, clusters=clusters, seed=7)
    compound_b = layout_elk_pipeline(edge_index, 4, node_sizes, clusters=clusters, seed=7)
    flat_a = layout_elk_pipeline(edge_index, 4, node_sizes, seed=7)
    flat_b = layout_elk_pipeline(edge_index, 4, node_sizes, seed=7)

    assert torch.equal(compound_a, compound_b)
    assert torch.equal(flat_a, flat_b)


def test_elk_compound_child_layout_uses_child_defaults() -> None:
    """Assert separate-children recursion around the existing flat pipeline.

    Returns
    -------
    None
        The cluster-internal chain must match a standalone flat ELK run with
        ELK child defaults, while the cross-hierarchy edge is ignored.
    """
    edge_index, node_sizes, clusters = _compound_inputs()
    compound = layout_elk_pipeline(edge_index, 4, node_sizes, clusters=clusters, seed=99)
    induced_child = torch.tensor(
        [[12.0, 12.0], [72.0, 12.0], [132.0, 12.0]],
        dtype=torch.float64,
    )

    cluster_offset = compound[0] - induced_child[0]
    torch.testing.assert_close(compound[:3] - cluster_offset, induced_child, rtol=0.0, atol=1.0e-7)
    assert float(compound[3, 1].item()) < float(compound[0, 1].item())


def test_elk_compound_op_records_cross_hierarchy_drops() -> None:
    """Pin structural diagnostics for ELK's dropped compound edges.

    Returns
    -------
    None
        The wrapper must report cross-hierarchy edges while retaining
        same-container edges.
    """
    from dagua.layout.ops.elk_compound import ElkRecursiveCompound, _CompoundOptions

    edge_index, node_sizes, clusters = _compound_inputs()
    state = ElkRecursiveCompound(
        _CompoundOptions(
            direction="DOWN",
            node_node_spacing=40.0,
            between_layers_spacing=60.0,
            cycle_breaking_strategy="greedy",
            layering_strategy="network_simplex",
            crossing_minimization_strategy="layer_sweep",
            node_placement_strategy="brandes_koepf",
            random_seed=3,
            thoroughness=7,
        )
    ).apply(
        LayoutProblem(
            edge_index=edge_index,
            num_nodes=4,
            node_sizes=node_sizes,
            clusters=clusters,
            seed=3,
        ),
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert state.extras["elk_compound"]["dropped_edges"] == [(2, 3)]
    assert state.extras["elk_compound"]["clusters"]["alpha"]["local_edges"] == [(0, 1), (1, 2)]


def test_elk_nested_compound_recurses_before_parent_layout() -> None:
    """Pin nested child containers as rigid boxes in their parent layout.

    Returns
    -------
    None
        The inner cluster layout must remain equivalent to a standalone child
        default run even when its parent also contains a direct leaf.
    """
    edge_index = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    node_sizes = torch.tensor([[40.0, 30.0], [40.0, 30.0], [40.0, 30.0]], dtype=torch.float64)
    clusters = {"outer": [0, 1, 2], "inner": [0, 1]}
    cluster_parents = {"inner": "outer"}

    compound = layout_elk_pipeline(
        edge_index,
        3,
        node_sizes,
        clusters=clusters,
        cluster_parents=cluster_parents,
        seed=5,
    )
    inner = torch.tensor([[12.0, 12.0], [72.0, 12.0]], dtype=torch.float64)

    inner_offset = compound[0] - inner[0]
    torch.testing.assert_close(compound[:2] - inner_offset, inner, rtol=0.0, atol=1.0e-7)


def test_layout_config_algorithm_elk_dispatches() -> None:
    """Exercise public engine dispatch for ``LayoutConfig(algorithm='elk')``.

    Returns
    -------
    None
        The engine must return one position per graph node.
    """
    graph = DaguaGraph.from_edge_list([("root", "left"), ("root", "right")])
    positions = layout(graph, LayoutConfig(algorithm="elk"))

    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()


def test_elk_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard production ELK source against Node or competitor delegation.

    Returns
    -------
    None
        Production source must not contain subprocess/reference adapter hooks.
    """
    source_paths = [
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "elk.py",
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "elk_compound.py",
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "elk.py",
    ]
    source = "\n".join(path.read_text() for path in source_paths)
    assert "subprocess" not in source
    assert "ElkLayered" not in source
    assert "node_modules" not in source


def test_layout_config_elk_named_variants_match_dedicated_wrappers() -> None:
    """Pin registry dispatch of the ELK named variants to the direct path.

    The named-variant wrappers previously used bare ``(*args, **kwargs)``
    signatures, so the engine's signature-filtered dispatch dropped every
    kwarg and the wrappers crashed via ``LayoutConfig(algorithm=...)``.

    Returns
    -------
    None
        Engine dispatch must equal the dedicated wrapper output (float32)
        for each named variant, proving graph tensors and seed reach the
        wrapper through the registry path.
    """
    variant_functions = {
        "elk_layered_ns": layout_elk_layered_ns_pipeline,
        "elk_layered_bk": layout_elk_layered_bk_pipeline,
        "elk_lp": layout_elk_lp_pipeline,
    }
    edges = [("root", "left"), ("root", "right"), ("left", "sink"), ("right", "sink")]
    for algorithm, variant_fn in variant_functions.items():
        engine_positions = layout(
            DaguaGraph.from_edge_list(edges),
            LayoutConfig(algorithm=algorithm, seed=11),
        )

        direct_graph = DaguaGraph.from_edge_list(edges)
        direct_graph.compute_node_sizes()
        direct_positions = variant_fn(
            direct_graph.edge_index,
            direct_graph.num_nodes,
            node_sizes=direct_graph.node_sizes,
            seed=11,
        ).to(torch.float32)

        assert engine_positions.shape == (4, 2)
        assert torch.isfinite(engine_positions).all()
        assert torch.equal(engine_positions, direct_positions), algorithm
