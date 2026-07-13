"""Pipeline pins and cached dagre.js fidelity checks."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List

import torch

from dagua.config import LayoutConfig
from dagua.eval.equivalence_metrics import anisotropic_procrustes, procrustes_rmsd
from dagua.graph import DaguaGraph
from dagua.layout.engine import layout
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.dagre import build_dagre_pipeline, layout_dagre_pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import get_op_class

_CACHE_PATH = Path(__file__).with_name("fixtures") / "dagre_reference_layouts.json"


def _diamond_inputs() -> tuple[torch.Tensor, torch.Tensor]:
    """Return the fixed diamond topology and box sizes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Edge index ``[2, 4]`` and node sizes ``[4, 2]``.
    """
    edge_index = torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long)
    node_sizes = torch.tensor([[40.0, 20.0]] * 4)
    return edge_index, node_sizes


def _load_reference_cache() -> Dict[str, Any]:
    """Load the checked-in one-layout-per-graph Dagre cache.

    Returns
    -------
    dict[str, Any]
        Parsed reference metadata and graph rows.
    """
    return json.loads(_CACHE_PATH.read_text())


def _edge_index(edges: List[List[int]]) -> torch.Tensor:
    """Convert cached edge pairs into a PyG edge-index tensor.

    Parameters
    ----------
    edges : list[list[int]]
        Cached ``[source, target]`` pairs.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def test_dagre_pipeline_and_ops_are_registered() -> None:
    """Register the public algorithm and every reusable stage.

    Returns
    -------
    None
        Registry lookups must resolve the Dagre entrypoint and op classes.
    """
    assert PIPELINE_REGISTRY["dagre"] == (
        "dagua.layout.ops.pipelines.dagre",
        "layout_dagre_pipeline",
    )
    assert get_pipeline_function("DAGRE") is layout_dagre_pipeline
    assert get_op_class("brandes_koepf_x_assignment").__name__ == "BrandesKoepfXAssignment"
    assert get_op_class("dagre_assign_ranks").__name__ == "DagreAssignRanks"


def test_dagre_pipeline_has_stage_composition() -> None:
    """Pin the algorithm as independent composable operations.

    Returns
    -------
    None
        The pipeline operation sequence must remain explicit.
    """
    pipeline = build_dagre_pipeline()
    assert [operation.name for operation in pipeline.ops] == [
        "dagre_prepare_graph",
        "dagre_make_acyclic",
        "dagre_assign_ranks",
        "dagre_normalize_edges",
        "dagre_order_nodes",
        "brandes_koepf_x_assignment",
        "dagre_assign_y",
        "dagre_finalize_coordinates",
    ]


def test_dagre_diamond_stage_and_position_pins() -> None:
    """Pin ranks, orders, and final positions on a small diamond.

    Returns
    -------
    None
        Stage snapshots and coordinates must match dagre.js 0.8.5.
    """
    edge_index, node_sizes = _diamond_inputs()
    problem = LayoutProblem(edge_index=edge_index, num_nodes=4, node_sizes=node_sizes)
    final_state = build_dagre_pipeline().apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )

    assert final_state.extras["dagre_ranks"] == [0, 2, 2, 4]
    assert final_state.extras["dagre_ordering"] == [0, 0, 1, 0]
    torch.testing.assert_close(
        final_state.pos,
        torch.tensor(
            [[65.0, 10.0], [20.0, 80.0], [110.0, 80.0], [65.0, 150.0]],
            dtype=torch.float64,
        ),
        rtol=0.0,
        atol=0.0,
    )


def test_dagre_variant_position_pins() -> None:
    """Pin direction, alignment, and spacing variant outputs.

    Returns
    -------
    None
        Public variant options must retain their dagre.js coordinates.
    """
    edge_index, node_sizes = _diamond_inputs()
    expected = {
        "BT": [[65.0, 150.0], [20.0, 80.0], [110.0, 80.0], [65.0, 10.0]],
        "LR": [[20.0, 45.0], [110.0, 10.0], [110.0, 80.0], [200.0, 45.0]],
        "UL": [[20.0, 10.0], [20.0, 80.0], [110.0, 80.0], [20.0, 150.0]],
        "spacing": [[45.0, 10.0], [20.0, 110.0], [70.0, 110.0], [45.0, 210.0]],
    }
    outputs = {
        "BT": layout_dagre_pipeline(edge_index, 4, node_sizes, rankdir="BT"),
        "LR": layout_dagre_pipeline(edge_index, 4, node_sizes, rankdir="LR"),
        "UL": layout_dagre_pipeline(edge_index, 4, node_sizes, align="UL"),
        "spacing": layout_dagre_pipeline(
            edge_index,
            4,
            node_sizes,
            nodesep=10.0,
            ranksep=80.0,
            edgesep=5.0,
        ),
    }
    for name, positions in outputs.items():
        torch.testing.assert_close(
            positions,
            torch.tensor(expected[name], dtype=torch.float64),
            rtol=0.0,
            atol=0.0,
        )


def test_dagre_cached_reference_layouts_are_similarity_exact() -> None:
    """Verify every cached small and larger graph against dagre.js.

    Returns
    -------
    None
        Ordinary and anisotropic residuals must remain below ``1e-9``.
    """
    cache = _load_reference_cache()
    assert cache["layouts_per_graph"] == 1
    for graph in cache["graphs"]:
        positions = layout_dagre_pipeline(
            edge_index=_edge_index(graph["edges"]),
            num_nodes=int(graph["num_nodes"]),
            node_sizes=torch.tensor(graph["node_sizes"], dtype=torch.float64),
            nodesep=40.0,
            ranksep=60.0,
            edgesep=20.0,
        ).to(dtype=torch.float32)
        reference = torch.tensor(graph["reference_positions"], dtype=torch.float32)
        residual = procrustes_rmsd(positions.numpy(), reference.numpy())
        anisotropic = anisotropic_procrustes(positions.numpy(), reference.numpy())
        assert residual < 1.0e-9, graph["name"]
        assert float(anisotropic["anisotropic_rmsd"]) < 1.0e-9, graph["name"]


def test_layout_config_algorithm_dagre_and_hard_pin_work() -> None:
    """Exercise public engine dispatch and resolved hard-pin projection.

    Returns
    -------
    None
        ``LayoutConfig(algorithm="dagre")`` must return and preserve the pin.
    """
    graph = DaguaGraph.from_edge_list([("root", "left"), ("root", "right")])
    graph.pin("left", x=123.0, y=-45.0)
    positions = layout(graph, LayoutConfig(algorithm="dagre"))
    left = graph._id_to_index["left"]

    assert positions.shape == (3, 2)
    assert positions[left].tolist() == [123.0, -45.0]


def test_dagre_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the implementation against Node or competitor delegation.

    Returns
    -------
    None
        Production source must not contain subprocess/reference adapter hooks.
    """
    source_paths = [
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "dagre.py",
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "brandes_koepf.py",
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "dagre.py",
    ]
    source = "\n".join(path.read_text() for path in source_paths)
    assert "subprocess" not in source
    assert "DagreCompetitor" not in source
    assert "node_modules" not in source
