"""Pipeline pins and reference checks for d3-force."""

from __future__ import annotations

from pathlib import Path

import torch

from dagua.config import LayoutConfig
from dagua.eval.competitors.d3force_competitor import D3ForceCompetitor
from dagua.eval.equivalence_metrics import procrustes_rmsd
from dagua.graph import DaguaGraph
from dagua.layout.engine import layout
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.d3force import build_d3force_pipeline, layout_d3force_pipeline
from dagua.layout.ops.taxonomy import get_op_class


def _graph_from_edges(num_nodes: int, edges: list[tuple[int, int]]) -> DaguaGraph:
    """Build a deterministic graph for d3-force checks.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : list[tuple[int, int]]
        Edge pairs.

    Returns
    -------
    DaguaGraph
        Graph with nodes ``0..N-1``.
    """
    graph = DaguaGraph.from_edge_list(edges, num_nodes=num_nodes)
    graph.compute_node_sizes()
    return graph


def test_d3force_pipeline_and_ops_are_registered() -> None:
    """Register public d3-force algorithms and stage ops.

    Returns
    -------
    None
        Registry lookups must resolve d3-force entries.
    """
    assert PIPELINE_REGISTRY["d3force"] == (
        "dagua.layout.ops.pipelines.d3force",
        "layout_d3force_pipeline",
    )
    assert PIPELINE_REGISTRY["d3force_default"] == (
        "dagua.layout.ops.pipelines.d3force",
        "layout_d3force_default_pipeline",
    )
    assert PIPELINE_REGISTRY["d3force_strong_repulsion"] == (
        "dagua.layout.ops.pipelines.d3force",
        "layout_d3force_strong_repulsion_pipeline",
    )
    assert get_pipeline_function("D3FORCE") is layout_d3force_pipeline
    assert get_op_class("d3force_initialize").__name__ == "D3ForceInitialize"
    assert get_op_class("d3force_link").__name__ == "D3ForceLink"
    assert get_op_class("d3force_many_body").__name__ == "D3ForceManyBody"


def test_d3force_pipeline_has_stage_composition() -> None:
    """Pin d3-force as an explicit op composition.

    Returns
    -------
    None
        The top-level and repeated stage names must remain visible.
    """
    pipeline = build_d3force_pipeline()
    assert [operation.name for operation in pipeline.ops] == [
        "d3force_initialize",
        "repeat",
    ]
    repeat = pipeline.ops[1]
    assert [operation.name for operation in repeat.inner.ops] == [
        "d3force_update_alpha",
        "d3force_link",
        "d3force_many_body",
        "d3force_center",
        "d3force_integrate",
    ]


def test_d3force_chain_reference_is_close() -> None:
    """Compare a small chain against the Node reference.

    Returns
    -------
    None
        Current direct n-body implementation must stay close to d3-force on
        a tiny graph where Barnes-Hut approximation has little effect.
    """
    graph = _graph_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    reference = D3ForceCompetitor().layout_with_variant(
        graph,
        seed=1,
        variant_params={"ticks": 30},
    )
    assert reference.error is None
    assert reference.pos is not None
    actual = layout_d3force_pipeline(graph.edge_index, graph.num_nodes, ticks=30)
    residual = procrustes_rmsd(actual.numpy(), reference.pos.numpy())
    assert residual < 0.1


def test_layout_config_algorithm_d3force_works() -> None:
    """Exercise public engine dispatch for d3-force.

    Returns
    -------
    None
        ``LayoutConfig(algorithm="d3force")`` must return an ``[N, 2]`` layout.
    """
    graph = _graph_from_edges(3, [(0, 1), (1, 2)])
    positions = layout(
        graph,
        LayoutConfig(
            algorithm="d3force",
            steps=5,
            seed=1,
            algorithm_params={"center": True},
        ),
    )
    assert positions.shape == (3, 2)
    assert torch.isfinite(positions).all()


def test_d3force_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the production pipeline against Node/reference delegation.

    Returns
    -------
    None
        Production d3-force source must not contain subprocess hooks.
    """
    source_paths = [
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "d3force.py",
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "d3force.py",
    ]
    source = "\n".join(path.read_text() for path in source_paths)
    assert "subprocess" not in source
    assert "D3ForceCompetitor" not in source
    assert "node_modules" not in source
