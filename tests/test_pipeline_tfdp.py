"""Pipeline pins and reference checks for t-FDP."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.eval.equivalence_metrics import procrustes_rmsd
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.tfdp import build_tfdp_pipeline, layout_tfdp_pipeline
from dagua.layout.ops.taxonomy import get_op_class
from dagua.layout.ops.tfdp import TFDPConfig


def _graph_from_edges(num_nodes: int, edges: list[tuple[int, int]]) -> DaguaGraph:
    """Build a deterministic graph for t-FDP checks.

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


def test_tfdp_pipeline_and_ops_are_registered() -> None:
    """Register public t-FDP algorithms and stage ops.

    Returns
    -------
    None
        Registry lookups must resolve t-FDP entries.
    """
    assert PIPELINE_REGISTRY["tfdp"] == (
        "dagua.layout.ops.pipelines.tfdp",
        "layout_tfdp_pipeline",
    )
    assert PIPELINE_REGISTRY["tfdp_exact"] == (
        "dagua.layout.ops.pipelines.tfdp",
        "layout_tfdp_exact_pipeline",
    )
    assert PIPELINE_REGISTRY["tfdp_random"] == (
        "dagua.layout.ops.pipelines.tfdp",
        "layout_tfdp_random_pipeline",
    )
    assert get_pipeline_function("TFDP") is layout_tfdp_pipeline
    assert get_op_class("tfdp_initialize").__name__ == "TFDPInitialize"
    assert get_op_class("tfdp_iteration").__name__ == "TFDPIteration"


def test_tfdp_pipeline_has_stage_composition() -> None:
    """Pin t-FDP as an explicit op composition.

    Returns
    -------
    None
        The top-level stage names must remain visible.
    """
    pipeline = build_tfdp_pipeline(TFDPConfig(max_iter=2, seed=3))
    assert [operation.name for operation in pipeline.ops] == [
        "tfdp_initialize",
        "tfdp_iteration",
    ]


def test_tfdp_exact_is_seed_deterministic() -> None:
    """Exact t-FDP should return identical positions for identical seeds.

    Returns
    -------
    None
        Repeated runs must match exactly.
    """
    graph = _graph_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    first = layout_tfdp_pipeline(graph.edge_index, graph.num_nodes, seed=11, max_iter=8)
    second = layout_tfdp_pipeline(graph.edge_index, graph.num_nodes, seed=11, max_iter=8)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert first.shape == (5, 2)


def test_tfdp_public_dispatch_returns_positions() -> None:
    """``dagua.layout`` should dispatch to the t-FDP pipeline.

    Returns
    -------
    None
        LayoutConfig dispatch must return finite coordinates.
    """
    graph = _graph_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    positions = dagua.layout(
        graph,
        LayoutConfig(
            algorithm="tfdp",
            steps=5,
            seed=7,
            algorithm_params={"init": "random"},
        ),
    )
    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_tfdp_single_step_regression_pin() -> None:
    """Pin one exact-mode small graph against the current native loop.

    Returns
    -------
    None
        The rotation-invariant residual against the pinned coordinates must
        remain tiny.
    """
    graph = _graph_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    actual = layout_tfdp_pipeline(
        graph.edge_index,
        graph.num_nodes,
        seed=5,
        max_iter=3,
        init="random",
    )
    expected = torch.tensor(
        [
            [1.9207534790, -1.0312495232],
            [1.0507655144, 1.0312495232],
            [0.3709546328, 0.6321423054],
            [-1.9207534790, -0.4083768129],
        ],
        dtype=torch.float32,
    )
    residual = procrustes_rmsd(actual.numpy(), expected.numpy())
    assert residual < 1.0e-6


def test_tfdp_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the production pipeline against reference delegation.

    Returns
    -------
    None
        Production source must not contain subprocess or reference hooks.
    """
    source_paths = [
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "tfdp.py",
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "tfdp.py",
    ]
    source = "\n".join(path.read_text() for path in source_paths)
    assert "subprocess" not in source
    assert "TFDPCompetitor" not in source
    assert "/tmp/tfdp-ref" not in source
    assert "source_code" not in source


def test_tfdp_competitor_is_not_imported_by_pipeline() -> None:
    """Pin pipeline source against eval-adapter imports.

    Returns
    -------
    None
        The native pipeline must not import the competitor adapter.
    """
    tfdp = importlib.import_module("dagua.layout.ops.pipelines.tfdp")

    source = inspect.getsource(tfdp)
    assert "competitors" not in source
