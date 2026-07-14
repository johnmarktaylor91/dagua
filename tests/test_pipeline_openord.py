"""Pipeline pins and reference checks for OpenOrd."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.eval.equivalence_metrics import procrustes_rmsd
from dagua.graph import DaguaGraph
from dagua.layout.ops.openord import _OPENORD_PRESETS, _resolve_openord_parameters
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.openord import build_openord_pipeline, layout_openord_pipeline
from dagua.layout.ops.taxonomy import get_op_class


def _graph_from_edges(num_nodes: int, edges: list[tuple[int, int]]) -> DaguaGraph:
    """Build a deterministic graph for OpenOrd checks.

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


def test_openord_pipeline_and_ops_are_registered() -> None:
    """Register public OpenOrd algorithms and stage ops.

    Returns
    -------
    None
        Registry lookups must resolve OpenOrd entries.
    """
    assert PIPELINE_REGISTRY["openord"] == (
        "dagua.layout.ops.pipelines.openord",
        "layout_openord_pipeline",
    )
    assert PIPELINE_REGISTRY["openord_refine"] == (
        "dagua.layout.ops.pipelines.openord",
        "layout_openord_refine_pipeline",
    )
    assert PIPELINE_REGISTRY["openord_final"] == (
        "dagua.layout.ops.pipelines.openord",
        "layout_openord_final_pipeline",
    )
    assert get_pipeline_function("OpenOrd") is layout_openord_pipeline
    assert get_op_class("openord_prepare_state").__name__ == "OpenOrdPrepareState"
    assert get_op_class("openord_initialize_positions").__name__ == "OpenOrdInitializePositions"
    assert get_op_class("openord_phase_solve").__name__ == "OpenOrdPhaseSolve"


def test_openord_pipeline_has_stage_composition() -> None:
    """Pin OpenOrd as an explicit op composition.

    Returns
    -------
    None
        The top-level stage names must remain visible.
    """
    pipeline = build_openord_pipeline(
        options={
            "liquid_iterations": 1,
            "expansion_iterations": 1,
            "cooldown_iterations": 1,
            "crunch_iterations": 1,
            "simmer_iterations": 1,
        }
    )
    assert [operation.name for operation in pipeline.ops] == [
        "openord_prepare_state",
        "openord_initialize_positions",
        "openord_phase_solve",
        "openord_finalize_positions",
    ]


def test_openord_default_phase_schedule_matches_source() -> None:
    """Pin the C++ OpenOrd default phase schedule.

    Returns
    -------
    None
        The source default liquid and expansion attractions must not regress to
        igraph DrL's swapped preset.
    """
    params = _resolve_openord_parameters("default")
    assert params.edge_cut == 32.0 / 40.0
    assert params.liquid.iterations == 200
    assert params.liquid.attraction == 2.0
    assert params.expansion.iterations == 200
    assert params.expansion.attraction == 10.0
    assert params.cooldown.temperature == 2000.0
    assert params.crunch.iterations == 50
    assert params.simmer.iterations == 100
    assert _OPENORD_PRESETS["final"].simmer.iterations == 25


def test_openord_is_seed_deterministic() -> None:
    """OpenOrd should return identical positions for identical seeds.

    Returns
    -------
    None
        Repeated runs must match exactly.
    """
    graph = _graph_from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    options = {
        "liquid_iterations": 2,
        "expansion_iterations": 2,
        "cooldown_iterations": 2,
        "crunch_iterations": 2,
        "simmer_iterations": 2,
    }
    first = layout_openord_pipeline(graph.edge_index, graph.num_nodes, seed=11, options=options)
    second = layout_openord_pipeline(graph.edge_index, graph.num_nodes, seed=11, options=options)
    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert first.shape == (5, 2)


def test_openord_public_dispatch_returns_positions() -> None:
    """``dagua.layout`` should dispatch to the OpenOrd pipeline.

    Returns
    -------
    None
        LayoutConfig dispatch must return finite coordinates.
    """
    graph = _graph_from_edges(4, [(0, 1), (1, 2), (2, 3)])
    positions = dagua.layout(
        graph,
        LayoutConfig(
            algorithm="openord",
            seed=7,
            algorithm_params={
                "options": {
                    "liquid_iterations": 1,
                    "expansion_iterations": 1,
                    "cooldown_iterations": 1,
                    "crunch_iterations": 1,
                    "simmer_iterations": 1,
                }
            },
        ),
    )
    assert positions.shape == (4, 2)
    assert torch.isfinite(positions).all()


def test_openord_short_schedule_regression_pin() -> None:
    """Pin one small graph against the current native OpenOrd loop.

    Returns
    -------
    None
        The rotation-invariant residual against pinned coordinates must remain
        tiny.
    """
    graph = _graph_from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)])
    actual = layout_openord_pipeline(
        graph.edge_index,
        graph.num_nodes,
        seed=5,
        options={
            "liquid_iterations": 1,
            "expansion_iterations": 1,
            "cooldown_iterations": 1,
            "crunch_iterations": 1,
            "simmer_iterations": 1,
        },
    )
    expected = torch.tensor(
        [
            [-1.1859790087, 3.0966091156],
            [-0.5091637969, 2.5390951633],
            [-0.8494190574, 1.6454149485],
            [0.5963082314, 2.1329045296],
        ],
        dtype=torch.float32,
    )
    residual = procrustes_rmsd(actual.numpy(), expected.numpy())
    assert residual < 1.0e-6


def test_openord_production_pipeline_has_no_runtime_delegation() -> None:
    """Guard the production pipeline against reference delegation.

    Returns
    -------
    None
        Production source must not contain subprocess or reference hooks.
    """
    source_paths = [
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "openord.py",
        Path(__file__).parents[1] / "dagua" / "layout" / "ops" / "pipelines" / "openord.py",
    ]
    source = "\n".join(path.read_text() for path in source_paths)
    assert "subprocess" not in source
    assert "OpenOrdCompetitor" not in source
    assert "/tmp/openord-ref" not in source
    assert "bin/layout" not in source


def test_openord_competitor_is_not_imported_by_pipeline() -> None:
    """Pin pipeline source against eval-adapter imports.

    Returns
    -------
    None
        The native pipeline must not import the competitor adapter.
    """
    openord = importlib.import_module("dagua.layout.ops.pipelines.openord")
    source = inspect.getsource(openord)
    assert "competitors" not in source
