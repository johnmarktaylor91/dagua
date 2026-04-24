"""Regression tests for dagua_native per-component decomposition."""

from __future__ import annotations

import copy

import torch

from dagua.config import LayoutConfig
from dagua.eval.graphs import get_test_graphs
from dagua.graph import DaguaGraph
from dagua.layout import layout
from dagua.layout.ops.pipelines import dagua_native as dagua_native_module
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.metrics import evaluate


def _bbox_for_nodes(
    pos: torch.Tensor,
    node_indices: list[int],
) -> tuple[float, float, float, float]:
    """Return the axis-aligned bounding box for a node subset.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_indices : list[int]
        Node ids included in the bounding box.

    Returns
    -------
    tuple[float, float, float, float]
        ``(min_x, max_x, min_y, max_y)`` for the selected nodes.
    """
    subset = pos[torch.tensor(node_indices, dtype=torch.long)]
    return (
        float(subset[:, 0].min().item()),
        float(subset[:, 0].max().item()),
        float(subset[:, 1].min().item()),
        float(subset[:, 1].max().item()),
    )


def _boxes_are_disjoint(
    left: tuple[float, float, float, float],
    right: tuple[float, float, float, float],
) -> bool:
    """Return whether two axis-aligned bounding boxes do not overlap.

    Parameters
    ----------
    left : tuple[float, float, float, float]
        ``(min_x, max_x, min_y, max_y)`` for the first box.
    right : tuple[float, float, float, float]
        ``(min_x, max_x, min_y, max_y)`` for the second box.

    Returns
    -------
    bool
        ``True`` when the boxes are separated along x or y.
    """
    return left[1] < right[0] or right[1] < left[0] or left[3] < right[2] or right[3] < left[2]


def _test_graph_by_name(name: str) -> DaguaGraph:
    """Return one named benchmark graph.

    Parameters
    ----------
    name : str
        Graph name in the synthetic benchmark corpus.

    Returns
    -------
    dagua.graph.DaguaGraph
        Matching graph object.
    """
    graphs = {graph.name: graph.graph for graph in get_test_graphs(max_nodes=250)}
    graph = copy.deepcopy(graphs[name])
    graph.compute_node_sizes()
    return graph


def _composite_score(graph_name: str, config: LayoutConfig) -> float:
    """Solve one benchmark graph and return its composite score.

    Parameters
    ----------
    graph_name : str
        Graph name in the benchmark corpus.
    config : LayoutConfig
        Layout configuration for the solve.

    Returns
    -------
    float
        Composite score from ``dagua.metrics.evaluate``.
    """
    graph = _test_graph_by_name(graph_name)
    pos = layout(graph, config)
    metrics = evaluate(graph, pos, tier="full")
    return float(metrics["composite_score"])


def test_component_wrapper_detects_and_tiles_three_components(monkeypatch) -> None:
    """Disconnected graphs should recurse per component and tile child layouts."""
    edge_index = torch.tensor(
        [
            [0, 1, 3, 5],
            [1, 2, 4, 6],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((7, 2), 20.0, dtype=torch.float32)
    observed_child_sizes: list[int] = []

    def _fake_run_native_problem(
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
        config: LayoutConfig,
    ) -> torch.Tensor:
        """Return deterministic child-local geometry keyed by component size."""
        del state, ctx, config
        observed_child_sizes.append(problem.num_nodes)
        if problem.num_nodes == 3:
            return torch.tensor(
                [[0.0, 0.0], [30.0, 0.0], [60.0, 0.0]],
                dtype=torch.float32,
                device=problem.edge_index.device,
            )
        return torch.tensor(
            [[0.0, 0.0], [0.0, 40.0]],
            dtype=torch.float32,
            device=problem.edge_index.device,
        )

    monkeypatch.setattr(dagua_native_module, "_run_native_problem", _fake_run_native_problem)

    pos = dagua_native_module.layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=7,
        node_sizes=node_sizes,
        config=LayoutConfig(seed=42, steps=10, decompose_components=True),
    )

    assert observed_child_sizes == [3, 2, 2]
    boxes = [
        _bbox_for_nodes(pos, [0, 1, 2]),
        _bbox_for_nodes(pos, [3, 4]),
        _bbox_for_nodes(pos, [5, 6]),
    ]
    assert _boxes_are_disjoint(boxes[0], boxes[1])
    assert _boxes_are_disjoint(boxes[0], boxes[2])
    assert _boxes_are_disjoint(boxes[1], boxes[2])


def test_single_component_path_is_unchanged_for_connected_graphs() -> None:
    """Connected graphs should bypass decomposition and keep identical output."""
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3],
            [1, 2, 3, 4],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((5, 2), 20.0, dtype=torch.float32)
    enabled = LayoutConfig(seed=42, steps=25, decompose_components=True)
    disabled = LayoutConfig(seed=42, steps=25, decompose_components=False)

    enabled_pos = dagua_native_module.layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        node_sizes=node_sizes,
        config=enabled,
    )
    disabled_pos = dagua_native_module.layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        node_sizes=node_sizes,
        config=disabled,
    )

    assert torch.allclose(enabled_pos, disabled_pos)


def test_dominant_component_graphs_skip_decomposition(monkeypatch) -> None:
    """Graphs with one dominant component should stay on the original path."""
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 4, 5, 6],
            [1, 2, 3, 4, 5, 6, 7],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((9, 2), 20.0, dtype=torch.float32)
    observed_problem_sizes: list[int] = []

    def _fake_run_native_problem(
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
        config: LayoutConfig,
    ) -> torch.Tensor:
        """Record the solved problem size and return deterministic positions."""
        del state, ctx, config
        observed_problem_sizes.append(problem.num_nodes)
        return torch.zeros(
            (problem.num_nodes, 2),
            dtype=torch.float32,
            device=problem.edge_index.device,
        )

    monkeypatch.setattr(dagua_native_module, "_run_native_problem", _fake_run_native_problem)

    pos = dagua_native_module.layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=9,
        node_sizes=node_sizes,
        config=LayoutConfig(seed=42, steps=10, decompose_components=True),
    )

    assert pos.shape == (9, 2)
    assert observed_problem_sizes == [9]


def test_disconnected_label_cycle_collage_score_improves_with_decomposition() -> None:
    """The benchmark regression graph should score better with decomposition enabled."""
    enabled_score = _composite_score(
        "disconnected_label_cycle_collage",
        LayoutConfig(seed=42, steps=80, decompose_components=True),
    )
    disabled_score = _composite_score(
        "disconnected_label_cycle_collage",
        LayoutConfig(seed=42, steps=80, decompose_components=False),
    )

    assert enabled_score > disabled_score
    assert enabled_score >= 70.0


def test_component_decomposition_edge_cases() -> None:
    """Empty, singleton, and all-singleton graphs should remain valid."""
    empty = dagua_native_module.layout_dagua_native_pipeline(
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        num_nodes=0,
        node_sizes=torch.zeros((0, 2), dtype=torch.float32),
        config=LayoutConfig(seed=42, decompose_components=True),
    )
    single = dagua_native_module.layout_dagua_native_pipeline(
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        num_nodes=1,
        node_sizes=torch.full((1, 2), 20.0, dtype=torch.float32),
        config=LayoutConfig(seed=42, decompose_components=True),
    )
    all_singletons = dagua_native_module.layout_dagua_native_pipeline(
        edge_index=torch.zeros((2, 0), dtype=torch.long),
        num_nodes=4,
        node_sizes=torch.full((4, 2), 20.0, dtype=torch.float32),
        config=LayoutConfig(seed=42, steps=10, decompose_components=True),
    )

    assert empty.shape == (0, 2)
    assert single.shape == (1, 2)
    assert torch.allclose(single, torch.zeros((1, 2), dtype=torch.float32))
    assert all_singletons.shape == (4, 2)
    assert torch.isfinite(all_singletons).all()
    assert torch.unique(all_singletons, dim=0).shape[0] == 4
