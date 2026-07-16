"""Tests for fidelity-mode dagua_native Graphviz-dot sub-components."""

from __future__ import annotations

from typing import Any

import pytest
import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.pipelines.dagua_native import (
    _anytime_fallback_positions,
    _apply_dot_cluster_fidelity_layout,
    _best_of_polish,
    _build_dot_cluster_skeletons,
    _collinear_dodge,
    _dot_rank_assignment,
    _is_graphviz_dot_cluster_fidelity_mode,
    layout_dagua_native_pipeline,
)


class _WorkerLayoutTimeoutError(RuntimeError):
    """Local stand-in for the benchmark worker alarm exception."""


def _gate_row_graph() -> tuple[torch.Tensor, torch.Tensor, int]:
    """Return a connected graph that triggers the old large-row fallback gate.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, int]
        Edge-index tensor, node-size tensor, and node count.
    """
    num_nodes = 250
    source = torch.arange(700, dtype=torch.long).remainder(num_nodes)
    target = (source * 37 + 11).remainder(num_nodes)
    edge_index = torch.stack((source, target), dim=0)
    node_sizes = torch.full((num_nodes, 2), 2.0)
    return edge_index, node_sizes, num_nodes


def _deadline_gate_config() -> LayoutConfig:
    """Build a config carrying benchmark-deadline metadata for gate tests.

    Returns
    -------
    LayoutConfig
        Native config that triggers the prelayout fallback registration path.
    """
    config = LayoutConfig(
        steps=1,
        edge_equalize_polish=False,
        decompose_components=False,
        route_flat_to_stress=False,
        force_pipeline="hybrid",
    )
    config._dagua_native_deadline_s = 9999999999.0
    config._dagua_native_total_budget_s = 300.0
    return config


def test_gate_row_deadline_runs_real_pipeline_not_prelayout_fallback(
    monkeypatch: Any,
) -> None:
    """A deadline-gated large row must not return the deterministic fallback."""
    import importlib

    native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    edge_index, node_sizes, num_nodes = _gate_row_graph()
    real_pipeline_pos = torch.stack(
        (
            torch.arange(num_nodes, dtype=torch.float32),
            torch.arange(num_nodes, dtype=torch.float32) + 1000.0,
        ),
        dim=1,
    )

    def fake_run_native_problem(
        problem: Any,
        state: Any,
        ctx: Any,
        config: Any,
    ) -> torch.Tensor:
        """Return a distinct finished pipeline tensor for the wired path."""
        del state, ctx, config
        return real_pipeline_pos.to(device=problem.edge_index.device)

    monkeypatch.setattr(native, "_run_native_problem", fake_run_native_problem)

    actual = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        config=_deadline_gate_config(),
        device="cpu",
    )
    fallback = _anytime_fallback_positions(
        edge_index,
        num_nodes,
        node_sizes,
        None,
        torch.device("cpu"),
    )

    assert torch.equal(actual, real_pipeline_pos)
    assert not torch.equal(actual, fallback)


def test_worker_timeout_returns_registered_prelayout_fallback(
    monkeypatch: Any,
) -> None:
    """Worker timeout exits return the anytime register, not live optimizer state."""
    import importlib

    native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    edge_index, node_sizes, num_nodes = _gate_row_graph()

    def raising_run_native_problem(
        problem: Any,
        state: Any,
        ctx: Any,
        config: Any,
    ) -> torch.Tensor:
        """Raise the benchmark worker-timeout sentinel after registration."""
        del problem, state, ctx, config
        raise _WorkerLayoutTimeoutError("worker layout timeout exceeded")

    monkeypatch.setattr(native, "_run_native_problem", raising_run_native_problem)

    actual = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        config=_deadline_gate_config(),
        device="cpu",
    )
    fallback = _anytime_fallback_positions(
        edge_index,
        num_nodes,
        node_sizes,
        None,
        torch.device("cpu"),
    )

    assert torch.equal(actual, fallback)
    assert bool(torch.isfinite(actual).all().item())


def test_worker_timeout_reraises_without_anytime_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-gate row with no admitted milestone must re-raise worker timeout."""
    import importlib

    native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 2.0)
    config = LayoutConfig(
        steps=1,
        edge_equalize_polish=False,
        decompose_components=False,
        route_flat_to_stress=False,
        force_pipeline="hybrid",
    )

    def raising_run_native_problem(
        problem: Any,
        state: Any,
        ctx: Any,
        prepared_config: Any,
    ) -> torch.Tensor:
        """Raise before any milestone can populate the anytime register."""
        del problem, state, ctx, prepared_config
        raise _WorkerLayoutTimeoutError("worker layout timeout exceeded")

    monkeypatch.setattr(native, "_run_native_problem", raising_run_native_problem)

    with pytest.raises(_WorkerLayoutTimeoutError):
        layout_dagua_native_pipeline(
            edge_index=edge_index,
            num_nodes=4,
            node_sizes=node_sizes,
            config=config,
            device="cpu",
        )


def test_worker_timeout_returns_cloned_anytime_register(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Timeout fallback must preserve the admitted tensor against later mutation."""
    import importlib

    native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 2.0)
    admitted = torch.tensor(
        [[0.0, 0.0], [10.0, 0.0], [20.0, 0.0], [30.0, 0.0]],
        dtype=torch.float32,
    )
    expected = admitted.clone()
    config = LayoutConfig(
        steps=1,
        edge_equalize_polish=False,
        decompose_components=False,
        route_flat_to_stress=False,
        force_pipeline="hybrid",
    )

    def mutating_run_native_problem(
        problem: Any,
        state: Any,
        ctx: Any,
        prepared_config: Any,
    ) -> torch.Tensor:
        """Register a milestone, mutate its source tensor, then time out."""
        del problem, state, ctx
        register_anytime_best = getattr(prepared_config, "_dagua_native_register_anytime_best")
        register_anytime_best(admitted, "post_base_contest")
        admitted.add_(1000.0)
        raise _WorkerLayoutTimeoutError("worker layout timeout exceeded")

    monkeypatch.setattr(native, "_run_native_problem", mutating_run_native_problem)

    actual = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=node_sizes,
        config=config,
        device="cpu",
    )

    assert torch.equal(actual, expected)
    assert not torch.equal(actual, admitted)


def test_collinear_dodge_moves_blocker_off_skip_edge() -> None:
    """A node centered on a non-incident skip edge is shifted perpendicular."""
    pos = torch.tensor([[0.0, 0.0], [0.0, 10.0], [0.0, 20.0]])
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)

    dodged = _collinear_dodge(pos, edge_index)

    assert dodged is not None
    assert torch.equal(dodged[[0, 2]], pos[[0, 2]])
    assert float(torch.abs(dodged[1, 0]).item()) > 0.0
    assert float(dodged[1, 1].item()) == float(pos[1, 1].item())


def test_directed_polish_rejects_degenerate_geometry_candidate(monkeypatch) -> None:
    """Shared directed polish routes geometry through the degeneracy guard."""
    import importlib

    native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")

    pos = torch.tensor([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    node_sizes = torch.full((3, 2), 10.0)

    monkeypatch.setattr(native, "_POLISH_SETTINGS", ())
    monkeypatch.setattr(native, "_collinear_dodge", lambda *args, **kwargs: torch.zeros_like(pos))
    polished = _best_of_polish(
        pos,
        edge_index,
        node_sizes,
        is_semantically_directed=True,
        declared_hierarchical=True,
    )

    assert torch.equal(polished, pos)


def test_polish_candidate_memory_error_is_skipped(monkeypatch: object) -> None:
    """A failing polish candidate must not sink the full solve."""
    import importlib

    native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")

    pos = torch.tensor([[0.0, 0.0], [0.0, 100.0], [0.0, 200.0]])
    edge_index = torch.tensor([[0, 0], [1, 2]], dtype=torch.long)
    node_sizes = torch.full((3, 2), 10.0)

    def raise_memory_error(*args: object, **kwargs: object) -> torch.Tensor:
        """Raise the same exception class as an oversized LP allocation."""
        del args, kwargs
        raise MemoryError("synthetic polish allocation failure")

    monkeypatch.setattr(native, "_POLISH_SETTINGS", ())
    monkeypatch.setattr(native, "_collinear_dodge", raise_memory_error)

    polished = _best_of_polish(
        pos,
        edge_index,
        node_sizes,
        is_semantically_directed=True,
        declared_hierarchical=True,
    )

    assert polished.shape == pos.shape
    assert bool(torch.isfinite(polished).all().item())


def test_polish_scores_cyclic_digraph_with_common_ruler(monkeypatch) -> None:
    """Cyclic directed polish candidates use the benchmark's common table."""
    import dagua.metrics as metrics

    pos = torch.tensor([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    node_sizes = torch.full((3, 2), 2.0)
    observed: list[tuple[bool, bool]] = []

    def fake_full(*args: object, **kwargs: object) -> dict[str, float]:
        """Return minimal numeric metrics for selector-routing inspection."""
        del args, kwargs
        return {"neighborhood_preservation_score": 1.0}

    def fake_composite_auto(numeric: dict[str, float], directed: bool) -> float:
        """Record the semantic and hierarchy flags passed by the selector."""
        observed.append((directed, bool(numeric["declared_hierarchical"])))
        return 50.0

    monkeypatch.setattr(metrics, "full", fake_full)
    monkeypatch.setattr(metrics, "composite_auto", fake_composite_auto)

    _best_of_polish(
        pos,
        edge_index,
        node_sizes,
        is_semantically_directed=True,
        declared_hierarchical=False,
        polish_battery="default",
    )

    assert observed
    assert set(observed) == {(True, False)}


def test_dense_collinear_dodge_is_skipped_before_blocker_scan() -> None:
    """Dense O(E*N) blocker detection is capped on a 300-node graph."""
    n = 300
    pos = torch.stack((torch.arange(n, dtype=torch.float32), torch.zeros(n)), dim=1)
    edge_index = torch.triu_indices(n, n, offset=1)

    assert _collinear_dodge(pos, edge_index) is None


def test_dot_cluster_skeleton_counts_match_cluster_c_build_skeleton() -> None:
    """Golden vector for Graphviz ``cluster.c:build_skeleton`` counters."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 3],
            [1, 2, 2, 4],
        ],
        dtype=torch.long,
    )
    ranks = (0, 1, 2, 0, 1)
    clusters = {"cluster_a": (0, 1, 2), "cluster_b": (3, 4)}

    skeletons = _build_dot_cluster_skeletons(
        clusters=clusters,
        cluster_parents=None,
        ranks=ranks,
        edge_index=edge_index,
    )

    by_name = {skeleton.name: skeleton for skeleton in skeletons}
    assert by_name["cluster_a"].rankleader_ranks == (0, 1, 2)
    assert by_name["cluster_a"].rankleader_uf_sizes == (1, 1, 1)
    assert by_name["cluster_a"].skeleton_edge_counts == (2, 2)
    assert by_name["cluster_b"].rankleader_ranks == (0, 1)
    assert by_name["cluster_b"].rankleader_uf_sizes == (1, 1)
    assert by_name["cluster_b"].skeleton_edge_counts == (1,)


def test_dot_cluster_skeleton_collapses_multi_node_rank_uf_size() -> None:
    """Graphviz decrements rankleader UF size when a rank has multiple nodes."""
    edge_index = torch.tensor([[0, 1], [2, 2]], dtype=torch.long)
    ranks = (0, 0, 1)

    (skeleton,) = _build_dot_cluster_skeletons(
        clusters={"cluster_a": (0, 1, 2)},
        cluster_parents=None,
        ranks=ranks,
        edge_index=edge_index,
    )

    assert skeleton.rankleader_ranks == (0, 1)
    assert skeleton.rankleader_uf_sizes == (1, 1)
    assert skeleton.skeleton_edge_counts == (2,)


def test_dot_cluster_fidelity_layout_separates_sibling_cluster_boxes() -> None:
    """Fidelity cluster layout should reserve non-overlapping sibling blocks."""
    edge_index = torch.tensor(
        [
            [0, 1, 3, 4, 2],
            [1, 2, 4, 5, 3],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((6, 2), 20.0, dtype=torch.float32)
    base_pos = torch.zeros((6, 2), dtype=torch.float32)
    clusters = {"left": (0, 1, 2), "right": (3, 4, 5)}
    ranks = _dot_rank_assignment(edge_index, 6)

    out = _apply_dot_cluster_fidelity_layout(
        base_pos,
        edge_index,
        node_sizes,
        clusters,
        cluster_parents=None,
    )

    left_max = float((out[[0, 1, 2], 0] + 10.0).max().item())
    right_min = float((out[[3, 4, 5], 0] - 10.0).min().item())
    assert right_min > left_max
    rank_mean = sum(ranks) / len(ranks)
    for node, rank in enumerate(ranks):
        assert float(out[node, 1].item()) == float((rank - rank_mean) * 72.0)


def test_dagua_native_pipeline_cluster_fidelity_mode_is_invokable() -> None:
    """The public native pipeline should accept the narrow cluster fidelity mode."""
    edge_index = torch.tensor(
        [
            [0, 0, 1, 3],
            [1, 3, 2, 4],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((5, 2), 24.0, dtype=torch.float32)
    config = LayoutConfig(
        algorithm="dagua_native",
        steps=2,
        edge_equalize_polish=False,
        force_pipeline="layered_dag",
    )

    out = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=5,
        node_sizes=node_sizes,
        config=config,
        clusters={"cluster_left": (0, 1, 2), "cluster_right": (3, 4)},
        fidelity_mode="dot_clusters",
    )

    assert _is_graphviz_dot_cluster_fidelity_mode("dot_clusters")
    assert out.shape == (5, 2)
    assert torch.isfinite(out).all()
