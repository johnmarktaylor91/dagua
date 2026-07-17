"""Tests for fidelity-mode dagua_native Graphviz-dot sub-components."""

from __future__ import annotations

from typing import Any, Optional, Sequence

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


def _install_proxy_honest_w5_fixture(
    monkeypatch: pytest.MonkeyPatch,
    base_pos: torch.Tensor,
    proxy_pos: torch.Tensor,
) -> None:
    """Install a fixture where proxy search and honest selection disagree.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    base_pos : torch.Tensor
        Position tensor that the honest final selector should keep.
    proxy_pos : torch.Tensor
        Position tensor that cheap proxy scoring should prefer.
    """
    import dagua.metrics as metrics
    from dagua.layout.ops.pipelines import dagua_native as native
    from dagua.layout.ops.pipelines import native_finisher, native_undirected

    def pos_key(pos: torch.Tensor) -> str:
        """Classify a candidate tensor by value for deterministic fake scoring."""
        if torch.allclose(pos.detach().cpu(), proxy_pos):
            return "proxy"
        if torch.allclose(pos.detach().cpu(), base_pos):
            return "base"
        return "other"

    def fake_quick(
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        node_sizes: Optional[torch.Tensor] = None,
    ) -> dict[str, float]:
        """Return a cheap proxy payload that prefers ``proxy_pos``."""
        del edge_index, node_sizes
        return {"proxy_score": 100.0 if pos_key(pos) == "proxy" else 10.0}

    def fake_full(
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, float]:
        """Return honest metrics that prefer ``base_pos`` over ``proxy_pos``."""
        del edge_index, kwargs
        score = 20.0 if pos_key(pos) == "base" else 15.0
        flow = 0.753 if pos_key(pos) == "base" else 1.0
        return {
            "directed_score": score,
            "undirected_score": score,
            "directed_flow_score": flow,
            "depth_order_score": 0.875,
            "ksm_score": 0.922,
            "edge_length_deviation_score": 0.876,
        }

    def fake_composite(numeric: dict[str, float]) -> float:
        """Return the directed honest score from the fake full payload."""
        return float(numeric["directed_score"])

    def fake_composite_undirected(numeric: dict[str, float]) -> float:
        """Return the undirected honest score from the fake full payload."""
        return float(numeric["undirected_score"])

    def fake_composite_auto(numeric: dict[str, float], directed: bool) -> float:
        """Return the proxy selector score while recording no ruler state."""
        del directed
        return float(numeric["proxy_score"])

    def fake_collinear_dodge(*args: object, **kwargs: object) -> torch.Tensor:
        """Produce the proxy-favored candidate from the polish battery."""
        del args, kwargs
        return proxy_pos.clone()

    def none_candidate(*args: object, **kwargs: object) -> None:
        """Disable unrelated polish candidates for a one-candidate fixture."""
        del args, kwargs
        return None

    def always_eligible(
        candidate: torch.Tensor,
        candidate_input: torch.Tensor,
        node_sizes: torch.Tensor,
        edge_index: torch.Tensor,
    ) -> tuple[bool, str]:
        """Admit the synthetic proxy candidate through the geometry guard."""
        del candidate, candidate_input, node_sizes, edge_index
        return True, ""

    monkeypatch.setattr(metrics, "quick", fake_quick)
    monkeypatch.setattr(metrics, "full", fake_full)
    monkeypatch.setattr(metrics, "composite", fake_composite)
    monkeypatch.setattr(metrics, "composite_undirected", fake_composite_undirected)
    monkeypatch.setattr(metrics, "composite_auto", fake_composite_auto)
    monkeypatch.setattr(native, "_POLISH_SETTINGS", ())
    monkeypatch.setattr(native, "_collinear_dodge", fake_collinear_dodge)
    for candidate_name in (
        "_y_layer_snap",
        "_orthogonal_align",
        "_overlap_jitter",
        "_swap_2opt_anti_crossing",
        "_per_layer_x_kmeans",
        "_global_depth_align",
        "_dot_lattice_lp",
        "_back_edge_relayer",
        "_tutte_cyclic_planar",
        "_gap_validated_layer_swaps",
        "_outerplanar_source_fan_spine",
        "_multi_component_row_major_repack",
        "_median_transpose_polish",
        "_lattice_uniform_centered_slots",
    ):
        monkeypatch.setattr(native, candidate_name, none_candidate)

    def no_predicted_skip(*args: object) -> None:
        """Allow the synthetic W5 call to run."""
        del args
        return None

    def fixed_finisher_slice(config: Optional[LayoutConfig]) -> float:
        """Return a nonzero W5 slice for the synthetic call."""
        del config
        return 1.0

    def ignore_w5_telemetry(*args: object) -> None:
        """Suppress W5 telemetry in the unit fixture."""
        del args
        return None

    monkeypatch.setattr(native_undirected, "_candidate_is_eligible", always_eligible)
    monkeypatch.setattr(native_finisher, "w5_predicted_skip_reason", no_predicted_skip)
    monkeypatch.setattr(native_finisher, "_finisher_slice_s", fixed_finisher_slice)
    monkeypatch.setattr(native_finisher, "log_w5_telemetry", ignore_w5_telemetry)


def _proxy_honest_fixture_tensors() -> tuple[
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
    torch.Tensor,
]:
    """Return tensors that trigger proxy finalist selection in polish tests.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Base positions, proxy positions, edge index, node-size tensors, and
        cluster ids.
    """
    base_pos = torch.tensor(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
        dtype=torch.float32,
    )
    proxy_pos = base_pos + torch.tensor([100.0, 0.0])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 1.0)
    cluster_ids = torch.arange(4, dtype=torch.long)
    return base_pos, proxy_pos, edge_index, node_sizes, cluster_ids


def _terminal_w5_fixture_tensors() -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return candidate-A, terminal, and W5 tensors for terminal-owner tests.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Candidate-A positions, final terminal positions, and synthetic W5
        winner positions, each with shape ``[4, 2]``.
    """
    candidate_a = torch.tensor(
        [[0.0, 0.0], [10.0, 0.0], [0.0, 10.0], [10.0, 10.0]],
        dtype=torch.float32,
    )
    terminal = candidate_a + torch.tensor([100.0, 0.0])
    w5_pos = terminal + torch.tensor([0.0, 25.0])
    return candidate_a, terminal, w5_pos


def _install_terminal_w5_metric_fixture(
    monkeypatch: pytest.MonkeyPatch,
    candidate_a: torch.Tensor,
    terminal: torch.Tensor,
    w5_pos: torch.Tensor,
) -> None:
    """Install deterministic metrics for terminal W5 ownership tests.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture.
    candidate_a : torch.Tensor
        Candidate-A tensor that must not be the terminal W5 incumbent.
    terminal : torch.Tensor
        Final tensor that terminal W5 must score as incumbent.
    w5_pos : torch.Tensor
        Synthetic W5 candidate tensor.
    """
    import dagua.metrics as metrics

    def pos_key(pos: torch.Tensor) -> str:
        """Classify fixture tensors by value."""
        cpu = pos.detach().cpu()
        if torch.allclose(cpu, terminal):
            return "terminal"
        if torch.allclose(cpu, candidate_a):
            return "candidate_a"
        if torch.allclose(cpu, w5_pos):
            return "w5"
        return "other"

    def fake_full(
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        **kwargs: object,
    ) -> dict[str, float]:
        """Return honest metrics keyed to the fixture tensor."""
        del edge_index, kwargs
        key = pos_key(pos)
        directed = {"terminal": 90.0, "candidate_a": 86.0, "w5": 90.2}.get(key, 50.0)
        undirected = {"terminal": 94.0, "candidate_a": 81.0, "w5": 94.0}.get(key, 50.0)
        flow = 0.753 if key == "terminal" else 1.0
        return {
            "directed_score": directed,
            "undirected_score": undirected,
            "directed_flow_score": flow,
            "depth_order_score": 0.875,
            "ksm_score": 0.922,
            "edge_length_deviation_score": 0.876,
        }

    def fake_composite(numeric: dict[str, float]) -> float:
        """Return directed fixture score."""
        return float(numeric["directed_score"])

    def fake_composite_undirected(numeric: dict[str, float]) -> float:
        """Return undirected fixture score."""
        return float(numeric["undirected_score"])

    monkeypatch.setattr(metrics, "full", fake_full)
    monkeypatch.setattr(metrics, "composite", fake_composite)
    monkeypatch.setattr(metrics, "composite_undirected", fake_composite_undirected)


def _terminal_w5_config() -> LayoutConfig:
    """Build a config that reaches the non-component terminal return quickly.

    Returns
    -------
    LayoutConfig
        Native config for unit-level terminal W5 tests.
    """
    return LayoutConfig(
        steps=1,
        edge_equalize_polish=True,
        decompose_components=False,
        route_flat_to_stress=False,
        force_pipeline="hybrid",
    )


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


def test_terminal_w5_incumbent_is_final_return_tensor_and_runs_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal W5 sees the true final tensor while inner W5 sites defer."""
    import importlib

    from dagua.layout.ops.pipelines.native_finisher import (
        W5FinisherResult,
        W5HonestAxes,
        W5ScorePair,
        W5Seed,
        make_w5_skip_result,
    )

    native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    candidate_a, terminal, w5_pos = _terminal_w5_fixture_tensors()
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 1.0)
    _install_terminal_w5_metric_fixture(monkeypatch, candidate_a, terminal, w5_pos)
    captured: dict[str, object] = {"calls": 0, "native_calls": 0}

    monkeypatch.setattr(native_finisher, "w5_predicted_skip_reason", lambda *args: None)
    monkeypatch.setattr(native_finisher, "_finisher_slice_s", lambda config: 1.0)
    monkeypatch.setattr(native_finisher, "log_w5_telemetry", lambda *args: None)

    def fake_run_w5_finisher(
        *,
        incumbent_pos: torch.Tensor,
        incumbent_score_pair: W5ScorePair,
        seeds: Sequence[W5Seed],
        edge_index: torch.Tensor,
        node_sizes: torch.Tensor,
        score_fn: object,
        is_semantically_directed: bool,
        declared_hierarchical: bool,
        direction_is_declared: bool = False,
        config: Optional[LayoutConfig] = None,
        accept_margin: float = 0.05,
        incumbent_axes: Optional[W5HonestAxes] = None,
    ) -> W5FinisherResult:
        """Capture terminal W5 inputs and return a no-op result."""
        del node_sizes, score_fn, accept_margin
        captured["calls"] = int(captured["calls"]) + 1
        captured["incumbent_pos"] = incumbent_pos.detach().cpu()
        captured["incumbent_score_pair"] = incumbent_score_pair
        captured["incumbent_axes"] = incumbent_axes
        captured["seed_names"] = [seed.name for seed in seeds]
        return make_w5_skip_result(
            incumbent_pos=incumbent_pos,
            incumbent_score_pair=incumbent_score_pair,
            reason="unit_noop",
            edge_index=edge_index,
            config=config,
            is_semantically_directed=is_semantically_directed,
            declared_hierarchical=declared_hierarchical,
            direction_is_declared=direction_is_declared,
        )

    def fake_run_native_problem(
        problem: Any,
        state: Any,
        ctx: Any,
        prepared_config: LayoutConfig,
    ) -> torch.Tensor:
        """Simulate a nested contest and an inner polish site before final choice."""
        del state, ctx
        captured["native_calls"] = int(captured["native_calls"]) + 1
        assert bool(getattr(prepared_config, "_dagua_native_defer_w5", False))
        if int(captured["native_calls"]) == 1:
            nested = layout_dagua_native_pipeline(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                node_sizes=problem.node_sizes,
                config=prepared_config,
                device="cpu",
            )
            assert torch.equal(nested, candidate_a)
            native._best_of_polish(
                candidate_a,
                problem.edge_index,
                problem.node_sizes,
                is_semantically_directed=True,
                declared_hierarchical=True,
                config=prepared_config,
            )
            return terminal.to(device=problem.edge_index.device)
        return candidate_a.to(device=problem.edge_index.device)

    monkeypatch.setattr(native_finisher, "run_w5_finisher", fake_run_w5_finisher)
    monkeypatch.setattr(native, "_run_native_problem", fake_run_native_problem)

    actual = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=node_sizes,
        config=_terminal_w5_config(),
        device="cpu",
    )

    assert int(captured["calls"]) == 1
    assert torch.equal(actual, terminal)
    assert torch.equal(captured["incumbent_pos"], terminal)
    assert captured["incumbent_score_pair"] == W5ScorePair(directed=90.0, undirected=94.0)
    assert captured["incumbent_axes"] == W5HonestAxes(
        flow=0.753,
        depth=0.875,
        ksm=0.922,
        edge_length=0.876,
    )
    assert captured["seed_names"][0] == "terminal_final"
    assert "candidate_a" in captured["seed_names"]


def test_terminal_w5_preserves_final_tensor_when_candidate_does_not_dominate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal W5 remains monotone against the exact final incumbent."""
    import importlib

    from dagua.layout.ops.pipelines.native_finisher import (
        W5Candidate,
        W5FinisherResult,
        W5HonestAxes,
        W5ScorePair,
        W5Seed,
    )

    native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    candidate_a, terminal, w5_pos = _terminal_w5_fixture_tensors()
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 1.0)
    one_sided_pair = W5ScorePair(directed=90.2, undirected=94.0)
    _install_terminal_w5_metric_fixture(monkeypatch, candidate_a, terminal, w5_pos)

    monkeypatch.setattr(native_finisher, "w5_predicted_skip_reason", lambda *args: None)
    monkeypatch.setattr(native_finisher, "_finisher_slice_s", lambda config: 1.0)
    monkeypatch.setattr(native_finisher, "log_w5_telemetry", lambda *args: None)

    def fake_run_native_problem(
        problem: Any,
        state: Any,
        ctx: Any,
        prepared_config: LayoutConfig,
    ) -> torch.Tensor:
        """Return the terminal incumbent chosen after candidate-A."""
        del state, ctx, prepared_config
        return terminal.to(device=problem.edge_index.device)

    def fake_run_w5_finisher(
        *,
        incumbent_pos: torch.Tensor,
        incumbent_score_pair: W5ScorePair,
        seeds: Sequence[W5Seed],
        edge_index: torch.Tensor,
        node_sizes: torch.Tensor,
        score_fn: object,
        is_semantically_directed: bool,
        declared_hierarchical: bool,
        direction_is_declared: bool = False,
        config: Optional[LayoutConfig] = None,
        accept_margin: float = 0.05,
        incumbent_axes: Optional[W5HonestAxes] = None,
    ) -> W5FinisherResult:
        """Return a one-sided W5 candidate that must be rejected."""
        del (
            incumbent_pos,
            seeds,
            edge_index,
            node_sizes,
            score_fn,
            is_semantically_directed,
            declared_hierarchical,
            direction_is_declared,
            config,
            accept_margin,
            incumbent_axes,
        )
        accepted = W5Candidate("w5_one_sided", w5_pos, one_sided_pair, "barrier_2d")
        return W5FinisherResult(
            winner_pos=w5_pos,
            incumbent_score_pair=incumbent_score_pair,
            winner_score_pair=one_sided_pair,
            winner_name="w5_one_sided",
            deadline_returned=False,
            accepted=(accepted,),
            rejected=(),
            checkpoints=(),
            mode="barrier_2d",
            steps=1,
        )

    monkeypatch.setattr(native, "_run_native_problem", fake_run_native_problem)
    monkeypatch.setattr(native_finisher, "run_w5_finisher", fake_run_w5_finisher)

    actual = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=node_sizes,
        config=_terminal_w5_config(),
        device="cpu",
    )

    assert torch.equal(actual, terminal)


def test_terminal_w5_noops_on_fidelity_no_budget_and_trivial_graph(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Terminal W5 does not optimize fidelity, no-budget, or n<2 paths."""
    import importlib

    native = importlib.import_module("dagua.layout.ops.pipelines.dagua_native")
    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    candidate_a, terminal, w5_pos = _terminal_w5_fixture_tensors()
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 1.0)
    _install_terminal_w5_metric_fixture(monkeypatch, candidate_a, terminal, w5_pos)
    calls = {"run_w5": 0, "telemetry": 0}

    def fake_run_native_problem(
        problem: Any,
        state: Any,
        ctx: Any,
        prepared_config: LayoutConfig,
    ) -> torch.Tensor:
        """Return a stable terminal tensor."""
        del state, ctx, prepared_config
        return terminal.to(device=problem.edge_index.device)

    def forbidden_run_w5(*args: object, **kwargs: object) -> None:
        """Record forbidden optimizer entry."""
        del args, kwargs
        calls["run_w5"] += 1
        raise AssertionError("terminal no-op path must not run W5")

    def record_telemetry(*args: object) -> None:
        """Record no-budget skip telemetry."""
        del args
        calls["telemetry"] += 1

    monkeypatch.setattr(native, "_run_native_problem", fake_run_native_problem)
    monkeypatch.setattr(native_finisher, "run_w5_finisher", forbidden_run_w5)
    monkeypatch.setattr(native_finisher, "w5_predicted_skip_reason", lambda *args: None)
    monkeypatch.setattr(native_finisher, "_finisher_slice_s", lambda config: None)
    monkeypatch.setattr(native_finisher, "log_w5_telemetry", record_telemetry)

    no_budget = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=node_sizes,
        config=_terminal_w5_config(),
        device="cpu",
    )
    fidelity_config = _terminal_w5_config()
    fidelity_config.fidelity_mode = "faithful"
    fidelity = layout_dagua_native_pipeline(
        edge_index=edge_index,
        num_nodes=4,
        node_sizes=node_sizes,
        config=fidelity_config,
        device="cpu",
    )
    trivial = layout_dagua_native_pipeline(
        edge_index=torch.empty((2, 0), dtype=torch.long),
        num_nodes=1,
        node_sizes=torch.full((1, 2), 1.0),
        config=_terminal_w5_config(),
        device="cpu",
    )

    assert torch.equal(no_budget, terminal)
    assert torch.equal(fidelity, terminal)
    assert torch.equal(trivial, torch.zeros((1, 2)))
    assert calls == {"run_w5": 0, "telemetry": 1}


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

    pos = torch.tensor([[0.0, 0.0], [10.0, 0.0], [5.0, 10.0], [15.0, 10.0]])
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    node_sizes = torch.full((4, 2), 2.0)
    cluster_ids = torch.arange(4, dtype=torch.long)
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
        cluster_ids=cluster_ids,
        polish_battery="default",
    )

    assert observed
    assert set(observed) == {(True, False)}


def test_best_of_polish_w5_receives_final_honest_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """W5 receives the final honest winner, not the pre-final proxy winner."""
    import importlib

    from dagua.layout.ops.pipelines.native_finisher import (
        W5FinisherResult,
        W5HonestAxes,
        W5ScorePair,
        W5Seed,
        make_w5_skip_result,
    )

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    base_pos, proxy_pos, edge_index, node_sizes, cluster_ids = _proxy_honest_fixture_tensors()
    _install_proxy_honest_w5_fixture(monkeypatch, base_pos, proxy_pos)
    captured: dict[str, object] = {"calls": 0}

    def fake_run_w5_finisher(
        *,
        incumbent_pos: torch.Tensor,
        incumbent_score_pair: W5ScorePair,
        seeds: Sequence[W5Seed],
        edge_index: torch.Tensor,
        node_sizes: torch.Tensor,
        score_fn: object,
        is_semantically_directed: bool,
        declared_hierarchical: bool,
        direction_is_declared: bool = False,
        config: Optional[LayoutConfig] = None,
        accept_margin: float = 0.05,
        incumbent_axes: Optional[W5HonestAxes] = None,
    ) -> W5FinisherResult:
        """Capture W5 inputs and return a no-op result."""
        del node_sizes, score_fn, accept_margin
        captured["calls"] = int(captured["calls"]) + 1
        captured["incumbent_pos"] = incumbent_pos.detach().cpu()
        captured["incumbent_score_pair"] = incumbent_score_pair
        captured["incumbent_axes"] = incumbent_axes
        captured["seed_names"] = [seed.name for seed in seeds]
        return make_w5_skip_result(
            incumbent_pos=incumbent_pos,
            incumbent_score_pair=incumbent_score_pair,
            reason="unit_noop",
            edge_index=edge_index,
            config=config,
            is_semantically_directed=is_semantically_directed,
            declared_hierarchical=declared_hierarchical,
            direction_is_declared=direction_is_declared,
        )

    monkeypatch.setattr(native_finisher, "run_w5_finisher", fake_run_w5_finisher)

    polished = _best_of_polish(
        base_pos,
        edge_index,
        node_sizes,
        is_semantically_directed=True,
        declared_hierarchical=True,
        cluster_ids=cluster_ids,
        config=LayoutConfig(),
    )

    assert int(captured["calls"]) == 1
    assert torch.equal(polished, base_pos)
    assert torch.equal(captured["incumbent_pos"], base_pos)
    assert captured["incumbent_score_pair"] == W5ScorePair(directed=20.0, undirected=20.0)
    assert captured["incumbent_axes"] == W5HonestAxes(
        flow=0.753,
        depth=0.875,
        ksm=0.922,
        edge_length=0.876,
    )
    assert captured["seed_names"][:2] == ["incumbent", "proxy_polish_winner"]


def test_best_of_polish_returns_w5_candidate_only_when_dominating_final_winner(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A W5 result can replace the final winner only by dual dominance."""
    import importlib

    from dagua.layout.ops.pipelines.native_finisher import (
        W5Candidate,
        W5FinisherResult,
        W5HonestAxes,
        W5ScorePair,
        W5Seed,
    )

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    base_pos, proxy_pos, edge_index, node_sizes, cluster_ids = _proxy_honest_fixture_tensors()
    _install_proxy_honest_w5_fixture(monkeypatch, base_pos, proxy_pos)
    w5_pos = base_pos + torch.tensor([0.0, 25.0])
    winner_pair = W5ScorePair(directed=20.2, undirected=20.2)
    captured: dict[str, W5ScorePair] = {}

    def fake_run_w5_finisher(
        *,
        incumbent_pos: torch.Tensor,
        incumbent_score_pair: W5ScorePair,
        seeds: Sequence[W5Seed],
        edge_index: torch.Tensor,
        node_sizes: torch.Tensor,
        score_fn: object,
        is_semantically_directed: bool,
        declared_hierarchical: bool,
        direction_is_declared: bool = False,
        config: Optional[LayoutConfig] = None,
        accept_margin: float = 0.05,
        incumbent_axes: Optional[W5HonestAxes] = None,
    ) -> W5FinisherResult:
        """Return a W5 winner that dominates the final honest incumbent."""
        del (
            incumbent_pos,
            seeds,
            edge_index,
            node_sizes,
            score_fn,
            is_semantically_directed,
            declared_hierarchical,
            direction_is_declared,
            config,
            accept_margin,
            incumbent_axes,
        )
        captured["incumbent"] = incumbent_score_pair
        accepted = W5Candidate("w5_unit", w5_pos, winner_pair, "barrier_2d")
        return W5FinisherResult(
            winner_pos=w5_pos,
            incumbent_score_pair=incumbent_score_pair,
            winner_score_pair=winner_pair,
            winner_name="w5_unit",
            deadline_returned=False,
            accepted=(accepted,),
            rejected=(),
            checkpoints=(),
            mode="barrier_2d",
            steps=1,
        )

    monkeypatch.setattr(native_finisher, "run_w5_finisher", fake_run_w5_finisher)

    polished = _best_of_polish(
        base_pos,
        edge_index,
        node_sizes,
        is_semantically_directed=True,
        declared_hierarchical=True,
        cluster_ids=cluster_ids,
        config=LayoutConfig(),
    )

    assert captured["incumbent"] == W5ScorePair(directed=20.0, undirected=20.0)
    assert winner_pair.directed > captured["incumbent"].directed + 0.05
    assert winner_pair.undirected > captured["incumbent"].undirected + 0.05
    assert torch.equal(polished, w5_pos)


def test_best_of_polish_preserves_final_winner_when_w5_does_not_dominate(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A non-dominating W5 winner is a no-op against the final honest winner."""
    import importlib

    from dagua.layout.ops.pipelines.native_finisher import (
        W5Candidate,
        W5FinisherResult,
        W5HonestAxes,
        W5ScorePair,
        W5Seed,
    )

    native_finisher = importlib.import_module("dagua.layout.ops.pipelines.native_finisher")
    base_pos, proxy_pos, edge_index, node_sizes, cluster_ids = _proxy_honest_fixture_tensors()
    _install_proxy_honest_w5_fixture(monkeypatch, base_pos, proxy_pos)
    w5_pos = base_pos + torch.tensor([0.0, 25.0])
    one_sided_pair = W5ScorePair(directed=20.2, undirected=20.0)

    def fake_run_w5_finisher(
        *,
        incumbent_pos: torch.Tensor,
        incumbent_score_pair: W5ScorePair,
        seeds: Sequence[W5Seed],
        edge_index: torch.Tensor,
        node_sizes: torch.Tensor,
        score_fn: object,
        is_semantically_directed: bool,
        declared_hierarchical: bool,
        direction_is_declared: bool = False,
        config: Optional[LayoutConfig] = None,
        accept_margin: float = 0.05,
        incumbent_axes: Optional[W5HonestAxes] = None,
    ) -> W5FinisherResult:
        """Return a W5 winner that fails the unchanged dual-ruler gate."""
        del (
            incumbent_pos,
            seeds,
            edge_index,
            node_sizes,
            score_fn,
            is_semantically_directed,
            declared_hierarchical,
            direction_is_declared,
            config,
            accept_margin,
            incumbent_axes,
        )
        accepted = W5Candidate("w5_one_sided", w5_pos, one_sided_pair, "barrier_2d")
        return W5FinisherResult(
            winner_pos=w5_pos,
            incumbent_score_pair=incumbent_score_pair,
            winner_score_pair=one_sided_pair,
            winner_name="w5_one_sided",
            deadline_returned=False,
            accepted=(accepted,),
            rejected=(),
            checkpoints=(),
            mode="barrier_2d",
            steps=1,
        )

    monkeypatch.setattr(native_finisher, "run_w5_finisher", fake_run_w5_finisher)

    polished = _best_of_polish(
        base_pos,
        edge_index,
        node_sizes,
        is_semantically_directed=True,
        declared_hierarchical=True,
        cluster_ids=cluster_ids,
        config=LayoutConfig(),
    )

    assert torch.equal(polished, base_pos)


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
