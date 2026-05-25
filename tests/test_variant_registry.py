"""Tests for the algorithm variant benchmark registry and runner helpers."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from pathlib import Path
from typing import Any, Mapping, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, get_competitor
from dagua.eval.competitors.classic_competitor import (
    _CLASSIC_LAYOUT_SPECS,
    ChainCompetitor,
    VariantCompetitor,
)
from dagua.eval.variants import (
    VARIANT_REGISTRY,
    engine_is_stochastic,
    get_variant,
    original_variant_name,
)
from dagua.graph import DaguaGraph
from scripts import run_benchmark
from scripts.run_benchmark import BenchmarkRecord, WorkItem

_PLANNED_BASE_ENGINES = frozenset({"cytoscape_fcose", "gephi_yifanhu"})


class _FakeCompetitor(CompetitorBase):
    """Test double for validating chained-competitor warm starts."""

    def __init__(
        self,
        name: str,
        pos: Optional[torch.Tensor],
        max_nodes: int,
        error: Optional[str] = None,
    ) -> None:
        """Initialize the fake competitor.

        Parameters
        ----------
        name : str
            Synthetic competitor name.
        pos : torch.Tensor | None
            Position tensor returned by the fake layout call.
        max_nodes : int
            Maximum graph size reported by the fake adapter.
        error : str | None, default=None
            Optional error message returned by the fake layout call.
        """
        self.name = name
        self.max_nodes = max_nodes
        self._pos = pos
        self._error = error
        self.calls: list[dict[str, Any]] = []

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Delegate to ``layout_with_variant`` for this test double.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional benchmark seed.

        Returns
        -------
        CompetitorResult
            Synthetic layout result.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Record the call parameters and return the configured result.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Maximum runtime in seconds.
        seed : int | None, default=None
            Optional benchmark seed.
        variant_params : Mapping[str, Any] | None, default=None
            Variant overrides supplied by the caller.

        Returns
        -------
        CompetitorResult
            Synthetic layout result with the configured position or error.
        """
        del graph
        self.calls.append(
            {
                "timeout": timeout,
                "seed": seed,
                "variant_params": None if variant_params is None else dict(variant_params),
            }
        )
        return CompetitorResult(
            name=self.name,
            pos=self._pos,
            runtime_seconds=0.0,
            error=self._error,
        )


def _callable_param_names(base_engine: str) -> set[str]:
    """Return supported keyword parameter names for one variant-capable base.

    Parameters
    ----------
    base_engine : str
        Base competitor name.

    Returns
    -------
    set[str]
        Supported layout keyword names excluding the common graph inputs.
    """
    competitor = get_competitor(base_engine)
    if competitor is not None and base_engine not in _CLASSIC_LAYOUT_SPECS:
        return set(competitor.variant_param_names)

    spec = _CLASSIC_LAYOUT_SPECS[base_engine]
    module = importlib.import_module(spec.import_path)
    function = getattr(module, spec.function_name)
    signature = inspect.signature(function)
    return {
        name
        for name in signature.parameters
        if name not in {"edge_index", "num_nodes", "node_sizes", "seed", "trace_every"}
    }


def _test_graph() -> DaguaGraph:
    """Create a tiny graph for grouped timeout tests.

    Returns
    -------
    DaguaGraph
        Two-edge path graph with computed node sizes.
    """
    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    graph.compute_node_sizes()
    return graph


def test_all_variants_have_valid_base_engine() -> None:
    """Every registry entry should point at a usable or planned base engine."""
    assert len(VARIANT_REGISTRY) == 120
    for variant in VARIANT_REGISTRY:
        assert (
            variant.base_engine in _CLASSIC_LAYOUT_SPECS
            or get_competitor(variant.base_engine) is not None
            or variant.base_engine in _PLANNED_BASE_ENGINES
        )
        assert get_variant(variant.variant_id) == variant


def test_variant_params_are_valid_for_reimpl() -> None:
    """Reimplementation variant params should match callable signatures."""
    for variant in VARIANT_REGISTRY:
        if variant.base_engine in _PLANNED_BASE_ENGINES:
            continue
        allowed_param_names = _callable_param_names(variant.base_engine)
        assert set(variant.reimpl_params).issubset(allowed_param_names)


def test_classic_fa2_defaults_enable_barnes_hut() -> None:
    """The FA2 benchmark defaults should match the reference BH mode."""
    default_params = _CLASSIC_LAYOUT_SPECS["classic_fa2"].default_params

    assert default_params["barnes_hut"] is True
    assert default_params["barnes_hut_theta"] == 1.2


def test_fa2_default_variants_enable_barnes_hut() -> None:
    """All FA2 reference-matched variants should opt into Barnes-Hut."""
    variant_ids = (
        "classic_fa2_default",
        "classic_fa2_gravity0",
        "classic_fa2_gravity2",
        "classic_fa2_scaling1",
        "classic_fa2_scaling4",
        "classic_fa2_strong_gravity",
        "classic_fa2_no_outbound",
        "classic_fa2_dissuade_hubs",
        "classic_fa2_linlog",
    )

    for variant_id in variant_ids:
        variant = get_variant(variant_id)
        assert variant is not None
        assert variant.reimpl_params["barnes_hut"] is True
        assert variant.reimpl_params["barnes_hut_theta"] == 1.2


def test_fa2_linlog_variant_is_not_true_original() -> None:
    """LinLog should use the proxy path because the reference asserts."""
    variant = get_variant("classic_fa2_linlog")

    assert variant is not None
    assert variant.is_true_original is False


def test_classic_sgd2_multi_registry_defaults_match_benchmark_reference() -> None:
    """Classic SGD2 defaults should include the benchmark learning rate."""
    assert _CLASSIC_LAYOUT_SPECS["classic_sgd2_multi"].default_params == {
        "criteria": {"stress": 1.0, "ideal_edge_length": 1.0},
        "lr": 0.01,
    }


def test_sgd2_multi_variants_pin_steps_and_grad_clamp() -> None:
    """All SGD2 variants should pin the agreed iteration and clamp settings."""
    variant_ids = (
        "classic_sgd2_multi_default",
        "classic_sgd2_multi_stress_only",
        "classic_sgd2_multi_with_crossing",
        "classic_sgd2_multi_with_aspect",
        "classic_sgd2_multi_lr001",
        "classic_sgd2_multi_lr01",
        "classic_sgd2_multi_batch8",
        "classic_sgd2_multi_batch128",
    )

    for variant_id in variant_ids:
        variant = get_variant(variant_id)
        assert variant is not None
        assert variant.reimpl_params["steps"] == 2000
        assert variant.reimpl_params["grad_clamp"] == 5.0
        assert variant.original_params["max_iter"] == 2000
        assert variant.original_params["grad_clamp"] == 5.0


def test_round31_infra_variant_max_node_caps() -> None:
    """Sample-recovery variants should cap graph sizes before dispatch."""
    expected_caps = {
        "classic_neulay_default": 1500,
        "classic_neulay_lr001": 1500,
        "classic_neulay_lr05": 1500,
        "classic_neulay_radius02": 1500,
        "classic_neulay_radius08": 1500,
        "classic_neulay_no_gcn": 1500,
        "classic_sgd2_multi_default": 2000,
        "classic_sgd2_multi_stress_only": 2000,
        "classic_sgd2_multi_with_crossing": 500,
        "classic_sgd2_multi_with_aspect": 2000,
        "classic_sgd2_multi_lr001": 2000,
        "classic_sgd2_multi_lr01": 2000,
        "classic_sgd2_multi_batch8": 2000,
        "classic_sgd2_multi_batch128": 2000,
        "classic_davidson_harel_rounds50": 300,
        "classic_davidson_harel_rounds100": 300,
        "classic_davidson_harel_rounds200": 300,
    }

    for variant_id, max_nodes in expected_caps.items():
        variant = get_variant(variant_id)
        assert variant is not None
        assert variant.max_nodes == max_nodes


def test_variant_competitor_honors_explicit_max_nodes() -> None:
    """Variant caps should be able to widen overly conservative base caps."""
    base = _FakeCompetitor(name="base", pos=None, max_nodes=50)
    variant = VariantCompetitor(
        base_competitor=base,
        variant_params={},
        name="base_variant",
        max_nodes=300,
    )

    assert variant.max_nodes == 300


def test_original_params_mappable_where_claimed() -> None:
    """Original-side params should only use adapter-supported names."""
    for variant in VARIANT_REGISTRY:
        if variant.original_engine is None:
            continue
        competitor = get_competitor(variant.original_engine)
        assert competitor is not None
        assert set(variant.original_params).issubset(set(competitor.variant_param_names))


def test_classic_embedding_variant_registry_matches_reference_defaults() -> None:
    """Embedding variants should keep the reference-aligned parameter budgets."""
    assert _CLASSIC_LAYOUT_SPECS["classic_neulay"].default_params == {
        "steps": 20_000,
        "gcn_steps": 2_000,
        "use_gcn": True,
        "lr": 0.1,
        "radius": 0.4,
    }

    tsnet_steps200 = get_variant("classic_tsnet_steps200")
    assert tsnet_steps200 is not None
    assert tsnet_steps200.reimpl_params["steps"] == 250
    assert tsnet_steps200.original_params["max_iter"] == 250

    neulay_no_gcn = get_variant("classic_neulay_no_gcn")
    assert neulay_no_gcn is not None
    assert "gcn_steps" not in neulay_no_gcn.reimpl_params
    assert "gcn_steps" not in neulay_no_gcn.original_params


def test_stochastic_flag_consistency() -> None:
    """Variant stochastic flags should match runner classification."""
    for variant in VARIANT_REGISTRY:
        assert variant.is_stochastic is engine_is_stochastic(variant.variant_id)
        original_name = original_variant_name(variant)
        if original_name is not None:
            competitor = get_competitor(variant.original_engine or "")
            assert competitor is not None
            assert engine_is_stochastic(original_name) is engine_is_stochastic(
                variant.original_engine or ""
            )


def test_planned_variant_bases_remain_unpaired() -> None:
    """Placeholder variant bases should not advertise original pairings."""
    for variant in VARIANT_REGISTRY:
        if variant.base_engine in _PLANNED_BASE_ENGINES:
            assert variant.original_engine is None


def test_chain_competitor_warmstarts_second_pass() -> None:
    """Chain competitors should forward first-pass positions into ``pos``."""
    graph = _test_graph()
    first_pos = torch.tensor([[0.0, 1.0], [1.0, 0.0], [2.0, 1.0]], dtype=torch.float32)
    second_pos = torch.tensor([[2.0, 2.0], [3.0, 3.0], [4.0, 4.0]], dtype=torch.float32)
    first = _FakeCompetitor(name="first", pos=first_pos, max_nodes=12)
    second = _FakeCompetitor(name="second", pos=second_pos, max_nodes=5)
    competitor = ChainCompetitor(
        first_competitor=first,
        second_competitor=second,
        name="chain",
        first_params={"steps": 50},
        second_params={"steps": 300},
    )

    result = competitor.layout_with_variant(
        graph,
        timeout=120.0,
        seed=11,
        variant_params={"steps": 125},
    )

    assert competitor.max_nodes == 5
    assert result.pos is second_pos
    assert first.calls == [{"timeout": 60.0, "seed": 11, "variant_params": {"steps": 50}}]
    assert len(second.calls) == 1
    assert 10.0 <= second.calls[0]["timeout"] <= 120.0
    assert second.calls[0]["seed"] == 11
    assert second.calls[0]["variant_params"] == {"steps": 125, "pos": first_pos}


def test_skip_after_timeout_serial(monkeypatch: Any, tmp_path: Path) -> None:
    """Grouped execution should skip remaining seeds after three timeouts."""
    graph = _test_graph()
    test_graph = types.SimpleNamespace(name="tiny_timeout_graph", graph=graph, tags=set())
    work_group = [
        WorkItem(
            graph_name="tiny_timeout_graph",
            engine_name="classic_fr_steps50",
            seed=seed,
            timeout_seconds=1.0,
            output_dir=str(tmp_path),
            save_positions=False,
        )
        for seed in (42, 43, 44, 45)
    ]

    timeout_record = BenchmarkRecord(
        graph_name="tiny_timeout_graph",
        engine_name="classic_fr_steps50",
        seed=42,
        status="timeout",
        runtime_seconds=1.0,
        error=None,
        positions_file=None,
        num_nodes=graph.num_nodes,
        num_edges=int(graph.edge_index.shape[1]),
        is_stochastic=True,
        skip_reason=None,
        original_for=[],
        reimpl_of=[],
    )

    call_count = {"count": 0}

    def _fake_run_single_work_item(work_item: WorkItem) -> dict[str, Any]:
        """Return three timeout records before grouped skipping kicks in."""
        del work_item
        call_count["count"] += 1
        return timeout_record.to_dict()

    monkeypatch.setattr(
        run_benchmark,
        "_ensure_worker_cache",
        lambda: ({}, {"tiny_timeout_graph": test_graph}),
    )
    monkeypatch.setattr(run_benchmark, "_run_single_work_item", _fake_run_single_work_item)

    payloads = run_benchmark._run_work_group(work_group)
    records = [BenchmarkRecord.from_dict(payload) for payload in payloads]

    assert call_count["count"] == 3
    assert [record.status for record in records] == ["timeout", "timeout", "timeout", "skipped"]
    assert records[-1].skip_reason == "skipped after 3 consecutive timeouts"


def test_cli_workers_auto(monkeypatch: Any) -> None:
    """Worker auto-detection should honor CPU, RAM, and the CLI parser."""

    class _VirtualMemory:
        """Minimal psutil ``virtual_memory`` return value."""

        available = 20 * 1024**3

    fake_psutil = types.SimpleNamespace(virtual_memory=lambda: _VirtualMemory())
    monkeypatch.setitem(sys.modules, "psutil", fake_psutil)
    monkeypatch.setattr(run_benchmark.os, "cpu_count", lambda: 8)

    monkeypatch.setattr(sys, "argv", ["run_benchmark.py", "--workers", "auto"])
    parsed_args = run_benchmark.parse_args()

    assert parsed_args.workers == "auto"
    assert run_benchmark.resolve_worker_count("auto") == 5
