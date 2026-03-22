"""Tests for the algorithm variant benchmark registry and runner helpers."""

from __future__ import annotations

import importlib
import inspect
import sys
import types
from pathlib import Path
from typing import Any

from dagua.eval.competitors.base import get_competitor
from dagua.eval.competitors.classic_competitor import _CLASSIC_LAYOUT_SPECS
from dagua.eval.variants import (
    VARIANT_REGISTRY,
    engine_is_stochastic,
    get_variant,
    original_variant_name,
)
from dagua.graph import DaguaGraph
from scripts import run_benchmark
from scripts.run_benchmark import BenchmarkRecord, WorkItem


def _callable_param_names(base_engine: str) -> set[str]:
    """Return supported keyword parameter names for one classic layout.

    Parameters
    ----------
    base_engine : str
        Base classic competitor name.

    Returns
    -------
    set[str]
        Supported layout keyword names excluding the common graph inputs.
    """
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
    """Every registry entry should point at a registered classic base engine."""
    assert len(VARIANT_REGISTRY) == 97
    for variant in VARIANT_REGISTRY:
        assert variant.base_engine in _CLASSIC_LAYOUT_SPECS
        assert get_variant(variant.variant_id) == variant


def test_variant_params_are_valid_for_reimpl() -> None:
    """Reimplementation variant params should match callable signatures."""
    for variant in VARIANT_REGISTRY:
        allowed_param_names = _callable_param_names(variant.base_engine)
        assert set(variant.reimpl_params).issubset(allowed_param_names)


def test_original_params_mappable_where_claimed() -> None:
    """Original-side params should only use adapter-supported names."""
    for variant in VARIANT_REGISTRY:
        if variant.original_engine is None:
            continue
        competitor = get_competitor(variant.original_engine)
        assert competitor is not None
        assert set(variant.original_params).issubset(set(competitor.variant_param_names))


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
