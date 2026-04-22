"""Sprint 0 Task 0.2 regression coverage: default dispatch contract.

Asserts that:
- `algorithm=None` routes through the dagua_native ops pipeline (build_dagua_pipeline called).
- `algorithm="dagua_native"` (explicit) goes through the same pipeline path.
- `algorithm="_legacy"` uses the pre-decomposition engine body (no pipeline call).
- `trace` argument forces the legacy path (op-level snapshots not yet wired) AND
  emits a DeprecationWarning so users see the fork.
- `relax_steps>0` likewise falls back to legacy with a DeprecationWarning.

These guard the central Sprint 0 routing contract; without them a future
refactor could silently flip the default away from the pipeline.
"""

from __future__ import annotations

import warnings
from unittest.mock import patch

import dagua
from dagua.layout.engine import layout as engine_layout
from dagua.layout.ops.pipelines import dagua_native as dn_module


def _build_chain_graph(n: int = 10) -> dagua.DaguaGraph:
    g = dagua.DaguaGraph()
    for i in range(n):
        g.add_node(f"n{i}")
    for i in range(n - 1):
        g.add_edge(f"n{i}", f"n{i + 1}")
    return g


def _trace_pipeline_calls():
    """Wrap build_dagua_pipeline to record invocations without changing behavior."""
    calls: list = []
    original = dn_module.build_dagua_pipeline

    def tracer(config):
        calls.append(config)
        return original(config)

    return calls, patch.object(dn_module, "build_dagua_pipeline", tracer)


def test_default_algorithm_none_routes_to_dagua_native():
    g = _build_chain_graph()
    calls, ctx = _trace_pipeline_calls()
    with ctx:
        pos = engine_layout(g, dagua.LayoutConfig(steps=10, seed=42))
    assert pos.shape == (10, 2)
    assert len(calls) == 1, "build_dagua_pipeline must be called exactly once on default"


def test_explicit_dagua_native_uses_pipeline():
    g = _build_chain_graph()
    calls, ctx = _trace_pipeline_calls()
    with ctx:
        pos = engine_layout(g, dagua.LayoutConfig(steps=10, seed=42, algorithm="dagua_native"))
    assert pos.shape == (10, 2)
    assert len(calls) == 1


def test_legacy_escape_does_not_use_pipeline():
    g = _build_chain_graph()
    calls, ctx = _trace_pipeline_calls()
    with ctx:
        pos = engine_layout(g, dagua.LayoutConfig(steps=10, seed=42, algorithm="_legacy"))
    assert pos.shape == (10, 2)
    assert len(calls) == 0, "_legacy must bypass the ops pipeline"


def test_trace_argument_forces_legacy_with_warning():
    """Animation path: trace forces legacy because op-level snapshots not yet wired."""

    class _NullTrace:
        def capture_layout_positions(self, *args, **kwargs):
            pass

    g = _build_chain_graph()
    calls, ctx = _trace_pipeline_calls()
    with ctx, warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pos = engine_layout(g, dagua.LayoutConfig(steps=10, seed=42), trace=_NullTrace())
    assert pos.shape == (10, 2)
    assert len(calls) == 0, "trace must NOT route through ops pipeline (yet)"
    assert any(
        issubclass(w.category, DeprecationWarning) and "trace" in str(w.message) for w in captured
    ), "trace fallback must emit a DeprecationWarning so users see the fork"


def test_relax_steps_forces_legacy_with_warning():
    g = _build_chain_graph()
    calls, ctx = _trace_pipeline_calls()
    with ctx, warnings.catch_warnings(record=True) as captured:
        warnings.simplefilter("always")
        pos = engine_layout(g, dagua.LayoutConfig(steps=10, relax_steps=5, seed=42))
    assert pos.shape == (10, 2)
    assert len(calls) == 0, "relax_steps>0 must NOT route through ops pipeline (yet)"
    assert any(
        issubclass(w.category, DeprecationWarning) and "relax_steps" in str(w.message)
        for w in captured
    ), "relax_steps fallback must emit a DeprecationWarning"


def test_other_pipeline_algorithm_still_works():
    """Sanity: non-dagua_native pipeline algorithms (fr, kk, etc.) unchanged."""
    g = _build_chain_graph()
    calls, ctx = _trace_pipeline_calls()
    with ctx:
        pos = engine_layout(g, dagua.LayoutConfig(steps=10, seed=42, algorithm="fr"))
    assert pos.shape == (10, 2)
    assert len(calls) == 0, "fr algorithm must NOT route through dagua_native"
