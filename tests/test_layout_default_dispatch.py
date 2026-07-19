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

import pytest
import torch

import dagua
from dagua.eval.graphs import _make_r8_lr_direction
from dagua.layout.engine import layout as engine_layout
from dagua.layout.ops.pipelines import dagua_native as dn_module
from dagua.layout.ops.pipelines.dagua_native import _apply_public_direction_frame
from dagua.layout.ops.pipelines.native_directed import _score_directed_candidate
from dagua.layout.ops.state import LayoutProblem
from dagua.metrics import quick


def _build_chain_graph(n: int = 10) -> dagua.DaguaGraph:
    """Build a deterministic directed chain graph.

    Parameters
    ----------
    n : int, default=10
        Number of nodes in the chain.

    Returns
    -------
    dagua.DaguaGraph
        Chain graph with edges ``n_i -> n_{i+1}``.
    """
    g = dagua.DaguaGraph()
    for i in range(n):
        g.add_node(f"n{i}")
    for i in range(n - 1):
        g.add_edge(f"n{i}", f"n{i + 1}")
    return g


def _trace_pipeline_calls() -> tuple[list[dagua.LayoutConfig], object]:
    """Wrap build_dagua_pipeline to record invocations without changing behavior."""
    calls: list[dagua.LayoutConfig] = []
    original = dn_module.build_dagua_pipeline

    def tracer(config: dagua.LayoutConfig) -> object:
        """Record one native pipeline build and return the real pipeline.

        Parameters
        ----------
        config : dagua.LayoutConfig
            Configuration passed to the native pipeline builder.

        Returns
        -------
        object
            Pipeline object returned by the original builder.
        """
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


def test_default_dagua_native_honors_graph_lr_direction_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The R8 LR fixture should return coordinates flowing on the LR axis once."""
    monkeypatch.setenv("DAGUA_NATIVE_DISABLE_W5", "1")
    test_graph = _make_r8_lr_direction()
    graph = test_graph.graph
    assert graph.node_sizes is not None
    node_sizes = graph.node_sizes
    config = dagua.LayoutConfig(device="cpu", seed=42, quality=0.25, cluster_aware=False)

    canonical_tb = dn_module.layout_dagua_native_pipeline(
        graph.edge_index,
        graph.num_nodes,
        node_sizes,
        config=dagua.LayoutConfig(
            device="cpu",
            seed=42,
            quality=0.25,
            cluster_aware=False,
            direction="TB",
        ),
        clusters=graph.clusters,
        cluster_parents=graph.cluster_parents,
        cluster_labels=graph.cluster_labels,
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        lr_pos = engine_layout(graph, config)

    wrong_frame_metrics = quick(
        canonical_tb,
        graph.edge_index,
        node_sizes=node_sizes,
        direction="LR",
        declared_hierarchical=True,
    )
    lr_metrics = quick(
        lr_pos,
        graph.edge_index,
        node_sizes=node_sizes,
        direction="LR",
        declared_hierarchical=True,
    )

    assert lr_metrics["directed_flow_score"] == pytest.approx(1.0)
    assert lr_metrics["depth_order_score"] == pytest.approx(1.0)
    assert lr_metrics["directed_flow_score"] > wrong_frame_metrics["directed_flow_score"] + 0.3
    assert lr_metrics["depth_order_score"] > wrong_frame_metrics["depth_order_score"] + 0.7
    assert torch.equal(lr_pos[:, 0], canonical_tb[:, 1])
    assert torch.equal(lr_pos[:, 1], canonical_tb[:, 0])


def test_directed_candidate_scoring_is_tb_lr_transpose_equivalent() -> None:
    """Directed candidate scoring should agree after transposing TB into LR."""
    test_graph = _make_r8_lr_direction()
    graph = test_graph.graph
    edge_index = graph.edge_index.detach().to(device="cpu", dtype=torch.long)
    num_nodes = int(graph.num_nodes)
    node_sizes = torch.full((num_nodes, 2), 60.0, dtype=torch.float32)
    index = torch.arange(num_nodes, dtype=torch.float32)
    canonical_tb = torch.stack((index.remainder(3.0) * 120.0, index * 80.0), dim=1)
    canonical_tb = canonical_tb - canonical_tb.mean(dim=0, keepdim=True)
    lr_pos = _apply_public_direction_frame(canonical_tb, "LR")

    tb_problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        direction="TB",
    )
    lr_problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        direction="LR",
    )

    tb_score = _score_directed_candidate(canonical_tb, tb_problem, cluster_ids=None)
    lr_score = _score_directed_candidate(lr_pos, lr_problem, cluster_ids=None)

    assert lr_score == pytest.approx(tb_score)
