"""Regression tests for registered layout pipeline dispatch."""

from __future__ import annotations

from typing import Any, Optional

import pytest
import torch

import dagua
from dagua.config import LayoutConfig
from dagua.flex import Flex, LayoutFlex
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY


def _small_seeded_graph() -> DaguaGraph:
    """Build the small graph used for registry dispatch smoke coverage.

    Returns
    -------
    DaguaGraph
        Ten-node DAG with enough branching to exercise layered and force
        pipelines without making the registry test slow.
    """
    edges = [
        ("n0", "n1"),
        ("n0", "n2"),
        ("n1", "n3"),
        ("n2", "n3"),
        ("n2", "n4"),
        ("n3", "n5"),
        ("n4", "n5"),
        ("n5", "n6"),
        ("n5", "n7"),
        ("n6", "n8"),
        ("n7", "n9"),
    ]
    return DaguaGraph.from_edge_list(edges)


@pytest.mark.parametrize("algorithm", sorted(PIPELINE_REGISTRY))
def test_registered_pipeline_dispatch_produces_finite_positions(algorithm: str) -> None:
    """Every registered algorithm should dispatch to a finite position tensor.

    Parameters
    ----------
    algorithm : str
        Registered algorithm name from ``PIPELINE_REGISTRY``.

    Returns
    -------
    None
        The assertion verifies the public ``LayoutConfig.algorithm`` dispatch
        path returns finite positions with shape ``[N, 2]``.
    """
    graph = _small_seeded_graph()

    pos = dagua.layout(graph, LayoutConfig(algorithm=algorithm, seed=42, steps=1))

    assert isinstance(pos, torch.Tensor)
    assert pos.shape == (graph.num_nodes, 2)
    assert torch.isfinite(pos).all()


def test_native_stress_pipeline_is_seed_deterministic() -> None:
    """``native_stress`` should produce identical coordinates for the same seed."""
    graph = DaguaGraph.from_edge_list(
        [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n2", "n3"),
            ("n3", "n0"),
            ("n0", "n2"),
        ]
    )
    config = LayoutConfig(algorithm="native_stress", seed=123, steps=4)

    first = dagua.layout(graph, config)
    second = dagua.layout(graph, config)

    assert torch.equal(first, second)


def test_dagua_native_force_pipeline_stress_dispatches() -> None:
    """``force_pipeline='stress'`` should expose native stress without auto-routing it."""
    graph = DaguaGraph.from_edge_list(
        [
            ("n0", "n1"),
            ("n1", "n2"),
            ("n2", "n3"),
            ("n3", "n0"),
        ]
    )
    pos = dagua.layout(
        graph,
        LayoutConfig(
            algorithm="dagua_native",
            force_pipeline="stress",
            seed=42,
            steps=3,
        ),
    )

    assert pos.shape == (graph.num_nodes, 2)
    assert torch.isfinite(pos).all()


def test_explicit_dagua_native_forwards_user_config_to_pipeline(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Explicit ``algorithm='dagua_native'`` should preserve user config kwargs.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to replace the native pipeline.

    Returns
    -------
    None
        The fake pipeline asserts that config, flex, direction, and cluster
        metadata reached the dispatch layer.
    """
    from dagua.layout.ops.pipelines import dagua_native

    captured: dict[str, Any] = {}

    def fake_native_pipeline(
        edge_index: torch.Tensor,
        num_nodes: int,
        node_sizes: torch.Tensor,
        config: Optional[LayoutConfig] = None,
        clusters: Optional[dict[str, Any]] = None,
        cluster_parents: Optional[dict[str, Optional[str]]] = None,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Capture native dispatch kwargs and return a valid position tensor.

        Parameters
        ----------
        edge_index : torch.Tensor
            Graph connectivity with shape ``[2, E]``.
        num_nodes : int
            Number of graph nodes.
        node_sizes : torch.Tensor
            Node sizes with shape ``[N, 2]``.
        config : LayoutConfig, optional
            Resolved layout configuration.
        clusters : dict[str, Any], optional
            Cluster metadata from the graph.
        cluster_parents : dict[str, str], optional
            Cluster parent metadata from the graph.
        **kwargs : Any
            Other accepted dispatch kwargs.

        Returns
        -------
        torch.Tensor
            Zero position tensor with shape ``[N, 2]``.
        """
        captured.update(
            {
                "edge_index": edge_index,
                "node_sizes": node_sizes,
                "config": config,
                "clusters": clusters,
                "cluster_parents": cluster_parents,
                "kwargs": kwargs,
            }
        )
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    monkeypatch.setattr(dagua_native, "layout_dagua_native_pipeline", fake_native_pipeline)

    graph = DaguaGraph.from_edge_list([("a", "b"), ("b", "c")])
    graph.add_cluster("outer", ["a", "b"])
    graph.add_cluster("inner", ["a"], parent="outer")
    flex = LayoutFlex(node_sep=Flex.firm(33.0))
    config = LayoutConfig(
        algorithm="dagua_native",
        edge_equalize_polish=False,
        direction="LR",
        flex=flex,
        seed=42,
    )

    pos = dagua.layout(graph, config)

    assert pos.shape == (graph.num_nodes, 2)
    # Dispatch forwards an equal COPY of the caller's config (copy-before-
    # mutate, WP-23): pin the forwarding contract, not object identity, and
    # pin that the caller's object is never mutated.
    assert captured["config"] == config
    assert captured["config"].edge_equalize_polish is False
    assert captured["config"].direction == "LR"
    assert captured["config"].flex == flex
    assert config.direction == "LR"
    assert config.flex is flex
    assert captured["clusters"] is graph.clusters
    assert captured["cluster_parents"] is graph.cluster_parents


def test_registry_counts_match_certified_inventory() -> None:
    """Pin the live registry counts against the certified inventory.

    Baseline was 385 ops (bare import) / 114 pipelines at 181d471c;
    WP-22a removed 5 zero-reference ops (of the 6 TRIAGE-approved
    candidates -- ``fmmm_uncoarsen_loop`` turned out to be live via the
    ``_UncoarsenLoop`` alias in ``pipelines/fmmm.py`` and was kept), so
    the bare-import count is 380. Two more ops
    (``directed_portfolio_route``, ``undirected_portfolio_route``)
    register lazily when the native support modules first import (any
    ``dagua_native`` run does this), so the test imports them explicitly
    and pins the order-independent all-in count of 382. Any other drift
    (a module silently dropping out of auto-discovery, an incidental
    transitive-import registration disappearing) must fail loudly here.
    """
    import dagua.layout.ops.pipelines.native_directed  # noqa: F401
    import dagua.layout.ops.pipelines.native_undirected  # noqa: F401
    from dagua.layout.ops import OP_REGISTRY

    assert len(OP_REGISTRY) == 382
    assert len(PIPELINE_REGISTRY) == 114


def test_trimmed_zero_reference_ops_stay_gone() -> None:
    """The 5 trimmed zero-reference ops must not silently reappear."""
    from dagua.layout.ops import OP_REGISTRY

    trimmed = (
        "crossing_swap_polish",
        "family_conditional_init",
        "gem_convergence_check",
        "graphopt_apply_displacement",
        "maxent_majorization_step",
    )
    for name in trimmed:
        assert name not in OP_REGISTRY


def test_register_op_rejects_unnamed_op_classes() -> None:
    """``@register_op`` must fail loudly on classes without a proper name.

    A silent skip would leave the op invisibly absent from the registry --
    the only silent-drop vector in op discovery.
    """
    from dagua.layout.ops.base import Op
    from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
    from dagua.layout.ops.taxonomy import register_op

    with pytest.raises(ValueError, match="name"):

        @register_op
        class _NamelessOp(Op):
            def apply(
                self,
                problem: LayoutProblem,
                state: SolveState,
                ctx: RuntimeContext,
            ) -> SolveState:
                del problem, ctx
                return state
