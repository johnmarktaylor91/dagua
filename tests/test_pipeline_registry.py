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
    assert captured["config"] is config
    assert captured["config"].edge_equalize_polish is False
    assert captured["config"].direction == "LR"
    assert captured["config"].flex is flex
    assert captured["clusters"] is graph.clusters
    assert captured["cluster_parents"] is graph.cluster_parents
