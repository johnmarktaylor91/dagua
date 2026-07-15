"""Tests for the CoRe-GD neural layout pipeline."""

from __future__ import annotations

import importlib
import inspect

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.coregd import (
    CoreGDConfig,
    _coarsen_edge_index,
    coregd_reference_forward,
    layout_coregd_pipeline,
)


def _path_edge_index(num_nodes: int) -> torch.Tensor:
    """Create a path edge tensor.

    Parameters
    ----------
    num_nodes : int
        Number of path nodes.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, max(N - 1, 0)]``.
    """
    if num_nodes <= 1:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(
        [list(range(num_nodes - 1)), list(range(1, num_nodes))],
        dtype=torch.long,
    )


def _small_config(seed: int = 7) -> CoreGDConfig:
    """Build a small fast CoRe-GD test config.

    Parameters
    ----------
    seed : int, default=7
        Deterministic seed.

    Returns
    -------
    CoreGDConfig
        Small model configuration.
    """
    return CoreGDConfig(
        hidden_dimension=8,
        hidden_state_factor=2.0,
        mlp_depth=0,
        random_in_channels=1,
        laplace_eigvec=2,
        use_beacons=True,
        num_beacons=1,
        encoding_size_per_beacon=4,
        alt_freq=1,
        knn_k=3,
        iterations=2,
        coarsen=False,
        seed=seed,
    )


def test_coregd_pipeline_is_registered() -> None:
    """The public pipeline registry should resolve ``algorithm='coregd'``."""
    assert PIPELINE_REGISTRY["coregd"] == (
        "dagua.layout.ops.pipelines.coregd",
        "layout_coregd_pipeline",
    )
    assert get_pipeline_function("coregd") is layout_coregd_pipeline


def test_coregd_forward_is_seed_deterministic() -> None:
    """CoRe-GD should return identical positions for identical random weights."""
    edge_index = _path_edge_index(6)
    config = _small_config(seed=11)

    first = layout_coregd_pipeline(edge_index, 6, config=config, seed=11)
    second = layout_coregd_pipeline(edge_index, 6, config=config, seed=11)

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert first.shape == (6, 2)


def test_coregd_public_dispatch_returns_positions() -> None:
    """``dagua.layout`` should dispatch to the CoRe-GD pipeline."""
    graph = DaguaGraph.from_edge_list([(f"n{i}", f"n{i + 1}") for i in range(5)])
    config = LayoutConfig(
        algorithm="coregd",
        algorithm_params=_small_config(seed=13).__dict__,
        seed=13,
        steps=2,
    )

    pos = dagua.layout(graph, config)

    assert pos.shape == (graph.num_nodes, 2)
    assert torch.isfinite(pos).all()


def test_coregd_rewiring_changes_message_path() -> None:
    """Disabling KNN overlay should change the same-weight forward result."""
    edge_index = _path_edge_index(7)
    with_rewire = _small_config(seed=17)
    without_rewire = CoreGDConfig(**{**with_rewire.__dict__, "knn_k": 0, "rewiring": "none"})

    pos_rewire = coregd_reference_forward(edge_index, 7, with_rewire)
    pos_no_rewire = coregd_reference_forward(edge_index, 7, without_rewire)

    assert not torch.allclose(pos_rewire, pos_no_rewire)


def test_coregd_coarsening_maps_all_fine_nodes() -> None:
    """The lightweight hierarchy should assign every fine node to a coarse node."""
    edge_index = _path_edge_index(12)

    coarse_edges, assignment, coarse_nodes = _coarsen_edge_index(edge_index, 12, 4)

    assert assignment.shape == (12,)
    assert int(assignment.min().item()) == 0
    assert int(assignment.max().item()) < coarse_nodes
    assert coarse_edges.shape[0] == 2


def test_coregd_pipeline_has_no_runtime_reference_import() -> None:
    """The Dagua pipeline must not import the cloned reference implementation."""
    coregd = importlib.import_module("dagua.layout.ops.pipelines.coregd")

    source = inspect.getsource(coregd)

    assert "neuraldrawer" not in source
    assert "/tmp/coregd-ref" not in source
