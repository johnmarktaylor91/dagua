"""Tests for the SmartGD neural layout pipeline."""

from __future__ import annotations

import importlib
import inspect

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.smartgd import (
    SmartGDConfig,
    build_smartgd_model,
    layout_smartgd_pipeline,
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


def _small_config(seed: int = 7) -> SmartGDConfig:
    """Build a small fast SmartGD test config.

    Parameters
    ----------
    seed : int, default=7
        Deterministic seed.

    Returns
    -------
    SmartGDConfig
        Small model configuration.
    """
    return SmartGDConfig(
        num_blocks=1,
        block_depth=1,
        block_width=4,
        block_output_dim=4,
        edge_net_depth=1,
        edge_net_width=8,
        use_reference_checkpoint=False,
        seed=seed,
    )


def test_smartgd_pipeline_is_registered() -> None:
    """The public pipeline registry should resolve ``algorithm='smartgd'``."""
    assert PIPELINE_REGISTRY["smartgd"] == (
        "dagua.layout.ops.pipelines.smartgd",
        "layout_smartgd_pipeline",
    )
    assert get_pipeline_function("smartgd") is layout_smartgd_pipeline


def test_smartgd_forward_is_seed_deterministic() -> None:
    """SmartGD should return identical positions for identical random weights."""
    edge_index = _path_edge_index(6)
    config = _small_config(seed=11)

    first = layout_smartgd_pipeline(edge_index, 6, config=config, seed=11)
    second = layout_smartgd_pipeline(edge_index, 6, config=config, seed=11)

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert first.shape == (6, 2)


def test_smartgd_public_dispatch_returns_positions() -> None:
    """``dagua.layout`` should dispatch to the SmartGD pipeline."""
    graph = DaguaGraph.from_edge_list([(f"n{i}", f"n{i + 1}") for i in range(5)])
    config = LayoutConfig(
        algorithm="smartgd",
        algorithm_params=_small_config(seed=13).__dict__,
        seed=13,
        steps=2,
    )

    pos = dagua.layout(graph, config)

    assert pos.shape == (graph.num_nodes, 2)
    assert torch.isfinite(pos).all()


def test_smartgd_pretrained_checkpoint_loads_strictly_when_available() -> None:
    """The shipped SmartGD checkpoint should load into the ported architecture."""
    checkpoint = "/tmp/smartgd-ref/generator_stress_only.pt"
    try:
        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
    except FileNotFoundError:
        return

    model = build_smartgd_model(SmartGDConfig(use_reference_checkpoint=False))

    model.load_state_dict(state_dict)


def test_smartgd_pipeline_has_no_runtime_reference_import() -> None:
    """The Dagua pipeline must not import the cloned reference implementation."""
    smartgd = importlib.import_module("dagua.layout.ops.pipelines.smartgd")

    source = inspect.getsource(smartgd)

    assert "from smartgd.model" not in source
    assert "import smartgd" not in source
