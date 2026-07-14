"""Tests for the DeepGD neural layout pipeline."""

from __future__ import annotations

import importlib
import inspect
from pathlib import Path

import torch

import dagua
from dagua.config import LayoutConfig
from dagua.graph import DaguaGraph
from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.deepgd import (
    DeepGDConfig,
    build_deepgd_model,
    layout_deepgd_pipeline,
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


def _small_config(seed: int = 7) -> DeepGDConfig:
    """Build a small fast DeepGD test config.

    Parameters
    ----------
    seed : int, default=7
        Deterministic seed.

    Returns
    -------
    DeepGDConfig
        Small model configuration.
    """
    return DeepGDConfig(
        num_blocks=1,
        block_depth=1,
        block_width=4,
        block_output_dim=4,
        edge_net_depth=1,
        edge_net_width=8,
        use_reference_checkpoint=False,
        seed=seed,
    )


def test_deepgd_pipeline_is_registered() -> None:
    """The public pipeline registry should resolve ``algorithm='deepgd'``."""
    assert PIPELINE_REGISTRY["deepgd"] == (
        "dagua.layout.ops.pipelines.deepgd",
        "layout_deepgd_pipeline",
    )
    assert get_pipeline_function("deepgd") is layout_deepgd_pipeline


def test_deepgd_forward_is_seed_deterministic() -> None:
    """DeepGD should return identical positions for identical random weights."""
    edge_index = _path_edge_index(6)
    config = _small_config(seed=11)

    first = layout_deepgd_pipeline(edge_index, 6, config=config, seed=11)
    second = layout_deepgd_pipeline(edge_index, 6, config=config, seed=11)

    assert torch.equal(first, second)
    assert torch.isfinite(first).all()
    assert first.shape == (6, 2)


def test_deepgd_public_dispatch_returns_positions() -> None:
    """``dagua.layout`` should dispatch to the DeepGD pipeline."""
    graph = DaguaGraph.from_edge_list([(f"n{i}", f"n{i + 1}") for i in range(5)])
    config = LayoutConfig(
        algorithm="deepgd",
        algorithm_params=_small_config(seed=13).__dict__,
        seed=13,
        steps=2,
    )

    pos = dagua.layout(graph, config)

    assert pos.shape == (graph.num_nodes, 2)
    assert torch.isfinite(pos).all()


def test_deepgd_pretrained_checkpoint_loads_strictly_when_available() -> None:
    """The shipped DeepGD checkpoint should load into the ported architecture."""
    checkpoint = Path.home() / "tools" / "dagua-refs" / "deepgd" / "model_stress_only.pt"
    try:
        state_dict = torch.load(checkpoint, map_location=torch.device("cpu"))
    except FileNotFoundError:
        return

    model = build_deepgd_model(DeepGDConfig(use_reference_checkpoint=False))

    model.load_state_dict(state_dict)


def test_deepgd_pipeline_has_no_runtime_reference_import() -> None:
    """The Dagua pipeline must not import the cloned reference implementation."""
    deepgd = importlib.import_module("dagua.layout.ops.pipelines.deepgd")

    source = inspect.getsource(deepgd)

    assert "from deepgd.model" not in source
    assert "import deepgd" not in source
