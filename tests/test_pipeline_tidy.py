"""Regression tests for the non-layered tidy tree pipeline."""

from __future__ import annotations

import subprocess
from typing import Any

import pytest
import torch

from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function


def _tree_edges() -> torch.Tensor:
    """Return a small directed tree edge tensor.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    return torch.tensor(
        [[0, 0, 1, 1, 2], [1, 2, 3, 4, 5]],
        dtype=torch.long,
    )


def _node_sizes() -> torch.Tensor:
    """Return variable node sizes for tidy tests.

    Returns
    -------
    torch.Tensor
        Node sizes with shape ``[N, 2]``.
    """
    return torch.tensor(
        [
            [20.0, 40.0],
            [10.0, 12.0],
            [14.0, 28.0],
            [8.0, 10.0],
            [12.0, 10.0],
            [9.0, 10.0],
        ],
        dtype=torch.float64,
    )


def test_tidy_registry_entry() -> None:
    """Register the public tidy algorithm.

    Returns
    -------
    None
        Assertions verify registry metadata and lazy resolution.
    """
    assert PIPELINE_REGISTRY["tidy"] == (
        "dagua.layout.ops.pipelines.tidy",
        "layout_tidy_pipeline",
    )
    assert get_pipeline_function("tidy").__name__ == "layout_tidy_pipeline"


def test_tidy_pipeline_is_deterministic() -> None:
    """Tidy should produce bitwise stable coordinates.

    Returns
    -------
    None
        Assertion checks deterministic coordinates.
    """
    layout = get_pipeline_function("tidy")
    edge_index = _tree_edges()
    sizes = _node_sizes()

    first = layout(edge_index, 6, sizes, parent_child_margin=7.0, peer_margin=5.0)
    second = layout(edge_index, 6, sizes, parent_child_margin=7.0, peer_margin=5.0)

    assert torch.equal(first, second)
    assert first.shape == (6, 2)
    assert torch.isfinite(first).all()


def test_tidy_uses_variable_parent_heights_for_y() -> None:
    """Non-layered tidy should use each parent height for child y.

    Returns
    -------
    None
        Assertions verify variable-height y placement.
    """
    pos = get_pipeline_function("tidy")(
        _tree_edges(),
        6,
        _node_sizes(),
        parent_child_margin=7.0,
        peer_margin=5.0,
        dtype=torch.float64,
    )

    assert pos[1, 1].item() == 47.0
    assert pos[2, 1].item() == 47.0
    assert pos[3, 1].item() == 66.0
    assert pos[4, 1].item() == 66.0
    assert pos[5, 1].item() == 82.0


def test_tidy_sibling_subtrees_do_not_overlap() -> None:
    """Sibling subtrees should be separated by at least the peer margin.

    Returns
    -------
    None
        Assertion checks sibling x extents at the child layer.
    """
    sizes = _node_sizes()
    peer_margin = 5.0
    pos = get_pipeline_function("tidy")(
        _tree_edges(),
        6,
        sizes,
        parent_child_margin=7.0,
        peer_margin=peer_margin,
        dtype=torch.float64,
    )

    left_right = pos[1, 0].item() + sizes[1, 0].item() / 2.0
    right_left = pos[2, 0].item() - sizes[2, 0].item() / 2.0

    assert right_left - left_right >= peer_margin


def test_tidy_forest_roots_are_tiled_apart() -> None:
    """Multiple roots should be placed as separate horizontal components.

    Returns
    -------
    None
        Assertion checks that forest roots are not stacked at the same x.
    """
    edge_index = torch.tensor([[0, 2], [1, 3]], dtype=torch.long)
    sizes = torch.full((4, 2), 10.0, dtype=torch.float64)

    pos = get_pipeline_function("tidy")(
        edge_index,
        4,
        sizes,
        peer_margin=6.0,
        dtype=torch.float64,
    )

    assert pos[2, 0].item() - pos[0, 0].item() >= 16.0


def test_tidy_pipeline_does_not_delegate_to_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Tidy must not call the Rust reference at runtime.

    Parameters
    ----------
    monkeypatch : pytest.MonkeyPatch
        Pytest monkeypatch fixture used to poison subprocess entrypoints.

    Returns
    -------
    None
        Successful layout proves no subprocess delegation occurred.
    """

    def fail_popen(*args: Any, **kwargs: Any) -> None:
        """Fail if runtime delegation is attempted.

        Parameters
        ----------
        *args : Any
            Positional arguments.
        **kwargs : Any
            Keyword arguments.

        Returns
        -------
        None
            Always raises.
        """
        raise AssertionError("tidy pipeline delegated to a subprocess.")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    pos = get_pipeline_function("tidy")(_tree_edges(), 6, _node_sizes())

    assert pos.shape == (6, 2)
