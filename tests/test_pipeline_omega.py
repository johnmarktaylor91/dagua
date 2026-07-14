"""Regression tests for the Omega/RDMDS pipeline."""

from __future__ import annotations

import subprocess
from typing import Any

import numpy as np
import pytest
import torch

from dagua.layout.ops.pipelines import PIPELINE_REGISTRY, get_pipeline_function
from dagua.layout.ops.pipelines.omega import OmegaConfig, _build_pairs, _rdmds_embedding


def _cycle_edges() -> torch.Tensor:
    """Return a small cycle graph edge tensor.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    return torch.tensor(
        [[0, 1, 2, 3, 0], [1, 2, 3, 0, 2]],
        dtype=torch.long,
    )


def test_omega_registry_entry() -> None:
    """Register the public Omega algorithm.

    Returns
    -------
    None
        Assertions verify registry metadata and lazy resolution.
    """
    assert PIPELINE_REGISTRY["omega"] == (
        "dagua.layout.ops.pipelines.omega",
        "layout_omega_pipeline",
    )
    assert get_pipeline_function("omega").__name__ == "layout_omega_pipeline"


def test_omega_pipeline_is_seed_deterministic() -> None:
    """Omega should produce identical positions for the same seed.

    Returns
    -------
    None
        Assertion checks bitwise deterministic coordinates.
    """
    layout = get_pipeline_function("omega")
    edge_index = _cycle_edges()

    first = layout(edge_index, 4, seed=7, k=4, sgd_iterations=5, dtype=torch.float64)
    second = layout(edge_index, 4, seed=7, k=4, sgd_iterations=5, dtype=torch.float64)

    assert torch.equal(first, second)
    assert first.shape == (4, 2)
    assert torch.isfinite(first).all()


def test_omega_seed_changes_sampled_sparse_sgd_path() -> None:
    """Different seeds should alter random pair sampling.

    Returns
    -------
    None
        Assertion checks that the seed reaches stochastic Omega stages.
    """
    import numpy as np

    edges = [(0, 1), (1, 2), (2, 3), (3, 4)]
    config = OmegaConfig(k=1)
    embedding = _rdmds_embedding(5, edges, config)

    first = _build_pairs(edges, embedding, config, np.random.default_rng(1))
    second = _build_pairs(edges, embedding, config, np.random.default_rng(2))

    assert [(p.i, p.j) for p in first] != [(p.i, p.j) for p in second]


def test_omega_random_pair_count_is_bounded_and_deterministic() -> None:
    """Random pair sampling should follow the reference duplicate-skip shape.

    Returns
    -------
    None
        Assertions validate deterministic pair construction.
    """
    config = OmegaConfig(k=3, seed=11)
    edges = [(0, 1), (1, 2)]
    embedding = _rdmds_embedding(4, edges, config)
    pairs_b = _build_pairs(edges, embedding, config, np.random.default_rng(11))
    pairs_c = _build_pairs(edges, embedding, config, np.random.default_rng(11))

    assert [(p.i, p.j, p.distance) for p in pairs_b] == [(p.i, p.j, p.distance) for p in pairs_c]
    assert len(pairs_b) <= len(edges) + 4 * config.k


def test_omega_pipeline_does_not_delegate_to_reference(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Omega must not call the Rust reference at runtime.

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
        raise AssertionError("Omega pipeline delegated to a subprocess.")

    monkeypatch.setattr(subprocess, "Popen", fail_popen)
    pos = get_pipeline_function("omega")(_cycle_edges(), 4, seed=3, k=1, sgd_iterations=2)

    assert pos.shape == (4, 2)
