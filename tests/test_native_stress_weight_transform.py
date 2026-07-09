"""Tests for the r80-S9 native-stress ``weight_transform`` knob.

``NativeStressConfig.weight_transform`` defaults to ``"none"`` everywhere,
preserving today's behavior (edge weights used as-is, i.e. as distances).
``"inverse"`` (``1 / w``) is opt-in only -- the r80-S9 weighted-similarity
portfolio challenger is the sole caller that sets it (see
``dagua/layout/ops/pipelines/native_undirected.py::_weighted_similarity_candidate``).
"""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops.pipelines.native_stress import (
    NativeStressConfig,
    _resolve_native_stress_config,
    layout_native_stress_pipeline,
)


def _ring_edge_index(num_nodes: int) -> torch.Tensor:
    """Return a simple ring edge index.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the ring.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, num_nodes]``.
    """
    edges = [(i, (i + 1) % num_nodes) for i in range(num_nodes)]
    return torch.tensor(list(zip(*edges)), dtype=torch.long)


def test_weight_transform_defaults_to_none() -> None:
    """The default config keeps today's untransformed weight semantics."""
    assert NativeStressConfig().weight_transform == "none"


def test_weight_transform_rejects_unknown_value() -> None:
    """An unsupported transform name raises during config resolution."""
    with pytest.raises(ValueError, match="weight_transform"):
        _resolve_native_stress_config(
            num_nodes=10,
            config=NativeStressConfig(weight_transform="bogus"),
        )


def test_weight_transform_none_preserves_default_positions() -> None:
    """Explicitly passing ``"none"`` must reproduce the no-config call.

    Locks the "no changes to default weight handling anywhere" contract:
    building the pipeline with an explicit ``weight_transform="none"``
    config must be bit-identical to omitting ``config`` entirely.
    """
    edge_index = _ring_edge_index(10)
    node_sizes = torch.full((10, 2), 20.0)
    weights = torch.linspace(0.5, 5.0, steps=10)

    default_pos = layout_native_stress_pipeline(
        edge_index=edge_index,
        num_nodes=10,
        node_sizes=node_sizes,
        edge_weights=weights,
        seed=7,
    )
    explicit_none_pos = layout_native_stress_pipeline(
        edge_index=edge_index,
        num_nodes=10,
        node_sizes=node_sizes,
        edge_weights=weights,
        seed=7,
        config=NativeStressConfig(weight_transform="none", seed=7),
    )

    torch.testing.assert_close(default_pos, explicit_none_pos, rtol=0.0, atol=0.0)


def test_weight_transform_inverse_changes_positions() -> None:
    """``"inverse"`` (1/w) produces a different, finite layout than "none".

    Heavier weights become SHORTER target distances under "inverse" and
    LONGER ones under "none" -- the two must diverge on a graph with
    non-uniform weights, and both must stay finite.
    """
    edge_index = _ring_edge_index(12)
    node_sizes = torch.full((12, 2), 20.0)
    generator = torch.Generator().manual_seed(0)
    weights = torch.rand(12, generator=generator) * 4.0 + 0.5

    none_pos = layout_native_stress_pipeline(
        edge_index=edge_index,
        num_nodes=12,
        node_sizes=node_sizes,
        edge_weights=weights,
        seed=3,
        config=NativeStressConfig(weight_transform="none", seed=3),
    )
    inverse_pos = layout_native_stress_pipeline(
        edge_index=edge_index,
        num_nodes=12,
        node_sizes=node_sizes,
        edge_weights=weights,
        seed=3,
        config=NativeStressConfig(weight_transform="inverse", seed=3),
    )

    assert bool(torch.isfinite(none_pos).all())
    assert bool(torch.isfinite(inverse_pos).all())
    assert not torch.allclose(none_pos, inverse_pos)


def test_weight_transform_inverse_matches_manual_preinversion() -> None:
    """``weight_transform="inverse"`` matches pre-inverting weights by hand.

    ``BuildAdjacencyConfig.weight_transform="inverse"`` computes ``1 / w``
    per edge (``dagua/layout/ops/preprocess.py::_resolve_weights``). This
    checks the adjacency build directly (rather than the full stochastic
    stress solve, which is legitimately sensitive to sub-ULP target-distance
    differences on a symmetric graph and can converge to a different, but
    equally valid, local optimum) so the assertion is deterministic.
    """
    from dagua.layout.ops.preprocess import BuildAdjacency, BuildAdjacencyConfig
    from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState

    edge_index = _ring_edge_index(9)
    weights = torch.linspace(0.25, 3.0, steps=9)
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))

    problem_inverse = LayoutProblem(
        edge_index=edge_index,
        num_nodes=9,
        node_sizes=torch.full((9, 2), 15.0),
        edge_weights=weights,
    )
    problem_manual = LayoutProblem(
        edge_index=edge_index,
        num_nodes=9,
        node_sizes=torch.full((9, 2), 15.0),
        edge_weights=1.0 / weights,
    )
    inverse_config = BuildAdjacencyConfig(
        weighted=True, dedup="min", format="list", directed=False, weight_transform="inverse"
    )
    none_config = BuildAdjacencyConfig(
        weighted=True, dedup="min", format="list", directed=False, weight_transform="none"
    )

    inverse_state = BuildAdjacency(inverse_config).apply(problem_inverse, SolveState(), ctx)
    manual_state = BuildAdjacency(none_config).apply(problem_manual, SolveState(), ctx)

    assert len(inverse_state.adjacency) == len(manual_state.adjacency)
    for inverse_row, manual_row in zip(inverse_state.adjacency, manual_state.adjacency):
        assert len(inverse_row) == len(manual_row)
        for (inverse_target, inverse_weight), (manual_target, manual_weight) in zip(
            inverse_row, manual_row
        ):
            assert inverse_target == manual_target
            assert inverse_weight == pytest.approx(manual_weight, rel=1e-5)
