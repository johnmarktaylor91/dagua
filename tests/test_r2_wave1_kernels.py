"""Focused regression tests for round-2 native kernel robustness fixes."""

from __future__ import annotations

import pytest
import torch

from dagua.layout.ops import _dot_mincross
from dagua.layout.ops.pipelines.fmmm import _graphviz_fdp_prism_delaunay_edges
from dagua.layout.ops.quadtree import graphviz_spring_electrical_repulsive_forces
from dagua.layout.ops.sfdp import (
    _SFDP_ALGORITHM_CONFIG,
    _repulsive_forces,
    _tiled_exact_repulsive_forces,
)


def _sample_mincross_inputs() -> tuple[
    list[list[int]],
    dict[int, list[tuple[int, int]]],
    dict[int, list[tuple[int, int]]],
]:
    """Return a layered graph that exercises transpose swaps and ties.

    Returns
    -------
    tuple[list[list[int]], dict[int, list[tuple[int, int]]], dict[int, list[tuple[int, int]]]]
        Mutable ranks plus incoming and outgoing neighbor maps in mincross
        format.
    """
    ranks = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
    incoming = {node: [] for rank in ranks for node in rank}
    outgoing = {node: [] for rank in ranks for node in rank}
    edges = [
        (0, 5, 1),
        (1, 3, 2),
        (2, 4, 1),
        (2, 6, 3),
        (3, 8, 1),
        (4, 7, 2),
        (5, 9, 1),
        (6, 7, 1),
    ]
    for source, target, penalty in edges:
        outgoing[source].append((target, penalty))
        incoming[target].append((source, penalty))
    return ranks, incoming, outgoing


def test_sfdp_fast_repulsion_is_finite_and_deterministic() -> None:
    """Compute finite deterministic native-default repulsion above the old BH threshold."""
    generator = torch.Generator().manual_seed(7)
    positions = torch.randn((80, 2), generator=generator)

    first = _repulsive_forces(
        positions=positions,
        repulsive_scale=0.2,
        repulsive_exponent=-1.0,
        theta=0.6,
        fidelity_mode=False,
    )
    second = _repulsive_forces(
        positions=positions,
        repulsive_scale=0.2,
        repulsive_exponent=-1.0,
        theta=0.6,
        fidelity_mode=False,
    )

    assert torch.isfinite(first).all()
    assert torch.equal(first, second)


def test_sfdp_faithful_mode_uses_graphviz_quadtree_bit_exactly() -> None:
    """Keep faithful repulsion pinned to the existing Graphviz quadtree path."""
    generator = torch.Generator().manual_seed(13)
    positions = torch.randn((64, 2), generator=generator)

    faithful = _repulsive_forces(
        positions=positions,
        repulsive_scale=0.2,
        repulsive_exponent=-1.0,
        theta=0.6,
        fidelity_mode=True,
    )
    expected = graphviz_spring_electrical_repulsive_forces(
        positions=positions,
        repulsive_scale=0.2,
        repulsive_exponent=-1.0,
        theta=0.6,
        max_level=_SFDP_ALGORITHM_CONFIG.max_quadtree_depth,
        quadtree_size=_SFDP_ALGORITHM_CONFIG.barnes_hut_threshold,
    )

    assert torch.equal(faithful, expected)


def test_sfdp_tiled_exact_repulsion_is_chunk_size_independent() -> None:
    """Match exact tiled forces across chunk sizes."""
    generator = torch.Generator().manual_seed(19)
    positions = torch.randn((37, 2), generator=generator)

    chunk_7 = _tiled_exact_repulsive_forces(
        positions=positions,
        repulsive_scale=0.2,
        repulsive_exponent=-1.0,
        chunk_size=7,
    )
    chunk_13 = _tiled_exact_repulsive_forces(
        positions=positions,
        repulsive_scale=0.2,
        repulsive_exponent=-1.0,
        chunk_size=13,
    )

    assert torch.allclose(chunk_7, chunk_13, atol=1.0e-6, rtol=1.0e-6)


def test_dot_transpose_numba_matches_python_fallback() -> None:
    """Require the optional numba transpose to preserve pure-Python order exactly."""
    if not _dot_mincross._NUMBA_AVAILABLE:
        pytest.skip("numba is not installed")
    ranks, incoming, outgoing = _sample_mincross_inputs()
    python_ranks = [list(rank) for rank in ranks]
    numba_ranks = [list(rank) for rank in ranks]

    _dot_mincross._transpose_python(
        ranks=python_ranks,
        incoming=incoming,
        outgoing=outgoing,
        reverse=True,
    )
    _dot_mincross._transpose_numba(
        ranks=numba_ranks,
        incoming=incoming,
        outgoing=outgoing,
        reverse=True,
    )

    assert numba_ranks == python_ranks


def test_fmmm_delaunay_edges_skip_nonfinite_points_without_crashing() -> None:
    """Avoid Qhull crashes when PRISM receives NaN or infinite coordinates."""
    edges = _graphviz_fdp_prism_delaunay_edges(
        x_positions=[0.0, float("nan"), 2.0, 3.0],
        y_positions=[0.0, 1.0, float("inf"), 3.0],
    )

    assert edges == set()
