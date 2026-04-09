"""Tests for FIX-S deterministic seeding in stochastic metric functions."""

from __future__ import annotations

import json
import subprocess
import sys
from typing import Tuple

import pytest
import torch

from dagua.metrics import (
    count_crossings,
    count_overlaps_detailed,
    quick,
    sampled_crossing_rate,
)


@pytest.fixture
def dense_overlap_graph() -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a graph that triggers overlap subsampling in crowded hash cells.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Position and node-size tensors for a dense graph with ``N > 2000`` so
        ``count_overlaps_detailed`` takes its spatial-hash path.
    """
    torch.manual_seed(0)
    n_nodes = 2501
    pos = torch.randn(n_nodes, 2) * 0.5
    node_sizes = torch.ones(n_nodes, 2) * 0.2
    return pos, node_sizes


@pytest.fixture
def many_edge_graph() -> Tuple[torch.Tensor, torch.Tensor]:
    """Build a graph that forces crossing-rate sampling.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor]
        Position and edge tensors for a graph with ``E > 500``.
    """
    torch.manual_seed(0)
    n_nodes = 100
    pos = torch.randn(n_nodes, 2)
    edges = torch.randint(0, n_nodes, (2, 2000))
    return pos, edges


def test_overlaps_seeded_reproducible(
    dense_overlap_graph: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify identical seeds reproduce the same overlap count.

    Returns
    -------
    None
        This test asserts exact dictionary equality.
    """
    pos, node_sizes = dense_overlap_graph
    first = count_overlaps_detailed(pos, node_sizes, seed=42)
    second = count_overlaps_detailed(pos, node_sizes, seed=42)
    assert first == second


def test_overlaps_seeded_distinct_seeds_differ(
    dense_overlap_graph: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify different seeds perturb the capped overlap sample.

    Returns
    -------
    None
        This test asserts the overlap dictionaries differ.
    """
    pos, node_sizes = dense_overlap_graph
    first = count_overlaps_detailed(pos, node_sizes, seed=1)
    second = count_overlaps_detailed(pos, node_sizes, seed=2)
    assert first != second


def test_overlaps_unseeded_is_stochastic(
    dense_overlap_graph: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify ``seed=None`` preserves stochastic behavior.

    Returns
    -------
    None
        This test asserts repeated calls are not all identical.
    """
    pos, node_sizes = dense_overlap_graph
    results = [count_overlaps_detailed(pos, node_sizes, seed=None) for _ in range(5)]
    unique_counts = {result["overlap_count"] for result in results}
    assert len(unique_counts) > 1


def test_sampled_crossing_seeded_reproducible(
    many_edge_graph: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify identical seeds reproduce the same sampled crossing estimate.

    Returns
    -------
    None
        This test asserts exact dictionary equality.
    """
    pos, edge_index = many_edge_graph
    first = sampled_crossing_rate(pos, edge_index, n_samples=500, seed=42)
    second = sampled_crossing_rate(pos, edge_index, n_samples=500, seed=42)
    assert first == second


def test_sampled_crossing_distinct_seeds_differ(
    many_edge_graph: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify different seeds change the sampled crossing estimate.

    Returns
    -------
    None
        This test asserts the sampled dictionaries differ.
    """
    pos, edge_index = many_edge_graph
    first = sampled_crossing_rate(pos, edge_index, n_samples=500, seed=1)
    second = sampled_crossing_rate(pos, edge_index, n_samples=500, seed=2)
    assert first != second


def test_sampled_crossing_unseeded_is_stochastic(
    many_edge_graph: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify unseeded sampled crossing remains stochastic.

    Returns
    -------
    None
        This test asserts repeated estimates are not all identical.
    """
    pos, edge_index = many_edge_graph
    results = [sampled_crossing_rate(pos, edge_index, n_samples=500, seed=None) for _ in range(5)]
    unique_rates = {result["crossing_rate"] for result in results}
    assert len(unique_rates) > 1


def test_count_crossings_seeded_reproducible_large_graph(
    many_edge_graph: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify seeded large-graph crossing counts are reproducible.

    Returns
    -------
    None
        This test asserts exact equality through the sampled branch.
    """
    pos, edge_index = many_edge_graph
    first = count_crossings(pos, edge_index, seed=42)
    second = count_crossings(pos, edge_index, seed=42)
    assert first == second


def test_count_crossings_distinct_seeds_differ(
    many_edge_graph: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify different seeds change the sampled crossing count.

    Returns
    -------
    None
        This test asserts the sampled totals differ.
    """
    pos, edge_index = many_edge_graph
    first = count_crossings(pos, edge_index, seed=1)
    second = count_crossings(pos, edge_index, seed=2)
    assert first != second


def test_count_crossings_unseeded_is_stochastic(
    many_edge_graph: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify unseeded large-graph crossing counts remain stochastic.

    Returns
    -------
    None
        This test asserts repeated sampled totals are not all identical.
    """
    pos, edge_index = many_edge_graph
    results = [count_crossings(pos, edge_index, seed=None) for _ in range(5)]
    assert len(set(results)) > 1


def test_quick_seed_reproducibility(
    dense_overlap_graph: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify ``quick`` forwards the seed to overlap counting.

    Returns
    -------
    None
        This test asserts the overlap field is reproducible.
    """
    pos, node_sizes = dense_overlap_graph
    n_nodes = pos.shape[0]
    edge_index = torch.stack([torch.arange(n_nodes - 1), torch.arange(1, n_nodes)])
    first = quick(pos, edge_index, node_sizes=node_sizes, seed=42)
    second = quick(pos, edge_index, node_sizes=node_sizes, seed=42)
    assert first["overlap_count"] == second["overlap_count"]


def test_quick_unseeded_still_works(
    dense_overlap_graph: Tuple[torch.Tensor, torch.Tensor],
) -> None:
    """Verify the new ``quick`` keyword remains backward compatible.

    Returns
    -------
    None
        This test asserts ``quick`` still returns overlap metrics.
    """
    pos, node_sizes = dense_overlap_graph
    n_nodes = pos.shape[0]
    edge_index = torch.stack([torch.arange(n_nodes - 1), torch.arange(1, n_nodes)])
    result = quick(pos, edge_index, node_sizes=node_sizes)
    assert "overlap_count" in result


def _run_overlap_subprocess(seed: int) -> dict[str, int]:
    """Evaluate the overlap metric inside a fresh Python subprocess.

    Parameters
    ----------
    seed : int
        Seed forwarded to ``count_overlaps_detailed``.

    Returns
    -------
    dict[str, int]
        Overlap statistics parsed from subprocess JSON output.
    """
    code = f"""
import json
import torch
from dagua.metrics import count_overlaps_detailed

torch.manual_seed(0)
n_nodes = 2501
pos = torch.randn(n_nodes, 2) * 0.5
node_sizes = torch.ones(n_nodes, 2) * 0.2
print(json.dumps(count_overlaps_detailed(pos, node_sizes, seed={seed}), sort_keys=True))
"""
    completed = subprocess.run(
        [sys.executable, "-c", code],
        check=True,
        capture_output=True,
        text=True,
    )
    return json.loads(completed.stdout.strip())


def test_overlaps_seeded_cross_process() -> None:
    """Verify the same seed reproduces across separate Python processes.

    Returns
    -------
    None
        This test asserts cross-process seeded outputs are identical.
    """
    result_a = _run_overlap_subprocess(42)
    result_b = _run_overlap_subprocess(42)
    assert result_a == result_b
