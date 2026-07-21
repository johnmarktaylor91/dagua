"""Regression tests for the scoring-only V3 ruler core."""

from __future__ import annotations

import math
from typing import Dict, Tuple

import pytest
import torch

from dagua.eval.ruler_v3 import (
    CORE_TIERS,
    PURE_GEOMETRY_FACETS,
    WHITESPACE_RATIO_HI,
    crossing_weight_multiplier,
    renormalized_score,
    score_core_v3,
)


def _probe_layout() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a compact graph that exercises all V3 CORE facets.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [4.0, 4.0],
            [0.0, 4.0],
            [8.0, 4.0],
            [12.0, 4.0],
        ],
        dtype=torch.float64,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 3, 0, 1, 3, 4],
            [1, 2, 3, 0, 2, 3, 4, 5],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.full((6, 2), 0.5, dtype=torch.float64)
    return pos, edge_index, node_sizes


def _chain_layout(
    node_count: int,
    gap: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a straight chain probe.

    Parameters
    ----------
    node_count : int
        Number of chain nodes.
    gap : float, optional
        Horizontal center spacing.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    pos = torch.stack(
        (
            torch.arange(node_count, dtype=torch.float64) * gap,
            torch.zeros(node_count, dtype=torch.float64),
        ),
        dim=1,
    )
    edge_index = torch.tensor(
        [
            list(range(node_count - 1)),
            list(range(1, node_count)),
        ],
        dtype=torch.long,
    )
    node_sizes = torch.ones((node_count, 2), dtype=torch.float64)
    return pos, edge_index, node_sizes


def _grid_layout(side: int, gap: float = 3.0) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a square grid probe.

    Parameters
    ----------
    side : int
        Number of nodes per grid side.
    gap : float, optional
        Center spacing.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    coords = [(float(x) * gap, float(y) * gap) for y in range(side) for x in range(side)]
    edges = []
    for y in range(side):
        for x in range(side):
            node = y * side + x
            if x + 1 < side:
                edges.append((node, node + 1))
                edges.append((node + 1, node))
            if y + 1 < side:
                edges.append((node, node + side))
                edges.append((node + side, node))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return (
        torch.tensor(coords, dtype=torch.float64),
        edge_index,
        torch.ones((side * side, 2), dtype=torch.float64),
    )


def _binary_tree_layout(
    depth: int,
    gap: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a tidy layered binary tree probe.

    Parameters
    ----------
    depth : int
        Maximum tree depth below the root.
    gap : float, optional
        Horizontal and vertical layer spacing.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    coords = []
    edges = []
    for layer in range(depth + 1):
        count = 2**layer
        start = count - 1
        offset = (count - 1) * gap / 2.0
        for index in range(count):
            coords.append((index * gap - offset, layer * gap))
            node = start + index
            left = 2 * node + 1
            right = 2 * node + 2
            if layer < depth:
                edges.extend([(node, left), (node, right)])
    node_count = len(coords)
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return (
        torch.tensor(coords, dtype=torch.float64),
        edge_index,
        torch.ones((node_count, 2), dtype=torch.float64),
    )


def _radial_star_layout(leaves: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a minimum-circumference radial star probe.

    Parameters
    ----------
    leaves : int
        Number of star leaves around the hub.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    radius = leaves / (2.0 * math.pi)
    coords = [(0.0, 0.0)]
    for index in range(leaves):
        theta = 2.0 * math.pi * index / leaves
        coords.append((radius * math.cos(theta), radius * math.sin(theta)))
    edge_index = torch.tensor(
        [[0 for _ in range(leaves)], list(range(1, leaves + 1))],
        dtype=torch.long,
    )
    return (
        torch.tensor(coords, dtype=torch.float64),
        edge_index,
        torch.ones((leaves + 1, 2), dtype=torch.float64),
    )


def _layered_dag_layout(
    layers: int,
    width: int,
    gap: float = 3.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a layered rank DAG probe.

    Parameters
    ----------
    layers : int
        Number of ranks.
    width : int
        Nodes per rank.
    gap : float, optional
        Rank and within-rank spacing.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    coords = []
    edges = []
    for layer in range(layers):
        offset = (width - 1) * gap / 2.0
        for column in range(width):
            coords.append((column * gap - offset, layer * gap))
            if layer + 1 < layers:
                edges.append((layer * width + column, (layer + 1) * width + column))
                if column + 1 < width:
                    edges.append((layer * width + column, (layer + 1) * width + column + 1))
    edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
    return (
        torch.tensor(coords, dtype=torch.float64),
        edge_index,
        torch.ones((layers * width, 2), dtype=torch.float64),
    )


def _disconnected_collinear_layout() -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Create a disconnected C1 probe with one collinear component.

    Returns
    -------
    Tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Positions ``[N, 2]``, edge index ``[2, E]``, and node sizes ``[N, 2]``.
    """
    pos = torch.tensor(
        [
            [0.0, 0.0],
            [4.0, 0.0],
            [0.0, 4.0],
            [8.0, 0.0],
            [11.0, 0.0],
            [14.0, 0.0],
            [17.0, 0.0],
            [20.0, 0.0],
            [23.0, 0.0],
        ],
        dtype=torch.float64,
    )
    edge_index = torch.tensor(
        [
            [0, 1, 2, 0, 3, 4, 5, 6, 7],
            [1, 2, 0, 2, 4, 5, 6, 7, 8],
        ],
        dtype=torch.long,
    )
    node_sizes = torch.ones((9, 2), dtype=torch.float64)
    return pos, edge_index, node_sizes


def test_unit_invariance_for_all_core_facets() -> None:
    """Joint scaling of positions and node sizes must preserve every facet."""
    pos, edge_index, node_sizes = _probe_layout()
    baseline = score_core_v3(pos, edge_index, node_sizes)

    for alpha in (0.02, 0.1, 1.0, 10.0, 50.0):
        scaled = score_core_v3(alpha * pos, edge_index, alpha * node_sizes)
        for code, facet in baseline.facets.items():
            assert scaled.facets[code].score == pytest.approx(facet.score, abs=1e-9)
        assert scaled.scores["tiered"] == pytest.approx(baseline.scores["tiered"], abs=1e-9)


def test_position_only_scale_invariance_for_pure_geometry_facets() -> None:
    """Pure-geometry facets must ignore position-only scale changes."""
    pos, edge_index, node_sizes = _probe_layout()
    baseline = score_core_v3(pos, edge_index, node_sizes)

    for alpha in (0.02, 0.1, 1.0, 10.0, 50.0):
        scaled = score_core_v3(alpha * pos, edge_index, node_sizes)
        for code in PURE_GEOMETRY_FACETS:
            assert scaled.facets[code].score == pytest.approx(
                baseline.facets[code].score,
                abs=1e-9,
            )


def test_c1_disconnected_collinear_position_scale_invariance() -> None:
    """C1 must not drift when a disconnected component has zero hull area."""
    pos, edge_index, node_sizes = _disconnected_collinear_layout()
    scores = []

    for alpha in (0.02, 0.1, 1.0, 10.0, 50.0):
        result = score_core_v3(alpha * pos, edge_index, node_sizes)
        assert result.facets["C1"].score is not None
        scores.append(result.facets["C1"].score)

    assert max(scores) - min(scores) < 1e-6


def test_c5_whitespace_band_flags_crater_and_stays_flat_on_legitimate_chain() -> None:
    """The C5 band should tolerate normal chain extent but penalize crater sprawl."""
    pos, edge_index, node_sizes = _chain_layout(10)

    legitimate = score_core_v3(pos, edge_index, node_sizes)
    crater = score_core_v3(1000.0 * pos, edge_index, node_sizes)

    assert legitimate.facets["C5"].score == pytest.approx(1.0)
    assert crater.facets["C5"].score is not None
    assert crater.facets["C5"].score < 0.01
    assert crater.facets["C5"].metadata["whitespace_ratio"] > WHITESPACE_RATIO_HI
    assert "SPRAWL" in crater.flags


def test_c5_legitimate_sprawl_probe_set_stays_on_plateau() -> None:
    """Structure floors should not false-positive on legitimate broad layouts."""
    probes = {
        "deep_tree": _binary_tree_layout(depth=8),
        "long_chain": _chain_layout(200),
        "radial_star": _radial_star_layout(leaves=1000),
        "layered_ranks": _layered_dag_layout(layers=12, width=30),
    }

    table: Dict[str, float] = {}
    for name, (pos, edge_index, node_sizes) in probes.items():
        result = score_core_v3(pos, edge_index, node_sizes)
        score = result.facets["C5"].score
        assert score == pytest.approx(1.0), name
        assert "SPRAWL" not in result.flags, name
        table[name] = float(result.facets["C5"].metadata["whitespace_ratio"])

    assert set(table) == {"deep_tree", "long_chain", "radial_star", "layered_ranks"}


def test_c5_crater_threshold_locks_still_fire() -> None:
    """The 10x and 50x area craters should remain hard SPRAWL failures."""
    pos, edge_index, node_sizes = _grid_layout(10)

    for alpha in (10.0, 50.0):
        crater = score_core_v3(alpha * pos, edge_index, node_sizes)
        score = crater.facets["C5"].score
        assert score is not None
        assert score < 0.01
        assert "SPRAWL" in crater.flags


def test_edgeless_margins_are_inapplicable_not_imputed() -> None:
    """C1, C3, and C10 should publish None on edgeless graphs."""
    pos = torch.tensor([[0.0, 0.0], [3.0, 0.0], [0.0, 3.0]], dtype=torch.float64)
    edge_index = torch.empty((2, 0), dtype=torch.long)
    node_sizes = torch.ones((3, 2), dtype=torch.float64)

    result = score_core_v3(pos, edge_index, node_sizes)

    for code in ("C1", "C3", "C10"):
        assert result.facets[code].score is None
        assert result.facets[code].applicable is False


def test_c4_occlusion_includes_label_extents() -> None:
    """C4 should detect overlap between declared label-inclusive extents."""
    pos = torch.tensor([[0.0, 0.0], [4.0, 0.0]], dtype=torch.float64)
    edge_index = torch.tensor([[0], [1]], dtype=torch.long)
    node_sizes = torch.ones((2, 2), dtype=torch.float64)
    label_sizes = torch.tensor([[4.0, 1.0], [4.0, 1.0]], dtype=torch.float64)
    label_offsets = torch.tensor([[1.5, 0.0], [-1.5, 0.0]], dtype=torch.float64)

    node_only = score_core_v3(pos, edge_index, node_sizes)
    label_inclusive = score_core_v3(
        pos,
        edge_index,
        node_sizes,
        label_sizes=label_sizes,
        label_offsets=label_offsets,
    )

    assert node_only.facets["C4"].score == pytest.approx(1.0)
    assert label_inclusive.facets["C4"].score is not None
    assert label_inclusive.facets["C4"].score < 1.0
    assert label_inclusive.facets["C4"].metadata["label_inclusive"] is True


def test_applicability_renormalization_excludes_missing_facets() -> None:
    """Missing facets should leave the denominator instead of receiving neutral fill."""
    score = renormalized_score(
        {"present": 0.5, "missing": None},
        {"present": 2.0, "missing": 100.0},
    )

    assert score == pytest.approx(50.0)


def test_triple_view_composite_and_tier_weights_are_published() -> None:
    """V3 should publish equal, tiered, and Tier-1-only scoring views."""
    pos, edge_index, node_sizes = _probe_layout()
    result = score_core_v3(pos, edge_index, node_sizes)

    assert set(result.scores) == {"tiered", "equal", "tier1_only"}
    assert result.facets["C1"].base_weight == pytest.approx(4.0)
    assert result.facets["C5"].base_weight == pytest.approx(2.0)
    assert result.facets["C9"].base_weight == pytest.approx(1.0)
    assert result.facets["C2"].effective_weight == pytest.approx(
        4.0 * crossing_weight_multiplier(pos.shape[0])
    )
    assert sum(1 for tier in CORE_TIERS.values() if tier == 1) == 3
    assert result.coverage["applicable_facets"] == 10
    assert 0.0 <= result.scores["equal"] <= 100.0
    assert 0.0 <= result.scores["tiered"] <= 100.0
    assert 0.0 <= result.scores["tier1_only"] <= 100.0
