"""Tests for router-v2, the lattice/grid route, and the community route.

Native-sprint r2 wave 2. Every probe graph here is GENERATED in-test (fresh
sizes/seeds not present in the benchmark corpus) so these tests double as the
out-of-corpus routing probes required by the anti-overfit protocol: a 7x9
grid must behave exactly like the corpus 6x8, a fresh brick-honeycomb like
the corpus 6x7, and node relabeling must never change a routing decision
(no name/id features exist to react to).
"""

from __future__ import annotations

import torch

from dagua.config import LayoutConfig
from dagua.eval.router_validation import (
    family_stratified_folds,
    held_out_fold,
    routing_change_accepted,
)
from dagua.layout.graph_classify import (
    classify_graph,
    label_propagation_communities,
    undirected_modularity,
)
from dagua.layout.ops.pipelines.dagua_native import (
    _choose_native_pipeline,
    _community_features_strong,
    _mesh_features_strong,
    _undirected_route_shortlist,
)
from dagua.layout.ops.pipelines.native_community import (
    layout_native_community_pipeline,
)
from dagua.layout.ops.pipelines.native_lattice_grid import (
    certificate_grid_positions,
    certify_rect_grid,
    layout_geodesic_stress_pipeline,
    layout_native_lattice_grid_pipeline,
)
from dagua.layout.ops.pipelines.native_undirected import _never_nan_winner
from dagua.layout.ops.state import LayoutProblem


def _grid_edges(rows: int, cols: int) -> torch.Tensor:
    """Return ascending-oriented rectangular grid edges.

    Parameters
    ----------
    rows : int
        Grid rows.
    cols : int
        Grid columns.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if col + 1 < cols:
                edges.append((node, node + 1))
            if row + 1 < rows:
                edges.append((node, node + cols))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _brick_honeycomb_edges(rows: int, cols: int) -> torch.Tensor:
    """Return a fresh brick-wall honeycomb patch (out-of-corpus geometry).

    Parameters
    ----------
    rows : int
        Lattice rows.
    cols : int
        Lattice columns.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    edges: list[tuple[int, int]] = []
    for row in range(rows):
        for col in range(cols):
            node = row * cols + col
            if row + 1 < rows:
                edges.append((node, node + cols))
            if col + 1 < cols and col % 2 == row % 2:
                edges.append((node, node + 1))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _planted_blocks_edges(block_size: int, num_blocks: int) -> torch.Tensor:
    """Return cliques joined by single bridges (planted community structure).

    Parameters
    ----------
    block_size : int
        Nodes per clique.
    num_blocks : int
        Number of cliques.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E]``.
    """
    edges: list[tuple[int, int]] = []
    for block in range(num_blocks):
        base = block * block_size
        for i in range(block_size):
            for j in range(i + 1, block_size):
                edges.append((base + i, base + j))
    for block in range(num_blocks - 1):
        edges.append((block * block_size, (block + 1) * block_size))
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _edge_length_cv(pos: torch.Tensor, edge_index: torch.Tensor) -> float:
    """Return the coefficient of variation of edge lengths.

    Parameters
    ----------
    pos : torch.Tensor
        Positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    float
        stddev / mean of edge lengths.
    """
    lengths = torch.linalg.vector_norm(pos[edge_index[1]] - pos[edge_index[0]], dim=1)
    return float((lengths.std(unbiased=False) / lengths.mean().clamp_min(1e-12)).item())


class _DeclaredUndirected:
    """Minimal graph stand-in carrying an explicit direction declaration."""

    is_semantically_directed = False
    edge_weights = None


# ---------------------------------------------------------------------------
# Certificate correctness
# ---------------------------------------------------------------------------


def test_certificate_fires_on_out_of_corpus_grid() -> None:
    """A fresh 7x9 grid (not in any corpus) must be certified exactly."""
    edge_index = _grid_edges(7, 9)
    certificate = certify_rect_grid(edge_index, 63)

    assert certificate is not None
    assert {certificate.rows, certificate.cols} == {7, 9}
    slots = certificate.row_index * max(certificate.cols, certificate.rows + 1)
    assert int(torch.unique(slots + certificate.col_index).numel()) == 63


def test_certificate_positions_have_zero_edge_length_variance() -> None:
    """Certified grid slots produce exactly uniform edge lengths."""
    edge_index = _grid_edges(6, 8)
    certificate = certify_rect_grid(edge_index, 48)
    assert certificate is not None
    node_sizes = torch.full((48, 2), 20.0)

    pos = certificate_grid_positions(certificate, node_sizes, node_sep=18.0)

    assert bool(torch.isfinite(pos).all().item())
    assert _edge_length_cv(pos, edge_index) < 1e-6


def test_certificate_abstains_on_near_grids() -> None:
    """Verify-then-emit: any structural deviation must abstain."""
    rows, cols = 6, 8
    grid = _grid_edges(rows, cols)

    diagonal = torch.tensor([[9], [9 + cols + 1]], dtype=torch.long)
    with_diagonal = torch.cat([grid, diagonal], dim=1)
    assert certify_rect_grid(with_diagonal, rows * cols) is None

    # Remove one interior rung (graph stays connected).
    keep = ~((grid[0] == 9) & (grid[1] == 10))
    missing_edge = grid[:, keep]
    assert certify_rect_grid(missing_edge, rows * cols) is None


def test_certificate_abstains_on_non_grid_topologies() -> None:
    """Cycles, regular graphs, and triangular meshes are not rect grids."""
    ring = torch.stack(
        [torch.arange(12, dtype=torch.long), (torch.arange(12, dtype=torch.long) + 1) % 12]
    )
    assert certify_rect_grid(ring, 12) is None

    # Petersen graph: 3-regular, no degree-2 corners.
    outer = [(i, (i + 1) % 5) for i in range(5)]
    spokes = [(i, i + 5) for i in range(5)]
    inner = [(5 + i, 5 + (i + 2) % 5) for i in range(5)]
    petersen = torch.tensor(outer + spokes + inner, dtype=torch.long).t()
    assert certify_rect_grid(petersen, 10) is None

    # Triangular mesh: max degree exceeds the grid bound.
    grid = _grid_edges(5, 5)
    diagonals = torch.tensor(
        [(r * 5 + c, (r + 1) * 5 + c + 1) for r in range(4) for c in range(4)],
        dtype=torch.long,
    ).t()
    triangular = torch.cat([grid, diagonals], dim=1)
    assert certify_rect_grid(triangular, 25) is None


def test_certificate_is_relabeling_invariant() -> None:
    """No name/id features: permuting node ids must not change the verdict."""
    edge_index = _grid_edges(5, 7)
    generator = torch.Generator().manual_seed(7)
    permutation = torch.randperm(35, generator=generator)
    permuted = permutation[edge_index]

    certificate = certify_rect_grid(permuted, 35)

    assert certificate is not None
    assert {certificate.rows, certificate.cols} == {5, 7}


# ---------------------------------------------------------------------------
# Geodesic stress route
# ---------------------------------------------------------------------------


def test_geodesic_stress_is_finite_and_uniform_on_grid() -> None:
    """The MDS + descent route recovers near-uniform grid geometry."""
    edge_index = _grid_edges(6, 8)
    node_sizes = torch.full((48, 2), 20.0)

    pos = layout_geodesic_stress_pipeline(
        edge_index=edge_index, num_nodes=48, node_sizes=node_sizes, seed=42
    )

    assert pos.shape == (48, 2)
    assert bool(torch.isfinite(pos).all().item())
    extent = pos.max(dim=0).values - pos.min(dim=0).values
    assert float(extent.min().item()) > 0.0
    assert _edge_length_cv(pos, edge_index) < 0.15


def test_geodesic_stress_handles_weighted_mesh() -> None:
    """Weighted meshes (Dijkstra geodesics) stay finite and non-degenerate."""
    grid = _grid_edges(5, 6)
    diagonals = torch.tensor([(0, 7), (13, 20)], dtype=torch.long).t()
    edge_index = torch.cat([grid, diagonals], dim=1)
    weights = torch.cat(
        [
            torch.full((grid.shape[1],), 2.0),
            torch.full((2,), 5.0),
        ]
    )

    pos = layout_geodesic_stress_pipeline(
        edge_index=edge_index,
        num_nodes=30,
        node_sizes=torch.full((30, 2), 18.0),
        edge_weights=weights,
        seed=42,
    )

    assert bool(torch.isfinite(pos).all().item())
    extent = pos.max(dim=0).values - pos.min(dim=0).values
    assert float(extent.min().item()) > 0.0


def test_lattice_grid_pipeline_prefers_certificate_for_exact_grids() -> None:
    """The combined route emits exact slots for grids, MDS for honeycombs."""
    grid = _grid_edges(7, 9)
    grid_pos = layout_native_lattice_grid_pipeline(
        edge_index=grid, num_nodes=63, node_sizes=torch.full((63, 2), 20.0), seed=42
    )
    assert _edge_length_cv(grid_pos, grid) < 1e-6

    honeycomb = _brick_honeycomb_edges(5, 9)
    honey_pos = layout_native_lattice_grid_pipeline(
        edge_index=honeycomb, num_nodes=45, node_sizes=torch.full((45, 2), 20.0), seed=42
    )
    assert bool(torch.isfinite(honey_pos).all().item())
    extent = honey_pos.max(dim=0).values - honey_pos.min(dim=0).values
    assert float(extent.min().item()) > 0.0


# ---------------------------------------------------------------------------
# Community structure
# ---------------------------------------------------------------------------


def test_label_propagation_is_deterministic_and_finds_planted_blocks() -> None:
    """LP must find the planted cliques and repeat bit-identically."""
    edge_index = _planted_blocks_edges(block_size=10, num_blocks=3)

    labels_a = label_propagation_communities(edge_index, 30)
    labels_b = label_propagation_communities(edge_index, 30)

    assert torch.equal(labels_a, labels_b)
    assert int(labels_a.max().item()) + 1 == 3
    for block in range(3):
        block_labels = labels_a[block * 10 : (block + 1) * 10]
        assert int(torch.unique(block_labels).numel()) == 1
    assert undirected_modularity(edge_index, labels_a, 30) > 0.4


def test_community_pipeline_separates_planted_blocks() -> None:
    """Community centroids must sit farther apart than intra spread."""
    edge_index = _planted_blocks_edges(block_size=10, num_blocks=3)
    node_sizes = torch.full((30, 2), 18.0)

    pos = layout_native_community_pipeline(
        edge_index=edge_index, num_nodes=30, node_sizes=node_sizes, seed=42
    )

    assert pos.shape == (30, 2)
    assert bool(torch.isfinite(pos).all().item())
    centroids = torch.stack([pos[b * 10 : (b + 1) * 10].mean(dim=0) for b in range(3)])
    intra = torch.stack(
        [
            torch.linalg.vector_norm(
                pos[b * 10 : (b + 1) * 10] - centroids[b].unsqueeze(0), dim=1
            ).mean()
            for b in range(3)
        ]
    )
    for a in range(3):
        for b in range(a + 1, 3):
            gap = float(torch.linalg.vector_norm(centroids[a] - centroids[b]).item())
            assert gap > float(intra.mean().item())


def test_community_pipeline_falls_back_without_structure() -> None:
    """A single clique has no mesoscale blocks; the flat core must serve."""
    clique = torch.tensor([(i, j) for i in range(8) for j in range(i + 1, 8)], dtype=torch.long).t()

    pos = layout_native_community_pipeline(
        edge_index=clique, num_nodes=8, node_sizes=torch.full((8, 2), 18.0), seed=42
    )

    assert pos.shape == (8, 2)
    assert bool(torch.isfinite(pos).all().item())


# ---------------------------------------------------------------------------
# Router features + shortlist per class
# ---------------------------------------------------------------------------


def test_classifier_populates_router_features() -> None:
    """New structural features must be measured on benchmark-scale graphs."""
    grid = _grid_edges(6, 8)
    structure = classify_graph(grid, 48)
    assert structure.degree_uniformity < 0.35
    assert structure.hub_edge_fraction < 0.45
    assert structure.diameter_estimate == 12  # (6-1) + (8-1)

    blocks = _planted_blocks_edges(10, 3)
    block_structure = classify_graph(blocks, 30)
    assert block_structure.community_score > 0.3
    assert block_structure.num_communities == 3

    star_edges = torch.stack(
        [torch.zeros(30, dtype=torch.long), torch.arange(1, 31, dtype=torch.long)]
    )
    star_structure = classify_graph(star_edges, 31)
    assert star_structure.hub_edge_fraction > 0.9
    assert not _mesh_features_strong(star_structure, 31)


def test_shortlist_matches_structure_classes() -> None:
    """Each structure class admits its candidate families, and only those."""
    grid_structure = classify_graph(_grid_edges(7, 9), 63)
    grid_shortlist = _undirected_route_shortlist(grid_structure, 63, has_edge_weights=False)
    assert "mesh" in grid_shortlist.classes
    assert "lattice_cert" in grid_shortlist.candidates
    assert "geodesic_stress" in grid_shortlist.candidates
    assert "community_scaffold" not in grid_shortlist.candidates

    block_structure = classify_graph(_planted_blocks_edges(10, 3), 30)
    block_shortlist = _undirected_route_shortlist(block_structure, 30, has_edge_weights=False)
    assert "community" in block_shortlist.classes
    assert "community_scaffold" in block_shortlist.candidates
    assert _community_features_strong(block_structure, 30)

    # Above the contest cap the shortlist is empty (incumbent runs alone).
    assert _undirected_route_shortlist(grid_structure, 5000, has_edge_weights=False) == (
        _undirected_route_shortlist(None, 5000, has_edge_weights=False)
    )
    assert not _undirected_route_shortlist(None, 5000, has_edge_weights=False).candidates


def test_out_of_corpus_probes_route_like_their_family() -> None:
    """Fresh lattice instances must route exactly like corpus siblings."""
    for rows, cols in ((6, 8), (7, 9)):  # corpus geometry vs fresh geometry
        edge_index = _grid_edges(rows, cols)
        num_nodes = rows * cols
        structure = classify_graph(edge_index, num_nodes, graph=_DeclaredUndirected())
        config = LayoutConfig(seed=42)
        setattr(config, "_dagua_native_num_nodes", num_nodes)
        assert _mesh_features_strong(structure, num_nodes)
        assert _choose_native_pipeline(structure, config) == "undirected_portfolio"

    honeycomb = _brick_honeycomb_edges(5, 9)  # fresh, corpus uses 6x7
    structure = classify_graph(honeycomb, 45, graph=_DeclaredUndirected())
    config = LayoutConfig(seed=42)
    setattr(config, "_dagua_native_num_nodes", 45)
    assert _mesh_features_strong(structure, 45)
    assert _choose_native_pipeline(structure, config) == "undirected_portfolio"


# ---------------------------------------------------------------------------
# Never-NaN fallback ladder
# ---------------------------------------------------------------------------


def test_never_nan_ladder_repairs_non_finite_winner() -> None:
    """A non-finite winner must come back finite via the safe core."""
    edge_index = _grid_edges(4, 5)
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=20,
        node_sizes=torch.full((20, 2), 18.0),
    )
    broken = torch.full((20, 2), float("nan"))

    repaired = _never_nan_winner(broken, problem, node_sep=18.0, seed=42)

    assert repaired.shape == (20, 2)
    assert bool(torch.isfinite(repaired).all().item())


def test_never_nan_ladder_passes_finite_winner_through_unchanged() -> None:
    """Finite winners must not be touched (bit-identical hot path)."""
    problem = LayoutProblem(
        edge_index=_grid_edges(3, 3),
        num_nodes=9,
        node_sizes=torch.full((9, 2), 18.0),
    )
    winner = torch.arange(18, dtype=torch.float32).reshape(9, 2)

    assert _never_nan_winner(winner, problem, node_sep=18.0, seed=42) is winner


# ---------------------------------------------------------------------------
# Rotating family-stratified validation protocol
# ---------------------------------------------------------------------------


def test_fold_assignment_is_stratified_and_rotates() -> None:
    """Folds balance every family and rotation shifts assignments."""
    families = {f"mesh_{i}": "mesh" for i in range(10)}
    families.update({f"sbm_{i}": "community" for i in range(10)})

    folds = family_stratified_folds(families, num_folds=5, rotation=0)
    for family in ("mesh", "community"):
        members = [folds[name] for name in folds if families[name] == family]
        counts = [members.count(fold) for fold in range(5)]
        assert max(counts) - min(counts) <= 1

    rotated = family_stratified_folds(families, num_folds=5, rotation=1)
    assert all(rotated[name] == (folds[name] + 1) % 5 for name in folds)
    assert held_out_fold(0) != held_out_fold(1)


def test_routing_change_promotion_rule() -> None:
    """The promotion rule blocks held-fold regressions and flat changes."""
    improving = {0: 2, 1: 1, 2: 0, 3: 1, 4: 0}
    assert routing_change_accepted(improving, rotation=4)
    held_regressed = {0: 2, 1: 1, 2: 0, 3: 1, 4: -1}
    assert not routing_change_accepted(held_regressed, rotation=4)
    training_regressed = {0: -1, 1: -2, 2: 3, 3: 1, 4: 0}
    assert not routing_change_accepted(training_regressed, rotation=4)
    no_strict_win = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0}
    assert not routing_change_accepted(no_strict_win, rotation=4)
