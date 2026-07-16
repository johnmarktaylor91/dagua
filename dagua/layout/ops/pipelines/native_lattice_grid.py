"""Lattice/grid native route (native-sprint r2 wave 2).

Two complementary mechanisms, both entered into the honest undirected
portfolio contest (never bypassing the measured-argmax referee):

1. **Exact rectangular-grid certificate** (:func:`certify_rect_grid`):
   verify-then-emit. The detector proposes integer ``(row, col)`` slots from
   two corner BFS sweeps, then checks that the proposed lattice reproduces
   the graph's undirected edge set EXACTLY; on any mismatch it abstains.
   This is immune to overfit by construction -- it fires on any isomorphic
   ``R x C`` grid instance (a 7x9 grid it has never seen) and never on
   anything else (a grid plus one diagonal, a grid minus one rung, a torus,
   Petersen, ... all fail the exact edge-set check). A certified grid's
   integer-slot layout (:func:`certificate_grid_positions`) has zero edge
   crossings, zero edge-length variance, and perfect neighborhood structure.

2. **Geodesic-MDS + differentiable stress descent**
   (:func:`layout_geodesic_stress_pipeline`): classical MDS (Torgerson) on
   exact BFS/Dijkstra geodesics unfolds lattice-LIKE and mesh structures
   (triangular/hexagonal patches, sierpinski gaskets, weighted meshes,
   random geometric graphs, small symmetric graphs) almost perfectly; a
   short Adam descent on SMACOF-weighted stress (``w = d^-2``) then perfects
   local geometry. Prototype-measured (r2_fable_protos/proto_lattice_mds.py):
   grid edge-length CV 0.119 -> 0.018, sierpinski stress-1 0.178 -> 0.101,
   all three dominant r83-ruler axes improved on every weak-family structure.

No graph names or corpus-specific constants appear anywhere in this module.
Everything is pure PyTorch; no external layout binaries are invoked.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any, Optional

import torch

_LOGGER = logging.getLogger(__name__)

# Certificate cost is a handful of BFS sweeps; safe well past contest sizes.
GRID_CERTIFICATE_MAX_NODES = 20_000
# Dense APSP + eigh cap for the geodesic route (matches MAX_CONTEST_NODES).
GEODESIC_MAX_NODES = 1_500
# Size-scheduled descent budget: full refinement on small graphs, bounded on
# contest-cap graphs (the O(N^2) pair term dominates).
GEODESIC_FULL_STEPS = 500
GEODESIC_MEDIUM_STEPS = 300
GEODESIC_LARGE_STEPS = 150
GEODESIC_FULL_NODE_CAP = 150
GEODESIC_MEDIUM_NODE_CAP = 600
GEODESIC_DENSE_WORK_BYTES_CAP = 200 * 1024 * 1024
GEODESIC_DENSE_WORK_ELEMENT_CAP = GEODESIC_MAX_NODES * GEODESIC_MAX_NODES
# Default spacing used when node sizes are unavailable (points).
DEFAULT_TARGET_EDGE_LENGTH = 54.0


@dataclass(frozen=True)
class GridCertificate:
    """Exact rectangular-grid structure proof.

    Attributes
    ----------
    rows : int
        Number of lattice rows.
    cols : int
        Number of lattice columns.
    row_index : torch.Tensor
        Integer row slot per node, shape ``[N]``.
    col_index : torch.Tensor
        Integer column slot per node, shape ``[N]``.
    """

    rows: int
    cols: int
    row_index: torch.Tensor
    col_index: torch.Tensor


def _undirected_neighbor_lists(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Return deduplicated undirected neighbor lists.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        Neighbor lists without self-loops or duplicate pairs.
    """
    neighbor_sets: list[set[int]] = [set() for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
        for source, target in zip(cpu_edges[0].tolist(), cpu_edges[1].tolist()):
            if source == target:
                continue
            neighbor_sets[source].add(target)
            neighbor_sets[target].add(source)
    return [sorted(neighbors) for neighbors in neighbor_sets]


def _bfs_distances(adjacency: list[list[int]], start: int) -> list[int]:
    """Return BFS hop distances from ``start`` (``-1`` = unreachable).

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected neighbor lists.
    start : int
        BFS source node.

    Returns
    -------
    list[int]
        Distances per node.
    """
    distances = [-1] * len(adjacency)
    distances[start] = 0
    frontier = [start]
    while frontier:
        next_frontier: list[int] = []
        for node in frontier:
            for neighbor in adjacency[node]:
                if distances[neighbor] < 0:
                    distances[neighbor] = distances[node] + 1
                    next_frontier.append(neighbor)
        frontier = next_frontier
    return distances


def certify_rect_grid(edge_index: torch.Tensor, num_nodes: int) -> Optional[GridCertificate]:
    """Return an exact rectangular-grid certificate, or ``None``.

    Detection is verify-then-emit: coordinates are PROPOSED from corner BFS
    distances (in an ``R x C`` grid with corner ``A=(0,0)`` and same-row
    corner ``B=(0,C-1)``, every node satisfies ``d_A = r + c`` and
    ``d_B = r + (C-1) - c``, so ``c = (d_A - d_B + (C-1)) / 2``), then the
    proposal is VERIFIED by exact undirected edge-set equality against the
    ideal lattice. Any structural deviation -- an extra diagonal, a missing
    rung, wraparound edges, non-grid topology -- fails verification and the
    detector abstains.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]`` (direction ignored).
    num_nodes : int
        Number of nodes.

    Returns
    -------
    GridCertificate | None
        Certificate when the graph IS an ``R x C`` grid (``R, C >= 2``),
        otherwise ``None``.
    """
    if num_nodes < 4 or num_nodes > GRID_CERTIFICATE_MAX_NODES or edge_index.numel() == 0:
        return None
    adjacency = _undirected_neighbor_lists(edge_index, num_nodes)
    degrees = [len(neighbors) for neighbors in adjacency]
    if min(degrees) < 2 or max(degrees) > 4:
        return None
    corners = [node for node, degree in enumerate(degrees) if degree == 2]
    # Every R x C grid with R, C >= 2 has exactly four degree-2 corners
    # (the 2x2 grid is the 4-cycle whose four nodes are all corners).
    if len(corners) != 4:
        return None
    actual_edges = {
        (node, neighbor)
        for node, neighbors in enumerate(adjacency)
        for neighbor in neighbors
        if node < neighbor
    }

    start = corners[0]
    dist_start = _bfs_distances(adjacency, start)
    if min(dist_start) < 0:
        return None
    for other in corners[1:]:
        certificate = _try_grid_assignment(
            dist_start,
            _bfs_distances(adjacency, other),
            width_minus_one=dist_start[other],
            num_nodes=num_nodes,
            actual_edges=actual_edges,
        )
        if certificate is not None:
            return certificate
    return None


def _try_grid_assignment(
    dist_a: list[int],
    dist_b: list[int],
    width_minus_one: int,
    num_nodes: int,
    actual_edges: set[tuple[int, int]],
) -> Optional[GridCertificate]:
    """Verify one corner-pair grid coordinate hypothesis.

    Parameters
    ----------
    dist_a : list[int]
        BFS distances from the anchor corner (hypothesized ``(0, 0)``).
    dist_b : list[int]
        BFS distances from the candidate same-row corner ``(0, C-1)``.
    width_minus_one : int
        Hypothesized ``C - 1``.
    num_nodes : int
        Number of nodes.
    actual_edges : set[tuple[int, int]]
        Canonical undirected edge set (``source < target``).

    Returns
    -------
    GridCertificate | None
        Certificate when the hypothesis verifies exactly.
    """
    if width_minus_one < 1 or min(dist_b) < 0:
        return None
    cols: list[int] = []
    rows: list[int] = []
    for d_a, d_b in zip(dist_a, dist_b):
        numerator = d_a - d_b + width_minus_one
        if numerator % 2 != 0:
            return None
        col = numerator // 2
        row = d_a - col
        if col < 0 or row < 0 or col > width_minus_one:
            return None
        cols.append(col)
        rows.append(row)
    num_cols = width_minus_one + 1
    num_rows = max(rows) + 1
    if num_rows * num_cols != num_nodes:
        return None
    slot_to_node: dict[int, int] = {}
    for node in range(num_nodes):
        slot = rows[node] * num_cols + cols[node]
        if slot in slot_to_node:
            return None
        slot_to_node[slot] = node
    expected_edges: set[tuple[int, int]] = set()
    for row in range(num_rows):
        for col in range(num_cols):
            node = slot_to_node[row * num_cols + col]
            if col + 1 < num_cols:
                right = slot_to_node[row * num_cols + col + 1]
                expected_edges.add((min(node, right), max(node, right)))
            if row + 1 < num_rows:
                below = slot_to_node[(row + 1) * num_cols + col]
                expected_edges.add((min(node, below), max(node, below)))
    if expected_edges != actual_edges:
        return None
    return GridCertificate(
        rows=num_rows,
        cols=num_cols,
        row_index=torch.tensor(rows, dtype=torch.long),
        col_index=torch.tensor(cols, dtype=torch.long),
    )


def _grid_pitch(node_sizes: Optional[torch.Tensor], node_sep: float) -> float:
    """Return the uniform slot pitch for a certified grid layout.

    A single pitch for both axes keeps every lattice edge exactly equal
    (edge-length CV of zero), and ``max node diagonal + node_sep`` keeps the
    layout overlap-free by construction.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Node bounding boxes shaped ``[N, 2]``.
    node_sep : float
        Configured node separation in points.

    Returns
    -------
    float
        Slot pitch in points.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return max(float(node_sep), 1.0) + DEFAULT_TARGET_EDGE_LENGTH
    sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    if sizes.ndim == 1:
        sizes = sizes.unsqueeze(1).expand(-1, 2)
    max_diagonal = float(torch.linalg.vector_norm(sizes, dim=1).max().item())
    return max(float(node_sep), 1.0) + max(max_diagonal, 1.0)


def certificate_grid_positions(
    certificate: GridCertificate,
    node_sizes: Optional[torch.Tensor],
    node_sep: float,
) -> torch.Tensor:
    """Return the idealized integer-slot layout for a certified grid.

    Parameters
    ----------
    certificate : GridCertificate
        Verified grid structure.
    node_sizes : torch.Tensor, optional
        Node bounding boxes shaped ``[N, 2]``.
    node_sep : float
        Configured node separation in points.

    Returns
    -------
    torch.Tensor
        Centered positions shaped ``[N, 2]`` with uniform pitch.
    """
    pitch = _grid_pitch(node_sizes, node_sep)
    x = certificate.col_index.to(dtype=torch.float32) * pitch
    y = certificate.row_index.to(dtype=torch.float32) * pitch
    positions = torch.stack([x, y], dim=1)
    return positions - positions.mean(dim=0, keepdim=True)


def _geodesic_descent_steps(num_nodes: int) -> int:
    """Return the size-scheduled stress-descent budget.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    int
        Adam descent steps.
    """
    if num_nodes <= GEODESIC_FULL_NODE_CAP:
        return GEODESIC_FULL_STEPS
    if num_nodes <= GEODESIC_MEDIUM_NODE_CAP:
        return GEODESIC_MEDIUM_STEPS
    return GEODESIC_LARGE_STEPS


def geodesic_dense_work_is_allowed(
    num_nodes: int,
    edge_count: int,
    steps: Optional[int] = None,
    *,
    bytes_cap: int = GEODESIC_DENSE_WORK_BYTES_CAP,
) -> bool:
    """Return whether dense geodesic/MDS work fits the guard budget.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edge_count : int
        Number of edges.
    steps : int, optional
        Stress-descent steps; defaults to the size schedule.
    bytes_cap : int, default=GEODESIC_DENSE_WORK_BYTES_CAP
        Maximum estimated resident bytes for dense float64 matrices.

    Returns
    -------
    bool
        ``True`` when the caller may build dense all-pairs geodesic tensors.
    """
    n = max(0, int(num_nodes))
    e = max(0, int(edge_count))
    if n > GEODESIC_MAX_NODES:
        return False
    dense_elements = n * n
    scheduled_steps = _geodesic_descent_steps(n) if steps is None else max(0, int(steps))
    estimated_dense_bytes = dense_elements * 8 * 4
    estimated_sparse_bytes = e * 2 * 8
    dense_step_work = dense_elements * max(1, scheduled_steps)
    return (
        dense_elements <= GEODESIC_DENSE_WORK_ELEMENT_CAP
        and estimated_dense_bytes + estimated_sparse_bytes <= int(bytes_cap)
        and dense_step_work <= GEODESIC_DENSE_WORK_ELEMENT_CAP * GEODESIC_LARGE_STEPS
    )


def _target_edge_length(node_sizes: Optional[torch.Tensor], node_sep: float) -> float:
    """Return the point-unit spacing target for geodesic layouts.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Node bounding boxes shaped ``[N, 2]``.
    node_sep : float
        Configured node separation in points.

    Returns
    -------
    float
        Target mean edge length in points.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return DEFAULT_TARGET_EDGE_LENGTH
    sizes = node_sizes.detach().to(device="cpu", dtype=torch.float32)
    if sizes.ndim == 1:
        sizes = sizes.unsqueeze(1).expand(-1, 2)
    mean_diagonal = float(torch.linalg.vector_norm(sizes, dim=1).mean().item())
    return max(mean_diagonal + max(float(node_sep), 1.0), 1.0)


def _deterministic_fallback_positions(num_nodes: int, spacing: float, seed: int) -> torch.Tensor:
    """Return a finite deterministic circle + jitter layout (terminal rung).

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    spacing : float
        Approximate neighbor spacing in points.
    seed : int
        Deterministic jitter seed.

    Returns
    -------
    torch.Tensor
        Finite positions shaped ``[N, 2]``.
    """
    if num_nodes <= 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    angles = torch.arange(num_nodes, dtype=torch.float32) * (2.0 * math.pi / max(num_nodes, 1))
    radius = spacing * max(num_nodes, 1) / (2.0 * math.pi)
    generator = torch.Generator().manual_seed(int(seed))
    jitter = (torch.rand((num_nodes, 2), generator=generator) - 0.5) * (0.05 * spacing)
    return torch.stack([radius * torch.cos(angles), radius * torch.sin(angles)], dim=1) + jitter


def layout_geodesic_stress_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    config: Optional[Any] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    steps: Optional[int] = None,
    node_sep: Optional[float] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run geodesic classical MDS + differentiable SMACOF stress descent.

    Exact shortest-path geodesics (BFS, or Dijkstra when ``edge_weights``
    carry distance semantics) feed a Torgerson classical-MDS unfold, then a
    short Adam descent on ``sum_ij w_ij (|x_i - x_j| - d_ij)^2`` with the
    SMACOF weighting ``w = d^-2`` perfects local geometry. The result is
    scaled into point units so overlap terms see real node boxes. Never
    returns non-finite coordinates: any numerical failure falls back to a
    deterministic finite layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]`` (direction ignored).
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node bounding boxes shaped ``[N, 2]``.
    config : Any, optional
        Optional layout configuration carrying ``node_sep``.
    seed : int, default=42
        Deterministic seed (used only by degenerate fallbacks).
    edge_weights : torch.Tensor, optional
        Optional per-edge distance costs shaped ``[E]``.
    steps : int, optional
        Explicit descent budget; defaults to the size schedule.
    node_sep : float, optional
        Node separation override in points.
    **kwargs : Any
        Compatibility keywords accepted by generic dispatchers.

    Returns
    -------
    torch.Tensor
        Finite positions shaped ``[N, 2]`` in point units.
    """
    del kwargs
    resolved_sep = float(
        node_sep if node_sep is not None else getattr(config, "node_sep", 36.0) or 36.0
    )
    target_length = _target_edge_length(node_sizes, resolved_sep)
    if num_nodes <= 0:
        return torch.zeros((0, 2), dtype=torch.float32)
    if num_nodes == 1 or edge_index.numel() == 0:
        return _deterministic_fallback_positions(num_nodes, target_length, seed)
    if num_nodes > GEODESIC_MAX_NODES:
        raise ValueError(
            f"geodesic stress route caps at {GEODESIC_MAX_NODES} nodes (got {num_nodes})."
        )
    edge_count = int(edge_index.shape[1]) if edge_index.numel() else 0
    if not geodesic_dense_work_is_allowed(num_nodes, edge_count, steps):
        raise ValueError(
            "geodesic stress dense-work cap exceeded "
            f"(n={num_nodes}, e={edge_count}, steps={steps or _geodesic_descent_steps(num_nodes)})"
        )

    from dagua.layout.ops.graph_utils import shortest_path_distances

    distances_np = shortest_path_distances(edge_index, num_nodes, edge_weights)
    distances = torch.tensor(distances_np, dtype=torch.float64)
    distances = torch.nan_to_num(distances, nan=1.0, posinf=1.0, neginf=1.0).clamp_min(0.0)

    positions = _classical_mds_2d(distances)
    if positions is None:
        positions = _deterministic_fallback_positions(num_nodes, 1.0, seed).to(torch.float64)
    positions = _smacof_stress_descent(
        positions,
        distances,
        steps if steps is not None else _geodesic_descent_steps(num_nodes),
    )

    positions = positions.to(dtype=torch.float32)
    if not bool(torch.isfinite(positions).all().item()):
        positions = _deterministic_fallback_positions(num_nodes, target_length, seed)

    cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    deltas = positions[cpu_edges[1]] - positions[cpu_edges[0]]
    mean_edge_length = float(torch.linalg.vector_norm(deltas, dim=1).mean().item())
    if mean_edge_length > 1.0e-9 and math.isfinite(mean_edge_length):
        positions = positions * (target_length / mean_edge_length)
    else:
        positions = _deterministic_fallback_positions(num_nodes, target_length, seed)
    return positions - positions.mean(dim=0, keepdim=True)


def _classical_mds_2d(distances: torch.Tensor) -> Optional[torch.Tensor]:
    """Return the 2D Torgerson classical-MDS embedding.

    Parameters
    ----------
    distances : torch.Tensor
        Finite distance matrix shaped ``[N, N]`` (float64).

    Returns
    -------
    torch.Tensor | None
        Embedding shaped ``[N, 2]``, or ``None`` when the spectrum is
        degenerate or the eigensolve fails.
    """
    n = distances.shape[0]
    try:
        squared = distances**2
        centering = torch.eye(n, dtype=torch.float64) - torch.full(
            (n, n), 1.0 / n, dtype=torch.float64
        )
        gram = -0.5 * centering @ squared @ centering
        gram = 0.5 * (gram + gram.T)
        eigenvalues, eigenvectors = torch.linalg.eigh(gram)
    except Exception:  # noqa: BLE001 -- degenerate spectra fall back safely
        return None
    top = eigenvalues.argsort(descending=True)[:2]
    lam = eigenvalues[top].clamp_min(0.0)
    if float(lam.max().item()) <= 0.0:
        return None
    embedding = eigenvectors[:, top] * lam.sqrt().unsqueeze(0)
    if not bool(torch.isfinite(embedding).all().item()):
        return None
    return embedding


def _smacof_stress_descent(
    positions: torch.Tensor,
    distances: torch.Tensor,
    steps: int,
) -> torch.Tensor:
    """Run Adam on SMACOF-weighted stress from a given initialization.

    Parameters
    ----------
    positions : torch.Tensor
        Initial embedding shaped ``[N, 2]`` (float64).
    distances : torch.Tensor
        Target distance matrix shaped ``[N, N]``.
    steps : int
        Descent iterations.

    Returns
    -------
    torch.Tensor
        Descended positions shaped ``[N, 2]`` (float64); returns the input
        unchanged when descent is not applicable or diverges.
    """
    n = positions.shape[0]
    if n < 3 or steps <= 0:
        return positions
    pair_index = torch.triu_indices(n, n, offset=1)
    target = distances[pair_index[0], pair_index[1]].clamp_min(1.0e-9)
    weights = target.pow(-2.0)
    solution = positions.detach().clone().requires_grad_(True)
    learning_rate = 0.05 * float(target.max().item())
    optimizer = torch.optim.Adam([solution], lr=learning_rate)
    best = positions.detach().clone()
    for _step in range(int(steps)):
        optimizer.zero_grad()
        pair_lengths = torch.linalg.vector_norm(
            solution[pair_index[0]] - solution[pair_index[1]], dim=1
        ).clamp_min(1.0e-9)
        loss = (weights * (pair_lengths - target) ** 2).sum()
        if not bool(torch.isfinite(loss).item()):
            return best
        loss.backward()
        optimizer.step()
    final = solution.detach()
    if not bool(torch.isfinite(final).all().item()):
        return best
    return final


def layout_native_lattice_grid_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    config: Optional[Any] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the lattice/grid route: exact certificate, else geodesic stress.

    The certificate path only fires for exact unweighted (or uniformly
    weighted) rectangular grids; every other mesh-like structure gets the
    geodesic-MDS + stress-descent treatment. Weighted meshes use their
    weights as Dijkstra distance costs so anisotropic spacing is honored.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node bounding boxes shaped ``[N, 2]``.
    config : Any, optional
        Optional layout configuration carrying ``node_sep``.
    seed : int, default=42
        Deterministic seed.
    edge_weights : torch.Tensor, optional
        Optional per-edge distance costs shaped ``[E]``.
    **kwargs : Any
        Compatibility keywords accepted by generic dispatchers.

    Returns
    -------
    torch.Tensor
        Finite positions shaped ``[N, 2]`` in point units.
    """
    del kwargs
    resolved_sep = float(getattr(config, "node_sep", 36.0) or 36.0)
    weights_are_uniform = edge_weights is None or (
        edge_weights.numel() > 0
        and bool((edge_weights == edge_weights.reshape(-1)[0]).all().item())
    )
    if weights_are_uniform:
        certificate = certify_rect_grid(edge_index, num_nodes)
        if certificate is not None:
            return certificate_grid_positions(certificate, node_sizes, resolved_sep)
    return layout_geodesic_stress_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        config=config,
        seed=seed,
        edge_weights=edge_weights,
        node_sep=resolved_sep,
    )


__all__ = [
    "GEODESIC_MAX_NODES",
    "GRID_CERTIFICATE_MAX_NODES",
    "GridCertificate",
    "certificate_grid_positions",
    "certify_rect_grid",
    "layout_geodesic_stress_pipeline",
    "layout_native_lattice_grid_pipeline",
]
