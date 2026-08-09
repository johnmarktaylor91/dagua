"""Planarity-guarded finisher polish for planar contest candidates.

Sprint2 W1-B: FPP/Schnyder-style planar drawings tie the specialist field
engines on crossings (both sit at zero) but lose the composite on angular
resolution, edge-length uniformity, and stress. This module polishes a
zero-crossing drawing under a two-layer planarity guard so the polished
candidate keeps ``crossings == 0`` while improving the movable facets:

1. CHEAP per-step screen -- face-sign line search. The realized embedding is
   extracted from the drawing itself (rotation system by angle, face walks by
   half-edge traversal) and every optimizer step is backtracked (step
   halving) until no face flips its reference winding sign.
2. EXACT batch acceptance -- face-winding preservation alone is NOT a
   sufficient planarity certificate (nonlocal crossings can appear with all
   faces un-flipped, e.g. when the guardable face set is empty around bridge
   trees). After every accepted step batch an exact segment-intersection
   count must be zero, otherwise the batch reverts to the last certified
   checkpoint.

Everything is deterministic: gradient descent from a deterministic input,
fixed iteration budgets, no RNG anywhere.
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_EPS = 1.0e-9
# Accepted-step batch size between exact planarity certificates. Small enough
# that a rare face-sign-clean nonlocal crossing costs at most 8 steps of
# rework; large enough that the O(E^2) exact check stays a minority cost.
_EXACT_CHECK_BATCH = 8
_MAX_STEP_HALVINGS = 6


def exact_crossing_count(pos: torch.Tensor, edge_index: torch.Tensor) -> int:
    """Return the exact crossing count for a straight-line drawing.

    ``dagua.metrics.count_crossings`` switches to a *sampled* (RNG-backed)
    estimate above 500 edges; a planarity certificate needs exactness at any
    gated size, so this helper runs the exact non-adjacent pair test
    directly.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    int
        Exact number of crossing edge pairs (collinear overlaps included).
    """
    from dagua.metrics import segments_intersect

    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edges.numel() == 0 or edges.shape[1] < 2:
        return 0
    work_pos = pos.detach().to(device="cpu", dtype=torch.float64)
    edge_count = int(edges.shape[1])
    source, target = edges[0], edges[1]
    first, second = torch.triu_indices(edge_count, edge_count, offset=1)
    shares_node = (
        (source[first] == source[second])
        | (source[first] == target[second])
        | (target[first] == source[second])
        | (target[first] == target[second])
    )
    valid = ~shares_node
    if not bool(valid.any().item()):
        return 0
    crossings = segments_intersect(
        work_pos[source[first[valid]]],
        work_pos[target[first[valid]]],
        work_pos[source[second[valid]]],
        work_pos[target[second[valid]]],
    )
    return int(crossings.sum().item())


def _simple_undirected_edges(edge_index: torch.Tensor) -> List[Tuple[int, int]]:
    """Return deduplicated undirected edges without self-loops.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    list[tuple[int, int]]
        Sorted unique ``(min, max)`` node pairs.
    """
    if edge_index.numel() == 0:
        return []
    pairs = {
        (min(int(u), int(v)), max(int(u), int(v)))
        for u, v in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist()
        if int(u) != int(v)
    }
    return sorted(pairs)


def drawing_faces(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Optional[List[List[int]]]:
    """Extract face walks of the embedding realized by a planar drawing.

    Builds the rotation system (neighbors sorted counter-clockwise by angle
    around each vertex) and traces face walks by half-edge traversal. The
    caller must ensure the drawing has zero crossings; the returned faces are
    then exactly the faces of the realized combinatorial embedding.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]] | None
        Face vertex walks, or ``None`` when a rotation system cannot be
        built (coincident adjacent endpoints).
    """
    edges = _simple_undirected_edges(edge_index)
    if not edges:
        return []
    coords = pos.detach().to(device="cpu", dtype=torch.float64)
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    scale = float((coords.max(dim=0).values - coords.min(dim=0).values).max().item())
    min_len = max(scale, 1.0) * 1.0e-12
    order_index: List[dict[int, int]] = [{} for _ in range(num_nodes)]
    for node in range(num_nodes):
        if not neighbors[node]:
            continue
        angles = []
        for other in neighbors[node]:
            dx = float(coords[other, 0] - coords[node, 0])
            dy = float(coords[other, 1] - coords[node, 1])
            if math.hypot(dx, dy) <= min_len:
                return None
            angles.append((math.atan2(dy, dx), other))
        angles.sort()
        neighbors[node] = [other for _, other in angles]
        order_index[node] = {other: i for i, other in enumerate(neighbors[node])}
    faces: List[List[int]] = []
    visited: set[Tuple[int, int]] = set()
    for start_u, start_v in (half for u, v in edges for half in ((u, v), (v, u))):
        if (start_u, start_v) in visited:
            continue
        face: List[int] = []
        current_u, current_v = start_u, start_v
        while (current_u, current_v) not in visited:
            visited.add((current_u, current_v))
            face.append(current_u)
            # Next half-edge in the face walk: at v, take the clockwise-next
            # neighbor after u (rotation system is CCW-sorted).
            ring = neighbors[current_v]
            position = order_index[current_v][current_u]
            next_target = ring[(position - 1) % len(ring)]
            current_u, current_v = current_v, next_target
        faces.append(face)
    return faces


def _face_signed_areas(pos: torch.Tensor, faces: Sequence[torch.Tensor]) -> torch.Tensor:
    """Return signed polygon areas for face walks.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    faces : sequence[torch.Tensor]
        Face vertex index tensors.

    Returns
    -------
    torch.Tensor
        Signed area per face with shape ``[F]``.
    """
    areas = []
    for face in faces:
        vertices = pos[face]
        shifted = torch.roll(vertices, shifts=-1, dims=0)
        areas.append(0.5 * (vertices[:, 0] * shifted[:, 1] - vertices[:, 1] * shifted[:, 0]).sum())
    if not areas:
        return pos.new_zeros(0)
    return torch.stack(areas)


class _FaceSignGuard:
    """Reference face windings of a drawing plus the cheap per-step screen."""

    def __init__(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
    ) -> None:
        """Extract the realized embedding's guardable faces.

        Faces whose reference area is degenerate (bridge walks, collinear
        chains) carry no stable winding sign and are dropped from the screen;
        the exact certificate covers those regions.

        Parameters
        ----------
        pos : torch.Tensor
            Zero-crossing reference positions with shape ``[N, 2]``.
        edge_index : torch.Tensor
            Edge tensor with shape ``[2, E]``.
        num_nodes : int
            Number of nodes.
        """
        walks = drawing_faces(pos, edge_index, num_nodes)
        self.valid = walks is not None
        self.faces: List[torch.Tensor] = []
        self.reference_signs = pos.new_zeros(0)
        if not walks:
            return
        reference = pos.detach().to(dtype=torch.float64)
        candidate_faces = [torch.tensor(walk, dtype=torch.long) for walk in walks if len(walk) >= 3]
        areas = _face_signed_areas(reference, candidate_faces)
        span = reference.max(dim=0).values - reference.min(dim=0).values
        area_floor = _EPS * float(span[0].item() * span[1].item() + 1.0)
        keep = [
            (face, float(area.item()))
            for face, area in zip(candidate_faces, areas)
            if abs(float(area.item())) > area_floor
        ]
        self.faces = [face for face, _ in keep]
        self.reference_signs = torch.tensor(
            [1.0 if area > 0.0 else -1.0 for _, area in keep], dtype=torch.float64
        )

    def signs_preserved(self, pos: torch.Tensor) -> bool:
        """Return whether every guarded face keeps its reference winding.

        Parameters
        ----------
        pos : torch.Tensor
            Candidate positions with shape ``[N, 2]``.

        Returns
        -------
        bool
            ``True`` when no guarded face flipped or degenerated.
        """
        if not self.faces:
            return True
        areas = _face_signed_areas(pos.detach().to(dtype=torch.float64), self.faces)
        return bool(((areas * self.reference_signs) > 0.0).all().item())


def _consecutive_incident_pairs(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return angularly consecutive incident-edge pairs per vertex.

    The rotation order is fixed by the input drawing; the face guard keeps
    it invariant during polish, so pairs precomputed once stay consecutive.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Center, first-neighbor, second-neighbor index tensors and the
        equiangular cosine target per pair.
    """
    edges = _simple_undirected_edges(edge_index)
    coords = pos.detach().to(device="cpu", dtype=torch.float64)
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    for u, v in edges:
        neighbors[u].append(v)
        neighbors[v].append(u)
    centers: List[int] = []
    first: List[int] = []
    second: List[int] = []
    targets: List[float] = []
    for node in range(num_nodes):
        ring = neighbors[node]
        degree = len(ring)
        if degree < 2:
            continue
        angles = sorted(
            (
                math.atan2(
                    float(coords[other, 1] - coords[node, 1]),
                    float(coords[other, 0] - coords[node, 0]),
                ),
                other,
            )
            for other in ring
        )
        ordered = [other for _, other in angles]
        cos_target = math.cos(2.0 * math.pi / degree)
        for index in range(degree if degree > 2 else 1):
            centers.append(node)
            first.append(ordered[index])
            second.append(ordered[(index + 1) % degree])
            targets.append(cos_target)
    return (
        torch.tensor(centers, dtype=torch.long),
        torch.tensor(first, dtype=torch.long),
        torch.tensor(second, dtype=torch.long),
        torch.tensor(targets, dtype=torch.float64),
    )


def _stress_pairs(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return all reachable node pairs with hop distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes (gated small, so the dense APSP is cheap).

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor]
        Pair index tensors ``[P]`` and hop distances ``[P]``.
    """
    from dagua.metrics import _all_pairs_unweighted, _build_csr

    offsets, targets = _build_csr(edge_index.detach().to(device="cpu"), num_nodes)
    hops = _all_pairs_unweighted(offsets, targets, num_nodes, max_dist=num_nodes)
    hop_tensor = torch.from_numpy(hops).to(dtype=torch.float64)
    row, col = torch.triu_indices(num_nodes, num_nodes, offset=1)
    distance = hop_tensor[row, col]
    reachable = distance > 0
    return row[reachable], col[reachable], distance[reachable]


class _PolishObjective:
    """Precomputed differentiable polish objective.

    Combines hop-distance stress (global shape), consecutive-angle angular
    resolution, edge-length uniformity, and node-box overlap -- the movable
    composite facets on which raw planar-grid drawings lose.
    """

    def __init__(
        self,
        pos: torch.Tensor,
        edge_index: torch.Tensor,
        num_nodes: int,
        node_sizes: Optional[torch.Tensor],
        *,
        stress_weight: float,
        angular_weight: float,
        uniformity_weight: float,
        overlap_weight: float,
    ) -> None:
        """Precompute pair structures from the input drawing.

        Parameters
        ----------
        pos : torch.Tensor
            Input positions with shape ``[N, 2]``.
        edge_index : torch.Tensor
            Edge tensor with shape ``[2, E]``.
        num_nodes : int
            Number of nodes.
        node_sizes : torch.Tensor, optional
            Node sizes with shape ``[N, 2]``.
        stress_weight : float
            Hop-stress term weight.
        angular_weight : float
            Angular-resolution term weight.
        uniformity_weight : float
            Edge-length coefficient-of-variation weight.
        overlap_weight : float
            Node-box overlap hinge weight.
        """
        self.edge_index = edge_index.detach().to(device="cpu", dtype=torch.long)
        self.node_sizes = (
            node_sizes.detach().to(device="cpu", dtype=torch.float64)
            if node_sizes is not None
            else None
        )
        self.stress_weight = stress_weight
        self.angular_weight = angular_weight
        self.uniformity_weight = uniformity_weight
        self.overlap_weight = overlap_weight
        coords = pos.detach().to(device="cpu", dtype=torch.float64)
        source = self.edge_index[0]
        target = self.edge_index[1]
        lengths = torch.linalg.norm(coords[target] - coords[source], dim=1)
        positive = lengths[lengths > 0]
        self.unit = float(positive.median().item()) if positive.numel() else 1.0
        self.unit = max(self.unit, 1.0e-6)
        self.row, self.col, hop = _stress_pairs(self.edge_index, num_nodes)
        self.stress_target = hop * self.unit
        self.stress_scale = (self.stress_target.square()).clamp(min=_EPS)
        centers, first, second, cos_targets = _consecutive_incident_pairs(
            pos, self.edge_index, num_nodes
        )
        self.angle_center = centers
        self.angle_first = first
        self.angle_second = second
        self.angle_cos_target = cos_targets

    def __call__(self, pos: torch.Tensor) -> torch.Tensor:
        """Evaluate the scalar polish loss.

        Parameters
        ----------
        pos : torch.Tensor
            Differentiable positions with shape ``[N, 2]``.

        Returns
        -------
        torch.Tensor
            Scalar loss.
        """
        from dagua.layout.ops.pipelines.native_surrogates import (
            edge_length_cv_loss,
            overlap_hinge_loss,
        )

        loss = pos.new_zeros(())
        if self.row.numel():
            deltas = pos[self.row] - pos[self.col]
            distances = torch.linalg.norm(deltas, dim=1).clamp(min=_EPS)
            loss = (
                loss
                + self.stress_weight
                * ((distances - self.stress_target).square() / self.stress_scale).mean()
            )
        if self.angle_center.numel():
            first_vec = torch.nn.functional.normalize(
                pos[self.angle_first] - pos[self.angle_center], dim=1, eps=_EPS
            )
            second_vec = torch.nn.functional.normalize(
                pos[self.angle_second] - pos[self.angle_center], dim=1, eps=_EPS
            )
            cosines = (first_vec * second_vec).sum(dim=1)
            # Angles below the equiangular target have cos above the target
            # cosine; wide angles are never penalized.
            loss = (
                loss
                + self.angular_weight * torch.relu(cosines - self.angle_cos_target).square().mean()
            )
        loss = loss + self.uniformity_weight * edge_length_cv_loss(pos, self.edge_index)
        if self.node_sizes is not None and self.overlap_weight > 0.0:
            loss = loss + self.overlap_weight * overlap_hinge_loss(pos, self.node_sizes)
        return loss


def guarded_descent(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    *,
    steps: int = 80,
    guarded: bool = True,
    stress_weight: float = 1.0,
    angular_weight: float = 1.0,
    uniformity_weight: float = 0.5,
    overlap_weight: float = 0.5,
) -> Optional[torch.Tensor]:
    """Polish a drawing by deterministic guarded gradient descent.

    Parameters
    ----------
    pos : torch.Tensor
        Input positions with shape ``[N, 2]``. When ``guarded``, the input
        must be an exact zero-crossing drawing or ``None`` is returned.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    steps : int, default=80
        Fixed descent iteration budget (iteration-based determinism).
    guarded : bool, default=True
        Apply the two-layer planarity guard. ``False`` runs plain bounded
        descent (used by the planar-seeded stress candidate, which is
        allowed to reintroduce crossings and is refereed honestly).
    stress_weight : float, default=1.0
        Hop-stress weight.
    angular_weight : float, default=1.0
        Angular-resolution weight.
    uniformity_weight : float, default=0.5
        Edge-length uniformity weight.
    overlap_weight : float, default=0.5
        Node-overlap hinge weight.

    Returns
    -------
    torch.Tensor | None
        Polished positions in the input dtype (guaranteed ``crossings == 0``
        when ``guarded``), or ``None`` when the guard cannot certify the
        input.
    """
    if num_nodes < 3 or edge_index.numel() == 0 or steps <= 0:
        return None
    original_dtype = pos.dtype
    original_device = pos.device
    work = pos.detach().to(device="cpu", dtype=torch.float64).clone()
    if not bool(torch.isfinite(work).all().item()):
        return None
    guard: Optional[_FaceSignGuard] = None
    if guarded:
        if exact_crossing_count(work, edge_index) != 0:
            return None
        guard = _FaceSignGuard(work, edge_index, num_nodes)
        if not guard.valid:
            return None
    objective = _PolishObjective(
        work,
        edge_index,
        num_nodes,
        node_sizes,
        stress_weight=stress_weight,
        angular_weight=angular_weight,
        uniformity_weight=uniformity_weight,
        overlap_weight=overlap_weight,
    )
    checkpoint = work.clone()
    accepted_since_check = 0
    for step in range(int(steps)):
        params = work.clone().requires_grad_(True)
        loss = objective(params)
        if not torch.isfinite(loss):
            break
        (gradient,) = torch.autograd.grad(loss, params)
        gradient = torch.nan_to_num(gradient, nan=0.0, posinf=0.0, neginf=0.0)
        max_norm = float(torch.linalg.norm(gradient, dim=1).max().item())
        if max_norm <= _EPS:
            break
        # Linearly decaying trust region: the largest single-node move goes
        # from 0.15 to 0.015 median-edge-lengths across the budget.
        step_fraction = 0.15 * (1.0 - 0.9 * step / max(int(steps) - 1, 1))
        delta = gradient * (-step_fraction * objective.unit / max_norm)
        moved = False
        for _ in range(_MAX_STEP_HALVINGS):
            candidate = work + delta
            if guard is None or guard.signs_preserved(candidate):
                work = candidate
                moved = True
                break
            delta = delta * 0.5
        if not moved:
            continue
        if guard is not None:
            accepted_since_check += 1
            if accepted_since_check >= _EXACT_CHECK_BATCH:
                accepted_since_check = 0
                if exact_crossing_count(work, edge_index) == 0:
                    checkpoint = work.clone()
                else:
                    work = checkpoint.clone()
    if guard is not None and exact_crossing_count(work, edge_index) != 0:
        work = checkpoint
    return work.to(device=original_device, dtype=original_dtype)


@register_op
class PlanarGuardedPolish(Op):
    """Polish ``state.pos`` under the two-layer planarity guard."""

    name: str = "planar_guarded_polish"
    category: OpCategory = OpCategory.OPTIMIZE
    reads: Tuple[str, ...] = ("pos",)
    writes: Tuple[str, ...] = ("pos",)

    def __init__(self, steps: int = 80) -> None:
        """Store the fixed iteration budget.

        Parameters
        ----------
        steps : int, default=80
            Descent iteration budget.
        """
        self.steps = int(steps)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Polish the current positions when the guard can certify them.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs.
        state : SolveState
            Mutable solve state with ``pos`` set.
        ctx : RuntimeContext
            Runtime context, unused.

        Returns
        -------
        SolveState
            State with polished positions, or unchanged when the input
            cannot be certified.
        """
        del ctx

        if state.pos is None:
            return state
        polished = guarded_descent(
            state.pos,
            problem.edge_index,
            int(problem.num_nodes),
            problem.node_sizes,
            steps=self.steps,
        )
        if polished is not None:
            state.pos = polished
        return state


__all__ = [
    "PlanarGuardedPolish",
    "drawing_faces",
    "exact_crossing_count",
    "guarded_descent",
]
