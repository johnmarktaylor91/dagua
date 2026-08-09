"""Planar-certificate contest arm: FPP/Schnyder candidates plus guarded polish.

Sprint2 W1-B. Exact planarity and the combinatorial embedding are already
computed and cached by ``graph_classify`` for every ``n <= 1500`` row, and
the deterministic FPP/Schnyder grid pipelines exist in-house -- but they were
never entered as contest candidates, leaving the planar rows to the external
specialist engines. This module builds the candidate family:

- ``planar_fpp`` / ``planar_schnyder`` -- verbatim pipeline output (parity
  floor with the reimplemented specialist engines; the pipelines are CALLED,
  never modified).
- ``planar_fpp_polished`` / ``planar_schnyder_polished`` -- finisher polish
  under the two-layer planarity guard (``planar_polish.guarded_descent``):
  crossings stay exactly zero while angular resolution, edge uniformity, and
  stress improve.
- ``planar_fpp_f{k}_polished`` / ``planar_schnyder_f{k}_polished`` -- the
  outer-face matrix (O19): 3-4 deterministically selected outer faces
  (largest, max-degree-incident, default, then the stable size ranking)
  crossed with BOTH embedding drawers -- the in-house FPP shift placement
  rooted at each face, and a true Schnyder-wood region-count drawing rooted
  at the same face. Every cell fails closed on a nonzero exact crossing
  count; downstream proxy stages (W1-C) cull the survivors cheaply.
- ``planar_tutte_f{k}_polished`` -- outer-face variants of the cached
  NetworkX embedding: distinct faces fixed as the convex boundary of a Tutte
  barycentric solve, then guard-polished. Only zero-crossing solves survive
  (the guard fails closed on non-3-connected collapses).
- ``planar_seeded_stress`` -- the embedding drawing as a warm start for
  bounded unguarded stress descent. It may reintroduce crossings; the honest
  referee compares it against the zero-crossing variants.

Everything is a REFEREED CANDIDATE: no route switching, the incumbent wins
ties, and on any row where :func:`planar_arm_admitted` is ``False`` no code
in this module runs (byte-inert gate-closed path).
"""

from __future__ import annotations

import logging
import math
from typing import Dict, List, Optional, Sequence, Tuple

import torch

from dagua.layout.ops.planar_polish import exact_crossing_count, guarded_descent
from dagua.layout.ops.state import LayoutProblem

_LOGGER = logging.getLogger(__name__)

# Structural gate ceiling. The exact-planarity embedding cache extends to the
# classifier ceiling (1500), but 500 bounds the polish's dense APSP/stress
# work and the O(E^2) exact planarity certificate to trivial cost; the six
# dev planar-loss rows sit at n <= ~100. Raise only on cousin-fitted cost
# evidence.
PLANAR_ARM_MAX_NODES = 500
# Guarded polish budget per candidate (iteration-based determinism).
PLANAR_ARM_POLISH_STEPS = 80
# Unguarded seeded-stress budget (may trade crossings for stress honestly).
PLANAR_ARM_SEEDED_STRESS_STEPS = 120
# Distinct outer faces tried for the FPP/Schnyder matrix and Tutte variants
# (spec: 3-4; the O(n^2) Schnyder region count stays trivial at the gate cap).
PLANAR_ARM_OUTER_FACES = 4
# Legacy structural prior seconds for the directed opaque-arm cost table.
PLANAR_ARM_PRIOR_S = 3.0
# OGDF grid unit shared with the FPP/Schnyder pipelines.
_GRID_SEPARATION = 40.0


def planar_arm_admitted(problem: LayoutProblem) -> bool:
    """Return whether the planar-certificate arm may build candidates.

    The gate is input-only structure: EXACT planarity (never the Euler
    hint -- ``is_planar`` is trusted only alongside a cached embedding, per
    the ``GraphStructure`` caveat), a single connected component (FPP and
    Schnyder assume connected input; the augmentation path exists but
    produces cross-component artifacts, so disconnected rows fail closed),
    and the polish cost ceiling.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem carrying the classifier output.

    Returns
    -------
    bool
        ``True`` when every structural condition holds.
    """
    structure = problem.structure
    if structure is None:
        return False
    if getattr(structure, "is_planar", None) is not True:
        return False
    if getattr(structure, "planar_embedding", None) is None:
        return False
    num_nodes = int(problem.num_nodes)
    if num_nodes < 4 or num_nodes > PLANAR_ARM_MAX_NODES:
        return False
    if int(getattr(structure, "num_components", 0)) != 1:
        return False
    if problem.edge_index.numel() == 0:
        return False
    return True


def _undirected_degrees(edge_index: torch.Tensor, num_nodes: int) -> List[int]:
    """Return undirected simple-graph degrees.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[int]
        Degree per node over deduplicated non-loop edges.
    """
    degrees = [0] * num_nodes
    seen = set()
    for u, v in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist():
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if int(u) == int(v) or key in seen:
            continue
        seen.add(key)
        degrees[key[0]] += 1
        degrees[key[1]] += 1
    return degrees


def _select_outer_faces(
    faces: Sequence[Sequence[int]],
    degrees: Sequence[int],
    limit: int,
    *,
    require_simple: bool = True,
) -> List[List[int]]:
    """Pick distinct candidate outer faces deterministically.

    Preference order: the largest face (most boundary room), the face with
    the highest total incident degree (hubs on the boundary relieve angular
    pressure inside), the embedding's first face (the default the grid
    pipelines implicitly use), then the remaining faces of the stable
    size-then-canonical ranking until ``limit`` is reached.

    Parameters
    ----------
    faces : sequence[sequence[int]]
        Face vertex walks from the cached embedding.
    degrees : sequence[int]
        Undirected node degrees.
    limit : int
        Maximum number of faces returned.
    require_simple : bool, default=True
        Skip non-simple walks (repeated vertices -- cut-vertex faces). A
        convex Tutte boundary needs a simple cycle; the FPP/Schnyder matrix
        only roots at one face EDGE, so it accepts any walk.

    Returns
    -------
    list[list[int]]
        Up to ``limit`` distinct face walks.
    """
    simple = [
        list(face)
        for face in faces
        if len(face) >= 3
        and (not require_simple or len(set(int(node) for node in face)) == len(face))
    ]
    if not simple:
        return []

    def canonical(face: Sequence[int]) -> tuple[int, ...]:
        rotations = [tuple(face[i:]) + tuple(face[:i]) for i in range(len(face))]
        reverse = list(reversed(face))
        rotations += [tuple(reverse[i:]) + tuple(reverse[:i]) for i in range(len(reverse))]
        return min(rotations)

    ranked: List[List[int]] = []
    by_size = sorted(simple, key=lambda face: (-len(face), canonical(face)))
    by_degree = sorted(
        simple,
        key=lambda face: (-sum(degrees[int(node)] for node in face), canonical(face)),
    )
    chosen_keys: set[tuple[int, ...]] = set()
    for candidate in [by_size[0], by_degree[0], simple[0], *by_size]:
        key = canonical(candidate)
        if key in chosen_keys:
            continue
        chosen_keys.add(key)
        ranked.append(candidate)
        if len(ranked) >= limit:
            break
    return ranked


def _inhouse_embedding_faces(embedding: object) -> List[List[int]]:
    """Extract face walks from the in-house ``PlanarEmbedding``.

    Mirrors ``native_planar._embedding_faces`` (which relies on NetworkX's
    ``traverse_face``) using the in-house half-edge API.

    Parameters
    ----------
    embedding : object
        ``dagua.layout.ops.pipelines.planar.PlanarEmbedding`` instance.

    Returns
    -------
    list[list[int]]
        Face vertex walks in deterministic embedding order.
    """
    data = embedding.get_data()  # type: ignore[attr-defined]
    visited: set[Tuple[int, int]] = set()
    faces: List[List[int]] = []
    for node, ring in data.items():
        for neighbor in ring:
            if (int(node), int(neighbor)) in visited:
                continue
            face: List[int] = []
            current = (int(node), int(neighbor))
            while current not in visited:
                visited.add(current)
                face.append(current[0])
                step = embedding.next_face_half_edge(*current)  # type: ignore[attr-defined]
                current = (int(step[0]), int(step[1]))
            if len(face) >= 3:
                faces.append(face)
    return faces


def _outer_triangles_for_face(triangulated: object, face: Sequence[int]) -> List[List[int]]:
    """Return candidate outer triangles rooting a drawing at one face.

    The canonical-ordering machinery takes a triangle ``[v1, v2, v3]`` of the
    fully triangulated embedding; fixing ``(v1, v2)`` to an edge of the
    requested face makes that face's corner the outer boundary root. The face
    walk's orientation inside the triangulated embedding is not knowable
    cheaply, so both orientations of the first edge are offered; callers try
    them in order and keep the first drawing that certifies.

    Parameters
    ----------
    triangulated : object
        Fully triangulated in-house ``PlanarEmbedding``.
    face : sequence[int]
        Simple face walk of the pre-triangulation embedding.

    Returns
    -------
    list[list[int]]
        Zero, one, or two ``[v1, v2, v3]`` outer-triangle candidates.
    """
    triangles: List[List[int]] = []
    first, second = int(face[0]), int(face[1])
    for v1, v2 in ((first, second), (second, first)):
        try:
            v3 = int(triangulated[v2][v1]["ccw"])  # type: ignore[index]
        except Exception:  # noqa: BLE001 -- edge missing means this rooting is invalid
            continue
        if len({v1, v2, v3}) == 3:
            triangles.append([v1, v2, v3])
    return triangles


def _fpp_positions_for_face(
    triangulated: object,
    face: Sequence[int],
    num_nodes: int,
    edge_index: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Run the FPP shift placement rooted at a chosen outer face.

    Parameters
    ----------
    triangulated : object
        Fully triangulated in-house ``PlanarEmbedding``.
    face : sequence[int]
        Simple face walk selected as the outer boundary.
    num_nodes : int
        Number of nodes.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]`` for the exact certificate.

    Returns
    -------
    torch.Tensor | None
        Zero-crossing OGDF-scale positions, or ``None`` when no rooting
        certifies (fail closed).
    """
    from dagua.layout.ops.pipelines.planar import (
        get_canonical_ordering,
        shift_placement_positions,
    )

    for outer in _outer_triangles_for_face(triangulated, face):
        try:
            node_list = get_canonical_ordering(triangulated, outer)  # type: ignore[arg-type]
            raw = shift_placement_positions(node_list)
        except Exception:  # noqa: BLE001 -- wrong-orientation rootings fail closed
            continue
        if len(raw) != num_nodes:
            continue
        output = torch.zeros((num_nodes, 2), dtype=torch.float64)
        y_max = max(int(value[1]) for value in raw.values())
        for node, (x_coord, y_coord) in raw.items():
            output[node, 0] = float(x_coord) * _GRID_SEPARATION
            output[node, 1] = float(y_max - int(y_coord)) * _GRID_SEPARATION
        if exact_crossing_count(output, edge_index) == 0:
            return output
    return None


def _schnyder_positions_for_face(
    triangulated: object,
    face: Sequence[int],
    num_nodes: int,
    edge_index: torch.Tensor,
) -> Optional[torch.Tensor]:
    """Draw a true Schnyder-wood barycentric layout rooted at a chosen face.

    The realizer comes from the canonical ordering of the fully triangulated
    embedding (leftmost contour neighbor -> tree 1, rightmost -> tree 2,
    cover events -> tree 3); each internal vertex is placed at the
    barycentric coordinates given by the number of internal faces in its
    three Schnyder regions (region face counts computed by a dual flood fill
    walled by the vertex's three tree paths). Every structural invariant is
    checked and any violation fails closed -- the exact zero-crossing
    certificate is the final acceptance test.

    Parameters
    ----------
    triangulated : object
        Fully triangulated in-house ``PlanarEmbedding``.
    face : sequence[int]
        Simple face walk selected as the outer boundary.
    num_nodes : int
        Number of nodes.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]`` for the exact certificate.

    Returns
    -------
    torch.Tensor | None
        Zero-crossing positions, or ``None`` when no rooting certifies.
    """
    from dagua.layout.ops.pipelines.planar import get_canonical_ordering

    for outer in _outer_triangles_for_face(triangulated, face):
        try:
            node_list = get_canonical_ordering(triangulated, outer)  # type: ignore[arg-type]
            positions = _schnyder_positions_from_ordering(triangulated, node_list, num_nodes)
        except Exception:  # noqa: BLE001 -- wrong-orientation rootings fail closed
            continue
        if positions is None:
            continue
        if exact_crossing_count(positions, edge_index) == 0:
            return positions
    return None


def _schnyder_positions_from_ordering(
    triangulated: object,
    node_list: Sequence[Tuple[int, List[int]]],
    num_nodes: int,
) -> Optional[torch.Tensor]:
    """Compute Schnyder region-count coordinates from a canonical ordering.

    Parameters
    ----------
    triangulated : object
        Fully triangulated in-house ``PlanarEmbedding``.
    node_list : sequence[tuple[int, list[int]]]
        Canonical ordering ``(node, contour_neighbors)`` entries.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor | None
        Barycentric positions on an equilateral frame, or ``None`` when a
        Schnyder invariant fails (fail closed).
    """
    if len(node_list) != num_nodes or num_nodes < 4:
        return None
    root1 = int(node_list[0][0])
    root2 = int(node_list[1][0])
    root3 = int(node_list[-1][0])

    # Realizer from the canonical ordering: when v_k arrives with contour
    # neighbors [w_p .. w_q], edge (v_k, w_p) joins tree 1, (v_k, w_q) joins
    # tree 2, and every intermediate w_i is covered -- its tree-3 parent is
    # v_k. Roots keep no parents.
    parent1: Dict[int, int] = {}
    parent2: Dict[int, int] = {}
    parent3: Dict[int, int] = {}
    for order_index in range(2, len(node_list)):
        node, contour = node_list[order_index]
        if len(contour) < 2:
            return None
        if node != root3:
            parent1[int(node)] = int(contour[0])
            parent2[int(node)] = int(contour[-1])
        for covered in contour[1:-1]:
            parent3[int(covered)] = int(node)

    internal = [node for node in range(num_nodes) if node not in (root1, root2, root3)]
    for node in internal:
        if node not in parent1 or node not in parent2 or node not in parent3:
            return None

    faces = [face for face in _inhouse_embedding_faces(triangulated) if len(face) == 3]
    outer_key = frozenset((root1, root2, root3))
    outer_indices = [index for index, face in enumerate(faces) if frozenset(face) == outer_key]
    if len(outer_indices) != 1:
        return None
    outer_index = outer_indices[0]
    inner_indices = [index for index in range(len(faces)) if index != outer_index]
    expected_inner = 2 * num_nodes - 5
    if len(inner_indices) != expected_inner:
        return None

    edge_to_faces: Dict[frozenset, List[int]] = {}
    for index in inner_indices:
        walk = faces[index]
        for position in range(3):
            key = frozenset((int(walk[position]), int(walk[(position + 1) % 3])))
            edge_to_faces.setdefault(key, []).append(index)

    def seed_face(a: int, b: int) -> Optional[int]:
        incident = edge_to_faces.get(frozenset((a, b)), [])
        return incident[0] if len(incident) == 1 else None

    seeds = (seed_face(root2, root3), seed_face(root1, root3), seed_face(root1, root2))
    if any(seed is None for seed in seeds):
        return None

    def tree_path_edges(start: int, parent: Dict[int, int], root: int) -> Optional[set]:
        edges: set = set()
        current = start
        for _ in range(num_nodes):
            if current == root:
                return edges
            nxt = parent.get(current)
            if nxt is None:
                return None
            edges.add(frozenset((current, nxt)))
            current = nxt
        return None

    side = _GRID_SEPARATION * float(expected_inner)
    corners = {
        root1: (0.0, 0.0),
        root2: (side, 0.0),
        root3: (side / 2.0, side * math.sqrt(3.0) / 2.0),
    }
    output = torch.zeros((num_nodes, 2), dtype=torch.float64)
    for root, coordinate in corners.items():
        output[root] = torch.tensor(coordinate, dtype=torch.float64)

    for node in internal:
        walls: set = set()
        for parent, root in ((parent1, root1), (parent2, root2), (parent3, root3)):
            path_edges = tree_path_edges(node, parent, root)
            if path_edges is None:
                return None
            walls |= path_edges
        labels: Dict[int, int] = {}
        counts = [0, 0, 0]
        for region, seed in enumerate(seeds):
            if seed in labels:
                return None
            stack = [seed]
            labels[seed] = region  # type: ignore[index]
            while stack:
                face_index = stack.pop()
                counts[region] += 1
                walk = faces[face_index]  # type: ignore[index]
                for position in range(3):
                    key = frozenset((int(walk[position]), int(walk[(position + 1) % 3])))
                    if key in walls:
                        continue
                    for neighbor_face in edge_to_faces.get(key, []):
                        if neighbor_face not in labels:
                            labels[neighbor_face] = region
                            stack.append(neighbor_face)
        if sum(counts) != expected_inner or min(counts) < 1:
            return None
        weight = torch.tensor(
            [float(count) / float(expected_inner) for count in counts],
            dtype=torch.float64,
        )
        frame = torch.tensor(
            [corners[root1], corners[root2], corners[root3]],
            dtype=torch.float64,
        )
        output[node] = weight @ frame
    return output


def _tutte_face_positions(
    problem: LayoutProblem,
    boundary: Sequence[int],
    spacing: float,
) -> Optional[torch.Tensor]:
    """Solve a Tutte barycentric layout with one face as the fixed boundary.

    Parameters
    ----------
    problem : LayoutProblem
        Layout inputs.
    boundary : sequence[int]
        Simple face walk fixed on a regular convex polygon.
    spacing : float
        Target boundary polygon edge length.

    Returns
    -------
    torch.Tensor | None
        Position tensor with shape ``[N, 2]``, or ``None`` when the solve
        degenerates (non-finite output).
    """
    from dagua.layout.ops.tutte import _edge_weights, _regular_polygon

    num_nodes = int(problem.num_nodes)
    boundary_ids = [int(node) for node in boundary]
    count = len(boundary_ids)
    radius = max(spacing, 1.0) / (2.0 * math.sin(math.pi / count))
    coordinates = _regular_polygon(boundary_ids, radius=radius)
    positions = torch.zeros((num_nodes, 2), dtype=torch.float64)
    for node, coordinate in coordinates.items():
        positions[node] = torch.tensor(coordinate, dtype=torch.float64)
    boundary_set = set(boundary_ids)
    interior = [node for node in range(num_nodes) if node not in boundary_set]
    if interior:
        interior_index = {node: index for index, node in enumerate(interior)}
        lhs = torch.zeros((len(interior), len(interior)), dtype=torch.float64)
        rhs = torch.zeros((len(interior), 2), dtype=torch.float64)
        for source, target, weight in _edge_weights(problem):
            for row_node, col_node in ((source, target), (target, source)):
                if row_node not in interior_index:
                    continue
                row = interior_index[row_node]
                lhs[row, row] += weight
                if col_node in interior_index:
                    lhs[row, interior_index[col_node]] -= weight
                else:
                    rhs[row] += weight * positions[col_node]
        try:
            positions[interior] = torch.linalg.solve(lhs, rhs)
        except RuntimeError:
            return None
    if not bool(torch.isfinite(positions).all().item()):
        return None
    return positions


def build_planar_arm_candidates(
    problem: LayoutProblem,
    *,
    node_sep: float,
) -> Dict[str, torch.Tensor]:
    """Build the planar-certificate candidate family for one gated row.

    Every candidate fails closed independently: an exception or a guard
    refusal drops that candidate only, never the arm or the solve. Exact
    duplicates of an earlier candidate are skipped (the default-face matrix
    cells often reproduce the base drawings byte-identically).

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem (``planar_arm_admitted`` must be ``True``).
    node_sep : float
        Contest node separation in points, used as the Tutte/seed spacing
        floor.

    Returns
    -------
    dict[str, torch.Tensor]
        Candidate name to raw positions.
    """
    from dagua.layout.ops.pipelines.fpp import layout_fpp_pipeline
    from dagua.layout.ops.pipelines.native_planar import (
        _embedding_faces,
        _embedding_positions,
    )
    from dagua.layout.ops.pipelines.planar import check_planarity, triangulate_embedding
    from dagua.layout.ops.pipelines.schnyder import layout_schnyder_pipeline

    num_nodes = int(problem.num_nodes)
    edge_index = problem.edge_index.detach().to(device="cpu", dtype=torch.long)
    node_sizes = (
        problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)
        if problem.node_sizes is not None
        else None
    )
    spacing = max(float(node_sep), 40.0)
    candidates: Dict[str, torch.Tensor] = {}

    def _is_duplicate(pos: torch.Tensor) -> bool:
        return any(
            existing.shape == pos.shape and torch.equal(existing, pos)
            for existing in candidates.values()
        )

    def _polish(name: str, raw: torch.Tensor, *, steps: int) -> None:
        try:
            polished = guarded_descent(
                raw,
                edge_index,
                num_nodes,
                node_sizes,
                steps=steps,
            )
        except Exception:  # noqa: BLE001 -- one candidate fails closed
            _LOGGER.warning("planar arm polish failed for %s", name, exc_info=True)
            return
        if polished is not None and not _is_duplicate(polished):
            candidates[name] = polished

    try:
        fpp_pos = layout_fpp_pipeline(edge_index=edge_index, num_nodes=num_nodes)
        # Parity floors carry the family certificate too: a verbatim pipeline
        # output that crosses (the Schnyder fallback grid is structure-blind)
        # must not enter under a planar-family name; it fails closed and the
        # certified matrix/polish variants carry the family on that row.
        if exact_crossing_count(fpp_pos, edge_index) == 0:
            candidates["planar_fpp"] = fpp_pos
        _polish("planar_fpp_polished", fpp_pos, steps=PLANAR_ARM_POLISH_STEPS)
    except Exception:  # noqa: BLE001 -- one candidate fails closed
        _LOGGER.warning("planar arm FPP candidate failed", exc_info=True)

    try:
        schnyder_pos = layout_schnyder_pipeline(edge_index=edge_index, num_nodes=num_nodes)
        if exact_crossing_count(schnyder_pos, edge_index) == 0:
            candidates["planar_schnyder"] = schnyder_pos
        _polish("planar_schnyder_polished", schnyder_pos, steps=PLANAR_ARM_POLISH_STEPS)
    except Exception:  # noqa: BLE001 -- one candidate fails closed
        _LOGGER.warning("planar arm Schnyder candidate failed", exc_info=True)

    # Outer-face matrix (O19): 3-4 deterministic outer faces x both embedding
    # drawers. Faces come from the in-house embedding the drawers actually
    # use; each cell is exact-certified before polish and fails closed alone.
    try:
        is_planar, inhouse_embedding = check_planarity(edge_index, num_nodes)
        if is_planar and inhouse_embedding is not None:
            triangulated, _ = triangulate_embedding(inhouse_embedding, fully_triangulate=True)
            inhouse_faces = _inhouse_embedding_faces(inhouse_embedding)
            degrees = _undirected_degrees(edge_index, num_nodes)
            for index, face in enumerate(
                _select_outer_faces(
                    inhouse_faces,
                    degrees,
                    PLANAR_ARM_OUTER_FACES,
                    require_simple=False,
                )
            ):
                fpp_variant = _fpp_positions_for_face(triangulated, face, num_nodes, edge_index)
                if fpp_variant is not None and not _is_duplicate(fpp_variant):
                    _polish(
                        f"planar_fpp_f{index}_polished",
                        fpp_variant,
                        steps=PLANAR_ARM_POLISH_STEPS,
                    )
                schnyder_variant = _schnyder_positions_for_face(
                    triangulated, face, num_nodes, edge_index
                )
                if schnyder_variant is not None and not _is_duplicate(schnyder_variant):
                    _polish(
                        f"planar_schnyder_f{index}_polished",
                        schnyder_variant,
                        steps=PLANAR_ARM_POLISH_STEPS,
                    )
    except Exception:  # noqa: BLE001 -- the matrix fails closed as a block
        _LOGGER.warning("planar arm outer-face matrix failed", exc_info=True)

    embedding = getattr(problem.structure, "planar_embedding", None)
    if embedding is not None:
        try:
            faces = _embedding_faces(embedding)
            degrees = _undirected_degrees(edge_index, num_nodes)
            for index, face in enumerate(
                _select_outer_faces(faces, degrees, PLANAR_ARM_OUTER_FACES)
            ):
                tutte_pos = _tutte_face_positions(problem, face, spacing)
                if tutte_pos is None:
                    continue
                if exact_crossing_count(tutte_pos, edge_index) != 0:
                    # Tutte only certifies planarity for 3-connected inputs;
                    # collapsed or crossing solves fail closed here.
                    continue
                if _is_duplicate(tutte_pos):
                    continue
                _polish(
                    f"planar_tutte_f{index}_polished",
                    tutte_pos,
                    steps=PLANAR_ARM_POLISH_STEPS,
                )
        except Exception:  # noqa: BLE001 -- one candidate fails closed
            _LOGGER.warning("planar arm Tutte variants failed", exc_info=True)

        try:
            seed_pos = _embedding_positions(embedding, problem, spacing)
            seeded = guarded_descent(
                seed_pos,
                edge_index,
                num_nodes,
                node_sizes,
                steps=PLANAR_ARM_SEEDED_STRESS_STEPS,
                guarded=False,
                stress_weight=3.0,
                angular_weight=0.25,
                uniformity_weight=0.25,
                overlap_weight=0.5,
            )
            if seeded is not None:
                candidates["planar_seeded_stress"] = seeded
        except Exception:  # noqa: BLE001 -- one candidate fails closed
            _LOGGER.warning("planar arm seeded stress failed", exc_info=True)

    return candidates


# Candidate names in this family that carry the zero-crossing certificate;
# the seeded-stress warm start is honestly allowed to reintroduce crossings.
def planar_candidate_requires_certificate(name: str) -> bool:
    """Return whether a planar-arm candidate must stay exactly crossing-free.

    Parameters
    ----------
    name : str
        Candidate family name produced by :func:`build_planar_arm_candidates`.

    Returns
    -------
    bool
        ``True`` for every planar-family candidate except the seeded-stress
        warm start.
    """
    return name.startswith("planar_") and name != "planar_seeded_stress"


__all__ = [
    "PLANAR_ARM_MAX_NODES",
    "PLANAR_ARM_OUTER_FACES",
    "PLANAR_ARM_POLISH_STEPS",
    "PLANAR_ARM_PRIOR_S",
    "PLANAR_ARM_SEEDED_STRESS_STEPS",
    "build_planar_arm_candidates",
    "planar_arm_admitted",
    "planar_candidate_requires_certificate",
]
