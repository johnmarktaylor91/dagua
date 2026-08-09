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
- ``planar_tutte_f{k}_polished`` -- outer-face variants: distinct faces of
  the cached embedding fixed as the convex boundary of a Tutte barycentric
  solve, then guard-polished. Only zero-crossing solves survive (the guard
  fails closed on non-3-connected collapses).
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
from typing import Dict, List, Optional, Sequence

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
# Distinct outer faces tried for the Tutte boundary variants.
PLANAR_ARM_OUTER_FACES = 3
# Legacy structural prior seconds for the directed opaque-arm cost table.
PLANAR_ARM_PRIOR_S = 3.0


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
) -> List[List[int]]:
    """Pick distinct simple candidate outer faces deterministically.

    Preference order: the largest face (most boundary room), the face with
    the highest total incident degree (hubs on the boundary relieve angular
    pressure inside), then the embedding's first face (the default the
    grid pipelines implicitly use). Non-simple walks (repeated vertices --
    cut-vertex faces) cannot serve as a convex Tutte boundary and are
    skipped.

    Parameters
    ----------
    faces : sequence[sequence[int]]
        Face vertex walks from the cached embedding.
    degrees : sequence[int]
        Undirected node degrees.
    limit : int
        Maximum number of faces returned.

    Returns
    -------
    list[list[int]]
        Up to ``limit`` distinct simple face walks.
    """
    simple = [
        list(face)
        for face in faces
        if len(face) >= 3 and len(set(int(node) for node in face)) == len(face)
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
    for candidate in (by_size[0], by_degree[0], simple[0]):
        key = canonical(candidate)
        if key in chosen_keys:
            continue
        chosen_keys.add(key)
        ranked.append(candidate)
        if len(ranked) >= limit:
            break
    return ranked


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
    refusal drops that candidate only, never the arm or the solve.

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
        if polished is not None:
            candidates[name] = polished

    try:
        fpp_pos = layout_fpp_pipeline(edge_index=edge_index, num_nodes=num_nodes)
        candidates["planar_fpp"] = fpp_pos
        _polish("planar_fpp_polished", fpp_pos, steps=PLANAR_ARM_POLISH_STEPS)
    except Exception:  # noqa: BLE001 -- one candidate fails closed
        _LOGGER.warning("planar arm FPP candidate failed", exc_info=True)

    try:
        schnyder_pos = layout_schnyder_pipeline(edge_index=edge_index, num_nodes=num_nodes)
        candidates["planar_schnyder"] = schnyder_pos
        _polish("planar_schnyder_polished", schnyder_pos, steps=PLANAR_ARM_POLISH_STEPS)
    except Exception:  # noqa: BLE001 -- one candidate fails closed
        _LOGGER.warning("planar arm Schnyder candidate failed", exc_info=True)

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


__all__ = [
    "PLANAR_ARM_MAX_NODES",
    "PLANAR_ARM_OUTER_FACES",
    "PLANAR_ARM_POLISH_STEPS",
    "PLANAR_ARM_PRIOR_S",
    "PLANAR_ARM_SEEDED_STRESS_STEPS",
    "build_planar_arm_candidates",
    "planar_arm_admitted",
]
