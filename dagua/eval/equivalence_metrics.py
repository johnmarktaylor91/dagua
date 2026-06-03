"""Layout equivalence metrics for fidelity evaluation.

This module compares a dagua reimplementation layout against a reference
layout using topology-aware and basis-invariant signals.  It is intended for
post-benchmark analysis only; it never delegates to reference layout engines.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any, Iterable, Optional, Sequence

import numpy as np
import torch

from dagua.layout.ops.graph_utils import shortest_path_distances
from dagua.metrics import count_crossings

EPSILON: float = 1.0e-12
DEFAULT_MAX_AUTOMORPHISMS: int = 20_000
DEFAULT_NEIGHBORHOOD_K: int = 10
FREE_ASPECT_ENGINES: set[str] = {"classic_sugiyama"}


@dataclass(frozen=True)
class EquivalenceMetrics:
    """Container for all layout-equivalence signals."""

    plain_procrustes_rmsd: float
    aut_procrustes_rmsd: float
    component_aligned_rmsd: float
    n_components: int
    anisotropic_rmsd: Optional[float]
    anisotropic_scale_x: Optional[float]
    anisotropic_scale_y: Optional[float]
    aut_group_size: int
    aut_capped: bool
    stress_dagua: float
    stress_reference: float
    stress_rel_delta: float
    crossings_dagua: int
    crossings_reference: int
    crossings_delta: int
    neighborhood_preservation_dagua: float
    neighborhood_preservation_reference: float
    neighborhood_preservation_delta: float
    dist_matrix_corr: float
    dist_matrix_rel_frob: float
    gram_eig_max_absdiff: float
    verdict: str

    def to_dict(self) -> dict[str, float | int | bool | str | None]:
        """Return a JSON-serializable dictionary.

        Returns
        -------
        dict[str, float | int | bool | str | None]
            Metric fields keyed by their public report names.
        """
        return asdict(self)


def as_float64_positions(
    positions: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
) -> np.ndarray:
    """Convert position-like input to a validated ``float64`` array.

    Parameters
    ----------
    positions : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Layout coordinates with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        CPU ``float64`` coordinates with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        Raised when the input is not a two-column coordinate matrix.
    """
    if isinstance(positions, torch.Tensor):
        array = positions.detach().cpu().numpy()
    else:
        array = np.asarray(positions)
    array64 = np.asarray(array, dtype=np.float64)
    if array64.ndim != 2 or array64.shape[1] != 2:
        raise ValueError(f"Expected positions with shape [N, 2], got {array64.shape}.")
    return array64


def as_edge_index(edge_index: np.ndarray | torch.Tensor | Sequence[Sequence[int]]) -> np.ndarray:
    """Convert edge-index-like input to a validated integer array.

    Parameters
    ----------
    edge_index : numpy.ndarray | torch.Tensor | Sequence[Sequence[int]]
        Edge index with shape ``[2, E]`` or edge list with shape ``[E, 2]``.

    Returns
    -------
    numpy.ndarray
        Integer edge index with shape ``[2, E]``.

    Raises
    ------
    ValueError
        Raised when the input cannot be interpreted as an edge index.
    """
    if isinstance(edge_index, torch.Tensor):
        array = edge_index.detach().cpu().numpy()
    else:
        array = np.asarray(edge_index)
    edge_array = np.asarray(array, dtype=np.int64)
    if edge_array.ndim != 2:
        raise ValueError(f"Expected edge_index with rank 2, got {edge_array.shape}.")
    if edge_array.shape[0] == 2:
        return edge_array
    if edge_array.shape[1] == 2:
        return edge_array.T.copy()
    raise ValueError(f"Expected edge_index shape [2, E] or [E, 2], got {edge_array.shape}.")


def procrustes_rmsd(positions_a: np.ndarray, positions_b: np.ndarray) -> float:
    """Compute the fidelity-report Procrustes RMSD.

    Parameters
    ----------
    positions_a : numpy.ndarray
        First coordinate matrix with shape ``[N, 2]``.
    positions_b : numpy.ndarray
        Second coordinate matrix with shape ``[N, 2]``.

    Returns
    -------
    float
        Scale-, translation-, rotation-, and reflection-invariant RMSD using
        the same unit-cloud normalization as ``scripts/fast_fidelity_report.py``.
    """
    a = as_float64_positions(positions_a)
    b = as_float64_positions(positions_b)
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    a_centered = a - a.mean(axis=0)
    b_centered = b - b.mean(axis=0)
    a_norm = float(np.linalg.norm(a_centered))
    b_norm = float(np.linalg.norm(b_centered))
    if a_norm < EPSILON or b_norm < EPSILON:
        return 0.0 if (a_norm < EPSILON and b_norm < EPSILON) else float(a_norm + b_norm)
    a_unit = a_centered / a_norm
    b_unit = b_centered / b_norm
    u_matrix, _singular_values, vt_matrix = np.linalg.svd(b_unit.T @ a_unit)
    rotation = u_matrix @ vt_matrix
    return float(np.linalg.norm((a_unit @ rotation.T) - b_unit))


def graph_automorphisms(
    edge_index: np.ndarray | torch.Tensor | Sequence[Sequence[int]],
    num_nodes: int,
    *,
    directed: bool = False,
    max_automorphisms: int = DEFAULT_MAX_AUTOMORPHISMS,
) -> tuple[list[list[int]], int, bool]:
    """Enumerate graph automorphisms with a safety cap.

    Parameters
    ----------
    edge_index : numpy.ndarray | torch.Tensor | Sequence[Sequence[int]]
        Graph edges with shape ``[2, E]`` or ``[E, 2]``.
    num_nodes : int
        Number of graph nodes.
    directed : bool, default=False
        Whether automorphisms should preserve edge direction.
    max_automorphisms : int, default=20000
        Maximum full-group permutations to enumerate.

    Returns
    -------
    tuple[list[list[int]], int, bool]
        Permutations to evaluate, estimated full group size, and whether the
        enumeration was capped.

    Notes
    -----
    The common fidelity holdouts are small symmetric graphs, so this function
    first asks igraph for the automorphism count and only fully enumerates with
    VF2 when the count is within ``max_automorphisms``.  If the group is larger,
    it evaluates identity plus BLISS generator permutations when Python igraph
    exposes them.  That fallback is intentionally conservative: it flags
    ``aut_capped=True`` instead of spending unbounded time generating a huge
    group.
    """
    try:
        import igraph as ig
    except ImportError:
        identity = list(range(num_nodes))
        return [identity], 1, True

    edges = _edge_tuples(edge_index)
    graph = ig.Graph(n=num_nodes, edges=edges, directed=directed)
    identity = list(range(num_nodes))
    try:
        group_size = int(graph.count_automorphisms_vf2())
    except Exception:
        group_size = max_automorphisms + 1
    if group_size <= max_automorphisms:
        try:
            permutations = [list(map(int, perm)) for perm in graph.get_automorphisms_vf2()]
            if not permutations:
                return [identity], 1, False
            return permutations, len(permutations), False
        except Exception:
            return [identity], group_size, True
    generators = _automorphism_generators(graph, num_nodes)
    candidates = _unique_permutations([identity, *generators])
    return candidates, group_size, True


def automorphism_aligned_procrustes(
    positions_dagua: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    positions_reference: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    edge_index: np.ndarray | torch.Tensor | Sequence[Sequence[int]],
    *,
    directed: bool = False,
    max_automorphisms: int = DEFAULT_MAX_AUTOMORPHISMS,
) -> dict[str, float | int | bool]:
    """Compute plain and automorphism-aligned Procrustes RMSD.

    Parameters
    ----------
    positions_dagua : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Reimplementation coordinates with shape ``[N, 2]``.
    positions_reference : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Reference coordinates with shape ``[N, 2]``.
    edge_index : numpy.ndarray | torch.Tensor | Sequence[Sequence[int]]
        Graph edges with shape ``[2, E]`` or ``[E, 2]``.
    directed : bool, default=False
        Whether automorphisms should preserve direction.
    max_automorphisms : int, default=20000
        Maximum full automorphism group size to enumerate.

    Returns
    -------
    dict[str, float | int | bool]
        ``plain_procrustes_rmsd``, ``aut_procrustes_rmsd``,
        ``aut_group_size``, and ``aut_capped``.
    """
    dagua = as_float64_positions(positions_dagua)
    reference = as_float64_positions(positions_reference)
    _validate_pair_shapes(dagua, reference)
    plain = procrustes_rmsd(dagua, reference)
    permutations, group_size, capped = graph_automorphisms(
        edge_index=edge_index,
        num_nodes=dagua.shape[0],
        directed=directed,
        max_automorphisms=max_automorphisms,
    )
    best = plain
    for permutation in permutations:
        permuted_reference = reference[np.asarray(permutation, dtype=np.int64)]
        best = min(best, procrustes_rmsd(dagua, permuted_reference))
    return {
        "plain_procrustes_rmsd": plain,
        "aut_procrustes_rmsd": best,
        "aut_group_size": group_size,
        "aut_capped": capped,
    }


def component_aligned_procrustes(
    positions_dagua: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    positions_reference: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    edge_index: np.ndarray | torch.Tensor | Sequence[Sequence[int]],
) -> dict[str, float | int]:
    """Align connected components independently under one global scale.

    Parameters
    ----------
    positions_dagua : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Reimplementation coordinates with shape ``[N, 2]``.
    positions_reference : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Reference coordinates with shape ``[N, 2]``.
    edge_index : numpy.ndarray | torch.Tensor | Sequence[Sequence[int]]
        Graph edges with shape ``[2, E]`` or ``[E, 2]``.

    Returns
    -------
    dict[str, float | int]
        ``component_aligned_rmsd`` and ``n_components``.  Connected graphs
        return the same normalized residual as ``procrustes_rmsd``.
    """
    dagua = as_float64_positions(positions_dagua)
    reference = as_float64_positions(positions_reference)
    _validate_pair_shapes(dagua, reference)
    components = _connected_components(edge_index, dagua.shape[0])
    if len(components) <= 1:
        return {
            "component_aligned_rmsd": procrustes_rmsd(dagua, reference),
            "n_components": len(components),
        }

    aligned_parts: list[np.ndarray] = []
    reference_parts: list[np.ndarray] = []
    for component in components:
        component_index = np.asarray(component, dtype=np.int64)
        dagua_centered = dagua[component_index] - dagua[component_index].mean(axis=0)
        reference_centered = reference[component_index] - reference[component_index].mean(axis=0)
        rotation = _orthogonal_alignment(dagua_centered, reference_centered)
        aligned_parts.append(dagua_centered @ rotation)
        reference_parts.append(reference_centered)

    aligned = np.vstack(aligned_parts)
    reference_centered_all = np.vstack(reference_parts)
    aligned_norm_sq = float(np.sum(np.square(aligned)))
    reference_norm = float(np.linalg.norm(reference_centered_all))
    if aligned_norm_sq < EPSILON or reference_norm < EPSILON:
        residual = 0.0 if aligned_norm_sq < EPSILON and reference_norm < EPSILON else reference_norm
    else:
        scale = float(np.sum(aligned * reference_centered_all) / aligned_norm_sq)
        residual = float(np.linalg.norm((scale * aligned) - reference_centered_all))
    return {
        "component_aligned_rmsd": residual / max(reference_norm, EPSILON),
        "n_components": len(components),
    }


def anisotropic_procrustes(
    positions_dagua: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    positions_reference: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
) -> dict[str, float]:
    """Align layouts with rotation/reflection/translation and per-axis scale.

    Parameters
    ----------
    positions_dagua : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Reimplementation coordinates with shape ``[N, 2]``.
    positions_reference : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Reference coordinates with shape ``[N, 2]``.

    Returns
    -------
    dict[str, float]
        ``anisotropic_rmsd`` plus fitted ``anisotropic_scale_x`` and
        ``anisotropic_scale_y``.  The residual uses the same reference-cloud
        normalization as ``procrustes_rmsd``.
    """
    dagua = as_float64_positions(positions_dagua)
    reference = as_float64_positions(positions_reference)
    _validate_pair_shapes(dagua, reference)
    dagua_centered = dagua - dagua.mean(axis=0)
    reference_centered = reference - reference.mean(axis=0)
    reference_norm = float(np.linalg.norm(reference_centered))
    if reference_norm < EPSILON:
        dagua_norm = float(np.linalg.norm(dagua_centered))
        residual = 0.0 if dagua_norm < EPSILON else dagua_norm
        return {
            "anisotropic_rmsd": residual,
            "anisotropic_scale_x": 1.0,
            "anisotropic_scale_y": 1.0,
        }

    candidates = [
        np.eye(2, dtype=np.float64),
        np.array([[-1.0, 0.0], [0.0, 1.0]], dtype=np.float64),
        np.array([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64),
        np.array([[0.0, 1.0], [1.0, 0.0]], dtype=np.float64),
        _orthogonal_alignment(dagua_centered, reference_centered),
    ]
    residual, scales = min(
        (
            _anisotropic_residual_for_rotation(dagua_centered, reference_centered, rotation)
            for rotation in candidates
        ),
        key=lambda result: result[0],
    )
    return {
        "anisotropic_rmsd": residual / reference_norm,
        "anisotropic_scale_x": float(scales[0]),
        "anisotropic_scale_y": float(scales[1]),
    }


def normalized_stress(
    positions: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    edge_index: np.ndarray | torch.Tensor | Sequence[Sequence[int]],
    *,
    all_pairs_distances: Optional[np.ndarray] = None,
) -> float:
    """Compute exact normalized graph-theoretic stress.

    Parameters
    ----------
    positions : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Layout coordinates with shape ``[N, 2]``.
    edge_index : numpy.ndarray | torch.Tensor | Sequence[Sequence[int]]
        Graph edges with shape ``[2, E]`` or ``[E, 2]``.
    all_pairs_distances : numpy.ndarray, optional
        Precomputed graph shortest-path distances with shape ``[N, N]``.

    Returns
    -------
    float
        ``sum(w_ij * (||x_i-x_j|| - d_ij)^2) / sum(w_ij * d_ij^2)`` over
        finite pairs with ``d_ij > 0`` and ``w_ij = 1 / d_ij^2``.
    """
    pos = as_float64_positions(positions)
    graph_dist = _shortest_paths(edge_index, pos.shape[0], all_pairs_distances)
    if pos.shape[0] < 2:
        return 0.0
    rows, cols = np.triu_indices(pos.shape[0], k=1)
    d_graph = graph_dist[rows, cols]
    valid = np.isfinite(d_graph) & (d_graph > 0.0)
    if not bool(valid.any()):
        return 0.0
    d_graph_valid = d_graph[valid]
    d_layout = np.linalg.norm(pos[rows[valid]] - pos[cols[valid]], axis=1)
    weights = 1.0 / np.square(d_graph_valid)
    numerator = float(np.sum(weights * np.square(d_layout - d_graph_valid)))
    denominator = float(np.sum(weights * np.square(d_graph_valid)))
    return numerator / max(denominator, EPSILON)


def stress_quality_equivalence(
    positions_dagua: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    positions_reference: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    edge_index: np.ndarray | torch.Tensor | Sequence[Sequence[int]],
    *,
    k: int = DEFAULT_NEIGHBORHOOD_K,
    all_pairs_distances: Optional[np.ndarray] = None,
) -> dict[str, float | int]:
    """Compute stress, crossing, and neighborhood preservation deltas.

    Parameters
    ----------
    positions_dagua : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Reimplementation coordinates with shape ``[N, 2]``.
    positions_reference : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Reference coordinates with shape ``[N, 2]``.
    edge_index : numpy.ndarray | torch.Tensor | Sequence[Sequence[int]]
        Graph edges with shape ``[2, E]`` or ``[E, 2]``.
    k : int, default=10
        Neighborhood size for graph/layout nearest-neighbor overlap.
    all_pairs_distances : numpy.ndarray, optional
        Precomputed graph shortest-path distances with shape ``[N, N]``.

    Returns
    -------
    dict[str, float | int]
        Exact quality metrics and absolute deltas.
    """
    dagua = as_float64_positions(positions_dagua)
    reference = as_float64_positions(positions_reference)
    _validate_pair_shapes(dagua, reference)
    edge_array = as_edge_index(edge_index)
    graph_dist = _shortest_paths(edge_array, dagua.shape[0], all_pairs_distances)
    stress_dagua = normalized_stress(dagua, edge_array, all_pairs_distances=graph_dist)
    stress_reference = normalized_stress(reference, edge_array, all_pairs_distances=graph_dist)
    stress_delta = abs(stress_dagua - stress_reference) / max(stress_reference, EPSILON)
    edge_tensor = torch.as_tensor(edge_array, dtype=torch.long)
    crossings_dagua = count_crossings(torch.as_tensor(dagua, dtype=torch.float64), edge_tensor)
    crossings_reference = count_crossings(
        torch.as_tensor(reference, dtype=torch.float64),
        edge_tensor,
    )
    neighborhood_dagua = neighborhood_preservation(dagua, graph_dist, k=k)
    neighborhood_reference = neighborhood_preservation(reference, graph_dist, k=k)
    return {
        "stress_dagua": stress_dagua,
        "stress_reference": stress_reference,
        "stress_rel_delta": stress_delta,
        "crossings_dagua": crossings_dagua,
        "crossings_reference": crossings_reference,
        "crossings_delta": abs(crossings_dagua - crossings_reference),
        "neighborhood_preservation_dagua": neighborhood_dagua,
        "neighborhood_preservation_reference": neighborhood_reference,
        "neighborhood_preservation_delta": abs(neighborhood_dagua - neighborhood_reference),
    }


def neighborhood_preservation(
    positions: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    all_pairs_distances: np.ndarray,
    *,
    k: int = DEFAULT_NEIGHBORHOOD_K,
) -> float:
    """Compare layout nearest neighbors against graph-distance neighbors.

    Parameters
    ----------
    positions : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Layout coordinates with shape ``[N, 2]``.
    all_pairs_distances : numpy.ndarray
        Graph shortest-path distances with shape ``[N, N]``.
    k : int, default=10
        Number of neighbors to compare per node.

    Returns
    -------
    float
        Mean per-node overlap fraction in ``[0, 1]``.
    """
    pos = as_float64_positions(positions)
    num_nodes = pos.shape[0]
    effective_k = min(k, max(num_nodes - 1, 0))
    if effective_k <= 0:
        return 1.0
    layout_distances = _pairwise_distances(pos)
    scores: list[float] = []
    for node in range(num_nodes):
        graph_row = np.asarray(all_pairs_distances[node], dtype=np.float64).copy()
        layout_row = layout_distances[node].copy()
        graph_row[node] = np.inf
        layout_row[node] = np.inf
        graph_candidates = np.where(np.isfinite(graph_row))[0]
        if graph_candidates.size == 0:
            continue
        graph_order = graph_candidates[np.argsort(graph_row[graph_candidates], kind="mergesort")]
        graph_neighbors = set(int(index) for index in graph_order[:effective_k])
        layout_neighbors = set(
            int(index) for index in np.argsort(layout_row, kind="mergesort")[: len(graph_neighbors)]
        )
        if graph_neighbors:
            scores.append(len(graph_neighbors & layout_neighbors) / len(graph_neighbors))
    return float(np.mean(scores)) if scores else 1.0


def spectrum_distance_diagnostic(
    positions_dagua: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    positions_reference: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
) -> dict[str, float]:
    """Compute rotation- and basis-invariant layout shape diagnostics.

    Parameters
    ----------
    positions_dagua : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Reimplementation coordinates with shape ``[N, 2]``.
    positions_reference : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Reference coordinates with shape ``[N, 2]``.

    Returns
    -------
    dict[str, float]
        Distance-matrix correlation, distance-matrix relative Frobenius error,
        and maximum absolute difference between normalized Gram eigenvalues.
    """
    dagua = as_float64_positions(positions_dagua)
    reference = as_float64_positions(positions_reference)
    _validate_pair_shapes(dagua, reference)
    dagua_dist = _pairwise_distances(dagua)
    reference_dist = _pairwise_distances(reference)
    rows, cols = np.triu_indices(dagua.shape[0], k=1)
    dagua_upper = dagua_dist[rows, cols]
    reference_upper = reference_dist[rows, cols]
    corr = _pearson_corr(dagua_upper, reference_upper)
    rel_frob = float(
        np.linalg.norm(dagua_dist - reference_dist)
        / max(float(np.linalg.norm(reference_dist)), EPSILON)
    )
    dagua_eig = _normalized_gram_eigenvalues(dagua)
    reference_eig = _normalized_gram_eigenvalues(reference)
    eig_diff = float(np.max(np.abs(dagua_eig - reference_eig))) if dagua_eig.size else 0.0
    return {
        "dist_matrix_corr": corr,
        "dist_matrix_rel_frob": rel_frob,
        "gram_eig_max_absdiff": eig_diff,
    }


def equivalence_verdict(
    metrics: dict[str, float | int | bool | str | None],
    *,
    engine_name: Optional[str] = None,
) -> str:
    """Classify a layout pair from its raw equivalence metrics.

    Parameters
    ----------
    metrics : dict[str, float | int | bool | str | None]
        Metric dictionary containing the public equivalence fields.
    engine_name : str, optional
        Benchmark engine name used for opt-in free-aspect invariance.

    Returns
    -------
    str
        ``"PRACTICALLY_EQUIVALENT"`` or ``"NOT_EQUIVALENT"``.
    """
    aut_ok = _required_float(metrics["aut_procrustes_rmsd"]) < 1.0e-3
    spectrum_ok = (
        _required_float(metrics["dist_matrix_corr"]) > 0.999
        and _required_float(metrics["gram_eig_max_absdiff"]) < 1.0e-3
    )
    quality_ok = (
        _required_float(metrics["stress_rel_delta"]) < 0.02
        and _required_float(metrics["neighborhood_preservation_delta"]) < 0.02
    )
    component_ok = _required_float(metrics["component_aligned_rmsd"]) < 1.0e-3
    anisotropic_value = metrics.get("anisotropic_rmsd")
    anisotropic_ok = (
        engine_name is not None
        and _engine_allows_free_aspect(engine_name)
        and anisotropic_value is not None
        and float(anisotropic_value) < 1.0e-3
    )
    return (
        "PRACTICALLY_EQUIVALENT"
        if (aut_ok or spectrum_ok or quality_ok or component_ok or anisotropic_ok)
        else "NOT_EQUIVALENT"
    )


def compute_equivalence_metrics(
    positions_dagua: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    positions_reference: np.ndarray | torch.Tensor | Sequence[Sequence[float]],
    edge_index: np.ndarray | torch.Tensor | Sequence[Sequence[int]],
    *,
    directed_automorphisms: bool = False,
    max_automorphisms: int = DEFAULT_MAX_AUTOMORPHISMS,
    neighborhood_k: int = DEFAULT_NEIGHBORHOOD_K,
    engine_name: Optional[str] = None,
) -> EquivalenceMetrics:
    """Compute all layout-equivalence metrics for one layout pair.

    Parameters
    ----------
    positions_dagua : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Reimplementation coordinates with shape ``[N, 2]``.
    positions_reference : numpy.ndarray | torch.Tensor | Sequence[Sequence[float]]
        Reference coordinates with shape ``[N, 2]``.
    edge_index : numpy.ndarray | torch.Tensor | Sequence[Sequence[int]]
        Graph edges with shape ``[2, E]`` or ``[E, 2]``.
    directed_automorphisms : bool, default=False
        Whether automorphism alignment should preserve edge direction.
    max_automorphisms : int, default=20000
        Maximum full automorphism group size to enumerate.
    neighborhood_k : int, default=10
        Neighborhood size for graph/layout nearest-neighbor overlap.
    engine_name : str, optional
        Benchmark engine name.  Anisotropic scaling is computed only for
        engines allowed by ``FREE_ASPECT_ENGINES``.

    Returns
    -------
    EquivalenceMetrics
        Structured metric result including the combined verdict.
    """
    dagua = as_float64_positions(positions_dagua)
    reference = as_float64_positions(positions_reference)
    _validate_pair_shapes(dagua, reference)
    edge_array = as_edge_index(edge_index)
    graph_dist = _shortest_paths(edge_array, dagua.shape[0], None)
    aut_metrics = automorphism_aligned_procrustes(
        dagua,
        reference,
        edge_array,
        directed=directed_automorphisms,
        max_automorphisms=max_automorphisms,
    )
    component_metrics = component_aligned_procrustes(dagua, reference, edge_array)
    if engine_name is not None and _engine_allows_free_aspect(engine_name):
        anisotropic_metrics: dict[str, float | None] = dict(
            anisotropic_procrustes(dagua, reference)
        )
    else:
        anisotropic_metrics = {
            "anisotropic_rmsd": None,
            "anisotropic_scale_x": None,
            "anisotropic_scale_y": None,
        }
    quality_metrics = stress_quality_equivalence(
        dagua,
        reference,
        edge_array,
        k=neighborhood_k,
        all_pairs_distances=graph_dist,
    )
    diagnostic_metrics = spectrum_distance_diagnostic(dagua, reference)
    combined: dict[str, float | int | bool | str | None] = {}
    combined.update(aut_metrics)
    combined.update(component_metrics)
    combined.update(anisotropic_metrics)
    combined.update(quality_metrics)
    combined.update(diagnostic_metrics)
    combined["verdict"] = equivalence_verdict(combined, engine_name=engine_name)
    return EquivalenceMetrics(
        plain_procrustes_rmsd=_required_float(combined["plain_procrustes_rmsd"]),
        aut_procrustes_rmsd=_required_float(combined["aut_procrustes_rmsd"]),
        component_aligned_rmsd=_required_float(combined["component_aligned_rmsd"]),
        n_components=_required_int(combined["n_components"]),
        anisotropic_rmsd=_optional_float(combined["anisotropic_rmsd"]),
        anisotropic_scale_x=_optional_float(combined["anisotropic_scale_x"]),
        anisotropic_scale_y=_optional_float(combined["anisotropic_scale_y"]),
        aut_group_size=_required_int(combined["aut_group_size"]),
        aut_capped=bool(combined["aut_capped"]),
        stress_dagua=_required_float(combined["stress_dagua"]),
        stress_reference=_required_float(combined["stress_reference"]),
        stress_rel_delta=_required_float(combined["stress_rel_delta"]),
        crossings_dagua=_required_int(combined["crossings_dagua"]),
        crossings_reference=_required_int(combined["crossings_reference"]),
        crossings_delta=_required_int(combined["crossings_delta"]),
        neighborhood_preservation_dagua=_required_float(
            combined["neighborhood_preservation_dagua"]
        ),
        neighborhood_preservation_reference=_required_float(
            combined["neighborhood_preservation_reference"]
        ),
        neighborhood_preservation_delta=_required_float(
            combined["neighborhood_preservation_delta"]
        ),
        dist_matrix_corr=_required_float(combined["dist_matrix_corr"]),
        dist_matrix_rel_frob=_required_float(combined["dist_matrix_rel_frob"]),
        gram_eig_max_absdiff=_required_float(combined["gram_eig_max_absdiff"]),
        verdict=str(combined["verdict"]),
    )


def _edge_tuples(
    edge_index: np.ndarray | torch.Tensor | Sequence[Sequence[int]],
) -> list[tuple[int, int]]:
    """Return an edge list suitable for igraph construction.

    Parameters
    ----------
    edge_index : numpy.ndarray | torch.Tensor | Sequence[Sequence[int]]
        Edge index with shape ``[2, E]`` or edge list with shape ``[E, 2]``.

    Returns
    -------
    list[tuple[int, int]]
        Integer ``(source, target)`` edge tuples.
    """
    edge_array = as_edge_index(edge_index)
    return [(int(src), int(tgt)) for src, tgt in edge_array.T.tolist() if int(src) != int(tgt)]


def _connected_components(
    edge_index: np.ndarray | torch.Tensor | Sequence[Sequence[int]],
    num_nodes: int,
) -> list[list[int]]:
    """Return undirected connected components.

    Parameters
    ----------
    edge_index : numpy.ndarray | torch.Tensor | Sequence[Sequence[int]]
        Graph edges with shape ``[2, E]`` or ``[E, 2]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Connected components as node-index lists, including isolates.
    """
    try:
        import igraph as ig
    except ImportError:
        return [list(range(num_nodes))]
    graph = ig.Graph(n=num_nodes, edges=_edge_tuples(edge_index), directed=False)
    return [list(map(int, component)) for component in graph.connected_components()]


def _orthogonal_alignment(source: np.ndarray, target: np.ndarray) -> np.ndarray:
    """Fit the reflection-allowing orthogonal map from source to target.

    Parameters
    ----------
    source : numpy.ndarray
        Centered source coordinates with shape ``[N, 2]``.
    target : numpy.ndarray
        Centered target coordinates with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Orthogonal matrix with shape ``[2, 2]``.
    """
    if source.size == 0 or float(np.linalg.norm(source)) < EPSILON:
        return np.eye(2, dtype=np.float64)
    u_matrix, _singular_values, vt_matrix = np.linalg.svd(source.T @ target)
    return u_matrix @ vt_matrix


def _anisotropic_residual_for_rotation(
    source: np.ndarray,
    target: np.ndarray,
    rotation: np.ndarray,
) -> tuple[float, np.ndarray]:
    """Fit per-axis scales for a fixed orthogonal alignment.

    Parameters
    ----------
    source : numpy.ndarray
        Centered source coordinates with shape ``[N, 2]``.
    target : numpy.ndarray
        Centered target coordinates with shape ``[N, 2]``.
    rotation : numpy.ndarray
        Orthogonal candidate matrix with shape ``[2, 2]``.

    Returns
    -------
    tuple[float, numpy.ndarray]
        Raw residual norm and fitted x/y scales.
    """
    aligned = source @ rotation
    scales = np.ones(2, dtype=np.float64)
    for axis in range(2):
        denominator = float(np.sum(np.square(aligned[:, axis])))
        if denominator >= EPSILON:
            scales[axis] = float(np.sum(aligned[:, axis] * target[:, axis]) / denominator)
    return float(np.linalg.norm((aligned * scales[None, :]) - target)), scales


def _engine_allows_free_aspect(engine_name: str) -> bool:
    """Return whether an engine owns free-aspect layout invariance.

    Parameters
    ----------
    engine_name : str
        Benchmark engine name.

    Returns
    -------
    bool
        ``True`` for exact allowlist entries and their named variants.

    Notes
    -----
    The allowlist stores conservative engine families, currently only
    ``classic_sugiyama``.  Benchmark variants append suffixes such as
    ``_default`` or ``_wide``, so those inherit the family-level justification.
    """
    return any(
        engine_name == allowed_engine or engine_name.startswith(f"{allowed_engine}_")
        for allowed_engine in FREE_ASPECT_ENGINES
    )


def _optional_float(value: float | int | bool | str | None) -> Optional[float]:
    """Convert optional metric values to optional floats.

    Parameters
    ----------
    value : float | int | bool | str | None
        Raw metric value.

    Returns
    -------
    float | None
        ``None`` when the metric is not applicable, otherwise ``float(value)``.
    """
    if value is None:
        return None
    return float(value)


def _required_float(value: float | int | bool | str | None) -> float:
    """Convert a required metric value to ``float``.

    Parameters
    ----------
    value : float | int | bool | str | None
        Raw metric value.

    Returns
    -------
    float
        Converted metric value.

    Raises
    ------
    ValueError
        Raised when a required metric is unexpectedly missing.
    """
    if value is None:
        raise ValueError("Required metric value is missing.")
    return float(value)


def _required_int(value: float | int | bool | str | None) -> int:
    """Convert a required metric value to ``int``.

    Parameters
    ----------
    value : float | int | bool | str | None
        Raw metric value.

    Returns
    -------
    int
        Converted metric value.

    Raises
    ------
    ValueError
        Raised when a required metric is unexpectedly missing.
    """
    if value is None:
        raise ValueError("Required metric value is missing.")
    return int(value)


def _automorphism_generators(graph: Any, num_nodes: int) -> list[list[int]]:
    """Extract BLISS automorphism generators from Python igraph when available.

    Parameters
    ----------
    graph : Any
        ``igraph.Graph`` instance.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Generator permutations, or an empty list if unavailable.
    """
    try:
        group = graph.automorphism_group()
    except Exception:
        return []
    raw_generators = getattr(group, "generators", None)
    if callable(raw_generators):
        raw_generators = raw_generators()
    if raw_generators is None and isinstance(group, Iterable):
        raw_generators = group
    generators: list[list[int]] = []
    if raw_generators is None:
        return generators
    for generator in raw_generators:
        permutation = list(map(int, generator))
        if len(permutation) == num_nodes:
            generators.append(permutation)
    return generators


def _unique_permutations(permutations: Sequence[Sequence[int]]) -> list[list[int]]:
    """Deduplicate permutations while preserving order.

    Parameters
    ----------
    permutations : Sequence[Sequence[int]]
        Candidate permutations.

    Returns
    -------
    list[list[int]]
        Unique permutations.
    """
    unique: list[list[int]] = []
    seen: set[tuple[int, ...]] = set()
    for permutation in permutations:
        key = tuple(int(index) for index in permutation)
        if key not in seen:
            seen.add(key)
            unique.append(list(key))
    return unique


def _shortest_paths(
    edge_index: np.ndarray | torch.Tensor | Sequence[Sequence[int]],
    num_nodes: int,
    all_pairs_distances: Optional[np.ndarray],
) -> np.ndarray:
    """Return a finite shortest-path distance matrix.

    Parameters
    ----------
    edge_index : numpy.ndarray | torch.Tensor | Sequence[Sequence[int]]
        Edge index with shape ``[2, E]`` or edge list with shape ``[E, 2]``.
    num_nodes : int
        Number of graph nodes.
    all_pairs_distances : numpy.ndarray, optional
        Precomputed graph distances.

    Returns
    -------
    numpy.ndarray
        ``float64`` graph distances with shape ``[N, N]``.
    """
    if all_pairs_distances is not None:
        return np.asarray(all_pairs_distances, dtype=np.float64)
    edge_tensor = torch.as_tensor(as_edge_index(edge_index), dtype=torch.long)
    return np.asarray(shortest_path_distances(edge_tensor, num_nodes), dtype=np.float64)


def _pairwise_distances(positions: np.ndarray) -> np.ndarray:
    """Compute a dense Euclidean distance matrix.

    Parameters
    ----------
    positions : numpy.ndarray
        Coordinate matrix with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Pairwise Euclidean distances with shape ``[N, N]``.
    """
    delta = positions[:, None, :] - positions[None, :, :]
    return np.linalg.norm(delta, axis=2)


def _pearson_corr(values_a: np.ndarray, values_b: np.ndarray) -> float:
    """Compute a robust Pearson correlation.

    Parameters
    ----------
    values_a : numpy.ndarray
        First vector.
    values_b : numpy.ndarray
        Second vector.

    Returns
    -------
    float
        Pearson correlation, with exact constant-vector matches reported as
        ``1.0``.
    """
    if values_a.size == 0:
        return 1.0
    a_std = float(np.std(values_a))
    b_std = float(np.std(values_b))
    if a_std < EPSILON or b_std < EPSILON:
        return 1.0 if np.allclose(values_a, values_b) else 0.0
    return float(np.corrcoef(values_a, values_b)[0, 1])


def _normalized_gram_eigenvalues(positions: np.ndarray) -> np.ndarray:
    """Return sorted eigenvalues of the unit-Frobenius centered Gram matrix.

    Parameters
    ----------
    positions : numpy.ndarray
        Coordinate matrix with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Sorted eigenvalues with shape ``[N]``.
    """
    centered = positions - positions.mean(axis=0)
    norm = float(np.linalg.norm(centered))
    if norm < EPSILON:
        centered = np.zeros_like(centered)
    else:
        centered = centered / norm
    gram = centered @ centered.T
    return np.sort(np.linalg.eigvalsh(gram))


def _validate_pair_shapes(positions_a: np.ndarray, positions_b: np.ndarray) -> None:
    """Validate that two layouts can be compared node-for-node.

    Parameters
    ----------
    positions_a : numpy.ndarray
        First coordinate matrix with shape ``[N, 2]``.
    positions_b : numpy.ndarray
        Second coordinate matrix with shape ``[N, 2]``.

    Returns
    -------
    None
        Returns ``None`` when shapes match.

    Raises
    ------
    ValueError
        Raised when layout shapes differ.
    """
    if positions_a.shape != positions_b.shape:
        raise ValueError(
            f"Layout shapes must match, got {positions_a.shape} and {positions_b.shape}."
        )
