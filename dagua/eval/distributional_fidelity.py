"""Statistical core for the r70 definitive distributional fidelity analysis."""

from __future__ import annotations

import hashlib
import math
from collections import deque
from typing import Any, Sequence

import numpy as np
from scipy import stats

from dagua.eval.equivalence_metrics import anisotropic_procrustes

EPSILON: float = 1.0e-12
NEAR_DETERMINISTIC_DISTANCE: float = 1.0e-3
SEED_NA_DISTANCE: float = 1.0e-2
EXACT_FALLBACK_CUTOFF: float = 1.0e-4
P_DIFF_PERMUTATIONS: int = 10_000
P_TRACK_PERMUTATIONS: int = 100_000
SPLIT_REPS: int = 1_000
CI_REPS: int = 2_000
STRESS_PAIR_LIMIT: int = 100_000


def pairwise_procrustes_matrix(
    layouts: Sequence[np.ndarray],
    free_aspect: bool = False,
) -> np.ndarray:
    """Compute the registered pairwise layout distance matrix.

    Parameters
    ----------
    layouts : Sequence[numpy.ndarray]
        Coordinate arrays with shape ``[N, 2]``.
    free_aspect : bool, default=False
        When ``True``, compute the symmetrized directed anisotropic residual
        used for free-aspect engines.  Otherwise use the complex Gram fast path
        plus exact-SVD fallback for distances below ``1e-4``.

    Returns
    -------
    numpy.ndarray
        Symmetric ``float64`` distance matrix with shape ``[M, M]``.
    """
    arrays = _as_layout_stack(layouts)
    if free_aspect:
        return _pairwise_anisotropic_matrix(arrays)
    return _pairwise_plain_procrustes_matrix(arrays)


def analyze_mode_a(
    D_layouts: Sequence[np.ndarray],
    R_layouts: Sequence[np.ndarray],
    rng: np.random.Generator,
    free_aspect: bool = False,
) -> dict[str, Any]:
    """Analyze a Mode A matched-seed two-sample combo.

    Parameters
    ----------
    D_layouts : Sequence[numpy.ndarray]
        Reimplementation layouts, one per matched seed, each shaped ``[N, 2]``.
    R_layouts : Sequence[numpy.ndarray]
        Reference layouts for the same seeds, each shaped ``[N, 2]``.
    rng : numpy.random.Generator
        Seeded generator owned by the caller.
    free_aspect : bool, default=False
        Whether comparative distances use symmetrized anisotropic residuals.

    Returns
    -------
    dict[str, Any]
        Registered Mode A statistics, diagnostics, and routing flags.
    """
    if len(D_layouts) != len(R_layouts):
        raise ValueError("Mode A requires matched D and R layout counts.")
    n = len(D_layouts)
    all_layouts = list(D_layouts) + list(R_layouts)
    distance = pairwise_procrustes_matrix(all_layouts, free_aspect=free_aspect)
    plain = pairwise_procrustes_matrix(all_layouts, free_aspect=False)
    w_d = distance[:n, :n]
    w_r = distance[n:, n:]
    b_mat = distance[:n, n:]
    plain_w_d = plain[:n, :n]
    plain_w_r = plain[n:, n:]
    stats_record = _mode_a_energy_stats(w_d, w_r, b_mat)
    stats_record.update(_mode_a_ci(w_d, w_r, b_mat, rng))
    stats_record["p_diff"] = _paired_swap_pvalue(distance, stats_record["E"], rng)
    stats_record.update(_split_calibration(distance, n, rng))
    stats_record.update(_tracking_stats(b_mat, rng))
    stats_record.update(_plain_guard_stats(plain_w_d, plain_w_r))
    stats_record["near_deterministic"] = (
        stats_record["plain_mean_W_D"] < NEAR_DETERMINISTIC_DISTANCE
        and stats_record["plain_mean_W_R"] < NEAR_DETERMINISTIC_DISTANCE
    )
    stats_record["one_sided_degenerate"] = (
        stats_record["plain_mean_W_D"] < NEAR_DETERMINISTIC_DISTANCE
    ) ^ (stats_record["plain_mean_W_R"] < NEAR_DETERMINISTIC_DISTANCE)
    stats_record["degenerate_heavy"] = _degenerate_fraction(D_layouts) > 0.10 or (
        _degenerate_fraction(R_layouts) > 0.10
    )
    stats_record["n"] = n
    stats_record["mode"] = "A"
    return stats_record


def analyze_mode_b(
    D_layouts: Sequence[np.ndarray],
    R_layout: np.ndarray,
    rng: np.random.Generator,
    free_aspect: bool = False,
) -> dict[str, Any]:
    """Analyze a Mode B one-sample deterministic-reference combo.

    Parameters
    ----------
    D_layouts : Sequence[numpy.ndarray]
        Reimplementation layouts with shape ``[N, 2]``.
    R_layout : numpy.ndarray
        Single deterministic reference layout with shape ``[N, 2]``.
    rng : numpy.random.Generator
        Seeded generator owned by the caller; accepted for API symmetry.
    free_aspect : bool, default=False
        Whether comparative distances use symmetrized anisotropic residuals.

    Returns
    -------
    dict[str, Any]
        Registered Mode B typicality diagnostics and routing flags.
    """
    del rng
    n = len(D_layouts)
    all_layouts = list(D_layouts) + [R_layout]
    distance = pairwise_procrustes_matrix(all_layouts, free_aspect=free_aspect)
    plain = pairwise_procrustes_matrix(all_layouts, free_aspect=False)
    w_d = distance[:n, :n]
    d_r = distance[:n, n]
    plain_w_d = plain[:n, :n]
    mean_plain_w_d = _offdiag_mean(plain_w_d)
    record: dict[str, Any] = {
        "mode": "B",
        "n": n,
        "mean_W_D": _offdiag_mean(w_d),
        "plain_mean_W_D": mean_plain_w_d,
        "d_R": float(np.mean(d_r)) if d_r.size else float("nan"),
        "d_R_nearest": float(np.min(d_r)) if d_r.size else float("nan"),
        "near_deterministic": mean_plain_w_d < NEAR_DETERMINISTIC_DISTANCE,
        "seed_na": True,
    }
    if record["near_deterministic"]:
        record.update(
            {
                "p_typ": float("nan"),
                "typicality_skipped": True,
                "ref_typical": record["d_R"] < NEAR_DETERMINISTIC_DISTANCE,
                "typicality_uninformative": False,
            }
        )
        return record
    scores_d = (np.sum(w_d, axis=1) + d_r) / max(n, 1)
    score_r = float(np.mean(d_r)) if d_r.size else float("nan")
    p_typ = (1.0 + float(np.count_nonzero(scores_d >= score_r))) / float(n + 1)
    percentile = float(np.mean(scores_d <= score_r)) if scores_d.size else float("nan")
    uninformative = mean_plain_w_d / math.sqrt(2.0) > 0.85
    record.update(
        {
            "scores_D": scores_d,
            "score_R": score_r,
            "score_R_percentile": percentile,
            "p_typ": p_typ,
            "typicality_skipped": False,
            "typicality_uninformative": uninformative,
            "ref_typical": (p_typ > 0.05) and not uninformative,
        }
    )
    return record


def prepare_graph_distances(edges: np.ndarray, n_nodes: int) -> np.ndarray:
    """Compute unweighted BFS shortest-path distances.

    Parameters
    ----------
    edges : numpy.ndarray
        Edge list with shape ``[E, 2]`` or edge index with shape ``[2, E]``.
    n_nodes : int
        Number of graph nodes.

    Returns
    -------
    numpy.ndarray
        ``float64`` distance matrix with shape ``[N, N]`` and ``inf`` between
        disconnected components.
    """
    edge_array = _as_edge_list(edges)
    adjacency: list[list[int]] = [[] for _ in range(n_nodes)]
    for src, dst in edge_array:
        if 0 <= src < n_nodes and 0 <= dst < n_nodes and src != dst:
            adjacency[src].append(dst)
            adjacency[dst].append(src)
    distances = np.full((n_nodes, n_nodes), np.inf, dtype=np.float64)
    for start in range(n_nodes):
        distances[start, start] = 0.0
        queue: deque[int] = deque([start])
        while queue:
            node = queue.popleft()
            next_distance = distances[start, node] + 1.0
            for neighbor in adjacency[node]:
                if not np.isfinite(distances[start, neighbor]):
                    distances[start, neighbor] = next_distance
                    queue.append(neighbor)
    return distances


def sample_pairs(dist_matrix: np.ndarray, graph_name: str) -> np.ndarray:
    """Return finite graph-distance node pairs for stress evaluation.

    Parameters
    ----------
    dist_matrix : numpy.ndarray
        Graph distance matrix with shape ``[N, N]``.
    graph_name : str
        Graph identifier used for deterministic pair sampling.

    Returns
    -------
    numpy.ndarray
        Integer pair array with shape ``[P, 2]``.
    """
    rows, cols = np.triu_indices(dist_matrix.shape[0], k=1)
    finite = np.isfinite(dist_matrix[rows, cols]) & (dist_matrix[rows, cols] > 0.0)
    pairs = np.column_stack((rows[finite], cols[finite])).astype(np.int64, copy=False)
    if pairs.shape[0] <= STRESS_PAIR_LIMIT:
        return pairs
    rng = _rng_from_purpose(f"r70::stressP::{graph_name}")
    selected = rng.choice(pairs.shape[0], size=STRESS_PAIR_LIMIT, replace=False)
    return pairs[np.sort(selected)]


def stress_per_layout(layout: np.ndarray, pairs: np.ndarray, dists: np.ndarray) -> float:
    """Compute normalized stress with closed-form optimal scale.

    Parameters
    ----------
    layout : numpy.ndarray
        Coordinate array with shape ``[N, 2]``.
    pairs : numpy.ndarray
        Pair index array with shape ``[P, 2]``.
    dists : numpy.ndarray
        Graph distance matrix with shape ``[N, N]``.

    Returns
    -------
    float
        Scale-optimized normalized stress over the supplied finite pairs.
    """
    if pairs.size == 0:
        return 0.0
    pos = _as_layout(layout)
    graph_d = dists[pairs[:, 0], pairs[:, 1]].astype(np.float64, copy=False)
    layout_d = np.linalg.norm(pos[pairs[:, 0]] - pos[pairs[:, 1]], axis=1)
    denominator_alpha = float(np.sum(np.square(layout_d)))
    alpha = 0.0
    if denominator_alpha >= EPSILON:
        alpha = float(np.sum(layout_d * graph_d) / denominator_alpha)
    numerator = float(np.sum(np.square((alpha * layout_d) - graph_d)))
    denominator = float(np.sum(np.square(graph_d)))
    return numerator / max(denominator, EPSILON)


def paired_tost(diffs: np.ndarray, margin: float) -> dict[str, Any]:
    """Run a paired-difference TOST against zero.

    Parameters
    ----------
    diffs : numpy.ndarray
        Paired differences.
    margin : float
        Equivalence margin.  Callers apply the registered floor.

    Returns
    -------
    dict[str, Any]
        T-based p-value, Wilcoxon annotation, and degenerate branch metadata.
    """
    values = np.asarray(diffs, dtype=np.float64)
    return _tost_values(values, 0.0, margin, paired=True)


def one_sample_tost(values: np.ndarray, target: float, margin: float) -> dict[str, Any]:
    """Run a one-sample TOST against a target value.

    Parameters
    ----------
    values : numpy.ndarray
        Observed sample values.
    target : float
        Registered comparison target.
    margin : float
        Equivalence margin.  Callers apply the registered floor.

    Returns
    -------
    dict[str, Any]
        T-based p-value, Wilcoxon annotation, and degenerate branch metadata.
    """
    array = np.asarray(values, dtype=np.float64)
    return _tost_values(array, target, margin, paired=False)


def bh_fdr(pvals: Sequence[float]) -> np.ndarray:
    """Compute standard Benjamini-Hochberg q-values.

    Parameters
    ----------
    pvals : Sequence[float]
        Raw p-values.

    Returns
    -------
    numpy.ndarray
        FDR-adjusted q-values in the original order.
    """
    p_array = np.asarray(pvals, dtype=np.float64)
    qvals = np.full_like(p_array, np.nan, dtype=np.float64)
    finite = np.isfinite(p_array)
    if not bool(finite.any()):
        return qvals
    finite_p = p_array[finite]
    order = np.argsort(finite_p, kind="mergesort")
    ranked = finite_p[order]
    ranks = np.arange(1, ranked.size + 1, dtype=np.float64)
    adjusted = ranked * ranked.size / ranks
    adjusted = np.minimum.accumulate(adjusted[::-1])[::-1]
    adjusted = np.clip(adjusted, 0.0, 1.0)
    out = np.empty_like(finite_p)
    out[order] = adjusted
    qvals[finite] = out
    return qvals


def assign_rung(record: dict[str, Any]) -> tuple[str, list[str]]:
    """Assign the registered per-combo verdict rung.

    Parameters
    ----------
    record : dict[str, Any]
        Analysis record containing raw flags and report-stage q-values.

    Returns
    -------
    tuple[str, list[str]]
        Rung label and annotations.
    """
    annotations = _record_annotations(record)
    if record.get("bit_exact", False):
        return "0", annotations
    if record.get("insufficient_data", False):
        return "INSUFFICIENT_DATA", annotations
    mode = str(record.get("mode", ""))
    quality = bool(record.get("quality_equivalent", False) or _q_lt(record.get("q_tost"), 0.05))
    quality_identical = bool(
        record.get("quality_identical", False) or _q_lt(record.get("q_battery"), 0.05)
    )
    if mode == "A":
        # Tracking must be earned from measured statistics. A bare boolean
        # "seed_tracking" field has no producer in this codebase and acted as
        # a silent benign-rung bypass in hand-compacted control files
        # (r78 gate_3_negative audit); it is deliberately ignored.
        tracking = (
            _q_lt(record.get("q_track"), 0.05) and float(record.get("track_ratio", math.inf)) <= 0.5
        )
        dist_equiv = bool(record.get("dist_equivalent", False))
        near_det = bool(record.get("near_deterministic", False))
        if near_det:
            diag = float(record.get("mean_diag_B", math.inf))
            if diag < NEAR_DETERMINISTIC_DISTANCE:
                return "1", annotations
            if quality_identical:
                return "3Q", annotations
            return ("3", annotations) if quality else ("4", annotations)
        if bool(record.get("one_sided_degenerate", False)):
            if quality_identical:
                return "3Q", annotations
            return ("3", annotations) if quality else ("4", annotations)
        if dist_equiv and tracking:
            return "1", annotations
        if tracking and not dist_equiv and "TRACKING_BUT_SHIFTED" not in annotations:
            annotations.append("TRACKING_BUT_SHIFTED")
        if dist_equiv:
            return "2", annotations
        if quality_identical:
            return "3Q", annotations
        return ("3", annotations) if quality else ("4", annotations)
    if mode == "B":
        if bool(record.get("near_deterministic", False)):
            return (
                ("2'", annotations)
                if float(record.get("d_R", math.inf)) < 1.0e-3
                else (
                    ("3Q", annotations)
                    if quality_identical
                    else (("3", annotations) if quality else ("4", annotations))
                )
            )
        # Typicality on a dispersed reimplementation cloud is a WEAK verdict:
        # sensitivity is high (39/39 on the gate_2 positive controls) but a
        # wrong-algorithm reference passes ~10% of the time (gate_3 negative
        # controls, r78 audit F4), because with high seed-variance any
        # sensible layout of the same graph scores "typical". It therefore
        # earns the distinct weak rung "2'w", which is never counted as a
        # primary/benign rung; the strong "2'" above requires the
        # near-deterministic positional d_R gate.
        informative = not bool(record.get("typicality_uninformative", False))
        ref_typical = bool(record.get("ref_typical", False)) or (
            informative and float(record.get("p_typ", 0.0)) > 0.05
        )
        if informative and ref_typical:
            return "2'w", annotations
        if quality_identical:
            return "3Q", annotations
        return ("3", annotations) if quality else ("4", annotations)
    return "4", annotations


def run_oc_simulation(rng: np.random.Generator) -> dict[str, Any]:
    """Run the registered q95-rule operating-characteristics simulation.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator supplied by the caller.

    Returns
    -------
    dict[str, Any]
        Pass-rate table for shift/spread/n simulation cells.
    """
    results: dict[str, Any] = {}
    for n in (30, 60, 100):
        for shift in (0.0, 0.25, 0.5, 1.0):
            for spread in (0.0, 0.25, 0.5, 1.0):
                passes = 0
                for _rep in range(500):
                    reference = _synthetic_clouds(rng, n, 40, shift=0.0, spread=0.0)
                    dagua = _synthetic_clouds(rng, n, 40, shift=shift, spread=spread)
                    distance = pairwise_procrustes_matrix(list(dagua) + list(reference))
                    passes += int(_split_calibration(distance, n, rng)["dist_equivalent"])
                key = f"n={n}|shift={shift}|spread={spread}"
                results[key] = {"passes": passes, "reps": 500, "pass_rate": passes / 500.0}
    return results


def _as_layout(layout: np.ndarray) -> np.ndarray:
    """Validate and convert one layout array.

    Parameters
    ----------
    layout : numpy.ndarray
        Position-like array.

    Returns
    -------
    numpy.ndarray
        ``float64`` coordinate array with shape ``[N, 2]``.
    """
    array = np.asarray(layout, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 2:
        raise ValueError(f"Expected layout shape [N, 2], got {array.shape}.")
    if not np.all(np.isfinite(array)):
        raise ValueError("Layout contains non-finite coordinates.")
    return array


def _as_layout_stack(layouts: Sequence[np.ndarray]) -> np.ndarray:
    """Convert a layout sequence to a homogeneous stack.

    Parameters
    ----------
    layouts : Sequence[numpy.ndarray]
        Layout arrays with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Stacked array with shape ``[M, N, 2]``.
    """
    arrays = [_as_layout(layout) for layout in layouts]
    if not arrays:
        return np.empty((0, 0, 2), dtype=np.float64)
    first_shape = arrays[0].shape
    if any(array.shape != first_shape for array in arrays):
        raise ValueError("All layouts must have the same shape.")
    return np.stack(arrays, axis=0)


def _pairwise_plain_procrustes_matrix(arrays: np.ndarray) -> np.ndarray:
    """Compute the plain pairwise Procrustes matrix.

    Parameters
    ----------
    arrays : numpy.ndarray
        Stacked layouts with shape ``[M, N, 2]``.

    Returns
    -------
    numpy.ndarray
        Pairwise plain-Procrustes distances with shape ``[M, M]``.
    """
    m = arrays.shape[0]
    if m == 0:
        return np.empty((0, 0), dtype=np.float64)
    centered = arrays - np.mean(arrays, axis=1, keepdims=True)
    norms = np.linalg.norm(centered.reshape(m, -1), axis=1)
    degenerate = norms < EPSILON
    unit = np.zeros_like(centered)
    unit[~degenerate] = centered[~degenerate] / norms[~degenerate, None, None]
    zed = unit[:, :, 0] + (1j * unit[:, :, 1])
    inner_rotation = zed @ zed.conj().T
    inner_reflection = zed @ zed.T
    best = np.maximum(np.abs(inner_rotation), np.abs(inner_reflection))
    matrix = np.sqrt(np.maximum(0.0, 2.0 - (2.0 * np.clip(best.real, 0.0, 1.0))))
    if bool(degenerate.any()):
        norm_sums = norms[:, None] + norms[None, :]
        both = degenerate[:, None] & degenerate[None, :]
        either = degenerate[:, None] | degenerate[None, :]
        matrix[either] = norm_sums[either]
        matrix[both] = 0.0
    fallback = np.argwhere(matrix < EXACT_FALLBACK_CUTOFF)
    for i, j in fallback:
        if i <= j:
            exact = _procrustes_rmsd_exact(arrays[i], arrays[j])
            matrix[i, j] = exact
            matrix[j, i] = exact
    np.fill_diagonal(matrix, 0.0)
    return matrix.astype(np.float64, copy=False)


def _pairwise_anisotropic_matrix(arrays: np.ndarray) -> np.ndarray:
    """Compute symmetrized anisotropic pairwise distances.

    Parameters
    ----------
    arrays : numpy.ndarray
        Stacked layouts with shape ``[M, N, 2]``.

    Returns
    -------
    numpy.ndarray
        Symmetric anisotropic distance matrix with shape ``[M, M]``.
    """
    m = arrays.shape[0]
    matrix = np.zeros((m, m), dtype=np.float64)
    for i in range(m):
        for j in range(i + 1, m):
            forward = float(anisotropic_procrustes(arrays[i], arrays[j])["anisotropic_rmsd"])
            backward = float(anisotropic_procrustes(arrays[j], arrays[i])["anisotropic_rmsd"])
            value = 0.5 * (forward + backward)
            matrix[i, j] = value
            matrix[j, i] = value
    return matrix


def _procrustes_rmsd_exact(a_array: np.ndarray, b_array: np.ndarray) -> float:
    """Compute the exact project-wide Procrustes RMSD.

    Parameters
    ----------
    a_array : numpy.ndarray
        First coordinate array with shape ``[N, 2]``.
    b_array : numpy.ndarray
        Second coordinate array with shape ``[N, 2]``.

    Returns
    -------
    float
        Exact O(2)-aligned residual.
    """
    if a_array.shape != b_array.shape or a_array.size == 0:
        return float("nan")
    a_centered = a_array - a_array.mean(axis=0)
    b_centered = b_array - b_array.mean(axis=0)
    a_norm = float(np.linalg.norm(a_centered))
    b_norm = float(np.linalg.norm(b_centered))
    if a_norm < EPSILON or b_norm < EPSILON:
        return 0.0 if (a_norm < EPSILON and b_norm < EPSILON) else float(a_norm + b_norm)
    a_unit = a_centered / a_norm
    b_unit = b_centered / b_norm
    u_matrix, _singular_values, vt_matrix = np.linalg.svd(b_unit.T @ a_unit)
    rotation = u_matrix @ vt_matrix
    return float(np.linalg.norm((a_unit @ rotation.T) - b_unit))


def _offdiag_mean(matrix: np.ndarray) -> float:
    """Return the off-diagonal mean of a square matrix.

    Parameters
    ----------
    matrix : numpy.ndarray
        Square numeric matrix.

    Returns
    -------
    float
        Mean over ``i != j`` or ``0`` for fewer than two rows.
    """
    n = matrix.shape[0]
    if n < 2:
        return 0.0
    return float((np.sum(matrix) - np.trace(matrix)) / float(n * (n - 1)))


def _mode_a_energy_stats(w_d: np.ndarray, w_r: np.ndarray, b_mat: np.ndarray) -> dict[str, float]:
    """Compute Mode A energy and dispersion statistics.

    Parameters
    ----------
    w_d : numpy.ndarray
        Reimplementation within-distance block.
    w_r : numpy.ndarray
        Reference within-distance block.
    b_mat : numpy.ndarray
        Cross-distance block.

    Returns
    -------
    dict[str, float]
        ``E``, ``e_rel``, ``disp``, and block means.
    """
    n = b_mat.shape[0]
    mean_w_d = _offdiag_mean(w_d)
    mean_w_r = _offdiag_mean(w_r)
    offdiag_mask = ~np.eye(n, dtype=bool)
    mean_b_offdiag = float(np.mean(b_mat[offdiag_mask])) if n > 1 else float("nan")
    mean_diag = float(np.mean(np.diag(b_mat))) if n else float("nan")
    energy = (2.0 * mean_b_offdiag) - mean_w_d - mean_w_r
    denominator = 0.5 * (mean_w_d + mean_w_r)
    e_rel = float("nan") if denominator < EPSILON else energy / denominator
    disp = float("nan") if mean_w_r < EPSILON else mean_w_d / mean_w_r
    return {
        "E": float(energy),
        "e_rel": float(e_rel),
        "disp": float(disp),
        "mean_W_D": mean_w_d,
        "mean_W_R": mean_w_r,
        "mean_B_offdiag": mean_b_offdiag,
        "mean_diag_B": mean_diag,
        "degenerate": denominator < EPSILON,
    }


def _mode_a_ci(
    w_d: np.ndarray,
    w_r: np.ndarray,
    b_mat: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float]:
    """Compute m-out-of-n recentered percentile CIs.

    Parameters
    ----------
    w_d : numpy.ndarray
        Reimplementation within-distance block.
    w_r : numpy.ndarray
        Reference within-distance block.
    b_mat : numpy.ndarray
        Cross-distance block.
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    dict[str, float]
        CI endpoints for ``E`` and ``e_rel``.
    """
    n = w_d.shape[0]
    m_sub = n // 2
    full = _mode_a_energy_stats(w_d, w_r, b_mat)
    if m_sub < 2:
        return {
            "E_ci_low": float("nan"),
            "E_ci_high": float("nan"),
            "e_rel_ci_low": float("nan"),
            "e_rel_ci_high": float("nan"),
        }
    estimates = np.empty((CI_REPS, 2), dtype=np.float64)
    for rep in range(CI_REPS):
        idx = rng.choice(n, size=m_sub, replace=False)
        estimate = _mode_a_energy_stats(
            w_d[np.ix_(idx, idx)],
            w_r[np.ix_(idx, idx)],
            b_mat[np.ix_(idx, idx)],
        )
        estimates[rep, 0] = estimate["E"]
        estimates[rep, 1] = estimate["e_rel"]
    scale = math.sqrt(m_sub / n)
    e_quantiles = np.nanpercentile(estimates[:, 0], [2.5, 97.5])
    erel_quantiles = np.nanpercentile(estimates[:, 1], [2.5, 97.5])
    return {
        "E_ci_low": float(full["E"] - ((full["E"] - e_quantiles[0]) * scale)),
        "E_ci_high": float(full["E"] + ((e_quantiles[1] - full["E"]) * scale)),
        "e_rel_ci_low": float(full["e_rel"] - ((full["e_rel"] - erel_quantiles[0]) * scale)),
        "e_rel_ci_high": float(full["e_rel"] + ((erel_quantiles[1] - full["e_rel"]) * scale)),
    }


def _paired_swap_pvalue(matrix: np.ndarray, observed: float, rng: np.random.Generator) -> float:
    """Compute the paired-swap Mode A permutation p-value.

    Parameters
    ----------
    matrix : numpy.ndarray
        Combined ``[D, R]`` distance matrix with shape ``[2N, 2N]``.
    observed : float
        Observed diagonal-excluded energy statistic.
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    float
        One-sided paired-swap p-value.
    """
    n = matrix.shape[0] // 2
    swaps = rng.random((P_DIFF_PERMUTATIONS, n)) < 0.5
    base = np.arange(n, dtype=np.int64)
    a_idx = np.where(swaps, base + n, base)
    b_idx = np.where(swaps, base, base + n)
    offdiag = ~np.eye(n, dtype=bool)
    w_a = matrix[a_idx[:, :, None], a_idx[:, None, :]][:, offdiag].mean(axis=1)
    w_b = matrix[b_idx[:, :, None], b_idx[:, None, :]][:, offdiag].mean(axis=1)
    cross = matrix[a_idx[:, :, None], b_idx[:, None, :]][:, offdiag].mean(axis=1)
    permuted = (2.0 * cross) - w_a - w_b
    return float((1 + np.count_nonzero(permuted >= observed)) / (P_DIFF_PERMUTATIONS + 1))


def _split_calibration(matrix: np.ndarray, n: int, rng: np.random.Generator) -> dict[str, Any]:
    """Compute q95 split-calibration equivalence statistics.

    Parameters
    ----------
    matrix : numpy.ndarray
        Combined ``[D, R]`` distance matrix with shape ``[2N, 2N]``.
    n : int
        Number of matched seeds.
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    dict[str, Any]
        Split-calibration arrays, percentiles, verdict, and fixed-m sensitivity.
    """
    half = n // 2
    e_self = np.empty(SPLIT_REPS, dtype=np.float64)
    e_cross = np.empty(SPLIT_REPS, dtype=np.float64)
    e_self_15 = np.empty(SPLIT_REPS, dtype=np.float64)
    e_cross_15 = np.empty(SPLIT_REPS, dtype=np.float64)
    for rep in range(SPLIT_REPS):
        perm = rng.permutation(n)
        a = perm[:half]
        b = perm[half : half * 2]
        e_self[rep] = _energy_two_sample(
            matrix[np.ix_(n + a, n + a)],
            matrix[np.ix_(n + b, n + b)],
            matrix[np.ix_(n + a, n + b)],
        )
        e_cross[rep] = _energy_two_sample(
            matrix[np.ix_(a, a)],
            matrix[np.ix_(n + b, n + b)],
            matrix[np.ix_(a, n + b)],
        )
        if n >= 30:
            a15 = a[:15]
            b15 = b[:15]
            e_self_15[rep] = _energy_two_sample(
                matrix[np.ix_(n + a15, n + a15)],
                matrix[np.ix_(n + b15, n + b15)],
                matrix[np.ix_(n + a15, n + b15)],
            )
            e_cross_15[rep] = _energy_two_sample(
                matrix[np.ix_(a15, a15)],
                matrix[np.ix_(n + b15, n + b15)],
                matrix[np.ix_(a15, n + b15)],
            )
        else:
            e_self_15[rep] = np.nan
            e_cross_15[rep] = np.nan
    q95_self = float(np.quantile(e_self, 0.95))
    med_cross = float(np.median(e_cross))
    equiv_percentile = float(np.mean(e_self <= med_cross))
    return {
        "E_self": e_self,
        "E_cross": e_cross,
        "E_self_q95": q95_self,
        "E_cross_median": med_cross,
        "E_cross_q95": float(np.quantile(e_cross, 0.95)),
        "dist_equivalent": med_cross <= q95_self,
        "equiv_percentile": equiv_percentile,
        "fixed_m15_E_self_q95": float(np.nanquantile(e_self_15, 0.95)),
        "fixed_m15_E_cross_median": float(np.nanmedian(e_cross_15)),
        "fixed_m15_dist_equivalent": bool(
            np.nanmedian(e_cross_15) <= np.nanquantile(e_self_15, 0.95)
        ),
    }


def _energy_two_sample(w_a: np.ndarray, w_b: np.ndarray, cross: np.ndarray) -> float:
    """Compute diagonal-excluded two-sample energy distance.

    Parameters
    ----------
    w_a : numpy.ndarray
        First within-distance block.
    w_b : numpy.ndarray
        Second within-distance block.
    cross : numpy.ndarray
        Cross-distance block.

    Returns
    -------
    float
        U-statistic energy distance.
    """
    return float((2.0 * np.mean(cross)) - _offdiag_mean(w_a) - _offdiag_mean(w_b))


def _tracking_stats(b_mat: np.ndarray, rng: np.random.Generator) -> dict[str, Any]:
    """Compute seed-specific tracking statistics.

    Parameters
    ----------
    b_mat : numpy.ndarray
        Cross-distance matrix with matched seeds on the diagonal.
    rng : numpy.random.Generator
        Seeded generator.

    Returns
    -------
    dict[str, Any]
        Tracking p-value, ratio, recovery, and rank diagnostics.
    """
    n = b_mat.shape[0]
    diag = np.diag(b_mat)
    offdiag = b_mat[~np.eye(n, dtype=bool)]
    observed = float(np.mean(diag))
    perms = np.argsort(rng.random((P_TRACK_PERMUTATIONS, n)), axis=1)
    sampled = b_mat[np.arange(n, dtype=np.int64)[None, :], perms].mean(axis=1)
    p_track = float((1 + np.count_nonzero(sampled <= observed)) / (P_TRACK_PERMUTATIONS + 1))
    ranks = np.empty(n, dtype=np.float64)
    for row in range(n):
        less = np.count_nonzero(b_mat[row] < b_mat[row, row])
        equal = np.count_nonzero(b_mat[row] == b_mat[row, row])
        ranks[row] = less + ((equal + 1.0) / 2.0)
    normalized = (ranks - 1.0) / max(n - 1.0, 1.0)
    return {
        "p_track": p_track,
        "track_ratio": observed / max(float(np.mean(offdiag)), EPSILON),
        "recovery_at_1": float(np.mean(np.isclose(ranks, 1.0))),
        "twin_rank_median": float(np.median(normalized)),
        "twin_rank_deciles": np.quantile(normalized, np.linspace(0.1, 0.9, 9)),
    }


def _plain_guard_stats(plain_w_d: np.ndarray, plain_w_r: np.ndarray) -> dict[str, float]:
    """Return plain-Procrustes block means used by absolute guards.

    Parameters
    ----------
    plain_w_d : numpy.ndarray
        Plain D within-distance block.
    plain_w_r : numpy.ndarray
        Plain R within-distance block.

    Returns
    -------
    dict[str, float]
        Plain block means.
    """
    return {"plain_mean_W_D": _offdiag_mean(plain_w_d), "plain_mean_W_R": _offdiag_mean(plain_w_r)}


def _degenerate_fraction(layouts: Sequence[np.ndarray]) -> float:
    """Return the fraction of degenerate centered layouts.

    Parameters
    ----------
    layouts : Sequence[numpy.ndarray]
        Layout arrays.

    Returns
    -------
    float
        Fraction with centered Frobenius norm below ``1e-12``.
    """
    if not layouts:
        return 0.0
    flags = [
        float(np.linalg.norm(_as_layout(layout) - np.mean(layout, axis=0)) < EPSILON)
        for layout in layouts
    ]
    return float(np.mean(flags))


def _as_edge_list(edges: np.ndarray) -> list[tuple[int, int]]:
    """Normalize edge input to a Python edge list.

    Parameters
    ----------
    edges : numpy.ndarray
        Edge list ``[E, 2]`` or edge index ``[2, E]``.

    Returns
    -------
    list[tuple[int, int]]
        Integer edge tuples.
    """
    array = np.asarray(edges, dtype=np.int64)
    if array.ndim != 2:
        raise ValueError("Edges must be rank-2.")
    if array.shape[0] == 2 and array.shape[1] != 2:
        array = array.T
    if array.shape[1] != 2:
        raise ValueError(f"Expected edge shape [E, 2] or [2, E], got {array.shape}.")
    return [(int(src), int(dst)) for src, dst in array.tolist()]


def _tost_values(
    values: np.ndarray,
    target: float,
    margin: float,
    *,
    paired: bool,
) -> dict[str, Any]:
    """Compute shared TOST values for paired and one-sample paths.

    Parameters
    ----------
    values : numpy.ndarray
        Values or paired differences.
    target : float
        Null target.
    margin : float
        Positive equivalence margin.
    paired : bool
        Whether the values came from a paired-difference test.

    Returns
    -------
    dict[str, Any]
        TOST result dictionary.
    """
    centered = np.asarray(values, dtype=np.float64) - target
    n = centered.size
    sd = float(np.std(centered, ddof=1)) if n > 1 else 0.0
    mean = float(np.mean(centered)) if n else float("nan")
    if n == 0:
        return {
            "p_tost": float("nan"),
            "wilcoxon_p_tost": float("nan"),
            "degenerate_sd": False,
            "equivalent_direct": False,
        }
    if sd < EPSILON:
        equivalent = abs(mean) <= margin
        return {
            "p_tost": 0.0 if equivalent else 1.0,
            "wilcoxon_p_tost": float("nan"),
            "degenerate_sd": True,
            "equivalent_direct": equivalent,
        }
    se = sd / math.sqrt(n)
    t_low = (mean + margin) / se
    t_high = (mean - margin) / se
    p_low = float(stats.t.sf(t_low, df=n - 1))
    p_high = float(stats.t.cdf(t_high, df=n - 1))
    wilcoxon = _wilcoxon_tost(centered, margin)
    return {
        "p_tost": max(p_low, p_high),
        "wilcoxon_p_tost": wilcoxon,
        "degenerate_sd": False,
        "equivalent_direct": False,
        "paired": paired,
    }


def _wilcoxon_tost(centered: np.ndarray, margin: float) -> float:
    """Compute a Wilcoxon signed-rank TOST annotation.

    Parameters
    ----------
    centered : numpy.ndarray
        Centered values.
    margin : float
        Positive equivalence margin.

    Returns
    -------
    float
        Maximum one-sided Wilcoxon p-value, or ``nan`` if undefined.
    """
    try:
        p_low = float(stats.wilcoxon(centered + margin, alternative="greater").pvalue)
        p_high = float(stats.wilcoxon(centered - margin, alternative="less").pvalue)
    except ValueError:
        return float("nan")
    return max(p_low, p_high)


def _record_annotations(record: dict[str, Any]) -> list[str]:
    """Extract registered annotations from a record.

    Parameters
    ----------
    record : dict[str, Any]
        Analysis record.

    Returns
    -------
    list[str]
        Annotation labels.
    """
    annotations: list[str] = []
    for key, label in (
        ("typicality_uninformative", "TYPICALITY_UNINFORMATIVE"),
        ("near_deterministic", "near_deterministic"),
        ("one_sided_degenerate", "one_sided_degenerate"),
        ("degenerate_heavy", "degenerate_heavy"),
        ("disconnected", "disconnected"),
        ("ref_seeds_lt_30", "ref_seeds_lt_30"),
    ):
        if bool(record.get(key, False)):
            annotations.append(label)
    if "p_diff" in record:
        annotations.append("p_diff")
    if "disp" in record:
        annotations.append("disp")
    return annotations


def _q_lt(value: Any, threshold: float) -> bool:
    """Return whether a q-value is finite and below threshold.

    Parameters
    ----------
    value : Any
        Candidate q-value.
    threshold : float
        Decision threshold.

    Returns
    -------
    bool
        ``True`` when finite and less than ``threshold``.
    """
    try:
        q_value = float(value)
    except (TypeError, ValueError):
        return False
    return math.isfinite(q_value) and q_value < threshold


def _rng_from_purpose(purpose: str) -> np.random.Generator:
    """Create a deterministic generator from a registered purpose string.

    Parameters
    ----------
    purpose : str
        Purpose string to hash.

    Returns
    -------
    numpy.random.Generator
        Deterministically seeded generator.
    """
    seed = int.from_bytes(hashlib.sha256(purpose.encode()).digest()[:8], "little")
    return np.random.default_rng(seed)


def _synthetic_clouds(
    rng: np.random.Generator,
    count: int,
    n_nodes: int,
    *,
    shift: float,
    spread: float,
) -> np.ndarray:
    """Generate synthetic 2D Gaussian-mixture layout clouds.

    Parameters
    ----------
    rng : numpy.random.Generator
        Seeded generator.
    count : int
        Number of clouds.
    n_nodes : int
        Number of points per cloud.
    shift : float
        Mean shift applied to the second component.
    spread : float
        Multiplicative spread perturbation.

    Returns
    -------
    numpy.ndarray
        Synthetic layouts with shape ``[count, n_nodes, 2]``.
    """
    centers = np.array([[-1.0, 0.0], [1.0 + shift, 0.0]], dtype=np.float64)
    labels = rng.integers(0, 2, size=(count, n_nodes))
    noise = rng.normal(scale=0.25 * (1.0 + spread), size=(count, n_nodes, 2))
    return centers[labels] + noise
