"""Deterministic modeled-cost table for native benchmark work admission.

The joint budget design names modeled wall-seconds (MWS) as the canonical
unit, with DWU as an equivalent design alias. The public B1 API requested
``*_dwu`` field names, so this module keeps those names while documenting that
one DWU is one modeled wall-second on the frozen calibration reference box.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Optional

import torch

CostTerms = dict[str, tuple[float, float]]
CostTable = dict[tuple[str, str], CostTerms]

PROVENANCE_REF = (
    "C1-2026-07-20 idle fidelity/runtime telemetry plus bounded directed-prior "
    "modeling (docs/native_cost_model_calibration_2026-07-20.md); fCoSE terms "
    "refit C2-2026-07-21 with a piecewise exact/Barnes-Hut regime split "
    "(docs/native_cost_model_calibration_2026-07-21.md)."
)

# Mirrors FCoSEConfig.max_exact_repulsion_nodes (the marketplace seam calls
# layout_fcose_pipeline with the default cap). At or below this node count the
# spring embedder uses exact vectorized pairwise repulsion and a full
# 2500-step arm costs ~0.2-0.6s on the calibration box; above it the per-step
# Barnes-Hut tree pass dominates and real cost jumps 3-4 orders of magnitude
# (~6.5 s/step at n=1000). One linear (alpha, beta) pair cannot represent both
# regimes: the old single line, anchored to price the n=1000 row out,
# over-priced small/medium rows 90-7000x and starved the winning fCoSE arms
# (the M2 4-row regression class).
FCOSE_EXACT_REPULSION_NODE_CAP = 512

# Frozen modeled wall-second constants. Tiny-row W5 priors are retained as
# protective measured/model anchors; directed flat arms use P90 idle telemetry
# with per-family sibling-load envelopes. fCoSE generation is piecewise by
# node count: "force" (exact-repulsion regime, N <= 512, intercept-slope fit)
# vs "force_bh" (Barnes-Hut regime, N > 512, zero-intercept rate fit from
# short-step samples). C2 constants are verbatim calibrate output on the
# 2026-07-21 telemetry (c2_fcose_telemetry_{exact,bh}.jsonl, envelope 2.0);
# the Barnes-Hut term keeps r8_nested_scale_1k_budget priced out by >100x
# budget, preserving the fCoSE-skip / Arm-S-admit scale anchor.
FROZEN_COST_TABLE: CostTable = {
    ("fcose", "cpu"): {
        "force": (5.49172932331e-06, 0.2094),
        "force_bh": (0.00164199078341, 0.0),
        "score": (0.000015, 1.0),
    },
    ("fcose", "cuda"): {
        "force": (3.12987012987e-07, 0.3726),
        "force_bh": (0.00427751152074, 0.0),
        "score": (0.000012, 1.0),
    },
    ("stress", "cpu"): {"pairs": (0.000010, 1.0), "score": (0.000015, 1.0)},
    ("stress", "cuda"): {"pairs": (0.000006, 0.8), "score": (0.000012, 0.8)},
    ("apsp", "cpu"): {"apsp": (0.000002, 0.5)},
    ("apsp", "cuda"): {"apsp": (0.000001, 0.4)},
    ("ruler", "cpu"): {"samples": (0.000015, 0.25)},
    ("ruler", "cuda"): {"samples": (0.000012, 0.20)},
    ("w5", "cpu"): {
        "step": (0.0437, 0.0),
        "referee": (0.019, 0.0),
    },
    ("w5", "cuda"): {
        "step": (0.0437, 0.0),
        "referee": (0.019, 0.0),
    },
    ("arm_s", "cpu"): {
        "stress_pairs": (0.000010, 2.0),
        "overlap_heal": (0.000020, 1.0),
        "score": (0.000015, 1.0),
    },
    ("arm_s", "cuda"): {
        "stress_pairs": (0.000008, 1.5),
        "overlap_heal": (0.000014, 0.8),
        "score": (0.000012, 0.8),
    },
    ("directed_mrtree", "cpu"): {"full_arm": (0.0, 4.0)},
    ("directed_mrtree", "cuda"): {"full_arm": (0.0, 4.0)},
    ("directed_ordering", "cpu"): {"full_arm": (0.0, 5.0)},
    ("directed_ordering", "cuda"): {"full_arm": (0.0, 5.0)},
    ("directed_pivot_mds", "cpu"): {"full_arm": (0.0, 6.0)},
    ("directed_pivot_mds", "cuda"): {"full_arm": (0.0, 5.0)},
    ("directed_recombinant", "cpu"): {"full_arm": (0.0, 2.5)},
    ("directed_recombinant", "cuda"): {"full_arm": (0.0, 2.5)},
    ("directed_stress_blend", "cpu"): {"full_arm": (0.0, 12.0)},
    ("directed_stress_blend", "cuda"): {"full_arm": (0.0, 10.0)},
    ("directed_sugiyama", "cpu"): {"full_arm": (0.0, 2.2)},
    ("directed_sugiyama", "cuda"): {"full_arm": (0.0, 2.2)},
    ("directed_yifanhu", "cpu"): {"full_arm": (0.0, 4.0)},
    ("directed_yifanhu", "cuda"): {"full_arm": (0.0, 3.5)},
    ("opaque", "cpu"): {"full_arm": (0.0, 30.0)},
    ("opaque", "cuda"): {"full_arm": (0.0, 30.0)},
}


@dataclass(frozen=True)
class NativeWorkCost:
    """Modeled native work cost debited by the deterministic ledger.

    Parameters
    ----------
    family : str
        Native candidate or phase family, such as ``"fcose"`` or ``"w5"``.
    generation_dwu : float
        Deterministic work units charged when the candidate is admitted.
    reserved_score_dwu : float
        Deterministic work units reserved for full scoring of the admitted
        candidate. This portion may be released only for deterministic
        rejection reasons.
    metadata : dict[str, object]
        JSON-compatible derivation details for decision logs and calibration
        replay.
    mandatory_tail_dwu : float
        Optional package-tail units retained for Sol/Fable parity with the
        joint design. The B1 ledger keeps global tail reservation on the
        ledger itself, but retaining this field makes model outputs complete.
    device_class : str
        Frozen cost-table device axis, usually ``"cpu"`` or ``"cuda"``.
    """

    family: str
    generation_dwu: float
    reserved_score_dwu: float
    metadata: dict[str, Any] = field(default_factory=dict)
    mandatory_tail_dwu: float = 0.0
    device_class: str = "cpu"


@dataclass(frozen=True)
class _ProblemSize:
    """Normalized graph size extracted from native problem-like objects.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    num_edges : int
        Number of graph edges.
    """

    num_nodes: int
    num_edges: int


def _safe_nonnegative_int(value: object, default: int = 0) -> int:
    """Return ``value`` as a non-negative integer.

    Parameters
    ----------
    value : object
        Candidate integer-like value.
    default : int, default=0
        Fallback when conversion fails.

    Returns
    -------
    int
        Converted integer clamped to zero or above.
    """
    if isinstance(value, (str, bytes, bytearray, int, float)):
        try:
            return max(0, int(value))
        except ValueError:
            return max(0, int(default))
    try:
        return max(0, int(default))
    except (TypeError, ValueError):
        return max(0, int(default))


def _safe_nonnegative_float(value: object, default: float = 0.0) -> float:
    """Return ``value`` as a non-negative float.

    Parameters
    ----------
    value : object
        Candidate float-like value.
    default : float, default=0.0
        Fallback when conversion fails.

    Returns
    -------
    float
        Converted float clamped to zero or above.
    """
    if isinstance(value, (str, bytes, bytearray, int, float)):
        try:
            return max(0.0, float(value))
        except ValueError:
            return max(0.0, float(default))
    return max(0.0, float(default))


def _edge_count_from_problem(problem: object) -> int:
    """Return an edge count from a problem-like object.

    Parameters
    ----------
    problem : object
        Native ``LayoutProblem`` or a lightweight test object/dict.

    Returns
    -------
    int
        Number of edges, or zero when no edge payload is available.
    """
    if isinstance(problem, Mapping):
        edge_value = problem.get("num_edges", problem.get("edges"))
        edge_index = problem.get("edge_index")
    else:
        edge_value = getattr(problem, "num_edges", None)
        edge_index = getattr(problem, "edge_index", None)
    if edge_value is not None:
        return _safe_nonnegative_int(edge_value)
    if isinstance(edge_index, torch.Tensor) and edge_index.ndim >= 2:
        return int(edge_index.shape[1])
    return 0


def _problem_size(problem: object) -> _ProblemSize:
    """Normalize a native problem-like object into node and edge counts.

    Parameters
    ----------
    problem : object
        Native ``LayoutProblem`` or a lightweight mapping with ``num_nodes`` and
        optional ``num_edges``/``edge_index`` fields.

    Returns
    -------
    _ProblemSize
        Non-negative graph size used by volume helpers.
    """
    if isinstance(problem, Mapping):
        node_value = problem.get("num_nodes", problem.get("n", 0))
    else:
        node_value = getattr(problem, "num_nodes", getattr(problem, "n", 0))
    return _ProblemSize(
        num_nodes=_safe_nonnegative_int(node_value),
        num_edges=_edge_count_from_problem(problem),
    )


def fcose_force_volume(num_nodes: int, num_edges: int, steps: int) -> float:
    """Return fCoSE force-update volume.

    Parameters
    ----------
    num_nodes : int
        Graph node count.
    num_edges : int
        Graph edge count.
    steps : int
        Planned force-iteration count.

    Returns
    -------
    float
        Linear ``(N + E) * steps`` unit volume.
    """
    return float(max(num_nodes, 0) + max(num_edges, 0)) * float(max(steps, 0))


def stress_pair_volume(num_nodes: int, steps: int, sample_pairs: Optional[int] = None) -> float:
    """Return stress-update pair volume.

    Parameters
    ----------
    num_nodes : int
        Graph node count.
    steps : int
        Stress optimization steps or epochs.
    sample_pairs : int, optional
        Explicit sampled pair count. ``None`` uses the dense ``N(N-1)/2`` pair
        count.

    Returns
    -------
    float
        Pair-update volume multiplied by planned steps.
    """
    dense_pairs = max(num_nodes, 0) * max(num_nodes - 1, 0) // 2
    pairs = dense_pairs if sample_pairs is None else max(int(sample_pairs), 0)
    return float(pairs) * float(max(steps, 0))


def apsp_volume(num_nodes: int, num_edges: int) -> float:
    """Return all-pairs shortest-path cache volume.

    Parameters
    ----------
    num_nodes : int
        Graph node count.
    num_edges : int
        Graph edge count.

    Returns
    -------
    float
        Sparse BFS-style ``N * (N + E)`` volume.
    """
    return float(max(num_nodes, 0)) * float(max(num_nodes, 0) + max(num_edges, 0))


def ruler_sample_volume(
    num_nodes: int,
    num_edges: int,
    samples: Optional[int] = None,
) -> float:
    """Return frozen-ruler sampled metric volume.

    Parameters
    ----------
    num_nodes : int
        Graph node count.
    num_edges : int
        Graph edge count.
    samples : int, optional
        Explicit sample budget. ``None`` uses a conservative edge-pair cap.

    Returns
    -------
    float
        Metric sample volume.
    """
    edge_pairs = max(num_edges, 0) * max(num_edges - 1, 0) // 2
    default_samples = min(max(edge_pairs, max(num_nodes, 0)), 100_000)
    return float(default_samples if samples is None else max(int(samples), 0))


def w5_step_volume(mode: str, steps: int, seeds: int = 1, checkpoints: int = 1) -> float:
    """Return W5 planned step/referee volume for a routed mode.

    Parameters
    ----------
    mode : str
        W5 route label. The current B1 model keeps one scalar step unit for all
        modes; calibration may split this later if residuals require it.
    steps : int
        Planned surrogate steps per seed.
    seeds : int, default=1
        Number of W5 seed layouts.
    checkpoints : int, default=1
        Number of checkpoint referee scores per seed.

    Returns
    -------
    float
        Planned W5 work volume. The tiny-row constants are encoded as
        per-step/per-referee costs in ``FROZEN_COST_TABLE``.
    """
    del mode
    return float(max(steps, 0) * max(seeds, 0) + max(checkpoints, 0) * max(seeds, 0))


def _term_cost(terms: CostTerms, term: str, volume: float) -> float:
    """Return modeled cost for one calibrated term.

    Parameters
    ----------
    terms : dict[str, tuple[float, float]]
        Frozen term constants keyed by term name.
    term : str
        Term name to price.
    volume : float
        Deterministic unit volume.

    Returns
    -------
    float
        ``alpha * volume + beta`` clamped to zero.
    """
    alpha, beta = terms.get(term, (0.0, 0.0))
    return max(0.0, float(alpha) * max(float(volume), 0.0) + float(beta))


def _steps(knobs: Mapping[str, object], default: int) -> int:
    """Return a non-negative step count from knob mappings.

    Parameters
    ----------
    knobs : Mapping[str, object]
        Cost-estimation knobs.
    default : int
        Default step count.

    Returns
    -------
    int
        Resolved step count.
    """
    return _safe_nonnegative_int(knobs.get("steps", default), default)


def estimate_native_work_cost(
    problem: object,
    family: str,
    knobs: Mapping[str, object],
    device_class: str,
) -> NativeWorkCost:
    """Estimate deterministic native work cost from frozen calibration constants.

    Parameters
    ----------
    problem : object
        Native ``LayoutProblem`` or problem-like mapping with graph size.
    family : str
        Candidate family to price.
    knobs : Mapping[str, object]
        Deterministic plan knobs such as ``steps``, ``samples``, ``seeds`` and
        ``checkpoints``.
    device_class : str
        Frozen device class axis, usually ``"cpu"`` or ``"cuda"``.

    Returns
    -------
    NativeWorkCost
        Modeled generation and reserved scoring DWU with metadata describing
        the chosen volumes and calibration provenance.
    """
    normalized_family = str(family).lower()
    normalized_device = str(device_class).lower()
    terms = FROZEN_COST_TABLE.get(
        (normalized_family, normalized_device),
        FROZEN_COST_TABLE[("opaque", normalized_device if normalized_device == "cuda" else "cpu")],
    )
    size = _problem_size(problem)
    metadata: dict[str, Any] = {
        "provenance": PROVENANCE_REF,
        "num_nodes": size.num_nodes,
        "num_edges": size.num_edges,
        "device_class": normalized_device,
        "terms": {},
    }

    if normalized_family == "fcose":
        steps = _steps(knobs, 250)
        force_volume = fcose_force_volume(size.num_nodes, size.num_edges, steps)
        score_volume = ruler_sample_volume(size.num_nodes, size.num_edges, knobs.get("samples"))  # type: ignore[arg-type]
        # Piecewise regime split mirroring FCoSEConfig.max_exact_repulsion_nodes.
        # Missing "force_bh" in a custom table falls back to the exact-regime
        # term rather than an accidental zero-cost (free-admission) estimate.
        barnes_hut = size.num_nodes > FCOSE_EXACT_REPULSION_NODE_CAP
        force_term = "force_bh" if barnes_hut and "force_bh" in terms else "force"
        generation = _term_cost(terms, force_term, force_volume)
        reserved = _term_cost(terms, "score", score_volume)
        metadata["terms"] = {force_term: force_volume, "score": score_volume}
        metadata["fcose_regime"] = "barnes_hut" if barnes_hut else "exact"
    elif normalized_family in {"stress", "arm_s"}:
        steps = _steps(knobs, 50)
        pair_volume = stress_pair_volume(size.num_nodes, steps, knobs.get("sample_pairs"))  # type: ignore[arg-type]
        score_volume = ruler_sample_volume(size.num_nodes, size.num_edges, knobs.get("samples"))  # type: ignore[arg-type]
        pair_term = "stress_pairs" if normalized_family == "arm_s" else "pairs"
        generation = _term_cost(terms, pair_term, pair_volume)
        if normalized_family == "arm_s":
            heal_volume = float(max(size.num_nodes, 0) * max(_steps(knobs, 10), 1))
            generation += _term_cost(terms, "overlap_heal", heal_volume)
            metadata["terms"] = {
                pair_term: pair_volume,
                "overlap_heal": heal_volume,
                "score": score_volume,
            }
        else:
            metadata["terms"] = {pair_term: pair_volume, "score": score_volume}
        reserved = _term_cost(terms, "score", score_volume)
    elif normalized_family == "apsp":
        volume = apsp_volume(size.num_nodes, size.num_edges)
        generation = _term_cost(terms, "apsp", volume)
        reserved = 0.0
        metadata["terms"] = {"apsp": volume}
    elif normalized_family == "ruler":
        volume = ruler_sample_volume(size.num_nodes, size.num_edges, knobs.get("samples"))  # type: ignore[arg-type]
        generation = 0.0
        reserved = _term_cost(terms, "samples", volume)
        metadata["terms"] = {"samples": volume}
    elif normalized_family == "w5":
        steps = _steps(knobs, 96)
        seeds = _safe_nonnegative_int(knobs.get("seeds", 1), 1)
        checkpoints = _safe_nonnegative_int(knobs.get("checkpoints", 1), 1)
        mode = str(knobs.get("mode", "default"))
        step_volume = float(max(steps, 0) * max(seeds, 0))
        referee_volume = float(max(checkpoints, 0) * max(seeds, 0))
        generation = _term_cost(terms, "step", step_volume)
        reserved = _term_cost(terms, "referee", referee_volume)
        metadata["terms"] = {
            "step": step_volume,
            "referee": referee_volume,
            "combined": w5_step_volume(mode, steps, seeds, checkpoints),
        }
        metadata["mode"] = mode
    elif normalized_family.startswith("directed_"):
        volume = _safe_nonnegative_float(knobs.get("volume", 1.0), 1.0)
        generation = _term_cost(terms, "full_arm", volume)
        reserved = 0.0
        metadata["terms"] = {"full_arm": volume}
    else:
        volume = _safe_nonnegative_float(knobs.get("volume", 1.0), 1.0)
        generation = _term_cost(terms, "full_arm", volume)
        reserved = 0.0
        metadata["terms"] = {"full_arm": volume}

    return NativeWorkCost(
        family=normalized_family,
        generation_dwu=generation,
        reserved_score_dwu=reserved,
        mandatory_tail_dwu=_safe_nonnegative_float(knobs.get("mandatory_tail_dwu", 0.0)),
        device_class=normalized_device,
        metadata=metadata,
    )


__all__ = [
    "FCOSE_EXACT_REPULSION_NODE_CAP",
    "FROZEN_COST_TABLE",
    "NativeWorkCost",
    "PROVENANCE_REF",
    "apsp_volume",
    "estimate_native_work_cost",
    "fcose_force_volume",
    "ruler_sample_volume",
    "stress_pair_volume",
    "w5_step_volume",
]
