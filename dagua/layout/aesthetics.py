"""User-facing aesthetic-priority knob (r80-S8).

Dagua already lets users tweak differentiable loss weights directly
(``LayoutConfig(w_crossing=5.0)``) and write custom constraints. What was
missing: the internal candidate-selection composite (the scorer the
undirected portfolio contest -- see
``dagua.layout.ops.pipelines.native_undirected`` -- uses to pick a winner
among several internally-generated layouts) had fixed weights, so a user's
aesthetic priorities could steer the optimizer but not which ENGINE won the
contest.

This module is the internal machinery for a single normalized "priority
profile" that plumbs into both:

- Wire-through A (selection): a NEW parallel reweighted-scoring path
  (:func:`reweighted_composite`) that the undirected portfolio contest can
  opt into. It intentionally duplicates the per-term subscore formulas from
  ``dagua.metrics.composite`` / ``composite_undirected`` rather than
  modifying those frozen functions (see the hard rule in the r80-S8 brief:
  the benchmark harness and the internal candidate contest both depend on
  ``composite`` / ``composite_undirected`` / ``composite_auto`` staying
  byte-for-byte unchanged).
- Wire-through B (losses): :func:`apply_loss_multipliers` maps the same
  profile onto the native gradient core's ``w_*`` loss-weight fields as
  MULTIPLICATIVE adjustments, so the differentiable optimizer itself is
  steered toward the same priorities the selection composite rewards.

Default identity is the load-bearing invariant: when a caller never sets an
aesthetic knob, :func:`resolve_aesthetic_profile` returns ``None`` and every
call site in this sprint falls through to the EXACT pre-existing code path
(``composite_auto`` called directly, no wrapper, no extra float ops) --
see ``native_undirected.py::_score_undirected_candidate`` and
``resolve.py::prepare_pipeline_config``. This guarantees the r80-S8 gate-1
default-identity sweep is bit-identical, not merely "close."

The public ``LayoutConfig`` surface (``prioritize`` / ``aesthetic_weights``)
is intentionally thin and documented as provisional in
``.project-context/research/r79_native/P15_AESTHETIC_KNOB.md`` -- this
module is the stable seam; the field names are the part JMT signs off on.
"""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import TYPE_CHECKING, Dict, Mapping, Optional, Tuple

if TYPE_CHECKING:  # pragma: no cover - import cycle guard, types only
    from dagua.config import LayoutConfig

# ---------------------------------------------------------------------------
# Term vocabulary -- mirrors dagua.metrics.composite() / composite_undirected()
# ---------------------------------------------------------------------------

#: Base weights for the directed composite (dagua.metrics.composite). Sums
#: to 100, matching the frozen function's docstring exactly.
DIRECTED_BASE_WEIGHTS: Dict[str, float] = {
    "dag_consistency": 22.0,
    "edge_length_cv": 18.0,
    "depth_spearman": 13.0,
    "overlap": 8.0,
    "straightness": 9.0,
    "crossing": 9.0,
    "stress": 10.0,
    "angular_res": 5.0,
    "cluster_sep": 6.0,
}

#: Base weights for the undirected composite (dagua.metrics.composite_undirected).
#: Sums to 100, matching the frozen function's docstring exactly. Every term
#: here is also a directed term (subset), so a single term vocabulary
#: (``DIRECTED_BASE_WEIGHTS`` keys) covers both composites.
UNDIRECTED_BASE_WEIGHTS: Dict[str, float] = {
    "edge_length_cv": 40.0,
    "overlap": 20.0,
    "crossing": 20.0,
    "angular_res": 10.0,
    "cluster_sep": 10.0,
}

#: All valid term names accepted in ``aesthetic_weights``. Directed-only
#: terms (dag_consistency, depth_spearman, straightness, stress) are simply
#: inert multipliers when the graph resolves to the undirected composite.
VALID_TERMS: Tuple[str, ...] = tuple(DIRECTED_BASE_WEIGHTS)

#: Named presets. Each maps a subset of terms to a multiplier; terms not
#: listed default to 1.0 (no change from today's weights).
PRESETS: Dict[str, Dict[str, float]] = {
    # Untangle the drawing: crossings are the single biggest readability
    # complaint for dense graphs. Boosts both the selection term and (via
    # apply_loss_multipliers) the routed/straight-line crossing losses.
    "crossings": {"crossing": 3.0},
    # Uniform edge lengths: the classic "tidy" force-directed look.
    "uniform_edges": {"edge_length_cv": 3.0},
    # Compactness: keep the drawing tight (graph-theoretic distance
    # fidelity) without letting nodes collide.
    "compactness": {"stress": 2.5, "overlap": 1.5},
    # Readability: broader bundle -- open angles, fewer crossings, straight
    # edges, and (for directed graphs) a clean depth hierarchy.
    "readability": {
        "angular_res": 2.0,
        "crossing": 1.5,
        "straightness": 1.5,
        "depth_spearman": 1.3,
    },
}

#: Multiplicative clamp applied to every resolved loss-weight multiplier so
#: a pathological ``aesthetic_weights`` value cannot blow up the optimizer.
LOSS_MULTIPLIER_CLAMP: Tuple[float, float] = (0.1, 5.0)

#: Wire-through B mapping: aesthetic term -> differentiable loss weight
#: field(s) on LayoutConfig that get the SAME multiplier applied. See
#: P15_AESTHETIC_KNOB.md for the full table with rationale per row.
#: ``depth_spearman`` has no direct differentiable-loss analogue today (the
#: closest structural lever, DAG ordering strength, is already driven by
#: the ``dag_consistency`` row) so it is intentionally absent here -- it
#: still participates in Wire-through A (selection).
LOSS_MULTIPLIER_MAP: Dict[str, Tuple[str, ...]] = {
    "dag_consistency": ("w_dag",),
    "edge_length_cv": ("w_length_variance",),
    "overlap": ("w_overlap",),
    "straightness": ("w_straightness",),
    "crossing": ("w_crossing", "w_edge_crossing", "w_edge_node_crossing"),
    "stress": ("w_stress",),
    "angular_res": ("w_fanout", "w_edge_angular_res"),
    "cluster_sep": ("w_cluster", "w_cluster_contain"),
}


@dataclass(frozen=True)
class AestheticProfile:
    """A resolved, normalized aesthetic-priority profile.

    Parameters
    ----------
    weights : Mapping[str, float]
        Term name -> multiplier (relative to the frozen composite's default
        weight for that term). Terms absent from this mapping default to a
        multiplier of ``1.0`` (no change). An empty mapping is a valid but
        pointless identity profile; callers should prefer ``None`` (see
        :func:`resolve_aesthetic_profile`) for the true default path.
    """

    weights: Mapping[str, float]

    def multiplier(self, term: str) -> float:
        """Return the resolved multiplier for one term, defaulting to 1.0.

        Parameters
        ----------
        term : str
            Aesthetic term name (one of ``VALID_TERMS``).

        Returns
        -------
        float
            The profile's multiplier for ``term``, or ``1.0`` when unset.
        """
        return float(self.weights.get(term, 1.0))


def resolve_aesthetic_profile(config: "LayoutConfig") -> Optional[AestheticProfile]:
    """Resolve ``config``'s aesthetic knob into a profile, or ``None``.

    ``None`` is the true identity signal: callers must skip the reweighted
    path entirely (not merely pass an all-ones profile through it), so the
    default output stays bit-identical to pre-knob behavior.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration, optionally carrying ``prioritize`` (a preset
        name) and/or ``aesthetic_weights`` (an explicit term -> multiplier
        override dict). Both default to ``None``.

    Returns
    -------
    AestheticProfile or None
        ``None`` when neither field is set (or ``aesthetic_weights`` is an
        empty dict). Otherwise a merged profile: preset weights first, then
        ``aesthetic_weights`` overriding per-key.

    Raises
    ------
    ValueError
        If ``prioritize`` names an unknown preset, if ``aesthetic_weights``
        contains an unknown term name, or if any explicit weight is not
        strictly positive.
    """
    prioritize = getattr(config, "prioritize", None)
    explicit = getattr(config, "aesthetic_weights", None)
    if prioritize is None and not explicit:
        return None

    merged: Dict[str, float] = {}
    if prioritize is not None:
        if prioritize not in PRESETS:
            raise ValueError(
                f"prioritize must be one of {sorted(PRESETS)} or None, got {prioritize!r}."
            )
        merged.update(PRESETS[prioritize])

    if explicit:
        for term, weight in explicit.items():
            if term not in DIRECTED_BASE_WEIGHTS:
                raise ValueError(
                    f"Unknown aesthetic term {term!r} in aesthetic_weights; "
                    f"valid terms: {sorted(DIRECTED_BASE_WEIGHTS)}."
                )
            weight_f = float(weight)
            if weight_f <= 0.0:
                raise ValueError(f"aesthetic_weights[{term!r}] must be positive, got {weight_f}.")
            merged[term] = weight_f

    if not merged:
        return None
    return AestheticProfile(weights=merged)


def _term_subscores(
    metrics: Mapping[str, float],
    is_directed: bool,
    degenerate: bool,
) -> Dict[str, float]:
    """Replicate the per-term [0, 1]-ish subscores from the frozen composites.

    This intentionally mirrors ``dagua.metrics.composite`` /
    ``composite_undirected`` formula-for-formula (minus the weight
    multiplication) so :func:`reweighted_composite` stays numerically
    consistent with the frozen scorer at multiplier=1.0. See those
    functions' docstrings for the degeneracy-guard rationale.

    Parameters
    ----------
    metrics : Mapping[str, float]
        Metric name to scalar value mapping from ``dagua.metrics.full``.
    is_directed : bool
        Whether to include the directed-only terms (dag_consistency,
        depth_spearman, straightness, stress).
    degenerate : bool
        Whether the layout is at a point-collapsed scale (see
        ``dagua.metrics._is_degenerate_scale``).

    Returns
    -------
    Dict[str, float]
        Term name -> raw subscore in ``[0, 1]``.
    """
    subscores: Dict[str, float] = {}
    if is_directed:
        subscores["dag_consistency"] = float(metrics.get("dag_consistency", 0.0))
        subscores["depth_spearman"] = max(0.0, float(metrics.get("depth_spearman_rho", 0.0)))
        straight_deg = float(metrics.get("edge_straightness_mean_deg", 45.0))
        subscores["straightness"] = max(0.0, 1.0 - straight_deg / 45.0)
        subscores["stress"] = max(0.0, 1.0 - float(metrics.get("sampled_stress", 1.0)))

    subscores["edge_length_cv"] = (
        0.0 if degenerate else max(0.0, 1.0 - float(metrics.get("edge_length_cv", 1.0)))
    )
    subscores["overlap"] = 1.0 if metrics.get("overlap_count", 1) == 0 else 0.0
    subscores["crossing"] = (
        0.0 if degenerate else max(0.0, 1.0 - float(metrics.get("crossing_rate", 0.5)) * 10.0)
    )
    subscores["angular_res"] = min(1.0, float(metrics.get("angular_res_mean_deg", 20.0)) / 40.0)
    if "cluster_mean_sep_ratio" in metrics:
        subscores["cluster_sep"] = min(1.0, float(metrics["cluster_mean_sep_ratio"]) / 5.0)
    elif not is_directed and "cluster_sep" in metrics:
        subscores["cluster_sep"] = min(1.0, float(metrics["cluster_sep"]))
    else:
        subscores["cluster_sep"] = 0.5
    return subscores


def reweighted_composite(
    metrics: Mapping[str, float],
    is_directed: bool,
    profile: AestheticProfile,
) -> float:
    """Score ``metrics`` with the frozen composite's terms, reweighted by ``profile``.

    A NEW parallel path -- never modifies ``dagua.metrics.composite`` /
    ``composite_undirected`` / ``composite_auto``. At the identity profile
    (every term multiplier == 1.0) this reproduces the frozen composite's
    formula exactly (same base weights, same subscore formulas, same
    100-point total); callers must still prefer calling ``composite_auto``
    directly when ``profile is None`` (see module docstring) to guarantee
    bit-identical defaults rather than relying on this function's float
    identity.

    Parameters
    ----------
    metrics : Mapping[str, float]
        Metric name to scalar value mapping from ``dagua.metrics.full``.
    is_directed : bool
        Whether to score with the directed (9-term, sum=100) or undirected
        (5-term, sum=100) term set.
    profile : AestheticProfile
        Resolved per-term multipliers.

    Returns
    -------
    float
        Reweighted composite score. Renormalized so the scaled base weights
        still sum to the original total (100), keeping scores comparable in
        magnitude to the frozen composite even under aggressive reweighting.
    """
    from dagua.metrics import _is_degenerate_scale

    base_weights = DIRECTED_BASE_WEIGHTS if is_directed else UNDIRECTED_BASE_WEIGHTS
    degenerate = _is_degenerate_scale(dict(metrics))
    subscores = _term_subscores(metrics, is_directed=is_directed, degenerate=degenerate)

    scaled = {term: base * profile.multiplier(term) for term, base in base_weights.items()}
    total_scaled = sum(scaled.values())
    total_base = sum(base_weights.values())
    factor = (total_base / total_scaled) if total_scaled > 0.0 else 1.0

    score = sum(scaled[term] * subscores[term] for term in base_weights) * factor
    return float(score)


def apply_loss_multipliers(config: "LayoutConfig", profile: AestheticProfile) -> "LayoutConfig":
    """Return a config copy with ``w_*`` fields scaled per ``LOSS_MULTIPLIER_MAP``.

    Multiplicative-only: every affected field becomes
    ``original_value * clamp(profile.multiplier(term))``. Fields for terms
    the profile leaves at multiplier 1.0 are untouched (not even
    reassigned), so this is a true no-op when ``profile`` carries no
    weights for a given term.

    Parameters
    ----------
    config : LayoutConfig
        Base configuration (already a per-problem shallow copy from the
        caller -- see ``dagua.layout.resolve.prepare_pipeline_config``).
    profile : AestheticProfile
        Resolved per-term multipliers.

    Returns
    -------
    LayoutConfig
        A new shallow copy of ``config`` with the mapped ``w_*`` fields
        scaled. The copy also carries
        ``_dagua_native_aesthetic_multipliers_applied`` (a dict of the
        field -> clamped multiplier actually applied) for debugging /
        evidence capture.
    """
    resolved = copy.copy(config)
    applied: Dict[str, float] = {}
    low, high = LOSS_MULTIPLIER_CLAMP
    for term, field_names in LOSS_MULTIPLIER_MAP.items():
        multiplier = profile.multiplier(term)
        if multiplier == 1.0:
            continue
        clamped = max(low, min(high, multiplier))
        for field_name in field_names:
            base_value = float(getattr(resolved, field_name, 0.0))
            setattr(resolved, field_name, base_value * clamped)
            applied[field_name] = clamped
    setattr(resolved, "_dagua_native_aesthetic_multipliers_applied", applied)
    return resolved


__all__ = [
    "AestheticProfile",
    "DIRECTED_BASE_WEIGHTS",
    "LOSS_MULTIPLIER_CLAMP",
    "LOSS_MULTIPLIER_MAP",
    "PRESETS",
    "UNDIRECTED_BASE_WEIGHTS",
    "VALID_TERMS",
    "apply_loss_multipliers",
    "resolve_aesthetic_profile",
    "reweighted_composite",
]
