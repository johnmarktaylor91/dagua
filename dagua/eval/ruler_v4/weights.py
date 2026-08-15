"""Declared edge-weight distance, order, and visual-encoding facets."""

# Contract identity docstrings intentionally keep each title and digest on one line.
# ruff: noqa: E501

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import DefaultDict, FrozenSet, List, Mapping, Optional, Tuple

import torch

from dagua.eval.ruler_v4._util import global_blend, graph_distances, primary_isotonic_fit, snap_unit
from dagua.eval.ruler_v4.scene import FacetResult, Scene, na_result, value_result
from dagua.eval.ruler_v4.structure import _distance_strata, _stress_from_fit

_DISTANCE_SEMANTICS = frozenset({"distance_cost", "connection_strength"})
_LOCAL_ORDER_SEMANTICS = frozenset({"distance_cost", "connection_strength"})

FITTED_DOF_CAP = 20
PRIOR_MASS_DISCLOSURE_GATE = 0.15
REQUIRED_PRIOR_FLOOR_FACETS = frozenset({"U12", "U13", "U34"})
GATE_DIAGNOSTIC_FACETS = frozenset(
    {
        "U02",
        "U04a",
        "U04b",
        "U05",
        "U06",
        "U14",
        "U15",
        "U19",
        "U20a",
        "U20b",
        "U37",
        "U39",
        "U40",
        "U42",
    }
)


@dataclass(frozen=True)
class SubtermWeight:
    """Declare one score-visible sub-term's fitted or fixed mass.

    Parameters
    ----------
    subterm_id : str
        Stable scored sub-term id from the frozen contract inventory.
    facet_id : str
        Owning facet id.
    group : str
        Reporting group used by composition attribution.
    weight : float
        Nonnegative composite mass supplied by the P5 fit or frozen prior.
    fitted_parameter : str or None
        Identity of the independently adjustable scalar that produced this
        mass. Reusing one identity across a fixed-ratio bundle counts one dof.
    prior_driven : bool
        Whether this mass enters PM-1's numerator for the active profile.
    diagnostic : bool
        Whether the term is carried at weight zero outside the headline.
    """

    subterm_id: str
    facet_id: str
    group: str
    weight: float
    fitted_parameter: Optional[str] = None
    prior_driven: bool = False
    diagnostic: bool = False


@dataclass(frozen=True)
class DofAccount:
    """Summarize fitted-cap accounting for one weight table.

    Parameters
    ----------
    d_power : int
        Information-limited fitted capacity.
    allowed : int
        Allowed fitted dof, ``min(20, D_power)``.
    used : int
        Independently adjustable score-visible scalar count.
    remaining : int
        Nonnegative unspent allowance.
    within_cap : bool
        Whether the table respects the allowed capacity.
    """

    d_power: int
    allowed: int
    used: int
    remaining: int
    within_cap: bool


@dataclass(frozen=True)
class PriorMassDisclosure:
    """Report PM-1 prior-driven mass for one observation profile.

    Parameters
    ----------
    numerator : float
        Applicable positive mass carried by prior-driven terms.
    denominator : float
        Total applicable positive headline mass after NA exclusion.
    fraction : float
        ``numerator / denominator``.
    partial : bool
        Whether the fraction is strictly above the preregistered gate.
    affected_facets : tuple[str, ...]
        Stable ids of applicable prior-driven facets.
    """

    numerator: float
    denominator: float
    fraction: float
    partial: bool
    affected_facets: Tuple[str, ...]


@dataclass(frozen=True)
class WeightTable:
    """Hold every explicit sub-term mass and its capacity provenance.

    Parameters
    ----------
    entries : tuple[SubtermWeight, ...]
        Per-sub-term masses. P5 supplies fitted values; this class contains no
        default weights.
    d_power : int
        Information-limited fitted capacity used by the A18 cap formula.
    prior_floors : mapping[str, float]
        Positive facet-level floors for U12, U13, and U34.
    other_fitted_parameters : tuple[str, ...]
        Score-visible fitted scalars outside sub-term masses, including group,
        aggregation, and headline-model parameters.
    """

    entries: Tuple[SubtermWeight, ...]
    d_power: int
    prior_floors: Mapping[str, float] = field(default_factory=dict)
    other_fitted_parameters: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        """Freeze mappings and validate local table invariants.

        Raises
        ------
        ValueError
            If ids, masses, floors, or parameter declarations are malformed.
        """

        if isinstance(self.d_power, bool) or not isinstance(self.d_power, int) or self.d_power < 0:
            raise ValueError("d_power must be a nonnegative integer")
        entries = tuple(self.entries)
        other_parameters = tuple(self.other_fitted_parameters)
        if len({entry.subterm_id for entry in entries}) != len(entries):
            raise ValueError("subterm ids must be unique")
        for entry in entries:
            if not entry.subterm_id or not entry.facet_id or not entry.group:
                raise ValueError("subterm, facet, and group ids must be nonempty")
            if not math.isfinite(entry.weight) or entry.weight < 0.0:
                raise ValueError("sub-term weights must be finite and nonnegative")
            if entry.diagnostic and entry.weight != 0.0:
                raise ValueError("diagnostic terms must carry weight zero")
            if entry.facet_id in GATE_DIAGNOSTIC_FACETS and not entry.diagnostic:
                raise ValueError(f"{entry.facet_id} must be carried as a diagnostic")
        floors = {facet_id: float(value) for facet_id, value in self.prior_floors.items()}
        if any(
            facet_id not in REQUIRED_PRIOR_FLOOR_FACETS or not math.isfinite(value) or value <= 0.0
            for facet_id, value in floors.items()
        ):
            raise ValueError("prior floors must be positive values for U12, U13, or U34")
        parameter_names = tuple(
            name
            for name in (
                *(entry.fitted_parameter for entry in entries),
                *other_parameters,
            )
            if name is not None
        )
        if any(not name for name in parameter_names):
            raise ValueError("fitted parameter identities must be nonempty")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "prior_floors", MappingProxyType(floors))
        object.__setattr__(self, "other_fitted_parameters", other_parameters)

    @property
    def by_subterm(self) -> Mapping[str, SubtermWeight]:
        """Return entries keyed by stable sub-term id.

        Returns
        -------
        mapping[str, SubtermWeight]
            Immutable lookup mapping.
        """

        return MappingProxyType({entry.subterm_id: entry for entry in self.entries})

    @property
    def dof_account(self) -> DofAccount:
        """Compute the strict independently-adjustable-scalar count.

        Returns
        -------
        DofAccount
            A18 capacity accounting with the information-limited cap.
        """

        parameter_names = {
            name
            for name in (
                *(entry.fitted_parameter for entry in self.entries),
                *self.other_fitted_parameters,
            )
            if name is not None
        }
        allowed = min(FITTED_DOF_CAP, self.d_power)
        used = len(parameter_names)
        return DofAccount(
            d_power=self.d_power,
            allowed=allowed,
            used=used,
            remaining=max(0, allowed - used),
            within_cap=used <= allowed,
        )

    def validate_for_contracts(self) -> None:
        """Validate completeness, gates-report diagnostics, and prior floors.

        Raises
        ------
        ValueError
            If the table cannot be used by the frozen 45-facet scorer.
        """

        from dagua.eval.ruler_v4.contracts import CONTRACTS

        ownership = {
            subterm_id: facet_id
            for facet_id, contract in CONTRACTS.items()
            for subterm_id in contract.scored_subterms
        }
        actual = set(self.by_subterm)
        expected = set(ownership)
        if actual != expected:
            raise ValueError(
                "weight table does not match frozen scored subterms; "
                f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
            )
        wrong_owners = sorted(
            entry.subterm_id
            for entry in self.entries
            if ownership[entry.subterm_id] != entry.facet_id
        )
        if wrong_owners:
            raise ValueError(f"weight table has incorrect facet ownership: {wrong_owners}")
        diagnostics = {entry.facet_id for entry in self.entries if entry.diagnostic}
        if diagnostics != GATE_DIAGNOSTIC_FACETS:
            raise ValueError("diagnostic facet set does not match the gates report")
        if set(self.prior_floors) != REQUIRED_PRIOR_FLOOR_FACETS:
            raise ValueError("U12, U13, and U34 prior floors are all required")
        facet_mass: DefaultDict[str, float] = defaultdict(float)
        for entry in self.entries:
            facet_mass[entry.facet_id] += entry.weight
        for facet_id, floor in self.prior_floors.items():
            if facet_mass[facet_id] < floor:
                raise ValueError(f"{facet_id} mass is below its preregistered prior floor")
        if not self.dof_account.within_cap:
            raise ValueError("fitted parameter count exceeds min(20, D_power)")

    def prior_mass_disclosure(self, applicable_subterms: FrozenSet[str]) -> PriorMassDisclosure:
        """Compute PM-1 after excluding NA and diagnostic terms.

        Parameters
        ----------
        applicable_subterms : frozenset[str]
            Scored sub-term ids available for this scene and profile.

        Returns
        -------
        PriorMassDisclosure
            Strict-gate disclosure record. Exactly 15 percent does not fire.
        """

        active = tuple(
            entry
            for entry in self.entries
            if entry.subterm_id in applicable_subterms
            and not entry.diagnostic
            and entry.weight > 0.0
        )
        denominator = math.fsum(entry.weight for entry in active)
        numerator = math.fsum(entry.weight for entry in active if entry.prior_driven)
        fraction = numerator / denominator if denominator > 0.0 else 0.0
        affected = tuple(sorted({entry.facet_id for entry in active if entry.prior_driven}))
        return PriorMassDisclosure(
            numerator=numerator,
            denominator=denominator,
            fraction=fraction,
            partial=fraction > PRIOR_MASS_DISCLOSURE_GATE,
            affected_facets=affected,
        )


def U35(scene: Scene) -> FacetResult:
    """Weighted distance fidelity. Frozen SHA-256: 7644f1fef1bcede526f923da3bc70e24005a0c28296369300eec9cc9d4743810."""

    if scene.graph.edge_weights is None:
        return na_result("WEIGHTS_ABSENT")
    if scene.graph.weight_semantics not in _DISTANCE_SEMANTICS:
        return na_result("WEIGHT_SEMANTICS_NOT_DISTANCE")
    declared = torch.tensor(scene.graph.edge_weights, dtype=torch.float64)
    if scene.graph.weight_semantics == "connection_strength":
        costs = torch.median(declared) / declared
    else:
        costs = declared
    weighted_graph = replace_edge_weights(scene, tuple(float(value) for value in costs))
    if not any(len(members) >= 3 for members in _component_members(scene)):
        return na_result("NO_THREE_NODE_WEIGHTED_COMPONENT")
    unweighted_strata = _distance_strata(scene)
    weighted_strata = _distance_strata(
        weighted_graph, graph_distances(weighted_graph, weighted=True)
    )
    logs = torch.log(costs)
    median = torch.median(logs)
    coefficient = (
        1.4826 * float(torch.median(torch.abs(logs - median))) / (abs(float(median)) + 1.0)
    )
    alpha = coefficient / (coefficient + 0.10)
    combined_stress = []
    stratum_weights = []
    for component_index in sorted({item[0] for item in unweighted_strata}):
        unweighted_component = [item for item in unweighted_strata if item[0] == component_index]
        weighted_component = [item for item in weighted_strata if item[0] == component_index]
        unweighted_order = torch.cat([item[2] for item in unweighted_component])
        weighted_order = torch.cat([item[2] for item in weighted_component])
        layout = torch.cat([item[3] for item in unweighted_component])
        fit_unweighted = primary_isotonic_fit(unweighted_order, layout)
        fit_weighted = primary_isotonic_fit(weighted_order, layout)
        cursor = 0
        for unweighted_item, weighted_item in zip(unweighted_component, weighted_component):
            count = unweighted_item[2].numel()
            stress_one = _stress_from_fit(
                unweighted_item[2],
                unweighted_item[3],
                fit_unweighted[cursor : cursor + count],
            )
            stress_weighted = _stress_from_fit(
                weighted_item[2],
                weighted_item[3],
                fit_weighted[cursor : cursor + count],
            )
            combined_stress.append((1.0 - alpha) * stress_one + alpha * stress_weighted)
            stratum_weights.append(float(count))
            cursor += count
    defect = global_blend(combined_stress, stratum_weights)
    return value_result(
        defect,
        {"U35.headline": defect},
        {"pair_count": int(sum(stratum_weights)), "alpha": alpha},
    )


def _component_members(scene: Scene) -> List[List[int]]:
    """Return connected components without introducing another public dependency.

    Parameters
    ----------
    scene : Scene
        Validated graph scene.

    Returns
    -------
    list[list[int]]
        Canonically ordered component memberships.
    """

    from dagua.eval.ruler_v4._util import components

    return components(scene)


def U36(scene: Scene) -> FacetResult:
    """Local weight monotonicity. Frozen SHA-256: 574b04b859567036912ca0815385311a2524dca0cb89ff0bda27acbb4c0ed703."""

    if scene.graph.edge_weights is None:
        return na_result("WEIGHTS_ABSENT")
    if scene.graph.weight_semantics not in _LOCAL_ORDER_SEMANTICS:
        return na_result("WEIGHT_SEMANTICS_NOT_LOCAL_ORDER")
    incident: DefaultDict[int, List[int]] = defaultdict(list)
    for edge_index, (source, target) in enumerate(scene.graph.edges):
        incident[source].append(edge_index)
        incident[target].append(edge_index)
    edges = torch.tensor(scene.graph.edges, dtype=torch.long)
    lengths = torch.linalg.vector_norm(
        scene.positions[edges[:, 0]] - scene.positions[edges[:, 1]], dim=1
    )
    strengths = torch.tensor(scene.graph.edge_weights, dtype=torch.float64)
    if scene.graph.weight_semantics == "distance_cost":
        strengths = 1.0 / strengths
    node_defects = []
    node_weights = []
    comparison_count = 0
    for edge_indices in incident.values():
        burdens = []
        for left_index, left in enumerate(edge_indices):
            for right in edge_indices[left_index + 1 :]:
                if strengths[left] == strengths[right]:
                    continue
                strong, weak = (
                    (left, right) if strengths[left] > strengths[right] else (right, left)
                )
                denominator = float(lengths[strong] + lengths[weak])
                margin = (
                    float(lengths[strong] - lengths[weak]) / denominator
                    if denominator > 0.0
                    else 0.0
                )
                scaled = max(-60.0, min(60.0, margin / 0.03))
                burdens.append(1.0 / (1.0 + math.exp(-scaled)))
        if burdens:
            node_defects.append(snap_unit(sum(burdens) / len(burdens)))
            node_weights.append(float(len(burdens)))
            comparison_count += len(burdens)
    if not node_defects:
        return na_result("NO_LOCAL_WEIGHT_ORDER")
    defect = global_blend(node_defects, node_weights)
    return value_result(
        defect,
        {"U36.headline": defect},
        {"comparison_count": comparison_count, "node_count": len(node_defects)},
    )


def U37(scene: Scene) -> FacetResult:
    """Thickness-only weight encoding. Frozen SHA-256: ad46f1330b123bd7941b47e9e49cd00b59803c0705b2d2432bda4da81ece53da."""

    if scene.graph.edge_weights is None or scene.graph.weight_visual_channel != "stroke_thickness":
        return na_result("THICKNESS_ENCODING_NOT_DECLARED")
    weights = torch.tensor(scene.graph.edge_weights, dtype=torch.float64)
    knots = torch.tensor(scene.graph.weight_encoding_knots, dtype=torch.float64)
    widths = torch.tensor(scene.style.edge_stroke_widths, dtype=torch.float64)
    target_widths = _log_linear_targets(weights, knots)
    log_error = torch.log(widths / target_widths)
    edge_losses = 1.0 - torch.exp(-((log_error / 0.10) ** 2))
    order = torch.argsort(weights, stable=True)
    order_losses = []
    for left, right in zip(order[:-1].tolist(), order[1:].tolist()):
        if weights[left] == weights[right]:
            continue
        argument = float((torch.log(widths[left]) - torch.log(widths[right])) / 0.02)
        argument = max(-60.0, min(60.0, argument))
        order_losses.append(1.0 / (1.0 + math.exp(-argument)))
    per_edge = snap_unit(float(torch.mean(edge_losses)))
    if order_losses:
        order_loss = snap_unit(sum(order_losses) / len(order_losses))
        defect = snap_unit(0.75 * per_edge + 0.25 * order_loss)
    else:
        order_loss = 0.0
        defect = per_edge
    return value_result(
        defect,
        {"U37.ell_e": per_edge, "U37.ell_ord": order_loss},
        {
            "target_widths": tuple(float(value) for value in target_widths),
            "derived_widths": tuple(float(value) for value in widths),
            "concordance_count": len(order_losses),
        },
    )


def _log_linear_targets(weights: torch.Tensor, knots: torch.Tensor) -> torch.Tensor:
    """Evaluate a positive piecewise-linear map in log-weight/log-width space.

    Parameters
    ----------
    weights : torch.Tensor
        Positive declared edge weights with shape ``[E]``.
    knots : torch.Tensor
        Positive monotone ``(weight, width)`` knots with shape ``[K, 2]``.

    Returns
    -------
    torch.Tensor
        Target widths with shape ``[E]``.
    """

    if knots.shape[0] == 1:
        return torch.full_like(weights, float(knots[0, 1]))
    log_weights = torch.log(weights)
    log_knot_weights = torch.log(knots[:, 0])
    log_knot_widths = torch.log(knots[:, 1])
    right = torch.searchsorted(log_knot_weights, log_weights, right=True)
    right = torch.clamp(right, min=1, max=knots.shape[0] - 1)
    left = right - 1
    fraction = (log_weights - log_knot_weights[left]) / (
        log_knot_weights[right] - log_knot_weights[left]
    )
    fraction = torch.clamp(fraction, 0.0, 1.0)
    return torch.exp(
        log_knot_widths[left] + fraction * (log_knot_widths[right] - log_knot_widths[left])
    )


def replace_edge_weights(scene: Scene, weights: tuple[float, ...]) -> Scene:
    """Return a scene with normalized graph-side distance costs.

    Parameters
    ----------
    scene : Scene
        Validated weighted scene.
    weights : tuple[float, ...]
        Positive normalized costs.

    Returns
    -------
    Scene
        Shallow immutable copy with replaced GraphSemantics weights.
    """

    from dataclasses import replace

    return replace(scene, graph=replace(scene.graph, edge_weights=weights))
