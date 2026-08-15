"""Pure top-level RULER V4 scoring entry point."""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from dagua.eval.ruler_v4.composition import (
    CompositionFamily,
    CompositionProfile,
    CompositionResult,
    SubtermContribution,
    compose,
)
from dagua.eval.ruler_v4.contracts import CONTRACTS
from dagua.eval.ruler_v4.headline import HeadlineProfile, HeadlineResult, ordinal_headline
from dagua.eval.ruler_v4.registry import evaluate_facet, validate_registry
from dagua.eval.ruler_v4.scene import FacetResult, ResultState, Scene, TemporalScene
from dagua.eval.ruler_v4.weight_table import ParameterProvenance, PriorMassDisclosure, WeightTable


class OutputType(str, Enum):
    """A17 provenance classes carried by the structured result."""

    TYPE_R = "TYPE-R"
    TYPE_M = "TYPE-M"


@dataclass(frozen=True)
class ScoringProfiles:
    """Bind all versioned, explicit scoring profiles for one call.

    Parameters
    ----------
    composition : CompositionProfile
        Across-group loss family and parameters.
    headline : HeadlineProfile
        Ordinal index mapping parameters.
    measurement_version : str
        V4-MEASUREMENT artifact version.
    policy_version : str
        V4-POLICY artifact version.
    alpha_grid_index : int or None
        Shared selected U17-family manifest grid row.
    crossing_gamma : float or None
        Optional U07 fitted crossing-severity scale.
    crossing_tail_weight : float or None
        Optional U07 fitted tail weight.
    parameter_provenance : mapping[str, ParameterProvenance]
        A18 provenance classification for every active score-visible profile
        scalar. The score path refuses an unclassified scalar, and a scalar
        classified as fitted must resolve to an identity in the weight
        table's dof ledger (V4_SPEC_r4 3.6 counting rule).
    """

    composition: CompositionProfile
    headline: HeadlineProfile
    measurement_version: str
    policy_version: str
    alpha_grid_index: Optional[int] = None
    crossing_gamma: Optional[float] = None
    crossing_tail_weight: Optional[float] = None
    parameter_provenance: Mapping[str, ParameterProvenance] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Require both frozen artifact version strings.

        Raises
        ------
        ValueError
            If either artifact version is empty or a provenance key is blank.
        """

        if not self.measurement_version or not self.policy_version:
            raise ValueError("measurement and policy versions must be nonempty")
        provenance = dict(self.parameter_provenance)
        if any(not name for name in provenance):
            raise ValueError("provenance keys must be nonempty scalar names")
        object.__setattr__(self, "parameter_provenance", MappingProxyType(provenance))

    def score_visible_scalars(self) -> Tuple[str, ...]:
        """Enumerate the active score-visible scalar parameters of this call.

        Returns
        -------
        tuple[str, ...]
            Stable names for every independently adjustable scalar that can
            move the published score under these profiles.
        """

        names = []
        if self.composition.family is CompositionFamily.P_MEAN:
            names.append("composition.power")
        else:
            names.append("composition.bottleneck_mix")
            names.append("composition.bottleneck_temperature")
            names.extend(
                f"composition.group_allowances.{group}"
                for group in sorted(self.composition.group_allowances)
            )
        names.append("headline.index_span")
        names.append("headline.loss_scale")
        if self.alpha_grid_index is not None:
            names.append("alpha_grid_index")
        if self.crossing_gamma is not None:
            names.append("crossing_gamma")
        if self.crossing_tail_weight is not None:
            names.append("crossing_tail_weight")
        return tuple(names)


@dataclass(frozen=True)
class TypeRContext:
    """Carry the facet-free input/render context legal before the hash gate.

    Parameters
    ----------
    output_type : OutputType
        Always ``TYPE-R``.
    profile_hash : str
        Canonical graph/profile/style hash.
    observation_profile : str
        Active immutable profile name.
    node_count : int
        Canonical graph node count.
    edge_count : int
        Canonical graph edge count.
    """

    output_type: OutputType
    profile_hash: str
    observation_profile: str
    node_count: int
    edge_count: int


@dataclass(frozen=True)
class FacetBreakdown:
    """Publish one facet result and all declared term attributions.

    Parameters
    ----------
    facet_id : str
        Frozen facet id.
    result : FacetResult
        Independent Phase-1 result.
    contributions : tuple[SubtermContribution, ...]
        Composition rows belonging to the facet.
    """

    facet_id: str
    result: FacetResult
    contributions: Tuple[SubtermContribution, ...]


@dataclass(frozen=True)
class TypeMMeasurement:
    """Carry all manifest-dependent values computed after the hash gate.

    Parameters
    ----------
    output_type : OutputType
        Always ``TYPE-M``.
    composition : CompositionResult
        Exact pre-map loss and attribution.
    headline : HeadlineResult
        Ordinal within-graph index.
    prior_mass : PriorMassDisclosure
        PM-1 profile disclosure.
    facets : mapping[str, FacetBreakdown]
        Stable full 45-facet table, including NA and diagnostics.
    """

    output_type: OutputType
    composition: CompositionResult
    headline: HeadlineResult
    prior_mass: PriorMassDisclosure
    facets: Mapping[str, FacetBreakdown]


@dataclass(frozen=True)
class ScoreResult:
    """Return separated TYPE-R context and TYPE-M measurement payloads.

    Parameters
    ----------
    type_r : TypeRContext
        Input/render-only structured context.
    type_m : TypeMMeasurement
        Manifest-dependent facet, composition, and headline values.
    measurement_version : str
        Frozen measurement artifact version.
    policy_version : str
        Frozen policy artifact version.
    """

    type_r: TypeRContext
    type_m: TypeMMeasurement
    measurement_version: str
    policy_version: str


def validate_parameter_provenance(profiles: ScoringProfiles, weight_table: WeightTable) -> None:
    """Refuse profile scalars that hide from the A18 dof account.

    Every active score-visible profile scalar must carry a provenance class,
    and every fitted one must resolve to an identity the weight table already
    declares, so ``DofAccount.used`` covers the whole score path
    (V4_SPEC_r4 3.6: independently adjustable score-visible scalar parameters
    across ALL fitting stages).

    Parameters
    ----------
    profiles : ScoringProfiles
        Profile set about to be scored.
    weight_table : WeightTable
        Table whose dof ledger must absorb the fitted profile scalars.

    Raises
    ------
    ValueError
        If a scalar is unclassified, a classification names an inactive
        scalar, or a fitted identity is absent from the table's ledger.
    """

    active = profiles.score_visible_scalars()
    declared = set(profiles.parameter_provenance)
    missing = sorted(set(active) - declared)
    if missing:
        raise ValueError(f"score-visible profile scalars lack provenance: {missing}")
    unknown = sorted(declared - set(active))
    if unknown:
        raise ValueError(f"provenance declared for inactive profile scalars: {unknown}")
    ledger = {
        name
        for name in (
            *(entry.fitted_parameter for entry in weight_table.entries),
            *weight_table.other_fitted_parameters,
        )
        if name is not None
    }
    unledgered = sorted(
        name
        for name in active
        if (identity := profiles.parameter_provenance[name].fitted_identity) is not None
        and identity not in ledger
    )
    if unledgered:
        raise ValueError(
            f"fitted profile scalars missing from the weight-table dof ledger: {unledgered}"
        )


def _evaluate_static_facets(
    scene: Scene,
    profiles: ScoringProfiles,
    temporal_scene: Optional[TemporalScene] = None,
) -> Mapping[str, FacetResult]:
    """Evaluate all 45 contracts under one explicit profile selection.

    Parameters
    ----------
    scene : Scene
        Validated static scene.
    profiles : ScoringProfiles
        Facet and scoring profile parameters.
    temporal_scene : TemporalScene or None
        Validated temporal scene routed to U40 so the pure entrypoint can
        publish the full 45-row table; U40 is structurally NA without it.

    Returns
    -------
    mapping[str, FacetResult]
        Contract-order independent facet mapping.
    """

    results = {}
    for facet_id in CONTRACTS:
        if facet_id == "U40" and temporal_scene is not None:
            results[facet_id] = evaluate_facet(facet_id, temporal_scene)
            continue
        results[facet_id] = evaluate_facet(
            facet_id,
            scene,
            alpha_grid_index=profiles.alpha_grid_index,
            gamma=profiles.crossing_gamma if facet_id == "U07" else None,
            lambda_T=profiles.crossing_tail_weight if facet_id == "U07" else None,
        )
    return results


def score(
    scene: Scene,
    weight_table: WeightTable,
    profiles: ScoringProfiles,
    temporal_scene: Optional[TemporalScene] = None,
) -> ScoreResult:
    """Score one validated scene deterministically with no I/O.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene from Phase-1 ingestion.
    weight_table : WeightTable
        Complete explicit per-sub-term weights and provenance.
    profiles : ScoringProfiles
        Frozen facet, composition, headline, measurement, and policy profiles.
    temporal_scene : TemporalScene or None
        Validated temporal scene for U40 (mental-map continuity), when
        ingestion produced one. Without it U40 publishes its structural NA.

    Returns
    -------
    ScoreResult
        Full 45-facet breakdown, exact composition, ordinal headline, and
        separated TYPE-R/TYPE-M payloads.

    Raises
    ------
    ValueError
        If registry/weight invariants fail or a facet returns INVALID.
    """

    validate_registry()
    weight_table.validate_for_contracts()
    validate_parameter_provenance(profiles, weight_table)
    facet_results = _evaluate_static_facets(scene, profiles, temporal_scene)
    invalid = {
        facet_id: result.reason
        for facet_id, result in facet_results.items()
        if result.state is ResultState.INVALID
    }
    if invalid:
        raise ValueError(f"validated scene produced invalid facet results: {invalid}")
    composition = compose(facet_results, weight_table, profiles.composition)
    applicable_subterms = frozenset(
        subterm_id
        for result in facet_results.values()
        if result.state is ResultState.VALUE
        for subterm_id in result.subterms
    )
    prior_mass = weight_table.prior_mass_disclosure(applicable_subterms)
    headline = ordinal_headline(composition.l_total, profiles.headline, prior_mass)
    contributions = {
        facet_id: FacetBreakdown(
            facet_id=facet_id,
            result=facet_results[facet_id],
            contributions=tuple(row for row in composition.subterms if row.facet_id == facet_id),
        )
        for facet_id in CONTRACTS
    }
    type_r = TypeRContext(
        output_type=OutputType.TYPE_R,
        profile_hash=scene.profile_hash,
        observation_profile=scene.profile.name,
        node_count=scene.node_count,
        edge_count=scene.edge_count,
    )
    type_m = TypeMMeasurement(
        output_type=OutputType.TYPE_M,
        composition=composition,
        headline=headline,
        prior_mass=prior_mass,
        facets=MappingProxyType(contributions),
    )
    return ScoreResult(
        type_r=type_r,
        type_m=type_m,
        measurement_version=profiles.measurement_version,
        policy_version=profiles.policy_version,
    )


def score_scene(
    scene: Scene,
    weight_table: WeightTable,
    profiles: ScoringProfiles,
    temporal_scene: Optional[TemporalScene] = None,
) -> ScoreResult:
    """Call :func:`score` using the explicit scene-oriented name.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene.
    weight_table : WeightTable
        Complete explicit per-sub-term weights.
    profiles : ScoringProfiles
        Frozen scoring profiles and version ids.
    temporal_scene : TemporalScene or None
        Validated temporal scene for U40, when available.

    Returns
    -------
    ScoreResult
        Deterministic structured score.
    """

    return score(scene, weight_table, profiles, temporal_scene)
