"""Pure top-level RULER V4 scoring entry point."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from types import MappingProxyType
from typing import Mapping, Optional, Tuple

from dagua.eval.ruler_v4.composition import (
    CompositionProfile,
    CompositionResult,
    SubtermContribution,
    compose,
)
from dagua.eval.ruler_v4.contracts import CONTRACTS
from dagua.eval.ruler_v4.headline import HeadlineProfile, HeadlineResult, ordinal_headline
from dagua.eval.ruler_v4.registry import evaluate_facet, validate_registry
from dagua.eval.ruler_v4.scene import FacetResult, ResultState, Scene
from dagua.eval.ruler_v4.weights import PriorMassDisclosure, WeightTable


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
    """

    composition: CompositionProfile
    headline: HeadlineProfile
    measurement_version: str
    policy_version: str
    alpha_grid_index: Optional[int] = None
    crossing_gamma: Optional[float] = None
    crossing_tail_weight: Optional[float] = None

    def __post_init__(self) -> None:
        """Require both frozen artifact version strings.

        Raises
        ------
        ValueError
            If either artifact version is empty.
        """

        if not self.measurement_version or not self.policy_version:
            raise ValueError("measurement and policy versions must be nonempty")


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


def _evaluate_static_facets(scene: Scene, profiles: ScoringProfiles) -> Mapping[str, FacetResult]:
    """Evaluate all 45 contracts under one explicit profile selection.

    Parameters
    ----------
    scene : Scene
        Validated static scene.
    profiles : ScoringProfiles
        Facet and scoring profile parameters.

    Returns
    -------
    mapping[str, FacetResult]
        Contract-order independent facet mapping.
    """

    results = {}
    for facet_id in CONTRACTS:
        results[facet_id] = evaluate_facet(
            facet_id,
            scene,
            alpha_grid_index=profiles.alpha_grid_index,
            gamma=profiles.crossing_gamma if facet_id == "U07" else None,
            lambda_T=profiles.crossing_tail_weight if facet_id == "U07" else None,
        )
    return results


def score(scene: Scene, weight_table: WeightTable, profiles: ScoringProfiles) -> ScoreResult:
    """Score one validated scene deterministically with no I/O.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene from Phase-1 ingestion.
    weight_table : WeightTable
        Complete explicit per-sub-term weights and provenance.
    profiles : ScoringProfiles
        Frozen facet, composition, headline, measurement, and policy profiles.

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
    facet_results = _evaluate_static_facets(scene, profiles)
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


def score_scene(scene: Scene, weight_table: WeightTable, profiles: ScoringProfiles) -> ScoreResult:
    """Call :func:`score` using the explicit scene-oriented name.

    Parameters
    ----------
    scene : Scene
        Validated canonical scene.
    weight_table : WeightTable
        Complete explicit per-sub-term weights.
    profiles : ScoringProfiles
        Frozen scoring profiles and version ids.

    Returns
    -------
    ScoreResult
        Deterministic structured score.
    """

    return score(scene, weight_table, profiles)
