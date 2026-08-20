"""Constrained pairwise likelihood for RULER V4 weight fitting."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import Mapping, Optional, Sequence, Tuple

import torch

from dagua.eval.ruler_v4.fit.bank import SplitPurpose
from dagua.eval.ruler_v4.fit.rescoring import RescoredPair
from dagua.eval.ruler_v4.weight_table import (
    ASSIGNABLE_DOF_BUCKETS,
    DOF_ALLOCATION,
    FITTED_DOF_CAP,
    GATE_DIAGNOSTIC_FACETS,
    REQUIRED_PRIOR_FLOOR_FACETS,
    WeightTable,
)

_MIN_PROBABILITY = 1.0e-12
_FROZEN_PRIOR_STRENGTH = 2.0


@dataclass(frozen=True)
class WeightParameter:
    """Declare one independently adjustable outer-weight scalar.

    Parameters
    ----------
    name : str
        Stable fitted-parameter identity from the dof ledger.
    bucket : str
        Assignable A18 bucket, normally ``universal`` or ``semantic``.
    prior : float
        Positive literature/preregistered prior.
    subterm_coefficients : mapping[str, float]
        Fixed nonnegative ratios from this scalar to scored sub-term masses.
    facet_ids : tuple[str, ...]
        Facets controlled by the scalar.
    lower, upper : float or None
        Explicit bounds. Defaults are the spec's ``[0.25x, 4x]`` prior range.
    """

    name: str
    bucket: str
    prior: float
    subterm_coefficients: Mapping[str, float]
    facet_ids: Tuple[str, ...]
    lower: Optional[float] = None
    upper: Optional[float] = None

    def __post_init__(self) -> None:
        """Freeze coefficients and enforce local parameter bounds.

        Raises
        ------
        ValueError
            If identities, ratios, facets, or bounds are invalid.
        """

        coefficients = {key: float(value) for key, value in self.subterm_coefficients.items()}
        facets = tuple(self.facet_ids)
        lower = self.prior * 0.25 if self.lower is None else float(self.lower)
        upper = self.prior * 4.0 if self.upper is None else float(self.upper)
        if not self.name or self.bucket not in ASSIGNABLE_DOF_BUCKETS:
            raise ValueError("weight parameters require a named assignable A18 bucket")
        if not math.isfinite(self.prior) or self.prior <= 0.0:
            raise ValueError("weight priors must be finite and positive")
        if not coefficients or any(
            not key or not math.isfinite(value) or value < 0.0
            for key, value in coefficients.items()
        ):
            raise ValueError("sub-term coefficients must be named, finite, and nonnegative")
        if math.fsum(coefficients.values()) <= 0.0:
            raise ValueError("a weight parameter must control positive mass")
        if not facets or any(not facet_id for facet_id in facets):
            raise ValueError("weight parameters require owning facets")
        if any(facet_id in GATE_DIAGNOSTIC_FACETS for facet_id in facets):
            raise ValueError("DIAG weight-0 facets cannot enter the fitting plan")
        if not (math.isfinite(lower) and math.isfinite(upper) and 0.0 < lower <= upper):
            raise ValueError("weight bounds must be finite, positive, and ordered")
        if not lower <= self.prior <= upper:
            raise ValueError("weight prior must lie inside its bounds")
        object.__setattr__(self, "subterm_coefficients", MappingProxyType(coefficients))
        object.__setattr__(self, "facet_ids", facets)
        object.__setattr__(self, "lower", lower)
        object.__setattr__(self, "upper", upper)


@dataclass(frozen=True)
class FittingPlan:
    """Bind the fitted weights to the complete A18 dof declaration.

    Parameters
    ----------
    weights : tuple[WeightParameter, ...]
        Adjustable outer-weight parameters in design-matrix order.
    other_fitted_parameter_buckets : mapping[str, str]
        Group-model and aggregation identities, including JND-HET and lapse.
    prior_floors : mapping[str, float]
        Frozen positive floors for any traceability facets present in ``weights``.
    prior_strength : float
        Frozen inverse prior variance on the ``log_4`` scale.
    """

    weights: Tuple[WeightParameter, ...]
    other_fitted_parameter_buckets: Mapping[str, str] = field(default_factory=dict)
    prior_floors: Mapping[str, float] = field(default_factory=dict)
    prior_strength: float = field(default=_FROZEN_PRIOR_STRENGTH, init=False)

    def __post_init__(self) -> None:
        """Freeze declarations and refuse off-ledger degrees of freedom.

        Raises
        ------
        ValueError
            If identities collide, buckets exceed A18, or a prior floor is violated.
        """

        weights = tuple(self.weights)
        others = dict(self.other_fitted_parameter_buckets)
        floors = {key: float(value) for key, value in self.prior_floors.items()}
        names = [parameter.name for parameter in weights]
        if len(set(names)) != len(names) or set(names) & set(others):
            raise ValueError("fitted parameter identities must be globally unique")
        facet_owners = [facet_id for parameter in weights for facet_id in parameter.facet_ids]
        if len(facet_owners) != len(set(facet_owners)):
            raise ValueError("each facet may be controlled by only one fitted scalar")
        if any(bucket not in ASSIGNABLE_DOF_BUCKETS for bucket in others.values()):
            raise ValueError("other fitted parameters require assignable A18 buckets")
        usage = {bucket: 0 for bucket in DOF_ALLOCATION}
        for parameter in weights:
            usage[parameter.bucket] += 1
        for bucket in others.values():
            usage[bucket] += 1
        over = sorted(
            bucket for bucket in ASSIGNABLE_DOF_BUCKETS if usage[bucket] > DOF_ALLOCATION[bucket]
        )
        if over:
            raise ValueError(f"A18 allocation buckets exceeded: {over}")
        if len(weights) + len(others) > FITTED_DOF_CAP - DOF_ALLOCATION["unspent"]:
            raise ValueError("fit consumes the permanently UNSPENT A18 degree of freedom")
        if any(
            facet_id not in REQUIRED_PRIOR_FLOOR_FACETS or not math.isfinite(value) or value <= 0.0
            for facet_id, value in floors.items()
        ):
            raise ValueError("prior floors are positive and limited to U12, U13, and U34")
        for parameter in weights:
            required = set(parameter.facet_ids) & REQUIRED_PRIOR_FLOOR_FACETS
            if required and len(parameter.facet_ids) != 1:
                raise ValueError(
                    "prior-floor validation requires one fitted scalar per traceability facet"
                )
            missing = sorted(required - set(floors))
            if missing:
                raise ValueError(f"fitted traceability facets require prior floors: {missing}")
            for facet_id in set(parameter.facet_ids) & set(floors):
                effective_lower = float(parameter.lower) * math.fsum(
                    parameter.subterm_coefficients.values()
                )
                if effective_lower < floors[facet_id]:
                    raise ValueError(f"{facet_id} fitted bound falls below its prior floor")
        object.__setattr__(self, "weights", weights)
        object.__setattr__(self, "other_fitted_parameter_buckets", MappingProxyType(others))
        object.__setattr__(self, "prior_floors", MappingProxyType(floors))

    @property
    def parameter_names(self) -> Tuple[str, ...]:
        """Return outer-weight identities in design order.

        Returns
        -------
        tuple[str, ...]
            Stable fitted outer-weight names.
        """

        return tuple(parameter.name for parameter in self.weights)


@dataclass(frozen=True)
class FitPair:
    """Hold one P-mean pair and its complete ordered-probit label.

    Parameters
    ----------
    numerator_a, numerator_b : tuple[float, ...]
        Per-parameter A/B contributions ``sum(ratio * defect**power)``.
    mass_coefficients : tuple[float, ...]
        Per-parameter applicable-mass coefficients.
    outcome : int
        ``-1`` for A, ``0`` for tie, and ``1`` for B.
    graded_verdict : int
        Required original A13 verdict in ``[-3, 3]``.
    confidence : int or None
        Original A13 confidence in ``[1, 3]`` when banked.
    fixed_numerator_a, fixed_numerator_b : float
        Contributions of non-fitted positive-mass sub-terms.
    fixed_mass : float
        Applicable non-fitted mass common to both sides.
    composition_power : float
        Frozen P-mean exponent, at least one.
    jnd : float
        Positive tie half-band on the score-difference scale.
    lapse_rate : float
        Uniform seven-category lapse mixture in ``[0, 1)``.
    primary_class, size_band, graph_hash, generator_family, era, instrument_hash : str
        Frozen diagnostic strata.
    observation_profile : str
        Opaque observation-profile likelihood stratum.
    purpose : SplitPurpose
        Frozen A15 consumption purpose.
    is_replication : bool
        Whether the judgment belongs to the cross-session replication line.
    base_pair_id, session_id, blind_id_a, blind_id_b : str
        Replication and displayed-order provenance.
    synthetic : bool
        Whether the row was constructed by the typed synthetic fixture factory.
    """

    numerator_a: Tuple[float, ...]
    numerator_b: Tuple[float, ...]
    mass_coefficients: Tuple[float, ...]
    outcome: int
    graded_verdict: int
    confidence: Optional[int]
    fixed_numerator_a: float
    fixed_numerator_b: float
    fixed_mass: float
    composition_power: float
    jnd: float
    lapse_rate: float
    primary_class: str
    size_band: str
    graph_hash: str
    generator_family: str
    era: str
    instrument_hash: str
    observation_profile: str
    purpose: SplitPurpose
    is_replication: bool
    base_pair_id: str
    session_id: str
    blind_id_a: str
    blind_id_b: str
    synthetic: bool

    def __post_init__(self) -> None:
        """Validate feature dimensions and likelihood constants.

        Raises
        ------
        ValueError
            If dimensions, values, or the outcome are invalid.
        """

        numerator_a = tuple(float(value) for value in self.numerator_a)
        numerator_b = tuple(float(value) for value in self.numerator_b)
        masses = tuple(float(value) for value in self.mass_coefficients)
        if (
            not numerator_a
            or len(numerator_a) != len(numerator_b)
            or len(numerator_a) != len(masses)
        ):
            raise ValueError("pair feature vectors must have one equal nonzero dimension")
        values = (
            *numerator_a,
            *numerator_b,
            *masses,
            self.fixed_numerator_a,
            self.fixed_numerator_b,
            self.fixed_mass,
        )
        if any(not math.isfinite(value) or value < 0.0 for value in values):
            raise ValueError("P-mean numerators and masses must be finite and nonnegative")
        if self.outcome not in (-1, 0, 1):
            raise ValueError("pair outcome must be -1, 0, or 1")
        if self.graded_verdict not in range(-3, 4):
            raise ValueError("graded verdict must lie in [-3, 3]")
        graded_verdict = self.graded_verdict
        graded_outcome = 0 if graded_verdict == 0 else 1 if graded_verdict > 0 else -1
        if graded_outcome != self.outcome:
            raise ValueError("graded verdict sign disagrees with the fitting outcome")
        if self.confidence is not None and self.confidence not in (1, 2, 3):
            raise ValueError("A13 confidence must lie in [1, 3]")
        if not math.isfinite(self.composition_power) or self.composition_power < 1.0:
            raise ValueError("composition power must be finite and at least one")
        if not math.isfinite(self.jnd) or self.jnd <= 0.0:
            raise ValueError("JND must be finite and positive")
        if not math.isfinite(self.lapse_rate) or not 0.0 <= self.lapse_rate < 1.0:
            raise ValueError("lapse rate must lie in [0, 1)")
        if not isinstance(self.purpose, SplitPurpose):
            raise ValueError("pair purpose must be a SplitPurpose")
        if not isinstance(self.synthetic, bool):
            raise ValueError("pair synthetic provenance must be boolean")
        if not all(
            (
                self.observation_profile,
                self.base_pair_id,
                self.session_id,
                self.blind_id_a,
                self.blind_id_b,
            )
        ):
            raise ValueError("pair likelihood and replication identities must be nonempty")
        object.__setattr__(self, "numerator_a", numerator_a)
        object.__setattr__(self, "numerator_b", numerator_b)
        object.__setattr__(self, "mass_coefficients", masses)
        object.__setattr__(self, "graded_verdict", graded_verdict)


def synthetic_fit_pair(
    numerator_a: Tuple[float, ...],
    numerator_b: Tuple[float, ...],
    mass_coefficients: Tuple[float, ...],
    graded_verdict: int,
    *,
    confidence: Optional[int] = None,
    fixed_numerator_a: float = 0.0,
    fixed_numerator_b: float = 0.0,
    fixed_mass: float = 0.0,
    composition_power: float = 1.0,
    jnd: float = 0.1,
    lapse_rate: float = 0.0,
    primary_class: str = "synthetic",
    size_band: str = "synthetic",
    graph_hash: str = "synthetic",
    generator_family: str = "synthetic",
    era: str = "synthetic",
    instrument_hash: str = "synthetic",
    observation_profile: str = "synthetic",
    is_replication: bool = False,
    base_pair_id: str = "synthetic",
    session_id: str = "synthetic",
    blind_id_a: str = "synthetic-a",
    blind_id_b: str = "synthetic-b",
) -> FitPair:
    """Build one explicitly synthetic ordered-response row.

    Parameters
    ----------
    numerator_a, numerator_b, mass_coefficients : tuple[float, ...]
        Synthetic P-mean feature vectors.
    graded_verdict : int
        Original seven-point response in ``[-3, 3]``; its sign supplies the
        three-way reporting projection.
    confidence : int or None, optional
        Diagnostic-only A13 confidence.
    fixed_numerator_a, fixed_numerator_b, fixed_mass : float, optional
        Synthetic non-fitted P-mean contributions.
    composition_power, jnd, lapse_rate : float, optional
        Synthetic likelihood constants.
    primary_class, size_band, graph_hash, generator_family : str, optional
        Synthetic diagnostic strata.
    era, instrument_hash, observation_profile : str, optional
        Synthetic non-poolable likelihood identity.
    is_replication : bool, default=False
        Whether this fixture row exercises replication behavior.
    base_pair_id, session_id, blind_id_a, blind_id_b : str, optional
        Synthetic replication provenance.

    Returns
    -------
    FitPair
        Fully populated synthetic-only objective row.
    """

    outcome = 0 if graded_verdict == 0 else 1 if graded_verdict > 0 else -1
    return FitPair(
        numerator_a=numerator_a,
        numerator_b=numerator_b,
        mass_coefficients=mass_coefficients,
        outcome=outcome,
        graded_verdict=graded_verdict,
        confidence=confidence,
        fixed_numerator_a=fixed_numerator_a,
        fixed_numerator_b=fixed_numerator_b,
        fixed_mass=fixed_mass,
        composition_power=composition_power,
        jnd=jnd,
        lapse_rate=lapse_rate,
        primary_class=primary_class,
        size_band=size_band,
        graph_hash=graph_hash,
        generator_family=generator_family,
        era=era,
        instrument_hash=instrument_hash,
        observation_profile=observation_profile,
        purpose=SplitPurpose.FIT,
        is_replication=is_replication,
        base_pair_id=base_pair_id,
        session_id=session_id,
        blind_id_a=blind_id_a,
        blind_id_b=blind_id_b,
        synthetic=True,
    )


def fit_pairs_from_rescoring(
    pairs: Sequence[RescoredPair],
    plan: FittingPlan,
    weight_table: WeightTable,
    composition_power: float,
    jnd_by_cell: Mapping[Tuple[str, str], float],
    lapse_rate: float,
) -> Tuple[FitPair, ...]:
    """Build exact P-mean pair rows from fresh TYPE-M sub-term values.

    Each fitted scalar replaces the masses of its declared sub-terms at fixed
    ratios. Every other applicable positive-mass non-DIAG row remains in the
    fixed P-mean numerator, preserving the production score's NA normalization.

    Parameters
    ----------
    pairs : sequence[RescoredPair]
        Judgments joined to fresh ``score()`` measurements.
    plan : FittingPlan
        Constrained outer-weight plan.
    weight_table : WeightTable
        Complete frozen/prior table defining fixed mass.
    composition_power : float
        Frozen P-mean exponent.
    jnd_by_cell : mapping[tuple[str, str], float]
        JND keyed by ``(primary_class, size_band)``.
    lapse_rate : float
        Frozen/fitted uniform lapse mixture.

    Returns
    -------
    tuple[FitPair, ...]
        Objective-ready rows in input order.

    Raises
    ------
    ValueError
        If fitted sub-terms overlap, disagree with the table, or are unavailable.
    """

    weight_table.validate_for_contracts()
    claimed = {}
    actual_facets_by_parameter = {}
    for index, parameter in enumerate(plan.weights):
        actual_facets = set()
        for subterm_id in parameter.subterm_coefficients:
            if subterm_id in claimed:
                raise ValueError(f"fitted sub-term is controlled twice: {subterm_id}")
            if subterm_id not in weight_table.by_subterm:
                raise ValueError(f"fitted sub-term absent from weight table: {subterm_id}")
            entry = weight_table.by_subterm[subterm_id]
            if entry.diagnostic or entry.weight == 0.0:
                raise ValueError(f"diagnostic or weight-0 sub-term cannot be fitted: {subterm_id}")
            if entry.facet_id not in parameter.facet_ids:
                raise ValueError(
                    f"{subterm_id} belongs to {entry.facet_id}, not {parameter.facet_ids}"
                )
            if entry.fitted_parameter != parameter.name:
                raise ValueError(
                    f"{subterm_id} fitted identity {entry.fitted_parameter!r} "
                    f"does not match plan identity {parameter.name!r}"
                )
            actual_facets.add(entry.facet_id)
            claimed[subterm_id] = index
        if actual_facets != set(parameter.facet_ids):
            raise ValueError(
                f"parameter {parameter.name} facet declaration does not match its sub-terms"
            )
        actual_facets_by_parameter[parameter.name] = actual_facets
    table_fitted = {
        entry.subterm_id: entry.fitted_parameter
        for entry in weight_table.entries
        if entry.fitted_parameter is not None
    }
    claimed_identities = {
        subterm_id: plan.weights[index].name for subterm_id, index in claimed.items()
    }
    if table_fitted != claimed_identities:
        raise ValueError("fitting plan is not bijective with the weight-table fitted declaration")
    for parameter in plan.weights:
        table_bucket = weight_table.fitted_parameter_buckets.get(parameter.name)
        if table_bucket != parameter.bucket:
            raise ValueError(f"{parameter.name} A18 bucket disagrees with the weight table")
        for facet_id in actual_facets_by_parameter[parameter.name] & REQUIRED_PRIOR_FLOOR_FACETS:
            table_floor = weight_table.prior_floors.get(facet_id)
            plan_floor = plan.prior_floors.get(facet_id)
            if table_floor != plan_floor:
                raise ValueError(f"{facet_id} prior floor disagrees with the weight table")
            facet_ratio = math.fsum(
                ratio
                for subterm_id, ratio in parameter.subterm_coefficients.items()
                if weight_table.by_subterm[subterm_id].facet_id == facet_id
            )
            if float(parameter.lower) * facet_ratio < float(table_floor):
                raise ValueError(f"{facet_id} effective fitted mass falls below its prior floor")
    result = []
    for pair in pairs:
        a = pair.side_a.subterms
        b = pair.side_b.subterms
        if set(a) != set(b):
            raise ValueError("pair sides have different applicable TYPE-M sub-terms")
        numerator_a = []
        numerator_b = []
        masses = []
        for parameter in plan.weights:
            missing = sorted(set(parameter.subterm_coefficients) - set(a))
            if missing:
                raise ValueError(f"fitted sub-terms unavailable on pair: {missing}")
            numerator_a.append(
                math.fsum(
                    ratio * a[subterm_id] ** composition_power
                    for subterm_id, ratio in parameter.subterm_coefficients.items()
                )
            )
            numerator_b.append(
                math.fsum(
                    ratio * b[subterm_id] ** composition_power
                    for subterm_id, ratio in parameter.subterm_coefficients.items()
                )
            )
            masses.append(math.fsum(parameter.subterm_coefficients.values()))
        fixed_entries = tuple(
            entry
            for entry in weight_table.entries
            if entry.subterm_id in a
            and entry.subterm_id not in claimed
            and not entry.diagnostic
            and entry.weight > 0.0
        )
        cell = pair.judgment.primary_class, pair.judgment.size_band
        if cell not in jnd_by_cell:
            raise ValueError(f"JND missing for cell: {cell}")
        result.append(
            FitPair(
                numerator_a=tuple(numerator_a),
                numerator_b=tuple(numerator_b),
                mass_coefficients=tuple(masses),
                outcome=pair.judgment.outcome,
                graded_verdict=pair.judgment.verdict,
                confidence=pair.judgment.confidence,
                fixed_numerator_a=math.fsum(
                    entry.weight * a[entry.subterm_id] ** composition_power
                    for entry in fixed_entries
                ),
                fixed_numerator_b=math.fsum(
                    entry.weight * b[entry.subterm_id] ** composition_power
                    for entry in fixed_entries
                ),
                fixed_mass=math.fsum(entry.weight for entry in fixed_entries),
                composition_power=composition_power,
                jnd=float(jnd_by_cell[cell]),
                lapse_rate=lapse_rate,
                primary_class=pair.judgment.primary_class,
                size_band=pair.judgment.size_band,
                graph_hash=pair.judgment.graph_hash,
                generator_family=pair.judgment.generator_family,
                era=pair.judgment.era,
                instrument_hash=pair.judgment.instrument_hash,
                observation_profile=pair.judgment.observation_profile,
                purpose=pair.judgment.purpose,
                is_replication=pair.judgment.is_replication,
                base_pair_id=pair.judgment.base_pair_id,
                session_id=pair.judgment.session_id,
                blind_id_a=pair.judgment.blind_id_a,
                blind_id_b=pair.judgment.blind_id_b,
                synthetic=False,
            )
        )
    return tuple(result)


class PairwiseObjective:
    """Evaluate the frozen seven-category ordered-probit likelihood.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Objective rows from one explicit profile/instrument likelihood.
    plan : FittingPlan
        Dof-guarded parameter plan.
    dtype : torch.dtype, default=torch.float64
        Deterministic CPU calculation dtype.
    """

    def __init__(
        self,
        pairs: Sequence[FitPair],
        plan: FittingPlan,
        dtype: torch.dtype = torch.float64,
    ) -> None:
        """Materialize immutable objective tensors.

        Parameters
        ----------
        pairs : sequence[FitPair]
            Objective rows.
        plan : FittingPlan
            Fitting plan.
        dtype : torch.dtype, default=torch.float64
            Tensor dtype.

        Raises
        ------
        ValueError
            If the objective is empty, dimensionally inconsistent, or crosses
            observation-profile strata upstream.
        """

        rows = tuple(pairs)
        if not rows:
            raise ValueError("pairwise objective requires at least one row")
        if any(row.purpose is not SplitPurpose.FIT for row in rows):
            raise ValueError("a non-train row reached the FIT-ORD likelihood")
        if all(
            row.fixed_mass == 0.0 and row.fixed_numerator_a == 0.0 and row.fixed_numerator_b == 0.0
            for row in rows
        ):
            raise ValueError(
                "fitted P-mean is scale-invariant without fixed mass; "
                "absolute weights are not identifiable"
            )
        dimension = len(plan.weights)
        if dimension == 0 or any(len(row.numerator_a) != dimension for row in rows):
            raise ValueError("pair dimensions must match the fitting plan")
        strata = {(row.instrument_hash, row.era, row.observation_profile) for row in rows}
        if len(strata) != 1:
            raise ValueError("one likelihood may not cross instrument/era/profile/purpose strata")
        self.pairs = rows
        self.plan = plan
        self.dtype = dtype
        self._a = torch.tensor([row.numerator_a for row in rows], dtype=dtype)
        self._b = torch.tensor([row.numerator_b for row in rows], dtype=dtype)
        self._mass = torch.tensor([row.mass_coefficients for row in rows], dtype=dtype)
        self._fixed_a = torch.tensor([row.fixed_numerator_a for row in rows], dtype=dtype)
        self._fixed_b = torch.tensor([row.fixed_numerator_b for row in rows], dtype=dtype)
        self._fixed_mass = torch.tensor([row.fixed_mass for row in rows], dtype=dtype)
        self._power = torch.tensor([row.composition_power for row in rows], dtype=dtype)
        self._jnd = torch.tensor([row.jnd for row in rows], dtype=dtype)
        self._lapse = torch.tensor([row.lapse_rate for row in rows], dtype=dtype)
        self._verdict_indices = torch.tensor(
            [row.graded_verdict + 3 for row in rows], dtype=torch.int64
        )
        self._priors = torch.tensor([parameter.prior for parameter in plan.weights], dtype=dtype)

    def score_differences(self, weights: torch.Tensor) -> torch.Tensor:
        """Compute exact A-minus-B P-mean score differences.

        Parameters
        ----------
        weights : torch.Tensor
            Positive fitted weights with shape ``[P]``.

        Returns
        -------
        torch.Tensor
            Score differences with shape ``[J]``; positive favors B.
        """

        mass = self._fixed_mass + (self._mass * weights.unsqueeze(0)).sum(dim=1)
        if bool(torch.any(mass <= 0.0)):
            raise ValueError("candidate weights leave a pair with no applicable mass")
        value_a = (self._fixed_a + self._a.mv(weights)) / mass
        value_b = (self._fixed_b + self._b.mv(weights)) / mass
        score_a = torch.pow(torch.clamp(value_a, min=0.0), 1.0 / self._power)
        score_b = torch.pow(torch.clamp(value_b, min=0.0), 1.0 / self._power)
        return score_a - score_b

    def outcome_probabilities(self, weights: torch.Tensor) -> torch.Tensor:
        """Return ordered-probit probabilities for verdicts ``-3`` through ``+3``.

        Parameters
        ----------
        weights : torch.Tensor
            Positive fitted weights with shape ``[P]``.

        Returns
        -------
        torch.Tensor
            Probability matrix with shape ``[J, 7]`` in verdict order.
        """

        difference = self.score_differences(weights)
        cutpoints = self._jnd.unsqueeze(1) * torch.tensor(
            (-3.0, -2.0, -1.0, 1.0, 2.0, 3.0), dtype=self.dtype
        ).unsqueeze(0)
        cdf = 0.5 * (1.0 + torch.erf((cutpoints - difference.unsqueeze(1)) / math.sqrt(2.0)))
        zeros = torch.zeros((len(self.pairs), 1), dtype=self.dtype)
        ones = torch.ones((len(self.pairs), 1), dtype=self.dtype)
        boundaries = torch.cat((zeros, cdf, ones), dim=1)
        probabilities = boundaries[:, 1:] - boundaries[:, :-1]
        probabilities = (1.0 - self._lapse.unsqueeze(1)) * probabilities
        probabilities = probabilities + self._lapse.unsqueeze(1) / 7.0
        return torch.clamp(probabilities, min=_MIN_PROBABILITY, max=1.0)

    def directional_probabilities(self, weights: torch.Tensor) -> torch.Tensor:
        """Project ordered probabilities onto the A/tie/B reporting scale.

        Parameters
        ----------
        weights : torch.Tensor
            Positive fitted weights with shape ``[P]``.

        Returns
        -------
        torch.Tensor
            Probability matrix with shape ``[J, 3]`` in A/tie/B order.
        """

        ordered = self.outcome_probabilities(weights)
        return torch.stack(
            (ordered[:, :3].sum(dim=1), ordered[:, 3], ordered[:, 4:].sum(dim=1)),
            dim=1,
        )

    def negative_log_likelihood(self, weights: torch.Tensor) -> torch.Tensor:
        """Return mean ordered-probit negative log likelihood.

        Parameters
        ----------
        weights : torch.Tensor
            Positive fitted weights with shape ``[P]``.

        Returns
        -------
        torch.Tensor
            Scalar mean negative log likelihood.
        """

        probabilities = self.outcome_probabilities(weights)
        selected = probabilities.gather(1, self._verdict_indices.unsqueeze(1)).squeeze(1)
        return -torch.log(selected).mean()

    def loss(self, weights: torch.Tensor) -> torch.Tensor:
        """Return likelihood plus preregistered log-prior shrinkage.

        Parameters
        ----------
        weights : torch.Tensor
            Positive fitted weights with shape ``[P]``.

        Returns
        -------
        torch.Tensor
            Scalar regularized fitting objective.
        """

        nll = self.negative_log_likelihood(weights)
        shrinkage = torch.square(torch.log(weights / self._priors) / math.log(4.0)).sum()
        return nll + self.plan.prior_strength * shrinkage / len(self.pairs)


@dataclass(frozen=True)
class FitOrdLines:
    """Separate the train likelihood from its replication-only JND profile.

    Parameters
    ----------
    train : tuple[FitPair, ...]
        All informative and replication train-role presentations contributing
        to outer weights and lapse.
    replication : tuple[FitPair, ...]
        Train-role cross-session replication presentations contributing to the
        JND block.
    """

    train: Tuple[FitPair, ...]
    replication: Tuple[FitPair, ...]


def partition_fit_ord_lines(pairs: Sequence[FitPair]) -> FitOrdLines:
    """Validate and partition FIT-ORD's two non-transferable consumption lines.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Candidate ordered-response rows.

    Returns
    -------
    FitOrdLines
        Complete train line and its replication-only JND subset.

    Raises
    ------
    ValueError
        If the input is empty, contains a non-train row, or has no replication
        line for the profiled JND block.
    """

    rows = tuple(pairs)
    if not rows:
        raise ValueError("FIT-ORD requires nonempty train-role rows")
    if any(pair.purpose is not SplitPurpose.FIT for pair in rows):
        raise ValueError("a non-train row reached FIT-ORD")
    replication = tuple(pair for pair in rows if pair.is_replication)
    if not replication:
        raise ValueError("FIT-ORD requires a train-role replication line")
    return FitOrdLines(train=rows, replication=replication)
