"""Explicit P5-supplied weight-table, dof-ledger, and PM-1 machinery.

Split out of ``weights.py`` (the U35-U37 edge-weight facet module) so the
two meanings of "weight" -- declared edge weights and composite sub-term
masses -- no longer share one file, and so ``composition.py`` does not pull
the torch-backed facet stack to reach ``SubtermWeight``.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field
from types import MappingProxyType
from typing import DefaultDict, FrozenSet, Mapping, Optional, Tuple

FITTED_DOF_CAP = 20
PRIOR_MASS_DISCLOSURE_GATE = 0.15
REQUIRED_PRIOR_FLOOR_FACETS = frozenset({"U12", "U13", "U34"})
# Frozen manifest provenance vocabulary (MANIFEST.json provenance_class).
PROVENANCE_CLASSES = frozenset(
    {"contract_frozen", "preregistered_prior", "fitted", "controlled_stimulus"}
)
FITTED_PROVENANCE_CLASSES = frozenset({"fitted", "controlled_stimulus"})
# A18 sec 7 preregistered dof allocation (PILOT_GATES_REPORT: N_u 9 + UNSPENT 1
# + N_s 3 + N_g 4 + N_t 3 = 20). UNSPENT is reserved headroom, never assignable.
DOF_ALLOCATION = MappingProxyType(
    {
        "universal": 9,
        "unspent": 1,
        "semantic": 3,
        "group_model": 4,
        "aggregation": 3,
    }
)
ASSIGNABLE_DOF_BUCKETS = frozenset(DOF_ALLOCATION) - {"unspent"}
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
        Reporting rollup label. Score-inert under the shipped p-mean family
        (V4_SPEC_r4 3.3: no weight semantics of its own).
    weight : float
        Nonnegative composite mass supplied by the P5 fit or frozen prior.
    fitted_parameter : str or None
        Identity of the independently adjustable scalar that produced this
        mass. Reusing one identity across a fixed-ratio bundle counts one dof
        only when the internal ratios were fixed a priori (V4_SPEC_r4 3.6);
        the manifest ``fitted_dof_declaration`` cross-check is a P5 gate.
    prior_driven : bool
        Whether this mass enters PM-1's numerator for the active profile.
        Per-profile by convention: one table serves exactly one observation
        profile (a table reused across profiles would report one profile's
        provenance for all).
    diagnostic : bool
        Whether the term is carried at weight zero outside the headline.
    provenance_class : str or None
        Frozen manifest provenance of this mass (``contract_frozen``,
        ``preregistered_prior``, ``fitted``, or ``controlled_stimulus``).
        Optional only while a table is under construction: the contract gate
        refuses any positive-mass or fitted-identity entry without one, the
        same fail-closed rule ``ScoringProfiles.parameter_provenance``
        applies to profile scalars (V4_SPEC_r4 3.6).
    """

    subterm_id: str
    facet_id: str
    group: str
    weight: float
    fitted_parameter: Optional[str] = None
    prior_driven: bool = False
    diagnostic: bool = False
    provenance_class: Optional[str] = None


@dataclass(frozen=True)
class ParameterProvenance:
    """Classify one score-visible profile scalar for the A18 dof ledger.

    Parameters
    ----------
    provenance_class : str
        Frozen manifest provenance vocabulary member.
    fitted_identity : str or None
        Dof-ledger identity, required exactly when the class is ``fitted`` or
        ``controlled_stimulus``. The identity must be declared in the weight
        table's ``other_fitted_parameters`` so the scalar enters the account.
    """

    provenance_class: str
    fitted_identity: Optional[str] = None

    def __post_init__(self) -> None:
        """Validate the class and its identity obligation.

        Raises
        ------
        ValueError
            If the class is unknown or the identity obligation is violated.
        """

        if self.provenance_class not in PROVENANCE_CLASSES:
            raise ValueError(f"unknown provenance class: {self.provenance_class}")
        if self.provenance_class in FITTED_PROVENANCE_CLASSES:
            if not self.fitted_identity:
                raise ValueError("fitted provenance requires a dof-ledger identity")
        elif self.fitted_identity is not None:
            raise ValueError("non-fitted provenance must not carry a dof identity")


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
    bucket_usage : mapping[str, int]
        Identities spent per preregistered A18 allocation bucket.
    unassigned_identities : tuple[str, ...]
        Fitted identities carrying no declared allocation bucket.
    within_buckets : bool
        Whether every identity has a bucket and no A18 bucket is exceeded.
    """

    d_power: int
    allowed: int
    used: int
    remaining: int
    within_cap: bool
    bucket_usage: Mapping[str, int]
    unassigned_identities: Tuple[str, ...]
    within_buckets: bool


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
        default weights. One table serves exactly one observation profile
        (PM-1 provenance is per-profile). The contract gate refuses any
        positive-mass or fitted entry whose provenance class is undeclared, so
        the mass surface cannot hide from the A18 dof account by omission.
    d_power : int
        Information-limited fitted capacity used by the A18 cap formula.
    prior_floors : mapping[str, float]
        Positive facet-level floors for U12, U13, and U34.
    other_fitted_parameters : tuple[str, ...]
        Score-visible fitted scalars outside sub-term masses, including group,
        aggregation, and headline-model parameters. The score path refuses a
        profile whose fitted scalars do not resolve into this ledger.
    fitted_parameter_buckets : mapping[str, str]
        Preregistered A18 allocation bucket per fitted identity. Required for
        every identity before the table can validate for contracts; the
        ``unspent`` reserve is never assignable.
    """

    entries: Tuple[SubtermWeight, ...]
    d_power: int
    prior_floors: Mapping[str, float] = field(default_factory=dict)
    other_fitted_parameters: Tuple[str, ...] = ()
    fitted_parameter_buckets: Mapping[str, str] = field(default_factory=dict)

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
        bundle_classes: DefaultDict[str, set] = defaultdict(set)
        for entry in entries:
            if entry.provenance_class is not None:
                if entry.provenance_class not in PROVENANCE_CLASSES:
                    raise ValueError(f"unknown provenance class: {entry.provenance_class}")
                if (
                    entry.provenance_class in FITTED_PROVENANCE_CLASSES
                    and entry.fitted_parameter is None
                ):
                    raise ValueError(
                        f"{entry.subterm_id}: fitted provenance requires a fitted identity"
                    )
                if (
                    entry.provenance_class not in FITTED_PROVENANCE_CLASSES
                    and entry.fitted_parameter is not None
                ):
                    raise ValueError(
                        f"{entry.subterm_id}: a fitted identity requires fitted provenance"
                    )
            if entry.fitted_parameter is not None:
                bundle_classes[entry.fitted_parameter].add(entry.provenance_class)
        inconsistent = sorted(
            identity for identity, classes in bundle_classes.items() if len(classes) > 1
        )
        if inconsistent:
            raise ValueError(f"bundles mix provenance classes: {inconsistent}")
        buckets = dict(self.fitted_parameter_buckets)
        for identity, bucket in buckets.items():
            if not identity:
                raise ValueError("bucket assignments require nonempty identities")
            if bucket not in ASSIGNABLE_DOF_BUCKETS:
                raise ValueError(f"{identity}: unknown or reserved A18 bucket {bucket!r}")
        object.__setattr__(self, "entries", entries)
        object.__setattr__(self, "prior_floors", MappingProxyType(floors))
        object.__setattr__(self, "other_fitted_parameters", other_parameters)
        object.__setattr__(self, "fitted_parameter_buckets", MappingProxyType(buckets))

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
        bucket_usage: DefaultDict[str, int] = defaultdict(int)
        unassigned = []
        for name in sorted(parameter_names):
            bucket = self.fitted_parameter_buckets.get(name)
            if bucket is None:
                unassigned.append(name)
            else:
                bucket_usage[bucket] += 1
        within_buckets = not unassigned and all(
            count <= DOF_ALLOCATION[bucket] for bucket, count in bucket_usage.items()
        )
        return DofAccount(
            d_power=self.d_power,
            allowed=allowed,
            used=used,
            remaining=max(0, allowed - used),
            within_cap=used <= allowed,
            bucket_usage=MappingProxyType(dict(bucket_usage)),
            unassigned_identities=tuple(unassigned),
            within_buckets=within_buckets,
        )

    def validate_for_contracts(self) -> None:
        """Validate completeness, diagnostics, floors, provenance, and dof.

        Raises
        ------
        ValueError
            If the table cannot be used by the frozen 45-facet scorer,
            including any positive-mass or fitted entry that declares no
            provenance class (V4_SPEC_r4 3.6 counting rule).
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
        # PM-1 numerator cross-check (MANIFEST capacity.prior_driven...):
        # a prior-floor facet's positive mass ships either at its
        # preregistered prior (flagged prior_driven) or as an evidence-fitted
        # value (carrying a fitted identity). A bare unflagged, unfitted row
        # would silently understate the disclosure fraction.
        for facet_id in self.prior_floors:
            unaccounted = sorted(
                entry.subterm_id
                for entry in self.entries
                if entry.facet_id == facet_id
                and entry.weight > 0.0
                and not entry.prior_driven
                and entry.fitted_parameter is None
            )
            if unaccounted:
                raise ValueError(
                    f"{facet_id} prior-floor mass is neither prior_driven nor "
                    f"evidence-fitted: {unaccounted}"
                )
        # 3.6 counting-rule closure on the mass surface (P2 review OB2): a
        # score-visible mass or a fitted identity with no declared provenance
        # class would let independently adjustable scalars cost zero dof, the
        # omission `validate_parameter_provenance` already refuses for
        # profile scalars. Construction stays permissive; the gate does not.
        unclassified = sorted(
            entry.subterm_id
            for entry in self.entries
            if entry.provenance_class is None
            and (entry.weight > 0.0 or entry.fitted_parameter is not None)
        )
        if unclassified:
            raise ValueError(
                f"score-visible sub-term masses lack a provenance class: {unclassified}"
            )
        account = self.dof_account
        if not account.within_cap:
            raise ValueError("fitted parameter count exceeds min(20, D_power)")
        if account.unassigned_identities:
            raise ValueError(
                "fitted identities lack a preregistered A18 allocation bucket: "
                f"{sorted(account.unassigned_identities)}"
            )
        if not account.within_buckets:
            over = sorted(
                bucket
                for bucket, count in account.bucket_usage.items()
                if count > DOF_ALLOCATION[bucket]
            )
            raise ValueError(f"A18 allocation buckets exceeded: {over}")

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
