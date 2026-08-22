"""Guarded orchestration for the synthetic FREEZE-1 profiled fit."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path
from types import MappingProxyType
from typing import Any, Mapping, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua.eval.ruler_v4.fit.access import AccessLedger
from dagua.eval.ruler_v4.fit.bank import (
    _FROZEN_A15_ROLE_HASH,
    SideSwapAuditRow,
    SplitPurpose,
)
from dagua.eval.ruler_v4.fit.diagnostics import SideSwapAuditResult, side_swap_audit
from dagua.eval.ruler_v4.fit.objective import (
    FitPair,
    FittingPlan,
    PairwiseObjective,
    partition_fit_ord_lines,
)
from dagua.eval.ruler_v4.fit.optimize import (
    FitResult,
    OuterWeightStability,
    _information_diagnostics,
    _profile_weight_intervals,
    apply_outer_weight_split_half,
)
from dagua.eval.ruler_v4.fit.uncertainty import (
    HalfAssignment,
    HJNDBranchResult,
    JNDFitConfig,
    JNDHeterogeneityFit,
    JNDProfileFit,
    _half_key,
    evaluate_h_jnd_branch,
    fit_jnd_heterogeneity,
    load_half_assignment,
    profile_jnd_block,
)
from dagua.eval.ruler_v4.weight_table import REQUIRED_PRIOR_FLOOR_FACETS, WeightTable

_JOINT_TOLERANCE = 1.0e-10
_MAX_FIXED_POINT_ITERATIONS = 25
_LAPSE_BOUNDS = (0.0, 0.25)
_LAPSE_INITIAL = 1.0 / 109.0
_LAPSE_BOUNDARY_ATOL = 1.0e-8
_PROFILE_DROP = 1.920729410347062
_LANDED_PRIOR_ADDENDUM = 29
_EXPECTED_MAP_SHA256 = (
    "fd6f7659c011f985bdbcb44686a782af6505f6ea83a61ab7f1f69925722d418a"  # pragma: allowlist secret
)
_DOF_BUCKETS = (
    ("N_u", 9, True, "universal"),
    ("UNSPENT", 1, False, "unspent"),
    ("N_s", 3, True, "semantic"),
    ("N_g", 4, True, "group_model"),
    ("N_t", 3, True, "aggregation"),
)
_BLIND_ASSERTIONS = frozenset(
    {
        "A1_QUARANTINE_DISJOINT",
        "A2_SCHEMA_BLIND",
        "A3_CODE_BLIND",
        "A4_RESOLUTION_COMPLETE",
    }
)
_BLIND_FORBIDDEN_KEYS = frozenset(
    {
        "engine",
        "layout",
        "renderer",
        "graph_name",
        "positions_path",
        "store",
        "record_id",
        "profile",
        "blind_id",
        "verdict",
        "tie",
        "abstain",
        "confidence",
        "reason",
        "free_note",
        "defects",
    }
)
_BLIND_MANDATED_KEY_EXCEPTIONS = frozenset(
    {"engine_identity_fields", "missing_blind_ids", "duplicate_blind_ids"}
)


class FitStartConditionError(RuntimeError):
    """Signal that a real FREEZE-1 fit has not met its start conditions."""


class FitConvergenceError(RuntimeError):
    """Signal that the profiled fixed point exhausted its iteration budget."""


@dataclass(frozen=True)
class RealFitStartConditions:
    """Declare owner-controlled gates that must precede any real fit.

    Parameters
    ----------
    campaign_complete : bool
        Whether the preregistered MAIN campaign has completed.
    protocol_start_authorized : bool
        Whether the owner has authorized the frozen fit start.
    lapse_prior_frozen : bool
        Whether DISCREPANCIES 56 has been resolved.
    graph_half_assignment_frozen : bool
        Whether DISCREPANCIES 57 has been resolved.
    fitted_dof_declaration_verified : bool
        Whether DOF-DECL(d/e) verified the external fitted-identity ledger.
    blind_map_attested : bool
        Whether the orchestration attested blind-map separation.
    """

    campaign_complete: bool
    protocol_start_authorized: bool
    lapse_prior_frozen: bool
    graph_half_assignment_frozen: bool
    fitted_dof_declaration_verified: bool
    blind_map_attested: bool

    @property
    def ready(self) -> bool:
        """Return whether every real-fit start gate is affirmative.

        Returns
        -------
        bool
            True only when all owner-controlled gates are satisfied.
        """

        return all(asdict(self).values())


@dataclass(frozen=True)
class FitDriverConfig:
    """Freeze deterministic orchestration constants.

    Parameters
    ----------
    seed : int
        Frozen deterministic seed.
    joint_tolerance : float
        Frozen convergence tolerance on the joint objective.
    maximum_iterations : int
        Fail-closed fixed-point iteration budget.
    protocol_addendum : int
        Landed addendum number deriving the lapse-prior gate.
    expected_map_sha256 : str
        Frozen ADDENDUM-19 map digest checked without opening the map.
    """

    seed: int = field(default=20260811, init=False)
    joint_tolerance: float = field(default=_JOINT_TOLERANCE, init=False)
    maximum_iterations: int = field(default=_MAX_FIXED_POINT_ITERATIONS, init=False)
    protocol_addendum: int = field(default=_LANDED_PRIOR_ADDENDUM, init=False)
    expected_map_sha256: str = field(default=_EXPECTED_MAP_SHA256, init=False)


@dataclass(frozen=True)
class FitDriverIteration:
    """Publish one profiled fixed-point trajectory row.

    Parameters
    ----------
    iteration : int
        One-based outer iteration.
    weights : mapping[str, float]
        Updated outer weights.
    lapse_rate : float
        Updated synthetic lapse MLE.
    mu, tau_class, tau_band : float
        Profiled JND-block values used for this update.
    train_objective, jnd_objective, joint_objective : float
        Summed objective components and their joint value.
    jnd_improvement : float or None
        JND-profile block decrease from the preceding fixed-point row.
    weight_lapse_improvement : float
        Train-objective decrease from this row's weight/lapse block.
    joint_improvement : float or None
        Sum of the two coordinate-block decreases; unavailable on the first row.
    """

    iteration: int
    weights: Mapping[str, float]
    lapse_rate: float
    mu: float
    tau_class: float
    tau_band: float
    train_objective: float
    jnd_objective: float
    joint_objective: float
    jnd_improvement: Optional[float]
    weight_lapse_improvement: float
    joint_improvement: Optional[float]

    def __post_init__(self) -> None:
        """Freeze the trajectory weight mapping."""

        object.__setattr__(self, "weights", MappingProxyType(dict(self.weights)))


@dataclass(frozen=True)
class LapseBoundaryDisclosure:
    """Publish a fitted lapse that reaches a frozen optimizer boundary.

    Parameters
    ----------
    bound : str
        ``lower`` or ``upper``.
    fitted_value : float
        Fitted scalar within ``1e-8`` of that bound.
    penalized_objective, unpenalized_objective : float
        Summed train objectives at the same optimum.
    """

    bound: str
    fitted_value: float
    penalized_objective: float
    unpenalized_objective: float

    def __post_init__(self) -> None:
        """Validate the named finite boundary publication.

        Raises
        ------
        ValueError
            If the bound is unknown or a publication value is nonfinite.
        """

        if self.bound not in {"lower", "upper"}:
            raise ValueError("lapse boundary disclosure must name lower or upper")
        if not all(
            math.isfinite(value)
            for value in (
                self.fitted_value,
                self.penalized_objective,
                self.unpenalized_objective,
            )
        ):
            raise ValueError("lapse boundary disclosure values must be finite")


@dataclass(frozen=True)
class LapsePriorFreeSensitivity:
    """Publish the mandatory fit with only LAPSE-PRIOR removed.

    Parameters
    ----------
    lapse_rate, mu, maximum_absolute_weight_change : float
        Prior-free lapse, associated JND location, and largest outer-weight delta.
    """

    lapse_rate: float
    mu: float
    maximum_absolute_weight_change: float


@dataclass(frozen=True)
class Freeze1FitResult:
    """Publish the complete guarded FREEZE-1 run.

    Parameters
    ----------
    weight_fit : FitResult
        Full-data outer-weight fit at the converged JND profile.
    outer_weight_stability : OuterWeightStability
        Graph-disjoint split-half response and shipped weights.
    lapse_rate : float
        Fitted synthetic seven-category lapse rate.
    lapse_interval : tuple[float, float]
        Penalized 95% profile-likelihood interval.
    lapse_prior_weight : float
        Realized ``111 / (111 + n_train_informative)`` prior weight.
    lapse_boundary_disclosure : LapseBoundaryDisclosure or None
        Named disclosure when the fitted lapse reaches either frozen bound.
    lapse_prior_free_sensitivity : LapsePriorFreeSensitivity
        Required refit with only the lapse prior removed.
    jnd_fit : JNDHeterogeneityFit
        Final W-13 point and uncertainty publications.
    h_jnd_branch : HJNDBranchResult
        Once-only ledgered H-JND branch decision.
    side_swap : SideSwapAuditResult or None
        FIT-ORD(b) audit for real runs; absent only for synthetic fixtures.
    start_conditions : RealFitStartConditions
        Six derived gates used for this run.
    half_assignment_digest : str or None
        Frozen real partition digest, absent only for synthetic fixtures.
    trajectory : tuple[FitDriverIteration, ...]
        Frozen joint-objective fixed-point trajectory.
    access_budget_before, access_budget_after : mapping[str, int]
        Persistent ledger usage surrounding the run.
    run_dir : pathlib.Path
        Newly created artifact directory.
    """

    weight_fit: FitResult
    outer_weight_stability: OuterWeightStability
    lapse_rate: float
    lapse_interval: Tuple[float, float]
    lapse_prior_weight: float
    lapse_boundary_disclosure: Optional[LapseBoundaryDisclosure]
    lapse_prior_free_sensitivity: LapsePriorFreeSensitivity
    jnd_fit: JNDHeterogeneityFit
    h_jnd_branch: HJNDBranchResult
    side_swap: Optional[SideSwapAuditResult]
    start_conditions: RealFitStartConditions
    half_assignment_digest: Optional[str]
    trajectory: Tuple[FitDriverIteration, ...]
    access_budget_before: Mapping[str, int]
    access_budget_after: Mapping[str, int]
    run_dir: Path

    def __post_init__(self) -> None:
        """Freeze driver budget mappings."""

        object.__setattr__(
            self, "access_budget_before", MappingProxyType(dict(self.access_budget_before))
        )
        object.__setattr__(
            self, "access_budget_after", MappingProxyType(dict(self.access_budget_after))
        )


def _atomic_write_json(path: Path, payload: object) -> None:
    """Atomically write one deterministic JSON artifact.

    Parameters
    ----------
    path : pathlib.Path
        Final artifact path inside a new run directory.
    payload : object
        JSON-serializable value.
    """

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False)
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, f"{encoded}\n".encode("utf-8"))
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _atomic_write_jsonl(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Atomically write deterministic JSON Lines.

    Parameters
    ----------
    path : pathlib.Path
        Final JSONL artifact path.
    rows : sequence[mapping[str, object]]
        Ordered trajectory records.
    """

    temporary = path.with_suffix(f"{path.suffix}.tmp")
    encoded = "".join(
        f"{json.dumps(dict(row), sort_keys=True, separators=(',', ':'), allow_nan=False)}\n"
        for row in rows
    ).encode("utf-8")
    descriptor = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    try:
        os.write(descriptor, encoded)
        os.fsync(descriptor)
    finally:
        os.close(descriptor)
    os.replace(temporary, path)


def _input_digest(pairs: Sequence[FitPair]) -> str:
    """Digest stable train-line identities and responses.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Complete train-role input.

    Returns
    -------
    str
        SHA-256 digest for artifact provenance.
    """

    digest = hashlib.sha256()
    for pair in pairs:
        fields = (
            pair.replicate_group_id,
            pair.base_pair_id,
            pair.session_id,
            pair.blind_id_a,
            pair.blind_id_b,
            str(pair.graded_verdict),
        )
        digest.update("\0".join(fields).encode("utf-8"))
        digest.update(b"\n")
    return digest.hexdigest()


def _sha256_file(path: Path) -> str:
    """Return a streaming SHA-256 digest for one artifact.

    Parameters
    ----------
    path : pathlib.Path
        File to digest.

    Returns
    -------
    str
        Lowercase hexadecimal SHA-256 digest.
    """

    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _allocation_block_sha256(source_path: Path) -> str:
    """Re-extract DOF-DECL's canonical A18 section-7 allocation digest.

    Parameters
    ----------
    source_path : pathlib.Path
        Live ``PREREG_V4_CALIBRATION.md`` supplied as frozen configuration.

    Returns
    -------
    str
        SHA-256 of the canonical allocation image from DOF-DECL(a-bis).

    Raises
    ------
    FitStartConditionError
        If the five allocation rows or their arithmetic cannot be parsed.
    """

    source = Path(source_path).read_text(encoding="utf-8")
    rows = re.findall(
        r"^\|\s*(`N_[usgt]`[^|]*|UNSPENT[^|]*)\|\s*(\d+)\s*\|([^|]*)\|",
        source,
        re.MULTILINE,
    )
    if len(rows) != 5:
        raise FitStartConditionError("DOF declaration source must contain five allocation rows")
    parsed = []
    for label, cap, contents in rows:
        match = re.match(r"`?(N_[usgt]|UNSPENT)`?", label.strip())
        if match is None:
            raise FitStartConditionError("DOF declaration source has an invalid bucket label")
        parsed.append((match.group(1), int(cap), contents))
    if tuple(row[0] for row in parsed) != tuple(bucket[0] for bucket in _DOF_BUCKETS):
        raise FitStartConditionError("DOF declaration source bucket order differs from A18")
    cap_match = re.search(r"allowed_fitted_dof = min\(20, (\d+)\) = \*\*(\d+)\*\*", source)
    power_match = re.search(r"D_power = floor\(min\(([\d.]+), (\d+)\)\) = (\d+)", source)
    controlled_match = re.search(r"`controlled_stimulus_fitted_dof = (\d+)`", source)
    if cap_match is None or power_match is None or controlled_match is None:
        raise FitStartConditionError("DOF declaration source arithmetic is incomplete")
    inputs = {}
    for name, pattern in (
        ("i_floor", r"I_floor\s*=\s*([\d,]+)"),
        ("j_min", r"J_min\s*=\s*([\d,]+)"),
        ("u_ms", r"U_ms\s*=\s*([\d,]+)"),
    ):
        match = re.search(pattern, source)
        if match is None:
            raise FitStartConditionError(f"DOF declaration source lacks {name}")
        inputs[name] = int(match.group(1).replace(",", ""))
    lines = [
        f"allowed_fitted_dof={int(cap_match.group(2))}",
        f"d_power={int(power_match.group(3))}",
        f"i_floor={inputs['i_floor']}",
        f"j_min={inputs['j_min']}",
        f"u_ms={inputs['u_ms']}",
    ]
    lines.extend(
        f"bucket={name} cap={cap} contents={' '.join(contents.split())}"
        for name, cap, contents in parsed
    )
    lines.append(f"controlled_stimulus_fitted_dof={int(controlled_match.group(1))}")
    return hashlib.sha256(("\n".join(lines) + "\n").encode("utf-8")).hexdigest()


def _verify_dof_declaration(
    declaration_path: Path,
    expected_sha256: str,
    source_path: Path,
    plan: FittingPlan,
    weight_table: WeightTable,
) -> Mapping[str, object]:
    """Verify DOF-DECL content and reconcile it with the realized fit ledger.

    Parameters
    ----------
    declaration_path : pathlib.Path
        Frozen ``FITTED_DOF_DECLARATION.json``.
    expected_sha256 : str
        Digest frozen by the declaration's latest landed transition.
    source_path : pathlib.Path
        Live A18 preregistration source for the allocation-block check.
    plan : FittingPlan
        Realized fit plan.
    weight_table : WeightTable
        Complete shipped table whose fitted identities must match the declaration.

    Returns
    -------
    mapping[str, object]
        Parsed verified declaration for verbatim manifest publication.

    Raises
    ------
    FitStartConditionError
        If any digest, schema, completeness, allocation, or fit cross-check fails.
    """

    path = Path(declaration_path)
    if _sha256_file(path) != expected_sha256:
        raise FitStartConditionError("fitted DOF declaration digest does not match")
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict) or raw.get("schema_version") != "v4-dof-decl-1":
        raise FitStartConditionError("fitted DOF declaration schema is invalid")
    if raw.get("source_allocation_sha256") != _allocation_block_sha256(source_path):
        raise FitStartConditionError("fitted DOF declaration allocation source is stale")
    if raw.get("assignment_complete") is not True or not raw.get("assignment_authority"):
        raise FitStartConditionError("fitted DOF declaration assignment is incomplete")
    buckets = raw.get("buckets")
    if not isinstance(buckets, list) or len(buckets) != len(_DOF_BUCKETS):
        raise FitStartConditionError("fitted DOF declaration bucket table is invalid")
    declared_by_code: dict[str, set[str]] = {}
    all_identities = []
    for entry, (name, cap, assignable, code_name) in zip(buckets, _DOF_BUCKETS):
        if not isinstance(entry, dict):
            raise FitStartConditionError("fitted DOF declaration bucket row is invalid")
        filled = entry.get("filled")
        if (
            entry.get("bucket") != name
            or entry.get("cap") != cap
            or entry.get("assignable") is not assignable
            or not isinstance(filled, list)
            or any(not isinstance(identity, str) or not identity for identity in filled)
        ):
            raise FitStartConditionError(f"fitted DOF declaration bucket {name} is invalid")
        if len(filled) != (cap if assignable else 0):
            raise FitStartConditionError(f"fitted DOF declaration bucket {name} is not filled")
        declared_by_code[code_name] = set(filled)
        all_identities.extend(filled)
    if len(all_identities) != len(set(all_identities)):
        raise FitStartConditionError("fitted identity appears in more than one DOF bucket")
    if raw.get("sum") != 20 or raw.get("allowed_fitted_dof") != 20:
        raise FitStartConditionError("fitted DOF declaration cap arithmetic differs from A18")
    if raw.get("controlled_stimulus_fitted_dof") != 0:
        raise FitStartConditionError("controlled-stimulus fitted DOF must remain zero")
    if set(raw.get("prior_floor_facets", [])) != set(REQUIRED_PRIOR_FLOOR_FACETS):
        raise FitStartConditionError("fitted DOF declaration prior-floor facets differ")

    table_outer = {
        entry.fitted_parameter
        for entry in weight_table.entries
        if entry.fitted_parameter is not None
    }
    plan_outer = {parameter.name for parameter in plan.weights}
    if table_outer != plan_outer:
        raise FitStartConditionError("fitting plan and WeightTable outer identities differ")
    actual_by_code: dict[str, set[str]] = {name: set() for name in declared_by_code}
    for identity, bucket in weight_table.fitted_parameter_buckets.items():
        if bucket in actual_by_code:
            actual_by_code[bucket].add(identity)
    if actual_by_code != declared_by_code:
        raise FitStartConditionError("realized fitted identities differ from the DOF declaration")
    account = weight_table.dof_account
    if (
        not account.within_cap
        or not account.within_buckets
        or account.used != len(all_identities)
        or weight_table.d_power != 20
    ):
        raise FitStartConditionError("realized fitted DOF accounting does not reconcile")
    if set(weight_table.prior_floors) != set(REQUIRED_PRIOR_FLOOR_FACETS):
        raise FitStartConditionError("realized WeightTable omits declared prior-floor facets")
    if dict(weight_table.prior_floors) != dict(plan.prior_floors):
        raise FitStartConditionError("realized plan changes a declared prior floor")
    return MappingProxyType(raw)


def _forbidden_attestation_key(key: str) -> bool:
    """Return whether one attestation key violates BLIND-ATTEST(d).

    Parameters
    ----------
    key : str
        JSON object key from an attestation line.

    Returns
    -------
    bool
        True for record-content fields outside the constitutional whitelist.
    """

    if key in _BLIND_MANDATED_KEY_EXCEPTIONS:
        return False
    lowered = key.lower()
    return any(token in lowered for token in _BLIND_FORBIDDEN_KEYS)


def _attestation_whitelist_valid(value: Any) -> bool:
    """Recursively enforce the amended BLIND-ATTEST content whitelist.

    Parameters
    ----------
    value : Any
        Parsed JSON value.

    Returns
    -------
    bool
        True only when no object key carries prohibited record content.
    """

    if isinstance(value, dict):
        return all(
            isinstance(key, str)
            and not _forbidden_attestation_key(key)
            and _attestation_whitelist_valid(child)
            for key, child in value.items()
        )
    if isinstance(value, list):
        return not value
    return isinstance(value, (str, int, bool)) and not isinstance(value, float)


def _verify_blind_attestation(
    attestation_path: Path,
    line_sha256: str,
    expected_map_sha256: str,
    fit_input_digest: str,
) -> Mapping[str, object]:
    """Verify one exact append-only attestation line without opening the map.

    Parameters
    ----------
    attestation_path : pathlib.Path
        Append-only JSONL attestation record.
    line_sha256 : str
        Orchestration-supplied digest of the exact attesting line bytes.
    expected_map_sha256 : str
        Frozen ADDENDUM-19 map digest.
    fit_input_digest : str
        Driver-recomputed digest of this delivered row set.

    Returns
    -------
    mapping[str, object]
        Parsed verified line for manifest and freeze-report publication.

    Raises
    ------
    FitStartConditionError
        If no exact line matches or any schema/content assertion is false.
    """

    matches = [
        line
        for line in Path(attestation_path).read_bytes().splitlines(keepends=True)
        if hashlib.sha256(line).hexdigest() == line_sha256
    ]
    if len(matches) != 1:
        raise FitStartConditionError("blind-map attestation line digest does not resolve once")
    raw = json.loads(matches[0])
    required = {
        "attestation_version",
        "date",
        "attester",
        "map_path",
        "map_sha256",
        "map_rows",
        "map_authority",
        "fit_input_digest",
        "assertions",
    }
    if not isinstance(raw, dict) or set(raw) != required:
        raise FitStartConditionError("blind-map attestation line schema fields differ")
    if raw.get("attestation_version") != "v4-blind-attest-1":
        raise FitStartConditionError("blind-map attestation schema version differs")
    if not _attestation_whitelist_valid(raw):
        raise FitStartConditionError("blind-map attestation violates the content whitelist")
    assertions = raw.get("assertions")
    if not isinstance(assertions, dict) or set(assertions) != _BLIND_ASSERTIONS:
        raise FitStartConditionError("blind-map attestation assertion set differs")
    for name, block in assertions.items():
        if (
            not isinstance(block, dict)
            or set(block) != {"assertion", "result", "evidence"}
            or block.get("assertion") != name
            or block.get("result") is not True
            or not isinstance(block.get("evidence"), dict)
        ):
            raise FitStartConditionError(f"blind-map attestation assertion is false: {name}")
    a2_evidence = assertions["A2_SCHEMA_BLIND"]["evidence"]
    if not isinstance(a2_evidence, dict) or a2_evidence.get("engine_identity_fields") != []:
        raise FitStartConditionError("blind-map attestation row schema is not engine-blind")
    if raw.get("fit_input_digest") != fit_input_digest:
        raise FitStartConditionError("blind-map attestation input digest differs from this run")
    if raw.get("map_sha256") != expected_map_sha256:
        raise FitStartConditionError("blind-map attestation map digest differs from frozen config")
    return MappingProxyType(raw)


def _pairs_with_profile(
    pairs: Sequence[FitPair], profile: JNDProfileFit, lapse_rate: float
) -> Tuple[FitPair, ...]:
    """Apply one profiled JND block and lapse to train rows.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Complete train line.
    profile : JNDProfileFit
        Current JND profile with pooled fallbacks.
    lapse_rate : float
        Current uniform lapse value.

    Returns
    -------
    tuple[FitPair, ...]
        Updated immutable likelihood rows.
    """

    pooled_jnd = math.exp(profile.mu)
    return tuple(
        replace(
            pair,
            jnd=profile.jnd_by_cell.get((pair.primary_class, pair.size_band), pooled_jnd),
            lapse_rate=lapse_rate,
        )
        for pair in pairs
    )


def _fit_weight_lapse_block(
    pairs: Sequence[FitPair],
    plan: FittingPlan,
    config: FitDriverConfig,
    initial_weights: Optional[Mapping[str, float]] = None,
    initial_lapse: Optional[float] = None,
    include_lapse_prior: bool = True,
    lapse_bounds: Tuple[float, float] = _LAPSE_BOUNDS,
) -> Tuple[FitResult, float, float]:
    """Jointly optimize synthetic outer weights and the uniform lapse.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Train rows at a fixed profiled JND block.
    plan : FittingPlan
        Frozen outer-weight plan.
    config : FitDriverConfig
        Deterministic seed and iteration provenance.
    initial_weights : mapping[str, float] or None
        Prior fixed-point weights, defaulting to frozen literature priors.
    initial_lapse : float or None
        Prior fixed-point lapse, defaulting to the frozen Beta-prior mode.
    include_lapse_prior : bool
        Whether to include LAPSE-PRIOR(d); false only for sensitivity reporting.
    lapse_bounds : tuple[float, float]
        Projected lapse interval for this fit.

    Returns
    -------
    tuple[FitResult, float, float]
        Weight publication, fitted lapse, and mean regularized train objective.
    """

    from scipy.optimize import minimize

    rows = tuple(pairs)
    names = plan.parameter_names
    starting_weights = (
        {parameter.name: float(parameter.prior) for parameter in plan.weights}
        if initial_weights is None
        else dict(initial_weights)
    )
    if set(starting_weights) != set(names):
        raise ValueError("initial weight identities do not match the fitting plan")
    parameter_bounds = []
    for parameter in plan.weights:
        if parameter.lower is None or parameter.upper is None:
            raise RuntimeError("fitting-plan weight bounds were not normalized")
        parameter_bounds.append((parameter.lower, parameter.upper))
    bounds_by_name = {
        parameter.name: parameter_bounds[index] for index, parameter in enumerate(plan.weights)
    }
    starting_lapse = _LAPSE_INITIAL if initial_lapse is None else initial_lapse
    initial = np.asarray(
        [starting_weights[name] for name in names] + [starting_lapse], dtype=np.float64
    )
    bounds = parameter_bounds + [lapse_bounds]

    def evaluate(candidate: np.ndarray) -> float:
        """Evaluate one joint synthetic weight/lapse candidate.

        Parameters
        ----------
        candidate : numpy.ndarray
            Weight coordinates followed by lapse.

        Returns
        -------
        float
            Mean regularized train objective.
        """

        objective = PairwiseObjective(
            tuple(replace(pair, lapse_rate=float(candidate[-1])) for pair in rows),
            plan,
        )
        vector = torch.tensor(candidate[:-1], dtype=torch.float64)
        loss = (
            objective.loss(vector)
            if include_lapse_prior
            else objective.loss_without_lapse_prior(vector)
        )
        return float(loss)

    accepted = [initial.copy()]
    losses = [evaluate(initial)]

    def record(candidate: np.ndarray) -> None:
        """Record one accepted L-BFGS-B iterate.

        Parameters
        ----------
        candidate : numpy.ndarray
            Accepted joint candidate.
        """

        accepted.append(np.asarray(candidate, dtype=np.float64).copy())
        losses.append(evaluate(candidate))

    # SciPy's stubs do not model the legal Powell callback/options combination.
    result = minimize(  # type: ignore[call-overload]
        evaluate,
        initial,
        method="Powell",
        bounds=bounds,
        callback=record,
        options={"ftol": 1.0e-13, "xtol": 1.0e-13, "maxiter": 500},
    )
    if not math.isfinite(float(result.fun)):
        raise FloatingPointError(f"joint weight/lapse optimization failed: {result.message}")
    final = np.asarray(result.x, dtype=np.float64)
    if not np.array_equal(accepted[-1], final):
        accepted.append(final.copy())
        losses.append(evaluate(final))
    weights = {name: float(final[index]) for index, name in enumerate(names)}
    lapse_rate = float(final[-1])
    objective = PairwiseObjective(
        tuple(replace(pair, lapse_rate=lapse_rate) for pair in rows),
        plan,
    )
    at_bounds = {
        parameter.name: (
            "fixed"
            if math.isclose(
                bounds_by_name[parameter.name][0],
                bounds_by_name[parameter.name][1],
                abs_tol=1.0e-12,
            )
            else (
                "lower"
                if math.isclose(
                    weights[parameter.name],
                    bounds_by_name[parameter.name][0],
                    abs_tol=1.0e-12,
                )
                else "upper"
            )
        )
        for parameter in plan.weights
        if math.isclose(weights[parameter.name], bounds_by_name[parameter.name][0], abs_tol=1.0e-12)
        or math.isclose(weights[parameter.name], bounds_by_name[parameter.name][1], abs_tol=1.0e-12)
    }
    intervals = _profile_weight_intervals(objective, weights)
    information_rank, condition_number = _information_diagnostics(objective, weights)
    fit = FitResult(
        weights=weights,
        weight_paths={
            name: tuple(float(candidate[index]) for candidate in accepted)
            for index, name in enumerate(names)
        },
        losses=tuple(losses),
        at_bounds=at_bounds,
        intervals=intervals,
        information_rank=information_rank,
        condition_number=condition_number,
        converged=bool(result.success),
        steps_completed=len(accepted) - 1,
        seed=config.seed,
    )
    return fit, lapse_rate, float(result.fun)


def _profile_lapse_interval(
    pairs: Sequence[FitPair],
    plan: FittingPlan,
    fitted_weights: Mapping[str, float],
    fitted_lapse: float,
) -> Tuple[float, float]:
    """Profile outer weights for the penalized lapse interval.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Final profiled train rows.
    plan : FittingPlan
        Frozen fitting plan.
    fitted_weights : mapping[str, float]
        Joint optimum used to initialize every profile solve.
    fitted_lapse : float
        Penalized lapse optimum.

    Returns
    -------
    tuple[float, float]
        Two-sided 95% penalized profile interval inside ``[0, 0.25]``.

    Raises
    ------
    RuntimeError
        If a profile optimization fails.
    """

    from scipy.optimize import brentq, minimize

    rows = tuple(pairs)
    names = plan.parameter_names
    initial = np.asarray([fitted_weights[name] for name in names], dtype=np.float64)
    bounds = [(float(parameter.lower), float(parameter.upper)) for parameter in plan.weights]
    optimum_objective = PairwiseObjective(
        tuple(replace(pair, lapse_rate=fitted_lapse) for pair in rows),
        plan,
    )
    optimum = float(optimum_objective.loss(torch.tensor(initial, dtype=torch.float64)))
    target = optimum + _PROFILE_DROP / len(rows)

    def profile(lapse: float) -> float:
        """Return the profiled mean objective minus the LR target.

        Parameters
        ----------
        lapse : float
            Fixed lapse candidate.

        Returns
        -------
        float
            Signed distance from the profile-likelihood target.
        """

        objective = PairwiseObjective(
            tuple(replace(pair, lapse_rate=lapse) for pair in rows),
            plan,
        )

        def evaluate(candidate: np.ndarray) -> float:
            """Evaluate one free outer-weight profile candidate.

            Parameters
            ----------
            candidate : numpy.ndarray
                Free outer weights.

            Returns
            -------
            float
                Penalized mean train objective.
            """

            return float(objective.loss(torch.tensor(candidate, dtype=torch.float64)))

        result = minimize(evaluate, initial, method="L-BFGS-B", bounds=bounds)
        if not result.success:
            raise RuntimeError(f"lapse profile optimization failed: {result.message}")
        return float(result.fun) - target

    lower_endpoint = 1.0e-12
    lower = (
        0.0
        if profile(lower_endpoint) <= 0.0
        else float(brentq(profile, lower_endpoint, fitted_lapse))
    )
    upper = (
        _LAPSE_BOUNDS[1]
        if profile(_LAPSE_BOUNDS[1]) <= 0.0
        else float(brentq(profile, fitted_lapse, _LAPSE_BOUNDS[1]))
    )
    return lower, upper


def _lapse_boundary_disclosure(
    pairs: Sequence[FitPair],
    plan: FittingPlan,
    weights: Mapping[str, float],
    lapse_rate: float,
) -> Optional[LapseBoundaryDisclosure]:
    """Build RIDER-1-style lapse boundary evidence when required.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Final train rows.
    plan : FittingPlan
        Frozen fitting plan.
    weights : mapping[str, float]
        Final outer weights.
    lapse_rate : float
        Final fitted lapse.

    Returns
    -------
    LapseBoundaryDisclosure or None
        Named disclosure within ``1e-8`` of a bound, otherwise ``None``.
    """

    bound = None
    if abs(lapse_rate - _LAPSE_BOUNDS[0]) <= _LAPSE_BOUNDARY_ATOL:
        bound = "lower"
    elif abs(lapse_rate - _LAPSE_BOUNDS[1]) <= _LAPSE_BOUNDARY_ATOL:
        bound = "upper"
    if bound is None:
        return None
    rows = tuple(replace(pair, lapse_rate=lapse_rate) for pair in pairs)
    objective = PairwiseObjective(rows, plan)
    vector = torch.tensor([weights[name] for name in plan.parameter_names], dtype=torch.float64)
    return LapseBoundaryDisclosure(
        bound=bound,
        fitted_value=lapse_rate,
        penalized_objective=float(objective.loss(vector)) * len(rows),
        unpenalized_objective=float(objective.loss_without_lapse_prior(vector)) * len(rows),
    )


def _require_real_start_conditions(
    conditions: Optional[RealFitStartConditions],
) -> None:
    """Fail closed unless every owner-controlled real-fit gate is satisfied.

    Parameters
    ----------
    conditions : RealFitStartConditions or None
        Explicit real-fit authorization state.

    Raises
    ------
    FitStartConditionError
        If any required start condition is absent.
    """

    if conditions is None or not conditions.ready:
        raise FitStartConditionError(
            "real FREEZE-1 requires campaign completion and every protocol start condition"
        )


def run_freeze1_fit(
    pairs: Sequence[FitPair],
    plan: FittingPlan,
    jnd_config: JNDFitConfig,
    run_dir: Path,
    ledger_root: Path,
    config: Optional[FitDriverConfig] = None,
    real_start_conditions: Optional[RealFitStartConditions] = None,
    family_map_path: Optional[Path] = None,
    dof_declaration_path: Optional[Path] = None,
    dof_declaration_sha256: Optional[str] = None,
    dof_source_path: Optional[Path] = None,
    weight_table: Optional[WeightTable] = None,
    blind_attestation_path: Optional[Path] = None,
    blind_attestation_sha256: Optional[str] = None,
    side_swap_audit_rows: Sequence[SideSwapAuditRow] = (),
) -> Freeze1FitResult:
    """Run guarded FREEZE-1 fitting and write complete artifacts.

    Parameters
    ----------
    pairs : sequence[FitPair]
        Complete train-role rows containing realized replication groups.
    plan : FittingPlan
        Frozen outer-weight plan.
    jnd_config : JNDFitConfig
        Frozen W-13 support and guard inputs.
    run_dir : pathlib.Path
        New, non-existing output directory.
    ledger_root : pathlib.Path
        Explicit ledger root; campaign state is legal only for a real run.
    config : FitDriverConfig or None
        Frozen driver configuration; ``None`` constructs the only valid values.
    real_start_conditions : RealFitStartConditions or None
        Campaign-complete and owner-authorization inputs. The remaining four
        fields are recomputed from frozen artifacts rather than trusted.
    family_map_path : pathlib.Path or None
        Frozen A15 family map required by real ``v4-half-1`` splitting.
    dof_declaration_path, dof_source_path : pathlib.Path or None
        Frozen declaration and live A18 allocation source required by DOF-DECL.
    dof_declaration_sha256 : str or None
        Latest landed declaration digest.
    weight_table : WeightTable or None
        Complete realized table for the external fitted-identity cross-check.
    blind_attestation_path : pathlib.Path or None
        Append-only attestation record containing the run's exact line.
    blind_attestation_sha256 : str or None
        Run parameter digest of the exact attesting line bytes.
    side_swap_audit_rows : sequence[SideSwapAuditRow]
        Audit-only exchanged-order controls, never likelihood rows.

    Returns
    -------
    Freeze1FitResult
        Complete fitted publications, branch decision, ledger state, and run path.

    Raises
    ------
    FitStartConditionError
        If real rows arrive before campaign and protocol start gates.
    FitConvergenceError
        If the profiled fixed point misses the frozen tolerance.
    FileExistsError
        If ``run_dir`` already exists.
    ValueError
        If rows violate train-only or deterministic configuration guards, or
        if a synthetic run targets the campaign ledger root.
    """

    rows = tuple(pairs)
    if not rows:
        raise ValueError("FREEZE-1 driver requires nonempty train rows")
    if any(pair.purpose is not SplitPurpose.FIT for pair in rows):
        raise ValueError("FREEZE-1 driver accepts train-role rows only")
    driver_config = FitDriverConfig() if config is None else config
    synthetic_only = all(pair.synthetic for pair in rows)
    if not synthetic_only and any(pair.synthetic for pair in rows):
        raise ValueError("FREEZE-1 cannot mix synthetic and real rows")
    input_digest = _input_digest(rows)
    half_assignment: Optional[HalfAssignment] = None
    dof_declaration: Optional[Mapping[str, object]] = None
    blind_attestation: Optional[Mapping[str, object]] = None
    side_swap_result: Optional[SideSwapAuditResult] = None
    start_conditions = RealFitStartConditions(False, False, False, False, False, False)
    if not synthetic_only:
        if (
            real_start_conditions is None
            or not real_start_conditions.campaign_complete
            or not real_start_conditions.protocol_start_authorized
        ):
            _require_real_start_conditions(real_start_conditions)
        required = {
            "family_map_path": family_map_path,
            "dof_declaration_path": dof_declaration_path,
            "dof_declaration_sha256": dof_declaration_sha256,
            "dof_source_path": dof_source_path,
            "weight_table": weight_table,
            "blind_attestation_path": blind_attestation_path,
            "blind_attestation_sha256": blind_attestation_sha256,
        }
        missing = sorted(name for name, value in required.items() if value is None)
        if missing:
            raise FitStartConditionError(f"real FREEZE-1 gate artifacts are missing: {missing}")
        assert family_map_path is not None
        assert dof_declaration_path is not None
        assert dof_declaration_sha256 is not None
        assert dof_source_path is not None
        assert weight_table is not None
        assert blind_attestation_path is not None
        assert blind_attestation_sha256 is not None
        try:
            half_assignment = load_half_assignment(family_map_path)
            dof_declaration = _verify_dof_declaration(
                dof_declaration_path,
                dof_declaration_sha256,
                dof_source_path,
                plan,
                weight_table,
            )
            blind_attestation = _verify_blind_attestation(
                blind_attestation_path,
                blind_attestation_sha256,
                driver_config.expected_map_sha256,
                input_digest,
            )
            side_swap_result = side_swap_audit(rows, side_swap_audit_rows)
        except (OSError, ValueError) as error:
            raise FitStartConditionError(f"real FREEZE-1 artifact gate failed: {error}") from error
        start_conditions = RealFitStartConditions(
            campaign_complete=real_start_conditions.campaign_complete,
            protocol_start_authorized=real_start_conditions.protocol_start_authorized,
            lapse_prior_frozen=driver_config.protocol_addendum >= _LANDED_PRIOR_ADDENDUM,
            graph_half_assignment_frozen=True,
            fitted_dof_declaration_verified=True,
            blind_map_attested=True,
        )
        _require_real_start_conditions(start_conditions)
    lines = partition_fit_ord_lines(rows)
    output = Path(run_dir)
    ledger = AccessLedger(ledger_root)
    if synthetic_only and ledger.is_campaign_root:
        raise ValueError("synthetic FREEZE-1 cannot target the campaign ledger root")
    output.mkdir(parents=False, exist_ok=False)
    budget_before = ledger.budget_usage(_FROZEN_A15_ROLE_HASH)
    _atomic_write_json(
        output / "manifest.json",
        {
            "input_digest": input_digest,
            "row_count": len(rows),
            "replication_row_count": len(lines.replication),
            "role_hash": _FROZEN_A15_ROLE_HASH,
            "seed": driver_config.seed,
            "joint_tolerance": driver_config.joint_tolerance,
            "maximum_iterations": driver_config.maximum_iterations,
            "synthetic_only": synthetic_only,
            "start_conditions": asdict(start_conditions),
            "half_assignment_digest": (
                None if half_assignment is None else half_assignment.table_sha256
            ),
            "dof_declaration": None if dof_declaration is None else dict(dof_declaration),
            "blind_attestation": None if blind_attestation is None else dict(blind_attestation),
            "access_budget_before": dict(budget_before),
        },
    )
    _atomic_write_json(output / "status.json", {"state": "RUNNING"})
    try:
        current_weights = {parameter.name: float(parameter.prior) for parameter in plan.weights}
        current_lapse = _LAPSE_INITIAL
        trajectory = []
        weight_fit: Optional[FitResult] = None
        profile: Optional[JNDProfileFit] = None
        for iteration in range(1, driver_config.maximum_iterations + 1):
            profile = profile_jnd_block(
                lines.replication,
                plan,
                current_weights,
                jnd_config,
                previous_profile=profile,
                half_assignment=half_assignment,
            )
            profiled_train = _pairs_with_profile(lines.train, profile, current_lapse)
            weight_fit, current_lapse, train_mean = _fit_weight_lapse_block(
                profiled_train,
                plan,
                driver_config,
                current_weights,
                current_lapse,
            )
            current_weights = dict(weight_fit.weights)
            train_objective_before = weight_fit.losses[0] * len(profiled_train)
            train_objective = train_mean * len(profiled_train)
            joint_objective = train_objective + profile.marginal_loss
            weight_lapse_improvement = max(train_objective_before - train_objective, 0.0)
            improvement = (
                None
                if profile.block_improvement is None
                else profile.block_improvement + weight_lapse_improvement
            )
            trajectory.append(
                FitDriverIteration(
                    iteration=iteration,
                    weights=current_weights,
                    lapse_rate=current_lapse,
                    mu=profile.mu,
                    tau_class=profile.tau_class,
                    tau_band=profile.tau_band,
                    train_objective=train_objective,
                    jnd_objective=profile.marginal_loss,
                    joint_objective=joint_objective,
                    jnd_improvement=profile.block_improvement,
                    weight_lapse_improvement=weight_lapse_improvement,
                    joint_improvement=improvement,
                )
            )
            if improvement is not None and improvement <= driver_config.joint_tolerance:
                break
        else:
            raise FitConvergenceError(
                "FREEZE-1 profiled fixed point exhausted its deterministic iteration budget"
            )
        if weight_fit is None or profile is None:
            raise RuntimeError("FREEZE-1 driver produced no fixed-point iteration")
        jnd_fit = fit_jnd_heterogeneity(
            lines.replication,
            plan,
            current_weights,
            jnd_config,
            half_assignment=half_assignment,
        )
        final_train = tuple(
            replace(
                pair,
                jnd=jnd_fit.jnd_by_cell.get(
                    (pair.primary_class, pair.size_band), math.exp(jnd_fit.mu)
                ),
                lapse_rate=current_lapse,
            )
            for pair in lines.train
        )
        lapse_interval = _profile_lapse_interval(
            final_train,
            plan,
            current_weights,
            current_lapse,
        )
        lapse_boundary = _lapse_boundary_disclosure(
            final_train,
            plan,
            current_weights,
            current_lapse,
        )
        sensitivity_fit, sensitivity_lapse, _ = _fit_weight_lapse_block(
            final_train,
            plan,
            driver_config,
            current_weights,
            current_lapse,
            include_lapse_prior=False,
            lapse_bounds=(1.0e-6, _LAPSE_BOUNDS[1]),
        )
        lapse_sensitivity = LapsePriorFreeSensitivity(
            lapse_rate=sensitivity_lapse,
            mu=jnd_fit.mu,
            maximum_absolute_weight_change=max(
                abs(sensitivity_fit.weights[name] - current_weights[name])
                for name in plan.parameter_names
            ),
        )
        lapse_prior_weight = (plan.lapse_prior_alpha + plan.lapse_prior_beta) / (
            plan.lapse_prior_alpha + plan.lapse_prior_beta + len(final_train)
        )
        halves = tuple(
            tuple(pair for pair in final_train if _half_key(pair, half_assignment) == half)
            for half in (0, 1)
        )
        if any(not half for half in halves):
            raise ValueError("v4-half-1 graph split leaves an empty outer-weight half")
        half_one, _, _ = _fit_weight_lapse_block(halves[0], plan, driver_config)
        half_two, _, _ = _fit_weight_lapse_block(halves[1], plan, driver_config)
        outer_stability = apply_outer_weight_split_half(weight_fit, half_one, half_two, plan)
        if jnd_fit.uncalibrated_classes:
            raise ValueError(
                f"rotation-envelope guard blocks classes: {list(jnd_fit.uncalibrated_classes)}"
            )
        branch = evaluate_h_jnd_branch(jnd_fit, ledger=ledger, synthetic_only=synthetic_only)
        budget_after = ledger.budget_usage(_FROZEN_A15_ROLE_HASH)
        trajectory_rows = [
            {
                "iteration": item.iteration,
                "weights": dict(item.weights),
                "lapse_rate": item.lapse_rate,
                "mu": item.mu,
                "tau_class": item.tau_class,
                "tau_band": item.tau_band,
                "train_objective": item.train_objective,
                "jnd_objective": item.jnd_objective,
                "joint_objective": item.joint_objective,
                "jnd_improvement": item.jnd_improvement,
                "weight_lapse_improvement": item.weight_lapse_improvement,
                "joint_improvement": item.joint_improvement,
            }
            for item in trajectory
        ]
        _atomic_write_jsonl(output / "trajectory.jsonl", trajectory_rows)
        _atomic_write_json(
            output / "result.json",
            {
                "weights": dict(weight_fit.weights),
                "shipped_weights": dict(outer_stability.weights),
                "weight_intervals": {
                    name: list(interval) for name, interval in weight_fit.intervals.items()
                },
                "lapse_rate": current_lapse,
                "lapse_interval": list(lapse_interval),
                "lapse_prior_weight": lapse_prior_weight,
                "lapse_boundary_disclosure": (
                    None if lapse_boundary is None else asdict(lapse_boundary)
                ),
                "lapse_prior_free_sensitivity": asdict(lapse_sensitivity),
                "jnd": {
                    "mu": jnd_fit.mu,
                    "tau_class": jnd_fit.tau_class,
                    "tau_band": jnd_fit.tau_band,
                    "spread": jnd_fit.spread,
                    "spread_ci_graph_clusters": list(jnd_fit.spread_ci_graph_clusters),
                    "spread_ci_generator_families": list(jnd_fit.spread_ci_generator_families),
                    "unestimated_cells": [list(cell) for cell in jnd_fit.unestimated_cells],
                    "effective_dof": jnd_fit.effective_dof,
                    "c06_component_audit": {
                        name: asdict(audit) for name, audit in jnd_fit.c06_component_audit.items()
                    },
                    "n_jnd": jnd_fit.n_jnd,
                    "variance_boundary_disclosures": [
                        asdict(disclosure) for disclosure in jnd_fit.variance_boundary_disclosures
                    ],
                    "split_half_frozen": list(jnd_fit.split_half.frozen_components),
                    "c06_shrink_actions": list(jnd_fit.c06_shrink_actions),
                    "c06_partial_declaration": jnd_fit.c06_partial_declaration,
                },
                "h_jnd_branch": asdict(branch),
                "side_swap_audit": (None if side_swap_result is None else asdict(side_swap_result)),
                "start_conditions": asdict(start_conditions),
                "half_assignment_digest": (
                    None if half_assignment is None else half_assignment.table_sha256
                ),
                "access_budget_after": dict(budget_after),
                "ledger_annulments": list(ledger.annulment_lines(_FROZEN_A15_ROLE_HASH)),
                "ledger_defects": list(ledger.ledger_defects(_FROZEN_A15_ROLE_HASH)),
                "iterations": len(trajectory),
                "converged": True,
            },
        )
        _atomic_write_json(output / "status.json", {"state": "COMPLETE"})
        return Freeze1FitResult(
            weight_fit=weight_fit,
            outer_weight_stability=outer_stability,
            lapse_rate=current_lapse,
            lapse_interval=lapse_interval,
            lapse_prior_weight=lapse_prior_weight,
            lapse_boundary_disclosure=lapse_boundary,
            lapse_prior_free_sensitivity=lapse_sensitivity,
            jnd_fit=jnd_fit,
            h_jnd_branch=branch,
            side_swap=side_swap_result,
            start_conditions=start_conditions,
            half_assignment_digest=(
                None if half_assignment is None else half_assignment.table_sha256
            ),
            trajectory=tuple(trajectory),
            access_budget_before=budget_before,
            access_budget_after=budget_after,
            run_dir=output,
        )
    except Exception as error:
        _atomic_write_json(
            output / "status.json",
            {"state": "FAILED", "error_type": type(error).__name__, "message": str(error)},
        )
        raise
