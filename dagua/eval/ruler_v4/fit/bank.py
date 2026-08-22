"""Era-aware loading of screened RULER V4 judgment-bank rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, replace
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple, Union, cast

PathLike = Union[str, Path]
_NON_FITTING_BUDGET_LINES = frozenset({"CONTROLS", "MULTI-CONFIG"})
_PROHIBITED_BANK_SUBTREES = frozenset({"pilot", "sealed"})
_FROZEN_A15_ROLE_HASH = (
    "efe09188d2f1680a6ca55857e1db0c5d7a9201c2e66600d4fdbec057514c0b02"  # pragma: allowlist secret
)
_FROZEN_A16_SCHEDULE_DIGEST = (
    "deb598b397523d17e8bd1d39d996de9965257414c72507daf3fe046865276020"  # pragma: allowlist secret
)
_TEST_ONLY_A16_DIGESTS_BY_ROLE_HASH: Mapping[str, str] = MappingProxyType({})
SEALED_TEST_ROLES = frozenset({"within-family-sealed", "cross-family-sealed"})


class SplitPurpose(str, Enum):
    """Declare how one frozen A15 graph role may be consumed."""

    FIT = "fit"
    VALIDATE = "validate"
    TEST = "test"
    REUSABLE_HOLDOUT = "reusable_holdout"
    DIAGNOSTIC = "diagnostic"


_ROLE_PURPOSE = MappingProxyType(
    {
        "train": SplitPurpose.FIT,
        "within-family-calibration": SplitPurpose.VALIDATE,
        "cross-family-calibration": SplitPurpose.VALIDATE,
        "within-family-sealed": SplitPurpose.TEST,
        "cross-family-sealed": SplitPurpose.TEST,
        "entire-class-holdout": SplitPurpose.REUSABLE_HOLDOUT,
        "adversarial": SplitPurpose.DIAGNOSTIC,
    }
)
CENSUS_BOUND_ROLES = frozenset(
    role for role, purpose in _ROLE_PURPOSE.items() if purpose is not SplitPurpose.FIT
)


@dataclass(frozen=True)
class ScheduledPair:
    """Hold the schedule-owned identities absent from a judged row.

    Parameters
    ----------
    presentation_id : str
        Unique scheduled presentation identity.
    session_id : str
        Session that was allowed to judge the presentation.
    base_pair_id : str
        Stable unordered drawing-pair identity.
    graph_hash : str
        Canonical graph identity.
    blind_id_a, blind_id_b : str
        Drawing identities in the displayed A/B order.
    profile_opaque_id : str
        Immutable observation-profile identity.
    budget_line : str
        Frozen campaign budget line.
    control_type : str or None
        Named control class, or ``None`` for a fitting-eligible presentation.
    replicate_group_id : str
        Realized repeat-group identity used instead of the advisory flag.
    """

    presentation_id: str
    session_id: str
    base_pair_id: str
    graph_hash: str
    blind_id_a: str
    blind_id_b: str
    profile_opaque_id: str
    budget_line: str
    control_type: Optional[str]
    replicate_group_id: str


@dataclass(frozen=True)
class JudgmentRow:
    """Represent one accepted, scheduled, non-control bank judgment.

    Parameters
    ----------
    presentation_id, session_id, base_pair_id, graph_hash : str
        Frozen campaign identities.
    blind_id_a, blind_id_b : str
        Schedule-owned drawing ids in displayed A/B order.
    instrument_hash : str
        Exact judge-instrument hash; never pooled implicitly.
    era : str
        Judge configuration suffix such as ``CF@1`` or ``CF@4``.
    observation_profile : str
        Immutable opaque observation-profile id.
    verdict : int
        Graded verdict in ``[-3, 3]``. Negative favors A; positive favors B.
    tie : bool
        Explicit A13 tie indicator.
    confidence : int
        A13 confidence response in ``[1, 3]``.
    is_replication : bool
        Whether the row belongs to the cross-session replication line.
    replicate_group_id : str
        Realized repeat-group identity.
    role : str
        Frozen A15 graph role.
    purpose : SplitPurpose
        Allowed fit/validation/test/diagnostic use of ``role``.
    primary_class, size_band, generator_family : str
        A15 strata used by JND-HET and robustness diagnostics.
    source_path : str
        JSONL bank file that supplied the judged row.
    """

    presentation_id: str
    session_id: str
    base_pair_id: str
    graph_hash: str
    blind_id_a: str
    blind_id_b: str
    instrument_hash: str
    era: str
    observation_profile: str
    verdict: int
    tie: bool
    confidence: int
    is_replication: bool
    replicate_group_id: str
    role: str
    purpose: SplitPurpose
    primary_class: str
    size_band: str
    generator_family: str
    source_path: str

    @property
    def outcome(self) -> int:
        """Return the three-way fitting outcome.

        Returns
        -------
        int
            ``-1`` for A, ``0`` for tie, and ``1`` for B.
        """

        if self.tie or self.verdict == 0:
            return 0
        return 1 if self.verdict > 0 else -1

    @property
    def era_stratum(self) -> Tuple[str, str]:
        """Return the non-poolable instrument/era stratum.

        Returns
        -------
        tuple[str, str]
            Exact ``(instrument_hash, era)`` key.
        """

        return self.instrument_hash, self.era


@dataclass(frozen=True)
class SideSwapAuditRow:
    """Carry one exchanged-order control leg outside every likelihood type.

    Parameters
    ----------
    replicate_group_id, base_pair_id, session_id : str
        Frozen join and presentation identities.
    blind_id_a, blind_id_b : str
        Schedule-owned displayed order for the control leg.
    graded_verdict : int
        A13 response in the control leg's displayed orientation.
    """

    replicate_group_id: str
    base_pair_id: str
    session_id: str
    blind_id_a: str
    blind_id_b: str
    graded_verdict: int

    def __post_init__(self) -> None:
        """Validate the audit-only row without widening fitting schemas.

        Raises
        ------
        ValueError
            If an identity is empty or the verdict is outside ``[-3, 3]``.
        """

        if not all(
            (
                self.replicate_group_id,
                self.base_pair_id,
                self.session_id,
                self.blind_id_a,
                self.blind_id_b,
            )
        ):
            raise ValueError("side-swap audit rows require complete join identities")
        if self.graded_verdict not in range(-3, 4):
            raise ValueError("side-swap audit verdict must lie in [-3, 3]")


@dataclass(frozen=True)
class JudgmentMetadata:
    """Expose only LOOK-LEDGER-whitelisted judgment metadata.

    Parameters
    ----------
    presentation_id, session_id, base_pair_id, graph_hash : str
        Frozen campaign identities.
    instrument_hash, era, observation_profile : str
        Provenance and non-poolable likelihood strata.
    is_replication : bool
        Whether the presentation belongs to the replication line.
    replicate_group_id : str
        Realized repeat-group identity.
    role : str
        Frozen A15 graph role.
    purpose : SplitPurpose
        Allowed consumption purpose.
    primary_class, size_band, generator_family : str
        Frozen A15 strata.

    Notes
    -----
    The type intentionally has no verdict, tie, abstain, confidence, reason,
    defect, free-note, departure, or raw model-fingerprint field. Unknown A13
    fields are gated by default and never enter this projection.
    """

    presentation_id: str
    session_id: str
    base_pair_id: str
    graph_hash: str
    instrument_hash: str
    era: str
    observation_profile: str
    is_replication: bool
    replicate_group_id: str
    role: str
    purpose: SplitPurpose
    primary_class: str
    size_band: str
    generator_family: str


def _metadata_from_fields(fields: Mapping[str, object]) -> JudgmentMetadata:
    """Project internal row fields onto the frozen metadata whitelist.

    Parameters
    ----------
    fields : mapping[str, object]
        Internal judgment constructor fields without labels.

    Returns
    -------
    JudgmentMetadata
        Whitelist-only immutable projection.
    """

    return JudgmentMetadata(
        presentation_id=str(fields["presentation_id"]),
        session_id=str(fields["session_id"]),
        base_pair_id=str(fields["base_pair_id"]),
        graph_hash=str(fields["graph_hash"]),
        instrument_hash=str(fields["instrument_hash"]),
        era=str(fields["era"]),
        observation_profile=str(fields["observation_profile"]),
        is_replication=bool(fields["is_replication"]),
        replicate_group_id=str(fields["replicate_group_id"]),
        role=str(fields["role"]),
        purpose=cast(SplitPurpose, fields["purpose"]),
        primary_class=str(fields["primary_class"]),
        size_band=str(fields["size_band"]),
        generator_family=str(fields["generator_family"]),
    )


@dataclass(frozen=True)
class _SealedJudgmentRef:
    """Retain TEST provenance without materializing its verdict.

    Parameters
    ----------
    source_path : str
        Bank JSONL path containing the guarded row.
    source_line : int
        One-based JSONL line number.
    row_fields : mapping[str, object]
        Non-label :class:`JudgmentRow` constructor fields.
    """

    source_path: str
    source_line: int
    row_fields: Mapping[str, object]

    def __post_init__(self) -> None:
        """Freeze non-label row metadata.

        Raises
        ------
        ValueError
            If the source locator is invalid.
        """

        if not self.source_path or self.source_line <= 0:
            raise ValueError("sealed judgment references require a source locator")
        object.__setattr__(self, "row_fields", MappingProxyType(dict(self.row_fields)))


@dataclass(frozen=True)
class BankLoadReport:
    """Publish deterministic loader inclusion and exclusion counts.

    Parameters
    ----------
    files : int
        Number of bank JSONL files read.
    raw_rows : int
        Total decoded bank rows.
    included_rows : int
        Accepted scheduled non-control rows returned.
    excluded_rejected_session : int
        Rows from sessions that did not pass screening.
    excluded_invalid : int
        Malformed or abstaining rows.
    excluded_controls : int
        Control and multi-config rows excluded from likelihood data.
    excluded_unscheduled : int
        Rows without an exact schedule join.
    excluded_era : int
        Rows removed by explicit era or instrument filters.
    side_swap_audit_rows : int
        Exchanged-order control legs retained only for FIT-ORD(b)'s audit.
    """

    files: int
    raw_rows: int
    included_rows: int
    excluded_rejected_session: int
    excluded_invalid: int
    excluded_controls: int
    excluded_unscheduled: int
    excluded_era: int
    side_swap_audit_rows: int


@dataclass(frozen=True)
class JudgmentBank:
    """Return immutable judgment rows plus their loader audit.

    Parameters
    ----------
    _rows : tuple[JudgmentRow, ...]
        Stable presentation-sorted train inputs. Every non-train label is absent.
    _calibration_refs : tuple[_SealedJudgmentRef, ...]
        Opaque calibration locators released only by the four-look ledger.
    _reusable_refs, _diagnostic_refs : tuple[_SealedJudgmentRef, ...]
        Opaque uncapped labels whose reads are nevertheless ledgered.
    _test_refs : tuple[_SealedJudgmentRef, ...]
        Stable opaque TEST source references without verdict or tie fields.
    _side_swap_audit_rows : tuple[SideSwapAuditRow, ...]
        Train-role exchanged-order controls excluded from likelihood data.
    role_hash : str
        Frozen A15 role-assignment identity binding the TEST access record.
    expected_test_base_pairs : mapping[str, tuple[str, ...]]
        Frozen A16 base-pair census for each guarded TEST role.
    expected_test_graphs : mapping[str, tuple[str, ...]]
        Frozen graph-hash census for each guarded TEST role.
    report : BankLoadReport
        Inclusion and exclusion audit.

    Notes
    -----
    Instances are intentionally not picklable. Their mapping proxies prevent
    multiprocessing or cache serialization from weakening sealed-label opacity.
    """

    _rows: Tuple[JudgmentRow, ...]
    _calibration_refs: Tuple[_SealedJudgmentRef, ...]
    _reusable_refs: Tuple[_SealedJudgmentRef, ...]
    _diagnostic_refs: Tuple[_SealedJudgmentRef, ...]
    _test_refs: Tuple[_SealedJudgmentRef, ...]
    _side_swap_audit_rows: Tuple[SideSwapAuditRow, ...]
    role_hash: str
    expected_test_base_pairs: Mapping[str, Tuple[str, ...]]
    expected_test_graphs: Mapping[str, Tuple[str, ...]]
    report: BankLoadReport

    def __post_init__(self) -> None:
        """Freeze the per-role guarded base-pair census."""

        census = {
            role: tuple(base_pairs) for role, base_pairs in self.expected_test_base_pairs.items()
        }
        graph_census = {
            role: tuple(graph_hashes) for role, graph_hashes in self.expected_test_graphs.items()
        }
        object.__setattr__(self, "expected_test_base_pairs", MappingProxyType(census))
        object.__setattr__(self, "expected_test_graphs", MappingProxyType(graph_census))

    @property
    def rows(self) -> Tuple[JudgmentRow, ...]:
        """Return reusable rows without exposing once-only TEST labels.

        Returns
        -------
        tuple[JudgmentRow, ...]
            Train-role rows only. All non-train labels require the ledger.
        """

        return self._rows

    @property
    def side_swap_audit_rows(self) -> Tuple[SideSwapAuditRow, ...]:
        """Return exchanged-order controls in their audit-only row type.

        Returns
        -------
        tuple[SideSwapAuditRow, ...]
            Stable audit rows that cannot enter the FIT-ORD objective.
        """

        return self._side_swap_audit_rows

    def metadata(
        self,
        purpose: Optional[SplitPurpose] = None,
        observation_profile: Optional[str] = None,
        era: Optional[str] = None,
        instrument_hash: Optional[str] = None,
        replication_only: bool = False,
    ) -> Tuple[JudgmentMetadata, ...]:
        """Return unlimited whitelist-only metadata without taking a look.

        Parameters
        ----------
        purpose : SplitPurpose or None
            Optional frozen A15 consumption purpose.
        observation_profile, era, instrument_hash : str or None
            Optional exact provenance selectors.
        replication_only : bool, default=False
            Restrict metadata to the replication line.

        Returns
        -------
        tuple[JudgmentMetadata, ...]
            Stable metadata projections satisfying every selector.
        """

        row_fields = [
            {
                key: value
                for key, value in vars(row).items()
                if key not in {"verdict", "tie", "confidence"}
            }
            for row in self._rows
        ]
        row_fields.extend(
            dict(ref.row_fields)
            for refs in (
                self._calibration_refs,
                self._reusable_refs,
                self._diagnostic_refs,
                self._test_refs,
            )
            for ref in refs
        )
        metadata = (_metadata_from_fields(fields) for fields in row_fields)
        return tuple(
            sorted(
                (
                    row
                    for row in metadata
                    if (purpose is None or row.purpose is purpose)
                    and (
                        observation_profile is None
                        or row.observation_profile == observation_profile
                    )
                    and (era is None or row.era == era)
                    and (instrument_hash is None or row.instrument_hash == instrument_hash)
                    and (not replication_only or row.is_replication)
                ),
                key=lambda row: (row.session_id, row.presentation_id),
            )
        )

    @property
    def guarded_test_count(self) -> int:
        """Return the hidden TEST-row count without exposing labels.

        Returns
        -------
        int
            Number of rows behind the once-only holdout guard.
        """

        return len(self._test_refs)

    def _partition_test_refs(self) -> Tuple[_SealedJudgmentRef, ...]:
        """Return opaque TEST references to the holdout boundary.

        Returns
        -------
        tuple[_SealedJudgmentRef, ...]
            Internal source references that contain no TEST labels.
        """

        return self._test_refs

    def _partition_gated_refs(self, purpose: SplitPurpose) -> Tuple[_SealedJudgmentRef, ...]:
        """Return opaque non-train references to the ledger boundary.

        Parameters
        ----------
        purpose : SplitPurpose
            Non-train purpose to retrieve.

        Returns
        -------
        tuple[_SealedJudgmentRef, ...]
            Label-free source references.

        Raises
        ------
        ValueError
            If FIT or TEST is requested through this helper.
        """

        by_purpose = {
            SplitPurpose.VALIDATE: self._calibration_refs,
            SplitPurpose.REUSABLE_HOLDOUT: self._reusable_refs,
            SplitPurpose.DIAGNOSTIC: self._diagnostic_refs,
        }
        if purpose not in by_purpose:
            raise ValueError(f"purpose has no reusable gated reference set: {purpose.value}")
        return by_purpose[purpose]

    def select(
        self,
        purpose: Optional[SplitPurpose] = None,
        observation_profile: Optional[str] = None,
        era: Optional[str] = None,
        instrument_hash: Optional[str] = None,
        replication_only: bool = False,
    ) -> Tuple[JudgmentRow, ...]:
        """Select one explicit, non-pooled likelihood stratum.

        Parameters
        ----------
        purpose : SplitPurpose or None
            Optional frozen A15 consumption purpose.
        observation_profile : str or None
            Optional immutable observation profile.
        era : str or None
            Optional judge era such as ``CF@4``.
        instrument_hash : str or None
            Optional exact instrument hash.
        replication_only : bool, default=False
            Restrict to the JND-HET replication line.

        Returns
        -------
        tuple[JudgmentRow, ...]
            Rows satisfying every declared selector.

        Raises
        ------
        ValueError
            If direct TEST selection is attempted outside the holdout guard.
        """

        if purpose is not None and purpose is not SplitPurpose.FIT:
            raise ValueError("non-train judged rows require a LOOK-LEDGER release")
        return tuple(
            row
            for row in self.rows
            if (purpose is None or row.purpose is purpose)
            and (observation_profile is None or row.observation_profile == observation_profile)
            and (era is None or row.era == era)
            and (instrument_hash is None or row.instrument_hash == instrument_hash)
            and (not replication_only or row.is_replication)
        )


def _jsonl_paths(inputs: Iterable[PathLike]) -> Tuple[Path, ...]:
    """Resolve files and directories into a stable JSONL path list.

    Parameters
    ----------
    inputs : iterable[path-like]
        JSONL files or directories recursively containing JSONL files.

    Returns
    -------
    tuple[pathlib.Path, ...]
        De-duplicated resolved paths in lexical order.
    """

    paths = set()
    for value in inputs:
        path = Path(value)
        if path.is_dir():
            paths.update(candidate.resolve() for candidate in path.rglob("*.jsonl"))
        elif path.is_file():
            paths.add(path.resolve())
        else:
            raise FileNotFoundError(path)
    return tuple(sorted(paths))


def _is_prohibited_bank_path(path: Path) -> bool:
    """Return whether a path resolves below ``bank/pilot`` or ``bank/sealed``.

    Parameters
    ----------
    path : pathlib.Path
        Resolved input or discovered JSONL path.

    Returns
    -------
    bool
        True only for one of the quarantined bank subtrees.
    """

    parts = path.resolve().parts
    return any(
        parts[index] == "bank" and parts[index + 1] in _PROHIBITED_BANK_SUBTREES
        for index in range(len(parts) - 1)
    )


def _bank_jsonl_paths(inputs: Iterable[PathLike]) -> Tuple[Path, ...]:
    """Resolve bank inputs while denying pilot and sealed subtrees.

    Parameters
    ----------
    inputs : iterable[path-like]
        Bank JSONL files or containing directories.

    Returns
    -------
    tuple[pathlib.Path, ...]
        Stable authorized bank paths.

    Raises
    ------
    PermissionError
        If an input or recursively discovered file is quarantined.
    """

    roots = tuple(Path(value).resolve() for value in inputs)
    prohibited_roots = [path for path in roots if _is_prohibited_bank_path(path)]
    if prohibited_roots:
        raise PermissionError(f"quarantined bank input denied: {prohibited_roots[0]}")
    paths = _jsonl_paths(roots)
    prohibited_paths = [path for path in paths if _is_prohibited_bank_path(path)]
    if prohibited_paths:
        raise PermissionError(f"recursive quarantined bank input denied: {prohibited_paths[0]}")
    return paths


def _read_jsonl(path: Path) -> Iterable[Mapping[str, Any]]:
    """Yield decoded object rows from one JSONL file.

    Parameters
    ----------
    path : pathlib.Path
        Existing JSONL path.

    Yields
    ------
    mapping[str, Any]
        One decoded object per nonblank line.

    Raises
    ------
    ValueError
        If a line is not a JSON object.
    """

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield value


def _read_jsonl_with_line_numbers(path: Path) -> Iterable[Tuple[int, Mapping[str, Any]]]:
    """Yield decoded JSON objects with stable one-based line numbers.

    Parameters
    ----------
    path : pathlib.Path
        Existing JSONL path.

    Yields
    ------
    tuple[int, mapping[str, Any]]
        Source line number and decoded object.

    Raises
    ------
    ValueError
        If a nonblank line is not a JSON object.
    """

    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"{path}:{line_number}: expected a JSON object")
            yield line_number, value


def load_schedule(inputs: Iterable[PathLike]) -> Mapping[Tuple[str, str], ScheduledPair]:
    """Load the exact session schedule used to authorize bank rows.

    Parameters
    ----------
    inputs : iterable[path-like]
        Session-manifest JSONL files or containing directories.

    Returns
    -------
    mapping[tuple[str, str], ScheduledPair]
        Immutable lookup keyed by ``(session_id, presentation_id)``.

    Raises
    ------
    ValueError
        If duplicate keys disagree or a schedule row lacks required identities.
    """

    schedule = {}
    for path in _jsonl_paths(inputs):
        for raw in _read_jsonl(path):
            if "presentation_id" not in raw or "blind_id_A" not in raw:
                raise ValueError(f"{path}: schedule row lacks presentation identities")
            try:
                row = ScheduledPair(
                    presentation_id=str(raw["presentation_id"]),
                    session_id=str(raw["session_id"]),
                    base_pair_id=str(raw["base_pair_id"]),
                    graph_hash=str(raw["graph_hash"]),
                    blind_id_a=str(raw["blind_id_A"]),
                    blind_id_b=str(raw["blind_id_B"]),
                    profile_opaque_id=str(
                        raw.get("profile_opaque_id", raw.get("opaque_profile_id", ""))
                    ),
                    budget_line=str(raw.get("budget_line", "PRIMARY")),
                    control_type=(
                        None if raw.get("control_type") is None else str(raw["control_type"])
                    ),
                    replicate_group_id=str(
                        raw.get("replicate_group_id", raw.get("base_pair_id", ""))
                    ),
                )
            except KeyError as error:
                raise ValueError(f"{path}: incomplete schedule row: {error}") from error
            if not all(
                (
                    row.presentation_id,
                    row.session_id,
                    row.base_pair_id,
                    row.graph_hash,
                    row.blind_id_a,
                    row.blind_id_b,
                    row.profile_opaque_id,
                    row.replicate_group_id,
                )
            ):
                raise ValueError(f"{path}: schedule identities must be nonempty")
            key = row.session_id, row.presentation_id
            prior = schedule.get(key)
            if prior is not None and prior != row:
                raise ValueError(f"conflicting duplicate schedule row: {key}")
            schedule[key] = row
    return MappingProxyType(schedule)


def _load_a15_family_map(
    path: PathLike,
) -> Tuple[Mapping[str, Mapping[str, Any]], str]:
    """Load and verify graph roles from the frozen A15 family map.

    Parameters
    ----------
    path : path-like
        Frozen ``A15_FAMILY_MAP.json`` path.

    Returns
    -------
    tuple[mapping[str, mapping[str, Any]], str]
        Graph metadata keyed by canonical hash and the verified role hash.

    Raises
    ------
    ValueError
        If the graph mapping or its frozen role hash is absent or inconsistent.
    """

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    graphs = payload.get("graphs") if isinstance(payload, dict) else None
    if not isinstance(graphs, dict):
        raise ValueError("A15 family map has no graph mapping")
    role_hash = payload.get("role_hash")
    if not isinstance(role_hash, str) or not role_hash:
        raise ValueError("A15 family map has no frozen role hash")
    role_lines = []
    for graph_hash, graph in sorted(graphs.items()):
        if not isinstance(graph, dict) or not isinstance(graph.get("role"), str):
            raise ValueError(f"A15 graph lacks a frozen role: {graph_hash}")
        role_lines.append(f"{graph_hash}\t{graph['role']}")
    computed = hashlib.sha256("\n".join(role_lines).encode("utf-8")).hexdigest()
    if computed != role_hash:
        raise ValueError("A15 family-map role hash does not match its graph census")
    if role_hash != _FROZEN_A15_ROLE_HASH:
        raise ValueError("A15 family-map role hash is not the frozen campaign identity")
    return graphs, role_hash


def _load_frozen_role_census(
    path: PathLike,
    role_hash: str,
    graph_map: Mapping[str, Mapping[str, Any]],
) -> Mapping[str, Tuple[str, ...]]:
    """Load every guarded-role base-pair census from pinned schedule bytes.

    Parameters
    ----------
    path : path-like
        Frozen A16-materialized presentation schedule.
    role_hash : str
        Verified A15 role hash selecting the pinned or synthetic-test digest.
    graph_map : mapping[str, mapping[str, Any]]
        Verified frozen A15 graph-role census.

    Returns
    -------
    mapping[str, tuple[str, ...]]
        Base-pair identities grouped by non-training role.

    Raises
    ------
    ValueError
        If the digest, schema, or A15/A16 role reconciliation disagrees.
    """

    expected_digest = _FROZEN_A16_SCHEDULE_DIGEST
    if role_hash != _FROZEN_A15_ROLE_HASH:
        expected_digest = _TEST_ONLY_A16_DIGESTS_BY_ROLE_HASH.get(role_hash, expected_digest)
    payload = Path(path).read_bytes()
    if hashlib.sha256(payload).hexdigest() != expected_digest:
        raise ValueError("frozen presentation-schedule digest does not match")
    census: dict[str, list[str]] = {role: [] for role in sorted(CENSUS_BOUND_ROLES)}
    for line_number, line in enumerate(payload.decode("utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        raw = json.loads(line)
        if not isinstance(raw, dict):
            raise ValueError(f"{path}:{line_number}: expected a JSON object")
        partition = str(raw.get("partition", ""))
        graph_hash = str(raw.get("graph_hash", ""))
        graph = graph_map.get(graph_hash)
        if graph is None or str(graph.get("role", "")) != partition:
            raise ValueError("frozen A15/A16 role census does not reconcile")
        if partition in CENSUS_BOUND_ROLES:
            base_pair_id = str(raw.get("base_pair_id", ""))
            if not base_pair_id:
                raise ValueError("frozen guarded schedule row lacks base-pair identity")
            census[partition].append(base_pair_id)
    return MappingProxyType(
        {role: tuple(sorted(set(base_pairs))) for role, base_pairs in census.items()}
    )


def _era(judge_id: object) -> str:
    """Extract the frozen judge configuration suffix.

    Parameters
    ----------
    judge_id : object
        Banked judge identity.

    Returns
    -------
    str
        Configuration suffix following ``/``.
    """

    value = str(judge_id)
    return value.rsplit("/", 1)[-1]


def load_bank(
    bank_inputs: Iterable[PathLike],
    schedule_inputs: Iterable[PathLike],
    family_map_path: PathLike,
    frozen_schedule_path: PathLike,
    era: Optional[str] = None,
    instrument_hash: Optional[str] = None,
) -> JudgmentBank:
    """Load accepted scheduled judgments without crossing frozen strata.

    Controls and multi-config rows never enter the returned likelihood data.
    Replication rows remain available because JND-HET is fitted exclusively
    from that line. The loader does not pool instrument hashes, observation
    profiles, eras, or A15 purposes; each remains an explicit row field.

    Parameters
    ----------
    bank_inputs : iterable[path-like]
        Bank JSONL files or directories.
    schedule_inputs : iterable[path-like]
        Exact session manifests that scheduled those bank rows.
    family_map_path : path-like
        Frozen ``A15_FAMILY_MAP.json``.
    frozen_schedule_path : path-like
        Frozen A16-materialized presentation schedule used only for the sealed census.
    era : str or None
        Optional exact era selector.
    instrument_hash : str or None
        Optional exact instrument selector.

    Returns
    -------
    JudgmentBank
        Immutable rows and a complete exclusion audit.

    Raises
    ------
    ValueError
        If a scheduled join disagrees with the bank or an A15 role is unknown.
    """

    paths = _bank_jsonl_paths(bank_inputs)
    schedule = load_schedule(schedule_inputs)
    graph_map, role_hash = _load_a15_family_map(family_map_path)
    expected_test_base_pairs = _load_frozen_role_census(frozen_schedule_path, role_hash, graph_map)
    expected_test_graphs = {
        role: tuple(
            sorted(
                graph_hash
                for graph_hash, graph in graph_map.items()
                if str(graph.get("role", "")) == role
            )
        )
        for role in sorted(CENSUS_BOUND_ROLES)
    }
    rows = []
    calibration_refs = []
    reusable_refs = []
    diagnostic_refs = []
    test_refs = []
    side_swap_audit_rows = []
    scheduled_matches = 0
    counts = {
        "raw_rows": 0,
        "excluded_rejected_session": 0,
        "excluded_invalid": 0,
        "excluded_controls": 0,
        "excluded_unscheduled": 0,
        "excluded_era": 0,
    }
    for path in paths:
        for source_line, raw in _read_jsonl_with_line_numbers(path):
            counts["raw_rows"] += 1
            if raw.get("session_accepted") is not True:
                counts["excluded_rejected_session"] += 1
                continue
            key = str(raw.get("session_id", "")), str(raw.get("presentation_id", ""))
            scheduled = schedule.get(key)
            if scheduled is None and key[0].startswith("main-"):
                scheduled = schedule.get((key[0].removeprefix("main-"), key[1]))
            if scheduled is None:
                counts["excluded_unscheduled"] += 1
                continue
            scheduled_matches += 1
            if int(raw.get("side_bit", -1)) not in (0, 1):
                raise ValueError(f"bank side_bit outside {{0, 1}}: {key}")
            # The schedule already stores the rendered A/B order. ``side_bit``
            # is only canonical-orientation metadata and must not flip it again.
            if any(
                (
                    str(raw.get("base_pair_id")) != scheduled.base_pair_id,
                    str(raw.get("graph_hash")) != scheduled.graph_hash,
                    str(raw.get("replicate_group_id", scheduled.replicate_group_id))
                    != scheduled.replicate_group_id,
                )
            ):
                raise ValueError(f"bank/schedule identity mismatch: {key}")
            control_type = raw.get("control_type", scheduled.control_type)
            budget_line = str(raw.get("budget_line", scheduled.budget_line))
            graph = graph_map.get(scheduled.graph_hash)
            if graph is None:
                raise ValueError(f"graph absent from frozen A15 map: {scheduled.graph_hash}")
            role = str(graph.get("role", ""))
            purpose = _ROLE_PURPOSE.get(role)
            if purpose is None:
                raise ValueError(f"unknown frozen A15 role: {role!r}")
            if control_type is not None or budget_line in _NON_FITTING_BUDGET_LINES:
                if control_type == "side-swap-repeat" and purpose is SplitPurpose.FIT:
                    if raw.get("malformed") is not True and raw.get("abstain") is not True:
                        verdict = int(raw.get("verdict", 0))
                        if verdict not in range(-3, 4):
                            raise ValueError(f"verdict outside A13 range: {verdict}")
                        side_swap_audit_rows.append(
                            SideSwapAuditRow(
                                replicate_group_id=scheduled.replicate_group_id,
                                base_pair_id=scheduled.base_pair_id,
                                session_id=key[0],
                                blind_id_a=scheduled.blind_id_a,
                                blind_id_b=scheduled.blind_id_b,
                                graded_verdict=verdict,
                            )
                        )
                counts["excluded_controls"] += 1
                continue
            row_era = _era(raw.get("judge_id", ""))
            row_instrument = str(raw.get("instrument_hash", ""))
            if (era is not None and row_era != era) or (
                instrument_hash is not None and row_instrument != instrument_hash
            ):
                counts["excluded_era"] += 1
                continue
            row_fields = {
                "presentation_id": scheduled.presentation_id,
                "session_id": key[0],
                "base_pair_id": scheduled.base_pair_id,
                "graph_hash": scheduled.graph_hash,
                "blind_id_a": scheduled.blind_id_a,
                "blind_id_b": scheduled.blind_id_b,
                "instrument_hash": row_instrument,
                "era": row_era,
                "observation_profile": scheduled.profile_opaque_id,
                "is_replication": False,
                "replicate_group_id": scheduled.replicate_group_id,
                "role": role,
                "purpose": purpose,
                "primary_class": str(graph.get("primary_class", "")),
                "size_band": str(graph.get("size_band", "")),
                "generator_family": str(graph.get("generator_family", "")),
                "source_path": str(path),
            }
            if purpose is SplitPurpose.FIT:
                if raw.get("malformed") is True or raw.get("abstain") is True:
                    counts["excluded_invalid"] += 1
                    continue
                verdict = int(raw.get("verdict", 0))
                if verdict < -3 or verdict > 3:
                    raise ValueError(f"verdict outside A13 range: {verdict}")
                tie = bool(raw.get("tie", verdict == 0))
                if verdict == 0 and not tie:
                    counts["excluded_invalid"] += 1
                    continue
                if tie != (verdict == 0):
                    raise ValueError("A13 verdict and tie fields are inconsistent")
                confidence = int(raw.get("confidence", 0))
                if confidence not in (1, 2, 3):
                    raise ValueError(f"confidence outside A13 range: {confidence}")
                rows.append(
                    JudgmentRow(
                        **row_fields,
                        verdict=verdict,
                        tie=tie,
                        confidence=confidence,
                    )
                )
            else:
                ref = _SealedJudgmentRef(
                    source_path=str(path), source_line=source_line, row_fields=row_fields
                )
                {
                    SplitPurpose.VALIDATE: calibration_refs,
                    SplitPurpose.REUSABLE_HOLDOUT: reusable_refs,
                    SplitPurpose.DIAGNOSTIC: diagnostic_refs,
                    SplitPurpose.TEST: test_refs,
                }[purpose].append(ref)
    if paths and scheduled_matches == 0:
        raise ValueError("bank/schedule join matched zero rows")
    by_replicate_group: dict[str, list[JudgmentRow]] = {}
    for row in rows:
        by_replicate_group.setdefault(row.replicate_group_id, []).append(row)
    qualified_replication_groups = {
        group_id
        for group_id, members in by_replicate_group.items()
        if len(members) >= 2 and len({member.session_id for member in members}) >= 2
    }
    ordered = tuple(
        sorted(
            (
                replace(
                    row,
                    is_replication=row.replicate_group_id in qualified_replication_groups,
                )
                for row in rows
            ),
            key=lambda row: (row.session_id, row.presentation_id),
        )
    )
    report = BankLoadReport(
        files=len(paths),
        raw_rows=counts["raw_rows"],
        included_rows=(
            len(ordered)
            + len(calibration_refs)
            + len(reusable_refs)
            + len(diagnostic_refs)
            + len(test_refs)
        ),
        excluded_rejected_session=counts["excluded_rejected_session"],
        excluded_invalid=counts["excluded_invalid"],
        excluded_controls=counts["excluded_controls"],
        excluded_unscheduled=counts["excluded_unscheduled"],
        excluded_era=counts["excluded_era"],
        side_swap_audit_rows=len(side_swap_audit_rows),
    )

    def ordered_refs(refs: list[_SealedJudgmentRef]) -> Tuple[_SealedJudgmentRef, ...]:
        """Sort opaque source references by stable campaign identity.

        Parameters
        ----------
        refs : list[_SealedJudgmentRef]
            Mutable loader accumulator.

        Returns
        -------
        tuple[_SealedJudgmentRef, ...]
            Stable immutable references.
        """

        return tuple(
            sorted(
                refs,
                key=lambda ref: (
                    str(ref.row_fields["session_id"]),
                    str(ref.row_fields["presentation_id"]),
                ),
            )
        )

    ordered_test_refs = ordered_refs(test_refs)
    return JudgmentBank(
        _rows=ordered,
        _calibration_refs=ordered_refs(calibration_refs),
        _reusable_refs=ordered_refs(reusable_refs),
        _diagnostic_refs=ordered_refs(diagnostic_refs),
        _test_refs=ordered_test_refs,
        _side_swap_audit_rows=tuple(
            sorted(
                side_swap_audit_rows,
                key=lambda row: (row.replicate_group_id, row.session_id, row.base_pair_id),
            )
        ),
        role_hash=role_hash,
        expected_test_base_pairs=expected_test_base_pairs,
        expected_test_graphs=expected_test_graphs,
        report=report,
    )
