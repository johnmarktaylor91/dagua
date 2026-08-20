"""Era-aware loading of screened RULER V4 judgment-bank rows."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from types import MappingProxyType
from typing import Any, Iterable, Mapping, Optional, Tuple, Union

PathLike = Union[str, Path]
_NON_FITTING_BUDGET_LINES = frozenset({"CONTROLS", "MULTI-CONFIG"})
_PROHIBITED_BANK_SUBTREES = frozenset({"pilot", "sealed"})


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
    """

    files: int
    raw_rows: int
    included_rows: int
    excluded_rejected_session: int
    excluded_invalid: int
    excluded_controls: int
    excluded_unscheduled: int
    excluded_era: int


@dataclass(frozen=True)
class JudgmentBank:
    """Return immutable judgment rows plus their loader audit.

    Parameters
    ----------
    _rows : tuple[JudgmentRow, ...]
        Stable presentation-sorted reusable inputs. TEST labels are absent.
    _test_refs : tuple[_SealedJudgmentRef, ...]
        Stable opaque TEST source references without verdict or tie fields.
    role_hash : str
        Frozen A15 role-assignment identity binding the TEST access record.
    expected_test_presentations : tuple[str, ...]
        Complete scheduled presentation census for guarded TEST roles.
    report : BankLoadReport
        Inclusion and exclusion audit.
    """

    _rows: Tuple[JudgmentRow, ...]
    _test_refs: Tuple[_SealedJudgmentRef, ...]
    role_hash: str
    expected_test_presentations: Tuple[str, ...]
    report: BankLoadReport

    @property
    def rows(self) -> Tuple[JudgmentRow, ...]:
        """Return reusable rows without exposing once-only TEST labels.

        Returns
        -------
        tuple[JudgmentRow, ...]
            Fit, validation, reusable holdout, and diagnostic rows only.
        """

        return self._rows

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

        if purpose is SplitPurpose.TEST:
            raise ValueError("A15 TEST rows require TestHoldoutGuard.consume()")
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


def _reveal_test_rows(refs: Tuple[_SealedJudgmentRef, ...]) -> Tuple[JudgmentRow, ...]:
    """Re-read guarded TEST labels from their exact bank locations.

    Parameters
    ----------
    refs : tuple[_SealedJudgmentRef, ...]
        Opaque references released by a :class:`JudgmentBank`.

    Returns
    -------
    tuple[JudgmentRow, ...]
        Materialized TEST rows in stable presentation order.

    Raises
    ------
    ValueError
        If a source row moved, changed identity, or has invalid labels.
    """

    by_path: dict[Path, dict[int, _SealedJudgmentRef]] = {}
    for ref in refs:
        by_path.setdefault(Path(ref.source_path), {})[ref.source_line] = ref
    rows = []
    for path, expected in sorted(by_path.items(), key=lambda item: str(item[0])):
        found = set()
        for line_number, raw in _read_jsonl_with_line_numbers(path):
            ref = expected.get(line_number)
            if ref is None:
                continue
            found.add(line_number)
            if (
                str(raw.get("session_id", "")) != ref.row_fields["session_id"]
                or str(raw.get("presentation_id", "")) != ref.row_fields["presentation_id"]
            ):
                raise ValueError("guarded TEST source identity changed")
            verdict = int(raw.get("verdict", 0))
            tie = bool(raw.get("tie", verdict == 0))
            if verdict < -3 or verdict > 3:
                raise ValueError(f"verdict outside A13 range: {verdict}")
            confidence = int(raw.get("confidence", 0))
            if confidence not in (1, 2, 3):
                raise ValueError(f"confidence outside A13 range: {confidence}")
            if tie != (verdict == 0):
                raise ValueError("guarded TEST verdict/tie fields are inconsistent")
            rows.append(
                JudgmentRow(
                    **ref.row_fields,
                    verdict=verdict,
                    tie=tie,
                    confidence=confidence,
                )
            )
        if found != set(expected):
            raise ValueError(f"guarded TEST source rows missing from {path}")
    return tuple(sorted(rows, key=lambda row: (row.session_id, row.presentation_id)))


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
    return graphs, role_hash


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
    expected_test_presentations = tuple(
        sorted(
            scheduled.presentation_id
            for scheduled in schedule.values()
            if scheduled.control_type is None
            and scheduled.budget_line not in _NON_FITTING_BUDGET_LINES
            and scheduled.graph_hash in graph_map
            and _ROLE_PURPOSE.get(str(graph_map[scheduled.graph_hash].get("role", "")))
            is SplitPurpose.TEST
        )
    )
    rows = []
    test_refs = []
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
            if raw.get("malformed") is True or raw.get("abstain") is True:
                counts["excluded_invalid"] += 1
                continue
            key = str(raw.get("session_id", "")), str(raw.get("presentation_id", ""))
            scheduled = schedule.get(key)
            if scheduled is None:
                counts["excluded_unscheduled"] += 1
                continue
            if int(raw.get("side_bit", -1)) not in (0, 1):
                raise ValueError(f"bank side_bit outside {{0, 1}}: {key}")
            # The schedule already stores the rendered A/B order. ``side_bit``
            # is only canonical-orientation metadata and must not flip it again.
            if any(
                (
                    str(raw.get("base_pair_id")) != scheduled.base_pair_id,
                    str(raw.get("graph_hash")) != scheduled.graph_hash,
                )
            ):
                raise ValueError(f"bank/schedule identity mismatch: {key}")
            control_type = raw.get("control_type", scheduled.control_type)
            budget_line = str(raw.get("budget_line", scheduled.budget_line))
            if control_type is not None or budget_line in _NON_FITTING_BUDGET_LINES:
                counts["excluded_controls"] += 1
                continue
            row_era = _era(raw.get("judge_id", ""))
            row_instrument = str(raw.get("instrument_hash", ""))
            if (era is not None and row_era != era) or (
                instrument_hash is not None and row_instrument != instrument_hash
            ):
                counts["excluded_era"] += 1
                continue
            graph = graph_map.get(scheduled.graph_hash)
            if graph is None:
                raise ValueError(f"graph absent from frozen A15 map: {scheduled.graph_hash}")
            role = str(graph.get("role", ""))
            purpose = _ROLE_PURPOSE.get(role)
            if purpose is None:
                raise ValueError(f"unknown frozen A15 role: {role!r}")
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
            row_fields = {
                "presentation_id": scheduled.presentation_id,
                "session_id": scheduled.session_id,
                "base_pair_id": scheduled.base_pair_id,
                "graph_hash": scheduled.graph_hash,
                "blind_id_a": scheduled.blind_id_a,
                "blind_id_b": scheduled.blind_id_b,
                "instrument_hash": row_instrument,
                "era": row_era,
                "observation_profile": scheduled.profile_opaque_id,
                "is_replication": bool(raw.get("is_replication", budget_line == "REPLICATION")),
                "role": role,
                "purpose": purpose,
                "primary_class": str(graph.get("primary_class", "")),
                "size_band": str(graph.get("size_band", "")),
                "generator_family": str(graph.get("generator_family", "")),
                "source_path": str(path),
            }
            if purpose is SplitPurpose.TEST:
                test_refs.append(
                    _SealedJudgmentRef(
                        source_path=str(path), source_line=source_line, row_fields=row_fields
                    )
                )
            else:
                rows.append(
                    JudgmentRow(
                        **row_fields,
                        verdict=verdict,
                        tie=tie,
                        confidence=confidence,
                    )
                )
    ordered = tuple(sorted(rows, key=lambda row: (row.session_id, row.presentation_id)))
    report = BankLoadReport(
        files=len(paths),
        raw_rows=counts["raw_rows"],
        included_rows=len(ordered) + len(test_refs),
        excluded_rejected_session=counts["excluded_rejected_session"],
        excluded_invalid=counts["excluded_invalid"],
        excluded_controls=counts["excluded_controls"],
        excluded_unscheduled=counts["excluded_unscheduled"],
        excluded_era=counts["excluded_era"],
    )
    ordered_refs = tuple(
        sorted(
            test_refs,
            key=lambda ref: (
                str(ref.row_fields["session_id"]),
                str(ref.row_fields["presentation_id"]),
            ),
        )
    )
    return JudgmentBank(
        _rows=ordered,
        _test_refs=ordered_refs,
        role_hash=role_hash,
        expected_test_presentations=expected_test_presentations,
        report=report,
    )
