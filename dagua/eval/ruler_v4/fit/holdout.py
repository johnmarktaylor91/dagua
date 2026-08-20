"""Fail-closed A15 fit/validation/test access discipline."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Tuple, Union

from dagua.eval.ruler_v4.fit.bank import (
    SEALED_TEST_ROLES,
    JudgmentBank,
    JudgmentRow,
    SplitPurpose,
    _read_jsonl_with_line_numbers,
    _SealedJudgmentRef,
)

_ACCESS_LEDGER_ROOT = Path(__file__).resolve().parents[4] / "p3/gate/ACCESS_LEDGER"


class TestHoldoutConsumedError(RuntimeError):
    """Signal that the once-only A15 test bank has already been touched."""


@dataclass(frozen=True)
class HoldoutPartitions:
    """Expose reusable roles while keeping test rows behind a guard.

    Parameters
    ----------
    fit : tuple[JudgmentRow, ...]
        A15 train-role rows.
    validate : tuple[JudgmentRow, ...]
        Within- and cross-family calibration rows.
    diagnostic : tuple[JudgmentRow, ...]
        Reusable adversarial diagnostic rows, never fitted.
    reusable_holdout : tuple[JudgmentRow, ...]
        Reusable entire-class holdout rows, never fitted.
    _test_refs_by_role : mapping[str, tuple[_SealedJudgmentRef, ...]]
        Opaque sealed-row locators grouped by frozen role.
    role_hash : str
        Frozen A15 role-assignment identity binding the persistent record.
    expected_test_presentations : mapping[str, tuple[str, ...]]
        Complete screened presentation census before loader selectors for each
        guarded TEST role.
    expected_test_graphs : mapping[str, tuple[str, ...]]
        Frozen graph-hash census for each guarded TEST role.

    Notes
    -----
    Instances are intentionally not picklable. Their mapping proxies keep
    guarded source references from becoming a serialization side channel.
    """

    fit: Tuple[JudgmentRow, ...]
    validate: Tuple[JudgmentRow, ...]
    reusable_holdout: Tuple[JudgmentRow, ...]
    diagnostic: Tuple[JudgmentRow, ...]
    _test_refs_by_role: Mapping[str, Tuple[_SealedJudgmentRef, ...]]
    role_hash: str
    expected_test_presentations: Mapping[str, Tuple[str, ...]]
    expected_test_graphs: Mapping[str, Tuple[str, ...]]

    @property
    def test_count(self) -> int:
        """Return the number of guarded test rows without exposing labels.

        Returns
        -------
        int
            Guarded test-row count.
        """

        return sum(len(refs) for refs in self._test_refs_by_role.values())


def partition_holdouts(
    rows: Union[JudgmentBank, Iterable[JudgmentRow]],
) -> HoldoutPartitions:
    """Partition judgments by their frozen A15 consumption purpose.

    Parameters
    ----------
    rows : JudgmentBank or iterable[JudgmentRow]
        A bank (required to partition its opaque TEST rows) or explicit rows.

    Returns
    -------
    HoldoutPartitions
        Stable purpose partitions with test labels kept private.
    """

    if isinstance(rows, JudgmentBank):
        source_rows = rows.rows
        test_refs = rows._partition_test_refs()
        test_refs_by_role = {
            role: tuple(ref for ref in test_refs if ref.row_fields["role"] == role)
            for role in sorted(SEALED_TEST_ROLES)
        }
        role_hash = rows.role_hash
        expected_test_presentations = rows.expected_test_presentations
        expected_test_graphs = rows.expected_test_graphs
    else:
        source_rows = tuple(rows)
        if any(row.purpose is SplitPurpose.TEST for row in source_rows):
            raise ValueError("explicit TEST rows bypass bank label opacity")
        test_refs_by_role = {}
        role_hash = ""
        expected_test_presentations = {}
        expected_test_graphs = {}
    grouped = {purpose: [] for purpose in SplitPurpose}
    for row in source_rows:
        grouped[row.purpose].append(row)
    return HoldoutPartitions(
        fit=tuple(grouped[SplitPurpose.FIT]),
        validate=tuple(grouped[SplitPurpose.VALIDATE]),
        reusable_holdout=tuple(grouped[SplitPurpose.REUSABLE_HOLDOUT]),
        diagnostic=tuple(grouped[SplitPurpose.DIAGNOSTIC]),
        _test_refs_by_role=MappingProxyType(test_refs_by_role),
        role_hash=role_hash,
        expected_test_presentations=expected_test_presentations,
        expected_test_graphs=expected_test_graphs,
    )


class TestHoldoutGuard:
    """Persist content-bound A15 TEST access with exclusive creation."""

    def __init__(self) -> None:
        """Initialize a guard without accepting a caller-chosen record path."""

        self._path: Union[Path, None] = None
        self._lock_path: Union[Path, None] = None
        self._consumed_in_process = False

    @property
    def consumed(self) -> bool:
        """Report whether TEST is already inaccessible.

        Returns
        -------
        bool
            True after local consumption or when the record exists.
        """

        return self._consumed_in_process or (self._path is not None and self._path.exists())

    def _reveal_test_rows(self, refs: Tuple[_SealedJudgmentRef, ...]) -> Tuple[JudgmentRow, ...]:
        """Re-read sealed labels only after this guard reserves a ledger slot.

        Parameters
        ----------
        refs : tuple[_SealedJudgmentRef, ...]
            Opaque source references for the role whose slot was reserved.

        Returns
        -------
        tuple[JudgmentRow, ...]
            Materialized sealed rows in stable presentation order.

        Raises
        ------
        RuntimeError
            If called before this guard successfully writes its access record.
        ValueError
            If a source row moved, changed identity, or has invalid labels.
        """

        if not self._consumed_in_process or self._lock_path is None or not self._lock_path.exists():
            raise RuntimeError("sealed labels require a successful access reservation")
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

    def consume(self, partitions: HoldoutPartitions, role: str) -> Tuple[JudgmentRow, ...]:
        """Touch and return one labelled A15 sealed role exactly once.

        The access record is reserved before rows are returned. A crash after
        reservation therefore spends the test set instead of allowing a retry.

        Parameters
        ----------
        partitions : HoldoutPartitions
            Partitions whose private TEST rows will be consumed.
        role : str
            ``within-family-sealed`` or ``cross-family-sealed``.

        Returns
        -------
        tuple[JudgmentRow, ...]
            Once-only test rows.

        Raises
        ------
        TestHoldoutConsumedError
            If TEST was touched previously in this or another process.
        ValueError
            If the role is invalid or its release is empty or partial.
        """

        if self._consumed_in_process:
            raise TestHoldoutConsumedError("A15 TEST has already been touched")
        if role not in SEALED_TEST_ROLES:
            raise ValueError(f"unknown A15 sealed role: {role!r}")
        refs = partitions._test_refs_by_role.get(role, ())
        if not refs:
            raise ValueError(f"cannot consume an empty A15 TEST role: {role}")
        if not partitions.role_hash:
            raise ValueError("A15 TEST partition lacks a frozen role hash")
        actual_presentations = tuple(sorted(str(ref.row_fields["presentation_id"]) for ref in refs))
        expected_presentations = partitions.expected_test_presentations.get(role, ())
        if actual_presentations != expected_presentations:
            raise ValueError(f"cannot consume a partial A15 TEST role: {role}")
        actual_graphs = tuple(sorted({str(ref.row_fields["graph_hash"]) for ref in refs}))
        expected_graphs = partitions.expected_test_graphs.get(role, ())
        if actual_graphs != expected_graphs:
            raise ValueError(f"A15 TEST role does not cover its frozen graph census: {role}")
        path = _ACCESS_LEDGER_ROOT / f"{partitions.role_hash}.jsonl"
        lock_path = _ACCESS_LEDGER_ROOT / f"{partitions.role_hash}.{role}.1.lock"
        if self._path is not None and self._path != path:
            raise ValueError("one guard instance cannot consume different TEST banks")
        self._path = path
        self._lock_path = lock_path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        for ref in refs:
            digest.update(str(ref.row_fields["presentation_id"]).encode("utf-8"))
            digest.update(b"\0")
        presentation_digest = digest.hexdigest()
        role_label = "within" if role == "within-family-sealed" else "cross"
        payload = (
            json.dumps(
                {
                    "state": "CONSUMED",
                    "role_hash": partitions.role_hash,
                    "role": role,
                    "label": role_label,
                    "budget": 1,
                    "row_count": len(refs),
                    "presentation_digest": presentation_digest,
                },
                sort_keys=True,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            lock_descriptor = os.open(self._lock_path, flags, 0o600)
        except FileExistsError as error:
            raise TestHoldoutConsumedError(
                f"A15 TEST role has already been touched: {role}"
            ) from error
        try:
            os.fsync(lock_descriptor)
        finally:
            os.close(lock_descriptor)
        ledger_flags = os.O_CREAT | os.O_APPEND | os.O_WRONLY
        descriptor = os.open(self._path, ledger_flags, 0o600)
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._consumed_in_process = True
        return self._reveal_test_rows(refs)
