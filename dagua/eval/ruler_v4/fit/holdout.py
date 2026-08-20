"""Fail-closed A15 fit/validation/test access discipline."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Union

from dagua.eval.ruler_v4.fit.bank import (
    JudgmentBank,
    JudgmentRow,
    SplitPurpose,
    _reveal_test_rows,
    _SealedJudgmentRef,
)

_TEST_HOLDOUT_STATE_ROOT = Path.home() / ".local/state/dagua/ruler_v4/test-holdout"


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
    _test_refs : tuple[_SealedJudgmentRef, ...]
        Opaque sealed-row locators without verdicts or tie labels.
    bank_identity : str
        Content identity binding this partition to one persistent record.
    """

    fit: Tuple[JudgmentRow, ...]
    validate: Tuple[JudgmentRow, ...]
    diagnostic: Tuple[JudgmentRow, ...]
    _test_refs: Tuple[_SealedJudgmentRef, ...]
    bank_identity: str

    @property
    def test_count(self) -> int:
        """Return the number of guarded test rows without exposing labels.

        Returns
        -------
        int
            Guarded test-row count.
        """

        return len(self._test_refs)


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
        bank_identity = rows.bank_identity
    else:
        source_rows = tuple(rows)
        if any(row.purpose is SplitPurpose.TEST for row in source_rows):
            raise ValueError("explicit TEST rows bypass bank label opacity")
        test_refs = ()
        bank_identity = ""
    grouped = {purpose: [] for purpose in SplitPurpose}
    for row in source_rows:
        grouped[row.purpose].append(row)
    return HoldoutPartitions(
        fit=tuple(grouped[SplitPurpose.FIT]),
        validate=tuple(grouped[SplitPurpose.VALIDATE]),
        diagnostic=tuple(grouped[SplitPurpose.DIAGNOSTIC]),
        _test_refs=test_refs,
        bank_identity=bank_identity,
    )


class TestHoldoutGuard:
    """Persist content-bound A15 TEST access with exclusive creation."""

    def __init__(self) -> None:
        """Initialize a guard without accepting a caller-chosen record path."""

        self._path: Union[Path, None] = None
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

    def consume(self, partitions: HoldoutPartitions) -> Tuple[JudgmentRow, ...]:
        """Touch and return A15 TEST labels exactly once.

        The access record is reserved before rows are returned. A crash after
        reservation therefore spends the test set instead of allowing a retry.

        Parameters
        ----------
        partitions : HoldoutPartitions
            Partitions whose private TEST rows will be consumed.

        Returns
        -------
        tuple[JudgmentRow, ...]
            Once-only test rows.

        Raises
        ------
        TestHoldoutConsumedError
            If TEST was touched previously in this or another process.
        """

        if self._consumed_in_process:
            raise TestHoldoutConsumedError("A15 TEST has already been touched")
        if not partitions._test_refs:
            raise ValueError("cannot consume an empty A15 TEST partition")
        if not partitions.bank_identity:
            raise ValueError("A15 TEST partition lacks a bank content identity")
        path = _TEST_HOLDOUT_STATE_ROOT / f"{partitions.bank_identity}.json"
        if self._path is not None and self._path != path:
            raise ValueError("one guard instance cannot consume different TEST banks")
        self._path = path
        self._path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        for ref in partitions._test_refs:
            digest.update(str(ref.row_fields["presentation_id"]).encode("utf-8"))
            digest.update(b"\0")
        presentation_digest = digest.hexdigest()
        payload = json.dumps(
            {
                "state": "CONSUMED",
                "bank_identity": partitions.bank_identity,
                "row_count": len(partitions._test_refs),
                "presentation_digest": presentation_digest,
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(self._path, flags, 0o600)
        except FileExistsError as error:
            existing = json.loads(self._path.read_text(encoding="utf-8"))
            if (
                existing.get("bank_identity") != partitions.bank_identity
                or existing.get("presentation_digest") != presentation_digest
            ):
                raise ValueError("A15 TEST access record disagrees with the partition") from error
            raise TestHoldoutConsumedError("A15 TEST has already been touched") from error
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._consumed_in_process = True
        return _reveal_test_rows(partitions._test_refs)
