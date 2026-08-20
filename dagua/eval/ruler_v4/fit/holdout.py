"""Fail-closed A15 fit/validation/test access discipline."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Tuple, Union

from dagua.eval.ruler_v4.fit.bank import JudgmentBank, JudgmentRow, SplitPurpose

PathLike = Union[str, Path]


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
    _test : tuple[JudgmentRow, ...]
        Sealed and entire-class-holdout rows; intentionally private.
    """

    fit: Tuple[JudgmentRow, ...]
    validate: Tuple[JudgmentRow, ...]
    diagnostic: Tuple[JudgmentRow, ...]
    _test: Tuple[JudgmentRow, ...]

    @property
    def test_count(self) -> int:
        """Return the number of guarded test rows without exposing labels.

        Returns
        -------
        int
            Guarded test-row count.
        """

        return len(self._test)


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

    source_rows = rows._partition_rows() if isinstance(rows, JudgmentBank) else rows
    grouped = {purpose: [] for purpose in SplitPurpose}
    for row in source_rows:
        grouped[row.purpose].append(row)
    return HoldoutPartitions(
        fit=tuple(grouped[SplitPurpose.FIT]),
        validate=tuple(grouped[SplitPurpose.VALIDATE]),
        diagnostic=tuple(grouped[SplitPurpose.DIAGNOSTIC]),
        _test=tuple(grouped[SplitPurpose.TEST]),
    )


class TestHoldoutGuard:
    """Persist the once-only A15 TEST access state with exclusive creation.

    Parameters
    ----------
    access_record_path : path-like
        New path reserved atomically on first access. An existing path means
        TEST is already consumed, including after a process restart or crash.
    """

    def __init__(self, access_record_path: PathLike) -> None:
        """Initialize a fail-closed persistent guard.

        Parameters
        ----------
        access_record_path : path-like
            Persistent access-record path.

        Raises
        ------
        ValueError
            If the path is empty or names a directory.
        """

        path = Path(access_record_path)
        if not str(path) or path.is_dir():
            raise ValueError("test access record must be a file path")
        self._path = path
        self._consumed_in_process = False

    @property
    def consumed(self) -> bool:
        """Report whether TEST is already inaccessible.

        Returns
        -------
        bool
            True after local consumption or when the record exists.
        """

        return self._consumed_in_process or self._path.exists()

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
        self._path.parent.mkdir(parents=True, exist_ok=True)
        digest = hashlib.sha256()
        for row in partitions._test:
            digest.update(row.presentation_id.encode("utf-8"))
            digest.update(b"\0")
        payload = json.dumps(
            {
                "state": "CONSUMED",
                "row_count": len(partitions._test),
                "presentation_digest": digest.hexdigest(),
            },
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            descriptor = os.open(self._path, flags, 0o600)
        except FileExistsError as error:
            raise TestHoldoutConsumedError("A15 TEST has already been touched") from error
        try:
            os.write(descriptor, payload)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)
        self._consumed_in_process = True
        return partitions._test
