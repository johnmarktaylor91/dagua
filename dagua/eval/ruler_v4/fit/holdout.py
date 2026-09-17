"""Fail-closed A15 fit/validation/test access discipline."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Iterable, Mapping, Tuple, Union

from dagua.eval.ruler_v4.fit.access import (
    AccessBudgetConsumedError,
    AccessLedger,
    LookReservation,
)
from dagua.eval.ruler_v4.fit.bank import (
    SEALED_TEST_ROLES,
    JudgmentBank,
    JudgmentMetadata,
    JudgmentRow,
    SplitPurpose,
    _metadata_from_fields,
    _read_jsonl_with_line_numbers,
    _SealedJudgmentRef,
)


class TestHoldoutConsumedError(RuntimeError):
    """Signal that the once-only A15 test bank has already been touched."""


class CalibrationLookConsumedError(RuntimeError):
    """Signal that a calibration schedule has exhausted its four looks."""


@dataclass(frozen=True)
class HoldoutPartitions:
    """Expose reusable roles while keeping test rows behind a guard.

    Parameters
    ----------
    fit : tuple[JudgmentRow, ...]
        A15 train-role rows.
    validate : tuple[JudgmentMetadata, ...]
        Whitelist-only within- and cross-family calibration metadata.
    diagnostic : tuple[JudgmentMetadata, ...]
        Whitelist-only adversarial diagnostic metadata.
    reusable_holdout : tuple[JudgmentMetadata, ...]
        Whitelist-only entire-class holdout metadata.
    _gated_refs_by_purpose : mapping[SplitPurpose, tuple[_SealedJudgmentRef, ...]]
        Opaque non-TEST locators grouped by frozen purpose.
    _test_refs_by_role : mapping[str, tuple[_SealedJudgmentRef, ...]]
        Opaque sealed-row locators grouped by frozen role.
    role_hash : str
        Frozen A15 role-assignment identity binding the persistent record.
    expected_test_base_pairs : mapping[str, tuple[str, ...]]
        Frozen A16 base-pair census for each guarded TEST role.
    expected_test_graphs : mapping[str, tuple[str, ...]]
        Frozen graph-hash census for each guarded TEST role.

    Notes
    -----
    Instances are intentionally not picklable. Their mapping proxies keep
    guarded source references from becoming a serialization side channel.
    """

    fit: Tuple[JudgmentRow, ...]
    validate: Tuple[JudgmentMetadata, ...]
    reusable_holdout: Tuple[JudgmentMetadata, ...]
    diagnostic: Tuple[JudgmentMetadata, ...]
    _gated_refs_by_purpose: Mapping[SplitPurpose, Tuple[_SealedJudgmentRef, ...]]
    _test_refs_by_role: Mapping[str, Tuple[_SealedJudgmentRef, ...]]
    role_hash: str
    expected_test_base_pairs: Mapping[str, Tuple[str, ...]]
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
        metadata = rows.metadata()
        test_refs = rows._partition_test_refs()
        test_refs_by_role = {
            role: tuple(ref for ref in test_refs if ref.row_fields["role"] == role)
            for role in sorted(SEALED_TEST_ROLES)
        }
        role_hash = rows.role_hash
        expected_test_base_pairs = rows.expected_test_base_pairs
        expected_test_graphs = rows.expected_test_graphs
        gated_refs_by_purpose = {
            purpose: rows._partition_gated_refs(purpose)
            for purpose in (
                SplitPurpose.VALIDATE,
                SplitPurpose.REUSABLE_HOLDOUT,
                SplitPurpose.DIAGNOSTIC,
            )
        }
    else:
        source_rows = tuple(rows)
        if any(row.purpose is not SplitPurpose.FIT for row in source_rows):
            raise ValueError("explicit non-FIT rows bypass LOOK-LEDGER label opacity")
        metadata = tuple(_metadata_from_fields(vars(row)) for row in source_rows)
        test_refs_by_role = {}
        gated_refs_by_purpose = {}
        role_hash = ""
        expected_test_base_pairs = {}
        expected_test_graphs = {}
    grouped = {purpose: [] for purpose in SplitPurpose}
    for row in source_rows:
        grouped[row.purpose].append(row)
    metadata_grouped = {purpose: [] for purpose in SplitPurpose}
    for row in metadata:
        metadata_grouped[row.purpose].append(row)
    return HoldoutPartitions(
        fit=tuple(grouped[SplitPurpose.FIT]),
        validate=tuple(metadata_grouped[SplitPurpose.VALIDATE]),
        reusable_holdout=tuple(metadata_grouped[SplitPurpose.REUSABLE_HOLDOUT]),
        diagnostic=tuple(metadata_grouped[SplitPurpose.DIAGNOSTIC]),
        _gated_refs_by_purpose=MappingProxyType(gated_refs_by_purpose),
        _test_refs_by_role=MappingProxyType(test_refs_by_role),
        role_hash=role_hash,
        expected_test_base_pairs=expected_test_base_pairs,
        expected_test_graphs=expected_test_graphs,
    )


class TestHoldoutGuard:
    """Persist content-bound A15 TEST access with exclusive creation."""

    def __init__(self) -> None:
        """Initialize a guard with the single frozen campaign ledger root."""

        self._ledger = AccessLedger()
        self._ledger_root = self._ledger._ledger_root
        self._path: Union[object, None] = None
        self._reserved_role: Union[str, None] = None
        self._consumed_in_process = False

    @property
    def consumed(self) -> bool:
        """Report whether TEST is already inaccessible.

        Returns
        -------
        bool
            True after local consumption or when the record exists.
        """

        return self._consumed_in_process

    def _reveal_test_rows(
        self, refs: Tuple[_SealedJudgmentRef, ...], role: str
    ) -> Tuple[JudgmentRow, ...]:
        """Re-read sealed labels only after this guard reserves a ledger slot.

        Parameters
        ----------
        refs : tuple[_SealedJudgmentRef, ...]
            Opaque source references for the role whose slot was reserved.
        role : str
            Frozen sealed role reserved by this guard.

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

        if (
            not self._consumed_in_process
            or self._reserved_role != role
            or any(ref.row_fields["role"] != role for ref in refs)
        ):
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
                if raw.get("malformed") is True or raw.get("abstain") is True:
                    continue
                verdict = int(raw.get("verdict", 0))
                tie = bool(raw.get("tie", verdict == 0))
                if verdict == 0 and not tie:
                    continue
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

    def _validate_role_census(
        self,
        partitions: HoldoutPartitions,
        refs: Tuple[_SealedJudgmentRef, ...],
        role: str,
        require_complete: bool,
    ) -> None:
        """Validate one guarded release against its frozen role census.

        Parameters
        ----------
        partitions : HoldoutPartitions
            Partitions carrying the artifact-derived census.
        refs : tuple[_SealedJudgmentRef, ...]
            Actual opaque rows proposed for release.
        role : str
            Frozen role being released.
        require_complete : bool
            Whether the release must cover the whole frozen role. This is true
            only for once-only sealed budgets.

        Raises
        ------
        ValueError
            If the role hash is absent, the release leaves the frozen census,
            or a required whole-role release is incomplete.
        """

        if not partitions.role_hash:
            raise ValueError(f"{role} partition lacks a frozen role hash")
        actual_base_pairs = {str(ref.row_fields["base_pair_id"]) for ref in refs}
        expected_base_pairs = set(partitions.expected_test_base_pairs.get(role, ()))
        extras = actual_base_pairs - expected_base_pairs
        if extras:
            raise ValueError(
                f"guarded row set is not a subset of its frozen role census: {role}; "
                f"extras={len(extras)}"
            )
        actual_graphs = {str(ref.row_fields["graph_hash"]) for ref in refs}
        expected_graphs = set(partitions.expected_test_graphs.get(role, ()))
        graph_extras = actual_graphs - expected_graphs
        if graph_extras:
            raise ValueError(
                f"guarded graph set is not a subset of its frozen role census: {role}; "
                f"extras={len(graph_extras)}"
            )
        if not require_complete:
            return
        missing = expected_base_pairs - actual_base_pairs
        if missing:
            raise ValueError(
                f"cannot consume a partial guarded role: {role}; missing={len(missing)}"
            )
        if actual_graphs != expected_graphs:
            raise ValueError(f"guarded role does not cover its frozen graph census: {role}")

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
        self._validate_role_census(partitions, refs, role, require_complete=True)
        role_label = "within" if role == "within-family-sealed" else "cross"
        try:
            self._ledger.reserve_once(
                partitions.role_hash,
                role,
                (str(ref.row_fields["presentation_id"]) for ref in refs),
                purpose=f"sealed-test-{role_label}",
            )
        except AccessBudgetConsumedError as error:
            raise TestHoldoutConsumedError(
                f"A15 TEST role has already been touched: {role}"
            ) from error
        self._reserved_role = role
        self._consumed_in_process = True
        return self._reveal_test_rows(refs, role)


class CalibrationLookGuard(TestHoldoutGuard):
    """Release calibration labels only after reserving the next look."""

    def __init__(self) -> None:
        """Initialize against the injected shared campaign ledger."""

        super().__init__()

    def consume(
        self,
        partitions: HoldoutPartitions,
        role: str,
        occasion: str,
        informative_judgments: int,
        decision: str,
    ) -> Tuple[Tuple[JudgmentRow, ...], LookReservation]:
        """Record and release one complete calibration-role look.

        Parameters
        ----------
        partitions : HoldoutPartitions
            Opaque bank partitions bound to the frozen role hash.
        role : str
            Within- or cross-family calibration role.
        occasion : str
            Next frozen occasion label.
        informative_judgments : int
            Campaign informative count divided by the frozen 8,520 denominator.
        decision : str
            Capacity unlock, stopping, or named per-stratum bar informed.

        Returns
        -------
        tuple[tuple[JudgmentRow, ...], LookReservation]
            Labelled rows and their alpha-spent reservation.

        Raises
        ------
        CalibrationLookConsumedError
            If the schedule has exhausted its four slots.
        ValueError
            If the role, census, or occasion is invalid.
        """

        if role not in {"within-family-calibration", "cross-family-calibration"}:
            raise ValueError(f"unknown calibration role: {role!r}")
        refs = tuple(
            ref
            for ref in partitions._gated_refs_by_purpose.get(SplitPurpose.VALIDATE, ())
            if ref.row_fields["role"] == role
        )
        if not refs:
            raise ValueError(f"cannot consume an empty calibration role: {role}")
        self._validate_role_census(partitions, refs, role, require_complete=False)
        try:
            reservation = self._ledger.reserve_look(
                partitions.role_hash,
                role,
                occasion,
                (str(ref.row_fields["presentation_id"]) for ref in refs),
                decision,
                informative_judgments,
            )
        except AccessBudgetConsumedError as error:
            raise CalibrationLookConsumedError(str(error)) from error
        self._reserved_role = role
        self._consumed_in_process = True
        rows = self._reveal_test_rows(refs, role)
        return rows, reservation


class ReusableJudgmentGuard(TestHoldoutGuard):
    """Ledger uncapped entire-class and adversarial judged-label reads."""

    def __init__(self) -> None:
        """Initialize against the injected shared campaign ledger."""

        super().__init__()

    def consume(
        self, partitions: HoldoutPartitions, purpose: SplitPurpose
    ) -> Tuple[JudgmentRow, ...]:
        """Record and release one uncapped non-fitting judged row set.

        Parameters
        ----------
        partitions : HoldoutPartitions
            Opaque bank partitions bound to the frozen role hash.
        purpose : SplitPurpose
            ``REUSABLE_HOLDOUT`` or ``DIAGNOSTIC``.

        Returns
        -------
        tuple[JudgmentRow, ...]
            Labelled rows after the read was ledgered.

        Raises
        ------
        ValueError
            If the purpose, role hash, or row set is invalid.
        """

        if purpose not in {SplitPurpose.REUSABLE_HOLDOUT, SplitPurpose.DIAGNOSTIC}:
            raise ValueError("reusable guard accepts holdout or diagnostic purpose only")
        refs = partitions._gated_refs_by_purpose.get(purpose, ())
        if not refs or not partitions.role_hash:
            raise ValueError("reusable labelled read requires a nonempty frozen partition")
        roles = {str(ref.row_fields["role"]) for ref in refs}
        if len(roles) != 1:
            raise ValueError("one reusable labelled read may release exactly one frozen role")
        role = next(iter(roles))
        self._validate_role_census(partitions, refs, role, require_complete=False)
        self._ledger.record_unbudgeted(
            partitions.role_hash,
            purpose.value,
            (str(ref.row_fields["presentation_id"]) for ref in refs),
        )
        self._reserved_role = role
        self._consumed_in_process = True
        return self._reveal_test_rows(refs, role)
