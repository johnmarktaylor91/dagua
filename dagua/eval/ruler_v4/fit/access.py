"""Content-bound LOOK-LEDGER access for non-training judgments."""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
from dataclasses import dataclass
from datetime import date, datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping, Optional, Tuple

from dagua.eval.ruler_v4.fit.bank import _FROZEN_A15_ROLE_HASH

_ACCESS_LEDGER_ROOT = Path("/home/jtaylor/.claude/research/dagua/ruler_v4/p3/gate/ACCESS_LEDGER")
_LOOK_OCCASIONS = ("post-M1", "post-M2", "post-M3", "stopping")
_CALIBRATION_KEYS = frozenset({"within-family-calibration", "cross-family-calibration"})
W08_LEDGER_KEY = "w08-off-distribution"
H_JND_LEDGER_KEY = "test-h-jnd-branch"
_PLANNED_PRESENTATIONS = 8520
_Z_ALPHA_OVER_TWO = NormalDist().inv_cdf(0.975)
_LICENSED_ANNULMENTS: frozenset[Tuple[int, str, int]] = frozenset()
_ANNULMENT_SCOPE_BASES = frozenset({"synthetic_only_manifest"})


class AccessBudgetConsumedError(RuntimeError):
    """Signal that a bounded LOOK-LEDGER key has no available release."""


@dataclass(frozen=True)
class LookReservation:
    """Publish one successfully recorded LOOK-LEDGER reservation.

    Parameters
    ----------
    ledger_key : str
        Frozen budget identity.
    slot_index : int
        One-based slot number.
    occasion : str
        Frozen occasion label.
    row_set_digest : str
        SHA-256 digest of the released row identities.
    information_fraction : float or None
        Capped information fraction for calibration schedules.
    cumulative_alpha, incremental_alpha : float or None
        Lan-DeMets O'Brien-Fleming spend for calibration schedules.
    """

    ledger_key: str
    slot_index: int
    occasion: str
    row_set_digest: str
    information_fraction: Optional[float]
    cumulative_alpha: Optional[float]
    incremental_alpha: Optional[float]


@dataclass(frozen=True)
class LedgerAnnulment:
    """Publish one valid append-only budget annulment.

    Parameters
    ----------
    ledger_key : str
        Budget identity of the annulled reservation.
    slot_index : int
        One-based slot restored by the annulment.
    annulled_line_sha256 : str
        SHA-256 of the exact persisted reservation-line bytes.
    reason : str
        One-sentence description of the specific defect.
    authority_addendum : int
        Landed preregistration addendum licensing the annulment.
    date : str
        UTC calendar date of the corrective entry.
    scope_basis : str
        Evidence class proving no judged content was released.
    """

    ledger_key: str
    slot_index: int
    annulled_line_sha256: str
    reason: str
    authority_addendum: int
    date: str
    scope_basis: str


@dataclass(frozen=True)
class _LedgerEntry:
    """Retain one parsed ledger record and its exact persisted bytes."""

    record: Mapping[str, object]
    exact_bytes: bytes
    sha256: str
    line_number: int


def _row_set_digest(row_ids: Iterable[str]) -> Tuple[str, int]:
    """Digest a nonempty set of stable row identities.

    Parameters
    ----------
    row_ids : iterable[str]
        Presentation or W-08 judgment identities.

    Returns
    -------
    tuple[str, int]
        SHA-256 digest and unique row count.

    Raises
    ------
    ValueError
        If no nonempty identity is supplied.
    """

    identities = tuple(sorted(set(str(value) for value in row_ids)))
    if not identities or any(not value for value in identities):
        raise ValueError("a ledger release requires nonempty row identities")
    digest = hashlib.sha256()
    for identity in identities:
        digest.update(identity.encode("utf-8"))
        digest.update(b"\0")
    return digest.hexdigest(), len(identities)


def _cumulative_alpha(information_fraction: float) -> float:
    """Return the frozen cumulative O'Brien-Fleming alpha spend.

    Parameters
    ----------
    information_fraction : float
        Information fraction in ``[0, 1]``.

    Returns
    -------
    float
        Cumulative two-sided alpha spend.
    """

    if information_fraction == 0.0:
        return 0.0
    return 2.0 - 2.0 * NormalDist().cdf(_Z_ALPHA_OVER_TWO / math.sqrt(information_fraction))


def _valid_annulment_reason(value: object) -> bool:
    """Return whether a value is a single, terminated sentence.

    Parameters
    ----------
    value : object
        Candidate ANNUL reason.

    Returns
    -------
    bool
        True for one nonempty line ending in sentence punctuation.
    """

    if not isinstance(value, str):
        return False
    reason = value.strip()
    if not reason or "\n" in reason or reason[-1] not in ".!?":
        return False
    sentence_body = reason[:-1]
    sentence_body = re.sub(r"(?<=\d)\.(?=\d)", "", sentence_body)
    sentence_body = re.sub(r"\b(?:[A-Za-z]\.){2,}", "", sentence_body)
    return not any(character in ".!?" for character in sentence_body)


def _valid_annulment_date(value: object) -> bool:
    """Return whether a value is an ISO 8601 calendar date.

    Parameters
    ----------
    value : object
        Candidate ANNUL date.

    Returns
    -------
    bool
        True only for a canonical ``YYYY-MM-DD`` calendar date.
    """

    if not isinstance(value, str):
        return False
    try:
        parsed = date.fromisoformat(value)
    except ValueError:
        return False
    return value == parsed.isoformat()


def _annulment_is_licensed(
    authority_addendum: object,
    ledger_key: object,
    slot_index: object,
) -> bool:
    """Return whether an addendum licenses one exact ledger correction.

    Parameters
    ----------
    authority_addendum : object
        Candidate preregistration addendum number.
    ledger_key : object
        Candidate bounded budget identity.
    slot_index : object
        Candidate one-based reservation slot.

    Returns
    -------
    bool
        True only for an exact tuple in the landed license registry.
    """

    if (
        isinstance(authority_addendum, bool)
        or not isinstance(authority_addendum, int)
        or not isinstance(ledger_key, str)
        or isinstance(slot_index, bool)
        or not isinstance(slot_index, int)
    ):
        return False
    return (authority_addendum, ledger_key, slot_index) in _LICENSED_ANNULMENTS


def _numeric_record_value(record: Mapping[str, object], key: str) -> float:
    """Read one required numeric value from a persisted ledger record.

    Parameters
    ----------
    record : mapping[str, object]
        Parsed ledger record.
    key : str
        Required numeric field name.

    Returns
    -------
    float
        Persisted numeric value.

    Raises
    ------
    ValueError
        If the persisted field is missing, boolean, or nonnumeric.
    """

    value = record.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"ledger record requires numeric {key}")
    return float(value)


class AccessLedger:
    """Append releases to the frozen campaign ledger with per-slot locks."""

    def __init__(self, ledger_root: Optional[Path] = None) -> None:
        """Use an explicit root or the injected campaign ledger root.

        Parameters
        ----------
        ledger_root : pathlib.Path or None, optional
            Explicit isolated ledger root. ``None`` selects the frozen campaign
            root for real access machinery.
        """

        self._ledger_root = _ACCESS_LEDGER_ROOT if ledger_root is None else Path(ledger_root)

    @property
    def is_campaign_root(self) -> bool:
        """Return whether this instance targets the campaign ledger tree.

        Returns
        -------
        bool
            True when the resolved root equals or is contained by the injected
            campaign root.
        """

        resolved_root = self._ledger_root.resolve()
        campaign_root = _ACCESS_LEDGER_ROOT.resolve()
        return resolved_root == campaign_root or campaign_root in resolved_root.parents

    def _path(self, role_hash: str) -> Path:
        """Resolve one role-hash ledger path.

        Parameters
        ----------
        role_hash : str
            Verified frozen A15 role hash.

        Returns
        -------
        pathlib.Path
            Append-only JSONL path.
        """

        if role_hash != _FROZEN_A15_ROLE_HASH:
            raise ValueError("ledger release requires the frozen A15 role hash")
        return self._ledger_root / f"{role_hash}.jsonl"

    def _entries(self, role_hash: str) -> Tuple[_LedgerEntry, ...]:
        """Read every ledger line while preserving its exact bytes.

        Parameters
        ----------
        role_hash : str
            Verified frozen A15 role hash.

        Returns
        -------
        tuple[_LedgerEntry, ...]
            Parsed entries in append order.
        """

        path = self._path(role_hash)
        if not path.exists():
            return ()
        entries = []
        with path.open("rb") as handle:
            for line_number, exact_bytes in enumerate(handle, start=1):
                if not exact_bytes.strip():
                    continue
                value = json.loads(exact_bytes.decode("utf-8"))
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: invalid access-ledger record")
                entries.append(
                    _LedgerEntry(
                        record=value,
                        exact_bytes=exact_bytes,
                        sha256=hashlib.sha256(exact_bytes).hexdigest(),
                        line_number=line_number,
                    )
                )
        return tuple(entries)

    @staticmethod
    def _scope_allows_annulment(target: Mapping[str, object], scope_basis: str) -> bool:
        """Return whether target evidence proves no judged-content release.

        Parameters
        ----------
        target : mapping[str, object]
            Reservation record named by an ANNUL entry.
        scope_basis : str
            Constitutional no-release evidence class.

        Returns
        -------
        bool
            True only when the record positively proves the supported
            synthetic-manifest situation without a judged-content release.
        """

        if target.get("state") == "RELEASED":
            return False
        if scope_basis == "synthetic_only_manifest":
            return target.get("synthetic_only") is True
        return False

    def _annulment_audit(
        self, role_hash: str
    ) -> Tuple[frozenset[str], Tuple[Mapping[str, object], ...], Tuple[str, ...]]:
        """Validate ANNUL lines and return their budget effects and defects.

        Parameters
        ----------
        role_hash : str
            Verified frozen A15 role hash.

        Returns
        -------
        tuple[frozenset[str], tuple[mapping[str, object], ...], tuple[str, ...]]
            Annulled exact-line digests, valid ANNUL records, and disclosed
            invalid-entry defects.
        """

        entries = self._entries(role_hash)
        earlier_by_digest: dict[str, _LedgerEntry] = {}
        annulled = set()
        valid = []
        defects = []
        for entry in entries:
            record = entry.record
            if record.get("state") != "ANNUL":
                earlier_by_digest[entry.sha256] = entry
                continue
            target_digest = record.get("annulled_line_sha256")
            target = earlier_by_digest.get(str(target_digest))
            defect: Optional[str] = None
            if record.get("role_hash") != role_hash:
                defect = "ANNUL role hash does not match its ledger"
            elif not isinstance(target_digest, str) or target is None:
                defect = "ANNUL does not identify an earlier exact ledger line"
            elif target_digest in annulled:
                defect = "ANNUL targets an already annulled reservation"
            elif record.get("ledger_key") != target.record.get("ledger_key"):
                defect = "ANNUL ledger key does not match its target"
            elif record.get("slot_index") != target.record.get("slot_index"):
                defect = "ANNUL slot index does not match its target"
            elif not _annulment_is_licensed(
                record.get("authority_addendum"),
                record.get("ledger_key"),
                record.get("slot_index"),
            ):
                defect = "ANNUL cites no landed licensing addendum"
            elif not _valid_annulment_reason(record.get("reason")):
                defect = "ANNUL reason is not one sentence"
            elif not _valid_annulment_date(record.get("date")):
                defect = "ANNUL date is not an ISO calendar date"
            elif record.get("scope_basis") not in _ANNULMENT_SCOPE_BASES:
                defect = "ANNUL scope basis is not licensed"
            elif not self._scope_allows_annulment(target.record, str(record["scope_basis"])):
                defect = "ANNUL target does not prove that judged content was unreleased"
            if defect is not None:
                defects.append(f"line {entry.line_number}: {defect}")
                continue
            annulled.add(str(target_digest))
            valid.append(record)
        return frozenset(annulled), tuple(valid), tuple(defects)

    def _records(self, role_hash: str, ledger_key: str) -> Tuple[Mapping[str, object], ...]:
        """Read active, non-annulled records for one budget key.

        Parameters
        ----------
        role_hash, ledger_key : str
            Frozen partition and budget identities.

        Returns
        -------
        tuple[mapping[str, object], ...]
            Active key-specific reservation records in append order.
        """

        annulled, _, _ = self._annulment_audit(role_hash)
        return tuple(
            entry.record
            for entry in self._entries(role_hash)
            if entry.record.get("state") != "ANNUL"
            and entry.sha256 not in annulled
            and entry.record.get("role_hash") == role_hash
            and entry.record.get("ledger_key") == ledger_key
        )

    def _append(
        self,
        role_hash: str,
        ledger_key: str,
        slot_index: int,
        payload: Mapping[str, object],
        bounded: bool,
    ) -> None:
        """Reserve a slot, append its record, and fsync both writes.

        Parameters
        ----------
        role_hash, ledger_key : str
            Frozen partition and budget identities.
        slot_index : int
            One-based release number.
        payload : mapping[str, object]
            Complete ledger record.
        bounded : bool
            Whether concurrent duplicate slot creation must fail closed.

        Raises
        ------
        AccessBudgetConsumedError
            If another process already reserved the bounded slot.
        """

        path = self._path(role_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        if bounded:
            _, valid_annulments, _ = self._annulment_audit(role_hash)
            generation = sum(
                int(
                    record.get("ledger_key") == ledger_key
                    and record.get("slot_index") == slot_index
                )
                for record in valid_annulments
            )
            lock_suffix = str(slot_index) + (f".a{generation}" if generation else "")
        else:
            lock_suffix = hashlib.sha256(
                json.dumps(payload, sort_keys=True).encode("utf-8")
            ).hexdigest()
        lock_path = path.parent / f"{role_hash}.{ledger_key}.{lock_suffix}.lock"
        flags = os.O_CREAT | os.O_EXCL | os.O_WRONLY
        try:
            lock_descriptor = os.open(lock_path, flags, 0o600)
        except FileExistsError as error:
            raise AccessBudgetConsumedError(
                f"ledger slot already consumed: {ledger_key} #{slot_index}"
            ) from error
        try:
            os.fsync(lock_descriptor)
        finally:
            os.close(lock_descriptor)
        self._append_line(path, payload)

    @staticmethod
    def _append_line(path: Path, payload: Mapping[str, object]) -> None:
        """Append and fsync one canonical JSON ledger line.

        Parameters
        ----------
        path : pathlib.Path
            Existing or new ledger JSONL path.
        payload : mapping[str, object]
            Complete append-only record.
        """

        encoded = (json.dumps(dict(payload), sort_keys=True, separators=(",", ":")) + "\n").encode(
            "utf-8"
        )
        descriptor = os.open(path, os.O_CREAT | os.O_APPEND | os.O_WRONLY, 0o600)
        try:
            os.write(descriptor, encoded)
            os.fsync(descriptor)
        finally:
            os.close(descriptor)

    def reserve_look(
        self,
        role_hash: str,
        ledger_key: str,
        occasion: str,
        row_ids: Iterable[str],
        decision: str,
        informative_judgments: Optional[int] = None,
    ) -> LookReservation:
        """Reserve the next occasion-ordered calibration or W-08 look.

        Parameters
        ----------
        role_hash : str
            Verified frozen A15 role hash.
        ledger_key : str
            One calibration role or ``w08-off-distribution``.
        occasion : str
            Next label in ``post-M1``, ``post-M2``, ``post-M3``, ``stopping``.
        row_ids : iterable[str]
            Exact judged rows released.
        decision : str
            Capacity unlock, stopping, or named per-stratum bar informed.
        informative_judgments : int or None
            Campaign-wide informative count, required for calibration and
            forbidden for W-08.

        Returns
        -------
        LookReservation
            Recorded slot and alpha-spending values.

        Raises
        ------
        AccessBudgetConsumedError
            If four looks have already been used.
        ValueError
            If the schedule, count, decision, or key is invalid.
        """

        allowed = _CALIBRATION_KEYS | {W08_LEDGER_KEY}
        if ledger_key not in allowed:
            raise ValueError(f"unknown four-look ledger key: {ledger_key!r}")
        if not decision.strip():
            raise ValueError("a look must name the decision it informed")
        digest, row_count = _row_set_digest(row_ids)
        records = self._records(role_hash, ledger_key)
        slot_index = len(records) + 1
        if slot_index > len(_LOOK_OCCASIONS):
            raise AccessBudgetConsumedError(f"four-look budget exhausted: {ledger_key}")
        expected_occasion = _LOOK_OCCASIONS[slot_index - 1]
        if occasion != expected_occasion:
            raise ValueError(f"out-of-order look for {ledger_key}: expected {expected_occasion!r}")
        information_fraction: Optional[float]
        cumulative_alpha: Optional[float]
        incremental_alpha: Optional[float]
        if ledger_key in _CALIBRATION_KEYS:
            if (
                informative_judgments is None
                or isinstance(informative_judgments, bool)
                or informative_judgments < 0
            ):
                raise ValueError("calibration looks require a nonnegative informative count")
            information_fraction = min(informative_judgments / _PLANNED_PRESENTATIONS, 1.0)
            previous_fraction = (
                0.0 if not records else _numeric_record_value(records[-1], "information_fraction")
            )
            if information_fraction < previous_fraction:
                raise ValueError("calibration information fraction cannot decrease")
            cumulative_alpha = _cumulative_alpha(information_fraction)
            previous_alpha = (
                0.0 if not records else _numeric_record_value(records[-1], "cumulative_alpha")
            )
            incremental_alpha = cumulative_alpha - previous_alpha
        else:
            if informative_judgments is not None:
                raise ValueError("W-08 looks carry no alpha arithmetic")
            information_fraction = None
            cumulative_alpha = None
            incremental_alpha = None
        timestamp = datetime.now(timezone.utc).isoformat()
        payload = {
            "state": "RELEASED",
            "role_hash": role_hash,
            "ledger_key": ledger_key,
            "budget": 4,
            "slot_index": slot_index,
            "occasion": occasion,
            "information_fraction": information_fraction,
            "cumulative_alpha": cumulative_alpha,
            "incremental_alpha": incremental_alpha,
            "row_set_digest": digest,
            "row_count": row_count,
            "decision": decision,
            "date": timestamp,
        }
        self._append(role_hash, ledger_key, slot_index, payload, bounded=True)
        return LookReservation(
            ledger_key=ledger_key,
            slot_index=slot_index,
            occasion=occasion,
            row_set_digest=digest,
            information_fraction=information_fraction,
            cumulative_alpha=cumulative_alpha,
            incremental_alpha=incremental_alpha,
        )

    def record_unbudgeted(self, role_hash: str, purpose: str, row_ids: Iterable[str]) -> str:
        """Append an uncapped reusable-holdout or diagnostic label read.

        Parameters
        ----------
        role_hash : str
            Verified frozen A15 role hash.
        purpose : str
            ``reusable_holdout`` or ``diagnostic``.
        row_ids : iterable[str]
            Exact labelled rows released.

        Returns
        -------
        str
            Row-set digest written to the ledger.

        Raises
        ------
        ValueError
            If the purpose is budgeted or the row set is empty.
        """

        if purpose not in {"reusable_holdout", "diagnostic"}:
            raise ValueError("only reusable holdout and diagnostic reads are uncapped")
        digest, row_count = _row_set_digest(row_ids)
        existing = self._records(role_hash, purpose)
        slot_index = len(existing) + 1
        payload = {
            "state": "RELEASED",
            "role_hash": role_hash,
            "ledger_key": purpose,
            "budget": None,
            "slot_index": slot_index,
            "row_set_digest": digest,
            "row_count": row_count,
            "purpose": purpose,
            "date": datetime.now(timezone.utc).isoformat(),
        }
        self._append(role_hash, purpose, slot_index, payload, bounded=False)
        return digest

    def reserve_once(
        self,
        role_hash: str,
        ledger_key: str,
        row_ids: Iterable[str],
        purpose: str,
        synthetic_only: bool = False,
    ) -> str:
        """Reserve one content-bound branch or sealed evaluation event.

        Parameters
        ----------
        role_hash, ledger_key : str
            Frozen partition and once-only budget identities.
        row_ids : iterable[str]
            Complete released row census.
        purpose : str
            Named use of the once-only event.
        synthetic_only : bool, default=False
            Whether the spending run's manifest proves it is synthetic-only.

        Returns
        -------
        str
            Row-set digest written to the ledger.

        Raises
        ------
        AccessBudgetConsumedError
            If the once-only event was already reserved.
        """

        if self._records(role_hash, ledger_key):
            raise AccessBudgetConsumedError(f"once-only budget exhausted: {ledger_key}")
        digest, row_count = _row_set_digest(row_ids)
        payload = {
            "state": "CONSUMED",
            "role_hash": role_hash,
            "ledger_key": ledger_key,
            "budget": 1,
            "slot_index": 1,
            "row_set_digest": digest,
            "row_count": row_count,
            "purpose": purpose,
            "synthetic_only": synthetic_only,
            "date": datetime.now(timezone.utc).isoformat(),
        }
        self._append(role_hash, ledger_key, 1, payload, bounded=True)
        return digest

    def annul_reservation(
        self,
        role_hash: str,
        ledger_key: str,
        slot_index: int,
        reason: str,
        authority_addendum: int,
        scope_basis: str,
    ) -> LedgerAnnulment:
        """Append a licensed ANNUL line for a no-release reservation defect.

        Parameters
        ----------
        role_hash, ledger_key : str
            Frozen partition and budget identities.
        slot_index : int
            One-based reservation slot to restore.
        reason : str
            One sentence naming the specific defect.
        authority_addendum : int
            Landed addendum licensing this specific annulment.
        scope_basis : str
            Supported RIDER-2 no-release evidence class.

        Returns
        -------
        LedgerAnnulment
            Validated corrective entry appended to the ledger.

        Raises
        ------
        ValueError
            If authority, reason, target identity, or constitutional scope is invalid.
        """

        if not _annulment_is_licensed(authority_addendum, ledger_key, slot_index):
            raise ValueError("ANNUL requires a landed licensing addendum")
        if not _valid_annulment_reason(reason):
            raise ValueError("ANNUL reason must be one sentence")
        if scope_basis not in _ANNULMENT_SCOPE_BASES:
            raise ValueError("ANNUL scope basis is not licensed")
        annulled, _, _ = self._annulment_audit(role_hash)
        targets = tuple(
            entry
            for entry in self._entries(role_hash)
            if entry.record.get("state") != "ANNUL"
            and entry.sha256 not in annulled
            and entry.record.get("role_hash") == role_hash
            and entry.record.get("ledger_key") == ledger_key
            and entry.record.get("slot_index") == slot_index
        )
        if len(targets) != 1:
            raise ValueError("ANNUL requires exactly one active target reservation")
        target = targets[0]
        if not self._scope_allows_annulment(target.record, scope_basis):
            raise ValueError("ANNUL target does not prove that judged content was unreleased")
        entry = LedgerAnnulment(
            ledger_key=ledger_key,
            slot_index=slot_index,
            annulled_line_sha256=target.sha256,
            reason=reason.strip(),
            authority_addendum=authority_addendum,
            date=datetime.now(timezone.utc).date().isoformat(),
            scope_basis=scope_basis,
        )
        payload = {
            "state": "ANNUL",
            "role_hash": role_hash,
            "ledger_key": entry.ledger_key,
            "slot_index": entry.slot_index,
            "annulled_line_sha256": entry.annulled_line_sha256,
            "reason": entry.reason,
            "authority_addendum": entry.authority_addendum,
            "date": entry.date,
            "scope_basis": entry.scope_basis,
        }
        path = self._path(role_hash)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._append_line(path, payload)
        return entry

    def annulment_lines(self, role_hash: str) -> Tuple[str, ...]:
        """Return every ANNUL line verbatim for freeze-report publication.

        Parameters
        ----------
        role_hash : str
            Frozen A15 role identity.

        Returns
        -------
        tuple[str, ...]
            Exact UTF-8 ANNUL lines, including their persisted newline.
        """

        return tuple(
            entry.exact_bytes.decode("utf-8")
            for entry in self._entries(role_hash)
            if entry.record.get("state") == "ANNUL"
        )

    def ledger_defects(self, role_hash: str) -> Tuple[str, ...]:
        """Return disclosed defects for void ANNUL lines.

        Parameters
        ----------
        role_hash : str
            Frozen A15 role identity.

        Returns
        -------
        tuple[str, ...]
            Validation defects in append order.
        """

        _, _, defects = self._annulment_audit(role_hash)
        return defects

    def budget_usage(self, role_hash: str) -> Mapping[str, int]:
        """Return persistent bounded-budget consumption without reading labels.

        Parameters
        ----------
        role_hash : str
            Frozen A15 role identity.

        Returns
        -------
        mapping[str, int]
            Consumed slots for every four-look and once-only budget.
        """

        keys = tuple(sorted(_CALIBRATION_KEYS | {W08_LEDGER_KEY, H_JND_LEDGER_KEY})) + tuple(
            sorted(("within-family-sealed", "cross-family-sealed"))
        )
        return {key: len(self._records(role_hash, key)) for key in keys}
