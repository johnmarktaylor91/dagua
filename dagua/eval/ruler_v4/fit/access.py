"""Content-bound LOOK-LEDGER access for non-training judgments."""

from __future__ import annotations

import hashlib
import json
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from statistics import NormalDist
from typing import Iterable, Mapping, Optional, Tuple

_ACCESS_LEDGER_ROOT = Path("/home/jtaylor/.claude/research/dagua/ruler_v4/p3/gate/ACCESS_LEDGER")
_LOOK_OCCASIONS = ("post-M1", "post-M2", "post-M3", "stopping")
_CALIBRATION_KEYS = frozenset({"within-family-calibration", "cross-family-calibration"})
W08_LEDGER_KEY = "w08-off-distribution"
H_JND_LEDGER_KEY = "test-h-jnd-branch"
_PLANNED_PRESENTATIONS = 8520
_Z_ALPHA_OVER_TWO = NormalDist().inv_cdf(0.975)


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


class AccessLedger:
    """Append releases to the frozen campaign ledger with per-slot locks."""

    def __init__(self) -> None:
        """Use the injected absolute campaign ledger root."""

        self._ledger_root = _ACCESS_LEDGER_ROOT

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

        if not role_hash:
            raise ValueError("ledger release requires a frozen role hash")
        return self._ledger_root / f"{role_hash}.jsonl"

    def _records(self, role_hash: str, ledger_key: str) -> Tuple[Mapping[str, object], ...]:
        """Read existing records for one budget key.

        Parameters
        ----------
        role_hash, ledger_key : str
            Frozen partition and budget identities.

        Returns
        -------
        tuple[mapping[str, object], ...]
            Existing key-specific records in append order.
        """

        path = self._path(role_hash)
        if not path.exists():
            return ()
        records = []
        with path.open("r", encoding="utf-8") as handle:
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"{path}:{line_number}: invalid access-ledger record")
                if value.get("role_hash") == role_hash and value.get("ledger_key") == ledger_key:
                    records.append(value)
        return tuple(records)

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
        lock_suffix = (
            str(slot_index)
            if bounded
            else hashlib.sha256(json.dumps(payload, sort_keys=True).encode("utf-8")).hexdigest()
        )
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
            previous_fraction = 0.0 if not records else float(records[-1]["information_fraction"])
            if information_fraction < previous_fraction:
                raise ValueError("calibration information fraction cannot decrease")
            cumulative_alpha = _cumulative_alpha(information_fraction)
            previous_alpha = 0.0 if not records else float(records[-1]["cumulative_alpha"])
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
            "date": datetime.now(timezone.utc).isoformat(),
        }
        self._append(role_hash, ledger_key, 1, payload, bounded=True)
        return digest
