"""JSON IO helpers for visual parity v2 stores."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Iterable, Mapping

from scripts.visual_parity.types import (
    CARD_MANIFEST_SCHEMA_VERSION,
    COVERAGE_MATRIX_SCHEMA_VERSION,
    LEDGER_SCHEMA_VERSION,
)

STORE_SCHEMAS: Dict[str, int] = {
    "coverage_matrix": COVERAGE_MATRIX_SCHEMA_VERSION,
    "ledger": LEDGER_SCHEMA_VERSION,
    "card_manifest": CARD_MANIFEST_SCHEMA_VERSION,
}

STORE_COLLECTIONS: Dict[str, Iterable[str]] = {
    "coverage_matrix": ("source_snapshots", "adapter_capabilities", "cells"),
    "ledger": ("rows", "knobs", "rounds", "residuals"),
    "card_manifest": ("cards",),
}


class SchemaVersionError(ValueError):
    """Raised when a visual parity JSON store has an unknown schema version."""


def _read_json(path: Path) -> Dict[str, Any]:
    """Read a JSON object from disk.

    Parameters
    ----------
    path
        JSON file path.

    Returns
    -------
    Dict[str, Any]
        Parsed JSON object.
    """

    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    if not isinstance(data, dict):
        raise TypeError(f"{path} must contain a JSON object")
    return data


def _canonical_json(data: Mapping[str, Any]) -> str:
    """Return canonical JSON text used by all visual parity stores.

    Parameters
    ----------
    data
        JSON-compatible mapping.

    Returns
    -------
    str
        Stable UTF-8 JSON text with a trailing newline.
    """

    return json.dumps(data, indent=2, sort_keys=True) + "\n"


def _validate_store(data: Mapping[str, Any], store: str) -> None:
    """Validate a visual parity store envelope.

    Parameters
    ----------
    data
        Parsed store object.
    store
        Store name: ``coverage_matrix``, ``ledger``, or ``card_manifest``.

    Returns
    -------
    None
        The function raises on invalid input.
    """

    expected = STORE_SCHEMAS[store]
    actual = data.get("schema_version")
    if actual != expected:
        raise SchemaVersionError(f"{store} schema_version must be {expected}; got {actual!r}")
    for key in STORE_COLLECTIONS[store]:
        if key not in data:
            raise KeyError(f"{store} missing required collection {key!r}")
        if not isinstance(data[key], list):
            raise TypeError(f"{store}.{key} must be a list")


def read_store(path: str | Path, store: str) -> Dict[str, Any]:
    """Read and validate a visual parity JSON store.

    Parameters
    ----------
    path
        Store JSON path.
    store
        Store name: ``coverage_matrix``, ``ledger``, or ``card_manifest``.

    Returns
    -------
    Dict[str, Any]
        Parsed and schema-version-checked store data.
    """

    if store not in STORE_SCHEMAS:
        raise KeyError(f"unknown visual parity store {store!r}")
    data = _read_json(Path(path))
    _validate_store(data, store)
    return data


def write_store(path: str | Path, store: str, data: Mapping[str, Any]) -> None:
    """Validate and write a visual parity JSON store in canonical form.

    Parameters
    ----------
    path
        Destination JSON path.
    store
        Store name: ``coverage_matrix``, ``ledger``, or ``card_manifest``.
    data
        JSON-compatible store data.

    Returns
    -------
    None
        The function writes the file or raises on invalid input.
    """

    if store not in STORE_SCHEMAS:
        raise KeyError(f"unknown visual parity store {store!r}")
    _validate_store(data, store)
    Path(path).write_text(_canonical_json(data), encoding="utf-8")


def read_coverage_matrix(path: str | Path) -> Dict[str, Any]:
    """Read a schema-version-checked coverage matrix.

    Parameters
    ----------
    path
        Coverage matrix JSON path.

    Returns
    -------
    Dict[str, Any]
        Parsed coverage matrix data.
    """

    return read_store(path, "coverage_matrix")


def write_coverage_matrix(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write a schema-version-checked coverage matrix.

    Parameters
    ----------
    path
        Coverage matrix JSON path.
    data
        JSON-compatible coverage matrix data.

    Returns
    -------
    None
        The function writes the file or raises on invalid input.
    """

    write_store(path, "coverage_matrix", data)


def read_ledger(path: str | Path) -> Dict[str, Any]:
    """Read a schema-version-checked ledger.

    Parameters
    ----------
    path
        Ledger JSON path.

    Returns
    -------
    Dict[str, Any]
        Parsed ledger data.
    """

    return read_store(path, "ledger")


def write_ledger(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write a schema-version-checked ledger.

    Parameters
    ----------
    path
        Ledger JSON path.
    data
        JSON-compatible ledger data.

    Returns
    -------
    None
        The function writes the file or raises on invalid input.
    """

    write_store(path, "ledger", data)


def read_card_manifest(path: str | Path) -> Dict[str, Any]:
    """Read a schema-version-checked card manifest.

    Parameters
    ----------
    path
        Card manifest JSON path.

    Returns
    -------
    Dict[str, Any]
        Parsed card manifest data.
    """

    return read_store(path, "card_manifest")


def write_card_manifest(path: str | Path, data: Mapping[str, Any]) -> None:
    """Write a schema-version-checked card manifest.

    Parameters
    ----------
    path
        Card manifest JSON path.
    data
        JSON-compatible card manifest data.

    Returns
    -------
    None
        The function writes the file or raises on invalid input.
    """

    write_store(path, "card_manifest", data)


def _self_test() -> None:
    """Run a small in-memory validation test for all supported stores.

    Returns
    -------
    None
        The function raises if a store cannot be validated.
    """

    read_write_pairs = {
        "coverage_matrix": {
            "schema_version": COVERAGE_MATRIX_SCHEMA_VERSION,
            "generated_at": "ISO-8601",
            "reference_pins": {},
            "source_snapshots": [],
            "adapter_capabilities": [],
            "cells": [],
        },
        "ledger": {
            "schema_version": LEDGER_SCHEMA_VERSION,
            "run_id": "visual_parity_v2_2026_07",
            "created_at": "ISO-8601",
            "updated_at": "ISO-8601",
            "environment": {},
            "ratchets": {"global_in_tol_floor_pct": 85.0},
            "auditor": {},
            "rows": [],
            "knobs": [],
            "rounds": [],
            "residuals": [],
        },
        "card_manifest": {
            "schema_version": CARD_MANIFEST_SCHEMA_VERSION,
            "generated_at": "ISO-8601",
            "cards": [],
        },
    }
    for store, data in read_write_pairs.items():
        _validate_store(data, store)


def main() -> int:
    """Run the visual parity IO command-line interface.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true", help="validate built-in store stubs")
    args = parser.parse_args()
    if args.self_test:
        _self_test()
        return 0
    parser.print_help()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
