"""Tests for visual parity v2 JSON IO."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict

import pytest

from scripts.visual_parity import io

RESEARCH_DIR = Path(".project-context/research/sprint_visual_parity_v2")
STORE_READERS: Dict[str, Callable[[Path], Dict[str, object]]] = {
    "coverage_matrix.json": io.read_coverage_matrix,
    "ledger.json": io.read_ledger,
    "card_manifest.json": io.read_card_manifest,
}
STORE_WRITERS: Dict[str, Callable[[Path, Dict[str, object]], None]] = {
    "coverage_matrix.json": io.write_coverage_matrix,
    "ledger.json": io.write_ledger,
    "card_manifest.json": io.write_card_manifest,
}


@pytest.mark.parametrize("filename", sorted(STORE_READERS))
def test_stub_round_trip_is_byte_stable(tmp_path: Path, filename: str) -> None:
    """Committed stubs should read and write without byte changes.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    filename
        Store filename under the research directory.

    Returns
    -------
    None
        The test asserts byte stability.
    """

    source = RESEARCH_DIR / filename
    target = tmp_path / filename
    target.write_bytes(source.read_bytes())

    before = target.read_bytes()
    data = STORE_READERS[filename](target)
    STORE_WRITERS[filename](target, data)

    assert target.read_bytes() == before


@pytest.mark.parametrize("filename", sorted(STORE_READERS))
def test_unknown_schema_version_is_rejected(tmp_path: Path, filename: str) -> None:
    """Stores with unknown schema versions should be refused.

    Parameters
    ----------
    tmp_path
        Pytest temporary directory.
    filename
        Store filename under the research directory.

    Returns
    -------
    None
        The test asserts a schema error is raised.
    """

    source = RESEARCH_DIR / filename
    data = json.loads(source.read_text(encoding="utf-8"))
    data["schema_version"] = 999
    target = tmp_path / filename
    target.write_text(json.dumps(data), encoding="utf-8")

    with pytest.raises(io.SchemaVersionError):
        STORE_READERS[filename](target)


def test_self_test_entrypoint() -> None:
    """The module self-test should validate all built-in empty stores.

    Returns
    -------
    None
        The test asserts the self-test returns a successful exit code.
    """

    io._self_test()
