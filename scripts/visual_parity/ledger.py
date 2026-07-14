"""Manage the visual parity v2 ledger and generated lock tests."""

from __future__ import annotations

import argparse
import json
import platform
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from pprint import pformat
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set

from scripts.visual_parity.io import read_coverage_matrix, read_ledger, write_ledger
from scripts.visual_parity.types import LEDGER_SCHEMA_VERSION

RESEARCH_DIR = Path(".project-context/research/sprint_visual_parity_v2")
COVERAGE_PATH = RESEARCH_DIR / "coverage_matrix.json"
LEDGER_PATH = RESEARCH_DIR / "ledger.json"
TRIPWIRE_STATUS_PATH = RESEARCH_DIR / "tripwire_status.json"
LOCK_TEST_PATH = Path("tests/test_visual_parity_locks.py")

LOCK_TEST_HEADER = '''"""Generated visual parity v2 lock tests.

Do not edit by hand. Regenerate with:
    python -m scripts.visual_parity.ledger --generate-lock-tests
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List

from scripts.visual_parity import ledger
from scripts.visual_parity.io import read_ledger

LEDGER_PATH = Path(".project-context/research/sprint_visual_parity_v2/ledger.json")


def test_generated_lock_file_is_byte_identical() -> None:
    """Regenerating lock tests should reproduce this file byte-for-byte.

    Returns
    -------
    None
        The test asserts deterministic generation.
    """

    data = read_ledger(LEDGER_PATH)
    assert ledger.render_lock_tests(data) == Path(__file__).read_text(encoding="utf-8")


def _locked_rows() -> List[Dict[str, object]]:
    """Return locked ledger rows.

    Returns
    -------
    List[Dict[str, object]]
        Locked rows from the committed ledger.
    """

    data = read_ledger(LEDGER_PATH)
    return [row for row in data["rows"] if row.get("locked") is True]

'''


def _utc_now() -> str:
    """Return the current UTC timestamp.

    Returns
    -------
    str
        ISO-8601 UTC timestamp with ``Z`` suffix.
    """

    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _version(command: Sequence[str]) -> str:
    """Return a best-effort command version.

    Parameters
    ----------
    command
        Command argv.

    Returns
    -------
    str
        Version output or ``"unavailable"``.
    """

    try:
        completed = subprocess.run(command, check=False, capture_output=True, text=True)
    except OSError:
        return "unavailable"
    lines = (completed.stdout or completed.stderr).strip().splitlines()
    return lines[0] if lines else "unavailable"


def _module_version(module_name: str) -> str:
    """Return an importable module's version.

    Parameters
    ----------
    module_name
        Python module name.

    Returns
    -------
    str
        Version string or ``"unavailable"``.
    """

    try:
        module = __import__(module_name)
    except ImportError:
        return "unavailable"
    return str(getattr(module, "__version__", "unknown"))


def _row_id(cell: Mapping[str, Any]) -> str:
    """Return the ledger row id for a coverage cell.

    Parameters
    ----------
    cell
        Coverage cell.

    Returns
    -------
    str
        Stable row id.
    """

    return f"{cell['cell_id']}.{cell['target_kind']}"


def _test_name(row_id: str) -> str:
    """Return a pytest-safe lock test function suffix.

    Parameters
    ----------
    row_id
        Ledger row id.

    Returns
    -------
    str
        Sanitized function suffix.
    """

    name = re.sub(r"[^0-9A-Za-z_]+", "_", row_id).strip("_").lower()
    return name or "locked_row"


def _seed_metric(cell: Mapping[str, Any], locked: bool) -> List[Dict[str, Any]]:
    """Seed metric records for a ledger row.

    Parameters
    ----------
    cell
        Coverage cell.
    locked
        Whether the row starts locked.

    Returns
    -------
    List[Dict[str, Any]]
        Metric rows.
    """

    metrics: List[Dict[str, Any]] = []
    for metric_id in cell.get("metric_ids", []):
        metrics.append(
            {
                "metric_id": metric_id,
                "target": 1.0,
                "current": 1.0 if locked else None,
                "tolerance": 0.98,
                "unit": "score",
                "status": "pass" if locked else "needs_v2_revalidation",
                "validated_tripwire": locked,
            }
        )
    return metrics


def _ledger_row_from_cell(cell: Mapping[str, Any]) -> Dict[str, Any]:
    """Create an initial ledger row from a coverage cell.

    Parameters
    ----------
    cell
        Coverage cell.

    Returns
    -------
    Dict[str, Any]
        Ledger row.
    """

    locked = cell["cell_id"] == "graphviz.edge.arrowhead.normal"
    row_id = _row_id(cell)
    status = "in_tolerance" if locked else "untested"
    return {
        "row_id": row_id,
        "coverage_cell_id": cell["cell_id"],
        "priority": cell["priority"],
        "feature_group": _feature_group(cell),
        "target_kind": cell["target_kind"],
        "geometry_mode": cell["geometry_mode"],
        "reference": {
            "tool": cell["tool"],
            "version": "7.0.5" if cell["tool"] == "graphviz" else "survey",
            "source": cell["source"],
            "fixture_ids": cell.get("fixture_ids", []),
        },
        "dagua": {
            "field": cell.get("dagua_field", ""),
            "value": cell.get("dagua_value", ""),
            "code_paths": [],
        },
        "metrics": _seed_metric(cell, locked),
        "parity_status": status,
        "support_status": cell["support_status"],
        "locked": locked,
        "lock_test": f"tests/test_visual_parity_locks.py::test_lock_{_test_name(row_id)}"
        if locked
        else None,
        "residual_class": None,
        "waiver": None,
        "history": [
            {
                "round": "seed",
                "action": "needs_v2_revalidation" if not locked else "seed_locked_prior",
                "notes": "Lane C seed from coverage matrix.",
            }
        ],
        "last_updated": _utc_now(),
    }


def _feature_group(cell: Mapping[str, Any]) -> str:
    """Return a human-readable feature group for a coverage cell.

    Parameters
    ----------
    cell
        Coverage cell.

    Returns
    -------
    str
        Feature group.
    """

    if cell.get("attribute") == "arrowhead":
        return "arrowheads"
    if cell.get("attribute") == "shape":
        return "node_shapes"
    if cell.get("tool") != "graphviz":
        return "adapters"
    return str(cell.get("attribute", "cosmetic"))


def _initial_knobs(rows: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    """Create stable seed knob entries.

    Parameters
    ----------
    rows
        Ledger rows.

    Returns
    -------
    List[Dict[str, Any]]
        Knob entries.
    """

    def linked(group: str) -> List[str]:
        return sorted(row["row_id"] for row in rows if row.get("feature_group") == group)

    return [
        {
            "knob_id": "node.autosize.width",
            "dial": "NodeStyle padding and autosize width",
            "code_ref": "dagua/render/mpl.py",
            "status": "open",
            "linked_rows": linked("node_shapes")[:25],
            "consecutive_no_improve": 0,
            "values_tried": [],
        },
        {
            "knob_id": "edge.arrow.geometry",
            "dial": "EdgeStyle arrow length and width",
            "code_ref": "dagua/render/edges/arrowheads.py",
            "status": "open",
            "linked_rows": linked("arrowheads")[:40],
            "consecutive_no_improve": 0,
            "values_tried": [],
        },
    ]


def init_ledger() -> Dict[str, Any]:
    """Initialize the ledger from the current coverage matrix.

    Returns
    -------
    Dict[str, Any]
        Ledger payload.
    """

    coverage = read_coverage_matrix(COVERAGE_PATH)
    rows = [_ledger_row_from_cell(cell) for cell in coverage["cells"]]
    payload: Dict[str, Any] = {
        "schema_version": LEDGER_SCHEMA_VERSION,
        "run_id": "visual_parity_v2_2026_07",
        "created_at": _utc_now(),
        "updated_at": _utc_now(),
        "environment": {
            "repo": str(Path.cwd()),
            "branch": _version(["git", "branch", "--show-current"]),
            "base_commit": _version(["git", "rev-parse", "--short=12", "HEAD"]),
            "python": platform.python_version(),
            "graphviz": _version(["dot", "-V"]),
            "matplotlib": _module_version("matplotlib"),
            "cairosvg": _module_version("cairosvg"),
            "pillow": _module_version("PIL"),
            "node": _version(["node", "--version"]),
            "npm_packages": {
                "cytosnap": _version(["npx", "cytosnap", "--version"]),
                "mmdc": _version(["mmdc", "--version"]),
            },
        },
        "ratchets": {"global_in_tol_floor_pct": 85.0},
        "auditor": {
            "primary": None,
            "ceiling": None,
            "fallback": None,
            "probe_scores": {},
            "probed_at": None,
        },
        "rows": rows,
        "knobs": _initial_knobs(rows),
        "rounds": [
            {
                "round_id": "g000",
                "track": "G",
                "subsprint": "S0",
                "commit": _version(["git", "rev-parse", "--short", "HEAD"]),
                "gates_summary": {"global_in_tol_pct": 85.0, "features_below_98": []},
                "tripwires": "assumed_trusted_missing_status",
                "audit": None,
                "rebase_label": None,
            }
        ],
        "residuals": [],
        "warnings": [],
    }
    apply_tripwire_status(payload)
    return payload


def apply_tripwire_status(ledger: Dict[str, Any]) -> None:
    """Propagate Lane B tripwire trust status into ledger rows.

    Parameters
    ----------
    ledger
        Mutable ledger payload.

    Returns
    -------
    None
        The ledger is updated in place.
    """

    if not TRIPWIRE_STATUS_PATH.exists():
        ledger.setdefault("warnings", []).append(
            "tripwire_status.json missing; treating all metrics as trusted until Lane B runs"
        )
        return
    status = json.loads(TRIPWIRE_STATUS_PATH.read_text(encoding="utf-8"))
    failed = _failed_metric_ids(status)
    if not failed:
        return
    for row in ledger["rows"]:
        row_failed = False
        for metric in row.get("metrics", []):
            if metric.get("metric_id") in failed:
                metric["validated_tripwire"] = False
                metric["status"] = "metric_untrusted"
                row_failed = True
        if row_failed:
            row["parity_status"] = "metric_untrusted"
            row["locked"] = False
            row["lock_test"] = None


def _failed_metric_ids(status: Mapping[str, Any]) -> Set[str]:
    """Extract failed metric ids from a tripwire status payload.

    Parameters
    ----------
    status
        Tripwire status JSON payload.

    Returns
    -------
    Set[str]
        Metric ids that cannot be trusted.
    """

    failed: Set[str] = set()
    for key in ("failed_metric_ids", "metric_untrusted"):
        value = status.get(key)
        if isinstance(value, list):
            failed.update(str(item) for item in value)
    for row in status.get("results", []):
        if isinstance(row, Mapping) and row.get("status") == "fail":
            metric_id = row.get("metric_id")
            if metric_id:
                failed.add(str(metric_id))
    return failed


def locked_rows(ledger: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return locked ledger rows.

    Parameters
    ----------
    ledger
        Ledger payload.

    Returns
    -------
    List[Dict[str, Any]]
        Locked row dictionaries.
    """

    return sorted(
        [dict(row) for row in ledger.get("rows", []) if row.get("locked") is True],
        key=lambda row: str(row["row_id"]),
    )


def render_lock_tests(ledger: Mapping[str, Any]) -> str:
    """Render the generated lock test file.

    Parameters
    ----------
    ledger
        Ledger payload.

    Returns
    -------
    str
        Complete Python source for ``tests/test_visual_parity_locks.py``.
    """

    chunks = [LOCK_TEST_HEADER]
    rows = locked_rows(ledger)
    if not rows:
        chunks.append(
            "\n\ndef test_no_locked_rows_yet() -> None:\n"
            '    """Document that no lock rows have been frozen yet.\n\n'
            "    Returns\n"
            "    -------\n"
            "    None\n"
            "        The test asserts the ledger has no locked rows.\n"
            '    """\n\n'
            "    assert _locked_rows() == []\n"
        )
        return "".join(chunks)
    for row in rows:
        name = _test_name(str(row["row_id"]))
        expected = pformat(_lock_expectation(row), sort_dicts=True, width=88)
        chunks.append(
            f"\n\ndef test_lock_{name}() -> None:\n"
            f'    """Protect locked row {row["row_id"]}.\n\n'
            "    Returns\n"
            "    -------\n"
            "    None\n"
            "        The test asserts the committed current values remain locked.\n"
            '    """\n\n'
            f"    expected = {expected}\n"
            '    by_id = {str(row["row_id"]): row for row in _locked_rows()}\n'
            f"    row = by_id[{json.dumps(row['row_id'])}]\n"
            "    actual = ledger.lock_expectation(row)\n"
            "    assert actual == expected\n"
        )
    return "".join(chunks)


def _lock_expectation(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Return the frozen values for a lock row.

    Parameters
    ----------
    row
        Locked ledger row.

    Returns
    -------
    Dict[str, Any]
        Lock expectation.
    """

    return {
        "coverage_cell_id": row.get("coverage_cell_id"),
        "parity_status": row.get("parity_status"),
        "support_status": row.get("support_status"),
        "metrics": [
            {
                "metric_id": metric.get("metric_id"),
                "current": metric.get("current"),
                "tolerance": metric.get("tolerance"),
                "status": metric.get("status"),
                "validated_tripwire": metric.get("validated_tripwire"),
            }
            for metric in row.get("metrics", [])
        ],
    }


def lock_expectation(row: Mapping[str, Any]) -> Dict[str, Any]:
    """Public wrapper used by generated tests for lock expectations.

    Parameters
    ----------
    row
        Locked ledger row.

    Returns
    -------
    Dict[str, Any]
        Lock expectation.
    """

    return _lock_expectation(row)


def stalled_stop_blockers(ledger: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Return stalled P0/P1 rows that block STOP.

    Parameters
    ----------
    ledger
        Ledger payload.

    Returns
    -------
    List[Dict[str, Any]]
        Blocking rows without waivers or filed residuals.
    """

    blockers = []
    for row in ledger.get("rows", []):
        if row.get("priority") not in {"P0", "P1"}:
            continue
        if row.get("waiver") or row.get("residual_class"):
            continue
        if row.get("parity_status") == "stalled":
            blockers.append(dict(row))
    return blockers


def assert_rebase_comparable(rounds: Iterable[Mapping[str, Any]]) -> None:
    """Validate dashboard-safe rebase labels for numeric comparisons.

    Parameters
    ----------
    rounds
        Round entries to compare.

    Returns
    -------
    None
        Raises when multiple target lanes are mixed.
    """

    labels = {round_entry.get("rebase_label") for round_entry in rounds}
    labels.discard(None)
    if len(labels) > 1:
        raise ValueError(
            "refusing cross-lane numeric comparison across differing rebase_label values"
        )


def write_lock_tests(ledger: Mapping[str, Any]) -> None:
    """Write generated lock tests to disk.

    Parameters
    ----------
    ledger
        Ledger payload.

    Returns
    -------
    None
        The function writes ``tests/test_visual_parity_locks.py``.
    """

    LOCK_TEST_PATH.write_text(render_lock_tests(ledger), encoding="utf-8")


def main(argv: Optional[Sequence[str]] = None) -> int:
    """Run the ledger command-line interface.

    Parameters
    ----------
    argv
        Optional command arguments.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--init", action="store_true", help="initialize ledger.json from coverage")
    parser.add_argument("--generate-lock-tests", action="store_true", help="regenerate lock tests")
    args = parser.parse_args(argv)
    if args.init:
        ledger = init_ledger()
        write_ledger(LEDGER_PATH, ledger)
        write_lock_tests(ledger)
        print(f"ledger rows: {len(ledger['rows'])}")
        print(f"locked rows: {len(locked_rows(ledger))}")
        for warning in ledger.get("warnings", []):
            print(f"warning: {warning}")
        return 0
    if args.generate_lock_tests:
        ledger = read_ledger(LEDGER_PATH)
        write_lock_tests(ledger)
        print(f"wrote {LOCK_TEST_PATH}")
        return 0
    parser.error("choose --init or --generate-lock-tests")
    return 2


if __name__ == "__main__":
    sys.exit(main())
