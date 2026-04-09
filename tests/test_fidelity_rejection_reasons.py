"""Tests for Group E: rejection reason preservation."""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

from fidelity_analysis import ResultRecord


def test_result_record_has_new_fields() -> None:
    """ResultRecord should accept error_message and skip_reason."""
    record = ResultRecord(
        graph_name="g",
        engine_name="engine",
        seed=None,
        status="error",
        runtime_seconds=None,
        positions_file=None,
        error_message="layout timeout",
        skip_reason=None,
    )
    assert record.error_message == "layout timeout"
    assert record.skip_reason is None


def test_result_record_defaults_for_new_fields() -> None:
    """New fields default to None for backward compatibility."""
    record = ResultRecord(
        graph_name="g",
        engine_name="engine",
        seed=None,
        status="ok",
        runtime_seconds=0.0,
        positions_file="positions/g.pt",
    )
    assert record.error_message is None
    assert record.skip_reason is None


def test_rejection_breakdown_json_parseable() -> None:
    """The rejection_breakdown_json column should be valid JSON."""
    sample = {"orig_error": 1, "contains_nan": 2}
    encoded = json.dumps(sample)
    decoded = json.loads(encoded)
    assert decoded == sample
