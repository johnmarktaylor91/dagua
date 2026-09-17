"""Smoke test for the deprecated pipeline-fidelity validator stub."""

from __future__ import annotations

import subprocess
import sys
from pathlib import Path

SCRIPT_PATH = Path(__file__).resolve().parents[2] / "scripts" / "validate_pipeline_fidelity.py"


def test_deprecated_stub_exits_nonzero_with_pointer() -> None:
    """The retired gate runs, fails loudly, and names its replacement."""
    result = subprocess.run(
        [sys.executable, str(SCRIPT_PATH)],
        capture_output=True,
        text=True,
        timeout=60,
    )

    assert result.returncode == 2
    assert "DEPRECATED" in result.stderr
    assert "compare_reimpl_vs_original.py" in result.stderr
