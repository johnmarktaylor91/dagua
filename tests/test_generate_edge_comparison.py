"""Tests for the extended edge comparison generator."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from scripts.generate_edge_comparison import (
    EXPECTED_OUTPUT_FILENAMES,
    build_edge_comparison_suite,
)


def test_build_edge_comparison_suite_emits_expected_pngs(tmp_path: Path) -> None:
    """The generator should emit the requested comparison image set."""

    if shutil.which("dot") is None:
        pytest.skip("Graphviz dot is not installed")

    result = build_edge_comparison_suite(output_dir=str(tmp_path))
    emitted = {Path(path).name for path in result.image_paths}

    assert result.output_dir == str(tmp_path.resolve())
    assert emitted == set(EXPECTED_OUTPUT_FILENAMES)
    assert len(result.image_paths) == len(EXPECTED_OUTPUT_FILENAMES)
    assert all((tmp_path / filename).exists() for filename in EXPECTED_OUTPUT_FILENAMES)
    assert all((tmp_path / filename).stat().st_size > 0 for filename in EXPECTED_OUTPUT_FILENAMES)
