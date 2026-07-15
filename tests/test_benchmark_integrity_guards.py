"""Smoke tests for benchmark integrity guardrails."""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

import pytest
import torch

_ROOT = Path(__file__).resolve().parents[1]

_VALIDATOR_SPEC = importlib.util.spec_from_file_location(
    "validate_benchmark_integrity",
    _ROOT / "scripts" / "validate_benchmark_integrity.py",
)
assert _VALIDATOR_SPEC is not None and _VALIDATOR_SPEC.loader is not None
validate_benchmark_integrity = importlib.util.module_from_spec(_VALIDATOR_SPEC)
sys.modules["validate_benchmark_integrity"] = validate_benchmark_integrity
_VALIDATOR_SPEC.loader.exec_module(validate_benchmark_integrity)

_FIDELITY_SPEC = importlib.util.spec_from_file_location(
    "definitive_fidelity_analysis",
    _ROOT / "scripts" / "definitive_fidelity_analysis.py",
)
assert _FIDELITY_SPEC is not None and _FIDELITY_SPEC.loader is not None
definitive_fidelity_analysis = importlib.util.module_from_spec(_FIDELITY_SPEC)
sys.modules["definitive_fidelity_analysis"] = definitive_fidelity_analysis
_FIDELITY_SPEC.loader.exec_module(definitive_fidelity_analysis)


def _write_position(path: Path, value: float) -> None:
    """Write a tiny position tensor.

    Parameters
    ----------
    path : Path
        Destination tensor path.
    value : float
        Fill value for the tensor.

    Returns
    -------
    None
        The tensor is written to disk.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(torch.full((3, 2), value, dtype=torch.float32), path)


def test_param_identical_reference_variants_fail(tmp_path: Path) -> None:
    """Non-whitelisted reference variants with identical tensors should fail."""
    data_dir = tmp_path / "bench"
    variant_a = "dummy_ref__for__classic_dummy_a"
    variant_b = "dummy_ref__for__classic_dummy_b"
    rows = []
    for variant in (variant_a, variant_b):
        for seed in (100, 101, 102):
            position_path = Path("positions") / f"{variant}_{seed}.pt"
            _write_position(data_dir / position_path, value=float(seed))
            rows.append(
                validate_benchmark_integrity.ResultRow(
                    key=f"tiny::{variant}::seed{seed}",
                    data_dir=data_dir,
                    payload={
                        "graph_name": "tiny",
                        "engine_name": variant,
                        "seed": seed,
                        "status": "ok",
                        "positions_file": str(position_path),
                    },
                )
            )

    errors = validate_benchmark_integrity.validate_param_sensitivity(rows)

    assert len(errors) == 1
    assert "PARAM-SENSITIVITY FAIL" in errors[0]
    assert variant_a in errors[0]
    assert variant_b in errors[0]


def test_definitive_analysis_refuses_existing_output_without_overwrite(tmp_path: Path) -> None:
    """Existing analysis output should not be appended by default."""
    output_path = tmp_path / "per_combo.jsonl"
    output_path.write_text('{"old": true}\n', encoding="utf-8")

    with pytest.raises(FileExistsError):
        definitive_fidelity_analysis.prepare_output_path(output_path, resume=False, overwrite=False)

    assert output_path.read_text(encoding="utf-8") == '{"old": true}\n'
