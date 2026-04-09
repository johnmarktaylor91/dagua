"""Schema checks for pairwise fidelity comparisons."""

from __future__ import annotations

from dataclasses import asdict

from scripts.fidelity_analysis import PairwiseComparison


def test_pairwise_comparison_supports_diagnostic_columns() -> None:
    """Verify PairwiseComparison carries the diagnostic CSV schema fields."""
    comparison = PairwiseComparison(
        comparison_type="orig-reimpl",
        seed_a=1,
        seed_b=2,
        procrustes_rmsd=0.25,
        scale_ratio=1.1,
        reflected=True,
        max_node_displacement=1.5,
        variant_id="demo_variant",
    )

    assert isinstance(comparison.variant_id, str)
    assert isinstance(comparison.reflected, bool)
    assert isinstance(comparison.max_node_displacement, float)

    record = asdict(comparison)

    assert record["variant_id"] == "demo_variant"
    assert record["reflected"] is True
    assert record["max_node_displacement"] == 1.5
