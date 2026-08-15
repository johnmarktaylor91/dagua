"""Tests for manifest-driven surrogate compilation."""

from dagua.eval.ruler_v4.contracts import CONTRACTS, SCORED_SUBTERM_COUNT
from dagua.eval.ruler_v4.surrogate.manifest import (
    CompilationRule,
    compile_surrogate_manifest,
)


def test_compiler_traces_every_frozen_manifest_subterm() -> None:
    """The compiler consumes all 91 rows without a second inventory."""

    compiled = compile_surrogate_manifest()
    expected = {
        subterm_id: (facet_id, contract.sha256)
        for facet_id, contract in CONTRACTS.items()
        for subterm_id in contract.scored_subterms
    }

    assert len(compiled.terms) == SCORED_SUBTERM_COUNT == 91
    assert set(compiled.by_subterm) == set(expected)
    for subterm_id, (facet_id, contract_sha256) in expected.items():
        trace = compiled.by_subterm[subterm_id]
        assert trace.facet_id == facet_id
        assert trace.contract_sha256 == contract_sha256
        assert trace.compilation_rule is CompilationRule.EXACT_DEFECT_IDENTITY
        assert trace.smoothing is None


def test_compiler_digest_is_deterministic() -> None:
    """Repeated compilation produces the same source-graph digest."""

    assert compile_surrogate_manifest().source_digest == compile_surrogate_manifest().source_digest


def test_classification_table_covers_every_scored_subterm() -> None:
    """CLASSIFICATION.md classifies all 91 rows exactly once, enum-valid.

    The P3FIX 91-row sweep (DISCREPANCIES entry 38's exemption rule) is
    published as a markdown table; this gate keeps it complete against
    the frozen manifest so no future row lands unclassified.
    """

    import re
    from pathlib import Path

    table_path = (
        Path(__file__).resolve().parents[3]
        / "dagua"
        / "eval"
        / "ruler_v4"
        / "surrogate"
        / "CLASSIFICATION.md"
    )
    rows = {}
    pattern = re.compile(
        r"^\| (U\S+) \| (naturally-smooth|contract-smoothed|irreducibly-discrete) \| ([^|]+) \|"
    )
    for line in table_path.read_text().splitlines():
        match = pattern.match(line)
        if match:
            subterm_id, class_name, channel = match.groups()
            assert subterm_id not in rows, f"duplicate classification row: {subterm_id}"
            rows[subterm_id] = (class_name, channel.strip())
    expected = {
        subterm_id for contract in CONTRACTS.values() for subterm_id in contract.scored_subterms
    }
    assert set(rows) == expected, (
        f"missing={sorted(expected - set(rows))} extra={sorted(set(rows) - expected)}"
    )
    assert len(rows) == 91
    valid_channels = {"live", "detached-channel", "input-owned", "diagnostic"}
    for subterm_id, (class_name, channel) in rows.items():
        base = channel.split(",")[0].strip()
        assert base in valid_channels, f"{subterm_id}: unknown channel {channel!r}"
