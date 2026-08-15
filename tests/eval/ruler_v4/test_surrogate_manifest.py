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
