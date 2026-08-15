"""Frozen event-manifold registry tests."""

from __future__ import annotations

from pathlib import Path

from dagua.eval.ruler_v4.events import evaluate_jump_bound, load_event_registry


def test_embedded_registry_is_complete() -> None:
    """Every contract either declares events or explicit event absence."""

    registry = load_event_registry()
    covered = {entry.facet_id for entry in registry.entries} | set(
        registry.facets_declaring_no_events
    )
    assert len(registry.entries) == 12
    assert len(covered) == 45


def test_external_frozen_registry_loads() -> None:
    """The production loader accepts the authoritative JSON schema."""

    path = Path.home() / ".claude/research/dagua/ruler_v4/p3/frozen/EVENT_REGISTRY.json"
    registry = load_event_registry(path)
    assert registry.schema_version == "A5-event-registry-1.0"
    assert registry.by_facet("U17")[0].jump_bound.value == 0.0


def test_u41_jump_bound_uses_input_face_opportunity() -> None:
    """The U41 bound decreases with the frozen graph-side face opportunity."""

    event = load_event_registry().by_facet("U41")[0]
    assert evaluate_jump_bound(event, {"F0": 1.0}) == 1.0
    assert evaluate_jump_bound(event, {"F0": 12.0}) == 0.25


def test_zero_onset_event_bound_is_zero() -> None:
    """Area-overlap contact events retain smooth zero onset."""

    event = load_event_registry().by_facet("U18")[0]
    assert evaluate_jump_bound(event) == 0.0


def test_amended_formula_string_refuses_the_stale_hardcoded_evaluator() -> None:
    """P2 tripwire: a regenerated registry formula cannot silently evaluate."""

    import pytest

    from dagua.eval.ruler_v4.events import EventManifold, JumpBound, evaluate_jump_bound

    drifted = EventManifold(
        "U41",
        "U41_FACE_SPLIT",
        "arrangement face split",
        True,
        JumpBound("closed_form", formula="min(1, 0.70*min(1,3/max(1,F0)))"),
    )
    with pytest.raises(ValueError, match="registry drift"):
        evaluate_jump_bound(drifted, {"F0": 100.0})
