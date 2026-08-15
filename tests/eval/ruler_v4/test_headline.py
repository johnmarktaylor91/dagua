"""Tests for the RENAME-compliant ordinal headline."""

from __future__ import annotations

import math

from dagua.eval.ruler_v4.headline import HeadlineProfile, ordinal_headline
from dagua.eval.ruler_v4.weights import PriorMassDisclosure


def _disclosure(*, partial: bool = False) -> PriorMassDisclosure:
    """Build a fixed PM-1 record for headline tests.

    Parameters
    ----------
    partial : bool
        Whether disclosure language must be included.

    Returns
    -------
    PriorMassDisclosure
        Deterministic disclosure record.
    """

    if partial:
        return PriorMassDisclosure(0.2, 1.0, 0.2, True, ("U12",))
    return PriorMassDisclosure(0.0, 1.0, 0.0, False, ())


def test_ordinal_index_is_monotone_without_categories() -> None:
    """More exact loss lowers the index without creating a quality category."""

    profile = HeadlineProfile(index_span=100.0, loss_scale=0.5, version="test-map")
    better = ordinal_headline(0.2, profile, _disclosure())
    worse = ordinal_headline(0.3, profile, _disclosure())

    assert better.name == "ordinal_within_graph_index"
    assert better.value > worse.value
    assert better.version == "test-map"
    assert "Ordinal within-graph index" in better.statement


def test_index_endpoints_come_only_from_explicit_profile_parameters() -> None:
    """Zero loss maps to the caller's span and the scale controls curvature."""

    profile = HeadlineProfile(index_span=73.0, loss_scale=2.0, version="explicit")

    top = ordinal_headline(0.0, profile, _disclosure())
    one_scale = ordinal_headline(2.0, profile, _disclosure())

    assert top.value == 73.0
    assert one_scale.value == 73.0 / math.e


def test_partial_profile_discloses_prior_mass_without_changing_index() -> None:
    """PM-1 changes publication language but never score mass or mapping."""

    profile = HeadlineProfile(index_span=100.0, loss_scale=1.0, version="test-map")

    ordinary = ordinal_headline(0.4, profile, _disclosure())
    partial = ordinal_headline(0.4, profile, _disclosure(partial=True))

    assert partial.value == ordinary.value
    assert "20.0%" in partial.statement
    assert "U12" in partial.statement
