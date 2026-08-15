"""Ordinal within-graph headline mapping for RULER V4."""

from __future__ import annotations

import math
from dataclasses import dataclass

from dagua.eval.ruler_v4.weights import PriorMassDisclosure


@dataclass(frozen=True)
class HeadlineProfile:
    """Declare the monotone ordinal-index map explicitly.

    Parameters
    ----------
    index_span : float
        Positive top of the published ordinal index range.
    loss_scale : float
        Positive lambda converting exact loss into index coordinates.
    version : str
        Frozen headline-map version carried into structured output.
    """

    index_span: float
    loss_scale: float
    version: str

    def __post_init__(self) -> None:
        """Validate map parameters without attaching policy categories.

        Raises
        ------
        ValueError
            If the map is non-finite, nonpositive, or unversioned.
        """

        if (
            not math.isfinite(self.index_span)
            or self.index_span <= 0.0
            or not math.isfinite(self.loss_scale)
            or self.loss_scale <= 0.0
            or not self.version
        ):
            raise ValueError("headline map requires positive finite parameters and a version")


@dataclass(frozen=True)
class HeadlineResult:
    """Publish one ordinal index with its certification language.

    Parameters
    ----------
    name : str
        RENAME-compliant headline label.
    value : float
        Monotone ordinal index value.
    version : str
        Headline-map version.
    statement : str
        Human-readable scope and any PM-1 disclosure.
    """

    name: str
    value: float
    version: str
    statement: str


def ordinal_headline(
    l_total: float,
    profile: HeadlineProfile,
    prior_mass: PriorMassDisclosure,
) -> HeadlineResult:
    """Map exact pre-map loss to the descriptive ordinal index.

    Parameters
    ----------
    l_total : float
        Nonnegative exact composition loss.
    profile : HeadlineProfile
        Explicit monotone-map parameters.
    prior_mass : PriorMassDisclosure
        Per-profile PM-1 record.

    Returns
    -------
    HeadlineResult
        Descriptive within-graph index with no categorical interpretation.

    Raises
    ------
    ValueError
        If ``l_total`` is negative or non-finite.
    """

    if not math.isfinite(l_total) or l_total < 0.0:
        raise ValueError("headline loss must be finite and nonnegative")
    value = profile.index_span * math.exp(-l_total / profile.loss_scale)
    scope = (
        "Ordinal within-graph index; certified only for drawings of the same graph, "
        "validated profile, scene profile, and ruler version."
    )
    if prior_mass.partial:
        scope = (
            f"{scope} Prior-driven composite mass is {prior_mass.fraction:.1%}; "
            f"affected facets: {', '.join(prior_mass.affected_facets)}."
        )
    return HeadlineResult(
        name="ordinal_within_graph_index",
        value=value,
        version=profile.version,
        statement=scope,
    )
