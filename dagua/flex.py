"""Flex system — soft layout targets for the differentiable engine.

Flex values express layout preferences as soft constraints rather than hard locks.
The layout engine converts them to loss terms with configurable weights:
- soft (weight=0.5): gentle preference, easily overridden by other forces
- firm (weight=2.0): noticeable pull, wins most conflicts
- rigid (weight=10.0): strong constraint, rarely overridden
- locked (weight=inf): hard constraint, enforced via post-step projection

Usage:
    from dagua.flex import Flex, LayoutFlex, AlignGroup

    flex = LayoutFlex(
        node_sep=Flex.firm(40),
        pins={"input": (Flex.locked(0), Flex.locked(0))},
        align_x=[AlignGroup(["a", "b", "c"])],
    )
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple


@dataclass(frozen=True)
class Flex:
    """A soft target value with a weight controlling how strongly the layout
    engine enforces it.

    Higher weight = stronger pull toward target.
    weight=inf means hard constraint (enforced via projection, not gradient).
    """

    target: float
    weight: float = 1.0

    @staticmethod
    def soft(value: float, weight: float = 0.5) -> Flex:
        """Gentle preference — easily overridden by other forces."""
        return Flex(target=value, weight=weight)

    @staticmethod
    def firm(value: float, weight: float = 2.0) -> Flex:
        """Noticeable pull — wins most conflicts."""
        return Flex(target=value, weight=weight)

    @staticmethod
    def rigid(value: float, weight: float = 10.0) -> Flex:
        """Strong constraint — rarely overridden."""
        return Flex(target=value, weight=weight)

    @staticmethod
    def locked(value: float) -> Flex:
        """Hard constraint — enforced via post-step projection."""
        return Flex(target=value, weight=float("inf"))

    @property
    def is_hard(self) -> bool:
        """Whether this flex value is a hard constraint (infinite weight)."""
        return self.weight == float("inf")


@dataclass
class AlignGroup:
    """A group of nodes that should share the same position on one axis."""

    nodes: List[Any]  # node IDs (resolved to indices by the engine)
    weight: float = 5.0


@dataclass
class LayoutFlex:
    """Flex targets for layout-affecting properties.

    Attach to a Graph or LayoutConfig to influence the optimization loop.
    All fields are optional — only set what you want to customize.
    """

    node_sep: Optional[Flex] = None
    rank_sep: Optional[Flex] = None
    pins: Optional[Dict[Any, Tuple[Optional[Flex], Optional[Flex]]]] = None
    align_x: Optional[List[AlignGroup]] = None  # groups sharing same x
    align_y: Optional[List[AlignGroup]] = None  # groups sharing same y
