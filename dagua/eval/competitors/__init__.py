"""Competitor layout adapters for benchmarking.

Each adapter wraps a third-party graph layout engine (Graphviz, ELK, dagre,
NetworkX) behind a common interface so the benchmark harness can run them
uniformly.
"""

# Import all competitor modules to trigger registration
from dagua.eval.competitors import (
    dagre_competitor,  # noqa: F401
    dagua_competitor,  # noqa: F401
    elk_competitor,  # noqa: F401
    graphviz_competitor,  # noqa: F401
    igraph_competitor,  # noqa: F401
    networkx_competitor,  # noqa: F401
)
from dagua.eval.competitors.base import (
    CompetitorBase,
    CompetitorResult,
    get_available_competitors,
    get_competitors,
    register,
)

__all__ = [
    "CompetitorBase",
    "CompetitorResult",
    "get_available_competitors",
    "get_competitors",
    "register",
]
