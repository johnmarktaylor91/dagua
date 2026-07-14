"""Competitor layout adapters for benchmarking.

Each adapter wraps a third-party graph layout engine (Graphviz, ELK, dagre,
NetworkX) behind a common interface so the benchmark harness can run them
uniformly.
"""

# Import all competitor modules to trigger registration
from dagua.eval.competitors import (
    classic_competitor,  # noqa: F401
    coregd_competitor,  # noqa: F401
    cytoscape_competitor,  # noqa: F401
    cytoscape_fcose_competitor,  # noqa: F401
    d3force_competitor,  # noqa: F401
    dagre_competitor,  # noqa: F401
    dagua_competitor,  # noqa: F401
    elk_competitor,  # noqa: F401
    fa2_competitor,  # noqa: F401
    gephi_competitor,  # noqa: F401
    graphviz_competitor,  # noqa: F401
    igraph_competitor,  # noqa: F401
    linlog_competitor,  # noqa: F401
    networkx_competitor,  # noqa: F401
    neulay_competitor,  # noqa: F401
    ogdf_competitor,  # noqa: F401
    sgd2_competitor,  # noqa: F401
    sgd2_multi_competitor,  # noqa: F401
    smacof_competitor,  # noqa: F401
    sparse_stress_competitor,  # noqa: F401
    tfdp_competitor,  # noqa: F401
    tsne_competitor,  # noqa: F401
    umap_competitor,  # noqa: F401
)
from dagua.eval.competitors.base import (
    _COMPETITORS,
    CompetitorBase,
    CompetitorResult,
    get_available_competitors,
    get_competitor,
    get_competitors,
    register,
)

__all__ = [
    "_COMPETITORS",
    "CompetitorBase",
    "CompetitorResult",
    "get_available_competitors",
    "get_competitor",
    "get_competitors",
    "register",
]
