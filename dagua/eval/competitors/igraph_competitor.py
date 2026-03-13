"""igraph competitor adapters — Sugiyama, Fruchterman-Reingold, Reingold-Tilford."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


def _graph_to_igraph(graph: DaguaGraph):
    """Convert DaguaGraph to igraph.Graph (minimal, for layout only)."""
    import igraph

    g = igraph.Graph(directed=True)
    g.add_vertices(graph.num_nodes)
    if graph.edge_index.numel() > 0:
        ei = graph.edge_index
        edges = [(ei[0, e].item(), ei[1, e].item()) for e in range(ei.shape[1])]
        g.add_edges(edges)
    return g


def _igraph_pos_to_tensor(layout, num_nodes: int) -> torch.Tensor:
    """Convert igraph Layout to [N, 2] tensor, scaled to dagua units."""
    pos = torch.zeros(num_nodes, 2)
    for i in range(min(len(layout), num_nodes)):
        # igraph layouts return coordinates in arbitrary units; scale up
        pos[i, 0] = layout[i][0] * 50.0
        pos[i, 1] = layout[i][1] * 50.0
    return pos


class _IgraphBase(CompetitorBase):
    """Base for igraph layout algorithms."""

    layout_algo: str = "sugiyama"
    layout_kwargs: dict = {}

    def layout(self, graph: DaguaGraph, timeout: float = 300.0) -> CompetitorResult:
        import igraph  # noqa: F401

        ig = _graph_to_igraph(graph)

        start = time.perf_counter()
        try:
            ig_layout = ig.layout(self.layout_algo, **self.layout_kwargs)
            elapsed = time.perf_counter() - start
            pos = _igraph_pos_to_tensor(ig_layout, graph.num_nodes)
            return CompetitorResult(
                name=self.name, pos=pos, runtime_seconds=elapsed
            )
        except Exception as e:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name, pos=None, runtime_seconds=elapsed, error=str(e)
            )

    def available(self) -> bool:
        try:
            import igraph  # noqa: F401
            return True
        except ImportError:
            return False


@register
class IgraphSugiyama(_IgraphBase):
    name = "igraph_sugiyama"
    max_nodes = 5_000
    layout_algo = "sugiyama"
    layout_kwargs = {}


@register
class IgraphFR(_IgraphBase):
    name = "igraph_fr"
    max_nodes = 50_000
    layout_algo = "fruchterman_reingold"
    layout_kwargs = {"niter": 500}


@register
class IgraphRT(_IgraphBase):
    name = "igraph_rt"
    max_nodes = 10_000
    layout_algo = "reingold_tilford"
    layout_kwargs = {}
