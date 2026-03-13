"""NetworkX competitor adapters — spring_layout and kamada_kawai_layout."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


def _graph_to_nx(graph: DaguaGraph):
    """Convert DaguaGraph to networkx.DiGraph."""
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(graph.num_nodes))
    if graph.edge_index.numel() > 0:
        ei = graph.edge_index
        for e in range(ei.shape[1]):
            G.add_edge(ei[0, e].item(), ei[1, e].item())
    return G


def _nx_pos_to_tensor(nx_pos: dict, num_nodes: int) -> torch.Tensor:
    """Convert networkx position dict to [N, 2] tensor, scaled to dagua units."""
    pos = torch.zeros(num_nodes, 2)
    for node_id, (x, y) in nx_pos.items():
        if node_id < num_nodes:
            # NetworkX layouts return ~[-1, 1] range; scale up for comparability
            pos[node_id, 0] = x * 500.0
            pos[node_id, 1] = y * 500.0
    return pos


class _NetworkXBase(CompetitorBase):
    """Base for NetworkX layout algorithms."""

    layout_func: str = "spring_layout"
    layout_kwargs: dict = {}

    def layout(self, graph: DaguaGraph, timeout: float = 300.0) -> CompetitorResult:
        import networkx as nx

        G = _graph_to_nx(graph)

        start = time.perf_counter()
        try:
            func = getattr(nx, self.layout_func)
            nx_pos = func(G, **self.layout_kwargs)
            elapsed = time.perf_counter() - start
            pos = _nx_pos_to_tensor(nx_pos, graph.num_nodes)
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
            import networkx  # noqa: F401

            return True
        except ImportError:
            return False


@register
class NetworkXSpring(_NetworkXBase):
    name = "nx_spring"
    max_nodes = 50_000
    layout_func = "spring_layout"
    layout_kwargs = {"seed": 42, "iterations": 50}


@register
class NetworkXKamadaKawai(_NetworkXBase):
    name = "nx_kamada_kawai"
    max_nodes = 5_000
    layout_func = "kamada_kawai_layout"
    layout_kwargs = {}
