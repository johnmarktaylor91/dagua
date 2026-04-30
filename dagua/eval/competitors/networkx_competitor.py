"""NetworkX competitor adapters — spring_layout and kamada_kawai_layout."""

from __future__ import annotations

import time
from typing import TYPE_CHECKING, Any, Mapping, Optional

import torch

from dagua.eval.competitors.base import (
    CompetitorBase,
    CompetitorResult,
    register,
)

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


def _graph_to_nx(graph: DaguaGraph) -> Any:
    """Convert a ``DaguaGraph`` to a weighted ``networkx.DiGraph``.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph whose topology and optional edge weights should be copied.

    Returns
    -------
    Any
        ``networkx.DiGraph`` with ``weight`` edge attributes when available.
    """
    import networkx as nx

    G = nx.DiGraph()
    G.add_nodes_from(range(graph.num_nodes))
    if graph.edge_index.numel() > 0:
        ei = graph.edge_index
        weights = graph.edge_weights
        for e in range(ei.shape[1]):
            source = ei[0, e].item()
            target = ei[1, e].item()
            if weights is not None:
                G.add_edge(source, target, weight=float(weights[e].item()))
            else:
                G.add_edge(source, target)
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
    layout_kwargs: dict[str, Any] = {}

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the configured NetworkX layout algorithm.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout parameter.
        seed : int | None, default=None
            Random seed forwarded when the underlying NetworkX layout accepts
            a ``seed`` keyword. Deterministic layouts ignore it.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the configured NetworkX layout algorithm.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout parameter.
        seed : int | None, default=None
            Random seed forwarded when the underlying NetworkX layout accepts
            a ``seed`` keyword. Deterministic layouts ignore it.

        Returns
        -------
        CompetitorResult
            Layout result and runtime information.
        """
        del timeout

        import networkx as nx

        G = _graph_to_nx(graph)

        start = time.perf_counter()
        try:
            func = getattr(nx, self.layout_func)
            layout_kwargs = dict(self.layout_kwargs)
            if variant_params is not None:
                layout_kwargs.update(dict(variant_params))
            if seed is not None and "seed" in layout_kwargs:
                layout_kwargs["seed"] = seed
            nx_pos = func(G, **layout_kwargs)
            elapsed = time.perf_counter() - start
            pos = _nx_pos_to_tensor(nx_pos, graph.num_nodes)
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=None, runtime_seconds=elapsed, error=str(e))

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
    layout_kwargs = {"seed": 42, "iterations": 50, "method": "force"}
    variant_param_names = frozenset({"gravity", "iterations", "k", "scale", "method"})


@register
class NetworkXKamadaKawai(_NetworkXBase):
    name = "nx_kamada_kawai"
    max_nodes = 5_000
    layout_func = "kamada_kawai_layout"
    layout_kwargs = {}
    variant_param_names = frozenset()


@register
class NetworkXSpectral(_NetworkXBase):
    """Competitor adapter for NetworkX's spectral layout."""

    name = "nx_spectral"
    max_nodes = 10_000
    layout_func = "spectral_layout"
    layout_kwargs = {"dim": 2}
    variant_param_names = frozenset({"dim", "scale"})
