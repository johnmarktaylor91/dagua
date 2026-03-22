"""igraph competitor adapters for selected layered and force-directed layouts."""

from __future__ import annotations

import random
import time
from contextlib import contextmanager
from typing import TYPE_CHECKING, Any, Iterator, Mapping, Optional

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph


@contextmanager
def _igraph_rng_seed(seed: Optional[int], enabled: bool) -> Iterator[None]:
    """Temporarily route igraph randomness through a seeded Python RNG.

    Parameters
    ----------
    seed : int | None
        Benchmark seed requested by the caller.
    enabled : bool
        Whether the wrapped layout algorithm depends on igraph's internal RNG.

    Returns
    -------
    Iterator[None]
        Context manager that restores igraph's default RNG afterwards.

    Notes
    -----
    python-igraph exposes only a setter for the global RNG. We therefore
    restore the library default PCG32 generator on exit instead of trying to
    preserve an unknown prior custom generator.
    """
    if seed is None or not enabled:
        yield
        return

    import igraph

    igraph.set_random_number_generator(random.Random(seed))
    try:
        yield
    finally:
        igraph.set_random_number_generator(None)


def _graph_to_igraph(graph: DaguaGraph) -> Any:
    """Convert a ``DaguaGraph`` into an ``igraph.Graph``.

    Parameters
    ----------
    graph : DaguaGraph
        Source graph whose node and edge structure should be mirrored.

    Returns
    -------
    Any
        ``igraph.Graph`` instance suitable for layout calls.
    """
    import igraph

    g = igraph.Graph(directed=True)
    g.add_vertices(graph.num_nodes)
    if graph.edge_index.numel() > 0:
        ei = graph.edge_index
        edges = [(ei[0, e].item(), ei[1, e].item()) for e in range(ei.shape[1])]
        g.add_edges(edges)
        if graph.edge_weights is not None:
            g.es["weight"] = graph.edge_weights.cpu().tolist()
    return g


def _igraph_pos_to_tensor(layout: Any, num_nodes: int) -> torch.Tensor:
    """Convert an igraph layout into a scaled ``[N, 2]`` tensor.

    Parameters
    ----------
    layout : Any
        igraph layout object exposing 2D coordinates by index.
    num_nodes : int
        Number of nodes expected in the output tensor.

    Returns
    -------
    torch.Tensor
        CPU tensor shaped ``[N, 2]`` in Dagua's coordinate scale.
    """
    pos = torch.zeros(num_nodes, 2)
    for i in range(min(len(layout), num_nodes)):
        # igraph layouts return coordinates in arbitrary units; scale up
        pos[i, 0] = layout[i][0] * 50.0
        pos[i, 1] = layout[i][1] * 50.0
    return pos


class _IgraphBase(CompetitorBase):
    """Base for igraph layout algorithms."""

    layout_algo: str = "sugiyama"
    layout_kwargs: dict[str, Any] = {}
    accepts_seed_matrix: bool = False
    uses_igraph_rng: bool = False

    def layout(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
    ) -> CompetitorResult:
        """Run the configured igraph layout algorithm.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout parameter.
        seed : int | None, default=None
            Random seed forwarded when the underlying igraph layout accepts a
            stochastic starting layout or RNG override.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        return self.layout_with_variant(graph, timeout=timeout, seed=seed, variant_params=None)

    def layout_with_variant(
        self,
        graph: DaguaGraph,
        timeout: float = 300.0,
        seed: Optional[int] = None,
        variant_params: Optional[Mapping[str, Any]] = None,
    ) -> CompetitorResult:
        """Run the configured igraph layout algorithm.

        Parameters
        ----------
        graph : DaguaGraph
            Graph to lay out.
        timeout : float, default=300.0
            Unused compatibility timeout parameter.
        seed : int | None, default=None
            Random seed forwarded only for igraph layouts that expose a
            ``seed`` keyword, currently Fruchterman-Reingold.

        Returns
        -------
        CompetitorResult
            Layout result and timing information.
        """
        del timeout

        import igraph  # noqa: F401

        ig = _graph_to_igraph(graph)

        start = time.perf_counter()
        try:
            kwargs = dict(self.layout_kwargs)
            if variant_params is not None:
                kwargs.update(dict(variant_params))
            if seed is not None and self.accepts_seed_matrix:
                # igraph FR's "seed" param is an initial position matrix, not an int.
                # Generate random initial positions from the integer seed instead.
                import numpy as np

                rng = np.random.RandomState(seed)
                kwargs["seed"] = rng.uniform(-1, 1, size=(graph.num_nodes, 2)).tolist()
            with _igraph_rng_seed(seed=seed, enabled=self.uses_igraph_rng):
                ig_layout = ig.layout(self.layout_algo, **kwargs)
            elapsed = time.perf_counter() - start
            pos = _igraph_pos_to_tensor(ig_layout, graph.num_nodes)
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as e:
            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=None, runtime_seconds=elapsed, error=str(e))

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
    variant_param_names = frozenset({"hgap", "maxiter", "vgap"})


@register
class IgraphFR(_IgraphBase):
    name = "igraph_fr"
    max_nodes = 50_000
    layout_algo = "fruchterman_reingold"
    layout_kwargs = {"niter": 500}
    accepts_seed_matrix = True
    variant_param_names = frozenset({"area", "grid", "niter"})


@register
class IgraphRT(_IgraphBase):
    name = "igraph_rt"
    max_nodes = 10_000
    layout_algo = "reingold_tilford"
    layout_kwargs = {}


@register
class IgraphDavidsonHarel(_IgraphBase):
    name = "igraph_davidson_harel"
    max_nodes = 500
    layout_algo = "davidson_harel"
    layout_kwargs = {}
    variant_param_names = frozenset({"cool_fact", "fineiter", "maxiter"})


@register
class IgraphKamadaKawai(_IgraphBase):
    name = "igraph_kamada_kawai"
    max_nodes = 5_000
    layout_algo = "kamada_kawai"
    layout_kwargs = {}
    variant_param_names = frozenset()


@register
class IgraphMDS(_IgraphBase):
    name = "igraph_mds"
    max_nodes = 5_000
    layout_algo = "mds"
    layout_kwargs = {}


@register
class IgraphGraphOpt(_IgraphBase):
    """GraphOpt: force-directed with simulated annealing hybrid."""

    name = "igraph_graphopt"
    max_nodes = 20_000
    layout_algo = "graphopt"
    layout_kwargs = {"niter": 500}
    accepts_seed_matrix = True
    variant_param_names = frozenset(
        {"niter", "node_charge", "node_mass", "spring_constant", "spring_length"}
    )


@register
class IgraphDRL(_IgraphBase):
    """DrL (Distributed Recursive Layout): multilevel force-directed for large graphs."""

    name = "igraph_drl"
    max_nodes = 100_000
    layout_algo = "drl"
    layout_kwargs = {}
    accepts_seed_matrix = True
    uses_igraph_rng = True
    variant_param_names = frozenset({"options", "seed", "weights"})


@register
class IgraphLGL(_IgraphBase):
    """LGL (Large Graph Layout): force-directed for large graphs."""

    name = "igraph_lgl"
    max_nodes = 100_000
    layout_algo = "lgl"
    layout_kwargs = {}
    uses_igraph_rng = True
    variant_param_names = frozenset(
        {"area", "cellsize", "coolexp", "maxdelta", "maxiter", "repulserad", "root"}
    )
