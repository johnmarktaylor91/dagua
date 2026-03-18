"""OGDF competitor adapters for selected energy-based and layered layouts."""

from __future__ import annotations

import importlib
import time
from typing import TYPE_CHECKING, Any, Callable, List, Tuple

import torch

from dagua.eval.competitors.base import CompetitorBase, CompetitorResult, register

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph

OGDFGraphBundle = Tuple[Any, Any, List[Any]]
OGDFGraphBuilder = Callable[["DaguaGraph", Any], OGDFGraphBundle]


def _load_ogdf() -> Any:
    """Load the OGDF binding namespace from ``ogdf_python``.

    Returns
    -------
    Any
        OGDF binding namespace exposing ``Graph``, ``GraphAttributes``, and
        the layout algorithm classes.

    Raises
    ------
    ImportError
        If ``ogdf_python`` is not importable or cannot load its native
        dependencies.
    OSError
        If the binding package loads but its shared libraries are unusable.
    """
    try:
        from ogdf_python import ogdf

        return ogdf
    except (ImportError, OSError):
        pass

    try:
        return importlib.import_module("ogdf_python.ogdf")
    except (ImportError, OSError):
        pass

    import ogdf_python as ogdf_module

    if hasattr(ogdf_module, "ogdf"):
        return ogdf_module.ogdf
    return ogdf_module


def _ogdf_available() -> bool:
    """Return whether the OGDF Python bindings are usable.

    Returns
    -------
    bool
        ``True`` when the binding package and its native dependencies can be
        imported successfully.
    """
    try:
        _load_ogdf()
    except Exception:
        return False
    return True


def _build_ogdf_graph(graph: DaguaGraph, ogdf: Any) -> OGDFGraphBundle:
    """Convert ``DaguaGraph`` into an OGDF graph for undirected layouts.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose ``edge_index`` is converted.
    ogdf : Any
        OGDF binding namespace returned by :func:`_load_ogdf`.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        OGDF graph object, its graph attributes, and node handles in Dagua
        index order.
    """
    ogdf_graph = ogdf.Graph()
    graph_attributes = ogdf.GraphAttributes(
        ogdf_graph,
        ogdf.GraphAttributes.nodeGraphics | ogdf.GraphAttributes.edgeGraphics,
    )
    nodes = [ogdf_graph.newNode() for _ in range(graph.num_nodes)]

    edge_index = graph.edge_index.cpu().numpy()
    seen_edges: set[tuple[int, int]] = set()
    for edge_idx in range(edge_index.shape[1]):
        source = int(edge_index[0, edge_idx])
        target = int(edge_index[1, edge_idx])
        if source == target:
            continue

        undirected_edge = (min(source, target), max(source, target))
        if undirected_edge in seen_edges:
            continue

        # Energy-based OGDF layouts are effectively undirected; add one
        # reciprocal pair per logical edge so already-bidirectional inputs do
        # not get double-counted.
        ogdf_graph.newEdge(nodes[source], nodes[target])
        ogdf_graph.newEdge(nodes[target], nodes[source])
        seen_edges.add(undirected_edge)

    return ogdf_graph, graph_attributes, nodes


def _build_ogdf_digraph(graph: DaguaGraph, ogdf: Any) -> OGDFGraphBundle:
    """Convert ``DaguaGraph`` into an OGDF graph for directed layouts.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose ``edge_index`` is converted.
    ogdf : Any
        OGDF binding namespace returned by :func:`_load_ogdf`.

    Returns
    -------
    tuple[Any, Any, list[Any]]
        OGDF graph object, its graph attributes, and node handles in Dagua
        index order.
    """
    ogdf_graph = ogdf.Graph()
    graph_attributes = ogdf.GraphAttributes(
        ogdf_graph,
        ogdf.GraphAttributes.nodeGraphics | ogdf.GraphAttributes.edgeGraphics,
    )
    nodes = [ogdf_graph.newNode() for _ in range(graph.num_nodes)]

    edge_index = graph.edge_index.cpu().numpy()
    seen_edges: set[tuple[int, int]] = set()
    for edge_idx in range(edge_index.shape[1]):
        source = int(edge_index[0, edge_idx])
        target = int(edge_index[1, edge_idx])
        edge = (source, target)
        if source == target or edge in seen_edges:
            continue

        ogdf_graph.newEdge(nodes[source], nodes[target])
        seen_edges.add(edge)

    return ogdf_graph, graph_attributes, nodes


def _read_positions(graph_attributes: Any, nodes: List[Any], num_nodes: int) -> torch.Tensor:
    """Extract an ``[N, 2]`` CPU tensor from OGDF graph attributes.

    Parameters
    ----------
    graph_attributes : Any
        OGDF ``GraphAttributes`` object after layout execution.
    nodes : list[Any]
        Node handles in Dagua index order.
    num_nodes : int
        Number of nodes in the source graph.

    Returns
    -------
    torch.Tensor
        Position tensor shaped ``[N, 2]`` on CPU.
    """
    pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node_idx, node in enumerate(nodes):
        pos[node_idx, 0] = float(graph_attributes.x(node))
        pos[node_idx, 1] = float(graph_attributes.y(node))
    return pos


class _OGDFBase(CompetitorBase):
    """Base class for OGDF layout adapters."""

    graph_builder: OGDFGraphBuilder = staticmethod(_build_ogdf_graph)

    def create_layout_algorithm(self, ogdf: Any) -> Any:
        """Construct the OGDF layout algorithm instance for this adapter.

        Parameters
        ----------
        ogdf : Any
            OGDF binding namespace returned by :func:`_load_ogdf`.

        Returns
        -------
        Any
            Configured OGDF layout algorithm object.

        Raises
        ------
        NotImplementedError
            If a subclass does not provide an algorithm implementation.
        """
        raise NotImplementedError

    def layout(self, graph: DaguaGraph, timeout: float = 300.0) -> CompetitorResult:
        """Run the configured OGDF layout and convert its positions to torch.

        Parameters
        ----------
        graph : DaguaGraph
            Input graph to lay out.
        timeout : float, optional
            Unused adapter timeout in seconds. Included for interface
            compatibility with the benchmark harness.

        Returns
        -------
        CompetitorResult
            Layout result with positions shaped ``[N, 2]`` on CPU, or an error
            payload if the third-party engine fails.
        """
        start = time.perf_counter()
        try:
            if graph.num_nodes <= 1:
                pos = torch.zeros((graph.num_nodes, 2), dtype=torch.float32)
                elapsed = time.perf_counter() - start
                return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)

            ogdf = _load_ogdf()
            ogdf_graph, graph_attributes, nodes = self.graph_builder(graph, ogdf)
            layout_algorithm = self.create_layout_algorithm(ogdf)
            layout_algorithm.call(graph_attributes)
            pos = _read_positions(graph_attributes, nodes, graph.num_nodes)

            # Keep the native graph object referenced until after coordinates
            # are read; cppyy-backed handles can otherwise become invalid.
            _ = ogdf_graph

            elapsed = time.perf_counter() - start
            return CompetitorResult(name=self.name, pos=pos, runtime_seconds=elapsed)
        except Exception as exc:
            elapsed = time.perf_counter() - start
            return CompetitorResult(
                name=self.name,
                pos=None,
                runtime_seconds=elapsed,
                error=str(exc),
            )

    def available(self) -> bool:
        """Report whether OGDF bindings are usable in the current environment.

        Returns
        -------
        bool
            ``True`` when the OGDF Python bindings import successfully.
        """
        return _ogdf_available()


@register
class OGDFGem(_OGDFBase):
    """Competitor adapter for OGDF's GEM layout."""

    name = "ogdf_gem"
    max_nodes = 20_000

    def create_layout_algorithm(self, ogdf: Any) -> Any:
        """Create the OGDF GEM layout instance.

        Parameters
        ----------
        ogdf : Any
            OGDF binding namespace returned by :func:`_load_ogdf`.

        Returns
        -------
        Any
            OGDF GEM layout instance.
        """
        return ogdf.energybased.GEMLayout()


@register
class OGDFFMMM(_OGDFBase):
    """Competitor adapter for OGDF's FM^3 multilevel layout."""

    name = "ogdf_fmmm"
    max_nodes = 100_000

    def create_layout_algorithm(self, ogdf: Any) -> Any:
        """Create the OGDF FM^3 layout instance.

        Parameters
        ----------
        ogdf : Any
            OGDF binding namespace returned by :func:`_load_ogdf`.

        Returns
        -------
        Any
            OGDF FM^3 layout instance.
        """
        return ogdf.energybased.FMMMLayout()


@register
class OGDFStress(_OGDFBase):
    """Competitor adapter for OGDF's stress minimization layout."""

    name = "ogdf_stress"
    max_nodes = 10_000

    def create_layout_algorithm(self, ogdf: Any) -> Any:
        """Create the OGDF stress minimization layout instance.

        Parameters
        ----------
        ogdf : Any
            OGDF binding namespace returned by :func:`_load_ogdf`.

        Returns
        -------
        Any
            OGDF stress minimization layout instance.
        """
        return ogdf.energybased.StressMinimization()


@register
class OGDFLinLog(_OGDFBase):
    """Competitor adapter for OGDF's LinLog layout."""

    name = "ogdf_linlog"
    max_nodes = 20_000

    def create_layout_algorithm(self, ogdf: Any) -> Any:
        """Create the OGDF LinLog layout instance.

        Parameters
        ----------
        ogdf : Any
            OGDF binding namespace returned by :func:`_load_ogdf`.

        Returns
        -------
        Any
            OGDF LinLog layout instance.
        """
        return ogdf.energybased.LinLogLayout()


@register
class OGDFPivotMDS(_OGDFBase):
    """Competitor adapter for OGDF's Pivot-MDS layout."""

    name = "ogdf_pivot_mds"
    max_nodes = 100_000

    def create_layout_algorithm(self, ogdf: Any) -> Any:
        """Create the OGDF Pivot-MDS layout instance.

        Parameters
        ----------
        ogdf : Any
            OGDF binding namespace returned by :func:`_load_ogdf`.

        Returns
        -------
        Any
            OGDF Pivot-MDS layout instance.
        """
        return ogdf.energybased.PivotMDS()


@register
class OGDFSugiyama(_OGDFBase):
    """Competitor adapter for OGDF's Sugiyama layered layout."""

    name = "ogdf_sugiyama"
    max_nodes = 20_000
    graph_builder: OGDFGraphBuilder = staticmethod(_build_ogdf_digraph)

    def create_layout_algorithm(self, ogdf: Any) -> Any:
        """Create the OGDF Sugiyama layout instance.

        Parameters
        ----------
        ogdf : Any
            OGDF binding namespace returned by :func:`_load_ogdf`.

        Returns
        -------
        Any
            OGDF Sugiyama layout instance.
        """
        return ogdf.layered.SugiyamaLayout()


@register
class OGDFDavidsonHarel(_OGDFBase):
    """Competitor adapter for OGDF's Davidson-Harel layout."""

    name = "ogdf_davidson_harel"
    max_nodes = 500

    def create_layout_algorithm(self, ogdf: Any) -> Any:
        """Create the OGDF Davidson-Harel layout instance.

        Parameters
        ----------
        ogdf : Any
            OGDF binding namespace returned by :func:`_load_ogdf`.

        Returns
        -------
        Any
            OGDF Davidson-Harel layout instance.
        """
        return ogdf.energybased.DavidsonHarelLayout()
