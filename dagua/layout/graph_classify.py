"""Detect common graph structures for fast-path layout dispatch.

Runs in O(V+E) and returns a :class:`GraphStructure` describing the detected
family. The layout engine uses this metadata to skip expensive general-purpose
optimization when a specialized shortcut is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Optional

import torch

from dagua.utils import longest_path_layering


class GraphFamily(Enum):
    """Detected graph structure family."""

    GENERAL = auto()
    TREE = auto()
    FOREST = auto()
    CHAIN = auto()
    BIPARTITE_DAG = auto()
    WIDE_LAYERED = auto()
    GRID = auto()


@dataclass(frozen=True)
class GraphStructure:
    """Result of graph structure analysis."""

    family: GraphFamily
    num_components: int
    max_degree: int
    num_layers: int
    avg_layer_width: float
    is_planar_hint: bool
    is_acyclic: bool = True
    max_layer_width: int = 0
    layer_width_cv: float = 0.0
    edge_to_node_ratio: float = 0.0
    is_directed_acyclic: bool = True
    topology_tags: tuple[str, ...] = ()


def _compute_degree(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return undirected node degrees for ``edge_index``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    torch.Tensor
        Degree tensor shaped ``[N]``.
    """
    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    degree = torch.zeros(num_nodes, dtype=torch.long)
    if num_edges == 0:
        return degree

    ones = torch.ones(num_edges, dtype=torch.long)
    degree.scatter_add_(0, edge_index[0].cpu(), ones)
    degree.scatter_add_(0, edge_index[1].cpu(), ones)
    return degree


def _find_root(parents: list[int], node: int) -> int:
    """Return the union-find root for ``node`` with path compression.

    Parameters
    ----------
    parents : list[int]
        Parent array for the union-find forest.
    node : int
        Node whose canonical root should be found.

    Returns
    -------
    int
        Canonical root index for ``node``.
    """
    while parents[node] != node:
        parents[node] = parents[parents[node]]
        node = parents[node]
    return node


def _count_components(edge_index: torch.Tensor, num_nodes: int) -> int:
    """Return the number of connected components in the undirected graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    int
        Number of connected components in the underlying undirected graph.
    """
    if num_nodes == 0:
        return 0

    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    if num_edges > num_nodes - 1:
        # Preserve the dense-graph fast path used by classifier callers that
        # only need to distinguish tree/forest shortcuts from general graphs.
        return 1

    parents = list(range(num_nodes))
    ranks = [0] * num_nodes

    if edge_index.numel() > 0:
        cpu_edges = edge_index.detach().cpu()
        sources = cpu_edges[0].tolist()
        targets = cpu_edges[1].tolist()

        for source, target in zip(sources, targets):
            if source == target:
                continue

            source_root = _find_root(parents, source)
            target_root = _find_root(parents, target)
            if source_root == target_root:
                continue

            if ranks[source_root] < ranks[target_root]:
                parents[source_root] = target_root
            elif ranks[source_root] > ranks[target_root]:
                parents[target_root] = source_root
            else:
                parents[target_root] = source_root
                ranks[source_root] += 1

    component_roots = {_find_root(parents, node) for node in range(num_nodes)}
    return len(component_roots)


def _is_undirected_acyclic(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Return whether the underlying undirected graph has no cycles.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    bool
        ``True`` when the undirected graph is acyclic.
    """
    if num_nodes == 0:
        return True

    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    if num_edges > num_nodes - 1:
        return False

    parents = list(range(num_nodes))
    ranks = [0] * num_nodes

    if edge_index.numel() > 0:
        cpu_edges = edge_index.detach().cpu()
        sources = cpu_edges[0].tolist()
        targets = cpu_edges[1].tolist()

        for source, target in zip(sources, targets):
            if source == target:
                return False

            source_root = _find_root(parents, source)
            target_root = _find_root(parents, target)
            if source_root == target_root:
                return False

            if ranks[source_root] < ranks[target_root]:
                parents[source_root] = target_root
            elif ranks[source_root] > ranks[target_root]:
                parents[target_root] = source_root
            else:
                parents[target_root] = source_root
                ranks[source_root] += 1

    return True


def _is_directed_acyclic(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Return whether the directed graph is acyclic.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    bool
        ``True`` when all directed edges admit a topological ordering.
    """
    if num_nodes == 0 or edge_index.numel() == 0:
        return True

    num_edges = edge_index.shape[1]
    if num_nodes > 50_000 and num_edges > num_nodes - 1:
        # Large dense DAGs can be acyclic, but aspect-policy tags are only
        # applied to small benchmark-scale families. Keep classification O(1)
        # for scale guards instead of allocating a Python adjacency list.
        return False

    cpu_edges = edge_index.detach().cpu()
    sources = cpu_edges[0].tolist()
    targets = cpu_edges[1].tolist()
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    indegree = [0] * num_nodes

    for source, target in zip(sources, targets):
        if source == target:
            return False
        adjacency[source].append(target)
        indegree[target] += 1

    queue = [node for node, degree in enumerate(indegree) if degree == 0]
    queue_index = 0
    visited = 0
    while queue_index < len(queue):
        node = queue[queue_index]
        queue_index += 1
        visited += 1
        for target in adjacency[node]:
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)

    return visited == num_nodes


def _resolve_layer_assignments(
    edge_index: torch.Tensor,
    num_nodes: int,
    layer_assignments: Optional[torch.Tensor],
) -> Optional[torch.Tensor]:
    """Return layer assignments as a CPU ``torch.long`` tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    layer_assignments : torch.Tensor, optional
        Pre-computed layer assignments.

    Returns
    -------
    torch.Tensor | None
        Layer assignments shaped ``[N]`` or ``None`` when not available.
    """
    if layer_assignments is not None:
        return layer_assignments.detach().to(device="cpu", dtype=torch.long)
    if num_nodes == 0 or edge_index.numel() == 0:
        return None

    prefer_device = "cuda" if torch.cuda.is_available() and torch.cuda.device_count() > 0 else "cpu"
    computed_layers = longest_path_layering(
        edge_index.detach().cpu(), num_nodes, device=prefer_device
    )
    if isinstance(computed_layers, torch.Tensor):
        return computed_layers.to(device="cpu", dtype=torch.long)
    return torch.tensor(computed_layers, dtype=torch.long)


def _analyze_layers(
    layer_assignments: Optional[torch.Tensor],
    num_nodes: int,
) -> tuple[int, float, int, float]:
    """Return layer-count and width metadata for the graph.

    Parameters
    ----------
    layer_assignments : torch.Tensor, optional
        Layer assignments shaped ``[N]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    tuple[int, float, int, float]
        ``(num_layers, avg_layer_width, max_layer_width, layer_width_cv)``.
        Values are zeroed when layering is unavailable.
    """
    if layer_assignments is None or layer_assignments.numel() == 0:
        return 0, 0.0, 0, 0.0

    max_layer = int(layer_assignments.max().item())
    layer_counts = torch.bincount(layer_assignments, minlength=max_layer + 1)
    nonempty_counts = layer_counts[layer_counts > 0]
    num_layers = int(nonempty_counts.numel())
    if num_layers == 0:
        return 0, 0.0, 0, 0.0

    widths = nonempty_counts.to(dtype=torch.float32)
    avg_layer_width = float(num_nodes) / float(num_layers)
    max_layer_width = int(nonempty_counts.max().item())
    if avg_layer_width <= 0.0:
        layer_width_cv = 0.0
    else:
        layer_width_cv = float(widths.std(unbiased=False).item()) / avg_layer_width
    return num_layers, avg_layer_width, max_layer_width, layer_width_cv


def _derive_topology_tags(
    family: GraphFamily,
    num_nodes: int,
    num_layers: int,
    max_layer_width: int,
    max_degree: int,
    edge_to_node_ratio: float,
    layer_width_cv: float,
    is_planar_hint: bool,
    is_directed_acyclic: bool,
) -> tuple[str, ...]:
    """Return conservative topology tags for aspect-ratio policy.

    Parameters
    ----------
    family : GraphFamily
        Existing fast-path structural family.
    num_nodes : int
        Number of nodes in the graph.
    num_layers : int
        Number of nonempty layers.
    max_layer_width : int
        Maximum number of nodes in any layer.
    max_degree : int
        Maximum undirected degree.
    edge_to_node_ratio : float
        Directed edge count divided by node count.
    layer_width_cv : float
        Coefficient of variation of nonempty layer widths.
    is_planar_hint : bool
        Whether the edge count satisfies the planar sparsity bound.
    is_directed_acyclic : bool
        Whether directed edges admit a topological ordering.

    Returns
    -------
    tuple[str, ...]
        Stable tag strings used by config-time policy resolution.
    """
    tags: list[str] = []
    if not is_directed_acyclic:
        return ()
    if family in {GraphFamily.TREE, GraphFamily.FOREST, GraphFamily.CHAIN}:
        return ()

    is_wide_layered = family in {GraphFamily.BIPARTITE_DAG, GraphFamily.WIDE_LAYERED} or (
        num_layers <= 3 and max_layer_width >= 24
    )
    if is_wide_layered:
        tags.append("wide_layered")
        if family == GraphFamily.BIPARTITE_DAG:
            tags.append("bipartite_dag")

    is_lattice_like = (
        is_planar_hint
        and 2 <= max_degree <= 6
        and 1.0 <= edge_to_node_ratio <= 2.2
        and num_layers >= 5
        and layer_width_cv <= 0.45
    )
    if is_lattice_like:
        tags.append("lattice_like")

    is_planar_dag = is_planar_hint and max_degree <= 4 and num_layers >= 5 and not is_lattice_like
    if is_planar_dag:
        tags.append("planar_dag")

    is_dense_dag = (
        edge_to_node_ratio >= 3.5 and max_layer_width <= 3 and num_layers >= int(0.6 * num_nodes)
    )
    if is_dense_dag:
        tags.append("dense_dag")

    return tuple(tags)


def classify_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    layer_assignments: Optional[torch.Tensor] = None,
) -> GraphStructure:
    """Classify graph structure in O(V+E).

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes.
    layer_assignments : torch.Tensor, optional
        Pre-computed layer assignments. When omitted, the classifier computes a
        longest-path layering to enable layered-graph heuristics.

    Returns
    -------
    GraphStructure
        Detected structure with metadata.
    """
    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0

    # At very large scale, classification is always GENERAL — skip the
    # expensive degree computation (20GB+ allocation) and union-find.
    if num_nodes > 10_000_000:
        resolved_layers = (
            _resolve_layer_assignments(edge_index, num_nodes, layer_assignments)
            if layer_assignments is not None
            else None
        )
        num_layers, avg_layer_width, max_layer_width, layer_width_cv = _analyze_layers(
            resolved_layers,
            num_nodes,
        )
        edge_to_node_ratio = float(num_edges) / float(num_nodes) if num_nodes > 0 else 0.0
        return GraphStructure(
            family=GraphFamily.GENERAL,
            num_components=1,
            max_degree=0,
            num_layers=num_layers,
            avg_layer_width=avg_layer_width,
            is_planar_hint=num_edges < 3 * num_nodes - 6,
            is_acyclic=True,
            max_layer_width=max_layer_width,
            layer_width_cv=layer_width_cv,
            edge_to_node_ratio=edge_to_node_ratio,
            is_directed_acyclic=True,
        )

    degree = _compute_degree(edge_index, num_nodes)
    max_degree = int(degree.max().item()) if degree.numel() > 0 else 0
    num_components = _count_components(edge_index, num_nodes)
    is_acyclic = _is_undirected_acyclic(edge_index, num_nodes)
    is_directed_acyclic = _is_directed_acyclic(edge_index, num_nodes)

    is_tree = num_nodes > 0 and num_components == 1 and is_acyclic and num_edges == num_nodes - 1
    is_forest = (
        num_nodes > 0
        and num_components >= 1
        and is_acyclic
        and num_edges == num_nodes - num_components
        and not is_tree
    )

    degree_one_count = int((degree == 1).sum().item()) if degree.numel() > 0 else 0
    is_chain = is_tree and max_degree <= 2 and (num_nodes <= 2 or degree_one_count == 2)

    resolved_layers = _resolve_layer_assignments(edge_index, num_nodes, layer_assignments)
    num_layers, avg_layer_width, max_layer_width, layer_width_cv = _analyze_layers(
        resolved_layers,
        num_nodes,
    )
    is_bipartite_dag = num_layers == 2
    is_wide_layered = (
        num_layers > 0
        and avg_layer_width >= 100.0
        and float(num_layers) <= max(float(num_nodes) / 100.0, 1.0)
    )

    is_planar_hint = (num_edges < 3 * num_nodes - 6) if num_nodes >= 3 else True
    edge_to_node_ratio = float(num_edges) / float(num_nodes) if num_nodes > 0 else 0.0

    if is_chain:
        family = GraphFamily.CHAIN
    elif is_tree:
        family = GraphFamily.TREE
    elif is_bipartite_dag:
        family = GraphFamily.BIPARTITE_DAG
    elif is_wide_layered:
        family = GraphFamily.WIDE_LAYERED
    elif is_forest:
        family = GraphFamily.FOREST
    else:
        family = GraphFamily.GENERAL
    topology_tags = _derive_topology_tags(
        family=family,
        num_nodes=num_nodes,
        num_layers=num_layers,
        max_layer_width=max_layer_width,
        max_degree=max_degree,
        edge_to_node_ratio=edge_to_node_ratio,
        layer_width_cv=layer_width_cv,
        is_planar_hint=is_planar_hint,
        is_directed_acyclic=is_directed_acyclic,
    )

    return GraphStructure(
        family=family,
        num_components=num_components,
        max_degree=max_degree,
        num_layers=num_layers,
        avg_layer_width=avg_layer_width,
        is_planar_hint=is_planar_hint,
        is_acyclic=is_acyclic,
        max_layer_width=max_layer_width,
        layer_width_cv=layer_width_cv,
        edge_to_node_ratio=edge_to_node_ratio,
        is_directed_acyclic=is_directed_acyclic,
        topology_tags=topology_tags,
    )
