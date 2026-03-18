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


def _count_components_and_acyclic(edge_index: torch.Tensor, num_nodes: int) -> tuple[int, bool]:
    """Return connected-component count and acyclicity of the undirected graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    tuple[int, bool]
        ``(num_components, is_acyclic)`` for the underlying undirected graph.
    """
    if num_nodes == 0:
        return 0, True

    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    # Trees and forests require E <= N - 1. Once that bound is exceeded, the
    # caller only needs a conservative "not acyclic" result, so we skip the
    # Python union-find entirely on dense graphs.
    if num_edges > num_nodes - 1:
        return 1, False

    parents = list(range(num_nodes))
    ranks = [0] * num_nodes
    is_acyclic = True

    if edge_index.numel() > 0:
        cpu_edges = edge_index.detach().cpu()
        sources = cpu_edges[0].tolist()
        targets = cpu_edges[1].tolist()

        for source, target in zip(sources, targets):
            if source == target:
                is_acyclic = False
                continue

            source_root = _find_root(parents, source)
            target_root = _find_root(parents, target)
            if source_root == target_root:
                is_acyclic = False
                continue

            if ranks[source_root] < ranks[target_root]:
                parents[source_root] = target_root
            elif ranks[source_root] > ranks[target_root]:
                parents[target_root] = source_root
            else:
                parents[target_root] = source_root
                ranks[source_root] += 1

    component_roots = {_find_root(parents, node) for node in range(num_nodes)}
    return len(component_roots), is_acyclic


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

    prefer_device = "cuda" if torch.cuda.is_available() else "cpu"
    computed_layers = longest_path_layering(
        edge_index.detach().cpu(), num_nodes, device=prefer_device
    )
    if isinstance(computed_layers, torch.Tensor):
        return computed_layers.to(device="cpu", dtype=torch.long)
    return torch.tensor(computed_layers, dtype=torch.long)


def _analyze_layers(layer_assignments: Optional[torch.Tensor], num_nodes: int) -> tuple[int, float]:
    """Return layer-count metadata for the graph.

    Parameters
    ----------
    layer_assignments : torch.Tensor, optional
        Layer assignments shaped ``[N]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    tuple[int, float]
        ``(num_layers, avg_layer_width)``. Both are zeroed when layering is
        unavailable.
    """
    if layer_assignments is None or layer_assignments.numel() == 0:
        return 0, 0.0

    max_layer = int(layer_assignments.max().item())
    layer_counts = torch.bincount(layer_assignments, minlength=max_layer + 1)
    num_layers = int((layer_counts > 0).sum().item())
    if num_layers == 0:
        return 0, 0.0
    return num_layers, float(num_nodes) / float(num_layers)


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
        num_layers, avg_layer_width = _analyze_layers(resolved_layers, num_nodes)
        return GraphStructure(
            family=GraphFamily.GENERAL,
            num_components=1,
            max_degree=0,
            num_layers=num_layers,
            avg_layer_width=avg_layer_width,
            is_planar_hint=num_edges < 3 * num_nodes - 6,
        )

    degree = _compute_degree(edge_index, num_nodes)
    max_degree = int(degree.max().item()) if degree.numel() > 0 else 0
    num_components, is_acyclic = _count_components_and_acyclic(edge_index, num_nodes)

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
    num_layers, avg_layer_width = _analyze_layers(resolved_layers, num_nodes)
    is_bipartite_dag = num_layers == 2
    is_wide_layered = (
        num_layers > 0
        and avg_layer_width >= 100.0
        and float(num_layers) <= max(float(num_nodes) / 100.0, 1.0)
    )

    is_planar_hint = (num_edges < 3 * num_nodes - 6) if num_nodes >= 3 else True

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

    return GraphStructure(
        family=family,
        num_components=num_components,
        max_degree=max_degree,
        num_layers=num_layers,
        avg_layer_width=avg_layer_width,
        is_planar_hint=is_planar_hint,
    )
