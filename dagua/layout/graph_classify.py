"""Detect common graph structures for fast-path layout dispatch.

Runs in O(V+E) and returns a :class:`GraphStructure` describing the detected
family. The layout engine uses this metadata to skip expensive general-purpose
optimization when a specialized shortcut is sufficient.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum, auto
from typing import Any, Optional

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
    FORCE_DIRECTED = auto()
    HYBRID = auto()


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
    is_semantically_directed: Optional[bool] = None
    num_layers_effective: int = 0
    cyclicity_ratio: float = 0.0
    has_dominant_component: bool = True
    is_planar: Optional[bool] = None
    planar_embedding: Any = None
    # Provenance of is_semantically_directed: True only when the graph
    # object carried an explicit declaration. Routing that optimizes an
    # undirected-flavored objective must not fire on low-confidence
    # inference alone (r80: two directed graphs were mis-inferred as
    # undirected and the portfolio picked a challenger that lost ~20
    # composite points under directed scoring).
    direction_is_declared: bool = False
    reciprocal_edge_ratio: float = 0.0
    # Router-v2 (native-sprint r2) features. All are STRUCTURAL -- computed
    # from topology only, never from graph names or corpus identity -- and
    # size-gated so the classifier stays cheap on large graphs (defaults of
    # zero mean "not measured"; consumers treat that as "gate closed").
    #
    # degree_uniformity: stddev/mean of undirected degree. Lattice/mesh
    # interiors have near-constant degree (low values); hubs raise it.
    degree_uniformity: float = 0.0
    # hub_edge_fraction: fraction of edges incident to the top-5%-degree
    # nodes. In a degree-regular mesh this sits near ~0.1; scale-free tails
    # concentrate 0.5+ of all edges on their hubs.
    hub_edge_fraction: float = 0.0
    # diameter_estimate: double-sweep BFS lower bound on the diameter.
    # 2D meshes have diameter ~ 2*sqrt(N); small-world/SBM sit at ~ log N.
    # One of the sharpest lattice-vs-community separators at two-BFS cost.
    diameter_estimate: int = 0
    # community_score: undirected Newman modularity of a deterministic
    # label-propagation partition; num_communities: its community count.
    community_score: float = 0.0
    num_communities: int = 0
    # has_edge_weights: whether the source graph object carried edge weights
    # (populated only when a graph object is provided to classify_graph).
    has_edge_weights: bool = False


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


def _largest_component_fraction(edge_index: torch.Tensor, num_nodes: int) -> float:
    """Return the largest weak-component fraction of all nodes.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    float
        Fraction in ``[0, 1]`` covered by the largest weak component.
    """
    if num_nodes <= 0:
        return 0.0

    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    if num_edges == 0:
        return 1.0 / float(num_nodes)
    if num_edges > num_nodes - 1:
        # Dense benchmark/scale graphs are not component-pack candidates; avoid
        # a second Python union-find pass on the classifier hot path.
        return 1.0

    parents = list(range(num_nodes))
    ranks = [0] * num_nodes
    sizes = [1] * num_nodes
    cpu_edges = edge_index.detach().cpu()

    for source, target in zip(cpu_edges[0].tolist(), cpu_edges[1].tolist()):
        if source == target:
            continue

        source_root = _find_root(parents, source)
        target_root = _find_root(parents, target)
        if source_root == target_root:
            continue

        if ranks[source_root] < ranks[target_root]:
            parents[source_root] = target_root
            sizes[target_root] += sizes[source_root]
        elif ranks[source_root] > ranks[target_root]:
            parents[target_root] = source_root
            sizes[source_root] += sizes[target_root]
        else:
            parents[target_root] = source_root
            sizes[source_root] += sizes[target_root]
            ranks[source_root] += 1

    largest = 1
    for node in range(num_nodes):
        root = _find_root(parents, node)
        largest = max(largest, sizes[root])
    return float(largest) / float(num_nodes)


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


def _effective_layer_count(layer_assignments: Optional[torch.Tensor]) -> int:
    """Return the number of non-singleton layers in a layering.

    Parameters
    ----------
    layer_assignments : torch.Tensor, optional
        Layer assignments shaped ``[N]``.

    Returns
    -------
    int
        Count of layers with at least two nodes, or ``1`` for a non-empty
        all-singleton layering. This keeps chain-like artifacts from looking
        like meaningful layered DAGs to the native dispatcher.
    """
    if layer_assignments is None or layer_assignments.numel() == 0:
        return 0

    normalized = layer_assignments.detach().to(device="cpu", dtype=torch.long)
    normalized = normalized - int(normalized.min().item())
    counts = torch.bincount(normalized)
    wide_layers = int((counts >= 2).sum().item())
    if wide_layers > 0:
        return wide_layers
    return 1


def _cyclicity_ratio(edge_index: torch.Tensor, num_nodes: int) -> float:
    """Return the fraction of edges reversed by the feedback-arc heuristic.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    float
        Ratio ``|back_edges| / |E|`` after greedy FAS cycle breaking.
    """
    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    if num_edges == 0:
        return 0.0
    if num_nodes > 50_000 or num_edges > 200_000:
        return 0.0

    from dagua.layout.cycle import make_acyclic_robust

    _, reversed_mask = make_acyclic_robust(edge_index.detach().to(device="cpu"), num_nodes)
    return float(reversed_mask.sum().item()) / float(num_edges)


def _reciprocal_edge_ratio(edge_index: torch.Tensor) -> float:
    """Return the fraction of directed edges with a reverse counterpart.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.

    Returns
    -------
    float
        Ratio in ``[0, 1]`` where ``1`` means every directed edge also appears
        in the opposite direction.
    """
    if edge_index.numel() == 0:
        return 0.0

    cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    edge_pairs = list(zip(cpu_edges[0].tolist(), cpu_edges[1].tolist()))
    pair_set = set(edge_pairs)
    reciprocal_count = sum(1 for source, target in edge_pairs if (target, source) in pair_set)
    return float(reciprocal_count) / float(len(edge_pairs))


def _infer_semantically_directed(
    edge_index: torch.Tensor,
    num_nodes: int,
    layer_assignments: Optional[torch.Tensor] = None,
) -> bool:
    """Infer whether graph edges carry semantic direction.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    layer_assignments : torch.Tensor, optional
        Optional layer assignment tensor shaped ``[N]``. When omitted, the
        classifier computes longest-path layers for small graphs.

    Returns
    -------
    bool
        ``False`` for likely undirected-origin graphs that were mechanically
        oriented into a DAG, otherwise ``True``.
    """
    if num_nodes <= 0 or edge_index.numel() == 0:
        return True

    if num_nodes > 10_000_000:
        resolved_layers = (
            _resolve_layer_assignments(edge_index, num_nodes, layer_assignments)
            if layer_assignments is not None
            else None
        )
        num_layers, _, _, _ = _analyze_layers(resolved_layers, num_nodes)
        return not (num_layers > 0 and float(num_layers) / float(num_nodes) >= 0.4)

    num_edges = edge_index.shape[1]
    degree = _compute_degree(edge_index, num_nodes)
    max_degree = int(degree.max().item()) if degree.numel() > 0 else 0
    num_components = _count_components(edge_index, num_nodes)
    is_acyclic = _is_undirected_acyclic(edge_index, num_nodes)
    is_tree = num_components == 1 and is_acyclic and num_edges == num_nodes - 1
    degree_one_count = int((degree == 1).sum().item()) if degree.numel() > 0 else 0
    is_chain = is_tree and max_degree <= 2 and (num_nodes <= 2 or degree_one_count == 2)
    if is_tree or is_chain:
        return True

    if _reciprocal_edge_ratio(edge_index) > 0.3:
        return False

    resolved_layers = _resolve_layer_assignments(edge_index, num_nodes, layer_assignments)
    num_layers, _, _, _ = _analyze_layers(resolved_layers, num_nodes)
    if num_layers > 0 and float(num_layers) / float(num_nodes) >= 0.4:
        # Deep layering alone is ambiguous: mechanically-oriented undirected
        # graphs (e.g. index-oriented dense graphs) and genuinely deep
        # directed pipelines (e.g. a long chain of transformer blocks) can
        # both produce num_layers close to num_nodes. Disambiguate using the
        # fraction of edges whose layer span is exactly 1 (adjacent-layer):
        # deep *chains* of meaningful stages are mostly adjacent-layer edges,
        # while mechanically oriented graphs are not.
        if _adjacent_layer_edge_fraction(edge_index, resolved_layers) >= 0.6:
            return True
        return False

    return True


def _adjacent_layer_edge_fraction(
    edge_index: torch.Tensor,
    layer_assignments: Optional[torch.Tensor],
) -> float:
    """Return the fraction of edges whose layer span (target - source) equals 1.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    layer_assignments : torch.Tensor, optional
        Layer assignments shaped ``[N]``. When unavailable, returns ``0.0``
        (treated as non-adjacent, preserving the conservative prior behavior).

    Returns
    -------
    float
        Fraction of edges with ``layer[target] - layer[source] == 1``.
    """
    if layer_assignments is None or edge_index.numel() == 0:
        return 0.0
    sources = edge_index[0].to(layer_assignments.device)
    targets = edge_index[1].to(layer_assignments.device)
    layer_span = layer_assignments[targets] - layer_assignments[sources]
    return float((layer_span == 1).to(dtype=torch.float32).mean().item())


# Router-v2 feature gates. Features above these sizes report their "not
# measured" defaults; the native contest cap (1500) sits far below both, so
# routing consumers never see unmeasured features in their active range.
ROUTER_FEATURE_MAX_NODES = 5_000
ROUTER_FEATURE_MAX_EDGES = 100_000
_LABEL_PROPAGATION_ROUNDS = 10
_HUB_DEGREE_TOP_FRACTION = 0.05


def _dedup_undirected_edges(edge_index: torch.Tensor) -> torch.Tensor:
    """Return deduplicated undirected edges as a ``[2, M]`` CPU tensor.

    Self-loops are dropped and each undirected pair is kept once with
    ``source < target``.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.

    Returns
    -------
    torch.Tensor
        Canonical undirected edge tensor shaped ``[2, M]``.
    """
    if edge_index.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    lo = torch.minimum(edges[0], edges[1])
    hi = torch.maximum(edges[0], edges[1])
    keep = lo != hi
    lo, hi = lo[keep], hi[keep]
    if lo.numel() == 0:
        return torch.zeros((2, 0), dtype=torch.long)
    keys = torch.unique(lo * (int(hi.max().item()) + 1) + hi)
    stride = int(hi.max().item()) + 1
    return torch.stack([torch.div(keys, stride, rounding_mode="floor"), keys % stride])


def label_propagation_communities(
    edge_index: torch.Tensor,
    num_nodes: int,
    rounds: int = _LABEL_PROPAGATION_ROUNDS,
) -> torch.Tensor:
    """Return a deterministic label-propagation community partition.

    Synchronous label propagation with min-label tie-breaking: every node
    adopts the most frequent label among its (undirected) neighbors each
    round, ties resolved to the smallest label id. Deterministic by
    construction (no RNG), so the classifier and the community route see the
    same partition on every call. Bounded to ``rounds`` iterations because
    synchronous updates can oscillate on bipartite-like structures.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]`` (direction ignored).
    num_nodes : int
        Number of nodes.
    rounds : int, default=10
        Maximum propagation rounds.

    Returns
    -------
    torch.Tensor
        Community ids shaped ``[N]`` relabeled densely to ``0..K-1``.
    """
    if num_nodes <= 0:
        return torch.zeros((0,), dtype=torch.long)
    labels = torch.arange(num_nodes, dtype=torch.long)
    if edge_index.numel() == 0:
        return labels
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    keep = edges[0] != edges[1]
    edges = edges[:, keep]
    if edges.numel() == 0:
        return labels
    src = torch.cat([edges[0], edges[1]])
    dst = torch.cat([edges[1], edges[0]])
    for _round in range(max(int(rounds), 1)):
        neighbor_labels = labels[src]
        keys = dst * num_nodes + neighbor_labels
        unique_keys, counts = torch.unique(keys, return_counts=True)
        nodes = torch.div(unique_keys, num_nodes, rounding_mode="floor")
        node_labels = unique_keys % num_nodes
        # unique() output is sorted by (node asc, label asc); a stable sort
        # by count desc followed by a stable sort by node keeps, per node,
        # the max-count label with the smallest id first.
        by_count = torch.argsort(counts, descending=True, stable=True)
        nodes_by_count = nodes[by_count]
        labels_by_count = node_labels[by_count]
        by_node = torch.argsort(nodes_by_count, stable=True)
        grouped_nodes = nodes_by_count[by_node]
        grouped_labels = labels_by_count[by_node]
        first_in_group = torch.ones_like(grouped_nodes, dtype=torch.bool)
        first_in_group[1:] = grouped_nodes[1:] != grouped_nodes[:-1]
        new_labels = labels.clone()
        new_labels[grouped_nodes[first_in_group]] = grouped_labels[first_in_group]
        if torch.equal(new_labels, labels):
            break
        labels = new_labels
    _, dense = torch.unique(labels, return_inverse=True)
    return dense


def undirected_modularity(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    num_nodes: int,
) -> float:
    """Return the Newman modularity of a partition on the undirected graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]`` (direction ignored, dedup applied).
    labels : torch.Tensor
        Community ids shaped ``[N]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    float
        Modularity ``Q = sum_c (e_c/m - (d_c/2m)^2)`` in ``[-0.5, 1)``.
    """
    if num_nodes <= 0:
        return 0.0
    undirected = _dedup_undirected_edges(edge_index)
    edge_count = int(undirected.shape[1])
    if edge_count == 0:
        return 0.0
    labels = labels.detach().to(device="cpu", dtype=torch.long)
    num_labels = int(labels.max().item()) + 1
    degrees = torch.zeros(num_nodes, dtype=torch.float64)
    ones = torch.ones(edge_count, dtype=torch.float64)
    degrees.scatter_add_(0, undirected[0], ones)
    degrees.scatter_add_(0, undirected[1], ones)
    intra = labels[undirected[0]] == labels[undirected[1]]
    intra_per_label = torch.bincount(
        labels[undirected[0]][intra],
        minlength=num_labels,
    ).to(dtype=torch.float64)
    degree_per_label = torch.zeros(num_labels, dtype=torch.float64)
    degree_per_label.scatter_add_(0, labels, degrees)
    m = float(edge_count)
    quality = intra_per_label / m - (degree_per_label / (2.0 * m)) ** 2
    return float(quality.sum().item())


def _build_undirected_adjacency_lists(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> list[list[int]]:
    """Return simple undirected adjacency lists for BFS helpers.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        Neighbor lists (deduplicated, no self-loops).
    """
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    undirected = _dedup_undirected_edges(edge_index)
    for source, target in zip(undirected[0].tolist(), undirected[1].tolist()):
        adjacency[source].append(target)
        adjacency[target].append(source)
    return adjacency


def _bfs_farthest(adjacency: list[list[int]], start: int) -> tuple[int, int]:
    """Return ``(farthest_node, distance)`` from a BFS sweep.

    Parameters
    ----------
    adjacency : list[list[int]]
        Undirected neighbor lists.
    start : int
        BFS source node.

    Returns
    -------
    tuple[int, int]
        Farthest reached node and its distance from ``start``.
    """
    num_nodes = len(adjacency)
    distance = [-1] * num_nodes
    distance[start] = 0
    frontier = [start]
    farthest_node, farthest_distance = start, 0
    while frontier:
        next_frontier: list[int] = []
        for node in frontier:
            for neighbor in adjacency[node]:
                if distance[neighbor] < 0:
                    distance[neighbor] = distance[node] + 1
                    next_frontier.append(neighbor)
                    if distance[neighbor] > farthest_distance:
                        farthest_distance = distance[neighbor]
                        farthest_node = neighbor
        frontier = next_frontier
    return farthest_node, farthest_distance


def _double_sweep_diameter(edge_index: torch.Tensor, num_nodes: int) -> int:
    """Return a double-sweep BFS diameter lower bound.

    BFS from node 0, then BFS from the farthest node found. Exact on trees,
    a tight lower bound on general graphs -- sufficient for the router's
    "mesh-scale vs log-scale" separation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    int
        Estimated diameter of the largest reachable region from node 0.
    """
    if num_nodes <= 1 or edge_index.numel() == 0:
        return 0
    adjacency = _build_undirected_adjacency_lists(edge_index, num_nodes)
    far_node, _ = _bfs_farthest(adjacency, 0)
    _, diameter = _bfs_farthest(adjacency, far_node)
    return int(diameter)


def _router_v2_features(
    edge_index: torch.Tensor,
    num_nodes: int,
    degree: torch.Tensor,
) -> tuple[float, float, int, float, int]:
    """Return the router-v2 structural feature tuple.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes.
    degree : torch.Tensor
        Undirected degrees shaped ``[N]``.

    Returns
    -------
    tuple[float, float, int, float, int]
        ``(degree_uniformity, hub_edge_fraction, diameter_estimate,
        community_score, num_communities)``. Size-gated features report
        zero defaults above the router gates.
    """
    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    if num_nodes <= 0 or num_edges == 0:
        return 0.0, 0.0, 0, 0.0, 0

    degree_float = degree.to(dtype=torch.float32)
    mean_degree = float(degree_float.mean().item())
    degree_uniformity = (
        float(degree_float.std(unbiased=False).item()) / mean_degree if mean_degree > 0 else 0.0
    )

    top_count = max(int(num_nodes * _HUB_DEGREE_TOP_FRACTION), 1)
    hub_nodes = torch.topk(degree_float, k=min(top_count, num_nodes)).indices
    hub_mask = torch.zeros(num_nodes, dtype=torch.bool)
    hub_mask[hub_nodes] = True
    cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    incident = hub_mask[cpu_edges[0]] | hub_mask[cpu_edges[1]]
    hub_edge_fraction = float(incident.to(dtype=torch.float32).mean().item())

    if num_nodes > ROUTER_FEATURE_MAX_NODES or num_edges > ROUTER_FEATURE_MAX_EDGES:
        return degree_uniformity, hub_edge_fraction, 0, 0.0, 0

    diameter_estimate = _double_sweep_diameter(edge_index, num_nodes)
    community_labels = label_propagation_communities(edge_index, num_nodes)
    num_communities = int(community_labels.max().item()) + 1 if community_labels.numel() else 0
    community_score = undirected_modularity(edge_index, community_labels, num_nodes)
    return (
        degree_uniformity,
        hub_edge_fraction,
        diameter_estimate,
        community_score,
        num_communities,
    )


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
    graph: Optional[Any] = None,
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
    graph : Any, optional
        Optional graph object with ``is_semantically_directed`` set to override
        heuristic inference.

    Returns
    -------
    GraphStructure
        Detected structure with metadata.
    """
    num_edges = edge_index.shape[1] if edge_index.numel() > 0 else 0
    explicit_direction = (
        getattr(graph, "is_semantically_directed", None) if graph is not None else None
    )

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
        num_layers_effective = _effective_layer_count(resolved_layers)
        edge_to_node_ratio = float(num_edges) / float(num_nodes) if num_nodes > 0 else 0.0
        if explicit_direction is not None:
            is_semantically_directed = bool(explicit_direction)
        else:
            is_semantically_directed = _infer_semantically_directed(
                edge_index,
                num_nodes,
                resolved_layers,
            )
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
            is_semantically_directed=is_semantically_directed,
            num_layers_effective=num_layers_effective,
            cyclicity_ratio=0.0,
            has_dominant_component=True,
            direction_is_declared=explicit_direction is not None,
        )

    large_dense_fast_path = num_nodes > 50_000 and num_edges > num_nodes - 1
    if large_dense_fast_path:
        resolved_layers = (
            _resolve_layer_assignments(edge_index, num_nodes, layer_assignments)
            if layer_assignments is not None
            else None
        )
        num_layers, avg_layer_width, max_layer_width, layer_width_cv = _analyze_layers(
            resolved_layers,
            num_nodes,
        )
        num_layers_effective = _effective_layer_count(resolved_layers)
        edge_to_node_ratio = float(num_edges) / float(num_nodes) if num_nodes > 0 else 0.0
        is_planar_hint = (num_edges < 3 * num_nodes - 6) if num_nodes >= 3 else True
        is_semantically_directed = (
            bool(explicit_direction) if explicit_direction is not None else True
        )
        topology_tags = _derive_topology_tags(
            family=GraphFamily.GENERAL,
            num_nodes=num_nodes,
            num_layers=num_layers,
            max_layer_width=max_layer_width,
            max_degree=0,
            edge_to_node_ratio=edge_to_node_ratio,
            layer_width_cv=layer_width_cv,
            is_planar_hint=is_planar_hint,
            is_directed_acyclic=False,
        )
        return GraphStructure(
            family=GraphFamily.GENERAL,
            num_components=1,
            max_degree=0,
            num_layers=num_layers,
            avg_layer_width=avg_layer_width,
            is_planar_hint=is_planar_hint,
            is_acyclic=False,
            max_layer_width=max_layer_width,
            layer_width_cv=layer_width_cv,
            edge_to_node_ratio=edge_to_node_ratio,
            is_directed_acyclic=False,
            topology_tags=topology_tags,
            is_semantically_directed=is_semantically_directed,
            num_layers_effective=num_layers_effective,
            cyclicity_ratio=0.0,
            has_dominant_component=True,
            is_planar=is_planar_hint if num_nodes > 1500 else None,
            planar_embedding=None,
            direction_is_declared=explicit_direction is not None,
            reciprocal_edge_ratio=0.0,
        )

    degree = _compute_degree(edge_index, num_nodes)
    max_degree = int(degree.max().item()) if degree.numel() > 0 else 0
    num_components = _count_components(edge_index, num_nodes)
    largest_component_fraction = _largest_component_fraction(edge_index, num_nodes)
    has_dominant_component = largest_component_fraction >= 0.8
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
    num_layers_effective = _effective_layer_count(resolved_layers)
    is_bipartite_dag = num_layers == 2
    is_wide_layered = (
        num_layers > 0
        and avg_layer_width >= 100.0
        and float(num_layers) <= max(float(num_nodes) / 100.0, 1.0)
    )

    is_planar_hint = (num_edges < 3 * num_nodes - 6) if num_nodes >= 3 else True
    edge_to_node_ratio = float(num_edges) / float(num_nodes) if num_nodes > 0 else 0.0
    cyclicity_ratio = 0.0 if is_directed_acyclic else _cyclicity_ratio(edge_index, num_nodes)

    if explicit_direction is not None:
        is_semantically_directed = bool(explicit_direction)
    else:
        is_semantically_directed = _infer_semantically_directed(
            edge_index,
            num_nodes,
            resolved_layers,
        )

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
    elif not is_directed_acyclic:
        if cyclicity_ratio > 0.3 or is_semantically_directed is False:
            family = GraphFamily.FORCE_DIRECTED
        else:
            family = GraphFamily.HYBRID
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
    is_planar, planar_embedding = _check_exact_planarity(
        edge_index, num_nodes, is_planar_hint=is_planar_hint
    )
    (
        degree_uniformity,
        hub_edge_fraction,
        diameter_estimate,
        community_score,
        num_communities,
    ) = _router_v2_features(edge_index, num_nodes, degree)
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
        is_semantically_directed=is_semantically_directed,
        num_layers_effective=num_layers_effective,
        cyclicity_ratio=cyclicity_ratio,
        has_dominant_component=has_dominant_component,
        is_planar=is_planar,
        planar_embedding=planar_embedding,
        direction_is_declared=explicit_direction is not None,
        reciprocal_edge_ratio=(_reciprocal_edge_ratio(edge_index) if num_nodes <= 100_000 else 0.0),
        degree_uniformity=degree_uniformity,
        hub_edge_fraction=hub_edge_fraction,
        diameter_estimate=diameter_estimate,
        community_score=community_score,
        num_communities=num_communities,
        has_edge_weights=(graph is not None and getattr(graph, "edge_weights", None) is not None),
    )


def _check_exact_planarity(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    is_planar_hint: bool,
) -> tuple[Optional[bool], Any]:
    """Run an exact planarity check via networkx for graphs that pass the hint.

    Returns ``(is_planar, planar_embedding)``. For very large graphs the cheap
    Euler-formula hint stands in for the exact check (None if the hint says
    not planar; otherwise the hint value with no embedding).
    """
    if num_nodes <= 1 or edge_index.numel() == 0:
        return True, None
    if not is_planar_hint:
        return False, None
    if num_nodes > 1500:
        return is_planar_hint, None
    try:
        import networkx as nx  # type: ignore
    except Exception:
        return is_planar_hint, None
    g = nx.Graph()
    g.add_nodes_from(range(num_nodes))
    edges = edge_index.detach().cpu().t().tolist()
    g.add_edges_from((int(u), int(v)) for u, v in edges if int(u) != int(v))
    try:
        is_planar, embedding = nx.check_planarity(g, counterexample=False)
    except Exception:
        return is_planar_hint, None
    return bool(is_planar), embedding if is_planar else None
