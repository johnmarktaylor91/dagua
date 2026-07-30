"""Community-aware geodesic stress candidates for native undirected contests."""

from __future__ import annotations

from typing import Any, Mapping, Optional, Sequence

import torch

COMMUNITY_STRESS_INTER_SCALES: tuple[float, float] = (1.5, 2.2)
MAX_COMMUNITY_LABEL_FRACTION = 0.5
MIN_COMMUNITY_MODULARITY = 0.30
CNM_GAIN_EPSILON = 1.0e-12


def greedy_modularity_communities(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Return deterministic weighted CNM greedy-modularity communities.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``. Direction is ignored.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional nonnegative edge weights shaped ``[E]``. Parallel undirected
        edges are summed before agglomeration.

    Returns
    -------
    torch.Tensor
        Dense community labels shaped ``[N]``.
    """
    if num_nodes <= 0:
        return torch.zeros((0,), dtype=torch.long)
    edge_weight_by_pair = _dedup_weighted_undirected_edges(edge_index, num_nodes, edge_weights)
    if not edge_weight_by_pair:
        return torch.arange(num_nodes, dtype=torch.long)

    total_weight = float(sum(edge_weight_by_pair.values()))
    if total_weight <= 0.0:
        return torch.arange(num_nodes, dtype=torch.long)

    labels = torch.arange(num_nodes, dtype=torch.long)
    degrees = [0.0] * num_nodes
    for (source, target), weight in edge_weight_by_pair.items():
        degrees[source] += weight
        degrees[target] += weight

    while True:
        communities = sorted(set(int(label) for label in labels.tolist()))
        degree_by_community = {
            community: sum(
                degrees[node] for node in range(num_nodes) if int(labels[node]) == community
            )
            for community in communities
        }
        coupling: dict[tuple[int, int], float] = {}
        for (source, target), weight in edge_weight_by_pair.items():
            source_community = int(labels[source])
            target_community = int(labels[target])
            if source_community == target_community:
                continue
            lo = min(source_community, target_community)
            hi = max(source_community, target_community)
            coupling[(lo, hi)] = coupling.get((lo, hi), 0.0) + weight

        best_pair: Optional[tuple[int, int]] = None
        best_gain = 0.0
        total_weight_sq = total_weight * total_weight
        for pair in sorted(coupling):
            lo, hi = pair
            gain = coupling[pair] / total_weight - (
                degree_by_community[lo] * degree_by_community[hi]
            ) / (2.0 * total_weight_sq)
            if gain > best_gain + 0.0:
                best_gain = gain
                best_pair = pair

        if best_pair is None or best_gain <= CNM_GAIN_EPSILON:
            break
        survivor, merged = best_pair
        labels[labels == merged] = survivor
        _, labels = torch.unique(labels, sorted=True, return_inverse=True)

    _, dense = torch.unique(labels, sorted=True, return_inverse=True)
    return dense.to(dtype=torch.long)


def resolve_community_labels(problem: Any) -> Optional[torch.Tensor]:
    """Resolve declared or detected community labels for a layout problem.

    Parameters
    ----------
    problem : Any
        Layout problem carrying ``edge_index``, ``num_nodes``, optional
        ``clusters``, ``cluster_parents``, ``edge_weights``, and ``structure``.

    Returns
    -------
    torch.Tensor or None
        Dense labels shaped ``[N]`` when a meaningful partition is available.
    """
    num_nodes = int(getattr(problem, "num_nodes", 0))
    if num_nodes <= 0:
        return None
    clusters = getattr(problem, "clusters", None)
    if clusters:
        labels = _declared_leaf_cluster_labels(
            clusters=clusters,
            cluster_parents=getattr(problem, "cluster_parents", None),
            num_nodes=num_nodes,
        )
    else:
        edge_weights = getattr(problem, "edge_weights", None)
        if edge_weights is None and not _existing_label_propagation_gate_fired(problem):
            return None
        labels = greedy_modularity_communities(
            getattr(problem, "edge_index"),
            num_nodes,
            edge_weights=edge_weights,
        )
    return _admissible_labels_or_none(
        edge_index=getattr(problem, "edge_index"),
        labels=labels,
        num_nodes=num_nodes,
    )


def layout_community_stress_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    community_labels: torch.Tensor,
    inter_scale: float,
    node_sizes: Optional[torch.Tensor] = None,
    config: Optional[Any] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    steps: Optional[int] = None,
    node_sep: Optional[float] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run geodesic stress with inflated inter-community target distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]`` (direction ignored).
    num_nodes : int
        Number of nodes.
    community_labels : torch.Tensor
        Dense community labels shaped ``[N]``.
    inter_scale : float
        Multiplicative target-distance scale for cross-community pairs.
    node_sizes : torch.Tensor, optional
        Node bounding boxes shaped ``[N, 2]``.
    config : Any, optional
        Optional layout configuration carrying ``node_sep``.
    seed : int, default=42
        Deterministic seed used by the geodesic fallback.
    edge_weights : torch.Tensor, optional
        Optional per-edge distance costs shaped ``[E]``.
    steps : int, optional
        Explicit descent budget.
    node_sep : float, optional
        Node separation override in points.
    **kwargs : Any
        Compatibility keywords accepted by generic dispatchers.

    Returns
    -------
    torch.Tensor
        Finite positions shaped ``[N, 2]`` in point units.
    """
    del kwargs
    from dagua.layout.ops.pipelines.native_lattice_grid import _layout_geodesic_stress_core

    if inter_scale > max(COMMUNITY_STRESS_INTER_SCALES):
        raise ValueError("community stress inter_scale must not exceed 2.2.")
    labels = community_labels.detach().to(device="cpu", dtype=torch.long)
    if int(labels.numel()) != num_nodes:
        raise ValueError("community_labels must have shape [N].")

    def _inflate_intercommunity_distances(distances: torch.Tensor) -> torch.Tensor:
        """Return distances with cross-community pairs inflated.

        Parameters
        ----------
        distances : torch.Tensor
            Finite APSP distance matrix shaped ``[N, N]``.

        Returns
        -------
        torch.Tensor
            Transformed distance matrix shaped ``[N, N]``.
        """
        transformed = distances.clone()
        inter_mask = labels.unsqueeze(0) != labels.unsqueeze(1)
        transformed[inter_mask] = transformed[inter_mask] * float(inter_scale)
        return transformed

    return _layout_geodesic_stress_core(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        config=config,
        seed=seed,
        edge_weights=edge_weights,
        steps=steps,
        node_sep=node_sep,
        distance_transform=_inflate_intercommunity_distances,
    )


def _dedup_weighted_undirected_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> dict[tuple[int, int], float]:
    """Return undirected edge weights keyed by sorted endpoint pair.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights shaped ``[E]``.

    Returns
    -------
    dict[tuple[int, int], float]
        Summed positive weights per undirected non-self edge.
    """
    weighted_edges: dict[tuple[int, int], float] = {}
    if edge_index.numel() == 0:
        return weighted_edges
    cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    weights = (
        torch.ones(int(cpu_edges.shape[1]), dtype=torch.float64)
        if edge_weights is None
        else edge_weights.detach().to(device="cpu", dtype=torch.float64)
    )
    for source, target, weight in zip(
        cpu_edges[0].tolist(),
        cpu_edges[1].tolist(),
        weights.tolist(),
    ):
        source_i = int(source)
        target_i = int(target)
        if source_i == target_i or source_i < 0 or target_i < 0:
            continue
        if source_i >= num_nodes or target_i >= num_nodes:
            continue
        lo = min(source_i, target_i)
        hi = max(source_i, target_i)
        weighted_edges[(lo, hi)] = weighted_edges.get((lo, hi), 0.0) + max(float(weight), 0.0)
    return {pair: weight for pair, weight in weighted_edges.items() if weight > 0.0}


def _declared_leaf_cluster_labels(
    clusters: Mapping[str, Any],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    num_nodes: int,
) -> torch.Tensor:
    """Return labels from declared leaf-most cluster membership.

    Parameters
    ----------
    clusters : Mapping[str, Any]
        Declared cluster membership mapping.
    cluster_parents : Mapping[str, str | None], optional
        Optional cluster hierarchy mapping.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Dense labels shaped ``[N]`` with non-clustered nodes as singletons.
    """
    from dagua.layout.ops.cluster_geometry import ClusterTree
    from dagua.utils import collect_cluster_leaves

    labels = torch.full((num_nodes,), -1, dtype=torch.long)
    if cluster_parents:
        normalized = {
            str(name): _cluster_member_sequence(members) for name, members in clusters.items()
        }
        tree = ClusterTree.from_flat_membership(normalized, cluster_parents)
        cluster_members = {
            name: sorted(members)
            for name, members in tree.leaves_per_cluster.items()
            if len(members) > 0
        }
    else:
        cluster_members = {
            str(name): sorted(int(index) for index in collect_cluster_leaves(members))
            for name, members in clusters.items()
        }

    next_label = 0
    for name in sorted(cluster_members):
        assigned = False
        for index in cluster_members[name]:
            if 0 <= int(index) < num_nodes:
                labels[int(index)] = next_label
                assigned = True
        if assigned:
            next_label += 1
    for index in range(num_nodes):
        if int(labels[index]) < 0:
            labels[index] = next_label
            next_label += 1
    _, dense = torch.unique(labels, sorted=True, return_inverse=True)
    return dense.to(dtype=torch.long)


def _cluster_member_sequence(members: Any) -> Sequence[int]:
    """Return integer leaves from a cluster membership payload.

    Parameters
    ----------
    members : Any
        Cluster membership value, either flat or nested.

    Returns
    -------
    Sequence[int]
        Leaf node indices.
    """
    from dagua.utils import collect_cluster_leaves

    if isinstance(members, dict):
        return tuple(int(index) for index in collect_cluster_leaves(members))
    return tuple(int(index) for index in members)


def _existing_label_propagation_gate_fired(problem: Any) -> bool:
    """Return whether the pre-existing community shortlist gate fired.

    Parameters
    ----------
    problem : Any
        Layout problem carrying optional ``structure``.

    Returns
    -------
    bool
        ``True`` when the old label-propagation community features pass.
    """
    structure = getattr(problem, "structure", None)
    if structure is None:
        return False
    num_nodes = int(getattr(problem, "num_nodes", 0))
    num_communities = int(getattr(structure, "num_communities", 0))
    return (
        float(getattr(structure, "community_score", 0.0)) >= MIN_COMMUNITY_MODULARITY
        and 2 <= num_communities <= MAX_COMMUNITY_LABEL_FRACTION * num_nodes
    )


def _admissible_labels_or_none(
    edge_index: torch.Tensor,
    labels: torch.Tensor,
    num_nodes: int,
) -> Optional[torch.Tensor]:
    """Return labels only when they pass community-quality guards.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor shaped ``[2, E]``.
    labels : torch.Tensor
        Candidate labels shaped ``[N]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor or None
        Dense labels or ``None`` when the partition is not meaningful.
    """
    from dagua.layout.graph_classify import undirected_modularity

    if int(labels.numel()) != num_nodes:
        return None
    _, dense = torch.unique(
        labels.detach().to(device="cpu", dtype=torch.long),
        sorted=True,
        return_inverse=True,
    )
    num_communities = int(dense.max().item()) + 1 if dense.numel() else 0
    if num_communities < 2 or num_communities > MAX_COMMUNITY_LABEL_FRACTION * num_nodes:
        return None
    if undirected_modularity(edge_index, dense, num_nodes) < MIN_COMMUNITY_MODULARITY:
        return None
    return dense.to(dtype=torch.long)


__all__ = [
    "COMMUNITY_STRESS_INTER_SCALES",
    "greedy_modularity_communities",
    "layout_community_stress_pipeline",
    "resolve_community_labels",
]
