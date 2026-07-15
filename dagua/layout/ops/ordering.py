"""Ordering operations for layered graph layouts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

from dagua.layout.init_placement import _transpose_heuristic
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


def _target_device(problem: LayoutProblem, state: SolveState) -> torch.device:
    """Resolve the device used for persisted ordering tensors.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.device
        Device for the updated ``state.ordering`` tensor.
    """
    if state.ordering is not None:
        return state.ordering.device
    if state.layers is not None:
        return state.layers.device
    return problem.edge_index.device


def _validate_layers(layers: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Return validated CPU layer assignments.

    Parameters
    ----------
    layers : torch.Tensor | None
        Candidate layer tensor with shape ``[N]``.
    num_nodes : int
        Expected node count.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[N]``.

    Raises
    ------
    ValueError
        If ``layers`` is missing or has an invalid shape.
    """
    if layers is None:
        raise ValueError("state.layers must be populated before ordering ops run")
    if layers.ndim != 1 or layers.shape[0] != num_nodes:
        raise ValueError(f"state.layers must have shape [{num_nodes}]")

    layers_cpu = layers.detach().to(device="cpu", dtype=torch.long)
    if layers_cpu.numel() > 0 and int(layers_cpu.min().item()) < 0:
        raise ValueError("state.layers cannot contain negative indices")
    return layers_cpu


def _validate_ordering(ordering: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Return validated CPU ordering indices.

    Parameters
    ----------
    ordering : torch.Tensor | None
        Candidate ordering tensor with shape ``[N]``.
    num_nodes : int
        Expected node count.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[N]``.

    Raises
    ------
    ValueError
        If ``ordering`` is missing or has an invalid shape.
    """
    if ordering is None:
        raise ValueError("state.ordering must be populated before this op runs")
    if ordering.ndim != 1 or ordering.shape[0] != num_nodes:
        raise ValueError(f"state.ordering must have shape [{num_nodes}]")
    return ordering.detach().to(device="cpu", dtype=torch.long)


def _validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return a validated CPU edge list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor expected to have shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[2, E]``.

    Raises
    ------
    ValueError
        If the edge tensor shape or node references are invalid.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("problem.edge_index must have shape [2, E]")
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() == 0:
        return edge_index_cpu
    if int(edge_index_cpu.min().item()) < 0 or int(edge_index_cpu.max().item()) >= num_nodes:
        raise ValueError("problem.edge_index references a node outside the valid range")
    return edge_index_cpu


def _expanded_layers_to_tensor(layers: Any, num_nodes: int) -> torch.Tensor:
    """Return per-node layer assignments from expanded-graph layer groups.

    Parameters
    ----------
    layers : Any
        Expanded layer data, either a tensor with shape ``[N]`` or a sequence
        of per-layer node-id sequences.
    num_nodes : int
        Expected node count for the expanded graph.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[N]``.

    Raises
    ------
    ValueError
        If the expanded layer data cannot be interpreted safely.
    """
    if isinstance(layers, torch.Tensor):
        return _validate_layers(layers, num_nodes)

    layer_tensor = torch.full((num_nodes,), -1, dtype=torch.long)
    try:
        for layer_index, layer_nodes in enumerate(layers):
            for node in layer_nodes:
                node_index = int(node)
                if not 0 <= node_index < num_nodes:
                    raise ValueError("expanded_graph.layers references an invalid node")
                layer_tensor[node_index] = int(layer_index)
    except TypeError as exc:
        raise ValueError("expanded_graph.layers must be a tensor or layer-node sequence") from exc

    if num_nodes > 0 and int(layer_tensor.min().item()) < 0:
        raise ValueError("expanded_graph.layers must assign every expanded node")
    return layer_tensor


def _solve_state_matches_node_count(state: SolveState, num_nodes: int) -> bool:
    """Return whether the active solve state has tensors for ``num_nodes``.

    Parameters
    ----------
    state : SolveState
        Mutable solve state that may carry original-node or expanded-node
        tensors.
    num_nodes : int
        Candidate active graph node count.

    Returns
    -------
    bool
        ``True`` when position or ordering tensors are already sized for the
        candidate graph.
    """
    if state.pos is not None and state.pos.ndim >= 1 and int(state.pos.shape[0]) == num_nodes:
        return True
    return (
        state.ordering is not None
        and state.ordering.ndim == 1
        and int(state.ordering.shape[0]) == num_nodes
    )


def _resolve_active_layered_graph(
    problem: LayoutProblem,
    state: SolveState,
) -> Tuple[torch.Tensor, torch.Tensor, int]:
    """Return the layered graph that ordering ops should use.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs for the original graph.
    state : SolveState
        Mutable solve state, optionally carrying ``extras["expanded_graph"]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, int]
        ``(edge_index, layers, num_nodes)`` on CPU. The expanded graph is used
        only when the active state tensors already match its node count.
    """
    expanded_graph = state.extras.get("expanded_graph")
    if expanded_graph is not None:
        expanded_edge_index = getattr(expanded_graph, "edge_index", None)
        expanded_layers = getattr(expanded_graph, "layers", None)
        expanded_num_nodes = getattr(expanded_graph, "num_nodes", None)
        if (
            isinstance(expanded_edge_index, torch.Tensor)
            and expanded_layers is not None
            and expanded_num_nodes is not None
        ):
            num_expanded = int(expanded_num_nodes)
            if _solve_state_matches_node_count(state, num_expanded):
                return (
                    _validate_edge_index(expanded_edge_index, num_expanded),
                    _expanded_layers_to_tensor(expanded_layers, num_expanded),
                    num_expanded,
                )

    num_nodes = int(problem.num_nodes)
    return (
        _validate_edge_index(problem.edge_index, num_nodes),
        _validate_layers(state.layers, num_nodes),
        num_nodes,
    )


def _resolve_initial_ordering(
    layers_cpu: torch.Tensor,
    state: SolveState,
    num_nodes: int,
) -> Optional[torch.Tensor]:
    """Use existing ordering when present, otherwise derive it from current x.

    Parameters
    ----------
    layers_cpu : torch.Tensor
        CPU layer assignment tensor with shape ``[N]``.
    state : SolveState
        Mutable solve state carrying optional ordering and positions.
    num_nodes : int
        Active graph node count.

    Returns
    -------
    torch.Tensor | None
        CPU long in-layer rank tensor with shape ``[N]`` when an ordering can
        be resolved, otherwise ``None``.
    """
    if (
        state.ordering is not None
        and state.ordering.ndim == 1
        and state.ordering.shape[0] == num_nodes
    ):
        return _validate_ordering(state.ordering, num_nodes)

    if state.pos is None or state.pos.ndim != 2 or state.pos.shape[0] != num_nodes:
        return None

    pos_x = state.pos.detach().to(device="cpu")[:, 0]
    ordered_layers = _initial_ordered_layers(layers_cpu)
    for layer_nodes in ordered_layers:
        layer_nodes.sort(key=lambda node_idx: (float(pos_x[node_idx].item()), node_idx))
    return _ordering_from_layers(
        ordered_layers=ordered_layers,
        num_nodes=num_nodes,
        device=torch.device("cpu"),
    )


def _apply_ordering_to_positions(
    state: SolveState,
    layers_cpu: torch.Tensor,
    ordering_cpu: torch.Tensor,
    num_nodes: int,
) -> None:
    """Project the resolved ordering back onto x coordinates.

    Parameters
    ----------
    state : SolveState
        Mutable solve state whose ``pos`` may be updated in place.
    layers_cpu : torch.Tensor
        CPU layer assignment tensor with shape ``[N]``.
    ordering_cpu : torch.Tensor
        CPU long ordering tensor with shape ``[N]``.
    num_nodes : int
        Active graph node count.

    Returns
    -------
    None
        ``state.pos`` is replaced when its node count matches ``num_nodes``.
    """
    if state.pos is None or state.pos.ndim != 2 or state.pos.shape[0] != num_nodes:
        return

    pos = state.pos
    pos_new = pos.detach().clone()
    ordered_layers = _initial_ordered_layers(layers_cpu)
    if ordered_layers and all(len(layer_nodes) <= 1 for layer_nodes in ordered_layers):
        # Longest-path DAGs such as dense_pair_50 can put every node in its
        # own layer. There is no same-layer permutation to express, so collapse
        # x to the median slot to remove arbitrary zig-zag crossings while
        # preserving the solved y coordinates.
        pos_new[:, 0] = torch.median(pos_new[:, 0])
        state.pos = pos_new.detach().requires_grad_(pos.requires_grad)
        return

    for layer_nodes in ordered_layers:
        if len(layer_nodes) < 2:
            continue
        members = torch.tensor(layer_nodes, dtype=torch.long, device=pos_new.device)
        order_values = ordering_cpu[members.cpu()]
        ordered_members_cpu = members.cpu()[torch.argsort(order_values, stable=True)]
        ordered_members = ordered_members_cpu.to(device=pos_new.device)
        sorted_x, _ = torch.sort(pos_new[members, 0])
        pos_new[ordered_members, 0] = sorted_x

    state.pos = pos_new.detach().requires_grad_(pos.requires_grad)


def _adjacency_as_weighted_lists(adjacency: Any, num_nodes: int) -> List[List[Tuple[int, float]]]:
    """Normalize multiple adjacency formats into weighted Python lists.

    Parameters
    ----------
    adjacency : Any
        Cached adjacency from ``BuildAdjacency``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[tuple[int, float]]]
        Weighted adjacency lists indexed by node id.

    Raises
    ------
    ValueError
        If the adjacency format is unsupported for ordering sweeps.
    """
    if adjacency is None:
        raise ValueError("state.adjacency must be populated before sweep ordering ops run")

    weighted: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]

    if isinstance(adjacency, list):
        if len(adjacency) != num_nodes:
            raise ValueError(f"state.adjacency must contain {num_nodes} rows")
        for node, neighbors in enumerate(adjacency):
            for entry in neighbors:
                if isinstance(entry, tuple):
                    if len(entry) != 2:
                        raise ValueError(
                            "adjacency tuple entries must have shape (neighbor, weight)"
                        )
                    neighbor, weight = entry
                else:
                    neighbor, weight = entry, 1.0
                weighted[node].append((int(neighbor), float(weight)))
        return weighted

    if isinstance(adjacency, torch.Tensor):
        if adjacency.ndim != 2 or adjacency.shape != (num_nodes, num_nodes):
            raise ValueError(f"dense adjacency must have shape [{num_nodes}, {num_nodes}]")
        dense_cpu = adjacency.detach().to(device="cpu", dtype=torch.float32)
        for node in range(num_nodes):
            for neighbor in range(num_nodes):
                if node == neighbor:
                    continue
                value = float(dense_cpu[node, neighbor].item())
                if not torch.isfinite(dense_cpu[node, neighbor]) or value == 0.0:
                    continue
                weighted[node].append((neighbor, value))
        return weighted

    if isinstance(adjacency, dict):
        indptr = adjacency.get("indptr")
        indices = adjacency.get("indices")
        data = adjacency.get("data", adjacency.get("weights"))
        if indptr is None or indices is None:
            raise ValueError("CSR adjacency must include indptr and indices")

        indptr_cpu = indptr.detach().to(device="cpu", dtype=torch.long)
        indices_cpu = indices.detach().to(device="cpu", dtype=torch.long)
        data_cpu = None
        if data is not None:
            data_cpu = data.detach().to(device="cpu", dtype=torch.float32)

        if indptr_cpu.numel() != num_nodes + 1:
            raise ValueError(f"CSR indptr must have length {num_nodes + 1}")
        for node in range(num_nodes):
            start = int(indptr_cpu[node].item())
            stop = int(indptr_cpu[node + 1].item())
            for offset in range(start, stop):
                neighbor = int(indices_cpu[offset].item())
                weight = float(data_cpu[offset].item()) if data_cpu is not None else 1.0
                weighted[node].append((neighbor, weight))
        return weighted

    raise ValueError("Unsupported adjacency format for ordering ops")


def _initial_ordered_layers(
    layers: torch.Tensor,
    ordering: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    """Build ordered layer lists from per-node layer and ordering tensors.

    Parameters
    ----------
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    ordering : torch.Tensor | None, optional
        CPU in-layer ordering values with shape ``[N]``.

    Returns
    -------
    list[list[int]]
        Node ids grouped by layer and sorted left-to-right.
    """
    if layers.numel() == 0:
        return []

    num_layers = int(layers.max().item()) + 1
    ordered_layers: List[List[int]] = [[] for _ in range(num_layers)]
    for node in range(layers.shape[0]):
        ordered_layers[int(layers[node].item())].append(node)

    for layer_nodes in ordered_layers:
        if ordering is None:
            layer_nodes.sort()
        else:
            layer_nodes.sort(key=lambda node_idx: (int(ordering[node_idx].item()), node_idx))
    return ordered_layers


def _ordering_from_layers(
    ordered_layers: Sequence[Sequence[int]],
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert ordered per-layer node lists into a per-node rank tensor.

    Parameters
    ----------
    ordered_layers : sequence[sequence[int]]
        Nodes grouped by layer in left-to-right order.
    num_nodes : int
        Number of graph nodes.
    device : torch.device
        Target device for the result tensor.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``[N]``.
    """
    ordering = torch.zeros((num_nodes,), dtype=torch.long)
    for layer_nodes in ordered_layers:
        for position, node in enumerate(layer_nodes):
            ordering[node] = position
    return ordering.to(device=device)


def _node_order_map(layers: Sequence[Sequence[int]]) -> Dict[int, float]:
    """Map each node id to its current in-layer position.

    Parameters
    ----------
    layers : sequence[sequence[int]]
        Ordered nodes per layer.

    Returns
    -------
    dict[int, float]
        Current left-to-right rank for every node.
    """
    order_map: Dict[int, float] = {}
    for layer_nodes in layers:
        for position, node in enumerate(layer_nodes):
            order_map[node] = float(position)
    return order_map


def _layered_neighbors_from_adjacency(
    adjacency: List[List[Tuple[int, float]]],
    layers: torch.Tensor,
    use_weights: bool,
) -> Tuple[List[List[int]], List[List[int]], List[Dict[int, float]], List[Dict[int, float]]]:
    """Split adjacency into lower-layer parents and upper-layer children.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Weighted adjacency lists indexed by node id.
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    use_weights : bool
        Whether edge weights should affect barycenter computations.

    Returns
    -------
    tuple
        ``(parents, children, parent_weights, child_weights)`` indexed by node id.
    """
    num_nodes = layers.shape[0]
    parents: List[List[int]] = [[] for _ in range(num_nodes)]
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    parent_weights: List[Dict[int, float]] = [dict() for _ in range(num_nodes)]
    child_weights: List[Dict[int, float]] = [dict() for _ in range(num_nodes)]

    for node, neighbors in enumerate(adjacency):
        node_layer = int(layers[node].item())
        for neighbor, raw_weight in neighbors:
            if not 0 <= neighbor < num_nodes:
                continue
            neighbor_layer = int(layers[neighbor].item())
            weight = float(raw_weight) if use_weights else 1.0
            if neighbor_layer < node_layer:
                if neighbor not in parent_weights[node]:
                    parents[node].append(neighbor)
                    parent_weights[node][neighbor] = 0.0
                parent_weights[node][neighbor] += weight
            elif neighbor_layer > node_layer:
                if neighbor not in child_weights[node]:
                    children[node].append(neighbor)
                    child_weights[node][neighbor] = 0.0
                child_weights[node][neighbor] += weight
    return parents, children, parent_weights, child_weights


def _layered_neighbors_from_edges(
    edge_index: torch.Tensor,
    layers: torch.Tensor,
    num_nodes: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Orient edges according to the layer assignment.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple
        ``(parents, children)`` indexed by node id.
    """
    parents: List[List[int]] = [[] for _ in range(num_nodes)]
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return parents, children

    for source, target in edge_index.t().tolist():
        source_layer = int(layers[source].item())
        target_layer = int(layers[target].item())
        if source_layer == target_layer:
            continue
        if source_layer < target_layer:
            if source not in parents[target]:
                parents[target].append(source)
            if target not in children[source]:
                children[source].append(target)
            continue
        if target not in parents[source]:
            parents[source].append(target)
        if source not in children[target]:
            children[target].append(source)
    return parents, children


def _barycenter_value(
    node: int,
    neighbors_by_node: Sequence[Sequence[int]],
    neighbor_weights_by_node: Sequence[Dict[int, float]],
    order_map: Dict[int, float],
) -> float:
    """Compute the weighted barycenter score for one node.

    Parameters
    ----------
    node : int
        Node being ranked.
    neighbors_by_node : sequence[sequence[int]]
        Layer-compatible neighbors used as references.
    neighbor_weights_by_node : sequence[dict[int, float]]
        Optional neighbor weights aligned with ``neighbors_by_node``.
    order_map : dict[int, float]
        Current in-layer rank for each node.

    Returns
    -------
    float
        Weighted barycenter score, or the current node rank when isolated.
    """
    neighbors = neighbors_by_node[node]
    if not neighbors:
        return order_map.get(node, 0.0)

    weighted_sum = 0.0
    total_weight = 0.0
    for neighbor in neighbors:
        weight = neighbor_weights_by_node[node].get(neighbor, 1.0)
        weighted_sum += order_map[neighbor] * weight
        total_weight += weight
    if total_weight == 0.0:
        return order_map.get(node, 0.0)
    return weighted_sum / total_weight


def _median_value(
    node: int,
    neighbors_by_node: Sequence[Sequence[int]],
    order_map: Dict[int, float],
) -> float:
    """Compute the median neighbor rank for one node.

    Parameters
    ----------
    node : int
        Node being ranked.
    neighbors_by_node : sequence[sequence[int]]
        Layer-compatible neighbors used as references.
    order_map : dict[int, float]
        Current in-layer rank for each node.

    Returns
    -------
    float
        Median neighbor rank, or the current node rank when isolated.
    """
    neighbors = neighbors_by_node[node]
    if not neighbors:
        return order_map.get(node, 0.0)

    values = sorted(order_map[neighbor] for neighbor in neighbors)
    midpoint = len(values) // 2
    if len(values) % 2 == 1:
        return values[midpoint]
    return 0.5 * (values[midpoint - 1] + values[midpoint])


def _sort_layer_by_scores(
    layer_nodes: List[int],
    score_map: Dict[int, float],
) -> None:
    """Sort one layer by computed scores with stable tie handling.

    Parameters
    ----------
    layer_nodes : list[int]
        Nodes in one layer.
    score_map : dict[int, float]
        Ordering scores keyed by node id.

    Returns
    -------
    None
        The layer is sorted in place.
    """
    stable_positions = {node: index for index, node in enumerate(layer_nodes)}
    layer_nodes.sort(key=lambda node: (score_map[node], stable_positions[node], node))


def _deterministic_fiedler_vector(
    edge_index: torch.Tensor,
    num_nodes: int,
    max_iterations: int,
) -> Optional[torch.Tensor]:
    """Compute a deterministic Fiedler vector using ``torch.lobpcg``.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    max_iterations : int
        Maximum ``torch.lobpcg`` iteration count.

    Returns
    -------
    torch.Tensor | None
        CPU float tensor with shape ``[N]`` when the eigensolve succeeds,
        otherwise ``None``.
    """
    if num_nodes < 2 or edge_index.numel() == 0:
        return None

    all_sources = torch.cat([edge_index[0], edge_index[1]])
    all_targets = torch.cat([edge_index[1], edge_index[0]])
    degree = torch.zeros((num_nodes,), dtype=torch.float32)
    degree.scatter_add_(0, all_sources, torch.ones_like(all_sources, dtype=torch.float32))

    diagonal = torch.arange(num_nodes, dtype=torch.long)
    indices = torch.cat(
        [
            torch.stack([all_sources, all_targets], dim=0),
            torch.stack([diagonal, diagonal], dim=0),
        ],
        dim=1,
    )
    values = torch.cat(
        [
            -torch.ones((all_sources.shape[0],), dtype=torch.float32),
            degree,
        ]
    )
    laplacian = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    # Use a deterministic initial subspace so the op is reproducible without RNG state.
    baseline = torch.linspace(-1.0, 1.0, steps=num_nodes, dtype=torch.float32)
    harmonic = torch.cos(torch.arange(num_nodes, dtype=torch.float32))
    initial_guess = torch.stack([baseline, harmonic], dim=1)

    try:
        _, eigenvectors = torch.lobpcg(
            laplacian,
            k=2,
            X=initial_guess,
            largest=False,
            niter=max_iterations,
        )
    except Exception:
        return None
    return eigenvectors[:, 1].detach().to(device="cpu", dtype=torch.float32)


@dataclass(frozen=True)
class BarycenterSweepConfig:
    """Configuration for :class:`BarycenterSweep`.

    Parameters
    ----------
    passes : int, default=24
        Number of full sweep passes to perform.
    direction : str, default="both"
        Sweep direction: ``"down"``, ``"up"``, or ``"both"``.
    use_weights : bool, default=True
        Whether adjacency weights should influence barycenter scores.
    """

    passes: int = 24
    direction: str = "both"
    use_weights: bool = True


@dataclass(frozen=True)
class MedianSweepConfig:
    """Configuration for :class:`MedianSweep`.

    Parameters
    ----------
    passes : int, default=24
        Number of full sweep passes to perform.
    """

    passes: int = 24


@dataclass(frozen=True)
class TransposeHeuristicConfig:
    """Configuration for :class:`TransposeHeuristic`.

    Parameters
    ----------
    passes : int, default=8
        Maximum number of local swap passes.
    """

    passes: int = 8


@dataclass(frozen=True)
class SpectralOrderConfig:
    """Configuration for :class:`SpectralOrder`.

    Parameters
    ----------
    max_iterations : int, default=32
        Maximum ``torch.lobpcg`` iterations used for the Fiedler-vector solve.
    """

    max_iterations: int = 32


@dataclass(frozen=True)
class ClusterContiguousOrderConfig:
    """Configuration for :class:`ClusterContiguousOrder`.

    Parameters
    ----------
    enabled : bool, default=True
        Master switch for the class-predicated cluster ordering pass.
    min_clusters : int, default=2
        Minimum number of valid clusters required before ordering changes.
    """

    enabled: bool = True
    min_clusters: int = 2


def _valid_cluster_members(
    clusters: Optional[Mapping[str, Sequence[int]]],
    num_nodes: int,
) -> Dict[str, frozenset[int]]:
    """Return valid integer members for each non-empty cluster.

    Parameters
    ----------
    clusters : mapping[str, sequence[int]] | None
        Cluster metadata from the layout problem.
    num_nodes : int
        Number of active original nodes.

    Returns
    -------
    dict[str, frozenset[int]]
        Cluster names mapped to in-range node ids.
    """
    if not clusters:
        return {}

    valid: Dict[str, frozenset[int]] = {}
    for name, members in clusters.items():
        indices: set[int] = set()
        for member in members:
            node = int(member)
            if 0 <= node < num_nodes:
                indices.add(node)
        if indices:
            valid[str(name)] = frozenset(indices)
    return valid


def _passes_cluster_order_structure_gate(problem: LayoutProblem, cluster_count: int) -> bool:
    """Return whether cluster ordering should run for this graph class.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs.
    cluster_count : int
        Count of valid non-empty clusters.

    Returns
    -------
    bool
        ``True`` for small directed-acyclic multi-cluster layered DAGs.
    """
    structure = problem.structure
    if structure is not None and not bool(
        getattr(structure, "is_directed_acyclic", getattr(structure, "is_acyclic", True))
    ):
        return False
    return cluster_count >= 5 and problem.num_nodes <= 120


def _cluster_children(
    cluster_names: Sequence[str],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
) -> Dict[Optional[str], List[str]]:
    """Group cluster names by parent.

    Parameters
    ----------
    cluster_names : sequence[str]
        Valid cluster names.
    cluster_parents : mapping[str, str | None] | None
        Optional nested-cluster parent metadata.

    Returns
    -------
    dict[str | None, list[str]]
        Parent name to child cluster names. Invalid parents are treated as
        roots.
    """
    names = set(cluster_names)
    children: Dict[Optional[str], List[str]] = {None: []}
    for name in cluster_names:
        raw_parent = cluster_parents.get(name) if cluster_parents is not None else None
        parent = str(raw_parent) if raw_parent in names else None
        children.setdefault(parent, []).append(name)
        children.setdefault(name, [])
    return children


def _stable_cluster_position(
    nodes: Sequence[int],
    positions: Mapping[int, int],
) -> int:
    """Return the current leftmost position for a cluster-like node set.

    Parameters
    ----------
    nodes : sequence[int]
        Node ids represented by the cluster or singleton group.
    positions : mapping[int, int]
        Current in-layer positions keyed by node id.

    Returns
    -------
    int
        Minimum current position, or a large sentinel for empty groups.
    """
    present = [positions[node] for node in nodes if node in positions]
    if not present:
        return 10**12
    return min(present)


def _cluster_order_for_layer(
    layer_nodes: Sequence[int],
    cluster_members: Mapping[str, frozenset[int]],
    children_by_parent: Mapping[Optional[str], Sequence[str]],
) -> List[int]:
    """Return one layer ordered with nested cluster subtrees contiguous.

    Parameters
    ----------
    layer_nodes : sequence[int]
        Current nodes in a single layer.
    cluster_members : mapping[str, frozenset[int]]
        Valid cluster membership sets.
    children_by_parent : mapping[str | None, sequence[str]]
        Nested cluster child mapping.

    Returns
    -------
    list[int]
        Reordered layer nodes. Nodes inside a cluster subtree are emitted
        contiguously while preserving current left-to-right sibling order.
    """
    layer_set = set(layer_nodes)
    positions = {node: index for index, node in enumerate(layer_nodes)}

    def emit_cluster(cluster_name: str) -> List[int]:
        """Emit direct leaves and child clusters for one cluster.

        Parameters
        ----------
        cluster_name : str
            Cluster whose layer-local members should be ordered.

        Returns
        -------
        list[int]
            Layer-local nodes in stable nested-cluster order.
        """
        child_names = list(children_by_parent.get(cluster_name, ()))
        child_union: set[int] = set()
        for child_name in child_names:
            child_union.update(cluster_members.get(child_name, frozenset()))
        direct_nodes = [
            node
            for node in layer_nodes
            if node in cluster_members[cluster_name] and node not in child_union
        ]
        groups: List[Tuple[int, List[int]]] = []
        if direct_nodes:
            groups.append((_stable_cluster_position(direct_nodes, positions), direct_nodes))
        for child_name in child_names:
            child_nodes = emit_cluster(child_name)
            if child_nodes:
                groups.append((_stable_cluster_position(child_nodes, positions), child_nodes))
        groups.sort(key=lambda item: item[0])
        return [node for _, nodes in groups for node in nodes]

    root_names = list(children_by_parent.get(None, ()))
    clustered_nodes: set[int] = set()
    top_groups: List[Tuple[int, List[int]]] = []
    for root_name in root_names:
        root_nodes = emit_cluster(root_name)
        if root_nodes:
            clustered_nodes.update(root_nodes)
            top_groups.append((_stable_cluster_position(root_nodes, positions), root_nodes))

    for node in layer_nodes:
        if node not in clustered_nodes and node in layer_set:
            top_groups.append((positions[node], [node]))
    top_groups.sort(key=lambda item: item[0])
    return [node for _, nodes in top_groups for node in nodes]


@register_op
class ClusterContiguousOrder(Op):
    """Make same-layer cluster members contiguous before coordinate compaction."""

    name: ClassVar[str] = "cluster_contiguous_order"
    category: ClassVar[OpCategory] = OpCategory.ORDERING
    reads: ClassVar[Tuple[str, ...]] = ("layers", "ordering", "pos")
    writes: ClassVar[Tuple[str, ...]] = ("ordering", "pos")
    requires: ClassVar[Tuple[str, ...]] = ("layers",)
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[ClusterContiguousOrderConfig] = None) -> None:
        """Store the cluster-contiguity configuration.

        Parameters
        ----------
        config : ClusterContiguousOrderConfig | None, optional
            Optional op configuration.
        """
        self.config = config or ClusterContiguousOrderConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Group cluster members contiguously within every populated layer.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing optional cluster metadata.
        state : SolveState
            Mutable solve state with layer assignments and optional positions.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with a cluster-contiguous ordering when the graph has
            at least ``min_clusters`` valid clusters. Non-clustered graphs are
            returned unchanged.
        """
        del ctx

        if not self.config.enabled:
            return state
        cluster_members = _valid_cluster_members(problem.clusters, problem.num_nodes)
        if len(cluster_members) < self.config.min_clusters:
            return state
        if not _passes_cluster_order_structure_gate(problem, len(cluster_members)):
            return state

        _, layers_cpu, num_nodes = _resolve_active_layered_graph(problem, state)
        ordering_cpu = _resolve_initial_ordering(
            layers_cpu=layers_cpu,
            state=state,
            num_nodes=num_nodes,
        )
        ordered_layers = _initial_ordered_layers(layers_cpu, ordering=ordering_cpu)
        children = _cluster_children(tuple(cluster_members), problem.cluster_parents)
        reordered_layers = [
            _cluster_order_for_layer(
                layer_nodes=layer_nodes,
                cluster_members=cluster_members,
                children_by_parent=children,
            )
            for layer_nodes in ordered_layers
        ]
        state.ordering = _ordering_from_layers(
            ordered_layers=reordered_layers,
            num_nodes=num_nodes,
            device=_target_device(problem, state),
        )
        _apply_ordering_to_positions(
            state=state,
            layers_cpu=layers_cpu,
            ordering_cpu=state.ordering.detach().to(device="cpu", dtype=torch.long),
            num_nodes=num_nodes,
        )
        return state


@register_op
class BarycenterSweep(Op):
    """Order each layer by repeated weighted barycenter sweeps."""

    name: ClassVar[str] = "barycenter_sweep"
    category: ClassVar[OpCategory] = OpCategory.ORDERING
    reads: ClassVar[Tuple[str, ...]] = ("layers", "adjacency")
    writes: ClassVar[Tuple[str, ...]] = ("ordering",)
    requires: ClassVar[Tuple[str, ...]] = ("layers", "adjacency")

    def __init__(self, config: Optional[BarycenterSweepConfig] = None) -> None:
        """Store the sweep configuration.

        Parameters
        ----------
        config : BarycenterSweepConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BarycenterSweepConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run repeated barycenter sweeps and persist per-node ordering ranks.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ``layers`` and ``adjacency`` populated.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with ``ordering`` populated.

        Raises
        ------
        ValueError
            If the configured direction or input tensors are invalid.
        """
        del ctx

        if self.config.direction not in {"down", "up", "both"}:
            raise ValueError("BarycenterSweep direction must be 'down', 'up', or 'both'")

        layers_cpu = _validate_layers(state.layers, problem.num_nodes)
        weighted_adjacency = _adjacency_as_weighted_lists(state.adjacency, problem.num_nodes)
        parents, children, parent_weights, child_weights = _layered_neighbors_from_adjacency(
            adjacency=weighted_adjacency,
            layers=layers_cpu,
            use_weights=self.config.use_weights,
        )
        ordered_layers = _initial_ordered_layers(layers_cpu)

        for _ in range(max(self.config.passes, 0)):
            order_map = _node_order_map(ordered_layers)

            if self.config.direction in {"down", "both"}:
                for layer_index in range(1, len(ordered_layers)):
                    score_map = {
                        node: _barycenter_value(node, parents, parent_weights, order_map)
                        for node in ordered_layers[layer_index]
                    }
                    _sort_layer_by_scores(ordered_layers[layer_index], score_map)
                    order_map = _node_order_map(ordered_layers)

            if self.config.direction in {"up", "both"}:
                for layer_index in range(len(ordered_layers) - 2, -1, -1):
                    score_map = {
                        node: _barycenter_value(node, children, child_weights, order_map)
                        for node in ordered_layers[layer_index]
                    }
                    _sort_layer_by_scores(ordered_layers[layer_index], score_map)
                    order_map = _node_order_map(ordered_layers)

        state.ordering = _ordering_from_layers(
            ordered_layers=ordered_layers,
            num_nodes=problem.num_nodes,
            device=_target_device(problem, state),
        )
        return state


@register_op
class MedianSweep(Op):
    """Order each layer by repeated median-of-neighbors sweeps."""

    name: ClassVar[str] = "median_sweep"
    category: ClassVar[OpCategory] = OpCategory.ORDERING
    reads: ClassVar[Tuple[str, ...]] = ("layers", "ordering", "pos", "adjacency")
    writes: ClassVar[Tuple[str, ...]] = ("ordering", "pos")
    requires: ClassVar[Tuple[str, ...]] = ("layers",)

    def __init__(self, config: Optional[MedianSweepConfig] = None) -> None:
        """Store the sweep configuration.

        Parameters
        ----------
        config : MedianSweepConfig | None, optional
            Optional op configuration.
        """
        self.config = config or MedianSweepConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run repeated median sweeps and persist per-node ordering ranks.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ``layers`` and ``adjacency`` populated.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with ``ordering`` populated.
        """
        del ctx

        edge_index_cpu, layers_cpu, num_nodes = _resolve_active_layered_graph(problem, state)
        try:
            weighted_adjacency = _adjacency_as_weighted_lists(state.adjacency, num_nodes)
            parents, children, _, _ = _layered_neighbors_from_adjacency(
                adjacency=weighted_adjacency,
                layers=layers_cpu,
                use_weights=False,
            )
        except ValueError:
            # The native pipeline does not always build adjacency. Derive the
            # directional layered neighborhoods from the active edge list.
            parents, children = _layered_neighbors_from_edges(
                edge_index=edge_index_cpu,
                layers=layers_cpu,
                num_nodes=num_nodes,
            )
        initial_ordering = _resolve_initial_ordering(
            layers_cpu=layers_cpu,
            state=state,
            num_nodes=num_nodes,
        )
        ordered_layers = _initial_ordered_layers(layers_cpu, ordering=initial_ordering)

        for _ in range(max(self.config.passes, 0)):
            order_map = _node_order_map(ordered_layers)
            for layer_index in range(1, len(ordered_layers)):
                score_map = {
                    node: _median_value(node, parents, order_map)
                    for node in ordered_layers[layer_index]
                }
                _sort_layer_by_scores(ordered_layers[layer_index], score_map)
                order_map = _node_order_map(ordered_layers)

            for layer_index in range(len(ordered_layers) - 2, -1, -1):
                score_map = {
                    node: _median_value(node, children, order_map)
                    for node in ordered_layers[layer_index]
                }
                _sort_layer_by_scores(ordered_layers[layer_index], score_map)
                order_map = _node_order_map(ordered_layers)

        state.ordering = _ordering_from_layers(
            ordered_layers=ordered_layers,
            num_nodes=num_nodes,
            device=_target_device(problem, state),
        )
        _apply_ordering_to_positions(
            state=state,
            layers_cpu=layers_cpu,
            ordering_cpu=state.ordering.detach().to(device="cpu", dtype=torch.long),
            num_nodes=num_nodes,
        )
        return state


@register_op
class TransposeHeuristic(Op):
    """Swap adjacent same-layer nodes when the swap reduces local crossings."""

    name: ClassVar[str] = "transpose_heuristic"
    category: ClassVar[OpCategory] = OpCategory.ORDERING
    reads: ClassVar[Tuple[str, ...]] = ("layers", "ordering", "pos")
    writes: ClassVar[Tuple[str, ...]] = ("ordering", "pos")
    requires: ClassVar[Tuple[str, ...]] = ("layers",)

    def __init__(self, config: Optional[TransposeHeuristicConfig] = None) -> None:
        """Store the transpose configuration.

        Parameters
        ----------
        config : TransposeHeuristicConfig | None, optional
            Optional op configuration.
        """
        self.config = config or TransposeHeuristicConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refine an existing ordering via adjacent swaps.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. ``problem.edge_index`` supplies the
            crossing relationships.
        state : SolveState
            Mutable solve state with ``layers`` and ``ordering`` populated.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with a refined ``ordering`` tensor.
        """
        del ctx

        edge_index_cpu, layers_cpu, num_nodes = _resolve_active_layered_graph(problem, state)
        ordering_cpu = _resolve_initial_ordering(
            layers_cpu=layers_cpu,
            state=state,
            num_nodes=num_nodes,
        )
        if ordering_cpu is None:
            ordering_cpu = _ordering_from_layers(
                ordered_layers=_initial_ordered_layers(layers_cpu),
                num_nodes=num_nodes,
                device=torch.device("cpu"),
            )
        parents, children = _layered_neighbors_from_edges(
            edge_index=edge_index_cpu,
            layers=layers_cpu,
            num_nodes=num_nodes,
        )
        ordered_layers = _initial_ordered_layers(layers=layers_cpu, ordering=ordering_cpu)
        layer_groups = {
            layer_index: list(layer_nodes) for layer_index, layer_nodes in enumerate(ordered_layers)
        }

        _transpose_heuristic(
            layer_groups=layer_groups,
            sorted_layers=list(range(len(ordered_layers))),
            children_of={node: child_nodes for node, child_nodes in enumerate(children)},
            parents_of={node: parent_nodes for node, parent_nodes in enumerate(parents)},
            num_passes=max(self.config.passes, 0),
        )

        refined_layers = [layer_groups[layer_index] for layer_index in range(len(ordered_layers))]
        state.ordering = _ordering_from_layers(
            ordered_layers=refined_layers,
            num_nodes=num_nodes,
            device=_target_device(problem, state),
        )
        _apply_ordering_to_positions(
            state=state,
            layers_cpu=layers_cpu,
            ordering_cpu=state.ordering.detach().to(device="cpu", dtype=torch.long),
            num_nodes=num_nodes,
        )
        return state


@register_op
@dataclass(frozen=True)
class SpectralOrder(Op):
    """Order each layer by the global Fiedler vector of the graph Laplacian."""

    config: SpectralOrderConfig = field(default_factory=SpectralOrderConfig)

    name: ClassVar[str] = "spectral_order"
    category: ClassVar[OpCategory] = OpCategory.ORDERING
    reads: ClassVar[Tuple[str, ...]] = ("layers",)
    writes: ClassVar[Tuple[str, ...]] = ("ordering",)
    requires: ClassVar[Tuple[str, ...]] = ("layers",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute a Fiedler-vector ordering and project it into each layer.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs supplying the graph Laplacian.
        state : SolveState
            Mutable solve state with ``layers`` populated.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with ``ordering`` populated.
        """
        del ctx

        layers_cpu = _validate_layers(state.layers, problem.num_nodes)
        edge_index_cpu = _validate_edge_index(problem.edge_index, problem.num_nodes)
        ordered_layers = _initial_ordered_layers(layers_cpu)
        fiedler = _deterministic_fiedler_vector(
            edge_index=edge_index_cpu,
            num_nodes=problem.num_nodes,
            max_iterations=self.config.max_iterations,
        )

        if fiedler is not None:
            for layer_nodes in ordered_layers:
                # Break ties with the pre-existing layer order so the spectral
                # solve stays deterministic even when coordinates are repeated.
                stable_positions = {node: index for index, node in enumerate(layer_nodes)}
                layer_nodes.sort(
                    key=lambda node: (
                        float(fiedler[node].item()),
                        stable_positions[node],
                        node,
                    )
                )

        state.ordering = _ordering_from_layers(
            ordered_layers=ordered_layers,
            num_nodes=problem.num_nodes,
            device=_target_device(problem, state),
        )
        return state
