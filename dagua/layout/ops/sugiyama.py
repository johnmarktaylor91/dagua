"""Sugiyama layered graph drawing operations.

This module hosts all Sugiyama-private helpers and the registered ops used by
the composable pipeline entrypoint.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import torch

from dagua.layout.cycle import make_acyclic_robust
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import (  # noqa: E402
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory, register_op  # noqa: E402

_NO_SHIFT = float("inf")
_SUGIYAMA_RESOLVED_SIZES_KEY = "sugiyama_resolved_sizes"
_SUGIYAMA_ACYCLIC_EDGES_KEY = "sugiyama_acyclic_edges"
_SUGIYAMA_REVERSED_MASK_KEY = "sugiyama_reversed_mask"
_SUGIYAMA_LAYER_ASSIGNMENTS_KEY = "sugiyama_layer_assignments"
_SUGIYAMA_EXPANDED_GRAPH_KEY = "sugiyama_expanded_graph"
_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY = "sugiyama_expanded_edge_weights"
_SUGIYAMA_PARENTS_KEY = "sugiyama_parents"
_SUGIYAMA_CHILDREN_KEY = "sugiyama_children"
_SUGIYAMA_PARENT_WEIGHTS_KEY = "sugiyama_parent_weights"
_SUGIYAMA_CHILD_WEIGHTS_KEY = "sugiyama_child_weights"
_SUGIYAMA_ORDERED_LAYERS_KEY = "sugiyama_ordered_layers"
_SUGIYAMA_TRACES_KEY = "sugiyama_traces"
_SUGIYAMA_EXPANDED_POSITIONS_KEY = "sugiyama_expanded_positions"
_SUGIYAMA_RANK_SEP_KEY = "sugiyama_rank_sep"
_SUGIYAMA_NODE_SEP_KEY = "sugiyama_node_sep"


@dataclass(frozen=True)
class _ExpandedLayeredGraph:
    """Store the dummy-node-expanded DAG used by Sugiyama sweeps."""

    edge_index: torch.Tensor
    layers: list[list[int]]
    node_sizes: torch.Tensor
    edge_paths: list[list[int]]
    num_nodes: int


@dataclass(frozen=True)
class _BarycenterOrderingConfig:
    """Configuration for :class:`_BarycenterOrdering`.

    Parameters
    ----------
    barycenter_passes : int, default=24
        Number of down/up sweeps used for crossing minimization.
    seed : int, default=42
        Retained for API compatibility with the classic entry point.
    trace_every : int, default=0
        Snapshot cadence in passes. Zero disables tracing.
    """

    barycenter_passes: int = 24
    seed: int = 42
    trace_every: int = 0


@dataclass(frozen=True)
class _StoreSpacingParamsConfig:
    """Configuration for :class:`_StoreSpacingParams`.

    Parameters
    ----------
    rank_sep : float, default=1.0
        Vertical spacing between layers.
    node_sep : float, default=1.0
        Horizontal spacing between nodes within a layer.
    """

    rank_sep: float = 1.0
    node_sep: float = 1.0


def _validate_layout_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    edge_weights: Optional[torch.Tensor],
) -> None:
    """Validate the public Sugiyama layout inputs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor expected to have shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor expected to have shape ``[N, 2]``.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor expected to have shape ``[E]``.

    Raises
    ------
    ValueError
        If any argument violates the required shape contract.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E]")
    if node_sizes is not None:
        if node_sizes.ndim != 2 or node_sizes.shape != (num_nodes, 2):
            raise ValueError("node_sizes must have shape [N, 2]")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} does not match "
            f"edge count {edge_index.shape[1]}"
        )


def _resolve_node_sizes(node_sizes: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Return CPU node sizes for coordinate spacing.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Optional input node sizes.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        CPU float tensor with shape ``[N, 2]``.
    """
    if node_sizes is None:
        return torch.zeros((num_nodes, 2), dtype=torch.float32)
    return node_sizes.detach().to(device="cpu", dtype=torch.float32)


def _prepare_acyclic_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Return a CPU ``edge_index`` with a robust acyclic orientation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge list of shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple
        ``(acyclic_edges, reversed_mask)`` where ``acyclic_edges`` is a CPU
        long tensor with shape ``[2, E]`` suitable for Kahn layering and
        ``reversed_mask`` marks input edges reversed during cycle breaking.
    """
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() == 0:
        return edge_index_cpu, torch.zeros((0,), dtype=torch.bool)
    return make_acyclic_robust(edge_index_cpu, num_nodes)


def _longest_path_layering(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Assign each node to a layer via Kahn topological traversal.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list of shape ``[2, E]`` on CPU.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Long tensor of shape ``[N]`` with layer indices.

    Raises
    ------
    ValueError
        If the graph remains cyclic after edge reversal.
    """
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()

    for src, dst in zip(src_nodes, dst_nodes):
        children[src].append(dst)
        in_degree[dst] += 1

    layers = [0] * num_nodes
    ready = [node for node, degree in enumerate(in_degree) if degree == 0]
    heapq.heapify(ready)

    processed = 0
    while ready:
        node = heapq.heappop(ready)
        processed += 1
        next_layer = layers[node] + 1
        for child in children[node]:
            if next_layer > layers[child]:
                layers[child] = next_layer
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heapq.heappush(ready, child)

    if processed != num_nodes:
        raise ValueError("graph must be acyclic after back-edge reversal")

    return torch.tensor(layers, dtype=torch.long)


def _group_nodes_by_layer(layer_assignments: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Group nodes into ordered per-layer lists.

    Parameters
    ----------
    layer_assignments : torch.Tensor
        Layer indices of shape ``[N]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list of list of int
        Node ids grouped by layer index.
    """
    if num_nodes == 0:
        return []

    num_layers = int(layer_assignments.max().item()) + 1
    layers: List[List[int]] = [[] for _ in range(num_layers)]
    for node in range(num_nodes):
        layer_index = int(layer_assignments[node].item())
        layers[layer_index].append(node)
    return layers


def _promote_layer_assignments(
    edge_index: torch.Tensor,
    layer_assignments: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Promote nodes downward to reduce outgoing dummy nodes.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list with shape ``[2, E]`` on CPU.
    layer_assignments : torch.Tensor
        Initial longest-path layer indices with shape ``[N]``.
    num_nodes : int
        Number of original graph nodes.

    Returns
    -------
    torch.Tensor
        Promoted layer indices with shape ``[N]``.

    Notes
    -----
    The promotion target uses the minimum successor layer minus one. This is
    the deepest feasible layer that preserves all outgoing edges while still
    removing avoidable dummy vertices on outgoing long edges.
    """
    if num_nodes == 0 or edge_index.numel() == 0:
        return layer_assignments

    _, children = _build_neighbor_lists(edge_index=edge_index, num_nodes=num_nodes)
    promoted_layers = layer_assignments.clone()

    changed = True
    while changed:
        changed = False
        node_order = sorted(
            range(num_nodes),
            key=lambda node: int(promoted_layers[node].item()),
            reverse=True,
        )
        for node in node_order:
            successor_layers = [int(promoted_layers[child].item()) for child in children[node]]
            if not successor_layers:
                continue

            current_layer = int(promoted_layers[node].item())
            min_successor_layer = min(successor_layers)
            if min_successor_layer < current_layer + 2:
                continue

            candidate_layer = min_successor_layer - 1
            if candidate_layer > current_layer:
                promoted_layers[node] = candidate_layer
                changed = True

    return promoted_layers


def _expand_long_edges_with_dummy_nodes(
    edge_index: torch.Tensor,
    layer_assignments: torch.Tensor,
    node_sizes: torch.Tensor,
    num_original_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> "_ExpandedLayeredGraph":
    """Insert dummy nodes for edges spanning more than one layer.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list with shape ``[2, E]`` on CPU.
    layer_assignments : torch.Tensor
        Original-node layer indices with shape ``[N]``.
    node_sizes : torch.Tensor
        Original-node sizes with shape ``[N, 2]`` on CPU.
    num_original_nodes : int
        Number of real graph nodes before dummy expansion.

    Returns
    -------
    _ExpandedLayeredGraph
        Expanded layered DAG with dummy nodes inserted on intermediate layers.
    """
    expanded_layers = _group_nodes_by_layer(
        layer_assignments=layer_assignments,
        num_nodes=num_original_nodes,
    )
    dummy_sizes: list[list[float]] = []
    expanded_sources: list[int] = []
    expanded_targets: list[int] = []
    expanded_weight_values: list[float] = []
    edge_paths: list[list[int]] = []
    next_dummy_index = num_original_nodes

    for edge_idx, (source, target) in enumerate(
        zip(edge_index[0].tolist(), edge_index[1].tolist())
    ):
        source_layer = int(layer_assignments[source].item())
        target_layer = int(layer_assignments[target].item())
        path = [source]
        previous = source
        orig_weight = float(edge_weights[edge_idx].item()) if edge_weights is not None else 1.0

        for layer_index in range(source_layer + 1, target_layer):
            dummy_index = next_dummy_index
            next_dummy_index += 1
            expanded_layers[layer_index].append(dummy_index)
            dummy_sizes.append([0.0, 0.0])
            expanded_sources.append(previous)
            expanded_targets.append(dummy_index)
            expanded_weight_values.append(orig_weight)
            path.append(dummy_index)
            previous = dummy_index

        expanded_sources.append(previous)
        expanded_targets.append(target)
        expanded_weight_values.append(orig_weight)
        path.append(target)
        edge_paths.append(path)

    if dummy_sizes:
        expanded_node_sizes = torch.cat(
            [
                node_sizes,
                torch.tensor(dummy_sizes, dtype=torch.float32),
            ],
            dim=0,
        )
    else:
        expanded_node_sizes = node_sizes.clone()

    expanded_edge_index = torch.tensor(
        [expanded_sources, expanded_targets],
        dtype=torch.long,
    )
    expanded_edge_weights: Optional[torch.Tensor] = None
    if edge_weights is not None:
        expanded_edge_weights = torch.tensor(expanded_weight_values, dtype=torch.float32)
    return _ExpandedLayeredGraph(
        edge_index=expanded_edge_index,
        layers=expanded_layers,
        node_sizes=expanded_node_sizes,
        edge_paths=edge_paths,
        num_nodes=next_dummy_index,
    ), expanded_edge_weights


def _build_edge_routes(
    positions: torch.Tensor,
    edge_paths: Sequence[Sequence[int]],
    reversed_edge_mask: torch.Tensor,
    output_device: torch.device,
) -> List[torch.Tensor]:
    """Convert dummy-node chains into routed edge polylines.

    Parameters
    ----------
    positions : torch.Tensor
        Expanded-node coordinates with shape ``[N_total, 2]``.
    edge_paths : sequence of sequence of int
        Expanded node ids visited by each original edge.
    reversed_edge_mask : torch.Tensor
        Boolean tensor marking edges reversed during cycle breaking.
    output_device : torch.device
        Device for returned route tensors.

    Returns
    -------
    list[torch.Tensor]
        Per-edge point sequences aligned to the input edge order.
    """
    routes: List[torch.Tensor] = []
    for edge_index, node_path in enumerate(edge_paths):
        route = positions[list(node_path)].to(device=output_device)
        if edge_index < reversed_edge_mask.numel() and bool(reversed_edge_mask[edge_index].item()):
            route = torch.flip(route, dims=[0])
        routes.append(route)
    return routes


def _build_neighbor_lists(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Build parent and child adjacency lists.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple
        ``(parents, children)`` where each entry is indexed by node id.
    """
    parents: List[List[int]] = [[] for _ in range(num_nodes)]
    children: List[List[int]] = [[] for _ in range(num_nodes)]

    for src, dst in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        parents[dst].append(src)
        children[src].append(dst)

    return parents, children


def _build_neighbor_weight_maps(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> Tuple[List[Dict[int, float]], List[Dict[int, float]]]:
    """Build parent and child edge-weight maps.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    tuple
        ``(parent_weights, child_weights)`` where each entry maps a neighbor
        node id to its accumulated edge weight.
    """
    parent_weights: List[Dict[int, float]] = [dict() for _ in range(num_nodes)]
    child_weights: List[Dict[int, float]] = [dict() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return parent_weights, child_weights

    weights_cpu = (
        torch.ones((edge_index.shape[1],), dtype=torch.float32)
        if edge_weights is None
        else edge_weights.detach().to(device="cpu", dtype=torch.float32)
    )
    for edge_id, (src, dst) in enumerate(zip(edge_index[0].tolist(), edge_index[1].tolist())):
        weight = float(weights_cpu[edge_id].item())
        parent_weights[dst][src] = parent_weights[dst].get(src, 0.0) + weight
        child_weights[src][dst] = child_weights[src].get(dst, 0.0) + weight
    return parent_weights, child_weights


def _barycenter_ordering(
    layers: List[List[int]],
    parents: List[List[int]],
    children: List[List[int]],
    parent_weights: List[Dict[int, float]],
    child_weights: List[Dict[int, float]],
    num_nodes: int,
    num_original_nodes: int,
    num_passes: int,
    seed: int,
    node_sizes: torch.Tensor,
    rank_sep: float,
    node_sep: float,
    trace_every: int,
    output_device: torch.device,
) -> Tuple[List[List[int]], List[torch.Tensor]]:
    """Minimize crossings via repeated barycenter sweeps.

    Parameters
    ----------
    layers : list of list of int
        Node ids grouped by layer.
    parents : list of list of int
        Parent adjacency for every node.
    children : list of list of int
        Child adjacency for every node.
    parent_weights : list of dict
        Parent edge weights for every node.
    child_weights : list of dict
        Child edge weights for every node.
    num_nodes : int
        Number of nodes.
    num_original_nodes : int
        Count of non-dummy nodes.
    num_passes : int
        Number of full down/up sweeps.
    seed : int
        Retained for signature compatibility.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]`` for trace snapshots.
    rank_sep : float
        Vertical layer spacing.
    node_sep : float
        Horizontal node spacing.
    trace_every : int
        Snapshot cadence in passes. Zero disables tracing.
    output_device : torch.device
        Device for the emitted trace tensors.

    Returns
    -------
    tuple
        ``(ordered_layers, traces)``.
    """
    ordered_layers = [sorted(layer) for layer in layers]
    if num_nodes == 0:
        return ordered_layers, []

    del seed
    traces: List[torch.Tensor] = []

    for pass_num in range(num_passes):
        order_index = _node_order_map(ordered_layers)

        for layer_idx in range(1, len(ordered_layers)):
            barycenters = _neighbor_barycenters(
                nodes=ordered_layers[layer_idx],
                neighbors_by_node=parents,
                neighbor_weights_by_node=parent_weights,
                order_index=order_index,
            )
            ordered_layers[layer_idx].sort(key=lambda node: barycenters[node])
            order_index = _node_order_map(ordered_layers)

        for layer_idx in range(len(ordered_layers) - 2, -1, -1):
            barycenters = _neighbor_barycenters(
                nodes=ordered_layers[layer_idx],
                neighbors_by_node=children,
                neighbor_weights_by_node=child_weights,
                order_index=order_index,
            )
            ordered_layers[layer_idx].sort(key=lambda node: barycenters[node])
            order_index = _node_order_map(ordered_layers)

        if trace_every > 0 and (pass_num + 1) % trace_every == 0:
            traces.append(
                _coordinate_assignment(
                    layers=ordered_layers,
                    parents=parents,
                    children=children,
                    node_sizes=node_sizes,
                    num_nodes=num_nodes,
                    num_original_nodes=num_original_nodes,
                    rank_sep=rank_sep,
                    node_sep=node_sep,
                    output_device=output_device,
                )
            )

    return ordered_layers, traces


def _neighbor_barycenters(
    nodes: Sequence[int],
    neighbors_by_node: Sequence[Sequence[int]],
    neighbor_weights_by_node: Sequence[Dict[int, float]],
    order_index: Dict[int, float],
) -> Dict[int, float]:
    """Compute barycenter values from already ordered neighboring layers.

    Parameters
    ----------
    nodes : sequence of int
        Nodes in the layer currently being sorted.
    neighbors_by_node : sequence of sequence of int
        Parent or child adjacency indexed by node.
    neighbor_weights_by_node : sequence of dict
        Parent or child edge-weight maps indexed by node.
    order_index : dict
        Current within-layer order position for every node.

    Returns
    -------
    dict
        Mapping from node id to barycenter score.
    """
    barycenters: Dict[int, float] = {}
    for node in nodes:
        neighbor_positions = [order_index[neighbor] for neighbor in neighbors_by_node[node]]
        if neighbor_positions:
            weighted_sum = 0.0
            total_weight = 0.0
            for neighbor in neighbors_by_node[node]:
                weight = neighbor_weights_by_node[node].get(neighbor, 1.0)
                weighted_sum += weight * order_index[neighbor]
                total_weight += weight
            if total_weight > 0.0:
                barycenters[node] = weighted_sum / total_weight
            else:
                barycenters[node] = sum(neighbor_positions) / float(len(neighbor_positions))
        else:
            barycenters[node] = order_index[node]
    return barycenters


def _node_order_map(layers: Sequence[Sequence[int]]) -> Dict[int, float]:
    """Map node ids to their current order within layers.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered layer contents.

    Returns
    -------
    dict
        Mapping from node id to its current in-layer position.
    """
    order_index: Dict[int, float] = {}
    for layer_nodes in layers:
        for index, node in enumerate(layer_nodes):
            order_index[node] = float(index)
    return order_index


def _coordinate_assignment(
    layers: Sequence[Sequence[int]],
    parents: Sequence[Sequence[int]],
    children: Sequence[Sequence[int]],
    node_sizes: torch.Tensor,
    num_nodes: int,
    num_original_nodes: int,
    rank_sep: float,
    node_sep: float,
    output_device: torch.device,
) -> torch.Tensor:
    """Assign ``(x, y)`` coordinates with Brandes-Kopf compaction.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer.
    parents : sequence of sequence of int
        Parent adjacency indexed by node id.
    children : sequence of sequence of int
        Child adjacency indexed by node id.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]``.
    num_nodes : int
        Number of nodes.
    num_original_nodes : int
        Count of non-dummy nodes. Dummy nodes occupy the trailing indices
        ``[num_original_nodes, num_nodes)`` after long-edge expansion.
    rank_sep : float
        Vertical distance between layers.
    node_sep : float
        Horizontal gap between node bounding boxes.
    output_device : torch.device
        Device for the returned position tensor.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for layer_idx, layer_nodes in enumerate(layers):
        if not layer_nodes:
            continue
        positions[layer_nodes, 1] = float(layer_idx) * rank_sep

    if num_nodes == 0:
        return positions.to(output_device)

    x_positions = _brandes_koepf_x_positions(
        layers=layers,
        parents=parents,
        children=children,
        node_sizes=node_sizes,
        num_nodes=num_nodes,
        num_original_nodes=num_original_nodes,
        node_sep=node_sep,
    )
    positions[:, 0] = torch.tensor(x_positions, dtype=torch.float32)
    return positions.to(output_device)


def _brandes_koepf_x_positions(
    layers: Sequence[Sequence[int]],
    parents: Sequence[Sequence[int]],
    children: Sequence[Sequence[int]],
    node_sizes: torch.Tensor,
    num_nodes: int,
    num_original_nodes: int,
    node_sep: float,
) -> List[float]:
    """Compute balanced horizontal coordinates with four BK passes.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer.
    parents : sequence of sequence of int
        Parent adjacency indexed by node id.
    children : sequence of sequence of int
        Child adjacency indexed by node id.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]``.
    num_nodes : int
        Number of nodes.
    num_original_nodes : int
        Count of non-dummy nodes.
    node_sep : float
        Horizontal gap between node bounding boxes.

    Returns
    -------
    list of float
        Final X coordinates for all expanded nodes.
    """
    if num_nodes == 0:
        return []

    dummy_mask = [node >= num_original_nodes for node in range(num_nodes)]
    orientation_specs = (
        ("ul", False, False),
        ("ur", False, True),
        ("dl", True, False),
        ("dr", True, True),
    )
    x_by_alignment: Dict[str, List[float]] = {}
    for alignment_name, reverse_layers, reverse_within in orientation_specs:
        transformed_layers = _transform_layers(
            layers=layers,
            reverse_layers=reverse_layers,
            reverse_within=reverse_within,
        )
        rank_of, pos_of = _layer_position_maps(layers=transformed_layers, num_nodes=num_nodes)
        predecessor_source = children if reverse_layers else parents
        predecessors = _ordered_transformed_neighbors(
            neighbors_by_node=predecessor_source,
            pos_of=pos_of,
        )
        conflicts = _find_type1_conflicts(
            layers=transformed_layers,
            predecessors=predecessors,
            pos_of=pos_of,
            dummy_mask=dummy_mask,
        )
        root, align = _vertical_alignment(
            layers=transformed_layers,
            predecessors=predecessors,
            pos_of=pos_of,
            conflicts=conflicts,
            num_nodes=num_nodes,
        )
        compacted = _horizontal_compaction(
            layers=transformed_layers,
            root=root,
            align=align,
            rank_of=rank_of,
            pos_of=pos_of,
            node_sizes=node_sizes,
            node_sep=node_sep,
            num_nodes=num_nodes,
        )
        if reverse_within:
            compacted = [-value for value in compacted]
        x_by_alignment[alignment_name] = compacted

    _align_compacted_coordinates(x_by_alignment=x_by_alignment)
    balanced = _median_balanced_coordinates(x_by_alignment=x_by_alignment, num_nodes=num_nodes)
    return _center_coordinates(values=balanced)


def _transform_layers(
    layers: Sequence[Sequence[int]],
    reverse_layers: bool,
    reverse_within: bool,
) -> List[List[int]]:
    """Return an orientation-specific view of the ordered layers.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer in the original top-down orientation.
    reverse_layers : bool
        Whether to scan layers from bottom to top.
    reverse_within : bool
        Whether to mirror each layer left-to-right.

    Returns
    -------
    list of list of int
        Oriented layer ordering for one Brandes-Kopf pass.
    """
    oriented_layers = [list(layer) for layer in layers]
    if reverse_layers:
        oriented_layers = list(reversed(oriented_layers))
    if reverse_within:
        oriented_layers = [list(reversed(layer)) for layer in oriented_layers]
    return oriented_layers


def _layer_position_maps(
    layers: Sequence[Sequence[int]],
    num_nodes: int,
) -> Tuple[List[int], List[int]]:
    """Build rank and order maps for one oriented layering.

    Parameters
    ----------
    layers : sequence of sequence of int
        Oriented layers for one pass.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple
        ``(rank_of, pos_of)`` arrays indexed by node id.
    """
    rank_of = [-1] * num_nodes
    pos_of = [-1] * num_nodes
    for rank_index, layer_nodes in enumerate(layers):
        for position, node in enumerate(layer_nodes):
            rank_of[node] = rank_index
            pos_of[node] = position
    return rank_of, pos_of


def _ordered_transformed_neighbors(
    neighbors_by_node: Sequence[Sequence[int]],
    pos_of: Sequence[int],
) -> List[List[int]]:
    """Sort neighbors by their position in the transformed layering.

    Parameters
    ----------
    neighbors_by_node : sequence of sequence of int
        Neighbor adjacency indexed by node id.
    pos_of : sequence of int
        Within-layer positions for the transformed layering.

    Returns
    -------
    list of list of int
        Neighbor ids sorted left-to-right in the transformed layering.
    """
    ordered_neighbors: List[List[int]] = []
    for neighbors in neighbors_by_node:
        ordered_neighbors.append(sorted(neighbors, key=lambda node: pos_of[node]))
    return ordered_neighbors


def _find_type1_conflicts(
    layers: Sequence[Sequence[int]],
    predecessors: Sequence[Sequence[int]],
    pos_of: Sequence[int],
    dummy_mask: Sequence[bool],
) -> Set[Tuple[int, int]]:
    """Mark Type 1 conflicts between inner and non-inner segments.

    Parameters
    ----------
    layers : sequence of sequence of int
        Oriented layers for one Brandes-Kopf pass.
    predecessors : sequence of sequence of int
        Oriented predecessor adjacency indexed by node id.
    pos_of : sequence of int
        Within-layer positions in the oriented layering.
    dummy_mask : sequence of bool
        Flags indicating which nodes are dummy vertices created for long
        edges.

    Returns
    -------
    set of tuple of int
        Conflicting oriented edges as ``(predecessor, node)`` pairs.
    """
    conflicts: Set[Tuple[int, int]] = set()
    for rank_index in range(1, len(layers)):
        north_layer = layers[rank_index - 1]
        south_layer = layers[rank_index]
        if not south_layer:
            continue

        left_boundary = 0
        scan_start = 0
        last_index = len(south_layer) - 1
        for south_index, node in enumerate(south_layer):
            inner_predecessor = _inner_segment_predecessor(
                node=node,
                predecessors=predecessors,
                dummy_mask=dummy_mask,
            )
            right_boundary = (
                pos_of[inner_predecessor] if inner_predecessor is not None else len(north_layer)
            )
            if inner_predecessor is None and south_index != last_index:
                continue

            for scan_node in south_layer[scan_start : south_index + 1]:
                for predecessor in predecessors[scan_node]:
                    predecessor_position = pos_of[predecessor]
                    is_inner_segment = dummy_mask[scan_node] and dummy_mask[predecessor]
                    if not is_inner_segment and (
                        predecessor_position < left_boundary
                        or predecessor_position > right_boundary
                    ):
                        conflicts.add((predecessor, scan_node))

            scan_start = south_index + 1
            left_boundary = right_boundary
    return conflicts


def _inner_segment_predecessor(
    node: int,
    predecessors: Sequence[Sequence[int]],
    dummy_mask: Sequence[bool],
) -> Optional[int]:
    """Return the predecessor of an inner segment dummy, if present.

    Parameters
    ----------
    node : int
        Candidate node on the lower layer of an oriented pass.
    predecessors : sequence of sequence of int
        Oriented predecessor adjacency indexed by node id.
    dummy_mask : sequence of bool
        Flags indicating dummy nodes.

    Returns
    -------
    int, optional
        The predecessor node id when ``node`` participates in a dummy-to-dummy
        inner segment; otherwise ``None``.
    """
    if not dummy_mask[node] or len(predecessors[node]) != 1:
        return None
    predecessor = predecessors[node][0]
    if not dummy_mask[predecessor]:
        return None
    return predecessor


def _vertical_alignment(
    layers: Sequence[Sequence[int]],
    predecessors: Sequence[Sequence[int]],
    pos_of: Sequence[int],
    conflicts: Set[Tuple[int, int]],
    num_nodes: int,
) -> Tuple[List[int], List[int]]:
    """Construct aligned blocks for one Brandes-Kopf orientation.

    Parameters
    ----------
    layers : sequence of sequence of int
        Oriented layers for one pass.
    predecessors : sequence of sequence of int
        Oriented predecessor adjacency indexed by node id.
    pos_of : sequence of int
        Within-layer positions in the oriented layering.
    conflicts : set of tuple of int
        Type 1 conflicts as ``(predecessor, node)`` pairs.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple
        ``(root, align)`` arrays indexed by node id.
    """
    root = list(range(num_nodes))
    align = list(range(num_nodes))

    for rank_index in range(1, len(layers)):
        previous_position = -1
        for node in layers[rank_index]:
            neighbor_nodes = predecessors[node]
            if not neighbor_nodes:
                continue

            median_start = (len(neighbor_nodes) - 1) // 2
            median_stop = len(neighbor_nodes) // 2
            for neighbor_index in range(median_start, median_stop + 1):
                predecessor = neighbor_nodes[neighbor_index]
                if align[node] != node:
                    break
                if pos_of[predecessor] <= previous_position:
                    continue
                if (predecessor, node) in conflicts:
                    continue

                align[predecessor] = node
                align[node] = root[node] = root[predecessor]
                previous_position = pos_of[predecessor]
    return root, align


def _horizontal_compaction(
    layers: Sequence[Sequence[int]],
    root: Sequence[int],
    align: Sequence[int],
    rank_of: Sequence[int],
    pos_of: Sequence[int],
    node_sizes: torch.Tensor,
    node_sep: float,
    num_nodes: int,
) -> List[float]:
    """Compact one aligned orientation into concrete X positions.

    Parameters
    ----------
    layers : sequence of sequence of int
        Oriented layers for one pass.
    root : sequence of int
        Block-root array from vertical alignment.
    align : sequence of int
        Cyclic alignment array from vertical alignment.
    rank_of : sequence of int
        Layer indices in the oriented layering.
    pos_of : sequence of int
        Within-layer positions in the oriented layering.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]``.
    node_sep : float
        Horizontal gap between node bounding boxes.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list of float
        Compacted X coordinates for the oriented pass.
    """
    sink = list(range(num_nodes))
    shift = [_NO_SHIFT] * num_nodes
    x: List[Optional[float]] = [None] * num_nodes

    for node in range(num_nodes):
        if root[node] == node:
            _place_compaction_block(
                block_root=node,
                layers=layers,
                root=root,
                align=align,
                rank_of=rank_of,
                pos_of=pos_of,
                node_sizes=node_sizes,
                node_sep=node_sep,
                sink=sink,
                shift=shift,
                x=x,
            )

    compacted = [0.0] * num_nodes
    for node in range(num_nodes):
        block_root = root[node]
        block_x = 0.0 if x[block_root] is None else x[block_root]
        sink_shift = shift[sink[block_root]]
        compacted[node] = block_x if sink_shift == _NO_SHIFT else block_x + sink_shift
    return compacted


def _place_compaction_block(
    block_root: int,
    layers: Sequence[Sequence[int]],
    root: Sequence[int],
    align: Sequence[int],
    rank_of: Sequence[int],
    pos_of: Sequence[int],
    node_sizes: torch.Tensor,
    node_sep: float,
    sink: List[int],
    shift: List[float],
    x: List[Optional[float]],
) -> None:
    """Recursively place one aligned block during horizontal compaction.

    Parameters
    ----------
    block_root : int
        Root node of the block currently being placed.
    layers : sequence of sequence of int
        Oriented layers for the current pass.
    root : sequence of int
        Block-root array from vertical alignment.
    align : sequence of int
        Cyclic alignment array from vertical alignment.
    rank_of : sequence of int
        Layer indices in the oriented layering.
    pos_of : sequence of int
        Within-layer positions in the oriented layering.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]``.
    node_sep : float
        Horizontal gap between node bounding boxes.
    sink : list of int
        Sink representative for each block root.
    shift : list of float
        Deferred class shifts indexed by sink root.
    x : list of float, optional
        Root coordinates under construction.
    """
    if x[block_root] is not None:
        return

    x[block_root] = 0.0
    current = block_root
    while True:
        layer_nodes = layers[rank_of[current]]
        position = pos_of[current]
        if position > 0:
            left_neighbor = layer_nodes[position - 1]
            left_root = root[left_neighbor]
            _place_compaction_block(
                block_root=left_root,
                layers=layers,
                root=root,
                align=align,
                rank_of=rank_of,
                pos_of=pos_of,
                node_sizes=node_sizes,
                node_sep=node_sep,
                sink=sink,
                shift=shift,
                x=x,
            )
            if sink[block_root] == block_root:
                sink[block_root] = sink[left_root]

            block_x = 0.0 if x[block_root] is None else x[block_root]
            left_x = 0.0 if x[left_root] is None else x[left_root]
            minimum_gap = _minimum_separation(
                left_node=left_neighbor,
                right_node=current,
                node_sizes=node_sizes,
                node_sep=node_sep,
            )
            if sink[block_root] != sink[left_root]:
                shift[sink[left_root]] = min(
                    shift[sink[left_root]],
                    block_x - left_x - minimum_gap,
                )
            else:
                x[block_root] = max(block_x, left_x + minimum_gap)

        current = align[current]
        if current == block_root:
            break


def _minimum_separation(
    left_node: int,
    right_node: int,
    node_sizes: torch.Tensor,
    node_sep: float,
) -> float:
    """Return the minimum center-to-center separation within one layer.

    Parameters
    ----------
    left_node : int
        Left node id in the current orientation.
    right_node : int
        Right node id in the current orientation.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]``.
    node_sep : float
        Horizontal gap between node bounding boxes.

    Returns
    -------
    float
        Required center-to-center distance between the node pair.
    """
    return (
        float(node_sizes[left_node, 0].item()) + float(node_sizes[right_node, 0].item())
    ) / 2.0 + node_sep


def _align_compacted_coordinates(x_by_alignment: Dict[str, List[float]]) -> None:
    """Shift the four compacted assignments into a common coordinate frame.

    Parameters
    ----------
    x_by_alignment : dict
        Mapping from alignment name (``ul``, ``ur``, ``dl``, ``dr``) to
        compacted X coordinates.
    """
    if not x_by_alignment:
        return

    anchor_name = min(
        x_by_alignment,
        key=lambda name: _coordinate_span(values=x_by_alignment[name]),
    )
    anchor_values = x_by_alignment[anchor_name]
    anchor_left = min(anchor_values)
    anchor_right = max(anchor_values)

    for alignment_name, values in x_by_alignment.items():
        if alignment_name == anchor_name:
            continue
        if alignment_name.endswith("l"):
            shift = anchor_left - min(values)
        else:
            shift = anchor_right - max(values)
        for index in range(len(values)):
            values[index] += shift


def _coordinate_span(values: Sequence[float]) -> float:
    """Return the span of a coordinate assignment.

    Parameters
    ----------
    values : sequence of float
        Coordinate values.

    Returns
    -------
    float
        ``max(values) - min(values)`` or zero for empty inputs.
    """
    if not values:
        return 0.0
    return max(values) - min(values)


def _median_balanced_coordinates(
    x_by_alignment: Dict[str, Sequence[float]],
    num_nodes: int,
) -> List[float]:
    """Take the per-node median across the four compacted assignments.

    Parameters
    ----------
    x_by_alignment : dict
        Mapping from alignment name to compacted X coordinates.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list of float
        Median-balanced X coordinates.
    """
    balanced = [0.0] * num_nodes
    for node in range(num_nodes):
        samples = sorted(values[node] for values in x_by_alignment.values())
        balanced[node] = (samples[1] + samples[2]) / 2.0
    return balanced


def _center_coordinates(values: Sequence[float]) -> List[float]:
    """Translate coordinates so the overall span is centered at zero.

    Parameters
    ----------
    values : sequence of float
        Coordinate values.

    Returns
    -------
    list of float
        Centered coordinates.
    """
    if not values:
        return []

    midpoint = (min(values) + max(values)) / 2.0
    return [value - midpoint for value in values]


@register_op
class _ValidateInputs(Op):
    """Validate Sugiyama layout inputs exactly like the classic entry point."""

    name: ClassVar[str] = "sugiyama_validate_inputs"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ()
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the classic validation checks on the problem inputs.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state. Unchanged by this op.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Unmodified state after validation passes.

        Raises
        ------
        ValueError
            If edge_index, node_sizes, or edge_weights have invalid shapes.
        """
        del ctx

        _validate_layout_inputs(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            node_sizes=problem.node_sizes,
            edge_weights=problem.edge_weights,
        )
        return state


@register_op
class _ResolveNodeSizes(Op):
    """Resolve node sizes to CPU float tensor for coordinate spacing."""

    name: ClassVar[str] = "sugiyama_resolve_node_sizes"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = (f"extras.{_SUGIYAMA_RESOLVED_SIZES_KEY}",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Store resolved node sizes in extras.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing optional node_sizes.
        state : SolveState
            Mutable solve state receiving resolved sizes.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with resolved node sizes cached in ``state.extras``.
        """
        del ctx

        resolved = _resolve_node_sizes(
            node_sizes=problem.node_sizes,
            num_nodes=problem.num_nodes,
        )
        state.extras[_SUGIYAMA_RESOLVED_SIZES_KEY] = resolved
        return state


@register_op
class _PrepareAcyclicEdges(Op):
    """Break cycles to produce an acyclic edge orientation."""

    name: ClassVar[str] = "sugiyama_prepare_acyclic_edges"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_ACYCLIC_EDGES_KEY}",
        f"extras.{_SUGIYAMA_REVERSED_MASK_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run cycle removal and store the acyclic edges and reversal mask.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing edge_index.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with acyclic edges and reversed mask in extras.
        """
        del ctx

        acyclic_edges, reversed_mask = _prepare_acyclic_edges(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
        )
        state.extras[_SUGIYAMA_ACYCLIC_EDGES_KEY] = acyclic_edges
        state.extras[_SUGIYAMA_REVERSED_MASK_KEY] = reversed_mask
        return state


@register_op
class _AssignLayers(Op):
    """Assign nodes to layers via longest-path layering with promotion."""

    name: ClassVar[str] = "sugiyama_assign_layers"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = (f"extras.{_SUGIYAMA_ACYCLIC_EDGES_KEY}",)
    writes: ClassVar[Tuple[str, ...]] = (f"extras.{_SUGIYAMA_LAYER_ASSIGNMENTS_KEY}",)
    requires: ClassVar[Tuple[str, ...]] = (f"extras.{_SUGIYAMA_ACYCLIC_EDGES_KEY}",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute longest-path layers and promote to reduce dummy nodes.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with acyclic edges already in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with promoted layer assignments in extras.
        """
        del ctx

        acyclic_edges = state.extras[_SUGIYAMA_ACYCLIC_EDGES_KEY]
        layer_assignments = _longest_path_layering(
            edge_index=acyclic_edges,
            num_nodes=problem.num_nodes,
        )
        layer_assignments = _promote_layer_assignments(
            edge_index=acyclic_edges,
            layer_assignments=layer_assignments,
            num_nodes=problem.num_nodes,
        )
        state.extras[_SUGIYAMA_LAYER_ASSIGNMENTS_KEY] = layer_assignments
        return state


@register_op
class _ExpandDummyNodes(Op):
    """Insert dummy nodes for edges spanning more than one layer."""

    name: ClassVar[str] = "sugiyama_expand_dummy_nodes"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_ACYCLIC_EDGES_KEY}",
        f"extras.{_SUGIYAMA_LAYER_ASSIGNMENTS_KEY}",
        f"extras.{_SUGIYAMA_RESOLVED_SIZES_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_ACYCLIC_EDGES_KEY}",
        f"extras.{_SUGIYAMA_LAYER_ASSIGNMENTS_KEY}",
        f"extras.{_SUGIYAMA_RESOLVED_SIZES_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Expand long edges with dummy nodes and store the expanded graph.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with layer assignments in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with expanded graph and edge weights in extras.
        """
        del ctx

        expanded_graph, expanded_edge_weights = _expand_long_edges_with_dummy_nodes(
            edge_index=state.extras[_SUGIYAMA_ACYCLIC_EDGES_KEY],
            layer_assignments=state.extras[_SUGIYAMA_LAYER_ASSIGNMENTS_KEY],
            node_sizes=state.extras[_SUGIYAMA_RESOLVED_SIZES_KEY],
            num_original_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        state.extras[_SUGIYAMA_EXPANDED_GRAPH_KEY] = expanded_graph
        state.extras[_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY] = expanded_edge_weights
        return state


@register_op
class _BuildNeighborStructures(Op):
    """Build parent/child adjacency lists and edge-weight maps."""

    name: ClassVar[str] = "sugiyama_build_neighbor_structures"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_PARENTS_KEY}",
        f"extras.{_SUGIYAMA_CHILDREN_KEY}",
        f"extras.{_SUGIYAMA_PARENT_WEIGHTS_KEY}",
        f"extras.{_SUGIYAMA_CHILD_WEIGHTS_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",)
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build adjacency lists and weight maps for crossing minimization.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with expanded graph in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with neighbor lists and weight maps in extras.
        """
        del ctx

        expanded_graph = state.extras[_SUGIYAMA_EXPANDED_GRAPH_KEY]
        parents, children = _build_neighbor_lists(
            edge_index=expanded_graph.edge_index,
            num_nodes=expanded_graph.num_nodes,
        )
        parent_weights, child_weights = _build_neighbor_weight_maps(
            edge_index=expanded_graph.edge_index,
            num_nodes=expanded_graph.num_nodes,
            edge_weights=state.extras[_SUGIYAMA_EXPANDED_EDGE_WEIGHTS_KEY],
        )
        state.extras[_SUGIYAMA_PARENTS_KEY] = parents
        state.extras[_SUGIYAMA_CHILDREN_KEY] = children
        state.extras[_SUGIYAMA_PARENT_WEIGHTS_KEY] = parent_weights
        state.extras[_SUGIYAMA_CHILD_WEIGHTS_KEY] = child_weights
        return state


@register_op
class _BarycenterOrdering(Op):
    """Minimize edge crossings via repeated barycenter sweeps."""

    config: _BarycenterOrderingConfig

    name: ClassVar[str] = "sugiyama_barycenter_ordering"
    category: ClassVar[OpCategory] = OpCategory.ORDERING
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_PARENTS_KEY}",
        f"extras.{_SUGIYAMA_CHILDREN_KEY}",
        f"extras.{_SUGIYAMA_PARENT_WEIGHTS_KEY}",
        f"extras.{_SUGIYAMA_CHILD_WEIGHTS_KEY}",
        f"extras.{_SUGIYAMA_RANK_SEP_KEY}",
        f"extras.{_SUGIYAMA_NODE_SEP_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_ORDERED_LAYERS_KEY}",
        f"extras.{_SUGIYAMA_TRACES_KEY}",
    )
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_PARENTS_KEY}",
        f"extras.{_SUGIYAMA_CHILDREN_KEY}",
        f"extras.{_SUGIYAMA_PARENT_WEIGHTS_KEY}",
        f"extras.{_SUGIYAMA_CHILD_WEIGHTS_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def __init__(
        self,
        barycenter_passes: int = 24,
        seed: int = 42,
        trace_every: int = 0,
        *,
        config: Optional[_BarycenterOrderingConfig] = None,
    ) -> None:
        """Store barycenter sweep parameters.

        Parameters
        ----------
        barycenter_passes : int, default=24
            Number of up/down sweeps for crossing minimization.
        seed : int, default=42
            Retained for API compatibility.
        trace_every : int, default=0
            Snapshot cadence in passes. Zero disables tracing.
        config : _BarycenterOrderingConfig | None, optional
            Optional configuration. When provided, it takes precedence over
            the scalar arguments.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or _BarycenterOrderingConfig(
            barycenter_passes=barycenter_passes,
            seed=seed,
            trace_every=trace_every,
        )

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run barycenter ordering sweeps on the expanded graph.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with neighbor structures in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with ordered layers and traces in extras.
        """
        del ctx

        expanded_graph = state.extras[_SUGIYAMA_EXPANDED_GRAPH_KEY]
        output_device = problem.edge_index.device
        if problem.node_sizes is not None:
            output_device = problem.node_sizes.device

        rank_sep = state.extras.get(_SUGIYAMA_RANK_SEP_KEY, 1.0)
        node_sep = state.extras.get(_SUGIYAMA_NODE_SEP_KEY, 1.0)

        ordered_layers, traces = _barycenter_ordering(
            layers=expanded_graph.layers,
            parents=state.extras[_SUGIYAMA_PARENTS_KEY],
            children=state.extras[_SUGIYAMA_CHILDREN_KEY],
            parent_weights=state.extras[_SUGIYAMA_PARENT_WEIGHTS_KEY],
            child_weights=state.extras[_SUGIYAMA_CHILD_WEIGHTS_KEY],
            num_nodes=expanded_graph.num_nodes,
            num_original_nodes=problem.num_nodes,
            num_passes=self.config.barycenter_passes,
            seed=self.config.seed,
            node_sizes=expanded_graph.node_sizes,
            rank_sep=rank_sep,
            node_sep=node_sep,
            trace_every=self.config.trace_every,
            output_device=output_device,
        )
        state.extras[_SUGIYAMA_ORDERED_LAYERS_KEY] = ordered_layers
        state.extras[_SUGIYAMA_TRACES_KEY] = traces
        return state


@register_op
class _CoordinateAssignment(Op):
    """Assign (x, y) coordinates with Brandes-Kopf compaction."""

    name: ClassVar[str] = "sugiyama_coordinate_assignment"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_ORDERED_LAYERS_KEY}",
        f"extras.{_SUGIYAMA_PARENTS_KEY}",
        f"extras.{_SUGIYAMA_CHILDREN_KEY}",
        f"extras.{_SUGIYAMA_RANK_SEP_KEY}",
        f"extras.{_SUGIYAMA_NODE_SEP_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos", f"extras.{_SUGIYAMA_EXPANDED_POSITIONS_KEY}")
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_ORDERED_LAYERS_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run Brandes-Kopf coordinate assignment on the ordered layers.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ordered layers in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with final positions in pos (original nodes only).
        """
        del ctx

        expanded_graph = state.extras[_SUGIYAMA_EXPANDED_GRAPH_KEY]
        output_device = problem.edge_index.device
        if problem.node_sizes is not None:
            output_device = problem.node_sizes.device

        rank_sep = state.extras.get(_SUGIYAMA_RANK_SEP_KEY, 1.0)
        node_sep = state.extras.get(_SUGIYAMA_NODE_SEP_KEY, 1.0)

        expanded_positions = _coordinate_assignment(
            layers=state.extras[_SUGIYAMA_ORDERED_LAYERS_KEY],
            parents=state.extras[_SUGIYAMA_PARENTS_KEY],
            children=state.extras[_SUGIYAMA_CHILDREN_KEY],
            node_sizes=expanded_graph.node_sizes,
            num_nodes=expanded_graph.num_nodes,
            num_original_nodes=problem.num_nodes,
            rank_sep=rank_sep,
            node_sep=node_sep,
            output_device=output_device,
        )
        # Keep the expanded coordinates for downstream edge routing before
        # slicing back to the original node set.
        state.extras[_SUGIYAMA_EXPANDED_POSITIONS_KEY] = expanded_positions
        state.pos = expanded_positions[: problem.num_nodes]
        return state


@register_op
class _BuildEdgeRoutes(Op):
    """Convert dummy-node chains into routed edge polylines."""

    name: ClassVar[str] = "sugiyama_build_edge_routes"
    category: ClassVar[OpCategory] = OpCategory.EDGE_ROUTE
    reads: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_EXPANDED_POSITIONS_KEY}",
        f"extras.{_SUGIYAMA_REVERSED_MASK_KEY}",
    )
    writes: ClassVar[Tuple[str, ...]] = ("edge_routes",)
    requires: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_EXPANDED_GRAPH_KEY}",
        f"extras.{_SUGIYAMA_EXPANDED_POSITIONS_KEY}",
        f"extras.{_SUGIYAMA_REVERSED_MASK_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build polyline edge routes from expanded positions and edge paths.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with expanded positions in extras.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with edge_routes populated.
        """
        del ctx

        expanded_graph = state.extras[_SUGIYAMA_EXPANDED_GRAPH_KEY]
        output_device = problem.edge_index.device
        if problem.node_sizes is not None:
            output_device = problem.node_sizes.device

        state.edge_routes = _build_edge_routes(
            positions=state.extras[_SUGIYAMA_EXPANDED_POSITIONS_KEY],
            edge_paths=expanded_graph.edge_paths,
            reversed_edge_mask=state.extras[_SUGIYAMA_REVERSED_MASK_KEY],
            output_device=output_device,
        )
        return state


@register_op
class _StoreSpacingParams(Op):
    """Store rank_sep and node_sep in extras for downstream ops."""

    config: _StoreSpacingParamsConfig

    name: ClassVar[str] = "sugiyama_store_spacing"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = (
        f"extras.{_SUGIYAMA_RANK_SEP_KEY}",
        f"extras.{_SUGIYAMA_NODE_SEP_KEY}",
    )
    access_pattern: ClassVar[str] = "global"

    def __init__(
        self,
        rank_sep: float = 1.0,
        node_sep: float = 1.0,
        *,
        config: Optional[_StoreSpacingParamsConfig] = None,
    ) -> None:
        """Store spacing parameters.

        Parameters
        ----------
        rank_sep : float, default=1.0
            Vertical spacing between layers.
        node_sep : float, default=1.0
            Horizontal spacing between nodes within a layer.
        config : _StoreSpacingParamsConfig | None, optional
            Optional configuration. When provided, it takes precedence over
            the scalar arguments.

        Returns
        -------
        None
            This constructor stores configuration only.
        """
        self.config = config or _StoreSpacingParamsConfig(rank_sep=rank_sep, node_sep=node_sep)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Write spacing parameters to extras.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state receiving spacing parameters.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with spacing params stored in extras.
        """
        del problem, ctx

        state.extras[_SUGIYAMA_RANK_SEP_KEY] = self.config.rank_sep
        state.extras[_SUGIYAMA_NODE_SEP_KEY] = self.config.node_sep
        return state
