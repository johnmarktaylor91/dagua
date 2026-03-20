"""Classic Sugiyama layered graph drawing (4-phase pipeline).

The standard approach for DAG visualization since 1981. Four discrete phases:
1. Layer assignment (longest-path)
2. Crossing minimization (barycenter heuristic)
3. Coordinate assignment (priority layout)
4. Edge routing (not implemented — just straight lines)

This is what Graphviz dot implements (with many refinements). Our version
is a clean, minimal implementation of the core algorithm for comparison.

Reference: Sugiyama, Tagawa & Toda, "Methods for Visual Understanding of
Hierarchical System Structures" (1981), IEEE Trans. SMC.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from dagua.layout.cycle import detect_back_edges, make_acyclic

_COORDINATE_REFINEMENT_SWEEPS = 16


@dataclass(frozen=True)
class _ExpandedLayeredGraph:
    """Store the dummy-node-expanded DAG used by Sugiyama sweeps."""

    edge_index: torch.Tensor
    layers: list[list[int]]
    node_sizes: torch.Tensor
    edge_paths: list[list[int]]
    num_nodes: int


def layout_sugiyama(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    rank_sep: float = 1.0,
    node_sep: float = 1.0,
    layer_sep: Optional[float] = None,
    seed: int = 42,
    barycenter_passes: int = 24,
    trace_every: int = 0,
    return_edge_routes: bool = False,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, List[torch.Tensor]],
    Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
]:
    """Run classic Sugiyama layered layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list, shape ``[2, E]``. The algorithm is designed for DAGs.
        Cyclic inputs are handled conservatively by reversing detected DFS
        back edges before the layering phase.
    num_nodes : int
        Number of nodes.
    node_sizes : torch.Tensor, optional
        Node sizes ``[N, 2]`` for spacing. Default: point-like nodes.
    rank_sep : float
        Vertical spacing between layers.
    node_sep : float
        Horizontal spacing between nodes within a layer.
    layer_sep : float, optional
        Alias for ``rank_sep``. When provided, it overrides ``rank_sep``.
    seed : int
        Retained for API compatibility. Stable node-order tie handling is used
        during barycenter sweeps to better match igraph.
    barycenter_passes : int
        Number of up/down sweeps for crossing minimization.
    trace_every : int
        If greater than zero, record position snapshots during barycenter
        sweeps after every ``trace_every`` full passes.
    return_edge_routes : bool, default=False
        If ``True``, also return a routed polyline for each input edge. Long
        edges are expanded through dummy nodes during ordering and coordinate
        assignment, then converted back into per-edge point sequences aligned
        to the input edge order.

    Returns
    -------
    torch.Tensor or tuple
        Final positions ``[N, 2]`` by default. If ``trace_every > 0``, returns
        ``(positions, traces)``. If ``return_edge_routes`` is enabled, returns
        ``(positions, edge_routes)`` or ``(positions, traces, edge_routes)``,
        where each route tensor has shape ``[P, 2]``.

    Raises
    ------
    ValueError
        If ``edge_index`` or ``node_sizes`` have invalid shapes, or if
        ``num_nodes`` or ``trace_every`` is negative.
    """
    _validate_layout_inputs(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)
    if trace_every < 0:
        raise ValueError("trace_every must be non-negative")
    if layer_sep is not None:
        rank_sep = layer_sep

    output_device = edge_index.device
    if node_sizes is not None:
        output_device = node_sizes.device

    resolved_sizes = _resolve_node_sizes(node_sizes=node_sizes, num_nodes=num_nodes)
    acyclic_edges, reversed_edge_mask = _prepare_acyclic_edges(
        edge_index=edge_index,
        num_nodes=num_nodes,
    )
    layer_assignments = _longest_path_layering(edge_index=acyclic_edges, num_nodes=num_nodes)
    layer_assignments = _promote_layer_assignments(
        edge_index=acyclic_edges,
        layer_assignments=layer_assignments,
        num_nodes=num_nodes,
    )
    expanded_graph = _expand_long_edges_with_dummy_nodes(
        edge_index=acyclic_edges,
        layer_assignments=layer_assignments,
        node_sizes=resolved_sizes,
        num_original_nodes=num_nodes,
    )
    parents, children = _build_neighbor_lists(
        edge_index=expanded_graph.edge_index,
        num_nodes=expanded_graph.num_nodes,
    )
    ordered_layers, traces = _barycenter_ordering(
        layers=expanded_graph.layers,
        parents=parents,
        children=children,
        num_nodes=expanded_graph.num_nodes,
        num_passes=barycenter_passes,
        seed=seed,
        node_sizes=expanded_graph.node_sizes,
        rank_sep=rank_sep,
        node_sep=node_sep,
        trace_every=trace_every,
        output_device=output_device,
    )
    expanded_positions = _coordinate_assignment(
        layers=ordered_layers,
        parents=parents,
        children=children,
        node_sizes=expanded_graph.node_sizes,
        num_nodes=expanded_graph.num_nodes,
        rank_sep=rank_sep,
        node_sep=node_sep,
        output_device=output_device,
    )
    positions = expanded_positions[:num_nodes]
    visible_traces = [trace[:num_nodes] for trace in traces]

    if not return_edge_routes:
        if trace_every > 0:
            return positions, visible_traces
        return positions

    edge_routes = _build_edge_routes(
        positions=expanded_positions,
        edge_paths=expanded_graph.edge_paths,
        reversed_edge_mask=reversed_edge_mask,
        output_device=output_device,
    )
    if trace_every > 0:
        return positions, visible_traces, edge_routes
    return positions, edge_routes


def _validate_layout_inputs(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
) -> None:
    """Validate the public Sugiyama layout inputs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor expected to have shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    node_sizes : torch.Tensor, optional
        Optional node size tensor expected to have shape ``[N, 2]``.

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


def _resolve_node_sizes(node_sizes: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Return CPU node sizes for coordinate spacing.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Optional input node sizes.
    num_nodes : int
        Number of nodes in the graph.

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
    """Return a CPU ``edge_index`` with detected back edges reversed.

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

    back_edge_mask = detect_back_edges(edge_index_cpu, num_nodes)
    if back_edge_mask.any():
        return make_acyclic(edge_index_cpu, back_edge_mask), back_edge_mask.to(dtype=torch.bool)
    return edge_index_cpu, back_edge_mask.to(dtype=torch.bool)


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
) -> _ExpandedLayeredGraph:
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
    edge_paths: list[list[int]] = []
    next_dummy_index = num_original_nodes

    for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        source_layer = int(layer_assignments[source].item())
        target_layer = int(layer_assignments[target].item())
        path = [source]
        previous = source

        for layer_index in range(source_layer + 1, target_layer):
            dummy_index = next_dummy_index
            next_dummy_index += 1
            expanded_layers[layer_index].append(dummy_index)
            dummy_sizes.append([0.0, 0.0])
            expanded_sources.append(previous)
            expanded_targets.append(dummy_index)
            path.append(dummy_index)
            previous = dummy_index

        expanded_sources.append(previous)
        expanded_targets.append(target)
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
    return _ExpandedLayeredGraph(
        edge_index=expanded_edge_index,
        layers=expanded_layers,
        node_sizes=expanded_node_sizes,
        edge_paths=edge_paths,
        num_nodes=next_dummy_index,
    )


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


def _barycenter_ordering(
    layers: List[List[int]],
    parents: List[List[int]],
    children: List[List[int]],
    num_nodes: int,
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
    num_nodes : int
        Number of nodes.
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
        Device for emitted trace tensors.

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
                order_index=order_index,
            )
            ordered_layers[layer_idx].sort(key=lambda node: barycenters[node])
            order_index = _node_order_map(ordered_layers)

        for layer_idx in range(len(ordered_layers) - 2, -1, -1):
            barycenters = _neighbor_barycenters(
                nodes=ordered_layers[layer_idx],
                neighbors_by_node=children,
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
                    rank_sep=rank_sep,
                    node_sep=node_sep,
                    output_device=output_device,
                )
            )

    return ordered_layers, traces


def _neighbor_barycenters(
    nodes: Sequence[int],
    neighbors_by_node: Sequence[Sequence[int]],
    order_index: Dict[int, float],
) -> Dict[int, float]:
    """Compute barycenter values from already ordered neighboring layers.

    Parameters
    ----------
    nodes : sequence of int
        Nodes in the layer currently being sorted.
    neighbors_by_node : sequence of sequence of int
        Parent or child adjacency indexed by node.
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
    rank_sep: float,
    node_sep: float,
    output_device: torch.device,
) -> torch.Tensor:
    """Assign centered ``(x, y)`` coordinates from the layer ordering.

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
    positions = _initialize_layer_positions(
        layers=layers,
        node_sizes=node_sizes,
        num_nodes=num_nodes,
        rank_sep=rank_sep,
        node_sep=node_sep,
    )
    if num_nodes == 0:
        return positions.to(output_device)

    for _ in range(_COORDINATE_REFINEMENT_SWEEPS):
        current_x = positions[:, 0].clone()
        for layer_idx, layer_nodes in enumerate(layers):
            if not layer_nodes:
                continue

            desired_x = [
                _neighbor_mean_x(
                    node=node,
                    current_x=current_x,
                    parents=parents,
                    children=children,
                )
                for node in layer_nodes
            ]
            resolved_x = _resolve_layer_overlaps(
                layer_nodes=layer_nodes,
                desired_x=desired_x,
                node_sizes=node_sizes,
                node_sep=node_sep,
            )
            for index, node in enumerate(layer_nodes):
                positions[node, 0] = resolved_x[index]
                positions[node, 1] = float(layer_idx) * rank_sep

    return positions.to(output_device)


def _initialize_layer_positions(
    layers: Sequence[Sequence[int]],
    node_sizes: torch.Tensor,
    num_nodes: int,
    rank_sep: float,
    node_sep: float,
) -> torch.Tensor:
    """Create centered initial coordinates for the ordered layers.

    Parameters
    ----------
    layers : sequence of sequence of int
        Ordered nodes per layer.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]``.
    num_nodes : int
        Number of nodes.
    rank_sep : float
        Vertical distance between layers.
    node_sep : float
        Horizontal gap between node bounding boxes.

    Returns
    -------
    torch.Tensor
        Initial position tensor with shape ``[N, 2]``.
    """
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for layer_idx, layer_nodes in enumerate(layers):
        if not layer_nodes:
            continue

        y = float(layer_idx) * rank_sep
        total_width = sum(float(node_sizes[node, 0].item()) for node in layer_nodes)
        total_width += node_sep * float(max(len(layer_nodes) - 1, 0))
        x_cursor = -total_width / 2.0
        for node in layer_nodes:
            node_width = float(node_sizes[node, 0].item())
            positions[node, 0] = x_cursor + node_width / 2.0
            positions[node, 1] = y
            x_cursor += node_width + node_sep
    return positions


def _neighbor_mean_x(
    node: int,
    current_x: torch.Tensor,
    parents: Sequence[Sequence[int]],
    children: Sequence[Sequence[int]],
) -> float:
    """Average the adjacent-layer X coordinates for one node.

    Parameters
    ----------
    node : int
        Node id whose desired X position is being computed.
    current_x : torch.Tensor
        Current X coordinates with shape ``[N]``.
    parents : sequence of sequence of int
        Parent adjacency indexed by node id.
    children : sequence of sequence of int
        Child adjacency indexed by node id.

    Returns
    -------
    float
        Desired X coordinate from parent and child barycenters.
    """
    neighbor_x = [float(current_x[parent].item()) for parent in parents[node]]
    neighbor_x.extend(float(current_x[child].item()) for child in children[node])
    if not neighbor_x:
        return float(current_x[node].item())
    return sum(neighbor_x) / float(len(neighbor_x))


def _resolve_layer_overlaps(
    layer_nodes: Sequence[int],
    desired_x: Sequence[float],
    node_sizes: torch.Tensor,
    node_sep: float,
) -> List[float]:
    """Push nodes apart within a layer while preserving their order.

    Parameters
    ----------
    layer_nodes : sequence of int
        Nodes in their fixed within-layer order.
    desired_x : sequence of float
        Desired X coordinates for each node in ``layer_nodes``.
    node_sizes : torch.Tensor
        CPU node sizes ``[N, 2]``.
    node_sep : float
        Horizontal gap between node bounding boxes.

    Returns
    -------
    list of float
        Resolved X coordinates without overlaps.
    """
    if not layer_nodes:
        return []

    resolved_x = list(desired_x)
    for index in range(1, len(layer_nodes)):
        left_node = layer_nodes[index - 1]
        right_node = layer_nodes[index]
        min_gap = (
            float(node_sizes[left_node, 0].item()) + float(node_sizes[right_node, 0].item())
        ) / 2.0 + node_sep
        resolved_x[index] = max(resolved_x[index], resolved_x[index - 1] + min_gap)

    desired_center = sum(desired_x) / float(len(desired_x))
    resolved_center = sum(resolved_x) / float(len(resolved_x))
    shift = desired_center - resolved_center
    return [x + shift for x in resolved_x]
