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
import random
from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch

from dagua.layout.cycle import detect_back_edges, make_acyclic


def layout_sugiyama(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    rank_sep: float = 50.0,
    node_sep: float = 28.0,
    seed: int = 42,
    barycenter_passes: int = 24,
    trace_every: int = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
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
        Node sizes ``[N, 2]`` for spacing. Default: 20x20.
    rank_sep : float
        Vertical spacing between layers.
    node_sep : float
        Horizontal spacing between nodes within a layer.
    seed : int
        Random seed for deterministic barycenter tie-breaking.
    barycenter_passes : int
        Number of up/down sweeps for crossing minimization.
    trace_every : int
        If greater than zero, record position snapshots during barycenter
        sweeps after every ``trace_every`` full passes.

    Returns
    -------
    torch.Tensor or tuple
        Final positions ``[N, 2]``, or ``(positions, traces)`` if tracing.

    Raises
    ------
    ValueError
        If ``edge_index`` or ``node_sizes`` have invalid shapes, or if
        ``num_nodes`` is negative.
    """
    _validate_layout_inputs(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)

    output_device = edge_index.device
    if node_sizes is not None:
        output_device = node_sizes.device

    resolved_sizes = _resolve_node_sizes(node_sizes=node_sizes, num_nodes=num_nodes)
    acyclic_edges = _prepare_acyclic_edges(edge_index=edge_index, num_nodes=num_nodes)
    layer_assignments = _longest_path_layering(edge_index=acyclic_edges, num_nodes=num_nodes)
    layers = _group_nodes_by_layer(layer_assignments=layer_assignments, num_nodes=num_nodes)
    parents, children = _build_neighbor_lists(edge_index=acyclic_edges, num_nodes=num_nodes)
    ordered_layers, traces = _barycenter_ordering(
        layers=layers,
        parents=parents,
        children=children,
        num_nodes=num_nodes,
        num_passes=barycenter_passes,
        seed=seed,
        node_sizes=resolved_sizes,
        rank_sep=rank_sep,
        node_sep=node_sep,
        trace_every=trace_every,
        output_device=output_device,
    )
    positions = _coordinate_assignment(
        layers=ordered_layers,
        node_sizes=resolved_sizes,
        num_nodes=num_nodes,
        rank_sep=rank_sep,
        node_sep=node_sep,
        output_device=output_device,
    )

    if trace_every > 0:
        return positions, traces
    return positions


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
        return torch.full((num_nodes, 2), 20.0, dtype=torch.float32)
    return node_sizes.detach().to(device="cpu", dtype=torch.float32)


def _prepare_acyclic_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return a CPU ``edge_index`` with detected back edges reversed.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge list of shape ``[2, E]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[2, E]`` suitable for Kahn layering.
    """
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() == 0:
        return edge_index_cpu

    back_edge_mask = detect_back_edges(edge_index_cpu, num_nodes)
    if back_edge_mask.any():
        return make_acyclic(edge_index_cpu, back_edge_mask)
    return edge_index_cpu


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
        Seed used for deterministic tie-breaking.
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

    rng = random.Random(seed)
    tie_break = {node: rng.random() for node in range(num_nodes)}
    traces: List[torch.Tensor] = []

    for pass_num in range(num_passes):
        order_index = _node_order_map(ordered_layers)

        for layer_idx in range(1, len(ordered_layers)):
            barycenters = _neighbor_barycenters(
                nodes=ordered_layers[layer_idx],
                neighbors_by_node=parents,
                order_index=order_index,
            )
            ordered_layers[layer_idx].sort(key=lambda node: (barycenters[node], tie_break[node]))
            order_index = _node_order_map(ordered_layers)

        for layer_idx in range(len(ordered_layers) - 2, -1, -1):
            barycenters = _neighbor_barycenters(
                nodes=ordered_layers[layer_idx],
                neighbors_by_node=children,
                order_index=order_index,
            )
            ordered_layers[layer_idx].sort(key=lambda node: (barycenters[node], tie_break[node]))
            order_index = _node_order_map(ordered_layers)

        if trace_every > 0 and (pass_num + 1) % trace_every == 0:
            traces.append(
                _coordinate_assignment(
                    layers=ordered_layers,
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

    return positions.to(output_device)
