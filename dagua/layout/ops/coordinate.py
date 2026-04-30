"""Coordinate assignment operations for layered and tree layouts."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, Dict, List, Optional, Sequence, Set, Tuple

import torch

from dagua.layout.graph_classify import (
    GraphStructure as TopologyGraphStructure,
)
from dagua.layout.graph_classify import classify_graph
from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import (
    build_directed_adjacency as _build_directed_adjacency,
)
from dagua.layout.ops.graph_utils import (
    build_undirected_adjacency as _build_undirected_adjacency,
)
from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_NO_SHIFT = float("inf")
_POSITION_OUTPUT_DIM = 2
_NODE_SIZE_SPACING_MULTIPLIER = 1.5
_WALKER_ROOT_NUMBER = 1
_WALKER_BASE_DISTANCE = 1.0
_COMPONENT_PADDING = 1.0
_BRANDES_KOEPF_APPLIED_KEY = "brandes_koepf_horizontal_refine_applied"


def _resolve_node_sizes(node_sizes: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Return CPU node sizes for coordinate spacing.

    Parameters
    ----------
    node_sizes : torch.Tensor | None
        Optional input node sizes.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        CPU float tensor with shape ``[N, 2]``.
    """
    if node_sizes is None:
        return torch.zeros((num_nodes, _POSITION_OUTPUT_DIM), dtype=torch.float32)
    return node_sizes.detach().to(device="cpu", dtype=torch.float32)


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
        oriented_layers = _transform_layers(
            layers=layers,
            reverse_layers=reverse_layers,
            reverse_within=reverse_within,
        )
        rank_of, pos_of = _layer_position_maps(layers=oriented_layers, num_nodes=num_nodes)
        predecessor_source = children if reverse_layers else parents
        predecessors = _ordered_transformed_neighbors(
            neighbors_by_node=predecessor_source,
            pos_of=pos_of,
        )
        conflicts = _find_type1_conflicts(
            layers=oriented_layers,
            predecessors=predecessors,
            pos_of=pos_of,
            dummy_mask=dummy_mask,
        )
        root, align = _vertical_alignment(
            layers=oriented_layers,
            predecessors=predecessors,
            pos_of=pos_of,
            conflicts=conflicts,
            num_nodes=num_nodes,
        )
        compacted = _horizontal_compaction(
            layers=oriented_layers,
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
    int | None
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


def _target_device(problem: LayoutProblem, state: SolveState) -> torch.device:
    """Resolve the device used for persisted position tensors.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.device
        Device for the updated ``state.pos`` tensor.
    """
    if state.pos is not None:
        return state.pos.device
    if problem.node_sizes is not None:
        return problem.node_sizes.device
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
        raise ValueError("state.layers must be populated before coordinate ops run")
    if layers.ndim != 1 or layers.shape[0] != num_nodes:
        raise ValueError(f"state.layers must have shape [{num_nodes}]")
    return layers.detach().to(device="cpu", dtype=torch.long)


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


def _ordered_layers_from_state(layers: torch.Tensor, ordering: torch.Tensor) -> List[List[int]]:
    """Build ordered layer lists from per-node layer and ordering tensors.

    Parameters
    ----------
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    ordering : torch.Tensor
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
        layer_nodes.sort(key=lambda node_idx: (int(ordering[node_idx].item()), node_idx))
    return ordered_layers


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


def _has_strict_forward_layering(edge_index: torch.Tensor, layers: torch.Tensor) -> bool:
    """Return whether every edge advances in the current layer assignment.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` when every edge satisfies ``layers[src] < layers[dst]``.
    """
    if edge_index.numel() == 0:
        return False
    layer_deltas = layers[edge_index[1]] - layers[edge_index[0]]
    return bool(torch.all(layer_deltas > 0).item())


def _ordering_from_current_x(layers: torch.Tensor, pos: torch.Tensor) -> torch.Tensor:
    """Build in-layer ordering ranks from current x coordinates.

    Parameters
    ----------
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        CPU ordering tensor with shape ``[N]``.
    """
    if layers.numel() == 0:
        return torch.zeros((0,), dtype=torch.long)

    pos_cpu = pos.detach().to(device="cpu", dtype=torch.float32)
    max_layer = int(layers.max().item())
    ordering = torch.zeros((layers.shape[0],), dtype=torch.long)
    for layer_index in range(max_layer + 1):
        layer_nodes = torch.where(layers == layer_index)[0].tolist()
        layer_nodes.sort(key=lambda node_idx: (float(pos_cpu[node_idx, 0].item()), node_idx))
        for position, node_idx in enumerate(layer_nodes):
            ordering[node_idx] = position
    return ordering


def _resolve_topology_structure(
    structure: Optional[TopologyGraphStructure],
    edge_index: torch.Tensor,
    layers: torch.Tensor,
    num_nodes: int,
) -> TopologyGraphStructure:
    """Return the topology structure used by the BK gate.

    Parameters
    ----------
    structure : TopologyGraphStructure | None
        Optional pre-classified graph structure.
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    TopologyGraphStructure
        Structure used for family-based gating.
    """
    if structure is not None:
        return structure
    return classify_graph(edge_index=edge_index, num_nodes=num_nodes, layer_assignments=layers)


def _active_coordinate_graph(
    problem: LayoutProblem,
    state: SolveState,
) -> tuple[int, torch.Tensor, torch.Tensor]:
    """Return the graph currently active for coordinate assignment.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable original graph inputs.
    state : SolveState
        Mutable solve state that may carry ``extras["expanded_graph"]``.

    Returns
    -------
    tuple[int, torch.Tensor, torch.Tensor]
        Active node count, edge tensor, and CPU node sizes. The expanded graph
        is used only when active layer tensors already match its node count.
    """
    expanded_graph = state.extras.get("expanded_graph")
    if (
        expanded_graph is not None
        and state.layers is not None
        and int(getattr(expanded_graph, "num_nodes", -1)) == int(state.layers.shape[0])
    ):
        num_nodes = int(expanded_graph.num_nodes)
        return (
            num_nodes,
            expanded_graph.edge_index,
            _resolve_node_sizes(expanded_graph.node_sizes, num_nodes),
        )
    return (
        problem.num_nodes,
        problem.edge_index,
        _resolve_node_sizes(problem.node_sizes, problem.num_nodes),
    )


def _should_apply_brandes_koepf_refine(
    structure: TopologyGraphStructure,
    edge_index: torch.Tensor,
    layers: torch.Tensor,
    num_nodes: int,
    min_layers: int,
) -> bool:
    """Return whether native BK horizontal refinement should run.

    Parameters
    ----------
    structure : TopologyGraphStructure
        Classified graph structure. Only stable family metadata is trusted.
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    num_nodes : int
        Number of graph nodes.
    min_layers : int
        Minimum layer count required for BK to be worthwhile.

    Returns
    -------
    bool
        ``True`` when the graph is a conservative BK candidate.

    Notes
    -----
    ``classify_graph().is_acyclic`` and ``classify_graph().num_components``
    are intentionally not used here because those fields are optimized for
    tree/forest shortcut detection, not exact general-DAG gating.
    """
    if num_nodes == 0:
        return False

    num_layers = int(layers.max().item()) + 1 if layers.numel() > 0 else 0
    if num_layers < min_layers:
        return False

    return _has_strict_forward_layering(edge_index=edge_index, layers=layers)


def _node_spacing(
    node_sizes: Optional[torch.Tensor],
    axis: int,
    default: float,
) -> float:
    """Estimate sibling and layer spacing from optional node sizes.

    Parameters
    ----------
    node_sizes : torch.Tensor | None
        Optional node-size tensor with shape ``[N, 2]``.
    axis : int
        Axis index to read from each node size.
    default : float
        Fallback spacing value when node sizes are missing or empty.

    Returns
    -------
    float
        Spacing multiplier for the requested axis.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return default
    max_size = float(node_sizes.to(dtype=torch.float32, device="cpu")[:, axis].max().item())
    return max(max_size * _NODE_SIZE_SPACING_MULTIPLIER, default)


def _root_candidates(edge_index: torch.Tensor, num_nodes: int) -> list[int]:
    """Order BFS roots by indegree and index for deterministic output.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[int]
        BFS roots sorted by ``(indegree != 0, indegree, node_id)``.
    """
    indegree = [0] * num_nodes
    if edge_index.numel() > 0:
        targets = edge_index.detach().to(device="cpu", dtype=torch.long)[1].tolist()
        for target in targets:
            indegree[target] += 1
    return sorted(
        range(num_nodes),
        key=lambda node_idx: (indegree[node_idx] != 0, indegree[node_idx], node_idx),
    )


def _rt_mode_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    traversal_mode: str,
) -> list[list[tuple[int, float]]]:
    """Build adjacency for an RT traversal mode.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    traversal_mode : str
        One of ``"out"``, ``"in"``, or ``"all"``.

    Returns
    -------
    list[list[tuple[int, float]]]
        Sorted adjacency list.
    """
    if traversal_mode == "all":
        return _build_undirected_adjacency(edge_index=edge_index, num_nodes=num_nodes)
    if traversal_mode == "out":
        return _build_directed_adjacency(edge_index=edge_index, num_nodes=num_nodes)
    if traversal_mode == "in":
        reversed_edges = edge_index[[1, 0], :] if edge_index.numel() > 0 else edge_index
        return _build_directed_adjacency(edge_index=reversed_edges, num_nodes=num_nodes)
    raise ValueError(f"Unsupported Reingold-Tilford traversal_mode: {traversal_mode!r}")


def _igraph_fidelity_root_candidates(
    edge_index: torch.Tensor,
    num_nodes: int,
    traversal_mode: str,
) -> list[int]:
    """Order RT roots with igraph-like reachability semantics.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    traversal_mode : str
        One of ``"out"``, ``"in"``, or ``"all"``.

    Returns
    -------
    list[int]
        Candidate roots ordered by directed reachability, deduped degree, and
        node id.
    """
    traversal_adjacency = _rt_mode_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        traversal_mode=traversal_mode,
    )
    reach_counts: list[int] = []
    for root in range(num_nodes):
        reached = {root}
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor, _ in traversal_adjacency[node]:
                if neighbor in reached:
                    continue
                reached.add(neighbor)
                queue.append(neighbor)
        reach_counts.append(len(reached))
    return sorted(
        range(num_nodes),
        key=lambda candidate: (
            -reach_counts[candidate],
            -len(traversal_adjacency[candidate]),
            candidate,
        ),
    )


def _bfs_forest(
    edge_index: torch.Tensor,
    num_nodes: int,
    traversal_mode: str = "all",
    root_candidates: Optional[list[int]] = None,
    root_depths: Optional[dict[int, int]] = None,
) -> tuple[list[int], list[list[int]], list[int]]:
    """Build a deterministic BFS forest from a possibly directed graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    traversal_mode : str, default="all"
        Edge direction mode used while expanding the BFS tree.
    root_candidates : list[int] | None, default=None
        Optional preordered root candidates. ``None`` uses Dagua's historical
        indegree ordering.
    root_depths : dict[int, int] | None, default=None
        Optional starting depth per root, used to mirror igraph ``rootlevel``
        fixtures without rebuilding the augmented graph.

    Returns
    -------
    tuple[list[int], list[list[int]], list[int]]
        Forest roots, child lists, and BFS depth per node.
    """
    adjacency = _rt_mode_adjacency(
        edge_index=edge_index,
        num_nodes=num_nodes,
        traversal_mode=traversal_mode,
    )
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    depths = [0] * num_nodes
    visited = [False] * num_nodes
    roots: list[int] = []

    roots_to_try = (
        _root_candidates(edge_index=edge_index, num_nodes=num_nodes)
        if root_candidates is None
        else root_candidates
    )
    for root in roots_to_try:
        if visited[root]:
            continue
        roots.append(root)
        visited[root] = True
        if root_depths is not None:
            depths[root] = root_depths.get(root, 0)
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor, _ in adjacency[node]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                depths[neighbor] = depths[node] + 1
                children[node].append(neighbor)
                queue.append(neighbor)

    return roots, children, depths


def _validate_rt_roots(roots: Optional[Sequence[int]], num_nodes: int) -> Optional[list[int]]:
    """Validate optional Reingold-Tilford root vertices.

    Parameters
    ----------
    roots : sequence of int | None
        Explicit root vertex ids, or ``None`` for automatic selection.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[int] | None
        Deduplicated explicit roots, preserving caller order.

    Raises
    ------
    ValueError
        If any root id is outside ``[0, num_nodes)``.
    """
    if roots is None:
        return None

    validated_roots: list[int] = []
    seen_roots: set[int] = set()
    for root in roots:
        root_idx = int(root)
        if root_idx < 0 or root_idx >= num_nodes:
            raise ValueError(f"Reingold-Tilford root out of range: {root_idx}")
        if root_idx in seen_roots:
            continue
        seen_roots.add(root_idx)
        validated_roots.append(root_idx)
    return validated_roots


def _validate_rt_rootlevels(
    rootlevels: Optional[Sequence[int]],
    roots: Optional[list[int]],
) -> dict[int, int]:
    """Validate optional igraph-style root levels.

    Parameters
    ----------
    rootlevels : sequence of int | None
        Explicit depth per supplied root.
    roots : list[int] | None
        Validated roots that root levels correspond to.

    Returns
    -------
    dict[int, int]
        Mapping from root vertex id to requested starting depth.

    Raises
    ------
    ValueError
        If levels are supplied without roots, have the wrong length, or are
        negative.
    """
    if rootlevels is None:
        return {}
    if roots is None:
        raise ValueError("Reingold-Tilford rootlevel requires explicit roots.")
    if len(rootlevels) != len(roots):
        raise ValueError("Reingold-Tilford rootlevel length must match roots length.")

    root_depths: dict[int, int] = {}
    for root, level in zip(roots, rootlevels):
        level_idx = int(level)
        if level_idx < 0:
            raise ValueError(f"Reingold-Tilford rootlevel must be non-negative: {level_idx}")
        root_depths[root] = level_idx
    return root_depths


@dataclass
class _WalkerNode:
    """Walker bookkeeping node for Buchheim-style tree coordinate assignment."""

    node_idx: int
    depth: int
    number: int
    parent: Optional["_WalkerNode"] = None
    children: list["_WalkerNode"] = field(default_factory=list)
    prelim: float = 0.0
    mod: float = 0.0
    shift: float = 0.0
    change: float = 0.0
    thread: Optional["_WalkerNode"] = None
    ancestor: Optional["_WalkerNode"] = None
    _leftmost_sibling: Optional["_WalkerNode"] = None

    def __post_init__(self) -> None:
        """Initialize the default ancestor to the current node."""
        self.ancestor = self

    def left(self) -> Optional["_WalkerNode"]:
        """Return the first leftward contour node."""
        if self.thread is not None:
            return self.thread
        if self.children:
            return self.children[0]
        return None

    def right(self) -> Optional["_WalkerNode"]:
        """Return the first rightward contour node."""
        if self.thread is not None:
            return self.thread
        if self.children:
            return self.children[-1]
        return None

    def left_brother(self) -> Optional["_WalkerNode"]:
        """Return the immediate left sibling, if any."""
        if self.parent is None:
            return None
        previous_sibling: Optional[_WalkerNode] = None
        for sibling in self.parent.children:
            if sibling is self:
                return previous_sibling
            previous_sibling = sibling
        return None

    def leftmost_sibling(self) -> Optional["_WalkerNode"]:
        """Return cached leftmost sibling in the current sibling set."""
        if self._leftmost_sibling is None and self.parent is not None:
            first_child = self.parent.children[0]
            if first_child is not self:
                self._leftmost_sibling = first_child
        return self._leftmost_sibling


def _build_walker_tree(
    root_idx: int,
    children: list[list[int]],
    depth: int,
    number: int,
    subtree_nodes: list[int],
    parent: Optional[_WalkerNode] = None,
) -> _WalkerNode:
    """Build tree state recursively for one BFS-rooted component."""
    subtree_nodes.append(root_idx)
    walker_node = _WalkerNode(
        node_idx=root_idx,
        depth=depth,
        number=number,
        parent=parent,
    )
    for child_number, child_idx in enumerate(children[root_idx], start=1):
        walker_node.children.append(
            _build_walker_tree(
                root_idx=child_idx,
                children=children,
                depth=depth + 1,
                number=child_number,
                subtree_nodes=subtree_nodes,
                parent=walker_node,
            )
        )
    return walker_node


def _walker_ancestor(
    left_inner: _WalkerNode,
    node: _WalkerNode,
    default_ancestor: _WalkerNode,
) -> _WalkerNode:
    """Resolve the representative ancestor for non-sibling conflict adjustment."""
    if node.parent is None:
        return default_ancestor
    candidate = left_inner.ancestor or default_ancestor
    if candidate in node.parent.children:
        return candidate
    return default_ancestor


def _move_subtree(left_subtree: _WalkerNode, right_subtree: _WalkerNode, shift: float) -> None:
    """Apply the deferred shift across a subtree interval."""
    subtree_count = right_subtree.number - left_subtree.number
    if subtree_count <= 0:
        return
    shift_per_subtree = shift / float(subtree_count)
    right_subtree.change -= shift_per_subtree
    right_subtree.shift += shift
    left_subtree.change += shift_per_subtree
    right_subtree.prelim += shift
    right_subtree.mod += shift


def _execute_shifts(node: _WalkerNode) -> None:
    """Apply accumulated sibling shifts from right to left."""
    shift = 0.0
    change = 0.0
    for child in reversed(node.children):
        child.prelim += shift
        child.mod += shift
        change += child.change
        shift += child.shift + change


def _apportion(node: _WalkerNode, default_ancestor: _WalkerNode, distance: float) -> _WalkerNode:
    """Resolve contour overlaps between the current node and previous siblings."""
    left_brother = node.left_brother()
    if left_brother is None:
        return default_ancestor

    inner_right = node
    outer_right = node
    inner_left = left_brother
    outer_left = node.leftmost_sibling()
    if outer_left is None:
        return default_ancestor

    sum_inner_right = node.mod
    sum_outer_right = node.mod
    sum_inner_left = inner_left.mod
    sum_outer_left = outer_left.mod

    while inner_left.right() is not None and inner_right.left() is not None:
        next_inner_left = inner_left.right()
        next_inner_right = inner_right.left()
        next_outer_left = outer_left.left()
        next_outer_right = outer_right.right()
        if (
            next_inner_left is None
            or next_inner_right is None
            or next_outer_left is None
            or next_outer_right is None
        ):
            break

        inner_left = next_inner_left
        inner_right = next_inner_right
        outer_left = next_outer_left
        outer_right = next_outer_right
        outer_right.ancestor = node

        shift = (inner_left.prelim + sum_inner_left) - (inner_right.prelim + sum_inner_right)
        shift += distance
        if shift > 0.0:
            ancestor = _walker_ancestor(inner_left, node, default_ancestor)
            _move_subtree(ancestor, node, shift)
            sum_inner_right += shift
            sum_outer_right += shift

        sum_inner_left += inner_left.mod
        sum_inner_right += inner_right.mod
        sum_outer_left += outer_left.mod
        sum_outer_right += outer_right.mod

    if inner_left.right() is not None and outer_right.right() is None:
        outer_right.thread = inner_left.right()
        outer_right.mod += sum_inner_left - sum_outer_right
    else:
        if inner_right.left() is not None and outer_left.left() is None:
            outer_left.thread = inner_right.left()
            outer_left.mod += sum_inner_right - sum_outer_left
        default_ancestor = node

    return default_ancestor


def _first_walk(node: _WalkerNode, distance: float) -> None:
    """Assign preliminary x-values in a post-order pass."""
    if not node.children:
        left_brother = node.left_brother()
        node.prelim = left_brother.prelim + distance if left_brother is not None else 0.0
        return

    default_ancestor = node.children[0]
    for child in node.children:
        _first_walk(child, distance=distance)
        default_ancestor = _apportion(child, default_ancestor, distance=distance)

    _execute_shifts(node)
    midpoint = 0.5 * (node.children[0].prelim + node.children[-1].prelim)
    left_brother = node.left_brother()
    if left_brother is None:
        node.prelim = midpoint
        return

    node.prelim = left_brother.prelim + distance
    node.mod = node.prelim - midpoint


def _second_walk(
    node: _WalkerNode,
    modifier: float,
    coordinates: dict[int, float],
) -> tuple[float, float]:
    """Propagate accumulated modifiers top-down to absolute x coordinates."""
    x_coordinate = node.prelim + modifier
    coordinates[node.node_idx] = x_coordinate
    min_x = x_coordinate
    max_x = x_coordinate
    for child in node.children:
        child_min, child_max = _second_walk(
            child,
            modifier=modifier + node.mod,
            coordinates=coordinates,
        )
        min_x = min(min_x, child_min)
        max_x = max(max_x, child_max)
    return min_x, max_x


def _assign_preliminary_x(
    root_idx: int,
    children: list[list[int]],
    depths: list[int],
    preliminary_x: list[float],
    component_offset: float,
    component_gap: float,
) -> float:
    """Generate absolute x positions for one BFS-rooted component."""
    subtree_nodes: list[int] = []
    root = _build_walker_tree(
        root_idx=root_idx,
        children=children,
        depth=depths[root_idx],
        number=_WALKER_ROOT_NUMBER,
        subtree_nodes=subtree_nodes,
    )
    _first_walk(root, distance=_WALKER_BASE_DISTANCE)

    coordinates: dict[int, float] = {}
    min_x, max_x = _second_walk(root, modifier=0.0, coordinates=coordinates)
    normalization_shift = component_offset - min_x
    for node_idx in subtree_nodes:
        preliminary_x[node_idx] = coordinates[node_idx] + normalization_shift

    return component_offset + (max_x - min_x) + _COMPONENT_PADDING + component_gap


@dataclass(frozen=True)
class ReingoldTilfordTreeConfig:
    """Configuration for :class:`ReingoldTilfordTree`.

    Parameters
    ----------
    sibling_sep : float | None, default=None
        Horizontal sibling spacing in tree units. ``None`` resolves from
        ``problem.node_sizes[:, 0]`` with a ``1.5x`` multiplier and a ``1.0``
        minimum floor.
    layer_sep : float | None, default=None
        Vertical layer spacing in tree units. ``None`` resolves from
        ``problem.node_sizes[:, 1]`` with a ``1.5x`` multiplier and a ``1.5``
        minimum floor.
    component_gap : float | None, default=None
        Extra spacing between disconnected components.
    default_component_gap_multiplier : float, default=2.0
        Multiplier applied to the resolved sibling spacing when
        ``component_gap`` is left unset.
    horizontal : bool, default=False
        Rotate the output so depth maps to x when ``True``.
    fidelity_mode : str | None, default=None
        Optional compatibility mode. ``"igraph"`` uses unit spacing,
        mode-sensitive BFS traversal, and synthetic-root packing for closer
        parity with igraph's RT implementation.
    traversal_mode : str, default="out"
        Edge direction mode used by ``fidelity_mode="igraph"``. Supported
        values are ``"out"``, ``"in"``, and ``"all"``.
    roots : sequence of int | None, default=None
        Optional explicit root vertices for igraph-style controlled tests.
    rootlevel : sequence of int | None, default=None
        Optional depth per explicit root.
    center_output : bool | None, default=None
        Whether to mean-center final coordinates. ``None`` preserves
        historical centering except in igraph fidelity mode.
    output_scale : float | None, default=None
        Uniform scale applied to final coordinates. ``None`` uses igraph's
        adapter scale in igraph fidelity mode and ``1.0`` otherwise.
    recursion_limit_multiplier : int, default=2
        Multiplier used when raising Python's recursion limit for large trees.
    """

    sibling_sep: Optional[float] = None
    layer_sep: Optional[float] = None
    component_gap: Optional[float] = None
    default_component_gap_multiplier: float = 2.0
    horizontal: bool = False
    fidelity_mode: Optional[str] = None
    traversal_mode: str = "out"
    roots: Optional[Sequence[int]] = None
    rootlevel: Optional[Sequence[int]] = None
    center_output: Optional[bool] = None
    output_scale: Optional[float] = None
    recursion_limit_multiplier: int = 2


@dataclass(frozen=True)
class BrandesKopf4PassConfig:
    """Configuration for :class:`BrandesKopf4Pass`.

    Parameters
    ----------
    node_sep : float, default=1.0
        Horizontal gap between neighboring nodes.
    rank_sep : float, default=1.0
        Vertical gap between consecutive layers.
    """

    node_sep: float = 1.0
    rank_sep: float = 1.0


@dataclass(frozen=True)
class BrandesKoepfHorizontalRefineConfig:
    """Configuration for :class:`BrandesKoepfHorizontalRefine`.

    Parameters
    ----------
    node_sep : float, default=1.0
        Horizontal separation used by the BK compaction pass.
    min_layers : int, default=6
        Minimum layer count before BK refinement is allowed.
    enabled : bool, default=True
        Master kill switch for rollback and A/B tests.
    structure : TopologyGraphStructure | None, default=None
        Optional pre-classified graph structure from native config resolution.
    """

    node_sep: float = 1.0
    min_layers: int = 6
    enabled: bool = True
    structure: Optional[TopologyGraphStructure] = None


@dataclass(frozen=True)
class BucheimWalkerTreeConfig:
    """Configuration for :class:`BucheimWalkerTree`.

    Parameters
    ----------
    sibling_sep : float, default=1.0
        Horizontal spacing multiplier between neighboring siblings.
    layer_sep : float, default=1.5
        Vertical spacing between consecutive tree depths.
    component_gap : float, default=2.0
        Horizontal gap between disconnected tree components.
    recursion_limit_multiplier : int, default=2
        Multiplier used when raising Python's recursion limit for large trees.
    """

    sibling_sep: float = 1.0
    layer_sep: float = 1.5
    component_gap: float = 2.0
    recursion_limit_multiplier: int = 2


@register_op
class BrandesKoepfHorizontalRefine(Op):
    """Reassign x coordinates with Brandes-Koepf while preserving y."""

    name: ClassVar[str] = "brandes_koepf_horizontal_refine"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = ("pos", "layers")
    writes: ClassVar[Tuple[str, ...]] = ("pos", "ordering", "extras")
    requires: ClassVar[Tuple[str, ...]] = ("pos", "layers")
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[BrandesKoepfHorizontalRefineConfig] = None) -> None:
        """Store the horizontal-refinement configuration.

        Parameters
        ----------
        config : BrandesKoepfHorizontalRefineConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BrandesKoepfHorizontalRefineConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run x-only BK compaction when the native DAG gate allows it.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ``pos`` shaped ``[N, 2]`` and
            ``layers`` shaped ``[N]``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            State with x coordinates refined when eligible. Y coordinates are
            copied through exactly.
        """
        del ctx

        state.extras[_BRANDES_KOEPF_APPLIED_KEY] = False
        if not self.config.enabled:
            return state
        if state.pos is None:
            raise ValueError("state.pos must be populated before BK horizontal refine")

        active_num_nodes, active_edge_index, active_node_sizes = _active_coordinate_graph(
            problem=problem,
            state=state,
        )
        layers_cpu = _validate_layers(state.layers, active_num_nodes)
        edge_index_cpu = _validate_edge_index(active_edge_index, active_num_nodes)
        structure = _resolve_topology_structure(
            structure=self.config.structure or problem.structure,
            edge_index=edge_index_cpu,
            layers=layers_cpu,
            num_nodes=active_num_nodes,
        )
        if not _should_apply_brandes_koepf_refine(
            structure=structure,
            edge_index=edge_index_cpu,
            layers=layers_cpu,
            num_nodes=active_num_nodes,
            min_layers=self.config.min_layers,
        ):
            return state

        ordering_cpu = _ordering_from_current_x(layers=layers_cpu, pos=state.pos)
        ordered_layers = _ordered_layers_from_state(layers_cpu, ordering_cpu)
        parents, children = _layered_neighbors_from_edges(
            edge_index=edge_index_cpu,
            layers=layers_cpu,
            num_nodes=active_num_nodes,
        )
        x_coordinates = _brandes_koepf_x_positions(
            layers=ordered_layers,
            parents=parents,
            children=children,
            node_sizes=active_node_sizes,
            num_nodes=active_num_nodes,
            num_original_nodes=problem.num_nodes,
            node_sep=self.config.node_sep,
        )

        refined_pos = state.pos.detach().clone()
        refined_pos[:, 0] = torch.tensor(
            x_coordinates,
            dtype=refined_pos.dtype,
            device=refined_pos.device,
        )
        state.pos = refined_pos
        state.ordering = ordering_cpu.to(device=refined_pos.device)
        state.extras[_BRANDES_KOEPF_APPLIED_KEY] = True
        return state


@register_op
class BrandesKopf4Pass(Op):
    """Assign layered coordinates via four-pass Brandes-Kopf horizontal compaction."""

    name: ClassVar[str] = "brandes_kopf_4pass"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = ("layers", "ordering")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("layers", "ordering")
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[BrandesKopf4PassConfig] = None) -> None:
        """Store the coordinate-assignment configuration.

        Parameters
        ----------
        config : BrandesKopf4PassConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BrandesKopf4PassConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Convert a layered ordering into concrete node coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ``layers`` and ``ordering`` populated.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with ``pos`` populated.
        """
        del ctx

        active_num_nodes, active_edge_index, active_node_sizes = _active_coordinate_graph(
            problem=problem,
            state=state,
        )
        layers_cpu = _validate_layers(state.layers, active_num_nodes)
        ordering_cpu = _validate_ordering(state.ordering, active_num_nodes)
        edge_index_cpu = _validate_edge_index(active_edge_index, active_num_nodes)
        ordered_layers = _ordered_layers_from_state(layers_cpu, ordering_cpu)
        parents, children = _layered_neighbors_from_edges(
            edge_index=edge_index_cpu,
            layers=layers_cpu,
            num_nodes=active_num_nodes,
        )

        positions = torch.zeros((active_num_nodes, _POSITION_OUTPUT_DIM), dtype=torch.float32)
        if active_num_nodes > 0:
            # The four orientation passes balance each other, so keep the
            # Brandes-Kopf output in local CPU tensors until the final transfer.
            x_coordinates = _brandes_koepf_x_positions(
                layers=ordered_layers,
                parents=parents,
                children=children,
                node_sizes=active_node_sizes,
                num_nodes=active_num_nodes,
                num_original_nodes=problem.num_nodes,
                node_sep=self.config.node_sep,
            )
            positions[:, 0] = torch.tensor(x_coordinates, dtype=torch.float32)
            # Layer indices already encode vertical rank, so only the configured
            # separation factor is needed here.
            positions[:, 1] = layers_cpu.to(dtype=torch.float32) * self.config.rank_sep

        state.pos = positions.to(device=_target_device(problem, state))
        return state


@register_op
class BucheimWalkerTree(Op):
    """Lay out the graph as a tidy BFS forest using Buchheim's linear-time algorithm."""

    name: ClassVar[str] = "bucheim_walker_tree"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ()
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[BucheimWalkerTreeConfig] = None) -> None:
        """Store the tree-layout configuration.

        Parameters
        ----------
        config : BucheimWalkerTreeConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BucheimWalkerTreeConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Assign tidy tree coordinates directly from the graph topology.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with ``pos`` populated.
        """
        del ctx

        edge_index_cpu = _validate_edge_index(problem.edge_index, problem.num_nodes)
        target_device = _target_device(problem, state)
        if problem.num_nodes == 0:
            state.pos = torch.empty(
                (0, _POSITION_OUTPUT_DIM), dtype=torch.float32, device=target_device
            )
            return state

        # Walker-style tree layout still uses recursion internally, so raise
        # the limit defensively for large forests before descending.
        sys.setrecursionlimit(
            max(
                sys.getrecursionlimit(),
                problem.num_nodes * self.config.recursion_limit_multiplier,
            )
        )
        roots, children, depths = _bfs_forest(
            edge_index=edge_index_cpu,
            num_nodes=problem.num_nodes,
        )

        preliminary_x = [0.0] * problem.num_nodes
        next_component_offset = 0.0
        for root in roots:
            next_component_offset = _assign_preliminary_x(
                root_idx=root,
                children=children,
                depths=depths,
                preliminary_x=preliminary_x,
                component_offset=next_component_offset,
                component_gap=self.config.component_gap,
            )

        positions = torch.zeros((problem.num_nodes, _POSITION_OUTPUT_DIM), dtype=torch.float32)
        for node_idx in range(problem.num_nodes):
            positions[node_idx, 0] = float(preliminary_x[node_idx]) * self.config.sibling_sep
            positions[node_idx, 1] = float(depths[node_idx]) * self.config.layer_sep

        # Re-center the whole forest so disconnected components remain balanced
        # around the origin like the historical tree ops.
        positions -= positions.mean(dim=0, keepdim=True)
        state.pos = positions.to(device=target_device)
        return state


@register_op
class ReingoldTilfordTree(Op):
    """Assign Reingold-Tilford tree coordinates with optional size-aware spacing."""

    name: ClassVar[str] = "reingold_tilford_tree"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = ()
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ()
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[ReingoldTilfordTreeConfig] = None) -> None:
        """Store the tree configuration.

        Parameters
        ----------
        config : ReingoldTilfordTreeConfig | None, optional
            Optional op configuration.
        """
        self.config = config or ReingoldTilfordTreeConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute exact Reingold-Tilford coordinates for one problem.

        Parameters
        ----------
        problem : LayoutProblem
            Input graph and optional node size hints.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context (unused).

        Returns
        -------
        SolveState
            Updated state with ``state.pos`` set.
        """
        del ctx

        output_device = _layout_device(edge_index=problem.edge_index, node_sizes=problem.node_sizes)
        edge_index_cpu = _validate_edge_index(problem.edge_index, problem.num_nodes)

        if problem.num_nodes == 0:
            state.pos = torch.empty(
                (0, _POSITION_OUTPUT_DIM), dtype=torch.float32, device=output_device
            )
            return state

        fidelity_mode = self.config.fidelity_mode
        if fidelity_mode not in {None, "igraph"}:
            raise ValueError(f"Unsupported Reingold-Tilford fidelity_mode: {fidelity_mode!r}")
        if self.config.traversal_mode not in {"out", "in", "all"}:
            raise ValueError(
                f"Unsupported Reingold-Tilford traversal_mode: {self.config.traversal_mode!r}"
            )

        if fidelity_mode == "igraph":
            sibling_sep = 1.0 if self.config.sibling_sep is None else self.config.sibling_sep
            layer_sep = 1.0 if self.config.layer_sep is None else self.config.layer_sep
            component_gap = 0.0 if self.config.component_gap is None else self.config.component_gap
            center_output = (
                False if self.config.center_output is None else self.config.center_output
            )
            output_scale = 50.0 if self.config.output_scale is None else self.config.output_scale
        else:
            # When explicit spacing is absent, derive it from node extents so
            # larger labels reserve proportionally more room between subtrees.
            sibling_sep = (
                _node_spacing(problem.node_sizes, axis=0, default=1.0)
                if self.config.sibling_sep is None
                else self.config.sibling_sep
            )
            layer_sep = (
                _node_spacing(problem.node_sizes, axis=1, default=1.5)
                if self.config.layer_sep is None
                else self.config.layer_sep
            )
            component_gap = (
                sibling_sep * self.config.default_component_gap_multiplier
                if self.config.component_gap is None
                else self.config.component_gap
            )
            center_output = True if self.config.center_output is None else self.config.center_output
            output_scale = 1.0 if self.config.output_scale is None else self.config.output_scale

        sys.setrecursionlimit(
            max(
                sys.getrecursionlimit(),
                problem.num_nodes * self.config.recursion_limit_multiplier,
            )
        )
        explicit_roots = _validate_rt_roots(roots=self.config.roots, num_nodes=problem.num_nodes)
        root_depths = _validate_rt_rootlevels(
            rootlevels=self.config.rootlevel,
            roots=explicit_roots,
        )
        if explicit_roots is not None:
            automatic_candidates = _igraph_fidelity_root_candidates(
                edge_index=edge_index_cpu,
                num_nodes=problem.num_nodes,
                traversal_mode=self.config.traversal_mode,
            )
            root_candidates = explicit_roots + [
                candidate for candidate in automatic_candidates if candidate not in explicit_roots
            ]
        elif fidelity_mode == "igraph":
            root_candidates = _igraph_fidelity_root_candidates(
                edge_index=edge_index_cpu,
                num_nodes=problem.num_nodes,
                traversal_mode=self.config.traversal_mode,
            )
        else:
            root_candidates = None
        roots, children, depths = _bfs_forest(
            edge_index=edge_index_cpu,
            num_nodes=problem.num_nodes,
            traversal_mode=self.config.traversal_mode if fidelity_mode == "igraph" else "all",
            root_candidates=root_candidates,
            root_depths=root_depths,
        )

        preliminary_x = [0.0] * problem.num_nodes
        if fidelity_mode == "igraph" and len(roots) > 1 and not root_depths:
            artificial_root = problem.num_nodes
            augmented_children = [list(child_nodes) for child_nodes in children]
            augmented_children.append(list(roots))
            augmented_depths = [depth + 1 for depth in depths]
            augmented_depths.append(0)
            augmented_x = [0.0] * (problem.num_nodes + 1)
            _assign_preliminary_x(
                root_idx=artificial_root,
                children=augmented_children,
                depths=augmented_depths,
                preliminary_x=augmented_x,
                component_offset=0.0,
                component_gap=0.0,
            )
            preliminary_x = augmented_x[: problem.num_nodes]
            depths = augmented_depths[: problem.num_nodes]
        else:
            next_component_offset = 0.0
            for root in roots:
                next_component_offset = _assign_preliminary_x(
                    root_idx=root,
                    children=children,
                    depths=depths,
                    preliminary_x=preliminary_x,
                    component_offset=next_component_offset,
                    component_gap=component_gap,
                )

        positions = torch.zeros((problem.num_nodes, _POSITION_OUTPUT_DIM), dtype=torch.float32)
        for node_idx in range(problem.num_nodes):
            positions[node_idx, 0] = float(preliminary_x[node_idx]) * sibling_sep
            positions[node_idx, 1] = float(depths[node_idx]) * layer_sep

        positions *= float(output_scale)
        if center_output:
            positions -= positions.mean(dim=0, keepdim=True)
        if self.config.horizontal:
            # Preserve the same coordinates but swap depth into x for
            # left-to-right tree renderers.
            positions = positions[:, [1, 0]]
        state.pos = positions.to(device=output_device, dtype=torch.float32)
        return state
