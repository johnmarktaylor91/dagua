"""d3-dag Sugiyama-compatible composable operations.

The implementation targets ``d3-dag`` 1.2.2's deterministic Sugiyama path:
``layeringSimplex`` or ``layeringLongestPath`` followed by ``decrossTwoLayer``
or small-graph optimal decrossing, then ``coordSimplex`` or ``coordGreedy``.
"""

from __future__ import annotations

import itertools
import math
from dataclasses import dataclass
from typing import ClassVar, Dict, Iterable, List, Optional, Sequence

import torch
from scipy.optimize import linprog

from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_D3DAG_GRAPH_KEY = "d3dag_graph"
_D3DAG_LAYERS_KEY = "d3dag_layers"
_D3DAG_LAYER_HEIGHT_KEY = "d3dag_layer_height"
_D3DAG_WIDTH_KEY = "d3dag_width"


@dataclass
class _D3DagNode:
    """Mutable node in the dummy-expanded d3-dag Sugiyama graph.

    Parameters
    ----------
    node_id : int
        Expanded-graph node identifier.
    role : str
        ``"node"`` for original graph nodes, ``"link"`` for dummy nodes.
    layer : int
        Layer containing this node.
    original : int | None
        Original node index when ``role`` is ``"node"``.
    edge_id : int | None
        Original edge index when ``role`` is ``"link"``.
    top_layer : int | None
        Top layer for original nodes.
    bottom_layer : int | None
        Bottom layer for original nodes.
    x : float
        Horizontal coordinate in d3-dag layout units.
    y : float
        Vertical coordinate in d3-dag layout units.
    """

    node_id: int
    role: str
    layer: int
    original: Optional[int] = None
    edge_id: Optional[int] = None
    top_layer: Optional[int] = None
    bottom_layer: Optional[int] = None
    x: float = 0.0
    y: float = 0.0


@dataclass
class _D3DagSugiGraph:
    """Dummy-expanded layered graph consumed by d3-dag decross/coord ops.

    Parameters
    ----------
    nodes : list[_D3DagNode]
        Expanded graph node records.
    layers : list[list[int]]
        Expanded node ids grouped by layer.
    parents : list[dict[int, int]]
        Parent adjacency counts per expanded node.
    children : list[dict[int, int]]
        Child adjacency counts per expanded node.
    node_widths : list[float]
        Width per original node.
    node_heights : list[float]
        Height per original node.
    num_original_nodes : int
        Original graph node count.
    """

    nodes: List[_D3DagNode]
    layers: List[List[int]]
    parents: List[Dict[int, int]]
    children: List[Dict[int, int]]
    node_widths: List[float]
    node_heights: List[float]
    num_original_nodes: int


def _edge_pairs(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    """Return edge pairs from a tensor in stable column order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    list[tuple[int, int]]
        Directed edge pairs.
    """
    if edge_index.numel() == 0:
        return []
    return [(int(src), int(dst)) for src, dst in edge_index.detach().cpu().t().tolist()]


def _node_sizes(problem: LayoutProblem) -> tuple[list[float], list[float]]:
    """Resolve strictly positive d3-dag node widths and heights.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs.

    Returns
    -------
    tuple[list[float], list[float]]
        Widths and heights for original nodes.
    """
    if problem.node_sizes is None:
        return ([1.0] * problem.num_nodes, [1.0] * problem.num_nodes)
    sizes = problem.node_sizes.detach().cpu().to(dtype=torch.float64)
    widths = [max(float(sizes[node, 0].item()), 1.0e-9) for node in range(problem.num_nodes)]
    heights = [max(float(sizes[node, 1].item()), 1.0e-9) for node in range(problem.num_nodes)]
    return widths, heights


def _topological_order(num_nodes: int, edges: Sequence[tuple[int, int]]) -> list[int]:
    """Return d3-dag-compatible stable topological order.

    Parameters
    ----------
    num_nodes : int
        Number of original nodes.
    edges : sequence[tuple[int, int]]
        Directed edges.

    Returns
    -------
    list[int]
        Stable node order.

    Raises
    ------
    ValueError
        If the graph is cyclic.
    """
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    indegree = [0] * num_nodes
    for source, target in edges:
        children[source].append(target)
        indegree[target] += 1
    ready = [node for node, degree in enumerate(indegree) if degree == 0]
    order: list[int] = []
    index = 0
    while index < len(ready):
        node = ready[index]
        index += 1
        order.append(node)
        for child in children[node]:
            indegree[child] -= 1
            if indegree[child] == 0:
                ready.append(child)
    if len(order) != num_nodes:
        raise ValueError("d3-dag Sugiyama requires an acyclic graph")
    return order


def _layers_from_values(values: Sequence[float]) -> list[int]:
    """Convert numeric d3-dag layer values into dense integer layers.

    Parameters
    ----------
    values : sequence[float]
        Raw layer coordinates.

    Returns
    -------
    list[int]
        Dense layer index for every node.
    """
    unique = sorted({float(value) for value in values})
    lookup = {value: index for index, value in enumerate(unique)}
    return [lookup[float(value)] for value in values]


def _simplex_layers(num_nodes: int, edges: Sequence[tuple[int, int]]) -> list[int]:
    """Assign layers with d3-dag's ``layeringSimplex`` objective.

    Parameters
    ----------
    num_nodes : int
        Number of original nodes.
    edges : sequence[tuple[int, int]]
        Directed acyclic edges.

    Returns
    -------
    list[int]
        Dense layer index for each node.
    """
    if num_nodes == 0:
        return []
    if not edges:
        return [0] * num_nodes

    c = [0.0] * num_nodes
    rows: list[list[float]] = []
    lower: list[float] = []
    upper: list[float] = []
    for source, target in edges:
        row = [0.0] * num_nodes
        row[target] = 1.0
        row[source] = -1.0
        rows.append(row)
        lower.append(1.0)
        upper.append(math.inf)
        c[target] += 1.0
        c[source] -= 1.0

    del upper
    result = linprog(
        c,
        A_ub=[[-value for value in row] for row in rows],
        b_ub=[-value for value in lower],
        bounds=[(None, None)] * num_nodes,
        method="highs",
    )
    if not result.success:
        raise ValueError(f"could not find a feasible d3-dag simplex layering: {result.message}")
    raw = [float(value) for value in result.x]
    minimum = min(raw)
    return _layers_from_values([value - minimum for value in raw])


def _longest_path_layers(num_nodes: int, edges: Sequence[tuple[int, int]]) -> list[int]:
    """Assign layers with d3-dag's top-down longest-path operator.

    Parameters
    ----------
    num_nodes : int
        Number of original nodes.
    edges : sequence[tuple[int, int]]
        Directed acyclic edges.

    Returns
    -------
    list[int]
        Dense layer index for each node.
    """
    order = _topological_order(num_nodes, edges)
    parents: list[list[int]] = [[] for _ in range(num_nodes)]
    children: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in edges:
        parents[target].append(source)
        children[source].append(target)

    y = [0.0] * num_nodes
    seen: set[int] = set()
    for node in reversed(order):
        neighbors = parents[node] + children[node]
        assigned = [y[neighbor] + 1.0 for neighbor in neighbors if neighbor in seen]
        y[node] = max([0.0, *assigned])
        seen.add(node)
    height = max((value + 1.0 for value in y), default=0.0)
    for node in order:
        candidates = [
            y[neighbor] - 1.0
            for neighbor in parents[node] + children[node]
            if y[node] < y[neighbor]
        ]
        if candidates:
            y[node] = min(candidates)
    return _layers_from_values([height - value for value in y])


def _add_edge(graph: _D3DagSugiGraph, source: int, target: int) -> None:
    """Add one expanded edge, accumulating duplicate counts.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph to mutate.
    source : int
        Expanded source node id.
    target : int
        Expanded target node id.

    Returns
    -------
    None
        The graph adjacency maps are updated in place.
    """
    graph.children[source][target] = graph.children[source].get(target, 0) + 1
    graph.parents[target][source] = graph.parents[target].get(source, 0) + 1


def _sugify(
    num_nodes: int,
    edges: Sequence[tuple[int, int]],
    layers: Sequence[int],
    widths: Sequence[float],
    heights: Sequence[float],
    y_gap: float,
) -> tuple[_D3DagSugiGraph, float]:
    """Build d3-dag's dummy-expanded layered graph.

    Parameters
    ----------
    num_nodes : int
        Original node count.
    edges : sequence[tuple[int, int]]
        Original directed edges.
    layers : sequence[int]
        Dense layer for each original node.
    widths : sequence[float]
        Original node widths.
    heights : sequence[float]
        Original node heights.
    y_gap : float
        Vertical gap between layers.

    Returns
    -------
    tuple[_D3DagSugiGraph, float]
        Expanded graph and final layout height.
    """
    max_layer = max(layers, default=0)
    nodes: list[_D3DagNode] = []
    grouped: list[list[int]] = [[] for _ in range(max_layer + 1)]
    parents: list[dict[int, int]] = []
    children: list[dict[int, int]] = []

    for node in range(num_nodes):
        layer = int(layers[node])
        node_id = len(nodes)
        nodes.append(
            _D3DagNode(
                node_id=node_id,
                role="node",
                layer=layer,
                original=node,
                top_layer=layer,
                bottom_layer=layer,
            )
        )
        grouped[layer].append(node_id)
        parents.append({})
        children.append({})

    graph = _D3DagSugiGraph(
        nodes=nodes,
        layers=grouped,
        parents=parents,
        children=children,
        node_widths=list(widths),
        node_heights=list(heights),
        num_original_nodes=num_nodes,
    )

    for edge_id, (source, target) in enumerate(edges):
        source_layer = int(layers[source])
        target_layer = int(layers[target])
        if target_layer <= source_layer:
            raise ValueError("d3-dag layering assigned an edge to a non-descending layer")
        previous = source
        for layer in range(source_layer + 1, target_layer):
            dummy_id = len(graph.nodes)
            graph.nodes.append(_D3DagNode(dummy_id, "link", layer=layer, edge_id=edge_id))
            graph.parents.append({})
            graph.children.append({})
            graph.layers[layer].append(dummy_id)
            _add_edge(graph, previous, dummy_id)
            previous = dummy_id
        _add_edge(graph, previous, target)

    height = -y_gap
    for layer in graph.layers:
        height += y_gap
        layer_height = max(
            [-y_gap]
            + [
                heights[graph.nodes[node_id].original]
                for node_id in layer
                if graph.nodes[node_id].role == "node" and graph.nodes[node_id].original is not None
            ]
        )
        y = height + layer_height / 2.0
        for node_id in layer:
            graph.nodes[node_id].y = y
        height += layer_height
    return graph, height


def _node_width(graph: _D3DagSugiGraph, node_id: int) -> float:
    """Return d3-dag width for an expanded node.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph.
    node_id : int
        Expanded node id.

    Returns
    -------
    float
        Node width; dummy nodes have zero width.
    """
    node = graph.nodes[node_id]
    if node.role == "node" and node.original is not None:
        return graph.node_widths[node.original]
    return 0.0


def _sep(graph: _D3DagSugiGraph, left: Optional[int], right: Optional[int], x_gap: float) -> float:
    """Return d3-dag sized horizontal separation.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph.
    left : int | None
        Left node id, or ``None`` for boundary separation.
    right : int | None
        Right node id, or ``None`` for boundary separation.
    x_gap : float
        Horizontal gap between adjacent nodes.

    Returns
    -------
    float
        Required separation.
    """
    left_width = _node_width(graph, left) if left is not None else 0.0
    right_width = _node_width(graph, right) if right is not None else 0.0
    base = (left_width + right_width) / 2.0
    return base + x_gap if left is not None and right is not None else base


def _count_crossings(graph: _D3DagSugiGraph, layers: Sequence[Sequence[int]]) -> int:
    """Count weighted adjacent-layer crossings.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph.
    layers : sequence[sequence[int]]
        Candidate layer ordering.

    Returns
    -------
    int
        Weighted crossing count.
    """
    crossings = 0
    for top_layer, bottom_layer in zip(layers, layers[1:]):
        bottom_index = {node_id: index for index, node_id in enumerate(bottom_layer)}
        for left_index, first in enumerate(top_layer):
            for second in top_layer[left_index + 1 :]:
                for child_a, count_a in graph.children[first].items():
                    if child_a not in bottom_index:
                        continue
                    for child_b, count_b in graph.children[second].items():
                        if child_b != child_a and child_b in bottom_index:
                            if bottom_index[child_a] > bottom_index[child_b]:
                                crossings += count_a * count_b
    return crossings


def _weighted_median(values: Iterable[int]) -> Optional[float]:
    """Return d3-dag's weighted-median aggregate.

    Parameters
    ----------
    values : iterable[int]
        Reference layer indices.

    Returns
    -------
    float | None
        Aggregate index, or ``None`` for empty input.
    """
    vals = sorted(values)
    if not vals:
        return None
    if len(vals) == 2:
        return (vals[0] + vals[1]) / 2.0
    if len(vals) % 2 == 0:
        ind = len(vals) // 2
        first = vals[0]
        left = vals[ind - 1]
        right = vals[ind]
        last = vals[-1]
        left_diff = left - first
        right_diff = last - right
        if left_diff + right_diff == 0:
            return (left + right) / 2.0
        return (left * right_diff + right * left_diff) / (left_diff + right_diff)
    return float(vals[(len(vals) - 1) // 2])


def _order_by_aggregate(
    layer: list[int],
    poses: dict[int, Optional[float]],
) -> None:
    """Order one layer with d3-dag's unassigned-node insertion rule.

    Parameters
    ----------
    layer : list[int]
        Layer to mutate.
    poses : dict[int, float | None]
        Aggregate positions for nodes in the layer.

    Returns
    -------
    None
        ``layer`` is reordered in place.
    """
    original_indices = {node: index for index, node in enumerate(layer)}
    assigned = [node for node in layer if poses[node] is not None]
    assigned.sort(key=lambda node: (float(poses[node]), original_indices[node]))
    unassigned = [node for node in layer if poses[node] is None]
    placements: list[int] = [0] * len(unassigned)

    def recurse(ustart: int, uend: int, ostart: int, oend: int) -> None:
        """Assign insertion slots recursively.

        Parameters
        ----------
        ustart : int
            Start index into unassigned nodes.
        uend : int
            End index into unassigned nodes.
        ostart : int
            Start slot among assigned nodes.
        oend : int
            End slot among assigned nodes.

        Returns
        -------
        None
            ``placements`` is updated in place.
        """
        if uend <= ustart:
            return
        umid = (ustart + uend) // 2
        node = unassigned[umid]
        node_index = original_indices[node]
        last = 0
        inversions = [last]
        for index in range(ostart, oend):
            last += -1 if original_indices[assigned[index]] < node_index else 1
            inversions.append(last)
        placement = ostart + inversions.index(min(inversions))
        placements[umid] = placement
        recurse(ustart, umid, ostart, placement)
        recurse(umid + 1, uend, placement, oend)

    recurse(0, len(unassigned), 0, len(assigned))
    placements.append(len(assigned) + 1)
    insert = 0
    unassigned_index = 0
    for index, node in enumerate(assigned):
        while unassigned_index < len(unassigned) and placements[unassigned_index] == index:
            layer[insert] = unassigned[unassigned_index]
            insert += 1
            unassigned_index += 1
        layer[insert] = node
        insert += 1
    while unassigned_index < len(unassigned) and placements[unassigned_index] == len(assigned):
        layer[insert] = unassigned[unassigned_index]
        insert += 1
        unassigned_index += 1


def _twolayer_agg(
    graph: _D3DagSugiGraph,
    top_layer: list[int],
    bottom_layer: list[int],
    top_down: bool,
) -> None:
    """Apply d3-dag's default aggregate two-layer ordering.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph.
    top_layer : list[int]
        Upper layer.
    bottom_layer : list[int]
        Lower layer.
    top_down : bool
        Whether to reorder the lower layer from upper references.

    Returns
    -------
    None
        One layer is reordered in place.
    """
    reordered, reference = (bottom_layer, top_layer) if top_down else (top_layer, bottom_layer)
    ref_indices = {node: index for index, node in enumerate(reference)}
    poses: dict[int, Optional[float]] = {}
    for node in reordered:
        own_index = ref_indices.get(node)
        if own_index is not None:
            poses[node] = float(own_index)
            continue
        neighbors = graph.parents[node] if top_down else graph.children[node]
        expanded = [
            ref_indices[neighbor]
            for neighbor, count in neighbors.items()
            if neighbor in ref_indices
            for _ in range(count)
        ]
        poses[node] = _weighted_median(expanded)
    _order_by_aggregate(reordered, poses)


def _swap_change(
    graph: _D3DagSugiGraph,
    stationary: Sequence[int],
    left: int,
    right: int,
    top_down: bool,
) -> float:
    """Return crossing reduction from swapping adjacent nodes.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph.
    stationary : sequence[int]
        Reference layer that is not being reordered.
    left : int
        Left candidate node.
    right : int
        Right candidate node.
    top_down : bool
        Whether the reordered layer is below the stationary layer.

    Returns
    -------
    float
        Positive values indicate the swap improves crossings.
    """
    stationary_indices = {node: index for index, node in enumerate(stationary)}
    if left in stationary_indices or right in stationary_indices:
        return -math.inf
    delta = 0.0
    left_neighbors = graph.parents[left] if top_down else graph.children[left]
    right_neighbors = graph.parents[right] if top_down else graph.children[right]
    for left_child, left_count in left_neighbors.items():
        if left_child not in stationary_indices:
            continue
        for right_child, right_count in right_neighbors.items():
            if right_child in stationary_indices:
                index_diff = stationary_indices[left_child] - stationary_indices[right_child]
                delta += math.copysign(left_count * right_count, index_diff)
    return delta


def _twolayer_greedy(
    graph: _D3DagSugiGraph,
    top_layer: list[int],
    bottom_layer: list[int],
    top_down: bool,
) -> None:
    """Apply d3-dag's aggregate plus adjacent-swap two-layer heuristic.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph.
    top_layer : list[int]
        Upper layer.
    bottom_layer : list[int]
        Lower layer.
    top_down : bool
        Whether to reorder the lower layer.

    Returns
    -------
    None
        One layer is reordered in place.
    """
    _twolayer_agg(graph, top_layer, bottom_layer, top_down)
    layer = bottom_layer if top_down else top_layer
    stationary = top_layer if top_down else bottom_layer
    ranges = [(0, len(layer))]
    while ranges:
        start, end = ranges.pop()
        if start >= end:
            continue
        best = 0.0
        best_index = end
        for index in range(start, end - 1):
            diff = _swap_change(graph, stationary, layer[index], layer[index + 1], top_down)
            if diff > best:
                best = diff
                best_index = index
        if best_index != end:
            layer[best_index], layer[best_index + 1] = layer[best_index + 1], layer[best_index]
            ranges.append((start, best_index + 1))
            ranges.append((best_index + 1, end))


def _dfs_order(graph: _D3DagSugiGraph, top_down: bool) -> list[list[int]]:
    """Return d3-dag DFS initializer ordering.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph.
    top_down : bool
        Whether to traverse from roots to leaves.

    Returns
    -------
    list[list[int]]
        New layer ordering.
    """
    visited: set[int] = set()
    ordered = [[] for _ in graph.layers]

    if top_down:
        starts = [
            node
            for layer in graph.layers
            for node in sorted(layer, key=lambda item: len(graph.children[item]), reverse=True)
            if not graph.parents[node]
        ]

        def neighbors(node: int) -> list[int]:
            """Return DFS children in d3-dag priority order."""
            return sorted(
                graph.children[node],
                key=lambda item: len(graph.children[item]),
                reverse=True,
            )

    else:
        starts = [
            node
            for layer in reversed(graph.layers)
            for node in sorted(layer, key=lambda item: len(graph.parents[item]), reverse=True)
            if not graph.children[node]
        ]

        def neighbors(node: int) -> list[int]:
            """Return DFS parents in d3-dag priority order."""
            return sorted(
                graph.parents[node],
                key=lambda item: len(graph.parents[item]),
                reverse=True,
            )

    stack = list(starts)
    while stack:
        node = stack.pop()
        if node in visited:
            continue
        visited.add(node)
        ordered[graph.nodes[node].layer].append(node)
        stack.extend(neighbor for neighbor in neighbors(node) if neighbor not in visited)
    for layer in graph.layers:
        for node in layer:
            if node not in visited:
                ordered[graph.nodes[node].layer].append(node)
    return ordered


def _decross_two_layer(graph: _D3DagSugiGraph, passes: int) -> None:
    """Apply d3-dag's default ``decrossTwoLayer`` heuristic.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph to reorder.
    passes : int
        Maximum top-down/bottom-up sweep count.

    Returns
    -------
    None
        ``graph.layers`` is mutated in place.
    """
    best = [layer[:] for layer in graph.layers]
    best_crossings = _count_crossings(graph, best)
    for top_down_init in (True, False):
        graph.layers = _dfs_order(graph, top_down=top_down_init)
        reversed_layers = list(reversed(graph.layers))
        changed = True
        for _ in range(passes):
            if not changed:
                break
            changed = False
            for upper, lower in zip(graph.layers, graph.layers[1:]):
                snapshot = lower[:]
                _twolayer_greedy(graph, upper, lower, True)
                if snapshot != lower:
                    changed = True
            top_crossings = _count_crossings(graph, graph.layers)
            if top_crossings < best_crossings:
                best_crossings = top_crossings
                best = [layer[:] for layer in graph.layers]
            for lower, upper in zip(reversed_layers, reversed_layers[1:]):
                snapshot = upper[:]
                _twolayer_greedy(graph, upper, lower, False)
                if snapshot != upper:
                    changed = True
            bottom_crossings = _count_crossings(graph, graph.layers)
            if bottom_crossings < best_crossings:
                best_crossings = bottom_crossings
                best = [layer[:] for layer in graph.layers]
    graph.layers = best


def _decross_opt(graph: _D3DagSugiGraph, max_permutations: int = 250_000) -> None:
    """Optimally minimize crossings by enumerating small layer permutations.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph to reorder.
    max_permutations : int, default=250000
        Safety cap for the permutation product.

    Returns
    -------
    None
        ``graph.layers`` is replaced by the minimum-crossing order.

    Raises
    ------
    ValueError
        If the layer permutation product exceeds ``max_permutations``.
    """
    product = 1
    for layer in graph.layers:
        product *= math.factorial(len(layer))
    if product > max_permutations:
        raise ValueError("d3dag decrossOpt graph is too large for exact enumeration")
    best_layers = [layer[:] for layer in graph.layers]
    best_crossings = _count_crossings(graph, best_layers)
    for candidate in itertools.product(*(itertools.permutations(layer) for layer in graph.layers)):
        candidate_layers = [list(layer) for layer in candidate]
        crossing_count = _count_crossings(graph, candidate_layers)
        if crossing_count < best_crossings:
            best_crossings = crossing_count
            best_layers = candidate_layers
    graph.layers = best_layers


def _coord_simplex(graph: _D3DagSugiGraph, x_gap: float) -> float:
    """Assign x coordinates using d3-dag's ``coordSimplex`` LP.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph.
    x_gap : float
        Horizontal gap.

    Returns
    -------
    float
        Final layout width.
    """
    ids = list(dict.fromkeys(node for layer in graph.layers for node in layer))
    if not ids:
        return 0.0
    index = {node: pos for pos, node in enumerate(ids)}
    num_x = len(ids)
    c: list[float] = [0.0] * num_x
    rows: list[list[float]] = []
    lower: list[float] = []

    def add_constraint(coeffs: dict[int, float], low: float) -> None:
        """Append one lower-bounded linear constraint."""
        row = [0.0] * len(c)
        for variable, value in coeffs.items():
            row[variable] = value
        rows.append(row)
        lower.append(low)

    def add_slack_pair(left: int, right: int, name_weight: float) -> None:
        """Add absolute-difference slack around two x variables."""
        slack = len(c)
        c.append(name_weight)
        for row in rows:
            row.append(0.0)
        add_constraint({slack: 1.0, left: 1.0, right: -1.0}, 0.0)
        add_constraint({slack: 1.0, left: -1.0, right: 1.0}, 0.0)

    for layer in graph.layers:
        for left, right in zip(layer, layer[1:]):
            add_constraint({index[right]: 1.0, index[left]: -1.0}, _sep(graph, left, right, x_gap))

    heights = [
        abs(graph.nodes[child].y - graph.nodes[node].y)
        for node in ids
        for child in graph.children[node]
        if child in index
    ]
    height_norm = sum(heights) / len(heights) if heights else 1.0
    primary_coeffs: list[float] = []
    for node in ids:
        for child in graph.children[node]:
            if child not in index:
                continue
            node_role_count = int(graph.nodes[node].role == "node") + int(
                graph.nodes[child].role == "node"
            )
            weight = {0: 8.0, 1: 2.0, 2: 1.0}[node_role_count]
            height = max((graph.nodes[child].y - graph.nodes[node].y) / height_norm, 1.0e-12)
            coeff = weight / height
            primary_coeffs.append(coeff)
            add_slack_pair(index[node], index[child], coeff)

    min_primary = min(primary_coeffs) if primary_coeffs else 1.0
    child_nodes = [node for node in ids if graph.children[node]]
    parent_nodes = [node for node in ids if graph.parents[node]]
    child_eps = min_primary / (len(child_nodes) + 1)
    parent_eps = min_primary / (len(parent_nodes) + 1)
    for node in child_nodes:
        slack = len(c)
        c.append(child_eps)
        for row in rows:
            row.append(0.0)
        coeff = {index[node]: 1.0, slack: 1.0}
        factor = 1.0 / len(graph.children[node])
        for child in graph.children[node]:
            if child in index:
                coeff[index[child]] = coeff.get(index[child], 0.0) - factor
        add_constraint(coeff, 0.0)
        coeff = {key: -value for key, value in coeff.items() if key != slack}
        coeff[slack] = 1.0
        add_constraint(coeff, 0.0)
    for node in parent_nodes:
        slack = len(c)
        c.append(parent_eps)
        for row in rows:
            row.append(0.0)
        coeff = {index[node]: 1.0, slack: 1.0}
        factor = 1.0 / len(graph.parents[node])
        for parent in graph.parents[node]:
            if parent in index:
                coeff[index[parent]] = coeff.get(index[parent], 0.0) - factor
        add_constraint(coeff, 0.0)
        coeff = {key: -value for key, value in coeff.items() if key != slack}
        coeff[slack] = 1.0
        add_constraint(coeff, 0.0)

    if rows:
        result = linprog(
            c,
            A_ub=[[-value for value in row] for row in rows],
            b_ub=[-value for value in lower],
            bounds=[(None, None)] * len(c),
            method="highs",
        )
        if not result.success:
            raise ValueError(
                f"could not find a feasible d3-dag coordSimplex solution: {result.message}"
            )
        for node, variable in index.items():
            graph.nodes[node].x = float(result.x[variable])
    else:
        for node in ids:
            graph.nodes[node].x = 0.0
    offset = 0.0
    width = 0.0
    for layer in graph.layers:
        first = layer[0]
        last = layer[-1]
        offset = min(offset, graph.nodes[first].x - _sep(graph, None, first, x_gap))
        width = max(width, graph.nodes[last].x + _sep(graph, last, None, x_gap))
    for node in ids:
        graph.nodes[node].x -= offset
    return width - offset


def _space_layer(graph: _D3DagSugiGraph, layer: Sequence[int], x_gap: float) -> None:
    """Space one layer according to d3-dag greedy coordinate assignment.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph.
    layer : sequence[int]
        Layer node ids.
    x_gap : float
        Horizontal gap.

    Returns
    -------
    None
        Node x coordinates are updated in place.
    """
    last = layer[-1]
    last_x = graph.nodes[last].x
    after = [last_x]
    for node in reversed(layer[:-1]):
        next_x = min(graph.nodes[node].x, last_x - _sep(graph, node, last, x_gap))
        after.append(next_x)
        last = node
        last_x = next_x
    after.reverse()
    last = layer[0]
    last_x = graph.nodes[last].x
    graph.nodes[last].x = (last_x + after[0]) / 2.0
    for index, node in enumerate(layer[1:], start=1):
        next_x = max(graph.nodes[node].x, last_x + _sep(graph, last, node, x_gap))
        graph.nodes[node].x = (next_x + after[index]) / 2.0
        last = node
        last_x = next_x


def _coord_greedy(graph: _D3DagSugiGraph, x_gap: float) -> float:
    """Assign x coordinates using d3-dag's greedy coordinate heuristic.

    Parameters
    ----------
    graph : _D3DagSugiGraph
        Expanded graph.
    x_gap : float
        Horizontal gap.

    Returns
    -------
    float
        Final layout width.
    """
    for layer in graph.layers:
        for index, node in enumerate(layer):
            graph.nodes[node].x = float(index)
        _space_layer(graph, layer, x_gap)
    for layer in graph.layers[1:]:
        for node in layer:
            refs = [graph.nodes[parent].x for parent in graph.parents[node]]
            if refs:
                graph.nodes[node].x = float(torch.tensor(refs, dtype=torch.float64).median().item())
        _space_layer(graph, layer, x_gap)
    for layer in reversed(graph.layers[:-1]):
        for node in layer:
            refs = [graph.nodes[child].x for child in graph.children[node]]
            if refs:
                graph.nodes[node].x = float(torch.tensor(refs, dtype=torch.float64).median().item())
        _space_layer(graph, layer, x_gap)
    start = math.inf
    end = -math.inf
    for layer in graph.layers:
        start = min(start, graph.nodes[layer[0]].x - _sep(graph, None, layer[0], x_gap))
        end = max(end, graph.nodes[layer[-1]].x + _sep(graph, layer[-1], None, x_gap))
    for node in {node for layer in graph.layers for node in layer}:
        graph.nodes[node].x -= start
    return end - start


@dataclass(frozen=True)
class D3DagPrepare:
    """Prepare original graph inputs for d3-dag Sugiyama stages.

    Parameters
    ----------
    x_gap : float, default=1.0
        Horizontal d3-dag gap.
    y_gap : float, default=1.0
        Vertical d3-dag gap.
    """

    x_gap: float = 1.0
    y_gap: float = 1.0
    name: ClassVar[str] = "d3dag_prepare"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Validate and store d3-dag input metadata.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context; unused.

        Returns
        -------
        SolveState
            State with d3-dag metadata in ``extras``.
        """
        del ctx
        edges = _edge_pairs(problem.edge_index)
        _topological_order(problem.num_nodes, edges)
        widths, heights = _node_sizes(problem)
        state.extras["d3dag_edges"] = edges
        state.extras["d3dag_widths"] = widths
        state.extras["d3dag_heights"] = heights
        state.extras["d3dag_x_gap"] = float(self.x_gap)
        state.extras["d3dag_y_gap"] = float(self.y_gap)
        return state


@dataclass(frozen=True)
class D3DagLayering:
    """Assign d3-dag Sugiyama layers.

    Parameters
    ----------
    method : str, default="simplex"
        ``"simplex"`` or ``"longestPath"``.
    """

    method: str = "simplex"
    name: ClassVar[str] = "d3dag_layering"
    category: ClassVar[OpCategory] = OpCategory.LAYERING

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Compute and store original-node layer assignments.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context; unused.

        Returns
        -------
        SolveState
            State with ``state.layers`` and d3-dag layer extras populated.
        """
        del ctx
        edges = state.extras["d3dag_edges"]
        normalized = self.method.lower().replace("_", "").replace("-", "")
        if normalized in {"simplex", "layeringsimplex"}:
            layers = _simplex_layers(problem.num_nodes, edges)
        elif normalized in {"longestpath", "layeringlongestpath"}:
            layers = _longest_path_layers(problem.num_nodes, edges)
        else:
            raise ValueError("d3dag layering must be 'simplex' or 'longestPath'.")
        state.layers = torch.tensor(layers, dtype=torch.long)
        state.extras["d3dag_stage_layers"] = layers
        return state


@dataclass(frozen=True)
class D3DagSugify:
    """Build the dummy-expanded d3-dag layered graph."""

    name: ClassVar[str] = "d3dag_sugify"
    category: ClassVar[OpCategory] = OpCategory.LAYERING

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Expand long edges into one-layer dummy chains.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state with ``state.layers`` assigned.
        ctx : RuntimeContext
            Execution context; unused.

        Returns
        -------
        SolveState
            State with expanded graph metadata.
        """
        del ctx
        if state.layers is None:
            raise ValueError("d3dag_sugify requires state.layers")
        graph, height = _sugify(
            num_nodes=problem.num_nodes,
            edges=state.extras["d3dag_edges"],
            layers=[int(value) for value in state.layers.detach().cpu().tolist()],
            widths=state.extras["d3dag_widths"],
            heights=state.extras["d3dag_heights"],
            y_gap=float(state.extras["d3dag_y_gap"]),
        )
        state.extras[_D3DAG_GRAPH_KEY] = graph
        state.extras[_D3DAG_LAYER_HEIGHT_KEY] = height
        return state


@dataclass(frozen=True)
class D3DagDecross:
    """Reorder d3-dag expanded layers to reduce crossings.

    Parameters
    ----------
    method : str, default="twoLayer"
        ``"twoLayer"`` for d3-dag's default heuristic or ``"opt"`` for
        exact small-graph crossing minimization.
    passes : int, default=24
        Maximum heuristic sweep count.
    """

    method: str = "twoLayer"
    passes: int = 24
    name: ClassVar[str] = "d3dag_decross"
    category: ClassVar[OpCategory] = OpCategory.ORDERING

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Run the selected d3-dag decross stage.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs; unused.
        state : SolveState
            Mutable state carrying the expanded graph.
        ctx : RuntimeContext
            Execution context; unused.

        Returns
        -------
        SolveState
            State with expanded-layer order and original ordering tensor.
        """
        del problem, ctx
        graph = state.extras[_D3DAG_GRAPH_KEY]
        normalized = self.method.lower().replace("_", "").replace("-", "")
        if normalized in {"twolayer", "decrosstwolayer"}:
            _decross_two_layer(graph, self.passes)
        elif normalized in {"opt", "decrossopt"}:
            _decross_opt(graph)
        elif normalized in {"dfs", "decrossdfs"}:
            graph.layers = _dfs_order(graph, top_down=True)
        else:
            raise ValueError("d3dag decross must be 'twoLayer', 'opt', or 'dfs'.")
        ordering = [0] * graph.num_original_nodes
        for layer in graph.layers:
            for index, node_id in enumerate(layer):
                node = graph.nodes[node_id]
                if node.role == "node" and node.original is not None:
                    ordering[node.original] = index
        state.ordering = torch.tensor(ordering, dtype=torch.long)
        state.extras["d3dag_stage_ordering"] = [layer[:] for layer in graph.layers]
        return state


@dataclass(frozen=True)
class D3DagCoordinate:
    """Assign final d3-dag Sugiyama coordinates.

    Parameters
    ----------
    method : str, default="simplex"
        ``"simplex"`` or ``"greedy"``.
    """

    method: str = "simplex"
    name: ClassVar[str] = "d3dag_coordinate"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Run d3-dag coordinate assignment and unsugify original nodes.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable state carrying the expanded graph.
        ctx : RuntimeContext
            Execution context; unused.

        Returns
        -------
        SolveState
            State with final ``state.pos`` tensor of shape ``[N, 2]``.
        """
        del ctx
        graph = state.extras[_D3DAG_GRAPH_KEY]
        normalized = self.method.lower().replace("_", "").replace("-", "")
        if normalized in {"simplex", "coordsimplex"}:
            width = _coord_simplex(graph, float(state.extras["d3dag_x_gap"]))
        elif normalized in {"greedy", "coordgreedy"}:
            width = _coord_greedy(graph, float(state.extras["d3dag_x_gap"]))
        else:
            raise ValueError("d3dag coord must be 'simplex' or 'greedy'.")
        positions = torch.zeros((problem.num_nodes, 2), dtype=torch.float64)
        for node in graph.nodes:
            if node.role == "node" and node.original is not None:
                positions[node.original, 0] = node.x
                positions[node.original, 1] = node.y
        state.pos = positions
        state.extras[_D3DAG_WIDTH_KEY] = width
        return state


@dataclass(frozen=True)
class D3DagCoffmanGrahamLayering:
    """Reusable deterministic Coffman-Graham-style layering op.

    Parameters
    ----------
    width : int
        Maximum number of original nodes per layer.
    """

    width: int
    name: ClassVar[str] = "d3dag_coffman_graham_layering"
    category: ClassVar[OpCategory] = OpCategory.LAYERING

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Assign width-bounded layers for a DAG.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context; unused.

        Returns
        -------
        SolveState
            State with ``state.layers`` populated.
        """
        del ctx
        if self.width <= 0:
            raise ValueError("Coffman-Graham width must be positive")
        edges = _edge_pairs(problem.edge_index)
        order = _topological_order(problem.num_nodes, edges)
        parents: list[list[int]] = [[] for _ in range(problem.num_nodes)]
        for source, target in edges:
            parents[target].append(source)
        layers = [0] * problem.num_nodes
        counts: list[int] = []
        for node in order:
            layer = max((layers[parent] + 1 for parent in parents[node]), default=0)
            while layer < len(counts) and counts[layer] >= self.width:
                layer += 1
            if layer == len(counts):
                counts.append(0)
            counts[layer] += 1
            layers[node] = layer
        state.layers = torch.tensor(layers, dtype=torch.long)
        return state


@dataclass(frozen=True)
class D3DagOptimalCrossingOrder:
    """Reusable exact crossing minimization op for small layered DAGs."""

    name: ClassVar[str] = "d3dag_optimal_crossing_order"
    category: ClassVar[OpCategory] = OpCategory.ORDERING

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Compute exact layer order for the active d3-dag expanded graph.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs; unused.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context; unused.

        Returns
        -------
        SolveState
            State with optimized expanded ordering.
        """
        del problem, ctx
        graph = state.extras.get(_D3DAG_GRAPH_KEY)
        if graph is None:
            raise ValueError("d3dag_optimal_crossing_order requires a d3-dag expanded graph")
        _decross_opt(graph)
        state.extras["d3dag_stage_ordering"] = [layer[:] for layer in graph.layers]
        return state


register_op(D3DagPrepare)
register_op(D3DagLayering)
register_op(D3DagSugify)
register_op(D3DagDecross)
register_op(D3DagCoordinate)
register_op(D3DagCoffmanGrahamLayering)
register_op(D3DagOptimalCrossingOrder)


__all__ = [
    "D3DagCoffmanGrahamLayering",
    "D3DagCoordinate",
    "D3DagDecross",
    "D3DagLayering",
    "D3DagOptimalCrossingOrder",
    "D3DagPrepare",
    "D3DagSugify",
]
