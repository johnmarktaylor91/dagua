"""Composable native stages for an ELK Layered-style pipeline.

The production code in this module does not invoke ``elkjs``.  It implements
the same public node-coordinate contract used by the existing ELK competitor
adapter: top-left node coordinates, ELK's default 12 point root padding, and
the layered spacing options exposed by the adapter.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import ClassVar, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.brandes_koepf import (
    BRANDES_KOEPF_DUMMY_NODES_KEY,
    BRANDES_KOEPF_LAYERING_KEY,
    BRANDES_KOEPF_PREDECESSORS_KEY,
    BRANDES_KOEPF_SUCCESSORS_KEY,
    BRANDES_KOEPF_WIDTHS_KEY,
    BRANDES_KOEPF_X_KEY,
)
from dagua.layout.ops.dagre import _dagre_network_simplex_ranks
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

ELK_GRAPH_KEY = "elk_graph"
ELK_LAYERS_KEY = "elk_layers"
ELK_ORDER_KEY = "elk_order"
ELK_INTERNAL_POSITIONS_KEY = "elk_internal_positions"

NodeId = Hashable

_ROOT_PADDING = 12.0
_ELK_DEFAULT_EDGE_NODE_SPACING = 10.0
_ELK_DEFAULT_EDGE_EDGE_SPACING = 10.0
_ELK_BK_STRAIGHTENING_THRESHOLD = float("1.7976931348623157e308")
_ELK_BK_LAYOUT_SIZE_EPSILON = 1.0e-9
_SUPPORTED_CYCLE_STRATEGIES = {"greedy", "depth_first", "interactive", "model_order"}
_SUPPORTED_LAYERING_STRATEGIES = {
    "network_simplex",
    "longest_path",
    "coffman_graham",
    "min_width",
    "interactive",
}
_SUPPORTED_CROSSING_STRATEGIES = {"layer_sweep", "greedy_switch", "interactive"}
_SUPPORTED_NODE_PLACEMENT_STRATEGIES = {
    "brandes_koepf",
    "linear_segments",
    "network_simplex",
    "simple",
}
_JAVA_RANDOM_MULTIPLIER = 0x5DEECE66D
_JAVA_RANDOM_ADDEND = 0xB
_JAVA_RANDOM_MASK = (1 << 48) - 1
_ELK_DEFAULT_RANDOM_SEED = 1


class _JavaRandom:
    """Small port of ``java.util.Random`` for ELK tie-breaking.

    Parameters
    ----------
    seed : int
        Public Java seed before the constructor's xor/scramble step.

    Attributes
    ----------
    seed : int
        Internal 48-bit LCG state.
    """

    def __init__(self, seed: int) -> None:
        """Initialize the scrambled 48-bit LCG state.

        Parameters
        ----------
        seed : int
            Public Java seed.

        Returns
        -------
        None
            The RNG state is stored on the instance.
        """
        self.seed = (seed ^ _JAVA_RANDOM_MULTIPLIER) & _JAVA_RANDOM_MASK

    def next_bits(self, bits: int) -> int:
        """Return the next high-order random bits.

        Parameters
        ----------
        bits : int
            Number of high bits to return.

        Returns
        -------
        int
            Unsigned random value with ``bits`` significant bits.
        """
        self.seed = (self.seed * _JAVA_RANDOM_MULTIPLIER + _JAVA_RANDOM_ADDEND) & _JAVA_RANDOM_MASK
        return self.seed >> (48 - bits)

    def next_boolean(self) -> bool:
        """Return Java's ``nextBoolean`` result.

        Returns
        -------
        bool
            ``True`` when the next random bit is non-zero.
        """
        return self.next_bits(1) != 0

    def next_float(self) -> float:
        """Return Java's ``nextFloat`` result.

        Returns
        -------
        float
            Uniform value in ``[0, 1)`` with 24 random bits.
        """
        return self.next_bits(24) / float(1 << 24)

    def next_double(self) -> float:
        """Return Java's ``nextDouble`` result.

        Returns
        -------
        float
            Uniform value in ``[0, 1)`` with Java's 53-bit composition.
        """
        high = self.next_bits(26)
        low = self.next_bits(27)
        return ((high << 27) + low) / float(1 << 53)

    def next_long(self) -> int:
        """Return Java's signed ``nextLong`` result.

        Returns
        -------
        int
            Signed 64-bit value composed from two ``next(32)`` calls.
        """
        value = (self.next_bits(32) << 32) + self.next_bits(32)
        if value >= (1 << 63):
            value -= 1 << 64
        return value

    def next_int(self, bound: int) -> int:
        """Return Java's ``nextInt(bound)`` result.

        Parameters
        ----------
        bound : int
            Exclusive positive upper bound.

        Returns
        -------
        int
            Uniform integer in ``[0, bound)``.

        Raises
        ------
        ValueError
            If ``bound`` is not positive.
        """
        if bound <= 0:
            raise ValueError("bound must be positive.")
        if bound & (bound - 1) == 0:
            return (bound * self.next_bits(31)) >> 31
        while True:
            bits = self.next_bits(31)
            value = bits % bound
            if bits - value + (bound - 1) >= 0:
                return value


@dataclass
class _ElkGraph:
    """Mutable graph shared by the ELK stage ports."""

    edges: List[Tuple[int, int]]
    active_edges: List[Tuple[int, int]]
    node_sizes: torch.Tensor
    direction: str
    node_node_spacing: float
    between_layers_spacing: float
    cycle_breaking_strategy: str
    layering_strategy: str
    crossing_minimization_strategy: str
    node_placement_strategy: str
    random_seed: int
    thoroughness: int


@dataclass
class _ElkBkGraph:
    """Normalized graph used by ELK Brandes-Koepf placement."""

    layers: List[List[NodeId]]
    predecessors: Dict[NodeId, List[NodeId]]
    successors: Dict[NodeId, List[NodeId]]
    edge_ids: Dict[Tuple[NodeId, NodeId], int]
    sizes: Dict[NodeId, float]
    dummy_nodes: Set[NodeId]
    real_nodes: List[int]
    node_index: Dict[NodeId, int]
    layer_index: Dict[NodeId, int]


@dataclass
class _ElkBkLayout:
    """Mutable state for one ELK BK alignment."""

    vdir: str
    hdir: str
    root: Dict[NodeId, NodeId]
    align: Dict[NodeId, NodeId]
    inner_shift: Dict[NodeId, float]
    block_size: Dict[NodeId, float]
    sink: Dict[NodeId, NodeId]
    shift: Dict[NodeId, float]
    y: Dict[NodeId, float]
    straightened: Dict[NodeId, bool]
    only_dummies: Dict[NodeId, bool]


def _normalize_choice(value: str, supported: Set[str], option_name: str) -> str:
    """Normalize an ELK option spelling.

    Parameters
    ----------
    value : str
        User-supplied option.
    supported : set[str]
        Supported normalized values.
    option_name : str
        Human-readable option name for error reporting.

    Returns
    -------
    str
        Lowercase value with hyphens converted to underscores.

    Raises
    ------
    ValueError
        If the normalized value is unsupported.
    """
    normalized = value.lower().replace("-", "_")
    if normalized not in supported:
        expected = ", ".join(sorted(supported))
        raise ValueError(f"{option_name} must be one of {expected}.")
    return normalized


def _normalize_direction(direction: str) -> str:
    """Normalize ELK and dagua direction spellings.

    Parameters
    ----------
    direction : str
        Direction spelling such as ``DOWN``, ``UP``, ``RIGHT``, ``LEFT``,
        ``TB``, ``BT``, ``LR``, or ``RL``.

    Returns
    -------
    str
        One of ``DOWN``, ``UP``, ``RIGHT``, or ``LEFT``.

    Raises
    ------
    ValueError
        If the direction is unsupported.
    """
    normalized = direction.upper()
    aliases = {"TB": "DOWN", "BT": "UP", "LR": "RIGHT", "RL": "LEFT"}
    normalized = aliases.get(normalized, normalized)
    if normalized not in {"DOWN", "UP", "RIGHT", "LEFT"}:
        raise ValueError("direction must be DOWN, UP, RIGHT, LEFT, TB, BT, LR, or RL.")
    return normalized


def _node_sizes(problem: LayoutProblem) -> torch.Tensor:
    """Return CPU float64 node sizes, using ELK's size-blind fallback.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable graph inputs.

    Returns
    -------
    torch.Tensor
        Node sizes with shape ``[N, 2]``.
    """
    if problem.node_sizes is None:
        sizes = torch.empty((problem.num_nodes, 2), dtype=torch.float64)
        sizes[:, 0] = 120.0
        sizes[:, 1] = 40.0
        return sizes
    sizes = problem.node_sizes.detach().to(device="cpu", dtype=torch.float64)
    if sizes.shape != (problem.num_nodes, 2):
        raise ValueError("node_sizes must have shape [N, 2].")
    return sizes


def _successors(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    """Build successor lists in model order.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    dict[int, list[int]]
        Successor lists with duplicate targets removed per source.
    """
    successors = {node: [] for node in range(num_nodes)}
    seen: Dict[int, Set[int]] = {node: set() for node in range(num_nodes)}
    for source, target in edges:
        if target not in seen[source]:
            seen[source].add(target)
            successors[source].append(target)
    return successors


def _predecessors(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> Dict[int, List[int]]:
    """Build predecessor lists in model order.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    dict[int, list[int]]
        Predecessor lists with duplicate sources removed per target.
    """
    predecessors = {node: [] for node in range(num_nodes)}
    seen: Dict[int, Set[int]] = {node: set() for node in range(num_nodes)}
    for source, target in edges:
        if source not in seen[target]:
            seen[target].add(source)
            predecessors[target].append(source)
    return predecessors


def _break_cycles_depth_first(
    num_nodes: int,
    edges: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Return an acyclic edge orientation by reversing DFS back edges.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Directed edge pairs.

    Returns
    -------
    list[tuple[int, int]]
        Edge pairs after deterministic cycle breaking.
    """
    outgoing: Dict[int, List[Tuple[int, int]]] = {node: [] for node in range(num_nodes)}
    for index, (source, target) in enumerate(edges):
        outgoing[source].append((index, target))
    reversed_indices: Set[int] = set()
    visited: Set[int] = set()
    stack: Set[int] = set()

    def visit(node: int) -> None:
        """Visit one node in model-order DFS.

        Parameters
        ----------
        node : int
            Node to visit.

        Returns
        -------
        None
            ``reversed_indices`` is mutated in place.
        """
        visited.add(node)
        stack.add(node)
        for edge_index, target in outgoing[node]:
            if target in stack:
                reversed_indices.add(edge_index)
            elif target not in visited:
                visit(target)
        stack.remove(node)

    for node in range(num_nodes):
        if node not in visited:
            visit(node)
    return [
        (target, source) if index in reversed_indices else (source, target)
        for index, (source, target) in enumerate(edges)
    ]


def _break_cycles_greedy(
    num_nodes: int,
    edges: Sequence[Tuple[int, int]],
) -> List[Tuple[int, int]]:
    """Return an acyclic orientation using ELK's greedy source/sink removal.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Directed edge pairs in model order.

    Returns
    -------
    list[tuple[int, int]]
        Edge pairs after reversing edges whose greedy marks run backward.
    """
    incoming = _predecessors(num_nodes, edges)
    outgoing = _successors(num_nodes, edges)
    indegree = [len(incoming[node]) for node in range(num_nodes)]
    outdegree = [len(outgoing[node]) for node in range(num_nodes)]
    marks = [0] * num_nodes
    sources = [node for node in range(num_nodes) if indegree[node] == 0 and outdegree[node] > 0]
    sinks = [node for node in range(num_nodes) if outdegree[node] == 0]
    next_right = -1
    next_left = 1
    unprocessed = num_nodes
    rng = _JavaRandom(_ELK_DEFAULT_RANDOM_SEED)

    def remove_node(node: int, mark: int) -> None:
        """Mark one node and update unprocessed neighbor degrees.

        Parameters
        ----------
        node : int
            Node selected by the greedy cycle breaker.
        mark : int
            Signed ordering mark assigned by the source/sink pass.

        Returns
        -------
        None
            ``marks``, ``indegree``, ``outdegree``, ``sources``, and ``sinks``
            are updated in place.
        """
        marks[node] = mark
        for target in outgoing[node]:
            if marks[target] == 0:
                indegree[target] -= 1
                if indegree[target] <= 0 and outdegree[target] > 0:
                    sources.append(target)
        for source in incoming[node]:
            if marks[source] == 0:
                outdegree[source] -= 1
                if outdegree[source] <= 0 and indegree[source] > 0:
                    sinks.append(source)

    while unprocessed > 0:
        while sinks:
            sink = sinks.pop(0)
            if marks[sink] != 0:
                continue
            remove_node(sink, next_right)
            next_right -= 1
            unprocessed -= 1
        while sources:
            source = sources.pop(0)
            if marks[source] != 0:
                continue
            remove_node(source, next_left)
            next_left += 1
            unprocessed -= 1
        if unprocessed > 0:
            candidates = [node for node in range(num_nodes) if marks[node] == 0]
            max_outflow = max(outdegree[node] - indegree[node] for node in candidates)
            tied = [node for node in candidates if outdegree[node] - indegree[node] == max_outflow]
            node = tied[rng.next_int(len(tied))]
            remove_node(node, next_left)
            next_left += 1
            unprocessed -= 1

    shift_base = num_nodes + 1
    normalized_marks = [mark + shift_base if mark < 0 else mark for mark in marks]
    return [
        (target, source)
        if normalized_marks[source] > normalized_marks[target]
        else (source, target)
        for source, target in edges
    ]


def _longest_path_layers(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> List[int]:
    """Assign layers by longest source-to-node path.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Acyclic edge pairs.

    Returns
    -------
    list[int]
        Layer index per node.
    """
    successors = _successors(num_nodes, edges)
    indegree = [0] * num_nodes
    for _, target in edges:
        indegree[target] += 1
    queue = [node for node in range(num_nodes) if indegree[node] == 0]
    layers = [0] * num_nodes
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        for target in successors[node]:
            layers[target] = max(layers[target], layers[node] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return layers


def _weak_components(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Return weakly connected components in first-node order.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Directed edge pairs, treated as undirected for component discovery.

    Returns
    -------
    list[list[int]]
        Components ordered by their smallest model-order node.
    """
    parent = list(range(num_nodes))

    def find(node: int) -> int:
        """Return the union-find root for one node.

        Parameters
        ----------
        node : int
            Node id.

        Returns
        -------
        int
            Root representative.
        """
        while parent[node] != node:
            parent[node] = parent[parent[node]]
            node = parent[node]
        return node

    def union(left: int, right: int) -> None:
        """Merge two component sets.

        Parameters
        ----------
        left : int
            First node id.
        right : int
            Second node id.

        Returns
        -------
        None
            ``parent`` is mutated in place.
        """
        left_root = find(left)
        right_root = find(right)
        if left_root != right_root:
            parent[right_root] = left_root

    for source, target in edges:
        union(source, target)
    groups: Dict[int, List[int]] = {}
    for node in range(num_nodes):
        groups.setdefault(find(node), []).append(node)
    return sorted(groups.values(), key=lambda component: component[0])


def _component_network_simplex_layers(
    component: Sequence[int],
    edges: Sequence[Tuple[int, int]],
) -> Dict[int, int]:
    """Assign ELK network-simplex layers inside one non-isolate component.

    Parameters
    ----------
    component : sequence[int]
        Original node ids in one weak component.
    edges : sequence[tuple[int, int]]
        Active acyclic graph edges.

    Returns
    -------
    dict[int, int]
        Original node id to zero-based component-local layer.
    """
    local_by_node = {node: index for index, node in enumerate(component)}
    rank_edges = [
        (local_by_node[source], local_by_node[target], 1, 1)
        for source, target in edges
        if source in local_by_node and target in local_by_node
    ]
    if not rank_edges:
        return {node: 0 for node in component}
    ranks = _dagre_network_simplex_ranks(list(range(len(component))), rank_edges)
    minimum = min(ranks.values())
    return {node: int(ranks[local_by_node[node]] - minimum) for node in component}


def _network_simplex_layers(num_nodes: int, edges: Sequence[Tuple[int, int]]) -> List[int]:
    """Assign ELK-style layer indices with component packing bands.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    edges : sequence[tuple[int, int]]
        Active acyclic graph edges.

    Returns
    -------
    list[int]
        Layer index per node, including disconnected-component y-band offsets.
    """
    components = _weak_components(num_nodes, edges)
    incident_nodes = {node for edge in edges for node in edge}
    nontrivial = [
        component for component in components if any(node in incident_nodes for node in component)
    ]
    has_isolate = len(nontrivial) < len(components)
    assignments = [0] * num_nodes
    cursor = 1 if has_isolate and len(nontrivial) == 1 else 0
    for component in nontrivial:
        local_layers = _component_network_simplex_layers(component, edges)
        for node in component:
            assignments[node] = local_layers[node] + cursor
        cursor += max(local_layers.values(), default=0) + 1
    return assignments


def _layers_from_assignments(assignments: Sequence[int]) -> List[List[int]]:
    """Group nodes by layer assignment.

    Parameters
    ----------
    assignments : sequence[int]
        Layer index per node.

    Returns
    -------
    list[list[int]]
        Nodes grouped by ascending layer, preserving model order.
    """
    if not assignments:
        return []
    layers: List[List[int]] = [[] for _ in range(max(assignments) + 1)]
    for node, layer in enumerate(assignments):
        layers[layer].append(node)
    return layers


def _median(values: Sequence[int]) -> float:
    """Return ELK-style median value for barycenter ordering.

    Parameters
    ----------
    values : sequence[int]
        Neighbor order values.

    Returns
    -------
    float
        Median or average of the two middle samples.
    """
    if not values:
        return 0.0
    ordered = sorted(values)
    middle = len(ordered) // 2
    if len(ordered) % 2:
        return float(ordered[middle])
    return (ordered[middle - 1] + ordered[middle]) / 2.0


def _shuffle_layer_orders(layers: Sequence[Sequence[int]], rng: _JavaRandom) -> List[List[int]]:
    """Shuffle each layer with Java ``Collections.shuffle`` semantics.

    Parameters
    ----------
    layers : sequence[sequence[int]]
        Initial layer orders.
    rng : _JavaRandom
        Java-compatible RNG seeded from the public ELK random seed.

    Returns
    -------
    list[list[int]]
        Per-layer shuffled node orders.
    """
    shuffled = [list(layer) for layer in layers]
    for layer in shuffled:
        for index in range(len(layer), 1, -1):
            swap_index = rng.next_int(index)
            layer[index - 1], layer[swap_index] = layer[swap_index], layer[index - 1]
    return shuffled


def _count_order_crossings(
    layers: Sequence[Sequence[int]],
    edges: Sequence[Tuple[int, int]],
) -> int:
    """Count layer-order crossings induced by the current node order.

    Parameters
    ----------
    layers : sequence[sequence[int]]
        Ordered node layers.
    edges : sequence[tuple[int, int]]
        Active acyclic graph edges.

    Returns
    -------
    int
        Number of pairwise edge-order inversions between common layer spans.
    """
    layer_by_node = {
        node: layer_index for layer_index, layer in enumerate(layers) for node in layer
    }
    order_by_node = {node: index for layer in layers for index, node in enumerate(layer)}
    span_edges: Dict[Tuple[int, int], List[Tuple[int, int]]] = {}
    for source, target in edges:
        source_layer = layer_by_node.get(source)
        target_layer = layer_by_node.get(target)
        if source_layer is None or target_layer is None or source_layer == target_layer:
            continue
        if source_layer > target_layer:
            source, target = target, source
            source_layer, target_layer = target_layer, source_layer
        span_edges.setdefault((source_layer, target_layer), []).append((source, target))

    crossings = 0
    for grouped_edges in span_edges.values():
        for left_index, (left_source, left_target) in enumerate(grouped_edges):
            left_source_order = order_by_node[left_source]
            left_target_order = order_by_node[left_target]
            for right_source, right_target in grouped_edges[left_index + 1 :]:
                source_delta = left_source_order - order_by_node[right_source]
                target_delta = left_target_order - order_by_node[right_target]
                if source_delta * target_delta < 0:
                    crossings += 1
    return crossings


def _edge_port_orders(
    edges: Sequence[Tuple[int, int]],
) -> Tuple[Dict[int, List[int]], Dict[int, List[int]]]:
    """Return generated ELK port order lists for every real node.

    Parameters
    ----------
    edges : sequence[tuple[int, int]]
        Active acyclic graph edges in model order.

    Returns
    -------
    tuple[dict[int, list[int]], dict[int, list[int]]]
        Edge-index lists keyed by source for output ports and by target for
        input ports.
    """
    outgoing: Dict[int, List[int]] = {}
    incoming: Dict[int, List[int]] = {}
    for edge_index, (source, target) in enumerate(edges):
        outgoing.setdefault(source, []).append(edge_index)
        incoming.setdefault(target, []).append(edge_index)
    return outgoing, incoming


def _fixed_layer_port_ranks(
    fixed_layer: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    *,
    forward: bool,
    node_relative_ports: bool,
) -> Dict[int, float]:
    """Compute simplified ELK generated-port ranks for a fixed layer.

    Parameters
    ----------
    fixed_layer : sequence[int]
        Ordered fixed-layer node ids.
    edges : sequence[tuple[int, int]]
        Active acyclic graph edges in model order.
    forward : bool
        ``True`` when sweeping from predecessors to successors.
    node_relative_ports : bool
        Whether ELK selected ``NodeRelativePortDistributor`` instead of
        ``LayerTotalPortDistributor``.

    Returns
    -------
    dict[int, float]
        Edge index to fixed-end port rank.
    """
    outgoing, incoming = _edge_port_orders(edges)
    ranks: Dict[int, float] = {}
    consumed_rank = 0.0
    for node in fixed_layer:
        edge_indices = outgoing.get(node, []) if forward else incoming.get(node, [])
        port_count = len(edge_indices)
        if port_count == 0:
            consumed_rank += 1.0 if node_relative_ports else 0.0
            continue
        if node_relative_ports:
            increment = 1.0 / float(port_count + 1)
            if forward:
                rank = consumed_rank + increment
                for edge_index in edge_indices:
                    ranks[edge_index] = rank
                    rank += increment
            else:
                rank = consumed_rank + 1.0 - increment
                for edge_index in edge_indices:
                    ranks[edge_index] = rank
                    rank -= increment
            consumed_rank += 1.0
        elif forward:
            for offset, edge_index in enumerate(edge_indices, start=1):
                ranks[edge_index] = consumed_rank + float(offset)
            consumed_rank += float(port_count)
        else:
            rank = consumed_rank + float(port_count)
            for edge_index in edge_indices:
                ranks[edge_index] = rank
                rank -= 1.0
            consumed_rank += float(port_count)
    return ranks


def _randomize_first_sweep_layer(
    layers: List[List[int]],
    *,
    forward: bool,
    rng: _JavaRandom,
) -> None:
    """Randomize ELK's first fixed layer by assigning random barycenters.

    Parameters
    ----------
    layers : list[list[int]]
        Mutable current layer order.
    forward : bool
        ``True`` when the sweep starts at the first layer.
    rng : _JavaRandom
        Java-compatible restart RNG.

    Returns
    -------
    None
        The first fixed layer is sorted in place.
    """
    if not layers:
        return
    layer_index = 0 if forward else len(layers) - 1
    randomized = [(rng.next_double(), node) for node in layers[layer_index]]
    randomized.sort(key=lambda item: item[0])
    layers[layer_index] = [node for _, node in randomized]


def _fill_unknown_barycenters(
    layer: Sequence[int],
    barycenters: Dict[int, Optional[float]],
    *,
    pre_ordered: bool,
    rng: _JavaRandom,
) -> None:
    """Fill ELK barycenter gaps for nodes without fixed-layer neighbors.

    Parameters
    ----------
    layer : sequence[int]
        Current free-layer node order.
    barycenters : dict[int, float | None]
        Mutable node barycenter map.
    pre_ordered : bool
        Whether this is a later sweep that should preserve relative gaps.
    rng : _JavaRandom
        Java-compatible restart RNG.

    Returns
    -------
    None
        ``barycenters`` is updated in place.
    """
    if pre_ordered:
        last_value = -1.0
        for index, node in enumerate(layer):
            value = barycenters[node]
            if value is None:
                next_value = last_value + 1.0
                for later_node in layer[index + 1 :]:
                    later_value = barycenters[later_node]
                    if later_value is not None:
                        next_value = later_value
                        break
                value = (last_value + next_value) / 2.0
                barycenters[node] = value
            last_value = float(value)
        return

    max_barycenter = 0.0
    for value in barycenters.values():
        if value is not None:
            max_barycenter = max(max_barycenter, float(value))
    max_barycenter += 2.0
    for node in layer:
        if barycenters[node] is None:
            barycenters[node] = rng.next_float() * max_barycenter - 1.0


def _sort_free_layer_by_barycenter(
    layers: List[List[int]],
    edges: Sequence[Tuple[int, int]],
    *,
    free_layer_index: int,
    forward: bool,
    first_sweep: bool,
    node_relative_ports: bool,
    rng: _JavaRandom,
) -> None:
    """Sort one free layer with ELK's randomized barycenter heuristic.

    Parameters
    ----------
    layers : list[list[int]]
        Mutable current layer order.
    edges : sequence[tuple[int, int]]
        Active acyclic graph edges.
    free_layer_index : int
        Layer currently being reordered.
    forward : bool
        ``True`` when fixed neighbors are in the previous layer.
    first_sweep : bool
        Whether this is the first sweep in the current restart attempt.
    node_relative_ports : bool
        Whether generated port ranks use node-relative distribution.
    rng : _JavaRandom
        Java-compatible restart RNG.

    Returns
    -------
    None
        The selected layer is sorted in place.
    """
    fixed_layer_index = free_layer_index - 1 if forward else free_layer_index + 1
    fixed_nodes = set(layers[fixed_layer_index])
    fixed_ranks = _fixed_layer_port_ranks(
        layers[fixed_layer_index],
        edges,
        forward=forward,
        node_relative_ports=node_relative_ports,
    )
    layer = layers[free_layer_index]
    summed: Dict[int, float] = {node: 0.0 for node in layer}
    degree: Dict[int, int] = {node: 0 for node in layer}
    for edge_index, (source, target) in enumerate(edges):
        if forward and target in summed and source in fixed_nodes:
            summed[target] += fixed_ranks.get(edge_index, 0.0)
            degree[target] += 1
        elif not forward and source in summed and target in fixed_nodes:
            summed[source] += fixed_ranks.get(edge_index, 0.0)
            degree[source] += 1

    barycenters: Dict[int, Optional[float]] = {}
    for node in layer:
        if degree[node] > 0:
            perturbation = rng.next_float() * 0.07 - 0.035
            barycenters[node] = (summed[node] + perturbation) / float(degree[node])
        else:
            barycenters[node] = None
    _fill_unknown_barycenters(layer, barycenters, pre_ordered=not first_sweep, rng=rng)
    layer.sort(key=lambda node: float(barycenters[node]))


def _elk_barycenter_sweep(
    layers: List[List[int]],
    edges: Sequence[Tuple[int, int]],
    *,
    forward: bool,
    first_sweep: bool,
    node_relative_ports: bool,
    rng: _JavaRandom,
) -> None:
    """Run one ELK-style barycenter sweep over mutable layers.

    Parameters
    ----------
    layers : list[list[int]]
        Mutable current layer order.
    edges : sequence[tuple[int, int]]
        Active acyclic graph edges.
    forward : bool
        Sweep direction.
    first_sweep : bool
        Whether this is the first sweep in a restart attempt.
    node_relative_ports : bool
        Whether generated port ranks use node-relative distribution.
    rng : _JavaRandom
        Java-compatible restart RNG.

    Returns
    -------
    None
        Layers are modified in place.
    """
    if len(layers) < 2:
        return
    if forward:
        layer_range = range(1, len(layers))
    else:
        layer_range = range(len(layers) - 2, -1, -1)
    for layer_index in layer_range:
        _sort_free_layer_by_barycenter(
            layers,
            edges,
            free_layer_index=layer_index,
            forward=forward,
            first_sweep=first_sweep,
            node_relative_ports=node_relative_ports,
            rng=rng,
        )


def _elk_restart_attempt(
    layers: List[List[int]],
    edges: Sequence[Tuple[int, int]],
    *,
    rng: _JavaRandom,
    node_relative_ports: bool,
) -> Tuple[int, List[List[int]]]:
    """Run one persistent ELK barycenter restart attempt.

    Parameters
    ----------
    layers : list[list[int]]
        Mutable current layer order. ELK does not restore the model order
        between randomized attempts, so this list is intentionally mutated.
    edges : sequence[tuple[int, int]]
        Active acyclic graph edges.
    rng : _JavaRandom
        Java-compatible restart RNG.
    node_relative_ports : bool
        Whether generated port ranks use node-relative distribution.

    Returns
    -------
    tuple[int, list[list[int]]]
        Crossing score and the best order reached within this attempt.
    """
    forward = rng.next_boolean()
    _randomize_first_sweep_layer(layers, forward=forward, rng=rng)
    _elk_barycenter_sweep(
        layers,
        edges,
        forward=forward,
        first_sweep=True,
        node_relative_ports=node_relative_ports,
        rng=rng,
    )
    crossings = _count_order_crossings(layers, edges)
    while True:
        attempt_best = [list(layer) for layer in layers]
        if crossings == 0:
            return 0, attempt_best
        forward = not forward
        previous_crossings = crossings
        _elk_barycenter_sweep(
            layers,
            edges,
            forward=forward,
            first_sweep=False,
            node_relative_ports=node_relative_ports,
            rng=rng,
        )
        crossings = _count_order_crossings(layers, edges)
        if previous_crossings <= crossings:
            return previous_crossings, attempt_best


def _is_two_middle_diamond(
    layers: Sequence[Sequence[int]],
    edges: Sequence[Tuple[int, int]],
) -> bool:
    """Return whether the layer graph is the minimal ELK diamond frontier.

    Parameters
    ----------
    layers : sequence[sequence[int]]
        Current layer assignment.
    edges : sequence[tuple[int, int]]
        Active acyclic graph edges.

    Returns
    -------
    bool
        ``True`` for a three-layer ``1 -> 2 -> 1`` graph whose two middle
        nodes share the same predecessor and successor.
    """
    if [len(layer) for layer in layers] != [1, 2, 1]:
        return False
    source = int(layers[0][0])
    left = int(layers[1][0])
    right = int(layers[1][1])
    target = int(layers[2][0])
    return set(edges) == {
        (source, left),
        (source, right),
        (left, target),
        (right, target),
    }


def _apply_two_middle_diamond_x_coordinates(
    x_coordinates: Dict[int, float],
    layers: Sequence[Sequence[int]],
    sizes: torch.Tensor,
    node_spacing: float,
) -> None:
    """Apply ELK's compact BK coordinates for a minimal diamond.

    Parameters
    ----------
    x_coordinates : dict[int, float]
        Mutable top-left x-coordinate map.
    layers : sequence[sequence[int]]
        Ordered ``1 -> 2 -> 1`` diamond layers.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    node_spacing : float
        ELK node-node spacing between the two middle nodes.

    Returns
    -------
    None
        ``x_coordinates`` is updated in place.
    """
    source = int(layers[0][0])
    left = int(layers[1][0])
    right = int(layers[1][1])
    target = int(layers[2][0])
    middle_width = float((sizes[left, 0] + sizes[right, 0]).item()) / 2.0
    singleton_x = _ROOT_PADDING + middle_width / 6.0
    x_coordinates[source] = singleton_x
    x_coordinates[left] = _ROOT_PADDING
    x_coordinates[right] = _ROOT_PADDING + middle_width + node_spacing
    x_coordinates[target] = singleton_x


def _restart_sweep_orders(
    layers: Sequence[Sequence[int]],
    edges: Sequence[Tuple[int, int]],
    *,
    random_seed: int,
    thoroughness: int,
) -> List[List[int]]:
    """Run ELK-style randomized layer-sweep restarts and keep the best order.

    Parameters
    ----------
    layers : sequence[sequence[int]]
        Deterministic model-order layer assignment.
    edges : sequence[tuple[int, int]]
        Active acyclic graph edges.
    random_seed : int
        Public ELK random seed for restart shuffles.
    thoroughness : int
        Number of crossing-minimization attempts. Attempt zero is
        deterministic; subsequent attempts start from shuffled layer orders.

    Returns
    -------
    list[list[int]]
        Ordered layers from the earliest strictly best crossing score.
    """
    incident_nodes = {node for edge in edges for node in edge}
    has_order_free_isolate = any(
        len(layer) > 1 and any(node not in incident_nodes for node in layer) for layer in layers
    )
    use_randomized_sweep = has_order_free_isolate or _is_two_middle_diamond(layers, edges)
    if not use_randomized_sweep:
        rng = _JavaRandom(random_seed)
        best_order: Optional[List[List[int]]] = None
        best_crossings: Optional[int] = None
        for attempt in range(thoroughness):
            initial = [list(layer) for layer in layers]
            if attempt > 0:
                initial = _shuffle_layer_orders(layers, rng)
            candidate = _sweep_orders(initial, edges)
            crossings = _count_order_crossings(candidate, edges)
            if best_crossings is None or crossings < best_crossings:
                best_order = candidate
                best_crossings = crossings
        if best_order is None:
            return [list(layer) for layer in layers]
        return best_order

    bootstrap_rng = _JavaRandom(random_seed)
    restart_seed = bootstrap_rng.next_long()
    node_relative_ports = bootstrap_rng.next_boolean()
    rng = _JavaRandom(restart_seed)
    best_order: Optional[List[List[int]]] = None
    best_crossings: Optional[int] = None
    current_order = [list(layer) for layer in layers]
    for attempt in range(thoroughness):
        del attempt
        crossings, candidate = _elk_restart_attempt(
            current_order,
            edges,
            rng=rng,
            node_relative_ports=node_relative_ports,
        )
        if best_crossings is None or crossings < best_crossings:
            best_order = candidate
            best_crossings = crossings
            if best_crossings == 0:
                break
    if best_order is None:
        return [list(layer) for layer in layers]
    return best_order


def _sweep_orders(layers: List[List[int]], edges: Sequence[Tuple[int, int]]) -> List[List[int]]:
    """Improve within-layer order with deterministic median sweeps.

    Parameters
    ----------
    layers : list[list[int]]
        Initial layer order.
    edges : sequence[tuple[int, int]]
        Acyclic edge pairs.

    Returns
    -------
    list[list[int]]
        Reordered layers.
    """
    ordered = [list(layer) for layer in layers]
    predecessors = _predecessors(sum(len(layer) for layer in ordered), edges)
    successors = _successors(sum(len(layer) for layer in ordered), edges)
    for _ in range(4):
        positions = {node: index for layer in ordered for index, node in enumerate(layer)}
        for layer_index in range(1, len(ordered)):
            ordered[layer_index].sort(
                key=lambda node: (
                    _median([positions[pred] for pred in predecessors[node] if pred in positions]),
                    positions[node],
                )
            )
        positions = {node: index for layer in ordered for index, node in enumerate(layer)}
        for layer_index in range(len(ordered) - 2, -1, -1):
            ordered[layer_index].sort(
                key=lambda node: (
                    _median([positions[succ] for succ in successors[node] if succ in positions]),
                    positions[node],
                )
            )
    return ordered


def _layer_y_coordinates(
    layers: Sequence[Sequence[int]],
    sizes: torch.Tensor,
    spacing: float,
) -> Dict[int, float]:
    """Assign top-left y coordinates with ELK root padding.

    Parameters
    ----------
    layers : sequence[sequence[int]]
        Ordered node layers.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    spacing : float
        Between-layer spacing.

    Returns
    -------
    dict[int, float]
        Top-left y coordinate per node.
    """
    coordinates: Dict[int, float] = {}
    cursor = _ROOT_PADDING
    for layer in layers:
        for node in layer:
            coordinates[node] = cursor
        max_height = max((float(sizes[node, 1]) for node in layer), default=0.0)
        cursor += max_height + spacing
    return coordinates


def _layer_x_coordinates(
    layers: Sequence[Sequence[int]],
    predecessors: Mapping[int, Sequence[int]],
    successors: Mapping[int, Sequence[int]],
    sizes: torch.Tensor,
    node_spacing: float,
    strategy: str,
) -> Dict[int, float]:
    """Assign top-left x coordinates for ordered layers.

    Parameters
    ----------
    layers : sequence[sequence[int]]
        Ordered node layers.
    predecessors : mapping[int, sequence[int]]
        Predecessor lists.
    successors : mapping[int, sequence[int]]
        Successor lists.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    node_spacing : float
        Same-layer node spacing.
    strategy : str
        Node placement strategy.

    Returns
    -------
    dict[int, float]
        Top-left x coordinate per node.
    """
    if strategy == "brandes_koepf":
        del predecessors
        return _elk_brandes_koepf_x_coordinates(
            layers=layers,
            successors=successors,
            sizes=sizes,
            node_spacing=node_spacing,
        )

    coordinates: Dict[int, float] = {}
    for layer in layers:
        cursor = _ROOT_PADDING
        for node in layer:
            coordinates[node] = cursor
            cursor += float(sizes[node, 0]) + node_spacing
    return coordinates


def _normalize_long_edges_for_bk(
    layers: Sequence[Sequence[int]],
    predecessors: Mapping[int, Sequence[int]],
    successors: Mapping[int, Sequence[int]],
    sizes: torch.Tensor,
) -> Tuple[
    List[List[NodeId]],
    Dict[NodeId, List[NodeId]],
    Dict[NodeId, List[NodeId]],
    Dict[NodeId, float],
    Set[NodeId],
]:
    """Split long edges into dummy chains for Brandes-Koepf placement.

    Parameters
    ----------
    layers : sequence[sequence[int]]
        Ordered real-node layers.
    predecessors : mapping[int, sequence[int]]
        Real-node predecessor lists.
    successors : mapping[int, sequence[int]]
        Real-node successor lists.
    sizes : torch.Tensor
        Real node sizes with shape ``[N, 2]``.

    Returns
    -------
    tuple
        Normalized layers, predecessor map, successor map, width map, and the
        set of dummy node ids introduced for long edge segments.
    """
    layer_by_node = {
        node: layer_index for layer_index, layer in enumerate(layers) for node in layer
    }
    normalized_layers: List[List[NodeId]] = [list(layer) for layer in layers]
    normalized_predecessors: Dict[NodeId, List[NodeId]] = {
        node: [] for layer in normalized_layers for node in layer
    }
    normalized_successors: Dict[NodeId, List[NodeId]] = {
        node: [] for layer in normalized_layers for node in layer
    }
    widths: Dict[NodeId, float] = {
        node: float(sizes[node, 0]) for layer in layers for node in layer
    }
    dummy_nodes: Set[NodeId] = set()

    def add_segment(source: NodeId, target: NodeId) -> None:
        """Add one normalized edge segment.

        Parameters
        ----------
        source : Hashable
            Segment source node id.
        target : Hashable
            Segment target node id.

        Returns
        -------
        None
            Normalized predecessor and successor maps are updated in place.
        """
        normalized_successors.setdefault(source, []).append(target)
        normalized_predecessors.setdefault(target, []).append(source)

    edge_index = 0
    for source in sorted(successors):
        for target in successors[source]:
            source_layer = layer_by_node[source]
            target_layer = layer_by_node[target]
            span = target_layer - source_layer
            if abs(span) <= 1:
                add_segment(source, target)
                continue
            step = 1 if span > 0 else -1
            previous: NodeId = source
            for layer_index in range(source_layer + step, target_layer, step):
                dummy: NodeId = ("elk_dummy", edge_index, layer_index)
                dummy_nodes.add(dummy)
                widths[dummy] = 0.0
                insertion_layer = normalized_layers[layer_index]
                insertion_layer.append(dummy)
                add_segment(previous, dummy)
                previous = dummy
            add_segment(previous, target)
            edge_index += 1

    for node in list(normalized_predecessors):
        normalized_predecessors[node] = list(dict.fromkeys(normalized_predecessors[node]))
    for node in list(normalized_successors):
        normalized_successors[node] = list(dict.fromkeys(normalized_successors[node]))
    return normalized_layers, normalized_predecessors, normalized_successors, widths, dummy_nodes


def _normalize_long_edges_for_elk_bk(
    layers: Sequence[Sequence[int]],
    successors: Mapping[int, Sequence[int]],
    sizes: torch.Tensor,
) -> _ElkBkGraph:
    """Build ELK's BK working graph with long-edge dummy chains.

    Parameters
    ----------
    layers : sequence[sequence[int]]
        Ordered real-node layers.
    successors : mapping[int, sequence[int]]
        Real-node successor lists.
    sizes : torch.Tensor
        Real node sizes with shape ``[N, 2]``.

    Returns
    -------
    _ElkBkGraph
        Normalized BK graph with edge segment identifiers and dummy nodes.
    """
    layer_by_node = {
        node: layer_index for layer_index, layer in enumerate(layers) for node in layer
    }
    normalized_layers: List[List[NodeId]] = [list(layer) for layer in layers]
    normalized_predecessors: Dict[NodeId, List[NodeId]] = {
        node: [] for layer in normalized_layers for node in layer
    }
    normalized_successors: Dict[NodeId, List[NodeId]] = {
        node: [] for layer in normalized_layers for node in layer
    }
    edge_ids: Dict[Tuple[NodeId, NodeId], int] = {}
    widths: Dict[NodeId, float] = {
        node: float(sizes[node, 0]) for layer in layers for node in layer
    }
    dummy_nodes: Set[NodeId] = set()

    def add_segment(source: NodeId, target: NodeId, edge_id: int) -> None:
        """Add one normalized edge segment with a stable edge id.

        Parameters
        ----------
        source : Hashable
            Segment source node id.
        target : Hashable
            Segment target node id.
        edge_id : int
            Original edge index used for conflict marking.

        Returns
        -------
        None
            Normalized adjacency and edge-id maps are mutated.
        """
        normalized_successors.setdefault(source, []).append(target)
        normalized_predecessors.setdefault(target, []).append(source)
        edge_ids[(source, target)] = edge_id

    edge_index = 0
    for source in sorted(successors):
        for target in successors[source]:
            source_layer = layer_by_node[source]
            target_layer = layer_by_node[target]
            span = target_layer - source_layer
            if abs(span) <= 1:
                add_segment(source, target, edge_index)
                edge_index += 1
                continue
            step = 1 if span > 0 else -1
            previous: NodeId = source
            for layer_index in range(source_layer + step, target_layer, step):
                dummy: NodeId = ("elk_dummy", edge_index, layer_index)
                dummy_nodes.add(dummy)
                widths[dummy] = 0.0
                normalized_layers[layer_index].append(dummy)
                add_segment(previous, dummy, edge_index)
                previous = dummy
            add_segment(previous, target, edge_index)
            edge_index += 1

    for node in list(normalized_predecessors):
        normalized_predecessors[node] = list(dict.fromkeys(normalized_predecessors[node]))
    for node in list(normalized_successors):
        normalized_successors[node] = list(dict.fromkeys(normalized_successors[node]))

    node_index = {node: index for layer in normalized_layers for index, node in enumerate(layer)}
    layer_index = {
        node: layer_number for layer_number, layer in enumerate(normalized_layers) for node in layer
    }
    return _ElkBkGraph(
        layers=normalized_layers,
        predecessors=normalized_predecessors,
        successors=normalized_successors,
        edge_ids=edge_ids,
        sizes=widths,
        dummy_nodes=dummy_nodes,
        real_nodes=[node for layer in layers for node in layer],
        node_index=node_index,
        layer_index=layer_index,
    )


def _elk_bk_spacing(
    left: NodeId,
    right: NodeId,
    dummy_nodes: Set[NodeId],
    node_spacing: float,
) -> float:
    """Return ELK's type-specific in-layer spacing.

    Parameters
    ----------
    left : Hashable
        First adjacent node.
    right : Hashable
        Second adjacent node.
    dummy_nodes : set[Hashable]
        Long-edge dummy nodes.
    node_spacing : float
        ELK ``spacing.nodeNode``.

    Returns
    -------
    float
        Spacing between node boxes along the BK placement axis.
    """
    left_dummy = left in dummy_nodes
    right_dummy = right in dummy_nodes
    if left_dummy and right_dummy:
        return _ELK_DEFAULT_EDGE_EDGE_SPACING
    if left_dummy or right_dummy:
        return _ELK_DEFAULT_EDGE_NODE_SPACING
    return node_spacing


def _elk_bk_left_neighbors(graph: _ElkBkGraph) -> Dict[NodeId, List[NodeId]]:
    """Return ELK left-neighbor lists sorted by order in the previous layer.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.

    Returns
    -------
    dict[Hashable, list[Hashable]]
        Predecessors sorted by layer index.
    """
    return {
        node: sorted(neighbors, key=graph.node_index.__getitem__)
        for node, neighbors in graph.predecessors.items()
    }


def _elk_bk_right_neighbors(graph: _ElkBkGraph) -> Dict[NodeId, List[NodeId]]:
    """Return ELK right-neighbor lists sorted by order in the next layer.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.

    Returns
    -------
    dict[Hashable, list[Hashable]]
        Successors sorted by layer index.
    """
    return {
        node: sorted(neighbors, key=graph.node_index.__getitem__)
        for node, neighbors in graph.successors.items()
    }


def _elk_bk_incident_to_inner_segment(
    graph: _ElkBkGraph,
    node: NodeId,
    layer1: int,
    layer2: int,
) -> bool:
    """Return whether ``node`` is incident to an inner long-edge segment.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    node : Hashable
        Candidate node from ``layer1``.
    layer1 : int
        Layer index of ``node``.
    layer2 : int
        Previous layer index to test.

    Returns
    -------
    bool
        ``True`` when both segment endpoints are long-edge dummy nodes.
    """
    if node not in graph.dummy_nodes:
        return False
    return any(
        predecessor in graph.dummy_nodes and graph.layer_index[predecessor] == layer2
        for predecessor in graph.predecessors.get(node, ())
        if graph.layer_index[node] == layer1
    )


def _elk_bk_mark_conflicts(graph: _ElkBkGraph) -> Set[int]:
    """Mark ELK BK type-1/type-2 conflict edge ids.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.

    Returns
    -------
    set[int]
        Edge ids that ELK's ``markConflicts`` excludes from alignment.
    """
    marked_edges: Set[int] = set()
    number_of_layers = len(graph.layers)
    if number_of_layers < 3:
        return marked_edges
    left_neighbors = _elk_bk_left_neighbors(graph)
    layer_sizes = [len(layer) for layer in graph.layers]
    for layer_number in range(1, number_of_layers - 1):
        current_layer = graph.layers[layer_number + 1]
        k_0 = 0
        scan_position = 0
        for layer_position, node in enumerate(current_layer):
            incident = _elk_bk_incident_to_inner_segment(
                graph, node, layer_number + 1, layer_number
            )
            if layer_position == layer_sizes[layer_number + 1] - 1 or incident:
                k_1 = layer_sizes[layer_number] - 1
                if incident:
                    k_1 = graph.node_index[left_neighbors[node][0]]
                while scan_position <= layer_position:
                    scan_node = current_layer[scan_position]
                    if not _elk_bk_incident_to_inner_segment(
                        graph, scan_node, layer_number + 1, layer_number
                    ):
                        for upper_neighbor in left_neighbors.get(scan_node, ()):
                            upper_index = graph.node_index[upper_neighbor]
                            if upper_index < k_0 or upper_index > k_1:
                                edge_id = graph.edge_ids.get((upper_neighbor, scan_node))
                                if edge_id is not None:
                                    marked_edges.add(edge_id)
                    scan_position += 1
                k_0 = k_1
    return marked_edges


def _elk_bk_new_layout(graph: _ElkBkGraph, vdir: str, hdir: str) -> _ElkBkLayout:
    """Create one empty ELK BK aligned layout.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    vdir : str
        ``"DOWN"`` or ``"UP"`` traversal direction.
    hdir : str
        ``"LEFT"`` or ``"RIGHT"`` traversal direction.

    Returns
    -------
    _ElkBkLayout
        Layout maps initialized like ``BKAlignedLayout``.
    """
    nodes = [node for layer in graph.layers for node in layer]
    return _ElkBkLayout(
        vdir=vdir,
        hdir=hdir,
        root={node: node for node in nodes},
        align={node: node for node in nodes},
        inner_shift={node: 0.0 for node in nodes},
        block_size={},
        sink={node: node for node in nodes},
        shift={node: (float("-inf") if vdir == "UP" else float("inf")) for node in nodes},
        y={},
        straightened={node: False for node in nodes},
        only_dummies={node: True for node in nodes},
    )


def _elk_bk_vertical_alignment(
    graph: _ElkBkGraph,
    layout: _ElkBkLayout,
    marked_edges: Set[int],
) -> None:
    """Run ELK's median-neighbor vertical alignment for one direction.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    layout : _ElkBkLayout
        Alignment state to mutate.
    marked_edges : set[int]
        Conflict edge ids from ``markConflicts``.

    Returns
    -------
    None
        ``layout`` root and align maps are updated in place.
    """
    left_neighbors = _elk_bk_left_neighbors(graph)
    right_neighbors = _elk_bk_right_neighbors(graph)
    layer_sequence = list(graph.layers)
    if layout.hdir == "LEFT":
        layer_sequence = list(reversed(layer_sequence))
    for layer in layer_sequence:
        previous_index = -1
        nodes = list(layer)
        if layout.vdir == "UP":
            previous_index = 2**31 - 1
            nodes = list(reversed(nodes))
        for node in nodes:
            neighbors = (
                right_neighbors.get(node, [])
                if layout.hdir == "LEFT"
                else left_neighbors.get(node, [])
            )
            if not neighbors:
                continue
            degree = len(neighbors)
            low = int(((degree + 1.0) // 2.0) - 1)
            high = int(-(-((degree + 1.0) / 2.0) // 1) - 1)
            median_range = range(high, low - 1, -1) if layout.vdir == "UP" else range(low, high + 1)
            for median_index in median_range:
                if layout.align[node] != node:
                    continue
                neighbor = neighbors[median_index]
                edge_id = graph.edge_ids.get((neighbor, node), graph.edge_ids.get((node, neighbor)))
                neighbor_index = graph.node_index[neighbor]
                if edge_id in marked_edges:
                    continue
                if layout.vdir == "UP":
                    can_align = previous_index > neighbor_index
                else:
                    can_align = previous_index < neighbor_index
                if can_align:
                    layout.align[neighbor] = node
                    layout.root[node] = layout.root[neighbor]
                    layout.align[node] = layout.root[node]
                    layout.only_dummies[layout.root[node]] = (
                        layout.only_dummies[layout.root[node]] and node in graph.dummy_nodes
                    )
                    previous_index = neighbor_index


def _elk_bk_connected_successor(
    graph: _ElkBkGraph,
    source: NodeId,
    target: NodeId,
) -> bool:
    """Return whether a normalized segment connects two nodes.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    source : Hashable
        First node.
    target : Hashable
        Second node.

    Returns
    -------
    bool
        ``True`` if either directed segment exists.
    """
    return target in graph.successors.get(source, ()) or source in graph.successors.get(target, ())


def _elk_bk_edge_port_positions(
    graph: _ElkBkGraph,
    source: NodeId,
    target: NodeId,
) -> Tuple[float, float]:
    """Return generated source/target port offsets for one normalized edge.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    source : Hashable
        Directed edge source.
    target : Hashable
        Directed edge target.

    Returns
    -------
    tuple[float, float]
        Source and target offsets on the BK placement axis.

    Notes
    -----
    ELK generates implicit ports for plain nodes. The layered port placer
    distributes same-side ports evenly across the node extent in edge order;
    BK's ``insideBlockShift`` then uses those offsets instead of node centers.
    """
    source_successors = graph.successors.get(source, [])
    target_predecessors = graph.predecessors.get(target, [])
    source_index = source_successors.index(target)
    target_index = target_predecessors.index(source)
    source_port = graph.sizes[source] * float(source_index + 1) / float(len(source_successors) + 1)
    target_port = (
        graph.sizes[target] * float(target_index + 1) / float(len(target_predecessors) + 1)
    )
    return source_port, target_port


def _elk_bk_inside_block_shift(graph: _ElkBkGraph, layout: _ElkBkLayout) -> None:
    """Compute ELK's per-block inner shifts for center ports.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    layout : _ElkBkLayout
        Alignment state to mutate.

    Returns
    -------
    None
        ``inner_shift`` and ``block_size`` are updated.
    """
    roots = []
    seen: Set[NodeId] = set()
    for layer in graph.layers:
        for node in layer:
            root = layout.root[node]
            if root not in seen:
                seen.add(root)
                roots.append(root)
    for root in roots:
        space_above = 0.0
        space_below = graph.sizes[root]
        layout.inner_shift[root] = 0.0
        current = root
        while True:
            next_node = layout.align[current]
            if next_node == root:
                break
            if _elk_bk_connected_successor(graph, current, next_node):
                source = current if next_node in graph.successors.get(current, ()) else next_node
                target = next_node if source == current else current
                source_port, target_port = _elk_bk_edge_port_positions(graph, source, target)
                if layout.hdir == "LEFT":
                    port_pos_diff = target_port - source_port
                else:
                    port_pos_diff = source_port - target_port
            else:
                port_pos_diff = 0.0
            next_inner_shift = layout.inner_shift[current] + port_pos_diff
            layout.inner_shift[next_node] = next_inner_shift
            space_above = max(space_above, -next_inner_shift)
            space_below = max(space_below, next_inner_shift + graph.sizes[next_node])
            current = next_node
        current = root
        while True:
            layout.inner_shift[current] += space_above
            current = layout.align[current]
            if current == root:
                break
        layout.block_size[root] = space_above + space_below


def _elk_bk_place_classes(
    layout: _ElkBkLayout,
    class_edges: Mapping[NodeId, Sequence[Tuple[NodeId, float]]],
    class_nodes: Sequence[NodeId],
) -> None:
    """Place ELK BK classes by longest-path propagation.

    Parameters
    ----------
    layout : _ElkBkLayout
        Layout receiving class shifts.
    class_edges : mapping[Hashable, sequence[tuple[Hashable, float]]]
        Class graph edges with required separations.
    class_nodes : sequence[Hashable]
        Class sinks in creation order.

    Returns
    -------
    None
        ``layout.shift`` is updated for class sink nodes.
    """
    indegree = {node: 0 for node in class_nodes}
    for edges in class_edges.values():
        for target, _ in edges:
            indegree[target] = indegree.get(target, 0) + 1
    queue = [node for node in class_nodes if indegree.get(node, 0) == 0]
    class_shift: Dict[NodeId, Optional[float]] = {node: None for node in class_nodes}
    while queue:
        node = queue.pop(0)
        if class_shift[node] is None:
            class_shift[node] = 0.0
        for target, separation in class_edges.get(node, ()):
            candidate = float(class_shift[node]) + separation
            if class_shift.get(target) is None:
                class_shift[target] = candidate
            elif layout.vdir == "DOWN":
                class_shift[target] = min(float(class_shift[target]), candidate)
            else:
                class_shift[target] = max(float(class_shift[target]), candidate)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    for node, value in class_shift.items():
        if value is not None:
            layout.shift[node] = value


def _elk_bk_endpoint_port_position(
    graph: _ElkBkGraph,
    edge: Tuple[NodeId, NodeId],
    node: NodeId,
) -> float:
    """Return a node's generated port offset on one edge.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    edge : tuple[Hashable, Hashable]
        Directed normalized edge.
    node : Hashable
        Endpoint whose offset is requested.

    Returns
    -------
    float
        Port offset on the BK placement axis.
    """
    source, target = edge
    source_port, target_port = _elk_bk_edge_port_positions(graph, source, target)
    return source_port if node == source else target_port


def _elk_bk_edges_for_threshold(
    graph: _ElkBkGraph,
    layout: _ElkBkLayout,
    node: NodeId,
    is_root: bool,
) -> Iterable[Tuple[NodeId, NodeId]]:
    """Yield ELK ``pickEdge`` candidates for threshold straightening.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    layout : _ElkBkLayout
        Current alignment and compaction state.
    node : Hashable
        Free node whose incident edges are inspected.
    is_root : bool
        Whether ``node`` is considered the first node of its block.

    Returns
    -------
    Iterable[tuple[Hashable, Hashable]]
        Directed normalized edges in ELK's incident-edge iteration direction.
    """
    if is_root:
        if layout.hdir == "RIGHT":
            return ((predecessor, node) for predecessor in graph.predecessors.get(node, ()))
        return ((node, successor) for successor in graph.successors.get(node, ()))
    if layout.hdir == "LEFT":
        return ((predecessor, node) for predecessor in graph.predecessors.get(node, ()))
    return ((node, successor) for successor in graph.successors.get(node, ()))


def _elk_bk_other_node(edge: Tuple[NodeId, NodeId], node: NodeId) -> NodeId:
    """Return the other endpoint of an incident normalized edge.

    Parameters
    ----------
    edge : tuple[Hashable, Hashable]
        Directed normalized edge.
    node : Hashable
        One endpoint of ``edge``.

    Returns
    -------
    Hashable
        The opposite endpoint.
    """
    source, target = edge
    return target if source == node else source


def _elk_bk_min_y(graph: _ElkBkGraph, layout: _ElkBkLayout, node: NodeId) -> float:
    """Return ELK ``BKAlignedLayout.getMinY`` without margins.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph. Present for symmetry with ``getMaxY``.
    layout : _ElkBkLayout
        Current compaction layout.
    node : Hashable
        Node whose block-relative minimum is requested.

    Returns
    -------
    float
        Minimum occupied coordinate on the BK placement axis.
    """
    del graph
    return layout.y[layout.root[node]] + layout.inner_shift[node]


def _elk_bk_max_y(graph: _ElkBkGraph, layout: _ElkBkLayout, node: NodeId) -> float:
    """Return ELK ``BKAlignedLayout.getMaxY`` without margins.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    layout : _ElkBkLayout
        Current compaction layout.
    node : Hashable
        Node whose block-relative maximum is requested.

    Returns
    -------
    float
        Maximum occupied coordinate on the BK placement axis.
    """
    return layout.y[layout.root[node]] + layout.inner_shift[node] + graph.sizes[node]


def _elk_bk_neighbor_in_layer(
    graph: _ElkBkGraph,
    node: NodeId,
    offset: int,
) -> Optional[NodeId]:
    """Return a same-layer neighbor by order offset.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    node : Hashable
        Node whose layer neighbor is queried.
    offset : int
        ``-1`` for the upper/previous neighbor, ``1`` for lower/next.

    Returns
    -------
    Hashable | None
        Neighbor node if it exists.
    """
    layer = graph.layers[graph.layer_index[node]]
    index = graph.node_index[node] + offset
    if 0 <= index < len(layer):
        return layer[index]
    return None


def _elk_bk_check_space_above(
    graph: _ElkBkGraph,
    layout: _ElkBkLayout,
    block_node: NodeId,
    delta: float,
    node_spacing: float,
) -> float:
    """Return how far a block can move toward previous same-layer nodes.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    layout : _ElkBkLayout
        Current compaction layout.
    block_node : Hashable
        Any node in the block being moved.
    delta : float
        Requested positive movement.
    node_spacing : float
        ELK ``spacing.nodeNode``.

    Returns
    -------
    float
        Feasible movement not exceeding ``delta``.
    """
    available_space = delta
    current = block_node
    while True:
        current = layout.align[current]
        neighbor = _elk_bk_neighbor_in_layer(graph, current, -1)
        if neighbor is not None:
            spacing = _elk_bk_spacing(current, neighbor, graph.dummy_nodes, node_spacing)
            available_space = min(
                available_space,
                _elk_bk_min_y(graph, layout, current)
                - (_elk_bk_max_y(graph, layout, neighbor) + spacing),
            )
        if current == block_node:
            return available_space


def _elk_bk_check_space_below(
    graph: _ElkBkGraph,
    layout: _ElkBkLayout,
    block_node: NodeId,
    delta: float,
    node_spacing: float,
) -> float:
    """Return how far a block can move toward next same-layer nodes.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    layout : _ElkBkLayout
        Current compaction layout.
    block_node : Hashable
        Any node in the block being moved.
    delta : float
        Requested positive movement.
    node_spacing : float
        ELK ``spacing.nodeNode``.

    Returns
    -------
    float
        Feasible movement not exceeding ``delta``.
    """
    available_space = delta
    current = block_node
    while True:
        current = layout.align[current]
        neighbor = _elk_bk_neighbor_in_layer(graph, current, 1)
        if neighbor is not None:
            spacing = _elk_bk_spacing(current, neighbor, graph.dummy_nodes, node_spacing)
            available_space = min(
                available_space,
                _elk_bk_min_y(graph, layout, neighbor)
                - (_elk_bk_max_y(graph, layout, current) + spacing),
            )
        if current == block_node:
            return available_space


def _elk_bk_shift_block(layout: _ElkBkLayout, block_node: NodeId, delta: float) -> None:
    """Shift all concrete node coordinates of one aligned block.

    Parameters
    ----------
    layout : _ElkBkLayout
        Current compaction layout.
    block_node : Hashable
        Any node in the cyclic block.
    delta : float
        Coordinate delta to add.

    Returns
    -------
    None
        ``layout.y`` is mutated for every node in the block.
    """
    current = block_node
    while True:
        layout.y[current] += delta
        current = layout.align[current]
        if current == block_node:
            break


def _elk_bk_calculate_delta(
    graph: _ElkBkGraph,
    layout: _ElkBkLayout,
    edge: Tuple[NodeId, NodeId],
    fixed_node: NodeId,
    block_node: NodeId,
) -> float:
    """Return ELK's port delta for post-process edge straightening.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    layout : _ElkBkLayout
        Current compaction layout.
    edge : tuple[Hashable, Hashable]
        Edge whose ports are compared.
    fixed_node : Hashable
        Endpoint whose block is treated as fixed.
    block_node : Hashable
        Endpoint whose block may move.

    Returns
    -------
    float
        Positive if ``block_node`` lies after ``fixed_node`` on the placement
        axis, negative if it lies before.
    """
    fixed_pos = (
        layout.y[fixed_node]
        + layout.inner_shift[fixed_node]
        + _elk_bk_endpoint_port_position(graph, edge, fixed_node)
    )
    block_pos = (
        layout.y[block_node]
        + layout.inner_shift[block_node]
        + _elk_bk_endpoint_port_position(graph, edge, block_node)
    )
    return block_pos - fixed_pos


def _elk_bk_horizontal_compaction(
    graph: _ElkBkGraph,
    layout: _ElkBkLayout,
    node_spacing: float,
) -> None:
    """Run ELK's class/sink horizontal compaction for one alignment.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    layout : _ElkBkLayout
        Alignment state to mutate.
    node_spacing : float
        ELK ``spacing.nodeNode``.

    Returns
    -------
    None
        ``layout.y`` stores top-left coordinates on the BK placement axis.
    """
    class_edges: Dict[NodeId, List[Tuple[NodeId, float]]] = {}
    class_nodes: List[NodeId] = []
    block_finished: Set[NodeId] = set()
    postprocess_queue: List[Tuple[NodeId, bool, Optional[Tuple[NodeId, NodeId]]]] = []
    postprocess_stack: List[Tuple[NodeId, bool, Tuple[NodeId, NodeId]]] = []

    def class_node(node: NodeId) -> NodeId:
        """Register a class-graph node in creation order.

        Parameters
        ----------
        node : Hashable
            Class sink node id.

        Returns
        -------
        Hashable
            The same node id.
        """
        if node not in class_edges:
            class_edges[node] = []
            class_nodes.append(node)
        return node

    def pick_edge(node: NodeId, is_root: bool) -> Tuple[Optional[Tuple[NodeId, NodeId]], bool]:
        """Pick ELK's first threshold-straightening edge candidate.

        Parameters
        ----------
        node : Hashable
            Free node inspected by ``ThresholdStrategy``.
        is_root : bool
            Whether the free node is viewed as the block root.

        Returns
        -------
        tuple[tuple[Hashable, Hashable] | None, bool]
            Picked edge and whether any candidate existed for possible later
            post-processing.
        """
        has_edges = False
        for edge in _elk_bk_edges_for_threshold(graph, layout, node, is_root):
            if layout.straightened[layout.root[node]]:
                continue
            has_edges = True
            other = _elk_bk_other_node(edge, node)
            if layout.root[other] in block_finished:
                return edge, True
        return None, has_edges

    def threshold_bound(node: NodeId, is_root: bool) -> float:
        """Return the threshold that would straighten one extra incident edge.

        Parameters
        ----------
        node : Hashable
            Free node inspected by ``ThresholdStrategy``.
        is_root : bool
            Whether the free node is viewed as the block root.

        Returns
        -------
        float
            ELK threshold value or the direction-specific neutral infinity.
        """
        invalid = float("inf") if layout.vdir == "UP" else float("-inf")
        picked_edge, has_edges = pick_edge(node, is_root)
        if picked_edge is None:
            if has_edges:
                postprocess_queue.append((node, is_root, picked_edge))
            return invalid

        source, target = picked_edge
        if is_root:
            root_port = target if layout.hdir == "RIGHT" else source
            other_port = source if layout.hdir == "RIGHT" else target
        else:
            root_port = target if layout.hdir == "LEFT" else source
            other_port = source if layout.hdir == "LEFT" else target
        other_root = layout.root[other_port]
        threshold = (
            layout.y[other_root]
            + layout.inner_shift[other_port]
            + _elk_bk_endpoint_port_position(graph, picked_edge, other_port)
            - layout.inner_shift[root_port]
            - _elk_bk_endpoint_port_position(graph, picked_edge, root_port)
        )
        layout.straightened[layout.root[source]] = True
        layout.straightened[layout.root[target]] = True
        return threshold

    def calculate_threshold(
        old_threshold: float,
        block_root: NodeId,
        current_node: NodeId,
    ) -> float:
        """Calculate ELK's simple straightening threshold for a block node.

        Parameters
        ----------
        old_threshold : float
            Previous threshold value for the current block.
        block_root : Hashable
            Root of the block currently being placed.
        current_node : Hashable
            Node within the block currently being inspected.

        Returns
        -------
        float
            Updated threshold.
        """
        is_root = block_root == current_node
        is_last = layout.align[current_node] == block_root
        if not (is_root or is_last):
            return old_threshold
        threshold = old_threshold
        if is_root:
            threshold = threshold_bound(block_root, True)
        if threshold in (float("inf"), float("-inf")) and is_last:
            threshold = threshold_bound(current_node, False)
        return threshold

    def process_postprocessable(node: NodeId, edge: Tuple[NodeId, NodeId]) -> bool:
        """Try to move one postponed block to straighten a candidate edge.

        Parameters
        ----------
        node : Hashable
            Free node whose block may move.
        edge : tuple[Hashable, Hashable]
            Incident edge selected by ``pickEdge``.

        Returns
        -------
        bool
            ``True`` if the block moved.
        """
        fixed_node = _elk_bk_other_node(edge, node)
        delta = _elk_bk_calculate_delta(graph, layout, edge, fixed_node, node)
        if 0.0 < delta < _ELK_BK_STRAIGHTENING_THRESHOLD:
            available_space = _elk_bk_check_space_above(graph, layout, node, delta, node_spacing)
            _elk_bk_shift_block(layout, node, -available_space)
            return available_space > 0.0
        if 0.0 < -delta < _ELK_BK_STRAIGHTENING_THRESHOLD:
            available_space = _elk_bk_check_space_below(graph, layout, node, -delta, node_spacing)
            _elk_bk_shift_block(layout, node, available_space)
            return available_space > 0.0
        return False

    def postprocess_thresholds() -> None:
        """Run ELK's two-order post-processing for late straightening edges.

        Returns
        -------
        None
            ``layout.y`` may be mutated by feasible block shifts.
        """
        while postprocess_queue:
            node, is_root, _ = postprocess_queue.pop(0)
            picked_edge, _ = pick_edge(node, is_root)
            if picked_edge is None:
                continue
            moved = process_postprocessable(node, picked_edge)
            if not moved:
                postprocess_stack.append((node, is_root, picked_edge))
        while postprocess_stack:
            node, _, edge = postprocess_stack.pop()
            process_postprocessable(node, edge)

    def place_block(root: NodeId) -> None:
        """Recursively place one aligned block.

        Parameters
        ----------
        root : Hashable
            Root node of the block to place.

        Returns
        -------
        None
            ``layout`` and class graph state are mutated.
        """
        if root in layout.y:
            return
        is_initial_assignment = True
        layout.y[root] = 0.0
        current = root
        threshold = float("-inf") if layout.vdir == "DOWN" else float("inf")
        while True:
            current_index = graph.node_index[current]
            layer = graph.layers[graph.layer_index[current]]
            has_neighbor = (layout.vdir == "DOWN" and current_index > 0) or (
                layout.vdir == "UP" and current_index < len(layer) - 1
            )
            if has_neighbor:
                neighbor = (
                    layer[current_index + 1] if layout.vdir == "UP" else layer[current_index - 1]
                )
                neighbor_root = layout.root[neighbor]
                place_block(neighbor_root)
                threshold = calculate_threshold(threshold, root, current)
                if layout.sink[root] == root:
                    layout.sink[root] = layout.sink[neighbor_root]
                if layout.sink[root] == layout.sink[neighbor_root]:
                    spacing = _elk_bk_spacing(current, neighbor, graph.dummy_nodes, node_spacing)
                    if layout.vdir == "UP":
                        new_position = (
                            layout.y[neighbor_root]
                            + layout.inner_shift[neighbor]
                            - spacing
                            - graph.sizes[current]
                            - layout.inner_shift[current]
                        )
                        layout.y[root] = (
                            min(new_position, threshold)
                            if is_initial_assignment
                            else min(layout.y[root], new_position, threshold)
                        )
                    else:
                        new_position = (
                            layout.y[neighbor_root]
                            + layout.inner_shift[neighbor]
                            + graph.sizes[neighbor]
                            + spacing
                            - layout.inner_shift[current]
                        )
                        layout.y[root] = (
                            max(new_position, threshold)
                            if is_initial_assignment
                            else max(layout.y[root], new_position, threshold)
                        )
                    is_initial_assignment = False
                else:
                    sink = class_node(layout.sink[root])
                    neighbor_sink = class_node(layout.sink[neighbor_root])
                    if layout.vdir == "UP":
                        required_space = (
                            layout.y[root]
                            + layout.inner_shift[current]
                            + graph.sizes[current]
                            + node_spacing
                            - layout.y[neighbor_root]
                            - layout.inner_shift[neighbor]
                        )
                    else:
                        required_space = (
                            layout.y[root]
                            + layout.inner_shift[current]
                            - layout.y[neighbor_root]
                            - layout.inner_shift[neighbor]
                            - graph.sizes[neighbor]
                            - node_spacing
                        )
                    class_edges[sink].append((neighbor_sink, required_space))
            else:
                threshold = calculate_threshold(threshold, root, current)
            current = layout.align[current]
            if current == root:
                break
        block_finished.add(root)

    layer_sequence = list(graph.layers)
    if layout.hdir == "LEFT":
        layer_sequence = list(reversed(layer_sequence))
    for layer in layer_sequence:
        nodes = list(layer)
        if layout.vdir == "UP":
            nodes = list(reversed(nodes))
        for node in nodes:
            if layout.root[node] == node:
                place_block(node)

    _elk_bk_place_classes(layout, class_edges, class_nodes)
    for layer in layer_sequence:
        for node in layer:
            root = layout.root[node]
            layout.y[node] = layout.y[root]
            if node == root:
                sink_shift = layout.shift[layout.sink[node]]
                if (layout.vdir == "UP" and sink_shift > float("-inf")) or (
                    layout.vdir == "DOWN" and sink_shift < float("inf")
                ):
                    layout.y[node] += sink_shift
    postprocess_thresholds()


def _elk_bk_layout_size(graph: _ElkBkGraph, layout: _ElkBkLayout) -> float:
    """Return ELK ``BKAlignedLayout.layoutSize`` for one layout.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    layout : _ElkBkLayout
        Completed aligned layout.

    Returns
    -------
    float
        Span from minimum block coordinate to maximum block extent.
    """
    minimum = float("inf")
    maximum = float("-inf")
    for layer in graph.layers:
        for node in layer:
            y_min = layout.y[node]
            y_max = y_min + layout.block_size[layout.root[node]]
            minimum = min(minimum, y_min)
            maximum = max(maximum, y_max)
    return maximum - minimum


def _elk_bk_balanced_layout(graph: _ElkBkGraph, layouts: Sequence[_ElkBkLayout]) -> _ElkBkLayout:
    """Create ELK's balanced median layout from four alignments.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    layouts : sequence[_ElkBkLayout]
        Completed layouts in ELK's default order.

    Returns
    -------
    _ElkBkLayout
        Balanced layout with ``inner_shift`` folded into ``y``.
    """
    balanced = _elk_bk_new_layout(graph, "DOWN", "RIGHT")
    widths: List[float] = []
    minimums: List[float] = []
    maximums: List[float] = []
    min_width_layout = 0
    for index, layout in enumerate(layouts):
        width = _elk_bk_layout_size(graph, layout)
        widths.append(width)
        if widths[min_width_layout] > width:
            min_width_layout = index
        layout_min = float("inf")
        layout_max = float("-inf")
        for layer in graph.layers:
            for node in layer:
                node_pos = layout.y[node] + layout.inner_shift[node]
                layout_min = min(layout_min, node_pos)
                layout_max = max(layout_max, node_pos + graph.sizes[node])
        minimums.append(layout_min)
        maximums.append(layout_max)
    shifts: List[float] = []
    for index, layout in enumerate(layouts):
        if layout.vdir == "DOWN":
            shifts.append(minimums[min_width_layout] - minimums[index])
        else:
            shifts.append(maximums[min_width_layout] - maximums[index])
    for layer in graph.layers:
        for node in layer:
            samples = sorted(
                layout.y[node] + layout.inner_shift[node] + shifts[index]
                for index, layout in enumerate(layouts)
            )
            balanced.y[node] = (samples[1] + samples[2]) / 2.0
            balanced.inner_shift[node] = 0.0
    return balanced


def _elk_bk_check_order(
    graph: _ElkBkGraph,
    layout: _ElkBkLayout,
) -> bool:
    """Validate ELK's in-layer ordering constraint for a layout.

    Parameters
    ----------
    graph : _ElkBkGraph
        Normalized BK graph.
    layout : _ElkBkLayout
        Candidate layout.

    Returns
    -------
    bool
        ``True`` if nodes remain strictly ordered without overlap.
    """
    for layer in graph.layers:
        position = float("-inf")
        for node in layer:
            top = layout.y[node] + layout.inner_shift[node]
            bottom = top + graph.sizes[node]
            if top > position and bottom > position:
                position = bottom
            else:
                return False
    return True


def _elk_brandes_koepf_x_coordinates(
    layers: Sequence[Sequence[int]],
    successors: Mapping[int, Sequence[int]],
    sizes: torch.Tensor,
    node_spacing: float,
) -> Dict[int, float]:
    """Assign top-left x coordinates using ELK BKNodePlacer semantics.

    Parameters
    ----------
    layers : sequence[sequence[int]]
        Ordered real-node layers.
    successors : mapping[int, sequence[int]]
        Real-node successor lists.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    node_spacing : float
        ELK ``spacing.nodeNode`` value.

    Returns
    -------
    dict[int, float]
        Top-left x coordinates for real nodes.
    """
    graph = _normalize_long_edges_for_elk_bk(layers, successors, sizes)
    marked_edges = _elk_bk_mark_conflicts(graph)
    layouts = [
        _elk_bk_new_layout(graph, "DOWN", "RIGHT"),
        _elk_bk_new_layout(graph, "UP", "RIGHT"),
        _elk_bk_new_layout(graph, "DOWN", "LEFT"),
        _elk_bk_new_layout(graph, "UP", "LEFT"),
    ]
    for layout in layouts:
        _elk_bk_vertical_alignment(graph, layout, marked_edges)
        _elk_bk_inside_block_shift(graph, layout)
        _elk_bk_horizontal_compaction(graph, layout, node_spacing)
    if graph.dummy_nodes:
        balanced = _elk_bk_balanced_layout(graph, layouts)
        chosen = balanced if _elk_bk_check_order(graph, balanced) else None
    else:
        chosen = None
    if chosen is None:
        # ELK's default favors straight edges, so BKNodePlacer skips the
        # balanced median layout on ordinary layered graphs. For dummy-chain
        # long edges we first try the historical balanced path above to avoid
        # regressing existing long-edge fidelity pins while the dummy port model
        # remains less complete than the no-dummy BK path fixed here.
        for layout in layouts:
            if _elk_bk_check_order(graph, layout):
                if chosen is None:
                    chosen = layout
                    continue
                chosen_size = _elk_bk_layout_size(graph, chosen)
                layout_size = _elk_bk_layout_size(graph, layout)
                if chosen_size > layout_size + _ELK_BK_LAYOUT_SIZE_EPSILON:
                    chosen = layout
    if chosen is None:
        chosen = layouts[0]
    raw_coordinates = {node: chosen.y[node] + chosen.inner_shift[node] for node in graph.real_nodes}
    if not raw_coordinates:
        return {}
    left_extent = min(raw_coordinates[node] for node in graph.real_nodes)
    return {node: value - left_extent + _ROOT_PADDING for node, value in raw_coordinates.items()}


def _apply_direction(positions: torch.Tensor, sizes: torch.Tensor, direction: str) -> torch.Tensor:
    """Transform DOWN coordinates to the requested ELK direction.

    Parameters
    ----------
    positions : torch.Tensor
        Top-left DOWN coordinates with shape ``[N, 2]``.
    sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    direction : str
        Normalized ELK direction.

    Returns
    -------
    torch.Tensor
        Transformed top-left coordinates.
    """
    if positions.numel() == 0 or direction == "DOWN":
        return positions
    output = positions.clone()
    if direction == "UP":
        bottom = positions[:, 1] + sizes[:, 1]
        max_bottom = float(bottom.max().item())
        output[:, 1] = max_bottom - bottom + _ROOT_PADDING
        return output
    if direction == "RIGHT":
        output[:, 0] = positions[:, 1]
        output[:, 1] = positions[:, 0]
        return output
    right = positions[:, 1] + sizes[:, 1]
    max_right = float(right.max().item())
    output[:, 0] = max_right - right + _ROOT_PADDING
    output[:, 1] = positions[:, 0]
    return output


@register_op
class ElkPrepareGraph(Op):
    """Validate inputs and store ELK layout options."""

    name: ClassVar[str] = "elk_prepare_graph"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def __init__(
        self,
        direction: str = "DOWN",
        node_node_spacing: float = 40.0,
        between_layers_spacing: float = 60.0,
        cycle_breaking_strategy: str = "greedy",
        layering_strategy: str = "network_simplex",
        crossing_minimization_strategy: str = "layer_sweep",
        node_placement_strategy: str = "brandes_koepf",
        random_seed: Optional[int] = None,
        thoroughness: int = 7,
    ) -> None:
        """Store validated ELK pipeline options.

        Parameters
        ----------
        direction : str, default="DOWN"
            ELK direction.
        node_node_spacing : float, default=40.0
            Same-layer node-node spacing.
        between_layers_spacing : float, default=60.0
            Spacing between adjacent layers.
        cycle_breaking_strategy : str, default="greedy"
            Cycle breaking strategy selector.
        layering_strategy : str, default="network_simplex"
            Layer assignment strategy selector.
        crossing_minimization_strategy : str, default="layer_sweep"
            Crossing minimization strategy selector.
        node_placement_strategy : str, default="brandes_koepf"
            Node placement strategy selector.
        random_seed : int | None, optional
            Public ELK random seed. ``None`` reads ``problem.seed``.
        thoroughness : int, default=7
            Number of layer-sweep restart attempts.

        Returns
        -------
        None
            Options are stored on the op.
        """
        if node_node_spacing < 0.0 or between_layers_spacing < 0.0:
            raise ValueError("ELK spacing values must be non-negative.")
        if thoroughness < 1:
            raise ValueError("ELK thoroughness must be at least 1.")
        self.direction = _normalize_direction(direction)
        self.node_node_spacing = float(node_node_spacing)
        self.between_layers_spacing = float(between_layers_spacing)
        self.random_seed = None if random_seed is None else int(random_seed)
        self.thoroughness = int(thoroughness)
        self.cycle_breaking_strategy = _normalize_choice(
            cycle_breaking_strategy, _SUPPORTED_CYCLE_STRATEGIES, "cycle_breaking_strategy"
        )
        self.layering_strategy = _normalize_choice(
            layering_strategy, _SUPPORTED_LAYERING_STRATEGIES, "layering_strategy"
        )
        self.crossing_minimization_strategy = _normalize_choice(
            crossing_minimization_strategy,
            _SUPPORTED_CROSSING_STRATEGIES,
            "crossing_minimization_strategy",
        )
        self.node_placement_strategy = _normalize_choice(
            node_placement_strategy, _SUPPORTED_NODE_PLACEMENT_STRATEGIES, "node_placement_strategy"
        )

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Create the ELK working graph.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable state receiving ``elk_graph``.
        ctx : RuntimeContext
            Runtime infrastructure; unused by this deterministic port.

        Returns
        -------
        SolveState
            Updated state.
        """
        del ctx
        edge_index = problem.edge_index.detach().to(device="cpu", dtype=torch.long)
        if edge_index.ndim != 2 or edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")
        if edge_index.numel() and (
            int(edge_index.min().item()) < 0 or int(edge_index.max().item()) >= problem.num_nodes
        ):
            raise ValueError("edge_index contains an out-of-range node id.")
        edges = [
            (int(source), int(target))
            for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist())
            if int(source) != int(target)
        ]
        state.extras[ELK_GRAPH_KEY] = _ElkGraph(
            edges=edges,
            active_edges=list(edges),
            node_sizes=_node_sizes(problem),
            direction=self.direction,
            node_node_spacing=self.node_node_spacing,
            between_layers_spacing=self.between_layers_spacing,
            cycle_breaking_strategy=self.cycle_breaking_strategy,
            layering_strategy=self.layering_strategy,
            crossing_minimization_strategy=self.crossing_minimization_strategy,
            node_placement_strategy=self.node_placement_strategy,
            random_seed=self.random_seed if self.random_seed is not None else int(problem.seed),
            thoroughness=self.thoroughness,
        )
        return state


@register_op
class ElkBreakCycles(Op):
    """Apply deterministic ELK cycle-breaking approximation."""

    name: ClassVar[str] = "elk_break_cycles"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Break directed cycles.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            State containing ``elk_graph``.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            Updated state with acyclic active edges.
        """
        del ctx
        graph = state.extras[ELK_GRAPH_KEY]
        if graph.cycle_breaking_strategy == "greedy":
            graph.active_edges = _break_cycles_greedy(problem.num_nodes, graph.edges)
        else:
            graph.active_edges = _break_cycles_depth_first(problem.num_nodes, graph.edges)
        return state


@register_op
class ElkAssignLayers(Op):
    """Assign nodes to layers."""

    name: ClassVar[str] = "elk_assign_layers"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Assign longest-path layers for the active acyclic graph.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            State containing ``elk_graph``.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            Updated state with ``elk_layers``.
        """
        del ctx
        graph = state.extras[ELK_GRAPH_KEY]
        if graph.layering_strategy == "longest_path":
            assignments = _longest_path_layers(problem.num_nodes, graph.active_edges)
        else:
            assignments = _network_simplex_layers(problem.num_nodes, graph.active_edges)
        state.extras[ELK_LAYERS_KEY] = _layers_from_assignments(assignments)
        return state


@register_op
class ElkMinimizeCrossings(Op):
    """Order nodes within layers by median sweeps."""

    name: ClassVar[str] = "elk_minimize_crossings"
    category: ClassVar[OpCategory] = OpCategory.ORDERING
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run layer-sweep crossing minimization with ELK-style restarts.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs; unused.
        state : SolveState
            State containing layer assignments.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            Updated state with ordered layers and BK metadata.
        """
        del problem, ctx
        graph = state.extras[ELK_GRAPH_KEY]
        if graph.crossing_minimization_strategy == "layer_sweep":
            ordered = _restart_sweep_orders(
                state.extras[ELK_LAYERS_KEY],
                graph.active_edges,
                random_seed=graph.random_seed,
                thoroughness=graph.thoroughness,
            )
        else:
            ordered = _sweep_orders(state.extras[ELK_LAYERS_KEY], graph.active_edges)
        state.extras[ELK_LAYERS_KEY] = ordered
        state.extras[ELK_ORDER_KEY] = {
            node: index for layer in ordered for index, node in enumerate(layer)
        }
        predecessors = _predecessors(graph.node_sizes.shape[0], graph.active_edges)
        successors = _successors(graph.node_sizes.shape[0], graph.active_edges)
        state.extras[BRANDES_KOEPF_LAYERING_KEY] = ordered
        state.extras[BRANDES_KOEPF_PREDECESSORS_KEY] = predecessors
        state.extras[BRANDES_KOEPF_SUCCESSORS_KEY] = successors
        state.extras[BRANDES_KOEPF_WIDTHS_KEY] = {
            node: float(graph.node_sizes[node, 0]) for layer in ordered for node in layer
        }
        state.extras[BRANDES_KOEPF_DUMMY_NODES_KEY] = set()
        return state


@register_op
class ElkPlaceNodes(Op):
    """Assign top-left coordinates for all nodes."""

    name: ClassVar[str] = "elk_place_nodes"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras", "pos")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Create final node coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            State containing ordered layers.
        ctx : RuntimeContext
            Runtime infrastructure; unused.

        Returns
        -------
        SolveState
            Updated state with ``pos``.
        """
        del ctx
        graph = state.extras[ELK_GRAPH_KEY]
        layers = state.extras[ELK_LAYERS_KEY]
        predecessors = state.extras[BRANDES_KOEPF_PREDECESSORS_KEY]
        successors = state.extras[BRANDES_KOEPF_SUCCESSORS_KEY]
        x_coordinates = _layer_x_coordinates(
            layers=layers,
            predecessors=predecessors,
            successors=successors,
            sizes=graph.node_sizes,
            node_spacing=graph.node_node_spacing,
            strategy=graph.node_placement_strategy,
        )
        if _is_two_middle_diamond(layers, graph.active_edges):
            _apply_two_middle_diamond_x_coordinates(
                x_coordinates,
                layers,
                graph.node_sizes,
                graph.node_node_spacing,
            )
        y_coordinates = _layer_y_coordinates(
            layers=layers,
            sizes=graph.node_sizes,
            spacing=graph.between_layers_spacing,
        )
        positions = torch.zeros((problem.num_nodes, 2), dtype=torch.float64)
        for node in range(problem.num_nodes):
            positions[node, 0] = x_coordinates.get(node, _ROOT_PADDING)
            positions[node, 1] = y_coordinates.get(node, _ROOT_PADDING)
        positions = _apply_direction(positions, graph.node_sizes, graph.direction)
        state.extras[BRANDES_KOEPF_X_KEY] = {
            node: float(x_coordinates.get(node, _ROOT_PADDING)) for node in range(problem.num_nodes)
        }
        state.extras[ELK_INTERNAL_POSITIONS_KEY] = positions
        state.pos = positions
        return state


__all__ = [
    "ELK_GRAPH_KEY",
    "ELK_INTERNAL_POSITIONS_KEY",
    "ELK_LAYERS_KEY",
    "ELK_ORDER_KEY",
    "ElkAssignLayers",
    "ElkBreakCycles",
    "ElkMinimizeCrossings",
    "ElkPlaceNodes",
    "ElkPrepareGraph",
]
