"""Shared compound machinery for Cytoscape CoSE-family layouts."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch

_CYTOSCAPE_LCG_MULTIPLIER = 1664525
_CYTOSCAPE_LCG_INCREMENT = 1013904223
_CYTOSCAPE_LCG_MODULUS = 4294967296.0
_DEFAULT_EDGE_LENGTH = 50.0
_DEFAULT_GRAPH_MARGIN = 15.0
_EMPTY_COMPOUND_NODE_SIZE = 40.0
_INITIAL_WORLD_BOUNDARY = 1000.0
_WORLD_CENTER_X = 1200.0
_WORLD_CENTER_Y = 900.0
_SIMPLE_NODE_SIZE = 40.0
_MIN_REPULSION_DIST = _DEFAULT_EDGE_LENGTH / 10.0
_MAX_NODE_DISPLACEMENT = 300.0
_DISPLACEMENT_THRESHOLD_PER_NODE = (3.0 * _DEFAULT_EDGE_LENGTH) / 100.0
_CONVERGENCE_CHECK_PERIOD = 100
_GRID_CALCULATION_CHECK_PERIOD = 10
_DEFAULT_COOLING_FACTOR_INCREMENTAL = 0.3
_DEFAULT_COMPONENT_SEPARATION = 60.0
_DEFAULT_RADIAL_SEPARATION = _DEFAULT_EDGE_LENGTH
_AREA_EPSILON = 1.0e-12


@dataclass(eq=False)
class _Rect:
    """Mutable rectangle using the layout-base top-left coordinate convention."""

    x: float = 0.0
    y: float = 0.0
    width: float = 0.0
    height: float = 0.0


@dataclass(eq=False)
class _Node:
    """Minimal layout-base ``LNode``/``CoSENode`` equivalent."""

    id: str
    rect: _Rect
    leaf_index: Optional[int] = None
    owner: Optional["_Graph"] = None
    child: Optional["_Graph"] = None
    padding_left: float = 0.0
    padding_top: float = 0.0
    padding_right: float = 0.0
    padding_bottom: float = 0.0
    edges: List["_Edge"] = field(default_factory=list)
    estimated_size: float = -math.inf
    inclusion_tree_depth: int = 0
    no_of_children: int = 1
    spring_force_x: float = 0.0
    spring_force_y: float = 0.0
    repulsion_force_x: float = 0.0
    repulsion_force_y: float = 0.0
    gravitation_force_x: float = 0.0
    gravitation_force_y: float = 0.0
    displacement_x: float = 0.0
    displacement_y: float = 0.0
    surrounding: List["_Node"] = field(default_factory=list)
    start_x: int = 0
    finish_x: int = 0
    start_y: int = 0
    finish_y: int = 0


@dataclass(eq=False)
class _Edge:
    """Minimal layout-base ``LEdge``/``CoSEEdge`` equivalent."""

    source: _Node
    target: _Node
    is_inter_graph: bool
    lca: Optional["_Graph"] = None
    source_in_lca: Optional[_Node] = None
    target_in_lca: Optional[_Node] = None
    ideal_length: float = _DEFAULT_EDGE_LENGTH
    length_x: float = 0.0
    length_y: float = 0.0
    length: float = 0.0
    is_overlapping_source_and_target: bool = False


@dataclass(eq=False)
class _Graph:
    """Minimal layout-base ``LGraph`` equivalent."""

    parent: Optional[_Node]
    nodes: List[_Node] = field(default_factory=list)
    edges: List[_Edge] = field(default_factory=list)
    estimated_size: float = -math.inf
    left: float = 0.0
    right: float = 0.0
    top: float = 0.0
    bottom: float = 0.0
    is_connected: bool = False


@dataclass
class _TileOrganization:
    """Row organization returned by the cose-base tiling strip packer."""

    rows: List[List[_Node]]
    row_width: List[float]
    row_height: List[float]
    width: float
    height: float
    vertical_padding: float
    horizontal_padding: float


@dataclass
class CoSECompoundOptions:
    """Options shared by CoSE-Bilkent and future fCoSE compound phases.

    Parameters
    ----------
    steps : int, default=2500
        Maximum spring-embedder iterations before tree regrowth starts.
    seed : int, default=42
        Seed for Cytoscape's adapter-level LCG.
    node_repulsion : float, default=4500.0
        Repulsion constant used by ``FDLayout.calcRepulsionForce``.
    ideal_edge_length : float, default=50.0
        Base ideal edge length before nesting-factor expansion.
    edge_elasticity : float, default=0.45
        Spring constant used by ``FDLayout.calcSpringForce``.
    nesting_factor : float, default=0.1
        Per-level ideal edge length factor for inter-graph edges.
    gravity : float, default=0.25
        Root gravity constant.
    gravity_range : float, default=3.8
        Root gravity range multiplier.
    gravity_compound : float, default=1.0
        Additional multiplier for non-root graph gravity.
    gravity_range_compound : float, default=1.5
        Non-root graph gravity range multiplier.
    tile : bool, default=True
        Whether to apply cose-base tiling pre/post processing.
    tiling_padding_vertical : float, default=10.0
        Vertical padding used by ``tileNodes``.
    tiling_padding_horizontal : float, default=10.0
        Horizontal padding used by ``tileNodes``.
    quality : str, default="default"
        Cytoscape quality tier: ``"draft"``, ``"default"``, or ``"proof"``.
    randomize : bool, default=True
        Whether to scatter initial positions with layout-base ``RandomSeed``.
    version : str, default="1.0.3"
        cose-base source family. The interface is versioned for fCoSE reuse.
    """

    steps: int = 2500
    seed: int = 42
    node_repulsion: float = 4500.0
    ideal_edge_length: float = 50.0
    edge_elasticity: float = 0.45
    nesting_factor: float = 0.1
    gravity: float = 0.25
    gravity_range: float = 3.8
    gravity_compound: float = 1.0
    gravity_range_compound: float = 1.5
    tile: bool = True
    tiling_padding_vertical: float = 10.0
    tiling_padding_horizontal: float = 10.0
    quality: str = "default"
    randomize: bool = True
    version: str = "1.0.3"


@dataclass
class CoSECompoundState:
    """Mutable compound-CoSE state exposed for structural fidelity checks.

    Parameters
    ----------
    root : _Graph
        Root inclusion graph.
    graphs : list[_Graph]
        Graph-manager graph list in layout-base encounter order.
    nodes_by_leaf : dict[int, _Node]
        Mapping from original Dagua node index to layout node.
    nodes_by_id : dict[str, _Node]
        Mapping from cluster or leaf ids to layout nodes.
    edges : list[_Edge]
        All unique layout edges.
    options : CoSECompoundOptions
        Runtime options.
    random_state : int
        Adapter LCG state used for tree regrowth tie breaks.
    sine_seed : int
        layout-base ``RandomSeed.seed`` state.
    """

    root: _Graph
    graphs: List[_Graph]
    nodes_by_leaf: Dict[int, _Node]
    nodes_by_id: Dict[str, _Node]
    edges: List[_Edge]
    options: CoSECompoundOptions
    random_state: int
    sine_seed: int = 1
    nodes_with_gravity: List[_Node] = field(default_factory=list)
    pruned_nodes_all: List[List[Tuple[_Node, _Edge, _Graph]]] = field(default_factory=list)
    tiled_member_pack: Dict[str, _TileOrganization] = field(default_factory=dict)
    tiled_zero_degree_pack: Dict[str, _TileOrganization] = field(default_factory=dict)
    member_groups: Dict[str, List[_Node]] = field(default_factory=dict)
    id_to_dummy_node: Dict[str, _Node] = field(default_factory=dict)
    compound_order: List[_Node] = field(default_factory=list)
    to_be_tiled: Dict[str, bool] = field(default_factory=dict)
    total_iterations: int = 0
    total_displacement: float = 0.0
    old_total_displacement: float = 0.0
    cooling_factor: float = 1.0
    initial_cooling_factor: float = 1.0
    max_iterations: int = 2500
    total_displacement_threshold: float = 0.0
    repulsion_range: float = 100.0
    cooling_cycle: int = 0
    max_cooling_cycle: float = 25.0
    final_temperature: float = 0.04
    cooling_adjuster: float = 1.0
    is_tree_growing: bool = False
    is_growth_finished: bool = False
    grow_tree_iterations: int = 0
    after_growth_iterations: int = 0
    grid: List[List[List[_Node]]] = field(default_factory=list)


def _center_x(node: _Node) -> float:
    """Return node center x-coordinate.

    Parameters
    ----------
    node : _Node
        Layout node.

    Returns
    -------
    float
        Center x-coordinate.
    """
    return node.rect.x + node.rect.width / 2.0


def _center_y(node: _Node) -> float:
    """Return node center y-coordinate.

    Parameters
    ----------
    node : _Node
        Layout node.

    Returns
    -------
    float
        Center y-coordinate.
    """
    return node.rect.y + node.rect.height / 2.0


def _set_center(node: _Node, x: float, y: float) -> None:
    """Set a node's center coordinates in-place.

    Parameters
    ----------
    node : _Node
        Layout node to move.
    x : float
        New center x-coordinate.
    y : float
        New center y-coordinate.

    Returns
    -------
    None
        Mutates ``node``.
    """
    node.rect.x = float(x) - node.rect.width / 2.0
    node.rect.y = float(y) - node.rect.height / 2.0


def _move_by(node: _Node, dx: float, dy: float) -> None:
    """Move a node rectangle by an offset.

    Parameters
    ----------
    node : _Node
        Layout node to move.
    dx : float
        X displacement.
    dy : float
        Y displacement.

    Returns
    -------
    None
        Mutates ``node``.
    """
    node.rect.x += dx
    node.rect.y += dy


def _sign(value: float) -> float:
    """Return the JavaScript-compatible sign used by layout-base.

    Parameters
    ----------
    value : float
        Numeric value.

    Returns
    -------
    float
        ``1`` for positive values, ``-1`` for negative values, otherwise ``0``.
    """
    if value > 0.0:
        return 1.0
    if value < 0.0:
        return -1.0
    return 0.0


def _next_cytoscape_random(raw_state: int) -> Tuple[int, float]:
    """Advance Cytoscape's adapter LCG by one draw.

    Parameters
    ----------
    raw_state : int
        Current unsigned 32-bit state.

    Returns
    -------
    tuple[int, float]
        Updated state and random value in ``[0, 1)``.
    """
    next_state = (
        _CYTOSCAPE_LCG_MULTIPLIER * int(raw_state) + _CYTOSCAPE_LCG_INCREMENT
    ) & 0xFFFFFFFF
    return next_state, next_state / _CYTOSCAPE_LCG_MODULUS


def _next_sine_random(seed: int) -> Tuple[int, float]:
    """Advance layout-base ``RandomSeed.nextDouble``.

    Parameters
    ----------
    seed : int
        Current module-level ``RandomSeed.seed``.

    Returns
    -------
    tuple[int, float]
        Updated seed and value in ``[0, 1)``.
    """
    value = math.sin(seed) * 10000.0
    return seed + 1, value - math.floor(value)


def _normalize_sizes(
    node_sizes: Optional[torch.Tensor],
    num_nodes: int,
) -> List[Tuple[float, float]]:
    """Return Cytoscape node dimensions as Python floats.

    Parameters
    ----------
    node_sizes : torch.Tensor | None
        Optional tensor with shape ``[N, 2]`` or compatible.
    num_nodes : int
        Number of leaf nodes.

    Returns
    -------
    list[tuple[float, float]]
        Width and height per node.
    """
    if node_sizes is None:
        return [(_SIMPLE_NODE_SIZE, _SIMPLE_NODE_SIZE) for _ in range(num_nodes)]
    sizes = node_sizes.detach().cpu().to(dtype=torch.float64)
    if sizes.ndim == 1:
        sizes = sizes[:, None].repeat(1, 2)
    if sizes.shape[1] == 1:
        sizes = sizes.repeat(1, 2)
    result: List[Tuple[float, float]] = []
    for index in range(num_nodes):
        width = float(sizes[index, 0].item())
        height = float(sizes[index, 1].item())
        result.append((width, height))
    return result


def _cluster_members(
    clusters: Optional[Mapping[str, Any]],
    num_nodes: int,
) -> Dict[str, List[int]]:
    """Normalize cluster memberships to sorted valid leaf indices.

    Parameters
    ----------
    clusters : mapping[str, Any] | None
        Cluster membership mapping from ``LayoutProblem``.
    num_nodes : int
        Number of leaf nodes.

    Returns
    -------
    dict[str, list[int]]
        Valid members keyed by cluster id.
    """
    normalized: Dict[str, List[int]] = {}
    if not clusters:
        return normalized
    for name, raw_members in clusters.items():
        if isinstance(raw_members, Mapping):
            continue
        members = sorted({int(member) for member in raw_members if 0 <= int(member) < num_nodes})
        if members:
            normalized[str(name)] = members
    return normalized


def _child_maps(
    members: Mapping[str, Sequence[int]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    num_nodes: int,
) -> Tuple[Dict[Optional[str], List[str]], Dict[Optional[str], List[int]]]:
    """Build direct child-cluster and direct leaf maps.

    Parameters
    ----------
    members : mapping[str, sequence[int]]
        Flattened leaf membership per cluster.
    cluster_parents : mapping[str, str | None] | None
        Cluster parent mapping.
    num_nodes : int
        Number of leaf nodes.

    Returns
    -------
    tuple[dict[str | None, list[str]], dict[str | None, list[int]]]
        Direct cluster children and direct leaf children keyed by parent cluster.
    """
    names = set(members)
    parents = cluster_parents or {}
    child_clusters: Dict[Optional[str], List[str]] = {None: []}
    for name in sorted(names):
        parent = parents.get(name)
        parent_key = str(parent) if parent in names else None
        child_clusters.setdefault(parent_key, []).append(name)
        child_clusters.setdefault(name, [])

    direct_nodes: Dict[Optional[str], List[int]] = {}
    assigned_to_child: Dict[Optional[str], set[int]] = {}
    for parent, children in child_clusters.items():
        child_members: set[int] = set()
        for child in children:
            child_members.update(int(node) for node in members[child])
        assigned_to_child[parent] = child_members

    root_cluster_members = (
        set().union(*(set(value) for value in members.values())) if members else set()
    )
    direct_nodes[None] = [node for node in range(num_nodes) if node not in root_cluster_members]
    for name, flattened in members.items():
        child_members = assigned_to_child.get(name, set())
        direct_nodes[name] = [node for node in flattened if node not in child_members]
    return child_clusters, direct_nodes


def _append_graph(state: CoSECompoundState, parent_node: _Node) -> _Graph:
    """Create and attach a child graph.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable layout state.
    parent_node : _Node
        Compound node that owns the child graph.

    Returns
    -------
    _Graph
        Newly attached graph.
    """
    graph = _Graph(parent=parent_node)
    parent_node.child = graph
    state.graphs.append(graph)
    return graph


def _add_node(graph: _Graph, node: _Node) -> _Node:
    """Add a node to a graph in layout-base encounter order.

    Parameters
    ----------
    graph : _Graph
        Owner graph.
    node : _Node
        Node to add.

    Returns
    -------
    _Node
        Added node.
    """
    node.owner = graph
    graph.nodes.append(node)
    return node


def _make_leaf_node(node_index: int, size: Tuple[float, float]) -> _Node:
    """Create one leaf layout node.

    Parameters
    ----------
    node_index : int
        Original Dagua node index.
    size : tuple[float, float]
        Node width and height.

    Returns
    -------
    _Node
        New leaf node.
    """
    width, height = size
    return _Node(
        id=str(node_index),
        rect=_Rect(-width / 2.0, -height / 2.0, width, height),
        leaf_index=node_index,
    )


def _make_compound_node(name: str) -> _Node:
    """Create one compound layout node.

    Parameters
    ----------
    name : str
        Cluster id.

    Returns
    -------
    _Node
        New compound node.
    """
    return _Node(id=name, rect=_Rect(0.0, 0.0, 1.0, 1.0))


def _populate_graph(
    state: CoSECompoundState,
    graph: _Graph,
    parent: Optional[str],
    child_clusters: Mapping[Optional[str], Sequence[str]],
    direct_nodes: Mapping[Optional[str], Sequence[int]],
    sizes: Sequence[Tuple[float, float]],
) -> None:
    """Populate one inclusion graph recursively.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable compound state.
    graph : _Graph
        Graph to populate.
    parent : str | None
        Parent cluster id or ``None`` for root.
    child_clusters : mapping[str | None, sequence[str]]
        Direct child clusters keyed by parent cluster.
    direct_nodes : mapping[str | None, sequence[int]]
        Direct leaf nodes keyed by parent cluster.
    sizes : sequence[tuple[float, float]]
        Leaf node dimensions.

    Returns
    -------
    None
        Mutates the graph manager state.
    """
    for cluster_name in child_clusters.get(parent, []):
        compound = _add_node(graph, _make_compound_node(cluster_name))
        state.nodes_by_id[cluster_name] = compound
        child_graph = _append_graph(state, compound)
        _populate_graph(state, child_graph, cluster_name, child_clusters, direct_nodes, sizes)
    for leaf_index in direct_nodes.get(parent, []):
        leaf = _add_node(graph, _make_leaf_node(leaf_index, sizes[leaf_index]))
        state.nodes_by_leaf[leaf_index] = leaf
        state.nodes_by_id[str(leaf_index)] = leaf


def _ancestor_graphs(node: _Node) -> List[_Graph]:
    """Return owner graph ancestors from owner to root.

    Parameters
    ----------
    node : _Node
        Node whose owner path is requested.

    Returns
    -------
    list[_Graph]
        Owner graph followed by ancestor graphs.
    """
    result: List[_Graph] = []
    graph = node.owner
    while graph is not None:
        result.append(graph)
        parent = graph.parent
        graph = parent.owner if parent is not None else None
    return result


def _child_in_graph(node: _Node, graph: _Graph) -> Optional[_Node]:
    """Return the descendant representative owned directly by ``graph``.

    Parameters
    ----------
    node : _Node
        Edge endpoint node.
    graph : _Graph
        Lowest common ancestor graph.

    Returns
    -------
    _Node | None
        Direct child of ``graph`` on the path to ``node``.
    """
    current = node
    while current.owner is not None and current.owner != graph:
        parent = current.owner.parent
        if parent is None:
            return None
        current = parent
    return current


def _edge_lca(source: _Node, target: _Node) -> Optional[_Graph]:
    """Return the lowest common owner graph of two nodes.

    Parameters
    ----------
    source : _Node
        Source endpoint.
    target : _Node
        Target endpoint.

    Returns
    -------
    _Graph | None
        Lowest common ancestor graph.
    """
    target_ancestors = set(_ancestor_graphs(target))
    for graph in _ancestor_graphs(source):
        if graph in target_ancestors:
            return graph
    return None


def _add_unique_edges(
    state: CoSECompoundState,
    edge_index: torch.Tensor,
    num_nodes: int,
) -> None:
    """Add de-duplicated non-self edges to the compound graph manager.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable layout state.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of leaf nodes.

    Returns
    -------
    None
        Mutates ``state``.
    """
    seen: set[Tuple[int, int]] = set()
    if edge_index.numel() == 0:
        return
    edges = edge_index.detach().cpu().long()
    for edge_pos in range(edges.shape[1]):
        source_index = int(edges[0, edge_pos].item())
        target_index = int(edges[1, edge_pos].item())
        if (
            source_index == target_index
            or source_index < 0
            or target_index < 0
            or source_index >= num_nodes
            or target_index >= num_nodes
        ):
            continue
        key = (
            (source_index, target_index)
            if source_index < target_index
            else (
                target_index,
                source_index,
            )
        )
        if key in seen:
            continue
        seen.add(key)
        source = state.nodes_by_leaf[source_index]
        target = state.nodes_by_leaf[target_index]
        source_graph = source.owner
        target_graph = target.owner
        is_inter_graph = source_graph is not target_graph
        edge = _Edge(source=source, target=target, is_inter_graph=is_inter_graph)
        if is_inter_graph:
            state.edges.append(edge)
        elif source_graph is not None:
            source_graph.edges.append(edge)
            state.edges.append(edge)
        source.edges.append(edge)
        target.edges.append(edge)


def build_cose_compound_state(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    clusters: Optional[Mapping[str, Any]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    options: CoSECompoundOptions,
) -> CoSECompoundState:
    """Build the shared compound-CoSE graph model.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of leaf nodes.
    node_sizes : torch.Tensor | None
        Optional node size tensor with shape ``[N, 2]``.
    clusters : mapping[str, Any] | None
        Cluster membership metadata.
    cluster_parents : mapping[str, str | None] | None
        Cluster hierarchy metadata.
    options : CoSECompoundOptions
        Runtime options.

    Returns
    -------
    CoSECompoundState
        Initialized compound state.
    """
    raw_state = int(options.seed) & 0xFFFFFFFF
    if raw_state == 0:
        raw_state = 1
    root = _Graph(parent=None)
    state = CoSECompoundState(
        root=root,
        graphs=[root],
        nodes_by_leaf={},
        nodes_by_id={},
        edges=[],
        options=options,
        random_state=raw_state,
    )
    members = _cluster_members(clusters, num_nodes)
    child_clusters, direct_nodes = _child_maps(members, cluster_parents, num_nodes)
    sizes = _normalize_sizes(node_sizes, num_nodes)
    _populate_graph(state, root, None, child_clusters, direct_nodes, sizes)
    _add_unique_edges(state, edge_index, num_nodes)
    return state


def _all_nodes(state: CoSECompoundState) -> List[_Node]:
    """Return all layout nodes in graph-manager order.

    Parameters
    ----------
    state : CoSECompoundState
        Compound state.

    Returns
    -------
    list[_Node]
        Nodes from every graph in encounter order.
    """
    return [node for graph in state.graphs for node in graph.nodes]


def _all_edges(state: CoSECompoundState) -> List[_Edge]:
    """Return all layout edges in graph-manager order.

    Parameters
    ----------
    state : CoSECompoundState
        Compound state.

    Returns
    -------
    list[_Edge]
        All edges.
    """
    return [
        edge
        for edge in state.edges
        if edge in edge.source.edges
        and edge in edge.target.edges
        and edge.source.owner is not None
        and edge.target.owner is not None
        and edge.source in edge.source.owner.nodes
        and edge.target in edge.target.owner.nodes
    ]


def _graph_margin(graph: _Graph) -> float:
    """Return the margin used by layout-base for one graph.

    Parameters
    ----------
    graph : _Graph
        Graph whose bound margin is requested.

    Returns
    -------
    float
        Parent padding if present, otherwise default graph margin.
    """
    if graph.parent is not None:
        return graph.parent.padding_left
    return _DEFAULT_GRAPH_MARGIN


def _update_graph_bounds(graph: _Graph, recursive: bool) -> None:
    """Update a graph and compound parent bounds.

    Parameters
    ----------
    graph : _Graph
        Graph to update.
    recursive : bool
        Whether to update child graphs first.

    Returns
    -------
    None
        Mutates graph and parent rectangles.
    """
    if recursive:
        for node in graph.nodes:
            if node.child is not None:
                _update_graph_bounds(node.child, True)
                _update_node_bounds(node)
    if not graph.nodes:
        if graph.parent is not None:
            left = graph.parent.rect.x
            top = graph.parent.rect.y
            graph.left = left
            graph.right = left + graph.parent.rect.width
            graph.top = top
            graph.bottom = top + graph.parent.rect.height
        return
    left = min(node.rect.x for node in graph.nodes)
    right = max(node.rect.x + node.rect.width for node in graph.nodes)
    top = min(node.rect.y for node in graph.nodes)
    bottom = max(node.rect.y + node.rect.height for node in graph.nodes)
    margin = _graph_margin(graph)
    graph.left = left - margin
    graph.right = right + margin
    graph.top = top - margin
    graph.bottom = bottom + margin


def _update_node_bounds(node: _Node) -> None:
    """Update a compound node rectangle from its child graph bounds.

    Parameters
    ----------
    node : _Node
        Compound node.

    Returns
    -------
    None
        Mutates ``node``.
    """
    if node.child is None:
        return
    graph = node.child
    node.rect.x = graph.left
    node.rect.y = graph.top
    node.rect.width = max(graph.right - graph.left, _EMPTY_COMPOUND_NODE_SIZE)
    node.rect.height = max(graph.bottom - graph.top, _EMPTY_COMPOUND_NODE_SIZE)


def _calc_estimated_size_node(node: _Node) -> float:
    """Calculate cose-base estimated size for one node.

    Parameters
    ----------
    node : _Node
        Layout node.

    Returns
    -------
    float
        Estimated size.
    """
    if node.child is None:
        node.estimated_size = math.sqrt(node.rect.width * node.rect.height)
        return node.estimated_size
    node.estimated_size = _calc_estimated_size_graph(node.child)
    return node.estimated_size


def _calc_estimated_size_graph(graph: _Graph) -> float:
    """Calculate cose-base estimated size for one graph.

    Parameters
    ----------
    graph : _Graph
        Inclusion graph.

    Returns
    -------
    float
        Estimated size.
    """
    size = sum(_calc_estimated_size_node(node) for node in graph.nodes)
    if size == 0.0:
        graph.estimated_size = _EMPTY_COMPOUND_NODE_SIZE
    else:
        graph.estimated_size = size / math.sqrt(len(graph.nodes))
    return graph.estimated_size


def _calc_inclusion_depths(state: CoSECompoundState) -> None:
    """Populate graph and node inclusion-tree depths.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable layout state.

    Returns
    -------
    None
        Mutates graph and node depth fields.
    """
    graph_depths: Dict[int, int] = {id(state.root): 1}
    for graph in state.graphs:
        graph_depth = graph_depths.get(id(graph), 1)
        for node in graph.nodes:
            node.inclusion_tree_depth = graph_depth + 1
            if node.child is not None:
                graph_depths[id(node.child)] = node.inclusion_tree_depth


def _calc_no_of_children(node: _Node) -> int:
    """Return the recursive leaf weight used by cose-base forces.

    Parameters
    ----------
    node : _Node
        Layout node.

    Returns
    -------
    int
        Number of descendant leaves, at least one.
    """
    if node.child is None:
        node.no_of_children = 1
        return 1
    total = sum(_calc_no_of_children(child) for child in node.child.nodes)
    node.no_of_children = max(total, 1)
    return node.no_of_children


def _calc_no_of_children_all(state: CoSECompoundState) -> None:
    """Populate all node weights.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable layout state.

    Returns
    -------
    None
        Mutates node weights.
    """
    for node in _all_nodes(state):
        _calc_no_of_children(node)


def _get_other_end(edge: _Edge, node: _Node) -> _Node:
    """Return the opposite endpoint of an edge.

    Parameters
    ----------
    edge : _Edge
        Edge incident to ``node``.
    node : _Node
        Endpoint.

    Returns
    -------
    _Node
        Opposite endpoint.
    """
    if edge.source is node:
        return edge.target
    if edge.target is node:
        return edge.source
    raise ValueError("node is not incident with edge")


def _get_other_end_in_graph(
    edge: _Edge,
    node: _Node,
    graph: _Graph,
    root: _Graph,
) -> Optional[_Node]:
    """Return the endpoint representative in a requested graph.

    Parameters
    ----------
    edge : _Edge
        Edge to traverse.
    node : _Node
        Current endpoint.
    graph : _Graph
        Graph whose direct representative is requested.
    root : _Graph
        Root inclusion graph.

    Returns
    -------
    _Node | None
        Neighbor representative in ``graph`` or ``None`` if outside.
    """
    other = _get_other_end(edge, node)
    while True:
        if other.owner is graph:
            return other
        if other.owner is root:
            break
        if other.owner is None or other.owner.parent is None:
            break
        other = other.owner.parent
    return None


def _update_connected(graph: _Graph, root: _Graph) -> None:
    """Update graph connectivity using layout-base child expansion.

    Parameters
    ----------
    graph : _Graph
        Graph to test.
    root : _Graph
        Root inclusion graph.

    Returns
    -------
    None
        Mutates ``graph.is_connected``.
    """
    if not graph.nodes:
        graph.is_connected = True
        return
    queue = list(_with_children(graph.nodes[0]))
    visited = set(queue)
    while queue:
        current = queue.pop(0)
        for edge in current.edges:
            neighbor = _get_other_end_in_graph(edge, current, graph, root)
            if neighbor is None or neighbor in visited:
                continue
            for child in _with_children(neighbor):
                if child not in visited:
                    visited.add(child)
                    queue.append(child)
    graph.is_connected = sum(1 for node in visited if node.owner is graph) == len(graph.nodes)


def _with_children(node: _Node) -> List[_Node]:
    """Return a node and all descendant layout nodes.

    Parameters
    ----------
    node : _Node
        Root node.

    Returns
    -------
    list[_Node]
        Node plus descendants.
    """
    result = [node]
    if node.child is not None:
        for child in node.child.nodes:
            result.extend(_with_children(child))
    return result


def _calculate_nodes_to_apply_gravitation_to(state: CoSECompoundState) -> List[_Node]:
    """Return nodes selected by cose-base compound gravity gating.

    Parameters
    ----------
    state : CoSECompoundState
        Compound state.

    Returns
    -------
    list[_Node]
        Nodes in graphs that are not internally connected.
    """
    result: List[_Node] = []
    for graph in state.graphs:
        _update_connected(graph, state.root)
        if not graph.is_connected:
            result.extend(graph.nodes)
    return result


def _calc_lowest_common_ancestors(state: CoSECompoundState) -> None:
    """Populate LCA metadata for all edges.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable layout state.

    Returns
    -------
    None
        Mutates edge LCA fields.
    """
    for edge in _all_edges(state):
        lca = _edge_lca(edge.source, edge.target)
        edge.lca = lca
        edge.source_in_lca = _child_in_graph(edge.source, lca) if lca is not None else None
        edge.target_in_lca = _child_in_graph(edge.target, lca) if lca is not None else None


def _calc_ideal_edge_lengths(state: CoSECompoundState) -> None:
    """Apply cose-base inter-graph nesting-factor edge lengths.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable layout state.

    Returns
    -------
    None
        Mutates ``edge.ideal_length``.
    """
    options = state.options
    for edge in _all_edges(state):
        edge.ideal_length = options.ideal_edge_length
        if not edge.is_inter_graph or edge.lca is None:
            continue
        source_in_lca = edge.source_in_lca
        target_in_lca = edge.target_in_lca
        if source_in_lca is None or target_in_lca is None:
            continue
        edge.ideal_length += (
            source_in_lca.estimated_size + target_in_lca.estimated_size - 2.0 * _SIMPLE_NODE_SIZE
        )
        lca_depth = _graph_depth(edge.lca, state.root)
        edge.ideal_length += (
            _DEFAULT_EDGE_LENGTH
            * options.nesting_factor
            * (edge.source.inclusion_tree_depth + edge.target.inclusion_tree_depth - 2 * lca_depth)
        )


def _graph_depth(graph: _Graph, root: _Graph) -> int:
    """Return layout-base inclusion depth for a graph.

    Parameters
    ----------
    graph : _Graph
        Graph whose depth is requested.
    root : _Graph
        Root graph.

    Returns
    -------
    int
        Root graph depth is one.
    """
    if graph is root:
        return 1
    parent = graph.parent
    if parent is None:
        return 1
    return parent.inclusion_tree_depth


def _scatter_node(state: CoSECompoundState, node: _Node) -> None:
    """Scatter one leaf or empty compound using layout-base ``RandomSeed``.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state carrying the sine RNG seed.
    node : _Node
        Node to scatter.

    Returns
    -------
    None
        Mutates node position and RNG seed.
    """
    state.sine_seed, random_x = _next_sine_random(state.sine_seed)
    state.sine_seed, random_y = _next_sine_random(state.sine_seed)
    node.rect.x = random_x * _INITIAL_WORLD_BOUNDARY
    node.rect.y = random_y * _INITIAL_WORLD_BOUNDARY


def _position_nodes_randomly(state: CoSECompoundState, graph: _Graph) -> None:
    """Recursively scatter leaves and update compound bounds.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.
    graph : _Graph
        Graph to position.

    Returns
    -------
    None
        Mutates node rectangles.
    """
    for node in graph.nodes:
        if node.child is None or len(node.child.nodes) == 0:
            _scatter_node(state, node)
        else:
            _position_nodes_randomly(state, node.child)
            _update_graph_bounds(node.child, True)
            _update_node_bounds(node)


def _get_flat_forest(root: _Graph) -> List[List[_Node]]:
    """Return flat forest components or an empty list if not a flat forest.

    Parameters
    ----------
    root : _Graph
        Root graph.

    Returns
    -------
    list[list[_Node]]
        Tree components when the root graph is flat and acyclic.
    """
    if any(node.child is not None for node in root.nodes):
        return []
    unprocessed = list(root.nodes)
    forest: List[List[_Node]] = []
    while unprocessed:
        queue = [unprocessed[0]]
        visited: set[_Node] = set()
        parents: Dict[_Node, _Node] = {}
        is_tree = True
        while queue and is_tree:
            current = queue.pop(0)
            visited.add(current)
            for edge in current.edges:
                neighbor = _get_other_end(edge, current)
                if parents.get(current) is neighbor:
                    continue
                if neighbor not in visited:
                    queue.append(neighbor)
                    parents[neighbor] = current
                else:
                    is_tree = False
                    break
        if not is_tree:
            return []
        component = list(visited)
        forest.append(component)
        for node in component:
            if node in unprocessed:
                unprocessed.remove(node)
    return forest


def _position_nodes_radially(state: CoSECompoundState, forest: Sequence[Sequence[_Node]]) -> None:
    """Apply cose-base radial forest initialization.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.
    forest : sequence[sequence[_Node]]
        Flat tree components.

    Returns
    -------
    None
        Mutates node positions.
    """
    if not forest:
        return
    number_of_columns = math.ceil(math.sqrt(len(forest)))
    height = 0.0
    current_y = 0.0
    current_x = 0.0
    point_x = 0.0
    point_y = 0.0
    for index, tree in enumerate(forest):
        if index % number_of_columns == 0:
            current_x = 0.0
            current_y = height
            if index != 0:
                current_y += _DEFAULT_COMPONENT_SEPARATION
            height = 0.0
        center = _find_center_of_tree(tree)
        point_x, point_y = _radial_layout(tree, center, current_x, current_y)
        if point_y > height:
            height = math.floor(point_y)
        current_x = math.floor(point_x + _DEFAULT_COMPONENT_SEPARATION)
    _transform_to(state, _WORLD_CENTER_X - point_x / 2.0, _WORLD_CENTER_Y - point_y / 2.0)


def _find_center_of_tree(tree: Sequence[_Node]) -> _Node:
    """Return a deterministic center node for a tree component.

    Parameters
    ----------
    tree : sequence[_Node]
        Tree nodes.

    Returns
    -------
    _Node
        Node with minimum eccentricity; encounter-order tie break.
    """
    best_node = tree[0]
    best_ecc = math.inf
    for node in tree:
        distances = _tree_distances(node)
        ecc = max(distances.values()) if distances else 0
        if ecc < best_ecc:
            best_ecc = ecc
            best_node = node
    return best_node


def _tree_distances(start: _Node) -> Dict[_Node, int]:
    """Return BFS distances from one tree node.

    Parameters
    ----------
    start : _Node
        BFS start node.

    Returns
    -------
    dict[_Node, int]
        Distances by node.
    """
    distances = {start: 0}
    queue = [start]
    while queue:
        current = queue.pop(0)
        for edge in current.edges:
            neighbor = _get_other_end(edge, current)
            if neighbor not in distances:
                distances[neighbor] = distances[current] + 1
                queue.append(neighbor)
    return distances


def _radial_layout(
    tree: Sequence[_Node],
    center: _Node,
    start_x: float,
    start_y: float,
) -> Tuple[float, float]:
    """Layout one tree radially and translate to a starting point.

    Parameters
    ----------
    tree : sequence[_Node]
        Tree component.
    center : _Node
        Center node.
    start_x : float
        Starting x-coordinate.
    start_y : float
        Starting y-coordinate.

    Returns
    -------
    tuple[float, float]
        Bottom-right point after translation.
    """
    radial_sep = max((math.hypot(node.rect.width, node.rect.height) for node in tree), default=0.0)
    radial_sep = max(radial_sep, _DEFAULT_RADIAL_SEPARATION)
    _branch_radial_layout(center, None, 0.0, 359.0, 0.0, radial_sep)
    left = min(node.rect.x for node in tree)
    top = min(node.rect.y for node in tree)
    right = max(node.rect.x + node.rect.width for node in tree)
    bottom = max(node.rect.y + node.rect.height for node in tree)
    dx = start_x - left
    dy = start_y - top
    for node in tree:
        _move_by(node, dx, dy)
    return right + dx, bottom + dy


def _branch_radial_layout(
    node: _Node,
    parent: Optional[_Node],
    start_angle: float,
    end_angle: float,
    distance: float,
    radial_separation: float,
) -> None:
    """Recursively place one radial tree branch.

    Parameters
    ----------
    node : _Node
        Current node.
    parent : _Node | None
        Parent node in the traversal.
    start_angle : float
        Start angle in degrees.
    end_angle : float
        End angle in degrees.
    distance : float
        Radius for this node.
    radial_separation : float
        Radius increment.

    Returns
    -------
    None
        Mutates node positions.
    """
    half_interval = ((end_angle - start_angle) + 1.0) / 2.0
    if half_interval < 0.0:
        half_interval += 180.0
    node_angle = (half_interval + start_angle) % 360.0
    theta = node_angle * math.tau / 360.0
    _set_center(node, distance * math.cos(theta), distance * math.sin(theta))
    neighbor_edges = list(node.edges)
    child_count = len(neighbor_edges) - (1 if parent is not None else 0)
    if child_count <= 0:
        return
    edges_to_parent = [
        edge
        for edge in neighbor_edges
        if parent is not None and _get_other_end(edge, node) is parent
    ]
    while len(edges_to_parent) > 1:
        edge = edges_to_parent.pop(0)
        if edge in neighbor_edges:
            neighbor_edges.remove(edge)
            child_count -= 1
    inc_edges_count = len(neighbor_edges)
    if inc_edges_count == 0 or child_count <= 0:
        return
    start_index = (
        (neighbor_edges.index(edges_to_parent[0]) + 1) % inc_edges_count
        if parent is not None and edges_to_parent
        else 0
    )
    step_angle = abs(end_angle - start_angle) / child_count
    branch_count = 0
    index = start_index
    while branch_count != child_count:
        neighbor = _get_other_end(neighbor_edges[index], node)
        if neighbor is not parent:
            child_start = (start_angle + branch_count * step_angle) % 360.0
            child_end = (child_start + step_angle) % 360.0
            _branch_radial_layout(
                neighbor,
                node,
                child_start,
                child_end,
                distance + radial_separation,
                radial_separation,
            )
            branch_count += 1
        index = (index + 1) % inc_edges_count


def _transform_to(state: CoSECompoundState, new_left: float, new_top: float) -> None:
    """Translate all nodes so the root graph left-top matches a target.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.
    new_left : float
        Target root left coordinate.
    new_top : float
        Target root top coordinate.

    Returns
    -------
    None
        Mutates all node positions.
    """
    _update_graph_bounds(state.root, True)
    if not state.root.nodes:
        return
    margin = _graph_margin(state.root)
    left_top_x = min(node.rect.x for node in state.root.nodes) - margin
    left_top_y = min(node.rect.y for node in state.root.nodes) - margin
    dx = new_left - left_top_x
    dy = new_top - left_top_y
    for node in _all_nodes(state):
        _move_by(node, dx, dy)


def _node_degree(node: _Node) -> int:
    """Return node degree excluding self loops.

    Parameters
    ----------
    node : _Node
        Layout node.

    Returns
    -------
    int
        Degree.
    """
    return sum(1 for edge in node.edges if edge.source is not edge.target)


def _node_degree_with_children(node: _Node) -> int:
    """Return degree including child contents.

    Parameters
    ----------
    node : _Node
        Layout node.

    Returns
    -------
    int
        Recursive degree.
    """
    degree = _node_degree(node)
    if node.child is None:
        return degree
    return degree + sum(_node_degree_with_children(child) for child in node.child.nodes)


def _get_to_be_tiled(state: CoSECompoundState, node: _Node) -> bool:
    """Return cose-base ``getToBeTiled`` for one compound node.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state with memoized tile decisions.
    node : _Node
        Candidate node.

    Returns
    -------
    bool
        Whether the node's children should be replaced by a tiled pack.
    """
    if node.id in state.to_be_tiled:
        return state.to_be_tiled[node.id]
    if node.child is None:
        state.to_be_tiled[node.id] = False
        return False
    for child in node.child.nodes:
        if _node_degree(child) > 0:
            state.to_be_tiled[node.id] = False
            return False
        if child.child is None:
            state.to_be_tiled[child.id] = False
            continue
        if not _get_to_be_tiled(state, child):
            state.to_be_tiled[node.id] = False
            return False
    state.to_be_tiled[node.id] = True
    return True


def _tile_nodes(
    nodes: Sequence[_Node],
    min_width: float,
    vertical_padding: float,
    horizontal_padding: float,
) -> _TileOrganization:
    """Port cose-base ``tileNodes`` row-filling strip packing.

    Parameters
    ----------
    nodes : sequence[_Node]
        Nodes to tile.
    min_width : float
        Minimum compound width.
    vertical_padding : float
        Row vertical padding.
    horizontal_padding : float
        Column horizontal padding.

    Returns
    -------
    _TileOrganization
        Row organization and resulting dimensions.
    """
    organization = _TileOrganization(
        rows=[],
        row_width=[],
        row_height=[],
        width=0.0,
        height=float(min_width),
        vertical_padding=float(vertical_padding),
        horizontal_padding=float(horizontal_padding),
    )
    sorted_nodes = sorted(
        nodes,
        key=lambda node: node.rect.width * node.rect.height,
        reverse=True,
    )
    for node in sorted_nodes:
        if not organization.rows:
            _insert_node_to_row(organization, node, 0, min_width)
        elif _can_add_horizontal(organization, node.rect.width, node.rect.height):
            _insert_node_to_row(organization, node, _shortest_row_index(organization), min_width)
        else:
            _insert_node_to_row(organization, node, len(organization.rows), min_width)
    return organization


def _insert_node_to_row(
    organization: _TileOrganization,
    node: _Node,
    row_index: int,
    min_width: float,
) -> None:
    """Insert one node into a tile row.

    Parameters
    ----------
    organization : _TileOrganization
        Mutable row organization.
    node : _Node
        Node to insert.
    row_index : int
        Target row index.
    min_width : float
        Minimum compound width.

    Returns
    -------
    None
        Mutates ``organization``.
    """
    if row_index == len(organization.rows):
        organization.rows.append([])
        organization.row_width.append(float(min_width))
        organization.row_height.append(0.0)
    width = organization.row_width[row_index] + node.rect.width
    if organization.rows[row_index]:
        width += organization.horizontal_padding
    organization.row_width[row_index] = width
    organization.width = max(organization.width, width)
    height = node.rect.height + (organization.vertical_padding if row_index > 0 else 0.0)
    if height > organization.row_height[row_index]:
        extra = height - organization.row_height[row_index]
        organization.row_height[row_index] = height
        organization.height += extra
    organization.rows[row_index].append(node)


def _shortest_row_index(organization: _TileOrganization) -> int:
    """Return the row with minimum width.

    Parameters
    ----------
    organization : _TileOrganization
        Row organization.

    Returns
    -------
    int
        Shortest row index, or ``-1`` when no rows exist.
    """
    if not organization.row_width:
        return -1
    return min(range(len(organization.row_width)), key=lambda index: organization.row_width[index])


def _can_add_horizontal(
    organization: _TileOrganization,
    extra_width: float,
    extra_height: float,
) -> bool:
    """Return whether adding a node to the shortest row improves aspect ratio.

    Parameters
    ----------
    organization : _TileOrganization
        Row organization.
    extra_width : float
        Width of the candidate node.
    extra_height : float
        Height of the candidate node.

    Returns
    -------
    bool
        Whether to add to an existing row.
    """
    shortest = _shortest_row_index(organization)
    if shortest < 0:
        return True
    min_width = organization.row_width[shortest]
    if min_width + organization.horizontal_padding + extra_width <= organization.width:
        return True
    height_diff = 0.0
    if organization.row_height[shortest] < extra_height and shortest > 0:
        height_diff = (
            extra_height + organization.vertical_padding - organization.row_height[shortest]
        )
    if organization.width - min_width >= extra_width + organization.horizontal_padding:
        add_to_row_ratio = (organization.height + height_diff) / (
            min_width + extra_width + organization.horizontal_padding
        )
    else:
        add_to_row_ratio = (organization.height + height_diff) / max(
            organization.width,
            _AREA_EPSILON,
        )
    height_diff = extra_height + organization.vertical_padding
    if organization.width < extra_width:
        add_new_row_ratio = (organization.height + height_diff) / max(extra_width, _AREA_EPSILON)
    else:
        add_new_row_ratio = (organization.height + height_diff) / max(
            organization.width,
            _AREA_EPSILON,
        )
    if add_new_row_ratio < 1.0:
        add_new_row_ratio = 1.0 / add_new_row_ratio
    if add_to_row_ratio < 1.0:
        add_to_row_ratio = 1.0 / add_to_row_ratio
    return add_to_row_ratio < add_new_row_ratio


def _adjust_locations(
    organization: _TileOrganization,
    x: float,
    y: float,
    compound_horizontal_margin: float,
    compound_vertical_margin: float,
) -> None:
    """Place tiled nodes relative to a compound top-left.

    Parameters
    ----------
    organization : _TileOrganization
        Row organization.
    x : float
        Compound top-left x-coordinate.
    y : float
        Compound top-left y-coordinate.
    compound_horizontal_margin : float
        Left padding.
    compound_vertical_margin : float
        Top padding.

    Returns
    -------
    None
        Mutates member node locations.
    """
    current_y = y + compound_vertical_margin
    left = x + compound_horizontal_margin
    for row in organization.rows:
        current_x = left
        max_height = 0.0
        for node in row:
            node.rect.x = current_x
            node.rect.y = current_y
            current_x += node.rect.width + organization.horizontal_padding
            max_height = max(max_height, node.rect.height)
        current_y += max_height + organization.vertical_padding


def _perform_dfs_on_compounds(state: CoSECompoundState) -> None:
    """Populate the inner-first compound tiling order.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates ``state.compound_order``.
    """
    state.compound_order = []
    _fill_compound_order_by_dfs(state, state.root.nodes)


def _fill_compound_order_by_dfs(state: CoSECompoundState, children: Sequence[_Node]) -> None:
    """DFS helper for compound tiling order.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.
    children : sequence[_Node]
        Child nodes to scan.

    Returns
    -------
    None
        Mutates ``state.compound_order``.
    """
    for child in children:
        if child.child is not None:
            _fill_compound_order_by_dfs(state, child.child.nodes)
        if _get_to_be_tiled(state, child):
            state.compound_order.append(child)


def _group_zero_degree_members(state: CoSECompoundState) -> None:
    """Port cose-base ``groupZeroDegreeMembers``.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates graph hierarchy with dummy compounds.
    """
    temp_groups: Dict[str, List[_Node]] = {}
    state.member_groups = {}
    state.id_to_dummy_node = {}
    for node in list(_all_nodes(state)):
        parent = node.owner.parent if node.owner is not None else None
        if parent is None:
            continue
        if _node_degree_with_children(node) == 0 and not _get_to_be_tiled(state, parent):
            temp_groups.setdefault(parent.id, []).append(node)
    for parent_id, grouped_nodes in temp_groups.items():
        if len(grouped_nodes) <= 1:
            continue
        parent = grouped_nodes[0].owner.parent if grouped_nodes[0].owner is not None else None
        if parent is None or parent.child is None:
            continue
        dummy_id = f"DummyCompound_{parent_id}"
        dummy = _Node(
            id=dummy_id,
            rect=_Rect(0.0, 0.0, 1.0, 1.0),
            padding_left=parent.padding_left,
            padding_top=parent.padding_top,
            padding_right=parent.padding_right,
            padding_bottom=parent.padding_bottom,
        )
        state.member_groups[dummy_id] = grouped_nodes
        state.id_to_dummy_node[dummy_id] = dummy
        dummy_graph = _append_graph(state, dummy)
        parent_graph = parent.child
        _add_node(parent_graph, dummy)
        for node in grouped_nodes:
            if node in parent_graph.nodes:
                parent_graph.nodes.remove(node)
            _add_node(dummy_graph, node)


def _clear_compounds(state: CoSECompoundState) -> None:
    """Port cose-base ``clearCompounds``.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Replaces tiled child graphs with compound boxes.
    """
    child_graph_map: Dict[str, List[_Node]] = {}
    id_to_node: Dict[str, _Node] = {}
    _perform_dfs_on_compounds(state)
    for compound in state.compound_order:
        if compound.child is None:
            continue
        id_to_node[compound.id] = compound
        child_graph_map[compound.id] = list(compound.child.nodes)
        if compound.child in state.graphs:
            state.graphs.remove(compound.child)
        compound.child = None
    _tile_compound_members(state, child_graph_map, id_to_node)


def _tile_compound_members(
    state: CoSECompoundState,
    child_graph_map: Mapping[str, Sequence[_Node]],
    id_to_node: Mapping[str, _Node],
) -> None:
    """Port cose-base ``tileCompoundMembers``.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.
    child_graph_map : mapping[str, sequence[_Node]]
        Removed children keyed by compound id.
    id_to_node : mapping[str, _Node]
        Compound node lookup.

    Returns
    -------
    None
        Mutates compound sizes.
    """
    state.tiled_member_pack = {}
    for compound_id, children in child_graph_map.items():
        compound = id_to_node[compound_id]
        organization = _tile_nodes(
            children,
            compound.padding_left + compound.padding_right,
            state.options.tiling_padding_vertical,
            state.options.tiling_padding_horizontal,
        )
        state.tiled_member_pack[compound_id] = organization
        compound.rect.width = organization.width
        compound.rect.height = organization.height


def _clear_zero_degree_members(state: CoSECompoundState) -> None:
    """Port cose-base ``clearZeroDegreeMembers``.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates dummy compound dimensions.
    """
    state.tiled_zero_degree_pack = {}
    for dummy_id, members in state.member_groups.items():
        compound = state.id_to_dummy_node[dummy_id]
        organization = _tile_nodes(
            members,
            compound.padding_left + compound.padding_right,
            state.options.tiling_padding_vertical,
            state.options.tiling_padding_horizontal,
        )
        state.tiled_zero_degree_pack[dummy_id] = organization
        compound.rect.width = organization.width
        compound.rect.height = organization.height


def _repopulate_zero_degree_members(state: CoSECompoundState) -> None:
    """Port cose-base ``repopulateZeroDegreeMembers``.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Restores zero-degree member positions inside dummy compounds.
    """
    for dummy_id, organization in state.tiled_zero_degree_pack.items():
        compound = state.id_to_dummy_node[dummy_id]
        _adjust_locations(
            organization,
            compound.rect.x,
            compound.rect.y,
            compound.padding_left,
            compound.padding_top,
        )


def _repopulate_compounds(state: CoSECompoundState) -> None:
    """Port cose-base ``repopulateCompounds``.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Restores tiled compound child positions.
    """
    for compound in reversed(state.compound_order):
        organization = state.tiled_member_pack.get(compound.id)
        if organization is None:
            continue
        _adjust_locations(
            organization,
            compound.rect.x,
            compound.rect.y,
            compound.padding_left,
            compound.padding_top,
        )


def _tiling_pre_layout(state: CoSECompoundState) -> None:
    """Apply cose-base tiling pre-layout sequence.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates hierarchy when tiling is enabled.
    """
    if not state.options.tile:
        return
    _group_zero_degree_members(state)
    _clear_compounds(state)
    _clear_zero_degree_members(state)


def _tiling_post_layout(state: CoSECompoundState) -> None:
    """Apply cose-base tiling post-layout sequence.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Restores tiled child positions.
    """
    if not state.options.tile:
        return
    _repopulate_zero_degree_members(state)
    _repopulate_compounds(state)


def _reduce_trees(state: CoSECompoundState) -> None:
    """Port cose-base ``reduceTrees`` leaf pruning.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates graph contents and stores pruned steps.
    """
    pruned_all: List[List[Tuple[_Node, _Edge, _Graph]]] = []
    contains_leaf = True
    while contains_leaf:
        all_nodes = list(_all_nodes(state))
        candidates: List[Tuple[_Node, _Edge, _Graph]] = []
        contains_leaf = False
        for node in all_nodes:
            if len(node.edges) == 1 and not node.edges[0].is_inter_graph and node.child is None:
                if node.owner is not None:
                    candidates.append((node, node.edges[0], node.owner))
                    contains_leaf = True
        if contains_leaf:
            pruned_step: List[Tuple[_Node, _Edge, _Graph]] = []
            for node, edge, owner in candidates:
                if len(node.edges) == 1:
                    pruned_step.append((node, edge, owner))
                    _remove_node(owner, node)
            pruned_all.append(pruned_step)
    state.pruned_nodes_all = pruned_all


def _remove_node(graph: _Graph, node: _Node) -> None:
    """Remove a node and incident edges from its owner graph.

    Parameters
    ----------
    graph : _Graph
        Owner graph.
    node : _Node
        Node to remove.

    Returns
    -------
    None
        Mutates graph and incident edge lists.
    """
    for edge in list(node.edges):
        other = _get_other_end(edge, node)
        if edge in other.edges:
            other.edges.remove(edge)
        if edge in graph.edges:
            graph.edges.remove(edge)
        node.edges.remove(edge)
    if node in graph.nodes:
        graph.nodes.remove(node)
    node.owner = graph


def _restore_pruned_edge(edge: _Edge) -> None:
    """Restore an edge into endpoint incidence and owner lists.

    Parameters
    ----------
    edge : _Edge
        Edge to restore.

    Returns
    -------
    None
        Mutates incidence lists.
    """
    if edge not in edge.source.edges:
        edge.source.edges.append(edge)
    if edge not in edge.target.edges:
        edge.target.edges.append(edge)
    if (
        not edge.is_inter_graph
        and edge.source.owner is not None
        and edge not in edge.source.owner.edges
    ):
        edge.source.owner.edges.append(edge)


def _update_grid(state: CoSECompoundState) -> None:
    """Update the FR-grid used by tree regrowth placement.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates ``state.grid`` and node grid coordinates.
    """
    width = max(state.root.right - state.root.left, state.repulsion_range)
    height = max(state.root.bottom - state.root.top, state.repulsion_range)
    size_x = max(int(math.ceil(width / state.repulsion_range)), 1)
    size_y = max(int(math.ceil(height / state.repulsion_range)), 1)
    grid: List[List[List[_Node]]] = [[[] for _ in range(size_y)] for _ in range(size_x)]
    for node in _all_nodes(state):
        start_x = int(math.floor((node.rect.x - state.root.left) / state.repulsion_range))
        finish_x = int(
            math.floor((node.rect.width + node.rect.x - state.root.left) / state.repulsion_range)
        )
        start_y = int(math.floor((node.rect.y - state.root.top) / state.repulsion_range))
        finish_y = int(
            math.floor((node.rect.height + node.rect.y - state.root.top) / state.repulsion_range)
        )
        start_x = max(0, min(start_x, size_x - 1))
        finish_x = max(0, min(finish_x, size_x - 1))
        start_y = max(0, min(start_y, size_y - 1))
        finish_y = max(0, min(finish_y, size_y - 1))
        node.start_x = start_x
        node.finish_x = finish_x
        node.start_y = start_y
        node.finish_y = finish_y
        for grid_x in range(start_x, finish_x + 1):
            for grid_y in range(start_y, finish_y + 1):
                grid[grid_x][grid_y].append(node)
    state.grid = grid


def _grow_tree(state: CoSECompoundState) -> None:
    """Port cose-base ``growTree`` for one pruned step.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Restores one pruned tree shell.
    """
    if not state.pruned_nodes_all:
        return
    pruned_step = state.pruned_nodes_all[-1]
    for node, edge, owner in pruned_step:
        _find_place_for_pruned_node(state, node, edge)
        _add_node(owner, node)
        _restore_pruned_edge(edge)
    state.pruned_nodes_all.pop()


def _find_place_for_pruned_node(state: CoSECompoundState, pruned_node: _Node, edge: _Edge) -> None:
    """Port cose-base ``findPlaceforPrunedNode``.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state carrying grid and LCG.
    pruned_node : _Node
        Node to restore.
    edge : _Edge
        Edge connecting restored node to the remaining graph.

    Returns
    -------
    None
        Mutates pruned node position.
    """
    node_to_connect = edge.target if pruned_node is edge.source else edge.source
    if not state.grid:
        _set_center(
            pruned_node,
            _center_x(node_to_connect) + _DEFAULT_EDGE_LENGTH + pruned_node.rect.width,
            _center_y(node_to_connect),
        )
        return
    control = [0, 0, 0, 0]
    if node_to_connect.start_y > 0:
        for grid_x in range(node_to_connect.start_x, node_to_connect.finish_x + 1):
            control[0] += (
                len(state.grid[grid_x][node_to_connect.start_y - 1])
                + len(state.grid[grid_x][node_to_connect.start_y])
                - 1
            )
    if node_to_connect.finish_x < len(state.grid) - 1:
        for grid_y in range(node_to_connect.start_y, node_to_connect.finish_y + 1):
            control[1] += (
                len(state.grid[node_to_connect.finish_x + 1][grid_y])
                + len(state.grid[node_to_connect.finish_x][grid_y])
                - 1
            )
    if node_to_connect.finish_y < len(state.grid[0]) - 1:
        for grid_x in range(node_to_connect.start_x, node_to_connect.finish_x + 1):
            control[2] += (
                len(state.grid[grid_x][node_to_connect.finish_y + 1])
                + len(state.grid[grid_x][node_to_connect.finish_y])
                - 1
            )
    if node_to_connect.start_x > 0:
        for grid_y in range(node_to_connect.start_y, node_to_connect.finish_y + 1):
            control[3] += (
                len(state.grid[node_to_connect.start_x - 1][grid_y])
                + len(state.grid[node_to_connect.start_x][grid_y])
                - 1
            )
    minimum = min(control)
    min_indices = [index for index, value in enumerate(control) if value == minimum]
    grid_for_pruned = min_indices[0]
    if minimum == 0 and len(min_indices) == 2:
        state.random_state, random_value = _next_cytoscape_random(state.random_state)
        grid_for_pruned = min_indices[int(math.floor(random_value * 2.0))]
    elif minimum == 0 and len(min_indices) == 4:
        state.random_state, random_value = _next_cytoscape_random(state.random_state)
        grid_for_pruned = int(math.floor(random_value * 4.0))
    if grid_for_pruned == 0:
        _set_center(
            pruned_node,
            _center_x(node_to_connect),
            _center_y(node_to_connect)
            - node_to_connect.rect.height / 2.0
            - _DEFAULT_EDGE_LENGTH
            - pruned_node.rect.height / 2.0,
        )
    elif grid_for_pruned == 1:
        _set_center(
            pruned_node,
            _center_x(node_to_connect)
            + node_to_connect.rect.width / 2.0
            + _DEFAULT_EDGE_LENGTH
            + pruned_node.rect.width / 2.0,
            _center_y(node_to_connect),
        )
    elif grid_for_pruned == 2:
        _set_center(
            pruned_node,
            _center_x(node_to_connect),
            _center_y(node_to_connect)
            + node_to_connect.rect.height / 2.0
            + _DEFAULT_EDGE_LENGTH
            + pruned_node.rect.height / 2.0,
        )
    else:
        _set_center(
            pruned_node,
            _center_x(node_to_connect)
            - node_to_connect.rect.width / 2.0
            - _DEFAULT_EDGE_LENGTH
            - pruned_node.rect.width / 2.0,
            _center_y(node_to_connect),
        )


def _rects_intersect(a: _Rect, b: _Rect) -> bool:
    """Return whether two rectangles intersect.

    Parameters
    ----------
    a : _Rect
        First rectangle.
    b : _Rect
        Second rectangle.

    Returns
    -------
    bool
        ``True`` when rectangles overlap or touch.
    """
    return not (
        a.x + a.width < b.x or b.x + b.width < a.x or a.y + a.height < b.y or b.y + b.height < a.y
    )


def _clip_delta(source: _Node, target: _Node) -> Tuple[float, float, bool]:
    """Return approximate clipped vector between node rectangles.

    Parameters
    ----------
    source : _Node
        Source node.
    target : _Node
        Target node.

    Returns
    -------
    tuple[float, float, bool]
        Delta from source to target after clipping and overlap flag.
    """
    result, overlaps = _intersection_points(source.rect, target.rect)
    if overlaps:
        return _center_x(target) - _center_x(source), _center_y(target) - _center_y(source), True
    # layout-base updateLength(rect target, rect source) stores sourceClip - targetClip.
    clipped_x = result[2] - result[0]
    clipped_y = result[3] - result[1]
    if abs(clipped_x) < 1.0:
        clipped_x = _sign(clipped_x)
    if abs(clipped_y) < 1.0:
        clipped_y = _sign(clipped_y)
    return clipped_x, clipped_y, False


def _intersection_points(rect_a: _Rect, rect_b: _Rect) -> Tuple[List[float], bool]:
    """Return layout-base ``IGeometry.getIntersection2`` clipping points.

    Parameters
    ----------
    rect_a : _Rect
        First rectangle.
    rect_b : _Rect
        Second rectangle.

    Returns
    -------
    tuple[list[float], bool]
        ``[a_x, a_y, b_x, b_y]`` and overlap flag.
    """
    p1x = rect_a.x + rect_a.width / 2.0
    p1y = rect_a.y + rect_a.height / 2.0
    p2x = rect_b.x + rect_b.width / 2.0
    p2y = rect_b.y + rect_b.height / 2.0
    if _rects_intersect(rect_a, rect_b):
        return [p1x, p1y, p2x, p2y], True
    top_left_ax = rect_a.x
    top_left_ay = rect_a.y
    top_right_ax = rect_a.x + rect_a.width
    bottom_left_ax = rect_a.x
    bottom_left_ay = rect_a.y + rect_a.height
    bottom_right_ax = rect_a.x + rect_a.width
    half_width_a = rect_a.width / 2.0
    half_height_a = rect_a.height / 2.0
    top_left_bx = rect_b.x
    top_left_by = rect_b.y
    top_right_bx = rect_b.x + rect_b.width
    bottom_left_bx = rect_b.x
    bottom_left_by = rect_b.y + rect_b.height
    bottom_right_bx = rect_b.x + rect_b.width
    half_width_b = rect_b.width / 2.0
    half_height_b = rect_b.height / 2.0
    result = [0.0, 0.0, 0.0, 0.0]
    if p1x == p2x:
        if p1y > p2y:
            return [p1x, top_left_ay, p2x, bottom_left_by], False
        if p1y < p2y:
            return [p1x, bottom_left_ay, p2x, top_left_by], False
        return result, False
    if p1y == p2y:
        if p1x > p2x:
            return [top_left_ax, p1y, top_right_bx, p2y], False
        if p1x < p2x:
            return [top_right_ax, p1y, top_left_bx, p2y], False
        return result, False

    slope_a = rect_a.height / rect_a.width
    slope_b = rect_b.height / rect_b.width
    slope_prime = (p2y - p1y) / (p2x - p1x)
    clip_a_found = False
    clip_b_found = False
    if -slope_a == slope_prime:
        if p1x > p2x:
            result[0] = bottom_left_ax
            result[1] = bottom_left_ay
        else:
            result[0] = top_right_ax
            result[1] = top_left_ay
        clip_a_found = True
    elif slope_a == slope_prime:
        if p1x > p2x:
            result[0] = top_left_ax
            result[1] = top_left_ay
        else:
            result[0] = bottom_right_ax
            result[1] = bottom_left_ay
        clip_a_found = True
    if -slope_b == slope_prime:
        if p2x > p1x:
            result[2] = bottom_left_bx
            result[3] = bottom_left_by
        else:
            result[2] = top_right_bx
            result[3] = top_left_by
        clip_b_found = True
    elif slope_b == slope_prime:
        if p2x > p1x:
            result[2] = top_left_bx
            result[3] = top_left_by
        else:
            result[2] = bottom_right_bx
            result[3] = bottom_left_by
        clip_b_found = True
    if clip_a_found and clip_b_found:
        return result, False
    if p1x > p2x:
        if p1y > p2y:
            cardinal_a = _cardinal_direction(slope_a, slope_prime, 4)
            cardinal_b = _cardinal_direction(slope_b, slope_prime, 2)
        else:
            cardinal_a = _cardinal_direction(-slope_a, slope_prime, 3)
            cardinal_b = _cardinal_direction(-slope_b, slope_prime, 1)
    elif p1y > p2y:
        cardinal_a = _cardinal_direction(-slope_a, slope_prime, 1)
        cardinal_b = _cardinal_direction(-slope_b, slope_prime, 3)
    else:
        cardinal_a = _cardinal_direction(slope_a, slope_prime, 2)
        cardinal_b = _cardinal_direction(slope_b, slope_prime, 4)
    if not clip_a_found:
        result[0], result[1] = _clip_point_for_cardinal(
            cardinal_a,
            p1x,
            p1y,
            top_left_ax,
            top_left_ay,
            bottom_left_ax,
            bottom_left_ay,
            bottom_right_ax,
            half_width_a,
            half_height_a,
            slope_prime,
        )
    if not clip_b_found:
        result[2], result[3] = _clip_point_for_cardinal(
            cardinal_b,
            p2x,
            p2y,
            top_left_bx,
            top_left_by,
            bottom_left_bx,
            bottom_left_by,
            bottom_right_bx,
            half_width_b,
            half_height_b,
            slope_prime,
        )
    return result, False


def _cardinal_direction(slope: float, slope_prime: float, line: int) -> int:
    """Return layout-base cardinal direction for clipping.

    Parameters
    ----------
    slope : float
        Rectangle diagonal slope.
    slope_prime : float
        Center-line slope.
    line : int
        Reference line index.

    Returns
    -------
    int
        Cardinal direction, where 1=N, 2=E, 3=S, 4=W.
    """
    return line if slope > slope_prime else 1 + line % 4


def _clip_point_for_cardinal(
    cardinal: int,
    center_x: float,
    center_y: float,
    top_left_x: float,
    top_left_y: float,
    bottom_left_x: float,
    bottom_left_y: float,
    bottom_right_x: float,
    half_width: float,
    half_height: float,
    slope_prime: float,
) -> Tuple[float, float]:
    """Return a rectangle clip point for one cardinal direction.

    Parameters
    ----------
    cardinal : int
        Cardinal direction.
    center_x : float
        Rectangle center x-coordinate.
    center_y : float
        Rectangle center y-coordinate.
    top_left_x : float
        Rectangle left x-coordinate.
    top_left_y : float
        Rectangle top y-coordinate.
    bottom_left_x : float
        Rectangle left x-coordinate.
    bottom_left_y : float
        Rectangle bottom y-coordinate.
    bottom_right_x : float
        Rectangle right x-coordinate.
    half_width : float
        Rectangle half width.
    half_height : float
        Rectangle half height.
    slope_prime : float
        Center-line slope.

    Returns
    -------
    tuple[float, float]
        Clip point.
    """
    if cardinal == 1:
        return center_x + (-half_height) / slope_prime, top_left_y
    if cardinal == 2:
        return bottom_right_x, center_y + half_width * slope_prime
    if cardinal == 3:
        return center_x + half_height / slope_prime, bottom_left_y
    return bottom_left_x, center_y + (-half_width) * slope_prime


def _calc_separation_amount(a: _Rect, b: _Rect) -> Tuple[float, float]:
    """Return an overlap separation vector matching layout-base intent.

    Parameters
    ----------
    a : _Rect
        First rectangle.
    b : _Rect
        Second rectangle.

    Returns
    -------
    tuple[float, float]
        X and Y separation amounts.
    """
    direction_x = -1.0 if a.x + a.width / 2.0 < b.x + b.width / 2.0 else 1.0
    direction_y = -1.0 if a.y + a.height / 2.0 < b.y + b.height / 2.0 else 1.0
    overlap_x = min(a.x + a.width, b.x + b.width) - max(a.x, b.x)
    overlap_y = min(a.y + a.height, b.y + b.height) - max(a.y, b.y)
    if a.x <= b.x and a.x + a.width >= b.x + b.width:
        overlap_x += min(b.x - a.x, a.x + a.width - (b.x + b.width))
    elif b.x <= a.x and b.x + b.width >= a.x + a.width:
        overlap_x += min(a.x - b.x, b.x + b.width - (a.x + a.width))
    if a.y <= b.y and a.y + a.height >= b.y + b.height:
        overlap_y += min(b.y - a.y, a.y + a.height - (b.y + b.height))
    elif b.y <= a.y and b.y + b.height >= a.y + a.height:
        overlap_y += min(a.y - b.y, b.y + b.height - (a.y + a.height))
    center_dx = (b.x + b.width / 2.0) - (a.x + a.width / 2.0)
    center_dy = (b.y + b.height / 2.0) - (a.y + a.height / 2.0)
    slope = abs(center_dy / center_dx) if center_dx != 0.0 else math.inf
    if center_dx == 0.0 and center_dy == 0.0:
        slope = 1.0
    move_by_y = slope * overlap_x
    move_by_x = overlap_y / slope if slope != 0.0 else math.inf
    if overlap_x < move_by_x:
        move_by_x = overlap_x
    else:
        move_by_y = overlap_y
    separation_buffer = _DEFAULT_EDGE_LENGTH / 2.0
    return (
        -direction_x * ((move_by_x / 2.0) + separation_buffer),
        -direction_y * ((move_by_y / 2.0) + separation_buffer),
    )


def _calc_spring_force(state: CoSECompoundState, edge: _Edge) -> None:
    """Apply cose-base spring force for one edge.

    Parameters
    ----------
    state : CoSECompoundState
        Compound state with spring options.
    edge : _Edge
        Edge to process.

    Returns
    -------
    None
        Mutates endpoint spring forces.
    """
    source = edge.source
    target = edge.target
    length_x, length_y, overlaps = _clip_delta(source, target)
    edge.is_overlapping_source_and_target = overlaps
    if overlaps:
        return
    edge.length_x = length_x
    edge.length_y = length_y
    edge.length = math.hypot(length_x, length_y)
    if edge.length == 0.0:
        return
    force_scalar = state.options.edge_elasticity * (edge.length - edge.ideal_length)
    source.spring_force_x += force_scalar * (edge.length_x / edge.length)
    source.spring_force_y += force_scalar * (edge.length_y / edge.length)
    target.spring_force_x -= force_scalar * (edge.length_x / edge.length)
    target.spring_force_y -= force_scalar * (edge.length_y / edge.length)


def _calc_repulsion_force(state: CoSECompoundState, node_a: _Node, node_b: _Node) -> None:
    """Apply cose-base sibling-scoped repulsion force.

    Parameters
    ----------
    state : CoSECompoundState
        Compound state.
    node_a : _Node
        First node.
    node_b : _Node
        Second node.

    Returns
    -------
    None
        Mutates node repulsion forces.
    """
    if node_a.owner is not node_b.owner:
        return
    if _rects_intersect(node_a.rect, node_b.rect):
        sep_x, sep_y = _calc_separation_amount(node_a.rect, node_b.rect)
        children_constant = (
            node_a.no_of_children
            * node_b.no_of_children
            / max(node_a.no_of_children + node_b.no_of_children, 1)
        )
        node_a.repulsion_force_x -= children_constant * 2.0 * sep_x
        node_a.repulsion_force_y -= children_constant * 2.0 * sep_y
        node_b.repulsion_force_x += children_constant * 2.0 * sep_x
        node_b.repulsion_force_y += children_constant * 2.0 * sep_y
        return
    distance_x, distance_y, _overlaps = _clip_delta(node_a, node_b)
    if abs(distance_x) < _MIN_REPULSION_DIST:
        distance_x = _sign(distance_x) * _MIN_REPULSION_DIST
    if abs(distance_y) < _MIN_REPULSION_DIST:
        distance_y = _sign(distance_y) * _MIN_REPULSION_DIST
    distance_squared = distance_x * distance_x + distance_y * distance_y
    distance = math.sqrt(distance_squared)
    repulsion = (
        state.options.node_repulsion
        * node_a.no_of_children
        * node_b.no_of_children
        / distance_squared
    )
    force_x = repulsion * distance_x / distance
    force_y = repulsion * distance_y / distance
    node_a.repulsion_force_x -= force_x
    node_a.repulsion_force_y -= force_y
    node_b.repulsion_force_x += force_x
    node_b.repulsion_force_y += force_y


def _calc_gravitational_force(state: CoSECompoundState, node: _Node) -> None:
    """Apply the exact cose-base compound gravity formula.

    Parameters
    ----------
    state : CoSECompoundState
        Compound state.
    node : _Node
        Node to apply gravity to.

    Returns
    -------
    None
        Mutates node gravity forces.
    """
    owner = node.owner
    if owner is None:
        return
    owner_center_x = (owner.right + owner.left) / 2.0
    owner_center_y = (owner.top + owner.bottom) / 2.0
    distance_x = _center_x(node) - owner_center_x
    distance_y = _center_y(node) - owner_center_y
    abs_distance_x = abs(distance_x) + node.rect.width / 2.0
    abs_distance_y = abs(distance_y) + node.rect.height / 2.0
    if owner is state.root:
        estimated_size = owner.estimated_size * state.options.gravity_range
        if abs_distance_x > estimated_size or abs_distance_y > estimated_size:
            node.gravitation_force_x = -state.options.gravity * distance_x
            node.gravitation_force_y = -state.options.gravity * distance_y
    else:
        estimated_size = owner.estimated_size * state.options.gravity_range_compound
        if abs_distance_x > estimated_size or abs_distance_y > estimated_size:
            node.gravitation_force_x = (
                -state.options.gravity * distance_x * state.options.gravity_compound
            )
            node.gravitation_force_y = (
                -state.options.gravity * distance_y * state.options.gravity_compound
            )


def _move_node(state: CoSECompoundState, node: _Node) -> None:
    """Move one node by the accumulated CoSE forces.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state carrying cooling parameters.
    node : _Node
        Node to move.

    Returns
    -------
    None
        Mutates node and resets force accumulators.
    """
    node.displacement_x = (
        state.cooling_factor
        * (node.spring_force_x + node.repulsion_force_x + node.gravitation_force_x)
        / max(node.no_of_children, 1)
    )
    node.displacement_y = (
        state.cooling_factor
        * (node.spring_force_y + node.repulsion_force_y + node.gravitation_force_y)
        / max(node.no_of_children, 1)
    )
    max_displacement = state.cooling_factor * _MAX_NODE_DISPLACEMENT
    if abs(node.displacement_x) > max_displacement:
        node.displacement_x = max_displacement * _sign(node.displacement_x)
    if abs(node.displacement_y) > max_displacement:
        node.displacement_y = max_displacement * _sign(node.displacement_y)
    if node.child is None or len(node.child.nodes) == 0:
        _move_by(node, node.displacement_x, node.displacement_y)
    else:
        _propagate_displacement_to_children(node, node.displacement_x, node.displacement_y)
    state.total_displacement += abs(node.displacement_x) + abs(node.displacement_y)
    node.spring_force_x = 0.0
    node.spring_force_y = 0.0
    node.repulsion_force_x = 0.0
    node.repulsion_force_y = 0.0
    node.gravitation_force_x = 0.0
    node.gravitation_force_y = 0.0
    node.displacement_x = 0.0
    node.displacement_y = 0.0


def _propagate_displacement_to_children(node: _Node, dx: float, dy: float) -> None:
    """Propagate compound movement to descendant leaves.

    Parameters
    ----------
    node : _Node
        Compound node.
    dx : float
        X displacement.
    dy : float
        Y displacement.

    Returns
    -------
    None
        Mutates descendant positions.
    """
    if node.child is None:
        return
    for child in node.child.nodes:
        if child.child is None:
            _move_by(child, dx, dy)
            child.displacement_x += dx
            child.displacement_y += dy
        else:
            _propagate_displacement_to_children(child, dx, dy)


def _calc_spring_forces(state: CoSECompoundState) -> None:
    """Apply spring forces for all edges.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates edge endpoint forces.
    """
    for edge in _all_edges(state):
        _calc_spring_force(state, edge)


def _calc_repulsion_forces(state: CoSECompoundState) -> None:
    """Apply sibling-scoped repulsion forces.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates node forces.
    """
    for graph in state.graphs:
        nodes = graph.nodes
        for i, node_a in enumerate(nodes):
            for node_b in nodes[i + 1 :]:
                _calc_repulsion_force(state, node_a, node_b)


def _calc_gravitational_forces(state: CoSECompoundState) -> None:
    """Apply gravity to the reference-selected node subset.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates node forces.
    """
    for node in state.nodes_with_gravity:
        if node.owner is not None:
            _calc_gravitational_force(state, node)


def _move_nodes(state: CoSECompoundState) -> None:
    """Move all nodes for one tick.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates node positions.
    """
    for node in _all_nodes(state):
        _move_node(state, node)


def _is_converged(state: CoSECompoundState) -> bool:
    """Return cose-base convergence status.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    bool
        Whether total displacement has converged or oscillated.
    """
    oscillating = False
    if state.total_iterations > state.max_iterations / 3.0:
        oscillating = abs(state.total_displacement - state.old_total_displacement) < 2.0
    converged = state.total_displacement < state.total_displacement_threshold
    state.old_total_displacement = state.total_displacement
    return converged or oscillating


def _init_spring_embedder(state: CoSECompoundState) -> None:
    """Initialize spring-embedder cooling and thresholds.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates cooling fields.
    """
    node_count = len(_all_nodes(state))
    state.cooling_factor = 1.0
    state.initial_cooling_factor = state.cooling_factor
    state.max_iterations = max(node_count * 5, int(state.options.steps))
    state.total_displacement_threshold = _DISPLACEMENT_THRESHOLD_PER_NODE * node_count
    state.repulsion_range = 2.0 * state.options.ideal_edge_length
    state.cooling_cycle = 0
    state.max_cooling_cycle = state.max_iterations / _CONVERGENCE_CHECK_PERIOD
    state.final_temperature = _CONVERGENCE_CHECK_PERIOD / max(state.max_iterations, 1)
    state.cooling_adjuster = 1.0


def _tick(state: CoSECompoundState) -> bool:
    """Run one cose-base spring-embedder tick.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    bool
        ``True`` when layout should stop.
    """
    state.total_iterations += 1
    if (
        state.total_iterations == state.max_iterations
        and not state.is_tree_growing
        and not state.is_growth_finished
    ):
        if state.pruned_nodes_all:
            state.is_tree_growing = True
        else:
            return True
    if (
        state.total_iterations % _CONVERGENCE_CHECK_PERIOD == 0
        and not state.is_tree_growing
        and not state.is_growth_finished
    ):
        if _is_converged(state):
            if state.pruned_nodes_all:
                state.is_tree_growing = True
            else:
                return True
        state.cooling_cycle += 1
        if state.options.quality == "draft":
            state.cooling_adjuster = float(state.cooling_cycle)
        elif state.options.quality == "default":
            state.cooling_adjuster = float(state.cooling_cycle) / 3.0
        exponent = math.log(
            100.0 * (state.initial_cooling_factor - state.final_temperature)
        ) / math.log(max(state.max_cooling_cycle, 1.0000001))
        state.cooling_factor = max(
            state.initial_cooling_factor
            - (math.pow(state.cooling_cycle, exponent) / 100.0) * state.cooling_adjuster,
            state.final_temperature,
        )
    if state.is_tree_growing:
        if state.grow_tree_iterations % 10 == 0:
            if state.pruned_nodes_all:
                _update_graph_bounds(state.root, True)
                _update_grid(state)
                _grow_tree(state)
                remaining = set(_all_nodes(state))
                state.nodes_with_gravity = [
                    node for node in state.nodes_with_gravity if node in remaining
                ]
                _update_graph_bounds(state.root, True)
                _update_grid(state)
                state.cooling_factor = _DEFAULT_COOLING_FACTOR_INCREMENTAL
            else:
                state.is_tree_growing = False
                state.is_growth_finished = True
        state.grow_tree_iterations += 1
    if state.is_growth_finished:
        if _is_converged(state):
            return True
        if state.after_growth_iterations % 10 == 0:
            _update_graph_bounds(state.root, True)
            _update_grid(state)
        state.cooling_factor = _DEFAULT_COOLING_FACTOR_INCREMENTAL * (
            (100.0 - state.after_growth_iterations) / 100.0
        )
        state.after_growth_iterations += 1
    state.total_displacement = 0.0
    _update_graph_bounds(state.root, True)
    _calc_estimated_size_graph(state.root)
    _calc_spring_forces(state)
    _calc_repulsion_forces(state)
    _calc_gravitational_forces(state)
    _move_nodes(state)
    return False


def _run_spring_embedder(state: CoSECompoundState) -> None:
    """Run ticks until cose-base stop criteria are met.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates positions.
    """
    layout_ended = False
    guard = state.max_iterations + 1000 + 10 * sum(len(step) for step in state.pruned_nodes_all)
    ticks = 0
    while not layout_ended and ticks < guard:
        layout_ended = _tick(state)
        ticks += 1
    _update_graph_bounds(state.root, True)


def _classic_layout(state: CoSECompoundState) -> None:
    """Run the cose-base ``classicLayout`` sequence.

    Parameters
    ----------
    state : CoSECompoundState
        Mutable state.

    Returns
    -------
    None
        Mutates positions.
    """
    state.nodes_with_gravity = _calculate_nodes_to_apply_gravitation_to(state)
    _calc_no_of_children_all(state)
    _calc_lowest_common_ancestors(state)
    _calc_inclusion_depths(state)
    _calc_estimated_size_graph(state.root)
    _calc_ideal_edge_lengths(state)
    if state.options.randomize:
        forest = _get_flat_forest(state.root)
        if forest:
            _position_nodes_radially(state, forest)
        else:
            _reduce_trees(state)
            remaining = set(_all_nodes(state))
            state.nodes_with_gravity = [
                node for node in state.nodes_with_gravity if node in remaining
            ]
            _position_nodes_randomly(state, state.root)
    _init_spring_embedder(state)
    _run_spring_embedder(state)


def _finalize_positions(state: CoSECompoundState, num_nodes: int) -> torch.Tensor:
    """Return leaf center positions as a tensor.

    Parameters
    ----------
    state : CoSECompoundState
        Compound state after layout.
    num_nodes : int
        Number of leaf nodes.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    _transform_to(state, 0.0, 0.0)
    _tiling_post_layout(state)
    result = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node_index, node in state.nodes_by_leaf.items():
        result[node_index, 0] = _center_x(node)
        result[node_index, 1] = _center_y(node)
    if result.numel() > 0:
        result = result - result.mean(dim=0, keepdim=True)
    return result


def layout_cose_base_compound(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    clusters: Optional[Mapping[str, Any]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    options: CoSECompoundOptions,
) -> torch.Tensor:
    """Run the shared compound-CoSE core.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of leaf nodes.
    node_sizes : torch.Tensor | None
        Optional node size tensor with shape ``[N, 2]``.
    clusters : mapping[str, Any] | None
        Cluster metadata.
    cluster_parents : mapping[str, str | None] | None
        Cluster hierarchy metadata.
    options : CoSECompoundOptions
        Runtime options.

    Returns
    -------
    torch.Tensor
        Leaf center positions with shape ``[N, 2]``.
    """
    if options.version not in {"1.0.3", "2.2.0"}:
        raise ValueError("version must be '1.0.3' or '2.2.0'.")
    if options.quality not in {"draft", "default", "proof"}:
        raise ValueError("quality must be one of 'draft', 'default', or 'proof'.")
    state = build_cose_compound_state(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        clusters=clusters,
        cluster_parents=cluster_parents,
        options=options,
    )
    _tiling_pre_layout(state)
    _classic_layout(state)
    return _finalize_positions(state, num_nodes)


__all__ = [
    "CoSECompoundOptions",
    "CoSECompoundState",
    "build_cose_compound_state",
    "layout_cose_base_compound",
]
