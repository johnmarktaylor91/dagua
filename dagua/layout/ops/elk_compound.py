"""Separate-children compound wrapper for the ELK layered pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_ELK_CHILD_DIRECTION = "RIGHT"
_ELK_CHILD_NODE_NODE_SPACING = 20.0
_ELK_CHILD_BETWEEN_LAYERS_SPACING = 20.0
_ELK_DEFAULT_PADDING = 12.0
_ELK_COMPONENT_COMPONENT_SPACING = 20.0
_ELK_LAYERED_ASPECT_RATIO = 1.6
_ELK_DEFAULT_CHILD_RANDOM_SEED = 1


@dataclass(frozen=True)
class _CompoundOptions:
    """ELK flat-pipeline options for one recursive layout run.

    Attributes
    ----------
    direction : str
        ELK direction for the current container.
    node_node_spacing : float
        Same-layer node spacing.
    between_layers_spacing : float
        Adjacent-layer spacing.
    cycle_breaking_strategy : str
        Cycle-breaking selector forwarded to the flat stages.
    layering_strategy : str
        Layer assignment selector forwarded to the flat stages.
    crossing_minimization_strategy : str
        Crossing minimization selector forwarded to the flat stages.
    node_placement_strategy : str
        Node placement selector forwarded to the flat stages.
    random_seed : int
        Java-compatible random seed for the flat run.
    thoroughness : int
        Number of ELK layer-sweep restarts.
    """

    direction: str
    node_node_spacing: float
    between_layers_spacing: float
    cycle_breaking_strategy: str
    layering_strategy: str
    crossing_minimization_strategy: str
    node_placement_strategy: str
    random_seed: int
    thoroughness: int


@dataclass(frozen=True)
class _ClusterTree:
    """Cluster membership and child order matching the elkjs adapter.

    Attributes
    ----------
    children_by_parent : dict[str | None, list[str]]
        Sorted cluster children keyed by parent cluster name.
    direct_members : dict[str | None, list[int]]
        Direct leaf nodes for each container. ``None`` stores root-level loose
        nodes.
    node_container : dict[int, str | None]
        Immediate container for each emitted node.
    """

    children_by_parent: Dict[Optional[str], List[str]]
    direct_members: Dict[Optional[str], List[int]]
    node_container: Dict[int, Optional[str]]


@dataclass(frozen=True)
class _Item:
    """One local graph item in a container layout.

    Attributes
    ----------
    key : str
        Stable item identifier.
    node_index : int | None
        Original node index for leaf nodes.
    cluster_name : str | None
        Cluster name for rigid child containers.
    size : tuple[float, float]
        Width and height in ELK point units.
    """

    key: str
    node_index: Optional[int]
    cluster_name: Optional[str]
    size: Tuple[float, float]


@dataclass(frozen=True)
class _ContainerLayout:
    """Resolved layout for a cluster container.

    Attributes
    ----------
    size : tuple[float, float]
        Container box width and height.
    leaf_positions : dict[int, torch.Tensor]
        Leaf top-left positions relative to the container top-left.
    extras : dict[str, Any]
        Structural diagnostics for tests and verification harnesses.
    """

    size: Tuple[float, float]
    leaf_positions: Dict[int, torch.Tensor]
    extras: Dict[str, Any]


@dataclass(frozen=True)
class _Component:
    """Connected component in a local ELK graph.

    Attributes
    ----------
    local_indices : list[int]
        Local item indices in DFS discovery order.
    edges : list[tuple[int, int]]
        Component edges remapped to ``local_indices`` order.
    positions : torch.Tensor
        Component-local top-left positions after the flat pipeline.
    size : tuple[float, float]
        Component graph size including ELK padding.
    """

    local_indices: List[int]
    edges: List[Tuple[int, int]]
    positions: torch.Tensor
    size: Tuple[float, float]


def _flatten_cluster_members(members: Any) -> List[int]:
    """Return flattened integer node members from a cluster payload.

    Parameters
    ----------
    members : Any
        Cluster member payload. Lists/tuples/sets contain node indices, and
        dictionaries are recursively flattened through their values.

    Returns
    -------
    list[int]
        Flattened node indices in payload traversal order.
    """
    if isinstance(members, Mapping):
        flattened: List[int] = []
        for child_members in members.values():
            flattened.extend(_flatten_cluster_members(child_members))
        return flattened
    if isinstance(members, Iterable) and not isinstance(members, (str, bytes)):
        return [int(member) for member in members]
    return []


def _cluster_members(
    clusters: Mapping[str, Any],
    cluster_name: str,
    num_nodes: int,
) -> List[int]:
    """Return valid flattened members for one named cluster.

    Parameters
    ----------
    clusters : mapping[str, Any]
        Cluster membership mapping.
    cluster_name : str
        Cluster name to read.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[int]
        Member node indices in the stored order, filtered to graph bounds.
    """
    members = _flatten_cluster_members(clusters.get(cluster_name, []))
    return [node for node in members if 0 <= node < num_nodes]


def _build_cluster_tree(
    clusters: Mapping[str, Any],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    num_nodes: int,
) -> _ClusterTree:
    """Build elkjs-compatible cluster order and direct membership.

    Parameters
    ----------
    clusters : mapping[str, Any]
        Cluster membership mapping.
    cluster_parents : mapping[str, str | None] | None
        Optional parent mapping.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    _ClusterTree
        Cluster child order, direct members, and node containers.
    """
    parents = cluster_parents or {}
    children_by_parent: Dict[Optional[str], List[str]] = {}
    for cluster_name in sorted(clusters):
        parent = parents.get(cluster_name)
        if parent not in clusters:
            parent = None
        children_by_parent.setdefault(parent, []).append(cluster_name)

    direct_members: Dict[Optional[str], List[int]] = {}
    node_container: Dict[int, Optional[str]] = {}
    emitted_nodes: Set[int] = set()

    def visit(parent_name: Optional[str]) -> None:
        """Populate direct members in elkjs emission order.

        Parameters
        ----------
        parent_name : str | None
            Container whose children should be traversed.

        Returns
        -------
        None
            The surrounding ``direct_members`` and ``node_container`` maps are
            mutated in place.
        """
        for cluster_name in children_by_parent.get(parent_name, []):
            visit(cluster_name)
            descendant_members: Set[int] = set()
            for child_name in children_by_parent.get(cluster_name, []):
                descendant_members.update(_cluster_members(clusters, child_name, num_nodes))
            members: List[int] = []
            for node_index in _cluster_members(clusters, cluster_name, num_nodes):
                if node_index in descendant_members or node_index in emitted_nodes:
                    continue
                members.append(node_index)
                emitted_nodes.add(node_index)
                node_container[node_index] = cluster_name
            direct_members[cluster_name] = members

    visit(None)
    root_members: List[int] = []
    for node_index in range(num_nodes):
        if node_index in emitted_nodes:
            continue
        root_members.append(node_index)
        node_container[node_index] = None
    direct_members[None] = root_members
    return _ClusterTree(
        children_by_parent=children_by_parent,
        direct_members=direct_members,
        node_container=node_container,
    )


def _edge_tensor(edges: Sequence[Tuple[int, int]]) -> torch.Tensor:
    """Convert local edge pairs to an edge-index tensor.

    Parameters
    ----------
    edges : sequence[tuple[int, int]]
        Directed local edge pairs.

    Returns
    -------
    torch.Tensor
        Long edge-index tensor with shape ``[2, E]``.
    """
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _run_flat_pipeline(
    edges: Sequence[Tuple[int, int]],
    node_sizes: torch.Tensor,
    options: _CompoundOptions,
) -> torch.Tensor:
    """Run the existing five-stage ELK pipeline on a flat subproblem.

    Parameters
    ----------
    edges : sequence[tuple[int, int]]
        Directed local edge pairs.
    node_sizes : torch.Tensor
        Local node sizes with shape ``[N, 2]``.
    options : _CompoundOptions
        ELK flat-pipeline options.

    Returns
    -------
    torch.Tensor
        Top-left local positions with shape ``[N, 2]``.

    Raises
    ------
    RuntimeError
        If the flat ELK pipeline does not produce positions.
    """
    if options.direction in {"RIGHT", "LEFT"}:
        swapped_sizes = node_sizes[:, [1, 0]].contiguous()
        down_options = _CompoundOptions(
            direction="DOWN",
            node_node_spacing=options.node_node_spacing,
            between_layers_spacing=options.between_layers_spacing,
            cycle_breaking_strategy=options.cycle_breaking_strategy,
            layering_strategy=options.layering_strategy,
            crossing_minimization_strategy=options.crossing_minimization_strategy,
            node_placement_strategy=options.node_placement_strategy,
            random_seed=options.random_seed,
            thoroughness=options.thoroughness,
        )
        down_positions = _run_flat_pipeline(edges, swapped_sizes, down_options)
        positions = torch.stack((down_positions[:, 1], down_positions[:, 0]), dim=1)
        if options.direction == "LEFT" and positions.shape[0] > 0:
            right = positions[:, 0] + node_sizes[:, 0]
            max_right = float(right.max().item())
            positions[:, 0] = max_right - right + _ELK_DEFAULT_PADDING
        return positions
    if node_sizes.shape[0] == 0:
        return torch.empty((0, 2), dtype=torch.float64)
    from dagua.layout.ops.pipelines.elk import build_elk_pipeline

    problem = LayoutProblem(
        edge_index=_edge_tensor(edges),
        num_nodes=int(node_sizes.shape[0]),
        node_sizes=node_sizes,
        seed=options.random_seed,
    )
    state = build_elk_pipeline(
        direction=options.direction,
        node_node_spacing=options.node_node_spacing,
        between_layers_spacing=options.between_layers_spacing,
        cycle_breaking_strategy=options.cycle_breaking_strategy,
        layering_strategy=options.layering_strategy,
        crossing_minimization_strategy=options.crossing_minimization_strategy,
        node_placement_strategy=options.node_placement_strategy,
        random_seed=options.random_seed,
        thoroughness=options.thoroughness,
    ).apply(problem, SolveState(), RuntimeContext(plan=ExecutionPlan(device="cpu")))
    if state.pos is None:
        raise RuntimeError("ELK flat sublayout did not produce positions.")
    return state.pos


def _content_size(positions: torch.Tensor, sizes: torch.Tensor) -> Tuple[float, float]:
    """Return ELK graph content size excluding leading graph padding.

    Parameters
    ----------
    positions : torch.Tensor
        Top-left positions with shape ``[N, 2]``.
    sizes : torch.Tensor
        Box sizes with shape ``[N, 2]``.

    Returns
    -------
    tuple[float, float]
        Width and height used by ``SimpleRowGraphPlacer``.
    """
    if positions.shape[0] == 0:
        return (0.0, 0.0)
    extents = positions + sizes
    return (
        max(0.0, float(extents[:, 0].max().item()) - _ELK_DEFAULT_PADDING),
        max(0.0, float(extents[:, 1].max().item()) - _ELK_DEFAULT_PADDING),
    )


def _component_indices(
    num_items: int,
    edges: Sequence[Tuple[int, int]],
) -> List[List[int]]:
    """Split a local graph into ELK-style connected components.

    Parameters
    ----------
    num_items : int
        Number of local graph items.
    edges : sequence[tuple[int, int]]
        Directed local edge pairs.

    Returns
    -------
    list[list[int]]
        Components in DFS start order, with nodes in DFS discovery order.
    """
    adjacency: List[List[int]] = [[] for _ in range(num_items)]
    for source, target in edges:
        adjacency[source].append(target)
        adjacency[target].append(source)
    visited = [False] * num_items
    components: List[List[int]] = []
    for start in range(num_items):
        if visited[start]:
            continue
        stack = [start]
        visited[start] = True
        component: List[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in reversed(adjacency[node]):
                if not visited[neighbor]:
                    visited[neighbor] = True
                    stack.append(neighbor)
        components.append(component)
    return components


def _layout_component(
    component_indices: Sequence[int],
    edges: Sequence[Tuple[int, int]],
    item_sizes: torch.Tensor,
    options: _CompoundOptions,
) -> _Component:
    """Layout one connected component as an independent ELK graph.

    Parameters
    ----------
    component_indices : sequence[int]
        Local item indices in this component.
    edges : sequence[tuple[int, int]]
        Full local graph edges.
    item_sizes : torch.Tensor
        Full local item sizes with shape ``[N, 2]``.
    options : _CompoundOptions
        ELK flat-pipeline options.

    Returns
    -------
    _Component
        Component layout and geometry.
    """
    index_map = {old_index: new_index for new_index, old_index in enumerate(component_indices)}
    component_edges = [
        (index_map[source], index_map[target])
        for source, target in edges
        if source in index_map and target in index_map
    ]
    sizes = item_sizes[list(component_indices), :].to(dtype=torch.float64)
    positions = _run_flat_pipeline(component_edges, sizes, options)
    return _Component(
        local_indices=list(component_indices),
        edges=component_edges,
        positions=positions,
        size=_content_size(positions, sizes),
    )


def _place_components(
    components: Sequence[_Component],
) -> Tuple[Dict[int, torch.Tensor], Tuple[float, float]]:
    """Port ELK ``SimpleRowGraphPlacer`` for component packing.

    Parameters
    ----------
    components : sequence[_Component]
        Component layouts before packing.

    Returns
    -------
    tuple[dict[int, torch.Tensor], tuple[float, float]]
        Packed local positions keyed by original local item index, and the
        combined graph size.
    """
    if not components:
        return {}, (0.0, 0.0)
    if len(components) == 1:
        component = components[0]
        positions = {
            old_index: component.positions[new_index].clone()
            for new_index, old_index in enumerate(component.local_indices)
        }
        return positions, component.size

    # ComponentsProcessor chooses SimpleRowGraphPlacer here. With default
    # priority zero, Java sorts by area ascending and TimSort preserves ties.
    sorted_components = sorted(
        enumerate(components),
        key=lambda indexed: (indexed[1].size[0] * indexed[1].size[1], indexed[0]),
    )
    ordered_components = [component for _, component in sorted_components]
    max_row_width = 0.0
    total_area = 0.0
    for component in ordered_components:
        width, height = component.size
        max_row_width = max(max_row_width, width)
        total_area += width * height
    max_row_width = max(max_row_width, math.sqrt(total_area) * _ELK_LAYERED_ASPECT_RATIO)

    positions: Dict[int, torch.Tensor] = {}
    xpos = 0.0
    ypos = 0.0
    highest_box = 0.0
    broadest_row = _ELK_COMPONENT_COMPONENT_SPACING
    for component in ordered_components:
        width, height = component.size
        if xpos + width > max_row_width:
            xpos = 0.0
            ypos += highest_box + _ELK_COMPONENT_COMPONENT_SPACING
            highest_box = 0.0
        offset = torch.tensor([xpos, ypos], dtype=torch.float64)
        for new_index, old_index in enumerate(component.local_indices):
            positions[old_index] = component.positions[new_index] + offset
        broadest_row = max(broadest_row, xpos + width)
        highest_box = max(highest_box, height)
        xpos += width + _ELK_COMPONENT_COMPONENT_SPACING
    return positions, (broadest_row, ypos + highest_box)


def _layout_local_graph(
    items: Sequence[_Item],
    edges: Sequence[Tuple[int, int]],
    options: _CompoundOptions,
) -> Tuple[Dict[int, torch.Tensor], Tuple[float, float], Dict[str, Any]]:
    """Layout and component-pack one local ELK graph.

    Parameters
    ----------
    items : sequence[_Item]
        Local child containers and direct node items.
    edges : sequence[tuple[int, int]]
        Directed local edges between direct node items.
    options : _CompoundOptions
        ELK flat-pipeline options.

    Returns
    -------
    tuple[dict[int, torch.Tensor], tuple[float, float], dict[str, Any]]
        Packed positions keyed by local item index, graph size, and structural
        diagnostics.
    """
    if not items:
        return (
            {},
            (0.0, 0.0),
            {
                "component_count": 0,
                "component_order": [],
            },
        )
    item_sizes = torch.tensor([item.size for item in items], dtype=torch.float64)
    components = [
        _layout_component(indices, edges, item_sizes, options)
        for indices in _component_indices(len(items), edges)
    ]
    positions, graph_size = _place_components(components)
    return (
        positions,
        graph_size,
        {
            "component_count": len(components),
            "component_order": [
                [items[index].key for index in component.local_indices] for component in components
            ],
            "packed_order": [
                [items[index].key for index in component.local_indices]
                for _, component in sorted(
                    enumerate(components),
                    key=lambda indexed: (indexed[1].size[0] * indexed[1].size[1], indexed[0]),
                )
            ],
        },
    )


def _local_edges_for_container(
    edges: Sequence[Tuple[int, int]],
    node_container: Mapping[int, Optional[str]],
    container_name: Optional[str],
    local_index_by_node: Mapping[int, int],
) -> Tuple[List[Tuple[int, int]], List[Tuple[int, int]]]:
    """Filter edges that survive in one separate-children container.

    Parameters
    ----------
    edges : sequence[tuple[int, int]]
        Original graph edge pairs.
    node_container : mapping[int, str | None]
        Immediate node container by original node index.
    container_name : str | None
        Current container.
    local_index_by_node : mapping[int, int]
        Local item index for direct nodes in the container.

    Returns
    -------
    tuple[list[tuple[int, int]], list[tuple[int, int]]]
        Surviving local edges and dropped original edges touching this
        container boundary.
    """
    local_edges: List[Tuple[int, int]] = []
    dropped_edges: List[Tuple[int, int]] = []
    for source, target in edges:
        source_container = node_container.get(source)
        target_container = node_container.get(target)
        if source_container == container_name and target_container == container_name:
            if source in local_index_by_node and target in local_index_by_node and source != target:
                local_edges.append((local_index_by_node[source], local_index_by_node[target]))
            elif source_container != target_container:
                dropped_edges.append((source, target))
        elif container_name in (source_container, target_container):
            dropped_edges.append((source, target))
    return local_edges, dropped_edges


def _edge_pairs(edge_index: torch.Tensor, num_nodes: int) -> List[Tuple[int, int]]:
    """Return valid non-self edge pairs from an edge-index tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge index with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[tuple[int, int]]
        Original directed edge pairs.
    """
    if edge_index.numel() == 0:
        return []
    cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    pairs: List[Tuple[int, int]] = []
    for source, target in zip(cpu_edges[0].tolist(), cpu_edges[1].tolist()):
        source_int = int(source)
        target_int = int(target)
        if source_int != target_int and 0 <= source_int < num_nodes and 0 <= target_int < num_nodes:
            pairs.append((source_int, target_int))
    return pairs


def _child_options(parent_options: _CompoundOptions) -> _CompoundOptions:
    """Return ELK default options for nested separate-children containers.

    Parameters
    ----------
    parent_options : _CompoundOptions
        Root options, used only for shared strategy selectors.

    Returns
    -------
    _CompoundOptions
        Child options. Root spacing and direction are deliberately not
        inherited, matching elkjs 0.11.1 probes and ELK option defaults.
    """
    return _CompoundOptions(
        direction=_ELK_CHILD_DIRECTION,
        node_node_spacing=_ELK_CHILD_NODE_NODE_SPACING,
        between_layers_spacing=_ELK_CHILD_BETWEEN_LAYERS_SPACING,
        cycle_breaking_strategy=parent_options.cycle_breaking_strategy,
        layering_strategy=parent_options.layering_strategy,
        crossing_minimization_strategy=parent_options.crossing_minimization_strategy,
        node_placement_strategy=parent_options.node_placement_strategy,
        random_seed=_ELK_DEFAULT_CHILD_RANDOM_SEED,
        thoroughness=parent_options.thoroughness,
    )


def _layout_container(
    container_name: Optional[str],
    tree: _ClusterTree,
    edges: Sequence[Tuple[int, int]],
    node_sizes: torch.Tensor,
    child_layouts: Mapping[str, _ContainerLayout],
    options: _CompoundOptions,
) -> _ContainerLayout:
    """Layout one cluster or root container from its direct children.

    Parameters
    ----------
    container_name : str | None
        Cluster name, or ``None`` for the root graph.
    tree : _ClusterTree
        Cluster hierarchy and direct memberships.
    edges : sequence[tuple[int, int]]
        Original graph edge pairs.
    node_sizes : torch.Tensor
        Original node sizes with shape ``[N, 2]``.
    child_layouts : mapping[str, _ContainerLayout]
        Already resolved child cluster layouts.
    options : _CompoundOptions
        ELK flat-pipeline options for this container.

    Returns
    -------
    _ContainerLayout
        Resolved container geometry and leaf positions.
    """
    items: List[_Item] = []
    for child_name in tree.children_by_parent.get(container_name, []):
        child_layout = child_layouts[child_name]
        items.append(
            _Item(
                key=f"cluster_{child_name}",
                node_index=None,
                cluster_name=child_name,
                size=child_layout.size,
            )
        )
    for node_index in tree.direct_members.get(container_name, []):
        items.append(
            _Item(
                key=str(node_index),
                node_index=node_index,
                cluster_name=None,
                size=(
                    float(node_sizes[node_index, 0].item()),
                    float(node_sizes[node_index, 1].item()),
                ),
            )
        )
    local_index_by_node = {
        item.node_index: index for index, item in enumerate(items) if item.node_index is not None
    }
    local_edges, dropped_edges = _local_edges_for_container(
        edges,
        tree.node_container,
        container_name,
        local_index_by_node,
    )
    item_positions, graph_size, extras = _layout_local_graph(items, local_edges, options)
    leaf_positions: Dict[int, torch.Tensor] = {}
    for index, item in enumerate(items):
        item_position = item_positions.get(index, torch.zeros(2, dtype=torch.float64))
        if item.node_index is not None:
            leaf_positions[item.node_index] = item_position
            continue
        if item.cluster_name is None:
            continue
        child_layout = child_layouts[item.cluster_name]
        for node_index, child_position in child_layout.leaf_positions.items():
            leaf_positions[node_index] = item_position + child_position
    extras.update(
        {
            "items": [item.key for item in items],
            "local_edges": local_edges,
            "dropped_edges": dropped_edges,
            "size": graph_size,
        }
    )
    container_size = (
        (graph_size[0] + 2.0 * _ELK_DEFAULT_PADDING, graph_size[1] + 2.0 * _ELK_DEFAULT_PADDING)
        if container_name is not None
        else graph_size
    )
    return _ContainerLayout(size=container_size, leaf_positions=leaf_positions, extras=extras)


def _postorder_clusters(
    children_by_parent: Mapping[Optional[str], Sequence[str]],
    parent_name: Optional[str],
) -> List[str]:
    """Return cluster names in child-before-parent order.

    Parameters
    ----------
    children_by_parent : mapping[str | None, sequence[str]]
        Cluster child mapping.
    parent_name : str | None
        Parent to traverse.

    Returns
    -------
    list[str]
        Postorder cluster names.
    """
    ordered: List[str] = []
    for child_name in children_by_parent.get(parent_name, []):
        ordered.extend(_postorder_clusters(children_by_parent, child_name))
        ordered.append(child_name)
    return ordered


def layout_elk_compound(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    clusters: Mapping[str, Any],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    root_options: _CompoundOptions,
) -> torch.Tensor:
    """Run ELK's default separate-children compound wrapper.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None
        Node sizes with shape ``[N, 2]``. Missing sizes default to ELK's
        historical ``120x40`` placeholder.
    clusters : mapping[str, Any]
        Cluster membership mapping.
    cluster_parents : mapping[str, str | None] | None
        Optional cluster parent mapping.
    root_options : _CompoundOptions
        Root flat-pipeline options.

    Returns
    -------
    torch.Tensor
        Final top-left leaf positions with shape ``[N, 2]``.
    """
    positions, _ = layout_elk_compound_with_diagnostics(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        clusters=clusters,
        cluster_parents=cluster_parents,
        root_options=root_options,
    )
    return positions


def layout_elk_compound_with_diagnostics(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    clusters: Mapping[str, Any],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    root_options: _CompoundOptions,
) -> Tuple[torch.Tensor, Dict[str, Any]]:
    """Run the compound wrapper and return structural diagnostics.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed graph edges with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None
        Node sizes with shape ``[N, 2]``.
    clusters : mapping[str, Any]
        Cluster membership mapping.
    cluster_parents : mapping[str, str | None] | None
        Optional cluster parent mapping.
    root_options : _CompoundOptions
        Root flat-pipeline options.

    Returns
    -------
    tuple[torch.Tensor, dict[str, Any]]
        Final positions and wrapper metadata including cross-hierarchy dropped
        edges and component-packing summaries.
    """
    sizes = (
        node_sizes.detach().to(device="cpu", dtype=torch.float64)
        if node_sizes is not None
        else torch.tensor([[120.0, 40.0]] * num_nodes, dtype=torch.float64)
    )
    tree = _build_cluster_tree(clusters, cluster_parents, num_nodes)
    edges = _edge_pairs(edge_index, num_nodes)
    dropped_edges = [
        (source, target)
        for source, target in edges
        if tree.node_container.get(source) != tree.node_container.get(target)
    ]
    child_options = _child_options(root_options)
    layouts: Dict[str, _ContainerLayout] = {}
    for cluster_name in _postorder_clusters(tree.children_by_parent, None):
        layouts[cluster_name] = _layout_container(
            cluster_name,
            tree,
            edges,
            sizes,
            layouts,
            child_options,
        )
    root_layout = _layout_container(None, tree, edges, sizes, layouts, root_options)
    positions = torch.zeros((num_nodes, 2), dtype=torch.float64)
    for node_index in range(num_nodes):
        position = root_layout.leaf_positions.get(node_index)
        if position is not None:
            positions[node_index] = position
    diagnostics: Dict[str, Any] = {
        "dropped_edges": dropped_edges,
        "children_by_parent": tree.children_by_parent,
        "direct_members": tree.direct_members,
        "root": root_layout.extras,
        "clusters": {cluster_name: layout.extras for cluster_name, layout in layouts.items()},
    }
    return positions, diagnostics


@register_op
class ElkRecursiveCompound(Op):
    """Apply ELK separate-children recursion as one composable op."""

    name = "elk_recursive_compound"
    category = OpCategory.COORDINATE
    reads = ("extras",)
    writes = ("extras", "pos")

    def __init__(self, root_options: _CompoundOptions) -> None:
        """Store root options for the recursive wrapper.

        Parameters
        ----------
        root_options : _CompoundOptions
            Root flat-pipeline options.

        Returns
        -------
        None
            Options are stored on the op.
        """
        self.root_options = root_options

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run recursive compound layout and store diagnostics.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs with cluster metadata.
        state : SolveState
            Mutable state receiving positions and ``elk_compound`` diagnostics.
        ctx : RuntimeContext
            Runtime infrastructure; unused by this deterministic wrapper.

        Returns
        -------
        SolveState
            Updated state with ``pos`` and diagnostics.

        Raises
        ------
        ValueError
            If no cluster mapping is present on the problem.
        """
        del ctx
        if not problem.clusters:
            raise ValueError("ElkRecursiveCompound requires non-empty problem.clusters.")
        positions, diagnostics = layout_elk_compound_with_diagnostics(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            node_sizes=problem.node_sizes,
            clusters=problem.clusters,
            cluster_parents=problem.cluster_parents,
            root_options=self.root_options,
        )
        state.pos = positions
        state.extras["elk_compound"] = diagnostics
        return state


__all__ = ["ElkRecursiveCompound", "layout_elk_compound", "layout_elk_compound_with_diagnostics"]
