"""FM^3 multilevel force-directed layout pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.cluster_geometry import ClusterTree
from dagua.layout.ops.fmmm import (
    _FinalizeFMMMPositions,
    _InitializeCoarsestLevel,
    _InitializeFMMMState,
    _InitializeFMMMStateConfig,
    _RefineCoarsestLevel,
    _SingleLevelFallback,
    _UncoarsenLoop,
)
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory

_FDP_COMPOUND_EDGE_ATTACHMENTS_KEY = "fmmm_fdp_compound_edge_attachments"
_FDP_COMPOUND_CLUSTER_OBSTACLES_KEY = "fmmm_fdp_compound_cluster_obstacles"
_FDP_COMPOUND_NODE_OBSTACLES_KEY = "fmmm_fdp_compound_node_obstacles"
_FDP_EPSILON = 1.0e-9

_ObjectKey = Tuple[str, Union[int, str]]


@dataclass(frozen=True)
class _FdpObstacleBox:
    """Axis-aligned obstacle box used by Graphviz fdp compound routing.

    Parameters
    ----------
    key : tuple[str, int | str]
        Stable object identity, either ``("node", index)`` or
        ``("cluster", name)``.
    x_min : float
        Lower x coordinate after any Graphviz-style expansion.
    y_min : float
        Lower y coordinate after any Graphviz-style expansion.
    x_max : float
        Upper x coordinate after any Graphviz-style expansion.
    y_max : float
        Upper y coordinate after any Graphviz-style expansion.
    """

    key: _ObjectKey
    x_min: float
    y_min: float
    x_max: float
    y_max: float


@dataclass(frozen=True)
class _FdpCompoundEdgeAttachment:
    """Compound-edge attachment metadata for one fdp edge.

    Parameters
    ----------
    edge_id : int
        Column index in the input edge tensor.
    source : int
        Source node index.
    target : int
        Target node index.
    tail_point : tuple[float, float]
        Tail attachment point after cluster-boundary clipping.
    head_point : tuple[float, float]
        Head attachment point after cluster-boundary clipping.
    tail_cluster : str, optional
        Deepest source-side cluster boundary crossed by the edge, if any.
    head_cluster : str, optional
        Deepest target-side cluster boundary crossed by the edge, if any.
    obstacle_keys : tuple[tuple[str, int | str], ...]
        Obstacles selected by the port of Graphviz ``objectList``.
    polyline : tuple[tuple[float, float], ...]
        Current route seed. Graphviz pathplan consumes the same endpoints and
        obstacle set to produce a visibility path; Dagua records the seed for
        downstream fidelity routing.
    """

    edge_id: int
    source: int
    target: int
    tail_point: Tuple[float, float]
    head_point: Tuple[float, float]
    tail_cluster: Optional[str]
    head_cluster: Optional[str]
    obstacle_keys: Tuple[_ObjectKey, ...]
    polyline: Tuple[Tuple[float, float], ...]


def _fdp_obstacle_vertices(box: _FdpObstacleBox) -> Tuple[Tuple[float, float], ...]:
    """Return Graphviz ``makeClustObs`` rectangle vertices in source order.

    Parameters
    ----------
    box : _FdpObstacleBox
        Obstacle box to convert.

    Returns
    -------
    tuple[tuple[float, float], ...]
        Four vertices ordered as lower-left, upper-left, upper-right,
        lower-right, matching ``clusteredges.c``.
    """
    return (
        (box.x_min, box.y_min),
        (box.x_min, box.y_max),
        (box.x_max, box.y_max),
        (box.x_max, box.y_min),
    )


def _fdp_expand_box(
    key: _ObjectKey,
    bounds: Tuple[float, float, float, float],
    expand: Tuple[float, float],
    do_add: bool,
) -> _FdpObstacleBox:
    """Apply Graphviz ``expand_t`` semantics to an obstacle box.

    Parameters
    ----------
    key : tuple[str, int | str]
        Stable node or cluster object identity.
    bounds : tuple[float, float, float, float]
        Box bounds as ``(x_min, y_min, x_max, y_max)``.
    expand : tuple[float, float]
        Expansion values corresponding to Graphviz ``pm->x`` and ``pm->y``.
    do_add : bool
        When ``True``, expand additively. When ``False``, scale about the box
        center using Graphviz's multiplicative branch.

    Returns
    -------
    _FdpObstacleBox
        Expanded box.
    """
    x_min, y_min, x_max, y_max = bounds
    expand_x, expand_y = expand
    center_x = (x_max + x_min) / 2.0
    center_y = (y_max + y_min) / 2.0
    if do_add:
        return _FdpObstacleBox(
            key=key,
            x_min=x_min - expand_x,
            y_min=y_min - expand_y,
            x_max=x_max + expand_x,
            y_max=y_max + expand_y,
        )

    delta_x = expand_x - 1.0
    delta_y = expand_y - 1.0
    return _FdpObstacleBox(
        key=key,
        x_min=expand_x * x_min - delta_x * center_x,
        y_min=expand_y * y_min - delta_y * center_y,
        x_max=expand_x * x_max - delta_x * center_x,
        y_max=expand_y * y_max - delta_y * center_y,
    )


def _fdp_graph_parent(tree: ClusterTree, graph_name: Optional[str]) -> Optional[str]:
    """Return the parent graph for a cluster graph.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    graph_name : str, optional
        Cluster graph name, or ``None`` for the root graph.

    Returns
    -------
    str or None
        Parent cluster graph, or ``None`` for root.
    """
    if graph_name is None:
        return None
    return tree.parents[graph_name]


def _fdp_graph_level(tree: ClusterTree, graph_name: Optional[str]) -> int:
    """Return the Graphviz-style nesting level for a graph.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    graph_name : str, optional
        Cluster graph name, or ``None`` for root.

    Returns
    -------
    int
        Root level is ``0`` and each nested cluster increments by one.
    """
    if graph_name is None:
        return 0
    level = 1
    parent = tree.parents[graph_name]
    while parent is not None:
        level += 1
        parent = tree.parents[parent]
    return level


def _fdp_deepest_cluster_by_node(tree: ClusterTree, num_nodes: int) -> Dict[int, Optional[str]]:
    """Map each node to its deepest containing cluster graph.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    dict[int, str | None]
        Deepest cluster name for each node, or ``None`` for root-owned nodes.
    """
    result: Dict[int, Optional[str]] = {index: None for index in range(num_nodes)}
    result_levels: Dict[int, int] = {index: 0 for index in range(num_nodes)}
    for cluster_name in tree.top_down_order():
        level = _fdp_graph_level(tree, cluster_name)
        for node_index in tree.descendants_per_cluster[cluster_name]:
            if 0 <= node_index < num_nodes and level >= result_levels[int(node_index)]:
                result[int(node_index)] = cluster_name
                result_levels[int(node_index)] = level
    return result


def _fdp_node_boxes(
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    expand: Tuple[float, float],
    do_add: bool,
) -> Dict[int, _FdpObstacleBox]:
    """Build node obstacle boxes from final fdp coordinates.

    Parameters
    ----------
    pos : torch.Tensor
        Node centers with shape ``[N, 2]``.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``. Missing sizes are treated as zero
        extents because the FMMM layout API does not require labels or sizes.
    expand : tuple[float, float]
        Graphviz obstacle expansion values.
    do_add : bool
        Whether expansion is additive.

    Returns
    -------
    dict[int, _FdpObstacleBox]
        Expanded node obstacle boxes keyed by node index.
    """
    boxes: Dict[int, _FdpObstacleBox] = {}
    if node_sizes is None:
        sizes = torch.zeros_like(pos)
    else:
        sizes = node_sizes.to(device=pos.device, dtype=pos.dtype)
    for node_index in range(pos.shape[0]):
        half_width = float(sizes[node_index, 0].item()) / 2.0
        half_height = float(sizes[node_index, 1].item()) / 2.0
        x_center = float(pos[node_index, 0].item())
        y_center = float(pos[node_index, 1].item())
        boxes[node_index] = _fdp_expand_box(
            key=("node", node_index),
            bounds=(
                x_center - half_width,
                y_center - half_height,
                x_center + half_width,
                y_center + half_height,
            ),
            expand=expand,
            do_add=do_add,
        )
    return boxes


def _fdp_cluster_boxes(
    tree: ClusterTree,
    pos: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    expand: Tuple[float, float],
    do_add: bool,
) -> Dict[str, _FdpObstacleBox]:
    """Build expanded cluster obstacles matching ``makeClustObs``.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    pos : torch.Tensor
        Node centers with shape ``[N, 2]``.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    expand : tuple[float, float]
        Graphviz obstacle expansion values.
    do_add : bool
        Whether expansion is additive.

    Returns
    -------
    dict[str, _FdpObstacleBox]
        Expanded cluster boxes keyed by cluster name.
    """
    boxes: Dict[str, _FdpObstacleBox] = {}
    if node_sizes is None:
        sizes = torch.zeros_like(pos)
    else:
        sizes = node_sizes.to(device=pos.device, dtype=pos.dtype)
    for cluster_name in tree.top_down_order():
        members = [
            int(node_index)
            for node_index in tree.descendants_per_cluster[cluster_name]
            if 0 <= int(node_index) < pos.shape[0]
        ]
        if not members:
            continue
        member_index = torch.tensor(members, dtype=torch.long, device=pos.device)
        member_pos = pos.index_select(0, member_index)
        member_sizes = sizes.index_select(0, member_index)
        half_sizes = member_sizes / 2.0
        lower = member_pos - half_sizes
        upper = member_pos + half_sizes
        bounds = (
            float(lower[:, 0].min().item()),
            float(lower[:, 1].min().item()),
            float(upper[:, 0].max().item()),
            float(upper[:, 1].max().item()),
        )
        boxes[cluster_name] = _fdp_expand_box(
            key=("cluster", cluster_name),
            bounds=bounds,
            expand=expand,
            do_add=do_add,
        )
    return boxes


def _fdp_add_graph_objects(
    obstacles: List[_FdpObstacleBox],
    graph_name: Optional[str],
    tail_exclude: Optional[_ObjectKey],
    head_exclude: Optional[_ObjectKey],
    tree: ClusterTree,
    node_parent: Dict[int, Optional[str]],
    node_boxes: Dict[int, _FdpObstacleBox],
    cluster_boxes: Dict[str, _FdpObstacleBox],
) -> None:
    """Append direct node and child-cluster obstacles for one graph level.

    Parameters
    ----------
    obstacles : list[_FdpObstacleBox]
        Mutable obstacle list receiving direct objects.
    graph_name : str, optional
        Graph level whose objects are added; ``None`` denotes root.
    tail_exclude : tuple[str, int | str], optional
        Tail object or containing graph to exclude.
    head_exclude : tuple[str, int | str], optional
        Head object or containing graph to exclude.
    tree : ClusterTree
        Cluster hierarchy.
    node_parent : dict[int, str | None]
        Deepest graph owning each node.
    node_boxes : dict[int, _FdpObstacleBox]
        Node obstacle lookup.
    cluster_boxes : dict[str, _FdpObstacleBox]
        Cluster obstacle lookup.

    Returns
    -------
    None
        Obstacles are appended in Graphviz iteration order.
    """
    for node_index in sorted(node_boxes):
        key: _ObjectKey = ("node", node_index)
        if node_parent.get(node_index) == graph_name and key not in {tail_exclude, head_exclude}:
            obstacles.append(node_boxes[node_index])

    for cluster_name in tree.top_down_order():
        key = ("cluster", cluster_name)
        if (
            tree.parents[cluster_name] == graph_name
            and key not in {tail_exclude, head_exclude}
            and cluster_name in cluster_boxes
        ):
            obstacles.append(cluster_boxes[cluster_name])


def _fdp_raise_level(
    obstacles: List[_FdpObstacleBox],
    graph_name: Optional[str],
    max_level: int,
    exclude: _ObjectKey,
    min_level: int,
    tree: ClusterTree,
    node_parent: Dict[int, Optional[str]],
    node_boxes: Dict[int, _FdpObstacleBox],
    cluster_boxes: Dict[str, _FdpObstacleBox],
) -> Optional[str]:
    """Mirror Graphviz ``raiseLevel`` for an endpoint graph.

    Parameters
    ----------
    obstacles : list[_FdpObstacleBox]
        Mutable obstacle list.
    graph_name : str, optional
        Starting endpoint graph.
    max_level : int
        Starting graph level.
    exclude : tuple[str, int | str]
        Endpoint object or previous containing graph to exclude.
    min_level : int
        Target graph level.
    tree : ClusterTree
        Cluster hierarchy.
    node_parent : dict[int, str | None]
        Deepest graph owning each node.
    node_boxes : dict[int, _FdpObstacleBox]
        Node obstacle lookup.
    cluster_boxes : dict[str, _FdpObstacleBox]
        Cluster obstacle lookup.

    Returns
    -------
    str or None
        Last cluster graph processed, matching the C function's ``*gp`` value.
    """
    current_graph = graph_name
    current_exclude = exclude
    for _level in range(max_level, min_level, -1):
        _fdp_add_graph_objects(
            obstacles=obstacles,
            graph_name=current_graph,
            tail_exclude=current_exclude,
            head_exclude=None,
            tree=tree,
            node_parent=node_parent,
            node_boxes=node_boxes,
            cluster_boxes=cluster_boxes,
        )
        if current_graph is None:
            return None
        current_exclude = ("cluster", current_graph)
        current_graph = _fdp_graph_parent(tree, current_graph)
    if current_exclude[0] == "cluster":
        return str(current_exclude[1])
    return None


def _fdp_compound_obstacle_list(
    source: int,
    target: int,
    tree: ClusterTree,
    node_parent: Dict[int, Optional[str]],
    node_boxes: Dict[int, _FdpObstacleBox],
    cluster_boxes: Dict[str, _FdpObstacleBox],
) -> List[_FdpObstacleBox]:
    """Port Graphviz fdp ``objectList`` for one non-loop edge.

    Parameters
    ----------
    source : int
        Tail node index.
    target : int
        Head node index.
    tree : ClusterTree
        Cluster hierarchy.
    node_parent : dict[int, str | None]
        Deepest graph owning each node.
    node_boxes : dict[int, _FdpObstacleBox]
        Node obstacle lookup.
    cluster_boxes : dict[str, _FdpObstacleBox]
        Cluster obstacle lookup.

    Returns
    -------
    list[_FdpObstacleBox]
        Obstacle list in Graphviz traversal order, excluding endpoints and
        graphs containing endpoints.
    """
    obstacles: List[_FdpObstacleBox] = []
    head_graph = node_parent.get(target)
    tail_graph = node_parent.get(source)
    head_exclude: _ObjectKey = ("node", target)
    tail_exclude: _ObjectKey = ("node", source)

    head_level = _fdp_graph_level(tree, head_graph)
    tail_level = _fdp_graph_level(tree, tail_graph)
    if head_level > tail_level:
        raised = _fdp_raise_level(
            obstacles,
            head_graph,
            head_level,
            head_exclude,
            tail_level,
            tree,
            node_parent,
            node_boxes,
            cluster_boxes,
        )
        head_exclude = ("cluster", raised) if raised is not None else head_exclude
        head_graph = _fdp_graph_parent(tree, raised)
    elif tail_level > head_level:
        raised = _fdp_raise_level(
            obstacles,
            tail_graph,
            tail_level,
            tail_exclude,
            head_level,
            tree,
            node_parent,
            node_boxes,
            cluster_boxes,
        )
        tail_exclude = ("cluster", raised) if raised is not None else tail_exclude
        tail_graph = _fdp_graph_parent(tree, raised)

    while head_graph != tail_graph:
        _fdp_add_graph_objects(
            obstacles,
            head_graph,
            tail_exclude=None,
            head_exclude=head_exclude,
            tree=tree,
            node_parent=node_parent,
            node_boxes=node_boxes,
            cluster_boxes=cluster_boxes,
        )
        _fdp_add_graph_objects(
            obstacles,
            tail_graph,
            tail_exclude=tail_exclude,
            head_exclude=None,
            tree=tree,
            node_parent=node_parent,
            node_boxes=node_boxes,
            cluster_boxes=cluster_boxes,
        )
        if head_graph is not None:
            head_exclude = ("cluster", head_graph)
        head_graph = _fdp_graph_parent(tree, head_graph)
        if tail_graph is not None:
            tail_exclude = ("cluster", tail_graph)
        tail_graph = _fdp_graph_parent(tree, tail_graph)

    _fdp_add_graph_objects(
        obstacles,
        tail_graph,
        tail_exclude=tail_exclude,
        head_exclude=head_exclude,
        tree=tree,
        node_parent=node_parent,
        node_boxes=node_boxes,
        cluster_boxes=cluster_boxes,
    )
    return obstacles


def _fdp_containing_chain(tree: ClusterTree, cluster_name: Optional[str]) -> List[str]:
    """Return a deepest-to-root cluster chain.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    cluster_name : str, optional
        Deepest cluster name.

    Returns
    -------
    list[str]
        Cluster chain beginning at ``cluster_name``.
    """
    chain: List[str] = []
    current = cluster_name
    while current is not None:
        chain.append(current)
        current = tree.parents[current]
    return chain


def _fdp_attachment_cluster(
    tree: ClusterTree,
    node_cluster: Optional[str],
    other_node: int,
) -> Optional[str]:
    """Choose the cluster boundary crossed by an inter-cluster edge.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    node_cluster : str, optional
        Deepest cluster containing the endpoint.
    other_node : int
        Opposite endpoint node index.

    Returns
    -------
    str or None
        Deepest containing cluster that does not also contain ``other_node``.
    """
    for cluster_name in _fdp_containing_chain(tree, node_cluster):
        if int(other_node) not in tree.descendants_per_cluster[cluster_name]:
            return cluster_name
    return None


def _fdp_intersect_ray_with_box(
    start: Tuple[float, float],
    end: Tuple[float, float],
    box: _FdpObstacleBox,
) -> Tuple[float, float]:
    """Intersect a ray from ``start`` toward ``end`` with a box boundary.

    Parameters
    ----------
    start : tuple[float, float]
        Ray origin, usually a node center inside a cluster.
    end : tuple[float, float]
        Point defining ray direction.
    box : _FdpObstacleBox
        Boundary box to intersect.

    Returns
    -------
    tuple[float, float]
        First boundary intersection in the ray direction. If the ray is
        degenerate, ``start`` is returned.
    """
    start_x, start_y = start
    end_x, end_y = end
    delta_x = end_x - start_x
    delta_y = end_y - start_y
    candidates: List[Tuple[float, float, float]] = []
    if abs(delta_x) > _FDP_EPSILON:
        x_boundary = box.x_max if delta_x > 0.0 else box.x_min
        scale = (x_boundary - start_x) / delta_x
        y_value = start_y + scale * delta_y
        if scale >= 0.0 and box.y_min - _FDP_EPSILON <= y_value <= box.y_max + _FDP_EPSILON:
            candidates.append((scale, x_boundary, y_value))
    if abs(delta_y) > _FDP_EPSILON:
        y_boundary = box.y_max if delta_y > 0.0 else box.y_min
        scale = (y_boundary - start_y) / delta_y
        x_value = start_x + scale * delta_x
        if scale >= 0.0 and box.x_min - _FDP_EPSILON <= x_value <= box.x_max + _FDP_EPSILON:
            candidates.append((scale, x_value, y_boundary))
    if not candidates:
        return start
    _, point_x, point_y = min(candidates, key=lambda item: item[0])
    return (point_x, point_y)


def _fdp_compute_compound_edge_attachments(
    problem: LayoutProblem,
    pos: torch.Tensor,
    expand: Tuple[float, float] = (0.0, 0.0),
    do_add: bool = True,
) -> Tuple[
    List[_FdpCompoundEdgeAttachment],
    Dict[str, _FdpObstacleBox],
    Dict[int, _FdpObstacleBox],
]:
    """Compute fdp compound-edge attachment metadata for fidelity mode.

    Parameters
    ----------
    problem : LayoutProblem
        Layout problem with edge tensor and optional cluster metadata.
    pos : torch.Tensor
        Final node positions with shape ``[N, 2]``.
    expand : tuple[float, float], default=(0.0, 0.0)
        Graphviz obstacle expansion values.
    do_add : bool, default=True
        Whether expansion is additive.

    Returns
    -------
    tuple[list[_FdpCompoundEdgeAttachment], dict[str, _FdpObstacleBox], dict[int, _FdpObstacleBox]]
        Edge attachment metadata, cluster obstacle boxes, and node obstacle
        boxes. Empty results are returned when the graph has no cluster tree.
    """
    tree = problem.get_cluster_tree()
    if tree is None or problem.edge_index.numel() == 0:
        return [], {}, {}

    work_pos = pos.detach().to(dtype=torch.float32, device="cpu")
    node_sizes = None
    if problem.node_sizes is not None:
        node_sizes = problem.node_sizes.detach().to(dtype=torch.float32, device="cpu")
    node_boxes = _fdp_node_boxes(work_pos, node_sizes, expand=expand, do_add=do_add)
    cluster_boxes = _fdp_cluster_boxes(tree, work_pos, node_sizes, expand=expand, do_add=do_add)
    node_parent = _fdp_deepest_cluster_by_node(tree, problem.num_nodes)
    attachments: List[_FdpCompoundEdgeAttachment] = []
    for edge_id, (source, target) in enumerate(problem.edge_index.t().tolist()):
        source_index = int(source)
        target_index = int(target)
        if source_index == target_index:
            continue
        source_point = (
            float(work_pos[source_index, 0].item()),
            float(work_pos[source_index, 1].item()),
        )
        target_point = (
            float(work_pos[target_index, 0].item()),
            float(work_pos[target_index, 1].item()),
        )
        tail_cluster = _fdp_attachment_cluster(
            tree,
            node_parent.get(source_index),
            target_index,
        )
        head_cluster = _fdp_attachment_cluster(
            tree,
            node_parent.get(target_index),
            source_index,
        )
        tail_point = source_point
        head_point = target_point
        if tail_cluster is not None and tail_cluster in cluster_boxes:
            tail_point = _fdp_intersect_ray_with_box(
                source_point,
                target_point,
                cluster_boxes[tail_cluster],
            )
        if head_cluster is not None and head_cluster in cluster_boxes:
            head_point = _fdp_intersect_ray_with_box(
                target_point,
                source_point,
                cluster_boxes[head_cluster],
            )
        obstacles = _fdp_compound_obstacle_list(
            source=source_index,
            target=target_index,
            tree=tree,
            node_parent=node_parent,
            node_boxes=node_boxes,
            cluster_boxes=cluster_boxes,
        )
        attachments.append(
            _FdpCompoundEdgeAttachment(
                edge_id=edge_id,
                source=source_index,
                target=target_index,
                tail_point=tail_point,
                head_point=head_point,
                tail_cluster=tail_cluster,
                head_cluster=head_cluster,
                obstacle_keys=tuple(obstacle.key for obstacle in obstacles),
                polyline=(tail_point, head_point),
            )
        )
    return attachments, cluster_boxes, node_boxes


@dataclass(frozen=True)
class _FdpCompoundEdgeAttachmentOp(Op):
    """Record Graphviz fdp compound-edge attachment metadata.

    Parameters
    ----------
    expand : tuple[float, float], default=(0.0, 0.0)
        Obstacle expansion values. The current FMMM public interface does not
        expose Graphviz ``esep``, so fidelity mode uses the zero-margin shape.
    do_add : bool, default=True
        Whether expansion is additive.
    """

    expand: Tuple[float, float] = (0.0, 0.0)
    do_add: bool = True

    name = "fmmm_fdp_compound_edge_attachment"
    category = OpCategory.POSTPROCESS
    reads = ("pos",)
    writes = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute and store compound-edge attachment metadata.

        Parameters
        ----------
        problem : LayoutProblem
            Layout problem with cluster metadata.
        state : SolveState
            Solve state containing final positions.
        ctx : RuntimeContext
            Runtime context, unused by this metadata op.

        Returns
        -------
        SolveState
            State with fidelity metadata stored in ``extras``.
        """
        del ctx
        if state.pos is None:
            state.extras[_FDP_COMPOUND_EDGE_ATTACHMENTS_KEY] = []
            state.extras[_FDP_COMPOUND_CLUSTER_OBSTACLES_KEY] = {}
            state.extras[_FDP_COMPOUND_NODE_OBSTACLES_KEY] = {}
            return state
        attachments, cluster_boxes, node_boxes = _fdp_compute_compound_edge_attachments(
            problem=problem,
            pos=state.pos,
            expand=self.expand,
            do_add=self.do_add,
        )
        state.extras[_FDP_COMPOUND_EDGE_ATTACHMENTS_KEY] = attachments
        state.extras[_FDP_COMPOUND_CLUSTER_OBSTACLES_KEY] = cluster_boxes
        state.extras[_FDP_COMPOUND_NODE_OBSTACLES_KEY] = node_boxes
        return state


_GRAPHVIZ_FDP_PACK_MARGIN = 4.0
_GRAPHVIZ_PACK_AVERAGE_POLYOMINO_SIZE = 100.0
_GRAPHVIZ_FDP_PORT_ANGLE_STEP = math.pi / 90.0


@dataclass(frozen=True)
class _FdpRecursionPort:
    """Boundary port induced by a parent derived-graph edge.

    Parameters
    ----------
    edge_id : int
        Original edge ordinal in ``edge_index``.
    node : int
        Original node inside the child cluster.
    alpha : float
        Port angle in radians.
    """

    edge_id: int
    node: int
    alpha: float


@dataclass(frozen=True)
class _FdpDerivedNode:
    """Node in the fdp recursion derived graph.

    Parameters
    ----------
    key : int or str
        Original node id, cluster name, or generated port key.
    kind : str
        One of ``"leaf"``, ``"cluster"``, or ``"port"``.
    members : frozenset[int]
        Original nodes represented by this derived node.
    """

    key: Union[int, str]
    kind: str
    members: frozenset[int]


@dataclass(frozen=True)
class _FdpDerivedEdge:
    """Edge in the fdp recursion derived graph.

    Parameters
    ----------
    source : int
        Local source node index.
    target : int
        Local target node index.
    real_edges : tuple[int, ...]
        Original edge ordinals represented by this derived edge.
    """

    source: int
    target: int
    real_edges: Tuple[int, ...]


@dataclass(frozen=True)
class _FdpDerivedGraph:
    """Collapsed Graphviz fdp derived graph for one recursion level.

    Parameters
    ----------
    nodes : tuple[_FdpDerivedNode, ...]
        Derived nodes in creation order.
    edges : tuple[_FdpDerivedEdge, ...]
        Unique derived edges.
    owner_by_node : Mapping[int, int | str]
        Original node id to owner key at this level.
    port_indices : frozenset[int]
        Derived node indices representing generated ports.
    """

    nodes: Tuple[_FdpDerivedNode, ...]
    edges: Tuple[_FdpDerivedEdge, ...]
    owner_by_node: Mapping[int, Union[int, str]]
    port_indices: frozenset[int]


@dataclass(frozen=True)
class _FdpLevelLayout:
    """Recursive fdp level layout result.

    Parameters
    ----------
    positions : Mapping[int, torch.Tensor]
        Original node positions in local coordinates.
    width : float
        Width of the level bbox.
    height : float
        Height of the level bbox.
    cluster_boxes : Mapping[str, tuple[float, float, float, float]]
        Cluster bboxes in local coordinates.
    """

    positions: Mapping[int, torch.Tensor]
    width: float
    height: float
    cluster_boxes: Mapping[str, Tuple[float, float, float, float]]


def _fdp_recursion_child_clusters(
    tree: ClusterTree,
    cluster_name: Optional[str],
) -> Tuple[str, ...]:
    """Return immediate child clusters for one fdp recursion level.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    cluster_name : str, optional
        Current cluster name, or ``None`` for the root graph.

    Returns
    -------
    tuple[str, ...]
        Immediate child cluster names in stable graph order.
    """
    if cluster_name is None:
        return tree.roots
    return tree.children_per_cluster[cluster_name]


def _fdp_recursion_direct_leaves(
    num_nodes: int,
    tree: ClusterTree,
    cluster_name: Optional[str],
) -> Tuple[int, ...]:
    """Return non-cluster leaf nodes owned directly by a recursion level.

    Parameters
    ----------
    num_nodes : int
        Total original node count.
    tree : ClusterTree
        Cluster hierarchy.
    cluster_name : str, optional
        Current cluster name, or ``None`` for root.

    Returns
    -------
    tuple[int, ...]
        Direct original node ids.
    """
    if cluster_name is not None:
        return tuple(sorted(int(index) for index in tree.leaves_per_cluster[cluster_name]))

    clustered_nodes: set[int] = set()
    for root_name in tree.roots:
        clustered_nodes.update(int(index) for index in tree.descendants_per_cluster[root_name])
    return tuple(index for index in range(num_nodes) if index not in clustered_nodes)


def _fdp_recursion_owner_map(
    tree: ClusterTree,
    cluster_name: Optional[str],
    child_clusters: Sequence[str],
    direct_leaves: Sequence[int],
) -> Dict[int, Union[int, str]]:
    """Map original nodes to derived owners for one recursion level.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy.
    cluster_name : str, optional
        Current cluster name, or ``None`` for root.
    child_clusters : Sequence[str]
        Child clusters represented as derived nodes.
    direct_leaves : Sequence[int]
        Direct original leaves represented as derived nodes.

    Returns
    -------
    dict[int, int | str]
        Original node id to derived node key.
    """
    owners: Dict[int, Union[int, str]] = {int(node): int(node) for node in direct_leaves}
    for child_name in child_clusters:
        for node_index in tree.descendants_per_cluster[child_name]:
            owners[int(node_index)] = child_name
    if cluster_name is not None:
        allowed = set(int(index) for index in tree.descendants_per_cluster[cluster_name])
        owners = {
            node_index: owner for node_index, owner in owners.items() if node_index in allowed
        }
    return owners


def _fdp_recursion_derive_graph(
    edge_index: torch.Tensor,
    num_nodes: int,
    tree: ClusterTree,
    cluster_name: Optional[str],
    ports: Sequence[_FdpRecursionPort] = (),
) -> _FdpDerivedGraph:
    """Create Graphviz fdp's cluster-collapsed derived graph.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original edge tensor with shape ``[2, E]``.
    num_nodes : int
        Total original node count.
    tree : ClusterTree
        Cluster hierarchy.
    cluster_name : str, optional
        Current cluster name, or ``None`` for root.
    ports : Sequence[_FdpRecursionPort], default=()
        Boundary ports generated from the parent level.

    Returns
    -------
    _FdpDerivedGraph
        Derived graph with child clusters collapsed to nodes.
    """
    child_clusters = _fdp_recursion_child_clusters(tree, cluster_name)
    direct_leaves = _fdp_recursion_direct_leaves(num_nodes, tree, cluster_name)
    owners = _fdp_recursion_owner_map(tree, cluster_name, child_clusters, direct_leaves)
    nodes: List[_FdpDerivedNode] = [
        _FdpDerivedNode(
            key=child_name,
            kind="cluster",
            members=frozenset(int(index) for index in tree.descendants_per_cluster[child_name]),
        )
        for child_name in child_clusters
    ]
    nodes.extend(
        _FdpDerivedNode(key=int(node_index), kind="leaf", members=frozenset({int(node_index)}))
        for node_index in direct_leaves
    )
    index_by_key: Dict[Union[int, str], int] = {node.key: index for index, node in enumerate(nodes)}

    grouped_edges: Dict[Tuple[int, int], List[int]] = {}
    edge_order: List[Tuple[int, int]] = []
    for edge_id, (source, target) in enumerate(edge_index.t().tolist()):
        source_owner = owners.get(int(source))
        target_owner = owners.get(int(target))
        if source_owner is None or target_owner is None or source_owner == target_owner:
            continue
        source_index = index_by_key[source_owner]
        target_index = index_by_key[target_owner]
        key = (
            (source_index, target_index)
            if source_index <= target_index
            else (target_index, source_index)
        )
        if key not in grouped_edges:
            grouped_edges[key] = []
            edge_order.append(key)
        grouped_edges[key].append(edge_id)

    port_indices: set[int] = set()
    for port_index, port in enumerate(ports):
        owner = owners.get(int(port.node))
        if owner is None:
            continue
        derived_index = len(nodes)
        nodes.append(
            _FdpDerivedNode(
                key=f"_port_{cluster_name or 'root'}_{port_index}",
                kind="port",
                members=frozenset({int(port.node)}),
            )
        )
        port_indices.add(derived_index)
        owner_index = index_by_key[owner]
        key = (
            (owner_index, derived_index)
            if owner_index <= derived_index
            else (derived_index, owner_index)
        )
        grouped_edges[key] = [int(port.edge_id)]
        edge_order.append(key)

    return _FdpDerivedGraph(
        nodes=tuple(nodes),
        edges=tuple(
            _FdpDerivedEdge(source=source, target=target, real_edges=tuple(grouped_edges[key]))
            for key in edge_order
            for source, target in [key]
        ),
        owner_by_node=owners,
        port_indices=frozenset(port_indices),
    )


def _fdp_recursion_components(derived: _FdpDerivedGraph) -> Tuple[Tuple[int, ...], ...]:
    """Find Graphviz fdp generalized connected components.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Derived graph whose port components should merge first.

    Returns
    -------
    tuple[tuple[int, ...], ...]
        Connected components in ``findCComp`` order.
    """
    adjacency: List[List[int]] = [[] for _node in derived.nodes]
    for edge in derived.edges:
        adjacency[edge.source].append(edge.target)
        adjacency[edge.target].append(edge.source)
    marked = [False] * len(derived.nodes)
    components: List[Tuple[int, ...]] = []

    def dfs(node_index: int, out: List[int]) -> None:
        """Append a connected component using Graphviz-style DFS.

        Parameters
        ----------
        node_index : int
            Derived node index to visit.
        out : list[int]
            Mutable component accumulator.

        Returns
        -------
        None
            ``marked`` and ``out`` are mutated in place.
        """
        marked[node_index] = True
        out.append(node_index)
        for other in adjacency[node_index]:
            if not marked[other]:
                dfs(other, out)

    if derived.port_indices:
        merged_ports: List[int] = []
        for port_index in sorted(derived.port_indices):
            if not marked[port_index]:
                dfs(port_index, merged_ports)
        components.append(tuple(merged_ports))

    for node_index in range(len(derived.nodes)):
        if marked[node_index]:
            continue
        component: List[int] = []
        dfs(node_index, component)
        components.append(tuple(component))
    return tuple(components)


def _fdp_recursion_component_edges(
    derived: _FdpDerivedGraph,
    component: Sequence[int],
) -> torch.Tensor:
    """Build a local edge tensor for a derived component.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Derived graph.
    component : Sequence[int]
        Derived node indices in the component.

    Returns
    -------
    torch.Tensor
        Local edge tensor with shape ``[2, E]``.
    """
    local_index = {node_index: index for index, node_index in enumerate(component)}
    edges = [
        (local_index[edge.source], local_index[edge.target])
        for edge in derived.edges
        if edge.source in local_index and edge.target in local_index
    ]
    if not edges:
        return torch.empty((2, 0), dtype=torch.long)
    return torch.tensor(edges, dtype=torch.long).t().contiguous()


def _fdp_recursion_component_sizes(
    derived: _FdpDerivedGraph,
    component: Sequence[int],
    node_sizes: Optional[torch.Tensor],
    child_layouts: Mapping[str, _FdpLevelLayout],
) -> torch.Tensor:
    """Return temporary sizes for a derived component layout or bbox.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Derived graph.
    component : Sequence[int]
        Derived node indices in a component.
    node_sizes : torch.Tensor, optional
        Original node sizes with shape ``[N, 2]``.
    child_layouts : Mapping[str, _FdpLevelLayout]
        Already-laid-out child clusters keyed by cluster name.

    Returns
    -------
    torch.Tensor
        Size tensor with shape ``[N_component, 2]``.
    """
    sizes: List[torch.Tensor] = []
    for derived_index in component:
        node = derived.nodes[derived_index]
        if node.kind == "leaf" and node_sizes is not None:
            sizes.append(node_sizes[int(node.key)].detach().to(dtype=torch.float32, device="cpu"))
        elif node.kind == "cluster" and str(node.key) in child_layouts:
            child = child_layouts[str(node.key)]
            sizes.append(torch.tensor([child.width, child.height], dtype=torch.float32))
        elif node.kind == "port":
            sizes.append(torch.zeros(2, dtype=torch.float32))
        else:
            sizes.append(torch.ones(2, dtype=torch.float32))
    if not sizes:
        return torch.empty((0, 2), dtype=torch.float32)
    return torch.stack(sizes)


def _fdp_recursion_layout_component(
    derived: _FdpDerivedGraph,
    component: Sequence[int],
    node_sizes: Optional[torch.Tensor],
    steps: int,
    seed: int,
) -> torch.Tensor:
    """Lay out a derived component with Dagua's FM^3 primitive.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Derived graph.
    component : Sequence[int]
        Derived component node indices.
    node_sizes : torch.Tensor, optional
        Original node sizes with shape ``[N, 2]``.
    steps : int
        FM^3 iteration budget.
    seed : int
        Deterministic seed.

    Returns
    -------
    torch.Tensor
        Component positions with shape ``[N_component, 2]``.
    """
    if len(component) == 0:
        return torch.empty((0, 2), dtype=torch.float32)
    if len(component) == 1:
        return torch.zeros((1, 2), dtype=torch.float32)
    return (
        layout_fmmm_pipeline(
            edge_index=_fdp_recursion_component_edges(derived, component),
            num_nodes=len(component),
            node_sizes=_fdp_recursion_component_sizes(derived, component, node_sizes, {}),
            steps=steps,
            seed=seed,
            reference_mode=True,
            fidelity_mode=False,
        )
        .detach()
        .to(dtype=torch.float32, device="cpu")
    )


def _fdp_recursion_expand_cluster_ports(
    derived: _FdpDerivedGraph,
    derived_positions: Mapping[int, torch.Tensor],
    cluster_index: int,
    edge_index: torch.Tensor,
) -> Tuple[_FdpRecursionPort, ...]:
    """Generate child ports using Graphviz fdp ``expandCluster`` ordering.

    Parameters
    ----------
    derived : _FdpDerivedGraph
        Positioned derived graph.
    derived_positions : Mapping[int, torch.Tensor]
        Derived-node positions keyed by derived node index.
    cluster_index : int
        Derived node index for the cluster being expanded.
    edge_index : torch.Tensor
        Original edge tensor with shape ``[2, E]``.

    Returns
    -------
    tuple[_FdpRecursionPort, ...]
        Generated child ports in Graphviz order.
    """
    center = derived_positions[cluster_index]
    incident: List[Tuple[float, float, int, _FdpDerivedEdge]] = []
    for edge_order, edge in enumerate(derived.edges):
        if edge.source != cluster_index and edge.target != cluster_index:
            continue
        other_index = edge.target if edge.source == cluster_index else edge.source
        other = derived_positions[other_index]
        dx = float((other[0] - center[0]).item())
        dy = float((other[1] - center[1]).item())
        incident.append((math.atan2(dy, dx), dx * dx + dy * dy, edge_order, edge))
    incident.sort(key=lambda item: (item[0], item[1], item[2]))

    adjusted: List[Tuple[float, float, int, _FdpDerivedEdge]] = []
    index = 0
    while index < len(incident):
        alpha = incident[index][0]
        end = index + 1
        while end < len(incident) and incident[end][0] == alpha:
            end += 1
        if end == index + 1:
            adjusted.append(incident[index])
        else:
            bound = math.pi if end == len(incident) else incident[end][0]
            delta = min((bound - alpha) / (end - index), _GRAPHVIZ_FDP_PORT_ANGLE_STEP)
            for offset, item in enumerate(incident[index:end]):
                adjusted.append((alpha + offset * delta, item[1], item[2], item[3]))
        index = end
    incident = adjusted

    ports: List[_FdpRecursionPort] = []
    first_alpha = incident[0][0] if incident else 0.0
    for item_index, (alpha, _dist2, _edge_order, edge) in enumerate(incident):
        bound = (
            incident[item_index + 1][0]
            if item_index + 1 < len(incident)
            else 2.0 * math.pi + first_alpha
        )
        real_edges = list(edge.real_edges)
        delta = min((bound - alpha) / max(len(real_edges), 1), _GRAPHVIZ_FDP_PORT_ANGLE_STEP)
        other_index = edge.target if edge.source == cluster_index else edge.source
        if cluster_index > other_index:
            real_edges.reverse()
            alpha += delta * (len(real_edges) - 1)
            delta = -delta
        for real_edge in real_edges:
            source = int(edge_index[0, real_edge].item())
            target = int(edge_index[1, real_edge].item())
            internal_node = (
                source
                if derived.owner_by_node.get(source) == derived.nodes[cluster_index].key
                else target
            )
            ports.append(
                _FdpRecursionPort(
                    edge_id=int(real_edge),
                    node=int(internal_node),
                    alpha=float(alpha),
                )
            )
            alpha += delta
    return tuple(ports)


def _fdp_recursion_bbox_from_positions(
    positions: Mapping[int, torch.Tensor],
    node_sizes: Optional[torch.Tensor],
) -> Tuple[float, float, float, float]:
    """Compute a bbox around original nodes.

    Parameters
    ----------
    positions : Mapping[int, torch.Tensor]
        Original node positions.
    node_sizes : torch.Tensor, optional
        Original node sizes with shape ``[N, 2]``.

    Returns
    -------
    tuple[float, float, float, float]
        Bounds as ``(x_min, y_min, x_max, y_max)``.
    """
    if not positions:
        return (0.0, 0.0, 0.0, 0.0)
    lower_parts: List[torch.Tensor] = []
    upper_parts: List[torch.Tensor] = []
    for node_index, position in positions.items():
        if node_sizes is None:
            size = torch.ones(2, dtype=torch.float32)
        else:
            size = node_sizes[int(node_index)].detach().to(dtype=torch.float32, device="cpu")
        half = size / 2.0
        lower_parts.append(position.to(dtype=torch.float32, device="cpu") - half)
        upper_parts.append(position.to(dtype=torch.float32, device="cpu") + half)
    lower = torch.stack(lower_parts).min(dim=0).values
    upper = torch.stack(upper_parts).max(dim=0).values
    return (
        float(lower[0].item()),
        float(lower[1].item()),
        float(upper[0].item()),
        float(upper[1].item()),
    )


def _fdp_recursion_shift_to_origin(
    positions: Mapping[int, torch.Tensor],
    node_sizes: Optional[torch.Tensor],
    cluster_boxes: Mapping[str, Tuple[float, float, float, float]],
) -> _FdpLevelLayout:
    """Translate a recursive level so the bbox lower-left is the origin.

    Parameters
    ----------
    positions : Mapping[int, torch.Tensor]
        Original node positions.
    node_sizes : torch.Tensor, optional
        Original node sizes with shape ``[N, 2]``.
    cluster_boxes : Mapping[str, tuple[float, float, float, float]]
        Cluster boxes in the same coordinates as ``positions``.

    Returns
    -------
    _FdpLevelLayout
        Shifted level layout.
    """
    x_min, y_min, x_max, y_max = _fdp_recursion_bbox_from_positions(positions, node_sizes)
    shift = torch.tensor([-x_min, -y_min], dtype=torch.float32)
    shifted_positions = {
        node_index: position.to(dtype=torch.float32, device="cpu") + shift
        for node_index, position in positions.items()
    }
    shifted_boxes = {
        name: (box[0] - x_min, box[1] - y_min, box[2] - x_min, box[3] - y_min)
        for name, box in cluster_boxes.items()
    }
    return _FdpLevelLayout(
        positions=shifted_positions,
        width=max(x_max - x_min, 0.0),
        height=max(y_max - y_min, 0.0),
        cluster_boxes=shifted_boxes,
    )


def _fdp_recursion_component_offsets(
    component_boxes: Sequence[Tuple[float, float]],
) -> List[torch.Tensor]:
    """Pack recursive components with Graphviz fdp tile packing.

    Parameters
    ----------
    component_boxes : Sequence[tuple[float, float]]
        Width and height for each component.

    Returns
    -------
    list[torch.Tensor]
        Translation offsets for each component.

    Notes
    -----
    The R36 tile-packing port covers Graphviz ``packGraphs`` behavior for
    component bounding boxes. Reusing it here keeps the clustered recursion path
    on the same fdp fidelity component instead of the earlier row-pack fallback.
    """
    boxes = [(0.0, 0.0, float(width), float(height)) for width, height in component_boxes]
    return [
        torch.tensor(offset, dtype=torch.float32) for offset in _graphviz_tile_pack_offsets(boxes)
    ]


def _fdp_recursion_layout_level(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    tree: ClusterTree,
    cluster_name: Optional[str],
    steps: int,
    seed: int,
    ports: Sequence[_FdpRecursionPort] = (),
) -> _FdpLevelLayout:
    """Lay out one graph or cluster using Graphviz fdp recursion.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original edge tensor with shape ``[2, E]``.
    num_nodes : int
        Total original node count.
    node_sizes : torch.Tensor, optional
        Original node sizes with shape ``[N, 2]``.
    tree : ClusterTree
        Cluster hierarchy.
    cluster_name : str, optional
        Current cluster name, or ``None`` for root.
    steps : int
        FM^3 iteration budget for each derived component.
    seed : int
        Deterministic seed.
    ports : Sequence[_FdpRecursionPort], default=()
        Parent-generated boundary ports.

    Returns
    -------
    _FdpLevelLayout
        Recursive level layout with original-node positions.
    """
    derived = _fdp_recursion_derive_graph(edge_index, num_nodes, tree, cluster_name, ports)
    if not derived.nodes:
        return _FdpLevelLayout(positions={}, width=0.0, height=0.0, cluster_boxes={})

    components = _fdp_recursion_components(derived)
    child_layouts: Dict[str, _FdpLevelLayout] = {}
    component_positions: List[Dict[int, torch.Tensor]] = []
    component_boxes: List[Tuple[float, float]] = []

    for component_offset, component in enumerate(components):
        local_tensor = _fdp_recursion_layout_component(
            derived=derived,
            component=component,
            node_sizes=node_sizes,
            steps=steps,
            seed=seed + component_offset,
        )
        local_positions = {
            derived_index: local_tensor[local_index]
            for local_index, derived_index in enumerate(component)
        }
        for derived_index in component:
            node = derived.nodes[derived_index]
            if node.kind != "cluster":
                continue
            child_ports = _fdp_recursion_expand_cluster_ports(
                derived=derived,
                derived_positions=local_positions,
                cluster_index=derived_index,
                edge_index=edge_index,
            )
            child_layouts[str(node.key)] = _fdp_recursion_layout_level(
                edge_index=edge_index,
                num_nodes=num_nodes,
                node_sizes=node_sizes,
                tree=tree,
                cluster_name=str(node.key),
                steps=steps,
                seed=seed + component_offset + 1,
                ports=child_ports,
            )

        sizes = _fdp_recursion_component_sizes(derived, component, node_sizes, child_layouts)
        if sizes.numel() == 0:
            component_boxes.append((0.0, 0.0))
        else:
            half_sizes = sizes / 2.0
            lower = local_tensor - half_sizes
            upper = local_tensor + half_sizes
            component_boxes.append(
                (
                    max(float((upper[:, 0].max() - lower[:, 0].min()).item()), 0.0),
                    max(float((upper[:, 1].max() - lower[:, 1].min()).item()), 0.0),
                )
            )
        component_positions.append(local_positions)

    offsets = _fdp_recursion_component_offsets(component_boxes)
    final_positions: Dict[int, torch.Tensor] = {}
    cluster_boxes: Dict[str, Tuple[float, float, float, float]] = {}
    for component, local_positions, offset in zip(components, component_positions, offsets):
        for derived_index in component:
            node = derived.nodes[derived_index]
            if node.kind == "port":
                continue
            position = local_positions[derived_index] + offset
            if node.kind == "leaf":
                final_positions[int(node.key)] = position
                continue
            child = child_layouts[str(node.key)]
            child_offset = position - torch.tensor(
                [child.width / 2.0, child.height / 2.0],
                dtype=torch.float32,
            )
            x_shift = float(child_offset[0].item())
            y_shift = float(child_offset[1].item())
            cluster_boxes[str(node.key)] = (
                x_shift,
                y_shift,
                x_shift + child.width,
                y_shift + child.height,
            )
            for child_name, child_box in child.cluster_boxes.items():
                cluster_boxes[child_name] = (
                    child_box[0] + x_shift,
                    child_box[1] + y_shift,
                    child_box[2] + x_shift,
                    child_box[3] + y_shift,
                )
            for node_index, child_position in child.positions.items():
                final_positions[int(node_index)] = child_position + child_offset

    return _fdp_recursion_shift_to_origin(final_positions, node_sizes, cluster_boxes)


def graphviz_fdp_fidelity(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 200,
    seed: int = 42,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
) -> torch.Tensor:
    """Run Graphviz fdp derived-graph recursion for clustered graphs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Original edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of original nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    steps : int, default=200
        FM^3 iteration budget for each derived component.
    seed : int, default=42
        Deterministic seed.
    clusters : Mapping[str, Sequence[int]], optional
        Flat descendant membership for each cluster.
    cluster_parents : Mapping[str, str | None], optional
        Parent mapping for clusters.

    Returns
    -------
    torch.Tensor
        Original node positions with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If inputs are invalid or cluster metadata is missing.

    Notes
    -----
    This ports Graphviz fdp's derived-graph recursion and boundary-port
    expansion. Graphviz's exact ``tLayout``, ``xLayout``, and ``packGraphs``
    numerical kernels remain integration assumptions for later R36 slices.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if not clusters:
        raise ValueError("graphviz_fdp_fidelity requires cluster metadata.")

    tree = ClusterTree.from_flat_membership(clusters, cluster_parents or {})
    cpu_edge_index = edge_index.detach().to(device="cpu", dtype=torch.long)
    cpu_node_sizes = (
        node_sizes.detach().to(device="cpu", dtype=torch.float32)
        if node_sizes is not None
        else None
    )
    layout = _fdp_recursion_layout_level(
        edge_index=cpu_edge_index,
        num_nodes=num_nodes,
        node_sizes=cpu_node_sizes,
        tree=tree,
        cluster_name=None,
        steps=steps,
        seed=seed,
    )
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node_index, position in layout.positions.items():
        positions[int(node_index)] = position.to(dtype=torch.float32, device="cpu")
    return positions.to(device=_layout_device(edge_index=edge_index, node_sizes=node_sizes))


def _weak_components(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Compute weak components in deterministic node order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Weak components as sorted parent node indices.
    """
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        source_index = int(source)
        target_index = int(target)
        if source_index == target_index:
            continue
        neighbors[source_index].append(target_index)
        neighbors[target_index].append(source_index)

    seen = [False] * num_nodes
    components: list[list[int]] = []
    for start in range(num_nodes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: list[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in neighbors[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _slice_component_edges(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    component: list[int],
) -> tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return component-local edges and optional weights.

    Parameters
    ----------
    edge_index : torch.Tensor
        Parent edge tensor with shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Optional parent edge weights with shape ``[E]``.
    component : list[int]
        Parent node indices in one weak component.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor or None]
        Relabeled component edge tensor and aligned weights.
    """
    local_by_parent = {node: index for index, node in enumerate(component)}
    sources: list[int] = []
    targets: list[int] = []
    weights: list[float] = []
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    weights_cpu = None if edge_weights is None else edge_weights.detach().to(device="cpu")
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        source_index = int(source)
        target_index = int(target)
        if source_index not in local_by_parent or target_index not in local_by_parent:
            continue
        sources.append(local_by_parent[source_index])
        targets.append(local_by_parent[target_index])
        if weights_cpu is not None:
            weights.append(float(weights_cpu[edge_id].item()))

    local_edges = torch.tensor([sources, targets], dtype=torch.long, device=edge_index.device)
    if edge_weights is None:
        return local_edges, None
    local_weights = torch.tensor(weights, dtype=edge_weights.dtype, device=edge_weights.device)
    return local_edges, local_weights


def _c_round(value: float) -> int:
    """Round like C99 ``round``.

    Parameters
    ----------
    value : float
        Input value.

    Returns
    -------
    int
        Nearest integer with half values rounded away from zero.
    """
    if value >= 0.0:
        return int(math.floor(value + 0.5))
    return int(math.ceil(value - 0.5))


def _c_int_div(numerator: int, denominator: int) -> int:
    """Divide integers like C99 truncation toward zero.

    Parameters
    ----------
    numerator : int
        Integer numerator.
    denominator : int
        Non-zero integer denominator.

    Returns
    -------
    int
        Truncated quotient.
    """
    quotient = abs(numerator) // abs(denominator)
    return quotient if numerator * denominator >= 0 else -quotient


def _graphviz_grid_count(width: float, step: int) -> int:
    """Return Graphviz ``GRID`` cell count for a positive span.

    Parameters
    ----------
    width : float
        Span length.
    step : int
        Grid cell size.

    Returns
    -------
    int
        Number of grid cells needed to cover the span.
    """
    return int(math.ceil(width / step))


def _graphviz_cell(value: float, step: int) -> int:
    """Return the rounded Graphviz grid cell containing a coordinate.

    Parameters
    ----------
    value : float
        Coordinate value.
    step : int
        Grid cell size.

    Returns
    -------
    int
        Rounded grid-cell coordinate.
    """
    if value >= 0.0:
        return _c_round(value / step)
    return _c_round(((value + 1.0) / step) - 1.0)


def _graphviz_pack_step(
    boxes: list[tuple[float, float, float, float]],
    margin: float,
) -> int:
    """Compute Graphviz pack.c's grid step size for component boxes.

    Parameters
    ----------
    boxes : list[tuple[float, float, float, float]]
        Bounding boxes as ``(llx, lly, urx, ury)``.
    margin : float
        Extra pack margin around each component.

    Returns
    -------
    int
        Positive grid step size.
    """
    count = len(boxes)
    a_value = _GRAPHVIZ_PACK_AVERAGE_POLYOMINO_SIZE * count - 1.0
    b_value = 0.0
    c_value = 0.0
    for llx, lly, urx, ury in boxes:
        width = urx - llx + 2.0 * margin
        height = ury - lly + 2.0 * margin
        b_value -= width + height
        c_value -= width * height
    discriminant = b_value * b_value - 4.0 * a_value * c_value
    root = int((-b_value + math.sqrt(discriminant)) / (2.0 * a_value))
    return root if root != 0 else 1


def _graphviz_box_cells(
    box: tuple[float, float, float, float],
    step: int,
    margin: float,
) -> tuple[list[tuple[int, int]], int]:
    """Generate the bbox polyomino cells used by Graphviz ``genBox``.

    Parameters
    ----------
    box : tuple[float, float, float, float]
        Bounding box as ``(llx, lly, urx, ury)``.
    step : int
        Grid cell size.
    margin : float
        Extra pack margin around the box.

    Returns
    -------
    tuple[list[tuple[int, int]], int]
        Occupied cells and half-perimeter sort key.
    """
    llx, lly, urx, ury = box
    rounded_llx = _c_round(llx)
    rounded_lly = _c_round(lly)
    rounded_urx = _c_round(urx)
    rounded_ury = _c_round(ury)
    low_x = _graphviz_cell(-margin, step)
    low_y = _graphviz_cell(-margin, step)
    high_x = _graphviz_cell(float(rounded_urx - rounded_llx) + margin, step)
    high_y = _graphviz_cell(float(rounded_ury - rounded_lly) + margin, step)

    cells = [
        (x_coord, y_coord)
        for x_coord in range(low_x, high_x + 1)
        for y_coord in range(low_y, high_y + 1)
    ]
    width_cells = _graphviz_grid_count(urx - llx + 2.0 * margin, step)
    height_cells = _graphviz_grid_count(ury - lly + 2.0 * margin, step)
    return cells, width_cells + height_cells


def _graphviz_fits(
    x_cell: int,
    y_cell: int,
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
) -> bool:
    """Return whether a translated polyomino does not overlap occupied cells.

    Parameters
    ----------
    x_cell : int
        Candidate x grid-cell offset.
    y_cell : int
        Candidate y grid-cell offset.
    cells : list[tuple[int, int]]
        Polyomino cells.
    occupied : set[tuple[int, int]]
        Already occupied cells.

    Returns
    -------
    bool
        ``True`` if every translated cell is available.
    """
    return all((x_coord + x_cell, y_coord + y_cell) not in occupied for x_coord, y_coord in cells)


def _graphviz_commit_fit(
    x_cell: int,
    y_cell: int,
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    box: tuple[float, float, float, float],
    step: int,
) -> tuple[float, float]:
    """Commit a fitted polyomino and return its Graphviz translation.

    Parameters
    ----------
    x_cell : int
        Accepted x grid-cell offset.
    y_cell : int
        Accepted y grid-cell offset.
    cells : list[tuple[int, int]]
        Polyomino cells.
    occupied : set[tuple[int, int]]
        Mutable occupied-cell set.
    box : tuple[float, float, float, float]
        Original component bounding box.
    step : int
        Grid cell size.

    Returns
    -------
    tuple[float, float]
        Translation in layout units.
    """
    for x_coord, y_coord in cells:
        occupied.add((x_coord + x_cell, y_coord + y_cell))
    return float(step * x_cell - _c_round(box[0])), float(step * y_cell - _c_round(box[1]))


def _graphviz_place_component(
    sorted_index: int,
    cells: list[tuple[int, int]],
    occupied: set[tuple[int, int]],
    box: tuple[float, float, float, float],
    step: int,
    margin: float,
) -> tuple[float, float]:
    """Place one component using Graphviz pack.c's spiral search.

    Parameters
    ----------
    sorted_index : int
        Position in descending polyomino-perimeter order.
    cells : list[tuple[int, int]]
        Polyomino cells for the component.
    occupied : set[tuple[int, int]]
        Mutable occupied-cell set.
    box : tuple[float, float, float, float]
        Original component bounding box.
    step : int
        Grid cell size.
    margin : float
        Extra pack margin around the component.

    Returns
    -------
    tuple[float, float]
        Translation for the component.
    """
    llx, lly, urx, ury = box
    if sorted_index == 0:
        width_cells = _graphviz_grid_count(urx - llx + 2.0 * margin, step)
        height_cells = _graphviz_grid_count(ury - lly + 2.0 * margin, step)
        x_cell = _c_int_div(-width_cells, 2)
        y_cell = _c_int_div(-height_cells, 2)
        if _graphviz_fits(x_cell, y_cell, cells, occupied):
            return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)

    if _graphviz_fits(0, 0, cells, occupied):
        return _graphviz_commit_fit(0, 0, cells, occupied, box, step)

    width = math.ceil(urx - llx)
    height = math.ceil(ury - lly)
    bound = 1
    while True:
        if width >= height:
            x_cell = 0
            y_cell = -bound
            while x_cell < bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                x_cell += 1
            while y_cell < bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                y_cell += 1
            while x_cell > -bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                x_cell -= 1
            while y_cell > -bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                y_cell -= 1
            while x_cell < 0:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                x_cell += 1
        else:
            y_cell = 0
            x_cell = -bound
            while y_cell > -bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                y_cell -= 1
            while x_cell < bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                x_cell += 1
            while y_cell < bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                y_cell += 1
            while x_cell > -bound:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                x_cell -= 1
            while y_cell > 0:
                if _graphviz_fits(x_cell, y_cell, cells, occupied):
                    return _graphviz_commit_fit(x_cell, y_cell, cells, occupied, box, step)
                y_cell -= 1
        bound += 1


def _graphviz_tile_pack_offsets(
    boxes: list[tuple[float, float, float, float]],
    margin: float = _GRAPHVIZ_FDP_PACK_MARGIN,
) -> list[tuple[float, float]]:
    """Pack component boxes with Graphviz's bbox polyomino tile search.

    Parameters
    ----------
    boxes : list[tuple[float, float, float, float]]
        Component bounding boxes as ``(llx, lly, urx, ury)``.
    margin : float, default=4.0
        Graphviz fdp's default pack margin in points, ``CL_OFFSET / 2``.

    Returns
    -------
    list[tuple[float, float]]
        Per-component translations in original component order.
    """
    if not boxes:
        return []
    step = _graphviz_pack_step(boxes, margin)
    packed_info: list[tuple[int, int, list[tuple[int, int]]]] = []
    for index, box in enumerate(boxes):
        cells, perimeter = _graphviz_box_cells(box, step, margin)
        packed_info.append((index, perimeter, cells))

    packed_info.sort(key=lambda item: -item[1])
    occupied: set[tuple[int, int]] = set()
    offsets = [(0.0, 0.0) for _ in boxes]
    for sorted_index, (box_index, _, cells) in enumerate(packed_info):
        offsets[box_index] = _graphviz_place_component(
            sorted_index=sorted_index,
            cells=cells,
            occupied=occupied,
            box=boxes[box_index],
            step=step,
            margin=margin,
        )
    return offsets


def _component_box(
    positions: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> tuple[float, float, float, float]:
    """Compute a component bounding box from positions and optional node sizes.

    Parameters
    ----------
    positions : torch.Tensor
        Component positions with shape ``[C, 2]``.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[C, 2]``.

    Returns
    -------
    tuple[float, float, float, float]
        Bounding box as ``(llx, lly, urx, ury)``.
    """
    positions_cpu = positions.detach().to(device="cpu", dtype=torch.float64)
    if positions_cpu.numel() == 0:
        return (0.0, 0.0, 0.0, 0.0)
    if node_sizes is None or node_sizes.numel() == 0:
        half_sizes = torch.zeros_like(positions_cpu)
    else:
        half_sizes = node_sizes.detach().to(device="cpu", dtype=torch.float64) / 2.0
    lower = positions_cpu - half_sizes
    upper = positions_cpu + half_sizes
    mins = lower.min(dim=0).values
    maxs = upper.max(dim=0).values
    return (
        float(mins[0].item()),
        float(mins[1].item()),
        float(maxs[0].item()),
        float(maxs[1].item()),
    )


def _translate_packed_components_to_origin(
    packed: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
) -> torch.Tensor:
    """Translate packed component coordinates so the lower-left box is at zero.

    Parameters
    ----------
    packed : torch.Tensor
        Packed positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Packed positions translated like Graphviz ``finalCC`` root output.
    """
    if packed.numel() == 0:
        return packed
    if node_sizes is None or node_sizes.numel() == 0:
        half_sizes = torch.zeros_like(packed)
    else:
        half_sizes = node_sizes.to(device=packed.device, dtype=packed.dtype) / 2.0
    lower = packed - half_sizes
    mins = lower.min(dim=0).values
    return packed - mins.unsqueeze(0)


def build_fmmm_pipeline(
    steps: int = 200,
    force_model: str = "ogdf_new",
    reference_mode: bool = False,
    fidelity_mode: bool = False,
) -> Pipeline:
    """Build an FM^3 multilevel force-directed pipeline.

    Reference fidelity
    ------------------
    Targets: Graphviz 7.0.5 fdp / Hachul and Junger (2004), "Drawing Large
        Graphs with a Potential-Field-Based Multilevel Algorithm".
    Fidelity mode: ``reference_mode=True`` or ``fidelity_mode=True`` enables
        OGDF/Graphviz-aligned coarsening, coarsest initialization, and force
        scaling choices used by evaluation competitors. ``fidelity_mode=True``
        also records fdp compound-edge attachment metadata in
        ``SolveState.extras`` for clustered graphs.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.067
        to 0.179 across step-count variants. Round 33 fdp bounded subset
        remained 0.121966.
    Known divergences:
        - Graphviz fdp exact ``tLayout``/``xLayout`` numeric kernels remain
          approximated by Dagua's FM^3 component solve.
        - Dagua keeps a fallback single-level solve when multilevel setup is
          unsuitable.

    Parameters
    ----------
    steps : int, default=200
        Total refinement budget distributed across hierarchy levels.
    force_model : str, default="ogdf_new"
        Spring-force model for edge attraction. ``"ogdf_new"`` matches
        OGDF's default; ``"fr"`` preserves Dagua's earlier coefficient for
        benchmark fallback selection.
    reference_mode : bool, default=False
        Use OGDF-aligned coarsening, coarsest initialization, and force
        scaling choices for fidelity comparisons.
    fidelity_mode : bool, default=False
        Alias for ``reference_mode`` used by evaluation competitors. Also
        enables Graphviz fdp compound-edge attachment metadata.

    Returns
    -------
    Pipeline
        Pipeline implementing the FM^3 algorithm. The pipeline produces final
        node coordinates by constructing a multilevel hierarchy, initializing
        the coarsest graph, refining that level, uncoarsening with per-level
        refinement, falling back to a single-level solve when needed, and
        normalizing the result.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    effective_reference_mode = reference_mode or fidelity_mode
    initialize_state = _InitializeFMMMState(
        config=_InitializeFMMMStateConfig(
            steps=steps,
            force_model=force_model,
            galaxy_choice="lower" if effective_reference_mode else "higher",
            coarsest_init="ogdf_random" if effective_reference_mode else "fr",
            ogdf_force_scaling=effective_reference_mode,
            sum_parallel_weights=not effective_reference_mode,
        )
    )
    initialize_coarsest = _InitializeCoarsestLevel()
    refine_coarsest = _RefineCoarsestLevel()
    uncoarsen_loop = _UncoarsenLoop()
    single_level_fallback = _SingleLevelFallback()
    finalize_positions = _FinalizeFMMMPositions()

    ops: List[Op] = [
        initialize_state,
        initialize_coarsest,
        refine_coarsest,
        uncoarsen_loop,
        single_level_fallback,
        finalize_positions,
    ]
    if fidelity_mode:
        ops.append(_FdpCompoundEdgeAttachmentOp())

    return Pipeline(ops, name="fmmm_pipeline")


def _run_fmmm_pipeline_once(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    steps: int,
    seed: int,
    edge_weights: Optional[torch.Tensor],
    force_model: str,
    reference_mode: bool,
    fidelity_mode: bool,
) -> torch.Tensor:
    """Run the FMMM op pipeline once without component decomposition.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    steps : int
        Total refinement budget.
    seed : int
        Random seed.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    force_model : str
        Spring-force model.
    reference_mode : bool
        Whether to use reference coarsening and force scaling.
    fidelity_mode : bool
        Evaluation alias for ``reference_mode``.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    RuntimeError
        If the pipeline does not produce positions.
    """
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_fmmm_pipeline(
        steps=steps,
        force_model=force_model,
        reference_mode=reference_mode,
        fidelity_mode=fidelity_mode,
    ).apply(
        problem,
        state,
        ctx,
    )
    if final_state.pos is None:
        raise RuntimeError("FM^3 pipeline did not produce final positions.")
    return final_state.pos


def _layout_fmmm_fidelity_components(
    edge_index: torch.Tensor,
    components: list[list[int]],
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    steps: int,
    seed: int,
    edge_weights: Optional[torch.Tensor],
    force_model: str,
    reference_mode: bool,
    fidelity_mode: bool,
) -> torch.Tensor:
    """Lay out weak components independently and pack them like Graphviz fdp.

    Parameters
    ----------
    edge_index : torch.Tensor
        Parent edge tensor with shape ``[2, E]``.
    components : list[list[int]]
        Weak components in parent node order.
    num_nodes : int
        Total parent node count.
    node_sizes : torch.Tensor, optional
        Optional parent node sizes with shape ``[N, 2]``.
    steps : int
        Total refinement budget for each component solve.
    seed : int
        Random seed reused for each component, matching fdp_tLayout seeding.
    edge_weights : torch.Tensor, optional
        Optional parent edge weights with shape ``[E]``.
    force_model : str
        Spring-force model.
    reference_mode : bool
        Whether to use reference coarsening and force scaling.
    fidelity_mode : bool
        Evaluation alias for ``reference_mode``.

    Returns
    -------
    torch.Tensor
        Packed parent coordinates with shape ``[N, 2]``.
    """
    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    component_positions: list[torch.Tensor] = []
    boxes: list[tuple[float, float, float, float]] = []
    for component in components:
        local_edges, local_weights = _slice_component_edges(edge_index, edge_weights, component)
        local_sizes = node_sizes[component] if node_sizes is not None else None
        local_pos = _run_fmmm_pipeline_once(
            edge_index=local_edges,
            num_nodes=len(component),
            node_sizes=local_sizes,
            steps=steps,
            seed=seed,
            edge_weights=local_weights,
            force_model=force_model,
            reference_mode=reference_mode,
            fidelity_mode=fidelity_mode,
        )
        component_positions.append(local_pos)
        boxes.append(_component_box(local_pos, local_sizes))

    offsets = _graphviz_tile_pack_offsets(boxes)
    dtype = component_positions[0].dtype
    packed = torch.zeros((num_nodes, 2), dtype=dtype, device=device)
    for component, local_pos, offset in zip(components, component_positions, offsets):
        offset_tensor = torch.tensor(offset, dtype=dtype, device=local_pos.device)
        packed[component] = (local_pos + offset_tensor).to(device=device, dtype=dtype)
    return _translate_packed_components_to_origin(packed, node_sizes).to(dtype=torch.float32)


def layout_fmmm_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 200,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    force_model: str = "ogdf_new",
    reference_mode: bool = False,
    fidelity_mode: bool = False,
    clusters: Optional[Mapping[str, Sequence[int]]] = None,
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    **kwargs: Any,
) -> torch.Tensor:
    """Run the FM^3 pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used for extent
        calculation and output-device selection.
    steps : int, default=200
        Total refinement budget distributed across hierarchy levels.
    seed : int, default=42
        Random seed for coarsening, coarse initialization, and prolongation
        jitter.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    force_model : str, default="ogdf_new"
        Spring-force model for edge attraction.
    reference_mode : bool, default=False
        Use OGDF-aligned reference behavior for algorithm fidelity runs.
    fidelity_mode : bool, default=False
        Alias for ``reference_mode`` used by evaluation competitors.
    clusters : Mapping[str, Sequence[int]], optional
        Cluster membership. Only used when ``fidelity_mode`` is enabled.
    cluster_parents : Mapping[str, str | None], optional
        Cluster parent mapping. Only used when ``fidelity_mode`` is enabled.
    **kwargs : Any
        Ignored compatibility keywords from generic layout dispatch.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``steps``, ``edge_weights``, or ``force_model`` are
        invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if force_model not in {"ogdf_new", "fr"}:
        raise ValueError("force_model must be either 'ogdf_new' or 'fr'.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge_count {edge_index.shape[1]}"
            )
    del kwargs

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)
    if fidelity_mode and clusters:
        return graphviz_fdp_fidelity(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            steps=steps,
            seed=seed,
            clusters=clusters,
            cluster_parents=cluster_parents,
        )

    effective_reference_mode = reference_mode or fidelity_mode
    if effective_reference_mode:
        components = _weak_components(edge_index=edge_index, num_nodes=num_nodes)
        if len(components) > 1:
            return _layout_fmmm_fidelity_components(
                edge_index=edge_index,
                components=components,
                num_nodes=num_nodes,
                node_sizes=node_sizes,
                steps=steps,
                seed=seed,
                edge_weights=edge_weights,
                force_model=force_model,
                reference_mode=reference_mode,
                fidelity_mode=fidelity_mode,
            )

    return _run_fmmm_pipeline_once(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        steps=steps,
        seed=seed,
        edge_weights=edge_weights,
        force_model=force_model,
        reference_mode=reference_mode,
        fidelity_mode=fidelity_mode,
    )


__all__ = ["build_fmmm_pipeline", "graphviz_fdp_fidelity", "layout_fmmm_pipeline"]
