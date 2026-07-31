"""Deterministic Graphviz osage-style packing pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, ClassVar, Dict, List, Mapping, Optional, Tuple

import numpy as np
import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.networkx_simple import graphviz_array_rects, graphviz_osage_array_positions
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op
from dagua.utils import collect_cluster_leaves

_OSAGE_DEFAULT_NODE_SIZE = (54.0, 36.0)
_OSAGE_DEFAULT_EMPTY_CLUSTER_SIZE = 18.0
_OSAGE_CLUSTER_LABEL_HEIGHT = 23.0
_OSAGE_CLUSTER_LABEL_XPAD = 16.0
_OSAGE_CLUSTER_LABEL_WIDTH_PER_CHAR = 7.0

_Rect = Tuple[float, float, float, float]


@dataclass
class _OsageClusterLayout:
    """Intermediate normalized cluster layout.

    Parameters
    ----------
    bbox : tuple[float, float, float, float]
        Normalized cluster bounding box as ``(0, 0, width, height)``.
    direct_positions : dict[int, tuple[float, float]]
        Direct child-node centers in the cluster-local y-up coordinate frame.
    child_bboxes : dict[str, tuple[float, float, float, float]]
        Direct child-cluster bounding boxes in the cluster-local y-up frame.
    """

    bbox: _Rect
    direct_positions: Dict[int, Tuple[float, float]]
    child_bboxes: Dict[str, _Rect]


@register_op
@dataclass
class OsageArrayPackLayout(Op):
    """Assign Graphviz osage-style array-packed coordinates.

    Parameters
    ----------
    separation : float, default=4.0
        Point gap between packed node boxes.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    """

    separation: float = 4.0
    dtype: torch.dtype = torch.float64

    name: ClassVar[str] = "graphviz_osage_array_layout"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[tuple[str, ...]] = ("N", "node_sizes", "clusters")
    writes: ClassVar[tuple[str, ...]] = ("pos", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Assign final osage-style coordinates to ``state.pos``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs, including node count ``N`` and optional
            ``node_sizes`` with shape ``[N, 2]``.
        state : SolveState
            Mutable solve state receiving ``pos`` and provenance extras.
        ctx : RuntimeContext
            Runtime context; accepted for composable-op API consistency.

        Returns
        -------
        SolveState
            State with osage-style positions and metadata populated.
        """
        del ctx
        device = layout_device(problem.edge_index, problem.node_sizes)
        if problem.clusters:
            state.pos = graphviz_osage_cluster_positions(
                num_nodes=problem.num_nodes,
                node_sizes=problem.node_sizes,
                clusters=problem.clusters,
                cluster_parents=problem.cluster_parents,
                cluster_labels=problem.cluster_labels,
                dtype=self.dtype,
                device=device,
                separation=self.separation,
            )
        else:
            state.pos = graphviz_osage_array_positions(
                num_nodes=problem.num_nodes,
                node_sizes=problem.node_sizes,
                dtype=self.dtype,
                device=device,
                separation=self.separation,
            )
        state.extras["osage"] = {
            "packmode": "array",
            "separation": self.separation,
            "compound": bool(problem.clusters),
        }
        return state


def _node_sizes_array(num_nodes: int, node_sizes: Optional[torch.Tensor]) -> np.ndarray:
    """Return node boxes in point units for osage packing.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None
        Optional node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    numpy.ndarray
        Size array with shape ``[N, 2]``.
    """
    if node_sizes is None:
        return np.tile(np.array([_OSAGE_DEFAULT_NODE_SIZE], dtype=np.float64), (num_nodes, 1))
    return np.round(node_sizes.detach().cpu().to(dtype=torch.float64).numpy())


def _cluster_members(clusters: Mapping[str, Any], cluster_name: str) -> List[int]:
    """Return flattened members for one declared cluster.

    Parameters
    ----------
    clusters : mapping[str, Any]
        Cluster membership payload from ``DaguaGraph``.
    cluster_name : str
        Cluster name to inspect.

    Returns
    -------
    list[int]
        Flattened member node indices in declaration order.
    """
    members = clusters.get(cluster_name, [])
    if isinstance(members, dict):
        return [int(index) for index in collect_cluster_leaves(members)]
    return [int(index) for index in members]


def _cluster_children(
    clusters: Mapping[str, Any],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
) -> Dict[Optional[str], List[str]]:
    """Build Graphviz-emission-order child clusters by parent.

    Parameters
    ----------
    clusters : mapping[str, Any]
        Cluster membership payload.
    cluster_parents : mapping[str, str | None] | None
        Optional parent lookup.

    Returns
    -------
    dict[str | None, list[str]]
        Parent-to-child cluster mapping. Child names are sorted to match the
        DOT serializer used by the Graphviz osage reference adapter.
    """
    parents = {} if cluster_parents is None else cluster_parents
    children: Dict[Optional[str], List[str]] = {}
    for cluster_name in sorted(str(name) for name in clusters):
        parent = parents.get(cluster_name)
        if parent not in clusters:
            parent = None
        children.setdefault(parent, []).append(cluster_name)
    return children


def _aggregate_members(
    cluster_name: str,
    clusters: Mapping[str, Any],
    children_by_parent: Mapping[Optional[str], List[str]],
) -> List[int]:
    """Return all nodes emitted under one cluster subtree.

    Parameters
    ----------
    cluster_name : str
        Cluster root to aggregate.
    clusters : mapping[str, Any]
        Cluster membership payload.
    children_by_parent : mapping[str | None, list[str]]
        Parent-to-child cluster mapping.

    Returns
    -------
    list[int]
        Node indices covered by the cluster and all descendant clusters.
    """
    seen: set[int] = set()
    ordered: List[int] = []
    for node in _cluster_members(clusters=clusters, cluster_name=cluster_name):
        if node not in seen:
            seen.add(node)
            ordered.append(node)
    for child_name in children_by_parent.get(cluster_name, []):
        for node in _aggregate_members(
            cluster_name=child_name,
            clusters=clusters,
            children_by_parent=children_by_parent,
        ):
            if node not in seen:
                seen.add(node)
                ordered.append(node)
    return ordered


def _direct_cluster_nodes(
    cluster_name: str,
    clusters: Mapping[str, Any],
    children_by_parent: Mapping[Optional[str], List[str]],
) -> List[int]:
    """Return nodes directly emitted inside one cluster block.

    Parameters
    ----------
    cluster_name : str
        Cluster to inspect.
    clusters : mapping[str, Any]
        Cluster membership payload.
    children_by_parent : mapping[str | None, list[str]]
        Parent-to-child cluster mapping.

    Returns
    -------
    list[int]
        Direct child nodes, excluding descendants of nested child clusters.
    """
    nested_nodes: set[int] = set()
    for child_name in children_by_parent.get(cluster_name, []):
        nested_nodes.update(
            _aggregate_members(
                cluster_name=child_name,
                clusters=clusters,
                children_by_parent=children_by_parent,
            )
        )
    return [
        node
        for node in _cluster_members(clusters=clusters, cluster_name=cluster_name)
        if node not in nested_nodes
    ]


def _label_size(
    cluster_name: str,
    cluster_labels: Optional[Mapping[str, str]],
    graphviz_cluster_label_sizes: Optional[Mapping[str, Tuple[float, float]]],
    graphviz_cluster_label_widths: Optional[Mapping[str, float]],
) -> Optional[Tuple[float, float]]:
    """Return padded Graphviz cluster-label dimensions when a label is present.

    Parameters
    ----------
    cluster_name : str
        Cluster whose label band is being sized.
    cluster_labels : mapping[str, str] | None
        Optional labels. Missing labels follow the reference serializer and use
        the cluster name; an explicit empty string disables the label.
    graphviz_cluster_label_sizes : mapping[str, tuple[float, float]] | None
        Optional exact padded label sizes in points.
    graphviz_cluster_label_widths : mapping[str, float] | None
        Optional exact padded label widths in points.

    Returns
    -------
    tuple[float, float] | None
        Padded label width and height in points, or ``None`` for no label.
    """
    label = str(cluster_labels.get(cluster_name, cluster_name)) if cluster_labels else cluster_name
    if label == "":
        return None
    if graphviz_cluster_label_sizes is not None and cluster_name in graphviz_cluster_label_sizes:
        width, height = graphviz_cluster_label_sizes[cluster_name]
        return (float(width), float(height))
    if graphviz_cluster_label_widths is not None and cluster_name in graphviz_cluster_label_widths:
        return (float(graphviz_cluster_label_widths[cluster_name]), _OSAGE_CLUSTER_LABEL_HEIGHT)
    return (
        float(len(label)) * _OSAGE_CLUSTER_LABEL_WIDTH_PER_CHAR + _OSAGE_CLUSTER_LABEL_XPAD,
        _OSAGE_CLUSTER_LABEL_HEIGHT,
    )


def _rect_size(rect: _Rect) -> Tuple[float, float]:
    """Return rectangle width and height.

    Parameters
    ----------
    rect : tuple[float, float, float, float]
        Rectangle as ``(llx, lly, urx, ury)``.

    Returns
    -------
    tuple[float, float]
        Rectangle width and height.
    """
    return (float(rect[2]) - float(rect[0]), float(rect[3]) - float(rect[1]))


def _bbox_from_rects(rects: List[_Rect]) -> _Rect:
    """Return the bounding box enclosing input rectangles.

    Parameters
    ----------
    rects : list[tuple[float, float, float, float]]
        Rectangles to enclose.

    Returns
    -------
    tuple[float, float, float, float]
        Enclosing bounding box.
    """
    if not rects:
        return (0.0, 0.0, 0.0, 0.0)
    return (
        min(rect[0] for rect in rects),
        min(rect[1] for rect in rects),
        max(rect[2] for rect in rects),
        max(rect[3] for rect in rects),
    )


def _normalize_rect(rect: _Rect, origin: Tuple[float, float]) -> _Rect:
    """Translate one rectangle by subtracting an origin.

    Parameters
    ----------
    rect : tuple[float, float, float, float]
        Rectangle to translate.
    origin : tuple[float, float]
        Origin ``(x, y)`` to subtract.

    Returns
    -------
    tuple[float, float, float, float]
        Translated rectangle.
    """
    return (
        float(rect[0]) - origin[0],
        float(rect[1]) - origin[1],
        float(rect[2]) - origin[0],
        float(rect[3]) - origin[1],
    )


def _layout_cluster(
    cluster_name: Optional[str],
    depth: int,
    num_nodes: int,
    node_sizes: np.ndarray,
    clusters: Mapping[str, Any],
    children_by_parent: Mapping[Optional[str], List[str]],
    cluster_labels: Optional[Mapping[str, str]],
    graphviz_cluster_label_sizes: Optional[Mapping[str, Tuple[float, float]]],
    graphviz_cluster_label_widths: Optional[Mapping[str, float]],
    separation: float,
) -> _OsageClusterLayout:
    """Recursively pack one osage cluster or the root graph.

    Parameters
    ----------
    cluster_name : str | None
        Cluster name, or ``None`` for the root graph.
    depth : int
        Graphviz osage recursion depth. Non-root clusters receive a half-margin
        border and label band.
    num_nodes : int
        Number of graph nodes.
    node_sizes : numpy.ndarray
        Node-size array with shape ``[N, 2]``.
    clusters : mapping[str, Any]
        Cluster membership payload.
    children_by_parent : mapping[str | None, list[str]]
        Parent-to-child cluster mapping.
    cluster_labels : mapping[str, str] | None
        Optional cluster label text.
    graphviz_cluster_label_sizes : mapping[str, tuple[float, float]] | None
        Optional exact padded cluster-label dimensions.
    graphviz_cluster_label_widths : mapping[str, float] | None
        Optional exact padded cluster-label widths.
    separation : float
        Graphviz pack margin in points.

    Returns
    -------
    _OsageClusterLayout
        Normalized cluster layout matching Graphviz 7.0.5 ``layout()``.
    """
    child_names = children_by_parent.get(cluster_name, [])
    child_layouts = {
        child_name: _layout_cluster(
            cluster_name=child_name,
            depth=depth + 1,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            clusters=clusters,
            children_by_parent=children_by_parent,
            cluster_labels=cluster_labels,
            graphviz_cluster_label_sizes=graphviz_cluster_label_sizes,
            graphviz_cluster_label_widths=graphviz_cluster_label_widths,
            separation=separation,
        )
        for child_name in child_names
    }

    if cluster_name is None:
        covered_nodes: set[int] = set()
        for child_name in child_names:
            covered_nodes.update(
                _aggregate_members(
                    cluster_name=child_name,
                    clusters=clusters,
                    children_by_parent=children_by_parent,
                )
            )
        direct_nodes = [node for node in range(num_nodes) if node not in covered_nodes]
    else:
        direct_nodes = _direct_cluster_nodes(
            cluster_name=cluster_name,
            clusters=clusters,
            children_by_parent=children_by_parent,
        )

    rects: List[_Rect] = []
    rect_kinds: List[Tuple[str, str | int]] = []
    for child_name in child_names:
        rects.append(child_layouts[child_name].bbox)
        rect_kinds.append(("cluster", child_name))
    for node in direct_nodes:
        width = float(node_sizes[node, 0])
        height = float(node_sizes[node, 1])
        rects.append((0.0, 0.0, width, height))
        rect_kinds.append(("node", node))

    if (
        not rects
        and cluster_name is not None
        and _label_size(
            cluster_name=cluster_name,
            cluster_labels=cluster_labels,
            graphviz_cluster_label_sizes=graphviz_cluster_label_sizes,
            graphviz_cluster_label_widths=graphviz_cluster_label_widths,
        )
        is None
    ):
        return _OsageClusterLayout(
            bbox=(0.0, 0.0, _OSAGE_DEFAULT_EMPTY_CLUSTER_SIZE, _OSAGE_DEFAULT_EMPTY_CLUSTER_SIZE),
            direct_positions={},
            child_bboxes={},
        )

    centers = graphviz_array_rects(rects=rects, margin=separation)
    placed_rects: List[_Rect] = []
    direct_positions: Dict[int, Tuple[float, float]] = {}
    child_bboxes: Dict[str, _Rect] = {}
    for index, rect in enumerate(rects):
        width, height = _rect_size(rect)
        center_x = float(centers[index, 0])
        center_y = float(centers[index, 1])
        placed = (
            center_x - width / 2.0,
            center_y - height / 2.0,
            center_x + width / 2.0,
            center_y + height / 2.0,
        )
        placed_rects.append(placed)
        kind, value = rect_kinds[index]
        if kind == "cluster":
            child_bboxes[str(value)] = placed
        else:
            direct_positions[int(value)] = (center_x, center_y)

    rootbb = _bbox_from_rects(placed_rects)
    if cluster_name is not None:
        label_size = _label_size(
            cluster_name=cluster_name,
            cluster_labels=cluster_labels,
            graphviz_cluster_label_sizes=graphviz_cluster_label_sizes,
            graphviz_cluster_label_widths=graphviz_cluster_label_widths,
        )
        if label_size is not None:
            label_width, label_height = label_size
            if not rects:
                rootbb = (0.0, 0.0, float(label_width), float(label_height))
            width_delta = float(label_width) - (rootbb[2] - rootbb[0])
            if width_delta > 0.0:
                half_delta = width_delta / 2.0
                rootbb = (
                    rootbb[0] - half_delta,
                    rootbb[1],
                    rootbb[2] + half_delta,
                    rootbb[3],
                )
            top_border = float(label_height)
        else:
            top_border = 0.0
        margin = float(separation) / 2.0 if depth > 0 else 0.0
        rootbb = (
            rootbb[0] - margin,
            rootbb[1] - margin,
            rootbb[2] + margin,
            rootbb[3] + margin + top_border,
        )

    origin = (rootbb[0], rootbb[1])
    normalized_direct = {
        node: (position[0] - origin[0], position[1] - origin[1])
        for node, position in direct_positions.items()
    }
    normalized_children = {
        name: _normalize_rect(rect=bbox, origin=origin) for name, bbox in child_bboxes.items()
    }
    normalized_bbox = _normalize_rect(rect=rootbb, origin=origin)
    return _OsageClusterLayout(
        bbox=normalized_bbox,
        direct_positions=normalized_direct,
        child_bboxes=normalized_children,
    )


def _reposition_cluster(
    cluster_name: Optional[str],
    origin: Tuple[float, float],
    layout_by_cluster: Mapping[Optional[str], _OsageClusterLayout],
    children_by_parent: Mapping[Optional[str], List[str]],
    positions: np.ndarray,
) -> None:
    """Apply Graphviz osage's top-down ``reposition`` translation.

    Parameters
    ----------
    cluster_name : str | None
        Cluster currently being translated, or ``None`` for the root graph.
    origin : tuple[float, float]
        Absolute lower-left origin for this cluster.
    layout_by_cluster : mapping[str | None, _OsageClusterLayout]
        Normalized layouts keyed by cluster name.
    children_by_parent : mapping[str | None, list[str]]
        Parent-to-child cluster mapping.
    positions : numpy.ndarray
        Mutable y-up position array with shape ``[N, 2]``.

    Returns
    -------
    None
        The function mutates ``positions`` in place.
    """
    layout = layout_by_cluster[cluster_name]
    for node, position in layout.direct_positions.items():
        positions[node, 0] = origin[0] + position[0]
        positions[node, 1] = origin[1] + position[1]
    for child_name in children_by_parent.get(cluster_name, []):
        child_bbox = layout.child_bboxes[child_name]
        child_origin = (origin[0] + child_bbox[0], origin[1] + child_bbox[1])
        _reposition_cluster(
            cluster_name=child_name,
            origin=child_origin,
            layout_by_cluster=layout_by_cluster,
            children_by_parent=children_by_parent,
            positions=positions,
        )


def _index_cluster_layouts(
    root: _OsageClusterLayout,
    **layout_kwargs: Any,
) -> Dict[Optional[str], _OsageClusterLayout]:
    """Build a lookup of normalized layouts for root and all clusters.

    Parameters
    ----------
    root : _OsageClusterLayout
        Precomputed root layout.
    **layout_kwargs : Any
        Keyword arguments forwarded to :func:`_layout_cluster`.

    Returns
    -------
    dict[str | None, _OsageClusterLayout]
        Layout lookup keyed by ``None`` for root and by cluster name.
    """
    children_by_parent = layout_kwargs["children_by_parent"]
    layouts: Dict[Optional[str], _OsageClusterLayout] = {None: root}
    for child_name in children_by_parent.get(None, []):
        _collect_cluster_layouts(
            cluster_name=child_name,
            layouts=layouts,
            **layout_kwargs,
        )
    return layouts


def _collect_cluster_layouts(
    cluster_name: str,
    layouts: Dict[Optional[str], _OsageClusterLayout],
    **layout_kwargs: Any,
) -> None:
    """Recursively collect normalized layouts for one cluster subtree.

    Parameters
    ----------
    cluster_name : str
        Cluster to collect.
    layouts : dict[str | None, _OsageClusterLayout]
        Mutable layout lookup.
    **layout_kwargs : Any
        Keyword arguments forwarded to :func:`_layout_cluster`.

    Returns
    -------
    None
        ``layouts`` is mutated in place.
    """
    children_by_parent = layout_kwargs["children_by_parent"]
    layout = _layout_cluster(cluster_name=cluster_name, **layout_kwargs)
    layouts[cluster_name] = layout
    for child_name in children_by_parent.get(cluster_name, []):
        _collect_cluster_layouts(
            cluster_name=child_name,
            layouts=layouts,
            **layout_kwargs,
        )


def graphviz_osage_cluster_positions(
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    clusters: Mapping[str, Any],
    cluster_parents: Optional[Mapping[str, Optional[str]]] = None,
    cluster_labels: Optional[Mapping[str, str]] = None,
    graphviz_cluster_label_sizes: Optional[Mapping[str, Tuple[float, float]]] = None,
    graphviz_cluster_label_widths: Optional[Mapping[str, float]] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
    separation: float = 4.0,
) -> torch.Tensor:
    """Return Graphviz osage-style compound cluster-packed node centers.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None
        Optional node-size tensor with shape ``[N, 2]`` in points.
    clusters : mapping[str, Any]
        Cluster membership metadata.
    cluster_parents : mapping[str, str | None] | None, optional
        Optional cluster parent lookup.
    cluster_labels : mapping[str, str] | None, optional
        Optional cluster label text. Missing labels use the cluster name, as in
        the Graphviz reference DOT serializer.
    graphviz_cluster_label_sizes : mapping[str, tuple[float, float]] | None, optional
        Optional exact padded cluster label sizes in points.
    graphviz_cluster_label_widths : mapping[str, float] | None, optional
        Optional exact padded cluster label widths in points.
    dtype : torch.dtype, default=torch.float64
        Output dtype.
    device : torch.device | None, optional
        Output device. ``None`` uses CPU.
    separation : float, default=4.0
        Graphviz pack margin in points.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]`` in Dagua's y-down frame.
    """
    out_device = torch.device("cpu") if device is None else device
    if num_nodes <= 0:
        return torch.empty((0, 2), dtype=dtype, device=out_device)
    if not clusters:
        return graphviz_osage_array_positions(
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            dtype=dtype,
            device=out_device,
            separation=separation,
        )

    node_size_array = _node_sizes_array(num_nodes=num_nodes, node_sizes=node_sizes)
    normalized_clusters = {str(name): value for name, value in clusters.items()}
    normalized_parents = (
        None
        if cluster_parents is None
        else {
            str(name): (None if parent is None else str(parent))
            for name, parent in cluster_parents.items()
        }
    )
    children_by_parent = _cluster_children(
        clusters=normalized_clusters,
        cluster_parents=normalized_parents,
    )
    layout_kwargs = dict(
        depth=0,
        num_nodes=num_nodes,
        node_sizes=node_size_array,
        clusters=normalized_clusters,
        children_by_parent=children_by_parent,
        cluster_labels=cluster_labels,
        graphviz_cluster_label_sizes=graphviz_cluster_label_sizes,
        graphviz_cluster_label_widths=graphviz_cluster_label_widths,
        separation=separation,
    )
    root_layout = _layout_cluster(cluster_name=None, **layout_kwargs)
    cluster_layout_kwargs = dict(layout_kwargs)
    cluster_layout_kwargs["depth"] = 1
    layout_by_cluster = _index_cluster_layouts(
        root=root_layout,
        **cluster_layout_kwargs,
    )

    positions = np.zeros((num_nodes, 2), dtype=np.float64)
    _reposition_cluster(
        cluster_name=None,
        origin=(0.0, 0.0),
        layout_by_cluster=layout_by_cluster,
        children_by_parent=children_by_parent,
        positions=positions,
    )
    positions[:, 1] *= -1.0
    return torch.as_tensor(positions, dtype=dtype, device=out_device)


def build_osage_pipeline(
    scale: float = 1.0,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> Pipeline:
    """Build the deterministic osage-style packing pipeline.

    Parameters
    ----------
    scale : float, default=1.0
        Accepted for API consistency; Graphviz osage uses node-size units.
    fidelity_dtype : torch.dtype | None, optional
        Output dtype for direct fidelity checks.

    Returns
    -------
    Pipeline
        Single-stage composable coordinate pipeline.
    """
    del scale
    return Pipeline(
        [OsageArrayPackLayout(dtype=torch.float64 if fidelity_dtype is None else fidelity_dtype)],
        name="osage_pipeline",
    )


def layout_osage_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: Optional[int] = 42,
    edge_weights: Optional[torch.Tensor] = None,
    scale: float = 1.0,
    fidelity_dtype: Optional[torch.dtype] = None,
    clusters: Optional[Dict[str, Any]] = None,
    cluster_parents: Optional[Dict[str, Optional[str]]] = None,
    cluster_labels: Optional[Dict[str, str]] = None,
    graphviz_cluster_label_sizes: Optional[Dict[str, Tuple[float, float]]] = None,
    graphviz_cluster_label_widths: Optional[Dict[str, float]] = None,
) -> torch.Tensor:
    """Run deterministic osage-style node packing.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor | None, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int | None, default=42
        Accepted for API consistency; osage layout is deterministic.
    edge_weights : torch.Tensor | None, optional
        Accepted for API consistency; osage layout ignores weights.
    scale : float, default=1.0
        Layout half-width.
    fidelity_dtype : torch.dtype | None, optional
        Output dtype for direct fidelity checks.
    clusters : dict[str, Any] | None, optional
        Cluster membership metadata.
    cluster_parents : dict[str, str | None] | None, optional
        Cluster parent metadata.
    cluster_labels : dict[str, str] | None, optional
        Cluster label text.
    graphviz_cluster_label_sizes : dict[str, tuple[float, float]] | None, optional
        Exact padded Graphviz cluster-label sizes in points.
    graphviz_cluster_label_widths : dict[str, float] | None, optional
        Exact padded Graphviz cluster-label widths in points.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    del seed, edge_weights
    if clusters:
        return graphviz_osage_cluster_positions(
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            clusters=clusters,
            cluster_parents=cluster_parents,
            cluster_labels=cluster_labels,
            graphviz_cluster_label_sizes=graphviz_cluster_label_sizes,
            graphviz_cluster_label_widths=graphviz_cluster_label_widths,
            dtype=torch.float64 if fidelity_dtype is None else fidelity_dtype,
            separation=4.0,
        )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        clusters=clusters,
        cluster_parents=cluster_parents,
        cluster_labels=cluster_labels,
    )
    state = build_osage_pipeline(scale=scale, fidelity_dtype=fidelity_dtype).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device="cpu")),
    )
    if state.pos is None:
        raise RuntimeError("Osage pipeline did not produce positions.")
    return state.pos


__all__ = [
    "OsageArrayPackLayout",
    "build_osage_pipeline",
    "graphviz_osage_cluster_positions",
    "layout_osage_pipeline",
]
