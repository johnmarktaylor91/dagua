"""Pure cluster hierarchy and geometry helpers for layout operations."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Optional, Sequence, Tuple

import torch


@dataclass(frozen=True)
class ClusterTree:
    """Tree representation of cluster hierarchy.

    Parameters
    ----------
    parents : Mapping[str, Optional[str]]
        Mapping from cluster name to parent cluster name. Root clusters map to
        ``None``.
    leaves_per_cluster : Mapping[str, frozenset[int]]
        Mapping from cluster name to leaf node indices that bottom out in this
        cluster and not in any deeper child cluster within this branch.
    descendants_per_cluster : Mapping[str, frozenset[int]]
        Mapping from cluster name to all leaf node indices reachable through
        this cluster. This matches dagua's existing flat membership convention.
    children_per_cluster : Mapping[str, tuple[str, ...]]
        Mapping from cluster name to immediate child cluster names.
    roots : tuple[str, ...]
        Cluster names whose parent is ``None``.

    Notes
    -----
    Construct from ``(clusters, cluster_parents)`` via
    :meth:`ClusterTree.from_flat_membership`. Both arguments use existing dagua
    conventions: ``clusters`` maps names to flat descendant leaf indices, and
    ``cluster_parents`` maps names to an optional parent cluster name.
    """

    parents: Mapping[str, Optional[str]]
    leaves_per_cluster: Mapping[str, frozenset[int]]
    descendants_per_cluster: Mapping[str, frozenset[int]]
    children_per_cluster: Mapping[str, Tuple[str, ...]]
    roots: Tuple[str, ...]

    @classmethod
    def from_flat_membership(
        cls,
        clusters: Mapping[str, Sequence[int]],
        cluster_parents: Mapping[str, Optional[str]],
    ) -> "ClusterTree":
        """Build a cluster tree from dagua flat cluster membership.

        Parameters
        ----------
        clusters : Mapping[str, Sequence[int]]
            Mapping from cluster name to member node indices. Existing graph
            builders may store either direct members or all descendants; child
            membership is unioned into each parent so both conventions produce
            a flat descendant lookup.
        cluster_parents : Mapping[str, Optional[str]]
            Mapping from cluster name to parent cluster name. Missing entries
            and parents outside ``clusters`` are treated as root membership.

        Returns
        -------
        ClusterTree
            Immutable hierarchy with child, root, leaf-only, and descendant
            membership lookup tables.
        """
        cluster_names = tuple(sorted(str(name) for name in clusters))
        declared_members = {
            str(name): frozenset(int(index) for index in members)
            for name, members in clusters.items()
        }
        parents = {}
        for name in cluster_names:
            parent_name = cluster_parents.get(name)
            parents[name] = parent_name if parent_name in declared_members else None
        children_lists: dict[str, list[str]] = {name: [] for name in cluster_names}
        for name in cluster_names:
            parent = parents[name]
            if parent is not None:
                children_lists[parent].append(name)

        children = {name: tuple(sorted(children_lists[name])) for name in cluster_names}
        expanded_descendants: dict[str, frozenset[int]] = {}

        def expand_descendants(name: str) -> frozenset[int]:
            """Return declared members plus all child descendants.

            Parameters
            ----------
            name : str
                Cluster name to expand.

            Returns
            -------
            frozenset[int]
                Full descendant node set for ``name``.
            """
            if name in expanded_descendants:
                return expanded_descendants[name]
            merged = set(declared_members[name])
            for child_name in children[name]:
                merged.update(expand_descendants(child_name))
            expanded = frozenset(merged)
            expanded_descendants[name] = expanded
            return expanded

        descendants = {name: expand_descendants(name) for name in cluster_names}
        leaves: dict[str, frozenset[int]] = {}
        for name in cluster_names:
            child_descendants: set[int] = set()
            for child_name in children[name]:
                child_descendants.update(descendants[child_name])
            leaves[name] = frozenset(declared_members[name].difference(child_descendants))

        roots = tuple(sorted(name for name in cluster_names if parents[name] is None))
        return cls(
            parents=parents,
            leaves_per_cluster=leaves,
            descendants_per_cluster=descendants,
            children_per_cluster=children,
            roots=roots,
        )

    def bottom_up_order(self) -> Tuple[str, ...]:
        """Return cluster names in child-before-parent order.

        Returns
        -------
        tuple[str, ...]
            Cluster names ordered so every child appears before its parent.
        """
        ordered: list[str] = []
        stack: list[tuple[str, bool]] = [(root_name, False) for root_name in reversed(self.roots)]
        while stack:
            name, children_done = stack.pop()
            if children_done:
                ordered.append(name)
                continue
            stack.append((name, True))
            for child_name in reversed(self.children_per_cluster[name]):
                stack.append((child_name, False))
        return tuple(ordered)

    def top_down_order(self) -> Tuple[str, ...]:
        """Return cluster names in parent-before-child order.

        Returns
        -------
        tuple[str, ...]
            Cluster names ordered so every parent appears before descendants.
        """
        ordered: list[str] = []
        stack = list(reversed(self.roots))
        while stack:
            name = stack.pop()
            ordered.append(name)
            for child_name in reversed(self.children_per_cluster[name]):
                stack.append(child_name)
        return tuple(ordered)


@dataclass(frozen=True)
class ClusterLabelMetrics:
    """Measured cluster label dimensions.

    Parameters
    ----------
    label_width_pt : float
        Label width in the same coordinate units as the member positions.
    label_height_pt : float
        Label height in the same coordinate units as the member positions.
    """

    label_width_pt: float
    label_height_pt: float


@dataclass(frozen=True)
class ClusterPlacementBox:
    """Placement-time cluster bounding box metadata.

    Parameters
    ----------
    width : float
        Full cluster footprint width.
    height : float
        Full cluster footprint height.
    anchor_offset : tuple[float, float]
        ``(dx, dy)`` from the centroid of ``inner_positions`` to the cluster
        bbox center.
    inner_bbox : tuple[float, float, float, float]
        Content bounds as ``(x_min, y_min, x_max, y_max)``.
    label_band_y_extent : tuple[float, float]
        Top and bottom ``y`` coordinates of the reserved top label band.
    """

    width: float
    height: float
    anchor_offset: Tuple[float, float]
    inner_bbox: Tuple[float, float, float, float]
    label_band_y_extent: Tuple[float, float]


@dataclass(frozen=True)
class ClusterProfileBox:
    """Derived axis-aligned geometry for one rendered cluster.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        Padded cluster box as ``(x_min, y_min, x_max, y_max)``.
    inner_bounds : tuple[float, float, float, float]
        Member content bounds before cluster padding and label band.
    label_bounds : Optional[tuple[float, float, float, float]]
        Reserved cluster-label band as ``(x_min, y_min, x_max, y_max)``, or
        ``None`` when no explicit label exists for this cluster.
    descendants : frozenset[int]
        Leaf node indices used to derive this cluster box.
    raw_leaf_bounds : tuple[float, float, float, float]
        Descendant leaf-node envelope before nested padding is introduced.
    """

    bounds: Tuple[float, float, float, float]
    inner_bounds: Tuple[float, float, float, float]
    label_bounds: Optional[Tuple[float, float, float, float]]
    descendants: frozenset[int]
    raw_leaf_bounds: Tuple[float, float, float, float]


@dataclass(frozen=True)
class ClusterGeometryProfile:
    """Read-only cluster geometry derived from placed nodes.

    Parameters
    ----------
    tree : ClusterTree
        Nested cluster membership tree.
    cluster_names : tuple[str, ...]
        Deterministic top-down cluster order.
    boxes : Mapping[str, ClusterProfileBox]
        Derived padded boxes, keyed by cluster name.
    node_bounds : tuple[tuple[float, float, float, float], ...]
        Node AABBs as ``(x_min, y_min, x_max, y_max)``.
    sibling_pairs : tuple[tuple[str, str], ...]
        Same-parent cluster pairs in deterministic order.

    Notes
    -----
    Boxes are derived from member positions and sizes via
    :func:`compute_cluster_placement_bbox`; they are never independent layout
    variables.
    """

    tree: ClusterTree
    cluster_names: Tuple[str, ...]
    boxes: Mapping[str, ClusterProfileBox]
    node_bounds: Tuple[Tuple[float, float, float, float], ...]
    sibling_pairs: Tuple[Tuple[str, str], ...]


def compute_cluster_placement_bbox(
    inner_positions: torch.Tensor,
    inner_sizes: torch.Tensor,
    label_metrics: ClusterLabelMetrics,
    side_padding_pt: float,
    label_band_pt: float,
    extra_top_band_pt: float = 0.0,
) -> ClusterPlacementBox:
    """Compute the placement-time bbox for a cluster from placed members.

    Parameters
    ----------
    inner_positions : torch.Tensor
        Member center positions with shape ``[N_inner, 2]``.
    inner_sizes : torch.Tensor
        Member box sizes as ``(width, height)`` with shape ``[N_inner, 2]``.
    label_metrics : ClusterLabelMetrics
        Measured label width and height in placement units.
    side_padding_pt : float
        Padding applied on the left, right, top, and bottom of the content
        bbox before adding label-band room.
    label_band_pt : float
        Vertical band reserved above the content for an internal top label.
    extra_top_band_pt : float, default=0.0
        Additional top clearance reserved for external-edge handling.

    Returns
    -------
    ClusterPlacementBox
        Full cluster footprint, anchor offset, raw content bbox, and reserved
        label-band extent.

    Raises
    ------
    ValueError
        If either tensor is not shaped ``[N_inner, 2]`` or contains no members.
    """
    if inner_positions.ndim != 2 or inner_positions.shape[1] != 2:
        raise ValueError("inner_positions must have shape [N_inner, 2]")
    if inner_sizes.ndim != 2 or inner_sizes.shape[1] != 2:
        raise ValueError("inner_sizes must have shape [N_inner, 2]")
    if inner_positions.shape[0] == 0:
        raise ValueError("cluster bbox requires at least one inner member")
    if inner_positions.shape[0] != inner_sizes.shape[0]:
        raise ValueError("inner_positions and inner_sizes must have matching row counts")

    half_sizes = inner_sizes / 2.0
    lower = inner_positions - half_sizes
    upper = inner_positions + half_sizes
    x_min = float(lower[:, 0].min().item())
    y_min = float(lower[:, 1].min().item())
    x_max = float(upper[:, 0].max().item())
    y_max = float(upper[:, 1].max().item())

    inner_width = max(x_max - x_min, 0.0)
    inner_height = max(y_max - y_min, 0.0)
    label_extra_width = max(0.0, float(label_metrics.label_width_pt) - inner_width)
    width = inner_width + 2.0 * float(side_padding_pt) + label_extra_width
    height = (
        inner_height
        + 2.0 * float(side_padding_pt)
        + float(label_band_pt)
        + float(extra_top_band_pt)
    )

    full_center_x = (x_min + x_max) / 2.0
    full_y_min = y_min - float(side_padding_pt)
    full_y_max = y_max + float(side_padding_pt) + float(label_band_pt) + float(extra_top_band_pt)
    full_center_y = (full_y_min + full_y_max) / 2.0
    inner_centroid = inner_positions.mean(dim=0)
    anchor_offset = (
        full_center_x - float(inner_centroid[0].item()),
        full_center_y - float(inner_centroid[1].item()),
    )
    label_band_top = full_y_max
    label_band_bottom = full_y_max - float(label_band_pt)

    return ClusterPlacementBox(
        width=width,
        height=height,
        anchor_offset=anchor_offset,
        inner_bbox=(x_min, y_min, x_max, y_max),
        label_band_y_extent=(label_band_top, label_band_bottom),
    )


def _estimate_label_metrics(label: str) -> ClusterLabelMetrics:
    """Estimate label footprint in placement units.

    Parameters
    ----------
    label : str
        Explicit cluster label text.

    Returns
    -------
    ClusterLabelMetrics
        Deterministic approximate label dimensions used for metric geometry.
    """
    return ClusterLabelMetrics(label_width_pt=max(8.0, 5.0 * len(label)), label_height_pt=10.0)


def _bounds_from_center_size(
    center: torch.Tensor, size: torch.Tensor
) -> Tuple[float, float, float, float]:
    """Return AABB bounds for a center/size pair.

    Parameters
    ----------
    center : torch.Tensor
        Item center with shape ``[2]``.
    size : torch.Tensor
        Item size as ``(width, height)`` with shape ``[2]``.

    Returns
    -------
    tuple[float, float, float, float]
        Bounds as ``(x_min, y_min, x_max, y_max)``.
    """
    half_size = size / 2.0
    lower = center - half_size
    upper = center + half_size
    return (
        float(lower[0].item()),
        float(lower[1].item()),
        float(upper[0].item()),
        float(upper[1].item()),
    )


def _union_bounds(
    bounds: Sequence[Tuple[float, float, float, float]],
) -> Tuple[float, float, float, float]:
    """Return the union of non-empty AABB bounds.

    Parameters
    ----------
    bounds : Sequence[tuple[float, float, float, float]]
        Bounds to merge.

    Returns
    -------
    tuple[float, float, float, float]
        Merged bounds as ``(x_min, y_min, x_max, y_max)``.

    Raises
    ------
    ValueError
        If ``bounds`` is empty.
    """
    if not bounds:
        raise ValueError("cannot union an empty bounds sequence")
    return (
        min(bound[0] for bound in bounds),
        min(bound[1] for bound in bounds),
        max(bound[2] for bound in bounds),
        max(bound[3] for bound in bounds),
    )


def _center_size_from_bounds(
    bounds: Tuple[float, float, float, float],
) -> Tuple[Tuple[float, float], Tuple[float, float]]:
    """Convert AABB bounds to a center/size pseudo-item.

    Parameters
    ----------
    bounds : tuple[float, float, float, float]
        Bounds as ``(x_min, y_min, x_max, y_max)``.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        Center and size tuples compatible with ``compute_cluster_placement_bbox``.
    """
    x_min, y_min, x_max, y_max = bounds
    return (
        ((x_min + x_max) / 2.0, (y_min + y_max) / 2.0),
        (max(0.0, x_max - x_min), max(0.0, y_max - y_min)),
    )


def build_cluster_geometry_profile(
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    labels: Optional[Mapping[str, str]],
    clusters: Optional[Mapping[str, Sequence[int]]],
    cluster_parents: Optional[Mapping[str, Optional[str]]],
    *,
    side_padding_pt: float = 8.0,
    label_band_pt: float = 26.0,
) -> Optional[ClusterGeometryProfile]:
    """Build the shared read-only profile for cluster-quality metrics.

    Parameters
    ----------
    positions : torch.Tensor
        Node center positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node box sizes with shape ``[N, 2]``.
    labels : Optional[Mapping[str, str]]
        Explicit cluster-label text keyed by cluster name. Missing labels do
        not create label-band occlusion terms.
    clusters : Optional[Mapping[str, Sequence[int]]]
        Cluster membership keyed by cluster name.
    cluster_parents : Optional[Mapping[str, Optional[str]]]
        Optional nested-cluster parent mapping.
    side_padding_pt : float, optional
        Padding used when deriving cluster boxes.
    label_band_pt : float, optional
        Reserved top label-band height for explicitly labelled clusters.

    Returns
    -------
    Optional[ClusterGeometryProfile]
        Derived immutable profile, or ``None`` when no valid cluster metadata
        is available.
    """
    if not clusters:
        return None
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must have shape [N, 2]")
    if node_sizes.ndim != 2 or node_sizes.shape[1] != 2:
        raise ValueError("node_sizes must have shape [N, 2]")
    if positions.shape[0] != node_sizes.shape[0]:
        raise ValueError("positions and node_sizes must have matching row counts")

    num_nodes = int(positions.shape[0])
    valid_clusters: dict[str, Tuple[int, ...]] = {}
    has_valid_member = False
    for name, members in clusters.items():
        valid_members = tuple(
            sorted({int(index) for index in members if 0 <= int(index) < num_nodes})
        )
        if valid_members:
            has_valid_member = True
        valid_clusters[str(name)] = valid_members
    if not has_valid_member:
        return None

    parent_lookup = cluster_parents or {}
    tree = ClusterTree.from_flat_membership(valid_clusters, parent_lookup)
    positions_cpu = positions.detach().cpu().to(dtype=torch.float64)
    sizes_cpu = node_sizes.detach().cpu().to(dtype=torch.float64)
    half_sizes = sizes_cpu / 2.0
    node_lower = positions_cpu - half_sizes
    node_upper = positions_cpu + half_sizes
    node_bounds = tuple(
        (
            float(node_lower[index, 0].item()),
            float(node_lower[index, 1].item()),
            float(node_upper[index, 0].item()),
            float(node_upper[index, 1].item()),
        )
        for index in range(num_nodes)
    )

    boxes: dict[str, ClusterProfileBox] = {}
    label_lookup = labels or {}
    for name in tree.bottom_up_order():
        descendants = frozenset(
            index for index in tree.descendants_per_cluster[name] if 0 <= index < num_nodes
        )
        if not descendants:
            continue
        direct_leaf_indices = sorted(
            index for index in tree.leaves_per_cluster[name] if 0 <= index < num_nodes
        )
        content_centers: list[tuple[float, float]] = []
        content_sizes: list[tuple[float, float]] = []
        for index in direct_leaf_indices:
            content_centers.append(
                (
                    float(positions_cpu[index, 0].item()),
                    float(positions_cpu[index, 1].item()),
                )
            )
            content_sizes.append(
                (
                    float(sizes_cpu[index, 0].item()),
                    float(sizes_cpu[index, 1].item()),
                )
            )
        for child_name in tree.children_per_cluster[name]:
            if child_name not in boxes:
                continue
            child_center, child_size = _center_size_from_bounds(boxes[child_name].bounds)
            content_centers.append(child_center)
            content_sizes.append(child_size)
        if not content_centers:
            continue
        label = str(label_lookup.get(name, ""))
        label_metrics = _estimate_label_metrics(label) if label else ClusterLabelMetrics(0.0, 0.0)
        effective_label_band = float(label_band_pt) if label else 0.0
        inner_positions = torch.tensor(content_centers, dtype=torch.float64)
        inner_sizes = torch.tensor(content_sizes, dtype=torch.float64)
        placement = compute_cluster_placement_bbox(
            inner_positions=inner_positions,
            inner_sizes=inner_sizes,
            label_metrics=label_metrics,
            side_padding_pt=float(side_padding_pt),
            label_band_pt=effective_label_band,
        )
        inner_x_min, inner_y_min, inner_x_max, inner_y_max = placement.inner_bbox
        bounds = (
            (inner_x_min + inner_x_max - placement.width) / 2.0,
            inner_y_min - float(side_padding_pt),
            (inner_x_min + inner_x_max + placement.width) / 2.0,
            inner_y_max + float(side_padding_pt) + effective_label_band,
        )
        label_bounds = None
        if label:
            label_top, label_bottom = placement.label_band_y_extent
            label_bounds = (bounds[0], label_bottom, bounds[2], label_top)
        raw_leaf_bounds = _union_bounds(
            [
                _bounds_from_center_size(positions_cpu[index], sizes_cpu[index])
                for index in sorted(descendants)
            ]
        )
        boxes[name] = ClusterProfileBox(
            bounds=bounds,
            inner_bounds=placement.inner_bbox,
            label_bounds=label_bounds,
            descendants=descendants,
            raw_leaf_bounds=raw_leaf_bounds,
        )

    sibling_pairs: list[tuple[str, str]] = []
    parent_groups: dict[Optional[str], list[str]] = {}
    for name in tree.top_down_order():
        if name in boxes:
            parent_groups.setdefault(tree.parents[name], []).append(name)
    for names in parent_groups.values():
        names.sort()
        for left_index, left_name in enumerate(names):
            for right_name in names[left_index + 1 :]:
                sibling_pairs.append((left_name, right_name))

    def sibling_pair_key(
        pair: tuple[str, str],
    ) -> tuple[
        tuple[tuple[int, ...], tuple[int, ...]],
        tuple[Tuple[float, float, float, float], Tuple[float, float, float, float]],
    ]:
        """Return a rename-invariant key for capped sibling-pair sampling.

        Parameters
        ----------
        pair : tuple[str, str]
            Sibling cluster names for one pair.

        Returns
        -------
        tuple
            Pair key based on member node indices and geometry, never cluster
            name strings.
        """
        left_name, right_name = pair
        left_members = tuple(sorted(boxes[left_name].descendants))
        right_members = tuple(sorted(boxes[right_name].descendants))
        member_key = (
            (left_members, right_members)
            if left_members <= right_members
            else (right_members, left_members)
        )
        left_bounds = boxes[left_name].bounds
        right_bounds = boxes[right_name].bounds
        bounds_key = (
            (left_bounds, right_bounds)
            if left_bounds <= right_bounds
            else (right_bounds, left_bounds)
        )
        return member_key, bounds_key

    sibling_pairs.sort(key=sibling_pair_key)

    return ClusterGeometryProfile(
        tree=tree,
        cluster_names=tuple(name for name in tree.top_down_order() if name in boxes),
        boxes=boxes,
        node_bounds=node_bounds,
        sibling_pairs=tuple(sibling_pairs),
    )


def cluster_descendants(tree: ClusterTree, name: str) -> frozenset[int]:
    """Return all leaf node descendants for a cluster.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy lookup table.
    name : str
        Cluster name.

    Returns
    -------
    frozenset[int]
        All leaf node indices reachable through ``name``.
    """
    return tree.descendants_per_cluster[name]


def cluster_leaves_only_at_level(tree: ClusterTree, name: str) -> frozenset[int]:
    """Return leaves owned directly by a cluster level.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy lookup table.
    name : str
        Cluster name.

    Returns
    -------
    frozenset[int]
        Leaf node indices in ``name`` after removing descendants of immediate
        child clusters.
    """
    return tree.leaves_per_cluster[name]


def cluster_subtree(tree: ClusterTree, name: str) -> Tuple[str, ...]:
    """All cluster names in the subtree rooted at ``name``.

    Parameters
    ----------
    tree : ClusterTree
        Cluster hierarchy lookup table.
    name : str
        Root cluster name for the requested subtree.

    Returns
    -------
    tuple[str, ...]
        Cluster names in preorder, including ``name``.
    """
    ordered: list[str] = []
    stack = [name]
    while stack:
        current = stack.pop()
        ordered.append(current)
        for child_name in reversed(tree.children_per_cluster[current]):
            stack.append(child_name)
    return tuple(ordered)
