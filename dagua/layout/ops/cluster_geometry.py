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
            Mapping from cluster name to all descendant leaf node indices.
        cluster_parents : Mapping[str, Optional[str]]
            Mapping from cluster name to parent cluster name. Missing entries
            and parents outside ``clusters`` are treated as root membership.

        Returns
        -------
        ClusterTree
            Immutable hierarchy with child, root, leaf-only, and descendant
            membership lookup tables.
        """
        cluster_names = tuple(clusters.keys())
        descendants = {
            name: frozenset(int(index) for index in members) for name, members in clusters.items()
        }
        parents = {
            name: cluster_parents.get(name) if cluster_parents.get(name) in clusters else None
            for name in cluster_names
        }
        children_lists: dict[str, list[str]] = {name: [] for name in cluster_names}
        for name in cluster_names:
            parent = parents[name]
            if parent is not None:
                children_lists[parent].append(name)

        children = {
            name: tuple(child for child in cluster_names if child in children_lists[name])
            for name in cluster_names
        }
        leaves: dict[str, frozenset[int]] = {}
        for name in cluster_names:
            child_descendants: set[int] = set()
            for child_name in children[name]:
                child_descendants.update(descendants[child_name])
            leaves[name] = frozenset(descendants[name].difference(child_descendants))

        roots = tuple(name for name in cluster_names if parents[name] is None)
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

        def visit(name: str) -> None:
            """Append ``name`` after all children have been visited.

            Parameters
            ----------
            name : str
                Cluster name to traverse.

            Returns
            -------
            None
                The traversal appends into ``ordered``.
            """
            for child_name in self.children_per_cluster[name]:
                visit(child_name)
            ordered.append(name)

        for root_name in self.roots:
            visit(root_name)
        return tuple(ordered)

    def top_down_order(self) -> Tuple[str, ...]:
        """Return cluster names in parent-before-child order.

        Returns
        -------
        tuple[str, ...]
            Cluster names ordered so every parent appears before descendants.
        """
        ordered: list[str] = []

        def visit(name: str) -> None:
            """Append ``name`` before all children are visited.

            Parameters
            ----------
            name : str
                Cluster name to traverse.

            Returns
            -------
            None
                The traversal appends into ``ordered``.
            """
            ordered.append(name)
            for child_name in self.children_per_cluster[name]:
                visit(child_name)

        for root_name in self.roots:
            visit(root_name)
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
    ordered = [name]
    for child_name in tree.children_per_cluster[name]:
        ordered.extend(cluster_subtree(tree, child_name))
    return tuple(ordered)
