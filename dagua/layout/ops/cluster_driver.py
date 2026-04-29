"""Recursive cluster-aware layout driver."""

from __future__ import annotations

import copy
from dataclasses import dataclass
from typing import Dict, Mapping, Optional, Sequence

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.cluster_geometry import (
    ClusterLabelMetrics,
    ClusterPlacementBox,
    ClusterTree,
    compute_cluster_placement_bbox,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

_OVERLAP_EPSILON = 1.0e-6
_OVERLAP_MAX_PASSES = 80


@dataclass
class ClusterPlacement:
    """Placed local geometry for one cluster.

    Parameters
    ----------
    positions : dict[int, torch.Tensor]
        Descendant leaf positions in this cluster's local coordinate system.
    anchor : torch.Tensor
        Center of the cluster placement rectangle with shape ``[2]``.
    bbox : ClusterPlacementBox
        Placement-time bbox metadata for this cluster.
    """

    positions: Dict[int, torch.Tensor]
    anchor: torch.Tensor
    bbox: ClusterPlacementBox


@dataclass(frozen=True)
class _PlacementItem:
    """A leaf node or child-cluster placeholder in one placement level.

    Parameters
    ----------
    key : int | str
        Original node index for leaf items, or cluster name for placeholders.
    kind : str
        Item kind, either ``"leaf"`` or ``"cluster"``.
    size : torch.Tensor
        Placement rectangle size with shape ``[2]``.
    """

    key: int | str
    kind: str
    size: torch.Tensor


@dataclass(frozen=True)
class ClusterAwareDriver(Op):
    """Wrap a leaf-placement pipeline into recursive cluster-aware placement.

    For each cluster, the driver first places child clusters, then runs the
    inner pipeline on the current level's direct leaves plus child cluster
    placeholders. Cluster placeholders are rigid rectangles whose sizes come
    from :func:`compute_cluster_placement_bbox`.

    Parameters
    ----------
    inner_pipeline : Sequence[Op]
        Ops to apply at each cluster level. The same ops are called
        recursively on each induced placement problem.
    side_padding_pt : float, default=8.0
        Padding inside cluster bbox on the left, right, and bottom.
    label_band_pt : float, default=26.0
        Reserved band at the top of the cluster bbox for the label.
    external_clearance_pt : float, default=10.0
        Extra padding on all sides so external nodes repel from the cluster
        perimeter with clearance.
    cluster_compactness_weight : float, default=1.0
        Reserved for a later compactness-loss refinement inside recursive
        subproblems.
    """

    inner_pipeline: Sequence[Op]
    side_padding_pt: float = 8.0
    label_band_pt: float = 26.0
    external_clearance_pt: float = 10.0
    cluster_compactness_weight: float = 1.0

    name = "cluster_aware_driver"
    category = OpCategory.CONTROL
    reads = ("pos",)
    writes = ("pos", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run recursive cluster-aware placement.

        Parameters
        ----------
        problem : LayoutProblem
            Full layout problem with original graph topology and cluster tree.
        state : SolveState
            Mutable state receiving the final original-node positions.
        ctx : RuntimeContext
            Runtime context forwarded to each inner pipeline invocation.

        Returns
        -------
        SolveState
            State with ``pos`` set to the final original-node coordinates.

        Raises
        ------
        ValueError
            If cluster metadata or node sizes are unavailable.
        RuntimeError
            If an inner pipeline fails to produce positions.
        """
        tree = problem.get_cluster_tree()
        if tree is None:
            return Pipeline(self.inner_pipeline, name="cluster_driver_inner").apply(
                problem,
                state,
                ctx,
            )
        if problem.node_sizes is None:
            raise ValueError("ClusterAwareDriver requires problem.node_sizes.")

        placements: dict[str, ClusterPlacement] = {}
        final_positions = torch.zeros(
            (problem.num_nodes, 2),
            dtype=torch.float32,
            device=problem.node_sizes.device,
        )
        root_placement = self._place_level(
            problem=problem,
            ctx=ctx,
            tree=tree,
            cluster_name=None,
            placements=placements,
        )
        for node_index, position in root_placement.positions.items():
            final_positions[int(node_index)] = position.to(
                device=final_positions.device,
                dtype=final_positions.dtype,
            )

        state.pos = final_positions
        state.extras["cluster_inner_layout"] = placements
        return state

    def _place_level(
        self,
        problem: LayoutProblem,
        ctx: RuntimeContext,
        tree: ClusterTree,
        cluster_name: Optional[str],
        placements: dict[str, ClusterPlacement],
    ) -> ClusterPlacement:
        """Place one cluster level and return its local leaf geometry.

        Parameters
        ----------
        problem : LayoutProblem
            Full original layout problem.
        ctx : RuntimeContext
            Runtime context for inner pipeline calls.
        tree : ClusterTree
            Cluster hierarchy.
        cluster_name : str or None
            Cluster to place, or ``None`` for the root placement level.
        placements : dict[str, ClusterPlacement]
            Accumulated child cluster placements.

        Returns
        -------
        ClusterPlacement
            Local geometry for this level. For the root level, ``bbox`` is a
            synthetic box around all placed items.
        """
        child_clusters = (
            tree.roots if cluster_name is None else tree.children_per_cluster[cluster_name]
        )
        for child_name in child_clusters:
            if child_name not in placements:
                placements[child_name] = self._place_level(
                    problem=problem,
                    ctx=ctx,
                    tree=tree,
                    cluster_name=child_name,
                    placements=placements,
                )

        items = self._build_items(problem, tree, cluster_name, child_clusters, placements)
        if not items:
            empty_box = ClusterPlacementBox(
                width=0.0,
                height=0.0,
                anchor_offset=(0.0, 0.0),
                inner_bbox=(0.0, 0.0, 0.0, 0.0),
                label_band_y_extent=(0.0, 0.0),
            )
            return ClusterPlacement(positions={}, anchor=torch.zeros(2), bbox=empty_box)

        sub_problem = self._build_subproblem(problem, tree, cluster_name, items)
        sub_state = self._run_inner(sub_problem, ctx)
        assert sub_state.pos is not None
        item_positions = self._resolve_rect_overlaps(
            positions=sub_state.pos.to(dtype=torch.float32),
            sizes=sub_problem.node_sizes.to(dtype=torch.float32),
        )
        item_positions = self._enforce_external_cluster_clearance(
            problem=problem,
            tree=tree,
            cluster_name=cluster_name,
            items=items,
            item_positions=item_positions,
            item_sizes=sub_problem.node_sizes.to(dtype=torch.float32),
        )
        item_positions = self._resolve_rect_overlaps(
            positions=item_positions,
            sizes=sub_problem.node_sizes.to(dtype=torch.float32),
        )
        leaf_positions = self._expand_item_positions(items, item_positions, placements)
        bbox = compute_cluster_placement_bbox(
            inner_positions=item_positions,
            inner_sizes=sub_problem.node_sizes.to(dtype=torch.float32),
            label_metrics=ClusterLabelMetrics(label_width_pt=0.0, label_height_pt=0.0),
            side_padding_pt=self.side_padding_pt + self.external_clearance_pt,
            label_band_pt=self.label_band_pt,
        )
        anchor = torch.tensor(
            [
                float(item_positions[:, 0].mean().item()) + bbox.anchor_offset[0],
                float(item_positions[:, 1].mean().item()) + bbox.anchor_offset[1],
            ],
            dtype=torch.float32,
        )
        return ClusterPlacement(positions=leaf_positions, anchor=anchor, bbox=bbox)

    def _build_items(
        self,
        problem: LayoutProblem,
        tree: ClusterTree,
        cluster_name: Optional[str],
        child_clusters: Sequence[str],
        placements: Mapping[str, ClusterPlacement],
    ) -> list[_PlacementItem]:
        """Build placement items for one hierarchy level.

        Parameters
        ----------
        problem : LayoutProblem
            Full original layout problem.
        tree : ClusterTree
            Cluster hierarchy.
        cluster_name : str or None
            Current cluster name, or ``None`` for root.
        child_clusters : Sequence[str]
            Immediate child clusters for this level.
        placements : Mapping[str, ClusterPlacement]
            Already-computed child placements.

        Returns
        -------
        list[_PlacementItem]
            Leaf and placeholder items in deterministic order.
        """
        assert problem.node_sizes is not None
        if cluster_name is None:
            clustered_nodes: set[int] = set()
            for root_name in tree.roots:
                clustered_nodes.update(tree.descendants_per_cluster[root_name])
            direct_leaves = [
                index for index in range(problem.num_nodes) if index not in clustered_nodes
            ]
        else:
            direct_leaves = sorted(tree.leaves_per_cluster[cluster_name])

        items = [
            _PlacementItem(
                key=int(node_index),
                kind="leaf",
                size=problem.node_sizes[int(node_index)].detach().to(dtype=torch.float32),
            )
            for node_index in direct_leaves
        ]
        for child_name in child_clusters:
            child_box = placements[child_name].bbox
            items.append(
                _PlacementItem(
                    key=child_name,
                    kind="cluster",
                    size=torch.tensor([child_box.width, child_box.height], dtype=torch.float32),
                )
            )
        return items

    def _build_subproblem(
        self,
        problem: LayoutProblem,
        tree: ClusterTree,
        cluster_name: Optional[str],
        items: Sequence[_PlacementItem],
    ) -> LayoutProblem:
        """Create the induced placement problem for one level.

        Parameters
        ----------
        problem : LayoutProblem
            Full original layout problem.
        tree : ClusterTree
            Cluster hierarchy.
        cluster_name : str or None
            Current cluster name, or ``None`` for root.
        items : Sequence[_PlacementItem]
            Placement items for the level.

        Returns
        -------
        LayoutProblem
            Subproblem with local node IDs and same edge semantics as the
            selected hierarchy level.
        """
        item_index = {item.key: local_index for local_index, item in enumerate(items)}
        owner_by_node = self._node_owner_at_level(tree, cluster_name, items)
        edges: list[tuple[int, int]] = []
        for source, target in problem.edge_index.t().tolist():
            source_owner = owner_by_node.get(int(source))
            target_owner = owner_by_node.get(int(target))
            if source_owner is None or target_owner is None or source_owner == target_owner:
                continue
            edges.append((item_index[source_owner], item_index[target_owner]))

        if edges:
            edge_index = torch.tensor(edges, dtype=torch.long).t().contiguous()
        else:
            edge_index = torch.empty((2, 0), dtype=torch.long)
        node_sizes = torch.stack([item.size for item in items]).to(dtype=torch.float32)
        return LayoutProblem(
            edge_index=edge_index,
            num_nodes=len(items),
            node_sizes=node_sizes,
            direction=problem.direction,
            seed=problem.seed,
        )

    def _node_owner_at_level(
        self,
        tree: ClusterTree,
        cluster_name: Optional[str],
        items: Sequence[_PlacementItem],
    ) -> dict[int, int | str]:
        """Map original nodes to their placement item at one level.

        Parameters
        ----------
        tree : ClusterTree
            Cluster hierarchy.
        cluster_name : str or None
            Current cluster name, or ``None`` for root.
        items : Sequence[_PlacementItem]
            Placement items at this level.

        Returns
        -------
        dict[int, int | str]
            Original node index to item key.
        """
        del cluster_name
        owners: dict[int, int | str] = {}
        for item in items:
            if item.kind == "leaf":
                owners[int(item.key)] = int(item.key)
            else:
                for node_index in tree.descendants_per_cluster[str(item.key)]:
                    owners[int(node_index)] = str(item.key)
        return owners

    def _run_inner(self, problem: LayoutProblem, ctx: RuntimeContext) -> SolveState:
        """Run the wrapped inner pipeline on a subproblem.

        Parameters
        ----------
        problem : LayoutProblem
            Local hierarchy-level layout problem.
        ctx : RuntimeContext
            Runtime context to reuse.

        Returns
        -------
        SolveState
            Final inner solve state.

        Raises
        ------
        RuntimeError
            If the wrapped pipeline does not produce positions.
        """
        local_state = SolveState()
        local_ctx = copy.copy(ctx)
        local_ctx.plan = copy.copy(ctx.plan)
        final_state = Pipeline(self.inner_pipeline, name="cluster_driver_inner").apply(
            problem,
            local_state,
            local_ctx,
        )
        if final_state.pos is None:
            raise RuntimeError("ClusterAwareDriver inner pipeline did not produce positions.")
        return final_state

    def _expand_item_positions(
        self,
        items: Sequence[_PlacementItem],
        item_positions: torch.Tensor,
        placements: Mapping[str, ClusterPlacement],
    ) -> dict[int, torch.Tensor]:
        """Translate child-cluster descendants into the current level.

        Parameters
        ----------
        items : Sequence[_PlacementItem]
            Placement items at this level.
        item_positions : torch.Tensor
            Item center positions with shape ``[N_items, 2]``.
        placements : Mapping[str, ClusterPlacement]
            Child placement geometry.

        Returns
        -------
        dict[int, torch.Tensor]
            Original node positions in the current level's local coordinates.
        """
        expanded: dict[int, torch.Tensor] = {}
        for item_index, item in enumerate(items):
            target = item_positions[item_index].detach().to(dtype=torch.float32)
            if item.kind == "leaf":
                expanded[int(item.key)] = target
                continue
            child = placements[str(item.key)]
            translation = target - child.anchor.to(dtype=torch.float32)
            for node_index, position in child.positions.items():
                expanded[int(node_index)] = position.to(dtype=torch.float32) + translation
        return expanded

    def _enforce_external_cluster_clearance(
        self,
        problem: LayoutProblem,
        tree: ClusterTree,
        cluster_name: Optional[str],
        items: Sequence[_PlacementItem],
        item_positions: torch.Tensor,
        item_sizes: torch.Tensor,
    ) -> torch.Tensor:
        """Move direct external leaves clear of child-cluster placeholders.

        Parameters
        ----------
        problem : LayoutProblem
            Full original layout problem.
        tree : ClusterTree
            Cluster hierarchy.
        cluster_name : str or None
            Current hierarchy level.
        items : Sequence[_PlacementItem]
            Placement items at this level.
        item_positions : torch.Tensor
            Current item centers with shape ``[N_items, 2]``.
        item_sizes : torch.Tensor
            Current item sizes with shape ``[N_items, 2]``.

        Returns
        -------
        torch.Tensor
            Positions with direct external leaves shifted outside child
            cluster rectangles for edges entering or leaving those clusters.
        """
        if not items or problem.edge_index.numel() == 0:
            return item_positions
        adjusted = item_positions.detach().clone()
        item_index = {item.key: local_index for local_index, item in enumerate(items)}
        owner_by_node = self._node_owner_at_level(tree, cluster_name, items)
        for source, target in problem.edge_index.t().tolist():
            source_owner = owner_by_node.get(int(source))
            target_owner = owner_by_node.get(int(target))
            if source_owner is None or target_owner is None or source_owner == target_owner:
                continue
            source_index = item_index[source_owner]
            target_index = item_index[target_owner]
            source_item = items[source_index]
            target_item = items[target_index]
            if source_item.kind == "leaf" and target_item.kind == "cluster":
                cluster_top = adjusted[target_index, 1] + item_sizes[target_index, 1] / 2.0
                min_y = cluster_top + item_sizes[source_index, 1] / 2.0 + self.external_clearance_pt
                adjusted[source_index, 1] = torch.maximum(adjusted[source_index, 1], min_y)
            elif source_item.kind == "cluster" and target_item.kind == "leaf":
                cluster_bottom = adjusted[source_index, 1] - item_sizes[source_index, 1] / 2.0
                max_y = (
                    cluster_bottom - item_sizes[target_index, 1] / 2.0 - self.external_clearance_pt
                )
                adjusted[target_index, 1] = torch.minimum(adjusted[target_index, 1], max_y)
        return adjusted

    def _resolve_rect_overlaps(self, positions: torch.Tensor, sizes: torch.Tensor) -> torch.Tensor:
        """Project overlapping placement rectangles apart.

        Parameters
        ----------
        positions : torch.Tensor
            Rectangle centers with shape ``[N, 2]``.
        sizes : torch.Tensor
            Rectangle sizes with shape ``[N, 2]``.

        Returns
        -------
        torch.Tensor
            Adjusted rectangle centers with shape ``[N, 2]``.
        """
        resolved = positions.detach().clone().to(dtype=torch.float32)
        if resolved.shape[0] < 2:
            return resolved
        half_sizes = sizes.to(dtype=torch.float32) / 2.0
        for _ in range(_OVERLAP_MAX_PASSES):
            moved = False
            for left in range(resolved.shape[0] - 1):
                for right in range(left + 1, resolved.shape[0]):
                    delta = resolved[right] - resolved[left]
                    min_delta = half_sizes[left] + half_sizes[right]
                    overlap = min_delta - delta.abs()
                    if float(overlap[0].item()) <= 0.0 or float(overlap[1].item()) <= 0.0:
                        continue
                    axis = 0 if float(overlap[0].item()) <= float(overlap[1].item()) else 1
                    sign = 1.0 if float(delta[axis].item()) >= 0.0 else -1.0
                    if abs(float(delta[axis].item())) < _OVERLAP_EPSILON:
                        sign = 1.0 if (right - left) % 2 else -1.0
                    shift = (float(overlap[axis].item()) + _OVERLAP_EPSILON) / 2.0
                    resolved[left, axis] -= sign * shift
                    resolved[right, axis] += sign * shift
                    moved = True
            if not moved:
                break
        return resolved
