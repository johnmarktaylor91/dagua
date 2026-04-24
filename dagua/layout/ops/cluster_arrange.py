"""Sprint 18: cluster centroid grid arrangement.

When a graph has clusters AND the gradient pipeline produced a
catastrophically narrow layout (clusters stacked vertically because
the gradient solver couldn't find horizontal pressure to spread
them), this op:

1. Computes each cluster's bbox + centroid.
2. Lays out cluster centroids on a roughly-square grid sized to fit
   the total intra-cluster footprint plus padding.
3. Translates each cluster's member positions to its new centroid.
   Intra-cluster geometry is preserved exactly.

Symptom this addresses: ``clustered_deep`` holdout family scored
77.50 with bbox 85 x 4783 (aspect 0.018) on n=96 because all 6
clusters of 16 nodes ended up vertically stacked at x=0. The
composite penalises aspect_ratio_deviation but the gradient solver
had no way to spread clusters horizontally — repulsion between
clusters is dominated by intra-cluster cohesion.

The ``BarycenterReorder`` polish pass doesn't help because there's
nothing to reorder within a column of single-node layers.
``AspectRatioFit`` doesn't help because rescaling x by N when
extent_x = 0 produces 0 (anything * 0 = 0). The geometric collapse
must be fixed by re-positioning cluster centroids, not by rescaling.

Sprint 18 fires only when ``problem.clusters`` is populated AND the
current aspect ratio is degenerate (< 0.15 or > 6.67). On
well-shaped cluster layouts (clustered_shallow, nested_3lvl) this
is a no-op. Tested in isolation on the 9 cluster-family holdout
graphs.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import ClassVar, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


@dataclass(frozen=True)
class ClusterGridArrangeConfig:
    """Configuration for :class:`ClusterGridArrange`.

    Parameters
    ----------
    centroid_collapse_threshold : float, default=0.1
        Fire when the fraction of layout extent spanned by cluster
        centroids (along the narrower axis) is below this. Catches
        the literal "cluster columns stacked vertically" symptom
        regardless of node-size-driven bbox aspect. 0.1 = clusters
        share less than 10% of the layout's narrower axis -> they
        are stacked.
    grid_padding_factor : float, default=0.3
        Extra spacing between cluster centroids beyond the largest
        cluster's bbox extent. 0.3 = 30% padding.
    enabled : bool, default=True
        Master opt-out.
    """

    centroid_collapse_threshold: float = 0.1
    grid_padding_factor: float = 0.3
    # Default OFF: investigation in Sprint 18 showed that re-arranging
    # clusters into a 2D grid breaks depth_spearman_rho (composite
    # weight 15), and the edge_length_cv improvement (composite weight
    # 20) is not enough to compensate. clustered_deep stuck at 77.50
    # appears to be a genuine local max for the current composite
    # metric. Op shipped for users with custom metrics that weight
    # aspect ratio explicitly; opt in via
    # ``ClusterGridArrangeConfig(enabled=True)``.
    enabled: bool = False


def _node_indices_for_cluster(cluster_members) -> list[int]:
    """Best-effort extraction of integer node indices from a cluster spec.

    Clusters in DaguaGraph can be stored as lists of node-ids (str) or
    integer indices. The pipeline already resolved them via
    ``_resolve_graph_flex`` for flex constraints, but cluster
    definitions on ``problem.clusters`` retain whatever form the user
    supplied. We accept both forms here defensively.
    """
    out: list[int] = []
    for m in cluster_members:
        if isinstance(m, int):
            out.append(m)
        elif hasattr(m, "__int__"):
            try:
                out.append(int(m))
            except (TypeError, ValueError):
                continue
    return out


@register_op
class ClusterGridArrange(Op):
    """Re-position cluster centroids on a grid when the layout is degenerate.

    Reads ``state.pos`` + ``problem.clusters``. Computes each cluster's
    current centroid, assigns each cluster a target position on a
    roughly-square grid, and translates members. Intra-cluster geometry
    (relative positions of members) is preserved.
    """

    name: ClassVar[str] = "cluster_grid_arrange"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: ClusterGridArrangeConfig | None = None) -> None:
        self.config = config or ClusterGridArrangeConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        del ctx
        if not self.config.enabled or state.pos is None:
            return state
        pos = state.pos
        if pos.shape[0] < 2:
            return state
        if not problem.clusters:
            return state

        # Build per-cluster member-index lists. Skip clusters whose
        # members aren't in [0, num_nodes).
        n = pos.shape[0]
        cluster_specs: list[tuple[str, torch.Tensor]] = []
        seen_members: set[int] = set()
        for cname, members in problem.clusters.items():
            indices = _node_indices_for_cluster(members)
            valid = [i for i in indices if 0 <= i < n and i not in seen_members]
            if not valid:
                continue
            seen_members.update(valid)
            cluster_specs.append(
                (
                    cname,
                    torch.tensor(valid, dtype=torch.long, device=pos.device),
                )
            )
        if len(cluster_specs) < 2:
            return state

        # Compute current centroid + bbox extent per cluster.
        centroids = []
        max_bbox_extent = 0.0
        for _, idx_tensor in cluster_specs:
            cluster_pos = pos[idx_tensor].detach()
            cx = float(cluster_pos[:, 0].mean().item())
            cy = float(cluster_pos[:, 1].mean().item())
            centroids.append((cx, cy))
            cw = float((cluster_pos[:, 0].max() - cluster_pos[:, 0].min()).item())
            ch = float((cluster_pos[:, 1].max() - cluster_pos[:, 1].min()).item())
            max_bbox_extent = max(max_bbox_extent, cw, ch)

        # Trigger: centroids span a tiny fraction of the layout. This
        # is the "stacked clusters" symptom -- nodes spread along one
        # axis (because intra-cluster forces did their job) but cluster
        # centroids share the other axis. Compare centroid spread to
        # the layout's narrower axis.
        x_min = float(pos[:, 0].detach().min().item())
        x_max = float(pos[:, 0].detach().max().item())
        y_min = float(pos[:, 1].detach().min().item())
        y_max = float(pos[:, 1].detach().max().item())
        layout_w = max(x_max - x_min, 1e-6)
        layout_h = max(y_max - y_min, 1e-6)
        narrower_axis = min(layout_w, layout_h)

        cx_vals = [c[0] for c in centroids]
        cy_vals = [c[1] for c in centroids]
        centroid_x_spread = max(cx_vals) - min(cx_vals)
        centroid_y_spread = max(cy_vals) - min(cy_vals)
        # If centroids span less than threshold * narrower_axis along
        # ANY axis, they are stacked along the orthogonal axis.
        x_stacked = centroid_x_spread < self.config.centroid_collapse_threshold * narrower_axis
        y_stacked = centroid_y_spread < self.config.centroid_collapse_threshold * narrower_axis
        if not (x_stacked or y_stacked):
            return state

        n_clusters = len(cluster_specs)
        cols = max(1, int(math.ceil(math.sqrt(n_clusters))))
        rows = int(math.ceil(n_clusters / cols))

        # Spacing = max cluster bbox + padding. Use a sensible floor when
        # max_bbox_extent is tiny (degenerate clusters).
        spacing = max(max_bbox_extent * (1.0 + self.config.grid_padding_factor), 50.0)

        # Sort clusters by current centroid y (preserves depth/sequence
        # ordering when present, e.g. compound_stage_0..N) and assign
        # row-major to grid cells.
        order = sorted(range(n_clusters), key=lambda i: centroids[i][1])

        pos_new = pos.clone()
        for grid_idx, src_idx in enumerate(order):
            row = grid_idx // cols
            col = grid_idx % cols
            target_x = (col - (cols - 1) / 2.0) * spacing
            target_y = (row - (rows - 1) / 2.0) * spacing
            cx, cy = centroids[src_idx]
            dx = target_x - cx
            dy = target_y - cy
            members = cluster_specs[src_idx][1]
            pos_new[members, 0] = pos[members, 0] + dx
            pos_new[members, 1] = pos[members, 1] + dy

        state.pos = pos_new.detach().requires_grad_(pos.requires_grad)
        return state


__all__ = ["ClusterGridArrange", "ClusterGridArrangeConfig"]
