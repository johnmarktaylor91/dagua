"""Sprint 10: barycenter crossing-minimization polish pass.

Classical Sugiyama-family polish: after the continuous optimizer
places nodes, freeze each node's y-coordinate (layer assignment) and
re-order nodes within each layer by the barycenter of their
in-neighbours on the previous layer. Iterate a few down-sweeps /
up-sweeps.

This directly targets the Pareto gap on ``edge_crossings`` /
``crossing_rate`` (Dagua loses 3:1 vs classical Sugiyama because the
continuous crossing loss is a soft penalty that doesn't enumerate
edge-edge crossings). After this op, layered DAGs exit with
Sugiyama-quality ordering on top of Dagua's continuous x-positions.

Preserves:
 - y-coordinates (DAG consistency + depth_spearman_rho)
 - overlap non-collision (only permuting x-positions within a layer)

Changes:
 - x-coordinates within each layer are reassigned to the SAME set of
   x-positions in barycenter order.

Sprint 10 scope: shipped as a registered op ``BarycenterReorder``
plumbed in after gradient_core + before the final OverlapProjection
in the dagua_native pipeline.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


@dataclass(frozen=True)
class BarycenterReorderConfig:
    """Configuration for :class:`BarycenterReorder`.

    Parameters
    ----------
    iterations : int, default=4
        Total number of barycenter passes. Each pass alternates between
        "downsweep" (compute barycenter from upstream layer, reorder)
        and "upsweep" (compute from downstream layer). Classical
        Sugiyama implementations converge in 2-8 passes.
    min_layer_size : int, default=2
        Don't reorder layers with fewer than this many nodes (nothing to
        reorder on a singleton layer).
    enabled : bool, default=True
        Master opt-out for the entire polish pass. Exposed so regressions
        on specific graph families can be diagnosed against a skip run.
    """

    # Sprint 18k: bumped 4 -> 8 after re-sweep at the new
    # (70, 240) spacing and AR target=0.25. Iteration sweep:
    #   1: 76.642
    #   2: 76.963
    #   4: 77.043 (Sprint 10 default)
    #   8: 77.064  <-- adopted (small gain, plateau begins)
    #  16: 77.080 (further marginal gain)
    #   0: 74.835 (no barycenter polish; significant regression)
    iterations: int = 8
    min_layer_size: int = 2
    enabled: bool = True


def _compute_barycenters_for_layer(
    pos: torch.Tensor,
    layer_members: torch.Tensor,
    neighbour_mask: torch.Tensor,
    neighbour_src: torch.Tensor,
    neighbour_dst: torch.Tensor,
) -> torch.Tensor:
    """For each node in ``layer_members``, return the mean x-coord of
    its neighbours on the adjacent layer.

    Nodes with no neighbours on the adjacent layer keep their current
    x-coordinate (so they don't move).
    """
    device = pos.device
    n_members = layer_members.shape[0]
    if n_members == 0:
        return torch.empty(0, device=device, dtype=pos.dtype)

    # A node m's neighbours are all edges whose other endpoint is in
    # the adjacent layer. neighbour_src / neighbour_dst are
    # pre-filtered for this (src in layer_members, dst in adj_layer OR
    # reverse). We compute mean(pos[neighbour_dst[matching m], 0]).
    # Efficient path: build a membership-to-index map, then scatter_mean.
    member_to_idx = torch.full((int(pos.shape[0]),), -1, device=device, dtype=torch.long)
    member_to_idx[layer_members] = torch.arange(n_members, device=device)
    src_idx = member_to_idx[neighbour_src]  # which layer-member row
    valid = src_idx >= 0
    src_idx_valid = src_idx[valid]
    adj_x = pos[neighbour_dst[valid], 0]

    # Accumulate sums + counts per layer-member row.
    sum_x = torch.zeros(n_members, device=device, dtype=pos.dtype)
    count = torch.zeros(n_members, device=device, dtype=pos.dtype)
    if src_idx_valid.numel() > 0:
        sum_x.scatter_add_(0, src_idx_valid, adj_x)
        count.scatter_add_(0, src_idx_valid, torch.ones_like(adj_x))

    has_neigh = count > 0
    current_x = pos[layer_members, 0]
    barycenter = torch.where(has_neigh, sum_x / count.clamp(min=1.0), current_x)
    return barycenter


def _active_edge_index(
    problem: LayoutProblem,
    state: SolveState,
    pos: torch.Tensor,
) -> torch.Tensor:
    """Return the edge tensor for the active barycenter graph.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable original graph inputs.
    state : SolveState
        Mutable solve state that may carry ``extras["expanded_graph"]``.
    pos : torch.Tensor
        Active position tensor with shape ``[N_active, 2]``.

    Returns
    -------
    torch.Tensor
        Edge tensor with shape ``[2, E_active]`` on ``pos.device``.
    """
    expanded_graph = state.extras.get("expanded_graph")
    if expanded_graph is not None and int(getattr(expanded_graph, "num_nodes", -1)) == int(
        pos.shape[0]
    ):
        return expanded_graph.edge_index.to(device=pos.device, dtype=torch.long)
    return problem.edge_index.to(device=pos.device, dtype=torch.long)


@register_op
@dataclass
class BarycenterReorder(Op):
    """Sugiyama barycenter polish pass over the finest-level layout.

    Reads ``state.pos`` + ``state.layers`` + ``state.layer_index`` +
    ``problem.edge_index``. For each layer (bottom -> top, then
    top -> bottom, iterated ``config.iterations`` times), compute each
    node's barycenter from its adjacent-layer neighbours and permute
    x-coordinates within the layer to match the barycenter order. The
    SET of x-positions per layer is preserved; only the assignment
    changes, so spacing / overlap / average layer position all stay put.
    """

    name: ClassVar[str] = "barycenter_reorder"
    category: ClassVar[OpCategory] = OpCategory.PROJECT
    reads: ClassVar[Tuple[str, ...]] = ("pos", "layers", "layer_index", "extras.expanded_graph")
    writes: ClassVar[Tuple[str, ...]] = ("pos",)
    requires: ClassVar[Tuple[str, ...]] = ("pos",)

    config: BarycenterReorderConfig = field(default_factory=BarycenterReorderConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        del ctx
        if not self.config.enabled:
            return state
        if state.pos is None or state.layers is None:
            return state
        if state.layer_index is None or state.layer_index.num_layers < 2:
            # Nothing to reorder on a single-layer graph.
            return state

        pos = state.pos
        layers = state.layers.to(device=pos.device, dtype=torch.long)
        layer_index = state.layer_index
        edge_index = _active_edge_index(problem=problem, state=state, pos=pos)
        src_all = edge_index[0]
        tgt_all = edge_index[1]
        src_layer = layers[src_all]
        tgt_layer = layers[tgt_all]

        # Work on a detached copy so we don't perturb any gradient graph
        # a caller might have (the polish is not differentiable).
        pos_new = pos.detach().clone()

        num_layers = layer_index.num_layers
        iterations = max(self.config.iterations, 0)
        min_size = max(self.config.min_layer_size, 2)

        for it in range(iterations):
            # Alternate direction: even iterations sweep bottom->top
            # (use upstream layer k-1 as neighbour reference), odd
            # iterations top->bottom (use downstream layer k+1).
            direction_up = (it % 2) == 0
            layer_order = range(1, num_layers) if direction_up else range(num_layers - 1, -1, -1)
            for k in layer_order:
                members = layer_index.nodes_in_layer(k).to(device=pos_new.device)
                if members.numel() < min_size:
                    continue

                if direction_up:
                    # Barycenter of in-neighbours (edges with tgt in layer k).
                    adj_layer_idx = k - 1
                    mask = (tgt_layer == k) & (src_layer == adj_layer_idx)
                    neighbour_src = tgt_all[mask]
                    neighbour_dst = src_all[mask]
                else:
                    adj_layer_idx = k + 1
                    if adj_layer_idx >= num_layers:
                        continue
                    mask = (src_layer == k) & (tgt_layer == adj_layer_idx)
                    neighbour_src = src_all[mask]
                    neighbour_dst = tgt_all[mask]

                barycenters = _compute_barycenters_for_layer(
                    pos_new,
                    members,
                    mask,
                    neighbour_src,
                    neighbour_dst,
                )
                # Stable sort by barycenter. Ties keep their previous
                # order (preserving determinism).
                order = torch.argsort(barycenters, stable=True)

                # Reassign the SAME sorted x-positions to members in the
                # new order. Y stays put.
                current_x = pos_new[members, 0]
                sorted_x, _ = torch.sort(current_x)
                new_member_order = members[order]
                pos_new[new_member_order, 0] = sorted_x

        state.pos = pos_new.detach().requires_grad_(pos.requires_grad)
        return state


__all__ = ["BarycenterReorder", "BarycenterReorderConfig"]
