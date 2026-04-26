"""Topology-dispatched adapter for dagua's native tensor layout engine."""

from __future__ import annotations

import copy
from typing import Any, Callable, Optional

import torch

from dagua.config import LayoutConfig
from dagua.layout.graph_classify import GraphFamily, GraphStructure, classify_graph
from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.pipelines import dagua_native_legacy
from dagua.layout.ops.pipelines._native_shared import (
    _prepare_native_config,
    _should_apply_brandes_koepf_refine,
    _should_decompose_components,
    _should_use_native_dummy_nodes,
    _should_use_native_median_transpose,
    _tile_component_positions,
    build_gradient_core,
)
from dagua.layout.ops.pipelines.native_force_directed import (
    build_native_force_directed_pipeline,
    layout_native_force_directed_pipeline,
)
from dagua.layout.ops.pipelines.native_hybrid import build_native_hybrid_pipeline
from dagua.layout.ops.pipelines.native_layered_dag import build_native_layered_dag_pipeline
from dagua.layout.ops.pipelines.native_planar import (
    PlanarityFailure,
    build_native_planar_pipeline,
)
from dagua.layout.ops.pipelines.native_tree import build_native_tree_pipeline
from dagua.layout.ops.postprocess import AspectRatioFit, AspectRatioFitConfig
from dagua.layout.ops.preprocess import DetectComponents
from dagua.layout.ops.state import (
    ExecutionPlan,
    FlexConstraints,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.resolve import build_flex_constraints, normalize_node_sizes

_COMPONENT_DOMINANCE_SKIP_FRACTION = 0.85


def _selected_force_pipeline(config: LayoutConfig) -> Optional[str]:
    """Return the user-selected native sub-pipeline override.

    Parameters
    ----------
    config : LayoutConfig
        Layout configuration.

    Returns
    -------
    str | None
        Normalized force-pipeline value.
    """
    value = getattr(config, "force_pipeline", None)
    if value is None:
        return None
    return str(value).lower()


def _choose_native_pipeline(structure: Optional[GraphStructure], config: LayoutConfig) -> str:
    """Choose a native sub-pipeline for one prepared problem.

    Parameters
    ----------
    structure : GraphStructure, optional
        Classified graph topology.
    config : LayoutConfig
        Prepared layout configuration.

    Returns
    -------
    str
        One of ``"tree"``, ``"layered_dag"``, ``"force_directed"``,
        ``"hybrid"``, or ``"legacy_monolith"``.
    """
    forced = _selected_force_pipeline(config)
    if forced in {"tree", "layered_dag", "force_directed", "hybrid", "planar", "legacy_monolith"}:
        return forced
    if structure is None:
        return "layered_dag"

    family = structure.family
    num_nodes = int(getattr(config, "_dagua_native_num_nodes", 0))
    small_tree_cutoff = int(getattr(config, "small_n_tree_cutoff", 64))
    if num_nodes <= small_tree_cutoff and family in {GraphFamily.TREE, GraphFamily.CHAIN}:
        return "tree"
    # Sprint-20g: planar dispatch when the classifier confirms exact
    # planarity AND the user has explicitly opted in via try_planar_first.
    # Default is False because the current Schnyder-init + flat-stress
    # planar pipeline drops the dag_consistency / depth_spearman bonus
    # that layered_dag earns on planar DAGs (loses 3-35 composite points
    # vs layered_dag on every benchmark candidate).
    if getattr(config, "try_planar_first", False) and bool(getattr(structure, "is_planar", False)):
        return "planar"
    cyclicity_ratio = float(getattr(structure, "cyclicity_ratio", 0.0))
    # Sprint-20g: removed auto-route to force_directed. Empirically the
    # PivotMDS+Stress force pipeline loses to layered_dag/hybrid on every
    # cyclic benchmark candidate today (2026-04-24 measurement). Users can
    # still opt in via force_pipeline="force_directed".
    if family == GraphFamily.FORCE_DIRECTED and cyclicity_ratio > 0.5:
        return "force_directed"
    if family == GraphFamily.HYBRID or cyclicity_ratio > 0.05:
        return "hybrid"
    return "layered_dag"


def build_dagua_pipeline(config: LayoutConfig) -> Pipeline:
    """Build the topology-selected native pipeline.

    Parameters
    ----------
    config : LayoutConfig
        Prepared native configuration with ``_dagua_native_*`` metadata.

    Returns
    -------
    Pipeline
        Selected native sub-pipeline.
    """
    structure = getattr(config, "_dagua_native_structure", None) or getattr(
        config,
        "structure",
        None,
    )
    selected = _choose_native_pipeline(structure=structure, config=config)
    if selected == "legacy_monolith":
        return dagua_native_legacy.build_dagua_pipeline(config)
    if selected == "tree":
        return build_native_tree_pipeline(config)
    if selected == "planar":
        return build_native_planar_pipeline(config)
    if selected == "force_directed":
        return build_native_force_directed_pipeline(config)
    if selected == "hybrid":
        return build_native_hybrid_pipeline(config)
    return build_native_layered_dag_pipeline(config)


def _has_pins(flex: Optional[FlexConstraints]) -> bool:
    """Return whether prepared flex constraints contain pinned nodes.

    Parameters
    ----------
    flex : FlexConstraints, optional
        Prepared flex constraints for the current problem.

    Returns
    -------
    bool
        ``True`` when at least one pin is present.
    """
    if flex is None or flex.pin_indices is None:
        return False
    return int(flex.pin_indices.numel()) > 0


def _has_cross_component_flex(
    flex: Optional[FlexConstraints],
    component_ids: Optional[torch.Tensor],
) -> bool:
    """Return whether any alignment group spans multiple weak components.

    Parameters
    ----------
    flex : FlexConstraints, optional
        Prepared flex constraints.
    component_ids : torch.Tensor, optional
        Weak-component labels with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` when an alignment group references multiple components.
    """
    if flex is None or component_ids is None or not flex.align_groups:
        return False

    labels = component_ids.to(dtype=torch.long)
    for group_indices, _, _ in flex.align_groups:
        members = group_indices.to(device=labels.device, dtype=torch.long)
        if members.numel() < 2:
            continue
        if torch.unique(labels[members], sorted=False).numel() > 1:
            return True
    return False


def _should_decompose_native_components(
    problem: LayoutProblem,
    config: LayoutConfig,
    component_ids: Optional[torch.Tensor],
) -> bool:
    """Return whether native dispatch should solve weak components separately.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared parent layout problem.
    config : LayoutConfig
        Prepared native configuration.
    component_ids : torch.Tensor, optional
        Weak-component labels with shape ``[N]``.

    Returns
    -------
    bool
        ``True`` when independent component solving is safe and useful.
    """
    forced = _selected_force_pipeline(config)
    if forced == "legacy_monolith":
        return _should_decompose_components(problem, config, component_ids)
    if not getattr(config, "decompose_components", True):
        return False
    if problem.num_nodes < 2 or problem.clusters or _has_pins(problem.flex):
        return False

    structure = problem.structure
    if structure is not None:
        if int(getattr(structure, "num_components", 1)) <= 1:
            return False
        if bool(getattr(structure, "has_dominant_component", False)):
            return False

    if component_ids is None or component_ids.numel() == 0:
        return False
    if int(component_ids.max().item()) <= 0:
        return False
    component_sizes = torch.bincount(component_ids.to(dtype=torch.long))
    if component_sizes.numel() > 0:
        largest_component = int(component_sizes.max().item())
        if largest_component / max(problem.num_nodes, 1) >= _COMPONENT_DOMINANCE_SKIP_FRACTION:
            return False
    if _has_cross_component_flex(problem.flex, component_ids):
        return False
    return True


def _subset_flex(
    flex: Optional[FlexConstraints],
    local_index: torch.Tensor,
) -> Optional[FlexConstraints]:
    """Project parent flex constraints into component-local node ids.

    Parameters
    ----------
    flex : FlexConstraints, optional
        Parent flex constraints.
    local_index : torch.Tensor
        Parent-to-local node map with shape ``[N_parent]``.

    Returns
    -------
    FlexConstraints | None
        Child-local flex constraints.
    """
    return dagua_native_legacy._subset_flex(flex, local_index)


def _extract_component_problem(
    parent_problem: LayoutProblem,
    parent_state: SolveState,
    component_nodes: torch.Tensor,
    layer_assignments: Optional[torch.Tensor] = None,
) -> tuple[LayoutProblem, SolveState, torch.Tensor, Optional[torch.Tensor]]:
    """Build one relabeled child problem for a weak component.

    Parameters
    ----------
    parent_problem : LayoutProblem
        Prepared parent problem.
    parent_state : SolveState
        Parent solve state.
    component_nodes : torch.Tensor
        Parent node ids in this component with shape ``[K]``.
    layer_assignments : torch.Tensor, optional
        Optional parent layer assignments with shape ``[N_parent]``.

    Returns
    -------
    tuple[LayoutProblem, SolveState, torch.Tensor, torch.Tensor | None]
        Child problem, child state, parent indices, and child layer assignments.
    """
    return dagua_native_legacy._extract_component_problem(
        parent_problem,
        parent_state,
        component_nodes,
        layer_assignments,
    )


def _run_native_problem(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    config: LayoutConfig,
) -> torch.Tensor:
    """Run the selected native sub-pipeline for one prepared problem.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared layout problem for one component or full graph.
    state : SolveState
        Mutable solve state.
    ctx : RuntimeContext
        Runtime execution context.
    config : LayoutConfig
        Prepared native configuration.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    structure = problem.structure or getattr(config, "_dagua_native_structure", None)
    if structure is None:
        structure = classify_graph(problem.edge_index, problem.num_nodes)
        problem.structure = structure

    selected = _choose_native_pipeline(structure=structure, config=config)
    if selected == "legacy_monolith":
        return dagua_native_legacy._run_native_problem(problem, state, ctx, config)
    if selected == "force_directed":
        return layout_native_force_directed_pipeline(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            node_sizes=problem.node_sizes,
            config=config,
            seed=problem.seed,
            edge_weights=problem.edge_weights,
        )

    try:
        final_state = build_dagua_pipeline(config).apply(problem, state, ctx)
    except PlanarityFailure:
        if _selected_force_pipeline(config) == "planar":
            raise
        # Auto-routed to planar but validation failed at runtime (e.g.
        # disconnected components). Fall back to the standard topology
        # selection without trying planar again.
        fallback_config = copy.copy(config)
        fallback_config.try_planar_first = False
        final_state = build_dagua_pipeline(fallback_config).apply(problem, state, ctx)
        selected = _choose_native_pipeline(structure=structure, config=fallback_config)
    if final_state.pos is None:
        raise RuntimeError(f"native {selected} pipeline did not produce final positions.")
    result = final_state.pos.detach()
    if result.shape[0] > problem.num_nodes:
        result = result[: problem.num_nodes]
    # Sprint-20k: best-of-polish edge-equalize. The gradient pipeline
    # converges to a local minimum where edge_length_variance_loss is
    # saturated (confirmed empirically: w=0..200 produces identical
    # output on the loss-bucket graphs). A direct constraint projection
    # toward the mean edge length, scored against the un-polished
    # baseline, escapes that minimum on most layered DAGs and lattices.
    # Gated by force_pipeline=None and bool flag for opt-out.
    if (
        getattr(config, "edge_equalize_polish", True)
        and _selected_force_pipeline(config) is None
        and selected in {"layered_dag", "tree", "hybrid", "force_directed"}
        and result.shape[0] >= 4
        and problem.edge_index.numel() > 0
        and problem.node_sizes is not None
    ):
        result = _best_of_polish(result, problem.edge_index, problem.node_sizes)
    return result


_POLISH_SETTINGS: tuple[tuple[int, float], ...] = (
    (5, 0.05),
    (10, 0.05),
    (20, 0.03),
    (10, 0.10),
    (30, 0.02),
    # Sprint-20l: aggressive variants picked up by petersen_10 (+3.95
    # composite) and disconnected_label_cycle_collage (+2.96). Other
    # graphs keep the un-polished baseline because the picker's 0.5-
    # margin gate filters out the regressions these two cause.
    (50, 0.05),
    (50, 0.20),
)

_Y_LAYER_SNAP_EPS = 0.5
_ORTHOGONAL_ALIGN_ITERS = 10
_ORTHOGONAL_ALIGN_STEP = 0.1
_OVERLAP_JITTER_MAX_NODES = 500
_OVERLAP_JITTER_PADDING = 2.0
_OVERLAP_JITTER_ITERS = 5
_OVERLAP_JITTER_STEP = 0.5
_ANTI_CROSSING_MAX_NODES = 200
_ANTI_CROSSING_MAX_EDGES = 400
_ANTI_CROSSING_MAX_SWAPS = 50
_LAYER_X_KMEANS_MIN_NODES = 24
_LAYER_X_KMEANS_MAX_NODES = 400
_LAYER_X_KMEANS_MIN_EDGE_NODE_RATIO = 1.2
_LAYER_X_KMEANS_MAX_EDGE_NODE_RATIO = 2.0
_LAYER_X_KMEANS_MAX_LAYER_WIDTH_CV = 0.30
_LAYER_X_KMEANS_ITERS = 8


def _equalize_edges(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    iters: int,
    step: float,
) -> torch.Tensor:
    """Run direct constraint projection toward the mean edge length.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    iters : int
        Number of projection iterations.
    step : float
        Per-iteration step size in [0, 1].

    Returns
    -------
    torch.Tensor
        Polished position tensor with shape ``[N, 2]``.
    """
    pos = pos.detach().clone()
    if edge_index.numel() == 0:
        return pos
    src = edge_index[0]
    tgt = edge_index[1]
    mask = src != tgt
    if not bool(mask.any().item()):
        return pos
    src = src[mask]
    tgt = tgt[mask]
    for _ in range(iters):
        diffs = pos[tgt] - pos[src]
        dists = diffs.pow(2).sum(-1).sqrt().clamp(min=1.0)
        target = float(dists.mean().item())
        unit = diffs / dists.unsqueeze(-1)
        delta = (dists - target).unsqueeze(-1) * unit * step
        pos.index_add_(0, src, delta * 0.5)
        pos.index_add_(0, tgt, -delta * 0.5)
    return pos


def _y_layer_snap(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    layer_eps: float = _Y_LAYER_SNAP_EPS,
) -> torch.Tensor:
    """Snap near-horizontal y-bands to their median ordinate.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``. Present for the polish-candidate
        call signature; y-band snapping only needs positions and node sizes.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    layer_eps : float, default=_Y_LAYER_SNAP_EPS
        Mean-node-height multiplier used to bucket nearby y coordinates.

    Returns
    -------
    torch.Tensor
        Position tensor with layer-local y jitter removed.
    """
    del edge_index
    cand = pos.detach().clone()
    if cand.shape[0] < 2 or node_sizes.numel() == 0:
        return cand
    band = float(node_sizes[:, 1].mean().item()) * layer_eps
    if band <= 1e-6:
        return cand
    buckets = torch.round(cand[:, 1] / band).to(dtype=torch.long)
    for bucket in torch.unique(buckets, sorted=False):
        idx = torch.nonzero(buckets == bucket, as_tuple=False).squeeze(1)
        if idx.numel() > 1:
            cand[idx, 1] = cand[idx, 1].median()
    return cand


def _orthogonal_align(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    iters: int = _ORTHOGONAL_ALIGN_ITERS,
    step: float = _ORTHOGONAL_ALIGN_STEP,
) -> torch.Tensor:
    """Nudge each edge toward its dominant horizontal or vertical axis.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``. Present for the polish-candidate
        call signature; orthogonal alignment only needs positions and edges.
    iters : int, default=_ORTHOGONAL_ALIGN_ITERS
        Number of nudge iterations.
    step : float, default=_ORTHOGONAL_ALIGN_STEP
        Per-iteration fraction of cross-axis displacement to remove.

    Returns
    -------
    torch.Tensor
        Position tensor with edge directions pulled toward cardinal axes.
    """
    del node_sizes
    cand = pos.detach().clone()
    if edge_index.numel() == 0:
        return cand
    src = edge_index[0]
    tgt = edge_index[1]
    mask = src != tgt
    if not bool(mask.any().item()):
        return cand
    src = src[mask]
    tgt = tgt[mask]
    for _ in range(iters):
        diffs = cand[tgt] - cand[src]
        is_vertical = diffs[:, 1].abs() >= diffs[:, 0].abs()
        delta = torch.zeros_like(diffs)
        # Positive deltas move endpoints toward each other on the
        # non-dominant axis; the sign in the research sketch was inverted.
        delta[is_vertical, 0] = diffs[is_vertical, 0] * step
        delta[~is_vertical, 1] = diffs[~is_vertical, 1] * step
        cand.index_add_(0, src, delta * 0.5)
        cand.index_add_(0, tgt, -delta * 0.5)
    return cand


def _overlap_jitter(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float = _OVERLAP_JITTER_PADDING,
    iters: int = _OVERLAP_JITTER_ITERS,
    step: float = _OVERLAP_JITTER_STEP,
    max_nodes: int = _OVERLAP_JITTER_MAX_NODES,
) -> torch.Tensor:
    """Push overlapping node boxes apart with a bounded pairwise pass.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``. Present for the polish-candidate
        call signature; overlap recovery only needs positions and node sizes.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    padding : float, default=_OVERLAP_JITTER_PADDING
        Additional box separation target in layout units.
    iters : int, default=_OVERLAP_JITTER_ITERS
        Number of pairwise recovery passes.
    step : float, default=_OVERLAP_JITTER_STEP
        Fraction of the minimum separating displacement to apply per pass.
    max_nodes : int, default=_OVERLAP_JITTER_MAX_NODES
        Largest graph size allowed for the O(N^2) pairwise tensor.

    Returns
    -------
    torch.Tensor
        Position tensor after deterministic overlap recovery.
    """
    del edge_index
    cand = pos.detach().clone()
    num_nodes = cand.shape[0]
    if num_nodes < 2 or num_nodes > max_nodes or node_sizes.numel() == 0:
        return cand
    eye = torch.eye(num_nodes, dtype=torch.bool, device=cand.device)
    node_ids = torch.arange(num_nodes, device=cand.device)
    fallback_sign = torch.where(
        node_ids[:, None] >= node_ids[None, :],
        torch.ones((num_nodes, num_nodes), dtype=cand.dtype, device=cand.device),
        -torch.ones((num_nodes, num_nodes), dtype=cand.dtype, device=cand.device),
    )
    for _ in range(iters):
        diffs = cand[:, None, :] - cand[None, :, :]
        dx = diffs[..., 0].abs()
        dy = diffs[..., 1].abs()
        half_w = (node_sizes[:, 0:1] + node_sizes[:, 0:1].T) * 0.5 + padding
        half_h = (node_sizes[:, 1:2] + node_sizes[:, 1:2].T) * 0.5 + padding
        overlap_x = (half_w - dx).clamp(min=0.0)
        overlap_y = (half_h - dy).clamp(min=0.0)
        overlaps = (overlap_x > 0) & (overlap_y > 0) & ~eye
        if not bool(overlaps.any().item()):
            break
        sign_x = torch.where(diffs[..., 0].abs() > 1e-6, torch.sign(diffs[..., 0]), fallback_sign)
        sign_y = torch.where(diffs[..., 1].abs() > 1e-6, torch.sign(diffs[..., 1]), fallback_sign)
        use_x = overlap_x <= overlap_y
        push = torch.zeros_like(diffs)
        push[..., 0] = torch.where(overlaps & use_x, sign_x * overlap_x, push[..., 0])
        push[..., 1] = torch.where(overlaps & ~use_x, sign_y * overlap_y, push[..., 1])
        cand = cand + push.sum(dim=1) * (step * 0.5)
    return cand


def _segments_cross_scalar(
    a: tuple[float, float],
    b: tuple[float, float],
    c: tuple[float, float],
    d: tuple[float, float],
) -> bool:
    """Return whether two open line segments cross.

    Parameters
    ----------
    a, b, c, d : tuple[float, float]
        Segment endpoints in two-dimensional coordinates.

    Returns
    -------
    bool
        ``True`` when the two non-collinear open segments intersect.
    """

    def cross(
        origin: tuple[float, float],
        left: tuple[float, float],
        right: tuple[float, float],
    ) -> float:
        """Return signed area for three scalar points.

        Parameters
        ----------
        origin : tuple[float, float]
            Origin point for the orientation test.
        left : tuple[float, float]
            First comparison point.
        right : tuple[float, float]
            Second comparison point.

        Returns
        -------
        float
            Signed twice-area of the triangle.
        """
        return (left[0] - origin[0]) * (right[1] - origin[1]) - (left[1] - origin[1]) * (
            right[0] - origin[0]
        )

    d1 = cross(c, d, a)
    d2 = cross(c, d, b)
    d3 = cross(a, b, c)
    d4 = cross(a, b, d)
    return ((d1 > 0.0) != (d2 > 0.0)) and ((d3 > 0.0) != (d4 > 0.0))


def _crossing_edge_pairs(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    max_pairs: int = 512,
) -> list[tuple[int, int]]:
    """Return exact non-incident crossing edge pairs for a small graph.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    max_pairs : int, default=512
        Maximum number of crossing pairs to collect.

    Returns
    -------
    list[tuple[int, int]]
        Crossing edge-index pairs.
    """
    if edge_index.numel() == 0 or edge_index.shape[1] < 2:
        return []
    cpu_pos = pos.detach().cpu().to(dtype=torch.float32)
    cpu_edges = edge_index.detach().cpu().to(dtype=torch.long)
    coords = [(float(x), float(y)) for x, y in cpu_pos.tolist()]
    pairs: list[tuple[int, int]] = []
    num_edges = int(cpu_edges.shape[1])
    for left in range(num_edges):
        u = int(cpu_edges[0, left].item())
        v = int(cpu_edges[1, left].item())
        if u == v:
            continue
        for right in range(left + 1, num_edges):
            a = int(cpu_edges[0, right].item())
            b = int(cpu_edges[1, right].item())
            if a == b or len({u, v, a, b}) < 4:
                continue
            if _segments_cross_scalar(coords[u], coords[v], coords[a], coords[b]):
                pairs.append((left, right))
                if len(pairs) >= max_pairs:
                    return pairs
    return pairs


def _y_layer_buckets(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    layer_eps: float = _Y_LAYER_SNAP_EPS,
) -> torch.Tensor:
    """Return y-band buckets inferred from positions and node heights.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    layer_eps : float, default=_Y_LAYER_SNAP_EPS
        Mean-node-height multiplier used to bucket nearby y coordinates.

    Returns
    -------
    torch.Tensor
        Integer bucket id for each node with shape ``[N]``.
    """
    if node_sizes.numel() == 0:
        return torch.arange(pos.shape[0], dtype=torch.long, device=pos.device)
    band = float(node_sizes[:, 1].mean().item()) * layer_eps
    if band <= 1e-6:
        return torch.arange(pos.shape[0], dtype=torch.long, device=pos.device)
    return torch.round(pos[:, 1] / band).to(dtype=torch.long)


def _swap_2opt_anti_crossing(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    score_fn: Callable[[torch.Tensor], float],
    max_swaps: int = _ANTI_CROSSING_MAX_SWAPS,
) -> torch.Tensor:
    """Try adjacent same-layer x swaps that improve composite score.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    score_fn : Callable[[torch.Tensor], float]
        Composite scoring function used to accept or reject local swaps.
    max_swaps : int, default=_ANTI_CROSSING_MAX_SWAPS
        Maximum number of adjacent swap attempts.

    Returns
    -------
    torch.Tensor
        Position tensor after accepted crossing-reduction swaps.
    """
    num_nodes = int(pos.shape[0])
    num_edges = int(edge_index.shape[1]) if edge_index.numel() > 0 else 0
    cand = pos.detach().clone()
    if (
        num_nodes > _ANTI_CROSSING_MAX_NODES
        or num_edges > _ANTI_CROSSING_MAX_EDGES
        or num_nodes < 4
        or num_edges < 2
    ):
        return cand
    crossing_pairs = _crossing_edge_pairs(cand, edge_index, max_pairs=512)
    if not crossing_pairs:
        return cand

    layers = _y_layer_buckets(cand, node_sizes)
    current_score = score_fn(cand)
    attempts = 0
    for _ in range(2):
        crossing_pairs = _crossing_edge_pairs(cand, edge_index, max_pairs=512)
        if not crossing_pairs:
            break
        crossing_edges = {edge_id for pair in crossing_pairs for edge_id in pair}
        crossing_nodes = torch.zeros(num_nodes, dtype=torch.bool, device=cand.device)
        for edge_id in crossing_edges:
            crossing_nodes[edge_index[0, edge_id]] = True
            crossing_nodes[edge_index[1, edge_id]] = True

        accepted_this_pass = False
        for layer in torch.unique(layers, sorted=True):
            layer_nodes = torch.nonzero(layers == layer, as_tuple=False).squeeze(1)
            if layer_nodes.numel() < 2:
                continue
            order = torch.argsort(cand[layer_nodes, 0], stable=True)
            ordered_nodes = layer_nodes[order]
            for left_idx in range(int(ordered_nodes.numel()) - 1):
                if attempts >= max_swaps:
                    return cand
                left_node = ordered_nodes[left_idx]
                right_node = ordered_nodes[left_idx + 1]
                if not bool((crossing_nodes[left_node] & crossing_nodes[right_node]).item()):
                    continue
                trial = cand.clone()
                left_x = trial[left_node, 0].clone()
                trial[left_node, 0] = trial[right_node, 0]
                trial[right_node, 0] = left_x
                attempts += 1
                if not bool(torch.isfinite(trial).all().item()):
                    continue
                try:
                    trial_score = score_fn(trial)
                except Exception:
                    continue
                if trial_score > current_score:
                    cand = trial
                    current_score = trial_score
                    accepted_this_pass = True
                    break
            if accepted_this_pass:
                break
        if not accepted_this_pass:
            break
    return cand


def _should_layer_x_kmeans(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Return whether a graph matches the lattice-like x-quantization gate.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    bool
        ``True`` when the graph satisfies the conservative layer-width,
        edge-density, and size gates from the sprint brief.
    """
    if (
        num_nodes < _LAYER_X_KMEANS_MIN_NODES
        or num_nodes > _LAYER_X_KMEANS_MAX_NODES
        or edge_index.numel() == 0
    ):
        return False
    num_edges = int(edge_index.shape[1])
    edge_to_node = float(num_edges) / float(max(num_nodes, 1))
    if not (
        _LAYER_X_KMEANS_MIN_EDGE_NODE_RATIO <= edge_to_node <= _LAYER_X_KMEANS_MAX_EDGE_NODE_RATIO
    ):
        return False
    try:
        structure = classify_graph(edge_index.detach().cpu(), num_nodes)
    except Exception:
        return False
    return (
        bool(getattr(structure, "is_directed_acyclic", True))
        and int(getattr(structure, "num_layers", 0)) >= 5
        and float(getattr(structure, "layer_width_cv", 1.0)) <= _LAYER_X_KMEANS_MAX_LAYER_WIDTH_CV
    )


def _per_layer_x_kmeans(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    max_iters: int = _LAYER_X_KMEANS_ITERS,
) -> torch.Tensor:
    """Quantize x coordinates by running 1-D K-means inside each layer.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``. Present for the polish-candidate
        call signature; layer x-quantization only needs positions and edges.
    max_iters : int, default=_LAYER_X_KMEANS_ITERS
        Maximum K-means iterations per layer.

    Returns
    -------
    torch.Tensor
        Position tensor with layer-local x coordinates snapped to centroids.
    """
    del node_sizes
    cand = pos.detach().clone()
    num_nodes = int(cand.shape[0])
    if not _should_layer_x_kmeans(edge_index, num_nodes):
        return cand
    try:
        from dagua.utils import longest_path_layering

        raw_layers = longest_path_layering(edge_index.detach().cpu(), num_nodes)
    except Exception:
        return cand
    layer_tensor = torch.as_tensor(raw_layers, dtype=torch.long, device=cand.device)
    unique_layers, counts = torch.unique(layer_tensor, sorted=True, return_counts=True)
    if unique_layers.numel() < 5:
        return cand
    median_width = int(torch.median(counts.to(dtype=torch.float32)).round().item())
    if median_width < 2:
        return cand

    for layer in unique_layers:
        idx = torch.nonzero(layer_tensor == layer, as_tuple=False).squeeze(1)
        layer_count = int(idx.numel())
        k = min(layer_count, median_width)
        if layer_count <= 2 or k >= layer_count or k < 2:
            continue
        values = cand[idx, 0]
        sorted_values = torch.sort(values).values
        init_positions = torch.linspace(0, layer_count - 1, k, device=cand.device).round().long()
        centers = sorted_values[init_positions].clone()
        labels = torch.zeros(layer_count, dtype=torch.long, device=cand.device)
        for _ in range(max_iters):
            distances = (values[:, None] - centers[None, :]).abs()
            labels = torch.argmin(distances, dim=1)
            new_centers = centers.clone()
            for center_idx in range(k):
                members = values[labels == center_idx]
                if members.numel() > 0:
                    new_centers[center_idx] = members.mean()
            if torch.allclose(new_centers, centers):
                break
            centers = new_centers
        cand[idx, 0] = centers[labels]
    return cand


def _should_dot_lattice_lp(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> bool:
    """Conservative gate for the dot-mimic LP polish candidate.

    The LP candidate is expensive (~10-200 ms per graph). Sprint-22
    area A measured large gains on layered DAGs with low hub-ratio and
    short edge-spans (hex_lattice +9.28, grid_5x5 +16.56), but losses
    on cyclic / hub graphs (parallel_cycles -9.92, hub_and_spoke
    -15.28). Restrict firing to the structural class where the LP is
    competitive.
    """
    if num_nodes < 12 or num_nodes > 2000 or edge_index.numel() == 0:
        return False
    src = edge_index[0]
    tgt = edge_index[1]
    non_self = src != tgt
    src = src[non_self]
    tgt = tgt[non_self]
    e = int(src.numel())
    if e == 0:
        return False
    indeg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    out_adj: list[list[int]] = [[] for _ in range(num_nodes)]
    for i in range(e):
        u = int(src[i].item())
        v = int(tgt[i].item())
        indeg[v] += 1
        out_adj[u].append(v)
    queue = [int(v.item()) for v in torch.nonzero(indeg == 0, as_tuple=False).squeeze(-1)]
    indeg_copy = indeg.clone()
    visited = 0
    while queue:
        u = queue.pop(0)
        visited += 1
        for v in out_adj[u]:
            indeg_copy[v] -= 1
            if int(indeg_copy[v].item()) == 0:
                queue.append(v)
    if visited != num_nodes:
        return False
    parent = list(range(num_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(e):
        a = find(int(src[i].item()))
        b = find(int(tgt[i].item()))
        if a != b:
            parent[a] = b
    if len({find(i) for i in range(num_nodes)}) > 1:
        return False
    deg = torch.zeros(num_nodes, dtype=torch.long, device=edge_index.device)
    for i in range(e):
        deg[int(src[i].item())] += 1
        deg[int(tgt[i].item())] += 1
    deg_sorted = torch.sort(deg).values.to(dtype=torch.float32)
    median_deg = float(deg_sorted[num_nodes // 2].item())
    max_deg = float(deg_sorted[-1].item())
    if median_deg <= 0 or max_deg / median_deg > 4.0:
        return False
    return True


def _dot_lattice_lp(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> torch.Tensor:
    """Replicate graphviz_dot's layered DAG layout via two LPs.

    Implements the Gansner-Koutsofios-North-Vo 1993 pipeline:
    rank-assignment LP -> virtual-node insertion -> median crossing
    reduction -> x-coordinate LP. Sprint-22 area A measured this
    candidate at +9.28 composite on hexagonal_lattice_42, +16.56 on
    grid_5x5, +10.97 on dependency_graph_100, +3.21 on
    complete_bipartite_8x12 over current dagua HEAD positions.

    Inputs match the polish-candidate signature; the candidate ignores
    ``pos`` and synthesizes coordinates from ``edge_index`` directly.
    The picker's 0.5-margin gate handles regression risk.
    """
    n = int(pos.shape[0])
    cand = pos.detach().clone()
    if not _should_dot_lattice_lp(edge_index, n):
        return cand
    try:
        import numpy as np
        from scipy.optimize import linprog
    except Exception:
        return cand

    src = edge_index[0]
    tgt = edge_index[1]
    non_self = src != tgt
    src = src[non_self]
    tgt = tgt[non_self]
    e = int(src.numel())
    if e == 0:
        return cand

    c_rank = np.zeros(n)
    rows: list[np.ndarray] = []
    rhs: list[float] = []
    for i in range(e):
        u = int(src[i].item())
        v = int(tgt[i].item())
        c_rank[v] += 1.0
        c_rank[u] -= 1.0
        row = np.zeros(n)
        row[u] = 1.0
        row[v] = -1.0
        rows.append(row)
        rhs.append(-1.0)
    bounds_rank = [(0, None)] * n
    try:
        res = linprog(
            c=c_rank,
            A_ub=np.array(rows),
            b_ub=np.array(rhs),
            bounds=bounds_rank,
            method="highs",
        )
    except Exception:
        return cand
    if not res.success:
        return cand
    rank_int = [int(round(r)) for r in res.x]
    rmin = min(rank_int)
    rank_int = [r - rmin for r in rank_int]

    new_rank = list(rank_int)
    new_edges: list[tuple[int, int, float]] = []
    for i in range(e):
        u = int(src[i].item())
        v = int(tgt[i].item())
        ru, rv = rank_int[u], rank_int[v]
        if rv <= ru:
            new_edges.append((u, v, 0.0))
            continue
        if rv == ru + 1:
            new_edges.append((u, v, 1.0))
            continue
        prev = u
        for kk in range(ru + 1, rv):
            virt = len(new_rank)
            new_rank.append(kk)
            new_edges.append((prev, virt, 8.0))
            prev = virt
        new_edges.append((prev, v, 8.0))

    n_total = len(new_rank)
    layers: dict[int, list[int]] = {}
    for i in range(n_total):
        layers.setdefault(new_rank[i], []).append(i)
    rmin_l = min(layers)
    rmax_l = max(layers)
    for r_l in layers:
        layers[r_l] = sorted(layers[r_l])
    in_e: list[list[int]] = [[] for _ in range(n_total)]
    out_e: list[list[int]] = [[] for _ in range(n_total)]
    for u, v, w in new_edges:
        if w == 0.0:
            continue
        if new_rank[v] > new_rank[u]:
            out_e[u].append(v)
            in_e[v].append(u)

    def _positions() -> dict[int, int]:
        out: dict[int, int] = {}
        for r_l in layers:
            for j, vv in enumerate(layers[r_l]):
                out[vv] = j
        return out

    for sweep in range(24):
        if sweep % 2 == 0:
            for r_l in range(rmin_l + 1, rmax_l + 1):
                pos_idx = _positions()

                def _key_down(v: int, r_l: int = r_l) -> float:
                    nbr = sorted(pos_idx[u] for u in in_e[v] if new_rank[u] == r_l - 1)
                    if not nbr:
                        return float(pos_idx[v])
                    m = len(nbr)
                    return float(
                        nbr[m // 2] if m % 2 == 1 else 0.5 * (nbr[m // 2 - 1] + nbr[m // 2])
                    )

                layers[r_l] = sorted(layers[r_l], key=_key_down)
        else:
            for r_l in range(rmax_l - 1, rmin_l - 1, -1):
                pos_idx = _positions()

                def _key_up(v: int, r_l: int = r_l) -> float:
                    nbr = sorted(pos_idx[w_v] for w_v in out_e[v] if new_rank[w_v] == r_l + 1)
                    if not nbr:
                        return float(pos_idx[v])
                    m = len(nbr)
                    return float(
                        nbr[m // 2] if m % 2 == 1 else 0.5 * (nbr[m // 2 - 1] + nbr[m // 2])
                    )

                layers[r_l] = sorted(layers[r_l], key=_key_up)

    nodesep = float(node_sizes[:, 0].mean().item()) * 1.5 if node_sizes.numel() else 72.0
    nodesep = max(nodesep, 1.0)
    ranksep = float(node_sizes[:, 1].mean().item()) * 2.0 if node_sizes.numel() else 72.0
    ranksep = max(ranksep, 1.0)

    edges_pos_w = [(u, v, w) for (u, v, w) in new_edges if w > 0]
    e_count = len(edges_pos_w)
    if e_count == 0:
        return cand
    n_vars = n_total + e_count
    cx = np.zeros(n_vars)
    for k, (_, _, w) in enumerate(edges_pos_w):
        cx[n_total + k] = w
    A_ub: list[np.ndarray] = []
    b_ub: list[float] = []
    for k, (u, v, _) in enumerate(edges_pos_w):
        r1 = np.zeros(n_vars)
        r1[n_total + k] = -1.0
        r1[v] = 1.0
        r1[u] = -1.0
        A_ub.append(r1)
        b_ub.append(0.0)
        r2 = np.zeros(n_vars)
        r2[n_total + k] = -1.0
        r2[v] = -1.0
        r2[u] = 1.0
        A_ub.append(r2)
        b_ub.append(0.0)
    for r_l, nodes_in_layer in layers.items():
        for i in range(len(nodes_in_layer) - 1):
            a = nodes_in_layer[i]
            b = nodes_in_layer[i + 1]
            row = np.zeros(n_vars)
            row[a] = 1.0
            row[b] = -1.0
            A_ub.append(row)
            b_ub.append(-nodesep)
    A_eq = np.zeros((1, n_vars))
    A_eq[0, 0] = 1.0
    b_eq = np.array([0.0])
    bounds_x = [(None, None)] * n_total + [(0, None)] * e_count
    try:
        res_x = linprog(
            c=cx,
            A_ub=np.array(A_ub),
            b_ub=np.array(b_ub),
            A_eq=A_eq,
            b_eq=b_eq,
            bounds=bounds_x,
            method="highs",
        )
    except Exception:
        return cand
    if not res_x.success:
        return cand
    x_vals = res_x.x[:n_total]
    x_vals = x_vals - x_vals.min()
    out = torch.zeros((n, 2), dtype=cand.dtype, device=cand.device)
    for v in range(n):
        out[v, 0] = float(x_vals[v])
        out[v, 1] = float(rank_int[v]) * ranksep
    out = out - out.mean(dim=0, keepdim=True)
    return out


def _global_depth_align(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    component_gap_factor: float = 1.5,
) -> torch.Tensor:
    """Align disconnected components on shared global-depth y-rows.

    The default per-component tile lays components row-major by node
    count and area. depth_spearman_rho is computed at the node level
    over ALL nodes globally, so components with overlapping local
    depths but different y-bands break the correlation. Sprint-22
    area C found that re-placing nodes on `y = global_depth * pitch`
    (with components stacked horizontally instead of row-major) lifts
    `disconnected_encoder_residual` from 74.01 to 86.19 (+12.17, flips
    a -1.62 close-loss into a +0.56 win vs elk_layered). The metric
    uses ``dagua.utils.longest_path_layering`` for depth, so this
    function MUST use the same.

    Cycle components (where all nodes share the same
    longest-path-layering "max+1" cycle layer) keep their local y-shape
    rescaled to 0.8 of the global pitch so the component still has
    visible vertical structure but is anchored to a shared row.

    Single-component graphs return ``pos`` unchanged.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``, typically the output of
        ``_tile_component_positions``.
    edge_index : torch.Tensor
        Parent edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``. Unused by the algorithm
        itself; kept for the polish-candidate signature.
    component_gap_factor : float, default=1.5
        Multiplier on the inferred row pitch used as the gap between
        adjacent component columns.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    del node_sizes
    cand = pos.detach().clone()
    n = int(cand.shape[0])
    if n < 4 or edge_index.numel() == 0:
        return cand

    # Undirected connected components.
    src = edge_index[0]
    tgt = edge_index[1]
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(int(edge_index.shape[1])):
        a = find(int(src[i].item()))
        b = find(int(tgt[i].item()))
        if a != b:
            parent[a] = b
    comp_of: dict[int, list[int]] = {}
    for i in range(n):
        root = find(i)
        comp_of.setdefault(root, []).append(i)
    comps = list(comp_of.values())
    if len(comps) < 2:
        return cand

    try:
        from dagua.utils import longest_path_layering

        global_depth = longest_path_layering(edge_index, n)
    except Exception:
        return cand
    depth_t = torch.as_tensor(global_depth, dtype=torch.float32, device=cand.device)

    # Inferred pitch: median per-component median y-step in the input.
    per_comp_pitch: list[float] = []
    for comp in comps:
        comp_idx = torch.tensor(comp, dtype=torch.long, device=cand.device)
        ys = torch.unique(cand[comp_idx, 1])
        if ys.numel() >= 2:
            sorted_ys = torch.sort(ys).values
            steps = sorted_ys[1:] - sorted_ys[:-1]
            steps = steps[steps > 1e-6]
            if steps.numel() > 0:
                per_comp_pitch.append(float(steps.median().item()))
    if not per_comp_pitch:
        return cand
    pitch = float(torch.tensor(per_comp_pitch).median().item())
    if pitch <= 1e-6:
        return cand

    # Vote on y-sign across components: deeper-node = larger y or smaller?
    sign_votes = 0
    for comp in comps:
        comp_idx = torch.tensor(comp, dtype=torch.long, device=cand.device)
        if comp_idx.numel() < 2:
            continue
        y_vals = cand[comp_idx, 1]
        d_vals = depth_t[comp_idx]
        if y_vals.std() <= 1e-6 or d_vals.std() <= 1e-6:
            continue
        cov = float(((y_vals - y_vals.mean()) * (d_vals - d_vals.mean())).mean().item())
        sign_votes += -1 if cov < 0 else 1
    y_sign = -1.0 if sign_votes < 0 else 1.0

    new_pos = cand.clone()
    cursor_x = 0.0
    gap = component_gap_factor * pitch
    for comp in comps:
        comp_idx = torch.tensor(comp, dtype=torch.long, device=cand.device)
        comp_depths = depth_t[comp_idx]
        comp_local_y = cand[comp_idx, 1]
        local_x = cand[comp_idx, 0]
        local_x_min = float(local_x.min().item())
        comp_width = float(local_x.max().item()) - local_x_min
        unique_depths = torch.unique(comp_depths).numel()
        if unique_depths <= 1:
            base_y = float(comp_depths[0].item()) * pitch * y_sign
            local_range = max(
                float(comp_local_y.max().item() - comp_local_y.min().item()),
                1e-6,
            )
            for k in range(comp_idx.numel()):
                node = int(comp_idx[k].item())
                norm_y = (
                    float(cand[node, 1].item()) - float(comp_local_y.min().item())
                ) / local_range
                offset = (norm_y - 0.5) * pitch * 0.8 * y_sign
                new_pos[node, 0] = cursor_x + (float(cand[node, 0].item()) - local_x_min)
                new_pos[node, 1] = base_y + offset
        else:
            for k in range(comp_idx.numel()):
                node = int(comp_idx[k].item())
                new_pos[node, 0] = cursor_x + (float(cand[node, 0].item()) - local_x_min)
                new_pos[node, 1] = float(depth_t[node].item()) * pitch * y_sign
        cursor_x += max(comp_width, pitch) + gap

    new_pos = new_pos - new_pos.mean(dim=0, keepdim=True)
    return new_pos


def _detect_back_edges_dfs(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return a boolean mask marking DFS back-edges + self-loops.

    The mask is shape ``[E]`` and is ``True`` for every edge that closes
    a directed cycle, plus every self-loop. This is a tree-edge / back-
    edge classifier on the directed graph, not a feedback-arc-set
    minimizer; it is sufficient for the relayer polish primitive
    because removing all back-edges always yields an acyclic forward
    graph.
    """
    if edge_index.numel() == 0:
        return torch.zeros(0, dtype=torch.bool)
    src = edge_index[0]
    tgt = edge_index[1]
    self_mask = src == tgt
    adj: list[list[tuple[int, int]]] = [[] for _ in range(num_nodes)]
    for i in range(edge_index.shape[1]):
        s_i = int(src[i].item())
        t_i = int(tgt[i].item())
        if s_i == t_i:
            continue
        adj[s_i].append((t_i, i))
    color = [0] * num_nodes
    back = torch.zeros(edge_index.shape[1], dtype=torch.bool)
    for start in range(num_nodes):
        if color[start] != 0:
            continue
        stack: list[tuple[int, Any]] = [(start, iter(adj[start]))]
        color[start] = 1
        while stack:
            u, it = stack[-1]
            advanced = False
            for v, eidx in it:
                if color[v] == 0:
                    color[v] = 1
                    stack.append((v, iter(adj[v])))
                    advanced = True
                    break
                if color[v] == 1:
                    back[eidx] = True
            if not advanced:
                color[u] = 2
                stack.pop()
    return back | self_mask


def _back_edge_relayer(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    blend: float = 1.0,
) -> torch.Tensor:
    """Re-layer cyclic graphs after removing detected back-edges.

    The gradient pipeline collapses cyclic graphs into compressed y bands
    when its back-edge handling saturates. Sprint-22 area E discovered
    that re-running longest-path layering on the forward DAG (i.e. with
    DFS back-edges removed) and placing each forward layer at uniform y
    pitch lifts cyclic targets by 5-9 composite points:

      * recurrent_feedback_cell  +8.17 (66.73 -> 74.90, beats every comp)
      * small_world_100          +8.65 (matches sprint-20i stress route)
      * small_world_500          +8.07 (1000x SNR confirmed)
      * braided_feedback_tails   +5.85
      * parallel_cycles_4x5      +5.03

    Acyclic graphs see ``back.sum() == 0`` and exit unchanged.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``. Used for x pitch only.
    blend : float, default=1.0
        Mixing factor between the original ``pos`` (0.0) and the
        re-layered output (1.0). The picker tries several blends.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    cand = pos.detach().clone()
    n = int(cand.shape[0])
    if n < 4 or edge_index.numel() == 0:
        return cand
    back = _detect_back_edges_dfs(edge_index, n)
    # Skip if no non-self back-edges -- a self-loop alone doesn't
    # justify rebuilding the layout.
    src = edge_index[0]
    tgt = edge_index[1]
    non_self_back = bool((back & (src != tgt)).any().item())
    if not non_self_back:
        return cand

    forward_ei = edge_index[:, ~back]
    try:
        from dagua.utils import longest_path_layering

        layers = longest_path_layering(forward_ei, n)
    except Exception:
        return cand
    layer_t = torch.as_tensor(layers, dtype=torch.long, device=cand.device)

    forward_mask = ~back
    if bool(forward_mask.any().item()):
        edge_lens = (cand[tgt[forward_mask]] - cand[src[forward_mask]]).pow(2).sum(-1).sqrt()
        edge_lens = edge_lens[edge_lens > 1e-6]
        pitch_y = float(edge_lens.median().item()) if edge_lens.numel() > 0 else 1.0
    else:
        pitch_y = 1.0
    pitch_y = max(pitch_y, 1.0)

    pitch_x = float(node_sizes[:, 0].mean().item()) * 1.5 if node_sizes.numel() else pitch_y
    pitch_x = max(pitch_x, 1.0)

    new_x = torch.zeros(n, dtype=cand.dtype, device=cand.device)
    new_y = layer_t.to(cand.dtype) * pitch_y
    for layer in torch.unique(layer_t):
        idx = torch.nonzero(layer_t == layer, as_tuple=False).squeeze(1)
        if idx.numel() == 0:
            continue
        order = torch.argsort(cand[idx, 0])
        ordered = idx[order]
        offsets = torch.arange(ordered.numel(), dtype=cand.dtype, device=cand.device)
        offsets = (offsets - (ordered.numel() - 1) / 2.0) * pitch_x
        new_x[ordered] = offsets
    relayered = torch.stack([new_x, new_y], dim=1)
    blend = max(0.0, min(1.0, blend))
    return (1.0 - blend) * cand + blend * relayered


def _should_tutte_cyclic_planar(edge_index: torch.Tensor, num_nodes: int) -> bool:
    """Gate the Tutte polish to disconnected simple directed-cycle graphs.

    The barycentric Tutte solve only beats the gradient pipeline on a very
    narrow target: graphs whose every connected component is a simple
    directed cycle (out-degree 1, in-degree 1, E_c == V_c). On lattice
    patches and 3-connected planar graphs the depth-warp tiebreak inflates
    edge_length_cv past the gradient baseline, so the gate has to be
    strict. See sprint-22 area B for the full empirical envelope:
    parallel_cycles_4x5 wins (+3.25), every other planar lattice loses.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Node count.

    Returns
    -------
    bool
        ``True`` when every component is a simple directed cycle.
    """
    if num_nodes < 6 or edge_index.numel() == 0:
        return False
    e_count = int(edge_index.shape[1])
    if e_count != num_nodes:
        return False  # disjoint cycles satisfy E == V exactly
    src = edge_index[0]
    tgt = edge_index[1]
    if bool((src == tgt).any().item()):
        return False
    indeg = torch.zeros(num_nodes, dtype=torch.long)
    outdeg = torch.zeros(num_nodes, dtype=torch.long)
    indeg.index_add_(0, tgt, torch.ones_like(tgt))
    outdeg.index_add_(0, src, torch.ones_like(src))
    if not bool((indeg == 1).all().item()):
        return False
    if not bool((outdeg == 1).all().item()):
        return False
    # Connected components via union-find on undirected edges.
    parent = list(range(num_nodes))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for i in range(e_count):
        a = find(int(src[i].item()))
        b = find(int(tgt[i].item()))
        if a != b:
            parent[a] = b
    roots = {find(i) for i in range(num_nodes)}
    if len(roots) < 2:
        return False  # require multi-component (single cycle is trivial)
    return True


def _tutte_cyclic_planar(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> torch.Tensor:
    """Per-component classical Tutte + monotone y-warp polish.

    Targets disconnected simple-directed-cycle graphs (parallel_cycles
    family). Each component is embedded by classical Tutte 2D (outer face
    on a regular polygon, interior solved via L_ii * pos = -L_ib *
    boundary), then y is replaced by ``depth * pitch`` from
    ``longest_path_layering`` to guarantee dag_consistency=1 and
    depth_spearman=1. Within-layer x is tiebroken by Tutte-rotation-x
    with a minimum gap of ``0.6 * x_pitch``. Components are packed
    horizontally with gap ``2 * x_pitch``.

    The pitch is inferred from the input ``pos`` so the polished output
    keeps the same scale as the gradient baseline; falls back to
    ``node_sizes`` mean when the input is degenerate.

    Returns the input unchanged when the gate
    (``_should_tutte_cyclic_planar``) rejects the topology, when scipy /
    networkx are unavailable, or when any per-component solve fails.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    cand = pos.detach().clone()
    n = int(cand.shape[0])
    if not _should_tutte_cyclic_planar(edge_index, n):
        return cand
    try:
        import networkx as nx
        import numpy as np
        import scipy.sparse as sp
        import scipy.sparse.linalg as spla

        from dagua.utils import longest_path_layering
    except Exception:
        return cand

    # Pitch inference: median y-step in the input. Sprint-22 area B used
    # equal x and y pitch (72 pt) and pitch ratio is the dominant
    # parameter -- aspect ratios far from 1:1 inflate edge_length_cv and
    # tank the win on parallel_cycles. Default to a single isotropic
    # pitch derived from the input's natural y-step.
    ys = torch.unique(cand[:, 1])
    pitch = float(node_sizes[:, 1].mean().item()) * 2.0 if node_sizes.numel() else 1.0
    if ys.numel() >= 2:
        sorted_ys = torch.sort(ys).values
        steps = sorted_ys[1:] - sorted_ys[:-1]
        steps = steps[steps > 1e-6]
        if steps.numel() > 0:
            pitch = float(steps.median().item())
    pitch = max(pitch, 1.0)
    pitch_y = pitch
    pitch_x = pitch

    raw_depth = longest_path_layering(edge_index, n)
    depth = (
        raw_depth.cpu().numpy()
        if isinstance(raw_depth, torch.Tensor)
        else np.asarray(raw_depth, dtype=np.int64)
    )

    G = nx.Graph()
    G.add_nodes_from(range(n))
    for s, t in edge_index.t().tolist():
        if int(s) != int(t):
            G.add_edge(int(s), int(t))

    def _outer_face(sub: nx.Graph) -> list[int]:
        is_planar, embedding = nx.check_planarity(sub, counterexample=False)
        if not is_planar:
            cb = nx.cycle_basis(sub)
            return cb[0] if cb else list(sub.nodes())[: max(3, len(sub.nodes()) // 4)]
        seen: set[tuple[int, int]] = set()
        faces: list[list[int]] = []
        for v in embedding.nodes():
            for w in embedding.neighbors_cw_order(v):
                if (v, w) in seen:
                    continue
                face = embedding.traverse_face(v, w)
                for i in range(len(face)):
                    seen.add((face[i], face[(i + 1) % len(face)]))
                faces.append(face)
        if not faces:
            return list(sub.nodes())[:3]
        faces.sort(key=lambda f: -len(f))
        return faces[0]

    final = np.zeros((n, 2), dtype=np.float64)
    x_offset = 0.0
    for comp in nx.connected_components(G):
        comp_nodes = sorted(comp)
        n_sub = len(comp_nodes)
        if n_sub == 0:
            continue
        if n_sub == 1:
            v = comp_nodes[0]
            final[v] = (x_offset, depth[v] * pitch_y)
            x_offset += pitch_x * 2.0
            continue
        old_to_new = {v: i for i, v in enumerate(comp_nodes)}
        new_to_old = {i: v for v, i in old_to_new.items()}
        sub = nx.relabel_nodes(G.subgraph(comp_nodes).copy(), old_to_new)
        sub_depth = depth[comp_nodes]

        radius = max(1.0, float(np.sqrt(n_sub))) * pitch_x * 0.5
        boundary = _outer_face(sub)
        if not boundary or len(boundary) < 3:
            boundary = list(range(min(3, n_sub)))
        boundary_set = set(boundary)

        pos2d = np.zeros((n_sub, 2), dtype=np.float64)
        n_b = len(boundary)
        for i, v in enumerate(boundary):
            theta = 2 * np.pi * i / n_b
            pos2d[v, 0] = radius * np.cos(theta)
            pos2d[v, 1] = radius * np.sin(theta)
        interior = [v for v in range(n_sub) if v not in boundary_set]
        if interior:
            int_idx = {v: i for i, v in enumerate(interior)}
            n_int = len(interior)
            edges_local = list(sub.edges())
            rows_ii: list[int] = []
            cols_ii: list[int] = []
            vals_ii: list[float] = []
            rows_ib: list[int] = []
            cols_ib: list[int] = []
            vals_ib: list[float] = []
            deg = np.zeros(n_sub, dtype=np.float64)
            for u, v in edges_local:
                deg[u] += 1.0
                deg[v] += 1.0
                u_in = u in int_idx
                v_in = v in int_idx
                if u_in and v_in:
                    iu, iv = int_idx[u], int_idx[v]
                    rows_ii.extend([iu, iv])
                    cols_ii.extend([iv, iu])
                    vals_ii.extend([-1.0, -1.0])
                elif u_in and not v_in:
                    rows_ib.append(int_idx[u])
                    cols_ib.append(v)
                    vals_ib.append(-1.0)
                elif v_in and not u_in:
                    rows_ib.append(int_idx[v])
                    cols_ib.append(u)
                    vals_ib.append(-1.0)
            diag_rows = list(range(n_int))
            diag_cols = list(range(n_int))
            diag_vals = [deg[v] for v in interior]
            l_ii = sp.csr_matrix(
                (vals_ii + diag_vals, (rows_ii + diag_rows, cols_ii + diag_cols)),
                shape=(n_int, n_int),
            )
            l_ib = sp.csr_matrix(
                (vals_ib, (rows_ib, cols_ib)),
                shape=(n_int, n_sub),
            )
            rhs_x = -l_ib @ pos2d[:, 0]
            rhs_y = -l_ib @ pos2d[:, 1]
            try:
                x_int = spla.spsolve(l_ii, rhs_x)
                y_int = spla.spsolve(l_ii, rhs_y)
            except Exception:
                l_reg = l_ii + sp.eye(n_int) * 1e-6
                try:
                    x_int = spla.spsolve(l_reg, rhs_x)
                    y_int = spla.spsolve(l_reg, rhs_y)
                except Exception:
                    return cand
            for v_loc, xv, yv in zip(interior, x_int, y_int):
                pos2d[v_loc, 0] = float(xv)
                pos2d[v_loc, 1] = float(yv)

        # Monotone y-warp + within-layer x-tiebreak.
        new_x = pos2d[:, 0].copy()
        layers: dict[int, list[int]] = {}
        for i in range(n_sub):
            layers.setdefault(int(sub_depth[i]), []).append(i)
        new_y = np.array([sub_depth[i] * pitch_y for i in range(n_sub)], dtype=np.float64)

        # Normalize new_x to span pitch_x * sqrt(n_sub) before tiebreak so
        # the gap enforcement is meaningful at the right scale.
        if new_x.max() - new_x.min() > 0:
            target_span = pitch_x * float(np.sqrt(n_sub))
            new_x = (new_x - new_x.min()) / (new_x.max() - new_x.min()) * target_span

        min_gap = 0.6 * pitch_x
        for d, members in layers.items():
            members_sorted = sorted(members, key=lambda i: new_x[i])
            n_layer = len(members_sorted)
            if n_layer > 1:
                xs = np.array([new_x[i] for i in members_sorted])
                for k in range(1, n_layer):
                    if xs[k] - xs[k - 1] < min_gap:
                        xs[k] = xs[k - 1] + min_gap
                for i, x_v in zip(members_sorted, xs):
                    new_x[i] = x_v

        for local_i in range(n_sub):
            global_v = new_to_old[local_i]
            final[global_v, 0] = x_offset + new_x[local_i]
            final[global_v, 1] = new_y[local_i]
        comp_width = float(new_x.max() - new_x.min()) if n_sub > 1 else 0.0
        x_offset += comp_width + pitch_x * 2.0

    out = torch.tensor(final, dtype=cand.dtype, device=cand.device)
    out = out - out.mean(dim=0, keepdim=True)
    return out


def _should_gap_swap_large_dag(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
) -> bool:
    """Gate the gap-validated x-swap polish to large dependency-style DAGs.

    The search is only worth running when (a) the graph is large enough
    that the gradient pipeline saturates without exploring all x-orderings
    and (b) edge-length variance is high enough for permutations to find
    real improvements. Sprint-22 area D measured the gain on
    ``dependency_500`` (N=500, baseline CV=0.91) at +0.98 composite; small
    graphs and low-CV graphs (random_dag_200, org_chart_deep,
    hub_fanout_label_skew) regress under forced equalization, so the gate
    must reject them.

    Parameters
    ----------
    pos : torch.Tensor
        Current positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    bool
        ``True`` when the topology and CV justify gap-search.
    """
    n = int(pos.shape[0])
    e = int(edge_index.shape[1]) if edge_index.numel() > 0 else 0
    if n < 200 or e < 2 * n:
        return False
    src = edge_index[0]
    tgt = edge_index[1]
    diffs = pos[tgt] - pos[src]
    lengths = diffs.pow(2).sum(-1).sqrt()
    finite = lengths[torch.isfinite(lengths)]
    if finite.numel() == 0:
        return False
    mean = float(finite.mean().item())
    if mean <= 1e-6:
        return False
    std = float(finite.std().item())
    cv = std / mean
    return cv >= 0.75


def _gap_validated_layer_swaps(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    score_fn: Callable[[torch.Tensor], float],
    max_candidates: int = 32,
) -> torch.Tensor:
    """Bounded adjacent-x-swap search with composite validation.

    Sprint-22 area D found that ``dependency_500`` saturates the gradient
    pipeline with edge_length_cv as the dominant residual term (0.91 at
    baseline, vs ELK 0.43). The fix is a small discrete permutation of
    same-layer x order: take the longest 10% of edges, look at adjacent
    same-layer node pairs that touch a long-edge endpoint, rank by cheap
    edge-CV delta, then validate the top candidates with full composite.

    The search uses ``longest_path_layering`` for layers (matching the
    metric's depth function) and only commits a swap when ``score_fn``
    confirms the trial improves. Runs with ``_should_gap_swap_large_dag``
    as the precondition; small graphs and low-CV graphs are skipped.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``. Unused but kept for the
        polish-candidate signature.
    score_fn : Callable[[torch.Tensor], float]
        Composite scoring function for trial acceptance.
    max_candidates : int, default=32
        Maximum number of CV-prefiltered swaps to validate.

    Returns
    -------
    torch.Tensor
        Position tensor after accepted swaps.
    """
    del node_sizes
    cand = pos.detach().clone()
    if not _should_gap_swap_large_dag(cand, edge_index):
        return cand
    n = int(cand.shape[0])
    try:
        from dagua.utils import longest_path_layering

        raw_depth = longest_path_layering(edge_index, n)
    except Exception:
        return cand
    layers = (
        raw_depth.to(torch.long)
        if isinstance(raw_depth, torch.Tensor)
        else torch.as_tensor(raw_depth, dtype=torch.long)
    )

    src = edge_index[0]
    tgt = edge_index[1]
    diffs = cand[tgt] - cand[src]
    lengths = diffs.pow(2).sum(-1).sqrt()
    if lengths.numel() == 0:
        return cand
    threshold = float(torch.quantile(lengths, 0.90).item())
    long_mask = lengths >= threshold
    long_endpoints = torch.zeros(n, dtype=torch.bool, device=cand.device)
    long_endpoints[src[long_mask]] = True
    long_endpoints[tgt[long_mask]] = True

    def edge_cv(p: torch.Tensor) -> float:
        d = (p[tgt] - p[src]).pow(2).sum(-1).sqrt()
        finite = d[torch.isfinite(d)]
        if finite.numel() == 0:
            return float("inf")
        m = float(finite.mean().item())
        if m <= 1e-6:
            return float("inf")
        return float(finite.std().item()) / m

    base_cv = edge_cv(cand)

    ranked: list[tuple[float, int, int]] = []
    for layer_val in torch.unique(layers, sorted=True):
        layer_nodes = torch.nonzero(layers == layer_val, as_tuple=False).squeeze(1)
        if layer_nodes.numel() < 2:
            continue
        order = torch.argsort(cand[layer_nodes, 0], stable=True)
        ordered = layer_nodes[order]
        for k in range(int(ordered.numel()) - 1):
            left = int(ordered[k].item())
            right = int(ordered[k + 1].item())
            if not (bool(long_endpoints[left].item()) or bool(long_endpoints[right].item())):
                continue
            trial = cand.clone()
            tmp = float(trial[left, 0].item())
            trial[left, 0] = trial[right, 0]
            trial[right, 0] = tmp
            cv_delta = edge_cv(trial) - base_cv
            ranked.append((cv_delta, left, right))

    if not ranked:
        return cand
    ranked.sort(key=lambda t: t[0])

    best = cand
    try:
        best_score = score_fn(best)
    except Exception:
        return cand

    for _, left, right in ranked[:max_candidates]:
        trial = best.clone()
        tmp = float(trial[left, 0].item())
        trial[left, 0] = trial[right, 0]
        trial[right, 0] = tmp
        if not bool(torch.isfinite(trial).all().item()):
            continue
        try:
            trial_score = score_fn(trial)
        except Exception:
            continue
        if trial_score > best_score:
            best = trial
            best_score = trial_score
    return best


def _best_of_polish(
    base_pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    margin: float = 0.5,
) -> torch.Tensor:
    """Try named polish candidates; return the best by composite.

    The gradient pipeline saturates on edge-length-variance for
    layered_dag and tree pipelines, so a direct constraint projection
    can escape the local minimum. Edge-equalize variants are tried first;
    sprint-21a projection primitives are then scored as named candidates.
    The un-polished baseline is preserved unless a candidate beats it by at
    least ``margin`` composite points.

    Parameters
    ----------
    base_pos : torch.Tensor
        Un-polished pipeline output with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    margin : float, default=0.5
        Minimum composite improvement to prefer a polished candidate.

    Returns
    -------
    torch.Tensor
        Best position tensor with shape ``[N, 2]``.
    """
    from dagua.metrics import composite, full

    def score(pos: torch.Tensor) -> float:
        torch.manual_seed(0)
        return float(composite(full(pos, edge_index, node_sizes=node_sizes)))

    def safe_score(pos: torch.Tensor) -> Optional[float]:
        """Return a finite composite score or ``None`` for invalid candidates.

        Parameters
        ----------
        pos : torch.Tensor
            Candidate position tensor with shape ``[N, 2]``.

        Returns
        -------
        float | None
            Composite score when scoring succeeds, otherwise ``None``.
        """
        if not bool(torch.isfinite(pos).all().item()):
            return None
        try:
            return score(pos)
        except Exception:
            return None

    best_pos = base_pos
    best_score = score(base_pos)

    edge_equalize_candidates: list[
        tuple[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]
    ] = [
        (
            f"edge_equalize_{iters}_{step:g}",
            lambda pos, edges, sizes, iters=iters, step=step: _equalize_edges(
                pos,
                edges,
                iters,
                step,
            ),
        )
        for iters, step in _POLISH_SETTINGS
    ]

    best_edge_pos = base_pos
    best_edge_score = best_score
    edge_seed_positions: list[tuple[str, torch.Tensor]] = []
    for edge_name, make_candidate in edge_equalize_candidates:
        cand = make_candidate(base_pos, edge_index, node_sizes)
        cand_score = safe_score(cand)
        if cand_score is None:
            continue
        edge_seed_positions.append((edge_name, cand))
        if cand_score > best_edge_score:
            best_edge_score = cand_score
            best_edge_pos = cand
        if cand_score > best_score + margin:
            best_score = cand_score
            best_pos = cand

    polish_candidates: list[
        tuple[str, Callable[[torch.Tensor, torch.Tensor, torch.Tensor], torch.Tensor]]
    ] = [
        (
            "y_layer_snap",
            lambda pos, edges, sizes: _y_layer_snap(best_edge_pos, edges, sizes),
        ),
        (
            "orthogonal_align",
            lambda pos, edges, sizes: _orthogonal_align(best_edge_pos, edges, sizes),
        ),
        (
            "overlap_jitter",
            lambda pos, edges, sizes: _overlap_jitter(best_edge_pos, edges, sizes),
        ),
        (
            "swap_2opt_anti_crossing",
            lambda pos, edges, sizes: _swap_2opt_anti_crossing(
                pos,
                edges,
                sizes,
                score_fn=score,
            ),
        ),
        (
            "per_layer_x_kmeans",
            lambda pos, edges, sizes: _per_layer_x_kmeans(pos, edges, sizes),
        ),
        (
            "global_depth_align",
            lambda pos, edges, sizes: _global_depth_align(
                base_pos,
                edges,
                sizes,
            ),
        ),
        (
            "dot_lattice_lp",
            lambda pos, edges, sizes: _dot_lattice_lp(
                base_pos,
                edges,
                sizes,
            ),
        ),
        (
            "back_edge_relayer_full",
            lambda pos, edges, sizes: _back_edge_relayer(
                base_pos,
                edges,
                sizes,
                blend=1.0,
            ),
        ),
        (
            "back_edge_relayer_quarter",
            lambda pos, edges, sizes: _back_edge_relayer(
                base_pos,
                edges,
                sizes,
                blend=0.25,
            ),
        ),
        (
            "back_edge_relayer_half",
            lambda pos, edges, sizes: _back_edge_relayer(
                base_pos,
                edges,
                sizes,
                blend=0.5,
            ),
        ),
        (
            "tutte_cyclic_planar",
            lambda pos, edges, sizes: _tutte_cyclic_planar(
                base_pos,
                edges,
                sizes,
            ),
        ),
        (
            "gap_validated_layer_swaps",
            lambda pos, edges, sizes: _gap_validated_layer_swaps(
                base_pos,
                edges,
                sizes,
                score_fn=score,
                max_candidates=32,
            ),
        ),
    ]
    for edge_name, seed_pos in edge_seed_positions:
        polish_candidates.extend(
            [
                (
                    f"y_layer_snap_after_{edge_name}",
                    lambda pos, edges, sizes, seed_pos=seed_pos: _y_layer_snap(
                        seed_pos,
                        edges,
                        sizes,
                    ),
                ),
                (
                    f"orthogonal_align_after_{edge_name}",
                    lambda pos, edges, sizes, seed_pos=seed_pos: _orthogonal_align(
                        seed_pos,
                        edges,
                        sizes,
                    ),
                ),
                (
                    f"orthogonal_align_overlap_jitter_after_{edge_name}",
                    lambda pos, edges, sizes, seed_pos=seed_pos: _overlap_jitter(
                        _orthogonal_align(seed_pos, edges, sizes),
                        edges,
                        sizes,
                    ),
                ),
            ]
        )
    for _, make_candidate in polish_candidates:
        cand = make_candidate(best_pos, edge_index, node_sizes)
        cand_score = safe_score(cand)
        if cand_score is None:
            continue
        if cand_score > best_score + margin:
            best_score = cand_score
            best_pos = cand
    return best_pos


def _score_native_result(
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
) -> float:
    """Return the composite metric score for one native layout candidate.

    Parameters
    ----------
    pos : torch.Tensor
        Candidate positions with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    float
        Higher-is-better composite score.
    """
    return dagua_native_legacy._score_native_result(pos, edge_index, node_sizes)


def layout_dagua_native_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: torch.Tensor,
    config: Optional[LayoutConfig] = None,
    device: Optional[str] = None,
    optimizer_type: str = "adam",
    init_pos: Optional[torch.Tensor] = None,
    clusters: Optional[dict[str, Any]] = None,
    cluster_parents: Optional[dict[str, str]] = None,
    layer_assignments: Optional[torch.Tensor] = None,
    prebuilt_layer_index: Optional[Any] = None,
    graph_structure: Optional[GraphStructure] = None,
    skip_classification: bool = False,
    seed: Optional[int] = None,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Run the topology-dispatched native pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    config : LayoutConfig, optional
        Layout configuration.
    device : str, optional
        Target execution device.
    optimizer_type : str, default="adam"
        Optimizer implementation for gradient sub-pipelines.
    init_pos : torch.Tensor, optional
        Optional initial positions with shape ``[N, 2]``.
    clusters : dict[str, Any], optional
        Cluster membership metadata.
    cluster_parents : dict[str, str], optional
        Nested-cluster parent metadata.
    layer_assignments : torch.Tensor, optional
        Optional layer assignments with shape ``[N]``.
    prebuilt_layer_index : Any, optional
        Optional pre-built layer index.
    graph_structure : GraphStructure, optional
        Optional pre-classified topology.
    skip_classification : bool, default=False
        Whether to skip classification during config preparation.
    seed : int, optional
        RNG seed override.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Detached position tensor with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")

    effective_config = copy.copy(config) if config is not None else LayoutConfig()
    if _selected_force_pipeline(effective_config) == "legacy_monolith":
        return dagua_native_legacy.layout_dagua_native_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            config=effective_config,
            device=device,
            optimizer_type=optimizer_type,
            init_pos=init_pos,
            clusters=clusters,
            cluster_parents=cluster_parents,
            layer_assignments=layer_assignments,
            prebuilt_layer_index=prebuilt_layer_index,
            graph_structure=graph_structure,
            skip_classification=skip_classification,
            seed=seed,
            edge_weights=edge_weights,
        )

    # Sprint-20i: stress route for degenerate-layering cyclic graphs. Ported
    # from the legacy monolith (sprint-20d) which was lost in the s20e
    # topology-dispatch refactor. Small-world / dense-cyclic graphs with a
    # ring or near-ring structure produce a fully degenerate post-FAS
    # layering (n_relayered == num_nodes, max layer count == 1) that the
    # layered_dag pipeline can't escape because every gradient-descent step
    # respects the chain init. Stress-SGD on the same input gives the
    # 2D embedding that scoring rewards (small_world_100 48.58 -> 57.18,
    # closing the -8.51 gap to igraph_sugiyama).
    if (
        _selected_force_pipeline(effective_config) is None
        and getattr(effective_config, "route_flat_to_stress", True)
        and getattr(effective_config, "algorithm", None) in (None, "dagua_native")
        and num_nodes >= 20
        and edge_index is not None
        and edge_index.numel() > 0
    ):
        try:
            from dagua.layout.cycle import detect_back_edges, make_acyclic_robust
            from dagua.utils import longest_path_layering

            if bool(detect_back_edges(edge_index, num_nodes).any().item()):
                self_loop_mask = edge_index[0] != edge_index[1]
                filtered = edge_index[:, self_loop_mask]
                if filtered.shape[1] > 0:
                    acyclic_edges, _ = make_acyclic_robust(filtered, num_nodes)
                    layers = longest_path_layering(acyclic_edges, num_nodes)
                    layer_seq = layers if isinstance(layers, list) else layers.tolist()
                    unique = set(layer_seq)
                    if len(unique) == num_nodes and max(layer_seq.count(v) for v in unique) == 1:
                        from dagua.layout.ops.pipelines.stress_sgd import (
                            layout_stress_sgd_pipeline,
                        )

                        stress_seed = seed if seed is not None else effective_config.seed
                        if stress_seed is None:
                            stress_seed = 42
                        stress_pos = layout_stress_sgd_pipeline(
                            edge_index=edge_index,
                            num_nodes=num_nodes,
                            node_sizes=node_sizes,
                            seed=int(stress_seed),
                        )
                        if stress_pos.shape[0] > 1:
                            mean_w = (
                                float(node_sizes[:, 0].mean().item())
                                if node_sizes is not None
                                else 60.0
                            )
                            target = max(mean_w * 1.3, 1.0)
                            centered = stress_pos - stress_pos.mean(dim=0, keepdim=True)
                            diffs = centered.unsqueeze(0) - centered.unsqueeze(1)
                            dists = diffs.pow(2).sum(-1).sqrt()
                            n = centered.shape[0]
                            mask = ~torch.eye(n, dtype=torch.bool, device=dists.device)
                            if mask.any():
                                current_min = float(dists[mask].min().item())
                                if current_min > 1e-6:
                                    stress_pos = centered * (target / current_min)
                            # Sprint-22a: also polish the stress-route output.
                            # Sprint-22 area E found the back-edge relayer
                            # adds +3.3 on small_world_500 ON TOP of the
                            # stress route's 52.19 baseline (final ~55-57).
                            # The picker margin gate handles regression risk.
                            if (
                                getattr(effective_config, "edge_equalize_polish", True)
                                and node_sizes is not None
                                and stress_pos.shape[0] >= 4
                            ):
                                stress_pos = _best_of_polish(
                                    stress_pos,
                                    edge_index,
                                    node_sizes,
                                )
                            return stress_pos
        except Exception:
            # Stress route is best-effort; fall through to the layered path.
            pass

    multi_start_k = int(getattr(effective_config, "multi_start_k", 1))
    if multi_start_k > 1:
        seed_base = seed if seed is not None else effective_config.seed
        if seed_base is None:
            seed_base = 42
        best_pos: Optional[torch.Tensor] = None
        best_score = float("-inf")
        for seed_offset in range(multi_start_k):
            candidate_seed = int(seed_base) + seed_offset
            candidate_config = copy.copy(effective_config)
            candidate_config.seed = candidate_seed
            candidate_config.multi_start_k = 1
            candidate_pos = layout_dagua_native_pipeline(
                edge_index=edge_index,
                num_nodes=num_nodes,
                node_sizes=node_sizes,
                config=candidate_config,
                device=device,
                optimizer_type=optimizer_type,
                init_pos=init_pos,
                clusters=clusters,
                cluster_parents=cluster_parents,
                layer_assignments=layer_assignments,
                prebuilt_layer_index=prebuilt_layer_index,
                graph_structure=graph_structure,
                skip_classification=skip_classification,
                seed=candidate_seed,
                edge_weights=edge_weights,
            )
            candidate_score = _score_native_result(candidate_pos, edge_index, node_sizes)
            if candidate_score > best_score:
                best_score = candidate_score
                best_pos = candidate_pos
        if best_pos is None:
            raise RuntimeError("dagua_native multi-start did not produce candidate positions.")
        return best_pos

    requested_device = device or effective_config.device
    if requested_device == "cuda" and not torch.cuda.is_available():
        requested_device = "cpu"
    target_device = torch.device(requested_device)
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=torch.float32, device=target_device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=target_device)

    normalized_node_sizes = normalize_node_sizes(node_sizes=node_sizes, device=target_device)
    prepared_edge_index = edge_index.to(device=target_device, dtype=torch.long)
    prepared_init_pos = (
        init_pos.to(device=target_device, dtype=torch.float32) if init_pos is not None else None
    )
    prepared_edge_weights = (
        edge_weights.to(device=target_device, dtype=torch.float32)
        if edge_weights is not None
        else None
    )
    prepared_layer_assignments = (
        layer_assignments.to(device=target_device, dtype=torch.long)
        if layer_assignments is not None
        else None
    )
    resolved_seed = seed if seed is not None else effective_config.seed
    if resolved_seed is not None:
        torch.manual_seed(int(resolved_seed))
        if target_device.type == "cuda":
            torch.cuda.manual_seed(int(resolved_seed))

    prepared_config = _prepare_native_config(
        config=effective_config,
        num_nodes=num_nodes,
        edge_index=prepared_edge_index,
        device=str(target_device),
        optimizer_type=optimizer_type,
        layer_assignments=prepared_layer_assignments,
        prebuilt_layer_index=prebuilt_layer_index,
        graph_structure=graph_structure,
        skip_classification=skip_classification,
    )
    flex_constraints = build_flex_constraints(
        config=prepared_config,
        num_nodes=num_nodes,
        device=target_device,
    )
    problem = LayoutProblem(
        edge_index=prepared_edge_index,
        num_nodes=num_nodes,
        node_sizes=normalized_node_sizes,
        direction=prepared_config.direction,
        clusters=clusters,
        cluster_parents=cluster_parents,
        structure=getattr(prepared_config, "_dagua_native_structure", None),
        flex=flex_constraints,
        edge_weights=prepared_edge_weights,
        seed=int(resolved_seed if resolved_seed is not None else 42),
    )
    state = SolveState(pos=prepared_init_pos)
    ctx = RuntimeContext(
        plan=ExecutionPlan(
            device=str(target_device),
            optimizer_type=optimizer_type,
        ),
    )
    component_ids: Optional[torch.Tensor] = None
    if (
        getattr(prepared_config, "decompose_components", True)
        and num_nodes >= 2
        and not problem.clusters
        and not _has_pins(problem.flex)
    ):
        component_state = DetectComponents().apply(problem, SolveState(), ctx)
        component_ids = component_state.component_ids

    if _should_decompose_native_components(problem, prepared_config, component_ids):
        component_results: list[tuple[torch.Tensor, torch.Tensor]] = []
        parent_layers = getattr(prepared_config, "_dagua_native_layer_assignments", None)
        assert component_ids is not None
        for component_id in torch.unique(component_ids, sorted=True).tolist():
            component_nodes = torch.nonzero(
                component_ids == component_id,
                as_tuple=False,
            ).squeeze(1)
            child_problem, child_state, parent_indices, child_layers = _extract_component_problem(
                problem,
                state,
                component_nodes,
                layer_assignments=parent_layers,
            )
            if child_problem.num_nodes <= 1:
                child_pos = torch.zeros(
                    (child_problem.num_nodes, 2),
                    dtype=torch.float32,
                    device=target_device,
                )
            else:
                child_config = _prepare_native_config(
                    config=effective_config,
                    num_nodes=child_problem.num_nodes,
                    edge_index=child_problem.edge_index,
                    device=str(target_device),
                    optimizer_type=optimizer_type,
                    layer_assignments=child_layers,
                    prebuilt_layer_index=None,
                    graph_structure=child_problem.structure,
                    skip_classification=False,
                )
                # Sprint-19d component packing is a protected win for cyclic
                # / general-family children. Sprint-21b: allow tree- and
                # chain-shaped children to re-classify into the dedicated
                # native_tree fast-path instead of forcing every child
                # through legacy_monolith. The original blanket override
                # cost +3.26 on disconnected_label_cycle_collage and small
                # wins on org_chart_deep, random_dag_50, kitchen_sink_hybrid_net
                # by preventing simple-component re-classification.
                child_structure = (
                    getattr(child_config, "_dagua_native_structure", None)
                    or child_problem.structure
                )
                child_is_simple = child_structure is not None and child_structure.family in {
                    GraphFamily.TREE,
                    GraphFamily.CHAIN,
                }
                if _selected_force_pipeline(child_config) is None and not child_is_simple:
                    child_config.force_pipeline = "legacy_monolith"
                child_pos = _run_native_problem(child_problem, child_state, ctx, child_config)
            component_results.append((parent_indices, child_pos))

        tiled_positions = _tile_component_positions(
            component_results,
            node_sep=float(
                getattr(prepared_config, "_dagua_native_node_sep", prepared_config.node_sep)
            ),
        )
        outer_state = AspectRatioFit(AspectRatioFitConfig()).apply(
            problem,
            SolveState(pos=tiled_positions),
            ctx,
        )
        if outer_state.pos is None:
            raise RuntimeError("dagua_native component tiling did not produce positions.")
        result = outer_state.pos.detach()
        # Sprint-20l: also polish the per-component-tiled output. Closes
        # +2.96 on disconnected_label_cycle_collage (the (50, 0.05)
        # variant lifts depth_spearman by repacking nodes around the
        # tile centers).
        if (
            getattr(effective_config, "edge_equalize_polish", True)
            and _selected_force_pipeline(effective_config) is None
            and result.shape[0] >= 4
            and edge_index.numel() > 0
            and node_sizes is not None
        ):
            result = _best_of_polish(result, edge_index, node_sizes)
        return result

    return _run_native_problem(problem, state, ctx, prepared_config)


__all__ = [
    "_choose_native_pipeline",
    "_prepare_native_config",
    "_run_native_problem",
    "_should_apply_brandes_koepf_refine",
    "_should_use_native_dummy_nodes",
    "_should_use_native_median_transpose",
    "build_dagua_pipeline",
    "build_gradient_core",
    "layout_dagua_native_pipeline",
]
