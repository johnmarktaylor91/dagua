"""Differentiable edge optimization — gradient descent on bezier control points.

Optimizes edge routing by minimizing a weighted combination of:
1. Edge-edge crossings (sigmoid-relaxed proxy)
2. Edge-node crossings (proximity penalty to node bboxes)
3. Port angular resolution (penalize small angles between incident edges)
4. Curvature consistency (variance of κ across edges)
5. Curvature penalty (prefer straighter paths)

Data representation:
- Control points: [E, 2, 2] tensor (cp1 and cp2 per edge, requires_grad)
- Endpoints: [E, 2, 2] tensor (p0 and p1, fixed from port positions)
- Bezier evaluation: [E, T, 2] for T sample points per curve
"""

from __future__ import annotations

import math
from typing import List, Optional, Sequence, Tuple

import torch

from dagua.edges import BezierCurve

_TAXI_STEP_MIN = 0.1
_TAXI_STEP_MAX = 0.9
_TAXI_STEP_DEFAULT = 0.35


def _build_cluster_data_for_edge_opt(graph, positions, node_sizes):
    """Pre-compute cluster bboxes and node membership for edge optimization.

    Returns:
        cluster_bboxes: [C, 4] tensor (x_min, y_min, x_max, y_max) per cluster
        node_cluster_mask: [N, C] bool tensor — True if node i is in cluster c
        None, None if no clusters
    """
    if graph is None or not hasattr(graph, "clusters") or not graph.clusters:
        return None, None

    from dagua.utils import collect_cluster_leaves

    pos = positions.detach().cpu()
    sizes = node_sizes.detach().cpu()
    N = pos.shape[0]
    cluster_names = sorted(graph.clusters.keys())
    C = len(cluster_names)
    if C == 0:
        return None, None

    bboxes = torch.zeros(C, 4)  # [C, 4]
    node_mask = torch.zeros(N, C, dtype=torch.bool)

    for ci, cname in enumerate(cluster_names):
        members = graph.clusters[cname]
        if isinstance(members, dict):
            members = collect_cluster_leaves(members)
        if not members:
            continue
        valid = [m for m in members if 0 <= m < N]
        if not valid:
            continue

        idx = torch.tensor(valid, dtype=torch.long)
        cstyle = graph.get_style_for_cluster(cname)
        pad = cstyle.padding

        member_pos = pos[idx]
        member_sizes = sizes[idx]
        half_sizes = member_sizes / 2

        bboxes[ci, 0] = (member_pos[:, 0] - half_sizes[:, 0]).min() - pad
        bboxes[ci, 1] = (member_pos[:, 1] - half_sizes[:, 1]).min() - pad
        bboxes[ci, 2] = (member_pos[:, 0] + half_sizes[:, 0]).max() + pad
        bboxes[ci, 3] = (member_pos[:, 1] + half_sizes[:, 1]).max() + pad

        node_mask[idx, ci] = True

    return bboxes, node_mask


_HIGH_QUALITY_THRESHOLD = 0.75


class _ForcedQualityEdgeConfig:
    """Config wrapper that forces the full loss-term weights for the r80-S7#3
    high-quality edge-refinement pass.

    ``LayoutConfig``'s own ``w_edge_*`` defaults are tuned conservatively for
    the always-on Sprint 6 adaptive-skip path (``w_edge_angular_res=0.0``,
    ``w_edge_curvature_consistency=0.0`` -- disabled). The orphaned
    :class:`~dagua.layout.ops.edge_route.BezierControlPointOpt` op ships a
    different, fuller default set via ``BezierControlPointOptConfig``. This
    wrapper reproduces that fuller set (crossing/node-crossing/angular-
    resolution/curvature-consistency/curvature-penalty/cluster-crossing) for
    the forced high/max-quality path only, without touching the global
    defaults every other draw() call still uses.
    """

    _OVERRIDES = {
        "w_edge_crossing": 5.0,
        "w_edge_node_crossing": 10.0,
        "w_edge_angular_res": 2.0,
        "w_edge_curvature_consistency": 1.0,
        "w_edge_curvature_penalty": 0.5,
        "w_edge_cluster_crossing": 8.0,
    }

    def __init__(self, base: object) -> None:
        self._base = base

    def __getattr__(self, name: str) -> object:
        try:
            return self._OVERRIDES[name]
        except KeyError:
            return getattr(self._base, name)


def maybe_refine_routes(
    curves: List[BezierCurve],
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    config: object,
    graph: Optional[object] = None,
    trace: Optional[object] = None,
) -> List[BezierCurve]:
    """Quality-gated differentiable route refinement (r80-S7#3).

    Wires the orphaned :class:`BezierControlPointOpt` optimizer's underlying
    mechanism (:func:`optimize_edges`, the exact function that op delegates
    to) as an opt-in high-quality refinement pass over heuristic routes:
    OFF at draft/balanced quality (< 0.75, the existing Sprint 6 adaptive-
    skip behavior is preserved unchanged), forced ON at high/max quality
    (>= 0.75), bypassing the adaptive-skip heuristic since the caller
    explicitly asked for maximum quality. Positions are a read-only input
    throughout -- this function never mutates ``positions``.

    Parameters
    ----------
    curves : list[BezierCurve]
        Heuristic routes from :func:`dagua.edges.route_edges` (already
        including node-avoidance and port-spread deflection).
    positions : torch.Tensor
        Node centers with shape ``[N, 2]``. Read-only.
    edge_index : torch.Tensor
        Edge pairs with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node widths and heights with shape ``[N, 2]``.
    config : Any
        LayoutConfig-like object with ``edge_routing``, ``edge_opt_steps``,
        ``quality``, and (optionally) ``w_edge_*`` fields.
    graph : object or None, default=None
        Optional DaguaGraph used for cluster-aware losses.
    trace : Any, default=None
        Optional trace recorder forwarded to :func:`optimize_edges`.

    Returns
    -------
    list[BezierCurve]
        Refined curves, or ``curves`` unchanged when the pass is skipped.
    """
    if not curves:
        return curves
    if getattr(config, "edge_routing", "differentiable") != "differentiable":
        return curves
    if getattr(config, "edge_opt_steps", 0) < 0:
        return curves

    quality = float(getattr(config, "quality", 0.5))
    force_high_quality = quality >= _HIGH_QUALITY_THRESHOLD

    if force_high_quality:
        forced_config = _ForcedQualityEdgeConfig(config)
        return optimize_edges(
            curves, positions, edge_index, node_sizes, forced_config, graph, trace
        )

    # draft/balanced: unchanged Sprint 6 adaptive-skip behavior -- only run
    # the optimizer when the heuristic router still leaves meaningful
    # crossings, or for rectilinear routing modes that need CP refinement
    # regardless of crossing count.
    from dagua.metrics import edge_node_crossing_count

    heur_crossings = edge_node_crossing_count(curves, positions, node_sizes, edge_index)[
        "edge_node_crossings"
    ]
    threshold = getattr(config, "edge_routing_auto_skip_threshold", 5)
    has_rectilinear_routes = any(_curve_routing(curve) in ("ortho", "taxi") for curve in curves)
    if heur_crossings < threshold and not has_rectilinear_routes:
        return curves

    return optimize_edges(curves, positions, edge_index, node_sizes, config, graph, trace)


def optimize_edges(
    curves: List[BezierCurve],
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    config: object,
    graph: Optional[object] = None,
    trace: Optional[object] = None,
) -> List[BezierCurve]:
    """Optimize routed edge geometry via gradient descent.

    Parameters
    ----------
    curves : list[BezierCurve]
        Initial routed curves from heuristic routing.
    positions : torch.Tensor
        Node centers with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge pairs with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node widths and heights with shape ``[N, 2]``.
    config : Any
        LayoutConfig-like object with edge optimization fields.
    graph : object or None, default=None
        Optional DaguaGraph used for cluster-aware losses.
    trace : Any, default=None
        Optional trace recorder.

    Returns
    -------
    list[BezierCurve]
        Optimized curves. Straight routes are returned unchanged because
        their endpoints are the only geometry available to optimize.
    """
    if not curves:
        return curves

    routing_modes = [_curve_routing(curve) for curve in curves]
    if all(routing == "straight" for routing in routing_modes):
        return curves
    if all(routing == "bezier" for routing in routing_modes):
        return _optimize_bezier_edges(
            curves, positions, edge_index, node_sizes, config, graph, trace
        )
    return _optimize_mixed_edges(curves, positions, edge_index, node_sizes, config, graph, trace)


def _optimize_bezier_edges(
    curves: List[BezierCurve],
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    config: object,
    graph: Optional[object] = None,
    trace: Optional[object] = None,
) -> List[BezierCurve]:
    """Optimize cubic bezier control points via the legacy Sprint 6 path.

    Parameters
    ----------
    curves : list[BezierCurve]
        Initial cubic curves from heuristic routing.
    positions : torch.Tensor
        Node centers with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge pairs with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node widths and heights with shape ``[N, 2]``.
    config : Any
        LayoutConfig-like object with edge optimization fields.
    graph : object or None, default=None
        Optional DaguaGraph used for cluster-aware losses.
    trace : Any, default=None
        Optional trace recorder.

    Returns
    -------
    list[BezierCurve]
        Optimized cubic curves.
    """
    E = len(curves)
    if E == 0:
        return curves

    steps = getattr(config, "edge_opt_steps", 0)
    if steps < 0:
        # Explicit skip (-1)
        return curves
    if steps == 0:
        # Auto-scale based on edge count
        steps = min(100, max(20, E * 2))

    lr = getattr(config, "edge_opt_lr", 0.1)

    # Extract endpoints and control points from curves
    endpoints = torch.zeros(E, 2, 2)  # [E, 2(p0/p1), 2(xy)]
    cp = torch.zeros(E, 2, 2)  # [E, 2(cp1/cp2), 2(xy)]

    for i, c in enumerate(curves):
        endpoints[i, 0, 0] = c.p0[0]
        endpoints[i, 0, 1] = c.p0[1]
        endpoints[i, 1, 0] = c.p1[0]
        endpoints[i, 1, 1] = c.p1[1]
        cp[i, 0, 0] = c.cp1[0]
        cp[i, 0, 1] = c.cp1[1]
        cp[i, 1, 0] = c.cp2[0]
        cp[i, 1, 1] = c.cp2[1]

    cp = cp.requires_grad_(True)
    pos = positions.detach().cpu().float()
    sizes = node_sizes.detach().cpu().float()

    # Loss weights from config
    w_crossing = getattr(config, "w_edge_crossing", 5.0)
    w_node_crossing = getattr(config, "w_edge_node_crossing", 10.0)
    w_angular = getattr(config, "w_edge_angular_res", 2.0)
    w_curv_consistency = getattr(config, "w_edge_curvature_consistency", 1.0)
    w_curv_penalty = getattr(config, "w_edge_curvature_penalty", 0.5)
    w_cluster_crossing = getattr(config, "w_edge_cluster_crossing", 8.0)

    # Pre-compute cluster data for edge-cluster crossing loss
    cluster_bboxes, node_cluster_mask = _build_cluster_data_for_edge_opt(
        graph, positions, node_sizes
    )

    # Pre-compute incident edge indices per node for angular resolution
    ei = edge_index.detach().cpu()
    src_list = ei[0].tolist()
    tgt_list = ei[1].tolist()

    optimizer = torch.optim.Adam([cp], lr=lr)

    # Pre-compute t samples for bezier evaluation
    T = 10
    t_samples = torch.linspace(0.0, 1.0, T).unsqueeze(0)  # [1, T]

    if trace is not None and hasattr(trace, "capture_edge_controls"):
        trace.capture_edge_controls(
            phase="heuristic_routing",
            step=0,
            total_steps=max(steps, 1),
            positions=positions.detach(),
            endpoints=endpoints,
            control_points=cp.detach(),
        )

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)

        # Evaluate bezier curves at T points: [E, T, 2]
        points = _evaluate_bezier_batch(endpoints, cp, t_samples)

        total_loss = torch.tensor(0.0)

        # 1. Edge crossing loss
        if w_crossing > 0 and E > 1:
            total_loss = total_loss + w_crossing * _edge_crossing_loss(points, E)

        # 2. Edge-node crossing loss
        if w_node_crossing > 0:
            total_loss = total_loss + w_node_crossing * _edge_node_crossing_loss(
                points, pos, sizes, src_list, tgt_list, E
            )

        # 3. Port angular resolution loss
        if w_angular > 0:
            total_loss = total_loss + w_angular * _port_angular_resolution_loss(
                endpoints, cp, src_list, tgt_list
            )

        # 4. Curvature consistency loss
        if w_curv_consistency > 0:
            total_loss = total_loss + w_curv_consistency * _curvature_consistency_loss(
                endpoints, cp, t_samples
            )

        # 5. Curvature penalty loss
        if w_curv_penalty > 0:
            total_loss = total_loss + w_curv_penalty * _curvature_penalty_loss(
                endpoints, cp, t_samples
            )

        # 6. Edge-cluster crossing loss
        if w_cluster_crossing > 0 and cluster_bboxes is not None:
            total_loss = total_loss + w_cluster_crossing * _edge_cluster_crossing_loss(
                points, cluster_bboxes, node_cluster_mask, src_list, tgt_list, E
            )

        if total_loss.requires_grad:
            total_loss.backward()
            # Replace NaN/Inf gradients with zero before stepping
            if cp.grad is not None and not torch.isfinite(cp.grad).all():
                cp.grad = torch.where(torch.isfinite(cp.grad), cp.grad, torch.zeros_like(cp.grad))
            torch.nn.utils.clip_grad_norm_([cp], max_norm=50.0)
            optimizer.step()
            # Clamp control points to stay within reasonable range of endpoints
            with torch.no_grad():
                if not torch.isfinite(cp).all():
                    # Revert NaN/Inf control points to linear interpolation
                    fallback_cp1 = endpoints[:, 0, :] * 0.667 + endpoints[:, 1, :] * 0.333
                    fallback_cp2 = endpoints[:, 0, :] * 0.333 + endpoints[:, 1, :] * 0.667
                    mask = ~torch.isfinite(cp[:, 0, :])
                    cp[:, 0, :] = torch.where(mask, fallback_cp1, cp[:, 0, :])
                    mask = ~torch.isfinite(cp[:, 1, :])
                    cp[:, 1, :] = torch.where(mask, fallback_cp2, cp[:, 1, :])

        if trace is not None and hasattr(trace, "capture_edge_controls"):
            trace.capture_edge_controls(
                phase="edge_optimization",
                step=step + 1,
                total_steps=steps,
                positions=positions.detach(),
                endpoints=endpoints,
                control_points=cp.detach(),
            )

    # Convert back to BezierCurve list (with final NaN safety check)
    cp_final = cp.detach()
    if not torch.isfinite(cp_final).all():
        # Optimization diverged — return original curves unchanged
        return curves

    result = []
    for i in range(E):
        result.append(
            BezierCurve(
                p0=(endpoints[i, 0, 0].item(), endpoints[i, 0, 1].item()),
                cp1=(cp_final[i, 0, 0].item(), cp_final[i, 0, 1].item()),
                cp2=(cp_final[i, 1, 0].item(), cp_final[i, 1, 1].item()),
                p1=(endpoints[i, 1, 0].item(), endpoints[i, 1, 1].item()),
            )
        )

    return result


def _optimize_mixed_edges(
    curves: List[BezierCurve],
    positions: torch.Tensor,
    edge_index: torch.Tensor,
    node_sizes: torch.Tensor,
    config: object,
    graph: Optional[object] = None,
    trace: Optional[object] = None,
) -> List[BezierCurve]:
    """Optimize mixed bezier, ortho, taxi, and straight routes.

    Parameters
    ----------
    curves : list[BezierCurve]
        Initial routed curves.
    positions : torch.Tensor
        Node centers with shape ``[N, 2]``.
    edge_index : torch.Tensor
        Edge pairs with shape ``[2, E]``.
    node_sizes : torch.Tensor
        Node widths and heights with shape ``[N, 2]``.
    config : Any
        LayoutConfig-like object with edge optimization fields.
    graph : object or None, default=None
        Optional DaguaGraph used for cluster-aware losses.
    trace : Any, default=None
        Optional trace recorder.

    Returns
    -------
    list[BezierCurve]
        Optimized mixed-route curves.
    """
    E = len(curves)
    if E == 0:
        return curves

    steps = getattr(config, "edge_opt_steps", 0)
    if steps < 0:
        return curves
    if steps == 0:
        steps = min(100, max(20, E * 2))

    endpoints = _curve_endpoints_tensor(curves)
    cp = _curve_control_tensor(curves).requires_grad_(True)
    ortho_indices = [
        index
        for index, curve in enumerate(curves)
        if _curve_routing(curve) == "ortho" and _ortho_lane_axis(curve) is not None
    ]
    taxi_indices = [
        index
        for index, curve in enumerate(curves)
        if _curve_routing(curve) == "taxi" and _is_non_degenerate_curve(curve)
    ]

    ortho_lanes = _initial_ortho_lanes(curves, ortho_indices)
    taxi_raw = _initial_taxi_raw(curves, taxi_indices)

    params: List[torch.Tensor] = []
    if any(_curve_routing(curve) == "bezier" for curve in curves):
        params.append(cp)
    if ortho_lanes is not None:
        params.append(ortho_lanes)
    if taxi_raw is not None:
        params.append(taxi_raw)
    if not params:
        return curves

    pos = positions.detach().cpu().float()
    sizes = node_sizes.detach().cpu().float()
    w_crossing = getattr(config, "w_edge_crossing", 5.0)
    w_node_crossing = getattr(config, "w_edge_node_crossing", 10.0)
    w_angular = getattr(config, "w_edge_angular_res", 2.0)
    w_curv_consistency = getattr(config, "w_edge_curvature_consistency", 1.0)
    w_curv_penalty = getattr(config, "w_edge_curvature_penalty", 0.5)
    w_cluster_crossing = getattr(config, "w_edge_cluster_crossing", 8.0)
    w_manhattan = getattr(config, "w_edge_manhattan_axis", 2.0)

    cluster_bboxes, node_cluster_mask = _build_cluster_data_for_edge_opt(
        graph, positions, node_sizes
    )
    ei = edge_index.detach().cpu()
    src_list = ei[0].tolist()
    tgt_list = ei[1].tolist()

    optimizer = torch.optim.Adam(params, lr=getattr(config, "edge_opt_lr", 0.1))
    t_samples = torch.linspace(0.0, 1.0, 10)

    if trace is not None and hasattr(trace, "capture_edge_controls"):
        trace.capture_edge_controls(
            phase="heuristic_routing",
            step=0,
            total_steps=max(steps, 1),
            positions=positions.detach(),
            endpoints=endpoints,
            control_points=cp.detach(),
        )

    for step in range(steps):
        optimizer.zero_grad(set_to_none=True)
        taxi_steps = _taxi_steps_from_raw(taxi_raw) if taxi_raw is not None else None
        points, tangent_controls, rect_segments, rect_segment_edges = _evaluate_mixed_routes(
            curves,
            endpoints,
            cp,
            ortho_indices,
            ortho_lanes,
            taxi_indices,
            taxi_steps,
            t_samples,
        )
        total_loss = torch.tensor(0.0)

        if w_crossing > 0 and E > 1:
            total_loss = total_loss + w_crossing * _edge_crossing_loss(points, E)
            if rect_segments.numel() > 0:
                total_loss = total_loss + w_crossing * _rectilinear_crossing_loss(
                    rect_segments, rect_segment_edges
                )
        if w_node_crossing > 0:
            total_loss = total_loss + w_node_crossing * _edge_node_crossing_loss(
                points, pos, sizes, src_list, tgt_list, E
            )
        if w_angular > 0:
            total_loss = total_loss + w_angular * _port_angular_resolution_loss(
                endpoints, tangent_controls, src_list, tgt_list
            )
        if w_curv_consistency > 0:
            total_loss = total_loss + w_curv_consistency * _sampled_curvature_consistency_loss(
                points
            )
        if w_curv_penalty > 0:
            total_loss = total_loss + w_curv_penalty * _sampled_curvature_penalty_loss(points)
        if w_cluster_crossing > 0 and cluster_bboxes is not None:
            total_loss = total_loss + w_cluster_crossing * _edge_cluster_crossing_loss(
                points, cluster_bboxes, node_cluster_mask, src_list, tgt_list, E
            )
        if w_manhattan > 0 and rect_segments.numel() > 0:
            total_loss = total_loss + w_manhattan * _manhattan_axis_penalty(rect_segments)

        if total_loss.requires_grad:
            total_loss.backward()
            for param in params:
                if param.grad is not None and not torch.isfinite(param.grad).all():
                    param.grad = torch.where(
                        torch.isfinite(param.grad), param.grad, torch.zeros_like(param.grad)
                    )
            torch.nn.utils.clip_grad_norm_(params, max_norm=50.0)
            optimizer.step()
            _clamp_rectilinear_params(curves, endpoints, ortho_indices, ortho_lanes)

        if trace is not None and hasattr(trace, "capture_edge_controls"):
            trace.capture_edge_controls(
                phase="edge_optimization",
                step=step + 1,
                total_steps=steps,
                positions=positions.detach(),
                endpoints=endpoints,
                control_points=cp.detach(),
            )

    taxi_steps = _taxi_steps_from_raw(taxi_raw) if taxi_raw is not None else None
    if not torch.isfinite(cp.detach()).all():
        return curves
    if ortho_lanes is not None and not torch.isfinite(ortho_lanes.detach()).all():
        return curves
    if taxi_steps is not None and not torch.isfinite(taxi_steps.detach()).all():
        return curves

    return _finalize_mixed_curves(
        curves,
        endpoints,
        cp.detach(),
        ortho_indices,
        None if ortho_lanes is None else ortho_lanes.detach(),
        taxi_indices,
        None if taxi_steps is None else taxi_steps.detach(),
    )


def _curve_routing(curve: BezierCurve) -> str:
    """Return the normalized routing mode for a curve.

    Parameters
    ----------
    curve : BezierCurve
        Curve whose routing metadata should be read.

    Returns
    -------
    str
        One of ``"bezier"``, ``"straight"``, ``"ortho"``, or ``"taxi"``.
    """
    routing = getattr(curve, "routing", "bezier")
    if routing in {"bezier", "straight", "ortho", "taxi"}:
        return str(routing)
    return "bezier"


def _curve_endpoints_tensor(curves: Sequence[BezierCurve]) -> torch.Tensor:
    """Convert curve endpoints to a tensor.

    Parameters
    ----------
    curves : sequence[BezierCurve]
        Curves to normalize.

    Returns
    -------
    torch.Tensor
        Endpoint tensor with shape ``[E, 2, 2]``.
    """
    endpoints = torch.zeros(len(curves), 2, 2)
    for index, curve in enumerate(curves):
        endpoints[index, 0, 0] = curve.p0[0]
        endpoints[index, 0, 1] = curve.p0[1]
        endpoints[index, 1, 0] = curve.p1[0]
        endpoints[index, 1, 1] = curve.p1[1]
    return endpoints


def _curve_control_tensor(curves: Sequence[BezierCurve]) -> torch.Tensor:
    """Convert curve control or bend anchors to a tensor.

    Parameters
    ----------
    curves : sequence[BezierCurve]
        Curves to normalize.

    Returns
    -------
    torch.Tensor
        Control tensor with shape ``[E, 2, 2]``.
    """
    controls = torch.zeros(len(curves), 2, 2)
    for index, curve in enumerate(curves):
        controls[index, 0, 0] = curve.cp1[0]
        controls[index, 0, 1] = curve.cp1[1]
        controls[index, 1, 0] = curve.cp2[0]
        controls[index, 1, 1] = curve.cp2[1]
    return controls


def _is_non_degenerate_curve(curve: BezierCurve) -> bool:
    """Return whether a curve has distinct endpoints.

    Parameters
    ----------
    curve : BezierCurve
        Curve to inspect.

    Returns
    -------
    bool
        ``True`` when source and target endpoints differ.
    """
    return not (
        math.isclose(curve.p0[0], curve.p1[0], abs_tol=1e-9)
        and math.isclose(curve.p0[1], curve.p1[1], abs_tol=1e-9)
    )


def _route_is_vertical_flow(curve: BezierCurve) -> bool:
    """Infer whether a rectilinear route uses vertical-first flow.

    Parameters
    ----------
    curve : BezierCurve
        Rectilinear curve to inspect.

    Returns
    -------
    bool
        ``True`` for ``TB``/``BT`` style routes.
    """
    direction = getattr(curve, "direction", "TB")
    if direction in ("TB", "BT"):
        return True
    if direction in ("LR", "RL"):
        return False
    if curve.waypoints is not None and len(curve.waypoints) >= 2:
        first = curve.waypoints[0]
        second = curve.waypoints[1]
        return math.isclose(first[0], second[0], abs_tol=1e-9)
    return True


def _ortho_lane_axis(curve: BezierCurve) -> Optional[int]:
    """Return the optimized lane axis for an orthogonal route.

    Parameters
    ----------
    curve : BezierCurve
        Orthogonal route.

    Returns
    -------
    int or None
        ``1`` for a shared y-lane, ``0`` for a shared x-lane, or ``None``
        when the route has no discrete lane to optimize.
    """
    if curve.waypoints is None or len(curve.waypoints) < 4:
        return None
    return 1 if _route_is_vertical_flow(curve) else 0


def _initial_ortho_lanes(
    curves: Sequence[BezierCurve], ortho_indices: Sequence[int]
) -> Optional[torch.Tensor]:
    """Build the learnable initial ortho lane tensor.

    Parameters
    ----------
    curves : sequence[BezierCurve]
        All curves.
    ortho_indices : sequence[int]
        Indices of orthogonal routes with an optimizable lane.

    Returns
    -------
    torch.Tensor or None
        Lane tensor with shape ``[O]`` and ``requires_grad=True``.
    """
    if not ortho_indices:
        return None
    values = []
    for curve_index in ortho_indices:
        curve = curves[curve_index]
        axis = _ortho_lane_axis(curve)
        if axis is None or curve.waypoints is None:
            values.append(0.0)
        else:
            # A tiny deterministic nudge breaks perfectly symmetric soft
            # crossing gradients without changing the route's visible topology.
            symmetry_nudge = 1e-3 if len(values) % 2 == 0 else -1e-3
            values.append(float(curve.waypoints[1][axis]) + symmetry_nudge)
    return torch.tensor(values, dtype=torch.float32, requires_grad=True)


def _initial_taxi_raw(
    curves: Sequence[BezierCurve], taxi_indices: Sequence[int]
) -> Optional[torch.Tensor]:
    """Build unconstrained taxi step parameters.

    Parameters
    ----------
    curves : sequence[BezierCurve]
        All curves.
    taxi_indices : sequence[int]
        Indices of taxi routes.

    Returns
    -------
    torch.Tensor or None
        Raw tensor with shape ``[T]`` and ``requires_grad=True``.
    """
    if not taxi_indices:
        return None
    raw_values = []
    for curve_index in taxi_indices:
        step = getattr(curves[curve_index], "step_fraction", _TAXI_STEP_DEFAULT)
        if isinstance(step, torch.Tensor):
            step_value = float(step.detach().cpu().item())
        elif step is None:
            step_value = _TAXI_STEP_DEFAULT
        else:
            step_value = float(step)
        normalized = (min(max(step_value, _TAXI_STEP_MIN), _TAXI_STEP_MAX) - _TAXI_STEP_MIN) / (
            _TAXI_STEP_MAX - _TAXI_STEP_MIN
        )
        normalized = min(max(normalized, 1e-4), 1.0 - 1e-4)
        raw_values.append(math.log(normalized / (1.0 - normalized)))
    return torch.tensor(raw_values, dtype=torch.float32, requires_grad=True)


def _taxi_steps_from_raw(raw: torch.Tensor) -> torch.Tensor:
    """Map unconstrained taxi parameters into the valid step-fraction range.

    Parameters
    ----------
    raw : torch.Tensor
        Raw learnable parameters with shape ``[T]``.

    Returns
    -------
    torch.Tensor
        Step fractions with shape ``[T]`` in ``[0.1, 0.9]``.
    """
    return _TAXI_STEP_MIN + (_TAXI_STEP_MAX - _TAXI_STEP_MIN) * torch.sigmoid(raw)


def _evaluate_mixed_routes(
    curves: Sequence[BezierCurve],
    endpoints: torch.Tensor,
    cp: torch.Tensor,
    ortho_indices: Sequence[int],
    ortho_lanes: Optional[torch.Tensor],
    taxi_indices: Sequence[int],
    taxi_steps: Optional[torch.Tensor],
    t_samples: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Evaluate mixed routes into sampled points and rectilinear segments.

    Parameters
    ----------
    curves : sequence[BezierCurve]
        Curves to evaluate.
    endpoints : torch.Tensor
        Endpoint tensor with shape ``[E, 2, 2]``.
    cp : torch.Tensor
        Bezier control tensor with shape ``[E, 2, 2]``.
    ortho_indices : sequence[int]
        Orthogonal route indices.
    ortho_lanes : torch.Tensor or None
        Learnable ortho lane tensor with shape ``[O]``.
    taxi_indices : sequence[int]
        Taxi route indices.
    taxi_steps : torch.Tensor or None
        Taxi step fractions with shape ``[T]``.
    t_samples : torch.Tensor
        Sample positions with shape ``[S]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]
        Sampled points ``[E, S, 2]``, tangent controls ``[E, 2, 2]``, and
        rectilinear segments ``[R, 2, 2]`` with owning edge indices ``[R]``.
    """
    ortho_lookup = {edge_index: param_index for param_index, edge_index in enumerate(ortho_indices)}
    taxi_lookup = {edge_index: param_index for param_index, edge_index in enumerate(taxi_indices)}
    bezier_points = _evaluate_bezier_batch(endpoints, cp, t_samples.unsqueeze(0))
    sampled = []
    tangent_controls = []
    rect_segments = []
    rect_segment_edges = []

    for index, curve in enumerate(curves):
        routing = _curve_routing(curve)
        if routing == "bezier":
            sampled.append(bezier_points[index])
            tangent_controls.append(cp[index])
            continue
        if routing == "straight":
            vertices = endpoints[index]
        elif routing == "ortho":
            lane = None
            if ortho_lanes is not None and index in ortho_lookup:
                lane = ortho_lanes[ortho_lookup[index]]
            vertices = _ortho_vertices(curve, endpoints[index], lane)
        elif routing == "taxi":
            step = None
            if taxi_steps is not None and index in taxi_lookup:
                step = taxi_steps[taxi_lookup[index]]
            vertices = _taxi_vertices(curve, endpoints[index], step)
        else:
            sampled.append(bezier_points[index])
            tangent_controls.append(cp[index])
            continue

        sampled.append(_evaluate_polyline_vertices(vertices, t_samples))
        tangent_controls.append(_polyline_tangent_controls(vertices))
        if routing in ("ortho", "taxi") and vertices.shape[0] > 1:
            edge_segments = torch.stack([vertices[:-1], vertices[1:]], dim=1)
            rect_segments.append(edge_segments)
            rect_segment_edges.append(
                torch.full((edge_segments.shape[0],), index, dtype=torch.long)
            )

    points = torch.stack(sampled, dim=0)
    controls = torch.stack(tangent_controls, dim=0)
    if rect_segments:
        segments = torch.cat(rect_segments, dim=0)
        segment_edges = torch.cat(rect_segment_edges, dim=0)
    else:
        segments = torch.zeros(0, 2, 2)
        segment_edges = torch.zeros(0, dtype=torch.long)
    return points, controls, segments, segment_edges


def _ortho_vertices(
    curve: BezierCurve, endpoints: torch.Tensor, lane: Optional[torch.Tensor]
) -> torch.Tensor:
    """Construct orthogonal route vertices from a lane parameter.

    Parameters
    ----------
    curve : BezierCurve
        Orthogonal curve metadata.
    endpoints : torch.Tensor
        Endpoint tensor with shape ``[2, 2]``.
    lane : torch.Tensor or None
        Shared lane coordinate. If ``None``, existing waypoint geometry is
        used as fixed topology.

    Returns
    -------
    torch.Tensor
        Vertex tensor with shape ``[P, 2]``.
    """
    p0 = endpoints[0]
    p1 = endpoints[1]
    if lane is None:
        return _fixed_waypoint_vertices(curve, endpoints)
    if _route_is_vertical_flow(curve):
        return torch.stack(
            [
                p0,
                torch.stack([p0[0], lane]),
                torch.stack([p1[0], lane]),
                p1,
            ]
        )
    return torch.stack(
        [
            p0,
            torch.stack([lane, p0[1]]),
            torch.stack([lane, p1[1]]),
            p1,
        ]
    )


def _taxi_vertices(
    curve: BezierCurve, endpoints: torch.Tensor, step: Optional[torch.Tensor]
) -> torch.Tensor:
    """Construct taxi route vertices from a step fraction.

    Parameters
    ----------
    curve : BezierCurve
        Taxi curve metadata.
    endpoints : torch.Tensor
        Endpoint tensor with shape ``[2, 2]``.
    step : torch.Tensor or None
        Step fraction in ``[0.1, 0.9]``.

    Returns
    -------
    torch.Tensor
        Vertex tensor with shape ``[6, 2]`` for non-degenerate taxi routes.
    """
    if step is None:
        return _fixed_waypoint_vertices(curve, endpoints)
    p0 = endpoints[0]
    p1 = endpoints[1]
    if _route_is_vertical_flow(curve):
        first_y = p0[1] + (p1[1] - p0[1]) * step
        second_y = p0[1] + (p1[1] - p0[1]) * (1.0 - step)
        mid_x = (p0[0] + p1[0]) / 2.0
        return torch.stack(
            [
                p0,
                torch.stack([p0[0], first_y]),
                torch.stack([mid_x, first_y]),
                torch.stack([mid_x, second_y]),
                torch.stack([p1[0], second_y]),
                p1,
            ]
        )
    first_x = p0[0] + (p1[0] - p0[0]) * step
    second_x = p0[0] + (p1[0] - p0[0]) * (1.0 - step)
    mid_y = (p0[1] + p1[1]) / 2.0
    return torch.stack(
        [
            p0,
            torch.stack([first_x, p0[1]]),
            torch.stack([first_x, mid_y]),
            torch.stack([second_x, mid_y]),
            torch.stack([second_x, p1[1]]),
            p1,
        ]
    )


def _fixed_waypoint_vertices(curve: BezierCurve, endpoints: torch.Tensor) -> torch.Tensor:
    """Return fixed waypoint vertices for a non-learnable polyline.

    Parameters
    ----------
    curve : BezierCurve
        Curve with optional waypoint metadata.
    endpoints : torch.Tensor
        Endpoint tensor with shape ``[2, 2]`` used as fallback.

    Returns
    -------
    torch.Tensor
        Vertex tensor with shape ``[P, 2]``.
    """
    if curve.waypoints is None:
        return endpoints
    return torch.tensor(curve.waypoints, dtype=torch.float32)


def _evaluate_polyline_vertices(vertices: torch.Tensor, t_samples: torch.Tensor) -> torch.Tensor:
    """Evaluate a fixed-topology polyline at normalized vertex-distance samples.

    Parameters
    ----------
    vertices : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.
    t_samples : torch.Tensor
        Sample positions with shape ``[S]``.

    Returns
    -------
    torch.Tensor
        Sampled points with shape ``[S, 2]``.
    """
    if vertices.shape[0] == 1:
        return vertices[0].unsqueeze(0).expand(t_samples.shape[0], -1)
    scaled = t_samples.clamp(0.0, 1.0) * float(vertices.shape[0] - 1)
    lower = torch.floor(scaled).long().clamp(max=vertices.shape[0] - 2)
    alpha = (scaled - lower.float()).unsqueeze(1)
    start = vertices[lower]
    stop = vertices[lower + 1]
    return start + (stop - start) * alpha


def _polyline_tangent_controls(vertices: torch.Tensor) -> torch.Tensor:
    """Return first and last bend anchors for angular-resolution loss.

    Parameters
    ----------
    vertices : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.

    Returns
    -------
    torch.Tensor
        Tangent-control tensor with shape ``[2, 2]``.
    """
    if vertices.shape[0] == 1:
        return torch.stack([vertices[0], vertices[0]])
    return torch.stack([vertices[1], vertices[-2]])


def _sampled_curvature_consistency_loss(points: torch.Tensor) -> torch.Tensor:
    """Penalize variance in discrete sampled curvature across routes.

    Parameters
    ----------
    points : torch.Tensor
        Sampled route points with shape ``[E, T, 2]``.

    Returns
    -------
    torch.Tensor
        Scalar curvature-consistency loss.
    """
    curvature = _sampled_curvature(points)
    if curvature.shape[0] < 2:
        return torch.tensor(0.0)
    return curvature.mean(dim=1).var()


def _sampled_curvature_penalty_loss(points: torch.Tensor) -> torch.Tensor:
    """Penalize total discrete sampled curvature.

    Parameters
    ----------
    points : torch.Tensor
        Sampled route points with shape ``[E, T, 2]``.

    Returns
    -------
    torch.Tensor
        Scalar curvature penalty.
    """
    curvature = _sampled_curvature(points)
    return (curvature**2).mean()


def _sampled_curvature(points: torch.Tensor) -> torch.Tensor:
    """Estimate curvature from sampled route points.

    Parameters
    ----------
    points : torch.Tensor
        Sampled route points with shape ``[E, T, 2]``.

    Returns
    -------
    torch.Tensor
        Per-edge sampled curvature with shape ``[E, T-2]``.
    """
    if points.shape[1] < 3:
        return torch.zeros(points.shape[0], 1)
    first = points[:, 1:-1, :] - points[:, :-2, :]
    second = points[:, 2:, :] - points[:, 1:-1, :]
    cross = first[:, :, 0] * second[:, :, 1] - first[:, :, 1] * second[:, :, 0]
    length = (first.norm(dim=2) + second.norm(dim=2)).clamp(min=1.0)
    return cross.abs() / (length**2)


def _manhattan_axis_penalty(segments: torch.Tensor) -> torch.Tensor:
    """Penalize segment angles that are not 0 or 90 degrees.

    Parameters
    ----------
    segments : torch.Tensor
        Segment tensor with shape ``[E*S, 2, 2]`` containing start and end
        points for each sub-segment.

    Returns
    -------
    torch.Tensor
        Scalar loss. The value is zero when all segments are axis-aligned.
    """
    dx = segments[:, 1, 0] - segments[:, 0, 0]
    dy = segments[:, 1, 1] - segments[:, 0, 1]
    length2 = dx * dx + dy * dy + 1e-9
    sin2 = (2 * dx * dy) / length2
    return (sin2**2).mean()


def _rectilinear_crossing_loss(segments: torch.Tensor, segment_edges: torch.Tensor) -> torch.Tensor:
    """Penalize soft intersections between rectilinear sub-segments.

    Parameters
    ----------
    segments : torch.Tensor
        Segment tensor with shape ``[R, 2, 2]``.
    segment_edges : torch.Tensor
        Owning edge index for each segment with shape ``[R]``.

    Returns
    -------
    torch.Tensor
        Scalar crossing proxy for horizontal-vs-vertical segment pairs.
    """
    segment_count = segments.shape[0]
    if segment_count < 2:
        return torch.tensor(0.0)
    idx = torch.triu_indices(segment_count, segment_count, offset=1)
    idx_a = idx[0]
    idx_b = idx[1]
    different_edges = segment_edges[idx_a] != segment_edges[idx_b]
    if not different_edges.any():
        return torch.tensor(0.0)
    idx_a = idx_a[different_edges]
    idx_b = idx_b[different_edges]

    seg_a = segments[idx_a]
    seg_b = segments[idx_b]
    delta_a = seg_a[:, 1, :] - seg_a[:, 0, :]
    delta_b = seg_b[:, 1, :] - seg_b[:, 0, :]
    a_horizontal = delta_a[:, 0].abs() >= delta_a[:, 1].abs()
    b_horizontal = delta_b[:, 0].abs() >= delta_b[:, 1].abs()
    hv = a_horizontal & ~b_horizontal
    vh = ~a_horizontal & b_horizontal
    if not (hv | vh).any():
        return torch.tensor(0.0)

    h_seg = torch.cat([seg_a[hv], seg_b[vh]], dim=0)
    v_seg = torch.cat([seg_b[hv], seg_a[vh]], dim=0)
    h_min_x = torch.minimum(h_seg[:, 0, 0], h_seg[:, 1, 0])
    h_max_x = torch.maximum(h_seg[:, 0, 0], h_seg[:, 1, 0])
    v_min_y = torch.minimum(v_seg[:, 0, 1], v_seg[:, 1, 1])
    v_max_y = torch.maximum(v_seg[:, 0, 1], v_seg[:, 1, 1])
    h_y = (h_seg[:, 0, 1] + h_seg[:, 1, 1]) / 2.0
    v_x = (v_seg[:, 0, 0] + v_seg[:, 1, 0]) / 2.0

    sharpness = 0.03
    inside_x = torch.sigmoid(sharpness * (v_x - h_min_x)) * torch.sigmoid(
        sharpness * (h_max_x - v_x)
    )
    inside_y = torch.sigmoid(sharpness * (h_y - v_min_y)) * torch.sigmoid(
        sharpness * (v_max_y - h_y)
    )
    return (inside_x * inside_y).mean()


def _clamp_rectilinear_params(
    curves: Sequence[BezierCurve],
    endpoints: torch.Tensor,
    ortho_indices: Sequence[int],
    ortho_lanes: Optional[torch.Tensor],
) -> None:
    """Clamp rectilinear parameters to finite, local geometry ranges.

    Parameters
    ----------
    curves : sequence[BezierCurve]
        Curves being optimized.
    endpoints : torch.Tensor
        Endpoint tensor with shape ``[E, 2, 2]``.
    ortho_indices : sequence[int]
        Indices of orthogonal routes.
    ortho_lanes : torch.Tensor or None
        Learnable ortho lane tensor.

    Returns
    -------
    None
    """
    if ortho_lanes is None:
        return
    with torch.no_grad():
        for param_index, edge_index in enumerate(ortho_indices):
            axis = _ortho_lane_axis(curves[edge_index])
            if axis is None:
                continue
            start = endpoints[edge_index, 0, axis]
            stop = endpoints[edge_index, 1, axis]
            span = (stop - start).abs().clamp(min=50.0)
            lo = torch.minimum(start, stop) - span
            hi = torch.maximum(start, stop) + span
            ortho_lanes[param_index].clamp_(float(lo.item()), float(hi.item()))


def _finalize_mixed_curves(
    curves: Sequence[BezierCurve],
    endpoints: torch.Tensor,
    cp: torch.Tensor,
    ortho_indices: Sequence[int],
    ortho_lanes: Optional[torch.Tensor],
    taxi_indices: Sequence[int],
    taxi_steps: Optional[torch.Tensor],
) -> List[BezierCurve]:
    """Convert optimized mixed-route tensors back to curve objects.

    Parameters
    ----------
    curves : sequence[BezierCurve]
        Original curves.
    endpoints : torch.Tensor
        Endpoint tensor with shape ``[E, 2, 2]``.
    cp : torch.Tensor
        Bezier control tensor with shape ``[E, 2, 2]``.
    ortho_indices : sequence[int]
        Orthogonal route indices.
    ortho_lanes : torch.Tensor or None
        Final ortho lane tensor.
    taxi_indices : sequence[int]
        Taxi route indices.
    taxi_steps : torch.Tensor or None
        Final taxi step fractions.

    Returns
    -------
    list[BezierCurve]
        Final routed curves.
    """
    ortho_lookup = {edge_index: param_index for param_index, edge_index in enumerate(ortho_indices)}
    taxi_lookup = {edge_index: param_index for param_index, edge_index in enumerate(taxi_indices)}
    result = []
    for index, curve in enumerate(curves):
        routing = _curve_routing(curve)
        if routing == "bezier":
            result.append(
                BezierCurve(
                    p0=(endpoints[index, 0, 0].item(), endpoints[index, 0, 1].item()),
                    cp1=(cp[index, 0, 0].item(), cp[index, 0, 1].item()),
                    cp2=(cp[index, 1, 0].item(), cp[index, 1, 1].item()),
                    p1=(endpoints[index, 1, 0].item(), endpoints[index, 1, 1].item()),
                    routing="bezier",
                    direction=getattr(curve, "direction", "TB"),
                )
            )
        elif routing == "straight":
            result.append(curve)
        elif routing == "ortho":
            lane = None
            if ortho_lanes is not None and index in ortho_lookup:
                lane = ortho_lanes[ortho_lookup[index]]
            vertices = _ortho_vertices(curve, endpoints[index], lane)
            result.append(
                _curve_from_vertices(vertices, "ortho", getattr(curve, "direction", "TB"), None)
            )
        elif routing == "taxi":
            step = None
            if taxi_steps is not None and index in taxi_lookup:
                step = taxi_steps[taxi_lookup[index]]
            vertices = _taxi_vertices(curve, endpoints[index], step)
            step_value = None if step is None else float(step.item())
            result.append(
                _curve_from_vertices(
                    vertices, "taxi", getattr(curve, "direction", "TB"), step_value
                )
            )
        else:
            result.append(curve)
    return result


def _curve_from_vertices(
    vertices: torch.Tensor,
    routing: str,
    direction: str,
    step_fraction: Optional[float],
) -> BezierCurve:
    """Build a BezierCurve wrapper from polyline vertices.

    Parameters
    ----------
    vertices : torch.Tensor
        Polyline vertices with shape ``[P, 2]``.
    routing : str
        Routing mode to store on the curve.
    direction : str
        Layout direction to store on the curve.
    step_fraction : float or None
        Taxi step fraction to store on the curve.

    Returns
    -------
    BezierCurve
        Routed curve with explicit waypoint geometry.
    """
    points = [(float(point[0].item()), float(point[1].item())) for point in vertices]
    deduped = []
    for point in points:
        if deduped and math.isclose(deduped[-1][0], point[0], abs_tol=1e-9):
            if math.isclose(deduped[-1][1], point[1], abs_tol=1e-9):
                continue
        deduped.append(point)
    if len(deduped) == 1:
        point = deduped[0]
        return BezierCurve(
            point,
            point,
            point,
            point,
            waypoints=(point,),
            routing=routing,
            direction=direction,
            step_fraction=step_fraction,
        )
    first_bend = deduped[1] if len(deduped) > 2 else deduped[0]
    last_bend = deduped[-2] if len(deduped) > 2 else deduped[-1]
    return BezierCurve(
        deduped[0],
        first_bend,
        last_bend,
        deduped[-1],
        waypoints=tuple(deduped),
        routing=routing,
        direction=direction,
        step_fraction=step_fraction,
    )


def _evaluate_bezier_batch(
    endpoints: torch.Tensor,
    cp: torch.Tensor,
    t_samples: torch.Tensor,
) -> torch.Tensor:
    """Evaluate cubic bezier curves at sample points.

    B(t) = (1-t)^3 P0 + 3(1-t)^2 t P1 + 3(1-t) t^2 P2 + t^3 P3

    Args:
        endpoints: [E, 2, 2] — P0 and P3
        cp: [E, 2, 2] — P1 (cp1) and P2 (cp2)
        t_samples: [1, T] — parameter values

    Returns:
        [E, T, 2] — evaluated points
    """
    t = t_samples.unsqueeze(-1)  # [1, T, 1]
    u = 1.0 - t

    p0 = endpoints[:, 0:1, :]  # [E, 1, 2]
    p3 = endpoints[:, 1:2, :]  # [E, 1, 2]
    p1 = cp[:, 0:1, :]  # [E, 1, 2]
    p2 = cp[:, 1:2, :]  # [E, 1, 2]

    points = u**3 * p0 + 3 * u**2 * t * p1 + 3 * u * t**2 * p2 + t**3 * p3
    return points  # [E, T, 2]


def _edge_crossing_loss(points: torch.Tensor, E: int) -> torch.Tensor:
    """Sigmoid-relaxed edge crossing proxy.

    Samples T points per curve, creates piecewise linear segments,
    checks pairs for crossings via soft intersection test.
    For E > 5K, sample edge pairs.
    """
    # Segments: [E, T-1, 2] start and end
    T = points.shape[1]
    seg_start = points[:, :-1, :]  # [E, T-1, 2]
    seg_end = points[:, 1:, :]  # [E, T-1, 2]

    S = T - 1  # segments per edge

    # For large E, sample pairs
    max_pairs = 5000
    if E * (E - 1) // 2 > max_pairs:
        idx1 = torch.randint(0, E, (max_pairs,))
        idx2 = torch.randint(0, E, (max_pairs,))
        valid = idx1 < idx2
        idx1 = idx1[valid]
        idx2 = idx2[valid]
    else:
        # All pairs
        idx = torch.triu_indices(E, E, offset=1)
        idx1, idx2 = idx[0], idx[1]

    if idx1.numel() == 0:
        return torch.tensor(0.0)

    # For each pair of edges, check one representative segment pair (midpoint segments)
    mid_s = S // 2
    p1 = seg_start[idx1, mid_s]  # [P, 2]
    p2 = seg_end[idx1, mid_s]  # [P, 2]
    p3 = seg_start[idx2, mid_s]  # [P, 2]
    p4 = seg_end[idx2, mid_s]  # [P, 2]

    # Soft crossing via signed area test
    d1 = p2 - p1
    d2 = p4 - p3
    cross = d1[:, 0] * d2[:, 1] - d1[:, 1] * d2[:, 0]

    d3 = p3 - p1
    # Clamp both sides to avoid division by near-zero (cross can be negative)
    safe_cross = cross.sign() * cross.abs().clamp(min=1e-6)
    t = (d3[:, 0] * d2[:, 1] - d3[:, 1] * d2[:, 0]) / safe_cross
    u = (d3[:, 0] * d1[:, 1] - d3[:, 1] * d1[:, 0]) / safe_cross

    # Sigmoid relaxation: crossing when both t,u in (0,1)
    sharpness = 10.0
    t_in = torch.sigmoid(sharpness * t) * torch.sigmoid(sharpness * (1 - t))
    u_in = torch.sigmoid(sharpness * u) * torch.sigmoid(sharpness * (1 - u))
    soft_crossing = t_in * u_in

    return soft_crossing.mean()


def _edge_node_crossing_loss(
    points: torch.Tensor,
    pos: torch.Tensor,
    sizes: torch.Tensor,
    src_list: list,
    tgt_list: list,
    E: int,
) -> torch.Tensor:
    """Penalty for edge sample points close to unrelated node bboxes.

    Uses relu(safety_margin - manhattan_dist_to_bbox)^2.
    """
    N = pos.shape[0]
    T = points.shape[1]
    safety_margin = 3.0

    # For efficiency, only check nearby nodes via spatial proximity
    # Flatten all sample points: [E*T, 2]
    flat_points = points.reshape(-1, 2)  # [E*T, 2]

    # Node bbox half-sizes
    half_w = sizes[:, 0] / 2  # [N]
    half_h = sizes[:, 1] / 2

    # For each sample point, distance to each node bbox
    # This is O(E*T*N) — use spatial hashing for large N
    if N > 500:
        # Sample subset of nodes
        sample_n = min(500, N)
        node_idx = torch.randperm(N)[:sample_n]
    else:
        node_idx = torch.arange(N)
        sample_n = N

    # Build source/target lookup for each edge's sample points
    edge_of_point = torch.arange(E).unsqueeze(1).expand(-1, T).reshape(-1)  # [E*T]

    # [E*T, sample_n]
    px = flat_points[:, 0:1]  # [E*T, 1]
    py = flat_points[:, 1:2]

    nx = pos[node_idx, 0:1].t()  # [1, sample_n]
    ny = pos[node_idx, 1:2].t()
    nhw = half_w[node_idx].unsqueeze(0)  # [1, sample_n]
    nhh = half_h[node_idx].unsqueeze(0)

    # Manhattan distance from point to bbox boundary (negative = inside)
    dx = (px - nx).abs() - nhw  # [E*T, sample_n]
    dy = (py - ny).abs() - nhh

    # Distance to bbox: max(dx, 0) + max(dy, 0) for outside;
    # for points inside, both dx and dy are negative
    dist_to_bbox = torch.relu(dx) + torch.relu(dy)

    # Mask out edges' own source/target nodes
    src_nodes = torch.tensor(src_list)  # [E]
    tgt_nodes = torch.tensor(tgt_list)  # [E]

    # Build mask: for each point, mask its edge's src and tgt
    edge_src = src_nodes[edge_of_point]  # [E*T]
    edge_tgt = tgt_nodes[edge_of_point]

    # Check if each node in sample is the src/tgt of each point's edge
    is_own_src = edge_src.unsqueeze(1) == node_idx.unsqueeze(0)  # [E*T, sample_n]
    is_own_tgt = edge_tgt.unsqueeze(1) == node_idx.unsqueeze(0)
    is_own = is_own_src | is_own_tgt

    # Penalty: relu(safety - dist)^2, zeroed for own nodes
    penalty = torch.relu(safety_margin - dist_to_bbox) ** 2
    penalty = penalty * (~is_own).float()

    return penalty.mean()


def _port_angular_resolution_loss(
    endpoints: torch.Tensor,
    cp: torch.Tensor,
    src_list: list,
    tgt_list: list,
) -> torch.Tensor:
    """Penalize small angles between incident edges at each node.

    Tangent at source: B'(0) = 3(P1 - P0)
    Tangent at target: B'(1) = 3(P3 - P2)
    """
    E = endpoints.shape[0]
    if E < 2:
        return torch.tensor(0.0)

    min_angle_target = math.pi / 8  # 22.5 degrees

    # Source tangents: 3 * (cp1 - p0) → direction leaving source
    src_tangent = cp[:, 0, :] - endpoints[:, 0, :]  # [E, 2]
    # Target tangents: 3 * (p3 - cp2) → direction arriving at target
    tgt_tangent = endpoints[:, 1, :] - cp[:, 1, :]  # [E, 2]

    # Group tangents by node
    src_nodes = torch.tensor(src_list)
    tgt_nodes = torch.tensor(tgt_list)
    all_nodes = torch.cat([src_nodes, tgt_nodes])
    all_tangents = torch.cat([src_tangent, tgt_tangent], dim=0)  # [2E, 2]

    unique_nodes = all_nodes.unique()
    total_loss = torch.tensor(0.0)
    count = 0

    for node in unique_nodes:
        mask = all_nodes == node
        node_tangents = all_tangents[mask]  # [K, 2]
        K = node_tangents.shape[0]
        if K < 2:
            continue

        # Normalize tangents
        norms = node_tangents.norm(dim=1, keepdim=True).clamp(min=1e-6)
        unit = node_tangents / norms

        # Angles between all pairs
        dots = (unit @ unit.t()).clamp(-1.0, 1.0)  # [K, K]
        angles = torch.acos(dots)

        # Only upper triangle (unique pairs)
        idx = torch.triu_indices(K, K, offset=1)
        pair_angles = angles[idx[0], idx[1]]

        # Penalty: relu(target - angle)^2
        penalty = torch.relu(min_angle_target - pair_angles) ** 2
        total_loss = total_loss + penalty.sum()
        count += pair_angles.numel()

    if count == 0:
        return torch.tensor(0.0)
    return total_loss / count


def _bezier_derivatives_batch(
    endpoints: torch.Tensor,
    cp: torch.Tensor,
    t_samples: torch.Tensor,
):
    """Compute first and second derivatives of bezier curves.

    B'(t) = 3[(1-t)^2(P1-P0) + 2(1-t)t(P2-P1) + t^2(P3-P2)]
    B''(t) = 6[(1-t)(P2-2P1+P0) + t(P3-2P2+P1)]

    Returns:
        d1: [E, T, 2] — first derivative
        d2: [E, T, 2] — second derivative
    """
    t = t_samples.unsqueeze(-1)  # [1, T, 1]
    u = 1.0 - t

    p0 = endpoints[:, 0:1, :]
    p3 = endpoints[:, 1:2, :]
    p1 = cp[:, 0:1, :]
    p2 = cp[:, 1:2, :]

    # First derivative
    d1 = 3 * (u**2 * (p1 - p0) + 2 * u * t * (p2 - p1) + t**2 * (p3 - p2))

    # Second derivative
    d2 = 6 * (u * (p2 - 2 * p1 + p0) + t * (p3 - 2 * p2 + p1))

    return d1, d2


def _curvature_consistency_loss(
    endpoints: torch.Tensor,
    cp: torch.Tensor,
    t_samples: torch.Tensor,
) -> torch.Tensor:
    """Penalize variance of curvature κ across all edges.

    κ = |B' × B''| / |B'|³
    """
    d1, d2 = _bezier_derivatives_batch(endpoints, cp, t_samples)

    # Cross product in 2D: d1_x * d2_y - d1_y * d2_x
    cross = d1[:, :, 0] * d2[:, :, 1] - d1[:, :, 1] * d2[:, :, 0]  # [E, T]
    d1_norm = d1.norm(dim=2).clamp(min=1e-6)  # [E, T]

    kappa = cross.abs() / d1_norm.clamp(min=1.0) ** 3  # [E, T]  clamp avoids blowup on short edges

    # Mean curvature per edge, then variance across edges
    mean_kappa = kappa.mean(dim=1)  # [E]
    if mean_kappa.numel() < 2:
        return torch.tensor(0.0)

    return mean_kappa.var()


def _curvature_penalty_loss(
    endpoints: torch.Tensor,
    cp: torch.Tensor,
    t_samples: torch.Tensor,
) -> torch.Tensor:
    """Penalize total curvature — prefer straighter paths.

    mean(κ²) across all edges and sample points.
    """
    d1, d2 = _bezier_derivatives_batch(endpoints, cp, t_samples)

    cross = d1[:, :, 0] * d2[:, :, 1] - d1[:, :, 1] * d2[:, :, 0]
    d1_norm = d1.norm(dim=2).clamp(min=1e-6)

    kappa = cross.abs() / d1_norm.clamp(min=1.0) ** 3  # clamp avoids blowup on short edges
    return (kappa**2).mean()


def _edge_cluster_crossing_loss(
    points: torch.Tensor,
    cluster_bboxes: torch.Tensor,
    node_cluster_mask: torch.Tensor,
    src_list: list,
    tgt_list: list,
    E: int,
) -> torch.Tensor:
    """Penalize edge sample points inside foreign cluster bboxes.

    A cluster is "foreign" to an edge if neither the source nor target node
    belongs to that cluster. Uses soft proximity: sigmoid penetration depth.

    Args:
        points: [E, T, 2] sampled bezier points
        cluster_bboxes: [C, 4] (x_min, y_min, x_max, y_max) per cluster
        node_cluster_mask: [N, C] bool — True if node i is in cluster c
        src_list: source node indices per edge
        tgt_list: target node indices per edge
        E: number of edges
    """
    T = points.shape[1]

    # Build foreign mask: [E, C] — True if neither src nor tgt belongs to cluster c
    src_t = torch.tensor(src_list[:E], dtype=torch.long)
    tgt_t = torch.tensor(tgt_list[:E], dtype=torch.long)
    src_in = node_cluster_mask[src_t]  # [E, C]
    tgt_in = node_cluster_mask[tgt_t]  # [E, C]
    foreign = ~(src_in | tgt_in)  # [E, C]

    if not foreign.any():
        return torch.tensor(0.0)

    # For each sample point, compute penetration into each cluster bbox
    # points: [E, T, 2], cluster_bboxes: [C, 4]
    px = points[:, :, 0]  # [E, T]
    py = points[:, :, 1]  # [E, T]

    # Expand for broadcasting: [E, T, C]
    bx_min = cluster_bboxes[:, 0].unsqueeze(0).unsqueeze(0)  # [1, 1, C]
    by_min = cluster_bboxes[:, 1].unsqueeze(0).unsqueeze(0)
    bx_max = cluster_bboxes[:, 2].unsqueeze(0).unsqueeze(0)
    by_max = cluster_bboxes[:, 3].unsqueeze(0).unsqueeze(0)

    px_exp = px.unsqueeze(2)  # [E, T, 1]
    py_exp = py.unsqueeze(2)  # [E, T, 1]

    # Penetration depth per axis (positive = inside bbox)
    pen_x = torch.relu(px_exp - bx_min) * torch.relu(bx_max - px_exp)  # [E, T, C]
    pen_y = torch.relu(py_exp - by_min) * torch.relu(by_max - py_exp)

    # Inside bbox when both pen_x > 0 and pen_y > 0
    # Use soft product as penetration measure
    penetration = pen_x * pen_y  # [E, T, C]

    # Mask by foreign: only penalize foreign clusters
    foreign_exp = foreign.unsqueeze(1).expand(-1, T, -1)  # [E, T, C]
    masked = penetration * foreign_exp.float()

    return masked.mean()
