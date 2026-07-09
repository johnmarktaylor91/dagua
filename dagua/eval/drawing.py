"""Helpers for full-drawing evaluation of EXTERNAL engine output (r80-S6).

External adapters (graphviz, ELK) capture routed edge geometry as plain
polylines -- ``list[list[tuple[float, float]]]`` aligned to ``edge_index``
columns, with ``None`` for edges the engine did not route. This module
converts those polylines into ``dagua.edges.BezierCurve`` objects so the
drawing metrics (:func:`dagua.metrics.routed_crossing_rate`,
:func:`dagua.metrics.composite_drawing`) can consume them through the same
interface as dagua's own router output.

Kept out of ``dagua/edges.py`` on purpose: that module is render-path code
and is frozen for this stream; this one is eval-only plumbing.
"""

from __future__ import annotations

from typing import List, Optional, Sequence, Tuple

import torch

Point = Tuple[float, float]
Polyline = List[Point]


def polyline_to_curve(points: Sequence[Point], routing: str = "external"):
    """Wrap a routed polyline in a ``BezierCurve`` for metric consumption.

    Parameters
    ----------
    points : Sequence[Point]
        Polyline vertices in draw order (>= 2 points).
    routing : str, default="external"
        Routing tag recorded on the curve.

    Returns
    -------
    dagua.edges.BezierCurve
        Waypoint-backed curve whose evaluation follows the polyline exactly.
    """
    from dagua.edges import BezierCurve

    pts: Polyline = [(float(x), float(y)) for x, y in points]
    if len(pts) == 0:
        raise ValueError("polyline_to_curve requires at least one point")
    if len(pts) == 1:
        pts = [pts[0], pts[0]]
    first_bend = pts[1] if len(pts) > 2 else pts[0]
    last_bend = pts[-2] if len(pts) > 2 else pts[-1]
    return BezierCurve(
        pts[0],
        first_bend,
        last_bend,
        pts[-1],
        waypoints=tuple(pts),
        routing=routing,
    )


def routes_to_curves(
    routes: Optional[Sequence[Optional[Sequence[Point]]]],
    pos: torch.Tensor,
    edge_index: torch.Tensor,
    routing: str = "external",
) -> Optional[list]:
    """Convert captured per-edge polylines into metric-ready curves.

    Edges whose captured route is ``None`` (or too short) fall back to the
    straight node-center segment, mirroring how an engine that emits no
    geometry for an edge would be rendered.

    Parameters
    ----------
    routes : sequence of (polyline | None) | None
        Captured routes aligned to ``edge_index`` columns. ``None`` for the
        whole argument returns ``None`` (no native routing at all).
    pos : torch.Tensor
        Node positions ``[N, 2]`` used for straight fallbacks.
    edge_index : torch.Tensor
        Edge tensor ``[2, E]``.
    routing : str, default="external"
        Routing tag recorded on wrapped curves.

    Returns
    -------
    list | None
        ``BezierCurve`` list aligned to edges, or ``None`` when ``routes``
        is ``None``.
    """
    from dagua.edges import BezierCurve

    if routes is None:
        return None
    E = int(edge_index.shape[1]) if edge_index.numel() > 0 else 0
    pos_cpu = pos.detach().cpu()
    curves = []
    for e in range(E):
        route = routes[e] if e < len(routes) else None
        if route is not None and len(route) >= 2:
            curves.append(polyline_to_curve(route, routing=routing))
            continue
        s = int(edge_index[0, e].item())
        t = int(edge_index[1, e].item())
        p0 = (float(pos_cpu[s, 0].item()), float(pos_cpu[s, 1].item()))
        p1 = (float(pos_cpu[t, 0].item()), float(pos_cpu[t, 1].item()))
        curves.append(BezierCurve(p0, p0, p1, p1, routing="straight"))
    return curves


def native_route_coverage(
    routes: Optional[Sequence[Optional[Sequence[Point]]]],
    num_edges: int,
) -> float:
    """Fraction of edges with a captured native route.

    Parameters
    ----------
    routes : sequence of (polyline | None) | None
        Captured routes.
    num_edges : int
        Total edge count.

    Returns
    -------
    float
        Coverage in ``[0, 1]``; ``0.0`` when ``routes`` is ``None`` or the
        graph has no edges.
    """
    if routes is None or num_edges <= 0:
        return 0.0
    covered = sum(1 for r in routes[:num_edges] if r is not None and len(r) >= 2)
    return covered / num_edges
