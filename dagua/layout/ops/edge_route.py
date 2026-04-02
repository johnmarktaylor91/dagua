"""Edge-routing operations for post-layout route refinement."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple, Union

import torch

from dagua.edges import BezierCurve
from dagua.layout.edge_optimization import optimize_edges
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_Point = Tuple[float, float]
_RouteLike = Union[BezierCurve, torch.Tensor, Sequence[Sequence[float]]]


def _require_positions(state: SolveState, op_name: str) -> torch.Tensor:
    """Return state positions or raise a descriptive error.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    op_name : str
        Operation name for error messages.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``state.pos`` is unavailable.
    """
    if state.pos is None:
        raise ValueError(f"{op_name} requires state.pos to be set.")
    return state.pos


def _resolve_node_sizes(problem: LayoutProblem) -> torch.Tensor:
    """Return node sizes suitable for edge optimization.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.

    Returns
    -------
    torch.Tensor
        Node-size tensor with shape ``[N, 2]`` on CPU.
    """
    if problem.node_sizes is None:
        return torch.zeros((problem.num_nodes, 2), dtype=torch.float32)
    return problem.node_sizes.detach().to(device="cpu", dtype=torch.float32)


def _polyline_points(route: _RouteLike) -> List[_Point]:
    """Normalize a route-like object into explicit polyline points.

    Parameters
    ----------
    route : BezierCurve or torch.Tensor or sequence of sequence of float
        Route description to normalize.

    Returns
    -------
    list[tuple[float, float]]
        Explicit route points in draw order.

    Raises
    ------
    ValueError
        If the route cannot be interpreted as a sequence of 2D points.
    """
    if isinstance(route, BezierCurve):
        if route.waypoints is not None:
            return [(float(x), float(y)) for x, y in route.waypoints]
        return [
            (float(route.p0[0]), float(route.p0[1])),
            (float(route.p1[0]), float(route.p1[1])),
        ]

    if isinstance(route, torch.Tensor):
        if route.ndim != 2 or route.shape[1] != 2:
            raise ValueError("Tensor edge routes must have shape [P, 2].")
        return [(float(point[0]), float(point[1])) for point in route.detach().cpu().tolist()]

    points = [(float(point[0]), float(point[1])) for point in route]
    if any(len(point) != 2 for point in route):
        raise ValueError("Sequence edge routes must contain 2D points.")
    return points


def _linear_control_points(start: _Point, stop: _Point) -> Tuple[_Point, _Point]:
    """Return cubic control points that lie on the straight segment.

    Parameters
    ----------
    start : tuple[float, float]
        Route start point.
    stop : tuple[float, float]
        Route end point.

    Returns
    -------
    tuple[tuple[float, float], tuple[float, float]]
        Straight-line cubic control points at one-third and two-thirds.
    """
    cp1 = (
        float(start[0] * (2.0 / 3.0) + stop[0] * (1.0 / 3.0)),
        float(start[1] * (2.0 / 3.0) + stop[1] * (1.0 / 3.0)),
    )
    cp2 = (
        float(start[0] * (1.0 / 3.0) + stop[0] * (2.0 / 3.0)),
        float(start[1] * (1.0 / 3.0) + stop[1] * (2.0 / 3.0)),
    )
    return cp1, cp2


def _route_to_bezier(route: _RouteLike) -> BezierCurve:
    """Convert a polyline-like route into a cubic Bezier representation.

    Parameters
    ----------
    route : BezierCurve or torch.Tensor or sequence of sequence of float
        Route description to convert.

    Returns
    -------
    BezierCurve
        Cubic route used by the edge optimizer.

    Raises
    ------
    ValueError
        If the route contains fewer than two points.
    """
    if isinstance(route, BezierCurve):
        return route

    points = _polyline_points(route)
    if len(points) < 2:
        raise ValueError("Edge routes require at least two points.")
    if len(points) == 2:
        cp1, cp2 = _linear_control_points(points[0], points[1])
        return BezierCurve(p0=points[0], cp1=cp1, cp2=cp2, p1=points[1])

    return BezierCurve(
        p0=points[0],
        cp1=points[1],
        cp2=points[-2],
        p1=points[-1],
        waypoints=tuple(points),
    )


def _reconstruct_routes(
    positions: torch.Tensor,
    edge_paths: Sequence[Sequence[int]],
    reversed_edge_mask: Optional[torch.Tensor],
) -> List[torch.Tensor]:
    """Build polyline routes from expanded-node paths.

    Parameters
    ----------
    positions : torch.Tensor
        Expanded-node coordinates with shape ``[N_total, 2]``.
    edge_paths : sequence of sequence of int
        Expanded node ids visited by each original edge.
    reversed_edge_mask : torch.Tensor, optional
        Boolean mask marking edges reversed during cycle breaking.

    Returns
    -------
    list[torch.Tensor]
        Polyline routes aligned to the original edge order.
    """
    routes: List[torch.Tensor] = []
    for edge_index, node_path in enumerate(edge_paths):
        route = positions[list(node_path)]
        is_reversed = (
            reversed_edge_mask is not None
            and edge_index < reversed_edge_mask.numel()
            and bool(reversed_edge_mask[edge_index].item())
        )
        if is_reversed:
            route = torch.flip(route, dims=[0])
        routes.append(route)
    return routes


@dataclass(frozen=True)
class BezierControlPointOptConfig:
    """Configuration for :class:`BezierControlPointOpt`.

    Parameters
    ----------
    lr : float, default=0.1
        Adam learning rate used for control-point optimization.
    steps : int, default=0
        Optimization steps. ``0`` uses the reference auto-scaling rule.
    w_edge_crossing : float, default=5.0
        Edge-edge crossing loss weight.
    w_edge_node_crossing : float, default=10.0
        Edge-node crossing loss weight.
    w_edge_angular_res : float, default=2.0
        Port angular resolution loss weight.
    w_edge_curvature_consistency : float, default=1.0
        Curvature consistency loss weight.
    w_edge_curvature_penalty : float, default=0.5
        Curvature penalty loss weight.
    w_edge_cluster_crossing : float, default=8.0
        Edge-cluster crossing loss weight.
    grad_clip : float, default=50.0
        Recorded gradient-clip threshold for parity with the native engine.
        The current optimizer implementation already clips to this value.
    """

    lr: float = 0.1
    steps: int = 0
    w_edge_crossing: float = 5.0
    w_edge_node_crossing: float = 10.0
    w_edge_angular_res: float = 2.0
    w_edge_curvature_consistency: float = 1.0
    w_edge_curvature_penalty: float = 0.5
    w_edge_cluster_crossing: float = 8.0
    grad_clip: float = 50.0


@register_op
@dataclass(frozen=True)
class BezierControlPointOpt(Op):
    """Optimize cubic Bezier control points for the current edge routes.

    The op delegates the differentiable control-point update to
    :func:`dagua.layout.edge_optimization.optimize_edges`. Randomness is not
    used by the optimizer itself; the update is deterministic for a fixed
    input state and floating-point backend.
    """

    config: BezierControlPointOptConfig = BezierControlPointOptConfig()

    name = "bezier_control_point_opt"
    category = OpCategory.EDGE_ROUTE
    reads = ("pos", "edge_routes")
    writes = ("edge_routes",)
    requires = ("pos", "edge_routes")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Optimize the current routes and store Bezier curves in state.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state carrying positions and existing routes.
        ctx : RuntimeContext
            Execution infrastructure used for logging.

        Returns
        -------
        SolveState
            State with ``edge_routes`` replaced by optimized ``BezierCurve``
            instances.

        Raises
        ------
        ValueError
            If positions or routes are unavailable, or if the route count does
            not match the edge count.
        """
        positions = _require_positions(state=state, op_name=self.name)
        if state.edge_routes is None:
            raise ValueError(f"{self.name} requires state.edge_routes to be set.")

        routes = list(state.edge_routes)
        num_edges = int(problem.edge_index.shape[1])
        if len(routes) != num_edges:
            raise ValueError(
                f"{self.name} expected {num_edges} edge routes, received {len(routes)}."
            )

        curves = [_route_to_bezier(route) for route in routes]
        optimized = optimize_edges(
            curves=curves,
            positions=positions,
            edge_index=problem.edge_index,
            node_sizes=_resolve_node_sizes(problem),
            config=self.config,
            graph=state.extras.get("graph"),
            trace=ctx.trace_sink,
        )
        state.edge_routes = optimized
        return state


@register_op
class ReconstructEdgeRoutes(Op):
    """Rebuild per-edge polyline routes from Sugiyama dummy-node chains."""

    name = "reconstruct_edge_routes"
    category = OpCategory.EDGE_ROUTE
    reads = ("pos", "back_edge_mask", "extras.expanded_graph")
    writes = ("edge_routes",)
    requires = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Reconstruct edge routes from expanded-graph dummy-node paths.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state carrying expanded positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with polyline routes stored in ``edge_routes``.

        Raises
        ------
        ValueError
            If the expanded graph metadata or required expanded positions are
            missing.
        """
        del problem, ctx
        positions = _require_positions(state=state, op_name=self.name)
        expanded_graph = state.extras.get("expanded_graph")
        if expanded_graph is None:
            raise ValueError(f"{self.name} requires state.extras['expanded_graph'].")
        if not hasattr(expanded_graph, "edge_paths"):
            raise ValueError(
                f"{self.name} requires expanded_graph.edge_paths for route reconstruction."
            )

        expected_nodes = getattr(expanded_graph, "num_nodes", positions.shape[0])
        if positions.shape[0] < int(expected_nodes):
            raise ValueError(
                f"{self.name} requires expanded positions for {expected_nodes} nodes, "
                f"received {positions.shape[0]}."
            )

        state.edge_routes = _reconstruct_routes(
            positions=positions,
            edge_paths=expanded_graph.edge_paths,
            reversed_edge_mask=state.back_edge_mask,
        )
        return state
