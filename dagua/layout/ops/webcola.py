"""WebCola-compatible stress descent and VPSC projection operations."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_ZERO_DISTANCE = 1.0e-10
_DERIVATIVE_DISTANCE_EPS = 1.0e-9
_LAGRANGIAN_TOLERANCE = -1.0e-4
_ZERO_UPPERBOUND = -1.0e-10
_SOLVE_COST_EPS = 1.0e-4
_DEFAULT_LINK_DISTANCE = 20.0

WebColaConstraint = Dict[str, Any]


def webcola_initial_positions(
    num_nodes: int, link_distance: float = _DEFAULT_LINK_DISTANCE
) -> torch.Tensor:
    """Return deterministic initial coordinates shared with the WebCola adapter.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    link_distance : float, default=20.0
        Radius scale used for the deterministic circle.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]`` and dtype ``torch.float64``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be nonnegative.")
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=torch.float64)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float64)
    radius = float(link_distance) * max(1.0, float(num_nodes) / (2.0 * math.pi))
    positions = torch.empty((num_nodes, 2), dtype=torch.float64)
    for node_index in range(num_nodes):
        angle = 2.0 * math.pi * float(node_index) / float(num_nodes)
        positions[node_index, 0] = radius * math.cos(angle)
        positions[node_index, 1] = radius * math.sin(angle)
    return positions


def webcola_distance_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    link_distance: float = _DEFAULT_LINK_DISTANCE,
) -> List[List[float]]:
    """Compute WebCola's undirected all-pairs shortest-path distances.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    link_distance : float, default=20.0
        Constant link length used as the Dijkstra edge cost.

    Returns
    -------
    list[list[float]]
        Dense ``N x N`` distance matrix with ``math.inf`` for disconnected
        pairs, matching WebCola's shortest-path calculator.
    """
    adjacency: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() > 0:
        edges = edge_index.detach().to(device="cpu", dtype=torch.long)
        for source, target in zip(edges[0].tolist(), edges[1].tolist()):
            u = int(source)
            v = int(target)
            if u == v or u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
                continue
            adjacency[u].append((v, float(link_distance)))
            adjacency[v].append((u, float(link_distance)))
    return [_dijkstra_webcola(adjacency, source) for source in range(num_nodes)]


def webcola_g_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> List[List[float]]:
    """Build WebCola's p-stress weight matrix.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``. WebCola defaults edges to 1
        and all nonedges to 2.

    Returns
    -------
    list[list[float]]
        Dense ``N x N`` matrix.
    """
    weights = [[2.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for index in range(num_nodes):
        weights[index][index] = 2.0
    if edge_index.numel() == 0:
        return weights
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    edge_weight_list = (
        edge_weights.detach().to(device="cpu", dtype=torch.float64).tolist()
        if edge_weights is not None
        else None
    )
    for edge_pos, (source, target) in enumerate(zip(edges[0].tolist(), edges[1].tolist())):
        u = int(source)
        v = int(target)
        if u == v or u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
            continue
        weight = 1.0 if edge_weight_list is None else float(edge_weight_list[edge_pos])
        weights[u][v] = weight
        weights[v][u] = weight
    return weights


def run_webcola_descent(
    initial_positions: torch.Tensor,
    distances: Sequence[Sequence[float]],
    steps: int,
    g_matrix: Optional[Sequence[Sequence[float]]] = None,
    constraints: Optional[Sequence[WebColaConstraint]] = None,
    threshold: float = 0.01,
) -> Tuple[torch.Tensor, List[float]]:
    """Run WebCola's Runge-Kutta stress descent.

    Parameters
    ----------
    initial_positions : torch.Tensor
        Initial coordinates with shape ``[N, 2]``.
    distances : sequence[sequence[float]]
        Dense ideal-distance matrix with shape ``[N, N]``.
    steps : int
        Maximum number of iterations.
    g_matrix : sequence[sequence[float]], optional
        Optional WebCola p-stress matrix. ``None`` matches the initial
        unconstrained stress phase.
    constraints : sequence[dict[str, Any]], optional
        Optional VPSC constraints applied through the Descent projection hook.
    threshold : float, default=0.01
        WebCola convergence threshold used by ``Descent.run``.

    Returns
    -------
    tuple[torch.Tensor, list[float]]
        Final positions with shape ``[N, 2]`` and per-iteration displacement
        stresses.
    """
    if steps < 0:
        raise ValueError("steps must be nonnegative.")
    positions = initial_positions.detach().to(device="cpu", dtype=torch.float64)
    x = [positions[:, 0].tolist(), positions[:, 1].tolist()]
    descent = _WebColaDescent(x=x, distances=distances, g_matrix=g_matrix, constraints=constraints)
    descent.threshold = float(threshold)
    stresses = descent.run(int(steps))
    result = torch.empty_like(positions)
    result[:, 0] = torch.tensor(descent.x[0], dtype=torch.float64)
    result[:, 1] = torch.tensor(descent.x[1], dtype=torch.float64)
    return result.to(device=initial_positions.device), stresses


def solve_vpsc_1d(
    desired_positions: Sequence[float],
    constraints: Sequence[Tuple[int, int, float, bool]],
    weights: Optional[Sequence[float]] = None,
) -> List[float]:
    """Project one coordinate axis with WebCola's VPSC active-set solver.

    Parameters
    ----------
    desired_positions : sequence[float]
        Desired scalar positions for each variable.
    constraints : sequence[tuple[int, int, float, bool]]
        Separation constraints as ``(left, right, gap, equality)``.
    weights : sequence[float], optional
        Per-variable quadratic weights. Omitted weights default to 1.

    Returns
    -------
    list[float]
        Projected scalar positions.
    """
    variables = [
        _VpscVariable(
            desired_position=float(position),
            weight=1.0 if weights is None else float(weights[index]),
        )
        for index, position in enumerate(desired_positions)
    ]
    vpsc_constraints = [
        _VpscConstraint(
            left=variables[int(left)],
            right=variables[int(right)],
            gap=float(gap),
            equality=bool(equality),
        )
        for left, right, gap, equality in constraints
    ]
    solver = _VpscSolver(variables, vpsc_constraints)
    solver.solve()
    return [variable.position() for variable in variables]


def project_webcola_constraints(
    x0: Sequence[float],
    y0: Sequence[float],
    proposed: Sequence[float],
    constraints: Sequence[WebColaConstraint],
    axis: str,
) -> List[float]:
    """Project a WebCola Descent step on one axis.

    Parameters
    ----------
    x0 : sequence[float]
        Starting x coordinates for this projection phase.
    y0 : sequence[float]
        Starting y coordinates for this projection phase.
    proposed : sequence[float]
        Proposed coordinates for the projected axis.
    constraints : sequence[dict[str, Any]]
        WebCola-style constraints.
    axis : {"x", "y"}
        Axis to project.

    Returns
    -------
    list[float]
        Projected coordinates for ``axis``.
    """
    del x0, y0
    axis_constraints = _axis_constraints(constraints, axis)
    if not axis_constraints:
        return [float(value) for value in proposed]
    return solve_vpsc_1d(proposed, axis_constraints)


def flex_to_webcola_constraints(problem: LayoutProblem) -> List[WebColaConstraint]:
    """Convert supported Dagua Flex constraints to WebCola constraint dictionaries.

    Parameters
    ----------
    problem : LayoutProblem
        Problem containing optional resolved ``FlexConstraints``.

    Returns
    -------
    list[dict[str, Any]]
        WebCola-compatible constraints. Pins are represented as equality
        constraints to synthetic fixed variables by the pipeline, so this
        function currently emits alignment and pairwise separation only.
    """
    flex = problem.flex
    constraints: List[WebColaConstraint] = []
    if flex is None:
        return constraints
    if flex.align_groups:
        for group in flex.align_groups:
            indices, _weight, axis_id = group
            axis = "x" if int(axis_id) == 0 else "y"
            values = indices.detach().to(device="cpu", dtype=torch.long).tolist()
            if len(values) < 2:
                continue
            anchor = int(values[0])
            for node_index in values[1:]:
                constraints.append(
                    {
                        "axis": axis,
                        "left": anchor,
                        "right": int(node_index),
                        "gap": 0.0,
                        "equality": True,
                    }
                )
    if flex.flex_node_sep is not None and flex.flex_node_sep_weight:
        gap = float(flex.flex_node_sep)
        for left in range(problem.num_nodes):
            for right in range(left + 1, problem.num_nodes):
                constraints.append({"axis": "x", "left": left, "right": right, "gap": gap})
    return constraints


@dataclass(frozen=True)
class ColaDescentConfig:
    """Configuration for the reusable WebCola descent op.

    Parameters
    ----------
    steps : int
        Number of Runge-Kutta iterations.
    link_distance : float
        Constant link distance for graph shortest paths.
    constrained : bool
        Whether to use constraints in ``state.extras["webcola_constraints"]``.
    p_stress : bool
        Whether to apply WebCola's p-stress matrix.
    threshold : float
        Convergence threshold.
    """

    steps: int = 50
    link_distance: float = _DEFAULT_LINK_DISTANCE
    constrained: bool = False
    p_stress: bool = False
    threshold: float = 0.01


@register_op
class InitializeWebColaPositions(Op):
    """Initialize positions with the pinned deterministic WebCola fixture start."""

    name = "webcola_init_positions"
    category = OpCategory.INIT
    writes = ("pos",)

    def __init__(self, link_distance: float = _DEFAULT_LINK_DISTANCE) -> None:
        """Create the initializer.

        Parameters
        ----------
        link_distance : float, default=20.0
            Radius scale passed to :func:`webcola_initial_positions`.

        Returns
        -------
        None
            Constructor only stores configuration.
        """
        self.link_distance = float(link_distance)

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Write deterministic initial coordinates to state.

        Parameters
        ----------
        problem : LayoutProblem
            Layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context.

        Returns
        -------
        SolveState
            State with ``pos`` set.
        """
        del ctx
        state.pos = webcola_initial_positions(problem.num_nodes, self.link_distance).to(
            device=problem.edge_index.device
        )
        return state


@register_op
class RunWebColaDescent(Op):
    """Run the WebCola Runge-Kutta stress descent op."""

    name = "webcola_run_descent"
    category = OpCategory.OPTIMIZE
    reads = ("pos", "edge_index", "extras")
    writes = ("pos", "extras")

    def __init__(self, config: ColaDescentConfig) -> None:
        """Create the descent op.

        Parameters
        ----------
        config : ColaDescentConfig
            WebCola descent configuration.

        Returns
        -------
        None
            Constructor only stores configuration.
        """
        self.config = config

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Run WebCola descent and store trace stresses.

        Parameters
        ----------
        problem : LayoutProblem
            Layout problem.
        state : SolveState
            Mutable solve state containing initial positions.
        ctx : RuntimeContext
            Runtime context.

        Returns
        -------
        SolveState
            State with updated positions.
        """
        del ctx
        distances = webcola_distance_matrix(
            problem.edge_index,
            problem.num_nodes,
            self.config.link_distance,
        )
        g_matrix = (
            webcola_g_matrix(problem.edge_index, problem.num_nodes, problem.edge_weights)
            if self.config.p_stress
            else None
        )
        constraints = state.extras.get("webcola_constraints") if self.config.constrained else None
        state.pos, stresses = run_webcola_descent(
            state.pos,
            distances,
            self.config.steps,
            g_matrix=g_matrix,
            constraints=constraints,
            threshold=self.config.threshold,
        )
        state.extras["webcola_stresses"] = stresses
        return state


@register_op
class BuildWebColaFlexConstraints(Op):
    """Build WebCola constraints from resolved Dagua Flex inputs."""

    name = "webcola_build_flex_constraints"
    category = OpCategory.PREPROCESS
    writes = ("extras",)

    def apply(self, problem: LayoutProblem, state: SolveState, ctx: RuntimeContext) -> SolveState:
        """Store WebCola constraints in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context.

        Returns
        -------
        SolveState
            State with ``webcola_constraints`` populated.
        """
        del ctx
        explicit = state.extras.get("webcola_constraints", [])
        state.extras["webcola_constraints"] = list(explicit) + flex_to_webcola_constraints(problem)
        return state


class _WebColaDescent:
    """Source-faithful Python port of WebCola ``Descent``."""

    def __init__(
        self,
        x: List[List[float]],
        distances: Sequence[Sequence[float]],
        g_matrix: Optional[Sequence[Sequence[float]]] = None,
        constraints: Optional[Sequence[WebColaConstraint]] = None,
    ) -> None:
        """Initialize descent work arrays.

        Parameters
        ----------
        x : list[list[float]]
            Mutable coordinates as ``[x_values, y_values]``.
        distances : sequence[sequence[float]]
            Dense ideal-distance matrix.
        g_matrix : sequence[sequence[float]], optional
            Optional p-stress matrix.
        constraints : sequence[dict[str, Any]], optional
            Optional projection constraints.

        Returns
        -------
        None
            Constructor initializes mutable work arrays.
        """
        self.x = x
        self.d = [[float(value) for value in row] for row in distances]
        self.g_matrix = (
            None if g_matrix is None else [[float(value) for value in row] for row in g_matrix]
        )
        self.constraints = list(constraints or [])
        self.k = len(x)
        self.n = len(x[0]) if x else 0
        self.threshold = 0.0001
        self.g = [[0.0] * self.n for _ in range(self.k)]
        self.h = [[[0.0] * self.n for _ in range(self.n)] for _ in range(self.k)]
        self.hd = [[0.0] * self.n for _ in range(self.k)]
        self.a = [[0.0] * self.n for _ in range(self.k)]
        self.b = [[0.0] * self.n for _ in range(self.k)]
        self.c = [[0.0] * self.n for _ in range(self.k)]
        self.d_work = [[0.0] * self.n for _ in range(self.k)]
        self.e = [[0.0] * self.n for _ in range(self.k)]
        self.ia = [[0.0] * self.n for _ in range(self.k)]
        self.ib = [[0.0] * self.n for _ in range(self.k)]
        self._min_d = self._minimum_positive_distance()

    def run(self, iterations: int) -> List[float]:
        """Run until convergence or iteration exhaustion.

        Parameters
        ----------
        iterations : int
            Maximum Runge-Kutta iterations.

        Returns
        -------
        list[float]
            Displacement stress per completed iteration.
        """
        stress = float("inf")
        stresses: List[float] = []
        while iterations > 0:
            current = self.runge_kutta()
            stresses.append(current)
            iterations -= 1
            converged = current != 0.0 and abs(stress / current - 1.0) < self.threshold
            stress = current
            if converged:
                break
        return stresses

    def runge_kutta(self) -> float:
        """Apply one WebCola Runge-Kutta iteration.

        Parameters
        ----------
        None
            Uses mutable descent state.

        Returns
        -------
        float
            Squared displacement stress returned by WebCola.
        """
        self._compute_next_position(self.x, self.a)
        self._mid(self.x, self.a, self.ia)
        self._compute_next_position(self.ia, self.b)
        self._mid(self.x, self.b, self.ib)
        self._compute_next_position(self.ib, self.c)
        self._compute_next_position(self.c, self.d_work)
        displacement = 0.0
        for axis in range(self.k):
            for node_index in range(self.n):
                value = (
                    self.a[axis][node_index]
                    + 2.0 * self.b[axis][node_index]
                    + 2.0 * self.c[axis][node_index]
                    + self.d_work[axis][node_index]
                ) / 6.0
                delta = self.x[axis][node_index] - value
                displacement += delta * delta
                self.x[axis][node_index] = value
        return displacement

    def compute_derivatives(self, x: List[List[float]]) -> None:
        """Compute WebCola gradient and Hessian diagonal blocks.

        Parameters
        ----------
        x : list[list[float]]
            Coordinates as ``[axis][node]``.

        Returns
        -------
        None
            The method mutates ``self.g`` and ``self.h``.
        """
        if self.n < 1:
            return
        for u in range(self.n):
            huu = [0.0 for _ in range(self.k)]
            for axis in range(self.k):
                self.g[axis][u] = 0.0
            for v in range(self.n):
                if u == v:
                    continue
                diffs = [0.0 for _ in range(self.k)]
                diff_squares = [0.0 for _ in range(self.k)]
                squared_distance = 0.0
                for axis in range(self.k):
                    diff = x[axis][u] - x[axis][v]
                    diffs[axis] = diff
                    diff_squares[axis] = diff * diff
                    squared_distance += diff_squares[axis]
                if squared_distance <= _DERIVATIVE_DISTANCE_EPS:
                    offset = self._offset_dir(u, v)
                    for axis in range(self.k):
                        x[axis][v] += offset[axis]
                    squared_distance = 0.0
                    for axis in range(self.k):
                        diff = x[axis][u] - x[axis][v]
                        diffs[axis] = diff
                        diff_squares[axis] = diff * diff
                        squared_distance += diff_squares[axis]
                length = math.sqrt(squared_distance)
                ideal = self.d[u][v]
                weight = self.g_matrix[u][v] if self.g_matrix is not None else 1.0
                if (weight > 1.0 and length > ideal) or not math.isfinite(ideal):
                    for axis in range(self.k):
                        self.h[axis][u][v] = 0.0
                    continue
                if weight > 1.0:
                    weight = 1.0
                ideal_sq = ideal * ideal
                gradient_scale = 2.0 * weight * (length - ideal) / (ideal_sq * length)
                length_cubed = length * length * length
                hessian_scale = -2.0 * weight / (ideal_sq * length_cubed)
                for axis in range(self.k):
                    self.g[axis][u] += diffs[axis] * gradient_scale
                    self.h[axis][u][v] = hessian_scale * (
                        length_cubed
                        + ideal * (diff_squares[axis] - squared_distance)
                        + length * squared_distance
                    )
                    huu[axis] -= self.h[axis][u][v]
            for axis in range(self.k):
                self.h[axis][u][u] = huu[axis]

    def compute_step_size(self, direction: List[List[float]]) -> float:
        """Compute WebCola's optimal scalar step size.

        Parameters
        ----------
        direction : list[list[float]]
            Descent direction with shape ``[2, N]``.

        Returns
        -------
        float
            Scalar step size.
        """
        numerator = 0.0
        denominator = 0.0
        for axis in range(self.k):
            numerator += _dot(self.g[axis], direction[axis])
            _right_multiply(self.h[axis], direction[axis], self.hd[axis])
            denominator += _dot(direction[axis], self.hd[axis])
        if denominator == 0.0 or not math.isfinite(denominator):
            return 0.0
        return numerator / denominator

    def _compute_next_position(self, x0: List[List[float]], result: List[List[float]]) -> None:
        """Compute one projected descent endpoint.

        Parameters
        ----------
        x0 : list[list[float]]
            Starting coordinates.
        result : list[list[float]]
            Mutable output coordinates.

        Returns
        -------
        None
            ``result`` is overwritten.
        """
        self.compute_derivatives(x0)
        alpha = self.compute_step_size(self.g)
        self._step_and_project(x0, result, self.g, alpha)
        if self.constraints:
            for axis in range(self.k):
                for node_index in range(self.n):
                    self.e[axis][node_index] = x0[axis][node_index] - result[axis][node_index]
            beta = max(0.2, min(self.compute_step_size(self.e), 1.0))
            self._step_and_project(x0, result, self.e, beta)

    def _step_and_project(
        self,
        x0: List[List[float]],
        result: List[List[float]],
        direction: List[List[float]],
        step_size: float,
    ) -> None:
        """Take a descent step and apply axis projections.

        Parameters
        ----------
        x0 : list[list[float]]
            Starting coordinates.
        result : list[list[float]]
            Mutable output coordinates.
        direction : list[list[float]]
            Descent direction.
        step_size : float
            Scalar step size.

        Returns
        -------
        None
            ``result`` is overwritten.
        """
        for axis in range(self.k):
            for node_index in range(self.n):
                result[axis][node_index] = x0[axis][node_index]
        self._take_descent_step(result[0], direction[0], step_size)
        if self.constraints:
            result[0][:] = project_webcola_constraints(
                x0[0], x0[1], result[0], self.constraints, "x"
            )
        self._take_descent_step(result[1], direction[1], step_size)
        if self.constraints:
            result[1][:] = project_webcola_constraints(
                result[0], x0[1], result[1], self.constraints, "y"
            )

    def _take_descent_step(
        self,
        values: List[float],
        direction: Sequence[float],
        step_size: float,
    ) -> None:
        """Apply ``values -= step_size * direction``.

        Parameters
        ----------
        values : list[float]
            Mutable coordinate vector.
        direction : sequence[float]
            Descent direction.
        step_size : float
            Scalar step size.

        Returns
        -------
        None
            ``values`` is mutated in place.
        """
        for index in range(self.n):
            values[index] -= step_size * direction[index]

    def _minimum_positive_distance(self) -> float:
        """Return the minimum positive finite ideal distance.

        Parameters
        ----------
        None
            Uses ``self.d``.

        Returns
        -------
        float
            Minimum positive distance, or 1 when no such distance exists.
        """
        min_distance = float("inf")
        for i in range(self.n):
            for j in range(i + 1, self.n):
                value = self.d[i][j]
                if 0.0 < value < min_distance:
                    min_distance = value
        return 1.0 if min_distance == float("inf") else min_distance

    def _offset_dir(self, u: int, v: int) -> List[float]:
        """Return a deterministic zero-distance displacement.

        Parameters
        ----------
        u : int
            First node index.
        v : int
            Second node index.

        Returns
        -------
        list[float]
            Offset vector with length equal to dimensionality.
        """
        values = []
        length_sq = 0.0
        for axis in range(self.k):
            raw = 0.01 + (0.99 * _lcg_unit(u, v, axis)) - 0.5
            values.append(raw)
            length_sq += raw * raw
        length = math.sqrt(length_sq)
        if length == 0.0:
            return [self._min_d if axis == 0 else 0.0 for axis in range(self.k)]
        return [value * self._min_d / length for value in values]

    def _mid(
        self,
        first: List[List[float]],
        second: List[List[float]],
        result: List[List[float]],
    ) -> None:
        """Compute midpoint arrays.

        Parameters
        ----------
        first : list[list[float]]
            First coordinate matrix.
        second : list[list[float]]
            Second coordinate matrix.
        result : list[list[float]]
            Mutable output matrix.

        Returns
        -------
        None
            ``result`` is overwritten.
        """
        for axis in range(self.k):
            for node_index in range(self.n):
                result[axis][node_index] = (
                    first[axis][node_index]
                    + (second[axis][node_index] - first[axis][node_index]) / 2.0
                )


class _VpscConstraint:
    """Internal VPSC constraint."""

    def __init__(
        self,
        left: "_VpscVariable",
        right: "_VpscVariable",
        gap: float,
        equality: bool = False,
    ) -> None:
        """Create an internal VPSC constraint.

        Parameters
        ----------
        left : _VpscVariable
            Left variable.
        right : _VpscVariable
            Right variable.
        gap : float
            Required minimum separation.
        equality : bool, default=False
            Whether the constraint is equality.

        Returns
        -------
        None
            Constructor stores references.
        """
        self.left = left
        self.right = right
        self.gap = float(gap)
        self.equality = bool(equality)
        self.lm = 0.0
        self.active = False
        self.unsatisfiable = False

    def slack(self) -> float:
        """Return current constraint slack.

        Parameters
        ----------
        None
            Uses variable positions.

        Returns
        -------
        float
            Positive values satisfy the constraint.
        """
        if self.unsatisfiable:
            return float("inf")
        return (
            self.right.scale * self.right.position()
            - self.gap
            - self.left.scale * self.left.position()
        )


class _VpscVariable:
    """Internal VPSC variable."""

    def __init__(self, desired_position: float, weight: float = 1.0, scale: float = 1.0) -> None:
        """Create an internal VPSC variable.

        Parameters
        ----------
        desired_position : float
            Desired scalar position.
        weight : float, default=1.0
            Quadratic weight.
        scale : float, default=1.0
            Variable scale.

        Returns
        -------
        None
            Constructor initializes solver fields.
        """
        self.desired_position = float(desired_position)
        self.weight = float(weight)
        self.scale = float(scale)
        self.offset = 0.0
        self.block: Optional[_VpscBlock] = None
        self.c_in: List[_VpscConstraint] = []
        self.c_out: List[_VpscConstraint] = []

    def dfdv(self) -> float:
        """Return objective derivative at this variable.

        Parameters
        ----------
        None
            Uses the current variable position.

        Returns
        -------
        float
            Objective derivative.
        """
        return 2.0 * self.weight * (self.position() - self.desired_position)

    def position(self) -> float:
        """Return current variable position.

        Parameters
        ----------
        None
            Uses owning block position.

        Returns
        -------
        float
            Scalar position.
        """
        if self.block is None:
            return self.desired_position
        return (self.block.ps_scale * self.block.posn + self.offset) / self.scale

    def visit_neighbours(
        self,
        previous: Optional["_VpscVariable"],
        visitor: Any,
    ) -> None:
        """Visit active neighbours in the block tree.

        Parameters
        ----------
        previous : _VpscVariable or None
            Variable to avoid revisiting.
        visitor : callable
            Callback receiving ``(constraint, next_variable)``.

        Returns
        -------
        None
            Visitor side effects only.
        """
        for constraint in self.c_out:
            if constraint.active and previous is not constraint.right:
                visitor(constraint, constraint.right)
        for constraint in self.c_in:
            if constraint.active and previous is not constraint.left:
                visitor(constraint, constraint.left)


class _VpscBlock:
    """Internal VPSC block."""

    def __init__(self, variable: _VpscVariable) -> None:
        """Create a block containing one variable.

        Parameters
        ----------
        variable : _VpscVariable
            Initial block variable.

        Returns
        -------
        None
            Constructor initializes aggregate position statistics.
        """
        self.variables: List[_VpscVariable] = []
        self.posn = 0.0
        self.ps_scale = variable.scale
        self.ab = 0.0
        self.ad = 0.0
        self.a2 = 0.0
        self.block_index = 0
        variable.offset = 0.0
        self.add_variable(variable)

    def add_variable(self, variable: _VpscVariable) -> None:
        """Add a variable to this block.

        Parameters
        ----------
        variable : _VpscVariable
            Variable to add.

        Returns
        -------
        None
            The block aggregate is updated.
        """
        variable.block = self
        self.variables.append(variable)
        ai = self.ps_scale / variable.scale
        bi = variable.offset / variable.scale
        self.ab += variable.weight * ai * bi
        self.ad += variable.weight * ai * variable.desired_position
        self.a2 += variable.weight * ai * ai
        self.posn = (self.ad - self.ab) / self.a2

    def update_weighted_position(self) -> None:
        """Refresh the weighted optimal block position.

        Parameters
        ----------
        None
            Uses block variables.

        Returns
        -------
        None
            The block position is updated.
        """
        self.ab = self.ad = self.a2 = 0.0
        for variable in self.variables:
            ai = self.ps_scale / variable.scale
            bi = variable.offset / variable.scale
            self.ab += variable.weight * ai * bi
            self.ad += variable.weight * ai * variable.desired_position
            self.a2 += variable.weight * ai * ai
        self.posn = (self.ad - self.ab) / self.a2

    def compute_lm(
        self,
        variable: _VpscVariable,
        previous: Optional[_VpscVariable],
        post_action: Any,
    ) -> float:
        """Compute Lagrange multipliers over the active tree.

        Parameters
        ----------
        variable : _VpscVariable
            Current variable.
        previous : _VpscVariable or None
            Previous variable.
        post_action : callable
            Callback applied after each constraint.

        Returns
        -------
        float
            Derivative contribution.
        """
        dfdv = variable.dfdv()

        def visit(constraint: _VpscConstraint, next_variable: _VpscVariable) -> None:
            nonlocal dfdv
            child = self.compute_lm(next_variable, variable, post_action)
            if next_variable is constraint.right:
                dfdv += child * constraint.left.scale
                constraint.lm = child
            else:
                dfdv += child * constraint.right.scale
                constraint.lm = -child
            post_action(constraint)

        variable.visit_neighbours(previous, visit)
        return dfdv / variable.scale

    def find_min_lm(self) -> Optional[_VpscConstraint]:
        """Find the active non-equality constraint with minimum multiplier.

        Parameters
        ----------
        None
            Uses the active block tree.

        Returns
        -------
        _VpscConstraint or None
            Split candidate.
        """
        best: Optional[_VpscConstraint] = None

        def capture(constraint: _VpscConstraint) -> None:
            nonlocal best
            if not constraint.equality and (best is None or constraint.lm < best.lm):
                best = constraint

        self.compute_lm(self.variables[0], None, capture)
        return best

    def populate_split_block(
        self,
        variable: _VpscVariable,
        previous: Optional[_VpscVariable],
    ) -> None:
        """Populate a split block recursively.

        Parameters
        ----------
        variable : _VpscVariable
            Current variable.
        previous : _VpscVariable or None
            Previous variable.

        Returns
        -------
        None
            Variables are added to this block.
        """

        def visit(constraint: _VpscConstraint, next_variable: _VpscVariable) -> None:
            next_variable.offset = variable.offset + (
                constraint.gap if next_variable is constraint.right else -constraint.gap
            )
            self.add_variable(next_variable)
            self.populate_split_block(next_variable, variable)

        variable.visit_neighbours(previous, visit)

    def is_active_directed_path_between(self, source: _VpscVariable, target: _VpscVariable) -> bool:
        """Return whether an active directed path exists from source to target.

        Parameters
        ----------
        source : _VpscVariable
            Path source.
        target : _VpscVariable
            Path target.

        Returns
        -------
        bool
            ``True`` when the active directed path exists.
        """
        if source is target:
            return True
        return any(
            constraint.active and self.is_active_directed_path_between(constraint.right, target)
            for constraint in source.c_out
        )

    def find_path(
        self,
        variable: _VpscVariable,
        previous: Optional[_VpscVariable],
        target: _VpscVariable,
        visitor: Any,
    ) -> bool:
        """Find an active path between two variables.

        Parameters
        ----------
        variable : _VpscVariable
            Current variable.
        previous : _VpscVariable or None
            Previous variable.
        target : _VpscVariable
            Target variable.
        visitor : callable
            Callback applied to constraints on the path.

        Returns
        -------
        bool
            Whether the target was found.
        """
        found = False

        def visit(constraint: _VpscConstraint, next_variable: _VpscVariable) -> None:
            nonlocal found
            if not found and (
                next_variable is target or self.find_path(next_variable, variable, target, visitor)
            ):
                found = True
                visitor(constraint, next_variable)

        variable.visit_neighbours(previous, visit)
        return found

    def find_min_lm_between(
        self,
        left_variable: _VpscVariable,
        right_variable: _VpscVariable,
    ) -> Optional[_VpscConstraint]:
        """Find the minimum multiplier on the path between two variables.

        Parameters
        ----------
        left_variable : _VpscVariable
            Left endpoint.
        right_variable : _VpscVariable
            Right endpoint.

        Returns
        -------
        _VpscConstraint or None
            Split candidate.
        """
        self.compute_lm(left_variable, None, lambda _constraint: None)
        best: Optional[_VpscConstraint] = None

        def capture(constraint: _VpscConstraint, next_variable: _VpscVariable) -> None:
            nonlocal best
            if (
                not constraint.equality
                and constraint.right is next_variable
                and (best is None or constraint.lm < best.lm)
            ):
                best = constraint

        self.find_path(left_variable, None, right_variable, capture)
        return best

    def split_between(
        self,
        left_variable: _VpscVariable,
        right_variable: _VpscVariable,
    ) -> Optional[Tuple[_VpscConstraint, "_VpscBlock", "_VpscBlock"]]:
        """Split this block between two variables if possible.

        Parameters
        ----------
        left_variable : _VpscVariable
            Left endpoint.
        right_variable : _VpscVariable
            Right endpoint.

        Returns
        -------
        tuple or None
            Split constraint and the two new blocks.
        """
        constraint = self.find_min_lm_between(left_variable, right_variable)
        if constraint is None:
            return None
        left_block, right_block = _split_block(constraint)
        return constraint, left_block, right_block

    def merge_across(
        self, block: "_VpscBlock", constraint: _VpscConstraint, distance: float
    ) -> None:
        """Merge another block into this one across an active constraint.

        Parameters
        ----------
        block : _VpscBlock
            Block to merge.
        constraint : _VpscConstraint
            Constraint activated by the merge.
        distance : float
            Offset adjustment.

        Returns
        -------
        None
            This block is mutated.
        """
        constraint.active = True
        for variable in block.variables:
            variable.offset += distance
            self.add_variable(variable)
        self.posn = (self.ad - self.ab) / self.a2

    def cost(self) -> float:
        """Return block objective cost.

        Parameters
        ----------
        None
            Uses block variables.

        Returns
        -------
        float
            Weighted squared error.
        """
        total = 0.0
        for variable in self.variables:
            delta = variable.position() - variable.desired_position
            total += delta * delta * variable.weight
        return total


class _VpscBlocks:
    """Internal VPSC block collection."""

    def __init__(self, variables: Sequence[_VpscVariable]) -> None:
        """Create one singleton block per variable.

        Parameters
        ----------
        variables : sequence[_VpscVariable]
            VPSC variables.

        Returns
        -------
        None
            Constructor initializes block list.
        """
        self.blocks = [_VpscBlock(variable) for variable in variables]
        for index, block in enumerate(self.blocks):
            block.block_index = index

    def cost(self) -> float:
        """Return total block cost.

        Parameters
        ----------
        None
            Uses all blocks.

        Returns
        -------
        float
            Total objective cost.
        """
        return sum(block.cost() for block in self.blocks)

    def insert(self, block: _VpscBlock) -> None:
        """Insert a block.

        Parameters
        ----------
        block : _VpscBlock
            Block to insert.

        Returns
        -------
        None
            The block collection is mutated.
        """
        block.block_index = len(self.blocks)
        self.blocks.append(block)

    def remove(self, block: _VpscBlock) -> None:
        """Remove a block by swap-pop.

        Parameters
        ----------
        block : _VpscBlock
            Block to remove.

        Returns
        -------
        None
            The block collection is mutated.
        """
        last = self.blocks[-1]
        self.blocks.pop()
        if block is not last:
            self.blocks[block.block_index] = last
            last.block_index = block.block_index

    def merge(self, constraint: _VpscConstraint) -> None:
        """Merge the blocks separated by a constraint.

        Parameters
        ----------
        constraint : _VpscConstraint
            Constraint to activate.

        Returns
        -------
        None
            Blocks are merged in place.
        """
        left = constraint.left.block
        right = constraint.right.block
        if left is None or right is None or left is right:
            return
        distance = constraint.right.offset - constraint.left.offset - constraint.gap
        if len(left.variables) < len(right.variables):
            right.merge_across(left, constraint, distance)
            self.remove(left)
        else:
            left.merge_across(right, constraint, -distance)
            self.remove(right)

    def update_block_positions(self) -> None:
        """Update all weighted block positions.

        Parameters
        ----------
        None
            Uses all blocks.

        Returns
        -------
        None
            Blocks are updated in place.
        """
        for block in self.blocks:
            block.update_weighted_position()

    def split(self, inactive: List[_VpscConstraint]) -> None:
        """Split blocks with negative Lagrange multipliers.

        Parameters
        ----------
        inactive : list[_VpscConstraint]
            Inactive constraint list to append deactivated constraints to.

        Returns
        -------
        None
            Blocks and inactive constraints are mutated.
        """
        self.update_block_positions()
        for block in list(self.blocks):
            constraint = block.find_min_lm()
            if constraint is not None and constraint.lm < _LAGRANGIAN_TOLERANCE:
                old_block = constraint.left.block
                for new_block in _split_block(constraint):
                    self.insert(new_block)
                if old_block is not None:
                    self.remove(old_block)
                inactive.append(constraint)


class _VpscSolver:
    """Internal WebCola VPSC active-set solver."""

    def __init__(self, variables: List[_VpscVariable], constraints: List[_VpscConstraint]) -> None:
        """Create a VPSC solver.

        Parameters
        ----------
        variables : list[_VpscVariable]
            Solver variables.
        constraints : list[_VpscConstraint]
            Solver constraints.

        Returns
        -------
        None
            Constructor wires variable adjacency.
        """
        self.variables = variables
        self.constraints = constraints
        for variable in self.variables:
            variable.c_in = []
            variable.c_out = []
        for constraint in self.constraints:
            constraint.left.c_out.append(constraint)
            constraint.right.c_in.append(constraint)
            constraint.active = False
        self.inactive = list(self.constraints)
        self.blocks: Optional[_VpscBlocks] = None

    def solve(self) -> float:
        """Solve the VPSC projection.

        Parameters
        ----------
        None
            Uses solver variables and constraints.

        Returns
        -------
        float
            Final objective cost.
        """
        self.satisfy()
        last_cost = float("inf")
        cost = self.cost()
        while abs(last_cost - cost) > _SOLVE_COST_EPS:
            self.satisfy()
            last_cost = cost
            cost = self.cost()
        return cost

    def cost(self) -> float:
        """Return current solver cost.

        Parameters
        ----------
        None
            Uses current block structure.

        Returns
        -------
        float
            Objective cost.
        """
        if self.blocks is None:
            return 0.0
        return self.blocks.cost()

    def satisfy(self) -> None:
        """Satisfy violated constraints by merging and splitting blocks.

        Parameters
        ----------
        None
            Uses current block structure.

        Returns
        -------
        None
            Solver state is mutated.
        """
        if self.blocks is None:
            self.blocks = _VpscBlocks(self.variables)
        self.blocks.split(self.inactive)
        violated = self._most_violated()
        while violated is not None and (
            violated.equality or (violated.slack() < _ZERO_UPPERBOUND and not violated.active)
        ):
            left_block = violated.left.block
            right_block = violated.right.block
            if left_block is not right_block:
                self.blocks.merge(violated)
            elif left_block is not None:
                if left_block.is_active_directed_path_between(violated.right, violated.left):
                    violated.unsatisfiable = True
                    violated = self._most_violated()
                    continue
                split = left_block.split_between(violated.left, violated.right)
                if split is None:
                    violated.unsatisfiable = True
                    violated = self._most_violated()
                    continue
                split_constraint, left_new, right_new = split
                self.blocks.insert(left_new)
                self.blocks.insert(right_new)
                self.blocks.remove(left_block)
                self.inactive.append(split_constraint)
                if violated.slack() >= 0.0:
                    self.inactive.append(violated)
                else:
                    self.blocks.merge(violated)
            violated = self._most_violated()

    def _most_violated(self) -> Optional[_VpscConstraint]:
        """Return and remove the most violated inactive constraint.

        Parameters
        ----------
        None
            Uses inactive constraints.

        Returns
        -------
        _VpscConstraint or None
            Most violated constraint.
        """
        min_slack = float("inf")
        violated: Optional[_VpscConstraint] = None
        delete_index = len(self.inactive)
        for index, constraint in enumerate(self.inactive):
            if constraint.unsatisfiable:
                continue
            slack = constraint.slack()
            if constraint.equality or slack < min_slack:
                min_slack = slack
                violated = constraint
                delete_index = index
                if constraint.equality:
                    break
        if (
            violated is not None
            and delete_index != len(self.inactive)
            and ((min_slack < _ZERO_UPPERBOUND and not violated.active) or violated.equality)
        ):
            self.inactive[delete_index] = self.inactive[-1]
            self.inactive.pop()
        return violated


def _dijkstra_webcola(adjacency: Sequence[Sequence[Tuple[int, float]]], source: int) -> List[float]:
    """Run Dijkstra with WebCola-equivalent undirected edge costs.

    Parameters
    ----------
    adjacency : sequence[sequence[tuple[int, float]]]
        Undirected adjacency list.
    source : int
        Source node.

    Returns
    -------
    list[float]
        Shortest distances from ``source``.
    """
    distances = [float("inf")] * len(adjacency)
    distances[source] = 0.0
    queue: List[Tuple[float, int]] = [(0.0, source)]
    while queue:
        current_distance, node = heapq.heappop(queue)
        if current_distance != distances[node]:
            continue
        for neighbor, cost in adjacency[node]:
            next_distance = current_distance + cost
            if distances[neighbor] > next_distance:
                distances[neighbor] = next_distance
                heapq.heappush(queue, (next_distance, neighbor))
    return distances


def _axis_constraints(
    constraints: Sequence[WebColaConstraint],
    axis: str,
) -> List[Tuple[int, int, float, bool]]:
    """Return VPSC constraints for a single axis.

    Parameters
    ----------
    constraints : sequence[dict[str, Any]]
        WebCola-style constraints.
    axis : str
        Axis name.

    Returns
    -------
    list[tuple[int, int, float, bool]]
        VPSC tuple constraints.
    """
    result: List[Tuple[int, int, float, bool]] = []
    for constraint in constraints:
        if constraint.get("axis") != axis:
            continue
        result.append(
            (
                int(constraint["left"]),
                int(constraint["right"]),
                float(constraint.get("gap", 0.0)),
                bool(constraint.get("equality", False)),
            )
        )
    return result


def _split_block(constraint: _VpscConstraint) -> Tuple[_VpscBlock, _VpscBlock]:
    """Split a block by deactivating one active constraint.

    Parameters
    ----------
    constraint : _VpscConstraint
        Active constraint to deactivate.

    Returns
    -------
    tuple[_VpscBlock, _VpscBlock]
        New left and right blocks.
    """
    constraint.active = False
    return _create_split_block(constraint.left), _create_split_block(constraint.right)


def _create_split_block(start_variable: _VpscVariable) -> _VpscBlock:
    """Create a split block from one variable.

    Parameters
    ----------
    start_variable : _VpscVariable
        Starting variable.

    Returns
    -------
    _VpscBlock
        New block populated through active constraints.
    """
    block = _VpscBlock(start_variable)
    block.populate_split_block(start_variable, None)
    return block


def _dot(left: Sequence[float], right: Sequence[float]) -> float:
    """Return a dot product.

    Parameters
    ----------
    left : sequence[float]
        Left vector.
    right : sequence[float]
        Right vector.

    Returns
    -------
    float
        Dot product.
    """
    total = 0.0
    for left_value, right_value in zip(left, right):
        total += left_value * right_value
    return total


def _right_multiply(
    matrix: Sequence[Sequence[float]], vector: Sequence[float], result: List[float]
) -> None:
    """Compute ``result = matrix * vector``.

    Parameters
    ----------
    matrix : sequence[sequence[float]]
        Matrix.
    vector : sequence[float]
        Vector.
    result : list[float]
        Mutable output vector.

    Returns
    -------
    None
        ``result`` is overwritten.
    """
    for row_index, row in enumerate(matrix):
        result[row_index] = _dot(row, vector)


def _lcg_unit(first: int, second: int, axis: int) -> float:
    """Return a deterministic unit interval value for zero-distance nudges.

    Parameters
    ----------
    first : int
        First node index.
    second : int
        Second node index.
    axis : int
        Coordinate axis.

    Returns
    -------
    float
        Value in ``[0, 1)``.
    """
    state = (first * 1664525 + second * 1013904223 + axis * 2654435761) & 0xFFFFFFFF
    state = (1664525 * state + 1013904223) & 0xFFFFFFFF
    return float(state) / 4294967296.0
