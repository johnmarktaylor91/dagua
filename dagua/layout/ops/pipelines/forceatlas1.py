"""Gephi ForceAtlas1 layout pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState

_JAVA_RANDOM_MULTIPLIER = 0x5DEECE66D
_JAVA_RANDOM_ADDEND = 0xB
_JAVA_RANDOM_MASK = (1 << 48) - 1
_JAVA_RANDOM_DOUBLE_SCALE = float(1 << 53)
_GEphi_INIT_OFFSET = 0.01
_GEphi_INIT_SCALE = 1000.0
_GEphi_INIT_SHIFT = 500.0
_REPULSION_FACTOR = 0.001
_ATTRACTION_FACTOR = 0.01
_GRAVITY_FACTOR = 0.0001
_GRAVITY_EPSILON = 0.0001
_FREEZE_SPEED_MULTIPLIER = 10.0
_FIXED_NODE_ATTRACTION_BONUS = 100.0


class _JavaRandom:
    """Minimal port of ``java.util.Random`` for ForceAtlas initialization."""

    def __init__(self, seed: int) -> None:
        """Initialize the Java-compatible linear congruential generator.

        Parameters
        ----------
        seed : int
            Signed or unsigned seed value passed to Gephi's layout initializer.

        Returns
        -------
        None
            Stores the scrambled 48-bit Java random seed.
        """
        self._seed = (int(seed) ^ _JAVA_RANDOM_MULTIPLIER) & _JAVA_RANDOM_MASK

    def _next(self, bits: int) -> int:
        """Return the next Java random integer fragment.

        Parameters
        ----------
        bits : int
            Number of high bits to return from the 48-bit state.

        Returns
        -------
        int
            Non-negative random fragment matching ``Random.next(bits)``.
        """
        self._seed = (
            (self._seed * _JAVA_RANDOM_MULTIPLIER) + _JAVA_RANDOM_ADDEND
        ) & _JAVA_RANDOM_MASK
        return self._seed >> (48 - bits)

    def next_double(self) -> float:
        """Return the next Java ``Random.nextDouble`` value.

        Returns
        -------
        float
            Double-precision value in ``[0, 1)``.
        """
        high = self._next(26)
        low = self._next(27)
        return float((high << 27) + low) / _JAVA_RANDOM_DOUBLE_SCALE


@dataclass
class _ForceAtlas1Node:
    """Mutable node state matching Gephi's float-backed layout data.

    Parameters
    ----------
    x : np.float32
        Current x-coordinate.
    y : np.float32
        Current y-coordinate.
    size : float
        Gephi node size used by ``adjustSizes`` anti-collision mode.
    fixed : bool
        Whether the node is fixed. Dagua's headless API does not expose fixed
        nodes, so this is always ``False`` for public calls.
    dx : np.float32
        Current x displacement accumulator.
    dy : np.float32
        Current y displacement accumulator.
    old_dx : np.float32
        Previous iteration x displacement accumulator.
    old_dy : np.float32
        Previous iteration y displacement accumulator.
    freeze : np.float32
        Gephi freeze-balance damping state.
    """

    x: np.float32
    y: np.float32
    size: float
    fixed: bool = False
    dx: np.float32 = np.float32(0.0)
    dy: np.float32 = np.float32(0.0)
    old_dx: np.float32 = np.float32(0.0)
    old_dy: np.float32 = np.float32(0.0)
    freeze: np.float32 = np.float32(0.0)


@dataclass(frozen=True)
class ForceAtlas1Config:
    """Configuration for Gephi ForceAtlas1.

    Attributes
    ----------
    steps : int
        Number of ForceAtlas1 iterations.
    attraction_strength : float
        Gephi attraction strength.
    repulsion_strength : float
        Gephi repulsion strength.
    inertia : float
        Fraction of previous displacement retained at the start of a step.
    outbound_attraction_distribution : bool
        Whether edge attraction is divided by ``1 + source degree``.
    adjust_sizes : bool
        Whether Gephi anti-collision force variants are used.
    freeze_balance : bool
        Whether Gephi freeze-balance damping is active.
    freeze_strength : float
        Freeze-balance strength.
    freeze_inertia : float
        Freeze state inertia.
    gravity : float
        Gephi gravity setting.
    speed : float
        Gephi speed setting.
    cooling : float
        Gephi cooling divisor.
    max_displacement : float
        Maximum per-iteration displacement length before damping.
    fidelity_mode : bool
        Whether to use the source-faithful float-backed reference port.
    fidelity_dtype : torch.dtype, optional
        Output dtype override for fidelity mode.
    """

    steps: int = 100
    attraction_strength: float = 10.0
    repulsion_strength: float = 200.0
    inertia: float = 0.1
    outbound_attraction_distribution: bool = False
    adjust_sizes: bool = False
    freeze_balance: bool = True
    freeze_strength: float = 80.0
    freeze_inertia: float = 0.2
    gravity: float = 30.0
    speed: float = 1.0
    cooling: float = 1.0
    max_displacement: float = 10.0
    fidelity_mode: bool = True
    fidelity_dtype: Optional[torch.dtype] = None


def _edge_arrays(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
) -> tuple[np.ndarray, np.ndarray]:
    """Convert graph edges and weights to reference-order NumPy arrays.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    tuple[np.ndarray, np.ndarray]
        Edge endpoint array with shape ``[E, 2]`` and float64 edge weights.
    """
    edges = edge_index.detach().cpu().numpy().T.astype(np.int64, copy=True)
    if edge_weights is None:
        weights = np.ones(edges.shape[0], dtype=np.float64)
    else:
        weights = edge_weights.detach().cpu().numpy().astype(np.float64, copy=True)
    return edges, weights


def _node_sizes_for_forceatlas1(
    node_sizes: Optional[torch.Tensor],
    num_nodes: int,
) -> np.ndarray:
    """Resolve Gephi-like scalar node sizes for anti-collision mode.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Optional Dagua node-size tensor with shape ``[N, 2]`` or scalar sizes
        with shape ``[N]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    numpy.ndarray
        Float64 scalar sizes with shape ``[N]``.
    """
    if node_sizes is None:
        return np.zeros(num_nodes, dtype=np.float64)
    sizes = node_sizes.detach().cpu().to(dtype=torch.float64)
    if sizes.ndim == 1:
        return sizes.numpy().astype(np.float64, copy=True)
    if sizes.ndim == 2 and sizes.shape[0] == num_nodes:
        return torch.amax(sizes, dim=1).numpy().astype(np.float64, copy=True)
    raise ValueError("node_sizes must have shape [N] or [N, 2].")


def _forceatlas1_degrees(edges: np.ndarray, num_nodes: int) -> np.ndarray:
    """Compute Gephi-style total degrees for ForceAtlas1 forces.

    Parameters
    ----------
    edges : numpy.ndarray
        Edge endpoint array with shape ``[E, 2]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    numpy.ndarray
        Float64 total degree vector with shape ``[N]``.
    """
    degree = np.zeros(num_nodes, dtype=np.float64)
    for source, target in edges:
        if source == target:
            degree[int(source)] += 2.0
            continue
        degree[int(source)] += 1.0
        degree[int(target)] += 1.0
    return degree


def _initial_forceatlas1_nodes(
    num_nodes: int,
    seed: int,
    node_sizes: Optional[torch.Tensor],
) -> list[_ForceAtlas1Node]:
    """Create ForceAtlas1 nodes using Gephi's all-zero initialization path.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    seed : int
        Java random seed.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N]`` or ``[N, 2]``.

    Returns
    -------
    list[_ForceAtlas1Node]
        Mutable node states in input node order.
    """
    sizes = _node_sizes_for_forceatlas1(node_sizes, num_nodes)
    rng = _JavaRandom(seed)
    nodes: list[_ForceAtlas1Node] = []
    for node_index in range(num_nodes):
        x_coord = np.float32((_GEphi_INIT_OFFSET + rng.next_double()) * _GEphi_INIT_SCALE)
        y_coord = np.float32((_GEphi_INIT_OFFSET + rng.next_double()) * _GEphi_INIT_SCALE)
        nodes.append(
            _ForceAtlas1Node(
                x=np.float32(float(x_coord) - _GEphi_INIT_SHIFT),
                y=np.float32(float(y_coord) - _GEphi_INIT_SHIFT),
                size=float(sizes[node_index]),
            )
        )
    return nodes


def _apply_forceatlas1_repulsion(
    nodes: list[_ForceAtlas1Node],
    degree: np.ndarray,
    config: ForceAtlas1Config,
) -> None:
    """Apply Gephi ForceAtlas1 pairwise repulsion.

    Parameters
    ----------
    nodes : list[_ForceAtlas1Node]
        Mutable node states in graph iteration order.
    degree : numpy.ndarray
        Total degree vector with shape ``[N]``.
    config : ForceAtlas1Config
        ForceAtlas1 configuration.

    Returns
    -------
    None
        Updates node displacement accumulators in place.
    """
    for source_index, source_node in enumerate(nodes):
        for target_index, target_node in enumerate(nodes):
            if source_index == target_index:
                continue
            coefficient = (
                config.repulsion_strength
                * (1.0 + float(degree[source_index]))
                * (1.0 + float(degree[target_index]))
            )
            x_dist = float(source_node.x) - float(target_node.x)
            y_dist = float(source_node.y) - float(target_node.y)
            euclidean = math.sqrt((x_dist * x_dist) + (y_dist * y_dist))
            distance = euclidean
            if config.adjust_sizes:
                distance = euclidean - source_node.size - target_node.size
            if distance > 0.0:
                force = _REPULSION_FACTOR * coefficient / distance
                source_node.dx = np.float32(float(source_node.dx) + (x_dist / distance) * force)
                source_node.dy = np.float32(float(source_node.dy) + (y_dist / distance) * force)
                target_node.dx = np.float32(float(target_node.dx) - (x_dist / distance) * force)
                target_node.dy = np.float32(float(target_node.dy) - (y_dist / distance) * force)
            elif config.adjust_sizes and distance != 0.0:
                force = -coefficient
                source_node.dx = np.float32(float(source_node.dx) + (x_dist / distance) * force)
                source_node.dy = np.float32(float(source_node.dy) + (y_dist / distance) * force)
                target_node.dx = np.float32(float(target_node.dx) - (x_dist / distance) * force)
                target_node.dy = np.float32(float(target_node.dy) - (y_dist / distance) * force)


def _apply_forceatlas1_attraction(
    nodes: list[_ForceAtlas1Node],
    edges: np.ndarray,
    weights: np.ndarray,
    degree: np.ndarray,
    config: ForceAtlas1Config,
) -> None:
    """Apply Gephi ForceAtlas1 edge attraction.

    Parameters
    ----------
    nodes : list[_ForceAtlas1Node]
        Mutable node states in graph iteration order.
    edges : numpy.ndarray
        Edge endpoint array with shape ``[E, 2]``.
    weights : numpy.ndarray
        Edge weights with shape ``[E]``.
    degree : numpy.ndarray
        Total degree vector with shape ``[N]``.
    config : ForceAtlas1Config
        ForceAtlas1 configuration.

    Returns
    -------
    None
        Updates node displacement accumulators in place.
    """
    for edge_offset, (source, target) in enumerate(edges):
        source_index = int(source)
        target_index = int(target)
        if source_index == target_index:
            continue
        source_node = nodes[source_index]
        target_node = nodes[target_index]
        bonus = _FIXED_NODE_ATTRACTION_BONUS if source_node.fixed or target_node.fixed else 1.0
        coefficient = bonus * float(weights[edge_offset]) * config.attraction_strength
        if config.outbound_attraction_distribution:
            coefficient /= 1.0 + float(degree[source_index])

        x_dist = float(source_node.x) - float(target_node.x)
        y_dist = float(source_node.y) - float(target_node.y)
        euclidean = math.sqrt((x_dist * x_dist) + (y_dist * y_dist))
        distance = euclidean
        if config.adjust_sizes:
            distance = euclidean - source_node.size - target_node.size
        if distance <= 0.0:
            continue
        force = -_ATTRACTION_FACTOR * coefficient * distance
        source_node.dx = np.float32(float(source_node.dx) + (x_dist / distance) * force)
        source_node.dy = np.float32(float(source_node.dy) + (y_dist / distance) * force)
        target_node.dx = np.float32(float(target_node.dx) - (x_dist / distance) * force)
        target_node.dy = np.float32(float(target_node.dy) - (y_dist / distance) * force)


def _apply_forceatlas1_gravity(
    nodes: list[_ForceAtlas1Node],
    gravity: float,
) -> None:
    """Apply Gephi ForceAtlas1 origin gravity.

    Parameters
    ----------
    nodes : list[_ForceAtlas1Node]
        Mutable node states in graph iteration order.
    gravity : float
        Gephi gravity setting.

    Returns
    -------
    None
        Updates node displacement accumulators in place.
    """
    for node in nodes:
        x_coord = float(node.x)
        y_coord = float(node.y)
        distance = _GRAVITY_EPSILON + math.sqrt((x_coord * x_coord) + (y_coord * y_coord))
        gravity_force = _GRAVITY_FACTOR * gravity * distance
        node.dx = np.float32(float(node.dx) - (gravity_force * x_coord / distance))
        node.dy = np.float32(float(node.dy) - (gravity_force * y_coord / distance))


def _apply_forceatlas1_speed_and_displacement(
    nodes: list[_ForceAtlas1Node],
    config: ForceAtlas1Config,
) -> None:
    """Apply Gephi ForceAtlas1 speed, freeze, cooling, and movement.

    Parameters
    ----------
    nodes : list[_ForceAtlas1Node]
        Mutable node states in graph iteration order.
    config : ForceAtlas1Config
        ForceAtlas1 configuration.

    Returns
    -------
    None
        Updates node coordinates and displacement accumulators in place.
    """
    speed_factor = config.speed * (_FREEZE_SPEED_MULTIPLIER if config.freeze_balance else 1.0)
    for node in nodes:
        node.dx = np.float32(float(node.dx) * speed_factor)
        node.dy = np.float32(float(node.dy) * speed_factor)

    for node in nodes:
        if node.fixed:
            continue
        displacement = _GRAVITY_EPSILON + math.sqrt(
            (float(node.dx) * float(node.dx)) + (float(node.dy) * float(node.dy))
        )
        if config.freeze_balance:
            old_delta_x = float(node.old_dx) - float(node.dx)
            old_delta_y = float(node.old_dy) - float(node.dy)
            freeze_target = (
                0.1
                * config.freeze_strength
                * math.sqrt(math.sqrt((old_delta_x * old_delta_x) + (old_delta_y * old_delta_y)))
            )
            node.freeze = np.float32(
                (config.freeze_inertia * float(node.freeze))
                + ((1.0 - config.freeze_inertia) * freeze_target)
            )
            ratio = min(
                displacement / (displacement * (1.0 + float(node.freeze))),
                config.max_displacement / displacement,
            )
        else:
            ratio = min(1.0, config.max_displacement / displacement)
        node.dx = np.float32(float(node.dx) * ratio / config.cooling)
        node.dy = np.float32(float(node.dy) * ratio / config.cooling)
        node.x = np.float32(float(node.x) + float(node.dx))
        node.y = np.float32(float(node.y) + float(node.dy))


def _layout_forceatlas1_source_port(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    node_sizes: Optional[torch.Tensor],
    steps: int,
    seed: int,
    edge_weights: Optional[torch.Tensor],
    config: ForceAtlas1Config,
    output_dtype: torch.dtype,
) -> torch.Tensor:
    """Run the source-faithful Gephi ForceAtlas1 port.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N]`` or ``[N, 2]``.
    steps : int
        Number of iterations.
    seed : int
        Java random seed for the fixed initial placement.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    config : ForceAtlas1Config
        ForceAtlas1 parameters.
    output_dtype : torch.dtype
        Output dtype for returned positions.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=output_dtype, device=edge_index.device)
    edges, weights = _edge_arrays(edge_index, edge_weights)
    if edge_weights is not None and weights.shape[0] != edges.shape[0]:
        raise ValueError("edge_weights must have shape [E].")
    degree = _forceatlas1_degrees(edges, num_nodes)
    nodes = _initial_forceatlas1_nodes(num_nodes, seed, node_sizes)

    for _ in range(steps):
        for node in nodes:
            node.old_dx = node.dx
            node.old_dy = node.dy
            node.dx = np.float32(float(node.dx) * config.inertia)
            node.dy = np.float32(float(node.dy) * config.inertia)
        _apply_forceatlas1_repulsion(nodes, degree, config)
        _apply_forceatlas1_attraction(nodes, edges, weights, degree, config)
        _apply_forceatlas1_gravity(nodes, config.gravity)
        _apply_forceatlas1_speed_and_displacement(nodes, config)

    pos = np.asarray([(float(node.x), float(node.y)) for node in nodes], dtype=np.float32)
    return torch.from_numpy(pos).to(device=edge_index.device, dtype=output_dtype)


@dataclass(frozen=True)
class ForceAtlas1Solve(Op):
    """Single ForceAtlas1 solve op wrapping the source-faithful loop.

    Parameters
    ----------
    config : ForceAtlas1Config
        Resolved ForceAtlas1 configuration.
    dtype : torch.dtype
        Output dtype.
    """

    config: ForceAtlas1Config
    dtype: torch.dtype

    name: str = "forceatlas1_solve"

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run ForceAtlas1 and store final coordinates in state.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state to receive ``pos``.
        ctx : RuntimeContext
            Execution context. The configured device is implicit in the input
            edge tensor, so this op does not otherwise inspect the context.

        Returns
        -------
        SolveState
            State with ``pos`` populated by ForceAtlas1.
        """
        del ctx
        state.pos = _layout_forceatlas1_source_port(
            problem.edge_index,
            problem.num_nodes,
            node_sizes=problem.node_sizes,
            steps=self.config.steps,
            seed=problem.seed,
            edge_weights=problem.edge_weights,
            config=self.config,
            output_dtype=self.dtype,
        )
        return state


def build_forceatlas1_pipeline(config: Optional[ForceAtlas1Config] = None) -> Pipeline:
    """Build the Gephi ForceAtlas1 pipeline.

    Parameters
    ----------
    config : ForceAtlas1Config, optional
        ForceAtlas1 parameters. Defaults match Gephi's ``resetPropertiesValues``.

    Returns
    -------
    Pipeline
        Pipeline containing the source-faithful ForceAtlas1 solve op.

    Raises
    ------
    ValueError
        If numeric parameters are outside the supported Gephi domain.
    """
    resolved = config or ForceAtlas1Config()
    if resolved.steps < 0:
        raise ValueError("steps must be non-negative.")
    if resolved.cooling == 0.0:
        raise ValueError("cooling must be non-zero.")
    if resolved.max_displacement <= 0.0:
        raise ValueError("max_displacement must be positive.")
    dtype = resolve_fidelity_dtype(resolved.fidelity_mode, resolved.fidelity_dtype)
    return Pipeline(
        [ForceAtlas1Solve(config=resolved, dtype=dtype)],
        name="forceatlas1_pipeline",
    )


def layout_forceatlas1_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 100,
    seed: int = 42,
    attraction_strength: float = 10.0,
    repulsion_strength: float = 200.0,
    inertia: float = 0.1,
    outbound_attraction_distribution: bool = False,
    outboundAttractionDistribution: Optional[bool] = None,
    adjust_sizes: bool = False,
    adjustSizes: Optional[bool] = None,
    freeze_balance: bool = True,
    freezeBalance: Optional[bool] = None,
    freeze_strength: float = 80.0,
    freeze_inertia: float = 0.2,
    gravity: float = 30.0,
    speed: float = 1.0,
    cooling: float = 1.0,
    max_displacement: float = 10.0,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_mode: bool = True,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the Gephi ForceAtlas1 pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node sizes with shape ``[N]`` or ``[N, 2]`` used only when
        ``adjust_sizes`` is enabled.
    steps : int, default=100
        Number of ForceAtlas1 iterations.
    seed : int, default=42
        Java-compatible initialization seed.
    attraction_strength : float, default=10.0
        Gephi attraction strength.
    repulsion_strength : float, default=200.0
        Gephi repulsion strength.
    inertia : float, default=0.1
        Fraction of previous displacement retained at each iteration start.
    outbound_attraction_distribution : bool, default=False
        Whether attraction is divided by ``1 + source degree``.
    outboundAttractionDistribution : bool, optional
        Java-style alias for ``outbound_attraction_distribution``.
    adjust_sizes : bool, default=False
        Whether anti-collision force variants are enabled.
    adjustSizes : bool, optional
        Java-style alias for ``adjust_sizes``.
    freeze_balance : bool, default=True
        Whether freeze-balance damping is enabled.
    freezeBalance : bool, optional
        Java-style alias for ``freeze_balance``.
    freeze_strength : float, default=80.0
        Freeze-balance strength.
    freeze_inertia : float, default=0.2
        Freeze-balance inertia.
    gravity : float, default=30.0
        Gephi gravity value.
    speed : float, default=1.0
        Gephi speed value.
    cooling : float, default=1.0
        Gephi cooling divisor.
    max_displacement : float, default=10.0
        Maximum movement magnitude before damping.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    fidelity_mode : bool, default=True
        Retained for consistency with other pipelines. ForceAtlas1 currently
        uses the source-faithful loop for both modes.
    fidelity_dtype : torch.dtype, optional
        Output dtype override. Defaults to float64 when ``fidelity_mode`` is
        true and float32 otherwise.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.
    """
    if outboundAttractionDistribution is not None:
        outbound_attraction_distribution = outboundAttractionDistribution
    if adjustSizes is not None:
        adjust_sizes = adjustSizes
    if freezeBalance is not None:
        freeze_balance = freezeBalance
    resolved_dtype = resolve_fidelity_dtype(fidelity_mode, fidelity_dtype)
    config = ForceAtlas1Config(
        steps=steps,
        attraction_strength=attraction_strength,
        repulsion_strength=repulsion_strength,
        inertia=inertia,
        outbound_attraction_distribution=outbound_attraction_distribution,
        adjust_sizes=adjust_sizes,
        freeze_balance=freeze_balance,
        freeze_strength=freeze_strength,
        freeze_inertia=freeze_inertia,
        gravity=gravity,
        speed=speed,
        cooling=cooling,
        max_displacement=max_displacement,
        fidelity_mode=fidelity_mode,
        fidelity_dtype=resolved_dtype,
    )
    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device)))
    final_state = build_forceatlas1_pipeline(config=config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("ForceAtlas1 pipeline did not produce final positions.")
    return final_state.pos


__all__ = [
    "ForceAtlas1Config",
    "ForceAtlas1Solve",
    "build_forceatlas1_pipeline",
    "layout_forceatlas1_pipeline",
]
