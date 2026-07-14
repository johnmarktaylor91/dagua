"""JUNG ISOM self-organizing-map layout pipeline."""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass
from typing import Optional

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory

_JAVA_RANDOM_MULTIPLIER = 0x5DEECE66D
_JAVA_RANDOM_ADDEND = 0xB
_JAVA_RANDOM_MASK = (1 << 48) - 1
_JAVA_RANDOM_DOUBLE_SCALE = float(1 << 53)
_DEFAULT_WIDTH = 600.0
_DEFAULT_HEIGHT = 600.0
_DEFAULT_RANDOM_MARGIN = 10.0
_DEFAULT_INITIAL_ADAPTION = 0.9
_DEFAULT_COOLING_FACTOR = 2.0
_DEFAULT_RADIUS = 5
_DEFAULT_MIN_RADIUS = 1
_DEFAULT_RADIUS_CONSTANT_TIME = 100
_DEFAULT_MAX_EPOCH = 2000


class JavaRandom:
    """Minimal ``java.util.Random`` port for JUNG-compatible ISOM runs."""

    def __init__(self, seed: int) -> None:
        """Initialize the Java-compatible linear congruential generator.

        Parameters
        ----------
        seed : int
            Signed or unsigned seed value.

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


@dataclass(frozen=True)
class IsomConfig:
    """Configuration for the JUNG ISOM source port.

    Attributes
    ----------
    steps : int
        Number of JUNG ``step()`` calls to execute.
    width : float
        Layout region width used by JUNG's random initializer.
    height : float
        Layout region height used by JUNG's random initializer.
    random_margin : float
        JUNG's random training point offset added to both axes.
    max_epoch : int
        Epoch at which JUNG stops updating.
    initial_adaption : float
        Initial Kohonen adaptation value.
    min_adaption : float
        Lower bound for the adaptation value.
    cooling_factor : float
        Exponential cooling multiplier.
    radius : int
        Initial graph-distance neighborhood radius.
    min_radius : int
        Minimum graph-distance neighborhood radius.
    radius_constant_time : int
        Epoch interval at which the radius decays by one.
    fidelity_dtype : torch.dtype | None
        Output dtype override for fidelity tests.
    """

    steps: int = _DEFAULT_MAX_EPOCH
    width: float = _DEFAULT_WIDTH
    height: float = _DEFAULT_HEIGHT
    random_margin: float = _DEFAULT_RANDOM_MARGIN
    max_epoch: int = _DEFAULT_MAX_EPOCH
    initial_adaption: float = _DEFAULT_INITIAL_ADAPTION
    min_adaption: float = 0.0
    cooling_factor: float = _DEFAULT_COOLING_FACTOR
    radius: int = _DEFAULT_RADIUS
    min_radius: int = _DEFAULT_MIN_RADIUS
    radius_constant_time: int = _DEFAULT_RADIUS_CONSTANT_TIME
    fidelity_dtype: Optional[torch.dtype] = None


def _validate_isom_config(config: IsomConfig) -> None:
    """Validate ISOM scalar configuration.

    Parameters
    ----------
    config : IsomConfig
        Configuration to validate.

    Returns
    -------
    None
        Raises on invalid values.

    Raises
    ------
    ValueError
        If a scalar option is outside the source-compatible domain.
    """
    if config.steps < 0:
        raise ValueError("steps must be non-negative.")
    if config.width <= 0.0 or config.height <= 0.0:
        raise ValueError("width and height must be positive.")
    if config.max_epoch < 0:
        raise ValueError("max_epoch must be non-negative.")
    if config.initial_adaption < 0.0 or config.min_adaption < 0.0:
        raise ValueError("adaption values must be non-negative.")
    if config.radius < 0 or config.min_radius < 0:
        raise ValueError("radius values must be non-negative.")
    if config.radius_constant_time <= 0:
        raise ValueError("radius_constant_time must be positive.")


def _validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Validate and return a CPU ``long`` edge-index tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N``.

    Returns
    -------
    torch.Tensor
        CPU edge-index tensor with shape ``[2, E]`` and dtype ``torch.long``.

    Raises
    ------
    ValueError
        If the tensor shape, dtype, or endpoint range is invalid.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("edge_index must use an integer dtype.")

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() == 0:
        return edge_index_cpu
    if int(edge_index_cpu.min().item()) < 0:
        raise ValueError("edge_index cannot contain negative node indices.")
    if int(edge_index_cpu.max().item()) >= num_nodes:
        raise ValueError("edge_index contains node indices outside num_nodes.")
    return edge_index_cpu


def _build_isom_neighbors(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build JUNG-style undirected neighbor lists in edge insertion order.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N``.

    Returns
    -------
    list[list[int]]
        Per-node neighbor indices. Each list preserves first-seen edge order.
    """
    neighbors: list[list[int]] = [[] for _ in range(num_nodes)]
    seen: list[set[int]] = [set() for _ in range(num_nodes)]
    for edge_pos in range(edge_index.shape[1]):
        source = int(edge_index[0, edge_pos].item())
        target = int(edge_index[1, edge_pos].item())
        if source == target:
            continue
        if target not in seen[source]:
            neighbors[source].append(target)
            seen[source].add(target)
        if source not in seen[target]:
            neighbors[target].append(source)
            seen[target].add(source)
    return neighbors


def _initial_isom_positions(
    num_nodes: int,
    width: float,
    height: float,
    seed: int,
) -> list[list[float]]:
    """Generate JUNG ``RandomLocationTransformer`` initial positions.

    Parameters
    ----------
    num_nodes : int
        Number of nodes ``N``.
    width : float
        Initialization width.
    height : float
        Initialization height.
    seed : int
        Java ``Random`` seed.

    Returns
    -------
    list[list[float]]
        Mutable coordinate pairs with shape ``[N, 2]``.
    """
    rng = JavaRandom(seed)
    return [[rng.next_double() * width, rng.next_double() * height] for _ in range(num_nodes)]


def _nearest_isom_node(positions: list[list[float]], target_x: float, target_y: float) -> int:
    """Return the first node nearest to the target point.

    Parameters
    ----------
    positions : list[list[float]]
        Mutable coordinate pairs with shape ``[N, 2]``.
    target_x : float
        Target x-coordinate.
    target_y : float
        Target y-coordinate.

    Returns
    -------
    int
        Index of the nearest node, with first-in-iteration tie behavior.
    """
    closest = 0
    min_distance = math.inf
    for node, (x_coord, y_coord) in enumerate(positions):
        dx = x_coord - target_x
        dy = y_coord - target_y
        distance = (dx * dx) + (dy * dy)
        if distance < min_distance:
            min_distance = distance
            closest = node
    return closest


def _adjust_isom_neighborhood(
    positions: list[list[float]],
    neighbors: list[list[int]],
    winner: int,
    target_x: float,
    target_y: float,
    adaption: float,
    radius: int,
) -> None:
    """Pull a winner and its graph-radius neighborhood toward a point.

    Parameters
    ----------
    positions : list[list[float]]
        Mutable coordinate pairs with shape ``[N, 2]``.
    neighbors : list[list[int]]
        Undirected graph-neighbor lists.
    winner : int
        Nearest node to the random point.
    target_x : float
        Random point x-coordinate.
    target_y : float
        Random point y-coordinate.
    adaption : float
        Current JUNG adaptation value.
    radius : int
        Current graph-distance adaptation radius.

    Returns
    -------
    None
        Mutates ``positions`` in place.
    """
    visited = [False] * len(positions)
    distances = [0] * len(positions)
    queue: deque[int] = deque([winner])
    visited[winner] = True

    while queue:
        current = queue.popleft()
        current_distance = distances[current]
        factor = adaption / math.pow(2.0, current_distance)
        current_pos = positions[current]
        current_pos[0] += factor * (target_x - current_pos[0])
        current_pos[1] += factor * (target_y - current_pos[1])

        if current_distance < radius:
            for child in neighbors[current]:
                if not visited[child]:
                    visited[child] = True
                    distances[child] = current_distance + 1
                    queue.append(child)


def _run_isom_epochs(
    positions: list[list[float]],
    neighbors: list[list[int]],
    seed: int,
    config: IsomConfig,
) -> None:
    """Run JUNG ISOM epochs in place.

    Parameters
    ----------
    positions : list[list[float]]
        Mutable coordinate pairs with shape ``[N, 2]``.
    neighbors : list[list[int]]
        Undirected graph-neighbor lists.
    seed : int
        Java ``Random`` seed for deterministic training points.
    config : IsomConfig
        Source-compatible algorithm configuration.

    Returns
    -------
    None
        Mutates ``positions`` in place.
    """
    if not positions:
        return

    epoch = 1
    adaption = config.initial_adaption
    radius = config.radius
    training_rng = JavaRandom(seed)

    for _step in range(config.steps):
        if epoch < config.max_epoch:
            target_x = config.random_margin + (training_rng.next_double() * config.width)
            target_y = config.random_margin + (training_rng.next_double() * config.height)
            winner = _nearest_isom_node(positions, target_x, target_y)
            _adjust_isom_neighborhood(
                positions=positions,
                neighbors=neighbors,
                winner=winner,
                target_x=target_x,
                target_y=target_y,
                adaption=adaption,
                radius=radius,
            )
            epoch += 1
            factor = math.exp(-1.0 * config.cooling_factor * (float(epoch) / config.max_epoch))
            adaption = max(config.min_adaption, factor * config.initial_adaption)
            if radius > config.min_radius and epoch % config.radius_constant_time == 0:
                radius -= 1


class BuildIsomGraph(Op):
    """Validate ISOM inputs and build source-order graph neighborhoods."""

    name = "build_isom_graph"
    category = OpCategory.PREPROCESS
    reads = ("edge_index",)
    writes = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build the graph neighborhood cache.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs with ``edge_index`` shape ``[2, E]``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context; unused because ISOM fidelity runs on CPU.

        Returns
        -------
        SolveState
            State with ``isom_neighbors`` and ``isom_edge_index`` extras.
        """
        del ctx
        edge_index = _validate_edge_index(problem.edge_index, problem.num_nodes)
        state.extras["isom_edge_index"] = edge_index
        state.extras["isom_neighbors"] = _build_isom_neighbors(edge_index, problem.num_nodes)
        return state


class InitIsomPositions(Op):
    """Initialize positions with JUNG's seeded random transformer."""

    name = "init_isom_positions"
    category = OpCategory.INIT
    reads = ("extras",)
    writes = ("extras",)

    def __init__(self, config: IsomConfig) -> None:
        """Store the ISOM configuration.

        Parameters
        ----------
        config : IsomConfig
            Source-compatible algorithm configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate mutable source-port positions.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs containing ``num_nodes`` and ``seed``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context; unused because ISOM fidelity runs on CPU.

        Returns
        -------
        SolveState
            State with ``isom_positions`` in ``extras``.
        """
        del ctx
        state.extras["isom_positions"] = _initial_isom_positions(
            num_nodes=problem.num_nodes,
            width=self.config.width,
            height=self.config.height,
            seed=problem.seed,
        )
        return state


class RunIsomEpochs(Op):
    """Run source-compatible JUNG ISOM training epochs."""

    name = "run_isom_epochs"
    category = OpCategory.OPTIMIZE
    reads = ("extras",)
    writes = ("extras",)

    def __init__(self, config: IsomConfig) -> None:
        """Store the ISOM configuration.

        Parameters
        ----------
        config : IsomConfig
            Source-compatible algorithm configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the configured number of epochs.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs containing the seed.
        state : SolveState
            Mutable solve state with ISOM positions and neighbors.
        ctx : RuntimeContext
            Execution context; unused because ISOM fidelity runs on CPU.

        Returns
        -------
        SolveState
            State with updated ``isom_positions``.

        Raises
        ------
        RuntimeError
            If prior ISOM initialization ops have not run.
        """
        del ctx
        positions = state.extras.get("isom_positions")
        neighbors = state.extras.get("isom_neighbors")
        if positions is None or neighbors is None:
            raise RuntimeError("ISOM positions and neighbors must be initialized before epochs.")
        _run_isom_epochs(
            positions=positions,
            neighbors=neighbors,
            seed=problem.seed,
            config=self.config,
        )
        return state


class FinalizeIsomPositions(Op):
    """Convert source-port ISOM coordinates into the public tensor output."""

    name = "finalize_isom_positions"
    category = OpCategory.POSTPROCESS
    reads = ("extras",)
    writes = ("pos",)

    def __init__(self, dtype: torch.dtype) -> None:
        """Store the output dtype.

        Parameters
        ----------
        dtype : torch.dtype
            Output tensor dtype.

        Returns
        -------
        None
            The op stores the dtype.
        """
        self.dtype = dtype

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Finalize positions.

        Parameters
        ----------
        problem : LayoutProblem
            Layout inputs containing ``num_nodes``.
        state : SolveState
            Mutable solve state with ``isom_positions``.
        ctx : RuntimeContext
            Execution context whose plan provides the output device.

        Returns
        -------
        SolveState
            State with ``pos`` tensor of shape ``[N, 2]``.

        Raises
        ------
        RuntimeError
            If ISOM positions are missing.
        """
        positions = state.extras.get("isom_positions")
        if positions is None:
            raise RuntimeError("ISOM positions were not initialized.")
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=self.dtype, device=ctx.plan.device)
        else:
            state.pos = torch.tensor(positions, dtype=self.dtype, device=ctx.plan.device)
        return state


def build_isom_pipeline(config: Optional[IsomConfig] = None) -> Pipeline:
    """Build the JUNG ISOM source-port pipeline.

    Parameters
    ----------
    config : IsomConfig, optional
        Source-compatible configuration. ``None`` uses JUNG defaults.

    Returns
    -------
    Pipeline
        Pipeline that validates input, initializes seeded locations, runs SOM
        epochs, and emits final coordinates.
    """
    resolved_config = config if config is not None else IsomConfig()
    _validate_isom_config(resolved_config)
    dtype = resolve_fidelity_dtype(True, resolved_config.fidelity_dtype)
    return Pipeline(
        [
            BuildIsomGraph(),
            InitIsomPositions(config=resolved_config),
            RunIsomEpochs(config=resolved_config),
            FinalizeIsomPositions(dtype=dtype),
        ],
        name="isom_pipeline",
    )


def layout_isom_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = _DEFAULT_MAX_EPOCH,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    fidelity_dtype: Optional[torch.dtype] = None,
    width: float = _DEFAULT_WIDTH,
    height: float = _DEFAULT_HEIGHT,
    random_margin: float = _DEFAULT_RANDOM_MARGIN,
    max_epoch: int = _DEFAULT_MAX_EPOCH,
    initial_adaption: float = _DEFAULT_INITIAL_ADAPTION,
    min_adaption: float = 0.0,
    cooling_factor: float = _DEFAULT_COOLING_FACTOR,
    radius: int = _DEFAULT_RADIUS,
    min_radius: int = _DEFAULT_MIN_RADIUS,
    radius_constant_time: int = _DEFAULT_RADIUS_CONSTANT_TIME,
    direction: str = "TB",
) -> torch.Tensor:
    """Run the JUNG ISOM source-port layout pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Accepted for dispatch compatibility; ISOM ignores node sizes.
    steps : int, default=2000
        Number of JUNG ``step()`` calls to execute.
    seed : int, default=42
        Java ``Random`` seed for initialization and deterministic training points.
    edge_weights : torch.Tensor, optional
        Accepted for dispatch compatibility; ISOM ignores edge weights.
    fidelity_dtype : torch.dtype, optional
        Output dtype override.
    width : float, default=600.0
        JUNG layout region width.
    height : float, default=600.0
        JUNG layout region height.
    random_margin : float, default=10.0
        Offset applied to each random training point coordinate.
    max_epoch : int, default=2000
        Epoch at which JUNG stops updating positions.
    initial_adaption : float, default=0.9
        Initial adaptation value.
    min_adaption : float, default=0.0
        Lower bound for adaptation.
    cooling_factor : float, default=2.0
        Exponential cooling multiplier.
    radius : int, default=5
        Initial graph-distance neighborhood radius.
    min_radius : int, default=1
        Minimum graph-distance neighborhood radius.
    radius_constant_time : int, default=100
        Epoch interval for radius decay.
    direction : str, default="TB"
        Accepted for dispatch compatibility; ISOM uses screen-space axes.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If inputs or scalar parameters are invalid.
    RuntimeError
        If the pipeline fails to produce positions.
    """
    del node_sizes, edge_weights, direction
    config = IsomConfig(
        steps=steps,
        width=width,
        height=height,
        random_margin=random_margin,
        max_epoch=max_epoch,
        initial_adaption=initial_adaption,
        min_adaption=min_adaption,
        cooling_factor=cooling_factor,
        radius=radius,
        min_radius=min_radius,
        radius_constant_time=radius_constant_time,
        fidelity_dtype=fidelity_dtype,
    )
    device = edge_index.device if edge_index.device.type != "meta" else torch.device("cpu")
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, seed=seed)
    final_state = build_isom_pipeline(config=config).apply(
        problem,
        SolveState(),
        RuntimeContext(plan=ExecutionPlan(device=str(device))),
    )
    if final_state.pos is None:
        raise RuntimeError("ISOM pipeline did not produce final positions.")
    return final_state.pos


__all__ = [
    "IsomConfig",
    "JavaRandom",
    "build_isom_pipeline",
    "layout_isom_pipeline",
]
