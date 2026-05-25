"""Initialization operations for composable layout pipelines."""

from __future__ import annotations

import importlib
import inspect
import random
from collections import defaultdict
from dataclasses import dataclass, field
from math import cos, pi, sin, sqrt
from typing import Any, Callable, ClassVar, DefaultDict, Dict, List, Optional, Tuple

import numpy as np
import torch
from scipy import sparse
from scipy.sparse import linalg as sparse_linalg

from dagua.layout.graph_classify import GraphFamily, GraphStructure
from dagua.layout.init_placement import init_positions
from dagua.layout.layers import LayerIndex, build_layer_index
from dagua.layout.ops._igraph_rng import make_igraph_default_rng
from dagua.layout.ops.base import Op
from dagua.layout.ops.distance import (
    AllPairsShortestPaths,
    AllPairsShortestPathsConfig,
    PivotDistanceQueries,
    PivotSelection,
    PivotSelectionConfig,
)
from dagua.layout.ops.graph_utils import layout_device as _layout_device
from dagua.layout.ops.graph_utils import rescale_layout as _rescale_layout
from dagua.layout.ops.preprocess import BuildAdjacency, BuildAdjacencyConfig
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op
from dagua.utils import longest_path_layering

_SPECTRAL_EIGEN_TOLERANCE = 1.0e-9
GRAPHOPT_INITIAL_POS_KEY = "graphopt_initial_pos"


def _target_device(problem: LayoutProblem, ctx: RuntimeContext) -> torch.device:
    """Resolve the output device for initialization ops.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    ctx : RuntimeContext
        Execution infrastructure, optionally carrying a requested device.

    Returns
    -------
    torch.device
        Device for the initialized position tensor.
    """
    if ctx.plan.device:
        return torch.device(ctx.plan.device)
    if problem.edge_index.numel() > 0:
        return problem.edge_index.device
    if problem.node_sizes is not None:
        return problem.node_sizes.device
    return torch.device("cpu")


def _scale_factor(num_nodes: int, rule: str) -> float:
    """Compute the multiplicative factor for scaled initializations.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    rule : str
        Scaling rule name.

    Returns
    -------
    float
        Multiplicative scale factor.

    Raises
    ------
    ValueError
        If ``rule`` is unsupported.
    """
    if rule in {"none", "unit"}:
        return 1.0
    if rule == "sqrt_n":
        return sqrt(float(max(num_nodes, 1)))
    raise ValueError(f"Unsupported init scale: {rule}")


def _torch_generator(problem: LayoutProblem, ctx: RuntimeContext) -> torch.Generator:
    """Resolve the CPU generator used by torch-backed random initializers.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs containing the fallback seed.
    ctx : RuntimeContext
        Execution infrastructure, optionally carrying a shared generator.

    Returns
    -------
    torch.Generator
        CPU generator seeded reproducibly.
    """
    if ctx.generator is not None:
        return ctx.generator
    generator = torch.Generator(device="cpu")
    generator.manual_seed(problem.seed)
    return generator


def _maybe_set_empty_or_single_positions(
    problem: LayoutProblem,
    state: SolveState,
    ctx: RuntimeContext,
    dim: int = 2,
) -> bool:
    """Handle ``N=0`` and ``N=1`` edge cases for init ops.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.
    ctx : RuntimeContext
        Execution infrastructure.
    dim : int, default=2
        Position dimensionality.

    Returns
    -------
    bool
        ``True`` when the edge case was handled and the caller should return.
    """
    if problem.num_nodes == 0:
        state.pos = None
        return True
    if problem.num_nodes == 1:
        state.pos = torch.zeros((1, dim), dtype=torch.float32, device=_target_device(problem, ctx))
        return True
    return False


def _layer_groups(layers: torch.Tensor) -> Dict[int, List[int]]:
    """Group nodes by their integer layer assignment.

    Parameters
    ----------
    layers : torch.Tensor
        Layer IDs with shape ``[N]``.

    Returns
    -------
    dict[int, list[int]]
        Nodes grouped by layer in input order.
    """
    groups: DefaultDict[int, List[int]] = defaultdict(list)
    for node, layer in enumerate(layers.tolist()):
        groups[int(layer)].append(node)
    return dict(groups)


def _parent_lists(edge_index: torch.Tensor, num_nodes: int) -> Dict[int, List[int]]:
    """Build parent lists for deterministic layered initialization.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    dict[int, list[int]]
        Parent node indices for each target.
    """
    parents: DefaultDict[int, List[int]] = defaultdict(list)
    if edge_index.numel() == 0:
        return dict(parents)
    for source, target in edge_index.detach().to(device="cpu", dtype=torch.long).t().tolist():
        if 0 <= source < num_nodes and 0 <= target < num_nodes:
            parents[target].append(source)
    return dict(parents)


def _barycenter_ordered_groups(
    edge_index: torch.Tensor,
    layers: torch.Tensor,
    num_nodes: int,
) -> Dict[int, List[int]]:
    """Order each layer by parent barycenter, preserving stable ties.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        Layer IDs with shape ``[N]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    dict[int, list[int]]
        Layer groups reordered by parent barycenter.
    """
    groups = _layer_groups(layers)
    parents = _parent_lists(edge_index=edge_index, num_nodes=num_nodes)
    current_order = {node: float(node) for node in range(num_nodes)}
    if not groups:
        return groups

    first_layer = min(groups)
    for layer in sorted(groups):
        ordered_nodes = groups[layer]
        if layer == first_layer:
            for index, node in enumerate(ordered_nodes):
                current_order[node] = float(index)
            continue

        enriched: List[Tuple[float, int, int]] = []
        for stable_index, node in enumerate(ordered_nodes):
            node_parents = parents.get(node, [])
            if node_parents:
                barycenter = sum(current_order[parent] for parent in node_parents) / float(
                    len(node_parents)
                )
            else:
                barycenter = current_order.get(node, float(stable_index))
            enriched.append((barycenter, stable_index, int(node)))

        enriched.sort()
        groups[layer] = [node for _, _, node in enriched]
        for index, node in enumerate(groups[layer]):
            current_order[node] = float(index)

    return groups


@register_op
class ValidateGraphOptInputs(Op):
    """Validate GraphOpt problem inputs before any state is initialized.

    Reads
    -----
    No ``SolveState`` fields.

    Writes
    ------
    Nothing. The op raises on invalid input instead of mutating state.

    Use this when
    -------------
    You want GraphOpt-style pipelines to fail early on malformed edge tensors
    or mismatched edge weights before allocating positions.
    """

    name = "validate_graphopt_inputs"
    category = OpCategory.INIT
    writes = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Validate the public GraphOpt arguments.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable GraphOpt problem definition.
        state : SolveState
            Mutable solve state (unused).
        ctx : RuntimeContext
            Execution context (unused).

        Returns
        -------
        SolveState
            Unmodified state.

        Raises
        ------
        ValueError
            If any GraphOpt input is invalid.
        """
        _ = ctx

        if problem.edge_index.ndim != 2 or problem.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")
        if problem.edge_weights is not None:
            if problem.edge_weights.ndim != 1:
                raise ValueError("edge_weights must have shape [E].")
            if problem.edge_weights.shape[0] != problem.edge_index.shape[1]:
                raise ValueError(
                    "edge_weights length "
                    f"{problem.edge_weights.shape[0]} != edge count "
                    f"{problem.edge_index.shape[1]}"
                )

        if problem.edge_index.numel() == 0:
            return state

        edge_index_cpu = problem.edge_index.to(device="cpu", dtype=torch.long)
        min_index = int(edge_index_cpu.min().item())
        max_index = int(edge_index_cpu.max().item())
        if min_index < 0:
            raise ValueError("edge_index cannot contain negative node indices.")
        if max_index >= problem.num_nodes:
            raise ValueError("edge_index contains node indices outside [0, num_nodes).")
        return state


@dataclass(frozen=True)
class ValidateFA2InputsConfig:
    """Configuration for :class:`ValidateFA2Inputs`.

    Parameters
    ----------
    steps : int, default=100
        Number of ForceAtlas2 iterations requested by the caller.
    barnes_hut_theta : float, default=1.2
        Barnes-Hut opening threshold. Must be positive.
    """

    steps: int = 100
    barnes_hut_theta: float = 1.2


@register_op
class ValidateFA2Inputs(Op):
    """Validate ForceAtlas2 problem inputs before preprocessing.

    Reads
    -----
    No ``SolveState`` fields.

    Writes
    ------
    Nothing. The op only raises on invalid public inputs.

    Use this when
    -------------
    You want FA2 pipelines to fail before building cached undirected state or
    allocating force buffers.
    """

    name = "validate_fa2_inputs"
    category = OpCategory.INIT
    writes = ()

    def __init__(self, config: Optional[ValidateFA2InputsConfig] = None) -> None:
        """Store the validation parameters.

        Parameters
        ----------
        config : ValidateFA2InputsConfig, optional
            FA2 validation settings.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or ValidateFA2InputsConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Validate public ForceAtlas2 arguments.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable ForceAtlas2 problem definition.
        state : SolveState
            Mutable solve state. Returned unchanged.
        ctx : RuntimeContext
            Execution context. Unused for deterministic validation.

        Returns
        -------
        SolveState
            Unmodified solve state.

        Raises
        ------
        ValueError
            If any ForceAtlas2 input is invalid.
        """
        del ctx

        if problem.num_nodes < 0:
            raise ValueError("num_nodes must be non-negative")
        if self.config.steps < 0:
            raise ValueError("steps must be non-negative")
        if self.config.barnes_hut_theta <= 0.0:
            raise ValueError("barnes_hut_theta must be positive")
        if problem.edge_index.dim() != 2 or problem.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E]")
        if problem.edge_index.dtype not in {
            torch.int8,
            torch.int16,
            torch.int32,
            torch.int64,
            torch.uint8,
        }:
            raise ValueError("edge_index must use an integer dtype")
        if problem.edge_weights is not None:
            if problem.edge_weights.dim() != 1:
                raise ValueError("edge_weights must be a one-dimensional tensor")
            if problem.edge_weights.shape[0] != problem.edge_index.shape[1]:
                raise ValueError("edge_weights length must match edge_index column count")

        if problem.edge_index.numel() == 0:
            return state

        edge_index_cpu = problem.edge_index.to(device="cpu", dtype=torch.long)
        min_index = int(edge_index_cpu.min().item())
        max_index = int(edge_index_cpu.max().item())
        if min_index < 0:
            raise ValueError("edge_index cannot contain negative node indices")
        if max_index >= problem.num_nodes:
            raise ValueError("edge_index contains node indices outside num_nodes")
        return state


@dataclass(frozen=True)
class GraphOptInitializePositionsConfig:
    """Configuration for :class:`GraphOptInitializePositions`.

    Parameters
    ----------
    position_dim : int, default=2
        Output dimensionality for the initialized position tensor.
    fidelity_mode : bool, default=False
        When ``True``, initialize from igraph's compiled default RNG stream if
        no explicit matrix is supplied.
    """

    position_dim: int = 2
    fidelity_mode: bool = False


@register_op
class GraphOptInitializePositions(Op):
    """Seed ``state.pos`` for GraphOpt iterations.

    Reads
    -----
    ``state.extras["graphopt_initial_pos"]`` when present. Otherwise uses
    ``problem.seed`` and ``problem.num_nodes``.

    Writes
    ------
    ``state.pos`` as a float64 tensor on the resolved target device.

    Use this when
    -------------
    You need GraphOpt-compatible random starts before running classic GraphOpt
    force updates. Fidelity mode mirrors igraph's native fallback random
    layout when the benchmark adapter has not supplied a seed matrix.
    """

    name = "graphopt_initialize_positions"
    category = OpCategory.INIT
    reads = (f"extras.{GRAPHOPT_INITIAL_POS_KEY}",)
    writes = ("pos",)

    def __init__(self, config: Optional[GraphOptInitializePositionsConfig] = None) -> None:
        """Store the GraphOpt initialization configuration.

        Parameters
        ----------
        config : GraphOptInitializePositionsConfig, optional
            GraphOpt position-initialization settings.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or GraphOptInitializePositionsConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize positions from a supplied matrix or a seeded RNG.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable GraphOpt inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context.

        Returns
        -------
        SolveState
            State with ``state.pos`` initialized.
        """
        initial_pos = state.extras.get(GRAPHOPT_INITIAL_POS_KEY)
        if initial_pos is not None:
            initial_tensor = torch.as_tensor(initial_pos, dtype=torch.float64)
            expected_shape = (problem.num_nodes, self.config.position_dim)
            if tuple(initial_tensor.shape) != expected_shape:
                raise ValueError(
                    f"{GRAPHOPT_INITIAL_POS_KEY} must have shape "
                    f"{expected_shape}, got {tuple(initial_tensor.shape)}"
                )
            state.pos = initial_tensor.to(device=_target_device(problem, ctx))
            return state

        if problem.num_nodes == 0:
            state.pos = torch.empty(
                (0, self.config.position_dim),
                dtype=torch.float64,
                device=_target_device(problem, ctx),
            )
            return state

        if self.config.fidelity_mode:
            rng = make_igraph_default_rng(problem.seed)
            positions = [
                [rng.uniform(-1.0, 1.0) for _ in range(self.config.position_dim)]
                for _ in range(problem.num_nodes)
            ]
        else:
            rng = random.Random(problem.seed)
            positions = [
                [rng.random() for _ in range(self.config.position_dim)]
                for _ in range(problem.num_nodes)
            ]
        state.pos = torch.tensor(
            positions, dtype=torch.float64, device=_target_device(problem, ctx)
        )
        return state


@dataclass(frozen=True)
class KamadaKawaiInitializePositionsConfig:
    """Configuration for :class:`KamadaKawaiInitializePositions`.

    Parameters
    ----------
    position_dim : int, default=2
        Output dimensionality for the initialized position tensor.
    """

    position_dim: int = 2


@register_op
class KamadaKawaiInitializePositions(Op):
    """Initialize KK positions from user input or a deterministic circle fallback.

    Reads
    -----
    ``state.extras["kk_initial_pos"]`` when present.

    Writes
    ------
    ``state.pos`` as a float64 tensor.

    Use this when
    -------------
    You want Kamada-Kawai to start from caller-supplied positions when
    available, otherwise from the classic circle layout.
    """

    name = "kamada_kawai_initialize_positions"
    category = OpCategory.INIT
    reads = ("extras.kk_initial_pos",)
    writes = ("pos",)

    def __init__(self, config: Optional[KamadaKawaiInitializePositionsConfig] = None) -> None:
        """Store the Kamada-Kawai initialization configuration.

        Parameters
        ----------
        config : KamadaKawaiInitializePositionsConfig, optional
            Kamada-Kawai initialization settings.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or KamadaKawaiInitializePositionsConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Use provided positions when available or a deterministic circle fallback.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable Kamada-Kawai inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context.

        Returns
        -------
        SolveState
            State with ``state.pos`` initialized.
        """
        del ctx

        if problem.num_nodes == 0:
            state.pos = torch.empty((0, self.config.position_dim), dtype=torch.float64)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, self.config.position_dim), dtype=torch.float64)
            return state

        initial_positions = state.extras.get("kk_initial_pos")
        if isinstance(initial_positions, torch.Tensor):
            expected_shape = (problem.num_nodes, self.config.position_dim)
            if initial_positions.shape != expected_shape:
                raise ValueError(
                    "kk_initial_pos must have shape "
                    f"{expected_shape}, got {tuple(initial_positions.shape)}"
                )
            state.pos = initial_positions.to(dtype=torch.float64)
            return state

        # The fallback mirrors the classic circular seed so KK starts from a
        # deterministic, non-degenerate configuration without needing input positions.
        theta = np.linspace(0, 1, num=problem.num_nodes + 1)[:-1] * (2.0 * np.pi)
        theta = theta.astype(np.float32)
        coordinates = np.column_stack((np.cos(theta), np.sin(theta))).astype(np.float64, copy=False)
        coordinates = torch.from_numpy(coordinates)
        state.pos = _rescale_layout(coordinates)
        return state


@dataclass(frozen=True)
class FA2InitializePositionsConfig:
    """Configuration for :class:`FA2InitializePositions`.

    Parameters
    ----------
    position_dim : int, default=2
        Output dimensionality for the initialized position tensor.
    dtype : torch.dtype, default=torch.float32
        Floating-point dtype for the initialized positions.
    """

    position_dim: int = 2
    dtype: torch.dtype = torch.float32


@register_op
class FA2InitializePositions(Op):
    """Seed ``state.pos`` with the reference FA2 Python-random initializer.

    Reads
    -----
    No ``SolveState`` fields. Uses ``problem.seed`` and ``problem.num_nodes``.

    Writes
    ------
    ``state.pos`` on ``problem.edge_index.device`` using the configured dtype.

    Use this when
    -------------
    You want FA2-compatible random starts before building FA2 force caches.
    """

    name = "fa2_initialize_positions"
    category = OpCategory.INIT
    writes = ("pos",)

    def __init__(self, config: Optional[FA2InitializePositionsConfig] = None) -> None:
        """Store the FA2 initialization configuration.

        Parameters
        ----------
        config : FA2InitializePositionsConfig, optional
            ForceAtlas2 initialization settings.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or FA2InitializePositionsConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize ``state.pos`` on ``[0, 1]^2`` using ``random.Random``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable FA2 inputs containing ``num_nodes`` and ``seed``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused because FA2 follows the edge tensor
            device exactly.

        Returns
        -------
        SolveState
            State with ``state.pos`` initialized using the configured dtype.
        """
        _ = ctx

        if problem.num_nodes == 0:
            state.pos = torch.zeros(
                (0, self.config.position_dim),
                dtype=self.config.dtype,
                device=problem.edge_index.device,
            )
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros(
                (1, self.config.position_dim),
                dtype=self.config.dtype,
                device=problem.edge_index.device,
            )
            return state

        rng = random.Random(problem.seed)
        positions = [
            [rng.random() for _ in range(self.config.position_dim)]
            for _ in range(problem.num_nodes)
        ]
        state.pos = torch.tensor(
            positions,
            dtype=self.config.dtype,
            device=problem.edge_index.device,
        )
        return state


@dataclass(frozen=True)
class RandomUniformInitConfig:
    """Configuration for :class:`RandomUniformInit`.

    Parameters
    ----------
    scale : str, default="sqrt_n"
        Output scaling rule. ``"unit"`` and ``"none"`` leave the random
        sample unchanged; ``"sqrt_n"`` multiplies by ``sqrt(max(N, 1))``.
    range : tuple[float, float], default=(0.0, 1.0)
        Inclusive lower bound and exclusive upper bound for the sample range.
    rng_backend : str, default="torch"
        Random backend: ``"torch"``, ``"python"``, or ``"numpy"``.
    position_dim : int, default=2
        Output dimensionality for the initialized position tensor.
    """

    scale: str = "sqrt_n"
    range: Tuple[float, float] = (0.0, 1.0)
    rng_backend: str = "torch"
    position_dim: int = 2


@register_op
class RandomUniformInit(Op):
    """Initialize positions from a reproducible uniform distribution.

    Reads
    -----
    No ``SolveState`` fields. Uses ``problem.seed`` and optional
    ``ctx.generator`` for the torch backend.

    Writes
    ------
    ``state.pos`` with the configured scale, range, and backend.

    Use this when
    -------------
    You want a generic random initializer for pipelines that do not need a
    layout-specific seed strategy.
    """

    name = "random_uniform_init"
    category = OpCategory.INIT
    writes = ("pos",)

    def __init__(self, config: Optional[RandomUniformInitConfig] = None) -> None:
        """Store the initialization configuration.

        Parameters
        ----------
        config : RandomUniformInitConfig, optional
            Uniform initialization configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or RandomUniformInitConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize ``state.pos`` with uniform random samples.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with initialized positions.
        """
        if (
            problem.num_nodes == 0
            and self.config.rng_backend == "numpy"
            and self.config.scale == "none"
        ):
            state.pos = torch.empty(
                (0, self.config.position_dim),
                dtype=torch.float64,
                device=_target_device(problem, ctx),
            )
            return state

        if _maybe_set_empty_or_single_positions(
            problem=problem,
            state=state,
            ctx=ctx,
            dim=self.config.position_dim,
        ):
            return state

        low, high = self.config.range
        if high < low:
            raise ValueError("RandomUniformInit range must satisfy high >= low.")

        if self.config.rng_backend == "torch":
            positions = torch.rand(
                (problem.num_nodes, self.config.position_dim),
                generator=_torch_generator(problem=problem, ctx=ctx),
                dtype=torch.float32,
            )
        elif self.config.rng_backend == "python":
            rng = random.Random(problem.seed)
            positions = torch.tensor(
                [rng.random() for _ in range(problem.num_nodes * self.config.position_dim)],
                dtype=torch.float32,
            ).reshape(problem.num_nodes, self.config.position_dim)
        elif self.config.rng_backend == "numpy":
            positions = torch.from_numpy(
                np.random.RandomState(problem.seed).rand(
                    problem.num_nodes, self.config.position_dim
                )
            )
        else:
            raise ValueError(
                f"Unsupported RandomUniformInit rng_backend: {self.config.rng_backend}"
            )

        positions = positions * (high - low) + low
        positions = positions * _scale_factor(num_nodes=problem.num_nodes, rule=self.config.scale)
        state.pos = positions.to(device=_target_device(problem, ctx))
        return state


@dataclass(frozen=True)
class RandomNormalInitConfig:
    """Configuration for :class:`RandomNormalInit`.

    Parameters
    ----------
    std : float, default=1.0e-4
        Standard deviation of the Gaussian sample.
    mean : float, default=0.0
        Mean of the Gaussian sample.
    scale : str, default="none"
        Additional output scaling rule.
    position_dim : int, default=2
        Output dimensionality for the initialized position tensor.
    """

    std: float = 1.0e-4
    mean: float = 0.0
    scale: str = "none"
    position_dim: int = 2


@register_op
class RandomNormalInit(Op):
    """Initialize positions from a reproducible normal distribution.

    Reads
    -----
    No ``SolveState`` fields. Uses ``problem.seed`` and optional
    ``ctx.generator``.

    Writes
    ------
    ``state.pos`` with Gaussian samples on the resolved target device.

    Use this when
    -------------
    You want a small-noise initializer around the origin for optimization
    pipelines that quickly impose their own structure.
    """

    name = "random_normal_init"
    category = OpCategory.INIT
    writes = ("pos",)

    def __init__(self, config: Optional[RandomNormalInitConfig] = None) -> None:
        """Store the normal initialization configuration.

        Parameters
        ----------
        config : RandomNormalInitConfig, optional
            Normal initialization configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or RandomNormalInitConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize ``state.pos`` with Gaussian random samples.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with initialized positions.
        """
        if _maybe_set_empty_or_single_positions(
            problem=problem,
            state=state,
            ctx=ctx,
            dim=self.config.position_dim,
        ):
            return state

        positions = torch.randn(
            (problem.num_nodes, self.config.position_dim),
            generator=_torch_generator(problem=problem, ctx=ctx),
            dtype=torch.float32,
        )
        positions = positions * self.config.std + self.config.mean
        positions = positions * _scale_factor(num_nodes=problem.num_nodes, rule=self.config.scale)
        state.pos = positions.to(device=_target_device(problem, ctx))
        return state


@dataclass(frozen=True)
class LinLogInitializePositionsConfig:
    """Configuration for :class:`LinLogInitializePositions`.

    Parameters
    ----------
    position_dim : int, default=2
        Output dimensionality for the initialized position tensor.
    """

    position_dim: int = 2


@register_op
class LinLogInitializePositions(Op):
    """Seed ``state.pos`` exactly like the classic LinLog initializer.

    Reads
    -----
    No ``SolveState`` fields. Uses ``problem.seed`` and the layout device
    derived from immutable problem inputs.

    Writes
    ------
    ``state.pos`` as a float32 tensor with gradients enabled.

    Use this when
    -------------
    You need parity with the historical LinLog random start before its energy
    optimization begins.
    """

    name = "linlog_initialize_positions"
    category = OpCategory.INIT
    writes = ("pos",)

    def __init__(self, config: Optional[LinLogInitializePositionsConfig] = None) -> None:
        """Store the LinLog initialization configuration.

        Parameters
        ----------
        config : LinLogInitializePositionsConfig, optional
            LinLog initialization settings.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or LinLogInitializePositionsConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Seed ``state.pos`` with LinLog-compatible random values.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing ``num_nodes`` and ``seed``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime infrastructure.

        Returns
        -------
        SolveState
            State with ``state.pos`` populated as a float32 tensor.
        """
        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        if problem.num_nodes == 0:
            state.pos = torch.empty(
                (0, self.config.position_dim),
                dtype=torch.float32,
                device=output_device,
            )
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros(
                (1, self.config.position_dim),
                dtype=torch.float32,
                device=output_device,
            )
            return state

        positions = torch.randn(
            (problem.num_nodes, self.config.position_dim),
            generator=torch.Generator(device="cpu").manual_seed(problem.seed),
            dtype=torch.float32,
            device=output_device,
        )
        state.pos = positions.requires_grad_(True)
        return state


@dataclass(frozen=True)
class CircularInitConfig:
    """Configuration for :class:`CircularInit`.

    Parameters
    ----------
    scale : float, default=1.0
        Radius of the output circle.
    position_dim : int, default=2
        Output dimensionality for the initialized position tensor.
    """

    scale: float = 1.0
    position_dim: int = 2


@register_op
class CircularInit(Op):
    """Place nodes uniformly on a circle in deterministic node order.

    Reads
    -----
    No ``SolveState`` fields.

    Writes
    ------
    ``state.pos`` with evenly spaced circle coordinates.

    Use this when
    -------------
    You need a cheap deterministic seed that preserves node order and avoids
    random variation between runs.
    """

    name = "circular_init"
    category = OpCategory.INIT
    writes = ("pos",)

    def __init__(self, config: Optional[CircularInitConfig] = None) -> None:
        """Store the circular initialization configuration.

        Parameters
        ----------
        config : CircularInitConfig, optional
            Circular initialization configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or CircularInitConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize ``state.pos`` with evenly spaced circle coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with initialized positions.
        """
        if _maybe_set_empty_or_single_positions(
            problem=problem,
            state=state,
            ctx=ctx,
            dim=self.config.position_dim,
        ):
            return state

        points = []
        for index in range(problem.num_nodes):
            theta = (2.0 * pi * float(index)) / float(problem.num_nodes)
            point = [0.0] * self.config.position_dim
            point[0] = self.config.scale * cos(theta)
            if self.config.position_dim > 1:
                point[1] = self.config.scale * sin(theta)
            points.append(point)
        state.pos = torch.tensor(points, dtype=torch.float32, device=_target_device(problem, ctx))
        return state


@dataclass(frozen=True)
class XavierInitConfig:
    """Configuration for :class:`XavierInit`.

    Parameters
    ----------
    dim : int, default=2
        Output embedding dimensionality.
    """

    dim: int = 2


@register_op
class XavierInit(Op):
    """Initialize positions with Xavier-uniform sampling.

    Reads
    -----
    No ``SolveState`` fields. Uses ``problem.seed`` to bracket the global CPU
    RNG state around ``torch.nn.init.xavier_uniform_``.

    Writes
    ------
    ``state.pos`` with Xavier-uniform samples on the resolved target device.

    Use this when
    -------------
    You want a scale-aware random initializer that matches common neural
    parameter initialization heuristics.
    """

    name = "xavier_init"
    category = OpCategory.INIT
    writes = ("pos",)

    def __init__(self, config: Optional[XavierInitConfig] = None) -> None:
        """Store the Xavier initialization configuration.

        Parameters
        ----------
        config : XavierInitConfig, optional
            Xavier initialization configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or XavierInitConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize ``state.pos`` with Xavier-uniform samples.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with initialized positions.
        """
        if _maybe_set_empty_or_single_positions(
            problem=problem,
            state=state,
            ctx=ctx,
            dim=self.config.dim,
        ):
            return state

        cpu_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(problem.seed)
            positions = torch.empty((problem.num_nodes, self.config.dim), dtype=torch.float32)
            # The gain matches the historical implementation, which scales the
            # Xavier envelope by graph size rather than leaving the default gain at 1.
            torch.nn.init.xavier_uniform_(positions, gain=sqrt(float(max(problem.num_nodes, 1))))
        finally:
            torch.random.set_rng_state(cpu_state)

        state.pos = positions.to(device=_target_device(problem, ctx))
        return state


@dataclass(frozen=True)
class DeterministicInitConfig:
    """Configuration for :class:`DeterministicInit`.

    Parameters
    ----------
    method : str, default="barycenter"
        Deterministic ordering rule. Supported values are ``"barycenter"``
        and ``"input"``.
    node_sep : float, default=25.0
        Horizontal spacing between consecutive nodes in a layer.
    rank_sep : float, default=50.0
        Vertical spacing between consecutive layers.
    position_dim : int, default=2
        Output dimensionality for the initialized position tensor.
    """

    method: str = "barycenter"
    node_sep: float = 25.0
    rank_sep: float = 50.0
    position_dim: int = 2


@register_op
class DeterministicInit(Op):
    """Initialize positions from layer order instead of randomness.

    Reads
    -----
    ``state.layers``.

    Writes
    ------
    ``state.pos`` with deterministic per-layer coordinates.

    Use this when
    -------------
    You already have a DAG layering and want a stable, interpretable seed for
    layered or Sugiyama-style downstream optimization.
    """

    name = "deterministic_init"
    category = OpCategory.INIT
    reads = ("layers",)
    writes = ("pos",)
    requires = ("layers",)

    def __init__(self, config: Optional[DeterministicInitConfig] = None) -> None:
        """Store the deterministic initialization configuration.

        Parameters
        ----------
        config : DeterministicInitConfig, optional
            Deterministic initialization configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or DeterministicInitConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize ``state.pos`` from layer assignments.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with initialized positions.

        Raises
        ------
        ValueError
            If layer assignments are required but missing, or if the method
            is unsupported.
        """
        if _maybe_set_empty_or_single_positions(
            problem=problem,
            state=state,
            ctx=ctx,
            dim=self.config.position_dim,
        ):
            return state
        if state.layers is None:
            raise ValueError("deterministic_init requires state.layers to be set.")

        layers = state.layers.detach().to(device="cpu", dtype=torch.long)
        if self.config.method == "barycenter":
            groups = _barycenter_ordered_groups(
                edge_index=problem.edge_index,
                layers=layers,
                num_nodes=problem.num_nodes,
            )
        elif self.config.method == "input":
            groups = _layer_groups(layers)
        else:
            raise ValueError(f"Unsupported DeterministicInit method: {self.config.method}")

        positions = torch.zeros((problem.num_nodes, self.config.position_dim), dtype=torch.float32)
        for layer in sorted(groups):
            y_value = float(layer) * self.config.rank_sep
            for index, node in enumerate(groups[layer]):
                positions[node, 0] = float(index) * self.config.node_sep
                if self.config.position_dim > 1:
                    positions[node, 1] = y_value

        state.pos = positions.to(device=_target_device(problem, ctx))
        return state


def _fallback_line_coordinates(num_nodes: int, dim: int) -> np.ndarray:
    """Build a deterministic line embedding for degenerate decompositions.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.
    dim : int
        Output embedding dimensionality.

    Returns
    -------
    numpy.ndarray
        Coordinate matrix with shape ``[N, dim]``.
    """
    coordinates = np.zeros((num_nodes, dim), dtype=np.float64)
    if num_nodes > 0 and dim > 0:
        coordinates[:, 0] = np.linspace(-1.0, 1.0, num_nodes, dtype=np.float64)
    return coordinates


def _build_sparse_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
) -> sparse.csr_matrix:
    """Build a SciPy sparse adjacency matrix from an edge list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse adjacency matrix with shape ``[N, N]``.

    Raises
    ------
    ValueError
        If ``edge_index`` has the wrong shape or ``edge_weights`` does not
        match the edge count.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
        )
    if edge_index.numel() == 0:
        return sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    rows = edge_index_cpu[0].numpy()
    cols = edge_index_cpu[1].numpy()
    if edge_weights is None:
        data = np.ones(rows.shape[0], dtype=np.float64)
    else:
        data = edge_weights.detach().to(device="cpu", dtype=torch.float64).numpy()
    return sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float64)


def _symmetrize_adjacency(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Convert a possibly directed adjacency into the undirected form used here.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix
        Input adjacency matrix.

    Returns
    -------
    scipy.sparse.csr_matrix
        Symmetric adjacency matrix.
    """
    if (adjacency - adjacency.T).nnz == 0:
        return adjacency
    return (adjacency + adjacency.T).tocsr()


def _laplacian_matrix(
    adjacency: sparse.csr_matrix,
    normalization: str,
) -> Tuple[sparse.csr_matrix, bool]:
    """Build the requested Laplacian matrix.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix
        Symmetric adjacency matrix with shape ``[N, N]``.
    normalization : str
        Laplacian normalization mode.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, bool]
        Laplacian matrix and whether it is symmetric.

    Raises
    ------
    ValueError
        If ``normalization`` is unsupported.
    """
    degrees = np.asarray(adjacency.sum(axis=1)).reshape(-1).astype(np.float64, copy=False)
    degree_matrix = sparse.diags(degrees, offsets=0, format="csr")

    if normalization == "unnormalized":
        return (degree_matrix - adjacency).tocsr(), True
    if normalization == "symmetric":
        inv_sqrt = np.zeros_like(degrees)
        nonzero_mask = degrees > 0.0
        inv_sqrt[nonzero_mask] = 1.0 / np.sqrt(degrees[nonzero_mask])
        normalized = sparse.diags(inv_sqrt, offsets=0, format="csr")
        identity = sparse.identity(adjacency.shape[0], format="csr", dtype=np.float64)
        return (identity - (normalized @ adjacency @ normalized)).tocsr(), True
    if normalization == "random_walk":
        inv_degree = np.zeros_like(degrees)
        nonzero_mask = degrees > 0.0
        inv_degree[nonzero_mask] = 1.0 / degrees[nonzero_mask]
        normalized = sparse.diags(inv_degree, offsets=0, format="csr")
        identity = sparse.identity(adjacency.shape[0], format="csr", dtype=np.float64)
        return (identity - (normalized @ adjacency)).tocsr(), False
    raise ValueError("normalization must be one of 'symmetric', 'random_walk', or 'unnormalized'.")


def _select_nontrivial_eigenvectors(
    eigenvalues: np.ndarray,
    eigenvectors: np.ndarray,
    dim: int,
) -> np.ndarray:
    """Select the first non-trivial Laplacian eigenvectors.

    Parameters
    ----------
    eigenvalues : numpy.ndarray
        Eigenvalues with shape ``[K]``.
    eigenvectors : numpy.ndarray
        Eigenvectors with shape ``[N, K]``.
    dim : int
        Output dimensionality.

    Returns
    -------
    numpy.ndarray
        Coordinate matrix with shape ``[N, dim]``.
    """
    sorted_indices = np.argsort(np.real(eigenvalues))
    nontrivial = [
        index
        for index in sorted_indices
        if abs(float(np.real(eigenvalues[index]))) > _SPECTRAL_EIGEN_TOLERANCE
    ][:dim]

    coordinates = np.zeros((eigenvectors.shape[0], dim), dtype=np.float64)
    if nontrivial:
        coordinates[:, : len(nontrivial)] = np.real(eigenvectors[:, nontrivial])
        return coordinates
    return _fallback_line_coordinates(eigenvectors.shape[0], dim)


def _dense_spectral_coordinates(
    laplacian: sparse.csr_matrix,
    dim: int,
    symmetric: bool,
) -> np.ndarray:
    """Compute dense spectral coordinates.

    Parameters
    ----------
    laplacian : scipy.sparse.csr_matrix
        Laplacian matrix with shape ``[N, N]``.
    dim : int
        Output dimensionality.
    symmetric : bool
        Whether the Laplacian is symmetric.

    Returns
    -------
    numpy.ndarray
        Coordinate matrix with shape ``[N, dim]``.
    """
    dense_laplacian = laplacian.toarray()
    if symmetric:
        eigenvalues, eigenvectors = np.linalg.eigh(dense_laplacian)
    else:
        eigenvalues, eigenvectors = np.linalg.eig(dense_laplacian)
    return _select_nontrivial_eigenvectors(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        dim=dim,
    )


def _sparse_spectral_coordinates(
    laplacian: sparse.csr_matrix,
    dim: int,
    symmetric: bool,
) -> np.ndarray:
    """Compute spectral coordinates with ARPACK.

    Parameters
    ----------
    laplacian : scipy.sparse.csr_matrix
        Laplacian matrix with shape ``[N, N]``.
    dim : int
        Output dimensionality.
    symmetric : bool
        Whether the Laplacian is symmetric.

    Returns
    -------
    numpy.ndarray
        Coordinate matrix with shape ``[N, dim]``.
    """
    num_nodes = int(laplacian.shape[0])
    eigen_count = min(num_nodes - 1, max(dim + 4, dim + 1))
    if eigen_count <= dim:
        return _dense_spectral_coordinates(laplacian=laplacian, dim=dim, symmetric=symmetric)

    lanczos_vectors = max((2 * eigen_count) + 1, int(np.sqrt(num_nodes)))
    ncv = min(max(lanczos_vectors, eigen_count + 2), num_nodes)
    if symmetric:
        eigenvalues, eigenvectors = sparse_linalg.eigsh(
            laplacian,
            k=eigen_count,
            which="SM",
            ncv=ncv,
        )
    else:
        eigenvalues, eigenvectors = sparse_linalg.eigs(
            laplacian,
            k=eigen_count,
            which="SR",
            ncv=ncv,
        )
    return _select_nontrivial_eigenvectors(
        eigenvalues=eigenvalues,
        eigenvectors=eigenvectors,
        dim=dim,
    )


def _double_center_squared_distances(distances: np.ndarray) -> np.ndarray:
    """Double-center a squared distance matrix into a Gram matrix.

    Parameters
    ----------
    distances : numpy.ndarray
        Pairwise distance matrix with shape ``[N, N]``.

    Returns
    -------
    numpy.ndarray
        Centered Gram matrix with shape ``[N, N]``.
    """
    squared = np.square(distances, dtype=np.float64)
    row_means = squared.mean(axis=1, keepdims=True)
    col_means = squared.mean(axis=0, keepdims=True)
    grand_mean = float(squared.mean())
    return -0.5 * (squared - row_means - col_means + grand_mean)


def _positive_eigh_coordinates(gram: np.ndarray, dim: int) -> np.ndarray:
    """Recover coordinates from a Gram matrix using positive eigenpairs.

    Parameters
    ----------
    gram : numpy.ndarray
        Gram matrix with shape ``[N, N]``.
    dim : int
        Output dimensionality.

    Returns
    -------
    numpy.ndarray
        Coordinate matrix with shape ``[N, dim]``.
    """
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    sorted_indices = np.argsort(eigenvalues)[::-1]
    positive = [index for index in sorted_indices if float(eigenvalues[index]) > 0.0][:dim]

    coordinates = np.zeros((gram.shape[0], dim), dtype=np.float64)
    if positive:
        selected_values = np.clip(eigenvalues[positive], a_min=0.0, a_max=None)
        selected_vectors = eigenvectors[:, positive]
        coordinates[:, : len(positive)] = selected_vectors * np.sqrt(selected_values)
        return coordinates
    return _fallback_line_coordinates(gram.shape[0], dim)


def _pivot_mds_coordinates(distance_matrix: torch.Tensor, dim: int) -> torch.Tensor:
    """Recover a low-rank embedding from pivot distances with SVD.

    Parameters
    ----------
    distance_matrix : torch.Tensor
        Pivot-to-node distance matrix with shape ``[P, N]``.
    dim : int
        Output dimensionality.

    Returns
    -------
    torch.Tensor
        Coordinate matrix with shape ``[N, dim]`` on CPU.
    """
    squared = distance_matrix.detach().to(device="cpu", dtype=torch.float64).square()
    row_means = squared.mean(dim=1, keepdim=True)
    col_means = squared.mean(dim=0, keepdim=True)
    grand_mean = squared.mean()
    centered = -0.5 * (squared - row_means - col_means + grand_mean)

    _, singular_values, vh = torch.linalg.svd(centered, full_matrices=False)
    coord_dims = min(dim, int(singular_values.shape[0]))
    if coord_dims == 0:
        return torch.from_numpy(_fallback_line_coordinates(distance_matrix.shape[1], dim)).to(
            dtype=torch.float32
        )

    scales = singular_values[:coord_dims].clamp_min(0.0)
    coordinates = vh[:coord_dims].transpose(0, 1) * scales.unsqueeze(0)
    if coord_dims < dim:
        padding = torch.zeros((coordinates.shape[0], dim - coord_dims), dtype=coordinates.dtype)
        coordinates = torch.cat((coordinates, padding), dim=1)
    if float(coordinates.abs().max().item()) <= _SPECTRAL_EIGEN_TOLERANCE:
        return torch.from_numpy(_fallback_line_coordinates(distance_matrix.shape[1], dim)).to(
            dtype=torch.float32
        )
    return coordinates.to(dtype=torch.float32)


def _resolve_layout_algorithm(algorithm: str) -> Callable[..., Any]:
    """Resolve a classic layout function by short algorithm name.

    Parameters
    ----------
    algorithm : str
        Algorithm short name such as ``"fr"``.

    Returns
    -------
    Callable[..., Any]
        Resolved layout function.

    Raises
    ------
    ValueError
        If the algorithm is unknown.
    """
    from dagua.layout import classic as classic_layouts

    function_name = f"layout_{algorithm}"
    if hasattr(classic_layouts, function_name):
        return getattr(classic_layouts, function_name)

    try:
        module = importlib.import_module(f"dagua.layout.classic.{algorithm}")
    except ModuleNotFoundError as exc:
        raise ValueError(f"Unsupported inner layout algorithm: {algorithm!r}.") from exc

    try:
        return getattr(module, function_name)
    except AttributeError as exc:
        raise ValueError(f"Unsupported inner layout algorithm: {algorithm!r}.") from exc


def _call_inner_layout(
    layout_fn: Callable[..., Any],
    problem: LayoutProblem,
    inner_config: Optional[Dict[str, Any]],
    inner_steps: int,
) -> torch.Tensor:
    """Invoke a classic layout function with reserved arguments enforced.

    Parameters
    ----------
    layout_fn : Callable[..., Any]
        Layout function to invoke.
    problem : LayoutProblem
        Immutable layout inputs.
    inner_config : dict[str, Any] | None
        Optional algorithm-specific keyword arguments.
    inner_steps : int
        Iteration budget forwarded as ``steps`` when supported.

    Returns
    -------
    torch.Tensor
        Returned position tensor with shape ``[N, 2]``.

    Raises
    ------
    TypeError
        If the inner algorithm does not return a tensor.
    ValueError
        If ``inner_config`` attempts to override reserved inputs or passes
        an unsupported keyword argument.
    """
    signature = inspect.signature(layout_fn)
    accepted_parameters = set(signature.parameters)
    reserved = {"edge_index", "num_nodes", "node_sizes", "edge_weights", "seed", "steps"}
    call_kwargs: Dict[str, Any] = {}

    base_kwargs: Dict[str, Any] = {
        "edge_index": problem.edge_index,
        "num_nodes": problem.num_nodes,
        "node_sizes": problem.node_sizes,
        "seed": problem.seed,
        "edge_weights": problem.edge_weights,
    }
    for key, value in base_kwargs.items():
        if key in accepted_parameters:
            call_kwargs[key] = value
    if "steps" in accepted_parameters:
        call_kwargs["steps"] = inner_steps

    if inner_config is not None:
        for key, value in inner_config.items():
            if key in reserved:
                raise ValueError(
                    f"inner_config may not override reserved argument {key!r}; "
                    "use the top-level config fields instead."
                )
            if key not in accepted_parameters:
                raise ValueError(
                    f"Inner algorithm {layout_fn.__name__} does not accept keyword {key!r}."
                )
            call_kwargs[key] = value

    result = layout_fn(**call_kwargs)
    positions = result[0] if isinstance(result, tuple) else result
    if not isinstance(positions, torch.Tensor):
        raise TypeError(
            f"Inner layout {layout_fn.__name__} returned {type(positions)!r}, not Tensor."
        )
    return positions


@dataclass(frozen=True)
class SpectralInitConfig:
    """Configuration for :class:`SpectralInit`.

    Parameters
    ----------
    normalization : str, default="symmetric"
        Laplacian normalization mode.
    sparse_threshold : int, default=500
        Graph-size cutoff for switching from dense eigendecomposition to
        sparse ARPACK.
    k : int, default=2
        Number of non-trivial eigenvectors to return as coordinates.
    """

    normalization: str = "symmetric"
    sparse_threshold: int = 500
    k: int = 2


@register_op
class SpectralInit(Op):
    """Initialize positions from graph Laplacian eigenvectors.

    Notes
    -----
    This op is deterministic for the dense path. The sparse ARPACK path has
    no explicit seed control because SciPy manages its internal initialization.

    Reads
    -----
    No ``SolveState`` fields.

    Writes
    ------
    ``state.laplacian`` and ``state.pos``.

    Use this when
    -------------
    You want a structure-aware deterministic initializer that reflects coarse
    graph connectivity before force-directed refinement.
    """

    name = "spectral_init"
    category = OpCategory.INIT
    writes = ("pos", "laplacian")

    def __init__(self, config: Optional[SpectralInitConfig] = None) -> None:
        """Store the spectral initialization configuration.

        Parameters
        ----------
        config : SpectralInitConfig, optional
            Spectral initialization configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or SpectralInitConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize ``state.pos`` from Laplacian eigenvectors.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with spectral positions and the cached Laplacian.

        Raises
        ------
        ValueError
            If ``k`` is not positive.
        """
        if self.config.k <= 0:
            raise ValueError("SpectralInit k must be positive.")
        if _maybe_set_empty_or_single_positions(
            problem=problem,
            state=state,
            ctx=ctx,
            dim=self.config.k,
        ):
            return state

        adjacency = _build_sparse_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        laplacian, is_symmetric = _laplacian_matrix(
            adjacency=_symmetrize_adjacency(adjacency),
            normalization=self.config.normalization,
        )
        state.laplacian = laplacian

        if problem.num_nodes < self.config.sparse_threshold:
            coordinates = _dense_spectral_coordinates(
                laplacian=laplacian,
                dim=self.config.k,
                symmetric=is_symmetric,
            )
        else:
            coordinates = _sparse_spectral_coordinates(
                laplacian=laplacian,
                dim=self.config.k,
                symmetric=is_symmetric,
            )

        state.pos = torch.from_numpy(coordinates).to(
            dtype=torch.float32,
            device=_target_device(problem, ctx),
        )
        return state


@dataclass(frozen=True)
class ClassicalMDSInitConfig:
    """Configuration for :class:`ClassicalMDSInit`.

    Parameters
    ----------
    unreachable_fill : str, default="max_plus_1"
        Fill strategy for unreachable graph distances.
    position_dim : int, default=2
        Output dimensionality for the recovered embedding.
    """

    unreachable_fill: str = "max_plus_1"
    position_dim: int = 2


@register_op
class ClassicalMDSInit(Op):
    """Initialize positions from all-pairs graph distances via classical MDS.

    Reads
    -----
    No pre-existing ``SolveState`` fields. The op builds its own adjacency and
    distance caches.

    Writes
    ------
    ``state.adjacency``, optional ``state.adjacency_weighted``,
    adjacency metadata in ``state.extras``, ``state.distance_matrix``, and
    ``state.pos``.

    Use this when
    -------------
    You want a dense, globally informed initializer for small-to-medium graphs
    where exact shortest paths are affordable.
    """

    name = "classical_mds_init"
    category = OpCategory.INIT
    writes = (
        "adjacency",
        "adjacency_weighted",
        "extras.adjacency_format",
        "extras.adjacency_directed",
        "extras.adjacency_weighted",
        "distance_matrix",
        "pos",
    )

    def __init__(self, config: Optional[ClassicalMDSInitConfig] = None) -> None:
        """Store the classical-MDS configuration.

        Parameters
        ----------
        config : ClassicalMDSInitConfig, optional
            Classical-MDS initialization configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or ClassicalMDSInitConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize ``state.pos`` from all-pairs shortest-path distances.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with APSP caches and classical-MDS positions.
        """
        if _maybe_set_empty_or_single_positions(problem=problem, state=state, ctx=ctx):
            return state

        BuildAdjacency(BuildAdjacencyConfig(weighted=problem.edge_weights is not None)).apply(
            problem, state, ctx
        )
        AllPairsShortestPaths(
            AllPairsShortestPathsConfig(unreachable_fill=self.config.unreachable_fill)
        ).apply(problem, state, ctx)

        if state.distance_matrix is None:
            raise RuntimeError("AllPairsShortestPaths did not populate state.distance_matrix.")

        distances = state.distance_matrix.detach().to(device="cpu", dtype=torch.float64).numpy()
        gram = _double_center_squared_distances(distances)
        coordinates = _positive_eigh_coordinates(gram=gram, dim=self.config.position_dim)
        state.pos = torch.from_numpy(coordinates).to(
            dtype=torch.float32,
            device=_target_device(problem, ctx),
        )
        return state


@dataclass(frozen=True)
class PivotMDSInitConfig:
    """Configuration for :class:`PivotMDSInit`.

    Parameters
    ----------
    n_pivots : int, default=50
        Maximum number of pivots to select.
    position_dim : int, default=2
        Output dimensionality for the recovered embedding.
    """

    n_pivots: int = 50
    position_dim: int = 2


@register_op
class PivotMDSInit(Op):
    """Initialize positions with landmark shortest paths and SVD.

    Reads
    -----
    No pre-existing ``SolveState`` fields. The op builds adjacency and pivot
    caches on demand.

    Writes
    ------
    ``state.adjacency``, optional ``state.adjacency_weighted``, adjacency
    metadata in ``state.extras``, ``state.pivot_indices``,
    ``state.pivot_distances``, and ``state.pos``.

    Use this when
    -------------
    You want a cheaper approximation than full classical MDS while still
    preserving coarse graph-distance structure.
    """

    name = "pivot_mds_init"
    category = OpCategory.INIT
    writes = (
        "adjacency",
        "adjacency_weighted",
        "extras.adjacency_format",
        "extras.adjacency_directed",
        "extras.adjacency_weighted",
        "pivot_indices",
        "pivot_distances",
        "pos",
    )

    def __init__(self, config: Optional[PivotMDSInitConfig] = None) -> None:
        """Store the pivot-MDS configuration.

        Parameters
        ----------
        config : PivotMDSInitConfig, optional
            Pivot-MDS initialization configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or PivotMDSInitConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize ``state.pos`` from pivot shortest-path distances.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with pivot caches and Pivot-MDS positions.

        Raises
        ------
        ValueError
            If ``n_pivots`` is not positive.
        """
        if self.config.n_pivots <= 0:
            raise ValueError("PivotMDSInit n_pivots must be positive.")
        if _maybe_set_empty_or_single_positions(problem=problem, state=state, ctx=ctx):
            return state

        BuildAdjacency(BuildAdjacencyConfig(weighted=problem.edge_weights is not None)).apply(
            problem, state, ctx
        )
        PivotSelection(PivotSelectionConfig(n_pivots=self.config.n_pivots)).apply(
            problem,
            state,
            ctx,
        )
        PivotDistanceQueries().apply(problem, state, ctx)

        if state.pivot_distances is None:
            raise RuntimeError("PivotDistanceQueries did not populate state.pivot_distances.")

        state.pos = _pivot_mds_coordinates(state.pivot_distances, dim=self.config.position_dim).to(
            device=_target_device(problem, ctx)
        )
        return state


@dataclass(frozen=True)
class FromAlgorithmInitConfig:
    """Configuration for :class:`FromAlgorithmInit`.

    Parameters
    ----------
    algorithm : str, default="fr"
        Inner classic layout algorithm short name.
    inner_config : dict[str, Any] | None, default=None
        Optional keyword arguments forwarded to the inner algorithm.
    inner_steps : int, default=50
        Iteration budget forwarded as ``steps`` when the inner algorithm
        supports it.
    """

    algorithm: str = "fr"
    inner_config: Optional[Dict[str, Any]] = None
    inner_steps: int = 50


@register_op
class FromAlgorithmInit(Op):
    """Initialize positions by delegating to another classic layout algorithm.

    Reads
    -----
    No ``SolveState`` fields.

    Writes
    ------
    ``state.pos`` from the delegated layout result.

    Use this when
    -------------
    You want to reuse an existing classic algorithm as an initializer inside a
    composable op pipeline.
    """

    name = "from_algorithm_init"
    category = OpCategory.INIT
    writes = ("pos",)

    def __init__(self, config: Optional[FromAlgorithmInitConfig] = None) -> None:
        """Store the delegation configuration.

        Parameters
        ----------
        config : FromAlgorithmInitConfig, optional
            Delegation configuration.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or FromAlgorithmInitConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize ``state.pos`` by running another layout algorithm.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with delegated initialization positions.

        Raises
        ------
        ValueError
            If ``inner_steps`` is negative.
        """
        if self.config.inner_steps < 0:
            raise ValueError("FromAlgorithmInit inner_steps must be non-negative.")
        if _maybe_set_empty_or_single_positions(problem=problem, state=state, ctx=ctx):
            return state

        layout_fn = _resolve_layout_algorithm(self.config.algorithm)
        positions = _call_inner_layout(
            layout_fn=layout_fn,
            problem=problem,
            inner_config=self.config.inner_config,
            inner_steps=self.config.inner_steps,
        )
        state.pos = positions.to(dtype=torch.float32, device=_target_device(problem, ctx))
        return state


@dataclass(frozen=True)
class NativeEngineInitConfig:
    """Configuration for :class:`NativeEngineInit`.

    Parameters
    ----------
    node_sep : float, default=25.0
        Horizontal separation target passed to ``init_positions()``.
    rank_sep : float, default=50.0
        Vertical separation target passed to ``init_positions()``.
    device : str or None, default=None
        Explicit device for initialized positions and layer metadata. When
        ``None``, derive the device from the runtime context.
    verbose : bool, default=False
        Forward verbose layering/projection messages to the native init path.
    layer_assignments : torch.Tensor or None, default=None
        Optional precomputed layer IDs with shape ``[N]``.
    prebuilt_layer_index : LayerIndex or None, default=None
        Optional precomputed layer index to reuse directly.
    """

    node_sep: float = 25.0
    rank_sep: float = 50.0
    device: Optional[str] = None
    verbose: bool = False
    layer_assignments: Optional[torch.Tensor] = None
    prebuilt_layer_index: Optional[LayerIndex] = None


@register_op
@dataclass(frozen=True)
class NativeEngineInit(Op):
    """Initialize the native engine positions and layered metadata.

    Notes
    -----
    This op mirrors the monolithic native engine's initialization phase:
    it seeds ``state.pos`` with ``init_positions()`` when a warm start is not
    already present, then prepares ``state.layers`` and ``state.layer_index``
    for the layer-aware losses and projections used later in the pipeline.
    """

    config: NativeEngineInitConfig = field(default_factory=NativeEngineInitConfig)

    name: ClassVar[str] = "native_engine_init"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ("pos", "layers", "layer_index")
    requires: ClassVar[Tuple[str, ...]] = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate native-engine initialization outputs on the solve state.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state that may already carry warm-start positions.
        ctx : RuntimeContext
            Execution infrastructure that may request a target device.

        Returns
        -------
        SolveState
            State with ``pos``, ``layers``, and ``layer_index`` populated.

        Raises
        ------
        ValueError
            If ``problem.node_sizes`` is missing.
        """
        target_device = torch.device(self.config.device or _target_device(problem, ctx))
        if problem.node_sizes is None:
            raise ValueError("NativeEngineInit requires problem.node_sizes to be set.")

        node_sizes = problem.node_sizes.to(device=target_device, dtype=torch.float32)
        if state.pos is None:
            state.pos = init_positions(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                node_sizes=node_sizes,
                node_sep=self.config.node_sep,
                rank_sep=self.config.rank_sep,
                device=str(target_device),
                verbose=self.config.verbose,
            )
        else:
            state.pos = state.pos.to(device=target_device, dtype=torch.float32)

        if self.config.prebuilt_layer_index is not None:
            prebuilt = self.config.prebuilt_layer_index
            state.layer_index = (
                prebuilt
                if prebuilt.node_to_layer.device == target_device
                else build_layer_index(
                    prebuilt.node_to_layer,
                    device=str(target_device),
                    verbose=self.config.verbose,
                )
            )
            state.layers = state.layer_index.node_to_layer
            return state

        if self.config.layer_assignments is not None:
            layers = self.config.layer_assignments.to(device=target_device, dtype=torch.long)
            state.layers = layers
            state.layer_index = build_layer_index(
                layers,
                device=str(target_device),
                verbose=self.config.verbose,
            )
            return state

        if problem.edge_index.numel() == 0:
            state.layers = None
            state.layer_index = None
            return state

        layering_device = "cuda" if torch.cuda.is_available() else "cpu"
        layers = longest_path_layering(
            problem.edge_index,
            problem.num_nodes,
            device=layering_device,
            verbose=self.config.verbose,
        )
        if not isinstance(layers, torch.Tensor):
            layers = torch.tensor(layers, dtype=torch.long)
        layers = layers.to(device=target_device, dtype=torch.long)
        state.layers = layers
        state.layer_index = build_layer_index(
            layers,
            device=str(target_device),
            verbose=self.config.verbose,
        )
        return state


@dataclass(frozen=True)
class FamilyConditionalInitConfig:
    """Configuration for :class:`FamilyConditionalInit`.

    Sprint 1 (2026-04-22): dispatches between `NativeEngineInit` (topological +
    barycenter) and `SpectralInit` based on graph structure. Layered, tree,
    and grid families get the structured init; connectivity-dominant graphs
    (low num_layers relative to N, or non-planar hint) get spectral.

    Parameters
    ----------
    native_config : NativeEngineInitConfig
        Config forwarded when the structured init wins.
    spectral_config : SpectralInitConfig
        Config forwarded when spectral wins.
    structure : GraphStructure or None
        Precomputed structure. When None, dispatches based on edge_index
        shape via ``classify_graph`` inside ``apply``.
    layer_ratio_threshold : float, default=0.2
        num_layers / num_nodes threshold. BELOW = spectral; ABOVE = native.
        Rationale: a "tall" DAG has many layers relative to nodes (chain =
        1.0, tree = log_b(N)/N). Flat/cyclic graphs like small_world have
        very few layers relative to nodes, so the spectral embedding
        gives a better starting geometry than longest-path layering.
    """

    native_config: Optional[NativeEngineInitConfig] = None
    spectral_config: Optional[SpectralInitConfig] = None
    structure: Optional[GraphStructure] = None
    layer_ratio_threshold: float = 0.2


@register_op
class FamilyConditionalInit(Op):
    """Dispatch initializer based on detected graph family.

    Sprint 1 weak-family fix (2026-04-22): the Sprint 0.5 held-out baseline
    showed undirected-ish families (small_world, bipartite, erdos_renyi,
    random_geometric) scoring 22-50 on composite while DAG families scored
    65-97. Part of that gap is rubric-based (DAG-weighted composite), but
    spectral init shifts undirected scores up because the initial geometry
    reflects connectivity instead of a forced topological stack.
    """

    name = "family_conditional_init"
    category = OpCategory.INIT
    writes = ("pos", "layers", "layer_index")

    def __init__(self, config: Optional[FamilyConditionalInitConfig] = None) -> None:
        self.config = config or FamilyConditionalInitConfig()

    def _use_spectral(self, problem: LayoutProblem) -> bool:
        structure = self.config.structure
        if structure is None:
            from dagua.layout.graph_classify import classify_graph

            structure = classify_graph(problem.edge_index, problem.num_nodes)
        # Structured families (TREE, CHAIN, BIPARTITE_DAG, WIDE_LAYERED, GRID)
        # always use native.
        if structure.family != GraphFamily.GENERAL:
            return False
        # GENERAL: check layer ratio. Flat graphs -> spectral.
        if problem.num_nodes == 0:
            return False
        layer_ratio = structure.num_layers / max(1, problem.num_nodes)
        return layer_ratio < self.config.layer_ratio_threshold

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Dispatch to native or spectral init based on structure.

        When spectral wins, we still layer the graph afterward so
        downstream layer-aware losses have valid layer_index; layers are
        derived from the spectral y-coordinate (quantile-bucketed into
        sqrt(N) layers).
        """
        if self._use_spectral(problem):
            spectral = SpectralInit(self.config.spectral_config)
            state = spectral.apply(problem, state, ctx)
            # Build fake layers from spectral y so layer-aware ops don't break.
            if state.pos is not None and problem.num_nodes > 0:
                y = state.pos[:, 1]
                num_layers = max(1, int(problem.num_nodes**0.5))
                if num_layers > 1:
                    quantiles = torch.linspace(0, 1, num_layers + 1, device=y.device)
                    thresholds = torch.quantile(y, quantiles[1:-1])
                    layers = torch.bucketize(y, thresholds).to(dtype=torch.long)
                else:
                    layers = torch.zeros(problem.num_nodes, dtype=torch.long, device=y.device)
                state.layers = layers
                state.layer_index = build_layer_index(
                    layers,
                    device=str(y.device),
                )
            return state
        native = NativeEngineInit(self.config.native_config or NativeEngineInitConfig())
        return native.apply(problem, state, ctx)


__all__ = [
    "CircularInit",
    "CircularInitConfig",
    "ClassicalMDSInit",
    "ClassicalMDSInitConfig",
    "DeterministicInit",
    "DeterministicInitConfig",
    "FA2InitializePositions",
    "FamilyConditionalInit",
    "FamilyConditionalInitConfig",
    "FromAlgorithmInit",
    "FromAlgorithmInitConfig",
    "LinLogInitializePositions",
    "NativeEngineInit",
    "NativeEngineInitConfig",
    "PivotMDSInit",
    "PivotMDSInitConfig",
    "RandomNormalInit",
    "RandomNormalInitConfig",
    "RandomUniformInit",
    "RandomUniformInitConfig",
    "SpectralInit",
    "SpectralInitConfig",
    "ValidateFA2Inputs",
    "ValidateFA2InputsConfig",
    "XavierInit",
    "XavierInitConfig",
]
