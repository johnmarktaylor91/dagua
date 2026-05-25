"""fCoSE-style force-directed layout operations.

This module implements the non-compound subset of Cytoscape fCoSE: a
spectral/BFS-distance initialization followed by CoSE-like spring embedding,
gravity, optional hard pins, and Barnes-Hut repulsion for larger graphs.
"""

from __future__ import annotations

import math
from collections import deque
from dataclasses import dataclass, field
from typing import ClassVar, Optional

import numpy as np
import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import (
    build_undirected_adjacency,
    normalize_positions,
    shortest_path_distances,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_MIN_DISTANCE = 1.0e-6
_DEFAULT_NODE_SEPARATION = 75.0
_DEFAULT_IDEAL_EDGE_LENGTH = 50.0
_DEFAULT_EDGE_ELASTICITY = 0.45
_DEFAULT_NODE_REPULSION = 4500.0
_DEFAULT_GRAVITY = 0.25
_DEFAULT_GRAVITY_RANGE = 3.8
_DEFAULT_INITIAL_ENERGY = 0.3
_DEFAULT_COOLING = 0.99
_PROOF_COOLING = 0.995
_MAX_EXACT_REPULSION_NODES = 512
_MAX_EXACT_INIT_NODES = 2000
_QUADTREE_MAX_DEPTH = 24
_QUADTREE_LEAF_SIZE = 1
_FCOSE_SPRING_EDGES_KEY = "fcose_spring_edges"
_FCOSE_DEGREES_KEY = "fcose_degrees"


@dataclass
class _FCoSEQuadTreeNode:
    """Barnes-Hut quadtree node for fCoSE repulsion.

    Parameters
    ----------
    center : torch.Tensor
        Cell center with shape ``[2]``.
    half_width : float
        Half-width of the square cell.
    indices : list[int]
        Point indices contained by this cell.
    level : int
        Tree depth.
    mass : float
        Number of points represented by this cell.
    center_of_mass : torch.Tensor | None
        Mean point coordinate with shape ``[2]``.
    children : list[_FCoSEQuadTreeNode]
        Child quadrants. Empty for leaves.
    """

    center: torch.Tensor
    half_width: float
    indices: list[int]
    level: int
    mass: float = 0.0
    center_of_mass: Optional[torch.Tensor] = None
    children: list["_FCoSEQuadTreeNode"] = field(default_factory=list)


@dataclass(frozen=True)
class FCoSEValidateInputsConfig:
    """Configuration for :class:`FCoSEValidateInputs`.

    Attributes
    ----------
    steps : int
        Number of spring-embedder iterations.
    node_separation : float
        Spectral initialization separation in Cytoscape coordinate units.
    ideal_edge_length : float
        Desired spring length.
    node_repulsion : float
        Pairwise repulsion coefficient.
    edge_elasticity : float
        Spring force coefficient.
    gravity : float
        Gravity coefficient pulling components toward their barycenter.
    """

    steps: int
    node_separation: float
    ideal_edge_length: float
    node_repulsion: float
    edge_elasticity: float
    gravity: float


@register_op
@dataclass(frozen=True)
class FCoSEValidateInputs(Op):
    """Validate fCoSE pipeline inputs."""

    name: ClassVar[str] = "fcose_validate_inputs"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[tuple[str, ...]] = ()
    writes: ClassVar[tuple[str, ...]] = ()
    config: FCoSEValidateInputsConfig

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Validate graph and scalar configuration.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused by this validation op.

        Returns
        -------
        SolveState
            Unmodified solve state.

        Raises
        ------
        ValueError
            If graph shapes or scalar hyperparameters are invalid.
        """
        del ctx
        if problem.num_nodes < 0:
            raise ValueError("num_nodes must be non-negative.")
        if problem.edge_index.ndim != 2 or problem.edge_index.shape[0] != 2:
            raise ValueError("edge_index must have shape [2, E].")
        if self.config.steps < 0:
            raise ValueError("steps must be non-negative.")
        positive_values = {
            "node_separation": self.config.node_separation,
            "ideal_edge_length": self.config.ideal_edge_length,
            "node_repulsion": self.config.node_repulsion,
            "edge_elasticity": self.config.edge_elasticity,
        }
        for name, value in positive_values.items():
            if value <= 0.0:
                raise ValueError(f"{name} must be positive.")
        if self.config.gravity < 0.0:
            raise ValueError("gravity must be non-negative.")
        return state


@dataclass(frozen=True)
class FCoSEInitialPlacementConfig:
    """Configuration for :class:`FCoSEInitialPlacement`.

    Attributes
    ----------
    node_separation : float
        Distance multiplier applied to graph shortest paths.
    randomize : bool
        Whether to ignore incoming positions and compute a fresh placement.
    max_exact_nodes : int
        Node cap for exact all-pairs-distance initialization.
    """

    node_separation: float = _DEFAULT_NODE_SEPARATION
    randomize: bool = True
    max_exact_nodes: int = _MAX_EXACT_INIT_NODES


@register_op
@dataclass(frozen=True)
class FCoSEInitialPlacement(Op):
    """Initialize positions with a spectral-style shortest-path embedding."""

    name: ClassVar[str] = "fcose_initial_placement"
    category: ClassVar[OpCategory] = OpCategory.INIT
    reads: ClassVar[tuple[str, ...]] = ("pos",)
    writes: ClassVar[tuple[str, ...]] = ("pos",)
    config: FCoSEInitialPlacementConfig = field(default_factory=FCoSEInitialPlacementConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate initial fCoSE positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context that provides the target device.

        Returns
        -------
        SolveState
            State with ``pos`` populated as a ``[N, 2]`` tensor.
        """
        if state.pos is not None and not self.config.randomize:
            state.pos = state.pos.to(device=torch.device(ctx.plan.device), dtype=torch.float32)
            return state

        device = torch.device(ctx.plan.device)
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=device)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32, device=device)
            return state

        if problem.num_nodes <= self.config.max_exact_nodes:
            positions = _distance_embedding(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                edge_weights=problem.edge_weights,
                node_separation=self.config.node_separation,
            )
        else:
            positions = _large_graph_seeded_components(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                seed=problem.seed,
                node_separation=self.config.node_separation,
            )
        state.pos = positions.to(device=device, dtype=torch.float32)
        return state


@dataclass(frozen=True)
class FCoSEPrepareStateConfig:
    """Configuration for :class:`FCoSEPrepareState`.

    Attributes
    ----------
    ideal_edge_length : float
        Desired edge length for all non-compound springs.
    edge_elasticity : float
        Spring elasticity for all edges.
    initial_energy : float
        Initial displacement temperature as a fraction of graph scale.
    quality : str
        fCoSE quality mode, one of ``"draft"``, ``"default"``, or ``"proof"``.
    """

    ideal_edge_length: float = _DEFAULT_IDEAL_EDGE_LENGTH
    edge_elasticity: float = _DEFAULT_EDGE_ELASTICITY
    initial_energy: float = _DEFAULT_INITIAL_ENERGY
    quality: str = "default"


@register_op
@dataclass(frozen=True)
class FCoSEPrepareState(Op):
    """Prepare spring and cooling state for fCoSE refinement."""

    name: ClassVar[str] = "fcose_prepare_state"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    reads: ClassVar[tuple[str, ...]] = ("pos",)
    writes: ClassVar[tuple[str, ...]] = ("temperature", "extras")
    config: FCoSEPrepareStateConfig = field(default_factory=FCoSEPrepareStateConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Prepare unique edge springs and initial temperature.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state with initialized positions.
        ctx : RuntimeContext
            Runtime context. Unused except for API consistency.

        Returns
        -------
        SolveState
            State with fCoSE extras and temperature populated.
        """
        del ctx
        pos = _require_positions(state)
        spring_edges = _unique_undirected_edges(problem.edge_index, problem.num_nodes, pos.device)
        state.extras[_FCOSE_SPRING_EDGES_KEY] = spring_edges
        state.spring_lengths = torch.full(
            (spring_edges.shape[1],),
            float(self.config.ideal_edge_length),
            dtype=pos.dtype,
            device=pos.device,
        )
        state.spring_strengths = torch.full(
            (spring_edges.shape[1],),
            float(self.config.edge_elasticity),
            dtype=pos.dtype,
            device=pos.device,
        )
        state.extras[_FCOSE_DEGREES_KEY] = _degree_from_edges(spring_edges, problem.num_nodes)

        span = float((pos.max(dim=0).values - pos.min(dim=0).values).max().item())
        temperature_base = max(span, float(self.config.ideal_edge_length), _MIN_DISTANCE)
        state.temperature = temperature_base * float(self.config.initial_energy)
        state.force_area = temperature_base * temperature_base
        return state


@dataclass(frozen=True)
class FCoSESpringEmbedderStepConfig:
    """Configuration for one fCoSE spring-embedder step.

    Attributes
    ----------
    node_repulsion : float
        Pairwise repulsion coefficient.
    gravity : float
        Barycenter gravity coefficient.
    gravity_range : float
        Soft radius multiplier before gravity reaches full strength.
    barnes_hut_theta : float
        Barnes-Hut opening threshold for approximate repulsion.
    max_exact_nodes : int
        Use exact pairwise repulsion at or below this node count.
    """

    node_repulsion: float = _DEFAULT_NODE_REPULSION
    gravity: float = _DEFAULT_GRAVITY
    gravity_range: float = _DEFAULT_GRAVITY_RANGE
    barnes_hut_theta: float = 0.8
    max_exact_nodes: int = _MAX_EXACT_REPULSION_NODES


@register_op
@dataclass(frozen=True)
class FCoSESpringEmbedderStep(Op):
    """Apply one CoSE-like spring-embedder force step."""

    name: ClassVar[str] = "fcose_spring_embedder_step"
    category: ClassVar[OpCategory] = OpCategory.FORCE
    reads: ClassVar[tuple[str, ...]] = ("pos", "spring_lengths", "spring_strengths")
    writes: ClassVar[tuple[str, ...]] = ("pos", "forces")
    config: FCoSESpringEmbedderStepConfig = field(default_factory=FCoSESpringEmbedderStepConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Advance positions by one temperature-clamped force update.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused except for API consistency.

        Returns
        -------
        SolveState
            State with updated ``pos`` and current ``forces``.
        """
        del problem, ctx
        pos = _require_positions(state)
        if state.temperature is None:
            raise ValueError("FCoSESpringEmbedderStep requires state.temperature.")

        forces = _fcose_repulsion(
            pos=pos,
            node_repulsion=self.config.node_repulsion,
            theta=self.config.barnes_hut_theta,
            max_exact_nodes=self.config.max_exact_nodes,
        )
        forces = forces + _fcose_spring_forces(state=state)
        forces = forces + _fcose_gravity(
            pos=pos,
            gravity=self.config.gravity,
            gravity_range=self.config.gravity_range,
            ideal_edge_length=_mean_spring_length(state),
        )
        displacement = _temperature_clamped_displacement(forces, state.temperature)
        state.pos = pos + displacement
        state.forces = forces
        return state


@dataclass(frozen=True)
class FCoSEApplyConstraintsConfig:
    """Configuration for :class:`FCoSEApplyConstraints`.

    Attributes
    ----------
    pin_strength : float
        Soft blend factor for pin targets when hard-pin metadata is absent.
    """

    pin_strength: float = 1.0


@register_op
@dataclass(frozen=True)
class FCoSEApplyConstraints(Op):
    """Apply the supported fCoSE user constraints."""

    name: ClassVar[str] = "fcose_apply_constraints"
    category: ClassVar[OpCategory] = OpCategory.PROJECT
    reads: ClassVar[tuple[str, ...]] = ("pos",)
    writes: ClassVar[tuple[str, ...]] = ("pos",)
    config: FCoSEApplyConstraintsConfig = field(default_factory=FCoSEApplyConstraintsConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Project supported fixed-node constraints onto current positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs, including optional flex pins.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused except for API consistency.

        Returns
        -------
        SolveState
            State with pinned nodes restored to their target coordinates.
        """
        del ctx
        if (
            problem.flex is None
            or problem.flex.pin_indices is None
            or problem.flex.pin_targets is None
        ):
            return state
        pos = _require_positions(state)
        pin_indices = problem.flex.pin_indices.to(device=pos.device, dtype=torch.long).flatten()
        pin_targets = problem.flex.pin_targets.to(device=pos.device, dtype=pos.dtype)
        if pin_indices.numel() == 0:
            return state
        strength = max(0.0, min(float(self.config.pin_strength), 1.0))
        pos = pos.clone()
        pos[pin_indices] = pos[pin_indices] * (1.0 - strength) + pin_targets * strength
        state.pos = pos
        return state


@dataclass(frozen=True)
class FCoSECoolConfig:
    """Configuration for fCoSE temperature cooling.

    Attributes
    ----------
    quality : str
        fCoSE quality mode. ``"proof"`` cools more slowly than ``"default"``.
    min_temperature : float
        Early-stop threshold.
    """

    quality: str = "default"
    min_temperature: float = 1.0


@register_op
@dataclass(frozen=True)
class FCoSECool(Op):
    """Apply fCoSE cooling and convergence detection."""

    name: ClassVar[str] = "fcose_cool"
    category: ClassVar[OpCategory] = OpCategory.ANNEAL
    reads: ClassVar[tuple[str, ...]] = ("temperature",)
    writes: ClassVar[tuple[str, ...]] = ("temperature", "converged")
    config: FCoSECoolConfig = field(default_factory=FCoSECoolConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Cool the current fCoSE temperature.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused by this op.

        Returns
        -------
        SolveState
            State with updated temperature and convergence flag.
        """
        del problem, ctx
        if state.temperature is None:
            raise ValueError("FCoSECool requires state.temperature.")
        cooling = _PROOF_COOLING if self.config.quality == "proof" else _DEFAULT_COOLING
        state.temperature *= cooling
        if state.temperature < self.config.min_temperature:
            state.converged = True
        return state


@dataclass(frozen=True)
class FCoSEFinalizeConfig:
    """Configuration for :class:`FCoSEFinalize`.

    Attributes
    ----------
    extent : float | None
        Optional target half-width. ``None`` preserves Cytoscape-like units.
    """

    extent: Optional[float] = None


@register_op
@dataclass(frozen=True)
class FCoSEFinalize(Op):
    """Finalize fCoSE output positions."""

    name: ClassVar[str] = "fcose_finalize"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[tuple[str, ...]] = ("pos",)
    writes: ClassVar[tuple[str, ...]] = ("pos",)
    config: FCoSEFinalizeConfig = field(default_factory=FCoSEFinalizeConfig)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Center and cast final positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs. Unused by this op.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context. Unused by this op.

        Returns
        -------
        SolveState
            State with final ``[N, 2]`` float32 positions.
        """
        del problem, ctx
        pos = _require_positions(state)
        if self.config.extent is not None:
            pos = normalize_positions(pos, extent=float(self.config.extent))
        else:
            pos = pos - pos.mean(dim=0, keepdim=True)
        state.pos = pos.to(dtype=torch.float32)
        return state


def _require_positions(state: SolveState) -> torch.Tensor:
    """Return current positions or raise an actionable error.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If positions are missing.
    """
    if state.pos is None:
        raise ValueError("fCoSE ops require state.pos to be populated.")
    return state.pos


def _distance_embedding(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor],
    node_separation: float,
) -> torch.Tensor:
    """Embed graph shortest-path distances into two dimensions.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor | None
        Optional edge weights with shape ``[E]``.
    node_separation : float
        Coordinate distance multiplier.

    Returns
    -------
    torch.Tensor
        Initial positions with shape ``[N, 2]``.
    """
    distances = shortest_path_distances(edge_index, num_nodes, edge_weights=edge_weights)
    distances = distances * float(node_separation)
    squared = distances * distances
    centering = np.eye(num_nodes, dtype=np.float64) - (np.ones((num_nodes, num_nodes)) / num_nodes)
    gram = -0.5 * centering @ squared @ centering
    eigenvalues, eigenvectors = np.linalg.eigh(gram)
    order = np.argsort(eigenvalues)[::-1]
    coordinates = np.zeros((num_nodes, 2), dtype=np.float64)
    for out_dim, eigen_index in enumerate(order[:2]):
        value = max(float(eigenvalues[eigen_index]), 0.0)
        if value <= 0.0:
            continue
        coordinates[:, out_dim] = eigenvectors[:, eigen_index] * math.sqrt(value)
    if not np.isfinite(coordinates).all() or float(np.abs(coordinates).max()) <= _MIN_DISTANCE:
        coordinates[:, 0] = np.linspace(
            -node_separation,
            node_separation,
            num_nodes,
            dtype=np.float64,
        )
    return torch.tensor(coordinates, dtype=torch.float32)


def _large_graph_seeded_components(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    node_separation: float,
) -> torch.Tensor:
    """Create a scalable component-aware random initialization.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    seed : int
        Random seed.
    node_separation : float
        Coordinate spacing scale.

    Returns
    -------
    torch.Tensor
        Initial position tensor with shape ``[N, 2]``.
    """
    generator = torch.Generator(device="cpu")
    generator.manual_seed(seed)
    positions = torch.randn((num_nodes, 2), generator=generator, dtype=torch.float32)
    positions *= float(node_separation)
    adjacency = build_undirected_adjacency(edge_index=edge_index, num_nodes=num_nodes)
    component_ids = _component_ids(adjacency)
    component_count = max(component_ids) + 1 if component_ids else 0
    if component_count <= 1:
        return positions
    grid_width = max(1, math.ceil(math.sqrt(component_count)))
    for component in range(component_count):
        node_indices = [index for index, value in enumerate(component_ids) if value == component]
        if not node_indices:
            continue
        row = component // grid_width
        column = component % grid_width
        offset = torch.tensor(
            [column * node_separation * 4.0, row * node_separation * 4.0],
            dtype=torch.float32,
        )
        index_tensor = torch.tensor(node_indices, dtype=torch.long)
        positions[index_tensor] += offset
    return positions


def _component_ids(adjacency: list[list[tuple[int, float]]]) -> list[int]:
    """Return connected-component ids for an undirected adjacency list.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Undirected adjacency list.

    Returns
    -------
    list[int]
        Component id for each node.
    """
    component_ids = [-1] * len(adjacency)
    component = 0
    for start in range(len(adjacency)):
        if component_ids[start] >= 0:
            continue
        component_ids[start] = component
        frontier: deque[int] = deque([start])
        while frontier:
            node = frontier.popleft()
            for neighbor, _ in adjacency[node]:
                if component_ids[neighbor] >= 0:
                    continue
                component_ids[neighbor] = component
                frontier.append(neighbor)
        component += 1
    return component_ids


def _unique_undirected_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Return unique non-self-loop undirected edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    device : torch.device
        Output device.

    Returns
    -------
    torch.Tensor
        Unique undirected edge tensor with shape ``[2, E_unique]``.
    """
    if edge_index.numel() == 0:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    edges: set[tuple[int, int]] = set()
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        if (
            source == target
            or source < 0
            or target < 0
            or source >= num_nodes
            or target >= num_nodes
        ):
            continue
        low = min(int(source), int(target))
        high = max(int(source), int(target))
        edges.add((low, high))
    if not edges:
        return torch.empty((2, 0), dtype=torch.long, device=device)
    ordered = torch.tensor(sorted(edges), dtype=torch.long, device=device)
    return ordered.transpose(0, 1).contiguous()


def _degree_from_edges(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute undirected degrees from unique edge pairs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Unique undirected edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Degree vector with shape ``[N]``.
    """
    degree = torch.zeros((num_nodes,), dtype=torch.float32, device=edge_index.device)
    if edge_index.numel() == 0:
        return degree
    ones = torch.ones((edge_index.shape[1],), dtype=degree.dtype, device=degree.device)
    degree.index_add_(0, edge_index[0], ones)
    degree.index_add_(0, edge_index[1], ones)
    return degree


def _fcose_repulsion(
    pos: torch.Tensor,
    node_repulsion: float,
    theta: float,
    max_exact_nodes: int,
) -> torch.Tensor:
    """Compute exact or Barnes-Hut fCoSE repulsive forces.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    node_repulsion : float
        Repulsion coefficient.
    theta : float
        Barnes-Hut opening threshold.
    max_exact_nodes : int
        Exact pairwise threshold.

    Returns
    -------
    torch.Tensor
        Repulsive force tensor with shape ``[N, 2]``.
    """
    if pos.shape[0] <= 1:
        return torch.zeros_like(pos)
    if pos.shape[0] <= max_exact_nodes:
        delta = pos.unsqueeze(1) - pos.unsqueeze(0)
        distance_sq = delta.square().sum(dim=2).clamp_min(_MIN_DISTANCE)
        distance_sq.fill_diagonal_(float("inf"))
        return (float(node_repulsion) * delta / distance_sq.unsqueeze(2)).sum(dim=1)

    tree = _build_quadtree(pos.detach())
    if tree is None:
        return torch.zeros_like(pos)
    forces = torch.zeros_like(pos)
    for index in range(pos.shape[0]):
        forces[index] = _barnes_hut_force(
            node=tree,
            positions=pos,
            index=index,
            theta=theta,
            node_repulsion=node_repulsion,
        )
    return forces


def _build_quadtree(positions: torch.Tensor) -> Optional[_FCoSEQuadTreeNode]:
    """Build a Barnes-Hut quadtree over current positions.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    _FCoSEQuadTreeNode | None
        Root node, or ``None`` when no positions are present.
    """
    if positions.numel() == 0:
        return None
    minimum = positions.min(dim=0).values
    maximum = positions.max(dim=0).values
    center = (minimum + maximum) * 0.5
    span = float((maximum - minimum).max().item())
    half_width = max(span * 0.5, _MIN_DISTANCE)
    return _build_quadtree_node(
        positions=positions,
        indices=list(range(positions.shape[0])),
        center=center,
        half_width=half_width,
        level=0,
    )


def _build_quadtree_node(
    positions: torch.Tensor,
    indices: list[int],
    center: torch.Tensor,
    half_width: float,
    level: int,
) -> _FCoSEQuadTreeNode:
    """Recursively build one Barnes-Hut quadtree node.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    indices : list[int]
        Point indices assigned to this node.
    center : torch.Tensor
        Cell center with shape ``[2]``.
    half_width : float
        Cell half-width.
    level : int
        Current tree depth.

    Returns
    -------
    _FCoSEQuadTreeNode
        Constructed quadtree node.
    """
    node = _FCoSEQuadTreeNode(center=center, half_width=half_width, indices=indices, level=level)
    if not indices:
        node.center_of_mass = center.clone()
        return node

    coords = positions[torch.tensor(indices, dtype=torch.long, device=positions.device)]
    node.mass = float(len(indices))
    node.center_of_mass = coords.mean(dim=0)
    if len(indices) <= _QUADTREE_LEAF_SIZE or level >= _QUADTREE_MAX_DEPTH:
        return node

    child_half_width = half_width * 0.5
    quadrants: list[list[int]] = [[], [], [], []]
    for index in indices:
        point = positions[index]
        horizontal = 1 if float(point[0].item()) >= float(center[0].item()) else 0
        vertical = 1 if float(point[1].item()) >= float(center[1].item()) else 0
        quadrants[vertical * 2 + horizontal].append(index)

    offsets = torch.tensor(
        [
            [-child_half_width, -child_half_width],
            [child_half_width, -child_half_width],
            [-child_half_width, child_half_width],
            [child_half_width, child_half_width],
        ],
        dtype=positions.dtype,
        device=positions.device,
    )
    for quadrant_index, quadrant_indices in enumerate(quadrants):
        if not quadrant_indices:
            continue
        child_center = center + offsets[quadrant_index]
        node.children.append(
            _build_quadtree_node(
                positions=positions,
                indices=quadrant_indices,
                center=child_center,
                half_width=child_half_width,
                level=level + 1,
            )
        )
    return node


def _barnes_hut_force(
    node: _FCoSEQuadTreeNode,
    positions: torch.Tensor,
    index: int,
    theta: float,
    node_repulsion: float,
) -> torch.Tensor:
    """Evaluate Barnes-Hut repulsion on one node.

    Parameters
    ----------
    node : _FCoSEQuadTreeNode
        Current quadtree node.
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    index : int
        Query node index.
    theta : float
        Opening threshold.
    node_repulsion : float
        Repulsion coefficient.

    Returns
    -------
    torch.Tensor
        Force vector with shape ``[2]``.
    """
    if node.mass <= 0.0 or node.center_of_mass is None:
        return torch.zeros(2, dtype=positions.dtype, device=positions.device)
    if len(node.indices) == 1 and node.indices[0] == index:
        return torch.zeros(2, dtype=positions.dtype, device=positions.device)

    delta = positions[index] - node.center_of_mass
    distance = float(torch.linalg.vector_norm(delta).item())
    width = node.half_width * 2.0
    should_open = distance > _MIN_DISTANCE and width / distance < theta
    if node.children and index in node.indices and not should_open:
        total = torch.zeros(2, dtype=positions.dtype, device=positions.device)
        for child in node.children:
            total = total + _barnes_hut_force(child, positions, index, theta, node_repulsion)
        return total
    if node.children and distance > _MIN_DISTANCE and width / distance < theta:
        return float(node_repulsion) * node.mass * delta / max(distance * distance, _MIN_DISTANCE)
    if node.children:
        total = torch.zeros(2, dtype=positions.dtype, device=positions.device)
        for child in node.children:
            total = total + _barnes_hut_force(child, positions, index, theta, node_repulsion)
        return total

    leaf_indices = [point_index for point_index in node.indices if point_index != index]
    if not leaf_indices:
        return torch.zeros(2, dtype=positions.dtype, device=positions.device)
    leaf_tensor = torch.tensor(leaf_indices, dtype=torch.long, device=positions.device)
    leaf_delta = positions[index].unsqueeze(0) - positions[leaf_tensor]
    distance_sq = leaf_delta.square().sum(dim=1).clamp_min(_MIN_DISTANCE)
    return (float(node_repulsion) * leaf_delta / distance_sq.unsqueeze(1)).sum(dim=0)


def _fcose_spring_forces(state: SolveState) -> torch.Tensor:
    """Compute spring forces for current fCoSE edges.

    Parameters
    ----------
    state : SolveState
        Mutable solve state containing fCoSE edge extras.

    Returns
    -------
    torch.Tensor
        Spring force tensor with shape ``[N, 2]``.
    """
    pos = _require_positions(state)
    spring_edges = state.extras.get(_FCOSE_SPRING_EDGES_KEY)
    if not isinstance(spring_edges, torch.Tensor) or spring_edges.numel() == 0:
        return torch.zeros_like(pos)
    if state.spring_lengths is None or state.spring_strengths is None:
        raise ValueError("fCoSE spring forces require spring_lengths and spring_strengths.")

    source = spring_edges[0].to(device=pos.device)
    target = spring_edges[1].to(device=pos.device)
    delta = pos[source] - pos[target]
    distance = torch.linalg.vector_norm(delta, dim=1).clamp_min(_MIN_DISTANCE)
    direction = delta / distance.unsqueeze(1)
    extension = distance - state.spring_lengths.to(device=pos.device, dtype=pos.dtype)
    strength = state.spring_strengths.to(device=pos.device, dtype=pos.dtype)
    edge_force = strength.unsqueeze(1) * extension.unsqueeze(1) * direction
    forces = torch.zeros_like(pos)
    forces.index_add_(0, source, -edge_force)
    forces.index_add_(0, target, edge_force)
    return forces


def _fcose_gravity(
    pos: torch.Tensor,
    gravity: float,
    gravity_range: float,
    ideal_edge_length: float,
) -> torch.Tensor:
    """Compute fCoSE barycenter gravity.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    gravity : float
        Gravity coefficient.
    gravity_range : float
        Range multiplier before gravity is fully active.
    ideal_edge_length : float
        Mean spring length used as the gravity range scale.

    Returns
    -------
    torch.Tensor
        Gravity force tensor with shape ``[N, 2]``.
    """
    if gravity <= 0.0 or pos.shape[0] == 0:
        return torch.zeros_like(pos)
    centered = pos - pos.mean(dim=0, keepdim=True)
    distance = torch.linalg.vector_norm(centered, dim=1).clamp_min(_MIN_DISTANCE)
    range_limit = max(float(gravity_range) * float(ideal_edge_length), _MIN_DISTANCE)
    range_scale = (distance / range_limit).clamp(max=1.0)
    return -float(gravity) * range_scale.unsqueeze(1) * centered


def _temperature_clamped_displacement(forces: torch.Tensor, temperature: float) -> torch.Tensor:
    """Clamp force vectors by the current fCoSE temperature.

    Parameters
    ----------
    forces : torch.Tensor
        Force tensor with shape ``[N, 2]``.
    temperature : float
        Maximum node displacement for this step.

    Returns
    -------
    torch.Tensor
        Displacement tensor with shape ``[N, 2]``.
    """
    lengths = torch.linalg.vector_norm(forces, dim=1).clamp_min(_MIN_DISTANCE)
    scale = torch.clamp(torch.full_like(lengths, float(temperature)) / lengths, max=1.0)
    return forces * scale.unsqueeze(1)


def _mean_spring_length(state: SolveState) -> float:
    """Return the mean active spring length.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    float
        Mean spring length, or the fCoSE default when no springs exist.
    """
    if state.spring_lengths is None or state.spring_lengths.numel() == 0:
        return _DEFAULT_IDEAL_EDGE_LENGTH
    return float(state.spring_lengths.mean().item())


__all__ = [
    "FCoSEApplyConstraints",
    "FCoSEApplyConstraintsConfig",
    "FCoSECool",
    "FCoSECoolConfig",
    "FCoSEFinalize",
    "FCoSEFinalizeConfig",
    "FCoSEInitialPlacement",
    "FCoSEInitialPlacementConfig",
    "FCoSEPrepareState",
    "FCoSEPrepareStateConfig",
    "FCoSESpringEmbedderStep",
    "FCoSESpringEmbedderStepConfig",
    "FCoSEValidateInputs",
    "FCoSEValidateInputsConfig",
]
