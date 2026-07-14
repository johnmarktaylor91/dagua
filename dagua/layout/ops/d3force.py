"""d3-force-compatible composable layout operations."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List, Optional

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_LCG_A = 1_664_525
_LCG_C = 1_013_904_223
_LCG_M = 4_294_967_296
_INITIAL_RADIUS = 10.0
_INITIAL_ANGLE = math.pi * (3.0 - math.sqrt(5.0))
_JIGGLE_SCALE = 1.0e-6
_DEFAULT_ALPHA_MIN = 0.001
_DEFAULT_ALPHA_TARGET = 0.0
_DEFAULT_MANY_BODY_STRENGTH = -30.0
_DEFAULT_LINK_DISTANCE = 30.0
_DEFAULT_VELOCITY_DECAY_FACTOR = 0.6
_DEFAULT_THETA = 0.9


class D3ForceLCG:
    """Linear congruential generator used by d3-force.

    Parameters
    ----------
    seed : int, default=1
        Initial unsigned 32-bit state. d3-force's built-in source starts at
        ``1``; exposing the seed mirrors ``simulation.randomSource``.
    """

    def __init__(self, seed: int = 1) -> None:
        self.state = int(seed) % _LCG_M

    def random(self) -> float:
        """Return the next d3-force LCG value.

        Returns
        -------
        float
            Uniform value in ``[0, 1)`` computed as ``state / 2**32`` after
            the d3-force LCG update.
        """
        self.state = (_LCG_A * self.state + _LCG_C) % _LCG_M
        return self.state / _LCG_M

    def jiggle(self) -> float:
        """Return d3-force's tiny coincident-point perturbation.

        Returns
        -------
        float
            ``(random() - 0.5) * 1e-6``.
        """
        return (self.random() - 0.5) * _JIGGLE_SCALE


def d3force_lcg_values(seed: int = 1, count: int = 20) -> List[float]:
    """Generate d3-force LCG values for tests and verification.

    Parameters
    ----------
    seed : int, default=1
        Initial unsigned 32-bit LCG state.
    count : int, default=20
        Number of values to return.

    Returns
    -------
    list[float]
        First ``count`` generated values.
    """
    rng = D3ForceLCG(seed=seed)
    return [rng.random() for _ in range(count)]


def d3force_phyllotaxis_positions(
    num_nodes: int,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return d3-force's initial phyllotaxis spiral coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of nodes to initialize.
    dtype : torch.dtype, default=torch.float64
        Output tensor dtype.
    device : torch.device, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    resolved_device = torch.device("cpu") if device is None else device
    pos = torch.zeros((num_nodes, 2), dtype=dtype, device=resolved_device)
    for index in range(num_nodes):
        radius = _INITIAL_RADIUS * math.sqrt(0.5 + float(index))
        angle = float(index) * _INITIAL_ANGLE
        pos[index, 0] = radius * math.cos(angle)
        pos[index, 1] = radius * math.sin(angle)
    return pos


def _edge_pairs(edge_index: torch.Tensor) -> list[tuple[int, int]]:
    """Convert edge-index tensor to stable Python edge pairs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    list[tuple[int, int]]
        Edge pairs in input column order.
    """
    if edge_index.numel() == 0:
        return []
    return [(int(source), int(target)) for source, target in edge_index.cpu().t().tolist()]


@dataclass(frozen=True)
class D3ForceConfig:
    """Configuration for d3-force-compatible operations.

    Parameters
    ----------
    ticks : int, default=300
        Number of simulation ticks.
    seed : int, default=1
        LCG seed. ``1`` matches d3-force's default source.
    many_body_strength : float, default=-30.0
        Constant charge strength for ``forceManyBody``.
    link_distance : float, default=30.0
        Constant link distance for ``forceLink``.
    link_iterations : int, default=1
        Number of link relaxation passes per tick.
    velocity_decay_factor : float, default=0.6
        Internal multiplier used during velocity Verlet integration. This is
        d3's ``1 - simulation.velocityDecay()`` value.
    theta : float, default=0.9
        Barnes-Hut theta exposed for API parity. The current op uses direct
        pairwise n-body evaluation and records this as a named fidelity gap.
    center : bool, default=True
        Whether to apply ``forceCenter(0, 0)``.
    """

    ticks: int = 300
    seed: int = 1
    many_body_strength: float = _DEFAULT_MANY_BODY_STRENGTH
    link_distance: float = 30.0
    link_iterations: int = 1
    velocity_decay_factor: float = _DEFAULT_VELOCITY_DECAY_FACTOR
    theta: float = _DEFAULT_THETA
    center: bool = True


@register_op
class D3ForceInitialize(Op):
    """Initialize positions, velocities, alpha, RNG, and link metadata."""

    name = "d3force_initialize"
    category = OpCategory.INIT
    writes = ("pos", "extras")

    def __init__(self, config: D3ForceConfig, dtype: torch.dtype = torch.float64) -> None:
        self.config = config
        self.dtype = dtype

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Initialize d3-force simulation state.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable state receiving ``pos`` and d3 extras.
        ctx : RuntimeContext
            Execution context providing the requested device.

        Returns
        -------
        SolveState
            State populated with d3-force working fields.
        """
        device = torch.device(ctx.plan.device or "cpu")
        state.pos = d3force_phyllotaxis_positions(problem.num_nodes, self.dtype, device)
        state.extras["d3force_vx"] = [0.0] * problem.num_nodes
        state.extras["d3force_vy"] = [0.0] * problem.num_nodes
        state.extras["d3force_alpha"] = 1.0
        state.extras["d3force_alpha_decay"] = 1.0 - math.pow(_DEFAULT_ALPHA_MIN, 1.0 / 300.0)
        state.extras["d3force_rng"] = D3ForceLCG(seed=self.config.seed)
        state.extras["d3force_edges"] = _edge_pairs(problem.edge_index)
        state.extras["d3force_link_count"] = self._link_counts(problem)
        return state

    def _link_counts(self, problem: LayoutProblem) -> list[int]:
        """Return d3-force link endpoint counts.

        Parameters
        ----------
        problem : LayoutProblem
            Graph inputs containing edge-index tensor.

        Returns
        -------
        list[int]
            Per-node incident link counts.
        """
        counts = [0] * problem.num_nodes
        for source, target in _edge_pairs(problem.edge_index):
            counts[source] += 1
            counts[target] += 1
        return counts


@register_op
class D3ForceUpdateAlpha(Op):
    """Apply d3-force alpha decay for one tick."""

    name = "d3force_update_alpha"
    category = OpCategory.ANNEAL
    reads = ("extras",)
    writes = ("extras",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Advance simulation alpha.

        Parameters
        ----------
        problem : LayoutProblem
            Unused immutable graph inputs.
        state : SolveState
            State containing d3-force alpha extras.
        ctx : RuntimeContext
            Unused execution context.

        Returns
        -------
        SolveState
            State with updated ``d3force_alpha``.
        """
        del problem, ctx
        alpha = float(state.extras["d3force_alpha"])
        alpha_decay = float(state.extras["d3force_alpha_decay"])
        state.extras["d3force_alpha"] = alpha + (_DEFAULT_ALPHA_TARGET - alpha) * alpha_decay
        return state


@register_op
class D3ForceLink(Op):
    """Apply d3-force ``forceLink`` velocity updates."""

    name = "d3force_link"
    category = OpCategory.FORCE
    reads = ("pos", "extras")
    writes = ("extras",)

    def __init__(self, config: D3ForceConfig) -> None:
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Relax links in d3-force edge order.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            State with positions, velocities, and link metadata.
        ctx : RuntimeContext
            Unused execution context.

        Returns
        -------
        SolveState
            State with updated velocity extras.
        """
        del problem, ctx
        if state.pos is None:
            return state
        pos = state.pos.detach().cpu().numpy()
        vx = state.extras["d3force_vx"]
        vy = state.extras["d3force_vy"]
        rng: D3ForceLCG = state.extras["d3force_rng"]
        alpha = float(state.extras["d3force_alpha"])
        counts = state.extras["d3force_link_count"]
        edges: list[tuple[int, int]] = state.extras["d3force_edges"]
        for _ in range(max(0, int(self.config.link_iterations))):
            for source, target in edges:
                x = float(pos[target, 0]) + vx[target] - float(pos[source, 0]) - vx[source]
                y = float(pos[target, 1]) + vy[target] - float(pos[source, 1]) - vy[source]
                if x == 0.0:
                    x = rng.jiggle()
                if y == 0.0:
                    y = rng.jiggle()
                length = math.sqrt(x * x + y * y)
                strength = 1.0 / float(min(counts[source], counts[target]))
                scale = (length - self.config.link_distance) / length * alpha * strength
                dx = x * scale
                dy = y * scale
                bias = counts[source] / float(counts[source] + counts[target])
                vx[target] -= dx * bias
                vy[target] -= dy * bias
                source_bias = 1.0 - bias
                vx[source] += dx * source_bias
                vy[source] += dy * source_bias
        return state


@register_op
class D3ForceManyBody(Op):
    """Apply d3-force-compatible direct many-body velocity updates."""

    name = "d3force_many_body"
    category = OpCategory.FORCE
    reads = ("pos", "extras")
    writes = ("extras",)

    def __init__(self, config: D3ForceConfig) -> None:
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply n-body repulsion to every ordered node pair.

        Parameters
        ----------
        problem : LayoutProblem
            Graph inputs supplying the node count.
        state : SolveState
            State with positions, velocities, alpha, and RNG.
        ctx : RuntimeContext
            Unused execution context.

        Returns
        -------
        SolveState
            State with updated velocity extras.

        Notes
        -----
        d3-force uses a Barnes-Hut quadtree. This direct evaluator matches
        the leaf force law and RNG discipline, but not internal-cell
        approximation order. Fidelity reports name this as the residual when
        full layouts diverge.
        """
        del ctx
        if state.pos is None:
            return state
        pos = state.pos.detach().cpu().numpy()
        vx = state.extras["d3force_vx"]
        vy = state.extras["d3force_vy"]
        rng: D3ForceLCG = state.extras["d3force_rng"]
        alpha = float(state.extras["d3force_alpha"])
        strength = float(self.config.many_body_strength)
        for node in range(problem.num_nodes):
            for other in range(problem.num_nodes):
                if other == node:
                    continue
                x = float(pos[other, 0]) - float(pos[node, 0])
                y = float(pos[other, 1]) - float(pos[node, 1])
                length2 = x * x + y * y
                if x == 0.0:
                    x = rng.jiggle()
                    length2 += x * x
                if y == 0.0:
                    y = rng.jiggle()
                    length2 += y * y
                if length2 < 1.0:
                    length2 = math.sqrt(length2)
                scale = strength * alpha / length2
                vx[node] += x * scale
                vy[node] += y * scale
        return state


@register_op
class D3ForceCenter(Op):
    """Apply d3-force ``forceCenter(0, 0)``."""

    name = "d3force_center"
    category = OpCategory.POSTPROCESS
    reads = ("pos",)
    writes = ("pos",)

    def __init__(self, enabled: bool = True) -> None:
        self.enabled = enabled

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Shift the current centroid to the origin.

        Parameters
        ----------
        problem : LayoutProblem
            Graph inputs supplying the node count.
        state : SolveState
            State containing position tensor ``[N, 2]``.
        ctx : RuntimeContext
            Unused execution context.

        Returns
        -------
        SolveState
            State with centered positions when enabled.
        """
        del ctx
        if not self.enabled or state.pos is None or problem.num_nodes == 0:
            return state
        state.pos = state.pos - state.pos.mean(dim=0, keepdim=True)
        return state


@register_op
class D3ForceIntegrate(Op):
    """Apply d3-force velocity Verlet integration."""

    name = "d3force_integrate"
    category = OpCategory.OPTIMIZE
    reads = ("pos", "extras")
    writes = ("pos", "extras")

    def __init__(self, config: D3ForceConfig) -> None:
        self.config = config

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Advance positions from velocity state.

        Parameters
        ----------
        problem : LayoutProblem
            Graph inputs supplying the node count.
        state : SolveState
            State with position tensor and velocity lists.
        ctx : RuntimeContext
            Unused execution context.

        Returns
        -------
        SolveState
            State with integrated positions and decayed velocities.
        """
        del ctx
        if state.pos is None:
            return state
        vx = state.extras["d3force_vx"]
        vy = state.extras["d3force_vy"]
        for node in range(problem.num_nodes):
            vx[node] *= self.config.velocity_decay_factor
            vy[node] *= self.config.velocity_decay_factor
            state.pos[node, 0] += vx[node]
            state.pos[node, 1] += vy[node]
        return state
