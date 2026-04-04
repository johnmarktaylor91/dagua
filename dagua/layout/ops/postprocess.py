"""Post-layout operations for coordinate cleanup and display transforms."""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from math import sqrt
from typing import DefaultDict, Dict, List, Optional, Sequence

import numpy as np
import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.graph_utils import (
    layout_extent as _layout_extent,
)
from dagua.layout.ops.graph_utils import (
    normalize_positions as _normalize_positions,
)
from dagua.layout.ops.graph_utils import (
    rescale_layout as _rescale_layout,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_MIN_SPAN = 1.0e-6
_OUTPUT_SCALE_FACTOR = 50.0


def _require_positions(state: SolveState, op_name: str) -> torch.Tensor:
    """Return the current position tensor or raise a descriptive error.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    op_name : str
        Name of the operation requesting positions.

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


def _centered_positions(positions: torch.Tensor) -> torch.Tensor:
    """Return positions translated to zero mean.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Centered position tensor with shape ``[N, 2]``.
    """
    if positions.shape[0] == 0:
        return positions.clone()
    return positions - positions.mean(dim=0, keepdim=True)


def _max_abs_coordinate(positions: torch.Tensor) -> float:
    """Return the maximum absolute coordinate magnitude.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    float
        Maximum absolute coordinate value, or ``0.0`` for empty tensors.
    """
    if positions.numel() == 0:
        return 0.0
    return float(positions.abs().max().item())


def _normalize_extent(
    problem: LayoutProblem,
    extent_fn: str,
    node_size_scale: float,
) -> float:
    """Compute the target half-width for normalized coordinates.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    extent_fn : str
        Extent rule name.
    node_size_scale : float
        Multiplier applied to the node-size-derived extent fallback.

    Returns
    -------
    float
        Target half-width after normalization.

    Raises
    ------
    ValueError
        If ``extent_fn`` is unsupported.
    """
    sqrt_n = sqrt(float(max(problem.num_nodes, 1)))
    if extent_fn == "sqrt_n_times_5":
        extent = sqrt_n * 5.0
    elif extent_fn == "sqrt_n_times_50":
        extent = sqrt_n * 50.0
    else:
        raise ValueError(f"Unsupported NormalizePositions extent_fn: {extent_fn}")

    if problem.node_sizes is None or problem.node_sizes.numel() == 0:
        return max(extent, 1.0)

    max_size = float(problem.node_sizes.to(dtype=torch.float32).max().item())
    sized_extent = max_size * sqrt_n * node_size_scale
    return max(extent, sized_extent, 1.0)


def _fallback_normalized_positions(
    positions: torch.Tensor,
    extent: float,
) -> torch.Tensor:
    """Create a deterministic normalized fallback for degenerate layouts.

    Parameters
    ----------
    positions : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    extent : float
        Target half-width after normalization.

    Returns
    -------
    torch.Tensor
        Position tensor with at least one non-zero span axis when possible.
    """
    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    fallback = torch.zeros_like(positions)
    fallback[:, 0] = torch.linspace(
        -extent,
        extent,
        steps=positions.shape[0],
        device=positions.device,
        dtype=positions.dtype,
    )
    return fallback


def _unique_children_in_order(children: Sequence[int]) -> List[int]:
    """Return children with duplicates removed while preserving order.

    Parameters
    ----------
    children : sequence[int]
        Child node indices gathered from outgoing edges.

    Returns
    -------
    list[int]
        Unique child node indices in first-seen order.
    """
    seen: Dict[int, None] = {}
    return [child for child in children if not (child in seen or seen.setdefault(child, None))]


@register_op
class CenterPositions(Op):
    """Translate positions so their centroid is at the origin."""

    name = "center_positions"
    category = OpCategory.POSTPROCESS
    reads = ("pos",)
    writes = ("pos",)
    requires = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Center the position tensor in-place on its existing device.

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
            State with centered positions.
        """
        _ = problem, ctx
        positions = _require_positions(state=state, op_name=self.name)
        state.pos = _centered_positions(positions)
        return state


@dataclass(frozen=True)
class ScalePositionsConfig:
    """Configuration for :class:`ScalePositions`.

    Parameters
    ----------
    method : str, default="max_abs"
        Scaling mode. ``"max_abs"`` divides by the maximum absolute
        coordinate before multiplying by ``factor``. ``"factor"`` applies
        the multiplicative factor directly.
    factor : float, default=1.0
        Output scale factor.
    """

    method: str = "max_abs"
    factor: float = 1.0


@register_op
class ScalePositions(Op):
    """Scale coordinates either by a fixed factor or to a target max extent."""

    name = "scale_positions"
    category = OpCategory.POSTPROCESS
    reads = ("pos",)
    writes = ("pos",)
    requires = ("pos",)

    def __init__(self, config: Optional[ScalePositionsConfig] = None) -> None:
        """Store the scaling configuration.

        Parameters
        ----------
        config : ScalePositionsConfig, optional
            Scaling configuration. Defaults to max-absolute normalization.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or ScalePositionsConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Scale the position tensor according to ``self.config``.

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
            State with scaled positions.

        Raises
        ------
        ValueError
            If the configured scaling method is unsupported.
        """
        _ = problem, ctx
        positions = _require_positions(state=state, op_name=self.name)
        if self.config.method == "factor":
            state.pos = positions * self.config.factor
            return state

        if self.config.method != "max_abs":
            raise ValueError(f"Unsupported ScalePositions method: {self.config.method}")

        limit = _max_abs_coordinate(positions)
        if limit <= 0.0:
            state.pos = positions.clone()
            return state
        state.pos = positions * (self.config.factor / limit)
        return state


@register_op
class FRFinalizePositions(Op):
    """Apply the FR-specific final centering, scaling, and float32 cast."""

    name = "fr_finalize_positions"
    category = OpCategory.POSTPROCESS
    reads: tuple[str, ...] = ("pos",)
    writes: tuple[str, ...] = ("pos",)
    requires: tuple[str, ...] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compose registered postprocess ops to match legacy FR finalization.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable layout state containing ``state.pos``.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with centered, scaled, float32 positions.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        _ = ctx
        if state.pos is None:
            raise ValueError("FRFinalizePositions requires state.pos to be set.")

        state = CenterPositions().apply(problem=problem, state=state, ctx=ctx)
        state = ScalePositions(
            ScalePositionsConfig(
                method="max_abs",
                factor=(sqrt(float(max(problem.num_nodes, 1))) * _OUTPUT_SCALE_FACTOR),
            ),
        ).apply(problem=problem, state=state, ctx=ctx)
        state.pos = state.pos.to(dtype=torch.float32)
        return state


@register_op
class GraphOptFinalizePositions(Op):
    """Cast GraphOpt positions to the classic output dtype and device."""

    name: str = "graphopt_finalize_positions"
    category: OpCategory = OpCategory.POSTPROCESS
    reads: tuple[str, ...] = ("pos",)
    writes: tuple[str, ...] = ("pos",)
    requires: tuple[str, ...] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Cast final GraphOpt positions onto the resolved output device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing final ``state.pos``.
        ctx : RuntimeContext
            Execution context.

        Returns
        -------
        SolveState
            State with ``state.pos`` as ``float32`` on the output device.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        _ = ctx
        if state.pos is None:
            raise ValueError("GraphOptFinalizePositions requires state.pos to be set.")

        output_device = _layout_device(edge_index=problem.edge_index, node_sizes=problem.node_sizes)
        if state.pos.numel() == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            return state

        state.pos = state.pos.to(dtype=torch.float32, device=output_device)
        return state


@register_op
class LinLogFinalizePositions(Op):
    """Finalize coordinates exactly as classic ``layout_linlog``."""

    name = "linlog_finalize_positions"
    category = OpCategory.POSTPROCESS
    reads: tuple[str, ...] = ("pos",)
    writes: tuple[str, ...] = ("pos",)
    requires: tuple[str, ...] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Normalize and cast final LinLog positions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable layout state containing final coordinates.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with final coordinates on the output device.
        """
        del ctx
        if state.pos is None:
            raise ValueError("LinLogFinalizePositions requires state.pos to be set.")

        output_device = _layout_device(edge_index=problem.edge_index, node_sizes=problem.node_sizes)
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32, device=output_device)
            return state

        extent = _layout_extent(num_nodes=problem.num_nodes, node_sizes=problem.node_sizes)
        normalized = _normalize_positions(state.pos.detach(), extent=extent)
        state.pos = normalized.to(dtype=torch.float32, device=output_device)
        return state


@dataclass(frozen=True)
class NormalizePositionsConfig:
    """Configuration for :class:`NormalizePositions`.

    Parameters
    ----------
    extent_fn : str, default="sqrt_n_times_5"
        Rule used to compute the target half-width.
    node_size_scale : float, default=2.0
        Multiplier for the node-size-derived extent fallback when node sizes
        are available on the problem.
    """

    extent_fn: str = "sqrt_n_times_5"
    node_size_scale: float = 2.0


@register_op
class NormalizePositions(Op):
    """Center and scale positions into a stable extent."""

    name = "normalize_positions"
    category = OpCategory.POSTPROCESS
    reads = ("pos",)
    writes = ("pos",)
    requires = ("pos",)

    def __init__(self, config: Optional[NormalizePositionsConfig] = None) -> None:
        """Store the normalization configuration.

        Parameters
        ----------
        config : NormalizePositionsConfig, optional
            Normalization configuration. Defaults to the MDS-style extent.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or NormalizePositionsConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Center and scale coordinates to the configured extent.

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
            State with normalized positions.
        """
        _ = ctx
        positions = _require_positions(state=state, op_name=self.name)
        if positions.shape[0] <= 1:
            state.pos = torch.zeros_like(positions)
            return state

        extent = _normalize_extent(
            problem=problem,
            extent_fn=self.config.extent_fn,
            node_size_scale=self.config.node_size_scale,
        )
        centered = _centered_positions(positions)
        span = _max_abs_coordinate(centered)
        if span < _MIN_SPAN:
            state.pos = _fallback_normalized_positions(positions=centered, extent=extent)
            return state

        state.pos = centered * (extent / span)
        return state


_KK_TRACE_KEY = "kk_traces"


@register_op
class KamadaKawaiFinalizePositions(Op):
    """Scale final Kamada-Kawai coordinates and move traces to output device."""

    name: str = "kamada_kawai_finalize_positions"
    category: OpCategory = OpCategory.POSTPROCESS
    reads: tuple[str, ...] = ("pos",)
    writes: tuple[str, ...] = ("pos", "extras")
    requires: tuple[str, ...] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Normalize solved Kamada-Kawai coordinates and output artifacts.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing solved coordinates.
        ctx : RuntimeContext
            Execution context.

        Returns
        -------
        SolveState
            State with float32 coordinates and traces on output device.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        _ = ctx
        if state.pos is None:
            raise ValueError("KamadaKawaiFinalizePositions requires state.pos to be set.")

        output_device = _layout_device(edge_index=problem.edge_index, node_sizes=problem.node_sizes)
        traces = state.extras.get(_KK_TRACE_KEY, [])
        state.extras[_KK_TRACE_KEY] = [trace.to(device=output_device) for trace in traces]

        if state.pos.shape[0] <= 1 or state.pos.numel() == 0:
            state.pos = state.pos.to(dtype=torch.float32, device=output_device)
            return state

        state.pos = _rescale_layout(state.pos).to(dtype=torch.float32, device=output_device)
        return state


def _rescale_spectral_layout(positions: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Center and scale coordinates like ``networkx.rescale_layout``."""
    positions = positions.copy()
    positions -= positions.mean(axis=0)
    limit = np.abs(positions).max()
    if limit > 0:
        positions *= scale / limit
    return positions


@register_op
class SpectralFinalizePositions(Op):
    """Apply spectral centering and scaling with stable output device placement."""

    name = "spectral_finalize_positions"
    category = OpCategory.POSTPROCESS
    reads = ("pos",)
    writes = ("pos",)
    requires = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Rescale coordinates like :func:`layout_spectral`."""
        _ = ctx
        if state.pos is None:
            raise ValueError("SpectralFinalizePositions requires state.pos to be set.")

        output_device = _layout_device(edge_index=problem.edge_index, node_sizes=problem.node_sizes)
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, 2), dtype=torch.float32, device=output_device)
            return state

        positions = state.pos.detach().to(device="cpu", dtype=torch.float64).numpy()
        state.pos = torch.from_numpy(_rescale_spectral_layout(positions=positions, scale=1.0)).to(
            dtype=torch.float32,
            device=output_device,
        )
        return state


def _normalize_classical_positions(positions: torch.Tensor, extent: float) -> torch.Tensor:
    """Apply classical-MDS span normalization with legacy fallbacks."""

    if positions.shape[0] <= 1:
        return torch.zeros_like(positions)

    centered = positions - positions.mean(dim=0, keepdim=True)
    span = float(centered.abs().max().item())
    if span < _MIN_SPAN:
        fallback = torch.zeros_like(centered)
        fallback[:, 0] = torch.linspace(
            -1.0,
            1.0,
            steps=centered.shape[0],
            device=centered.device,
            dtype=centered.dtype,
        )
        centered = fallback
        span = float(centered.abs().max().item())

    return centered * (extent / max(span, _MIN_SPAN))


@register_op
class ClassicalMDSFinalizePositions(Op):
    """Apply legacy classical-MDS final normalization and casting."""

    name: str = "classical_mds_finalize_positions"
    category: OpCategory = OpCategory.POSTPROCESS
    reads: tuple[str, ...] = ("pos",)
    writes: tuple[str, ...] = ("pos",)
    requires: tuple[str, ...] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Normalize final classical-MDS coordinates and resolve output device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing raw embedding coordinates.
        ctx : RuntimeContext
            Execution context.

        Returns
        -------
        SolveState
            State with normalized ``float32`` coordinates.

        Raises
        ------
        ValueError
            If ``state.pos`` is missing.
        """
        _ = ctx
        if state.pos is None:
            raise ValueError("ClassicalMDSFinalizePositions requires state.pos to be set.")

        output_device = _layout_device(edge_index=problem.edge_index, node_sizes=problem.node_sizes)
        extent = _layout_extent(num_nodes=problem.num_nodes, node_sizes=problem.node_sizes)
        normalized = _normalize_classical_positions(
            positions=state.pos.to(device=output_device),
            extent=extent,
        )
        state.pos = normalized.to(dtype=torch.float32, device=output_device)
        return state


@register_op
class PivotMDSFinalizePositions(Op):
    """Apply Pivot-MDS-specific final normalization and output casting."""

    name = "pivot_mds_finalize_positions"
    category = OpCategory.POSTPROCESS
    reads = ("pos",)
    writes = ("pos",)
    requires = ()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Finalize Pivot-MDS output with deterministic extent normalization.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state containing raw Pivot-MDS coordinates.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with final ``pos`` on the resolved output device and ``float32``.

        Notes
        -----
        ``PivotMDSInit`` intentionally returns ``None`` for ``state.pos`` when
        ``num_nodes == 0``. This final op normalizes that edge case to the
        expected empty ``[0, 2]`` tensor after selecting the output device.
        """
        _ = ctx
        output_device = _layout_device(
            edge_index=problem.edge_index,
            node_sizes=problem.node_sizes,
        )
        if state.pos is None or state.pos.numel() == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=output_device)
            return state

        extent = _layout_extent(num_nodes=problem.num_nodes, node_sizes=problem.node_sizes)
        normalized = _normalize_classical_positions(
            positions=state.pos.to(device=output_device),
            extent=extent,
        )
        state.pos = normalized.to(dtype=torch.float32, device=output_device)
        return state


@dataclass(frozen=True)
class DirectionTransformConfig:
    """Configuration for :class:`DirectionTransform`.

    Parameters
    ----------
    direction : str, default="TB"
        Output flow direction. Supported values are ``TB``, ``BT``, ``LR``,
        and ``RL``.
    """

    direction: str = "TB"


@register_op
class DirectionTransform(Op):
    """Rotate or flip coordinates into the requested layout direction."""

    name = "direction_transform"
    category = OpCategory.POSTPROCESS
    reads = ("pos",)
    writes = ("pos",)
    requires = ("pos",)

    def __init__(self, config: Optional[DirectionTransformConfig] = None) -> None:
        """Store the direction transform configuration.

        Parameters
        ----------
        config : DirectionTransformConfig, optional
            Direction configuration. Defaults to top-to-bottom.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or DirectionTransformConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Apply the configured direction transform to ``state.pos``.

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
            State with direction-adjusted coordinates.

        Raises
        ------
        ValueError
            If the configured direction is unsupported.
        """
        _ = problem, ctx
        positions = _require_positions(state=state, op_name=self.name)
        direction = self.config.direction.upper()
        if direction == "TB":
            state.pos = positions.clone()
            return state
        if direction == "BT":
            transformed = positions.clone()
            transformed[:, 1] = -transformed[:, 1]
            state.pos = transformed
            return state
        if direction == "LR":
            state.pos = positions[:, [1, 0]].clone()
            return state
        if direction == "RL":
            transformed = positions[:, [1, 0]].clone()
            transformed[:, 0] = -transformed[:, 0]
            state.pos = transformed
            return state
        raise ValueError(f"Unsupported DirectionTransform direction: {self.config.direction}")


@register_op
class StripDummyNodes(Op):
    """Remove dummy-node coordinates introduced by layered expansions."""

    name = "strip_dummy_nodes"
    category = OpCategory.POSTPROCESS
    reads = ("pos", "extras.expanded_graph")
    writes = ("pos",)
    requires = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Truncate the position tensor back to the original node count.

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
            State whose positions only include original graph nodes.
        """
        _ = ctx
        positions = _require_positions(state=state, op_name=self.name)
        expanded_graph = state.extras.get("expanded_graph")
        if expanded_graph is not None:
            expanded_num_nodes = getattr(expanded_graph, "num_nodes", positions.shape[0])
            if expanded_num_nodes < problem.num_nodes:
                raise ValueError(
                    "expanded_graph.num_nodes cannot be smaller than problem.num_nodes"
                )
        visible_nodes = min(problem.num_nodes, positions.shape[0])
        state.pos = positions[:visible_nodes].clone()
        return state


@dataclass(frozen=True)
class SpreadFanoutChildrenConfig:
    """Configuration for :class:`SpreadFanoutChildren`.

    Parameters
    ----------
    hub_threshold : int, default=8
        Minimum unique out-neighbor count required to treat a node as a hub.
    widening : float, default=1.5
        Multiplier applied to the observed child spacing before re-spreading.
    """

    hub_threshold: int = 8
    widening: float = 1.5


@register_op
class SpreadFanoutChildren(Op):
    """Redistribute high-fanout child groups around each hub's x-coordinate."""

    name = "spread_fanout_children"
    category = OpCategory.POSTPROCESS
    reads = ("pos", "layers")
    writes = ("pos",)
    requires = ("pos", "layers")

    def __init__(self, config: Optional[SpreadFanoutChildrenConfig] = None) -> None:
        """Store the hub spreading configuration.

        Parameters
        ----------
        config : SpreadFanoutChildrenConfig, optional
            Hub spreading configuration. Defaults to the engine-init values.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or SpreadFanoutChildrenConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Spread children of high-fanout hubs while preserving child order.

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
            State with widened child x-coordinates for detected hubs.
        """
        _ = ctx
        positions = _require_positions(state=state, op_name=self.name)
        if positions.shape[0] <= 1 or problem.edge_index.numel() == 0:
            return state
        if state.layers is None:
            raise ValueError("spread_fanout_children requires state.layers to be set.")

        updated = positions.clone()
        edge_index = problem.edge_index.detach().to(device="cpu", dtype=torch.long)
        children_of: DefaultDict[int, List[int]] = defaultdict(list)
        for source, target in edge_index.t().tolist():
            if 0 <= source < problem.num_nodes and 0 <= target < problem.num_nodes:
                children_of[source].append(target)

        for hub, raw_children in children_of.items():
            children = _unique_children_in_order(raw_children)
            if len(children) < self.config.hub_threshold:
                continue

            hub_x = float(updated[hub, 0].item())
            by_layer: DefaultDict[int, List[int]] = defaultdict(list)
            for child in children:
                by_layer[int(state.layers[child].item())].append(child)

            for layer_children in by_layer.values():
                if len(layer_children) <= 1:
                    continue

                ordered_children = sorted(layer_children, key=lambda node: float(updated[node, 0]))
                left = float(updated[ordered_children[0], 0].item())
                right = float(updated[ordered_children[-1], 0].item())
                current_span = max(right - left, float(len(ordered_children) - 1))
                step = max(
                    (current_span / float(max(len(ordered_children) - 1, 1)))
                    * self.config.widening,
                    1.0,
                )
                start = hub_x - (step * float(len(ordered_children) - 1) / 2.0)
                for index, child in enumerate(ordered_children):
                    updated[child, 0] = start + step * float(index)

        state.pos = updated
        return state


__all__ = [
    "CenterPositions",
    "DirectionTransform",
    "DirectionTransformConfig",
    "FRFinalizePositions",
    "GraphOptFinalizePositions",
    "KamadaKawaiFinalizePositions",
    "ClassicalMDSFinalizePositions",
    "LinLogFinalizePositions",
    "NormalizePositions",
    "NormalizePositionsConfig",
    "ScalePositions",
    "ScalePositionsConfig",
    "SpreadFanoutChildren",
    "SpreadFanoutChildrenConfig",
    "StripDummyNodes",
]
