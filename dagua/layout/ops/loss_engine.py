"""Engine loss operations for the composable layout pipeline.

This module wraps the native engine's differentiable losses as ``LossOp``
subclasses so they can participate in ``LossGroup`` execution.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, ClassVar, Optional, Tuple

import torch
import torch.nn.functional as F

from dagua.layout.constraints import (
    _ClusterCache,
    alignment_loss,
    back_edge_compactness_loss,
    cluster_compactness_loss,
    cluster_containment_loss,
    cluster_separation_loss,
    crossing_loss,
    dag_ordering_loss,
    edge_attraction_loss,
    edge_length_variance_loss,
    edge_straightness_loss,
    fanout_distribution_loss,
    flex_spacing_loss,
    position_pin_loss,
    spacing_consistency_loss,
)
from dagua.layout.ops.base import LossOp
from dagua.layout.ops.spatial_hash import cell_list_candidate_pairs
from dagua.layout.ops.state import FlexConstraints, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_CELL_LIST_EXACT_THRESHOLD = 500
_REPULSION_CUTOFF_DIAMETER_FACTOR = 5.0


def _require_pos(state: SolveState) -> torch.Tensor:
    """Return the active position tensor.

    Parameters
    ----------
    state : SolveState
        Mutable solve state that should contain ``pos``.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If the solve state does not contain positions.
    """
    if state.pos is None:
        raise ValueError("Loss evaluation requires state.pos to be set.")
    return state.pos


def _require_node_sizes(problem: LayoutProblem) -> torch.Tensor:
    """Return the problem's node-size tensor.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable problem inputs.

    Returns
    -------
    torch.Tensor
        Node-size tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If node sizes are required but missing from the problem.
    """
    if problem.node_sizes is None:
        raise ValueError("Loss evaluation requires problem.node_sizes to be set.")
    return problem.node_sizes


def _expanded_graph_for_state(state: SolveState, pos: torch.Tensor) -> Optional[Any]:
    """Return the active expanded graph when positions match it.

    Parameters
    ----------
    state : SolveState
        Mutable solve state that may carry ``extras["expanded_graph"]``.
    pos : torch.Tensor
        Active positions with shape ``[N_active, 2]``.

    Returns
    -------
    Any | None
        Expanded graph metadata when ``pos`` is expanded, otherwise ``None``.
    """
    expanded_graph = state.extras.get("expanded_graph")
    if expanded_graph is None:
        return None
    if int(getattr(expanded_graph, "num_nodes", -1)) != int(pos.shape[0]):
        return None
    return expanded_graph


def _active_edge_index(
    problem: LayoutProblem,
    state: SolveState,
    pos: torch.Tensor,
) -> torch.Tensor:
    """Return the edge tensor used by edge-centric losses.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable original graph inputs.
    state : SolveState
        Mutable solve state.
    pos : torch.Tensor
        Active positions with shape ``[N_active, 2]``.

    Returns
    -------
    torch.Tensor
        Edge tensor on ``pos.device``.
    """
    expanded_graph = _expanded_graph_for_state(state=state, pos=pos)
    if expanded_graph is not None:
        return expanded_graph.edge_index.to(device=pos.device, dtype=torch.long)
    return problem.edge_index.to(device=pos.device, dtype=torch.long)


def _active_node_sizes(
    problem: LayoutProblem,
    state: SolveState,
    pos: torch.Tensor,
) -> torch.Tensor:
    """Return node sizes matching the active edge-loss graph.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable original graph inputs.
    state : SolveState
        Mutable solve state.
    pos : torch.Tensor
        Active positions with shape ``[N_active, 2]``.

    Returns
    -------
    torch.Tensor
        Node-size tensor with shape ``[N_active, 2]``.
    """
    expanded_graph = _expanded_graph_for_state(state=state, pos=pos)
    if expanded_graph is not None:
        return expanded_graph.node_sizes.to(device=pos.device, dtype=pos.dtype)
    return _require_node_sizes(problem)


def _visible_original_pos(
    problem: LayoutProblem,
    state: SolveState,
    pos: torch.Tensor,
) -> torch.Tensor:
    """Return original-node positions for box-centric losses.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable original graph inputs.
    state : SolveState
        Mutable solve state.
    pos : torch.Tensor
        Active positions with shape ``[N_active, 2]``.

    Returns
    -------
    torch.Tensor
        Original node block with shape ``[N_original, 2]``.
    """
    if _expanded_graph_for_state(state=state, pos=pos) is None:
        return pos
    return pos[: problem.num_nodes]


def _visible_original_layer_index(state: SolveState, pos: torch.Tensor) -> Optional[object]:
    """Return original-node layer index when the active state is expanded.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    pos : torch.Tensor
        Active positions with shape ``[N_active, 2]``.

    Returns
    -------
    object | None
        Layer index for original nodes when available.
    """
    if _expanded_graph_for_state(state=state, pos=pos) is None:
        return state.layer_index
    return state.extras.get("original_layer_index", state.layer_index)


def _zero_scalar(pos: torch.Tensor) -> torch.Tensor:
    """Return a differentiable zero scalar on the position device.

    Parameters
    ----------
    pos : torch.Tensor
        Active position tensor used for device and dtype selection.

    Returns
    -------
    torch.Tensor
        Scalar zero tensor.
    """
    return pos.sum() * 0.0


def _get_cluster_cache(
    problem: LayoutProblem,
    state: SolveState,
    device: torch.device,
) -> _ClusterCache:
    """Return a per-pipeline-call cluster cache, building it lazily.

    Cluster losses iterate over the same cluster dict every step. The cache
    flattens (cluster_id, node_idx) pairs and precomputes sibling/containment
    pair tensors + per-cluster max sizes (since node_sizes are constant), so
    each step is a few scatter ops instead of a Python loop.

    The cache is invalidated when ``problem.clusters`` identity, cluster_parents
    identity, or device changes -- safe across V-cycle level transitions which
    set ``problem.clusters = None`` for coarse levels.
    """
    cached = state.extras.get("_cluster_cache")
    key = (
        id(problem.clusters),
        id(problem.cluster_parents),
        device,
        id(problem.node_sizes),
    )
    if cached is not None and cached[0] == key:
        return cached[1]
    cache = _ClusterCache(
        problem.clusters or {},
        problem.cluster_parents,
        problem.node_sizes,
        device,
    )
    state.extras["_cluster_cache"] = (key, cache)
    return cache


def _exact_repulsion_loss(
    pos: torch.Tensor,
    distance_epsilon: float,
    node_sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute the exact all-pairs repulsion term from ``constraints.py``.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    distance_epsilon : float
        Positive additive floor applied to squared distances.
    node_sizes : torch.Tensor | None, default=None
        Optional node sizes with shape ``[N, 2]`` for size-aware scaling.

    Returns
    -------
    torch.Tensor
        Scalar repulsion loss.
    """
    num_nodes = pos.shape[0]
    if num_nodes <= 1:
        return _zero_scalar(pos)

    diff = pos.unsqueeze(0) - pos.unsqueeze(1)
    dist_sq = diff.square().sum(dim=2) + distance_epsilon
    mask = ~torch.eye(num_nodes, dtype=torch.bool, device=pos.device)

    if node_sizes is not None:
        combined_w = node_sizes[:, 0].unsqueeze(0) + node_sizes[:, 0].unsqueeze(1)
        combined_h = node_sizes[:, 1].unsqueeze(0) + node_sizes[:, 1].unsqueeze(1)
        size_factor = (combined_w * combined_h) / (combined_w * combined_h).mean()
        return (size_factor[mask] / dist_sq[mask]).mean()

    return (1.0 / dist_sq[mask]).mean()


def _exact_overlap_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
) -> torch.Tensor:
    """Compute the exact all-pairs overlap penalty from ``constraints.py``.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    padding : float
        Extra separation margin applied around each node.

    Returns
    -------
    torch.Tensor
        Scalar overlap loss.
    """
    num_nodes = pos.shape[0]
    if num_nodes <= 1:
        return _zero_scalar(pos)

    dx_abs = torch.abs(pos.unsqueeze(0)[:, :, 0] - pos.unsqueeze(1)[:, :, 0])
    dy_abs = torch.abs(pos.unsqueeze(0)[:, :, 1] - pos.unsqueeze(1)[:, :, 1])
    min_dx = (node_sizes.unsqueeze(0)[:, :, 0] + node_sizes.unsqueeze(1)[:, :, 0]) / 2 + padding
    min_dy = (node_sizes.unsqueeze(0)[:, :, 1] + node_sizes.unsqueeze(1)[:, :, 1]) / 2 + padding
    overlap = F.relu(min_dx - dx_abs) * F.relu(min_dy - dy_abs)
    mask = ~torch.eye(num_nodes, dtype=torch.bool, device=pos.device)
    return overlap[mask].mean()


def _unique_pair_count(num_nodes: int) -> int:
    """Return the number of unordered pairs in a node set.

    Parameters
    ----------
    num_nodes : int
        Number of nodes.

    Returns
    -------
    int
        Count of unique unordered pairs.
    """
    return num_nodes * (num_nodes - 1) // 2


def _size_factor_mean(node_sizes: torch.Tensor) -> torch.Tensor:
    """Return the exact all-pairs mean for repulsion size normalization.

    Parameters
    ----------
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Scalar mean of ``(w_i + w_j) * (h_i + h_j)`` over all ordered pairs.
    """
    widths = node_sizes[:, 0]
    heights = node_sizes[:, 1]
    return 2.0 * (widths * heights).mean() + 2.0 * widths.mean() * heights.mean()


def _cell_list_repulsion_cutoff(
    node_sizes: Optional[torch.Tensor],
    cutoff_diameter_factor: float,
) -> float:
    """Return the radius used by cell-list repulsion.

    Parameters
    ----------
    node_sizes : torch.Tensor | None
        Optional node sizes with shape ``[N, 2]``.
    cutoff_diameter_factor : float
        Multiplier applied to the maximum node diameter.

    Returns
    -------
    float
        Positive cutoff radius for candidate pair generation.
    """
    if node_sizes is None or int(node_sizes.numel()) == 0:
        return max(float(cutoff_diameter_factor), 1.0)
    max_diameter = float(node_sizes.detach().max().item())
    return max(max_diameter * cutoff_diameter_factor, 1.0)


def _cell_list_overlap_cutoff(node_sizes: torch.Tensor, padding: float) -> float:
    """Return a radius guaranteed to include all overlapping boxes.

    Parameters
    ----------
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    padding : float
        Extra separation margin applied around each node.

    Returns
    -------
    float
        Positive cutoff radius for candidate pair generation.
    """
    max_axis = float(node_sizes.detach().max().item()) + float(padding)
    return max(math.sqrt(2.0) * max_axis, 1.0)


def _cell_list_repulsion_loss(
    pos: torch.Tensor,
    distance_epsilon: float,
    cutoff_radius: float,
    node_sizes: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Compute cutoff repulsion over spatial-hash candidate pairs.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    distance_epsilon : float
        Positive additive floor applied to squared distances.
    cutoff_radius : float
        Radius for retaining local repulsion pairs.
    node_sizes : torch.Tensor | None, default=None
        Optional node sizes with shape ``[N, 2]`` for size-aware scaling.

    Returns
    -------
    torch.Tensor
        Scalar repulsion loss normalized by all possible unordered pairs.
    """
    num_nodes = int(pos.shape[0])
    if num_nodes <= 1:
        return _zero_scalar(pos)
    pairs = cell_list_candidate_pairs(pos, cutoff_radius=cutoff_radius)
    if int(pairs.shape[1]) == 0:
        return _zero_scalar(pos)

    source = pairs[0]
    target = pairs[1]
    diff = pos[source] - pos[target]
    raw_dist_sq = diff.square().sum(dim=1)
    in_radius = raw_dist_sq <= cutoff_radius * cutoff_radius
    if not bool(in_radius.any().item()):
        return _zero_scalar(pos)

    source = source[in_radius]
    target = target[in_radius]
    dist_sq = raw_dist_sq[in_radius] + distance_epsilon
    if node_sizes is not None:
        combined_w = node_sizes[source, 0] + node_sizes[target, 0]
        combined_h = node_sizes[source, 1] + node_sizes[target, 1]
        size_factor = (combined_w * combined_h) / _size_factor_mean(node_sizes)
        pair_values = size_factor / dist_sq
    else:
        pair_values = 1.0 / dist_sq
    return pair_values.sum() / float(_unique_pair_count(num_nodes))


def _cell_list_overlap_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    cutoff_radius: float,
) -> torch.Tensor:
    """Compute exact local overlap loss over spatial-hash candidate pairs.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    padding : float
        Extra separation margin applied around each node.
    cutoff_radius : float
        Radius for candidate pair generation.

    Returns
    -------
    torch.Tensor
        Scalar overlap loss normalized by all possible unordered pairs.
    """
    num_nodes = int(pos.shape[0])
    if num_nodes <= 1:
        return _zero_scalar(pos)
    pairs = cell_list_candidate_pairs(pos, cutoff_radius=cutoff_radius)
    if int(pairs.shape[1]) == 0:
        return _zero_scalar(pos)

    source = pairs[0]
    target = pairs[1]
    dx_abs = torch.abs(pos[source, 0] - pos[target, 0])
    dy_abs = torch.abs(pos[source, 1] - pos[target, 1])
    min_dx = (node_sizes[source, 0] + node_sizes[target, 0]) / 2.0 + padding
    min_dy = (node_sizes[source, 1] + node_sizes[target, 1]) / 2.0 + padding
    overlap = F.relu(min_dx - dx_abs) * F.relu(min_dy - dy_abs)
    return overlap.sum() / float(_unique_pair_count(num_nodes))


def _repulsion_loss(
    pos: torch.Tensor,
    distance_epsilon: float,
    node_sizes: Optional[torch.Tensor],
    exact: bool,
    exact_threshold: int,
    cutoff_diameter_factor: float,
) -> torch.Tensor:
    """Compute repulsion with exact fallback or spatial-hash cutoff.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    distance_epsilon : float
        Positive additive floor applied to squared distances.
    node_sizes : torch.Tensor | None
        Optional node sizes with shape ``[N, 2]`` for size-aware scaling.
    exact : bool
        Whether to force the legacy all-pairs path.
    exact_threshold : int
        Node-count threshold below which exact all-pairs is cheaper.
    cutoff_diameter_factor : float
        Radius multiplier applied to the maximum node diameter.

    Returns
    -------
    torch.Tensor
        Scalar repulsion loss.
    """
    if exact or int(pos.shape[0]) < exact_threshold:
        return _exact_repulsion_loss(
            pos,
            distance_epsilon=distance_epsilon,
            node_sizes=node_sizes,
        )
    return _cell_list_repulsion_loss(
        pos,
        distance_epsilon=distance_epsilon,
        cutoff_radius=_cell_list_repulsion_cutoff(node_sizes, cutoff_diameter_factor),
        node_sizes=node_sizes,
    )


def _overlap_loss(
    pos: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
    exact: bool,
    exact_threshold: int,
) -> torch.Tensor:
    """Compute overlap loss with exact fallback or spatial-hash candidates.

    Parameters
    ----------
    pos : torch.Tensor
        Node positions with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.
    padding : float
        Extra separation margin applied around each node.
    exact : bool
        Whether to force the legacy all-pairs path.
    exact_threshold : int
        Node-count threshold below which exact all-pairs is cheaper.

    Returns
    -------
    torch.Tensor
        Scalar overlap loss.
    """
    if exact or int(pos.shape[0]) < exact_threshold:
        return _exact_overlap_loss(pos, node_sizes, padding=padding)
    return _cell_list_overlap_loss(
        pos,
        node_sizes,
        padding=padding,
        cutoff_radius=_cell_list_overlap_cutoff(node_sizes, padding),
    )


def _flex(problem: LayoutProblem) -> Optional[FlexConstraints]:
    """Return prepared flex constraints when available.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable problem inputs.

    Returns
    -------
    FlexConstraints | None
        Prepared flex constraint payload, if present.
    """
    return problem.flex


@dataclass(frozen=True)
class DagOrderingLossConfig:
    """Configuration for :class:`DagOrderingLoss`."""

    rank_sep: float = 50.0


@dataclass(frozen=True)
class EdgeAttractionLossConfig:
    """Configuration for :class:`EdgeAttractionLoss`."""

    x_bias: float = 4.0


@dataclass(frozen=True)
class EdgeStraightnessLossConfig:
    """Configuration for :class:`EdgeStraightnessLoss`."""


@dataclass(frozen=True)
class EdgeLengthVarianceLossConfig:
    """Configuration for :class:`EdgeLengthVarianceLoss`."""


@dataclass(frozen=True)
class RepulsionLossConfig:
    """Configuration for :class:`RepulsionLoss`.

    Parameters
    ----------
    threshold : int, default=500
        Node-count threshold below which the exact all-pairs path is cheaper
        than building the spatial hash.
    sample_k : int, default=128
        Reserved sample count for compatibility with older sampled paths.
    rvs_threshold : int, default=5000
        Reserved threshold for future random-vantage-set repulsion paths.
    rvs_nn_k : int, default=20
        Reserved neighbor count for future random-vantage-set paths.
    distance_epsilon : float, default=1e-4
        Positive additive floor applied to squared pairwise distances.
    exact : bool, default=False
        Whether to force the legacy all-pairs loss path.
    cutoff_diameter_factor : float, default=5.0
        Spatial-hash repulsion radius as a multiple of maximum node diameter.
    """

    threshold: int = _CELL_LIST_EXACT_THRESHOLD
    sample_k: int = 128
    rvs_threshold: int = 5000
    rvs_nn_k: int = 20
    distance_epsilon: float = 1.0e-4
    exact: bool = False
    cutoff_diameter_factor: float = _REPULSION_CUTOFF_DIAMETER_FACTOR


@dataclass(frozen=True)
class OverlapAvoidanceLossConfig:
    """Configuration for :class:`OverlapAvoidanceLoss`."""

    padding: float = 2.0
    rvs_threshold: int = 100000
    # Cell-list path handles N >= exact_threshold exactly for overlap pairs
    # without allocating the legacy [N, N] tensors.
    exact_threshold: int = _CELL_LIST_EXACT_THRESHOLD
    sample_k: int = 128
    exact: bool = False


@dataclass(frozen=True)
class CrossingLossConfig:
    """Configuration for :class:`CrossingLoss`."""

    alpha: float = 5.0
    max_pairs: int = 2000


@dataclass(frozen=True)
class ClusterCompactnessLossConfig:
    """Configuration for :class:`ClusterCompactnessLoss`."""


@dataclass(frozen=True)
class ClusterSeparationLossConfig:
    """Configuration for :class:`ClusterSeparationLoss`."""

    padding: float = 10.0


@dataclass(frozen=True)
class ClusterContainmentLossConfig:
    """Configuration for :class:`ClusterContainmentLoss`."""

    padding: float = 18.0


@dataclass(frozen=True)
class SpacingConsistencyLossConfig:
    """Configuration for :class:`SpacingConsistencyLoss`."""

    target_gap: float = 25.0


@dataclass(frozen=True)
class FanoutDistributionLossConfig:
    """Configuration for :class:`FanoutDistributionLoss`."""

    degree_threshold: int = 5


@dataclass(frozen=True)
class BackEdgeCompactnessLossConfig:
    """Configuration for :class:`BackEdgeCompactnessLoss`."""


@dataclass(frozen=True)
class PositionPinLossConfig:
    """Configuration for :class:`PositionPinLoss`."""


@dataclass(frozen=True)
class AlignmentLossConfig:
    """Configuration for :class:`AlignmentLoss`."""


@dataclass(frozen=True)
class FlexSpacingLossConfig:
    """Configuration for :class:`FlexSpacingLoss`."""


@register_op
@dataclass(frozen=True)
class DagOrderingLoss(LossOp):
    """Penalize edges whose targets drift above their sources."""

    config: DagOrderingLossConfig = field(default_factory=DagOrderingLossConfig)

    name: ClassVar[str] = "dag_ordering_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "edge_batch_context")
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "edge"
    weight_key: ClassVar[str] = "w_dag"
    default_weight: ClassVar[float] = 10.0

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the DAG ordering penalty.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        pos = _require_pos(state)
        node_sizes = _active_node_sizes(problem=problem, state=state, pos=pos)
        return dag_ordering_loss(
            pos,
            _active_edge_index(problem=problem, state=state, pos=pos),
            node_sizes,
            rank_sep=self.config.rank_sep,
            edge_ctx=state.edge_batch_context,
        )


@register_op
@dataclass(frozen=True)
class EdgeAttractionLoss(LossOp):
    """Pull connected nodes together with an x-axis bias."""

    config: EdgeAttractionLossConfig = field(default_factory=EdgeAttractionLossConfig)

    name: ClassVar[str] = "edge_attraction_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "edge_batch_context")
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "edge"
    weight_key: ClassVar[str] = "w_attract"
    default_weight: ClassVar[float] = 2.0

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the edge-attraction loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        pos = _require_pos(state)
        return edge_attraction_loss(
            pos,
            _active_edge_index(problem=problem, state=state, pos=pos),
            x_bias=self.config.x_bias,
            edge_ctx=state.edge_batch_context,
        )


@register_op
@dataclass(frozen=True)
class EdgeStraightnessLoss(LossOp):
    """Penalize horizontal displacement along edges."""

    config: EdgeStraightnessLossConfig = field(default_factory=EdgeStraightnessLossConfig)

    name: ClassVar[str] = "edge_straightness_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "edge_batch_context")
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "edge"
    weight_key: ClassVar[str] = "w_straightness"
    default_weight: ClassVar[float] = 2.2

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the edge-straightness loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        pos = _require_pos(state)
        return edge_straightness_loss(
            pos,
            _active_edge_index(problem=problem, state=state, pos=pos),
            edge_ctx=state.edge_batch_context,
        )


@register_op
@dataclass(frozen=True)
class EdgeLengthVarianceLoss(LossOp):
    """Penalize variance in edge lengths."""

    config: EdgeLengthVarianceLossConfig = field(default_factory=EdgeLengthVarianceLossConfig)

    name: ClassVar[str] = "edge_length_variance_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "edge_batch_context")
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "edge"
    weight_key: ClassVar[str] = "w_length_variance"
    default_weight: ClassVar[float] = 0.7

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the edge-length variance loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        pos = _require_pos(state)
        return edge_length_variance_loss(
            pos,
            _active_edge_index(problem=problem, state=state, pos=pos),
            edge_ctx=state.edge_batch_context,
        )


@register_op
@dataclass(frozen=True)
class RepulsionLoss(LossOp):
    """Repel all nodes from one another."""

    config: RepulsionLossConfig = field(default_factory=RepulsionLossConfig)

    name: ClassVar[str] = "repulsion_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "sampled"
    weight_key: ClassVar[str] = "w_repel"
    default_weight: ClassVar[float] = 0.1

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the repulsion loss.

        Tiered strategy:
        - exact=True or N < self.config.threshold: legacy exact O(N^2)
        - otherwise: uniform spatial hash with a radius cutoff, normalized as
          omitted far pairs contributing zero to the all-pairs mean.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        pos = _visible_original_pos(problem, state, _require_pos(state))
        node_sizes = problem.node_sizes
        return _repulsion_loss(
            pos,
            distance_epsilon=self.config.distance_epsilon,
            node_sizes=node_sizes,
            exact=self.config.exact,
            exact_threshold=self.config.threshold,
            cutoff_diameter_factor=self.config.cutoff_diameter_factor,
        )


@register_op
@dataclass(frozen=True)
class OverlapAvoidanceLoss(LossOp):
    """Penalize node bounding-box overlap."""

    config: OverlapAvoidanceLossConfig = field(default_factory=OverlapAvoidanceLossConfig)

    name: ClassVar[str] = "overlap_avoidance_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "sampled"
    weight_key: ClassVar[str] = "w_overlap"
    default_weight: ClassVar[float] = 5.0

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the overlap-avoidance loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        pos = _visible_original_pos(problem, state, _require_pos(state))
        node_sizes = _require_node_sizes(problem)
        return _overlap_loss(
            pos,
            node_sizes,
            padding=self.config.padding,
            exact=self.config.exact,
            exact_threshold=self.config.exact_threshold,
        )


@register_op
@dataclass(frozen=True)
class CrossingLoss(LossOp):
    """Differentiable crossing proxy loss."""

    config: CrossingLossConfig = field(default_factory=CrossingLossConfig)

    name: ClassVar[str] = "crossing_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "layers")
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"
    weight_key: ClassVar[str] = "w_crossing"
    default_weight: ClassVar[float] = 1.8

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the crossing proxy loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions and optional layers.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        pos = _require_pos(state)
        return crossing_loss(
            pos,
            _active_edge_index(problem=problem, state=state, pos=pos),
            alpha=self.config.alpha,
            max_pairs=self.config.max_pairs,
            layer_assignments=state.layers,
        )


@register_op
@dataclass(frozen=True)
class ClusterCompactnessLoss(LossOp):
    """Pull nodes toward their cluster centroids."""

    config: ClusterCompactnessLossConfig = field(default_factory=ClusterCompactnessLossConfig)

    name: ClassVar[str] = "cluster_compactness_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"
    weight_key: ClassVar[str] = "w_cluster"
    default_weight: ClassVar[float] = 1.0

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the cluster-compactness loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        pos = _require_pos(state)
        if not problem.clusters:
            return _zero_scalar(pos)
        cache = _get_cluster_cache(problem, state, pos.device)
        return cluster_compactness_loss(pos, problem.clusters, device=pos.device, cache=cache)


@register_op
@dataclass(frozen=True)
class ClusterSeparationLoss(LossOp):
    """Repel overlapping sibling cluster bounding boxes."""

    config: ClusterSeparationLossConfig = field(default_factory=ClusterSeparationLossConfig)

    name: ClassVar[str] = "cluster_separation_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"
    weight_key: ClassVar[str] = "w_cluster_sep"
    default_weight: ClassVar[float] = 0.5

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the sibling-cluster separation loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        pos = _require_pos(state)
        if not problem.clusters:
            return _zero_scalar(pos)
        node_sizes = _require_node_sizes(problem)
        cache = _get_cluster_cache(problem, state, pos.device)
        return cluster_separation_loss(
            pos,
            node_sizes,
            problem.clusters,
            padding=self.config.padding,
            device=pos.device,
            cluster_parents=problem.cluster_parents,
            cache=cache,
        )


@register_op
@dataclass(frozen=True)
class ClusterContainmentLoss(LossOp):
    """Keep child cluster bounding boxes inside their parents."""

    config: ClusterContainmentLossConfig = field(default_factory=ClusterContainmentLossConfig)

    name: ClassVar[str] = "cluster_containment_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"
    weight_key: ClassVar[str] = "w_cluster_contain"
    default_weight: ClassVar[float] = 2.0

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the cluster-containment loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        pos = _require_pos(state)
        if not problem.clusters or not problem.cluster_parents:
            return _zero_scalar(pos)
        node_sizes = _require_node_sizes(problem)
        cache = _get_cluster_cache(problem, state, pos.device)
        return cluster_containment_loss(
            pos,
            node_sizes,
            problem.clusters,
            problem.cluster_parents,
            padding=self.config.padding,
            device=pos.device,
            cache=cache,
        )


@register_op
@dataclass(frozen=True)
class SpacingConsistencyLoss(LossOp):
    """Penalize deviation from the target same-layer gap."""

    config: SpacingConsistencyLossConfig = field(default_factory=SpacingConsistencyLossConfig)

    name: ClassVar[str] = "spacing_consistency_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "layer_index")
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"
    weight_key: ClassVar[str] = "w_spacing"
    default_weight: ClassVar[float] = 0.3

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the spacing-consistency loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions and a layer index.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        full_pos = _require_pos(state)
        pos = _visible_original_pos(problem, state, full_pos)
        node_sizes = _require_node_sizes(problem)
        return spacing_consistency_loss(
            pos,
            node_sizes,
            _visible_original_layer_index(state, full_pos),
            target_gap=self.config.target_gap,
        )


@register_op
@dataclass(frozen=True)
class FanoutDistributionLoss(LossOp):
    """Penalize uneven angular child distribution around hubs."""

    config: FanoutDistributionLossConfig = field(default_factory=FanoutDistributionLossConfig)

    name: ClassVar[str] = "fanout_distribution_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "edge_batch_context")
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "edge"
    weight_key: ClassVar[str] = "w_fanout"
    default_weight: ClassVar[float] = 0.3

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the fanout-distribution loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        pos = _visible_original_pos(problem, state, _require_pos(state))
        return fanout_distribution_loss(
            pos,
            problem.edge_index,
            degree_threshold=self.config.degree_threshold,
            edge_ctx=state.edge_batch_context,
            step=state.step,
            edge_is_sampled=state.edge_batch_context is not None,
        )


@register_op
@dataclass(frozen=True)
class BackEdgeCompactnessLoss(LossOp):
    """Penalize wide back-edge spans."""

    config: BackEdgeCompactnessLossConfig = field(default_factory=BackEdgeCompactnessLossConfig)

    name: ClassVar[str] = "back_edge_compactness_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "edge_batch_context")
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "edge"
    weight_key: ClassVar[str] = "w_back_edge"
    default_weight: ClassVar[float] = 0.3

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the back-edge compactness loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        pos = _visible_original_pos(problem, state, _require_pos(state))
        return back_edge_compactness_loss(
            pos,
            problem.edge_index,
            edge_ctx=state.edge_batch_context,
        )


@register_op
@dataclass(frozen=True)
class PositionPinLoss(LossOp):
    """Pull softly pinned nodes toward their targets."""

    config: PositionPinLossConfig = field(default_factory=PositionPinLossConfig)

    name: ClassVar[str] = "position_pin_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"
    weight_key: ClassVar[str] = "w_pin"
    default_weight: ClassVar[float] = 1.0

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the soft-pin loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        full_pos = _require_pos(state)
        pos = _visible_original_pos(problem, state, full_pos)
        flex = _flex(problem)
        if (
            flex is None
            or flex.pin_indices is None
            or flex.pin_targets is None
            or flex.pin_weights is None
            or flex.soft_pin_mask is None
        ):
            return _zero_scalar(full_pos)
        # Propagated coarse flex may be on the hierarchy tensor's device,
        # which can differ from the active pos device on mixed-device runs
        # (CPU hierarchy + CUDA refinement). Move to pos.device before
        # indexing pos.
        device = pos.device
        return position_pin_loss(
            pos,
            flex.pin_indices.to(device=device, dtype=torch.long),
            flex.pin_targets.to(device=device, dtype=pos.dtype),
            flex.pin_weights.to(device=device, dtype=pos.dtype),
            flex.soft_pin_mask.to(device=device),
        )


@register_op
@dataclass(frozen=True)
class AlignmentLoss(LossOp):
    """Penalize variance along the configured alignment axis."""

    config: AlignmentLossConfig = field(default_factory=AlignmentLossConfig)

    name: ClassVar[str] = "alignment_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos",)
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"
    weight_key: ClassVar[str] = "w_align_flex"
    default_weight: ClassVar[float] = 1.0

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the alignment variance loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        full_pos = _require_pos(state)
        pos = _visible_original_pos(problem, state, full_pos)
        flex = _flex(problem)
        if flex is None or not flex.align_groups:
            return _zero_scalar(full_pos)
        # Propagated coarse align groups may live on a different device
        # than the active pos (mixed-device runs); migrate index tensors
        # before the loss indexes pos.
        device = pos.device
        migrated = [
            (g[0].to(device=device, dtype=torch.long), g[1], g[2]) for g in flex.align_groups
        ]
        return alignment_loss(pos, migrated)


@register_op
@dataclass(frozen=True)
class FlexSpacingLoss(LossOp):
    """Spacing consistency weighted by the prepared flex spacing strength."""

    config: FlexSpacingLossConfig = field(default_factory=FlexSpacingLossConfig)

    name: ClassVar[str] = "flex_spacing_loss"
    category: ClassVar[OpCategory] = OpCategory.LOSS
    reads: ClassVar[Tuple[str, ...]] = ("pos", "layer_index")
    writes: ClassVar[Tuple[str, ...]] = ()
    requires: ClassVar[Tuple[str, ...]] = ("pos",)
    access_pattern: ClassVar[str] = "global"
    weight_key: ClassVar[str] = "w_flex_spacing"
    default_weight: ClassVar[float] = 1.0

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Return the flex-weighted spacing loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs.
        state : SolveState
            Mutable solve state containing positions and a layer index.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar loss tensor.
        """
        del ctx
        full_pos = _require_pos(state)
        pos = _visible_original_pos(problem, state, full_pos)
        node_sizes = _require_node_sizes(problem)
        flex = _flex(problem)
        if flex is None or flex.flex_node_sep is None or flex.flex_node_sep_weight is None:
            return _zero_scalar(full_pos)
        return flex_spacing_loss(
            pos,
            node_sizes,
            _visible_original_layer_index(state, full_pos),
            flex.flex_node_sep,
            flex.flex_node_sep_weight,
        )


__all__ = [
    "AlignmentLoss",
    "AlignmentLossConfig",
    "BackEdgeCompactnessLoss",
    "BackEdgeCompactnessLossConfig",
    "ClusterCompactnessLoss",
    "ClusterCompactnessLossConfig",
    "ClusterContainmentLoss",
    "ClusterContainmentLossConfig",
    "ClusterSeparationLoss",
    "ClusterSeparationLossConfig",
    "CrossingLoss",
    "CrossingLossConfig",
    "DagOrderingLoss",
    "DagOrderingLossConfig",
    "EdgeAttractionLoss",
    "EdgeAttractionLossConfig",
    "EdgeLengthVarianceLoss",
    "EdgeLengthVarianceLossConfig",
    "EdgeStraightnessLoss",
    "EdgeStraightnessLossConfig",
    "FanoutDistributionLoss",
    "FanoutDistributionLossConfig",
    "FlexSpacingLoss",
    "FlexSpacingLossConfig",
    "OverlapAvoidanceLoss",
    "OverlapAvoidanceLossConfig",
    "PositionPinLoss",
    "PositionPinLossConfig",
    "RepulsionLoss",
    "RepulsionLossConfig",
    "SpacingConsistencyLoss",
    "SpacingConsistencyLossConfig",
]
