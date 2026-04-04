"""Classic algorithm loss operations.

This module exposes loss ops that wrap the reference implementations in
``dagua.layout.classic`` so the op vocabulary can reproduce classic
objectives without re-deriving their math inside the ops layer.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Tuple, Union

import torch

from dagua.layout._archive.classic import davidson_harel as _dh
from dagua.layout._archive.classic import linlog as _linlog
from dagua.layout._archive.classic import maxent_stress as _maxent
from dagua.layout._archive.classic import neulay as _neulay
from dagua.layout._archive.classic import sgd2_multi as _sgd2
from dagua.layout._archive.classic import tsnet as _tsnet
from dagua.layout._archive.classic import umap_layout as _umap
from dagua.layout.ops.base import LossOp, Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_UMAP_DEFAULT_MIN_DIST = 0.1
_UMAP_DEFAULT_SPREAD = 1.0
_TSNE_DEFAULT_PERPLEXITY = 30.0
_SGD2_SAMPLER_KEY = "sgd2_samplers"
_SGD2_PREPARED_STATE_KEY = "sgd2_prepared_state"
_SGD2_CROSSING_STATE_KEY = "sgd2_crossing_state"
_SGD2_VERTEX_RESOLUTION_STATE_KEY = "sgd2_vertex_resolution_state"
_SGD2_BATCH_SIZE_KEY = "sgd2_batch_size"
_SGD2_ACTIVE_CRITERION_KEY = "sgd2_active_criterion"
_KD_TREE_PAIR_KEY = "neulay_kdtree_pairs"


@dataclass(frozen=True)
class ExactPairStressLossConfig:
    """Configuration for :class:`ExactPairStressLoss`.

    Parameters
    ----------
    weight_fn : str, default="inverse_sq"
        Stress weight transform. Supported values are ``"inverse_sq"``,
        ``"inverse"``, and ``"uniform"``.
    """

    weight_fn: str = "inverse_sq"


@dataclass(frozen=True)
class KLDivergenceLossConfig:
    """Configuration for :class:`KLDivergenceLoss`.

    Parameters
    ----------
    exaggeration : float, default=12.0
        Early-exaggeration multiplier applied before ``exaggeration_steps``.
    exaggeration_steps : int, default=250
        Number of leading steps that use early exaggeration.
    """

    exaggeration: float = 12.0
    exaggeration_steps: int = 250


@dataclass(frozen=True)
class UMAPCrossEntropyLossConfig:
    """Configuration for :class:`UMAPCrossEntropyLoss`.

    Parameters
    ----------
    neg_rate : int, default=5
        Negative samples per positive sample.
    repulsion_strength : float, default=1.0
        UMAP negative-sample repulsion coefficient ``gamma``.
    """

    neg_rate: int = 5
    repulsion_strength: float = 1.0


@dataclass(frozen=True)
class LinLogAttractionLossConfig:
    """Configuration for :class:`LinLogAttractionLoss`.

    Parameters
    ----------
    exponent_a : float, default=1.0
        Attraction exponent ``a`` from the classic LinLog objective.
    """

    exponent_a: float = 1.0


@dataclass(frozen=True)
class LinLogRepulsionLossConfig:
    """Configuration for :class:`LinLogRepulsionLoss`.

    Parameters
    ----------
    exponent_r : float, default=0.0
        Repulsion exponent ``r`` from the classic LinLog objective.
    """

    exponent_r: float = 0.0


@dataclass(frozen=True)
class LinLogLossConfig:
    """Configuration for :class:`LinLogLoss`.

    Parameters
    ----------
    exponent_a : float, default=1.0
        Attraction exponent ``a`` in ``|p_i - p_j|^a``.
    exponent_r : float, default=0.0
        Repulsion exponent ``r`` in ``-|p_i - p_j|^r``.
    """

    exponent_a: float = 1.0
    exponent_r: float = 0.0


@dataclass(frozen=True)
class EntropyLossConfig:
    """Configuration for :class:`EntropyLoss`.

    Parameters
    ----------
    alpha : float, default=1.0
        Entropy-loss scaling applied to the non-edge term.
    """

    alpha: float = 1.0


@dataclass(frozen=True)
class DavidsonHarelEnergyLossConfig:
    """Configuration for :class:`DavidsonHarelEnergyLoss`.

    Parameters
    ----------
    w_distribution : float, default=1.0
        Weight for node-distribution energy.
    w_border : float, default=0.1
        Weight for border repulsion.
    w_edge_length : float, default=0.2
        Weight for edge-length energy.
    w_crossing : float, default=2.0
        Weight for edge crossing count.
    w_node_edge : float, default=0.5
        Weight for node-edge proximity.
    """

    w_distribution: float = 1.0
    w_border: float = 0.1
    w_edge_length: float = 0.2
    w_crossing: float = 2.0
    w_node_edge: float = 0.5


@dataclass(frozen=True)
class KDTreeRepulsionLossConfig:
    """Configuration for :class:`KDTreeRepulsionLoss`.

    Parameters
    ----------
    radius : float, default=0.4
        Gaussian repulsion radius.
    magnitude : float or str or None, default="auto"
        Repulsion magnitude. ``"auto"`` and ``None`` both resolve to the
        NeuLay adaptive formula ``100 * N^(1/3) * radius``.
    """

    radius: float = 0.4
    magnitude: Union[float, str, None] = "auto"


@dataclass(frozen=True)
class SGD2CriterionLossConfig:
    """Configuration for :class:`SGD2CriterionLoss`.

    Parameters
    ----------
    criterion : str, default="stress"
        One criterion name from the reference multicriteria optimizer.
    batch_size : int, default=16
        Mini-batch size for the sampled criterion evaluation.
    """

    criterion: str = "stress"
    batch_size: int = 16


@dataclass(frozen=True)
class SGD2CrossingDetectorStepConfig:
    """Configuration for :class:`SGD2CrossingDetectorStep`.

    Parameters
    ----------
    inner_steps : int, default=2
        Number of detector training steps before evaluating position loss.
    detector_lr : float, default=0.01
        Adam learning rate for the crossing detector.
    """

    inner_steps: int = 2
    detector_lr: float = 0.01


@dataclass(frozen=True)
class CyclicSamplerConfig:
    """Configuration for :class:`CyclicSampler`.

    Parameters
    ----------
    pool_size : int, default=0
        Explicit sampler pool size. ``0`` means infer the pool size from the
        active SGD2 criterion and prepared state.
    """

    pool_size: int = 0


def _require_positions(state: SolveState) -> torch.Tensor:
    """Return the position tensor or raise a descriptive error.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Current positions with shape ``[N, 2]``.
    """
    if state.pos is None:
        raise ValueError("This op requires `state.pos` to be initialized.")
    return state.pos


def _problem_device(problem: LayoutProblem, state: SolveState) -> torch.device:
    """Resolve the compute device for helper tensors.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.device
        Preferred compute device.
    """
    if state.pos is not None:
        return state.pos.device
    if problem.edge_index.numel() > 0:
        return problem.edge_index.device
    if problem.node_sizes is not None:
        return problem.node_sizes.device
    return torch.device("cpu")


def _resolve_distance_matrix(problem: LayoutProblem, state: SolveState) -> torch.Tensor:
    """Return the all-pairs distance matrix, computing it when absent.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Distance matrix with shape ``[N, N]``.
    """
    if state.distance_matrix is not None:
        return state.distance_matrix

    device = _problem_device(problem, state)
    adjacency = _sgd2._build_adjacency(
        edge_index=problem.edge_index,
        num_nodes=problem.num_nodes,
        edge_weights=problem.edge_weights,
    )
    return _sgd2._all_pairs_shortest_paths(
        adjacency=adjacency,
        device=device,
        weighted=problem.edge_weights is not None,
    )


def _resolve_tsne_probabilities(problem: LayoutProblem, state: SolveState) -> torch.Tensor:
    """Resolve the symmetric t-SNE probability matrix.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Symmetric affinity matrix ``P`` with shape ``[N, N]``.
    """
    if state.affinity_matrix is not None:
        return state.affinity_matrix
    if "tsne_probabilities" in state.extras:
        return state.extras["tsne_probabilities"]

    perplexity = float(state.extras.get("tsne_perplexity", _TSNE_DEFAULT_PERPLEXITY))
    distances = _resolve_distance_matrix(problem, state)
    probabilities = _tsnet._high_dimensional_affinities(
        distances.to(device="cpu", dtype=torch.float32),
        min(perplexity, float(max(problem.num_nodes - 1, 1))),
    )
    return probabilities.to(device=_problem_device(problem, state))


def _resolve_umap_graph(
    problem: LayoutProblem,
    state: SolveState,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]:
    """Resolve the positive graph and curve parameters for UMAP.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.
    device : torch.device
        Device for returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor, torch.Tensor, float, float]
        Positive edge heads, positive edge tails, positive weights, and the
        fitted UMAP ``a`` and ``b`` parameters.
    """
    if {"umap_head", "umap_tail", "umap_weight"} <= state.extras.keys():
        head = state.extras["umap_head"].to(device=device, dtype=torch.long)
        tail = state.extras["umap_tail"].to(device=device, dtype=torch.long)
        weight = state.extras["umap_weight"].to(device=device, dtype=torch.float32)
    else:
        unique_edges, unique_weights = _dh._unique_edges(
            problem.edge_index,
            problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        if unique_edges:
            head = torch.tensor(
                [source for source, _target in unique_edges],
                dtype=torch.long,
                device=device,
            )
            tail = torch.tensor(
                [target for _source, target in unique_edges],
                dtype=torch.long,
                device=device,
            )
            weight = unique_weights.to(device=device, dtype=torch.float32)
        else:
            head = torch.empty((0,), dtype=torch.long, device=device)
            tail = torch.empty((0,), dtype=torch.long, device=device)
            weight = torch.empty((0,), dtype=torch.float32, device=device)

    min_dist = float(state.extras.get("umap_min_dist", _UMAP_DEFAULT_MIN_DIST))
    spread = float(state.extras.get("umap_spread", _UMAP_DEFAULT_SPREAD))
    fit_a, fit_b = _umap._fit_ab(min_dist=min_dist, spread=spread)
    a = float(state.extras.get("umap_a", fit_a))
    b = float(state.extras.get("umap_b", fit_b))
    return head, tail, weight, a, b


def _resolve_exact_stress_pairs(
    problem: LayoutProblem,
    state: SolveState,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Build exact stress distances and upper-triangle node pairs.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.
    device : torch.device
        Device for returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Pair indices with shape ``[2, P]`` and graph distances with shape
        ``[P]``.
    """
    distances = _resolve_distance_matrix(problem, state).to(device=device, dtype=torch.float32)
    upper = torch.triu_indices(distances.shape[0], distances.shape[1], offset=1, device=device)
    pair_distances = distances[upper[0], upper[1]]
    mask = torch.isfinite(pair_distances) & (pair_distances > 0)
    return upper[:, mask], pair_distances[mask]


def _stress_weights(targets: torch.Tensor, weight_fn: str) -> torch.Tensor:
    """Compute stress weights from graph distances.

    Parameters
    ----------
    targets : torch.Tensor
        Positive graph distances with shape ``[P]``.
    weight_fn : str
        Weight transform name.

    Returns
    -------
    torch.Tensor
        Stress weights with shape ``[P]``.
    """
    if weight_fn == "inverse_sq":
        return targets.reciprocal().square()
    if weight_fn == "inverse":
        return targets.reciprocal()
    if weight_fn == "uniform":
        return torch.ones_like(targets)
    raise ValueError(f"Unsupported stress weight_fn: {weight_fn!r}")


def _edge_weight_vector(
    problem: LayoutProblem,
    device: torch.device,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Resolve per-edge weights or ones for the input edge list.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    device : torch.device
        Device for the result.
    dtype : torch.dtype
        Floating dtype for the result.

    Returns
    -------
    torch.Tensor
        Per-edge weights with shape ``[E]``.
    """
    if problem.edge_weights is None:
        return torch.ones((problem.edge_index.shape[1],), dtype=dtype, device=device)
    return problem.edge_weights.to(device=device, dtype=dtype)


def _resolve_non_edge_pairs(
    problem: LayoutProblem,
    state: SolveState,
    device: torch.device,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Resolve exact non-edge pairs for maxent-style entropy.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.
    device : torch.device
        Device for returned tensors.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Non-edge source and destination indices.
    """
    if {"maxent_non_edge_src", "maxent_non_edge_dst"} <= state.extras.keys():
        return (
            state.extras["maxent_non_edge_src"].to(device=device, dtype=torch.long),
            state.extras["maxent_non_edge_dst"].to(device=device, dtype=torch.long),
        )

    adjacency = _maxent._build_undirected_adjacency(
        problem.edge_index,
        problem.num_nodes,
        edge_weights=problem.edge_weights,
    )
    src, dst = _maxent._full_non_edge_pairs(adjacency)
    return src.to(device=device), dst.to(device=device)


def _resolve_kdtree_pairs(pos: torch.Tensor, state: SolveState, radius: float) -> Any:
    """Resolve or refresh cached NeuLay cKDTree pair queries.

    Parameters
    ----------
    pos : torch.Tensor
        Current positions with shape ``[N, 2]``.
    state : SolveState
        Mutable solve state.
    radius : float
        NeuLay Gaussian radius.

    Returns
    -------
    Any
        NumPy array of nearby node pairs.
    """
    query_radius = _neulay._PAIR_QUERY_RADIUS_FACTOR * radius
    cached_pairs = state.extras.get(_KD_TREE_PAIR_KEY)
    cached_radius = state.extras.get("neulay_kdtree_query_radius")
    if cached_pairs is not None and cached_radius == query_radius:
        return cached_pairs

    pairs = _neulay._kdtree_repulsion_pairs(pos=pos, query_radius=query_radius)
    state.extras[_KD_TREE_PAIR_KEY] = pairs
    state.extras["neulay_kdtree_query_radius"] = query_radius
    return pairs


def _resolve_kdtree_magnitude(
    num_nodes: int,
    radius: float,
    magnitude: Union[float, str, None],
) -> float:
    """Resolve the NeuLay repulsion magnitude.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    radius : float
        NeuLay Gaussian radius.
    magnitude : float or str or None
        User-configured magnitude.

    Returns
    -------
    float
        Effective repulsion magnitude.
    """
    if magnitude in {None, "auto"}:
        return 100.0 * float(max(num_nodes, 1)) ** (1.0 / 3.0) * radius
    if isinstance(magnitude, str):
        raise ValueError(f"Unsupported KD-tree repulsion magnitude: {magnitude!r}")
    return float(magnitude)


def _resolve_sgd2_state(
    problem: LayoutProblem,
    state: SolveState,
    device: torch.device,
) -> _sgd2._PreparedState:
    """Resolve the precomputed shared state for SGD2 criteria.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.
    device : torch.device
        Device for the returned prepared state.

    Returns
    -------
    _sgd2._PreparedState
        Shared criterion-precompute state.
    """
    prepared = state.extras.get(_SGD2_PREPARED_STATE_KEY)
    if prepared is not None:
        return prepared

    active_criterion = str(state.extras.get(_SGD2_ACTIVE_CRITERION_KEY, "stress"))
    needs_distances = active_criterion in {"stress", "vertex_resolution"}
    needs_incident = active_criterion == "angular_resolution"
    needs_non_incident = active_criterion in {"crossings", "crossing_angle_maximization"}
    prepared = _sgd2._prepare_state(
        edge_index=problem.edge_index,
        num_nodes=problem.num_nodes,
        device=device,
        needs_distances=needs_distances,
        needs_incident_edge_pairs=needs_incident,
        needs_non_incident_edge_pairs=needs_non_incident,
        edge_weights=problem.edge_weights,
    )
    state.extras[_SGD2_PREPARED_STATE_KEY] = prepared
    return prepared


def _resolve_sgd2_sampler_store(state: SolveState) -> Dict[str, _sgd2._CyclicSampler]:
    """Return the mutable SGD2 sampler dictionary from ``state.extras``.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    dict[str, _sgd2._CyclicSampler]
        Sampler mapping stored in ``state.extras``.
    """
    samplers = state.extras.get(_SGD2_SAMPLER_KEY)
    if samplers is None:
        samplers = {}
        state.extras[_SGD2_SAMPLER_KEY] = samplers
    return samplers


def _infer_sgd2_pool_size(prepared: _sgd2._PreparedState, criterion: str, num_nodes: int) -> int:
    """Infer a cyclic-sampler pool size for one SGD2 criterion.

    Parameters
    ----------
    prepared : _sgd2._PreparedState
        Shared criterion-precompute state.
    criterion : str
        Criterion name.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    int
        Effective pool size.
    """
    if criterion in {"stress", "vertex_resolution"} and prepared.stress_pairs is not None:
        return int(prepared.stress_pairs.shape[1])
    if criterion == "ideal_edge_length":
        return int(prepared.edges.shape[1])
    if criterion in {"neighborhood_preservation", "aspect_ratio"}:
        return num_nodes
    if criterion == "angular_resolution" and prepared.incident_edge_pairs is not None:
        return int(prepared.incident_edge_pairs.shape[1])
    if criterion in {"crossings", "crossing_angle_maximization"}:
        if prepared.non_incident_edge_pairs is None:
            return 0
        return int(prepared.non_incident_edge_pairs.shape[1])
    return 0


def _resolve_sgd2_sampler(
    problem: LayoutProblem,
    state: SolveState,
    criterion: str,
    pool_size: int,
    device: torch.device,
) -> Optional[_sgd2._CyclicSampler]:
    """Resolve or lazily create the cyclic sampler for one criterion.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout problem.
    state : SolveState
        Mutable solve state.
    criterion : str
        Criterion name.
    pool_size : int
        Explicit pool size. ``0`` means infer from prepared state.
    device : torch.device
        Sampler device.

    Returns
    -------
    _sgd2._CyclicSampler or None
        Criterion sampler, or ``None`` when the criterion has no pool.
    """
    samplers = _resolve_sgd2_sampler_store(state)
    if criterion in samplers:
        return samplers[criterion]

    prepared = _resolve_sgd2_state(problem, state, device)
    total = (
        pool_size
        if pool_size > 0
        else _infer_sgd2_pool_size(prepared, criterion, problem.num_nodes)
    )
    if total <= 0:
        return None
    sampler = _sgd2._CyclicSampler(total, device)
    samplers[criterion] = sampler
    return sampler


def _resolve_sgd2_vertex_resolution_state(
    state: SolveState,
    device: torch.device,
) -> Optional[_sgd2._VertexResolutionState]:
    """Resolve the persistent vertex-resolution smoothing state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    device : torch.device
        Device for new tensors.

    Returns
    -------
    _sgd2._VertexResolutionState or None
        Smoothing state used by the vertex-resolution criterion.
    """
    resolved = state.extras.get(_SGD2_VERTEX_RESOLUTION_STATE_KEY)
    if resolved is not None:
        return resolved

    resolved = _sgd2._VertexResolutionState(
        prev_target_dist=torch.tensor(1.0, dtype=torch.float32, device=device),
        prev_weight=0.0,
    )
    state.extras[_SGD2_VERTEX_RESOLUTION_STATE_KEY] = resolved
    return resolved


def _resolve_sgd2_crossing_state(
    state: SolveState,
    device: torch.device,
    inner_steps: int,
    detector_lr: float,
) -> _sgd2._CrossingLossState:
    """Resolve the persistent SGD2 crossing-detector state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.
    device : torch.device
        Device for the detector.
    inner_steps : int
        Detector update count used by the helper op.
    detector_lr : float
        Detector optimizer learning rate.

    Returns
    -------
    _sgd2._CrossingLossState
        Crossing detector state.
    """
    resolved = state.extras.get(_SGD2_CROSSING_STATE_KEY)
    if resolved is not None:
        resolved.inner_steps = inner_steps
        return resolved

    detector = _sgd2._CrossingDetector().to(device=device)
    resolved = _sgd2._CrossingLossState(
        detector=detector,
        optimizer=torch.optim.Adam(detector.parameters(), lr=detector_lr),
        train_loss=torch.nn.BCELoss(),
        position_loss=torch.nn.BCELoss(reduction="sum"),
    )
    # The dataclass is mutable, so storing the helper-op setting on the object
    # keeps the public config separate from the reference dataclass shape.
    resolved.inner_steps = inner_steps  # type: ignore[attr-defined]
    state.extras[_SGD2_CROSSING_STATE_KEY] = resolved
    return resolved


def _crossings_loss_with_override_steps(
    pos: torch.Tensor,
    left: torch.Tensor,
    right: torch.Tensor,
    crossing_state: _sgd2._CrossingLossState,
    inner_steps: int,
) -> torch.Tensor:
    """Evaluate SGD2's crossing loss with configurable detector steps.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.
    left : torch.Tensor
        Left edge batch with shape ``[2, B]``.
    right : torch.Tensor
        Right edge batch with shape ``[2, B]``.
    crossing_state : _sgd2._CrossingLossState
        Persistent detector state.
    inner_steps : int
        Detector updates to run before evaluating the position loss.

    Returns
    -------
    torch.Tensor
        Scalar crossing loss.
    """
    edge_pair_pos = _sgd2._edge_pair_positions(pos=pos, left=left, right=right)
    if edge_pair_pos.numel() == 0:
        return pos.sum() * 0.0

    labels = _sgd2._are_edge_pairs_crossed(edge_pair_pos.detach()).to(
        device=pos.device,
        dtype=pos.dtype,
    )
    crossing_state.detector.train()
    for _ in range(inner_steps):
        preds = crossing_state.detector(edge_pair_pos.detach()).view(-1)
        train_loss = crossing_state.train_loss(preds, labels)
        crossing_state.optimizer.zero_grad(set_to_none=True)
        train_loss.backward()
        crossing_state.optimizer.step()

    crossing_state.detector.eval()
    preds = crossing_state.detector(edge_pair_pos).view(-1)
    return crossing_state.position_loss(preds, torch.zeros_like(preds))


@register_op
class ExactPairStressLoss(LossOp):
    """Exact graph-stress loss over all finite node pairs."""

    name = "exact_pair_stress_loss"
    category = OpCategory.LOSS
    reads = ("pos", "distance_matrix")
    requires = ("pos",)
    weight_key = "stress"

    def __init__(self, config: Optional[ExactPairStressLossConfig] = None) -> None:
        """Store the exact-stress configuration.

        Parameters
        ----------
        config : ExactPairStressLossConfig, optional
            Weighting configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or ExactPairStressLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate exact weighted graph stress.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar stress loss.
        """
        del ctx
        pos = _require_positions(state)
        pairs, targets = _resolve_exact_stress_pairs(problem, state, pos.device)
        if pairs.numel() == 0:
            return pos.sum() * 0.0
        lengths = torch.linalg.norm(pos[pairs[0]] - pos[pairs[1]], dim=1)
        weights = _stress_weights(targets, self.config.weight_fn)
        return (weights * (targets - lengths).square()).sum()


@register_op
class PivotApproxStressLoss(LossOp):
    """Pivot-approximated maxent-stress objective."""

    name = "pivot_approx_stress_loss"
    category = OpCategory.LOSS
    reads = ("pos", "pivot_indices", "pivot_distances")
    requires = ("pos", "pivot_distances")
    weight_key = "stress"

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the pivot-approximated stress term.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar stress loss.
        """
        del problem, ctx
        pos = _require_positions(state)
        pivot_indices = (
            torch.empty((0,), dtype=torch.long, device=pos.device)
            if state.pivot_indices is None
            else state.pivot_indices.to(device=pos.device, dtype=torch.long)
        )
        if state.pivot_distances is None:
            raise ValueError("PivotApproxStressLoss requires `state.pivot_distances`.")
        pivot_distances = state.pivot_distances.to(device=pos.device, dtype=pos.dtype)
        empty_long = torch.empty((0,), dtype=torch.long, device=pos.device)
        empty_float = torch.empty((0,), dtype=pos.dtype, device=pos.device)
        return _maxent._stress_term(
            positions=pos,
            stress_src=empty_long,
            stress_dst=empty_long,
            stress_lengths=empty_float,
            pivot_indices=pivot_indices,
            pivot_distances=pivot_distances,
        )


@register_op
class KLDivergenceLoss(LossOp):
    """Exact t-SNE KL divergence with early exaggeration."""

    name = "kl_divergence_loss"
    category = OpCategory.LOSS
    reads = ("pos", "affinity_matrix", "distance_matrix")
    requires = ("pos",)
    weight_key = "kl"

    def __init__(self, config: Optional[KLDivergenceLossConfig] = None) -> None:
        """Store the t-SNE KL configuration.

        Parameters
        ----------
        config : KLDivergenceLossConfig, optional
            Early-exaggeration settings.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or KLDivergenceLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the t-SNE KL divergence.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar KL divergence.
        """
        del ctx
        pos = _require_positions(state)
        probabilities = _resolve_tsne_probabilities(problem, state).to(
            device=pos.device,
            dtype=pos.dtype,
        )
        exaggeration = (
            self.config.exaggeration if state.step < self.config.exaggeration_steps else 1.0
        )
        return _tsnet._tsne_loss(pos, probabilities * exaggeration)


@register_op
class UMAPCrossEntropyLoss(LossOp):
    """UMAP cross-entropy loss with negative sampling."""

    name = "umap_cross_entropy_loss"
    category = OpCategory.LOSS
    reads = ("pos", "extras.umap_head", "extras.umap_tail", "extras.umap_weight")
    requires = ("pos",)
    weight_key = "umap_ce"
    access_pattern = "sampled"

    def __init__(self, config: Optional[UMAPCrossEntropyLossConfig] = None) -> None:
        """Store the UMAP loss configuration.

        Parameters
        ----------
        config : UMAPCrossEntropyLossConfig, optional
            Negative-sampling configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or UMAPCrossEntropyLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the sampled UMAP cross-entropy objective.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar UMAP loss.
        """
        pos = _require_positions(state)
        head, tail, weight, a, b = _resolve_umap_graph(problem, state, pos.device)
        if head.numel() == 0:
            return pos.sum() * 0.0

        generator = ctx.generator
        if generator is None:
            generator = torch.Generator(device="cpu")
            generator.manual_seed(problem.seed + state.step + 1)

        positive_diff = pos[head] - pos[tail]
        positive_distance_sq = positive_diff.square().sum(dim=1)
        positive_prob = (1.0 + (a * positive_distance_sq.pow(b))).reciprocal()
        positive_loss = -(weight * positive_prob.clamp(min=_umap._EPSILON).log()).sum()

        if self.config.neg_rate <= 0 or self.config.repulsion_strength == 0.0:
            return positive_loss

        negatives = torch.randint(
            0,
            problem.num_nodes,
            (head.shape[0], self.config.neg_rate),
            generator=generator,
            dtype=torch.long,
        ).to(device=pos.device)
        source = head.unsqueeze(1).expand_as(negatives)
        negative_diff = pos[source] - pos[negatives]
        negative_distance_sq = negative_diff.square().sum(dim=2)
        negative_prob = (1.0 + (a * negative_distance_sq.pow(b))).reciprocal()
        negative_loss = (
            -self.config.repulsion_strength
            * torch.log(1.0 - negative_prob.clamp(max=1.0 - _umap._EPSILON)).sum()
        )
        return positive_loss + negative_loss


@register_op
class LinLogAttractionLoss(LossOp):
    """LinLog edge-attraction term from the classic objective."""

    name = "linlog_attraction_loss"
    category = OpCategory.LOSS
    reads = ("pos",)
    requires = ("pos",)
    weight_key = "linlog_attract"

    def __init__(self, config: Optional[LinLogAttractionLossConfig] = None) -> None:
        """Store the LinLog attraction configuration.

        Parameters
        ----------
        config : LinLogAttractionLossConfig, optional
            Attraction exponent configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or LinLogAttractionLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the attraction-only LinLog term.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar attraction loss.
        """
        del ctx
        pos = _require_positions(state)
        if problem.edge_index.numel() == 0:
            return pos.sum() * 0.0
        src = problem.edge_index[0].to(device=pos.device, dtype=torch.long)
        dst = problem.edge_index[1].to(device=pos.device, dtype=torch.long)
        edge_lengths = torch.linalg.norm(pos[src] - pos[dst], dim=1).clamp(
            min=_linlog._MIN_DISTANCE
        )
        weights = _edge_weight_vector(problem, pos.device, edge_lengths.dtype)
        return (weights * edge_lengths.pow(self.config.exponent_a)).sum()


@register_op
class LinLogRepulsionLoss(LossOp):
    """LinLog all-pairs repulsion term from the classic objective."""

    name = "linlog_repulsion_loss"
    category = OpCategory.LOSS
    reads = ("pos",)
    requires = ("pos",)
    weight_key = "linlog_repel"

    def __init__(self, config: Optional[LinLogRepulsionLossConfig] = None) -> None:
        """Store the LinLog repulsion configuration.

        Parameters
        ----------
        config : LinLogRepulsionLossConfig, optional
            Repulsion exponent configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or LinLogRepulsionLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the repulsion-only LinLog term.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar repulsion loss.
        """
        del problem, ctx
        pos = _require_positions(state)
        num_nodes = int(pos.shape[0])
        pair_src, pair_dst = _linlog._full_all_pairs(num_nodes=num_nodes, device=pos.device)
        if pair_src.numel() == 0:
            return pos.sum() * 0.0
        distances = torch.linalg.norm(pos[pair_src] - pos[pair_dst], dim=1).clamp(
            min=_linlog._MIN_DISTANCE
        )
        if self.config.exponent_r == 0.0:
            return -torch.log(distances).sum()
        return -distances.pow(self.config.exponent_r).sum()


@register_op
class LinLogLoss(LossOp):
    """Evaluate the full classic LinLog objective (attraction + repulsion)."""

    name = "linlog_loss"
    category = OpCategory.LOSS
    reads = ("pos", "step")
    requires = ("pos",)

    def __init__(self, config: Optional[LinLogLossConfig] = None) -> None:
        """Store full objective exponents for the LinLog criterion.

        Parameters
        ----------
        config : LinLogLossConfig, optional
            Attraction and repulsion exponents.

        Returns
        -------
        None
            The op stores only its resolved configuration.
        """
        self.config = config or LinLogLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the full objective via the archived helper.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem definition.
        state : SolveState
            Mutable state containing current positions.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar classic LinLog energy value.
        """
        del ctx
        positions = _require_positions(state)
        return _linlog._linlog_loss(
            positions=positions,
            edge_index=problem.edge_index,
            seed=problem.seed,
            step=state.step,
            a=self.config.exponent_a,
            r=self.config.exponent_r,
            edge_weights=problem.edge_weights,
        )


@register_op
class EntropyLoss(LossOp):
    """Maxent-stress non-edge entropy regularizer."""

    name = "entropy_loss"
    category = OpCategory.LOSS
    reads = ("pos", "extras.maxent_non_edge_src", "extras.maxent_non_edge_dst")
    requires = ("pos",)
    weight_key = "entropy"

    def __init__(self, config: Optional[EntropyLossConfig] = None) -> None:
        """Store the entropy-loss configuration.

        Parameters
        ----------
        config : EntropyLossConfig, optional
            Entropy scaling configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or EntropyLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the exact non-edge entropy term.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar entropy loss.
        """
        del ctx
        pos = _require_positions(state)
        src, dst = _resolve_non_edge_pairs(problem, state, pos.device)
        return self.config.alpha * _maxent._entropy_term(pos, src, dst, scale=1.0)


@register_op
class DavidsonHarelEnergyLoss(LossOp):
    """Five-term Davidson-Harel simulated-annealing energy."""

    name = "davidson_harel_energy_loss"
    category = OpCategory.LOSS
    reads = ("pos",)
    requires = ("pos",)
    weight_key = "davidson_harel"

    def __init__(self, config: Optional[DavidsonHarelEnergyLossConfig] = None) -> None:
        """Store the Davidson-Harel energy weights.

        Parameters
        ----------
        config : DavidsonHarelEnergyLossConfig, optional
            Energy-term weights.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or DavidsonHarelEnergyLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate the Davidson-Harel energy scalar.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar energy value.
        """
        del ctx
        pos = _require_positions(state)
        extent = _dh._layout_extent(problem.num_nodes, problem.node_sizes)
        edges, unique_edge_weights = _dh._unique_edges(
            problem.edge_index,
            problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        num_nodes = int(pos.shape[0])
        distribution = torch.tensor(0.0, dtype=pos.dtype, device=pos.device)
        if num_nodes > 1:
            src, dst = torch.triu_indices(num_nodes, num_nodes, offset=1, device=pos.device)
            squared_distances = (
                (pos[src] - pos[dst]).square().sum(dim=1).clamp(min=_dh._MIN_DISTANCE)
            )
            distribution = squared_distances.reciprocal().sum()

        border_distances = torch.stack(
            [
                pos[:, 0] + extent,
                extent - pos[:, 0],
                pos[:, 1] + extent,
                extent - pos[:, 1],
            ],
            dim=1,
        ).clamp(min=_dh._MIN_DISTANCE)
        border = border_distances.reciprocal().square().sum()

        edge_length = torch.tensor(0.0, dtype=pos.dtype, device=pos.device)
        if edges:
            edge_weight_tensor = unique_edge_weights.to(device=pos.device, dtype=pos.dtype)
            edge_lengths = [
                torch.linalg.norm(pos[source] - pos[target]).square() * edge_weight_tensor[index]
                for index, (source, target) in enumerate(edges)
            ]
            edge_length = torch.stack(edge_lengths).sum()

        crossings = 0.0
        for index, (a, b) in enumerate(edges):
            for c, d in edges[index + 1 :]:
                if len({a, b, c, d}) < 4:
                    continue
                if _dh._segments_intersect(pos[a], pos[b], pos[c], pos[d]):
                    crossings += 1.0
        crossing_energy = torch.tensor(crossings, dtype=pos.dtype, device=pos.device)

        penalties = []
        for node in range(num_nodes):
            for source, target in edges:
                if node in (source, target):
                    continue
                distance = _dh._point_segment_distance(pos[node], pos[source], pos[target])
                penalties.append(distance.clamp(min=_dh._MIN_DISTANCE).reciprocal().square())
        node_edge = (
            torch.stack(penalties).sum()
            if penalties
            else torch.tensor(0.0, dtype=pos.dtype, device=pos.device)
        )

        edge_count = len(edges)
        distribution_scale = _dh._scale_denominator(num_nodes * max(num_nodes - 1, 1) // 2)
        border_scale = _dh._scale_denominator(num_nodes)
        edge_length_scale = _dh._scale_denominator(edge_count)
        crossing_scale = _dh._scale_denominator(edge_count * edge_count)
        node_edge_scale = _dh._scale_denominator(num_nodes * edge_count)
        return (
            self.config.w_distribution * (distribution / distribution_scale)
            + self.config.w_border * (border / border_scale)
            + self.config.w_edge_length * (edge_length / edge_length_scale)
            + self.config.w_crossing * (crossing_energy / crossing_scale)
            + self.config.w_node_edge * (node_edge / node_edge_scale)
        )

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Evaluate the non-differentiable energy without calling backward.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with ``prev_loss`` updated.
        """
        loss = self.evaluate(problem, state, ctx)
        state.prev_loss = float(loss.detach().item())
        return state


@register_op
class ElasticLoss(LossOp):
    """NeuLay elastic edge-attraction loss."""

    name = "elastic_loss"
    category = OpCategory.LOSS
    reads = ("pos",)
    requires = ("pos",)
    weight_key = "elastic"

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate NeuLay's elastic loss.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar elastic loss.
        """
        del ctx
        pos = _require_positions(state)
        edge_index = problem.edge_index.to(device=pos.device, dtype=torch.long)
        return _neulay._elastic_loss(pos=pos, edge_index=edge_index)


@register_op
class KDTreeRepulsionLoss(LossOp):
    """NeuLay Gaussian repulsion over cached KD-tree pairs."""

    name = "kdtree_repulsion_loss"
    category = OpCategory.LOSS
    reads = ("pos", "extras.neulay_kdtree_pairs")
    writes = ("extras.neulay_kdtree_pairs",)
    requires = ("pos",)
    weight_key = "kdtree_repel"
    access_pattern = "sampled"

    def __init__(self, config: Optional[KDTreeRepulsionLossConfig] = None) -> None:
        """Store the KD-tree repulsion configuration.

        Parameters
        ----------
        config : KDTreeRepulsionLossConfig, optional
            Repulsion radius and magnitude configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or KDTreeRepulsionLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate NeuLay's Gaussian KD-tree repulsion term.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar repulsion loss.
        """
        del ctx
        pos = _require_positions(state)
        pairs = _resolve_kdtree_pairs(pos=pos, state=state, radius=self.config.radius)
        magnitude = _resolve_kdtree_magnitude(
            num_nodes=problem.num_nodes,
            radius=self.config.radius,
            magnitude=self.config.magnitude,
        )
        return _neulay._kdtree_repulsion_loss(
            pos=pos,
            pairs=pairs,
            radius=self.config.radius,
            magnitude=magnitude,
        )


@register_op
class SGD2CriterionLoss(LossOp):
    """One sampled criterion from the classic (SGD)^2 optimizer."""

    name = "sgd2_criterion_loss"
    category = OpCategory.LOSS
    reads = ("pos", "extras.sgd2_prepared_state", "extras.sgd2_samplers")
    writes = ("extras.sgd2_prepared_state", "extras.sgd2_samplers")
    requires = ("pos",)
    weight_key = "sgd2_criterion"
    access_pattern = "sampled"

    def __init__(self, config: Optional[SGD2CriterionLossConfig] = None) -> None:
        """Store the SGD2 criterion configuration.

        Parameters
        ----------
        config : SGD2CriterionLossConfig, optional
            Criterion name and batch size.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or SGD2CriterionLossConfig()

    def evaluate(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Evaluate one (SGD)^2 criterion on a mini-batch.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        torch.Tensor
            Scalar criterion loss.
        """
        del ctx
        pos = _require_positions(state)
        state.extras[_SGD2_ACTIVE_CRITERION_KEY] = self.config.criterion
        state.extras[_SGD2_BATCH_SIZE_KEY] = self.config.batch_size
        prepared = _resolve_sgd2_state(problem, state, pos.device)
        sampler = _resolve_sgd2_sampler(
            problem=problem,
            state=state,
            criterion=self.config.criterion,
            pool_size=0,
            device=pos.device,
        )
        vertex_resolution_state = None
        if self.config.criterion == "vertex_resolution":
            vertex_resolution_state = _resolve_sgd2_vertex_resolution_state(state, pos.device)
        crossing_state = None
        if self.config.criterion == "crossings":
            crossing_state = _resolve_sgd2_crossing_state(
                state=state,
                device=pos.device,
                inner_steps=_sgd2._CROSSING_DETECTOR_TRAIN_STEPS,
                detector_lr=_sgd2._CROSSING_DETECTOR_LR,
            )
        return _sgd2._criterion_loss(
            name=self.config.criterion,
            pos=pos,
            state=prepared,
            batch_size=self.config.batch_size,
            sampler=sampler,
            vertex_resolution_state=vertex_resolution_state,
            crossing_state=crossing_state,
        )


@register_op
class SGD2CrossingDetectorStep(Op):
    """Train the SGD2 crossing detector and backpropagate crossing loss."""

    name = "sgd2_crossing_detector_step"
    category = OpCategory.LOSS
    reads = ("pos", "extras.sgd2_prepared_state", "extras.sgd2_samplers")
    writes = ("prev_loss", "extras.sgd2_crossing_state", "extras.sgd2_prepared_state")
    requires = ("pos",)
    access_pattern = "sampled"

    def __init__(self, config: Optional[SGD2CrossingDetectorStepConfig] = None) -> None:
        """Store the crossing-detector configuration.

        Parameters
        ----------
        config : SGD2CrossingDetectorStepConfig, optional
            Detector training-step configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or SGD2CrossingDetectorStepConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run one crossing-detector training/evaluation step.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with ``prev_loss`` updated after backpropagation.
        """
        del ctx
        pos = _require_positions(state)
        state.extras[_SGD2_ACTIVE_CRITERION_KEY] = "crossings"
        prepared = _resolve_sgd2_state(problem, state, pos.device)
        crossing_state = _resolve_sgd2_crossing_state(
            state=state,
            device=pos.device,
            inner_steps=self.config.inner_steps,
            detector_lr=self.config.detector_lr,
        )
        sampler = _resolve_sgd2_sampler(
            problem=problem,
            state=state,
            criterion="crossings",
            pool_size=0,
            device=pos.device,
        )
        batch_size = int(state.extras.get(_SGD2_BATCH_SIZE_KEY, 0))
        if batch_size <= 0:
            batch_size = _infer_sgd2_pool_size(prepared, "crossings", problem.num_nodes)
        if prepared.non_incident_edge_pairs is None or batch_size <= 0:
            loss = pos.sum() * 0.0
        else:
            if sampler is not None:
                sample_index = sampler.sample(batch_size)
                pair_batch = prepared.non_incident_edge_pairs[:, sample_index]
            else:
                pair_batch = _sgd2._sample_pairs(
                    prepared.non_incident_edge_pairs,
                    batch_size=batch_size,
                )
            loss = _crossings_loss_with_override_steps(
                pos=pos,
                left=pair_batch[:2],
                right=pair_batch[2:],
                crossing_state=crossing_state,
                inner_steps=self.config.inner_steps,
            )
        loss.backward()
        state.prev_loss = float(loss.detach().item())
        state.extras["sgd2_crossing_loss"] = loss.detach()
        return state


@register_op
class CyclicSampler(Op):
    """Create or refresh an SGD2 cyclic sampler in ``state.extras``."""

    name = "cyclic_sampler"
    category = OpCategory.UTILITY
    reads = ("extras.sgd2_active_criterion", "extras.sgd2_prepared_state")
    writes = ("extras.sgd2_samplers",)

    def __init__(self, config: Optional[CyclicSamplerConfig] = None) -> None:
        """Store the sampler configuration.

        Parameters
        ----------
        config : CyclicSamplerConfig, optional
            Explicit or inferred pool-size configuration.

        Returns
        -------
        None
            The op stores the supplied config.
        """
        self.config = config or CyclicSamplerConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Create or replace the active SGD2 cyclic sampler.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure.

        Returns
        -------
        SolveState
            State with the sampler stored in ``extras``.
        """
        del ctx
        criterion = str(state.extras.get(_SGD2_ACTIVE_CRITERION_KEY, "stress"))
        device = _problem_device(problem, state)
        prepared = _resolve_sgd2_state(problem, state, device)
        total = self.config.pool_size
        if total <= 0:
            total = _infer_sgd2_pool_size(prepared, criterion, problem.num_nodes)
        samplers = _resolve_sgd2_sampler_store(state)
        if total > 0:
            samplers[criterion] = _sgd2._CyclicSampler(total, device)
        return state
