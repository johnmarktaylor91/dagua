"""Bounded guardrail planning for native scale cluster candidates."""

from __future__ import annotations

import time
from dataclasses import dataclass
from typing import Any, Callable, Iterable, Mapping, Optional, Sequence

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.state import LayoutProblem

SMALL_EXACT = "small_exact"
SAMPLED_POLISH = "sampled_polish"
CLUSTER_SKELETON_ONLY = "cluster_skeleton_only"
INCUMBENT_ONLY = "incumbent_only"

DEFAULT_SMALL_EXACT_NODE_CAP = 250
DEFAULT_SMALL_EXACT_WORK_CAP = 50_000
DEFAULT_SAMPLED_POLISH_WORK_CAP = 500_000
DEFAULT_SKELETON_WORK_CAP = 2_000_000
DEFAULT_SIBLING_PAIR_SAMPLE_CAP = 4_096
DEFAULT_EXCLUSION_PAIR_SAMPLE_CAP = 8_192
DEFAULT_EDGE_CLUSTER_OBSTACLE_SAMPLE_CAP = 8_192
DEFAULT_DEADLINE_RESERVE_S = 5.0
DEFAULT_PREDICTED_COST_MULTIPLIER = 2.0


@dataclass(frozen=True)
class NativeGuardrailCaps:
    """Fixed work caps for scale cluster guardrails.

    Parameters
    ----------
    small_exact_node_cap : int
        Largest graph size eligible for exact cluster work.
    small_exact_work_cap : int
        Maximum predicted work units for exact mode.
    sampled_polish_work_cap : int
        Maximum predicted work units for sampled-polish mode.
    skeleton_work_cap : int
        Maximum predicted work units for cluster-skeleton-only mode.
    sibling_pair_sample_cap : int
        Maximum sampled sibling-cluster pairs.
    exclusion_pair_sample_cap : int
        Maximum sampled node/cluster exclusion pairs.
    edge_cluster_obstacle_sample_cap : int
        Maximum sampled edge/cluster obstacle pairs.
    deadline_reserve_s : float
        Seconds reserved for returning an incumbent before a hard deadline.
    predicted_cost_multiplier : float
        Safety multiplier for candidate-level predicted-cost admission.
    """

    small_exact_node_cap: int = DEFAULT_SMALL_EXACT_NODE_CAP
    small_exact_work_cap: int = DEFAULT_SMALL_EXACT_WORK_CAP
    sampled_polish_work_cap: int = DEFAULT_SAMPLED_POLISH_WORK_CAP
    skeleton_work_cap: int = DEFAULT_SKELETON_WORK_CAP
    sibling_pair_sample_cap: int = DEFAULT_SIBLING_PAIR_SAMPLE_CAP
    exclusion_pair_sample_cap: int = DEFAULT_EXCLUSION_PAIR_SAMPLE_CAP
    edge_cluster_obstacle_sample_cap: int = DEFAULT_EDGE_CLUSTER_OBSTACLE_SAMPLE_CAP
    deadline_reserve_s: float = DEFAULT_DEADLINE_RESERVE_S
    predicted_cost_multiplier: float = DEFAULT_PREDICTED_COST_MULTIPLIER


@dataclass(frozen=True)
class NativeGuardrailCostInputs:
    """Structural cost sketch for bounded cluster-scale candidates.

    Parameters
    ----------
    num_nodes : int
        Node count ``N``.
    num_edges : int
        Edge count ``E``.
    cluster_count : int
        Cluster count ``C``.
    max_depth : int
        Maximum cluster nesting depth ``D``.
    total_ancestor_memberships : int
        Sum of ancestor memberships over normalized cluster leaves.
    sibling_pair_count : int
        Count of same-parent cluster pairs before sampling.
    edge_cluster_obstacle_count : int
        Count of candidate edge/cluster obstacle pairs before sampling.
    max_rank_width : int
        Widest available rank/layer.
    estimated_dummy_expansion : int
        Estimated number of dummy nodes implied by long directed edges.
    """

    num_nodes: int
    num_edges: int
    cluster_count: int
    max_depth: int
    total_ancestor_memberships: int
    sibling_pair_count: int
    edge_cluster_obstacle_count: int
    max_rank_width: int
    estimated_dummy_expansion: int

    @property
    def predicted_work_units(self) -> int:
        """Return a deterministic scalar work estimate.

        Returns
        -------
        int
            Additive work sketch. Pairwise terms are represented by their
            capped/sampled counters elsewhere; this property never allocates
            proportional working sets.
        """
        rank_dummy_work = self.max_rank_width * max(1, self.estimated_dummy_expansion)
        return int(
            self.num_nodes
            + self.num_edges
            + self.total_ancestor_memberships
            + self.sibling_pair_count
            + self.edge_cluster_obstacle_count
            + rank_dummy_work
        )


@dataclass(frozen=True)
class NativeGuardrailSamples:
    """Bounded deterministic sample ids for expensive cluster relations.

    Parameters
    ----------
    sibling_pairs : tuple[tuple[str, str], ...]
        Sampled same-parent cluster pairs.
    exclusion_pairs : tuple[tuple[int, str], ...]
        Sampled node/cluster exclusion pairs.
    edge_cluster_obstacles : tuple[tuple[int, str], ...]
        Sampled edge/cluster obstacle pairs.
    """

    sibling_pairs: tuple[tuple[str, str], ...] = ()
    exclusion_pairs: tuple[tuple[int, str], ...] = ()
    edge_cluster_obstacles: tuple[tuple[int, str], ...] = ()


@dataclass(frozen=True)
class NativeGuardrailPlan:
    """Admission and degrade decision for a cluster candidate.

    Parameters
    ----------
    mode : str
        One of ``small_exact``, ``sampled_polish``,
        ``cluster_skeleton_only``, or ``incumbent_only``.
    admitted : bool
        Whether cluster candidate work may start.
    cost_inputs : NativeGuardrailCostInputs
        Structural cost sketch used for this decision.
    caps : NativeGuardrailCaps
        Fixed cap set used for this decision.
    samples : NativeGuardrailSamples
        Bounded deterministic samples available to candidate code.
    predicted_cost_s : float
        Predicted candidate cost in seconds for deadline admission.
    available_work_s : float | None
        Available seconds before reserve, or ``None`` when no deadline exists.
    skip_reason : str | None
        Machine-readable rejection reason.
    """

    mode: str
    admitted: bool
    cost_inputs: NativeGuardrailCostInputs
    caps: NativeGuardrailCaps
    samples: NativeGuardrailSamples
    predicted_cost_s: float
    available_work_s: Optional[float]
    skip_reason: Optional[str] = None


def _flatten_members(members: Any) -> tuple[int, ...]:
    """Return sorted integer leaf ids from nested cluster membership data.

    Parameters
    ----------
    members : Any
        Cluster membership value containing ints, lists, tuples, sets, or
        nested dictionaries.

    Returns
    -------
    tuple[int, ...]
        Sorted unique integer node ids.
    """
    if isinstance(members, int):
        return (int(members),)
    if isinstance(members, Mapping):
        flattened: list[int] = []
        for value in members.values():
            flattened.extend(_flatten_members(value))
        return tuple(sorted(set(flattened)))
    if isinstance(members, (list, tuple, set, frozenset)):
        flattened = []
        for value in members:
            flattened.extend(_flatten_members(value))
        return tuple(sorted(set(flattened)))
    return ()


def _normalized_clusters(problem: LayoutProblem) -> dict[str, tuple[int, ...]]:
    """Return sorted in-range cluster leaves for a problem.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared problem carrying optional cluster metadata.

    Returns
    -------
    dict[str, tuple[int, ...]]
        Normalized cluster leaves keyed by stable cluster name.
    """
    if not problem.clusters:
        return {}
    num_nodes = int(problem.num_nodes)
    return {
        str(name): tuple(node for node in _flatten_members(members) if 0 <= node < num_nodes)
        for name, members in sorted(problem.clusters.items(), key=lambda item: str(item[0]))
    }


def _cluster_depth(name: str, parents: Mapping[str, Optional[str]]) -> int:
    """Return bounded ancestor depth for one cluster.

    Parameters
    ----------
    name : str
        Cluster name.
    parents : Mapping[str, str | None]
        Parent mapping keyed by cluster name.

    Returns
    -------
    int
        Number of ancestors including the cluster itself. Cycles terminate
        conservatively at the first repeated name.
    """
    depth = 0
    seen: set[str] = set()
    current: Optional[str] = name
    while current is not None and current not in seen:
        seen.add(current)
        depth += 1
        current = parents.get(current)
    return depth


def _sibling_groups(
    cluster_names: Sequence[str],
    parents: Mapping[str, Optional[str]],
) -> list[list[str]]:
    """Group clusters by parent without enumerating sibling pairs.

    Parameters
    ----------
    cluster_names : Sequence[str]
        Stable cluster names.
    parents : Mapping[str, str | None]
        Parent mapping keyed by cluster name.

    Returns
    -------
    list[list[str]]
        Sorted groups of sibling cluster names.
    """
    grouped: dict[Optional[str], list[str]] = {}
    for name in cluster_names:
        grouped.setdefault(parents.get(name), []).append(name)
    return [sorted(group) for _, group in sorted(grouped.items(), key=lambda item: str(item[0]))]


def _rank_stats(edge_index: torch.Tensor, num_nodes: int) -> tuple[int, int]:
    """Estimate rank width and dummy expansion from a deterministic DAG pass.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Node count.

    Returns
    -------
    tuple[int, int]
        Maximum rank width and estimated dummy-node expansion. Cyclic
        leftovers stay on rank zero, matching the conservative cost-sketch
        purpose rather than asserting DAG fidelity.
    """
    if num_nodes <= 0:
        return 0, 0
    if edge_index.numel() == 0:
        return num_nodes, 0
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
    indegree = [0] * num_nodes
    for tail_raw, head_raw in edges.t().tolist():
        tail = int(tail_raw)
        head = int(head_raw)
        if 0 <= tail < num_nodes and 0 <= head < num_nodes:
            outgoing[tail].append(head)
            indegree[head] += 1
    queue = [index for index, degree in enumerate(indegree) if degree == 0]
    ranks = [0] * num_nodes
    visited = 0
    cursor = 0
    while cursor < len(queue):
        node = queue[cursor]
        cursor += 1
        visited += 1
        for head in sorted(outgoing[node]):
            ranks[head] = max(ranks[head], ranks[node] + 1)
            indegree[head] -= 1
            if indegree[head] == 0:
                queue.append(head)
    if visited < num_nodes:
        ranks = [rank if indegree[index] == 0 else 0 for index, rank in enumerate(ranks)]
    rank_counts: dict[int, int] = {}
    for rank in ranks:
        rank_counts[rank] = rank_counts.get(rank, 0) + 1
    dummy_expansion = 0
    for tail_raw, head_raw in edges.t().tolist():
        tail = int(tail_raw)
        head = int(head_raw)
        if 0 <= tail < num_nodes and 0 <= head < num_nodes:
            dummy_expansion += max(0, abs(ranks[head] - ranks[tail]) - 1)
    return max(rank_counts.values(), default=0), int(dummy_expansion)


def build_native_guardrail_cost_inputs(problem: LayoutProblem) -> NativeGuardrailCostInputs:
    """Build the reusable native scale-guardrail cost sketch.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared native layout problem.

    Returns
    -------
    NativeGuardrailCostInputs
        Structural counts over ``N``, ``E``, cluster hierarchy, sibling pairs,
        edge/cluster obstacle pairs, rank width, and dummy expansion.
    """
    num_nodes = int(problem.num_nodes)
    num_edges = int(problem.edge_index.shape[1]) if problem.edge_index.ndim == 2 else 0
    clusters = _normalized_clusters(problem)
    parents = {
        str(key): (None if value is None else str(value))
        for key, value in (problem.cluster_parents or {}).items()
    }
    names = tuple(sorted(clusters))
    depths = {name: _cluster_depth(name, parents) for name in names}
    sibling_pair_count = sum(
        len(group) * (len(group) - 1) // 2 for group in _sibling_groups(names, parents)
    )
    total_ancestor_memberships = sum(len(clusters[name]) * depths[name] for name in names)
    rank_width = int(problem.structure.max_layer_width) if problem.structure is not None else 0
    dummy_expansion = 0
    if rank_width <= 0:
        rank_width, dummy_expansion = _rank_stats(problem.edge_index, num_nodes)
    return NativeGuardrailCostInputs(
        num_nodes=num_nodes,
        num_edges=num_edges,
        cluster_count=len(names),
        max_depth=max(depths.values(), default=0),
        total_ancestor_memberships=int(total_ancestor_memberships),
        sibling_pair_count=int(sibling_pair_count),
        edge_cluster_obstacle_count=int(num_edges * len(names)),
        max_rank_width=int(rank_width),
        estimated_dummy_expansion=int(dummy_expansion),
    )


def _sample_index(total: int, ordinal: int, salt: int) -> int:
    """Return one deterministic sample index in ``[0, total)``.

    Parameters
    ----------
    total : int
        Population size.
    ordinal : int
        Sample ordinal.
    salt : int
        Stable per-family salt.

    Returns
    -------
    int
        Population index.
    """
    if total <= 0:
        return 0
    stride = max(1, (total // max(1, min(total, 997))) | 1)
    return int((ordinal * stride + salt) % total)


def _pair_at(group: Sequence[str], pair_index: int) -> tuple[str, str]:
    """Return a same-parent sibling pair without materializing all pairs.

    Parameters
    ----------
    group : Sequence[str]
        Sorted sibling cluster names.
    pair_index : int
        Zero-based pair index within ``len(group) choose 2``.

    Returns
    -------
    tuple[str, str]
        Stable cluster-name pair.
    """
    remaining = int(pair_index)
    for left_index, left_name in enumerate(group[:-1]):
        width = len(group) - left_index - 1
        if remaining < width:
            return left_name, group[left_index + 1 + remaining]
        remaining -= width
    return group[0], group[-1]


def _dedupe_preserve_order(items: Iterable[tuple[Any, ...]]) -> tuple[tuple[Any, ...], ...]:
    """Deduplicate sampled tuple items while preserving first occurrence.

    Parameters
    ----------
    items : Iterable[tuple[Any, ...]]
        Sampled tuple payloads.

    Returns
    -------
    tuple[tuple[Any, ...], ...]
        Deduplicated tuple payloads.
    """
    out: list[tuple[Any, ...]] = []
    seen: set[tuple[Any, ...]] = set()
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return tuple(out)


def build_native_guardrail_samples(
    problem: LayoutProblem,
    caps: NativeGuardrailCaps = NativeGuardrailCaps(),
) -> NativeGuardrailSamples:
    """Build deterministic capped samples for expensive cluster relations.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared native layout problem.
    caps : NativeGuardrailCaps, default=NativeGuardrailCaps()
        Fixed sample caps.

    Returns
    -------
    NativeGuardrailSamples
        Bounded sampled sibling, exclusion, and edge-obstacle relations.
    """
    clusters = _normalized_clusters(problem)
    names = tuple(sorted(clusters))
    parents = {
        str(key): (None if value is None else str(value))
        for key, value in (problem.cluster_parents or {}).items()
    }
    sibling_groups = _sibling_groups(names, parents)
    sibling_total = sum(len(group) * (len(group) - 1) // 2 for group in sibling_groups)
    sibling_budget = min(int(caps.sibling_pair_sample_cap), sibling_total)
    sibling_samples: list[tuple[str, str]] = []
    cumulative = []
    total_seen = 0
    for group in sibling_groups:
        count = len(group) * (len(group) - 1) // 2
        cumulative.append((total_seen, total_seen + count, group))
        total_seen += count
    for ordinal in range(sibling_budget):
        sampled = _sample_index(sibling_total, ordinal, 17)
        for start, end, group in cumulative:
            if start <= sampled < end:
                sibling_samples.append(_pair_at(group, sampled - start))
                break

    exclusion_total = int(problem.num_nodes) * len(names)
    exclusion_budget = min(int(caps.exclusion_pair_sample_cap), exclusion_total)
    exclusion_samples = []
    for ordinal in range(exclusion_budget):
        sampled = _sample_index(exclusion_total, ordinal, 31)
        name_count = max(1, len(names))
        exclusion_samples.append((sampled // name_count, names[sampled % name_count]))

    edge_count = int(problem.edge_index.shape[1]) if problem.edge_index.ndim == 2 else 0
    obstacle_total = edge_count * len(names)
    obstacle_budget = min(int(caps.edge_cluster_obstacle_sample_cap), obstacle_total)
    obstacle_samples = []
    for ordinal in range(obstacle_budget):
        sampled = _sample_index(obstacle_total, ordinal, 47)
        name_count = max(1, len(names))
        obstacle_samples.append((sampled // name_count, names[sampled % name_count]))
    return NativeGuardrailSamples(
        sibling_pairs=tuple(_dedupe_preserve_order(sibling_samples)),
        exclusion_pairs=tuple(_dedupe_preserve_order(exclusion_samples)),
        edge_cluster_obstacles=tuple(_dedupe_preserve_order(obstacle_samples)),
    )


def native_guardrail_available_work_s(
    config: Optional[LayoutConfig],
    reserve_s: float = DEFAULT_DEADLINE_RESERVE_S,
) -> Optional[float]:
    """Return deadline seconds available before the incumbent-return reserve.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared native configuration carrying optional deadline attributes.
    reserve_s : float, default=DEFAULT_DEADLINE_RESERVE_S
        Seconds held back for returning the already-registered incumbent.

    Returns
    -------
    float | None
        Available seconds clamped to zero, or ``None`` when no deadline is
        known.
    """
    if config is None:
        return None
    wall_deadline = getattr(config, "_dagua_native_deadline_s", None)
    if wall_deadline is not None:
        remaining = float(wall_deadline) - time.perf_counter()
        if remaining <= float(reserve_s):
            return 0.0
    process_deadline = getattr(config, "_dagua_native_process_deadline_s", None)
    if process_deadline is None:
        return None
    return max(0.0, float(process_deadline) - time.process_time() - float(reserve_s))


def _mode_for_cost(cost: NativeGuardrailCostInputs, caps: NativeGuardrailCaps) -> str:
    """Return the highest-quality bounded mode admitted by structural cost.

    Parameters
    ----------
    cost : NativeGuardrailCostInputs
        Structural cost sketch.
    caps : NativeGuardrailCaps
        Fixed cap set.

    Returns
    -------
    str
        Degrade mode name.
    """
    work_units = cost.predicted_work_units
    if cost.num_nodes <= caps.small_exact_node_cap and work_units <= caps.small_exact_work_cap:
        return SMALL_EXACT
    if work_units <= caps.sampled_polish_work_cap:
        return SAMPLED_POLISH
    if work_units <= caps.skeleton_work_cap:
        return CLUSTER_SKELETON_ONLY
    return INCUMBENT_ONLY


def build_native_guardrail_plan(
    problem: LayoutProblem,
    config: Optional[LayoutConfig],
    *,
    caps: NativeGuardrailCaps = NativeGuardrailCaps(),
    prior_cost_s: float = 1.0,
) -> NativeGuardrailPlan:
    """Build a deterministic candidate-level admission/degrade plan.

    Parameters
    ----------
    problem : LayoutProblem
        Prepared native layout problem.
    config : LayoutConfig, optional
        Prepared configuration with optional deadline metadata.
    caps : NativeGuardrailCaps, default=NativeGuardrailCaps()
        Fixed structural and sample caps.
    prior_cost_s : float, default=1.0
        Candidate-family prior cost in process seconds. Callers may update
        this from telemetry after observed runs; source constants stay fixed.

    Returns
    -------
    NativeGuardrailPlan
        Deterministic plan containing mode, bounded samples, and skip reason.
    """
    cost = build_native_guardrail_cost_inputs(problem)
    mode = _mode_for_cost(cost, caps)
    samples = build_native_guardrail_samples(problem, caps)
    available = native_guardrail_available_work_s(config, reserve_s=caps.deadline_reserve_s)
    predicted_cost_s = float(prior_cost_s) * float(caps.predicted_cost_multiplier)
    if mode == INCUMBENT_ONLY:
        return NativeGuardrailPlan(
            mode=mode,
            admitted=False,
            cost_inputs=cost,
            caps=caps,
            samples=samples,
            predicted_cost_s=predicted_cost_s,
            available_work_s=available,
            skip_reason="structural_cap",
        )
    if available is not None and available <= predicted_cost_s:
        return NativeGuardrailPlan(
            mode=INCUMBENT_ONLY,
            admitted=False,
            cost_inputs=cost,
            caps=caps,
            samples=samples,
            predicted_cost_s=predicted_cost_s,
            available_work_s=available,
            skip_reason="insufficient_predicted_budget",
        )
    return NativeGuardrailPlan(
        mode=mode,
        admitted=True,
        cost_inputs=cost,
        caps=caps,
        samples=samples,
        predicted_cost_s=predicted_cost_s,
        available_work_s=available,
    )


def register_native_guardrail_observation(
    config: Optional[LayoutConfig],
    *,
    candidate: str,
    mode: str,
    elapsed_s: float,
) -> None:
    """Record observed guardrail cost telemetry on the runtime config.

    Parameters
    ----------
    config : LayoutConfig, optional
        Prepared configuration receiving private telemetry.
    candidate : str
        Stable candidate-family name.
    mode : str
        Degrade mode used by the observed run.
    elapsed_s : float
        Non-negative observed process seconds.

    Returns
    -------
    None
        Telemetry is stored on ``config`` only; no source constants or
        row-specific behavior are mutated.
    """
    if config is None:
        return
    records = list(getattr(config, "_dagua_native_guardrail_observations", []))
    records.append(
        {
            "candidate": str(candidate),
            "mode": str(mode),
            "elapsed_s": max(0.0, float(elapsed_s)),
        }
    )
    setattr(config, "_dagua_native_guardrail_observations", records)


def run_guarded_native_cluster_candidate(
    *,
    plan: NativeGuardrailPlan,
    incumbent_pos: torch.Tensor,
    candidate_fn: Callable[[NativeGuardrailPlan], torch.Tensor],
) -> torch.Tensor:
    """Run an admitted cluster candidate or return the full incumbent.

    Parameters
    ----------
    plan : NativeGuardrailPlan
        Admission/degrade plan.
    incumbent_pos : torch.Tensor
        Full incumbent positions with shape ``[N, 2]`` registered before any
        cluster work starts.
    candidate_fn : Callable[[NativeGuardrailPlan], torch.Tensor]
        Candidate factory invoked only when the plan is admitted.

    Returns
    -------
    torch.Tensor
        Full finite positions. Deadline/degrade and invalid candidate outputs
        return the incumbent, never a partial recursive expansion.
    """
    if not plan.admitted or plan.mode == INCUMBENT_ONLY:
        return incumbent_pos
    try:
        candidate = candidate_fn(plan)
    except TimeoutError:
        return incumbent_pos
    if not isinstance(candidate, torch.Tensor):
        return incumbent_pos
    if candidate.shape != incumbent_pos.shape or not bool(torch.isfinite(candidate).all().item()):
        return incumbent_pos
    return candidate.to(device=incumbent_pos.device, dtype=incumbent_pos.dtype)
