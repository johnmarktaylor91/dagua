"""Strongly connected component condensation operations for hybrid layouts."""

from __future__ import annotations

import copy
import math
import sys
from dataclasses import dataclass, field
from typing import ClassVar, Tuple

import torch

from dagua.config import LayoutConfig
from dagua.layout.ops.base import Op
from dagua.layout.ops.pipelines.native_layered_dag import layout_native_layered_dag_pipeline
from dagua.layout.ops.pipelines.native_stress import (
    NativeStressConfig,
    layout_native_stress_pipeline,
)
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_SCC_EXTRA_KEY = "scc_condensation"
_SCC_INTERNAL_OFFSETS_KEY = "scc_internal_offsets"
_SCC_BBOX_SIZES_KEY = "scc_bbox_sizes"
_SCC_META_POS_KEY = "scc_meta_pos"
_SCC_EXPANDED_BBOX_KEY = "scc_expanded_bbox"
_DEFAULT_INTERNAL_MIN = 5
_DEFAULT_BBOX_PADDING = 24.0


@dataclass(frozen=True)
class SCCPredicateStats:
    """Summary statistics used by hybrid-v2 routing.

    Parameters
    ----------
    total_nodes : int
        Number of input nodes.
    covered_nodes : int
        Nodes inside nontrivial SCCs. A singleton self-loop is nontrivial.
    max_scc_size : int
        Largest nontrivial SCC size.
    coverage_ratio : float
        ``covered_nodes / total_nodes`` with ``0`` for an empty graph.
    nontrivial_count : int
        Number of nontrivial SCCs.
    """

    total_nodes: int
    covered_nodes: int
    max_scc_size: int
    coverage_ratio: float
    nontrivial_count: int


@dataclass(frozen=True)
class SCCCondensation:
    """Condensed SCC graph and expansion metadata.

    Parameters
    ----------
    component_ids : torch.Tensor
        Original-node to SCC id mapping with shape ``[N]``.
    components : tuple[tuple[int, ...], ...]
        Original node ids per SCC, indexed by SCC id.
    node_local_indices : torch.Tensor
        Original-node local index inside its SCC with shape ``[N]``.
    meta_edge_index : torch.Tensor
        Deduplicated condensation DAG edges with shape ``[2, E_meta]``.
    meta_edge_multiplicity : torch.Tensor
        Edge multiplicity for each condensation edge with shape ``[E_meta]``.
    internal_edge_indices : tuple[torch.Tensor, ...]
        Per-SCC local edge tensors, each shaped ``[2, E_internal]``.
    meta_layers : torch.Tensor
        Longest-path layers on the condensation DAG with shape ``[C]``.
    stats : SCCPredicateStats
        Routing statistics for the original graph.
    """

    component_ids: torch.Tensor
    components: Tuple[Tuple[int, ...], ...]
    node_local_indices: torch.Tensor
    meta_edge_index: torch.Tensor
    meta_edge_multiplicity: torch.Tensor
    internal_edge_indices: Tuple[torch.Tensor, ...]
    meta_layers: torch.Tensor
    stats: SCCPredicateStats


@dataclass(frozen=True)
class SCCCondenseConfig:
    """Configuration for SCC condensation.

    Parameters
    ----------
    include_self_loops : bool, default=True
        Whether singleton self-loops count as nontrivial SCC coverage.
    """

    include_self_loops: bool = True


@dataclass(frozen=True)
class SCCInternalLayoutConfig:
    """Configuration for internal SCC layouts.

    Parameters
    ----------
    internal_min : int, default=5
        SCC size at which the normal native-stress budget is used.
    small_steps : int, default=24
        Stress-SGD steps for small SCCs.
    large_steps : int, default=0
        Stress-SGD steps for large SCCs. ``0`` uses native-stress auto budget.
    bbox_padding : float, default=24.0
        Extra reserved padding around each SCC bounding box.
    seed : int, default=42
        Deterministic base seed for per-SCC stress layouts.
    """

    internal_min: int = _DEFAULT_INTERNAL_MIN
    small_steps: int = 24
    large_steps: int = 0
    bbox_padding: float = _DEFAULT_BBOX_PADDING
    seed: int = 42


@dataclass(frozen=True)
class SCCMetaLayoutConfig:
    """Configuration for condensation-DAG layout.

    Parameters
    ----------
    node_sep : float, default=70.0
        Horizontal spacing forwarded to the layered machinery.
    rank_sep : float, default=240.0
        Vertical rank spacing forwarded to the layered machinery.
    steps : int, default=0
        Native layered-DAG optimization steps.
    seed : int, default=42
        Deterministic layout seed.
    device : str, default="cpu"
        Execution device.
    """

    node_sep: float = 70.0
    rank_sep: float = 240.0
    steps: int = 0
    seed: int = 42
    device: str = "cpu"


@dataclass(frozen=True)
class SCCExpandConfig:
    """Configuration for expanding SCC members around meta-node positions.

    Parameters
    ----------
    center_output : bool, default=True
        Whether to center expanded positions around the origin.
    """

    center_output: bool = True


def compute_scc_predicate_stats(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    include_self_loops: bool = True,
) -> SCCPredicateStats:
    """Compute SCC coverage statistics for routing.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    include_self_loops : bool, default=True
        Whether singleton self-loop SCCs count as nontrivial.

    Returns
    -------
    SCCPredicateStats
        Nontrivial SCC coverage and max-size summary.
    """
    condensation = build_scc_condensation(
        edge_index=edge_index,
        num_nodes=num_nodes,
        include_self_loops=include_self_loops,
    )
    return condensation.stats


def hybrid_v2_predicate_matches(
    stats: SCCPredicateStats,
    *,
    min_coverage: float = 0.25,
    min_max_scc_size: int = 10,
) -> bool:
    """Return whether SCC statistics satisfy the hybrid-v2 routing gate.

    Parameters
    ----------
    stats : SCCPredicateStats
        SCC coverage summary for the candidate graph.
    min_coverage : float, default=0.25
        Minimum nontrivial SCC coverage fraction.
    min_max_scc_size : int, default=10
        Minimum largest nontrivial SCC size.

    Returns
    -------
    bool
        ``True`` when the dossier predicate is satisfied.
    """
    return stats.coverage_ratio > min_coverage and stats.max_scc_size >= min_max_scc_size


def build_scc_condensation(
    edge_index: torch.Tensor,
    num_nodes: int,
    *,
    include_self_loops: bool = True,
) -> SCCCondensation:
    """Build SCCs and the condensation DAG.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    include_self_loops : bool, default=True
        Whether singleton self-loops count as nontrivial SCCs in stats.

    Returns
    -------
    SCCCondensation
        SCC membership, condensation edges, internal edges, and rank metadata.
    """
    edge_cpu = _validate_edge_index(edge_index=edge_index, num_nodes=num_nodes)
    adjacency = _directed_adjacency(edge_cpu=edge_cpu, num_nodes=num_nodes)
    components_raw = _tarjan_scc(adjacency=adjacency)
    components = tuple(tuple(sorted(component)) for component in components_raw)
    component_ids, node_local_indices = _component_index_tensors(
        components=components,
        num_nodes=num_nodes,
    )
    self_loop_nodes = _self_loop_nodes(edge_cpu=edge_cpu, num_nodes=num_nodes)
    meta_edge_index, meta_edge_multiplicity = _condensation_edges(
        edge_cpu=edge_cpu,
        component_ids=component_ids,
    )
    internal_edge_indices = _internal_edges_by_component(
        edge_cpu=edge_cpu,
        component_ids=component_ids,
        node_local_indices=node_local_indices,
        components=components,
    )
    meta_layers = _longest_path_layers(edge_index=meta_edge_index, num_nodes=len(components))
    stats = _predicate_stats(
        components=components,
        self_loop_nodes=self_loop_nodes,
        total_nodes=num_nodes,
        include_self_loops=include_self_loops,
    )
    return SCCCondensation(
        component_ids=component_ids,
        components=components,
        node_local_indices=node_local_indices,
        meta_edge_index=meta_edge_index,
        meta_edge_multiplicity=meta_edge_multiplicity,
        internal_edge_indices=internal_edge_indices,
        meta_layers=meta_layers,
        stats=stats,
    )


def _validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return a CPU long edge tensor after shape and bounds validation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Candidate edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        CPU long edge tensor with shape ``[2, E]``.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_cpu.ndim != 2 or edge_cpu.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_cpu.numel() == 0:
        return edge_cpu
    if int(edge_cpu.min().item()) < 0 or int(edge_cpu.max().item()) >= num_nodes:
        raise ValueError("edge_index contains node ids outside [0, num_nodes).")
    return edge_cpu


def _directed_adjacency(edge_cpu: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Build directed adjacency lists.

    Parameters
    ----------
    edge_cpu : torch.Tensor
        CPU long edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Out-neighbor lists indexed by source node.
    """
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in zip(edge_cpu[0].tolist(), edge_cpu[1].tolist()):
        adjacency[int(source)].append(int(target))
    return adjacency


def _tarjan_scc(adjacency: list[list[int]]) -> list[list[int]]:
    """Compute strongly connected components with Tarjan's algorithm.

    Parameters
    ----------
    adjacency : list[list[int]]
        Directed adjacency lists.

    Returns
    -------
    list[list[int]]
        Components in deterministic discovery order.
    """
    num_nodes = len(adjacency)
    sys.setrecursionlimit(max(sys.getrecursionlimit(), (2 * num_nodes) + 100))
    index = 0
    stack: list[int] = []
    on_stack = [False] * num_nodes
    indices = [-1] * num_nodes
    lowlinks = [0] * num_nodes
    components: list[list[int]] = []

    def strongconnect(node: int) -> None:
        """Visit one DFS subtree and emit completed SCCs."""
        nonlocal index
        indices[node] = index
        lowlinks[node] = index
        index += 1
        stack.append(node)
        on_stack[node] = True

        for target in adjacency[node]:
            if indices[target] == -1:
                strongconnect(target)
                lowlinks[node] = min(lowlinks[node], lowlinks[target])
            elif on_stack[target]:
                lowlinks[node] = min(lowlinks[node], indices[target])

        if lowlinks[node] != indices[node]:
            return
        component: list[int] = []
        while True:
            member = stack.pop()
            on_stack[member] = False
            component.append(member)
            if member == node:
                break
        components.append(component)

    for node in range(num_nodes):
        if indices[node] == -1:
            strongconnect(node)
    return components


def _component_index_tensors(
    components: Tuple[Tuple[int, ...], ...],
    num_nodes: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return node-to-component and node-to-local-index tensors.

    Parameters
    ----------
    components : tuple[tuple[int, ...], ...]
        Node ids per SCC.
    num_nodes : int
        Number of original graph nodes.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        ``(component_ids, node_local_indices)`` each shaped ``[N]``.
    """
    component_ids = torch.empty((num_nodes,), dtype=torch.long)
    node_local_indices = torch.empty((num_nodes,), dtype=torch.long)
    for component_id, members in enumerate(components):
        for local_index, node in enumerate(members):
            component_ids[int(node)] = int(component_id)
            node_local_indices[int(node)] = int(local_index)
    return component_ids, node_local_indices


def _self_loop_nodes(edge_cpu: torch.Tensor, num_nodes: int) -> set[int]:
    """Return nodes that have at least one self-loop.

    Parameters
    ----------
    edge_cpu : torch.Tensor
        CPU long edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    set[int]
        Node ids with ``u -> u`` edges.
    """
    del num_nodes
    return {
        int(source)
        for source, target in zip(edge_cpu[0].tolist(), edge_cpu[1].tolist())
        if int(source) == int(target)
    }


def _condensation_edges(
    edge_cpu: torch.Tensor,
    component_ids: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Deduplicate inter-SCC edges and retain multiplicities.

    Parameters
    ----------
    edge_cpu : torch.Tensor
        CPU long edge tensor with shape ``[2, E]``.
    component_ids : torch.Tensor
        Original-node to SCC id mapping with shape ``[N]``.

    Returns
    -------
    tuple[torch.Tensor, torch.Tensor]
        Meta edge index ``[2, E_meta]`` and multiplicities ``[E_meta]``.
    """
    counts: dict[tuple[int, int], int] = {}
    for source, target in zip(edge_cpu[0].tolist(), edge_cpu[1].tolist()):
        source_component = int(component_ids[int(source)].item())
        target_component = int(component_ids[int(target)].item())
        if source_component == target_component:
            continue
        key = (source_component, target_component)
        counts[key] = counts.get(key, 0) + 1
    if not counts:
        return torch.zeros((2, 0), dtype=torch.long), torch.zeros((0,), dtype=torch.float32)
    ordered = sorted(counts)
    meta_edge_index = torch.tensor(ordered, dtype=torch.long).t().contiguous()
    meta_edge_multiplicity = torch.tensor([counts[key] for key in ordered], dtype=torch.float32)
    return meta_edge_index, meta_edge_multiplicity


def _internal_edges_by_component(
    edge_cpu: torch.Tensor,
    component_ids: torch.Tensor,
    node_local_indices: torch.Tensor,
    components: Tuple[Tuple[int, ...], ...],
) -> Tuple[torch.Tensor, ...]:
    """Build local internal edge tensors for each SCC.

    Parameters
    ----------
    edge_cpu : torch.Tensor
        CPU long edge tensor with shape ``[2, E]``.
    component_ids : torch.Tensor
        Original-node to SCC id mapping with shape ``[N]``.
    node_local_indices : torch.Tensor
        Original-node local index inside its SCC with shape ``[N]``.
    components : tuple[tuple[int, ...], ...]
        Node ids per SCC.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Per-SCC local edge tensors.
    """
    buckets: list[list[tuple[int, int]]] = [[] for _ in components]
    for source, target in zip(edge_cpu[0].tolist(), edge_cpu[1].tolist()):
        source_component = int(component_ids[int(source)].item())
        target_component = int(component_ids[int(target)].item())
        if source_component != target_component:
            continue
        buckets[source_component].append(
            (
                int(node_local_indices[int(source)].item()),
                int(node_local_indices[int(target)].item()),
            )
        )
    out: list[torch.Tensor] = []
    for bucket in buckets:
        if bucket:
            out.append(torch.tensor(bucket, dtype=torch.long).t().contiguous())
        else:
            out.append(torch.zeros((2, 0), dtype=torch.long))
    return tuple(out)


def _longest_path_layers(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute longest-path layers for a DAG edge set.

    Parameters
    ----------
    edge_index : torch.Tensor
        DAG edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of DAG nodes.

    Returns
    -------
    torch.Tensor
        Layer assignments with shape ``[N]``.
    """
    if num_nodes <= 0:
        return torch.zeros((0,), dtype=torch.long)
    indegree = [0] * num_nodes
    outgoing: list[list[int]] = [[] for _ in range(num_nodes)]
    for source, target in zip(edge_index[0].tolist(), edge_index[1].tolist()):
        outgoing[int(source)].append(int(target))
        indegree[int(target)] += 1
    queue = [node for node, degree in enumerate(indegree) if degree == 0]
    queue_index = 0
    layers = [0] * num_nodes
    while queue_index < len(queue):
        node = queue[queue_index]
        queue_index += 1
        for target in outgoing[node]:
            layers[target] = max(layers[target], layers[node] + 1)
            indegree[target] -= 1
            if indegree[target] == 0:
                queue.append(target)
    return torch.tensor(layers, dtype=torch.long)


def _predicate_stats(
    components: Tuple[Tuple[int, ...], ...],
    self_loop_nodes: set[int],
    total_nodes: int,
    *,
    include_self_loops: bool,
) -> SCCPredicateStats:
    """Summarize nontrivial SCC coverage.

    Parameters
    ----------
    components : tuple[tuple[int, ...], ...]
        Node ids per SCC.
    self_loop_nodes : set[int]
        Nodes with self-loop edges.
    total_nodes : int
        Number of graph nodes.
    include_self_loops : bool
        Whether singleton self-loops count as nontrivial.

    Returns
    -------
    SCCPredicateStats
        Coverage summary.
    """
    covered_nodes = 0
    max_scc_size = 0
    nontrivial_count = 0
    for members in components:
        is_nontrivial = len(members) > 1 or (
            include_self_loops and len(members) == 1 and int(members[0]) in self_loop_nodes
        )
        if not is_nontrivial:
            continue
        covered_nodes += len(members)
        max_scc_size = max(max_scc_size, len(members))
        nontrivial_count += 1
    coverage_ratio = float(covered_nodes) / float(total_nodes) if total_nodes > 0 else 0.0
    return SCCPredicateStats(
        total_nodes=total_nodes,
        covered_nodes=covered_nodes,
        max_scc_size=max_scc_size,
        coverage_ratio=coverage_ratio,
        nontrivial_count=nontrivial_count,
    )


def _component_node_sizes(problem: LayoutProblem, members: Tuple[int, ...]) -> torch.Tensor:
    """Return node sizes for one SCC.

    Parameters
    ----------
    problem : LayoutProblem
        Parent layout problem.
    members : tuple[int, ...]
        Original node ids in component-local order.

    Returns
    -------
    torch.Tensor
        CPU float node-size tensor with shape ``[K, 2]``.
    """
    if problem.node_sizes is None:
        return torch.full((len(members), 2), 20.0, dtype=torch.float32)
    index = torch.tensor(members, dtype=torch.long, device=problem.node_sizes.device)
    return problem.node_sizes.index_select(0, index).detach().to(device="cpu", dtype=torch.float32)


def _ring_positions(num_nodes: int, node_sizes: torch.Tensor) -> torch.Tensor:
    """Return deterministic circular positions for tiny SCC fallbacks.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the SCC.
    node_sizes : torch.Tensor
        Node sizes with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Centered circular positions with shape ``[N, 2]``.
    """
    if num_nodes <= 1:
        return torch.zeros((num_nodes, 2), dtype=torch.float32)
    radius = max(float(node_sizes.max().item()), 1.0) * max(float(num_nodes), 3.0) / math.tau
    angles = torch.linspace(0.0, math.tau, num_nodes + 1, dtype=torch.float32)[:-1]
    return torch.stack([torch.cos(angles) * radius, torch.sin(angles) * radius], dim=1)


def _center_offsets(pos: torch.Tensor) -> torch.Tensor:
    """Center an offset tensor around the origin.

    Parameters
    ----------
    pos : torch.Tensor
        Position tensor with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Centered position tensor with shape ``[N, 2]``.
    """
    if pos.numel() == 0:
        return pos.detach().to(device="cpu", dtype=torch.float32)
    centered = pos.detach().to(device="cpu", dtype=torch.float32)
    return centered - centered.mean(dim=0, keepdim=True)


def _bbox_size(
    offsets: torch.Tensor,
    node_sizes: torch.Tensor,
    padding: float,
) -> torch.Tensor:
    """Return the reserved bounding-box size for one SCC.

    Parameters
    ----------
    offsets : torch.Tensor
        Centered member offsets with shape ``[K, 2]``.
    node_sizes : torch.Tensor
        Node sizes with shape ``[K, 2]``.
    padding : float
        Extra padding around the component.

    Returns
    -------
    torch.Tensor
        Width/height tensor with shape ``[2]``.
    """
    if offsets.numel() == 0:
        return torch.zeros((2,), dtype=torch.float32)
    half_sizes = node_sizes * 0.5
    low = (offsets - half_sizes).min(dim=0).values
    high = (offsets + half_sizes).max(dim=0).values
    return (high - low).clamp_min(1.0) + (2.0 * float(padding))


@register_op
@dataclass(frozen=True)
class SCCCondense(Op):
    """Compute SCCs and the condensation DAG."""

    config: SCCCondenseConfig = field(default_factory=SCCCondenseConfig)

    name: ClassVar[str] = "scc_condense"
    category: ClassVar[OpCategory] = OpCategory.PREPROCESS
    writes: ClassVar[Tuple[str, ...]] = ("extras.scc_condensation",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Store SCC condensation data in ``state.extras``.

        Parameters
        ----------
        problem : LayoutProblem
            Directed graph layout problem.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused by this deterministic op.

        Returns
        -------
        SolveState
            State with ``extras["scc_condensation"]`` populated.
        """
        del ctx
        state.extras[_SCC_EXTRA_KEY] = build_scc_condensation(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            include_self_loops=self.config.include_self_loops,
        )
        return state


@register_op
@dataclass(frozen=True)
class SCCLayoutInternals(Op):
    """Lay out each SCC internally and compute reserved boxes."""

    config: SCCInternalLayoutConfig = field(default_factory=SCCInternalLayoutConfig)

    name: ClassVar[str] = "scc_layout_internals"
    category: ClassVar[OpCategory] = OpCategory.EMBED
    reads: ClassVar[Tuple[str, ...]] = ("extras.scc_condensation",)
    writes: ClassVar[Tuple[str, ...]] = ("extras.scc_internal_offsets", "extras.scc_bbox_sizes")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute centered member offsets for every SCC.

        Parameters
        ----------
        problem : LayoutProblem
            Parent graph problem.
        state : SolveState
            Mutable solve state containing SCC condensation metadata.
        ctx : RuntimeContext
            Runtime context. The device is inherited by native-stress calls.

        Returns
        -------
        SolveState
            State with internal offsets and bounding boxes populated.
        """
        condensation = _require_condensation(state)
        offsets: list[torch.Tensor] = []
        bbox_sizes: list[torch.Tensor] = []
        for component_id, members in enumerate(condensation.components):
            node_sizes = _component_node_sizes(problem=problem, members=members)
            if len(members) <= 1:
                component_offsets = torch.zeros((len(members), 2), dtype=torch.float32)
            else:
                component_offsets = self._layout_component(
                    condensation=condensation,
                    component_id=component_id,
                    node_sizes=node_sizes,
                    ctx=ctx,
                )
            component_offsets = _center_offsets(component_offsets)
            offsets.append(component_offsets)
            bbox_sizes.append(_bbox_size(component_offsets, node_sizes, self.config.bbox_padding))
        state.extras[_SCC_INTERNAL_OFFSETS_KEY] = tuple(offsets)
        state.extras[_SCC_BBOX_SIZES_KEY] = (
            torch.stack(bbox_sizes).to(dtype=torch.float32)
            if bbox_sizes
            else torch.zeros((0, 2), dtype=torch.float32)
        )
        return state

    def _layout_component(
        self,
        condensation: SCCCondensation,
        component_id: int,
        node_sizes: torch.Tensor,
        ctx: RuntimeContext,
    ) -> torch.Tensor:
        """Run native-stress or circular fallback for one SCC.

        Parameters
        ----------
        condensation : SCCCondensation
            Parent SCC metadata.
        component_id : int
            SCC id to lay out.
        node_sizes : torch.Tensor
            Component node sizes with shape ``[K, 2]``.
        ctx : RuntimeContext
            Runtime context containing the preferred device.

        Returns
        -------
        torch.Tensor
            Component-local offsets with shape ``[K, 2]``.
        """
        num_nodes = int(node_sizes.shape[0])
        edge_index = condensation.internal_edge_indices[component_id]
        if num_nodes < self.config.internal_min:
            stress_config = NativeStressConfig(
                steps=self.config.small_steps,
                late_steps=max(4, self.config.small_steps // 6),
                n_pivots=max(2, num_nodes),
                smacof_iters=1,
                overlap_iterations=3,
                target_aspect=1.0,
                seed=self.config.seed + component_id,
            )
        else:
            stress_config = NativeStressConfig(
                steps=self.config.large_steps,
                target_aspect=1.0,
                seed=self.config.seed + component_id,
            )
        try:
            return layout_native_stress_pipeline(
                edge_index=edge_index.to(device=ctx.plan.device),
                num_nodes=num_nodes,
                node_sizes=node_sizes.to(device=ctx.plan.device),
                config=stress_config,
                seed=self.config.seed + component_id,
                target_aspect=1.0,
            ).detach()
        except Exception:
            return _ring_positions(num_nodes=num_nodes, node_sizes=node_sizes)


@register_op
@dataclass(frozen=True)
class SCCLayoutCondensationDAG(Op):
    """Lay out the condensation DAG while reserving SCC bounding boxes."""

    config: SCCMetaLayoutConfig = field(default_factory=SCCMetaLayoutConfig)

    name: ClassVar[str] = "scc_layout_condensation_dag"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = ("extras.scc_condensation", "extras.scc_bbox_sizes")
    writes: ClassVar[Tuple[str, ...]] = ("extras.scc_meta_pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the existing layered machinery on the SCC condensation DAG.

        Parameters
        ----------
        problem : LayoutProblem
            Parent graph problem. Used for seed fallback only.
        state : SolveState
            Mutable solve state with SCC boxes populated.
        ctx : RuntimeContext
            Runtime context containing the active device.

        Returns
        -------
        SolveState
            State with meta-node positions populated.
        """
        condensation = _require_condensation(state)
        bbox_sizes = _require_bbox_sizes(state)
        num_meta = len(condensation.components)
        if num_meta == 0:
            state.extras[_SCC_META_POS_KEY] = torch.zeros((0, 2), dtype=torch.float32)
            return state
        if num_meta == 1:
            state.extras[_SCC_META_POS_KEY] = torch.zeros((1, 2), dtype=torch.float32)
            return state

        seed = self.config.seed if self.config.seed is not None else problem.seed
        meta_config = LayoutConfig(
            steps=self.config.steps,
            node_sep=self.config.node_sep,
            rank_sep=self.config.rank_sep,
            device=self.config.device or ctx.plan.device,
            seed=seed,
            force_pipeline="layered_dag",
            edge_equalize_polish=False,
        )
        meta_config.insert_dummy_nodes = False
        meta_config.brandes_koepf_refine = True
        meta_pos = layout_native_layered_dag_pipeline(
            edge_index=condensation.meta_edge_index.to(device=ctx.plan.device),
            num_nodes=num_meta,
            node_sizes=bbox_sizes.to(device=ctx.plan.device),
            config=meta_config,
            device=ctx.plan.device,
            seed=seed,
            edge_weights=condensation.meta_edge_multiplicity.to(device=ctx.plan.device),
        )
        state.extras[_SCC_META_POS_KEY] = meta_pos.detach().to(device="cpu", dtype=torch.float32)
        return state


@register_op
@dataclass(frozen=True)
class SCCExpand(Op):
    """Expand SCC member positions around their condensation meta-nodes."""

    config: SCCExpandConfig = field(default_factory=SCCExpandConfig)

    name: ClassVar[str] = "scc_expand"
    category: ClassVar[OpCategory] = OpCategory.POSTPROCESS
    reads: ClassVar[Tuple[str, ...]] = (
        "extras.scc_condensation",
        "extras.scc_internal_offsets",
        "extras.scc_meta_pos",
    )
    writes: ClassVar[Tuple[str, ...]] = ("pos", "layers", "extras.scc_expanded_bbox")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Combine meta positions and internal SCC offsets.

        Parameters
        ----------
        problem : LayoutProblem
            Parent layout problem.
        state : SolveState
            Mutable state carrying SCC metadata and meta positions.
        ctx : RuntimeContext
            Runtime context containing the output device.

        Returns
        -------
        SolveState
            State with original-node positions restored in ``state.pos``.
        """
        condensation = _require_condensation(state)
        internal_offsets = _require_internal_offsets(state)
        meta_pos = _require_meta_pos(state)
        output = torch.zeros((problem.num_nodes, 2), dtype=torch.float32)
        for component_id, members in enumerate(condensation.components):
            base = meta_pos[component_id].view(1, 2)
            offsets = internal_offsets[component_id]
            for local_index, node in enumerate(members):
                output[int(node)] = base[0] + offsets[int(local_index)]
        if self.config.center_output and output.numel() > 0:
            output -= output.mean(dim=0, keepdim=True)
        state.pos = output.to(device=ctx.plan.device, dtype=torch.float32)
        state.layers = condensation.meta_layers[condensation.component_ids].to(
            device=ctx.plan.device
        )
        state.extras[_SCC_EXPANDED_BBOX_KEY] = copy.copy(state.extras.get(_SCC_BBOX_SIZES_KEY))
        return state


def _require_condensation(state: SolveState) -> SCCCondensation:
    """Return SCC condensation metadata from state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    SCCCondensation
        Stored SCC metadata.
    """
    condensation = state.extras.get(_SCC_EXTRA_KEY)
    if not isinstance(condensation, SCCCondensation):
        raise ValueError("SCCCondense must run before SCC hybrid-v2 ops.")
    return condensation


def _require_bbox_sizes(state: SolveState) -> torch.Tensor:
    """Return SCC bounding-box sizes from state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Box sizes with shape ``[C, 2]``.
    """
    bbox_sizes = state.extras.get(_SCC_BBOX_SIZES_KEY)
    if not isinstance(bbox_sizes, torch.Tensor):
        raise ValueError("SCCLayoutInternals must populate SCC bbox sizes.")
    return bbox_sizes


def _require_internal_offsets(state: SolveState) -> Tuple[torch.Tensor, ...]:
    """Return SCC internal offsets from state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    tuple[torch.Tensor, ...]
        Per-SCC offset tensors.
    """
    offsets = state.extras.get(_SCC_INTERNAL_OFFSETS_KEY)
    if not isinstance(offsets, tuple) or not all(
        isinstance(item, torch.Tensor) for item in offsets
    ):
        raise ValueError("SCCLayoutInternals must populate SCC offsets.")
    return offsets


def _require_meta_pos(state: SolveState) -> torch.Tensor:
    """Return condensation-DAG positions from state.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor
        Meta-node positions with shape ``[C, 2]``.
    """
    meta_pos = state.extras.get(_SCC_META_POS_KEY)
    if not isinstance(meta_pos, torch.Tensor):
        raise ValueError("SCCLayoutCondensationDAG must populate meta positions.")
    return meta_pos


__all__ = [
    "SCCCondensation",
    "SCCCondense",
    "SCCCondenseConfig",
    "SCCExpand",
    "SCCExpandConfig",
    "SCCInternalLayoutConfig",
    "SCCLayoutCondensationDAG",
    "SCCLayoutInternals",
    "SCCMetaLayoutConfig",
    "SCCPredicateStats",
    "build_scc_condensation",
    "compute_scc_predicate_stats",
    "hybrid_v2_predicate_matches",
]
