"""Preprocessing operations for composable layout pipelines.

These ops prepare graph structure before layout proper: cycle detection,
acyclic rewrites, topology classification, adjacency construction, and
connected-component detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import numpy as np
import torch
from scipy import sparse

from dagua.layout.cycle import detect_back_edges, make_acyclic, make_acyclic_robust
from dagua.layout.graph_classify import classify_graph as _classify_graph_reference
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import GraphStructure, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_ACYCLIC_EDGE_KEY = "preprocess_edge_index"
_ADJ_FORMAT_KEY = "adjacency_format"
_ADJ_DIRECTED_KEY = "adjacency_directed"
_ADJ_WEIGHTED_KEY = "adjacency_weighted"
_SYMMETRIC_FLAG_KEY = "spectral_is_symmetric"

AdjacencyList = List[List[Tuple[int, float]]]
CSRAdjacency = Dict[str, torch.Tensor]
AdjacencyValue = Union[AdjacencyList, CSRAdjacency, torch.Tensor]


def _resolved_edge_index(problem: LayoutProblem, state: SolveState) -> torch.Tensor:
    """Return the active edge tensor for preprocessing.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state that may carry an acyclic edge rewrite.

    Returns
    -------
    torch.Tensor
        Active edge tensor with shape ``[2, E]``.
    """
    edge_index = state.extras.get(_ACYCLIC_EDGE_KEY)
    if isinstance(edge_index, torch.Tensor):
        return edge_index
    return problem.edge_index


def _resolve_layer_assignments(state: SolveState) -> Optional[torch.Tensor]:
    """Return optional layer assignments on CPU.

    Parameters
    ----------
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.Tensor | None
        Layer assignments with shape ``[N]`` on CPU, or ``None`` when absent.
    """
    if state.layers is None:
        return None
    return state.layers.detach().to(device="cpu", dtype=torch.long)


def _layer_stats(layer_assignments: Optional[torch.Tensor]) -> Tuple[int, int]:
    """Compute depth and maximum layer width from layer assignments.

    Parameters
    ----------
    layer_assignments : torch.Tensor | None
        Layer IDs with shape ``[N]``.

    Returns
    -------
    tuple[int, int]
        ``(depth, max_layer_width)``. Both are zero when layering is absent.
    """
    if layer_assignments is None or layer_assignments.numel() == 0:
        return 0, 0
    max_layer = int(layer_assignments.max().item())
    counts = torch.bincount(layer_assignments, minlength=max_layer + 1)
    max_width = int(counts.max().item()) if counts.numel() > 0 else 0
    return max_layer, max_width


def _family_name(family: Any) -> str:
    """Normalize graph-family enums into the solve-state schema.

    Parameters
    ----------
    family : Any
        Family enum or string from the reference classifier.

    Returns
    -------
    str
        Lowercase family name.
    """
    if hasattr(family, "name"):
        return str(family.name).lower()
    return str(family).lower()


def _build_problem_structure(
    problem: LayoutProblem,
    state: SolveState,
    large_graph_cutoff: int,
) -> GraphStructure:
    """Classify the active graph and map the result into ``ops.state.GraphStructure``.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    GraphStructure
        Structure descriptor stored on ``problem.structure``.
    """
    edge_index = _resolved_edge_index(problem, state)
    layers = _resolve_layer_assignments(state)
    if problem.num_nodes > large_graph_cutoff:
        num_layers = 0 if layers is None or layers.numel() == 0 else int(layers.max().item()) + 1
        avg_layer_width = 0.0 if num_layers == 0 else float(problem.num_nodes) / float(num_layers)
        reference: Any = {
            "family": "general",
            "max_degree": 0,
            "num_layers": num_layers,
            "avg_layer_width": avg_layer_width,
            "num_components": 1,
        }
    else:
        reference = _classify_graph_reference(
            edge_index=edge_index.detach().to(device="cpu", dtype=torch.long),
            num_nodes=problem.num_nodes,
            layer_assignments=layers,
        )
    depth, max_layer_width = _layer_stats(layers)
    reference_num_layers = int(
        reference["num_layers"] if isinstance(reference, dict) else reference.num_layers
    )
    reference_avg_width = float(
        reference["avg_layer_width"] if isinstance(reference, dict) else reference.avg_layer_width
    )
    reference_components = int(
        reference["num_components"] if isinstance(reference, dict) else reference.num_components
    )
    reference_max_degree = int(
        reference["max_degree"] if isinstance(reference, dict) else reference.max_degree
    )
    reference_family = reference["family"] if isinstance(reference, dict) else reference.family
    if layers is None and reference_num_layers > 0:
        depth = max(reference_num_layers - 1, 0)
        if problem.num_nodes > 0:
            estimated_width = problem.num_nodes / float(reference_num_layers)
            max_layer_width = max(int(round(estimated_width)), 1)

    if state.back_edge_mask is None:
        is_dag = not bool(detect_back_edges(edge_index, problem.num_nodes).any())
    else:
        is_dag = not bool(state.back_edge_mask.any())

    num_edges = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
    return GraphStructure(
        family=_family_name(reference_family),
        num_nodes=problem.num_nodes,
        num_edges=num_edges,
        max_degree=reference_max_degree,
        depth=depth,
        max_layer_width=max_layer_width,
        num_layers=reference_num_layers,
        avg_layer_width=reference_avg_width,
        is_dag=is_dag,
        num_components=reference_components,
    )


def _resolve_weights(
    problem: LayoutProblem,
    num_edges: int,
    weighted: bool,
    weight_transform: str,
) -> List[float]:
    """Resolve per-edge costs for adjacency construction.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    num_edges : int
        Number of edges in the active ``edge_index``.
    weighted : bool
        Whether to use ``problem.edge_weights``.
    weight_transform : str
        Weight transform name.

    Returns
    -------
    list[float]
        One transformed cost per edge.

    Raises
    ------
    ValueError
        If the weight transform is unknown.
    """
    if weighted and problem.edge_weights is not None:
        weights = problem.edge_weights.detach().to(device="cpu", dtype=torch.float64).tolist()
    else:
        weights = [1.0] * num_edges

    if weight_transform == "none":
        return [float(weight) for weight in weights]
    if weight_transform == "inverse":
        return [float("inf") if float(weight) == 0.0 else 1.0 / float(weight) for weight in weights]
    raise ValueError(f"Unsupported weight_transform: {weight_transform!r}.")


def _aggregate_neighbor_weights(
    values: Sequence[float],
    dedup: str,
) -> List[float]:
    """Aggregate duplicate edge weights for one neighbor bucket.

    Parameters
    ----------
    values : sequence[float]
        All edge weights for the same source-target pair.
    dedup : str
        Aggregation policy.

    Returns
    -------
    list[float]
        One or more output weights for the pair.

    Raises
    ------
    ValueError
        If the dedup mode is unknown.
    """
    if dedup == "keep_all":
        return [float(value) for value in values]
    if dedup == "min":
        return [float(min(values))]
    if dedup == "sum":
        return [float(sum(values))]
    raise ValueError(f"Unsupported dedup mode: {dedup!r}.")


def _build_fr_adjacency_matrix(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    """Build the directed, dense adjacency matrix used by FR force ops.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed edge list with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional per-edge weights with shape ``[E]``.

    Returns
    -------
    torch.Tensor
        Dense adjacency matrix with shape ``[N, N]`` and dtype ``float64``.

    Raises
    ------
    ValueError
        If ``edge_index`` is malformed or contains out-of-range indices.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    adjacency = torch.zeros((num_nodes, num_nodes), dtype=torch.float64)
    if edge_index.numel() == 0:
        return adjacency

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    sources = edge_index_cpu[0]
    targets = edge_index_cpu[1]
    if (
        torch.any(sources < 0)
        or torch.any(sources >= num_nodes)
        or torch.any(targets < 0)
        or torch.any(targets >= num_nodes)
    ):
        raise ValueError("edge_index contains a node index outside [0, num_nodes).")

    if edge_weights is not None:
        weights = edge_weights.detach().to(device="cpu", dtype=torch.float64)
    else:
        weights = torch.ones(edge_index_cpu.shape[1], dtype=torch.float64)

    # NetworkX's DiGraph keeps the last inserted edge attribute for repeated
    # directed pairs. Assign in input order instead of relying on repeated
    # advanced-index writes, whose ordering is backend-dependent.
    for source, target, weight in zip(sources.tolist(), targets.tolist(), weights.tolist()):
        adjacency[int(source), int(target)] = float(weight)
    return adjacency


def _append_edge(
    adjacency: List[Dict[int, List[float]]],
    source: int,
    target: int,
    weight: float,
) -> None:
    """Append one weighted edge into an intermediate adjacency map.

    Parameters
    ----------
    adjacency : list[dict[int, list[float]]]
        Intermediate adjacency container.
    source : int
        Source node index.
    target : int
        Target node index.
    weight : float
        Edge cost.

    Returns
    -------
    None
        The adjacency map is mutated in place.
    """
    bucket = adjacency[source].setdefault(target, [])
    bucket.append(weight)


def _build_list_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    weights: Sequence[float],
    directed: bool,
    dedup: str,
    keep_multiplicity: bool,
) -> AdjacencyList:
    """Build a deterministic list adjacency from the edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes.
    weights : sequence[float]
        Per-edge costs aligned to ``edge_index``.
    directed : bool
        Whether to preserve edge direction.
    dedup : str
        Duplicate-edge aggregation policy.
    keep_multiplicity : bool
        Whether to preserve duplicate edges as separate entries.

    Returns
    -------
    list[list[tuple[int, float]]]
        Weighted adjacency list.
    """
    if int(edge_index.numel()) == 0:
        return [[] for _ in range(num_nodes)]

    src = edge_index.detach().to(device="cpu", dtype=torch.long)[0].tolist()
    tgt = edge_index.detach().to(device="cpu", dtype=torch.long)[1].tolist()

    if keep_multiplicity:
        adjacency: AdjacencyList = [[] for _ in range(num_nodes)]
        for source, target, weight in zip(src, tgt, weights):
            if not directed and source == target:
                continue
            adjacency[source].append((target, float(weight)))
            if not directed:
                adjacency[target].append((source, float(weight)))
        return [sorted(neighbors, key=lambda item: (item[0], item[1])) for neighbors in adjacency]

    adjacency_map: List[Dict[int, List[float]]] = [{} for _ in range(num_nodes)]
    for source, target, weight in zip(src, tgt, weights):
        if not directed and source == target:
            continue
        _append_edge(adjacency_map, source, target, float(weight))
        if not directed:
            _append_edge(adjacency_map, target, source, float(weight))

    adjacency: AdjacencyList = [[] for _ in range(num_nodes)]
    for node_idx, neighbors in enumerate(adjacency_map):
        expanded: List[Tuple[int, float]] = []
        for neighbor, values in neighbors.items():
            for reduced in _aggregate_neighbor_weights(values, dedup=dedup):
                expanded.append((neighbor, reduced))
        adjacency[node_idx] = sorted(expanded, key=lambda item: (item[0], item[1]))
    return adjacency


def _build_dense_adjacency(
    adjacency: AdjacencyList,
    num_nodes: int,
) -> torch.Tensor:
    """Build a dense adjacency-cost matrix from the list representation.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Weighted adjacency list.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Dense ``[N, N]`` cost matrix with ``inf`` for missing off-diagonal
        edges and zeros on the diagonal.
    """
    dense = torch.full((num_nodes, num_nodes), float("inf"), dtype=torch.float64)
    if num_nodes == 0:
        return dense
    dense.fill_diagonal_(0.0)
    for source, neighbors in enumerate(adjacency):
        for target, weight in neighbors:
            dense[source, target] = float(weight)
    return dense


def _build_csr_adjacency(adjacency: AdjacencyList) -> CSRAdjacency:
    """Build a CSR-like adjacency payload from the list representation.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Weighted adjacency list.

    Returns
    -------
    dict[str, torch.Tensor]
        CSR payload containing ``indptr``, ``indices``, and ``weights``.
    """
    indptr_values: List[int] = [0]
    index_values: List[int] = []
    weight_values: List[float] = []
    for neighbors in adjacency:
        for target, weight in neighbors:
            index_values.append(int(target))
            weight_values.append(float(weight))
        indptr_values.append(len(index_values))

    return {
        "indptr": torch.tensor(indptr_values, dtype=torch.long),
        "indices": torch.tensor(index_values, dtype=torch.long),
        "weights": torch.tensor(weight_values, dtype=torch.float64),
    }


def _find_root(parents: List[int], node: int) -> int:
    """Find a union-find root with path compression.

    Parameters
    ----------
    parents : list[int]
        Union-find parent array.
    node : int
        Node index to resolve.

    Returns
    -------
    int
        Canonical component root.
    """
    while parents[node] != node:
        parents[node] = parents[parents[node]]
        node = parents[node]
    return node


def _component_labels(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Compute weakly connected components from an edge tensor.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        Component IDs with shape ``[N]``.
    """
    if num_nodes == 0:
        return torch.empty((0,), dtype=torch.long)

    parents = list(range(num_nodes))
    ranks = [0] * num_nodes
    if int(edge_index.numel()) > 0:
        cpu_edges = edge_index.detach().to(device="cpu", dtype=torch.long)
        sources = cpu_edges[0].tolist()
        targets = cpu_edges[1].tolist()
        for source, target in zip(sources, targets):
            source_root = _find_root(parents, source)
            target_root = _find_root(parents, target)
            if source_root == target_root:
                continue
            if ranks[source_root] < ranks[target_root]:
                parents[source_root] = target_root
            elif ranks[source_root] > ranks[target_root]:
                parents[target_root] = source_root
            else:
                parents[target_root] = source_root
                ranks[source_root] += 1

    root_to_component: Dict[int, int] = {}
    labels = torch.empty((num_nodes,), dtype=torch.long)
    next_component = 0
    for node_idx in range(num_nodes):
        root = _find_root(parents, node_idx)
        if root not in root_to_component:
            root_to_component[root] = next_component
            next_component += 1
        labels[node_idx] = root_to_component[root]
    return labels


@dataclass(frozen=True)
class DetectCyclesConfig:
    """Configuration for ``DetectCycles``.

    Parameters
    ----------
    method : str, default="dfs_then_greedy"
        Cycle-detection strategy. ``"dfs"`` keeps the plain DFS back-edge
        mask from :mod:`dagua.layout.cycle`. ``"dfs_then_greedy"`` upgrades
        that mask with the robust greedy fallback when DFS alone does not
        produce an acyclic orientation.
    """

    method: str = "dfs_then_greedy"


@register_op
class DetectCycles(Op):
    """Detect cycle-forming edges and store the reversal mask.

    Reads
    -----
    ``state.extras["preprocess_edge_index"]`` when an earlier op already
    rewrote the active edge view.

    Writes
    ------
    ``state.back_edge_mask``.

    Use this when
    -------------
    You need a reusable mask for acyclic rewrites, DAG-only layering, or graph
    classification without mutating ``problem.edge_index``.
    """

    name = "detect_cycles"
    category = OpCategory.PREPROCESS
    reads = ("extras.preprocess_edge_index",)
    writes = ("back_edge_mask",)

    def __init__(self, config: Optional[DetectCyclesConfig] = None) -> None:
        """Initialize the cycle-detection op.

        Parameters
        ----------
        config : DetectCyclesConfig | None, optional
            Optional op configuration.
        """
        self.config = config or DetectCyclesConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Detect edges that should be reversed to break cycles.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``back_edge_mask`` populated.

        Raises
        ------
        ValueError
            If the configured detection method is unsupported.
        """
        _ = ctx
        edge_index = _resolved_edge_index(problem, state)
        if self.config.method == "dfs":
            state.back_edge_mask = detect_back_edges(edge_index, problem.num_nodes)
            return state
        if self.config.method == "dfs_then_greedy":
            # The robust helper preserves the DFS fast path but falls back to a
            # greedy repair when the plain back-edge mask still leaves cycles.
            _, reversed_mask = make_acyclic_robust(edge_index, problem.num_nodes)
            state.back_edge_mask = reversed_mask.to(dtype=torch.bool)
            return state
        raise ValueError(f"Unsupported cycle-detection method: {self.config.method!r}.")


@register_op
class MakeAcyclic(Op):
    """Materialize an acyclic edge view into ``state.extras``.

    Reads
    -----
    ``state.back_edge_mask`` when already computed.

    Writes
    ------
    ``state.extras["preprocess_edge_index"]``.

    Use this when
    -------------
    Downstream ops should consume a DAG view while the original
    ``problem.edge_index`` remains unchanged for provenance or later reuse.
    """

    name = "make_acyclic"
    category = OpCategory.PREPROCESS
    reads = ("back_edge_mask",)
    writes = ("extras.preprocess_edge_index",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Reverse the detected back edges into a transient edge tensor.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``extras["preprocess_edge_index"]`` populated.
        """
        _ = ctx
        back_edge_mask = state.back_edge_mask
        if back_edge_mask is None:
            # Missing masks mean "no reversals requested", which keeps this op
            # safe to run in conservative pipelines that skip cycle detection.
            back_edge_mask = torch.zeros(problem.edge_index.shape[1], dtype=torch.bool)
        state.extras[_ACYCLIC_EDGE_KEY] = make_acyclic(problem.edge_index, back_edge_mask)
        return state


@dataclass(frozen=True)
class ClassifyGraphConfig:
    """Configuration for ``ClassifyGraph``.

    Parameters
    ----------
    large_graph_cutoff : int, default=10_000_000
        Node-count threshold above which the classifier stays on the reference
        large-graph path.
    """

    large_graph_cutoff: int = 10_000_000


@register_op
class ClassifyGraph(Op):
    """Classify graph structure and write it onto ``problem.structure``.

    Reads
    -----
    ``state.layers``, ``state.back_edge_mask``, and the active edge override in
    ``state.extras["preprocess_edge_index"]`` when present.

    Writes
    ------
    ``problem.structure``.

    Use this when
    -------------
    Later pipeline stages need topology hints such as DAG-ness, layer width, or
    component counts to choose specialized algorithms.
    """

    name = "classify_graph"
    category = OpCategory.PREPROCESS
    reads = ("layers", "back_edge_mask", "extras.preprocess_edge_index")
    writes = ("problem.structure",)

    def __init__(self, config: Optional[ClassifyGraphConfig] = None) -> None:
        """Initialize the graph-classification op.

        Parameters
        ----------
        config : ClassifyGraphConfig | None, optional
            Optional op configuration.
        """
        self.config = config or ClassifyGraphConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Classify the active graph structure.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            The original state. The side effect is on ``problem.structure``.
        """
        _ = ctx
        problem.structure = _build_problem_structure(
            problem,
            state,
            large_graph_cutoff=self.config.large_graph_cutoff,
        )
        return state


@dataclass(frozen=True)
class BuildAdjacencyConfig:
    """Configuration for ``BuildAdjacency``.

    Parameters
    ----------
    directed : bool, default=False
        Whether the output should preserve edge direction.
    weighted : bool, default=False
        Whether to use ``problem.edge_weights``.
    dedup : str, default="min"
        Duplicate-edge policy: ``"min"``, ``"sum"``, or ``"keep_all"``.
    format : str, default="list"
        Output format: ``"list"``, ``"dense"``, or ``"csr"``.
    keep_multiplicity : bool, default=False
        Preserve duplicate edges as separate entries. This is only supported
        for ``"list"`` and ``"csr"`` outputs.
    weight_transform : str, default="none"
        Optional weight transform. ``"inverse"`` converts similarities into
        costs via ``1 / w``.
    """

    directed: bool = False
    weighted: bool = False
    dedup: str = "min"
    format: str = "list"
    keep_multiplicity: bool = False
    weight_transform: str = "none"


@register_op
class BuildAdjacency(Op):
    """Build a cached adjacency representation from the active edge tensor.

    Reads
    -----
    ``state.extras["preprocess_edge_index"]`` when present.

    Writes
    ------
    ``state.adjacency``, optional ``state.adjacency_weighted``, and adjacency
    metadata in ``state.extras``.

    Use this when
    -------------
    Traversal, distance, or embedding ops need a shared adjacency cache in list,
    dense, or CSR form.
    """

    name = "build_adjacency"
    category = OpCategory.PREPROCESS
    reads = ("extras.preprocess_edge_index",)
    writes = (
        "adjacency",
        "adjacency_weighted",
        "extras.adjacency_format",
        "extras.adjacency_directed",
        "extras.adjacency_weighted",
    )

    def __init__(self, config: Optional[BuildAdjacencyConfig] = None) -> None:
        """Initialize the adjacency builder.

        Parameters
        ----------
        config : BuildAdjacencyConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BuildAdjacencyConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build and store adjacency in the configured representation.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``adjacency`` (and ``adjacency_weighted`` for weighted
            builds) populated.

        Raises
        ------
        ValueError
            If the output format is unsupported.
        """
        _ = ctx
        edge_index = _resolved_edge_index(problem, state)
        num_edges = int(edge_index.shape[1]) if edge_index.ndim == 2 else 0
        weights = _resolve_weights(
            problem=problem,
            num_edges=num_edges,
            weighted=self.config.weighted,
            weight_transform=self.config.weight_transform,
        )
        adjacency_list = _build_list_adjacency(
            edge_index=edge_index,
            num_nodes=problem.num_nodes,
            weights=weights,
            directed=self.config.directed,
            dedup=self.config.dedup,
            keep_multiplicity=self.config.keep_multiplicity or self.config.dedup == "keep_all",
        )

        state.adjacency = None
        state.adjacency_weighted = None
        if self.config.weighted:
            # Keep the weighted list cache even when the public adjacency output
            # is dense or CSR so downstream distance ops can reuse exact weights.
            state.adjacency_weighted = adjacency_list

        if self.config.format == "list":
            state.adjacency = adjacency_list
        elif self.config.format == "dense":
            if self.config.keep_multiplicity or self.config.dedup == "keep_all":
                raise ValueError("Dense adjacency cannot preserve edge multiplicity exactly.")
            state.adjacency = _build_dense_adjacency(adjacency_list, num_nodes=problem.num_nodes)
        elif self.config.format == "csr":
            state.adjacency = _build_csr_adjacency(adjacency_list)
        else:
            raise ValueError(f"Unsupported adjacency format: {self.config.format!r}.")

        state.extras[_ADJ_FORMAT_KEY] = self.config.format
        state.extras[_ADJ_DIRECTED_KEY] = self.config.directed
        state.extras[_ADJ_WEIGHTED_KEY] = self.config.weighted
        return state


def _build_spectral_adjacency(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_weights: Optional[torch.Tensor] = None,
) -> sparse.csr_matrix:
    """Build a directed sparse adjacency matrix used by spectral layout.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.

    Returns
    -------
    scipy.sparse.csr_matrix
        Sparse weighted adjacency matrix with shape ``[N, N]``.

    Raises
    ------
    ValueError
        If ``edge_index`` has an invalid shape or out-of-range endpoints.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")

    if edge_index.numel() == 0:
        return sparse.csr_matrix((num_nodes, num_nodes), dtype=np.float64)

    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    rows = edge_index_cpu[0].numpy()
    cols = edge_index_cpu[1].numpy()
    if (
        np.any(rows < 0)
        or np.any(rows >= num_nodes)
        or np.any(cols < 0)
        or np.any(cols >= num_nodes)
    ):
        raise ValueError("edge_index contains a node index outside [0, num_nodes).")

    if edge_weights is not None:
        data = edge_weights.detach().to(device="cpu").numpy().astype(np.float64)
    else:
        data = np.ones(rows.shape[0], dtype=np.float64)

    return sparse.csr_matrix((data, (rows, cols)), shape=(num_nodes, num_nodes), dtype=np.float64)


def _symmetrize_spectral_adjacency(adjacency: sparse.csr_matrix) -> sparse.csr_matrix:
    """Return symmetric adjacency by mirroring directed entries when needed."""
    difference = adjacency - adjacency.T
    if difference.nnz == 0:
        return adjacency
    return (adjacency + adjacency.T).tocsr()


def _spectral_laplacian(
    adjacency: sparse.csr_matrix,
    normalization: str,
) -> tuple[sparse.csr_matrix, bool]:
    """Build the requested graph Laplacian.

    Parameters
    ----------
    adjacency : scipy.sparse.csr_matrix
        Symmetric adjacency matrix with shape ``[N, N]``.
    normalization : str
        One of ``"symmetric"``, ``"random_walk"``, or ``"unnormalized"``.

    Returns
    -------
    tuple[scipy.sparse.csr_matrix, bool]
        Laplacian matrix and whether it is symmetric.
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


@dataclass(frozen=True)
class SpectralPrepareStateConfig:
    """Configuration for :class:`SpectralPrepareState`.

    Parameters
    ----------
    position_dim : int, default=2
        Output dimensionality for the placeholder position tensor used by
        empty and single-node edge cases.
    """

    position_dim: int = 2


@register_op
class SpectralPrepareState(Op):
    """Build the state required by spectral eigenpair ops.

    Reads
    -----
    No ``SolveState`` fields.

    Writes
    ------
    ``state.pos`` for trivial graphs, ``state.laplacian``, and
    ``state.extras["spectral_is_symmetric"]``.

    Use this when
    -------------
    You want spectral pipelines to separate Laplacian construction from the
    later eigendecomposition step.
    """

    name = "spectral_prepare_state"
    category = OpCategory.PREPROCESS
    reads = ()
    writes = ("pos", "laplacian", "extras.spectral_is_symmetric")

    def __init__(
        self,
        normalization: str,
        config: Optional[SpectralPrepareStateConfig] = None,
        networkx_fidelity: bool = False,
    ) -> None:
        """Store the spectral normalization mode.

        Parameters
        ----------
        normalization : str
            One of ``"symmetric"``, ``"random_walk"``, or ``"unnormalized"``.
        config : SpectralPrepareStateConfig, optional
            Spectral preprocessing settings.
        networkx_fidelity : bool, default=False
            Whether to apply NetworkX-compatible trivial graph handling.
        """
        self.normalization = normalization
        self.config = config or SpectralPrepareStateConfig()
        self.networkx_fidelity = bool(networkx_fidelity)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build a cached Laplacian and mark symmetric eigensolve mode.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused.

        Returns
        -------
        SolveState
            The updated state.
        """
        _ = ctx
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, self.config.position_dim), dtype=torch.float32)
            return state
        if problem.num_nodes == 1:
            state.pos = torch.zeros((1, self.config.position_dim), dtype=torch.float32)
            return state
        if self.networkx_fidelity and problem.num_nodes == 2:
            # NetworkX returns both nodes at the default center before any
            # eigensolve, so preserve that exact degenerate edge case.
            state.pos = torch.zeros((2, self.config.position_dim), dtype=torch.float32)
            return state

        adjacency = _build_spectral_adjacency(
            edge_index=problem.edge_index,
            num_nodes=problem.num_nodes,
            edge_weights=problem.edge_weights,
        )
        laplacian, is_symmetric = _spectral_laplacian(
            adjacency=_symmetrize_spectral_adjacency(adjacency),
            normalization=self.normalization,
        )
        state.laplacian = laplacian
        state.extras[_SYMMETRIC_FLAG_KEY] = is_symmetric
        return state


@dataclass(frozen=True)
class FRPrepareAdjacencyConfig:
    """Configuration for :class:`FRPrepareAdjacency`.

    Parameters
    ----------
    default_force_area : float, default=1.0
        Default unit-square area used by FR force calculations when callers do
        not supply an override later in the pipeline.
    k : float, optional
        Explicit NetworkX-style optimal node spacing. When provided,
        ``force_area`` is set to ``k * k * num_nodes`` so the force op resolves
        the requested spacing through its existing area-based helper.
    """

    default_force_area: float = 1.0
    k: Optional[float] = None


@register_op
class FRPrepareAdjacency(Op):
    """Build a dense FR adjacency matrix and default force metadata.

    Reads
    -----
    No ``SolveState`` fields.

    Writes
    ------
    ``state.dense_adjacency`` and ``state.force_area``.

    Use this when
    -------------
    The Fruchterman-Reingold force op expects a dense adjacency matrix and a
    resolved initial force area.
    """

    name = "fr_prepare_adjacency"
    category = OpCategory.PREPROCESS
    reads: tuple[str, ...] = ()
    writes: tuple[str, ...] = ("dense_adjacency", "force_area")

    def __init__(self, config: Optional[FRPrepareAdjacencyConfig] = None) -> None:
        """Store the FR adjacency-preparation configuration.

        Parameters
        ----------
        config : FRPrepareAdjacencyConfig, optional
            FR preparation settings.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or FRPrepareAdjacencyConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Set FR state fields required for force-directed updates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable state receiving FR cache values.
        ctx : RuntimeContext
            Execution infrastructure. Unused for deterministic prep.

        Returns
        -------
        SolveState
            State with ``dense_adjacency`` and ``force_area`` populated.
        """
        _ = ctx
        if problem.num_nodes == 0:
            state.dense_adjacency = torch.zeros((0, 0), dtype=torch.float64)
        else:
            state.dense_adjacency = _build_fr_adjacency_matrix(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                edge_weights=problem.edge_weights,
            )
        if self.config.k is not None:
            if self.config.k <= 0.0:
                raise ValueError("FRPrepareAdjacency k must be positive when provided.")
            state.force_area = float(self.config.k) * float(self.config.k) * float(
                max(problem.num_nodes, 1)
            )
        else:
            state.force_area = self.config.default_force_area
        return state


@dataclass(frozen=True)
class FA2PrepareStateConfig:
    """Configuration for :class:`FA2PrepareState`.

    Parameters
    ----------
    outbound_attraction_distribution : bool, default=True
        Whether FA2 should divide attraction by the source-node mass and apply
        the matching mean-mass compensation.
    mass_offset : float, default=1.0
        Constant added to degree to produce FA2 mass.
    force_dim : int, default=2
        Dimensionality of the cached ``old_forces`` tensor.
    initial_speed : float, default=1.0
        Initial FA2 speed scalar.
    initial_speed_efficiency : float, default=1.0
        Initial FA2 speed-efficiency scalar.
    dtype : torch.dtype, default=torch.float32
        Floating-point dtype for FA2 mass, weights, and force history.
    """

    outbound_attraction_distribution: bool = True
    mass_offset: float = 1.0
    force_dim: int = 2
    initial_speed: float = 1.0
    initial_speed_efficiency: float = 1.0
    dtype: torch.dtype = torch.float32


@register_op
class FA2PrepareState(Op):
    """Build the cached undirected graph state required by FA2 iterations.

    Reads
    -----
    No ``SolveState`` fields.

    Writes
    ------
    ``state.degree``, ``state.old_forces``, and the FA2 cache entries in
    ``state.extras``.

    Use this when
    -------------
    ForceAtlas2 steps need undirected unique edges, node masses, and speed
    control scalars prepared exactly once up front.
    """

    name = "fa2_prepare_state"
    category = OpCategory.PREPROCESS
    reads = ()
    writes = (
        "degree",
        "old_forces",
        "extras.fa2_undirected_edges",
        "extras.fa2_undirected_weights",
        "extras.fa2_mass",
        "extras.fa2_outbound_att_compensation",
        "extras.fa2_speed",
        "extras.fa2_speed_efficiency",
    )

    def __init__(self, config: Optional[FA2PrepareStateConfig] = None) -> None:
        """Store the FA2 preprocessing configuration.

        Parameters
        ----------
        config : FA2PrepareStateConfig, optional
            FA2 preprocessing settings.

        Returns
        -------
        None
            The operation stores the resolved configuration.
        """
        self.config = config or FA2PrepareStateConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Populate FA2 adjacency, degree, mass, and speed-control caches.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable state receiving FA2 caches.
        ctx : RuntimeContext
            Execution infrastructure. Unused for deterministic preprocessing.

        Returns
        -------
        SolveState
            State with FA2 extras populated.
        """
        _ = ctx

        device = problem.edge_index.device
        if problem.edge_index.numel() == 0:
            undirected_edges = torch.empty((2, 0), dtype=torch.long, device=device)
            undirected_weights = None
        else:
            source = problem.edge_index[0].to(dtype=torch.long)
            target = problem.edge_index[1].to(dtype=torch.long)
            non_self = source != target
            if bool(non_self.any().item()):
                source = source[non_self]
                target = target[non_self]
                # FA2 works on unique undirected pairs here so degree, mass, and
                # attraction compensation stay aligned with the later force step.
                lower = torch.minimum(source, target)
                upper = torch.maximum(source, target)
                pairs = torch.stack([lower, upper], dim=1)
                unique_pairs, inverse = torch.unique(pairs, dim=0, return_inverse=True)
                undirected_edges = unique_pairs.transpose(0, 1).contiguous()
                if problem.edge_weights is None:
                    undirected_weights = None
                else:
                    weights = problem.edge_weights[non_self].to(
                        dtype=self.config.dtype,
                        device=device,
                    )
                    undirected_weights = torch.zeros(
                        unique_pairs.shape[0],
                        dtype=self.config.dtype,
                        device=device,
                    )
                    undirected_weights.scatter_add_(0, inverse, weights)
            else:
                undirected_edges = torch.empty((2, 0), dtype=torch.long, device=device)
                undirected_weights = None

        degree = torch.zeros(problem.num_nodes, dtype=self.config.dtype, device=device)
        if undirected_edges.numel() > 0:
            ones = torch.ones(undirected_edges.shape[1], dtype=self.config.dtype, device=device)
            degree.scatter_add_(0, undirected_edges[0], ones)
            degree.scatter_add_(0, undirected_edges[1], ones)

        mass = degree + self.config.mass_offset
        state.degree = degree
        state.old_forces = torch.zeros(
            (problem.num_nodes, self.config.force_dim),
            dtype=self.config.dtype,
            device=device,
        )
        state.extras["fa2_undirected_edges"] = undirected_edges
        state.extras["fa2_undirected_weights"] = undirected_weights
        state.extras["fa2_mass"] = mass
        state.extras["fa2_outbound_att_compensation"] = (
            float(mass.mean().item()) if self.config.outbound_attraction_distribution else 1.0
        )
        state.extras["fa2_speed"] = self.config.initial_speed
        state.extras["fa2_speed_efficiency"] = self.config.initial_speed_efficiency
        return state


@register_op
class DetectComponents(Op):
    """Detect weakly connected components from the active edge tensor.

    Reads
    -----
    ``state.extras["preprocess_edge_index"]`` when present.

    Writes
    ------
    ``state.component_ids``.

    Use this when
    -------------
    Later ops need stable weak-component labels for disconnected-graph logic or
    batched processing.
    """

    name = "detect_components"
    category = OpCategory.PREPROCESS
    reads = ("extras.preprocess_edge_index",)
    writes = ("component_ids",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Assign one component label to each node.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution context. Unused for this deterministic op.

        Returns
        -------
        SolveState
            State with ``component_ids`` populated.
        """
        _ = ctx
        edge_index = _resolved_edge_index(problem, state)
        state.component_ids = _component_labels(edge_index, problem.num_nodes)
        return state
