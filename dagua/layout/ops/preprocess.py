"""Preprocessing operations for composable layout pipelines.

These ops prepare graph structure before layout proper: cycle detection,
acyclic rewrites, topology classification, adjacency construction, and
connected-component detection.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

import torch

from dagua.layout.cycle import detect_back_edges, make_acyclic, make_acyclic_robust
from dagua.layout.graph_classify import classify_graph as _classify_graph_reference
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import GraphStructure, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_ACYCLIC_EDGE_KEY = "preprocess_edge_index"
_ADJ_FORMAT_KEY = "adjacency_format"
_ADJ_DIRECTED_KEY = "adjacency_directed"
_ADJ_WEIGHTED_KEY = "adjacency_weighted"

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

    Notes
    -----
    This op is deterministic and uses no randomness.
    """

    name = "detect_cycles"
    category = OpCategory.PREPROCESS
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
            _, reversed_mask = make_acyclic_robust(edge_index, problem.num_nodes)
            state.back_edge_mask = reversed_mask.to(dtype=torch.bool)
            return state
        raise ValueError(f"Unsupported cycle-detection method: {self.config.method!r}.")


@register_op
class MakeAcyclic(Op):
    """Materialize an acyclic edge view into ``state.extras``.

    Notes
    -----
    This op does not mutate ``problem.edge_index``. The rewritten edge tensor is
    stored in ``state.extras["preprocess_edge_index"]`` so downstream ops can
    opt into the acyclic view while preserving the original graph topology.
    """

    name = "make_acyclic"
    category = OpCategory.PREPROCESS
    reads = ("back_edge_mask",)
    writes = ("extras.preprocess_edge_index",)
    requires = ("back_edge_mask",)

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
    """Classify graph structure and write it onto ``problem.structure``."""

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
    """Build a cached adjacency representation from the active edge tensor."""

    name = "build_adjacency"
    category = OpCategory.PREPROCESS
    reads = ("extras.preprocess_edge_index",)
    writes = ("adjacency", "extras.adjacency_format", "extras.adjacency_directed")

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
            State with ``adjacency`` populated.

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


@register_op
class DetectComponents(Op):
    """Detect weakly connected components from the active edge tensor."""

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
