"""SFDP multilevel force-directed layout pipeline."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import ClassVar, Optional, Tuple

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.graph_utils import (
    layout_device as _layout_device,
)
from dagua.layout.ops.sfdp import (
    _BASE_GRAPH_KEY,
    _GENERATOR_KEY,
    _GRAPH_KEY,
    _MAPPING_KEY,
    BuildSFDPGraph,
    BuildSFDPHierarchy,
    GraphData,
    InitSFDPCoarsestPositions,
    SFDPFinalizePositions,
    SFDPHierarchyConfig,
    SFDPProlongateAndRefineLevels,
    SFDPRefineCoarsestLevel,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)
from dagua.layout.ops.taxonomy import OpCategory

_DEFAULT_THETA = 0.6
_DEFAULT_P = -1.0
_GRAPHVIZ_MAX_CLUSTER_SIZE = 4


def _decompose_graphviz_supervariables(graph: GraphData) -> list[list[int]]:
    """Group nodes with Graphviz's sparse-matrix supervariable refinement.

    Parameters
    ----------
    graph : GraphData
        Symmetric SFDP graph whose adjacency lists represent sparse matrix rows.

    Returns
    -------
    list[list[int]]
        Supervariable groups in Graphviz creation order. Each inner list contains
        fine node IDs that have the same sparse-matrix column pattern.
    """
    num_nodes = graph.num_nodes
    if num_nodes == 0:
        return []

    super_ids = [0 for _ in range(num_nodes)]
    super_sizes = [0 for _ in range(num_nodes + 1)]
    mask = [-1 for _ in range(num_nodes)]
    newmap = [0 for _ in range(num_nodes)]
    super_sizes[0] = num_nodes
    next_super_id = 1

    for row_index, neighbors in enumerate(graph.adjacency):
        neighbor_ids = [neighbor for neighbor, _ in neighbors]
        for neighbor in neighbor_ids:
            super_sizes[super_ids[neighbor]] -= 1

        for neighbor in neighbor_ids:
            old_super = super_ids[neighbor]
            if mask[old_super] < row_index:
                mask[old_super] = row_index
                if super_sizes[old_super] == 0:
                    super_sizes[old_super] = 1
                    newmap[old_super] = old_super
                else:
                    new_super = next_super_id
                    next_super_id += 1
                    newmap[old_super] = new_super
                    super_sizes[new_super] = 1
                    super_ids[neighbor] = new_super
            else:
                mapped_super = newmap[old_super]
                super_ids[neighbor] = mapped_super
                super_sizes[mapped_super] += 1

    groups: list[list[int]] = [[] for _ in range(next_super_id)]
    for node, super_id in enumerate(super_ids):
        groups[super_id].append(node)
    return groups


def _graphviz_sfdp_cluster_nodes(
    graph: GraphData,
    generator: torch.Generator,
    config: SFDPHierarchyConfig,
) -> Optional[torch.Tensor]:
    """Build Graphviz SFDP fine-to-coarse clusters for one internal pass.

    Parameters
    ----------
    graph : GraphData
        Fine graph for the current internal coarsening pass.
    generator : torch.Generator
        CPU generator used for the unmatched-node permutation.
    config : SFDPHierarchyConfig
        Graphviz-compatible coarsening stop thresholds.

    Returns
    -------
    torch.Tensor | None
        Fine-to-coarse assignment with shape ``[N]`` when this pass produces a
        valid Graphviz internal coarsening, otherwise ``None``.
    """
    num_nodes = graph.num_nodes
    if num_nodes < config.min_coarse_size:
        return None

    matched = [False for _ in range(num_nodes)]
    clusters: list[list[int]] = []

    for super_group in _decompose_graphviz_supervariables(graph):
        if len(super_group) <= 1:
            continue
        for start in range(0, len(super_group), _GRAPHVIZ_MAX_CLUSTER_SIZE):
            chunk = super_group[start : start + _GRAPHVIZ_MAX_CLUSTER_SIZE]
            for node in chunk:
                matched[node] = True
            clusters.append(chunk)

    for node in torch.randperm(num_nodes, generator=generator).tolist():
        if matched[node]:
            continue

        partner = -1
        partner_weight = -1.0
        for neighbor, weight in graph.adjacency[node]:
            if matched[neighbor]:
                continue
            if weight > partner_weight:
                partner = neighbor
                partner_weight = weight

        if partner >= 0:
            matched[node] = True
            matched[partner] = True
            clusters.append([node, partner])

    for node, is_matched in enumerate(matched):
        if not is_matched:
            clusters.append([node])

    coarse_num_nodes = len(clusters)
    if coarse_num_nodes == num_nodes or coarse_num_nodes < config.min_coarse_size:
        return None

    fine_to_coarse = torch.empty((num_nodes,), dtype=torch.long)
    for coarse_node, cluster in enumerate(clusters):
        for fine_node in cluster:
            fine_to_coarse[fine_node] = coarse_node
    return fine_to_coarse


def _build_graphviz_matrix_coarse_graph(
    graph: GraphData,
    fine_to_coarse: torch.Tensor,
    coarse_num_nodes: int,
) -> GraphData:
    """Aggregate a coarse graph through Graphviz's ``R * A * P`` semantics.

    Parameters
    ----------
    graph : GraphData
        Fine graph represented as unique undirected weighted edges.
    fine_to_coarse : torch.Tensor
        Fine-to-coarse assignment with shape ``[N_fine]``.
    coarse_num_nodes : int
        Number of coarse clusters.

    Returns
    -------
    GraphData
        Coarse graph with diagonal entries removed after matrix aggregation.
    """
    coarse_edges: dict[tuple[int, int], float] = {}
    for edge_id in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge_id].item())
        target = int(graph.edge_index[1, edge_id].item())
        coarse_source = int(fine_to_coarse[source].item())
        coarse_target = int(fine_to_coarse[target].item())
        if coarse_source == coarse_target:
            continue
        lower = min(coarse_source, coarse_target)
        upper = max(coarse_source, coarse_target)
        coarse_edges[(lower, upper)] = coarse_edges.get((lower, upper), 0.0) + float(
            graph.edge_weight[edge_id].item()
        )

    if coarse_edges:
        ordered_edges = sorted(coarse_edges.items())
        edge_pairs = [edge for edge, _ in ordered_edges]
        edge_weights = [weight for _, weight in ordered_edges]
        edge_index = torch.tensor(edge_pairs, dtype=torch.long).transpose(0, 1).contiguous()
        weight_tensor = torch.tensor(edge_weights, dtype=torch.float32)
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)
        weight_tensor = torch.empty((0,), dtype=torch.float32)

    adjacency: list[list[tuple[int, float]]] = [[] for _ in range(coarse_num_nodes)]
    for edge_id in range(edge_index.shape[1]):
        source = int(edge_index[0, edge_id].item())
        target = int(edge_index[1, edge_id].item())
        weight = float(weight_tensor[edge_id].item())
        adjacency[source].append((target, weight))
        adjacency[target].append((source, weight))

    return GraphData(
        num_nodes=coarse_num_nodes,
        edge_index=edge_index,
        edge_weight=weight_tensor,
        adjacency=adjacency,
    )


def _graphviz_sfdp_coarsen(
    graph: GraphData,
    generator: torch.Generator,
    config: SFDPHierarchyConfig,
) -> Optional[tuple[torch.Tensor, GraphData]]:
    """Coarsen one Graphviz SFDP multilevel edge using matrix aggregation.

    Parameters
    ----------
    graph : GraphData
        Fine graph at the current hierarchy level.
    generator : torch.Generator
        CPU generator used by unmatched-node heavy-edge passes.
    config : SFDPHierarchyConfig
        Coarsening thresholds. ``min_coarsen_reduction`` controls the wrapper
        loop that forces sufficient reduction across internal passes.

    Returns
    -------
    tuple[torch.Tensor, GraphData] | None
        Composed fine-to-coarse mapping and coarse graph, or ``None`` if the
        Graphviz internal pass cannot produce a usable coarse graph.
    """
    original_num_nodes = graph.num_nodes
    current_graph = graph
    composed_mapping: Optional[torch.Tensor] = None
    last_result: Optional[tuple[torch.Tensor, GraphData]] = None

    while True:
        internal_mapping = _graphviz_sfdp_cluster_nodes(
            graph=current_graph,
            generator=generator,
            config=config,
        )
        if internal_mapping is None:
            return last_result

        coarse_num_nodes = int(internal_mapping.max().item()) + 1
        coarse_graph = _build_graphviz_matrix_coarse_graph(
            graph=current_graph,
            fine_to_coarse=internal_mapping,
            coarse_num_nodes=coarse_num_nodes,
        )
        composed_mapping = (
            internal_mapping
            if composed_mapping is None
            else internal_mapping[composed_mapping].to(dtype=torch.long)
        )
        last_result = (composed_mapping.clone(), coarse_graph)

        if coarse_num_nodes <= config.min_coarsen_reduction * float(original_num_nodes):
            return last_result

        current_graph = coarse_graph


@dataclass(frozen=True)
class BuildGraphvizSFDPMatrixHierarchy(Op):
    """Build Graphviz-compatible SFDP hierarchy using matrix coarsening."""

    config: SFDPHierarchyConfig = field(default_factory=SFDPHierarchyConfig)

    name: ClassVar[str] = "sfdp_graphviz_matrix_coarsen_hierarchy"
    category: ClassVar[OpCategory] = OpCategory.COARSEN
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    requires: ClassVar[Tuple[str, ...]] = (f"extras.{_BASE_GRAPH_KEY}",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Iteratively build the fidelity-mode Graphviz SFDP hierarchy.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs containing the deterministic seed.
        state : SolveState
            Mutable solve state. Reads the base SFDP graph from ``extras``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State with SFDP graph levels, mappings, and generator populated.
        """
        del ctx
        generator = torch.Generator(device="cpu")
        generator.manual_seed(problem.seed)

        base_graph: GraphData = state.extras[_BASE_GRAPH_KEY]
        graphs: list[GraphData] = [base_graph]
        mappings: list[torch.Tensor] = []
        current_graph = base_graph

        while True:
            coarsened = _graphviz_sfdp_coarsen(
                graph=current_graph,
                generator=generator,
                config=self.config,
            )
            if coarsened is None:
                break

            fine_to_coarse, coarse_graph = coarsened
            mappings.append(fine_to_coarse)
            graphs.append(coarse_graph)
            current_graph = coarse_graph

        state.extras[_GRAPH_KEY] = graphs
        state.extras[_MAPPING_KEY] = mappings
        state.extras[_GENERATOR_KEY] = generator
        return state


def build_sfdp_pipeline(
    steps: int = 500,
    theta: float = _DEFAULT_THETA,
    repulsive_exponent: float = _DEFAULT_P,
    fidelity_mode: bool = False,
) -> Pipeline:
    """Build an SFDP multilevel force-directed pipeline.

    Reference fidelity
    ------------------
    Targets: Graphviz 7.0.5 sfdp / Hu (2005), "Efficient, High-Quality
        Force-Directed Graph Drawing".
    Fidelity mode: ``fidelity_mode=True`` switches the hierarchy builder to
        Graphviz SFDP's supervariable-first matrix coarsening path.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.079
        to 0.100. Round 33 force-law alignment improved the bounded subset to
        median RMSD 0.004724.
    Known divergences:
        - Remaining residual is dominated by ``parallel_multiedge_bundle``.
        - Sequential in-place updates remain unported.
        - Unmatched-node permutation still uses Dagua's seeded torch generator
          rather than Graphviz's process-global ``gv_random`` stream.

    Parameters
    ----------
    steps : int, default=500
        Maximum number of spring-electrical iterations per level.
    theta : float, default=0.6
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=-1.0
        SFDP repulsion exponent ``p``.
    fidelity_mode : bool, default=False
        When ``True``, build the coarsening hierarchy with Graphviz SFDP's
        matrix-based supervariable clustering instead of Dagua's historical
        heavy-edge matching path.

    Returns
    -------
    Pipeline
        Pipeline implementing the classical SFDP algorithm. The pipeline
        produces final node coordinates by building the multilevel graph,
        initializing the coarsest level, refining with spring-electrical
        updates, prolongating through finer levels, and normalizing the final
        positions.

    Raises
    ------
    ValueError
        If ``steps`` is negative.
    """
    if steps < 0:
        raise ValueError("steps must be non-negative.")

    hierarchy_op: Op
    if fidelity_mode:
        hierarchy_op = BuildGraphvizSFDPMatrixHierarchy()
    else:
        hierarchy_op = BuildSFDPHierarchy()

    return Pipeline(
        [
            BuildSFDPGraph(),
            hierarchy_op,
            InitSFDPCoarsestPositions(),
            SFDPRefineCoarsestLevel(
                steps=steps,
                theta=theta,
                repulsive_exponent=repulsive_exponent,
            ),
            SFDPProlongateAndRefineLevels(
                steps=steps,
                theta=theta,
                repulsive_exponent=repulsive_exponent,
            ),
            SFDPFinalizePositions(),
        ],
        name="sfdp_pipeline",
    )


def layout_sfdp_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    steps: int = 500,
    seed: int = 123,
    theta: float = _DEFAULT_THETA,
    repulsive_exponent: float = _DEFAULT_P,
    edge_weights: Optional[torch.Tensor] = None,
    direction: str = "TB",
    fidelity_mode: bool = False,
) -> torch.Tensor:
    """Run the SFDP pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only for output
        scaling.
    steps : int, default=500
        Maximum number of spring-electrical iterations per level.
    seed : int, default=123
        Random seed for coarsening order, coarsest initialization, and
        prolongation noise.
    theta : float, default=0.6
        Barnes-Hut opening angle threshold.
    repulsive_exponent : float, default=-1.0
        SFDP repulsion exponent ``p``.
    edge_weights : torch.Tensor, optional
        Optional edge weights with shape ``[E]``.
    direction : str, default="TB"
        Requested layout flow direction: ``TB``, ``BT``, ``LR``, or ``RL``.
    fidelity_mode : bool, default=False
        Enable Graphviz-compatible matrix coarsening. Existing behavior remains
        unchanged when this flag is ``False``.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If inputs are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if steps < 0:
        raise ValueError("steps must be non-negative.")
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    if edge_index.dtype not in {
        torch.int8,
        torch.int16,
        torch.int32,
        torch.int64,
        torch.uint8,
    }:
        raise ValueError("edge_index must use an integer dtype.")
    if edge_weights is not None and edge_weights.shape[0] != edge_index.shape[1]:
        raise ValueError(
            f"edge_weights length {edge_weights.shape[0]} does not match "
            f"edge count {edge_index.shape[1]}"
        )
    if edge_index.numel() != 0:
        edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
        if int(edge_index_cpu.min().item()) < 0:
            raise ValueError("edge_index cannot contain negative node indices.")
        if int(edge_index_cpu.max().item()) >= num_nodes:
            raise ValueError("edge_index contains node indices outside num_nodes.")

    device = _layout_device(edge_index=edge_index, node_sizes=node_sizes)
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32, device=device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=torch.float32, device=device)

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
        direction=direction,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_sfdp_pipeline(
        steps=steps,
        theta=theta,
        repulsive_exponent=repulsive_exponent,
        fidelity_mode=fidelity_mode,
    ).apply(problem, state, ctx)

    if final_state.pos is None:
        raise RuntimeError("SFDP pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_sfdp_pipeline", "layout_sfdp_pipeline"]
