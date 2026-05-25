"""Pivot-MDS layout pipeline."""

from __future__ import annotations

from typing import Dict, List, Optional, Tuple, Union

import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.distance import (
    PivotDistanceQueries,
    PivotDistanceQueriesConfig,
    PivotSelection,
    PivotSelectionConfig,
)
from dagua.layout.ops.embed import PivotMDSComputeCoordinates
from dagua.layout.ops.postprocess import PivotMDSFinalizePositions
from dagua.layout.ops.preprocess import BuildAdjacency, BuildAdjacencyConfig
from dagua.layout.ops.state import (  # noqa: E402
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)


def build_pivot_mds_pipeline(
    n_pivots: int = 50,
    weighted: bool = False,
    first_pivot_index: Optional[int] = None,
    first_pivot: str = "random",
    compute_dtype: Union[torch.dtype, str] = torch.float32,
    distance_scale: float = 1.0,
) -> Pipeline:
    """Build a Pivot-MDS pipeline.

    Reference fidelity
    ------------------
    Targets: OGDF PivotMDS / Brandes and Pich (2007), "Eigensolver Methods
        for Progressive Multidimensional Scaling of Large Data".
    Fidelity mode: no single flag; reference callers use ``first_pivot`` /
        ``first_pivot_index``, dtype, and ``distance_scale`` to match OGDF
        variants.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.000
        across 10, 50, 100, and 200 pivot variants.
    Known divergences:
        - Pivot selection and distance preparation are native tensor/Python ops.
        - Path graphs have an OGDF-style special case in the public wrapper.

    Parameters
    ----------
    n_pivots : int, default=50
        Maximum number of pivots to select.
    weighted : bool, default=False
        Whether to treat edges as weighted during adjacency construction.
    first_pivot_index : int | None, default=None
        Optional deterministic first pivot used by reference-compatible
        callers. ``None`` preserves the seeded Pivot-MDS default.
    first_pivot : str, default="random"
        Named first-pivot strategy. Use ``"first_node"`` for OGDF
        compatibility.
    compute_dtype : torch.dtype | str, default=torch.float32
        Internal dtype for pivot distances and centering/SVD.
    distance_scale : float, default=1.0
        Multiplicative scale for graph distances before embedding.

    Returns
    -------
    Pipeline
        Pipeline implementing the Pivot-MDS algorithm. The pipeline produces
        final node coordinates by building adjacency, selecting pivots,
        computing pivot-to-node distances, solving the low-rank MDS embedding,
        and finalizing the layout.

    Raises
    ------
    ValueError
        If ``n_pivots`` is not positive.
    """
    if n_pivots <= 0:
        raise ValueError("n_pivots must be positive.")
    if distance_scale <= 0.0:
        raise ValueError("distance_scale must be positive.")
    resolved_dtype = _resolve_compute_dtype(compute_dtype)

    return Pipeline(
        [
            BuildAdjacency(
                BuildAdjacencyConfig(
                    weighted=weighted,
                    dedup="min",
                    format="list",
                ),
            ),
            PivotSelection(
                PivotSelectionConfig(
                    n_pivots=n_pivots,
                    first_pivot_index=first_pivot_index,
                    first_pivot=first_pivot,
                )
            ),
            PivotDistanceQueries(
                PivotDistanceQueriesConfig(
                    dtype=resolved_dtype,
                    distance_scale=distance_scale,
                ),
            ),
            PivotMDSComputeCoordinates(compute_dtype=resolved_dtype),
            PivotMDSFinalizePositions(),
        ],
        name="pivot_mds_pipeline",
    )


def _resolve_compute_dtype(compute_dtype: Union[torch.dtype, str]) -> torch.dtype:
    """Resolve a user-facing dtype token to a torch dtype.

    Parameters
    ----------
    compute_dtype : torch.dtype | str
        Requested internal compute dtype.

    Returns
    -------
    torch.dtype
        ``torch.float32`` or ``torch.float64``.

    Raises
    ------
    ValueError
        If ``compute_dtype`` is unsupported.
    """
    if compute_dtype in (torch.float32, "float32"):
        return torch.float32
    if compute_dtype in (torch.float64, "float64"):
        return torch.float64
    raise ValueError("compute_dtype must be torch.float32, torch.float64, 'float32', or 'float64'.")


def _ogdf_path_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    edge_cost: float,
) -> Optional[torch.Tensor]:
    """Return OGDF-style path coordinates when the graph is a simple path.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.
    edge_cost : float
        Horizontal distance between adjacent path nodes.

    Returns
    -------
    torch.Tensor | None
        Raw OGDF path coordinates with shape ``[N, 2]`` when the simplified
        undirected graph is a path, otherwise ``None``.
    """
    if num_nodes <= 1:
        return torch.zeros((num_nodes, 2), dtype=torch.float32)

    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    undirected_edges: set[Tuple[int, int]] = set()
    for edge_pos in range(int(edges.shape[1])):
        source = int(edges[0, edge_pos].item())
        target = int(edges[1, edge_pos].item())
        if source == target:
            continue
        low, high = (source, target) if source < target else (target, source)
        undirected_edges.add((low, high))

    if len(undirected_edges) != num_nodes - 1:
        return None

    neighbors: Dict[int, List[int]] = {node: [] for node in range(num_nodes)}
    for source, target in undirected_edges:
        neighbors[source].append(target)
        neighbors[target].append(source)
    endpoints = [node for node, node_neighbors in neighbors.items() if len(node_neighbors) == 1]
    if len(endpoints) != 2 or any(len(node_neighbors) > 2 for node_neighbors in neighbors.values()):
        return None

    order: List[int] = []
    previous = -1
    current = endpoints[0]
    while current != -1:
        order.append(current)
        next_nodes = [node for node in neighbors[current] if node != previous]
        previous, current = current, next_nodes[0] if next_nodes else -1
    if len(order) != num_nodes:
        return None

    pos = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for path_index, node in enumerate(order):
        pos[node, 0] = float(path_index) * float(edge_cost)
    return pos


def layout_pivot_mds_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    n_pivots: int = 50,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    first_pivot_index: Optional[int] = None,
    first_pivot: str = "random",
    compute_dtype: Union[torch.dtype, str] = torch.float32,
    distance_scale: float = 1.0,
    ogdf_path_special_case: bool = False,
) -> torch.Tensor:
    """Run the Pivot-MDS pipeline as a drop-in replacement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used to scale the
        final drawing extent.
    n_pivots : int, default=50
        Maximum number of pivots to select.
    seed : int, default=42
        Random seed for the first pivot.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    first_pivot_index : int | None, default=None
        Optional deterministic first pivot used by reference-compatible
        callers. ``None`` preserves the seeded Pivot-MDS default.
    first_pivot : str, default="random"
        Named first-pivot strategy. Use ``"first_node"`` for OGDF
        compatibility.
    compute_dtype : torch.dtype | str, default=torch.float32
        Internal dtype for pivot distances and centering/SVD.
    distance_scale : float, default=1.0
        Multiplicative scale for graph distances before embedding.
    ogdf_path_special_case : bool, default=False
        Whether to return OGDF's raw straight-line layout for simple paths.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes``, ``n_pivots``, or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline fails to populate final positions.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if n_pivots <= 0:
        raise ValueError("n_pivots must be positive.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )
    if distance_scale <= 0.0:
        raise ValueError("distance_scale must be positive.")

    resolved_dtype = _resolve_compute_dtype(compute_dtype)
    if ogdf_path_special_case:
        path_pos = _ogdf_path_positions(edge_index, num_nodes, edge_cost=distance_scale)
        if path_pos is not None:
            return path_pos

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=edge_weights,
        seed=seed,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_pivot_mds_pipeline(
        n_pivots=n_pivots,
        weighted=problem.edge_weights is not None,
        first_pivot_index=first_pivot_index,
        first_pivot=first_pivot,
        compute_dtype=resolved_dtype,
        distance_scale=distance_scale,
    ).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("Pivot-MDS pipeline did not produce final positions.")
    return final_state.pos


__all__ = ["build_pivot_mds_pipeline", "layout_pivot_mds_pipeline"]
