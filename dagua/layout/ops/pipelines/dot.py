"""Graphviz dot-compatible public layout pipeline."""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple, Union

import torch

from dagua.layout.ops.pipelines.sugiyama import layout_sugiyama_pipeline


def layout_dot_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    rank_sep: Optional[float] = None,
    node_sep: Optional[float] = None,
    layer_sep: Optional[float] = None,
    seed: int = 42,
    barycenter_passes: int = 24,
    trace_every: int = 0,
    edge_weights: Optional[torch.Tensor] = None,
    return_edge_routes: bool = False,
    fidelity_dtype: torch.dtype = torch.float32,
    use_node_sizes_for_spacing: Optional[bool] = None,
    center_coordinates: Optional[bool] = None,
    graphviz_node_sizes: Optional[torch.Tensor] = None,
    graphviz_typed_node_sizes: Optional[torch.Tensor] = None,
    graphviz_edge_label_sizes: Optional[torch.Tensor] = None,
    clusters: Optional[Dict[str, Any]] = None,
    cluster_parents: Optional[Dict[str, Optional[str]]] = None,
    graphviz_cluster_label_widths: Optional[Dict[str, float]] = None,
    graphviz_apply_cluster_constraints: bool = False,
    graphviz_enable_cluster_skeleton: bool = False,
    graphviz_expected_x_inventory: Optional[
        Union[
            Tuple[int, Tuple[Tuple[int, int, int], ...]],
            Tuple[int, Tuple[Tuple[int, int, int], ...], str],
            Tuple[int, Tuple[Tuple[int, int, int], ...], str, float],
        ]
    ] = None,
    config: Optional[Any] = None,
) -> Union[
    torch.Tensor,
    Tuple[torch.Tensor, List[torch.Tensor]],
    Tuple[torch.Tensor, List[torch.Tensor], List[torch.Tensor]],
]:
    """Run the existing Graphviz DOT fidelity path as a first-class algorithm.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    rank_sep : float, optional
        Vertical center-to-center layer spacing.
    node_sep : float, optional
        Horizontal gap between node bounding boxes.
    layer_sep : float, optional
        Alias for ``rank_sep``.
    seed : int, default=42
        Seed forwarded for deterministic tie handling.
    barycenter_passes : int, default=24
        Number of crossing-minimization sweeps.
    trace_every : int, default=0
        Trace interval for ordering snapshots.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    return_edge_routes : bool, default=False
        Whether to return reconstructed edge routes with positions.
    fidelity_dtype : torch.dtype, default=torch.float32
        Internal fidelity dtype forwarded for dispatcher compatibility.
    use_node_sizes_for_spacing : bool, optional
        Whether horizontal compaction includes node widths.
    center_coordinates : bool, optional
        Whether to center final horizontal coordinates.
    graphviz_node_sizes : torch.Tensor, optional
        Point-unit Graphviz DOT node boxes with shape ``[N, 2]``.
    graphviz_typed_node_sizes : torch.Tensor, optional
        Source-exact point-unit node boxes for the typed cluster path.
    graphviz_edge_label_sizes : torch.Tensor, optional
        Point-unit edge-label boxes with shape ``[E, 2]``.
    clusters : dict[str, Any], optional
        Cluster membership metadata.
    cluster_parents : dict[str, str | None], optional
        Cluster parent metadata.
    graphviz_cluster_label_widths : dict[str, float], optional
        Padded Graphviz cluster-label widths in point units.
    graphviz_apply_cluster_constraints : bool, default=False
        Whether to enable Graphviz-dot cluster x-boundary machinery.
    graphviz_enable_cluster_skeleton : bool, default=False
        Whether to enable the inactive cluster rank/mincross prototype.
    graphviz_expected_x_inventory : tuple, optional
        Instrumented inventory guard for the typed cluster solve.
    config : Any, optional
        Full layout configuration supplied by the engine.

    Returns
    -------
    torch.Tensor or tuple
        Final positions with shape ``[N, 2]`` and optional traces/routes.
    """
    return layout_sugiyama_pipeline(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        rank_sep=rank_sep,
        node_sep=node_sep,
        layer_sep=layer_sep,
        seed=seed,
        barycenter_passes=barycenter_passes,
        trace_every=trace_every,
        edge_weights=edge_weights,
        return_edge_routes=return_edge_routes,
        fidelity_mode="graphviz",
        fidelity_dtype=fidelity_dtype,
        use_node_sizes_for_spacing=use_node_sizes_for_spacing,
        center_coordinates=center_coordinates,
        graphviz_node_sizes=graphviz_node_sizes,
        graphviz_typed_node_sizes=graphviz_typed_node_sizes,
        graphviz_edge_label_sizes=graphviz_edge_label_sizes,
        clusters=clusters,
        cluster_parents=cluster_parents,
        graphviz_cluster_label_widths=graphviz_cluster_label_widths,
        graphviz_apply_cluster_constraints=graphviz_apply_cluster_constraints,
        graphviz_enable_cluster_skeleton=graphviz_enable_cluster_skeleton,
        graphviz_expected_x_inventory=graphviz_expected_x_inventory,
        config=config,
    )


__all__ = ["layout_dot_pipeline"]
