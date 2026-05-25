"""Graphviz neato-compatible layout pipeline."""

from __future__ import annotations

from typing import List, Optional, Tuple, Union

import torch

from dagua.layout.ops.pipelines.stress_majorization import layout_stress_majorization_pipeline


def _weak_components(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Compute weak components in deterministic node order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Weakly connected components, each expressed as sorted node indices.
    """
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    for source, target in zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist()):
        if source == target:
            continue
        neighbors[source].append(target)
        neighbors[target].append(source)

    seen = [False] * num_nodes
    components: List[List[int]] = []
    for start in range(num_nodes):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        component: List[int] = []
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in neighbors[node]:
                if not seen[neighbor]:
                    seen[neighbor] = True
                    stack.append(neighbor)
        components.append(sorted(component))
    return components


def _slice_component_edges(
    edge_index: torch.Tensor,
    edge_weights: Optional[torch.Tensor],
    component: List[int],
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Return component-local edges and aligned optional weights.

    Parameters
    ----------
    edge_index : torch.Tensor
        Parent edge tensor with shape ``[2, E]``.
    edge_weights : torch.Tensor, optional
        Optional parent edge weights with shape ``[E]``.
    component : list[int]
        Parent node indices in one weak component.

    Returns
    -------
        tuple[torch.Tensor, torch.Tensor or None]
        Component-local edge tensor and matching weights.
    """
    node_to_local = {node: index for index, node in enumerate(component)}
    sources: List[int] = []
    targets: List[int] = []
    weight_values: List[float] = []
    edge_index_cpu = edge_index.to(device="cpu", dtype=torch.long)
    weights_cpu = None if edge_weights is None else edge_weights.detach().to(device="cpu")
    for edge_id, (source, target) in enumerate(
        zip(edge_index_cpu[0].tolist(), edge_index_cpu[1].tolist())
    ):
        if source not in node_to_local or target not in node_to_local:
            continue
        sources.append(node_to_local[source])
        targets.append(node_to_local[target])
        if weights_cpu is not None:
            weight_values.append(float(weights_cpu[edge_id].item()))
    local_edges = torch.tensor([sources, targets], dtype=torch.long, device=edge_index.device)
    if weights_cpu is None:
        return local_edges, None
    local_weights = torch.tensor(
        weight_values,
        dtype=edge_weights.dtype,
        device=edge_weights.device,
    )
    return local_edges, local_weights


def _pack_component_positions(
    components: List[List[int]],
    component_positions: List[torch.Tensor],
    num_nodes: int,
    gap: float,
) -> torch.Tensor:
    """Pack component layouts into a row-major grid.

    Parameters
    ----------
    components : list[list[int]]
        Parent node indices for each component.
    component_positions : list[torch.Tensor]
        Local component coordinates, each with shape ``[C, 2]``.
    num_nodes : int
        Total number of parent graph nodes.
    gap : float
        Padding between component bounding boxes.

    Returns
    -------
    torch.Tensor
        Packed parent coordinates with shape ``[N, 2]``.
    """
    if not component_positions:
        return torch.empty((0, 2), dtype=torch.float32)
    device = component_positions[0].device
    dtype = component_positions[0].dtype
    packed = torch.zeros((num_nodes, 2), dtype=dtype, device=device)
    cols = max(1, int(len(component_positions) ** 0.5 + 0.999))
    x_cursor = 0.0
    y_cursor = 0.0
    row_height = 0.0
    for index, (component, local_pos) in enumerate(zip(components, component_positions)):
        local = local_pos - local_pos.mean(dim=0, keepdim=True)
        mins = local.min(dim=0).values
        maxs = local.max(dim=0).values
        size = (maxs - mins).clamp(min=1.0)
        local = local - mins + torch.tensor([x_cursor, y_cursor], dtype=dtype, device=device)
        packed[component] = local
        x_cursor += float(size[0].item()) + gap
        row_height = max(row_height, float(size[1].item()))
        if (index + 1) % cols == 0:
            x_cursor = 0.0
            y_cursor += row_height + gap
            row_height = 0.0
    return packed - packed.mean(dim=0, keepdim=True)


def layout_neato_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    maxiter: int = 200,
    iterations: Optional[int] = None,
    epsilon: float = 0.0001,
    mode: str = "major",
    model: str = "shortpath",
    pack: bool = True,
    trace_every: int = 0,
) -> Union[torch.Tensor, Tuple[torch.Tensor, List[torch.Tensor]]]:
    """Run Graphviz neato's default stress-majorization mode.

    Reference fidelity
    ------------------
    Targets: Graphviz 7.0.5 neato / Gansner, Koren, and North (2004), "Graph
        Drawing by Stress Majorization".
    Fidelity mode: this public entry point always calls stress majorization
        with ``fidelity_mode="graphviz_neato"`` for supported options.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.065.
        Round 33 bounded neato subset median remained 0.009117.
    Known divergences:
        - Only ``mode="major"`` and ``model="shortpath"`` are supported.
        - Exact CG solver behavior, raw ``drand48`` initialization parity, edge
          ``len`` semantics, and Graphviz post-processing remain unported.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]``.
    seed : int, default=42
        Random seed for the neato-style random initialization.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    maxiter : int, default=200
        Maximum SMACOF iteration count.
    iterations : int, optional
        Alias for ``maxiter`` used by existing dagua stress variants.
    epsilon : float, default=0.0001
        Relative stress-delta convergence threshold.
    mode : str, default="major"
        Graphviz neato mode. Only ``"major"`` is supported in this adapter.
    model : str, default="shortpath"
        Graphviz neato model. Only ``"shortpath"`` is supported.
    pack : bool, default=True
        Whether to pack disconnected components independently.
    trace_every : int, default=0
        If positive, return periodic position snapshots.

    Returns
    -------
    torch.Tensor or tuple[torch.Tensor, list[torch.Tensor]]
        Final position tensor with shape ``[N, 2]``. Traces are returned when
        ``trace_every > 0`` and component packing is not required.
    """
    if mode != "major":
        raise ValueError("neato pipeline currently supports only mode='major'.")
    if model != "shortpath":
        raise ValueError("neato pipeline currently supports only model='shortpath'.")
    resolved_iterations = maxiter if iterations is None else iterations
    if not pack or num_nodes <= 1:
        return layout_stress_majorization_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            iterations=resolved_iterations,
            seed=seed,
            edge_weights=edge_weights,
            trace_every=trace_every,
            fidelity_mode="graphviz_neato",
            epsilon=epsilon,
        )

    components = _weak_components(edge_index=edge_index, num_nodes=num_nodes)
    if len(components) <= 1:
        return layout_stress_majorization_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            iterations=resolved_iterations,
            seed=seed,
            edge_weights=edge_weights,
            trace_every=trace_every,
            fidelity_mode="graphviz_neato",
            epsilon=epsilon,
        )

    component_positions: List[torch.Tensor] = []
    for component_index, component in enumerate(components):
        local_edges, local_weights = _slice_component_edges(edge_index, edge_weights, component)
        local_sizes = node_sizes[component] if node_sizes is not None else None
        local_pos = layout_stress_majorization_pipeline(
            edge_index=local_edges,
            num_nodes=len(component),
            node_sizes=local_sizes,
            iterations=resolved_iterations,
            seed=seed + component_index,
            edge_weights=local_weights,
            trace_every=0,
            fidelity_mode="graphviz_neato",
            epsilon=epsilon,
        )
        component_positions.append(local_pos)
    gap = 1.0
    if node_sizes is not None and node_sizes.numel() > 0:
        gap = max(float(node_sizes.to(dtype=torch.float32, device="cpu").max().item()), 1.0)
    packed = _pack_component_positions(
        components=components,
        component_positions=component_positions,
        num_nodes=num_nodes,
        gap=gap,
    )
    return (packed, []) if trace_every > 0 else packed


__all__ = ["layout_neato_pipeline"]
