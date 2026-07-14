"""NetworkX simple deterministic layout operations.

These ops are source ports of the small deterministic layouts in
``networkx.drawing.layout``.  They intentionally do not import NetworkX; the
reference package is used only by competitor adapters and fidelity scripts.
"""

from __future__ import annotations

import math
import warnings
from collections import deque
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np
import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.graph_utils import layout_device
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


def nx_rescale_layout(pos: np.ndarray, scale: float = 1.0) -> np.ndarray:
    """Rescale positions using NetworkX's in-place layout normalization.

    Parameters
    ----------
    pos : numpy.ndarray
        Coordinate array with shape ``[N, 2]``.
    scale : float, default=1.0
        Target maximum absolute coordinate after centering.

    Returns
    -------
    numpy.ndarray
        The input array, centered per axis and scaled in place.
    """
    pos -= pos.mean(axis=0)
    lim = np.abs(pos).max()
    if lim > 0:
        pos *= scale / lim
    return pos


def _empty_positions(num_nodes: int, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Return an empty or singleton NetworkX-compatible coordinate tensor.

    Parameters
    ----------
    num_nodes : int
        Number of nodes in the graph.
    dtype : torch.dtype
        Output floating-point dtype.
    device : torch.device
        Output device.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    return torch.zeros((num_nodes, 2), dtype=dtype, device=device)


def _to_tensor(pos: np.ndarray, dtype: torch.dtype, device: torch.device) -> torch.Tensor:
    """Convert a NumPy coordinate array to the pipeline output tensor.

    Parameters
    ----------
    pos : numpy.ndarray
        Coordinate array with shape ``[N, 2]``.
    dtype : torch.dtype
        Output floating-point dtype.
    device : torch.device
        Output device.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    return torch.as_tensor(pos, dtype=dtype, device=device)


def _sorted_adjacency(edge_index: torch.Tensor, num_nodes: int) -> List[List[int]]:
    """Build NetworkX-order undirected adjacency lists.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Neighbor lists in first-seen edge order with duplicate neighbors
        removed.
    """
    neighbors: List[List[int]] = [[] for _ in range(num_nodes)]
    seen: List[set[int]] = [set() for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return neighbors
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    for source, target in zip(edges[0].tolist(), edges[1].tolist()):
        source = int(source)
        target = int(target)
        if target not in seen[source]:
            neighbors[source].append(target)
            seen[source].add(target)
        if source not in seen[target]:
            neighbors[target].append(source)
            seen[target].add(source)
    return neighbors


def nx_bfs_layers(edge_index: torch.Tensor, num_nodes: int, start: int = 0) -> Dict[int, List[int]]:
    """Compute NetworkX ``bfs_layers``-style layers from a pinned source.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    start : int, default=0
        Source node for the first BFS component.

    Returns
    -------
    dict[int, list[int]]
        Layer index to nodes. Disconnected components are appended in node
        order so headless Dagua layouts still cover every node.
    """
    if num_nodes <= 0:
        return {}
    neighbors = _sorted_adjacency(edge_index=edge_index, num_nodes=num_nodes)
    start = min(max(int(start), 0), num_nodes - 1)
    visited = [False] * num_nodes
    layers: Dict[int, List[int]] = {}
    next_layer = 0
    for root in [start] + [node for node in range(num_nodes) if node != start]:
        if visited[root]:
            continue
        visited[root] = True
        queue: deque[tuple[int, int]] = deque([(root, next_layer)])
        component_max = next_layer
        while queue:
            node, layer = queue.popleft()
            layers.setdefault(layer, []).append(node)
            component_max = max(component_max, layer)
            for neighbor in neighbors[node]:
                if visited[neighbor]:
                    continue
                visited[neighbor] = True
                queue.append((neighbor, layer + 1))
        next_layer = component_max + 1
    return layers


def nx_bipartite_node_set(edge_index: torch.Tensor, num_nodes: int, start: int = 0) -> List[int]:
    """Choose a deterministic bipartite side for headless pipeline input.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    start : int, default=0
        Source node whose even-distance side is used as NetworkX's ``nodes``
        argument.

    Returns
    -------
    list[int]
        Nodes assigned to the left/top side, in node order.
    """
    layers = nx_bfs_layers(edge_index=edge_index, num_nodes=num_nodes, start=start)
    top = [node for layer, nodes in layers.items() if layer % 2 == 0 for node in nodes]
    return sorted(top)


def nx_circular_positions(
    num_nodes: int,
    scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return NetworkX ``circular_layout`` coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    scale : float, default=1.0
        NetworkX layout scale.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    device : torch.device | None, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    out_device = torch.device("cpu") if device is None else device
    if num_nodes <= 1:
        return _empty_positions(num_nodes=num_nodes, dtype=dtype, device=out_device)
    theta = np.linspace(0, 1, num_nodes + 1)[:-1] * 2 * np.pi
    theta = theta.astype(np.float32)
    pos = np.column_stack([np.cos(theta), np.sin(theta), np.zeros((num_nodes, 0))])
    pos = nx_rescale_layout(pos, scale=scale)
    return _to_tensor(pos=pos, dtype=dtype, device=out_device)


def nx_shell_positions(
    num_nodes: int,
    nlist: Optional[List[List[int]]] = None,
    rotate: Optional[float] = None,
    scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return NetworkX ``shell_layout`` coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    nlist : list[list[int]] | None, optional
        Shell membership. ``None`` mirrors NetworkX by placing every node in
        one shell.
    rotate : float | None, optional
        Per-shell starting-angle increment in radians.
    scale : float, default=1.0
        NetworkX layout scale.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    device : torch.device | None, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    out_device = torch.device("cpu") if device is None else device
    if num_nodes <= 1:
        return _empty_positions(num_nodes=num_nodes, dtype=dtype, device=out_device)
    shells = [list(range(num_nodes))] if nlist is None else nlist
    pos = np.zeros((num_nodes, 2), dtype=np.float64)
    radius_bump = float(scale) / len(shells)
    radius = 0.0 if len(shells[0]) == 1 else radius_bump
    resolved_rotate = np.pi / len(shells) if rotate is None else rotate
    first_theta = resolved_rotate
    for nodes in shells:
        theta = (
            np.linspace(0, 2 * np.pi, len(nodes), endpoint=False, dtype=np.float32) + first_theta
        )
        layer_pos = radius * np.column_stack([np.cos(theta), np.sin(theta)])
        for node, coords in zip(nodes, layer_pos):
            pos[node] = coords
        radius += radius_bump
        first_theta += resolved_rotate
    return _to_tensor(pos=pos, dtype=dtype, device=out_device)


def nx_spiral_positions(
    num_nodes: int,
    scale: float = 1.0,
    resolution: float = 0.35,
    equidistant: bool = False,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return NetworkX ``spiral_layout`` coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    scale : float, default=1.0
        NetworkX layout scale.
    resolution : float, default=0.35
        Spiral compactness parameter.
    equidistant : bool, default=False
        Whether to use NetworkX's equidistant chord update path.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    device : torch.device | None, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    out_device = torch.device("cpu") if device is None else device
    if num_nodes <= 1:
        return _empty_positions(num_nodes=num_nodes, dtype=dtype, device=out_device)
    if equidistant:
        chord = 1.0
        step = 0.5
        theta = float(resolution)
        theta += chord / (step * theta)
        rows: List[List[float]] = []
        for _ in range(num_nodes):
            r = step * theta
            theta += chord / r
            rows.append([np.cos(theta) * r, np.sin(theta) * r])
        pos = np.array(rows)
    else:
        dist = np.arange(num_nodes, dtype=float)
        angle = float(resolution) * dist
        pos = np.transpose(dist * np.array([np.cos(angle), np.sin(angle)]))
    pos = nx_rescale_layout(np.array(pos), scale=scale)
    return _to_tensor(pos=pos, dtype=dtype, device=out_device)


def nx_bipartite_positions(
    num_nodes: int,
    nodes: List[int],
    align: str = "vertical",
    scale: float = 1.0,
    aspect_ratio: float = 4.0 / 3.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return NetworkX ``bipartite_layout`` coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    nodes : list[int]
        Pinned NetworkX ``nodes`` argument for the left/top side.
    align : {"vertical", "horizontal"}, default="vertical"
        NetworkX alignment mode.
    scale : float, default=1.0
        NetworkX layout scale.
    aspect_ratio : float, default=4/3
        Width-to-height ratio before final rescale.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    device : torch.device | None, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    if align not in ("vertical", "horizontal"):
        raise ValueError("align must be either vertical or horizontal.")
    out_device = torch.device("cpu") if device is None else device
    if num_nodes <= 0:
        return _empty_positions(num_nodes=num_nodes, dtype=dtype, device=out_device)
    top = set(nodes)
    bottom = set(range(num_nodes)) - top
    ordered_nodes = list(top) + list(bottom)
    height = 1.0
    width = float(aspect_ratio) * height
    offset = (width / 2.0, height / 2.0)
    left_xs = np.repeat(0, len(top))
    right_xs = np.repeat(width, len(bottom))
    left_ys = np.linspace(0, height, len(top))
    right_ys = np.linspace(0, height, len(bottom))
    top_pos = np.column_stack([left_xs, left_ys]) - offset
    bottom_pos = np.column_stack([right_xs, right_ys]) - offset
    raw = np.concatenate([top_pos, bottom_pos])
    raw = nx_rescale_layout(raw, scale=scale)
    if align == "horizontal":
        raw = raw[:, ::-1]
    pos = np.zeros((num_nodes, 2), dtype=raw.dtype)
    for node, coords in zip(ordered_nodes, raw):
        pos[node] = coords
    return _to_tensor(pos=pos, dtype=dtype, device=out_device)


def nx_multipartite_positions(
    layers: Dict[int, List[int]],
    num_nodes: int,
    align: str = "vertical",
    scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return NetworkX ``multipartite_layout`` coordinates from pinned layers.

    Parameters
    ----------
    layers : dict[int, list[int]]
        Layer mapping used as NetworkX's dict-like ``subset_key`` argument.
    num_nodes : int
        Number of graph nodes.
    align : {"vertical", "horizontal"}, default="vertical"
        NetworkX alignment mode.
    scale : float, default=1.0
        NetworkX layout scale.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    device : torch.device | None, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    if align not in ("vertical", "horizontal"):
        raise ValueError("align must be either vertical or horizontal.")
    out_device = torch.device("cpu") if device is None else device
    if num_nodes <= 0:
        return _empty_positions(num_nodes=num_nodes, dtype=dtype, device=out_device)
    sorted_layers = dict(sorted(layers.items()))
    rows: List[np.ndarray] = []
    ordered_nodes: List[int] = []
    width = len(sorted_layers)
    for i, layer in enumerate(sorted_layers.values()):
        height = len(layer)
        xs = np.repeat(i, height)
        ys = np.arange(0, height, dtype=float)
        offset = ((width - 1) / 2.0, (height - 1) / 2.0)
        rows.append(np.column_stack([xs, ys]) - offset)
        ordered_nodes.extend(layer)
    raw = nx_rescale_layout(np.concatenate(rows), scale=scale)
    if align == "horizontal":
        raw = raw[:, ::-1]
    pos = np.zeros((num_nodes, 2), dtype=raw.dtype)
    for node, coords in zip(ordered_nodes, raw):
        pos[node] = coords
    return _to_tensor(pos=pos, dtype=dtype, device=out_device)


def nx_arf_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    scaling: float = 1.0,
    a: float = 1.1,
    etol: float = 1.0e-6,
    dt: float = 1.0e-3,
    max_iter: int = 1000,
    seed: Optional[int] = None,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return NetworkX ``arf_layout`` coordinates.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    scaling : float, default=1.0
        Radius scale from NetworkX equation 10.
    a : float, default=1.1
        Spring strength for connected node pairs. Must be greater than one.
    etol : float, default=1e-6
        Sum-gradient convergence tolerance.
    dt : float, default=1e-3
        Integration timestep.
    max_iter : int, default=1000
        Maximum iteration guard matching NetworkX's ``n_iter > max_iter``.
    seed : int | None, optional
        NumPy RandomState seed for the random-layout initializer.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    device : torch.device | None, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    if a <= 1:
        raise ValueError("The parameter a should be larger than 1")
    out_device = torch.device("cpu") if device is None else device
    if num_nodes <= 0:
        return _empty_positions(num_nodes=num_nodes, dtype=dtype, device=out_device)
    rng = np.random.RandomState(seed) if seed is not None else np.random.mtrand._rand
    p = rng.rand(num_nodes, 2).astype(np.float32)
    k_matrix = np.ones((num_nodes, num_nodes)) - np.eye(num_nodes)
    if edge_index.numel() > 0:
        edges = edge_index.detach().to(device="cpu", dtype=torch.long)
        for source, target in zip(edges[0].tolist(), edges[1].tolist()):
            if int(source) != int(target):
                k_matrix[int(source), int(target)] = float(a)
    rho = float(scaling) * np.sqrt(num_nodes)
    error = float(etol) + 1.0
    n_iter = 0
    while error > etol:
        diff = p[:, np.newaxis] - p[np.newaxis]
        distances = np.linalg.norm(diff, axis=-1)[..., np.newaxis]
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            change = k_matrix[..., np.newaxis] * diff - rho / distances * diff
        change = np.nansum(change, axis=0)
        p += change * float(dt)
        error = np.linalg.norm(change, axis=-1).sum()
        if n_iter > max_iter:
            break
        n_iter += 1
    return _to_tensor(pos=p, dtype=dtype, device=out_device)


def trivial_star_center(edge_index: torch.Tensor, num_nodes: int) -> int:
    """Choose the deterministic star center by maximum undirected degree.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    int
        Center node index. Ties are resolved by the lowest node index.
    """
    if num_nodes <= 0:
        return 0
    degrees = [0] * num_nodes
    if edge_index.numel() > 0:
        edges = edge_index.detach().to(device="cpu", dtype=torch.long)
        for source, target in zip(edges[0].tolist(), edges[1].tolist()):
            if 0 <= int(source) < num_nodes:
                degrees[int(source)] += 1
            if 0 <= int(target) < num_nodes and int(target) != int(source):
                degrees[int(target)] += 1
    return max(range(num_nodes), key=lambda node: (degrees[node], -node))


def nx_star_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    center: Optional[int] = None,
    scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return igraph-style star coordinates.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]`` used for default center selection.
    num_nodes : int
        Number of graph nodes.
    center : int | None, optional
        Explicit center node. ``None`` uses maximum undirected degree.
    scale : float, default=1.0
        Circle radius for leaf nodes.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    device : torch.device | None, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    out_device = torch.device("cpu") if device is None else device
    if num_nodes <= 1:
        return _empty_positions(num_nodes=num_nodes, dtype=dtype, device=out_device)
    center_node = trivial_star_center(edge_index, num_nodes) if center is None else int(center)
    center_node = min(max(center_node, 0), num_nodes - 1)
    pos = np.zeros((num_nodes, 2), dtype=np.float64)
    leaves = [node for node in range(num_nodes) if node != center_node]
    for leaf_index, node in enumerate(leaves):
        angle = 2.0 * np.pi * leaf_index / len(leaves)
        pos[node, 0] = math.cos(angle) * float(scale)
        pos[node, 1] = math.sin(angle) * float(scale)
    return _to_tensor(pos=pos, dtype=dtype, device=out_device)


def nx_concentric_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return deterministic degree-ring concentric coordinates.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    scale : float, default=1.0
        Outer-ring radius.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    device : torch.device | None, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    out_device = torch.device("cpu") if device is None else device
    if num_nodes <= 1:
        return _empty_positions(num_nodes=num_nodes, dtype=dtype, device=out_device)
    degrees = [0] * num_nodes
    if edge_index.numel() > 0:
        edges = edge_index.detach().to(device="cpu", dtype=torch.long)
        for source, target in zip(edges[0].tolist(), edges[1].tolist()):
            degrees[int(source)] += 1
            if int(target) != int(source):
                degrees[int(target)] += 1
    degree_values = sorted(set(degrees), reverse=True)
    radius_step = float(scale) / max(len(degree_values) - 1, 1)
    pos = np.zeros((num_nodes, 2), dtype=np.float64)
    for ring_index, degree in enumerate(degree_values):
        nodes = [node for node in range(num_nodes) if degrees[node] == degree]
        radius = radius_step * ring_index
        if len(nodes) == num_nodes and ring_index == 0:
            radius = float(scale)
        if radius == 0.0 and len(nodes) == 1:
            continue
        if radius == 0.0:
            radius = radius_step
        for offset, node in enumerate(nodes):
            angle = 2.0 * np.pi * offset / len(nodes)
            pos[node, 0] = math.cos(angle) * radius
            pos[node, 1] = math.sin(angle) * radius
    return _to_tensor(pos=pos, dtype=dtype, device=out_device)


def nx_arc_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    start: int = 0,
    scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return deterministic BFS/input-order arc-diagram coordinates.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    start : int, default=0
        Source node for BFS ordering.
    scale : float, default=1.0
        Half-width after centering.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    device : torch.device | None, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    out_device = torch.device("cpu") if device is None else device
    if num_nodes <= 1:
        return _empty_positions(num_nodes=num_nodes, dtype=dtype, device=out_device)
    layers = nx_bfs_layers(edge_index=edge_index, num_nodes=num_nodes, start=start)
    order = [node for layer in sorted(layers) for node in layers[layer]]
    xs = np.linspace(-float(scale), float(scale), num_nodes)
    pos = np.zeros((num_nodes, 2), dtype=np.float64)
    for index, node in enumerate(order):
        pos[node, 0] = xs[index]
    return _to_tensor(pos=pos, dtype=dtype, device=out_device)


def nx_circlepack_positions(
    num_nodes: int,
    scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return deterministic one-level circle-packing coordinates.

    Parameters
    ----------
    num_nodes : int
        Number of graph nodes.
    scale : float, default=1.0
        Outer packing radius.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    device : torch.device | None, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    out_device = torch.device("cpu") if device is None else device
    if num_nodes <= 1:
        return _empty_positions(num_nodes=num_nodes, dtype=dtype, device=out_device)
    radius = float(scale) * (1.0 - 1.0 / (1.0 + math.sqrt(num_nodes)))
    pos = np.zeros((num_nodes, 2), dtype=np.float64)
    for node in range(num_nodes):
        angle = 2.0 * np.pi * node / num_nodes
        pos[node, 0] = math.cos(angle) * radius
        pos[node, 1] = math.sin(angle) * radius
    return _to_tensor(pos=pos, dtype=dtype, device=out_device)


def nx_planar_positions(
    edge_index: torch.Tensor,
    num_nodes: int,
    scale: float = 1.0,
    dtype: torch.dtype = torch.float64,
    device: Optional[torch.device] = None,
) -> torch.Tensor:
    """Return NetworkX Chrobak-Payne planar-layout coordinates.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.
    scale : float, default=1.0
        NetworkX layout scale.
    dtype : torch.dtype, default=torch.float64
        Output floating-point dtype.
    device : torch.device | None, optional
        Output device. ``None`` uses CPU.

    Returns
    -------
    torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    """
    out_device = torch.device("cpu") if device is None else device
    nx = __import__("networkx")
    graph = nx.Graph()
    graph.add_nodes_from(range(num_nodes))
    if edge_index.numel() > 0:
        graph.add_edges_from((int(s), int(t)) for s, t in edge_index.detach().cpu().t().tolist())
    pos_map = nx.planar_layout(graph, scale=scale)
    pos = np.vstack([pos_map[node] for node in range(num_nodes)]).astype(np.float64)
    return _to_tensor(pos=pos, dtype=dtype, device=out_device)


@register_op
@dataclass
class NetworkXSimpleLayout(Op):
    """Assign positions using one NetworkX simple-layout source port.

    Parameters
    ----------
    algorithm : str
        Layout name: ``circular``, ``shell``, ``spiral``, ``bipartite``,
        ``multipartite``, ``bfs``, ``arf``, ``star``, ``concentric``,
        ``circlepack``, ``arc``, or ``planar``.
    params : dict[str, Any]
        Algorithm-specific keyword parameters.
    """

    algorithm: str
    params: Dict[str, Any]

    name = "networkx_simple_layout"
    category = OpCategory.COORDINATE
    reads = ("edge_index", "N")
    writes = ("pos", "extras")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Assign final coordinates to ``state.pos``.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable graph inputs, including ``edge_index`` with shape
            ``[2, E]`` and node count ``N``.
        state : SolveState
            Mutable solve state receiving ``pos`` and provenance extras.
        ctx : RuntimeContext
            Runtime context; accepted for composable-op API consistency.

        Returns
        -------
        SolveState
            State with NetworkX-simple positions and metadata populated.
        """
        del ctx
        dtype = self.params.get("dtype", torch.float64)
        device = layout_device(problem.edge_index, problem.node_sizes)
        scale = float(self.params.get("scale", 1.0))
        algorithm = self.algorithm.lower()
        if algorithm == "circular":
            pos = nx_circular_positions(problem.num_nodes, scale=scale, dtype=dtype, device=device)
            extras: Dict[str, Any] = {}
        elif algorithm == "shell":
            nlist = self.params.get("nlist")
            if nlist == "bfs":
                nlist = list(nx_bfs_layers(problem.edge_index, problem.num_nodes).values())
            pos = nx_shell_positions(
                problem.num_nodes,
                nlist=nlist,
                rotate=self.params.get("rotate"),
                scale=scale,
                dtype=dtype,
                device=device,
            )
            extras = {"nlist": nlist}
        elif algorithm == "spiral":
            pos = nx_spiral_positions(
                problem.num_nodes,
                scale=scale,
                resolution=float(self.params.get("resolution", 0.35)),
                equidistant=bool(self.params.get("equidistant", False)),
                dtype=dtype,
                device=device,
            )
            extras = {}
        elif algorithm == "bipartite":
            nodes = self.params.get("nodes")
            if nodes is None:
                nodes = nx_bipartite_node_set(problem.edge_index, problem.num_nodes)
            pos = nx_bipartite_positions(
                problem.num_nodes,
                nodes=list(nodes),
                align=str(self.params.get("align", "vertical")),
                scale=scale,
                aspect_ratio=float(self.params.get("aspect_ratio", 4.0 / 3.0)),
                dtype=dtype,
                device=device,
            )
            extras = {"nodes": list(nodes)}
        elif algorithm in {"multipartite", "bfs"}:
            start = int(self.params.get("start", 0))
            layers = self.params.get("layers")
            if layers is None:
                layers = nx_bfs_layers(problem.edge_index, problem.num_nodes, start=start)
            pos = nx_multipartite_positions(
                layers=layers,
                num_nodes=problem.num_nodes,
                align=str(self.params.get("align", "vertical")),
                scale=scale,
                dtype=dtype,
                device=device,
            )
            extras = {"layers": layers, "start": start}
        elif algorithm == "arf":
            pos = nx_arf_positions(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                scaling=float(self.params.get("scaling", 1.0)),
                a=float(self.params.get("a", 1.1)),
                etol=float(self.params.get("etol", 1.0e-6)),
                dt=float(self.params.get("dt", 1.0e-3)),
                max_iter=int(self.params.get("max_iter", 1000)),
                seed=self.params.get("seed"),
                dtype=dtype,
                device=device,
            )
            extras = {"seed": self.params.get("seed")}
        elif algorithm == "star":
            pos = nx_star_positions(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                center=self.params.get("center"),
                scale=scale,
                dtype=dtype,
                device=device,
            )
            extras = {"center": self.params.get("center")}
        elif algorithm == "concentric":
            pos = nx_concentric_positions(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                scale=scale,
                dtype=dtype,
                device=device,
            )
            extras = {}
        elif algorithm == "circlepack":
            pos = nx_circlepack_positions(
                num_nodes=problem.num_nodes,
                scale=scale,
                dtype=dtype,
                device=device,
            )
            extras = {}
        elif algorithm == "arc":
            start = int(self.params.get("start", 0))
            pos = nx_arc_positions(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                start=start,
                scale=scale,
                dtype=dtype,
                device=device,
            )
            extras = {"start": start}
        elif algorithm == "planar":
            pos = nx_planar_positions(
                edge_index=problem.edge_index,
                num_nodes=problem.num_nodes,
                scale=scale,
                dtype=dtype,
                device=device,
            )
            extras = {}
        else:
            raise ValueError(f"Unsupported NetworkX simple layout: {self.algorithm!r}.")
        state.pos = pos
        state.extras[algorithm] = extras
        return state


__all__ = [
    "NetworkXSimpleLayout",
    "nx_arf_positions",
    "nx_bfs_layers",
    "nx_bipartite_node_set",
    "nx_bipartite_positions",
    "nx_arc_positions",
    "nx_circular_positions",
    "nx_circlepack_positions",
    "nx_concentric_positions",
    "nx_multipartite_positions",
    "nx_planar_positions",
    "nx_rescale_layout",
    "nx_shell_positions",
    "nx_spiral_positions",
    "nx_star_positions",
    "trivial_star_center",
]
