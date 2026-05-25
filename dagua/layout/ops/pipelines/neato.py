"""Graphviz neato-compatible layout pipeline."""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

import torch

from dagua.layout.ops.pipelines.stress_majorization import layout_stress_majorization_pipeline

_GRAPHVIZ_VPSC_DEFAULT_GAP = 1.0 / 9.0
_GRAPHVIZ_VPSC_X_SCALE = 1.0001
_VPSC_FEASIBILITY_TOLERANCE = 1.0e-9


@dataclass
class _VPSCConstraint:
    """One-dimensional separation constraint for Graphviz VPSC overlap removal.

    Parameters
    ----------
    left : int
        Index of the variable constrained to remain on the lower side.
    right : int
        Index of the variable constrained to remain on the upper side.
    gap : float
        Minimum separation such that ``place[right] >= place[left] + gap``.
    """

    left: int
    right: int
    gap: float


@dataclass
class _VPSCBlock:
    """Active set block used by the VPSC feasibility projection.

    Parameters
    ----------
    variables : list[int]
        Variable indices in the block.
    offsets : dict[int, float]
        Per-variable offset from the block reference position.
    position : float
        Current block reference coordinate.
    """

    variables: List[int]
    offsets: dict[int, float]
    position: float


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


def _rectangles_from_positions(
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    gap: Tuple[float, float],
    x_size_scale: float,
) -> List[Tuple[float, float, float, float]]:
    """Build Graphviz-style node rectangles for overlap-constraint generation.

    Parameters
    ----------
    positions : torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    gap : tuple[float, float]
        Total horizontal and vertical gap added to node dimensions.
    x_size_scale : float
        Scale applied to node widths. Graphviz's x pass uses ``1.0001`` so a
        horizontally resolved overlap is not regenerated as a vertical overlap.

    Returns
    -------
    list[tuple[float, float, float, float]]
        Rectangles as ``(min_x, max_x, min_y, max_y)``.
    """
    pos_cpu = positions.detach().to(device="cpu", dtype=torch.float64)
    sizes_cpu = node_sizes.detach().to(device="cpu", dtype=torch.float64)
    rectangles: List[Tuple[float, float, float, float]] = []
    for index in range(pos_cpu.shape[0]):
        center_x = float(pos_cpu[index, 0].item())
        center_y = float(pos_cpu[index, 1].item())
        width = x_size_scale * float(sizes_cpu[index, 0].item()) + gap[0]
        height = float(sizes_cpu[index, 1].item()) + gap[1]
        rectangles.append(
            (
                center_x - width / 2.0,
                center_x + width / 2.0,
                center_y - height / 2.0,
                center_y + height / 2.0,
            )
        )
    return rectangles


def _rectangle_overlap_x(
    left: Tuple[float, float, float, float],
    right: Tuple[float, float, float, float],
) -> float:
    """Return horizontal intersection width for two rectangles.

    Parameters
    ----------
    left : tuple[float, float, float, float]
        First rectangle as ``(min_x, max_x, min_y, max_y)``.
    right : tuple[float, float, float, float]
        Second rectangle as ``(min_x, max_x, min_y, max_y)``.

    Returns
    -------
    float
        Positive overlap width, zero at separation or tangency, or a negative
        gap when the rectangles are disjoint horizontally.
    """
    return min(left[1], right[1]) - max(left[0], right[0])


def _rectangle_overlap_y(
    lower: Tuple[float, float, float, float],
    upper: Tuple[float, float, float, float],
) -> float:
    """Return vertical intersection height for two rectangles.

    Parameters
    ----------
    lower : tuple[float, float, float, float]
        First rectangle as ``(min_x, max_x, min_y, max_y)``.
    upper : tuple[float, float, float, float]
        Second rectangle as ``(min_x, max_x, min_y, max_y)``.

    Returns
    -------
    float
        Positive overlap height, zero at separation or tangency, or a negative
        gap when the rectangles are disjoint vertically.
    """
    return min(lower[3], upper[3]) - max(lower[2], upper[2])


def _generate_vpsc_x_constraints(
    rectangles: List[Tuple[float, float, float, float]],
) -> List[_VPSCConstraint]:
    """Generate Graphviz-style horizontal non-overlap constraints.

    Parameters
    ----------
    rectangles : list[tuple[float, float, float, float]]
        Node rectangles as ``(min_x, max_x, min_y, max_y)``.

    Returns
    -------
    list[_VPSCConstraint]
        Horizontal separation constraints generated by the Adaptagrams
        sweep-line neighbor-list heuristic used by Graphviz's VPSC pass.
    """
    centers = [0.5 * (rect[0] + rect[1]) for rect in rectangles]
    events: List[Tuple[float, int, int]] = []
    for index, rect in enumerate(rectangles):
        events.append((rect[2], 0, index))
        events.append((rect[3], 1, index))
    events.sort(key=lambda event: (event[0], event[1], event[2]))

    scanline: List[int] = []
    left_neighbors: List[set[int]] = [set() for _ in rectangles]
    right_neighbors: List[set[int]] = [set() for _ in rectangles]
    constraints: List[_VPSCConstraint] = []
    for _, event_type, node in events:
        if event_type == 0:
            scanline.append(node)
            scanline.sort(key=lambda idx: (centers[idx], idx))
            position = scanline.index(node)

            for candidate in reversed(scanline[:position]):
                overlap_x = _rectangle_overlap_x(rectangles[candidate], rectangles[node])
                if overlap_x <= 0.0:
                    left_neighbors[node].add(candidate)
                    break
                if overlap_x <= _rectangle_overlap_y(rectangles[candidate], rectangles[node]):
                    left_neighbors[node].add(candidate)

            for candidate in scanline[position + 1 :]:
                overlap_x = _rectangle_overlap_x(rectangles[candidate], rectangles[node])
                if overlap_x <= 0.0:
                    right_neighbors[node].add(candidate)
                    break
                if overlap_x <= _rectangle_overlap_y(rectangles[candidate], rectangles[node]):
                    right_neighbors[node].add(candidate)
            continue

        for candidate in sorted(left_neighbors[node], key=lambda idx: (centers[idx], idx)):
            width = rectangles[node][1] - rectangles[node][0]
            candidate_width = rectangles[candidate][1] - rectangles[candidate][0]
            constraints.append(_VPSCConstraint(candidate, node, (width + candidate_width) / 2.0))
            right_neighbors[candidate].discard(node)
        for candidate in sorted(right_neighbors[node], key=lambda idx: (centers[idx], idx)):
            width = rectangles[node][1] - rectangles[node][0]
            candidate_width = rectangles[candidate][1] - rectangles[candidate][0]
            constraints.append(_VPSCConstraint(node, candidate, (width + candidate_width) / 2.0))
            left_neighbors[candidate].discard(node)
        if node in scanline:
            scanline.remove(node)
    return constraints


def _generate_vpsc_y_constraints(
    rectangles: List[Tuple[float, float, float, float]],
) -> List[_VPSCConstraint]:
    """Generate Graphviz-style vertical non-overlap constraints.

    Parameters
    ----------
    rectangles : list[tuple[float, float, float, float]]
        Node rectangles as ``(min_x, max_x, min_y, max_y)``.

    Returns
    -------
    list[_VPSCConstraint]
        Vertical separation constraints generated after the horizontal VPSC
        pass has updated x coordinates.
    """
    centers = [0.5 * (rect[2] + rect[3]) for rect in rectangles]
    events: List[Tuple[float, int, int]] = []
    for index, rect in enumerate(rectangles):
        events.append((rect[0], 0, index))
        events.append((rect[1], 1, index))
    events.sort(key=lambda event: (event[0], event[1], event[2]))

    scanline: List[int] = []
    first_above: List[Optional[int]] = [None for _ in rectangles]
    first_below: List[Optional[int]] = [None for _ in rectangles]
    constraints: List[_VPSCConstraint] = []
    for _, event_type, node in events:
        if event_type == 0:
            scanline.append(node)
            scanline.sort(key=lambda idx: (centers[idx], idx))
            position = scanline.index(node)
            if position > 0:
                candidate = scanline[position - 1]
                first_above[node] = candidate
                first_below[candidate] = node
            if position + 1 < len(scanline):
                candidate = scanline[position + 1]
                first_below[node] = candidate
                first_above[candidate] = node
            continue

        above = first_above[node]
        below = first_below[node]
        if above is not None:
            height = rectangles[node][3] - rectangles[node][2]
            candidate_height = rectangles[above][3] - rectangles[above][2]
            constraints.append(_VPSCConstraint(above, node, (height + candidate_height) / 2.0))
            first_below[above] = below
        if below is not None:
            height = rectangles[node][3] - rectangles[node][2]
            candidate_height = rectangles[below][3] - rectangles[below][2]
            constraints.append(_VPSCConstraint(node, below, (height + candidate_height) / 2.0))
            first_above[below] = above
        if node in scanline:
            scanline.remove(node)
    return constraints


def _block_positions(blocks: List[_VPSCBlock], num_variables: int) -> List[float]:
    """Materialize per-variable positions from active-set blocks.

    Parameters
    ----------
    blocks : list[_VPSCBlock]
        Current active-set blocks.
    num_variables : int
        Number of variables to materialize.

    Returns
    -------
    list[float]
        Variable coordinates in index order.
    """
    positions = [0.0 for _ in range(num_variables)]
    for block in blocks:
        for variable in block.variables:
            positions[variable] = block.position + block.offsets[variable]
    return positions


def _merge_vpsc_blocks(
    left_block: _VPSCBlock,
    right_block: _VPSCBlock,
    constraint: _VPSCConstraint,
    desired: List[float],
) -> _VPSCBlock:
    """Merge two active-set blocks across a violated VPSC constraint.

    Parameters
    ----------
    left_block : _VPSCBlock
        Block containing the lower-side variable.
    right_block : _VPSCBlock
        Block containing the upper-side variable.
    constraint : _VPSCConstraint
        Violated separation constraint used as the active merge edge.
    desired : list[float]
        Original desired coordinates for each variable.

    Returns
    -------
    _VPSCBlock
        Merged block with offsets satisfying ``constraint`` at equality.
    """
    shift = (
        left_block.offsets[constraint.left] + constraint.gap - right_block.offsets[constraint.right]
    )
    offsets = dict(left_block.offsets)
    variables = list(left_block.variables)
    for variable in right_block.variables:
        variables.append(variable)
        offsets[variable] = right_block.offsets[variable] + shift
    position = sum(desired[variable] - offsets[variable] for variable in variables) / float(
        len(variables)
    )
    return _VPSCBlock(variables=variables, offsets=offsets, position=position)


def _solve_vpsc_1d(
    desired: List[float],
    constraints: List[_VPSCConstraint],
) -> List[float]:
    """Project one coordinate onto VPSC separation constraints.

    Parameters
    ----------
    desired : list[float]
        Original coordinates for one axis.
    constraints : list[_VPSCConstraint]
        Separation constraints for that axis.

    Returns
    -------
    list[float]
        Feasible coordinates that stay close to ``desired``. This ports the
        merge-satisfy part of Graphviz's VPSC solver; the block-splitting
        refinement is intentionally left for the sibling solver integration.
    """
    if not constraints:
        return list(desired)

    blocks: List[_VPSCBlock] = [
        _VPSCBlock(variables=[index], offsets={index: 0.0}, position=value)
        for index, value in enumerate(desired)
    ]
    variable_to_block = list(range(len(desired)))
    max_merges = max(1, len(desired) * max(1, len(constraints)) * 2)
    for _ in range(max_merges):
        positions = _block_positions(blocks=blocks, num_variables=len(desired))
        min_slack = 0.0
        violated: Optional[_VPSCConstraint] = None
        for constraint in constraints:
            slack = positions[constraint.right] - positions[constraint.left] - constraint.gap
            if slack < min_slack:
                min_slack = slack
                violated = constraint
        if violated is None or min_slack >= -_VPSC_FEASIBILITY_TOLERANCE:
            return positions

        left_id = variable_to_block[violated.left]
        right_id = variable_to_block[violated.right]
        if left_id == right_id:
            continue
        merged = _merge_vpsc_blocks(
            left_block=blocks[left_id],
            right_block=blocks[right_id],
            constraint=violated,
            desired=desired,
        )
        keep_id = min(left_id, right_id)
        drop_id = max(left_id, right_id)
        blocks[keep_id] = merged
        del blocks[drop_id]
        for block_id, block in enumerate(blocks):
            for variable in block.variables:
                variable_to_block[variable] = block_id
    return _block_positions(blocks=blocks, num_variables=len(desired))


def _has_axis_aligned_overlap(
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    gap: Tuple[float, float],
) -> bool:
    """Return whether any expanded node rectangles overlap.

    Parameters
    ----------
    positions : torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    gap : tuple[float, float]
        Total horizontal and vertical gap added to node dimensions.

    Returns
    -------
    bool
        ``True`` when at least one pair overlaps in both axes.
    """
    rectangles = _rectangles_from_positions(
        positions=positions,
        node_sizes=node_sizes,
        gap=gap,
        x_size_scale=1.0,
    )
    for left in range(len(rectangles)):
        for right in range(left + 1, len(rectangles)):
            if (
                _rectangle_overlap_x(rectangles[left], rectangles[right]) > 0.0
                and _rectangle_overlap_y(rectangles[left], rectangles[right]) > 0.0
            ):
                return True
    return False


def _prism_scale_until_disjoint(
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    gap: Tuple[float, float],
    max_iterations: int,
) -> torch.Tensor:
    """Apply Graphviz PRISM-style uniform scaling before VPSC cleanup.

    Parameters
    ----------
    positions : torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    gap : tuple[float, float]
        Total horizontal and vertical gap added to node dimensions.
    max_iterations : int
        Maximum number of scale doublings.

    Returns
    -------
    torch.Tensor
        Scaled coordinates centered around their input centroid.
    """
    if positions.shape[0] < 2:
        return positions
    centered = positions - positions.mean(dim=0, keepdim=True)
    if not _has_axis_aligned_overlap(centered, node_sizes, gap):
        return positions
    scale = 1.0
    scaled = centered
    for _ in range(max_iterations):
        scale *= 2.0
        scaled = centered * scale
        if not _has_axis_aligned_overlap(scaled, node_sizes, gap):
            break
    return scaled + positions.mean(dim=0, keepdim=True)


def _remove_overlaps_vpsc(
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    gap: Tuple[float, float],
    method: str,
    prism_scaling_iterations: int,
) -> torch.Tensor:
    """Remove node overlaps with Graphviz VPSC-style x/y separation passes.

    Parameters
    ----------
    positions : torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor
        Node-size tensor with shape ``[N, 2]``.
    gap : tuple[float, float]
        Total horizontal and vertical gap added to node dimensions.
    method : str
        Overlap method. ``"vpsc"`` performs the static two-axis VPSC pass;
        ``"prism"`` adds a PRISM-compatible uniform scaling prepass.
    prism_scaling_iterations : int
        Maximum scale doublings for the PRISM-compatible prepass.

    Returns
    -------
    torch.Tensor
        Overlap-adjusted coordinate tensor with the same dtype and device as
        ``positions``.

    Raises
    ------
    ValueError
        If ``method`` is not ``"vpsc"`` or ``"prism"``.
    """
    if method not in {"vpsc", "prism"}:
        raise ValueError("overlap_method must be 'vpsc' or 'prism'.")
    if positions.shape[0] < 2:
        return positions

    working = positions.detach().to(device="cpu", dtype=torch.float64).clone()
    sizes = node_sizes.detach().to(device="cpu", dtype=torch.float64)
    if method == "prism":
        working = _prism_scale_until_disjoint(
            positions=working,
            node_sizes=sizes,
            gap=gap,
            max_iterations=prism_scaling_iterations,
        )

    x_rectangles = _rectangles_from_positions(
        positions=working,
        node_sizes=sizes,
        gap=gap,
        x_size_scale=_GRAPHVIZ_VPSC_X_SCALE,
    )
    x_constraints = _generate_vpsc_x_constraints(x_rectangles)
    x_desired = [float(value) for value in working[:, 0].tolist()]
    working[:, 0] = torch.tensor(
        _solve_vpsc_1d(x_desired, x_constraints),
        dtype=working.dtype,
        device=working.device,
    )

    y_rectangles = _rectangles_from_positions(
        positions=working,
        node_sizes=sizes,
        gap=gap,
        x_size_scale=1.0,
    )
    y_constraints = _generate_vpsc_y_constraints(y_rectangles)
    y_desired = [float(value) for value in working[:, 1].tolist()]
    working[:, 1] = torch.tensor(
        _solve_vpsc_1d(y_desired, y_constraints),
        dtype=working.dtype,
        device=working.device,
    )
    return working.to(device=positions.device, dtype=positions.dtype)


def remove_neato_overlap_fidelity(
    positions: torch.Tensor,
    node_sizes: Optional[torch.Tensor],
    overlap_method: str = "vpsc",
    overlap_gap: float = _GRAPHVIZ_VPSC_DEFAULT_GAP,
    prism_scaling_iterations: int = 32,
) -> torch.Tensor:
    """Run the opt-in Graphviz neato overlap-removal fidelity component.

    Parameters
    ----------
    positions : torch.Tensor
        Coordinate tensor with shape ``[N, 2]``.
    node_sizes : torch.Tensor, optional
        Node-size tensor with shape ``[N, 2]``. When absent, Graphviz's
        rectangle constraints cannot be generated and coordinates are returned
        unchanged.
    overlap_method : str, default="vpsc"
        Overlap method to port. ``"vpsc"`` runs static VPSC post-processing;
        ``"prism"`` runs a PRISM-style scaling prepass followed by VPSC.
    overlap_gap : float, default=1/9
        Total separation gap added to both node dimensions. The default matches
        Graphviz 7.0.5's postscript-point default margin conversion.
    prism_scaling_iterations : int, default=32
        Maximum number of scale doublings for the PRISM-compatible prepass.

    Returns
    -------
    torch.Tensor
        Adjusted coordinate tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If tensor shapes or scalar options are invalid.
    """
    if positions.ndim != 2 or positions.shape[1] != 2:
        raise ValueError("positions must be shape [N, 2].")
    if node_sizes is None:
        return positions
    if node_sizes.ndim != 2 or node_sizes.shape != positions.shape:
        raise ValueError("node_sizes must match positions with shape [N, 2].")
    if overlap_gap < 0.0:
        raise ValueError("overlap_gap must be non-negative.")
    if prism_scaling_iterations < 0:
        raise ValueError("prism_scaling_iterations must be non-negative.")
    return _remove_overlaps_vpsc(
        positions=positions,
        node_sizes=node_sizes,
        gap=(overlap_gap, overlap_gap),
        method=overlap_method,
        prism_scaling_iterations=prism_scaling_iterations,
    )


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
    fidelity_mode: Union[bool, str] = False,
    fidelity_dtype: torch.dtype = torch.float32,
    overlap_removal: bool = True,
    overlap_method: str = "vpsc",
    overlap_gap: float = _GRAPHVIZ_VPSC_DEFAULT_GAP,
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
    fidelity_mode : bool or str, default=False
        Whether to enable opt-in Graphviz post-processing fidelity behavior.
        Passing ``"graphviz"`` also enables the Graphviz PCA initialization
        and packed conjugate-gradient stress solver.
    overlap_removal : bool, default=True
        Whether to run neato overlap removal when ``fidelity_mode`` is true.
    overlap_method : {"vpsc", "prism"}, default="vpsc"
        Graphviz overlap-removal method to apply in fidelity mode.
    overlap_gap : float, default=1/9
        Total gap added to node dimensions during overlap removal.

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
    if isinstance(fidelity_mode, str) and fidelity_mode not in {"graphviz", "graphviz_neato"}:
        raise ValueError("neato fidelity_mode must be bool, 'graphviz', or 'graphviz_neato'.")
    resolved_iterations = maxiter if iterations is None else iterations
    use_graphviz_solver = fidelity_mode == "graphviz"
    stress_fidelity_mode = "graphviz" if use_graphviz_solver else "graphviz_neato"
    # Graphviz default neato does not run VPSC unless the graph's overlap
    # adjustment mode requests it. The compatibility path keeps the older
    # opt-in behavior; the packed-CG solver path follows the default reference.
    postprocess_fidelity = bool(fidelity_mode) and not use_graphviz_solver
    if not pack or num_nodes <= 1:
        direct_result = layout_stress_majorization_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            iterations=resolved_iterations,
            seed=seed,
            edge_weights=edge_weights,
            trace_every=trace_every,
            fidelity_mode=stress_fidelity_mode,
            epsilon=epsilon,
        )
        if trace_every > 0:
            direct_pos, traces = direct_result
            if postprocess_fidelity and overlap_removal:
                direct_pos = remove_neato_overlap_fidelity(
                    positions=direct_pos,
                    node_sizes=node_sizes,
                    overlap_method=overlap_method,
                    overlap_gap=overlap_gap,
                )
            return direct_pos, traces
        if postprocess_fidelity and overlap_removal:
            direct_result = remove_neato_overlap_fidelity(
                positions=direct_result,
                node_sizes=node_sizes,
                overlap_method=overlap_method,
                overlap_gap=overlap_gap,
            )
        return direct_result

    components = _weak_components(edge_index=edge_index, num_nodes=num_nodes)
    if len(components) <= 1:
        connected_result = layout_stress_majorization_pipeline(
            edge_index=edge_index,
            num_nodes=num_nodes,
            node_sizes=node_sizes,
            iterations=resolved_iterations,
            seed=seed,
            edge_weights=edge_weights,
            trace_every=trace_every,
            fidelity_mode=stress_fidelity_mode,
            epsilon=epsilon,
        )
        if trace_every > 0:
            connected_pos, traces = connected_result
            if postprocess_fidelity and overlap_removal:
                connected_pos = remove_neato_overlap_fidelity(
                    positions=connected_pos,
                    node_sizes=node_sizes,
                    overlap_method=overlap_method,
                    overlap_gap=overlap_gap,
                )
            return connected_pos, traces
        if postprocess_fidelity and overlap_removal:
            connected_result = remove_neato_overlap_fidelity(
                positions=connected_result,
                node_sizes=node_sizes,
                overlap_method=overlap_method,
                overlap_gap=overlap_gap,
            )
        return connected_result

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
            fidelity_mode=stress_fidelity_mode,
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
    if postprocess_fidelity and overlap_removal:
        packed = remove_neato_overlap_fidelity(
            positions=packed,
            node_sizes=node_sizes,
            overlap_method=overlap_method,
            overlap_gap=overlap_gap,
        )
    return (packed, []) if trace_every > 0 else packed


__all__ = ["layout_neato_pipeline", "remove_neato_overlap_fidelity"]
