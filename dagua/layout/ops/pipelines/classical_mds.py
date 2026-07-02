"""Classical multidimensional scaling layout pipeline."""

from __future__ import annotations

import ctypes
import math
import random
from typing import Optional

import numpy as np
import scipy.linalg
import torch

from dagua.layout.ops.base import Pipeline
from dagua.layout.ops.distance import ClassicalMDSDistanceMatrix
from dagua.layout.ops.embed import ClassicalMDSComputeEmbedding, ClassicalMDSComputeEmbeddingConfig
from dagua.layout.ops.graph_utils import shortest_path_distances as _shortest_path_distances
from dagua.layout.ops.pipelines import resolve_fidelity_dtype
from dagua.layout.ops.postprocess import (
    ClassicalMDSFinalizePositions,
    ClassicalMDSFinalizePositionsConfig,
)
from dagua.layout.ops.state import (
    ExecutionPlan,
    LayoutProblem,
    RuntimeContext,
    SolveState,
)

_OGDF_EDGE_COST = 100.0
_OGDF_POWER_EPSILON = 1.0 - 1e-10
_OGDF_CENTERING_FACTOR = -0.5
_LIBC_RAND_MAX = 2_147_483_647
_IGRAPH_LAYOUT_SCALE = 50.0
_IGRAPH_DLA_GRID_STEPS = 200
_IGRAPH_DLA_AREA_FACTOR = 5.0
_IGRAPH_DLA_WALK_STEP_DIVISOR = 100.0
_IGRAPH_DLA_KILL_RADIUS_PAD = 5.0
_IGRAPH_DLA_MAX_TOTAL_STEPS = 10_000_000
_IGRAPH_DLA_MAX_RESTARTS = 1_000_000


def build_classical_mds_pipeline(
    *,
    igraph_fidelity: bool = False,
    ogdf_fidelity: bool = False,
) -> Pipeline:
    """Build a classical multidimensional scaling pipeline.

    Reference fidelity
    ------------------
    Targets: igraph 1.0.0 MDS layout / Torgerson (1952) classical metric MDS.
    Fidelity mode: ``igraph_fidelity=True`` uses igraph-compatible raw
        embedding and final scaling semantics.
    Verified at: final 100-seed report, strong equivalent; median RMSD 0.000
        for both default and igraph-fidelity variants.
    Known divergences:
        - Graph distances are prepared in Dagua tensor ops rather than through
          igraph's C path.
        - Repeated top eigenvalues are irreducible for bit-exact matching
          without porting igraph's vendored LAPACK 3.4.2 ``dsyevr`` path:
          igraph asks ``dsyevr`` for the largest algebraic dimensions with
          ``range='I'``, ``uplo='U'``, and ``abstol=1e-14``. When the largest
          eigenvalue multiplicity exceeds the requested two layout dimensions,
          LAPACK returns an implementation-dependent 2D basis from that larger
          eigenspace. The chosen basis depends on the tridiagonal reduction and
          inverse-iteration details inside igraph's vendored LAPACK. A direct
          ``scipy.linalg.lapack.dsyevr`` call with igraph's parameters
          (``range='I'``, ``uplo='U'``, ``abstol=1e-14``, same eigenvalue
          index range) was tested and still chose a different basis, so SciPy's
          exposed ``evr``/``evx``/``dsyevr`` drivers can match the eigenspace
          but not the selected basis on symmetric fixtures such as Petersen
          and complete graphs.
        - Disconnected-graph behavior follows the benchmark distance matrix
          contract, not arbitrary user-provided dissimilarities.

    Parameters
    ----------
    igraph_fidelity : bool, default=False
        If ``True``, opt into igraph-compatible raw embedding and final scaling
        semantics for benchmark parity checks.
    ogdf_fidelity : bool, default=False
        Reserved for the public wrapper's OGDF-compatible full-PivotMDS path.

    Returns
    -------
    Pipeline
        Pipeline implementing classical MDS. The pipeline produces final node
        coordinates by computing the all-pairs graph distance matrix, solving
        the double-centered eigendecomposition, and finalizing the embedding
        into a 2D layout.
    """
    if ogdf_fidelity:
        raise ValueError("ogdf_fidelity is only supported by layout_classical_mds_pipeline.")
    return Pipeline(
        [
            ClassicalMDSDistanceMatrix(),
            ClassicalMDSComputeEmbedding(
                config=ClassicalMDSComputeEmbeddingConfig(igraph_fidelity=igraph_fidelity)
            ),
            ClassicalMDSFinalizePositions(
                config=ClassicalMDSFinalizePositionsConfig(igraph_fidelity=igraph_fidelity)
            ),
        ],
        name="classical_mds_pipeline",
    )


def layout_classical_mds_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    seed: int = 42,
    edge_weights: Optional[torch.Tensor] = None,
    igraph_fidelity: bool = False,
    ogdf_fidelity: bool = False,
    fidelity_dtype: Optional[torch.dtype] = None,
) -> torch.Tensor:
    """Run the classical multidimensional scaling pipeline.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of nodes ``N`` in the graph.
    node_sizes : torch.Tensor, optional
        Optional node-size tensor with shape ``[N, 2]`` used only to pick a
        stable output extent.
    seed : int, default=42
        Accepted for interface compatibility. Classical MDS is deterministic
        once graph distances are fixed.
    edge_weights : torch.Tensor, optional
        Optional edge-weight tensor with shape ``[E]``.
    igraph_fidelity : bool, default=False
        If ``True``, ignore edge weights and use igraph-compatible embedding
        and scaling semantics. This is intended for fidelity benchmarking
        against ``igraph.layout("mds")``.
    ogdf_fidelity : bool, default=False
        If ``True``, run OGDF ``PivotMDS`` with all nodes as pivots. This
        matches OGDF's documented classical-MDS mode and uses uniform edge cost
        ``100`` plus OGDF's fixed ``srand(0)`` power-iteration basis.
    fidelity_dtype : torch.dtype, optional
        Internal and returned dtype for fidelity-mode comparisons. ``None``
        defaults to ``torch.float64`` when fidelity mode is selected.

    Returns
    -------
    torch.Tensor
        Final position tensor with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If ``num_nodes`` or ``edge_weights`` are invalid.
    RuntimeError
        If the pipeline does not produce final positions.
    """
    _ = seed, node_sizes

    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if igraph_fidelity and ogdf_fidelity:
        raise ValueError("igraph_fidelity and ogdf_fidelity are mutually exclusive.")
    if edge_weights is not None:
        if edge_weights.ndim != 1:
            raise ValueError("edge_weights must have shape [E].")
        if edge_weights.shape[0] != edge_index.shape[1]:
            raise ValueError(
                f"edge_weights length {edge_weights.shape[0]} != edge count {edge_index.shape[1]}"
            )
    if ogdf_fidelity:
        return _layout_ogdf_classical_mds(
            edge_index=edge_index,
            num_nodes=num_nodes,
            fidelity_dtype=resolve_fidelity_dtype(True, fidelity_dtype),
        )
    if igraph_fidelity:
        return _layout_igraph_classical_mds(
            edge_index=edge_index,
            num_nodes=num_nodes,
            seed=seed,
            output_dtype=torch.float32 if fidelity_dtype is None else fidelity_dtype,
            use_two_node_special=igraph_fidelity,
        )
    if edge_weights is None:
        components = _weak_components_igraph_order(edge_index=edge_index, num_nodes=num_nodes)
        if len(components) > 1:
            return _layout_igraph_classical_mds(
                edge_index=edge_index,
                num_nodes=num_nodes,
                seed=seed,
                output_dtype=torch.float32 if fidelity_dtype is None else fidelity_dtype,
                use_two_node_special=True,
            )

    problem = LayoutProblem(
        edge_index=edge_index,
        num_nodes=num_nodes,
        node_sizes=node_sizes,
        edge_weights=None if igraph_fidelity else edge_weights,
    )
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device="cpu"))
    final_state = build_classical_mds_pipeline(igraph_fidelity=igraph_fidelity).apply(
        problem,
        state,
        ctx,
    )
    if final_state.pos is None:
        raise RuntimeError("Classical MDS pipeline did not produce final positions.")
    return final_state.pos


def _layout_igraph_classical_mds(
    edge_index: torch.Tensor,
    num_nodes: int,
    seed: int,
    output_dtype: torch.dtype,
    use_two_node_special: bool,
) -> torch.Tensor:
    """Run igraph-compatible classical MDS for connected benchmark graphs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes ``N``.
    seed : int
        Seed for the igraph-compatible disconnected-component DLA merge.
    output_dtype : torch.dtype
        Dtype used for returned coordinates.
    use_two_node_special : bool
        Whether to use igraph's ``[[0, 0], [1, 1]]`` raw layout for two-node
        graphs. The public default keeps the legacy two-node behavior.

    Returns
    -------
    torch.Tensor
        igraph-scaled coordinates with shape ``[N, 2]``.

    Notes
    -----
    igraph 1.0.0 computes unweighted all-pairs distances, squares the distance
    matrix in place, double-centers via row means and a grand mean, calls
    LAPACK ``dsyevr`` for the two largest algebraic eigenpairs, and writes the
    selected dimensions in reverse column order. Classical MDS itself has no
    RNG; disconnected-graph DLA packing is the only stochastic behavior.
    """
    output_device = edge_index.device if edge_index.numel() > 0 else torch.device("cpu")
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=output_dtype, device=output_device)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=output_dtype, device=output_device)
    components = _weak_components_igraph_order(edge_index=edge_index, num_nodes=num_nodes)
    if len(components) > 1:
        distances = _shortest_path_distances(
            edge_index=edge_index,
            num_nodes=num_nodes,
            edge_weights=None,
        )
        coordinates = _layout_igraph_disconnected_mds(
            distances=distances,
            components=components,
            seed=seed,
        )
        return torch.from_numpy(coordinates * _IGRAPH_LAYOUT_SCALE).to(
            dtype=output_dtype,
            device=output_device,
        )
    if num_nodes == 2 and use_two_node_special:
        return (
            torch.tensor([[0.0, 0.0], [1.0, 1.0]], dtype=torch.float64, device=output_device)
            * _IGRAPH_LAYOUT_SCALE
        ).to(dtype=output_dtype)

    distances = _shortest_path_distances(
        edge_index=edge_index,
        num_nodes=num_nodes,
        edge_weights=None,
    )
    coordinates = _igraph_mds_single_from_distances(
        distances=distances,
        use_two_node_special=use_two_node_special,
    )

    return torch.from_numpy(coordinates * _IGRAPH_LAYOUT_SCALE).to(
        dtype=output_dtype,
        device=output_device,
    )


def _igraph_mds_single_from_distances(
    distances: np.ndarray,
    use_two_node_special: bool,
) -> np.ndarray:
    """Run igraph's single-component classical MDS kernel.

    Parameters
    ----------
    distances : numpy.ndarray
        Dense graph-distance matrix with shape ``[N, N]`` for one component.
    use_two_node_special : bool
        Whether to use igraph's two-node raw layout ``[[0, 0], [1, 1]]``.

    Returns
    -------
    numpy.ndarray
        Raw unscaled coordinates with shape ``[N, 2]``.
    """
    num_nodes = int(distances.shape[0])
    if num_nodes == 0:
        return np.zeros((0, 2), dtype=np.float64)
    if num_nodes == 1:
        return np.zeros((1, 2), dtype=np.float64)
    if num_nodes == 2 and use_two_node_special:
        return np.array([[0.0, 0.0], [1.0, 1.0]], dtype=np.float64)

    gram = np.array(distances, dtype=np.float64, order="F", copy=True)
    for column in range(num_nodes):
        for row in range(num_nodes):
            gram[row, column] *= gram[row, column]

    row_means = gram.mean(axis=1)
    grand_mean = float(row_means.sum() / float(num_nodes))
    gram += grand_mean
    for column in range(num_nodes):
        for row in range(num_nodes):
            gram[row, column] -= row_means[row] + row_means[column]
            gram[row, column] *= _OGDF_CENTERING_FACTOR

    eigenvalues, eigenvectors = scipy.linalg.eigh(
        gram,
        subset_by_index=(num_nodes - 2, num_nodes - 1),
        driver="evr",
        lower=True,
        check_finite=False,
    )

    coordinates = np.zeros((num_nodes, 2), dtype=np.float64)
    selected_count = int(eigenvalues.shape[0])
    for eigen_index in range(selected_count):
        output_column = selected_count - 1 - eigen_index
        coordinates[:, output_column] = (
            math.sqrt(abs(float(eigenvalues[eigen_index]))) * eigenvectors[:, eigen_index]
        )

    return coordinates


def _weak_components_igraph_order(edge_index: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Find weak components in igraph's first-unseen-vertex order.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[int]]
        Components as vertex-index lists. Component discovery follows the
        first unseen vertex, and each component is sorted to mirror igraph's
        subcomponent vertex vector on benchmark fixtures.
    """
    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    for edge_pos in range(int(edges.shape[1])):
        source = int(edges[0, edge_pos].item())
        target = int(edges[1, edge_pos].item())
        if source == target:
            continue
        adjacency[source].append(target)
        adjacency[target].append(source)

    seen = [False] * num_nodes
    components: list[list[int]] = []
    for start in range(num_nodes):
        if seen[start]:
            continue
        seen[start] = True
        queue = [start]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            for neighbor in adjacency[node]:
                if seen[neighbor]:
                    continue
                seen[neighbor] = True
                queue.append(neighbor)
        components.append(sorted(queue))
    return components


def _layout_igraph_disconnected_mds(
    distances: np.ndarray,
    components: list[list[int]],
    seed: int,
) -> np.ndarray:
    """Lay out disconnected components and merge them with igraph DLA.

    Parameters
    ----------
    distances : numpy.ndarray
        Whole-graph finite distance matrix with shape ``[N, N]``.
    components : list[list[int]]
        Weak components in igraph discovery order.
    seed : int
        Seed for the Python RNG stream used by the DLA walk.

    Returns
    -------
    numpy.ndarray
        Raw unscaled coordinates with shape ``[N, 2]`` in original vertex
        order.
    """
    layouts: list[np.ndarray] = []
    vertex_order = [0] * int(distances.shape[0])
    processed_vertex_count = 0
    for component in components:
        indices = np.array(component, dtype=np.int64)
        sub_distances = np.array(distances[np.ix_(indices, indices)], dtype=np.float64, copy=True)
        layouts.append(
            _igraph_mds_single_from_distances(
                distances=sub_distances,
                use_two_node_special=True,
            )
        )
        for vertex in component:
            vertex_order[vertex] = processed_vertex_count
            processed_vertex_count += 1

    merged = _igraph_layout_merge_dla(layouts=layouts, seed=seed)
    result = np.zeros_like(merged)
    for vertex, merged_row in enumerate(vertex_order):
        result[vertex, :] = merged[merged_row, :]
    return result


def _igraph_layout_merge_dla(layouts: list[np.ndarray], seed: int) -> np.ndarray:
    """Merge component layouts using igraph's DLA component packer.

    Parameters
    ----------
    layouts : list[numpy.ndarray]
        Per-component raw coordinates, each shaped ``[Ni, 2]``.
    seed : int
        Seed for ``random.Random``. The benchmark reference routes igraph's
        ``RNG_UNIF`` through Python's seeded RNG, so this mirrors call order.

    Returns
    -------
    numpy.ndarray
        Merged raw coordinates in concatenated component-row order.
    """
    coords_len = len(layouts)
    if coords_len == 0:
        return np.zeros((0, 2), dtype=np.float64)

    sizes = [float(layout.shape[0]) for layout in layouts]
    x_positions = [0.0] * coords_len
    y_positions = [0.0] * coords_len
    radii = [math.pow(size, 0.75) for size in sizes]
    centers_x = [0.0] * coords_len
    centers_y = [0.0] * coords_len
    native_radii = [0.0] * coords_len

    area = 0.0
    all_nodes = 0
    for index, layout in enumerate(layouts):
        all_nodes += int(layout.shape[0])
        area += radii[index] * radii[index]
        centers_x[index], centers_y[index], native_radii[index] = _igraph_layout_sphere_2d(layout)

    order = sorted(range(coords_len), key=lambda index: (-sizes[index], index))
    minx = miny = -math.sqrt(_IGRAPH_DLA_AREA_FACTOR * area)
    maxx = maxy = math.sqrt(_IGRAPH_DLA_AREA_FACTOR * area)
    grid = _IgraphMergeGrid(
        minx=minx,
        maxx=maxx,
        stepsx=_IGRAPH_DLA_GRID_STEPS,
        miny=miny,
        maxy=maxy,
        stepsy=_IGRAPH_DLA_GRID_STEPS,
    )

    first_component = order[0]
    grid.place_sphere(0.0, 0.0, radii[first_component], first_component)
    rng = random.Random(seed)
    for component in order[1:]:
        x_positions[component], y_positions[component] = _igraph_layout_merge_dla_walk(
            grid=grid,
            radius=radii[component],
            center_x=0.0,
            center_y=0.0,
            start_radius=maxx,
            kill_radius=maxx + _IGRAPH_DLA_KILL_RADIUS_PAD,
            rng=rng,
        )
        grid.place_sphere(
            x_positions[component],
            y_positions[component],
            radii[component],
            component,
        )

    result = np.zeros((all_nodes, 2), dtype=np.float64)
    result_pos = 0
    for index, layout in enumerate(layouts):
        scale = 1.0 if native_radii[index] == 0.0 else radii[index] / native_radii[index]
        for row in range(int(layout.shape[0])):
            result[result_pos, 0] = scale * (layout[row, 0] - centers_x[index])
            result[result_pos, 1] = scale * (layout[row, 1] - centers_y[index])
            result[result_pos, 0] += x_positions[index]
            result[result_pos, 1] += y_positions[index]
            result_pos += 1
    return result


def _igraph_layout_sphere_2d(coords: np.ndarray) -> tuple[float, float, float]:
    """Compute igraph's 2D bounding sphere from a coordinate bounding box.

    Parameters
    ----------
    coords : numpy.ndarray
        Coordinate matrix with shape ``[N, 2]``.

    Returns
    -------
    tuple[float, float, float]
        Center x, center y, and half-diagonal radius.
    """
    xmin = xmax = float(coords[0, 0])
    ymin = ymax = float(coords[0, 1])
    for row in range(1, int(coords.shape[0])):
        x_coord = float(coords[row, 0])
        y_coord = float(coords[row, 1])
        if x_coord < xmin:
            xmin = x_coord
        elif x_coord > xmax:
            xmax = x_coord
        if y_coord < ymin:
            ymin = y_coord
        elif y_coord > ymax:
            ymax = y_coord
    center_x = (xmin + xmax) / 2.0
    center_y = (ymin + ymax) / 2.0
    radius = math.sqrt((xmax - xmin) * (xmax - xmin) + (ymax - ymin) * (ymax - ymin)) / 2.0
    return center_x, center_y, radius


def _rng_unif(rng: random.Random, low: float, high: float) -> float:
    """Draw a uniform value matching igraph's ``RNG_UNIF`` call sites.

    Parameters
    ----------
    rng : random.Random
        Seeded Python RNG used by the benchmark igraph adapter.
    low : float
        Inclusive lower bound.
    high : float
        Exclusive upper bound.

    Returns
    -------
    float
        Uniform draw in ``[low, high)``.
    """
    return low + (high - low) * rng.random()


def _igraph_layout_merge_dla_walk(
    grid: "_IgraphMergeGrid",
    radius: float,
    center_x: float,
    center_y: float,
    start_radius: float,
    kill_radius: float,
    rng: random.Random,
) -> tuple[float, float]:
    """Run igraph's random walk for one component sphere.

    Parameters
    ----------
    grid : _IgraphMergeGrid
        Occupancy grid containing already placed component spheres.
    radius : float
        Radius of the component sphere to place.
    center_x : float
        X coordinate of the DLA center.
    center_y : float
        Y coordinate of the DLA center.
    start_radius : float
        Maximum radius for new particle starts.
    kill_radius : float
        Radius at which a walk attempt is abandoned.
    rng : random.Random
        Seeded RNG for ``RNG_UNIF`` calls.

    Returns
    -------
    tuple[float, float]
        Last non-colliding coordinate adjacent to an occupied sphere.

    Raises
    ------
    RuntimeError
        If development guardrails are exceeded.
    """
    sphere = -1
    x_coord = 0.0
    y_coord = 0.0
    total_steps = 0
    restarts = 0
    rng_random = rng.random
    two_pi = 2.0 * math.pi
    start_low = 0.5 * start_radius
    start_width = start_radius - start_low
    walk_step_high = start_radius / _IGRAPH_DLA_WALK_STEP_DIVISOR
    get_sphere = grid.get_sphere
    cos = math.cos
    sin = math.sin
    hypot = math.hypot
    while sphere < 0:
        restarts += 1
        if restarts > _IGRAPH_DLA_MAX_RESTARTS:
            raise RuntimeError("igraph DLA restart guardrail exceeded.")
        while True:
            angle = two_pi * rng_random()
            length = start_low + start_width * rng_random()
            x_coord = center_x + length * cos(angle)
            y_coord = center_y + length * sin(angle)
            sphere = get_sphere(x_coord, y_coord, radius)
            if sphere < 0:
                break

        while sphere < 0 and hypot(x_coord - center_x, y_coord - center_y) < kill_radius:
            total_steps += 1
            if total_steps > _IGRAPH_DLA_MAX_TOTAL_STEPS:
                raise RuntimeError("igraph DLA step guardrail exceeded.")
            angle = two_pi * rng_random()
            length = walk_step_high * rng_random()
            next_x = x_coord + length * cos(angle)
            next_y = y_coord + length * sin(angle)
            sphere = get_sphere(next_x, next_y, radius)
            if sphere < 0:
                x_coord = next_x
                y_coord = next_y
    return x_coord, y_coord


class _IgraphMergeGrid:
    """Scalar port of igraph's rasterized merge occupancy grid."""

    def __init__(
        self,
        minx: float,
        maxx: float,
        stepsx: int,
        miny: float,
        maxy: float,
        stepsy: int,
    ) -> None:
        """Initialize an igraph-style merge grid.

        Parameters
        ----------
        minx : float
            Minimum x coordinate covered by the grid.
        maxx : float
            Maximum x coordinate covered by the grid.
        stepsx : int
            Number of x cells.
        miny : float
            Minimum y coordinate covered by the grid.
        maxy : float
            Maximum y coordinate covered by the grid.
        stepsy : int
            Number of y cells.

        Returns
        -------
        None
            Initializes the occupancy array.
        """
        self.minx = minx
        self.maxx = maxx
        self.stepsx = stepsx
        self.deltax = (maxx - minx) / float(stepsx)
        self.miny = miny
        self.maxy = maxy
        self.stepsy = stepsy
        self.deltay = (maxy - miny) / float(stepsy)
        self.data = np.zeros((stepsx, stepsy), dtype=np.int64)
        self.occupied_x = np.empty(0, dtype=np.int64)
        self.occupied_y = np.empty(0, dtype=np.int64)
        self.cell_x_coords = minx + np.arange(stepsx, dtype=np.float64) * self.deltax
        self.cell_y_coords = miny + np.arange(stepsy, dtype=np.float64) * self.deltay

    def _mat_index(self, x_index: int, y_index: int) -> int:
        """Return igraph's flattened ``MAT(i, j)`` storage index.

        Parameters
        ----------
        x_index : int
            X grid index.
        y_index : int
            Y grid index.

        Returns
        -------
        int
            Flat occupancy-array index.
        """
        return self.stepsy * y_index + x_index

    def _get_mat(self, x_index: int, y_index: int) -> int:
        """Read one igraph occupancy cell.

        Parameters
        ----------
        x_index : int
            X grid index.
        y_index : int
            Y grid index.

        Returns
        -------
        int
            Stored component id plus one, or zero for empty.
        """
        return int(self.data[x_index, y_index])

    def _set_mat(self, x_index: int, y_index: int, value: int) -> None:
        """Write one igraph occupancy cell.

        Parameters
        ----------
        x_index : int
            X grid index.
        y_index : int
            Y grid index.
        value : int
            Stored component id plus one.

        Returns
        -------
        None
            Updates the occupancy array in place.
        """
        self.data[x_index, y_index] = value

    def which(self, x_coord: float, y_coord: float) -> tuple[int, int]:
        """Map coordinates to igraph merge-grid cell indices.

        Parameters
        ----------
        x_coord : float
            X coordinate.
        y_coord : float
            Y coordinate.

        Returns
        -------
        tuple[int, int]
            X and y grid indices.
        """
        if x_coord <= self.minx:
            x_index = 0
        elif x_coord >= self.maxx:
            x_index = self.stepsx - 1
        else:
            x_index = math.floor((x_coord - self.minx) / self.deltax)

        if y_coord <= self.miny:
            y_index = 0
        elif y_coord >= self.maxy:
            y_index = self.stepsy - 1
        else:
            y_index = math.floor((y_coord - self.miny) / self.deltay)
        return int(x_index), int(y_index)

    def _distance_to_cell(
        self,
        x_coord: float,
        y_coord: float,
        cell_x: float,
        cell_y: float,
    ) -> float:
        """Compute igraph's cell-center distance expression.

        Parameters
        ----------
        x_coord : float
            Query x coordinate.
        y_coord : float
            Query y coordinate.
        cell_x : float
            Cell x coordinate.
        cell_y : float
            Cell y coordinate.

        Returns
        -------
        float
            Euclidean distance.
        """
        return math.sqrt(
            (x_coord - cell_x) * (x_coord - cell_x) + (y_coord - cell_y) * (y_coord - cell_y)
        )

    def place_sphere(
        self,
        x_coord: float,
        y_coord: float,
        radius: float,
        component_id: int,
    ) -> None:
        """Rasterize a placed component sphere into the occupancy grid.

        Parameters
        ----------
        x_coord : float
            Sphere center x coordinate.
        y_coord : float
            Sphere center y coordinate.
        radius : float
            Sphere radius.
        component_id : int
            Component index stored as ``id + 1`` in the grid.

        Returns
        -------
        None
            Mutates grid occupancy in place.
        """
        center_x, center_y = self.which(x_coord, y_coord)
        value = component_id + 1
        self._set_mat(center_x, center_y, value)

        i = 0
        while (
            center_x + i < self.stepsx
            and self._place_distance(x_coord, y_coord, center_x, center_y, i, 0, 1, 1) < radius
        ):
            j = 0
            while (
                center_y + j < self.stepsy
                and self._place_distance(x_coord, y_coord, center_x, center_y, i, j, 1, 1) < radius
            ):
                self._set_mat(center_x + i, center_y + j, value)
                j += 1
            i += 1

        i = 0
        while (
            center_x + i < self.stepsx
            and self._place_distance(x_coord, y_coord, center_x, center_y, i, 0, 1, -1) < radius
        ):
            j = 1
            while (
                center_y - j > 0
                and self._place_distance(x_coord, y_coord, center_x, center_y, i, j, 1, -1) < radius
            ):
                self._set_mat(center_x + i, center_y - j, value)
                j += 1
            i += 1

        i = 1
        while (
            center_x - i > 0
            and self._place_distance(x_coord, y_coord, center_x, center_y, i, 0, -1, 1) < radius
        ):
            j = 0
            while (
                center_y + j < self.stepsy
                and self._place_distance(x_coord, y_coord, center_x, center_y, i, j, -1, 1) < radius
            ):
                self._set_mat(center_x - i, center_y + j, value)
                j += 1
            i += 1

        i = 1
        while (
            center_x - i > 0
            and self._place_distance(x_coord, y_coord, center_x, center_y, i, 0, -1, -1) < radius
        ):
            j = 1
            while (
                center_y - j > 0
                and self._place_distance(x_coord, y_coord, center_x, center_y, i, j, -1, -1)
                < radius
            ):
                self._set_mat(center_x - i, center_y - j, value)
                j += 1
            i += 1
        self._refresh_occupied_cells()

    def _place_distance(
        self,
        x_coord: float,
        y_coord: float,
        center_x: int,
        center_y: int,
        offset_x: int,
        offset_y: int,
        sign_x: int,
        sign_y: int,
    ) -> float:
        """Compute one quadrant distance from ``place_sphere``.

        Parameters
        ----------
        x_coord : float
            Sphere center x coordinate.
        y_coord : float
            Sphere center y coordinate.
        center_x : int
            Grid center x index.
        center_y : int
            Grid center y index.
        offset_x : int
            Non-negative x offset.
        offset_y : int
            Non-negative y offset.
        sign_x : int
            X quadrant sign, either ``1`` or ``-1``.
        sign_y : int
            Y quadrant sign, either ``1`` or ``-1``.

        Returns
        -------
        float
            Distance to the igraph cell coordinate used by that quadrant.
        """
        cell_x_index = center_x + offset_x if sign_x > 0 else center_x - offset_x + 1
        cell_y_index = center_y + offset_y if sign_y > 0 else center_y - offset_y + 1
        cell_x = self.minx + cell_x_index * self.deltax
        cell_y = self.miny + cell_y_index * self.deltay
        return self._distance_to_cell(x_coord, y_coord, cell_x, cell_y)

    def get_sphere(self, x_coord: float, y_coord: float, radius: float) -> int:
        """Find any occupied sphere colliding with a candidate sphere.

        Parameters
        ----------
        x_coord : float
            Candidate center x coordinate.
        y_coord : float
            Candidate center y coordinate.
        radius : float
            Candidate radius.

        Returns
        -------
        int
            Component id of the colliding sphere, or ``-1`` when none is found.
        """
        if (
            x_coord - radius <= self.minx
            or x_coord + radius >= self.maxx
            or y_coord - radius <= self.miny
            or y_coord + radius >= self.maxy
        ):
            return -1

        if self.occupied_x.size == 0:
            return -1

        x_start = max(0, math.floor((x_coord - radius - self.minx) / self.deltax) + 1)
        x_stop = min(self.stepsx, math.ceil((x_coord + radius - self.minx) / self.deltax))
        y_start = max(0, math.floor((y_coord - radius - self.miny) / self.deltay) + 1)
        y_stop = min(self.stepsy, math.ceil((y_coord + radius - self.miny) / self.deltay))
        if x_start >= x_stop or y_start >= y_stop:
            return -1

        local_occupied_x, local_occupied_y = np.nonzero(self.data[x_start:x_stop, y_start:y_stop])
        if local_occupied_x.size == 0:
            return -1

        hit_x_indices = x_start + local_occupied_x
        hit_y_indices = y_start + local_occupied_y
        delta_x = x_coord - self.cell_x_coords[hit_x_indices]
        delta_y = y_coord - self.cell_y_coords[hit_y_indices]
        hits = delta_x * delta_x + delta_y * delta_y < radius * radius
        if not bool(hits.any()):
            return -1
        hit_x = hit_x_indices[hits][0]
        hit_y = hit_y_indices[hits][0]
        return self._get_mat(int(hit_x), int(hit_y)) - 1

    def _refresh_occupied_cells(self) -> None:
        """Refresh cached occupied-cell coordinates after rasterization.

        Parameters
        ----------
        None
            This method reads the current occupancy array.

        Returns
        -------
        None
            Updates cached occupied x and y index arrays.
        """
        self.occupied_x, self.occupied_y = np.nonzero(self.data)


def _layout_ogdf_classical_mds(
    edge_index: torch.Tensor,
    num_nodes: int,
    fidelity_dtype: torch.dtype,
) -> torch.Tensor:
    """Run OGDF's all-pivots PivotMDS implementation.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes ``N``.
    fidelity_dtype : torch.dtype
        Output dtype for fidelity comparisons.

    Returns
    -------
    torch.Tensor
        Raw OGDF-compatible coordinates with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If the graph is disconnected, matching OGDF PivotMDS' connected-graph
        precondition.
    RuntimeError
        If OGDF's power iteration diverges numerically.
    """
    if num_nodes == 0:
        return torch.zeros((0, 2), dtype=fidelity_dtype)
    if num_nodes == 1:
        return torch.zeros((1, 2), dtype=fidelity_dtype)

    adjacency: list[list[int]] = [[] for _ in range(num_nodes)]
    edges = edge_index.detach().to(device="cpu", dtype=torch.long)
    for edge_pos in range(int(edges.shape[1])):
        source = int(edges[0, edge_pos].item())
        target = int(edges[1, edge_pos].item())
        if source == target:
            continue
        adjacency[source].append(target)
        adjacency[target].append(source)

    simple_neighbors = [list(dict.fromkeys(neighbors)) for neighbors in adjacency]
    endpoints = [node for node, neighbors in enumerate(simple_neighbors) if len(neighbors) == 1]
    if len(endpoints) == 2 and not any(
        len(neighbors) > 2 or len(neighbors) == 0 for neighbors in simple_neighbors
    ):
        positions = torch.zeros((num_nodes, 2), dtype=fidelity_dtype)
        previous = -1
        current = endpoints[0]
        for path_index in range(num_nodes):
            positions[current, 0] = float(path_index) * _OGDF_EDGE_COST
            next_nodes = [
                neighbor for neighbor in simple_neighbors[current] if neighbor != previous
            ]
            previous, current = current, next_nodes[0] if next_nodes else -1
            if current == -1 and path_index == num_nodes - 1:
                return positions

    visited = [False] * num_nodes
    queue = [0]
    visited[0] = True
    head = 0
    while head < len(queue):
        node = queue[head]
        head += 1
        for neighbor in adjacency[node]:
            if visited[neighbor]:
                continue
            visited[neighbor] = True
            queue.append(neighbor)
    if not all(visited):
        raise ValueError("OGDF classical MDS fidelity requires a connected graph.")

    pivot_matrix: list[list[float]] = []
    min_distances = [math.inf] * num_nodes
    pivot_node = 0
    for _ in range(num_nodes):
        distances = [math.inf] * num_nodes
        distances[pivot_node] = 0.0
        queue = [pivot_node]
        head = 0
        while head < len(queue):
            node = queue[head]
            head += 1
            next_distance = distances[node] + _OGDF_EDGE_COST
            for neighbor in adjacency[node]:
                if math.isfinite(distances[neighbor]):
                    continue
                distances[neighbor] = next_distance
                queue.append(neighbor)
        pivot_matrix.append(distances)
        min_distances[pivot_node] = 0.0
        for node in range(num_nodes):
            min_distances[node] = min(min_distances[node], distances[node])
            if min_distances[node] > min_distances[pivot_node]:
                pivot_node = node

    normalization_factor = 0.0
    col_normalization = [0.0] * num_nodes
    for pivot_idx in range(num_nodes):
        row_col_normalizer = 0.0
        for node_idx in range(num_nodes):
            row_col_normalizer += (
                pivot_matrix[pivot_idx][node_idx] * pivot_matrix[pivot_idx][node_idx]
            )
        normalization_factor += row_col_normalizer
        col_normalization[pivot_idx] = row_col_normalizer / float(num_nodes)
    normalization_factor /= float(num_nodes * num_nodes)
    for node_idx in range(num_nodes):
        row_col_normalizer = 0.0
        for pivot_idx in range(num_nodes):
            square = pivot_matrix[pivot_idx][node_idx] * pivot_matrix[pivot_idx][node_idx]
            pivot_matrix[pivot_idx][node_idx] = (
                square + normalization_factor - col_normalization[pivot_idx]
            )
            row_col_normalizer += square
        row_col_normalizer /= float(num_nodes)
        for pivot_idx in range(num_nodes):
            pivot_matrix[pivot_idx][node_idx] = _OGDF_CENTERING_FACTOR * (
                pivot_matrix[pivot_idx][node_idx] - row_col_normalizer
            )

    product_matrix = [[0.0 for _ in range(num_nodes)] for _ in range(num_nodes)]
    for row_idx in range(num_nodes):
        for col_idx in range(row_idx + 1):
            total = 0.0
            for node_idx in range(num_nodes):
                total += pivot_matrix[row_idx][node_idx] * pivot_matrix[col_idx][node_idx]
            product_matrix[row_idx][col_idx] = total
            product_matrix[col_idx][row_idx] = total

    libc = ctypes.CDLL(None)
    libc.srand(0)
    eigenvectors = [
        [float(libc.rand()) / float(_LIBC_RAND_MAX) for _ in range(num_nodes)] for _ in range(2)
    ]
    eigenvalues = [0.0, 0.0]
    for dim_idx in range(2):
        norm = math.sqrt(sum(value * value for value in eigenvectors[dim_idx]))
        eigenvalues[dim_idx] = norm
        if norm != 0.0:
            eigenvectors[dim_idx] = [value / norm for value in eigenvectors[dim_idx]]

    convergence = 0.0
    while convergence < _OGDF_POWER_EPSILON:
        if math.isnan(convergence) or math.isinf(convergence):
            raise RuntimeError("OGDF classical MDS power iteration diverged.")
        previous = [row.copy() for row in eigenvectors]
        eigenvectors = [[0.0 for _ in range(num_nodes)] for _ in range(2)]
        for dim_idx in range(2):
            for row_idx in range(num_nodes):
                for col_idx in range(num_nodes):
                    eigenvectors[dim_idx][col_idx] += (
                        product_matrix[row_idx][col_idx] * previous[dim_idx][row_idx]
                    )
        denominator = sum(value * value for value in eigenvectors[0])
        factor = (
            sum(eigenvectors[0][index] * eigenvectors[1][index] for index in range(num_nodes))
            / denominator
        )
        for node_idx in range(num_nodes):
            eigenvectors[1][node_idx] -= factor * eigenvectors[0][node_idx]
        for dim_idx in range(2):
            norm = math.sqrt(sum(value * value for value in eigenvectors[dim_idx]))
            eigenvalues[dim_idx] = norm
            if norm != 0.0:
                eigenvectors[dim_idx] = [value / norm for value in eigenvectors[dim_idx]]
        convergence = 1.0
        for dim_idx in range(2):
            product = sum(
                eigenvectors[dim_idx][index] * previous[dim_idx][index]
                for index in range(num_nodes)
            )
            convergence = min(convergence, abs(product))

    coordinate_rows = [[0.0 for _ in range(num_nodes)] for _ in range(2)]
    for dim_idx in range(2):
        eigenvalues[dim_idx] = math.sqrt(eigenvalues[dim_idx])
        for node_idx in range(num_nodes):
            for pivot_idx in range(num_nodes):
                coordinate_rows[dim_idx][node_idx] += (
                    pivot_matrix[pivot_idx][node_idx] * eigenvectors[dim_idx][pivot_idx]
                )
    for dim_idx in range(2):
        norm = math.sqrt(sum(value * value for value in coordinate_rows[dim_idx]))
        if norm != 0.0:
            coordinate_rows[dim_idx] = [value / norm for value in coordinate_rows[dim_idx]]
    for dim_idx in range(2):
        eigenvalues[dim_idx] = math.sqrt(eigenvalues[dim_idx])
        for node_idx in range(num_nodes):
            coordinate_rows[dim_idx][node_idx] *= eigenvalues[dim_idx]

    return torch.tensor(
        [
            [coordinate_rows[0][node_idx], coordinate_rows[1][node_idx]]
            for node_idx in range(num_nodes)
        ],
        dtype=fidelity_dtype,
    )


__all__ = ["build_classical_mds_pipeline", "layout_classical_mds_pipeline"]
