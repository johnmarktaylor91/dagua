"""Ordering operations for layered graph layouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Sequence, Tuple

import torch

from dagua.layout.init_placement import _transpose_heuristic
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


def _target_device(problem: LayoutProblem, state: SolveState) -> torch.device:
    """Resolve the device used for persisted ordering tensors.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.device
        Device for the updated ``state.ordering`` tensor.
    """
    if state.ordering is not None:
        return state.ordering.device
    if state.layers is not None:
        return state.layers.device
    return problem.edge_index.device


def _validate_layers(layers: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Return validated CPU layer assignments.

    Parameters
    ----------
    layers : torch.Tensor | None
        Candidate layer tensor with shape ``[N]``.
    num_nodes : int
        Expected node count.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[N]``.

    Raises
    ------
    ValueError
        If ``layers`` is missing or has an invalid shape.
    """
    if layers is None:
        raise ValueError("state.layers must be populated before ordering ops run")
    if layers.ndim != 1 or layers.shape[0] != num_nodes:
        raise ValueError(f"state.layers must have shape [{num_nodes}]")

    layers_cpu = layers.detach().to(device="cpu", dtype=torch.long)
    if layers_cpu.numel() > 0 and int(layers_cpu.min().item()) < 0:
        raise ValueError("state.layers cannot contain negative indices")
    return layers_cpu


def _validate_ordering(ordering: Optional[torch.Tensor], num_nodes: int) -> torch.Tensor:
    """Return validated CPU ordering indices.

    Parameters
    ----------
    ordering : torch.Tensor | None
        Candidate ordering tensor with shape ``[N]``.
    num_nodes : int
        Expected node count.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[N]``.

    Raises
    ------
    ValueError
        If ``ordering`` is missing or has an invalid shape.
    """
    if ordering is None:
        raise ValueError("state.ordering must be populated before this op runs")
    if ordering.ndim != 1 or ordering.shape[0] != num_nodes:
        raise ValueError(f"state.ordering must have shape [{num_nodes}]")
    return ordering.detach().to(device="cpu", dtype=torch.long)


def _validate_edge_index(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Return a validated CPU edge list.

    Parameters
    ----------
    edge_index : torch.Tensor
        Input edge tensor expected to have shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[2, E]``.

    Raises
    ------
    ValueError
        If the edge tensor shape or node references are invalid.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("problem.edge_index must have shape [2, E]")
    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() == 0:
        return edge_index_cpu
    if int(edge_index_cpu.min().item()) < 0 or int(edge_index_cpu.max().item()) >= num_nodes:
        raise ValueError("problem.edge_index references a node outside the valid range")
    return edge_index_cpu


def _adjacency_as_weighted_lists(adjacency: Any, num_nodes: int) -> List[List[Tuple[int, float]]]:
    """Normalize multiple adjacency formats into weighted Python lists.

    Parameters
    ----------
    adjacency : Any
        Cached adjacency from ``BuildAdjacency``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    list[list[tuple[int, float]]]
        Weighted adjacency lists indexed by node id.

    Raises
    ------
    ValueError
        If the adjacency format is unsupported for ordering sweeps.
    """
    if adjacency is None:
        raise ValueError("state.adjacency must be populated before sweep ordering ops run")

    weighted: List[List[Tuple[int, float]]] = [[] for _ in range(num_nodes)]

    if isinstance(adjacency, list):
        if len(adjacency) != num_nodes:
            raise ValueError(f"state.adjacency must contain {num_nodes} rows")
        for node, neighbors in enumerate(adjacency):
            for entry in neighbors:
                if isinstance(entry, tuple):
                    if len(entry) != 2:
                        raise ValueError(
                            "adjacency tuple entries must have shape (neighbor, weight)"
                        )
                    neighbor, weight = entry
                else:
                    neighbor, weight = entry, 1.0
                weighted[node].append((int(neighbor), float(weight)))
        return weighted

    if isinstance(adjacency, torch.Tensor):
        if adjacency.ndim != 2 or adjacency.shape != (num_nodes, num_nodes):
            raise ValueError(f"dense adjacency must have shape [{num_nodes}, {num_nodes}]")
        dense_cpu = adjacency.detach().to(device="cpu", dtype=torch.float32)
        for node in range(num_nodes):
            for neighbor in range(num_nodes):
                if node == neighbor:
                    continue
                value = float(dense_cpu[node, neighbor].item())
                if not torch.isfinite(dense_cpu[node, neighbor]) or value == 0.0:
                    continue
                weighted[node].append((neighbor, value))
        return weighted

    if isinstance(adjacency, dict):
        indptr = adjacency.get("indptr")
        indices = adjacency.get("indices")
        data = adjacency.get("data", adjacency.get("weights"))
        if indptr is None or indices is None:
            raise ValueError("CSR adjacency must include indptr and indices")

        indptr_cpu = indptr.detach().to(device="cpu", dtype=torch.long)
        indices_cpu = indices.detach().to(device="cpu", dtype=torch.long)
        data_cpu = None
        if data is not None:
            data_cpu = data.detach().to(device="cpu", dtype=torch.float32)

        if indptr_cpu.numel() != num_nodes + 1:
            raise ValueError(f"CSR indptr must have length {num_nodes + 1}")
        for node in range(num_nodes):
            start = int(indptr_cpu[node].item())
            stop = int(indptr_cpu[node + 1].item())
            for offset in range(start, stop):
                neighbor = int(indices_cpu[offset].item())
                weight = float(data_cpu[offset].item()) if data_cpu is not None else 1.0
                weighted[node].append((neighbor, weight))
        return weighted

    raise ValueError("Unsupported adjacency format for ordering ops")


def _initial_ordered_layers(
    layers: torch.Tensor,
    ordering: Optional[torch.Tensor] = None,
) -> List[List[int]]:
    """Build ordered layer lists from per-node layer and ordering tensors.

    Parameters
    ----------
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    ordering : torch.Tensor | None, optional
        CPU in-layer ordering values with shape ``[N]``.

    Returns
    -------
    list[list[int]]
        Node ids grouped by layer and sorted left-to-right.
    """
    if layers.numel() == 0:
        return []

    num_layers = int(layers.max().item()) + 1
    ordered_layers: List[List[int]] = [[] for _ in range(num_layers)]
    for node in range(layers.shape[0]):
        ordered_layers[int(layers[node].item())].append(node)

    for layer_nodes in ordered_layers:
        if ordering is None:
            layer_nodes.sort()
        else:
            layer_nodes.sort(key=lambda node_idx: (int(ordering[node_idx].item()), node_idx))
    return ordered_layers


def _ordering_from_layers(
    ordered_layers: Sequence[Sequence[int]],
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert ordered per-layer node lists into a per-node rank tensor.

    Parameters
    ----------
    ordered_layers : sequence[sequence[int]]
        Nodes grouped by layer in left-to-right order.
    num_nodes : int
        Number of graph nodes.
    device : torch.device
        Target device for the result tensor.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``[N]``.
    """
    ordering = torch.zeros((num_nodes,), dtype=torch.long)
    for layer_nodes in ordered_layers:
        for position, node in enumerate(layer_nodes):
            ordering[node] = position
    return ordering.to(device=device)


def _node_order_map(layers: Sequence[Sequence[int]]) -> Dict[int, float]:
    """Map each node id to its current in-layer position.

    Parameters
    ----------
    layers : sequence[sequence[int]]
        Ordered nodes per layer.

    Returns
    -------
    dict[int, float]
        Current left-to-right rank for every node.
    """
    order_map: Dict[int, float] = {}
    for layer_nodes in layers:
        for position, node in enumerate(layer_nodes):
            order_map[node] = float(position)
    return order_map


def _layered_neighbors_from_adjacency(
    adjacency: List[List[Tuple[int, float]]],
    layers: torch.Tensor,
    use_weights: bool,
) -> Tuple[List[List[int]], List[List[int]], List[Dict[int, float]], List[Dict[int, float]]]:
    """Split adjacency into lower-layer parents and upper-layer children.

    Parameters
    ----------
    adjacency : list[list[tuple[int, float]]]
        Weighted adjacency lists indexed by node id.
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    use_weights : bool
        Whether edge weights should affect barycenter computations.

    Returns
    -------
    tuple
        ``(parents, children, parent_weights, child_weights)`` indexed by node id.
    """
    num_nodes = layers.shape[0]
    parents: List[List[int]] = [[] for _ in range(num_nodes)]
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    parent_weights: List[Dict[int, float]] = [dict() for _ in range(num_nodes)]
    child_weights: List[Dict[int, float]] = [dict() for _ in range(num_nodes)]

    for node, neighbors in enumerate(adjacency):
        node_layer = int(layers[node].item())
        for neighbor, raw_weight in neighbors:
            if not 0 <= neighbor < num_nodes:
                continue
            neighbor_layer = int(layers[neighbor].item())
            weight = float(raw_weight) if use_weights else 1.0
            if neighbor_layer < node_layer:
                if neighbor not in parent_weights[node]:
                    parents[node].append(neighbor)
                    parent_weights[node][neighbor] = 0.0
                parent_weights[node][neighbor] += weight
            elif neighbor_layer > node_layer:
                if neighbor not in child_weights[node]:
                    children[node].append(neighbor)
                    child_weights[node][neighbor] = 0.0
                child_weights[node][neighbor] += weight
    return parents, children, parent_weights, child_weights


def _layered_neighbors_from_edges(
    edge_index: torch.Tensor,
    layers: torch.Tensor,
    num_nodes: int,
) -> Tuple[List[List[int]], List[List[int]]]:
    """Orient edges according to the layer assignment.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    tuple
        ``(parents, children)`` indexed by node id.
    """
    parents: List[List[int]] = [[] for _ in range(num_nodes)]
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    if edge_index.numel() == 0:
        return parents, children

    for source, target in edge_index.t().tolist():
        source_layer = int(layers[source].item())
        target_layer = int(layers[target].item())
        if source_layer == target_layer:
            continue
        if source_layer < target_layer:
            if source not in parents[target]:
                parents[target].append(source)
            if target not in children[source]:
                children[source].append(target)
            continue
        if target not in parents[source]:
            parents[source].append(target)
        if source not in children[target]:
            children[target].append(source)
    return parents, children


def _barycenter_value(
    node: int,
    neighbors_by_node: Sequence[Sequence[int]],
    neighbor_weights_by_node: Sequence[Dict[int, float]],
    order_map: Dict[int, float],
) -> float:
    """Compute the weighted barycenter score for one node.

    Parameters
    ----------
    node : int
        Node being ranked.
    neighbors_by_node : sequence[sequence[int]]
        Layer-compatible neighbors used as references.
    neighbor_weights_by_node : sequence[dict[int, float]]
        Optional neighbor weights aligned with ``neighbors_by_node``.
    order_map : dict[int, float]
        Current in-layer rank for each node.

    Returns
    -------
    float
        Weighted barycenter score, or the current node rank when isolated.
    """
    neighbors = neighbors_by_node[node]
    if not neighbors:
        return order_map.get(node, 0.0)

    weighted_sum = 0.0
    total_weight = 0.0
    for neighbor in neighbors:
        weight = neighbor_weights_by_node[node].get(neighbor, 1.0)
        weighted_sum += order_map[neighbor] * weight
        total_weight += weight
    if total_weight == 0.0:
        return order_map.get(node, 0.0)
    return weighted_sum / total_weight


def _median_value(
    node: int,
    neighbors_by_node: Sequence[Sequence[int]],
    order_map: Dict[int, float],
) -> float:
    """Compute the median neighbor rank for one node.

    Parameters
    ----------
    node : int
        Node being ranked.
    neighbors_by_node : sequence[sequence[int]]
        Layer-compatible neighbors used as references.
    order_map : dict[int, float]
        Current in-layer rank for each node.

    Returns
    -------
    float
        Median neighbor rank, or the current node rank when isolated.
    """
    neighbors = neighbors_by_node[node]
    if not neighbors:
        return order_map.get(node, 0.0)

    values = sorted(order_map[neighbor] for neighbor in neighbors)
    midpoint = len(values) // 2
    if len(values) % 2 == 1:
        return values[midpoint]
    return 0.5 * (values[midpoint - 1] + values[midpoint])


def _sort_layer_by_scores(
    layer_nodes: List[int],
    score_map: Dict[int, float],
) -> None:
    """Sort one layer by computed scores with stable tie handling.

    Parameters
    ----------
    layer_nodes : list[int]
        Nodes in one layer.
    score_map : dict[int, float]
        Ordering scores keyed by node id.

    Returns
    -------
    None
        The layer is sorted in place.
    """
    stable_positions = {node: index for index, node in enumerate(layer_nodes)}
    layer_nodes.sort(key=lambda node: (score_map[node], stable_positions[node], node))


def _deterministic_fiedler_vector(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Optional[torch.Tensor]:
    """Compute a deterministic Fiedler vector using ``torch.lobpcg``.

    Parameters
    ----------
    edge_index : torch.Tensor
        CPU edge tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    torch.Tensor | None
        CPU float tensor with shape ``[N]`` when the eigensolve succeeds,
        otherwise ``None``.
    """
    if num_nodes < 2 or edge_index.numel() == 0:
        return None

    all_sources = torch.cat([edge_index[0], edge_index[1]])
    all_targets = torch.cat([edge_index[1], edge_index[0]])
    degree = torch.zeros((num_nodes,), dtype=torch.float32)
    degree.scatter_add_(0, all_sources, torch.ones_like(all_sources, dtype=torch.float32))

    diagonal = torch.arange(num_nodes, dtype=torch.long)
    indices = torch.cat(
        [
            torch.stack([all_sources, all_targets], dim=0),
            torch.stack([diagonal, diagonal], dim=0),
        ],
        dim=1,
    )
    values = torch.cat(
        [
            -torch.ones((all_sources.shape[0],), dtype=torch.float32),
            degree,
        ]
    )
    laplacian = torch.sparse_coo_tensor(indices, values, (num_nodes, num_nodes)).coalesce()

    # Use a deterministic initial subspace so the op is reproducible without RNG state.
    baseline = torch.linspace(-1.0, 1.0, steps=num_nodes, dtype=torch.float32)
    harmonic = torch.cos(torch.arange(num_nodes, dtype=torch.float32))
    initial_guess = torch.stack([baseline, harmonic], dim=1)

    try:
        _, eigenvectors = torch.lobpcg(
            laplacian,
            k=2,
            X=initial_guess,
            largest=False,
            niter=32,
        )
    except Exception:
        return None
    return eigenvectors[:, 1].detach().to(device="cpu", dtype=torch.float32)


@dataclass(frozen=True)
class BarycenterSweepConfig:
    """Configuration for :class:`BarycenterSweep`.

    Parameters
    ----------
    passes : int, default=24
        Number of full sweep passes to perform.
    direction : str, default="both"
        Sweep direction: ``"down"``, ``"up"``, or ``"both"``.
    use_weights : bool, default=True
        Whether adjacency weights should influence barycenter scores.
    """

    passes: int = 24
    direction: str = "both"
    use_weights: bool = True


@dataclass(frozen=True)
class MedianSweepConfig:
    """Configuration for :class:`MedianSweep`.

    Parameters
    ----------
    passes : int, default=24
        Number of full sweep passes to perform.
    """

    passes: int = 24


@dataclass(frozen=True)
class TransposeHeuristicConfig:
    """Configuration for :class:`TransposeHeuristic`.

    Parameters
    ----------
    passes : int, default=8
        Maximum number of local swap passes.
    """

    passes: int = 8


@register_op
class BarycenterSweep(Op):
    """Order each layer by repeated weighted barycenter sweeps."""

    name = "barycenter_sweep"
    category = OpCategory.ORDERING
    reads = ("layers", "adjacency")
    writes = ("ordering",)
    requires = ("layers", "adjacency")

    def __init__(self, config: Optional[BarycenterSweepConfig] = None) -> None:
        """Store the sweep configuration.

        Parameters
        ----------
        config : BarycenterSweepConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BarycenterSweepConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run repeated barycenter sweeps and persist per-node ordering ranks.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ``layers`` and ``adjacency`` populated.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with ``ordering`` populated.

        Raises
        ------
        ValueError
            If the configured direction or input tensors are invalid.
        """
        del ctx

        if self.config.direction not in {"down", "up", "both"}:
            raise ValueError("BarycenterSweep direction must be 'down', 'up', or 'both'")

        layers_cpu = _validate_layers(state.layers, problem.num_nodes)
        weighted_adjacency = _adjacency_as_weighted_lists(state.adjacency, problem.num_nodes)
        parents, children, parent_weights, child_weights = _layered_neighbors_from_adjacency(
            adjacency=weighted_adjacency,
            layers=layers_cpu,
            use_weights=self.config.use_weights,
        )
        ordered_layers = _initial_ordered_layers(layers_cpu)

        for _ in range(max(self.config.passes, 0)):
            order_map = _node_order_map(ordered_layers)

            if self.config.direction in {"down", "both"}:
                for layer_index in range(1, len(ordered_layers)):
                    score_map = {
                        node: _barycenter_value(node, parents, parent_weights, order_map)
                        for node in ordered_layers[layer_index]
                    }
                    _sort_layer_by_scores(ordered_layers[layer_index], score_map)
                    order_map = _node_order_map(ordered_layers)

            if self.config.direction in {"up", "both"}:
                for layer_index in range(len(ordered_layers) - 2, -1, -1):
                    score_map = {
                        node: _barycenter_value(node, children, child_weights, order_map)
                        for node in ordered_layers[layer_index]
                    }
                    _sort_layer_by_scores(ordered_layers[layer_index], score_map)
                    order_map = _node_order_map(ordered_layers)

        state.ordering = _ordering_from_layers(
            ordered_layers=ordered_layers,
            num_nodes=problem.num_nodes,
            device=_target_device(problem, state),
        )
        return state


@register_op
class MedianSweep(Op):
    """Order each layer by repeated median-of-neighbors sweeps."""

    name = "median_sweep"
    category = OpCategory.ORDERING
    reads = ("layers", "adjacency")
    writes = ("ordering",)
    requires = ("layers", "adjacency")

    def __init__(self, config: Optional[MedianSweepConfig] = None) -> None:
        """Store the sweep configuration.

        Parameters
        ----------
        config : MedianSweepConfig | None, optional
            Optional op configuration.
        """
        self.config = config or MedianSweepConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run repeated median sweeps and persist per-node ordering ranks.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ``layers`` and ``adjacency`` populated.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with ``ordering`` populated.
        """
        del ctx

        layers_cpu = _validate_layers(state.layers, problem.num_nodes)
        weighted_adjacency = _adjacency_as_weighted_lists(state.adjacency, problem.num_nodes)
        parents, children, _, _ = _layered_neighbors_from_adjacency(
            adjacency=weighted_adjacency,
            layers=layers_cpu,
            use_weights=False,
        )
        ordered_layers = _initial_ordered_layers(layers_cpu)

        for _ in range(max(self.config.passes, 0)):
            order_map = _node_order_map(ordered_layers)
            for layer_index in range(1, len(ordered_layers)):
                score_map = {
                    node: _median_value(node, parents, order_map)
                    for node in ordered_layers[layer_index]
                }
                _sort_layer_by_scores(ordered_layers[layer_index], score_map)
                order_map = _node_order_map(ordered_layers)

            for layer_index in range(len(ordered_layers) - 2, -1, -1):
                score_map = {
                    node: _median_value(node, children, order_map)
                    for node in ordered_layers[layer_index]
                }
                _sort_layer_by_scores(ordered_layers[layer_index], score_map)
                order_map = _node_order_map(ordered_layers)

        state.ordering = _ordering_from_layers(
            ordered_layers=ordered_layers,
            num_nodes=problem.num_nodes,
            device=_target_device(problem, state),
        )
        return state


@register_op
class TransposeHeuristic(Op):
    """Swap adjacent same-layer nodes when the swap reduces local crossings."""

    name = "transpose_heuristic"
    category = OpCategory.ORDERING
    reads = ("layers", "ordering")
    writes = ("ordering",)
    requires = ("layers", "ordering")

    def __init__(self, config: Optional[TransposeHeuristicConfig] = None) -> None:
        """Store the transpose configuration.

        Parameters
        ----------
        config : TransposeHeuristicConfig | None, optional
            Optional op configuration.
        """
        self.config = config or TransposeHeuristicConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Refine an existing ordering via adjacent swaps.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. ``problem.edge_index`` supplies the
            crossing relationships.
        state : SolveState
            Mutable solve state with ``layers`` and ``ordering`` populated.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with a refined ``ordering`` tensor.
        """
        del ctx

        layers_cpu = _validate_layers(state.layers, problem.num_nodes)
        ordering_cpu = _validate_ordering(state.ordering, problem.num_nodes)
        edge_index_cpu = _validate_edge_index(problem.edge_index, problem.num_nodes)
        parents, children = _layered_neighbors_from_edges(
            edge_index=edge_index_cpu,
            layers=layers_cpu,
            num_nodes=problem.num_nodes,
        )
        ordered_layers = _initial_ordered_layers(layers=layers_cpu, ordering=ordering_cpu)
        layer_groups = {
            layer_index: list(layer_nodes) for layer_index, layer_nodes in enumerate(ordered_layers)
        }

        _transpose_heuristic(
            layer_groups=layer_groups,
            sorted_layers=list(range(len(ordered_layers))),
            children_of={node: child_nodes for node, child_nodes in enumerate(children)},
            parents_of={node: parent_nodes for node, parent_nodes in enumerate(parents)},
            num_passes=max(self.config.passes, 0),
        )

        refined_layers = [layer_groups[layer_index] for layer_index in range(len(ordered_layers))]
        state.ordering = _ordering_from_layers(
            ordered_layers=refined_layers,
            num_nodes=problem.num_nodes,
            device=_target_device(problem, state),
        )
        return state


@register_op
class SpectralOrder(Op):
    """Order each layer by the global Fiedler vector of the graph Laplacian."""

    name = "spectral_order"
    category = OpCategory.ORDERING
    reads = ("layers",)
    writes = ("ordering",)
    requires = ("layers",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute a Fiedler-vector ordering and project it into each layer.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs supplying the graph Laplacian.
        state : SolveState
            Mutable solve state with ``layers`` populated.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with ``ordering`` populated.
        """
        del ctx

        layers_cpu = _validate_layers(state.layers, problem.num_nodes)
        edge_index_cpu = _validate_edge_index(problem.edge_index, problem.num_nodes)
        ordered_layers = _initial_ordered_layers(layers_cpu)
        fiedler = _deterministic_fiedler_vector(
            edge_index=edge_index_cpu,
            num_nodes=problem.num_nodes,
        )

        if fiedler is not None:
            for layer_nodes in ordered_layers:
                stable_positions = {node: index for index, node in enumerate(layer_nodes)}
                layer_nodes.sort(
                    key=lambda node: (
                        float(fiedler[node].item()),
                        stable_positions[node],
                        node,
                    )
                )

        state.ordering = _ordering_from_layers(
            ordered_layers=ordered_layers,
            num_nodes=problem.num_nodes,
            device=_target_device(problem, state),
        )
        return state
