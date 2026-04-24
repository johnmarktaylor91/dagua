"""Layering operations for hierarchical graph layouts.

These ops expose the core Sugiyama layering primitives through the
composable ``Op`` interface.
"""

from __future__ import annotations

import heapq
from dataclasses import dataclass, field
from typing import ClassVar, List, Optional, Tuple

import torch

from dagua.layout.layers import build_layer_index
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


@dataclass(frozen=True)
class BuildLayerIndexConfig:
    """Configuration for :class:`BuildLayerIndex`.

    Parameters
    ----------
    enable_cuda_sort : bool, default=True
        Whether CUDA argsort may be used when available and memory-safe.
    """

    enable_cuda_sort: bool = True


@dataclass(frozen=True)
class InsertDummyNodesConfig:
    """Configuration for :class:`InsertDummyNodes`.

    Parameters
    ----------
    dummy_width : float, default=0.0
        Width assigned to inserted dummy nodes.
    dummy_height : float, default=0.0
        Height assigned to inserted dummy nodes.
    """

    dummy_width: float = 0.0
    dummy_height: float = 0.0


@dataclass(frozen=True)
class ExpandedGraph:
    """Dummy-node-expanded layered DAG.

    Parameters
    ----------
    edge_index : torch.Tensor
        Expanded edge list with shape ``[2, E_expanded]`` on CPU.
    layers : list[list[int]]
        Node IDs grouped by layer after dummy insertion.
    node_sizes : torch.Tensor
        Expanded node sizes with shape ``[N_expanded, 2]`` on CPU.
    edge_paths : list[list[int]]
        Expanded node chains for each original edge.
    num_nodes : int
        Total node count after dummy insertion.
    """

    edge_index: torch.Tensor
    layers: list[list[int]]
    node_sizes: torch.Tensor
    edge_paths: list[list[int]]
    num_nodes: int


@dataclass(frozen=True)
class _ExpandedLayeredGraph:
    """Expanded DAG after dummy-node insertion for long edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Expanded edge list with shape ``[2, E_expanded]``.
    layers : list[list[int]]
        Node ids grouped by layer.
    node_sizes : torch.Tensor
        Expanded node sizes with shape ``[N_expanded, 2]``.
    edge_paths : list[list[int]]
        Expanded node chains for each original edge.
    num_nodes : int
        Total expanded node count.
    """

    edge_index: torch.Tensor
    layers: list[list[int]]
    node_sizes: torch.Tensor
    edge_paths: list[list[int]]
    num_nodes: int


def _resolve_node_sizes(node_sizes: torch.Tensor | None, num_nodes: int) -> torch.Tensor:
    """Return CPU node sizes for layer expansion.

    Parameters
    ----------
    node_sizes : torch.Tensor | None
        Optional input node sizes.
    num_nodes : int
        Number of original nodes.

    Returns
    -------
    torch.Tensor
        CPU float tensor with shape ``[N, 2]``.
    """
    if node_sizes is None:
        return torch.zeros((num_nodes, 2), dtype=torch.float32)
    return node_sizes.detach().to(device="cpu", dtype=torch.float32)


def _build_neighbor_lists(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> tuple[list[list[int]], list[list[int]]]:
    """Build parent and child adjacency lists from an edge index.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge list with shape ``[2, E]``.
    num_nodes : int
        Number of nodes in the graph.

    Returns
    -------
    tuple[list[list[int]], list[list[int]]]
        ``(parents, children)`` adjacency lists.
    """
    parents: List[List[int]] = [[] for _ in range(num_nodes)]
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    sources = edge_index[0].tolist()
    targets = edge_index[1].tolist()
    for source, target in zip(sources, targets):
        children[source].append(target)
        parents[target].append(source)
    return parents, children


def _longest_path_layering(edge_index: torch.Tensor, num_nodes: int) -> torch.Tensor:
    """Assign each node to a layer via Kahn traversal.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list of shape ``[2, E]`` on CPU.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    torch.Tensor
        Long tensor with shape ``[N]`` and layer indices.

    Raises
    ------
    ValueError
        If the graph remains cyclic after edge reversal.
    """
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    in_degree = [0] * num_nodes
    src_nodes = edge_index[0].tolist()
    dst_nodes = edge_index[1].tolist()

    for source, target in zip(src_nodes, dst_nodes):
        children[source].append(target)
        in_degree[target] += 1

    layers = [0] * num_nodes
    ready = [node for node, degree in enumerate(in_degree) if degree == 0]
    heapq.heapify(ready)

    processed = 0
    while ready:
        node = heapq.heappop(ready)
        processed += 1
        next_layer = layers[node] + 1
        for child in children[node]:
            if next_layer > layers[child]:
                layers[child] = next_layer
            in_degree[child] -= 1
            if in_degree[child] == 0:
                heapq.heappush(ready, child)

    if processed != num_nodes:
        raise ValueError("graph must be acyclic after back-edge reversal")

    return torch.tensor(layers, dtype=torch.long)


def _group_nodes_by_layer(layer_assignments: torch.Tensor, num_nodes: int) -> list[list[int]]:
    """Group nodes into ordered per-layer lists.

    Parameters
    ----------
    layer_assignments : torch.Tensor
        Layer indices for each node.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    list[list[int]]
        Node ids grouped by layer.
    """
    if num_nodes == 0:
        return []

    num_layers = int(layer_assignments.max().item()) + 1
    layers: List[List[int]] = [[] for _ in range(num_layers)]
    for node in range(num_nodes):
        layer_index = int(layer_assignments[node].item())
        layers[layer_index].append(node)
    return layers


def _promote_layer_assignments(
    edge_index: torch.Tensor,
    layer_assignments: torch.Tensor,
    num_nodes: int,
) -> torch.Tensor:
    """Push nodes down where outgoing edges skip multiple layers.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list with shape ``[2, E]`` on CPU.
    layer_assignments : torch.Tensor
        Initial layer assignments with shape ``[N]``.
    num_nodes : int
        Number of original nodes.

    Returns
    -------
    torch.Tensor
        Promoted layer assignments.
    """
    if num_nodes == 0 or edge_index.numel() == 0:
        return layer_assignments

    _, children = _build_neighbor_lists(edge_index=edge_index, num_nodes=num_nodes)
    promoted_layers = layer_assignments.clone()

    changed = True
    while changed:
        changed = False
        node_order = sorted(
            range(num_nodes),
            key=lambda node: int(promoted_layers[node].item()),
            reverse=True,
        )
        for node in node_order:
            successor_layers = [int(promoted_layers[child].item()) for child in children[node]]
            if not successor_layers:
                continue

            current_layer = int(promoted_layers[node].item())
            min_successor_layer = min(successor_layers)
            if min_successor_layer < current_layer + 2:
                continue

            candidate_layer = min_successor_layer - 1
            if candidate_layer > current_layer:
                promoted_layers[node] = candidate_layer
                changed = True

    return promoted_layers


def _expand_long_edges_with_dummy_nodes(
    edge_index: torch.Tensor,
    layer_assignments: torch.Tensor,
    node_sizes: torch.Tensor,
    num_original_nodes: int,
    dummy_size: tuple[float, float],
    edge_weights: torch.Tensor | None = None,
) -> tuple[_ExpandedLayeredGraph, torch.Tensor | None]:
    """Insert dummy nodes for edges spanning multiple layers.

    Parameters
    ----------
    edge_index : torch.Tensor
        Acyclic edge list with shape ``[2, E]`` on CPU.
    layer_assignments : torch.Tensor
        Original-node layer indices with shape ``[N]``.
    node_sizes : torch.Tensor
        Original node sizes with shape ``[N, 2]`` on CPU.
    num_original_nodes : int
        Number of real graph nodes before expansion.
    dummy_size : tuple[float, float]
        Size assigned to inserted dummy nodes.
    edge_weights : torch.Tensor | None, default=None
        Optional edge weights.

    Returns
    -------
    tuple[_ExpandedLayeredGraph, torch.Tensor | None]
        Expanded graph and expanded edge weights.
    """
    expanded_layers = _group_nodes_by_layer(
        layer_assignments=layer_assignments,
        num_nodes=num_original_nodes,
    )
    dummy_sizes: list[list[float]] = []
    expanded_sources: list[int] = []
    expanded_targets: list[int] = []
    expanded_weight_values: list[float] = []
    edge_paths: list[list[int]] = []
    next_dummy_index = num_original_nodes

    for edge_idx, (source, target) in enumerate(
        zip(edge_index[0].tolist(), edge_index[1].tolist())
    ):
        source_layer = int(layer_assignments[source].item())
        target_layer = int(layer_assignments[target].item())
        path = [source]
        previous = source
        orig_weight = float(edge_weights[edge_idx].item()) if edge_weights is not None else 1.0

        for layer_index in range(source_layer + 1, target_layer):
            dummy_index = next_dummy_index
            next_dummy_index += 1
            expanded_layers[layer_index].append(dummy_index)
            dummy_sizes.append([dummy_size[0], dummy_size[1]])
            expanded_sources.append(previous)
            expanded_targets.append(dummy_index)
            expanded_weight_values.append(orig_weight)
            path.append(dummy_index)
            previous = dummy_index

        expanded_sources.append(previous)
        expanded_targets.append(target)
        expanded_weight_values.append(orig_weight)
        path.append(target)
        edge_paths.append(path)

    if dummy_sizes:
        expanded_node_sizes = torch.cat(
            [
                node_sizes,
                torch.tensor(dummy_sizes, dtype=torch.float32),
            ],
            dim=0,
        )
    else:
        expanded_node_sizes = node_sizes.clone()

    expanded_edge_index = torch.tensor(
        [expanded_sources, expanded_targets],
        dtype=torch.long,
    )
    expanded_edge_weights: torch.Tensor | None = None
    if edge_weights is not None:
        expanded_edge_weights = torch.tensor(expanded_weight_values, dtype=torch.float32)

    return _ExpandedLayeredGraph(
        edge_index=expanded_edge_index,
        layers=expanded_layers,
        node_sizes=expanded_node_sizes,
        edge_paths=edge_paths,
        num_nodes=next_dummy_index,
    ), expanded_edge_weights


def _validate_problem_edge_index(problem: LayoutProblem) -> torch.Tensor:
    """Return a validated CPU ``edge_index`` tensor.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[2, E]``.

    Raises
    ------
    ValueError
        If the edge tensor shape or node references are invalid.
    """
    edge_index = problem.edge_index
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("problem.edge_index must have shape [2, E]")

    edge_index_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    if edge_index_cpu.numel() == 0:
        return edge_index_cpu

    min_index = int(edge_index_cpu.min().item())
    max_index = int(edge_index_cpu.max().item())
    if min_index < 0:
        raise ValueError("problem.edge_index cannot contain negative node indices")
    if max_index >= problem.num_nodes:
        raise ValueError("problem.edge_index references a node outside problem.num_nodes")
    return edge_index_cpu


def _validate_layer_assignments(
    layers: Optional[torch.Tensor],
    num_nodes: int,
) -> torch.Tensor:
    """Return validated CPU layer assignments.

    Parameters
    ----------
    layers : torch.Tensor or None
        Candidate layer tensor.
    num_nodes : int
        Expected node count.

    Returns
    -------
    torch.Tensor
        CPU long tensor with shape ``[N]``.

    Raises
    ------
    ValueError
        If ``layers`` is missing or violates the shape contract.
    """
    if layers is None:
        raise ValueError("state.layers must be populated before this op runs")
    if layers.ndim != 1 or layers.shape[0] != num_nodes:
        raise ValueError(f"state.layers must have shape [{num_nodes}]")

    layers_cpu = layers.detach().to(device="cpu", dtype=torch.long)
    if layers_cpu.numel() > 0 and int(layers_cpu.min().item()) < 0:
        raise ValueError("state.layers cannot contain negative layer indices")
    return layers_cpu


def _target_layer_device(problem: LayoutProblem, state: SolveState) -> torch.device:
    """Choose the device for persisted layer tensors.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.device
        Device matching the current structural tensors when possible.
    """
    if state.layers is not None:
        return state.layers.device
    if state.pos is not None:
        return state.pos.device
    return problem.edge_index.device


def _edge_weights_cpu(problem: LayoutProblem) -> Optional[torch.Tensor]:
    """Return validated CPU edge weights when provided.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.

    Returns
    -------
    torch.Tensor or None
        CPU float tensor with shape ``[E]`` when weights are present.

    Raises
    ------
    ValueError
        If the edge-weight shape does not match the edge count.
    """
    if problem.edge_weights is None:
        return None
    edge_weights = problem.edge_weights.detach().to(device="cpu", dtype=torch.float32)
    if edge_weights.ndim != 1 or edge_weights.shape[0] != problem.edge_index.shape[1]:
        raise ValueError("problem.edge_weights must have shape [E]")
    return edge_weights


@register_op
class LongestPathLayering(Op):
    """Assign layers by longest-path depth in an acyclic graph."""

    name: ClassVar[str] = "longest_path_layering"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    writes: ClassVar[Tuple[str, ...]] = ("layers",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute Kahn-style longest-path layers.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. ``problem.edge_index`` must already be
            acyclic; cyclic inputs raise ``ValueError``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with ``layers`` populated.
        """
        del ctx

        edge_index = _validate_problem_edge_index(problem)
        layers_cpu = _longest_path_layering(edge_index=edge_index, num_nodes=problem.num_nodes)
        state.layers = layers_cpu.to(device=_target_layer_device(problem, state))
        return state


@register_op
class LayerPromotion(Op):
    """Push nodes to the deepest legal layer after longest-path assignment."""

    name: ClassVar[str] = "layer_promotion"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("layers",)
    writes: ClassVar[Tuple[str, ...]] = ("layers",)
    requires: ClassVar[Tuple[str, ...]] = ("layers",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Promote nodes downward without violating edge directions.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. ``problem.edge_index`` must already be
            acyclic.
        state : SolveState
            Mutable solve state with existing layer assignments.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with promoted ``layers``.
        """
        del ctx

        edge_index = _validate_problem_edge_index(problem)
        layers_cpu = _validate_layer_assignments(state.layers, problem.num_nodes)
        promoted_cpu = _promote_layer_assignments(
            edge_index=edge_index,
            layer_assignments=layers_cpu,
            num_nodes=problem.num_nodes,
        )
        state.layers = promoted_cpu.to(device=_target_layer_device(problem, state))
        return state


@register_op
class BuildLayerIndex(Op):
    """Build a reusable per-layer index for layer-aware losses and projections."""

    name: ClassVar[str] = "build_layer_index"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("layers",)
    writes: ClassVar[Tuple[str, ...]] = ("layer_index",)
    requires: ClassVar[Tuple[str, ...]] = ("layers",)

    def __init__(self, config: Optional[BuildLayerIndexConfig] = None) -> None:
        """Initialize the op with an optional config.

        Parameters
        ----------
        config : BuildLayerIndexConfig, optional
            Layer-index build settings. Defaults to the standard config.
        """
        self.config = config or BuildLayerIndexConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Build and store the layer index on the current layer device.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state with existing layer assignments.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with ``layer_index`` populated.
        """
        del problem, ctx

        if state.layers is None:
            raise ValueError("state.layers must be populated before BuildLayerIndex runs")
        layer_device = str(state.layers.device)
        state.layer_index = build_layer_index(
            layer_assignments=state.layers,
            device=layer_device,
            enable_cuda_sort=self.config.enable_cuda_sort,
        )
        return state


@register_op
@dataclass(frozen=True)
class InsertDummyNodes(Op):
    """Expand long edges through dummy-node chains for layered routing."""

    config: InsertDummyNodesConfig = field(default_factory=InsertDummyNodesConfig)

    name: ClassVar[str] = "insert_dummy_nodes"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("layers",)
    writes: ClassVar[Tuple[str, ...]] = ("extras.expanded_graph",)
    requires: ClassVar[Tuple[str, ...]] = ("layers",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Insert dummy nodes on edges spanning multiple layers.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Node sizes default to zeros when absent.
        state : SolveState
            Mutable solve state with existing layer assignments.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            Updated state with ``extras["expanded_graph"]`` populated.
        """
        del ctx

        edge_index = _validate_problem_edge_index(problem)
        layers_cpu = _validate_layer_assignments(state.layers, problem.num_nodes)
        node_sizes = _resolve_node_sizes(node_sizes=problem.node_sizes, num_nodes=problem.num_nodes)
        expanded_graph, _ = _expand_long_edges_with_dummy_nodes(
            edge_index=edge_index,
            layer_assignments=layers_cpu,
            node_sizes=node_sizes,
            num_original_nodes=problem.num_nodes,
            dummy_size=(self.config.dummy_width, self.config.dummy_height),
            edge_weights=_edge_weights_cpu(problem),
        )
        # Preserve the original edge order via edge_paths so coordinate and
        # routing ops can collapse the expanded graph back to per-edge routes.
        state.extras["expanded_graph"] = ExpandedGraph(
            edge_index=expanded_graph.edge_index,
            layers=expanded_graph.layers,
            node_sizes=expanded_graph.node_sizes,
            edge_paths=[list(path) for path in expanded_graph.edge_paths],
            num_nodes=expanded_graph.num_nodes,
        )
        return state


def _expanded_layers_to_tensor(
    expanded_layers: list[list[int]],
    num_nodes: int,
    device: torch.device,
) -> torch.Tensor:
    """Convert expanded layer groups into a per-node tensor.

    Parameters
    ----------
    expanded_layers : list[list[int]]
        Node IDs grouped by layer for the expanded graph.
    num_nodes : int
        Number of expanded graph nodes.
    device : torch.device
        Target device for the returned tensor.

    Returns
    -------
    torch.Tensor
        Layer assignments with shape ``[N_expanded]``.

    Raises
    ------
    ValueError
        If any expanded node is missing from the grouped layers.
    """
    layers = torch.full((num_nodes,), -1, dtype=torch.long)
    for layer_index, nodes in enumerate(expanded_layers):
        if nodes:
            layers[torch.tensor(nodes, dtype=torch.long)] = layer_index
    if bool((layers < 0).any().item()):
        raise ValueError("expanded_graph.layers must cover every expanded node")
    return layers.to(device=device)


def _seed_expanded_positions(
    pos: torch.Tensor,
    edge_paths: list[list[int]],
    expanded_num_nodes: int,
) -> torch.Tensor:
    """Interpolate dummy positions along each original edge path.

    Parameters
    ----------
    pos : torch.Tensor
        Original-node positions with shape ``[N_original, 2]``.
    edge_paths : list[list[int]]
        Expanded path for each original edge.
    expanded_num_nodes : int
        Total number of expanded graph nodes.

    Returns
    -------
    torch.Tensor
        Expanded positions with shape ``[N_expanded, 2]``.
    """
    expanded = torch.zeros((expanded_num_nodes, 2), dtype=pos.dtype, device=pos.device)
    expanded[: pos.shape[0]] = pos
    for path in edge_paths:
        if len(path) <= 2:
            continue
        start = pos[path[0]]
        end = pos[path[-1]]
        denom = float(len(path) - 1)
        for step, node in enumerate(path[1:-1], start=1):
            alpha = float(step) / denom
            expanded[node] = start + ((end - start) * alpha)
    return expanded


@register_op
class ActivateExpandedGraphState(Op):
    """Promote active positions and layers to a dummy-expanded graph."""

    name: ClassVar[str] = "activate_expanded_graph_state"
    category: ClassVar[OpCategory] = OpCategory.LAYERING
    reads: ClassVar[Tuple[str, ...]] = ("pos", "layers", "layer_index", "extras.expanded_graph")
    writes: ClassVar[Tuple[str, ...]] = (
        "pos",
        "layers",
        "layer_index",
        "ordering",
        "extras.original_layers",
        "extras.original_layer_index",
    )
    requires: ClassVar[Tuple[str, ...]] = ("pos", "extras.expanded_graph")

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Expand active solve tensors after dummy-node insertion.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs. Unused by this op.
        state : SolveState
            Mutable solve state with original-node positions and expansion
            metadata in ``extras["expanded_graph"]``.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this op.

        Returns
        -------
        SolveState
            State whose ``pos``, ``layers``, and ``layer_index`` match the
            expanded graph.
        """
        del problem, ctx
        if state.pos is None:
            raise ValueError("ActivateExpandedGraphState requires state.pos")
        expanded_graph = state.extras.get("expanded_graph")
        if expanded_graph is None:
            return state

        state.extras["original_layers"] = None if state.layers is None else state.layers.clone()
        state.extras["original_layer_index"] = state.layer_index

        expanded_pos = _seed_expanded_positions(
            pos=state.pos.detach(),
            edge_paths=expanded_graph.edge_paths,
            expanded_num_nodes=int(expanded_graph.num_nodes),
        )
        expanded_layers = _expanded_layers_to_tensor(
            expanded_layers=expanded_graph.layers,
            num_nodes=int(expanded_graph.num_nodes),
            device=state.pos.device,
        )
        state.pos = expanded_pos.requires_grad_(state.pos.requires_grad)
        state.layers = expanded_layers
        state.layer_index = build_layer_index(expanded_layers, device=str(state.pos.device))
        state.ordering = None
        return state
