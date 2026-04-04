"""Coordinate assignment operations for layered and tree layouts."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from typing import List, Optional, Tuple

import torch

from dagua.layout._archive.classic.reingold_tilford import _assign_preliminary_x, _bfs_forest
from dagua.layout._archive.classic.sugiyama import _brandes_koepf_x_positions, _resolve_node_sizes
from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op


def _target_device(problem: LayoutProblem, state: SolveState) -> torch.device:
    """Resolve the device used for persisted position tensors.

    Parameters
    ----------
    problem : LayoutProblem
        Immutable layout inputs.
    state : SolveState
        Mutable solve state.

    Returns
    -------
    torch.device
        Device for the updated ``state.pos`` tensor.
    """
    if state.pos is not None:
        return state.pos.device
    if problem.node_sizes is not None:
        return problem.node_sizes.device
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
        raise ValueError("state.layers must be populated before coordinate ops run")
    if layers.ndim != 1 or layers.shape[0] != num_nodes:
        raise ValueError(f"state.layers must have shape [{num_nodes}]")
    return layers.detach().to(device="cpu", dtype=torch.long)


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


def _ordered_layers_from_state(layers: torch.Tensor, ordering: torch.Tensor) -> List[List[int]]:
    """Build ordered layer lists from per-node layer and ordering tensors.

    Parameters
    ----------
    layers : torch.Tensor
        CPU layer assignments with shape ``[N]``.
    ordering : torch.Tensor
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
        layer_nodes.sort(key=lambda node_idx: (int(ordering[node_idx].item()), node_idx))
    return ordered_layers


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


@dataclass(frozen=True)
class BrandesKopf4PassConfig:
    """Configuration for :class:`BrandesKopf4Pass`.

    Parameters
    ----------
    node_sep : float, default=1.0
        Horizontal gap between neighboring nodes.
    rank_sep : float, default=1.0
        Vertical gap between consecutive layers.
    """

    node_sep: float = 1.0
    rank_sep: float = 1.0


@dataclass(frozen=True)
class BucheimWalkerTreeConfig:
    """Configuration for :class:`BucheimWalkerTree`.

    Parameters
    ----------
    sibling_sep : float, default=1.0
        Horizontal spacing multiplier between neighboring siblings.
    layer_sep : float, default=1.5
        Vertical spacing between consecutive tree depths.
    component_gap : float, default=2.0
        Horizontal gap between disconnected tree components.
    """

    sibling_sep: float = 1.0
    layer_sep: float = 1.5
    component_gap: float = 2.0


@register_op
class BrandesKopf4Pass(Op):
    """Assign x coordinates via four-pass Brandes-Kopf compaction."""

    name = "brandes_kopf_4pass"
    category = OpCategory.COORDINATE
    reads = ("layers", "ordering")
    writes = ("pos",)
    requires = ("layers", "ordering")

    def __init__(self, config: Optional[BrandesKopf4PassConfig] = None) -> None:
        """Store the coordinate-assignment configuration.

        Parameters
        ----------
        config : BrandesKopf4PassConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BrandesKopf4PassConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Convert a layered ordering into concrete node coordinates.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state with ``layers`` and ``ordering`` populated.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with ``pos`` populated.
        """
        del ctx

        layers_cpu = _validate_layers(state.layers, problem.num_nodes)
        ordering_cpu = _validate_ordering(state.ordering, problem.num_nodes)
        edge_index_cpu = _validate_edge_index(problem.edge_index, problem.num_nodes)
        ordered_layers = _ordered_layers_from_state(layers_cpu, ordering_cpu)
        parents, children = _layered_neighbors_from_edges(
            edge_index=edge_index_cpu,
            layers=layers_cpu,
            num_nodes=problem.num_nodes,
        )
        node_sizes_cpu = _resolve_node_sizes(problem.node_sizes, problem.num_nodes)

        positions = torch.zeros((problem.num_nodes, 2), dtype=torch.float32)
        if problem.num_nodes > 0:
            x_coordinates = _brandes_koepf_x_positions(
                layers=ordered_layers,
                parents=parents,
                children=children,
                node_sizes=node_sizes_cpu,
                num_nodes=problem.num_nodes,
                num_original_nodes=problem.num_nodes,
                node_sep=self.config.node_sep,
            )
            positions[:, 0] = torch.tensor(x_coordinates, dtype=torch.float32)
            positions[:, 1] = layers_cpu.to(dtype=torch.float32) * self.config.rank_sep

        state.pos = positions.to(device=_target_device(problem, state))
        return state


@register_op
class BucheimWalkerTree(Op):
    """Lay out the graph as a tidy BFS forest using Buchheim's algorithm."""

    name = "bucheim_walker_tree"
    category = OpCategory.COORDINATE
    writes = ("pos",)

    def __init__(self, config: Optional[BucheimWalkerTreeConfig] = None) -> None:
        """Store the tree-layout configuration.

        Parameters
        ----------
        config : BucheimWalkerTreeConfig | None, optional
            Optional op configuration.
        """
        self.config = config or BucheimWalkerTreeConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Assign tidy tree coordinates directly from the graph topology.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable layout inputs.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Execution infrastructure. Unused by this deterministic op.

        Returns
        -------
        SolveState
            Updated state with ``pos`` populated.
        """
        del ctx

        edge_index_cpu = _validate_edge_index(problem.edge_index, problem.num_nodes)
        target_device = _target_device(problem, state)
        if problem.num_nodes == 0:
            state.pos = torch.empty((0, 2), dtype=torch.float32, device=target_device)
            return state

        sys.setrecursionlimit(max(sys.getrecursionlimit(), problem.num_nodes * 2))
        roots, children, depths = _bfs_forest(
            edge_index=edge_index_cpu,
            num_nodes=problem.num_nodes,
        )

        preliminary_x = [0.0] * problem.num_nodes
        next_component_offset = 0.0
        for root in roots:
            next_component_offset = _assign_preliminary_x(
                root_idx=root,
                children=children,
                depths=depths,
                preliminary_x=preliminary_x,
                component_offset=next_component_offset,
                component_gap=self.config.component_gap,
            )

        positions = torch.zeros((problem.num_nodes, 2), dtype=torch.float32)
        for node_idx in range(problem.num_nodes):
            positions[node_idx, 0] = float(preliminary_x[node_idx]) * self.config.sibling_sep
            positions[node_idx, 1] = float(depths[node_idx]) * self.config.layer_sep

        positions -= positions.mean(dim=0, keepdim=True)
        state.pos = positions.to(device=target_device)
        return state
