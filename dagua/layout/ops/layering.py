"""Layering operations for hierarchical graph layouts.

These ops expose the core Sugiyama layering primitives through the
composable ``Op`` interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import torch

from dagua.layout._archive.classic.sugiyama import (
    _expand_long_edges_with_dummy_nodes,
    _longest_path_layering,
    _promote_layer_assignments,
    _resolve_node_sizes,
)
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

    name = "longest_path_layering"
    category = OpCategory.LAYERING
    writes = ("layers",)

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

    name = "layer_promotion"
    category = OpCategory.LAYERING
    reads = ("layers",)
    writes = ("layers",)
    requires = ("layers",)

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

    name = "build_layer_index"
    category = OpCategory.LAYERING
    reads = ("layers",)
    writes = ("layer_index",)
    requires = ("layers",)

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
class InsertDummyNodes(Op):
    """Expand long edges through dummy-node chains for layered routing."""

    name = "insert_dummy_nodes"
    category = OpCategory.LAYERING
    reads = ("layers",)
    writes = ("extras.expanded_graph",)
    requires = ("layers",)

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
            edge_weights=_edge_weights_cpu(problem),
        )
        state.extras["expanded_graph"] = ExpandedGraph(
            edge_index=expanded_graph.edge_index,
            layers=expanded_graph.layers,
            node_sizes=expanded_graph.node_sizes,
            edge_paths=[list(path) for path in expanded_graph.edge_paths],
            num_nodes=expanded_graph.num_nodes,
        )
        return state
