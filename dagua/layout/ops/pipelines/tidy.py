"""Non-layered tidy tree pipeline with variable node heights.

This is a deterministic source port of the non-layered behavior from
``zxch3n/tidy``: each child y-coordinate is based on its actual parent's bottom
edge, so tall nodes push only their own descendants down. The x-coordinate pass
uses tidy-tree centering plus contour separation to avoid overlapping sibling
subtrees with variable node widths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple, Union

import torch

from dagua.layout.ops.base import Op, Pipeline
from dagua.layout.ops.state import ExecutionPlan, LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

_DEFAULT_PARENT_CHILD_MARGIN = 10.0
_DEFAULT_PEER_MARGIN = 10.0
_DEFAULT_NODE_WIDTH = 1.0
_DEFAULT_NODE_HEIGHT = 1.0


@dataclass(frozen=True)
class TidyConfig:
    """Configuration for the non-layered tidy tree pipeline.

    Parameters
    ----------
    parent_child_margin : float, default=10.0
        Vertical gap between a parent bottom and each direct child top.
    peer_margin : float, default=10.0
        Minimum horizontal gap between sibling subtree bounding boxes.
    dtype : torch.dtype, default=torch.float32
        Output tensor dtype.
    """

    parent_child_margin: float = _DEFAULT_PARENT_CHILD_MARGIN
    peer_margin: float = _DEFAULT_PEER_MARGIN
    dtype: torch.dtype = torch.float32


@dataclass
class _TidyNode:
    """Mutable tree node used by the tidy port.

    Parameters
    ----------
    index : int
        Original graph node index.
    width : float
        Node box width.
    height : float
        Node box height.
    parent : _TidyNode, optional
        Parent node.
    """

    index: int
    width: float
    height: float
    parent: Optional["_TidyNode"] = None
    children: List["_TidyNode"] = None  # type: ignore[assignment]
    x: float = 0.0
    y: float = 0.0
    prelim: float = 0.0

    def __post_init__(self) -> None:
        """Initialize the child list.

        Returns
        -------
        None
            The node receives its own mutable child list.
        """
        if self.children is None:
            self.children = []


def _node_dimensions(
    node_sizes: Optional[torch.Tensor],
    num_nodes: int,
) -> Tuple[List[float], List[float]]:
    """Return per-node widths and heights.

    Parameters
    ----------
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple[list[float], list[float]]
        Widths and heights for all nodes.
    """
    if node_sizes is None or node_sizes.numel() == 0:
        return (
            [_DEFAULT_NODE_WIDTH for _ in range(num_nodes)],
            [_DEFAULT_NODE_HEIGHT for _ in range(num_nodes)],
        )
    sizes = node_sizes.detach().cpu().to(torch.float64)
    widths = [max(float(sizes[i, 0]), _DEFAULT_NODE_WIDTH) for i in range(num_nodes)]
    heights = [max(float(sizes[i, 1]), _DEFAULT_NODE_HEIGHT) for i in range(num_nodes)]
    return widths, heights


def _children_from_edges(
    edge_index: torch.Tensor,
    num_nodes: int,
) -> Tuple[List[List[int]], List[int]]:
    """Build deterministic child lists from directed tree edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]`` interpreted as ``parent -> child``.
    num_nodes : int
        Number of nodes.

    Returns
    -------
    tuple[list[list[int]], list[int]]
        Child indices per node and root indices.
    """
    children: List[List[int]] = [[] for _ in range(num_nodes)]
    parent: List[Optional[int]] = [None for _ in range(num_nodes)]
    if edge_index.numel() != 0:
        for raw_u, raw_v in edge_index.detach().cpu().t().tolist():
            u = int(raw_u)
            v = int(raw_v)
            if u == v or u < 0 or v < 0 or u >= num_nodes or v >= num_nodes:
                continue
            if parent[v] is not None:
                continue
            parent[v] = u
            children[u].append(v)
    roots = [idx for idx, value in enumerate(parent) if value is None]
    return children, roots


def _build_tree(
    root_index: int,
    child_indices: List[List[int]],
    widths: List[float],
    heights: List[float],
) -> _TidyNode:
    """Build a pointer-free tree for one component.

    Parameters
    ----------
    root_index : int
        Root graph node index.
    child_indices : list[list[int]]
        Child indices per graph node.
    widths : list[float]
        Node widths.
    heights : list[float]
        Node heights.

    Returns
    -------
    _TidyNode
        Root of the copied tree component.
    """
    root = _TidyNode(root_index, widths[root_index], heights[root_index])
    stack: List[Tuple[_TidyNode, int]] = [(root, root_index)]
    while stack:
        node, graph_index = stack.pop()
        for child_index in reversed(child_indices[graph_index]):
            child = _TidyNode(
                child_index,
                widths[child_index],
                heights[child_index],
                parent=node,
            )
            node.children.insert(0, child)
            stack.append((child, child_index))
    return root


def _assign_y(node: _TidyNode, parent_child_margin: float) -> None:
    """Assign non-layered y coordinates using parent heights.

    Parameters
    ----------
    node : _TidyNode
        Root of the subtree.
    parent_child_margin : float
        Vertical gap after each parent box.

    Returns
    -------
    None
        The tree is updated in place.
    """
    for child in node.children:
        child.y = node.y + node.height + parent_child_margin
        _assign_y(child, parent_child_margin)


def _subtree_contour(node: _TidyNode) -> Dict[float, Tuple[float, float]]:
    """Return horizontal extents grouped by y band.

    Parameters
    ----------
    node : _TidyNode
        Root of the subtree.

    Returns
    -------
    dict[float, tuple[float, float]]
        Mapping from node top y to ``(left, right)`` extents.
    """
    contour: Dict[float, Tuple[float, float]] = {}
    stack = [node]
    while stack:
        current = stack.pop()
        left = current.prelim - current.width / 2.0
        right = current.prelim + current.width / 2.0
        if current.y in contour:
            old_left, old_right = contour[current.y]
            contour[current.y] = (min(old_left, left), max(old_right, right))
        else:
            contour[current.y] = (left, right)
        stack.extend(current.children)
    return contour


def _required_shift(left: _TidyNode, right: _TidyNode, peer_margin: float) -> float:
    """Compute the horizontal shift needed between sibling subtrees.

    Parameters
    ----------
    left : _TidyNode
        Previous sibling subtree.
    right : _TidyNode
        Current sibling subtree.
    peer_margin : float
        Required horizontal gap.

    Returns
    -------
    float
        Non-negative shift for the right subtree.
    """
    left_contour = _subtree_contour(left)
    right_contour = _subtree_contour(right)
    shift = 0.0
    for y_value, (_, left_right) in left_contour.items():
        if y_value not in right_contour:
            continue
        right_left, _ = right_contour[y_value]
        shift = max(shift, left_right - right_left + peer_margin)
    return max(shift, 0.0)


def _shift_subtree(node: _TidyNode, shift: float) -> None:
    """Move a subtree horizontally.

    Parameters
    ----------
    node : _TidyNode
        Root of the subtree.
    shift : float
        Horizontal offset.

    Returns
    -------
    None
        The subtree prelim values are updated in place.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        current.prelim += shift
        stack.extend(current.children)


def _first_walk(node: _TidyNode, peer_margin: float) -> None:
    """Compute preliminary tidy x coordinates.

    Parameters
    ----------
    node : _TidyNode
        Root of the subtree.
    peer_margin : float
        Minimum sibling subtree gap.

    Returns
    -------
    None
        ``prelim`` values are updated in place.
    """
    if not node.children:
        node.prelim = 0.0
        return

    previous: Optional[_TidyNode] = None
    for child in node.children:
        _first_walk(child, peer_margin)
        if previous is not None:
            _shift_subtree(child, _required_shift(previous, child, peer_margin))
        previous = child

    first = node.children[0].prelim
    last = node.children[-1].prelim
    node.prelim = (first + last) / 2.0


def _normalize_tree(node: _TidyNode) -> None:
    """Shift a component so its root is centered at x=0.

    Parameters
    ----------
    node : _TidyNode
        Root of the component.

    Returns
    -------
    None
        Coordinates are updated in place.
    """
    root_x = node.prelim
    stack = [node]
    while stack:
        current = stack.pop()
        current.x = current.prelim - root_x
        stack.extend(current.children)


def _translate_tree_x(node: _TidyNode, shift: float) -> None:
    """Translate final x coordinates for a component.

    Parameters
    ----------
    node : _TidyNode
        Root of the component.
    shift : float
        Final x-coordinate offset.

    Returns
    -------
    None
        The component x coordinates are updated in place.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        current.x += shift
        stack.extend(current.children)


def _collect_positions(node: _TidyNode, positions: torch.Tensor) -> None:
    """Write tree positions to the output tensor.

    Parameters
    ----------
    node : _TidyNode
        Root of the component.
    positions : torch.Tensor
        Output positions with shape ``[N, 2]``.

    Returns
    -------
    None
        ``positions`` is updated in place.
    """
    stack = [node]
    while stack:
        current = stack.pop()
        positions[current.index, 0] = current.x
        positions[current.index, 1] = current.y
        stack.extend(current.children)


def _component_width(node: _TidyNode) -> Tuple[float, float]:
    """Return component horizontal extents including node widths.

    Parameters
    ----------
    node : _TidyNode
        Root of the component.

    Returns
    -------
    tuple[float, float]
        Minimum and maximum x extents.
    """
    min_x = float("inf")
    max_x = float("-inf")
    stack = [node]
    while stack:
        current = stack.pop()
        min_x = min(min_x, current.x - current.width / 2.0)
        max_x = max(max_x, current.x + current.width / 2.0)
        stack.extend(current.children)
    return min_x, max_x


def _layout_tidy_forest(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor],
    config: TidyConfig,
) -> torch.Tensor:
    """Lay out a forest with non-layered tidy placement.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]`` interpreted as parent-child edges.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``.
    config : TidyConfig
        Layout configuration.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    widths, heights = _node_dimensions(node_sizes, num_nodes)
    child_indices, roots = _children_from_edges(edge_index, num_nodes)
    positions = torch.zeros((num_nodes, 2), dtype=config.dtype, device=edge_index.device)
    offset = 0.0
    for root_index in roots:
        root = _build_tree(root_index, child_indices, widths, heights)
        _assign_y(root, config.parent_child_margin)
        _first_walk(root, config.peer_margin)
        _normalize_tree(root)
        min_x, max_x = _component_width(root)
        _translate_tree_x(root, offset - min_x)
        _collect_positions(root, positions)
        offset += (max_x - min_x) + config.peer_margin
    return positions


@register_op
@dataclass
class RunTidyTreeLayout(Op):
    """Run non-layered tidy-tree placement."""

    config: TidyConfig
    name: str = "run_tidy_tree_layout"
    category: OpCategory = OpCategory.COORDINATE
    writes: Tuple[str, ...] = ("pos",)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Compute tidy tree positions.

        Parameters
        ----------
        problem : LayoutProblem
            Graph topology and node sizes.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context; accepted for pipeline compatibility.

        Returns
        -------
        SolveState
            State with final ``pos`` populated.
        """
        del ctx
        state.pos = _layout_tidy_forest(
            problem.edge_index,
            problem.num_nodes,
            problem.node_sizes,
            self.config,
        )
        return state


def build_tidy_pipeline(config: Optional[TidyConfig] = None) -> Pipeline:
    """Build the non-layered tidy tree pipeline.

    Parameters
    ----------
    config : TidyConfig, optional
        Pipeline configuration. ``None`` uses reference-style defaults.

    Returns
    -------
    Pipeline
        Single-stage tidy placement pipeline.
    """
    resolved = TidyConfig() if config is None else config
    if resolved.parent_child_margin < 0.0:
        raise ValueError("parent_child_margin must be non-negative.")
    if resolved.peer_margin < 0.0:
        raise ValueError("peer_margin must be non-negative.")
    return Pipeline([RunTidyTreeLayout(resolved)], name="tidy_pipeline")


def layout_tidy_pipeline(
    edge_index: torch.Tensor,
    num_nodes: int,
    node_sizes: Optional[torch.Tensor] = None,
    *,
    parent_child_margin: float = _DEFAULT_PARENT_CHILD_MARGIN,
    peer_margin: float = _DEFAULT_PEER_MARGIN,
    dtype: Union[torch.dtype, str] = torch.float32,
    **kwargs: object,
) -> torch.Tensor:
    """Lay out a tree or forest with the non-layered tidy algorithm.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]`` interpreted as parent-child edges.
    num_nodes : int
        Number of graph nodes.
    node_sizes : torch.Tensor, optional
        Node sizes with shape ``[N, 2]``. Heights directly affect descendant y
        positions, matching the non-layered reference mode.
    parent_child_margin : float, default=10.0
        Vertical gap between a parent bottom and its children.
    peer_margin : float, default=10.0
        Horizontal gap between sibling subtrees.
    dtype : torch.dtype or str, default=torch.float32
        Output dtype.
    **kwargs : object
        Additional dispatch kwargs accepted for compatibility.

    Returns
    -------
    torch.Tensor
        Final positions with shape ``[N, 2]``.
    """
    del kwargs
    resolved_dtype = getattr(torch, dtype) if isinstance(dtype, str) else dtype
    config = TidyConfig(
        parent_child_margin=parent_child_margin,
        peer_margin=peer_margin,
        dtype=resolved_dtype,
    )
    problem = LayoutProblem(edge_index=edge_index, num_nodes=num_nodes, node_sizes=node_sizes)
    state = SolveState()
    ctx = RuntimeContext(plan=ExecutionPlan(device=str(edge_index.device)))
    final_state = build_tidy_pipeline(config).apply(problem, state, ctx)
    if final_state.pos is None:
        raise RuntimeError("tidy pipeline did not produce positions.")
    return final_state.pos.to(device=edge_index.device, dtype=resolved_dtype)


__all__ = [
    "RunTidyTreeLayout",
    "TidyConfig",
    "build_tidy_pipeline",
    "layout_tidy_pipeline",
]
