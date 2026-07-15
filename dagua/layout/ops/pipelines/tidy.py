"""Non-layered tidy tree pipeline with variable node heights.

This is a deterministic source port of the non-layered behavior from
``zxch3n/tidy``: each child y-coordinate is based on its actual parent's bottom
edge, so tall nodes push only their own descendants down. The x-coordinate pass
uses tidy-tree centering plus contour separation to avoid overlapping sibling
subtrees with variable node widths.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple, Union

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
    relative_x: float = 0.0
    shift_acceleration: float = 0.0
    shift_change: float = 0.0
    modifier_to_subtree: float = 0.0
    modifier_thread_left: float = 0.0
    modifier_thread_right: float = 0.0
    modifier_extreme_left: float = 0.0
    modifier_extreme_right: float = 0.0
    thread_left: Optional["_TidyNode"] = None
    thread_right: Optional["_TidyNode"] = None
    extreme_left: Optional["_TidyNode"] = None
    extreme_right: Optional["_TidyNode"] = None

    def __post_init__(self) -> None:
        """Initialize the child list.

        Returns
        -------
        None
            The node receives its own mutable child list.
        """
        if self.children is None:
            self.children = []


@dataclass
class _LinkedYList:
    """Linked list of contour bottoms used by the tidy apportion pass.

    Parameters
    ----------
    index : int
        Child index associated with this contour segment.
    y : float
        Bottom y-coordinate of the segment.
    next : _LinkedYList, optional
        Next segment with a lower active bottom.
    """

    index: int
    y: float
    next: Optional["_LinkedYList"] = None


@dataclass
class _TidyContour:
    """Stateful left or right contour cursor.

    Parameters
    ----------
    is_left : bool
        ``True`` for a left contour, ``False`` for a right contour.
    current : _TidyNode
        Current contour node.
    modifier_sum : float
        Accumulated modifier from the subtree root to ``current``.
    """

    is_left: bool
    current: Optional[_TidyNode]
    modifier_sum: float


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


def _init_tidy_node(node: _TidyNode) -> None:
    """Reset layout fields before running the tidy pass.

    Parameters
    ----------
    node : _TidyNode
        Node to reset.

    Returns
    -------
    None
        The node's mutable layout fields are reset in place.
    """
    node.x = 0.0
    node.y = 0.0
    node.relative_x = 0.0
    node.shift_acceleration = 0.0
    node.shift_change = 0.0
    node.modifier_to_subtree = 0.0
    node.modifier_thread_left = 0.0
    node.modifier_thread_right = 0.0
    node.modifier_extreme_left = 0.0
    node.modifier_extreme_right = 0.0
    node.thread_left = None
    node.thread_right = None
    node.extreme_left = None
    node.extreme_right = None


def _init_tidy_tree(root: _TidyNode) -> None:
    """Reset every node in a tidy component.

    Parameters
    ----------
    root : _TidyNode
        Root of the component.

    Returns
    -------
    None
        All layout fields are reset in place.
    """
    stack = [root]
    while stack:
        node = stack.pop()
        _init_tidy_node(node)
        stack.extend(node.children)


def _set_extreme(node: _TidyNode) -> None:
    """Update cached extreme contour nodes for a subtree.

    Parameters
    ----------
    node : _TidyNode
        Subtree root.

    Returns
    -------
    None
        Extreme fields and their modifiers are updated in place.
    """
    if not node.children:
        node.extreme_left = node
        node.extreme_right = node
        node.modifier_extreme_left = 0.0
        node.modifier_extreme_right = 0.0
        return
    first = node.children[0]
    last = node.children[-1]
    node.extreme_left = first.extreme_left
    node.modifier_extreme_left = first.modifier_to_subtree + first.modifier_extreme_left
    node.extreme_right = last.extreme_right
    node.modifier_extreme_right = last.modifier_to_subtree + last.modifier_extreme_right


def _extreme_left(node: _TidyNode) -> _TidyNode:
    """Return the cached left extreme node.

    Parameters
    ----------
    node : _TidyNode
        Subtree root.

    Returns
    -------
    _TidyNode
        Left extreme node.
    """
    if node.extreme_left is None:
        raise RuntimeError("tidy left extreme was not initialized.")
    return node.extreme_left


def _extreme_right(node: _TidyNode) -> _TidyNode:
    """Return the cached right extreme node.

    Parameters
    ----------
    node : _TidyNode
        Subtree root.

    Returns
    -------
    _TidyNode
        Right extreme node.
    """
    if node.extreme_right is None:
        raise RuntimeError("tidy right extreme was not initialized.")
    return node.extreme_right


def _position_root(node: _TidyNode) -> None:
    """Center an internal node over its first and last child.

    Parameters
    ----------
    node : _TidyNode
        Internal node.

    Returns
    -------
    None
        ``relative_x`` and ``modifier_to_subtree`` are updated in place.
    """
    first = node.children[0]
    last = node.children[-1]
    first_child_pos = first.relative_x + first.modifier_to_subtree
    last_child_pos = last.relative_x + last.modifier_to_subtree
    node.relative_x = (first_child_pos + last_child_pos) / 2.0
    node.modifier_to_subtree = -node.relative_x


def _add_child_spacing(node: _TidyNode) -> None:
    """Apply deferred sibling spacing shifts to direct children.

    Parameters
    ----------
    node : _TidyNode
        Parent whose children carry deferred shift fields.

    Returns
    -------
    None
        Child modifiers are updated in place and deferred fields are cleared.
    """
    speed = 0.0
    delta = 0.0
    for child in node.children:
        speed += child.shift_acceleration
        delta += speed + child.shift_change
        child.modifier_to_subtree += delta
        child.shift_acceleration = 0.0
        child.shift_change = 0.0


def _linked_y_bottom(y_list: _LinkedYList) -> float:
    """Return a linked contour segment bottom.

    Parameters
    ----------
    y_list : _LinkedYList
        Current contour segment.

    Returns
    -------
    float
        Segment bottom coordinate.
    """
    return y_list.y


def _linked_y_update(y_list: _LinkedYList, index: int, y_value: float) -> _LinkedYList:
    """Insert or replace the active contour segment.

    Parameters
    ----------
    y_list : _LinkedYList
        Current linked y-list head.
    index : int
        Child index for the new segment.
    y_value : float
        Bottom y-coordinate for the new segment.

    Returns
    -------
    _LinkedYList
        Updated list head.
    """
    node = y_list
    while node.y <= y_value:
        if node.next is None:
            return _LinkedYList(index=index, y=y_value)
        node = node.next
    return _LinkedYList(index=index, y=y_value, next=node)


def _contour_left(contour: _TidyContour) -> float:
    """Return the current contour node's left edge.

    Parameters
    ----------
    contour : _TidyContour
        Active contour cursor.

    Returns
    -------
    float
        Absolute left edge in the contour coordinate frame.
    """
    if contour.current is None:
        raise RuntimeError("tidy contour has no current node.")
    return contour.modifier_sum + contour.current.relative_x - contour.current.width / 2.0


def _contour_right(contour: _TidyContour) -> float:
    """Return the current contour node's right edge.

    Parameters
    ----------
    contour : _TidyContour
        Active contour cursor.

    Returns
    -------
    float
        Absolute right edge in the contour coordinate frame.
    """
    if contour.current is None:
        raise RuntimeError("tidy contour has no current node.")
    return contour.modifier_sum + contour.current.relative_x + contour.current.width / 2.0


def _contour_bottom(contour: _TidyContour) -> float:
    """Return the current contour node's bottom y-coordinate.

    Parameters
    ----------
    contour : _TidyContour
        Active contour cursor.

    Returns
    -------
    float
        Bottom y-coordinate, or zero for a finished contour.
    """
    if contour.current is None:
        return 0.0
    return contour.current.y + contour.current.height


def _contour_next(contour: _TidyContour) -> None:
    """Advance a contour cursor using children or threads.

    Parameters
    ----------
    contour : _TidyContour
        Mutable contour cursor.

    Returns
    -------
    None
        The cursor is advanced in place.
    """
    current = contour.current
    if current is None:
        return
    if contour.is_left:
        if current.children:
            contour.current = current.children[0]
            contour.modifier_sum += contour.current.modifier_to_subtree
        else:
            contour.modifier_sum += current.modifier_thread_left
            contour.current = current.thread_left
    elif current.children:
        contour.current = current.children[-1]
        contour.modifier_sum += contour.current.modifier_to_subtree
    else:
        contour.modifier_sum += current.modifier_thread_right
        contour.current = current.thread_right


def _move_subtree(
    node: _TidyNode,
    current_index: int,
    from_index: int,
    distance: float,
) -> None:
    """Move the right sibling block and defer spacing between siblings.

    Parameters
    ----------
    node : _TidyNode
        Parent whose child block is moved.
    current_index : int
        Index of the current right subtree.
    from_index : int
        Index that anchors deferred spacing.
    distance : float
        Horizontal distance to add.

    Returns
    -------
    None
        Child modifier and deferred shift fields are updated in place.
    """
    child = node.children[current_index]
    child.modifier_to_subtree += distance
    if from_index == current_index - 1:
        return
    index_diff = float(current_index - from_index)
    node.children[from_index + 1].shift_acceleration += distance / index_diff
    node.children[current_index].shift_acceleration -= distance / index_diff
    node.children[current_index].shift_change -= distance - distance / index_diff


def _set_left_thread(
    node: _TidyNode,
    current_index: int,
    target: _TidyNode,
    modifier: float,
) -> None:
    """Attach a left thread after the left contour finishes first.

    Parameters
    ----------
    node : _TidyNode
        Parent of the sibling subtrees.
    current_index : int
        Current right child index.
    target : _TidyNode
        Contour node to thread to.
    modifier : float
        Current contour modifier sum.

    Returns
    -------
    None
        Thread and extreme fields are updated in place.
    """
    first = node.children[0]
    current = node.children[current_index]
    diff = modifier - first.modifier_extreme_left - first.modifier_to_subtree
    left_extreme = _extreme_left(first)
    left_extreme.thread_left = target
    left_extreme.modifier_thread_left = diff
    first.extreme_left = current.extreme_left
    first.modifier_extreme_left = (
        current.modifier_extreme_left + current.modifier_to_subtree - first.modifier_to_subtree
    )


def _set_right_thread(
    node: _TidyNode,
    current_index: int,
    target: _TidyNode,
    modifier: float,
) -> None:
    """Attach a right thread after the right contour finishes first.

    Parameters
    ----------
    node : _TidyNode
        Parent of the sibling subtrees.
    current_index : int
        Current right child index.
    target : _TidyNode
        Contour node to thread to.
    modifier : float
        Current contour modifier sum.

    Returns
    -------
    None
        Thread and extreme fields are updated in place.
    """
    current = node.children[current_index]
    diff = modifier - current.modifier_extreme_right - current.modifier_to_subtree
    right_extreme = _extreme_right(current)
    right_extreme.thread_right = target
    right_extreme.modifier_thread_right = diff
    previous = node.children[current_index - 1]
    current.extreme_right = previous.extreme_right
    current.modifier_extreme_right = (
        previous.modifier_extreme_right + previous.modifier_to_subtree - current.modifier_to_subtree
    )


def _separate(
    node: _TidyNode,
    child_index: int,
    y_list: _LinkedYList,
    peer_margin: float,
) -> _LinkedYList:
    """Separate adjacent sibling subtrees using tidy contours.

    Parameters
    ----------
    node : _TidyNode
        Parent node.
    child_index : int
        Index of the right sibling currently being apportioned.
    y_list : _LinkedYList
        Active y-list from prior siblings.
    peer_margin : float
        Minimum horizontal gap between overlapping contour boxes.

    Returns
    -------
    _LinkedYList
        Updated y-list.
    """
    left = _TidyContour(
        is_left=False,
        current=node.children[child_index - 1],
        modifier_sum=node.children[child_index - 1].modifier_to_subtree,
    )
    right = _TidyContour(
        is_left=True,
        current=node.children[child_index],
        modifier_sum=node.children[child_index].modifier_to_subtree,
    )
    while left.current is not None and right.current is not None:
        while _contour_bottom(left) > _linked_y_bottom(y_list):
            if y_list.next is None:
                raise RuntimeError("tidy y-list ended before the active contour.")
            y_list = y_list.next

        distance = _contour_right(left) - _contour_left(right) + peer_margin
        if distance > 0.0:
            right.modifier_sum += distance
            _move_subtree(node, child_index, y_list.index, distance)

        left_bottom = _contour_bottom(left)
        right_bottom = _contour_bottom(right)
        if left_bottom <= right_bottom:
            _contour_next(left)
        if left_bottom >= right_bottom:
            _contour_next(right)

    if left.current is None and right.current is not None:
        _set_left_thread(node, child_index, right.current, right.modifier_sum)
    elif left.current is not None and right.current is None:
        _set_right_thread(node, child_index, left.current, left.modifier_sum)
    return y_list


def _first_walk(node: _TidyNode, peer_margin: float) -> None:
    """Compute tidy relative x coordinates and subtree modifiers.

    Parameters
    ----------
    node : _TidyNode
        Root of the subtree.
    peer_margin : float
        Minimum sibling subtree gap.

    Returns
    -------
    None
        Tidy modifier fields are updated in place.
    """
    if not node.children:
        _set_extreme(node)
        return

    _first_walk(node.children[0], peer_margin)
    right_extreme = _extreme_right(node.children[0])
    y_list = _LinkedYList(index=0, y=right_extreme.y + right_extreme.height)
    for index in range(1, len(node.children)):
        current_child = node.children[index]
        _first_walk(current_child, peer_margin)
        max_y = _extreme_left(current_child).y + _extreme_left(current_child).height
        y_list = _separate(node, index, y_list, peer_margin)
        y_list = _linked_y_update(y_list, index, max_y)

    _position_root(node)
    _set_extreme(node)


def _second_walk(node: _TidyNode, modifier_sum: float) -> None:
    """Resolve final absolute x positions from tidy modifiers.

    Parameters
    ----------
    node : _TidyNode
        Root of the subtree.
    modifier_sum : float
        Accumulated modifier from ancestors.

    Returns
    -------
    None
        ``x`` fields are updated in place.
    """
    next_modifier_sum = modifier_sum + node.modifier_to_subtree
    node.x = node.relative_x + next_modifier_sum
    _add_child_spacing(node)
    for child in node.children:
        _second_walk(child, next_modifier_sum)


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
        _init_tidy_tree(root)
        _assign_y(root, config.parent_child_margin)
        _first_walk(root, config.peer_margin)
        _second_walk(root, 0.0)
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
