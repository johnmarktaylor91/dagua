"""d3-hierarchy tree and cluster source ports."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

import torch

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

Separation = Callable[["D3HierarchyNode", "D3HierarchyNode"], float]

_DEFAULT_DX = 1.0
_DEFAULT_DY = 1.0


@dataclass
class D3HierarchyNode:
    """Hierarchy node matching the d3-hierarchy fields used by tree layouts.

    Parameters
    ----------
    index : int
        Original graph node index.
    parent : D3HierarchyNode | None, optional
        Parent hierarchy node.
    children : list[D3HierarchyNode]
        Ordered child nodes.
    depth : int, default=0
        Root-relative depth.
    x : float, default=0.0
        Horizontal or angular coordinate.
    y : float, default=0.0
        Vertical or radial coordinate.
    """

    index: int
    parent: Optional["D3HierarchyNode"] = None
    children: List["D3HierarchyNode"] = field(default_factory=list)
    depth: int = 0
    x: float = 0.0
    y: float = 0.0


@dataclass
class _TreeNode:
    """Internal Buchheim-Walker wrapper mirroring d3's ``TreeNode``.

    Parameters
    ----------
    node : D3HierarchyNode | None
        Wrapped hierarchy node. The artificial parent stores ``None``.
    i : int
        Sibling index.
    parent : _TreeNode | None, optional
        Parent wrapper.
    children : list[_TreeNode] | None, optional
        Ordered child wrappers.
    A : _TreeNode | None, optional
        Default ancestor slot used by d3's apportion pass.
    a : _TreeNode | None, optional
        Ancestor pointer.
    z : float, default=0.0
        Preliminary coordinate.
    m : float, default=0.0
        Modifier.
    c : float, default=0.0
        Change accumulator.
    s : float, default=0.0
        Shift accumulator.
    t : _TreeNode | None, optional
        Thread pointer.
    """

    node: Optional[D3HierarchyNode]
    i: int
    parent: Optional["_TreeNode"] = None
    children: Optional[List["_TreeNode"]] = None
    A: Optional["_TreeNode"] = None
    a: Optional["_TreeNode"] = None
    z: float = 0.0
    m: float = 0.0
    c: float = 0.0
    s: float = 0.0
    t: Optional["_TreeNode"] = None

    def __post_init__(self) -> None:
        """Initialize d3's self-ancestor default.

        Returns
        -------
        None
            The instance is updated in place.
        """
        if self.a is None:
            self.a = self


def default_separation(left: D3HierarchyNode, right: D3HierarchyNode) -> float:
    """Return d3-hierarchy's default tree-node separation.

    Parameters
    ----------
    left : D3HierarchyNode
        First hierarchy node.
    right : D3HierarchyNode
        Second hierarchy node.

    Returns
    -------
    float
        ``1`` for siblings and ``2`` otherwise.
    """
    return 1.0 if left.parent is right.parent else 2.0


def _empty_edge_index(edge_index: torch.Tensor) -> bool:
    """Return whether an edge-index tensor has no edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Edge tensor with shape ``[2, E]``.

    Returns
    -------
    bool
        ``True`` when no edges are present.
    """
    return edge_index.numel() == 0 or int(edge_index.shape[1]) == 0


def build_d3_hierarchy(edge_index: torch.Tensor, num_nodes: int) -> D3HierarchyNode:
    """Build d3-style ordered hierarchy from directed graph edges.

    Parameters
    ----------
    edge_index : torch.Tensor
        Directed parent-child edges with shape ``[2, E]``. Edges are consumed
        in column order to preserve d3 child order. Nodes with an existing
        parent ignore later incoming edges, yielding a deterministic spanning
        hierarchy for non-tree graphs.
    num_nodes : int
        Number of graph nodes.

    Returns
    -------
    D3HierarchyNode
        Root hierarchy node. Forest roots after the first are attached under
        the first root so every input node receives coordinates.
    """
    if num_nodes <= 0:
        raise ValueError("d3 hierarchy layouts require at least one node.")

    children_by_parent: Dict[int, List[int]] = {node: [] for node in range(num_nodes)}
    parent_by_child: Dict[int, int] = {}
    if not _empty_edge_index(edge_index):
        cpu_edges = edge_index.detach().cpu()
        for edge_id in range(int(cpu_edges.shape[1])):
            source = int(cpu_edges[0, edge_id].item())
            target = int(cpu_edges[1, edge_id].item())
            if source == target or source < 0 or target < 0:
                continue
            if source >= num_nodes or target >= num_nodes:
                continue
            if target in parent_by_child:
                continue
            parent_by_child[target] = source
            children_by_parent[source].append(target)

    roots = [node for node in range(num_nodes) if node not in parent_by_child]
    if not roots:
        roots = [0]
        parent_by_child.pop(0, None)
    root_index = roots[0]
    for extra_root in roots[1:]:
        if extra_root != root_index:
            parent_by_child[extra_root] = root_index
            children_by_parent[root_index].append(extra_root)

    nodes: Dict[int, D3HierarchyNode] = {}
    visiting: set[int] = set()

    def build(index: int, parent: Optional[D3HierarchyNode], depth: int) -> D3HierarchyNode:
        """Recursively create hierarchy nodes while cutting cycles.

        Parameters
        ----------
        index : int
            Current graph node index.
        parent : D3HierarchyNode | None
            Parent hierarchy node.
        depth : int
            Current hierarchy depth.

        Returns
        -------
        D3HierarchyNode
            Constructed hierarchy node.
        """
        node = D3HierarchyNode(index=index, parent=parent, depth=depth)
        nodes[index] = node
        visiting.add(index)
        for child_index in children_by_parent[index]:
            if child_index in visiting:
                continue
            node.children.append(build(child_index, node, depth + 1))
        visiting.remove(index)
        return node

    root = build(root_index, None, 0)
    for index in range(num_nodes):
        if index not in nodes:
            root.children.append(build(index, root, 1))
    return root


def hierarchy_nodes_preorder(root: D3HierarchyNode) -> List[D3HierarchyNode]:
    """Return hierarchy nodes in d3 ``eachBefore`` order.

    Parameters
    ----------
    root : D3HierarchyNode
        Root hierarchy node.

    Returns
    -------
    list[D3HierarchyNode]
        Preorder node list.
    """
    ordered: List[D3HierarchyNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        ordered.append(node)
        stack.extend(reversed(node.children))
    return ordered


def _postorder(root: D3HierarchyNode) -> List[D3HierarchyNode]:
    """Return hierarchy nodes in d3 ``eachAfter`` order.

    Parameters
    ----------
    root : D3HierarchyNode
        Root hierarchy node.

    Returns
    -------
    list[D3HierarchyNode]
        Postorder node list.
    """
    discovered: List[D3HierarchyNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        discovered.append(node)
        stack.extend(node.children)
    return list(reversed(discovered))


def _tree_root(root: D3HierarchyNode) -> _TreeNode:
    """Build the d3 internal tree wrapper and artificial parent.

    Parameters
    ----------
    root : D3HierarchyNode
        Public hierarchy root.

    Returns
    -------
    _TreeNode
        Wrapped root with an artificial parent.
    """
    wrapper_by_node: Dict[int, _TreeNode] = {}

    def wrap(node: D3HierarchyNode, sibling_index: int, parent: Optional[_TreeNode]) -> _TreeNode:
        """Wrap a hierarchy node and descendants.

        Parameters
        ----------
        node : D3HierarchyNode
            Node to wrap.
        sibling_index : int
            Sibling index.
        parent : _TreeNode | None
            Wrapped parent.

        Returns
        -------
        _TreeNode
            Wrapped tree node.
        """
        wrapped = _TreeNode(node=node, i=sibling_index, parent=parent)
        wrapper_by_node[node.index] = wrapped
        if node.children:
            wrapped.children = [
                wrap(child, child_index, wrapped) for child_index, child in enumerate(node.children)
            ]
        return wrapped

    tree = wrap(root, 0, None)
    artificial = _TreeNode(node=None, i=0, children=[tree])
    tree.parent = artificial
    return tree


def _next_left(node: _TreeNode) -> Optional[_TreeNode]:
    """Return d3's left contour successor.

    Parameters
    ----------
    node : _TreeNode
        Current internal tree node.

    Returns
    -------
    _TreeNode | None
        Leftmost child or thread.
    """
    return node.children[0] if node.children else node.t


def _next_right(node: _TreeNode) -> Optional[_TreeNode]:
    """Return d3's right contour successor.

    Parameters
    ----------
    node : _TreeNode
        Current internal tree node.

    Returns
    -------
    _TreeNode | None
        Rightmost child or thread.
    """
    return node.children[-1] if node.children else node.t


def _move_subtree(left: _TreeNode, right: _TreeNode, shift: float) -> None:
    """Apply d3's subtree shift bookkeeping.

    Parameters
    ----------
    left : _TreeNode
        Left sibling subtree.
    right : _TreeNode
        Right sibling subtree.
    shift : float
        Required positive contour shift.

    Returns
    -------
    None
        Tree-node fields are updated in place.
    """
    change = shift / float(right.i - left.i)
    right.c -= change
    right.s += shift
    left.c += change
    right.z += shift
    right.m += shift


def _execute_shifts(node: _TreeNode) -> None:
    """Execute d3's accumulated sibling shifts.

    Parameters
    ----------
    node : _TreeNode
        Internal node whose children have shift/change accumulators.

    Returns
    -------
    None
        Child preliminary coordinates are updated in place.
    """
    shift = 0.0
    change = 0.0
    if node.children is None:
        return
    for child in reversed(node.children):
        child.z += shift
        child.m += shift
        change += child.c
        shift += child.s + change


def _next_ancestor(left_inner: _TreeNode, node: _TreeNode, ancestor: _TreeNode) -> _TreeNode:
    """Return d3's greatest uncommon ancestor fallback.

    Parameters
    ----------
    left_inner : _TreeNode
        Current right contour node from the left subtree.
    node : _TreeNode
        Current subtree root.
    ancestor : _TreeNode
        Default ancestor.

    Returns
    -------
    _TreeNode
        Ancestor selected by d3.
    """
    if left_inner.a is not None and left_inner.a.parent is node.parent:
        return left_inner.a
    return ancestor


def _apportion(
    node: _TreeNode,
    previous: Optional[_TreeNode],
    ancestor: _TreeNode,
    separation: Separation,
) -> _TreeNode:
    """Run d3's Buchheim-Walker apportion step.

    Parameters
    ----------
    node : _TreeNode
        Current internal subtree root.
    previous : _TreeNode | None
        Previous sibling subtree root.
    ancestor : _TreeNode
        Current default ancestor.
    separation : callable
        d3-compatible separation function.

    Returns
    -------
    _TreeNode
        Updated default ancestor.
    """
    if previous is None:
        return ancestor
    vip = node
    vop = node
    vim = previous
    if vip.parent is None or vip.parent.children is None:
        return ancestor
    vom = vip.parent.children[0]
    sip = vip.m
    sop = vop.m
    sim = vim.m
    som = vom.m
    while True:
        next_vim = _next_right(vim)
        next_vip = _next_left(vip)
        if next_vim is None or next_vip is None:
            break
        vim = next_vim
        vip = next_vip
        next_vom = _next_left(vom)
        next_vop = _next_right(vop)
        if next_vom is None or next_vop is None:
            break
        vom = next_vom
        vop = next_vop
        vop.a = node
        if vim.node is None or vip.node is None:
            raise RuntimeError("d3 tree contour unexpectedly reached artificial node.")
        shift = vim.z + sim - vip.z - sip + separation(vim.node, vip.node)
        if shift > 0.0:
            _move_subtree(_next_ancestor(vim, node, ancestor), node, shift)
            sip += shift
            sop += shift
        sim += vim.m
        sip += vip.m
        som += vom.m
        sop += vop.m
    if _next_right(vim) is not None and _next_right(vop) is None:
        vop.t = _next_right(vim)
        vop.m += sim - sop
    if _next_left(vip) is not None and _next_left(vom) is None:
        vom.t = _next_left(vip)
        vom.m += sip - som
        ancestor = node
    return ancestor


def _tree_nodes_postorder(root: _TreeNode) -> List[_TreeNode]:
    """Return internal tree wrappers in postorder.

    Parameters
    ----------
    root : _TreeNode
        Wrapped tree root.

    Returns
    -------
    list[_TreeNode]
        Postorder wrapper list.
    """
    discovered: List[_TreeNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        discovered.append(node)
        if node.children:
            stack.extend(node.children)
    return list(reversed(discovered))


def _tree_nodes_preorder(root: _TreeNode) -> List[_TreeNode]:
    """Return internal tree wrappers in preorder.

    Parameters
    ----------
    root : _TreeNode
        Wrapped tree root.

    Returns
    -------
    list[_TreeNode]
        Preorder wrapper list.
    """
    ordered: List[_TreeNode] = []
    stack = [root]
    while stack:
        node = stack.pop()
        ordered.append(node)
        if node.children:
            stack.extend(reversed(node.children))
    return ordered


def layout_d3_tree_nodes(
    root: D3HierarchyNode,
    *,
    dx: float = _DEFAULT_DX,
    dy: float = _DEFAULT_DY,
    node_size: bool = True,
    separation: Separation = default_separation,
) -> None:
    """Assign d3 ``tree`` coordinates to hierarchy nodes.

    Parameters
    ----------
    root : D3HierarchyNode
        Root hierarchy node.
    dx : float, default=1.0
        d3 horizontal size or node-size scale.
    dy : float, default=1.0
        d3 vertical size or node-size scale.
    node_size : bool, default=True
        Whether to use ``tree.nodeSize([dx, dy])`` semantics. ``False`` uses
        ``tree.size([dx, dy])`` normalization.
    separation : callable, default=default_separation
        d3-compatible separation function.

    Returns
    -------
    None
        Coordinates are written to ``root`` descendants.
    """
    tree = _tree_root(root)

    def first_walk(node: _TreeNode) -> None:
        """Compute d3 preliminary coordinates for one internal node.

        Parameters
        ----------
        node : _TreeNode
            Current internal node.

        Returns
        -------
        None
            Internal tree fields are updated in place.
        """
        if node.parent is None or node.parent.children is None:
            return
        siblings = node.parent.children
        previous = siblings[node.i - 1] if node.i else None
        if node.children:
            _execute_shifts(node)
            midpoint = (node.children[0].z + node.children[-1].z) / 2.0
            if previous is not None:
                if node.node is None or previous.node is None:
                    raise RuntimeError("d3 tree first walk reached artificial node.")
                node.z = previous.z + separation(node.node, previous.node)
                node.m = node.z - midpoint
            else:
                node.z = midpoint
        elif previous is not None:
            if node.node is None or previous.node is None:
                raise RuntimeError("d3 tree first walk reached artificial node.")
            node.z = previous.z + separation(node.node, previous.node)
        node.parent.A = _apportion(node, previous, node.parent.A or siblings[0], separation)

    for wrapped in _tree_nodes_postorder(tree):
        first_walk(wrapped)
    if tree.parent is None:
        raise RuntimeError("d3 tree root is missing artificial parent.")
    tree.parent.m = -tree.z
    for wrapped in _tree_nodes_preorder(tree):
        if wrapped.node is None or wrapped.parent is None:
            continue
        wrapped.node.x = wrapped.z + wrapped.parent.m
        wrapped.m += wrapped.parent.m

    if node_size:
        for node in hierarchy_nodes_preorder(root):
            node.x *= dx
            node.y = node.depth * dy
        return

    ordered = hierarchy_nodes_preorder(root)
    left = min(ordered, key=lambda item: item.x)
    right = max(ordered, key=lambda item: item.x)
    bottom = max(ordered, key=lambda item: item.depth)
    scale = 1.0 if left is right else separation(left, right) / 2.0
    tx = scale - left.x
    kx = dx / (right.x + scale + tx)
    ky = dy / float(bottom.depth or 1)
    for node in ordered:
        node.x = (node.x + tx) * kx
        node.y = node.depth * ky


def _mean_x(children: Sequence[D3HierarchyNode]) -> float:
    """Return d3 cluster mean child x coordinate.

    Parameters
    ----------
    children : sequence[D3HierarchyNode]
        Child hierarchy nodes.

    Returns
    -------
    float
        Mean ``x`` coordinate.
    """
    return sum(child.x for child in children) / float(len(children))


def _max_y(children: Sequence[D3HierarchyNode]) -> float:
    """Return d3 cluster max child y plus one.

    Parameters
    ----------
    children : sequence[D3HierarchyNode]
        Child hierarchy nodes.

    Returns
    -------
    float
        Maximum child ``y`` plus one.
    """
    return 1.0 + max(child.y for child in children)


def _leaf_left(node: D3HierarchyNode) -> D3HierarchyNode:
    """Return the leftmost leaf descendant.

    Parameters
    ----------
    node : D3HierarchyNode
        Starting hierarchy node.

    Returns
    -------
    D3HierarchyNode
        Leftmost leaf.
    """
    while node.children:
        node = node.children[0]
    return node


def _leaf_right(node: D3HierarchyNode) -> D3HierarchyNode:
    """Return the rightmost leaf descendant.

    Parameters
    ----------
    node : D3HierarchyNode
        Starting hierarchy node.

    Returns
    -------
    D3HierarchyNode
        Rightmost leaf.
    """
    while node.children:
        node = node.children[-1]
    return node


def layout_d3_cluster_nodes(
    root: D3HierarchyNode,
    *,
    dx: float = _DEFAULT_DX,
    dy: float = _DEFAULT_DY,
    node_size: bool = False,
    separation: Separation = default_separation,
) -> None:
    """Assign d3 ``cluster`` coordinates to hierarchy nodes.

    Parameters
    ----------
    root : D3HierarchyNode
        Root hierarchy node.
    dx : float, default=1.0
        d3 horizontal size or node-size scale.
    dy : float, default=1.0
        d3 vertical size or node-size scale.
    node_size : bool, default=False
        Whether to use ``cluster.nodeSize([dx, dy])`` semantics. ``False``
        uses d3's default ``cluster.size([dx, dy])``.
    separation : callable, default=default_separation
        d3-compatible separation function.

    Returns
    -------
    None
        Coordinates are written to ``root`` descendants.
    """
    previous_node: Optional[D3HierarchyNode] = None
    x_coord = 0.0
    for node in _postorder(root):
        if node.children:
            node.x = _mean_x(node.children)
            node.y = _max_y(node.children)
        else:
            if previous_node is not None:
                x_coord += separation(node, previous_node)
            node.x = x_coord
            node.y = 0.0
            previous_node = node

    left = _leaf_left(root)
    right = _leaf_right(root)
    x0 = left.x - separation(left, right) / 2.0
    x1 = right.x + separation(right, left) / 2.0
    for node in _postorder(root):
        if node_size:
            node.x = (node.x - root.x) * dx
            node.y = (root.y - node.y) * dy
        else:
            node.x = (node.x - x0) / (x1 - x0) * dx
            node.y = (1.0 - (node.y / root.y if root.y else 1.0)) * dy


def hierarchy_to_tensor(
    root: D3HierarchyNode,
    num_nodes: int,
    *,
    radial: bool = False,
    dtype: torch.dtype = torch.float64,
) -> torch.Tensor:
    """Convert hierarchy coordinates to an ``[N, 2]`` tensor.

    Parameters
    ----------
    root : D3HierarchyNode
        Root hierarchy node with assigned coordinates.
    num_nodes : int
        Number of original graph nodes.
    radial : bool, default=False
        Whether to interpret ``x`` as radians and ``y`` as radius.
    dtype : torch.dtype, default=torch.float64
        Output floating dtype.

    Returns
    -------
    torch.Tensor
        Position tensor with shape ``[N, 2]``.
    """
    positions = torch.zeros((num_nodes, 2), dtype=dtype)
    for node in hierarchy_nodes_preorder(root):
        if radial:
            angle = node.x - math.pi / 2.0
            positions[node.index, 0] = node.y * math.cos(angle)
            positions[node.index, 1] = node.y * math.sin(angle)
        else:
            positions[node.index, 0] = node.x
            positions[node.index, 1] = node.y
    return positions


@register_op
class D3TreeLayout(Op):
    """Composable op for d3-hierarchy ``tree`` layout."""

    name = "d3_tree_layout"
    category = OpCategory.COORDINATE
    reads = ("edge_index",)
    writes = ("pos",)

    def __init__(
        self,
        *,
        dx: float = _DEFAULT_DX,
        dy: float = _DEFAULT_DY,
        node_size: bool = True,
        radial: bool = False,
    ) -> None:
        """Initialize the d3 tree op.

        Parameters
        ----------
        dx : float, default=1.0
            d3 horizontal size or node-size scale.
        dy : float, default=1.0
            d3 vertical size or node-size scale.
        node_size : bool, default=True
            Whether to use ``tree.nodeSize([dx, dy])`` semantics.
        radial : bool, default=False
            Whether to apply the d3 radial polar transform.

        Returns
        -------
        None
            Instance fields are initialized.
        """
        self.dx = float(dx)
        self.dy = float(dy)
        self.node_size = bool(node_size)
        self.radial = bool(radial)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run d3 tree coordinate assignment.

        Parameters
        ----------
        problem : LayoutProblem
            Graph layout problem with ``edge_index`` shape ``[2, E]``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context, accepted for op API compatibility.

        Returns
        -------
        SolveState
            State with ``pos`` set to d3 tree coordinates.
        """
        del ctx
        root = build_d3_hierarchy(problem.edge_index, problem.num_nodes)
        layout_d3_tree_nodes(root, dx=self.dx, dy=self.dy, node_size=self.node_size)
        state.pos = hierarchy_to_tensor(root, problem.num_nodes, radial=self.radial)
        state.extras["d3_hierarchy_root"] = root
        return state


@register_op
class D3ClusterLayout(Op):
    """Composable op for d3-hierarchy ``cluster`` layout."""

    name = "d3_cluster_layout"
    category = OpCategory.COORDINATE
    reads = ("edge_index",)
    writes = ("pos",)

    def __init__(
        self,
        *,
        dx: float = _DEFAULT_DX,
        dy: float = _DEFAULT_DY,
        node_size: bool = False,
        radial: bool = False,
    ) -> None:
        """Initialize the d3 cluster op.

        Parameters
        ----------
        dx : float, default=1.0
            d3 horizontal size or node-size scale.
        dy : float, default=1.0
            d3 vertical size or node-size scale.
        node_size : bool, default=False
            Whether to use ``cluster.nodeSize([dx, dy])`` semantics.
        radial : bool, default=False
            Whether to apply the d3 radial polar transform.

        Returns
        -------
        None
            Instance fields are initialized.
        """
        self.dx = float(dx)
        self.dy = float(dy)
        self.node_size = bool(node_size)
        self.radial = bool(radial)

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run d3 cluster coordinate assignment.

        Parameters
        ----------
        problem : LayoutProblem
            Graph layout problem with ``edge_index`` shape ``[2, E]``.
        state : SolveState
            Mutable solve state.
        ctx : RuntimeContext
            Runtime context, accepted for op API compatibility.

        Returns
        -------
        SolveState
            State with ``pos`` set to d3 cluster coordinates.
        """
        del ctx
        root = build_d3_hierarchy(problem.edge_index, problem.num_nodes)
        layout_d3_cluster_nodes(root, dx=self.dx, dy=self.dy, node_size=self.node_size)
        state.pos = hierarchy_to_tensor(root, problem.num_nodes, radial=self.radial)
        state.extras["d3_hierarchy_root"] = root
        return state


__all__ = [
    "D3ClusterLayout",
    "D3HierarchyNode",
    "D3TreeLayout",
    "build_d3_hierarchy",
    "default_separation",
    "hierarchy_nodes_preorder",
    "hierarchy_to_tensor",
    "layout_d3_cluster_nodes",
    "layout_d3_tree_nodes",
]
