"""Pure Python igraph-compatible Reingold-Tilford tree layout."""

from __future__ import annotations

import sys
from collections import deque
from dataclasses import dataclass
from typing import Optional, Sequence

import torch


@dataclass
class _RtVertex:
    """Contour state for igraph's Reingold-Tilford implementation."""

    parent: int = -1
    level: int = -1
    offset: float = 0.0
    left_contour: int = -1
    right_contour: int = -1
    offset_to_left_contour: float = 0.0
    offset_to_right_contour: float = 0.0
    left_extreme: int = -1
    right_extreme: int = -1
    offset_to_left_extreme: float = 0.0
    offset_to_right_extreme: float = 0.0


def _validated_edge_list(edge_index: torch.Tensor, num_nodes: int) -> list[tuple[int, int]]:
    """Return validated CPU edge pairs.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph vertices.

    Returns
    -------
    list[tuple[int, int]]
        Edge pairs as Python integers.

    Raises
    ------
    ValueError
        If the edge tensor shape or vertex ids are invalid.
    """
    if edge_index.ndim != 2 or edge_index.shape[0] != 2:
        raise ValueError("edge_index must have shape [2, E].")
    edge_cpu = edge_index.detach().to(device="cpu", dtype=torch.long)
    edges: list[tuple[int, int]] = []
    for edge_id in range(edge_cpu.shape[1]):
        source = int(edge_cpu[0, edge_id].item())
        target = int(edge_cpu[1, edge_id].item())
        if source < 0 or source >= num_nodes or target < 0 or target >= num_nodes:
            raise ValueError(
                f"edge_index contains vertex outside [0, {num_nodes}): {(source, target)}"
            )
        edges.append((source, target))
    return edges


def _adjacency(
    edges: Sequence[tuple[int, int]],
    num_nodes: int,
    traversal_mode: str,
) -> list[list[int]]:
    """Build igraph-style traversal adjacency without loops or parallel edges.

    Parameters
    ----------
    edges : sequence of tuple[int, int]
        Directed edge pairs.
    num_nodes : int
        Number of graph vertices.
    traversal_mode : str
        Traversal mode: ``"out"``, ``"in"``, or ``"all"``.

    Returns
    -------
    list[list[int]]
        Sorted unique neighbor ids for each vertex.
    """
    neighbors: list[set[int]] = [set() for _ in range(num_nodes)]
    for source, target in edges:
        if source == target:
            continue
        if traversal_mode == "out":
            neighbors[source].add(target)
        elif traversal_mode == "in":
            neighbors[target].add(source)
        elif traversal_mode == "all":
            neighbors[source].add(target)
            neighbors[target].add(source)
        else:
            raise ValueError(f"Unsupported Reingold-Tilford traversal_mode: {traversal_mode!r}")
    return [sorted(node_neighbors) for node_neighbors in neighbors]


def _degrees(
    edges: Sequence[tuple[int, int]],
    num_nodes: int,
    traversal_mode: str,
) -> list[int]:
    """Count degrees for igraph's automatic root sorting.

    Parameters
    ----------
    edges : sequence of tuple[int, int]
        Directed edge pairs. Parallel edges are counted; loops are ignored.
    num_nodes : int
        Number of graph vertices.
    traversal_mode : str
        Degree mode: ``"out"``, ``"in"``, or ``"all"``.

    Returns
    -------
    list[int]
        Degree per vertex.
    """
    degrees = [0] * num_nodes
    for source, target in edges:
        if source == target:
            continue
        if traversal_mode == "out":
            degrees[source] += 1
        elif traversal_mode == "in":
            degrees[target] += 1
        elif traversal_mode == "all":
            degrees[source] += 1
            degrees[target] += 1
        else:
            raise ValueError(f"Unsupported Reingold-Tilford traversal_mode: {traversal_mode!r}")
    return degrees


def _sort_indices_like_igraph(values: Sequence[int], descending: bool) -> list[int]:
    """Sort indices with igraph's non-stable qsort tie behavior.

    Parameters
    ----------
    values : sequence of int
        Values indexed by vertex id.
    descending : bool
        Whether larger values should sort first.

    Returns
    -------
    list[int]
        Vertex ids ordered like ``igraph_vector_int_sort_ind``.
    """
    indices = list(range(len(values)))

    def compare(left: int, right: int) -> int:
        """Compare two vertex ids by their associated values.

        Parameters
        ----------
        left : int
            Left vertex id.
        right : int
            Right vertex id.

        Returns
        -------
        int
            Negative, zero, or positive comparison result.
        """
        if values[left] < values[right]:
            return 1 if descending else -1
        if values[left] > values[right]:
            return -1 if descending else 1
        return 0

    def med3(first: int, middle: int, last: int) -> int:
        """Return the median-of-three pivot position used by igraph qsort.

        Parameters
        ----------
        first : int
            First candidate position.
        middle : int
            Middle candidate position.
        last : int
            Last candidate position.

        Returns
        -------
        int
            Pivot position.
        """
        if compare(indices[first], indices[middle]) < 0:
            if compare(indices[middle], indices[last]) < 0:
                return middle
            if compare(indices[first], indices[last]) < 0:
                return last
            return first
        if compare(indices[middle], indices[last]) > 0:
            return middle
        if compare(indices[first], indices[last]) < 0:
            return first
        return last

    def insertion_sort(start: int, count: int) -> None:
        """Sort a small partition with igraph's insertion-sort fallback.

        Parameters
        ----------
        start : int
            Partition start index.
        count : int
            Partition length.

        Returns
        -------
        None
            ``indices`` is modified in place.
        """
        for cursor in range(start + 1, start + count):
            position = cursor
            while position > start and compare(indices[position - 1], indices[position]) > 0:
                indices[position], indices[position - 1] = indices[position - 1], indices[position]
                position -= 1

    def quicksort(start: int, count: int) -> None:
        """Run igraph's Bentley-McIlroy qsort variant on one partition.

        Parameters
        ----------
        start : int
            Partition start index.
        count : int
            Partition length.

        Returns
        -------
        None
            ``indices`` is modified in place.
        """
        while True:
            if count < 2:
                return
            if count < 7:
                insertion_sort(start=start, count=count)
                return

            pivot = start + count // 2
            if count > 7:
                left = start
                right = start + count - 1
                if count > 40:
                    stride = count // 8
                    left = med3(left, left + stride, left + 2 * stride)
                    pivot = med3(pivot - stride, pivot, pivot + stride)
                    right = med3(right - 2 * stride, right - stride, right)
                pivot = med3(left, pivot, right)

            indices[start], indices[pivot] = indices[pivot], indices[start]
            pa = start + 1
            pb = start + 1
            pc = start + count - 1
            pd = start + count - 1
            swap_count = 0

            while True:
                while pb <= pc:
                    comparison = compare(indices[pb], indices[start])
                    if comparison > 0:
                        break
                    if comparison == 0:
                        swap_count = 1
                        indices[pa], indices[pb] = indices[pb], indices[pa]
                        pa += 1
                    pb += 1
                while pb <= pc:
                    comparison = compare(indices[pc], indices[start])
                    if comparison < 0:
                        break
                    if comparison == 0:
                        swap_count = 1
                        indices[pc], indices[pd] = indices[pd], indices[pc]
                        pd -= 1
                    pc -= 1
                if pb > pc:
                    break
                indices[pb], indices[pc] = indices[pc], indices[pb]
                swap_count = 1
                pb += 1
                pc -= 1

            if swap_count == 0:
                insertion_sort(start=start, count=count)
                return

            left_equal = min(pa - start, pb - pa)
            for offset in range(left_equal):
                left_pos = start + offset
                right_pos = pb - left_equal + offset
                indices[left_pos], indices[right_pos] = indices[right_pos], indices[left_pos]

            right_equal = min(pd - pc, start + count - pd - 1)
            for offset in range(right_equal):
                left_pos = pb + offset
                right_pos = start + count - right_equal + offset
                indices[left_pos], indices[right_pos] = indices[right_pos], indices[left_pos]

            left_count = pb - pa
            right_count = pd - pc
            if left_count <= right_count:
                if left_count > 1:
                    quicksort(start=start, count=left_count)
                if right_count <= 1:
                    return
                start = start + count - right_count
                count = right_count
            else:
                if right_count > 1:
                    quicksort(start=start + count - right_count, count=right_count)
                if left_count <= 1:
                    return
                count = left_count

    quicksort(start=0, count=len(indices))
    return indices


def _weak_components(edges: Sequence[tuple[int, int]], num_nodes: int) -> tuple[list[int], int]:
    """Compute weak component membership.

    Parameters
    ----------
    edges : sequence of tuple[int, int]
        Directed edge pairs.
    num_nodes : int
        Number of graph vertices.

    Returns
    -------
    tuple[list[int], int]
        Component id per vertex and component count.
    """
    graph = _adjacency(edges=edges, num_nodes=num_nodes, traversal_mode="all")
    membership = [-1] * num_nodes
    component_count = 0
    for root in range(num_nodes):
        if membership[root] >= 0:
            continue
        membership[root] = component_count
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if membership[neighbor] >= 0:
                    continue
                membership[neighbor] = component_count
                queue.append(neighbor)
        component_count += 1
    return membership, component_count


def _strong_components(edges: Sequence[tuple[int, int]], num_nodes: int) -> tuple[list[int], int]:
    """Compute strong component membership.

    Parameters
    ----------
    edges : sequence of tuple[int, int]
        Directed edge pairs.
    num_nodes : int
        Number of graph vertices.

    Returns
    -------
    tuple[list[int], int]
        Component id per vertex and component count.
    """
    outgoing = _adjacency(edges=edges, num_nodes=num_nodes, traversal_mode="out")
    incoming = _adjacency(edges=edges, num_nodes=num_nodes, traversal_mode="in")
    visited = [False] * num_nodes
    order: list[int] = []
    for root in range(num_nodes):
        if visited[root]:
            continue
        stack: list[tuple[int, bool]] = [(root, False)]
        while stack:
            node, expanded = stack.pop()
            if expanded:
                order.append(node)
                continue
            if visited[node]:
                continue
            visited[node] = True
            stack.append((node, True))
            for neighbor in reversed(outgoing[node]):
                if not visited[neighbor]:
                    stack.append((neighbor, False))

    components: list[list[int]] = []
    assigned = [False] * num_nodes
    for root in reversed(order):
        if assigned[root]:
            continue
        component: list[int] = []
        assigned[root] = True
        stack = [root]
        while stack:
            node = stack.pop()
            component.append(node)
            for neighbor in reversed(incoming[node]):
                if assigned[neighbor]:
                    continue
                assigned[neighbor] = True
                stack.append(neighbor)
        components.append(sorted(component))
    components.sort(key=lambda members: members[0])

    membership = [-1] * num_nodes
    for component_id, members in enumerate(components):
        for node in members:
            membership[node] = component_id
    return membership, len(components)


def _component_degrees(
    edges: Sequence[tuple[int, int]],
    membership: Sequence[int],
    component_count: int,
    traversal_mode: str,
) -> list[int]:
    """Count directed inter-component degrees.

    Parameters
    ----------
    edges : sequence of tuple[int, int]
        Directed edge pairs.
    membership : sequence of int
        Component id per vertex.
    component_count : int
        Number of components.
    traversal_mode : str
        ``"out"`` counts source components; ``"in"`` counts target components.

    Returns
    -------
    list[int]
        Inter-component degree per component.
    """
    degrees = [0] * component_count
    for source, target in edges:
        source_component = membership[source]
        target_component = membership[target]
        if source_component == target_component:
            continue
        component = source_component if traversal_mode == "out" else target_component
        degrees[component] += 1
    return degrees


def _eccentricities(
    edges: Sequence[tuple[int, int]],
    num_nodes: int,
    traversal_mode: str,
) -> list[int]:
    """Compute finite eccentricity for each vertex.

    Parameters
    ----------
    edges : sequence of tuple[int, int]
        Directed edge pairs.
    num_nodes : int
        Number of graph vertices.
    traversal_mode : str
        Traversal mode for shortest paths.

    Returns
    -------
    list[int]
        Maximum finite distance reachable from each vertex.
    """
    graph = _adjacency(edges=edges, num_nodes=num_nodes, traversal_mode=traversal_mode)
    values: list[int] = []
    for root in range(num_nodes):
        distances = [-1] * num_nodes
        distances[root] = 0
        maximum = 0
        queue: deque[int] = deque([root])
        while queue:
            node = queue.popleft()
            for neighbor in graph[node]:
                if distances[neighbor] >= 0:
                    continue
                distances[neighbor] = distances[node] + 1
                maximum = max(maximum, distances[neighbor])
                queue.append(neighbor)
        values.append(maximum)
    return values


def _select_roots(
    edges: Sequence[tuple[int, int]],
    num_nodes: int,
    traversal_mode: str,
) -> list[int]:
    """Select automatic roots with igraph's tree-layout heuristic.

    Parameters
    ----------
    edges : sequence of tuple[int, int]
        Directed edge pairs.
    num_nodes : int
        Number of graph vertices.
    traversal_mode : str
        Traversal mode.

    Returns
    -------
    list[int]
        Root vertices for the graph or forest.
    """
    if num_nodes == 0:
        return []
    if num_nodes < 500:
        values = _degrees(edges=edges, num_nodes=num_nodes, traversal_mode=traversal_mode)
        order = _sort_indices_like_igraph(values=values, descending=True)
    else:
        values = _eccentricities(edges=edges, num_nodes=num_nodes, traversal_mode=traversal_mode)
        order = _sort_indices_like_igraph(values=values, descending=False)

    if traversal_mode == "all":
        membership, component_count = _weak_components(edges=edges, num_nodes=num_nodes)
        roots = [-1] * component_count
        seen = 0
        for node in order:
            component = membership[node]
            if roots[component] >= 0:
                continue
            roots[component] = node
            seen += 1
            if seen == component_count:
                break
        return roots

    membership, component_count = _strong_components(edges=edges, num_nodes=num_nodes)
    reverse_mode = "in" if traversal_mode == "out" else "out"
    cluster_degrees = _component_degrees(
        edges=edges,
        membership=membership,
        component_count=component_count,
        traversal_mode=reverse_mode,
    )
    roots = [-1] * component_count
    for node in order:
        component = membership[node]
        if cluster_degrees[component] == 0 and roots[component] == -1:
            roots[component] = node
    return [root for root in roots if root >= 0]


def _validated_roots(roots: Optional[Sequence[int]], num_nodes: int) -> Optional[list[int]]:
    """Validate optional explicit roots.

    Parameters
    ----------
    roots : sequence of int | None
        Optional explicit root vertices.
    num_nodes : int
        Number of original graph vertices.

    Returns
    -------
    list[int] | None
        Root ids preserving caller order, or ``None``.

    Raises
    ------
    ValueError
        If a root id is outside the graph.
    """
    if roots is None:
        return None
    root_list = [int(root) for root in roots]
    for root in root_list:
        if root < 0 or root >= num_nodes:
            raise ValueError(f"Invalid Reingold-Tilford root vertex: {root}")
    return root_list


def _validated_rootlevel(
    rootlevel: Optional[Sequence[int]],
    roots: Optional[Sequence[int]],
) -> Optional[list[int]]:
    """Validate optional explicit root levels.

    Parameters
    ----------
    rootlevel : sequence of int | None
        Optional level for each explicit root.
    roots : sequence of int | None
        Explicit roots, if any.

    Returns
    -------
    list[int] | None
        Root levels, or ``None`` when absent.

    Raises
    ------
    ValueError
        If levels are negative or mismatched with multiple explicit roots.
    """
    if rootlevel is None:
        return None
    levels = [int(level) for level in rootlevel]
    if roots is not None and len(roots) > 1 and len(levels) != len(roots):
        raise ValueError("Reingold-Tilford roots and rootlevel lengths differ.")
    for level in levels:
        if level < 0:
            raise ValueError(f"Reingold-Tilford rootlevel must be non-negative: {level}")
    return levels


def _reachable(
    edges: Sequence[tuple[int, int]],
    num_nodes: int,
    traversal_mode: str,
    root: int,
) -> list[bool]:
    """Mark vertices reachable from ``root``.

    Parameters
    ----------
    edges : sequence of tuple[int, int]
        Directed edge pairs.
    num_nodes : int
        Number of graph vertices.
    traversal_mode : str
        Traversal mode.
    root : int
        BFS start vertex.

    Returns
    -------
    list[bool]
        Reachability mask.
    """
    graph = _adjacency(edges=edges, num_nodes=num_nodes, traversal_mode=traversal_mode)
    visited = [False] * num_nodes
    queue: deque[int] = deque([root])
    while queue:
        node = queue.popleft()
        if visited[node]:
            continue
        visited[node] = True
        for neighbor in graph[node]:
            if not visited[neighbor]:
                queue.append(neighbor)
    return visited


def _prepare_graph(
    edges: Sequence[tuple[int, int]],
    num_nodes: int,
    traversal_mode: str,
    roots: Optional[Sequence[int]],
    rootlevel: Optional[Sequence[int]],
) -> tuple[list[tuple[int, int]], int, int]:
    """Prepare igraph's extended graph and synthetic root.

    Parameters
    ----------
    edges : sequence of tuple[int, int]
        Original directed edge pairs.
    num_nodes : int
        Number of original graph vertices.
    traversal_mode : str
        Traversal mode.
    roots : sequence of int | None
        Optional explicit roots.
    rootlevel : sequence of int | None
        Optional root levels for explicit multi-root layouts.

    Returns
    -------
    tuple[list[tuple[int, int]], int, int]
        Extended edge list, extended vertex count, and real root id.
    """
    root_list = _validated_roots(roots=roots, num_nodes=num_nodes)
    levels = _validated_rootlevel(rootlevel=rootlevel, roots=root_list)
    if root_list is None or len(root_list) == 0:
        root_list = _select_roots(edges=edges, num_nodes=num_nodes, traversal_mode=traversal_mode)

    extended_edges = list(edges)
    extended_count = num_nodes
    if levels is not None and len(levels) > 0 and len(root_list) > 1:
        for root_index, level in enumerate(levels):
            if level == 0:
                continue
            root_node = root_list[root_index]
            if traversal_mode != "in":
                extended_edges.append((extended_count, root_node))
                for _ in range(level - 1):
                    extended_edges.append((extended_count + 1, extended_count))
                    extended_count += 1
            else:
                extended_edges.append((root_node, extended_count))
                for _ in range(level - 1):
                    extended_edges.append((extended_count, extended_count + 1))
                    extended_count += 1
            root_list[root_index] = extended_count
            extended_count += 1

    if len(root_list) == 1:
        real_root = root_list[0]
    else:
        real_root = extended_count
        extended_count += 1
        for root in root_list:
            extended_edges.append((real_root, root))

    visited = _reachable(
        edges=extended_edges,
        num_nodes=extended_count,
        traversal_mode=traversal_mode,
        root=real_root,
    )
    for node in range(extended_count):
        if visited[node]:
            continue
        if traversal_mode != "in":
            extended_edges.append((real_root, node))
        else:
            extended_edges.append((node, real_root))
    return extended_edges, extended_count, real_root


def _postorder(vertices: list[_RtVertex], node: int) -> None:
    """Run igraph's postorder contour placement for one subtree.

    Parameters
    ----------
    vertices : list[_RtVertex]
        Mutable per-vertex RT state.
    node : int
        Subtree root.

    Returns
    -------
    None
        Vertex contour state is modified in place.
    """
    childcount = 0
    for child in range(len(vertices)):
        if child != node and vertices[child].parent == node:
            childcount += 1
            _postorder(vertices=vertices, node=child)
    if childcount == 0:
        return

    minsep = 1.0
    leftroot = -1
    avg = 0.0
    sibling_index = 0
    for child in range(len(vertices)):
        if child == node or vertices[child].parent != node:
            continue
        if leftroot >= 0:
            lnode = leftroot
            rnode = child
            rootsep = vertices[leftroot].offset + minsep
            loffset = vertices[leftroot].offset
            roffset = loffset + minsep
            vertices[node].right_contour = child
            vertices[node].offset_to_right_contour = rootsep
            while lnode >= 0 and rnode >= 0:
                if vertices[lnode].right_contour >= 0:
                    loffset += vertices[lnode].offset_to_right_contour
                    lnode = vertices[lnode].right_contour
                else:
                    if vertices[rnode].left_contour >= 0:
                        auxnode = vertices[node].left_extreme
                        newoffset = (
                            vertices[node].offset_to_right_extreme
                            - vertices[node].offset_to_left_extreme
                            + minsep
                            + vertices[rnode].offset_to_left_contour
                        )
                        vertices[auxnode].left_contour = vertices[rnode].left_contour
                        vertices[auxnode].right_contour = vertices[rnode].left_contour
                        vertices[auxnode].offset_to_left_contour = newoffset
                        vertices[auxnode].offset_to_right_contour = newoffset
                        vertices[node].left_extreme = vertices[child].left_extreme
                        vertices[node].right_extreme = vertices[child].right_extreme
                        vertices[node].offset_to_left_extreme = (
                            vertices[child].offset_to_left_extreme + rootsep
                        )
                        vertices[node].offset_to_right_extreme = (
                            vertices[child].offset_to_right_extreme + rootsep
                        )
                    else:
                        vertices[node].right_extreme = vertices[child].right_extreme
                        vertices[node].offset_to_right_extreme = (
                            vertices[child].offset_to_right_extreme + rootsep
                        )
                    lnode = -1

                if vertices[rnode].left_contour >= 0:
                    roffset += vertices[rnode].offset_to_left_contour
                    rnode = vertices[rnode].left_contour
                else:
                    if lnode >= 0:
                        auxnode = vertices[child].right_extreme
                        newoffset = loffset - rootsep - vertices[child].offset_to_right_extreme
                        vertices[auxnode].left_contour = lnode
                        vertices[auxnode].right_contour = lnode
                        vertices[auxnode].offset_to_left_contour = newoffset
                        vertices[auxnode].offset_to_right_contour = newoffset
                    rnode = -1

                if lnode >= 0 and rnode >= 0 and roffset - loffset < minsep:
                    rootsep += minsep - roffset + loffset
                    roffset = loffset + minsep
                    vertices[node].offset_to_right_contour = rootsep

            vertices[child].offset = rootsep
            vertices[node].offset_to_right_contour = rootsep
            avg = (avg * sibling_index) / (sibling_index + 1) + rootsep / (sibling_index + 1)
            leftroot = child
        else:
            leftroot = child
            vertices[node].left_contour = child
            vertices[node].right_contour = child
            vertices[node].offset_to_left_contour = 0.0
            vertices[node].offset_to_right_contour = 0.0
            vertices[node].left_extreme = vertices[child].left_extreme
            vertices[node].right_extreme = vertices[child].right_extreme
            vertices[node].offset_to_left_extreme = vertices[child].offset_to_left_extreme
            vertices[node].offset_to_right_extreme = vertices[child].offset_to_right_extreme
            avg = vertices[child].offset
        sibling_index += 1

    vertices[node].offset_to_left_contour -= avg
    vertices[node].offset_to_right_contour -= avg
    vertices[node].offset_to_left_extreme -= avg
    vertices[node].offset_to_right_extreme -= avg
    for child in range(len(vertices)):
        if child != node and vertices[child].parent == node:
            vertices[child].offset -= avg


def _calc_coords(
    vertices: Sequence[_RtVertex],
    positions: list[list[float]],
    node: int,
    xpos: float,
) -> None:
    """Propagate offsets into absolute x coordinates.

    Parameters
    ----------
    vertices : sequence of _RtVertex
        Final RT vertex state.
    positions : list[list[float]]
        Mutable coordinate matrix.
    node : int
        Current subtree root.
    xpos : float
        Absolute x coordinate.

    Returns
    -------
    None
        Coordinates are written to ``positions`` in place.
    """
    positions[node][0] = xpos
    for child in range(len(vertices)):
        if child != node and vertices[child].parent == node:
            _calc_coords(
                vertices=vertices,
                positions=positions,
                node=child,
                xpos=xpos + vertices[child].offset,
            )


def _layout_units(
    edges: Sequence[tuple[int, int]],
    num_nodes: int,
    traversal_mode: str,
    root: int,
) -> list[list[float]]:
    """Lay out an igraph-prepared tree in unscaled units.

    Parameters
    ----------
    edges : sequence of tuple[int, int]
        Extended directed edge pairs.
    num_nodes : int
        Number of extended graph vertices.
    traversal_mode : str
        Traversal mode.
    root : int
        Real root id.

    Returns
    -------
    list[list[float]]
        Coordinates in igraph layout units.
    """
    graph = _adjacency(edges=edges, num_nodes=num_nodes, traversal_mode=traversal_mode)
    vertices = [_RtVertex(left_extreme=node, right_extreme=node) for node in range(num_nodes)]
    positions = [[0.0, 0.0] for _ in range(num_nodes)]
    vertices[root].parent = root
    vertices[root].level = 0

    queue: deque[tuple[int, int]] = deque([(root, 0)])
    while queue:
        node, distance = queue.popleft()
        for neighbor in graph[node]:
            if vertices[neighbor].parent >= 0:
                continue
            positions[neighbor][1] = float(distance + 1)
            vertices[neighbor].parent = node
            vertices[neighbor].level = distance + 1
            queue.append((neighbor, distance + 1))

    _postorder(vertices=vertices, node=root)
    _calc_coords(vertices=vertices, positions=positions, node=root, xpos=vertices[root].offset)
    return positions


def layout_igraph_reingold_tilford(
    edge_index: torch.Tensor,
    num_nodes: int,
    traversal_mode: str = "out",
    roots: Optional[Sequence[int]] = None,
    rootlevel: Optional[Sequence[int]] = None,
    horizontal: bool = False,
    center_output: Optional[bool] = None,
    output_scale: Optional[float] = None,
) -> torch.Tensor:
    """Compute igraph-compatible Reingold-Tilford coordinates.

    Parameters
    ----------
    edge_index : torch.Tensor
        Graph connectivity tensor with shape ``[2, E]``.
    num_nodes : int
        Number of graph vertices.
    traversal_mode : str, default="out"
        Traversal mode: ``"out"``, ``"in"``, or ``"all"``.
    roots : sequence of int | None, default=None
        Optional explicit root vertices.
    rootlevel : sequence of int | None, default=None
        Optional root levels for explicit multi-root layouts.
    horizontal : bool, default=False
        Whether to swap output axes after layout.
    center_output : bool | None, default=None
        Optional mean-centering override.
    output_scale : float | None, default=None
        Uniform output scale. ``None`` uses the igraph adapter scale ``50.0``.

    Returns
    -------
    torch.Tensor
        Scaled coordinates with shape ``[N, 2]``.

    Raises
    ------
    ValueError
        If graph dimensions, roots, or traversal mode are invalid.
    """
    if num_nodes < 0:
        raise ValueError("num_nodes must be non-negative.")
    if traversal_mode not in {"out", "in", "all"}:
        raise ValueError(f"Unsupported Reingold-Tilford traversal_mode: {traversal_mode!r}")
    if num_nodes == 0:
        return torch.empty((0, 2), dtype=torch.float32)

    edges = _validated_edge_list(edge_index=edge_index, num_nodes=num_nodes)
    extended_edges, extended_count, real_root = _prepare_graph(
        edges=edges,
        num_nodes=num_nodes,
        traversal_mode=traversal_mode,
        roots=roots,
        rootlevel=rootlevel,
    )
    sys.setrecursionlimit(max(sys.getrecursionlimit(), extended_count * 2 + 100))
    layout = _layout_units(
        edges=extended_edges,
        num_nodes=extended_count,
        traversal_mode=traversal_mode,
        root=real_root,
    )

    scale = 50.0 if output_scale is None else float(output_scale)
    positions = torch.zeros((num_nodes, 2), dtype=torch.float32)
    for node in range(num_nodes):
        positions[node, 0] = float(layout[node][0]) * scale
        positions[node, 1] = float(layout[node][1]) * scale
    if center_output:
        positions -= positions.mean(dim=0, keepdim=True)
    if horizontal:
        positions = positions[:, [1, 0]]
    return positions


__all__ = ["layout_igraph_reingold_tilford"]
