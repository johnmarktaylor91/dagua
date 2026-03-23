"""Lightweight graph view objects for user-facing navigation.

The graph keeps its layout-oriented storage in tensors and parallel lists.
These views add object-style navigation and concise reprs without copying data.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional, Tuple

if TYPE_CHECKING:
    from dagua.graph import DaguaGraph
    from dagua.render.crossings import EdgeCrossing
    from dagua.styles import ClusterStyle, EdgeStyle, NodeStyle


class NodeView:
    """Lightweight view of a single node in a :class:`~dagua.graph.DaguaGraph`."""

    __slots__ = ("_graph", "_index")

    def __init__(self, graph: DaguaGraph, index: int) -> None:
        """Store the owning graph and node index.

        Parameters
        ----------
        graph : DaguaGraph
            Graph owning the referenced node.
        index : int
            Integer node index within ``graph``.

        Returns
        -------
        None
        """
        self._graph = graph
        self._index = index

    @property
    def index(self) -> int:
        """Return the integer node index.

        Returns
        -------
        int
            Zero-based node index in the owning graph.
        """
        return self._index

    @property
    def id(self) -> Any:
        """Return the original user-provided node identifier.

        Returns
        -------
        Any
            Node identifier supplied by the user, or the integer index when no
            reverse mapping is available.
        """
        if self._index < len(self._graph._index_to_id):
            return self._graph._index_to_id[self._index]
        return self._index

    @property
    def label(self) -> str:
        """Return the user-facing node label.

        Returns
        -------
        str
            Node label stored on the graph.
        """
        return self._graph.node_labels[self._index]

    @property
    def type(self) -> str:
        """Return the node type used by the style cascade.

        Returns
        -------
        str
            Node type string.
        """
        return self._graph.node_types[self._index]

    @property
    def style(self) -> NodeStyle:
        """Return the effective node style after cascade resolution.

        Returns
        -------
        NodeStyle
            Effective style for this node.
        """
        return self._graph.get_style_for_node(self._index)

    @property
    def style_override(self) -> Optional[NodeStyle]:
        """Return the per-node style override, if any.

        Returns
        -------
        Optional[NodeStyle]
            Explicit node style override or ``None`` when the node uses only
            inherited defaults.
        """
        if self._index < len(self._graph.node_styles):
            return self._graph.node_styles[self._index]
        return None

    @property
    def size(self) -> Optional[Tuple[float, float]]:
        """Return the cached node size in data coordinates.

        Returns
        -------
        Optional[Tuple[float, float]]
            ``(width, height)`` tuple when node sizes have been computed,
            otherwise ``None``.
        """
        if self._graph.node_sizes is not None and self._index < self._graph.node_sizes.shape[0]:
            size = self._graph.node_sizes[self._index]
            return (float(size[0].item()), float(size[1].item()))
        return None

    @property
    def position(self) -> Optional[Tuple[float, float]]:
        """Return the cached layout position.

        Returns
        -------
        Optional[Tuple[float, float]]
            ``(x, y)`` tuple when the graph has cached positions, otherwise
            ``None``.
        """
        positions = self._graph.last_positions
        if positions is not None and self._index < positions.shape[0]:
            return (
                float(positions[self._index, 0].item()),
                float(positions[self._index, 1].item()),
            )
        return None

    @property
    def degree(self) -> int:
        """Return the total incident edge count.

        Returns
        -------
        int
            Number of incoming or outgoing edges touching this node.
        """
        edge_index = self._graph.edge_index
        if edge_index.numel() == 0:
            return 0
        return int(((edge_index[0] == self._index) | (edge_index[1] == self._index)).sum().item())

    @property
    def in_degree(self) -> int:
        """Return the number of incoming edges.

        Returns
        -------
        int
            Number of edges whose target is this node.
        """
        edge_index = self._graph.edge_index
        if edge_index.numel() == 0:
            return 0
        return int((edge_index[1] == self._index).sum().item())

    @property
    def out_degree(self) -> int:
        """Return the number of outgoing edges.

        Returns
        -------
        int
            Number of edges whose source is this node.
        """
        edge_index = self._graph.edge_index
        if edge_index.numel() == 0:
            return 0
        return int((edge_index[0] == self._index).sum().item())

    @property
    def edges(self) -> list[EdgeView]:
        """Return all incoming and outgoing edges touching this node.

        Returns
        -------
        list[EdgeView]
            Connected edges in graph edge-order.
        """
        edge_index = self._graph.edge_index
        if edge_index.numel() == 0:
            return []
        mask = (edge_index[0] == self._index) | (edge_index[1] == self._index)
        indices = mask.nonzero(as_tuple=True)[0].tolist()
        return [EdgeView(self._graph, edge_idx) for edge_idx in indices]

    @property
    def outgoing_edges(self) -> list[EdgeView]:
        """Return the outgoing edges for this node.

        Returns
        -------
        list[EdgeView]
            Outgoing edges in graph edge-order.
        """
        edge_index = self._graph.edge_index
        if edge_index.numel() == 0:
            return []
        indices = (edge_index[0] == self._index).nonzero(as_tuple=True)[0].tolist()
        return [EdgeView(self._graph, edge_idx) for edge_idx in indices]

    @property
    def incoming_edges(self) -> list[EdgeView]:
        """Return the incoming edges for this node.

        Returns
        -------
        list[EdgeView]
            Incoming edges in graph edge-order.
        """
        edge_index = self._graph.edge_index
        if edge_index.numel() == 0:
            return []
        indices = (edge_index[1] == self._index).nonzero(as_tuple=True)[0].tolist()
        return [EdgeView(self._graph, edge_idx) for edge_idx in indices]

    @property
    def neighbors(self) -> list[NodeView]:
        """Return directly connected neighboring nodes.

        Returns
        -------
        list[NodeView]
            Neighboring nodes with direction ignored and duplicates removed.
        """
        edge_index = self._graph.edge_index
        if edge_index.numel() == 0:
            return []
        out_targets = edge_index[1, edge_index[0] == self._index].tolist()
        in_sources = edge_index[0, edge_index[1] == self._index].tolist()
        unique = sorted(set(out_targets + in_sources) - {self._index})
        return [NodeView(self._graph, node_idx) for node_idx in unique]

    @property
    def successors(self) -> list[NodeView]:
        """Return nodes reachable by outgoing edges from this node.

        Returns
        -------
        list[NodeView]
            Unique successor nodes in ascending index order.
        """
        edge_index = self._graph.edge_index
        if edge_index.numel() == 0:
            return []
        targets = edge_index[1, edge_index[0] == self._index].tolist()
        unique = sorted(set(targets))
        return [NodeView(self._graph, node_idx) for node_idx in unique]

    @property
    def predecessors(self) -> list[NodeView]:
        """Return nodes with edges pointing to this node.

        Returns
        -------
        list[NodeView]
            Unique predecessor nodes in ascending index order.
        """
        edge_index = self._graph.edge_index
        if edge_index.numel() == 0:
            return []
        sources = edge_index[0, edge_index[1] == self._index].tolist()
        unique = sorted(set(sources))
        return [NodeView(self._graph, node_idx) for node_idx in unique]

    @property
    def clusters(self) -> list[ClusterView]:
        """Return clusters containing this node.

        Returns
        -------
        list[ClusterView]
            Cluster views in graph cluster-order.
        """
        result: list[ClusterView] = []
        for name, members in self._graph.clusters.items():
            if isinstance(members, list) and self._index in members:
                result.append(ClusterView(self._graph, name))
        return result

    def __repr__(self) -> str:
        """Return a concise node summary for interactive use."""
        parts = [f"Node({self._index}", f"label={self.label!r}"]
        if self.type != "default":
            parts.append(f"type={self.type!r}")
        degree = self.degree
        if degree > 0:
            parts.append(f"degree={degree}")
        position = self.position
        if position is not None:
            parts.append(f"pos=({position[0]:.1f}, {position[1]:.1f})")
        clusters = self.clusters
        if clusters:
            parts.append(f"clusters=[{', '.join(cluster.name for cluster in clusters)}]")
        return ", ".join(parts) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether two views reference the same graph node."""
        if not isinstance(other, NodeView):
            return False
        return self._graph is other._graph and self._index == other._index

    def __hash__(self) -> int:
        """Return a stable hash based on graph identity and node index."""
        return hash((id(self._graph), self._index))


class EdgeView:
    """Lightweight view of a single edge in a :class:`~dagua.graph.DaguaGraph`."""

    __slots__ = ("_graph", "_index")

    def __init__(self, graph: DaguaGraph, index: int) -> None:
        """Store the owning graph and edge index.

        Parameters
        ----------
        graph : DaguaGraph
            Graph owning the referenced edge.
        index : int
            Integer edge index within ``graph``.

        Returns
        -------
        None
        """
        self._graph = graph
        self._index = index

    @property
    def index(self) -> int:
        """Return the integer edge index.

        Returns
        -------
        int
            Zero-based edge index in the owning graph.
        """
        return self._index

    @property
    def source(self) -> NodeView:
        """Return the source node view.

        Returns
        -------
        NodeView
            Source endpoint of this edge.
        """
        source_idx = int(self._graph.edge_index[0, self._index].item())
        return NodeView(self._graph, source_idx)

    @property
    def target(self) -> NodeView:
        """Return the target node view.

        Returns
        -------
        NodeView
            Target endpoint of this edge.
        """
        target_idx = int(self._graph.edge_index[1, self._index].item())
        return NodeView(self._graph, target_idx)

    @property
    def label(self) -> Optional[str]:
        """Return the optional edge label.

        Returns
        -------
        Optional[str]
            Edge label or ``None`` when unset.
        """
        if self._index < len(self._graph.edge_labels):
            return self._graph.edge_labels[self._index]
        return None

    @property
    def type(self) -> str:
        """Return the edge type used by the style cascade.

        Returns
        -------
        str
            Edge type string.
        """
        if self._index < len(self._graph.edge_types):
            return self._graph.edge_types[self._index]
        return "normal"

    @property
    def weight(self) -> Optional[float]:
        """Return the optional edge weight.

        Returns
        -------
        Optional[float]
            Edge weight as a Python float, or ``None`` when weights are absent.
        """
        if self._graph.edge_weights is not None and self._index < self._graph.edge_weights.shape[0]:
            return float(self._graph.edge_weights[self._index].item())
        return None

    @property
    def style(self) -> EdgeStyle:
        """Return the effective edge style after cascade resolution.

        Returns
        -------
        EdgeStyle
            Effective style for this edge.
        """
        return self._graph.get_style_for_edge(self._index)

    @property
    def style_override(self) -> Optional[EdgeStyle]:
        """Return the per-edge style override, if any.

        Returns
        -------
        Optional[EdgeStyle]
            Explicit edge style override or ``None`` when the edge uses only
            inherited defaults.
        """
        if self._index < len(self._graph.edge_styles):
            return self._graph.edge_styles[self._index]
        return None

    @property
    def is_back_edge(self) -> bool:
        """Return whether the edge is marked as a back edge.

        Returns
        -------
        bool
            ``True`` when cycle handling marked this edge as a back edge.
        """
        mask = self._graph.back_edge_mask
        if mask is not None and self._index < mask.shape[0]:
            return bool(mask[self._index].item())
        return False

    @property
    def crossings(self) -> list["EdgeCrossing"]:
        """Return cached crossings touching this edge.

        Returns
        -------
        list[EdgeCrossing]
            Crossing records involving this edge. Returns an empty list when
            the renderer has not computed crossings for the current graph state.
        """
        cached = self._graph._cached_crossings
        if cached is None:
            return []
        return [
            crossing
            for crossing in cached
            if crossing.edge_a == self._index or crossing.edge_b == self._index
        ]

    def __repr__(self) -> str:
        """Return a concise edge summary for interactive use."""
        parts = [f"Edge({self.source.label!r} -> {self.target.label!r}"]
        label = self.label
        if label:
            parts.append(f"label={label!r}")
        weight = self.weight
        if weight is not None:
            parts.append(f"weight={weight:.2g}")
        if self.is_back_edge:
            parts.append("back=True")
        return ", ".join(parts) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether two views reference the same graph edge."""
        if not isinstance(other, EdgeView):
            return False
        return self._graph is other._graph and self._index == other._index

    def __hash__(self) -> int:
        """Return a stable hash based on graph identity and edge index."""
        return hash((id(self._graph), "edge", self._index))


class ClusterView:
    """Lightweight view of a cluster in a :class:`~dagua.graph.DaguaGraph`."""

    __slots__ = ("_graph", "_name")

    def __init__(self, graph: DaguaGraph, name: str) -> None:
        """Store the owning graph and cluster name.

        Parameters
        ----------
        graph : DaguaGraph
            Graph owning the referenced cluster.
        name : str
            Cluster name within ``graph``.

        Returns
        -------
        None
        """
        self._graph = graph
        self._name = name

    @property
    def name(self) -> str:
        """Return the cluster name.

        Returns
        -------
        str
            Internal cluster name.
        """
        return self._name

    @property
    def label(self) -> str:
        """Return the display label for this cluster.

        Returns
        -------
        str
            Explicit cluster label or the cluster name when unset.
        """
        return self._graph.cluster_labels.get(self._name, self._name)

    @property
    def members(self) -> list[NodeView]:
        """Return the member node views.

        Returns
        -------
        list[NodeView]
            Nodes listed for this cluster.
        """
        indices = self._graph.clusters.get(self._name, [])
        return [NodeView(self._graph, node_idx) for node_idx in indices]

    @property
    def member_count(self) -> int:
        """Return the member node count.

        Returns
        -------
        int
            Number of nodes listed for this cluster.
        """
        return len(self._graph.clusters.get(self._name, []))

    @property
    def children(self) -> list[ClusterView]:
        """Return child clusters in the hierarchy.

        Returns
        -------
        list[ClusterView]
            Direct child clusters.
        """
        child_names = self._graph.cluster_children(self._name)
        return [ClusterView(self._graph, child_name) for child_name in child_names]

    @property
    def parent(self) -> Optional[ClusterView]:
        """Return the parent cluster, if any.

        Returns
        -------
        Optional[ClusterView]
            Parent cluster view or ``None`` for root clusters.
        """
        parent_name = self._graph.cluster_parents.get(self._name)
        if parent_name is None:
            return None
        return ClusterView(self._graph, parent_name)

    @property
    def depth(self) -> int:
        """Return the cluster hierarchy depth.

        Returns
        -------
        int
            Zero-based cluster depth.
        """
        return self._graph.cluster_depth(self._name)

    @property
    def style(self) -> ClusterStyle:
        """Return the effective cluster style.

        Returns
        -------
        ClusterStyle
            Effective style for this cluster.
        """
        return self._graph.get_style_for_cluster(self._name)

    @property
    def style_override(self) -> Optional[ClusterStyle]:
        """Return the per-cluster style override, if any.

        Returns
        -------
        Optional[ClusterStyle]
            Explicit cluster style override or ``None`` when the cluster uses
            inherited theme defaults only.
        """
        return self._graph.cluster_styles.get(self._name)

    def __repr__(self) -> str:
        """Return a concise cluster summary for interactive use."""
        parts = [f"Cluster({self._name!r}", f"members={self.member_count}"]
        children = self.children
        if children:
            parts.append(f"children=[{', '.join(child.name for child in children)}]")
        parent = self.parent
        if parent is not None:
            parts.append(f"parent={parent.name!r}")
        depth = self.depth
        if depth > 0:
            parts.append(f"depth={depth}")
        return ", ".join(parts) + ")"

    def __eq__(self, other: object) -> bool:
        """Return whether two views reference the same graph cluster."""
        if not isinstance(other, ClusterView):
            return False
        return self._graph is other._graph and self._name == other._name

    def __hash__(self) -> int:
        """Return a stable hash based on graph identity and cluster name."""
        return hash((id(self._graph), "cluster", self._name))
