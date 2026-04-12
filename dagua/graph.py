"""DaguaGraph — the central data structure for graph layout and rendering."""

from __future__ import annotations

import copy as _copy
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any, Dict, Iterator, List, Optional, Tuple, Union

import torch

from dagua.styles import (
    GRAPHVIZ_THEME,
    ClusterStyle,
    EdgeStyle,
    GraphStyle,
    NodeStyle,
    Theme,
    resolve_cluster_style,
    resolve_edge_style,
    resolve_node_style,
)
from dagua.utils import compute_node_size, fit_font_size_to_node

if TYPE_CHECKING:
    from dagua.render.crossings import EdgeCrossing
    from dagua.views import ClusterView, EdgeView, NodeView

_DTYPE_NAME_TO_TORCH = {
    "int32": torch.int32,
    "int64": torch.int64,
    "float16": torch.float16,
    "float32": torch.float32,
    "float64": torch.float64,
}

_CLUSTER_PREFIX = "cluster_"


def _updated_style_instance(
    current_style: Optional[Union[NodeStyle, EdgeStyle, ClusterStyle]],
    style_type: type[Union[NodeStyle, EdgeStyle, ClusterStyle]],
    updates: Dict[str, Any],
) -> Union[NodeStyle, EdgeStyle, ClusterStyle]:
    """Return a style instance with graph-level updates applied.

    Parameters
    ----------
    current_style : NodeStyle | EdgeStyle | ClusterStyle | None
        Existing graph-level default style, if any.
    style_type : type[NodeStyle] | type[EdgeStyle] | type[ClusterStyle]
        Dataclass type to instantiate when no style exists yet.
    updates : dict[str, Any]
        Field names and values to assign to the style object.

    Returns
    -------
    NodeStyle | EdgeStyle | ClusterStyle
        Updated style object. Existing instances are deep-copied so callers do
        not mutate shared objects by reference.
    """
    style = _copy.deepcopy(current_style) if current_style is not None else style_type()
    for field_name, value in updates.items():
        setattr(style, field_name, value)
    return style


@dataclass
class DaguaGraph:
    """A directed graph ready for layout and rendering.

    Core representation uses PyG-convention edge_index [2, E] LongTensor.
    Node IDs are mapped to integer indices internally.

    Edges are accumulated in a Python list during construction and lazily
    converted to a tensor when ``edge_index`` is first accessed, avoiding
    O(E²) ``torch.cat`` calls during incremental ``add_edge()`` usage.
    """

    # Core topology
    num_nodes: int = 0

    # Node properties
    node_labels: List[str] = field(default_factory=list)
    node_sizes: Optional[torch.Tensor] = None  # [N, 2] width, height
    node_font_sizes: Optional[torch.Tensor] = None  # [N] effective font size per node
    node_types: List[str] = field(default_factory=list)
    node_styles: List[Optional[NodeStyle]] = field(default_factory=list)

    # Edge properties
    edge_labels: List[Optional[str]] = field(default_factory=list)
    edge_types: List[str] = field(default_factory=list)
    edge_styles: List[Optional[EdgeStyle]] = field(default_factory=list)
    edge_weights: Optional[torch.Tensor] = None  # [E] per-edge weight, float32

    # Cluster hierarchy: name -> list of node indices
    clusters: Dict[str, Any] = field(default_factory=dict)
    cluster_styles: Dict[str, ClusterStyle] = field(default_factory=dict)
    cluster_labels: Dict[str, str] = field(default_factory=dict)
    cluster_parents: Dict[str, Optional[str]] = field(default_factory=dict)

    # Layout direction
    direction: str = "TB"  # TB, BT, LR, RL

    # Graph-level style defaults (cascade level 4)
    default_node_style: Optional[NodeStyle] = None
    default_edge_style: Optional[EdgeStyle] = None
    default_cluster_style: Optional[ClusterStyle] = None

    # Flex layout constraints
    flex: Optional[Any] = None  # LayoutFlex — typed as Any to avoid circular import

    # Storage precision controls
    index_dtype: Optional[torch.dtype] = None
    size_dtype: Optional[torch.dtype] = None

    # ID mapping
    _id_to_index: Dict[Any, int] = field(default_factory=dict, repr=False)
    _index_to_id: List[Any] = field(default_factory=list, repr=False)
    _theme: Any = field(
        default_factory=lambda: _copy.deepcopy(GRAPHVIZ_THEME), repr=False
    )  # Theme or Dict[str, NodeStyle]

    # Internal edge storage — not part of the public API
    _pending_edges: List[Tuple[int, int]] = field(default_factory=list, repr=False)
    _pending_edge_weights: List[Optional[float]] = field(default_factory=list, repr=False)
    _edge_index_tensor: Optional[torch.Tensor] = field(default=None, repr=False)

    # Cycle support — populated lazily by has_cycles / back_edge_mask
    _back_edge_mask: Optional[torch.Tensor] = field(default=None, repr=False)
    _original_edge_index: Optional[torch.Tensor] = field(default=None, repr=False)

    # Lifecycle / cached layout state
    revision: int = 0
    _node_sizes_revision: int = field(default=-1, repr=False)
    _layout_positions: Optional[torch.Tensor] = field(default=None, repr=False)
    _layout_revision: int = field(default=-1, repr=False)
    _layout_curves: Optional[Any] = field(default=None, repr=False)
    _routing_revision: int = field(default=-1, repr=False)
    _layout_label_positions: Optional[List[Optional[Tuple[float, float]]]] = field(
        default=None, repr=False
    )
    _label_revision: int = field(default=-1, repr=False)
    _cached_crossings: Optional[List["EdgeCrossing"]] = field(default=None, repr=False)

    @property
    def edge_index(self) -> torch.Tensor:
        """Return the [2, E] edge_index tensor, finalizing pending edges first."""
        self._finalize_edges()
        return self._edge_index_tensor  # type: ignore[return-value]

    @edge_index.setter
    def edge_index(self, value: torch.Tensor) -> None:
        """Set the edge_index tensor directly.

        Parameters
        ----------
        value : torch.Tensor
            Edge tensor with shape ``[2, E]``. Any pending edge additions and
            pending edge weights are discarded before storing the tensor.

        Returns
        -------
        None
        """
        self._pending_edges.clear()
        self._pending_edge_weights.clear()
        value = value.to(dtype=self.index_dtype)
        self._validate_index_range(value)
        self._edge_index_tensor = value
        self._back_edge_mask = None  # invalidate cycle cache
        if self.edge_weights is not None and self.edge_weights.shape[0] != value.shape[1]:
            self.edge_weights = None
        self.invalidate_layout()
        self._touch()

    def __post_init__(self) -> None:
        from dagua.defaults import get_default_index_dtype, get_default_size_dtype

        # Auto-convert bare Dict[str, NodeStyle] to Theme for backwards compat
        if isinstance(self._theme, dict):
            self._theme = Theme(node_styles=dict(self._theme))
        self.index_dtype = self._normalize_index_dtype(
            self.index_dtype if self.index_dtype is not None else get_default_index_dtype()
        )
        self.size_dtype = self._normalize_size_dtype(
            self.size_dtype if self.size_dtype is not None else get_default_size_dtype()
        )
        # Ensure the tensor is initialized when no edges are provided
        if self._edge_index_tensor is None and not self._pending_edges:
            self._edge_index_tensor = torch.zeros(2, 0, dtype=self.index_dtype)
        elif self._edge_index_tensor is not None:
            self._edge_index_tensor = self._edge_index_tensor.to(dtype=self.index_dtype)

    def __repr__(self) -> str:
        """Return a compact human-readable summary of the graph state."""
        edge_count = self.num_edges
        parts = [f"DaguaGraph({self.num_nodes} nodes, {edge_count} edges"]
        if self.clusters:
            cluster_word = "cluster" if len(self.clusters) == 1 else "clusters"
            parts.append(f"{len(self.clusters)} {cluster_word}")
        parts.append(f"direction={self.direction!r}")
        status = self.layout_status
        if status != "missing":
            parts.append(f"layout={status}")
        if self.edge_weights is not None:
            parts.append("weighted=True")
        if self.is_cyclic:
            parts.append("cyclic=True")
        return ", ".join(parts) + ")"

    def configure(self, **kwargs: Any) -> None:
        """Set graph-scoped style defaults from flat keyword arguments.

        Parameters
        ----------
        **kwargs : Any
            Style overrides routed by field name.

            Unprefixed node-style fields such as ``font_size`` or
            ``overflow_policy`` update ``default_node_style``.
            Prefixed edge-style fields such as ``edge_width`` update
            ``default_edge_style``. Unprefixed edge-only field names such as
            ``color`` and ``routing`` are also accepted.
            Cluster-style fields must use the ``cluster_`` prefix, for example
            ``cluster_padding``.

        Returns
        -------
        None

        Raises
        ------
        TypeError
            If any keyword cannot be routed to a supported graph-level style
            default.
        """
        from dagua.defaults import (
            _EDGE_PREFIX,
            _EDGE_STYLE_FIELDS,
            _NODE_STYLE_FIELDS,
            _did_you_mean,
        )

        node_updates: Dict[str, Any] = {}
        edge_updates: Dict[str, Any] = {}
        cluster_updates: Dict[str, Any] = {}
        cluster_fields = {
            field_def.name for field_def in ClusterStyle.__dataclass_fields__.values()
        }

        for key, value in kwargs.items():
            if key.startswith(_CLUSTER_PREFIX):
                cluster_field = key[len(_CLUSTER_PREFIX) :]
                if cluster_field in cluster_fields:
                    cluster_updates[cluster_field] = value
                    continue
            elif key in _NODE_STYLE_FIELDS:
                node_updates[key] = value
                continue
            elif key.startswith(_EDGE_PREFIX) and key[len(_EDGE_PREFIX) :] in _EDGE_STYLE_FIELDS:
                edge_updates[key[len(_EDGE_PREFIX) :]] = value
                continue
            elif key in _EDGE_STYLE_FIELDS and key not in _NODE_STYLE_FIELDS:
                edge_updates[key] = value
                continue

            hint = _did_you_mean(key)
            raise TypeError(f"Unknown configure() option: {key!r}.{hint}")

        if node_updates:
            self.default_node_style = _updated_style_instance(
                self.default_node_style,
                NodeStyle,
                node_updates,
            )
        if edge_updates:
            self.default_edge_style = _updated_style_instance(
                self.default_edge_style,
                EdgeStyle,
                edge_updates,
            )
        if cluster_updates:
            self.default_cluster_style = _updated_style_instance(
                self.default_cluster_style,
                ClusterStyle,
                cluster_updates,
            )

        if node_updates or edge_updates or cluster_updates:
            self.invalidate_layout()
            self._touch()

    @staticmethod
    def _normalize_index_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
        normalized: Optional[torch.dtype]
        if isinstance(dtype, str):
            normalized = _DTYPE_NAME_TO_TORCH.get(dtype)
        else:
            normalized = dtype
        if normalized not in (torch.int32, torch.int64):
            raise TypeError("index_dtype must be torch.int32 or torch.int64")
        return normalized

    @staticmethod
    def _normalize_size_dtype(dtype: Union[torch.dtype, str]) -> torch.dtype:
        normalized: Optional[torch.dtype]
        if isinstance(dtype, str):
            normalized = _DTYPE_NAME_TO_TORCH.get(dtype)
        else:
            normalized = dtype
        if normalized not in (torch.float16, torch.float32, torch.float64):
            raise TypeError("size_dtype must be torch.float16, torch.float32, or torch.float64")
        return normalized

    def _validate_index_range(self, tensor: torch.Tensor) -> None:
        if tensor.numel() == 0 or self.index_dtype != torch.int32:
            return
        min_idx = int(tensor.min().item())
        max_idx = int(tensor.max().item())
        if min_idx < 0 or max_idx > torch.iinfo(torch.int32).max:
            raise ValueError("edge indices do not fit in int32 storage")

    def _finalize_edges(self) -> None:
        """Flush pending edges and weights into tensor storage.

        Returns
        -------
        None
        """
        if not self._pending_edges:
            return

        new_edges = torch.tensor(self._pending_edges, dtype=self.index_dtype).t()  # [2, K]
        self._validate_index_range(new_edges)

        if self._edge_index_tensor is None or self._edge_index_tensor.numel() == 0:
            self._edge_index_tensor = new_edges
        else:
            self._edge_index_tensor = torch.cat([self._edge_index_tensor, new_edges], dim=1)

        has_any_weight = any(weight is not None for weight in self._pending_edge_weights)
        if has_any_weight or self.edge_weights is not None:
            new_weights = torch.tensor(
                [weight if weight is not None else 1.0 for weight in self._pending_edge_weights],
                dtype=torch.float32,
            )
            if self.edge_weights is not None:
                self.edge_weights = torch.cat([self.edge_weights, new_weights])
            else:
                existing_count = self._edge_index_tensor.shape[1] - len(self._pending_edges)
                if existing_count > 0:
                    backfill = torch.ones(existing_count, dtype=torch.float32)
                    self.edge_weights = torch.cat([backfill, new_weights])
                else:
                    self.edge_weights = new_weights

        self._pending_edges.clear()
        self._pending_edge_weights.clear()
        self._back_edge_mask = None  # invalidate cycle cache

    def _touch(self) -> None:
        """Bump graph revision after a user-visible mutation."""
        self.revision += 1

    @property
    def _edge_count(self) -> int:
        """Return the current edge count without forcing finalization.

        Returns
        -------
        int
            Count of finalized and pending edges.
        """
        return self.num_edges

    @property
    def num_edges(self) -> int:
        """Return the edge count without forcing edge tensor finalization.

        Returns
        -------
        int
            Total number of pending and finalized edges.
        """
        pending = len(self._pending_edges)
        finalized = (
            self._edge_index_tensor.shape[1]
            if self._edge_index_tensor is not None and self._edge_index_tensor.numel() > 0
            else 0
        )
        return pending + finalized

    def invalidate_layout(self) -> None:
        """Drop cached layout-derived artifacts."""
        self._layout_positions = None
        self._layout_revision = -1
        self._layout_curves = None
        self._routing_revision = -1
        self._layout_label_positions = None
        self._label_revision = -1
        self._cached_crossings = None
        self._node_sizes_revision = -1
        self.node_sizes = None
        self.node_font_sizes = None

    def _invalidate_routing(self) -> None:
        """Drop cached post-layout routing artifacts while keeping positions."""
        self._layout_curves = None
        self._routing_revision = -1
        self._layout_label_positions = None
        self._label_revision = -1
        self._cached_crossings = None

    @property
    def has_fresh_layout(self) -> bool:
        """Whether cached positions match the current graph revision."""
        return self._layout_positions is not None and self._layout_revision == self.revision

    @property
    def layout_status(self) -> str:
        """Human-readable layout cache state: missing, fresh, or stale."""
        if self._layout_positions is None:
            return "missing"
        if self._layout_revision == self.revision:
            return "fresh"
        return "stale"

    @property
    def last_positions(self) -> Optional[torch.Tensor]:
        """Most recently cached layout positions, if any."""
        return self._layout_positions

    @property
    def last_curves(self):
        """Most recently cached routed curves, if any."""
        return self._layout_curves

    @property
    def last_label_positions(self) -> Optional[List[Optional[Tuple[float, float]]]]:
        """Most recently cached edge label positions, if any."""
        return self._layout_label_positions

    def cache_layout(self, positions: torch.Tensor) -> torch.Tensor:
        """Cache positions as the current fresh layout."""
        self._layout_positions = positions.detach().cpu().clone()
        self._layout_revision = self.revision
        self._invalidate_routing()
        return positions

    def cache_routing(
        self, curves, label_positions: Optional[List[Optional[Tuple[float, float]]]] = None
    ) -> None:
        """Cache routed edges and optional label positions for the current revision."""
        self._layout_curves = curves
        self._routing_revision = self.revision
        self._layout_label_positions = label_positions
        self._label_revision = self.revision if label_positions is not None else -1

    def node(self, id_or_index: Any) -> "NodeView":
        """Return a node view by user ID or integer index.

        Parameters
        ----------
        id_or_index : Any
            User-provided node identifier or integer node index.

        Returns
        -------
        NodeView
            View object referencing the requested node.

        Raises
        ------
        IndexError
            If an integer index is outside the valid range.
        KeyError
            If a non-index identifier is unknown.
        """
        from dagua.views import NodeView

        if isinstance(id_or_index, int) and id_or_index not in self._id_to_index:
            if id_or_index < 0 or id_or_index >= self.num_nodes:
                raise IndexError(f"Node index {id_or_index} out of range [0, {self.num_nodes})")
            return NodeView(self, id_or_index)
        if id_or_index not in self._id_to_index:
            raise KeyError(f"Unknown node ID: {id_or_index!r}")
        return NodeView(self, self._id_to_index[id_or_index])

    def edge(self, index: int) -> "EdgeView":
        """Return an edge view by integer index.

        Parameters
        ----------
        index : int
            Zero-based edge index.

        Returns
        -------
        EdgeView
            View object referencing the requested edge.

        Raises
        ------
        IndexError
            If ``index`` is outside the valid edge range.
        """
        from dagua.views import EdgeView

        self._finalize_edges()
        edge_count = self._edge_index_tensor.shape[1] if self._edge_index_tensor is not None else 0
        if index < 0 or index >= edge_count:
            raise IndexError(f"Edge index {index} out of range [0, {edge_count})")
        return EdgeView(self, index)

    def cluster(self, name: str) -> "ClusterView":
        """Return a cluster view by name.

        Parameters
        ----------
        name : str
            Cluster name.

        Returns
        -------
        ClusterView
            View object referencing the requested cluster.

        Raises
        ------
        KeyError
            If ``name`` is not a known cluster.
        """
        from dagua.views import ClusterView

        if name not in self.clusters:
            raise KeyError(f"Unknown cluster: {name!r}")
        return ClusterView(self, name)

    @property
    def nodes(self) -> Iterator["NodeView"]:
        """Iterate over all nodes as lightweight views.

        Returns
        -------
        Iterator[NodeView]
            Generator yielding node views in index order.
        """
        from dagua.views import NodeView

        for index in range(self.num_nodes):
            yield NodeView(self, index)

    @property
    def nodes_iter(self) -> Iterator["NodeView"]:
        """Iterate over all nodes via the deprecated alias.

        Returns
        -------
        Iterator[NodeView]
            Generator yielding node views in index order.
        """
        return self.nodes

    @property
    def edges(self) -> Iterator["EdgeView"]:
        """Iterate over all edges as lightweight views.

        Returns
        -------
        Iterator[EdgeView]
            Generator yielding edge views in edge order.
        """
        from dagua.views import EdgeView

        self._finalize_edges()
        for index in range(self.num_edges):
            yield EdgeView(self, index)

    @property
    def edges_view(self) -> Iterator["EdgeView"]:
        """Iterate over all edges via the compatibility alias.

        Returns
        -------
        Iterator[EdgeView]
            Generator yielding edge views in edge order.
        """
        return self.edges

    @property
    def edges_iter(self) -> Iterator["EdgeView"]:
        """Iterate over all edges via the deprecated alias.

        Returns
        -------
        Iterator[EdgeView]
            Generator yielding edge views in edge order.
        """
        return self.edges

    @property
    def clusters_view(self) -> Iterator["ClusterView"]:
        """Iterate over all clusters as lightweight views.

        Returns
        -------
        Iterator[ClusterView]
            Generator yielding cluster views in insertion order.
        """
        from dagua.views import ClusterView

        for name in self.clusters:
            yield ClusterView(self, name)

    @property
    def clusters_iter(self) -> Iterator["ClusterView"]:
        """Iterate over all clusters via the deprecated alias.

        Returns
        -------
        Iterator[ClusterView]
            Generator yielding cluster views in insertion order.
        """
        return self.clusters_view

    def __contains__(self, node_id: Any) -> bool:
        """Return whether a node ID exists in the graph.

        Parameters
        ----------
        node_id : Any
            User-facing node identifier to check.

        Returns
        -------
        bool
            ``True`` when ``node_id`` resolves to a graph node.
        """
        return node_id in self._id_to_index

    def __len__(self) -> int:
        """Return the number of nodes in the graph.

        Returns
        -------
        int
            Current node count.
        """
        return self.num_nodes

    def __getitem__(self, key: Any) -> "NodeView":
        """Return a node view for ``graph[key]`` access.

        Parameters
        ----------
        key : Any
            Node identifier or integer node index.

        Returns
        -------
        NodeView
            Node view referencing the selected node.
        """
        return self.node(key)

    def node_id(self, index: int) -> Any:
        """Return the user-provided ID for a node index.

        Parameters
        ----------
        index : int
            Integer node index.

        Returns
        -------
        Any
            Original user-provided node ID, or ``index`` if no reverse mapping
            exists for that slot.
        """
        if index < len(self._index_to_id):
            return self._index_to_id[index]
        return index

    def edges_for_node(self, node: Any) -> list["EdgeView"]:
        """Return all edges incident to a node.

        Parameters
        ----------
        node : Any
            Node identifier or integer node index.

        Returns
        -------
        list[EdgeView]
            Incoming and outgoing edges touching the node.
        """
        return self.node(node).edges

    def edges_between(self, source: Any, target: Any) -> list["EdgeView"]:
        """Return all directed edges from ``source`` to ``target``.

        Parameters
        ----------
        source : Any
            Source node identifier or integer node index.
        target : Any
            Target node identifier or integer node index.

        Returns
        -------
        list[EdgeView]
            Edge views whose endpoints match the requested direction.
        """
        from dagua.views import EdgeView

        src_idx = self._id_to_index.get(source, source if isinstance(source, int) else None)
        tgt_idx = self._id_to_index.get(target, target if isinstance(target, int) else None)
        if src_idx is None or tgt_idx is None:
            return []
        edge_index = self.edge_index
        mask = (edge_index[0] == src_idx) & (edge_index[1] == tgt_idx)
        indices = mask.nonzero(as_tuple=True)[0].tolist()
        return [EdgeView(self, edge_idx) for edge_idx in indices]

    def clusters_for_node(self, node: Any) -> list["ClusterView"]:
        """Return all clusters containing a node.

        Parameters
        ----------
        node : Any
            Node identifier or integer node index.

        Returns
        -------
        list[ClusterView]
            Clusters containing the requested node.
        """
        return self.node(node).clusters

    @property
    def summary(self) -> str:
        """Return a multi-line human-readable graph summary.

        Returns
        -------
        str
            Summary including topology, style-relevant distributions, clusters,
            and cached layout state.
        """
        from collections import Counter

        lines = [repr(self)]
        type_counts = Counter(self.node_types)
        if len(type_counts) > 1 or (len(type_counts) == 1 and "default" not in type_counts):
            types_str = ", ".join(
                f"{node_type}: {count}" for node_type, count in type_counts.most_common()
            )
            lines.append(f"  Node types: {types_str}")
        edge_count = self.edge_index.shape[1]
        if edge_count > 0 and self.edge_weights is not None:
            weights = self.edge_weights
            assert weights is not None
            lines.append(
                "  Edge weights: "
                f"min={float(weights.min().item()):.2g}, "
                f"max={float(weights.max().item()):.2g}, "
                f"mean={float(weights.mean().item()):.2g}"
            )
        if self.clusters:
            for name in self.clusters:
                cluster_view = self.cluster(name)
                indent = "  " * (cluster_view.depth + 1)
                lines.append(f"{indent}Cluster {name!r}: {cluster_view.member_count} members")
        lines.append(f"  Layout: {self.layout_status}")
        if self.has_cycles:
            back_count = (
                int(self.back_edge_mask.sum().item()) if self.back_edge_mask is not None else 0
            )
            lines.append(f"  Cycles: {back_count} back edges")
        return "\n".join(lines)

    def add_node(
        self,
        node_id: Any,
        label: Optional[str] = None,
        type: str = "default",
        style: Optional[NodeStyle] = None,
    ) -> int:
        """Add a node and return its integer index."""
        if node_id in self._id_to_index:
            return self._id_to_index[node_id]

        idx = self.num_nodes
        self._id_to_index[node_id] = idx
        self._index_to_id.append(node_id)
        self.num_nodes += 1
        self.node_labels.append(label if label is not None else str(node_id))
        self.node_types.append(type)
        self.node_styles.append(style)
        self.invalidate_layout()
        self._touch()
        return idx

    def add_edge(
        self,
        source: Any,
        target: Any,
        label: Optional[str] = None,
        type: str = "normal",
        style: Optional[EdgeStyle] = None,
        weight: Optional[float] = None,
    ) -> None:
        """Add an edge between two nodes.

        Parameters
        ----------
        source : Any
            Source node identifier. The node is created if it does not exist.
        target : Any
            Target node identifier. The node is created if it does not exist.
        label : Optional[str], default=None
            Optional label associated with the edge.
        type : str, default="normal"
            Edge type used by the style cascade.
        style : Optional[EdgeStyle], default=None
            Optional per-edge style override.
        weight : Optional[float], default=None
            Optional edge weight. Missing weights are backfilled to ``1.0`` once
            any edge in the graph carries an explicit weight.

        Returns
        -------
        None
        """
        if source not in self._id_to_index:
            self.add_node(source)
        if target not in self._id_to_index:
            self.add_node(target)

        src_idx = self._id_to_index[source]
        tgt_idx = self._id_to_index[target]

        self._pending_edges.append((src_idx, tgt_idx))
        self._pending_edge_weights.append(weight)

        self.edge_labels.append(label)
        self.edge_types.append(type)
        self.edge_styles.append(style)
        self.invalidate_layout()
        self._touch()

    def add_cluster(
        self,
        name: str,
        members: Union[List[Any], Dict],
        style: Optional[ClusterStyle] = None,
        label: Optional[str] = None,
        parent: Optional[str] = None,
        strict: bool = True,
    ) -> None:
        """Add a cluster. Members can be node IDs/indices or nested dict.

        Args:
            name: Cluster name.
            members: List of node IDs/indices, or dict-of-dicts for nesting.
            style: Optional ClusterStyle override.
            label: Optional display label.
            parent: Optional parent cluster name (for hierarchy).
        """
        if isinstance(members, dict):
            # Dict-of-dicts: keys are child cluster names, values are member lists or nested dicts.
            # Recursively flatten to get leaf indices, and auto-populate cluster_parents.
            from dagua.utils import collect_cluster_leaves

            all_indices = collect_cluster_leaves(members)
            self.clusters[name] = all_indices
            # Auto-create child clusters from dict keys
            for child_name, child_members in members.items():
                if isinstance(child_members, dict):
                    self.add_cluster(child_name, child_members, parent=name)
                else:
                    # child_members is a list of node IDs
                    indices = []
                    for m in child_members:
                        if isinstance(m, int) and m < self.num_nodes:
                            indices.append(m)
                        elif m in self._id_to_index:
                            indices.append(self._id_to_index[m])
                        elif strict:
                            raise KeyError(
                                f"Unknown cluster member {m!r} for cluster {child_name!r}. "
                                "Add the node first or pass strict=False."
                            )
                    self.clusters[child_name] = indices
                    self.cluster_parents[child_name] = name
        else:
            indices = []
            for m in members:
                if isinstance(m, int) and m < self.num_nodes:
                    indices.append(m)
                elif m in self._id_to_index:
                    indices.append(self._id_to_index[m])
                elif strict:
                    raise KeyError(
                        f"Unknown cluster member {m!r} for cluster {name!r}. "
                        "Add the node first or pass strict=False."
                    )
            self.clusters[name] = indices

        if parent is not None:
            # Cycle detection: parent can't be a descendant of name
            cur: Optional[str] = parent
            while cur is not None:
                if cur == name:
                    raise ValueError(
                        f"Cluster cycle detected: '{name}' cannot have "
                        f"'{parent}' as parent (it is a descendant)"
                    )
                cur = self.cluster_parents.get(cur)
            self.cluster_parents[name] = parent

        if style is not None:
            self.cluster_styles[name] = style
        if label is not None:
            self.cluster_labels[name] = label
        self.invalidate_layout()
        self._touch()

    def compute_node_sizes(
        self,
        font_family: str = "",
        font_size: float = 8.5,
    ) -> None:
        """Compute node sizes from labels if not already set.

        Uses per-node style for padding, shape, font_weight, font_style,
        text wrapping/transform/rotation, min_width, min_height,
        overflow_policy, and min_font_size. Populates both node_sizes and
        node_font_sizes tensors.
        """
        if (
            self.node_sizes is not None
            and self.node_sizes.ndim == 2
            and self.node_sizes.shape[0] == self.num_nodes
            and self.node_sizes.shape[1] == 2
            and self._node_sizes_revision != self.revision
            and len(self.node_labels) < self.num_nodes
        ):
            self._node_sizes_revision = self.revision
            return

        if (
            self.node_sizes is not None
            and self.node_sizes.shape[0] == self.num_nodes
            and self._node_sizes_revision == self.revision
        ):
            return

        sizes = []
        font_sizes = []
        for i in range(self.num_nodes):
            label = self.node_labels[i] if i < len(self.node_labels) else ""
            style = self.get_style_for_node(i)
            padding = style.padding
            ff = font_family if style.font_family in ("", font_family) else style.font_family
            fs = style.font_size if style.font_size != 8.5 else font_size
            w, h, efs = compute_node_size(
                label,
                ff,
                fs,
                padding,
                shape=style.shape,
                font_weight=style.font_weight,
                font_style=style.font_style,
                text_wrap=style.text_wrap,
                text_max_width=style.text_max_width,
                text_transform=style.text_transform,
                overflow_policy=style.overflow_policy,
                min_font_size=style.min_font_size,
                label_format=style.label_format,
                text_rotation=style.text_rotation,
            )
            # For shrink_text/overflow, min_width and min_height act as both
            # floor AND cap: the node stays at that size and text adapts
            # (shrinks or overflows).  For expand_node (default), they are
            # floors only -- the node can grow beyond them.
            if style.overflow_policy in ("shrink_text", "overflow"):
                if style.min_width is not None:
                    w = style.min_width
                if style.min_height is not None:
                    h = style.min_height
            else:
                if style.min_width is not None:
                    w = max(w, style.min_width)
                if style.min_height is not None:
                    h = max(h, style.min_height)
            if style.overflow_policy == "shrink_text":
                efs = fit_font_size_to_node(
                    label,
                    w,
                    h,
                    font_family=ff,
                    font_size=fs,
                    padding=padding,
                    shape=style.shape,
                    font_weight=style.font_weight,
                    font_style=style.font_style,
                    text_wrap=style.text_wrap,
                    text_max_width=style.text_max_width,
                    text_transform=style.text_transform,
                    min_font_size=style.min_font_size,
                    label_format=style.label_format,
                    text_rotation=style.text_rotation,
                )
            # Enforce aspect ratio: grow the smaller dimension to match.
            # aspect_ratio = width / height, so height = width / ratio.
            if style.aspect_ratio is not None and style.aspect_ratio > 0:
                current_ratio = w / h if h > 0 else 1.0
                if current_ratio < style.aspect_ratio:
                    # Too tall -- widen to match
                    w = h * style.aspect_ratio
                else:
                    # Too wide -- heighten to match
                    h = w / style.aspect_ratio
            sizes.append([w, h])
            font_sizes.append(efs)

        self.node_sizes = torch.tensor(sizes, dtype=self.size_dtype)
        self.node_font_sizes = torch.tensor(font_sizes, dtype=torch.float32)
        self._node_sizes_revision = self.revision

    def get_style_for_node(self, idx: int) -> NodeStyle:
        """Get effective style for a node via 5-level cascade.

        Priority (highest first):
        1. Per-element override (node_styles[idx])
        2. Deepest cluster's member_node_style
        3. Graph default_node_style
        4. Theme type lookup
        5. Global defaults (dagua.configure())
        """
        per_element = self.node_styles[idx] if idx < len(self.node_styles) else None
        node_type = self.node_types[idx] if idx < len(self.node_types) else "default"
        theme_style = self._theme.get_node_style(node_type)

        # Collect cluster member styles (deepest first)
        cluster_member_styles = self._get_cluster_member_node_styles(idx)

        # Global defaults
        global_default = None
        try:
            from dagua.defaults import get_default_node_style_overrides

            overrides = get_default_node_style_overrides()
            if overrides:
                global_default = NodeStyle(**overrides)
        except ImportError:
            pass

        # Fast path: no cascade needed if only theme matters
        if (
            per_element is None
            and not cluster_member_styles
            and self.default_node_style is None
            and global_default is None
        ):
            return theme_style

        return resolve_node_style(
            per_element=per_element,
            cluster_member_styles=cluster_member_styles,
            theme_style=theme_style,
            graph_default=self.default_node_style,
            global_default=global_default,
        )

    def get_style_for_edge(self, idx: int) -> EdgeStyle:
        """Get effective style for an edge via 5-level cascade.

        Priority (highest first):
        1. Per-element override (edge_styles[idx])
        2. Cluster member_edge_style (deepest cluster containing source node)
        3. Graph default_edge_style
        4. Theme type lookup (back edge > edge_type > default)
        5. Global defaults (dagua.configure())
        """
        per_element = self.edge_styles[idx] if idx < len(self.edge_styles) else None

        # Determine edge type for theme lookup
        if (
            self._back_edge_mask is not None
            and idx < self._back_edge_mask.shape[0]
            and self._back_edge_mask[idx].item()
        ):
            edge_type = "back"
        else:
            edge_type = self.edge_types[idx] if idx < len(self.edge_types) else "default"
        theme_style = self._theme.get_edge_style(edge_type)

        # Collect cluster member edge styles for the source node
        cluster_member_styles = None
        self._finalize_edges()
        if (
            self._edge_index_tensor is not None
            and self._edge_index_tensor.numel() > 0
            and idx < self._edge_index_tensor.shape[1]
        ):
            src_idx = int(self._edge_index_tensor[0, idx].item())
            cluster_member_styles = self._get_cluster_member_edge_styles(src_idx)

        # Global defaults
        global_default = None
        try:
            from dagua.defaults import get_default_edge_style_overrides

            overrides = get_default_edge_style_overrides()
            if overrides:
                global_default = EdgeStyle(**overrides)
        except ImportError:
            pass

        # Fast path
        if (
            per_element is None
            and not cluster_member_styles
            and self.default_edge_style is None
            and global_default is None
        ):
            return theme_style

        return resolve_edge_style(
            per_element=per_element,
            cluster_member_styles=cluster_member_styles,
            theme_style=theme_style,
            graph_default=self.default_edge_style,
            global_default=global_default,
        )

    # --- Cluster member style helpers ---

    def _get_cluster_member_node_styles(self, node_idx: int) -> Optional[List[Optional[NodeStyle]]]:
        """Collect member_node_style from clusters containing this node, deepest first."""
        if not self.clusters:
            return None
        result = []
        for name, members in self.clusters.items():
            if isinstance(members, list) and node_idx in members:
                style = self.cluster_styles.get(name)
                if style is not None and hasattr(style, "member_node_style"):
                    result.append((self.cluster_depth(name), style.member_node_style))
        if not result:
            return None
        # Sort deepest first
        result.sort(key=lambda x: -x[0])
        return [s for _, s in result]

    def _get_cluster_member_edge_styles(self, node_idx: int) -> Optional[List[Optional[EdgeStyle]]]:
        """Collect member_edge_style from clusters containing this node, deepest first."""
        if not self.clusters:
            return None
        result = []
        for name, members in self.clusters.items():
            if isinstance(members, list) and node_idx in members:
                style = self.cluster_styles.get(name)
                if style is not None and hasattr(style, "member_edge_style"):
                    result.append((self.cluster_depth(name), style.member_edge_style))
        if not result:
            return None
        result.sort(key=lambda x: -x[0])
        return [s for _, s in result]

    # --- Pin, align, export helpers ---

    def pin(
        self,
        node_id: Any,
        x: Optional[float] = None,
        y: Optional[float] = None,
        weight: float = float("inf"),
    ) -> None:
        """Pin a node's position (soft or hard).

        Args:
            node_id: The node to pin.
            x: Target x position (None = unconstrained).
            y: Target y position (None = unconstrained).
            weight: Constraint strength (inf = hard pin).
        """
        from dagua.flex import Flex, LayoutFlex

        if self.flex is None:
            self.flex = LayoutFlex()
        if self.flex.pins is None:
            self.flex.pins = {}

        fx = Flex(target=x, weight=weight) if x is not None else None
        fy = Flex(target=y, weight=weight) if y is not None else None
        self.flex.pins[node_id] = (fx, fy)
        self.invalidate_layout()
        self._touch()

    def align(self, node_ids: List[Any], axis: str = "x", weight: float = 5.0) -> None:
        """Align a group of nodes on an axis.

        Args:
            node_ids: Nodes that should share the same x or y coordinate.
            axis: 'x' (vertical alignment) or 'y' (horizontal alignment).
            weight: Constraint strength.
        """
        from dagua.flex import AlignGroup, LayoutFlex

        if self.flex is None:
            self.flex = LayoutFlex()

        group = AlignGroup(nodes=list(node_ids), weight=weight)
        if axis == "x":
            if self.flex.align_x is None:
                self.flex.align_x = []
            self.flex.align_x.append(group)
        elif axis == "y":
            if self.flex.align_y is None:
                self.flex.align_y = []
            self.flex.align_y.append(group)
        else:
            raise ValueError(f"axis must be 'x' or 'y', got {axis!r}")
        self.invalidate_layout()
        self._touch()

    def export_style(self, path: str) -> None:
        """Export this graph's style settings to a YAML/JSON file."""
        import json as _json
        from pathlib import Path

        data: Dict[str, Any] = {}

        if self.default_node_style is not None:
            import dataclasses as _dc

            data["default_node_style"] = _dc.asdict(self.default_node_style)
        if self.default_edge_style is not None:
            import dataclasses as _dc

            data["default_edge_style"] = _dc.asdict(self.default_edge_style)
        if self.default_cluster_style is not None:
            import dataclasses as _dc

            data["default_cluster_style"] = _dc.asdict(self.default_cluster_style)

        p = Path(path)
        if p.suffix in (".yaml", ".yml"):
            try:
                import yaml  # type: ignore[import-untyped]
            except ImportError:
                raise ImportError("PyYAML required: pip install pyyaml")
            with open(path, "w") as f:
                yaml.dump(data, f, default_flow_style=False, sort_keys=False)
        else:
            with open(path, "w") as f:
                _json.dump(data, f, indent=2)

    # --- Cycle support ---

    @property
    def has_cycles(self) -> bool:
        """Whether this graph contains cycles. Lazy detection, cached."""
        mask = self.back_edge_mask
        return bool(mask is not None and bool(mask.any().item()))

    @property
    def is_cyclic(self) -> bool:
        """Return whether the graph contains one or more cycles.

        Returns
        -------
        bool
            ``True`` when cycle detection found any back edges.
        """
        return self.has_cycles

    @property
    def back_edge_mask(self) -> Optional[torch.Tensor]:
        """BoolTensor[E] marking back edges, or None for DAGs.

        Lazy: runs DFS on first access, caches until edges change.
        """
        self._finalize_edges()
        if self._back_edge_mask is not None:
            return self._back_edge_mask
        if self._edge_index_tensor is None or self._edge_index_tensor.numel() == 0:
            return None
        from dagua.layout.cycle import detect_back_edges

        mask = detect_back_edges(self._edge_index_tensor, self.num_nodes)
        if mask.any():
            self._back_edge_mask = mask
            return mask
        return None

    def set_back_edge_mask(self, mask: torch.Tensor) -> None:
        """Manually override which edges are back edges.

        Args:
            mask: BoolTensor of shape [E] matching current edge count.
        """
        self._finalize_edges()
        E = self._edge_index_tensor.shape[1] if self._edge_index_tensor is not None else 0
        if mask.shape[0] != E:
            raise ValueError(f"mask length {mask.shape[0]} != edge count {E}")
        self._back_edge_mask = mask.bool()

    def _prepare_for_layout(self) -> None:
        """Detect cycles, reverse back edges so the engine sees a DAG.

        Skips auto-detection for graphs > 1M nodes (O(V+E) DFS is too slow
        in Python). Users can still call set_back_edge_mask() for large graphs.
        """
        self._finalize_edges()
        # Use cached mask or auto-detect for reasonably sized graphs
        if self._back_edge_mask is not None:
            mask: Optional[torch.Tensor] = self._back_edge_mask
        elif self.num_nodes > 1_000_000:
            self._original_edge_index = None
            return
        else:
            mask = self.back_edge_mask
        assert mask is None or isinstance(mask, torch.Tensor)
        if mask is None or not mask.any():
            self._original_edge_index = None
            return
        from dagua.layout.cycle import make_acyclic

        assert self._edge_index_tensor is not None
        self._original_edge_index = self._edge_index_tensor.clone()
        self._edge_index_tensor = make_acyclic(self._edge_index_tensor, mask)

    def _restore_after_layout(self) -> None:
        """Restore original edge directions after layout."""
        if self._original_edge_index is not None:
            self._edge_index_tensor = self._original_edge_index
            self._original_edge_index = None

    def get_style_for_cluster(self, name: str) -> ClusterStyle:
        """Get effective style for a cluster.

        Parameters
        ----------
        name : str
            Cluster identifier.

        Returns
        -------
        ClusterStyle
            Effective cluster style using the cascade
            ``per-cluster > graph default > theme``.
        """
        per_cluster = self.cluster_styles.get(name)
        theme_style = self._theme.cluster_style
        if per_cluster is None and self.default_cluster_style is None:
            return theme_style
        return resolve_cluster_style(
            per_cluster=per_cluster,
            theme_style=theme_style,
            graph_default=self.default_cluster_style,
        )

    # --- Cluster hierarchy methods ---

    def cluster_depth(self, name: str) -> int:
        """Walk parent chain, return number of hops (0 = root-level cluster)."""
        d = 0
        cur: Optional[str] = name
        while cur is not None and self.cluster_parents.get(cur) is not None:
            cur = self.cluster_parents[cur]
            d += 1
        return d

    def cluster_children(self, name: str) -> List[str]:
        """Return direct children of a cluster."""
        return [c for c, p in self.cluster_parents.items() if p == name]

    def leaf_cluster_members(self, name: str) -> List[int]:
        """Recursively collect all leaf node indices (own members + children's members)."""
        from dagua.utils import collect_cluster_leaves

        members = self.clusters.get(name, [])
        if isinstance(members, dict):
            result = set(collect_cluster_leaves(members))
        else:
            result = set(members)
        for child in self.cluster_children(name):
            result.update(self.leaf_cluster_members(child))
        return sorted(result)

    @property
    def max_cluster_depth(self) -> int:
        """Maximum depth across all clusters (0 if no clusters or no hierarchy)."""
        if not self.clusters:
            return 0
        return max(self.cluster_depth(name) for name in self.clusters)

    @property
    def cluster_ids(self) -> Optional[torch.Tensor]:
        """Per-node [N] LongTensor, each node assigned to its deepest cluster (-1 = unassigned).

        Cluster indices correspond to the sorted order of cluster names.
        """
        if not self.clusters or self.num_nodes == 0:
            return None
        from dagua.utils import collect_cluster_leaves

        ids = torch.full((self.num_nodes,), -1, dtype=torch.long)
        node_depth = [-1] * self.num_nodes  # track deepest assignment per node
        cluster_name_list = sorted(self.clusters.keys())
        name_to_idx = {n: i for i, n in enumerate(cluster_name_list)}
        for name in cluster_name_list:
            members = self.clusters[name]
            if isinstance(members, dict):
                members = collect_cluster_leaves(members)
            depth = self.cluster_depth(name)
            for node_idx in members:
                if 0 <= node_idx < self.num_nodes and depth > node_depth[node_idx]:
                    ids[node_idx] = name_to_idx[name]
                    node_depth[node_idx] = depth
        return ids

    @property
    def graph_style(self) -> GraphStyle:
        """Get graph-level style from the theme."""
        return self._theme.graph_style

    def to(self, device: str) -> DaguaGraph:
        """Move graph tensors to a device.

        Parameters
        ----------
        device : str
            Torch device string such as ``"cpu"`` or ``"cuda"``.

        Returns
        -------
        DaguaGraph
            The graph instance with tensor fields moved in place.
        """
        self._finalize_edges()
        self._edge_index_tensor = self._edge_index_tensor.to(device)  # type: ignore[union-attr]
        if self.edge_weights is not None:
            self.edge_weights = self.edge_weights.to(device)
        if self.node_sizes is not None:
            self.node_sizes = self.node_sizes.to(device)
        if self.node_font_sizes is not None:
            self.node_font_sizes = self.node_font_sizes.to(device)
        return self

    # --- Class methods for construction ---

    @classmethod
    def from_edge_list(cls, edges: List[Tuple[Any, ...]], **kwargs) -> DaguaGraph:
        """Create a graph from edge tuples.

        Parameters
        ----------
        edges : List[Tuple[Any, ...]]
            Edge tuples of the form ``(source, target)`` or
            ``(source, target, weight)``.
        **kwargs : Any
            Additional keyword arguments forwarded to ``DaguaGraph``.

        Returns
        -------
        DaguaGraph
            Graph populated from the provided edge list.
        """
        num_nodes = kwargs.pop("num_nodes", None)
        g = cls(**kwargs)
        if num_nodes is not None:
            for i in range(num_nodes):
                g.add_node(i)
        for edge in edges:
            if len(edge) == 3:
                src, tgt, weight = edge
                g.add_edge(src, tgt, weight=weight)
            else:
                src, tgt = edge
                g.add_edge(src, tgt)
        g._finalize_edges()
        return g

    @classmethod
    def from_networkx(cls, nx_graph: Any, **kwargs: Any) -> DaguaGraph:
        """Create a graph from a NetworkX graph.

        Parameters
        ----------
        nx_graph : Any
            NetworkX graph object supplying node and edge attributes.
        **kwargs : Any
            Additional keyword arguments forwarded to ``DaguaGraph``.

        Returns
        -------
        DaguaGraph
            Graph populated from the NetworkX object.
        """
        g = cls(**kwargs)
        for node in nx_graph.nodes():
            label = nx_graph.nodes[node].get("label", str(node))
            node_type = nx_graph.nodes[node].get("type", "default")
            g.add_node(node, label=label, type=node_type)

        for u, v in nx_graph.edges():
            label = nx_graph.edges[u, v].get("label", None)
            weight = nx_graph.edges[u, v].get("weight", None)
            if weight is not None:
                weight = float(weight)
            g.add_edge(u, v, label=label, weight=weight)

        g._finalize_edges()
        return g

    @classmethod
    def from_edge_index(
        cls,
        edge_index: torch.Tensor,
        num_nodes: int,
        edge_weights: Optional[torch.Tensor] = None,
        **kwargs: Any,
    ) -> DaguaGraph:
        """Create a graph from a PyG-style edge tensor.

        Parameters
        ----------
        edge_index : torch.Tensor
            Edge tensor with shape ``[2, E]``.
        num_nodes : int
            Number of nodes in the graph.
        edge_weights : Optional[torch.Tensor], default=None
            Optional weight tensor with shape ``[E]``.
        **kwargs : Any
            Additional keyword arguments forwarded to ``DaguaGraph``.

        Returns
        -------
        DaguaGraph
            Graph populated from the provided tensors.

        Raises
        ------
        ValueError
            If ``edge_index`` contains invalid node indices or
            ``edge_weights`` does not match the edge count.
        """
        requested_dtype = kwargs.get("index_dtype")
        if requested_dtype is None:
            from dagua.defaults import get_default_index_dtype

            requested_dtype = get_default_index_dtype()
        requested_dtype = cls._normalize_index_dtype(requested_dtype)
        ei = edge_index.to(dtype=requested_dtype)
        if ei.numel() > 0:
            max_idx = ei.max().item()
            if max_idx >= num_nodes:
                raise ValueError(
                    f"edge_index contains index {max_idx} but num_nodes={num_nodes}. "
                    f"All indices must be < num_nodes."
                )
            min_idx = ei.min().item()
            if min_idx < 0:
                raise ValueError(
                    f"edge_index contains negative index {min_idx}. All indices must be >= 0."
                )

        g = cls(**kwargs)
        g.edge_index = ei
        g.num_nodes = num_nodes
        g.node_labels = [str(i) for i in range(num_nodes)]
        g.node_types = ["default"] * num_nodes
        g.node_styles = [None] * num_nodes
        g.edge_labels = [None] * edge_index.shape[1]
        g.edge_types = ["normal"] * edge_index.shape[1]
        g.edge_styles = [None] * edge_index.shape[1]
        g._id_to_index = {i: i for i in range(num_nodes)}
        g._index_to_id = list(range(num_nodes))
        if edge_weights is not None:
            if edge_weights.shape[0] != edge_index.shape[1]:
                raise ValueError(
                    f"edge_weights length {edge_weights.shape[0]} != "
                    f"edge count {edge_index.shape[1]}"
                )
            g.edge_weights = edge_weights.to(dtype=torch.float32)
        return g

    @classmethod
    def from_json(cls, data: Union[Dict, str, Any]) -> DaguaGraph:
        """Create graph from JSON (dict, JSON string, or .json file path).

        See ``dagua.io.graph_from_json`` for full documentation.
        """
        from dagua.io import graph_from_json

        return graph_from_json(data)

    def to_json(self) -> Dict[str, Any]:
        """Serialize this graph to a JSON-compatible dict.

        See ``dagua.io.graph_to_json`` for full documentation.
        """
        from dagua.io import graph_to_json

        return graph_to_json(self)

    @classmethod
    def load(cls, source) -> DaguaGraph:
        """Load graph from file (YAML/JSON), dict, or string.

        See ``dagua.io.load`` for full documentation.
        """
        from dagua.io import load

        return load(source)

    def save(self, path, format=None):
        """Save graph to file (YAML/JSON). Format auto-detected from extension.

        See ``dagua.io.save`` for full documentation.
        """
        from dagua.io import save

        save(self, path, format=format)

    @classmethod
    def from_yaml(cls, data) -> DaguaGraph:
        """Create graph from YAML string or .yaml file path.

        See ``dagua.io.graph_from_yaml`` for full documentation.
        """
        from dagua.io import graph_from_yaml

        return graph_from_yaml(data)

    def to_yaml(self, path=None) -> str:
        """Serialize to YAML string (or write to file if path given).

        See ``dagua.io.graph_to_yaml`` for full documentation.
        """
        from dagua.io import graph_to_yaml

        return graph_to_yaml(self, path)

    def to_networkx(self):
        """Export to NetworkX DiGraph. See ``dagua.io.to_networkx``."""
        from dagua.io import to_networkx

        return to_networkx(self)

    def to_igraph(self):
        """Export to igraph.Graph. See ``dagua.io.to_igraph``."""
        from dagua.io import to_igraph

        return to_igraph(self)

    def to_pyg(self):
        """Export to torch_geometric.data.Data. See ``dagua.io.to_pyg``."""
        from dagua.io import to_pyg

        return to_pyg(self)

    def to_scipy(self):
        """Export to scipy.sparse.csr_matrix. See ``dagua.io.to_scipy``."""
        from dagua.io import to_scipy

        return to_scipy(self)

    @classmethod
    def from_igraph(cls, ig_graph, **kwargs) -> DaguaGraph:
        """Create graph from igraph.Graph. See ``dagua.io.from_igraph``."""
        from dagua.io import from_igraph

        return from_igraph(ig_graph, **kwargs)

    @classmethod
    def from_scipy(cls, adj_matrix, labels=None, **kwargs) -> DaguaGraph:
        """Create graph from scipy sparse adjacency. See ``dagua.io.from_scipy``."""
        from dagua.io import from_scipy

        return from_scipy(adj_matrix, labels=labels, **kwargs)

    @classmethod
    def from_dot(cls, dot_string: str, **kwargs) -> DaguaGraph:
        """Create graph from DOT string. See ``dagua.io.from_dot``."""
        from dagua.io import from_dot

        return from_dot(dot_string, **kwargs)

    @classmethod
    def from_torchlens(
        cls,
        model_log,
        mode: str = "unrolled",
        show_buffer_layers: bool = False,
        direction: str = "BT",
        **kwargs,
    ) -> DaguaGraph:
        """Create graph from TorchLens ModelLog.

        Args:
            model_log: TorchLens ModelLog object
            mode: 'unrolled' (each pass = node) or 'rolled' (multi-pass collapsed)
            show_buffer_layers: whether to include buffer nodes
            direction: layout direction (BT = bottom-up, default for DNN)
        """
        g = cls(direction=direction, **kwargs)

        # Select entries based on mode
        if mode == "unrolled":
            entries = list(model_log.layer_dict_main_keys.values())
        else:
            entries = list(model_log.layer_logs.values())

        # Filter buffer layers
        if not show_buffer_layers:
            entries = [e for e in entries if not getattr(e, "is_buffer_layer", False)]

        # Add nodes
        for entry in entries:
            node_id = entry.layer_label
            label = _build_torchlens_node_label(entry, mode)
            node_type = _classify_torchlens_node_type(entry)
            g.add_node(node_id, label=label, type=node_type)

            # Dashed border for nodes without input ancestry
            if not getattr(entry, "has_input_ancestor", True):
                style = NodeStyle(stroke_dash="dashed")
                g.node_styles[g._id_to_index[node_id]] = style

        node_ids = set(g._id_to_index.keys())

        # Add edges
        edges_seen = set()
        for entry in entries:
            parent_id = entry.layer_label
            if parent_id not in node_ids:
                continue

            for child_label in getattr(entry, "child_layers", None) or []:
                if child_label not in node_ids:
                    continue

                edge_key = (parent_id, child_label)
                if edge_key in edges_seen:
                    continue
                edges_seen.add(edge_key)

                edge_type = _classify_torchlens_edge_type(entry, child_label)
                edge_label = _build_torchlens_edge_label(edge_type)
                edge_style = None
                if not getattr(entry, "has_input_ancestor", True):
                    edge_style = EdgeStyle(style="dashed")

                g.add_edge(
                    parent_id, child_label, label=edge_label, type=edge_type, style=edge_style
                )

        # Build clusters from module nesting
        _build_torchlens_clusters(g, entries, model_log)

        return g


def _build_torchlens_node_label(entry, mode: str) -> str:
    """Build multi-line node label from TorchLens entry."""
    lines = []

    # Title
    if mode == "rolled" and getattr(entry, "num_passes", 1) > 1:
        lines.append(f"{entry.layer_label} (x{entry.num_passes})")
    else:
        lines.append(entry.layer_label)

    # Boolean indicator
    if getattr(entry, "is_terminal_bool_layer", False):
        val = getattr(entry, "scalar_bool_value", None)
        if val is not None:
            lines.append(str(val).upper())

    # Shape
    shape = getattr(entry, "tensor_shape", None)
    if shape:
        shape_str = "x".join(str(d) for d in shape)
        lines.append(shape_str)

    return "\n".join(lines)


def _classify_torchlens_node_type(entry) -> str:
    """Classify TorchLens entry into a node type for styling."""
    if getattr(entry, "is_input_layer", False):
        return "input"
    if getattr(entry, "is_output_layer", False):
        return "output"
    if getattr(entry, "is_buffer_layer", False):
        return "buffer"
    if getattr(entry, "is_terminal_bool_layer", False):
        return "bool"
    if getattr(entry, "uses_params", False):
        return "trainable_params"
    return "default"


def _classify_torchlens_edge_type(parent_entry, child_label: str) -> str:
    """Classify edge type from TorchLens conditional branch data."""
    then_children = getattr(parent_entry, "cond_branch_then_children", None) or []
    if child_label in then_children:
        return "then"
    if_children = getattr(parent_entry, "cond_branch_start_children", None) or []
    if child_label in if_children:
        return "if"
    if getattr(parent_entry, "is_buffer_layer", False):
        return "buffer"
    return "normal"


def _build_torchlens_edge_label(edge_type: str) -> Optional[str]:
    """Build edge label text."""
    if edge_type == "if":
        return "IF"
    elif edge_type == "then":
        return "THEN"
    return None


def _build_torchlens_clusters(g: DaguaGraph, entries, model_log) -> None:
    """Build hierarchical clusters from TorchLens module nesting."""
    for entry in entries:
        containing = getattr(entry, "containing_modules", None)
        if not containing:
            continue

        node_id = entry.layer_label
        if node_id not in g._id_to_index:
            continue
        node_idx = g._id_to_index[node_id]

        for module_addr in containing:
            cluster_name = (
                module_addr.split(":")[0] if ":" in str(module_addr) else str(module_addr)
            )
            if cluster_name == "self":
                continue

            if cluster_name not in g.clusters:
                g.clusters[cluster_name] = []
            if isinstance(g.clusters[cluster_name], list):
                if node_idx not in g.clusters[cluster_name]:
                    g.clusters[cluster_name].append(node_idx)

    # Infer cluster_parents from dot-separated module addresses.
    # E.g. "1.conv1" -> parent "1", "encoder.layer.0.attn" -> parent "encoder.layer.0".
    # Only set parent if the parent cluster actually exists in g.clusters.
    for cluster_name in list(g.clusters):
        if "." in cluster_name:
            parent_name = cluster_name.rsplit(".", 1)[0]
            if parent_name in g.clusters and cluster_name not in g.cluster_parents:
                g.cluster_parents[cluster_name] = parent_name

    # Add cluster labels from module metadata
    if hasattr(model_log, "modules"):
        try:
            for addr, module_log in model_log.modules._module_logs.items():
                if addr == "self":
                    continue
                class_name = getattr(module_log, "module_class_name", "")
                if class_name:
                    g.cluster_labels[addr] = f"{addr} ({class_name})"
                else:
                    g.cluster_labels[addr] = addr
        except Exception:
            pass
