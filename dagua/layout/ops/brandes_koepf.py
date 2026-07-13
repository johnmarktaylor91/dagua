"""Dagre-compatible Brandes-Koepf horizontal coordinate assignment.

This module ports the coordinate stage from dagre.js 0.8.5.  It is kept
independent of Dagre's ranking and ordering stages so other layered engines,
including ELK, can reuse the four-alignment coordinate primitive.
"""

from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from typing import ClassVar, Dict, Hashable, List, Mapping, Optional, Sequence, Set, Tuple

from dagua.layout.ops.base import Op
from dagua.layout.ops.state import LayoutProblem, RuntimeContext, SolveState
from dagua.layout.ops.taxonomy import OpCategory, register_op

LayerNode = Hashable

BRANDES_KOEPF_LAYERING_KEY = "brandes_koepf_layering"
BRANDES_KOEPF_PREDECESSORS_KEY = "brandes_koepf_predecessors"
BRANDES_KOEPF_SUCCESSORS_KEY = "brandes_koepf_successors"
BRANDES_KOEPF_WIDTHS_KEY = "brandes_koepf_widths"
BRANDES_KOEPF_DUMMY_NODES_KEY = "brandes_koepf_dummy_nodes"
BRANDES_KOEPF_X_KEY = "brandes_koepf_x"


@dataclass(frozen=True)
class BrandesKoepfConfig:
    """Configure Dagre's four-alignment Brandes-Koepf solve.

    Parameters
    ----------
    node_sep : float, default=50.0
        Gap between adjacent real-node boxes.
    edge_sep : float, default=20.0
        Gap contributed by adjacent dummy edge nodes.
    align : str | None, default=None
        Optional alignment selector: ``UL``, ``UR``, ``DL``, or ``DR``.
        ``None`` balances the middle two of all four assignments.
    """

    node_sep: float = 50.0
    edge_sep: float = 20.0
    align: Optional[str] = None


def _add_conflict(conflicts: Set[frozenset[LayerNode]], left: LayerNode, right: LayerNode) -> None:
    """Record one symmetric alignment conflict.

    Parameters
    ----------
    conflicts : set[frozenset[Hashable]]
        Mutable conflict collection.
    left : Hashable
        First incident node.
    right : Hashable
        Second incident node.

    Returns
    -------
    None
        The conflict set is mutated in place.
    """
    conflicts.add(frozenset((left, right)))


def _find_type1_conflicts(
    layering: Sequence[Sequence[LayerNode]],
    predecessors: Mapping[LayerNode, Sequence[LayerNode]],
    dummy_nodes: Set[LayerNode],
) -> Set[frozenset[LayerNode]]:
    """Find non-inner segments that cross inner dummy segments.

    Parameters
    ----------
    layering : sequence[sequence[Hashable]]
        Ordered nodes for each rank.
    predecessors : mapping[Hashable, sequence[Hashable]]
        Ordered predecessor ids for each node.
    dummy_nodes : set[Hashable]
        Nodes representing normalized edge segments.

    Returns
    -------
    set[frozenset[Hashable]]
        Symmetric node-pair conflicts matching dagre.js ``findType1Conflicts``.
    """
    conflicts: Set[frozenset[LayerNode]] = set()
    if len(layering) < 2:
        return conflicts

    for previous_layer, layer in zip(layering, layering[1:]):
        previous_positions = {node: index for index, node in enumerate(previous_layer)}
        previous_inner_position = 0
        scan_position = 0
        last_node = layer[-1] if layer else None

        for layer_index, node in enumerate(layer):
            other_inner: Optional[LayerNode] = None
            if node in dummy_nodes:
                for predecessor in predecessors.get(node, ()):
                    if predecessor in dummy_nodes:
                        other_inner = predecessor
                        break
            next_inner_position = (
                previous_positions[other_inner] if other_inner is not None else len(previous_layer)
            )
            if other_inner is not None or node == last_node:
                for scan_node in layer[scan_position : layer_index + 1]:
                    for predecessor in predecessors.get(scan_node, ()):
                        predecessor_position = previous_positions[predecessor]
                        if (
                            predecessor_position < previous_inner_position
                            or next_inner_position < predecessor_position
                        ) and not (predecessor in dummy_nodes and scan_node in dummy_nodes):
                            _add_conflict(conflicts, predecessor, scan_node)
                scan_position = layer_index + 1
                previous_inner_position = next_inner_position
    return conflicts


def _vertical_alignment(
    layering: Sequence[Sequence[LayerNode]],
    conflicts: Set[frozenset[LayerNode]],
    neighbors: Mapping[LayerNode, Sequence[LayerNode]],
) -> Tuple[Dict[LayerNode, LayerNode], Dict[LayerNode, LayerNode]]:
    """Build vertical blocks by aligning nodes with median neighbors.

    Parameters
    ----------
    layering : sequence[sequence[Hashable]]
        Layering transformed for one vertical/horizontal orientation.
    conflicts : set[frozenset[Hashable]]
        Alignment conflicts found in the canonical layering.
    neighbors : mapping[Hashable, sequence[Hashable]]
        Predecessors for upper sweeps or successors for lower sweeps.

    Returns
    -------
    tuple[dict[Hashable, Hashable], dict[Hashable, Hashable]]
        Root and circular alignment maps.
    """
    root: Dict[LayerNode, LayerNode] = {}
    align: Dict[LayerNode, LayerNode] = {}
    positions: Dict[LayerNode, int] = {}
    for layer in layering:
        for order, node in enumerate(layer):
            root[node] = node
            align[node] = node
            positions[node] = order

    for layer in layering:
        previous_index = -1
        for node in layer:
            ordered_neighbors = sorted(neighbors.get(node, ()), key=positions.__getitem__)
            if not ordered_neighbors:
                continue
            median_position = (len(ordered_neighbors) - 1) / 2.0
            first_median = int(median_position // 1)
            last_median = int(-(-median_position // 1))
            for median_index in range(first_median, last_median + 1):
                neighbor = ordered_neighbors[median_index]
                if (
                    align[node] == node
                    and previous_index < positions[neighbor]
                    and frozenset((node, neighbor)) not in conflicts
                ):
                    align[neighbor] = node
                    align[node] = root[node] = root[neighbor]
                    previous_index = positions[neighbor]
    return root, align


def _separation(
    left: LayerNode,
    right: LayerNode,
    widths: Mapping[LayerNode, float],
    dummy_nodes: Set[LayerNode],
    node_sep: float,
    edge_sep: float,
) -> float:
    """Return Dagre's center separation for adjacent layer nodes.

    Parameters
    ----------
    left : Hashable
        Left node in the current oriented layer.
    right : Hashable
        Right node in the current oriented layer.
    widths : mapping[Hashable, float]
        Node widths in the adjusted coordinate system.
    dummy_nodes : set[Hashable]
        Normalized edge and self-edge dummy nodes.
    node_sep : float
        Real-node separation.
    edge_sep : float
        Dummy-edge separation.

    Returns
    -------
    float
        Required center-to-center distance.
    """
    left_gap = edge_sep if left in dummy_nodes else node_sep
    right_gap = edge_sep if right in dummy_nodes else node_sep
    return widths[left] / 2.0 + left_gap / 2.0 + right_gap / 2.0 + widths[right] / 2.0


def _block_graph(
    layering: Sequence[Sequence[LayerNode]],
    root: Mapping[LayerNode, LayerNode],
    widths: Mapping[LayerNode, float],
    dummy_nodes: Set[LayerNode],
    node_sep: float,
    edge_sep: float,
) -> Tuple[
    List[LayerNode],
    Dict[LayerNode, List[LayerNode]],
    Dict[LayerNode, List[LayerNode]],
    Dict[Tuple[LayerNode, LayerNode], float],
]:
    """Construct Dagre's separation-constraint block graph.

    Parameters
    ----------
    layering : sequence[sequence[Hashable]]
        Oriented layer ordering.
    root : mapping[Hashable, Hashable]
        Vertical-alignment root per node.
    widths : mapping[Hashable, float]
        Adjusted node widths.
    dummy_nodes : set[Hashable]
        Dummy nodes receiving edge separation.
    node_sep : float
        Real-node separation.
    edge_sep : float
        Dummy-edge separation.

    Returns
    -------
    tuple
        Block insertion order, predecessor lists, successor lists, and edge
        separation weights.
    """
    block_order: List[LayerNode] = []
    seen_blocks: Set[LayerNode] = set()
    predecessors: Dict[LayerNode, List[LayerNode]] = {}
    successors: Dict[LayerNode, List[LayerNode]] = {}
    weights: Dict[Tuple[LayerNode, LayerNode], float] = {}

    def add_block(block: LayerNode) -> None:
        """Add a block while preserving first-seen insertion order.

        Parameters
        ----------
        block : Hashable
            Alignment block root.

        Returns
        -------
        None
            Local graph collections are mutated.
        """
        if block not in seen_blocks:
            seen_blocks.add(block)
            block_order.append(block)
            predecessors[block] = []
            successors[block] = []

    for layer in layering:
        previous: Optional[LayerNode] = None
        for node in layer:
            node_root = root[node]
            add_block(node_root)
            if previous is not None:
                previous_root = root[previous]
                add_block(previous_root)
                edge = (previous_root, node_root)
                gap = _separation(
                    previous,
                    node,
                    widths=widths,
                    dummy_nodes=dummy_nodes,
                    node_sep=node_sep,
                    edge_sep=edge_sep,
                )
                if edge not in weights:
                    predecessors[node_root].append(previous_root)
                    successors[previous_root].append(node_root)
                    weights[edge] = gap
                else:
                    weights[edge] = max(weights[edge], gap)
            previous = node
    return block_order, predecessors, successors, weights


def _dependency_order(
    block_order: Sequence[LayerNode],
    adjacency: Mapping[LayerNode, Sequence[LayerNode]],
) -> List[LayerNode]:
    """Reproduce Dagre's iterative dependency traversal order.

    Parameters
    ----------
    block_order : sequence[Hashable]
        Block-graph insertion order.
    adjacency : mapping[Hashable, sequence[Hashable]]
        Dependencies to visit before each block.

    Returns
    -------
    list[Hashable]
        Nodes in the order their post-visit callback fires.
    """
    stack = list(block_order)
    visited: Set[LayerNode] = set()
    output: List[LayerNode] = []
    while stack:
        node = stack.pop()
        if node in visited:
            output.append(node)
            continue
        visited.add(node)
        stack.append(node)
        stack.extend(adjacency.get(node, ()))
    return output


def _horizontal_compaction(
    layering: Sequence[Sequence[LayerNode]],
    root: Mapping[LayerNode, LayerNode],
    align: Mapping[LayerNode, LayerNode],
    widths: Mapping[LayerNode, float],
    dummy_nodes: Set[LayerNode],
    node_sep: float,
    edge_sep: float,
) -> Dict[LayerNode, float]:
    """Compact one alignment with Dagre's modified two-pass block solve.

    Parameters
    ----------
    layering : sequence[sequence[Hashable]]
        Oriented layer ordering.
    root : mapping[Hashable, Hashable]
        Alignment root per node.
    align : mapping[Hashable, Hashable]
        Circular alignment links.
    widths : mapping[Hashable, float]
        Adjusted node widths.
    dummy_nodes : set[Hashable]
        Dummy nodes receiving edge separation.
    node_sep : float
        Real-node separation.
    edge_sep : float
        Dummy-edge separation.

    Returns
    -------
    dict[Hashable, float]
        Compacted x coordinate per node.
    """
    block_order, predecessors, successors, weights = _block_graph(
        layering,
        root,
        widths=widths,
        dummy_nodes=dummy_nodes,
        node_sep=node_sep,
        edge_sep=edge_sep,
    )
    x_coordinates: Dict[LayerNode, float] = {}
    for block in _dependency_order(block_order, predecessors):
        x_coordinates[block] = max(
            (
                x_coordinates[predecessor] + weights[(predecessor, block)]
                for predecessor in predecessors[block]
            ),
            default=0.0,
        )
    for block in _dependency_order(block_order, successors):
        upper_bound = min(
            (
                x_coordinates[successor] - weights[(block, successor)]
                for successor in successors[block]
            ),
            default=float("inf"),
        )
        if upper_bound != float("inf"):
            x_coordinates[block] = max(x_coordinates[block], upper_bound)
    return {node: x_coordinates[root_node] for node, root_node in root.items()}


def _alignment_width(
    x_coordinates: Mapping[LayerNode, float],
    widths: Mapping[LayerNode, float],
) -> float:
    """Return the bounding-box width of one assignment.

    Parameters
    ----------
    x_coordinates : mapping[Hashable, float]
        X coordinate per node.
    widths : mapping[Hashable, float]
        Node widths.

    Returns
    -------
    float
        Full horizontal span including half-width extents.
    """
    if not x_coordinates:
        return 0.0
    minimum = min(x_coordinates[node] - widths[node] / 2.0 for node in x_coordinates)
    maximum = max(x_coordinates[node] + widths[node] / 2.0 for node in x_coordinates)
    return maximum - minimum


def brandes_koepf_x_assignment(
    layering: Sequence[Sequence[LayerNode]],
    predecessors: Mapping[LayerNode, Sequence[LayerNode]],
    successors: Mapping[LayerNode, Sequence[LayerNode]],
    widths: Mapping[LayerNode, float],
    dummy_nodes: Set[LayerNode],
    node_sep: float = 50.0,
    edge_sep: float = 20.0,
    align: Optional[str] = None,
) -> Dict[LayerNode, float]:
    """Assign x coordinates using dagre.js 0.8.5 Brandes-Koepf semantics.

    Parameters
    ----------
    layering : sequence[sequence[Hashable]]
        Nodes grouped by rank in final order.
    predecessors : mapping[Hashable, sequence[Hashable]]
        Ordered predecessor ids for each node.
    successors : mapping[Hashable, sequence[Hashable]]
        Ordered successor ids for each node.
    widths : mapping[Hashable, float]
        Adjusted node widths.
    dummy_nodes : set[Hashable]
        Normalized edge and self-edge dummy nodes.
    node_sep : float, default=50.0
        Gap between real-node boxes.
    edge_sep : float, default=20.0
        Gap contributed by dummy edge nodes.
    align : str | None, default=None
        Optional ``UL``, ``UR``, ``DL``, or ``DR`` alignment selector.

    Returns
    -------
    dict[Hashable, float]
        Balanced x coordinate per node.

    Raises
    ------
    ValueError
        If an invalid alignment name is supplied or metadata is incomplete.
    """
    normalized_align = align.lower() if align is not None else None
    if normalized_align not in (None, "ul", "ur", "dl", "dr"):
        raise ValueError("align must be one of UL, UR, DL, DR, or None.")
    nodes = [node for layer in layering for node in layer]
    missing_widths = [node for node in nodes if node not in widths]
    if missing_widths:
        raise ValueError(f"Missing widths for Brandes-Koepf nodes: {missing_widths[:3]}")
    if not nodes:
        return {}

    conflicts = _find_type1_conflicts(layering, predecessors, dummy_nodes)
    assignments: "OrderedDict[str, Dict[LayerNode, float]]" = OrderedDict()
    for vertical in ("u", "d"):
        vertical_layers = [list(layer) for layer in layering]
        if vertical == "d":
            vertical_layers.reverse()
        neighbor_map = predecessors if vertical == "u" else successors
        for horizontal in ("l", "r"):
            oriented_layers = [list(layer) for layer in vertical_layers]
            if horizontal == "r":
                oriented_layers = [list(reversed(layer)) for layer in oriented_layers]
            root, alignment = _vertical_alignment(oriented_layers, conflicts, neighbor_map)
            coordinates = _horizontal_compaction(
                oriented_layers,
                root,
                alignment,
                widths=widths,
                dummy_nodes=dummy_nodes,
                node_sep=node_sep,
                edge_sep=edge_sep,
            )
            if horizontal == "r":
                coordinates = {node: -value for node, value in coordinates.items()}
            assignments[vertical + horizontal] = coordinates

    anchor_name = min(
        assignments,
        key=lambda name: _alignment_width(assignments[name], widths),
    )
    anchor = assignments[anchor_name]
    anchor_min = min(anchor.values())
    anchor_max = max(anchor.values())
    for name, coordinates in list(assignments.items()):
        if name == anchor_name:
            continue
        if name.endswith("l"):
            delta = anchor_min - min(coordinates.values())
        else:
            delta = anchor_max - max(coordinates.values())
        if delta:
            assignments[name] = {node: value + delta for node, value in coordinates.items()}

    if normalized_align is not None:
        return dict(assignments[normalized_align])
    balanced: Dict[LayerNode, float] = {}
    for node in nodes:
        samples = sorted(coordinates[node] for coordinates in assignments.values())
        balanced[node] = (samples[1] + samples[2]) / 2.0
    return balanced


@register_op
class BrandesKoepfXAssignment(Op):
    """Assign x coordinates from layered metadata stored in ``SolveState``."""

    name: ClassVar[str] = "brandes_koepf_x_assignment"
    category: ClassVar[OpCategory] = OpCategory.COORDINATE
    reads: ClassVar[Tuple[str, ...]] = ("extras",)
    writes: ClassVar[Tuple[str, ...]] = ("extras",)
    access_pattern: ClassVar[str] = "global"

    def __init__(self, config: Optional[BrandesKoepfConfig] = None) -> None:
        """Store coordinate-assignment options.

        Parameters
        ----------
        config : BrandesKoepfConfig | None, optional
            Dagre-compatible spacing and alignment controls.

        Returns
        -------
        None
            The immutable config is stored on the op.
        """
        self.config = config or BrandesKoepfConfig()

    def apply(
        self,
        problem: LayoutProblem,
        state: SolveState,
        ctx: RuntimeContext,
    ) -> SolveState:
        """Run the four-alignment coordinate solve.

        Parameters
        ----------
        problem : LayoutProblem
            Immutable problem inputs. The topology is prepared by a preceding
            layered-engine op.
        state : SolveState
            State containing Brandes-Koepf metadata in ``extras``.
        ctx : RuntimeContext
            Runtime infrastructure; unused because the port is deterministic.

        Returns
        -------
        SolveState
            State with ``brandes_koepf_x`` added to ``extras``.
        """
        del problem, ctx
        state.extras[BRANDES_KOEPF_X_KEY] = brandes_koepf_x_assignment(
            layering=state.extras[BRANDES_KOEPF_LAYERING_KEY],
            predecessors=state.extras[BRANDES_KOEPF_PREDECESSORS_KEY],
            successors=state.extras[BRANDES_KOEPF_SUCCESSORS_KEY],
            widths=state.extras[BRANDES_KOEPF_WIDTHS_KEY],
            dummy_nodes=state.extras[BRANDES_KOEPF_DUMMY_NODES_KEY],
            node_sep=self.config.node_sep,
            edge_sep=self.config.edge_sep,
            align=self.config.align,
        )
        return state


__all__ = [
    "BRANDES_KOEPF_DUMMY_NODES_KEY",
    "BRANDES_KOEPF_LAYERING_KEY",
    "BRANDES_KOEPF_PREDECESSORS_KEY",
    "BRANDES_KOEPF_SUCCESSORS_KEY",
    "BRANDES_KOEPF_WIDTHS_KEY",
    "BRANDES_KOEPF_X_KEY",
    "BrandesKoepfConfig",
    "BrandesKoepfXAssignment",
    "brandes_koepf_x_assignment",
]
