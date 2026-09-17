"""Immutable scene, semantic, style, and result contracts for RULER V4."""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, FrozenSet, Mapping, Optional, Tuple, Union

import torch

from dagua.eval.ruler_v4._tracing import (
    TRACED_BOUND_TOLERANCE,
    Scalar,
    SurrogateTraceError,
    record_subterm,
)


class IngestionErrorCode(str, Enum):
    """Typed invalid-scene error codes."""

    MALFORMED_POSITIONS = "malformed_positions"
    NONFINITE_GEOMETRY = "nonfinite_geometry"
    TOPOLOGY_MISMATCH = "topology_mismatch"
    MISSING_REQUIRED_PRIMITIVE = "missing_required_primitive"
    IMPOSSIBLE_STYLE = "impossible_style"
    UNSUPPORTED_COMPOSITING = "unsupported_compositing"
    PRODUCER_EXTENTS_FORBIDDEN = "producer_extents_forbidden"
    MALFORMED_ROUTE = "malformed_route"
    MALFORMED_METADATA = "malformed_metadata"


class ResultState(str, Enum):
    """Facet execution states."""

    VALUE = "VALUE"
    NA = "NA"
    INVALID = "INVALID"


@dataclass(frozen=True)
class Route:
    """One producer-owned vector edge route.

    Parameters
    ----------
    edge_index : int
        Index of the corresponding declared graph edge.
    points : torch.Tensor
        Route vertices with shape ``[P, 2]`` and float64 coordinates.
    kind : str
        Declared route geometry kind. Phase 1 accepts flattened polylines.
    """

    edge_index: int
    points: torch.Tensor
    kind: str = "polyline"


@dataclass(frozen=True)
class ObservationProfile:
    """Input-owned declaration of channels visible to one observation profile.

    Parameters
    ----------
    name : str
        Stable profile name.
    visible_channels : frozenset[str]
        Channels that are visible in this profile.
    required_channels : frozenset[str]
        Channels that every valid drawing must carry.
    optional_channels : frozenset[str]
        Channels whose graph-wide absence is a valid NA condition.
    scale_normalized : bool
        Whether position scale has been removed by the profile contract.
    viewport : tuple[float, float] or None
        Optional physical viewport dimensions.
    """

    name: str = "ordinary"
    visible_channels: FrozenSet[str] = frozenset({"nodes", "routes"})
    required_channels: FrozenSet[str] = frozenset({"nodes"})
    optional_channels: FrozenSet[str] = frozenset()
    scale_normalized: bool = False
    viewport: Optional[Tuple[float, float]] = None


@dataclass(frozen=True)
class PortDeclaration:
    """One immutable semantic port declaration.

    Parameters
    ----------
    port_id : str
        Canonical port identifier.
    node_id : int
        Owning canonical node index.
    side : str
        Cardinal node-local side ``N``, ``E``, ``S``, or ``W``.
    side_coordinate : float
        Anchor coordinate in ``[0, 1]`` along the declared side.
    order : int
        Total order within the node side.
    normal : tuple[float, float]
        Declared outward unit normal.
    expected_approach : tuple[float, float]
        Expected unit route tangent pointing away from the node.
    """

    port_id: str
    node_id: int
    side: str
    side_coordinate: float
    order: int
    normal: Tuple[float, float]
    expected_approach: Tuple[float, float]


@dataclass(frozen=True)
class ChannelDeclaration:
    """One corpus-owned non-geometric visual channel.

    Parameters
    ----------
    channel_id : str
        Stable channel identifier.
    primitive_kind : str
        Target population, ``node`` or ``edge``.
    attribute : str
        GraphSemantics categorical-attribute key encoded by the channel.
    visual_property : str
        Encoded property, currently ``fill_color`` or ``stroke_color``.
    value_map : mapping[str, tuple[float, float, float]]
        Category-to-sRGB map with channels in ``[0, 1]``.
    """

    channel_id: str
    primitive_kind: str
    attribute: str
    visual_property: str
    value_map: Mapping[str, Tuple[float, float, float]]


@dataclass(frozen=True)
class GraphSemantics:
    """Corpus-owned graph and optional semantic declarations.

    Parameters
    ----------
    node_ids : tuple[str, ...]
        Canonical node identifiers.
    edges : tuple[tuple[int, int], ...]
        Canonical directed or undirected edge multiset.
    directed : bool
        Whether edge direction is declared.
    node_labels : tuple[str | None, ...]
        Optional declared node label strings.
    edge_labels : tuple[str | None, ...]
        Optional declared edge label strings.
    clusters : mapping[str, tuple[int, ...]]
        Declared cluster memberships.
    cluster_parents : mapping[str, str]
        Child-to-parent cluster hierarchy.
    ranks : tuple[int, ...] or None
        Optional declared node ranks.
    roots : tuple[int, ...]
        Optional declared roots.
    feedback : tuple[bool, ...] or None
        Optional immutable per-edge feedback mask.
    edge_weights : tuple[float, ...] or None
        Optional positive declared edge weights.
    edge_styles : tuple[str, ...] or None
        Optional immutable per-edge route style declarations.
    edge_bundles : tuple[str | None, ...] or None
        Optional immutable per-edge declared bundle ids.
    node_attributes : mapping[str, tuple[str, ...]]
        Optional declared categorical values in canonical node order.
    edge_attributes : mapping[str, tuple[str, ...]]
        Optional declared categorical values in canonical edge order.
    legends : mapping[str, mapping[str, tuple[float, float, float]]]
        Declared channel legends used for completeness validation.
    weight_semantics : str or None
        Interpretation of declared edge weights.
    ports : mapping[int, tuple[PortDeclaration | None, PortDeclaration | None]]
        Optional edge endpoint port declarations.
    temporal_ids : tuple[str, ...] or None
        Optional cross-frame node identities.
    required_primitives : frozenset[str]
        Render channels required by graph semantics.
    planarity_certificate : mapping[str, Any] or None
        Optional input-owned planarity certificate.
    flow_axis : tuple[float, float] or None
        Optional input-owned unit direction axis.
    symmetry_generators : tuple[tuple[int, ...], ...]
        Certified non-identity automorphism permutations.
    node_masses : tuple[float, ...] or None
        Optional positive input-owned node masses.
    declared_graph_class : str or None
        Optional input-owned graph class used by frozen facet exemptions.
    lattice_dimensions : tuple[int, int] or None
        Optional positive declared dimensions for lattice/grid classes.
    tree_parents : tuple[int | None, ...] or None
        Optional declared parent per node for rooted tree semantics.
    tree_depths : tuple[int, ...] or None
        Optional declared integer tree depth per node.
    tree_layout : str or None
        Declared tree display mode, ``layered`` or ``radial``.
    ordered_children : mapping[int, tuple[int, ...]]
        Optional immutable child order per parent.
    weight_visual_channel : str or None
        Optional declared visual encoding channel for edge weights.
    weight_encoding_knots : tuple[tuple[float, float], ...]
        Positive ``(weight, target_width)`` knots for log-log interpolation.
    """

    node_ids: Tuple[str, ...]
    edges: Tuple[Tuple[int, int], ...]
    directed: bool = False
    node_labels: Tuple[Optional[str], ...] = ()
    edge_labels: Tuple[Optional[str], ...] = ()
    clusters: Mapping[str, Tuple[int, ...]] = field(default_factory=dict)
    cluster_parents: Mapping[str, str] = field(default_factory=dict)
    ranks: Optional[Tuple[int, ...]] = None
    roots: Tuple[int, ...] = ()
    feedback: Optional[Tuple[bool, ...]] = None
    edge_weights: Optional[Tuple[float, ...]] = None
    edge_styles: Optional[Tuple[str, ...]] = None
    edge_bundles: Optional[Tuple[Optional[str], ...]] = None
    node_attributes: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    edge_attributes: Mapping[str, Tuple[str, ...]] = field(default_factory=dict)
    legends: Mapping[str, Mapping[str, Tuple[float, float, float]]] = field(default_factory=dict)
    weight_semantics: Optional[str] = None
    ports: Mapping[int, Tuple[Optional[PortDeclaration], Optional[PortDeclaration]]] = field(
        default_factory=dict
    )
    temporal_ids: Optional[Tuple[str, ...]] = None
    required_primitives: FrozenSet[str] = frozenset({"nodes"})
    planarity_certificate: Optional[Mapping[str, Any]] = None
    flow_axis: Optional[Tuple[float, float]] = None
    symmetry_generators: Tuple[Tuple[int, ...], ...] = ()
    node_masses: Optional[Tuple[float, ...]] = None
    declared_graph_class: Optional[str] = None
    lattice_dimensions: Optional[Tuple[int, int]] = None
    tree_parents: Optional[Tuple[Optional[int], ...]] = None
    tree_depths: Optional[Tuple[int, ...]] = None
    tree_layout: Optional[str] = None
    ordered_children: Mapping[int, Tuple[int, ...]] = field(default_factory=dict)
    weight_visual_channel: Optional[str] = None
    weight_encoding_knots: Tuple[Tuple[float, float], ...] = ()


@dataclass(frozen=True)
class StyleContract:
    """Corpus-owned extent and primitive model.

    Parameters
    ----------
    font_size : float
        Node-label font size in canonical scene units.
    average_character_width : float
        Character width as a multiple of ``font_size``.
    line_height : float
        Label line height as a multiple of ``font_size``.
    padding_x, padding_y : float
        Node padding in canonical scene units.
    minimum_node_width, minimum_node_height : float
        Minimum visible node dimensions.
    route_stroke_width : float
        Declared visible route stroke width.
    label_gap : float
        Gap between node center and externally placed label center.
    coordinate_scale : float
        Unit re-expression scale applied to all dimensional style fields.
    opaque : bool
        Whether all declared primitives use the supported opaque model.
    allowed_route_kinds : frozenset[str]
        Route kinds accepted at ingestion.
    flattening_tolerance : float
        Input-owned route flattening tolerance.
    edge_stroke_widths : tuple[float, ...]
        Corpus-owned derived visible width for each semantic edge.
    minimum_feature_separation : float
        Smallest visible feature separation in intrinsic-unit multiples.
    physical_output : mapping[str, float] or None
        Optional complete physical viewport, font-height, and legibility-floor block.
    channel_set : tuple[ChannelDeclaration, ...]
        Corpus-owned non-geometric visual channel declarations.
    canvas_background : tuple[float, float, float]
        Opaque sRGB canvas backdrop.
    """

    font_size: float = 1.0
    average_character_width: float = 0.52
    line_height: float = 1.0
    padding_x: float = 0.25
    padding_y: float = 0.20
    minimum_node_width: float = 1.0
    minimum_node_height: float = 1.0
    route_stroke_width: float = 0.08
    label_gap: float = 0.20
    coordinate_scale: float = 1.0
    opaque: bool = True
    allowed_route_kinds: FrozenSet[str] = frozenset({"polyline"})
    flattening_tolerance: float = 1e-3
    edge_stroke_widths: Tuple[float, ...] = ()
    minimum_feature_separation: float = 0.05
    physical_output: Optional[Mapping[str, float]] = None
    channel_set: Tuple[ChannelDeclaration, ...] = ()
    canvas_background: Tuple[float, float, float] = (1.0, 1.0, 1.0)


@dataclass(frozen=True)
class DrawingScene:
    """Producer-owned drawing fields.

    Parameters
    ----------
    positions : torch.Tensor
        Node centers with shape ``[N, 2]`` in scene coordinates.
    routes : tuple[Route, ...]
        Exact flattened visible routes.
    z_order : tuple[str, ...]
        Back-to-front primitive identifiers.
    node_label_offsets : torch.Tensor or None
        Optional external label offsets with shape ``[N, 2]``.
    edge_label_positions : torch.Tensor or None
        Optional edge-label centers with shape ``[E, 2]``.
    """

    positions: torch.Tensor
    routes: Tuple[Route, ...] = ()
    z_order: Tuple[str, ...] = ()
    node_label_offsets: Optional[torch.Tensor] = None
    edge_label_positions: Optional[torch.Tensor] = None


@dataclass(frozen=True)
class BoxGeometry:
    """One derived axis-aligned primitive box.

    Parameters
    ----------
    center : torch.Tensor
        Box center with shape ``[2]``.
    half_extents : torch.Tensor
        Positive half-width and half-height with shape ``[2]``.
    owner : int
        Owning node or edge index.
    """

    center: torch.Tensor
    half_extents: torch.Tensor
    owner: int


@dataclass(frozen=True)
class Scene:
    """Validated canonical scene consumed by facets.

    Parameters
    ----------
    graph : GraphSemantics
        Canonical input-owned semantics.
    style : StyleContract
        Canonical input-owned primitive contract.
    profile : ObservationProfile
        Active observation profile.
    positions : torch.Tensor
        Validated float64 positions with shape ``[N, 2]``.
    routes : tuple[Route, ...]
        Validated float64 routes.
    z_order : tuple[str, ...]
        Producer-supplied primitive ordering.
    node_boxes : tuple[BoxGeometry, ...]
        Style-derived node boxes.
    node_label_boxes : tuple[BoxGeometry, ...]
        Style-derived visible node-label boxes.
    edge_label_boxes : tuple[BoxGeometry, ...]
        Style-derived visible edge-label boxes.
    cluster_label_boxes : mapping[str, BoxGeometry]
        Style- and region-derived visible cluster-label boxes.
    intrinsic_unit : float
        Median diagonal of declared node primitives.
    profile_hash : str
        Canonical hash of graph/profile/style inputs.
    graph_hash : str
        Canonical hash of graph semantics alone.
    """

    graph: GraphSemantics
    style: StyleContract
    profile: ObservationProfile
    positions: torch.Tensor
    routes: Tuple[Route, ...]
    z_order: Tuple[str, ...]
    node_boxes: Tuple[BoxGeometry, ...]
    node_label_boxes: Tuple[BoxGeometry, ...]
    edge_label_boxes: Tuple[BoxGeometry, ...]
    cluster_label_boxes: Mapping[str, BoxGeometry]
    intrinsic_unit: float
    profile_hash: str
    graph_hash: str

    @property
    def node_count(self) -> int:
        """Return the number of nodes.

        Returns
        -------
        int
            Number of canonical nodes.
        """

        return len(self.graph.node_ids)

    @property
    def edge_count(self) -> int:
        """Return the number of declared edges.

        Returns
        -------
        int
            Number of edges in the multiset.
        """

        return len(self.graph.edges)


@dataclass(frozen=True)
class ValidScene:
    """Successful ingestion result.

    Parameters
    ----------
    scene : Scene
        Fully validated scene.
    """

    scene: Scene


@dataclass(frozen=True)
class ValidAbsence:
    """Valid graph-wide absence of an optional semantic block.

    Parameters
    ----------
    reason : str
        Machine-readable NA reason.
    missing_channels : tuple[str, ...]
        Optional channels absent for every drawing under the profile.
    """

    reason: str
    missing_channels: Tuple[str, ...]


@dataclass(frozen=True)
class InvalidScene:
    """Typed malformed-scene ingestion result.

    Parameters
    ----------
    code : IngestionErrorCode
        Stable invalidity category.
    message : str
        Human-readable detail without a score.
    path : str or None
        Field path responsible for the error.
    """

    code: IngestionErrorCode
    message: str
    path: Optional[str] = None


@dataclass(frozen=True)
class FacetResult:
    """One independent facet result.

    Parameters
    ----------
    state : ResultState
        Value, valid absence, or typed invalid state.
    value : float or None
        Defect in ``[0, 1]`` for value results.
    reason : str or None
        Machine-readable NA or invalid reason.
    subterms : mapping[str, float]
        Score-visible sub-term values keyed by manifest id.
    raw : mapping[str, Any]
        Published sufficient statistics and diagnostics.
    temporal_headline : float or None
        Sequence-only headline used by temporal facets and excluded from static scoring.
    """

    state: ResultState
    value: Optional[float]
    reason: Optional[str]
    subterms: Mapping[str, float] = field(default_factory=dict)
    raw: Mapping[str, Any] = field(default_factory=dict)
    temporal_headline: Optional[float] = None


@dataclass(frozen=True)
class TemporalTransition:
    """Input-owned semantics for one consecutive frame transition.

    Parameters
    ----------
    states : mapping[str, str]
        Temporal node id to ``unchanged``, ``changed``, ``enter``, or ``exit``.
    expected_displacements : mapping[str, float]
        Expected displacement in intrinsic-unit multiples for common nodes.
    elapsed_time : float
        Positive input-owned transition duration.
    """

    states: Mapping[str, str]
    expected_displacements: Mapping[str, float]
    elapsed_time: float = 1.0


@dataclass(frozen=True)
class TemporalScene:
    """Validated ordered temporal drawing consumed by U40.

    Parameters
    ----------
    frames : tuple[Scene, ...]
        At least two validated static scenes in chronological order.
    transitions : tuple[TemporalTransition, ...]
        Exactly one input-owned declaration per consecutive frame pair.
    """

    frames: Tuple[Scene, ...]
    transitions: Tuple[TemporalTransition, ...]


@dataclass(frozen=True)
class ValidTemporalScene:
    """Successful temporal ingestion result.

    Parameters
    ----------
    scene : TemporalScene
        Fully validated temporal scene.
    """

    scene: TemporalScene


IngestionResult = Union[ValidScene, ValidAbsence, InvalidScene]
TemporalIngestionResult = Union[ValidTemporalScene, ValidAbsence, InvalidScene]


def value_result(
    value: Scalar,
    subterms: Optional[Mapping[str, Scalar]] = None,
    raw: Optional[Mapping[str, Any]] = None,
) -> FacetResult:
    """Build a finite, bounded facet value.

    Tensor-valued inputs occur only on the traced surrogate path (they can
    exist only inside a ``trace_subterms`` context, where the ``keep`` seam
    stops casting): each tensor subterm is recorded into the active trace
    buffer with its autograd graph intact, and the published ``FacetResult``
    carries the detached float exactly as before, so the frozen result shape
    and validation are unchanged.

    Parameters
    ----------
    value : float or torch.Tensor
        Scalar defect expected in ``[0, 1]``.
    subterms : mapping[str, float or torch.Tensor] or None
        Optional scored sub-term mapping.
    raw : mapping[str, Any] or None
        Optional raw statistic mapping.

    Returns
    -------
    FacetResult
        Validated value result.

    Raises
    ------
    ValueError
        If the value or a sub-term is non-finite or out of range on the
        frozen float path.
    SurrogateTraceError
        If a TRACED (tensor-valued) value is non-finite or violates the
        ``[0, 1]`` bound by more than ``TRACED_BOUND_TOLERANCE``. A traced
        value past the bound by at most the tolerance is a saturating
        row's accumulation-order noise (the traced-vs-exact gap is 1-3
        ULP): it is clamped, in both the recorded tensor (the boundary
        clamp's zero gradient is the honest subgradient at saturation)
        and the published float, instead of aborting the trace with a
        bare ``ValueError`` from inside a facet.
    """

    def traced_scalar(label: str, item: torch.Tensor) -> Tuple[torch.Tensor, float]:
        """Validate one traced scalar with the ULP tolerance, clamping."""

        raw_value = float(item.detach().item())
        if not math.isfinite(raw_value):
            raise SurrogateTraceError(f"traced value for {label} is non-finite")
        if raw_value < -TRACED_BOUND_TOLERANCE or raw_value > 1.0 + TRACED_BOUND_TOLERANCE:
            raise SurrogateTraceError(
                f"traced value for {label} is {raw_value!r}, outside [0, 1] "
                f"beyond the {TRACED_BOUND_TOLERANCE} tolerance"
            )
        if 0.0 <= raw_value <= 1.0:
            return item, raw_value
        return item.clamp(0.0, 1.0), min(max(raw_value, 0.0), 1.0)

    values = {}
    for key, item in (subterms or {}).items():
        if isinstance(item, torch.Tensor):
            tensor, published = traced_scalar(key, item)
            record_subterm(key, tensor)
            values[key] = published
        else:
            values[key] = float(item)
    if isinstance(value, torch.Tensor):
        _, headline = traced_scalar("the headline", value)
    else:
        headline = float(value)
    candidates = [headline, *values.values()]
    if any(
        not torch.isfinite(torch.tensor(item, dtype=torch.float64)).item() for item in candidates
    ):
        raise ValueError("facet values must be finite")
    if any(item < 0.0 or item > 1.0 for item in candidates):
        raise ValueError("facet values must lie in [0, 1]")
    return FacetResult(ResultState.VALUE, headline, None, values, dict(raw or {}))


def na_result(reason: str, raw: Optional[Mapping[str, Any]] = None) -> FacetResult:
    """Build a valid-absence facet result.

    Parameters
    ----------
    reason : str
        Machine-readable reason.
    raw : mapping[str, Any] or None
        Optional published context.

    Returns
    -------
    FacetResult
        NA result with no numeric score.
    """

    return FacetResult(ResultState.NA, None, reason, {}, dict(raw or {}))


def invalid_result(reason: str, raw: Optional[Mapping[str, Any]] = None) -> FacetResult:
    """Build a typed invalid facet result.

    Parameters
    ----------
    reason : str
        Machine-readable invalidity reason.
    raw : mapping[str, Any] or None
        Optional published context.

    Returns
    -------
    FacetResult
        Invalid result with no numeric score.
    """

    return FacetResult(ResultState.INVALID, None, reason, {}, dict(raw or {}))
