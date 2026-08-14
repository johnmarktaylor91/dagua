"""Three-way RULER V4 scene ingestion and canonical extent derivation."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import fields
from typing import Any, Dict, Mapping, Optional, Tuple

import torch

from dagua.eval.ruler_v4.scene import (
    BoxGeometry,
    DrawingScene,
    GraphSemantics,
    IngestionErrorCode,
    IngestionResult,
    InvalidScene,
    ObservationProfile,
    Route,
    Scene,
    StyleContract,
    TemporalIngestionResult,
    TemporalScene,
    TemporalTransition,
    ValidAbsence,
    ValidScene,
    ValidTemporalScene,
)

_FORBIDDEN_DRAWING_FIELDS = frozenset(
    {"half_extents", "node_boxes", "label_boxes", "stroke_widths", "marker_sizes", "glyph_metrics"}
)
_WEIGHT_SEMANTICS = frozenset(
    {"distance_cost", "connection_strength", "generator_declared_edge_weights"}
)
_PORT_DIRECTIONS = frozenset({"north", "east", "south", "west"})


def _invalid(code: IngestionErrorCode, message: str, path: Optional[str] = None) -> InvalidScene:
    """Construct a typed invalid result.

    Parameters
    ----------
    code : IngestionErrorCode
        Stable invalidity code.
    message : str
        Human-readable detail.
    path : str or None
        Responsible field path.

    Returns
    -------
    InvalidScene
        Typed error result.
    """

    return InvalidScene(code, message, path)


def _validate_graph(graph: GraphSemantics) -> Optional[InvalidScene]:
    """Validate graph-owned topology and optional metadata.

    Parameters
    ----------
    graph : GraphSemantics
        Graph declaration to validate.

    Returns
    -------
    InvalidScene or None
        First lexicographic invalidity, if any.
    """

    node_count = len(graph.node_ids)
    if len(set(graph.node_ids)) != node_count:
        return _invalid(
            IngestionErrorCode.MALFORMED_METADATA, "node_ids must be unique", "graph.node_ids"
        )
    for edge_index, (source, target) in enumerate(graph.edges):
        if source < 0 or target < 0 or source >= node_count or target >= node_count:
            return _invalid(
                IngestionErrorCode.TOPOLOGY_MISMATCH,
                "edge endpoint is outside the canonical node range",
                f"graph.edges[{edge_index}]",
            )
    optional_lengths = (
        ("node_labels", graph.node_labels, node_count),
        ("edge_labels", graph.edge_labels, len(graph.edges)),
        ("feedback", graph.feedback, len(graph.edges)),
        ("edge_weights", graph.edge_weights, len(graph.edges)),
        ("ranks", graph.ranks, node_count),
        ("temporal_ids", graph.temporal_ids, node_count),
    )
    for name, values, expected in optional_lengths:
        if values is not None and len(values) not in {0, expected}:
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                f"{name} length must be {expected}",
                f"graph.{name}",
            )
    if graph.edge_weights is not None:
        weights = torch.tensor(graph.edge_weights, dtype=torch.float64)
        if not bool(torch.isfinite(weights).all()) or bool((weights <= 0.0).any()):
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "edge weights must be finite and positive",
                "graph.edge_weights",
            )
        if graph.weight_semantics not in _WEIGHT_SEMANTICS:
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "weighted graphs require a closed-enum weight_semantics value",
                "graph.weight_semantics",
            )
    for name, members in graph.clusters.items():
        if not name or any(member < 0 or member >= node_count for member in members):
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "cluster members must reference canonical nodes",
                f"graph.clusters.{name}",
            )
    if graph.flow_axis is not None:
        axis = torch.tensor(graph.flow_axis, dtype=torch.float64)
        if (
            axis.shape != (2,)
            or not bool(torch.isfinite(axis).all())
            or not math.isclose(
                float(torch.linalg.vector_norm(axis)), 1.0, rel_tol=0.0, abs_tol=1e-12
            )
        ):
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "flow_axis must be a finite unit vector",
                "graph.flow_axis",
            )
    edge_multiset = sorted(graph.edges)
    for index, permutation in enumerate(graph.symmetry_generators):
        if sorted(permutation) != list(range(node_count)) or tuple(permutation) == tuple(
            range(node_count)
        ):
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "symmetry generators must be non-identity node permutations",
                f"graph.symmetry_generators[{index}]",
            )
        mapped = sorted(
            (permutation[source], permutation[target]) for source, target in graph.edges
        )
        if not graph.directed:
            mapped = sorted(tuple(sorted(edge)) for edge in mapped)
            reference = sorted(tuple(sorted(edge)) for edge in graph.edges)
        else:
            reference = edge_multiset
        if mapped != reference:
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "symmetry generator does not preserve graph topology",
                f"graph.symmetry_generators[{index}]",
            )
    if graph.weight_visual_channel is not None:
        if graph.weight_visual_channel != "stroke_thickness":
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "unsupported weight visual channel",
                "graph.weight_visual_channel",
            )
        if graph.edge_weights is None or len(graph.weight_encoding_knots) == 0:
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "thickness encoding requires weights and encoding knots",
                "graph.weight_encoding_knots",
            )
        knots = torch.tensor(graph.weight_encoding_knots, dtype=torch.float64)
        if (
            knots.ndim != 2
            or knots.shape[1] != 2
            or not bool(torch.isfinite(knots).all())
            or bool((knots <= 0.0).any())
            or (knots.shape[0] > 1 and bool((knots[1:, 0] <= knots[:-1, 0]).any()))
            or (knots.shape[0] > 1 and bool((knots[1:, 1] < knots[:-1, 1]).any()))
        ):
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "encoding knots must be positive and monotone",
                "graph.weight_encoding_knots",
            )
    for edge_index, directions in graph.ports.items():
        if (
            edge_index < 0
            or edge_index >= len(graph.edges)
            or len(directions) != 2
            or any(
                direction is not None and direction not in _PORT_DIRECTIONS
                for direction in directions
            )
        ):
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "ports must reference an edge and use supported cardinal directions",
                f"graph.ports.{edge_index}",
            )
    return None


def _validate_style(style: StyleContract) -> Optional[InvalidScene]:
    """Validate the input-owned primitive model.

    Parameters
    ----------
    style : StyleContract
        Style declaration to validate.

    Returns
    -------
    InvalidScene or None
        Typed error when dimensions or compositing are unsupported.
    """

    if not style.opaque:
        return _invalid(
            IngestionErrorCode.UNSUPPORTED_COMPOSITING,
            "v4.0 supports opaque primitives only",
            "style.opaque",
        )
    numeric = (
        style.font_size,
        style.average_character_width,
        style.line_height,
        style.padding_x,
        style.padding_y,
        style.minimum_node_width,
        style.minimum_node_height,
        style.route_stroke_width,
        style.coordinate_scale,
        style.flattening_tolerance,
    )
    tensor = torch.tensor(numeric, dtype=torch.float64)
    if not bool(torch.isfinite(tensor).all()) or bool((tensor <= 0.0).any()):
        return _invalid(
            IngestionErrorCode.IMPOSSIBLE_STYLE,
            "all dimensional style values must be finite and positive",
            "style",
        )
    if style.label_gap < 0.0 or not torch.isfinite(torch.tensor(style.label_gap)).item():
        return _invalid(
            IngestionErrorCode.IMPOSSIBLE_STYLE,
            "label_gap must be finite and nonnegative",
            "style.label_gap",
        )
    return None


def _clone_float64(value: torch.Tensor) -> torch.Tensor:
    """Clone geometry into canonical CPU float64 storage.

    Parameters
    ----------
    value : torch.Tensor
        Input geometry tensor.

    Returns
    -------
    torch.Tensor
        Detached contiguous CPU float64 tensor.
    """

    return value.detach().to(device="cpu", dtype=torch.float64).contiguous().clone()


def _text_extent(text: Optional[str], style: StyleContract) -> Tuple[float, float]:
    """Derive one label extent from frozen text metrics.

    Parameters
    ----------
    text : str or None
        Declared text.
    style : StyleContract
        Frozen metric model.

    Returns
    -------
    tuple[float, float]
        Full width and height in scene coordinates.
    """

    content = text or ""
    scale = style.coordinate_scale
    width = max(1, len(content)) * style.average_character_width * style.font_size * scale
    height = style.line_height * style.font_size * scale
    return width, height


def _derive_boxes(
    graph: GraphSemantics,
    drawing: DrawingScene,
    style: StyleContract,
    profile: ObservationProfile,
) -> Tuple[Tuple[BoxGeometry, ...], Tuple[BoxGeometry, ...], Tuple[BoxGeometry, ...]]:
    """Derive node and label boxes from graph text and the style contract.

    Parameters
    ----------
    graph : GraphSemantics
        Corpus-owned labels and topology.
    drawing : DrawingScene
        Producer-owned centers and optional label placements.
    style : StyleContract
        Corpus-owned extent model.
    profile : ObservationProfile
        Input-owned visible-channel declaration.

    Returns
    -------
    tuple
        Node boxes, node-label boxes, and edge-label boxes.
    """

    positions = _clone_float64(drawing.positions)
    labels = graph.node_labels or tuple(None for _ in graph.node_ids)
    node_boxes = []
    node_label_boxes = []
    for index, center in enumerate(positions):
        label_width, label_height = _text_extent(labels[index], style)
        scale = style.coordinate_scale
        node_width = max(
            style.minimum_node_width * scale, label_width + 2.0 * style.padding_x * scale
        )
        node_height = max(
            style.minimum_node_height * scale, label_height + 2.0 * style.padding_y * scale
        )
        node_boxes.append(
            BoxGeometry(
                center.clone(),
                torch.tensor([node_width / 2.0, node_height / 2.0], dtype=torch.float64),
                index,
            )
        )
        if labels[index] is not None and "node_labels" in profile.visible_channels:
            offset = (
                _clone_float64(drawing.node_label_offsets)[index]
                if drawing.node_label_offsets is not None
                else torch.zeros(2, dtype=torch.float64)
            )
            node_label_boxes.append(
                BoxGeometry(
                    center.clone() + offset,
                    torch.tensor([label_width / 2.0, label_height / 2.0], dtype=torch.float64),
                    index,
                )
            )
    edge_label_boxes = []
    if drawing.edge_label_positions is not None:
        edge_centers = _clone_float64(drawing.edge_label_positions)
        edge_labels = graph.edge_labels or tuple(None for _ in graph.edges)
        for index, center in enumerate(edge_centers):
            if edge_labels[index] is None:
                continue
            width, height = _text_extent(edge_labels[index], style)
            edge_label_boxes.append(
                BoxGeometry(
                    center.clone(),
                    torch.tensor([width / 2.0, height / 2.0], dtype=torch.float64),
                    index,
                )
            )
    return tuple(node_boxes), tuple(node_label_boxes), tuple(edge_label_boxes)


def _profile_hash(graph: GraphSemantics, style: StyleContract, profile: ObservationProfile) -> str:
    """Hash canonical input-owned comparison identity.

    Parameters
    ----------
    graph : GraphSemantics
        Graph declaration.
    style : StyleContract
        Style declaration.
    profile : ObservationProfile
        Observation profile.

    Returns
    -------
    str
        SHA-256 hexadecimal digest.
    """

    payload: Dict[str, Any] = {
        "node_ids": graph.node_ids,
        "edges": graph.edges,
        "directed": graph.directed,
        "node_labels": graph.node_labels,
        "edge_labels": graph.edge_labels,
        "clusters": sorted((key, tuple(value)) for key, value in graph.clusters.items()),
        "cluster_parents": sorted(graph.cluster_parents.items()),
        "ranks": graph.ranks,
        "roots": graph.roots,
        "feedback": graph.feedback,
        "edge_weights": graph.edge_weights,
        "weight_semantics": graph.weight_semantics,
        "ports": sorted((key, value) for key, value in graph.ports.items()),
        "temporal_ids": graph.temporal_ids,
        "required_primitives": sorted(graph.required_primitives),
        "flow_axis": graph.flow_axis,
        "symmetry_generators": graph.symmetry_generators,
        "weight_visual_channel": graph.weight_visual_channel,
        "weight_encoding_knots": graph.weight_encoding_knots,
        "style": {
            item.name: getattr(style, item.name)
            for item in fields(style)
            if item.name != "coordinate_scale"
        },
        "profile": {item.name: getattr(profile, item.name) for item in fields(profile)},
    }

    def normalize(value: Any) -> Any:
        """Normalize sets and mappings for canonical JSON serialization."""

        if isinstance(value, (set, frozenset)):
            return sorted(normalize(item) for item in value)
        if isinstance(value, Mapping):
            return {
                str(key): normalize(item)
                for key, item in sorted(value.items(), key=lambda pair: str(pair[0]))
            }
        if isinstance(value, tuple):
            return [normalize(item) for item in value]
        return value

    encoded = json.dumps(normalize(payload), sort_keys=True, separators=(",", ":"), allow_nan=False)
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def ingest(
    graph: GraphSemantics,
    drawing: DrawingScene,
    style: StyleContract,
    profile: ObservationProfile,
) -> IngestionResult:
    """Ingest one scene into valid, valid-absence, or invalid state.

    Parameters
    ----------
    graph : GraphSemantics
        Corpus-owned graph declaration.
    drawing : DrawingScene
        Producer-owned vector geometry.
    style : StyleContract
        Corpus-owned primitive extent model.
    profile : ObservationProfile
        Input-owned observation contract.

    Returns
    -------
    IngestionResult
        Exactly one of ``ValidScene``, ``ValidAbsence``, or ``InvalidScene``.
    """

    graph_error = _validate_graph(graph)
    if graph_error is not None:
        return graph_error
    style_error = _validate_style(style)
    if style_error is not None:
        return style_error
    if graph.weight_visual_channel == "stroke_thickness":
        if len(style.edge_stroke_widths) != len(graph.edges):
            return _invalid(
                IngestionErrorCode.MISSING_REQUIRED_PRIMITIVE,
                "thickness encoding requires one derived stroke width per edge",
                "style.edge_stroke_widths",
            )
        widths = torch.tensor(style.edge_stroke_widths, dtype=torch.float64)
        if not bool(torch.isfinite(widths).all()) or bool((widths <= 0.0).any()):
            return _invalid(
                IngestionErrorCode.IMPOSSIBLE_STYLE,
                "derived edge stroke widths must be finite and positive",
                "style.edge_stroke_widths",
            )
    positions = drawing.positions
    if not isinstance(positions, torch.Tensor) or positions.ndim != 2 or positions.shape[1] != 2:
        return _invalid(
            IngestionErrorCode.MALFORMED_POSITIONS,
            "positions must be a tensor with shape [N, 2]",
            "drawing.positions",
        )
    if positions.shape[0] != len(graph.node_ids):
        return _invalid(
            IngestionErrorCode.TOPOLOGY_MISMATCH,
            "position count does not match canonical node count",
            "drawing.positions",
        )
    positions64 = _clone_float64(positions)
    if not bool(torch.isfinite(positions64).all()):
        return _invalid(
            IngestionErrorCode.NONFINITE_GEOMETRY,
            "positions contain NaN or infinity",
            "drawing.positions",
        )
    available = {"nodes"}
    if drawing.routes:
        available.add("routes")
    if (
        any(label is not None for label in graph.node_labels)
        and "node_labels" in profile.visible_channels
    ):
        available.add("node_labels")
    if drawing.edge_label_positions is not None:
        available.add("edge_labels")
    required = set(profile.required_channels) | set(graph.required_primitives)
    missing_required = sorted(required - available)
    if missing_required:
        return _invalid(
            IngestionErrorCode.MISSING_REQUIRED_PRIMITIVE,
            f"missing required primitives: {', '.join(missing_required)}",
            "drawing",
        )
    missing_optional = tuple(sorted(set(profile.optional_channels) - available))
    if missing_optional:
        return ValidAbsence(f"no_declared_{missing_optional[0]}", missing_optional)
    if drawing.node_label_offsets is not None:
        offsets = drawing.node_label_offsets
        if offsets.shape != positions.shape or not bool(
            torch.isfinite(_clone_float64(offsets)).all()
        ):
            return _invalid(
                IngestionErrorCode.NONFINITE_GEOMETRY,
                "node label offsets must be finite with shape [N, 2]",
                "drawing.node_label_offsets",
            )
    if drawing.edge_label_positions is not None:
        label_positions = drawing.edge_label_positions
        expected = (len(graph.edges), 2)
        if tuple(label_positions.shape) != expected or not bool(
            torch.isfinite(_clone_float64(label_positions)).all()
        ):
            return _invalid(
                IngestionErrorCode.NONFINITE_GEOMETRY,
                "edge label positions must be finite with shape [E, 2]",
                "drawing.edge_label_positions",
            )
    seen_edges = set()
    routes = []
    for route_index, route in enumerate(drawing.routes):
        if (
            route.edge_index < 0
            or route.edge_index >= len(graph.edges)
            or route.edge_index in seen_edges
        ):
            return _invalid(
                IngestionErrorCode.MALFORMED_ROUTE,
                "routes must map uniquely to declared edges",
                f"drawing.routes[{route_index}].edge_index",
            )
        if route.kind not in style.allowed_route_kinds:
            return _invalid(
                IngestionErrorCode.MALFORMED_ROUTE,
                f"unsupported route kind: {route.kind}",
                f"drawing.routes[{route_index}].kind",
            )
        if (
            route.points.ndim != 2
            or tuple(route.points.shape)[1:] != (2,)
            or route.points.shape[0] < 2
        ):
            return _invalid(
                IngestionErrorCode.MALFORMED_ROUTE,
                "route points must have shape [P, 2] with P >= 2",
                f"drawing.routes[{route_index}].points",
            )
        points = _clone_float64(route.points)
        if not bool(torch.isfinite(points).all()):
            return _invalid(
                IngestionErrorCode.NONFINITE_GEOMETRY,
                "route contains NaN or infinity",
                f"drawing.routes[{route_index}].points",
            )
        seen_edges.add(route.edge_index)
        routes.append(Route(route.edge_index, points, route.kind))
    simple_keys = [tuple(sorted(edge)) for edge in graph.edges if edge[0] != edge[1]]
    has_parallel = len(simple_keys) != len(set(simple_keys))
    has_loop = any(source == target for source, target in graph.edges)
    if (has_parallel or has_loop) and seen_edges != set(range(len(graph.edges))):
        return _invalid(
            IngestionErrorCode.MISSING_REQUIRED_PRIMITIVE,
            "multiedges and self-loops require one distinct route per edge id",
            "drawing.routes",
        )
    if not set(graph.ports).issubset(seen_edges):
        return _invalid(
            IngestionErrorCode.MISSING_REQUIRED_PRIMITIVE,
            "every port-bound edge requires a visible route",
            "drawing.routes",
        )
    node_boxes, node_label_boxes, edge_label_boxes = _derive_boxes(graph, drawing, style, profile)
    diagonals = torch.stack(
        [2.0 * torch.linalg.vector_norm(box.half_extents) for box in node_boxes]
    )
    intrinsic_unit = float(torch.median(diagonals).item())
    if intrinsic_unit <= 0.0 or not torch.isfinite(torch.tensor(intrinsic_unit)).item():
        return _invalid(
            IngestionErrorCode.IMPOSSIBLE_STYLE,
            "derived primitive diagonal must be finite and positive",
            "style",
        )
    scene = Scene(
        graph=graph,
        style=style,
        profile=profile,
        positions=positions64,
        routes=tuple(routes),
        z_order=tuple(drawing.z_order),
        node_boxes=node_boxes,
        node_label_boxes=node_label_boxes,
        edge_label_boxes=edge_label_boxes,
        intrinsic_unit=intrinsic_unit,
        profile_hash=_profile_hash(graph, style, profile),
    )
    return ValidScene(scene)


def ingest_temporal(
    frames: Tuple[Scene, ...], transitions: Tuple[TemporalTransition, ...]
) -> TemporalIngestionResult:
    """Validate an ordered temporal scene without re-ingesting static frames.

    Parameters
    ----------
    frames : tuple[Scene, ...]
        Previously validated static frames in chronological order.
    transitions : tuple[TemporalTransition, ...]
        Input-owned declarations for consecutive frame pairs.

    Returns
    -------
    TemporalIngestionResult
        Valid temporal scene, typed valid absence, or typed invalid scene.
    """

    if len(frames) < 2:
        return ValidAbsence("TEMPORAL_PROFILE_ABSENT", ("temporal_frames",))
    if len(transitions) != len(frames) - 1:
        return _invalid(
            IngestionErrorCode.MALFORMED_METADATA,
            "temporal transition count must be frame count minus one",
            "transitions",
        )
    first = frames[0]
    if any(
        frame.style != first.style or frame.profile.name != first.profile.name for frame in frames
    ):
        return _invalid(
            IngestionErrorCode.MALFORMED_METADATA,
            "temporal frames must share one style and profile family",
            "frames",
        )
    for frame_index, frame in enumerate(frames):
        identifiers = frame.graph.temporal_ids
        if identifiers is None or len(set(identifiers)) != len(identifiers):
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "every temporal frame needs unique temporal_ids",
                f"frames[{frame_index}].graph.temporal_ids",
            )
    allowed_states = frozenset({"unchanged", "changed", "enter", "exit"})
    for index, (before, after, transition) in enumerate(zip(frames, frames[1:], transitions)):
        if transition.elapsed_time <= 0.0 or not math.isfinite(transition.elapsed_time):
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "elapsed_time must be finite and positive",
                f"transitions[{index}].elapsed_time",
            )
        before_ids = set(before.graph.temporal_ids or ())
        after_ids = set(after.graph.temporal_ids or ())
        expected_ids = before_ids | after_ids
        if set(transition.states) != expected_ids or any(
            state not in allowed_states for state in transition.states.values()
        ):
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "transition states must cover exactly the temporal-id union",
                f"transitions[{index}].states",
            )
        common = before_ids & after_ids
        if set(transition.expected_displacements) != common:
            return _invalid(
                IngestionErrorCode.MALFORMED_METADATA,
                "expected displacement must cover exactly the common temporal ids",
                f"transitions[{index}].expected_displacements",
            )
        for identifier in expected_ids:
            state = transition.states[identifier]
            if identifier not in before_ids:
                expected_state = "enter"
            elif identifier not in after_ids:
                expected_state = "exit"
            else:
                expected_state = None
            if expected_state is not None and state != expected_state:
                return _invalid(
                    IngestionErrorCode.MALFORMED_METADATA,
                    "enter/exit state contradicts frame identities",
                    f"transitions[{index}].states.{identifier}",
                )
        for identifier, magnitude in transition.expected_displacements.items():
            state = transition.states[identifier]
            if (
                magnitude < 0.0
                or not math.isfinite(magnitude)
                or (state == "unchanged" and magnitude != 0.0)
                or (state == "changed" and magnitude <= 0.0)
                or state not in {"unchanged", "changed"}
            ):
                return _invalid(
                    IngestionErrorCode.MALFORMED_METADATA,
                    "typed displacement magnitude contradicts node state",
                    f"transitions[{index}].expected_displacements.{identifier}",
                )
    return ValidTemporalScene(TemporalScene(frames, transitions))


def ingest_record(
    graph: GraphSemantics,
    record: Mapping[str, Any],
    style: StyleContract,
    profile: ObservationProfile,
) -> IngestionResult:
    """Ingest an untrusted mapping while rejecting producer-owned extent fields.

    Parameters
    ----------
    graph : GraphSemantics
        Corpus-owned graph declaration.
    record : mapping[str, Any]
        Untrusted DrawingScene-shaped record.
    style : StyleContract
        Corpus-owned extent contract.
    profile : ObservationProfile
        Input-owned observation contract.

    Returns
    -------
    IngestionResult
        Three-way ingestion outcome.
    """

    forbidden = sorted(_FORBIDDEN_DRAWING_FIELDS.intersection(record))
    if forbidden:
        return _invalid(
            IngestionErrorCode.PRODUCER_EXTENTS_FORBIDDEN,
            f"producer extent fields are forbidden: {', '.join(forbidden)}",
            forbidden[0],
        )
    unknown = sorted(set(record) - {item.name for item in fields(DrawingScene)})
    if unknown:
        return _invalid(
            IngestionErrorCode.MALFORMED_METADATA,
            f"unknown drawing fields: {', '.join(unknown)}",
            unknown[0],
        )
    try:
        raw_routes = record.get("routes", ())
        routes = tuple(
            item
            if isinstance(item, Route)
            else Route(
                int(item["edge_index"]),
                torch.as_tensor(item["points"]),
                str(item.get("kind", "polyline")),
            )
            for item in raw_routes
        )
        drawing = DrawingScene(
            positions=torch.as_tensor(record["positions"]),
            routes=routes,
            z_order=tuple(record.get("z_order", ())),
            node_label_offsets=(
                torch.as_tensor(record["node_label_offsets"])
                if record.get("node_label_offsets") is not None
                else None
            ),
            edge_label_positions=(
                torch.as_tensor(record["edge_label_positions"])
                if record.get("edge_label_positions") is not None
                else None
            ),
        )
    except (KeyError, TypeError, ValueError) as error:
        return _invalid(IngestionErrorCode.MALFORMED_METADATA, str(error), "record")
    return ingest(graph, drawing, style, profile)
