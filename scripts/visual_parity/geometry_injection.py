"""Graphviz geometry extraction for visual parity comparisons."""

from __future__ import annotations

import copy
import hashlib
import re
import shutil
import subprocess
from dataclasses import replace as dc_replace
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import torch

from dagua.edges import BezierCurve
from scripts.visual_parity.types import InjectedGeometry, SplineSegment

POINTS_PER_INCH = 72.0
SPLINE_SAMPLES_PER_SEGMENT = 24
_ATTR_BLOCK_RE = re.compile(
    r"(?P<id>[\w:.-]+)\s+(?:->\s+(?P<target>[\w:.-]+)\s+)?\[(?P<attrs>.*?)\];",
    re.S,
)
_SUBGRAPH_RE = re.compile(r"subgraph\s+(?P<id>cluster[\w:.-]*)\s*\{(?P<body>.*?)\}", re.S)


def dot_source_hash(dot_source: str) -> str:
    """Return the stable content hash used for Graphviz reference cache keys.

    Parameters
    ----------
    dot_source
        DOT source text.

    Returns
    -------
    str
        SHA-256 hex digest for ``dot_source``.
    """

    return hashlib.sha256(dot_source.encode("utf-8")).hexdigest()


def graphviz_tool_version(engine: str = "dot") -> str:
    """Return the installed Graphviz tool version string.

    Parameters
    ----------
    engine
        Graphviz executable name.

    Returns
    -------
    str
        Version output, or ``"unknown"`` when the executable cannot be run.
    """

    try:
        result = subprocess.run(
            [engine, "-V"],
            check=False,
            capture_output=True,
            text=True,
            timeout=10,
        )
    except (FileNotFoundError, subprocess.SubprocessError):
        return "unknown"
    output = (result.stderr or result.stdout).strip()
    return output or "unknown"


def _quote_attr_value(value: str) -> str:
    """Strip Graphviz quote wrappers from an attribute value.

    Parameters
    ----------
    value
        Raw attribute value.

    Returns
    -------
    str
        Unquoted attribute text.
    """

    cleaned = value.strip()
    if len(cleaned) >= 2 and cleaned[0] == cleaned[-1] == '"':
        cleaned = cleaned[1:-1]
    return cleaned.replace('\\"', '"')


def parse_dot_attrs(raw_attrs: str) -> Dict[str, str]:
    """Parse a Graphviz attribute list.

    Parameters
    ----------
    raw_attrs
        Text inside ``[...]`` from Graphviz ``-Tdot`` output.

    Returns
    -------
    dict[str, str]
        Attribute values keyed by name.
    """

    attrs: Dict[str, str] = {}
    token = []
    in_quote = False
    parts: List[str] = []
    for char in raw_attrs:
        if char == '"' and (not token or token[-1] != "\\"):
            in_quote = not in_quote
        if char == "," and not in_quote:
            parts.append("".join(token))
            token = []
        else:
            token.append(char)
    if token:
        parts.append("".join(token))
    for part in parts:
        if "=" not in part:
            continue
        key, value = part.split("=", 1)
        attrs[key.strip()] = _quote_attr_value(value)
    return attrs


def parse_point(raw: str) -> Tuple[float, float]:
    """Parse a Graphviz point token.

    Parameters
    ----------
    raw
        Point token such as ``"27,36.1"`` or ``"e,27,36.1"``.

    Returns
    -------
    tuple[float, float]
        Parsed ``(x, y)`` point.
    """

    parts = raw.split(",")
    if parts[0] in {"e", "s"}:
        parts = parts[1:]
    if len(parts) != 2:
        raise ValueError(f"invalid Graphviz point token: {raw!r}")
    return float(parts[0]), float(parts[1])


def parse_spline_pos(edge_id: str, pos: str) -> List[SplineSegment]:
    """Parse a Graphviz ``pos`` spline into cubic segments.

    Parameters
    ----------
    edge_id
        Stable edge identifier used in emitted segments.
    pos
        Graphviz ``pos`` attribute with optional endpoint/startpoint tokens.

    Returns
    -------
    list[ scripts.visual_parity.types.SplineSegment ]
        Cubic spline segments. The control-point grammar is ``3k+1`` points.
    """

    tokens = [token for token in pos.replace("\\\n", " ").split() if token]
    endpoint: Optional[Tuple[float, float]] = None
    startpoint: Optional[Tuple[float, float]] = None
    points: List[Tuple[float, float]] = []
    for token in tokens:
        if token.startswith("e,"):
            endpoint = parse_point(token)
        elif token.startswith("s,"):
            startpoint = parse_point(token)
        else:
            points.append(parse_point(token))
    if startpoint is not None:
        points.insert(0, startpoint)
    if (len(points) - 1) % 3 != 0:
        raise ValueError(f"{edge_id}: expected 3k+1 spline points, got {len(points)}")
    segments: List[SplineSegment] = []
    for offset in range(0, len(points) - 1, 3):
        segments.append(
            SplineSegment(
                edge_id=edge_id,
                segment_index=len(segments),
                start=points[offset],
                control_1=points[offset + 1],
                control_2=points[offset + 2],
                end=points[offset + 3],
                endpoint=endpoint,
            )
        )
    return segments


def _parse_bb(raw: str) -> Tuple[float, float, float, float]:
    """Parse a Graphviz bounding-box attribute.

    Parameters
    ----------
    raw
        Comma-separated ``x0,y0,x1,y1`` text.

    Returns
    -------
    tuple[float, float, float, float]
        Parsed bounding box.
    """

    values = [float(part) for part in raw.split(",")]
    if len(values) != 4:
        raise ValueError(f"invalid Graphviz bb: {raw!r}")
    return values[0], values[1], values[2], values[3]


def parse_graphviz_dot_geometry(
    xdot_dot: str,
    *,
    case_id: str,
    tool_version: str,
    source_hash: str,
    engine: str = "dot",
) -> InjectedGeometry:
    """Parse Graphviz ``-Tdot`` output into shared injected geometry.

    Parameters
    ----------
    xdot_dot
        DOT output emitted by Graphviz after layout.
    case_id
        Case id for provenance.
    tool_version
        Graphviz version string.
    source_hash
        Hash of the original DOT source.
    engine
        Graphviz engine used to produce the output.

    Returns
    -------
    scripts.visual_parity.types.InjectedGeometry
        Parsed geometry snapshot.
    """

    node_positions: Dict[str, Tuple[float, float]] = {}
    node_sizes: Dict[str, Tuple[float, float]] = {}
    edge_splines: Dict[str, List[SplineSegment]] = {}
    graph_attrs: Dict[str, Any] = {}
    cluster_rects: Dict[str, Tuple[float, float, float, float]] = {}

    graph_match = re.search(r"\bgraph\s+\[(?P<attrs>.*?)\];", xdot_dot, re.S)
    if graph_match:
        graph_attrs.update(parse_dot_attrs(graph_match.group("attrs")))
    if "bb" in graph_attrs:
        x0, y0, x1, y1 = _parse_bb(str(graph_attrs["bb"]))
        canvas_pt = (max(x1 - x0, 0.0), max(y1 - y0, 0.0))
    else:
        canvas_pt = (0.0, 0.0)

    for cluster_match in _SUBGRAPH_RE.finditer(xdot_dot):
        cluster_attrs_match = re.search(
            r"\bgraph\s+\[(?P<attrs>.*?)\];",
            cluster_match.group("body"),
            re.S,
        )
        if cluster_attrs_match:
            attrs = parse_dot_attrs(cluster_attrs_match.group("attrs"))
            if "bb" in attrs:
                cluster_rects[cluster_match.group("id")] = _parse_bb(attrs["bb"])

    for match in _ATTR_BLOCK_RE.finditer(xdot_dot):
        source_id = match.group("id")
        target_id = match.group("target")
        attrs = parse_dot_attrs(match.group("attrs"))
        if target_id is None:
            if "pos" in attrs:
                node_positions[source_id] = parse_point(attrs["pos"])
            if "width" in attrs and "height" in attrs:
                node_sizes[source_id] = (
                    float(attrs["width"]) * POINTS_PER_INCH,
                    float(attrs["height"]) * POINTS_PER_INCH,
                )
        elif "pos" in attrs:
            edge_id = f"{source_id}->{target_id}"
            edge_splines[edge_id] = parse_spline_pos(edge_id, attrs["pos"])

    return InjectedGeometry(
        case_id=case_id,
        tool="graphviz",
        tool_version=tool_version,
        dot_source_hash=source_hash,
        engine=engine,
        canvas_pt=canvas_pt,
        node_positions=node_positions,
        node_sizes=node_sizes,
        edge_splines=edge_splines,
        cluster_rects=cluster_rects,
        graph_attrs=graph_attrs,
    )


def _run_dot_command(
    engine: str,
    output_format: str,
    dot_path: Path,
    output_path: Path,
    *,
    dpi: Optional[int] = None,
) -> None:
    """Run one Graphviz output command.

    Parameters
    ----------
    engine
        Graphviz executable.
    output_format
        Output format passed as one ``-T`` flag.
    dot_path
        Input DOT path.
    output_path
        Destination path.
    dpi
        Optional raster DPI for PNG output.

    Returns
    -------
    None
        The output file is written.
    """

    command = [engine, f"-T{output_format}"]
    if dpi is not None:
        command.append(f"-Gdpi={dpi}")
    command.extend([str(dot_path), "-o", str(output_path)])
    result = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        timeout=60,
    )
    if result.returncode != 0:
        raise RuntimeError(f"{' '.join(command)} failed: {result.stderr.strip()}")


def graphviz_geometry(
    dot_source_or_graph: Union[str, Any],
    *,
    case_id: str,
    refcache_dir: Union[str, Path],
    engine: str = "dot",
    dpi: int = 200,
) -> InjectedGeometry:
    """Produce cached Graphviz geometry and references from one DOT source.

    Parameters
    ----------
    dot_source_or_graph
        DOT source text, or a graph accepted by ``graphviz_theme_comparison.graph_to_dot``.
    case_id
        Stable case id.
    refcache_dir
        Root cache directory for Graphviz reference artifacts.
    engine
        Graphviz executable to run.
    dpi
        PNG raster DPI.

    Returns
    -------
    scripts.visual_parity.types.InjectedGeometry
        Parsed geometry with cache provenance in ``graph_attrs``.
    """

    if shutil.which(engine) is None:
        raise RuntimeError(f"Graphviz executable not found on PATH: {engine}")
    if isinstance(dot_source_or_graph, str):
        dot_source = dot_source_or_graph
    else:
        import scripts.graphviz_theme_comparison as gthc

        dot_source = gthc.graph_to_dot(dot_source_or_graph)

    source_hash = dot_source_hash(dot_source)
    tool_version = graphviz_tool_version(engine)
    cache_key = f"{case_id}_{engine}_{source_hash[:16]}"
    cache_root = Path(refcache_dir) / "graphviz" / cache_key
    cache_root.mkdir(parents=True, exist_ok=True)
    dot_path = cache_root / "source.dot"
    svg_path = cache_root / "reference.svg"
    xdot_path = cache_root / "layout.dot"
    png_path = cache_root / "reference.png"
    dot_path.write_text(dot_source, encoding="utf-8")

    if not svg_path.exists():
        _run_dot_command(engine, "svg", dot_path, svg_path)
    if not xdot_path.exists():
        _run_dot_command(engine, "dot", dot_path, xdot_path)
    if not png_path.exists():
        _run_dot_command(engine, "png", dot_path, png_path, dpi=dpi)

    geometry = parse_graphviz_dot_geometry(
        xdot_path.read_text(encoding="utf-8"),
        case_id=case_id,
        tool_version=tool_version,
        source_hash=source_hash,
        engine=engine,
    )
    geometry.graph_attrs.update(
        {
            "cache_key": cache_key,
            "dot_path": str(dot_path),
            "svg_path": str(svg_path),
            "xdot_path": str(xdot_path),
            "png_path": str(png_path),
        }
    )
    return geometry


def _sample_cubic(segment: SplineSegment, samples: int) -> List[Tuple[float, float]]:
    """Sample one cubic spline segment.

    Parameters
    ----------
    segment
        Cubic segment to sample.
    samples
        Number of points to emit, including both endpoints.

    Returns
    -------
    list[tuple[float, float]]
        Sampled polyline points.
    """

    points: List[Tuple[float, float]] = []
    p0 = segment.start
    p1 = segment.control_1
    p2 = segment.control_2
    p3 = segment.end
    for index in range(samples):
        t = index / max(samples - 1, 1)
        u = 1.0 - t
        x = (u**3 * p0[0]) + (3.0 * u * u * t * p1[0]) + (3.0 * u * t * t * p2[0]) + (t**3 * p3[0])
        y = (u**3 * p0[1]) + (3.0 * u * u * t * p1[1]) + (3.0 * u * t * t * p2[1]) + (t**3 * p3[1])
        points.append((float(x), float(y)))
    return points


def to_bezier_curves(
    splines: Mapping[str, Sequence[SplineSegment]],
    *,
    samples_per_segment: int = SPLINE_SAMPLES_PER_SEGMENT,
) -> Dict[str, BezierCurve]:
    """Convert parsed Graphviz splines to Dagua waypoint curves.

    Parameters
    ----------
    splines
        Edge spline segments keyed by edge id.
    samples_per_segment
        Number of waypoint samples per cubic segment.

    Returns
    -------
    dict[str, dagua.edges.BezierCurve]
        Curves keyed by edge id, each with ``routing="graphviz_spline"``.
    """

    curves: Dict[str, BezierCurve] = {}
    for edge_id, segments in splines.items():
        waypoints: List[Tuple[float, float]] = []
        for segment in segments:
            samples = _sample_cubic(segment, samples_per_segment)
            if waypoints:
                samples = samples[1:]
            waypoints.extend(samples)
        if not waypoints:
            continue
        first = waypoints[0]
        last = waypoints[-1]
        cp1 = waypoints[1] if len(waypoints) > 1 else first
        cp2 = waypoints[-2] if len(waypoints) > 1 else last
        curves[edge_id] = BezierCurve(
            first,
            cp1,
            cp2,
            last,
            waypoints=tuple(waypoints),
            routing="graphviz_spline",
        )
    return curves


def node_positions_tensor(
    geometry: InjectedGeometry,
    node_ids: Sequence[str],
    *,
    flip_y: bool = False,
) -> torch.Tensor:
    """Build a position tensor from injected Graphviz node positions.

    Parameters
    ----------
    geometry
        Injected geometry snapshot.
    node_ids
        Node ids in tensor order.
    flip_y
        Whether to negate y coordinates for a y-down coordinate system.

    Returns
    -------
    torch.Tensor
        Float tensor with shape ``[N, 2]``.
    """

    values: List[Tuple[float, float]] = []
    for node_id in node_ids:
        x, y = geometry.node_positions[node_id]
        values.append((x, -y if flip_y else y))
    return torch.tensor(values, dtype=torch.float32)


def apply_graphviz_coordinate_dance(graph: Any) -> Any:
    """Apply the shared Graphviz y-up rendering convention to a graph clone.

    Parameters
    ----------
    graph
        Dagua graph whose edge styles should be interpreted in Graphviz y-up space.

    Returns
    -------
    Any
        Deep-copied graph with ``direction="BT"`` and head arrows moved to tails.
    """

    adjusted = copy.deepcopy(graph)
    adjusted.direction = "BT"
    for edge_index in range(int(adjusted.edge_index.shape[1])):
        style = adjusted.get_style_for_edge(edge_index)
        if style.arrow != "none" and style.tail_arrow == "none":
            adjusted.edge_styles[edge_index] = dc_replace(
                style,
                arrow="none",
                tail_arrow=style.arrow,
            )
    return adjusted


def edge_id_for_index(graph: Any, edge_index: int) -> str:
    """Return the Graphviz edge id for a Dagua edge index.

    Parameters
    ----------
    graph
        Dagua graph with ``edge_index``.
    edge_index
        Edge index to map.

    Returns
    -------
    str
        Edge id in the ``n{source}->n{target}`` form emitted by the comparison DOT writer.
    """

    source = int(graph.edge_index[0, edge_index].item())
    target = int(graph.edge_index[1, edge_index].item())
    return f"n{source}->n{target}"
