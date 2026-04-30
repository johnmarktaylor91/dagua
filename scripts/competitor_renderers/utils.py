"""Shared helpers for cosmetic competitor reference renderers."""

from __future__ import annotations

import json
import logging
import shutil
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import torch
from PIL import Image

LOGGER = logging.getLogger(__name__)
MAX_COMPARISON_SIDE_PX = 2000


def command_available(command: str) -> bool:
    """Return whether an executable can be found on ``PATH``.

    Parameters
    ----------
    command : str
        Executable name to locate.

    Returns
    -------
    bool
        ``True`` when ``command`` is available.
    """

    return shutil.which(command) is not None


def write_json(path: Path, payload: Mapping[str, object]) -> None:
    """Write a JSON payload with stable formatting.

    Parameters
    ----------
    path : pathlib.Path
        Destination path.
    payload : Mapping[str, object]
        JSON-serializable data to write.

    Returns
    -------
    None
        The file is written to disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _style_to_dict(style: object) -> Dict[str, object]:
    """Convert a style object into plain JSON-compatible fields.

    Parameters
    ----------
    style : object
        Dagua style dataclass or mapping.

    Returns
    -------
    dict[str, object]
        Shallow style mapping.
    """

    if isinstance(style, Mapping):
        return dict(style)
    if is_dataclass(style):
        return dict(asdict(style))
    return {}


def normalize_positions(
    positions: Sequence[Tuple[float, float]] | torch.Tensor,
    dimensions: Tuple[int, int],
    padding: float = 90.0,
) -> List[Tuple[float, float]]:
    """Map layout positions into image pixel coordinates.

    Parameters
    ----------
    positions : Sequence[tuple[float, float]] | torch.Tensor
        Node positions in Dagua layout coordinates with shape ``[N, 2]``.
    dimensions : tuple[int, int]
        Target image size as ``(width_px, height_px)``.
    padding : float, default=90.0
        Minimum pixel margin around the normalized graph.

    Returns
    -------
    list[tuple[float, float]]
        Pixel coordinates with y-axis flipped for browser/SVG renderers.
    """

    if isinstance(positions, torch.Tensor):
        raw_positions = [(float(x), float(y)) for x, y in positions.detach().cpu().tolist()]
    else:
        raw_positions = [(float(x), float(y)) for x, y in positions]
    if not raw_positions:
        return []

    width, height = dimensions
    xs = [point[0] for point in raw_positions]
    ys = [point[1] for point in raw_positions]
    min_x, max_x = min(xs), max(xs)
    min_y, max_y = min(ys), max(ys)
    span_x = max(max_x - min_x, 1.0)
    span_y = max(max_y - min_y, 1.0)
    usable_width = max(width - (2.0 * padding), 1.0)
    usable_height = max(height - (2.0 * padding), 1.0)
    scale = min(usable_width / span_x, usable_height / span_y)
    used_width = span_x * scale
    used_height = span_y * scale
    x_offset = (width - used_width) / 2.0
    y_offset = (height - used_height) / 2.0
    return [
        (x_offset + ((x - min_x) * scale), y_offset + ((max_y - y) * scale))
        for x, y in raw_positions
    ]


def ensure_png_dimensions(path: Path, dimensions: Tuple[int, int]) -> Path:
    """Resize or canvas-normalize a PNG to exact dimensions.

    Parameters
    ----------
    path : pathlib.Path
        PNG path to normalize in place.
    dimensions : tuple[int, int]
        Required output dimensions as ``(width_px, height_px)``.

    Returns
    -------
    pathlib.Path
        The same ``path`` after normalization.
    """

    width, height = dimensions
    longest = max(width, height)
    if longest > MAX_COMPARISON_SIDE_PX:
        raise ValueError(f"Comparison image exceeds {MAX_COMPARISON_SIDE_PX}px: {dimensions}")
    with Image.open(path) as opened:
        image = opened.convert("RGBA")
        if image.size == dimensions:
            image.save(path)
            return path
        image.thumbnail(dimensions, Image.LANCZOS)
        canvas = Image.new("RGBA", dimensions, "#FFFFFF")
        paste_x = (width - image.width) // 2
        paste_y = (height - image.height) // 2
        canvas.paste(image, (paste_x, paste_y), image)
        canvas.convert("RGB").save(path)
    return path


def graph_spec_from_dagua(graph: object) -> Dict[str, object]:
    """Build a unified graph spec from a ``DaguaGraph`` instance.

    Parameters
    ----------
    graph : object
        Dagua graph-like object exposing ``nodes``, ``edge_index``, and style
        lookup methods.

    Returns
    -------
    dict[str, object]
        Unified graph spec with ``nodes``, ``edges``, and ``clusters`` keys.
    """

    node_ids = list(getattr(graph, "_index_to_id", []))
    nodes: List[Dict[str, object]] = []
    for index, node in enumerate(list(getattr(graph, "nodes"))):
        style = _style_to_dict(graph.get_style_for_node(index))  # type: ignore[attr-defined]
        fallback_label = node_ids[index] if index < len(node_ids) else index
        nodes.append(
            {
                "id": node_ids[index] if index < len(node_ids) else str(index),
                "label": str(getattr(node, "label", fallback_label)),
                "style": style,
                "shape": style.get("shape", "ellipse"),
                "fill": style.get("fill", "#FFFFFF"),
                "stroke": style.get("stroke", "#000000"),
            }
        )

    edges: List[Dict[str, object]] = []
    edge_index = getattr(graph, "edge_index", None)
    if edge_index is not None:
        edge_tensor = (
            edge_index.detach().cpu() if isinstance(edge_index, torch.Tensor) else edge_index
        )
        for edge_num in range(int(edge_tensor.shape[1])):
            src_index = int(edge_tensor[0, edge_num])
            tgt_index = int(edge_tensor[1, edge_num])
            style = _style_to_dict(graph.get_style_for_edge(edge_num))  # type: ignore[attr-defined]
            edges.append(
                {
                    "src": node_ids[src_index] if src_index < len(node_ids) else str(src_index),
                    "tgt": node_ids[tgt_index] if tgt_index < len(node_ids) else str(tgt_index),
                    "label": "",
                    "style": style,
                }
            )

    return {"nodes": nodes, "edges": edges, "clusters": []}


def node_by_id(graph_spec: Mapping[str, object]) -> Dict[str, Mapping[str, object]]:
    """Return a lookup for unified node records.

    Parameters
    ----------
    graph_spec : Mapping[str, object]
        Unified graph specification.

    Returns
    -------
    dict[str, Mapping[str, object]]
        Node records keyed by node ID.
    """

    return {
        str(node.get("id")): node
        for node in graph_spec.get("nodes", [])
        if isinstance(node, Mapping)
    }


def unsupported_feature_warning(
    tool_name: str,
    feature_overrides: Optional[Mapping[str, object]],
) -> None:
    """Log that a tool cannot render requested feature overrides.

    Parameters
    ----------
    tool_name : str
        Competitor renderer name.
    feature_overrides : Mapping[str, object] | None
        Feature override payload requested by the harness.

    Returns
    -------
    None
        A warning is emitted through the module logger.
    """

    LOGGER.warning("%s cannot render feature overrides: %s", tool_name, feature_overrides)


def sanitize_id(value: object) -> str:
    """Return a renderer-safe identifier string.

    Parameters
    ----------
    value : object
        Input identifier.

    Returns
    -------
    str
        Identifier containing only simple ASCII characters.
    """

    cleaned = "".join(ch if ch.isalnum() else "_" for ch in str(value))
    return cleaned or "node"


def all_pngs_within_cap(paths: Iterable[Path], cap_px: int = MAX_COMPARISON_SIDE_PX) -> bool:
    """Return whether every PNG's longest side is within a cap.

    Parameters
    ----------
    paths : Iterable[pathlib.Path]
        PNG paths to inspect.
    cap_px : int, default=MAX_COMPARISON_SIDE_PX
        Longest-side cap in pixels.

    Returns
    -------
    bool
        ``True`` when all images satisfy the cap.
    """

    for path in paths:
        with Image.open(path) as image:
            if max(image.size) > cap_px:
                return False
    return True
