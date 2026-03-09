"""NodeStyle, EdgeStyle, ClusterStyle dataclasses and theme system."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional, Tuple


@dataclass
class NodeStyle:
    """Visual style for a node."""

    shape: str = "roundrect"  # rect, roundrect, ellipse, diamond, circle
    fill: str = "#E8F0FE"
    stroke: str = "#333333"
    stroke_width: float = 1.0
    stroke_dash: str = "solid"  # solid, dashed
    font_family: str = "monospace"
    font_size: float = 9.0
    font_color: str = "#333333"
    padding: Tuple[float, float] = (8.0, 4.0)  # horizontal, vertical
    corner_radius: float = 4.0
    opacity: float = 1.0


@dataclass
class EdgeStyle:
    """Visual style for an edge."""

    color: str = "#666666"
    width: float = 1.0
    arrow: str = "normal"  # normal, none
    style: str = "solid"  # solid, dashed, dotted
    opacity: float = 0.8


@dataclass
class ClusterStyle:
    """Visual style for a cluster box."""

    fill: str = "#F5F5F5"
    stroke: str = "#CCCCCC"
    stroke_width: float = 1.0
    stroke_dash: str = "solid"
    corner_radius: float = 8.0
    padding: float = 15.0
    label_position: str = "top"
    font_size: float = 10.0
    font_weight: str = "bold"
    font_color: str = "#555555"
    opacity: float = 0.3


DEFAULT_THEME: Dict[str, NodeStyle] = {
    "default": NodeStyle(),
    "input": NodeStyle(fill="#98FB98", shape="ellipse"),
    "output": NodeStyle(fill="#FF9999", shape="ellipse"),
    "buffer": NodeStyle(fill="#888888", shape="ellipse", font_color="#FFFFFF"),
    "bool": NodeStyle(fill="#F7D460", shape="ellipse"),
    "trainable_params": NodeStyle(fill="#D9D9D9", shape="roundrect"),
    "frozen_params": NodeStyle(fill="#B0B0B0", shape="roundrect"),
    "mixed_params": NodeStyle(fill="#D9D9D9", shape="roundrect"),
    "module": NodeStyle(fill="#E0E0FF", shape="rect"),
}

GRAPHVIZ_MATCH_THEME: Dict[str, NodeStyle] = {
    "default": NodeStyle(
        fill="#FFFFFF",
        stroke="#000000",
        shape="ellipse",
        font_family="serif",
        font_size=14.0,
        font_color="#000000",
    ),
    "input": NodeStyle(
        fill="#98FB98",
        stroke="#000000",
        shape="ellipse",
        font_family="serif",
        font_size=14.0,
        font_color="#000000",
    ),
    "output": NodeStyle(
        fill="#FF9999",
        stroke="#000000",
        shape="ellipse",
        font_family="serif",
        font_size=14.0,
        font_color="#000000",
    ),
}
