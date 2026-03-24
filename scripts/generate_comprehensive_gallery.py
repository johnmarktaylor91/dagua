#!/usr/bin/env python
# ruff: noqa: E402
"""Generate multi-panel cosmetic sweep galleries for Dagua styles.

This script renders one image per cosmetic option, with each image showing the
entire value range for that option. When Graphviz has a practical analogue for
the option, the gallery includes a second comparison row.
"""

from __future__ import annotations

import argparse
import json
import math
import shutil
import sys
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import numpy as np
import torch
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

from dagua import DaguaGraph
from dagua.styles import ClusterStyle, EdgeStyle, GraphStyle, NodeStyle
from scripts.generate_cosmetic_album import (
    EDGE_COLOR,
    NODE_FILL,
    NODE_STROKE,
    PANEL_SIZE,
    RAW_RENDER_DPI,
    WHITE,
    AlbumCase,
    CosmeticAlbumResult,
    GraphvizRenderSpec,
    _apply_graph_style,
    _base_cluster_style,
    _base_edge_style,
    _base_node_style,
    _build_graphviz_dot,
    _pair_graph,
    _render_dagua_png,
    _render_graphviz_png,
    _set_all_edge_styles,
    _set_all_node_styles,
    _single_node_graph,
)

DEFAULT_OUTPUT_DIR = "eval_output/comprehensive_gallery"
GRAPHVIZ_LABEL = "Graphviz"
# Keep six-value sweeps on a single row so they do not leave a lone orphan cell.
MAX_COLUMNS = 6
CONTENT_CROP_PADDING = 8
PANEL_MARGIN = 18
SWEEP_PANEL_SIZE: Tuple[int, int] = (PANEL_SIZE[0] // 4 - 12, PANEL_SIZE[1] // 4 - 12)
BASE_CLUSTER_FILL = "#EAF1F8"
BASE_CLUSTER_STROKE = "#A9B8C7"
LIGHT_NODE_FILL = "#DCE7F2"
LIGHT_NODE_STROKE = "#A6B7C8"
LIGHT_EDGE_COLOR = "#D7E3EE"
DARK_FONT_COLOR = "#F7F8FA"
GRADIENT_FILL_COLOR = "#4C7AE6"
GRADIENT_COLOR = "#FF9A4A"
EDGE_GRADIENT_START = "#0057FF"
EDGE_GRADIENT_END = "#FF6A00"
TEXT_OUTLINE_COLOR = "#0072B2"
SHADOW_COLOR_SOFT = "#00000060"
SHADOW_COLOR_LIGHT = "#00000040"
DARK_NODE_FILL = "#24303B"
DARK_NODE_STROKE = "#8FA1B3"
DARK_EDGE_COLOR = "#9FB0C0"
DARK_CLUSTER_FILL = "#141C24"
DARK_CLUSTER_STROKE = "#5E7285"
DARK_LABEL_BACKGROUND = "#111820"
FILL_PATTERN_PALETTE = ["#56B4E9", "#0072B2", "#009E73", "#D55E00", "#CC79A7", "#F0E442"]
DEFAULT_PADDING = (11.0, 9.0)
NON_INSETTABLE_SHAPES = {
    "double_circle",
    "cloud",
    "stadium",
    "tab",
    "note",
    "document",
    "box3d",
}
SMALL_INTERIOR_NODE_SHAPES = {"star", "note", "box3d", "tab", "document"}
ARROW_DEMO_SWEEPS = {
    "edge_arrow_types",
    "edge_arrow_fill",
    "edge_arrow_length",
    "edge_arrow_width",
}
COMMON_ARROW_TYPES: Tuple[str, ...] = (
    "normal",
    "vee",
    "dot",
    "diamond",
    "tee",
    "crow",
    "circle",
    "open",
)
SHAPE_SWEEP_MIN_WIDTH = 80.0
SHAPE_SWEEP_MIN_HEIGHT = 60.0
TEXT_ALIGN_SWEEP_MIN_WIDTH = 120.0
TEXT_ALIGN_SWEEP_MIN_HEIGHT = 90.0
CROSSING_SWEEP_EDGE_WIDTH = 3.0
CROSSING_STYLE_DEMO_SIZE = 12.0
DIRECTION_SWEEP_NODE_WIDTH = 72.0
DIRECTION_SWEEP_NODE_HEIGHT = 42.0
NODE_STROKE_SWEEP_MIN_WIDTH = 124.0
NODE_STROKE_SWEEP_MIN_HEIGHT = 78.0
TEXT_WRAP_SWEEP_MIN_WIDTH = 128.0
TEXT_WRAP_SWEEP_MIN_HEIGHT = 78.0
EDGE_ARROW_DEMO_GAP = 180.0
EDGE_COLOR_GRADIENT_GAP = 240.0
EDGE_HEAD_TAIL_LABEL_GAP = 184.0
GRAPH_DIRECTION_HORIZONTAL_MARGIN = 32.0
WIDE_DIRECTION_PANEL_SIZE: Tuple[int, int] = (SWEEP_PANEL_SIZE[0] + 80, SWEEP_PANEL_SIZE[1])
PORT_STYLE_PANEL_SIZE: Tuple[int, int] = (SWEEP_PANEL_SIZE[0] + 18, SWEEP_PANEL_SIZE[1] + 12)
EXTERNAL_LABEL_PANEL_SIZE: Tuple[int, int] = (SWEEP_PANEL_SIZE[0] + 44, SWEEP_PANEL_SIZE[1])
OVERFLOW_POLICY_PANEL_SIZE: Tuple[int, int] = (SWEEP_PANEL_SIZE[0] + 56, SWEEP_PANEL_SIZE[1])
SWEEP_COLUMN_OVERRIDES: Dict[str, int] = {"edge_arrow_types": 4}
GRAPH_MARGIN_OUTLINE_COLOR = "#8FA3B8"
GRAPH_MARGIN_OUTLINE_STYLE = (0, (4, 3))
GRAPH_MARGIN_OUTLINE_BASE_INSET = 0.03
GRAPH_MARGIN_OUTLINE_MAX_INSET = 0.22
SHAPE_NODE_LABELS: Dict[str, str] = {
    "parallelogram": "parallel",
    "double_circle": "dbl_circle",
    "box3d": "box3d",
}


@dataclass(frozen=True)
class SweepConfig:
    """Definition of a single cosmetic sweep.

    Parameters
    ----------
    name : str
        Stable sweep identifier.
    category : str
        Output category path relative to the gallery root.
    description : str
        Human-readable summary for the sweep figure title.
    target : str
        Style target: ``"node"``, ``"edge"``, ``"cluster"``, or ``"graph"``.
    field : str
        Target field name on the style dataclass.
    values : list[Any]
        Ordered sweep values.
    labels : list[str]
        Human-readable labels aligned with ``values``.
    graph_builder : str
        Fixed graph-builder identifier.
    gv_map : dict[str, dict[str, str] | None] | None
        Graphviz attribute mapping for comparable values.
    """

    name: str
    category: str
    description: str
    target: str
    field: str
    values: List[Any]
    labels: List[str]
    graph_builder: str
    gv_map: Optional[Dict[str, Optional[Dict[str, str]]]]


SWEEPS: List[SweepConfig] = [
    SweepConfig(
        name="node_shape",
        category="nodes/shapes",
        description="All supported node shapes",
        target="node",
        field="shape",
        values=[
            "rect",
            "roundrect",
            "ellipse",
            "diamond",
            "circle",
            "triangle",
            "hexagon",
            "parallelogram",
            "pentagon",
            "octagon",
            "star",
            "cylinder",
            "trapezoid",
            "double_circle",
            "tab",
            "note",
            "box3d",
        ],
        labels=[
            "rect",
            "roundrect",
            "ellipse",
            "diamond",
            "circle",
            "triangle",
            "hexagon",
            "parallel.",
            "pentagon",
            "octagon",
            "star",
            "cylinder",
            "trapezoid",
            "dbl_circle",
            "tab",
            "note",
            "box3d",
        ],
        graph_builder="single_node",
        gv_map={
            "rect": {"shape": "box"},
            "roundrect": {"shape": "box", "style": "filled,rounded"},
            "ellipse": {"shape": "ellipse"},
            "diamond": {"shape": "diamond"},
            "circle": {"shape": "circle"},
            "triangle": {"shape": "triangle"},
            "hexagon": {"shape": "hexagon"},
            "parallelogram": {"shape": "parallelogram"},
            "pentagon": {"shape": "pentagon"},
            "octagon": {"shape": "octagon"},
            "star": {"shape": "star"},
            "cylinder": {"shape": "cylinder"},
            "trapezoid": {"shape": "trapezium"},
            "double_circle": {"shape": "doublecircle"},
            "tab": {"shape": "tab"},
            "note": {"shape": "note"},
            "box3d": {"shape": "box3d"},
        },
    ),
    SweepConfig(
        name="node_gradient",
        category="nodes/fills",
        description="Node gradient modes",
        target="node",
        field="gradient",
        values=["none", "linear", "radial"],
        labels=["None", "Linear", "Radial"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_gradient_angle",
        category="nodes/fills",
        description="Linear gradient angle variations",
        target="node",
        field="gradient_angle",
        values=[0.0, 45.0, 90.0, 135.0, 180.0, 270.0],
        labels=["0 deg", "45 deg", "90 deg", "135 deg", "180 deg", "270 deg"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_fill_pattern",
        category="nodes/fills",
        description="Node fill pattern types",
        target="node",
        field="fill_pattern",
        values=["solid", "striped", "hatched", "pie"],
        labels=["Solid", "Striped", "Hatched", "Pie"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_pie_chart",
        category="nodes/fills",
        description="Pie chart fill with varying slice counts",
        target="node",
        field="fill_pattern_values",
        values=[
            [1.0],
            [1.0, 1.0],
            [3.0, 2.0, 1.0],
            [1.0, 2.0, 3.0, 4.0],
            [1.0, 1.0, 1.0, 1.0, 1.0, 1.0],
        ],
        labels=["1 slice", "2 slices", "3 slices", "4 slices", "6 slices"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_donut",
        category="nodes/fills",
        description="Donut hole size variations",
        target="node",
        field="fill_pattern_hole",
        values=[0.0, 0.2, 0.4, 0.6, 0.8],
        labels=["0 (pie)", "0.2", "0.4", "0.6", "0.8"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_opacity",
        category="nodes/fills",
        description="Node opacity range",
        target="node",
        field="opacity",
        values=[0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0.2", "0.4", "0.6", "0.8", "1.0"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_stroke_width",
        category="nodes/borders",
        description="Border thickness range",
        target="node",
        field="stroke_width",
        values=[0.5, 1.5, 3.0, 5.0],
        labels=["0.5pt", "1.5pt", "3.0pt", "5.0pt"],
        graph_builder="single_node",
        gv_map={
            "0.5": {"penwidth": "1.0"},
            "1.5": {"penwidth": "3.0"},
            "3.0": {"penwidth": "6.0"},
            "5.0": {"penwidth": "10.0"},
        },
    ),
    SweepConfig(
        name="node_stroke_dash",
        category="nodes/borders",
        description="Border dash styles",
        target="node",
        field="stroke_dash",
        values=["solid", "dashed", "dotted"],
        labels=["Solid", "Dashed", "Dotted"],
        graph_builder="single_node",
        gv_map={
            "solid": {"style": "filled"},
            "dashed": {"style": "filled,dashed"},
            "dotted": {"style": "filled,dotted"},
        },
    ),
    SweepConfig(
        name="node_border_opacity",
        category="nodes/borders",
        description="Border opacity range",
        target="node",
        field="border_opacity",
        values=[0.0, 0.2, 0.5, 0.8, 1.0],
        labels=["0.0", "0.2", "0.5", "0.8", "1.0"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_corner_radius",
        category="nodes/borders",
        description="Corner radius range",
        target="node",
        field="corner_radius",
        values=[0.0, 4.0, 8.0, 12.0, 20.0],
        labels=["0 (sharp)", "4", "8", "12", "20 (pill)"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_border_count",
        category="nodes/borders",
        description="Single vs double border",
        target="node",
        field="border_count",
        values=[1, 2],
        labels=["Single", "Double"],
        graph_builder="single_node",
        gv_map={"1": {"peripheries": "1"}, "2": {"peripheries": "2"}},
    ),
    SweepConfig(
        name="node_border_position",
        category="nodes/borders",
        description="Border position relative to node edge",
        target="node",
        field="border_position",
        values=["center", "inside", "outside"],
        labels=["Center", "Inside", "Outside"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_stroke_cap",
        category="nodes/borders",
        description="Stroke cap styles",
        target="node",
        field="stroke_cap",
        values=["butt", "round", "square"],
        labels=["Butt", "Round", "Square"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_stroke_join",
        category="nodes/borders",
        description="Stroke join styles",
        target="node",
        field="stroke_join",
        values=["miter", "bevel", "round"],
        labels=["Miter", "Bevel", "Round"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_font_size",
        category="nodes/text",
        description="Font size range",
        target="node",
        field="font_size",
        values=[5.0, 7.0, 9.0, 12.0, 16.0, 24.0],
        labels=["5pt", "7pt", "9pt (default)", "12pt", "16pt", "24pt"],
        graph_builder="single_node",
        gv_map={
            "5.0": {"fontsize": "8"},
            "7.0": {"fontsize": "11"},
            "9.0": {"fontsize": "14"},
            "12.0": {"fontsize": "18"},
            "16.0": {"fontsize": "24"},
            "24.0": {"fontsize": "36"},
        },
    ),
    SweepConfig(
        name="node_font_weight",
        category="nodes/text",
        description="Font weight options",
        target="node",
        field="font_weight",
        values=["regular", "bold"],
        labels=["Regular", "Bold"],
        graph_builder="single_node",
        gv_map={
            "regular": {"fontname": "Helvetica"},
            "bold": {"fontname": "Helvetica Bold"},
        },
    ),
    SweepConfig(
        name="node_font_style",
        category="nodes/text",
        description="Font style options",
        target="node",
        field="font_style",
        values=["normal", "italic"],
        labels=["Normal", "Italic"],
        graph_builder="single_node",
        gv_map={
            "normal": {"fontname": "Helvetica"},
            "italic": {"fontname": "Helvetica Oblique"},
        },
    ),
    SweepConfig(
        name="node_text_align",
        category="nodes/text",
        description="Horizontal text alignment",
        target="node",
        field="text_align",
        values=["left", "center", "right"],
        labels=["Left", "Center", "Right"],
        graph_builder="wide_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_text_valign",
        category="nodes/text",
        description="Vertical text alignment",
        target="node",
        field="text_valign",
        values=["top", "center", "bottom"],
        labels=["Top", "Center", "Bottom"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_text_rotation",
        category="nodes/text",
        description="Text rotation angles",
        target="node",
        field="text_rotation",
        values=[0.0, 45.0, 90.0, 180.0],
        labels=["0 deg", "45 deg", "90 deg", "180 deg"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_text_wrap",
        category="nodes/text",
        description="Text wrapping modes",
        target="node",
        field="text_wrap",
        values=["none", "wrap", "ellipsis"],
        labels=["None", "Wrap", "Ellipsis"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_text_transform",
        category="nodes/text",
        description="Text transform options",
        target="node",
        field="text_transform",
        values=["none", "uppercase", "lowercase"],
        labels=["None", "UPPERCASE", "lowercase"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_text_outline",
        category="nodes/text",
        description="Text outline on or off",
        target="node",
        field="text_outline",
        values=[False, True],
        labels=["Off", "On"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_text_background",
        category="nodes/text",
        description="Text background variations",
        target="node",
        field="text_background",
        values=["", "#FFE0B2", "#C8E6C9"],
        labels=["None", "Orange bg", "Green bg"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_external_label",
        category="nodes/text",
        description="External label positions",
        target="node",
        field="external_label_position",
        values=["top", "bottom", "left", "right"],
        labels=["Top", "Bottom", "Left", "Right"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_shadow",
        category="nodes/effects",
        description="Shadow on or off",
        target="node",
        field="shadow",
        values=[False, True],
        labels=["Off", "On"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_shadow_blur",
        category="nodes/effects",
        description="Shadow blur radius",
        target="node",
        field="shadow_blur",
        values=[0.0, 2.0, 5.0],
        labels=["0 (sharp)", "2", "5"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_padding",
        category="nodes/effects",
        description="Node padding range",
        target="node",
        field="padding",
        values=[
            (5.0, 3.0),
            (8.0, 6.0),
            (11.0, 9.0),
            (18.0, 14.0),
            (25.0, 20.0),
        ],
        labels=["Tight", "Compact", "Default", "Spacious", "Generous"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="node_overflow_policy",
        category="nodes/effects",
        description="Text overflow handling",
        target="node",
        field="overflow_policy",
        values=["shrink_text", "expand_node", "overflow"],
        labels=["Shrink text", "Expand node", "Overflow"],
        graph_builder="single_node",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_arrow_types",
        category="edges/arrows",
        description="Common arrowhead types",
        target="edge",
        field="arrow",
        values=list(COMMON_ARROW_TYPES),
        labels=list(COMMON_ARROW_TYPES),
        graph_builder="pair",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_arrow_fill",
        category="edges/arrows",
        description="Filled vs hollow arrows",
        target="edge",
        field="arrow_fill",
        values=["filled", "hollow"],
        labels=["Filled", "Hollow"],
        graph_builder="pair",
        gv_map={"filled": {"arrowhead": "normal"}, "hollow": {"arrowhead": "empty"}},
    ),
    SweepConfig(
        name="edge_arrow_length",
        category="edges/arrows",
        description="Arrow length range",
        target="edge",
        field="arrow_length",
        values=[5.0, 12.0, 20.0, 30.0, 45.0],
        labels=["5pt", "12pt", "20pt", "30pt", "45pt"],
        graph_builder="pair",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_arrow_width",
        category="edges/arrows",
        description="Arrow width range",
        target="edge",
        field="arrow_width",
        values=[3.0, 7.0, 14.0, 25.0],
        labels=["3pt", "7pt", "14pt", "25pt"],
        graph_builder="pair",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_width",
        category="edges/styles",
        description="Edge line width range",
        target="edge",
        field="width",
        values=[0.5, 1.0, 1.5, 2.5, 4.0, 6.0],
        labels=["0.5pt", "1.0pt", "1.5pt", "2.5pt", "4.0pt", "6.0pt"],
        graph_builder="pair",
        gv_map={
            "0.5": {"penwidth": "0.5"},
            "1.0": {"penwidth": "1.0"},
            "1.5": {"penwidth": "1.5"},
            "2.5": {"penwidth": "2.5"},
            "4.0": {"penwidth": "4.0"},
            "6.0": {"penwidth": "6.0"},
        },
    ),
    SweepConfig(
        name="edge_line_style",
        category="edges/styles",
        description="Edge line dash styles",
        target="edge",
        field="style",
        values=["solid", "dashed", "dotted"],
        labels=["Solid", "Dashed", "Dotted"],
        graph_builder="pair",
        gv_map={"solid": {}, "dashed": {"style": "dashed"}, "dotted": {"style": "dotted"}},
    ),
    SweepConfig(
        name="edge_opacity",
        category="edges/styles",
        description="Edge opacity range",
        target="edge",
        field="opacity",
        values=[0.2, 0.4, 0.65, 0.8, 1.0],
        labels=["0.2", "0.4", "0.65 (default)", "0.8", "1.0"],
        graph_builder="pair",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_curvature",
        category="edges/styles",
        description="Bezier curvature control",
        target="edge",
        field="curvature",
        values=[0.0, 0.2, 0.4, 0.6, 0.8, 1.0],
        labels=["0", "0.2", "0.4", "0.6", "0.8", "1.0"],
        graph_builder="pair",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_routing",
        category="edges/routing",
        description="Edge routing modes",
        target="edge",
        field="routing",
        values=["bezier", "straight", "ortho", "taxi"],
        labels=["Bezier", "Straight", "Ortho", "Taxi"],
        graph_builder="pair",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_taper",
        category="edges/advanced",
        description="Edge taper end-width ratios",
        target="edge",
        field="taper_width_end",
        values=[3.0, 2.0, 1.0, 0.5, 0.1],
        labels=["Off (3pt->3pt)", "3pt->2pt", "3pt->1pt", "3pt->0.5pt", "3pt->0.1pt"],
        graph_builder="pair",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_color_gradient",
        category="edges/advanced",
        description="Edge color gradient modes",
        target="edge",
        field="color_gradient",
        values=["none", "source_to_target", "source_to_target_wide"],
        labels=["None", "Source to Target", "Source to Target (4pt)"],
        graph_builder="pair",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_crossing_style",
        category="edges/advanced",
        description="Edge crossing jump styles",
        target="edge",
        field="crossing_style",
        values=["none", "arc", "gap", "sharp"],
        labels=["None", "Arc", "Gap", "Sharp"],
        graph_builder="crossing4",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_crossing_size",
        category="edges/advanced",
        description="Crossing jump size range",
        target="edge",
        field="crossing_size",
        values=[4.0, 8.0, 12.0, 18.0],
        labels=["4pt", "8pt", "12pt", "18pt"],
        graph_builder="crossing4",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_label_position",
        category="edges/labels",
        description="Label position along edge",
        target="edge",
        field="label_position",
        values=[0.2, 0.35, 0.5, 0.65, 0.8],
        labels=["0.2", "0.35", "0.5 (mid)", "0.65", "0.8"],
        graph_builder="pair",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_head_tail_labels",
        category="edges/labels",
        description="Head and tail endpoint labels",
        target="edge",
        field="head_label",
        values=["none", "head_tail", "in_out", "src_dst"],
        labels=["None", "Head/Tail", "In/Out", "Src/Dst"],
        graph_builder="pair",
        gv_map=None,
    ),
    SweepConfig(
        name="edge_port_style",
        category="edges/labels",
        description="Port distribution styles",
        target="edge",
        field="port_style",
        values=["distributed", "center"],
        labels=["Distributed", "Center"],
        graph_builder="fan6",
        gv_map=None,
    ),
    SweepConfig(
        name="cluster_stroke_dash",
        category="clusters",
        description="Cluster border dash styles",
        target="cluster",
        field="stroke_dash",
        values=["solid", "dashed", "dotted"],
        labels=["Solid", "Dashed", "Dotted"],
        graph_builder="cluster",
        gv_map={
            "solid": {"style": "filled"},
            "dashed": {"style": "filled,dashed"},
            "dotted": {"style": "filled,dotted"},
        },
    ),
    SweepConfig(
        name="cluster_corner_radius",
        category="clusters",
        description="Cluster corner radius range",
        target="cluster",
        field="corner_radius",
        values=[0.0, 4.0, 8.0, 15.0, 25.0],
        labels=["0 (sharp)", "4", "8 (default)", "15", "25"],
        graph_builder="cluster",
        gv_map=None,
    ),
    SweepConfig(
        name="cluster_padding",
        category="clusters",
        description="Cluster padding range",
        target="cluster",
        field="padding",
        values=[20.0, 35.0, 50.0, 65.0, 80.0],
        labels=["20pt", "35pt", "50pt", "65pt", "80pt"],
        graph_builder="cluster",
        gv_map=None,
    ),
    SweepConfig(
        name="cluster_label_position",
        category="clusters",
        description="Cluster label positions",
        target="cluster",
        field="label_position",
        values=["top-left", "top-center", "top-right"],
        labels=["Top-Left", "Top-Center", "Top-Right"],
        graph_builder="cluster",
        gv_map=None,
    ),
    SweepConfig(
        name="cluster_opacity",
        category="clusters",
        description="Cluster opacity range",
        target="cluster",
        field="opacity",
        values=[0.1, 0.3, 0.5, 0.7, 1.0],
        labels=["0.1", "0.3", "0.5", "0.7", "1.0"],
        graph_builder="cluster",
        gv_map=None,
    ),
    SweepConfig(
        name="cluster_depth_coloring",
        category="clusters",
        description="Nested cluster depth fill step",
        target="cluster",
        field="depth_fill_step",
        values=[0.0, 0.05, 0.1, 0.2],
        labels=["0", "0.05", "0.1", "0.2"],
        graph_builder="nested_cluster",
        gv_map=None,
    ),
    SweepConfig(
        name="graph_background",
        category="graph",
        description="Background color variations",
        target="graph",
        field="background_color",
        values=["#FFFFFF", "#FAFAFA", "#F5F5F0", "#1A1E24", "#0F0F10"],
        labels=["White", "Warm white", "Paper", "Dark", "Near black"],
        graph_builder="pair",
        gv_map=None,
    ),
    SweepConfig(
        name="graph_direction",
        category="graph",
        description="Graph layout directions",
        target="graph",
        field="direction",
        values=["TB", "BT", "LR", "RL"],
        labels=["Top-Bottom", "Bottom-Top", "Left-Right", "Right-Left"],
        graph_builder="chain3",
        gv_map={
            "TB": {"rankdir": "TB"},
            "BT": {"rankdir": "BT"},
            "LR": {"rankdir": "LR"},
            "RL": {"rankdir": "RL"},
        },
    ),
    SweepConfig(
        name="graph_margin",
        category="graph",
        description="Graph margin range",
        target="graph",
        field="margin",
        values=[0.0, 10.0, 30.0, 60.0],
        labels=["0", "10", "30", "60"],
        graph_builder="pair",
        gv_map=None,
    ),
]


def build_sweep_catalog() -> List[SweepConfig]:
    """Return the full sweep catalog after basic validation.

    Returns
    -------
    list[SweepConfig]
        Sweep catalog.
    """

    for sweep in SWEEPS:
        if len(sweep.values) != len(sweep.labels):
            raise ValueError(f"Sweep {sweep.name!r} has mismatched values and labels.")
    return list(SWEEPS)


def _graphviz_available() -> bool:
    """Return whether Graphviz's ``dot`` executable is available.

    Returns
    -------
    bool
        ``True`` when ``dot`` is on ``PATH``.
    """

    return shutil.which("dot") is not None


def _value_key(value: Any) -> str:
    """Convert a sweep value to a manifest and lookup key.

    Parameters
    ----------
    value : Any
        Sweep value.

    Returns
    -------
    str
        Stable string form for mapping lookups.
    """

    return str(value)


def _is_dark_background(color: str) -> bool:
    """Return whether a hex background color is perceptually dark.

    Parameters
    ----------
    color : str
        Hex color such as ``"#1A1E24"``.

    Returns
    -------
    bool
        ``True`` for dark colors.
    """

    hex_value = color.lstrip("#")
    if len(hex_value) != 6:
        return False
    red = int(hex_value[0:2], 16)
    green = int(hex_value[2:4], 16)
    blue = int(hex_value[4:6], 16)
    luminance = (0.2126 * red) + (0.7152 * green) + (0.0722 * blue)
    return luminance < 128.0


def _pattern_colors(count: int) -> List[str]:
    """Return a stable palette for fill-pattern demos.

    Parameters
    ----------
    count : int
        Number of colors needed.

    Returns
    -------
    list[str]
        Palette slice repeated as needed.
    """

    repeats = math.ceil(count / len(FILL_PATTERN_PALETTE))
    palette = FILL_PATTERN_PALETTE * repeats
    return palette[:count]


def _crop_to_content(image: Image.Image) -> Image.Image:
    """Crop a rendered image down to non-white content.

    Parameters
    ----------
    image : PIL.Image.Image
        Source image.

    Returns
    -------
    PIL.Image.Image
        Cropped image, or the original image when no content is found.
    """

    rgba = image.convert("RGBA")
    data = np.asarray(rgba)
    visible_mask = (data[:, :, 3] > 0) & np.any(data[:, :, :3] < 250, axis=2)
    if not bool(visible_mask.any()):
        return rgba
    ys, xs = np.nonzero(visible_mask)
    left = max(int(xs.min()) - CONTENT_CROP_PADDING, 0)
    top = max(int(ys.min()) - CONTENT_CROP_PADDING, 0)
    right = min(int(xs.max()) + CONTENT_CROP_PADDING + 1, image.width)
    bottom = min(int(ys.max()) + CONTENT_CROP_PADDING + 1, image.height)
    return rgba.crop((left, top, right, bottom))


def _normalize_panel_image(
    image_path: Path,
    panel_size: Tuple[int, int] = SWEEP_PANEL_SIZE,
    crop_content: bool = True,
) -> np.ndarray:
    """Resize and center a render onto a fixed white panel.

    Parameters
    ----------
    image_path : Path
        Source PNG path.
    panel_size : tuple[int, int], default=SWEEP_PANEL_SIZE
        Target panel width and height in pixels.
    crop_content : bool, default=True
        Whether to crop away empty whitespace before resizing.

    Returns
    -------
    numpy.ndarray
        RGB image array suitable for ``imshow``.
    """

    with Image.open(image_path) as image:
        normalized = _crop_to_content(image) if crop_content else image.convert("RGBA")
        normalized.thumbnail(
            (
                panel_size[0] - PANEL_MARGIN,
                panel_size[1] - PANEL_MARGIN,
            ),
            Image.LANCZOS,
        )
        canvas = Image.new("RGBA", panel_size, WHITE)
        offset = (
            (panel_size[0] - normalized.width) // 2,
            (panel_size[1] - normalized.height) // 2,
        )
        canvas.paste(normalized, offset, normalized)
    return np.asarray(canvas.convert("RGB"))


def _panel_size_for_sweep(sweep: SweepConfig) -> Tuple[int, int]:
    """Return the normalized panel size for a sweep.

    Parameters
    ----------
    sweep : SweepConfig
        Sweep definition.

    Returns
    -------
    tuple[int, int]
        Panel width and height in pixels.
    """

    if sweep.name == "graph_direction":
        return WIDE_DIRECTION_PANEL_SIZE
    if sweep.name == "edge_port_style":
        return PORT_STYLE_PANEL_SIZE
    if sweep.name == "node_external_label":
        return EXTERNAL_LABEL_PANEL_SIZE
    if sweep.name == "node_overflow_policy":
        return OVERFLOW_POLICY_PANEL_SIZE
    return SWEEP_PANEL_SIZE


def _column_count_for_sweep(sweep: SweepConfig) -> int:
    """Return the number of columns used when laying out a sweep.

    Parameters
    ----------
    sweep : SweepConfig
        Sweep definition.

    Returns
    -------
    int
        Number of columns to render.
    """

    preferred = SWEEP_COLUMN_OVERRIDES.get(sweep.name, MAX_COLUMNS)
    return min(preferred, len(sweep.values))


def _figure_height_for_sweep(sweep: SweepConfig, total_rows: int) -> float:
    """Return the figure height for a rendered sweep.

    Parameters
    ----------
    sweep : SweepConfig
        Sweep definition.
    total_rows : int
        Total subplot rows after Graphviz comparisons are expanded.

    Returns
    -------
    float
        Figure height in inches.
    """

    if sweep.name == "edge_port_style":
        return max(2.9, 2.05 * total_rows + 0.85)
    return max(3.0, 2.35 * total_rows + 0.55)


def _tight_layout_rect_for_sweep(sweep: SweepConfig) -> Tuple[float, float, float, float]:
    """Return the ``tight_layout`` bounds for a sweep figure.

    Parameters
    ----------
    sweep : SweepConfig
        Sweep definition.

    Returns
    -------
    tuple[float, float, float, float]
        ``matplotlib`` rect tuple.
    """

    if sweep.name == "edge_port_style":
        return (0.01, 0.01, 1.0, 0.94)
    return (0.01, 0.01, 1.0, 0.972)


def _draw_graph_margin_outline(axis: plt.Axes, margin: float) -> None:
    """Overlay a guide box that visualizes the graph margin inset.

    Parameters
    ----------
    axis : matplotlib.axes.Axes
        Target panel axis.
    margin : float
        Graph margin value in points.

    Returns
    -------
    None
        The axis is updated in place.
    """

    max_margin = max(float(v) for v in (0.0, 10.0, 30.0, 60.0))
    inset = GRAPH_MARGIN_OUTLINE_BASE_INSET
    if max_margin > 0.0:
        inset += (float(margin) / max_margin) * GRAPH_MARGIN_OUTLINE_MAX_INSET
    outline = Rectangle(
        (inset, inset),
        1.0 - (2.0 * inset),
        1.0 - (2.0 * inset),
        transform=axis.transAxes,
        fill=False,
        linewidth=1.6,
        edgecolor=GRAPH_MARGIN_OUTLINE_COLOR,
        linestyle=GRAPH_MARGIN_OUTLINE_STYLE,
    )
    axis.add_patch(outline)


def _shape_node_label(shape_name: str) -> str:
    """Return the in-panel label for a node-shape demo.

    Parameters
    ----------
    shape_name : str
        Stable shape identifier.

    Returns
    -------
    str
        Shape label tuned for legibility inside the node.
    """

    return SHAPE_NODE_LABELS.get(shape_name, shape_name)


def _port_style_annotation(port_style: str) -> str:
    """Return the explanatory overlay for a port-style demo panel.

    Parameters
    ----------
    port_style : str
        Port style identifier.

    Returns
    -------
    str
        Short panel annotation.
    """

    if port_style == "distributed":
        return "Ports spread across the hub edge"
    if port_style == "center":
        return "Ports stack at the hub midpoint"
    raise ValueError(f"Unsupported port style: {port_style}")


def _pair_positions(direction: str = "TB", gap: float = 80.0) -> List[Tuple[float, float]]:
    """Return fixed pair positions for the requested direction.

    Parameters
    ----------
    direction : str, default="TB"
        Direction identifier.
    gap : float, default=80.0
        Distance between the two nodes.

    Returns
    -------
    list[tuple[float, float]]
        Positions for a source-target pair.
    """

    half_gap = gap / 2.0
    if direction == "TB":
        return [(0.0, half_gap), (0.0, -half_gap)]
    if direction == "BT":
        return [(0.0, -half_gap), (0.0, half_gap)]
    if direction == "LR":
        return [(-half_gap, 0.0), (half_gap, 0.0)]
    if direction == "RL":
        return [(half_gap, 0.0), (-half_gap, 0.0)]
    raise ValueError(f"Unsupported direction: {direction}")


def _chain_positions(direction: str = "TB", spacing: float = 110.0) -> List[Tuple[float, float]]:
    """Return fixed three-node chain positions for a direction sweep.

    Parameters
    ----------
    direction : str, default="TB"
        Direction identifier.
    spacing : float, default=110.0
        Distance between consecutive nodes.

    Returns
    -------
    list[tuple[float, float]]
        Chain positions in node order.
    """

    if direction == "TB":
        return [(0.0, 120.0), (0.0, 10.0), (0.0, -100.0)]
    if direction == "BT":
        return [(0.0, -100.0), (0.0, 10.0), (0.0, 120.0)]
    if direction == "LR":
        return [(-spacing, 0.0), (0.0, 0.0), (spacing, 0.0)]
    if direction == "RL":
        return [(spacing, 0.0), (0.0, 0.0), (-spacing, 0.0)]
    raise ValueError(f"Unsupported direction: {direction}")


def _configure_graph_defaults(graph: DaguaGraph) -> None:
    """Apply shared node and edge defaults to a freshly built graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to mutate.

    Returns
    -------
    None
        The graph is updated in place.
    """

    _set_all_node_styles(graph, _base_node_style())
    if graph.num_edges > 0:
        _set_all_edge_styles(graph, _base_edge_style())


def _build_single_node_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a single-node demo graph.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and position tensor with shape ``[1, 2]``.
    """

    graph, positions = _single_node_graph("Sample")
    _configure_graph_defaults(graph)
    return graph, positions


def _build_pair_graph(direction: str = "TB") -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a simple two-node graph for edge and graph sweeps.

    Parameters
    ----------
    direction : str, default="TB"
        Graph direction.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and position tensor with shape ``[2, 2]``.
    """

    graph, positions = _pair_graph(
        _pair_positions(direction=direction),
        ["Source", "Target"],
        direction,
    )
    _configure_graph_defaults(graph)
    return graph, positions


def _build_chain3_graph(direction: str = "TB") -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a three-node chain graph.

    Parameters
    ----------
    direction : str, default="TB"
        Graph direction.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and position tensor with shape ``[3, 2]``.
    """

    graph = DaguaGraph(direction=direction)
    _apply_graph_style(graph)
    graph.add_node("A", label="Stage A")
    graph.add_node("B", label="Stage B")
    graph.add_node("C", label="Stage C")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    _configure_graph_defaults(graph)
    positions = torch.tensor(_chain_positions(direction=direction), dtype=torch.float32)
    return graph, positions


def _build_cluster_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a simple clustered chain graph.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and position tensor with shape ``[3, 2]``.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="One")
    graph.add_node("B", label="Two")
    graph.add_node("C", label="Three")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_cluster(
        "group",
        ["A", "B", "C"],
        style=_base_cluster_style(),
        label="Cluster",
    )
    _configure_graph_defaults(graph)
    positions = torch.tensor([[-70.0, 80.0], [0.0, 0.0], [70.0, -80.0]], dtype=torch.float32)
    return graph, positions


def _build_nested_cluster_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a nested-cluster graph for depth coloring demos.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and position tensor with shape ``[4, 2]``.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="Outer A")
    graph.add_node("B", label="Inner B")
    graph.add_node("C", label="Inner C")
    graph.add_node("D", label="Outer D")
    graph.add_edge("A", "B")
    graph.add_edge("B", "C")
    graph.add_edge("C", "D")
    graph.add_cluster(
        "outer",
        ["A", "B", "C", "D"],
        style=_base_cluster_style(),
        label="Outer",
    )
    graph.add_cluster(
        "inner",
        ["B", "C"],
        style=_base_cluster_style(),
        label="Inner",
        parent="outer",
    )
    _configure_graph_defaults(graph)
    positions = torch.tensor(
        [[-80.0, 110.0], [-25.0, 25.0], [25.0, -25.0], [80.0, -110.0]],
        dtype=torch.float32,
    )
    return graph, positions


def _build_crossing4_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a crossing-edge graph for jump-style sweeps.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and position tensor with shape ``[4, 2]``.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("A", label="A")
    graph.add_node("B", label="B")
    graph.add_node("C", label="C")
    graph.add_node("D", label="D")
    graph.add_edge("A", "D")
    graph.add_edge("B", "C")
    _configure_graph_defaults(graph)
    positions = torch.tensor(
        [[-120.0, 90.0], [120.0, 90.0], [-120.0, -90.0], [120.0, -90.0]],
        dtype=torch.float32,
    )
    return graph, positions


def _build_fan6_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a six-leaf fan-out graph for port-style sweeps.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and position tensor with shape ``[7, 2]``.
    """

    graph = DaguaGraph()
    _apply_graph_style(graph)
    graph.add_node("C", label="Hub")
    for node_id, label in zip(
        ("L1", "L2", "L3", "L4", "L5", "L6"),
        ("Leaf 1", "Leaf 2", "Leaf 3", "Leaf 4", "Leaf 5", "Leaf 6"),
    ):
        graph.add_node(node_id, label=label)
        graph.add_edge("C", node_id)
    _configure_graph_defaults(graph)
    positions = torch.tensor(
        [
            [0.0, 126.0],
            [-210.0, -40.0],
            [-126.0, -40.0],
            [-42.0, -40.0],
            [42.0, -40.0],
            [126.0, -40.0],
            [210.0, -40.0],
        ],
        dtype=torch.float32,
    )
    return graph, positions


def _build_wide_node_graph() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build a single node with multiple lines of text.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Graph and position tensor with shape ``[1, 2]``.
    """

    graph, positions = _single_node_graph("First Line\nSecond Line\nThird Line")
    _configure_graph_defaults(graph)
    if graph.node_styles[0] is not None:
        graph.node_styles[0].shape = "rect"
        graph.node_styles[0].mark_set("shape")
        graph.node_styles[0].corner_radius = 0.0
        graph.node_styles[0].min_width = TEXT_ALIGN_SWEEP_MIN_WIDTH
        graph.node_styles[0].min_height = TEXT_ALIGN_SWEEP_MIN_HEIGHT
    return graph, positions


def build_graph(sweep: SweepConfig, value: Any) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the fixed-position graph used for one sweep value.

    Parameters
    ----------
    sweep : SweepConfig
        Sweep definition.
    value : Any
        Sweep value for context-sensitive builders.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Configured graph and fixed positions.
    """

    if sweep.graph_builder == "single_node":
        return _build_single_node_graph()
    if sweep.graph_builder == "pair":
        if sweep.name == "edge_curvature":
            graph, positions = _build_pair_graph(direction="LR")
            positions = torch.tensor([[-90.0, 55.0], [90.0, -55.0]], dtype=torch.float32)
        else:
            graph, positions = _build_pair_graph(direction="TB")
            if sweep.name == "edge_routing":
                positions = torch.tensor([[-92.0, 62.0], [92.0, -62.0]], dtype=torch.float32)
            elif sweep.name == "edge_color_gradient":
                positions = torch.tensor(
                    _pair_positions(direction="LR", gap=EDGE_COLOR_GRADIENT_GAP),
                    dtype=torch.float32,
                )
            elif sweep.name in {
                "edge_arrow_types",
                "edge_arrow_fill",
                "edge_arrow_length",
                "edge_arrow_width",
            }:
                positions = torch.tensor(
                    _pair_positions(direction="TB", gap=EDGE_ARROW_DEMO_GAP),
                    dtype=torch.float32,
                )
            elif sweep.name == "edge_head_tail_labels":
                positions = torch.tensor(
                    _pair_positions(direction="TB", gap=EDGE_HEAD_TAIL_LABEL_GAP),
                    dtype=torch.float32,
                )
        return graph, positions
    if sweep.graph_builder == "chain3":
        direction = str(value) if sweep.name == "graph_direction" else "TB"
        return _build_chain3_graph(direction=direction)
    if sweep.graph_builder == "cluster":
        return _build_cluster_graph()
    if sweep.graph_builder == "nested_cluster":
        return _build_nested_cluster_graph()
    if sweep.graph_builder == "crossing4":
        return _build_crossing4_graph()
    if sweep.graph_builder == "fan6":
        return _build_fan6_graph()
    if sweep.graph_builder == "wide_node":
        return _build_wide_node_graph()
    raise ValueError(f"Unsupported graph builder: {sweep.graph_builder}")


def _node_style(graph: DaguaGraph) -> NodeStyle:
    """Return the shared node style for a sweep graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph with node styles assigned.

    Returns
    -------
    NodeStyle
        Shared node style object.
    """

    style = graph.node_styles[0]
    if style is None:
        raise ValueError("Expected a concrete node style.")
    return style


def _edge_style(graph: DaguaGraph) -> EdgeStyle:
    """Return the shared edge style for a sweep graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph with edge styles assigned.

    Returns
    -------
    EdgeStyle
        Shared edge style object.
    """

    style = graph.edge_styles[0]
    if style is None:
        raise ValueError("Expected a concrete edge style.")
    return style


def _cluster_styles(graph: DaguaGraph) -> List[ClusterStyle]:
    """Return all concrete cluster styles for a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph containing cluster styles.

    Returns
    -------
    list[ClusterStyle]
        Cluster style objects.
    """

    return [style for style in graph.cluster_styles.values()]


def apply_sweep_value(graph: DaguaGraph, sweep: SweepConfig, value: Any) -> None:
    """Apply a sweep value and all companion settings to a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to mutate.
    sweep : SweepConfig
        Sweep definition.
    value : Any
        Selected sweep value.

    Returns
    -------
    None
        The graph is updated in place.
    """

    if sweep.target == "node":
        style = _node_style(graph)
        setattr(style, sweep.field, value)
        if sweep.name == "node_gradient":
            style.fill = GRADIENT_FILL_COLOR
            style.gradient_color = GRADIENT_COLOR
            style.font_color = DARK_FONT_COLOR
            style.min_width = 100.0
            style.min_height = 60.0
        elif sweep.name == "node_gradient_angle":
            style.gradient = "linear"
            style.fill = GRADIENT_FILL_COLOR
            style.gradient_color = GRADIENT_COLOR
            style.font_color = DARK_FONT_COLOR
        elif sweep.name == "node_shape":
            graph.node_labels[0] = _shape_node_label(str(value))
            style.min_width = SHAPE_SWEEP_MIN_WIDTH
            style.min_height = SHAPE_SWEEP_MIN_HEIGHT
            if value in NON_INSETTABLE_SHAPES:
                # The current border inset helper does not support the
                # double-circle family of custom node paths, so keep the
                # gallery runnable by omitting the stroke for those shapes.
                style.stroke_width = 0.0
        elif sweep.name == "node_fill_pattern":
            style.min_width = 116.0
            style.min_height = 72.0
            graph.node_labels[0] = str(value).replace("_", " ").title()
            if value == "striped":
                style.fill_pattern_colors = _pattern_colors(3)
                style.fill_pattern_angle = 0.0
            elif value == "hatched":
                style.fill_pattern_colors = ["#56B4E9", "#D55E00"]
                style.fill_pattern_angle = 45.0
            elif value == "pie":
                style.fill_pattern_values = [3.0, 2.0, 1.0]
                style.fill_pattern_colors = _pattern_colors(3)
            else:
                style.fill_pattern_colors = None
                style.fill_pattern_values = None
                style.font_color = "#253140"
        elif sweep.name == "node_pie_chart":
            style.fill_pattern = "pie"
            style.fill_pattern_colors = _pattern_colors(len(value))
            style.shape = "circle"
            style.mark_set("shape")
            style.min_width = 96.0
            style.min_height = 96.0
            graph.node_labels[0] = ""
        elif sweep.name == "node_donut":
            style.fill_pattern = "pie"
            style.fill_pattern_values = [3.0, 2.0, 1.0, 4.0]
            style.fill_pattern_colors = _pattern_colors(4)
            style.shape = "circle"
            style.mark_set("shape")
            style.min_width = 96.0
            style.min_height = 96.0
            graph.node_labels[0] = ""
        elif sweep.name == "node_stroke_width":
            style.min_width = NODE_STROKE_SWEEP_MIN_WIDTH
            style.min_height = NODE_STROKE_SWEEP_MIN_HEIGHT
        elif sweep.name == "node_corner_radius":
            style.shape = "roundrect"
            style.min_width = 108.0
            style.min_height = 54.0
            # Mark shape as explicitly set so the cascade respects it
            # even though "roundrect" matches the NodeStyle class default.
            style.mark_set("shape")
        elif sweep.name == "node_border_position":
            style.shape = "roundrect"
            style.stroke_width = 4.0
            style.min_width = 108.0
            style.min_height = 54.0
        elif sweep.name == "node_stroke_cap":
            style.shape = "rect"
            style.corner_radius = 0.0
            style.stroke_dash = "dashed"
            style.stroke_width = 4.0
            style.min_width = 126.0
            style.min_height = 86.0
        elif sweep.name == "node_stroke_join":
            style.shape = "star"
            style.stroke_width = 5.0
            style.corner_radius = 0.0
            style.min_width = 120.0
            style.min_height = 120.0
            graph.node_labels[0] = ""
        elif sweep.name == "node_text_valign":
            style.shape = "rect"
            style.corner_radius = 0.0
            style.min_height = 100.0
        elif sweep.name == "node_text_rotation":
            graph.node_labels[0] = "Rotate"
            style.shape = "rect"
            style.mark_set("shape")
            style.corner_radius = 0.0
            style.min_width = 100.0
            style.min_height = 100.0
        elif sweep.name == "node_text_wrap":
            wrap_labels = {
                "none": "Readable label sample",
                "wrap": "Wrap this sample label across a few lines",
                "ellipsis": "Ellipsis keeps this longer sample readable",
            }
            graph.node_labels[0] = wrap_labels[str(value)]
            style.shape = "rect"
            style.mark_set("shape")
            style.corner_radius = 0.0
            if str(value) == "wrap":
                style.text_max_width = 80.0
                style.min_width = 110.0
                style.min_height = 72.0
            else:
                style.text_max_width = 92.0
                style.min_width = TEXT_WRAP_SWEEP_MIN_WIDTH
                style.min_height = TEXT_WRAP_SWEEP_MIN_HEIGHT
        elif sweep.name == "node_text_outline" and bool(value):
            style.text_outline_color = TEXT_OUTLINE_COLOR
            style.text_outline_width = 2.0
        elif sweep.name == "node_external_label":
            style.external_label = "ID 42"
            graph._theme.graph_style.margin = 20.0
        elif sweep.name == "node_shadow":
            style.fill = NODE_FILL
            style.stroke = NODE_STROKE
            style.min_width = 108.0
            style.min_height = 62.0
            style.shadow_offset = (5.0, -5.0)
            style.shadow_color = SHADOW_COLOR_LIGHT
            style.shadow_blur = 4.0
        elif sweep.name == "node_shadow_blur":
            style.shadow = True
            style.shadow_offset = (3.0, -3.0)
            style.shadow_color = SHADOW_COLOR_SOFT
        elif sweep.name == "node_border_opacity":
            style.stroke_width = 3.0
        elif sweep.name == "node_overflow_policy":
            graph.node_labels[0] = "Overflow demo"
            style.shape = "rect"
            style.mark_set("shape")
            style.corner_radius = 0.0
            style.font_size = 5.5
            style.min_width = 84.0
            style.min_height = 40.0
        elif sweep.name == "node_text_background":
            style.text_background_opacity = 0.9
        elif sweep.name == "node_text_align":
            style.shape = "rect"
            style.mark_set("shape")
            style.corner_radius = 0.0
            style.min_width = TEXT_ALIGN_SWEEP_MIN_WIDTH
            style.min_height = TEXT_ALIGN_SWEEP_MIN_HEIGHT
        elif sweep.name == "node_text_transform":
            graph.node_labels[0] = "Sample Text"
    elif sweep.target == "edge":
        style = _edge_style(graph)
        if sweep.name == "edge_taper":
            style.taper_width_start = 3.0
            style.taper_width_end = float(value)
            style.taper = float(value) < 3.0
        elif sweep.name == "edge_head_tail_labels":
            label_pairs = {
                "none": ("", ""),
                "head_tail": ("Head", "Tail"),
                "in_out": ("In", "Out"),
                "src_dst": ("Src", "Dst"),
            }
            head_label, tail_label = label_pairs[str(value)]
            style.head_label = head_label
            style.tail_label = tail_label
        else:
            setattr(style, sweep.field, value)
        if sweep.name in ARROW_DEMO_SWEEPS:
            for index in range(graph.num_nodes):
                node_style = graph.node_styles[index]
                if node_style is None:
                    continue
                node_style.shape = "rect"
                node_style.mark_set("shape")
                node_style.corner_radius = 0.0
                node_style.min_width = 96.0
                node_style.min_height = 46.0
            if graph.num_nodes >= 2:
                graph.node_labels[0] = "A"
                graph.node_labels[1] = "B"
            style.width = 2.4
            style.arrow_node_fraction = 0.0
            if sweep.name in {"edge_arrow_types", "edge_arrow_fill"}:
                style.arrow_length = 20.0
                style.arrow_width = 14.0
            if sweep.name in {"edge_arrow_length", "edge_arrow_width"}:
                style.width = 2.5
        elif sweep.name == "edge_color_gradient":
            style.color = EDGE_GRADIENT_START
            style.width = 4.0 if value == "source_to_target_wide" else 3.0
            if value in {"source_to_target", "source_to_target_wide"}:
                style.color_gradient = "source_to_target"
                style.color_gradient_end = EDGE_GRADIENT_END
            else:
                style.color_gradient = "none"
                style.color_gradient_end = ""
            graph.node_labels[0] = "Source\n#0057FF"
            graph.node_labels[1] = "Target\n#FF6A00"
            for node_style in graph.node_styles:
                if node_style is None:
                    continue
                node_style.shape = "rect"
                node_style.mark_set("shape")
                node_style.corner_radius = 0.0
                node_style.min_width = 118.0
                node_style.min_height = 56.0
        elif sweep.name == "edge_crossing_style":
            for edge_style in graph.edge_styles:
                if edge_style is not None:
                    edge_style.crossing_style = str(value)
                    edge_style.crossing_size = CROSSING_STYLE_DEMO_SIZE
                    edge_style.width = CROSSING_SWEEP_EDGE_WIDTH
        elif sweep.name == "edge_crossing_size":
            for edge_style in graph.edge_styles:
                if edge_style is not None:
                    edge_style.crossing_style = "arc"
                    edge_style.crossing_size = float(value)
                    edge_style.width = CROSSING_SWEEP_EDGE_WIDTH
        elif sweep.name == "edge_label_position":
            graph.edge_labels = ["weight=1.0"]
        elif sweep.name == "edge_head_tail_labels":
            node_style = _node_style(graph)
            node_style.shape = "rect"
            node_style.mark_set("shape")
            node_style.corner_radius = 0.0
            node_style.min_width = 104.0
            node_style.min_height = 48.0
            style.label_font_size = 12.0
            style.head_label_offset = 16.0
            style.tail_label_offset = 16.0
        elif sweep.name == "edge_arrow_fill" and value == "hollow":
            style.arrow_color = EDGE_COLOR
        elif sweep.name == "edge_routing":
            node_style = _node_style(graph)
            node_style.shape = "rect"
            node_style.mark_set("shape")
            node_style.corner_radius = 0.0
            node_style.min_width = 96.0
            node_style.min_height = 46.0
            style.width = 2.2
        if sweep.name == "edge_port_style":
            style.routing = "straight"
            style.width = 2.0
    elif sweep.target == "cluster":
        for style in _cluster_styles(graph):
            setattr(style, sweep.field, value)
        if sweep.name == "cluster_label_position":
            offsets = {
                "top-left": (12.0, 10.0),
                "top-center": (0.0, 10.0),
                "top-right": (-12.0, 10.0),
            }
            for style in _cluster_styles(graph):
                style.label_offset = offsets[str(value)]
    elif sweep.target == "graph":
        graph_style = graph._theme.graph_style
        if not isinstance(graph_style, GraphStyle):
            raise ValueError("Expected graph._theme.graph_style to be a GraphStyle.")
        setattr(graph_style, sweep.field, value)
        if sweep.name == "graph_background" and _is_dark_background(str(value)):
            node_style = _node_style(graph)
            node_style.fill = DARK_NODE_FILL
            node_style.stroke = DARK_NODE_STROKE
            node_style.font_color = DARK_FONT_COLOR
            if graph.num_edges > 0:
                edge_style = _edge_style(graph)
                edge_style.color = DARK_EDGE_COLOR
                edge_style.label_font_color = DARK_FONT_COLOR
                edge_style.label_background = DARK_LABEL_BACKGROUND
            for cluster_style in _cluster_styles(graph):
                cluster_style.fill = DARK_CLUSTER_FILL
                cluster_style.stroke = DARK_CLUSTER_STROKE
                cluster_style.font_color = DARK_FONT_COLOR
        elif sweep.name == "graph_direction":
            graph.direction = str(value)
            graph.node_labels = ["A", "B", "C"]
            if value in {"LR", "RL"}:
                graph_style.margin = GRAPH_DIRECTION_HORIZONTAL_MARGIN
            else:
                graph_style.margin = 20.0
            for node_style in graph.node_styles:
                if node_style is not None:
                    node_style.min_width = DIRECTION_SWEEP_NODE_WIDTH
                    node_style.min_height = DIRECTION_SWEEP_NODE_HEIGHT


def get_graphviz_attrs(
    sweep: SweepConfig,
    value: Any,
) -> Optional[Dict[str, str]]:
    """Return Graphviz attributes for one sweep value.

    Parameters
    ----------
    sweep : SweepConfig
        Sweep definition.
    value : Any
        Selected sweep value.

    Returns
    -------
    dict[str, str] | None
        Graphviz attributes, or ``None`` when the value has no analogue.
    """

    if sweep.gv_map is None:
        return None
    return sweep.gv_map.get(_value_key(value))


def _graphviz_cluster_base_attrs() -> Dict[str, str]:
    """Return the default cluster DOT attributes for comparisons.

    Returns
    -------
    dict[str, str]
        Base cluster DOT attributes.
    """

    return {
        "style": "filled",
        "fillcolor": BASE_CLUSTER_FILL,
        "color": BASE_CLUSTER_STROKE,
        "penwidth": "2.0",
        "fontname": "Helvetica",
        "fontsize": "13",
        "fontcolor": "#374151",
    }


def build_graphviz_spec(
    graph: DaguaGraph,
    sweep: SweepConfig,
    value: Any,
) -> Optional[GraphvizRenderSpec]:
    """Build the Graphviz render specification for one sweep value.

    Parameters
    ----------
    graph : DaguaGraph
        Graph being rendered.
    sweep : SweepConfig
        Sweep definition.
    value : Any
        Selected sweep value.

    Returns
    -------
    GraphvizRenderSpec | None
        Graphviz spec when comparison is possible.
    """

    attrs = get_graphviz_attrs(sweep, value)
    if attrs is None:
        return None
    spec = GraphvizRenderSpec(competitor_label=GRAPHVIZ_LABEL)
    if sweep.target == "node":
        spec.default_node_attrs.update(attrs)
    elif sweep.target == "edge":
        spec.default_edge_attrs.update(attrs)
    elif sweep.target == "cluster":
        cluster_attrs = _graphviz_cluster_base_attrs()
        cluster_attrs.update(attrs)
        for cluster_name in graph.clusters:
            spec.cluster_attrs[cluster_name] = dict(cluster_attrs)
    elif sweep.target == "graph":
        spec.graph_attrs.update(attrs)
    return spec


def _render_panel_case(
    graph: DaguaGraph,
    positions: torch.Tensor,
    sweep: SweepConfig,
    value: Any,
) -> AlbumCase:
    """Build a transient panel case for one sweep value.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to render.
    positions : torch.Tensor
        Fixed node positions with shape ``[N, 2]``.
    sweep : SweepConfig
        Sweep definition.
    value : Any
        Sweep value.

    Returns
    -------
    AlbumCase
        Transient render case carrying graph and Graphviz metadata.
    """

    return AlbumCase(
        case_id=f"{sweep.name}:{_value_key(value)}",
        category=sweep.category,
        filename=f"{sweep.name}_{_value_key(value)}.png",
        title=sweep.description,
        graph=graph,
        positions=positions,
        settings={"sweep": sweep.name, "value": _value_key(value)},
        graphviz=build_graphviz_spec(graph, sweep, value),
    )


def render_sweep(
    sweep: SweepConfig,
    output_dir: Path,
    graphviz_available: bool,
) -> Path:
    """Render a single sweep to a multi-panel PNG.

    Parameters
    ----------
    sweep : SweepConfig
        Sweep definition to render.
    output_dir : Path
        Root output directory.
    graphviz_available : bool
        Whether Graphviz is available in the current environment.

    Returns
    -------
    Path
        Final output image path.
    """

    n_values = len(sweep.values)
    n_columns = _column_count_for_sweep(sweep)
    n_value_rows = math.ceil(n_values / n_columns)
    has_graphviz = graphviz_available and sweep.gv_map is not None
    n_renderer_rows = 2 if has_graphviz else 1
    total_rows = n_value_rows * n_renderer_rows
    panel_size = _panel_size_for_sweep(sweep)
    panel_width_scale = panel_size[0] / SWEEP_PANEL_SIZE[0]

    fig_width = max(4.2, 2.7 * n_columns * panel_width_scale)
    fig_height = _figure_height_for_sweep(sweep, total_rows)
    fig, axes = plt.subplots(
        total_rows,
        n_columns,
        figsize=(fig_width, fig_height),
        squeeze=False,
    )
    fig.patch.set_facecolor(WHITE)
    fig.suptitle(sweep.description, fontsize=14, fontweight="bold", y=0.985)

    for row in axes:
        for axis in row:
            axis.axis("off")

    with tempfile.TemporaryDirectory(prefix=f"{sweep.name}_", dir=str(output_dir)) as temp_dir_str:
        temp_dir = Path(temp_dir_str)
        for index, (value, label) in enumerate(zip(sweep.values, sweep.labels)):
            value_row = index // n_columns
            column = index % n_columns
            dagua_row = value_row * n_renderer_rows

            graph, positions = build_graph(sweep, value)
            apply_sweep_value(graph, sweep, value)
            panel_case = _render_panel_case(graph, positions, sweep, value)

            dagua_path = temp_dir / f"dagua_{index}.png"
            _render_dagua_png(
                panel_case.graph,
                panel_case.positions,
                dagua_path,
                dpi=RAW_RENDER_DPI,
            )
            dagua_axis = axes[dagua_row, column]
            crop_content = sweep.name != "graph_margin"
            dagua_axis.imshow(
                _normalize_panel_image(
                    dagua_path,
                    panel_size=panel_size,
                    crop_content=crop_content,
                )
            )
            dagua_axis.set_title(label, fontsize=9)
            dagua_axis.axis("off")
            if sweep.name == "graph_margin":
                _draw_graph_margin_outline(dagua_axis, float(value))
            if sweep.name == "edge_port_style":
                dagua_axis.text(
                    0.5,
                    0.05,
                    _port_style_annotation(str(value)),
                    transform=dagua_axis.transAxes,
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    color="#4B5563",
                    bbox={
                        "boxstyle": "round,pad=0.25",
                        "facecolor": "#FFFFFFCC",
                        "edgecolor": "none",
                    },
                )
            if column == 0:
                dagua_axis.set_ylabel(
                    "Dagua",
                    fontsize=10,
                    fontweight="bold",
                    rotation=0,
                    labelpad=34,
                )

            if not has_graphviz:
                continue

            graphviz_axis = axes[dagua_row + 1, column]
            graphviz_axis.axis("off")
            graphviz_axis.set_title(label, fontsize=9)
            if column == 0:
                graphviz_axis.set_ylabel(
                    GRAPHVIZ_LABEL,
                    fontsize=10,
                    fontweight="bold",
                    rotation=0,
                    labelpad=34,
                )

            if panel_case.graphviz is None:
                graphviz_axis.text(0.5, 0.5, "N/A", ha="center", va="center", fontsize=12)
                continue

            graphviz_path = temp_dir / f"graphviz_{index}.png"
            dot_source = _build_graphviz_dot(panel_case.graph, panel_case.graphviz)
            _render_graphviz_png(
                dot_source,
                graphviz_path,
                panel_case.graphviz.engine,
                dpi=RAW_RENDER_DPI,
            )
            graphviz_axis.imshow(_normalize_panel_image(graphviz_path, panel_size=panel_size))

    plt.tight_layout(rect=_tight_layout_rect_for_sweep(sweep), pad=0.35, w_pad=0.2, h_pad=0.45)
    category_dir = output_dir / sweep.category
    category_dir.mkdir(parents=True, exist_ok=True)
    output_path = category_dir / f"sweep_{sweep.name}.png"
    fig.savefig(output_path, dpi=180, bbox_inches="tight", facecolor=WHITE)
    plt.close(fig)
    return output_path


def _summary_lines(
    sweeps: Sequence[SweepConfig],
    image_paths: Sequence[Path],
    graphviz_available: bool,
) -> List[str]:
    """Build the markdown summary contents.

    Parameters
    ----------
    sweeps : sequence[SweepConfig]
        Rendered sweeps.
    image_paths : sequence[Path]
        Rendered image paths.
    graphviz_available : bool
        Whether Graphviz was available during rendering.

    Returns
    -------
    list[str]
        Markdown lines.
    """

    category_counts: DefaultDict[str, int] = defaultdict(int)
    for sweep in sweeps:
        category_counts[sweep.category] += 1

    lines = [
        "# Comprehensive Cosmetic Gallery",
        "",
        f"- Generated at: {datetime.now(timezone.utc).isoformat()}",
        f"- Graphviz available: {'yes' if graphviz_available else 'no'}",
        f"- Total sweeps: {len(sweeps)}",
        f"- Total images: {len(image_paths)}",
        "",
        "## Categories",
        "",
    ]
    for category in sorted(category_counts):
        lines.append(f"- {category}: {category_counts[category]}")
    return lines


def build_comprehensive_gallery(
    output_dir: str = DEFAULT_OUTPUT_DIR,
    category: Optional[str] = None,
    sweep: Optional[str] = None,
) -> CosmeticAlbumResult:
    """Render the comprehensive gallery and emit a manifest.

    Parameters
    ----------
    output_dir : str, default=DEFAULT_OUTPUT_DIR
        Root output directory.
    category : str | None, default=None
        Optional category prefix filter.
    sweep : str | None, default=None
        Optional exact sweep-name filter.

    Returns
    -------
    CosmeticAlbumResult
        Result record with manifest and image paths.
    """

    root = Path(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    sweeps = build_sweep_catalog()
    if category is not None:
        sweeps = [entry for entry in sweeps if entry.category.startswith(category)]
    if sweep is not None:
        sweeps = [entry for entry in sweeps if entry.name == sweep]
    if not sweeps:
        raise ValueError("No sweeps matched the requested filters.")

    graphviz_available = _graphviz_available()
    image_paths: List[Path] = []
    manifest_sweeps: List[Dict[str, object]] = []
    for entry in sweeps:
        image_path = render_sweep(entry, root, graphviz_available)
        image_paths.append(image_path)
        manifest_sweeps.append(
            {
                "name": entry.name,
                "category": entry.category,
                "description": entry.description,
                "n_values": len(entry.values),
                "has_graphviz_comparison": entry.gv_map is not None,
                "image_path": image_path.relative_to(root).as_posix(),
            }
        )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "graphviz_available": graphviz_available,
        "total_sweeps": len(sweeps),
        "total_images": len(image_paths),
        "filters": {"category": category, "sweep": sweep},
        "sweeps": manifest_sweeps,
    }
    manifest_path = root / "manifest.json"
    manifest_path.write_text(f"{json.dumps(manifest, indent=2)}\n", encoding="utf-8")

    summary_path = root / "summary.md"
    summary_path.write_text(
        "\n".join(_summary_lines(sweeps, image_paths, graphviz_available)).rstrip() + "\n",
        encoding="utf-8",
    )

    return CosmeticAlbumResult(
        output_dir=str(root),
        manifest_path=str(manifest_path),
        image_paths=[str(path) for path in image_paths],
    )


def _build_parser() -> argparse.ArgumentParser:
    """Build the command-line parser for the gallery generator.

    Returns
    -------
    argparse.ArgumentParser
        Configured parser.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-dir",
        default=DEFAULT_OUTPUT_DIR,
        help="Output directory for gallery artifacts.",
    )
    parser.add_argument(
        "--category",
        default=None,
        help="Category prefix filter such as 'nodes/shapes'.",
    )
    parser.add_argument(
        "--sweep",
        default=None,
        help="Exact sweep name to render.",
    )
    return parser


def main() -> int:
    """Run the CLI entry point.

    Returns
    -------
    int
        Process exit code.
    """

    parser = _build_parser()
    args = parser.parse_args()
    result = build_comprehensive_gallery(
        output_dir=args.output_dir,
        category=args.category,
        sweep=args.sweep,
    )
    print(result.manifest_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
