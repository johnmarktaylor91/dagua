#!/usr/bin/env python
# ruff: noqa: E402
"""Build fixed-size gallery audit cards for LLM-friendly visual review.

This script creates atomic reference cards, Graphviz comparison cards, combo
cards, navigation boards, and a JSONL index under ``eval_output/gallery_audit``.
The artifacts are intentionally plain and deterministic so iterative review can
focus on one visual variable at a time. Recent render-polish fixtures also rely
on three gallery-specific adaptations documented here: dark headers switch to a
separate palette when a card's graph background is visually dark, decorative
fill cases apply per-shape overrides so labels remain legible inside box3d,
cloud, and patterned nodes, and strip cards allocate equal panel widths so
side-by-side comparisons do not drift as captions or dividers change.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import shutil
import subprocess
import sys
import tempfile
from dataclasses import dataclass, fields
from pathlib import Path
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import matplotlib
import torch
from PIL import Image, ImageDraw, ImageFont

Image.MAX_IMAGE_PIXELS = None

matplotlib.use("Agg")
import matplotlib.pyplot as plt

logging.getLogger("matplotlib.font_manager").setLevel(logging.ERROR)

from dagua import DaguaGraph, render
from dagua.styles import GRAPHVIZ_STRICT_THEME, ClusterStyle, EdgeStyle, GraphStyle, NodeStyle
from scripts.generate_cosmetic_album import VARIED_EXTERNAL_LABELS, build_case_catalog

WHITE = "#FFFFFF"
DEFAULT_COMPARISON_STROKE = "#222222"
TEXT_COLOR = "#17212B"
MUTED_TEXT_COLOR = "#4A5868"
LABEL_BAR = "#F4F6F8"
LABEL_BAR_DARK = "#1A2332"
LABEL_TEXT_DARK = "#E8ECF0"
NODE_FILL = "#DCEBFA"
NODE_STROKE = "#4C77A3"
EDGE_COLOR = "#5F6C7B"
CLUSTER_FILL = "#EAF1F8"
SATURATED_CLUSTER_FILL = "#4A90D9"
CLUSTER_STROKE = "#A9B8C7"
GRADIENT_FILL = "#2196F3"
GRADIENT_COLOR = "#FF9800"
PATTERN_COLORS = ["#56B4E9", "#D55E00", "#009E73", "#E69F00"]
DARK_NODE_FILL = "#374151"
DARK_NODE_STROKE = "#CBD5E1"
DARK_EDGE_COLOR = "#E2E8F0"
DARK_CLUSTER_FILL = "#1F2937"
DARK_CLUSTER_STROKE = "#94A3B8"
DARK_LABEL_BG = "#111827"
SHADOW_COLOR = "#0000002A"
EDGE_GRADIENT_END = "#FF9800"
TEXT_OUTLINE_COLOR = "#FFFFFF"
OVERFLOW_DEMO_LABEL = "Processing validation stage output results"
OVERFLOW_EXPAND_LABEL = "Processing validation stage output expanded"
WRAP_DEMO_LABEL = "Process validation stage output results"
ELLIPSIS_DEMO_LABEL = "A long label for truncation"
PANEL_HALF_WIDTH = 800
PANEL_HEIGHT = 600
RENDER_DPI = 100
CARD_DPI = 200
CARD_SIZE: Tuple[int, int] = (1600, 1200)
PANEL_SIZE: Tuple[int, int] = (PANEL_HALF_WIDTH, PANEL_HEIGHT)
COMPARISON_SIZE: Tuple[int, int] = (1600, 600)
STRIP_CARD_SIZE: Tuple[int, int] = (1568, 600)
CARD_CONTENT_INSET: Tuple[int, int, int, int] = (72, 112, 72, 72)
PANEL_CONTENT_INSET: Tuple[int, int, int, int] = (42, 84, 42, 42)
THUMBNAIL_SIZE: Tuple[int, int] = (800, 600)
BOARD_SIZE: Tuple[int, int] = (1600, 1200)
HEADER_HEIGHT = 88
BOARD_TOP_ROW_MARGIN = 64
BOARD_GRID_TOP = HEADER_HEIGHT + BOARD_TOP_ROW_MARGIN
BOARD_CELL_SIZE: Tuple[int, int] = (
    BOARD_SIZE[0] // 2,
    (BOARD_SIZE[1] - BOARD_GRID_TOP) // 2,
)
STRIP_HEADER_HEIGHT = 56
STRIP_PANEL_LABEL_HEIGHT = 52
STRIP_PANEL_DIVIDER_WIDTH = 2
STRIP_PANEL_INSET: Tuple[int, int, int, int] = (
    24,
    STRIP_PANEL_LABEL_HEIGHT + 18,
    24,
    24,
)
CARD_FIGSIZE: Tuple[float, float] = (8.0, 6.0)
PANEL_FIGSIZE: Tuple[float, float] = (
    PANEL_HALF_WIDTH / RENDER_DPI,
    PANEL_HEIGHT / RENDER_DPI,
)
CONTENT_CROP_PADDING = 12
PAIR_DEFAULT_GAP = 260.0
PAIR_SCALAR_COMPARISON_GAP = 180.0
PAIR_ARROW_GAP = 130.0
CURVATURE_CARD_MARGIN = 140.0
STRIP_CURVATURE_CARD_MARGIN = 80.0
ARROW_DEMO_EDGE_WIDTH = 3.0
ARROW_DEMO_NODE_FRACTION = 0.5
NODE_SHAPE_PARITY_CARD_IDS = frozenset(
    f"nodes_shapes_{slug}"
    for slug in (
        "rect",
        "roundrect",
        "ellipse",
        "diamond",
        "circle",
        "triangle",
        "hexagon",
        "pentagon",
        "octagon",
        "star",
        "cylinder",
        "parallelogram",
        "trapezoid",
        "double_circle",
        "cloud",
        "stadium",
        "tab",
        "note",
        "document",
        "box3d",
    )
)
NODE_PAIR_PARITY_CARD_IDS = frozenset(
    f"nodes_{category}_{feature}_{slug}"
    for category, feature, slugs in (
        ("borders", "stroke_dash", ("solid", "dashed", "dotted")),
        ("borders", "stroke_width", ("0_5", "1_5", "3_0", "5_0")),
        ("borders", "border_opacity", ("0_2", "0_5", "0_8", "1_0")),
        ("borders", "border_count", ("1_vs_2", "2_vs_3")),
        ("borders", "border_position", ("inside", "outside")),
        ("fills", "gradient", ("solid", "linear", "radial")),
        ("fills", "fill_pattern", ("solid", "striped", "hatched", "pie")),
        ("fills", "opacity", ("0_2", "0_5", "0_8", "1_0")),
    )
    for slug in slugs
)
EDGE_PAIR_PARITY_CARD_IDS = frozenset(
    f"edges_{category}_{slug}" if feature == "" else f"edges_{category}_{feature}_{slug}"
    for category, feature, slugs in (
        (
            "arrows",
            "",
            ("normal", "vee", "dot", "diamond", "tee", "crow", "circle", "open"),
        ),
        ("arrows", "arrow_fill", ("filled", "hollow")),
        ("styles", "style", ("solid", "dashed", "dotted")),
        ("styles", "port_indicator", ("circle", "diamond")),
        ("styles", "width", ("0_5", "1_5", "3_0", "5_0")),
    )
    for slug in slugs
)
DECORATIVE_FILL_CARD_IDS = frozenset(
    NODE_SHAPE_PARITY_CARD_IDS | NODE_PAIR_PARITY_CARD_IDS | EDGE_PAIR_PARITY_CARD_IDS
)
DECORATIVE_FILL_CARD_MIN_HEIGHT = 110.0
DECORATIVE_FILL_CARD_PADDING: Tuple[float, float] = (8.0, 4.0)
GRAPHVIZ_PARITY_MAX_NODE_WIDTH = 200.0
GRAPHVIZ_PARITY_MAX_NODE_HEIGHT = DECORATIVE_FILL_CARD_MIN_HEIGHT
SCALAR_NODE_COMPARISON_FEATURES = frozenset(
    {
        "font_size",
        "opacity",
        "corner_radius",
        "border_count",
        "border_opacity",
        "stroke_width",
        "border_position",
    }
)
STRIP_REFERENCE_FEATURES = frozenset(
    {
        ("nodes/borders", "stroke_width"),
        ("nodes/fills", "opacity"),
        ("nodes/borders", "corner_radius"),
        ("nodes/borders", "border_opacity"),
        ("edges/styles", "width"),
        ("edges/routing", "curvature"),
        ("edges/advanced", "taper"),
    }
)
COMBO_INTERNAL_FIELDS = frozenset({"combo", "preserve_crossing_width"})

GRAPHVIZ_SHAPE_MAP: Dict[str, Dict[str, str]] = {
    "rect": {"shape": "box"},
    "roundrect": {"shape": "box", "style": "filled,rounded"},
    "ellipse": {"shape": "ellipse"},
    "diamond": {"shape": "diamond"},
    "circle": {"shape": "circle"},
    "triangle": {"shape": "triangle"},
    "hexagon": {"shape": "hexagon"},
    "pentagon": {"shape": "pentagon"},
    "octagon": {"shape": "octagon"},
    "star": {"shape": "star"},
    "cylinder": {"shape": "cylinder"},
    "parallelogram": {"shape": "parallelogram"},
    "trapezoid": {"shape": "trapezium"},
    "double_circle": {"shape": "doublecircle"},
    "tab": {"shape": "tab"},
    "note": {"shape": "note"},
    "box3d": {"shape": "box3d"},
}
GRAPHVIZ_ARROW_MAP: Dict[str, Dict[str, str]] = {
    "normal": {"arrowhead": "normal"},
    "vee": {"arrowhead": "vee"},
    "dot": {"arrowhead": "dot"},
    "diamond": {"arrowhead": "diamond"},
    "tee": {"arrowhead": "tee"},
    "crow": {"arrowhead": "crow"},
    "circle": {"arrowhead": "circle"},
    "open": {"arrowhead": "open"},
}
GRAPH_DIRECTION_FIELDS = {"direction"}
GRAPHVIZ_NODE_SHAPES = frozenset(
    {
        "rect",
        "roundrect",
        "ellipse",
        "circle",
        "diamond",
        "triangle",
        "hexagon",
        "pentagon",
        "octagon",
        "star",
        "cylinder",
        "parallelogram",
        "trapezoid",
        "tab",
        "note",
        "box3d",
    }
)
GRAPHVIZ_ARROWS = frozenset(
    {
        "normal",
        "inv",
        "dot",
        "box",
        "vee",
        "tee",
        "crow",
        "diamond",
        "curve",
        "icurve",
        "simple",
        "fancy",
        "wedge",
        "bracket",
        "none",
        "open",
        "circle",
    }
)
CYTOSCAPE_ARROWS = frozenset(
    {
        "crows_foot_one",
        "crows_foot_many",
        "crows_foot_one_mandatory",
        "crows_foot_many_mandatory",
        "crows_foot_many_optional",
        "triangle_tee",
    }
)


def _classify_tier(competitor_tools: Sequence[str]) -> str:
    """Classify a card or spec from its competitor tool availability.

    Parameters
    ----------
    competitor_tools : Sequence[str]
        Ordered competitor preference tuple.

    Returns
    -------
    str
        ``"A"`` for Graphviz-backed cards, ``"B"`` for non-Graphviz
        automated references, and ``"C"`` for heuristic-only cards.
    """

    if "graphviz" in competitor_tools:
        return "A"
    if competitor_tools:
        return "B"
    return "C"


def _feature_competitor_tools(category: str, feature: str) -> Tuple[str, ...]:
    """Return conservative competitor tools for a feature family.

    Parameters
    ----------
    category : str
        Atomic spec category path.
    feature : str
        Atomic feature name.

    Returns
    -------
    tuple[str, ...]
        Ordered preferred competitor names.
    """

    feature_key = (category, feature)
    mapping: Dict[Tuple[str, str], Tuple[str, ...]] = {
        ("nodes/shapes", "shape"): ("graphviz", "mermaid"),
        ("nodes/fills", "gradient"): ("graphviz", "cytoscape"),
        ("nodes/fills", "fill_pattern"): ("graphviz", "cytoscape"),
        ("nodes/fills", "opacity"): ("graphviz",),
        ("nodes/borders", "stroke_width"): ("graphviz",),
        ("nodes/borders", "border_opacity"): ("graphviz",),
        ("nodes/borders", "stroke_dash"): ("cytoscape",),
        ("nodes/borders", "border_count"): ("cytoscape",),
        ("nodes/borders", "border_position"): ("cytoscape",),
        ("nodes/text", "font_weight"): ("graphviz",),
        ("nodes/text", "font_style"): ("graphviz",),
        ("nodes/text", "font_size"): ("graphviz",),
        ("nodes/text", "font_family"): ("graphviz",),
        ("nodes/text", "font_color"): ("graphviz",),
        ("nodes/text", "text_align"): ("graphviz",),
        ("nodes/text", "text_valign"): ("graphviz",),
        ("nodes/text", "external_label"): ("graphviz",),
        ("nodes/text", "text_wrap"): ("cytoscape",),
        ("nodes/text", "text_ellipsis"): ("cytoscape",),
        ("nodes/text", "text_transform"): ("cytoscape",),
        ("nodes/text", "text_rotation"): ("cytoscape",),
        ("nodes/labels", "external_label"): ("graphviz",),
        ("edges/arrows", "arrow"): ("graphviz", "cytoscape"),
        ("edges/routing", "routing"): ("graphviz", "cytoscape"),
        ("edges/styles", "style"): ("graphviz",),
        ("edges/styles", "width"): ("graphviz", "mermaid"),
        ("edges/styles", "stroke_dash"): ("cytoscape",),
        ("edges/advanced", "taper"): ("graphviz",),
        ("edges/advanced", "color_gradient"): ("cytoscape",),
        ("edges/advanced", "line_cap"): ("cytoscape",),
        ("edges/advanced", "line_join"): ("cytoscape",),
        ("edges/labels", "external_label"): ("graphviz",),
        ("edges/labels", "label_position"): ("graphviz",),
        ("edges/labels", "head_tail_label"): ("graphviz",),
        ("clusters/styles", "cluster_style"): ("graphviz",),
        ("clusters/styles", "fill"): ("graphviz",),
        ("clusters/styles", "stroke"): ("graphviz",),
        ("clusters/styles", "border"): ("graphviz",),
        ("clusters", "stroke_dash"): ("graphviz",),
        ("clusters", "label_position"): ("graphviz",),
        ("clusters", "opacity"): ("graphviz",),
        ("graph", "background_color"): ("graphviz",),
        ("graph", "direction"): ("graphviz",),
        ("graph", "margin"): ("graphviz",),
    }
    return mapping.get(feature_key, ())


@dataclass(frozen=True)
class FeatureValue:
    """One concrete value for a reference feature.

    Parameters
    ----------
    slug : str
        Stable filename-safe value identifier.
    label : str
        Human-readable value label.
    params : dict[str, object]
        JSON-serializable parameters used to apply the value.
    graphviz_attrs : dict[str, str] | None, default=None
        DOT attributes for comparison renders when available.
    """

    slug: str
    label: str
    params: Dict[str, object]
    graphviz_attrs: Optional[Dict[str, str]] = None


@dataclass(frozen=True)
class AtomicCardSpec:
    """Definition of one atomic reference feature family.

    Parameters
    ----------
    target : str
        Style target: ``"node"``, ``"edge"``, ``"cluster"``, or ``"graph"``.
    category : str
        Output category path below ``cards/reference``.
    feature : str
        Stable feature identifier.
    fields : tuple[str, ...]
        Backing style fields validated against the dataclass surface.
    fixture : str
        Canonical fixture name used for all values in the category.
    values : tuple[FeatureValue, ...]
        Ordered values for the feature family.
    filename_prefix : str, default=""
        Optional stem prefix for categories that contain multiple features.
    sensitivity : str, default="coarse"
        Audit sensitivity classification written into the JSONL index.
    competitor_tools : tuple[str, ...], default=()
        Ordered competitor preference for this feature family.
    tier : str, default=""
        Audit tier. Empty values are derived from ``competitor_tools``.
    """

    target: str
    category: str
    feature: str
    fields: Tuple[str, ...]
    fixture: str
    values: Tuple[FeatureValue, ...]
    filename_prefix: str = ""
    sensitivity: str = "coarse"
    competitor_tools: Tuple[str, ...] = ()
    tier: str = ""

    def __post_init__(self) -> None:
        """Derive the audit tier when the caller leaves it empty.

        Returns
        -------
        None
            Frozen dataclass fields are updated through ``object.__setattr__``.
        """

        if not self.tier:
            object.__setattr__(self, "tier", _classify_tier(self.competitor_tools))


@dataclass(frozen=True)
class ReferenceCardItem:
    """One resolved atomic reference card to render.

    Parameters
    ----------
    card_id : str
        Stable identifier used in the index and tests.
    spec : AtomicCardSpec
        Parent feature specification.
    value : FeatureValue
        Concrete feature value to render.
    relative_path : str
        Relative PNG path below the gallery audit root.
    comparison_relative_path : str | None
        Relative comparison PNG path when available.
    """

    card_id: str
    spec: AtomicCardSpec
    value: FeatureValue
    relative_path: str
    comparison_relative_path: Optional[str]


@dataclass(frozen=True)
class ComboCardSpec:
    """Definition of one combo card imported from the cosmetic album catalog.

    Parameters
    ----------
    case_id : str
        Stable combo identifier.
    combo_kind : str
        Directory name such as ``"2way"`` or ``"5way"``.
    title : str
        Human-readable combo title.
    settings : dict[str, object]
        JSON-serializable cosmetic settings imported from the source catalog.
    competitor_tools : tuple[str, ...], default=()
        Ordered competitor preference for this combo.
    tier : str, default=""
        Audit tier. Empty values are derived from ``competitor_tools``.
    """

    case_id: str
    combo_kind: str
    title: str
    settings: Dict[str, object]
    competitor_tools: Tuple[str, ...] = ()
    tier: str = ""

    def __post_init__(self) -> None:
        """Derive the audit tier when the caller leaves it empty.

        Returns
        -------
        None
            Frozen dataclass fields are updated through ``object.__setattr__``.
        """

        if not self.tier:
            object.__setattr__(self, "tier", _classify_tier(self.competitor_tools))


@dataclass(frozen=True)
class ComboCardItem:
    """One resolved combo card to render.

    Parameters
    ----------
    card_id : str
        Stable identifier used in the index and tests.
    spec : ComboCardSpec
        Parent combo specification.
    relative_path : str
        Relative PNG path below the gallery audit root.
    """

    card_id: str
    spec: ComboCardSpec
    relative_path: str


@dataclass(frozen=True)
class EvilCardSpec:
    """Specification for one evil stress-test card.

    Parameters
    ----------
    case_id : str
        Stable evil-case identifier.
    title : str
        Human-readable case title imported from the cosmetic album catalog.
    settings : dict[str, object]
        JSON-serializable cosmetic settings imported from the source catalog.
    graph : DaguaGraph
        Pre-built stress graph to render.
    positions : torch.Tensor
        Fixed node positions with shape ``[N, 2]``.
    competitor_tools : tuple[str, ...], default=()
        Ordered competitor preference for this evil stress card.
    tier : str, default=""
        Audit tier. Empty values are derived from ``competitor_tools``.
    """

    case_id: str
    title: str
    settings: Dict[str, object]
    graph: DaguaGraph
    positions: torch.Tensor
    competitor_tools: Tuple[str, ...] = ()
    tier: str = ""

    def __post_init__(self) -> None:
        """Derive the audit tier when the caller leaves it empty.

        Returns
        -------
        None
            Frozen dataclass fields are updated through ``object.__setattr__``.
        """

        if not self.tier:
            object.__setattr__(self, "tier", _classify_tier(self.competitor_tools))


@dataclass(frozen=True)
class EvilCardItem:
    """Resolved evil card ready for rendering.

    Parameters
    ----------
    card_id : str
        Stable identifier used in the index and tests.
    spec : EvilCardSpec
        Parent evil-card specification.
    relative_path : str
        Relative PNG path below the gallery audit root.
    """

    card_id: str
    spec: EvilCardSpec
    relative_path: str


@dataclass(frozen=True)
class StripCardItem:
    """One resolved strip card to render.

    Parameters
    ----------
    card_id : str
        Stable identifier used in the index and tests.
    spec : AtomicCardSpec
        Parent feature specification shown in the strip.
    members : tuple[ReferenceCardItem, ...]
        Atomic reference cards rendered side by side in the strip.
    relative_path : str
        Relative PNG path below the gallery audit root.
    """

    card_id: str
    spec: AtomicCardSpec
    members: Tuple[ReferenceCardItem, ...]
    relative_path: str


@dataclass(frozen=True)
class GalleryAuditResult:
    """Summary of generated gallery audit artifacts.

    Parameters
    ----------
    output_dir : str
        Root output directory.
    index_path : str
        JSONL index path, or an empty string when not built.
    reference_count : int
        Number of reference cards written.
    comparison_count : int
        Number of comparison cards written.
    combo_count : int
        Number of combo cards written.
    evil_count : int
        Number of evil stress cards written.
    board_count : int
        Number of board images written.
    """

    output_dir: str
    index_path: str
    reference_count: int
    comparison_count: int
    combo_count: int
    evil_count: int
    board_count: int


def _value(
    slug: str,
    label: str,
    params: Mapping[str, object],
    graphviz_attrs: Optional[Mapping[str, str]] = None,
) -> FeatureValue:
    """Build a feature value with copied parameter dictionaries.

    Parameters
    ----------
    slug : str
        Stable filename-safe slug.
    label : str
        Human-readable label.
    params : Mapping[str, object]
        JSON-serializable parameter mapping.
    graphviz_attrs : Mapping[str, str] | None, optional
        Optional DOT attribute mapping used for comparisons.

    Returns
    -------
    FeatureValue
        Frozen feature value record.
    """

    gv_attrs = dict(graphviz_attrs) if graphviz_attrs is not None else None
    return FeatureValue(slug=slug, label=label, params=dict(params), graphviz_attrs=gv_attrs)


def _spec(
    target: str,
    category: str,
    feature: str,
    fields_: Sequence[str],
    fixture: str,
    values: Sequence[FeatureValue],
    filename_prefix: str = "",
    competitor_tools: Optional[Sequence[str]] = None,
    tier: str = "",
) -> AtomicCardSpec:
    """Build an atomic feature spec from simple constructor arguments.

    Parameters
    ----------
    target : str
        Style target.
    category : str
        Output category below ``cards/reference``.
    feature : str
        Stable feature identifier.
    fields_ : Sequence[str]
        Style fields or graph-level fields backing the feature.
    fixture : str
        Canonical fixture name.
    values : Sequence[FeatureValue]
        Ordered concrete values to render.
    filename_prefix : str, default=""
        Optional filename prefix for categories with several feature families.
    competitor_tools : Sequence[str] | None, optional
        Ordered competitor preference. Derived from the feature inventory when
        omitted.
    tier : str, default=""
        Optional explicit tier.

    Returns
    -------
    AtomicCardSpec
        Frozen feature specification.
    """

    return AtomicCardSpec(
        target=target,
        category=category,
        feature=feature,
        fields=tuple(fields_),
        fixture=fixture,
        values=tuple(values),
        filename_prefix=filename_prefix,
        competitor_tools=tuple(competitor_tools)
        if competitor_tools is not None
        else _feature_competitor_tools(category, feature),
        tier=tier,
    )


def _style_field_names(style_type: type[Any]) -> set[str]:
    """Return the dataclass field names for a style type.

    Parameters
    ----------
    style_type : type[Any]
        Dataclass type to introspect.

    Returns
    -------
    set[str]
        Field names defined on the dataclass.
    """

    return {field_info.name for field_info in fields(style_type)}


def _validate_reference_specs(specs: Sequence[AtomicCardSpec]) -> None:
    """Validate that requested feature fields exist on the backing types.

    Parameters
    ----------
    specs : Sequence[AtomicCardSpec]
        Reference specs to validate.

    Returns
    -------
    None

    Raises
    ------
    ValueError
        Raised when a spec references an unknown style field.
    """

    allowed_fields = {
        "node": _style_field_names(NodeStyle),
        "edge": _style_field_names(EdgeStyle),
        "cluster": _style_field_names(ClusterStyle),
        "graph": _style_field_names(GraphStyle) | GRAPH_DIRECTION_FIELDS,
    }
    errors: List[str] = []
    for spec in specs:
        unknown = sorted(set(spec.fields) - allowed_fields[spec.target])
        if unknown:
            errors.append(
                f"{spec.category}:{spec.feature} references unknown {spec.target} fields {unknown}"
            )
    if errors:
        raise ValueError("; ".join(errors))


def _base_graph_style() -> GraphStyle:
    """Return the Graphviz-strict base graph style for audit fixtures.

    Returns
    -------
    GraphStyle
        Deep copy of the graph style from ``GRAPHVIZ_STRICT_THEME``.
    """

    return copy.deepcopy(GRAPHVIZ_STRICT_THEME.graph_style)


def _base_node_style() -> NodeStyle:
    """Return the Graphviz-strict base node style for audit fixtures.

    Returns
    -------
    NodeStyle
        Deep copy of the default node style from ``GRAPHVIZ_STRICT_THEME``.
    """

    return copy.deepcopy(GRAPHVIZ_STRICT_THEME.node_styles["default"])


def _base_edge_style() -> EdgeStyle:
    """Return the Graphviz-strict base edge style for audit fixtures.

    Returns
    -------
    EdgeStyle
        Deep copy of the default edge style from ``GRAPHVIZ_STRICT_THEME``.
    """

    return copy.deepcopy(GRAPHVIZ_STRICT_THEME.edge_styles["default"])


def _base_cluster_style() -> ClusterStyle:
    """Return the Graphviz-strict base cluster style for audit fixtures.

    Returns
    -------
    ClusterStyle
        Deep copy of the cluster style from ``GRAPHVIZ_STRICT_THEME``.
    """

    return copy.deepcopy(GRAPHVIZ_STRICT_THEME.cluster_style)


def _set_all_node_styles(graph: DaguaGraph, style: NodeStyle) -> None:
    """Assign copied node styles to every node in a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to mutate.
    style : NodeStyle
        Style copied to each node.

    Returns
    -------
    None
        The graph is mutated in place.
    """

    graph.node_styles = [copy.deepcopy(style) for _ in range(graph.num_nodes)]


def _set_all_edge_styles(graph: DaguaGraph, style: EdgeStyle) -> None:
    """Assign copied edge styles to every edge in a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to mutate.
    style : EdgeStyle
        Style copied to each edge.

    Returns
    -------
    None
        The graph is mutated in place.
    """

    graph.edge_styles = [copy.deepcopy(style) for _ in range(graph.edge_index.shape[1])]


def _configure_fixture_defaults(graph: DaguaGraph) -> None:
    """Apply shared graph, node, and edge defaults to a new fixture.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to configure.

    Returns
    -------
    None
        The graph is mutated in place.
    """

    graph._theme.graph_style = _base_graph_style()
    _set_all_node_styles(graph, _base_node_style())
    if graph.edge_index.shape[1] > 0:
        _set_all_edge_styles(graph, _base_edge_style())


def _pair_positions(node_gap: float = PAIR_DEFAULT_GAP, layout: str = "vertical") -> torch.Tensor:
    """Return fixed pair fixture positions.

    Parameters
    ----------
    node_gap : float, default=PAIR_DEFAULT_GAP
        Center-to-center spacing between the two nodes.
    layout : str, default="vertical"
        Pair orientation. Supported values are ``"vertical"`` and
        ``"horizontal"``.

    Returns
    -------
    torch.Tensor
        Pair positions with shape ``[2, 2]``.

    Raises
    ------
    ValueError
        Raised when ``layout`` is unsupported.
    """

    half_gap = node_gap / 2.0
    if layout == "vertical":
        coords = [[0.0, half_gap], [0.0, -half_gap]]
    elif layout == "horizontal":
        coords = [[-half_gap, 0.0], [half_gap, 0.0]]
    else:
        raise ValueError(f"Unsupported pair layout: {layout}")
    return torch.tensor(coords, dtype=torch.float32)


def _fan_positions(leaf_count: int) -> torch.Tensor:
    """Return fixed hub-and-leaf positions for a requested fan size.

    Parameters
    ----------
    leaf_count : int
        Number of leaf nodes connected to the hub.

    Returns
    -------
    torch.Tensor
        Fan positions with shape ``[leaf_count + 1, 2]``.

    Raises
    ------
    ValueError
        Raised when ``leaf_count`` is smaller than one.
    """

    if leaf_count < 1:
        raise ValueError("Fan fixtures require at least one leaf node.")
    if leaf_count == 4:
        coords = [
            [0.0, 160.0],
            [-240.0, -20.0],
            [-80.0, -100.0],
            [80.0, -100.0],
            [240.0, -20.0],
        ]
        return torch.tensor(coords, dtype=torch.float32)

    leaf_xs = torch.linspace(-280.0, 280.0, steps=leaf_count, dtype=torch.float32)
    coords = [[0.0, 180.0]]
    for index, x_coord in enumerate(leaf_xs.tolist()):
        arch = 48.0 * abs((index - (leaf_count - 1) / 2.0) / max(leaf_count - 1, 1))
        coords.append([float(x_coord), -80.0 + arch])
    return torch.tensor(coords, dtype=torch.float32)


def _chain_positions(direction: str = "TB") -> torch.Tensor:
    """Return fixed chain positions for a graph direction.

    Parameters
    ----------
    direction : str, default="TB"
        Direction identifier.

    Returns
    -------
    torch.Tensor
        Chain positions with shape ``[3, 2]``.
    """

    if direction == "TB":
        coords = [(0.0, 180.0), (0.0, 20.0), (0.0, -140.0)]
    elif direction == "BT":
        coords = [(0.0, -140.0), (0.0, 20.0), (0.0, 180.0)]
    elif direction == "LR":
        coords = [(-240.0, 0.0), (0.0, 0.0), (240.0, 0.0)]
    elif direction == "RL":
        coords = [(240.0, 0.0), (0.0, 0.0), (-240.0, 0.0)]
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return torch.tensor(coords, dtype=torch.float32)


def _combo_flow_positions(direction: str = "TB") -> torch.Tensor:
    """Return fixed positions for the five-node workflow combo fixture.

    Parameters
    ----------
    direction : str, default="TB"
        Direction identifier.

    Returns
    -------
    torch.Tensor
        Workflow positions with shape ``[5, 2]``.
    """

    if direction == "TB":
        coords = [
            (0.0, 210.0),
            (-170.0, 70.0),
            (170.0, 70.0),
            (-170.0, -90.0),
            (170.0, -200.0),
        ]
    elif direction == "BT":
        coords = [
            (0.0, -210.0),
            (-170.0, -70.0),
            (170.0, -70.0),
            (-170.0, 90.0),
            (170.0, 200.0),
        ]
    elif direction == "LR":
        coords = [
            (-280.0, 0.0),
            (-120.0, 120.0),
            (-120.0, -120.0),
            (120.0, 120.0),
            (280.0, -120.0),
        ]
    elif direction == "RL":
        coords = [
            (280.0, 0.0),
            (120.0, 120.0),
            (120.0, -120.0),
            (-120.0, 120.0),
            (-280.0, -120.0),
        ]
    else:
        raise ValueError(f"Unsupported direction: {direction}")
    return torch.tensor(coords, dtype=torch.float32)


def _build_pair_fixture() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the canonical two-node pair fixture.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Pair graph and positions.
    """

    graph = DaguaGraph(direction="TB")
    graph.add_node("source", label="Source")
    graph.add_node("target", label="Target")
    graph.add_edge("source", "target", label=None)
    _configure_fixture_defaults(graph)
    return graph, _pair_positions()


def _build_single_node_fixture() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the canonical single-node fixture.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Single-node graph and positions.
    """

    graph = DaguaGraph(direction="TB")
    graph.add_node("focus", label="Target")
    _configure_fixture_defaults(graph)
    positions = torch.tensor([[0.0, 0.0]], dtype=torch.float32)
    return graph, positions


def _build_chain_fixture(direction: str = "TB") -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the canonical three-node chain fixture.

    Parameters
    ----------
    direction : str, default="TB"
        Direction identifier.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Chain graph and positions.
    """

    graph = DaguaGraph(direction=direction)
    graph.add_node("a", label="Stage A")
    graph.add_node("b", label="Stage B")
    graph.add_node("c", label="Stage C")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    _configure_fixture_defaults(graph)
    return graph, _chain_positions(direction)


def _build_fan_fixture(leaf_count: int = 4) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the canonical hub-and-leaves fan fixture.

    Parameters
    ----------
    leaf_count : int, default=4
        Number of leaf nodes connected to the hub.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Fan graph and positions.
    """

    graph = DaguaGraph(direction="TB")
    graph.add_node("hub", label="Hub")
    for index in range(leaf_count):
        leaf_id = f"leaf_{index}"
        graph.add_node(leaf_id, label=f"Leaf {index + 1}")
        graph.add_edge("hub", leaf_id)
    _configure_fixture_defaults(graph)
    return graph, _fan_positions(leaf_count)


def _build_diamond_fixture() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the canonical four-node diamond fixture.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Diamond graph and positions.
    """

    graph = DaguaGraph(direction="TB")
    graph.add_node("a", label="A")
    graph.add_node("b", label="B")
    graph.add_node("c", label="C")
    graph.add_node("d", label="D")
    graph.add_edge("a", "b")
    graph.add_edge("a", "c")
    graph.add_edge("b", "d")
    graph.add_edge("c", "d")
    _configure_fixture_defaults(graph)
    positions = torch.tensor(
        [[0.0, 180.0], [-170.0, 20.0], [170.0, 20.0], [0.0, -160.0]],
        dtype=torch.float32,
    )
    return graph, positions


def _build_crossing_fixture() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the canonical crossing-edge fixture.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Crossing graph and positions.
    """

    graph = DaguaGraph(direction="TB")
    graph.add_node("a", label="A")
    graph.add_node("b", label="B")
    graph.add_node("c", label="C")
    graph.add_node("d", label="D")
    graph.add_edge("a", "d")
    graph.add_edge("b", "c")
    _configure_fixture_defaults(graph)
    positions = torch.tensor(
        [[-160.0, 150.0], [160.0, 150.0], [-160.0, -110.0], [160.0, -110.0]],
        dtype=torch.float32,
    )
    return graph, positions


def _build_cluster_simple_fixture() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the canonical simple-cluster fixture.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Clustered graph and positions.
    """

    graph = DaguaGraph(direction="TB")
    graph.add_node("a", label="Group A")
    graph.add_node("b", label="Group B")
    graph.add_node("c", label="Group C")
    graph.add_node("d", label="Outside")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", "d")
    graph.add_cluster(
        "primary",
        ["a", "b", "c"],
        style=_base_cluster_style(),
        label="Primary cluster",
    )
    _configure_fixture_defaults(graph)
    positions = torch.tensor(
        [[-180.0, 90.0], [0.0, 20.0], [180.0, 90.0], [0.0, -160.0]],
        dtype=torch.float32,
    )
    return graph, positions


def _build_cluster_nested_fixture() -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the canonical nested-cluster fixture.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Nested-cluster graph and positions.
    """

    graph = DaguaGraph(direction="TB")
    graph.add_node("a", label="Outer A")
    graph.add_node("b", label="Inner B")
    graph.add_node("c", label="Inner C")
    graph.add_node("d", label="Outer D")
    graph.add_edge("a", "b")
    graph.add_edge("b", "c")
    graph.add_edge("c", "d")
    graph.add_cluster(
        "outer",
        ["a", "b", "c", "d"],
        style=_base_cluster_style(),
        label="Outer",
    )
    graph.add_cluster(
        "inner",
        ["b", "c"],
        style=_base_cluster_style(),
        label="Inner",
        parent="outer",
    )
    _configure_fixture_defaults(graph)
    positions = torch.tensor(
        [[-140.0, 160.0], [-60.0, 40.0], [60.0, -40.0], [140.0, -170.0]],
        dtype=torch.float32,
    )
    return graph, positions


def _build_combo_flow_fixture(direction: str = "TB") -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the canonical five-node workflow fixture used by combo cards.

    Parameters
    ----------
    direction : str, default="TB"
        Direction identifier.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Workflow graph and positions.
    """

    graph = DaguaGraph(direction=direction)
    graph.add_node("ingest", label="Ingest")
    graph.add_node("validate", label="Validate")
    graph.add_node("review", label="Review")
    graph.add_node("approve", label="Approve")
    graph.add_node("ship", label="Ship")
    graph.add_edge("ingest", "validate")
    graph.add_edge("ingest", "review")
    graph.add_edge("validate", "approve")
    graph.add_edge("review", "ship")
    _configure_fixture_defaults(graph)
    return graph, _combo_flow_positions(direction)


def _build_fixture(name: str, direction: str = "TB") -> Tuple[DaguaGraph, torch.Tensor]:
    """Build one canonical fixture by name.

    Parameters
    ----------
    name : str
        Fixture identifier.
    direction : str, default="TB"
        Direction override for directional fixtures.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Configured graph and fixed positions.
    """

    if name == "pair":
        return _build_pair_fixture()
    if name == "single_node":
        return _build_single_node_fixture()
    if name == "chain":
        return _build_chain_fixture(direction=direction)
    if name == "fan":
        return _build_fan_fixture()
    if name == "diamond":
        return _build_diamond_fixture()
    if name == "crossing":
        return _build_crossing_fixture()
    if name == "cluster_simple":
        return _build_cluster_simple_fixture()
    if name == "cluster_nested":
        return _build_cluster_nested_fixture()
    if name == "combo_flow":
        return _build_combo_flow_fixture(direction=direction)
    raise ValueError(f"Unsupported fixture: {name}")


def _node_styles(graph: DaguaGraph) -> List[NodeStyle]:
    """Return concrete node styles for a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to inspect.

    Returns
    -------
    list[NodeStyle]
        Node styles in graph order.
    """

    styles: List[NodeStyle] = []
    for style in graph.node_styles:
        if style is None:
            raise ValueError("Expected concrete node styles for every node.")
        styles.append(style)
    return styles


def _edge_styles(graph: DaguaGraph) -> List[EdgeStyle]:
    """Return concrete edge styles for a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to inspect.

    Returns
    -------
    list[EdgeStyle]
        Edge styles in edge order.
    """

    styles: List[EdgeStyle] = []
    for style in graph.edge_styles:
        if style is None:
            raise ValueError("Expected concrete edge styles for every edge.")
        styles.append(style)
    return styles


def _cluster_styles(graph: DaguaGraph) -> List[ClusterStyle]:
    """Return concrete cluster styles for a graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to inspect.

    Returns
    -------
    list[ClusterStyle]
        Cluster styles in insertion order.
    """

    return list(graph.cluster_styles.values())


def _apply_overrides(style: Any, overrides: Mapping[str, object]) -> None:
    """Apply attribute overrides to a style object.

    Parameters
    ----------
    style : Any
        Style instance to mutate.
    overrides : Mapping[str, object]
        Attribute values keyed by field name.

    Returns
    -------
    None
        The style object is mutated in place.
    """

    for field_name, value in overrides.items():
        setattr(style, field_name, copy.deepcopy(value))


def _apply_indexed_overrides(
    styles: Sequence[Any],
    overrides: object,
    style_kind: str,
) -> None:
    """Apply per-index overrides to a homogeneous style sequence.

    Parameters
    ----------
    styles : Sequence[Any]
        Concrete style objects in graph order.
    overrides : object
        Expected to be a list of mapping objects with the same length as
        ``styles``.
    style_kind : str
        Readable style-family name used in validation errors.

    Returns
    -------
    None
        The style sequence is mutated in place.

    Raises
    ------
    ValueError
        Raised when the override payload shape does not match ``styles``.
    """

    if overrides is None:
        return
    if not isinstance(overrides, list):
        raise ValueError(f"Expected {style_kind}_style_overrides to be a list.")
    if len(overrides) != len(styles):
        raise ValueError(
            f"Expected {len(styles)} {style_kind} overrides, received {len(overrides)}"
        )
    for style, style_overrides in zip(styles, overrides):
        if not isinstance(style_overrides, Mapping):
            raise ValueError(f"Expected each {style_kind} override to be a mapping.")
        _apply_overrides(style, style_overrides)


def _apply_dark_palette(graph: DaguaGraph) -> None:
    """Switch fixture colors to a dark-background palette.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to recolor.

    Returns
    -------
    None
        The graph styles are mutated in place.
    """

    for style in _node_styles(graph):
        style.fill = DARK_NODE_FILL
        style.stroke = DARK_NODE_STROKE
        style.font_color = WHITE
        style.text_outline_color = DARK_LABEL_BG
    for style in _edge_styles(graph):
        style.color = DARK_EDGE_COLOR
        style.width = max(float(style.width), 2.0)
        style.label_font_color = WHITE
        style.label_background = DARK_LABEL_BG
    for style in _cluster_styles(graph):
        style.fill = DARK_CLUSTER_FILL
        style.stroke = DARK_CLUSTER_STROKE
        style.font_color = WHITE


def _apply_label_values(
    labels: object,
    graph_labels: List[Optional[str]],
    expected_count: int,
) -> List[Optional[str]]:
    """Expand one-or-many label inputs to match a target count.

    Parameters
    ----------
    labels : object
        Either a single label string or a list of label strings.
    graph_labels : list[str | None]
        Current labels used as a fallback when no override is provided.
    expected_count : int
        Expected label count.

    Returns
    -------
    list[str | None]
        Resolved labels with ``expected_count`` entries.
    """

    if isinstance(labels, str):
        return [labels for _ in range(expected_count)]
    if isinstance(labels, list):
        if len(labels) != expected_count:
            raise ValueError(f"Expected {expected_count} labels, received {len(labels)}")
        return [str(label) if label is not None else None for label in labels]
    return graph_labels


def _apply_reference_params(
    graph: DaguaGraph,
    positions: torch.Tensor,
    params: Mapping[str, object],
    fixture: str,
) -> torch.Tensor:
    """Apply JSON-serializable parameter overrides to a fixture graph.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to mutate.
    positions : torch.Tensor
        Current fixed positions with shape ``[N, 2]``.
    params : Mapping[str, object]
        Parameter mapping carried by the card definition.
    fixture : str
        Canonical fixture name.

    Returns
    -------
    torch.Tensor
        Final positions after any direction-driven updates.
    """

    next_positions = positions
    node_overrides = params.get("node")
    if isinstance(node_overrides, Mapping):
        for style in _node_styles(graph):
            _apply_overrides(style, node_overrides)
    _apply_indexed_overrides(_node_styles(graph), params.get("node_style_overrides"), "node")

    edge_overrides = params.get("edge")
    if isinstance(edge_overrides, Mapping):
        for style in _edge_styles(graph):
            _apply_overrides(style, edge_overrides)
    _apply_indexed_overrides(_edge_styles(graph), params.get("edge_style_overrides"), "edge")

    cluster_overrides = params.get("cluster")
    if isinstance(cluster_overrides, Mapping):
        for style in _cluster_styles(graph):
            _apply_overrides(style, cluster_overrides)
            if style.stroke_dash == "dotted":
                style.stroke_width = max(style.stroke_width, 1.5)

    graph_overrides = params.get("graph")
    if isinstance(graph_overrides, Mapping):
        graph_style = graph._theme.graph_style
        if not isinstance(graph_style, GraphStyle):
            raise ValueError("Expected graph._theme.graph_style to be a GraphStyle.")
        _apply_overrides(graph_style, graph_overrides)

    if "node_labels" in params:
        graph.node_labels = _apply_label_values(
            params["node_labels"],
            graph.node_labels,
            graph.num_nodes,
        )

    varied_external_labels = params.get("varied_external_labels")
    if isinstance(varied_external_labels, list):
        for index, style in enumerate(_node_styles(graph)):
            style.external_label = str(varied_external_labels[index % len(varied_external_labels)])

    if "edge_labels" in params:
        graph.edge_labels = _apply_label_values(
            params["edge_labels"],
            graph.edge_labels,
            graph.edge_index.shape[1],
        )

    if bool(params.get("blank_node_labels")):
        graph.node_labels = ["" for _ in range(graph.num_nodes)]

    cluster_label_offset = params.get("cluster_label_offset")
    if isinstance(cluster_label_offset, list) and len(cluster_label_offset) == 2:
        offset = (float(cluster_label_offset[0]), float(cluster_label_offset[1]))
        for style in _cluster_styles(graph):
            style.label_offset = offset

    direction = params.get("direction")
    if isinstance(direction, str):
        graph.direction = direction
        if params.get("position_variant") == "chain_direction" and fixture == "chain":
            next_positions = _chain_positions(direction)
        elif params.get("position_variant") == "combo_flow_direction" and fixture == "combo_flow":
            next_positions = _combo_flow_positions(direction)
    if fixture == "pair":
        pair_layout = params.get("pair_layout")
        pair_gap = params.get("pair_gap")
        if isinstance(pair_layout, str):
            resolved_gap = (
                float(pair_gap) if isinstance(pair_gap, (float, int)) else PAIR_DEFAULT_GAP
            )
            next_positions = _pair_positions(node_gap=resolved_gap, layout=pair_layout)

    if bool(params.get("dark_background")):
        _apply_dark_palette(graph)
    if bool(params.get("hide_edges")):
        for style in _edge_styles(graph):
            style.arrow = "none"
            style.width = 0.0
            style.opacity = 0.0

    return next_positions


def _copy_params_without_keys(
    params: Mapping[str, object],
    excluded_keys: Sequence[str],
) -> Dict[str, object]:
    """Copy a parameter mapping while omitting selected top-level keys.

    Parameters
    ----------
    params : Mapping[str, object]
        Source parameter mapping.
    excluded_keys : Sequence[str]
        Keys to omit from the returned copy.

    Returns
    -------
    dict[str, object]
        Deep-copied parameter mapping without the excluded keys.
    """

    excluded = set(excluded_keys)
    return {key: copy.deepcopy(value) for key, value in params.items() if key not in excluded}


def _graph_background_color(graph: DaguaGraph) -> str:
    """Return the effective graph background color for canvas normalization.

    Parameters
    ----------
    graph : DaguaGraph
        Graph whose style should determine the canvas background.

    Returns
    -------
    str
        Matplotlib-compatible background color string.
    """

    return str(graph.graph_style.background_color)


def _scalar_default_node_overrides(item: ReferenceCardItem) -> Dict[str, object]:
    """Return per-feature overrides for the left-hand comparison node.

    Parameters
    ----------
    item : ReferenceCardItem
        Scalar comparison card metadata.

    Returns
    -------
    dict[str, object]
        Style overrides applied to the baseline comparison node.
    """

    overrides: Dict[str, object] = {}
    if item.spec.feature == "border_opacity":
        overrides.update({"stroke_width": 3.0, "stroke": DEFAULT_COMPARISON_STROKE})
    if item.spec.feature == "border_position":
        overrides.update(
            {
                "shape": "rect",
                "stroke_width": 50.0,
                "border_position": "center",
                "min_width": 80.0,
                "min_height": 60.0,
                "fill": "#FFE0B2",
                "stroke": "#E65100",
            }
        )
    if item.spec.feature == "border_count" and item.value.slug == "1_vs_2":
        overrides.update({"border_count": 1})
    if item.spec.feature == "border_count" and item.value.slug == "2_vs_3":
        overrides.update({"border_count": 2, "stroke_width": 3.0})
    if item.spec.feature == "corner_radius" and item.value.slug == "12":
        overrides.update({"shape": "roundrect", "corner_radius": 0.0})
    return overrides


def _scalar_value_node_overrides(item: ReferenceCardItem) -> Dict[str, object]:
    """Return per-feature overrides for the right-hand comparison node.

    Parameters
    ----------
    item : ReferenceCardItem
        Scalar comparison card metadata.

    Returns
    -------
    dict[str, object]
        Additional style overrides layered on the swept comparison node.
    """

    overrides: Dict[str, object] = {}
    if item.spec.feature == "border_count" and item.value.slug == "1_vs_2":
        overrides.update({"border_count": 2})
    if item.spec.feature == "corner_radius" and item.value.slug == "12":
        overrides.update({"shape": "roundrect", "corner_radius": 12.0})
    return overrides


def _scalar_comparison_labels(item: ReferenceCardItem) -> List[str]:
    """Return the node labels shown for a scalar comparison card.

    Parameters
    ----------
    item : ReferenceCardItem
        Scalar comparison card metadata.

    Returns
    -------
    list[str]
        Pair labels shown on the left and right comparison nodes.
    """

    if item.spec.feature == "border_count" and item.value.slug == "1_vs_2":
        return ["1", "2"]
    if item.spec.feature == "border_count" and item.value.slug == "2_vs_3":
        return ["2", "3"]
    if item.spec.feature == "corner_radius" and item.value.slug == "12":
        return ["0", "12"]
    if item.spec.feature == "border_position":
        return ["Center", item.value.label]
    return ["Default", item.value.label]


def _is_scalar_node_comparison_card(item: ReferenceCardItem) -> bool:
    """Return whether a reference card should render with default-vs-sweep context.

    Parameters
    ----------
    item : ReferenceCardItem
        Reference card metadata.

    Returns
    -------
    bool
        ``True`` when the card is a simple node scalar sweep that benefits from
        side-by-side default comparison context.
    """

    return (
        item.spec.target == "node"
        and item.spec.fixture == "pair"
        and item.spec.feature in SCALAR_NODE_COMPARISON_FEATURES
    )


def _apply_scalar_node_comparison_context(
    graph: DaguaGraph,
    item: ReferenceCardItem,
) -> torch.Tensor:
    """Render a scalar node sweep as default-vs-swept vertical pair nodes.

    Parameters
    ----------
    graph : DaguaGraph
        Pair fixture graph to mutate.
    item : ReferenceCardItem
        Scalar-sweep reference card metadata.

    Returns
    -------
    torch.Tensor
        Side-by-side pair positions with shape ``[2, 2]``.

    Raises
    ------
    ValueError
        Raised when the card does not carry node overrides.
    """

    node_overrides = item.value.params.get("node")
    if not isinstance(node_overrides, Mapping):
        raise ValueError("Scalar comparison cards require node overrides.")

    non_node_params = _copy_params_without_keys(
        item.value.params,
        excluded_keys=("node", "node_labels", "blank_node_labels"),
    )
    _apply_reference_params(graph, _pair_positions(), non_node_params, item.spec.fixture)
    left_style, right_style = _node_styles(graph)
    _apply_overrides(left_style, _scalar_default_node_overrides(item))
    _apply_overrides(right_style, node_overrides)
    _apply_overrides(right_style, _scalar_value_node_overrides(item))
    graph.node_labels = _scalar_comparison_labels(item)

    # The comparison should focus on node styling, not the connecting pair edge.
    for style in _edge_styles(graph):
        style.arrow = "none"
        style.width = 0.0
        style.opacity = 0.0

    # Use tighter gap for border_position so the fill-area difference
    # between center/inside/outside is immediately obvious.
    gap = 140.0 if item.spec.feature == "border_position" else PAIR_SCALAR_COMPARISON_GAP
    return _pair_positions(node_gap=gap, layout="horizontal")


def _build_reference_fixture(item: ReferenceCardItem) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build the fixture graph best suited for one reference card.

    Parameters
    ----------
    item : ReferenceCardItem
        Reference card metadata.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Configured graph and fixed positions.
    """

    if item.spec.feature == "port_style" and item.value.slug == "center":
        return _build_fan_fixture(leaf_count=6)
    return _build_fixture(item.spec.fixture)


def _apply_reference_card_tweaks(
    item: ReferenceCardItem,
    graph: DaguaGraph,
    positions: torch.Tensor,
    render_context: str = "reference",
) -> torch.Tensor:
    """Apply one-off visual tweaks that do not map cleanly to shared params.

    Parameters
    ----------
    item : ReferenceCardItem
        Reference card metadata.
    graph : DaguaGraph
        Styled graph to mutate in place.
    positions : torch.Tensor
        Current fixed positions with shape ``[N, 2]``.
    render_context : str, default="reference"
        Rendering context. ``"reference"`` keeps standalone-card framing,
        while ``"strip"`` allows tighter strip-panel framing.

    Returns
    -------
    torch.Tensor
        Updated fixed positions.
    """

    if item.card_id in DECORATIVE_FILL_CARD_IDS:
        for style in _node_styles(graph):
            style.min_height = max(float(style.min_height), DECORATIVE_FILL_CARD_MIN_HEIGHT)
            style.min_width = max(float(style.min_width), GRAPHVIZ_PARITY_MAX_NODE_WIDTH)
            # Decorative fills must use the same footprint as plain nodes; the
            # fixed-extent metric now treats fill-specific size drift as signal.
            style.padding = DECORATIVE_FILL_CARD_PADDING
    if item.card_id in NODE_SHAPE_PARITY_CARD_IDS:
        for style in _node_styles(graph):
            if style.min_width is not None:
                style.min_width = GRAPHVIZ_PARITY_MAX_NODE_WIDTH
            if style.min_height is not None:
                style.min_height = GRAPHVIZ_PARITY_MAX_NODE_HEIGHT
    if item.spec.fixture == "pair" and item.spec.category.startswith("nodes/"):
        for style in _edge_styles(graph):
            style.arrow = "normal"
            style.arrow_length = max(float(style.arrow_length), 28.0)
            style.arrow_width = max(float(style.arrow_width), 20.0)
    if item.spec.feature == "external_label" and item.value.slug == "top":
        styles = _node_styles(graph)
        if len(styles) >= 2:
            styles[1].external_label = ""
    if item.spec.feature == "curvature" and item.value.slug == "0_8":
        margin_target = (
            STRIP_CURVATURE_CARD_MARGIN if render_context == "strip" else CURVATURE_CARD_MARGIN
        )
        graph.graph_style.margin = max(float(graph.graph_style.margin), margin_target)
    # Offset the middle node for routing cards so ortho/taxi produce visible bends.
    if item.spec.feature == "routing" and item.spec.fixture == "chain":
        positions = positions.clone()
        positions[1, 0] += 120.0
    # No special edge handling for pointed shapes needed -- the renderer's
    # _adjust_port_for_shape() and ray_polygon_intersection() now handle
    # shape-aware edge endpoints for all polygon shapes.
    return positions


def _apply_arrow_demo_tweaks(
    item: ReferenceCardItem,
    graph: DaguaGraph,
    positions: torch.Tensor,
) -> torch.Tensor:
    """Amplify arrow demo cards so the arrowhead remains the focal element.

    Parameters
    ----------
    item : ReferenceCardItem
        Reference card metadata.
    graph : DaguaGraph
        Styled graph to mutate.
    positions : torch.Tensor
        Current fixed positions with shape ``[N, 2]``.

    Returns
    -------
    torch.Tensor
        Updated fixed positions.
    """

    if item.spec.category != "edges/arrows" or item.spec.fixture != "pair":
        return positions

    for style in _edge_styles(graph):
        style.width = ARROW_DEMO_EDGE_WIDTH
        style.arrow_node_fraction = ARROW_DEMO_NODE_FRACTION
    # Open/stroked arrowheads (crow, tee, bracket) need wider edges
    # to make their thin strokes visually prominent.
    if item.value.slug in ("crow", "bracket"):
        for style in _edge_styles(graph):
            style.width = ARROW_DEMO_EDGE_WIDTH * 1.8

    return _pair_positions(node_gap=PAIR_ARROW_GAP)


def _prepare_reference_render(
    item: ReferenceCardItem,
    render_context: str = "reference",
) -> Tuple[DaguaGraph, torch.Tensor]:
    """Build and tweak a reference-card graph before image rendering.

    Parameters
    ----------
    item : ReferenceCardItem
        Reference card metadata.
    render_context : str, default="reference"
        Rendering context. ``"reference"`` prepares the standalone card, while
        ``"strip"`` prepares a tighter strip-panel variant.

    Returns
    -------
    tuple[DaguaGraph, torch.Tensor]
        Prepared graph and fixed positions.
    """

    graph, positions = _build_reference_fixture(item)
    if _is_scalar_node_comparison_card(item):
        positions = _apply_scalar_node_comparison_context(graph, item)
    else:
        positions = _apply_reference_params(graph, positions, item.value.params, item.spec.fixture)
    positions = _apply_arrow_demo_tweaks(item, graph, positions)
    positions = _apply_reference_card_tweaks(item, graph, positions, render_context)
    return graph, positions


def _escape_dot(text: str) -> str:
    """Escape a string for inclusion in DOT source.

    Parameters
    ----------
    text : str
        Source text.

    Returns
    -------
    str
        DOT-safe string.
    """

    return text.replace("\\", "\\\\").replace('"', '\\"')


def _graphviz_node_attrs(
    graph: DaguaGraph,
    value: FeatureValue,
) -> Dict[str, str]:
    """Return default Graphviz node attributes for a comparison card.

    Parameters
    ----------
    graph : DaguaGraph
        Graph being rendered.
    value : FeatureValue
        Feature value that may carry comparison attributes.

    Returns
    -------
    dict[str, str]
        DOT node attributes.
    """

    attrs = {
        "style": "filled",
        "fillcolor": NODE_FILL,
        "color": NODE_STROKE,
        "fontname": "Helvetica",
        "fontsize": "18",
        "fontcolor": TEXT_COLOR,
        "penwidth": "2.0",
    }
    node_params = value.params.get("node", {})
    if isinstance(node_params, Mapping):
        if "stroke_width" in node_params:
            attrs["penwidth"] = str(node_params["stroke_width"])
        if "border_opacity" in node_params:
            opacity = min(max(float(node_params["border_opacity"]), 0.0), 1.0)
            attrs["color"] = f"{NODE_STROKE}{int(round(opacity * 255.0)):02X}"
        if "opacity" in node_params:
            opacity = min(max(float(node_params["opacity"]), 0.0), 1.0)
            attrs["fillcolor"] = f"{NODE_FILL}{int(round(opacity * 255.0)):02X}"
        if "text_align" in node_params:
            attrs["labeljust"] = {"left": "l", "center": "c", "right": "r"}.get(
                str(node_params["text_align"]),
                "c",
            )
        if "text_valign" in node_params:
            attrs["labelloc"] = {"top": "t", "center": "c", "bottom": "b"}.get(
                str(node_params["text_valign"]),
                "c",
            )
        if "external_label" in node_params:
            attrs["xlabel"] = str(node_params["external_label"])
    if value.graphviz_attrs is not None and "shape" in value.graphviz_attrs:
        attrs.update(value.graphviz_attrs)
    return attrs


def _graphviz_edge_attrs(value: FeatureValue) -> Dict[str, str]:
    """Return default Graphviz edge attributes for a comparison card.

    Parameters
    ----------
    value : FeatureValue
        Feature value that may carry comparison attributes.

    Returns
    -------
    dict[str, str]
        DOT edge attributes.
    """

    attrs = {
        "color": EDGE_COLOR,
        "penwidth": "2.2",
        "arrowsize": "1.2",
    }
    edge_params = value.params.get("edge", {})
    if isinstance(edge_params, Mapping):
        if "width" in edge_params:
            attrs["penwidth"] = str(edge_params["width"])
        if "label_position" in edge_params:
            attrs["xlabel"] = value.params.get("edge_labels", [""])[0]
        if bool(edge_params.get("taper")):
            attrs["style"] = "tapered"
    if value.graphviz_attrs is not None:
        attrs.update(value.graphviz_attrs)
    return attrs


def _dot_attr_list(attrs: Mapping[str, str]) -> str:
    """Serialize DOT attributes into a bracketed list.

    Parameters
    ----------
    attrs : Mapping[str, str]
        Attribute mapping.

    Returns
    -------
    str
        DOT attribute list.
    """

    parts = [f'{name}="{_escape_dot(value)}"' for name, value in attrs.items()]
    return ", ".join(parts)


def _graphviz_cluster_attrs(style: ClusterStyle, item: ReferenceCardItem) -> Dict[str, str]:
    """Return Graphviz cluster attributes for one comparison card.

    Parameters
    ----------
    style : ClusterStyle
        Cluster style prepared for the current card.
    item : ReferenceCardItem
        Reference card metadata carrying feature overrides.

    Returns
    -------
    dict[str, str]
        DOT graph attributes for a cluster subgraph.
    """

    attrs = {
        "color": str(style.stroke),
        "penwidth": str(style.stroke_width),
        "fontname": str(style.font_family or "Helvetica"),
        "fontsize": str(style.font_size),
        "fontcolor": str(style.font_color),
        "labelloc": "t",
    }
    cluster_params = item.value.params.get("cluster", {})
    if isinstance(cluster_params, Mapping):
        opacity = min(max(float(cluster_params.get("opacity", style.opacity)), 0.0), 1.0)
        border_opacity = min(
            max(float(cluster_params.get("border_opacity", style.border_opacity or opacity)), 0.0),
            1.0,
        )
        attrs["color"] = f"{style.stroke}{int(round(border_opacity * 255.0)):02X}"
        fill = str(cluster_params.get("fill", "") or "")
        fill_opacity = min(max(float(cluster_params.get("fill_opacity", 0.0)), 0.0), 1.0)
        if fill:
            attrs["style"] = "filled"
            attrs["fillcolor"] = f"{fill}{int(round(fill_opacity * opacity * 255.0)):02X}"
    return attrs


def _cluster_dot_lines(
    graph: DaguaGraph,
    item: ReferenceCardItem,
    parent: Optional[str] = None,
    indent: str = "  ",
) -> List[str]:
    """Build DOT subgraph lines for Dagua clusters.

    Parameters
    ----------
    graph : DaguaGraph
        Graph with cluster membership.
    item : ReferenceCardItem
        Current comparison card metadata.
    parent : str | None, default=None
        Parent cluster identifier to render.
    indent : str, default="  "
        DOT indentation string.

    Returns
    -------
    list[str]
        DOT lines for nested cluster subgraphs.
    """

    cluster_parents = getattr(graph, "cluster_parents", {}) or {}
    children = [name for name in graph.clusters if cluster_parents.get(name) == parent]
    lines: List[str] = []
    for name in children:
        style = graph.get_style_for_cluster(name)
        cluster_id = "".join(ch if ch.isalnum() or ch == "_" else "_" for ch in str(name))
        lines.append(f"{indent}subgraph cluster_{cluster_id} {{")
        attrs = _graphviz_cluster_attrs(style, item)
        attrs["label"] = str(graph.cluster_labels.get(name, name))
        lines.append(f"{indent}  graph [{_dot_attr_list(attrs)}];")
        lines.extend(_cluster_dot_lines(graph, item, parent=name, indent=f"{indent}  "))
        child_members = {
            int(member)
            for child_name, child_parent in cluster_parents.items()
            if child_parent == name
            for member in graph.leaf_cluster_members(child_name)
        }
        for member in graph.leaf_cluster_members(name):
            if int(member) in child_members:
                continue
            lines.append(f"{indent}  n{int(member)};")
        lines.append(f"{indent}}}")
    return lines


def _build_comparison_dot_source(graph: DaguaGraph, item: ReferenceCardItem) -> str:
    """Build DOT source for one Graphviz comparison card.

    Parameters
    ----------
    graph : DaguaGraph
        Styled graph prepared for the reference value.
    item : ReferenceCardItem
        Reference card metadata.

    Returns
    -------
    str
        DOT source string.
    """

    graph_attrs = {"bgcolor": "white", "rankdir": "TB", "margin": "0.3"}
    graph_params = item.value.params.get("graph", {})
    if isinstance(graph_params, Mapping):
        if "background_color" in graph_params:
            graph_attrs["bgcolor"] = str(graph_params["background_color"])
        if "margin" in graph_params:
            graph_attrs["margin"] = str(graph_params["margin"])
    if isinstance(item.value.params.get("direction"), str):
        graph_attrs["rankdir"] = str(item.value.params["direction"]).upper()

    lines = ["digraph G {"]
    lines.append(f"  graph [{_dot_attr_list(graph_attrs)}];")
    lines.append(f"  node [{_dot_attr_list(_graphviz_node_attrs(graph, item.value))}];")
    lines.append(f"  edge [{_dot_attr_list(_graphviz_edge_attrs(item.value))}];")
    for index, label in enumerate(graph.node_labels):
        node_label = "" if label is None else str(label)
        lines.append(f'  n{index} [label="{_escape_dot(node_label)}"];')
    lines.extend(_cluster_dot_lines(graph, item))
    for edge_index in range(graph.edge_index.shape[1]):
        source = int(graph.edge_index[0, edge_index].item())
        target = int(graph.edge_index[1, edge_index].item())
        lines.append(f"  n{source} -> n{target};")
    lines.append("}")
    return "\n".join(lines)


def _render_dagua_png(
    graph: DaguaGraph,
    positions: torch.Tensor,
    output_path: Path,
    size_px: Tuple[int, int],
) -> None:
    """Render a Dagua graph to a PNG file.

    Parameters
    ----------
    graph : DaguaGraph
        Graph to render.
    positions : torch.Tensor
        Fixed positions with shape ``[N, 2]``.
    output_path : Path
        Target PNG path.
    size_px : tuple[int, int]
        Requested raw canvas size in pixels.

    Returns
    -------
    None
        The PNG is written to ``output_path``.
    """

    graph.compute_node_sizes()
    bg_color = _graph_background_color(graph)
    fig, ax = render(
        graph,
        positions,
        dpi=RENDER_DPI,
        figsize=(size_px[0] / RENDER_DPI, size_px[1] / RENDER_DPI),
    )
    fig.patch.set_facecolor(bg_color)
    ax.set_facecolor(bg_color)
    if graph.node_sizes is not None and graph.num_nodes:
        pos = positions.detach().cpu()
        sizes = graph.node_sizes.detach().cpu()
        x_min = float((pos[:, 0] - sizes[:, 0] / 2.0).min())
        x_max = float((pos[:, 0] + sizes[:, 0] / 2.0).max())
        y_min = float((pos[:, 1] - sizes[:, 1] / 2.0).min())
        y_max = float((pos[:, 1] + sizes[:, 1] / 2.0).max())
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        ax.set_xlim(x_center - size_px[0] / 2.0, x_center + size_px[0] / 2.0)
        ax.set_ylim(y_center - size_px[1] / 2.0, y_center + size_px[1] / 2.0)
    fig.savefig(
        output_path,
        dpi=RENDER_DPI,
        bbox_inches=None,
        pad_inches=0.0,
        facecolor=bg_color,
        edgecolor=bg_color,
        transparent=False,
    )
    plt.close(fig)


def _render_graphviz_png(dot_source: str, output_path: Path) -> None:
    """Render DOT source to a PNG using Graphviz ``dot``.

    Parameters
    ----------
    dot_source : str
        DOT source string.
    output_path : Path
        Target PNG path.

    Returns
    -------
    None
        The PNG is written to ``output_path``.

    Raises
    ------
    RuntimeError
        Raised when Graphviz is unavailable or the subprocess fails.
    """

    if shutil.which("dot") is None:
        raise RuntimeError("Graphviz 'dot' is required to build comparison cards.")
    result = subprocess.run(
        ["dot", "-Gdpi=200", "-Tpng", "-o", str(output_path)],
        input=dot_source,
        text=True,
        capture_output=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Graphviz render failed")


def _content_crop_box(image: Image.Image) -> Optional[Tuple[int, int, int, int]]:
    """Return a padded crop box around non-white image content.

    Parameters
    ----------
    image : Image.Image
        Source image.

    Returns
    -------
    tuple[int, int, int, int] | None
        Crop box in PIL coordinates, or ``None`` when no content is found.
    """

    rgba = image.convert("RGBA")
    data = rgba.load()
    xs: List[int] = []
    ys: List[int] = []
    for y in range(rgba.height):
        for x in range(rgba.width):
            red, green, blue, alpha = data[x, y]
            if alpha > 0 and (red < 252 or green < 252 or blue < 252):
                xs.append(x)
                ys.append(y)
    if not xs or not ys:
        return None
    left = max(min(xs) - CONTENT_CROP_PADDING, 0)
    right = min(max(xs) + CONTENT_CROP_PADDING + 1, rgba.width)
    top = max(min(ys) - CONTENT_CROP_PADDING, 0)
    bottom = min(max(ys) + CONTENT_CROP_PADDING + 1, rgba.height)
    return left, top, right, bottom


def _place_render_on_canvas(
    image_path: Path,
    canvas_size: Tuple[int, int],
    inset: Tuple[int, int, int, int],
    canvas_color: str = WHITE,
    downscale_overflow: bool = False,
) -> Image.Image:
    """Place a fixed-extent render onto a fixed canvas.

    Parameters
    ----------
    image_path : Path
        Source image path.
    canvas_size : tuple[int, int]
        Final canvas size.
    inset : tuple[int, int, int, int]
        Left, top, right, and bottom insets for the content area.
    canvas_color : str, default=WHITE
        Background color used for the normalized card canvas.
    downscale_overflow : bool, default=False
        Whether to shrink native renders that exceed the target canvas. Keep
        disabled for Dagua so node-size signal is not normalized away; enable
        for competitor tools that can emit oversized native canvases.

    Returns
    -------
    Image.Image
        Fixed-size RGB image.
    """

    with Image.open(image_path) as opened:
        rgba = opened.convert("RGBA")
        available_width = canvas_size[0] - inset[0] - inset[2]
        available_height = canvas_size[1] - inset[1] - inset[3]
        if rgba.width > canvas_size[0] or rgba.height > canvas_size[1]:
            logging.warning(
                "Fixed render %s overflows canvas %sx%s with native size %sx%s",
                image_path,
                canvas_size[0],
                canvas_size[1],
                rgba.width,
                rgba.height,
            )
            if downscale_overflow:
                rgba.thumbnail(canvas_size, Image.LANCZOS)
        canvas = Image.new("RGBA", canvas_size, canvas_color)
        paste_x = inset[0] + (available_width - rgba.width) // 2
        paste_y = inset[1] + (available_height - rgba.height) // 2
        canvas.paste(rgba, (paste_x, paste_y), rgba)
    return canvas.convert("RGB")


def _load_font(size: int, bold: bool = False) -> ImageFont.ImageFont:
    """Load a local truetype font, falling back to PIL's default font.

    Parameters
    ----------
    size : int
        Requested point size.
    bold : bool, default=False
        Whether to prefer a bold face.

    Returns
    -------
    ImageFont.ImageFont
        Loaded font object.
    """

    font_names = (
        ["DejaVuSans-Bold.ttf", "Arial Bold.ttf", "Arial.ttf"]
        if bold
        else ["DejaVuSans.ttf", "Arial.ttf"]
    )
    for name in font_names:
        try:
            return ImageFont.truetype(name, size=size)
        except OSError:
            continue
    return ImageFont.load_default()


def _draw_header(
    image: Image.Image,
    title: str,
    subtitle: str,
    right_label: Optional[str] = None,
    dark: bool = False,
) -> None:
    """Draw a simple audit header on an image.

    Parameters
    ----------
    image : Image.Image
        Image to annotate in place.
    title : str
        Primary title.
    subtitle : str
        Secondary subtitle.
    right_label : str | None, optional
        Optional right-aligned label.
    dark : bool, default=False
        Whether to use the dark-background header palette.

    Returns
    -------
    None
        The image is annotated in place.
    """

    bar_color = LABEL_BAR_DARK if dark else LABEL_BAR
    title_color = LABEL_TEXT_DARK if dark else TEXT_COLOR
    secondary_color = LABEL_TEXT_DARK if dark else MUTED_TEXT_COLOR

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, HEADER_HEIGHT), fill=bar_color)
    draw.line((0, HEADER_HEIGHT, image.width, HEADER_HEIGHT), fill="#D7DDE3", width=2)
    title_font = _load_font(34, bold=True)
    subtitle_font = _load_font(22, bold=False)
    draw.text((42, 18), title, fill=title_color, font=title_font)
    draw.text((42, 54), subtitle, fill=secondary_color, font=subtitle_font)
    if right_label:
        right_font = _load_font(20, bold=True)
        width = draw.textbbox((0, 0), right_label, font=right_font)[2]
        draw.text(
            (image.width - width - 42, 30),
            right_label,
            fill=secondary_color,
            font=right_font,
        )


def _draw_strip_header(image: Image.Image, title: str, dark: bool = False) -> None:
    """Draw a compact header for a strip card.

    Parameters
    ----------
    image : Image.Image
        Strip card image to annotate in place.
    title : str
        Primary strip title.
    dark : bool, default=False
        Whether to use the dark-background header palette.

    Returns
    -------
    None
        The image is annotated in place.
    """

    bar_color = LABEL_BAR_DARK if dark else LABEL_BAR
    title_color = LABEL_TEXT_DARK if dark else TEXT_COLOR

    draw = ImageDraw.Draw(image)
    draw.rectangle((0, 0, image.width, STRIP_HEADER_HEIGHT), fill=bar_color)
    draw.line(
        (0, STRIP_HEADER_HEIGHT, image.width, STRIP_HEADER_HEIGHT),
        fill="#D7DDE3",
        width=2,
    )
    title_font = _load_font(26, bold=True)
    draw.text((28, 14), title, fill=title_color, font=title_font)


def _is_dark_background(params: Mapping[str, object]) -> bool:
    """Return whether the configured graph background is visually dark.

    Parameters
    ----------
    params : Mapping[str, object]
        Reference or combo parameter mapping.

    Returns
    -------
    bool
        ``True`` when the configured background luminance is below the audit threshold.

    Notes
    -----
    The dark-header adaptation keeps gallery chrome readable when the fixture
    itself switches to dark graph backgrounds, without requiring each card
    recipe to duplicate header color overrides.
    """
    graph_params = params.get("graph", {})
    if not isinstance(graph_params, Mapping):
        return False
    background = graph_params.get("background_color", "#FAFAFA")
    if not isinstance(background, str) or not background.startswith("#") or len(background) != 7:
        return False
    red = int(background[1:3], 16)
    green = int(background[3:5], 16)
    blue = int(background[5:7], 16)
    luminance = (0.299 * red + 0.587 * green + 0.114 * blue) / 255.0
    return luminance < 0.45


def _render_reference_canvas(
    item: ReferenceCardItem,
    size_px: Tuple[int, int],
    inset: Tuple[int, int, int, int],
    render_context: str = "reference",
) -> Image.Image:
    """Render a reference item onto a normalized fixed-size canvas.

    Parameters
    ----------
    item : ReferenceCardItem
        Reference card metadata.
    size_px : tuple[int, int]
        Final canvas size in pixels.
    inset : tuple[int, int, int, int]
        Left, top, right, and bottom content insets.
    render_context : str, default="reference"
        Rendering context used to prepare the graph.

    Returns
    -------
    Image.Image
        Rendered RGB canvas.
    """

    graph, positions = _prepare_reference_render(item, render_context=render_context)
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = Path(temp_dir) / "reference.png"
        _render_dagua_png(graph, positions, raw_path, size_px)
        return _place_render_on_canvas(
            raw_path,
            size_px,
            inset,
            canvas_color=_graph_background_color(graph),
        )


def _reference_card_inset(item: ReferenceCardItem) -> Tuple[int, int, int, int]:
    """Return the content inset for one reference card.

    Parameters
    ----------
    item : ReferenceCardItem
        Reference card metadata.

    Returns
    -------
    tuple[int, int, int, int]
        Card content inset.
    """

    return CARD_CONTENT_INSET


def _reference_card_annotation(item: ReferenceCardItem) -> Optional[str]:
    """Return an optional large footer annotation for one reference card.

    Parameters
    ----------
    item : ReferenceCardItem
        Reference card metadata.

    Returns
    -------
    str | None
        Footer annotation text when the card benefits from one.
    """

    return None


def _panel_widths(
    total_width: int,
    panel_count: int,
    divider_width: int = 0,
) -> Tuple[int, List[int]]:
    """Split a strip width into equal panel slots with optional dividers.

    Parameters
    ----------
    total_width : int
        Total width to split.
    panel_count : int
        Number of strip panels.
    divider_width : int, default=0
        Fixed width reserved between adjacent panels.

    Returns
    -------
    tuple[int, list[int]]
        Left inset plus equal per-panel widths. Any remainder is kept outside
        the panels so every panel receives the same width allocation.
    """
    if panel_count <= 0:
        raise ValueError("panel_count must be positive")
    safe_divider_width = max(int(divider_width), 0)
    total_divider_width = safe_divider_width * max(panel_count - 1, 0)
    available_panel_width = max(total_width - total_divider_width, panel_count)
    panel_width = max(available_panel_width // panel_count, 1)
    occupied_width = (panel_width * panel_count) + total_divider_width
    left_inset = max((total_width - occupied_width) // 2, 0)
    return left_inset, [panel_width] * panel_count


def _save_image(image: Image.Image, destination: Path) -> None:
    """Save an image after ensuring the destination directory exists.

    Parameters
    ----------
    image : Image.Image
        Image to save.
    destination : Path
        Target image path.

    Returns
    -------
    None
        The image is written to ``destination``.
    """

    destination.parent.mkdir(parents=True, exist_ok=True)
    image.save(destination)


def _render_reference_card(item: ReferenceCardItem, output_root: Path) -> None:
    """Render one atomic reference card and its JSON sidecar.

    Parameters
    ----------
    item : ReferenceCardItem
        Card metadata.
    output_root : Path
        Gallery audit root directory.

    Returns
    -------
    None
        The card PNG and JSON sidecar are written to disk.
    """

    destination = output_root / item.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    card = _render_reference_canvas(item, CARD_SIZE, _reference_card_inset(item))
    _draw_header(
        card,
        title=f"{item.spec.feature}: {item.value.label}",
        subtitle=f"fixture: {item.spec.fixture}",
        dark=_is_dark_background(item.value.params),
    )
    annotation = _reference_card_annotation(item)
    if annotation is not None:
        draw = ImageDraw.Draw(card)
        annotation_font = _load_font(36, bold=True)
        annotation_box = draw.textbbox((0, 0), annotation, font=annotation_font)
        annotation_width = annotation_box[2] - annotation_box[0]
        annotation_x = max((card.width - annotation_width) // 2, 42)
        draw.text(
            (annotation_x, card.height - 114),
            annotation,
            fill=TEXT_COLOR,
            font=annotation_font,
        )
    _save_image(card, destination)
    sidecar = {
        "id": item.card_id,
        "kind": "reference",
        "category": item.spec.category,
        "feature": item.spec.feature,
        "value": item.value.label,
        "value_slug": item.value.slug,
        "fixture": item.spec.fixture,
        "fields": list(item.spec.fields),
        "params": item.value.params,
        "competitor_tools": list(_card_competitor_tools(item)),
        "tier": _classify_tier(_card_competitor_tools(item)),
        "tier_c_reason": _tier_c_reason(item) if not _card_competitor_tools(item) else "",
    }
    _write_json(destination.with_suffix(".json"), sidecar)


def _render_strip_card(item: StripCardItem, output_root: Path) -> None:
    """Render one strip card and its JSON sidecar.

    Parameters
    ----------
    item : StripCardItem
        Strip card metadata.
    output_root : Path
        Gallery audit root directory.

    Returns
    -------
    None
        The strip PNG and JSON sidecar are written to disk.
    """

    destination = output_root / item.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    strip = Image.new("RGB", STRIP_CARD_SIZE, WHITE)
    _draw_strip_header(
        strip,
        title=f"{item.spec.feature}: strip",
        dark=any(_is_dark_background(member.value.params) for member in item.members),
    )

    body_height = STRIP_CARD_SIZE[1] - STRIP_HEADER_HEIGHT
    label_font = _load_font(24, bold=True)
    x_offset, panel_widths = _panel_widths(
        STRIP_CARD_SIZE[0],
        len(item.members),
        divider_width=STRIP_PANEL_DIVIDER_WIDTH,
    )
    for member_index, (member, panel_width) in enumerate(zip(item.members, panel_widths)):
        panel = _render_reference_canvas(
            member,
            (panel_width, body_height),
            STRIP_PANEL_INSET,
            render_context="strip",
        )
        draw = ImageDraw.Draw(panel)
        label_box = draw.textbbox((0, 0), member.value.label, font=label_font)
        label_width = label_box[2] - label_box[0]
        label_x = max((panel.width - label_width) // 2, 18)
        draw.text((label_x, 14), member.value.label, fill=TEXT_COLOR, font=label_font)
        draw.line(
            (0, STRIP_PANEL_LABEL_HEIGHT, panel.width, STRIP_PANEL_LABEL_HEIGHT),
            fill="#E2E8F0",
            width=2,
        )
        strip.paste(panel, (x_offset, STRIP_HEADER_HEIGHT))
        x_offset += panel_width
        if member_index < len(item.members) - 1:
            divider = Image.new("RGB", (STRIP_PANEL_DIVIDER_WIDTH, body_height), "#D7DDE3")
            strip.paste(divider, (x_offset, STRIP_HEADER_HEIGHT))
            x_offset += STRIP_PANEL_DIVIDER_WIDTH

    _save_image(strip, destination)
    sidecar = {
        "id": item.card_id,
        "kind": "reference_strip",
        "category": item.spec.category,
        "feature": item.spec.feature,
        "fixture": item.spec.fixture,
        "fields": list(item.spec.fields),
        "member_card_ids": [member.card_id for member in item.members],
        "values": [member.value.label for member in item.members],
        "value_slugs": [member.value.slug for member in item.members],
        "competitor_tools": list(item.spec.competitor_tools),
        "tier": item.spec.tier,
    }
    _write_json(destination.with_suffix(".json"), sidecar)


def _render_comparison_card(item: ReferenceCardItem, output_root: Path) -> None:
    """Render one Dagua-vs-Graphviz comparison card.

    Parameters
    ----------
    item : ReferenceCardItem
        Reference card metadata with a comparison path.
    output_root : Path
        Gallery audit root directory.

    Returns
    -------
    None
        The comparison PNG is written to disk.
    """

    if item.comparison_relative_path is None:
        return
    graph, positions = _prepare_reference_render(item)
    destination = output_root / item.comparison_relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        dagua_raw = Path(temp_dir) / "dagua.png"
        graphviz_raw = Path(temp_dir) / "graphviz.png"
        _render_dagua_png(graph, positions, dagua_raw, PANEL_SIZE)
        _render_graphviz_png(_build_comparison_dot_source(graph, item), graphviz_raw)
        dagua_panel = _place_render_on_canvas(
            dagua_raw,
            PANEL_SIZE,
            PANEL_CONTENT_INSET,
            canvas_color=_graph_background_color(graph),
        )
        try:
            graphviz_panel = _place_render_on_canvas(
                graphviz_raw,
                PANEL_SIZE,
                PANEL_CONTENT_INSET,
                downscale_overflow=True,
            )
        except Exception as exc:
            logging.warning("Skipping Graphviz comparison for %s: %s", item.card_id, exc)
            destination.unlink(missing_ok=True)
            return
    canvas = Image.new("RGB", COMPARISON_SIZE, WHITE)
    canvas.paste(dagua_panel, (0, 0))
    canvas.paste(graphviz_panel, (PANEL_SIZE[0], 0))
    _draw_header(
        canvas,
        title=f"{item.spec.feature}: {item.value.label}",
        subtitle="dagua | Graphviz dot",
        dark=_is_dark_background(item.value.params),
    )
    draw = ImageDraw.Draw(canvas)
    draw.line((PANEL_SIZE[0], 0, PANEL_SIZE[0], COMPARISON_SIZE[1]), fill="#D7DDE3", width=2)
    label_font = _load_font(22, bold=True)
    draw.text((42, 96), "dagua", fill=TEXT_COLOR, font=label_font)
    draw.text((PANEL_SIZE[0] + 42, 96), "Graphviz dot", fill=TEXT_COLOR, font=label_font)
    _save_image(canvas, destination)


def _write_json(path: Path, data: Mapping[str, object]) -> None:
    """Write JSON with stable formatting.

    Parameters
    ----------
    path : Path
        Target JSON path.
    data : Mapping[str, object]
        JSON-serializable mapping to write.

    Returns
    -------
    None
        The JSON file is written to disk.
    """

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{json.dumps(data, indent=2, sort_keys=True)}\n", encoding="utf-8")


def _reference_filename_stem(item: ReferenceCardItem) -> str:
    """Return the filename stem for a reference card.

    Parameters
    ----------
    item : ReferenceCardItem
        Reference card metadata.

    Returns
    -------
    str
        Filename stem without an extension.
    """

    if item.spec.filename_prefix:
        return f"{item.spec.filename_prefix}_{item.value.slug}"
    return item.value.slug


def _reference_card_id(spec: AtomicCardSpec, value: FeatureValue) -> str:
    """Return the stable index identifier for a reference card.

    Parameters
    ----------
    spec : AtomicCardSpec
        Parent feature specification.
    value : FeatureValue
        Concrete feature value.

    Returns
    -------
    str
        Stable card identifier.
    """

    parts = [segment.replace("-", "_") for segment in spec.category.split("/")]
    stem = (
        value.slug.replace("-", "_")
        if not spec.filename_prefix
        else f"{spec.filename_prefix}_{value.slug}".replace("-", "_")
    )
    parts.append(stem)
    return "_".join(parts)


def _strip_card_id(spec: AtomicCardSpec) -> str:
    """Return the stable index identifier for a strip card.

    Parameters
    ----------
    spec : AtomicCardSpec
        Parent feature specification.

    Returns
    -------
    str
        Stable strip card identifier.
    """

    parts = [segment.replace("-", "_") for segment in spec.category.split("/")]
    parts.append(f"strip_{spec.feature}".replace("-", "_"))
    return "_".join(parts)


def _tuple_from_first_supported(candidates: Sequence[str]) -> Tuple[str, ...]:
    """Return candidates as an immutable preference tuple.

    Parameters
    ----------
    candidates : Sequence[str]
        Candidate competitor names in preference order.

    Returns
    -------
    tuple[str, ...]
        Candidate tuple with empty strings removed.
    """

    return tuple(candidate for candidate in candidates if candidate)


def _atomic_value_competitor_tools(spec: AtomicCardSpec, value: FeatureValue) -> Tuple[str, ...]:
    """Return value-specific competitor tools for an atomic card.

    Parameters
    ----------
    spec : AtomicCardSpec
        Atomic feature family.
    value : FeatureValue
        Concrete feature value.

    Returns
    -------
    tuple[str, ...]
        Ordered competitor preference.
    """

    node_params = value.params.get("node", {})
    edge_params = value.params.get("edge", {})
    if not isinstance(node_params, Mapping):
        node_params = {}
    if not isinstance(edge_params, Mapping):
        edge_params = {}

    if spec.category == "nodes/shapes" and spec.feature == "shape":
        shape = str(node_params.get("shape", value.slug))
        if shape == "double_circle":
            return ("graphviz", "mermaid")
        if shape in {"cloud", "document", "stadium"}:
            return ("mermaid",)
        if shape in GRAPHVIZ_NODE_SHAPES:
            return ("graphviz",)
        return ()
    if spec.category == "edges/arrows":
        arrow = str(edge_params.get("arrow", value.slug))
        if arrow in GRAPHVIZ_ARROWS:
            return ("graphviz",)
        if arrow in CYTOSCAPE_ARROWS:
            return ("cytoscape",)
        return ()
    if spec.category == "edges/routing":
        routing = str(edge_params.get("routing", value.slug))
        if routing in {"bezier", "straight", "ortho"}:
            return ("graphviz",)
        if routing in {"taxi", "round-taxi"}:
            return ("cytoscape",)
        return ()
    if spec.category == "edges/styles" and spec.feature == "style":
        return ("graphviz",)
    if spec.category == "edges/styles" and spec.feature == "width":
        return ("graphviz", "mermaid")
    if spec.feature in {"text_outline", "text_background"}:
        return ()
    if spec.feature == "shadow":
        # Phase-7 smoke explicitly exercises shadow as a Tier-C heuristic path.
        return ()
    return spec.competitor_tools


def _settings_competitor_tools(settings: Mapping[str, object]) -> Tuple[str, ...]:
    """Return the best competitor tools for combo/evil settings.

    Parameters
    ----------
    settings : Mapping[str, object]
        Cosmetic settings dictionary.

    Returns
    -------
    tuple[str, ...]
        Ordered competitor preference.
    """

    tools_by_feature: List[Tuple[str, ...]] = []
    node_settings = settings.get("node", {})
    edge_settings = settings.get("edge", {})
    if isinstance(node_settings, Mapping):
        for nested_name, nested_value in node_settings.items():
            if nested_name in {"gradient", "fill_pattern"} and bool(nested_value):
                tools_by_feature.append(("graphviz", "cytoscape"))
            elif nested_name in {"border_count", "border_position"} and bool(nested_value):
                tools_by_feature.append(("cytoscape",))
            elif nested_name == "shadow" and bool(nested_value):
                tools_by_feature.append(())
    if isinstance(edge_settings, Mapping):
        for nested_name, nested_value in edge_settings.items():
            if nested_name == "routing" and bool(nested_value):
                tools_by_feature.append(
                    ("cytoscape",) if nested_value in {"taxi", "round-taxi"} else ("graphviz",)
                )
            elif nested_name in {"line_cap", "line_join", "color_gradient"} and bool(nested_value):
                tools_by_feature.append(("cytoscape",))
    for name, value in settings.items():
        if name in COMBO_INTERNAL_FIELDS or not bool(value):
            continue
        if name == "cluster":
            tools_by_feature.append(("graphviz",))
        elif name == "edge_style":
            tools_by_feature.append(("graphviz",))
        elif name == "routing":
            tools_by_feature.append(
                ("cytoscape",) if value in {"taxi", "round-taxi"} else ("graphviz",)
            )
        elif name == "shape":
            shape = str(value)
            if shape in {"cloud", "document", "stadium"}:
                tools_by_feature.append(("mermaid",))
            elif shape in GRAPHVIZ_NODE_SHAPES:
                tools_by_feature.append(("graphviz",))
            else:
                tools_by_feature.append(())
        elif name in {"gradient", "fill_pattern"}:
            tools_by_feature.append(("graphviz", "cytoscape"))
        elif name in {"border_count", "border_position", "line_cap", "line_join", "color_gradient"}:
            tools_by_feature.append(("cytoscape",))
        elif name in {"font_weight", "font_style", "font_size", "font_family", "font_color"}:
            tools_by_feature.append(("graphviz",))
        elif name in {"text_outline", "text_background", "shadow", "bevel"}:
            tools_by_feature.append(())

    if not tools_by_feature:
        return ()
    coverage: Dict[str, int] = {}
    for feature_tools in tools_by_feature:
        for tool in feature_tools:
            coverage[tool] = coverage.get(tool, 0) + 1
    if not coverage:
        return ()
    preference = {"graphviz": 0, "cytoscape": 1, "mermaid": 2, "d3": 3, "gephi": 4}
    max_coverage = max(coverage.values())
    ordered = sorted(
        (tool for tool, count in coverage.items() if count == max_coverage),
        key=lambda tool: preference.get(tool, 99),
    )
    return _tuple_from_first_supported(ordered)


def _card_competitor_tools(card: object) -> Tuple[str, ...]:
    """Return competitor tools for a resolved card item or spec.

    Parameters
    ----------
    card : object
        Reference, combo, evil item, or spec.

    Returns
    -------
    tuple[str, ...]
        Ordered competitor preference.
    """

    if isinstance(card, ReferenceCardItem):
        return _atomic_value_competitor_tools(card.spec, card.value)
    if isinstance(card, AtomicCardSpec):
        return card.competitor_tools
    if isinstance(card, ComboCardItem):
        return _settings_competitor_tools(card.spec.settings) or card.spec.competitor_tools
    if isinstance(card, ComboCardSpec):
        return _settings_competitor_tools(card.settings) or card.competitor_tools
    if isinstance(card, EvilCardItem):
        return _settings_competitor_tools(card.spec.settings) or card.spec.competitor_tools
    if isinstance(card, EvilCardSpec):
        return _settings_competitor_tools(card.settings) or card.competitor_tools
    return ()


def _tier_c_reason(card: object) -> str:
    """Return a one-line reason for Tier C cards.

    Parameters
    ----------
    card : object
        Card item or spec.

    Returns
    -------
    str
        Human-readable reason.
    """

    if isinstance(card, ReferenceCardItem):
        if card.spec.feature in {"text_outline", "text_background"}:
            return "render layer feature with no automated competitor"
        if card.spec.feature == "shadow":
            return "no competitor"
        return "feature dagua-original or not explicitly mapped"
    return "mixed or dagua-original features with no single automated competitor"


def build_reference_items() -> Tuple[ReferenceCardItem, ...]:
    """Build the resolved reference card inventory.

    Returns
    -------
    tuple[ReferenceCardItem, ...]
        Ordered reference card items.
    """

    items: List[ReferenceCardItem] = []
    for spec in build_reference_specs():
        for value in spec.values:
            stem = (
                value.slug if not spec.filename_prefix else f"{spec.filename_prefix}_{value.slug}"
            )
            relative_path = f"cards/reference/{spec.category}/{stem}.png"
            comparison_path = None
            probe_item = ReferenceCardItem(
                card_id=_reference_card_id(spec, value),
                spec=spec,
                value=value,
                relative_path=relative_path,
                comparison_relative_path=None,
            )
            if "graphviz" in _card_competitor_tools(probe_item):
                comparison_path = f"cards/comparisons/{spec.category}/{value.slug}_vs_graphviz.png"
            items.append(
                ReferenceCardItem(
                    card_id=probe_item.card_id,
                    spec=spec,
                    value=value,
                    relative_path=relative_path,
                    comparison_relative_path=comparison_path,
                )
            )
    return tuple(items)


def build_strip_items(reference_items: Sequence[ReferenceCardItem]) -> Tuple[StripCardItem, ...]:
    """Build the resolved strip-card inventory from atomic reference items.

    Parameters
    ----------
    reference_items : Sequence[ReferenceCardItem]
        Reference items selected for the current run.

    Returns
    -------
    tuple[StripCardItem, ...]
        Ordered strip-card items.
    """

    grouped: Dict[Tuple[str, str], List[ReferenceCardItem]] = {}
    for item in reference_items:
        grouped.setdefault((item.spec.category, item.spec.feature), []).append(item)

    strip_items: List[StripCardItem] = []
    for spec in build_reference_specs():
        feature_key = (spec.category, spec.feature)
        if feature_key not in STRIP_REFERENCE_FEATURES:
            continue
        members = grouped.get(feature_key, [])
        if len(members) != len(spec.values):
            continue
        strip_items.append(
            StripCardItem(
                card_id=_strip_card_id(spec),
                spec=spec,
                members=tuple(members),
                relative_path=f"cards/reference/{spec.category}/strip_{spec.feature}.png",
            )
        )
    return tuple(strip_items)


def _gallery_extension_combo_specs() -> Tuple[ComboCardSpec, ...]:
    """Return gallery-only combo specs not sourced from the cosmetic album.

    Returns
    -------
    tuple[ComboCardSpec, ...]
        Additional combo specs appended after imported album cases.
    """

    return (
        ComboCardSpec(
            case_id="combo_arrow_gradient",
            combo_kind="2way",
            title="Arrow + Linear Gradient",
            settings={"shape": "arrow", "gradient": "linear"},
        ),
        ComboCardSpec(
            case_id="combo_arrow_shadow",
            combo_kind="2way",
            title="Arrow + Shadow",
            settings={"shape": "arrow", "shadow": True},
        ),
        ComboCardSpec(
            case_id="combo_bevel_gradient",
            combo_kind="2way",
            title="Bevel + Linear Gradient",
            settings={"bevel": True, "gradient": "linear"},
        ),
        ComboCardSpec(
            case_id="combo_bevel_shadow",
            combo_kind="2way",
            title="Bevel + Shadow",
            settings={"bevel": True, "shadow": True},
        ),
        ComboCardSpec(
            case_id="combo_bridge_thick_edge",
            combo_kind="2way",
            title="Bridge Crossing + Thick Edge",
            settings={
                "crossing_style": "bridge",
                "width": 3.0,
                "preserve_crossing_width": True,
            },
        ),
        ComboCardSpec(
            case_id="combo_per_corner_gradient",
            combo_kind="2way",
            title="Per-Corner Radius + Gradient",
            settings={"corner_radius": (0.0, 12.0, 12.0, 0.0), "gradient": "linear"},
        ),
        ComboCardSpec(
            case_id="combo_port_circle_dashed",
            combo_kind="2way",
            title="Port Indicator Circle + Dashed Edge",
            settings={"port_indicator": "circle", "edge_style": "dashed"},
        ),
        ComboCardSpec(
            case_id="combo_arrow_bevel",
            combo_kind="2way",
            title="Arrow + Bevel",
            settings={"shape": "arrow", "bevel": True},
        ),
        ComboCardSpec(
            case_id="combo_arrow_bevel_gradient",
            combo_kind="3way",
            title="Arrow + Bevel + Gradient",
            settings={"shape": "arrow", "bevel": True, "gradient": "linear"},
        ),
        ComboCardSpec(
            case_id="combo_bevel_shadow_gradient",
            combo_kind="3way",
            title="Bevel + Shadow + Gradient",
            settings={"bevel": True, "shadow": True, "gradient": "linear"},
        ),
        ComboCardSpec(
            case_id="combo_bridge_taper_gradient_edge",
            combo_kind="3way",
            title="Bridge Crossing + Taper + Edge Gradient",
            settings={
                "crossing_style": "bridge",
                "taper": True,
                "color_gradient": "source_to_target",
            },
        ),
    )


def build_combo_specs() -> Tuple[ComboCardSpec, ...]:
    """Import combo definitions and append gallery-local combo extensions.

    Returns
    -------
    tuple[ComboCardSpec, ...]
        Ordered combo card specs for 2-way through 5-way cases.
    """

    combo_specs: List[ComboCardSpec] = []
    for case in build_case_catalog():
        if case.category not in {"combo_2way", "combo_3way", "combo_4way", "combo_5way"}:
            continue
        combo_specs.append(
            ComboCardSpec(
                case_id=case.case_id,
                combo_kind=case.category.removeprefix("combo_"),
                title=case.title.removeprefix("Combo: "),
                settings=dict(case.settings),
                competitor_tools=_settings_competitor_tools(case.settings),
            )
        )
    seen_case_ids = {spec.case_id for spec in combo_specs}
    for spec in _gallery_extension_combo_specs():
        if spec.case_id not in seen_case_ids:
            combo_specs.append(
                ComboCardSpec(
                    case_id=spec.case_id,
                    combo_kind=spec.combo_kind,
                    title=spec.title,
                    settings=spec.settings,
                    competitor_tools=_settings_competitor_tools(spec.settings),
                )
            )
    return tuple(combo_specs)


def build_combo_items() -> Tuple[ComboCardItem, ...]:
    """Build the resolved combo card inventory.

    Returns
    -------
    tuple[ComboCardItem, ...]
        Ordered combo card items.
    """

    items: List[ComboCardItem] = []
    for spec in build_combo_specs():
        stem = spec.case_id.removeprefix("combo_")
        items.append(
            ComboCardItem(
                card_id=spec.case_id,
                spec=spec,
                relative_path=f"cards/combos/{spec.combo_kind}/{stem}.png",
            )
        )
    return tuple(items)


def _build_local_evil_spec(
    case_id: str,
    title: str,
    fixture: str,
    params: Mapping[str, object],
) -> EvilCardSpec:
    """Build one gallery-local evil spec from a canonical fixture and params.

    Parameters
    ----------
    case_id : str
        Stable evil-case identifier.
    title : str
        Human-readable card title.
    fixture : str
        Canonical fixture name understood by ``_build_fixture``.
    params : Mapping[str, object]
        Reference-style parameter mapping applied to the fixture.

    Returns
    -------
    EvilCardSpec
        Fully prepared evil card definition.
    """

    direction = str(params.get("direction", "TB"))
    graph, positions = _build_fixture(fixture, direction=direction)
    positions = _apply_reference_params(graph, positions, params, fixture)
    return EvilCardSpec(
        case_id=case_id,
        title=title,
        settings=dict(params),
        graph=graph,
        positions=positions,
    )


def _gallery_extension_evil_specs() -> Tuple[EvilCardSpec, ...]:
    """Return gallery-only evil specs for the newly added parity features.

    Returns
    -------
    tuple[EvilCardSpec, ...]
        Additional evil specs appended after imported album cases.
    """

    return (
        _build_local_evil_spec(
            case_id="evil_arrow_bevel_gradient_shadow",
            title="Evil Combo: Arrow + Bevel + Gradient + Shadow",
            fixture="combo_flow",
            params={
                "node": {
                    "shape": "arrow",
                    "bevel": True,
                    "bevel_intensity": 0.35,
                    "gradient": "linear",
                    "fill": GRADIENT_FILL,
                    "gradient_color": GRADIENT_COLOR,
                    "shadow": True,
                    "shadow_offset": (7.0, -7.0),
                    "shadow_color": "#00000040",
                    "shadow_blur": 5.0,
                    "min_width": 120.0,
                    "min_height": 66.0,
                    "font_color": "#FFFFFF",
                },
                "node_labels": ["In", "Val", "Rev", "OK", "Out"],
            },
        ),
        _build_local_evil_spec(
            case_id="evil_per_corner_bevel_striped",
            title="Evil Combo: Per-Corner + Bevel + Striped Fill",
            fixture="combo_flow",
            params={
                "node": {
                    "shape": "roundrect",
                    "corner_radius": (0.0, 14.0, 14.0, 0.0),
                    "bevel": True,
                    "bevel_intensity": 0.35,
                    "fill_pattern": "striped",
                    "fill_pattern_colors": ["#90CAF9", "#FFAB91"],
                    "fill_pattern_angle": 30.0,
                    "text_background": "#FFFFFF",
                    "text_background_opacity": 0.92,
                    "text_background_padding": (6.0, 3.0),
                    "text_background_corner_radius": 4.0,
                    "min_width": 126.0,
                    "min_height": 70.0,
                },
                "node_labels": ["A", "B", "C", "D", "E"],
            },
        ),
        _build_local_evil_spec(
            case_id="evil_all_new_features",
            title="Evil Combo: All New Features",
            fixture="crossing",
            params={
                "node": {"bevel": True, "bevel_intensity": 0.35},
                "node_style_overrides": [
                    {
                        "shape": "arrow",
                        "fill": "#82B1FF",
                        "stroke": "#4E7CC4",
                        "font_color": "#FFFFFF",
                        "min_width": 138.0,
                        "min_height": 72.0,
                    },
                    {
                        "shape": "arrow",
                        "fill": "#82B1FF",
                        "stroke": "#4E7CC4",
                        "font_color": "#FFFFFF",
                        "min_width": 168.0,
                        "min_height": 92.0,
                    },
                    {
                        "shape": "roundrect",
                        "scale_corner_radius": True,
                        "corner_radius": (0.0, 0.18, 0.18, 0.0),
                        "min_width": 108.0,
                        "min_height": 56.0,
                    },
                    {
                        "shape": "roundrect",
                        "scale_corner_radius": True,
                        "corner_radius": (0.0, 0.18, 0.18, 0.0),
                        "min_width": 196.0,
                        "min_height": 108.0,
                    },
                ],
                "edge": {
                    "crossing_style": "bridge",
                    "crossing_size": 12.0,
                    "width": 3.0,
                    "port_indicator": "circle",
                    "port_indicator_size": 4.0,
                },
                "node_labels": ["A", "B", "C", "D"],
            },
        ),
        _build_local_evil_spec(
            case_id="evil_taxi_gradient_multiborder",
            title="Evil Combo: Taxi + Gradient + Multi-Border",
            fixture="combo_flow",
            params={
                "node": {
                    "shape": "roundrect",
                    "gradient": "linear",
                    "fill": GRADIENT_FILL,
                    "gradient_color": GRADIENT_COLOR,
                    "border_count": 3,
                    "border_position": "outside",
                    "stroke_width": 2.5,
                    "min_width": 124.0,
                    "min_height": 68.0,
                    "font_color": "#FFFFFF",
                },
                "edge": {
                    "routing": "taxi",
                    "width": 2.4,
                },
                "node_labels": ["In", "Val", "Rev", "OK", "Out"],
            },
        ),
    )


def build_evil_specs() -> Tuple[EvilCardSpec, ...]:
    """Import evil case definitions and append gallery-local stress cases.

    Returns
    -------
    tuple[EvilCardSpec, ...]
        Ordered evil-card specs copied from the album case catalog.
    """

    specs: List[EvilCardSpec] = []
    for case in build_case_catalog():
        if case.category != "evil_combos":
            continue
        specs.append(
            EvilCardSpec(
                case_id=case.case_id,
                title=case.title,
                settings=dict(case.settings),
                graph=case.graph,
                positions=case.positions,
                competitor_tools=_settings_competitor_tools(case.settings),
            )
        )
    seen_case_ids = {spec.case_id for spec in specs}
    for spec in _gallery_extension_evil_specs():
        if spec.case_id not in seen_case_ids:
            specs.append(
                EvilCardSpec(
                    case_id=spec.case_id,
                    title=spec.title,
                    settings=spec.settings,
                    graph=spec.graph,
                    positions=spec.positions,
                    competitor_tools=_settings_competitor_tools(spec.settings),
                )
            )
    return tuple(specs)


def build_evil_items() -> Tuple[EvilCardItem, ...]:
    """Build the resolved evil card inventory.

    Returns
    -------
    tuple[EvilCardItem, ...]
        Ordered evil card items.
    """

    items: List[EvilCardItem] = []
    for spec in build_evil_specs():
        stem = spec.case_id
        items.append(
            EvilCardItem(
                card_id=spec.case_id,
                spec=spec,
                relative_path=f"cards/evil/{stem}.png",
            )
        )
    return tuple(items)


def _choose_combo_fixture(settings: Mapping[str, object]) -> str:
    """Choose the best canonical fixture for a combo case.

    Parameters
    ----------
    settings : Mapping[str, object]
        Imported combo settings.

    Returns
    -------
    str
        Canonical fixture name.
    """

    if bool(settings.get("cluster")):
        return "cluster_simple"
    if "crossing_style" in settings:
        return "crossing"
    return "combo_flow"


def _combo_params(settings: Mapping[str, object], fixture: str) -> Dict[str, object]:
    """Translate imported combo settings into reference-style params.

    Parameters
    ----------
    settings : Mapping[str, object]
        Imported combo settings.
    fixture : str
        Selected fixture name.

    Returns
    -------
    dict[str, object]
        Parameter mapping understood by ``_apply_reference_params``.
    """

    params: Dict[str, object] = {
        "node": {},
        "edge": {},
        "graph": {},
    }
    node_fields = _style_field_names(NodeStyle)
    edge_fields = _style_field_names(EdgeStyle)
    for name, value in settings.items():
        if name in COMBO_INTERNAL_FIELDS or name == "cluster":
            continue
        if name == "external_label_varied":
            if bool(value):
                params["varied_external_labels"] = list(VARIED_EXTERNAL_LABELS)
            continue
        if name == "edge_style":
            edge_block = params["edge"]
            if not isinstance(edge_block, dict):
                raise ValueError("Expected edge params to be a dictionary.")
            edge_block["style"] = value
            continue
        if name == "direction":
            params["direction"] = value
            params["position_variant"] = "combo_flow_direction"
            graph_block = params["graph"]
            if not isinstance(graph_block, dict):
                raise ValueError("Expected graph params to be a dictionary.")
            graph_block["margin"] = 32.0 if str(value) in {"LR", "RL"} else 20.0
            continue
        if name in node_fields:
            node_block = params["node"]
            if not isinstance(node_block, dict):
                raise ValueError("Expected node params to be a dictionary.")
            node_block[name] = value
            continue
        if name in edge_fields:
            edge_block = params["edge"]
            if not isinstance(edge_block, dict):
                raise ValueError("Expected edge params to be a dictionary.")
            edge_block[name] = value

    node_block = params["node"]
    edge_block = params["edge"]
    if not isinstance(node_block, dict) or not isinstance(edge_block, dict):
        raise ValueError("Expected node and edge params to be dictionaries.")

    if str(node_block.get("gradient", "none")) != "none":
        node_block.setdefault("fill", GRADIENT_FILL)
        node_block.setdefault("gradient_color", GRADIENT_COLOR)
    if bool(node_block.get("shadow")):
        node_block.setdefault("shadow_offset", (7.0, -7.0))
        node_block.setdefault("shadow_color", "#00000040")
        node_block.setdefault("shadow_blur", 5.0)
    if str(node_block.get("fill_pattern", "solid")) == "striped":
        # Use softer stripe colors than the primary palette so the
        # pattern is visible without overwhelming the label text.
        node_block.setdefault("fill_pattern_colors", ["#90CAF9", "#FFAB91"])
        node_block.setdefault("fill_pattern_angle", 30.0)
    if str(node_block.get("fill_pattern", "solid")) == "hatched":
        node_block.setdefault("fill_pattern_colors", ["#F7FAFC", "#4A5568"])
    if str(node_block.get("fill_pattern", "solid")) == "pie":
        # Pie charts need enough space to show slices clearly.
        node_block.setdefault("min_width", 120.0)
        node_block.setdefault("min_height", 80.0)
    if "opacity" in node_block:
        node_block.setdefault("border_opacity", float(node_block["opacity"]))
    if str(edge_block.get("color_gradient", "none")) == "source_to_target":
        edge_block.setdefault("color", GRADIENT_FILL)
        # Use a warm red-orange that stays visible on white backgrounds
        # (the original #FF9800 fades to near-invisible olive at low opacity).
        edge_block.setdefault("color_gradient_end", "#E53935")
    if bool(edge_block.get("taper")):
        edge_block.setdefault("taper_width_start", 4.5)
        edge_block.setdefault("taper_width_end", 1.0)
    if "crossing_style" in edge_block:
        edge_block.setdefault("crossing_size", 20.0)
        requested_width = float(edge_block.get("width", settings.get("width", 4.0)))
        if bool(settings.get("preserve_crossing_width")):
            edge_block["width"] = requested_width
        else:
            # Crossing demos need slightly heavier strokes so the jump treatment
            # remains visible after the audit card is downscaled.
            edge_block["width"] = max(requested_width, 4.0)
    if "width" in edge_block:
        edge_block["width"] = float(edge_block["width"])
    if "text_background" in node_block and node_block["text_background"]:
        node_block.setdefault("text_background_opacity", 0.9)
    if str(node_block.get("text_wrap", "none")) == "wrap":
        node_block.setdefault("text_max_width", 88.0)
        # Rectangles accommodate wrapped multi-line text better than ellipses.
        node_block.setdefault("shape", "roundrect")
        node_block.setdefault("min_height", 72.0)
        params["node_labels"] = [
            "Submit intake request",
            "Validate field mapping",
            "Review change note",
            "Approve with policy",
            "Ship signed result",
        ]
    if fixture == "cluster_simple":
        cluster_block = {
            "fill": CLUSTER_FILL,
            "opacity": 0.6,
        }
        params["cluster"] = cluster_block
        # Cluster fixtures need more margin to fit the cluster box and
        # the outside node without cramming content.
        graph_block = params.get("graph")
        if isinstance(graph_block, dict):
            graph_block.setdefault("margin", 30.0)
    # Corner radius requires a rectangular base shape to be visible.
    if "corner_radius" in node_block and node_block.get("shape") not in (
        "rect",
        "roundrect",
    ):
        node_block.setdefault("shape", "roundrect")
    # Star shapes need short labels to avoid text clipping.
    if str(node_block.get("shape", "")) == "star":
        params.setdefault(
            "node_labels",
            ["In", "Val", "Rev", "OK", "Out"],
        )
    # Cloud shapes have irregular boundaries -- use short labels.
    if str(node_block.get("shape", "")) == "cloud":
        params.setdefault(
            "node_labels",
            ["Svc", "API", "Hub", "Log", "Out"],
        )
    # Box3D visible face is smaller than full bounds -- use short labels.
    if str(node_block.get("shape", "")) == "box3d":
        params.setdefault(
            "node_labels",
            ["DB", "Srv", "App", "Web", "Out"],
        )
    # External labels need a reasonable offset from the node edge.
    if node_block.get("external_label") or "varied_external_labels" in params:
        node_block.setdefault("external_label_font_size", 10.0)
        node_block.setdefault("external_label_offset", 8.0)
    # Text outlines need a visible color and width to show clearly.
    if bool(node_block.get("text_outline")):
        node_block.setdefault("text_outline_color", "#FFFFFF")
        node_block.setdefault("text_outline_width", 2.0)
    # Increase taper range so the width change is clearly visible.
    if bool(edge_block.get("taper")):
        edge_block.setdefault("taper_width_start", 4.0)
        edge_block.setdefault("taper_width_end", 0.4)
    return params


def _render_combo_card(item: ComboCardItem, output_root: Path) -> None:
    """Render one combo card and its JSON sidecar.

    Parameters
    ----------
    item : ComboCardItem
        Combo card metadata.
    output_root : Path
        Gallery audit root directory.

    Returns
    -------
    None
        The combo PNG and JSON sidecar are written to disk.
    """

    fixture = _choose_combo_fixture(item.spec.settings)
    direction = str(item.spec.settings.get("direction", "TB"))
    graph, positions = _build_fixture(fixture, direction=direction)
    params = _combo_params(item.spec.settings, fixture)
    positions = _apply_reference_params(graph, positions, params, fixture)
    destination = output_root / item.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = Path(temp_dir) / "combo.png"
        _render_dagua_png(graph, positions, raw_path, CARD_SIZE)
        card = _place_render_on_canvas(raw_path, CARD_SIZE, CARD_CONTENT_INSET)
    features = ", ".join(
        name.replace("_", " ")
        for name in item.spec.settings
        if name not in COMBO_INTERNAL_FIELDS and bool(item.spec.settings[name])
    )
    _draw_header(
        card,
        title=item.spec.title,
        subtitle=f"fixture: {fixture} | features: {features}",
        right_label=f"combo {item.spec.combo_kind}",
        dark=_is_dark_background(params),
    )
    _save_image(card, destination)
    sidecar = {
        "id": item.card_id,
        "kind": "combo",
        "category": f"combos/{item.spec.combo_kind}",
        "feature": "combo",
        "value": item.spec.title,
        "fixture": fixture,
        "settings": item.spec.settings,
        "competitor_tools": list(_card_competitor_tools(item)),
        "tier": _classify_tier(_card_competitor_tools(item)),
        "tier_c_reason": _tier_c_reason(item) if not _card_competitor_tools(item) else "",
    }
    _write_json(destination.with_suffix(".json"), sidecar)


def _render_evil_card(item: EvilCardItem, output_root: Path) -> None:
    """Render one evil stress-test card using its pre-built graph.

    Parameters
    ----------
    item : EvilCardItem
        Evil-card metadata.
    output_root : Path
        Gallery audit root directory.

    Returns
    -------
    None
        The evil PNG and JSON sidecar are written to disk.
    """

    destination = output_root / item.relative_path
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory() as temp_dir:
        raw_path = Path(temp_dir) / "evil.png"
        _render_dagua_png(item.spec.graph, item.spec.positions, raw_path, CARD_SIZE)
        card = _place_render_on_canvas(raw_path, CARD_SIZE, CARD_CONTENT_INSET)
    _draw_header(
        card,
        title=item.spec.title,
        subtitle="stress test",
        right_label="evil",
        dark=_is_dark_background(
            {"graph": {"background_color": _graph_background_color(item.spec.graph)}}
        ),
    )
    _save_image(card, destination)
    sidecar = {
        "id": item.card_id,
        "kind": "evil",
        "category": "evil",
        "feature": "stress_test",
        "value": item.spec.title,
        "settings": item.spec.settings,
        "competitor_tools": list(_card_competitor_tools(item)),
        "tier": _classify_tier(_card_competitor_tools(item)),
        "tier_c_reason": _tier_c_reason(item) if not _card_competitor_tools(item) else "",
    }
    _write_json(destination.with_suffix(".json"), sidecar)


def _board_title(category: str, index: int) -> str:
    """Build a readable board title from a category path.

    Parameters
    ----------
    category : str
        Board category identifier.
    index : int
        One-based board index within the category.

    Returns
    -------
    str
        Board title string.
    """

    readable = category.replace("_", " ").replace("/", " / ")
    return f"{readable} ({index})"


def _compose_board(
    image_paths: Sequence[Path],
    output_path: Path,
    title: str,
    dark: bool = False,
) -> None:
    """Compose a 2x2 navigation board from up to four card images.

    Parameters
    ----------
    image_paths : Sequence[Path]
        Card images to place on the board.
    output_path : Path
        Final board image path.
    title : str
        Board title.
    dark : bool, default=False
        Whether to use the dark-background header palette.

    Returns
    -------
    None
        The board PNG is written to disk.
    """

    canvas = Image.new("RGB", BOARD_SIZE, WHITE)
    for slot_index in range(4):
        tile = Image.new("RGB", BOARD_CELL_SIZE, WHITE)
        if slot_index < len(image_paths):
            with Image.open(image_paths[slot_index]) as opened:
                render_tile = opened.convert("RGB")
                render_tile.thumbnail(BOARD_CELL_SIZE, Image.LANCZOS)
                paste_x = (BOARD_CELL_SIZE[0] - render_tile.width) // 2
                paste_y = (BOARD_CELL_SIZE[1] - render_tile.height) // 2
                tile.paste(render_tile, (paste_x, paste_y))
        x_offset = 0 if slot_index % 2 == 0 else BOARD_CELL_SIZE[0]
        y_offset = BOARD_GRID_TOP + (0 if slot_index < 2 else BOARD_CELL_SIZE[1])
        canvas.paste(tile, (x_offset, y_offset))
    _draw_header(canvas, title=title, subtitle="2x2 board", right_label=None, dark=dark)
    _save_image(canvas, output_path)


def _reset_output_dir(path: Path) -> None:
    """Remove and recreate an output subdirectory.

    Parameters
    ----------
    path : Path
        Directory to reset.

    Returns
    -------
    None
        The directory exists and is empty afterwards.
    """

    shutil.rmtree(path, ignore_errors=True)
    path.mkdir(parents=True, exist_ok=True)


def _group_paths_by_category(
    pairs: Sequence[Tuple[str, Path]],
) -> Dict[str, List[Path]]:
    """Group card image paths by category.

    Parameters
    ----------
    pairs : Sequence[tuple[str, Path]]
        Category-path pairs.

    Returns
    -------
    dict[str, list[Path]]
        Grouped image paths.
    """

    grouped: Dict[str, List[Path]] = {}
    for category, path in pairs:
        grouped.setdefault(category, []).append(path)
    return grouped


def _write_index(
    output_root: Path,
    strip_items: Sequence[StripCardItem],
    reference_items: Sequence[ReferenceCardItem],
    combo_items: Sequence[ComboCardItem],
    evil_items: Sequence[EvilCardItem],
    comparison_lookup: Mapping[str, str],
) -> Path:
    """Write the machine-readable JSONL card index.

    Parameters
    ----------
    output_root : Path
        Gallery audit root directory.
    strip_items : Sequence[StripCardItem]
        Strip cards rendered in the current run.
    reference_items : Sequence[ReferenceCardItem]
        Reference cards rendered in the current run.
    combo_items : Sequence[ComboCardItem]
        Combo cards rendered in the current run.
    evil_items : Sequence[EvilCardItem]
        Evil cards rendered in the current run.
    comparison_lookup : Mapping[str, str]
        Comparison path lookup keyed by reference card ID.

    Returns
    -------
    Path
        Written JSONL index path.
    """

    index_path = output_root / "index.jsonl"
    lines: List[str] = []
    for item in strip_items:
        entry = {
            "id": item.card_id,
            "kind": "reference_strip",
            "path": item.relative_path,
            "category": item.spec.category,
            "feature": item.spec.feature,
            "value": "strip",
            "value_slugs": [member.value.slug for member in item.members],
            "fixture": item.spec.fixture,
            "sensitivity": item.spec.sensitivity,
            "has_comparison": False,
            "comparison_path": None,
            "is_primary_review_artifact": True,
            "member_card_ids": [member.card_id for member in item.members],
            "competitor_tools": list(item.spec.competitor_tools),
            "tier": item.spec.tier,
            "tier_c_reason": ""
            if item.spec.competitor_tools
            else "strip has no single automated competitor",
        }
        lines.append(json.dumps(entry, sort_keys=True))
    for item in reference_items:
        entry = {
            "id": item.card_id,
            "kind": "reference",
            "path": item.relative_path,
            "category": item.spec.category,
            "feature": item.spec.feature,
            "value": item.value.slug,
            "fixture": item.spec.fixture,
            "sensitivity": item.spec.sensitivity,
            "has_comparison": item.card_id in comparison_lookup,
            "comparison_path": comparison_lookup.get(item.card_id),
            "competitor_tools": list(_card_competitor_tools(item)),
            "tier": _classify_tier(_card_competitor_tools(item)),
            "tier_c_reason": _tier_c_reason(item) if not _card_competitor_tools(item) else "",
        }
        lines.append(json.dumps(entry, sort_keys=True))
    for item in combo_items:
        entry = {
            "id": item.card_id,
            "kind": "combo",
            "path": item.relative_path,
            "category": f"combos/{item.spec.combo_kind}",
            "feature": "combo",
            "value": item.spec.title,
            "fixture": _choose_combo_fixture(item.spec.settings),
            "sensitivity": "coarse",
            "has_comparison": False,
            "comparison_path": None,
            "combo_features": item.spec.settings,
            "competitor_tools": list(_card_competitor_tools(item)),
            "tier": _classify_tier(_card_competitor_tools(item)),
            "tier_c_reason": _tier_c_reason(item) if not _card_competitor_tools(item) else "",
        }
        lines.append(json.dumps(entry, sort_keys=True))
    for item in evil_items:
        entry = {
            "id": item.card_id,
            "kind": "evil",
            "path": item.relative_path,
            "category": "evil",
            "feature": "stress_test",
            "value": item.spec.title,
            "has_comparison": False,
            "comparison_path": None,
            "settings": item.spec.settings,
            "competitor_tools": list(_card_competitor_tools(item)),
            "tier": _classify_tier(_card_competitor_tools(item)),
            "tier_c_reason": _tier_c_reason(item) if not _card_competitor_tools(item) else "",
        }
        lines.append(json.dumps(entry, sort_keys=True))
    index_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return index_path


def build_reference_specs() -> Tuple[AtomicCardSpec, ...]:
    """Build the full atomic reference manifest.

    Returns
    -------
    tuple[AtomicCardSpec, ...]
        Ordered atomic reference specs.
    """

    specs = (
        _spec(
            "node",
            "nodes/shapes",
            "shape",
            ("shape",),
            "pair",
            (
                _value(
                    "rect",
                    "Rect",
                    {"node": {"shape": "rect", "corner_radius": 0.0}},
                    GRAPHVIZ_SHAPE_MAP["rect"],
                ),
                _value(
                    "roundrect",
                    "Roundrect",
                    {"node": {"shape": "roundrect", "corner_radius": 12.0}},
                    GRAPHVIZ_SHAPE_MAP["roundrect"],
                ),
                _value(
                    "arrow",
                    "Arrow",
                    {"node": {"shape": "arrow", "min_width": 152.0, "min_height": 78.0}},
                ),
                _value(
                    "ellipse",
                    "Ellipse",
                    {"node": {"shape": "ellipse", "min_width": 116.0, "min_height": 72.0}},
                    GRAPHVIZ_SHAPE_MAP["ellipse"],
                ),
                _value(
                    "diamond",
                    "Diamond",
                    {
                        "node": {"shape": "diamond", "min_width": 124.0, "min_height": 88.0},
                        "node_labels": ["A", "B"],
                    },
                    GRAPHVIZ_SHAPE_MAP["diamond"],
                ),
                _value(
                    "circle",
                    "Circle",
                    {"node": {"shape": "circle", "min_width": 108.0, "min_height": 108.0}},
                    GRAPHVIZ_SHAPE_MAP["circle"],
                ),
                _value(
                    "triangle",
                    "Triangle",
                    {
                        "node": {"shape": "triangle", "min_width": 120.0, "min_height": 102.0},
                        "node_labels": ["A", "B"],
                    },
                    GRAPHVIZ_SHAPE_MAP["triangle"],
                ),
                _value(
                    "hexagon",
                    "Hexagon",
                    {"node": {"shape": "hexagon", "min_width": 132.0, "min_height": 80.0}},
                    GRAPHVIZ_SHAPE_MAP["hexagon"],
                ),
                _value(
                    "pentagon",
                    "Pentagon",
                    {"node": {"shape": "pentagon", "min_width": 124.0, "min_height": 90.0}},
                    GRAPHVIZ_SHAPE_MAP["pentagon"],
                ),
                _value(
                    "octagon",
                    "Octagon",
                    {"node": {"shape": "octagon", "min_width": 132.0, "min_height": 88.0}},
                    GRAPHVIZ_SHAPE_MAP["octagon"],
                ),
                _value(
                    "star",
                    "Star",
                    {
                        "node": {"shape": "star", "min_width": 128.0, "min_height": 128.0},
                        "node_labels": ["A", "B"],
                    },
                    GRAPHVIZ_SHAPE_MAP["star"],
                ),
                _value(
                    "cylinder",
                    "Cylinder",
                    {"node": {"shape": "cylinder", "min_width": 120.0, "min_height": 88.0}},
                    GRAPHVIZ_SHAPE_MAP["cylinder"],
                ),
                _value(
                    "parallelogram",
                    "Parallelogram",
                    {
                        "node": {
                            "shape": "parallelogram",
                            "min_width": 140.0,
                            "min_height": 76.0,
                        }
                    },
                    GRAPHVIZ_SHAPE_MAP["parallelogram"],
                ),
                _value(
                    "trapezoid",
                    "Trapezoid",
                    {"node": {"shape": "trapezoid", "min_width": 138.0, "min_height": 80.0}},
                    GRAPHVIZ_SHAPE_MAP["trapezoid"],
                ),
                _value(
                    "double_circle",
                    "Double Circle",
                    {
                        "node": {
                            "shape": "double_circle",
                            "min_width": 116.0,
                            "min_height": 116.0,
                            "stroke_width": 0.0,
                        }
                    },
                    GRAPHVIZ_SHAPE_MAP["double_circle"],
                ),
                _value(
                    "cloud",
                    "Cloud",
                    {"node": {"shape": "cloud", "min_width": 136.0, "min_height": 86.0}},
                ),
                _value(
                    "stadium",
                    "Stadium",
                    {"node": {"shape": "stadium", "min_width": 148.0, "min_height": 70.0}},
                ),
                _value(
                    "tab",
                    "Tab",
                    {
                        "node": {
                            "shape": "tab",
                            "min_width": 126.0,
                            "min_height": 82.0,
                            "stroke_width": 0.0,
                        }
                    },
                    GRAPHVIZ_SHAPE_MAP["tab"],
                ),
                _value(
                    "note",
                    "Note",
                    {
                        "node": {
                            "shape": "note",
                            "min_width": 126.0,
                            "min_height": 94.0,
                            "stroke_width": 0.0,
                        }
                    },
                    GRAPHVIZ_SHAPE_MAP["note"],
                ),
                _value(
                    "document",
                    "Document",
                    {"node": {"shape": "document", "min_width": 136.0, "min_height": 90.0}},
                ),
                _value(
                    "box3d",
                    "3D Box",
                    {
                        "node": {
                            "shape": "box3d",
                            "min_width": 126.0,
                            "min_height": 84.0,
                            "stroke_width": 1.5,
                        }
                    },
                    GRAPHVIZ_SHAPE_MAP["box3d"],
                ),
            ),
        ),
        _spec(
            "node",
            "nodes/fills",
            "gradient",
            ("gradient", "gradient_color"),
            "pair",
            (
                _value(
                    "solid",
                    "Solid",
                    {"node": {"gradient": "none", "fill": NODE_FILL, "gradient_color": ""}},
                ),
                _value(
                    "linear",
                    "Linear",
                    {
                        "node": {
                            "gradient": "linear",
                            "fill": GRADIENT_FILL,
                            "gradient_color": GRADIENT_COLOR,
                            "font_color": "#FFFFFF",
                        }
                    },
                ),
                _value(
                    "radial",
                    "Radial",
                    {
                        "node": {
                            "gradient": "radial",
                            "fill": GRADIENT_FILL,
                            "gradient_color": GRADIENT_COLOR,
                        }
                    },
                ),
            ),
            filename_prefix="gradient",
        ),
        _spec(
            "node",
            "nodes/borders",
            "stroke_dash",
            ("stroke_dash",),
            "pair",
            (
                _value("solid", "Solid", {"node": {"stroke_dash": "solid"}}),
                _value("dashed", "Dashed", {"node": {"stroke_dash": "dashed"}}),
                _value("dotted", "Dotted", {"node": {"stroke_dash": "dotted"}}),
            ),
            filename_prefix="stroke_dash",
        ),
        _spec(
            "node",
            "nodes/borders",
            "stroke_width",
            ("stroke_width",),
            "pair",
            (
                _value("0_5", "0.5", {"node": {"stroke_width": 0.5}}),
                _value("1_5", "1.5", {"node": {"stroke_width": 1.5}}),
                _value("3_0", "3.0", {"node": {"stroke_width": 3.0}}),
                _value("5_0", "5.0", {"node": {"stroke_width": 5.0}}),
            ),
            filename_prefix="stroke_width",
        ),
        _spec(
            "node",
            "nodes/borders",
            "corner_radius",
            ("corner_radius",),
            "pair",
            (
                _value(
                    "0",
                    "0",
                    {
                        "node": {
                            "shape": "roundrect",
                            "corner_radius": 0.0,
                            "min_width": 140.0,
                            "min_height": 80.0,
                        }
                    },
                ),
                _value(
                    "12",
                    "12",
                    {
                        "node": {
                            "shape": "roundrect",
                            "corner_radius": 12.0,
                            "min_width": 140.0,
                            "min_height": 80.0,
                        }
                    },
                ),
                _value(
                    "24",
                    "24",
                    {
                        "node": {
                            "shape": "roundrect",
                            "corner_radius": 24.0,
                            "min_width": 140.0,
                            "min_height": 80.0,
                        }
                    },
                ),
                _value(
                    "40",
                    "40",
                    {
                        "node": {
                            "shape": "roundrect",
                            "corner_radius": 40.0,
                            "min_width": 140.0,
                            "min_height": 80.0,
                        }
                    },
                ),
            ),
            filename_prefix="corner_radius",
        ),
        _spec(
            "node",
            "nodes/borders",
            "corner_radius_per_corner",
            ("corner_radius",),
            "pair",
            (
                _value(
                    "corner_radius_per_corner",
                    "Per-Corner Radius",
                    {
                        "pair_layout": "horizontal",
                        "pair_gap": 220.0,
                        "hide_edges": True,
                        "node_style_overrides": [
                            {
                                "shape": "roundrect",
                                "corner_radius": 12.0,
                                "min_width": 140.0,
                                "min_height": 80.0,
                            },
                            {
                                "shape": "roundrect",
                                "corner_radius": (0.0, 24.0, 0.0, 24.0),
                                "min_width": 140.0,
                                "min_height": 80.0,
                            },
                        ],
                        "node_labels": ["Uniform", "Per-corner"],
                    },
                ),
            ),
        ),
        _spec(
            "node",
            "nodes/borders",
            "scale_corner_radius",
            ("corner_radius", "scale_corner_radius"),
            "pair",
            (
                _value(
                    "scale_corner_radius",
                    "Scale Corner Radius",
                    {
                        "pair_layout": "horizontal",
                        "pair_gap": 300.0,
                        "hide_edges": True,
                        "node_style_overrides": [
                            {
                                "shape": "roundrect",
                                "scale_corner_radius": True,
                                "corner_radius": 0.15,
                                "min_width": 96.0,
                                "min_height": 54.0,
                            },
                            {
                                "shape": "roundrect",
                                "scale_corner_radius": True,
                                "corner_radius": 0.15,
                                "min_width": 192.0,
                                "min_height": 108.0,
                            },
                        ],
                        "node_labels": ["Small", "Large"],
                    },
                ),
            ),
        ),
        _spec(
            "node",
            "nodes/borders",
            "border_opacity",
            ("border_opacity",),
            "pair",
            (
                _value(
                    "0_2",
                    "0.2",
                    {"node": {"opacity": 1.0, "border_opacity": 0.2, "stroke_width": 4.0}},
                ),
                _value(
                    "0_5",
                    "0.5",
                    {"node": {"opacity": 1.0, "border_opacity": 0.5, "stroke_width": 4.0}},
                ),
                _value(
                    "0_8",
                    "0.8",
                    {"node": {"opacity": 1.0, "border_opacity": 0.8, "stroke_width": 4.0}},
                ),
                _value(
                    "1_0",
                    "1.0",
                    {"node": {"opacity": 1.0, "border_opacity": 1.0, "stroke_width": 4.0}},
                ),
            ),
            filename_prefix="border_opacity",
        ),
        _spec(
            "node",
            "nodes/fills",
            "opacity",
            ("opacity",),
            "pair",
            (
                _value(
                    "0_2",
                    "0.2",
                    {"node": {"fill": NODE_FILL, "opacity": 0.2, "border_opacity": 0.2}},
                ),
                _value(
                    "0_5",
                    "0.5",
                    {"node": {"fill": NODE_FILL, "opacity": 0.5, "border_opacity": 0.5}},
                ),
                _value(
                    "0_8",
                    "0.8",
                    {"node": {"fill": NODE_FILL, "opacity": 0.8, "border_opacity": 0.8}},
                ),
                _value(
                    "1_0",
                    "1.0",
                    {"node": {"fill": NODE_FILL, "opacity": 1.0, "border_opacity": 1.0}},
                ),
            ),
            filename_prefix="opacity",
        ),
        _spec(
            "node",
            "nodes/effects",
            "shadow",
            ("shadow", "shadow_offset", "shadow_blur"),
            "pair",
            (
                _value("off", "Off", {"node": {"shadow": False}}),
                _value(
                    "on",
                    "On",
                    {
                        "node": {
                            "shadow": True,
                            "shadow_offset": (5.0, -5.0),
                            "shadow_color": SHADOW_COLOR,
                            "shadow_blur": 4.0,
                        }
                    },
                ),
            ),
        ),
        _spec(
            "node",
            "nodes/effects",
            "bevel",
            ("bevel", "bevel_intensity"),
            "pair",
            (
                _value(
                    "off",
                    "Off",
                    {
                        "node": {
                            "bevel": False,
                            "fill": "#7FA7D2",
                            "stroke": "#476E97",
                            "font_color": "#FFFFFF",
                            "min_width": 136.0,
                            "min_height": 72.0,
                        }
                    },
                ),
                _value(
                    "on",
                    "On",
                    {
                        "node": {
                            "bevel": True,
                            "bevel_intensity": 0.35,
                            "fill": "#7FA7D2",
                            "stroke": "#476E97",
                            "font_color": "#FFFFFF",
                            "min_width": 136.0,
                            "min_height": 72.0,
                        }
                    },
                ),
            ),
            filename_prefix="bevel",
        ),
        _spec(
            "node",
            "nodes/borders",
            "border_count",
            ("border_count",),
            "pair",
            (
                _value(
                    "1_vs_2",
                    "1 vs 2",
                    {"node": {"border_count": 2, "stroke_width": 3.0}},
                ),
                _value(
                    "2_vs_3",
                    "2 vs 3",
                    {"node": {"border_count": 3, "stroke_width": 3.0}},
                ),
            ),
            filename_prefix="border_count",
        ),
        _spec(
            "node",
            "nodes/borders",
            "border_position",
            ("border_position",),
            "pair",
            (
                _value(
                    "inside",
                    "Inside",
                    {
                        "node": {
                            "shape": "rect",
                            "border_position": "inside",
                            "stroke_width": 50.0,
                            "min_width": 80.0,
                            "min_height": 60.0,
                            "fill": "#FFE0B2",
                            "stroke": "#E65100",
                        }
                    },
                ),
                _value(
                    "outside",
                    "Outside",
                    {
                        "node": {
                            "shape": "rect",
                            "border_position": "outside",
                            "stroke_width": 50.0,
                            "min_width": 80.0,
                            "min_height": 60.0,
                            "fill": "#FFE0B2",
                            "stroke": "#E65100",
                        }
                    },
                ),
            ),
            filename_prefix="border_position",
        ),
        _spec(
            "node",
            "nodes/text",
            "text_align",
            ("text_align",),
            "pair",
            (
                _value(
                    "left",
                    "Left",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "min_width": 176.0,
                            "min_height": 92.0,
                            "text_align": "left",
                        },
                        "node_labels": ["Left aligned\ntext sample", "Left aligned\ntext sample"],
                    },
                ),
                _value(
                    "center",
                    "Center",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "min_width": 176.0,
                            "min_height": 92.0,
                            "text_align": "center",
                        },
                        "node_labels": [
                            "Center aligned\ntext sample",
                            "Center aligned\ntext sample",
                        ],
                    },
                ),
                _value(
                    "right",
                    "Right",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "min_width": 176.0,
                            "min_height": 92.0,
                            "text_align": "right",
                        },
                        "node_labels": ["Right aligned\ntext sample", "Right aligned\ntext sample"],
                    },
                ),
            ),
            filename_prefix="text_align",
        ),
        _spec(
            "node",
            "nodes/text",
            "text_valign",
            ("text_valign",),
            "pair",
            (
                _value(
                    "top",
                    "Top",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "min_width": 176.0,
                            "min_height": 120.0,
                            "text_valign": "top",
                        },
                        "node_labels": ["Top\naligned", "Top\naligned"],
                    },
                ),
                _value(
                    "center",
                    "Center",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "min_width": 176.0,
                            "min_height": 120.0,
                            "text_valign": "center",
                        },
                        "node_labels": ["Center\naligned", "Center\naligned"],
                    },
                ),
                _value(
                    "bottom",
                    "Bottom",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "min_width": 176.0,
                            "min_height": 140.0,
                            "text_valign": "bottom",
                        },
                        "node_labels": ["Bottom\naligned", "Bottom\naligned"],
                    },
                ),
            ),
            filename_prefix="text_valign",
        ),
        _spec(
            "node",
            "nodes/text",
            "font_weight",
            ("font_weight",),
            "pair",
            (
                _value("regular", "Regular", {"node": {"font_weight": "regular"}}),
                _value("bold", "Bold", {"node": {"font_weight": "bold"}}),
            ),
            filename_prefix="font_weight",
        ),
        _spec(
            "node",
            "nodes/text",
            "font_style",
            ("font_style",),
            "pair",
            (
                _value("normal", "Normal", {"node": {"font_style": "normal"}}),
                _value("italic", "Italic", {"node": {"font_style": "italic"}}),
            ),
            filename_prefix="font_style",
        ),
        _spec(
            "node",
            "nodes/text",
            "text_rotation",
            ("text_rotation",),
            "pair",
            (
                _value(
                    "0",
                    "0",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "min_width": 116.0,
                            "min_height": 116.0,
                            "text_rotation": 0.0,
                        },
                        "node_labels": ["Rotate", "Rotate"],
                    },
                ),
                _value(
                    "45",
                    "45",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "min_width": 116.0,
                            "min_height": 116.0,
                            "text_rotation": 45.0,
                        },
                        "node_labels": ["Rotate", "Rotate"],
                    },
                ),
                _value(
                    "90",
                    "90",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "min_width": 116.0,
                            "min_height": 116.0,
                            "text_rotation": 90.0,
                        },
                        "node_labels": ["Rotate", "Rotate"],
                    },
                ),
            ),
            filename_prefix="text_rotation",
        ),
        _spec(
            "node",
            "nodes/text",
            "text_wrap",
            ("text_wrap", "text_max_width"),
            "pair",
            (
                _value(
                    "none",
                    "None",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "min_width": 148.0,
                            "min_height": 78.0,
                            "text_wrap": "none",
                            "text_max_width": 92.0,
                        },
                        "node_labels": ["Readable label sample", "Readable label sample"],
                    },
                ),
                _value(
                    "wrap",
                    "Wrap",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "min_width": 116.0,
                            "min_height": 112.0,
                            "text_wrap": "wrap",
                            "text_max_width": 60.0,
                        },
                        "node_labels": [WRAP_DEMO_LABEL, WRAP_DEMO_LABEL],
                    },
                ),
                _value(
                    "ellipsis",
                    "Ellipsis",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "min_width": 148.0,
                            "min_height": 78.0,
                            "text_wrap": "ellipsis",
                            "text_max_width": 74.0,
                        },
                        "node_labels": [ELLIPSIS_DEMO_LABEL, ELLIPSIS_DEMO_LABEL],
                    },
                ),
            ),
            filename_prefix="text_wrap",
        ),
        _spec(
            "node",
            "nodes/text",
            "text_background",
            ("text_background",),
            "pair",
            (
                _value("none", "None", {"node": {"text_background": ""}}),
                _value(
                    "orange",
                    "Orange",
                    {"node": {"text_background": "#FFE0B2", "text_background_opacity": 0.9}},
                ),
                _value(
                    "green",
                    "Green",
                    {"node": {"text_background": "#C8F2D1", "text_background_opacity": 0.9}},
                ),
            ),
            filename_prefix="text_background",
        ),
        _spec(
            "node",
            "nodes/text",
            "text_outline",
            ("text_outline",),
            "pair",
            (
                _value("off", "Off", {"node": {"text_outline": False}}),
                _value(
                    "on",
                    "On",
                    {
                        "node": {
                            "fill": NODE_FILL,
                            "stroke": NODE_STROKE,
                            "text_outline": True,
                            "text_outline_color": TEXT_OUTLINE_COLOR,
                            "text_outline_width": 3.0,
                        }
                    },
                ),
            ),
            filename_prefix="text_outline",
        ),
        _spec(
            "node",
            "nodes/text",
            "external_label",
            ("external_label", "external_label_position"),
            "pair",
            (
                _value(
                    "top",
                    "Top",
                    {
                        "node": {"external_label": "ID 42", "external_label_position": "top"},
                        "graph": {"margin": 42.0},
                    },
                ),
                _value(
                    "bottom",
                    "Bottom",
                    {
                        "node": {"external_label": "ID 42", "external_label_position": "bottom"},
                        "graph": {"margin": 42.0},
                    },
                ),
                _value(
                    "left",
                    "Left",
                    {
                        "node": {"external_label": "ID 42", "external_label_position": "left"},
                        "graph": {"margin": 42.0},
                    },
                ),
                _value(
                    "right",
                    "Right",
                    {
                        "node": {"external_label": "ID 42", "external_label_position": "right"},
                        "graph": {"margin": 42.0},
                    },
                ),
            ),
            filename_prefix="external_label",
        ),
        _spec(
            "node",
            "nodes/fills",
            "fill_pattern",
            ("fill_pattern",),
            "pair",
            (
                _value(
                    "solid",
                    "Solid",
                    {
                        "node": {
                            "fill_pattern": "solid",
                            "fill_pattern_colors": None,
                            "fill_pattern_values": None,
                        }
                    },
                ),
                _value(
                    "striped",
                    "Striped",
                    {
                        "node": {
                            "fill_pattern": "striped",
                            "fill_pattern_colors": PATTERN_COLORS[:2],
                            "fill_pattern_angle": 30.0,
                        }
                    },
                ),
                _value(
                    "hatched",
                    "Hatched",
                    {
                        "node": {
                            "fill_pattern": "hatched",
                            "fill_pattern_colors": PATTERN_COLORS[:2],
                            "fill_pattern_angle": 45.0,
                        }
                    },
                ),
                _value(
                    "pie",
                    "Pie",
                    {
                        "node": {
                            "shape": "circle",
                            "fill_pattern": "pie",
                            "fill_pattern_colors": PATTERN_COLORS[:3],
                            "fill_pattern_values": [3.0, 2.0, 1.0],
                            "min_width": 112.0,
                            "min_height": 112.0,
                        },
                        "node_labels": ["A", "B"],
                    },
                ),
            ),
            filename_prefix="fill_pattern",
        ),
        _spec(
            "node",
            "nodes/text",
            "overflow_policy",
            ("overflow_policy",),
            "pair",
            (
                _value(
                    "shrink_text",
                    "Shrink Text",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "overflow_policy": "shrink_text",
                            "font_size": 11.0,
                            "min_width": 108.0,
                            "min_height": 52.0,
                        },
                        "node_labels": [OVERFLOW_DEMO_LABEL, OVERFLOW_DEMO_LABEL],
                    },
                ),
                _value(
                    "expand_node",
                    "Expand Node",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "overflow_policy": "expand_node",
                            "font_size": 11.0,
                            "min_width": 108.0,
                            "min_height": 52.0,
                        },
                        "node_labels": [OVERFLOW_EXPAND_LABEL, OVERFLOW_EXPAND_LABEL],
                    },
                ),
                _value(
                    "overflow",
                    "Overflow",
                    {
                        "node": {
                            "shape": "rect",
                            "corner_radius": 0.0,
                            "overflow_policy": "overflow",
                            "font_size": 11.0,
                            "min_width": 108.0,
                            "min_height": 52.0,
                        },
                        "node_labels": [OVERFLOW_DEMO_LABEL, OVERFLOW_DEMO_LABEL],
                    },
                ),
            ),
            filename_prefix="overflow_policy",
        ),
        _spec(
            "edge",
            "edges/arrows",
            "arrow",
            ("arrow",),
            "pair",
            (
                _value(
                    "normal",
                    "Normal",
                    {"edge": {"arrow": "normal", "arrow_length": 20.0, "arrow_width": 14.0}},
                    GRAPHVIZ_ARROW_MAP["normal"],
                ),
                _value(
                    "vee",
                    "Vee",
                    {"edge": {"arrow": "vee", "arrow_length": 20.0, "arrow_width": 14.0}},
                    GRAPHVIZ_ARROW_MAP["vee"],
                ),
                _value(
                    "dot",
                    "Dot",
                    {"edge": {"arrow": "dot", "arrow_length": 20.0, "arrow_width": 14.0}},
                    GRAPHVIZ_ARROW_MAP["dot"],
                ),
                _value(
                    "diamond",
                    "Diamond",
                    {"edge": {"arrow": "diamond", "arrow_length": 20.0, "arrow_width": 14.0}},
                    GRAPHVIZ_ARROW_MAP["diamond"],
                ),
                _value(
                    "tee",
                    "Tee",
                    {"edge": {"arrow": "tee", "arrow_length": 20.0, "arrow_width": 14.0}},
                    GRAPHVIZ_ARROW_MAP["tee"],
                ),
                _value(
                    "crow",
                    "Crow",
                    {"edge": {"arrow": "crow", "arrow_length": 36.0, "arrow_width": 28.0}},
                    GRAPHVIZ_ARROW_MAP["crow"],
                ),
                _value(
                    "circle",
                    "Circle",
                    {"edge": {"arrow": "circle", "arrow_length": 20.0, "arrow_width": 14.0}},
                    GRAPHVIZ_ARROW_MAP["circle"],
                ),
                _value(
                    "open",
                    "Open",
                    {"edge": {"arrow": "open", "arrow_length": 20.0, "arrow_width": 14.0}},
                    GRAPHVIZ_ARROW_MAP["open"],
                ),
            ),
        ),
        _spec(
            "edge",
            "edges/arrows",
            "arrow_fill",
            ("arrow_fill",),
            "pair",
            (
                _value(
                    "filled",
                    "Filled",
                    {
                        "edge": {
                            "arrow_fill": "filled",
                            "arrow": "normal",
                            "arrow_length": 20.0,
                            "arrow_width": 14.0,
                        }
                    },
                ),
                _value(
                    "hollow",
                    "Hollow",
                    {
                        "edge": {
                            "arrow_fill": "hollow",
                            "arrow_color": EDGE_COLOR,
                            "arrow": "normal",
                            "arrow_length": 20.0,
                            "arrow_width": 14.0,
                        }
                    },
                ),
            ),
            filename_prefix="arrow_fill",
        ),
        _spec(
            "edge",
            "edges/styles",
            "style",
            ("style",),
            "pair",
            (
                _value("solid", "Solid", {"edge": {"style": "solid"}}),
                _value("dashed", "Dashed", {"edge": {"style": "dashed"}}),
                _value("dotted", "Dotted", {"edge": {"style": "dotted"}}),
            ),
            filename_prefix="style",
        ),
        _spec(
            "edge",
            "edges/styles",
            "port_indicator",
            ("port_indicator", "port_indicator_size"),
            "pair",
            (
                _value(
                    "circle",
                    "Circle",
                    {
                        "pair_layout": "horizontal",
                        "pair_gap": 260.0,
                        "edge": {
                            "arrow": "none",
                            "routing": "straight",
                            "width": 3.0,
                            "port_indicator": "circle",
                            "port_indicator_size": 4.0,
                        },
                    },
                ),
                _value(
                    "diamond",
                    "Diamond",
                    {
                        "pair_layout": "horizontal",
                        "pair_gap": 260.0,
                        "edge": {
                            "arrow": "none",
                            "routing": "straight",
                            "width": 3.0,
                            "port_indicator": "diamond",
                            "port_indicator_size": 4.0,
                        },
                    },
                ),
            ),
            filename_prefix="port_indicator",
        ),
        _spec(
            "edge",
            "edges/styles",
            "width",
            ("width",),
            "pair",
            (
                _value("0_5", "0.5", {"edge": {"width": 0.5}}),
                _value("1_5", "1.5", {"edge": {"width": 1.5}}),
                _value("3_0", "3.0", {"edge": {"width": 3.0}}),
                _value("5_0", "5.0", {"edge": {"width": 5.0}}),
            ),
            filename_prefix="width",
        ),
        _spec(
            "edge",
            "edges/routing",
            "routing",
            ("routing",),
            "chain",
            (
                _value("bezier", "Bezier", {"edge": {"routing": "bezier", "width": 2.2}}),
                _value("straight", "Straight", {"edge": {"routing": "straight", "width": 2.2}}),
                _value("ortho", "Ortho", {"edge": {"routing": "ortho", "width": 2.2}}),
                _value("taxi", "Taxi", {"edge": {"routing": "taxi", "width": 2.2}}),
            ),
            filename_prefix="routing",
        ),
        _spec(
            "edge",
            "edges/routing",
            "curvature",
            ("curvature",),
            "fan",
            (
                _value(
                    "0_0", "0.0", {"edge": {"routing": "bezier", "curvature": 0.0, "width": 2.2}}
                ),
                _value(
                    "0_4", "0.4", {"edge": {"routing": "bezier", "curvature": 0.4, "width": 2.2}}
                ),
                _value(
                    "0_8", "0.8", {"edge": {"routing": "bezier", "curvature": 0.8, "width": 2.2}}
                ),
            ),
            filename_prefix="curvature",
        ),
        _spec(
            "edge",
            "edges/advanced",
            "taper",
            ("taper", "taper_width_start", "taper_width_end"),
            "crossing",
            (
                _value(
                    "off",
                    "Off",
                    {"edge": {"taper": False, "taper_width_start": 3.0, "taper_width_end": 3.0}},
                ),
                _value(
                    "3_to_1",
                    "3->1",
                    {"edge": {"taper": True, "taper_width_start": 3.0, "taper_width_end": 1.0}},
                ),
                _value(
                    "3_to_0_5",
                    "3->0.5",
                    {"edge": {"taper": True, "taper_width_start": 3.0, "taper_width_end": 0.5}},
                ),
            ),
            filename_prefix="taper",
        ),
        _spec(
            "edge",
            "edges/advanced",
            "crossing_style",
            ("crossing_style",),
            "crossing",
            (
                _value("none", "None", {"edge": {"crossing_style": "none", "width": 3.0}}),
                _value(
                    "arc",
                    "Arc",
                    {"edge": {"crossing_style": "arc", "crossing_size": 10.0, "width": 3.0}},
                ),
                _value(
                    "gap",
                    "Gap",
                    {"edge": {"crossing_style": "gap", "crossing_size": 10.0, "width": 3.0}},
                ),
                _value(
                    "sharp",
                    "Sharp",
                    {"edge": {"crossing_style": "sharp", "crossing_size": 10.0, "width": 3.0}},
                ),
                _value(
                    "bridge",
                    "Bridge",
                    {"edge": {"crossing_style": "bridge", "crossing_size": 10.0, "width": 3.0}},
                ),
            ),
            filename_prefix="crossing_style",
        ),
        _spec(
            "edge",
            "edges/advanced",
            "color_gradient",
            ("color_gradient", "color_gradient_end"),
            "crossing",
            (
                _value(
                    "none",
                    "None",
                    {"edge": {"color_gradient": "none", "color_gradient_end": "", "width": 3.0}},
                ),
                _value(
                    "source_to_target",
                    "Source to Target",
                    {
                        "edge": {
                            "color_gradient": "source_to_target",
                            "color": GRADIENT_FILL,
                            "color_gradient_end": EDGE_GRADIENT_END,
                            "width": 3.0,
                        }
                    },
                ),
            ),
            filename_prefix="color_gradient",
        ),
        _spec(
            "edge",
            "edges/labels",
            "label_position",
            ("label_position",),
            "pair",
            (
                _value(
                    "0_2", "0.2", {"edge": {"label_position": 0.2}, "edge_labels": ["weight=1.0"]}
                ),
                _value(
                    "0_5", "0.5", {"edge": {"label_position": 0.5}, "edge_labels": ["weight=1.0"]}
                ),
                _value(
                    "0_8", "0.8", {"edge": {"label_position": 0.8}, "edge_labels": ["weight=1.0"]}
                ),
            ),
        ),
        _spec(
            "edge",
            "edges/routing",
            "port_style",
            ("port_style",),
            "fan",
            (
                _value(
                    "distributed",
                    "Distributed",
                    {"edge": {"port_style": "distributed", "routing": "straight", "width": 2.0}},
                ),
                _value(
                    "center",
                    "Center",
                    {"edge": {"port_style": "center", "routing": "straight", "width": 2.0}},
                ),
            ),
            filename_prefix="port_style",
        ),
        _spec(
            "cluster",
            "clusters",
            "stroke_dash",
            ("stroke_dash",),
            "cluster_nested",
            (
                _value("solid", "Solid", {"cluster": {"stroke_dash": "solid"}}),
                _value("dashed", "Dashed", {"cluster": {"stroke_dash": "dashed"}}),
                _value(
                    "dotted",
                    "Dotted",
                    {"cluster": {"stroke_dash": "dotted", "stroke_width": 2.5}},
                ),
            ),
            filename_prefix="stroke_dash",
        ),
        _spec(
            "cluster",
            "clusters",
            "label_position",
            ("label_position",),
            "cluster_nested",
            (
                _value(
                    "top_left",
                    "Top Left",
                    {
                        "cluster": {"label_position": "top-left"},
                        "cluster_label_offset": [12.0, 10.0],
                    },
                ),
                _value(
                    "top_center",
                    "Top Center",
                    {
                        "cluster": {"label_position": "top-center"},
                        "cluster_label_offset": [0.0, 10.0],
                    },
                ),
                _value(
                    "top_right",
                    "Top Right",
                    {
                        "cluster": {"label_position": "top-right"},
                        "cluster_label_offset": [-12.0, 10.0],
                    },
                ),
            ),
            filename_prefix="label_position",
        ),
        _spec(
            "cluster",
            "clusters",
            "opacity",
            ("opacity",),
            "cluster_nested",
            (
                _value(
                    "0_3",
                    "0.3",
                    {
                        "cluster": {
                            "fill": SATURATED_CLUSTER_FILL,
                            "fill_opacity": 1.0,
                            "border_opacity": 1.0,
                            "opacity": 0.3,
                        }
                    },
                ),
                _value(
                    "0_6",
                    "0.6",
                    {
                        "cluster": {
                            "fill": SATURATED_CLUSTER_FILL,
                            "fill_opacity": 1.0,
                            "border_opacity": 1.0,
                            "opacity": 0.6,
                        }
                    },
                ),
                _value(
                    "1_0",
                    "1.0",
                    {
                        "cluster": {
                            "fill": SATURATED_CLUSTER_FILL,
                            "fill_opacity": 1.0,
                            "border_opacity": 1.0,
                            "opacity": 1.0,
                        }
                    },
                ),
            ),
            filename_prefix="opacity",
        ),
        _spec(
            "cluster",
            "clusters",
            "corner_radius",
            ("corner_radius",),
            "cluster_nested",
            (
                _value("0", "0", {"cluster": {"corner_radius": 0.0}}),
                _value("8", "8", {"cluster": {"corner_radius": 8.0}}),
                _value("16", "16", {"cluster": {"corner_radius": 16.0}}),
            ),
            filename_prefix="corner_radius",
        ),
        _spec(
            "graph",
            "graph",
            "background_color",
            ("background_color",),
            "chain",
            (
                _value("white", "White", {"graph": {"background_color": WHITE}}),
                _value(
                    "dark",
                    "Dark",
                    {"graph": {"background_color": "#1F2937"}, "dark_background": True},
                ),
                _value(
                    "near_black",
                    "Near Black",
                    {"graph": {"background_color": "#05070B"}, "dark_background": True},
                ),
            ),
            filename_prefix="background",
        ),
        _spec(
            "graph",
            "graph",
            "direction",
            ("direction",),
            "chain",
            (
                _value(
                    "tb",
                    "TB",
                    {
                        "direction": "TB",
                        "position_variant": "chain_direction",
                        "graph": {"margin": 20.0},
                    },
                ),
                _value(
                    "bt",
                    "BT",
                    {
                        "direction": "BT",
                        "position_variant": "chain_direction",
                        "graph": {"margin": 20.0},
                    },
                ),
                _value(
                    "lr",
                    "LR",
                    {
                        "direction": "LR",
                        "position_variant": "chain_direction",
                        "graph": {"margin": 32.0},
                    },
                ),
                _value(
                    "rl",
                    "RL",
                    {
                        "direction": "RL",
                        "position_variant": "chain_direction",
                        "graph": {"margin": 32.0},
                    },
                ),
            ),
            filename_prefix="direction",
        ),
        _spec(
            "graph",
            "graph",
            "margin",
            ("margin",),
            "chain",
            (
                _value("0", "0", {"graph": {"margin": 0.0}}),
                _value("15", "15", {"graph": {"margin": 15.0}}),
                _value("40", "40", {"graph": {"margin": 40.0}}),
            ),
            filename_prefix="margin",
        ),
    )
    _validate_reference_specs(specs)
    return specs


def build_gallery_audit(
    output_dir: str = "eval_output/gallery_audit",
    *,
    cards: bool = True,
    boards: bool = True,
    comparisons: bool = True,
    combos: bool = True,
    evil: bool = True,
    index: bool = True,
    reference_card_ids: Optional[Sequence[str]] = None,
    combo_card_ids: Optional[Sequence[str]] = None,
    evil_card_ids: Optional[Sequence[str]] = None,
) -> GalleryAuditResult:
    """Build the gallery audit artifact tree.

    Parameters
    ----------
    output_dir : str, default="eval_output/gallery_audit"
        Root output directory.
    cards : bool, default=True
        Whether to build atomic reference cards.
    boards : bool, default=True
        Whether to build 2x2 navigation boards.
    comparisons : bool, default=True
        Whether to build Graphviz comparison cards.
    combos : bool, default=True
        Whether to build combo cards.
    evil : bool, default=True
        Whether to build evil stress cards.
    index : bool, default=True
        Whether to write ``index.jsonl``.
    reference_card_ids : Sequence[str] | None, optional
        Optional subset of reference card IDs for tests.
    combo_card_ids : Sequence[str] | None, optional
        Optional subset of combo card IDs for tests.
    evil_card_ids : Sequence[str] | None, optional
        Optional subset of evil card IDs for tests.

    Returns
    -------
    GalleryAuditResult
        Summary of generated artifacts.
    """

    output_root = Path(output_dir)
    output_root.mkdir(parents=True, exist_ok=True)

    reference_items = list(build_reference_items())
    if reference_card_ids is not None:
        selected = set(reference_card_ids)
        reference_items = [item for item in reference_items if item.card_id in selected]

    combo_items = list(build_combo_items())
    if combo_card_ids is not None:
        selected_combo = set(combo_card_ids)
        combo_items = [item for item in combo_items if item.card_id in selected_combo]
    evil_items = list(build_evil_items())
    if evil_card_ids is not None:
        selected_evil = set(evil_card_ids)
        evil_items = [item for item in evil_items if item.card_id in selected_evil]

    written_reference_items: List[ReferenceCardItem] = []
    written_strip_items: List[StripCardItem] = []
    written_combo_items: List[ComboCardItem] = []
    written_evil_items: List[EvilCardItem] = []
    comparison_lookup: Dict[str, str] = {}
    reference_pairs: List[Tuple[str, Path]] = []
    combo_pairs: List[Tuple[str, Path]] = []
    board_count = 0

    if cards:
        _reset_output_dir(output_root / "cards" / "reference")
        for item in reference_items:
            _render_reference_card(item, output_root)
            written_reference_items.append(item)
        for item in build_strip_items(written_reference_items):
            _render_strip_card(item, output_root)
            written_strip_items.append(item)
            reference_pairs.append((item.spec.category, output_root / item.relative_path))
        for item in written_reference_items:
            reference_pairs.append((item.spec.category, output_root / item.relative_path))

    if comparisons:
        _reset_output_dir(output_root / "cards" / "comparisons")
        for item in reference_items:
            if item.comparison_relative_path is None:
                continue
            _render_comparison_card(item, output_root)
            comparison_lookup[item.card_id] = item.comparison_relative_path

    if combos:
        _reset_output_dir(output_root / "cards" / "combos")
        for item in combo_items:
            _render_combo_card(item, output_root)
            written_combo_items.append(item)
            combo_pairs.append((item.spec.combo_kind, output_root / item.relative_path))

    if evil:
        _reset_output_dir(output_root / "cards" / "evil")
        for item in evil_items:
            _render_evil_card(item, output_root)
            written_evil_items.append(item)

    if boards:
        reference_grouped = _group_paths_by_category(reference_pairs)
        combo_grouped = _group_paths_by_category(combo_pairs)
        _reset_output_dir(output_root / "boards")
        for category, paths in reference_grouped.items():
            board_dir = output_root / "boards" / "reference"
            board_dir.mkdir(parents=True, exist_ok=True)
            for index_in_category, start in enumerate(range(0, len(paths), 4), start=1):
                board_path = board_dir / f"{category.replace('/', '_')}_{index_in_category:02d}.png"
                _compose_board(
                    paths[start : start + 4], board_path, _board_title(category, index_in_category)
                )
                board_count += 1
        for combo_kind, paths in combo_grouped.items():
            board_dir = output_root / "boards" / "combos"
            board_dir.mkdir(parents=True, exist_ok=True)
            for index_in_category, start in enumerate(range(0, len(paths), 4), start=1):
                board_path = board_dir / f"{combo_kind}_{index_in_category:02d}.png"
                _compose_board(
                    paths[start : start + 4],
                    board_path,
                    _board_title(combo_kind, index_in_category),
                )
                board_count += 1

    index_path = Path()
    if index:
        index_path = _write_index(
            output_root,
            written_strip_items,
            written_reference_items,
            written_combo_items,
            written_evil_items,
            comparison_lookup,
        )

    return GalleryAuditResult(
        output_dir=str(output_root),
        index_path=str(index_path) if index else "",
        reference_count=len(written_reference_items),
        comparison_count=len(comparison_lookup),
        combo_count=len(written_combo_items),
        evil_count=len(written_evil_items),
        board_count=board_count,
    )


def main() -> int:
    """Parse CLI arguments and build the requested artifact set.

    Returns
    -------
    int
        Process exit code.
    """

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", default="eval_output/gallery_audit")
    parser.add_argument("--cards", action="store_true")
    parser.add_argument("--boards", action="store_true")
    parser.add_argument("--comparisons", action="store_true")
    parser.add_argument("--combos", action="store_true")
    parser.add_argument("--evil", action="store_true")
    parser.add_argument("--index", action="store_true")
    parser.add_argument("--all", action="store_true")
    args = parser.parse_args()

    if args.all or not any(
        [args.cards, args.boards, args.comparisons, args.combos, args.evil, args.index]
    ):
        args.cards = True
        args.boards = True
        args.comparisons = True
        args.combos = True
        args.evil = True
        args.index = True

    result = build_gallery_audit(
        output_dir=args.output_dir,
        cards=args.cards,
        boards=args.boards,
        comparisons=args.comparisons,
        combos=args.combos,
        evil=args.evil,
        index=args.index,
    )
    print(result.output_dir)
    print(f"reference_cards={result.reference_count}")
    print(f"comparison_cards={result.comparison_count}")
    print(f"combo_cards={result.combo_count}")
    print(f"evil_cards={result.evil_count}")
    print(f"boards={result.board_count}")
    if result.index_path:
        print(result.index_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
