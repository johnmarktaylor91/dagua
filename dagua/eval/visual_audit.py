"""Visual iteration and audit suite for Dagua.

This module builds a curated set of generated artifacts meant to help with
visual iteration:

- complexity ladder from simple to complex
- renderer decomposition views
- style kill-switch matrices
- side-by-side visual diff dashboards
- typography stress sheets
- edge language sheets
- metric cards pairing layout metrics with thumbnails
- frozen baseline copies of key renders
"""

from __future__ import annotations

import copy
import json
import shutil
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
import torch

from dagua import DaguaGraph, LayoutConfig, layout
from dagua.edges import place_edge_labels, route_edges
from dagua.eval.competitors import get_available_competitors
from dagua.eval.competitors.base import CompetitorBase
from dagua.eval.graphs import get_test_graphs
from dagua.metrics import full
from dagua.render.mpl import (
    RESOLVED_FONT,
    _draw_clusters,
    _draw_edge_labels,
    _draw_edges,
    _draw_node_labels,
    _draw_nodes,
)
from dagua.styles import (
    ClusterStyle,
    EdgeStyle,
    GraphStyle,
    MINIMAL_THEME,
    NodeStyle,
    Theme,
)


@dataclass(frozen=True)
class AuditSpec:
    graph_name: str
    rung_title: str
    rationale: str
    failure_modes: Tuple[str, ...]
    direction: Optional[str] = None


@dataclass
class VisualAuditResult:
    output_dir: str
    manifest_path: str
    readme_path: str
    ladder_paths: List[str] = field(default_factory=list)
    decomposition_paths: List[str] = field(default_factory=list)
    kill_switch_paths: List[str] = field(default_factory=list)
    diff_paths: List[str] = field(default_factory=list)
    competitor_paths: List[str] = field(default_factory=list)
    sheet_paths: List[str] = field(default_factory=list)
    metric_paths: List[str] = field(default_factory=list)
    frozen_paths: List[str] = field(default_factory=list)


_LADDER_SPECS: Tuple[AuditSpec, ...] = (
    AuditSpec("linear_3layer_mlp", "Rung 1: Trivial chain", "Baseline reading order and whitespace.", ("baseline", "reading-order")),
    AuditSpec("deep_chain_20", "Rung 2: Deep chain", "Tests vertical rhythm and depth clarity.", ("depth", "spacing")),
    AuditSpec("residual_block", "Rung 3: Single skip", "The first non-trivial skip connection.", ("skip-routing", "symmetry")),
    AuditSpec("inception_block", "Rung 4: Wide parallel", "Checks branch alignment and merge balance.", ("branch-alignment", "merge-balance")),
    AuditSpec("nested_shallow_enc_dec", "Rung 5: Shallow clusters", "First hierarchy test with simple containment.", ("cluster-clarity", "containment")),
    AuditSpec("hierarchical_residual_stage", "Rung 6: Deep hierarchy", "Residuals crossing hierarchy boundaries.", ("cluster-crosstalk", "long-skips")),
    AuditSpec("transformer_layer", "Rung 7: Transformer block", "Parallel attention branches plus residuals.", ("motif-recognition", "mixed-density")),
    AuditSpec("recurrent_feedback_cell", "Rung 8: Feedback", "Late cycles and recurrent arcs.", ("feedback", "loop-clarity")),
    AuditSpec("interleaved_cluster_crosstalk", "Rung 9: Sibling crosstalk", "Interleaved cluster edges that can turn to mud.", ("cluster-crosstalk", "sibling-separation")),
    AuditSpec("kitchen_sink_hybrid_net", "Rung 10: Kitchen sink", "Everything at once for adversarial visual pressure.", ("all-of-the-above", "stress")),
)


def build_visual_audit_suite(
    output_dir: str = "eval_output/visual_audit",
    steps: int = 80,
    edge_opt_steps: int = 12,
    graph_names: Optional[Sequence[str]] = None,
) -> VisualAuditResult:
    """Build the full visual iteration suite."""
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    specs = [spec for spec in _LADDER_SPECS if graph_names is None or spec.graph_name in set(graph_names)]
    graph_map = {tg.name: tg for tg in get_test_graphs()}
    result = VisualAuditResult(
        output_dir=str(out),
        manifest_path=str(out / "visual_audit_manifest.json"),
        readme_path=str(out / "README.md"),
    )

    ladder_dir = out / "complexity_ladder"
    decomp_dir = out / "decomposition"
    kill_dir = out / "kill_switches"
    diff_dir = out / "diff_dashboard"
    competitor_dir = out / "competitor_stepwise"
    sheet_dir = out / "sheets"
    metric_dir = out / "metric_cards"
    frozen_dir = out / "frozen_baselines" / "current"
    for d in (ladder_dir, decomp_dir, kill_dir, diff_dir, competitor_dir, sheet_dir, metric_dir, frozen_dir):
        d.mkdir(parents=True, exist_ok=True)

    manifest: Dict[str, object] = {
        "ladder": [],
        "decomposition": [],
        "kill_switches": [],
        "diff_dashboard": [],
        "competitor_stepwise": [],
        "sheets": [],
        "metric_cards": [],
        "frozen_baselines": [],
    }

    metric_rows = []
    competitors = _audit_competitors()
    for spec in specs:
        tg = graph_map[spec.graph_name]
        graph = copy.deepcopy(tg.graph)
        if spec.direction:
            graph.direction = spec.direction
        cfg = LayoutConfig(steps=steps, edge_opt_steps=edge_opt_steps, direction=graph.direction, seed=42)
        pos = layout(graph, cfg)
        graph.compute_node_sizes()
        curves = route_edges(pos, graph.edge_index, graph.node_sizes, graph.direction, graph)
        label_positions = place_edge_labels(curves, pos, graph.node_sizes, graph.edge_labels, graph)
        metrics = full(
            pos,
            graph.edge_index,
            node_sizes=graph.node_sizes,
            curves=curves,
            label_positions=label_positions,
            edge_labels=graph.edge_labels,
            direction=graph.direction,
        )

        ladder_path = ladder_dir / f"{spec.graph_name}_ladder.png"
        _render_complexity_ladder(graph, pos, curves, label_positions, spec, ladder_path)
        result.ladder_paths.append(str(ladder_path))
        manifest["ladder"].append({"graph": spec.graph_name, "path": str(ladder_path), "failure_modes": list(spec.failure_modes)})

        if len(result.decomposition_paths) < 4:
            decomp_path = decomp_dir / f"{spec.graph_name}_decomposition.png"
            _render_decomposition(graph, pos, curves, label_positions, spec, decomp_path)
            result.decomposition_paths.append(str(decomp_path))
            manifest["decomposition"].append({"graph": spec.graph_name, "path": str(decomp_path)})

            kill_path = kill_dir / f"{spec.graph_name}_kill_switches.png"
            _render_kill_switch_matrix(graph, pos, curves, label_positions, spec, kill_path)
            result.kill_switch_paths.append(str(kill_path))
            manifest["kill_switches"].append({"graph": spec.graph_name, "path": str(kill_path)})

            diff_path = diff_dir / f"{spec.graph_name}_diff.png"
            _render_diff_dashboard(graph, pos, curves, label_positions, spec, diff_path)
            result.diff_paths.append(str(diff_path))
            manifest["diff_dashboard"].append({"graph": spec.graph_name, "path": str(diff_path)})

            competitor_path = competitor_dir / f"{spec.graph_name}_competitors.png"
            _render_competitor_stepwise(graph, spec, competitors, competitor_path, steps, edge_opt_steps)
            result.competitor_paths.append(str(competitor_path))
            manifest["competitor_stepwise"].append({"graph": spec.graph_name, "path": str(competitor_path)})

        thumb_rel = Path("..") / "complexity_ladder" / ladder_path.name
        metric_path = metric_dir / f"{spec.graph_name}.json"
        metric_path.write_text(json.dumps({
            "graph": spec.graph_name,
            "rung_title": spec.rung_title,
            "rationale": spec.rationale,
            "failure_modes": list(spec.failure_modes),
            "metrics": {k: _json_safe(v) for k, v in metrics.items()},
            "thumbnail": str(thumb_rel),
        }, indent=2), encoding="utf-8")
        result.metric_paths.append(str(metric_path))
        manifest["metric_cards"].append({"graph": spec.graph_name, "path": str(metric_path)})
        metric_rows.append((spec, metrics, thumb_rel))

        frozen_target = frozen_dir / ladder_path.name
        shutil.copy2(ladder_path, frozen_target)
        result.frozen_paths.append(str(frozen_target))
        manifest["frozen_baselines"].append({"graph": spec.graph_name, "path": str(frozen_target)})

    typography_path = sheet_dir / "typography_stress.png"
    _render_typography_sheet(copy.deepcopy(graph_map["extreme_mixed_width_transformer"].graph), steps, edge_opt_steps, typography_path)
    result.sheet_paths.append(str(typography_path))
    manifest["sheets"].append({"kind": "typography", "path": str(typography_path)})

    edge_sheet_path = sheet_dir / "edge_language_sheet.png"
    _render_edge_language_sheet(copy.deepcopy(graph_map["edge_label_braid"].graph), steps, edge_opt_steps, edge_sheet_path)
    result.sheet_paths.append(str(edge_sheet_path))
    manifest["sheets"].append({"kind": "edge_language", "path": str(edge_sheet_path)})

    metric_md = metric_dir / "README.md"
    metric_md.write_text(_metric_cards_markdown(metric_rows), encoding="utf-8")
    result.metric_paths.append(str(metric_md))

    Path(result.manifest_path).write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    Path(result.readme_path).write_text(_suite_readme(result, specs), encoding="utf-8")
    return result


def _render_complexity_ladder(graph, pos, curves, label_positions, spec: AuditSpec, path: Path) -> None:
    import matplotlib.pyplot as plt

    titles = ["Placement", "Nodes+Edges", "Node Labels", "Hierarchy", "Full"]
    layers = [
        ("placement",),
        ("edges", "nodes"),
        ("edges", "nodes", "node_labels"),
        ("clusters", "edges", "nodes", "node_labels"),
        ("clusters", "edges", "nodes", "node_labels", "edge_labels"),
    ]
    fig, axes = plt.subplots(1, len(layers), figsize=(18, 4.6))
    fig.patch.set_facecolor(graph.graph_style.background_color)
    for ax, title, layer_set in zip(axes, titles, layers):
        _render_layers(ax, graph, pos, curves, label_positions, layer_set, title)
    fig.suptitle(spec.rung_title, fontsize=12, fontfamily=RESOLVED_FONT, y=1.02)
    fig.text(0.01, -0.01, f"{spec.rationale}  Failure modes: {', '.join(spec.failure_modes)}", fontsize=9, fontfamily=RESOLVED_FONT)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _render_decomposition(graph, pos, curves, label_positions, spec: AuditSpec, path: Path) -> None:
    import matplotlib.pyplot as plt

    panels = [
        ("Clusters", ("clusters",)),
        ("Edges", ("edges",)),
        ("Nodes", ("nodes",)),
        ("Node Labels", ("nodes", "node_labels")),
        ("Edge Labels", ("edges", "edge_labels")),
        ("Full", ("clusters", "edges", "nodes", "node_labels", "edge_labels")),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.patch.set_facecolor(graph.graph_style.background_color)
    for ax, (title, layer_set) in zip(axes.flat, panels):
        _render_layers(ax, graph, pos, curves, label_positions, layer_set, title)
    fig.suptitle(f"Renderer decomposition — {spec.graph_name}", fontsize=12, fontfamily=RESOLVED_FONT)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _render_kill_switch_matrix(graph, pos, curves, label_positions, spec: AuditSpec, path: Path) -> None:
    import matplotlib.pyplot as plt

    variants = [
        ("Full", graph, curves, label_positions),
        ("No Labels", _variant_no_labels(graph), curves, [None] * len(curves)),
        ("No Clusters", _variant_no_clusters(graph), curves, label_positions),
        ("Neutral Theme", _variant_neutral(graph), curves, label_positions),
        ("Placement Only", graph, curves, label_positions),
        ("No Edge Labels", _variant_no_edge_labels(graph), curves, [None] * len(curves)),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8))
    fig.patch.set_facecolor(graph.graph_style.background_color)
    for ax, (title, variant_graph, variant_curves, variant_lps) in zip(axes.flat, variants):
        if title == "Placement Only":
            _render_layers(ax, variant_graph, pos, variant_curves, variant_lps, ("placement",), title)
        else:
            _render_layers(ax, variant_graph, pos, variant_curves, variant_lps, ("clusters", "edges", "nodes", "node_labels", "edge_labels"), title)
    fig.suptitle(f"Kill switches — {spec.graph_name}", fontsize=12, fontfamily=RESOLVED_FONT)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _render_diff_dashboard(graph, pos, curves, label_positions, spec: AuditSpec, path: Path) -> None:
    import matplotlib.pyplot as plt

    current = _render_to_array(graph, pos, curves, label_positions, ("clusters", "edges", "nodes", "node_labels", "edge_labels"), "Current")
    neutral_graph = _variant_neutral(graph)
    neutral = _render_to_array(neutral_graph, pos, curves, label_positions, ("clusters", "edges", "nodes", "node_labels", "edge_labels"), "Neutral")
    placement = _render_to_array(graph, pos, curves, label_positions, ("placement",), "Placement")
    diff = np.abs(current.astype(np.int16) - neutral.astype(np.int16)).astype(np.uint8)

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    for ax, img, title in zip(
        axes.flat,
        (current, neutral, diff, placement),
        ("Current", "Neutral", "Absolute diff", "Placement only"),
    ):
        ax.imshow(img)
        ax.set_title(title, fontfamily=RESOLVED_FONT)
        ax.axis("off")
    fig.suptitle(f"Diff dashboard — {spec.graph_name}", fontsize=12, fontfamily=RESOLVED_FONT)
    fig.tight_layout()
    fig.savefig(path, dpi=160, bbox_inches="tight")
    plt.close(fig)


def _render_typography_sheet(graph: DaguaGraph, steps: int, edge_opt_steps: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    cfg = LayoutConfig(steps=steps, edge_opt_steps=edge_opt_steps, direction=graph.direction, seed=42)
    pos = layout(graph, cfg)
    graph.compute_node_sizes()
    curves = route_edges(pos, graph.edge_index, graph.node_sizes, graph.direction, graph)
    lps = place_edge_labels(curves, pos, graph.node_sizes, graph.edge_labels, graph)

    variants = [
        ("Compact", _variant_scale_type(graph, 0.85), curves, lps),
        ("Baseline", graph, curves, lps),
        ("Large", _variant_scale_type(graph, 1.18), curves, lps),
        ("Muted secondary", _variant_secondary_scale(graph, 0.68), curves, lps),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (title, vg, vc, vl) in zip(axes.flat, variants):
        _render_layers(ax, vg, pos, vc, vl, ("clusters", "edges", "nodes", "node_labels", "edge_labels"), title)
    fig.suptitle("Typography stress sheet", fontsize=12, fontfamily=RESOLVED_FONT)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _render_edge_language_sheet(graph: DaguaGraph, steps: int, edge_opt_steps: int, path: Path) -> None:
    import matplotlib.pyplot as plt

    cfg = LayoutConfig(steps=steps, edge_opt_steps=edge_opt_steps, direction=graph.direction, seed=42)
    pos = layout(graph, cfg)
    graph.compute_node_sizes()

    variants = [
        ("Straight / quiet", _variant_edge_language(graph, routing="straight", curvature=0.0, opacity=0.45, width=0.9)),
        ("Bezier / baseline", _variant_edge_language(graph, routing="bezier", curvature=0.35, opacity=0.68, width=1.15)),
        ("Ortho / crisp", _variant_edge_language(graph, routing="ortho", curvature=0.1, opacity=0.62, width=1.05)),
        ("Bezier / expressive", _variant_edge_language(graph, routing="bezier", curvature=0.6, opacity=0.78, width=1.35)),
    ]
    fig, axes = plt.subplots(2, 2, figsize=(12, 9))
    for ax, (title, vg) in zip(axes.flat, variants):
        vc = route_edges(pos, vg.edge_index, vg.node_sizes if vg.node_sizes is not None else graph.node_sizes, vg.direction, vg)
        vl = place_edge_labels(vc, pos, vg.node_sizes if vg.node_sizes is not None else graph.node_sizes, vg.edge_labels, vg)
        _render_layers(ax, vg, pos, vc, vl, ("clusters", "edges", "nodes", "node_labels", "edge_labels"), title)
    fig.suptitle("Edge language sheet", fontsize=12, fontfamily=RESOLVED_FONT)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _render_layers(ax, graph, positions, curves, label_positions, layers: Sequence[str], title: str) -> None:
    pos = positions.detach().cpu().numpy()
    graph.compute_node_sizes()
    sizes = graph.node_sizes.detach().cpu().numpy()
    gs = graph.graph_style
    bg = gs.background_color
    margin = gs.margin
    x_min = (pos[:, 0] - sizes[:, 0] / 2).min() - margin
    x_max = (pos[:, 0] + sizes[:, 0] / 2).max() + margin
    y_min = (pos[:, 1] - sizes[:, 1] / 2).min() - margin
    y_max = (pos[:, 1] + sizes[:, 1] / 2).max() + margin
    ax.set_facecolor(bg)

    if "clusters" in layers:
        _draw_clusters(ax, graph, pos, sizes)
    if "edges" in layers:
        _draw_edges(ax, graph, curves)
    if "nodes" in layers:
        _draw_nodes(ax, graph, pos, sizes)
    if "placement" in layers:
        ax.scatter(pos[:, 0], pos[:, 1], s=32, c="#2563EB", alpha=0.9, zorder=3)
        if graph.num_nodes <= 24:
            for i, (x, y) in enumerate(pos):
                ax.text(x, y - 6, str(graph.node_labels[i]), ha="center", va="top", fontsize=7, fontfamily=RESOLVED_FONT)
    if "node_labels" in layers:
        _draw_node_labels(ax, graph, pos, sizes)
    if "edge_labels" in layers:
        _draw_edge_labels(ax, graph, curves, label_positions=label_positions)

    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=10, fontfamily=RESOLVED_FONT)


def _render_to_array(graph, positions, curves, label_positions, layers: Sequence[str], title: str) -> np.ndarray:
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(1, 1, figsize=(5.6, 4.2))
    fig.patch.set_facecolor(graph.graph_style.background_color)
    _render_layers(ax, graph, positions, curves, label_positions, layers, title)
    fig.tight_layout()
    fig.canvas.draw()
    arr = np.asarray(fig.canvas.buffer_rgba())[..., :3].copy()
    plt.close(fig)
    return arr


def _variant_no_labels(graph: DaguaGraph) -> DaguaGraph:
    g = copy.deepcopy(graph)
    g.edge_labels = [None] * len(g.edge_labels)
    g.node_labels = [label.split("\n")[0] if isinstance(label, str) else label for label in g.node_labels]
    return g


def _variant_no_edge_labels(graph: DaguaGraph) -> DaguaGraph:
    g = copy.deepcopy(graph)
    g.edge_labels = [None] * len(g.edge_labels)
    return g


def _variant_no_clusters(graph: DaguaGraph) -> DaguaGraph:
    g = copy.deepcopy(graph)
    g.clusters = {}
    g.cluster_labels = {}
    g.cluster_styles = {}
    g.cluster_parents = {}
    return g


def _variant_neutral(graph: DaguaGraph) -> DaguaGraph:
    g = copy.deepcopy(graph)
    neutral = copy.deepcopy(MINIMAL_THEME)
    neutral.graph_style = GraphStyle(
        background_color="#F8F8F7",
        margin=18.0,
        title_font_size=9.0,
        title_font_color="#111827",
        edge_label_font_size=6.5,
        edge_label_background="#F8F8F7",
        edge_label_background_opacity=0.8,
        node_label_secondary_scale=0.74,
    )
    neutral.cluster_style = ClusterStyle(fill="#F1F3F5", stroke="#C7CDD5", opacity=0.18, stroke_width=0.7)
    neutral.node_styles["default"] = NodeStyle(
        fill="#FCFCFB",
        stroke="#374151",
        font_color="#111827",
        font_size=7.8,
        shape="roundrect",
        padding=(7.0, 4.5),
    )
    neutral.edge_styles["default"] = EdgeStyle(color="#6B7280", width=1.0, opacity=0.58, curvature=0.22)
    g._theme = neutral
    return g


def _variant_scale_type(graph: DaguaGraph, scale: float) -> DaguaGraph:
    g = copy.deepcopy(graph)
    g._theme.graph_style.node_label_secondary_scale *= scale if scale < 1 else min(scale, 1.0)
    for i, style in enumerate(g.node_styles):
        if style is not None:
            g.node_styles[i] = copy.deepcopy(style)
            g.node_styles[i].font_size *= scale
    for i, style in enumerate(g.edge_styles):
        if style is not None:
            g.edge_styles[i] = copy.deepcopy(style)
            g.edge_styles[i].label_font_size *= scale
    g.node_sizes = None
    return g


def _variant_secondary_scale(graph: DaguaGraph, scale: float) -> DaguaGraph:
    g = copy.deepcopy(graph)
    g._theme.graph_style.node_label_secondary_scale = scale
    g.node_sizes = None
    return g


def _variant_edge_language(graph: DaguaGraph, routing: str, curvature: float, opacity: float, width: float) -> DaguaGraph:
    g = copy.deepcopy(graph)
    for i in range(len(g.edge_styles)):
        style = copy.deepcopy(g.edge_styles[i]) if g.edge_styles[i] is not None else EdgeStyle()
        style.routing = routing
        style.curvature = curvature
        style.opacity = opacity
        style.width = width
        g.edge_styles[i] = style
    return g


def _metric_cards_markdown(rows: Sequence[Tuple[AuditSpec, Dict[str, float], Path]]) -> str:
    lines = ["# Metric Cards", ""]
    for spec, metrics, thumb_rel in rows:
        lines.append(f"## {spec.rung_title}")
        lines.append(f"![{spec.graph_name}]({thumb_rel.as_posix()})")
        lines.append("")
        lines.append(f"- Graph: `{spec.graph_name}`")
        lines.append(f"- Rationale: {spec.rationale}")
        lines.append(f"- Failure modes: {', '.join(spec.failure_modes)}")
        for key in ("overall_quality", "dag_consistency", "edge_length_cv", "overlap_count", "edge_crossings", "edge_node_crossings", "label_overlaps"):
            if key in metrics:
                lines.append(f"- `{key}`: {metrics[key]:.4f}" if isinstance(metrics[key], float) else f"- `{key}`: {metrics[key]}")
        lines.append("")
    return "\n".join(lines) + "\n"


def _suite_readme(result: VisualAuditResult, specs: Sequence[AuditSpec]) -> str:
    return (
        "# Visual Audit Suite\n\n"
        "This directory is the visual iteration toolkit for Dagua. It is meant to be used like a visual unit-test surface.\n\n"
        "Contents:\n"
        "- `complexity_ladder/`: simple-to-complex progression with staged render passes\n"
        "- `decomposition/`: renderer layer breakdowns\n"
        "- `kill_switches/`: quick isolation of labels/clusters/theme effects\n"
        "- `diff_dashboard/`: current vs neutral vs placement-only views\n"
        "- `competitor_stepwise/`: the same stepwise graphs shown side by side with competing engines\n"
        "- `sheets/`: typography and edge-language stress sheets\n"
        "- `metric_cards/`: metrics paired with thumbnails and failure-mode tags\n"
        "- `frozen_baselines/current/`: current frozen key renders for future comparison\n\n"
        f"Graphs covered: {', '.join(spec.graph_name for spec in specs)}\n"
    )


def _json_safe(value):
    if isinstance(value, (np.floating,)):
        return float(value)
    if isinstance(value, (np.integer,)):
        return int(value)
    return value


def _audit_competitors() -> List[CompetitorBase]:
    preferred = {"dagua", "graphviz_dot", "elk_layered", "dagre", "graphviz_sfdp", "nx_spring"}
    competitors = [c for c in get_available_competitors() if c.name in preferred]
    order = ["dagua", "graphviz_dot", "elk_layered", "dagre", "graphviz_sfdp", "nx_spring"]
    competitors.sort(key=lambda c: order.index(c.name) if c.name in order else 999)
    return competitors


def _render_competitor_stepwise(
    graph: DaguaGraph,
    spec: AuditSpec,
    competitors: Sequence[CompetitorBase],
    path: Path,
    steps: int,
    edge_opt_steps: int,
) -> None:
    import matplotlib.pyplot as plt

    if not competitors:
        return

    positions: Dict[str, Optional[torch.Tensor]] = {}
    graph.compute_node_sizes()
    dagua_cfg = LayoutConfig(steps=steps, edge_opt_steps=edge_opt_steps, direction=graph.direction, seed=42)
    positions["dagua"] = layout(copy.deepcopy(graph), dagua_cfg)

    for competitor in competitors:
        if competitor.name == "dagua":
            continue
        if graph.num_nodes > competitor.max_nodes:
            positions[competitor.name] = None
            continue
        try:
            result = competitor.layout(copy.deepcopy(graph), timeout=120.0)
            positions[competitor.name] = result.pos
        except Exception:
            positions[competitor.name] = None

    ordered_names = [c.name for c in competitors if c.name in positions]
    cols = min(3, max(1, len(ordered_names)))
    rows = (len(ordered_names) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.8, rows * 4.0))
    axes_list = _flatten_axes(axes)
    fig.patch.set_facecolor(graph.graph_style.background_color)

    for ax, comp_name in zip(axes_list, ordered_names):
        pos = positions.get(comp_name)
        if pos is None:
            _status_panel(ax, comp_name, "N/A")
            continue
        norm_pos = _normalize_positions_for_audit(pos, graph.node_sizes)
        curves = route_edges(norm_pos, graph.edge_index, graph.node_sizes, graph.direction, graph)
        label_positions = place_edge_labels(curves, norm_pos, graph.node_sizes, graph.edge_labels, graph)
        _render_layers(
            ax,
            graph,
            norm_pos,
            curves,
            label_positions,
            ("clusters", "edges", "nodes", "node_labels", "edge_labels"),
            comp_name,
        )
    for ax in axes_list[len(ordered_names):]:
        ax.axis("off")
    fig.suptitle(f"Competitor stepwise — {spec.graph_name}", fontsize=12, fontfamily=RESOLVED_FONT)
    fig.tight_layout()
    fig.savefig(path, dpi=170, bbox_inches="tight")
    plt.close(fig)


def _flatten_axes(axes) -> List:
    if isinstance(axes, np.ndarray):
        return [ax for ax in axes.flat]
    return [axes]


def _status_panel(ax, title: str, subtitle: str) -> None:
    ax.set_facecolor("#F3F4F6")
    ax.text(0.5, 0.58, title, ha="center", va="center", transform=ax.transAxes, fontsize=11, color="#374151")
    ax.text(0.5, 0.42, subtitle, ha="center", va="center", transform=ax.transAxes, fontsize=8, color="#6B7280")
    ax.axis("off")


def _normalize_positions_for_audit(
    positions: torch.Tensor,
    node_sizes: torch.Tensor,
    target_width: float = 600.0,
    target_height: float = 420.0,
    padding: float = 30.0,
) -> torch.Tensor:
    pos = positions.detach().cpu().clone()
    sizes = node_sizes.detach().cpu()
    x_min = (pos[:, 0] - sizes[:, 0] / 2).min()
    x_max = (pos[:, 0] + sizes[:, 0] / 2).max()
    y_min = (pos[:, 1] - sizes[:, 1] / 2).min()
    y_max = (pos[:, 1] + sizes[:, 1] / 2).max()
    width = max((x_max - x_min).item(), 1.0)
    height = max((y_max - y_min).item(), 1.0)
    scale = min((target_width - 2 * padding) / width, (target_height - 2 * padding) / height)
    pos[:, 0] = (pos[:, 0] - x_min) * scale + padding
    pos[:, 1] = (pos[:, 1] - y_min) * scale + padding
    return pos
