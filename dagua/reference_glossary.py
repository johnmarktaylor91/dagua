"""Generated glossary/reference builder for Dagua.

Builds a reference manual with:
- public API signatures
- config/data-structure fields
- loss functions
- metric functions and metric keys
- glossary terms and optimization stages
- generated explanatory visuals
- LaTeX source and optional PDF
"""

from __future__ import annotations

import dataclasses
import inspect
import json
import shutil
import subprocess
from dataclasses import MISSING, dataclass, fields, is_dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch

import dagua as dagua_api
from dagua import DaguaGraph, LayoutConfig
from dagua.animation import AnimationConfig, PosterConfig, TourConfig
from dagua.config import PARAM_REGISTRY
from dagua.edges import place_edge_labels, route_edges
from dagua.eval.graphs import get_test_graphs
from dagua.flex import AlignGroup, Flex, LayoutFlex
from dagua.layout import constraints as constraint_mod
from dagua.layout import edge_optimization as edge_opt_mod
from dagua.layout import layout
from dagua.metrics import full, quick
from dagua.render.mpl import (
    _draw_clusters,
    _draw_edge_labels,
    _draw_edges,
    _draw_node_labels,
    _draw_nodes,
)
from dagua.styles import ClusterStyle, EdgeStyle, GraphStyle, NodeStyle


@dataclass
class GlossaryBuildResult:
    output_dir: str
    tex_path: str
    pdf_path: Optional[str]
    manifest_path: str
    figures_dir: str


_BASICS = [
    ("DAG", "Directed acyclic graph. Edges encode direction, and there are no directed cycles."),
    ("Rank", "A layer index assigned during hierarchical layout. Nodes in the same rank are peers."),
    ("Cluster", "A visual grouping box around related nodes, optionally nested."),
    ("Port", "The attachment location where an edge exits or enters a node."),
    ("Routing", "The post-layout step that turns node positions into edge curves."),
    ("Projection", "A hard corrective step used to enforce constraints like overlap removal or pins."),
    ("Coarsening", "Compression of the graph into a smaller hierarchy level before layout."),
    ("Prolongation", "Expansion of a coarse solution back to a finer graph level."),
    ("Refinement", "Short optimization passes applied after prolongation."),
    ("Flex", "A soft target with a weight that biases layout without forcing a hard lock."),
    ("Pin", "A node position target, usually hard when weight is infinite."),
    ("Tier-1 metric", "A metric cheap enough to compute routinely at any scale."),
    ("Tier-2 metric", "A sampled or costlier metric used when feasible."),
    ("Tier-3 metric", "A hierarchy- or DAG-specific metric that relies on structural metadata."),
]

_STAGES = [
    ("Graph Construction", "Nodes, edges, labels, clusters, and styles are assembled into a DaguaGraph."),
    ("Node Sizing", "Text measurement and padding determine node bounding boxes before layout."),
    ("Layering", "Topological depth is converted into ordered ranks consistent with graph direction."),
    ("Coarsening", "Large graphs are compressed into smaller layer-aware hierarchy levels."),
    ("Coarse Optimization", "The smallest hierarchy level is optimized first to establish global structure."),
    ("Prolongation + Refinement", "The coarse layout is expanded and refined level by level."),
    ("Edge Routing", "Node positions are converted into routed edge curves and ports."),
    ("Edge Optimization", "Bezier control points are refined to improve crossings and curvature."),
    ("Label Placement", "Edge labels are placed after routing to avoid collisions where possible."),
    ("Rendering", "Nodes, clusters, edges, and labels are turned into raster/vector output."),
]

_VISUAL_CAPTIONS = {
    "pipeline_stages": "Pipeline schematic from graph construction through rendering.",
    "direction_modes": "Direction modes change the global flow convention while preserving structure.",
    "spacing_sweep": "Node and rank separation alter whitespace and breathing room.",
    "crossing_sweep": "Crossing weight trades local compactness for reduced edge tangles.",
    "routing_styles": "Edge routing style and curvature change the visual character of the same topology.",
    "flex_constraints": "Pins and alignment constraints visibly reshape the same graph.",
}


def _format_annotation(annotation: Any) -> str:
    if annotation is inspect._empty:
        return "Any"
    if getattr(annotation, "__module__", "") == "builtins":
        return annotation.__name__
    return str(annotation).replace("typing.", "")


def _format_default(field: dataclasses.Field) -> str:
    if field.default is not MISSING:
        return repr(field.default)
    if field.default_factory is not MISSING:  # type: ignore[comparison-overlap]
        return f"{field.default_factory.__name__}()"  # type: ignore[union-attr]
    return "<required>"


def _public_api_entries() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for name in dagua_api.__all__:
        obj = getattr(dagua_api, name, None)
        if obj is None:
            continue
        if inspect.isfunction(obj):
            entries.append(
                {
                    "name": name,
                    "signature": f"{name}{inspect.signature(obj)}",
                    "summary": (inspect.getdoc(obj) or "").splitlines()[0] if inspect.getdoc(obj) else "",
                    "source": inspect.getmodule(obj).__name__ if inspect.getmodule(obj) else "",
                }
            )
    return sorted(entries, key=lambda item: item["name"])


def _graph_method_entries() -> List[Dict[str, str]]:
    entries: List[Dict[str, str]] = []
    for name, obj in inspect.getmembers(DaguaGraph, inspect.isfunction):
        if name.startswith("_"):
            continue
        if inspect.getmodule(obj) is None or inspect.getmodule(obj).__name__ != "dagua.graph":
            continue
        doc = inspect.getdoc(obj) or ""
        entries.append(
            {
                "name": name,
                "signature": f"{name}{inspect.signature(obj)}",
                "summary": doc.splitlines()[0] if doc else "",
            }
        )
    return sorted(entries, key=lambda item: item["name"])


def _dataclass_sections() -> List[Tuple[str, Any]]:
    return [
        ("LayoutConfig", LayoutConfig),
        ("NodeStyle", NodeStyle),
        ("EdgeStyle", EdgeStyle),
        ("ClusterStyle", ClusterStyle),
        ("GraphStyle", GraphStyle),
        ("AnimationConfig", AnimationConfig),
        ("PosterConfig", PosterConfig),
        ("TourConfig", TourConfig),
        ("Flex", Flex),
        ("LayoutFlex", LayoutFlex),
        ("AlignGroup", AlignGroup),
    ]


def _dataclass_field_entries(cls: Any) -> List[Dict[str, str]]:
    result = []
    for field in fields(cls):
        result.append(
            {
                "name": field.name,
                "type": _format_annotation(field.type),
                "default": _format_default(field),
            }
        )
    return result


def _loss_entries() -> List[Dict[str, str]]:
    entries = []
    for module in (constraint_mod, edge_opt_mod):
        for name, fn in inspect.getmembers(module, inspect.isfunction):
            if "loss" not in name or name.startswith("__"):
                continue
            doc = inspect.getdoc(fn) or ""
            entries.append(
                {
                    "name": name,
                    "signature": f"{name}{inspect.signature(fn)}",
                    "summary": doc.splitlines()[0] if doc else "",
                    "source": module.__name__,
                }
            )
    return sorted(entries, key=lambda item: item["name"])


def _metric_entries() -> Tuple[List[Dict[str, str]], List[str], List[str]]:
    import dagua.metrics as metrics_mod

    entries = []
    for name, fn in inspect.getmembers(metrics_mod, inspect.isfunction):
        if name.startswith("_"):
            continue
        if inspect.getmodule(fn) is not metrics_mod:
            continue
        doc = inspect.getdoc(fn) or ""
        entries.append(
            {
                "name": name,
                "signature": f"{name}{inspect.signature(fn)}",
                "summary": doc.splitlines()[0] if doc else "",
            }
        )

    sample_graph = DaguaGraph.from_edge_list([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")])
    sample_graph.compute_node_sizes()
    pos = layout(sample_graph, LayoutConfig(steps=12, edge_opt_steps=-1, seed=42))
    quick_keys = sorted(quick(pos, sample_graph.edge_index, node_sizes=sample_graph.node_sizes).keys())
    full_keys = sorted(
        full(
            pos,
            sample_graph.edge_index,
            topo_depth=torch.tensor([0, 1, 1, 2]),
            node_sizes=sample_graph.node_sizes,
            cluster_ids=sample_graph.cluster_ids,
        ).keys()
    )
    return sorted(entries, key=lambda item: item["name"]), quick_keys, full_keys


def _hyperparameter_entries() -> List[Dict[str, str]]:
    return [
        {
            "name": param.name,
            "display_name": param.display_name,
            "default": repr(param.default),
            "category": param.category,
            "description": param.description,
            "visual_effect": param.visual_effect,
        }
        for param in PARAM_REGISTRY
    ]


def _render_graph_on_ax(ax, graph, pos: torch.Tensor, title: str) -> None:
    graph.compute_node_sizes()
    pos_np = pos.detach().cpu().numpy()
    sizes = graph.node_sizes.detach().cpu().numpy()
    curves = route_edges(pos, graph.edge_index, graph.node_sizes, graph.direction, graph)
    label_positions = place_edge_labels(curves, pos, graph.node_sizes, graph.edge_labels, graph)
    _draw_clusters(ax, graph, pos_np, sizes)
    _draw_edges(ax, graph, curves)
    _draw_nodes(ax, graph, pos_np, sizes)
    _draw_node_labels(ax, graph, pos_np, sizes)
    _draw_edge_labels(ax, graph, curves, label_positions)
    x_min = (pos_np[:, 0] - sizes[:, 0] / 2).min() - 20
    x_max = (pos_np[:, 0] + sizes[:, 0] / 2).max() + 20
    y_min = (pos_np[:, 1] - sizes[:, 1] / 2).min() - 20
    y_max = (pos_np[:, 1] + sizes[:, 1] / 2).max() + 20
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect("equal")
    ax.axis("off")
    ax.set_title(title, fontsize=9)


def _sample_graphs() -> Dict[str, DaguaGraph]:
    lookup = {tg.name: tg.graph for tg in get_test_graphs(max_nodes=120)}
    return {
        "residual": lookup["residual_block"],
        "clustered": lookup["nested_shallow_enc_dec"],
        "skip": lookup["dense_skip_200"] if "dense_skip_200" in lookup else DaguaGraph.from_edge_list([("a", "b"), ("a", "c"), ("b", "d"), ("c", "d")]),
    }


def _visual_pipeline(path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(9, 2.8), dpi=180)
    ax.axis("off")
    stages = ["Graph", "Layering", "Coarsen", "Optimize", "Refine", "Route", "Render"]
    x_positions = [0.05, 0.19, 0.33, 0.47, 0.61, 0.75, 0.89]
    for x, stage in zip(x_positions, stages):
        ax.add_patch(plt.Rectangle((x - 0.055, 0.35), 0.11, 0.28, facecolor="#EAF4F8", edgecolor="#2F7EA8", linewidth=1.2))
        ax.text(x, 0.49, stage, ha="center", va="center", fontsize=10)
    for left, right in zip(x_positions[:-1], x_positions[1:]):
        ax.annotate("", xy=(right - 0.06, 0.49), xytext=(left + 0.06, 0.49), arrowprops=dict(arrowstyle="->", color="#5F6368"))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _visual_direction(path: Path, steps: int) -> None:
    g = DaguaGraph.from_edge_list([("input", "stem"), ("stem", "left"), ("stem", "right"), ("left", "merge"), ("right", "merge"), ("merge", "head")])
    directions = ["TB", "BT", "LR", "RL"]
    fig, axes = plt.subplots(2, 2, figsize=(8, 6), dpi=180)
    for ax, direction in zip(axes.flat, directions):
        pos = layout(g, LayoutConfig(steps=steps, edge_opt_steps=-1, direction=direction, seed=42))
        _render_graph_on_ax(ax, g, pos, direction)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _visual_spacing(path: Path, steps: int) -> None:
    g = DaguaGraph.from_edge_list([("a", "c"), ("b", "c"), ("c", "d"), ("c", "e"), ("d", "f"), ("e", "f")])
    configs = [("Tight", 14, 28), ("Default", 28, 50), ("Wide", 50, 90)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=180)
    for ax, (title, node_sep, rank_sep) in zip(axes.flat, configs):
        pos = layout(g, LayoutConfig(steps=steps, edge_opt_steps=-1, node_sep=node_sep, rank_sep=rank_sep, seed=42))
        _render_graph_on_ax(ax, g, pos, title)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _visual_crossing(path: Path, steps: int) -> None:
    g = DaguaGraph.from_edge_list(
        [
            ("in1", "m1"),
            ("in2", "m2"),
            ("m1", "out2"),
            ("m2", "out1"),
            ("in1", "out1"),
            ("in2", "out2"),
        ]
    )
    configs = [("Low crossing", 0.3), ("Default", 1.8), ("High crossing", 4.0)]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=180)
    for ax, (title, weight) in zip(axes.flat, configs):
        pos = layout(g, LayoutConfig(steps=steps, edge_opt_steps=-1, w_crossing=weight, seed=42))
        _render_graph_on_ax(ax, g, pos, title)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _visual_routing(path: Path, steps: int) -> None:
    base = DaguaGraph.from_edge_list([("input", "fork"), ("fork", "a"), ("fork", "b"), ("a", "merge"), ("b", "merge"), ("merge", "out")])
    configs = [
        ("Bezier", EdgeStyle(routing="bezier", curvature=0.4)),
        ("Straight", EdgeStyle(routing="straight", curvature=0.0)),
        ("Orthogonal", EdgeStyle(routing="ortho", curvature=0.0)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=180)
    for ax, (title, edge_style) in zip(axes.flat, configs):
        g = DaguaGraph.from_json(base.to_json())
        g.default_edge_style = edge_style
        pos = layout(g, LayoutConfig(steps=steps, edge_opt_steps=-1, seed=42))
        _render_graph_on_ax(ax, g, pos, title)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _visual_flex(path: Path, steps: int) -> None:
    g = DaguaGraph.from_edge_list([("in", "a"), ("in", "b"), ("a", "out"), ("b", "out")])
    cfg_default = LayoutConfig(steps=steps, edge_opt_steps=-1, seed=42)
    cfg_flex = LayoutConfig(
        steps=steps,
        edge_opt_steps=-1,
        seed=42,
        flex=LayoutFlex(
            pins={"in": (Flex.locked(0), Flex.locked(0))},
            align_y=[AlignGroup(["a", "b"], weight=5.0)],
            rank_sep=Flex.firm(80),
        ),
    )
    fig, axes = plt.subplots(1, 2, figsize=(8, 3.6), dpi=180)
    _render_graph_on_ax(axes[0], g, layout(g, cfg_default), "Default")
    _render_graph_on_ax(axes[1], g, layout(g, cfg_flex), "Pinned + aligned")
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _render_visuals(figures_dir: Path, steps: int) -> Dict[str, str]:
    figures_dir.mkdir(parents=True, exist_ok=True)
    visuals = {
        "pipeline_stages": figures_dir / "pipeline_stages.png",
        "direction_modes": figures_dir / "direction_modes.png",
        "spacing_sweep": figures_dir / "spacing_sweep.png",
        "crossing_sweep": figures_dir / "crossing_sweep.png",
        "routing_styles": figures_dir / "routing_styles.png",
        "flex_constraints": figures_dir / "flex_constraints.png",
    }
    _visual_pipeline(visuals["pipeline_stages"])
    _visual_direction(visuals["direction_modes"], steps)
    _visual_spacing(visuals["spacing_sweep"], steps)
    _visual_crossing(visuals["crossing_sweep"], steps)
    _visual_routing(visuals["routing_styles"], steps)
    _visual_flex(visuals["flex_constraints"], steps)
    return {k: str(v) for k, v in visuals.items()}


def _latex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("$", "\\$")
        .replace("#", "\\#")
        .replace("_", "\\_")
        .replace("{", "\\{")
        .replace("}", "\\}")
    )


def _render_tex(output_dir: Path, visuals: Dict[str, str], sample_steps: int) -> str:
    api_entries = _public_api_entries()
    loss_entries = _loss_entries()
    metric_entries, quick_keys, full_keys = _metric_entries()
    hyper_entries = _hyperparameter_entries()

    sections = []
    sections.append("\\section{Basics and Terminology}\n\\begin{description}")
    for name, desc in _BASICS:
        sections.append(f"\\item[{_latex_escape(name)}] {_latex_escape(desc)}")
    sections.append("\\end{description}")

    sections.append("\\section{Optimization Stages}\n\\begin{description}")
    for name, desc in _STAGES:
        sections.append(f"\\item[{_latex_escape(name)}] {_latex_escape(desc)}")
    sections.append("\\end{description}")

    sections.append("\\section{Public API Functions}\n\\begin{longtable}{p{0.22\\linewidth}p{0.73\\linewidth}}")
    sections.append("\\toprule Name & Signature and summary \\\\ \\midrule")
    for entry in api_entries:
        rhs = f"\\texttt{{{_latex_escape(entry['signature'])}}}\\\\{_latex_escape(entry['summary'])}"
        sections.append(f"{_latex_escape(entry['name'])} & {rhs} \\\\")
    sections.append("\\bottomrule\\end{longtable}")

    graph_methods = _graph_method_entries()

    sections.append("\\section{Graph Construction and Orchestration Methods}")
    sections.append("\\begin{longtable}{p{0.24\\linewidth}p{0.71\\linewidth}}")
    sections.append("\\toprule Method & Signature and summary \\\\ \\midrule")
    for entry in graph_methods:
        rhs = f"\\texttt{{{_latex_escape(entry['signature'])}}}\\\\{_latex_escape(entry['summary'])}"
        sections.append(f"{_latex_escape(entry['name'])} & {rhs} \\\\")
    sections.append("\\bottomrule\\end{longtable}")

    sections.append("\\section{Configs and Data Structures}")
    for title, cls in _dataclass_sections():
        sections.append(f"\\subsection{{{_latex_escape(title)}}}")
        if inspect.getdoc(cls):
            sections.append(_latex_escape(inspect.getdoc(cls).splitlines()[0]))
        sections.append("\\begin{longtable}{p{0.24\\linewidth}p{0.24\\linewidth}p{0.18\\linewidth}p{0.26\\linewidth}}")
        sections.append("\\toprule Field & Type & Default & Notes \\\\ \\midrule")
        for entry in _dataclass_field_entries(cls):
            sections.append(
                f"{_latex_escape(entry['name'])} & {_latex_escape(entry['type'])} & "
                f"\\texttt{{{_latex_escape(entry['default'])}}} & {_latex_escape(title)} field \\\\"
            )
        sections.append("\\bottomrule\\end{longtable}")

    sections.append("\\section{Loss Functions}\\begin{longtable}{p{0.28\\linewidth}p{0.67\\linewidth}}")
    sections.append("\\toprule Loss & Signature and summary \\\\ \\midrule")
    for entry in loss_entries:
        rhs = f"\\texttt{{{_latex_escape(entry['signature'])}}}\\\\{_latex_escape(entry['summary'])}"
        sections.append(f"{_latex_escape(entry['name'])} & {rhs} \\\\")
    sections.append("\\bottomrule\\end{longtable}")

    sections.append("\\section{Metric Functions}\\begin{longtable}{p{0.24\\linewidth}p{0.71\\linewidth}}")
    sections.append("\\toprule Metric & Signature and summary \\\\ \\midrule")
    for entry in metric_entries:
        rhs = f"\\texttt{{{_latex_escape(entry['signature'])}}}\\\\{_latex_escape(entry['summary'])}"
        sections.append(f"{_latex_escape(entry['name'])} & {rhs} \\\\")
    sections.append("\\bottomrule\\end{longtable}")
    sections.append("\\subsection{Metric Keys from quick()}")
    sections.append("\\texttt{" + _latex_escape(", ".join(quick_keys)) + "}")
    sections.append("\\subsection{Metric Keys from full()}")
    sections.append("\\texttt{" + _latex_escape(", ".join(full_keys)) + "}")

    sections.append("\\section{Optimization Hyperparameters}\\begin{longtable}{p{0.2\\linewidth}p{0.2\\linewidth}p{0.12\\linewidth}p{0.4\\linewidth}}")
    sections.append("\\toprule Name & Display name & Category & Description \\\\ \\midrule")
    for entry in hyper_entries:
        sections.append(
            f"{_latex_escape(entry['name'])} & {_latex_escape(entry['display_name'])} & "
            f"{_latex_escape(entry['category'])} & {_latex_escape(entry['description'])} \\\\"
        )
    sections.append("\\bottomrule\\end{longtable}")

    sections.append("\\section{Visual Guide}")
    for key, caption in _VISUAL_CAPTIONS.items():
        fig_rel = Path("figures") / Path(visuals[key]).name
        sections.append("\\begin{figure}[h]\\centering")
        sections.append(f"\\includegraphics[width=0.92\\linewidth]{{{fig_rel.as_posix()}}}")
        sections.append(f"\\caption{{{_latex_escape(caption)}}}")
        sections.append("\\end{figure}")

    tex = f"""
\\documentclass[11pt]{{article}}
\\usepackage[margin=1in]{{geometry}}
\\usepackage{{graphicx}}
\\usepackage{{booktabs}}
\\usepackage{{longtable}}
\\usepackage{{hyperref}}
\\title{{Dagua Exhaustive Glossary and Reference}}
\\date{{Generated automatically}}
\\begin{{document}}
\\maketitle
\\tableofcontents
\\paragraph{{Generation note.}} This reference was generated from the codebase, curated glossary metadata, and scripted visuals. Sample visuals use {sample_steps} layout steps to keep rebuilds practical.
{''.join(sections)}
\\end{{document}}
"""
    tex_path = output_dir / "dagua_glossary.tex"
    tex_path.write_text(tex, encoding="utf-8")
    return str(tex_path)


def _compile_pdf(tex_path: Path) -> Optional[str]:
    if shutil.which("pdflatex") is None:
        return None
    subprocess.run(
        ["pdflatex", "-interaction=nonstopmode", tex_path.name],
        cwd=tex_path.parent,
        capture_output=True,
        text=True,
        check=False,
        timeout=120,
    )
    for suffix in (".aux", ".log", ".out", ".toc"):
        aux_path = tex_path.with_suffix(suffix)
        if aux_path.exists():
            aux_path.unlink()
    pdf_path = tex_path.with_suffix(".pdf")
    return str(pdf_path) if pdf_path.exists() else None


def build_glossary(
    output_dir: str = "docs/glossary",
    compile_pdf: bool = True,
    sample_steps: int = 30,
) -> GlossaryBuildResult:
    out = Path(output_dir)
    figures_dir = out / "figures"
    out.mkdir(parents=True, exist_ok=True)

    visuals = _render_visuals(figures_dir, sample_steps)
    tex_path = _render_tex(out, visuals, sample_steps)
    manifest = {
        "output_dir": str(out),
        "figures": visuals,
        "public_api_count": len(_public_api_entries()),
        "graph_method_count": len(_graph_method_entries()),
        "loss_count": len(_loss_entries()),
        "hyperparameter_count": len(_hyperparameter_entries()),
        "glossary_terms": [name for name, _ in _BASICS],
        "stages": [name for name, _ in _STAGES],
    }
    manifest_path = out / "glossary_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    pdf_path = _compile_pdf(Path(tex_path)) if compile_pdf else None
    return GlossaryBuildResult(
        output_dir=str(out),
        tex_path=tex_path,
        pdf_path=pdf_path,
        manifest_path=str(manifest_path),
        figures_dir=str(figures_dir),
    )
