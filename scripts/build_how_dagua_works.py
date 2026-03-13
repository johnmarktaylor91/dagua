#!/usr/bin/env python
"""Build visuals for the public Dagua algorithm explainer."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

import dagua
from dagua import DaguaGraph, EdgeStyle, LayoutConfig
from dagua.reference_glossary import _render_graph_on_ax


def _pipeline_overview(path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(10, 3), dpi=180)
    ax.axis("off")
    stages = [
        ("Graph", "#EAF4F8"),
        ("Sizes", "#F7F2E8"),
        ("Layering", "#E9F4EA"),
        ("Coarsen", "#EEF0FA"),
        ("Optimize", "#FBEDEE"),
        ("Route", "#F3F0F8"),
        ("Render", "#EAF4F8"),
    ]
    xs = [0.07, 0.2, 0.33, 0.46, 0.59, 0.72, 0.85]
    for x, (label, fill) in zip(xs, stages):
        ax.add_patch(plt.Rectangle((x - 0.055, 0.36), 0.11, 0.28, facecolor=fill, edgecolor="#3A556A", linewidth=1.2))
        ax.text(x, 0.5, label, ha="center", va="center", fontsize=10)
    for left, right in zip(xs[:-1], xs[1:]):
        ax.annotate("", xy=(right - 0.06, 0.5), xytext=(left + 0.06, 0.5), arrowprops=dict(arrowstyle="->", color="#5F6368"))
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _multilevel_hierarchy(path: Path) -> None:
    fig, ax = plt.subplots(1, 1, figsize=(8, 4.5), dpi=180)
    ax.axis("off")
    levels = [
        ("Full graph", 10, 0.82, "#D7EAF6"),
        ("Coarse level 1", 6, 0.55, "#E8EAF7"),
        ("Coarse level 2", 3, 0.28, "#F5E8F0"),
    ]
    for label, count, y, fill in levels:
        xs = [0.1 + i * (0.8 / max(count - 1, 1)) for i in range(count)]
        for x in xs:
            ax.add_patch(plt.Circle((x, y), 0.03, facecolor=fill, edgecolor="#3A556A", linewidth=1.0))
        ax.text(0.5, y + 0.08, label, ha="center", va="bottom", fontsize=11)
    for x in [0.14, 0.3, 0.46, 0.62, 0.78]:
        ax.annotate("", xy=(x, 0.61), xytext=(x, 0.74), arrowprops=dict(arrowstyle="->", color="#70757A"))
    for x in [0.22, 0.5, 0.78]:
        ax.annotate("", xy=(x, 0.34), xytext=(x, 0.47), arrowprops=dict(arrowstyle="->", color="#70757A"))
    ax.text(0.5, 0.05, "Optimize small first, then prolong and refine back upward", ha="center", fontsize=10)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _routing_comparison(path: Path) -> None:
    g = DaguaGraph.from_edge_list([
        ("input", "fork"),
        ("fork", "a"),
        ("fork", "b"),
        ("a", "merge"),
        ("b", "merge"),
        ("merge", "out"),
    ])
    configs = [
        ("Straight", EdgeStyle(routing="straight", curvature=0.0)),
        ("Bezier", EdgeStyle(routing="bezier", curvature=0.4)),
        ("Orthogonal", EdgeStyle(routing="ortho", curvature=0.0)),
    ]
    fig, axes = plt.subplots(1, 3, figsize=(11, 3.6), dpi=180)
    for ax, (title, style) in zip(axes.flat, configs):
        gg = DaguaGraph.from_json(g.to_json())
        gg.default_edge_style = style
        pos = dagua.layout(gg, LayoutConfig(steps=60, edge_opt_steps=-1, seed=42))
        _render_graph_on_ax(ax, gg, pos, title)
    fig.savefig(path, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def _loss_constraint_story(path: Path) -> None:
    g = DaguaGraph.from_edge_list([
        ("input", "a"),
        ("input", "b"),
        ("a", "merge"),
        ("b", "merge"),
        ("merge", "output"),
    ])
    g.pin("input", x=0.0, y=0.0)
    g.pin("output", x=240.0, y=120.0)
    result = dagua.animate(
        g,
        LayoutConfig(steps=45, edge_opt_steps=8, seed=42),
        output=str(path),
        animation_config=dagua.AnimationConfig(
            fps=16,
            dpi=110,
            max_layout_frames=18,
            max_edge_frames=8,
            camera="focus",
            center_on="merge",
        ),
    )
    return result


def build(output_dir: str = "/home/jtaylor/projects/dagua/docs/how_dagua_works/figures") -> None:
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    _pipeline_overview(out / "pipeline_overview.png")
    _multilevel_hierarchy(out / "multilevel_hierarchy.png")
    _routing_comparison(out / "routing_comparison.png")
    _loss_constraint_story(out / "pinning_constraint.gif")


if __name__ == "__main__":
    build()
