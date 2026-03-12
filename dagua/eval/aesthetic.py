"""Offline aesthetic iteration workflow for Dagua defaults.

Runs a fast local loop over a representative graph subset:
1. Layout + render each graph with the current theme/config.
2. Compute structural quality metrics.
3. Apply a strict local critic heuristic.
4. Mutate theme/layout defaults for the next round.

Artifacts are written to `aesthetic_review/` by default. That directory is
git-ignored, so full round logs stay local while curated outputs can be copied
into the repo when a run is accepted.
"""

from __future__ import annotations

import copy
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

from dagua.config import LayoutConfig
from dagua.eval.graphs import TestGraph, get_test_graphs
from dagua.graph import DaguaGraph
from dagua.io import save_style
from dagua.layout import layout
from dagua.metrics import compute_all_metrics
from dagua.render import render
from dagua.styles import (
    ClusterStyle,
    DEFAULT_THEME_OBJ,
    EdgeStyle,
    GraphStyle,
    NodeStyle,
    Theme,
)


REPRESENTATIVE_GRAPH_NAMES = [
    "linear_3layer_mlp",
    "residual_block",
    "unet_small",
    "nested_shallow_enc_dec",
    "kitchen_sink_hybrid_net",
    "edge_label_braid",
    "shape_and_routing_matrix",
    "disconnected_label_cycle_collage",
]


@dataclass
class GraphEval:
    name: str
    tags: List[str]
    description: str
    expected_challenges: str
    metrics: Dict[str, float]
    image_path: str


@dataclass
class CriticResponse:
    overall_rating: float
    what_works: List[str]
    what_fails: List[str]
    priority_fixes: List[str]
    detailed_notes: Dict[str, str]
    inspiration: str


def _theme_to_config_dict(theme: Theme) -> Dict[str, Any]:
    return {
        "theme": {
            "name": theme.name,
            "node_styles": {name: asdict(style) for name, style in theme.node_styles.items()},
            "edge_styles": {name: asdict(style) for name, style in theme.edge_styles.items()},
            "cluster_style": asdict(theme.cluster_style),
            "graph_style": asdict(theme.graph_style),
        }
    }


def _layout_to_dict(config: LayoutConfig) -> Dict[str, Any]:
    data = asdict(config)
    data.pop("flex", None)
    return data


def get_representative_graphs(max_nodes: int = 200) -> List[TestGraph]:
    graphs = get_test_graphs(max_nodes=max_nodes)
    by_name = {g.name: g for g in graphs}
    selected = [by_name[name] for name in REPRESENTATIVE_GRAPH_NAMES if name in by_name]
    if selected:
        return selected
    return graphs[:8]


def clone_graph(graph: DaguaGraph) -> DaguaGraph:
    return DaguaGraph.from_json(graph.to_json())


def _apply_theme(graph: DaguaGraph, theme: Theme) -> None:
    graph._theme = theme.copy()


def _mean(values: Sequence[float]) -> float:
    return sum(values) / max(len(values), 1)


def _aggregate_metrics(graph_evals: Sequence[GraphEval]) -> Dict[str, float]:
    metric_names = sorted({k for ge in graph_evals for k in ge.metrics})
    return {name: _mean([ge.metrics.get(name, 0.0) for ge in graph_evals]) for name in metric_names}


def _score_theme_and_layout(theme: Theme, config: LayoutConfig, aggregate: Dict[str, float]) -> float:
    score = 10.0
    score -= min(4.0, aggregate.get("edge_crossings", 0.0) * 0.25)
    score -= min(3.0, aggregate.get("node_overlaps", 0.0) * 1.0)
    score -= min(1.5, max(0.0, 1.0 - aggregate.get("dag_fraction", 1.0)) * 8.0)
    score -= min(1.0, aggregate.get("edge_straightness", 0.0) / 45.0)
    score -= min(0.8, abs(aggregate.get("aspect_ratio", 1.3) - 1.6) * 0.25)
    score -= min(0.8, aggregate.get("edge_length_cv", 0.0) * 0.5)
    score -= min(0.8, (aggregate.get("label_overlaps", 0.0) + aggregate.get("label_node_overlaps", 0.0)) * 0.5)

    default_node = theme.get_node_style("default")
    default_edge = theme.get_edge_style("default")
    cluster = theme.cluster_style
    graph_style = theme.graph_style

    if not (0.45 <= default_node.stroke_width <= 0.8):
        score -= 0.3
    if not (0.95 <= default_node.opacity <= 1.0):
        score -= 0.1
    if not (0.95 <= default_edge.width <= 1.4):
        score -= 0.25
    if not (0.5 <= default_edge.opacity <= 0.72):
        score -= 0.35
    if not (0.18 <= cluster.opacity <= 0.4):
        score -= 0.3
    if not (8.5 <= default_node.font_size <= 9.5):
        score -= 0.2
    if not (14.0 <= graph_style.margin <= 24.0):
        score -= 0.2
    if not (24.0 <= config.node_sep <= 38.0):
        score -= 0.3
    if not (40.0 <= config.rank_sep <= 64.0):
        score -= 0.3

    return max(1.0, min(10.0, score))


def critique_round(theme: Theme, config: LayoutConfig, graph_evals: Sequence[GraphEval]) -> CriticResponse:
    aggregate = _aggregate_metrics(graph_evals)
    rating = _score_theme_and_layout(theme, config, aggregate)

    what_works: List[str] = []
    what_fails: List[str] = []
    fixes: List[str] = []

    if aggregate.get("dag_fraction", 1.0) >= 0.98:
        what_works.append("Flow direction is legible; the layered read survives across the representative set.")
    else:
        what_fails.append("Hierarchy is leaking. Too many edges still read as lateral detours instead of disciplined flow.")
        fixes.append("Increase DAG and straightness pressure while giving layers a touch more separation.")

    if aggregate.get("edge_crossings", 0.0) <= 2.0:
        what_works.append("Crossing pressure is under control on the medium-complexity cases.")
    else:
        what_fails.append("Crossing density is still too high. The eye has to negotiate clutter instead of following structure.")
        fixes.append("Push crossing minimization and open the horizontal spacing slightly.")

    if aggregate.get("edge_straightness", 45.0) <= 18.0:
        what_works.append("Edges mostly hold a clean vertical cadence instead of meandering.")
    else:
        what_fails.append("Edges are too wavy for a default. Curves should solve exceptions, not narrate the whole graph.")
        fixes.append("Lower default curvature and reinforce vertical edge preference.")

    if theme.cluster_style.opacity > 0.35:
        what_fails.append("Cluster fills are muddying figure-ground separation.")
        fixes.append("Reduce cluster fill opacity and lighten cluster framing.")
    else:
        what_works.append("Cluster framing stays subordinate to the nodes instead of competing with them.")

    if theme.get_edge_style("default").opacity < 0.52:
        what_fails.append("Edges recede a bit too much; some long paths are under-articulated.")
        fixes.append("Give the default edge tone slightly more authority without thickening it.")

    if not what_works:
        what_works.append("Nothing is catastrophically broken; the system has a coherent baseline to refine.")
    if not what_fails:
        what_fails.append("The remaining issues are mostly about refinement: rhythm, whitespace balance, and edge authority.")
    if not fixes:
        fixes.append("Tighten spacing and edge emphasis together; the defaults are close but still too neutral.")

    details = {
        "nodes": (
            f"Node boxes are using font_size={theme.get_node_style('default').font_size:.1f}, "
            f"padding={theme.get_node_style('default').padding}, stroke_width={theme.get_node_style('default').stroke_width:.2f}."
        ),
        "edges": (
            f"Mean crossings={aggregate.get('edge_crossings', 0.0):.2f}, "
            f"straightness={aggregate.get('edge_straightness', 0.0):.2f} deg, "
            f"default width={theme.get_edge_style('default').width:.2f}, "
            f"opacity={theme.get_edge_style('default').opacity:.2f}, "
            f"curvature={theme.get_edge_style('default').curvature:.2f}."
        ),
        "text": (
            f"Default node text sits at {theme.get_node_style('default').font_size:.1f}pt; "
            f"cluster labels at {theme.cluster_style.font_size:.1f}pt bold."
        ),
        "spacing": (
            f"node_sep={config.node_sep:.1f}, rank_sep={config.rank_sep:.1f}, "
            f"edge_length_cv={aggregate.get('edge_length_cv', 0.0):.2f}, "
            f"aspect_ratio={aggregate.get('aspect_ratio', 0.0):.2f}."
        ),
        "clusters": (
            f"padding={theme.cluster_style.padding:.1f}, opacity={theme.cluster_style.opacity:.2f}, "
            f"depth_fill_step={theme.cluster_style.depth_fill_step:.2f}."
        ),
        "color": (
            f"Background {theme.graph_style.background_color}, "
            f"default edge color {theme.get_edge_style('default').color}, "
            f"default base node color {theme.get_node_style('default').base_color}."
        ),
        "composition": (
            f"overall_quality={aggregate.get('overall_quality', 0.0):.2f}; "
            "the target is calm hierarchy with visible path continuity and no debug-dump feel."
        ),
    }

    return CriticResponse(
        overall_rating=round(rating, 2),
        what_works=what_works[:3],
        what_fails=what_fails[:3],
        priority_fixes=fixes[:3],
        detailed_notes=details,
        inspiration="Graphviz for restraint, Netron for module legibility, but with softer contemporary spacing.",
    )


def _clamp(value: float, lo: float, hi: float) -> float:
    return max(lo, min(hi, value))


def mutate_theme_and_config(theme: Theme, config: LayoutConfig, critic: CriticResponse, aggregate: Dict[str, float]) -> Tuple[Theme, LayoutConfig, List[str]]:
    next_theme = theme.copy()
    next_config = copy.deepcopy(config)
    changes: List[str] = []

    if aggregate.get("edge_crossings", 0.0) > 2.0:
        next_config.w_crossing = _clamp(next_config.w_crossing + 0.25, 0.5, 4.0)
        next_config.node_sep = _clamp(next_config.node_sep + 2.0, 24.0, 40.0)
        changes.append("Increased crossing pressure and nudged node_sep upward.")

    if aggregate.get("edge_straightness", 0.0) > 18.0:
        next_config.w_straightness = _clamp(next_config.w_straightness + 0.3, 1.5, 4.0)
        next_config.w_attract_x_bias = _clamp(next_config.w_attract_x_bias + 0.3, 1.5, 4.0)
        for style in next_theme.edge_styles.values():
            style.curvature = _clamp(style.curvature - 0.05, 0.18, 0.55)
        changes.append("Reinforced vertical edge discipline and reduced default curvature slightly.")

    if aggregate.get("edge_length_cv", 0.0) > 0.55:
        next_config.w_length_variance = _clamp(next_config.w_length_variance + 0.1, 0.4, 1.5)
        next_config.rank_sep = _clamp(next_config.rank_sep + 2.0, 42.0, 64.0)
        changes.append("Increased edge length regularity pressure and expanded rank spacing.")

    if aggregate.get("overall_quality", 0.0) < 70.0:
        next_config.steps = min(max(next_config.steps, 80) + 10, 140)
        changes.append("Granted the optimizer a few more steps on the representative suite.")

    default_node = next_theme.get_node_style("default")
    if default_node.stroke_width > 0.58:
        for style in next_theme.node_styles.values():
            style.stroke_width = _clamp(style.stroke_width - 0.03, 0.48, 0.62)
        changes.append("Trimmed node border weight for less visual noise.")

    if next_theme.cluster_style.opacity > 0.32:
        next_theme.cluster_style.opacity = _clamp(next_theme.cluster_style.opacity - 0.04, 0.18, 0.32)
        next_theme.cluster_style.stroke_width = _clamp(next_theme.cluster_style.stroke_width - 0.05, 0.55, 0.85)
        changes.append("Lightened cluster treatment to improve figure-ground separation.")

    if next_theme.get_edge_style("default").opacity < 0.56:
        for style in next_theme.edge_styles.values():
            style.opacity = _clamp(style.opacity + 0.04, 0.54, 0.7)
            style.width = _clamp(style.width - 0.05, 0.95, 1.25)
        changes.append("Raised edge opacity while shaving width to keep paths visible but quiet.")

    if next_theme.graph_style.margin < 18.0:
        next_theme.graph_style.margin = _clamp(next_theme.graph_style.margin + 1.5, 16.0, 24.0)
        changes.append("Added a little more outer margin for a cleaner frame.")

    if not changes:
        next_config.node_sep = _clamp(next_config.node_sep + 1.0, 24.0, 38.0)
        next_config.rank_sep = _clamp(next_config.rank_sep + 1.0, 42.0, 62.0)
        changes.append("Small spacing refinement pass to keep the loop moving.")

    return next_theme, next_config, changes


def render_and_evaluate_round(
    graphs: Sequence[TestGraph],
    theme: Theme,
    config: LayoutConfig,
    round_dir: Path,
) -> List[GraphEval]:
    round_dir.mkdir(parents=True, exist_ok=True)
    graph_evals: List[GraphEval] = []

    for tg in graphs:
        try:
            graph = clone_graph(tg.graph)
            _apply_theme(graph, theme)
            graph.compute_node_sizes()
            pos = layout(graph, config)
            image_path = round_dir / f"{tg.name}.png"
            render(graph, pos, config, output=str(image_path), dpi=160, figsize=(6.5, 4.5))
            metrics = compute_all_metrics(
                pos,
                graph.edge_index,
                graph.node_sizes,
                clusters=graph.clusters,
                direction=graph.direction,
            )
            graph_evals.append(
                GraphEval(
                    name=tg.name,
                    tags=sorted(tg.tags),
                    description=tg.description,
                    expected_challenges=tg.expected_challenges,
                    metrics=metrics,
                    image_path=str(image_path),
                )
            )
        except Exception as exc:
            with open(round_dir / f"{tg.name}.error.txt", "w") as f:
                f.write(str(exc) + "\n")

    return graph_evals


def _write_round_artifacts(
    round_dir: Path,
    round_idx: int,
    theme: Theme,
    config: LayoutConfig,
    graph_evals: Sequence[GraphEval],
    critic: CriticResponse,
    changes_for_next_round: Sequence[str],
) -> None:
    aggregate = _aggregate_metrics(graph_evals)
    payload = {
        "round": round_idx,
        "visual_defaults": _theme_to_config_dict(theme),
        "hyperparameters": _layout_to_dict(config),
        "aggregate_metrics": aggregate,
        "graphs": [asdict(ge) for ge in graph_evals],
        "critic_response": asdict(critic),
        "changes_for_next_round": list(changes_for_next_round),
    }
    with open(round_dir / "round.json", "w") as f:
        json.dump(payload, f, indent=2)


def _write_final_outputs(
    output_dir: Path,
    theme: Theme,
    config: LayoutConfig,
    graph_evals: Sequence[GraphEval],
    round_logs: Sequence[Dict[str, Any]],
) -> None:
    final_dir = output_dir / "final"
    final_dir.mkdir(parents=True, exist_ok=True)

    save_style(_theme_to_config_dict(theme), final_dir / "default_aesthetic_theme.yaml")
    with open(final_dir / "default_layout_config.json", "w") as f:
        json.dump(_layout_to_dict(config), f, indent=2)

    aggregate = _aggregate_metrics(graph_evals)
    lines = [
        "# Dagua Default Aesthetic Summary",
        "",
        "## Final Position",
        f"- Overall local critic rating: {round_logs[-1]['critic_response']['overall_rating']:.2f}/10",
        f"- Mean overall quality metric: {aggregate.get('overall_quality', 0.0):.2f}",
        "",
        "## Key Choices",
        f"- Node style: rounded rectangles, stroke_width={theme.get_node_style('default').stroke_width:.2f}, font_size={theme.get_node_style('default').font_size:.1f}",
        f"- Edge style: width={theme.get_edge_style('default').width:.2f}, opacity={theme.get_edge_style('default').opacity:.2f}, curvature={theme.get_edge_style('default').curvature:.2f}",
        f"- Cluster style: padding={theme.cluster_style.padding:.1f}, opacity={theme.cluster_style.opacity:.2f}, stroke_width={theme.cluster_style.stroke_width:.2f}",
        f"- Layout spacing: node_sep={config.node_sep:.1f}, rank_sep={config.rank_sep:.1f}, w_crossing={config.w_crossing:.2f}, w_straightness={config.w_straightness:.2f}",
        "",
        "## Critic Pressure Points",
    ]
    for item in round_logs[-1]["critic_response"]["what_fails"]:
        lines.append(f"- {item}")
    lines.extend([
        "",
        "## Representative Graphs",
    ])
    for ge in graph_evals:
        lines.append(f"- `{ge.name}`: {ge.description} | tags={', '.join(ge.tags)}")

    with open(final_dir / "summary.md", "w") as f:
        f.write("\n".join(lines) + "\n")


def run_aesthetic_iteration(
    rounds: int = 10,
    output_dir: str = "aesthetic_review",
    max_nodes: int = 200,
    steps: int = 80,
    seed: int = 42,
) -> Dict[str, Any]:
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    graphs = get_representative_graphs(max_nodes=max_nodes)
    theme = copy.deepcopy(DEFAULT_THEME_OBJ)
    config = LayoutConfig(
        steps=steps,
        edge_opt_steps=-1,
        seed=seed,
        node_sep=28.0,
        rank_sep=50.0,
        w_crossing=1.8,
        w_straightness=2.2,
        w_length_variance=0.7,
        w_attract_x_bias=2.4,
    )

    round_logs: List[Dict[str, Any]] = []
    final_graph_evals: List[GraphEval] = []

    for round_idx in range(1, rounds + 1):
        round_dir = output_path / f"round_{round_idx:02d}"
        graph_evals = render_and_evaluate_round(graphs, theme, config, round_dir)
        critic = critique_round(theme, config, graph_evals)
        aggregate = _aggregate_metrics(graph_evals)
        next_theme, next_config, changes = mutate_theme_and_config(theme, config, critic, aggregate)
        _write_round_artifacts(round_dir, round_idx, theme, config, graph_evals, critic, changes)
        round_logs.append({
            "round": round_idx,
            "critic_response": asdict(critic),
            "aggregate_metrics": aggregate,
            "changes_for_next_round": changes,
        })
        theme, config = next_theme, next_config
        final_graph_evals = graph_evals

    _write_final_outputs(output_path, theme, config, final_graph_evals, round_logs)

    result = {
        "output_dir": str(output_path),
        "rounds": rounds,
        "graphs": [g.name for g in graphs],
        "final_theme": _theme_to_config_dict(theme),
        "final_layout_config": _layout_to_dict(config),
        "final_rating": round_logs[-1]["critic_response"]["overall_rating"] if round_logs else 0.0,
    }
    with open(output_path / "run_summary.json", "w") as f:
        json.dump(result, f, indent=2)
    return result


def main(argv: Optional[Sequence[str]] = None) -> None:
    import argparse

    parser = argparse.ArgumentParser(description="Offline aesthetic iteration for Dagua defaults")
    parser.add_argument("--rounds", type=int, default=10)
    parser.add_argument("--output-dir", default="aesthetic_review")
    parser.add_argument("--max-nodes", type=int, default=200)
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args(argv)

    result = run_aesthetic_iteration(
        rounds=args.rounds,
        output_dir=args.output_dir,
        max_nodes=args.max_nodes,
        steps=args.steps,
        seed=args.seed,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
