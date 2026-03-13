"""Interactive notebook playground for layout intuition and tuning.

This module provides an optional ``ipywidgets``-based playground for exploring
how layout hyperparameters affect graph structure and metrics in real time.
It is intentionally notebook-first: a low-friction surface for developers and
curious users to build intuition before changing defaults.
"""

from __future__ import annotations

import copy
from dataclasses import replace
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import matplotlib.pyplot as plt
import torch

from dagua.config import LayoutConfig, PARAM_REGISTRY_DICT
from dagua.eval.graphs import TestGraph, get_test_graphs
from dagua.layout import layout
from dagua.metrics import composite, quick
from dagua.render import render


PLAYGROUND_PARAM_NAMES: Tuple[str, ...] = (
    "node_sep",
    "rank_sep",
    "w_crossing",
    "w_dag",
    "w_overlap",
    "w_straightness",
    "w_attract",
    "w_attract_x_bias",
    "w_repel",
    "steps",
)

PLAYGROUND_GRAPH_LADDER: Tuple[str, ...] = (
    "linear_3layer_mlp",
    "deep_chain_20",
    "wide_parallel_200",
    "residual_block",
    "long_range_residual_ladder",
    "nested_shallow_enc_dec",
    "interleaved_cluster_crosstalk",
    "tl_transformer_1layer",
    "kitchen_sink_hybrid_net",
    "random_sparse_500",
    "random_dense_300",
)

PLAYGROUND_PANEL_PRESETS: Mapping[str, Tuple[str, ...]] = {
    "basics": (
        "linear_3layer_mlp",
        "deep_chain_20",
        "wide_parallel_200",
        "residual_block",
    ),
    "skips": (
        "residual_block",
        "long_range_residual_ladder",
        "multiscale_skip_cascade",
        "tl_resnet_2block",
    ),
    "clusters": (
        "nested_shallow_enc_dec",
        "interleaved_cluster_crosstalk",
        "kitchen_sink_hybrid_net",
        "braided_feedback_tails",
    ),
    "money": (
        "residual_block",
        "tl_transformer_1layer",
        "interleaved_cluster_crosstalk",
        "width_skew_late_merge",
    ),
}


def _graph_catalog(max_nodes: int = 2000) -> Dict[str, TestGraph]:
    """Return the curated graph catalog used by the interactive playground."""
    catalog = {entry.name: entry for entry in get_test_graphs(max_nodes=max_nodes)}
    ordered: Dict[str, TestGraph] = {}
    for name in PLAYGROUND_GRAPH_LADDER:
        if name in catalog:
            ordered[name] = catalog[name]
    for name, entry in catalog.items():
        if name not in ordered:
            ordered[name] = entry
    return ordered


def _panel_graph_names(preset: str, catalog: Mapping[str, TestGraph]) -> List[str]:
    """Resolve a panel preset to available graph names."""
    names = list(PLAYGROUND_PANEL_PRESETS.get(preset, ()))
    resolved = [name for name in names if name in catalog]
    if resolved:
        return resolved
    return list(catalog.keys())[:4]


def _base_playground_config(device: str = "cpu") -> LayoutConfig:
    """Return the default layout config used by the interactive playground."""
    return LayoutConfig(device=device, edge_opt_steps=-1, seed=42)


def _apply_overrides(config: LayoutConfig, overrides: Mapping[str, float]) -> LayoutConfig:
    """Return a new config with widget-driven overrides applied."""
    updated = config
    for name, value in overrides.items():
        updated = replace(updated, **{name: value})
    return updated


def _graph_clone(test_graph: TestGraph):
    """Clone a graph entry so the playground does not mutate the shared corpus."""
    return copy.deepcopy(test_graph.graph)


def _ensure_node_sizes(graph: Any) -> torch.Tensor:
    """Return node sizes for metrics and rendering."""
    sizes = graph.compute_node_sizes()
    if sizes.ndim == 1:
        sizes = torch.stack([sizes, sizes], dim=1)
    elif sizes.ndim == 2 and sizes.shape[1] == 1:
        sizes = sizes.repeat(1, 2)
    return sizes


def _placement_metrics(graph: Any, pos: torch.Tensor, direction: str) -> Dict[str, float]:
    """Compute lightweight placement metrics for the playground."""
    metrics = quick(
        pos,
        graph.edge_index,
        node_sizes=_ensure_node_sizes(graph),
        direction=direction,
        back_edge_mask=getattr(graph, "_back_edge_mask", None),
    )
    score = composite(metrics)
    result = {
        "composite": float(score),
        "edge_crossings": float(metrics.get("edge_crossings", 0.0)),
        "overlap_count": float(metrics.get("overlap_count", 0.0)),
        "dag_consistency": float(metrics.get("dag_consistency", 0.0)),
        "edge_length_cv": float(metrics.get("edge_length_cv", 0.0)),
    }
    return result


def _metrics_delta(current: Mapping[str, float], baseline: Mapping[str, float]) -> Dict[str, float]:
    """Compute current-vs-baseline deltas for the main live metrics."""
    return {key: float(current[key] - baseline.get(key, 0.0)) for key in current}


def _metrics_html(
    graph_name: str,
    current: Mapping[str, float],
    baseline: Mapping[str, float],
) -> str:
    """Render a compact HTML summary for a single graph's live metrics."""
    delta = _metrics_delta(current, baseline)
    rows = []
    for key in ("composite", "edge_crossings", "overlap_count", "dag_consistency", "edge_length_cv"):
        cur = current[key]
        d = delta[key]
        sign = "+" if d >= 0 else ""
        rows.append(
            f"<tr><td><code>{key}</code></td><td>{cur:.3f}</td><td>{sign}{d:.3f}</td></tr>"
        )
    return (
        f"<h4 style='margin:0 0 6px 0'>{graph_name}</h4>"
        "<table style='border-collapse:collapse;width:100%'>"
        "<thead><tr><th align='left'>Metric</th><th align='left'>Current</th>"
        "<th align='left'>Δ vs default</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _panel_metrics_html(
    graph_names: Sequence[str],
    currents: Sequence[Mapping[str, float]],
    baselines: Sequence[Mapping[str, float]],
) -> str:
    """Render an aggregate metrics summary for the current panel selection."""
    if not currents:
        return "<p>No metrics available.</p>"
    keys = ("composite", "edge_crossings", "overlap_count", "dag_consistency", "edge_length_cv")
    summary: Dict[str, float] = {key: 0.0 for key in keys}
    baseline_summary: Dict[str, float] = {key: 0.0 for key in keys}
    for current, baseline in zip(currents, baselines):
        for key in keys:
            summary[key] += float(current[key])
            baseline_summary[key] += float(baseline[key])
    count = float(len(currents))
    rows = []
    for key in keys:
        cur = summary[key] / count
        base = baseline_summary[key] / count
        d = cur - base
        sign = "+" if d >= 0 else ""
        rows.append(
            f"<tr><td><code>{key}</code></td><td>{cur:.3f}</td><td>{sign}{d:.3f}</td></tr>"
        )
    return (
        "<h4 style='margin:0 0 6px 0'>Panel summary</h4>"
        f"<p style='margin:0 0 6px 0'>{', '.join(graph_names)}</p>"
        "<table style='border-collapse:collapse;width:100%'>"
        "<thead><tr><th align='left'>Metric</th><th align='left'>Mean current</th>"
        "<th align='left'>Δ vs default</th></tr></thead>"
        f"<tbody>{''.join(rows)}</tbody></table>"
    )


def _transition_positions(old: Optional[torch.Tensor], new: torch.Tensor, frames: int) -> Iterable[torch.Tensor]:
    """Yield interpolated positions for smooth notebook transitions."""
    if old is None or old.shape != new.shape or frames <= 1:
        yield new
        return
    for alpha in torch.linspace(0.0, 1.0, frames):
        yield old * (1.0 - alpha) + new * alpha


def launch_playground(
    *,
    device: str = "cpu",
    max_nodes: int = 2000,
    animation_frames: int = 6,
) -> Any:
    """Return an interactive widget for exploring layout parameter intuition.

    The playground is notebook-oriented and intentionally optional. It requires
    ``ipywidgets`` and IPython display support.
    """
    try:
        import ipywidgets as widgets
        from IPython.display import clear_output, display
    except ImportError as exc:  # pragma: no cover - import path depends on env
        raise ImportError(
            "launch_playground() requires ipywidgets in a notebook environment. "
            "Install ipywidgets and run this from Jupyter."
        ) from exc

    catalog = _graph_catalog(max_nodes=max_nodes)
    if not catalog:
        raise RuntimeError("No graphs available for the interactive playground.")

    base_config = _base_playground_config(device=device)
    graph_options = [(name.replace("_", " "), name) for name in catalog]

    mode = widgets.ToggleButtons(
        options=[("Single graph", "single"), ("Panel", "panel")],
        value="single",
        description="View",
    )
    graph_select = widgets.Dropdown(options=graph_options, value=graph_options[0][1], description="Graph")
    panel_select = widgets.Dropdown(
        options=[(name.replace("_", " "), name) for name in PLAYGROUND_PANEL_PRESETS],
        value="money",
        description="Panel",
    )
    direction = widgets.Dropdown(
        options=[("Top to bottom", "TB"), ("Left to right", "LR"), ("Bottom to top", "BT"), ("Right to left", "RL")],
        value=base_config.direction,
        description="Direction",
    )
    animate = widgets.Checkbox(value=True, description="Animate transitions")
    reset = widgets.Button(description="Reset defaults", button_style="")

    slider_widgets: Dict[str, Any] = {}
    for name in PLAYGROUND_PARAM_NAMES:
        param = PARAM_REGISTRY_DICT[name]
        min_val, max_val = param.sweep_range
        if name == "steps":
            slider = widgets.IntSlider(
                value=int(param.default),
                min=int(min_val),
                max=int(max_val),
                step=5,
                description=param.display_name,
                continuous_update=False,
                readout_format="d",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="98%"),
            )
        else:
            slider = widgets.FloatSlider(
                value=float(param.default),
                min=float(min_val),
                max=float(max_val),
                step=float((max_val - min_val) / 40.0) if max_val > min_val else 0.1,
                description=param.display_name,
                continuous_update=False,
                readout_format=".2f",
                style={"description_width": "initial"},
                layout=widgets.Layout(width="98%"),
            )
        slider_widgets[name] = slider

    metrics_out = widgets.HTML()
    figure_out = widgets.Output()

    state: Dict[str, Optional[torch.Tensor]] = {"single": None}
    panel_state: Dict[str, Optional[torch.Tensor]] = {}
    baseline_cache: Dict[Tuple[str, str], Tuple[torch.Tensor, Dict[str, float]]] = {}

    def current_config() -> LayoutConfig:
        overrides: Dict[str, float] = {}
        for name, slider in slider_widgets.items():
            overrides[name] = float(slider.value)
        cfg = _apply_overrides(base_config, overrides)
        return replace(cfg, direction=direction.value)

    def baseline_payload(graph_name: str, direction_value: str) -> Tuple[torch.Tensor, Dict[str, float]]:
        key = (graph_name, direction_value)
        if key in baseline_cache:
            return baseline_cache[key]
        baseline_graph = _graph_clone(catalog[graph_name])
        baseline_cfg = replace(base_config, direction=direction_value)
        baseline_pos = layout(baseline_graph, baseline_cfg)
        baseline_metrics = _placement_metrics(baseline_graph, baseline_pos, direction_value)
        baseline_cache[key] = (baseline_pos.detach().cpu().clone(), baseline_metrics)
        return baseline_cache[key]

    def render_single(cfg: LayoutConfig) -> None:
        graph_name = graph_select.value
        graph = _graph_clone(catalog[graph_name])
        pos = layout(graph, cfg)
        current_metrics = _placement_metrics(graph, pos, cfg.direction)
        _, baseline_metrics = baseline_payload(graph_name, cfg.direction)

        metrics_out.value = _metrics_html(graph_name, current_metrics, baseline_metrics)
        frames = animation_frames if animate.value else 1
        old_pos = state["single"]
        with figure_out:
            for interp in _transition_positions(old_pos, pos, frames):
                clear_output(wait=True)
                fig, _ = render(graph, interp, cfg, title=graph_name.replace("_", " "))
                display(fig)
                plt.close(fig)
        state["single"] = pos.detach().cpu().clone()

    def render_panel(cfg: LayoutConfig) -> None:
        graph_names = _panel_graph_names(panel_select.value, catalog)
        currents: List[Mapping[str, float]] = []
        baselines: List[Mapping[str, float]] = []
        frames = animation_frames if animate.value else 1

        graph_payload: List[Tuple[str, Any, torch.Tensor, Optional[torch.Tensor]]] = []
        for graph_name in graph_names:
            graph = _graph_clone(catalog[graph_name])
            pos = layout(graph, cfg)
            old_pos = panel_state.get(graph_name)
            graph_payload.append((graph_name, graph, pos, old_pos))
            currents.append(_placement_metrics(graph, pos, cfg.direction))
            _, baseline_metrics = baseline_payload(graph_name, cfg.direction)
            baselines.append(baseline_metrics)

        metrics_out.value = _panel_metrics_html(graph_names, currents, baselines)

        with figure_out:
            for frame_idx in range(frames):
                clear_output(wait=True)
                grid = widgets.GridspecLayout(2, 2, width="100%")
                for idx, (graph_name, graph, pos, old_pos) in enumerate(graph_payload[:4]):
                    alpha = 1.0 if frames <= 1 else float(frame_idx) / float(max(frames - 1, 1))
                    interp = pos if old_pos is None or old_pos.shape != pos.shape else old_pos * (1.0 - alpha) + pos * alpha
                    out = widgets.Output()
                    with out:
                        fig, _ = render(graph, interp, cfg, title=graph_name.replace("_", " "), figsize=(5, 3.8))
                        display(fig)
                        plt.close(fig)
                    grid[idx // 2, idx % 2] = out
                display(grid)

        for graph_name, _, pos, _ in graph_payload:
            panel_state[graph_name] = pos.detach().cpu().clone()

    def refresh(*_: Any) -> None:
        cfg = current_config()
        if mode.value == "single":
            render_single(cfg)
        else:
            render_panel(cfg)

    def reset_defaults(_: Any) -> None:
        for name, slider in slider_widgets.items():
            slider.value = int(PARAM_REGISTRY_DICT[name].default) if name == "steps" else float(PARAM_REGISTRY_DICT[name].default)
        direction.value = base_config.direction

    reset.on_click(reset_defaults)
    for control in [mode, graph_select, panel_select, direction, animate, *slider_widgets.values()]:
        control.observe(refresh, names="value")

    accordion = widgets.Accordion(children=[widgets.VBox(list(slider_widgets.values()))], selected_index=0)
    accordion.set_title(0, "Layout dials")

    controls = widgets.VBox(
        [
            widgets.HTML("<h3 style='margin:0'>Dagua Interactive Tuning Playground</h3>"
                         "<p style='margin:4px 0 10px 0'>Turn the dials, watch layouts move, and see the main placement metrics update live.</p>"),
            widgets.HBox([mode, graph_select, panel_select]),
            widgets.HBox([direction, animate, reset]),
            accordion,
            metrics_out,
        ],
        layout=widgets.Layout(width="34%"),
    )

    body = widgets.HBox(
        [controls, figure_out],
        layout=widgets.Layout(align_items="flex-start", width="100%"),
    )
    refresh()
    return body


__all__ = [
    "PLAYGROUND_GRAPH_LADDER",
    "PLAYGROUND_PANEL_PRESETS",
    "PLAYGROUND_PARAM_NAMES",
    "launch_playground",
]
