"""User-facing CLI for Dagua tooling."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from pathlib import Path
from typing import Optional, Sequence

from dagua import LayoutConfig, poster, tour
from dagua.animation import PosterConfig, TourConfig
import torch

from dagua.eval.benchmark import (
    benchmark_run_status,
    get_rare_suite_graphs,
    get_standard_suite_graphs,
    merge_latest_results,
)
from dagua.eval.report import (
    generate_benchmark_deltas,
    generate_placement_dashboard_artifacts,
    generate_placement_summary_artifacts,
    generate_report,
)
from dagua.eval.visual_audit import build_visual_audit_suite, freeze_visual_audit_baseline
from dagua.eval.visual_audit import build_visual_review_session
from dagua.io import load


def _add_layout_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--steps", type=int, default=120, help="Node optimization steps")
    parser.add_argument("--edge-opt-steps", type=int, default=-1, help="Edge optimization steps (-1 skips)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed")
    parser.add_argument("--device", default="cpu", help="Layout device, e.g. cpu or cuda")


def _layout_config_from_args(args: argparse.Namespace) -> LayoutConfig:
    return LayoutConfig(
        steps=args.steps,
        edge_opt_steps=args.edge_opt_steps,
        seed=args.seed,
        device=args.device,
    )


def _lookup_benchmark_graph(graph_name: str):
    for bg in list(get_standard_suite_graphs()) + list(get_rare_suite_graphs()):
        if bg.test_graph.name == graph_name:
            return bg.test_graph.graph
    raise KeyError(f"Unknown benchmark graph {graph_name!r}")


def _resolve_benchmark_positions(
    output_dir: str,
    suite: str,
    graph_name: str,
    competitor: str,
) -> torch.Tensor:
    run_root = Path(output_dir) / "benchmark_db" / suite
    latest_link = run_root / "latest"
    results_path = latest_link / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No latest {suite} benchmark run found at {results_path}")
    payload = json.loads(results_path.read_text(encoding="utf-8"))
    graph_payload = payload["graphs"][graph_name]
    result = graph_payload["competitors"][competitor]
    rel_path = result.get("positions_path")
    if not rel_path:
        raise FileNotFoundError(f"No saved positions for {graph_name}/{competitor}")
    return torch.load((latest_link / rel_path).resolve())


def _suite_root(output_dir: str, suite: str) -> Path:
    return Path(output_dir) / "benchmark_db" / suite


def _load_run_payload(output_dir: str, suite: str, run_id: Optional[str]) -> dict:
    suite_root = _suite_root(output_dir, suite)
    run_dir = suite_root / (run_id or "latest")
    results_path = run_dir / "results.json"
    if not results_path.exists():
        raise FileNotFoundError(f"No benchmark results found at {results_path}")
    return json.loads(results_path.read_text(encoding="utf-8"))


def _resolve_run_dir(output_dir: str, suite: str, run_id: Optional[str]) -> Path:
    run_dir = _suite_root(output_dir, suite) / (run_id or "latest")
    if not run_dir.exists():
        raise FileNotFoundError(f"No benchmark run found at {run_dir}")
    return run_dir.resolve()


def _load_graph_and_positions(args: argparse.Namespace):
    if getattr(args, "benchmark_graph", None):
        graph = _lookup_benchmark_graph(args.benchmark_graph)
        positions = _resolve_benchmark_positions(
            output_dir=args.output_dir,
            suite=args.benchmark_suite,
            graph_name=args.benchmark_graph,
            competitor=args.competitor,
        )
        return graph, positions
    if args.graph is None:
        raise ValueError("Either a graph path or --benchmark-graph must be provided.")
    graph = load(args.graph)
    return graph, None


def _run_poster(args: argparse.Namespace) -> int:
    graph, positions = _load_graph_and_positions(args)
    result = poster(
        graph,
        positions=positions,
        config=_layout_config_from_args(args),
        output=args.output,
        poster_config=PosterConfig(
            format=args.format,
            scene=args.scene,
            keyframe_index=args.keyframe_index,
            dpi=args.dpi,
            lod_threshold=args.lod_threshold,
            detail_node_limit=args.detail_node_limit,
            label_node_limit=args.label_node_limit,
            edge_sample_limit=args.edge_sample_limit,
            show_titles=not args.no_titles,
        ),
    )
    print(json.dumps({"output": result.output, "format": result.format, "used_large_lod": result.used_large_lod}))
    return 0


def _run_tour(args: argparse.Namespace) -> int:
    graph, positions = _load_graph_and_positions(args)
    result = tour(
        graph,
        positions=positions,
        config=_layout_config_from_args(args),
        output=args.output,
        tour_config=TourConfig(
            format=args.format,
            scene=args.scene,
            fps=args.fps,
            dpi=args.dpi,
            lod_threshold=args.lod_threshold,
            detail_node_limit=args.detail_node_limit,
            label_node_limit=args.label_node_limit,
            edge_sample_limit=args.edge_sample_limit,
            show_titles=not args.no_titles,
        ),
    )
    print(json.dumps({"output": result.output, "format": result.format, "frame_count": result.frame_count}))
    return 0


def _run_benchmark_status(args: argparse.Namespace) -> int:
    payload = benchmark_run_status(output_dir=args.output_dir, suite=args.suite)
    print(json.dumps(payload, indent=2))
    return 0


def _run_benchmark_list(args: argparse.Namespace) -> int:
    suite_root = _suite_root(args.output_dir, args.suite)
    runs = []
    if suite_root.exists():
        for path in sorted((p for p in suite_root.iterdir() if p.is_dir() and p.name != "latest"), key=lambda p: p.name):
            payload_path = path / "results.json"
            partial_path = path / "results.partial.json"
            state = "missing"
            graph_count = 0
            if payload_path.exists():
                payload = json.loads(payload_path.read_text(encoding="utf-8"))
                graph_count = len(payload.get("graphs", {}))
                state = "complete"
            elif partial_path.exists():
                payload = json.loads(partial_path.read_text(encoding="utf-8"))
                graph_count = len(payload.get("graphs", {}))
                state = "partial"
            runs.append({"run_id": path.name, "state": state, "graphs": graph_count})
    print(json.dumps({"suite": args.suite, "runs": runs}, indent=2))
    return 0


def _run_benchmark_show(args: argparse.Namespace) -> int:
    payload = _load_run_payload(args.output_dir, args.suite, args.run_id)
    graph_payload = payload["graphs"][args.graph]
    if args.competitor:
        graph_payload = {
            "graph": args.graph,
            "competitor": args.competitor,
            "result": graph_payload["competitors"][args.competitor],
        }
    else:
        graph_payload = {"graph": args.graph, **graph_payload}
    print(json.dumps(graph_payload, indent=2))
    return 0


def _run_benchmark_freeze(args: argparse.Namespace) -> int:
    source_dir = _resolve_run_dir(args.output_dir, args.suite, args.run_id)
    frozen_root = _suite_root(args.output_dir, args.suite) / "frozen"
    frozen_root.mkdir(parents=True, exist_ok=True)
    target_dir = frozen_root / args.label
    if target_dir.exists():
        if not args.overwrite:
            raise FileExistsError(f"Frozen benchmark label already exists: {target_dir}")
        shutil.rmtree(target_dir)
    shutil.copytree(source_dir, target_dir)
    metadata_path = target_dir / "freeze_metadata.json"
    metadata_path.write_text(
        json.dumps(
            {
                "suite": args.suite,
                "source_run_id": source_dir.name,
                "source_dir": str(source_dir),
                "label": args.label,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(json.dumps({"label": args.label, "source_run_id": source_dir.name, "target_dir": str(target_dir)}, indent=2))
    return 0


def _run_benchmark_compare_runs(args: argparse.Namespace) -> int:
    payload_a = _load_run_payload(args.output_dir, args.suite, args.run_a)
    payload_b = _load_run_payload(args.output_dir, args.suite, args.run_b)
    overlapping = sorted(set(payload_a.get("graphs", {})) & set(payload_b.get("graphs", {})))
    graph_deltas = {}
    score_deltas = []
    runtime_deltas = []
    for graph_name in overlapping:
        result_a = payload_a["graphs"][graph_name]["competitors"].get(args.competitor, {})
        result_b = payload_b["graphs"][graph_name]["competitors"].get(args.competitor, {})
        if result_a.get("status") != "OK" or result_b.get("status") != "OK":
            continue
        score_delta = None
        runtime_delta = None
        if result_a.get("composite_score") is not None and result_b.get("composite_score") is not None:
            score_delta = float(result_b["composite_score"]) - float(result_a["composite_score"])
            score_deltas.append(score_delta)
        if result_a.get("runtime_seconds") is not None and result_b.get("runtime_seconds") is not None:
            runtime_delta = float(result_b["runtime_seconds"]) - float(result_a["runtime_seconds"])
            runtime_deltas.append(runtime_delta)
        graph_deltas[graph_name] = {
            "score_delta": score_delta,
            "runtime_delta_seconds": runtime_delta,
        }
    output = {
        "suite": args.suite,
        "competitor": args.competitor,
        "run_a": payload_a.get("run_id", args.run_a),
        "run_b": payload_b.get("run_id", args.run_b),
        "aggregate": {
            "graphs_compared": len(graph_deltas),
            "mean_score_delta": sum(score_deltas) / len(score_deltas) if score_deltas else None,
            "mean_runtime_delta_seconds": sum(runtime_deltas) / len(runtime_deltas) if runtime_deltas else None,
        },
        "graphs": graph_deltas,
    }
    print(json.dumps(output, indent=2))
    return 0


def _run_benchmark_watch(args: argparse.Namespace) -> int:
    remaining = args.iterations
    while True:
        payload = benchmark_run_status(output_dir=args.output_dir, suite=args.suite)
        print(json.dumps(payload, indent=2))
        if not args.follow:
            return 0
        if remaining is not None:
            remaining -= 1
            if remaining <= 0:
                return 0
        time.sleep(args.interval)


def _run_benchmark_report(args: argparse.Namespace) -> int:
    artifacts = generate_report(output_dir=args.output_dir, compile_pdf=not args.no_pdf)
    print(json.dumps(artifacts, indent=2))
    return 0


def _run_benchmark_deltas(args: argparse.Namespace) -> int:
    merge_latest_results(output_dir=args.output_dir)
    json_path, md_path = generate_benchmark_deltas(output_dir=args.output_dir)
    print(json.dumps({"benchmark_deltas_json": json_path, "benchmark_deltas_md": md_path}, indent=2))
    return 0


def _run_placement_sprint(args: argparse.Namespace) -> int:
    combined = merge_latest_results(output_dir=args.output_dir)
    placement_summary_json, placement_summary_md = generate_placement_summary_artifacts(
        output_dir=args.output_dir,
        combined_results=combined,
    )
    placement_dashboard_json, placement_dashboard_md = generate_placement_dashboard_artifacts(
        output_dir=args.output_dir,
        combined_results=combined,
    )
    benchmark_deltas_json, benchmark_deltas_md = generate_benchmark_deltas(
        output_dir=args.output_dir,
        combined_results=combined,
    )

    frozen_dir = None
    if args.freeze_label:
        source_dir = _resolve_run_dir(args.output_dir, "standard", args.run_id)
        frozen_root = _suite_root(args.output_dir, "standard") / "frozen"
        frozen_root.mkdir(parents=True, exist_ok=True)
        target_dir = frozen_root / args.freeze_label
        if target_dir.exists():
            if not args.overwrite:
                raise FileExistsError(f"Frozen benchmark label already exists: {target_dir}")
            shutil.rmtree(target_dir)
        shutil.copytree(source_dir, target_dir)
        (target_dir / "freeze_metadata.json").write_text(
            json.dumps(
                {
                    "suite": "standard",
                    "source_run_id": source_dir.name,
                    "source_dir": str(source_dir),
                    "label": args.freeze_label,
                },
                indent=2,
            ),
            encoding="utf-8",
        )
        frozen_dir = str(target_dir)

    print(
        json.dumps(
            {
                "placement_summary_json": placement_summary_json,
                "placement_summary_md": placement_summary_md,
                "placement_dashboard_json": placement_dashboard_json,
                "placement_dashboard_md": placement_dashboard_md,
                "benchmark_deltas_json": benchmark_deltas_json,
                "benchmark_deltas_md": benchmark_deltas_md,
                "frozen_dir": frozen_dir,
            },
            indent=2,
        )
    )
    return 0


def _run_visual_audit_build(args: argparse.Namespace) -> int:
    result = build_visual_audit_suite(
        output_dir=args.output_dir,
        steps=args.steps,
        edge_opt_steps=args.edge_opt_steps,
        graph_names=args.graphs,
        compare_to_baseline=args.compare_to_baseline,
        panels=args.panels,
    )
    print(json.dumps({
        "output_dir": result.output_dir,
        "manifest_path": result.manifest_path,
        "readme_path": result.readme_path,
        "ladder_count": len(result.ladder_paths),
        "competitor_count": len(result.competitor_paths),
        "baseline_diff_count": len(result.baseline_diff_paths),
    }, indent=2))
    return 0


def _run_visual_audit_freeze(args: argparse.Namespace) -> int:
    target = freeze_visual_audit_baseline(
        output_dir=args.output_dir,
        label=args.label,
        overwrite=args.overwrite,
    )
    print(json.dumps({"baseline_dir": target, "label": args.label}, indent=2))
    return 0


def _run_visual_session_build(args: argparse.Namespace) -> int:
    result = build_visual_review_session(
        output_dir=args.output_dir,
        steps=args.steps,
        edge_opt_steps=args.edge_opt_steps,
        graph_names=args.graphs,
    )
    print(
        json.dumps(
            {
                "output_dir": result.output_dir,
                "manifest_path": result.manifest_path,
                "readme_path": result.readme_path,
                "notes_path": result.notes_path,
                "image_count": len(result.image_paths),
            },
            indent=2,
        )
    )
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dagua CLI tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("benchmark-status", help="Show latest benchmark run status")
    status_parser.add_argument("--output-dir", default="eval_output")
    status_parser.add_argument("--suite", choices=["standard", "rare"], default="standard")
    status_parser.set_defaults(func=_run_benchmark_status)

    list_parser = subparsers.add_parser("benchmark-list", help="List stored benchmark runs")
    list_parser.add_argument("--output-dir", default="eval_output")
    list_parser.add_argument("--suite", choices=["standard", "rare"], default="standard")
    list_parser.set_defaults(func=_run_benchmark_list)

    show_parser = subparsers.add_parser("benchmark-show", help="Show stored results for one graph")
    show_parser.add_argument("graph")
    show_parser.add_argument("--output-dir", default="eval_output")
    show_parser.add_argument("--suite", choices=["standard", "rare"], default="standard")
    show_parser.add_argument("--run-id", default=None, help="Specific run id; defaults to latest")
    show_parser.add_argument("--competitor", default=None, help="Optional competitor to narrow the output")
    show_parser.set_defaults(func=_run_benchmark_show)

    freeze_parser = subparsers.add_parser("benchmark-freeze", help="Copy a run into a frozen named baseline")
    freeze_parser.add_argument("label")
    freeze_parser.add_argument("--output-dir", default="eval_output")
    freeze_parser.add_argument("--suite", choices=["standard", "rare"], default="standard")
    freeze_parser.add_argument("--run-id", default=None, help="Specific run id; defaults to latest")
    freeze_parser.add_argument("--overwrite", action="store_true")
    freeze_parser.set_defaults(func=_run_benchmark_freeze)

    compare_parser = subparsers.add_parser("benchmark-compare-runs", help="Compare two stored runs")
    compare_parser.add_argument("run_a")
    compare_parser.add_argument("run_b")
    compare_parser.add_argument("--output-dir", default="eval_output")
    compare_parser.add_argument("--suite", choices=["standard", "rare"], default="standard")
    compare_parser.add_argument("--competitor", default="dagua")
    compare_parser.set_defaults(func=_run_benchmark_compare_runs)

    watch_parser = subparsers.add_parser("benchmark-watch", help="Poll benchmark status repeatedly")
    watch_parser.add_argument("--output-dir", default="eval_output")
    watch_parser.add_argument("--suite", choices=["standard", "rare"], default="standard")
    watch_parser.add_argument("--follow", action="store_true")
    watch_parser.add_argument("--interval", type=float, default=10.0)
    watch_parser.add_argument("--iterations", type=int, default=None)
    watch_parser.set_defaults(func=_run_benchmark_watch)

    report_parser = subparsers.add_parser("benchmark-report", help="Generate report artifacts from stored benchmark results")
    report_parser.add_argument("--output-dir", default="eval_output")
    report_parser.add_argument("--no-pdf", action="store_true")
    report_parser.set_defaults(func=_run_benchmark_report)

    deltas_parser = subparsers.add_parser("benchmark-deltas", help="Generate round-over-round benchmark deltas")
    deltas_parser.add_argument("--output-dir", default="eval_output")
    deltas_parser.set_defaults(func=_run_benchmark_deltas)

    sprint_parser = subparsers.add_parser("placement-sprint", help="Regenerate placement-facing benchmark artifacts in one shot")
    sprint_parser.add_argument("--output-dir", default="eval_output")
    sprint_parser.add_argument("--run-id", default=None, help="Specific standard run id to freeze when using --freeze-label")
    sprint_parser.add_argument("--freeze-label", default=None, help="Optional standard benchmark baseline label to freeze")
    sprint_parser.add_argument("--overwrite", action="store_true")
    sprint_parser.set_defaults(func=_run_placement_sprint)

    audit_parser = subparsers.add_parser("visual-audit-build", help="Build the visual iteration / audit suite")
    audit_parser.add_argument("--output-dir", default="eval_output/visual_audit")
    audit_parser.add_argument("--graphs", nargs="*", default=None)
    audit_parser.add_argument(
        "--panels",
        nargs="*",
        default=None,
        help="Optional subset: ladder decomposition kill_switches diff_dashboard competitor_stepwise metric_cards sheets frozen_baselines run_to_run_diff readme manifest",
    )
    audit_parser.add_argument("--compare-to-baseline", default="reference")
    _add_layout_args(audit_parser)
    audit_parser.set_defaults(func=_run_visual_audit_build)

    audit_freeze_parser = subparsers.add_parser("visual-audit-freeze", help="Freeze the current visual-audit baseline under a label")
    audit_freeze_parser.add_argument("label")
    audit_freeze_parser.add_argument("--output-dir", default="eval_output/visual_audit")
    audit_freeze_parser.add_argument("--overwrite", action="store_true")
    audit_freeze_parser.set_defaults(func=_run_visual_audit_freeze)

    session_parser = subparsers.add_parser("visual-session-build", help="Build a numbered side-by-side review folder for collaborative visual iteration")
    session_parser.add_argument("--output-dir", default="eval_output/visual_review_session")
    session_parser.add_argument("--graphs", nargs="*", default=None)
    _add_layout_args(session_parser)
    session_parser.set_defaults(func=_run_visual_session_build)

    poster_parser = subparsers.add_parser("poster", help="Export a cinematic still render")
    poster_parser.add_argument("graph", nargs="?", help="Input graph file (JSON/YAML)")
    poster_parser.add_argument("output", help="Output still path")
    poster_parser.add_argument("--benchmark-graph", default=None, help="Graph name from the benchmark DB")
    poster_parser.add_argument("--benchmark-suite", choices=["standard", "rare"], default="standard")
    poster_parser.add_argument("--competitor", default="dagua", help="Competitor name for saved benchmark positions")
    poster_parser.add_argument("--output-dir", default="eval_output", help="Benchmark output dir when using --benchmark-graph")
    poster_parser.add_argument("--scene", default="auto")
    poster_parser.add_argument("--format", default=None)
    poster_parser.add_argument("--keyframe-index", type=int, default=None)
    poster_parser.add_argument("--dpi", type=int, default=220)
    poster_parser.add_argument("--lod-threshold", type=int, default=100_000)
    poster_parser.add_argument("--detail-node-limit", type=int, default=12_000)
    poster_parser.add_argument("--label-node-limit", type=int, default=160)
    poster_parser.add_argument("--edge-sample-limit", type=int, default=60_000)
    poster_parser.add_argument("--no-titles", action="store_true")
    _add_layout_args(poster_parser)
    poster_parser.set_defaults(func=_run_poster)

    tour_parser = subparsers.add_parser("tour", help="Export a cinematic graph tour")
    tour_parser.add_argument("graph", nargs="?", help="Input graph file (JSON/YAML)")
    tour_parser.add_argument("output", help="Output animation path")
    tour_parser.add_argument("--benchmark-graph", default=None, help="Graph name from the benchmark DB")
    tour_parser.add_argument("--benchmark-suite", choices=["standard", "rare"], default="standard")
    tour_parser.add_argument("--competitor", default="dagua", help="Competitor name for saved benchmark positions")
    tour_parser.add_argument("--output-dir", default="eval_output", help="Benchmark output dir when using --benchmark-graph")
    tour_parser.add_argument("--scene", default="auto")
    tour_parser.add_argument("--format", default=None)
    tour_parser.add_argument("--fps", type=int, default=24)
    tour_parser.add_argument("--dpi", type=int, default=160)
    tour_parser.add_argument("--lod-threshold", type=int, default=100_000)
    tour_parser.add_argument("--detail-node-limit", type=int, default=8_000)
    tour_parser.add_argument("--label-node-limit", type=int, default=120)
    tour_parser.add_argument("--edge-sample-limit", type=int, default=40_000)
    tour_parser.add_argument("--no-titles", action="store_true")
    _add_layout_args(tour_parser)
    tour_parser.set_defaults(func=_run_tour)

    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
