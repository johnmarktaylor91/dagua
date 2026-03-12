"""User-facing CLI for Dagua tooling."""

from __future__ import annotations

import argparse
import json
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
from dagua.eval.report import generate_benchmark_deltas, generate_report
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
