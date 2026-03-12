"""User-facing CLI for Dagua tooling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Optional, Sequence

from dagua import LayoutConfig, poster, tour
from dagua.animation import PosterConfig, TourConfig
from dagua.eval.benchmark import benchmark_run_status
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


def _run_poster(args: argparse.Namespace) -> int:
    graph = load(args.graph)
    result = poster(
        graph,
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
    graph = load(args.graph)
    result = tour(
        graph,
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


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Dagua CLI tooling")
    subparsers = parser.add_subparsers(dest="command", required=True)

    status_parser = subparsers.add_parser("benchmark-status", help="Show latest benchmark run status")
    status_parser.add_argument("--output-dir", default="eval_output")
    status_parser.add_argument("--suite", choices=["standard", "rare"], default="standard")
    status_parser.set_defaults(func=_run_benchmark_status)

    poster_parser = subparsers.add_parser("poster", help="Export a cinematic still render")
    poster_parser.add_argument("graph", help="Input graph file (JSON/YAML)")
    poster_parser.add_argument("output", help="Output still path")
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
    tour_parser.add_argument("graph", help="Input graph file (JSON/YAML)")
    tour_parser.add_argument("output", help="Output animation path")
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
