"""Build the Dagua visual audit suite."""

from __future__ import annotations

import argparse

from dagua.eval.visual_audit import build_visual_audit_suite


def main() -> None:
    parser = argparse.ArgumentParser(description="Build Dagua visual iteration and audit artifacts.")
    parser.add_argument("--output-dir", default="eval_output/visual_audit")
    parser.add_argument("--steps", type=int, default=80)
    parser.add_argument("--edge-opt-steps", type=int, default=12)
    parser.add_argument("--graphs", nargs="*", default=None, help="Optional subset of graph names.")
    args = parser.parse_args()

    result = build_visual_audit_suite(
        output_dir=args.output_dir,
        steps=args.steps,
        edge_opt_steps=args.edge_opt_steps,
        graph_names=args.graphs,
    )
    print(result.output_dir)


if __name__ == "__main__":
    main()
