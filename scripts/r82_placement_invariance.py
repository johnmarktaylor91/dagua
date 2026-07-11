"""r82: placement-invariance check for routing-only changes (P10 protocol).

Runs dagua.layout() (seed 42, steps=15, CPU, single-threaded) on 5 spot
graphs and saves positions. Run once at base ("--save before"), once after
the routing prototype ("--save after"), then "--compare" reports
max-abs-diff (must be 0.0: routing is layered on top of placement and must
never move nodes).

Usage:
  .venv/bin/python scripts/r82_placement_invariance.py --save /tmp/r82_pos_before.pt
  .venv/bin/python scripts/r82_placement_invariance.py --save /tmp/r82_pos_after.pt
  .venv/bin/python scripts/r82_placement_invariance.py \
      --compare /tmp/r82_pos_before.pt /tmp/r82_pos_after.pt
"""

from __future__ import annotations

import argparse
import sys

import torch

torch.set_num_threads(1)

SPOT_GRAPHS = [
    "citation_dag_300",
    "random_dag_200",
    "clustered_medium_5x20",
    "r79_nested_clusters_3x2x10",
    "heavy_tail_weights_50",
]


def compute_positions():
    import dagua
    from dagua.eval.graphs import get_test_graphs

    by_name = {tg.name: tg for tg in get_test_graphs(max_nodes=500)}
    out = {}
    for name in SPOT_GRAPHS:
        g = by_name[name].graph
        g.compute_node_sizes()
        config = dagua.LayoutConfig(seed=42, steps=15, device="cpu")
        pos = dagua.layout(g, config)
        if isinstance(pos, dict):
            pos = pos.get("positions", pos)
        out[name] = pos.detach().cpu()
        print(f"[inv] {name}: {tuple(out[name].shape)}", flush=True)
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--save", type=str, default=None)
    parser.add_argument("--compare", nargs=2, default=None)
    args = parser.parse_args()

    if args.compare:
        before = torch.load(args.compare[0], weights_only=True)
        after = torch.load(args.compare[1], weights_only=True)
        worst = 0.0
        for name in before:
            d = (before[name] - after[name]).abs().max().item()
            print(f"[inv] {name}: max_abs_diff={d}")
            worst = max(worst, d)
        print(f"[inv] WORST max_abs_diff={worst}")
        print("[inv] PASS" if worst == 0.0 else "[inv] FAIL")
        return 0 if worst == 0.0 else 1

    if args.save:
        torch.save(compute_positions(), args.save)
        print(f"[inv] saved {args.save}")
        return 0

    parser.print_help()
    return 2


if __name__ == "__main__":
    sys.exit(main())
