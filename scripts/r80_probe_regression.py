"""Instrumented probe: name the mechanism behind the two sweep regressions.

Reruns the dagua engine on the two regressing graphs (rgg_500 and
r79_weighted_hub_spoke_4x18) with counters wrapped around
``_project_exact`` and ``_grid_spread_residual_overlaps`` to establish
whether the grid-spread deadlock valve fired and how many projector calls /
passes were consumed. Read-only with respect to the baseline store.

Usage: python scripts/r80_probe_regression.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from r79_baseline import SEED, TIMEOUT_SECONDS, build_corpus, get_competitor

from dagua.layout import projection as projection_module

TARGETS = ["r79_weighted_hub_spoke_4x18", "rgg_500"]

stats = {"project_exact_calls": 0, "grid_spread_calls": 0, "grid_spread_sizes": []}

_orig_project_exact = projection_module._project_exact
_orig_grid_spread = projection_module._grid_spread_residual_overlaps


def counted_project_exact(pos, node_sizes, padding, iterations, *args, **kwargs):
    """Count exact-projector invocations, then delegate."""
    stats["project_exact_calls"] += 1
    return _orig_project_exact(pos, node_sizes, padding, iterations, *args, **kwargs)


def counted_grid_spread(pos, node_sizes, padding, node_indices):
    """Count grid-spread activations and residual-set sizes, then delegate."""
    stats["grid_spread_calls"] += 1
    stats["grid_spread_sizes"].append(int(node_indices.shape[0]))
    return _orig_grid_spread(
        pos=pos, node_sizes=node_sizes, padding=padding, node_indices=node_indices
    )


projection_module._project_exact = counted_project_exact
projection_module._grid_spread_residual_overlaps = counted_grid_spread


def main() -> int:
    """Run the instrumented probe.

    Returns
    -------
    int
        Process exit status.
    """
    corpus = {g.name: g for g in build_corpus()}
    competitor = get_competitor("dagua")
    for name in TARGETS:
        for key in stats:
            stats[key] = [] if key == "grid_spread_sizes" else 0
        test_graph = corpus[name]
        t0 = time.perf_counter()
        competitor.layout(test_graph.graph, timeout=TIMEOUT_SECONDS, seed=SEED)
        elapsed = time.perf_counter() - t0
        print(
            f"{name}: layout {elapsed:.1f}s | project_exact calls: "
            f"{stats['project_exact_calls']} | grid_spread activations: "
            f"{stats['grid_spread_calls']} | residual-set sizes: "
            f"{stats['grid_spread_sizes'][:20]}",
            flush=True,
        )
    return 0


if __name__ == "__main__":
    sys.exit(main())
