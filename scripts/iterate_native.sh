#!/bin/bash
# MVP fast iteration harness for the native dagua pipeline.
#
# Sprint 0 Task 0.4: prints scalar composite score, runtime, AND image path
# in <=8s (chain_100 baseline; larger graphs may exceed). Sprint 0.5 adds the
# full registry, rolling subset, and competitor delta.
#
# Usage:
#   scripts/iterate_native.sh [graph_id] [steps] [--no-image]
#
# Image emission is DEFAULT (per Sprint 0 exit criterion 3). Use --no-image
# to skip rendering when only the score matters (faster iteration).
#
# Supported graph_id (MVP, chain-only):
#   chain_<N>         e.g., chain_100, chain_500
#   random_dag_<N>    e.g., random_dag_200
#   diamond_<N>       e.g., diamond_40

set -euo pipefail

GRAPH="${1:-chain_100}"
STEPS="${2:-40}"
IMAGE_FLAG="${3:-}"

# Resolve output directory relative to script, not CWD.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(dirname "${SCRIPT_DIR}")"
OUTPUT_DIR="${REPO_ROOT}/eval_output/native_algo"
mkdir -p "${OUTPUT_DIR}"

TIMESTAMP="$(date +%Y%m%d_%H%M%S)"
IMAGE_PATH="${OUTPUT_DIR}/iter_${TIMESTAMP}_${GRAPH}.png"

EMIT_IMAGE=1
if [ "${IMAGE_FLAG}" = "--no-image" ]; then
    EMIT_IMAGE=0
fi

python - "${GRAPH}" "${STEPS}" "${IMAGE_PATH}" "${EMIT_IMAGE}" <<'PY'
import sys
import time

from dagua.config import LayoutConfig
from dagua.eval.graphs import make_chain, make_diamond, make_random_dag
from dagua.layout.engine import layout as engine_layout
from dagua.metrics import composite, composite_large, full, quick

GRAPH = sys.argv[1]
STEPS = int(sys.argv[2])
IMAGE_PATH = sys.argv[3]
EMIT_IMAGE = bool(int(sys.argv[4]))


def resolve_graph(spec: str):
    """MVP registry. Sprint 0.5 replaces with graph_generator.py."""
    if spec.startswith("chain_"):
        n = int(spec.split("_", 1)[1])
        return make_chain(n, seed=42).graph
    if spec.startswith("random_dag_"):
        n = int(spec.rsplit("_", 1)[1])
        return make_random_dag(n, density=2.0, seed=42).graph
    if spec.startswith("diamond_"):
        n = int(spec.split("_", 1)[1])
        return make_diamond(n, seed=42).graph
    raise SystemExit(
        f"MVP harness: unknown graph '{spec}' (Sprint 0.5 adds registry; "
        f"supported: chain_N, random_dag_N, diamond_N)"
    )


def main() -> None:
    g = resolve_graph(GRAPH)
    n = g.num_nodes
    g.compute_node_sizes()

    t0 = time.perf_counter()
    pos = engine_layout(g, LayoutConfig(steps=STEPS, seed=42))
    wall = time.perf_counter() - t0

    # Sprint 0 rule (Task 0.8): full() + composite() at N<=2000 (profile_small);
    # quick() + composite_large() at N>2000 (profile_large). NEVER call
    # composite() with quick-only metrics (silent-default blind spot).
    if n <= 2000:
        m = full(pos, g.edge_index, node_sizes=g.node_sizes)
        score = composite(m)
        profile = "profile_small"
    else:
        m = quick(pos, g.edge_index, node_sizes=g.node_sizes)
        score = composite_large(m)
        profile = "profile_large"

    overlap_count = m.get("overlap_count", 0)
    crossing_rate = m.get("crossing_rate", 0.0)
    dag_consistency = m.get("dag_consistency", 0.0)
    edge_length_cv = m.get("edge_length_cv", 0.0)

    print(f"graph={GRAPH} n={n} steps={STEPS} profile={profile}")
    print(f"score={score:.2f}")
    print(f"runtime_ms={wall * 1000:.1f}")
    print(
        f"overlap_count={overlap_count} crossing_rate={crossing_rate:.3f} "
        f"dag_consistency={dag_consistency:.3f} edge_length_cv={edge_length_cv:.3f}"
    )

    if EMIT_IMAGE:
        try:
            import matplotlib

            matplotlib.use("Agg")
            from dagua.render.mpl import render

            render(g, pos, output=IMAGE_PATH)
            print(f"image={IMAGE_PATH}")
        except Exception as e:
            print(f"image=SKIPPED (render error: {type(e).__name__}: {e})")


main()
PY
