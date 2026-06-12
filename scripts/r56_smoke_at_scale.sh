#!/usr/bin/env bash
# R56 smoke-at-scale: verify dagua's bit-exact ports hold on real benchmark
# graphs (not just 8-node smoke topologies). Picks 9 diverse graphs across
# size bins, runs every variant at seed 42 vs reference, computes per-variant
# RMSDs. Reports any engine with >1e-3 max RMSD as a scale-extension regression.

set -u
cd "$(dirname "$0")/.."

# Use instrumented graphviz so the report compares apples to apples
export PATH=/tmp/graphviz_instr/bin:$PATH
export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:${LD_LIBRARY_PATH:-}

SCRATCH_DIR=eval_output/algo_fidelity/round_56/scratch
RESULTS_JSON="$SCRATCH_DIR/results.json"
RMSD_REPORT="eval_output/algo_fidelity/round_56/per_variant_rmsd.md"

# Pick 9 graphs spanning size + topology diversity
GRAPHS="braided_feedback_tails,densenet_block,regular_3_30,planar_60,dense_pair_50,real_lesmis_77,wide_1_100_1,powerlaw_500,rgg_2000"

mkdir -p "$SCRATCH_DIR"

echo "=== R56 smoke-at-scale $(date -Iseconds) ==="
echo "graphviz binary: $(which dot)  ($(dot -V 2>&1))"
echo "Running benchmark: 1 seed, 9 graphs, all variants..."

python3 scripts/run_benchmark.py \
    --seeds 1 --variants \
    --output-dir "$SCRATCH_DIR" \
    --graphs "$GRAPHS" \
    --workers 8 \
    --engines all \
    --timeout 600 \
    --watchdog-timeout 720 || echo "benchmark exited non-zero (continuing to analyze partial results)"

echo ""
echo "=== Analyzing per-variant RMSDs ==="

python3 - <<'PY'
import json
import math
from pathlib import Path
from collections import defaultdict

import numpy as np

results_path = Path("eval_output/algo_fidelity/round_56/scratch/results.json")
report_path = Path("eval_output/algo_fidelity/round_56/per_variant_rmsd.md")
if not results_path.is_file():
    print(f"FATAL: {results_path} missing")
    raise SystemExit(1)

with results_path.open() as f:
    results = json.load(f)
print(f"Loaded {len(results)} result rows")

# Group by (graph, seed) - find reimpl vs reference pairs
from dagua.eval.variants import VARIANT_REGISTRY, original_variant_name

pair_map = {}
for v in VARIANT_REGISTRY:
    on = original_variant_name(v)
    if on is None:
        continue
    pair_map[v.variant_id] = on

def procrustes_rmsd(a: np.ndarray, b: np.ndarray) -> float:
    if a.shape != b.shape or a.size == 0:
        return float("nan")
    a_c = a - a.mean(0); b_c = b - b.mean(0)
    a_n = float(np.linalg.norm(a_c)); b_n = float(np.linalg.norm(b_c))
    if a_n < 1e-12 or b_n < 1e-12:
        return 0.0 if (a_n < 1e-12 and b_n < 1e-12) else float(a_n + b_n)
    a_u = a_c / a_n; b_u = b_c / b_n
    u, _, vt = np.linalg.svd(b_u.T @ a_u)
    rotation = u @ vt
    return float(np.linalg.norm((a_u @ rotation.T) - b_u))

# Index results by (engine, graph, seed)
by_key = {}
for row_id, row in results.items():
    eng = row.get("engine_name", "")
    graph = row.get("graph_name", "")
    seed = row.get("seed", -1)
    pos = row.get("positions")
    if pos is not None and row.get("status") == "ok":
        by_key[(eng, graph, seed)] = np.asarray(pos, dtype=float)

per_variant = defaultdict(list)
for reimp_name, ref_name in pair_map.items():
    for (eng, graph, seed), reimp_pos in by_key.items():
        if eng != reimp_name:
            continue
        ref_pos = by_key.get((ref_name, graph, seed))
        if ref_pos is None:
            continue
        rmsd = procrustes_rmsd(reimp_pos, ref_pos)
        per_variant[reimp_name].append((graph, seed, rmsd))

# Build report
report_lines = ["# R56 Smoke-at-scale per-variant RMSD\n",
                "Engines tested with seed=42 against reference adapter on 9 diverse benchmark graphs.\n",
                "Bit-exact threshold: 1e-3.\n",
                "| Variant | N | Mean | Median | Max | Verdict |",
                "|---|---:|---:|---:|---:|:--|"]

pass_c = 0; fail_c = 0; warn_c = 0
for variant, entries in sorted(per_variant.items()):
    rmsds = [r for (_g, _s, r) in entries if math.isfinite(r)]
    if not rmsds:
        report_lines.append(f"| {variant} | 0 | -- | -- | -- | NO DATA |")
        continue
    arr = np.asarray(rmsds)
    mean_r = float(arr.mean()); median_r = float(np.median(arr)); max_r = float(arr.max())
    if max_r < 1e-3:
        verdict = "OK"; pass_c += 1
    elif max_r < 1e-2:
        verdict = "WARN"; warn_c += 1
    else:
        verdict = "FAIL"; fail_c += 1
    report_lines.append(f"| {variant} | {len(rmsds)} | {mean_r:.3e} | {median_r:.3e} | {max_r:.3e} | {verdict} |")

report_lines.append(f"\nTotals: {pass_c} OK / {warn_c} WARN (1e-3 to 1e-2) / {fail_c} FAIL (>=1e-2)")

report_path.parent.mkdir(parents=True, exist_ok=True)
report_path.write_text("\n".join(report_lines))
print(f"\nWrote per-variant RMSD report to {report_path}")
print(f"Totals: {pass_c} OK, {warn_c} WARN, {fail_c} FAIL")

if fail_c > 0 or warn_c > 0:
    print(f"\n=== Engines with non-bit-exact behavior at scale ===")
    for variant, entries in sorted(per_variant.items()):
        rmsds = [r for (_g, _s, r) in entries if math.isfinite(r)]
        if not rmsds:
            continue
        max_r = max(rmsds)
        if max_r >= 1e-3:
            print(f"  {variant}: max={max_r:.3e} ({len(rmsds)} runs)")
            for (g, s, r) in entries:
                if math.isfinite(r) and r >= 1e-3:
                    print(f"    {g} seed={s}: {r:.3e}")
PY

echo ""
echo "=== R56 smoke-at-scale done ==="
