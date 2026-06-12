#!/usr/bin/env bash
# r71 P1d: 100-seed seeded-reference benchmark (plan sec. 2d).
# Per-engine loop (run_benchmark --graphs is GLOBAL -- the p3b lesson). Synthetic
# `<ref>__for__<variant>` engines from failing_map for the probe-SEEDABLE families
# (+ fdp ensemble-eligible). Seeds 42-141 via --seed-refs (run-scoped override).
set -uo pipefail
cd /home/jtaylor/projects/dagua
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 NUMEXPR_NUM_THREADS=1
OUT=eval_output/benchmark_100seed_seeded_refs
SEEDABLE_BASES="graphviz_neato,graphviz_sfdp,graphviz_fdp,ogdf_fmmm,ogdf_gem,ogdf_stress,igraph_mds"

DISK_FLOOR_GB=15
check_disk() {
  local free_gb
  free_gb=$(df --output=avail -BG / | tail -1 | tr -dc '0-9')
  if [ "$free_gb" -lt "$DISK_FLOOR_GB" ]; then
    echo "P1D_DISK_FLOOR free=${free_gb}G < ${DISK_FLOOR_GB}G -- ABORT"
    exit 3
  fi
}

# Build per-engine work list (ref_synthetic<TAB>graphs_csv) from failing_map
python3 - <<'PYEOF' > /tmp/r71_p1d_worklist.tsv
import json
fm = json.load(open('.project-context/research/sprint_rng_matching/failing_map_final.json'))
seedable = ("graphviz_neato","graphviz_sfdp","graphviz_fdp","ogdf_fmmm","ogdf_gem","ogdf_stress","igraph_mds")
for variant, ent in sorted(fm.items()):
    ref = ent["ref"]
    base = ref.split("__for__")[0]
    if base in seedable:
        print(f"{ref}\t{','.join(sorted(ent['graphs']))}")
PYEOF
N=$(wc -l < /tmp/r71_p1d_worklist.tsv)
echo "P1D worklist: $N reference engines"

i=0
while IFS=$'\t' read -r ENGINE GRAPHS; do
  i=$((i+1))
  check_disk
  echo "=== [$i/$N] $ENGINE ==="
  python3 scripts/run_benchmark.py --seeds 100 --seed-start 42 --variants \
    --engines "$ENGINE" --graphs "$GRAPHS" \
    --seed-refs "$SEEDABLE_BASES" \
    --output-dir "$OUT" --resume --workers 10 --timeout 300 --watchdog-timeout 600 \
    || echo "P1D_ENGINE_FAILED $ENGINE rc=$?"
done < /tmp/r71_p1d_worklist.tsv
echo "P1D_COMPLETE"
