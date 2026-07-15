#!/usr/bin/env bash
# r77: regenerate seed-42-era / low-power stochastic references at seeds 100-199.
# Plan: /tmp/r77_era_regen_plan.json (engine -> graph list, extracted from per_combo_r76).
set -uo pipefail; cd /home/jtaylor/projects/dagua
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-r77
OUT=eval_output/benchmark_100seed_r77_era_refs
for ENGINE in $(python3 -c "import json;print(' '.join(json.load(open('/tmp/r77_era_regen_plan.json')).keys()))"); do
  GRAPHS=$(python3 -c "import json;print(','.join(json.load(open('/tmp/r77_era_regen_plan.json'))['$ENGINE']))")
  echo "=== REGEN $ENGINE ($(echo $GRAPHS | tr ',' '\n' | wc -l) graphs) ==="
  nice -n 15 python3 scripts/run_benchmark.py --variants \
    --engines "$ENGINE" --seed-refs "$ENGINE" --graphs "$GRAPHS" \
    --max-nodes 0 --seeds 100 --seed-start 100 --workers 5 \
    --timeout 3600 --watchdog-timeout 7200 --resume --output-dir "$OUT"
  echo "=== $ENGINE rc=$? ==="
done
echo "R77_ERA_REGEN_COMPLETE"
