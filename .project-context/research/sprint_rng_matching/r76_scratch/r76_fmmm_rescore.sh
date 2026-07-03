#!/usr/bin/env bash
set -uo pipefail; cd /home/jtaylor/projects/dagua
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-r75
nice -n 15 python3 scripts/definitive_fidelity_analysis.py --mode full \
  --data-dir eval_output/benchmark_100seed_escalation_final \
  --data-dir eval_output/benchmark_100seed_seeded_refs \
  --data-dir eval_output/benchmark_100seed_drlref_realfix \
  --data-dir eval_output/benchmark_100seed_umap_realfix \
  --data-dir eval_output/benchmark_100seed_gem_realfix \
  --data-dir eval_output/benchmark_100seed_r72_fixes \
  --data-dir eval_output/benchmark_100seed_fmmm_r3 \
  --data-dir eval_output/benchmark_100seed_fdp_fix \
  --data-dir eval_output/benchmark_100seed_r73_fixes \
  --data-dir eval_output/benchmark_100seed_r75_fixes \
  --data-dir eval_output/benchmark_100seed_r75_mds_topup \
  --data-dir eval_output/benchmark_100seed_r75_topup2 \
  --data-dir eval_output/benchmark_100seed_r76_refs \
  --combos-file /tmp/r76_fmmm_combos.txt --workers 4 \
  --output eval_output/fidelity_definitive/r76_fmmm_rescore.jsonl
echo "rc=$?"
python3 - <<'PY'
import json
new=[json.loads(l) for l in open('eval_output/fidelity_definitive/r76_fmmm_rescore.jsonl')]
flips=[r['combo_id'] for r in new if r.get('quality_identical_raw')]
print(f'fmmm rescored={len(new)} now_quality_identical={len(flips)} still={len(new)-len(flips)}')
for r in new:
    if not r.get('quality_identical_raw'):
        print('  still', r['combo_id'], 'disc=', r.get('disconnected'))
PY
echo "R76_FMMM_RESCORE_COMPLETE"
