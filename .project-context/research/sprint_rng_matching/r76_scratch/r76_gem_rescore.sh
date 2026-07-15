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
  --data-dir eval_output/benchmark_100seed_r76_gem_fix \
  --combos-file /tmp/r76_gem_all_combos.txt --workers 6 \
  --output eval_output/fidelity_definitive/r76_gem_rescore.jsonl
echo "rc=$?"
python3 - <<'PY'
import json
old={}
for l in open('eval_output/fidelity_definitive/per_combo_r75.jsonl'):
    r=json.loads(l)
    if 'classic_gem' in r.get('engine',''): old[r['combo_id']]=r
new=[json.loads(l) for l in open('eval_output/fidelity_definitive/r76_gem_rescore.jsonl')]
qi=[r for r in new if r.get('quality_identical_raw')]
still=[r for r in new if not r.get('quality_identical_raw') and not r.get('insufficient_data')]
print(f'gem full-family rescored={len(new)} quality_identical={len(qi)} not-identical={len(still)}')
for r in still[:20]: print('  still', r['combo_id'], 'disc=', r.get('disconnected'))
PY
echo "R76_GEM_RESCORE_COMPLETE"
