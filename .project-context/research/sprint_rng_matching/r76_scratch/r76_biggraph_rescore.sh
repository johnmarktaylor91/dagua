#!/usr/bin/env bash
# r76 D1: big-graph tier hang-safe rescore (>300-node combos, landmark APSP + sampled crossings)
set -uo pipefail; cd /home/jtaylor/projects/dagua
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-r76
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
  --data-dir eval_output/benchmark_100seed_r76_umap_refs \
  --data-dir eval_output/benchmark_100seed_r76_umap_refs2 \
  --combos-file /tmp/r76_biggraph_combos.txt --workers 5 \
  --output eval_output/fidelity_definitive/r76_biggraph.jsonl
echo "rc=$?"
python3 - <<'PY'
import json, collections
rows=[json.loads(l) for l in open('eval_output/fidelity_definitive/r76_biggraph.jsonl')]
last={}
for r in rows: last[r['combo_id']]=r
rows=list(last.values())
c=collections.Counter()
for r in rows:
    if r.get('insufficient_data'): c['insufficient:'+str(r.get('insufficient_reason'))]+=1
    elif r.get('no_canonical_reference'): c['no_canonical']+=1
    elif r.get('quality_identical_raw'): c['identical']+=1
    elif r.get('quality_equivalent_raw'): c['equivalent']+=1
    else: c['divergent']+=1
print("BIG-GRAPH TIER SUMMARY:", dict(c), "total", len(rows))
PY
echo "R76_BIGGRAPH_COMPLETE"
