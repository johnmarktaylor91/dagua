#!/usr/bin/env bash
# r76 umap stage-1 disposition test: bench 5 graphs from the UNCOMMITTED worktree code,
# then rescore vs the existing reference chain. Decides whether attempt-2 umap code
# reaches rung-2 (distributional equivalence) despite the spectral-init eigenbasis chaos.
set -uo pipefail
WT=/home/jtaylor/.claude/worktrees/dagua-umap-port
MAIN=/home/jtaylor/projects/dagua
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-r76
export PYTHONPATH=$WT

cd $WT
nice -n 10 python3 scripts/run_benchmark.py --variants \
  --engines classic_umap \
  --graphs parallel_multiedge_bundle,random_dag_50,random_dag_200,citation_dag_300,clustered_longlabel_handoffs \
  --seeds 100 --seed-start 100 --workers 6 --timeout 2400 --watchdog-timeout 3600 \
  --output-dir $MAIN/eval_output/benchmark_100seed_r76_umap_fix
echo "BENCH_RC=$?"

cd $MAIN
unset PYTHONPATH
nice -n 10 python3 scripts/definitive_fidelity_analysis.py --mode full \
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
  --data-dir eval_output/benchmark_100seed_r76_umap_fix \
  --combos-file /tmp/r76_umap_combos.txt --workers 6 \
  --output eval_output/fidelity_definitive/r76_umap_stage1.jsonl
echo "RESCORE_RC=$?"

python3 - <<'PY'
import json
rows=[json.loads(l) for l in open('eval_output/fidelity_definitive/r76_umap_stage1.jsonl')]
last={}
for r in rows: last[r['combo_id']]=r
rows=list(last.values())
div7={"parallel_multiedge_bundle::classic_umap_default","parallel_multiedge_bundle::classic_umap_mindist001",
"parallel_multiedge_bundle::classic_umap_nn5","parallel_multiedge_bundle::classic_umap_spread2",
"parallel_multiedge_bundle::classic_umap_nn30","random_dag_50::classic_umap_nn5","random_dag_200::classic_umap_nn5"}
print("=== the 7 previously-divergent ===")
for r in sorted(rows,key=lambda r:r['combo_id']):
    if r['combo_id'] in div7:
        print(f"  {r['combo_id']:55s} ident={r.get('quality_identical_raw')} equiv={r.get('quality_equivalent_raw')}")
print("=== regression sample (must stay identical) ===")
for r in sorted(rows,key=lambda r:r['combo_id']):
    if r['combo_id'].split('::')[0] in ('citation_dag_300','clustered_longlabel_handoffs'):
        print(f"  {r['combo_id']:55s} ident={r.get('quality_identical_raw')} equiv={r.get('quality_equivalent_raw')}")
PY
echo "R76_UMAP_STAGE1_COMPLETE"
