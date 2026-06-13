#!/usr/bin/env bash
# r71 final assembly: wait for gem rebench, then union re-analysis across all overlay
# stores -> merged per_combo -> report v2. Zero-LLM; resumable.
set -uo pipefail
cd /home/jtaylor/projects/dagua
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG(){ echo "[$(date '+%F %T')] $*"; }

GEM_WRAP="${1:-}"
if [ -n "$GEM_WRAP" ]; then
  LOG "waiting for gem rebench wrapper $GEM_WRAP..."
  while kill -0 "$GEM_WRAP" 2>/dev/null; do sleep 120; done
fi
grep -q "R71_GEM_REBENCH_COMPLETE" /tmp/r71_gem_rebench.log 2>/dev/null && LOG "gem rebench COMPLETE" || LOG "WARN gem rebench may be incomplete"

# Union re-analysis: overlay order = base, seeded refs, fixed refs, fixed natives (last wins/key)
LOG "running union re-analysis (2626 combos)..."
nice -n 15 python3 scripts/definitive_fidelity_analysis.py --mode full \
  --data-dir eval_output/benchmark_100seed_escalation_final \
  --data-dir eval_output/benchmark_100seed_seeded_refs \
  --data-dir eval_output/benchmark_100seed_drlref_realfix \
  --data-dir eval_output/benchmark_100seed_umap_realfix \
  --data-dir eval_output/benchmark_100seed_gem_realfix \
  --combos-file /tmp/r71_union_combos.txt --workers 6 \
  --output eval_output/fidelity_definitive/r71_union_analysis.jsonl \
  >> /tmp/r71_union_analysis.log 2>&1
LOG "union re-analysis rc=$? rows=$(wc -l < eval_output/fidelity_definitive/r71_union_analysis.jsonl)"

# Merge: r70 per_combo (untouched combos) overlaid by r71 union verdicts -> per_combo_r71.jsonl
LOG "merging into per_combo_r71.jsonl..."
python3 - <<'PY' >> /tmp/r71_union_analysis.log 2>&1
import json
base={json.loads(l)['combo_id']:json.loads(l) for l in open('eval_output/fidelity_definitive/per_combo.jsonl')}
for src in ['r71_union_analysis','r71_drl_realfix_analysis','r71_umap_realfix_analysis','r71_p1e_seeded_analysis']:
    try:
        for l in open(f'eval_output/fidelity_definitive/{src}.jsonl'):
            r=json.loads(l); base[r['combo_id']]=r
    except FileNotFoundError: pass
with open('eval_output/fidelity_definitive/per_combo_r71.jsonl','w') as f:
    for r in base.values(): f.write(json.dumps(r)+'\n')
print('per_combo_r71 rows:', len(base))
PY

# Scorecard
LOG "computing final scorecard..."
python3 - <<'PY' >> /tmp/r71_union_analysis.log 2>&1
import json
rows=[json.loads(l) for l in open('eval_output/fidelity_definitive/per_combo_r71.jsonl')]
def fam(c): return c.split('::')[1].replace('classic_','').rsplit('_',1)[0]
def verdict(r):
    if r.get('insufficient_reason'): return 'insuf'
    m=r.get('mode')
    if m=='B':
        fl=r.get('flags') or []
        if 'near_deterministic' in fl: return 'equiv'
        if (r.get('p_typ') or 0)>0.05: return 'equiv'
        return 'equiv' if (r.get('quality_equivalent_raw') or r.get('stress_direct_equivalent')) else 'diff'
    if r.get('dist_equivalent'): return 'equiv'
    if r.get('quality_equivalent_raw') or r.get('stress_direct_equivalent') or r.get('one_sided_degenerate'): return 'equiv'
    return 'diff'
from collections import Counter
vc=Counter(); fd=Counter()
for r in rows:
    v=verdict(r); vc[v]+=1
    if v=='diff': fd[fam(r['combo_id'])]+=1
det=[json.loads(l) for l in open('eval_output/fidelity_definitive/deterministic_verdicts.jsonl')]
ddiff=sum(1 for d in det if d.get('deterministic_verdict')=='DIFFERENT')
deq=sum(1 for d in det if d.get('deterministic_verdict') in ('INVARIANCE_EQUIVALENT','QUALITY_EQUIVALENT'))
scored=vc['equiv']+vc['diff']
out={'escalation_equiv':vc['equiv'],'escalation_divergent':vc['diff'],'insufficient':vc['insuf'],
     'deterministic_equiv':deq,'deterministic_divergent':ddiff,
     'divergent_pct_escalation':round(100*vc['diff']/scored,1),
     'divergent_pct_esc_plus_det':round(100*(vc['diff']+ddiff)/(scored+deq+ddiff),1),
     'divergent_by_family':dict(fd.most_common())}
json.dump(out,open('eval_output/fidelity_definitive/r71_final_scorecard.json','w'),indent=1)
print(json.dumps(out,indent=1))
PY
LOG "R71_FINAL_ASSEMBLY_DONE"
