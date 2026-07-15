#!/usr/bin/env bash
# r75 FINAL ANALYSIS: all 409 target combos, FIXED loader (per-combo freshest-dir),
# corrected metrics, all r75 layout fixes + regenerated OGDF references.
set -uo pipefail; cd /home/jtaylor/projects/dagua
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-r75

echo "=== [1] r75 final rescore: 409 combos, 12-dir chain, fixed overlay ==="
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
  --data-dir eval_output/benchmark_100seed_r74_fixes \
  --data-dir eval_output/benchmark_100seed_r75_fixes \
  --data-dir eval_output/benchmark_100seed_r75_mds_topup \
  --data-dir eval_output/benchmark_100seed_r75_topup2 \
  --combos-file /tmp/r75_truebase_combos.txt --workers 8 \
  --output eval_output/fidelity_definitive/r75_final.jsonl
echo "analysis rc=$? rows=$(wc -l < eval_output/fidelity_definitive/r75_final.jsonl 2>/dev/null)"

echo ""
echo "=== [2] scorecard vs r74 phase2 (stale baseline) + families ==="
python3 - <<'PY'
import json
from collections import Counter
old={json.loads(l)['combo_id']: json.loads(l) for l in open('eval_output/fidelity_definitive/r74_phase2_rescore.jsonl')}
new={json.loads(l)['combo_id']: json.loads(l) for l in open('eval_output/fidelity_definitive/r75_final.jsonl')}
def fam(c): return c.split('::')[1].replace('classic_','').split('_')[0]
def state(r):
    if r.get('no_canonical_reference'): return 'no_canonical_reference'
    if r.get('quality_identical_raw'): return 'quality_identical'
    if r.get('quality_superior_distinct'): return 'divergent_superior_distinct'
    if r.get('insufficient_data'): return 'insufficient'
    return 'divergent'
states=Counter(state(r) for r in new.values())
print('r75 FINAL states over the 409:', dict(states))
print()
div=[c for c,r in new.items() if state(r) in ('divergent','divergent_superior_distinct')]
print(f'divergent remaining: {len(div)} by family:', dict(Counter(fam(c) for c in div).most_common()))
flips=[c for c in new if c in old and not old[c].get('quality_identical_raw') and new[c].get('quality_identical_raw')]
print(f'flips vs r74-phase2: {len(flips)} by family:', dict(Counter(fam(c) for c in flips).most_common()))
regr=[c for c in new if c in old and old[c].get('quality_identical_raw') and state(new[c]) in ('divergent','divergent_superior_distinct')]
print(f'regressions: {len(regr)}')
for c in regr[:20]: print('  REGR', c)
PY

echo ""
echo "=== [3] anti-laundering controls (gate_5 must be 0/40) ==="
python3 scripts/definitive_fidelity_report.py --controls \
  --controls-dir eval_output/fidelity_definitive/controls_full \
  --output-dir /tmp/r75_final_controls > /tmp/r75_final_controls.log 2>&1
echo "controls rc=$? (nonzero expected: gate_3 pre-existing)"
python3 - <<'PY'
import json
g=json.load(open('/tmp/r75_final_controls/controls/gate_results.json'))
for k,v in g.get('gates', g).items():
    if isinstance(v, dict): print(k, 'passed=', v.get('passed'))
PY
echo "R75_FINAL_ANALYSIS_COMPLETE"
