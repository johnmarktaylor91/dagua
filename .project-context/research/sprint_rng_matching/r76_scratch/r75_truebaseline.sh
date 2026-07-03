#!/usr/bin/env bash
# r75 TRUE-BASELINE rescore: corrected (Phase-2) metrics x FRESHEST positions
# (9-dir r73 chain + benchmark_100seed_r74_fixes appended freshest-last).
# The r74 Phase-2 rescore used stale pre-r74-fix positions; this run gives the
# honest current divergent set. Analysis-only -- no benching.
set -uo pipefail; cd /home/jtaylor/projects/dagua
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-r75

echo "=== r75 true-baseline rescore: 409 combos, 10-dir overlay ==="
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
  --data-dir eval_output/benchmark_100seed_r74_fixes \
  --combos-file /tmp/r75_truebase_combos.txt --workers 8 \
  --output eval_output/fidelity_definitive/r75_truebaseline.jsonl
rc=$?
echo "analysis rc=$rc rows=$(wc -l < eval_output/fidelity_definitive/r75_truebaseline.jsonl 2>/dev/null)"

echo ""
echo "=== diff vs r74_phase2_rescore (stale-position baseline) ==="
python3 - <<'PY'
import json
from collections import Counter
old={json.loads(l)['combo_id']: json.loads(l) for l in open('eval_output/fidelity_definitive/r74_phase2_rescore.jsonl')}
new={}
try:
    new={json.loads(l)['combo_id']: json.loads(l) for l in open('eval_output/fidelity_definitive/r75_truebaseline.jsonl')}
except FileNotFoundError:
    print('no output produced'); raise SystemExit(1)
def qi(r): return bool(r.get('quality_identical_raw'))
flips=[c for c in new if c in old and not qi(old[c]) and qi(new[c])]
regr=[c for c in new if c in old and qi(old[c]) and not qi(new[c])]
still=[c for c in new if not qi(new[c])]
print(f'rescored={len(new)} newly_quality_identical={len(flips)} regressions={len(regr)} still_divergent={len(still)}')
fam=Counter(c.split('::')[1].replace('classic_','').split('_')[0] for c in flips)
print('flips by family:', dict(fam.most_common()))
fam2=Counter(c.split('::')[1].replace('classic_','').split('_')[0] for c in still)
print('still-divergent by family:', dict(fam2.most_common()))
for c in flips[:40]: print('  FLIP', c)
for c in regr[:20]: print('  REGR', c)
PY
echo "R75_TRUEBASELINE_COMPLETE"
