#!/usr/bin/env bash
# r71 unattended weekend chain (zero-LLM): when P1d completes, auto-run the P1e
# re-analysis on the seeded-ref upgrades; summarize everything for the Tuesday resume.
set -uo pipefail
cd /home/jtaylor/projects/dagua
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1
LOG() { echo "[$(date '+%F %T')] $*"; }

# 1. Wait for P1d (wrapper pid in arg 1, fallback: log marker)
P1D_PID="${1:-1110197}"
LOG "waiting for P1d (pid $P1D_PID)..."
while kill -0 "$P1D_PID" 2>/dev/null; do sleep 120; done
if ! grep -q "P1D_COMPLETE" /tmp/r71_p1d.log; then
  LOG "P1d exited WITHOUT P1D_COMPLETE -- attempting one resume relaunch"
  setsid nice -n 19 ionice -c 3 bash scripts/r71_p1d_seeded_refs.sh >> /tmp/r71_p1d.log 2>&1 < /dev/null &
  sleep 10
  NEWPID=$(pgrep -f "bash scripts/r71_p1d_seeded_refs.sh" | head -1)
  if [ -n "$NEWPID" ]; then
    LOG "relaunched as $NEWPID; waiting"
    while kill -0 "$NEWPID" 2>/dev/null; do sleep 120; done
  fi
fi
grep -q "P1D_COMPLETE" /tmp/r71_p1d.log || { LOG "P1d still incomplete -- stopping chain (resume pass will handle)"; exit 1; }
LOG "P1d COMPLETE"

# 2. Build the seedable Mode-B combos file
python3 - <<'PYEOF'
import json
fm = json.load(open('.project-context/research/sprint_rng_matching/failing_map_final.json'))
seedable = ("graphviz_neato","graphviz_sfdp","graphviz_fdp","ogdf_fmmm","ogdf_gem","ogdf_stress","igraph_mds")
n = 0
with open('/tmp/r71_p1e_combos.txt','w') as f:
    for v, ent in sorted(fm.items()):
        if ent['ref'].split('__for__')[0] in seedable:
            for g in sorted(ent['graphs']):
                f.write(f"{g}::{v}\n"); n += 1
print(f"p1e combos: {n}")
PYEOF

# 3. P1e re-analysis: escalation store + seeded-refs overlay
LOG "running P1e re-analysis..."
nice -n 19 python3 scripts/definitive_fidelity_analysis.py --mode full \
  --data-dir eval_output/benchmark_100seed_escalation_final \
  --data-dir eval_output/benchmark_100seed_seeded_refs \
  --combos-file /tmp/r71_p1e_combos.txt --workers 6 \
  --output eval_output/fidelity_definitive/r71_p1e_seeded_analysis.jsonl \
  >> /tmp/r71_p1e.log 2>&1
LOG "P1e analysis rc=$?"

# 4. Summary for the Tuesday resume pass
python3 - <<'PYEOF'
import json
from collections import Counter
out = {}
try:
    rows = [json.loads(l) for l in open('eval_output/fidelity_definitive/r71_p1e_seeded_analysis.jsonl')]
    modes = Counter(r.get('mode') or 'INSUF' for r in rows)
    a = [r for r in rows if r.get('mode') == 'A' and not r.get('insufficient_reason')]
    out['p1e'] = {
        'rows': len(rows), 'modes': dict(modes),
        'modeA_scored': len(a),
        'dist_equivalent': sum(1 for r in a if r.get('dist_equivalent')),
        'tracking_raw': sum(1 for r in a if (r.get('p_track') or 1) < 0.001 and (r.get('track_ratio') or 1) <= 0.5),
    }
except Exception as exc:
    out['p1e'] = {'error': str(exc)[:200]}
try:
    det = [json.loads(l) for l in open('eval_output/fidelity_definitive/deterministic_verdicts.jsonl')]
    out['w4'] = dict(Counter(r.get('deterministic_verdict') or 'NO_DATA' for r in det))
    out['w4_timeouts_remaining'] = sum(1 for r in det if r.get('toolkit_timeout'))
except Exception as exc:
    out['w4'] = {'error': str(exc)[:200]}
json.dump(out, open('eval_output/fidelity_definitive/r71_weekend_summary.json','w'), indent=1)
print(json.dumps(out, indent=1))
PYEOF
LOG "R71_WEEKEND_CHAIN_DONE"
