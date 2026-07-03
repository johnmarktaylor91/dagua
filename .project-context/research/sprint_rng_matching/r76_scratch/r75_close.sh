#!/usr/bin/env bash
set -uo pipefail; cd /home/jtaylor/projects/dagua
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 MPLCONFIGDIR=/tmp/mpl-r75
echo "=== merge per_combo_r73 <- r75_final -> per_combo_r75 ==="
python3 - <<'PY'
import json
base={}
for l in open('eval_output/fidelity_definitive/per_combo_r73.jsonl'):
    r=json.loads(l); base[r['combo_id']]=r
upd=0
for l in open('eval_output/fidelity_definitive/r75_final.jsonl'):
    r=json.loads(l); base[r['combo_id']]=r; upd+=1
with open('eval_output/fidelity_definitive/per_combo_r75.jsonl','w') as f:
    for cid in sorted(base): f.write(json.dumps(base[cid])+'\n')
print(f'overlaid={upd} total={len(base)}')
PY
echo "=== official r75 report ==="
python3 scripts/definitive_fidelity_report.py \
  --per-combo eval_output/fidelity_definitive/per_combo_r75.jsonl \
  --output-dir eval_output/fidelity_definitive_r75 \
  --data-dir eval_output/benchmark_100seed_escalation_final \
  --controls-dir eval_output/fidelity_definitive/controls_full \
  --no-strict > /tmp/r75_report.log 2>&1
echo "report rc=$?"
echo "=== official rung distribution ==="
python3 - <<'PY'
import json
from collections import Counter
rows=json.load(open('eval_output/fidelity_definitive_r75/per_combo.json'))
allc=Counter(str(r.get('final_rung')) for r in rows)
print('r75 official full dist:', dict(sorted(allc.items())))
pre=json.load(open('eval_output/fidelity_definitive_r73/per_combo.json'))
prec=Counter(str(r.get('final_rung')) for r in pre)
print('r73 official full dist:', dict(sorted(prec.items())))
PY
echo "R75_CLOSE_COMPLETE"
