#!/usr/bin/env bash
# Job wrapper for the fresh-pairs layout sweep. Usage: fp_job_layout.sh TAG [ENGINES] [WORKERS]
set -u
TAG=${1:-main}; ENGINES=${2:-}; WORKERS=${3:-3}
export NODE_PATH=$HOME/data/dagua/refs/node_modules
PY=$HOME/anaconda3/envs/py311/bin/python
mkdir -p out smoke
EXTRA=()
[ -n "$ENGINES" ] && EXTRA=(--engines "$ENGINES")
$PY scripts/fp_layout.py --out smoke --tag "$TAG" --limit 2 --workers 2 "${EXTRA[@]}" > smoke/log.txt 2>&1 || { echo smoke-failed; cp -r smoke out/; exit 3; }
if [ "$TAG" = main ]; then
  $PY - <<'PY' > smoke/render.log 2>&1 || { echo render-smoke-failed; cp -r smoke out/; exit 4; }
import json, sys
sys.path.insert(0, "scripts")
import fp_render as r
g = json.load(open("smoke/graphs.json"))
lay = r.load_layouts([__import__("pathlib").Path("smoke/layouts-main.jsonl")])
name = next(iter(g))
engs = sorted({k[1] for k in lay if k[0] == name})
print(name, engs)
pairs = [{"pair_id": "smoke1", "graph": name, "x": {"layout": engs[0]}, "y": {"layout": engs[-1]}}]
json.dump(pairs, open("smoke/pairs.json", "w"))
r.main(["--graphs", "smoke/graphs.json", "--layouts", "smoke/layouts-main.jsonl", "--pairs", "smoke/pairs.json", "--out", "smoke/render"])
PY
fi
cp -r smoke out/smoke
$PY scripts/fp_layout.py --out out --tag "$TAG" --workers "$WORKERS" "${EXTRA[@]}" > out/layout.log 2>&1
rc=$?
D=$HOME/scratch/dagua-freshpairs/$TAG
mkdir -p "$D" && cp out/graphs.json out/layouts-"$TAG".jsonl out/layout.log "$D"/
echo "layout rc=$rc"
exit $rc
