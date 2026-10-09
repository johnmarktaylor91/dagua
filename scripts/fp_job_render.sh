#!/usr/bin/env bash
# Job wrapper: render the selected fresh pairs from the layout files left in the shared scratch folder.
set -u
export PYTHONPATH=$PWD
S=$HOME/scratch/dagua-freshpairs
PY=$HOME/anaconda3/envs/py311/bin/python
mkdir -p out
$PY scripts/fp_render.py --graphs "$S/main/graphs.json" --layouts "$S/main/layouts-main.jsonl" "$S/sprint2/layouts-sprint2.jsonl" \
  --pairs "$S/sel/pairs.json" --out out/render > out/render.log 2>&1
rc=$?
mkdir -p "$S/render" && cp -r out/render/. "$S/render/"
echo "render rc=$rc"
exit $rc
