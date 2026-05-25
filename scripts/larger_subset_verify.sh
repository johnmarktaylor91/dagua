#!/usr/bin/env bash
# Larger-graph fidelity verification helper.
#
# The bounded 5-graph subset (linear_3layer_mlp, parallel_multiedge_bundle,
# nested_shallow_enc_dec, tl_mlp_3layer, mixed_width_labels) uses graphs at
# N=3-26 nodes. Several R31/R32 codexes flagged: "the new fixes target
# dense/multi-component cases that the tiny verification graphs don't
# exercise." E.g. umap multi-component spectral init never fires at N<10.
#
# This helper adds a medium-size graph to the bounded comparison so codex
# fixes get a more representative signal before declaring a regression.
#
# Usage:
#   bash scripts/larger_subset_verify.sh <dagua_engine> <reference> [<output_dir>]
#
# Example:
#   bash scripts/larger_subset_verify.sh classic_umap_default umap_graph /tmp/umap_verify

set -u
cd "$(dirname "$0")/.."

if [ "$#" -lt 2 ]; then
    echo "Usage: $0 <dagua_engine> <reference> [<output_dir>]" >&2
    exit 2
fi

DAGUA="$1"
TARGET="$2"
OUT="${3:-eval_output/algo_fidelity/larger_subset_verify/${DAGUA}_vs_${TARGET}}"

# 5 small + 5 medium graphs spanning N=14-200, mix of topologies
GRAPHS="linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels,asymmetric_hourglass_hub,small_world_100,scale_free_ba_120,citation_dag_300,sbm_4x30"

echo "[verify] $DAGUA vs $TARGET on $(echo $GRAPHS | tr ',' '\n' | wc -l) graphs (5 small + 5 medium)"
echo "[verify] output: $OUT"

mkdir -p "$OUT"
python3 scripts/algo_fidelity_live_compare.py "$DAGUA" "$TARGET" \
    --seeds 30 \
    --graphs "$GRAPHS" \
    --output-dir "$OUT"
