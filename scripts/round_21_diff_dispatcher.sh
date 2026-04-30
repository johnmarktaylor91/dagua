#!/usr/bin/env bash
# round_21_diff_dispatcher.sh
#
# Scheduled to run at 6am via nohup+sleep. Dispatches adversarial line-by-line
# diff codexes for all dagua algorithm families that have NOT been diffed in
# Round 19. Each codex produces an exhaustive ROUND_21_DIFF_<algo>.md report
# (DIAGNOSIS-ONLY -- no code changes).
#
# Targets (15 algos): all VARIANT_REGISTRY base engines except those already
# covered in Round 19 (davidson_harel, drl, graphopt, neulay, tsnet) and
# already covered via graphviz Round 9 (sfdp).

set -euo pipefail

REPO="/home/jtaylor/projects/dagua"
PROMPT_DIR="$REPO/.project-context/research/sprint_algo_fidelity"
SPRINT_LOG_DIR="/tmp"
DISPATCHER_LOG="/tmp/round_21_dispatcher.log"

echo "[$(date)] dispatcher starting" >> "$DISPATCHER_LOG"

# Reference root mappings (general guidance for codex; actual ref location may need codex search)
REF_HINTS="\
For igraph_*: source at /home/jtaylor/projects/_references/igraph/src/layout/<algo>.c (or .cpp). \
For ogdf_*: source at /home/jtaylor/projects/_references/ogdf/src/ogdf/<category>/<algo>.cpp + headers in /home/jtaylor/projects/_references/ogdf/include/ogdf/<category>/. \
For nx_* (networkx): source at /home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/networkx/drawing/layout.py. \
For fa2_*: /home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/fa2_modified/ (preferred) or fa2/. \
For sgd2_*: search site-packages for sgd2 or its installed name. \
For umap_*: /home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/. \
For tsne_*: /home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/sklearn/manifold/_t_sne.py."

# (dagua_engine, target_engine, ref_dir_or_file_hint, slug)
# Each line is one algo. slug becomes filename suffix for diff doc.
ALGOS=(
  "classic_classical_mds:igraph_mds:igraph/src/layout/mds.c:classical_mds"
  "classic_fa2:fa2_ref:site-packages/fa2_modified:fa2"
  "classic_fmmm:ogdf_fmmm:ogdf/src/ogdf/energybased/FMMM:fmmm"
  "classic_fr:nx_spring:networkx/drawing/layout.py:fr"
  "classic_gem:ogdf_gem:ogdf/src/ogdf/energybased/GEM:gem"
  "classic_kk:nx_kamada_kawai:networkx/drawing/layout.py:kk"
  "classic_lgl:igraph_lgl:igraph/src/layout/large_graph_layout.c:lgl"
  "classic_maxent_stress:ogdf_stress:ogdf/src/ogdf/energybased:maxent_stress"
  "classic_pivot_mds:ogdf_pivot_mds:ogdf/src/ogdf/energybased/PivotMDS:pivot_mds"
  "classic_rt:igraph_rt:igraph/src/layout/reingold_tilford.c:rt"
  "classic_sgd2_multi:sgd2_multi_ref:site-packages/sgd2:sgd2_multi"
  "classic_spectral:nx_spectral:networkx/drawing/layout.py:spectral"
  "classic_stress_maj:ogdf_stress:ogdf/src/ogdf/energybased/StressMinimization:stress_maj"
  "classic_stress_sgd:sgd2:site-packages/sgd2:stress_sgd"
  "classic_sugiyama:igraph_sugiyama:igraph/src/layout/sugiyama:sugiyama"
  "classic_umap:umap_graph:site-packages/umap:umap"
)

generate_prompt() {
  local dagua_engine="$1"
  local target_engine="$2"
  local ref_hint="$3"
  local slug="$4"
  local prompt_path="$PROMPT_DIR/PROMPT_21_diff_${slug}.md"

  cat > "$prompt_path" <<PROMPT_EOF
<task>
You are Codex on the dagua project. Repo: \`/home/jtaylor/projects/dagua\`. Branch: \`develop\`.

Round 21 ADVERSARIAL DIFF for **${slug}** family (dagua \`${dagua_engine}\` vs reference \`${target_engine}\`).

This is part of an exhaustive sweep covering EVERY dagua-vs-reference
pairing. Even if the family is currently \`strong_equivalent\` in the
mega-run, the user wants every last divergence catalogued -- GPT-5.5
may find something new.

## Inputs

**Dagua side (READ ALL):**
- Locate \`dagua/layout/ops/${slug}.py\` or related ops files for this engine.
- Locate \`dagua/layout/ops/pipelines/${slug}.py\` (the pipeline wiring).
- \`dagua/eval/variants.py\` for variant configs.
- \`dagua/eval/competitors/\` for the adapter that runs ${target_engine}.

**Reference side (READ ALL):**
- Reference path hint: \`${ref_hint}\`
- Search for the actual implementation if the hint path doesn't exist:
  ${REF_HINTS}
- For composite/hybrid engines, locate all relevant files.

**Existing analysis to skim:**
- \`eval_output/fidelity_report/report.md\` for the current verdict on ${slug}.
- \`.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md\` for sprint context.

## What to do

**This is a DIAGNOSIS-ONLY round.** Do NOT edit any source files. No commits.

Produce ONE document: \`.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_${slug}.md\`

Sections (be brutally exhaustive):

1. **Files read** -- list every source file you read on both sides.
2. **Overall pipeline structure** -- compare the high-level flow of the dagua and reference implementations.
3. **Energy / loss / objective** -- per-term comparison; cite formulas with file:line refs on both sides.
4. **Force / gradient computation** -- if applicable.
5. **Initialization** -- random scheme, scale, RNG type (numpy/torch/python random).
6. **Iteration / convergence** -- step count, learning-rate schedule, convergence test.
7. **Hyperparameter alignment table** -- exhaustive Y/N match per param + dagua default vs reference default.
8. **Edge cases** -- self-loops, multi-edges, disconnected components, weighted edges, empty graph.
9. **Numerical precision** -- float32 vs float64, dtype boundaries, summation order.
10. **RNG semantics** -- specifically does dagua's torch seed produce same sequence as reference's RNG?
11. **Edge-case bugs** -- anything that looks like an off-by-one, wrong sign, wrong direction, etc.
12. **Ranked fix list** -- 5+ items ranked by expected RMSD impact, each with file:line refs and proposed fix size estimate.
13. **Recommended Round 22+ fix scope** -- bundle of top-K levers for one followup round.

Be exhaustive. Cite specific line:line refs throughout. If the family
is \`strong_equivalent\` already, focus on residual sub-percent
divergences -- e.g., float precision, summation order, RNG semantics --
even if no obvious algorithmic divergence exists.

## End state
ONE markdown report at the path above. NO code changes. NO commits.
</task>

<scope_constraints>DIAGNOSIS-ONLY. NO file edits. NO commits. Read-only.</scope_constraints>

<verification_loop>File ROUND_21_DIFF_${slug}.md exists and is exhaustive (>10KB) with line:line refs throughout.</verification_loop>
PROMPT_EOF

  echo "$prompt_path"
}

# Generate prompts and dispatch
for entry in "${ALGOS[@]}"; do
  IFS=":" read -r dagua_engine target_engine ref_hint slug <<< "$entry"
  echo "[$(date)] preparing $slug ($dagua_engine vs $target_engine)" >> "$DISPATCHER_LOG"
  prompt_path=$(generate_prompt "$dagua_engine" "$target_engine" "$ref_hint" "$slug")
  log_path="$SPRINT_LOG_DIR/algo_fid_21_${slug}.log"

  # Dispatch with codex-bg.sh (handles backgrounding and watchdog)
  ~/.claude/scripts/codex-bg.sh \
    "$log_path" \
    "$prompt_path" \
    --cd "$REPO" \
    --sandbox danger-full-access \
    --effort medium >> "$DISPATCHER_LOG" 2>&1

  echo "[$(date)] dispatched $slug" >> "$DISPATCHER_LOG"
  # Throttle: 5 second gap between dispatches so codex auth pool isn't pummeled
  sleep 5
done

echo "[$(date)] all 16 codexes dispatched" >> "$DISPATCHER_LOG"
echo "Logs: /tmp/algo_fid_21_*.log" >> "$DISPATCHER_LOG"
echo "Diff outputs will land at: $PROMPT_DIR/ROUND_21_DIFF_*.md" >> "$DISPATCHER_LOG"
