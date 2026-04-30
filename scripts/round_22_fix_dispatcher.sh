#!/usr/bin/env bash
# round_22_fix_dispatcher.sh
#
# Dispatches Round 22 ADVERSARIAL FIX codexes for all 16 algos that had
# Round 21 diff reports. Each codex reads its ROUND_21_DIFF_<family>.md spec
# and applies the recommended Round 22 scope (top 3 levers per family).

set -euo pipefail

REPO="/home/jtaylor/projects/dagua"
PROMPT_DIR="$REPO/.project-context/research/sprint_algo_fidelity"
DISPATCHER_LOG="/tmp/round_22_dispatcher.log"

echo "[$(date)] Round 22 fix dispatcher starting" >> "$DISPATCHER_LOG"

# (slug, dagua_engine, target_engine, scope_constraints) -- families that match
# Round 21 diff suffixes
ALGOS=(
  "classical_mds:classic_classical_mds:igraph_mds"
  "fa2:classic_fa2:fa2_ref"
  "fmmm:classic_fmmm:ogdf_fmmm"
  "fr:classic_fr:nx_spring"
  "gem:classic_gem:ogdf_gem"
  "kk:classic_kk:nx_kamada_kawai"
  "lgl:classic_lgl:igraph_lgl"
  "maxent_stress:classic_maxent_stress:ogdf_stress"
  "pivot_mds:classic_pivot_mds:ogdf_pivot_mds"
  "rt:classic_rt:igraph_rt"
  "sgd2_multi:classic_sgd2_multi:sgd2_multi_ref"
  "spectral:classic_spectral:nx_spectral"
  "stress_maj:classic_stress_maj:ogdf_stress"
  "stress_sgd:classic_stress_sgd:sgd2"
  "sugiyama:classic_sugiyama:igraph_sugiyama"
  "umap:classic_umap:umap_graph"
)

generate_prompt() {
  local slug="$1"
  local dagua_engine="$2"
  local target_engine="$3"
  local prompt_path="$PROMPT_DIR/PROMPT_22_fix_${slug}.md"

  cat > "$prompt_path" <<PROMPT_EOF
<task>
You are Codex on the dagua project. Repo: \`/home/jtaylor/projects/dagua\`. Branch: \`develop\`.

Round 22 ADVERSARIAL FIX for **${slug}** family (\`${dagua_engine}\` vs \`${target_engine}\`).

## SPEC

Your spec is the diff document at:
\`.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_${slug}.md\`

Read it END-TO-END. The "Recommended Round 22+ Fix Scope" section
contains the bundle for this round. The "Ranked Fix List" has details.

Apply the **top 3 highest-impact fixes** from the ranked list as a
single bundle. Each fix should be small (1-50 lines net per fix; total
< 200 lines). If the spec recommends a smaller staged scope, follow that.

## Process

1. Read \`ROUND_21_DIFF_${slug}.md\` end-to-end.
2. Multi-seed baseline (3 seeds, 5 small graphs):
   \`\`\`
   python scripts/algo_fidelity_live_compare.py ${dagua_engine} ${target_engine} \\
       --seeds 3 \\
       --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \\
       --output-dir eval_output/algo_fidelity/round_22/${slug}/baseline
   \`\`\`
3. Apply the top 3 levers from the spec as a bundle. Be precise --
   cite line:line refs from the diff doc.
4. Run pytest tests/test_layout/ -x --tb=short -q -k "${slug}" (or
   whatever test selector matches the family).
5. Re-measure on the same subset.
6. **COMMIT criterion** (relaxed for diversity):
   - Median improves by >= 0.03, OR
   - Aggregate TOST verdict moves up one tier, OR
   - The fix is a clean opt-in fidelity_mode/flag with regression tests
     (even if median unchanged because mode is opt-in, this is valuable
     infrastructure -- commit it)
7. If COMMITTED: \`feat(fidelity): round 22 ${slug} -- <short fix description>\`
8. If criterion missed: revert. Write \`ROUND_22_RESIDUAL_${slug}.md\`.
9. Per-round summary: \`eval_output/algo_fidelity/round_22/${slug}/SUMMARY.md\`

## Scope

**Allowed**:
- The dagua ops/pipeline files for ${slug} (located via the ROUND_21_DIFF doc's "Files Read" section)
- \`dagua/layout/ops/state.py\` ONLY if SolveState field needed
- Specific support files mentioned in the diff doc (e.g. graph_utils.py for one specific function, init.py for the family-specific class only)
- \`scripts/ogdf_runner.cpp\` IF the family is OGDF-targeted and the diff doc explicitly recommends runner-side changes
- \`dagua/eval/competitors/<family>_competitor.py\` IF the diff doc explicitly recommends adapter changes (only for adapter-bug fixes)
- \`eval_output/algo_fidelity/round_22/${slug}/**\`
- \`.project-context/research/sprint_algo_fidelity/ROUND_22_*${slug}*.md\`
- \`tests/test_layout/test_*${slug}*.py\` for regression tests + snapshot updates

**HARD do-not-touch**:
- \`dagua/render/**\`, \`dagua/styles.py\`, \`scripts/graphviz_theme_comparison.py\`
- \`tests/test_render/**\`
- \`.project-context/research/sprint_clusters/**\`
- \`.project-context/research/sprint_graphviz_parity/**\`
- Any other family's pipeline/ops files (you only own ${slug})

## Verification
- pytest layout tests for this family pass
- live_compare runs cleanly
- \`git diff --stat HEAD~0\` shows only allowed scope

ONE commit on develop only IF criterion met.
</task>

<scope_constraints>${slug} family files only. NO other family code.</scope_constraints>
PROMPT_EOF

  echo "$prompt_path"
}

for entry in "${ALGOS[@]}"; do
  IFS=":" read -r slug dagua_engine target_engine <<< "$entry"
  echo "[$(date)] preparing $slug ($dagua_engine vs $target_engine)" >> "$DISPATCHER_LOG"
  prompt_path=$(generate_prompt "$slug" "$dagua_engine" "$target_engine")
  log_path="/tmp/algo_fid_22_${slug}.log"

  ~/.claude/scripts/codex-bg.sh \
    "$log_path" \
    "$prompt_path" \
    --cd "$REPO" \
    --sandbox danger-full-access \
    --effort medium >> "$DISPATCHER_LOG" 2>&1

  echo "[$(date)] dispatched $slug" >> "$DISPATCHER_LOG"
  sleep 5
done

echo "[$(date)] all 16 Round 22 codexes dispatched" >> "$DISPATCHER_LOG"
