<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 22 ADVERSARIAL FIX for **classical_mds** family (`classic_classical_mds` vs `igraph_mds`).

## SPEC

Your spec is the diff document at:
`.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_classical_mds.md`

Read it END-TO-END. The "Recommended Round 22+ Fix Scope" section
contains the bundle for this round. The "Ranked Fix List" has details.

Apply the **top 3 highest-impact fixes** from the ranked list as a
single bundle. Each fix should be small (1-50 lines net per fix; total
< 200 lines). If the spec recommends a smaller staged scope, follow that.

## Process

1. Read `ROUND_21_DIFF_classical_mds.md` end-to-end.
2. Multi-seed baseline (3 seeds, 5 small graphs):
   ```
   python scripts/algo_fidelity_live_compare.py classic_classical_mds igraph_mds \
       --seeds 3 \
       --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
       --output-dir eval_output/algo_fidelity/round_22/classical_mds/baseline
   ```
3. Apply the top 3 levers from the spec as a bundle. Be precise --
   cite line:line refs from the diff doc.
4. Run pytest tests/test_layout/ -x --tb=short -q -k "classical_mds" (or
   whatever test selector matches the family).
5. Re-measure on the same subset.
6. **COMMIT criterion** (relaxed for diversity):
   - Median improves by >= 0.03, OR
   - Aggregate TOST verdict moves up one tier, OR
   - The fix is a clean opt-in fidelity_mode/flag with regression tests
     (even if median unchanged because mode is opt-in, this is valuable
     infrastructure -- commit it)
7. If COMMITTED: `feat(fidelity): round 22 classical_mds -- <short fix description>`
8. If criterion missed: revert. Write `ROUND_22_RESIDUAL_classical_mds.md`.
9. Per-round summary: `eval_output/algo_fidelity/round_22/classical_mds/SUMMARY.md`

## Scope

**Allowed**:
- The dagua ops/pipeline files for classical_mds (located via the ROUND_21_DIFF doc's "Files Read" section)
- `dagua/layout/ops/state.py` ONLY if SolveState field needed
- Specific support files mentioned in the diff doc (e.g. graph_utils.py for one specific function, init.py for the family-specific class only)
- `scripts/ogdf_runner.cpp` IF the family is OGDF-targeted and the diff doc explicitly recommends runner-side changes
- `dagua/eval/competitors/<family>_competitor.py` IF the diff doc explicitly recommends adapter changes (only for adapter-bug fixes)
- `eval_output/algo_fidelity/round_22/classical_mds/**`
- `.project-context/research/sprint_algo_fidelity/ROUND_22_*classical_mds*.md`
- `tests/test_layout/test_*classical_mds*.py` for regression tests + snapshot updates

**HARD do-not-touch**:
- `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`
- `tests/test_render/**`
- `.project-context/research/sprint_clusters/**`
- `.project-context/research/sprint_graphviz_parity/**`
- Any other family's pipeline/ops files (you only own classical_mds)

## Verification
- pytest layout tests for this family pass
- live_compare runs cleanly
- `git diff --stat HEAD~0` shows only allowed scope

ONE commit on develop only IF criterion met.
</task>

<scope_constraints>classical_mds family files only. NO other family code.</scope_constraints>
