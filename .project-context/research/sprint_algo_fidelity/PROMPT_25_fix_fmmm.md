<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 25 STRAGGLER FIX for **fmmm** family (`classic_fmmm` vs `ogdf_fmmm`).

## Round 24 measurement (post Round 22 + 23 fixes)

5 graphs, 30 seeds:
- median RMSD: **0.0803**
- max RMSD: **0.247** (parallel_multiedge_bundle)
- per-graph: linear_3layer_mlp 0.026, mixed_width_labels 0.075, nested_shallow_enc_dec 0.026, parallel_multiedge_bundle 0.247, tl_mlp_3layer 0.028
- TOST verdicts: not_tested (within_target was None because the OGDF FMMM cache has no multi-seed reference)

**The Round 22 + 23 fixes did NOT improve the median (still 0.0803 across rounds 22, 23, and 24).**

This means the remaining gap is something Rounds 21 / 22 / 23 either:
- Identified but did NOT actually patch (verify each ranked item ACTUALLY changed code, not just tests/docs/scaffolding)
- Missed entirely (deeper algorithmic divergence)

## Reference clones
- OGDF: `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/FMMMLayout.cpp` and `src/ogdf/energybased/fmmm/`
- OGDF includes: `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/FMMMLayout.h`
- Dagua FMMM source: `dagua/layout/ops/fmmm.py`, `dagua/layout/ops/pipelines/fmmm.py`

## Spec

Primary diff: `.project-context/research/sprint_algo_fidelity/ROUND_21_DIFF_fmmm.md`
Secondary: `.project-context/research/sprint_algo_fidelity/ROUND_22_RESIDUAL_fmmm.md` (if exists), eval_output/algo_fidelity/round_23/fmmm/SUMMARY.md

The Round 21 diff identified: galaxy choice INVERTED (highest vs lowest star mass) plus several other items. Rounds 22 + 23 claimed fixes for the top items but the empirical median didn't budge.

**Your job:**
1. **Verify each Round 22 / 23 fmmm commit actually changed runtime behavior** (not just exposed knobs gated by `fidelity_mode`). Run with the most-stringent fidelity-mode default and confirm:
   ```bash
   git log --oneline --grep "round 2[23] fmmm" -- dagua/layout/ops/fmmm.py dagua/layout/ops/pipelines/fmmm.py dagua/eval/competitors/classic_competitor.py
   ```
2. **Read the worst graph (parallel_multiedge_bundle) divergence in detail.** Generate dagua and ogdf positions side by side; visualize. Why is dagua so different on this 3-node graph specifically?
3. **Fresh line-by-line comparison of the OGDF FMMM main loop** vs dagua FMMM. Read OGDF `FMMMLayout.cpp` `call()` method, the FR-style force-law module, and the Galaxy multi-level if applicable. Walk through both line by line to find the next concrete divergence beyond the R21 list.
4. Apply the fix. Verify it actually moves the median.
5. If you find the R22/R23 commits exposed knobs but didn't FLIP the default to OGDF-faithful, FLIP the defaults under `fidelity_mode=True` AND also update the `classic_competitor.py` so `default_params` for `classic_fmmm` requests fidelity mode.

## Verification

Run BEFORE and AFTER on the bounded 5-graph 30-seed comparison:
```bash
python scripts/algo_fidelity_live_compare.py classic_fmmm ogdf_fmmm \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_25/fmmm/{baseline,post_fix}
```

Required: median RMSD improvement >= 0.01 OR worst-graph (parallel_multiedge_bundle) RMSD reduction >= 0.05. If neither, mark `principled_residual` with explicit classification.

## Scope constraints

- **DO NOT TOUCH** any of: `dagua/render/**`, `dagua/styles.py`, `scripts/graphviz_theme_comparison.py`, `scripts/build_gallery_audit.py`, `tests/test_render/**`, `.project-context/research/sprint_clusters/**`, `.project-context/research/sprint_graphviz_parity/**`. The cluster sprint owns those.
- Stage commits with explicit `git add <files>`; NO `git add -A` or `git commit -a`.
- Commit format: `feat(fidelity): round 25 fmmm -- <terse desc>`

## Tests

- After each commit: `pytest tests/test_layout/ -x --tb=short -q -k "fmmm" --ignore=tests/test_layout/test_gem_fidelity.py`
- Final summary: `eval_output/algo_fidelity/round_25/fmmm/SUMMARY.md` (before/after medians, per-graph RMSDs, list of fixes applied, residual rationale).

</task>

<completeness_contract>
- Either: (a) measurable improvement (median delta >= 0.01 or worst-graph reduction >= 0.05) with commit on develop, OR (b) `principled_residual` documentation in SUMMARY explaining why no further code change can help.
- Per-round SUMMARY.md mandatory.
- Surface remaining residuals so a future round can target them.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going. Only stop for missing details that change correctness, safety, or irreversible actions.
</default_follow_through_policy>

<verification_loop>
1. Read source: OGDF FMMMLayout.cpp + dagua fmmm.py
2. Run BEFORE comparison
3. Apply fix
4. pytest -k fmmm
5. Run AFTER comparison
6. If improved: commit + write SUMMARY
7. If not improved: revert + try next divergence in your discovered list
8. After 3 unsuccessful revert cycles: write principled_residual SUMMARY
</verification_loop>
