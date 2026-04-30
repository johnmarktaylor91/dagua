<task>
You are Codex on the dagua project. Repo: `/home/jtaylor/projects/dagua`. Branch: `develop`.

Round 28 STRAGGLER FIX for **sfdp** (`classic_sfdp` vs `graphviz_sfdp`).

## Round 27 baseline + ranked findings

Source: `.project-context/research/sprint_algo_fidelity/ROUND_27_DIFF_sfdp.md`

Baseline (5 graphs, 30 seeds): median 0.018990, p95 0.114511, worst
parallel_multiedge_bundle 0.132595.

Top R28-fixable items per the diff doc:
1. **Fine-level cooling**: graphviz still multiplies step by 0.90 even when
   `adaptive_cooling=false`. Dagua doesn't. Small line-local fix.
2. **Force-norm calculation**: graphviz `Fnorm` sums per-node force magnitudes;
   dagua uses different aggregation. Tied to cooling.
3. **Per-iteration recentering**: dagua does it; graphviz doesn't. Remove or
   gate behind a compatibility option.
4. **Quadtree threshold**: graphviz uses 45; dagua uses different. Lower or
   parameterize.

Items 5-7 (sequential update, component packing, matrix coarsening) are
larger rewrites — defer.

## Your job

Apply items 1-4 in order. For each:
- Make the smallest line-local change that matches graphviz behavior under
  fidelity_mode=True (or equivalent flag for sfdp).
- Re-run the bounded measurement after each item to confirm direction.
- If item regresses median by >0.005, revert that item and document.

Keep individual commits per item or bundle related items.

## Reference

- `.project-context/research/sprint_algo_fidelity/ROUND_27_DIFF_sfdp.md` (full
  ranked list with file:line on both sides)
- Graphviz source: `/home/jtaylor/projects/_references/graphviz/lib/sfdpgen/`
- Dagua: `dagua/layout/ops/pipelines/sfdp.py`

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_sfdp graphviz_sfdp \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_28/sfdp/{baseline,post_fix}
```

Required: median improvement >= 0.005 OR worst-graph reduction >= 0.05. If
neither, write principled_residual SUMMARY.

## Scope

- DO NOT TOUCH render/styles, cluster sprint files, parallel sprint files
- Stage commits explicitly (`git add <files>`); commit format
  `feat(fidelity): round 28 sfdp -- <terse>`
- Multiple micro-commits OK
- After each commit: `pytest tests/test_layout/ -x --tb=short -q -k "sfdp"`

## Output

Per-round SUMMARY at `eval_output/algo_fidelity/round_28/sfdp/SUMMARY.md`.

</task>

<completeness_contract>
Apply R27 items 1-4. Either measurable improvement OR explicit per-item
revert with rationale. SUMMARY mandatory.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
