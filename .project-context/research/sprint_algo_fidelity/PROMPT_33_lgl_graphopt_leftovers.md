<task>
R33 IMPLEMENTATION for lgl/graphopt remaining items + ogdf_competitor del-seed bug.

## lgl leftovers (anything past L1-L4 from R31 Claude PLAN)

Read `eval_output/algo_fidelity/round_31/lgl/PLAN_claude.md`. R31 lgl codex applied L1-L4. Plan may have more items (L5+). Implement remaining ones if they exist and look correct.

## graphopt G3: zero-distance predicate parity

Reference at `/home/jtaylor/projects/_references/igraph/src/layout/graphopt.c` checks `if (d2 < 1e-5)`. Dagua may use different epsilon. Match.

Files: `dagua/layout/ops/force.py` (GraphOpt* ops).

## ogdf_competitor del-seed bug

Older TODO: `dagua/eval/competitors/ogdf_competitor.py:203` had a `del seed` line that drops the seed before passing to OGDF runner. R28 fixed it for fmmm/gem/stress/maxent_stress/pivot_mds via runner rebuild. Check if there are OTHER OGDF engines still affected (e.g., ogdf_davidson_harel, ogdf_sugiyama).

Inspect `dagua/eval/competitors/ogdf_competitor.py` line ~203 (and other places where `del seed` or `seed=` is dropped). If any engine still drops seed, thread it through like R28 did.

## Implementation

Use commit-safe wrapper. Each fix its own commit.

Verification per family (extended subset):
```bash
python scripts/algo_fidelity_live_compare.py classic_lgl igraph_lgl --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels,small_world_100,scale_free_ba_120 --output-dir eval_output/algo_fidelity/round_33/lgl/post_impl
python scripts/algo_fidelity_live_compare.py classic_graphopt igraph_graphopt --seeds 30 --graphs ... --output-dir eval_output/algo_fidelity/round_33/graphopt/post_impl
```

## Output
`eval_output/algo_fidelity/round_33/lgl_graphopt_leftovers/SUMMARY.md`.
</task>

<completeness_contract>
Apply remaining items per spec. Revert if regress.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
