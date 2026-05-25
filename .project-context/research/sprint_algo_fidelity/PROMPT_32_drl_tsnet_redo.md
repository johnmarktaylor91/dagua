<task>
R32 RE-DO drl + tsnet (small-commit version to dodge hook rollback).

Background: R31 codex landed both family fixes but pre-commit hook rolled them back during commit. Working tree was lost. Re-implement carefully.

## drl (smaller scope this time)

Read first:
- eval_output/algo_fidelity/round_31/drl/PLAN_claude.md (963 lines)
- eval_output/algo_fidelity/round_31/drl/PLAN_jtaylor_zmachine_20260524_174604.md
- eval_output/algo_fidelity/round_31/drl/SUMMARY.md (if exists)

Implement ONLY the clearly-bug fixes:
1. **FINAL preset table** in `dagua/layout/ops/drl.py` (around line 217): change `_PhaseParameters(50, 2000.0, 2.0, 1.0)` to match igraph `drl_layout.cpp:380-388` values (50, 50, 0.1, 0.25). REFINE preset similar.
2. **Jump sign** (1 line): match igraph `drl_graph.cpp:939-941` `(.5 - RNG_UNIF01()) * jump_length` not `rng.uniform(-0.5, 0.5)`.

SKIP for now (more invasive): init range, edge cutting, density grid. They regressed bounded subset in R31.

Commit as `fix(layout): round 32 drl -- preset + jump sign`. Single commit, small diff, less hook surface.

## tsnet

Read first:
- eval_output/algo_fidelity/round_31/tsnet/PLAN_claude.md (note: Claude's c=4 claim was empirically disproved by codex during R31)
- eval_output/algo_fidelity/round_31/tsnet/PLAN_jtaylor_zmachine_20260524_174631.md
- eval_output/algo_fidelity/round_31/tsnet/SUMMARY.md

Implement ONLY:
1. NumPy `RandomState` init in fidelity_mode (vs torch.Generator) -- aligns with sklearn
2. sklearn-style convergence checks (grad_norm <= 1e-7 or 300-iter no-progress)

SKIP: `c=4` gradient scale (codex empirically showed dagua autograd already matches sklearn at scale 1.0).

Commit as `fix(layout): round 32 tsnet -- numpy init + sklearn convergence`.

## Verification (each)

```bash
python scripts/algo_fidelity_live_compare.py classic_drl igraph_drl --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_32/drl/post_impl
python scripts/algo_fidelity_live_compare.py classic_tsnet tsne_graph --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_32/tsnet/post_impl
```

If a fix regresses bounded RMSD by >0.01 vs baseline, document and skip committing THAT specific item.

## Scope
- DO NOT TOUCH: render/styles, cluster sprint, existing benchmark_100seed_final outputs
- Stage with explicit `git add <files>`. Use small, focused commits to avoid hook rollback.
- Run `pytest tests/test_layout/ -x --tb=short -q -k "drl or tsnet"` after each commit.

## Output
SUMMARY at `eval_output/algo_fidelity/round_32/drl/SUMMARY.md` and `eval_output/algo_fidelity/round_32/tsnet/SUMMARY.md`.
</task>

<completeness_contract>
At least 1 commit per family (drl preset, tsnet init). Multiple micro-commits OK.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
