<task>
Round 31 IMPLEMENTATION for graphopt.

Read first:
- eval_output/algo_fidelity/round_31/graphopt/PLAN_claude.md
- eval_output/algo_fidelity/round_31/graphopt/PLAN_jtaylor_zmachine_20260524_174643.md
- eval_output/algo_fidelity/round_31/ROUND_31_INTEGRATED_PLAN.md (A6)

## Implement

### G1: Benchmark-reference init parity
Reference adapter passes `np.random.RandomState(seed).uniform(-1, 1, size=(N, 2))` row-major as seed matrix. Dagua's `GraphOptInitializePositions` uses Python `random.Random(seed)` sampling `[0, 1)` row-major.

Add a fidelity_mode that accepts an externally-provided init matrix and uses it instead of internal RNG. Wire `classic_graphopt` competitor to pass the same seed matrix.

### G2: Drop edge_weights from spring magnitudes (fidelity_mode)
`dagua/layout/ops/force.py:1471-1479`: dagua multiplies spring magnitudes by `edge_weights`. igraph GraphOpt ignores edge weights. Gate via fidelity_mode.

### G3: Zero-distance predicate parity
Match igraph's exact zero-distance epsilon.

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_graphopt igraph_graphopt \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_31/graphopt/post_impl
```

Expected: median 0.10-0.17 -> ~0.05-0.10 (some variants strong, others weak; architectural RNG floor remains).

## Scope
- DO NOT TOUCH: render/styles, cluster sprint, existing fidelity_report/benchmark outputs
- Explicit git add. Commit: `fix(layout): round 31 graphopt -- <terse>`.

## Output
`eval_output/algo_fidelity/round_31/graphopt/SUMMARY.md`.
</task>

<completeness_contract>
G1 + G2 minimum. G3 if cheap.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
