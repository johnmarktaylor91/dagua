<task>
Round 31 IMPLEMENTATION for lgl.

Read first:
- eval_output/algo_fidelity/round_31/lgl/PLAN_claude.md
- eval_output/algo_fidelity/round_31/lgl/PLAN_jtaylor_zmachine_20260524_174616.md
- eval_output/algo_fidelity/round_31/ROUND_31_INTEGRATED_PLAN.md (A4)

## Implement (bundle, ~70 LoC)

### L1: Shell-1 placement
Dagua deterministically places shell-1 children at equal angles around root. Reading igraph carefully: that branch is dead code; igraph actually places shell-1 via `RNG_UNIF(-1, 1)` like all deeper shells. Replace deterministic-angle with uniform-random.

### L2: Init draw order
Dagua interleaves `(x0, y0, x1, y1, ...)`. igraph is column-major `(x0..xN, y0..yN)`. Match column-major in fidelity_mode.

### L3: Grid neighbor enumeration
igraph visits 4 cells `{(0,0), (+1,0), (0,+1), (+1,+1)}`. Dagua visits 5 cells including `(+1,-1)`. Match igraph's 4-cell pattern.

### L4: Repulsion epsilon
Dagua uses 1e-12 zero-distance fallback; igraph uses 1e-5. At pair collision dagua's repulsion is 10^7x larger. Match 1e-5.

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_lgl igraph_lgl \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_31/lgl/post_impl
```

Expected: median 0.13-0.15 -> ~0.05-0.08 (strong_equivalent).

## Scope
- DO NOT TOUCH: render/styles, cluster sprint, existing fidelity_report/benchmark outputs
- Explicit git add. Commit: `fix(layout): round 31 lgl -- <terse>`.

## Output
`eval_output/algo_fidelity/round_31/lgl/SUMMARY.md`.
</task>

<completeness_contract>
L1-L4 all. Bundle commit OK.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
