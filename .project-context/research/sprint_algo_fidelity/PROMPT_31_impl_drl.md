<task>
Round 31 IMPLEMENTATION for drl.

Read first:
- eval_output/algo_fidelity/round_31/drl/PLAN_claude.md
- eval_output/algo_fidelity/round_31/drl/PLAN_jtaylor_zmachine_20260524_174604.md
- eval_output/algo_fidelity/round_31/ROUND_31_INTEGRATED_PLAN.md (section A2)

## Implement (in priority order, commit each as you go)

### F1: FINAL + REFINE preset table fix (~10 LoC)
`dagua/layout/ops/drl.py:217` (FINAL) and adjacent REFINE preset. igraph has at `drl_layout.cpp:380-388`:
```c
options->expansion_iterations   = 50;
options->expansion_temperature  = 50;
options->expansion_attraction   = .1;
options->expansion_damping_mult = .25;
```
Dagua has `_PhaseParameters(50, 2000.0, 2.0, 1.0)` — 40x temp, 20x attraction wrong.

### F2: Init range mismatch (~5-10 LoC)
Reference adapter uses NumPy `uniform(-1, 1)` seed matrix; dagua uses Python `random()` in `[0, 1)`. Add `fidelity_mode` that accepts a seed matrix and uses [-1, 1] init.

### F3: Jump sign (1 line)
`drl_graph.cpp:939-941` has `(.5 - RNG_UNIF01()) * jump_length` (positive bias subtraction); dagua has `rng.uniform(-0.5, 0.5)`. Match the reference sign convention.

### F4: Edge cutting semantics (~30 LoC)
Per `drl_graph.cpp:1130-1133`: igraph cuts only the current node's neighbor map. Dagua removes symmetrically. Match igraph's one-sided cut.

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_drl igraph_drl \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_31/drl/post_impl
```

Expected: median RMSD 0.12 -> ~0.06 after F1+F2+F3; drl_final 0.165 -> ~0.05.

## Scope
- DO NOT TOUCH: render/styles, cluster sprint, existing fidelity_report or benchmark_100seed_final outputs
- Explicit git add. Commit format: `fix(layout): round 31 drl -- <terse>`.

## Output
`eval_output/algo_fidelity/round_31/drl/SUMMARY.md` with before/after.
</task>

<completeness_contract>
F1 + F2 + F3 minimum. F4 if cheap. Multi-commit OK.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
