<task>
R33 IMPLEMENT igraph-compatible MT19937 port (architectural floor closer).

R32 mt19937 research codex recommended deferring. User explicitly said "no more punting". Implement.

## Scope

Port igraph's MT19937 to pure Python. ~220-390 LoC per codex estimate. Used by drl, lgl, graphopt fidelity_mode.

## Read first

- `eval_output/algo_fidelity/round_32/mt19937_bridge/REPORT.md` (R32 research codex output)
- `/home/jtaylor/projects/_references/igraph/src/random/random.c` (high-level RNG interface)
- `/home/jtaylor/projects/_references/igraph/src/random/rng_mt19937.c` (actual MT19937 implementation)
- `/home/jtaylor/projects/_references/igraph/src/random/rng_pcg32.c` (default PCG32; reference codex flagged this as the actual local default, but task focus is MT19937 for paired comparison)

Confirm which RNG igraph_layout_drl actually uses (call site → which generator). If MT19937: implement MT19937. If PCG32: implement PCG32 instead. Whichever is in the actual `igraph_layout_drl` path.

## Implement

1. Create `dagua/layout/ops/_igraph_rng.py` (or wherever fits):
   - Class `IgraphMT19937` (or `IgraphPCG32` depending on which igraph uses)
   - Seed function, advance function, uniform double, uniform int (Lemire bounded)
   - Bit-exact match to igraph's draw sequence
2. Add regression tests with golden vectors (capture from igraph C output or document the canonical seed→sequence).
3. Wire into drl, lgl, graphopt fidelity_mode (replace `random.Random(seed)` with the bridge).

## Verify

After implementation:
```bash
python scripts/algo_fidelity_live_compare.py classic_drl igraph_drl --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels,small_world_100 --output-dir eval_output/algo_fidelity/round_33/mt19937_port/drl_post
python scripts/algo_fidelity_live_compare.py classic_graphopt igraph_graphopt --seeds 30 --graphs ... --output-dir eval_output/algo_fidelity/round_33/mt19937_port/graphopt_post
python scripts/algo_fidelity_live_compare.py classic_lgl igraph_lgl --seeds 30 --graphs ... --output-dir eval_output/algo_fidelity/round_33/mt19937_port/lgl_post
```

Expected: closing of architectural RNG floor → strong_equivalent on graphs where init range / draw order is dominant.

## Implementation strategy

Commit incrementally:
1. RNG class + tests (no engine wiring): `feat(layout): round 33 mt19937 -- pure-python rng`
2. Wire to drl: `fix(layout): round 33 drl -- mt19937 rng bridge`
3. Wire to graphopt: `fix(layout): round 33 graphopt -- mt19937 rng bridge`
4. Wire to lgl: `fix(layout): round 33 lgl -- mt19937 rng bridge`

Use commit-safe wrapper.

## Output
`eval_output/algo_fidelity/round_33/mt19937_port/SUMMARY.md` with RMSD deltas per engine.
</task>

<completeness_contract>
At minimum: RNG class + tests. Engine wiring if successful.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
