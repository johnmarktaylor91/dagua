<task>
R33 drl edge cutting + scheduler boundaries (deeper alignment past R31 F5/R32 ab7cfaa).

R31 Claude PLAN identified F5 (one-sided cut) — landed in commit ab7cfaa.
Remaining items past F5:

## DE1: candidate acceptance rule (per Claude F-5+ analysis)
Reference scoring uses current-node degree; dagua uses neighbor degree. R20 attempted partial fix; revert measured 0.0174 delta but didn't commit. Try again with cleaner scope.

## DE2: scheduler boundary sweeps
Reference at `drl_layout.cpp:240-480` calls `update_nodes()` BEFORE stage control on every `ReCompute()`. Dagua does not execute these init/boundary/final sweeps.

## DE3: parallel/multiedge handling
igraph map overwrites duplicate neighbor weights; dagua sums them.

## Read

- `eval_output/algo_fidelity/round_31/drl/PLAN_claude.md` sections F5-F8
- `eval_output/algo_fidelity/round_31/drl/PLAN_jtaylor_*.md`
- `/home/jtaylor/projects/_references/igraph/src/layout/drl/drl_graph.cpp` lines 909-1135 (acceptance), 1257-1268 (multiedge)

## Implementation

Each item separate commit via commit-safe wrapper:
1. `fix(layout): round 33 drl -- candidate acceptance current-node degree`
2. `fix(layout): round 33 drl -- scheduler boundary sweeps`
3. `fix(layout): round 33 drl -- multiedge overwrite semantics`

Measure after each on extended bounded subset (add small_world_100, scale_free_ba_120):
```bash
python scripts/algo_fidelity_live_compare.py classic_drl igraph_drl --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels,small_world_100,scale_free_ba_120 --output-dir eval_output/algo_fidelity/round_33/drl_edge_deep/post_<item>
```

Revert any item that regresses >0.005.

## Output
`eval_output/algo_fidelity/round_33/drl_edge_deep/SUMMARY.md`.
</task>

<completeness_contract>
At least 1 item committed if any work. Revert items that regress.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
