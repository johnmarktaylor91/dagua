<task>
R32 drl edge-cutting + density grid alignment (deeper work).

R31 codex F-5 + F-6 attempted these but didn't land cleanly. Try again, smaller scope.

## Read

- eval_output/algo_fidelity/round_31/drl/PLAN_claude.md
- eval_output/algo_fidelity/round_31/drl/PLAN_jtaylor_zmachine_20260524_174604.md
- /home/jtaylor/projects/_references/igraph/src/layout/drl/drl_graph.cpp (full)
- /home/jtaylor/projects/_references/igraph/src/layout/drl/DensityGrid.cpp (full)

## Implement

### F5: Edge cutting one-sided semantics
Reference at `drl_graph.cpp:1130-1133`: `neighbors[ node_ind ].erase( maxIndex );` -- erases ONLY current node's neighbor map.
Dagua removes both directions. Match one-sided.

### F6: Density grid kernel
Reference uses separable product kernel (per Codex finding). Dagua uses radial cone. Port separable product.

### F7: Density grid boundary
Reference has boundary penalty/throws + fine bins populated only after `fineDensity=true`.
Dagua: always-populated fine buckets + `1e-12` fine guard.

## Verification

```bash
python scripts/algo_fidelity_live_compare.py classic_drl igraph_drl --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels --output-dir eval_output/algo_fidelity/round_32/drl/post_edge_cut
```

If RMSD regresses, document and revert specific items.

## Scope
- Small commits. Each item separate.
- Commit: `fix(layout): round 32 drl -- <terse>`.

## Output
`eval_output/algo_fidelity/round_32/drl/SUMMARY_edge_cut.md`.
</task>

<completeness_contract>
At least F5 (one-sided edge cut) if measurable.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
