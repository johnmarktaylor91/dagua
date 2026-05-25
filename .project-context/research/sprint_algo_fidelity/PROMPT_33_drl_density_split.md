<task>
R33 IMPLEMENTATION for drl density grid (split into 3 items, apply one at a time).

R32 drl_edge codex tried F6+F7 together → regressed mixed_width_labels (0.089 → 0.106) → reverted both. Try splitting and measuring each:

## DG1: Separable product kernel (~30 LoC)
Reference uses separable product (kx * ky) for density contribution. Dagua uses radial cone. Per Claude R31 plan.
Files: `dagua/layout/ops/drl.py` (look for DensityGrid op or similar)
Reference: /home/jtaylor/projects/_references/igraph/src/layout/drl/DensityGrid.cpp

## DG2: Boundary penalty + bin-throw (~20 LoC)
Reference: nodes near grid boundary get small penalty + throw to nearest interior cell.
Dagua: clamps to boundary.

## DG3: Fine-bin lifecycle (~15 LoC)
Reference: fine bins populated ONLY after `fineDensity=true` flag flips (during phase 2+).
Dagua: always populates fine bins.

## Implementation strategy

**APPLY ONE AT A TIME, MEASURE EACH, REVERT IF REGRESSES.**

Use commit-safe wrapper. Commit each as `fix(layout): round 33 drl_density -- <item>`.

Bounded subset (extended): see scripts/larger_subset_verify.sh
```bash
python scripts/algo_fidelity_live_compare.py classic_drl igraph_drl --seeds 30 --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels,asymmetric_hourglass_hub,small_world_100,scale_free_ba_120 --output-dir eval_output/algo_fidelity/round_33/drl_density/<phase>
```

## Output
`eval_output/algo_fidelity/round_33/drl_density/SUMMARY.md` with per-item before/after.
</task>

<completeness_contract>
Try all 3, keep only ones that don't regress >0.005 on the extended bounded subset.
</completeness_contract>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation and keep going.
</default_follow_through_policy>
