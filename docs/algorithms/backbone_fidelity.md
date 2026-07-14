# Backbone Fidelity

## Status

`algorithm="backbone"` implements the graphlayouts-style sparsify-then-stress
pipeline without runtime R delegation.

- Reference: `schochastics/graphlayouts::layout_as_backbone`
- Runtime delegation: none
- Verification: `python scripts/verify_backbone_fidelity.py`
- Current residual name: `oaqc_edge_orbit_embeddedness_when_reference_unavailable`

## Method

The native pipeline has two bisection stages:

1. `backbone_compute`: score edges by simmelian embeddedness, apply
   graphlayouts' maximum-prefix Jaccard reweighting, union the keep-fraction
   filter with the union maximum spanning tree, and expose selected backbone
   edges.
2. `backbone_stress`: run a graphlayouts-compatible dense stress sweep on the
   selected backbone graph, including MDS initialization, fixed stress seed 42,
   disconnected component layout, and row packing.

The graphlayouts R reference additionally uses `oaqc::oaqc(...,
non_ind_freq=TRUE)` edge orbit column 11 before prefix-Jaccard reweighting.
Dagua does not add an `oaqc` runtime dependency, so the native port uses
common-neighbor embeddedness for that simmelian edge score. This is the first
named stage that can diverge from exact graphlayouts when `oaqc` is available.

## Latest Local Verification

On this environment, `Rscript` is present, but `graphlayouts`/`oaqc` were not
fully available during implementation. The verifier therefore reports
clean-room quality tiers unless those packages are installed.

Expected verifier output shape:

```text
reference_r_package_ran: no
named_residual: oaqc_edge_orbit_embeddedness_when_reference_unavailable
graph,residual,tier,quality,backbone_edge_set_matched
path4,NA,quality-tier-clean-room,...
cycle_diagonal,NA,quality-tier-clean-room,...
triangle_tail,NA,quality-tier-clean-room,...
two_components,NA,quality-tier-clean-room,...
```

When `igraph`, `graphlayouts`, and `oaqc` are installed in R, the same script
runs the reference adapter and reports rotation-invariant Procrustes residuals,
per-graph tier, quality score, and whether the selected backbone edge set
matches.
