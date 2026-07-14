# Backbone Fidelity

## Status

`algorithm="backbone"` implements the graphlayouts-style sparsify-then-stress
pipeline without runtime R delegation.

- Reference: `schochastics/graphlayouts::layout_as_backbone`
- Runtime delegation: none
- Verification: `python scripts/verify_backbone_fidelity.py`
- Current tier: `reference-verified-partial`
- Current residual name: `stress_initialization_mds_rng_parity`

## Method

The native pipeline has two bisection stages:

1. `backbone_compute`: score edges by the same non-induced C4 edge orbit used
   by `oaqc::oaqc(..., non_ind_freq=TRUE)$e_orbits_non_ind[, 11]`, apply
   graphlayouts' maximum-prefix Jaccard reweighting, union the keep-fraction
   filter with the union maximum spanning tree, and expose selected backbone
   edges.
2. `backbone_stress`: run a graphlayouts-compatible dense stress sweep on the
   selected backbone graph, including MDS initialization, fixed stress seed 42,
   disconnected component layout, and row packing.

Dagua computes the `oaqc` edge orbit directly as the count of distinct
length-3 paths between each edge's endpoints. It does not add an `oaqc` or R
runtime dependency.

## Latest Local Verification

On this environment, `Rscript` is present and the R reference stack installs
and runs. `oaqc` installed from CRAN. `igraph` and `graphlayouts` installed
from CRAN after installing `lattice` from CRAN and `Matrix 1.6-5` from the CRAN
archive, because current CRAN `Matrix` is not available for R 4.2.

Measured verifier output:

```text
reference_r_package_ran: yes
named_residual: stress_initialization_mds_rng_parity
graph,residual,tier,quality,backbone_edge_set_matched
path4,0.00586404,partial,48.561,yes
cycle_diagonal,0.000936847,similarity-exact,31.877,yes
triangle_tail,0.0058558,partial,34.342,yes
two_components,0.0189826,partial,58.480,yes
overall_tier: reference-verified-partial
all_backbone_edge_sets_matched: yes
```

The backbone edge set now matches the R reference on the verification corpus.
The remaining residual is isolated to stress initialization parity:
graphlayouts uses `igraph::layout_with_mds(...) + stats::runif(...)` under R's
seed 42, while Dagua uses a NumPy-native MDS and RNG initialization before the
same stress majorization update.
