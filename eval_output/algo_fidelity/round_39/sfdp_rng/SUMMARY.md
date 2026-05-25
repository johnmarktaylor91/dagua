# Round 39 SFDP RNG Port Summary

## Graphviz RNG Identified

- `lib/sparse/general.c:24-25`: SFDP `drand()` is `rand() / (double) RAND_MAX`.
- `lib/util/random.c:15-32`: `gv_permutation(bound)` initializes `[0, bound - 1]`
  and applies Fisher-Yates swaps.
- `lib/util/random.c:36-58`: `gv_random(bound)` uses rejection sampling over
  `rand()` to avoid modulo bias.
- `lib/sfdpgen/spring_electrical.c:280-282`: coarsest random initialization
  calls `srand(ctrl->random_seed)` and fills `x[i] = drand()`.
- `lib/sfdpgen/spring_electrical.c:855-867`: prolongation jitter uses
  `drand() - 0.5`.
- `lib/sfdpgen/Multilevel.c:102-104`: unmatched-node coarsening order uses
  `gv_permutation(m)`.

## Implementation Summary

- Added `GraphvizRandom` in `dagua/layout/ops/sfdp.py`.
  - Ports the glibc `srand`/`rand` additive-feedback sequence used by the local
    Graphviz reference build.
  - Implements Graphviz `drand()`, `gv_random(bound)`, and `gv_permutation`.
- Wired `fidelity_mode="graphviz"` SFDP paths to the Graphviz RNG:
  - unmatched-node matrix coarsening permutation,
  - coarsest random placement,
  - prolongation sibling jitter.
- Mirrored Graphviz's RNG reset boundary:
  - matrix coarsening uses the process-default `rand` stream seeded as `1`,
  - coarsest random placement resets to `ctrl->random_seed` before `drand`.
- Added regression coverage for the `srand(1)` golden `rand()` sequence and
  `gv_permutation(8)`.
- Added `round_39/sfdp_rng/smoke_check.py` for path, star, and clustered
  topologies across seeds `1`, `2`, and `3`.

## Smoke RMSD

Before values are from
`eval_output/algo_fidelity/round_38/residual_debug/SUMMARY.md`.

| Topology | Before seeds 1/2/3 | After seeds 1/2/3 |
| --- | ---: | ---: |
| path | 0.023935233 / 0.022571208 / 0.017429141 | 0.023626077 / 0.019741366 / 0.013578818 |
| star | 0.353847238 / 0.296679793 / 0.356800226 | 0.165420600 / 0.002054337 / 0.164348903 |
| clustered | 0.044169177 / 0.000266634 / 0.039216829 | 0.000179138 / 0.052798253 / 0.000205787 |

## Residual Diagnosis

The RNG port reduces the star residual but does not hit the `<0.05` target for
seeds `1` and `3`. The remaining star residual is dominated by symmetric leaf
label permutation, not geometry:

| Seed | Raw star RMSD | Best leaf-permuted RMSD |
| --- | ---: | ---: |
| 1 | 0.165420600 | 0.000861034 |
| 2 | 0.002054337 | 0.001760757 |
| 3 | 0.164348903 | 0.001108516 |

This points to one remaining integration mismatch in node/leaf ordering after
random initialization or final orientation on hub-symmetric graphs. Output
normalization is still Dagua-specific, but Procrustes plus the leaf-permutation
check indicates the dominant residual is assignment/order rather than scale.

## Verdict

Numerical floor for path and most clustered cases; still-residual for symmetric
star seeds `1` and `3` due to leaf label permutation. Not bit-exact yet.
