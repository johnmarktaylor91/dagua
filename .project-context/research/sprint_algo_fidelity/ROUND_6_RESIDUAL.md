# Round 6 Residual -- SFDP

## Classification

`attempted_lever_no_signal: graphviz_sfdp_attractive_distance_factor`

## Baseline

Live `classic_sfdp` vs `graphviz_sfdp` on the current node-size context:

| Metric | Value |
|---|---:|
| graphs | 24 |
| median RMSD | 0.091496 |
| p25 | 0.036583 |
| p75 | 0.208596 |
| p95 | 0.417776 |
| worst | center_port_backedge_hub 0.475146 |

This matches the Round 1 cached SFDP family baseline of roughly 0.0915.

## Source Comparison

Graphviz SFDP source confirms:

- `random_start = true`, `random_seed = 123`
- `C = 0.2`
- `bh = 0.6`
- `maxiter = 500`
- `step = 0.1`
- `adaptive_cooling = true` initially
- `tol = 0.001`
- `K < 0` resolves to `average_edge_length(A, x)`
- finer levels use `ctrl->K = ctrl->K * 0.75`, `ctrl->adaptive_cooling = false`, `ctrl->step = .1`

Dagua already matches the major defaults and the multilevel K decay / finer-level
adaptive-cooling switch. Dagua also initializes the coarsest graph from seeded
uniform random positions.

## Attempted Lever

Graphviz applies attractive force as:

```text
CRK * (x_j - x_i) * ||x_i - x_j||
```

Dagua's `_spring_forces` used:

```text
CRK * (x_j - x_i)
```

I tested adding the missing distance factor as a single focused force-law
alignment. The patch was reverted because it regressed the median.

## Measurements

| Metric | Baseline | Attempted | Delta |
|---|---:|---:|---:|
| median RMSD | 0.091496 | 0.106249 | +0.014753 |
| worst RMSD | 0.475146 | 0.479776 | +0.004630 |
| worst graph | center_port_backedge_hub | center_port_backedge_hub | unchanged |

The post-fix run was repeated once and produced the same median and worst graph,
so this was not stochastic drift at the 0.01 threshold.

Largest attempted regressions:

| Graph | Baseline | Attempted | Delta |
|---|---:|---:|---:|
| extreme_mixed_width_transformer | 0.046919 | 0.114633 | +0.067713 |
| hierarchical_residual_stage | 0.064534 | 0.082151 | +0.017617 |
| residual_block | 0.040841 | 0.053372 | +0.012530 |
| center_port_backedge_hub | 0.475146 | 0.479776 | +0.004630 |

Simple-graph regression check:

| Graph | Baseline | Attempted | Delta |
|---|---:|---:|---:|
| linear_3layer_mlp | 0.005576 | 0.008567 | +0.002991 |
| nested_shallow_enc_dec | 0.005578 | 0.008571 | +0.002993 |

Both stayed below 0.02, but the median regression fails the Round 6 commit
criterion.

## Recommendation

Stay on SFDP for Round 7 because `flail_count_sfdp=1`. The next higher-confidence
lever is not another scalar force coefficient; it is likely a trajectory-level
mismatch:

- Graphviz updates nodes sequentially in the slow / hybrid-small solver, so later
  nodes in an iteration see earlier nodes' moved coordinates. Dagua computes
  vectorized synchronous forces from the previous full position tensor.
- Graphviz C `drand()` / `gv_permutation()` may produce different random streams
  from Torch even with seed 123, affecting coarsening and initial coordinates.

Try the sequential-update alignment first only if it can be done narrowly for
small graphs without wholesale SFDP rewrite. Otherwise classify the remaining
gap as an implementation-trajectory residual and move to neato.
