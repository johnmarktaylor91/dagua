# Round 41 Hungarian-Matched RMSD Summary

## Implementation

- Added `hungarian_matched_rmsd(positions_a, positions_b)` in
  `scripts/fidelity_analysis.py`.
- The metric uses the existing scale-normalized Procrustes behavior first,
  builds a pairwise point-distance cost matrix, and solves the assignment with
  `scipy.optimize.linear_sum_assignment`.
- The default Procrustes RMSD is unchanged. Hungarian RMSD is emitted beside it
  as an alternative geometric-only metric in:
  - `pairwise_similarity.csv`: `hungarian_rmsd`
  - `per_graph_detail.csv`: `hungarian_rmsd_mean`, `hungarian_rmsd_std`,
    `hungarian_rmsd_max`
  - `algorithm_summary.csv`: `hungarian_rmsd_mean`,
    `hungarian_rmsd_median`, `hungarian_rmsd_max`
- Exact assignment is guarded at 2,000 nodes; larger comparisons report `NaN`
  for Hungarian RMSD to avoid large quadratic cost matrices during full
  fidelity reruns.

## SFDP Smoke Check

Command: inline Python smoke using the Round 39 SFDP harness with
`classic_sfdp_graphviz_fidelity` against the Graphviz SFDP reference.

| Topology | Seed | Raw RMSD | Hungarian RMSD |
| --- | ---: | ---: | ---: |
| path | 1 | 0.023626077 | 0.023626077 |
| path | 2 | 0.019741366 | 0.019741366 |
| path | 3 | 0.013578818 | 0.013578818 |
| star | 1 | 0.165420600 | 0.000937754 |
| star | 2 | 0.002054337 | 0.002054337 |
| star | 3 | 0.164348903 | 0.002743147 |
| clustered | 1 | 0.000179138 | 0.000179138 |
| clustered | 2 | 0.052798253 | 0.000196836 |
| clustered | 3 | 0.000205787 | 0.000205787 |

## Interpretation

- Star seeds 1 and 3 reproduce the Round 39 diagnosis: raw Procrustes RMSD is
  high because symmetric leaves are label-permuted, while Hungarian RMSD drops
  to near the numerical floor.
- Path layouts are unchanged, which is expected when the label correspondence
  is already aligned.
- The clustered seed 2 drop shows the same metric-level behavior can apply
  outside the star case when a symmetric component is geometrically matched but
  label-assigned differently.

## Cross-Engine Spot Check

Same 7-leaf star graph, seed 1:

| Variant | Raw RMSD | Hungarian RMSD |
| --- | ---: | ---: |
| `classic_sfdp_graphviz_fidelity` | 0.165420600 | 0.000937754 |
| `classic_spectral_nx_fidelity` | 0.000000000 | 0.000000000 |
| `classic_kk_steps300` | 0.000000000 | 0.000000000 |

Only SFDP shows the symmetric-label residual in this smoke set. The other
engines stay unchanged, which is the expected no-op behavior when labels
already match.
