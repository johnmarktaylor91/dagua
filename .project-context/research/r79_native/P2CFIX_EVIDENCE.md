# P2c Row Spacing Fix Evidence

Date: 2026-07-06
Branch: `r79/p2c-fix`

## Change Summary

- Added row-adjacent minimum center spacing after `ClusterAwareXCompaction`.
- Added the same final spacing repair after `RankRowSnap`.
- Default row gap is `0.35 * node_sep`; the benchmark default is `24.5`.
- Required same-row center spacing is `max(width_i, width_j) + min_gap`.

## Affected-Graph Composite Gate

Before rows are from `eval_output/r79_baseline` at the branch point. After rows
are from rerunning Dagua with this patch and rescoring with corrected semantics.

| Graph | Before | After | Delta | Best External | After vs Best | Overlaps After |
| --- | ---: | ---: | ---: | --- | ---: | ---: |
| dependency_500 | 54.555887 | 54.555887 | 0.000000 | elk_layered 55.657253 | -1.101366 | 0 |
| dependency_graph_100 | 57.114050 | 57.114050 | 0.000000 | elk_layered 56.202980 | 0.911069 | 0 |
| clustered_medium_5x20 | 65.272650 | 65.306764 | 0.034114 | graphviz_dot 66.863664 | -1.556901 | 0 |
| r79_nested_clusters_3x2x10 | 70.900095 | 70.921482 | 0.021387 | graphviz_dot 71.276253 | -0.354772 | 0 |

No affected graph dropped by more than 0.5 composite points.

## De-Collision Ratios

Ratio is `min(row-adjacent center distance / required spacing)`, grouping exact
rank rows from the rendered positions. `inf` means no multi-node snapped rows.

| Graph | Before Ratio | Before Distance | Before Required | After Ratio | After Distance | After Required |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| dependency_500 | 1.384050 | 111.103516 | 80.274220 | 1.384050 | 111.103516 | 80.274220 |
| dependency_graph_100 | inf | inf | inf | inf | inf | inf |
| clustered_medium_5x20 | 2.798469 | 305.139160 | 109.037888 | 2.798469 | 305.139160 | 109.037888 |
| r79_nested_clusters_3x2x10 | inf | inf | inf | inf | inf | inf |

The two required graphs are both >= 1.0 after the fix.

## Item 2: Overlap-Metric Blindness Verdict

Verdict: (a) true node boxes do not overlap on the post-P2c
`dependency_500` positions used by the benchmark.

Evidence:
- `count_overlaps_detailed(post_p2c_dependency_500_pos, true_node_sizes, seed=42)`
  returned `{"overlap_count": 0}`.
- `dependency_500` has 500 nodes, so `dagua/metrics.py:492` uses the exact
  pairwise branch at `dagua/metrics.py:531`, not the spatial-hash sampling
  branch. This is not a sampling miss.
- The visual bars are therefore consistent with render-marker or near-touching
  opacity, not true benchmark-box overlap. The row-spacing cap remains useful
  as a visual guardrail for future snapped/compacted rows.

## Jitter Check

Gaussian positional jitter, sigma 0.5, 8 trials, full composite rescoring.

| Graph | Min Delta | Max Delta | Max Abs Delta |
| --- | ---: | ---: | ---: |
| dependency_500 | -0.079537 | -0.078949 | 0.079537 |
| dependency_graph_100 | -0.003623 | 0.001491 | 0.003623 |
| clustered_medium_5x20 | -0.026457 | -0.000988 | 0.026457 |
| r79_nested_clusters_3x2x10 | -0.004678 | 0.004393 | 0.004678 |

All jitter deltas stayed well inside the 0.5 tie band.

## Full Sweep

Command:

```bash
timeout 9000 .venv/bin/python scripts/r79_baseline.py --dagua-only
timeout 9000 .venv/bin/python scripts/r79_baseline.py --dagua-only --resume
```

The first command reached the 9000 second wrapper timeout while near the end of
the corpus. The second command resumed the same staging store and completed.
Final W/T/L:

| Population | W | T | L |
| --- | ---: | ---: | ---: |
| legacy | 56 | 8 | 29 |
| extended | 8 | 2 | 5 |

This matches the branch-point baseline and is not worse.

## Gallery

Rendered two-panel PNGs are in:

`.project-context/research/r79_native/gallery_p2cfix/`

Files:
- `dependency_500.png`
- `dependency_graph_100.png`
- `clustered_medium_5x20.png`
- `r79_nested_clusters_3x2x10.png`
