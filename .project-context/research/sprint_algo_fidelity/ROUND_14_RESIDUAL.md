# Round 14 Residual -- DRL

Status: RESIDUAL
Family: drl
Date: 2026-04-30

## Classification

`numerical_residual / stochastic_floor_partial`

The small-subset default run is close to igraph's own multi-seed floor on most
graphs, but not cleanly equivalent across the subset. Baseline graph-level TOST
was equivalent at `1x` on four of five evaluated graphs; the exception was
`parallel_multiedge_bundle`, which was only equivalent at `1.5x`.

## Why No Commit Landed

The strongest default-path lever tested was igraph's two-candidate node
acceptance rule. It improved median RMSD but did not meet the sprint's commit
criterion:

```text
baseline: 0.206197
attempt:  0.188797
delta:    0.017400
required: >= 0.030000
```

The attempted code change was reverted, leaving only measurement and research
artifacts for Round 14.

## Remaining High-Confidence Leads

- `classic_drl_final` has a clear phase-parameter mismatch against igraph's
  `FINAL` preset. This should be targeted only with a `classic_drl_final`
  live compare, not the default compare used in Round 14.
- Edge cutting differs: igraph removes the selected long edge from only the
  current node's neighbor map, while dagua removes it symmetrically. This may be
  a larger lever but is more invasive because it changes the directed traversal
  state across later node updates.
- Density-grid boundary behavior differs. igraph returns high density near grid
  edges; dagua clamps to the nearest cell. This is likely lower priority for the
  current small graphs than node acceptance and edge cutting.
