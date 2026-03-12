# Benchmark Artifact Guide

This is the short answer to: which benchmark artifact should I read, and why?

## First Read

If you only open one file while iterating on placement, open:

- `eval_output/report/placement_dashboard.md`

That is the fastest way to answer:

- is Dagua winning or losing on placement?
- on which graphs?
- on which core metrics?

## Artifact Map

### `benchmark_deltas.md`

Use when:

- comparing the latest run against the previous run
- asking whether a recent code change helped or hurt overall

Best for:

- round-over-round engineering iteration

### `layout_similarity.md`

Use when:

- asking whether Dagua and another engine actually found different layouts
- separating geometry differences from styling differences

Best for:

- intellectually honest interpretation
- avoiding fake narratives based on rendering alone

### `placement_summary.md`

Use when:

- you want a clean placement-only summary without visual-language discussion
- you want per-graph metric snapshots across competitors

Best for:

- placement review
- benchmark reading without aesthetic noise

### `placement_dashboard.md`

Use when:

- you are in the trenches and need the fastest view of wins/losses
- the question is simply “how is Dagua’s placement doing?”

Best for:

- optimization sprints
- deciding what graph/metric to target next

### `scaling_curve.png`

Use when:

- telling the performance story
- showing where competitors stop and Dagua continues

Best for:

- scale narrative
- external communication

### `visuals/comparisons/`

Use when:

- you need side-by-side output reference from competitor engines
- you want to inspect actual rendered layouts, not just metrics

Best for:

- visual iteration
- style/language comparison

## Recommended Reading Order

For placement iteration:

1. `placement_dashboard.md`
2. `placement_summary.md`
3. `benchmark_deltas.md`
4. `layout_similarity.md`
5. selected visuals

For visual redesign:

1. competitor stepwise visual audit
2. decomposition / kill-switch audit panels
3. frozen baseline diffs
4. only then the broader gallery

For scale storytelling:

1. `scaling_curve.png`
2. rare-suite benchmark results
3. poster/tour assets
