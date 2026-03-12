# Money Graphs

These are the graphs to look at first when judging progress.

Do not start every iteration by staring at the whole corpus. Start here.

## Placement Money Graphs

These are the graphs most useful for judging whether Dagua's node placement is actually improving.

- `residual_block`
  - small, readable, skip-connection sanity check
- `long_range_residual_ladder`
  - tests long skip bridges over a deep backbone
- `interleaved_cluster_crosstalk`
  - tests whether cluster-heavy structure stays legible under cross-links
- `multiscale_skip_cascade`
  - tests cross-scale alignment and repeated handoffs
- `hub_skip_superfan`
  - tests hub-dominated skip pressure

Use with:

- `placement_dashboard.md`
- `placement_summary.md`
- competitor stepwise visual audit

## Visual-Language Money Graphs

These are the graphs most useful for judging the eventual text/edge/cluster redesign.

- `mixed_width_labels`
  - text hierarchy pressure
- `small_label_storm`
  - edge labels + cluster labels in a cramped space
- `nested_cluster_label_stack`
  - nested containment + cluster-label discipline
- `kitchen_sink_hybrid_net`
  - many visual features at once
- `edge_label_braid`
  - whether edge language becomes clear or collapses into clutter

Use with:

- decomposition views
- kill-switch matrices
- typography / edge-language sheets
- frozen baseline diffs

## Scale / Showcase Money Graphs

These are not the best for pixel-polish judgment. They are the best for telling the Dagua story.

- `scale_20k`
  - medium-scale transition point
- `scale_50k`
  - large but still benchmark-friendly
- `scale_100k`
  - edge of normal competitor comfort
- `scale_500000` and above
  - scaling story
- `1b` benchmark run
  - the “why Dagua exists” case

Use with:

- scaling curve
- poster / tour exports
- benchmark report

## Rule

If a change looks good only on a graph that is not in this list, do not trust it yet.
