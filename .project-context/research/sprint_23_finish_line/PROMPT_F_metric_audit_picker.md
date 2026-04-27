# Sprint 23 Area F: Metric audit + picker tuning

## Mandate

Some close-losses might be metric-noise-driven rather than algorithm-
limited. Sprint-22b da58b14 fixed the seed but sampled metrics
(crossing_rate at 1M samples, neighborhood_preservation at sparse
samples) still have small variance. Tightening these or using exact
counts on small graphs might re-classify 1-2 close-losses as ties.

Additionally, the picker's margin gate is fixed at 0.5. For some
graphs a margin of 0.3 or 0.7 might be optimal -- 0.5 was set
empirically in sprint-20k and not re-tuned since.

## Research questions

1. On the 5 close-losses (small_world_500, clustered_medium_5x20,
   outerplanar_dag_20, multi_component_80, hexagonal_lattice_42),
   re-score with crossing_samples=5_000_000 and exact crossing
   count for N <= 200. Does any close-loss become a tie under
   tighter scoring?

2. Audit the picker's 0.5 margin: sweep margin in {0.0, 0.1, 0.25,
   0.4, 0.5, 0.6, 0.75, 1.0} on the full 93-graph benchmark. Does
   any graph win MORE under a different margin? What's the
   winning margin in aggregate (best-or-tied %)?

3. Audit composite weights: dag_consistency=25, edge_length_cv=20,
   depth_spearman=15, overlap=10, edge_straightness=10,
   crossing_rate=10, angular_resolution=5, cluster_separation=5.
   Sprint-19/20 may have left the weights at non-optimal values.
   On the 6-graph close-loss bucket, would a small re-weight
   (+/- 5 on any one term) flip outcomes? IMPORTANT: any re-weight
   must hold the win bucket constant (no regressions).

## Output spec

File: `.project-context/research/sprint_23_finish_line/F_metric_audit_picker__<agent>.md`

Sections:
- TL;DR
- Tighter-scoring re-classification table (5 close-losses x 2
  scoring tiers: current vs tightened)
- Picker margin sweep result table
- Composite weight sensitivity analysis
- Recommended adjustments (if any) and their net effect on
  best-or-tied %

## Constraints

- READ-ONLY on dagua/
- HEAD = sprint-22e finalize commit `d27fced`
- Use existing dagua.metrics.composite + dagua.metrics.full
- Use /tmp/h2h_buckets.py as the basis for full-suite scoring
- This is the ONLY area that can recommend metric changes; if it
  recommends them, the change must be empirically backed (not
  philosophical)

## Word budget

1500-2500 words.
