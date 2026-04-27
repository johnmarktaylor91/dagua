# Sprint 19 Improvement Scan -- Shared Context

## Goal
Exhaustively identify every actionable improvement dagua can make to:
1. Composite metric score on 93-graph benchmark_full (≤500 nodes).
2. Layout runtime (total time per graph).

Dagua currently beats every competitor on mean composite score (77.29 vs
dot 73.71, dagre 70.93, sugiyama 67.71, elk 65.87 after sprint-19a/b).
But we have specific per-graph losses we want to close.

## Current Head-to-Head (post sprint-19a; sprint-19b closes kitchen_sink -8)

Worst losses vs best competitor on each graph:
- disconnected_label_cycle_collage: n=7, cycles, dagua 62.08 vs elk 75.19 (-13)
- dependency_500: n=500, large sparse DAG, dagua 51.96 vs elk 62.82 (-11)
- kitchen_sink_hybrid_net: CLOSED by sprint-19b (now +0)
- small_world_100: n=100, cyclic flat, dagua 49.19 vs sugiyama 57.08 (-8)
- recurrent_feedback_cell: n=5, cycle, dagua 62.56 vs sugiyama 69.41 (-7)
- hexagonal_lattice_42: DAG, planar, dagua 82.42 vs dot 88.99 (-7)
- sierpinski_42: DAG, planar, dagua 78.35 vs dot 84.29 (-6)
- extreme_mixed_width_transformer: n=10 DAG, dagua 73.82 vs dagre 78.49 (-5)
- dense_pair_50: n=50 DAG dense, dagua 71.81 vs dot 76.33 (-5)
- small_world_500: n=500 cyclic flat, dagua 49.81 vs elk 54.26 (-4)

Biggest dagua wins (leave these alone, don't regress them):
- random_dag_200: +27, org_chart_deep: +23, random_dag_50: +22,
  hub_fanout_label_skew: +16, org_chart_1_5_4_8: +16

## Composite Metric Breakdown (dagua/dagua/metrics.py, `composite` at L1147)
Weights (sum=100):
- dag_consistency: 25
- edge_length_cv: 20 (LOWER is better, variance of edge length)
- depth_spearman: 15 (correlation between graph-depth and y-position)
- overlap_count: 10 (binary: 0 overlaps = 10, else 0)
- edge_straightness: 10 (LOWER deg from layer axis is better)
- crossing_rate: 10
- angular_resolution: 5
- cluster_separation: 5

## Key Files

Layout:
- dagua/layout/engine.py -- optimization loop + pipeline dispatch
- dagua/layout/init_placement.py -- topological sort + barycenter init
- dagua/layout/ops/pipelines/dagua_native.py -- current default pipeline
- dagua/layout/ops/ -- 268 composable primitives
- dagua/layout/cycle.py -- DFS back-edge + greedy FAS

Eval:
- dagua/eval/benchmark.py -- benchmark runner
- dagua/metrics.py -- all metric formulas + composite
- eval_output/variant_bench_full/positions/*.pt -- cached competitor positions

Benchmark:
- eval_output/variant_bench_full/ -- cached runs from 8 competitors
- /tmp/h2h2.py -- h2h script template
- /tmp/diag_single.py -- per-graph breakdown script

## Competitors (cached positions in eval_output/variant_bench_full/positions/)
- graphviz_dot -- Sugiyama-family hierarchical
- graphviz_sfdp -- scalable force directed
- elk_layered -- ELK (Eclipse layered)
- dagre -- hierarchical JS port
- igraph_sugiyama -- Sugiyama in igraph
- igraph_kamada_kawai -- KK force
- nx_spring -- networkx spring

## Output
Each research agent should write a structured findings file:
`.project-context/research/sprint_19_improvement_scan/<area>_<agent>.md`

With sections:
1. TL;DR (3 bullets)
2. Findings: each finding has {title, severity (high/med/low), evidence, proposed fix, expected impact}
3. Runtime observations (if applicable)
4. Recommended action queue ordered by expected impact per effort

## Reference commands
- Run dagua on one graph (with CPU only):
  CUDA_VISIBLE_DEVICES="" python /tmp/diag_single.py <graph_name>
- Inspect competitor positions:
  torch.load("eval_output/variant_bench_full/positions/<graph>__<engine>.pt")
