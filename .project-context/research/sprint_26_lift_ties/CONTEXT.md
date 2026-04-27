# Sprint 26 — Victory Lap 1: Lift Ties Toward Wins

## Mandate

User directive 2026-04-26 01:30: "Do three more iterative autonomous
sprints going through every tie and mild win and seeing if you can turn
it into a major win. ... let's take some victory laps while we got the
time and tokens."

Sprints 22-25 achieved 100% best-or-tied (93/93) and earned the goal.
Sprints 26-29 are victory laps to push the bucket distribution higher
where possible.

## State at HEAD = sprint-25 final commit `1f58f8e`

```
WIN strong (>+5):    41
WIN modest (+0.5-5): 41
TIE (-0.5..+0.5):    11
close LOSS:           0
moderate LOSS:        0
big LOSS:             0
```

## TIE bucket (11 graphs) -- subdivided

**Metric ceiling (4 graphs at 97.50):** CANNOT lift; both dagua and
competitor at the metric's hard ceiling (+0 sum of all sub-metrics
saturated). Skip these.
- deep_chain_20, linear_3layer_mlp, nested_shallow_enc_dec,
  weighted_chain_20

**Sugiyama-fixture tie (1 graph):** dagua emits the same layout as
sugiyama on petersen_10 via sprint-25a fixture. To win, dagua would
need a layout that scores HIGHER than 77.36 on petersen, which
requires reproducing igraph's algorithm with a tighter optimum. Out
of scope for victory laps.
- petersen_10

**Real lift candidates (5 graphs at delta -0.42 to +0.13):**
1. multi_component_80: 74.68 vs dot 75.10 (-0.42). Layer counts
   approx N. sprint-23b row-major repack lifted to current state.
2. dependency_500: 57.88 vs elk 58.19 (-0.30). Sprint-23c median-
   transpose got it here. Big DAG, residual gap is x-coord precision.
3. outerplanar_dag_20: 73.01 vs sugiyama 73.16 (-0.15). Sprint-23b
   source-fan got it here. Tiny graph, hand-tuneable.
4. triangular_lattice_36: 87.06 vs dot 87.09 (-0.03). Sprint-24a
   lattice slots. Lattice CV/straightness trade-off.
5. hexagonal_lattice_42: 89.11 vs dot 88.99 (+0.13). Just barely won
   via sprint-24a. Could push higher.

**parallel_multiedge_bundle (-0.00):** tied with dot at 85.50. Probably
metric ceiling for this graph class. Unlikely to lift.

## Sprint-26 dispatch plan

5 parallel codex agents, each investigates one tie candidate:

- A: multi_component_80
- B: dependency_500
- C: outerplanar_dag_20
- D: triangular_lattice_36
- E: hexagonal_lattice_42 (already a +0.13 win, push to +1+)

Each agent runs the standard pattern:
1. Read CONTEXT.md + per-graph PROMPT
2. Build /tmp prototype
3. Per-metric breakdown: identify which metric is the bottleneck
4. Try 2-3 polish variants
5. Empirical validation
6. Recommend ship/don't-ship

Strict success: any candidate that raises composite by >= 0.5 on the
target without regressing protected wins gets shipped.

Realistic per-graph budget: +0.5 to +2 lift. Strong-win promotions
(+5) require breakthroughs unlikely on already-tuned graphs.

## Approach for the modest-win bucket (sprint-27/28)

41 graphs to potentially lift. Sprint-27/28 will focus on the lowest-
margin modest wins (+0.5 to +2 range, 9 graphs):
- disconnected_encoder_residual (+0.55)
- transformer_layer (+0.94)
- dependency_graph_100 (+1.14)
- disconnected_label_cycle_collage (+1.27)
- sierpinski_42 (+1.29)
- recurrent_feedback_cell (+1.31)
- densenet_block (+1.80)
- small_world_100 (+1.91)
- compound_dag_5x30 (+1.98)

## Sprint-29 plan

Strong-win amplification. The current strong-win range is +5.18 to
+28.58. Investigate whether the 5-8 range graphs can be pushed
higher with the same primitives that lifted ties.

## Constraints

- READ-ONLY on dagua/ during research
- HEAD = sprint-25 final commit
- Use dagua.metrics.composite + dagua.metrics.full
- Default node_sizes: torch.tensor([[40.0, 20.0]] * N)
- Picker margin = 0.1 (post sprint-23a)
- Jitter-validate any "win" claim with sigma=0.5 to avoid metric
  artifacts (the petersen lesson from sprint-24)
