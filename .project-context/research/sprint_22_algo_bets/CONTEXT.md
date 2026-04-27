# Sprint 22 — Algorithmic Bets for "Tied or Best on Every Graph"

## Mandate

JMT directive: "Do another sprint, claudes and codexes for research.
Feel free to try the bigger algorithmic bets properly. We're so close
to the dream of 'tied or best at every single graph structure.' Keep
going. Be as exhaustive as possible. Everything remains on the table."

Sprint-21 landed several picker-safe polish primitives (sprint-21a
dd440e5: y_layer_snap, ortho_align, overlap_jitter, swap_2opt_anti_crossing,
per_layer_x_kmeans) plus a F bug fix (sprint-21b c821eb6: tree/chain
component re-classification). Best-or-tied went 88% -> 89%, competitive
97% -> 98%. The remaining gaps are at algorithm ceiling — they need
real algorithm design, not picker variants. This is research only,
implementation comes after.

## State at HEAD = `c821eb6` (sprint-21b)

Bucket distribution (deterministic seed=0 scoring):

```
WIN strong (>+5):        40  (43%)
WIN modest (+0.5..+5):   36  (39%)
TIE (-0.5..+0.5):         7  (8%)
close LOSS (-2..-0.5):    8  (9%)
moderate LOSS (-5..-2):   2  (2%)
big LOSS (<-5):           0  (0%)

best-or-tied: 83/93 = 89%
competitive:  91/93 = 98%
```

**The 11 graphs that aren't dominated:**

Moderate losses (2):

| Graph | dagua | best | comp | delta | known diagnosis |
|---|---|---|---|---|---|
| dependency_500 | 55.28 | elk_layered | 58.19 | -2.90 | edge_length_cv 0.95 vs 0.79; gradient saturated; polish would regress so picker keeps baseline. Large DAG (N=500). |
| petersen_10 | 74.64 | igraph_sugiyama | 77.36 | -2.72 | non-planar 3-regular; sugiyama wins on this class structurally. Sprint-21 B Claude flagged this may be stale — fresh measure showed +3.42 at sprint-20l HEAD. RE-VERIFY at sprint-21b HEAD. |

Close losses (8):

| Graph | dagua | best | comp | delta |
|---|---|---|---|---|
| small_world_500 | 52.19 | elk_layered | 54.15 | -1.96 |
| disconnected_encoder_residual | 84.01 | elk_layered | 85.63 | -1.62 |
| triangular_lattice_36 | 85.48 | graphviz_dot | 87.09 | -1.61 |
| clustered_medium_5x20 | 69.78 | graphviz_dot | 71.20 | -1.41 |
| outerplanar_dag_20 | 72.42 | igraph_sugiyama | 73.16 | -0.74 |
| multi_component_80 | 74.46 | graphviz_dot | 75.10 | -0.64 |
| hexagonal_lattice_42 | 88.35 | graphviz_dot | 88.99 | -0.63 |
| parallel_cycles_4x5 | 62.11 | graphviz_sfdp | 62.73 | -0.62 |

Ties (7):

| Graph | dagua | best | comp | delta | note |
|---|---|---|---|---|---|
| recurrent_feedback_cell | 73.18 | igraph_sugiyama | 73.58 | -0.39 | flippable |
| parallel_multiedge_bundle | 85.50 | graphviz_dot | 85.50 | 0.00 | tied |
| deep_chain_20 | 97.50 | graphviz_dot | 97.50 | 0.00 | metric ceiling |
| linear_3layer_mlp | 97.50 | graphviz_dot | 97.50 | 0.00 | metric ceiling |
| nested_shallow_enc_dec | 97.50 | igraph_sugiyama | 97.50 | 0.00 | metric ceiling |
| weighted_chain_20 | 97.50 | graphviz_dot | 97.50 | 0.00 | metric ceiling |
| small_world_100 | 57.18 | igraph_sugiyama | 57.09 | +0.09 | already beat by 0.09 |

## What's been ruled out across sprints 19, 20, 21

- All gradient weight tuning is saturated (w_dag, w_attract, w_repel,
  w_length_variance, w_straightness — every value gives identical
  layouts on the moderate-loss bucket).
- multi_start_k 1..20 produces identical output on saturated targets.
- Lattice aspect target 0.05..1.0: sprint-19e's 0.05 is empirically
  optimal for hex_lattice; raising regresses by 5+ points.
- All sprint-21a polish primitives (y_layer_snap, ortho_align,
  overlap_jitter, swap_2opt_anti_crossing, per_layer_x_kmeans):
  picker chose them where they helped. Where they didn't help, they
  were rejected.
- Simple band-permute depth-rank tiling (sprint-21c attempted): no-op
  because depth_spearman is node-level.
- BFS-layer + within-layer x-rank lattice synth (sprint-21c attempted):
  produces uniform grid that loses on straightness/dag_consistency
  more than it gains on edge_length_cv. Picker rejects.
- Force_directed and planar pipelines auto-routing: empirically lose
  almost everywhere; only useful as explicit user override.

## Bigger algorithmic bets for this sprint

These are the ones I bailed on in sprint-21 due to time / care budget.
Sprint-22 mandate: do them properly.

### Bet 1: Match graphviz_dot's actual lattice positions (not uniform grid)

`hexagonal_lattice_42`: dot positions have **18 unique x-values and
12 unique y-values** but edges are NOT uniform length (CV = 0.10, not
0). dot uses a non-trivial layer-x heuristic that's tighter than my
uniform-pitch synth. Read dot's source or reverse-engineer from its
positions.

Reference dot positions for hex_lattice_42:
```
x range: 27..459 (pitch varies per row)
y range: -810..-18  (12 layers, pitch ~72)
first 8 nodes: (117,-810), (153,-738), (261,-666), (279,-594),
               (369,-522), (387,-450), (459,-378), (81,-738)
```

Targets: hexagonal_lattice_42 (-0.63), triangular_lattice_36 (-1.61),
parallel_cycles_4x5 (-0.62), and possibly close-loss tail.

### Bet 2: Conformal / harmonic embedding for true planar lattices

D Claude flagged arXiv:2506.20541 (June 2025) "conformal-rigidity-
guided Tutte/harmonic refinement" as the genuinely new post-2024
result with applicability to dagua's metric. Predicted +5..+7
composite on hex/tri lattices. ~250 LOC.

Targets: same as Bet 1 if Bet 1 doesn't work.

### Bet 3: Node-level global-depth y-alignment for multi-component DAGs

`disconnected_encoder_residual`: depth_spearman 0.644 vs elk's 1.000
is the entire -1.62 gap. Components have overlapping depth ranges, so
band-permute (sprint-21c attempt) didn't help — needs node-level
re-y-coordinate fix that puts depth-0 nodes on the same row globally,
depth-1 nodes on the next row, etc., across all components.

Targets: disconnected_encoder_residual (-1.62), multi_component_80
(-0.64).

### Bet 4: Large-DAG dependency_500 escape

dependency_500 N=500, E=1471, family=GENERAL, max_degree=53. Gradient
saturated. D codex proposed gap-constrained layered local search:
identify long edges, run adjacent swaps that lower
`crossings + alpha*gaps + beta*edge_span_cv`. 120-200 LOC. E codex
proposed `aspect_preserving_equalize` (locks bbox during projection)
+0.5..+1.5 on dependency_500.

Targets: dependency_500 (-2.90).

### Bet 5: small_world_500 spectral/structural finishing

small_world_500 -1.96. dagua wins edge_length_cv (+15.45) but loses
dag_consistency (-12.35) and straightness (-6.12) to elk. Sugiyama
wins because it imposes hierarchy via FAS even on cyclic graphs.
Stress route already helps small_world_100 (+0.09). 500-node case
needs different handling.

Targets: small_world_500 (-1.96).

### Bet 6: Petersen verification

B Claude said petersen +3.42 at sprint-20l HEAD (CONTEXT.md was
stale). Verify at sprint-21b HEAD before any work. If it's no longer
a loss, this whole bet is moot.

## Output contract

Each agent writes a markdown findings file at:
`.project-context/research/sprint_22_algo_bets/<area>_<agent>.md`

Each report MUST include:
1. TL;DR (4-6 bullets) — single biggest call.
2. Algorithm sketch with **complete, working pseudocode** (not handwave).
3. Empirical validation: hand-implement in /tmp/, test on real graphs,
   quote actual measured composite delta numbers.
4. Risk / regression analysis with specific protected wins to verify.
5. Implementation order.

This time: BIGGER BETS done PROPERLY. No 30-minute hacks. If a bet
needs 200-500 LOC and 2 hours of careful design, that's what it gets.

DO NOT write any code in dagua/ or commit. Findings markdown files
only. /tmp/ scripts allowed for measurement.
