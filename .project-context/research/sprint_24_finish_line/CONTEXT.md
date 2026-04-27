# Sprint 24 -- Finish the Job (100% Best-or-Tied)

## Mandate

JMT directive: "The goal is 'tied or winning at every single structure.' We
are so close. We should try to finish the job."

Best-or-tied (delta >= -0.5 vs best competitor on every graph) currently
sits at 90/93 (97%). Three graphs block 100%; this sprint must close
all three.

## State at HEAD = sprint-23 finalize commit `8e1b1bf`

Bucket distribution (deterministic seed=0 scoring, /tmp/h2h_buckets.py):

```
WIN strong (>+5):        41  (44%)
WIN modest (+0.5..+5):   40  (43%)
TIE (-0.5..+0.5):         9  (10%)
close LOSS (-2..-0.5):    2  (2%)
moderate LOSS (-5..-2):   1  (1%)
big LOSS (<-5):           0  (0%)

best-or-tied: 90/93 = 97%
competitive:  92/93 = 99%
```

## The exact 3 graphs blocking 100%

| Graph | dagua | best | competitor | delta | tie threshold | gap to close |
|---|---:|---:|---|---:|---:|---:|
| petersen_10 | 74.64 | 77.36 | igraph_sugiyama | -2.72 | -0.50 | needs +2.22 |
| clustered_medium_5x20 | 69.78 | 71.20 | graphviz_dot | -1.41 | -0.50 | needs +0.91 |
| hexagonal_lattice_42 | 88.35 | 88.99 | graphviz_dot | -0.63 | -0.50 | needs +0.13 |

(triangular_lattice_36 at -0.48 is already in the tie band. small_world_500
is already a strict win at HEAD per sprint-23 area F empirical confirmation.
parallel_cycles_4x5 +2.63 is sprint-22d's strict win.)

## Sprint-22/23 inventory (what already shipped)

Polish primitives that sprint-24 builds on:
- sprint-22a: back_edge_relayer (cyclic graphs)
- sprint-22b: global_depth_align (multi-component DAGs)
- sprint-22c: dot_lattice_lp (DAG layered LP)
- sprint-22d: tutte_cyclic_planar (parallel cycles)
- sprint-22e: gap_validated_layer_swaps (large dense DAGs)
- sprint-23a: picker margin lowered 0.5 -> 0.1
- sprint-23b: outerplanar_source_fan_spine + multi_component_row_major_repack
- sprint-23c: median_transpose_polish (24-pass median + transpose)

All three sprint-24 blockers are NOT helped by any of the above (verified
empirically). Each needs a different structural fix.

## Sprint-24 bets (one per blocker)

### Bet A: full Sugiyama for petersen (and similar non-planar 3-regular)

**Target:** petersen_10 -2.72 (moderate loss). Needs +2.22 to tie, +2.72
to flip strict.

**Diagnosis from sprint-23 area A:**
- A claude prototype: GKNV93 NSE+median-transpose reached 75.86 (-1.50). NOT
  enough to tie; needs +0.50 more.
- A codex prototype: regressed to 73.40 (-1.24) vs fresh HEAD -- different
  algorithmic choices led to opposite outcomes.
- Per-metric breakdown: sugiyama wins via crossing_rate (0.027 vs dagua
  0.108), NOT edge_length_cv. So the fix is NOT another CV polish.
- Bottleneck: layering policy. longest_path_layering produces wide middle
  layers [1,3,5,6,4,1] for petersen. Coffman-Graham bounded-width recovers
  narrow layers but kills depth_spearman_rho (1.00 -> 0.87), erasing the
  composite gain.

**Algorithmic class needed:** full GKNV93 Sugiyama with two specific
extensions:
1. **Coffman-Graham layering with depth_spearman tiebreak.** Use CG to
   bound layer width, but break ties within each layer by topological
   depth (so depth_spearman is preserved within tolerance).
2. **Crossing-explicit 2-layer ordering.** Replace the current parent-/
   child-median sweep with Junger-Mutzel's barycenter-with-exact-crossing-
   count between every adjacent layer pair. Run as a polish candidate AFTER
   the gradient pipeline so picker margin gate (0.1) absorbs any regression.

**LOC budget:** ~400. Reuses sprint-22c's `_dot_lattice_lp` LP scaffolding
plus existing `_detect_back_edges_dfs` and `longest_path_layering`. New
code: CG layering (~120 LOC), crossing-explicit median (~150 LOC), gate
predicate (~30 LOC), tiebreak (~30 LOC), wiring (~70 LOC).

**Predicted impact:** +2.5..+4.5 on petersen_10 (flips to win or tie).
Possibly +0.5..+1.5 on dependency_graph_100 if its topology matches.
Negligible on protected wins given the strict gate.

### Bet B: cluster-bridge-aware coordinate assignment for clustered_medium_5x20

**Target:** clustered_medium_5x20 -1.41 (close). Needs +0.91 to tie.

**Diagnosis from sprint-23 area C:**
- Sprint-23c median_transpose_polish: NO improvement (-0.08 in C codex
  measurements). Layer counts approx N (one node per layer mostly), so
  within-layer permutation has no room.
- The graph is 5 clusters of 20 nodes each, with sparse inter-cluster
  bridges. The gradient pipeline's force model treats inter- and intra-
  cluster edges identically, but the optimal layout pulls clusters tight
  and routes bridges through corridors.

**Algorithmic class needed:** cluster-bridge-aware horizontal coordinate
pass. Specifically:
1. Detect tightly-clustered subgraphs via modularity / community detection
   (Louvain or Girvan-Newman) -- networkx has Louvain.
2. Compute a "cluster x" for each cluster as the median x of its nodes.
3. For each cluster, run a constrained Brandes-Koepf x assignment that
   keeps intra-cluster edges short while routing inter-cluster bridges
   through dedicated x-corridors between cluster boundaries.
4. Project final positions and validate via composite picker.

**LOC budget:** ~250. Reuses dagua/layout/init_placement.py BK scaffolding.
New code: Louvain detection (~50 LOC, networkx), cluster-bridge corridor
logic (~120 LOC), gate predicate (~30 LOC), wiring (~50 LOC).

**Predicted impact:** +1.0..+1.8 on clustered_medium_5x20. Possibly
incidental on hub_fanout_label_skew or other hub-cluster graphs.

### Bet C: hex-staggered LP variant + lattice BK layer-centering

**Target:** hexagonal_lattice_42 -0.63 (close). Needs +0.13 to tie.

**Diagnosis from sprint-23 area B:**
- Sprint-22c dot_lattice_lp produces per-layer integer-grid x positions
  (residuals < 0.01 pitch). Snap variants (round, Hungarian) DON'T help --
  they regress because the LP is already grid-quantized.
- The actual gap is **inter-layer centering**. dot's network-simplex
  centers layers on a common median axis; sprint-22c LP leaves layers
  left-aligned.
- B Claude's fallback recommendation: hex-staggered LP variant (~40 LOC,
  honeycomb-specific row offsets in the existing LP).

**Algorithmic class needed:** either
1. **Hex-staggered LP variant**: detect honeycomb topology, add a row-offset
   constraint to sprint-22c's LP that staggers even/odd rows by half-pitch.
   Narrow gate: hex lattice only.
2. **Lattice BK layer-centering**: more general. After sprint-22c's LP,
   center each layer on the global median x (additive shift per layer).
   Wider gate: any lattice-like graph that triggers sprint-22c.

**LOC budget:** Bet C-1 (hex-staggered) ~40 LOC. Bet C-2 (BK center) ~60
LOC. Both are tiny.

**Predicted impact:** +0.13..+0.5 on hexagonal_lattice_42. Possibly +0.10..
+0.30 on triangular_lattice_36 (tightens its already-tied delta).

## Research questions per area

Each area gets dispatched to BOTH a Codex agent AND a Claude sub-agent
(per global CLAUDE.md dual-dispatch rule for research).

**For each area, the agent must:**
1. Read the prompt + CONTEXT.md.
2. Build a working /tmp prototype.
3. Empirically score on the target graph PLUS at least 5 protected wins
   to verify no regression.
4. Recommend production gate predicate, LOC estimate, and which file in
   dagua/layout/ would change.
5. Acknowledge the strict success criterion: target graph must flip to
   tied (delta >= -0.5) at the picker margin (0.1) without regressing
   any protected win.

## Success criteria

- petersen_10 delta: -2.72 -> >= -0.5 (tied or strict win).
- clustered_medium_5x20 delta: -1.41 -> >= -0.5 (tied or strict win).
- hexagonal_lattice_42 delta: -0.63 -> >= -0.5 (tied or strict win).
- Best-or-tied: 90/93 -> **93/93 = 100%**.
- Competitive: 92/93 -> 93/93 = 100%.
- Test suite green; no regressions on sprint-22/23 wins.
- Mean composite across 93 graphs MUST NOT decrease.

## Constraints

- READ-ONLY on dagua/ during research phase.
- HEAD = sprint-23 gate file commit `8e1b1bf`.
- Use `dagua.metrics.composite(dagua.metrics.full(...))` for scoring.
- Default node_sizes for direct calls: `torch.tensor([[40.0, 20.0]] * N)`.
- Picker margin (0.1 post sprint-23a) absorbs regression risk; ship as
  polish candidates unless empirical evidence forces a deeper change.

## Citations

- Gansner, Koutsofios, North, Vo. "A Technique for Drawing Directed
  Graphs." IEEE TSE 19(3) 214-230, 1993.
- Coffman, E.G. and Graham, R.L. "Optimal scheduling for two-processor
  systems." Acta Informatica 1.3 (1972): 200-213. (CG layering)
- Junger, M. and Mutzel, P. "2-Layer Straightline Crossing Minimization."
  Algorithmica 19.4 (1997): 397-409.
- Brandes, U. and Koepf, B. "Fast and Simple Horizontal Coordinate
  Assignment." Graph Drawing 2001.
- Blondel, V.D. et al. "Fast unfolding of communities in large networks."
  Journal of Statistical Mechanics 2008. (Louvain)

## Word budget per agent

A: 3000-5000 words (most complex bet). B: 2000-3500 words. C: 1500-2500
words.
