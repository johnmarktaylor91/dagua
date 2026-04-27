# Sprint 24 Area B: Cluster-bridge-aware coordinate assignment for clustered_medium_5x20

## Mandate

clustered_medium_5x20 is one of two close-losses blocking dagua's 100%
best-or-tied milestone (delta -1.41 vs graphviz_dot 71.20, dagua=69.78).
Needs +0.91 to flip to tied.

The graph is structured as 5 tight clusters of 20 nodes each with
sparse inter-cluster bridges. Sprint-23 area C empirically confirmed
the median-transpose polish does NOTHING on this graph (-0.08
regression in C codex measurements) because layer counts approx N --
within-layer permutation has no room.

## Diagnosis

The gradient pipeline's force model treats inter- and intra-cluster
edges identically. The optimal layout pulls clusters tight and routes
bridges through corridors -- this is what graphviz_dot does
implicitly via its hierarchical record-based layout. dagua needs an
explicit cluster-aware coordinate assignment pass.

## Research questions

1. Build a working /tmp prototype implementing cluster-bridge-aware
   horizontal coordinate assignment in /tmp/sprint24_b_<agent>/:

   a. **Cluster detection**: use networkx Louvain community detection
      (or Girvan-Newman if Louvain is unstable) on the undirected
      graph. Verify the detected communities match the 5 expected
      clusters of clustered_medium_5x20.

   b. **Bridge edge identification**: an edge is a "bridge" if its
      endpoints are in different clusters. Count bridges per cluster
      pair.

   c. **Cluster-tight x assignment**: for each cluster, compute a
      "cluster x center" as the median x of its nodes. Run a
      constrained Brandes-Koepf x assignment that keeps intra-cluster
      edges short (small horizontal extent within each cluster).

   d. **Bridge corridor routing**: between cluster x centers, allocate
      a horizontal corridor proportional to the bridge count. Bridge
      endpoints should land at the cluster boundary, not the cluster
      center.

   e. **Composite picker validation**: score the candidate; accept
      only if composite improves by >= 0.1 (sprint-23a margin) and
      overlap count doesn't increase.

2. Empirically measure on:
   - clustered_medium_5x20 (primary target)
   - hub_fanout_label_skew (potential incidental win or regression)
   - random_dag_200 (protected, multi-cluster-ish)
   - org_chart_deep (protected, hierarchical)
   - dependency_500 (protected, dense DAG -- must NOT regress
     sprint-23c's +1.61 lift)
   - small_world_100, small_world_500 (cyclic, gate must reject)
   - hex_lattice_42, tri_lattice_36 (lattice, gate must reject)

3. Per-metric breakdown for clustered_medium_5x20: dag_consistency,
   edge_length_cv, depth_spearman_rho, overlap_count, edge_straightness,
   crossing_rate, cluster_separation. The win likely comes from
   cluster_separation (5 pts in composite) and possibly
   edge_straightness.

## Output spec

File:
`.project-context/research/sprint_24_finish_line/B_cluster_bridge__<agent>.md`

Sections:
- **TL;DR (5 bullets)** -- ship/don't ship, measured delta, gate.
- **Cluster detection diagnosis** -- what Louvain returns on the
  target graph; does it find 5 clusters as expected? Per-cluster size,
  inter-cluster bridge count.
- **Algorithm sketch** (Python pseudocode, ~150-250 LOC).
- **Empirical validation table** -- per-graph composite +
  per-metric breakdown for clustered_medium_5x20 + protected wins.
- **Risk / regression analysis** -- which graphs MIGHT regress and
  the gate to keep them safe.
- **Recommended implementation** -- gate predicate, where it slots in
  dagua/layout/ (likely a new private helper in dagua_native.py near
  _global_depth_align or a new candidate slot in _best_of_polish),
  LOC estimate.

## Strict success criterion

clustered_medium_5x20 composite >= 70.70 (delta >= -0.5, the tie
threshold). 70.30 (post-sprint-23 baseline) is NOT enough. We need
+0.92 over baseline.

If the candidate cannot reach 70.70, identify what's blocking it.
The honest "this approach doesn't close the gap" answer beats a
shipped-but-insufficient candidate.

## Constraints

- READ-ONLY on dagua/. Experiments in /tmp/sprint24_b_<agent>/.
- HEAD = sprint-23 gate file commit `8e1b1bf`.
- Use networkx for community detection (Louvain via
  `networkx.algorithms.community.louvain_communities`).
- Reference dagua/layout/init_placement.py for existing BK
  implementation that you'll constrain.
- Reference sprint-23 C research at
  `.project-context/research/sprint_23_finish_line/C_dense_dag_ordering__codex.md`
  for why median-transpose doesn't help on this graph.

## Citations

- Brandes, U. and Koepf, B. "Fast and Simple Horizontal Coordinate
  Assignment." Graph Drawing 2001.
- Blondel, V.D. et al. "Fast unfolding of communities in large
  networks." Journal of Statistical Mechanics 2008.
- Sander, G. "Graph Layout for Applications in Compiler Construction."
  Theoretical Computer Science 217.2 (1999): 175-214. (Hierarchical
  cluster-aware layered drawing)

## Word budget

2000-3500 words.
