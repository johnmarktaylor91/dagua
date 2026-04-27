# Sprint 23 Area A: Network-simplex x-coordinate for non-planar 3-regular

## Mandate

Petersen_10 is the SINGLE non-competitive graph in the dagua benchmark
suite (delta -2.72 vs igraph_sugiyama, dagua=74.64, sugiyama=77.36).
Closing this gap takes us from 99% competitive to 100% competitive.

The gap is structural: petersen is non-planar 3-regular (every node has
degree 3, graph cannot be embedded without edge crossings). Sugiyama
wins because it implements integer-grid x-coordinate assignment via
network simplex (Gansner-Koutsofios-North-Vo 1993, IEEE TSE 19(3)
section 4.2 "A Technique for Drawing Directed Graphs"). dagua's
gradient pipeline can't represent the integer-grid optimum in
continuous coordinates.

## Research questions

1. Read the GKNV93 paper (especially section 4.2). Implement the
   network-simplex x-coordinate step in /tmp/sprint23_a/. Use HiGHS
   from scipy as the LP solver but tighten with branch-and-bound
   for integer constraints.

2. Generalize to non-planar by first removing back-edges (FAS), then
   layering the residual DAG via longest-path or Coffman-Graham. Use
   dagua's existing `_detect_back_edges_dfs` for back-edge detection
   and `dagua.utils.longest_path_layering`.

3. Empirically measure on ALL 93 benchmark graphs to find the
   envelope where this candidate wins / loses / ties. Include:
   - petersen_10 (the target)
   - other 3-regular candidates: complete_bipartite_*, peterson-like
     synthetic graphs if any
   - non-planar large: small_world_100, small_world_500
   - protected wins: don't break hex_lattice, deep_chain, etc.

4. Pose as a polish candidate (gated narrowly, picker margin 0.5).
   Estimate LOC budget for production implementation.

## Output spec

File: `.project-context/research/sprint_23_finish_line/A_petersen_3regular__<agent>.md`

Sections:
- TL;DR (5 bullets max, single biggest call)
- Algorithm sketch (Python pseudocode, 50-100 LOC working)
- Empirical validation table (petersen + 5-10 other graphs):
  graph, baseline composite, candidate composite, delta, picker
  decision (would margin gate accept it?)
- Risk / regression analysis: which graphs MIGHT regress, what's
  the minimum gate to keep them safe
- Recommended implementation: gate predicate, pipeline structure,
  LOC estimate, what files in dagua/ would change

## Constraints

- READ-ONLY on dagua/ -- experiments live in /tmp/sprint23_a/
- HEAD = sprint-22e finalize commit `d27fced`
- Use `dagua.metrics.composite(dagua.metrics.full(...))` for scoring
- Default node_sizes for direct calls: torch.tensor([[40.0, 20.0]] * N)
- Use longest_path_layering for layer assignments
- Reference sprint-22 area A research (dot_lattice_lp) at
  `.project-context/research/sprint_22_algo_bets/A_dot_lattice_mimic__codex.md`
  -- the gate logic for layered DAGs is reusable; new code is the
  back-edge / non-planar dummy expansion

## Citations

Always cite GKNV93 IEEE TSE 19(3) when referencing the algorithm. If
you find newer papers (post-2020) on integer-coordinate Sugiyama, cite
those too with arXiv IDs.

## Word budget

2500-4000 words. TL;DR is the load-bearing part.
