# Sprint 24 Area A: Full Sugiyama for petersen_10

## Mandate

petersen_10 is the SINGLE moderate-loss in the dagua benchmark suite at
delta -2.72 vs igraph_sugiyama. This sprint MUST flip it to tied or
strict win. Best-or-tied at 100% requires it.

Petersen is non-planar 3-regular. Sprint-23 area A research dual-
dispatched on this exact problem and found: GKNV93 NSE alone is
insufficient. A claude reached 75.86 (-1.50, still close-loss). A
codex regressed to 73.40 (-1.24). Neither flipped the graph.

The diagnosis was definitive (sprint-23 A claude empirical
breakdown):

- sugiyama wins via crossing_rate (0.027 vs dagua's 0.108), NOT
  edge_length_cv.
- longest_path_layering produces wide middle layers [1, 3, 5, 6, 4, 1]
  for petersen. Coffman-Graham bounded-width recovers narrower layers
  [2, 3, 3, 2] but kills depth_spearman_rho (1.00 -> 0.87). The
  composite gain from CV improvement is erased.

## Research questions

1. Implement full Sugiyama with TWO specific extensions in
   /tmp/sprint24_a/ scratch:
   a. **Coffman-Graham layering with depth_spearman tiebreak**: use CG
      for layer width bounds, but break ties within each layer by
      topological depth from longest_path_layering, so depth_spearman
      stays >= 0.95.
   b. **Crossing-explicit 2-layer ordering**: replace median sweep with
      Junger-Mutzel barycenter that explicitly counts and minimizes
      crossings between every adjacent layer pair (not just
      parent-/child-median). Run as polish candidate AFTER the gradient
      pipeline so picker margin absorbs regression risk.

2. Empirically score on petersen_10 PLUS:
   - Other 3-regular: complete_bipartite_8x12, regular_3_30 (if in
     suite), heawood_14, mcgee_24, moebius_kantor_16 (synthesize via
     networkx if not in suite -- code them and add to /tmp scoring)
   - Protected wins: random_dag_200, org_chart_deep, deep_chain_20,
     hub_fanout_label_skew, linear_3layer_mlp, hexagonal_lattice_42,
     dependency_500
   - Sanity: small_world_500, parallel_cycles_4x5

3. Per-metric breakdown REQUIRED for petersen_10: dag_consistency,
   edge_length_cv, depth_spearman_rho, overlap_count,
   edge_straightness, crossing_rate, angular_resolution. Show that the
   crossing_rate term improves AND depth_spearman doesn't collapse.

4. Verify the picker margin gate (0.1) accepts the candidate on
   petersen_10 and rejects it on every protected win.

## Output spec

File:
`.project-context/research/sprint_24_finish_line/A_full_sugiyama__<agent>.md`

Sections:
- **TL;DR (5 bullets)** -- single biggest call: ship/don't ship,
  measured petersen_10 delta, what protected wins regressed if any.
- **Algorithm sketch (Python pseudocode, 200-400 LOC working)** --
  CG layering + crossing-explicit ordering + NSE x-coord + projection
  back onto baseline x-slot multiset.
- **Empirical validation table** -- per-graph composite + per-metric
  breakdown for petersen_10 + at least 8 other graphs.
- **Risk / regression analysis** -- which protected wins MIGHT regress
  and the gate predicate that keeps them safe.
- **Recommended implementation** -- gate predicate, pipeline
  structure, LOC estimate, which file in dagua/layout/ would change
  (likely a new private helper in dagua_native.py near
  _dot_lattice_lp).

## Strict success criterion

The candidate MUST achieve petersen_10 composite >= 76.86 (delta
>= -0.5, the tie threshold) at the picker margin (0.1). 75.86 is NOT
enough -- that's what sprint-23 A claude reached and it's still
-1.50. We need at least +1.00 more.

If empirical evidence shows the candidate cannot reach 76.86, say so
directly and identify what additional algorithmic component would
close the residual gap. Honesty here is more valuable than a
shipped-but-insufficient candidate.

## Constraints

- READ-ONLY on dagua/. Experiments in /tmp/sprint24_a_<agent>/.
- HEAD = sprint-23 gate file commit `8e1b1bf`.
- Use dagua.metrics.composite + dagua.metrics.full with default
  node_sizes [40, 20] * N.
- Reference sprint-23 area A research at
  `.project-context/research/sprint_23_finish_line/A_petersen_3regular__claude.md`
  and `__codex.md`. The diagnosis there is definitive; sprint-24 builds
  on it.
- Reference sprint-22c `_dot_lattice_lp` at
  `dagua/layout/ops/pipelines/dagua_native.py` line ~1006 -- the
  rank-LP scaffolding is reusable for the network-simplex x-coord step.

## Citations

Always cite GKNV93 IEEE TSE 19(3) section 4.2 for NSE. Cite
Junger-Mutzel Algorithmica 19.4 (1997) for the 2-layer crossing
analysis. Cite Coffman-Graham Acta Informatica 1.3 (1972) for the
layering bound. If post-2020 papers extend these, cite them too.

## Word budget

3000-5000 words. The TL;DR is the load-bearing part; if the candidate
flips petersen, the full empirical envelope is the proof.
