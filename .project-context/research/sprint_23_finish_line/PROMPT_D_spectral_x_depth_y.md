# Sprint 23 Area D: Spectral-x + depth-y for non-planar lattices

## Mandate

small_world_500 (-1.96, post sprint-22a back_edge_relayer) is the
single largest close-loss. The graph is non-planar with cycles; the
gradient pipeline's stress route handles it OK but never quite reaches
elk_layered's 54.15.

A different approach: use the second-smallest eigenvector of the
graph Laplacian for x positions (the "Fiedler vector" -- minimizes
total squared edge length on connected graphs), then warp y to match
longest_path_layering for dag_consistency=1 and depth_spearman=1.
This is a 1D Tutte/spectral embedding combined with depth anchoring.

B Codex's sprint-22 sketch was partially measured but not implemented
beyond the dot_lattice_lp variant. Sprint-23 Area D probes whether
this is the right finisher for small_world_500 specifically.

## Research questions

1. Implement spectral_x_depth_y in /tmp/sprint23_d/:
   - Build symmetric Laplacian L = D - A (undirected adjacency)
   - Compute eigsh(L, k=2, sigma=0) -- the Fiedler vector x_fiedler
   - Layer assignment via longest_path_layering, set y = layer * pitch
   - Place x = x_fiedler, normalized to span pitch * sqrt(N)
   - Optional: compose with sprint-22a back_edge_relayer for
     graphs with strong cycle structure

2. Empirically measure on small_world_500, small_world_100,
   recurrent_feedback_cell, parallel_cycles_4x5, hex_lattice_42,
   tri_lattice_36, planar_60, dependency_500, deep_chain_20.
   Per-metric breakdown matters here -- spectral wins on edge_length_cv
   but might lose on edge_straightness.

3. Decide whether to ship: gate predicates, picker margin tolerance,
   what protected wins might regress.

## Output spec

File: `.project-context/research/sprint_23_finish_line/D_spectral_x_depth_y__<agent>.md`

Sections:
- TL;DR (5 bullets)
- Algorithm sketch (Python pseudocode, ~80 LOC)
- Empirical validation: per-graph composite + per-metric breakdown
  (CV, straightness, dag, rho, crossings) for ~10 graphs
- Picker decision: ship narrow / ship broad / don't ship
- Implementation: gate predicate, LOC estimate

## Constraints

- READ-ONLY on dagua/
- HEAD = sprint-22e finalize commit `d27fced`
- scipy.sparse.linalg.eigsh is available
- Reference sprint-22 area B research at
  `.project-context/research/sprint_22_algo_bets/B_conformal_embedding__codex.md`

## Word budget

1500-2500 words.
