# Sprint 23 Area B: Lattice-aware grid-snap for hex/tri/square

## Mandate

hexagonal_lattice_42 (-0.63 vs graphviz_dot 88.99) and
triangular_lattice_36 (post sprint-22c estimated -0.48) are close-losses
where graphviz_dot wins by edge_length_cv (~0.10) via integer-grid x
positions. Sprint-22c dot_lattice_lp uses HiGHS LP with default
tolerances and minimizes total edge length subject to layer-rank
constraints; this gets close but not flush against dot.

The structural difference: dot's network-simplex x-step uses
INTEGER-GRID positions via branch-and-bound, plus per-layer "tight
constraints" that pull adjacent same-layer nodes to integer grid
lines. Pure LP relaxation gives fractional positions that score
slightly worse on edge_length_cv.

## Research questions

1. Reverse-engineer graphviz_dot's exact lattice positions by loading
   `eval_output/benchmark_full/positions/hexagonal_lattice_42__graphviz_dot.pt`.
   Verify the integer-grid hypothesis: count unique x and y values,
   compare CV to dagua's current output and to sprint-22c LP output.

2. Implement a lattice-snap step that takes the LP output and snaps
   x-positions to integer grid lines (computed from per-layer node
   counts and the LP's continuous output). Two variants:
   - Variant 1: round to nearest integer-times-pitch
   - Variant 2: post-LP integer projection via Hungarian matching
     (assign each LP-x-position to the closest integer-grid slot
     given per-layer slot counts)

3. Empirically measure on:
   - hexagonal_lattice_42 (target)
   - triangular_lattice_36 (target)
   - grid_5x5 (currently +X.XX win for dagua; verify NO regression)
   - sierpinski_42, planar_60, parallel_multiedge_bundle (sanity
     checks for protected wins)

4. Pose as an additional polish candidate stacked on sprint-22c's
   dot_lattice_lp. Picker margin gate handles regression risk.

## Output spec

File: `.project-context/research/sprint_23_finish_line/B_lattice_grid_snap__<agent>.md`

Sections:
- TL;DR (single biggest call: ship/don't ship, where it wins, where
  it loses)
- graphviz_dot reverse-engineering: per-graph table of unique-x /
  unique-y / CV / aspect ratio for hex_42, tri_36, grid_5x5
- Algorithm sketch (Python pseudocode for both variants)
- Empirical table of variants vs LP baseline vs gradient baseline
  vs dot, on all target graphs
- Risk: which graphs would the snap REGRESS on, what's the gate
- Recommended implementation: 1 variant or both, gate predicate,
  LOC estimate

## Constraints

- READ-ONLY on dagua/
- HEAD = sprint-22e finalize commit `d27fced`
- Sprint-22c implementation lives at
  `dagua/layout/ops/pipelines/dagua_native.py` `_dot_lattice_lp`
  (lines 1006+) -- READ first to understand the LP structure before
  proposing the snap
- Use scipy.optimize.linear_sum_assignment for Hungarian matching

## Word budget

2000-3500 words.
