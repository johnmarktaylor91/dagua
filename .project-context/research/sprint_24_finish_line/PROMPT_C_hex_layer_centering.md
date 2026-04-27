# Sprint 24 Area C: Lattice layer-centering for hexagonal_lattice_42

## Mandate

hexagonal_lattice_42 is the second close-loss blocking 100%
best-or-tied (delta -0.63 vs graphviz_dot 88.99, dagua=88.35). The
smallest gap of the three sprint-24 blockers -- needs only +0.13 to
flip to tied.

Sprint-23 area B research reverse-engineered graphviz_dot's lattice
positions and concluded:
- Sprint-22c `_dot_lattice_lp` produces per-layer integer-grid x
  positions with residuals < 0.01 pitch -- already grid-quantized.
- The actual gap to dot is **inter-layer centering**: dot's
  network-simplex centers each layer on a common median axis;
  sprint-22c LP leaves layers left-aligned.
- Snap variants (round, Hungarian) DON'T help -- they regress.

## Research questions

1. Build prototype in /tmp/sprint24_c_<agent>/ implementing TWO
   variants:

   **Variant 1: hex-staggered LP (B Claude's fallback recommendation)**
   - Detect honeycomb topology (hex face structure).
   - Add row-offset constraint to sprint-22c's LP that staggers
     even/odd rows by half-pitch -- mimics dot's hex output.
   - Narrow gate: hex lattice topology only.

   **Variant 2: lattice BK layer-center (more general)**
   - After sprint-22c LP solves x positions, compute the global
     median x.
   - Apply per-layer additive shift so each layer's median x equals
     the global median. This is the inter-layer centering that B
     research identified.
   - Wider gate: any planar lattice that triggers sprint-22c.

2. Empirically score BOTH variants on:
   - hexagonal_lattice_42 (primary target)
   - triangular_lattice_36 (currently -0.48 tied, must not regress)
   - grid_5x5 (current sprint-22c rejected, must not regress)
   - sierpinski_42 (planar but not lattice, gate must reject)
   - planar_60 (3-connected planar, must not regress)
   - parallel_multiedge_bundle (planar, must not regress)
   - dependency_500 (DAG, gate must reject -- it triggers sprint-22c
     LP today via gap_validated_layer_swaps)

3. Per-metric breakdown for hexagonal_lattice_42:
   edge_length_cv, depth_spearman_rho, edge_straightness,
   crossing_rate. The gap to dot is most likely on edge_length_cv;
   verify.

4. Decision: ship Variant 1 (narrow), Variant 2 (broader), or both
   (composed picker).

## Output spec

File:
`.project-context/research/sprint_24_finish_line/C_hex_layer_centering__<agent>.md`

Sections:
- **TL;DR** -- ship which variant, measured deltas, gate.
- **Algorithm sketch for both variants** (Python pseudocode, ~30-60 LOC each).
- **Empirical table** -- both variants vs sprint-22c LP baseline vs dot
  on the target graphs, per-metric breakdown for hex_42.
- **Recommended implementation** -- LOC estimate, where it slots in
  dagua_native.py.

## Strict success criterion

hexagonal_lattice_42 composite >= 88.49 (delta >= -0.5, the tie
threshold). 88.35 baseline + 0.14 lift is the bare minimum.

This is the smallest gap of the sprint-24 blockers -- if it can't be
closed with a 40-60 LOC variant, that's a strong signal we're at a
metric-definition floor and the gap will need either a different
metric or a different baseline.

## Constraints

- READ-ONLY on dagua/. Experiments in /tmp/sprint24_c_<agent>/.
- HEAD = sprint-23 gate file commit `8e1b1bf`.
- Reference sprint-22c `_dot_lattice_lp` at
  `dagua/layout/ops/pipelines/dagua_native.py` line ~1006.
- Reference sprint-23 area B research at
  `.project-context/research/sprint_23_finish_line/B_lattice_grid_snap__codex.md`
  and `__claude.md`. Both confirmed snap variants don't help; this
  prompt is the next-step layer-centering bet.
- Use saved competitor positions:
  `eval_output/benchmark_full/positions/hexagonal_lattice_42__graphviz_dot.pt`.

## Citations

- Gansner, Koutsofios, North, Vo. "A Technique for Drawing Directed
  Graphs." IEEE TSE 19(3) 214-230, 1993. Section 4.2 for NSE x-coord;
  the layer-centering behavior we're trying to match is implicit in
  dot's code, not the paper -- expect to read `lib/dotgen/position.c`
  in the graphviz source on GitHub if you need ground truth.
- Brandes, U. and Koepf, B. "Fast and Simple Horizontal Coordinate
  Assignment." Graph Drawing 2001.

## Word budget

1500-2500 words.
