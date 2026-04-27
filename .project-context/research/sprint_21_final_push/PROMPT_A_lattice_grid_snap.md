# Area A — Lattice grid-snap algorithm

## Question

Design a placement algorithm that closes the lattice gap for
`hexagonal_lattice_42` (-2.52), `triangular_lattice_36` (-2.20), and
`sierpinski_42` (already in TIE-or-better but improvable).

graphviz_dot wins these by deterministic grid placement: nodes snapped
to integer grid points, edges all unit-length. dagua's gradient
pipeline + edge-equalize polish reaches edge_length_cv ≈ 0.43 on
hex_lattice; dot achieves 0.10. The 20-pt-weight metric component
gives dot ~6 free composite points purely from grid-uniformity.

## Specific evidence

`hexagonal_lattice_42` post-sprint-20l (dagua=86.46, dot=88.99,
delta=-2.52):

| metric | weight | dagua | dot | delta |
|---|---|---|---|---|
| dag_consistency | 25 | 1.00 | 1.00 | 0 |
| edge_length_cv | 20 | 0.43 | 0.10 | -6.6 (DAGUA LOSES) |
| depth_spearman | 15 | 0.998 | 0.823 | +2.6 |
| edge_straightness | 10 | 1.000 | 0.45 | +5.5 |
| crossing_rate | 10 | 0 | 0 | 0 |
| overlap | 10 | 0 | 0 | 0 |

dagua trades straightness for length variance. The grid snap would
flip this: nodes snap to a regular hex grid, edge_length becomes
exactly the grid step (CV=0), straightness goes from 1.0 to whatever
the hex geometry produces (~0.5).

dot's positions on hex_lattice_42:
```
graphviz_dot:
  x range: 27 to 459 = 432 units wide
  y range: -810 to -18 = 792 units tall
  18 unique x values, 12 unique y values  ← grid-snapped
  e.g. (117, -810), (153, -738), (261, -666), (279, -594), ...
```

## Research targets

1. **Detect when a graph is "lattice-like enough" to benefit from
   grid-snap**: classifier already exposes `lattice_like` tag (used
   for aspect_target). Is the right gate "is_planar AND
   degree_distribution_uniform AND edge_count_within_planar_bound"?

2. **Algorithm**: given polished positions, find a grid orientation
   + cell size that minimizes total node displacement when each node
   is snapped to its nearest grid intersection. Lloyd-relaxation-on-
   grid or a one-shot orientation fit then snap.

3. **Non-regression**: the snap must preserve edge_straightness and
   crossing_rate gains. Hex lattice's straight edges shouldn't curl
   when snapped.

4. **Generality**: hex grid for hex_lattice, triangular grid for
   triangular_lattice, more general "minimum-displacement grid fit"
   for irregular lattice-like inputs (sierpinski_42 fractal).

## Output format

Write your findings to:
`.project-context/research/sprint_21_final_push/A_lattice_grid_snap__<your_agent_name>.md`

Include:
- TL;DR (4-6 bullets) — single biggest call
- Algorithm sketch with pseudocode (post-pipeline projection step)
- Expected composite delta per target graph (quantified)
- Risk: which sprint-19 protected wins or sprint-20k polish wins are
  at risk?
- Recommended gate (when to fire the snap vs leave the polish-only path)
- Open questions if any

## Constraints

- READ-ONLY. Do NOT write code or commit. Findings file only.
- Read `.project-context/research/sprint_21_final_push/CONTEXT.md` first.
- Budget your reasoning: aim for 1500-2500 words of findings, not
  10000.
