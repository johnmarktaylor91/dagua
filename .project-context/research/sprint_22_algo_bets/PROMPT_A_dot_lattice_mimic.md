# Area A — Reverse-engineer graphviz_dot's lattice algorithm

## Question

graphviz_dot wins on `hexagonal_lattice_42` (88.99 vs 88.35), `triangular_lattice_36` (87.09 vs 85.48), and `parallel_cycles_4x5` (62.73 sfdp vs 62.11). Sprint-21c's BFS-layer + within-layer x-rank lattice synth produced uniform pitch grids that LOST to baseline because they sacrificed straightness/dag_consistency to gain edge_length_cv.

dot's secret: positions are NOT a uniform grid. For hex_lattice_42 dot uses 18 unique x-values and 12 unique y-values; CV of edge lengths is 0.10, NOT 0. The layer-x pattern is non-trivial.

## Cached dot positions for hex_lattice_42

```python
torch.load("eval_output/variant_bench_full/positions/hexagonal_lattice_42__graphviz_dot.pt")
```

First 8 nodes: `(117,-810), (153,-738), (261,-666), (279,-594), (369,-522), (387,-450), (459,-378), (81,-738), ...`

Range: x in [27, 459], y in [-810, -18]. Pitch_y appears uniform at ~72; pitch_x varies per row.

## Research targets

1. **Reverse-engineer the layer-x pattern.** Group nodes by y, sort by x within each layer, compute step distribution. Is it constant per layer but varying across layers? Is it driven by edge constraints? Read graphviz `dot` source if findable.

2. **Identify the principle.** Possibilities: (a) dot uses a network simplex for x-coordinates (each layer's x positions minimize edge length sum subject to spacing constraints — the classic Sugiyama-Gansner-North algorithm). (b) dot applies a per-layer median heuristic with a spacing constant. (c) dot uses brandes-koepf alignment which dagua already has but applied differently.

3. **Implement a working version.** This is not a hack — produce real pseudocode that dagua can call as a polish candidate (or a new post-pipeline op).

4. **Empirical test.** Implement in /tmp/, run on hex_lattice_42, triangular_lattice_36, parallel_cycles_4x5, and several non-lattice graphs (verify no regression). Quote real measured composite delta.

## Output

`.project-context/research/sprint_22_algo_bets/A_dot_lattice_mimic__<your_agent>.md`

- TL;DR (4-6 bullets)
- The algorithm dot uses (or your best theory) with citations
- Working pseudocode (200+ lines if needed)
- Measured deltas from /tmp/ implementation on at least 6 graphs
- Recommended integration point in dagua (polish candidate vs new pipeline)
- Risk / regression analysis with specific protected wins to verify

## Constraints

- READ-ONLY in dagua/. /tmp/ scripts allowed.
- Read CONTEXT.md first.
- BIGGER BET version: aim for 3000-5000 word findings if needed, with real empirical validation.
- No 30-minute hacks. If the algorithm needs 200-500 LOC of careful design, write it carefully.
