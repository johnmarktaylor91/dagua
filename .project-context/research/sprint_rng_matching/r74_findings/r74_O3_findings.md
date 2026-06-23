# r74 Cluster O3 — FMMM Fidelity Findings (READ-ONLY survey)

Scope: fmmm engines (classic_fmmm_steps10/100/200, classic_fmmm_graphviz_fdp_fidelity).
Source verdicts: `eval_output/fidelity_definitive_r73/per_combo.json`.
Reference C: `/home/jtaylor/projects/_references/ogdf` and `/home/jtaylor/projects/_references/graphviz`.

## Verdict census (fmmm, 383 combos, all mode A except 9 INSUFFICIENT)

| rung | count |
|------|------:|
| 1 (bit-exact) | 140 |
| 2 (stat) | 95 |
| 3 | 22 |
| 3Q (quality) | 32 |
| 4 DIVERGENT | **85** |
| INSUFFICIENT_DATA | 9 |

Divergent by engine: steps10=39, steps200=19, steps100=17, fdp_fidelity=10.
INSUFFICIENT: all 9 are classic_fmmm_graphviz_fdp_fidelity (large graphs, `matched_seeds_lt_30` timeout).

The r72/r73 thread4 bucketing (M1..M6) matches this data; r73 already landed MAARPacking
Best-Fit (79a2ac5) and multi-edge collapse (`_graphviz_fdp_collapse_parallel_edges`,
fmmm.py:5980 — confirms M5b is FIXED). The remaining 85 break down below.

---

## AVENUE A (TOP ROI): Multi-component rotation canonicalization gap — REAL, not floor

### What it is
OGDF FMMM's `pack_subGraph_drawings` (FMMMLayout.cpp:746) runs
`rotate_components_and_calculate_bounding_rectangles` for EVERY graph because the default
`stepsForRotatingComponents()==10` (FMMMLayout.cpp:269). For each component it:
- rotates through 10 angles `pi/2 * j/11`, j=1..10 (FMMMLayout.cpp:842-862),
- keeps the orientation with **minimum bounding-box area** — `calculate_area` with
  `comp_nr != 1` returns plain `width*height` (FMMMLayout.h:934-948; scaling=1.0 unless
  there is exactly one component),
- then applies an aspect-ratio "tipping" (90deg) branch (FMMMLayout.cpp:885-901).

The `comp_nr == 1` branch is DIFFERENT: it uses **aspect-ratio-scaled area** and the extra
`act_area_PI_half_rotated` test (FMMMLayout.cpp:858-879).

### dagua evidence (the gap)
- Single-component path: `_ogdf_fmmm_pack_single_component` (fmmm.py:1331, called at 1576
  inside both `_layout_ogdf_fmmm_small_fidelity` and `_layout_ogdf_fmmm_multilevel_fidelity`)
  rotates using `_ogdf_fmmm_square_aspect_area` (fmmm.py:927) + the `width/height<1` flip
  (fmmm.py:1367-1369). This correctly mirrors OGDF's `comp_nr == 1` case ONLY.
- Multi-component path: `_layout_ogdf_fmmm_component_fidelity` (fmmm.py:1755) lays out each
  component by CALLING those same single-level functions (1814/1822) — so each component gets
  the WRONG (aspect-ratio, single-component) rotation+flip baked in — and then computes a
  raw angle-0 bounding box (`_ogdf_fmmm_component_rectangle`, 1840) and packs via
  `_ogdf_maar_pack_component_transforms` (1850). **It never runs OGDF's per-component
  min-AREA rotation search.** Two bugs: (1) per-component orientation uses the wrong area
  metric/flip; (2) the multi-component min-area rotation loop is entirely absent.

### Why this is the M1 residual (and provably not FP-chaos)
Thread4 measured disp CONSTANT across steps10/100/200 for the multi-component graphs
(random_dag_50 1.238/1.196/1.193; multi_component_80 0.851/0.818/0.815;
parallel_cycles_4x5 -/0.671/0.667). FP-chaos shrinks with iters; a packing/rotation mismatch
does not. The per-component internal layouts already match (single-component combos for the
same topology are rung 1/2); only the relative orientation/placement of components differs.

### Combos it flips + to what tier
Multi-component disconnected graphs across steps variants:
random_dag_50, multi_component_80, disconnected_encoder_residual, disconnected_label_cycle_collage,
parallel_cycles_4x5, kitchen_sink_platform_graph, random_dag_200 — roughly **12-15 of the 85**
(the steps10/100/200 divergent rows whose graphs are multi-component). Expected landing: most
to rung 1/2 if internal component layouts already match at that step count (they do at
steps100/200 for the connected analogs). random_dag_50/random_dag_200 carry huge cross_R
(327-9000) — heavy crossing overshoot consistent with wrong component placement; these are the
strongest candidates to flip. A few may only reach 3/3Q if a residual MAARPacking ordering
detail remains.

### Confidence / effort
Confidence HIGH on root cause (direct source diff). Effort MEDIUM: port
`rotate_components_and_calculate_bounding_rectangles` faithfully — add a multi-component
rotation pass that (a) does NOT apply the single-component aspect-ratio rotation inside
per-component layout, (b) rotates each packed component by 10 angles picking min plain area,
(c) applies the tipping branch, (d) recomputes boxes before `_ogdf_maar_pack_component_transforms`.
Must verify `minDistCC()/2` offset (FMMMLayout.cpp:806-809) matches dagua's
`_ogdf_fmmm_component_rectangle` ±15 margin (fmmm.py:920-923). VERIFY on benchmark path,
matched seed, no runtime delegation.

---

## AVENUE B (PERF, unblocks INSUFFICIENT): fdp tlayout vectorization — NOT an algorithm gap

### What it is
The 9 INSUFFICIENT fdp combos timeout (`matched_seeds_lt_30`) because
`_graphviz_fdp_tlayout` (fmmm.py:5147; called via `_graphviz_fdp_component_layout`:5942)
runs **600 iterations of pure-Python scalar arithmetic**. Thread4's claim of "O(N^2)
all-pairs repulsion" is WRONG: dagua already ports Graphviz's spatial grid faithfully
(cell_size=3*K, bucket nodes into cells, repel only within-cell + 8 neighbor cells;
fmmm.py:5189-5243). Graphviz fdp `tlayout.c` uses exactly this grid by default
(`useGrid` on; DFLT_maxIters=600, DFLT_K=0.3 — constants match dagua at fmmm.py:2831-2832).
So complexity is already ~O(N) per iter; the cost is Python interpreter overhead per pair
(`_graphviz_fdp_apply_tlayout_repulsion_lists`, fmmm.py:4820, one Python call per node pair
per iteration).

### The win
Torch-vectorize the grid-cell repulsion: gather per-cell-neighborhood pair indices, compute
the repulsion in batched float64 tensor ops (preserving Graphviz's exact `K^2/(dist*dist2)`
form and zero-distance dispersal). Keeps faithfulness (same grid, same force, same iter
order semantics within a cell-neighborhood as long as accumulation order is reproduced or
proven order-invariant). Estimated 30-200x speedup at N=200-500, enough to collect 30 seeds.

### Combos it flips + to what tier
Recovers up to **9 INSUFFICIENT -> a real verdict** (ba_500, citation_dag_300, er_500,
grid_20x20, hub_spoke variants, powerlaw_500, rgg_500, sbm_5x50, small_world_500). Landing
unknown: these are large fdp graphs; thread4 (M5a) shows fdp spring FP-divergence is a FLOOR
for small/dense graphs, so several may land rung 4 anyway. Honest expectation: convert 9
INSUFFICIENT to determinate verdicts; perhaps 2-4 reach rung 2/3, the rest expose FP floor.
This is a measurement-completeness win more than a fidelity win.

### Confidence / effort
Confidence HIGH that it's pure Python overhead (grid already faithful). Effort MEDIUM-HIGH:
vectorizing within the grid while reproducing Graphviz's per-pair accumulation order is
delicate (float64 summation order affects bit-exactness; but these are mode-A stochastic, so
order-invariant batched sums are acceptable for stat tiers). Barnes-Hut is NOT needed and
would be LESS faithful than Graphviz's own grid — do not implement BH (thread4's BH
recommendation is misguided here; the grid IS the approximation Graphviz uses).

---

## AVENUE C (NEGATIVE): per-variant scale / param gap — no one-line win

Checked `_ogdf_fmmm_adapt_to_ideal_edge_length` (fmmm.py:853, applied at 1558/1575) and the
finalize/translate path (`_translate_packed_components_to_origin`, fmmm.py:6385). Scaling is
identical across steps10/100/200; the only per-variant difference is iteration count via
`_ogdf_fmmm_max_mult_iter` (fmmm.py:1468) / `max(100, 10*steps)`. No systematic per-variant
scale or param mismatch. The constant-disp signature of M1 graphs further rules out a scale
bug (a scale gap would shift disp uniformly but not survive Procrustes alignment, which
removes global scale). NEGATIVE — no cheap win here.

---

## FLOOR (genuine FP basin chaos — PROVE, don't chase): ~58 of 85

These are connected/compound graphs where dagua's single-/multi-level FMMM and OGDF diverge
into different attractor basins purely from libm vs torch/math FP differences:
- M2 steps10 floor (~21): all pass at steps100/200 (proves iters overcome FP, but 100/10
  iters land in a different basin). e.g. sierpinski_42 steps10 disp 11.3 -> rung2 at 100.
- M3 converging-but-not-converged (~24): disp monotonically shrinking with iters
  (deep_chain_20 2.140/0.255/0.078) — would pass at steps400-1000; bench only runs 10/100/200.
- M4 compound connected FP floor (~14): plain-graph FMMM iters, cluster info correctly
  ignored in fidelity mode; resnet_stack_4x16 / tl_transformer_1layer plateau at constant
  disp (random-basin floor).
- M5a fdp spring FP divergence (~10): small/dense + disconnected fdp; pure libm-sqrt floor.
- M6 stochastic border (~3): org_chart_deep / protein_ppi_200 alternate pass/fail across
  step counts — no monotone signal.

Floor evidence is solid: the SAME graph passes at a different step count (M2/M3) or plateaus
at the median Procrustes distance between two distinct energy minima (M4/M5a/M6). Bit-level
libm emulation in Python is infeasible. Accept and document; FP-chaos EVIDENCE is the
constant/monotone disp-vs-iters curve already in the data.

---

## HONEST SPLIT of the 85 DIVERGENT

| bucket | ~combos | fixable? |
|--------|--------:|----------|
| Multi-component rotation/packing (Avenue A) | 12-15 | YES (MEDIUM) |
| FP basin chaos (M2/M3/M4/M5a/M6) | ~58 | FLOOR |
| Residual ambiguous (multi-comp that may stay 3/3Q after A) | ~3-5 | partial |
| INSUFFICIENT (Avenue B, separate from the 85) | 9 | recover verdict (MEDIUM-HIGH) |

ROI order: **A (rotation, 12-15 flips, source-faithful, medium effort) >> B (perf, recovers
9 verdicts, few fidelity flips) >> C (negative)**. The floor (~58) is genuine FP chaos —
prove via existing disp-vs-iters curves; do not chase.

## Guardrail notes
- Avenue A is a source-faithful port of FMMMLayout.cpp:816-911 — NOT delegation, NOT
  laundering. Verify on benchmark path with matched OGDF runner params (fixedIterations =
  steps, randSeed) and matched seed; expect disp -> near 0 only if internal layouts already
  match at that step count.
- Avenue B must preserve Graphviz grid semantics; do NOT substitute Barnes-Hut (less
  faithful). Mode-A stochastic tiers tolerate order-invariant batched sums.
