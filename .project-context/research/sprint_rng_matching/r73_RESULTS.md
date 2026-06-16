# r73 Drive-to-Zero Fidelity Sprint -- RESULTS (DEFINITIVE, supersedes r72)

**Date:** 2026-06-16. **Status:** COMPLETE. Current authoritative fidelity verdict (supersedes
r72_RESULTS.md and all earlier). Scorecard: `eval_output/fidelity_definitive/r73_scorecard_final.json`.
Report: `eval_output/fidelity_definitive_r73/`.

## Headline

| metric | r72 | r73 | delta |
|---|---|---|---|
| total divergent | 617 | **574** | **-43** (0 regressions) |
| escalation-divergent (Mode-A) | 331 | **309** | -22 |
| quality-identical (3Q) | 32 | **36** | +4 |

Escalation-divergent series: **705 (r70) -> 463 -> 418 -> 381 -> 331 -> 309 (r73)** = **-56% from r70**.

## The 43 eliminations (0 regressions)

| engine | n | fix | confidence |
|---|---|---|---|
| umap | 18 | parallel-edge multiplicity: `_build_undirected_adjacency` used `set` (dedup parallel edges -> weight 1.0); reference CSR sums them (-> 2.0). set->dict accumulator. (7e2f7ae) | HIGH |
| pivot_mds | 16 | OGDF-scale: `PivotMDSFinalizePositions` normalized to sqrt(N)*5; OGDF emits raw scale 100. skip_normalization on fidelity path + unweighted ref. (0c01c00) | HIGH |
| classical_mds | 5 | weighted `default` passed edge-weights to dagua while igraph uses unweighted BFS. Added to `_UNWEIGHTED_REFERENCE_LAYOUTS`. Resolved exactly the 5 weighted graphs. (24ae7d3) | HIGH |
| fmmm | 3 | OGDF MAARPacking Best-Fit component packing + multi-edge aggregation. (79a2ac5) | partial |
| neato | 1 | Graphviz polyomino component packing. (786f32b) | partial |

## What was NOT fixed (and why -- honesty over a pretty number)

- **The adversarial Codex critique STOPPED a ~218-combo LAUNDERING.** A research thread proposed
  reclassifying degenerate-optimal/quality-equal combos to 3Q by dropping the BH correction. The
  critique empirically ran the anti-laundering gate: the proposed rule passes 5/40 chance controls
  (12.5%), and the claimed "161" actually came from an inverted convention passing 11/40 (27.5%),
  with no cross-combo FDR. **The ~218 stay divergent** -- they have real quality gaps (e.g. ba_500
  sugiyama: 22344 crossings vs 2805 reference -- genuinely worse, not equal-quality).
- **The critique also corrected wrong packing-algorithm diagnoses** (Sonnet agents guessed
  "shelf"/"TileToRows"; actual source = Graphviz polyomino + OGDF MAARPacking Best-Fit). This made
  the packing fixes source-faithful (they'd have failed otherwise).
- **gem (22):** the parity guardrail FAILED -- dagua-gem != ogdf-gem at matched seed (the diagnosis
  was wrong; gem was already seeded). Stays divergent. The guardrail working IS the win.
- **sugiyama (~80 attempted):** 0 verified flips. The coord/mincross fixes need a deep `position.c`
  port (virtual/slack node weights, set_xcoords geometry) + deeper mincross. Discarded. The LP layer
  degeneracy (135) is genuinely-worse-quality, not quality-identical. LP-canonical-vertex spike =
  scoped future direction.

## Remaining ~574 = bottom of the well

Deep multi-layer ports (sugiyama position.c, fmmm component-rotation, neato edge-spline bboxes --
each a multi-week port where a verdict flips only when ALL layers match) + verified FP-basin floor
(sfdp 137, etc.). These are not absurd, but past the point of sprint-scale returns.

## Methodology notes (reusable -- two overlay traps hit this sprint)

1. **Seed-count match:** overlay re-bench must match the base data's seed COUNT or it Frankensteins
   (mixed-seed combos flip mode). mds/pivot at 5 seeds left 95 stale seeds -> fixed (Pass A2, 100 seeds).
2. **Seeded-ref keys:** overlay re-bench of a SEEDED-ref engine must re-run the ref WITH `--seed-refs`
   (same 100 keys) or old ref keys persist -> mixed ref cloud. igraph_mds (in SEEDABLE_BASES) -> fixed
   (Pass A3). Side effect: cosmetic mds B->A relabel (30 combos, same verdict; raw ModeA shows 339,
   real ModeA = 309).

## Artifacts
- Report: `eval_output/fidelity_definitive_r73/` (DEFINITIVE_FIDELITY_REPORT.md, per_combo.json, controls/)
- Merged verdicts: `eval_output/fidelity_definitive/per_combo_r73.jsonl`
- Scorecard: `r73_scorecard_final.json`
- Benchmark: `benchmark_100seed_r73_fixes`
- commits on develop: 24ae7d3, 7e2f7ae, 0c01c00, 79a2ac5, 786f32b
