# Postmortem: Graphviz Cosmetic Parity Sprint (2026-04-27)

## What happened

Drove dagua's `graphviz_strict` theme toward visual parity with native graphviz `dot` across 18 rounds (9 implementation + 9 audit). Each round used a fresh Opus 4.7 picky audit to drive the next codex / Opus implementation round.

Round-by-round audit verdicts (PASS / PARTIAL / FAIL of the prior round's claimed fixes):
- R4 (initial picky audit): 19 departures found (8 HIGH, 7 MED, 4 LOW)
- R6 audit (post R5): 6 P / 5 partial / 2 F / 2 deferred + 1 new regression
- R8 audit (post R7): 3 P / 5 partial / 1 F + 4 new regressions
- R10 audit (post R9): mostly green + 3 new HIGH (puffy nodes, edge labels, arrow inconsistency)
- R12 audit (post R11): 0 P / 1 partial / 2 F — direction inverted
- R14 audit (post R13): 1 P / 4 partial / 1 F
- R16 audit (post R15): 1 P / 2 partial / 2 F (overshoot inversions)
- R18 audit (post R17): 1 P / 2 partial / 1 F + 2 new

Final user-flagged complaints status: font CLOSED, text-centering CLOSED, cluster-boxes CLOSED (modulo H4/H5 deferred as layout-side), arrows still PARTIAL.

## What worked

1. **Opus over Sonnet for visual audits.** Empirically substantial. Sonnet's R4 STOP verdict missed 19 departures that Opus caught, including a wrong-font issue requiring `fc-match` and a structural star stroke geometric defect. Sonnet should never drive visual parity audits.
2. **Subagent delegation pattern.** Visual audits in subagent context, main agent reads markdown reports — saved an order of magnitude of main-context tokens. Critical for any multi-panel comparison.
3. **Two-way pre-cropping.** Once we hit Opus's 2000px image cap, cropping 3-way galleries to 2-way kept the loop running. Should have done this from round 1.
4. **Theme/layout orthogonality enforcement.** When implementer drift toward layout work surfaced (cluster bbox sizing), correctly deferring those as architecture-protected kept the cosmetic sprint coherent.
5. **Per-round commits with descriptive messages.** Bisectability + clear rollback points. Did not need to use them, but the discipline kept changes small.

## What didn't work

1. **Zigzag oscillation around scalar targets.** Font size, min_height, edge_label_ratio, arrow proportions ALL went through correction → over-correction → correction cycles. We crossed the optimum each time. Symptom of single-parameter-step tuning without a global error metric.
2. **Qualitative deltas.** Audits said "bring within 10%" or "match dot's apparent size." Implementer interpreted as a guess. Next audit found it overshot 5% or undershot 15%. The audit→implement→audit loop has no shared numeric target.
3. **Coupled parameters tweaked in parallel.** Round 11 changed font size, padding, min_w, min_h, AND introduced compact_shape_factors — five interacting changes. When the result was puffy nodes, you couldn't tell which knob caused it. Each round tweaked 5+ parameters across 5+ dimensions; nothing got locked.
4. **No regression tests for "passing" parameters.** Round 3's font fix was correct but kept being tweaked in rounds 5/7/9 as side effects of other changes. Should have been frozen with a unit test that fails on drift.
5. **No upfront target extraction.** Native dot's SVG output contains literal numeric targets for ellipse semi-axes, font-size attributes, arrow polygon vertices, cluster rectangle positions. We discovered these ad-hoc per round instead of harvesting all targets at sprint start.
6. **Token cost.** ~2M+ tokens across 9 implementation rounds (each ~150-250K) + 9 audit rounds (each ~50-150K). Should have converged in 3-4 rounds with a tighter loop.
7. **No "stop" criteria.** Sprint kept going because the audit kept finding things — Opus is incentivized to be picky, so naturally found new departures. Without a quantitative gate, "fully satisfied" is unreachable.

## Diagnosis

The loop architecture is wrong for this kind of work. We treated cosmetic parity as a textual review-and-revise cycle (good for code logic) when it's actually a numeric optimization problem (parameters to dial in to match a reference). Visual parity is a distance-minimization task; the loop should reflect that.

## Better approaches

### A. Anchor numeric targets upfront

Before round 1, parse native graphviz `dot -Tsvg` output for each test panel. Extract:
- Font family, font size (literal pt value), font weight
- Ellipse semi-axes for each node
- Arrow polygon vertices (gives length, width, fill)
- Cluster rectangle (x, y, w, h), stroke color, fill color
- Edge stroke width, edge endpoint positions

Convert to a target table: `{panel: {node: {ellipse_w: X, ellipse_h: Y, ...}, ...}}`. Audit becomes "for each target, is dagua within tolerance?" Convergence is well-defined.

### B. Pixel-diff infrastructure

Render dagua and dot at the same canvas size with same node positions (already done). Compute per-region scalar diffs:
- Mean L1 RGB error per panel region (node, edge, cluster, label)
- Per-feature extractors: ellipse aspect ratio, label cap-height, arrow filled-area, etc.

Track a vector of error metrics across rounds. Each round's commit must reduce total error AND not regress any region by more than ε.

### C. Decompose into orthogonal sub-sprints

Instead of "fix everything cosmetic," split:
- **Sprint A — Typography.** Lock everything else; sweep font_family, font_size, label_font_size until typography metrics in tolerance. Ship.
- **Sprint B — Arrows.** Lock typography (regression tests). Sweep arrow_length, arrow_width, fill flags, arrow primitives. Ship.
- **Sprint C — Clusters.** Lock prior. Sweep cluster fill, opacity, stroke, label.
- **Sprint D — Ellipses + nodes.** Lock prior. Sweep node padding, min sizes, ellipse aspect.
- **Sprint E — Edges.** Lock prior. Sweep edge width, curvature, color, endpoint trim.

Each sprint converges in 1-3 rounds because there's only one dimension active. Locking via regression tests prevents collateral damage.

### D. Parallel value sweeps within a round

For each tunable parameter, render the gallery with N candidate values in parallel (e.g. font_size in {12, 13, 14, 15, 16}). Compute the error metric for each. Pick the minimizer. One round = one parameter converged.

### E. VLM as gradient oracle, not target setter

Audits should answer:
- For each parameter currently in flight: "is X too big, too small, or right? Confidence 0-1."
- "What did this round close? What did it open?" (delta from prior round)

NOT:
- "Recommend value Y for parameter X." (audit's numeric estimates have been systematically wrong by 25%+)

VLM provides the direction; the actual value comes from sweep + metric, or from anchor targets.

### F. Hard stop criteria

Define before sprint:
- Per-region error budget (e.g. RGB-L1 < 5 per pixel)
- Per-feature tolerance (e.g. ellipse aspect within 5% of dot's)
- Acceptable residual list (font hinting, AA, B-spline routing)

When all metrics in tolerance OR no actionable improvement direction, STOP. Don't ask Opus "is this perfect?" — the answer is always "no."

## Recommended workflow for visual parity tasks

```
1. SETUP
   - Render reference (dot, design tool, target system) for ~10-20 representative cases
   - Extract numeric targets from reference (SVG path data, computed sizes, colors)
   - Define error metric: per-region pixel diff + per-feature deviation
   - Define stop criteria (error budget per region, tolerance per feature)
   - Decompose into orthogonal sub-sprints (typography, arrows, clusters, etc.)

2. ITERATION (per sub-sprint)
   - Identify the parameter set for THIS sub-sprint (lock everything else)
   - Sweep candidate values in parallel; render N variants
   - Compute error metric for each variant; pick minimizer
   - Apply; verify no regression in locked dimensions (regression tests)
   - If error is in tolerance, FREEZE the parameter; add unit test
   - VLM audit only as a sanity check / catch-blind-spots, not as primary loop driver

3. STOPPING
   - Sub-sprint converges when all its parameters frozen
   - Sprint converges when all sub-sprints frozen
   - Then a final VLM audit catches anything the metric missed

4. MAINTENANCE
   - Frozen parameters have regression tests; future changes can't break them
   - Acceptable residuals documented (e.g. font hinting differences)
```

## Sharpened plan to finish this sprint

If/when resuming graphviz parity work after this postmortem:

1. **Build the diff infrastructure first.** Write `scripts/parity_metrics.py` that:
   - Parses dot's SVG for each panel
   - Extracts target features per panel
   - Renders dagua's strict gallery
   - Computes per-feature deltas (ellipse aspect, font size, arrow geom, cluster props, edge stroke)
   - Outputs a JSON table of {panel: {feature: {target: X, dagua: Y, delta: Z, in_tolerance: bool}}}
2. **Run baseline.** Sees where round 17 actually stands quantitatively. Probably ~80% of features in tolerance.
3. **Pick one out-of-tolerance dimension at a time.** Use the metric to pick the right value (binary search if needed) instead of guessing-and-correcting.
4. **Freeze + regression-test as you go.** Each fixed parameter gets a unit test asserting it stays within tolerance.
5. **Stop when metric says stop.** Document residuals. Send pictures.

Estimated cost: 2-3 hours total for diff infrastructure + 1-2 rounds to close all in-tolerance items. Compare to ~10+ hours and 9 rounds we just spent.

## Generalized lessons

1. **Visual parity is optimization, not review.** Treat it numerically, not textually.
2. **Reference is data.** Parse SVG / design specs / measurements as concrete targets. Don't let VLM estimate.
3. **VLM = direction, not magnitude.** Ask "bigger or smaller?", not "by how much?"
4. **Decompose. Lock. Test.** One dimension at a time. Freeze passing values with tests.
5. **Pixel diff > qualitative review.** Build the metric early; iterate against it.
6. **Define stop criteria upfront.** "Until perfect" is undefined and unreachable; "until error < ε" is achievable.
7. **Sweep, don't step.** Parallel candidate values + minimizer beats single-step adjustments.
