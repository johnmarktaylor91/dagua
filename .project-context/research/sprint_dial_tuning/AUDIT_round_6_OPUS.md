# Round 6 Audit -- dial_tuning

## Verdict

- New audit: **PASS** (round 6 surgical revert successful; metric and visual corroborate)
- Stop criteria status: **CONTINUE** (one focused round 7 -- 3 actionable cosmetic gaps remain that round 4/5 left open and round 6 did NOT pick up)
- Mean Tier A L1: **2.971 -- VERIFIED** (recomputed from per_card summary)
- Median Tier A L1: 1.541
- Tier A cards with L1 < 5: **144 / 181 (79.6%)**
- Tier A cards with L1 < 3: 127 / 181 (70.2%)

The round-5 catastrophic regression has been undone. The 75x50pt theme + DECORATIVE_FILL_CARD pair-card override + reverted parity tolerances landed cleanly. The cluster z-order win from round 5 is preserved. Mean L1 at 2.971 is BACK in line with the round-4 best of 2.454, indicating the structural choice was correct -- BUT visually three round-4-flagged cosmetic gaps remain unfixed, and they are actionable in cosmetic scope.

## Round 6 wins verified (Group 1)

Opened side-by-side panels for the previously-stuck simple-shape and decorative-fill cards. All round-6 wins genuine:

| Card | Round-5 L1 | Round-6 L1 | Visual verdict |
|------|---:|---:|---|
| ellipse | (regressed) | 0.606 | Dagua ellipse is reasonable size (~258x130 px), no longer giant or tiny. Hairline border + white interior STILL there from round 4 (R4-N1 NOT closed) but size is fine |
| circle | (regressed) | 0.645 | Dagua circles fit panel, no overlap. Hairline border + white interior persist |
| rect | (regressed) | 1.268 | Dagua rect ~258x130, reasonable. Hairline border persists |
| gradient_linear | 12.474 | 8.714 | Gradient applied (orange/blue), node size matches round 4. Slightly smaller than graphviz still. |
| striped | (regressed) | 8.724 | Diagonal stripes applied; same size as gradient |
| pie | n/a | 9.131 | Pie wedges visible (green/orange/blue thirds), node ~260x130 |
| combo_bold_shadow_gradient_rounded | ~50 | 8.269 | Combo no longer 5x giant; nodes now match the small-graphviz scale |
| cluster_opacity_1_0 | 40.8 | 29.253 | Inner cluster border VISIBLE at full opacity (z-order fix preserved); fill opacity progression intact |
| cluster_opacity_0_6 | (high) | 19.161 | Both bboxes drawn, opacity progression correct |
| cluster_opacity_0_3 | (high) | 10.702 | Both bboxes faded correctly |

The pair-card DECORATIVE_FILL_CARD override is working for fill-pattern fixtures. BUT note: I see no evidence the override scaled simple-shape (ellipse/circle/rect) up to the ~290x163 graphviz match -- they look the same ~258x130 as the gradient/striped/pie cards. This is fine for L1 (the size is reasonable across all cards), but the round-4 visible cosmetic gap of "dagua nodes look hairline / no fill / smaller than graphviz" is STILL present on simple-shape comparison panels.

## Worst-10 root cause classification

| # | Card | L1 | Class | Round-7 actionable? |
|---|---|---:|---|---|
| 1 | nodes_borders_border_position_outside (B) | 30.06 | competitor_dial_semantic_mismatch | NO |
| 2 | clusters_opacity_1_0 (A) | 29.25 | layout_coupled (fixture topology + cluster-fill drawn over full layout bbox) | **PARTIAL** -- fixture redesign or cluster-fill clamp possible |
| 3 | nodes_borders_border_position_inside (B) | 28.77 | competitor_dial_semantic_mismatch | NO |
| 4 | evil_taxi_gradient_multiborder (B) | 19.67 | competitor_side_glitch (cytoscape can't compose multiborder + gradient + taxi) | NO |
| 5 | clusters_opacity_0_6 (A) | 19.16 | layout_coupled (same as #2) | **PARTIAL** |
| 6 | evil_taxi_self_loop (B) | 16.63 | competitor_side_glitch (cytoscape uses massive 600x500 ellipse for self-loop) | NO |
| 7 | combo_kitchen_sink_6 (B) | 14.40 | layout_coupled (taxi routing renders fundamentally differently) | NO |
| 8 | combo_kitchen_sink_5 (A) | 14.07 | multi_feature_density_combo (graphviz auto-shrinks to ~50x30 per node; dagua doesn't auto-shrink) | NO |
| 9 | combo_pie_shadow_gradient_bold (A) | 14.02 | multi_feature_density_combo (same as #8) | NO |
| 10 | combo_taxi_crossing_gap_gradient (B) | 13.10 | layout_coupled (cytoscape taxi routing differs) | NO |

8 of 10 are NOT actionable in cosmetic scope. Two cluster_opacity items are PARTIALLY actionable (Group 4 details below). The two `competitor_dial_semantic_mismatch` and three `competitor_side_glitch` items together make ~85 L1-points of unavoidable Tier B residual.

## Cluster_opacity layout coupling -- actionable or layout-scope?

I directly compared cluster_opacity_1_0's per_card panel:
- **Dagua side:** Outer cluster bbox spans ~370x380 px (Outer A top-left, Outer D bottom-right -- TB layout staircase); Inner cluster bbox is panel-spanning rectangle around Inner B + Inner C (which are also placed wide-apart-diagonal). Cluster fill paints the entire layout-engine bbox.
- **Graphviz side:** Outer cluster is a tight ~100x200 vertical strip around 4 nodes in single column; Inner cluster is a ~80x100 nested strip.

Two cosmetic-scope-actionable interventions:

### Option A: Fixture redesign (HIGH-LEVERAGE)
The cluster_opacity fixture builder spreads Outer A through Outer D into a wide topology. If the FIXTURE were rewritten to use a vertical chain (Outer A -> Inner B -> Inner C -> Outer D, all stacked), dagua's TB layout would naturally produce a tight strip matching graphviz. **This is a fixture-builder change, NOT a layout-engine change** -- entirely in cosmetic-scope. Risk: changes the visual semantics of the fixture (the user originally chose a 4-node graph to stress the cluster bbox; a chain of 4 doesn't stress it the same). Recommend: discuss with maintainer; if acceptable, this drops cluster_opacity_* from worst-list.

### Option B: Cluster-fill clamping (LOWER-LEVERAGE BUT PURE COSMETIC)
Currently dagua's cluster fill paints across the full layout-engine cluster bbox (which spans the layout). Could paint only within a tighter bbox = `union of immediate-children's tight bboxes + small padding`. This decouples the FILL region from the layout spread, while keeping the LABEL position. Risk: the result might look weird (nodes "outside" the fill region). Less recommended but mentioned for completeness.

### Option C: Accept as layout_coupled residual
Both above interventions are non-trivial to validate; the round-4 audit already classified this as out-of-scope. Three rounds of audits have flagged the layout coupling but no fix has landed. **Default classification: accepted_residual** unless maintainer wants to redesign fixtures.

My recommendation: classify as `layout_coupled / accepted_residual` for round 7, BUT propose to JMT that fixture-redesign for cluster_opacity_* is a one-shot fix worth considering as a separate task.

## Tier B border_position investigation

Direct inspection of `nodes_borders_border_position_inside_vs_cytoscape.png` and `..._outside_vs_cytoscape.png`:

**Dagua side:** Renders TWO nodes (Center + Inside / Center + Outside) at LARGE size (~300x200 px each), side-by-side. Border is rendered as a wide orange OUTER FRAME (~80px thick) around a small cream interior label. The two nodes overlap horizontally because they're both big.

**Cytoscape side:** Two SEPARATE centered nodes (~280x150 each), well-spaced. Border is rendered as a chunky ~30px orange ring around an internal cream label. The Inside vs Outside dial moves the border slightly -- but cytoscape paints both nodes at the same nominal size with the border varying.

**Verdict: `competitor_dial_semantic_mismatch`.** Cytoscape and dagua interpret `border_position: inside` / `outside` differently:
- Cytoscape: nominal node size constant; border drawn at slightly different position (subtle visual difference).
- Dagua: nominal node bbox grows to accommodate the border (border is rendered as part of the node's outer frame, expanding the visible footprint).

This is NOT a fixable cosmetic dial -- the two engines have a different semantic model of what `border_position` means. The Tier B L1 of 30.1 / 28.8 is an unavoidable consequence. Accept as residual.

(Side note: dagua's visualization is actually clearer about WHAT inside-vs-outside means -- but L1 metric punishes any visual disagreement with the cytoscape comparator.)

## Routing-fixture caveat (Group 6)

Codex left `edges_routing_*` (fan/chain fixtures) WITHOUT the DECORATIVE_FILL_CARD override. I checked `routing_bezier_vs_graphviz.png`:

- Dagua: 3 small ellipses (Stage A, B, C) at ~75x40 each in a small downward chain
- Graphviz: 3 large ellipses (~290x163) at full theme size in a vertical chain

The L1 numbers are LOW (bezier 0.889, straight 0.889, ortho 1.084, taxi 2.304) because most pixels are white-on-white. The routing dial WORKS (bezier curves traced, ortho corners, taxi steps) -- but at a smaller scale than graphviz. **Codex's choice to leave routing fixtures at theme default was CORRECT** per his note: "applying 200x110 footprint there pushed mean L1 above 3.0". The routing comparison panels don't visually match graphviz at 1:1 size, but the metric is healthy and bumping size makes things worse on the multi-node fan layouts.

Classification: `accepted_residual / metric_artifact` -- the dials are correct, the visual gap is metric-friendly, and fixing it via larger nodes regresses other cards.

## Tier C combos / evil cards (Group 5)

Spot-checked:
- `evil/evil_huge_arrows.png` -- tiny ellipses + small filled-triangle. The fixture is named "huge arrows" but renders small. NOT a round-6 regression -- this was small in round 4 too. Low priority.
- `evil/evil_max_opacity_stack.png` -- multi-color opacity-stacked ellipses cleanly visible (pink/orange/cyan/yellow). Z-order working.
- `combos/2way/bevel_shadow.png` -- bevel + shadow combine cleanly on Ingest/Validate/Review/Approve/Ship. Layout reasonable. Healthy.

No combo-card regressions from round 6. Dials cooperating across the gallery.

## Round-4 carryover items NOT closed by round 6

Three round-4 / round-5 findings explicitly recommended for fixing did not land in round 6's surgical revert:

### Round-7 priority 1 (CRITICAL): R4-N3 / R4-N5 cluster bbox border in label_position fixtures
Direct inspection of `clusters_label_position_top_center_vs_graphviz.png`:
- Dagua: Outer + Inner labels visible. Two horizontal lines (top and bottom of rectangles). **NO vertical strokes -- the cluster bbox is broken into top-edge and bottom-edge only.**
- Graphviz: Full visible rectangles around clusters with thin black border.

This is a **clear actionable real_cosmetic_gap + fixable_theme_or_render**. The cluster bbox stroke is being clipped or not drawn on left/right edges in label_position fixtures. Round 4 R4-N3 flagged this; round 5 z-order fix didn't address it; round 6 didn't address it.

### Round-7 priority 2 (CRITICAL): R4-N2 stroke_width 5.0 dial broken at high values
Direct inspection of `nodes_borders_5_0_vs_graphviz.png`:
- Dagua: Two ellipses side-by-side (Default + 5.0). "Default" has hairline-thin stroke. "5.0" has only ~1pt visible stroke -- nothing close to graphviz's chunky 5pt stroke.
- Graphviz: Stacked vertical with chunky ~4-5pt blue strokes on both nodes.

**Clear real_cosmetic_gap + fixable_theme_or_render.** The stroke_width=5.0 dial is rendering at a fraction of its specified width. This is a render-bug, not a layout issue. Independent of size.

### Round-7 priority 3 (HIGH): R4-N1 simple-shape light-blue fill + chunky border in comparison panels
Direct inspection of ellipse/circle/rect/roundrect comparison panels:
- Dagua: hairline-thin black border, white/transparent interior, no end-of-edge arrowhead.
- Graphviz: ~2pt blue border, light-blue fill, clear filled-triangle arrowhead.

The DECORATIVE_FILL_CARD override (per round-6 codex commit) is keyed on `nodes_shapes_*` group but appears NOT to apply default light-blue fill + chunky border to simple-shape comparison fixtures. Either (a) the override was scoped only to size and not to fill_color/border_width, or (b) the override doesn't actually fire on these card paths. Bumping fill_color to graphviz's `#E0EFFF` and border_width to ~1.5pt for these fixtures would close the visible gap that the metric is under-reporting (because most pixels are white-on-white).

Same for the comparison-panel arrowheads (R4-N4): there's a tiny ~6pt downward triangle near the Target node, but graphviz shows a clean ~10pt filled triangle right at the edge. Fix is fixture-local arrowhead size restoration on simple-shape comparison panels.

## STOP recommendation

**Recommend CONTINUE one more round (round 7).** Three actionable cosmetic gaps remain (cluster bbox border in label_position; stroke_width 5.0 broken at high values; simple-shape fill/border/arrow). All three fall squarely under `real_cosmetic_gap + fixable_theme_or_render`, all three were flagged in round 4, and round 6 did not address them.

If round 7 closes those three, the remaining residuals are:
- Cluster_opacity_* (layout_coupled or accepted_residual unless fixture-redesign)
- Tier B border_position_inside / outside (competitor_dial_semantic_mismatch)
- Tier B evil_taxi_*, kitchen_sink_6, taxi_crossing_gap_gradient (competitor_side_glitch / layout_coupled)
- combo_pie_*/donut_*/kitchen_sink_5 (multi_feature_density_combo -- structural; graphviz auto-shrinks, dagua doesn't)

After round 7, **the bar of "indistinguishable save for documented rendering-stack residuals" should be MET on ~94%+ of Tier A cards.** The remaining ~6% are residuals with documented reasons (layout-coupling, density-combo, dial-semantic-mismatch). At that point recommend STOP.

**Honest assessment: we are NOT at ceiling yet.** Three round-4 findings have survived two rounds without being addressed (R4-N1 arrowheads + simple-shape fill, R4-N2 stroke_width 5.0, R4-N3 cluster bbox border). They're all visible cosmetic gaps that the auditor flagged across rounds. Round 6 was a metric-rescue (un-broke the round-5 regression); round 7 should be the closer that addresses these specific three carryover items. If the maintainer wants ceiling, round 7 is the path. If the maintainer wants to ship at "metrically healthy" (mean L1 < 3.0 achieved), round 6 is acceptable but the visible gaps remain.

## Sprint summary suggestion (if STOP after a round 7)

Across rounds 1-6, dial_tuning sprint:
1. Mean Tier A L1 went from 35+ (round 1, with bbox-tight rescaling artifacts) to 2.971 (round 6) -- 11.8x improvement after metric pipeline fix and cosmetic dials
2. 144/181 Tier A cards now under L1=5.0; 127/181 under L1=3.0
3. Donut central transparency wired (round 2 R3-N14 closed)
4. Striped/hatched/pie patterns wired (round 3 R3-N17 closed)
5. Cluster z-order bug fixed (round 5 -- inner border visible above solid fill at opacity 1.0)
6. Pair-card DECORATIVE_FILL override pattern landed (round 4) and re-stabilized (round 6 surgical revert)
7. Parity-metrics regression lock at 2.0pt rx/ry/wedge tolerances re-established post-round-6

Remaining residuals (after a hypothetical round 7):
- Cluster_opacity_* (layout-fixture-coupled; fixture redesign possible but out-of-scope)
- Tier B border_position semantic mismatch with cytoscape (~58 L1)
- multi_feature_density_combos (graphviz auto-shrinks per-node; dagua doesn't -- structural)
- Tier B taxi-routing topology mismatches (~50 L1)

These are documented, principled residuals consistent with rendering-stack floor.

End round 6 audit.
