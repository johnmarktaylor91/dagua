# Round 4 Audit -- dial_tuning

## Verdict

- New audit: **PARTIAL** (with major reservation -- see "Visual reality check")
- Stop criteria status: **CONTINUE** (1 more focused round, then likely STOP)
- Round 4 fix closure: **3/4 phases genuinely landed**, but the headline metric improvement is partly an artifact of how white-pixel-dominance interacts with the new metric pipeline -- the visual gap on simple/gradient/striped paths has actually MOVED IN THE WRONG DIRECTION on simple paths
- Mean Tier A L1: codex-reported 2.454 -- **CONFIRMED in the per_card summary**, but **DISPUTED** as a faithful "we got closer" signal -- see below

## Visual reality check (round 4 wins) -- the headline finding is more nuanced than the metric suggests

The L1 numbers genuinely dropped (e.g. ellipse 1.042 -> 0.413, rect 0.989 -> 0.731, gradient_linear 35.0 -> 2.288, striped 36.5 -> 2.001). I OPENED each of these comparison panels and the visual reality is:

**The L1 dropped because the bbox-tight + thumbnail rescaling was removed AND because most of each panel is white-on-white background, NOT because dagua and graphviz now look indistinguishable.** Concretely, in the simple-shape comparisons (`circle_vs_graphviz.png`, `ellipse_vs_graphviz.png`, `rect_vs_graphviz.png`, `roundrect_vs_graphviz.png`):

- **dagua's nodes are ~4-5x SMALLER than graphviz's** in linear pixel size (ellipse: dagua ~80x40 vs graphviz ~400x180; rect: dagua ~80x35 vs graphviz ~360x130; circle: dagua ~80x80 vs graphviz ~360x360).
- **dagua's nodes have NO fill color** -- they show white interior. Graphviz nodes have light-blue fill.
- **dagua's nodes have a hairline-thin border** -- graphviz nodes have a chunky 2-3pt darker-blue border.
- **dagua has NO end-of-edge arrowhead** -- graphviz shows a clear filled-triangle arrowhead at Target (R3-N8 PARTIALLY closed in pair-fixture but NOT in comparison panels).

The metric pipeline change correctly stopped renormalizing, but the dagua side is so much smaller than graphviz on so many cards that **most pixels are whitespace on both sides** -- so per-pixel L1 is small even though the visible content is grossly different.

This is not an L1 measurement that tracks visual indistinguishability. It tracks "how much non-white pixels disagree", and when both sides have ~5% non-white coverage on tiny different shapes, that is a small per-pixel-mean number.

The codex-flagged caveat ("Kept global GRAPHVIZ_STRICT_THEME node size unchanged because parity metrics lock fails otherwise; node-size tuning is fixture-local") is the explanation: **the size restoration was applied in fixture builders, not in the theme defaults**. Net effect: simple-shape comparison fixtures are now using even smaller nodes than round 3, while the gradient/pie/striped fixtures got the size restored. That's why gradient_linear visually looks like the gradient is now small but applied, while plain ellipse looks tinier than ever.

## Codex-claimed wins -- card-by-card visual verdict

| Card | L1 | Visual genuinely matches graphviz? |
|---|---|---|
| nodes_shapes_ellipse | 0.413 | NO -- dagua ellipse is ~5x smaller, no fill, hairline border, no arrowhead |
| nodes_shapes_rect | 0.731 | NO -- same story; dagua rect is tiny, no fill, no arrowhead, hairline border. R3-N6 NOT FIXED |
| nodes_shapes_roundrect | 0.710 | NO -- same; R3-N7 NOT FIXED |
| nodes_shapes_circle | 0.402 | NO -- dagua circle ~4x smaller, white interior, hairline outline |
| nodes_fills_gradient_linear | 2.288 | PARTIAL -- gradient applied (orange/blue), shadow visible, but node size still small relative to graphviz |
| nodes_fills_gradient_radial | 2.497 | PARTIAL -- radial gradient visible (orange center, dark edge), text contrast still poor in radial center |
| nodes_fills_fill_pattern_striped | 2.001 | PARTIAL -- diagonal stripes visible, applied correctly; size still small |
| graph_background_dark | 0.812 | PARTIAL -- bg fills full canvas, but dagua nodes are tiny dark-blue ellipses on dark bg (low contrast) vs graphviz chunky-light-blue on dark bg |
| graph_background_near_black | 1.426 | PARTIAL -- same as dark |
| evil_pie_shadow_gradient | 3.899 | YES (good) -- pie shadow gradient looks healthy on its own as a stress-test; layout works |
| evil_donut_diamond | 2.118 | YES (good) -- donut diamond cooperates (donut center IS now transparent showing diamond through; round-2 N14 actually CLOSED) |

Six of the eleven still have a real visible cosmetic gap; the metric is misleading for those.

## Round 3 finding closure update (re-test selected items)

| Round-3 Finding | Round-4 Status | Evidence |
|---|---|---|
| R3-N1 metric pipeline rescaling | CLOSED -- pipeline no longer renormalizes (bbox_inches=tight + thumbnail removed/disabled) | per_card_pixel_diff_summary.md headline numbers visibly reflect actual content |
| R3-N2 node size over-corrected on simple path | NOT CLOSED -- and arguably WORSE (codex applied size restoration only fixture-local; theme default still tiny) | comparisons/nodes/shapes/{ellipse,rect,circle,roundrect}_vs_graphviz.png |
| R3-N3 size bifurcation gradient/pie path | PARTIALLY CLOSED -- gradient/striped/pie fixtures now consistent with their size; but theme default still bifurcated from gradient sizing | comparisons/nodes/fills/{linear,radial,striped}_vs_graphviz.png |
| R3-N4 cluster bbox panel-spanning | NOT CLOSED -- dagua's cluster bbox still spans the panel where graphviz uses tight bbox; cluster_opacity_1_0 stays at L1=28.6 for this reason (down from 62 because layout-coupling residual still present but rescaling removed inflated multiplier) | comparisons/clusters/{1_0,0_6,0_3}_vs_graphviz.png |
| R3-N5 dark/near-black bg + node body residual | PARTIALLY CLOSED -- bg fills correctly, but tiny dagua nodes still don't match graphviz chunky nodes on dark bg | comparisons/graph/{dark,near_black}_vs_graphviz.png |
| R3-N6 rect outline visibility | NOT CLOSED -- rect outline still hairline-thin, no fill | comparisons/nodes/shapes/rect_vs_graphviz.png |
| R3-N7 roundrect outline visibility | NOT CLOSED -- same | comparisons/nodes/shapes/roundrect_vs_graphviz.png |
| R3-N8 pair-fixture comparison-panel arrows | PARTIALLY CLOSED -- pair-fixture reference renders may have arrows but COMPARISON panels still don't show end-of-edge arrowhead between Source and Target (verified for circle, ellipse, rect, roundrect, gradient panels, stroke_width 5.0) | comparisons/nodes/shapes/* + comparisons/nodes/borders/5_0_vs_graphviz.png |
| R3-N9 cluster border invisible | NOT CLOSED -- cluster_label_position cards now show NO cluster bbox at all on dagua side; round-4 explicitly flagged "cluster bbox + border" but the border is still missing in label_position_top_center comparison and 4-way combo | comparisons/clusters/top_center_vs_graphviz.png + combos/4way/cluster_gradient_shadow_double_border.png |
| R3-N10/N11 gradient text contrast | NOT CLOSED in radial; PARTIAL in linear | comparisons/nodes/fills/{linear,radial}_vs_graphviz.png |
| R3-N12 layout aspect ratio | OUT OF SCOPE -- layout-scope item, deferred |
| R3-N13 kitchen_sink_5 external label inverted | NOT VERIFIED in this audit (per metric still ~14.07 -- did not drop dramatically, indicating still wrong) | per_card summary |
| R3-N14 donut central cutout transparency | CLOSED -- evil_donut_diamond shows donut center IS now transparent showing diamond shape underneath | evil/evil_donut_diamond.png |
| R3-N15 metric pipeline node-size signal | CLOSED with caveat -- metric now reads size, but tiny nodes vs big graphviz nodes both leave panels mostly-white so the signal is weaker than wanted | per_card_pixel_diff_summary.md |
| R3-N16 layout coupling cluster fixture | OUT OF SCOPE (layout) |
| R3-N17 striped pattern path | CLOSED -- striped pattern now applied (diagonal stripes visible) and node size in fixture matches | comparisons/nodes/fills/striped_vs_graphviz.png |
| R3-N18 simple-shapes overall size | NOT CLOSED -- simple shapes are now even smaller; theme-default not bumped |
| R3-N19 arrowhead size pair-fixture | NOT VERIFIED |
| R3-N20 external_label vs internal label | NOT VERIFIED |
| R3-N21 cluster bbox layout-dependent | OUT OF SCOPE (layout) |

## Still-stuck cards -- root cause analysis

### clusters_opacity_1_0 (L1=28.6) and clusters_opacity_0_6 (L1=18.7)

**Root cause is COMBINED:**

1. **Cluster bbox panel-spanning** (the dominant residual). Dagua positions Outer A in upper-left, Outer D in lower-right -- forcing Outer cluster bbox to span ~80% of the panel. Graphviz uses a tight vertical strip ~250px wide. Even with tight bbox-padding (which round 4 may have addressed in fixture-local form), the LAYOUT topology still requires bbox to enclose all four nodes, and dagua's layout produces a wide spread. This is layout-scope, not theme-scope.
2. **Inner cluster bbox missing on dagua side at 1_0 opacity**. At opacity 1.0 the Inner cluster shows only as a thin LINE on dagua side -- the Inner bbox isn't drawn / has zero stroke. Graphviz shows a clear thin rectangle around Inner B and Inner C. At opacity 0.3 and 0.6 the Inner cluster IS drawn (visible darker-blue fill rectangle) -- so the issue is solid-fill specifically.
3. **Dagua nodes are pale/white inside on this card** vs graphviz's light-blue with darker borders.

The opacity DIAL itself works monotonically (0.3 -> 0.6 -> 1.0 progresses faded -> medium -> solid as expected). So the FEATURE is correctly wired. The remaining L1 is a combination of (1) layout-scope bbox shape, (2) Inner cluster bbox missing at opacity 1.0 specifically, and (3) the persistent node-size/fill mismatch.

A round-5 fix could close (2) and (3) which would drop L1 by maybe 30-40%. (1) is layout-scope.

## New findings (round-4 introduced or surfaced)

| # | Severity | Card | Tier | Tool | Element/Region | Finding | Class | Action | Evidence |
|---|---|---|---|---|---|---|---|---|---|
| R4-N1 | CRITICAL | comparisons/nodes/shapes/* | A | graphviz | dagua node interior + size + border | Dagua's plain-shape nodes are ~4-5x smaller than graphviz's, have no fill (white interior), hairline-thin border, and no end arrowhead. Despite low L1 (0.4-0.7) the visual gap is substantial. The metric is being fooled by white-on-white pixel agreement; the cosmetic gap is real | real_cosmetic_gap | fixable_theme_or_render -- bump GRAPHVIZ_STRICT_THEME default node size to ~75x50, restore default light-blue fill_color, restore visible border stroke width | comparisons/nodes/shapes/{circle,ellipse,rect,roundrect,triangle,etc}_vs_graphviz.png |
| R4-N2 | CRITICAL | comparisons/nodes/borders/5_0_vs_graphviz.png | A | graphviz | dagua border stroke at width 5.0 | At stroke_width=5.0, dagua's border looks like 0.5-1.0pt -- maybe 1/10th of graphviz's chunky 5pt stroke. The dial is broken at high values. Layout also puts nodes side-by-side instead of vertical | real_cosmetic_gap | fixable_theme_or_render | comparisons/nodes/borders/5_0_vs_graphviz.png |
| R4-N3 | CRITICAL | comparisons/clusters/top_center_vs_graphviz.png + 4-way combo | A | graphviz | cluster border on label_position fixture | Cluster bbox is COMPLETELY INVISIBLE on dagua side in label_position cards and in 4-way cluster_gradient_shadow_double_border combo. Only "Outer"/"Inner" labels visible, no bbox lines. Round 3's collateral persists. Round 4 claimed phase-2 cluster bbox + border fix but it didn't reach these cases | real_cosmetic_gap | fixable_theme_or_render -- restore cluster default border to "#000000" 1pt while keeping fill off | comparisons/clusters/top_center_vs_graphviz.png + combos/4way/cluster_gradient_shadow_double_border.png |
| R4-N4 | HIGH | comparisons/nodes/shapes/* (all 17 shape comparisons) | A | graphviz | end-of-edge arrowhead | NONE of the simple-shape comparison panels show an arrowhead at Target. Graphviz consistently shows a filled-triangle arrowhead. Round-4 phase 3B claim re: pair-fixture arrows did not propagate to comparison panels. Same problem on stroke_width comparison panels | real_cosmetic_gap | fixable_theme_or_render | comparisons/nodes/shapes/*_vs_graphviz.png + comparisons/nodes/borders/5_0_vs_graphviz.png |
| R4-N5 | HIGH | clusters_opacity_1_0 | A | graphviz | Inner cluster bbox at opacity 1.0 only | At opacity_0_3 and opacity_0_6 the dagua Inner cluster bbox IS visible as a faded/medium rectangle. At opacity_1_0 it disappears -- shows only a thin line. The inner-cluster fill at full opacity is hiding the bbox stroke beneath, OR the bbox stroke is not drawn at all. Inconsistent with the dial values 0.3/0.6 where the bbox renders | real_cosmetic_gap | fixable_theme_or_render -- ensure cluster bbox is drawn ABOVE solid fill, not below | comparisons/clusters/{0_3,0_6,1_0}_vs_graphviz.png |
| R4-N6 | HIGH | comparisons/clusters/* (all opacity cards) | A | graphviz | cluster bbox aspect ratio | Even with the dial right, the dagua cluster bbox is roughly square (panel-wide) while graphviz is a tight vertical strip. This is dominantly layout-driven (Outer A and Outer D get placed at opposite corners) | layout_coupling | layout_scope -- defer | comparisons/clusters/*_vs_graphviz.png |
| R4-N7 | HIGH | comparisons/graph/{dark,near_black}_vs_graphviz.png | A | graphviz | dagua nodes on dark bg | Dagua's dark-bg nodes are tiny dark-blue ellipses, almost invisible on the dark bg. Graphviz's are chunky light-blue. The fill_color or auto-contrast on dark bg is wrong | real_cosmetic_gap | fixable_theme_or_render -- on dark bg, switch fill_color or invert per WCAG contrast | comparisons/graph/{dark,near_black}_vs_graphviz.png |
| R4-N8 | MEDIUM | comparisons/nodes/fills/radial_vs_graphviz.png | A | graphviz | text contrast on radial gradient | Text 'Source' and 'Target' on the radial gradient sits in the orange center, dark text on dark center -- low readability. R3-N10 not closed | real_cosmetic_gap | fixable_theme_or_render -- compute text color from gradient center luminance | comparisons/nodes/fills/radial_vs_graphviz.png |
| R4-N9 | MEDIUM | combos/4way/cluster_gradient_shadow_double_border.png | C | n/a | combo cluster integrity | When cluster + gradient + shadow + double-border combine: gradient applied, shadow applied, double-border applied (visible), but cluster bbox is GONE. Only "Primary cluster" label visible. Same root cause as R4-N3 | real_cosmetic_gap | fixable_theme_or_render | combos/4way/cluster_gradient_shadow_double_border.png |
| R4-N10 | MEDIUM | nodes_text_external_label_*_vs_graphviz | A | graphviz | external label position dial | Cards exist as Tier A (round 3 phase 11 fix) at L1=0.21-0.25 -- low, suggesting the dial works. Not deeply re-inspected this round | n/a | accepted_residual | per_card summary |
| R4-N11 | LOW | combos/2way/shadow_gradient.png | C | n/a | combo healthy | Combo correctly renders shadow + gradient on Ingest/Validate/Approve/Review/Ship -- gradient visible, shadow visible, layout reasonable. No defect | n/a | n/a | combos/2way/shadow_gradient.png |
| R4-N12 | LOW | evil/evil_donut_diamond.png | C | n/a | donut central transparency | Donut center IS now transparent (showing diamond underneath). R3-N14 closed | n/a | n/a | evil/evil_donut_diamond.png |
| R4-N13 | LOW | evil/evil_pie_shadow_gradient.png | A | graphviz | pie + shadow + gradient combo | Cooperates well: shadow under pie, gradient on pie wedges, label centered. Healthy | n/a | n/a | evil/evil_pie_shadow_gradient.png |
| R4-N14 | LOW | evil/evil_max_opacity_stack.png | C | n/a | max opacity stack | Stack of varying-opacity ellipses renders cleanly with all colors visible. No defect | n/a | n/a | evil/evil_max_opacity_stack.png |
| R4-N15 | LOW | evil/evil_huge_arrows.png | C | n/a | "huge arrows" don't look huge | The fixture's promise is huge arrows; what's rendered is small nodes with a small filled-triangle. Probably the same theme-default tiny-node issue | real_cosmetic_gap | fixable_theme_or_render via R4-N1 | evil/evil_huge_arrows.png |
| R4-N16 | LOW | clusters_corner_radius_* | C | n/a | corner radius dial | Tier C, no competitor. Per file inspection elsewhere these dial monotonically | n/a | n/a |
| R4-N17 | INFO | The codex caveat | n/a | n/a | fixture-local size | Codex applied size restoration in fixture builders not theme defaults because parity-metrics-lock fails. This is hidden coupling: the gallery test harness has a pinned snapshot of theme defaults that must not change. Net effect: gallery tests pass, but downstream API consumers using GRAPHVIZ_STRICT_THEME directly get the round-3 over-corrected (tiny) size | api_consistency_concern | dagua_design_decision_required -- either bump theme default + update parity snapshot, or accept that theme default is "node small, gallery fixtures override" | (no card; design issue) |

**Total NEW findings: 17.** (10 minimum met. Of these: 8 are real_cosmetic_gap + fixable_theme_or_render; 1 is layout_scope; 1 is api_consistency_concern; 7 are LOW/INFO/accepted.)

## Combination integrity assessment (Tier C combos + evils)

Looking at Pie Shadow Gradient, Donut Diamond, Max Opacity Stack, Shadow+Gradient 2-way, and the 4-way Cluster + Gradient + Shadow + Double Border:

- **Gradient + shadow combine cleanly** -- visible across multiple combos.
- **Pie + shadow + gradient cooperate** in the evil_pie_shadow_gradient stress.
- **Donut + diamond shape cooperate** with central transparency through to diamond shape -- this is round-2 N14 closed.
- **Multiple opacity stack** renders all layers without z-order failure.
- **Cluster + N-feature combos break the cluster bbox visibility** -- the cluster border stays invisible across label_position, 4-way cluster combos. This is a recurring issue.

So combination integrity is generally GOOD except for any combination involving cluster bbox visibility, which is broken across the board.

## Tier B status (post round-4)

| Card | Round-3 L1 (input) | Round-4 L1 | Visual reality |
|---|---|---|---|
| nodes_borders_border_position_inside | (was 22.96 in round 1) | 18.679 | Cytoscape native renders inside-stroke; dagua's stroke is on-axis. Real gap remains |
| evil_taxi_self_loop | (was 17.83) | 16.633 | self-loop renders cleanly on dagua side; comparison file not regenerated for cytoscape. Cytoscape side has a different topology |
| nodes_shapes_cloud | (was ~0.8) | 0.572 | Mermaid-anchored; lower L1 indicates pipeline fix helped |
| nodes_shapes_stadium | (was ~0.9) | 0.463 | Same |
| evil_taxi_gradient_multiborder | (was 26.901) | 19.673 | Cytoscape side glitch persists; partial residual; mark accepted |
| evil_self_loop_styled | (was 11.831) | 11.831 | Static -- still real gap |

Tier B saw across-the-board L1 reduction from the metric pipeline fix, consistent with the Tier A improvement source. None reached zero.

## Recommended fix order for round 5 (or STOP)

I am NOT recommending STOP. There are still clear `real_cosmetic_gap + fixable_theme_or_render` items.

### Round 5 priorities (P0 -> P2)

**P0 (high impact, fixable in cosmetic scope):**

1. **Bump theme-default node size, fill, stroke** (R4-N1, R4-N2, R4-N4, R4-N7, R4-N15). Either:
   - (a) Bump GRAPHVIZ_STRICT_THEME default node width/height to ~75x50pt, default fill_color to graphviz's `#E0EFFF`, default border_width to ~1.5pt -- AND update the parity-metrics-lock snapshot atomically, OR
   - (b) Document explicitly that simple comparison fixtures get a fixture-local large-node profile that matches graphviz, so the comparison panels DO show the size match.

   Either way, the comparison panels should land at "dagua circle/ellipse/rect VISUALLY match graphviz at 1:1 size", not at "low L1 because both are mostly white".

2. **Restore cluster default border** (R4-N3, R4-N5, R4-N9). Set cluster default border to `"#000000"` 1pt with `fill_color=""` and `fill_opacity=0.0`. Verify cluster bbox is drawn ABOVE solid fill at opacity_1_0 (z-order fix). Verify cluster bbox visible in label_position_top_center comparison + 4-way cluster combo.

3. **Restore comparison-panel end arrows** (R4-N4). Pair-fixture comparison renders for {circle, ellipse, rect, roundrect, all gradient/fill cards, stroke_width} should show the filled-triangle arrowhead at Target. Round 4 phase 3B was meant to do this; it propagated to reference renders only.

**P1 (visual quality):**

4. **Gradient text contrast** (R4-N8). Compute text color from gradient center luminance.

5. **Inner cluster bbox at opacity_1_0** (R4-N5). Stroke + fill z-order: stroke must be drawn after fill.

**P2 (low value, save for stretch round):**

6. Bump rect/roundrect outline visibility specifically (R3-N6/N7) if not already covered by P0(1).

7. Verify combo_kitchen_sink_5 / combo_external_label_diamond_shadow if maintainer wants.

### The codex CAVEAT -- is "fixture-local node size" acceptable?

I assess this as **api_consistency_concern, dagua_design_decision_required**.

- If GRAPHVIZ_STRICT_THEME is meant to match graphviz visually for downstream consumers (e.g., a maintainer using `dagua.set_theme('graphviz_strict')` and expecting graphviz-shaped output), the fixture-local override is hiding a theme default that doesn't actually look like graphviz.
- If GRAPHVIZ_STRICT_THEME is purely a regression-test snapshot, then fixture-local is fine and the user-facing "graphviz parity" is a separate, gallery-only concept.

Recommend: ask the maintainer (JMT) which interpretation. If (1), round 5 must update both theme + the parity-metrics-lock snapshot atomically. If (2), document that GRAPHVIZ_STRICT_THEME != "looks like graphviz" -- it's a frozen-snapshot baseline.

## STOP verdict

**CONTINUE for one more round (round 5) targeting the P0 items above (3 of them).**

After round 5:
- If R4-N1/N3/N4 close, then the visible cosmetic gap closes -- comparison panels would actually look like graphviz at 1:1 size.
- Layout-scope items (R3-N12, R3-N16, R3-N21, R4-N6) remain but those are out of scope.
- Combination integrity is healthy except for cluster bbox visibility, which round 5 fixes.

After that round 5, if zero `real_cosmetic_gap+fixable_theme_or_render` findings remain, recommend STOP.

**Estimate: we are 1 round away from ceiling, NOT at ceiling now.**

The dial-tuning sprint has done substantial work; mean Tier A L1 is genuinely lower (the metric pipeline fix unlocked real signal); donut transparency, opacity dial, taper, text outline, white label plate, skipped-comparison renderer, striped pattern, evil donut all visibly closed across the rounds. But the simple-shape parity (the most-watched comparison) still has visible gaps that the metric is under-reporting. One more round is warranted.

End round 4 audit.
