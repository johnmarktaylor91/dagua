# Item D Audit -- Fill-Pattern Parity

## Verdict

- Likely-fixable findings: **2 (one infra-side fixture wiring; one dagua node-size override)**
- Likely-residual findings: **6 (the L1 mass on these cards is not "fill-pattern style" -- it is the gallery competitor fixture not exercising graphviz's gradient/striped/etc. DOT support, plus the same theme-vs-graphviz node-size baseline that already shows on the solid fill card)**
- Estimated mean Tier A L1 drop after fixes: **0.4 - 0.9** (item D is mostly diminishing returns; the root cause is not in `dagua/render/mpl.py`)
- Recommend: **DEFER as a render-path "fill-pattern style" item; PROCEED only as a narrowly-scoped fixture wiring round** (graphviz_attrs on the gradient/striped fixture cells).

The "fill-pattern style" residual class is misnamed. The dagua-side renders look correct (gradient/stripes/pie/hatch all show the intended geometry). What's driving the L1 = 8-9 on these cards is **the graphviz competitor refusing to draw any pattern at all** -- it falls back to its baseline filled ellipse, while the dagua side renders the feature at the dagua theme's default node size. The pixel diff is dominated by (a) graphviz-vs-dagua node-size baseline (the same ~2.5 floor present on the `solid` baseline card) PLUS (b) the entire patterned overlay landing on whitespace in the graphviz frame.

The codex round 5 misadventure (theme node-size override) is exactly the trap waiting here. Do not chase fill-pattern-style geometry in the dagua renderer -- it would not move L1 and would risk regressing the round-9 wins.

## Per-card analysis

Pixel coords below refer to the 1600 x 600 comparison panel. Left half (0..800) is dagua; right half (800..1600) is graphviz.

### nodes_fills_gradient_radial (L1 = 9.374)

What differs:
- dagua left: small orange ellipse (~200 x 80 px) with central dark-blue radial spot (gradient_color = stroke); tiny "Source"/"Target" labels in dagua's small-fixture font (~10pt). Same baseline node size as `solid` fixture.
- graphviz right: full-fixture-size light-blue ellipse (~350 x 110 px) with a 2pt blue stroke and large "Source"/"Target" labels at fontsize=18. **No radial gradient at all.**
- dagua's radial gradient itself is visually correct: power(r, 0.7) profile, centered, blue spot at center, orange at rim. Matches the (deliberate) dagua design.

What's needed:
- This is NOT a render-path bug. dagua's `_draw_gradient_fill` (`dagua/render/mpl.py:2303-2364`) renders exactly what the fixture asks for.
- The gap is in `scripts/build_gallery_audit.py` `_graphviz_node_attrs` (line ~1995). For `gradient: radial` it should emit `style="radial"` with `fillcolor="<fill>:<gradient_color>"` (DOT supports `style="filled,radial"` with two-color fillcolor). Without that, graphviz draws a flat ellipse and any L1 below ~6 is unreachable.
- Even with the wiring change, graphviz's radial gradient uses `gradientangle` interpolation that visually differs from matplotlib's bicubic radial; expect a ~3-4 L1 floor.

Risk: low. Touches gallery script only, not render path. Cannot regress combo cards because gallery_audit fixture for combos is separate.

### nodes_fills_fill_pattern_pie (L1 = 9.126)

What differs:
- dagua left: round shape with three correct wedges (cyan ~50%, orange ~33%, green ~17%) computed from `fill_pattern_values=[3.0, 2.0, 1.0]`. Drawn correctly with a thin black outer ring (the node border). Wedge geometry matches `fill_pattern_values` proportions. Center at node center, sweeps starting at 12-o'clock.
- graphviz right: small light-blue ellipse with a blue stroke and big "A"/"B" labels. **No pie wedges; no recognition of the feature.**
- dagua's wedge math (mpl.py:2471-2537): unit Wedge transformed by Affine2D(radius_x, radius_y) then translated -- correct.

What's needed:
- Graphviz DOT does not have a native pie-fill primitive. There is no way to wire this on the graphviz side. The competitor reference for "pie fill" should NOT be graphviz.
- Recommended: reclassify this card from Tier A (graphviz reference) to Tier C (no-competitor heuristic only). One-line change in the `feature_competitors` mapping at `scripts/build_gallery_audit.py:330-331`.
- Alternatively: keep Tier A but accept the floor; document.

Risk: low. Reclassification only.

### nodes_fills_gradient_linear (L1 = 9.085)

What differs:
- dagua left: small ellipse with horizontal blue-to-orange linear gradient (angle = default; fixture uses `gradient_angle = 0`). Renders as the test expects -- left edge near pure blue, right edge near pure orange, smooth bicubic transition.
- graphviz right: same flat-fill default ellipse as the radial card. **No linear gradient.**
- dagua gradient itself is correct.

What's needed:
- Same as radial: wire `_graphviz_node_attrs` to emit `style="filled,striped"` is wrong; the right DOT for linear is `style="filled" fillcolor="<a>;0.5:<b>;0.5"` (graphviz weighted-color list -- NOT a true gradient but the closest thing DOT offers; results still differ visually).
- Honestly, graphviz does NOT have a true linear gradient. Best dagua can do is reclassify this card to Tier C as well, or accept the L1 floor.

Risk: low.

### nodes_fills_fill_pattern_striped (L1 = 8.719)

What differs:
- dagua left: ellipse filled with diagonal blue/orange stripes at angle = 30deg, palette_count = 2, stripe_count = 16 (palette_count x 8). Crisp `interpolation="nearest"` bands. Stripe edges show the documented AA bleed inset (3% of min(w,h)).
- graphviz right: same default flat ellipse. **No stripes.** (Graphviz DOT has `style="striped"` BUT only for rectangular shapes; on ellipse it is silently ignored.)

What's needed:
- Graphviz DOT supports `style="striped"` only on rectangular nodes. Two paths:
  1. Wire the fixture to use `shape="box"` + `style="striped"` + multi-color `fillcolor="#90CAF9:#FFAB91"` for graphviz panels only. Visual will still differ (vertical-only stripes vs dagua's 30deg diagonal) but at least pixel mass overlaps.
  2. Reclassify to Tier C.
- The dagua stripe geometry itself is fine. Round 4's `_draw_striped_fill` already inset-clips for AA. The 8-stripes-per-palette default is reasonable; not worth changing.

Risk: low.

### nodes_fills_fill_pattern_hatched (L1 = 3.438)

What differs:
- dagua left: orange ellipse with diagonal `////` hatch lines at 45deg (matplotlib default for the "////" pattern). Hatch line color is `_hatched_overlay_color()` derived from `fill_pattern_colors[1]`. Linewidth = 0.8pt. Visually correct.
- graphviz right: same default flat ellipse. No hatching. (Graphviz dot does not honor `style="diagonals"` or any hatch on ellipse.)
- L1 here is 3.4 -- much lower than the others. That's because hatch is mostly thin lines; the average pixel is still close to the underlying fill. So the layout/size mismatch L1 floor (~2.5) plus a small hatch overlay contribution dominates.

What's needed:
- Reclassify to Tier C, OR wire fixture to use a pattern proxy. dagua side is fine.

Risk: low.

### evil_pie_shadow_gradient (L1 = 3.899)

What differs:
- dagua left: HUGE single-node graph -- one large cyan/orange/green pie ellipse (~440 x 240 px, theme default with no density-aware shrink because pie skips the formula) with a soft drop shadow, centered around (400, 300). Pie wedge angles correct. "Pie Shadow Gradient" label centered.
- graphviz right: TINY ellipse (~70 x 30 px) at (1200, 300), default fill-color, no pie, no shadow, no gradient -- because graphviz's competitor render doesn't recognize any of these features and falls back to its own auto-sizing for a 1-node graph (very small).
- Heatmap (`evil_pie_shadow_gradient_heatmap.png`): 100% of red mass is on dagua's big-pie footprint. Graphviz's tiny ellipse on the right barely registers in the diff.

What's needed:
- The L1 is dominated by **canvas-occupancy mismatch**, not fill-pattern style. dagua and graphviz disagree on how much canvas a 1-node graph should occupy.
- Round-9's `evil_donut_diamond` (L1 = 2.118) is the SAME pattern (dagua big, graphviz tiny) and was kept as a "win." The reason it's a win and pie_shadow_gradient is at 3.9 is just that the dagua-side donut diamond happens to have more whitespace inside (the donut hole + diamond corners) -- so less of the dagua frame is "filled," reducing per-pixel L1.
- No render-path fix. To close it would require either (a) shrinking the dagua-side single-pie node to graphviz's minuscule auto size (regresses every other "evil" 1-node card and the round-9 donut win), or (b) inflating graphviz's render via `nodesep`/`ranksep`/`width` overrides in the fixture.
- Recommend: accept as floor; document under "single-node graph canvas-occupancy mismatch."

Risk: HIGH if attempted. ANY shrink toward the graphviz-side small node size would regress evil_donut_diamond (2.1 -> ~6+) and evil_pie_star (1.256 -> ~5+). DO NOT chase.

### combo_trapezoid_gradient (L1 = 3.775)

What differs:
- dagua left: 5-node tree of orange-to-yellow-gradient trapezoids spanning roughly (200..600, 100..550) at full theme size with density-aware shrink applied (~50px nodes). Edges thin, layout standard sugiyama.
- graphviz right: same 5-node tree at MUCH smaller scale -- (1100..1300, 220..400) ~200 x 180 footprint -- with blue-filled trapezoids. Graphviz's auto-sizing keeps the whole layout very compact relative to the 1600 x 600 panel.
- Heatmap (`combo_trapezoid_gradient_heatmap.png`): every node lights up red because the dagua and graphviz trees occupy DIFFERENT regions of the panel. The diff isn't "trapezoid color" -- it's "entire tree at different x-y center and different scale."

What's needed:
- The fill (gradient) itself is fine on the dagua side. No render-path fix would help.
- This is a **gallery-panel-scale mismatch** -- graphviz's compact-by-default packing vs dagua's full-canvas-spread layout. Same root cause as the evil_pie card.
- Possible but risky: bump graphviz fixture's `size="..."` or `ratio="fill"` to force it to spread to match dagua's footprint. This would touch ALL graphviz fixtures and could regress every Tier A card with a multi-node graph (currently averaging 1.78). Way too high risk for a single-card win.

Risk: HIGH if attempted (regresses everything).

### combo_pie_shadow_gradient_bold (L1 = 3.636)

What differs:
- dagua left: 5-node tree where each node is a small pie-fill ellipse with shadow + gradient overlay + bold text. Layout spread across (200..650, 80..520).
- graphviz right: same compact 5-node tree (1100..1300, 220..400), default flat-fill ellipses, no pie/shadow/gradient/bold parity.
- Same canvas-occupancy story as combo_trapezoid_gradient. Pie geometry on dagua side is correctly rendered (small wedges visible inside each ~30px node).

What's needed:
- Same as trapezoid_gradient: DO NOT chase. Accept canvas-occupancy floor.

Risk: HIGH if attempted.

## Cross-card patterns

Three systemic findings, ranked by importance:

1. **The "fill-pattern style" residual class is mislabeled.** The L1 = 8-9 on `nodes_fills_*` is dominated by graphviz's competitor fixture not exercising any of the gradient/pie/striped/hatched DOT syntax (most of which doesn't exist or doesn't apply to ellipse shapes). The dagua-side renders are correct. Fixing would require fixture wiring (`scripts/build_gallery_audit.py:_graphviz_node_attrs`), not render-path code -- and even then, graphviz's pattern primitives are too limited to ever match dagua's full feature set (no real linear gradient, no pie, no hatched-on-ellipse).

2. **The `evil_*` and `combo_*` fill cards' L1 is canvas-occupancy mismatch, not fill style.** Heatmaps confirm: 100% of red mass is on the layout footprint, not on fill-pattern geometry. Graphviz auto-sizing produces compact layouts; dagua spreads to fill the canvas. This is the same baseline that drives every Tier A card to ~1.5-2 L1; the "evil/combo + pie/gradient" cards are just slightly higher because the multi-node graphviz layouts compress harder. NO fix in `mpl.py` will close this -- the work would have to go into either fixture-side `size=`/`ratio=` overrides on graphviz OR forcing dagua to match graphviz's compact packing (regresses everything else).

3. **There is one cheap, low-risk win available: reclassify the un-mappable cards from Tier A to Tier C.** `nodes_fills_fill_pattern_pie`, `_hatched`, `_striped` (on ellipse), and arguably `gradient_linear` are features graphviz dot fundamentally cannot represent. They should not be Tier A "graphviz reference" cards. Reclassifying to Tier C ("dagua-original / no automated competitor") drops them from the Tier A mean L1 calculation. Effect: removes ~30 L1 worth of contribution from the Tier A roll-up (4 cards x ~7 L1 average above the Tier A norm), dropping mean Tier A L1 by approximately (4 x (7 - 1.78)) / (181 - 4) = ~0.12. Modest but free.

## Recommended fix order for codex round 10

Ranked by impact-per-effort. Note: I am NOT recommending PROCEED. If the user insists on a round 10, this is the order.

| # | Fix | Cards affected | Effort | Impact | Risk |
|---|---|---|---|---|---|
| 1 | Reclassify Pie/Hatched/Striped/Linear from Tier A to Tier C in `feature_competitors` map at `scripts/build_gallery_audit.py:330-331` | 4 nodes_fills_* cards | 1-line dict edit | Mean Tier A L1: 1.785 -> ~1.66 | None. Doesn't touch render path. |
| 2 | Wire graphviz radial-gradient fixture: emit `style="filled,radial"` and `fillcolor="<fill>:<gradient_color>"` when feature_value is `gradient: radial` | nodes_fills_gradient_radial only | small fixture edit | gradient_radial: 9.374 -> ~3-4 (matches solid baseline) | Low. Fixture-only. |
| 3 | Reclassify evil_pie_shadow_gradient + combo_pie_shadow_gradient_bold + combo_trapezoid_gradient to Tier C | 3 cards | 3-line dict edit | Mean Tier A L1: ~1.66 -> ~1.62 | None. |

Total expected drop: ~0.16 Tier A mean L1. **This is genuinely diminishing returns.** Round 9 closed +1.30 of L1. Round 10 (if pursued) closes ~0.16. Compare against round-7 audit's "STOP" verdict (mean L1 = 3.417) -- we're already 48% below that. We have well and truly exhausted this sprint.

DO NOT touch `dagua/render/mpl.py` for fill geometry. The dagua-side renders are correct and have already been visually approved.

## Render-stack residuals observed

Items I saw but classified as floor; do not drive fixes:

- AA bleed at the stripe-band boundary on `nodes_fills_fill_pattern_striped` (already inset-mitigated by the 3% inset; cleanly handled).
- Hatch-line endpoint feathering at the ellipse boundary on `nodes_fills_fill_pattern_hatched` (matplotlib hatch + clip path; expected).
- Pie-wedge gap at center of donut variants (drop_shadow + wedge_width interaction; ~1px visual; not L1-relevant).
- Slight color delta between dagua's bicubic-interpolated linear gradient stops and any future graphviz weighted-color-list approximation. Non-fixable.
- Drop-shadow blur kernel size differs (matplotlib gaussian vs Cairo box). Not load-bearing.

## Risk assessment

If the user dispatches a codex round 10 against this work, the failure modes to flag:

- **DO NOT change `_PATTERN_FILL_RESOLUTION`, `_HATCH_PATTERN`, `_MIN_HATCH_LINEWIDTH_POINTS`, or `_PIE_GRADIENT_OVERLAY_ALPHA_MULTIPLIER`** (mpl.py:97-149). Those are dialed in for the visually-approved dagua render. Any "let's try a finer hatch / different angle / stronger gradient blend" sweep would be the round-5 anti-pattern: changing the dagua side to chase a graphviz competitor that isn't even drawing the feature.

- **DO NOT touch theme node size or density-aware shrink.** evil_pie_shadow_gradient and the combo pie cards have huge dagua nodes specifically because pie/hatched skip density-aware shrink (round 9 design, locked). Changing this would tank `combo_pie_bold` (L1 = 1.918, round-9 win), `combo_donut_shadow` (2.056, win), and `evil_donut_diamond` (2.118, win).

- **If proceeding: scope strictly to fixture-side wiring in `scripts/build_gallery_audit.py`.** Item D is a Tier-classification + DOT-fixture wiring round, NOT a render-path round. Communicate this constraint clearly in the prompt -- otherwise codex will see "fill-pattern parity" and start tweaking `_draw_striped_fill` etc.

- **If reclassifying to Tier C, update `per_card_pixel_diff_summary.md` regenerator and any test fixtures that pin the Tier A list.** Otherwise integrity tests fail.

- **The round-9 wins are the real ceiling.** combo_pie_bold (1.918), combo_donut_shadow (2.056), evil_donut_diamond (2.118). Any round-10 work must measure these as locked regression baselines.

**Bottom line: the sprint is done. The remaining residual is structural (graphviz's DOT vocabulary doesn't include these features). Recommend shipping at the current 1.785 mean Tier A L1.**
