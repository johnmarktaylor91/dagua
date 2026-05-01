# Sprint C Round 2 Audit — Opus 4.7

**Date:** 2026-04-30
**Audited commit:** d13cf02 (`fit_to_canvas` render mode + gallery_audit hookup)
**Verdict:** **CONTINUE_ROUND_3** — root cause is layout fixture vertical gap, not the canvas-fit math. Path D (fixture y-gap reduction) is recommended; Path A (lower margin) is insufficient (~11% gain available; we need ~98%).

---

## TL;DR

The `fit_to_canvas` math in `dagua/render/mpl.py:1392-1447` is **mathematically correct**. The reason dagua box3d still renders at ~110px instead of ~250px in `box3d_vs_graphviz.png` is **not** a sign error or a too-generous margin — it is that the **layout fixture for the node-shapes comparison panel uses `PAIR_DEFAULT_GAP = 260.0`** in `scripts/build_gallery_audit.py:111`, which produces a layout extent of ~96 × 304 data-units. With panel content area 716 × 474 px and uniform-aspect canvas-fit, the binding constraint is layout HEIGHT — yielding scale ≈ 1.40 px/data-unit. Graphviz's panel renders at 200 DPI with its own (much tighter) rank-separation, giving ~2.78 px/pt — exactly 2× more.

**The simplest fix that gets dagua box3d to ≥90% of graphviz size:** drop `PAIR_DEFAULT_GAP` from 260 → ~110 (or fixture-specific override) for `nodes/shapes` comparison cards.

---

## Part 1: Implementation review

### `_coerce_canvas_fit_margin` (lines 1359-1389)
Correct. Maps `True` → 0.05, `False` → `None`, validates explicit floats in `[0.0, 0.5)`. No bug.

### `_canvas_fit_bounds` (lines 1392-1447)
The math:
```
content_width  = max(x_max - x_min, 1.0)
content_height = max(y_max - y_min, 1.0)
inner_fraction = 1 - 2*margin_fraction       # 0.9 with default margin
target_aspect  = figsize[0] / figsize[1]
desired_width  = content_width  / inner_fraction
desired_height = content_height / inner_fraction
if desired_width/desired_height < target_aspect:
    desired_width = desired_height * target_aspect    # widen to fit aspect
else:
    desired_height = desired_width / target_aspect    # tallen to fit aspect
```
This is **correct**:
- Inflates each data-axis by `1/inner_fraction` (so margin_fraction of axis is empty padding).
- Then expands the *shorter* axis to match figure aspect (preserves uniform px-per-data-unit, no stretching).
- Returns axes bounds centered on content center.

No sign flip. No one-axis-only bug. The "tighten axis limits → bigger nodes" intuition is preserved: smaller `desired_width × desired_height` ⇒ larger `figsize/desired` ⇒ more px-per-data-unit ⇒ bigger nodes.

### Application sites in `render()` (lines 1700-1850)
- Line 1710-1718: First `_canvas_fit_bounds` call after computing layout extent + cluster expansion + self-loop padding (lines 1570-1689). Inputs include `gs.margin = 4.0` already added.
- Line 1720-1727: Figure created with `subplots_adjust(0,0,1,1)` so the axes fill the entire figure rectangle (no white border eating canvas). Correct.
- Line 1730-1734: `ax.set_aspect("auto")` when fitting (so x and y can have independent display scales). The `_canvas_fit_bounds` math compensates by widening one axis to match `target_aspect`, which keeps `px_per_data_x == px_per_data_y`. This is the pattern that makes the math consistent.
- Line 1737: `_expand_axes_for_clusters(ax, ...)` runs even in fit mode. This **mutates `ax.xlim/ylim`** if cluster headers need extra space.
- Line 1738-1750: Second `_canvas_fit_bounds` call re-reads the (possibly expanded) limits and re-fits. Correct (handles cluster expansion).

No bugs. Math is sound.

---

## Part 2: Pixel probe of `box3d_vs_graphviz.png`

Image: 1600 × 600. Two side-by-side 800 × 600 panels, content area inset (42, 84, 42, 42) → 716 × 474 px content per panel.

### Detected geometry (vertical edges + horizontal edges)
Panel composition: each side has TWO box3d nodes stacked vertically with arrow between (PAIR fixture).

**LEFT panel — DAGUA box3d:**
- Top box3d:    rows 133–180 (h ≈ 47 px), cols 343–456 (w ≈ 113 px)
- Bottom box3d: rows 461–508 (h ≈ 47 px), cols 346–454 (w ≈ 108 px)
- Edge connector: vertical at cols 398–401, rows 178–464

**RIGHT panel — GRAPHVIZ box3d:**
- Top box3d:    rows 169–273 (h ≈ 104 px), cols 323–476 (w ≈ 153 px)
- Bottom box3d: rows 369–472 (h ≈ 104 px), cols 323–476 (w ≈ 153 px)

### Measured ratios
| Dim | Dagua | Graphviz | Ratio |
|-----|-------|----------|-------|
| Width | 113 px | 153 px | **74%** |
| Height | 47 px | 104 px | **45%** |
| Area | 5,311 px² | 15,912 px² | **33%** |

The brief's "44%" matches the height ratio. The width ratio is slightly better (74%). The **height** is the binding visual problem. The aspect ratio of dagua box3d (113 × 47 = 2.4:1) is much wider than graphviz (153 × 104 = 1.47:1) — which is the visible "squashed" appearance.

### Round 1 vs Round 2 delta
Brief reported Round 1: ~75 px width. Round 2: ~110 px width (matches my 113 px measurement). That's a +47% increase. The math model below explains this exactly.

---

## Part 3: Diagnosis — math model vs observed

I instrumented the box3d card's actual graph + positions:
```
positions: [[0, 130], [0, -130]]
node_sizes: [[87.8, 36.0], [84.1, 36.0]]
gs.margin: 4.0
layout extent (after margin): w=95.8 data-units, h=304.0 data-units
target_aspect (panel content): 716/474 = 1.5105
```

Applying `_canvas_fit_bounds(margin_fraction=0.05)`:
```
inner = 0.9
desired_w = 95.8 / 0.9 = 106.4
desired_h = 304.0 / 0.9 = 337.8
desired_w/desired_h = 0.315 < target_aspect 1.5105
=> widen w to desired_h * target_aspect = 510.2
final desired axes range: 510.2 × 337.8
```

Pixel-per-data-unit scale:
```
px/data_x = 716 / 510.2 = 1.4034
px/data_y = 474 / 337.8 = 1.4034   (uniform — by design)
```

Predicted node pixel size: **87.8 × 1.40 = 123 px wide × 36 × 1.40 = 50 px tall.**
Observed: **113 × 47.** Match within 8% (residual from cluster-expansion second-pass and 0.5 px line-width effects).

**The math is correct. The issue is upstream: the layout extent height is far too large relative to what graphviz produces.**

### Why graphviz appears bigger
Graphviz `dot -Gdpi=200`:
- Node height = 0.5 in × 72 pt = 36 pt = 36 × (200/72) = **100 px** ✓ (observed 104)
- Node width (auto-fit to label "box3d" + padding) ≈ 0.75 in = **150 px** ✓ (observed 153)
- Rank separation = 0.5 in = 100 px between nodes (centers ~200 px apart)

Graphviz's *layout* is in pt-space, where a 36 pt-tall node + 36 pt rank gap = 72 pt center-to-center. Then the SVG → PNG render at DPI=200 converts at 1pt → 2.78 px.

Dagua's *layout* is in data-unit space, where the fixture explicitly puts nodes at y = ±130 (gap = 260 data-units, surface-to-surface 224). With `node_sizes[1] = 36` data-units, the rank gap is 224/36 = **6.2× wider than graphviz's natural 1× rank gap.**

When canvas-fit then maps that to 474 px of vertical panel space, the scale is throttled to 1.40 px/data-unit. The node — only 36 data-units tall — gets only 50 px of vertical canvas. It's not the math; **it's that dagua's fixture wasted ~80% of the layout-height on rank gap.**

### Verdict on the three hypotheses
| Hypothesis | Status |
|-----------|--------|
| **A) Margin too generous (0.05 → 0.01)** | **PARTIAL.** Reducing margin → 0 raises scale from 1.40 to 1.56 px/data (~+11%). Insufficient — would push dagua node from 47 px tall to ~52 px tall. Still ~50% of graphviz. |
| **B) Scale calc has a math/sign error** | **NO BUG.** Verified by manual computation — math matches observed pixels. |
| **C) Layout-extent computation is wrong** | **PARTIAL — not "wrong" but "honest".** The 304-data-unit height accurately reflects fixture node positions plus margins; nothing is over-counted. The fixture itself is generous. |
| **D (NEW) — fixture rank-gap too wide** | **YES — this is the real cause.** `PAIR_DEFAULT_GAP = 260` produces 6.2× excess vertical empty space relative to graphviz's natural 36-pt rank gap. |

---

## Part 4: Round 3 recommendation

### Fix path: Reduce `PAIR_DEFAULT_GAP` for shape-comparison fixtures

Modeled outcomes (margin=0.05, panel 716×474, node 88×36):

| `node_gap` | layout h | scale | node px (W × H) | vs graphviz (153 × 104) |
|-----------:|---------:|------:|----------------:|:-----------------------|
| 260 (current) | 304 | 1.40 | 123 × 51   | width 81%, height 49%  |
| 110 | 154 | 2.77 | 243 × 100  | width 159% (overshoot), height 96% |
|  90 | 134 | 3.18 | 280 × 115  | overshoot both |
|  72 | 116 | 3.68 | 323 × 132  | severe overshoot |

The aspect-mismatch widens `desired_w` to fit 1.51 panel aspect, so width gets oversized as gap shrinks. To target ≥90% on both width AND height while not overshooting:

**Optimal: `node_gap ≈ 110`** for `nodes/shapes` pair fixtures gives:
- height: 100 px ≈ graphviz 104 → **96% match (target ≥ 90% ✓)**
- width: 243 px vs graphviz 153 → 159% (overshoot)

Width overshoot is because the layout WIDTH (96 data-units) is much narrower than the panel aspect demands. The canvas-fit then **widens** the axis range, cutting the px/data scale less aggressively in width than in height. To prevent width overshoot we need **either**:
- (i) a horizontal pair layout (gap on x-axis) — but this is a vertical-pair fixture by design, OR
- (ii) center the layout *without aspect-widening* — but that would produce non-uniform px/data, violating the diff-rendering invariant, OR
- (iii) **increase content_width to match the panel aspect** — i.e., add horizontal "ghost" padding to layout extent so it has aspect 1.51 natively.

### Recommended Round 3 implementation (Path D + iii)

**Spec for codex:**

1. **Primary change** — `scripts/build_gallery_audit.py:111`:
   - Add a fixture-specific gap override (don't change `PAIR_DEFAULT_GAP` globally; that affects many other panels).
   - Introduce `PAIR_SHAPE_COMPARISON_GAP = 110.0` and use it in `_pair_positions()` when the card is in `NODE_SHAPE_PARITY_CARD_IDS`.
   - Apply via `_apply_reference_card_tweaks` (line 1825) by re-emitting positions when `item.card_id in NODE_SHAPE_PARITY_CARD_IDS`.

2. **Aspect-padding in `_canvas_fit_bounds`** — `dagua/render/mpl.py:1392-1447`:
   - Current code expands the **smaller** dimension to match `target_aspect`, which preserves uniform scale but **loses density** when content aspect differs from panel aspect.
   - Add a `prefer_density: bool = True` parameter (or new helper) that, when set, picks the AXIS that gives **larger px/data scale**, then pads the OTHER axis with empty space (as the current code does, but choosing the higher-scale axis).
   - Concretely: instead of always widening the shorter axis, the code already DOES choose the shorter axis correctly — the issue is that with `inner_fraction=0.9`, the inflation is applied symmetrically. **Lower the default `margin_fraction` from 0.05 to 0.02** (`_coerce_canvas_fit_margin` line 1383). This alone gains ~6% scale (1.40 → 1.49). Combine with fixture fix.

3. **Tier A L1 metric verification:**
   - After both changes, re-run `dagua benchmark` and check Tier A L1 mean. Round 1→2 went 1.233 → 1.233 (no movement) because the visual change was small. Round 3 with proper rank-gap fix should drop Tier A L1 substantially since the comparison panels currently penalize size mismatches in `cards/comparisons/nodes/shapes/*`.

4. **Cross-check residual cards:**
   - Beyond box3d, the same gap issue likely affects ALL `nodes/shapes` comparison cards (rect, ellipse, hexagon, etc.). Verify after the gap change that other shape comparison cards also improved (not just box3d).
   - Pair cards in `EDGE_PAIR_PARITY_CARD_IDS` use `PAIR_ARROW_GAP = 130.0`; check whether those panels also need adjustment.

### Specific code locations for Round 3 spec
- `scripts/build_gallery_audit.py:111` — add `PAIR_SHAPE_COMPARISON_GAP = 110.0`
- `scripts/build_gallery_audit.py:1854-1861` — extend the `NODE_SHAPE_PARITY_CARD_IDS` block in `_apply_reference_card_tweaks` to override positions via `_pair_positions(node_gap=PAIR_SHAPE_COMPARISON_GAP)`.
- `dagua/render/mpl.py:1383` — change default margin from `0.05` → `0.02` (small, but stacks with fixture fix).
- Round 3 test: re-run `python scripts/build_gallery_audit.py` and pixel-probe at least 3 shape cards (box3d, hexagon, cylinder). Validate dagua node-height ≥ 90% of graphviz node-height for each.

### Why NOT Path A alone (lower margin)
- Lowering margin from 0.05 → 0 inflates scale from 1.40 → 1.56 (+11%).
- Node grows from 47 px tall → 52 px tall.
- Still 50% of graphviz. **Path A alone fails the ≥90% goal.**

### Why NOT a math fix
- Verified math is correct. There is no Path B fix.

### Why NOT Path C alone (broader extent computation)
- The extent computation already correctly reports 96 × 304 from positions + margins. Trimming it further would be incorrect — it's a faithful bbox.

---

## Recommended Round 3 work order

1. Apply fixture fix (`PAIR_SHAPE_COMPARISON_GAP = 110`) for shape-parity cards.
2. Lower default `fit_to_canvas` margin from 0.05 → 0.02.
3. Re-render `box3d_vs_graphviz.png`. Pixel-probe to confirm node H ≥ 90 px (≥ 87% of graphviz 104).
4. Spot-check 4 other shape comparison cards (rect, ellipse, hexagon, cylinder).
5. Re-run benchmark, compare Tier A L1 mean.

If after these changes width overshoot is severe (predicted 243 vs graphviz 153), add a fourth step:

6. Cap node-pixel-width by adding aspect-padding TO content_width (not by changing scale): if the dagua content aspect is much narrower than panel aspect, pad `x_min/x_max` symmetrically with empty space *before* `_canvas_fit_bounds`. This keeps the scale capped to the height-binding scale and produces empty horizontal space rather than overshoot. Implement as a helper on top of `_canvas_fit_bounds` keyed off `content_aspect / target_aspect` ratio.

---

## Confidence

- Pixel measurements: **HIGH** (direct PIL/numpy probe of saved PNG, multiple methods cross-checked).
- Math diagnosis: **HIGH** (manual calc replicates observed pixels within 8%).
- Fix prediction: **HIGH** for height (deterministic linear scaling), **MEDIUM** for width (depends on aspect-padding behavior; may need step 6).
- Cross-fixture impact: **MEDIUM** — only verified for box3d. Other shape-cards likely behave identically but should be spot-checked.
