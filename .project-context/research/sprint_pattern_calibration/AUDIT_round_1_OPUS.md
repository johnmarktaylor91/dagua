# Sprint E Round 1 -- Opus 4.7 Visual Audit

## Verdict

**`CONTINUE_ROUND_2_FIXABLE` -- but not for the reason Sprint D suspected.**

The L1-blind class identified by SSIM_loss is *not* primarily a dash/dot/italic
rendering defect. It is a **layout-scale mismatch + thin-edge vanishing** problem
for simple 2-node fixtures, which gets perceptually amplified by dashed/dotted
patterns. Italic rendering is actually a *graphviz* limitation (graphviz dot
silently drops italic styling on default sans-serif fonts), and dagua is
correctly rendering italic. The "italic defect" is a parity question, not a
dagua bug.

The real, fixable defects are:

1. **Edge stroke + arrowhead vanish on long edges** in 2-node `edges_styles_*`
   fixtures (most visible on `edges_styles_style_dashed`, also affects `solid` and
   `dotted` and the various `width_*` cards).
2. **Layout scale mismatch** between dagua and the GRAPHVIZ_STRICT_THEME
   reference: same fixture, same theme, but dagua spreads Source/Target
   ~3x farther apart than the graphviz competitor's reference rasterization.
3. **Dotted edge cadence too tight at thin stroke width** -- dots blur into a
   continuous gray line at ~0.25 line width, defeating the purpose of "dotted."

## Part 1 -- Visual inspection (per card)

### `edges_styles_style_dotted` (SSIM_loss 0.024, rank gap 43)

- Dagua: Source at (~400,175), Target at (~400,420). Edge is rendered as a
  continuous-looking thin gray line with no visible dot cadence at viewing
  size. Arrowhead is missing or invisible.
- Graphviz: Source at (~400,270), Target at (~400,330). Tight pair, clearly
  dotted edge with arrowhead.
- Defect class: **dot cadence too dense + node separation 3x too large**.
  At width=0.25 the on-length is `0.15 * 0.25 = 0.0375` data units, dots blur to
  pixels at gallery zoom. Compounded by the edge being ~4x longer than graphviz.

### `edges_styles_style_dashed` (SSIM_loss 0.023, rank gap 40)

- Dagua: Source/Target same wide separation as dotted. Edge is **completely
  invisible** -- only a single dash near top of Target survives. No arrowhead.
- Graphviz: tight pair with clean dashed edge with arrowhead.
- Defect class: **vanishing edge + arrowhead** on long edges. The dashes appear
  to be placed correctly cadence-wise (4.0 on, 2.75 off at width=0.25 = ~1.0
  on, ~0.69 off in data units) but the rendered output shows essentially no ink.
  Either the dashes are being placed beyond a clipping region, or the linewidth
  in rendered points is below 0.5pt and Agg drops sub-pixel strokes.

### `combo_parallelogram_dotted` (SSIM_loss 0.065, rank gap 59)

- Dagua: 5-node tree (Ingest -> Validate, Review -> Approve, Ship), nodes
  scaled tiny (~50px wide) and spread enormously (canvas ~700px tall).
  Parallelogram borders are drawn but text labels are illegible (8-10px).
  Dotted edges visible but thin.
- Graphviz: same 5-node tree at sane scale (canvas ~150px tall), parallelograms
  ~100px wide, text legible, dotted edges with arrowheads visible.
- Defect class: **layout scale mismatch** dominates here. Dotted rendering on
  dagua actually looks fine at this zoom; the scale mismatch is the SSIM driver.

### `combo_star_dotted` (SSIM_loss 0.065, rank gap 56)

- Dagua: 5-star tree at 3x graphviz's scale. Star nodes are tiny line drawings
  (no fill). Edges between stars are SOLID, not dotted -- the dotted attribute
  appears to have been lost or overridden by a solid edge style on the star
  combo fixture.
- Graphviz: tight 5-star tree, clean strokes, edges appear solid here too
  (graphviz may also be losing the dotted attribute for edges between
  star-shape nodes in this fixture).
- Defect class: **layout scale mismatch + possible style attribute loss**.
  Both renderers show solid edges, so the SSIM gap is dominated by scale.

### `combo_dashed_diamond_opacity` (SSIM_loss 0.061, rank gap 43)

- Dagua: 5-diamond tree at 2.5x graphviz's scale. Diamond borders show faint
  dashed pattern (correct). Edges drawn solid + heavy black with arrowheads.
- Graphviz: tight 5-diamond tree, dashed diamond borders, solid edges.
- Defect class: **layout scale mismatch** with the dashed-pattern rendering on
  borders looking correct in both. Edges are solid in both.

### `combo_diamond_dashed_opacity_italic` (SSIM_loss 0.060, rank gap 42)

- Same as `combo_dashed_diamond_opacity` plus italic labels. Dagua text is
  clearly italic; graphviz reference text is also italic-styled here (this
  fixture passes the italic through, perhaps because the combo path differs
  from the simple `nodes_text_font_style_italic` path).
- Defect class: **layout scale mismatch**, italic is fine.

### `nodes_text_font_style_italic` (SSIM_loss 0.023, rank gap 49)

- Dagua: clearly italic Source / Target with visible slant.
- Graphviz: UPRIGHT Source / Target. **Graphviz's competitor rendering ignores
  the italic style attribute** for the simple ellipse default font.
- Defect class: **graphviz-side limitation, not a dagua defect**. The
  perceptual divergence is real but the cause is "graphviz drops italic on
  Times sans-serif default" -- a known graphviz behavior. Dagua is rendering
  correctly per the requested style.

## Part 2 -- Defect classes

### Class A: edge + arrowhead vanish on long edges (FIXABLE)

The 2-node `edges_styles_*` fixture has a node separation that yields edges
~250px long in screen space, and at GRAPHVIZ_STRICT_THEME's thin stroke width,
dagua's edge body and/or arrowhead drop below the visibility floor.

Severity: **HIGH** (causes visible edge loss in default-themed renders).

Likely causes:
- Stroke linewidth in points falls below `_MIN_VISIBLE_STROKE_POINTS` floor and
  the floor isn't being applied to dashed/dotted body strokes.
- Arrowhead is being placed on the curve at a parameter that puts it inside the
  Target node bbox, then clipped away.
- For dashed: dashes start at a phase that places no on-segment near the visible
  portion of the edge body.

### Class B: layout scale mismatch on simple 2-node and 5-node trees (FIXABLE
but probably NOT in scope here)

Same fixture, same theme, dagua's pipeline lays out Source/Target at ~3x the
node separation that the graphviz competitor lays out. This is a layout-engine
concern (algo_fidelity territory) and is locked under the brief's guardrails.

Severity: **HIGH** for SSIM but the brief explicitly forbids touching algo
territory, so this stays parked.

### Class C: dotted edge cadence too dense at thin widths (FIXABLE, low risk)

`DOTTED_ON_RATIO = 0.15` * scaled_width = 0.0375 data units at width 0.25.
On a 250px edge that's roughly 1px on / 12px off. With round caps eating one
linewidth per side, the on-segment is dominated by cap circles which blur into
a continuous line at gallery zoom.

Severity: **MEDIUM** (cadence visible only at near-pixel resolution; not the
top SSIM driver).

### Class D: italic parity is a policy choice, not a defect

Graphviz dot drops italic on its default font; dagua honors the style. The
"defect" only exists in SSIM space because of the divergent rendering. Two
options:

- **Option D1: keep dagua honoring italic (current).** Italic text is visually
  correct, graphviz parity is sacrificed.
- **Option D2: match graphviz behavior** under GRAPHVIZ_STRICT_THEME by
  ignoring italic style requests. This is bizarre behavior for a graphviz
  drop-in but technically improves SSIM_loss.

Recommendation: **Option D1, keep current**. Italic rendering is working as
documented. SSIM_loss difference is principled.

## Part 3 -- Round 2 fix path

### Fix 1: floor edge linewidth for visible strokes (Class A)

File: `dagua/render/edges/dashes.py` (~line 30) and the edge body assembly
path (look for `_MIN_VISIBLE_STROKE_POINTS` enforcement).

Goal: ensure the dashed/dotted edge BODY stroke applies the same minimum
visible linewidth floor as solid edges. The current floor `_MIN_VISIBLE_STROKE_POINTS`
is one of the locked constants per the brief, so do NOT change its value -- but
verify it is actually being applied to the dashed/dotted body. Likely the
solid body uses one path and dashed/dotted uses another that bypasses the floor.

Concrete check:
```bash
grep -rn "_MIN_VISIBLE_STROKE_POINTS" dagua/render/
```
Walk the dashed-body assembly path; assert the same floor is enforced before
matplotlib `LineCollection`/`PathCollection` draws.

### Fix 2: arrowhead placement on short visible spans (Class A)

File: `dagua/render/edges/arrowheads.py`.

Goal: when the edge body terminates inside the Target node bbox (which the
dashed pattern can do at the wrong phase), the arrowhead placement falls
inside-clip and disappears. Either:

- Solve arrowhead position from the OUTSIDE of the Target intersection,
  independent of dash phase.
- If using the last dash segment endpoint, fall back to the analytic curve-
  bbox intersection for arrowhead placement.

This is a render-path level fix (arrowhead geometry only). Does NOT touch
layout, themes, or locked constants.

### Fix 3 (optional, low-priority): dotted cadence visibility floor (Class C)

File: `dagua/render/edges/dashes.py:24-27`.

Constants `DOTTED_ON_RATIO=0.15` and `DOTTED_OFF_RATIO=1.8` are NOT in the
locked-constants list (they appear tunable per their docstrings). Adding a
visibility floor in points (e.g. ensure rendered dot diameter >= 1.5 device
pixels) would make the dot cadence read as discrete dots.

**FLAG**: this is a tunable, not a locked constant. Recommend Round 2 leaves
this alone if Fix 1 + Fix 2 close the SSIM gap, since the bigger wins are
upstream. If Round 2 still shows mediocre SSIM after Fix 1+2, revisit Fix 3.

### NOT recommended

- **DO NOT touch layout scale (Class B).** The brief explicitly forbids
  touching algo_fidelity territory. The 2-node fixture's layout placement is
  a layout-engine choice.
- **DO NOT match graphviz's italic-drop behavior (Class D).** Italic rendering
  is documented and correct.
- **DO NOT touch any GRAPHVIZ_STRICT_THEME numerics.**
- **DO NOT touch `_MIN_VISIBLE_STROKE_POINTS` value.** Only verify enforcement.

## Part 4 -- Strictness check

Maximum-strictness review applied:

- Several of the cards I inspected (combo_*) are dominated by layout-scale
  mismatch which is OUT OF SCOPE per the brief's guardrails.
- `edges_styles_style_dashed` and `_dotted` show real, visible defects on the
  dagua side that a graphviz drop-in user would notice (missing arrowhead,
  invisible edge). These ARE in scope.
- The italic divergence is a graphviz-side limitation; pursuing parity by
  intentionally degrading dagua's italic rendering is wrong.

After accounting for guardrails, the *fixable* perceptual gap is:
- Edge body + arrow visibility on long thin-stroke edges -> Fix 1, 2.
- Possibly dotted cadence at thin widths -> Fix 3 (defer).

Predicted SSIM_loss impact: closing Fix 1 + Fix 2 should reduce SSIM_loss for
the two `edges_styles_style_*` cards from ~0.024 toward ~0.015 (matching the
solid edge baseline). The combo_* cards' SSIM_loss is dominated by Class B
(layout scale) and Round 2 will NOT meaningfully shrink them under the
guardrails. Sprint E's SECONDARY criterion ("mean L1-blind class SSIM_loss
toward 0.03") is unlikely to be met without touching layout, so set
expectation accordingly.

## Recommended Round 2 dispatch

Single codex round, medium effort:

1. Walk the dashed/dotted edge body rendering path in `dagua/render/edges/`
   and verify `_MIN_VISIBLE_STROKE_POINTS` is enforced before matplotlib draws
   the body. If not, plumb it through (Fix 1).
2. In `dagua/render/edges/arrowheads.py`, decouple arrowhead placement from
   dash phase -- always place at the analytic edge-vs-Target intersection
   (Fix 2).
3. Add unit tests for `edges_styles_style_dashed` and `_dotted` that assert
   the rendered PNG has visible non-white pixels along the expected edge
   centerline AND in the arrowhead region.
4. Re-run the per_card_pixel_diff harness for the 7 cards in this audit;
   record SSIM_loss before/after.

If Round 2 closes the two `edges_styles_style_*` cards but the four `combo_*`
cards barely move, that is the **expected and acceptable** outcome under the
guardrails. Mark Sprint E DONE with the layout-scale residual documented as
out-of-scope-for-pattern-sprint and queued for an algo_fidelity sprint if
ever revisited.

## Files referenced

- `/home/jtaylor/projects/dagua/dagua/render/edges/dashes.py`
- `/home/jtaylor/projects/dagua/dagua/render/edges/arrowheads.py`
- `/home/jtaylor/projects/dagua/dagua/render/edges/geometry.py`
- `/home/jtaylor/projects/dagua/eval_output/gallery_audit/per_card_pixel_diff/comparisons/`
- `/home/jtaylor/projects/dagua/eval_output/gallery_audit/per_card_pixel_diff/dagua/`
- `/home/jtaylor/projects/dagua/eval_output/gallery_audit/per_card_pixel_diff/competitors/graphviz/`
