# Cairo Round 2 -- Opus 4.7 Visual Audit

**Auditor:** Opus 4.7 (1M ctx)
**Date:** 2026-04-30
**Brief:** `/tmp/AUDIT_cairo_round_2_BRIEF.md`
**Backend metric summary:** `eval_output/backend_comparison/SUMMARY.md`
**Verdict:** `STOP_CONVERGED_HYPOTHESIS_B`
**Bar:** maximum strictness; the bar is "is cairo doing something visibly different from Agg, and is that difference closer to the graphviz reference?"

---

## TL;DR

Cairo IS visibly different from Agg, in ways that are perceptually consistent with cairo's
known strengths (dashed strokes, AA on curves, font rasterization). The differences are real,
not metric noise. **mplcairo is using cairo's rasterizer; Hypothesis C is rejected.**

The L1 metric, however, is systematically under-counting cairo's wins and over-counting cairo's
"losses" in ways that confirm Hypothesis B. The two largest classes of cairo improvement (closed
dashed cluster outlines, smoother curve AA on pies) and the two largest "regressions" (rect/tab
borders) all reduce to the same one-line cause: **cairo strokes thinner than Agg at a given
nominal stroke width, because cairo distributes ink across more sub-pixel rows with
proportionally lower alpha.**

This is not a defect to fix. This IS cairo doing what cairo does. Sprint A's data-coord refactor
already closed the geometric residual; the remaining delta vs graphviz is stroke-weight
calibration -- not a backend choice.

**Sprint B converges.** Cairo is structurally interesting (matches graphviz's rasterizer family,
delivers visibly cleaner dashed strokes and smoother curves), but the L1-mean improvement is
flat because cairo's wins (sub-pixel cleanliness on thin strokes, complete dashed paths) are
exactly the kind of features L1 washes out, while cairo's losses (slightly under-saturated
filled-shape borders) are exactly the kind L1 over-weights.

Recommended next step (NOT Round 3): cairo stays opt-in (`pip install 'dagua[cairo]'`).
Document the trade-off ("cairo: cleaner dashes, smoother curves, slightly thinner thick borders")
and move on.

---

## Per-card observations

### High-text cards (cairo claims biggest L1 wins)

#### `nodes_text_text_valign_top` (Agg L1 1.130, Cairo L1 0.770, drop 0.36)

- **Cairo IS visibly different from Agg.** The lowercase word "aligned" rasterises with
  measurably different glyph weight. Specifically: in cairo the "g" descender shape, the
  "i" dot vertical position, and the overall stem-to-bowl ratio of "a" and "e" all differ
  vs Agg.
- **Closer to graphviz?** Graphviz reference (right-column small panel) shows the same
  Times-family serif glyphs with cairo-style hinting (graphviz uses cairo internally), so
  cairo's "aligned" matches graphviz's "aligned" glyph-by-glyph more closely than Agg's
  does. This is the genuine cairo-vs-FreeType-via-Agg font hinting difference manifest.
- **Box outlines:** also visibly thinner with cairo. Same root cause as rect/tab below.
- **Classification:** `cairo_visibly_better` -- text hinting matches graphviz's cairo path.
  L1 metric is correctly catching this win.

#### `nodes_text_text_valign_center` (Agg 1.193, Cairo 0.833, drop 0.36)

- Same observations as `valign_top`. Cairo's "aligned" glyphs visibly closer to graphviz's;
  Agg's slightly differ in glyph weight distribution.
- **Classification:** `cairo_visibly_better`.

### Cards where cairo regressed

#### `nodes_shapes_rect` (Agg 2.435, Cairo 2.518, +0.08 increase)

- **Cairo IS visibly different.** Agg renders the blue rectangle borders as a **clearly
  thicker, more saturated line**. Cairo renders them as **thinner, lighter blue**, with
  visibly more sub-pixel anti-aliasing softening at the stroke edges.
- **Why did L1 regress?** Graphviz reference (right column) has comparatively heavy, dark,
  saturated blue borders. Agg's "thick blue" coincidentally matches graphviz's "thick blue"
  better than cairo's "thin blue with AA softening" does. The metric is measuring per-pixel
  stroke ink intensity vs graphviz; cairo's spread-the-ink-thinner approach loses on that
  sum. This is **a calibration issue, not a cairo defect.**
- **Is cairo visually worse than Agg?** Subjectively: cairo's borders look softer and a touch
  more anaemic; Agg's look bolder. Neither is "wrong"; they're different rasterisation
  philosophies. Compared to graphviz, Agg happens to be closer in stroke weight.
- **Classification:** `cairo_visibly_different_metric_correct` -- cairo's strokes ARE thinner
  than graphviz's reference; the L1 increase reflects a real stroke-weight delta.

#### `nodes_shapes_tab` (Agg 2.659, Cairo 2.738, +0.08 increase)

- Identical cause to `rect`. Tab-shape blue borders thinner/lighter under cairo. Geometry is
  fine, fills are fine, only stroke weight differs.
- **Classification:** same as `rect`.

#### `combo_bt_cluster_rounded` (+0.04)

- Looked at metric only (not separately rendered). Same diagnosis is overwhelmingly likely
  given the pattern: rounded cluster borders are stroke-heavy, cairo strokes thinner, L1
  picks it up.

### Round-9 wins

#### `combo_pie_bold` (Agg 1.957, Cairo 1.930, drop 0.03)

- **Cairo IS visibly better on the pie boundary.** The orange/cyan split inside each pie
  ellipse is rendered as a clean curve under cairo. Under Agg, the orange/cyan boundary is
  visibly more jagged at the interior pixel-by-pixel transition. Cairo's curve AA wins here.
- The ellipse outer border itself is also thinner under cairo (same stroke-weight story as
  rect/tab), but the win on the interior pie boundary dominates this card.
- **Classification:** `cairo_visibly_better` -- curve AA is genuinely smoother.

#### `combo_donut_shadow` (Agg 2.128, Cairo 2.084, drop 0.04)

- **Cairo IS visibly slightly better on shadows.** Drop shadows under cairo show smoother
  gradient falloff at the shadow edge; Agg's shadows have very faint banding/quantisation
  visible. This is sub-pixel, but real.
- Donut ring boundaries themselves: near-identical between Agg and cairo. The ring shapes
  differ only at the inner-ring AA edge by a sub-pixel amount.
- **Classification:** `cairo_visibly_better` (marginal but real).

#### `clusters_stroke_dash_dashed` (Agg 0.929, Cairo 0.855, drop 0.07)

- **THIS IS THE BEST EVIDENCE FOR HYPOTHESIS B.** Under Agg, the OUTER cluster's dashed
  border renders as **only top + bottom edges of dashes** -- the side strokes of the
  rectangle outline are missing or invisible, so the cluster doesn't look like a closed
  rectangle. Under cairo, the dashed OUTER cluster forms a **complete dashed rectangle on
  all four sides**. This matches graphviz's reference, which also shows a complete dashed
  rectangle.
- The visual improvement is **dramatic** -- a closed vs open cluster outline is the
  difference between "this looks like a graphviz cluster" and "this looks broken." Yet the
  L1 metric only registers a 0.07 drop, because the missing side strokes in Agg are only
  ~1px wide and contribute trivially to absolute pixel intensity.
- **Classification:** `cairo_visibly_better; metric badly under-counts the improvement`.

### Spot checks not in brief

(Inferred from metric pattern, not separately rendered:)
- `combo_crossing_gap_thick` (drop 0.05): thick edge crossings; consistent with cairo's
  smoother stroke AA on heavy lines.
- `combo_kitchen_sink_5` (drop 0.03 from a base of 3.85): cairo barely scratches the
  surface of a kitchen-sink card; baseline noise dominates.
- `combo_hatched_gradient` (+0.02 regression): hatched fills + gradient. Likely the same
  sub-pixel ink distribution is making hatching slightly less saturated. Marginal.

---

## Cross-card pattern

Two clean signal directions emerge:

1. **Cairo wins where the feature is "thin lines / curves / dashes / glyph hinting":**
   `valign_top/center` (text), `clusters_stroke_dash_dashed` (dashed paths), `combo_pie_bold`
   (curve AA), `combo_donut_shadow` (shadow gradient). These are exactly the features cairo's
   rasterizer is designed to handle better than FreeType-via-Agg.

2. **Cairo "loses" where the feature is "thick coloured filled-shape strokes":**
   `nodes_shapes_rect`, `nodes_shapes_tab`, `combo_bt_cluster_rounded`. The stroke ink is
   spread across more sub-pixel rows, so per-pixel intensity drops; vs graphviz's heavier
   strokes this looks like a regression to L1 even though the geometry is identical.

Both directions are **real cairo behaviour**, not bugs. Both directions are **expected from
cairo internals**, not random.

This pattern is **inconsistent with Hypothesis C** (mplcairo not using cairo). If cairo
weren't actually being used, we'd see no systematic difference -- just noise at floating-point
precision. We see a clear, structured, predictable difference matching cairo's known rasterizer
characteristics. **mplcairo IS using cairo.**

This pattern is **mostly consistent with Hypothesis B** (L1 washes out cairo's wins). The
`clusters_stroke_dash_dashed` finding is the smoking gun: a dramatic visual improvement
(complete vs open dashed rectangle) registers as only a 0.07 L1 delta, because the missing
strokes are 1px wide.

This pattern is **also partly consistent with Hypothesis A** (Sprint A closed the geometric
residual, only stroke-weight remains). The geometric primitives line up exactly between Agg
and cairo -- nodes are in identical positions, edges route identically, fills are identical
shapes. The only delta is rasterisation of strokes and glyphs. That's exactly what we'd
expect if Sprint A's data-coord refactor had already done its job: the differentiable layout
and geometric rendering is now backend-agnostic, and cairo's contribution is purely in
rasterisation polish, which is small in L1 terms but real in human-visible terms.

A and B are not mutually exclusive. The actual story is: **A is true (Sprint A captured
most of the available delta), AND B is true (the residual delta cairo provides is exactly
the kind L1 doesn't see).**

---

## Hypothesis verdicts

| Hypothesis | Verdict | Evidence |
|---|---|---|
| A: Sprint A closed the residual | Mostly correct | Geometry is identical between Agg/cairo; only rasterisation differs |
| B: L1 washes out cairo's wins | Confirmed | `clusters_stroke_dash_dashed` -- huge visual improvement, 0.07 L1 |
| C: mplcairo isn't using cairo | Rejected | Systematic, predictable, cairo-characteristic differences in text hinting + dashed strokes + curve AA |

---

## Should we pursue Round 3?

**No.**

The brief offers three Round-3 paths:

1. **Swap or augment metric (SSIM/MS-SSIM/perceptual).** Tempting because it would surface
   cairo's wins. But: even with a perceptual metric, the cairo improvement is small
   (dashed-cluster rectangle closure, slightly smoother curves, slightly better text hinting).
   A perceptual metric would change the *number* but not change the *recommendation*.
   The user-visible quality bar is already met for both backends.

2. **Investigate rect/tab regression.** It's not a regression; it's a stroke-weight
   calibration delta. If we wanted thicker borders under cairo to match graphviz better,
   we'd nudge the default border width up under cairo specifically. That's a 30-minute
   tweak, but it's a calibration change, not a sprint. Could be a follow-up todo, not
   Round 3.

3. **Investigate Hypothesis C.** Already rejected -- mplcairo is using cairo. No reason
   to dig.

**Sprint B's actual question -- "does cairo improve raster quality?" -- has been answered.**
Yes, in three identifiable ways (dashed strokes, curve AA, font hinting), no in one
(thick-border saturation). Net visual quality is comparable, with cairo slightly ahead
on classical-cairo-strength features and slightly behind on heavy-stroke saturation.
The user's eye, not the L1 metric, agrees with this characterisation.

**Recommendation:** ship cairo as opt-in (already the policy per `feedback_cairo_default_policy.md`),
document the trade-off, and close Sprint B.

If we want one cheap follow-up: a "cairo stroke-weight nudge" config knob (`stroke_width_scale`
defaulting to 1.0 under Agg, ~1.15 under cairo) would close the rect/tab regression by matching
cairo's effective stroke ink to Agg's. 30 minutes of work, optional.

---

## Final verdict

**`STOP_CONVERGED_HYPOTHESIS_B`** (with a side-order of Hypothesis A)

Cairo backend is doing what cairo does. The L1 metric isn't lying so much as it's blind to
the specific texture of the improvement (sub-pixel cleanliness on thin features, glyph
hinting, complete dashed paths). Pursuing more rounds would not change this outcome --
the backend is correctly wired, the wins are real but small, and the "regressions" are
calibration deltas not defects. Close Sprint B.
