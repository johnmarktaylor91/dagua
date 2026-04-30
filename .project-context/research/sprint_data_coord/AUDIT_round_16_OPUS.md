# Round 16 Final Convergence Audit -- Opus 4.7 (Sprint A)

**Date**: 2026-04-30
**Auditor**: Opus 4.7, maximum strictness
**Subject**: dagua data-coordinate-everything sprint, post round-15 (commit 042a73d)
**Verdict**: **`STOP_CONVERGED`**

## Executive summary

Sprint A is genuinely done. Every `linewidth=` and `fontsize=`/`font_size=` site
in `dagua/render/` has been classified, and zero unaccounted leakages remain.
All round-15 fixes landed correctly: text outline + bold emphasis use
`_glyph_stroke_ribbon_paths` data-coord ribbons, port indicators are pure
Path A (data-coord with `display_scale`), and `SCALING.md` faithfully encodes
the 2026-03-23 directive. All 6 regression tests pass; all 6 visual gates I
spot-checked preserve their intended outcomes (round-9, -11, -12, -14, -15
wins survive). The cairo sprint can begin.

## Part 1 -- Round-15 fix verification

### 1a. Text glyph outline stroke (`dagua/render/text/collection.py:633-655`)

Verified data-coord ribbon construction:

- `outline_width = _outline_linewidth(spec) * safe_scale` (data units)
- `_glyph_stroke_ribbon_paths(_segment_path(...), outline_width)` builds a
  filled ribbon polygon enclosing the glyph stroke
- `PathPatch(outline_path, facecolor=spec.outline_color, edgecolor="none",
  linewidth=0.0, ...)` -- zero stroke, fill-only

This is the canonical post-round-13 pattern. Round 15 fix landed correctly.

### 1b. Bold emphasis stroke (`dagua/render/text/collection.py:661-679`)

Verified identical pattern:

- `emphasis_width = _bold_emphasis_linewidth(spec) * safe_scale`
- `_glyph_stroke_ribbon_paths(_segment_path(...), emphasis_width)`
- `PathPatch(..., edgecolor="none", linewidth=0.0, ...)`

Round 15 fix landed correctly.

### 1c. Port indicator markers (`dagua/render/mpl.py:8395-8416`)

**Path A taken** (preferred). The port indicator now constructs filled
data-coordinate paths (no `Line2D` markers, no display-point `markersize`):

- `outer_radius = size_points * display_scale * 0.5` -- explicit point->data
  conversion at the boundary
- `border_width = min(_PORT_INDICATOR_BORDER_WIDTH_POINTS * display_scale,
  outer_radius * 0.45)` -- locked-constant border width converted via
  `display_scale`
- `_port_indicator_path(center, radius, indicator)` returns a closed
  matplotlib `Path` in data coordinates (verified in helper docstring
  starting at line 8419: "Return a filled data-coordinate path for a port
  indicator glyph.")
- Both `outline_patch` and `fill_patch` are `PathPatch` with
  `edgecolor="none", linewidth=0.0`

Path A is the strictly preferable outcome. No principled-residual carve-out
was needed. This is exactly what `_compute_display_scale(ax)` is for.

### 1d. `dagua/render/SCALING.md`

Read end-to-end (85 lines). Verifies the post-round-13/15 philosophy:

| Required content | Found |
|---|---|
| 2026-03-23 directive (data coords by default; display-points opt-in) | Yes (lines 5-7) |
| Differentiable-layout / optimizer-manifold structural argument | Yes (lines 9-14) |
| `_compute_display_scale(ax)` pattern documented with code example | Yes (lines 36-49) |
| `linewidth=0.0` + `edgecolor="none"` ribbon idiom shown | Yes (lines 46-49) |
| Minimum-width clamp pattern (`_MIN_VISIBLE_STROKE_POINTS * display_scale`) | Yes (lines 59-66) |
| Two legitimate display-point categories distinguished (user overrides vs. principled internal residuals) | Yes (lines 72-84) |
| Concise (~50-100 lines) | Yes (85 lines) |

The file is well-targeted and grep-able. Round 15 doc rewrite landed correctly.

## Part 2 -- Final `linewidth=` / `fontsize=` / `font_size=` classification sweep

`grep -rnE "linewidth=|fontsize=|font_size=" dagua/render/` returned the
matches below. Every match falls into a principled category; **zero
leakages remain.**

### `dagua/render/mpl.py`

| Lines | Pattern | Classification |
|---|---|---|
| 1685, 7976, 8082, 8671, 8862, 8934, 9179 | `font_size=_effective_font_size_points(font_size_data, display_scale)` | OK -- helper input flowing into `_*_font_size_data` data-coord conversion |
| 2002, 2013, 2023, 2048, 2065, 2076, 2086, 2096, 2106, 2116, 2135, 2153, 2187, 2204 | `linewidth=linewidth` inside `_build_node_patch` | OK -- helper parameter, dead in production (only `tests/`, `scripts/generate_node_border_comparisons.py` call it). Already documented in round 14/15 audits. |
| 2366, 2381, 2741, 2769, 2792, 4982, 5080, 5091, 6169, 6905, 7026, 7272, 8184, 8230, 8249, 8281, 8326, 8351, 8403, 8410 | `linewidth=0.0` | OK -- canonical zero-stroke fill-only |
| 2777 | `linewidth=_MIN_HATCH_LINEWIDTH_POINTS` | OK -- locked constant (hatch rasterization floor) |
| 6583 | `linewidth=_CROSSING_BRIDGE_STROKE_WIDTH_POINTS` | OK -- locked constant; the bridge stroke is a hairline knockout border whose visual stability is itself the design intent |
| 7753, 7793 | `label_font_size=float(style.label_font_size)` | OK -- assignment to `DaguaEdge` constructor field; downstream conversion via `_compute_display_scale` at render time |
| 3764 | `font_size=font_size_data` | OK -- already in data units |
| 7989 | `min_font_size=style.min_font_size` | OK -- helper input forwarded to data-coord layout (consumed by `_layout_*_text` with `min_size_data = min_font_size * safe_scale` at collection.py:554) |

### `dagua/render/text/collection.py`

| Lines | Pattern | Classification |
|---|---|---|
| 321 | `fontsize=font_size_pts` | OK -- principled residual. Inside `_matplotlib_text_background_path`, used solely on an `alpha=0.0` invisible probe text whose only purpose is `probe.get_window_extent(renderer)` to read the rasterized bbox, which is then immediately inverted via `ax.transData.inverted()` (lines 339-...) back to data coordinates. This is the "font rendering inverse boundary" case explicitly named in `SCALING.md` lines 54-57. |
| 557, 569, 584 | `font_size=size_data`, etc. | OK -- already data-unit sizes flowing into glyph layout |
| 623, 648, 672, 684, 714, 736 | `linewidth=0.0` | OK -- canonical zero-stroke fill-only |

### `dagua/render/edges/collection.py`

| Lines | Pattern | Classification |
|---|---|---|
| 1605 | `font_size=edge.label_font_size` | OK -- `DaguaText` constructor field assignment; `font_size` field type is points-input, converted to data units inside `render_text` via `safe_scale` (collection.py:553). The boundary conversion happens at the render leaf, not the data class. |

### `dagua/render/borders/collection.py`

| Lines | Pattern | Classification |
|---|---|---|
| 28 | `linewidth=0.0` | OK -- canonical zero-stroke clip-proxy |

### Aggregate

- `linewidth=0.0` (zero-stroke fill-only): 26 sites
- Locked constants (`_MIN_HATCH_LINEWIDTH_POINTS`, `_CROSSING_BRIDGE_STROKE_WIDTH_POINTS`): 2 sites
- `_PORT_INDICATOR_BORDER_WIDTH_POINTS * display_scale` (locked-constant times scale): 1 site
- `_*_VISIBLE_STROKE_POINTS * display_scale` minimum-width clamp: 1 site (line 5325, not in the linewidth-keyword grep)
- Helper inputs flowing into `_effective_font_size_points` / `_*_font_size_data` data-coord helpers: 7 sites
- `DaguaText.font_size` / `DaguaEdge.label_font_size` constructor field assignments (converted at render leaf): 4 sites
- `_build_node_patch` dead-in-production helper params: 14 sites
- `_matplotlib_text_background_path` invisible-probe (documented inverse boundary): 1 site

**Leakages found: 0.**

## Part 3 -- `tests/test_render_dpi_invariance.py` completeness

Read end-to-end (496 lines). Test inventory:

1. `test_pair_fixture_geometry_ratios_are_dpi_invariant` -- exercises basic
   node border, label font, edge stroke ratios.
2. `test_cluster_border_dpi_invariance` -- exercises solid cluster ribbon
   (the round-14 fix path).
3. `test_double_circle_inner_ring_dpi_invariance` -- exercises compound
   ring inner-stroke ribbon (the round-14 fix path).
4. `test_cylinder_rim_dpi_invariance` -- exercises cylinder rim ribbon
   (the round-14 fix path).

All 4 verify DPI invariance at {100, 150, 200, 300} with a 5% tolerance,
which is the right operational guard for catching new display-point leaks:
a `linewidth=style.field` regression would scale ratios by ~3x between
DPI 100 and 300 and trip the assertion immediately.

### Coverage assessment

The four fixtures exercise:

- node border ribbons (pair fixture)
- label text geometry (pair fixture)
- edge stroke ribbons (pair fixture)
- cluster border ribbons (cluster fixture)
- compound annular rings (double_circle fixture)
- cap-shape rims (cylinder fixture)

Two primitive types that *use* data-coord ribbons but are NOT directly
exercised:

- **Text outline ribbons** (round-15 fix). The pair fixture's label uses
  `outline=False` by default, so the round-15 outline change isn't exercised
  by a DPI-invariance assertion. A future regression that re-introduced a
  `linewidth=_outline_linewidth(spec)` display-point leak in `text/collection.py:639-655`
  would not be caught by these tests.
- **Bold emphasis ribbons** (round-15 fix). Similarly not exercised.
- **Port indicator markers** (round-15 fix). The pair fixture's edge has
  no port indicators set on either endpoint, so a regression to
  `Line2D` + `markersize=size_points` would not trip the test.
- **Crossing bridges** (locked constant; lower priority but still uncovered).

These are genuine coverage gaps. However, classifying severity:

- The round-15 text/port fixes have **zero metric impact** (mean L1 went
  1.516 -> 1.515) and are **already locked-down structurally** by the data-coord
  pattern itself: any regression would manifest as `linewidth=<expr>` with
  `<expr> != 0.0` and `<expr>` not one of the named locked constants, which
  the next code review or grep sweep catches. The DPI-invariance test is
  defense in depth, not the primary lock.
- The minor cost of three additional fixtures (text-outline, port-indicator,
  embolden) is low; the marginal value is "future audits can be entirely
  test-driven instead of needing a code sweep." This is a **suggested
  follow-up**, not a sprint blocker.

### Recommendation

**Not a blocker for `STOP_CONVERGED`.** Optional follow-up: add three
small fixtures in a future PR -- `_build_text_outline_graph(...)` exercising
`outline=True`, `_build_port_indicator_graph(...)` exercising
`port_indicator="dot"` on an edge, `_build_bold_graph(...)` exercising
`font_weight="bold"`. Each is ~30 lines and would close the structural
DPI-coverage gap. The bar of "every primitive type that uses data-coord
MUST have at least one fixture" is a reasonable target post-cairo.

## Part 4 -- Visual gates regression check

I read each of the requested gate images and verified the expected
outcome:

| Gate | Image | Outcome |
|---|---|---|
| Round-11 win: 3D box edge stem | `cards/comparisons/nodes/shapes/box3d_vs_graphviz.png` | Edge stem clearly visible between dagua source and target nodes; arrowhead crisp; right-side 3D edges of the box render correctly. PASS. |
| Round-12 FONT_FLOOR=0.5 win: combo_pie_bold labels | `per_card_pixel_diff/comparisons/combo_pie_bold_vs_graphviz.png` | Labels (Ingest, Validate, Review, Approve, Ship) readable on the dagua side at the smaller scale; FONT_FLOOR=0.5 keeping density labels above the legibility floor. PASS. |
| Round-14 win: double_circle inner ring | `cards/comparisons/nodes/shapes/double_circle_vs_graphviz.png` | Inner ring is a thin visible blue band on both Source and Target; not collapsed/missing. PASS. |
| Round-14 win: cylinder rim | `cards/comparisons/nodes/shapes/cylinder_vs_graphviz.png` | Top and bottom rim curves visible on both Source and Target cylinders. PASS. |
| Round-9/14 path: cluster opacity_1_0 border | `per_card_pixel_diff/comparisons/clusters_opacity_1_0_vs_graphviz.png` | Outer cluster solid blue rectangle visible; inner cluster nested correctly with own border; nodes still visible with white-fill ellipses against blue background. PASS. |
| Round-15 spot check: text_outline_on | `cards/reference/nodes/text/text_outline_on.png` | Source/Target labels are readable, outlined glyph fills present (subtle but consistent). PASS. |
| Round-15 spot check: font_weight_bold | `cards/reference/nodes/text/font_weight_bold.png` | Source/Target labels visibly bolder than non-bold reference (confirmed weight differential). Bold emphasis ribbon is doing its job. PASS. |

All 7 visual gates preserved. No regression.

## Part 5 -- Tests

```
$ pytest tests/test_render_dpi_invariance.py tests/test_render_pair_edges.py \
    tests/test_render_density_label.py -x --tb=short
============================== 6 passed in 1.64s ==============================
```

All 6 regression tests green.

## Verdict: `STOP_CONVERGED`

Sprint A (data-coord-everything) is genuinely done.

Justification against the bar set in the brief:

- **Zero `leakage` findings.** The Part 2 sweep classified every match. The
  bar "a `linewidth=` call passing user-facing data is leakage even if
  matplotlib needs points there" is satisfied: every site is either zero,
  a locked named constant, a `display_scale`-converted ribbon width, a
  helper-input flowing into a data-coord conversion, a constructor field
  whose conversion happens downstream at the render leaf, or
  `_matplotlib_text_background_path`'s invisible-probe inverse-boundary
  use which `SCALING.md` explicitly authorizes.
- **Visual gates pass.** Round-9/11/12/14/15 wins all preserved.
- **Regression tests green.** All 6 pass.
- **`SCALING.md` reflects the post-round-15 philosophy.** Reads cleanly,
  encodes the 2026-03-23 directive with structural argument, code idiom,
  clamp pattern, and two legitimate residual categories.

The dpi-invariance test could expand to three more fixtures (text outline,
port indicator, bold emphasis), but that's a marginal coverage improvement,
not a sprint blocker. The data-coord pattern is structurally locked
elsewhere -- by the grep-discoverability of any `linewidth=<non-zero>` and
by the existing audit cadence.

**Recommendation: close Sprint A. Begin cairo sprint.**

## Optional follow-ups (not blockers)

1. Add 3 dpi-invariance fixtures (text outline, port indicator, bold
   emphasis) to lock down the round-15 fixes test-side. ~90 lines total.
2. Consider deleting `_build_node_patch` since it's only used by
   `tests/` and `scripts/generate_node_border_comparisons.py`. Deferring
   this lets the diff comparison script keep working; it is not a leak
   risk because no production caller uses it. (Already noted in round
   14/15 audits.)
3. Cairo sprint will close `_CROSSING_BRIDGE_STROKE_WIDTH_POINTS` and
   `_MIN_HATCH_LINEWIDTH_POINTS` naturally (cairo can rasterize ribbons
   at sub-pixel widths without rasterization-floor tricks). Not urgent.
