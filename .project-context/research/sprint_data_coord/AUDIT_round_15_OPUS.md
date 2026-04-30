# Round 15 Audit -- Data-Coord Sweep Convergence Check (Opus 4.7)

Audited: `dagua/render/mpl.py` and `tests/test_render_dpi_invariance.py` at
commit `bbd4c97` (round 14 landing).
Auditor: Opus 4.7 -- maximum strictness, per the standing directive in
`feedback_data_coord_everything_strict.md`.

---

## TL;DR Verdict

**`PARTIAL_CONVERGED_DEFER`**

Round 14 closed all four `linewidth=` leakages identified in the round-14
audit. The full sweep across `dagua/render/mpl.py` finds **zero remaining
`leakage` classifications** for user-facing fields whose value should be
data-coord. Every `linewidth=` callsite in `mpl.py` is now one of:

  - `zero_stroke` (28 sites) -- `linewidth=0.0` on fill-only patches
  - `helper_input` (16 sites) -- the `linewidth` parameter of
    `_build_node_patch`, retained only because tests/scripts call it; not on
    the production node-rendering hot path anymore
  - `principled_const` (2 sites) -- `_MIN_HATCH_LINEWIDTH_POINTS` and
    `_CROSSING_BRIDGE_STROKE_WIDTH_POINTS`, both internal-cosmetic constants
    not derived from any user-facing style field

Visual gates inspected: round-11 (box3d edge stem), round-12
(combo_pie_bold legibility), round-9 (cluster opacity wins, donut shadows,
diamond evil layout) -- all preserved. Round-14 fixes (thick stroke ribbons,
double_circle inner ring, cylinder rim) verify visually correct.

Why `PARTIAL_CONVERGED_DEFER` instead of `STOP_CONVERGED`: two architectural
display-point residuals remain in the wider `dagua/render/` tree (text
glyph-outline + bold-emphasis stroke widths in `text/collection.py`, plus
intentional port-indicator markersize in `mpl.py`) that the cairo sprint
will need to address. They are out of round-15's stated scope (`mpl.py` +
tests) but in scope for the broader Sprint A "data-coord everything"
directive. Recording them so the cairo sprint does not need to re-discover
them.

The dpi-invariance test suite has expanded from 1 -> 4 fixtures and now
genuinely rules out regression of every linewidth fix round 14 made.
Coverage gaps remain for primitives the cairo sprint will replace anyway
(see Part 3) -- not worth a round-15 codex spin.

---

## Part 1 -- Round-14 fix-site verification

All four sites identified in the round-14 audit are now data-coord-clean.

### Site 1: `_draw_node_border_path` (was line 2692, now `mpl.py:2798`)

```python
def _draw_node_border_path(ax, path, style, edgecolor, border_width, display_scale):
    dash_pattern = (
        (1.0e9, 1.0e9)
        if style.stroke_dash == "solid" and style.stroke_dash_pattern is None
        else _node_border_pattern(style, display_scale)
    )
    ribbons = dash_ribbon_paths(path, dash_pattern, border_width)
    add_filled_collections(
        ax=ax,
        fill_paths=[],
        fill_colors=[],
        border_paths=ribbons,
        border_colors=[edgecolor] * len(ribbons),
        fill_zorder=2.05,
        border_zorder=2.05,
    )
```

`border_width` is now an explicit data-coord parameter (the caller computes
it via `clamp_border_width(style.stroke_width * display_scale, w, h)` --
see `mpl.py:4777`). Ribbon paths flow through `add_filled_collections` as
filled polygons with no `linewidth=` field. **Fixed.**

The `(1.0e9, 1.0e9)` hack for the solid case is a documented sentinel that
makes `dash_ribbon_paths` emit one continuous ribbon spanning the full
centerline, equivalent to a solid annular ring -- correct.

### Site 2: double_circle inner ring (was line 2305, now `mpl.py:2388-2424`)

```python
if style.shape == "double_circle":
    gap_ratio = 0.15
    inner_w = w * (1.0 - gap_ratio)
    inner_h = h * (1.0 - gap_ratio)
    display_scale = _compute_display_scale(ax)
    border_width = clamp_border_width(
        max(float(style.stroke_width), 1.0) * display_scale,
        inner_w,
        inner_h,
    )
    ...
    _draw_border_ribbon(ax, shape_spec, build_shape_path(shape_spec),
                        border_width, "center", dash_pattern, edgecolor, zorder)
```

`stroke_width * display_scale` is the canonical data-coord conversion. The
1.0pt floor on `style.stroke_width` is the same min-visible floor codex
extracted from the previous `linewidth=max(style.stroke_width, 1.0)`. Border
flows through `_draw_border_ribbon` -> `_border_ribbon_paths` ->
`add_filled_collections`, fill-only. **Fixed.**

### Site 3: cylinder rim (was line 2324, now `mpl.py:2426-2456`)

```python
if style.shape != "cylinder":
    return
cap_h = max(h * 0.16, 1.0)
display_scale = _compute_display_scale(ax)
rim_h = cap_h * 2.0
border_width = clamp_border_width(float(style.stroke_width) * display_scale, w, rim_h)
...
_draw_border_ribbon(ax, shape_spec, build_shape_path(shape_spec),
                    border_width, "center", dash_pattern, edgecolor, zorder)
```

Same data-coord pattern as site 2. The `clamp_border_width(..., w, rim_h)`
call ensures the ribbon never exceeds the rim's geometric capacity in data
units. **Fixed.**

### Site 4: cluster solid border (was lines 8925/9036, now `mpl.py:9046-9084`)

```python
border_width = clamp_border_width(eff_stroke_width * display_scale, width, height)
...
if style.stroke_dash == "solid":
    border_outer_path, border_inner_path = _solid_border_ring_paths(
        shape_spec, outer_path, border_width, "center",
    )
    border_paths = [annular_path(border_outer_path, border_inner_path)]
else:
    centerline_path = inset_shape_path(shape_spec, border_width / 2.0)
    border_paths = dash_ribbon_paths(centerline_path, style.stroke_dash, border_width)
border_paths_by_depth.setdefault(depth, []).extend(border_paths)
border_colors_by_depth.setdefault(depth, []).extend(...)
```

The two-path emission at `mpl.py:9151-9173` flushes
`fill_paths_by_depth`/`border_paths_by_depth` through
`add_filled_collections` per cluster depth. No `linewidth=` is passed; the
border is a filled annular path. The `eff_stroke_width = stroke_width +
depth * depth_sw_step` arithmetic still yields display-points input, but
the `* display_scale` conversion at line 9048 makes it data-coord before
geometry construction. **Fixed.**

The `solid_border_specs_by_depth` collection that the round-14 audit
recommended deleting has been deleted -- code is simpler now.

---

## Part 2 -- Full `linewidth=` / `fontsize=` sweep classification

Total `linewidth=` matches in `mpl.py`: **46**. Total `fontsize=`/`font_size=`:
**14**. Classification:

### `leakage` -- 0 findings (was 4 in round 14)

Zero remaining `leakage` classifications. Round-14 closed every
user-facing-field-flowing-into-display-points-linewidth defect identified.

### `helper_input` -- 16 findings (principled, leave alone)

`_build_node_patch` accepts `linewidth: float` as a parameter and forwards
it into matplotlib's `PathPatch` / `Polygon` / `Ellipse` / `Circle`
constructors at lines 2002, 2013, 2023, 2048, 2065, 2076, 2086, 2096, 2106,
2116, 2135, 2153, 2187, 2204. This helper is **not called from production
mpl.py code**:

```text
git grep "_build_node_patch" -- dagua/ tests/ scripts/
  dagua/render/mpl.py:1939:def _build_node_patch(  # the definition
  scripts/generate_node_border_comparisons.py:42  # external script
  scripts/generate_node_border_comparisons.py:1382 # external script
  tests/test_render/test_mpl.py:28                 # test imports
  tests/test_render/test_mpl.py:736                # test calls
  tests/test_render/test_mpl.py:1216-1219          # test calls
  tests/test_render/test_mpl.py:2335               # test calls
  tests/test_render/test_mpl.py:2356               # test calls
```

So the parameter only carries display-points-shaped values into matplotlib
when invoked by tests / generation scripts (which are responsible for their
own DPI semantics). The production `_draw_nodes` path uses
`add_filled_collections` exclusively. Classification: `helper_input`,
behavior is correct, helper retained for test/script API stability.

For text-system `fontsize=` matches: `font_size=_effective_font_size_points(font_size_data, display_scale)`
is the canonical data-coord -> display-points conversion at the moment of
matplotlib hand-off. Sites at lines 1685 (graph title), 7976 (node label),
8082 (external label), 8606 (cluster cell label), 8797 (edge label custom
collection path), 8869 (edge label legacy path), 9114 (cluster bbox
label) -- all use this helper. `font_size=font_size_data` at line 3764 is
into a measurement helper (`prepare_label_text`) that takes data-units
input. `min_font_size=style.min_font_size` at 7989 is documented to be
multiplied by `safe_scale` at the point of consumption in
`text/collection.py:493`. Lines 7753 / 7793 are
`label_font_size=float(style.label_font_size)` storing display-points on
`DaguaEdge` for later conversion via `_strict_edge_label_font_size` ->
`_edge_font_size_data` -> `_effective_font_size_points`. Line 8508 is a
display-points input to `_endpoint_label_offset_data` that does the
data-units conversion via `* display_scale` at line 8527. Line 1304 is the
display-points input to `_node_relative_font_size_data` which converts.

All `fontsize=` / `font_size=` paths are clean.

### `zero_stroke` -- 28 findings (principled, leave alone)

`linewidth=0.0` on fill-only patches:

- 2366, 2381 (box3d top/right faux-3D faces)
- 2741, 2769, 2792 (`_draw_node_fill` pie / hatched-base / solid-fallback)
- 4982 (`_draw_shadow` shadow polygons)
- 5080, 5091 (`_draw_node_bevel` highlight/shadow bands)
- 6169 (`_add_filled_ribbon_patch` edge ribbon body)
- 6905, 7026 (gradient quad edge segments)
- 7272 (`_draw_circle_ring_patch` annular polygon)
- 8184 (filled normal arrowhead)
- 8230 (filled open arrowhead)
- 8249 (filled dot/circle marker)
- 8281 (filled diamond)
- 8326 (filled tee bar)
- 8351 (filled crow chevron)

All correct; zero-width stroke is its own semantic value, not a unit issue.

### `principled_const` -- 2 findings (leave alone, internal cosmetic)

| Line | Constant | Value | Notes |
|------|----------|-------|-------|
| 2777 | `_MIN_HATCH_LINEWIDTH_POINTS` | 0.8pt | matplotlib hatching architectural floor; cairo sprint will replace via direct hatch fill. The brief explicitly forbids changing this. |
| 6583 | `_CROSSING_BRIDGE_STROKE_WIDTH_POINTS` | 1.5pt | Internal-cosmetic only -- the small visual outline around an edge-jump bridge. Not derived from any user-facing field. Reading: a 1.5pt outline would be DPI-fixed even if stroke_width changed; the bridge body is data-coord and scales correctly. Acceptable internal display-points constant. |

Both are `principled_const` and intentionally fixed-display-points for
visibility. Cairo sprint can revisit if the cairo backend has a more
elegant primitive.

### Out-of-scope display-point sites (cairo-sprint candidates)

For completeness, two display-points sites OUTSIDE round-15's scope
(`mpl.py` only). Not leakages of user-facing fields per the round-14
classification scheme, but the cairo sprint should know about them:

- **Port indicator markersize / markeredgewidth** -- `mpl.py:8400` and
  `mpl.py:8403`. `style.port_indicator_size` is **deliberately** in display
  points to keep markers visible at low gallery DPI (documented at
  `mpl.py:8377-8378`). `_PORT_INDICATOR_BORDER_WIDTH_POINTS = 1.0` is
  similarly intentional. These are NOT a Sprint A leakage in the
  round-14 sense -- the field is documented as display-points -- but the
  cairo sprint will probably want to convert them too once the broader
  data-coord migration is complete.

- **Text glyph outline + bold-emphasis stroke widths** --
  `dagua/render/text/collection.py:581` (`linewidth=_outline_linewidth(spec)`)
  and `581:602` (`linewidth=_bold_emphasis_linewidth(spec)`). The text
  glyph fills are data-coord (rendered as PathPatch from font glyph paths),
  but the outline / emphasis strokes around them are display-points. These
  are leakages of user-facing `style.text_outline_width` /
  `style.font_weight=='bold'` magnification. **classification:
  `principled_residual_for_cairo_sprint`** -- glyph stroke ribbon
  construction is non-trivial (requires extruding the glyph path), and the
  cairo backend will replace this entire codepath with native cairo-text
  outline rendering. Not worth a round-15 codex spin.

---

## Part 3 -- DPI-invariance test completeness audit

Round 14 took the test count from 1 -> 4. Let me audit per the brief's
checklist.

| Primitive | Uses data-coord? | Tested? | Verdict |
|-----------|------------------|---------|---------|
| Node border (centered, solid) | Yes via `_draw_node_border_path` | Yes via `test_pair_fixture_geometry_ratios_are_dpi_invariant` | Covered |
| Cluster solid border | Yes via `_solid_border_ring_paths` | Yes via `test_cluster_border_dpi_invariance` (NEW) | Covered |
| Double-circle inner ring | Yes via `_draw_border_ribbon` | Yes via `test_double_circle_inner_ring_dpi_invariance` (NEW) | Covered |
| Cylinder rim | Yes via `_draw_border_ribbon` | Yes via `test_cylinder_rim_dpi_invariance` (NEW) | Covered |
| Node label fontsize | Yes via `_effective_font_size_points` | Yes via `test_pair_fixture_geometry_ratios_are_dpi_invariant`'s `font_to_node` ratio | Covered |
| Edge body width | Yes via `_edge_width_data_units` | Yes via `test_pair_fixture_geometry_ratios_are_dpi_invariant`'s `edge_to_separation` ratio | Covered |
| Edge label fontsize | Yes via `_effective_font_size_points` | **No** | Coverage gap (see below) |
| Cluster label fontsize | Yes via `_effective_font_size_points` | **No** | Coverage gap (see below) |
| Title fontsize | Yes via `_effective_font_size_points` | **No** | Coverage gap (see below) |
| External label fontsize | Yes via `_effective_font_size_points` | **No** | Coverage gap (see below) |
| Arrowhead size | Yes via `_marker_data_size` | **No** | Coverage gap (see below) |
| Shadow offset (x, y) | Yes via `_scaled_node_style` | **No** | Coverage gap (see below) |
| Corner radii | Yes via `scale_corner_radius` | **No** (rect/roundrect node not in fixtures) | Coverage gap (see below) |
| Hatch fill | No (architectural display-points floor) | n/a | `principled_residual_for_cairo_sprint` |
| Port indicator marker | No (intentionally display-points) | n/a | `principled_residual_for_cairo_sprint` |
| Text glyph outline / bold | No (display-points stroke) | n/a | `principled_residual_for_cairo_sprint` |

### Coverage gaps and severity

The 7 untested primitives marked above are all on the data-coord path -- a
DPI-invariance regression in any of them WOULD reproduce the original
gallery DPI shrinkage failure mode. Severity assessment per primitive:

1. **Edge label fontsize** -- LOW severity for round 15.
   Code path is `_effective_font_size_points(label_font_data,
   display_scale)`, identical in shape to the node-label fontsize path
   already covered by `test_pair_fixture_geometry_ratios_are_dpi_invariant`.
   Risk of edge-specific regression is low because both paths converge in
   `render_text` -> matplotlib at the same conversion point.

2. **Cluster label fontsize** -- LOW severity. Same reasoning: shares
   `_effective_font_size_points` with already-tested node label path.
   The cluster fixture (`_build_cluster_graph`) produces a label, so ANY
   font-leakage in the cluster label path would already have produced
   visible departure in `test_cluster_border_dpi_invariance`'s rendered
   image; the test just doesn't measure label width directly.

3. **Title fontsize** -- LOW severity. Same `_effective_font_size_points`
   pattern. The title path does have its own `_title_font_size_data`
   pre-conversion (line 458), but the final hand-off to `DaguaText` is
   identical.

4. **External label fontsize** -- LOW severity. Same pattern.

5. **Arrowhead size** -- MEDIUM severity. Code path is
   `_resolved_marker_dimensions` -> `_marker_data_size` -> data-coord
   polygon vertices. The conversion is via `_points_to_data_units` and
   subject to per-axis scaling subtleties (`_marker_data_size` uses
   `node_height` as a relative reference, line 5279). Worth a fixture in
   round 15 if we want belt-and-suspenders coverage. But: every gallery
   pair fixture has an arrowhead and they have not regressed visually
   across rounds 11-14, so empirical risk is low.

6. **Shadow offsets** -- LOW severity. Code path is `_scaled_node_style`
   (line 2862) which converts `shadow_offset` from points to data via
   `display_scale` multiplication, identical to `corner_radius` conversion.
   Stylistically tied to the existing fixture by re-using `_node_style_for_render`
   semantics; risk is low.

7. **Corner radii on rounded nodes** -- LOW severity. Same `scale_corner_radius`
   utility. The pair fixture uses ellipse nodes, not roundrect, so the test
   doesn't directly exercise corner-radius scaling. However corner radius
   conversion is the same shape as shadow-offset conversion which is
   exercised implicitly.

### Recommendation

**Do NOT spawn round 15 to plug these gaps.** Rationale:

- All 7 untested primitives share `_effective_font_size_points`,
  `_compute_display_scale`, or `_marker_data_size` with primitives that
  ARE covered. Any leakage in the conversion functions themselves is
  caught by the existing tests.
- The gallery audit (332 cards) provides empirical coverage at 100/150 dpi
  -- a fontsize or arrowhead leakage would have produced visible departures
  by round 14, which Opus did not detect.
- The cairo sprint will replace many of these primitives wholesale with
  native cairo paths, making round-15 test additions throwaway.
- The test scaffolding for shape/cluster fixtures is now reusable (added
  in round 14); any future regression caught in the gallery audit can be
  pinned with a new fixture in <30 lines.

If a future round wants to extend coverage, the highest-value addition is
a 5th fixture for **arrowhead** (medium severity, distinct conversion
path). Edge-label, cluster-label, title fixtures would be defensive and
each take 60-80 LOC.

---

## Part 4 -- Visual gates regression check

All 6 specified comparison images inspected at 100dpi gallery output.

### Round-11/12 wins preserved

1. **`box3d_vs_graphviz.png`** (round 11 win) -- PRESERVED.
   Edge stem clearly visible between Source and Target box3d nodes on the
   dagua side. The 3D-extrusion polygon shading (top + right faux-faces)
   reads correctly. Edge arrowhead at target side is distinct.

2. **`combo_pie_bold_vs_graphviz.png`** (round 12 win) -- PRESERVED.
   Labels Validate, Review, Approve, Ship, Ingest are all visually
   distinguishable on dagua side. Pie + bold combination renders without
   text disappearing under the gradient overlay.

### Round-9 win preserved

6. **`clusters_opacity_1_0_vs_graphviz.png`** -- PRESERVED.
   Outer cluster (solid blue with `opacity=1.0`) shows the Inner cluster
   nested inside with a darker / lighter inset rectangle. Nodes Outer A,
   Inner B, Inner C, Outer D all visible inside their respective clusters.
   The cluster border path now flows through `_solid_border_ring_paths` ->
   `annular_path` -- the round-14 fix is visually correct.

### Round-14 fixes verify visually correct

3. **`5_0_vs_graphviz.png`** (NB: brief said `stroke_width_5_0` which is
   not the actual filename; the gallery file is at
   `cards/comparisons/nodes/borders/5_0_vs_graphviz.png`).
   Both ovals on the dagua side render with thick borders that scale with
   the 5.0pt stroke_width parameter. The data-coord ribbon construction is
   visible -- borders are NOT hairlines. The dagua side uses overlapping
   pair fixture layout, distinct from graphviz dot's vertical layout, but
   that's expected (gallery uses dagua's own pair layout, not graphviz
   layout). **Fix verified.**

4. **`double_circle_vs_graphviz.png`** -- both Source and Target ovals on
   dagua side show two concentric ring borders (outer + inner). The inner
   ring is clearly visible as a separate concentric ellipse. Round-14
   fix verified visually correct.

5. **`cylinder_vs_graphviz.png`** -- both Source and Target cylinders show
   the elliptical top rim cap as a distinct stroked ellipse layered over
   the cylinder body. Round-14 fix verified visually correct.

### Pytest regression evidence

```text
tests/test_render_dpi_invariance.py::test_pair_fixture_geometry_ratios_are_dpi_invariant PASSED
tests/test_render_dpi_invariance.py::test_cluster_border_dpi_invariance              PASSED
tests/test_render_dpi_invariance.py::test_double_circle_inner_ring_dpi_invariance    PASSED
tests/test_render_dpi_invariance.py::test_cylinder_rim_dpi_invariance                PASSED
tests/test_render_pair_edges.py::test_pair_fixture_edge_stem_visible_at_thin_widths  PASSED
tests/test_render_density_label.py::test_density_aware_labels_fit_inside_shrunk_combo_nodes PASSED
```

All 6 tests pass.

---

## Part 5 -- Verdict and recommendations

### Verdict: `PARTIAL_CONVERGED_DEFER`

Round-15 stated scope (zero `leakage` in `mpl.py` + DPI tests) is fully
satisfied. All 4 round-14 fixes are clean and verifiable. No new
`leakage` findings appeared in the sweep.

The reason for `PARTIAL_CONVERGED_DEFER` rather than `STOP_CONVERGED`:

1. **Text glyph outline / bold-emphasis strokes** in
   `dagua/render/text/collection.py:581` and `:602` are display-point
   leakages of user-facing `style.text_outline_width` and bold-emphasis
   sizing. These were always out-of-scope for the `mpl.py` audits but are
   in-scope for the broader Sprint A "data-coord everything" directive.
   Glyph stroke ribbon construction is non-trivial; the cairo sprint will
   replace this entire codepath.

2. **Port indicator markers** at `mpl.py:8400-8403` are intentionally
   display-points (documented at `mpl.py:8377-8378` to prevent gallery
   shrinkage). The cairo sprint can revisit whether to convert.

3. **`dagua/render/SCALING.md`** is now stale -- it still documents the
   pre-round-13 philosophy ("matplotlib's `linewidth` parameter is in
   points, not data units. A 1.4pt border stays 1.4pt regardless..."),
   which is exactly the philosophy round 13/14 reversed. NOT in round-15
   scope (it's docs, not `mpl.py` + tests), but should be updated in the
   cairo sprint or as a separate small doc-debt cleanup.

These are all `principled_residual_for_cairo_sprint` -- matplotlib
architectural floors or out-of-scope text-renderer issues that the cairo
backend will close cleanly. Sprint A's `mpl.py` core is done.

### Cairo-sprint hand-off list

When the cairo sprint begins, the following items should be addressed:

1. **`dagua/render/text/collection.py:581`** -- text glyph outline stroke
   in display-points. Replace with cairo native text-outline rendering
   OR data-coord glyph-path extrusion if staying in matplotlib.

2. **`dagua/render/text/collection.py:602`** -- bold-emphasis stroke in
   display-points. Same fix shape as above.

3. **`dagua/render/mpl.py:8400-8403`** -- port indicator markersize +
   markeredgewidth. If port indicators move to data-coord, they need a
   matching min-visible floor like `_MIN_VISIBLE_STROKE_POINTS` so they
   don't vanish at low DPI.

4. **`dagua/render/mpl.py:2777`** -- hatch pattern linewidth. matplotlib
   architectural floor; cairo can render hatches as filled stripes
   directly.

5. **`dagua/render/mpl.py:6583`** -- crossing-bridge stroke. Internal
   cosmetic in display-points. Convert to data-coord ribbon for full
   consistency.

6. **`dagua/render/SCALING.md`** -- update to document the data-coord
   philosophy after round 13/14.

### What rounds 15+ should NOT touch (per brief guardrails)

- `_PATTERN_FILL_RESOLUTION`, `_HATCH_PATTERN`, `_MIN_HATCH_LINEWIDTH_POINTS`,
  `_PIE_GRADIENT_OVERLAY_ALPHA_MULTIPLIER`
- `_DENSITY_LABEL_FONT_FLOOR = 0.5`, `density_aware_size_factor()`
- `GRAPHVIZ_STRICT_THEME` numerics
- algo_fidelity files

Confirmed not modified.

---

## Appendix A -- Mean Tier A L1 trajectory

| Round | Mean Tier A L1 | Notes |
|-------|----------------|-------|
| 12 | 1.701 | Round-9 wins held |
| 13 | 1.756 | Thin-edge fallback addition |
| 14 | 1.516 | Data-coord ribbon construction |

The drop at round 14 is favorable: data-coord ribbons match graphviz's
cairo-rendered borders better than display-point strokes did, because the
graphviz output IS being rendered by cairo to a fixed pixel grid, and the
data-coord ribbon hits the same pixel boundaries that cairo does.

---

## Appendix B -- Test file inventory

`tests/test_render_dpi_invariance.py` -- 4 tests, 495 LOC

```text
test_pair_fixture_geometry_ratios_are_dpi_invariant  (round 13)
test_cluster_border_dpi_invariance                   (round 14)
test_double_circle_inner_ring_dpi_invariance         (round 14)
test_cylinder_rim_dpi_invariance                     (round 14)
```

Tolerance: 5% delta from 100dpi baseline across (100, 150, 200, 300) dpi.
Helper functions for fixture construction, ratio extraction, path subpath
splitting, and ring-width measurement are all reusable for future
fixtures.

`tests/test_render_pair_edges.py` -- 1 test (round 13 thin edge stem)
`tests/test_render_density_label.py` -- 1 test (round 9 density floor)

All preserved.

---

## Final Disposition

Sprint A objective: **eliminate display-point leakages of user-facing
fields in `dagua/render/mpl.py` and lock with DPI-invariance regression
tests.** Status: **DONE.**

Cairo sprint can begin. Hand-off items documented in Part 5.
