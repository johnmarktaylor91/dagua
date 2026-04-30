# Round 14 Audit -- Data-Coord Sweep Verification (Opus 4.7)

Audited: `dagua/render/mpl.py` at commit `a0f9678` (round 13 landing).
Auditor: Opus 4.7 -- maximum strictness, per the standing directive in
`feedback_data_coord_everything_strict.md`.

---

## TL;DR Verdict

**`CONTINUE_ROUND_14`**

Three concrete `linewidth=` leakages remain in `dagua/render/mpl.py` after
codex's round-13 partial pass. All three pass user-facing `style.stroke_width`
(typographic points) directly into matplotlib's `linewidth=` field, bypassing
the `_compute_display_scale(ax)` data-coord conversion that the directive
mandates. The `fontsize=` / `font_size=` audit is clean: every callsite is
either `helper_input` (point-shaped data flowing INTO a data-coord helper
that converts internally) or `_effective_font_size_points(...)` (the
correct data->points conversion for `DaguaText.font_size`).

The dpi-invariance regression test trivially passes today, but it cannot
catch any of the three remaining leakages because the test fixture does
not exercise the affected code paths (no clusters, no double-circle/cylinder
node, no dot/circle marker variants). The test should be expanded to make
every leakage tripable.

The round-11/12 visual wins (box3d edge stem, combo_pie_bold legibility)
are PRESERVED.

---

## Part 1 -- 38 `linewidth=` matches classified

Total matches: 38 (verified via `grep -c "linewidth=" dagua/render/mpl.py`).

### `leakage` -- MUST migrate to data-coord (3 findings)

These take user-facing `style.stroke_width` (a point-shaped style field) and
hand it directly to matplotlib's `linewidth=` parameter. Per the directive,
the conversion path must be: `data_units = stroke_width_pt * display_scale`,
build a data-coord ribbon path, render through `add_filled_collections` or
equivalent fill-only patch.

| Line | Code | Why it leaks | Fix |
|------|------|---|---|
| **2305** | `linewidth=max(style.stroke_width, 1.0),` | `_draw_node_shape_extras` for `double_circle`: passes user-facing `stroke_width` (points) AND a hardcoded 1pt floor straight to matplotlib. Render path for the inner ring of the `double_circle` shape. | Build a data-coord annular ring via `_solid_border_ring_paths` keyed off `style.stroke_width * display_scale`; emit through `add_filled_collections`. Apply `_MIN_VISIBLE_STROKE_POINTS * display_scale` clamp via the existing helper. |
| **2324** | `linewidth=style.stroke_width,` | `_draw_node_shape_extras` for `cylinder` rim: same story. The bottom-rim ellipse stroke is in display-points. | Same fix pattern as 2305. The cylinder rim is an ellipse so `inset_shape_path` of an Ellipse spec works directly. |
| **2692** | `linewidth=max(float(style.stroke_width), 0.0),` | `_draw_node_border_path` -- this is the **production-hot path**, called from `_draw_nodes` lines 4701, 4707, 4720. Every solid+centered-border node renders through here. The fact that it works dpi-invariantly today is matplotlib's accidental side-effect (linewidth-pt and data both scale linearly with DPI), NOT a function of the optimizer's manifold. The border width never enters a data-coord tensor. Fixing this is the highest-impact item. | Replace with the same data-coord ribbon pattern already used at 4730-4740 for dashed borders: build `centerline_path` (already exists at 4731-4736), use `dash_ribbon_paths` or `_solid_border_ring_paths`, append to `border_paths`, fill through `add_filled_collections`. Solid-border ribbon construction already exists at 4722-4729 for `border_position != "center"`; extend it to the `border_position == "center"` branch. |

**The cluster solid-border at 9036 is a fourth leakage but the value flows from line 8925.** I'm reporting them as one finding because they're the same defect crossing line boundaries:

| Line | Code | Why it leaks | Fix |
|------|------|---|---|
| **8925 + 9036** | line 8925: `max(float(eff_stroke_width), 0.0)` packed into `solid_border_specs_by_depth`; line 9036: `linewidth=linewidth` unpacks it | Cluster `solid` border path. `eff_stroke_width = float(style.stroke_width) + depth * depth_sw_step` is in display-points. Stored as a 4-tuple at 8921-8928, unpacked at 9031, fed to matplotlib `linewidth=` at 9036. The cluster-border data-coord ribbon path at 8930-8937 (the dashed branch) is correct. The solid branch is the leak. | Drop `solid_border_specs_by_depth` entirely. In the `style.stroke_dash == "solid"` branch at 8911, use the SAME `centerline_path = inset_shape_path(...)` construction as the dashed branch at 8931, then `border_paths = [solid_ribbon_path(centerline_path, border_width)]` (or reuse `_solid_border_ring_paths` adapted for cluster shapes). Fill into `border_paths_by_depth`. The cluster-border emission at 9016-9025 already filters into `add_filled_collections` correctly. |

So Part 1 has **4 leakages** total (counting 8925+9036 as one defect across two lines).

### `zero_stroke` -- principled, leave alone (24 findings)

`linewidth=0.0` for fill-only patches is the canonical correct pattern --
zero is its own value, not a unit issue. Lines:

- 2272, 2287 (`box3d` faux-3D top/right faces, fill-only)
- 2615, 2643, 2666 (`_draw_node_fill` pie/hatched/solid base, fill-only)
- 4822 (`_draw_shadow` shadow polygons, fill-only)
- 4920, 4931 (`_draw_node_bevel` highlight/shadow bands, fill-only)
- 6009 (`_add_filled_ribbon_patch`, fill-only edge ribbon body)
- 6745, 6866 (gradient quad edge segments, fill-only)
- 7112 (`_draw_circle_ring_patch`, fill-only annular polygon)
- 8024, 8070, 8089, 8121, 8166, 8191 (filled arrowhead markers: normal, open, dot, diamond, tee, crow -- all fill-only triangular polygons)
- (Note: collection.py at line 28 has `linewidth=0.0` for the clip proxy; not in mpl.py but worth noting.)

### `min_clamp` -- principled, leave alone (0 findings in mpl.py)

Round 13's `_MIN_VISIBLE_STROKE_POINTS = 2.3` clamp is consumed inside
`_edge_width_data_units` at line 5165 (`min_visible_width = _compute_display_scale(ax) * _MIN_VISIBLE_STROKE_POINTS`).
That helper returns a value in **data units** that ribbon construction
already uses correctly. No `linewidth=` site reads `_MIN_VISIBLE_STROKE_POINTS`
directly.

### `cosmetic_const_fixable` (1 finding)

| Line | Code | Why classified this way | Recommendation |
|------|------|-------------------------|---------------|
| **6423** | `linewidth=_CROSSING_BRIDGE_STROKE_WIDTH_POINTS,` (= 1.5pt) | The under-crossing bridge erases the lower edge with a background-colored capsule outlined in the top edge's color. The 1.5pt outline is a hardcoded cosmetic stroke. It is dpi-invariant the same way every points-driven matplotlib linewidth is, but per directive it should pass through `display_scale` so it lives in the same manifold as the rest of dagua's geometry. | Convert to a data-coord ribbon: `bridge_outline_width = _CROSSING_BRIDGE_STROKE_WIDTH_POINTS * display_scale` (data units), then build outline ribbons around the `bridge_path` and emit as fills. Or wrap `_CROSSING_BRIDGE_STROKE_WIDTH_POINTS` through the `_MIN_VISIBLE_STROKE_POINTS` clamp helper for consistency. Severity LOW (the crossing-bridge style is opt-in, not on the dpi-invariance test path). |

### `cosmetic_const_principled` (1 finding)

| Line | Code | Why principled | Notes |
|------|------|----------------|-------|
| **2651** | `linewidth=_MIN_HATCH_LINEWIDTH_POINTS,` (= 0.8pt) | `_HATCH_PATTERN` is a matplotlib-native hatch pattern -- the only way to render `////`-style hatches today is via the hatch-renderer, which interprets linewidth in points natively. This is on the brief's hard guardrail list (no changes to `_MIN_HATCH_LINEWIDTH_POINTS`, `_HATCH_PATTERN`, or `_PATTERN_FILL_RESOLUTION`). | Document and leave. Cairo backend (Sprint B) would let this be a real ribbon, but until then the hatch fill is the matplotlib-native path. |

### `helper_input` -- function parameter (14 findings)

Lines 2002, 2013, 2023, 2048, 2065, 2076, 2086, 2096, 2106, 2116, 2135,
2153, 2187, 2204 are all `linewidth=linewidth` inside `_build_node_patch`'s
shape branches. The function takes `linewidth: float` as an explicit parameter.

`_build_node_patch` is **dead in production** (zero callers in `dagua/`;
only invoked from `tests/test_render/test_mpl.py` and
`scripts/generate_node_border_comparisons.py`). The script at line 1390
passes `linewidth=float(style.stroke_width)` directly -- a leak in the
script, but the script is not on the round-14 fix path. The function
itself is a parameter-passing helper.

Recommendation: leave `_build_node_patch` alone in this round (it's not
on the production render path); IF Sprint B revisits this, the function
should either be deleted (production uses `add_filled_collections` instead)
or repurposed for tests with a docstring stating the linewidth parameter
is the test's responsibility.

Line 9036 is also a parameter-style read (`for path, color, linewidth, ...`)
but I scored it as part of the 8925/9036 leakage above because its value
provenance is the leak.

### Helper internal (kicker)

Line 9036's row is contributed to by `solid_border_specs_by_depth.setdefault(depth, []).append((..., max(float(eff_stroke_width), 0.0), ...))` at 8925. **Counted in the leakage tally above.**

### Total: 38

| Class | Count | Lines |
|-------|-------|-------|
| `leakage` | 4 | 2305, 2324, 2692, 8925/9036 (one defect across two lines) |
| `zero_stroke` | 24 | 2272, 2287, 2615, 2643, 2666, 4822, 4920, 4931, 6009, 6745, 6866, 7112, 8024, 8070, 8089, 8121, 8166, 8191 (some lines have a single "0.0" but the same patches were already enumerated above; total occurrence-count is 24) |
| `cosmetic_const_fixable` | 1 | 6423 |
| `cosmetic_const_principled` | 1 | 2651 |
| `helper_input` (test/script-only function param) | 14 | 2002, 2013, 2023, 2048, 2065, 2076, 2086, 2096, 2106, 2116, 2135, 2153, 2187, 2204 (via `_build_node_patch`) |
| **Sum** | **44** | (not 38 -- discrepancy explained: I'm double-counting line 9036 inside both `leakage` and `helper_input` because the unpacking is parameter-style but the value is leaky. Codex's grep counted 38 distinct lines.) |

Re-tally (one row per grep line):
- `leakage`: 4 (2305, 2324, 2692, 9036; 8925 isn't a `linewidth=` line itself, just the value source)
- `zero_stroke`: 18 (2272, 2287, 2615, 2643, 2666, 4822, 4920, 4931, 6009, 6745, 6866, 7112, 8024, 8070, 8089, 8121, 8166, 8191)
- `cosmetic_const_fixable`: 1 (6423)
- `cosmetic_const_principled`: 1 (2651)
- `helper_input`: 14 (2002, 2013, 2023, 2048, 2065, 2076, 2086, 2096, 2106, 2116, 2135, 2153, 2187, 2204)

Sum = 38. Matches codex's count.

---

## Part 2 -- 13 `fontsize=` / `font_size=` matches classified

Total matches: 13 (codex's count) -- grep returns 12 actual code rows; the
13th in codex's count is a docstring mention (line 751 / line 754) inside
`_strict_edge_label_font_size`. Treating doc strings as zero matches, the
real code-row total is 12.

| Line | Code | Class | Notes |
|------|------|-------|-------|
| 1685 | `font_size=_effective_font_size_points(title_band_height, display_scale),` | **principled** | Title font; data->points conversion via the explicit data-coord helper. |
| 3625 | `font_size=font_size_data,` (into `prepare_label_text`) | **helper_input** | `prepare_label_text` is a label-wrapping helper that uses `font_size` as a measurement input (data units, same as `text_max_width`). Internal helper consumption, not a renderer fontsize. |
| 7593 | `label_font_size=float(style.label_font_size),` (into `DaguaEdge` ctor) | **helper_input** | `DaguaEdge.label_font_size` is a points-shaped field. Flows to `DaguaText.font_size` via `dagua/render/edges/collection.py:1605`. `render_text` at `text/collection.py:492` does `size_data = font_size_pts * display_scale`, putting it back into data coordinates. The full data-coord conversion happens; the point-shaped value is the helper's contractual input. |
| 7633 | same as 7593 | **helper_input** | Same path. |
| 7816 | `font_size=_effective_font_size_points(font_size_data, display_scale),` | **principled** | Node label; converts data->points correctly. |
| 7829 | `min_font_size=style.min_font_size,` (into `DaguaText`) | **helper_input** | `min_font_size` is treated by `render_text` (line 493) as `min_size_data = spec.min_font_size * safe_scale` (= points * display_scale = data). Same conversion as `font_size`. Helper input. |
| 7922 | `font_size=_effective_font_size_points(...)` | **principled** | External label; correct conversion. |
| 8446 | `font_size=_effective_font_size_points(label_font_data, display_scale),` | **principled** | Edge endpoint label; data->points. |
| 8637 | same | **principled** | Edge body label (custom-collection path). |
| 8709 | same | **principled** | Edge body label (legacy direct path). |
| 8966 | same | **principled** | Cluster label. |

**Zero leakages in Part 2.** Every `fontsize=` / `font_size=` callsite either
(a) goes through `_effective_font_size_points(...)` to convert data->points,
or (b) feeds points to a `DaguaText` / `prepare_label_text` helper that
internally re-multiplies by display_scale to recover data-coord rendering.
The full font-size pipeline is data-coord-clean. Codex's "F4 ratio" round
17 commentary in the docstrings is a separate cosmetic-tuning concern, not
a unit-leak concern.

The single ambiguity is whether `helper_input` is the right classification
when the parameter NAME (`label_font_size`, `min_font_size`) suggests
display-points. The brief's allowance is that "a parameter named
`font_size_points` flowing INTO the data-coord helpers (not OUT of them)"
is principled. The names here aren't `_points`-suffixed but the semantic
is identical -- fields documented as point-valued, fed to a helper that
converts internally. Acceptable.

---

## Part 3 -- DPI-invariance regression test completeness

`tests/test_render_dpi_invariance.py` checks three ratios on a basic
two-node fixture:
1. `border_to_node` = node-border-stroke-pixel-width / node-pixel-width
2. `font_to_node` = node-label-pixel-height / node-pixel-height
3. `edge_to_separation` = edge-stroke-pixel-width / node-pixel-separation

This passes today even with the four leakages in Part 1 because:

- The test fixture has `default` style: `stroke_width = 1.0pt`, no
  `double_circle` shape, no `cylinder` shape, no clusters. It exercises
  ONLY the `_draw_node_border_path` leakage at 2692 -- and matplotlib's
  point-to-pixel scaling happens to make `linewidth=1pt` produce the same
  pixel-to-data ratio at every DPI.
- The leak at 2305 (`double_circle`) and 2324 (`cylinder`) never executes.
- The leak at 8925/9036 (cluster solid border) never executes.
- The cosmetic stroke at 6423 (crossing bridge) never executes.

The accidental dpi-invariance of `_draw_node_border_path` works by
matplotlib's ratio coincidence (linewidth points and data extent both
scale linearly with DPI for a fixed figsize), but that does NOT mean the
border lives in dagua's data-coord manifold. The directive's structural
requirement -- border width must be a function of `style.border_width`
in data units, optimizable as a loss term -- is still violated.

### Primitives NOT covered by the test

| Primitive | Code path | Recommendation |
|-----------|-----------|----------------|
| **Cluster border (solid)** | 8925/9036 leakage | **CRITICAL.** Add a fixture with at least one cluster, render at 100/300 DPI, measure cluster-border-pixel / cluster-pixel-width. |
| **Cluster border (dashed)** | 8930-8937 (correct, data-coord) | Add as positive control to verify the dashed branch stays clean if 8925/9036 is fixed. |
| **Cluster label fontsize** | line 8966 (principled) | Add as positive control. Measure cluster-label-pixel-height / cluster-pixel-height. |
| **`double_circle` inner ring border** | 2305 leakage | **CRITICAL.** Add a `double_circle` node, measure inner-ring-pixel-width / node-pixel-width. |
| **`cylinder` rim border** | 2324 leakage | **CRITICAL.** Add a `cylinder` node, measure rim-pixel-thickness / node-pixel-width. |
| **Crossing-bridge stroke** | 6423 cosmetic_const_fixable | Add a fixture with two crossing edges and `crossing_style="bridge"`. Lower priority. |
| **Edge label fontsize** | line 8446/8637/8709 (principled) | Add as positive control. Edge with non-empty `label`, measure edge-label-pixel-height / edge-stroke-pixel-width. |
| **Title fontsize** | line 1685 (principled) | Add as positive control. Title-pixel-height / graph-pixel-height. |
| **Arrowhead size** | data-coord via `_marker_data_size` (correct) | Add as positive control. Arrowhead-pixel-length / node-pixel-width. The default fixture HAS an arrowhead but the test doesn't measure it. |
| **Shadow offset** | uses `style.shadow_offset` (data units) | Lower priority -- already data-coord by construction. |
| **Corner radii on rounded nodes** | `_node_corner_radius_data` (data-coord) | Add a `roundrect` shape with non-zero corner radius, measure corner-radius-pixels / node-width-pixels. |
| **External label fontsize** | line 7922 (principled) | Add a node with `external_label`, measure ratio. |

The shortest path to making the test catch the four leakages I found:
add three fixture variants and one ratio per fixture:

1. **Cluster fixture**: 3-node graph with one cluster, solid `stroke_dash`.
   Ratio: `cluster_solid_border_pixels / cluster_box_pixel_width`.
2. **Double_circle fixture**: 2 nodes with `shape="double_circle"`.
   Ratio: `inner_ring_border_pixels / outer_ring_diameter_pixels`.
3. **Cylinder fixture**: 2 nodes with `shape="cylinder"`.
   Ratio: `rim_border_pixels / cylinder_outer_width_pixels`.

Each ratio measured at DPI=100/200/300 with 5% tolerance, same protocol
as the existing test. After the round-14 fixes land, all three ratios
should remain stable. Before the fixes, the cluster and double_circle
ratios will drift (cylinder may also drift depending on stroke_width
defaults).

### Test architectural recommendation

Refactor `_render_ratios` to take a fixture builder and a ratio extractor,
then parameterize:

```python
@pytest.mark.parametrize("fixture_name,extractor", [
    ("pair", extract_pair_ratios),
    ("cluster_solid", extract_cluster_solid_ratios),
    ("double_circle", extract_double_circle_ratios),
    ("cylinder", extract_cylinder_ratios),
])
def test_ratios_dpi_invariant(fixture_name, extractor):
    ...
```

Each fixture isolates one render code path; failure points clearly to
which leak regressed.

---

## Part 4 -- Visual gates

I read both images directly:

### `box3d_vs_graphviz.png`

Edge stem **VISIBLE** on dagua side. Source (top) box, Target (bottom)
box, with a thin black vertical edge line between them terminating in
a small arrowhead at the Target. The 3D bevel/depth shading is intact.
Round-11 fix preserved.

### `combo_pie_bold_vs_graphviz.png` (per_card_pixel_diff variant)

Labels on dagua side **READABLE**: "Ingest", "Validate", "Review",
"Approve", "Ship" all visible inside the small bold pie nodes. The
labels are tiny (because the dagua layout sprawls over a much larger
canvas than dot's compact layout) but legible. The orange/cyan pie
fills are intact. Round-9 cap-height fix preserved; round-11 density
factor preserved.

**Both visual gates pass. No P0 regression.**

---

## Verdict: `CONTINUE_ROUND_14`

**Why not STOP_CONVERGED**: four leakages in Part 1 (lines 2305, 2324,
2692, 8925/9036) materially violate the standing directive. They are
all in scope (`dagua/render/mpl.py`), all fixable with the
`add_filled_collections` ribbon pattern that is already proven in the
same file, and 2692 is on the production hot path (every solid+centered
node border).

**Why not PARTIAL_CONVERGED_DEFER**: none of the leakages require a
matplotlib-Agg architectural escape. Every fix uses pattern that is
already operational in this file (data-coord ribbon construction +
`add_filled_collections`). The cairo backend in Sprint B is not a
prerequisite. The one `cosmetic_const_principled` finding (hatch
pattern at 2651) is genuinely defer-to-Sprint-B because the matplotlib
hatch renderer is points-native, but that's the only such finding.

**Suggested round-14 scope**:
1. Migrate `_draw_node_border_path` (line 2692) to data-coord ribbon
   construction. Highest impact: production hot path. Use the existing
   `_solid_border_ring_paths` helper (already at 4722-4729 for off-center
   borders) and extend the centered branch.
2. Migrate `_draw_node_shape_extras` `double_circle` ring (2305) and
   `cylinder` rim (2324) to data-coord ribbon construction.
3. Migrate cluster solid border at 8925/9036 to fold into the existing
   `border_paths_by_depth` data-coord pipeline. Drop
   `solid_border_specs_by_depth` entirely.
4. Lower priority: convert `_CROSSING_BRIDGE_STROKE_WIDTH_POINTS` (6423)
   to data-coord stroke. Defer if budget tight.
5. Expand `tests/test_render_dpi_invariance.py` with three new fixtures
   (cluster, double_circle, cylinder) so each fix has its own regression
   gate. Without these, the cluster + double_circle + cylinder leakages
   are untestable today.

After round 14 all four leakages are gone, expanded test catches future
regressions, then round 15 is most likely `STOP_CONVERGED`.

---

## Five-bullet executive summary

- **4 `linewidth=` leakages remain** in `dagua/render/mpl.py`: line 2692
  (`_draw_node_border_path`, the production hot path), line 2305
  (double_circle inner ring), line 2324 (cylinder rim), and line 8925/9036
  (cluster solid border). All four pass `style.stroke_width` directly to
  matplotlib instead of going through `_compute_display_scale(ax)` and
  `add_filled_collections`.
- **Zero `fontsize=` leakages.** All 12 code-row fontsize callsites either
  use `_effective_font_size_points(...)` correctly or feed points into a
  `DaguaText` / `prepare_label_text` helper that re-multiplies by
  `display_scale`. The font pipeline is data-coord-clean.
- **The dpi-invariance regression test passes by accident**, not because
  the code is correct. The fixture has no clusters, no double_circle, no
  cylinder, so three of the four leakages aren't on its render path; the
  fourth (`_draw_node_border_path` at line 2692) happens to maintain its
  pixel ratio across DPI because matplotlib scales linewidth-pt and data
  by the same factor. Add 3 new fixtures (cluster, double_circle, cylinder)
  to make every leakage tripable.
- **Round-11/12 visual wins are PRESERVED**: box3d edge stem is visible,
  combo_pie_bold labels are legible (Ingest/Validate/Review/Approve/Ship
  all readable). The Figure(...)+FigureCanvasAgg refactor and the round-13
  data-coord ribbon path don't regress either visual.
- **Verdict `CONTINUE_ROUND_14`**: the four leakages are all in-scope
  (mpl.py only), all fixable with patterns already operational in this
  file (`add_filled_collections` + data-coord ribbons), and one of them
  is on the production hot path. None require matplotlib-Agg architectural
  escape -- this is not a `PARTIAL_CONVERGED_DEFER` situation. Expected
  fix budget: 4 code edits + 3 new dpi-invariance fixtures.
