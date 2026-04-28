<task>
Round 5 of cosmetic parity work for `graphviz_strict` theme. Background: a Sonnet audit prematurely recommended STOP at round 4. An expert user overruled and a max-pickiness Opus 4.7 re-audit found 19 distinct departures (8 HIGH, 7 MEDIUM, 4 LOW). You are implementing the next round of fixes.

Read these for full context (mandatory):
- `.project-context/research/sprint_graphviz_parity/AUDIT_round_4_OPUS.md` — the picky audit driving this round (READ IT FULLY before starting; it has measurements, panel evidence, and likely-fix locations for every issue)
- `.project-context/research/sprint_graphviz_parity/REPORT_round_3.md` — your prior implementation report (round 3 commit 602daae)

SCOPE: COSMETIC RENDERING ONLY for `graphviz_strict` theme.
- DO NOT touch dagua/layout/.
- DO NOT modify scripts/graphviz_theme_comparison.py (the harness).
- DO NOT spend cycles on the IMPROVED `graphviz` theme — it is explicitly deferred. Only `graphviz_strict` matters this round.
- OK to touch: dagua/styles.py, dagua/render/, tests/.

THE FIX LIST (priority ordered)
================================

## TIER 1 — TRIVIAL PARAMETER CHANGES (do first; biggest impact-per-line)

### F1 (H1): Font family — Times New Roman → TeX Gyre Termes
The user's "different font" complaint has a confirmed root cause: native dot's "Times-Roman" PostScript font name resolves to **TeX Gyre Termes** on this Linux system (`fc-match` verified by Opus audit), while dagua's `font_family="Times New Roman"` resolves to Microsoft Core Fonts' Times New Roman. These are physically different fonts. In `dagua/styles.py`:
- Change every occurrence of `font_family="Times New Roman"` in GRAPHVIZ_STRICT_THEME (and `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE`) to `font_family="TeX Gyre Termes"`.
- Verify the font is installed (`fc-match "TeX Gyre Termes"` should resolve to `qtmr.pfb`).
- Optionally add a fallback chain like `"TeX Gyre Termes, Nimbus Roman No9 L, Times-Roman, serif"` if matplotlib supports comma-separated families.

### F2 (H2): Node font size 10.5pt → 12.0pt
Round 3 over-corrected (DPI normalization formula was right but matplotlib's effective DPI on this machine is 100, not 96). Per Opus measurement: dot renders "Preprocess" letters at ~28-30px height in a ~70px ellipse (ratio 0.40-0.42); strict renders the same at ~18-20px in a ~60px ellipse (ratio 0.32-0.33). Raise to ~12.0pt empirically — verify by overlaying LEFT and MIDDLE pipeline.png panels and matching cap-height. Adjust to 12.5 if still slightly small.
- Change `_GRAPHVIZ_STRICT_DEFAULT_NODE_STYLE.font_size` 10.5 → 12.0
- Change EdgeStyle's `label_font_size` (in graphviz_strict edge_styles default) 10.5 → 12.0
- Change GraphStyle's `edge_label_font_size` 10.5 → 12.0

### F3 (M3): Cluster border lighter and thinner
- Change cluster `stroke` from `"#AAAAAA"` to `"#CCCCCC"` in graphviz_strict ClusterStyle
- Change cluster `stroke_width` from 0.8 to 0.5

### F4 (M4): Cluster fill_opacity 0.15 → 0.08
The audit measured dot's cluster fill at ~0.05-0.10 alpha; current 0.15 reads as visible gray. Drop to 0.08.

### F5 (M2): Arrow proportions toward more equilateral
Current 10×7pt reads as elongated/pointy vs dot's stout/equilateral arrowheads. Try `arrow_length=8.0, arrow_width=8.0` or `arrow_length=9.0, arrow_width=7.5`. Verify on pipeline.png and arrow_types.png — strict arrowheads should be visibly squat/wide, not pointy.

### F6 (M7): Edge body stroke 1.0 → 0.75
Round 3 moved node stroke from 1.0 to 0.75; do the same for the edge body width since the same matplotlib AA bias applies. Change EdgeStyle.width from 1.0 to 0.75 in graphviz_strict default edge style.

## TIER 2 — RENDERER CHANGES (medium effort, high impact)

### F7 (H3): Ellipse aspect ratio — use sqrt(2) circumscription
Native dot computes ellipse semi-axes as `width = max(label_width + 2*margin, node_width) * sqrt(2) / 2` and same for height (sqrt-2 makes the ellipse circumscribe the label box rather than match it tightly — produces rounder shapes). Dagua currently uses tighter circumscription, producing flatter/wider ellipses (Opus measurements: dot's "Preprocess" ratio ~2.0, dagua's ~3.2).
- Find the ellipse-fitting routine in `dagua/render/mpl.py` (or `dagua/render/borders/` shapes module). Look for ellipse path generation that takes a bounding box and produces semi-axes.
- Change the semi-axis multiplier so the ellipse circumscribes the label text bounding box at the sqrt(2) factor, NOT just contains it.
- Verify on pipeline.png and diamond.png — strict ellipses should look more rounded/circular than they currently do.

### F8 (H4 + H5 + H8): Cluster bounding box defects
ALL three issues likely live in the same cluster bounding-box computation pass:
- **H4**: Outer Group cluster top-edge cuts through node A on nested_clusters. Cluster bounding box not leaving room above for label, OR not accounting for external nodes near the cluster's top boundary.
- **H5**: Sibling clusters Right Branch / Left Branch overlap; their labels nearly collide. Sibling-cluster horizontal padding too small (or zero).
- **H8**: Cluster border stroke crosses through cluster label text (visible on nested_clusters, deep_nesting_4, cluster_showcase). Dot breaks the top-stroke around the label OR puts label outside the box; dagua does neither.

Investigation path:
- Locate cluster-rendering code (likely `dagua/render/clusters.py` or a sibling in `dagua/render/`). Search for where the cluster rectangle path is generated.
- For H4: cluster top edge y-coordinate should be `min(y of contained nodes) - cluster_top_padding - label_height`, ensuring label fits ABOVE the contents and the box doesn't overlap external nodes positioned above it.
- For H5: when computing horizontal extent of sibling sub-clusters, leave `cluster_horizontal_separation` (e.g. 15-20pt) of whitespace between siblings.
- For H8: pick whichever is cleanest:
  - (a) draw a white-fill rectangle behind the cluster label (matplotlib z-order: label backdrop above stroke, label text on top)
  - (b) split the top stroke into two segments that flank the label, leaving a gap of `label_width + 2*padding` centered at label position
  - (c) set the label's `bbox` parameter in matplotlib `Text` artist to a white-fill rectangle (cleanest if available)

Verify on nested_clusters.png, deep_nesting_4.png, cluster_showcase.png. All three should: (1) cluster boxes never cross external nodes, (2) sibling clusters separated by visible whitespace, (3) cluster label text never crossed by the box stroke.

### F9 (H6): Back-edge curvature absolute floor for long chords
`curvature=0.2` is fraction-of-chord, so long chords (e.g. state_machine's Done→Idle, multi_cycle's G→A) render as near-straight lines passing through the body of the graph. Native dot routes them around the side as visible curves.
- Find back-edge curvature application in `dagua/render/edges/` (likely `geometry.py` or `collection.py`).
- Change the formula from `offset = curvature * chord_length` to `offset = max(curvature * chord_length, MIN_BACK_EDGE_OFFSET_PT)` where `MIN_BACK_EDGE_OFFSET_PT` is e.g. 30-40pt. This ensures long chords still produce a visible side-routing curve.
- Verify on state_machine.png and multi_cycle.png — back-edges should now arc visibly to one side rather than cutting straight through.

### F10 (H7): Open arrow forms (vee, open, circle) render as filled
Currently graphviz_strict renders `vee`, `open`, and `circle` as filled shapes; native dot renders them as outline-only / hollow. The graphviz semantic is: the "o" prefix opens an arrowhead (e.g. `ovee`, `ocircle`), but bare `vee` is itself open (just two strokes, no fill). Investigation:
- In `dagua/render/edges/arrowheads.py`, locate the per-arrowhead-name rendering.
- For `vee`: should render as two strokes meeting at the tip, no triangular fill.
- For `open`: should render as triangle outline only (stroke, no fill).
- For `circle`: should render as ring (stroke, no fill) — ALIAS may need fixing again. Round-1 changed `circle` alias from `odot` (hollow) to `dot` (filled); the audit says native dot's `circle` is hollow. Re-verify by running `dot -Tsvg` on a graph using `arrowhead=circle` and checking the SVG path. Whichever native dot does, match it.
- The `tee` fix in round 3 (perpendicular bar) is correct; don't undo it.

Verify on arrow_types.png — each named arrowhead (normal/vee/dot/diamond/tee/crow/circle/open/none) should match dot's panel: filled vs hollow distinction is correct per name.

## TIER 3 — POLISH (do if time permits)

### F11 (M1): Arrow tip-to-boundary spacing
Currently inconsistent: gap on pipeline (~1-2px), overlap on diamond. Edge-trim routine that computes "where to stop the edge body" is slightly off relative to arrow length.
- Find edge-trim logic (likely `dagua/render/edges/collection.py` or `geometry.py`).
- Standard fix: compute the boundary intersection point on the target ellipse (parametric ellipse intersection), place arrow tip exactly there, and trim edge body to `tip_pt - arrow_length * tangent_direction`.

### F12 (M5): Edge label collision avoidance
On state_machine.png the labels "retry" and "resume" sit so close they read as one word. Dot offsets one of them along-edge or perpendicular to avoid the collision.
- Find edge label placement code in `dagua/render/edges/labels.py` or similar.
- Detect when two edge labels would overlap (their bounding boxes intersect) and offset one perpendicular to the edge centerline by `label_height + small_padding`.

### F13 (M6): X11 named-color RGB table verification
Audit notes color saturation drift on colors_showcase. Native dot's red is "lightcoral" (#F08080), yellow is the X11 "yellow" (#FFFF00). Verify dagua's name-to-RGB table for fillcolor lookups matches X11 classic palette exactly. If dagua uses different RGBs for these names, align them.

## SKIP THIS ROUND (low priority)
- L1 title font weight (panel decoration, not graph content)
- L2 panel top margin
- L3 long-label padding (largely covered by F7 ellipse ratio)
- L4 vertical centering 1-2px (sub-pixel territory)
</task>

<completeness_contract>
Not done until:
1. ALL of F1-F10 implemented (TIER 1 + TIER 2). F11-F13 highly preferred but acceptable to defer with justification in REPORT_round_5.md.
2. `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"` passes for IN-SCOPE files (the pre-existing layout-import errors from prior reports are still out of scope).
3. Re-render the gallery: `python scripts/graphviz_theme_comparison.py --output-dir eval_output/graphviz_theme_round_5`.
4. Visually verify by reading at LEAST these PNGs from the new gallery: `pipeline.png` (font + size + ellipse + arrows), `nested_clusters.png` (cluster boxes — H4/H5/H8), `state_machine.png` (back-edges — H6, edge labels — F12), `arrow_types.png` (open vs filled — H7), `cluster_showcase.png` (cluster fill/stroke — F3/F4/H8), `colors_showcase.png` (font + colors — H1/F13).
5. ONE commit: `feat(theme): graphviz_strict cosmetic round 5 — TeX Gyre Termes font, ellipse sqrt(2) ratio, cluster box fixes, back-edge curvature floor, open arrow forms, polish` with the body listing each fix.
6. REPORT_round_5.md at `.project-context/research/sprint_graphviz_parity/REPORT_round_5.md` with: what landed, what didn't (and why), font verification (`fc-match "TeX Gyre Termes"` output), back-edge offset value used, ellipse ratio formula change details, any deviations.

Same scope/safety rules as prior rounds. Single commit on `develop`.
</completeness_contract>

<verification_loop>
For Tier 1 (F1-F6): tier-1 targeted tests (`pytest tests/test_style.py tests/test_render -x --tb=short -q`) after each batch of changes.

For Tier 2 (F7-F10): each fix needs a re-render of at minimum 1-2 panels to confirm visual change. Don't move to next fix until the prior one's visual change is verified.

For F8 (clusters) specifically: verify on at least nested_clusters.png AND deep_nesting_4.png AND cluster_showcase.png — each exercises slightly different cluster scenarios.

For F9 (back-edges): verify the new floor produces a visible arc on long chords by reading state_machine.png and multi_cycle.png after the fix.

For F10 (open arrows): the audit notes the round-1 "circle" alias change may have been wrong. Verify by running `dot -Tsvg` on a small test graph and inspecting SVG paths for filled vs unfilled. Document the verification in REPORT_round_5.md.

After all fixes: full tier-2 run once, then re-render full gallery.
</verification_loop>

<missing_context_gating>
Default to most reasonable interpretation. Investigate before changing if uncertain. Document deviations in REPORT_round_5.md.

If a fix seems harder than its tier suggests (e.g. F8 cluster fixes turn out to require layout-touching code), document the issue, implement what you can without crossing the layout boundary, and flag for next round.
</missing_context_gating>

<action_safety>
Theme + render + tests only. develop branch. ONE commit at end.
</action_safety>

<default_follow_through_policy>
Default to most reasonable low-risk interpretation. Pre-existing test failures unrelated to your changes are not your problem. Keep going. Document deviations.
</default_follow_through_policy>
