NEW SESSION: Read this file first. Then read CLAUDE.md, AGENTS.md,
.project-context/knowledge/cosmetic_inventory.md, and
.project-context/knowledge/gotchas.md. After reading, your first action
should be: continue the cosmetic tuning sprint per the user's direction.

PRIORITY: Cosmetic tuning sprint -- continuing from gallery audit expansion.

## What Just Happened (massive session)

### Gallery Audit Expansion (COMPLETE -- committed)
- Added 41 new non-evil combo cards (79 total, was 38)
- Added 20 new evil stress-test cards (35 total, was 15)
- 6 rounds of LLM critic review + Codex renderer fixes
- ALL non-evil combos at 9+ critic score
- ALL evil cards at 7+ (no catastrophes)
- Commit: fcc5c93 (test fix) on top of b4033fd (main work)

### Renderer Bugs Fixed (generalizable, not card-specific)
1. inset_shape_path: non-polygon shapes (cloud/stadium/document/tab/note/box3d)
   fall back to scaled ShapeSpec instead of crashing on polygon_vertices()
2. Shadow contours: shadows now follow actual node shape path, not rect bbox
3. Bold/italic font: fontweight/fontstyle properly passed to matplotlib ax.text()
4. Text outline: uses matplotlib patheffects.withStroke
5. Hatched fill: hatch parameter applied with visible pattern
6. Sharp crossing geometry: V-notch size proportional to edge width
7. Head/tail label clearance: offset from arrowhead tips
8. overflow_policy shrink_text: enforced on curved node shapes
9. Deep cluster viewport: bounds expanded for nested cluster padding

### New Combo Coverage
2-way (22 new): taxi/straight routing, pie/donut fills, external labels,
text outlines, cylinder/cloud/stadium/tab/note/document/box3d/parallelogram/
trapezoid/pentagon/octagon shapes, crossing gap/sharp, BT/RL directions,
crow arrowheads, hatched+gradient

3-way (10 new): taxi+shadow+gradient, pie+gradient+bold, ext_label+diamond+shadow,
cloud+gradient+italic, stadium+striped+shadow, hatched+shadow+bold,
head_tail_labels+ortho, bt+cluster+rounded, text_outline+shadow+bold,
color_gradient+taper+thick

4-way (5 new): pie+shadow+gradient+bold, cylinder+dashed+shadow+gradient,
taxi+crossing_gap+gradient, cloud+striped+shadow+italic,
ext_label+hexagon+gradient+bold

5-way (3 new): kitchen_sink_4/5/6

### New Evil Coverage (20)
Self-loops on star/diamond/triangle, long wrapped text in concave shapes,
24-edge mega-hub, zero-width edges, mixed overflow policies, empty labels,
unicode labels, negative curvature, 100-node grid, 8-deep clusters,
pie-on-star, donut-on-diamond, taxi self-loop, all-arrowheads hub,
white-on-white gradient, extreme taper crossing, contradictory per-node styles

### Also Added (by Codex, not explicitly requested)
- Van Essen and Cajal neuroscience themes in dagua/styles.py (~5000 lines)
- These were TODO items that Codex picked up opportunistically
- NOT yet reviewed or critic-scored -- may need attention

## Test State
- 29 tests passing in test_generate_cosmetic_album + test_build_gallery_audit
- Pre-existing failures: test_bench_large (checkpoint), test_animation (MP4),
  test_cascade (style cascade), test_graphviz_theme_comparison (script test)

## Git State
- Branch: feat/bench-and-aesthetics
- Latest commits: fcc5c93, b4033fd
- Clean working tree (modified files are uncommitted .project-context/ and
  untracked docs/gallery/, scripts/, tests/ from prior sessions)

## Key Files

| File | What |
|------|------|
| scripts/generate_cosmetic_album.py | Source catalog (~5100 lines, 182 cases) |
| scripts/build_gallery_audit.py | Gallery audit generator (~4400 lines) |
| eval_output/gallery_audit_v33/ | Latest gallery (133 ref + 79 combo + 35 evil) |
| dagua/render/mpl.py | Main renderer (~5800 lines) |
| dagua/render/borders/inset.py | Shape insetting (non-polygon fix) |
| dagua/render/text/paths.py | Text path rendering (bold/italic fix) |
| dagua/render/crossings.py | Crossing detection + rendering |
| dagua/graph.py | Node sizing, overflow_policy |
| dagua/utils.py | Node size computation |

## How the Gallery Audit Pipeline Works

1. `generate_cosmetic_album.py`: defines AlbumCase objects with graph/positions/settings
2. `build_gallery_audit.py`: imports cases, builds fixtures, applies params, renders cards
3. Combo cases flow through `build_combo_specs()` -> `_combo_params()` -> `_apply_reference_params()`
4. Evil cases flow through `build_evil_specs()` -> `_render_evil_card()` (uses pre-built graph directly)
5. `_combo_params()` has smart defaults for: gradient, shadow, striped/hatched fills,
   opacity, color_gradient, taper, crossing, text_wrap, cluster, corner_radius,
   star/cloud/box3d short labels, external_label offset, text_outline, pie min_width

## What to Continue With

The user wants to keep going with cosmetic tuning. Possible next directions:
- Graphviz theme calibration (min>=8, mean>=9 target from todos)
- More combo coverage (e.g., compound arrowheads, port styles, border_position)
- Visual audit of existing 133 reference cards for regressions after renderer changes
- Review the Van Essen/Cajal themes Codex added
- Address remaining bugs from todos.md (arrowhead placement, cluster label collision)
- Ask the user what they want to focus on next

## Critic Review Process (for reference)
- Spawn 2-3 parallel review agents, each reading ~10-20 card PNGs
- Score 1-10 on: label readability, feature visibility, visual polish, layout
- Non-evil target: 9+. Evil target: 7+
- Batch fixes by type (systemic renderer > params > card-specific)
- Dispatch fixes to Codex, re-render, re-review until targets met
