<task>
Phase 5 of the cluster sprint — corrective round. Phases 1-4 landed but the post-Phase-4 audit (`/home/jtaylor/projects/dagua/.project-context/research/sprint_clusters/AUDIT_post_phase_4.md`) found 5 HIGH-severity defects + a critical instrument gap. The cluster sprint is NOT bulletproof yet.

Read `/home/jtaylor/projects/dagua/.project-context/research/sprint_clusters/AUDIT_post_phase_4.md` FULLY before starting — it has detailed per-defect descriptions and likely fix locations.

This is NOT the originally-deferred "Phase 5 (Sugiyama+clusters)" — that's still deferred. This is a corrective round addressing real bugs the prior phases introduced or didn't catch.

Repo: `/home/jtaylor/projects/dagua` (already on `develop`). Single working branch.

## Fix list (priority order)

### F1 (CRITICAL bug) — Inner cluster rectangles not drawing on nested_clusters
On `nested_clusters` panel, only thin vertical fragments of inner cluster boxes are visible — top/bottom edges entirely missing. This is a REGRESSION from Phase 3 (likely z-order fix or path build).

Investigate `_draw_clusters` (`dagua/render/mpl.py:7513-7770`):
- Trace what happens when an inner cluster is rendered for `nested_clusters` topology
- Check the annular path build for degenerate-segment case (when inner cluster has narrow leaf bbox)
- Likely causes: (a) annular path collapses when outer-minus-inner ring has zero/negative width somewhere; (b) z-order change in Phase 3 reordered something; (c) min-width clamp produces a path that doesn't close.

Verify by reading rendered `nested_clusters` panel after fix — all cluster rectangles must be fully drawn.

### F2 (CRITICAL bug) — "Large Cluster With Longer Label" rectangle COMPLETELY MISSING on cluster_showcase
Same root cause class as F1 but more severe — only the floating label is drawn, no rectangle.

Investigate min-width clamp + label-fit-width clamp interaction (around `_draw_clusters`):
- The clamp formula `(min_w - cw)/2` may have a sign error when `min_w < cw` and the label width forces expansion in one axis but not the other
- Possible: `min_cluster_width = cluster_height * 0.65` produces a bbox dimension where one of the path coords goes negative, causing matplotlib to skip drawing

Add a print probe to trace the path coords for the "Large Cluster" case, fix the math, verify rectangle draws.

### F3 (HIGH) — Cluster top-edge stroke visible through label text on every panel
The `@background` mask added in Phase 3 uses a too-narrow padding (or computes width from char-count not rasterized text bbox). On every panel, stroke shows through the label.

Fix:
- In `_draw_clusters`, when computing the label background rectangle, use `matplotlib.text.Text.get_window_extent()` (or the equivalent rasterized text bbox) to size the mask, not nominal `font_size_pt * len(label)` math.
- Increase `label_background_padding` default from `(4.0, 2.0)` to `(6.0, 4.0)` for graphviz themes — gives extra clearance on either side.

Verify on nested_clusters and deep_nesting_4 — stroke must be cleanly masked behind every cluster label.

### F4 (HIGH) — Phase 4 edge clipping not engaging visually
Audit reports external→internal arrowheads still land INSIDE cluster bboxes; bypass edges pierce perimeters.

Investigate `dagua/edges.py` clipping logic added in Phase 4:
- Verify the clipping uses the RENDERED cluster bbox (with render-time padding), NOT the placement-time bbox.
- The issue may be that the clipping operates on an unshifted bbox while the rendered cluster has been expanded by render-padding (depth-stepped), producing a perimeter that doesn't match.
- Trace one edge in `transformer_block` end-to-end: what bbox does the clip routine see vs what the renderer draws.

Fix: route the rendered bbox (post all clamps and padding) to the clipping function. Verify visually.

### F5 (HIGH) — Sibling overlap on cross_cluster_edges, microservices
Audit notes this is partly harness artifact (the comparison gallery uses dot's positions, bypassing Phase 2's placement). But also partly a render-side issue: cluster bboxes computed at render time can grow past their placement-time footprint via min-width / label-fit clamps, producing rectangles that overlap even when placement said they don't.

Two-part fix:
- (a) Cap the cluster bbox at render time so it can NEVER exceed its placement-time footprint by more than 2pt. Either: skip the min-width/label-fit clamps when `cluster_aware=True`, OR raise placement-time bbox to include those clamps.
- (b) Add a `--use-dagua-placement` flag to `scripts/graphviz_theme_comparison.py` so Phase 2's gains are visible. Default unchanged for backward compat.

### F6 (instrument gap) — parity_metrics blind to rectangle presence
Audit critique: declarative metric reports 41/41 cluster features in tolerance because it only checks fill/stroke/font, not whether the cluster RECTANGLE actually drew.

Add cluster-rectangle-presence assertion to `scripts/parity_metrics.py`:
- For each cluster in dot's SVG, extract its polygon vertices (always present).
- For each cluster in dagua's render, verify the cluster patch was actually emitted (non-zero path length, finite vertices).
- If dot has N cluster rectangles and dagua has M < N: report (N - M) missing rectangles per panel as `cluster_rect_missing` feature deltas.

This catches F1/F2-class regressions automatically going forward.

## Verification

For each fix above, after implementing, render the affected panels and read them to verify visually. Image budget ≤ 12 reads.

After all fixes:
1. Visual confirmation on nested_clusters, cluster_showcase, transformer_block, cross_cluster_edges, deep_nesting_4 — all cluster rectangles drawn, labels clearly readable above strokes, edges cleanly clipped at perimeters.
2. `python scripts/parity_metrics.py` — should now report `cluster_rect_missing` if any panels still have missing rectangles. Aim for 100% passing on this new feature.
3. `python scripts/parity_pixel_diff.py` — mean SSIM should improve on cluster panels.
4. `pytest tests/test_layout/ tests/test_render/ tests/test_style.py tests/test_parity_metrics.py -x --tb=short -q` — all pass.

## Out of scope

- Sugiyama+clusters (still deferred to a future sprint)
- Don't break non-cluster rendering
- Don't undo Phase 1-4 commits — fix forward

## Completeness contract

Not done until:
1. F1, F2, F3, F4, F5, F6 all implemented (or documented as truly infeasible).
2. New cluster_rect_missing metric in parity_metrics.py reports 0 missing rectangles on all 45 panels.
3. Visual verification on 5 cluster panels — rectangles drawn, labels masked, edges clipped.
4. Existing tests pass (excluding pre-existing test_classic_drl.py import error).
5. Parity metric stays >= 99%.
6. ONE commit on develop: `feat(cluster): phase 5 — corrective fixes (rectangle drawing, label mask, edge clip wiring, instrument gap)`.
7. REPORT at `.project-context/research/sprint_clusters/REPORT_phase_5.md` with: per-fix outcome, before/after visual observations, deviations.

## Reply format

Per-fix outcome (F1-F6), commit SHA, before/after observations on the 5 cluster panels. ≤300 words.
</task>

<missing_context_gating>
F1 and F2 are CRITICAL bugs — cluster rectangles missing entirely. Spend whatever investigation time needed to root-cause these BEFORE moving on; they're regressions and must be fixed.

If F4 (edge clipping) is structurally hard because Phase 4's wiring is fundamentally broken, document and propose a Phase 6 to redo.

The instrument gap (F6) is mandatory — without it we can't trust future audits.
</missing_context_gating>

<action_safety>
develop branch. ONE commit at end. Theme + render + tests + parity_metrics.py infrastructure only.
</action_safety>
