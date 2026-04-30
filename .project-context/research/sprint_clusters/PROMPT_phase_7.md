<task>
Phase 7 of the cluster sprint — narrow corrective for two render bugs found in the post-Phase-6 audit. Per `/home/jtaylor/projects/dagua/.project-context/research/sprint_clusters/AUDIT_FINAL.md`.

Read AUDIT_FINAL.md FULLY before starting — it has the precise diagnoses for H1 (top edges missing) and H2 (label fragmentation).

Repo: `/home/jtaylor/projects/dagua` (already on `develop`). Single working branch.

## Fix list (only 2 — narrow scope)

### H1 (HIGH render bug) — Top edges missing on 5 cluster panels
Affects: `nested_clusters`, `transformer_block`, `cross_cluster_edges`, `data_pipeline`, `flat_many_clusters`. Cluster rectangles render with both vertical strokes but the TOP edge is missing entirely (showing only `[`-shaped fragments).

Per the audit: Phase 5+6 bbox cap collapses sibling cluster top edges to zero height in some scenarios. This is a path-builder bug in cluster rendering.

Investigation:
1. Look at `_draw_clusters` (or wherever Phase 6 added the bbox cap logic).
2. Trace what path is built when sibling clusters are constrained by the parent's interior.
3. Identify the case where the cluster's top y-coordinate ends up equal to or below its bottom y-coordinate (zero or negative height).
4. The bug is likely: parent.interior_top - sibling.label_band > sibling.interior_bottom, but the clamp pushes top down past bottom.

Fix:
- Ensure `cluster.bbox.height >= label_band + min_inner_height` always.
- If parent's interior is too small to fit a child cluster's full label_band + content, expand the parent's bbox to fit (rather than clamping the child to zero).

Verify by reading the rendered panels — every cluster rectangle must have a visible top edge.

### H2 (HIGH render bug) — "Outer Group" label fragmented on nested_clusters
Phase 6 claimed to fix label z-order so labels render above node fills, but the rendered panel still shows "O...ap" (label fragmented by node A's white fill masking the middle).

Investigation:
1. Look at the z-order changes Phase 6 made.
2. Confirm whether the label glyph z was raised but the LABEL BACKGROUND MASK z was NOT — that would let the mask sit BEHIND the node fill, exposing the stroke through the label, then the high-z glyph paints over only some of the label area.
3. Both the mask AND the glyph need to be ABOVE node fills, OR neither needs special z-order if the mask covers nodes too.

Fix:
- Ensure cluster label background mask is at the SAME z-order as the label glyph (both above node fills).
- OR extend mask coverage to extend across overlapping nodes.

Verify on `nested_clusters` — "Outer Group" label must be fully readable, not fragmented.

## Verification

1. Visual inspection on the 5 H1-affected panels + nested_clusters for H2 (≤12 image reads).
2. Run `python scripts/parity_pixel_diff.py --hires nested_clusters,transformer_block,cross_cluster_edges,data_pipeline,flat_many_clusters` and read the dagua-side hi-res to verify visually.
3. `pytest tests/test_layout/ tests/test_render/ tests/test_style.py tests/test_parity_metrics.py -x --tb=short -q` — all pass.
4. Parity metric `cluster_rect_missing` stays at 100%.

## Out of scope

- H3 (cluster-aware placement collapsing directed-flow graphs into a ball) — that's the cluster-aware-Sugiyama sprint, separate
- Any other audit findings (MED/LOW)
- Don't break non-cluster rendering

## Completeness contract

Not done until:
1. H1 fixed: all cluster rectangles have visible top edges on the 5 affected panels.
2. H2 fixed: nested_clusters "Outer Group" label fully readable.
3. Tests pass.
4. ONE commit on develop: `feat(cluster): phase 7 — render fixes (top edges, label z-order final)`.
5. REPORT at `.project-context/research/sprint_clusters/REPORT_phase_7.md`.

## Reply format

Per-fix outcome (H1, H2), commit SHA, before/after observations on the 5 panels. ≤200 words.
</task>

<missing_context_gating>
If H1 turns out to be a deeper bbox-computation issue (not a path-builder bug), document and propose a focused Phase 8. Don't burn cycles trying to fix it both ways.
</missing_context_gating>

<action_safety>
develop branch. ONE commit at end. Render code only.
</action_safety>
