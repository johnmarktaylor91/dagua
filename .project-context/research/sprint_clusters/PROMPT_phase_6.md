<task>
Phase 6 of the cluster sprint — final corrective round. Phase 5 closed F1-F3 cleanly but introduced 2 regressions and left the bypass-edge gap open. Per `/home/jtaylor/projects/dagua/.project-context/research/sprint_clusters/AUDIT_post_phase_5.md` (read this fully before starting).

Read for context:
- AUDIT_post_phase_5.md (drives this round)
- REPORT_phase_5.md (what Phase 5 did + the open issues it flagged)

Repo: `/home/jtaylor/projects/dagua` (already on `develop`). Single working branch.

## Fix list (priority order)

### G1 (HIGH regression from Phase 5) — deep_nesting_4 levels stack adjacent instead of concentric
Phase 5's render bbox cap (`placement footprint + 2pt`) is too aggressive — it prevents child clusters from being rendered INSIDE the parent's interior region. On deep_nesting_4, Levels 1..4 are drawn stacked adjacent instead of concentric (nested).

Fix: the cap must expand child cluster bboxes to fit fully INSIDE the parent's interior, not clamp to the child's own placement footprint. The semantic invariant: `parent_bbox` always strictly contains `child_bbox` plus `cluster_padding` on every side.

Implementation:
- In `_draw_clusters` (or wherever the Phase 5 cap was added), when iterating clusters in render order (parent-first or child-first), enforce `child.bbox ⊂ parent.bbox.shrunk(padding)`.
- If the cap is violated, expand the cluster bbox to fit inside its parent (with appropriate padding), not clamp to the placement footprint.
- For root-level clusters (no parent), keep the placement footprint as the cap.

Verify on deep_nesting_4 — Levels must be drawn concentric (Level 4 inside Level 3 inside Level 2 inside Level 1).

### G2 (HIGH regression from Phase 5) — Edge bodies clipped to stubs
On microservices "Search Service → 4 DBs" and deep_nesting_4 "Source→Outer 1", edge bodies are clipped to just arrowheads — the body is gone. Endpoint clip + perimeter clip are composing incorrectly.

Investigate the edge clipping pipeline (Phase 4 + Phase 5 changes):
- Trace one Search Service → Order DB edge end-to-end.
- The defect is likely: endpoint clip removes the source-side stub, perimeter clip removes the cluster-bbox-side stub, leaving only the arrowhead.

Fix: clip body ONCE per (cluster, node) intersection, not iteratively. After clipping at the cluster perimeter, the remaining body is the visible portion — don't apply additional endpoint trim.

Verify on microservices and deep_nesting_4 — edge bodies must be visible (not just arrowheads).

### G3 (HIGH) — Cluster label masks should mask through overlapping nodes too
On nested_clusters, node A pierces the Outer Group top stroke and the label "Outer Group" fragments through A's white fill (z-order issue).

Fix:
- Render cluster labels AFTER node fills in z-order (currently labels are at z=0.12 + depth*0.01, nodes at z=1.0 — labels are BEHIND nodes).
- OR extend the label mask to cover overlapping node fill regions too.

Cleanest: bump cluster label z-order ABOVE node fills (e.g. 1.5 + depth*0.01). Verify nodes still render on top of cluster fills (which is desired).

### G4 (HIGH, was deferred from Phase 4) — Bypass edges through foreign clusters
On transformer_block, bypass edges that ENTER and EXIT a foreign cluster (e.g. an edge that goes "around" FFN cluster but visually passes through its right side) are not gapped.

Fix: Phase 4 only handled (src in, tgt out) and (src out, tgt in). Now also handle (src out, tgt out, BUT polyline crosses cluster). For these, segment the edge body so the portion inside the foreign cluster is invisible (gapped), with the visible body before-and-after the cluster perimeter.

This may require the edge renderer to support segmented body curves. Two approaches:
- (a) Render two separate bezier patches for the visible portions
- (b) Apply a clip-path to the edge body that excludes foreign cluster interiors

If both are complex, document and partially implement.

VERIFY on transformer_block — bypass edges no longer cross cluster strokes uninterrupted.

### G5 (CRITICAL infra) — Re-run gallery with --use-dagua-placement
Phase 5 added a `--use-dagua-placement` flag to `scripts/graphviz_theme_comparison.py` but the gallery wasn't regenerated with it. Re-run the gallery with this flag to verify Phase 2's cluster-aware placement gains. Without this, the audits keep reporting harness artifacts as cosmetic gaps.

After all G1-G4 fixes, run:
```
python scripts/graphviz_theme_comparison.py --output-dir eval_output/cluster_phase_6_check --use-dagua-placement
```
Then visually verify cluster_showcase, cross_cluster_edges, microservices — sibling-cluster overlap should be reduced now that dagua's actual placement is used.

If `--use-dagua-placement` was implemented incorrectly (e.g., the harness only applies it to dagua's render but still feeds dot's positions), debug and fix.

## Verification

1. Visual inspection on deep_nesting_4, microservices, nested_clusters, transformer_block, cluster_showcase, cross_cluster_edges (≤12 image reads).
2. Both galleries:
   - `eval_output/cluster_phase_6_check_dot_positions/` (default — dot positions for dagua)
   - `eval_output/cluster_phase_6_check_dagua_placement/` (--use-dagua-placement)
   Compare cluster_showcase / cross_cluster_edges between the two to demonstrate Phase 2's gains.
3. `pytest tests/test_layout/ tests/test_render/ tests/test_style.py tests/test_parity_metrics.py -x --tb=short -q` — all pass.
4. Parity metrics: `python scripts/parity_metrics.py` — `cluster_rect_missing` stays at 100%.

## Out of scope

- Sugiyama+clusters (still deferred)
- Don't break non-cluster rendering

## Completeness contract

Not done until:
1. G1, G2, G3, G4, G5 all attempted with verification.
2. deep_nesting_4 levels render concentric (G1).
3. microservices edge bodies are visible (G2).
4. nested_clusters Outer Group label not pierced by node A (G3).
5. transformer_block bypass edges either gapped or documented as deferred-to-Phase-7 (G4).
6. Both galleries generated and compared (G5).
7. Tests pass.
8. ONE commit on develop: `feat(cluster): phase 6 — corrective (concentric nesting, edge body composition, label z-order, bypass edges, dagua placement audit)`.
9. REPORT at `.project-context/research/sprint_clusters/REPORT_phase_6.md`.

## Reply format

Per-fix outcome (G1-G5), commit SHA, before/after observations on the panels. ≤300 words.
</task>

<missing_context_gating>
G4 (bypass edges) is the hardest. If after 30 min you cannot get a clean implementation, document the partial state and move on — we have other fixes to land.

If `--use-dagua-placement` flag turns out to be incorrectly implemented in Phase 5, fix it as part of G5 — that's critical for future audits.
</missing_context_gating>

<action_safety>
develop branch. ONE commit at end.
</action_safety>
