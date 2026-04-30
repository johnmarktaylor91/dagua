<task>
Phase 4 of the cluster sprint — edge clipping at cluster perimeter. Per `/home/jtaylor/projects/dagua/.project-context/research/sprint_clusters/DESIGN.md` §6 Phase 4 + §3.4 + §5.5.

Read DESIGN.md fully. Especially §3.4 (current external-edge-cluster crossing behavior) and §5.5 (3-step approach).

Repo: `/home/jtaylor/projects/dagua` (already on `develop`). Single working branch.

## Context

Phase 2 made placement cluster-aware (FR/KK/FA2/SFDP). Phase 3 made cluster rendering match dot visually (top-center labels, opaque background masks). Remaining defect class: **edges from outside a cluster to inside that cluster currently render with the visible polyline starting at the source's port, passing through empty space, and entering the cluster bbox by traversing the cluster's interior region BEFORE reaching the target's port.** Visually the edge appears to "punch through" the cluster's stroke / interior region.

dot's behavior: the visible polyline starts at the source's port, terminates at the cluster's perimeter intersection, and the rest of the edge body inside the cluster is clipped (the arrowhead sits at the target's port, but the body's appearance suggests it enters cleanly through the cluster boundary).

## Goal

Implement edge clipping at cluster perimeter for cosmetic edges that cross cluster boundaries:
- For each edge (src, tgt): determine all clusters the edge polyline crosses where exactly one endpoint is inside.
- For each such cluster, clip the visible edge body so it terminates at the cluster's outer-stroke perimeter on the side the edge enters from.
- Arrowhead position remains at target node port (not on cluster perimeter — the edge body becomes shorter, not the arrow).
- Apply only when `cluster_aware=True` AND the cluster has a visible stroke (`stroke_opacity > 0` or default).

## Files to touch

### 1. `dagua/render/edges/` or `dagua/routing.py` — clipping logic

Find the existing edge polyline / Bezier rendering code. Add a post-routing clipping pass:

```python
def clip_edge_at_cluster_boundaries(
    edge_polyline: list[tuple[float, float]],  # ordered points along the edge
    src_idx: int,
    tgt_idx: int,
    cluster_membership: dict[int, list[str]],  # node_idx -> list of cluster names containing it
    cluster_bboxes: dict[str, ClusterPlacementBox],
    skip_inner_cluster: bool = True,  # don't clip to clusters that contain BOTH endpoints
) -> list[tuple[float, float]]:
    """Clip an edge polyline at cluster perimeters.

    For each cluster crossed by the edge where exactly one endpoint is a member,
    truncate the visible polyline at the cluster's outer stroke perimeter.

    Returns a (possibly shorter) polyline. The endpoint segments that fall inside
    the perimeter are removed.
    """
```

Geometry helpers needed:
- `polyline_intersect_rect(polyline, rect) -> Optional[index, point]` — find first intersection of polyline with rectangle perimeter.
- The cluster bbox is rectangular; intersection is straightforward parametric line-vs-rectangle for each polyline segment.

### 2. Wire the clip into the render path

In `dagua/render/edges/collection.py` or wherever edge polylines are produced for matplotlib:
- After computing the polyline (Bezier control points → flattened polyline), but BEFORE the arrowhead/label is positioned, call the clipping helper.
- Arrowhead remains at target. Body is clipped.

If the existing render path uses Bezier patches directly (not flattened polylines), the clip should be done as a parametric `t` adjustment: find `t*` where the Bezier crosses the perimeter, and render only `t ∈ [t*, 1]` (target side).

### 3. Edge case handling

- **Both endpoints in same cluster**: no clipping (skip_inner_cluster=True).
- **Neither endpoint in cluster**: no clipping (no membership change).
- **Nested clusters**: clip at the OUTERMOST cluster the edge enters (the one closest to the source). Don't clip at inner clusters.
- **Self-loops**: skip clipping.
- **Backedges**: clip same as forward edges.
- **Cluster has no visible stroke**: skip clipping (no visible perimeter to clip to).

### 4. Tests

In `tests/test_render/test_edge_cluster_clip.py` (new):
- Render a graph with one external node A and one internal node B inside cluster C. Verify the rendered edge body terminates at cluster C's perimeter (the polyline's last point near the C bbox edge).
- Same with multiple clusters: nested A → outer → inner → B. Verify clipping happens at outer's perimeter, not inner's.
- `cluster_aware=False`: verify clipping doesn't happen (legacy mode preserved).

## Verification

1. **New tests pass**: `pytest tests/test_render/test_edge_cluster_clip.py -x --tb=short -q`.
2. **Existing tests pass**: `pytest tests/test_layout/ tests/test_render/ tests/test_style.py tests/test_parity_metrics.py -x --tb=short -q`.
3. **Visual cluster check** on `transformer_block`, `nested_clusters`, `cross_cluster_edges`:
   - Render via `python scripts/graphviz_theme_comparison.py --output-dir eval_output/cluster_phase_4_check`.
   - Read those panels — confirm edges no longer punch through cluster strokes.
4. **Parity metric stays >= 99%**.

## Out of scope

- Phase 5 (Sugiyama+clusters)
- Layout-level placement re-tuning (Phase 2 territory)
- Cluster bbox computation changes (Phase 1 territory)

## Completeness contract

Not done until:
1. Clipping helper implemented + wired in render path.
2. Edge cases handled (both-in, neither-in, nested, self-loop, backedge, no-stroke).
3. New tests pass.
4. Existing tests pass.
5. Visual verification on 3 cluster panels (no edge-cluster-stroke punch-through).
6. ONE commit on develop: `feat(cluster): phase 4 — edge clipping at cluster perimeter`.
7. REPORT at `.project-context/research/sprint_clusters/REPORT_phase_4.md`.

## Reply format

Per-step outcome, commit SHA, before/after observations on the 3 cluster panels. ≤200 words.
</task>

<missing_context_gating>
If the existing render path uses parametric Bezier rendering (not polyline flattening), the cleanest implementation may be: solve for the parametric t where the Bezier crosses the rectangle, and render only the portion from t* to 1. If that's complex, fall back to flattening the Bezier to a polyline at fine resolution, clipping the polyline, and converting back to a Bezier patch (lossy but works).

If clipping interacts badly with the arrowhead positioning (e.g. arrow ends up rotated weird because the clipped polyline tail has a different tangent than the unclipped Bezier tail), keep the arrowhead positioning logic on the UNCLIPPED Bezier — clip ONLY the body. Document this choice.
</missing_context_gating>

<action_safety>
develop branch. ONE commit at end.
</action_safety>
