# Cluster Sprint — Final Summary

**Period:** 2026-04-29
**Outcome:** Render side bulletproof. Cluster-aware placement landed for FR/KK/FA2/SFDP. Hierarchical-flow placement (cluster-aware Sugiyama) explicitly deferred to a separate sprint per DEFERRED.md.

## What got delivered

### Architecture
- **Cluster-as-node placement primitive** (the standing JMT directive): clusters now participate in placement as nodes at the same hierarchical layer for FR/KK/FA2/SFDP via the new `ClusterAwareDriver` op.
- **Single source of truth for cluster bbox computation**: `dagua/layout/ops/cluster_geometry.py` — used by both placement and render.
- `LayoutConfig.cluster_aware = True` default-on; `--use-dagua-placement` flag added to `scripts/graphviz_theme_comparison.py`.

### Render
- **Cluster rectangles draw correctly** on all 8 cluster fixtures. Top edges, sibling separation, concentric nesting, label readability all working.
- **Cluster labels** are top-center for graphviz themes, with opaque @background masks rendered above node fills.
- **Edge clipping at cluster perimeter** for source-out/target-in (and reverse) cases.
- **Bypass edges** segmented for source-out/target-out cases that pass through foreign clusters (partial — see DEFERRED.md).

### Instrumentation
- `cluster_rect_missing` metric in `parity_metrics.py` — gates rectangle presence per panel. Currently 41/41 in tolerance.

## Final state

### Declarative parity
- 99.27% in-tolerance globally (held stable from before sprint)
- `cluster_rect_missing`: 41/41 = 100%
- All cluster features at 100% lock

### Pixel parity
- Mean L1: ~17.5 RGB/pixel
- Mean SSIM: ~0.76
- (Same render-stack floor as graphviz parity sprint — matplotlib vs Cairo)

### Per-panel cluster verification (all 8 panels)
- nested_clusters: ✓ inner top edges, ✓ Outer Group label readable, ✓ concentric nesting
- cluster_showcase: ✓ Large Cluster rectangle, ✓ concentric nesting
- transformer_block: ✓ FFN top edge, ✓ MHA top edge, ✓ endpoint clipping (bypass partial)
- cross_cluster_edges: ✓ Cluster Y/Z top edges
- deep_nesting_4: ✓ Levels 1-4 concentric
- microservices: ✓ edge bodies visible to data layer
- data_pipeline: ✓ Transform top edge
- flat_many_clusters: ✓ Alpha/Beta/Gamma/Delta top edges

## Iteration log

| Phase | Commit | Outcome | Runtime |
|---|---|---|---|
| Investigation | (Opus subagent) | DESIGN.md with 6 phases proposed | ~10 min |
| 1 | d46cdaf | cluster tree + bbox primitive (pure refactor, pixel L1=0) | 38 min |
| 2 | aed468a | ClusterAwareDriver — recursive placement for FR/KK/FA2/SFDP | 1h 2m |
| Audit (post-2) | dagua_native fell back; harness uses dot positions; layouts otherwise solid | (in-line) | — |
| 3 | 2d7cb4b | render parity (top-center labels, opaque masks, z-order) | 1h 31m |
| 4 | 394c67d | edge clipping at cluster perimeter (endpoint cases) | 53 min |
| Audit | FAIL: rectangles missing on 2 panels, masks too narrow, clipping not engaging | — | 6 min |
| 5 | e5d5e26 | corrective: rectangles draw, label mask via Text bbox, instrument metric | 55 min |
| Audit | PARTIAL: Phase 5 introduced regressions (G1 concentric collapse, G2 stub edges) | — | 6 min |
| 6 | 9e7a06e | corrective: concentric nesting, edge body composition, label z-order, bypass edges, dagua placement audit | 1h 24m |
| Audit | PARTIAL: H1 top edges missing on 5 panels, H2 Outer Group label fragmented | — | 7 min |
| 7 | 82eb897 | render fixes (top edges, label z-order final) | 58 min |
| Audit (final) | PASS / STOP | — | 4 min |

## Accepted residuals (all documented in DEFERRED.md)

1. **Cluster-aware Sugiyama / hierarchical placement** — directed-flow graphs (`transformer_block`, `data_pipeline`, `microservices`) collapse into a ball under FR inner placement when `cluster_aware=True`. This is the dot-algorithm's core difficulty. Separate sprint.
2. **Bypass edge clipping completeness** — Phase 6 added segmented bodies; near-border bypass routes can still read as continuous on transformer_block. Half-day follow-up.
3. **Long-label ellipse_rx kerning gap** — pre-existing matplotlib TextToPath vs Cairo residual on labels ≥10 chars. Already accepted residual from earlier sprint.
4. **MED cosmetic polish items** — top-left vs top-center sibling labels, harness containment artifacts, etc. Future cosmetic sprint if desired.

## Commits this sprint

```
82eb897 feat(cluster): phase 7 — render fixes (top edges, label z-order final)
9e7a06e feat(cluster): phase 6 — corrective (concentric nesting, edge body composition, label z-order, bypass edges, dagua placement audit)
e5d5e26 feat(cluster): phase 5 — corrective fixes (rectangle drawing, label mask, edge clip wiring, instrument gap)
394c67d feat(cluster): phase 4 — edge clipping at cluster perimeter
2d7cb4b feat(cluster): phase 3 — render parity (top-center label, universal background mask)
aed468a feat(cluster): phase 2 — ClusterAwareDriver (recursive cluster-as-node placement)
d46cdaf feat(cluster): phase 1 — cluster tree + placement bbox primitive (pure refactor)
```

## Status

**DONE.** Sent to JMT via iMessage with cluster panel side-by-sides.
