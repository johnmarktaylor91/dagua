# Cluster Sprint — Deferred Issues

Items tabled for future sprints. None of these are blocking the current cluster sprint's "render bulletproof" goal — they are larger-scope items that require their own focused work.

## High-priority deferrals

### Cluster-aware Sugiyama / hierarchical placement (originally Phase 5 in DESIGN.md, plus AUDIT_FINAL H3)

**Problem:** Dagua's cluster-aware placement currently uses force-directed inner placement (FR) for all algorithms, including ones that should be hierarchical. On directed-flow graphs (`transformer_block`, `data_pipeline`, `microservices`, `flat_many_clusters`), the FR inner placement collapses contained nodes into a ball or short line — they don't lay out as the user expects.

**Why deferred:** Cluster-aware Sugiyama is a known-hard problem (the actual graphviz dot algorithm). It requires:
- Cluster-aware rank assignment (all members of a cluster occupy a contiguous rank range)
- Within-rank ordering must keep cluster members contiguous in the parent layer's sibling order
- Rank assignment loop must be cluster-bbox-aware

**Files to touch (per DESIGN.md §6 Phase 5):**
- `dagua/layout/ops/layering.py` — `assign_layers_with_cluster_constraints` op variant
- `dagua/layout/ops/ordering.py` — within-rank cluster contiguity constraint
- `dagua/layout/ops/pipelines/sugiyama.py` — wire cluster-aware variants

**Verification target:**
- `transformer_block` with `algorithm="sugiyama"` produces all MHA cluster nodes within a contiguous y-range, all FFN cluster nodes likewise, no interleaving.
- Edge crossings remain reasonable.

**Risk:** HIGH — Sugiyama with cluster constraints is the dot-algorithm's core difficulty.

**Estimated scope:** separate sprint, multi-day.

### Bypass edge clipping completeness (Phase 4 + Phase 6 partial)

**Problem:** Phase 4 added edge clipping at cluster perimeters for source-out/target-in (and reverse). Phase 6 added segmented bodies for source-out/target-out edges that pass through a foreign cluster. But near-border bypass routes can still read as continuous in some cases.

**Why deferred:** the bypass-edge segmenting requires either (a) more sophisticated path-segmenting in matplotlib (multiple bezier patches per edge), or (b) clip-path approach. Both are non-trivial and the audit said "improved but watch list."

**Files involved:** `dagua/edges.py`, `dagua/render/edges/collection.py`.

**Verification target:** transformer_block bypass edges visibly segmented or rerouted around foreign clusters with no continuous path through cluster interiors.

**Estimated scope:** half-day to full-day work, lower risk.

### Cluster-rect-missing metric extension to edges

**Problem:** `cluster_rect_missing` metric (added in Phase 5) catches missing cluster RECTANGLES but not missing or visibly-broken edge bodies (Phase 5/6 had edge body regressions that the metric reported as 100% passing).

**Why deferred:** instrument-side improvement, not user-facing.

**Fix path:** add `edge_body_visible` feature to `parity_metrics.py` that asserts each edge has at least N pixels of visible body between its endpoints (not just the arrowhead).

**Estimated scope:** ~2 hours.

## Medium-priority deferrals

### MED findings from AUDIT_FINAL.md

The audit lists 6 MED + 4 LOW findings beyond H1-H3. Most are cosmetic polish on cluster panels. Worth a future cosmetic sprint if user wants to push past ceiling.

### Per-edge `arrowsize` attribute extraction completeness

Phase 5+6 may not fully wire all edge-attribute paths through to the render. arrow_types panel still shows some arrow_width inconsistencies.

### Long-label ellipse_rx kerning gap

From the graphviz parity sprint — matplotlib TextToPath doesn't apply kerning, so labels >=10 chars accumulate ~0.3pt/char width gap vs Cairo. Two prior phases attempted, both reverted (broad regression). Not currently in the metric's out-of-tolerance list (already accepted as render-stack residual).

## Low-priority deferrals

### Cosmetic polish items from prior cosmetic-parity sprint

Documented in `~/.claude/projects/-home-jtaylor-projects-dagua/memory/` and `.project-context/knowledge/visual_tuning_workflow.md`. The parity sprint hit 95.74% in tolerance; the remaining 4.26% is matplotlib-vs-Cairo render-stack floor (font hinting, AA, B-spline routing).

## Process notes

This sprint had 7 phases (Phases 1-4 planned, Phase 5+6+7 corrective rounds for regressions and audit-found defects). Each round took ~1-1.5h codex runtime. Audits found new findings each round — some real, some metric-extraction artifacts.

**Anti-flail signal:** Phase 7 must close H1+H2 cleanly. If a future audit still finds the same defect class on the same panels after Phase 7, mark as `principled residual` and stop — the loop has converged on what theme/render code can fix without entering the deferred algorithm-scope work.
