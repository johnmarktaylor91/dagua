<task>
Phase 2 of the cluster sprint — the core architectural change. Per `/home/jtaylor/projects/dagua/.project-context/research/sprint_clusters/DESIGN.md` §6 Phase 2 + §5.

Read DESIGN.md fully — especially §5 (architecture proposal), §5.2 (recursive placement pseudocode), §5.9 (algorithm-pipeline integration), and §6 Phase 2 (this round's spec).

Repo: `/home/jtaylor/projects/dagua` (already on `develop`). Single working branch.

## Architectural decisions already made (from STATE.md)

1. `cluster_aware=True` default-on (this phase makes it the default for `LayoutConfig`)
2. Cluster placeholder anchor: leaves-centroid (matches `cluster_compactness_loss` semantics)
3. Cross-cluster edges in inner placement: ignored (first sprint pass)
4. External-edge clearance: ~10pt
5. Boolean API: `LayoutConfig(cluster_aware=True)`
6. Backward compat: keep legacy params, warn when set with `cluster_aware=True`

## Goal

Introduce a `ClusterAwareDriver` op that wraps any leaf-placement pipeline into a cluster-aware recursive driver. After this phase, `LayoutConfig(algorithm="fr", cluster_aware=True)` produces a layout where:
- No sibling-cluster bbox overlap (rigid rectangle obstacles in the parent layer)
- External nodes outside the cluster bbox by `cluster_external_clearance` (default 10pt)
- Parent-cluster bbox strictly contains child cluster bboxes (structural guarantee from recursion)

This is the **single integration op** that makes ALL 23 algorithm pipelines cluster-aware, not per-pipeline patches.

## Files to touch

### 1. `dagua/layout/ops/cluster_driver.py` (new)

Define `ClusterAwareDriver`:

```python
@dataclass(frozen=True)
class ClusterAwareDriver:
    """Wraps a leaf-placement pipeline into recursive cluster-aware placement.

    For each cluster (bottom-up), runs the inner pipeline on its leaves +
    already-placed child cluster placeholders. After the recursion completes,
    translates all descendants by the cluster's final placement.

    Parameters
    ----------
    inner_pipeline : Sequence[Op]
        Ops to apply at each cluster level. Same ops, called recursively.
    side_padding_pt : float, default=8.0
        Padding inside cluster bbox (left/right/bottom).
    label_band_pt : float, default=26.0
        Reserved at top of cluster bbox for label.
    external_clearance_pt : float, default=10.0
        Extra padding all sides for external-node-to-cluster gap.
    cluster_compactness_weight : float, default=1.0
        Strength of the within-cluster compactness force during inner placement.
    """

    inner_pipeline: Sequence[Op]
    side_padding_pt: float = 8.0
    label_band_pt: float = 26.0
    external_clearance_pt: float = 10.0
    cluster_compactness_weight: float = 1.0

    def apply(self, problem: LayoutProblem, state: SolveState) -> SolveState:
        # Build cluster_tree from problem if not set
        tree = problem.get_cluster_tree() or _build_trivial_tree(problem)

        # Bottom-up recursion: place each cluster's interior, compute its bbox,
        # then place its parent layer with the child as a rigid placeholder.
        for cluster_name in tree.bottom_up_order():
            sub_problem = self._build_subproblem(problem, state, tree, cluster_name)
            sub_state = self._run_inner(sub_problem, state)
            inner_pos, inner_bbox = self._extract_inner_placement(sub_state, sub_problem)
            placeholder = self._make_cluster_placeholder(
                inner_pos, inner_bbox, label_metrics, ...
            )
            state.cluster_placements[cluster_name] = ClusterPlacement(
                inner_pos=inner_pos, anchor=placeholder.anchor, bbox=placeholder.bbox
            )

        # Top: place root layer with all top-level clusters as placeholders + true root nodes
        root_set = build_root_placement_set(tree, state.cluster_placements)
        root_pos = run_inner_pipeline(root_set, problem.edges_at_root_level)

        # Translate each cluster's descendants by (final_root_pos - inner_anchor)
        for cluster_name in tree.top_down_order():
            translation = root_pos[cluster_name] - state.cluster_placements[cluster_name].anchor
            translate_cluster_descendants(state, tree, cluster_name, translation)

        return state
```

Use `compute_cluster_placement_bbox` from Phase 1's `cluster_geometry.py` for the bbox math.

For now, edges:
- During internal placement: include only edges where BOTH endpoints are in this cluster's leaves-only-at-level set (or in immediate-child-cluster placeholder positions)
- External-to-cluster edges: ignored during internal placement (per §5.8 first-sprint rule)
- After full recursion completes, all edges become valid for post-processing (edge optimization, render)

### 2. `dagua/config.py` — add `cluster_aware` flag

```python
@dataclass
class LayoutConfig:
    ...
    # Cluster-aware placement (Phase 2)
    cluster_aware: bool = True
    cluster_side_padding_pt: float = 8.0
    cluster_label_band_pt: float = 26.0
    cluster_external_clearance_pt: float = 10.0
```

When `cluster_aware=True` and `graph.clusters` is non-empty, the engine wraps the chosen pipeline in `ClusterAwareDriver`.

When `cluster_aware=False`, the legacy path runs unchanged (cluster-compactness/separation/containment losses still wired in the native pipeline as today).

### 3. `dagua/layout/engine.py` — engine entry detects cluster_aware

In `_layout_inner` or wherever the pipeline is dispatched:
```python
inner_pipeline = build_pipeline(config.algorithm, ...)
if config.cluster_aware and graph.clusters:
    pipeline = [ClusterAwareDriver(inner_pipeline, ...)]
else:
    pipeline = inner_pipeline  # legacy path
```

### 4. Tests

In `tests/test_layout/test_cluster_driver.py` (new):
- Build a graph with `nested_clusters` topology (outer + 2 sibling sub-clusters), apply `cluster_aware=True` with `algorithm="fr"`. Verify:
  - Sibling cluster bboxes don't overlap
  - Outer cluster bbox strictly contains both children's bboxes
  - External nodes (those above the outer cluster) are at least `external_clearance_pt` above the outer cluster top
- Same for `cluster_showcase` topology
- Bottom-up ordering: change a deep-nested cluster, verify all parent bboxes update consistently
- `cluster_aware=False`: verify legacy path still produces the same result as before this phase (no regression in non-cluster-aware mode)

Also extend `tests/test_layout/test_cluster_geometry.py` with cluster-tree bottom-up ordering tests.

### 5. Backward-compat warnings

In `dagua/layout/constraints.py`, add a one-line `warnings.warn(...)` when `w_cluster_separation` or `w_cluster_containment` is set non-default AND `cluster_aware=True`. (Tag with `DeprecationWarning`.)

## Verification

1. **New tests pass**: `pytest tests/test_layout/test_cluster_driver.py tests/test_layout/test_cluster_geometry.py -x --tb=short -q`.
2. **No regression on existing layout tests**: `pytest tests/test_layout/ -x --tb=short -q`. (~225 passed at Phase 1 commit.)
3. **Visual cluster check on representative panels** (nested_clusters, cluster_showcase, transformer_block):
   - Run `python scripts/graphviz_theme_comparison.py --output-dir eval_output/cluster_phase_2_check`.
   - Read each cluster panel's `dagua_strict/<slug>.png`.
   - Visually confirm: no sibling-cluster bbox overlap, no external-node-cluster collision.
   - Pixel L1 vs Phase 1 baseline: expected to CHANGE (this phase changes layout). Document the diff.
4. **Parity metric stays >= 95%**: `python scripts/parity_metrics.py` should still report ≥ 95% in tolerance globally (we may briefly dip on cluster panels but rest should hold).

## Out of scope

- Cluster render path (Phase 3)
- Edge clipping at cluster perimeter (Phase 4)
- Sugiyama+clusters (Phase 5, separate sprint)
- Don't deprecate the old loss code yet (just warn)

## Completeness contract

Not done until:
1. `ClusterAwareDriver` op implemented with full type hints + NumPy docstrings.
2. `LayoutConfig.cluster_aware = True` default; engine wraps pipeline accordingly.
3. New cluster-driver tests pass.
4. All `tests/test_layout/` pass.
5. Visual cluster check passes (no overlaps, no protrusions on the 3 panels).
6. ONE commit on develop: `feat(cluster): phase 2 — ClusterAwareDriver (recursive cluster-as-node placement)`.
7. REPORT at `.project-context/research/sprint_clusters/REPORT_phase_2.md` with: per-fix outcome, before/after panel observations, deviations.

## Reply format

Per-step outcome, commit SHA, before/after observations on the 3 cluster panels. ≤250 words.
</task>

<missing_context_gating>
This phase is medium risk because it touches the dispatch path. If the recursive driver breaks an existing layout pipeline (e.g. native pipeline relies on flat-tensor optimization), gate `cluster_aware` to OFF for that pipeline only and document. Don't try to fix all pipelines in one round — get the common case (FR, KK, FA2, SFDP) working first, document gaps for layered algorithms (Sugiyama).
</missing_context_gating>

<action_safety>
develop branch. ONE commit at end.
</action_safety>
