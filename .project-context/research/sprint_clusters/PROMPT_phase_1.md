<task>
Phase 1 of the cluster sprint per `/home/jtaylor/projects/dagua/.project-context/research/sprint_clusters/DESIGN.md` §6 Phase 1.

Read the design doc fully before starting — it has the full architectural context. Especially read §1-§3 (current state mapping), §5.3 (bbox formula), and §6 Phase 1 (this round's spec).

## Goal

Extract a single source-of-truth helper for cluster bbox computation that both placement (Phase 2) and render (Phase 3) will use. **Pure refactor — no behavior change.** Successful Phase 1 leaves rendered output bit-for-bit identical (or within 2pt rounding) to current.

Repo: `/home/jtaylor/projects/dagua` (already on `develop` branch). Single working branch.

## Files to touch

### 1. New module: `dagua/layout/ops/cluster_geometry.py`

Pure helpers, no graph object dependency. Public API:

```python
@dataclass(frozen=True)
class ClusterTree:
    """Tree representation of cluster hierarchy.

    Parameters
    ----------
    parents : Mapping[str, Optional[str]]
        cluster_name -> parent cluster name (None for root clusters)
    leaves_per_cluster : Mapping[str, frozenset[int]]
        cluster_name -> set of leaf node indices that bottom out in this cluster
        (i.e. NOT in any deeper child cluster within this branch)
    descendants_per_cluster : Mapping[str, frozenset[int]]
        cluster_name -> set of ALL leaf node indices reachable through this cluster
        (matches the existing flat membership convention)
    children_per_cluster : Mapping[str, tuple[str, ...]]
        cluster_name -> tuple of immediate child cluster names
    roots : tuple[str, ...]
        cluster names with parent=None

    Notes
    -----
    Construct from `(clusters: Mapping[str, Sequence[int]], cluster_parents: Mapping[str, Optional[str]])`
    via `ClusterTree.from_flat_membership(...)`. Both args use the existing dagua conventions.
    """

def compute_cluster_placement_bbox(
    inner_positions: torch.Tensor,  # [N_inner, 2] positions of placement-set members at this level
    inner_sizes: torch.Tensor,  # [N_inner, 2] (width, height) of each member
    label_metrics: ClusterLabelMetrics,  # label_width_pt, label_height_pt
    side_padding_pt: float,  # default 8.0 (dot parity)
    label_band_pt: float,  # default label_height + 12 (8 top + 4 bottom)
    extra_top_band_pt: float = 0.0,  # external-edge clearance, optional
) -> ClusterPlacementBox:
    """Compute the placement-time bbox for a cluster from its placed inner members.

    Returns a `ClusterPlacementBox` with:
        width, height : float — full footprint of this cluster as a placement node
        anchor_offset : tuple[float, float] — (dx, dy) from inner_positions centroid to bbox center
        inner_bbox : tuple[float, float, float, float] — (x_min, y_min, x_max, y_max) of contents
        label_band_y_extent : tuple[float, float] — (y_top_of_label_band, y_bottom_of_label_band)

    The function is pure. It does not consult any global state.
    """
```

(Define `ClusterLabelMetrics` and `ClusterPlacementBox` as frozen dataclasses in the same file.)

The label_band_y_extent is needed by Phase 3 render to know where to put the label and where to break the top stroke. For now (Phase 1) it's just metadata in the return value.

Also include:
```python
def cluster_descendants(tree: ClusterTree, name: str) -> frozenset[int]: ...
def cluster_leaves_only_at_level(tree: ClusterTree, name: str) -> frozenset[int]: ...
def cluster_subtree(tree: ClusterTree, name: str) -> tuple[str, ...]:
    """All cluster names in the subtree rooted at `name` (inclusive)."""
```

Add full NumPy-style docstrings, type hints throughout.

### 2. `dagua/layout/ops/state.py` — add `cluster_tree` to `LayoutProblem`

Add an optional field `cluster_tree: Optional[ClusterTree]` to `LayoutProblem`. If both `clusters` and `cluster_parents` are populated and `cluster_tree` is None, construct it lazily at first access via a property/method (must remain thread-safe via simple memoization since LayoutProblem is read by multiple ops).

Don't break existing API — fields are added optional, lazy construction.

### 3. `dagua/render/mpl.py` — refactor existing bbox computation to delegate

Find `_compute_cluster_y_maxes` and `_compute_cluster_y_mins` (around `mpl.py:3607-3775` per the design doc).

Refactor them to call `compute_cluster_placement_bbox` for the bbox math. The render-side wrappers can supply the additional render-only parameters (theme padding, depth-stepped padding) but the core geometry must come from the new helper.

CRITICAL: render output must remain identical. Verify with byte-wise PNG diff (or pixel L1 ≤ small epsilon) on a few representative panels before declaring done.

## Verification

1. **New tests** in `tests/test_layout/test_cluster_geometry.py` (file already exists per the design doc — extend it, don't replace):
   - `ClusterTree.from_flat_membership` round-trip on simple hierarchies (1 cluster, 1 nested cluster, 3 siblings, deep nesting 4 levels).
   - `compute_cluster_placement_bbox` formula sanity: with known inner positions/sizes and label metrics, returns expected width/height/anchor.
   - `cluster_descendants`, `cluster_leaves_only_at_level`, `cluster_subtree` produce expected sets.

2. **Visual regression** on at least 3 cluster panels (nested_clusters, cluster_showcase, transformer_block):
   - Render with current dagua before any of your changes (capture `eval_output/cluster_phase_1_baseline/` via `python scripts/graphviz_theme_comparison.py --output-dir eval_output/cluster_phase_1_baseline/ --quick` if quick mode covers these, otherwise full run).
   - Apply your changes.
   - Re-render to `eval_output/cluster_phase_1_check/`.
   - Confirm pixel L1 difference ≤ 2 between baseline and check on each cluster panel.
   - If pixel L1 differs by >2, the refactor changed behavior — fix or document.

3. **Existing tests pass**: `pytest tests/test_layout/test_cluster_geometry.py tests/test_layout/test_engine.py tests/test_render/ -x --tb=short -q`. Update assertions if you intentionally changed any (you shouldn't have).

4. **Tier-1 parity tests still pass**: `pytest tests/test_parity_metrics.py -x --tb=short -q`.

## Out of scope for Phase 1

- DO NOT change placement behavior. The cluster losses, the engine, the algorithm pipelines remain untouched.
- DO NOT touch `dagua/render/edges/` (Phase 4 territory).
- DO NOT modify `LayoutConfig` or add `cluster_aware` flag (Phase 2 territory).

## Completeness contract

Not done until:
1. New module created with full NumPy docstrings + type hints.
2. `LayoutProblem.cluster_tree` field added (optional, lazy).
3. Render-side bbox computation refactored to delegate.
4. New tests added (≥ 6 in test_cluster_geometry.py).
5. Visual regression confirms pixel L1 ≤ 2 on the 3 cluster panels.
6. All targeted tests pass: `pytest tests/test_layout/test_cluster_geometry.py tests/test_layout/test_engine.py tests/test_render/ tests/test_parity_metrics.py -x --tb=short -q`.
7. ONE commit on develop: `feat(cluster): phase 1 — cluster tree + placement bbox primitive (pure refactor)`.
8. REPORT at `.project-context/research/sprint_clusters/REPORT_phase_1.md` with: per-fix outcome, before/after pixel L1 on the 3 cluster panels, deviations.

## Reply format

Per-step outcome, commit SHA, before/after pixel L1 on the 3 cluster panels. ≤200 words.
</task>

<missing_context_gating>
If `LayoutProblem` is a `@dataclass(frozen=True)` and you can't add a mutable cached field, use a property that lazily constructs the tree on first access (recompute is fine — caching is an optimization, not correctness). Document the choice.
</missing_context_gating>

<action_safety>
Pure refactor. develop branch. ONE commit at end.
</action_safety>
