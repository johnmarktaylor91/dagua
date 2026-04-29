# Phase 3 Report: Render Parity Polish

## Changes

- `dagua/styles.py`
  - Added cluster label mask style fields: `label_background`,
    `label_background_opacity`, `label_background_padding`, and
    `label_background_corner_radius`.
  - Set `GRAPHVIZ_STRICT_THEME.cluster_style` and `GRAPHVIZ_THEME.cluster_style`
    to `label_position="top-center"`, `label_background="@background"`,
    `label_background_padding=(4.0, 2.0)`, and
    `label_background_opacity=1.0`.
  - Preserved the global `ClusterStyle.label_position="top-left"` default.

- `dagua/render/mpl.py`
  - Added `_resolve_cluster_label_background()` for resolving explicit mask
    colors and the `@background` sentinel to `graph.graph_style.background_color`.
  - Generalized cluster-label background masks to any theme that opts in via
    `ClusterStyle.label_background`.
  - Raised cluster-label z-order so the background patch paints above the same
    cluster border and masks stroke-through-label artifacts.

- `tests/test_render/test_cluster_label.py`
  - Added regression tests for top-center cluster labels, `@background` mask
    resolution, and graphviz-strict nested-cluster label masking.

- `tests/test_render/test_mpl.py`
  - Updated label-patch helpers/assertions to distinguish text glyph patches
    from the new background mask patches.

- `tests/test_style.py`
  - Locked graphviz theme cluster label defaults and preserved the global
    top-left default.

## Assumptions

- I kept `ClusterStyle.label_background` default empty rather than globally
  enabling `@background`. The task text asks graphviz themes to opt in and says
  the dagua default label-position choice should not change; keeping the mask
  opt-in is the conservative interpretation.
- I skipped DESIGN §5.6 Option B path breaks. The opaque mask now matches dot's
  painter-order behavior for raster targets, and SVG path splitting remains a
  follow-up.

## Verification

- `ruff check . --fix` passed.
- `mypy --follow-imports=silent dagua/cli.py` passed.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q` passed:
  `270 passed, 6 warnings`.
- `pytest tests/test_render/ tests/test_style.py tests/test_parity_metrics.py -x --tb=short -q`
  passed: `158 passed, 2 warnings`.
- `pytest tests/test_layout/ tests/test_render/ tests/test_style.py tests/test_parity_metrics.py -x --tb=short -q`
  passed: `391 passed, 8 warnings`.
- `python scripts/graphviz_theme_comparison.py --output-dir eval_output/cluster_phase_3_check`
  completed and wrote 45 comparison rows.
- Visual check:
  - `nested_clusters`: graphviz themes use top-centered labels with visible
    masks; strict still shows placement/routing differences outside Phase 3.
  - `cluster_showcase`: graphviz-theme labels are top-centered; improved theme
    masks strokes clearly under large labels.
  - `transformer_block`: graphviz-theme labels are top-centered and readable;
    remaining edge/cluster perimeter routing is Phase 4 scope.
- `python scripts/parity_metrics.py` passed the parity floor:
  `in-tolerance 7317/7371 = 99.27%`.

## Known Gate Blockers

- `pytest tests/ --ignore=tests/test_classic_drl.py -x --tb=short -q -m "not slow and not benchmark and not rare"`
  fails during collection before this render change is exercised:
  `ImportError: cannot import name 'layout_fa2' from 'dagua.layout.classic'`.
- Ignoring `test_classic_fa2.py` then fails on `test_classic_fr.py` for the
  same missing `dagua.layout.classic` exports.
- Ignoring all `test_classic_*.py` then fails on `tests/test_layout_ops.py`:
  `ImportError: cannot import name 'MultilevelVCycle' from 'dagua.layout.ops'`.
- I did not change these packaging/export issues because they are outside the
  Phase 3 render parity scope.

## Controversial Choices

- The cluster label spec z-order changed from `0.12 + depth*0.01` to
  `0.16 + depth*0.01` so text backgrounds render at `0.06 + depth*0.01`,
  above same-depth cluster borders at `0.05 + depth*0.01`.
- Graphviz-style themes inherit the new mask through the normal style cascade;
  tests that counted all label-related patches now explicitly count glyph
  patches separately from background masks.

## Concerns

- The full non-slow suite has multiple collection-time import/export failures
  unrelated to this render work, so it cannot currently be used as a final green
  gate without broader cleanup.
- Phase 4 edge clipping is still needed for visible edge/cluster perimeter
  differences in some comparison panels.

## Knowledge

- `top-center` cluster label anchoring already existed in
  `_cluster_label_anchor`; this phase only needed theme defaults and mask
  generalization.
- `DaguaText` background patches render at `spec.zorder - 0.1`, so callers must
  account for that when using backgrounds as masks over other artists.
