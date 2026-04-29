# Phase 2 Report — ClusterAwareDriver

## Outcome

- Implemented `dagua/layout/ops/cluster_driver.py` with `ClusterAwareDriver`.
- Added recursive placement over `ClusterTree`: child clusters are solved first, then parent
  levels place direct leaves plus rigid child-cluster placeholders.
- Added rectangle overlap projection and parent-level external-edge clearance projection for
  direct leaf-to-child-cluster edges.
- Added `LayoutConfig.cluster_aware=True`, `cluster_side_padding_pt`,
  `cluster_label_band_pt`, and `cluster_external_clearance_pt`.
- Wired engine wrapping for supported inner pipelines: `fr`, `kk`, `fa2`, `sfdp`.
- Gated `dagua_native` off after it failed on a layered ordering precondition inside recursive
  subproblems. The engine falls back to flat placement with a `RuntimeWarning`.
- Added `ClusterTree.bottom_up_order()` and `top_down_order()`.
- Added `DeprecationWarning` helper for non-default legacy containment weight with
  `cluster_aware=True`.

## Tests

- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/test_cluster_driver.py tests/test_layout/test_cluster_geometry.py -x --tb=short -q`:
  passed, `14 passed`; warnings only for expected `dagua_native` fallback in legacy geometry tests.
- `pytest tests/test_layout/ -x --tb=short -q`: passed, `233 passed`; warnings:
  expected `dagua_native` fallback plus existing `ConstantInputWarning`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  failed during collection before cluster tests:
  `ImportError: cannot import name 'layout_drl' from 'dagua.layout.classic'`.
  Root cause appears unrelated to this phase: `dagua/layout/classic/` lacks `__init__.py` while
  archived classic exports exist under `dagua/layout/_archive/classic/__init__.py`. I did not patch
  this because it is outside the Phase 2 cluster scope.

## Visual Check

Command run:

```bash
python scripts/graphviz_theme_comparison.py --output-dir eval_output/cluster_phase_2_check
```

- `nested_clusters`: Dagua strict panel still shows node `A` overlapping the outer cluster top and
  child cluster boxes visually touching/overlapping. This command renders Dagua theme on Graphviz
  positions, so the remaining defect is render-bbox parity, not the recursive driver output.
- `cluster_showcase`: Dagua strict panel still shows several rendered cluster bboxes overlapping
  on Graphviz positions. Same render-path limitation as above.
- `transformer_block`: no sibling-cluster overlap observed; the `Add` node sits just outside the
  MHA cluster and between MHA/FFN.

Parity:

```text
in-tolerance %: 99.27%
```

## Deviations

- The "all 23 pipelines" target is not complete in this round. Recursive wrapping is enabled for
  FR/KK/FA2/SFDP and explicitly gated for other algorithms.
- `dagua_native` was attempted, failed on recursive subproblem ordering, and was gated off per the
  prompt's medium-risk fallback instruction.
- The requested visual cluster check does not fully pass for `nested_clusters` or
  `cluster_showcase` because the specified comparison script uses Graphviz positions and exercises
  render-side cluster bbox behavior, which is Phase 3 scope.

## Knowledge

- Existing algorithm dispatch resolves layout functions; Phase 2 needed a builder bridge to obtain
  op `Pipeline` instances for recursive reuse.
- Classic FR does not make rectangle sizes hard constraints, so the driver must project rectangle
  overlaps after inner placement to make placeholders act as rigid obstacles.
- Render padding can exceed placement padding. The engine now uses effective cluster style padding
  for placeholder footprints when cluster-aware placement is active.
