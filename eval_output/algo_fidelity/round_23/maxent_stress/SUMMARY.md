# Round 23 maxent_stress Summary

## Scope

Family: `classic_maxent_stress` vs `ogdf_stress`.

Baseline and post-fix comparisons used the bounded 5-graph subset:
`linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels`
with `--seeds 3`.

## Measurement

- Baseline: median `0.000000`, p95 `0.000106`, worst `mixed_width_labels 0.000133`.
- Post-fix: median `0.000000`, p95 `0.000107`, worst `mixed_width_labels 0.000134`.

The default subset was already effectively identical. Round 23 changes are edge-case
fidelity fixes and parameter plumbing rather than expected median-RMSD improvements.

## Ranked Items

1. Runner precision: already applied in Round 22. No Round 23 change.
2. OGDF stress iterations: already applied in Round 22. No Round 23 change.
3. Dagua step variants on majorization: already applied in Round 22. No Round 23 change.
4. OGDF-compatible PivotMDS initialization: partially applied in `dd20365`.
   - Added deterministic first-pivot plumbing to `layout_pivot_mds_pipeline`.
   - `MaxentInitializePositions(for_majorization=True)` now requests first pivot `0`.
   - Deferred exact OGDF SVD/power-iteration parity because it is larger than the
     round cap and would require reproducing OGDF's randomized power iteration.
5. OGDF path fast path: applied in `667a068`.
   - Added simple-path detection and raw line warm start for maxent majorization.
   - Added regression coverage.
6. Disconnected-component initialization: skipped/deferred.
   - Estimated size is M/L and likely over 200 net lines because OGDF's
     `ComponentSplitterLayout` packing behavior needs component-wise PivotMDS plus
     deterministic packing parity.
7. Majorization distance float64 round-trip: already applied in Round 22. No Round 23 change.
8. Weighted wrappers/runner: partially applied in `dd20365`; runner side skipped.
   - Direct `ClassicMaxentStress` now forwards `graph.edge_weights`.
   - OGDF runner weighted parsing was attempted, but rebuild failed because local OGDF
     headers are missing `ogdf/basic/internal/config_autogen.h`. Per spec fallback,
     runner-side changes were not kept.
9. Entropy variants using `ogdf_stress`: skipped as interpretation/metadata only.
   - Existing variant metadata already marks these as non-true-original proxy comparisons.
   - A true entropy reference implementation is outside the small-fix cap.
10. Specialized `ClassicMaxentStress` wrapper params: applied/verified in `dd20365`.
    - Direct wrapper now forwards edge weights.
    - Variant parameter routing already uses `_ClassicBase.layout_with_variant`.

## Commits

- `667a068` `feat(fidelity): round 23 maxent_stress -- warm start parity`
  - `dagua/layout/ops/maxent_stress.py`
  - `tests/test_layout/test_maxent_stress_fidelity.py`
- `dd20365` `feat(fidelity): round 23 maxent_stress -- pivot plumbing`
  - `dagua/eval/competitors/classic_competitor.py`
  - `dagua/layout/ops/pipelines/pivot_mds.py`

## Tests

- `ruff check dagua/layout/ops/maxent_stress.py dagua/layout/ops/distance.py dagua/layout/ops/pipelines/pivot_mds.py dagua/eval/competitors/ogdf_competitor.py dagua/eval/competitors/classic_competitor.py tests/test_layout/test_maxent_stress_fidelity.py --fix`
  - Passed.
- `pytest tests/test_layout/test_maxent_stress_fidelity.py -q`
  - Passed: `7 passed in 0.05s`.
- `pytest tests/test_layout/ -x --tb=short -q -k "maxent_stress"`
  - Passed: `7 passed, 327 deselected in 0.24s`.
- OGDF runner rebuild:
  - Failed: `/home/jtaylor/projects/_references/ogdf/include/ogdf/basic/internal/config.h:34:10: fatal error: ogdf/basic/internal/config_autogen.h: No such file or directory`.
  - Runner-side weighted support was reverted/skipped per spec fallback.

## Concerns

- The post-fix p95/worst changed only by `~1e-6`; this is below the level where the
  bounded default subset is a useful signal for the applied edge-case fixes.
- Full OGDF PivotMDS parity remains incomplete: exact SVD/power iteration and component
  splitter packing are still open larger tasks.
- The workspace contains unrelated dirty/staged files from parallel family and cosmetic
  work; the maxent commits were created with an isolated temporary index to avoid staging
  those changes.
