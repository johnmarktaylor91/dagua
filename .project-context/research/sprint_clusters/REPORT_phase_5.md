# Phase 5 Corrective Round Report

## Per-Fix Outcome

- F1: Fixed/verified. `nested_clusters` inner cluster rectangles render as complete boxes after render-time bbox caps preserve finite, positive cluster paths.
- F2: Fixed/verified. `cluster_showcase` "Large Cluster With Longer Label" renders a complete rectangle; the label mask intentionally hides the top stroke behind the label.
- F3: Fixed. Text backgrounds now use Matplotlib `Text.get_window_extent()` for non-rotated plain text masks, with graphviz theme padding raised to `(6.0, 4.0)`.
- F4: Partially fixed. Endpoint cluster crossings now move the rendered edge curve and arrowhead to the clipped perimeter instead of leaving arrowheads at internal node boundaries. True bypass crossings remain structurally limited because the edge collection has one continuous body curve and cannot represent a middle gap without a follow-up multi-segment edge body.
- F5: Fixed for render expansion/harness visibility. Cluster-aware render bboxes cap min-width and label-fit expansion to the placement footprint plus 2pt horizontally, and `scripts/graphviz_theme_comparison.py` now has `--use-dagua-placement`.
- F6: Fixed. `scripts/parity_metrics.py` reports `cluster_rect_missing`; current result is `41/41` in tolerance.

## Before/After Visual Observations

- `nested_clusters`: before audit noted inner boxes as fragments and labels bisected; after render shows full inner rectangles and clean label masks. Outer label/node overlap remains driven by the Graphviz-position harness.
- `cluster_showcase`: before audit reported the large rectangle missing; after render shows the large rectangle present with masked top stroke.
- `transformer_block`: before arrowheads landed inside cluster boxes; after endpoint-clipped arrowheads terminate at perimeters. Bypass edges crossing through foreign clusters remain a Phase 6 candidate.
- `cross_cluster_edges`: rectangles remain present; sibling overlap is reduced by render bbox caps but still visible under Graphviz-position injection.
- `deep_nesting_4`: all nested rectangles draw; label masks now clear the top strokes.

## Deviations

- F4 bypass clipping is documented as structurally incomplete rather than hidden. A full fix should add segmented/gapped edge bodies or perimeter-aware rerouting.

## Verification

- `python scripts/parity_metrics.py`: pass, 99.27% in tolerance, `cluster_rect_missing` 100%.
- `python scripts/parity_pixel_diff.py`: pass, mean SSIM 0.761140; cluster panel SSIM nudged up on `nested_clusters` and `transformer_block`.
- `pytest tests/test_layout/ tests/test_render/ tests/test_style.py tests/test_parity_metrics.py -x --tb=short -q`: pass, 395 passed, 8 warnings.
- `ruff check . --fix`: pass.
- `mypy --follow-imports=silent dagua/cli.py`: pass.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: pass, 270 passed, 6 warnings.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`: blocked by pre-existing `tests/test_classic_drl.py` import error for `layout_drl`.
