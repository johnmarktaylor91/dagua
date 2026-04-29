# Phase 7 Cluster Sprint Report

## Changes

- H1: Added a render-time minimum cluster bbox height in `dagua/render/mpl.py` so top caps cannot collapse a cluster below its label band plus member content. Parent containment still expands parents around child boxes instead of shrinking children to zero-height paths.
- H2: Raised cluster labels to a stable layer above node fills and made cluster label background masks render at the same z-order as their glyphs in `dagua/render/text/collection.py`.
- Tests: Added regressions in `tests/test_render/test_cluster_label.py` for cluster label masks above node fills and top-cap height preservation.

## Visual Verification

Regenerated:

```bash
python scripts/parity_pixel_diff.py --hires nested_clusters,transformer_block,cross_cluster_edges,data_pipeline,flat_many_clusters
```

Inspected Dagua-side hi-res outputs:

- `nested_clusters`: Right Branch and Left Branch top strokes are visible around label masks. `Outer Group` is fully readable over node A.
- `transformer_block`: Feed-Forward Network top stroke is visible around the label mask.
- `cross_cluster_edges`: Cluster Y and Cluster Z top strokes are visible around label masks.
- `data_pipeline`: Transform top stroke is visible around the label mask.
- `flat_many_clusters`: Alpha, Beta, Gamma, and Delta top strokes are visible around label masks.

## Metrics

`python scripts/parity_metrics.py`:

- Overall: `7358/7412` features in tolerance, `99.27%`.
- `cluster_rect_missing`: `41/41` in tolerance, `100.00%`.

## Tests

```bash
ruff check . --fix
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_render/ -x --tb=short -q
pytest tests/test_layout/ tests/test_render/ tests/test_style.py tests/test_parity_metrics.py -x --tb=short -q
```

Results:

- Ruff: passed.
- Mypy: `Success: no issues found in 1 source file`.
- Render tests: `136 passed, 2 warnings`.
- Required final pytest slice: `399 passed, 8 warnings in 1276.32s`.

## Assumptions

- The H1 corrective was kept to render bbox construction after top caps. I did not change placement or sibling ordering.
- The H2 corrective was limited to cluster label background masks; non-cluster text backgrounds keep their prior z-order offset.

## Concerns

- H3 remains out of scope: cluster-aware placement on hierarchical-flow graphs still needs the separate Sugiyama-focused sprint.
- The required pytest slice was slow because another local Codex process was running a long algo-fidelity comparison in the same checkout.
