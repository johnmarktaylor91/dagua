# Round 62 UMAP Layout Real Port

## Verdict

The composable UMAP layout pipeline no longer delegates to `umap-learn`.
`dagua/layout/ops/pipelines/umap_layout.py` now always runs the native op
pipeline, and `dagua/layout/ops/umap.py` contains the ported fuzzy simplicial
set, spectral initialization, and Euclidean SGD optimizer pieces.

Smoke checks against `umap.UMAP(metric="precomputed")` reached bit-exact output
for the measured graph-distance cases:

| Case | Seed | Max raw diff | Procrustes RMSD |
| --- | ---: | ---: | ---: |
| path-5 | 42 | 0.0 | 1.1041893068757026e-16 |
| path-10 | 42 | 0.0 | 9.848586031720603e-17 |
| path-12 | 42 | 0.0 | 1.0935997856313817e-16 |
| weighted-6 | 17 | 0.0 | 5.25404566686952e-17 |

## Source Ported

- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/umap_.py`:
  `smooth_knn_dist`, `compute_membership_strengths`, `fuzzy_simplicial_set`,
  `find_ab_params`, `make_epochs_per_sample`, and `simplicial_set_embedding`
  setup semantics.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/spectral.py`:
  normalized Laplacian construction, ARPACK parameters, float32 degree handling,
  random initialization advancement, and noisy coordinate scaling.
- `/home/jtaylor/anaconda3/envs/py311/lib/python3.11/site-packages/umap/layouts.py`:
  `optimize_layout_euclidean` epoch scheduling, `move_other=True` pair updates,
  gradient clipping, and taus88 negative-sampling RNG semantics.

## Fidelity Notes

- The fitted `a,b` curve parameters must use `curve_fit(curve, xv, yv)` with
  SciPy defaults. Adding `maxfev=10000` changes the last decimal places and
  desynchronizes chaotic SGD trajectories.
- Spectral initialization must preserve umap-learn's float32 degree vector.
  Promoting the inverse degree vector to float64 changes ARPACK output enough
  to desynchronize negative-sampling states.
- Negative sampling must use numba's declared `int32` return cast for
  `tau_rand_int` before modulo. Python-level signed conversion gives different
  negative sample indices.

## Verification

- `rg -n "import umap|from umap" dagua/layout/ops/pipelines/umap_layout.py dagua/layout/ops/umap.py`: no matches.
- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/test_umap_fidelity.py -q`: `12 passed, 2 warnings`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: failed after
  `389 passed` on the now-stale expectation that `_fit_ab` passes
  `maxfev=10000`; the test was updated afterward to match umap-learn source
  defaults.
- `pytest tests/test_pipeline_umap_layout.py -q`: failed because those tests
  still compare the native real port to `dagua.layout.classic.umap_layout`,
  which is a historical normalized implementation rather than the umap-learn
  reference.
