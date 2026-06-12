# Round 64 FR Chaotic Residual Check

## Verdict

No current chaotic-amplification floor was reproduced for FR on the R56 worst
graph/seed pair. The saved R56 artifacts showed a large `classic_fr_steps*`
residual against the NetworkX spring reference on `dense_pair_50`, seed `42`,
but the current implementation is numerically aligned with both the NetworkX
adapter path and the pure igraph fidelity path at high iteration counts.

No fix was applied. This round documents the diagnosis only.

## Scope Note

The task scope named `dagua/layout/ops/fr.py`, but that file does not exist in
this checkout. FR implementation code is in `dagua/layout/ops/pipelines/fr.py`.
No FR runtime code was changed.

## Phase A: R56 Artifact Sweep

R56 saved positions for `dense_pair_50`, seed `42`, had the largest FR residual
against the paired NetworkX adapter:

| Iterations | R56 RMSD |
| ---: | ---: |
| 50 | 2.08456369339e-01 |
| 100 | 1.34324122793e-01 |
| 200 | 1.85493776203e-01 |
| 500 | 2.09807687384e-01 |

ASCII plot, normalized to the maximum R56 RMSD:

```text
  50  ####################
 100  #############
 200  ##################
 500  ####################
```

This is not an exponential growth curve. It was an old large residual already
present at low iteration counts, not a high-iteration basin fork.

## Phase A: Current NetworkX Sweep

Current `layout_fr_pipeline(..., networkx_compat=True)` versus direct
`networkx.spring_layout(..., method="force")` on the same graph and seed:

| Iterations | Procrustes RMSD | Direct RMSD | Scale ratio |
| ---: | ---: | ---: | ---: |
| 50 | 3.11761203286e-09 | 4.11031184897e-06 | 9.99999993594e-01 |
| 100 | 3.28958129094e-09 | 4.50844193017e-06 | 1.00000001060e+00 |
| 200 | 3.30793312251e-09 | 3.75312379019e-06 | 9.99999999170e-01 |
| 500 | 3.04861417779e-09 | 3.19315881773e-06 | 9.99999998483e-01 |
| 1000 | 3.12741794939e-08 | 3.16931813919e-05 | 1.00000000874e+00 |

ASCII plot, normalized to the maximum current Procrustes RMSD:

```text
  50  ##
 100  ##
 200  ##
 500  ##
1000  ####################
```

The absolute values remain well below the `1e-6` bit-exact gate through 500
iterations and only reach `3.13e-08` at 1000.

## Phase B: Igraph High-Precision Check

Direct comparison of the current pure igraph FR reference loop against
python-igraph on `dense_pair_50`, seed `42`, using float64 output and exact
iteration counts:

| Iterations | Procrustes RMSD | Direct RMSD | Scale ratio |
| ---: | ---: | ---: | ---: |
| 50 | 3.17929231097e-17 | 4.94047998515e-15 | 1.00000000000e+00 |
| 100 | 4.62801092083e-17 | 1.69511823991e-15 | 1.00000000000e+00 |
| 200 | 2.15817516457e-17 | 1.73777390727e-15 | 1.00000000000e+00 |
| 500 | 3.22429549555e-17 | 2.38550735799e-15 | 1.00000000000e+00 |
| 1000 | 6.28732538261e-17 | 8.32723621945e-15 | 1.00000000000e+00 |

There is no algorithmic jump and no exponential growth. The pure igraph path
tracks python-igraph at double-precision noise through 1000 iterations.

## Phase C

No code fix and no chaotic floor documentation in runtime docs is needed. The
current FR implementation has already eliminated the R56 residual for the
reproduced graph/seed pair.

One wrapper detail matters for future smoke scripts: public
`layout_fr_pipeline(..., fidelity_mode="igraph", steps=50)` intentionally maps
the unchanged 50-step default to igraph's default `niter=500`. Explicit
iteration-count investigations should call the internal reference loop or pass
non-default counts deliberately.

## Verification

- `git diff -- dagua/layout/ops/pipelines/fr.py dagua/layout/ops/fr.py`: no
  runtime diff; `dagua/layout/ops/fr.py` is absent.
- Runtime delegation grep: no new `igraph`, `subprocess`, or external runtime
  delegation was added in FR code.
- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: passed,
  `435 passed, 8 warnings in 1449.21s`.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  failed during collection on the existing `ImportError: cannot import name
  'layout_drl' from 'dagua.layout.classic'` in `tests/test_classic_drl.py`.
