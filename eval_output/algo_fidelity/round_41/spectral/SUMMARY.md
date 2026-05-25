# Round 41 Spectral Fidelity Summary

## Reference source lines identified

- `/home/jtaylor/projects/_references/igraph/src/properties/spectral.c:128-146`:
  normalized Laplacians write a diagonal `1` only when degree is positive.
  Isolated vertices keep a zero diagonal.
- `/home/jtaylor/projects/_references/igraph/src/properties/spectral.c:154-181`:
  edge contributions are applied in graph edge order, with undirected mode
  adding the reverse contribution.
- `/home/jtaylor/projects/_references/igraph/src/misc/embedding.c:698-717`:
  ARPACK setup uses `nev = no` and `ncv = min(nev + 3, n)` for the embedding
  solve.

## Sub-component diagnosis

Dominant residual was disconnected/isolated normalized-Laplacian handling plus
zero-mode selection. Dagua's existing default builds normalized Laplacians from
a full identity and filters all near-zero eigenvalues. The igraph-compatible
reference keeps isolated normalized diagonals at zero and skips only the first
trivial eigenvector, retaining additional zero modes for disconnected graphs.

RNG, node order, edge order, and force kernels are not active for this dense
smoke path. Spectral is deterministic for these cases.

## Port implementation summary

- Added opt-in `fidelity_mode="igraph"` / `fidelity_mode=True` to
  `layout_spectral_pipeline()` and `build_spectral_pipeline()`.
- Routed igraph fidelity through `SpectralPrepareState` to use igraph's
  normalized isolated-vertex diagonal convention.
- Routed igraph fidelity through `SpectralEmbed` to skip only the first
  eigenvector and to use igraph-style sparse ARPACK `ncv = min(k + 3, n)`.
- Left default and `networkx_fidelity=True` behavior unchanged.

## Before/after smoke RMSD

Smoke harness: path/star/clustered/grid, seeds 0/1/2, Procrustes RMSD against a
source-derived igraph symmetric-normalized Laplacian reference. The clustered
case includes one isolate to exercise the diagnosed divergence.

| Topology | Seed | Before | After |
|---|---:|---:|---:|
| path | 0 | 0.000000004 | 0.000000004 |
| path | 1 | 0.000000004 | 0.000000004 |
| path | 2 | 0.000000004 | 0.000000004 |
| star | 0 | 0.000000004 | 0.000000004 |
| star | 1 | 0.000000004 | 0.000000004 |
| star | 2 | 0.000000004 | 0.000000004 |
| clustered | 0 | 0.174393428 | 0.000000003 |
| clustered | 1 | 0.174393428 | 0.000000003 |
| clustered | 2 | 0.174393428 | 0.000000003 |
| grid | 0 | 0.000000004 | 0.000000004 |
| grid | 1 | 0.000000004 | 0.000000004 |
| grid | 2 | 0.000000004 | 0.000000004 |
| mean | - | 0.043598360 | 0.000000004 |

## Final verdict

Bit-exact for the dense smoke target within float32 tensor readback noise. The
observed post-port floor is approximately `0.000000004` RMSD, from converting
the float64 eigensolver output to the pipeline's public float32 positions.

## Verification notes

- `ruff check . --fix`: blocked by an unrelated existing line-length issue in
  `dagua/layout/ops/init.py:551`.
- `ruff check dagua/layout/ops/pipelines/spectral.py dagua/layout/ops/preprocess.py dagua/layout/ops/embed.py tests/test_layout/test_spectral_fidelity.py --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_layout/test_spectral_fidelity.py tests/test_pipeline_spectral.py -x --tb=short -q`: passed, `23 passed`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: did not complete in this session; the process was terminated after partial progress output.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  blocked during collection by unrelated `tests/test_classic_drl.py` import
  failure: `cannot import name 'layout_drl' from 'dagua.layout.classic'`.
