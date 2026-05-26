# Round 61 FR Real Port

## Verdict

BIT-EXACT verdict for the Procrustes smoke contract: the pure Python/PyTorch
fidelity path reached mean RMSD `4.2370838979e-09`, below the `<1e-6` target.
No runtime delegation to python-igraph, subprocesses, or external services was
added in `dagua/layout/ops/pipelines/fr.py`.

The requested harness path
`eval_output/algo_fidelity/round_41/fr/smoke_check.py` is absent in this
checkout, so the exact command failed with `[Errno 2] No such file or directory`.
The after numbers below come from an equivalent measurement-only script using
the existing python-igraph reference adapter semantics.

## Igraph Source Lines Ported

- `/home/jtaylor/projects/_references/igraph/src/layout/fruchterman_reingold.c:48-50`:
  weak connectivity and disconnected-component constant.
- `/home/jtaylor/projects/_references/igraph/src/layout/fruchterman_reingold.c:65-83`:
  displacement reset, connected all-pairs repulsion order, zero-distance jitter,
  and direct `dx / dlen` arithmetic.
- `/home/jtaylor/projects/_references/igraph/src/layout/fruchterman_reingold.c:87-105`:
  disconnected all-pairs repulsion arithmetic.
- `/home/jtaylor/projects/_references/igraph/src/layout/fruchterman_reingold.c:111-122`:
  edge-order attraction.
- `/home/jtaylor/projects/_references/igraph/src/layout/fruchterman_reingold.c:127-139`:
  displacement jitter, temperature clamp, and position update.
- `/home/jtaylor/projects/_references/igraph/src/layout/align.c:107-123`:
  center layout at the origin.
- `/home/jtaylor/projects/_references/igraph/src/layout/align.c:133-199`:
  build nematic tensor from edge vectors, then centered vertex vectors when
  edge vectors have zero norm.
- `/home/jtaylor/projects/_references/igraph/src/layout/align.c:213-257`:
  normalize tensor, compute eigenvectors, and retry once after removing the
  saved correction vector for near-symmetric tensors.
- `/home/jtaylor/projects/_references/igraph/src/layout/align.c:259-300`:
  rotate layout and reorder axes by descending extent.

## Torch-to-Python Loop Replacements

- Kept the FR fidelity loop in sequential Python over `source` then
  `target = source + 1`, matching the C all-pairs order.
- Kept attraction as a sequential Python loop over input edge order.
- Tightened connected repulsion from factored `1.0 / dlen` multiplication to
  direct `dx / dlen` and `dy / dlen`, matching the C expression order. This was
  the step that moved the residual from about `1.46e-4` to `4.24e-9`.
- Added `_igraph_layout_align_positions()` as an in-place 2D port of
  `igraph_layout_align`.

## Smoke RMSD

Procrustes RMSD, dagua vs measurement-only python-igraph adapter.

| Topology | Seed | Before | After |
| --- | ---: | ---: | ---: |
| path | 1 | 0.000218 | 0.000000003439 |
| path | 2 | 0.000118 | 0.000000006271 |
| path | 3 | 0.000122 | 0.000000004846 |
| star | 1 | 0.000172 | 0.000000005221 |
| star | 2 | 0.000000 | 0.000000005233 |
| star | 3 | 0.000113 | 0.000000003656 |
| clustered | 1 | 0.000012 | 0.000000003918 |
| clustered | 2 | 0.000010 | 0.000000004123 |
| clustered | 3 | 0.000055 | 0.000000003487 |
| grid | 1 | 0.000297 | 0.000000003420 |
| grid | 2 | 0.000001 | 0.000000003125 |
| grid | 3 | 0.000240 | 0.000000004106 |

- Before mean: `0.000113`
- After mean: `0.000000004237`

## Verification

- `git diff dagua/layout/ops/pipelines/fr.py | grep -E "^\\+.*(import igraph|subprocess|from igraph)"`:
  empty output, exit code 1.
- `PATH=/tmp/graphviz_instr/bin:$PATH python eval_output/algo_fidelity/round_41/fr/smoke_check.py`:
  failed because the file is missing from this checkout.
- `ruff check . --fix`: passed.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `pytest tests/test_pipeline_fr.py -x --tb=short -q`: `14 passed, 2 warnings`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: did not
  complete; stale pytest processes ran for 23-50 minutes and were terminated.
- `timeout 900 pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  failed during collection on pre-existing `ImportError: cannot import name
  'layout_drl' from 'dagua.layout.classic'` in `tests/test_classic_drl.py`.
