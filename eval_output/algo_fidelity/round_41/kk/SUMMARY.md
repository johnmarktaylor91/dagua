# Round 41 KK Summary

## Reference Source Lines

- `/home/jtaylor/projects/_references/igraph/src/layout/kamada_kawai.c:27-30`:
  `KK_EPS = 1e-13`.
- `/home/jtaylor/projects/_references/igraph/src/layout/kamada_kawai.c:160-169`:
  non-seeded circular initialization scaled by `0.36 * sqrt(vcount)`.
- `/home/jtaylor/projects/_references/igraph/src/layout/kamada_kawai.c:181-215`:
  all-pairs Dijkstra with `IGRAPH_ALL`, disconnected pairs replaced by graph
  diameter, then `kij = kkconst / dij^2` and `lij = sqrt(vcount) / diameter * dij`.
- `/home/jtaylor/projects/_references/igraph/src/layout/kamada_kawai.c:219-255`:
  cached gradient initialization and max-gradient vertex selection.
- `/home/jtaylor/projects/_references/igraph/src/layout/kamada_kawai.c:260-298`:
  2x2 Newton system for the selected vertex.
- `/home/jtaylor/projects/_references/igraph/src/layout/kamada_kawai.c:314-340`:
  incremental gradient cache update after moving one vertex.

## Sub-Component Diagnosis

Smoke harness: inline Python harness comparing four synthetic topologies
(`path`, `star`, `clustered`, `grid`) at seeds `1, 2, 3` against
`IgraphKamadaKawai` with Procrustes RMSD.

Dominant residuals before this round:

- **Shortest paths:** Dagua's historical KK path used directed NetworkX-style
  APSP and filled unreachable pairs with `1e6`; igraph uses undirected
  `IGRAPH_ALL` distances and fills unreachable pairs with the largest finite
  distance.
- **Solver kernel:** Dagua used the NetworkX/SciPy L-BFGS energy optimizer;
  igraph uses one max-delta vertex Newton update per outer iteration with a
  cached gradient vector.
- **Initialization:** python-igraph's benchmark adapter passes a seeded
  `np.random.RandomState(seed).uniform(-1, 1, [N, 2])` matrix. The old Dagua
  direct path ignored the seed unless explicit `pos` was provided.
- **Output scale:** the igraph adapter multiplies coordinates by `50.0`; this
  is now mirrored in `fidelity_mode="igraph"`.

## Port Implementation Summary

Changed `dagua/layout/ops/pipelines/kk.py` only:

- Added an explicit `fidelity_mode=True` / `fidelity_mode="igraph"` path.
- Added igraph-compatible seeded/circular initialization.
- Added igraph-compatible undirected APSP, positive-weight validation, and
  diameter fill for disconnected pairs.
- Ported the 2D igraph max-delta Newton loop, including `KK_EPS`, `kij`, `lij`,
  max-gradient node selection, and incremental gradient cache updates.
- Kept the existing NetworkX/L-BFGS pipeline as the default when
  `fidelity_mode` is false.

## Before/After Smoke RMSD

| Topology | Seed | Before RMSD | After RMSD |
|---|---:|---:|---:|
| path | 1 | 0.001741880 | 0.000000005358 |
| path | 2 | 0.001752330 | 0.000000004625 |
| path | 3 | 0.001735412 | 0.000000003401 |
| star | 1 | 0.225293476 | 0.000000001994 |
| star | 2 | 0.218339865 | 0.000000001994 |
| star | 3 | 0.245460250 | 0.000000001994 |
| clustered | 1 | 0.143927631 | 0.000000005989 |
| clustered | 2 | 0.169080586 | 0.000000003641 |
| clustered | 3 | 0.155843243 | 0.000000003641 |
| grid | 1 | 0.135313534 | 0.000000002847 |
| grid | 2 | 0.135313534 | 0.000000002847 |
| grid | 3 | 0.135313534 | 0.000000002847 |
| **Mean** |  | **0.130759606** | **0.000000003431** |
| **Max** |  | **0.245460250** | **0.000000005989** |

## Final Verdict

Bit-exact under the sprint's Procrustes RMSD criterion. The remaining smoke
floor is approximately `6e-09` RMSD, caused by Python/NumPy vs C double
rounding plus the reference adapter's final float32 tensor conversion. It is
well below the `<0.001` target.

## Verification

- `ruff check dagua/layout/ops/pipelines/kk.py --fix`: passed.
- `python -m py_compile dagua/layout/ops/pipelines/kk.py`: passed.
- `pytest tests/test_pipeline_kk.py tests/test_layout/test_kk_fidelity.py -x --tb=short -q`:
  passed, `20 passed, 2 warnings in 0.27s`.
- `mypy --follow-imports=silent dagua/cli.py`: passed.
- `ruff check . --fix`: blocked by unrelated `F841` in
  `dagua/layout/ops/drl.py:1382`.
- `pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q`: interrupted
  after multiple minutes with only four progress dots while several parallel
  R41 agents were running the same broad command in the checkout.
- `pytest tests/ -x --tb=short -q -m "not slow and not benchmark and not rare"`:
  blocked during collection by unrelated `ImportError: cannot import name
  'layout_drl' from 'dagua.layout.classic'` in `tests/test_classic_drl.py:10`.
