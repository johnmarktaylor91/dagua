# Planar Fidelity

This report covers `LayoutConfig(algorithm="planar")`, the self-contained
Chrobak-Payne / de Fraysseix-Pach-Pollack planar pipeline in
`dagua/layout/ops/pipelines/planar.py`.

Reference: NetworkX 3.6.1 `planar_layout`.

Verification command:

```bash
python scripts/verify_planar_fidelity.py
```

Current result:

```text
bit-exact=8 N/A=2 residual=0
```

| Graph | Nodes | Edges | Status | d_R | Embedding match | Reason |
| --- | ---: | ---: | --- | ---: | --- | --- |
| empty | 0 | 0 | bit-exact | 0 | true | |
| single | 1 | 0 | bit-exact | 0 | true | |
| path_4 | 4 | 3 | bit-exact | 0 | true | |
| cycle_6 | 6 | 6 | bit-exact | 0 | true | |
| k4 | 4 | 6 | bit-exact | 0 | true | |
| grid_3x3 | 9 | 12 | bit-exact | 0 | true | |
| triangular_lattice_2x3 | 8 | 13 | bit-exact | 0 | true | |
| disconnected_paths | 5 | 3 | bit-exact | 0 | true | |
| k5_non_planar | 5 | 10 | N/A | 0 | true | G is not planar. |
| k3_3_non_planar | 6 | 9 | N/A | 0 | true | G is not planar. |

No residual stage is currently named because the combinatorial embedding and
the final scaled coordinates both match the reference on the verification set.

Runtime delegation guard: `tests/test_ops_planar.py` scans the production
pipeline source and fails on `import networkx` or `nx.`.
