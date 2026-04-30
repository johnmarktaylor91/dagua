# Round 25 Pivot-MDS Straggler Fix

## Diagnosis

`mixed_width_labels` is the only graph in the five-graph Pivot-MDS check whose
simplified undirected topology is non-path and has two unequal nonzero embedding
components: a 4-cycle with a tail:

```text
x -> MultiHeadAttention(embed_dim=512, num_heads=8)
MultiHeadAttention(embed_dim=512, num_heads=8) -> LayerNorm(normalized_shape=(512,))
LayerNorm(normalized_shape=(512,)) -> +
x -> +
+ -> ReLU
ReLU -> out
```

That exposes a Dagua/OGDF coordinate-scale mismatch. OGDF's
`singularValueDecomposition()` normalizes `C^T u`, then `pivotMDSLayout()` scales
coordinates by `sqrt(sigma)`. Dagua was scaling by `sigma`, which changes the
aspect ratio only when both output dimensions are active with unequal singular
values. The other four measured graphs were path-like or degenerate enough that
the mismatch was hidden after Procrustes alignment.

## Changes

- `dagua/layout/ops/embed.py`
  - `_pivot_mds_coordinates()` now applies OGDF's `sqrt(sigma)` coordinate
    scale instead of `sigma`.
- `scripts/ogdf_runner.cpp`
  - Applies the already-forwarded `numberOfPivots` option with
    `PivotMDS::setNumberOfPivots()`.
- `tests/test_layout/test_pivot_mds_fidelity.py`
  - Adds a regression test for OGDF's `sqrt(sigma)` Pivot-MDS coordinate scale.

## Verification

Post-fix command:

```bash
python scripts/algo_fidelity_live_compare.py classic_pivot_mds ogdf_pivot_mds \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_25/pivot_mds/post_fix
```

Post-fix results:

| graph | median RMSD | p95 RMSD |
| --- | ---: | ---: |
| linear_3layer_mlp | 0.0000728 | 0.000124 |
| mixed_width_labels | 0.000000330 | 0.000000333 |
| nested_shallow_enc_dec | 0.0000728 | 0.000124 |
| parallel_multiedge_bundle | 0.000000281 | 0.000000295 |
| tl_mlp_3layer | 0.0000728 | 0.0000835 |

Overall median: `0.000073`; worst: `tl_mlp_3layer 0.000073`.

## Test Commands

Passed:

```bash
mypy --follow-imports=silent dagua/cli.py
pytest tests/test_layout/ tests/test_graph.py -x --tb=short -q -k "pivot"
pytest tests/test_layout/ -x --tb=short -q -k "pivot"
ruff check dagua/layout/ops/embed.py tests/test_layout/test_pivot_mds_fidelity.py --fix
```

Blocked by unrelated workspace state:

```bash
ruff check . --fix
```

`ruff check . --fix` stops on an existing unused local variable in the unrelated
untracked file `scripts/round_24_aggregate.py`.

Could not syntax-check or rebuild `scripts/ogdf_runner.cpp` locally because OGDF
headers/libraries are not available in this shell:

```text
fatal error: ogdf/basic/Graph.h: No such file or directory
scripts/ogdf_runner: error while loading shared libraries: libOGDF.so.2025.10.01
```
