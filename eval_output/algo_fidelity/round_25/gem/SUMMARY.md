# Round 25 GEM Fidelity Summary

## Scope

Implemented `fidelity_mode` for the GEM family by adding:

- glibc-compatible `rand()` reproduction in `dagua/layout/ops/gem.py`
- OGDF runner-style initial positions from `rand() % 1000 / 10.0`
- `fidelity_mode` plumbing through `InitializeGEMPositions`, `GEMPrepareState`,
  `build_gem_pipeline()`, and `layout_gem_pipeline()`
- classic GEM competitor defaults/call-site wiring for `fidelity_mode=True`
- regression tests for the seed-42 rand sequence and OGDF runner initialization

## Fidelity Compare

Command:

```bash
python scripts/algo_fidelity_live_compare.py classic_gem ogdf_gem \
    --seeds 30 \
    --graphs linear_3layer_mlp,parallel_multiedge_bundle,nested_shallow_enc_dec,tl_mlp_3layer,mixed_width_labels \
    --output-dir eval_output/algo_fidelity/round_25/gem/{baseline,post_fix}
```

Baseline:

- median RMSD: `0.035869`
- worst graph: `mixed_width_labels` at `0.188649`

Post-fix:

- median RMSD: `0.038132`
- worst graph: `mixed_width_labels` at `0.199858`

Per-graph median RMSD:

| graph | baseline | post_fix | delta |
| --- | ---: | ---: | ---: |
| linear_3layer_mlp | 0.035869 | 0.038132 | +0.002263 |
| parallel_multiedge_bundle | 0.026163 | 0.023863 | -0.002300 |
| nested_shallow_enc_dec | 0.048393 | 0.048834 | +0.000441 |
| tl_mlp_3layer | 0.024108 | 0.022518 | -0.001590 |
| mixed_width_labels | 0.188649 | 0.199858 | +0.011209 |

## Principled Residual

The OGDF-faithful runner initialization did not improve the aggregate RMSD gate
for this graph set. Two graphs improved slightly, but the median and worst-graph
metrics regressed. The implemented helper follows the explicit seed-42 fixture
from the task (`[166, 740, 881, 241, 12, 758]` modulo 1000), which corresponds
to glibc `rand()` output after `srand(42)`.

The remaining mismatch is therefore likely outside the initial position fixture:
GEM still differs from OGDF in later stochastic node permutation scheduling,
connected-component packing, or exact OGDF graph-attribute geometry handling.
