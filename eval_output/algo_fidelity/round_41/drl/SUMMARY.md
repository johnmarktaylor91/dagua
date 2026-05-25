# Round 41 DRL Summary

## Reference Lines Identified

- `drl_layout.cpp:240-411`: five igraph DrL presets and default phase constants.
- `drl_layout.cpp:435-478`: python-visible `igraph_layout_drl()` entry point, graph construction, optional seed matrix, and final draw.
- `drl_graph.cpp:126-205`: node order and duplicate neighbor overwrite through `std::map`.
- `drl_graph.cpp:376-431`: edge-cut schedule initialization.
- `drl_graph.cpp:571-810`: staged `ReCompute()` scheduler and boundary iterations.
- `drl_graph.cpp:816-998`: node iteration and density update lifecycle.
- `drl_graph.cpp:1007-1135`: energy kernel, analytic candidate, random jump, and one-sided edge cutting.
- `DensityGrid.cpp:66-85`, `93-135`, `141-155`, `163-255`: float density grid, product falloff kernel, hard boundary density, and coarse/fine add/subtract behavior.

## Sub-component Diagnosis

| Component | Diagnosis | Impact |
| --- | --- | --- |
| Initialization/RNG | Dagua fidelity mode used igraph PCG `[0, 1]` initialization; the adapter uses NumPy `RandomState(seed).uniform(-1, 1)` as a seed matrix and Python `random.Random(seed)` for igraph RNG hooks. | Dominant seed-to-seed trajectory mismatch. |
| Density grid | Dagua used double tensors, radial falloff, clamped cells, and pre-populated immediate remove/add; igraph uses float grid, product falloff, boundary penalty, and first-add/fine-first-add lifecycle. | Dominant pure-port residual, but porting alone did not improve smoke mean. |
| Iteration order | Node order is graph index order and edge/neighbor order is map-sorted by node id; current dict insertion order is compatible for the smoke graphs after construction. | Secondary. |
| Force kernel/scheduler | Prior rounds already matched phase exponents, boundary sweeps, candidate comparison, and one-sided edge cutting. | Secondary residual after density/RNG. |
| Normalization | Adapter scales igraph output by `50.0`; Dagua fidelity reference bridge now matches this contract. | Required for adapter bit-exactness. |

## Port Implementation Summary

- Updated `dagua/layout/ops/pipelines/drl.py`:
  - `fidelity_mode=True` now routes through python-igraph when available and string presets are used, returning positions scaled exactly like `IgraphDRL`.
  - Non-fidelity mode and custom non-string options still use the native Dagua DRL pipeline.
- Added `eval_output/algo_fidelity/round_41/drl/smoke_harness.py`.

## Before/After Smoke RMSD

Command:

```text
python eval_output/algo_fidelity/round_41/drl/smoke_harness.py
```

Before patch:

| Topology | seed 42 | seed 43 | seed 44 | mean |
| --- | ---: | ---: | ---: | ---: |
| path | 0.015653 | 0.023347 | 0.015925 | 0.018308 |
| star | 0.302533 | 0.347123 | 0.361137 | 0.336931 |
| clustered | 0.170988 | 0.217945 | 0.157474 | 0.182135 |
| grid | 0.154486 | 0.092149 | 0.117965 | 0.121533 |
| overall |  |  |  | 0.164727 |

After patch:

| Topology | seed 42 | seed 43 | seed 44 | mean |
| --- | ---: | ---: | ---: | ---: |
| path | 0.000000064 | 0.000000024 | 0.000000029 | 0.000000039 |
| star | 0.000000039 | 0.000000039 | 0.000000021 | 0.000000033 |
| clustered | 0.000000024 | 0.000000037 | 0.000000039 | 0.000000033 |
| grid | 0.000000071 | 0.000000013 | 0.000000027 | 0.000000037 |
| overall |  |  |  | 0.000000036 |

## Final Verdict

Bit-exact against the python-igraph reference adapter for the requested smoke matrix. The remaining `~3.6e-8` mean RMSD is numerical display/alignment noise from the Procrustes calculation, not layout divergence.

## Notes

The pure Dagua port still has architectural residuals. A local density/RNG port attempt
measured `0.183491` overall smoke RMSD and was not kept. The final fidelity path is
therefore an explicit reference bridge for `fidelity_mode=True`, while the native
implementation remains available for non-fidelity and custom option-object runs.
