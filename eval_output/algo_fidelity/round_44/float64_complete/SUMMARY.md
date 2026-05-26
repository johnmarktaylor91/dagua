# Round 44 Float64 Completion

## Audit Findings

R44 found that `LayoutConfig.fidelity_dtype=torch.float32` made the engine
silently pass float32 even when a user only set `fidelity_mode`. The default is
now optional: fidelity-mode pipelines resolve unset dtype to `torch.float64`,
while the graph-level public API casts returned tensors back to `torch.float32`.

Fixed float32 hot spots:

- `classical_mds`: OGDF PivotMDS direct path now accepts `fidelity_dtype` and
  returns float64 when requested.
- `davidson_harel`: python-igraph adapter tensor now preserves requested
  fidelity dtype.
- `drl`: python-igraph adapter tensor now preserves requested fidelity dtype.
- `fa2`: exact reference path and FA2 config resolve unset fidelity dtype to
  float64; explicit float32 still downgrades.
- `gem`: OGDF sequential initial/final tensors and finalizer now preserve
  requested fidelity dtype. Star seed 43 remains chaotic.
- `graphopt`: finalizer output dtype is configurable from fidelity dtype.
- `lgl`: finalizer output dtype is configurable from fidelity dtype.
- `stress_sgd`: OGDF serial sweep output and warm-start dtype now preserve
  requested fidelity dtype.
- `tsnet`: sklearn exact path resolves unset fidelity dtype to float64.
- `umap_layout`: wrapper accepts `fidelity_dtype`; umap-learn still computes
  float32 internally.

Engines with no meaningful dtype-controlled branch in the round 41 harness:
`maxent_stress`, `reingold_tilford`. `pivot_mds` uses `compute_dtype`, not
`fidelity_dtype`.

## Smoke Results

Round 41 smoke harnesses were rerun with the new default float64 fidelity mode.
The comparison wrapper also ran explicit float32 vs float64 where the harness
exposed a dtype path.

| Engine | RMSD float32 | RMSD float64 | Reduction factor | Diagnosis |
|---|---:|---:|---:|---|
| classical_mds | 2.72834154587e-17 | 5.25782182583e-09 | 0.0x | Harness reference is float32; default smoke remains 0.000000001. |
| davidson_harel | 6.67910388859e-17 | 7.1631652918e-09 | 0.0x | Harness reference is float32; default smoke remains zero. |
| drl | 6.04707943961e-17 | 4.3584093154e-08 | 0.0x | Adapter/runtime residual around 1e-8, not a dtype hot spot. |
| fa2 | 0.000771713646 | 0.0000751178925 | 10.3x | Exact path is roundoff; Barnes-Hut tree residual dominates. |
| gem | 0.0278484129692 | 0.000410417792 | 67.9x | Star seed 43 remains 0.00437629951; chaotic/algorithmic. |
| graphopt | 7.08973118583e-09 | 9.77659212136e-17 | 72517407.9x | Float64 closes the finalization floor. |
| lgl | 0.171220751265 | 0.171220750561 | 1.0x | RNG/grid/order residual dominates. |
| maxent_stress | 2.23814173121e-08 | 2.23814173121e-08 | 1.0x | No fidelity_dtype branch; reproducibility smoke already near 2e-8. |
| pivot_mds | 0.0840845698714 | 4.8608512317e-09 | 17298322.0x | Uses `compute_dtype`; float64 closes the OGDF SVD floor. |
| reingold_tilford | 0.0854571145 | 0 | inf | Combinatorial traversal, not dtype limited. |
| stress_sgd | 6.54926443918e-17 | 8.02301942312e-09 | 0.0x | Harness reference is float32; default smoke remains zero. |

## Helped

Clear numeric wins: `graphopt`, `pivot_mds`, `gem`, `fa2`.

Default-fidelity smoke checks still at or near zero after the dtype default
change: `classical_mds`, `davidson_harel`, `stress_sgd`, `reingold_tilford`.

## Did Not Help

- `gem` star seed 43 did not drop under float64. It remains the R43 chaotic
  residual, not a plain float32 floor.
- `lgl` is algorithmically floored by RNG/grid/update-order differences.
- `fa2` Barnes-Hut rows are algorithmically floored by tree implementation
  differences; non-Barnes-Hut exact rows are roundoff.
- `maxent_stress` has no fidelity dtype branch in the harness.
- Some round 41 references return float32 tensors, so explicit float64
  comparisons can appear worse than explicit float32 even though the pipeline
  now honors float64 internally.
