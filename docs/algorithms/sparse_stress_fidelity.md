# sparse-stress fidelity

Reference: MarkOrtmann/sparse-stress Java implementation, built manually with `javac`
because Gradle 2.5 cannot parse Java 17
(`Could not determine java version from '17.0.19'`).

Native implementation: sampler and sparse term aggregation ported from source; PivotMDS
uses NumPy's deterministic symmetric eigensolver on the same centered kernel.

Named residual stage: `initialization_eigensolver`; pivots/sampler and sparse terms are
matched in isolation for the pinned cases.

## Results

| graph | residual | tier | stress delta | neighborhood delta | quality |
| --- | ---: | --- | ---: | ---: | --- |
| diamond | 3.36635e-10 | BIT/SIMILARITY_EXACT | 4.09484e-15 | 0 | PRACTICALLY_EQUIVALENT |
| complete_5 | 0.665782 | DISTRIBUTIONAL | 0.117925 | 0 | NOT_EQUIVALENT |
| wheel_6 | 6.28976e-08 | BIT/SIMILARITY_EXACT | 2.3756e-08 | 0 | PRACTICALLY_EQUIVALENT |
| grid_3x3 | 0.0267465 | POSITIONAL | 0.00695204 | 0 | PRACTICALLY_EQUIVALENT |

## Notes

- The production pipeline never calls this adapter or any subprocess.
- Reference input is restricted to connected simple undirected graphs, matching the reference README.
- Remaining residual is expected to appear first at PivotMDS eigenvector orientation/order, not in sampler or sparse terms.
