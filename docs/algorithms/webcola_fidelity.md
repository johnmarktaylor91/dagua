# WebCola Fidelity

Reference: WebCola 3.4.0 via `dagua.eval.competitors.webcola_competitor.WebColaCompetitor`.

Scope: placement only. GridRouter/routing is intentionally excluded.

Adapter policy: initial positions are explicitly pinned to the deterministic circle used by the native pipeline, and WebCola disconnected-component packing is disabled so the stress/VPSC core is measured directly.

| variant | graph | nodes | edges | max_abs | procrustes | tier | residual |
|---|---:|---:|---:|---:|---:|---|---|
| webcola | single_node | 1 | 0 | 0.000e+00 | 0.000e+00 | bit-exact | none |
| webcola_constrained | single_node | 1 | 0 | 0.000e+00 | 0.000e+00 | bit-exact | none |
| webcola | small_chain | 4 | 3 | 1.776e-15 | 2.671e-16 | bit-exact | none |
| webcola_constrained | small_chain | 4 | 3 | 1.066e-14 | 2.388e-16 | bit-exact | none |
| webcola | binary_tree | 7 | 6 | 1.421e-14 | 3.050e-16 | bit-exact | none |
| webcola_constrained | binary_tree | 7 | 6 | 1.421e-14 | 3.525e-16 | bit-exact | none |
| webcola | diamond | 4 | 4 | 5.329e-15 | 5.280e-16 | bit-exact | none |
| webcola_constrained | diamond | 4 | 4 | 7.105e-15 | 2.831e-16 | bit-exact | none |
| webcola | grid_5x5 | 25 | 40 | 1.755e-08 | 1.476e-10 | positional | float-order residual in Runge-Kutta descent reduction |
| webcola_constrained | grid_5x5 | 25 | 40 | 1.421e-14 | 2.286e-16 | bit-exact | none |
| webcola | org_chart_small | 8 | 7 | 2.487e-14 | 4.077e-16 | bit-exact | none |
| webcola_constrained | org_chart_small | 8 | 7 | 1.421e-14 | 2.400e-16 | bit-exact | none |
| webcola | long_skip | 6 | 6 | 3.155e-30 | 6.161e-17 | bit-exact | none |
| webcola_constrained | long_skip | 6 | 6 | 3.553e-15 | 4.402e-16 | bit-exact | none |
| webcola | disconnected | 6 | 3 | 0.000e+00 | 2.169e-16 | bit-exact | none |
| webcola_constrained | disconnected | 6 | 3 | 1.776e-14 | 4.906e-16 | bit-exact | none |
| webcola | cycle_4 | 4 | 4 | 1.341e-29 | 4.398e-18 | bit-exact | none |
| webcola_constrained | cycle_4 | 4 | 4 | 1.066e-14 | 2.884e-16 | bit-exact | none |
| webcola | random_dag_50 | 50 | 28 | 5.684e-14 | 2.786e-16 | bit-exact | none |
| webcola_constrained | random_dag_50 | 50 | 28 | 9.948e-14 | 2.780e-16 | bit-exact | none |
| webcola | org_chart_deep | 10 | 9 | 2.132e-14 | 2.731e-16 | bit-exact | none |
| webcola_constrained | org_chart_deep | 10 | 9 | 2.132e-14 | 3.244e-16 | bit-exact | none |

## Summary

- Bit-exact rows: 21/22.
- Positional rows: 1/22.
- Named residual: none for bit-exact rows; any positional constrained rows are attributed to floating-point order in the VPSC active-set projection.
