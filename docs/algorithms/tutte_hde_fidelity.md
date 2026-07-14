# Tutte/HDE fidelity verification

References: pinned local deterministic adapters in `scripts/verify_tutte_hde_fidelity.py`. Production pipelines do not invoke external reference engines.

## tutte

Result: **6/6 bit-exact**, **0 positional**, thresholds `bit < 1e-09`, positional `< 1e-06`.

| graph | N | E | Procrustes d_R | anisotropic d_R | max raw diff | verdict | first divergent/N-A stage |
|---|---:|---:|---:|---:|---:|---|---|
| triangle | 3 | 3 | 2.256e-16 | 0.000e+00 | 0.000e+00 | bit-exact | none |
| wheel_5 | 5 | 8 | 6.123e-17 | 0.000e+00 | 0.000e+00 | bit-exact | none |
| path_5 | 5 | 4 | 1.010e-17 | 0.000e+00 | 0.000e+00 | bit-exact | no peripheral cycle; all nodes fixed on convex polygon |
| disconnected | 6 | 3 | 1.319e-16 | 0.000e+00 | 0.000e+00 | bit-exact | no peripheral cycle; all nodes fixed on convex polygon |
| grid_3x3 | 9 | 12 | 2.982e-17 | 0.000e+00 | 0.000e+00 | bit-exact | none |
| lollipop | 7 | 7 | 3.034e-16 | 5.113e-17 | 1.110e-16 | bit-exact | none |

## hde

Result: **6/6 bit-exact**, **0 positional**, thresholds `bit < 1e-09`, positional `< 1e-06`.

| graph | N | E | Procrustes d_R | anisotropic d_R | max raw diff | verdict | first divergent/N-A stage |
|---|---:|---:|---:|---:|---:|---|---|
| triangle | 3 | 3 | 1.585e-16 | 1.188e-16 | 1.110e-16 | bit-exact | none |
| wheel_5 | 5 | 8 | 4.930e-32 | 0.000e+00 | 2.311e-33 | bit-exact | none |
| path_5 | 5 | 4 | 4.421e-17 | 5.526e-17 | 2.220e-16 | bit-exact | none |
| disconnected | 6 | 3 | 4.913e-16 | 5.222e-16 | 1.608e-15 | bit-exact | none |
| grid_3x3 | 9 | 12 | 4.382e-16 | 4.503e-16 | 3.464e+00 | bit-exact | none |
| lollipop | 7 | 7 | 1.050e-16 | 6.553e-17 | 6.661e-16 | bit-exact | none |

## Notes

Tutte uses a chordless peripheral cycle as the fixed convex boundary in the headless tensor API. Graphs without such a cycle are finite but marked N/A for theorem fidelity because all nodes are fixed on the fallback polygon.

HDE is also exposed as the reusable `hde_project_pivot_distances` init op; the public pipeline composes adjacency, farthest-first pivots, distance queries, and that init op.
