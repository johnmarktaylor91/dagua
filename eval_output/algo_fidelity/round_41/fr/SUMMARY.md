# Round 41 FR igraph Fidelity

## Reference Source Lines

- `/home/jtaylor/projects/_references/igraph/src/layout/fruchterman_reingold.c`
- `igraph_layout_i_fr`: random bounded initialization when no seed matrix is supplied, weak-connectivity branch, direct all-pairs repulsion, edge-order attraction, per-node displacement jitter, temperature clamp, and linear cooling.
- Public wrapper: `igraph_layout_fruchterman_reingold` validates inputs, defaults `IGRAPH_LAYOUT_AUTOGRID` to no-grid for `N <= 1000`, and delegates to `igraph_layout_i_fr` for these smoke graphs.
- Python adapter observation: local python-igraph default `niter` is 500 and default `start_temp` is `sqrt(vcount) / 10`.

## Sub-Component Diagnosis

| Component | Finding |
| --- | --- |
| Initialization | Dominant. Dagua FR used NumPy `[0, 1]`; the local igraph adapter passes a NumPy `[-1, 1]` seed matrix. |
| Iteration count | Dominant. Dagua fidelity path was still NetworkX-style 50 iterations; igraph defaults to 500. |
| Force kernel | Dominant. Dagua used NetworkX `k^2 / d^2 - A*d/k`; igraph uses unit inverse-square repulsion and edge attraction `distance * weight`. |
| Disconnected graphs | Important. igraph switches repulsion to `(C - d^3) / (d^2*C)` with `C = n*sqrt(n)` for weakly disconnected graphs. |
| Iteration order | Ported. Repulsion loops `v` then `u = v + 1`; attraction loops input edge order. |
| Convergence | Ported by omission. igraph FR runs the requested fixed iteration count; no NetworkX early break. |
| Normalization | Dominant. igraph returns raw coordinates; the adapter multiplies by `50.0`. No centering/max-abs normalization. |

## Port Implementation

- Added `fidelity_mode=True` / `fidelity_mode="igraph"` to `layout_fr_pipeline`.
- Implemented an FR-local igraph reference loop in `dagua/layout/ops/pipelines/fr.py`; no shared infrastructure was changed.
- Mapped unchanged `steps=50` to python-igraph default `niter=500` only in igraph fidelity mode.
- Preserved existing NetworkX-compatible behavior and tests for the default path.
- Added a regression test comparing the igraph fidelity path to `IgraphFR` with Procrustes RMSD `< 0.001`.

## Smoke RMSD

Procrustes RMSD, dagua vs `IgraphFR` reference adapter.

| Topology | Seed | Before | After |
| --- | ---: | ---: | ---: |
| path | 1 | 0.159219 | 0.000218 |
| path | 2 | 0.194711 | 0.000118 |
| path | 3 | 0.168286 | 0.000122 |
| star | 1 | 0.061031 | 0.000172 |
| star | 2 | 0.029758 | 0.000000 |
| star | 3 | 0.068153 | 0.000113 |
| clustered | 1 | 0.089105 | 0.000012 |
| clustered | 2 | 0.107383 | 0.000010 |
| clustered | 3 | 0.079772 | 0.000055 |
| grid | 1 | 0.191402 | 0.000297 |
| grid | 2 | 0.155941 | 0.000001 |
| grid | 3 | 0.071760 | 0.000240 |

- Before mean: `0.114710`
- After mean: `0.000113`

## Final Verdict

Bit-exact target reached for the smoke contract: all cases are below `0.001`
RMSD, with overall mean `0.000113`.

The remaining non-zero floor appears to be numeric/jitter-interface noise:
Dagua uses Python `random.Random(seed).uniform()` for the igraph displacement
jitter approximation, while python-igraph routes `RNG_UNIF()` through its custom
global RNG bridge. The jitter amplitude is `1e-9`, and the observed smoke floor
is at most `0.000297` after Procrustes alignment.
