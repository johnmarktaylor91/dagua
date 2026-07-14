# d3-force Fidelity

Reference: `d3-force` via `dagua.eval.competitors.d3force_competitor.D3ForceCompetitor`.
Seed: `1`. Ticks: `300`.

LCG verification: bit-for-bit match for the first 20 values.

Summary: 1 bit-exact, 7 close, 3 distributional.

| graph | Procrustes d_R | anisotropic d_R | tier |
| --- | ---: | ---: | --- |
| single_node | 0 | 0 | bit-exact |
| small_chain | 0.0167837004925 | 0.0167557890797 | close |
| binary_tree | 0.0115535283876 | 0.0100997906141 | close |
| diamond | 0.262152522607 | 0.2389640433 | distributional |
| grid_5x5 | 0.0941326191263 | 0.090198147234 | close |
| org_chart_small | 0.167932710946 | 0.167286225833 | distributional |
| long_skip | 0.020000994797 | 0.0198791797239 | close |
| disconnected | 0.011461026406 | 0.0114484361846 | close |
| cycle_4 | 0.151969700827 | 0.121289778669 | distributional |
| random_dag_50 | 0.0898549648805 | 0.0891825417078 | close |
| org_chart_deep | 0.093632669604 | 0.0934151936671 | close |

Named residual: full-layout divergence first appears at the many-body force stage.
The current native op matches d3-force's LCG, phyllotaxis initialization, link force,
center force, and velocity-Verlet order, but uses direct ordered-pair n-body
evaluation instead of d3-quadtree Barnes-Hut traversal. This is a mathematical
summation/approximation-order gap, not runtime delegation.
