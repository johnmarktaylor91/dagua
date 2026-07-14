# d3-force Fidelity

Reference: `d3-force` via `dagua.eval.competitors.d3force_competitor.D3ForceCompetitor`.
Seed: `1`. Ticks: `300`.

LCG verification: bit-for-bit match for the first 20 values.

Summary: 11 bit-exact, 0 close, 0 distributional.

| graph | Procrustes d_R | anisotropic d_R | tier |
| --- | ---: | ---: | --- |
| single_node | 0 | 0 | bit-exact |
| small_chain | 3.55173776281e-16 | 3.03747114537e-16 | bit-exact |
| binary_tree | 1.20007262331e-15 | 1.18007526388e-15 | bit-exact |
| diamond | 1.71658754945e-16 | 0 | bit-exact |
| grid_5x5 | 6.74163432075e-16 | 6.43356898107e-16 | bit-exact |
| org_chart_small | 1.50813630553e-15 | 1.4349081638e-15 | bit-exact |
| long_skip | 4.70055648695e-16 | 2.43697148186e-16 | bit-exact |
| disconnected | 2.05489693247e-16 | 1.75041865429e-16 | bit-exact |
| cycle_4 | 8.44152876808e-17 | 0 | bit-exact |
| random_dag_50 | 2.20168717934e-14 | 2.19939345192e-14 | bit-exact |
| org_chart_deep | 5.2022766769e-14 | 5.09124350173e-14 | bit-exact |

Named residual: none. The native pipeline matches d3-force's LCG, phyllotaxis
initialization, link force, d3-quadtree Barnes-Hut many-body traversal, center
force, and velocity-Verlet order within the bit-exact threshold.
