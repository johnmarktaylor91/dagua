# SMACOF Nonmetric + Radial Tree Fidelity

Reference adapters:

- `smacof_nonmetric`: `sklearn.manifold.smacof(metric=False, n_init=1)` on graph geodesic distances.
- `radial_tree`: `python-igraph 1.0.0` `Graph.layout_reingold_tilford_circular(mode="out")`.

Production guard: neither pipeline calls its reference runtime. SMACOF ports the sklearn nonmetric loop and isotonic wrapper behavior; radial tree composes the local igraph-compatible RT port with igraph's documented circular transform.

| algorithm | graph | N | E | d_R | max centered abs | tier | note |
| --- | ---: | ---: | ---: | ---: | ---: | --- | --- |
| smacof_nonmetric | path6 | 6 | 5 | 4.670e-16 | 2.220e-16 | bit/similarity-exact | isotonic disparities + Guttman update matched |
| smacof_nonmetric | branched7 | 7 | 6 | 5.303e-16 | 5.274e-16 | bit/similarity-exact | isotonic disparities + Guttman update matched |
| smacof_nonmetric | cycle_chord6 | 6 | 7 | 1.898e-15 | 1.998e-15 | bit/similarity-exact | isotonic disparities + Guttman update matched |
| radial_tree | star5 | 5 | 4 | 2.564e-16 | 0.000e+00 | bit/similarity-exact | RT tidy coords + igraph polar transform matched |
| radial_tree | binary7 | 7 | 6 | 1.989e-16 | 0.000e+00 | bit/similarity-exact | RT tidy coords + igraph polar transform matched |
| radial_tree | unbalanced6 | 6 | 5 | 1.963e-17 | 0.000e+00 | bit/similarity-exact | RT tidy coords + igraph polar transform matched |

Named residuals:

- `smacof_nonmetric`: no first-divergent stage on the verification set; any remaining scalar stress drift is one-ulp summation noise while positions match the sklearn run.
- `radial_tree`: no first-divergent stage on the verification set; raw coordinates match igraph within float64/trigonometric tolerance.

Tier thresholds: `d_R < 1e-9` is bit/similarity-exact; `d_R < 1e-3` is positional.
