# ELK Secondary Fidelity

Reference: `elkjs` with flat graphs, seed `1`, and pinned secondary algorithms.
Runtime pipelines do not import `elkjs`, spawn Node, or call the reference adapter.

| algorithm | graph | N | E | d_R | d_A | tier |
|---|---:|---:|---:|---:|---:|---|
| elk_force | single | 1 | 0 | 0.000e+00 | 0.000e+00 | bit/similarity-exact |
| elk_force | chain5 | 5 | 4 | 4.276e-08 | 3.985e-08 | positional |
| elk_force | binary7 | 7 | 6 | 3.557e-08 | 3.469e-08 | positional |
| elk_force | diamond | 4 | 4 | 3.211e-08 | 2.836e-08 | positional |
| elk_force | cycle4 | 4 | 4 | 3.467e-08 | 3.457e-08 | positional |
| elk_stress | single | 1 | 0 | 0.000e+00 | 0.000e+00 | bit/similarity-exact |
| elk_stress | chain5 | 5 | 4 | 4.315e-08 | 4.262e-08 | positional |
| elk_stress | binary7 | 7 | 6 | 3.971e-08 | 3.571e-08 | positional |
| elk_stress | diamond | 4 | 4 | 5.983e-08 | 5.364e-08 | positional |
| elk_stress | cycle4 | 4 | 4 | 4.779e-08 | 3.813e-08 | positional |
| elk_mrtree | single | 1 | 0 | 0.000e+00 | 0.000e+00 | bit/similarity-exact |
| elk_mrtree | chain5 | 5 | 4 | 1.755e-16 | 0.000e+00 | bit/similarity-exact |
| elk_mrtree | binary7 | 7 | 6 | 3.208e-02 | 9.333e-17 | positional |
| elk_radial | single | 1 | 0 | 0.000e+00 | 0.000e+00 | bit/similarity-exact |
| elk_radial | chain5 | 5 | 4 | 3.024e-08 | 3.024e-08 | positional |
| elk_radial | binary7 | 7 | 6 | 1.954e-08 | 1.918e-08 | positional |

## Residuals

- `elk_force`: first-divergent stage is final JavaScript/Java numeric serialization; the local port matches `ForceLayoutProvider`, `AbstractForceModel.layout`, and `FruchtermanReingoldModel` init/order against the cached `elkjs` references to float-rounding residuals.
- `elk_stress`: first-divergent stage is final JavaScript/Java numeric serialization; the local port now uses `StressLayoutProvider`'s Force warm start and `StressMajorization.computeNewPosition` Gauss-Seidel update order.
- `elk_mrtree`: first-divergent stage is ELK treeification / ordering / compaction; local implementation uses the existing Reingold-Tilford tidy-tree op.
- `elk_radial`: first-divergent stage is ELK radial treeification / angular ordering; local implementation is distinct from `radial_tree` and uses concentric RT depths.
