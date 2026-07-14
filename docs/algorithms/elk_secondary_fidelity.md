# ELK Secondary Fidelity

Reference: `elkjs` with flat graphs, seed `1`, and pinned secondary algorithms.
Runtime pipelines do not import `elkjs`, spawn Node, or call the reference adapter.

| algorithm | graph | N | E | d_R | d_A | tier |
|---|---:|---:|---:|---:|---:|---|
| elk_force | single | 1 | 0 | 0.000e+00 | 0.000e+00 | bit/similarity-exact |
| elk_force | chain5 | 5 | 4 | 7.157e-01 | 5.507e-01 | distributional |
| elk_force | binary7 | 7 | 6 | 1.127e+00 | 8.479e-01 | distributional |
| elk_force | diamond | 4 | 4 | 5.387e-01 | 3.993e-01 | distributional |
| elk_force | cycle4 | 4 | 4 | 9.682e-01 | 7.033e-01 | distributional |
| elk_stress | single | 1 | 0 | 0.000e+00 | 0.000e+00 | bit/similarity-exact |
| elk_stress | chain5 | 5 | 4 | 5.596e-02 | 5.535e-02 | positional |
| elk_stress | binary7 | 7 | 6 | 5.906e-01 | 5.164e-01 | distributional |
| elk_stress | diamond | 4 | 4 | 5.495e-03 | 3.959e-03 | positional |
| elk_stress | cycle4 | 4 | 4 | 9.417e-01 | 7.078e-01 | distributional |
| elk_mrtree | single | 1 | 0 | 0.000e+00 | 0.000e+00 | bit/similarity-exact |
| elk_mrtree | chain5 | 5 | 4 | 1.755e-16 | 0.000e+00 | bit/similarity-exact |
| elk_mrtree | binary7 | 7 | 6 | 3.208e-02 | 9.333e-17 | positional |
| elk_radial | single | 1 | 0 | 0.000e+00 | 0.000e+00 | bit/similarity-exact |
| elk_radial | chain5 | 5 | 4 | 3.024e-08 | 3.024e-08 | positional |
| elk_radial | binary7 | 7 | 6 | 1.954e-08 | 1.918e-08 | positional |

## Residuals

- `elk_force`: first-divergent stage is initial graph import / node micro-layout; local port matches the documented Eades/FR displacement loop but not ELK's pre-layout coordinates.
- `elk_stress`: first-divergent stage is initial graph import; the majorization update follows the ELK loop, but starts from local deterministic coordinates.
- `elk_mrtree`: first-divergent stage is ELK treeification / ordering / compaction; local implementation uses the existing Reingold-Tilford tidy-tree op.
- `elk_radial`: first-divergent stage is ELK radial treeification / angular ordering; local implementation is distinct from `radial_tree` and uses concentric RT depths.
