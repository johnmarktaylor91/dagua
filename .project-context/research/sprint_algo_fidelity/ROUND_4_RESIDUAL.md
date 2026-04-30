# Round 4 Residual: fmmm-vs-fdp

## Classification

`attempted_lever_no_signal: graphviz_fdp_K_alignment`

Stay on the `fdp` family for one more round with a different lever. The most
likely remaining gaps are Graphviz fdp's exact FR force law, initialization
distribution, and post-layout overlap expansion, not the scalar default `K`
alone.

## Baseline

Two live runs were deterministic:

| Run | Median RMSD | Worst graph | Worst RMSD | Graphs > 0.15 |
|---|---:|---|---:|---:|
| baseline_run1 | 0.247474 | center_port_backedge_hub | 0.440077 | 20 |
| baseline_run2 | 0.247474 | center_port_backedge_hub | 0.440077 | 20 |

This is lower than Round 1's cached median `0.2918`, but preserves the fdp
family floor pattern.

## Lever Tried

Graphviz fdp documents `K=0.3` inches as its default spring constant. Its source
initializes edge `len` from `fdp_parms->K` and uses `DFLT_K 0.3`, so the direct
analogy to Round 3 was to align Dagua FMMM's refinement ideal length to
`0.3 * 72 = 21.6` points instead of deriving it from the drawing extent.

Temporary change tested:

- Added a Graphviz fdp ideal-edge constant of `21.6` points.
- Set FMMM `refinement_area = (ideal_edge_length * sqrt(N)) ** 2`, so
  `_fr_ideal_length(area, N)` became the Graphviz default edge length.

## Result

The lever did not meet the commit criterion and was reverted.

| Metric | Before | After attempted lever | Delta |
|---|---:|---:|---:|
| Median RMSD | 0.247474 | 0.251977 | +0.004503 |
| Worst RMSD | 0.440077 | 0.443168 | +0.003092 |
| Graphs > 0.15 | 20 | 18 | -2 |

Median and worst graph both regressed, so no code change was committed.

## Diagnosis

The failed `K`-only alignment suggests the fdp floor is not primarily a scalar
ideal-length mismatch. Dagua's FMMM implementation still differs mechanically
from Graphviz fdp in ways that affect shape after Procrustes normalization:

- Graphviz fdp is documented as Fruchterman-Reingold with a multigrid solver,
  while Dagua's current default is OGDF-style FMMM with logarithmic attraction.
- Graphviz fdp source uses a random rectangle/ellipse initialization scaled by
  `K * (sqrt(N) + 1)`, while Dagua seeds the coarsest/single level through the
  FR pipeline and then refines.
- Graphviz fdp has a second expansion/overlap phase (`fdp_xLayout`) that Dagua
  FMMM does not mirror.

## Recommendation

Round 5 on fdp should test one of these focused levers:

1. Add a `graphviz_fdp` force model matching Graphviz source formulas:
   repulsion proportional to `K^2 / d^2` in vector scale and attraction
   proportional to `(d - len) / d`.
2. Add Graphviz fdp-style random initialization for single-level graphs before
   refinement.
3. Investigate whether overlap expansion, not force equilibrium, dominates the
   cached graphviz_fdp target positions for labeled graphs.
