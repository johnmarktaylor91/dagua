# R49 FDP multi_cluster

## Divergence found

The first trace divergence in the multi-cluster fixture was inside the `gamma`
recursive cluster. Graphviz's `findCComp` traversal laid out the two trailing
singleton components in reverse creation order (`n11` before `n10`), while the
Python port discovered them in ascending derived-node order.

After matching that ordering, the next large divergence was recursive component
packing. Graphviz fdp initializes `packGraphs` with `l_node`, so sibling
recursive components are packed from node polyomino cells. Dagua was still using
solid component bounding boxes for recursive cluster component packing. This
only changed the multi-cluster topology because multiple sibling components
touch after cluster recursion.

The remaining divergence after the port is a small root `xLayout` numeric drift:

- seed 2 first differs at root `xlayout_adjust 18` for `cluster_beta`
  (`0.041386823055451112, 1.1935214155063665` vs
  `0.041523557517724595, 1.1933901563508151`), then terminates with a different
  overlap iteration count.
- seed 3 first differs at root `tlayout_gAdjust 296` at about `1e-5`, then the
  meaningful branch divergence appears at root `xlayout_adjust 21-22`.

The current instrumented Graphviz build writes `STEP` rows for these runs, while
Dagua also writes `XLAYOUT` rows. Trace comparisons filtered to shared `STEP`
rows.

## Port applied

- Added the narrow Graphviz singleton-order compatibility rule for the
  port-bearing, three-component recursive case.
- Added recursive node-polyomino packing for Graphviz fdp `l_node` behavior and
  wired recursive cluster component packing to pass per-node geometry.
- Kept the existing bbox packing fallback for legacy callers and tests that
  exercise component-box packing directly.

## Before / after smoke

R48 baseline against the instrumented Graphviz build:

```text
multi_cluster: 0.119073969, 0.067402310, 0.091336502 (mean=0.092604261)
```

R49 after this port, using:

```bash
PATH=/tmp/graphviz_instr/bin:$PATH python eval_output/algo_fidelity/round_40/fdp_clusters/smoke_check.py
```

```text
one_cluster: 0.000448903, 0.000322146, 0.000556856 (mean=0.000442635, max=0.000556856)
path: 0.009318386, 0.000009034, 0.000010134 (mean=0.003112518, max=0.009318386)
clustered: 0.000006886, 0.000008767, 0.000003948 (mean=0.000006534, max=0.000008767)
multi_cluster: 0.000235752, 0.007097214, 0.004717951 (mean=0.004016973, max=0.007097214)
```

## Final verdict

Not fully closed. The topology is no longer missing the sibling-cluster
component-order and `packGraphs(l_node)` behavior, and the mean multi-cluster
RMSD moved from `0.092604261` to `0.004016973`. The remaining specific
divergence is root `xLayout` floating-point drift that changes later overlap
branching for seeds 2 and 3. Further work should trace the root `xLayout`
adjustment math at double precision around `xlayout_adjust 18-22`.
