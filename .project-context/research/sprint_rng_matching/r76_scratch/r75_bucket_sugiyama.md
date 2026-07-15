# BUCKET: sugiyama (129 divergent combos <=300 nodes + a large-graph tail -- THE structural prize)

Target list: r75_targets_sugiyama.json (BUCKET=sugiyama). This bucket is believed GENUINELY
divergent (not metric artifact): e.g. ba_500 dagua 22344 crossings vs igraph 2805. Prior rounds
already fixed cycle-breaking (iterative, r74 6563d98) and the igraph LP objective (169ce7b).

Dagua side: dagua/layout/ops/pipelines/sugiyama.py + ops (ranking, ordering, coordinate ops).
References: BOTH graphviz dot (lib/dotgen/) and igraph's Sugiyama -- split your analysis by
variant family (classic_sugiyama_graphviz_fidelity* vs classic_sugiyama_igraph* vs default):
the target JSON's engine field tells you which reference each combo compares against, and the
fix path differs per family.

Your deliverable is a STAGED PORT SPEC, not just a diagnosis. The Sugiyama pipeline has 4 stages
(cycle-break -> ranking -> crossing-minimization ordering -> x-coordinate assignment); positions
diverge at the FIRST stage that differs, so:

1. DIVERGENCE-STAGE INVENTORY: for a representative sample of failing combos (small graphs
   first: binary_tree, small DAGs), determine the FIRST stage where dagua's intermediate output
   differs from the reference's. Practical approach for graphviz: dot can dump ranks/order via
   debug output, or instrument a tiny C harness, or reason from final positions (same y-ranks
   but different x -> divergence is at ordering or x-coord; different ranks -> ranking).
   For igraph: igraph's sugiyama is pure C (src/layout/sugiyama.c) -- read it and compare
   stage-by-stage against dagua's igraph-fidelity variant ops.
2. GRAPHVIZ X-COORD PORT SPEC: graphviz assigns x-coords via a second network-simplex pass on an
   auxiliary graph (lib/dotgen/position.c, set_xcoords, plus virtual-node chains and omega
   weights from class1/class2.c). Spec out precisely: auxiliary graph construction (aux nodes per
   edge, omega weight table by node-type pair), the network simplex reuse (lib/common/ns.c),
   priority/tie-break details, flat-edge handling (flat.c), cluster/port constraints we can skip.
   Estimate port size (LOC, which dagua ops change/added) and a stage-by-stage verification plan
   (rank match -> order match -> x match on a ladder of graphs).
3. MINCROSS DELTA: compare dagua's ordering op against lib/dotgen/mincross.c specifics --
   init_order DFS details, median heuristic (in/out passes, weighted medians of -1 handling),
   transpose() with reverse sweep alternation, iteration counts (MinQuit=8, MaxIter=24,
   Convergence=.995), local-opt exchange conditions. List every concrete behavioral difference
   found (file:line both sides).
4. IGRAPH FAMILY: same but for igraph's implementation (ranking via longest-path? ordering via
   barycenter/median variant? x-coords via its LP with integrality? -- read the actual source,
   src/layout/sugiyama.c) vs dagua's igraph-fidelity ops. r74's LP-objective fix already landed;
   what remains?
5. For the crossings-only failures (11 combos fail ONLY crossings): are these ordering-stage
   divergence with coincidentally-matching stress? Would the mincross deltas in (3) explain them?

Answer explicitly: if we land (2)+(3) faithfully, which fraction of the 129 (+large-graph tail)
should become bit-exact or distributionally matched? What CANNOT be fixed by these ports (list
combos + why)? Note the free-aspect anisotropic-Procrustes exception already forgives per-axis
scale for sugiyama variants, so axis-scale is NOT a divergence source to chase.
