# Competitor Weaving -- Extracting Ideas from Every Library Worth Stealing From

The native default wins by absorbing the best idea from every competitor
without inheriting their ceiling. This file pairs each sprint with concrete
extraction targets and explicit "weave it in" ops. Research is already
catalogued in 07; this file turns research into per-sprint action.

## Competitor matrix

23 competitor pipelines already reimplemented in `dagua/layout/ops/pipelines/`:
fr, kk, fa2, stress_sgd, stress_majorization, sfdp, umap, tsnet, sugiyama,
spectral, classical_mds, pivot_mds, drl, gem, graphopt, lgl, linlog, fmmm,
davidson_harel, maxent_stress, neulay, sgd2_multi, reingold_tilford.

Plus 6 external adapters: graphviz (dot, neato, fdp, sfdp), elk, dagre,
networkx (spring, kamada_kawai), igraph (sugiyama, fr, kk).

Total: 29 points of comparison. All benchmark-cached, so head-to-head
comparison on the iteration suite costs only Dagua's run.

## Per-sprint extraction map

Each sprint comes with (a) the graph family it most targets, (b) the best
competitor(s) on that family per the current benchmark, (c) the specific
technique to extract, (d) the name of the op that captures it.

### Sprint 1 -- Initialization + Gradient Core + Memory Port

Goal families: all. The initializer dominates convergence speed and quality
floor for every family.

Extractions (each -> a registered op):
- **From spectral (NetworkX spectral_layout, scipy eigensolver)**:
  `SpectralInit` op using normalized Laplacian; best for undirected and
  disconnected. Already partially in `dagua/layout/ops/embed.py` as spectral
  embedding; confirm and wire as initializer option.
- **From dagre / grandalf (Sugiyama pipeline)**: `LongestPathLayering` +
  `BarycenterOrdering` combination. Already present as separate ops; bundle
  into `SugiyamaInit` initializer for DAG family.
- **From sgd2_multi pipeline (already in Dagua)**: `WarmStartSGD2` op that
  runs 10-20 sgd2 iterations cheaply and hands positions to the gradient
  core. Works well for near-undirected or medium-density graphs.
- **From UMAP (umap-learn)**: `UMAPInit` op using existing `umap.py` ops as
  a cheap init for clustered undirected graphs. Only if it outperforms
  spectral on the same family.

Gradient core memory port:
- **From legacy `_layout_inner`**: extract per-loss backward pattern into
  `LossPerLossBackward` op.
- **From `torch.utils.checkpoint`**: `GradientCheckpoint` op wrapping a
  loss group.
- **From hybrid device logic in engine.py**: `HybridDeviceOffload` op.

Exit deliverable: per-family dispatcher tested on 20 graphs (5 per family)
shows >=5% composite improvement vs Sprint 0 on 3 of 7 families.

### Sprint 2 -- Multilevel V-Cycle + Hierarchy Memory Parity

Goal families: large-sparse, large-dense, 100K+.

IMPORTANT: the V-cycle's internal coarsening groups are a PERFORMANCE
strategy, NOT user-visible clusters. They never appear in the rendered
output as cluster boxes. User-defined clusters (from `Graph.clusters`)
are handled separately in Sprint 4. If the user provides NO clusters,
Sprint 2 still coarsens for performance but the result is a flat
hierarchy at render time.

Extractions:
- **From SFDP (Graphviz, Hu 2005)**: coarsening schedule -- heavy-edge
  matching plus star contraction; prolongation adds small gaussian noise
  to coarse positions before refinement at finer level. Op:
  `SFDPCoarseningSchedule` wrapping existing `coarsen.py` utilities.
- **From FM^3 (OGDF, Hachul-Junger 2004)**: multilevel step schedule
  (geometric decay of steps per level). Op: `FM3StepSchedule`.
- **From Walshaw multilevel framework**: post-prolongation local refinement
  pass using the SGD2 warm-start pattern. Op: `WalshawLocalRefine`.
- **From cose-bilkent (cytoscape.js)**: cluster-aware coarsening that
  respects user-provided cluster boundaries. Op: `ClusterAwareCoarsen`,
  wired in Sprint 4.
- **From Graphviz sfdp source**: the actual floor on node spacing at each
  level (not just a factor) -- coarse levels use absolute minimum spacing
  tuned to the level's node count.

### Sprint 3 -- Hybrid Classical Steps

Goal families: directed DAG, tree, Sugiyama-style hierarchical.

Extractions:
- **From Graphviz dot (Gansner 1993, network simplex)**: approximate
  network simplex for y-coordinates. Full simplex is hard to make
  differentiable; extract the OBJECTIVE (sum of weight * |rank(tgt) -
  rank(src)|) as a differentiable loss that approximates it. **Multi-op
  bundle (reclassified per adversarial review):**
  (a) `EdgeWeightedRankLoss` -- differentiable surrogate objective
  (not the simplex algorithm itself).
  (b) `DiscreteRankAssignPolish` -- non-differentiable polish that runs
  a classical longest-path + Coffman-Graham on the final rank ordering.
  Labeled approximation + polish, NOT "network simplex extraction".
  Sprint 3 mini-spec defines both.
- **From Brandes-Koepf 2001 + 2020 erratum**: x-coordinate alignment via
  four corner alignments + block structure. Non-differentiable but cheap
  polish op: `BrandesKoepfPolish` after gradient core converges.
- **From OGDF greedy-switch**: post-polish crossing reduction. Op:
  `GreedySwitchRefine`. Drop-in after gradient core on DAG graphs.
- **From ELK layer-sweep**: barycenter -> median -> greedy-switch triad.
  Already partially present; formalize as composable op chain
  `ELKLayerSweepChain`.
- **From igraph layout_sugiyama**: coffman-graham layering as a width
  constraint option. Op: `CoffmanGrahamLayering`.
- **From maxent_stress (already in Dagua)**: entropy-regularized stress
  loss; add as a gradient-core option for undirected family to reduce
  local minima lock-in.

### Sprint 4 -- USER-Cluster-as-Node + Hierarchical Flex

Goal families: nested-shallow, nested-deep, mixed-width -- specifically
graphs that arrive WITH user-defined clusters. Sprint 4 does nothing at
layout time for graphs without user clusters (those just use Sprint 2's
V-cycle auto-coarsening for performance). Extractions below all target
USER-cluster handling, not auto-coarsening.

Extractions:
- **From Sander 1996 (dagre)**: border-node insertion between real-node
  layers. Op: `BorderNodeInsert`. Lets sibling clusters have their own
  layer-local layout without fighting global layer assignment.
- **From ELK compound layout**: recursive per-cluster layout instances.
  **Multi-op bundle (reclassified per adversarial review):** pure ops
  cannot run nested subproblems on a linear Pipeline. Expressed as:
  (a) `ClusterSubproblemConstruct` -- builds a per-cluster
  `LayoutProblem` snapshot.
  (b) `RunNestedPipelineForCluster` -- invokes an inner Pipeline via
  recursion in a dedicated op class; updates cluster-internal positions.
  (c) `ClusterResultMerge` -- writes nested results back to parent
  SolveState via controlled mutation.
  Sprint 4 mini-spec defines required state mutations and the recursion
  depth cap. This IS a controlled bypass of strict composability; the
  sprint exit note documents it as such.
- **From cose-bilkent**: cluster-aware force blending (parent pulls children,
  siblings repel as a group). Op: `ClusterForceBlend`.
- **From cola.js constraints**: hull constraint for cluster containment.
  Op: `ClusterHullConstraint` (differentiable via convex hull soft proxy).
- **From Dogrusoz 2009 force-directed compound**: gravity toward parent
  center. Op: `ParentGravityPull`. Simple, effective for flat clusters.

### Sprint 5 -- Pinning + Flex End-to-End

Goal families: any with user pinning; especially DAGs with mandatory roots.

Extractions:
- **From Dwyer & Koren 2005 (constrained stress majorization)**: hard-pin
  handling via constraint majorization. We already have hard-pin projection
  in `HardPinProjection`; extend to multi-level pin propagation. Op:
  `MultilevelPinPropagate`.
- **From cola.js**: gradient projection for pinned positions. Already in
  hard-pin projection; verify semantics match cola.js behavior for
  simultaneous pins.
- **From Graphviz rank=same constraints**: align-group handling through
  coarsening. Op: `AlignGroupPropagate` that lifts alignment groups to
  super-nodes at coarse levels and re-applies at fine levels.
- **From WebCola**: soft-then-hard constraint sweep ("slack to zero") --
  start constraints as soft losses, anneal to hard. Pattern:
  `PinSlackAnneal` schedule.

### Sprint 6 -- Differentiable Edge Routing

Goal families: skip-heavy, diamond, mixed-width, dense.

Extractions:
- **From Holten & van Wijk 2009 FDEB**: force-directed edge bundling via
  compatibility (angle, scale, position, visibility). Op:
  `FDEBAttraction` operating on control points. Already sketched in
  `loss_functions.md`.
- **From Pupyrev 2013 stub bundling**: confluence at endpoints only. Op:
  `StubBundleConfluence`. Sketched too; finalize differentiable form.
- **From Dickerson-Bach confluent drawings**: track-assignment approach.
  Too architectural to port directly; extract the READABILITY principle
  (shared paths must unambiguously reveal original edges) as a
  post-bundling check.
- **From Graphviz splines**: visibility-graph routing. Non-differentiable
  but cheap polish: op `VisibilityGraphPolish` after differentiable edges
  settle. Catches "edge passes through node" cases the loss misses.
- **From Hobby 1986**: aesthetic curve fitting parameters (curl, tension).
  Op: `HobbyTensionTune` for final polish on Bezier control points.

### Sprint 7 -- Node Size + Text Polish

Goal families: mixed-width, large-dense, nested-deep.

Extractions:
- **From Kakoulis-Tollis 1998**: label placement algorithm with precedence
  rules. Non-differentiable polish op: `KakoulisTollisLabelPlace`.
- **From ELK label placement**: port/edge-label slot reservation. Op:
  `ELKLabelSlots` (heuristic).
- **From Graphviz label handling**: label-aware node sizing before layout
  starts. We already have `compute_node_sizes`; add a feedback loop
  iteration where post-layout label collisions resize nodes and re-run
  layout. **Multi-op bundle (reclassified per adversarial review)**:
  (a) `LabelCollisionDetect` -- measures overlaps, writes to
  SolveState.extras["label_collisions"].
  (b) `NodeSizeExpand` -- reads collisions, produces new sizes in extras,
  does NOT mutate LayoutProblem.
  (c) `LayoutProblemReseed` -- top-level control op that rebuilds a new
  LayoutProblem with expanded sizes and re-invokes the pipeline up to
  2 extra passes.
  This IS a controlled bypass of LayoutProblem immutability; sprint 7
  exit note documents it, and there's a ceiling of 2 feedback passes.

### Sprint 8 -- Scale Ladder Hardening

Goal: 100K, 1M, 10M runs that beat competitors in wall-time at similar
quality.

Extractions:
- **From Gephi ForceAtlas2 (Jacomy 2014) Barnes-Hut**: quad-tree
  repulsion approximation. Op: `BarnesHutRepulsion` as a swap-in for
  the exact + sampled repulsion on N>100K.
- **From cuGraph FA2**: GPU-native Barnes-Hut with warp-level primitives.
  Reference only -- port the algorithmic idea, don't link the library.
- **From SFDP at scale**: coarse-grained coarsening for N>1M (heavier
  edge-weight matching + star contraction). Already partially in
  Sprint 2's `SFDPCoarseningSchedule`; scale up there.
- **From OGDF fast multipole method (FMMM)**: multipole expansion for
  far-field repulsion. Op: `FMMRepulsion` if Barnes-Hut insufficient
  at 10M.

### Sprint 9 -- Aesthetic Dial-In

Goal: tuning off iteration logs + matching best-competitor weights.

Extractions:
- **From OGDF default parameters**: reference parameter ranges for each
  algorithm family. Use as priors for Optuna search space.
- **From ELK published defaults** (Domros 2023): spacing, aspect ratio
  targets. Use as target values in tuning.
- **From published benchmark values (graph-tool, GraphvizRep surveys)**:
  expected metric ranges per family. Sanity-check our results aren't
  outside plausible.

## Authoritative competitor matrix (frozen at Sprint 0.5)

Revised twice per 2026-04-22 adversarial reviews. Uses the ACTUAL adapter
names present in `dagua/eval/competitors/__init__.py`:

| Competitor variant | Adapter module | Family authority | Device |
|--------------------|-----------------|------------------|--------|
| graphviz_dot | graphviz_competitor | directed DAG (small/medium) | CPU |
| graphviz_sfdp | graphviz_competitor | large undirected + multilevel | CPU |
| graphviz_neato | graphviz_competitor | small undirected stress | CPU |
| graphviz_fdp | graphviz_competitor | force-directed undirected | CPU |
| elk_layered | elk_competitor | deep hierarchical, compound | CPU (JVM) |
| dagre | dagre_competitor | JS user parity, compound | CPU (Node) |
| igraph_sugiyama | igraph_competitor | fast C-based directed | CPU |
| igraph_fr | igraph_competitor | fast C-based FR | CPU |
| igraph_kamada_kawai | igraph_competitor | fast C-based Kamada-Kawai | CPU |
| nx_spring | networkx_competitor | Python spring reference | CPU |
| nx_kamada_kawai | networkx_competitor | Python KK reference | CPU |
| sgd2_multi_ref | sgd2_multi_competitor | differentiable general-purpose | CPU/GPU |
| gephi_yifanhu | gephi_competitor | multilevel force-directed | CPU |
| fa2_ref | fa2_competitor | Barnes-Hut ForceAtlas2 reference | CPU |
| ogdf_fmmm | ogdf_competitor | FMMM multilevel reference | CPU |
| cytoscape_fcose | cytoscape_fcose_competitor | compound + constraint | CPU |

This is the AUTHORITATIVE set. All 16 entries bind the sprint-exit
head-to-head gate (see head-to-head section below).

Version / binary hash / date recorded per competitor at every refresh.
`scripts/refresh_competitors.sh --capture-versions` writes the manifest.

Device-normalization rule (Q18 resolved at Sprint 0.5): same-device
comparison only. Dagua-CPU vs CPU competitors; Dagua-GPU vs GPU competitors
(currently only sgd2_multi_ref on GPU; cuGraph if linkable in Sprint 8).
Per-graph competitor binding is to the registered name in
`dagua/eval/competitors/__init__.py`; aliases used in plan prose match
those names exactly.
Per-family device choice:
- Small DAG / tree (N<=1K): Dagua CPU vs graphviz_dot/elk/dagre/igraph CPU
- Medium DAG (1K<=N<=20K): same-device CPU
- Large undirected (N>=20K): GPU-preferred for Dagua
- Ultra (N>=100K): GPU required, sgd2_multi GPU as peer

## Refresh protocol (single authoritative policy)

Revised round 2: the earlier "Sprint 0, 5, 9 only" rule is superseded.
The binding policy is:

Every sprint exit runs `scripts/refresh_competitors.sh --check`. If any
competitor's binary hash differs from the committed manifest, the sprint's
competitor cache is invalidated and a refresh runs before Pareto scoring.

Full refresh (re-running all 16 competitors on iteration + held-out)
happens:
- Automatically at Sprint 0.5 exit (first authoritative baseline).
- On any hash change detected by `--check`.
- Manually at Sprint 5 (midpoint health) and Sprint 9 (release).

A full refresh costs 2-4 hours; it runs in background via dispatch.sh,
not inside the iteration loop's clock budget.

## Head-to-head competitor benchmark at every sprint exit

Revised round 2: binds to the FULL 16-competitor authoritative matrix
above, not a subset. Sprint exit runs dagua head-to-head against every
competitor in the matrix (cached competitor results; only dagua re-runs
on graphs where the cache is valid).

For each graph in the iteration + held-out suite we record:
- composite score per competitor
- runtime per competitor (cached; invalidated by `--check` version hash)
- Pareto classification: dagua optimal / dominated / tied

Exit gates (per 10_iteration_loop.md): measured against the full
authoritative matrix. Ramp calibrated from Sprint 0.5 baseline, Sprint 9
target 90% iter / 80% held-out, family floors enforced.

(Refresh protocol is defined above, in the "Refresh protocol" section.
No duplicate rule here.)

## Competitor adapter correctness

Before trusting any head-to-head comparison:
- Adapter seed produces different outputs for different seeds (2-seed
  sanity check). From gotchas: this has bit us. See
  `.project-context/knowledge/gotchas.md` [BENCH] entries.
- Adapter config is audited at Sprint 0 and after any refresh.
- External tool subprocess paths verified (graphviz, elk, dagre, igraph).

## What we don't extract

- Competitor UX / API surface: irrelevant to the placement algo.
- Competitor rendering: we have our own.
- Competitor parameter naming conventions: we have `LayoutConfig`.
- Paper-only ideas without installed implementations, unless a sprint
  explicitly needs one AND the paper is unambiguous (rare).

## Extraction success criteria

An extraction is "successful" only when:
1. It lands as a registered op (not inline).
2. Its unit test covers the extracted idea against a hand-computed reference.
3. Its ablation (with vs without op in the pipeline) shows measurable gain
   on at least one family in the sprint's iteration loop.
4. It does not regress any other family > 3%.

Failed extractions are documented in the sprint exit note as "attempted,
did not land" with a one-line reason. These are valuable -- they prevent
re-attempting the same dead end.

## The north-star question

After every iteration, ask: is Dagua Pareto-optimal on the weak graph yet?
If not, and we've tried 3 Dagua-native hypotheses, the FOURTH hypothesis
is mandatory: look at the best competitor on this graph and extract. No
graph stays weak because we were too proud to borrow.
