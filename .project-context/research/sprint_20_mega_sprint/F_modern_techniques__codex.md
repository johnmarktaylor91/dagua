# Sprint 20 Agent F: Modern Graph-Drawing Techniques Not Yet Used by Default

Date: 2026-04-24

Scope: literature survey for graph-drawing techniques from roughly 2022-2026,
plus older underused methods explicitly named in the dispatch, focused on
Dagua's remaining tail losses:

- `ragged_feature_pyramid`: Dagua 69.52 vs elk 79.56, delta -10.04.
- `planar_60`: Dagua 65.82 vs elk 75.06, delta -9.25.
- `small_world_100`: Dagua 48.58 vs igraph_sugiyama 57.08, delta -8.51.
- `small_world_500`: Dagua 49.34 vs elk 54.16, delta -4.82.
- `regular_3_30`: Dagua 68.37 vs dot 72.23, delta -3.86.
- `hexagonal_lattice_42`: Dagua 85.21 vs dot 88.99, delta -3.77.

Assumption: I interpret "not implemented" as "not used by the default native
pipeline, not sufficiently integrated for topology dispatch, or missing a key
modern variant." Dagua already has opt-in implementations for several named
families.

## TL;DR

1. **Highest impact per effort: add a clean undirected force/stress lane and
   dispatch small-world/regular graphs to it.** Dagua has FR, FA2, SFDP,
   MaxEnt-Stress, Stress-SGD, SMACOF, Pivot-MDS, LinLog, DrL, FMMM, etc. in
   `dagua/layout/ops/pipelines/__init__.py`, but the default `dagua_native`
   lane still starts from layered native initialization and only adds a flat
   2D fallback. The largest non-DAG losses are exactly where a dedicated
   force-directed/stress lane should win.
2. **Use Pivot-MDS or MaxEnt-Stress as an initializer, not merely as opt-in
   algorithms.** Dagua already has Pivot-MDS prep and MaxEnt-Stress pipelines,
   but the native default only uses pivot distances when `w_stress > 0`.
   For `small_world_100/500` and `regular_3_30`, Pivot-MDS -> short
   Stress-SGD or t-FDP/FA2 refinement is the most conservative improvement.
3. **Planar/lattice losses need an embedding-aware lane, not more DAG
   crossing polish.** `planar_60`, hexagonal lattice, Sierpinski-like graphs,
   and `ragged_feature_pyramid` need planarity testing/embedding, Tutte or
   circle-packing/conformal seeds, and then metric-preserving relaxation. The
   current classifier only uses a sparsity hint and tags such as
   `lattice_like`/`planar_dag`; it does not compute a planar embedding.
4. **The modern 2022-2026 additions worth prototyping are t-FDP (Zhong et al.
   2023), FORBID overlap removal (Giovannangeli et al. 2022), CoRe-GD-style
   hierarchical learned refinement (Grotschla et al. 2024), GUMAP/NNP-NET
   ideas (2025), and GPU FA2 interop.** Of these, t-FDP and FORBID compose
   cleanly with Dagua's differentiable/non-differentiable spirit.
5. **Skip full neural layout as a default for this sprint.** DeepGD/SmartGD and
   CoRe-GD are real, but training data, model packaging, determinism, and
   dependency load are high. Use their ideas as seed/refinement modules or
   offline candidate generators first.

## Existing Pipeline Coverage

The registry currently exposes 24 named pipelines in
`dagua/layout/ops/pipelines/__init__.py`:

- `classical_mds`: exact all-pairs distance MDS. Covered for small/medium
  metric embeddings, but O(N^2) memory/time makes it a seed only.
- `davidson_harel`: simulated annealing layout. Covered but expensive; not an
  obvious default candidate for 93-graph CPU h2h.
- `drl`: Distributed Recursive Layout. Covered as a large-graph force family.
- `dagua_native`: default pipeline. It is layered/DAG-biased, with Adam
  gradient core, pivot-stress optional prep, median/transpose ordering,
  Brandes-Kopf x-refinement, dummy-node long-edge handling, overlap projection,
  aspect fitting, and component tiling.
- `fa2`: ForceAtlas2. Covered as an opt-in force-directed pipeline.
- `fmmm`: FM^3 multilevel force-directed. Covered as an opt-in classic
  multilevel method.
- `fr`: Fruchterman-Reingold. Covered.
- `gem`: GEM graph embedder. Covered.
- `graphopt`: GraphOpt force-directed. Covered.
- `kk`: Kamada-Kawai with LBFGS machinery. Covered.
- `lgl`: Large Graph Layout. Covered.
- `linlog`: LinLog energy. Covered for community separation.
- `maxent_stress`: Gansner/Hu/North MaxEnt-Stress. Covered as an opt-in
  majorization/gradient pipeline; not a default initializer/refiner.
- `neulay`: neural layout family exists in the archive/pipeline set; not a
  modern GNN default.
- `pivot_mds`: Brandes/Pich-style Pivot-MDS. Covered as an opt-in pipeline and
  as native pivot-distance prep when stress loss is enabled.
- `reingold_tilford`: tidy tree. Covered.
- `sfdp`: Graphviz-style scalable force-directed placement with multilevel
  hierarchy and Barnes-Hut. Covered as opt-in.
- `sgd2_multi`: multicriteria scalable graph drawing via SGD. Covered as
  opt-in and potentially relevant to crossing/angle/stress tradeoffs.
- `spectral`: spectral layout. Covered.
- `stress_majorization`: dense SMACOF. Covered for smaller connected graphs.
- `stress_sgd`: Zheng/Pawar/Goodman Stress-SGD. Covered, including approximate
  pivot branch.
- `sugiyama`: classical layered. Covered.
- `tsnet`: t-SNE network embedding. Covered.
- `umap`: UMAP graph layout. Covered.

The important implementation detail is that coverage is not the same as use.
The native pipeline still has a single broad lane. The sprint context says it
now performs: native init -> `Force2DInitIfFlat` for collapsed cyclic graphs ->
optional dummy expansion -> Adam gradient core -> layered crossing reducers ->
Brandes-Kopf x refinement -> overlap projection -> aspect ratio fit. In the
source, `Force2DInitIfFlat` is explicitly a rescue for small-world/social-net
graphs collapsed to one y-layer. That is a useful patch, but the algorithmic
center of gravity remains layered. The default does not dispatch small-world
graphs into a true undirected force/stress pipeline, nor does it dispatch
planar/lattice graphs into a planarity-preserving embedding pipeline.

## Detailed Survey Table

| Technique | Paper(s), year | Expected impact bucket | Runtime cost | Dagua fit and integration point | Diff spirit? |
|---|---|---|---|---|---|
| GNN-based initialization | DeepGD, Wang/Yen/Hu/Shen 2021/2023; SmartGD, Wang/Yen/Hu/Shen 2022/2024; CoRe-GD, Grotschla/Mathys/Veres/Wattenhofer 2024; GraphSAGE, Hamilton/Ying/Leskovec 2017 | general, small_world, regular | Training expensive; inference roughly O(E) per GNN layer plus coarsening for CoRe-GD | Add as optional initializer/candidate generator, not default. Use model output as seed for existing differentiable core. | Yes, if model is PyTorch and final refinement remains Dagua loss-based; risky for packaging/determinism. |
| Pivot-MDS + local refinement | Brandes and Pich 2007; common in tsNET/MaxEnt pipelines | small_world, regular, general 100-500 nodes | O(P(E+N)) BFS + small SVD; cheap for P=32-64 and N<=500 | Already implemented. Promote to default initializer for flat, connected, undirected-ish graphs, then Stress-SGD/FA2/t-FDP refinement. | Yes. Distance query non-diff, refinement diff/non-diff as selected. |
| MaxEnt-Stress | Gansner, Hu, North 2012/2013 | small_world, regular, general | Exact branch dense for small graphs; pivot approximation scales; more costly than FA2 but stable | Already implemented. Use for high-dimensional/small-world graphs with overlap/clutter risk, or as short refiner after Pivot-MDS. | Mostly yes. Majorization is non-Adam but objective is clear and composable. |
| Constrained stress / IPSep-CoLa | Dwyer, Koren, Marriott 2009; Dwyer and Marriott diagonally scaled gradient projection 2008; fCoSE, Balci/Dogrusoz et al. 2021/2022 | planar, ragged pyramid, constrained DAGs, grids | Moderate; projection/separation constraints add O(C log C) to O(C^2) depending solver | Dagua has flex/pin/align and projection, but not a full separation-constraint stress solver. Add x/y separation constraints for grid/lattice/layer order. | Strong fit: Dagua already mixes differentiable losses and projections. |
| Neural force fields / DRGraph-like negative sampling | DRGraph, Zhu/Chen/Hu/Hou/Liu/Zhang 2020/2021; t-FDP, Zhong et al. 2023; CoRe-GD 2024 | small_world, large general, regular | DRGraph/t-FDP target near-linear or N log N with approximations; FFT/GPU variants fast but implementation-heavy | Do not replace everything. Add t-FDP-style bounded short-range repulsion as a force op/refiner, or DRGraph negative sampling to Stress-SGD. | Good for t-FDP/negative sampling; trained neural fields are higher risk. |
| Topology-preserving planar drawing | Booth/Lueker 1976 PQ trees; Chiba/Nishizeki/Abe/Ozawa 1985 PQ embedding; Boyer-Myrvold 2004; Tutte spring theorem; Chrobak-Payne/Schnyder-style grid drawings; Eppstein/Goodrich/Illickan 2024 Bezier planar/1-planar | planar, lattice, hex, Sierpinski, planar_60 | Planarity test O(N); embedding O(N); Tutte solve sparse linear system | Missing as default. Add actual planarity test/embedding, then seed coordinates by Tutte/canonical grid. Use Dagua refinement with crossing guard. | Mixed: embedding is non-diff, refinement can be diff. Very aligned with "non-diff when it boosts performance." |
| Constraint-based grids / orthogonal layout | Tamassia 1987; OGDF topology-shape-metrics; fCoSE 2022; bend-minimum series-parallel work by Didimo et al. 2022 | planar, regular, ragged pyramid, grids | Orthogonal TSM can be expensive but fine for N<=500; exact bend-minimum often class-restricted | Use only for classified low-degree planar/lattice graphs. Seed with grid/orthogonal coordinates, then optional straight-line relaxation or edge routing. | Mostly non-diff, but can feed diff final polish. |
| GPU-accelerated multilevel / RAPIDS cuGraph | RAPIDS cuGraph ForceAtlas2; Graphistry/cuGraph docs; BigGraphVis 2021 | small_world_500+, large general | Very fast with CUDA, but dependency/platform heavy; CPU fallback required | Since Dagua is already PyTorch/GPU-aware, integrate conceptually first: batched torch FA2/Barnes-Hut or optional backend. | Yes if torch-native; external cuGraph less composable. |
| Differentiable Sinkhorn assignment/order | Gumbel-Sinkhorn, Mena et al. 2018; Cuturi et al. 2019 differentiable sorting/ranking by OT; SoftSort, Prillo/Eisenschlos 2020 | layered x-order, ragged pyramid, planar order | O(L * k^2) per layer for Sinkhorn iterations; manageable for layer widths under a few hundred | Use for soft x-order/crossing reduction where current median/transpose is discrete. Better as experiment than first-line fix. | Very high: pure torch and differentiable. |
| Newton/LBFGS/CG refinement after Adam | Kamada-Kawai Newton-Raphson 1989; Gansner/Koren/North stress majorization 2004; Zheng/Pawar/Goodman Stress-SGD 2019 | general, small_world, regular | Small graphs: cheap. Dense Hessian not ok; LBFGS/CG sparse is ok. | Dagua has LBFGS in KK and Adam in native. Add final 10-30 step LBFGS/CG on stable losses for N<=500 candidates. | Yes, optimizer swap only. |
| Multi-start ensemble and candidate selection | Standard force-directed practice; metric-based selection; relates to SmartGD multi-aesthetic evaluation | all tail losses, especially small_world/regular | K times selected cheap pipelines; K=4-8 feasible for N<=500 | Cheapest big win: run Pivot-MDS, FA2, SFDP, native, MaxEnt short, maybe planar seed; score with existing composite proxy; keep best. | Excellent. Non-diff outer loop, diff/non-diff inner candidates. |
| Conformal/circle-packing seeds | Koebe-Andreev-Thurston theorem; Stephenson circle packing; Renssen 2019 exact algorithm; conformal uniformization work 2023 | planar lattice, Sierpinski, hex | Planarity/triangulation plus iterative packing; moderate; N<=500 fine | Use as seed for 3-connected/triangulated planar graphs and Sierpinski/hex-like families. Not a default all-graph method. | Non-diff seed, diff polish. |
| Edge-bundling-aware drawing | FFTEB, Lhuillier/Hurter/Telea 2017; Pixel-Based Edge Bundling, Wu et al. 2023; Eppstein/Goodrich/Illickan 2024 curved planar edges | visual clutter, dense/parallel cycles | Mostly render/postprocess; FFT/WebGL fast but does not improve straight-line metrics directly | Dagua metrics reward straightness/crossings, so bundling can hurt unless metrics support routed/curved edges. Use in renderer, not layout score optimization. | Not layout-diff; rendering-stage non-diff. |
| FORBID overlap removal | Giovannangeli/Lalanne/Giot/Bourqui 2022 | general, labels, dense small graphs | SGD overlap removal; cheap postprocess | Dagua already has overlap projection, but FORBID's stress+scaling objective may preserve topology better. Good postprocess for label-sized nodes. | Strong fit: stochastic gradient objective plus projection-style postprocess. |
| GUMAP / UMAP graph layout refinements | SS-GUMAP/SL-GUMAP/SSSL-GUMAP 2025; Dagua already has UMAP | small_world, general | Faster than tsNET per paper claims; similar to UMAP negative sampling | Dagua has UMAP. Investigate if current UMAP misses graph-specific schedules/large-graph variants; use as candidate in ensemble. | Yes. |
| NNP-NET / neural t-SNE acceleration | Hartskeerl/Mchedlidze/van Wageningen/Vangorp/Telea 2025 | large general, tsNET replacement | Training/inference complexity; intended for very large graphs | Low relevance to N<=500 tail losses; useful later if tsNET becomes bottleneck. | Partially. |
| Resistance-distance stress / Omega | Recent 2025 arXiv "Graph Drawing Stress Model with Resistance Distances" | regular, lattice, general | Claims linear-time embedding plus sampled SGD; experimental | Potentially useful for regular/lattice where shortest-path distances can overconstrain. Needs validation; not first sprint target. | Yes if implemented as sampled stress objective. |

## Technique Notes and Fit

### 1. GNN-based initialization

DeepGD (Wang, Yen, Hu, Shen, 2021/2023) proposes a GNN framework that maps graph
structure to layouts and trains against multiple aesthetics. SmartGD (same
group, 2022/2024) moves further toward GAN-based optimization of differentiable
and non-differentiable goals. CoRe-GD (Grotschla et al., 2024) is more directly
interesting for Dagua because it is hierarchical and stress-oriented: coarsen,
draw coarsest graph, then refine/uncoarsen with GNN message passing and
positional rewiring.

For Dagua's immediate losses, a trained neural replacement is too large a bet.
It needs training data, stored weights, versioning, CPU/GPU determinism, and a
fallback story. However, the initialization idea is sound. The low-risk variant
is not "learn the whole layout" but "learn or compute one better seed, then let
Dagua's metric/loss core polish." A practical non-training proxy is
GraphSAGE/role features -> PCA/MDS -> Dagua local refinement. Features could be
degree, clustering coefficient, eccentricity/pivot distances, layer index when
available, and component id. This mimics the inductive embedding flavor without
shipping a model.

Expected impact: medium for `small_world_100/500` and `regular_3_30` if used as
a seed among candidates; low for `planar_60` unless combined with planarity
features. Effort: high for true DeepGD/CoRe-GD; medium for feature-embedding
seed.

### 2. Pivot-MDS + local refinement

Brandes and Pich (2007) introduced a sampling-based approximation to classical
MDS using pivots, producing fast large-graph layouts and progressive
refinement. Dagua already has `pivot_mds` and native pivot-distance prep. The
gap is dispatch: Pivot-MDS is not the default seed for no-hierarchy graphs.

This is the cleanest small-world story. Small-world graphs are often high
diameter-ish locally but no DAG hierarchy exists. Layered initialization is a
poor inductive bias; random-y rescue is not enough. Pivot-MDS gives global
geodesic structure, then a short local force/stress stage improves edge-length
CV and angular resolution. For `small_world_500`, P=32 or P=64 BFS rows are
cheap. For `small_world_100`, exact or pivot stress is fine.

Integration sketch: in topology resolution, add a `flat_undirected_like` or
`small_world_like` tag when directed acyclicity is false or layer count <= 2,
edge-to-node ratio is sparse/moderate, and connected component count is one.
Run `PivotMDSInit(n_pivots=min(64, sqrt(N)*4))`, normalize, then `StressSGD`
or a new force refiner for 100-300 iterations. Keep native as another candidate
until h2h proves dispatch safe.

Expected impact: high for `small_world_100/500`, medium for `regular_3_30`.
Effort: low because most pieces exist.

### 3. MaxEnt-Stress

Gansner, Hu, and North (2012/2013) target a known stress failure mode: approximate
stress/MDS can overlap nodes or compress degrees of freedom in high-dimensional
graphs. MaxEnt-Stress adds an entropy-like repulsive term to avoid clutter while
preserving graph distances. Dagua already has `maxent_stress`, with majorization
and gradient branches.

The likely opportunity is using MaxEnt-Stress as a specialized force/stress
lane rather than an opt-in algorithm. It is particularly suited to
`small_world_100/500` because graph-theoretic distances in small-world graphs
are compressed and many node pairs look similarly close; maximum-entropy
repulsion discourages degenerate clusters. It may also help `regular_3_30`,
where symmetry and uniform edge lengths matter.

Expected impact: medium-high for small_world, medium for regular, low for
planar lattices if it destroys embedding. Runtime: medium. Recommended as one
ensemble candidate, not sole default.

### 4. Constrained stress and IPSep-CoLa

Dwyer, Koren, and Marriott (2009) combine stress majorization with linear
constraints via gradient projection. The literature around CoLa/IPSep focuses
on separation constraints, orthogonal ordering, non-overlap, and maintaining
mental-map or user constraints. fCoSE (Balci/Dogrusoz et al., 2021/2022) is a
modern compound graph layout with constraint support.

Dagua has flex constraints, pin/alignment losses, and hard projections, but it
does not appear to have a full separation-constraint stress optimizer that can
say "these two nodes must remain in this x/y order by at least gap g" while
optimizing stress. This matters for `ragged_feature_pyramid` and lattice/planar
families because the issue is often not a missing attractive force; it is that
the solver can violate a combinatorial order that ELK/dot/orthogonal methods
preserve by construction.

Integration sketch: add a topology-derived separation constraint stage for
selected graph families. For layered/pyramid graphs, constraints come from
layer x-order and parent-child intervals. For grids/lattices, constraints come
from inferred grid rows/columns or planar embedding faces. Use existing
projection style first: after every K gradient steps, project x/y coordinates
onto separation constraints. Then consider a proper IPSep solver if projection
is too unstable.

Expected impact: high for `ragged_feature_pyramid`, medium-high for planar and
regular. Effort: medium-high.

### 5. t-FDP and DRGraph-style force fields

DRGraph (Zhu et al., 2020/2021) casts graph layout as nonlinear dimensionality
reduction with sparse distance approximation, negative sampling, and multilevel
optimization. t-FDP (Zhong et al., 2023) revisits force-directed layouts with a
Student-t-distribution-based bounded short-range force. The t-FDP paper reports
better neighborhood preservation with low stress and an efficient FFT/GPU
implementation.

Dagua already has UMAP/tsNET, ForceAtlas2, SFDP, FMMM, and Stress-SGD, so the
new piece is not "another force-directed algorithm" but the bounded short-range
force. Current repulsion can over-expand or produce round blobs; bounded
short-range behavior can preserve neighborhoods without excessive local
explosion. That maps well to `small_world_500` and regular sparse graphs.

Integration sketch: implement a `TForceRefine` op as a local refiner after
Pivot-MDS or spectral initialization. Start CPU/torch dense for N<=500, no FFT.
Expose t-force parameters in `algorithm_params` or the topology lane. If it
improves target graphs, later add FFT/GPU acceleration.

Expected impact: high for small_world/regular if tuned; medium effort.

### 6. Planarity-preserving drawing via actual embeddings

The planar tail losses are the strongest evidence that Dagua needs graph
topology, not just continuous optimization. The current classifier uses a
planar sparsity hint (`E < 3N - 6`) and DAG/lattice tags based on max degree,
edge ratio, layer counts, and layer width CV. It does not compute a planar
embedding or preserve faces. Planar graph layout algorithms based on PQ-trees,
Boyer-Myrvold planarity testing, canonical orderings, Tutte embeddings, and
grid drawings solve a different problem: preserve the embedding first, optimize
geometry second.

For `planar_60`, a true planar lane could beat ELK/dot by eliminating crossings
at the source. For hexagonal and Sierpinski families, embedding preservation
also protects the lattice holes/faces that force-directed layout can distort.
Tutte embeddings are especially attractive: fix the outer face to a convex
polygon, solve a sparse linear system for interior vertices, get crossing-free
straight-line drawings for 3-connected planar graphs. For non-3-connected
graphs, use block decomposition or a planar drawing library/algorithm to seed.

Integration sketch: add a planarity backend behind the classifier. First
prototype using NetworkX planarity if dependency policy allows, or a small
Boyer-Myrvold implementation if not. Produce a combinatorial embedding and
outer face. Seed with Tutte/canonical coordinates. Then run constrained
low-weight stress/edge-length polish with crossing guard; if a move creates
crossings, reject or project back. This is non-differentiable, but exactly in
Dagua's stated spirit.

Expected impact: high for `planar_60` and lattice family. Effort: high but
strategic.

### 7. Orthogonal/grid layout for low-degree planar graphs

OGDF's topology-shape-metrics tradition and orthogonal layout algorithms are
not new, but they remain what production competitors use when graphs are
diagrammatic or low-degree planar. The dispatch topic names `ragged_feature_pyramid`,
`planar_60`, and `regular_3_30`; these are exactly cases where a grid or
orthogonal skeleton may outperform a continuous force model under the composite
metric because it controls edge direction, uniform spacing, and crossings.

This does not mean Dagua should become OGDF. A lighter approach is to infer
grid coordinates for lattice-like graphs and preserve row/column separation.
For degree <= 4 planar graphs, an orthogonal seed can avoid the "near-planar
but skewed" look. For triangular/hex lattices, a 60-degree axial coordinate
system is better than 90-degree orthogonal layout.

Expected impact: medium-high for planar/lattice/regular; high for ragged
pyramid if it is diagrammatic. Effort: medium for grid inference; high for full
TSM orthogonal layout.

### 8. GPU-accelerated multilevel and cuGraph

RAPIDS cuGraph exposes GPU ForceAtlas2, and Graphistry wraps cuGraph and its
own GPU layouts for interactive large graphs. Dagua is already PyTorch-based,
so the direction is attractive. But the sprint target graphs are <=500 nodes.
The immediate bottleneck is not raw speed; it is choosing the right objective
and avoiding bad local minima. GPU acceleration matters if Dagua wants to run
multi-start ensembles or scale the same lane to 20K+ nodes.

Recommendation: do not add a hard cuGraph dependency for this sprint. Instead,
make the force/stress lane batchable and torch-native. A K-candidate ensemble
can run multiple initial states on GPU in one tensor if the loss ops are shaped
for `[K, N, 2]`, or it can run serially on CPU for now.

Expected impact on h2h score: low directly, high as enabler for ensembles and
large graphs. Effort: medium to high.

### 9. Differentiable Sinkhorn for assignment/order

Gumbel-Sinkhorn (Mena et al., 2018), differentiable sorting/ranking via optimal
transport (Cuturi et al., 2019), and SoftSort (Prillo and Eisenschlos, 2020)
provide continuous relaxations of permutations. Dagua's layered crossing
reduction currently uses discrete barycenter, median sweep, transpose, and
Brandes-Kopf x refinement after the gradient core. Those are strong classical
methods, but they are not differentiable and only activate in acyclic/layered
contexts.

Sinkhorn is relevant for `ragged_feature_pyramid` and possibly `transformer_layer`,
not for small-world. It could optimize an x-order distribution within each
layer against crossing/straightness losses, then snap to a permutation with
Hungarian/argsort. That avoids dummy-node proliferation and lets Dagua's
gradient reason about order before committing.

Expected impact: medium for layered/pyramid; low for planar/small_world. Effort:
medium-high; risk: soft permutations can be numerically mushy and slow on wide
layers.

### 10. Newton-Raphson / LBFGS / CG refinement after Adam

Kamada-Kawai uses Newton-style local optimization; stress majorization solves a
linearized system per iteration; Dagua's KK pipeline already uses LBFGS. The
question is whether native Adam stalls above a better local minimum on hard
cases. For N<=500, a short second-order or quasi-Newton polish is cheap enough
to test.

The best use is not a dense Hessian over all Dagua losses. Use a restricted
smooth objective: stress/edge-length variance/straightness/repulsion, maybe
excluding non-smooth crossing and overlap terms. Run 10-30 LBFGS steps from
the final native or force-lane position. Then re-run hard overlap projection
and candidate scoring.

Expected impact: medium for regular/small_world if Adam underconverges; low if
bad initialization is the root cause. Effort: low-medium because optimizer
infrastructure exists.

### 11. Multi-start ensemble and metric-based candidate selection

This is the cheapest big win. The composite metric is known. Competitors win on
different structures. Dagua already has many pipelines. Rather than betting on
one new default, run a small portfolio for graphs <=500:

- Native lane, preserving current DAG wins.
- Pivot-MDS -> Stress-SGD/t-FDP.
- MaxEnt-Stress short.
- SFDP or FA2.
- Planar/Tutte seed if planar test passes.

Score candidates with the same metric bundle or a fast proxy. Keep the best.
This is not literature-novel, but it is strongly supported by the reality of
graph drawing: local minima and aesthetic tradeoffs are severe. SmartGD's
motivation also reinforces the point that multi-aesthetic optimization has no
single universal method.

Expected impact: high across all target losses. Runtime: Kx, but for N<=500
and K<=5 it is acceptable, and it can be gated to uncertain/tail-risk graph
families. Effort: low-medium.

### 12. Conformal/circle-packing seeds for planar lattices

Circle packing and discrete conformal mapping are classical but still active.
They give canonical geometry for planar graphs, often revealing faces and
symmetries better than generic spring forces. For Sierpinski and hexagonal
lattices, preserving holes and local angular structure matters more than
minimizing raw stress.

Practical approach: if a graph is planar and biconnected/triangulatable, create
a triangulation, compute a circle-packing or Tutte-like seed, then remove
auxiliary vertices. Full robust circle packing is more work than Tutte, so I
would prototype Tutte first. Circle packing becomes attractive if Tutte
over-compresses interiors or fails on non-3-connected cases.

Expected impact: medium-high for Sierpinski/hex/planar lattice. Effort: high
unless using a library.

### 13. Edge-bundling-aware drawing

The named "FFTBundle 2023" appears to line up more closely with older FFTEB
(FFT edge bundling, 2017) and newer web/pixel-based edge bundling work such as
PBEB (Wu et al., 2023). Edge bundling reduces visual clutter, but it does not
usually improve straight-line layout metrics; it may lower perceived clutter
while leaving node positions unchanged. Dagua's metric bundle currently rewards
edge straightness and crossing rate, so bundled curved edges could be penalized
unless metrics are route-aware.

Recommendation: skip as a layout fix for this sprint. Add later as a renderer
option or as route-aware metrics work. The exception is `parallel_cycles_4x5`,
where edge routing/bundling might improve human readability, but it is not the
top target here.

Expected impact on current composite: low or negative. Effort: medium.

### 14. FORBID overlap removal

FORBID (Giovannangeli et al., 2022) models overlap removal as joint stress and
scaling optimized by SGD. Dagua already has overlap losses and projections, but
FORBID's selling point is preserving the initial topology while removing
overlaps. That matters for labels and node sizes. Dagua's composite has a
binary 10-point overlap score, so a topology-preserving overlap postprocess can
protect wins while eliminating catastrophic overlap failures.

Integration sketch: add a postprocess candidate after each layout candidate:
current overlap projection vs FORBID-style stress+scale SGD. Pick the result
with better overlap and lower displacement/stress. This is especially useful
when a good planar or force seed has a few label overlaps.

Expected impact: low-medium on current target losses, high as guardrail.
Effort: medium.

### 15. GUMAP / NNP-NET / newer DR graph-layout variants

Dagua already has UMAP and tsNET. 2025 work such as GUMAP variants and NNP-NET
suggests that graph-specific UMAP/t-SNE acceleration remains active. For the
current <=500-node tail, the main value is not acceleration; it is another
candidate family with strong neighborhood preservation. GUMAP claims faster
tsNET-like results and quality gains on stress/crossing/shape metrics, but it
is too new to treat as high-confidence without reproducing.

Recommendation: include Dagua's existing UMAP/tsNET in a research ensemble only
if cheap. Do not build NNP-NET until large-graph tsNET runtime becomes a real
blocker.

Expected impact: medium for small_world, low for planar. Effort: low for using
existing UMAP/tsNET; high for new neural acceleration.

### 16. Resistance-distance stress / Omega

Recent 2025 work on resistance-distance stress challenges shortest-path stress
as the only distance model. This is plausible for regular/lattice graphs:
shortest-path distances can make many nodes equally distant and create diamond
or square artifacts, while resistance distances encode global connectivity and
can be smoother. The claimed linear-time/sampled SGD approach is attractive but
new and not yet obviously production-proven.

Recommendation: high-risk exploratory bet. Implement as an offline notebook or
opt-in pipeline after the simpler Pivot-MDS/Stress-SGD and planar lanes have
been tested.

Expected impact: unknown; maybe medium for regular/lattice. Effort: medium.

## Top-3 High-Confidence Recommendations

### 1. Topology-dispatched force/stress lane for flat graphs

Target: `small_world_100`, `small_world_500`, `regular_3_30`, maybe
`parallel_cycles_4x5`.

Rough integration:

1. Extend classification with `flat_undirected_like`, `small_world_like`, and
   `regular_like` tags. Use weak signals first: not directed acyclic, layer
   count <= 2 or bad depth/y signal, connected, sparse/moderate edge ratio,
   degree CV low for regular-like.
2. Build `layout_force_stress_pipeline`: Pivot-MDS seed -> Stress-SGD or
   MaxEnt-Stress/t-FDP refiner -> overlap projection -> aspect fit.
3. In default dispatch, run native and force/stress candidates for these tags
   until h2h proves direct dispatch safe.
4. Protect current wins by never using this lane for DAG tags unless explicitly
   selected.

Why first: most code exists. The target losses explicitly say "no hierarchy;
needs force-directed." This is the most direct answer to the benchmark.

### 2. Planar embedding seed lane for planar/lattice graphs

Target: `planar_60`, `hexagonal_lattice_42`, Sierpinski, planar lattice family,
possibly `ragged_feature_pyramid`.

Rough integration:

1. Add real planarity testing/embedding behind `graph_classify`, separate from
   the current sparsity hint.
2. For planar connected graphs under a size threshold, compute a planar
   embedding and outer face. Seed with Tutte/canonical/grid coordinates.
3. Add a crossing-preservation guard during polish: reject refinement steps
   that introduce crossings or apply a strong crossing barrier only after the
   planar seed.
4. Use lattice-specific axial/grid coordinate recovery when degree/face pattern
   indicates hex/triangular/square lattice.

Why second: higher effort, but it directly addresses the largest planar loss
and avoids trying to approximate combinatorial embedding with generic forces.

### 3. Small candidate ensemble with metric selection

Target: all tail losses, with K<=5 for N<=500.

Rough integration:

1. Add an internal candidate runner gated by graph size and risk tags.
2. Candidates: current native, force/stress lane, MaxEnt short, SFDP/FA2, and
   planar seed lane when eligible.
3. Score with the existing composite metric or a fast proxy mirroring the
   metric weights: dag consistency, length CV, overlap, crossing, angular
   resolution, depth Spearman if directed.
4. Cache candidate scores in verbose diagnostics to explain dispatch choices.

Why third: graph drawing has local minima and incompatible aesthetics. Ensemble
selection can harvest existing pipelines without prematurely making one new
algorithm the default. The runtime cost is acceptable for the benchmark scale.

## Top-3 High-Risk, High-Reward Bets

### 1. CoRe-GD/DeepGD-inspired learned initializer

Build a PyTorch initializer that consumes structural features and emits a 2D
seed. Start without training by using GraphSAGE-like message passing over
handcrafted node features and train later on Dagua/competitor cached layouts.
Reward: a single learned prior could improve small-world, regular, and messy
general graphs. Risk: training, determinism, model maintenance, and regression
debuggability.

### 2. Differentiable Sinkhorn x-ordering

Replace or augment median/transpose for selected layered graphs with a soft
permutation objective. Reward: Dagua can optimize ordering jointly with its
losses and potentially fix `ragged_feature_pyramid` without more dummy nodes.
Risk: soft ordering may be slow, hard to tune, and inferior to classical
discrete sweeps on small layers.

### 3. Resistance-distance / conformal planar metric lane

For regular and planar graphs, use resistance distances or conformal/circle
packing geometry instead of shortest-path stress. Reward: preserves lattice
symmetry and global shape better than force-directed methods. Risk: newer or
more specialized math, library complexity, and uncertain effect on Dagua's
specific composite.

## Things to Skip for Now

- **Full DeepGD/SmartGD as a default layout replacement.** Good research, wrong
  immediate integration cost. Use as initializer/candidate research only.
- **Edge bundling as a fix for current losses.** Bundling is a renderer/clutter
  technique. Dagua's current scoring is straight-line/routed-layout oriented,
  so bundling could reduce perceived clutter while not improving the metric.
- **Hard cuGraph dependency.** GPU FA2 is useful, but the target graphs are
  small. Torch-native batching and candidate selection will pay off sooner.
- **Another generic force algorithm without dispatch.** Dagua already has many
  opt-in force pipelines. The missing layer is topology-aware selection and
  integration with native scoring.
- **Full OGDF clone.** Orthogonal/TSM ideas are valuable, but wholesale
  replacement is too large. Start with planar embedding and grid/orthogonal
  seeds for low-degree cases.

## Risk and Regression Analysis

The protected wins are mostly hierarchical/DAG-like: deep org charts, random
DAGs, bipartite layered graphs, label-heavy fanout, and weighted karate. These
will regress if a generic force lane becomes the default for all graphs. The
main architectural rule should be: **directed acyclic/layered graphs stay on
native unless the candidate scorer proves otherwise.** The force/stress lane is
for flat/no-hierarchy graphs.

Planar embedding also has risk. A planar seed may score poorly on
`dag_consistency` or depth Spearman if applied to directed graphs with
meaningful edge direction. Therefore, planar dispatch should distinguish
undirected/weakly directed planar graphs from planar DAGs. For planar DAGs,
embedding can be an x-seed while y remains layer-constrained.

Multi-start ensemble risk is runtime and score overfitting. Use it only under
N<=500 or under diagnostic/benchmark mode until enough h2h evidence exists.
Candidate scoring must use the same metric semantics as evaluation, or a proxy
validated against it, to avoid choosing layouts that look better to the proxy
but worse in h2h.

## Implementation Order

1. **Measure existing opt-in pipelines on the six named target losses.** Run
   native, `pivot_mds`, `stress_sgd`, `maxent_stress`, `sfdp`, `fa2`, `umap`,
   and `sgd2_multi` with a fixed seed. This tells whether dispatch alone can
   close the gap.
2. **Build force/stress candidate lane using existing ops.** Pivot-MDS init and
   Stress-SGD/MaxEnt refinement should be possible with minimal new code.
3. **Add small ensemble scorer for N<=500 target-risk graphs.** Keep native as
   candidate to protect DAG strengths.
4. **Prototype real planarity test + Tutte/canonical seed.** Validate on
   `planar_60`, hexagonal, Sierpinski, and regular planar cases.
5. **Only after those land, try t-FDP bounded forces and FORBID overlap.**
   These are good refinements but less urgent than dispatch and planar
   topology.
6. **Longer term: learned initializers and Sinkhorn ordering.** These are
   promising but should not block the high-confidence classical improvements.

## References

- Wang, Yen, Hu, Shen. "DeepGD: A Deep Learning Framework for Graph Drawing
  Using GNN." arXiv/IEEE TVCG, 2021/2023.
  https://arxiv.org/abs/2106.15347
- Wang, Yen, Hu, Shen. "SmartGD: A GAN-Based Graph Drawing Framework for
  Diverse Aesthetic Goals." arXiv/IEEE TVCG, 2022/2024.
  https://arxiv.org/abs/2206.06434
- Grotschla, Mathys, Veres, Wattenhofer. "CoRe-GD: A Hierarchical Framework
  for Scalable Graph Visualization with GNNs." ICLR, 2024.
  https://arxiv.org/abs/2402.06706
- Hamilton, Ying, Leskovec. "Inductive Representation Learning on Large
  Graphs." NeurIPS, 2017. https://papers.nips.cc/paper/6703-inductive-representation-learning-on-large-graphs
- Brandes, Pich. "Eigensolver Methods for Progressive Multidimensional Scaling
  of Large Data." GD, 2007.
  https://www.research-collection.ethz.ch/handle/20.500.11850/667207
- Gansner, Hu, North. "A Maxent-Stress Model for Graph Layout." PacificVis,
  2012/2013. https://irc.cs.sdu.edu.cn/vis/course_M/papers/GraphLayout.pdf
- Dwyer, Koren, Marriott. "Constrained Graph Layout by Stress Majorization and
  Gradient Projection." Discrete Mathematics, 2009.
  https://doi.org/10.1016/j.disc.2007.12.103
- Zheng, Pawar, Goodman. "Graph Drawing by Stochastic Gradient Descent." IEEE
  TVCG, 2019. https://arxiv.org/abs/1710.04626
- Zhong, Xue, Zhang, Zhang, Ban, Deussen, Wang. "Force-Directed Graph Layouts
  Revisited: A New Force Based on the T-Distribution." arXiv/TVCG, 2023.
  https://arxiv.org/abs/2303.03964
- Zhu, Chen, Hu, Hou, Liu, Zhang. "DRGraph: An Efficient Graph Layout Algorithm
  for Large-scale Graphs by Dimensionality Reduction." IEEE TVCG, 2020/2021.
  https://arxiv.org/abs/2008.07799
- Giovannangeli, Lalanne, Giot, Bourqui. "FORBID: Fast Overlap Removal by
  Stochastic GradIent Descent for Graph Drawing." GD, 2022.
  https://arxiv.org/abs/2208.10334
- Balci, Dogrusoz, et al. "fCoSE: A Fast Compound Graph Layout Algorithm with
  Constraint Support." IEEE TVCG, 2021/2022.
  https://doi.org/10.1109/TVCG.2021.3095303
- Mena, Belanger, Linderman, Snoek. "Learning Latent Permutations with
  Gumbel-Sinkhorn Networks." ICLR, 2018. https://arxiv.org/abs/1802.08665
- Cuturi, Teboul, Vert. "Differentiable Ranking and Sorting using Optimal
  Transport." NeurIPS, 2019.
  https://papers.nips.cc/paper/8910-differentiable-ranking-and-sorting-using-optimal-transport
- Prillo, Eisenschlos. "SoftSort: A Continuous Relaxation for the argsort
  Operator." ICML, 2020. https://proceedings.mlr.press/v119/prillo20a.html
- Eppstein, Goodrich, Illickan. "Drawing Planar Graphs and 1-Planar Graphs
  Using Cubic Bezier Curves with Bounded Curvature." GD, 2024.
  https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2024.39
- Hartskeerl, Mchedlidze, van Wageningen, Vangorp, Telea. "NNP-NET:
  Accelerating t-SNE Graph Drawing for Very Large Graphs by Neural Networks."
  GD, 2025. https://drops.dagstuhl.de/entities/document/10.4230/LIPIcs.GD.2025.22
- RAPIDS/cuGraph and Graphistry GPU ForceAtlas2 documentation.
  https://hub.graphistry.com/docs/graph-algorithms/cugraphex/
