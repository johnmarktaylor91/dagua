# Round 21 adversarial diff: fmmm (`classic_fmmm` vs `ogdf_fmmm`)

Diagnosis-only pass. No source files were edited. Current mega-run verdict is already
`strong_equivalent` for all three FMMM variants, but this report catalogs residual and
structural divergences that can still explain sub-percent-to-moderate Procrustes RMSD.

## 1. Files read

Dagua implementation and wiring:

- `dagua/layout/ops/fmmm.py`: primary ops implementation, including force formulas, hierarchy
  construction, prolongation, refinement, and final normalization (`dagua/layout/ops/fmmm.py:45-76`,
  `dagua/layout/ops/fmmm.py:409-653`, `dagua/layout/ops/fmmm.py:741-954`,
  `dagua/layout/ops/fmmm.py:1202-1846`).
- `dagua/layout/ops/pipelines/fmmm.py`: public pipeline construction and `layout_fmmm_pipeline`
  wrapper (`dagua/layout/ops/pipelines/fmmm.py:28-76`,
  `dagua/layout/ops/pipelines/fmmm.py:79-158`).
- `dagua/layout/ops/pipelines/fr.py`: coarsest-level initializer used by FMMM
  (`dagua/layout/ops/pipelines/fr.py:127-179`, `dagua/layout/ops/pipelines/fr.py:182-262`).
- `dagua/layout/ops/init.py`: FR random initialization backend and dtype behavior
  (`dagua/layout/ops/init.py:736-859`).
- `dagua/layout/ops/force.py`: FR force field used by coarsest initialization
  (`dagua/layout/ops/force.py:837-912`).
- `dagua/layout/ops/anneal.py`: FR linear cooling and FMMM-adjacent exponential cooling conventions
  (`dagua/layout/ops/anneal.py:357-422`, `dagua/layout/ops/anneal.py:426-492`).
- `dagua/layout/ops/postprocess.py`: FR finalization and shared normalization semantics
  (`dagua/layout/ops/postprocess.py:323-405`, `dagua/layout/ops/postprocess.py:670-738`).
- `dagua/layout/ops/graph_utils.py`: `layout_extent` and normalization helper used by FMMM
  (`dagua/layout/ops/graph_utils.py:194-213`).
- `dagua/layout/_archive/classic/fmmm.py`: symlink target for `dagua/layout/classic/fmmm.py`,
  older monolithic implementation with the same broad simplifications
  (`dagua/layout/_archive/classic/fmmm.py:1-24`, `dagua/layout/_archive/classic/fmmm.py:73-216`).
- `dagua/eval/variants.py`: fmmm variant definitions and stochastic flags
  (`dagua/eval/variants.py:1001-1033`, `dagua/eval/variants.py:1838-1857`).
- `dagua/eval/competitors/classic_competitor.py`: `classic_fmmm` adapter and small-graph selector
  (`dagua/eval/competitors/classic_competitor.py:229-233`,
  `dagua/eval/competitors/classic_competitor.py:1469-1564`).
- `dagua/eval/competitors/ogdf_competitor.py`: `ogdf_fmmm` subprocess adapter and seed discard
  (`dagua/eval/competitors/ogdf_competitor.py:105-171`,
  `dagua/eval/competitors/ogdf_competitor.py:179-217`,
  `dagua/eval/competitors/ogdf_competitor.py:255-261`).
- `scripts/ogdf_runner.cpp`: compiled C++ runner used by the OGDF adapter
  (`scripts/ogdf_runner.cpp:145-157`, `scripts/ogdf_runner.cpp:219-238`).

OGDF reference implementation:

- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/FMMMLayout.h`: option
  documentation and defaults (`FMMMLayout.h:71-235`).
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/fmmm/FMMMOptions.h`: enum
  definitions for all option families (`FMMMOptions.h:38-148`).
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/FMMMLayout.cpp`: top-level flow,
  preprocessing, options, forces, iterations, postprocessing, component packing, integer rounding
  (`FMMMLayout.cpp:69-124`, `FMMMLayout.cpp:136-184`, `FMMMLayout.cpp:186-250`,
  `FMMMLayout.cpp:252-311`, `FMMMLayout.cpp:375-458`, `FMMMLayout.cpp:550-620`,
  `FMMMLayout.cpp:710-930`, `FMMMLayout.cpp:933-1340`).
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/fmmm/Multilevel.cpp`: solar-system
  coarsening and prolongation (`Multilevel.cpp:55-88`, `Multilevel.cpp:90-124`,
  `Multilevel.cpp:126-223`, `Multilevel.cpp:225-403`, `Multilevel.cpp:405-675`).
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/fmmm/Multilevel.h`:
  multilevel API and placement helper intent (`Multilevel.h:51-177`).
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/fmmm/Set.cpp`: OGDF random node-set
  selection for galaxy suns (`Set.cpp:51-79`, `Set.cpp:96-147`).
- `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/fmmm/Set.h`: node-set contract
  and lower/higher star mass APIs (`Set.h:44-121`).
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/fmmm/FruchtermanReingold.cpp`:
  exact/grid repulsive force code used by OGDF FMMM fallback paths
  (`FruchtermanReingold.cpp:51-77`, `FruchtermanReingold.cpp:79-162`).
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/fmmm/NewMultipoleMethod.cpp`:
  NMM repulsion setup and exact fallback cutoff (`NewMultipoleMethod.cpp:121-192`,
  `NewMultipoleMethod.cpp:320-372`, `NewMultipoleMethod.cpp:374-482`).
- `/home/jtaylor/projects/_references/ogdf/src/ogdf/energybased/fmmm/numexcept.cpp` and
  `/home/jtaylor/projects/_references/ogdf/include/ogdf/energybased/fmmm/numexcept.h`: repulsive
  scalar and numeric exception behavior (`numexcept.cpp:120-181`, `numexcept.h:77-81`).

Existing analysis:

- `eval_output/fidelity_report/report.md`: fmmm verdicts are strong-equivalent with median RMSD
  0.141, 0.130, and 0.125 (`eval_output/fidelity_report/report.md:31-33`).
- `.project-context/research/sprint_algo_fidelity/algo_fidelity_SUMMARY.md`: sprint note flags the
  OGDF seed-discard bug but says current OGDF-targeted families including fmmm are already
  strong-equivalent (`algo_fidelity_SUMMARY.md:194-197`).

## 2. Overall pipeline structure

Dagua `classic_fmmm` in evaluation is a Python competitor wrapper over
`layout_fmmm_pipeline`: the registry points `classic_fmmm` to
`dagua.layout.ops.pipelines.fmmm.layout_fmmm_pipeline` with default `steps=200`
(`dagua/eval/competitors/classic_competitor.py:229-233`). The public pipeline is six ops:
initialize hierarchy, initialize coarsest, refine coarsest, uncoarsen loop, single-level fallback,
finalize (`dagua/layout/ops/pipelines/fmmm.py:57-76`). The callable wrapper validates inputs,
special-cases `N=0` and `N=1`, builds a `LayoutProblem`, runs the CPU execution plan, and returns
`float32` positions (`dagua/layout/ops/pipelines/fmmm.py:122-158`).

The evaluation wrapper adds a non-reference selector on small graphs. For graphs with node sizes,
`N <= 2000`, and `E <= 5000`, it runs three candidates `(100, "fr")`, `(100, "ogdf_new")`,
and `(200, "ogdf_new")`, scores them with dagua metrics, and returns the best score
(`dagua/eval/competitors/classic_competitor.py:1511-1547`). For larger graphs, it runs exactly
`steps=200`, `force_model="ogdf_new"` (`dagua/eval/competitors/classic_competitor.py:1548-1554`).
This selector is a deliberate evaluation-layer divergence from OGDF FMMM, whose runner constructs
one `ogdf::FMMMLayout` and calls it once for `"fmmm"` (`scripts/ogdf_runner.cpp:154-157`).

OGDF's top-level flow is materially richer. `FMMMLayout::call` imports node/edge attributes,
initializes individual ideal edge lengths, reduces the graph to a simple loop-free copy,
runs divide-et-impera over connected components, adjusts final positions, and exports coordinates
(`FMMMLayout.cpp:90-116`). The divide-et-impera step decomposes into connected components, lays
each out, then packs component drawings (`FMMMLayout.cpp:136-154`). Each connected subgraph gets
a true OGDF multilevel representation (`FMMMLayout.cpp:157-184`): create all levels, initialize
the coarsest level with OGDF's own random/grid placement, refine each level, prolong lower levels,
and postprocess only at level 0 (`FMMMLayout.cpp:173-183`, `FMMMLayout.cpp:223-250`).

Dagua's multilevel flow is recognizably inspired by OGDF but compressed. It builds one hierarchy
for the full graph (`dagua/layout/ops/fmmm.py:1517-1524`), initializes the coarsest graph by
calling dagua FR (`dagua/layout/ops/fmmm.py:1588-1596`), optionally refines the coarsest graph
(`dagua/layout/ops/fmmm.py:1637-1660`), prolongs/refines back to the original graph
(`dagua/layout/ops/fmmm.py:1706-1732`), and normalizes the final drawing
(`dagua/layout/ops/fmmm.py:1838-1845`). There is no connected-component layout/packing phase,
no OGDF integer-position adjustment, and no OGDF postprocessing/fine-tuning loop.

## 3. Energy / loss / objective

Neither side is implemented as a single explicit energy minimization; both are force-directed
iterations. The closest "objective" terms are attractive edge forces, repulsive pair forces, and
step limiting.

Attraction:

- OGDF computes `vector_v_minus_u = pos[v] - pos[u]`, distance `d`, scalar
  `f_attr_scalar(d, E[e].get_length()) / d`, and applies `+f_u` to the source and `-f_u` to the
  target (`FMMMLayout.cpp:1091-1107`). With default `ForceModel::New`, the scalar is
  `log2(d / L) * d^2 / L^3` (`FMMMLayout.cpp:1110-1140`). Because OGDF then divides by `d`,
  the vector coefficient on `(v-u)` is `log2(d/L) * d / L^3`.
- Dagua's `_attractive_force_scale` for `"ogdf_new"` returns
  `log2(distances / desired_lengths) * (distances / desired_lengths.pow(3))`
  (`dagua/layout/ops/fmmm.py:45-76`). `_attractive_force_with_lengths` multiplies that scalar
  by `delta = positions[dst] - positions[src]` and accumulates source `+`, destination `-`
  (`dagua/layout/ops/fmmm.py:938-954`). This matches the OGDF vector coefficient for nonzero
  distances, modulo dtype, distance floors, and edge-length construction.
- Dagua's fallback `"fr"` force model returns `distances / desired_lengths.pow(3)`
  (`dagua/layout/ops/fmmm.py:72-76`). OGDF's `ForceModel::FruchtermanReingold` scalar is
  `d*d/L^3`, divided by `d`, hence the same vector coefficient `d/L^3`
  (`FMMMLayout.cpp:1113-1116`). The issue is not formula mismatch; it is that the wrapper may
  select this non-default model for small graphs (`classic_competitor.py:1520-1531`), while
  `ogdf_fmmm` uses default `ForceModel::New` (`FMMMLayout.cpp:283-285`).

Repulsion:

- OGDF's exact repulsion evaluates `numexcept::f_rep_u_on_v`, which returns
  `(pos_v - pos_u) * (1/d) / d = (pos_v - pos_u) / d^2` (`numexcept.cpp:169-181`,
  `numexcept.h:77-81`). The exact method adds that force to `v` and subtracts it from `u`
  (`FruchtermanReingold.cpp:68-75`).
- Dagua's exact repulsion forms `delta = positions.unsqueeze(1) - positions.unsqueeze(0)`,
  clamps distances, sets factor `1/(dist*dist)`, zeroes the diagonal, and sums
  (`dagua/layout/ops/fmmm.py:792-815`). For node `i`, the term against `j` is
  `(pos_i - pos_j)/d^2`, which is the same direction as OGDF's force on `i` due to `j`.
- OGDF default repulsion method is NMM, not Barnes-Hut. Defaults set
  `repulsiveForcesCalculation(NMM)`, `nmParticlesInLeaves(25)`, and `nmPrecision(4)`
  (`FMMMLayout.cpp:283-310`). NMM uses exact fallback for `N < 175`; otherwise it builds a
  reduced quadtree and evaluates direct, multipole, and local expansion terms
  (`NewMultipoleMethod.cpp:121-192`, `NewMultipoleMethod.cpp:139-168`).
- Dagua uses exact repulsion for `N <= 500` and a simple Barnes-Hut center-of-mass quadtree above
  that (`dagua/layout/ops/fmmm.py:1254-1263`, `dagua/layout/ops/fmmm.py:741-789`,
  `dagua/layout/ops/fmmm.py:818-854`). This can be close distributionally but is not the same
  approximation, threshold, tree construction, or summation order as OGDF's NMM.

Force scaling/objective composition:

- OGDF combines attraction and repulsion as
  `f = springStrength * F_attr + repForcesStrength * F_rep`, multiplies both coordinates by
  `average_ideal_edgelength^2`, then limits by `min(norm_f * cool_factor * forceScalingFactor,
  max_radius(iter)) / norm_f` (`FMMMLayout.cpp:1168-1204`). Defaults are spring strength 1,
  repulsion strength 1, `forceScalingFactor=0.05`, and `coolTemperature=false`
  (`FMMMLayout.cpp:283-294`).
- Dagua sums `repulsive + attractive` directly, normalizes by its norm, and caps displacement by
  a temperature initialized to FR ideal length and exponentially decayed by 0.99
  (`dagua/layout/ops/fmmm.py:1139-1166`, `dagua/layout/ops/fmmm.py:1179-1199`,
  `dagua/layout/ops/fmmm.py:1250-1272`). There is no `average_ideal_edgelength^2` multiplier,
  no `forceScalingFactor=0.05`, no `max_radius(iter)`, and no OGDF oscillation damping.

Postprocessing objective:

- OGDF runs 10 extra force steps, rescales to ideal average edge length, runs
  `fineTuningIterations()` more steps, and rescales again (`FMMMLayout.cpp:231-250`). Defaults
  are `fineTuningIterations=20`, `fineTuneScalar=0.2`, dynamic post repulsion, and
  `postSpringStrength=2.0` (`FMMMLayout.cpp:296-303`).
- Dagua has no equivalent postprocessing force loop. It finalizes by centering and scaling to a
  dagua layout extent (`dagua/layout/ops/fmmm.py:1800-1846`,
  `dagua/layout/ops/graph_utils.py:194-213`).

## 4. Force / gradient computation

Dagua computes force vectors in PyTorch tensors, not autograd gradients. `FMMMForceStep` selects
exact or Barnes-Hut repulsion, computes attractive force with either uniform ideal length or
per-edge desired lengths, sums both, caps each node displacement, and writes `state.pos`
(`dagua/layout/ops/fmmm.py:1088-1166`). The exact path is dense `torch.cdist` plus tensor
summation (`dagua/layout/ops/fmmm.py:811-815`). The approximate path builds a recursive Python
quadtree with center-of-mass averaging (`dagua/layout/ops/fmmm.py:680-738`) and applies an
opening criterion `width / distance < theta` (`dagua/layout/ops/fmmm.py:773-779`).

OGDF force computation is procedural C++ over `NodeArray<DPoint>`. Every force iteration calls
`adjust_positions`, `calculate_attractive_forces`, `calculate_repulsive_forces`,
`add_attr_rep_forces`, `prevent_oscillations`, `move_nodes`, and box update
(`FMMMLayout.cpp:972-983`). That sequence matters: integer restriction happens before each force
calculation under default `allowedPositions=Integer` (`FMMMLayout.cpp:550-620`,
`FMMMLayout.cpp:972-980`). Dagua does no per-iteration integer floor.

OGDF also dampens oscillation using angle between the new and previous movement. The factor table
is indexed by `ceil(angle / (pi/6))`, and if `norm_old * factor / norm_new < 1`, the new force is
scaled down (`FMMMLayout.cpp:1283-1317`). Dagua has no last-movement memory in FMMM force steps.

## 5. Initialization

OGDF runner initialization has two layers. The helper binary seeds OGDF global RNG and C `srand`
with 42, then assigns initial GraphAttributes using `std::rand() % 1000 / 10.0`
(`scripts/ogdf_runner.cpp:219-228`). FMMM then imports those positions (`FMMMLayout.cpp:349-357`)
but default FMMM coarsest initial placement overwrites coarsest coordinates via
`initialPlacementForces(RandomRandIterNr)` (`FMMMLayout.cpp:292-294`, `FMMMLayout.cpp:1041-1057`).
That random placement sets OGDF seed to `randSeed()` and samples `randomNumber(0, 1e9) / 1e9`
inside the current computation box (`FMMMLayout.cpp:1029-1038`, `FMMMLayout.cpp:1054-1056`).
Default `randSeed` is 100 (`FMMMLayout.cpp:260-263`), not the runner's seed 42 unless the C++
caller changes the FMMM option.

Dagua initializes the coarsest graph by calling `layout_fr_pipeline`, not OGDF FMMM's
`create_initial_placement_random` or uniform grid (`dagua/layout/ops/fmmm.py:1588-1596`). The FR
pipeline uses `RandomUniformInit` with `rng_backend="numpy"` and `scale="none"`
(`dagua/layout/ops/pipelines/fr.py:153-164`), which calls
`np.random.RandomState(problem.seed).rand(N, dim)` and produces `float64` positions
(`dagua/layout/ops/init.py:854-859`). It then runs standard FR forces and finalizes by centering,
scaling to `sqrt(N)*50`, and casting to `float32` (`dagua/layout/ops/pipelines/fr.py:165-176`,
`dagua/layout/ops/postprocess.py:398-405`). This is a major structural mismatch despite the
strong-equivalent verdict.

Prolongation is closer but still diverges. OGDF's advanced placement adds same-solar-system
adjacency-informed candidate positions before lambda-list candidates (`Multilevel.cpp:444-461`,
`Multilevel.cpp:583-605`), computes placement sectors from adjacent higher-level positions
(`Multilevel.cpp:492-565`), and samples random angles/radii using OGDF `randomNumber` in the open
interval formula `(randomNumber(1, BILLION)+1)/(BILLION+2)` (`Multilevel.cpp:634-657`). Dagua
prolongation uses lambda lists and moon interpolation, but does not implement the advanced
same-system `calculate_position` candidate pass or placement sectors; fallback random placement
always uses `[0, 2*pi]` (`dagua/layout/ops/fmmm.py:1392-1458`). Dagua samples with Python
`random.Random(seed).random()` (`dagua/layout/ops/fmmm.py:1277-1310`), which is not OGDF's RNG.

## 6. Iteration / convergence

OGDF default stop criterion is `FixedIterationsOrThreshold`, with `fixedIterations=30` and
`threshold=0.01` (`FMMMLayout.cpp:283-291`). `running` stops when either the max iteration count is
exceeded or average force vector length drops below threshold (`FMMMLayout.cpp:186-197`,
`FMMMLayout.cpp:215-220`). Per-level max iterations are not a single global `steps / levels`.
Defaults use `LinearlyDecreasing`, `maxIterFactor=10`: coarsest level can get up to
`10 * fixedIterations`, finest gets `fixedIterations`, and small levels with `node_nr <= 500`
get at least 100 iterations (`FMMMLayout.cpp:274-280`, `FMMMLayout.cpp:933-970`). After the main
level-0 loop, OGDF adds 10 postprocessing iterations plus `fineTuningIterations` (default 20)
(`FMMMLayout.cpp:231-250`, `FMMMLayout.cpp:296-300`).

Dagua uses a fixed total budget parameter. `_InitializeFMMMState` stores
`fmmm_level_budget = max(10, steps // len(levels))` (`dagua/layout/ops/fmmm.py:1526-1532`).
`FMMMRefineLevel` then runs exactly that many iterations per refined level with no threshold
check (`dagua/layout/ops/fmmm.py:1269-1272`). Coarsest initialization separately runs FR for
`max(50, steps)` iterations (`dagua/layout/ops/fmmm.py:1588-1595`). The public variants set
`steps=10`, `100`, and `200` (`dagua/eval/variants.py:1001-1033`), but the raw OGDF reference
adapter exposes no equivalent iteration option (`dagua/eval/competitors/ogdf_competitor.py:138-144`,
`scripts/ogdf_runner.cpp:145-157`).

Cooling is also not aligned. OGDF default `coolTemperature=false`, so `cool_factor=1.0` during
normal force calculation; force scaling is governed by `forceScalingFactor()` and
`max_radius(iter)` (`FMMMLayout.cpp:1147-1204`). Dagua always applies exponential cooling by
0.99 to `state.ideal_length` after every FMMM step (`dagua/layout/ops/fmmm.py:1179-1199`,
`dagua/layout/ops/fmmm.py:1264-1271`).

## 7. Hyperparameter alignment table

| Parameter | Dagua default / behavior | OGDF default / behavior | Match? | Line refs |
| --- | --- | --- | --- | --- |
| Public default seed | `layout_fmmm_pipeline(seed=42)` | FMMM `randSeed(100)`; runner seeds OGDF/C RNG with 42 but FMMM random placement resets to 100 | N | `fmmm.py:83-87`; `FMMMLayout.cpp:260-263`; `FMMMLayout.cpp:1054-1056`; `ogdf_runner.cpp:219-228` |
| Small-graph wrapper behavior | Candidate selector over `(100, fr)`, `(100, ogdf_new)`, `(200, ogdf_new)` | Single FMMMLayout call | N | `classic_competitor.py:1511-1547`; `ogdf_runner.cpp:154-157` |
| Min graph size / coarsest target | `_COARSE_TARGET=50` | `minGraphSize(50)` | Y | `fmmm.py:30-32`; `FMMMLayout.cpp:274-276` |
| Galaxy choice | Highest star mass sampling | `NonUniformProbLowerMass` | N | `fmmm.py:218-259`, `fmmm.py:449-453`; `FMMMLayout.cpp:274-277`; `Multilevel.cpp:155-164` |
| Random tries | 20 | 20 | Y | `fmmm.py:34`; `FMMMLayout.cpp:274-278` |
| Edge linearity break | bad counter after `current > 0.8 previous`, break after >5 | same conceptual rule | Mostly Y | `fmmm.py:634-642`; `Multilevel.cpp:90-103` |
| Edge length measurement | Input lengths start at 1.0; no radius addition | BoundingCircle: `length*unitEdgeLength + radii` | N | `fmmm.py:615-626`; `FMMMLayout.cpp:375-389` |
| Unit edge length | implicit 1.0 / dagua extent | `LayoutStandards::defaultNodeSeparation()` documented/default 100 | N | `fmmm.py:280-347`; `FMMMLayout.h:87-88`; `FMMMLayout.cpp:252-257` |
| Parallel edges | Average lengths, sum weights | Average lengths, no public weights | Partial | `fmmm.py:311-347`; `FMMMLayout.cpp:460-539`; `Multilevel.cpp:339-403` |
| Self-loops | Dropped | Dropped in reduced copy | Y | `fmmm.py:321-325`; `FMMMLayout.cpp:409-429` |
| Connected components | One hierarchy over entire graph | Separate connected subgraphs plus packing | N | `fmmm.py:1517-1524`; `FMMMLayout.cpp:136-154`; `FMMMLayout.cpp:746-760` |
| Coarsest initial placement | Dagua FR pipeline | OGDF random/grid/keep placement; default random from `randSeed` | N | `fmmm.py:1588-1596`; `FMMMLayout.cpp:1004-1057` |
| FR/NMM exact cutoff | exact repulsion for `N <= 500`, BH above | exact fallback inside NMM for `N < 175`, NMM above | N | `fmmm.py:1260`; `NewMultipoleMethod.cpp:176-191` |
| Approx repulsion | Barnes-Hut COM, max depth 10, theta 1.0 | NMM multipole/local/direct expansions, precision 4, particles 25 | N | `fmmm.py:30-33`, `fmmm.py:741-854`; `NewMultipoleMethod.cpp:121-168`; `FMMMLayout.cpp:305-310` |
| Attractive force model | `"ogdf_new"` default; wrapper may select `"fr"` | `ForceModel::New` default | Partial | `fmmm.py:72-76`; `classic_competitor.py:1520-1531`; `FMMMLayout.cpp:283-285` |
| Spring strength | hardcoded 1 by omission; edge weights optional | `springStrength(1)` | Partial | `fmmm.py:945-953`; `FMMMLayout.cpp:283-286` |
| Repulsion strength | hardcoded 1 | `repForcesStrength(1)` | Y | `fmmm.py:1140-1165`; `FMMMLayout.cpp:283-287` |
| Force scale | displacement capped by temperature | multiply by avg edge length squared, `forceScalingFactor=0.05`, max radius | N | `fmmm.py:1162-1165`; `FMMMLayout.cpp:1183-1204` |
| Cooling | Always 0.99 exponential | `coolTemperature(false)` by default; `coolValue=0.99` unused in normal phase | N | `fmmm.py:33`, `fmmm.py:1179-1199`; `FMMMLayout.cpp:291-294`, `FMMMLayout.cpp:1147-1156` |
| Stop criterion | fixed per-level budget only | fixed-iterations-or-threshold | N | `fmmm.py:1269-1272`; `FMMMLayout.cpp:186-220` |
| Iterations | `steps // levels`, min 10; FR coarse `max(50, steps)` | level-dependent 30..300, small levels min 100, plus postprocessing | N | `fmmm.py:1530`, `fmmm.py:1588-1595`; `FMMMLayout.cpp:933-970`; `FMMMLayout.cpp:231-250` |
| Oscillation prevention | none | angle-based movement damping | N | `FMMMLayout.cpp:1283-1317` |
| Postprocessing | final normalize only | 10 force steps, resize, fine tune, resize | N | `fmmm.py:1800-1846`; `FMMMLayout.cpp:231-250` |
| Final position rounding | float32 normalized coordinates | default integer floor after bounding restriction | N | `fmmm.py:1841-1845`; `FMMMLayout.cpp:550-620` |
| Output dtype in adapter | `torch.float32` | C++ double output parsed into `torch.float32` | Partial | `fmmm.py:137-140`; `ogdf_competitor.py:162-171`; `ogdf_runner.cpp:232-238` |

## 8. Edge cases

Self-loops: both discard them before force/layout. Dagua skips `source == target` in
`_unique_edges_with_lengths` (`dagua/layout/ops/fmmm.py:318-325`). OGDF marks self-loop copy edges
as `nullptr` and only creates reduced edges for non-self-loops (`FMMMLayout.cpp:420-429`).

Multi-edges: both average ideal lengths for parallel simple-edge collapse, but dagua also sums
optional attraction weights (`dagua/layout/ops/fmmm.py:311-347`, `dagua/layout/ops/fmmm.py:522-563`).
OGDF averages lengths when deleting parallel edges in both initial simplification and multilevel
collapse (`FMMMLayout.cpp:460-539`, `Multilevel.cpp:339-403`). Since the OGDF runner payload has
only edge endpoints, not weights (`ogdf_competitor.py:138-144`), weighted dagua FMMM cannot align
with `ogdf_fmmm`.

Disconnected components: this is a major edge-case divergence. OGDF explicitly computes connected
components, lays each component independently, rotates rectangles, packs them with MAAR packing,
and exports translated component positions (`FMMMLayout.cpp:136-154`, `FMMMLayout.cpp:710-760`,
`FMMMLayout.cpp:816-930`). Dagua runs one hierarchy over the full edge tensor and therefore
disconnected components repel each other during the same force solve (`dagua/layout/ops/fmmm.py:1517-1524`,
`dagua/layout/ops/fmmm.py:1706-1732`). For disconnected graphs, strong-equivalent verdicts are
probably coming from Procrustes/metric tolerance, not implementation identity.

Weighted edges: dagua accepts `edge_weights` and applies them to attractive forces
(`dagua/layout/ops/fmmm.py:896-903`, `dagua/layout/ops/fmmm.py:947-954`), and the classic wrapper
passes no explicit weights in the selector (`classic_competitor.py:1525-1532`). OGDF FMMM accepts
an `EdgeArray<double> edgeLength`, but the runner only serializes endpoints and calls `layout.call`
with default edge lengths (`scripts/ogdf_runner.cpp:214-230`, `FMMMLayout.cpp:69-73`).

Empty and singleton graph: dagua returns an empty `[0,2]` tensor or a single zero immediately
(`dagua/layout/ops/pipelines/fmmm.py:136-140`). OGDF runner creates zero or one graph nodes; FMMM
only handles `G.numberOfNodes() == 1` by setting `(0,0)` and otherwise does nothing for zero-node
graphs (`FMMMLayout.cpp:98-123`). The Python OGDF adapter special-cases `num_nodes == 0` before
subprocess execution and returns zeros (`dagua/eval/competitors/ogdf_competitor.py:131-132`).

Degenerate coincident positions: OGDF has numeric exception logic that randomly perturbs equal
positions before repulsion (`numexcept.cpp:169-181`, `numexcept.cpp:48-111`). Dagua clamps
distances to `_MIN_DISTANCE=1e-3` and zeroes the diagonal (`dagua/layout/ops/fmmm.py:30`,
`dagua/layout/ops/fmmm.py:811-815`), so exact coincident non-identical nodes produce zero delta
and no random separating direction.

## 9. Numerical precision

Dagua uses mixed precision. FR initialization returns `float64` positions internally when using
NumPy random initialization (`dagua/layout/ops/init.py:854-859`), but FR finalization casts to
`float32` (`dagua/layout/ops/postprocess.py:398-405`), and FMMM coarsest positions are explicitly
cast to `float32` (`dagua/layout/ops/fmmm.py:1596`). FMMM hierarchy edge lengths and weights are
created as `torch.float32` (`dagua/layout/ops/fmmm.py:333-347`, `dagua/layout/ops/fmmm.py:556-563`).
Final output is also `float32` (`dagua/layout/ops/fmmm.py:1841-1845`).

OGDF computes geometry in C++ `double` `DPoint` and `double` edge lengths throughout the FMMM
implementation (`FMMMLayout.cpp:360-389`, `FMMMLayout.cpp:1083-1140`,
`FMMMLayout.cpp:1183-1204`). The runner prints text JSON without setting high precision
(`scripts/ogdf_runner.cpp:232-238`), and the Python adapter parses to `torch.float32`
(`dagua/eval/competitors/ogdf_competitor.py:157-171`). Thus reference-side numerical precision is
double during layout but lossy at output serialization and adapter ingestion.

Summation order also differs. Dagua exact repulsion uses vectorized tensor reductions in row order
(`dagua/layout/ops/fmmm.py:811-815`), while OGDF loops upper-triangle pairs in graph node order
and accumulates pairwise into two nodes (`FruchtermanReingold.cpp:62-75`). Dagua attraction uses
`index_add_`, whose accumulation order is tensor backend dependent (`dagua/layout/ops/fmmm.py:901-903`,
`dagua/layout/ops/fmmm.py:952-954`); OGDF iterates `G.edges` and mutates `NodeArray`s in edge
iteration order (`FMMMLayout.cpp:1091-1107`). For sub-percent residual RMSD, these order/dtype
differences are enough even when formulas match.

## 10. RNG semantics

Dagua's torch seed does not produce the same sequence as OGDF's RNG. In fact, FMMM mostly does not
use torch RNG. Hierarchy coarsening and prolongation use Python `random.Random(problem.seed)`
(`dagua/layout/ops/fmmm.py:632`, `dagua/layout/ops/fmmm.py:1714`); coarsest FR initialization uses
NumPy `RandomState(problem.seed)` (`dagua/layout/ops/pipelines/fr.py:153-164`,
`dagua/layout/ops/init.py:854-859`). Dagua's optional torch random helper exists elsewhere, but
not in this FMMM path (`dagua/layout/ops/init.py:90-109`).

OGDF uses its own global RNG through `setSeed` and `randomNumber`. The runner seeds OGDF global RNG
to 42 before creating initial GraphAttributes (`scripts/ogdf_runner.cpp:219-228`), but FMMM default
coarsest random placement resets to `randSeed()`, whose default is 100 (`FMMMLayout.cpp:260-263`,
`FMMMLayout.cpp:1054-1056`). Multilevel coarsening also calls `setSeed(rand_seed)` and the node
set calls `set_seed(rand_seed)` (`Multilevel.cpp:55-61`, `Multilevel.cpp:133-136`,
`Set.cpp:51-52`).

Additionally, the Python OGDF adapter ignores the competitor seed: `_OGDFBase.layout` explicitly
`del seed` because the helper binary exposes no seed parameter (`dagua/eval/competitors/ogdf_competitor.py:179-204`).
The sprint summary already flags this as a known seed-discard bug
(`algo_fidelity_SUMMARY.md:194-197`). Therefore `classic_fmmm(seed=42)` and `ogdf_fmmm(seed=42)`
are not seed-aligned in any strong sense; they are only benchmark-deterministic by separate RNGs.

## 11. Edge-case bugs

1. Likely sign/selection mismatch in galaxy choice: dagua always uses
   `get_random_node_with_highest_star_mass` (`dagua/layout/ops/fmmm.py:449-453`), while OGDF
   default is `NonUniformProbLowerMass` and calls `get_random_node_with_lowest_star_mass`
   (`FMMMLayout.cpp:274-277`, `Multilevel.cpp:155-164`, `Set.cpp:141-147`). This is not subtle;
   it changes the hierarchy.

2. Temperature state key is unused/misnamed. `FMMMForceStep` and `FMMMCoolStep` carry
   `temperature_key`, but force reads/writes `state.ideal_length`, not `state.extras[...]`
   (`dagua/layout/ops/fmmm.py:1106-1138`, `dagua/layout/ops/fmmm.py:1179-1199`). This is not a
   runtime bug for current behavior, but it obscures temperature semantics and makes OGDF cooling
   alignment harder.

3. Dagua single-level fallback uses uniform ideal length, not edge lengths. If no hierarchy is
   built, `_SingleLevelFallback` passes `edge_weights` but not `edge_lengths`
   (`dagua/layout/ops/fmmm.py:1779-1795`). OGDF always initializes individual ideal edge lengths
   before simplification (`FMMMLayout.cpp:103-110`, `FMMMLayout.cpp:375-389`). For small graphs,
   this loses per-edge node-radius/edge-length information entirely.

4. Dagua exact coincident-node repulsion can be zero. Because clamping happens after delta is zero,
   a non-diagonal coincident pair contributes zero vector (`dagua/layout/ops/fmmm.py:811-815`).
   OGDF perturbs coincident points randomly before computing repulsion (`numexcept.cpp:169-181`).
   This can trap symmetric isolated/duplicate starts differently.

5. Dagua final normalization erases OGDF's integer floor and ideal-edge resize. OGDF floors final
   positions under default integer mode (`FMMMLayout.cpp:605-619`) and rescales against actual vs
   ideal edge lengths in postprocessing (`FMMMLayout.cpp:1319-1340`). Dagua normalizes to its own
   extent (`dagua/layout/ops/fmmm.py:1841-1845`, `dagua/layout/ops/graph_utils.py:194-213`).
   This is not a crash bug, but it is a fidelity bug for exact reference matching.

6. The wrapper selector can hide raw pipeline regressions. `classic_fmmm` may return a 100-step FR
   force-model candidate because it scores better (`classic_competitor.py:1518-1547`), even though
   the named variant may be `classic_fmmm_steps200` in `variants.py` (`dagua/eval/variants.py:1024-1033`).
   This improves benchmark verdicts but makes diagnosis of raw `layout_fmmm_pipeline` vs OGDF less
   direct.

7. OGDF output precision is truncated by the runner. The C++ runner streams doubles with default
   iostream precision (`scripts/ogdf_runner.cpp:232-238`), then Python casts to `float32`
   (`dagua/eval/competitors/ogdf_competitor.py:165-171`). This may be intentional, but for
   sub-percent residual analysis it is a reference-adapter precision bug.

## 12. Ranked fix list

1. Align galaxy choice to OGDF default lower-star-mass sampling.
   Expected RMSD impact: high for graphs that actually coarsen. Dagua currently selects highest
   star mass (`dagua/layout/ops/fmmm.py:449-453`); OGDF default is lower mass
   (`FMMMLayout.cpp:274-277`, `Multilevel.cpp:159-164`). Proposed fix size: S, add a galaxy-choice
   config enum and default to `"lower"` for OGDF fidelity, preserving `"higher"` only as a legacy
   option.

2. Replace coarsest FR initialization with OGDF-style random placement inside computed box.
   Expected RMSD impact: high. Dagua calls FR (`dagua/layout/ops/fmmm.py:1588-1596`), while OGDF
   initializes a box and samples random positions (`FMMMLayout.cpp:985-1059`). Proposed fix size:
   M, because it requires adding OGDF boxlength semantics and likely an option to preserve current
   quality-oriented behavior.

3. Implement OGDF force scaling, fixed/threshold stop criterion, and oscillation damping.
   Expected RMSD impact: high. Dagua caps by exponentially cooled ideal length
   (`dagua/layout/ops/fmmm.py:1162-1165`, `dagua/layout/ops/fmmm.py:1179-1199`), while OGDF uses
   average edge length squared, `forceScalingFactor`, `max_radius`, threshold stopping, and
   movement damping (`FMMMLayout.cpp:186-220`, `FMMMLayout.cpp:1143-1204`,
   `FMMMLayout.cpp:1283-1317`). Proposed fix size: L.

4. Add OGDF postprocessing: 10 preliminary post iterations, ideal-edge resize, fine tuning, and
   second resize.
   Expected RMSD impact: medium-high, especially final edge-length scale and local untangling.
   Dagua has only final normalization (`dagua/layout/ops/fmmm.py:1800-1846`); OGDF postprocess is
   `FMMMLayout.cpp:231-250` plus resize formula at `FMMMLayout.cpp:1319-1340`. Proposed fix size:
   M/L.

5. Add disconnected-component solve and MAAR-like packing or at least component-separate layout
   with deterministic rectangle packing.
   Expected RMSD impact: high on disconnected graphs, low on connected graphs. OGDF separates and
   packs components (`FMMMLayout.cpp:136-154`, `FMMMLayout.cpp:746-760`); dagua solves all nodes
   together (`dagua/layout/ops/fmmm.py:1517-1524`). Proposed fix size: L.

6. Match individual ideal edge lengths including node radii and unit edge length.
   Expected RMSD impact: medium. OGDF default `BoundingCircle` adds endpoint radii to
   `edgeLength * unitEdgeLength` (`FMMMLayout.cpp:375-389`), while dagua starts from length 1
   (`dagua/layout/ops/fmmm.py:311-347`). Proposed fix size: M, because dagua has node sizes but
   not OGDF GraphAttributes units in the FMMM path.

7. Replace Barnes-Hut approximation with OGDF NMM threshold/precision behavior, or lower the exact
   cutoff to OGDF's `N < 175` and expose a reference mode.
   Expected RMSD impact: medium for larger graphs. Dagua's approximation is Barnes-Hut
   (`dagua/layout/ops/fmmm.py:741-854`); OGDF NMM uses multipole/local/direct expansion and exact
   fallback below 175 (`NewMultipoleMethod.cpp:121-192`). Proposed fix size: XL for true NMM, S
   for cutoff-only reference mode.

8. Make `ogdf_fmmm` seed-controllable and set FMMM `randSeed` in the runner.
   Expected RMSD impact: medium for reproducibility, potentially high for seed-sensitive graphs.
   Python adapter deletes seed (`ogdf_competitor.py:203`), runner hardcodes 42
   (`scripts/ogdf_runner.cpp:219-222`), and FMMM defaults to randSeed 100
   (`FMMMLayout.cpp:260-263`). Proposed fix size: M, requiring JSON schema and runner option
   plumbing.

9. Match OGDF integer-position adjustment or make finalization reference-mode configurable.
   Expected RMSD impact: low-to-medium after Procrustes normalization, but nonzero. OGDF floors
   positions (`FMMMLayout.cpp:605-619`); dagua emits normalized float32 (`dagua/layout/ops/fmmm.py:1841-1845`).
   Proposed fix size: S/M.

10. Increase reference adapter output precision.
    Expected RMSD impact: low, but important for residual sub-percent analysis. Runner uses default
    stream precision (`scripts/ogdf_runner.cpp:232-238`); adapter casts to `float32`
    (`ogdf_competitor.py:165-171`). Proposed fix size: S.

## 13. Recommended Round 22+ fix scope

Recommended one-round bundle: implement a `reference_mode` for `layout_fmmm_pipeline` and the
classic wrapper that targets the largest algorithmic gaps without attempting full NMM.

Top-K scope:

1. Add galaxy-choice config and default reference mode to OGDF's `NonUniformProbLowerMass`.
   This is small and likely high leverage (`dagua/layout/ops/fmmm.py:449-453` vs
   `Multilevel.cpp:159-164`).

2. Add OGDF-style coarsest random placement using `randSeed` semantics as closely as practical,
   but keep Python RNG behind an option until the runner seed story is fixed
   (`FMMMLayout.cpp:1029-1057`, `dagua/layout/ops/fmmm.py:1588-1596`).

3. Replace dagua's exponential FMMM cooling with an OGDF-compatible force scaling path:
   average ideal edge length squared, `forceScalingFactor=0.05`, and oscillation damping
   (`FMMMLayout.cpp:1147-1204`, `FMMMLayout.cpp:1283-1317`).

4. Add OGDF postprocessing resize/fine-tune loop after the finest level
   (`FMMMLayout.cpp:231-250`, `FMMMLayout.cpp:1319-1340`).

5. Fix seed plumbing in `ogdf_runner`/`ogdf_competitor` so the next adversarial sweep can separate
   algorithmic differences from seed differences (`ogdf_competitor.py:179-204`,
   `scripts/ogdf_runner.cpp:219-228`).

Do not include true NMM in the same follow-up round unless the goal is a larger implementation
sprint. Barnes-Hut vs NMM is real (`dagua/layout/ops/fmmm.py:741-854` vs
`NewMultipoleMethod.cpp:139-168`), but the first four levers above should explain larger residual
RMSD while keeping patch size reviewable.
