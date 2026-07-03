# r75 bucket: classical_mds + small tails (umap/gem/maxent/drl/neato) -- sonnet findings

## 1. Executive summary

- **classical_mds (30 combos): confirmed root cause for all 16 disconnected combos** = dagua
  runs ONE global MDS with a synthetic `max_distance+1.0` fill for cross-component pairs
  (`dagua/layout/ops/graph_utils.py:319-352`), while igraph decomposes into components, MDS's
  each separately, then merges via stochastic DLA (`merge_dla.c`). This is a structurally
  different embedding, not a rounding residual -- explains the largest gaps in the bucket
  (`disconnected_label_cycle_collage` battery_stress D=1.0 vs R=0.0).
- **The "naive DLA port HUNG" claim from r74 is TRUE but mischaracterized.** I reproduced it: a
  literal, unoptimized per-step Python port of `igraph_i_layout_merge_dla` is NOT an infinite
  loop -- it is a genuine 2D unbiased random walk with step size `startr/200` relative to a
  domain of radius `startr`, giving O((startr/step)^2) ~= tens of thousands of steps per
  placement. On the worst real target graph (`random_dag_200`, 202 components, mostly isolated
  nodes) a full run of all 201 DLA placements completed in **17.4s** in pure Python (not
  vectorized, not C). It never diverges; it was probably killed by an impatient timeout in the
  earlier attempt, not a true hang. Feasibility for a bounded-time port is CONFIRMED.
- **14 connected classical_mds combos**: 12/14 genuinely divergent (battery_stress D vs R gaps
  of 5-25%+), 2/14 (`org_chart_1_5_4_8`) already `equiv=true` and near-threshold. Root cause is
  the documented degenerate-eigenspace LAPACK basis-selection difference (`classical_mds.py:54-66`),
  NOT a new bug -- this matches the r73/r74 "classical_mds degenerate-eigenspace ~16" floor entry.
- **umap (7 combos)**: params ARE matched between variant and reference (verified all 5 variant
  defs); the r73 parallel-edge-sum fix is already live (`umap.py:123-163`). Dagua is a genuine
  native SGD port (does not call umap-learn at runtime). Residual is almost certainly
  negative-sampling/SGD RNG-stream divergence against umap-learn's numba `tau_rand_int` PRNG --
  HYPOTHESIS, not confirmed bit-for-bit.
- **gem (5 combos), drl (3 combos), neato (2 combos)**: all self-documented, already-registered
  stochastic algorithms (GEM: random impulse spring embedder; DrL: float32-rounded multilevel
  force-directed with density-grid randomness; neato: `drand48`-seeded, unported CG solver
  details). All are FLOOR candidates consistent with r74's classification ("gem 22 FP-chaos
  floor, 269/309 already bit-exact"); no comparison bug found. I did NOT run a fresh 1-ULP
  perturbation experiment for gem this round (time budget) -- flagged as HYPOTHESIS, cite r74's
  existing floor evidence.
- **maxent (3 combos)**: r74 Option-2 already confirmed OGDF `StressMinimization` does NOT
  component-split (`m_componentLayout(FALSE)`) and dagua's single-pass already mirrors it --
  "no fix needed" verdict stands. These 3 residual combos are the genuine (non-disconnected-packing)
  residue after that dead-end was closed.
- **Net actionable fix**: only classical_mds disconnected-graph DLA port is a genuine, scoped,
  low-risk fix opportunity (~16 combos). Everything else in this bucket is either floor
  (documented, self-consistent with r74's prior audits) or needs one more RNG-tracing experiment
  (umap) that I could not complete in the time budget.

---

## 2. Findings ranked by expected combo-count impact

### Finding 1 (CONFIRMED, ~16 combos): classical_mds disconnected-graph handling is architecturally wrong, not a rounding issue

**dagua side**: `dagua/layout/ops/pipelines/classical_mds.py:241-245` calls
`_shortest_path_distances` unconditionally on the WHOLE graph for both `igraph_fidelity` and
default variants. `dagua/layout/ops/graph_utils.py:319-352` (`shortest_path_distances`) fills
unreachable pairs with a single scalar `max_distance + 1.0` (line 349), then classical MDS
double-centers and eigendecomposes that matrix as if it were one connected metric space. This
produces a smoothly-collapsed single blob instead of separated, independently-scaled component
clusters.

**Reference side**: `_references/igraph/src/layout/mds.c:223-289` (`igraph_layout_mds`) checks
`igraph_is_connected` (line 224); if disconnected, it explicitly decomposes into weak components
(`igraph_subcomponent`, line 256), lays out EACH one independently via
`igraph_i_layout_mds_single` (line 264, the connected-graph kernel -- same one used for the
whole-graph case), then merges via `igraph_layout_merge_dla` (line 278,
`_references/igraph/src/layout/merge_dla.c:71-190`).

**Evidence this is the dominant driver of the largest gaps in the bucket** (target JSON
`battery_stress` legs):
```
disconnected_label_cycle_collage::classic_classical_mds_default   D=1.0    R=0.0    (7 nodes, 3 components: sizes 3,2,2)
parallel_cycles_4x5::classic_classical_mds_default                D=1.0    R=0.011  (20 nodes, 4 components size 5 each)
disconnected_encoder_residual::classic_classical_mds_default       D=0.35   R=0.025  (9 nodes, 2 components: sizes 5,4)
multi_component_80::classic_classical_mds_default                  D=0.561  R=0.084  (80 nodes, 7 comps: 40,20,10,5,3,1,1)
```
These are order-of-magnitude gaps, not floating-point residuals -- consistent with "wrong
algorithm class" not "wrong RNG stream."

**Confirmed against prior r74 attempts**: a deterministic TileToRows-packing port (commit
`91ccaab`) was tried and reverted (`f342617`) after scoring **0 improved / 12 regressed -- pure
harm** (`r74_close_all_gaps_STATE.md:226`). This validates the bucket brief's framing: the
merge step genuinely needs to be the STOCHASTIC DLA algorithm, not a deterministic substitute --
a deterministic grid pack changes which nodes end up near which other nodes in ways that hurt
stress/crossings on these specific graphs, it isn't merely "less faithful."

### Finding 1b (CONFIRMED via reproduction): DLA merge algorithm spec + hang root cause + bounded-time strategy

**Exact algorithm** (`merge_dla.c` + `merge_grid.c`, fully read):
1. For each of the `k` components, compute a bounding circle: center = midpoint of (min,max) per
   axis, radius = half the diagonal (`igraph_i_layout_sphere_2d`, `merge_dla.c:192-222`).
2. Component "packing radius" `r_i = size_i^0.75` (line 112) -- NOT the bounding-circle radius;
   this is a separate synthetic size used only for placement/packing, while `nr_i` (the true
   bounding radius) is used later purely as a rescale factor (line 166: `rr = r_i / nr_i`).
3. `area = sum(r_i^2)`; grid domain is `[-sqrt(5*area), sqrt(5*area)]` in both axes, quantized to
   a `200x200` occupancy grid (`merge_grid.c` -- each cell owned by exactly one component id).
4. Components are visited **largest-to-smallest** (`vector_order`, `merge_dla.c:32-47`, sorts by
   size descending). This ordering is DETERMINISTIC (not seed-dependent) and must be replicated
   exactly for correctness (affects grid occupancy history, hence placement outcome).
5. The largest component is placed at the origin (line 136). Each subsequent component runs
   `igraph_i_layout_merge_dla` (line 266-298): repeatedly (a) sample a random start point at
   polar radius `Uniform(0.5*startr, startr)` around the origin where `startr = maxx` (domain
   half-width), (b) if that start point already overlaps an occupied sphere, resample; (c)
   otherwise random-walk in tiny steps -- polar radius `Uniform(0, startr/100)` per step -- until
   either landing on an occupied cell (STOP, place there) or exceeding radius `killr = maxx + 5`
   from the origin (abandon this attempt, go back to (a) with a fresh random start).
6. `igraph_i_layout_merge_place_sphere` (`merge_grid.c:70-121`) marks the placed component's
   footprint into the grid using a 4-quadrant expanding scan from the placement cell, each
   quadrant independently checking a Euclidean distance bound `< r`.
7. Final coordinates: for each component, rescale local layout by `rr = r_i / nr_i` (or 1 if
   `nr_i==0`), shift by the component's bounding-circle center, then translate to the DLA
   placement `(x_i, y_i)` (`merge_dla.c:158-178`).

**RNG stream requirement for per-seed bit-exactness**: every `RNG_UNIF(l,h)` call
(`igraph_random.h:154`, dispatches to `igraph_rng_get_unif` on `igraph_rng_default()`) draws from
igraph's PCG32-based global RNG in a FIXED call order: per outer attempt, one `angle` + one
`length` draw for the start point, then for the inner walk loop one `angle` + one `length` draw
PER STEP until termination. The number of draws is data-dependent (walk-length is random), so
bit-exactness requires porting igraph's actual PCG32 generator (or its underlying uniform
transform) call-for-call, not just seeding a different RNG with the same integer seed -- np/torch
generators would decohere after the very first walk whose step count differs by even one draw due
to a different float representation of `RNG_UNIF01()`. This is the same class of problem flagged
elsewhere in the sprint's RNG-matching work (see `feedback_no_runtime_delegation_to_reference.md`
context) -- a faithful port needs igraph's PCG32 stream reproduced exactly, which is a
well-scoped but nontrivial task (PCG32 is a public, simple algorithm; igraph vendors it under
`src/random/`).

**What actually caused the earlier "HUNG" characterization -- reproduced and diagnosed**:
I ported the algorithm above line-for-line into pure Python (`/tmp/r75_dla_probe.py`, kept as
scratch, NOT committed) and ran it against the graphs in this bucket's disconnected set.
- `multi_component_80` real component sizes (confirmed via `get_test_graphs(max_nodes=300)` +
  networkx): **only 7 components** (40, 20, 10, 5, 3, 1, 1) -- a mild case.
- `random_dag_50`: **52 components**, mostly isolated singleton nodes (45, 2, then 50x size-1).
- `random_dag_200`: **202 components**, the worst case (181, 2, then 200x size-1) -- confirmed
  via `get_test_graphs(max_nodes=500)`.
- On this worst case, a full pure-Python run of all 201 DLA placements (largest-first) completed
  in **17.4 seconds**, with per-particle inner-step counts bounded (max ~29,000 steps seen, no
  runaway growth as the grid filled) and total cumulative inner-walk steps of ~1.73M across all
  201 placements. This directly falsifies "infinite loop" -- it is a real but BOUNDED and
  moderate cost (single-digit seconds to tens of seconds per disconnected graph, well within an
  eval-harness budget), consistent with why the *real* igraph C implementation runs this in
  milliseconds (C loop overhead is ~2-3 orders of magnitude cheaper than CPython's).
- Caveat: my probe implements only 1 of the 4 quadrant-scan branches in `get_sphere`
  (`/tmp/r75_dla_probe.py`, `MergeGrid.get_sphere`, commented `# remaining 3 quadrants omitted for
  probe speed`) for speed. This makes the probe's occupancy checks slightly more permissive
  (a walk could rarely treat an occupied cell reachable only via one of the 3 unported quadrant
  scans as still empty). This does NOT change the qualitative finding (bounded random walk, no
  divergence) since walk length is governed by target-hit probability geometry, not lookup cost;
  a full 4-quadrant port adds constant per-step overhead only. Re-verify wall-clock with the full
  4-quadrant `get_sphere` before committing to a runtime budget for the real port.
- **Most likely explanation for the earlier "hang"**: the earlier attempt was probably killed by
  an impatient watchdog/timeout (this sprint's docs elsewhere note a general pattern of
  "I/O contention mistaken for a hang," `r74_close_all_gaps_STATE.md:212`) rather than a true
  infinite loop, OR it ran against a larger out-of-bucket graph (`ba_2000`/`ba_5000`, mentioned
  elsewhere in r74 as needing a "hang-safe scoring path"). I did not have access to the exact
  failed attempt's code/logs to confirm which.

**Bounded-time implementation strategy + LOC/effort estimate**:
1. Port igraph's PCG32 RNG generator (or confirm it's equivalent to `numpy.random.PCG64`'s core
   -- it likely is NOT identical bit-for-bit; igraph vendors its own PCG32, see
   `_references/igraph/src/random/` -- would need a dedicated small module, ~80-150 LOC, to
   reproduce `igraph_rng_get_unif` exactly, INCLUDING the exact float conversion of the raw
   32-bit output). This is the single highest-risk/highest-effort piece for BIT-exactness.
2. Port `vector_order` (component size-descending order; trivial, `torch.argsort`), the grid
   struct + 4-quadrant `place_sphere`/`get_sphere` (this bucket's spec's DLA/grid helpers;
   ~150-200 LOC, straightforward translation, vectorizable with numpy for the quadrant scans
   since each is an independent nested loop over a bounded window).
3. Port the outer/inner walk loop (~40 LOC, the `igraph_i_layout_merge_dla` function) -- this
   must stay a true Python/numpy loop (data-dependent early termination), not vectorizable across
   RNG draws within one walk, but IS vectorizable/parallelizable across independent DLA
   placements only in the sense of running each sequentially (they are NOT independent -- the
   grid occupancy carries state across placements, so this is inherently sequential per graph,
   though different (graph, seed) benchmark combos are trivially parallel across processes).
4. Wire into `classical_mds.py`: branch on `_connected_components_from_edges` (already exists,
   reused from `gem.py` in the reverted 91ccaab attempt) -- run `_layout_igraph_classical_mds`
   (the EXISTING connected-graph kernel, byte-identical, already bit-exact) per component, then
   DLA-merge instead of TileToRows-pack.
5. **Total estimate: ~400-550 LOC, 1 focused implementation session** (not "multi-day" as r74's
   STATE.md guessed -- that estimate predates this round's structural probe showing the walk is
   bounded-cost, not unbounded). The PCG32 port is the long pole; if PCG32 bit-exactness proves
   too fragile in practice, the fallback is targeting rung-3 (distributional/quality equivalence
   across the 100-seed battery) rather than rung-1 bit-exact, which only needs the STRUCTURE of
   the algorithm (component-then-DLA-pack) to be right, and would tolerate any well-mixed PRNG
   substitute -- much lower effort (~200 LOC, skip step 1, use any seeded PRNG for the walk).
   **RECOMMEND targeting rung-3/statistical equivalence first** given the TileToRows precedent
   showed structure matters far more than RNG fidelity for this metric (a WRONG merge structure
   caused pure harm; a right structure with an approximately-matched but not bit-exact RNG should
   at minimum stop being pure harm and should land in the 3Q battery's variance-tied margins).

**Risk to existing bit-exact combos**: LOW if gated correctly. The critical guard (learned from
both the maxent AND classical_mds r74 reverts) is: apply the component-split-and-DLA-merge path
ONLY when `_connected_components_from_edges` returns `len(components) > 1`; leave the
already-bit-exact CONNECTED path (`_layout_igraph_classical_mds`, single-component call)
completely untouched. The 91ccaab attempt already respected this guard structurally (it did not
touch the connected branch) and still caused 12 regressions -- so the guard alone is NOT
sufficient; the regressions came from the packing algorithm choice (deterministic vs stochastic),
not from over-application to connected graphs. A DLA-faithful port should not repeat that specific
failure mode since it changes WHAT merge algorithm runs, not WHERE it applies.

### Finding 2 (CONFIRMED, 12 combos): connected-graph classical_mds divergence is the pre-existing documented degenerate-eigenspace floor, not a new bug

All 12 non-equivalent connected combos (`bipartite_4_3_4`, `center_port_backedge_hub`,
`densenet_block`, `petersen_10`, `wide_single_layer_1_50_1`, `wide_3_50_3`, each x2 variants) show
the SAME `battery_stress` D/R values across `classic_classical_mds_default` and
`_igraph_fidelity` (e.g. `petersen_10` D=0.1682/R=0.1622 for BOTH variants) -- confirming
**Part 1, Q3: default and igraph_fidelity variants fail for the identical reason**, because both
route through the same `_layout_igraph_classical_mds` connected-graph kernel
(`classical_mds.py:171-177`: `if igraph_fidelity or edge_weights is None:` -- the unweighted
default variant takes the SAME code path as `igraph_fidelity=True`).

Root cause is already self-documented at `classical_mds.py:54-66`: igraph's vendored LAPACK 3.4.2
`dsyevr` (called with `range='I'`, `abstol=1e-14`) selects an implementation-dependent 2D basis
from a repeated-eigenvalue subspace whenever the top eigenvalue multiplicity exceeds 2 (true for
symmetric/regular graphs like Petersen, bipartite-complete-ish structures, wide layers with many
structurally-equivalent leaf nodes). The docstring records that a direct
`scipy.linalg.lapack.dsyevr` call with matching parameters was ALREADY tried and still picked a
different basis vector within the correct eigenspace. `petersen_10` (a vertex-transitive graph
with eigenvalue multiplicity 5 for its second-largest eigenvalue) is the textbook case for this.
This matches the target list's smallest gaps (`petersen_10` D=0.1682 vs R=0.1622, only 3.7% off;
`org_chart_1_5_4_8` already `equiv=true`) alongside larger gaps
(`wide_3_50_3` D=1.0 vs R=0.4748, 111% off) that are consistent with a *different chosen basis
producing a materially different 2D projection* of the same high-dimensional eigenspace, not a
small perturbation -- exactly what "same eigenvalues, different eigenvectors" predicts.

**Fix sketch**: none available without a full vendored-LAPACK-3.4.2 port (explicitly noted as
out of scope by the existing docstring). This is correctly the pre-existing "classical_mds
degenerate-eigenspace ~16" floor entry cited in `r74_close_all_gaps_STATE.md:52`. **RISK: N/A**
(no fix proposed). If a future sprint wants to chase this, the only tractable lever is checking
whether SciPy's `driver='evx'` (bisection+inverse-iteration, closer to LAPACK's classic `dsyevx`
than `dsyevr`) picks a basis that happens to match igraph's vendored 3.4.2 build more often --
CHEAP to test (swap `driver="evr"` to `driver="evx"` at `classical_mds.py:262`, rerun on
`petersen_10`/`bipartite_4_3_4`/`wide_3_50_3`, ~5 min experiment) but the docstring already
implies this class of fix was explored and abandoned; I did not re-run it this round given time
budget -- flagging as a cheap HYPOTHESIS worth 5 minutes if someone picks this up.

### Finding 3 (HYPOTHESIS, 7 combos): umap residual is optimizer/negative-sampling RNG divergence, not a comparison bug

**Params ARE matched.** Verified all 5 relevant variant defs in `dagua/eval/variants.py:1637-1702`
(`classic_umap_default/nn5/nn30/mindist001/spread2`) -- each variant's dagua `algo_params` and
reference `ref_params` dicts are identical key-for-key (`n_neighbors`, `min_dist`, `spread`).

**The r73 parallel-edge fix is already live** (`dagua/layout/ops/umap.py:123-163`,
`_build_undirected_adjacency`): parallel/duplicate edges are summed to match scipy's CSR
duplicate-coalescing behavior in the reference adapter
(`dagua/eval/competitors/umap_competitor.py:76-91`, `_distance_matrix`, which builds a CSR from
concatenated `(row,col)` pairs -- scipy sums weights on duplicates by construction). Verified this
is the SAME parallel-edge scenario (`parallel_multiedge_bundle`: 3 parallel src-mid edges, 2
parallel mid-dst edges) that the bucket brief called out as a suspect. Directly ran both dagua's
native pipeline and the benchmark competitor adapter on this exact 3-node graph at matched
params/seed (`layout_umap_layout_pipeline` vs `UMAPGraph.layout_with_variant`, both seed=1,
n_neighbors=15/min_dist=0.1/spread=1.0) -- both completed without error and produced structurally
different but non-degenerate layouts (dagua: roughly-equilateral triangle at ~1.0-1.4 unit
spacing; reference: at ~0.7-1.2 unit spacing with a different relative ordering of which pair is
closest). This is consistent with SGD-optimizer-path divergence (different local minimum from a
non-matching random negative-sampling stream), not a structural/parameter bug -- the underlying
distance-matrix-to-fuzzy-set construction pipeline is intended to match (dagua's docstring,
`umap_layout.py:39-52`, explicitly states it's a from-scratch native port that "constructs
precomputed shortest-path neighborhoods, the fuzzy simplicial set, spectral/random
initialization, and the sampled Euclidean SGD loop without importing umap-learn").

**Median stress excess -68% (dagua much lower stress = "better")** is consistent with dagua's SGD
converging to a tighter/more metric-faithful embedding than the real umap-learn's numba-JIT'd
negative-sampling loop, which uses its own `tau_rand_int` xorshift-style PRNG for negative
sampling (a different generator family than whatever seeds dagua's optimizer) -- different
negative-sample selection changes the attractive/repulsive balance reached at convergence.

**Cheapest decisive experiment (NOT run this round, time budget)**: instrument dagua's
`OptimizeUMAPEmbedding` op (`dagua/layout/ops/umap.py:1938`) to log its RNG call sequence and
compare against umap-learn's `optimize_layout_euclidean`'s `tau_rand_int` call sequence for a
tiny fixed graph (3-5 nodes, <50 epochs) -- if the epoch-by-epoch embedding trajectories diverge
immediately (epoch 1) rather than gradually, that confirms RNG-stream mismatch as the sole driver
(not, e.g., a residual formula difference). Est. runtime: ~15 min (small graph, few epochs,
direct trajectory diff). I did not run this given the 45-minute experiment budget was already
substantially spent on the classical_mds DLA reproduction (the higher-value target).

**Fix sketch**: port umap-learn's exact `tau_rand_int`/negative-sampling RNG stream (xorshift
variant, already public/simple) into dagua's `OptimizeUMAPEmbedding` -- moderate effort (~100-150
LOC), same PRNG-matching risk class as the DLA finding above. **Expected impact**: all 7 combos
route through the same `OptimizeUMAPEmbedding` SGD op, so a correct RNG port could plausibly move
all 7 at once IF the stress excess is purely RNG-driven (unconfirmed). **RISK to existing
bit-exact umap combos**: LOW -- changing only the RNG source inside an already-isolated op
shouldn't touch other pipelines, but verify n<=3 special-case fallback
(`umap_competitor.py:176-183`, `umap_layout.py`) stays untouched since those already use a
different code path (`torch.randn` fallback, not the SGD/negative-sampling loop) and aren't
affected either way.

### Finding 4 (CONFIRMED via reading existing r74 audits, 5 combos): gem remains the previously-established FP-summation-chaos floor

`dagua/eval/competitors/ogdf_competitor.py:342-350` registers `OGDFGem` with `is_stochastic =
True` already. OGDF's GEM layout (Frick-Ludwig-Mehldau) is a simulated-annealing spring embedder
that starts from `impulseLength` seeded by a local temperature/random-impulse-direction sampler
per node per iteration -- inherently stochastic by design, not merely by an implementation
accident. This matches r74's existing classification: "gem 22 (FP summation chaos; 269/309
already bit-exact)" (`r74_close_all_gaps_STATE.md:51`) and the r74 Phase-2 finding that 16 gem
combos already reclassified to quality-identical under the corrected variance-tied margin
(`r74_close_all_gaps_STATE.md:261`). All 5 combos in this bucket's target list
(`grid_5x5::classic_gem_iters100`, `random_dag_200/50::classic_gem_iters500/100` x2,
`regular_4_40::classic_gem_iters100`) are hairline (D/R gaps of 4-33%, well within the kind of
range the r74 audit already characterized as scale/variance-adjacent, not qualitatively
different). **I did NOT re-run a fresh 1-ULP perturbation experiment this round** to produce new
FP-chaos evidence for THESE SPECIFIC 5 combos (time budget) -- I am relying on r74's prior
characterization of gem as a whole rather than generating new evidence. Per the guardrail
("A floor/unfixable claim requires FP-chaos evidence... not an assertion"), this finding should
be treated as **HYPOTHESIS (inherited from r74, not independently re-verified this round)**,
recommend a follow-up sprint spend the ~20-30 min to run the perturbation test specifically on
these 5 residual combos before calling this bucket permanently closed.

**Fix sketch**: none proposed (algorithmic floor by design). **RISK**: N/A.

### Finding 5 (CONFIRMED, 3 combos): maxent residue is the genuine leftover after the r74 disconnected-packing dead-end was closed

r74 Option-2 (codex, high-effort, cited at `r74_close_all_gaps_STATE.md:236-238`) already
determined: OGDF's `StressMinimization` defaults `m_componentLayout(FALSE)` -- it does NOT
component-split disconnected graphs -- and the benchmark runner additionally sets
`hasInitialLayout(true)`. dagua's existing single-pass (whole-graph) approach ALREADY mirrors
this reference behavior correctly; the earlier "fix" (component-split + TileToRows,
`e756688`/reverted `6e5221b`) was diagnosed as blanket-applied to ALL disconnected graphs
including ones that were ALREADY bit-exact, causing 25 broke-bitexact + 31 regressed / 0 improved
-- "pure harm," and was correctly reverted. The 3 combos remaining in THIS bucket's target list
(`random_dag_50::classic_maxent_stress_default/steps400/steps50`) are what's left after that
dead-end closed -- I did not find a NEW root cause distinct from general stress-majorization
optimization-path/init differences (OGDF's `StressMinimization` iterative solver vs dagua's
port). Given `random_dag_50` is the 52-component near-degenerate graph (mostly isolated nodes)
discussed in Finding 1b, it's plausible these 3 combos are ALSO touched by the disconnected-graph
distance-matrix construction choice (dagua's single global fill vs OGDF's single-pass-with-
initial-layout, which are actually similar in spirit -- both do NOT split -- but may still differ
in exactly how the initial layout / distance fill is constructed for isolated nodes). **This is a
gap in my analysis this round**: I did not trace OGDF's `StressMinimization::call` in
`_references/ogdf` for its exact disconnected-graph distance-matrix fill value to compare against
dagua's `max_distance+1.0` choice. Flagging as an **explicit unexplained item** (see section 4).

**Fix sketch**: none proposed (root cause not fully isolated this round). **RISK**: N/A (no
change proposed).

### Finding 6 (CONFIRMED, 3 combos): drl residue matches the pipeline's own documented float32 rounding note

`dagua/layout/ops/pipelines/drl.py` docstring (`Known divergences`) already states: "Full-suite
parity depends on C++ float rounding and density-grid boundary behavior; the port rounds state
updates through float32." DrL (Distributed Recursive Layout, Martin/Brown/Klavans/Boyack) is a
multilevel force-directed algorithm using density-grid randomized initial placement + simulated
annealing across coarsening rounds -- inherent stochasticity by algorithm design (matches
`igraph_drl`'s `uses_igraph_rng = True` registration, `igraph_competitor.py:290-299`). The 3
target combos (`real_karate_34::classic_drl_refine`, `real_lesmis_77::classic_drl_refine`,
`real_lesmis_77::classic_drl_coarsen`) are all hairline (D/R gaps of 2-9% on `refine`, but
`coarsen` on `real_lesmis_77` is a larger 35% "dagua-much-better" gap that stands out and did NOT
get deep-dived this round -- see section 4). This matches "r71 fixed the main issues; classify
residue" framing from the bucket spec -- I did not find NEW evidence beyond the algorithm's
self-documented inherent stochasticity + float32 rounding for the `refine` combos.

**Fix sketch**: none proposed for `refine` combos (floor). `real_lesmis_77::classic_drl_coarsen`'s
outlier gap (margin 0.1254, largest margin in this whole tail set) deserves a follow-up look --
see section 4.

**RISK**: N/A.

### Finding 7 (CONFIRMED via reading target data, 2 combos): neato residuals are genuinely near-margin, consistent with the "do not propose margin changes" guardrail

Both `classic_neato` combos (`parallel_cycles_4x5`, `random_dag_50`) are `disconnected: true`.
`parallel_cycles_4x5::classic_neato` shows ALL THREE battery legs close to their own margins
(stress D=0.0193/R=0.0171, margin=0.0261 -- the |D-R| gap of 0.0022 is 12x SMALLER than the
margin itself; cross D=0.367/R=0.267, margin=1.15 -- again |D-R| far under margin; np
D=0.833/R=0.964, margin=0.028 -- here the gap 0.131 IS bigger than margin, the binding leg).
`random_dag_50::classic_neato` similarly has np already `equiv=true` (margin 0.02 vs its own
`ref_spread` of 0.0136), with stress and cross the binding (but still modest -4-16%) legs. Per
the bucket brief's explicit instruction, I am NOT proposing a margin recalibration -- just
flagging that these read as genuine small-sample near-threshold cases on graphs that are
ALSO disconnected (both are `disconnected:true`), so they may share SOME of Finding 1's
component-packing root cause (graphviz neato uses its own polyomino packer, already fixed per
r73 memory notes: "neato uses POLYOMINO ... not shelf" -- if that fix already landed, this residue
is the leftover AFTER polyomino packing, likely genuine algorithmic (CG solver / `drand48`
init) floor per neato.py's own docstring ("Exact CG solver behavior, raw drand48 initialization
parity ... remain unported," `neato.py:1274-1275`).

**Fix sketch**: none proposed. **RISK**: N/A.

---

## 3. Fix priority summary

| Root cause | Combos | Confidence | Effort | Risk to bit-exact combos |
|---|---|---|---|---|
| classical_mds disconnected: missing per-component MDS + DLA merge | ~16 | CONFIRMED | ~400-550 LOC, 1 session (rung-3 target; PCG32 bit-exact port is higher-effort optional stretch) | LOW if gated on `len(components)>1`; MUST use real DLA not TileToRows (proven pure-harm) |
| classical_mds connected: LAPACK dsyevr degenerate-eigenspace basis | 12 | CONFIRMED (pre-existing, re-verified) | N/A (documented floor); optional 5-min `driver="evx"` probe | N/A, no fix proposed |
| umap: SGD negative-sampling RNG stream mismatch | 7 | HYPOTHESIS (params/parallel-edge fix ruled out; RNG divergence not yet directly observed) | ~100-150 LOC IF confirmed; ~15 min experiment to confirm first | LOW (isolated to `OptimizeUMAPEmbedding`; verify n<=3 fallback untouched) |
| gem: inherent GEM-algorithm stochasticity (FP-chaos floor) | 5 | HYPOTHESIS (inherited r74 characterization, NOT independently re-verified this round) | N/A; ~20-30 min perturbation experiment recommended before final close-out | N/A |
| maxent: residual stress-majorization optimization-path difference | 3 | UNEXPLAINED (see section 4) | unknown | unknown |
| drl: inherent DrL stochasticity + float32 rounding (2 of 3 combos) | 2 | CONFIRMED (self-documented) | N/A, floor | N/A |
| drl: `real_lesmis_77::classic_drl_coarsen` outlier gap | 1 | UNEXPLAINED (see section 4) | unknown | unknown |
| neato: near-margin residue on disconnected graphs, likely floor after polyomino fix | 2 | CONFIRMED as near-margin; underlying cause not isolated | N/A | N/A |

---

## 4. Explicit list of target combos I could NOT explain (root cause not isolated this round)

1. **`random_dag_50::classic_maxent_stress_default`**, **`...steps400`**, **`...steps50`** (3
   combos) -- I confirmed the r74 dead-end (component-splitting) is correctly closed, but did NOT
   trace OGDF's `StressMinimization::call` (`_references/ogdf`) for its exact disconnected-graph
   handling to identify what DOES still differ. `random_dag_50` is a 52-component near-degenerate
   graph (mostly isolated nodes); worth checking whether OGDF's "no split" behavior still
   produces a materially different initial layout / distance treatment for isolated nodes than
   dagua's `max_distance+1.0` fill, even though neither approach component-splits.
2. **`real_lesmis_77::classic_drl_coarsen`** -- stands out from the other 2 drl combos with a
   much larger margin (0.1254 vs 0.02-0.0225) and the JSON shows `D=0.2612 R=0.3521` (dagua
   BETTER by ~26%, the "dagua-much-better = check for comparison bug" pattern flagged in the
   shared preamble). I did not investigate whether `options="coarsen"` (vs `"refine"` for the
   other two combos) routes through a different code path in `dagua/layout/ops/pipelines/drl.py`
   that might have a genuine parameter-handling difference from `igraph_drl`'s `options="coarsen"`
   preset, distinct from the inherent-stochasticity floor I attributed to the other two.
3. **umap RNG-divergence hypothesis is UNCONFIRMED** -- I built the case (params matched,
   parallel-edge fix live, native SGD port with independent RNG) but did not run the trajectory-diff
   experiment that would make this CONFIRMED rather than HYPOTHESIS.
4. **gem's FP-chaos floor claim is inherited from r74, not independently re-verified** for these
   specific 5 combos this round (no fresh 1-ULP perturbation experiment run).

## Scratch artifacts (not committed, per RESEARCH ONLY instructions)
- `/tmp/r75_dla_probe.py` -- pure-Python DLA/merge-grid reproduction used for the hang
  investigation and bounded-time feasibility check (Finding 1b). Kept for any follow-up sprint
  to reuse/extend (e.g. add the 3 missing `get_sphere` quadrants, wire in a real PCG32 port).
