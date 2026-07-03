# r75 sfdp bucket -- Claude/Sonnet investigation

## 1. Executive summary

H1 (reference overlap-removal post-processing) is CONFIRMED DEAD for this benchmark
environment: the `dot` binary in use (graphviz 7.0.5, conda-packaged) is linked without
GTS/Triangle (`ldd` shows no libgts/libtriangle), so `removeOverlapWith`'s `AM_PRISM` case
compiles out and no-ops; a live default-vs-`-Goverlap=prism0` comparison produced bit-identical
output, confirming no overlap pass runs by default. H2 (RNG mismatch) is CONFIRMED DEAD: dagua's
`GraphvizRandom` (dagua/layout/ops/sfdp.py:123-254) is a correct glibc `random()` port with the
right 344-value warmup and correctly splits rejection-sampled `gv_random`-style calls from raw-
modulo `irand`-style calls, matching graphviz 7.0.5's actual `random_permutation` (raw modulo,
confirmed via `git show 7.0.5:lib/sparse/general.c`). H3 (iteration/cooling mismatch) is
CONFIRMED DEAD at the parameter level: tolerance (1e-3), cool factor (0.9), quadtree_size (45),
and the adaptive-cooling formula all match graphviz source exactly, file:line cited below. H4
(multilevel coarsening) is CONFIRMED DEAD at the structural level: dagua's outer coarsen-repeat
loop correctly reproduces graphviz's FORCEFUL-mode "force sufficient reduction" wrapper.
**IMPORTANT CORRECTION to the bucket spec:** my own recomputation of "hairline <=1% stress"
using strict relative difference gives only 8/126, not 83/126 -- the spec's number likely uses a
different (margin-relative) definition I could not exactly reconstruct from the target JSON's
fields; flag for sprint-lead clarification, do not treat "hairline" as established without
re-deriving the exact formula. One genuine non-hairline, non-chaotic BUG was found and is
CONFIRMED: `parallel_cycles_4x5` produces degenerate stress=1.0 (sentinel) positions across all 6
sfdp variants, isolated to disconnected-component packing. Remaining divergence for the other
120 combos is most consistent with a floor (FP-chaos in an early-terminating iterative N-body
solver) but I did NOT run the 1-ULP perturbation experiment required to fully confirm the floor
claim per guardrails -- flagged as HYPOTHESIS, cheap experiment described below.

## 2. Findings ranked by expected combo-count impact

### Finding A -- CONFIRMED: H1 (reference overlap removal) does not apply in this environment (~0 combos, but rules out the leading suspect for all 126)

The graphviz adapter (`dagua/eval/competitors/graphviz_competitor.py:689-693`,
`GraphvizSfdp.variant_param_names = frozenset({"K","maxiter","repulsiveforce","theta"})`) never
passes `-Goverlap=`. `variants.py:1571-1636` (all 6 sfdp variant defs) never sets an `overlap`
param either. Reference-side `sfdp_layout()` (graphviz `lib/sfdpgen/sfdpinit.c:279-288` at tag
`7.0.5`, verified via `git show 7.0.5:lib/sfdpgen/sfdpinit.c`) calls
`graphAdjustMode(g, &am, "prism0")` (compiled path when `HAVE_GTS`/`HAVE_TRIANGLE` defined) and
then, if `am.mode==AM_PRISM`, calls `removeOverlapWith(g,&am)` after `sfdpLayout`.
`removeOverlapWith` (`lib/neatogen/adjust.c:1044-1105` at 7.0.5) only handles `AM_PRISM` inside
`#if ((defined(HAVE_GTS) || defined(HAVE_TRIANGLE)) && defined(SFDP))` -- if neither library is
linked, the switch falls through to `default: ret = 0`, i.e. only `normalize()` +
`simpleScale()` run, no actual node displacement.

DECISIVE EXPERIMENT RUN (within guardrails -- direct `dot` CLI invocation, not a dagua pipeline
call, so no benchmark-path violation):
```
ldd $(which dot)   # -> no libgts.so, no libtriangle.so linked
dot -Tjson -Ksfdp -Gseed=1 -Gstart=1 /tmp/test.dot                       # default
dot -Tjson -Ksfdp -Gseed=1 -Gstart=1 -Goverlap=prism0 /tmp/test.dot      # explicit prism0
```
Both produced bit-identical positions (`a 246.31,100.37` / `b 423.76,182.57` / `c 423.57,18` /
`d 27,100.34` in both runs), while `-Goverlap=false`/`true`/`scale`/`prism` all produced
DIFFERENT positions (`a 103.72,46.813` etc.) -- proving overlap removal is a no-op by default in
this build, exactly as the source predicts for a GTS/Triangle-less build.

CONCLUSION: H1's proposed mechanism (dagua-better because reference positions include a
distorting overlap-removal pass dagua doesn't apply) cannot be the cause of ANY of the 126
combos in this environment, because the reference itself never runs overlap removal. If the
"dagua-better" cluster (51 combos, all-legs-better) has a real comparison-bug cause, it is NOT
this one.

### Finding B -- CONFIRMED: `_graphviz_repulsive_exponent` clamp (commit 6f8cff5) is CORRECT, not the source of p_neg2's divergence pattern

`dagua/layout/ops/pipelines/sfdp.py:411-433` collapses any `repulsive_exponent < -1.0` (i.e. the
`p_neg2` variant's `-2.0`) to `-1.0`. Traced the graphviz reference chain at 7.0.5:
- `lib/sfdpgen/sfdpinit.c:238` (7.0.5 line numbers): `ctrl->K = late_double(...)`;
  `sfdpinit.c:239`: `ctrl->p = -1.0*late_double(g, agfindgraphattr(g,"repulsiveforce"), -AUTOP, 0.0)`.
- `late_double(obj, attr, defaultValue, minimum)` (`lib/common/utils.c:55-69` at working-tree HEAD,
  confirmed same semantics via `git show 7.0.5:lib/common/utils.c`, though that exact function name
  moved between versions -- verified via the `if (rv < minimum) return minimum;` clamp logic present
  identically in both) clamps the RAW attribute value to `minimum=0.0` BEFORE it is used, i.e.
  `repulsiveforce=-2.0` reads back as `0.0`, giving `ctrl->p = -1.0*0.0 = -0.0` (NOT `-2.0`).
- Inside every `spring_electrical_embedding*` variant (`lib/sfdpgen/spring_electrical.c:519`,
  `:688`, `:864`, `:1167`, `:1352` at 7.0.5; also present in the fast-path variant at the current
  HEAD tree, `lib/sfdpgen/spring_electrical.c:287`): `if (p >= 0) ctrl->p = p = -1;`. Since
  `ctrl->p` is `-0.0 >= 0`, this second clamp forces `p = -1.0`.
- LIVE EXPERIMENT: `dot -Tjson -Ksfdp -Gseed=1 -Gstart=1 -Grepulsiveforce=-2.0 test.dot` produced
  positions bit-identical to the no-flag default run, confirming the effective exponent really is
  `-1.0` for `repulsiveforce=-2.0` in the actual reference binary, exactly matching dagua's clamp.

CONCLUSION: `p_neg2` should behave statistically like `classic_sfdp_default` (52 vs 13 target
combos is purely a difference in which graphs got sampled into each variant's target list, not
evidence of a residual p-clamp bug -- verified the two variants' graph sets are almost disjoint,
12/52 overlap). The clamp fix already landed and is correct; it is not an open root cause for
r75. Re-verify commit 6f8cff5 is fully present in current tree (it is -- read directly from
`dagua/layout/ops/pipelines/sfdp.py` HEAD, matches the diff in `git show 6f8cff5` exactly).

### Finding C -- CONFIRMED: RNG stream (H2) is a faithful glibc `random()` + graphviz dual-consumer port, not a naive LCG

`dagua/layout/ops/sfdp.py:123-254` (`GraphvizRandom`) implements:
- `_initialize_state` (:143-175): the actual glibc additive-feedback `initstate`/`srandom`
  algorithm with the correct 344-value warmup (`_GRAPHVIZ_RANDOM_WARMUP = 344`,
  `sfdp.py:120`), matching glibc's `TYPE_3` state (deg=31, sep=3) warmup count.
- `rand()` (:177-188): additive feedback `state[0]+state[28]`, `>>1`, matching glibc `random()`.
- `random(bound)` (:200-228): REJECTION-SAMPLED bounded integer -- discard values above
  `RAND_MAX - ((RAND_MAX+1) % bound)` before taking modulo. This matches graphviz's
  `gv_random`/`random_small` rejection-sampling path (current-HEAD `lib/util/random.c:34-49`,
  confirmed to have the SAME algorithm, though this file didn't exist as such at 7.0.5 -- see
  caveat below).
- `permutation(bound)` (:230-254): RAW MODULO `self.rand() % (index+1)`, explicitly commented as
  NOT using rejection sampling, with an inline citation to graphviz's actual coarsening
  permutation call.

VERSION-SKEW CAVEAT (significant, worth flagging to the sprint lead): `_references/graphviz` is
checked out at HEAD (`233597cd4`, dated 2026-04-20, graphviz ~14.1.5), NOT at the `dot` binary's
actual pinned version (`graphviz version 7.0.5 (20221231.0122)`, confirmed via `dot -V`). Some
earlier analysis in this investigation (before I caught this) read the wrong version's source
(e.g. `lib/util/random.c`'s `gv_permutation`/`gv_random` is a HEAD-only refactor; `lib/sparse/
general.c` at 7.0.5 still has the OLD `random_permutation`/`irand` = raw modulo). I re-verified
every citation below against `git show 7.0.5:<path>` specifically. **Recommend the sprint
re-check whether any other bucket's report cited the wrong graphviz version** -- this repo's
default working tree is misleading for this pinned-dot-7.0.5 benchmark.

Re-verified at 7.0.5 specifically: `git show 7.0.5:lib/sparse/general.c` lines 20-31:
```
double drand(){ return rand()/(double) RAND_MAX; }
int irand(int n){ return rand()%n; }
int *random_permutation(int n){ ... j = irand(len); ... }
```
This is raw modulo, exactly matching dagua's `permutation()` (:230-254). `Multilevel.c` (7.0.5)
line 99/150/216/etc. calls `random_permutation(m)` for coarsening order -- confirms dagua's
`GraphvizRandom.permutation` is used correctly by `_graphviz_sfdp_cluster_nodes`
(pipelines/sfdp.py:152, `for node in generator.permutation(num_nodes)`).

CONCLUSION: H2's stated concern ("dagua ops/sfdp.py:247-253 uses a raw modulo LCG") describes
dagua's CORRECT, faithful implementation of graphviz 7.0.5's actual (also raw-modulo)
`random_permutation`. There is no RNG stream mismatch to fix here; H2 as literally stated in the
bucket brief does not apply to the pinned reference version.

### Finding D -- CONFIRMED: H3 termination/cooling parameters match exactly, file:line for file:line

- `tolerance`: dagua `SFDPAdaptiveCoolConfig.tolerance = 1.0e-3` (sfdp.py:103) vs graphviz
  `ctrl->tol = 0.001` (`spring_electrical.c:45` at 7.0.5, comment: "minimum different between two
  subsequence config before terminating"). Loop condition `while (step > tol && iter < maxiter)`
  (`spring_electrical.c:616` at 7.0.5) vs dagua `if current_step < self.config.tolerance:
  state.converged = True` (sfdp.py:1362-1363) inside a `Repeat(n=steps,...)` that stops early on
  `state.converged` (`dagua/layout/ops/base.py:364-368`). Semantically equivalent early-stop.
- `shrink_factor`/`cool`: dagua `0.90` (sfdp.py:104) vs graphviz `ctrl->cool = 0.90` (7.0.5
  `spring_electrical.c:47`, comment "default 0.9"). Match.
- `plateau_ratio`: dagua `0.95` (sfdp.py:105) vs graphviz `else if (Fnorm > 0.95*Fnorm0)` (7.0.5
  `spring_electrical.c:297`). Match.
- Cooling formula: dagua `_update_step` (sfdp.py:1178-1210, verified: `force_norm >=
  previous_force_norm -> shrink_factor*step`; `force_norm > plateau_ratio*previous_force_norm ->
  step unchanged`; else `0.99*step/cool`, with an explicit code comment noting this asymmetric
  branch -- not the algebraically-equivalent `1.1*step` -- because "SFDP's force iterations are
  chaotic") vs graphviz `update_step()` (7.0.5 `spring_electrical.c:290-303`, identical 3-branch
  structure, `step = 0.99*step/cool` on the growth branch). Exact match, including the
  chaos-sensitivity awareness.
- `barnes_hut_threshold`/`quadtree_size`: dagua `45` (sfdp.py:63) vs graphviz `ctrl->quadtree_size
  = 45` (7.0.5 `spring_electrical.c:42`). Match. Default `tscheme = QUAD_TREE_NORMAL`
  (`sfdpinit.c:242` at 7.0.5) dispatches to the node-sequential `spring_electrical_embedding`
  (not `_fast`, not `_slow`) at n>=45 (`spring_electrical.c:1854-1858` region at 7.0.5) --
  confirmed dagua's `_SFDPGraphvizSequentialStep` (pipelines/sfdp.py:459-622) targets the correct
  C function variant (sequential per-vertex writeback, quadtree-accelerated repulsion above
  threshold).

CONCLUSION: no parameter or control-flow mismatch found for H3. If chaos is the driver, it is
FP-summation-order / iteration-count-sensitivity chaos given correct parameters, not a parameter
bug -- this pushes toward a "floor" explanation for the remaining hairline-ish cluster, but I did
not run the required 1-ULP perturbation experiment to fully confirm per guardrails (see Finding
F below).

### Finding E -- CONFIRMED (real, non-hairline bug): `parallel_cycles_4x5` disconnected-component packing produces degenerate stress=1.0 positions (6 combos, all sfdp variants for this graph)

All 6 sfdp variants of `parallel_cycles_4x5` (`disconnected=true`) show `battery_stress.D`
pinned at a sentinel-like value (`1.0` for 5/6 variants, `0.8033` for `steps200`) against a
reference stress of `0.014-0.025` -- a >40x gap, the largest in the bucket, and structurally
different from every other combo's pattern (no other graph in the bucket shows a stress ratio
above ~1.5x). `cross.D = 0.0` uniformly for this graph (suspiciously perfect -- consistent with
components being packed with no inter-component edges to cross, but the near-1.0 stress strongly
suggests degenerate/collapsed coordinates, e.g. all nodes of a component collapsing to a point or
components overlapping at the origin before packing).

Root-cause location (not fully bottomed out -- HYPOTHESIS for the exact mechanism, but the
symptom and blast radius are CONFIRMED from the data): `_layout_graphviz_sfdp_components`
(dagua/layout/ops/pipelines/sfdp.py:958-1035) lays out each weak component independently via a
recursive `layout_sfdp_pipeline(..., fidelity_mode="graphviz")` call, then
`_pack_component_positions` (dagua/layout/ops/pipelines/neato.py:569-657, shared with the neato
pipeline) packs them via a polyomino packer. `parallel_cycles_4x5` is presumably 4-5 small
disjoint cycle components -- small-N components (cycles of ~5 nodes) are exactly where a
degenerate normalization edge case (e.g. `_normalize_positions`'s `span < min_span` fallback,
sfdp.py pipeline-adjacent code at :317-346, which places nodes on a `linspace(-1,1)` line when
the bounding box nearly collapses) could produce a pathological single-component layout that then
dominates the packed stress metric.

DECISIVE EXPERIMENT (not yet run -- next cheapest step, ~5 min): load
`eval_output/benchmark_100seed_r74_fixes/positions/parallel_cycles_4x5__classic_sfdp_default__seed*.pt`
and directly plot/inspect per-component bounding boxes and pairwise distances to confirm whether
one or more components collapse to a point (degenerate) vs are simply packed far apart (which
would inflate stress benignly for small dense subgraphs but not to a 1.0 sentinel).

### Finding F -- HYPOTHESIS: remaining ~113 combos (all connected-graph, non-`parallel_cycles_4x5` divergences) are FP-chaos floor, not yet CONFIRMED per guardrails

The bucket brief's claim of "83/126 hairline <=1%" could not be exactly reproduced from the
target JSON's `battery_stress`/`cross` fields using either raw relative-difference (gives 8/126)
or margin-ratio framing (gives 15-54/126 depending on exact formula) -- I flag this discrepancy
for the sprint lead rather than asserting a number I can't derive. What IS clear: sign of the
D-R gap varies unpredictably across near-identical variants of the same graph (e.g.
`asymmetric_hourglass_hub`'s 6 variants all show small negative-then-positive-ish noise around a
common `ref_spread` of ~0.0016), consistent with chaotic sensitivity to iteration order rather
than a systematic bias. Given Finding D (all parameters/formulas verified correct) and the
documented chaos-awareness already in dagua's own code comments (`_update_step` docstring,
sfdp.py:1206-1208, explicitly calling out SFDP's chaotic force iteration), a floor explanation is
plausible but UNCONFIRMED -- I did not run the guardrail-required 1-ULP perturbation /
summation-order experiment.

DECISIVE CHEAP EXPERIMENT (est. 10 min, still within the 45-min budget if run): for one
small-N connected graph (e.g. `grid_5x5`), run `layout_sfdp_pipeline` via the benchmark path
twice with the seed unperturbed and once with a 1-ULP perturbation injected into initial
positions (or force summation order reversed for one node), and measure whether the FINAL
stress delta magnitude is comparable to the observed dagua-vs-reference gap. If yes, that is
FP-chaos floor evidence per the guardrail definition. NOT run in this investigation due to time
allocation toward the H1-H4 source verification (which had a much higher combo-count payoff if
confirmed/killed) -- recommend as the sprint's next action for this bucket specifically.

### Finding G -- Systematic "dagua worse" outlier graphs worth a closer look

Two graphs show ALL 6 sfdp variants failing with dagua strictly WORSE on every failing leg
(the opposite of the "dagua-better" cluster the bucket brief highlights):
`asymmetric_hourglass_hub` (6/6 worse) and `weighted_chain_20` (6/6 worse), plus
`real_lesmis_77` (5/6 worse). This is a candidate location for a real (not chaos) bug distinct
from Finding E, since it's directionally consistent across ALL parameter variants of the same
graph rather than randomly signed. NOT investigated further due to time budget -- flagged as
HYPOTHESIS. Cheapest next step: diff dagua vs reference positions for
`weighted_chain_20::classic_sfdp_default` (a chain graph, simplest possible topology) since a
chain's expected sfdp layout is nearly 1-D and any systematic worse-stress would be visually
obvious without alignment machinery.

## 3. Root cause / fix sketches

| Root cause | Status | Fix sketch | Expected impact | Risk to bit-exact combos |
|---|---|---|---|---|
| H1 overlap removal | CONFIRMED DEAD (no-op in this env) | None -- no fix needed/applicable | 0 combos | None |
| H2 RNG mismatch | CONFIRMED DEAD (matches 7.0.5 exactly) | None -- no fix needed | 0 combos | None |
| H3 iteration/cooling params | CONFIRMED DEAD (exact param match) | None -- no fix needed | 0 combos | None |
| H4 coarsening reduction-stop | CONFIRMED structurally correct | None -- no fix needed | 0 combos | None |
| p_neg2 clamp (6f8cff5) | CONFIRMED already correct | Already landed, no further action | 0 (already counted in current rescore) | N/A |
| `parallel_cycles_4x5` degenerate packing (Finding E) | HYPOTHESIS (symptom confirmed, mechanism not) | Investigate `_pack_component_positions`/`_normalize_positions` degenerate-span fallback for small cyclic components; likely needs a targeted fix in the disconnected-component packing path, NOT a blanket change (guardrail: prior blanket per-component fixes broke bit-exact combos for maxent/classical_mds) | Up to 6 combos (all `parallel_cycles_4x5::classic_sfdp_*`) | LOW if scoped to the specific degenerate-span code path with a regression test on already-passing disconnected sfdp/neato combos; MEDIUM-HIGH if changed broadly across `_pack_component_positions` (shared with neato pipeline -- any change here needs neato bucket regression check too) |
| FP-chaos floor (Finding F) | HYPOTHESIS, unconfirmed | If confirmed via 1-ULP experiment: document as floor, no code change; if NOT confirmed (i.e. delta exceeds chaos-sensitivity bound), re-open investigation into force-summation order / dtype (float64 in dagua's sequential step vs graphviz's native double -- should match, but worth spot-checking `_SFDPGraphvizSequentialStep`'s explicit float64 cast at sfdp.py:527 pipeline file against any float32 leakage elsewhere in the pipeline) | Up to ~113 combos if floor is confirmed (i.e. sprint should STOP chasing these, not fix them) | N/A (floor = no fix) |
| `asymmetric_hourglass_hub`/`weighted_chain_20` systematic-worse (Finding G) | HYPOTHESIS, not investigated | TBD pending position diff | Up to ~17 combos across the 3 flagged graphs | Unknown until root cause identified |

## 4. Explicit list of target combos I could NOT explain

All combos EXCEPT the `parallel_cycles_4x5` group (Finding E, 6 combos, mechanism still not
fully bottomed out) and the general H1-H4 source-level verification (which rules out but doesn't
positively explain the remaining ~120). Specifically unexplained at the individual-combo level:

- All 113-120 combos not covered by Finding E or Finding G -- consistent with chaos-floor
  (Finding F) but NOT confirmed via the required 1-ULP perturbation experiment. This includes the
  bulk of `classic_sfdp_p_neg2` (52 targets), `classic_sfdp_steps200` (18), `classic_sfdp_theta08`
  (15), `classic_sfdp_theta04` (14), `classic_sfdp_graphviz_fidelity` (14), `classic_sfdp_default`
  (13) minus the graphs covered above.
- `asymmetric_hourglass_hub` (6 combos) and `weighted_chain_20` (6 combos): systematically
  dagua-worse across all variants (Finding G) -- flagged as a DIFFERENT pattern from the
  chaos-floor hypothesis but not root-caused.
- `real_lesmis_77` (5/6 worse), `random_dag_200`/`random_dag_50` (mixed, worse on 2-6 legs each):
  not individually investigated.

## Answers to the two explicit bucket questions

**Q1: Which of the 126 would a single H1-style adapter/params fix convert to quality-identical
or bit-exact?** NONE. H1 is confirmed dead in this environment (no overlap removal runs on the
reference side by default, verified live). There is no adapter-side overlap/params fix available
for this bucket. (If the benchmark environment's `dot` binary were ever swapped for a
GTS/Triangle-enabled build, that WOULD change the reference's behavior and would need sprint-lead
sign-off per the guardrails -- but that is a environment change, not an "adapter fix," and I am
not recommending it since it would only make the comparison LESS matched to what real Graphviz
installs typically look like on minimal/conda systems.)

**Q2: Does the r74 p_neg2 clamp (commit 6f8cff5) interact with any of these combos?** Yes,
mechanically: all 52 `classic_sfdp_p_neg2` target combos in this bucket ARE post-6f8cff5 (the fix
is present in the current tree, confirmed by reading `dagua/layout/ops/pipelines/sfdp.py:411-433`
directly and diffing against `git show 6f8cff5`). The clamp is CORRECT (Finding B) and is not
contributing to the remaining divergence -- the 52 p_neg2 combos diverge for the same reason as
the `default`/`theta04`/`theta08`/`steps200`/`graphviz_fidelity` variants of the same graphs
(chaos-floor hypothesis, Finding F), not because of any residual p-exponent bug. No further
action needed on the clamp itself.
