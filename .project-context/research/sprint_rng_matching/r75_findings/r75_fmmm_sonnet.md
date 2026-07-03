# r75 fmmm bucket -- sonnet findings

## 1. Executive summary

- All 32 target combos set `fidelity_mode=True`, which routes through
  `_layout_ogdf_fmmm_component_fidelity` in `dagua/layout/ops/pipelines/fmmm.py`
  (NOT the older `ops/fmmm.py` op-graph path with `_InitializeCoarsestLevel` /
  `_build_hierarchy` -- that path is dead code for these variants; see Finding 4
  for why it still matters for a 3-4 combo sub-bucket).
- 25/32 targets are single-component graphs with <=48 nodes, which take the
  `_layout_ogdf_fmmm_small_fidelity` single-level path (OGDF's own
  `numberOfNodes() > minGraphSize(50)` gate never fires below 50 nodes, so
  multilevel/galaxy coarsening is provably NOT involved for this majority
  sub-bucket -- ruled out with source citations, see Finding 5).
- Repulsive-force NMM-vs-exact was a strong initial hypothesis but is RULED OUT
  for every target graph except `random_dag_200` (383 nodes): OGDF's NMM
  approximation only activates above `MIN_NODE_NUMBER = 175`
  (`NewMultipoleMethod.cpp:122`); below that it silently falls back to the same
  exact O(N^2) method dagua uses.
- CONFIRMED, highest-impact, single-line bug (Finding 1): OGDF's
  `numexcept::f_rep_u_on_v` (`numexcept.cpp:169-182`) jitters two exactly-coincident
  node positions apart by drawing from the OGDF RNG stream before computing the
  repulsion force; dagua's `_ogdf_fmmm_tensor_repulsive_forces`
  (`pipelines/fmmm.py:562-585`) instead silently zeroes the force for any
  distance-0 pair and never perturbs. Positions are floored to integers every
  iteration (`_ogdf_fmmm_adjust_positions`), which makes coincidences plausible
  at these small N. This both changes forces AND desyncs the RNG stream for
  every later random draw once triggered -- a stream desync cascades and would
  explain the "genuinely worse, structurally different" symptom on small graphs
  far better than a small local force error would.
- HYPOTHESIS, second-priority (Finding 2): `_ogdf_fmmm_angle`
  (`pipelines/fmmm.py:168-196`) computes the oscillation-damping angle via the
  cross/dot `atan2(ax*by-ay*bx, ax*bx+ay*by)` identity, while OGDF's real
  `DPoint::angle` (`geometry.h:135-150`) computes
  `atan2(dy2,dx2) - atan2(dy1,dx1)`. Mathematically equivalent, but the
  `ceil(angle/pi_over_6)` bucket lookup used for oscillation damping
  (`prevent_oscillations`, `FMMMLayout.cpp:1298`) is extremely sensitive to
  which side of a `pi/6` multiple the float lands on -- different rounding paths
  can flip a damping factor bucket on any iteration where the true angle is
  near a boundary. Cheap to test (see fix sketch).
- HYPOTHESIS, scoped to 3-4 combos (Finding 3): the coarsening RNG for the
  multilevel path (`random_dag_50`, `random_dag_200`, `multi_component_80`,
  used via `_layout_ogdf_fmmm_multilevel_fidelity`) uses one continuous Python
  stdlib `random.Random(seed)` stream shared across ALL hierarchy levels
  (`_build_hierarchy`, `ops/fmmm.py:708-733`), while OGDF's real
  `Multilevel::create_suns_and_planets` calls `Node_Set.set_seed(rand_seed)`
  --  the *same fixed seed* -- at the start of EVERY level
  (`Multilevel.cpp:60,135`). Neither side is even using the same RNG algorithm
  (stdlib Mersenne Twister vs OGDF's `Set::get_random_element` reading the
  ported `_OgdfMt19937`/`randomNumber()` stream), so this sub-bucket is not
  bit-exact-repairable without a much larger galaxy-partitioning RNG port.
- Confirmed NOT the cause: `stopCriterion` early-stop (the actual reference
  binary always forces `FixedIterations`, ruling out threshold-based early
  termination as a source of divergence -- see "ruled out" section).
- Empirical symptom check on `deep_chain_20`/`classic_fmmm_steps10`/seed100
  (single-level path, benchmark competitor calls, matched params): dagua's
  layout is a similar overall spiral shape to the reference (both curl the
  same direction, similar centroid/extent) but has much noisier edge lengths
  (std 5.7 vs 2.8, min 34.9 vs 43.0, ideal=48.3) -- consistent with an RNG
  stream desync partway through the force loop rather than a systematically
  wrong formula (a wrong formula would produce a globally different shape, not
  just added per-iteration noise).

## 2. Findings ranked by expected combo-count impact

### Finding 1 (CONFIRMED code-diff, HYPOTHESIS for combo-count impact): missing RNG-jitter on coincident-position repulsion

- dagua: `dagua/layout/ops/pipelines/fmmm.py:562-585`
  (`_ogdf_fmmm_tensor_repulsive_forces`):
  ```python
  delta = positions.unsqueeze(1) - positions.unsqueeze(0)
  distances = torch.linalg.norm(delta, dim=2)
  factor = torch.zeros_like(distances)
  nonzero = distances > 0.0
  factor[nonzero] = 1.0 / distances[nonzero].square()
  ```
  When `distances == 0` (coincident positions), `factor` stays `0` -- the pair
  contributes **zero** repulsive force, permanently.
- OGDF: `src/ogdf/energybased/fmmm/numexcept.cpp:169-182`
  (`numexcept::f_rep_u_on_v`):
  ```cpp
  if (pos_u == pos_v) {
      pos_u = choose_distinct_random_point_in_radius_epsilon(pos_u);
  }
  ```
  `choose_distinct_random_point_in_radius_epsilon` ->
  `choose_distinct_random_point_in_disque` (`numexcept.cpp:48-103`) runs a
  `do { ... } while(...)` loop drawing `randomNumber(1, BILLION)` pairs from
  the SAME global `s_random` mt19937 stream (`basic.cpp:120,133,143`) that
  everything else in FMMM reads from (initial placement, force iterations
  downstream). Every triggered collision consumes an unpredictable number of
  RNG draws (loop repeats until the jittered point clears a distance
  threshold), permanently desyncing dagua's mt19937 stream from OGDF's from
  that point forward.
- Why plausible at these scales: `_ogdf_fmmm_adjust_positions`
  (`pipelines/fmmm.py:277-325`, ported faithfully from
  `FMMMLayout.cpp:550-619`) floors every node's position to an integer on
  every iteration. With boxlength on the order of a few hundred units and
  20-50 nodes, exact-integer collisions after flooring are far from
  vanishingly rare, especially in the first few iterations right after random
  placement, or once the oscillation-damping factors near-freeze several
  nodes onto the same pixel.
- CONFIRMED as a code difference. HYPOTHESIS on causing this specific bucket:
  I did not instrument a live run to count actual triggered collisions per
  target graph (would need to run the reference C++ runner under a debugger
  or add stderr counters, which is out of the <45 min budget). Cheapest
  decisive experiment (~10 min): add a Python-side counter to
  `_ogdf_fmmm_tensor_repulsive_forces` for `nonzero.numel() - nonzero.sum()`
  summed over all iterations of a `deep_chain_20`/seed100 run; a nonzero count
  strongly implicates this path given the RNG-stream-desync mechanism.
- Fix sketch: port `numexcept::choose_distinct_random_point_in_disque` /
  `f_rep_near_machine_precision` exactly (reuse the already-ported
  `_OgdfMt19937` class at `pipelines/fmmm.py:66-149`) and call it from the
  scalar/loop-based `_ogdf_fmmm_repulsive_forces` (the CPU list-based
  reference implementation already present at `pipelines/fmmm.py:328-356` --
  note the tensor path at :562 is a *fast* reimplementation of the same
  function and currently silently diverges from its own sibling function's
  intent). The tensor path cannot vectorize RNG-jitter cleanly since draws are
  sequential and history-dependent; likely needs a fallback to the CPU/scalar
  loop whenever any zero-distance pair is detected in a batch, which is rare
  enough not to hurt performance.
- Expected impact: potentially most/all 25 single-level small-graph combos,
  since RNG desync cascades through the rest of the run (explains "genuinely
  worse", not just noisier). Also plausibly explains part of Finding 3's
  multilevel bucket (same function is called at every level).
- Risk to existing bit-exact/3Q fmmm combos: LOW if implemented as a
  detect-and-fallback (only activates on an actual zero-distance pair, which
  by definition never happens in currently-passing combos or they'd already
  show this symptom). Do not change the fast path's numerics when no collision
  occurs.

### Finding 2 (CONFIRMED code-diff, HYPOTHESIS for impact): oscillation-angle formula uses a different (mathematically equivalent but numerically distinct) atan2 identity

- dagua: `dagua/layout/ops/pipelines/fmmm.py:168-196` (`_ogdf_fmmm_angle`):
  `atan2(ax*by - ay*bx, ax*bx + ay*by)` where `a = first-center`,
  `b = second-center`.
- OGDF: `include/ogdf/basic/geometry.h:135-150` (`GenericPoint::angle`):
  `atan2(dy2,dx2) - atan2(dy1,dx1)`, wrapped into `[0,2pi)`.
- Both formulas compute the same mathematical angle, but floating-point
  rounding differs, and this angle feeds directly into
  `factors[int(ceil(angle/0.52359878))]` (`prevent_oscillations`,
  `FMMMLayout.cpp:1298`; dagua mirror at `pipelines/fmmm.py:526` and the
  tensor version at :752). A `ceil()` bucket lookup is a step function --
  any angle landing within float-epsilon of a `pi/6` multiple can select a
  different damping factor (values range 0.33 to 2.0 across neighboring
  buckets), producing a discontinuous divergence in movement magnitude for
  that node on that iteration. This is exactly the kind of high-sensitivity,
  low-frequency-trigger bug that would show up as scattered noise (matching
  the elevated edge-length std observed empirically) rather than a systematic
  shift.
- Decisive experiment (~5 min): swap `_ogdf_fmmm_angle`'s formula to the
  OGDF subtraction form and diff output positions bit-for-bit against the
  current tensor implementation on a few of the smaller target graphs; if
  positions are bit-identical, this formula difference has zero real-world
  effect (both round to identical floats at these magnitudes) and can be
  eliminated as a candidate. This is the cheapest test in this report and
  should be run FIRST before investing in Finding 1's fix.
- Fix sketch: replace `_ogdf_fmmm_angle` (and its tensor sibling
  `_ogdf_fmmm_tensor_prevent_oscillations`'s `atan2(cross, dot)` at line 750)
  with the literal `atan2(dy2,dx2) - atan2(dy1,dx1)` subtraction form to
  match OGDF exactly, including its explicit zero-vector early return
  (`geometry.h:140-142`, currently unneeded since callers already guard
  `norm_new>0 and norm_old>0`, but keep for defensiveness/fidelity).
- Risk to existing bit-exact/3Q combos: LOW-MEDIUM. If the two formulas are
  NOT bit-identical in practice (likely, given IEEE754 rounding is
  path-dependent), this fix could shift results for currently-passing
  bit-exact combos too, since ALL fmmm fidelity runs call this function.
  MUST re-run the full fmmm 33-combo bit-exact/3Q regression set after this
  change, not just the 32 divergent targets.

### Finding 3 (CONFIRMED code-diff, scoped HYPOTHESIS): multilevel coarsening RNG doesn't reseed per-level and uses the wrong generator entirely

- Affects: `random_dag_50` (97 nodes, 4 variant combos in target list),
  `random_dag_200` (383 nodes, 1 combo), `multi_component_80` (80 nodes total,
  1 combo) -- up to 6 of the 32 targets, all multi-component or >50-node
  single components that hit `_layout_ogdf_fmmm_multilevel_fidelity`
  (`pipelines/fmmm.py:1587`) via the `numberOfNodes() > minGraphSize(50)` gate
  confirmed at `FMMMLayout.cpp:69,163-171`.
- dagua: `_build_hierarchy` (`dagua/layout/ops/fmmm.py:657-734`) creates ONE
  `rng = random.Random(seed)` (line 708) and threads it through every
  `_coarsen_level` call across the whole coarsening loop (line 720-725) --
  a single continuous stdlib Mersenne Twister stream.
- OGDF: `Multilevel::create_multilevel_representations`
  (`Multilevel.cpp:55-84`) calls `setSeed(rand_seed)` ONCE up front (line 60,
  reseeding the global `s_random`), but then for EVERY coarsening level calls
  `partition_galaxy_into_solar_systems` -> `create_suns_and_planets`
  (`Multilevel.cpp:119-137`), which itself calls
  `Node_Set.set_seed(rand_seed)` (line 135) with the SAME fixed `rand_seed`
  value every time -- i.e. OGDF's `Set` RNG state (a distinct structure from
  the global `s_random`, see `Set::get_random_element` at `Set.cpp:78`) is
  reset to an identical starting point at the start of every level, not
  advanced continuously.
- Additionally, dagua's stream is Python stdlib `random.Random` (Mersenne
  Twister with different seeding/tempering constants from a raw
  `std::mt19937`), not the already-ported `_OgdfMt19937` class
  (`pipelines/fmmm.py:66-149`) that the single-level and coarsest-level-init
  paths correctly use (`_ogdf_fmmm_random_placement`,
  `pipelines/fmmm.py:252-274,1624,1899`). This is a second, independent
  reason the multilevel coarsening cannot currently be bit-exact regardless
  of the reseed-per-level fix.
- This is a genuinely different, larger scope of work than Findings 1-2: it
  requires porting `Set::get_random_element` / `Set::get_random_element_with_lowest_star_mass`
  (`Set.cpp`) as a faithful `_OgdfMt19937`-backed structure, not just fixing
  the reseed cadence. Not attempted here (out of the <45 min budget) --
  flagged as scoped future work, matching the sprint's precedent of
  "seeded refs proved bit-exact" wins on other engines (r71).
- Fix sketch (future sprint): (a) replace `random.Random(seed)` with
  `_OgdfMt19937(seed)` in `_build_hierarchy`; (b) reseed to the SAME seed
  value at the top of every `_coarsen_level` call instead of threading one
  continuous rng object; (c) port `Set`'s node-selection algorithm
  (`get_random_node_with_lowest_star_mass`/`_highest_star_mass`) to consume
  the OGDF-equivalent RNG stream exactly, including the `random_tries=20`
  retry-selection semantics (`Multilevel.cpp` uses `randomTries()` default
  20, need to verify dagua's `_SOLAR_RANDOM_TRIES` matches).
- Risk: this sub-bucket is disjoint from the single-level path (different
  function, `_layout_ogdf_fmmm_multilevel_fidelity` vs
  `_layout_ogdf_fmmm_small_fidelity`), so a fix here has essentially ZERO risk
  to the 25 single-level target combos or their currently-passing siblings.
  Risk is confined to other multilevel-eligible fmmm combos not in this
  divergent list (need to check whether any currently-bit-exact/3Q fmmm
  combos also route through the multilevel path before touching this code --
  not verified in this pass).

## 3. Ruled-out candidates (with evidence)

- **Multilevel/galaxy threshold divergence for the 25-combo majority bucket**:
  RULED OUT. `minGraphSize` defaults to 50
  (`include/ogdf/energybased/FMMMLayout.h:138`, confirmed via `resetOptions()`
  at `FMMMLayout.cpp:275`) and the coarsening `while` loop condition is
  `act_Graph_ptr->numberOfNodes() > min_Graph_size` (`Multilevel.cpp:69`).
  All but 3 target graphs (`random_dag_50`=97, `random_dag_200`=383,
  `multi_component_80`=80 nodes total) have <=48 nodes measured directly via
  `dagua.eval.graphs.get_test_graphs()`. `ogdf_runner.cpp` never overrides
  `minGraphSize` or `singleLevel`, so the reference binary takes the same
  single-level path dagua's `fidelity_mode` branch does for these 25 combos
  (confirmed at `pipelines/fmmm.py:1787-1803`,
  `len(components)<=1 and num_nodes<=50 -> _layout_ogdf_fmmm_small_fidelity`).
- **NMM (New Multipole Method) approximate repulsion vs dagua's exact O(N^2)**:
  RULED OUT for all targets except `random_dag_200`. OGDF's own
  `NewMultipoleMethod::make_initialisations`
  (`NewMultipoleMethod.cpp:176-192`) only sets `using_NMM=true` when
  `G.numberOfNodes() >= MIN_NODE_NUMBER` where `MIN_NODE_NUMBER=175`
  (`NewMultipoleMethod.cpp:122`); below that it calls
  `ExactMethod.make_initialisations` and `calculate_repulsive_forces_by_exact_method`
  (`NewMultipoleMethod.cpp:171-174,188-190`), which is the same exact-summation
  method dagua's tensor path implements. Force-magnitude formula also verified
  identical: OGDF `f_rep_scalar(d) = 1/d` combined with the outer `/d` division
  in `f_rep_u_on_v` (`numexcept.h:78-81`, `numexcept.cpp:178`) yields `1/d^2`,
  matching dagua's `1.0 / distances.square()` exactly (`pipelines/fmmm.py:583`).
  Only `random_dag_200` (383 nodes > 175) would actually diverge on this axis
  and even then only through the multilevel path's coarsest level or any
  level whose node count stays above 175.
- **`stopCriterion` early termination (threshold-based stop) causing dagua to
  over-iterate relative to reference**: RULED OUT for THIS benchmark's data.
  OGDF's true default is `StopCriterion::FixedIterationsOrThreshold` with
  `threshold(0.01)` (`FMMMLayout.cpp:288-289`), which could stop early. BUT
  the actual C++ runner that generates our `ogdf_fmmm__for__classic_fmmm_*`
  reference positions explicitly overrides this:
  `scripts/ogdf_runner.cpp:324`: `layout.stopCriterion(ogdf::FMMMOptions::StopCriterion::FixedIterations)`.
  Both dagua's fidelity path (`max_iterations = max(100, 10*steps)`, always
  runs to completion, `pipelines/fmmm.py:1905`) and the reference binary run
  the full fixed-iteration budget, so no early-stop asymmetry exists in this
  dataset. (This WOULD be a real bug if anyone re-generated references using
  OGDF's true unmodified defaults instead of the runner's override -- worth a
  one-line comment in `ogdf_runner.cpp` flagging this intentional deviation
  from upstream defaults.)
- **Fine-tuning / postprocessing formula mismatch**: spot-checked
  `_ogdf_fmmm_postprocess_fidelity` / `_ogdf_fmmm_combined_forces` /
  `_ogdf_fmmm_adjust_positions` / `adapt_to_ideal_edge_length` /
  `pack_single_component` (MAARPacking Best-Fit) against
  `FMMMLayout.cpp:231-250,896-1070,1319-1345` and `MAARPacking.cpp` -- all
  formulas, constants (`fineTuningIterations=20`, `fineTuneScalar=0.2`,
  `postSpringStrength=2.0`, box-scaling `1.1`, `pageRatio=1.0`,
  10-step rotation search) match line-for-line. No divergence found here.
- **Adapter parameter mirroring** (`dagua/eval/competitors/ogdf_competitor.py`,
  `scripts/ogdf_runner.cpp`): confirmed `randSeed(seed)` and
  `fixedIterations(fmmmFixedIterations)` are both set identically on both
  sides for every `classic_fmmm_steps{10,100,200}` variant
  (`ogdf_runner.cpp:320-328`, `classic_competitor.py:1113-1119`). No parameter
  mismatch found. `classic_fmmm_graphviz_fdp_fidelity` correctly compares
  against `graphviz_fdp` (a deliberate cross-engine fidelity target, not an
  OGDF comparison bug) per `classic_competitor.py:371-372`.

## 4. Target combos not explained

I could not produce a root-cause-confirmed (as opposed to code-diff-confirmed)
explanation for any specific combo within the time budget -- Finding 1 and
Finding 2 are both CONFIRMED as real code differences from OGDF but NOT yet
confirmed as the specific cause via an isolated before/after position diff
(that experiment -- toggle one fix at a time and re-run the r74
`definitive_fidelity_analysis` scorer against the 25-combo single-level
sub-bucket -- is the natural next step and was out of the <45 min research
budget). Concretely unexplained/untested:

- Whether Finding 1's coincidence-jitter actually triggers on any of the 25
  small-graph combos (no instrumented run was executed).
- Whether Finding 2's angle-formula swap changes any bit of output at all
  (the cheapest test in this report, recommended to run first).
- `extreme_mixed_width_transformer` / `parallel_cycles_4x5` /
  `random_dag_50` under `classic_fmmm_graphviz_fdp_fidelity`: these compare
  against `graphviz_fdp`, a different reference engine and code path
  (`graphviz_fdp_fidelity` / `_layout_fmmm_fidelity_components`,
  `pipelines/fmmm.py:6755-6829`) that I did not investigate at all -- Findings
  1-3 apply only to the `ogdf_fmmm`-referenced combos (29/32 targets). The
  3 `graphviz_fdp_fidelity` combos need a separate pass against
  `_references/graphviz`'s `fdp` source, not OGDF's FMMM.
- `heavy_tail_weights_50`, `sparse_pair_50` (both exactly 50 nodes): sit
  exactly AT the `minGraphSize` boundary (`> 50` is the coarsening condition,
  so 50 nodes take the single-level path, confirmed), but their edge-weight
  handling (`heavy_tail_weights_50` presumably has non-uniform edge weights)
  was not separately checked against OGDF's `set_average_ideal_edgelength`
  weighted-length logic.
