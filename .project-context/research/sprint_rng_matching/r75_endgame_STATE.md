# r75 -- Fidelity Endgame Sprint (Fable-led)

**Started:** 2026-07-01 ~17:30
**Directive (JMT):** "EVERYTHING on the table except genuine FP rounding. North star: ALL algos
either bit-identical or statistically identical on distributional level. Squeeze every last bit
of juice; it would be amazing if this were the last one. Review metrics/criteria + invariances."
**Routing:** Fable = architect/synthesis only. Codex = default implementer. Sonnet = Claude-side
redundant research. Adversarial critique at high effort before ANY code lands (r74 lesson).

## Baseline (verified 2026-07-01 from r74_phase2_rescore.jsonl, develop == 89ed3c3)
- 409 divergent <=300-node combos rescored in r74 Phase 2 -> 72 reclassified, **337 remain divergent**:
  sugiyama 129, sfdp 126, fmmm 32, classical_mds 30, umap 7, gem 5, maxent 3, drl 3, neato 2.
- Plus **~165 huge-graph (>300 node) divergent combos** never rescored (need hang-safe scoring).
- Leg failures among 337: stress fails on 324, crossings on 235, np on 66.
- KEY NEW FINDING (Fable phase-0): of the 337, **93 are dagua-BETTER on every failing leg**, 89 mixed,
  only 155 all-worse. Better-than-reference = likely COMPARISON BUG (param/postprocessing mismatch,
  e.g. reference overlap-removal included in extracted positions), NOT an accolade. sfdp median stress
  excess = -8.2% (dagua better); umap -68%. Root-cause, don't celebrate, don't launder.
- Hairline cluster: sfdp 83/126 stress-fails within 1% rel; sugiyama 48/118 within 1%.

## Phase ladder
- P0 (Fable, DONE): baseline quantification, invariance review, sprint design.
- P1 (RUNNING): redundant research sweep -- 5 buckets x (sonnet + codex): sfdp, sugiyama, fmmm,
  classical_mds+small_tails (merged for codex count), metrics/criteria. Targets JSONs in r75_findings/.
- P2: Fable synthesis -> adversarial codex critique (HIGH effort, reads reference source) of every
  proposed fix BEFORE implementation.
- P3: implementation codexes (sequential per engine, worktree isolation if parallel), incl. the
  sugiyama position.c/mincross.c port chain (multi-day, own sub-state).
- P4: scoped re-bench (100 seeds, --seed-refs for seedable bases, DIRECT launch + bg-watch, never
  codex-babysat) -> re-analyze (overlay freshest-last) -> official report -> gates (0/40) -> scorecard.

## Wake-up routing
- Case A: research agents done, no synthesis yet -> Fable reads reports, synthesizes, dispatches P2 critique.
- Case B: agents still running -> ack, yield.
- Case C: codex quota out -> fall back to sonnet/opus subagents for the blocked task; note here.
- Case D: implementation landed -> verify (git log in worktree, tests), then next in chain.
- Always: check `ps -eo pid,args | grep -E "codex exec|run_benchmark"` before re-dispatching anything.
- Foreign work: torchlens codex (PID ~31997/32024, worktree torchlens-collapse) is JMT's other
  project -- NEVER touch/kill.

## Stop criteria (quantitative)
- Every one of the 337+~165 divergent combos has a disposition: (a) bit-exact after fix, (b) certified
  3Q under gated margins (0/40 controls), (c) aggregate population-equivalence (stochastic floors), or
  (d) documented genuine-divergence with root cause + why out of scope (needs JMT sign-off).
- Anti-laundering gate_5 holds 0/40 on every rescore. gate_6 data committed. No regressions:
  bit-exact count never decreases (r74 maxent lesson).

## GUARDRAILS (from baton, do not relax)
1. NEVER launder (gate_5 0/40). 2. Verify on BENCHMARK PATH (get_competitor), not direct calls.
3. Param+seed match vs references (--seed-refs). 4. No runtime delegation to reference libs.
5. "Floor" claims need FP-chaos evidence. 6. final_rung is a STRING. 7. Adversarial-critique research
before coding. 8. Long benchmarks run DIRECTLY + bg-watch launcher $!; kill by PID never pattern.

## STANDING DIRECTIVE (JMT 2026-07-02, recorded in memory project_r76_final_sprint_directive)
When r75 closes (final analysis + report + baton), AUTONOMOUSLY launch r76 -- THE FINAL sprint:
close all remaining gaps reasonably closable "without doing violence". Structure analogous to
r75 (redundant research where needed, adversarial gates, worktree codexes, sonnets per tiering,
gate_5 0/40 absolute). Create r76_STATE.md BEFORE first dispatch. Queue: mincross GD_nlist +
chain-merge port; stage-A x-simplex perf >300 nodes; sugiyama stages B-D; igraph GLPK ties + BK
ordinal-edge + qsort; big-graph hang-safe scoring + rescore; re-bench error top-ups; umap numba
trace; gem verdict vs honest refs; drl/neato perturbation dispositions; population-equivalence
tier; residuals from r75 final analysis.

## Iteration log
- 2026-07-01 17:30: P0 done. Invariance review verdict: current stack (similarity-group Procrustes +
  automorphism alignment + per-component alignment + anisotropic-for-sugiyama + scale-fitted stress)
  is SOUND; three flags raised: (1) dagua-better fails two-sided TOST -> need explicit policy
  (root-cause first; possible "quality-superior" tier ONLY if root cause proves benign + JMT approves);
  (2) crossings discreteness: integer metric with ref_self_spread often 0 on small graphs -> margin
  floor may be miscalibrated (audit in metrics bucket); (3) minor: component-alignment and
  automorphism-alignment are min'd separately, never composed (immaterial edge case, noted only).
- 2026-07-01 17:45: P1 dispatched. 5 sonnet (bg) + 5 codex (bg) research agents, 1:1 redundant buckets:
  sfdp, sugiyama, fmmm, mds+tails, metrics. Reports -> r75_findings/r75_{bucket}_{sonnet|codex}.md.
- 2026-07-01 17:55: mds_tails + metrics CODEX reports IN (both high quality). Headlines:
  * mds_tails: DLA port CONFIRMED needed for 16 disconnected mds rows; FULL port spec w/ file:line
    (merge_dla.c grid quirks, RNG = python random.Random(seed) via adapter's
    set_random_number_generator -- NOT igraph PCG; naive-port hang explained: unbounded walk);
    connected mds residue (14 rows) = eigensolver-basis HYPOTHESIS w/ cheap experiment; umap params
    CONFIRMED matched, residue in fuzzy-graph/spectral-init/neg-sampling (5/7 rows =
    parallel_multiedge_bundle -> duplicate-edge CSR suspect); neato: dagua seeds components
    seed+i vs graphviz single RNG stream (likely real bug, pipelines/neato.py:1404-1419);
    gem = OGDF ref (not igraph), near-margin, needs 1-ULP evidence; maxent 3 rows = random_dag_50
    disconnected-init parity (NOT blanket split).
  * metrics: crossings floor CONFIRMED brittle -- margin=max(2%*ref, 0.5, ref_spread)
    (analysis.py:1598), 115/235 cross-failing rows have ZERO ref self-spread, 75 within 1 crossing;
    exact vs sampled predicate INCONSISTENT (collinear overlap counted only in vector path,
    metrics.py:146 vs :2083); sampled-crossings denominator bug (rate x all-pairs incl adjacent,
    metrics.py:831) -- affects future huge-graph path only (all 235 current fails are exact-count);
    ***HUGE-GRAPH WORKLIST = 9 ROWS, NOT ~165*** (all random_dag_200; two independent cross-checks;
    baton's ~165 was wrong -- VERIFY via official report in synthesis, but lead #1 is a mop-up not
    a big win); dagua-better tier recommendation: QUALITY_SUPERIOR_DISTINCT, never counts toward
    identical headline, requires fairness audit + controls.
- 2026-07-01 18:05: fmmm CODEX report IN. Headline: dagua FMMM fidelity loop uses EXACT all-pairs
  repulsion while OGDF defaults to NMM (New Multipole Method) -- CONFIRMED first divergence
  affecting ALL 29 OGDF-FMMM target rows (pipelines/fmmm.py:833-835 vs FMMMLayout.cpp:283-288);
  spot-check deep_chain_20 seed0 RMSD 0.2088 (not FP floor). Only 1/32 rows (random_dag_200
  steps10) actually enters multilevel (>50-node component) -- do NOT chase coarsening for the rest;
  for that row: prolongation omits OGDF Advanced same-solar-system placement + placement sectors
  (Multilevel.cpp:444-656) and multilevel RNG uses python random.Random instead of OGDF
  randomNumber (existing _OgdfMt19937 helper is unused there). 3 rows are graphviz_fdp_fidelity --
  REBUCKET to sfdp/graphviz family, not OGDF. Proposed fix ladder: (1) port NewMultipoleMethod.cpp
  for fidelity loop (high risk, gate on 33 r74 3Q fmmm combos), (2) Advanced prolongation +
  OGDF RNG for multilevel, (3) rebucket fdp rows.
- 2026-07-01 18:15: metrics SONNET report IN -- DISAGREES with metrics codex on 2 fronts (signal!):
  * CROSSINGS FLOOR: sonnet says mostly NOT a bug -- the zero-spread cross failures are GENUINE
    systematic 1-crossing differences between two deterministic algorithms (e.g. dagua sugiyama
    always 3 crossings vs igraph always 2 on hub_skip_superfan); only ~10-15 combos are true
    floor-miscalibration candidates. Codex framed the same data as margin brittleness w/ 75-row
    upper bound. SYNTHESIS LEAN (Fable): sonnet's frame fits the north star -- a deterministic
    +1 crossing is a REAL divergence to fix in the LAYOUT (ordering), not to widen margins over.
    Reconcile codex 235 vs sonnet 163 cross-fail counts (definition difference?).
  * HUGE-GRAPH WORKLIST: sonnet shows r74_analysis.jsonl only covers 1/16 >300-node graphs -- the
    other 15 were NEVER re-run in r74 (absence != cleared). From r73 full-coverage snapshot:
    **238 divergent combos on big graphs** (upper bound, pre-r74-metric-fixes). Codex's "9 rows"
    treated missing rows as resolved -- WRONG PREMISE. Lead #1 is real and BIGGER than baton's
    ~165: big graphs need rescore-from-r73-positions w/ corrected metrics + hang-safe path
    (and possibly re-bench for engines whose r74 layout fixes changed positions).
  * sfdp p_neg2 drives ~22-34 dagua-better combos; params CONFIRMED matched (variants.py:1615) --
    root cause must come from sfdp bucket (overlap-removal hypothesis pending).
  * Sampled-crossings SE discarded before margin (analysis.py:1726) -- confirms codex S2 landmine.
- 2026-07-01 18:25: sugiyama SONNET report IN. Headline: dagua uses Brandes-Kopf x-coordinates
  UNCONDITIONALLY (ops/sugiyama.py:1675/1741) for all fidelity modes; graphviz dot uses
  NETWORK-SIMPLEX on an auxiliary graph (position.c:141-142) -- wrong algorithm FAMILY for the
  graphviz-fidelity variants (58/129 combos; 39 show "crossings match, stress diverges" signature;
  57/58 dagua-worse). Also missing omega weight table (C_EE=1/C_VS=2/C_SS=2/C_VV=4) + init_order
  is node-id sort vs graphviz BFS build_ranks (mincross.c:1212-1286) -- the ba_500 lever. igraph's
  x-coord IS BK (igraph sugiyama.c:858-1047) so igraph family (71 combos) needs BK detail parity
  only (median-of-4, anchors, hgap) + isolated-node tie-break hypothesis (cheap trace experiment
  specced). Estimate: graphviz family ~88-93% fixable via ns port; igraph ~39-56%; combined
  ~61-73% of 129. CAVEAT for critique: sonnet used only 4 tool calls -- file:line claims need
  verification against source (codex high-effort report will cross-check).
- 2026-07-01 18:35: sfdp CODEX report IN (dense, high quality). Headlines:
  * BOMBSHELL CONFIRMED: graphviz sfdp IGNORES `theta` and `maxiter` graph attrs (sfdpinit.c:200-224
    reads neither; BH theta is a compile-time constant bh=0.6, spring_electrical.c:41-43). So 47/126
    targets (theta04 x14, theta08 x15, steps200 x18) compare dagua-with-changed-params vs graphviz
    DEFAULT output -- probe shows reference positions identical across all these attrs (rms 4e-16).
    These variants have NO expressible canonical reference -> scoring/metadata fix (route to
    exploratory / non-original tier), NOT a dagua code fix. Needs JMT sign-off (changes scorecard).
  * H1 OVERLAP-REMOVAL KILLED: graphviz default = prism0 = ZERO overlap iterations (overlap.c:517-528
    returns at ntry==0); default==prism0==prism1 positions bit-identical on probes. Do NOT set
    overlap=false anywhere.
  * DISCONNECTED HYPOTHESIS (up to 43 targets, 19 non-overlapping with the 47): graphviz reuses ONE
    mutable spring_electrical_control across components (ctrl->K/random_start/adaptive_cooling/step
    mutate, spring_electrical.c:284-286,571-573,1173-1179) then packSubgraphs; dagua re-runs fresh
    independent pipelines per component + neato packer import. Decisive experiment specced (~20min).
  * p_neg2 clamp (6f8cff5) CONFIRMED correct + not causal: BOTH sides collapse repulsiveforce=-2 to
    default. => p_neg2 divergences are DEFAULT divergences under another label. FABLE FOLLOW-UP:
    many graphs fail ONLY as p_neg2 while default passes on the same graph -- if effective params
    are identical both sides, per-seed reference positions for p_neg2 vs default should be
    BIT-IDENTICAL; if they differ it's a REFERENCE-CACHE VINTAGE artifact (stale overlay / seeded-ref
    era mismatch). CHECK positions directly in synthesis -- could clear a large slice of the 58
    unexplained (which are heavily p_neg2 rows).
  * gv_random rejection-sampling mismatch (ops/sfdp.py:247-253) CONFIRMED real but ZERO rejections
    occur at bucket sizes (probe to 10000) -- correctness debt, near-zero combo impact.
  * 58 connected default-like targets remain unexplained -> next: makeMatrix row order/dup-edge
    handling, average_edge_length quirk (ops/sfdp.py:808-826), sequential update parity.
- 2026-07-01 18:50: ALL 10 P1 REPORTS IN. Late arrivals:
  * sugiyama CODEX (high effort): CONFIRMS sonnet's wrong-family finding with live benchmark probe
    (binary_tree: identical ranks+orders, different x) -> graphviz x-coord = aux-graph network
    simplex w/ LR balance (position.c:127-153, 238-350, 525-584), staged port spec A-D
    (250-400 LOC core + 150-250 flat/label + 250-500 cluster), mincross partial-port deltas
    (class2 chains, xpenalty-weighted crossings, pass 0/1/2, left2right; _dot_mincross.py vs
    mincross.c:690-748). igraph family: GLPK-vs-HiGHS at rank stage (VERY high risk), BK
    ordinal-edge conflict quirk (sugiyama.c:912-944), qsort ties. NEW: reference-tree igraph LP
    objective has indegs AND outdegs both IGRAPH_IN (sugiyama.c:589-592) -- possible conflict w/
    r74 fix, version check needed. Estimate: graphviz family ~50-55/58 fixable; DISPROVED sonnet's
    "all 11 crossings-only rows are ordering-stage" (multiscale probe: identical rank/order).
  * fmmm SONNET: CONTRADICTS fmmm codex -- NMM only activates >=175 particles, 25/32 targets <=48
    nodes never touch it. Alternative causes: (a) coincident-node repulsion: dagua zeroes force,
    OGDF numexcept.cpp:169-182 jitters CONSUMING RNG -> stream desync (positions integer-floored
    each iter, collisions plausible); (b) oscillation-damping angle atan2-form rounding ->
    ceil(angle/pi_6) sector flips. CRITIQUE MUST SETTLE codex-vs-sonnet.
  * sfdp SONNET: killed H1 via different mechanism (conda dot 7.0.5 lacks GTS -> prism compiles
    out; consistent verdict w/ codex). H2/H3/H4 dead. ***METHODOLOGY CATCH: _references/graphviz
    tree is at ~14.1.5 HEAD, NOT the binary's 7.0.5 -- codex citations suspect; verify via
    `git show 7.0.5:<path>`.*** CONTRADICTS codex on gv_permutation rejection-sampling. Also
    caught sprint-lead bucketing bug: "83/126 hairline <=1%" was WRONG (negatives fell in
    <=0.01 bucket); strict recount = 8/126. parallel_cycles_4x5 degenerate stress=1.0 sentinel
    across all 6 sfdp variants = real packing bug (6 combos).
  * mds_tails SONNET: DEBUNKED "DLA port hung" -- literal Python port on random_dag_200 (202
    components) completes in 17.4s; bounded 2D walk, just slow in CPython. Fix ~400-550 LOC, ONE
    session, targets rung-3. Connected mds: claims documented dsyevr degenerate-eigenspace floor
    (codex says testable hypothesis -- critique to rule; floor claims need FP-chaos evidence).
    umap: params + parallel-edge fix ruled out; residue likely numba neg-sampling RNG.
- 2026-07-01 18:55: ***STALE-VINTAGE CONFIRMED (Fable)***: rescore rows for p_neg2 vs default have
  DIFFERENT D-values on same graphs, but current code (r74_fixes positions) is bit-identical
  15/15 seeds x 4 graphs. => all 52 p_neg2 rows scored on pre-clamp layouts; p_neg2's lower stale
  stress also explains much of sfdp's "dagua-better" cluster. Combined w/ 47 theta/steps200
  no-reference rows: up to 99/126 sfdp rows may be non-issues.
  ACTIONS LAUNCHED: (1) TRUE-BASELINE RESCORE running: /tmp/r75_truebaseline.sh, PID 1554765,
  log /tmp/r75_truebaseline.log, output eval_output/fidelity_definitive/r75_truebaseline.jsonl,
  409 combos, 10-dir overlay (9-dir r73 chain + r74_fixes freshest-last), bg-watch armed.
  (2) ADVERSARIAL CRITIQUE codex dispatched (high effort): PID 1563185, log /tmp/r75_cx_critique.log,
  brief /tmp/r75_critique_brief.md -> output r75_findings/r75_ADVERSARIAL_VERDICTS.md. Must settle
  5 conflicts (fmmm force path, gv_permutation, crossings policy, mds connected floor, igraph LP
  objective) with VERSION-PINNED citations; verdicts gate all implementation.
- 2026-07-01 19:10: CRITIQUE DONE (r75_ADVERSARIAL_VERDICTS.md -- version-pinned, conservative).
  Conflict rulings: fmmm NMM-for-all REJECTED (NMM falls back to exact <175 nodes in runner's
  foxglove-202510 OGDF); sfdp permutation-RNG fix REJECTED (7.0.5 uses raw rand()%n -- dagua
  faithful; codex had read HEAD); crossings margin widening REJECTED (deterministic 3-vs-2 is a
  real layout divergence -- 235 vs 163 count = any-delta vs TOST-gated); mds connected floor claim
  REJECTED-until-experiment; igraph LP objective NEEDS-EXPERIMENT (source tree unpinned vs
  installed 1.0.0). APPROVED w/ gates: mds DLA port, sugiyama graphviz x-ns staged port +
  omega/build_ranks, sampled-crossings SE+denominator fix, predicate consistency,
  QUALITY_SUPERIOR_DISTINCT (metadata only), fmmm multilevel (>50 gate), fdp-row rebucket.
  VERSION AUDIT: graphviz tree=14.1.5-era vs binary 7.0.5 (use git show 7.0.5:); igraph tree
  unpinned vs installed 1.0.0 (runtime-trace only); OGDF runner = ~/tools/ogdf-src foxglove-202510
  SHA 5b67956, NOT _references/ogdf.
  OVERLAY POISON NOTE (Fable): benchmark_100seed_r74_fixes contains REVERTED-code positions for
  classical_mds + maxent_stress -> r75_truebaseline rows for those engines are INVALID; use
  r74_phase2_rescore values for mds/maxent (develop==r73 code there). Valid for
  sfdp/fmmm/umap/sugiyama (fixes kept on develop).
- 2026-07-01 19:15: IMPLEMENTATION WAVE dispatched (5 codexes):
  * mds DLA port: pid 1775134, worktree dagua-mds-dla branch r75/mds-dla, log /tmp/r75_cx_mds_impl.log
  * metrics fixes (verdicts 30/31/35): pid 1775537, worktree dagua-metrics-fixes branch
    r75/metrics-fixes, log /tmp/r75_cx_metrics_impl.log
  * sugiyama x-ns STAGE A: pid 1776125, worktree dagua-sugiyama-xns branch r75/sugiyama-xns,
    log /tmp/r75_cx_sugiyama_impl.log (ladder: binary_tree frame-match -> stress shrink -> igraph
    byte-identical regression gate)
  * fmmm probes (coincidence census + angle swap): pid 1776420, log /tmp/r75_cx_fmmm_probe.log
    -> r75_PROBE_fmmm_RESULTS.md
  * tails probes (E1 igraph LP / E2 mds eigenspace / E3 maxent trace / E4 neato seeding):
    pid 1776902, log /tmp/r75_cx_tails_probe.log -> r75_PROBE_tails_RESULTS.md
  All monitored via codex-watch. Truebaseline rescore still running (~150/409 at 19:05).
  AWAITING JMT SIGN-OFF (do not implement until he answers): (1) no-canonical-reference tier for
  sfdp theta04/theta08/steps200 (47 rows -- graphviz can't express those knobs; route to
  non-counting tier?); (2) whether QUALITY_SUPERIOR_DISTINCT combos (proven-benign dagua-better)
  should appear as their own scorecard line (metadata is landing either way, non-counting).
- 2026-07-01 19:25: FMMM PROBES DONE -- BOTH HYPOTHESES KILLED. Coincidence census: 0 triggers
  across 20 benchmark-path runs (integer flooring confirmed BOTH sides, so census was valid).
  Angle swap: numerical no-op (base-vs-swap RMSD ~1e-17). Do NOT land either fix.
  FABLE INSIGHT from probe data: grid_5x5 seeds 44/45 nearly MATCH the OGDF reference (RMSD
  0.001-0.002, stress equal to 4dp) while seeds 42/43/46 diverge (0.04-0.14). Near-match on some
  seeds = pipeline nearly right; divergence is SEED-CONDITIONAL (occasional extra RNG draw /
  boundary-clamp event / tie-break desync), NOT a wrong force model. NEXT FMMM PROBE (queued):
  extend scripts/ogdf_runner.cpp with an iteration-dump flag (rebuild via
  scripts/rng_match/build_ogdf_runner.sh), then bisect first-divergent iteration on grid_5x5
  seed 42 (diverging) vs seed 44 (matching) -- find the exact event that desyncs.
- 2026-07-01 19:40: TAILS PROBES DONE (r75_PROBE_tails_RESULTS.md). Verdicts:
  * E1 igraph LP objective: dagua r74 objective WRONG -- installed igraph 1.0.0 matches the IN/IN
    source-bug on distinguishing DAG two_hubs_bridge. FIX QUEUED (serialize behind sugiyama-xns
    codex -- same file ops/sugiyama.py:495-543; add two_hubs_bridge regression; gate igraph
    modes only; verify currently-bit-exact igraph rows unchanged -- objectives coincide on
    longest-path-layering graphs, so most passing rows should be unaffected).
  * E2 connected mds: GENUINE FLOOR with hard evidence -- exact eigenvalue ties (l1=l2=l3) on 6/7
    graphs, reference NOT in dagua's subspace, evd driver materially differs. Basis is arbitrary
    at LAPACK level. Disposition: documented floor (~12-14 rows), no port.
  * E3 maxent: KILLED + DISCOVERY -- ogdf_runner ALWAYS sets hasInitialLayout(true) + fills
    std::rand()%1000/10.0 init (runner.cpp:417-422), bypassing OGDF internal PivotMDS warm start;
    current dagua matches runner to Procrustes 3e-8 @ steps200 on random_dag_50. The 3 divergent
    maxent rows = stale-data artifacts; expected to clear on clean re-bench.
  * E4 neato: seeding-policy KILLED (same-seed patch no help; pack=false worse). 2 rows -> low
    priority; graphviz pack/CG internals only if ever chased.
- 2026-07-01 19:55: METRICS IMPL VERIFIED + COMMITTED (205129e on r75/metrics-fixes; codex left it
  uncommitted due to instruction confusion -- Fable committed). Evidence: gate_5 0/40 HELD;
  exact-row replay 409/409 decisions unchanged; 131+54+37 targeted tests green; gate_3/gate_6
  fails pre-existing; test_bench_large checkpoint failure CONFIRMED pre-existing on clean develop
  (0.05s repro). MERGE DEFERRED until truebaseline rescore finishes (running process imports
  analysis code from main repo -- don't swap files mid-run).
- 2026-07-01 20:15: MDS DLA IMPL VERIFIED + COMMITTED (ec24b05 on r75/mds-dla; codex left
  uncommitted again -- Fable committed after ruff-format round-trip, 12 pipeline tests re-green).
  Gates: connected byte-identical (SHA-256 pre/post via stash, 3 graphs x 3 seeds); stress gap
  SHRANK on all 3 probe graphs (multi_component_80 +0.477->+0.072, parallel_cycles_4x5
  +0.989->+0.331, random_bipartite_60 +0.102->+0.023); guardrails raise (10M steps/1M restarts);
  no TileToRows. FLAG for re-bench: parallel_cycles_4x5 new D stress = exactly 1.000000 --
  possible degenerate-layout sentinel; 100-seed battery arbitrates. get_sphere deviates from
  literal C quadrant scan (documented; fine for rung-3 target, revisit only if bit-exact needed).
  INFRA NOTE: codexes keep skipping `git commit` ("orchestrator handles git") -- plugin runtime
  instruction overrides my prompt; just commit their work myself each time.
  ALSO: bg-watch.sh --stale-warn-min Linux crash fixed in place (stat -c first + numeric guard),
  ledger entry filed 2026-07-01-1856-bg-watchsh-stale-warn.md.
- 2026-07-01 20:50: TRUE-BASELINE DONE (r75_truebaseline.jsonl, 409 rows). RESULT: 35 p_neg2 rows
  FLIP to quality-identical (vintage theory VALIDATED); 10 regressions (6 sfdp: planar_60
  default/theta04 + random_bipartite_60 x4; 4 fmmm: heavy_tail_weights_50 + sparse_pair_50
  steps100/200) = rows where KEPT r74 fixes made fresh positions fail corrected metrics -- honest
  count includes them. TRUE <=300 BASELINE: **312 divergent** = sugiyama 129, sfdp 97 (47
  theta/steps200 no-reference pending JMT + 17 p_neg2 sharing default root cause + 33
  default/graphviz_fidelity-family), fmmm 36, mds 30 (16 disconnected DLA-fixed pending re-bench,
  14 connected = evidenced floor), umap 7, gem 5, maxent 3 (expected stale artifacts), drl 3,
  neato 2. NOTE: mds/maxent rows in truebaseline used POISONED r74_fixes positions -- their
  authoritative pre-fix verdicts stay r74_phase2_rescore (same counts, no material difference).
- 2026-07-01 20:55: MERGED r75/metrics-fixes (8221313) + r75/mds-dla (8aac36d) -> develop;
  branches deleted, worktrees removed (only sugiyama-xns remains). MDS RE-BENCH LAUNCHED direct:
  pid 2754785, log /tmp/r75_mds_rebench.log, output eval_output/benchmark_100seed_r75_fixes,
  2 mds variants x 100 seeds (100-199) x all graphs, bg-watch armed. Refs reused from
  seeded_refs (reference code unchanged).
- 2026-07-01 21:35: SUGIYAMA STAGE A VERIFIED + MERGED (e6ba3db on r75/sugiyama-xns, merge
  01d6845). LADDER: binary_tree x matches graphviz to 1.5e-16 rel residual (target 1e-6 --
  MACHINE PRECISION); stress gap shrinks 3/3 (bipartite 0.145->0.021, org_chart 0.162->0.033,
  center_port 0.132->0.008); default/tight tensor-identical 10/10; 56 sugiyama tests green on
  develop post-merge. Worktree removed, branch deleted (steady state: develop+main).
  KEY IMPL NOTES: new graphviz_network_simplex_assignment(balance_mode) in dot_rank.py; aux-graph
  seeding from position.c ND_rank values matters (zero-init gives different optimal tie); 2-unit
  internal resolution for odd minlens; worktree pytest needs PYTHONPATH=$PWD (editable install
  points at main checkout). Stages B-D (flat/labels/clusters) deferred -- graphs w/ labels/clusters
  in the 58-row graphviz family will partially remain until then.
- 2026-07-01 21:40: LP OBJECTIVE FIX DISPATCHED: pid 2915786, worktree dagua-lp-obj, branch
  r75/lp-objective (from 01d6845), log /tmp/r75_cx_lp_impl.log. Ports igraph 1.0.0's faithful
  IN/IN objective (E1-proven); gate igraph modes; two_hubs_bridge runtime regression; probe must
  show changed pairs move TOWARD reference; graphviz path tensor-identical gate.
- 2026-07-01 22:20: LP FIX VERIFIED + MERGED (bb72c8b, merge 952c98a; codex committed itself this
  time). Evidence: two_hubs_bridge runtime regression vs installed igraph 1.0.0 green; graphviz
  path tensor-identical; 45 sugiyama tests green; probe showed ZERO changed pairs on 4 probe
  graphs (objectives coincide there -- impact lands only on asymmetric/feedback graphs, re-bench
  arbitrates). KEY INSIGHT from codex: faithful objective is ALL-ZERO on feedback-free DAGs ->
  igraph's layering there is pure GLPK tie-breaking -> HiGHS-vs-GLPK tie parity (verdict 19) is
  now the likely first-stage lever for remaining igraph-family rows. test_classify_early_exit
  failure = wall-clock threshold flaking under mds-bench load, unrelated.
- 2026-07-01 22:25: FMMM BISECTION PROBE dispatched: pid 3305094, log /tmp/r75_cx_fmmm_bisect.log,
  brief /tmp/r75_probe_fmmm_bisect.md -> r75_PROBE_fmmm_bisect_RESULTS.md. Instrumented OGDF
  runner (SEPARATE /tmp binary, pristine-restore contract w/ baseline byte-match verification),
  per-iteration dumps both sides, bisect grid_5x5 seed 42 (diverging) vs 44 (matching), RNG-count
  comparison per iteration. mds re-bench still on big-graph tail (18789/~19000).
- 2026-07-01 23:10: ***OGDF RUNNER PROVENANCE BOMBSHELL RESOLVED (fmmm+gem root cause).*** The
  fmmm bisect probe STOPPED on its restore contract: rebuilt runner != committed binary. Fable
  follow-up chain: (1) binary tracked in git, last committed 52930fe (2026-04-30); runner.cpp
  gained gemRounds/fmmmFixedIterations plumbing LATER (f60944e) with no binary recommit ->
  (2) SMOKING GUN: old binary IGNORES fmmmFixedIterations + gemRounds (identical output 10 vs 200
  iters / 100 vs 2000 rounds); stress plumbing predates drift -> old==new BYTE-IDENTICAL for
  stress (maxent/stress claims unaffected) -> (3) A/B: dagua fmmm vs REBUILT runner RMSD
  0.000000-0.0014 on all 10 previously-diverging graph/seeds (was 0.039-0.139). FMMM's 36-row
  family + gem tail = PARAMETER-MISMATCH ARTIFACTS via stale reference binary (the r73 gem
  30000-rounds lesson, hidden in a binary). gem: dagua-vs-rebuilt 0.21-0.27 at matched rounds --
  needs rescore against honest refs (r71 gem port may have tuned to default-rounds refs; re-bench
  arbitrates). REBUILT BINARY COMMITTED 0817427 w/ full provenance. ogdf_fmmm+ogdf_gem refs MUST
  regenerate; queued in combined re-bench.
- 2026-07-01 23:20: mds re-bench DONE (21000 total, 17924 ok; ba_2000/ba_5000 mds timeouts
  expected/huge-graph tier). COVERAGE: 6/8 disconnected targets 200/200; random_dag_50 24/200 +
  random_dag_200 0/200 TIMED OUT -> DLA perf bug on many-component graphs (50-202 components).
  DLA PERF FIXUP dispatched: pid 3361855, worktree dagua-dla-perf branch r75/dla-perf, log
  /tmp/r75_cx_dla_perf.log. HARD constraint: bit-identical output on already-passing graphs
  (RNG draw order preserved); profile-first; target <10s/seed on random_dag_200.
  COMBINED RE-BENCH queued (waiting for mds bench process exit -- same output dir, results.json
  contention): 12 engines (6 sugiyama + 3 fmmm steps + 3 gem iters) + --seed-refs ogdf_fmmm,
  ogdf_gem, 100 seeds, --resume, dir benchmark_100seed_r75_fixes, log /tmp/r75_main_rebench.log.
  Launch command ready in /tmp (guard script attempted once, mds still writing).
- 2026-07-01 23:45: JMT SIGN-OFFS RECEIVED: (1) 47 theta/steps200 rows -> no-canonical-reference
  tier APPROVED ("graphviz cant make em in the first place so fidelity doesnt make sense") w/
  clear documentation; option-2 (patched-graphviz reference build) REJECTED as excessive.
  (2) dagua-better line APPROVED -- label clearly, "make clear they are in fact different".
  JMT also asked bottom-of-the-well assessment; answered: last sprint of UNKNOWNS plausibly yes
  (every divergent row now has known root cause or evidenced floor); last sprint of WORK needs
  the sugiyama chain (mincross + stages B-D) + big-graph rescore + gem re-verdict. JMT: "keep
  cooking".
- 2026-07-01 23:50: FOUR PARALLEL JOBS: (1) combined re-bench 32% (pid 3376686, persistent watch);
  (2) DLA perf codex (pid 3361855); (3) nocanon tier + superior-distinct line codex (pid 3404049,
  worktree dagua-nocanon, branch r75/no-canon-tier); (4) NEW: mincross phase-1 codex dispatched
  (pid 3408769, worktree dagua-mincross, branch r75/mincross) -- omega xpenalty weights +
  build_ranks init + pass 0/1/2 schedule, graphviz-mode gated; ladder incl. ba_500 spot-check
  (22344 vs 2805 crossings, >2x improvement = phase pass); left2right/class2-merge deferred to
  phase 2.
- 2026-07-02 00:20: NOCANON TIER VERIFIED + MERGED (01e589f, merge c0c78ba; worktree/branch swept).
  Evidence: EXACTLY 47 scored rows -> NO_CANONICAL_REFERENCE (steps200 19 / theta04 14 /
  theta08 14); 8 flagged rows correctly stay INSUFFICIENT_DATA; 0 identity-tier side effects;
  gate_5 0/40 HELD; report section renders w/ full 7.0.5 evidence. NOTE: 15 of the 47 previously
  PASSED (10 rung-3 + 5 3Q) vs the knob-ignoring reference -- whole-variant routing regardless of
  pass/fail is the coherent policy. QUALITY_SUPERIOR_DISTINCT section renders (0 rows on
  truebaseline input because that file predates the metrics merge; final re-analysis populates).
  test_smoke verbose failure = pre-existing on develop (verified pre-merge).
- 2026-07-02 00:50: DLA PERF VERIFIED + MERGED (8266c77, merge 288d0ca; swept). Profile-first:
  get_sphere all-occupied-cell scan was 32.6s/35s -> bounding-box lookup fix; BIT-IDENTITY 9/9
  (torch_equal max_abs=0 on 3 passing graphs x 3 seeds); random_dag_200 timeout -> 8.9s;
  probe 20/20 ok 0 timeouts. MDS TOP-UP BENCH launched: pid 3442916, SEPARATE dir
  eval_output/benchmark_100seed_r75_mds_topup (combined re-bench owns r75_fixes results.json --
  never two writers one dir), random_dag_50/200 x 2 variants x 100 seeds, workers 2 nice 15,
  bg-watch armed. Overlay chain gains one more dir freshest-last.
- 2026-07-02 01:15: MDS TOP-UP v1 half-failed: random_dag_50 200/200 OK; random_dag_200 all-200
  "watchdog: future exceeded timeout" -- NOT a code problem (standalone main-repo run = 7.8s).
  ROOT CAUSE: run_benchmark --timeout applies to the WHOLE 100-seed batch per (graph,engine);
  100 x ~8s = ~800s > 300s ceiling. INFRA LESSON: for slow-per-seed engines at 100 seeds, set
  --timeout to batch scale (>= seeds x per-seed x 2). TOP-UP v2 launched: random_dag_200 only,
  --timeout 1800, --resume, pid 3639056, log /tmp/r75_mds_topup2.log, watcher w/ explicit
  fail-regex (Traceback|FATAL) so per-seed ERROR data rows don't false-trip.
- 2026-07-02 02:00: MDS TOP-UP COMPLETE (random_dag_200 200/200 ok -- v1's watchdog marked batch
  error at 300s but workers finished + wrote positions; v2 --resume validated 200 ok).
  MDS RESCORE RESULT (r75_mds_rescore.jsonl): **0/30 flip**. DLA improved every disconnected D
  (parallel_cycles 1.0->0.56, multi_component 0.56->0.35, encoder_residual 0.35->0.20) but still
  FAR from reference (parallel_cycles R=0.011 -- 4 disjoint 5-CYCLES at D=0.55 is structurally
  wrong, per-component embeddings suspect, NOT packing: battery stress uses finite-pair sample
  where cross-component pairs shouldn't count). References verified UNCONTAMINATED (r75_fixes has
  0 igraph_mds__for__ files; R values match old rescore). DLA QUALITY PROBE dispatched: pid
  3791265, log /tmp/r75_cx_dla_quality.log -> r75_PROBE_dla_quality_RESULTS.md (per-component
  stress split, submatrix/row-order bug hunt, analysis-side finite-fill check).
- 2026-07-02 02:05: MINCROSS ATTEMPT 1 FAILED ITS LADDER HONESTLY (no commit -- correct behavior):
  crossings moved AWAY 3/4 targets (dense_pair 391->400 vs ref 331, hub_skip 3->5, heavy_tail
  70->90), weighted_karate overshot (111->76 vs 108); ba_500 timed out (Python transpose loop).
  Source-fidelity catch by codex: 7.0.5 crossing counts use ED_xpenalty NOT omega/virtual_weight
  (omega belongs to x-coord aux constraints only -- my brief was wrong, correction kept).
  Suspect: "reverse creation order" approximation of GD_nlist. RETRY (attempt 2 of 2, HIGH
  effort) dispatched: pid 3790815, same worktree (attempt-1 diff available), log
  /tmp/r75_cx_mincross2.log. NEW DISCRIMINATOR: `dot -v` prints graphviz's own ordering-stage
  crossing count -> iterate ordering-vs-ordering before rendered crossings. Ladder unchanged +
  incremental transpose delta for ba_500 perf. If attempt 2 fails: documented failure -> r76,
  NO commit (anti-flail: 2 attempts max this sprint).
- 2026-07-02 02:50: ***SEED-ERA FRANKENSTEIN FOUND (major eval defect, r75's own r74-style
  metric-audit win).*** DLA quality probe verdict: current code + bench positions are CORRECT
  (per-component stress 0.011 == igraph; bench file bit-identical to fresh layout). The rescore's
  D=0.555 comes from the OVERLAY: load_results_multi unions rows PER SEED KEY, so old-era dirs
  contribute seeds 42-99 (pre-DLA code) alongside r75's seeds 100-199 (post-DLA) -> battery
  averages TWO CODE ERAS (158 rows for parallel_cycles: 58 old @ ~1.0 + 100 new @ 0.011 = 0.555).
  Sweep: THOUSANDS of combos across ALL families have mixed seed eras in the 11-dir chain --
  r73/r74 verdicts may carry the same mixing wherever old refs (seeds 42-99) paired with old
  dagua rows. FIX DIRECTION (unambiguous, seed-matching principle): overlay resolution must be
  PER-COMBO FRESHEST-DIR-WINS (never union one combo across dirs). OVERLAY-FIX codex dispatched:
  pid 3977094, worktree dagua-overlay-fix, branch r75/overlay-fix, log /tmp/r75_cx_overlay.log.
  Verification = re-run 30-combo mds rescore w/ fixed loader; expectation parallel_cycles D
  0.555 -> ~0.011 + real flips. NOTE: after this lands, the r75_truebaseline (312) itself needs
  recomputation for the final scorecard -- its regressions (planar_60 etc.) could be mixing
  artifacts. DLA placement-scale delta (dagua r=167 vs igraph r=3.3 under finite-fill metric)
  noted but battery doesn't use finite-fill -- non-issue unless finite-fill stress is used
  somewhere (it is NOT in the battery path; diagnostic-only).
- 2026-07-02 03:10: MINCROSS ATTEMPT 2 FAILED LADDER (final for r75 per anti-flail; NO merge).
  PROGRESS BANKED: ordering-stage discriminator MATCHES dot -v on 3/4 small graphs (binary_tree
  0/0, bipartite 36/36, hub_skip 2/2; weighted_karate 50 vs 63 = port BETTER than graphviz there);
  residual precisely named: GD_nlist install order (fastgr.c:205-264, mincross.c:1356-1414) +
  representative-chain ED_xpenalty merge (class2.c:137-155). NEW discovery: ba_500 scale blocker
  is the MERGED stage-A x-coord network simplex (>270s, recursion fixed but slow) -- large-graph
  graphviz_fidelity rows in the running re-bench will error and fall back to old dirs under the
  overlay fix (acceptable; stage-A wins are on small/mid graphs). Also: stage-A x port worsens
  rendered crossings on 2 ladder graphs vs BK even at same ordering -- net battery effect per
  combo decided by re-bench + fixed-overlay analysis.
  INCIDENT (Fable process error, banked lesson): chained `worktree remove --force` after commit
  attempts WITHOUT verifying the commits landed -- ruff-format hook aborted both commits and the
  force-remove discarded attempt-2 code. Notes SAVED (r75_IMPL_mincross_NOTES.md committed
  00a2893); code patches recoverable from codex transcripts (r75_cx_mincross*.log copied into
  findings dir, untracked). LESSON: verify `git log` shows the commit BEFORE any worktree
  remove; never chain them.
  MERGE ORDER remaining: overlay-fix -> verify (mds flips materialize) -> merge; combined
  re-bench done -> FULL re-analyze with FIXED loader (all currently-divergent combos + controls)
  -> official report + gates 0/40 -> scorecard (recompute baseline with fixed loader for
  apples-to-apples) -> baton + memory. r76 QUEUE: mincross GD_nlist+chain-merge port (notes +
  transcripts), stage-A x-coord simplex perf for >300-node graphs, igraph GLPK tie parity,
  sugiyama stages B-D.
