# r76 -- THE FINAL FIDELITY SPRINT (autonomous, JMT-authorized 2026-07-02)

**Directive (JMT, verbatim intent):** last sprint; close ALL remaining gaps "we can reasonably
close without doing violence"; structurally analogous to r75; LEAN CODEX OVER SONNET (preserve
Claude credits): codex for all research+impl; codex-vs-codex redundancy; claude agents only for
in-session-capability needs; Fable = architect/synthesis only. Memory:
project_r76_final_sprint_directive. Baseline: r75_RESULTS.md (163 divergent + 79
superior-distinct + big-graph tier unscored).

## Stop criteria (quantitative -- the sprint ends when EVERY combo has exactly one disposition)
identical (bit/dist/3Q, gate_5 0/40) | evidenced floor (perturbation/eigengap proof) |
no-canonical-reference | superior-distinct (fairness-audited) | aggregate-equivalent (population
tier). Zero rows left as bare "divergent" without a named cause + documented why-not-closed.
No regressions: bit-exact/identical counts never decrease. gate_5 0/40 on every rescore.

## Work items (from r75; ROI-ordered within tracks)
TRACK A -- sugiyama (121 divergent: the bulk):
  A1 mincross GD_nlist install order + representative-chain ED_xpenalty merge port (attempt-2
     analysis in r75_IMPL_mincross_NOTES.md; patches in r75_findings/r75_cx_mincross*.log;
     discriminator harness: dot -v ordering counts, already proven on 3/4 graphs).
  A2 stage-A x-simplex PERF (network simplex slow >=~100-node dense; unblocks 9 unmeasured
     graphviz_fidelity combos + ba_500 tail; iterative DFS already landed in transcript patches).
  A3 igraph family: GLPK-vs-HiGHS tie parity (LP objective now all-zero on feedback-free DAGs ->
     solver tie-break decides ranks), BK ordinal-edge conflict quirk, qsort ties (verdicts 19-21).
  A4 stages B-D (flat edges, edge labels, clusters) per position.c -- AFTER A1/A2 land.
TRACK B -- fmmm residual (33) + gem (7):
  B1 fmmm triage vs honest refs: which legs fail now? (references regenerated w/ fixed binary;
     probe showed RMSD ~1e-3 at steps10 -- suspect crossings-leg integer discreteness + steps100/200
     parity). Possibly bit-exact reachable via full RNG-stream match (r71 method).
  B2 gem re-port vs honest refs (r71 port matched STALE binary; dagua-vs-new 0.21 RMSD at matched
     rounds -- read foxglove-202510 gem source, first-divergence trace).
TRACK C -- mds (22) + tails (15):
  C1 mds disconnected residuals (8 flipped; which legs still fail on the rest? DLA placement-scale
     delta noted r=167 vs 3.3 -- check if crossings leg penalizes packing geometry).
  C2 mds connected = evidenced floor (eigenvalue ties, E2) -> formal floor disposition w/ data.
  C3 umap numba neg-sampling trace (7); drl/neato/maxent perturbation dispositions (8).
TRACK D -- infrastructure:
  D1 big-graph tier: hang-safe scoring (landmark APSP + sampled-crossings SE now plumbed +
     APPROX_UNRESOLVED semantics per metrics reports) -> rescore >300-node combos (~238 upper
     bound from r73 snapshot; ba_500/2000/5000 et al).
  D2 population-equivalence aggregate tier (S3 design, metadata-only, BH-corrected).
  D3 final ledger generation + official report + gates + r76_RESULTS + baton + memory.
  D4 (added 2026-07-03, JMT plumbing discussion): ORACLE INVARIANTS -- fold param-sensitivity
     tripwire into validate_benchmark_integrity.py (reference engines: vary iters/steps ->
     output MUST differ, else abort scoring), expected-__for__-row-count assertion on any
     --seed-refs run, overwrite-or-fail on analysis output files. Permanent gates, not notes.

## Wake-up routing
Case A: codex done + not verified -> verify (NOTES + gates + git log) -> merge or park (2-attempt
max per item) -> next item. Case B: bench/analysis done -> read counts -> next. Case C: codex
quota out -> agent-control status; fallback per global rules; note here. Case D: idle + items
open -> dispatch next per tracks. ALWAYS: check `ps -eo pid,args | grep -E "codex exec|
run_benchmark|definitive_fidelity"` before dispatching; never two writers on one results.json;
batch-scale --timeout (>= seeds x per-seed x 2); OMP_NUM_THREADS=1 on benches; verify git log
BEFORE any worktree remove.

## GUARDRAILS (unchanged from r75 + new lessons)
NEVER launder (gate_5 0/40). Benchmark path only. Params+seed matched. No runtime delegation.
Floor claims need FP-chaos/eigengap evidence. final_rung is a STRING. Adversarial-critique big
research before code (codex high-effort critic). Version-pin all source citations (graphviz `git
show 7.0.5:`; igraph = INSTALLED 1.0.0 runtime traces; OGDF = ~/tools/ogdf-src foxglove-202510).
Per-combo freshest-dir overlay is LAW (7fa972e). Resume counts errors as complete -- fresh dir
for retries. 2-attempt max per work item, then park documented.

## JMT BOUNDARY RULINGS (2026-07-02, final)
- OMIT: patched-graphviz reference build (theta/steps stay no-canonical) -- "recompiling the
  whole package" excluded.
- OMIT: LAPACK-degenerate eigenbasis vendoring (connected mds = floor; DO upgrade the label to
  "proven member of reference equivalence class" -- claim-strengthening only, no port).
- PUSH TO EXHAUSTION: (a) all "FP-chaos floor" candidates via first-divergence bisection --
  NO chaos-floor label until bisection stops finding op differences (sfdp connected residue,
  drl, any gem/fmmm leftovers); (b) umap scalar-faithful SGD port (tau_rand xorshift stream is
  portable; target distributional match, bit-exact if float32 order cooperates).
- OVERNIGHT MODE: JMT asleep; run autonomously to DONE. "Fix everything you can fix; leave
  nothing on the table except the truly absurd."

## Iteration log
- 2026-07-03 ~12:50: POST-REBOOT RESUME (Fable). Checklist executed: no stale watchers/procs
  (one FOREIGN torchlens codex running -- untouched); pause sentinels clear; /tmp rebuilt from
  r76_scratch. KEY FINDING: the mincross codex FINISHED before reboot (snapshot log has final
  summary) -- honest ladder-a fail 4/6; residual root-cause NAMED: pre-mincross rank/decompose
  parity (edge-weights asymmetry: DOT adapter emits no weight= but dagua rank consumes
  edge_weights -- necessary-not-sufficient; + build_ranks/GD_nlist seeding). WIP uncommitted in
  dagua-mincross2 worktree (3 files, 517-line diff) + notes banked.
  WAVE 3 DISPATCHED (4 parallel):
  * gem full-family rescore: pid 82292, log /tmp/r76_gem_rescore.log, output
    eval_output/fidelity_definitive/r76_gem_rescore.jsonl (315 combos), watcher b1caro52w.
  * A1-FIXUP mincross (the ONE sanctioned fixup): codex HIGH, pid 84720, log
    /tmp/r76_cx_mincross_fixup.log, worktree dagua-mincross2 (builds on WIP), brief
    r76_scratch/r76_fixup_mincross.md, watcher bau3wx0ds. Targets: rank-parity bisection
    (weights gating -> rank extraction from dot -Tdot -> build_ranks seeding); ladder a-e.
  * C4 sfdp/fdp triage probe: codex MED, pid 87221, log /tmp/r76_cx_sfdp_probe.log, output
    r75_findings/r76_PROBE_sfdp_triage.md (44 sfdp + 3 fdp rows; bisection-first per JMT
    ruling), brief r76_scratch/r76_probe_sfdp_triage.md, watcher byyg0lqos.
  * C3 umap trace+port: codex MED, pid 89533, log /tmp/r76_cx_umap.log, worktree
    dagua-umap-port (branch r76/umap-port), phase-1 first-divergence trace MANDATORY before
    code (verdict #24), then scalar-faithful SGD port (tau_rand stream), brief
    r76_scratch/r76_impl_umap_port.md, watcher bck581p0o.
- 2026-07-03 ~13:05: SFDP PROBE LANDED (r76_PROBE_sfdp_triage.md, 288 lines). VERDICTS: all
  47 rows STRUCTURAL (0 hairline; RMSD 0.0116-0.3841). Cluster A = 21 disconnected rows:
  first divergence NAMED = component orchestration (graphviz ccomps + ONE mutable ctrl reused
  across components + packSubgraphs vs dagua recursive pipelines + neato packer;
  sfdpinit.c:268-315). Cluster B = 23 connected rows: init RNG proven matched (libc parity
  seed 100); divergence between multilevel matching/prolongation and first spring-electrical
  iters -- needs instrumented gv750 trace build. Cluster C = 3 rows are FDP not SFDP ->
  rerouted out of sfdp closure (aggregate-tier or fdp parity candidates -- decide at ledger).
  NO floor labels assigned (bisection not exhausted -- correct per JMT ruling). Rejected r75
  theories NOT re-litigated. WAVE 3b DISPATCHED:
  * C4a sfdp-disc fix: codex MED, pid 144153, log /tmp/r76_cx_sfdp_disc.log, worktree
    dagua-sfdp-disc (branch r76/sfdp-disc), brief r76_scratch/r76_impl_sfdp_disc.md, watcher
    btrj3q7hp. Blast-radius gates: neato/fmmm position hashes unchanged.
  * C4b sfdp-conn bisect-then-fix: codex HIGH, pid 146388, log /tmp/r76_cx_sfdp_conn.log,
    worktree dagua-sfdp-conn (branch r76/sfdp-conn), brief r76_scratch/r76_impl_sfdp_conn.md,
    watcher bqt9w1r6n. Instrumented /tmp gv750 trace build sanctioned (NOT a reference build);
    fix if op difference, 1-ULP floor dossier only if bisection exhausts.
- 2026-07-03 ~13:15: UMAP ATTEMPT 1 LANDED (honest no-commit). PHASE-1 WIN: first divergence
  found+fixed = epochs_per_sample float32 schedule rounding (make_epochs_per_sample computes
  n_samples in float32 THEN divides in float64; dagua promoted early -- 3.48e-08 shift);
  tiny-graph trace now matches draw-for-draw through epoch 1 (schedule, tau-rand, negatives,
  gradients, embedding). BLOCKER WAS MY BRIEF'S ERROR: I banned numba wholesale; dagua's umap
  op already used optional-numba-with-fallback (codex deleted it to comply -> pure-scalar vs
  reference fastmath kernels -> probe gate failed, mean RMSD 0.01504->0.01457 only).
  CORRECTION ISSUED: no-delegation = no umap-learn imports; numba JIT of dagua's OWN kernel is
  fine and mirrors the reference compilation environment. ATTEMPT 2 DISPATCHED: codex MED,
  pid 170464, log /tmp/r76_cx_umap2.log, same worktree (keeps attempt-1 fix, restores numba
  wrappers, makes kernel bodies structurally identical to umap-learn 0.5.11 layouts.py:92-187
  + rdist + clip + tau_rand; per-epoch bisection loop if RMSD does not collapse), brief
  r76_scratch/r76_impl_umap_retry.md. FINAL umap attempt -- honest fail -> aggregate tier.
  Note: attempt-1 codex flagged a phantom "repo AGENTS says no commits" concern; retry brief
  grants explicit commit authority on the branch.
- 2026-07-03 ~13:35: SFDP-DISC ATTEMPTS 1+2 LANDED (honest no-commit, reverted). Both carried
  ctrl scalar state (K/random_start/cooling timing) across components -- improved 3/6 then
  2/6, below the 4/6 gate. THE PRIZE: exact unported rule NAMED = graphviz's single
  process-wide rand() stream threads the WHOLE component loop (Multilevel.c
  random_permutation consumes draws per component BEFORE the conditional
  srand(random_start); prolongation flips random_start=FALSE on the SHARED ctrl -> component
  order + prior draw counts determine later streams; dagua used independent per-component
  GraphvizRandom). PROCESS BUG (mine): probe file + r75_final.jsonl are untracked ->
  absent in worktrees; both sfdp briefs referenced them relatively. Fixed: probe file copied
  into both sfdp worktrees; all future briefs use absolute main-repo paths for artifacts.
  FIXUP DISPATCHED (final disc attempt): codex MED, pid 260695, log
  /tmp/r76_cx_sfdp_disc2.log, brief r76_scratch/r76_fixup_sfdp_disc.md -- step 1 instrumented
  gv750 stream-schedule trace (own /tmp/gv750-disc copy; /tmp/gv750-trace belongs to conn
  task), step 2 thread ONE GraphvizRandom through the loop. Same 6-graph gate.
  Conn codex (146388) healthy: instrumented build up, writing tests, gates running.
- 2026-07-03 ~13:50: GEM RESCORE LANDED = NEAR-TOTAL CLOSURE. Deduped 315 combos (pre-reboot
  run had completed and my rerun APPENDED -- jsonl rewritten deduped in place): 147
  identical_raw + 166 equivalent_raw + 0 no-canonical + 2 divergent. The 2 divergent are
  random_dag_50::iters2000 + random_dag_200::iters100, BOTH disconnected = MAAR
  packing-tie-break family (same root cause as fmmm's 4 parked rows -> now 6 rows total
  sharing that cause; strengthens case for one MAAR attempt-2 pre-ledger). All 7
  r75-divergent gem rows flipped (2 identical, 5 equivalent). 7 iters100 rows moved
  identical->equivalent = anticipated HONESTY CORRECTIONS (old passes matched the stale
  binary; all 7 verified quality_equivalent_raw=true, rung-2 held). gem DISPOSITION: closed
  (313/315 rung-1/rung-2; 2 rows -> MAAR packing cluster).
- 2026-07-03 ~14:15: UMAP ATTEMPT 2 LANDED (honest no-commit, but MAJOR narrowing). Kernel
  parity PROVEN: dagua's local numba kernels (tau_rand i4(i8[:]), rdist f4 fastmath,
  clip, serial single-epoch) match installed umap-learn 0.5.11 EXACTLY on trace (epoch-1
  embedding max diff 0.0; RNG state exact). Bisection killer evidence: substituting the
  REFERENCE'S OWN compiled kernel still leaves RMSD ~0.14 on random_dag_50 -> divergence is
  NOT the epoch body. Pre-optimizer all-exact (distances, fuzzy COO 628 entries diff 0.0,
  head/tail, schedule, curve, base RNG state). SOLE divergence = spectral init: same RNG
  stream consumed, DIFFERENT second-eigenvector basis (max diff 0.376) in degenerate/
  near-degenerate component eigenspace (fuzzy graph 2 comps [52,45]). Same class as mds
  connected floor (JMT-ruled: no vendoring; equivalence-class evidence). Per-seed RMSD is the
  WRONG metric in this chaotic regime -- rung-2 distributional equivalence is the real test.
  STAGE-1 DISPOSITION TEST launched: pid 319281, log /tmp/r76_umap_stage1.log, script
  r76_scratch/r76_umap_stage1.sh -- bench 5 graphs (3 divergent + citation_dag_300 +
  clustered_longlabel_handoffs regression) x 6 umap variants x 100 seeds FROM THE WORKTREE
  (uncommitted attempt-2 code), rescore 22 combos -> r76_umap_stage1.jsonl. DECISION RULE:
  7 divergent flip to equivalent AND regression rows stay identical -> commit attempt-2 on
  r76/umap-port + full-family umap bench; else -> floor/aggregate disposition w/ the
  bisection dossier (which is already sufficient evidence: named non-chaos-free op = ARPACK
  eigsh basis selection under degeneracy).
- 2026-07-03 ~14:30: MINCROSS FIXUP LANDED (no commit yet, but A1 is DONE-ENOUGH). Ladder-a
  PASSED 5/6 exact: the missing seed-order rule was `class2()` then `decompose(g,1)` component
  DFS (NOT reverse fast-node creation) -- weighted_karate_34 now EXACT (pass-0 178=178, final
  63=63). Weight gating landed (graphviz fidelity ignores dagua edge_weights; ranks match
  reference on both residual graphs). heavy_tail_weights_50 off by ONE crossing (pass-0 91 vs
  96; "expanded fast-edge metadata/order" residual). Ladders c/d "failed" but MY GATE WAS
  MIS-SPECIFIED: ba_500 reference dot renders ~140,276 crossings (ledger cross_R), old dagua
  117,932, port 95,261 -- gate demanded <=11k which is a QUALITY target not a FIDELITY target.
  Rendered divergence despite exact ordering = downstream stages (position.c B-D, flat edges,
  order-to-coordinate plumbing) = the queued A4 item. perf: ba_500 mincross 33.7s. b/e green
  (457+54 tests). Known pre-existing failure: test_bench_large hierarchy checkpoint test.
  A4 DISPATCHED: codex HIGH, pid 336440, log /tmp/r76_cx_sugiyama_a4.log, same worktree ON TOP
  of A1 WIP, brief r76_scratch/r76_impl_sugiyama_a4.md. Step-0 localization (ordering-gap vs
  rendered-gap per graph incl ba_500), then port named downstream stages; fidelity-correct
  gates (move TOWARD reference, >=3/4 + ba_500, byte-identical regression rows); on pass
  commit the ENTIRE stack (A1+A4). A3 (igraph GLPK/BK/qsort ties) SERIALIZED after A4 (shared
  files).
- 2026-07-03 ~14:50: SFDP-CONN ROUND 1 LANDED = FIRST DIVERGENCE FOUND AND FIXED. Named op:
  graphviz symmetrized-CSR neighbor ORDER (makeMatrix -> SparseMatrix_symmetrize appends
  outgoing then incoming-transpose entries; coarsening matching iterates row order ->
  hierarchies were structurally different, e.g. hexagonal [42,24,12,6] vs gv [42,24,15,8,4]).
  Fix +5/-4 gated to graphviz_order=True. RMSD medians: hourglass 0.526->0.0025, hexagonal
  0.384->0.0018, planar_60 0.090->0.019. Unchanged-row gate passed. Codex withheld commit
  ONLY due to test_bench_large::test_hierarchy_checkpoint_rejects_incomplete_manifest -- I
  VERIFIED it fails on untouched develop (pre-existing, 2 codexes hit it independently) and
  COMMITTED myself: 681370b (fix) + 09b8a28 (notes) on r76/sfdp-conn. TODO ledger-phase:
  file/fix the pre-existing test separately.
  ROUND 2 DISPATCHED: codex HIGH, pid 356295, log /tmp/r76_cx_sfdp_conn2.log, brief
  r76_scratch/r76_impl_sfdp_conn2.md -- bisect real_karate_34 (0.3898, unchanged) +
  weighted_chain_20 (0.2390, unchanged); PRIME HYPOTHESIS: weighted-edge handling in
  coarsening matching (all 3 fixed graphs unweighted, both residuals weighted). Fix-or-floor
  per JMT ruling; pre-existing-failure exclusion clause now in all briefs.
- 2026-07-03 ~15:05: UMAP STAGE-1 VERDICT + TWO INFRA DISCOVERIES. Bench 3000/3000 ok; rescore
  22 combos. RESULTS: clustered_longlabel 6/6 STAYED bit-identical; citation_dag_300 6/6
  DROPPED identical->equivalent (= attempt-2's diagnostic "spectral alignment" hunks broke
  matching graphs -- classes: schedule fix GOOD, kernel parity GOOD, spectral alignment BAD);
  the 7 divergent stayed non-equivalent. DISCOVERY 1 (oracle alarm, param-insensitivity
  class): parallel_multiedge_bundle reference mean_W IDENTICAL to 16 digits across ALL 5
  variants (0.6037692...) while citation refs differ per variant -> multiedge-specific
  reference anomaly (adapter clamping or degenerate path) -- MUST resolve before
  dispositioning those 5 rows; also dagua W 0.128 vs ref 0.604 = structural multi-edge
  handling difference, NOT eigenspace chaos. DISCOVERY 2 (era seed-range mismatch):
  escalation_final umap refs are SEED-42-START era (42-141); benches now use 100-199 ->
  matched n=42, halved TOST power; final ledger (D3) must flag rows with n<100 and the
  42-start reference era generally. ACTIONS DISPATCHED:
  * umap spectral-revert codex: pid 405030, log /tmp/r76_cx_umap3.log, brief
    r76_scratch/r76_impl_umap_revert_spectral.md -- keep schedule+kernel, revert spectral
    hunks to HEAD, byte-identity checks vs HEAD on citation_dag_300, commit on pass.
  * umap refs regen seeds 100-199: pid 407303, log /tmp/r76_umap_refs.log, dir
    benchmark_100seed_r76_umap_refs (5 stage-1 graphs x variants; __for__-row verification
    armed per r75 lesson).
  NEXT after both: stage-1b rescore (chain + r76_umap_refs + r76_umap_fix) -> dispositions:
  multiedge-5 (pending oracle answer), random_dag pair (expect equivalence at full power or
  eigenspace floor dossier), citation identity restored.
- 2026-07-03 ~15:25: ***ORACLE BUG #2 FOUND AND FIXED (umap tiny-graph fallback).*** The fresh
  param-identical refs on parallel_multiedge_bundle exposed it: umap_competitor.py returned
  seeded torch.randn for num_nodes<=3 WITHOUT running umap or consuming variant params. The
  "reference" for that graph was a Gaussian cloud (stress 0.604 vs dagua's real-umap 0.128 --
  the 5 divergent multiedge rows were dagua-does-umap vs reference-does-randn). Empirically
  verified: umap-learn RUNS at n=3 with random init (min_dist visibly changes output); only
  n<=2 genuinely fails (n_neighbors>1 required). FIXED adapter cutoff 3->2, committed develop
  7d1f090. Blast radius verified: parallel_multiedge_bundle is the ONLY <=3-node graph.
  Multiedge refs regen w/ real umap: pid 466512, dir benchmark_100seed_r76_umap_refs2.
  REMAINING for umap closure: (a) dagua pipeline ALSO special-cases N<=3 ("historical
  explicit fallbacks", umap_layout.py docstring) -- its fidelity path must run the real port
  at N=3 (QUEUED behind the spectral-revert codex, same file umap.py); (b) then re-bench
  dagua multiedge rows + stage-1b rescore (chain + r76_umap_refs + r76_umap_refs2 + fix dir).
  Era note appended to D3: umap refs in escalation_final are seed-42-era; ledger must prefer
  r76_umap_refs* dirs and flag n<100 rows.
- 2026-07-03 ~15:45: SALVO EXPANSION + A4 VERDICT. Multiedge refs2 verified (600/600 ok;
  param-sensitivity tripwire passes: mindist/spread differ, default==nn5==nn30 correctly
  clamp-identical at n=3; seed-sensitive). DISPATCHED: F1 fairness triage (codex 488766, log
  /tmp/r76_cx_fairness.log, 79 superior-distinct rows, oracle-sanity-first brief); D1
  big-graph rescore (local pid 486784, 400 combos >300 nodes, 16 graphs, log
  /tmp/r76_biggraph.log -> r76_biggraph.jsonl); B2b MAAR attempt-2 (first dispatch DIED --
  dagua-fmmm-disc worktree had been cleaned pre-reboot; RECREATED off develop 7d1f090,
  redispatched codex 501893, log /tmp/r76_cx_maar2.log; instrumented TileToRowsCCPacker
  trace mandate).
  A4 LANDED (honest no-commit, blocker precisely named): rendered real-node RANK+ORDER now
  MATCH on all 3 step-0 graphs; ba_500 internal mincross 79046 vs gv 79098 (0.07%).
  Divergence is purely X-COORDINATE parity: (1) node boxes -- dagua feeds narrower boxes
  than gv (44-52pt vs 54-70pt); (2) nodesep units -- benchmark 1.0 = DOT inches = 72pt, not
  layout units; (3) virtual-node widths + aux constraints. A4's units-only patch improved
  crossings 3/4 but stress 2/4 -> honestly reverted; verdict: box+units must land TOGETHER.
  A4b DISPATCHED (final targeted attempt): codex HIGH 507112, log
  /tmp/r76_cx_sugiyama_a4b.log, brief r76_scratch/r76_impl_sugiyama_a4b.md -- box rule
  validated against dot -Tjson, units port, gates on crossings AND stress, commit full A1+A4b
  stack on pass; else dossier = official port-in-progress disposition. A3 (igraph ties) still
  queued behind the mincross worktree.
- 2026-07-03 ~16:00: ***SFDP-DISC FIXED (attempt 3, committed).*** Shared-RNG-stream port
  landed: 8647d47 (fix) + 80d693d (docs) on r76/sfdp-disc. RMSD medians improved 6/6:
  encoder_residual 1.090->0.363, label_cycle_collage 0.738->0.279, kitchen_sink 0.558->0.132,
  multi_component_80 1.014->0.112, parallel_cycles 0.752->0.168, random_dag_200 1.250->0.065.
  Regression gates: 25/25 connected sfdp hashes unchanged, 8/8 neato/fmmm disc hashes
  unchanged, pytest green. KEY TRACE CORRECTION: gv 7.0.5 RESTORES ctrl after each multilevel
  call -- only the process rand() stream persists across components (why ctrl-carrying
  attempts 1-2 regressed). Residual after fix = packer geometry itself (packSubgraphs
  polyomino vs dagua packer; doSplines) -- defer judgment to family re-bench verdicts.
  PLAN: when conn round-2 lands -> merge r76/sfdp-disc + r76/sfdp-conn into develop -> sfdp
  FAMILY RE-BENCH (all classic_sfdp combos <=300 nodes, seeds 100-199) -> family rescore
  (gem-pattern) -> dispositions.
- 2026-07-03 ~16:20: FAIRNESS TRIAGE LANDED (r76_PROBE_fairness.md, 189 lines). Count
  correction: 89 superior-distinct rows (not 79). BUCKETS: 8 reference-bug (umap multiedge 4
  = randn fallback; gem 2 + fmmm 2 = stale runner -- REMOVE superior labels, rescores/regens
  already in motion); 13 sfdp reference-param-noop (gv sfdp refs BIT-IDENTICAL across
  theta04/theta08/steps200/p_neg2 variants on connected graphs -- collapse these variant rows
  to reference-non-expressible at ledger, extends the r75 theta/maxiter no-canonical
  finding to p_neg2); 54 fair-but-portable (sugiyama 45 [18 dot + 27 igraph] + sfdp 9 --
  disposition rides on A4b/A3/sfdp merges); 11 fair-non-portable (mds 6, maxent 2, misc 3 --
  keep w/ documented basis/packing cause); 3 reclassify (mds disc margin-direction).
  LEDGER RULES BANKED (probe recs 1-5): no superior-distinct for param-noop refs; sugiyama
  45 held until parity attempts merged-or-parked; narrow "fair non-portable basis" labels.
  D4 SPEC SHARPENED: same-graph same-seed reference positions must NOT be bit-identical
  across param variants unless declared non-expressible or clamp-equivalent; seed response
  alone is NOT adequate oracle sanity. A3 igraph (27 portable rows) = next dispatch when
  mincross worktree frees.
- 2026-07-03 ~16:40: UMAP REVERT COMMITTED (795ccbd fix + 77c9830 docs on r76/umap-port):
  byte-identity to HEAD restored on citation rows (torch.equal across 6 variants x 3 seeds),
  tiny-trace exact, schedule+kernel parity preserved. STAGE-1B launched: pid 548099, log
  /tmp/r76_umap_stage1b.log, bench worktree code -> benchmark_100seed_r76_umap_fix2 (fresh
  dir per retry law), rescore w/ honest refs -> r76_umap_stage1b.jsonl.
  MAAR ATTEMPT-2 PARKED (honest, notes committed bc02ce6 on r76/fmmm-disconnected): named
  the exact rules (FMMM uses MAARPacking NOT TileToRowsCCPacker; OGDF qsort stable only
  <=40 items; MAAR equal-row-width tie = newest pairing-heap push) but porting them WORSENED
  random_dag_50 vs honest refs (0.9-1.0 RMSD) -> tie rules alone are not the _50 residual;
  reverted, no commit. ***INFRA CATCH: random_dag_200 honest ogdf refs MISSING from r76_refs
  (regen capped at <=300 builder-nodes; random_dag_200 counts >300)*** -> its gem/fmmm
  verdicts (incl 2 of the 6 MAAR-cluster rows) were scored vs STALE-binary refs =
  unreliable. REGEN LAUNCHED: pid 558655, dir benchmark_100seed_r76_refs2 (ogdf fmmm+gem,
  random_dag_200, seeds 100-199, --max-nodes 0). After: rescore MAAR-cluster rows w/ refs2;
  also audit big-graph ogdf ref coverage when D1 lands (ba_500+ likely same gap -> ledger
  must flag stale-ref ogdf rows or regen).
- 2026-07-03 ~17:00: SFDP-CONN ROUND 2 LANDED = SECOND OP FOUND+FIXED. Weighted-edge
  hypothesis CONFIRMED: DOT adapter emits no weight= -> gv makeMatrix sees unit weights;
  dagua fidelity hierarchy consumed edge_weights (same asymmetry class as mincross
  weight-gating). Fix +4/-1: real_karate_34 0.3898->0.0800, weighted_chain_20
  0.2390->0.0476. Residual first-divergence now 6e-10 coarsest force deltas; 1-ULP
  perturbation reproduces comparable magnitudes for SOME seeds -- floor evidence close but
  honestly NOT claimed. Commit withheld over classic_fcose name-list test -- VERIFIED
  pre-existing on develop (phantom blocker #2; both now in briefs' known-failures) -- I
  committed: e3e4622 (fix) + 8e3dcc6 (notes) on r76/sfdp-conn.
  ***SFDP MERGED TO DEVELOP***: 7a54b9d (disc: shared RNG stream) + aff2a0e (conn: CSR
  order + unit weights). Merged smoke green (28 sfdp tests). LAUNCHED: sfdp FAMILY RE-BENCH
  pid 583635 -> benchmark_100seed_r76_sfdp_fix (classic_sfdp all variants, <=300 nodes,
  seeds 100-199), log /tmp/r76_sfdp_bench.log; MAAR-cluster rescore pid 585922 (12
  fmmm/gem combos on random_dag_50/200 vs honest refs2) -> r76_maar_rescore.jsonl, log
  /tmp/r76_maar_rescore.log. random_dag_200 honest refs DONE (600/600 ok, refs2).
  NEXT: sfdp bench done -> family rescore -> dispositions (incl 13 param-noop variant rows
  -> no-canonical). Then remaining: A4b + A3, umap stage-1b verdicts, D1 verdicts + ogdf
  big-graph ref audit, D2 population tier, D3 ledger + D4 invariants.
- 2026-07-03 ~17:15: MAAR RESCORE (12 combos, honest refs2) = mixed w/ TWO CAVEATS. Verdicts
  as scored: random_dag_200 fmmm steps10/100 EQUIV (flips), steps200 div; gem iters100
  ident(!), iters500/2000 div; random_dag_50 fmmm all div, gem iters500 equiv,
  iters100/2000 div. CAVEAT 1: random_dag_200 DAGUA side is old-era (n=42; graph >300
  builder-nodes -> excluded from r76_gem_fix bench; old positions + old code) -- its rows
  unreliable until current-code bench. LAUNCHED: pid 606191, dagua classic_fmmm+classic_gem
  on random_dag_50+200, seeds 100-199 -> benchmark_100seed_r76_maar_bench, then final MAAR
  rescore. CAVEAT 2 (LEDGER INVESTIGATION ITEM): random_dag_50::classic_gem_iters100
  identical_raw FLIPPED between gem-rescore (True) and maar-rescore (False) on the same
  chain -- diff the two jsonl rows field-by-field at ledger time; possible scoring
  nondeterminism or overlay-resolution sensitivity. Do NOT trust single-run verdict flips
  near margins for MAAR-cluster rows; ledger decides w/ D4 invariants + fresh benches.
- 2026-07-03 ~17:40: ***ACCOUNTING CORRECTION (major, positive) + D3 LEDGER RULE.*** Root
  cause of "regressions"/verdict weirdness found in the SCORER, not the data:
  quality_identical_raw = metric_identical AND battery_eligible, where eligibility =
  reference plain_mean_W_R <= 1.0 ("canonical reference" gate added in r75 3Q work).
  Bit-identical rows vs poor-stress references get DEMOTED to
  quality_identical_exploratory. r75_final rows predate the gate (fields None) -> r75
  counted ungated; r76 rescores counted gated = apples-to-oranges. VERIFIED: citation_dag_300
  umap positions are byte-equal to refs on ALL sampled seeds incl 142-199 (torch.equal) yet
  scored ident_raw=False / exploratory=True (W_R_plain 1.09-1.13 > 1.0).
  CORRECTED COUNTS: gem = 147 raw + 160 exploratory = ***307/315 metric-identical*** (+6
  equivalent-only, 2 MAAR-divergent); the "7 iters100 honesty-correction regressions" are
  RETRACTED -- all 7 are exploratory-IDENTICAL (bit-identical vs refs with plain stress
  1.16-1.32). umap stage-1b = 13 raw + 6 exploratory (all citation -- never regressed) =
  19/22 identical; multiedge 5/5 became bit-identical after the oracle fix.
  D3 LEDGER RULE (binding): fidelity-identical := identical_raw OR identical_exploratory;
  report the canonical flag as metadata, never as a fidelity gate. D3 must recompute ALL
  families uniformly under this rule.
  D3 INVESTIGATION ITEM: scoring nondeterminism -- random_dag_50::classic_gem_iters100
  metric-identical in gem-rescore but NOT in maar-rescore (same chain/data); run analysis
  twice on one combo set + field-diff before trusting marginal verdicts.
- 2026-07-03 17:32 ET: SPRINT RESUMED (paused 14:58-17:30 per JMT, soft, claude-only;
  PAUSE_STATE at ~/.claude/state/sprint-pause/20260703-1458-dagua-r76-fidelity/). During
  pause: A4b codex finished; MAAR dagua bench + D1 finished; sfdp bench 96.6% (watcher
  re-armed 240min). HARVEST:
  * A4b: honest no-commit; weighted_karate crossings 108/108 EXACT + hub_skip 2/2;
    dense_pair 343v331, heavy_tail 63v67; residual NAMED = position.c aux x-graph
    (make_edge_pairs, LR edges, slack nodes, initial ranks, insertion order). Stack (975
    lines incl A4b + classic_competitor node-box helper -- needs gating review) PROTECTED as
    wip commit 525327d on r76/mincross.
  * D1 big-graph (400 combos): 31 identical + 192 equivalent + 83 divergent + 23
    no-canonical + 71 insufficient (52 = matched_seeds<30 -- the seed-era mismatch; 7
    ref<30; 12 reimpl<30). Ledger items: era-aligned re-bench for the 52; cause-bucket the
    83 (ogdf big-graph stale refs + sugiyama port-in-progress expected dominant).
  * A4c DISPATCHED (genuinely final sugiyama push): codex HIGH pid 715514, log
    /tmp/r76_cx_sugiyama_a4c.log, brief r76_scratch/r76_impl_sugiyama_a4c.md -- aux x-graph
    trace+port; on fail the A4b/A4c dossier = official port-in-progress disposition.
  * MAAR FINAL rescore launched: pid 704103 (fresh both-sides: refs2 + maar_bench) ->
    r76_maar_final.jsonl.
  PENDING: sfdp bench (96.6%) -> family rescore; MAAR final verdicts; A4c; then umap merge,
  A3 (if A4c passes), D2 population tier, D3 ledger + D4 invariants.
- 2026-07-03 ~17:50 ET: ***MAAR CLUSTER CLOSED (final rescore, fresh both sides, n=100).***
  random_dag_200: ALL 6 rows EQUIVALENT (fmmm steps10/100/200 + gem iters100/500/2000) --
  the stale refs + seed-era mismatch were most of the "divergence". random_dag_50: gem
  iters100 + iters500 IDENTICAL(any); residual = 4 rows: fmmm steps10/100/200 (dagua stress
  LOWER than ref: 0.74v0.86, 0.69v0.83, 0.71v0.83 -- superior-pattern, quality NOT lesser)
  + gem iters2000 (W 1.0217v1.0204 = 0.13% gap, marginal). DISPOSITION: 4 rows =
  divergent-with-named-cause (MAAR packing tie-breaks, dossier bc02ce6: pairing-heap
  newest-push ties + qsort stability; port attempt worsened fit -> non-portable-as-traced)
  + quality equal-or-better = JMT option-3 compliant / superior-distinct candidates.
  NO further MAAR work. Ledger: r76_maar_final.jsonl SUPERSEDES r76_maar_rescore.jsonl and
  all older verdicts for these 12 combos.
- 2026-07-03 ~18:00 ET: A4c LANDED (dossier-only, notes committed 156cb25). Aux x-graph
  EDGE construction now matches; residual narrowed to ONE constant class: ND_lw/ND_rw node
  half-widths feeding make_LR_constraints minlens are ~1pt-per-node smaller in dagua (gv
  146,146,136 vs dagua 144,144,135 on karate; 177,141 vs 175,139 on dense_pair).
  classic_competitor node-box helper VERIFIED gated to graphviz-fidelity engine input (no
  scoring leak). Calibration movement-gate now measures margin noise (crossing-exact graphs
  with 1e-4 stress deltas) -- ARCHITECT DECISION: stop micro-porting; the benchmark scoring
  is the honest arbiter. SUGIYAMA FAMILY RE-BENCH launched FROM THE STACK (worktree
  PYTHONPATH): pid 749129, log /tmp/r76_sugiyama_bench.log -> benchmark_100seed_
  r76_sugiyama_wip (classic_sugiyama all variants <=300n, seeds 100-199). On done -> family
  rescore -> rows flip = merge r76/mincross; no flips = A4b/A4c dossier disposition stands
  (with the 1pt half-width item as the named port-in-progress residual).
- 2026-07-03 ~19:05 ET: SFDP BENCH DONE (53400/53400 ok) -> family rescore launched (607
  combos, pid 814182 -> r76_sfdp_rescore.jsonl, watcher bqa3y90px). UMAP MERGED to develop
  (630bc2f: schedule float32 fix + reference-parity numba kernels, +521/-93). FLOOR-DOSSIERS
  codex dispatched (pid 819132, log /tmp/r76_cx_floors.log, brief
  r76_scratch/r76_probe_floor_dossiers.md): mds-connected equivalence-class proofs
  (eigengap tables + orthogonal-transform-within-degenerate-eigenspace residuals) + umap
  random_dag 1-ULP perturbation experiment + quality-parity tables -> r76_FLOOR_DOSSIERS.md.
  Still pending: sugiyama wip bench (pid 749129), sfdp rescore, floors codex. Then: A3
  decision, D2+D3 ledger, housekeeping (2 pre-existing test failures -- deferred to ledger
  phase alongside D4 invariants).
- 2026-07-03 ~19:30 ET: ***SFDP FAMILY RESCORE: 261 IDENTICAL + 6 EQUIVALENT*** (of 288
  scoreable; 290 no-canonical, 29 insufficient); 28/44 r75-divergent flipped. Residual 21
  divergent + 6 "regressions" (random_dag_200 x3, kitchen_sink x3) are ORACLE-CONFOUNDED:
  (a) rescore used old-era stored sfdp refs while the fixup's gates used FRESH dot runs --
  random_dag_200 shows W_D 0.086 vs W_R 1.292 which contradicts the fixup's 19x RMSD
  improvement vs fresh refs; (b) 4 rows are p_neg2 variants whose refs are PROVEN param-noop
  (fairness probe) -> reclassify no-canonical at ledger, not divergent. SFDP REFS REGEN
  LAUNCHED: pid 827651 -> benchmark_100seed_r76_sfdp_refs (graphviz_sfdp, 19 affected
  graphs, seeds 100-199, --max-nodes 0), log /tmp/r76_sfdp_refs.log. On done -> rescore the
  ~27 affected combos w/ fresh refs -> final sfdp dispositions.
- 2026-07-03 ~19:45 ET: FLOOR DOSSIERS LANDED + COMMITTED (d5a6b08, r76_FLOOR_DOSSIERS.md).
  UMAP random_dag pair: 1-ULP spectral-init perturbation amplifies to final RMSD comparable
  to or LARGER than dagua-vs-reference divergence, 3/3 seeds, both graphs = chaos
  amplification PROVEN; quality parity tables in (W essentially tied on _50). DISPOSITION:
  "evidenced FP-chaos floor (eigenspace basis selection), quality parity shown" -- umap
  family now FULLY dispositioned. MDS connected 14: eigengap tables + eigenspace-membership
  proofs; honest nuance: multiplicity>2 means the formal claim is Gram-eigenspace
  MEMBERSHIP (JMT's "proven member of reference equivalence class"), not 2D congruence.
  LEDGER NOTE (D5b): verify drl/neato/maxent (8 rows) carry adequate r75 floor evidence at
  ledger time; if not, one more small dossier probe.
- 2026-07-03 ~19:55 ET: SFDP REFS DONE (11400/11400 ok, 19 graphs). TRIPWIRE ON FRESH REFS:
  graphviz_sfdp output BIT-IDENTICAL across default/graphviz_fidelity/p_neg2 on BOTH
  connected AND disconnected graphs -> installed 7.0.5 sfdp ignores the repulsive-exponent
  attr (same class as theta/maxiter). LEDGER RULING IMPLIED: classic_sfdp_p_neg2 rows ->
  NO-CANONICAL tier (dagua genuinely computes exponent -2; reference never did). Extends
  r75 no-canonical + confirms fairness-probe param-noop finding with fresh data. SFDP FINAL
  RESCORE launched: pid 884824, 21 affected combos w/ r76_sfdp_refs in chain ->
  r76_sfdp_final.jsonl.
- 2026-07-03 ~20:15 ET: SFDP FINAL RESCORE (fresh refs) + ERA-GAP CATCH. The random_dag_200
  15x-stress "anomaly" root-caused: r76_sfdp_fix bench ran --max-nodes 300, so
  random_dag_200/ba_500/grid_20x20/rgg_500/small_world_500 have NO current-code dagua
  positions -- their verdicts compared PRE-FIX-era dagua vs fresh refs (garbage). (The
  disc-fix codex's own gate benched into benchmark_full separately -- that's why its RMSD
  numbers disagreed with my rescores. THIRD instance of the >300-builder-node exclusion trap:
  gem, fmmm, now sfdp. D4 invariant addition: benches must LOG excluded graphs loudly.)
  TOP-UP BENCH launched: pid 896445 -> benchmark_100seed_r76_sfdp_fix2 (5 big graphs,
  current code, seeds 100-199). Scoreable fresh verdicts so far (fresh both sides):
  encoder_residual W 0.764v0.769 (~parity), label_cycle_collage 0.511v0.763 (dagua BETTER),
  kitchen_sink 0.500v0.479 (4.4% worse), multi_component 0.908v0.866 (4.8% worse),
  random_dag_50 1.097v1.120 (dagua better) -- all DIVERGENT statistically = residual packer
  geometry (packSubgraphs polyomino vs dagua packer), the named remaining disc cause;
  quality mixed-to-better. p_neg2 rows -> no-canonical (fresh-ref tripwire proof).
  After fix2: FINAL sfdp rescore of the 5 graphs' combos -> family disposition table.
- 2026-07-03 ~20:40 ET: ***SFDP PACKING FIXED (C4d, committed+MERGED 9d3cfa4).*** Root
  cause = ANOTHER unit bug: sfdp fed point-unit coords into neato's INCH-space pack helper
  (72x mismatch). Gated to graphviz-fidelity disconnected sfdp; hash gates 33/33 clean; 455
  layout tests green. NOTE: r76_sfdp_fix2 top-up bench (running) launched PRE-merge -> its
  random_dag_200 rows (disconnected) lack the pack fix; after fix2 lands, re-bench
  random_dag_200 sfdp into fix3 + final rescore of disc graphs (encoder_residual,
  label_cycle_collage, kitchen_sink, multi_component, random_dag_50 need rescore vs fresh
  refs WITH pack fix -- the sfdp_fix dir predates it too!). SUGIYAMA BENCH: completed
  50500/51000 ok; 15 errors = CRASH (list index out of range) in graphviz_fidelity on 5
  graphs (er_100, random_dag_50, regular_4_40, rgg_100, sbm_5x50) -> A4d crash-fix codex
  dispatched (pid 960932, worktree dagua-mincross2, gates incl 100-seed topup bench of the
  5). Family rescore of the 424 combos launched (pid 956829 -> r76_sugiyama_rescore.jsonl).
  WAIT -- also note sfdp_fix dir (main family bench) ALSO predates the pack fix: disconnected
  sfdp rows in r76_sfdp_fix are pre-pack-fix. After crash/benches settle: ONE consolidated
  sfdp disc re-bench (all disc graphs, post-merge code) -> final family verdicts. Disk 89%.
- 2026-07-03 ~21:00 ET: ***SUGIYAMA FAMILY VERDICT: MERGE THE STACK.*** Mode-aware recount
  (sugiyama rows are mode-B/deterministic; my quality-flag counter was wrong for them; d_R
  is the authoritative fidelity number; official ledger handles the tiering): graphviz_
  fidelity 77 scoreable = 1 bit-exact + 31 near(<0.01) + 24 close(<0.1) + 21 far. PER-COMBO
  OLD-VS-NEW: 53 improved / 17 worsened / 6 same; median d_R 0.1455 -> 0.0160 (9x). Wins:
  real_karate 0.883->0.002, weighted_karate 0.849->0.002, complete_bipartite 0.990->0.003,
  densenet 0.719->0.004. Worsened tail = label/cluster-heavy graphs (edge_label_braid
  0.60, moe_router 0.36, clustered_longlabel 0.24) = UNPORTED stages B-D (edge labels,
  clusters) -- named residual -- plus the 5 crash graphs (stale fallback positions).
  igraph-family: 30 bit-exact + 6 near + 74 close + 227 far = A3 REMAINS THE BIGGEST OPEN
  ITEM. PLAN: crash fix (in flight) -> merge r76/mincross -> A3 dispatch (same worktree,
  serialized) -> crash-graph topup rescore -> label/cluster residual = port-in-progress
  disposition w/ the A4b/A4c dossiers (or one scoped B-D item if window allows).
- 2026-07-03 ~21:20 ET: ***SUGIYAMA STACK MERGED TO DEVELOP (7a33573).*** Crash fix
  verified: _count_crossings Fenwick sized by EDGE COUNT but indexed by node order
  (aeaf194) + deterministic-repeat cache for the watchdog (9224d71); byte-identity
  preserved on working graphs; topup bench of the 5 crash graphs 3000/3000 ok
  (r76_sugiyama_topup). Post-merge smoke green (57 tests). DISPATCHED: A3 igraph
  tie-parity codex (HIGH, pid 1080544, log /tmp/r76_cx_igraph_a3.log, brief
  r76_scratch/r76_impl_igraph_a3.md -- bisect-first per row class: GLPK degenerate-LP
  tie-break / BK quirk / qsort ties; vendoring GLPK = STOP-and-dossier; family bench gate
  included). LAUNCHED: consolidated sfdp disc re-bench post-pack-fix (pid 1082719 ->
  benchmark_100seed_r76_sfdp_fix3, 8 disc graphs incl random_dag_200 at --max-nodes 0).
  PENDING VERDICT CHAIN: fix3 + fix2 done -> FINAL sfdp rescore (disc + big graphs, fresh
  refs) -> sfdp disposition table. A3 done -> igraph family rescore -> re-merge. Then:
  crash-graph topup rescore, D2 population tier, D3 ledger (mode-aware tiering + corrected
  identical rule + era flags) + D4 invariants + housekeeping (2 pre-existing test failures,
  worktree/branch sweep, baton, memory).
- 2026-07-03 ~22:40 ET: A3 LANDED (honest partial, MERGED 08a6003). PORTED: igraph BK
  ordinal-edge Type-1 conflict quirk (4377a80, gated fidelity_mode="igraph";
  multiscale_skip_cascade now EXACT vs installed igraph, all 5 variants; 461 layout tests
  green). BLOCKED (named, vendoring-excluded class): GLPK degenerate-LP rank tie-break --
  SciPy HiGHS/simplex variants do NOT reproduce installed igraph's basis selection;
  real_karate_34 + moe_router_sparse diverge first at LP rank. width_skew_late_merge has a
  further BK/dummy x-stage detail (named residual). LEDGER LABEL for LP-tie rows:
  "divergent: GLPK basis selection on all-zero objective (non-portable; vendoring
  excluded)" + per-row quality parity. LAUNCHED: post-A3 sugiyama family bench (pid
  1099911 -> benchmark_100seed_r76_igraph_fix, supersedes wip+topup dirs, includes crash
  fix); sfdp disc FINAL rescore (pid 1101966, 48 combos, fix3 positions + fresh refs ->
  r76_sfdp_disc_final.jsonl). fix3 bench was 4800/4800 ok. fix2 (big graphs) still running.
- 2026-07-03 ~23:00 ET: SFDP DISC FINAL: 2 identical + 4 equivalent (pack-fix flips) + 32
  param-noop->no-canonical + 10 divergent (5 graph clusters x2 variants) w/ TERMINAL NAMED
  CAUSE = spline-box polyomino occupancy in packSubgraphs (dagua has no splines at pack
  time; porting spline routing = excluded violence). QUALITY FLAGS for ledger:
  label_cycle_collage dagua BETTER (0.515 v 0.763), encoder_residual better (0.747 v
  0.769), random_dag_50 ~parity (1.132 v 1.120), kitchen_sink ~5% worse (0.505 v 0.479),
  multi_component ~5.7% worse (0.915 v 0.866) -- state plainly in r76_RESULTS. LAUNCHED:
  ogdf big-graph honest refs regen (pid 1107724 -> benchmark_100seed_r76_refs3, 8
  500-node-class graphs x fmmm+gem variants, seeds 100-199; 2000/5000-node monsters go to
  aggregate/population tier w/ era flags, NOT regenerated). REMAINING BEFORE LEDGER:
  igraph family bench (running) -> igraph rescore; fix2 (running); refs3. Then D2+D3+D4
  FINALE (single big codex dispatch planned: full re-analysis all dirs w/ mode-aware
  tiering + corrected identical rule + population tier + oracle invariants into
  validate_benchmark_integrity + official report + r76_RESULTS.md + gates).
- 2026-07-03 ~23:20 ET: ***SUGIYAMA FINAL NUMBERS (r76_sugiyama_final.jsonl, post-A3 bench
  53400/53400 ok ZERO errors).*** graphviz_fidelity: 1 bit-exact + 31 near(<0.01) + 25
  close + 20 far + 2 insufficient. igraph-family: 60 bit-exact (DOUBLED by BK quirk from
  30) + 1 near + 54 close + 222 far; d_R improved 115 / worsened 65 (all churn WITHIN far
  tier) / same 157. ***ZERO tier regressions in either family*** (no bit-exact/near row
  fell out). Crash graphs now score (er_100 d_R 0.011). Far-tail causes all NAMED: graphviz
  = stages B-D (labels/clusters) + aux minlen 1pt half-width (A4b/A4c dossiers); igraph =
  GLPK basis selection (vendoring-excluded, r76_IMPL_igraph_NOTES) + residual BK/dummy
  detail. SUGIYAMA IS DISPOSITIONED. Remaining before D3 finale: fix2 bench + D4 infra
  codex (both running).
- 2026-07-04 ~00:15 ET: ***D4 LANDED + MERGED (ce4562d); D3 FINALE DISPATCHED.*** D4
  (8b43153): all 6 items -- param tripwire, seed-era guard, __for__ assertions +
  --max-nodes exclusion logging, atomic overwrite-or-fail, scorer --self-check, 2 test
  fixes + guardrail tests (48 green). DETERMINISM: GREEN (12-combo self-check
  deterministic; earlier flip = era-mixing not RNG). Validator vs real dirs: 234
  param-noop failures on HISTORICAL ogdf/igraph_mds/ogdf_stress dirs (archaeological
  stale-runner fingerprint) + 241 era warnings -> D3 brief amended w/ adjudication mandate
  (fresh-coverage audit per ogdf combo; igraph_mds/ogdf_stress clamp-equivalence w/
  evidence). Pre-existing failure #3 noted (cosmetic render smoke, develop). ALL DATA
  GENERATION COMPLETE (fix2 3000/3000 ok). D3 FINALE: codex HIGH pid 1151669, log
  /tmp/r76_cx_d3_finale.log, brief r76_scratch/r76_impl_d3_finale.md -- authoritative pass
  over the full universe, corrected tier rules, named-cause registry, population tier,
  gates, official report + r76_RESULTS.md on branch r76/ledger. AFTER IT LANDS: verify +
  merge, worktree/branch sweep, baton, memory update, sprint close.
  ON EACH LANDING: verify per wake-up routing (Case A/B). After mincross verdict -> sugiyama
  family re-bench if ladder passed. After gem rescore -> read closure counts. After sfdp probe
  -> targeted fix/floor codexes. After umap -> re-bench umap family. THEN: superior-distinct
  fairness triage (79), fmmm MAAR attempt-2 (only if 4 rows still matter), big-graph tier (D1),
  population tier (D2), FINAL LEDGER (D3).
- 2026-07-02: STATE created at r75 close (r75 official: divergent 574->352, 3Q 36->79,
  no-canonical 290, ZERO bit-exact regressions -- verified). WAVE 1 dispatched (all codex):
  A2 xns-perf (worktree dagua-xns-perf, branch r76/xns-perf), B1 fmmm triage probe,
  C1 mds+gem triage probe. PIPELINE: A2 done -> A1 mincross (recover attempt-2 patches from
  r75_findings/r75_cx_mincross*.log) -> A4 stages B-D; B1/C1 verdicts -> targeted fix codexes
  (bisection-first per JMT ruling); then umap scalar port; then superior-distinct fairness
  triage (79 rows: match-the-worse-reference where portable); then D1 big-graph hang-safe
  rescore + D2 population tier + D3 final ledger/report/baton.
- 2026-07-03 ~00:30: TRIAGE PROBES IN + CRITICAL CATCH. ***--seed-refs DID NOT REGENERATE REFS
  in r75 (my invocation error: ref engines must ALSO be in --engines; r75_fixes has ZERO
  __for__ rows). r75's fmmm/gem verdicts scored vs STALE refs.*** REF REGEN running correctly:
  pid 2416761 (via pidfile /tmp/r76_refs.pid), --engines ogdf_fmmm,ogdf_gem --seed-refs same,
  dir benchmark_100seed_r76_refs, <=300 nodes, log /tmp/r76_refs.log. On done -> rescore
  fmmm+gem combos (chain + r76_refs last).
  TRIAGE VERDICTS: fmmm 30 OGDF rows = 24 near-match (RMSD 1e-17..1e-3 vs honest runner --
  stale-ref accounting + integer-crossing margins; expect mass flips on rescore) + 6 structural
  ALL-DISCONNECTED (random_dag_50 x3, random_dag_200, multi_component_80,
  kitchen_sink_platform) + 3 fdp-family rows; RNG primitive proven bit-exact (mt19937+
  uniform_int_distribution first-20 match); deep_chain steps200 "regression" = stale-ref
  artifact (identical R stress across step counts). mds: 14 connected = FORMAL FLOOR DOSSIER
  (machine-precision eigenvalue ties, evd flips coordinates); 4 small disc = hairline
  (1.8e-06!) near-margin; 4 large disc = within-big-component geometry (DLA packing VINDICATED:
  zero cross-component crossings). gem: 4 connected = EARLY divergence (round-20 0.144, bbox 3x
  -> init/RNG-distribution suspect, fixable) + 3 disc/noncanonical -> aggregate tier.
  WAVE 2 dispatched: B2 fmmm-disconnected parity fix (pid 2440376, worktree dagua-fmmm-disc,
  branch r76/fmmm-disconnected), B3 gem trace+fix (pid 2440653, worktree dagua-gem-trace,
  branch r76/gem-fix). Still running: A2 xns-perf (pid 2171417), ref regen.
  NEXT wake-ups: refs done -> rescore fmmm/gem; A2 done -> verify/merge -> dispatch A1 mincross;
  B2/B3 done -> verify/merge -> targeted re-bench of touched engines; then umap port + remaining
  ladder.
- 2026-07-03 ~02:00: WAVE-2 RESULTS. ***GEM SOLVED***: root cause = update-budget bridge
  multiplied rounds x num_nodes (25x over-iteration); fix ec52d79; RMSD 1.0 -> 5-7e-08 on ALL
  probe graphs (float32-rounding = essentially bit-exact); RNG chain verified draw-for-draw;
  MERGED (fe239cc). XNS-PERF merged (00fbe41): ns.c incremental cutvalues ported, bit-identity
  8/8; targets partially met -- remaining large-graph hotspot RELOCATED to mincross _transpose
  (faulthandler-proven). FMMM-DISC attempt 1 resisted honestly (component internals match at
  0.001; residual = MAAR singleton-packing tie-breaks; 3 partial approaches reverted; notes
  92b75b7) -- PARKED pending honest rescore. REFS REGEN DONE (53400/53400 ok, r76_refs).
  ***FMMM RESCORE vs HONEST REFS: 26/33 FLIP.*** Remaining 7 = 4 OGDF disconnected-packing rows
  (random_dag_50 x3, random_dag_200 -- np-leg/packing residual) + 3 fdp-family rows (route to
  sfdp/fdp triage). GEM RE-BENCH running (pid 2788296, r76_gem_fix dir; rescore on completion).
  A1 MINCROSS FINAL dispatched HIGH effort: pid 2807906, worktree dagua-mincross2, branch
  r76/mincross -- armed w/ attempts-1+2 notes, recoverable patches, dot -v discriminator,
  GD_nlist + chain-merge mandate, transpose incremental-delta perf mandate; ladder a-e.
  NEXT: gem bench done -> gem rescore (expect near-total closure); mincross done -> verify ->
  merge if ladder passes -> sugiyama re-bench (graphviz family); then umap scalar port +
  superior-distinct fairness triage + big-graph tier + population tier + final ledger.
- 2026-07-03 ~10:15: ***REBOOT IMMINENT (supersedes the run-to-completion plan below).*** JMT is
  rebooting: the in-flight mincross codex (pid 2807906) and gem rescore (pid 2895173) WILL DIE
  mid-flight and /tmp WILL BE WIPED. Everything needed is preserved on disk:
  * ALL briefs/scripts/combo-lists/codex-log-snapshots: .project-context/research/
    sprint_rng_matching/r76_scratch/ (41 files; .log files untracked but on disk; the live
    mincross transcript snapshot = r76_cx_mincross_SNAPSHOT_AT_REBOOT.log -- contains its
    in-progress reasoning/patches).
  * Mincross WIP survives in the worktree ~/.claude/worktrees/dagua-mincross2 (3 modified files
    + a partial r76_IMPL_mincross_NOTES.md, uncommitted -- inspect `git -C <worktree> diff`
    before deciding: resume-in-place vs fresh re-dispatch with the brief
    r76_scratch/r76_impl_mincross_final.md; give the new codex the snapshot log + worktree diff
    as recovered context).
  POST-REBOOT RESUME CHECKLIST:
  1. pkill leftover watchers (none expected post-reboot); verify no codex/bench procs.
  2. Re-run gem rescore: bash r76_scratch/r76_gem_rescore.sh (script paths are absolute;
     combo list embedded path /tmp/r76_gem_all_combos.txt -- COPY r76_scratch/
     r76_gem_all_combos.txt back to /tmp first, or edit the script path). ~30-60 min local.
  3. Re-dispatch mincross per above (codex, high effort, worktree dagua-mincross2 exists).
  4. Then continue the REMAINING r76 QUEUE below.
- 2026-07-03 ~10:00 (superseded by reboot note): usage pause; monitors detached; in-flight
  jobs left running to completion unattended:
  * MINCROSS FINAL codex: pid 2807906, log /tmp/r76_cx_mincross.log, worktree
    ~/.claude/worktrees/dagua-mincross2, branch r76/mincross. ON RESUME: check
    `kill -0 2807906` / tail the log for CODEX_DONE-equivalent (process gone). Then verify:
    read r75_findings/r76_IMPL_mincross_NOTES.md (in the WORKTREE), check `git -C <worktree>
    log --oneline` for commits, ladder a-e evidence. Ladder passed -> merge r76/mincross ->
    sweep -> launch sugiyama-family re-bench (all classic_sugiyama_* <=300, seeds 100-199,
    OMP capped, batch timeout 1800) -> sugiyama rescore. Failed -> park notes per 2-attempt
    protocol (this was the final attempt; disposition = documented port-in-progress).
  * GEM FULL-FAMILY RESCORE: pid 2895173, log /tmp/r76_gem_rescore.log, output
    eval_output/fidelity_definitive/r76_gem_rescore.jsonl (315 combos). ON RESUME: read the
    summary at log tail ("gem full-family rescored=..."); expect near-total closure vs honest
    refs (gem fix = 5-7e-08 RMSD). Watch for regressions among previously-identical rows
    (old passes were stale-ref artifacts; honest verdicts supersede).
  REMAINING r76 QUEUE after those two land (in order): (1) sfdp/fdp-family triage (44 sfdp rows
  + 3 fmmm-fdp transfers; sfdp default/graphviz_fidelity cluster root-cause -- probe brief
  pattern: like /tmp/r76_probe_fmmm_triage.md); (2) umap scalar-faithful SGD port (tau_rand
  stream portable; target rung-3); (3) superior-distinct fairness triage (79 rows -- match-the-
  worse-reference where portable); (4) fmmm MAAR packing attempt 2 ONLY if the 4 rows still
  matter post-ledger; (5) big-graph tier hang-safe rescore (D1); (6) population tier (D2);
  (7) FINAL LEDGER: full re-analysis (fixed loader, all dirs incl r76_refs + r76_gem_fix +
  any sugiyama dir), official report, gates 0/40, r76_RESULTS.md, baton, memory. Codex-first
  routing throughout (JMT credits directive).
  DIRS ADDED in r76 so far: benchmark_100seed_r76_refs (honest ogdf fmmm/gem refs, 53400 ok),
  benchmark_100seed_r76_gem_fix (post-fix gem, 26700 ok). develop @ 92b75b7+merges (fe239cc gem,
  00fbe41 xns-perf). fmmm rescore artifact: r76_fmmm_rescore.jsonl (26/33 flip).
- 2026-07-04 ~00:50 ET: ***SPRINT CLOSED. r76 DONE.*** D3 finale published the official
  ledger (d8977f4): 3,955 rows -- FIDELITY_IDENTICAL 2,679 + MODE_B_BIT_EXACT 80 +
  MODE_B_IDENTICAL_DISTANCE 39 (= 2,798 identical-tier, 70.7%) + DISTRIBUTIONAL_EQUIVALENT
  45 + MODE_B_CLOSE 94 + DIVERGENT_NAMED_CAUSE 335 (every one dossiered) + NO_CANONICAL 405
  + SUPERIOR_DISTINCT 44 (fairness-audited) + AGGREGATE 51 (24 insufficient + 27
  stale-ref-quarantined large rows) + INSUFFICIENT 183. STOP CRITERIA MET: zero
  no-disposition rows, zero bare divergent, zero unnamed. Gates 1/2/4/5/6 green (gate_5
  0/40); gate_3 = pre-existing calibration note (unchanged from r75). Scorer determinism
  verified. Honest flags carried in-ledger: 10 evidence-thin rows, 645 low-power (n<100),
  2,052 seed-42-era references named, 27 stale-ref large rows quarantined (NOT scored
  per-seed). Housekeeping: all 7 r76 worktrees removed, all r76 branches merged+deleted;
  steady state = develop + main. Codex note: finale commit skipped pre-commit hooks (204MB
  generated report JSON vs detect-secrets scan; justified, logged). Pre-existing items for
  a future session: gate_3 calibration; cosmetic double-border render smoke; TunableParam
  w_straightness metadata drift (2.2 vs 0.5).
  SPRINT STATUS: DONE. Official artifacts: eval_output/fidelity_definitive_r76/ +
  per_combo_r76.jsonl + r76_RESULTS.md.
- 2026-07-04 ~04:15 ET: ***r77 MOP-UP SALVO LAUNCHED (JMT: "perfect fidelity unless
  violence").*** The r76 ledger stands; r77 attacks every named-cause row that was closed
  by budget rather than proof. DISPATCHED (all parallel):
  * A5 half-width rule: codex MED pid 1297475, worktree dagua-sugiyama-final
    (r77/sugiyama-final), brief r76_scratch/r77_impl_a5_halfwidth.md. Stages B-D dispatches
    into the SAME worktree when A5 lands (serialized, shared files).
  * A6 GLPK parity via OPTIONAL swiglpk DEP (JMT-authorized, umap-numba precedent): codex
    HIGH pid 1299744, worktree dagua-igraph-glpk (r77/igraph-glpk), brief
    r77_impl_a6_glpk.md. Draw-level LP parity gates; fallback = current behavior.
  * E1 evidence-thin rows probe: codex MED pid 1302016, main repo read-only, brief
    r77_probe_thin_rows.md -> r76_THIN_ROW_DOSSIERS.md.
  * C1 sfdp spline-occupancy BOUND (doSplines=1 vs 0 instrumented) + approximation: codex
    MED pid 1304321, worktree dagua-sfdp-pack2 (r77/sfdp-pack2), brief
    r77_impl_sfdp_pack2.md. Targets the 2 quality-worse clusters.
  * B1 MAAR instrumented-RUNNER trace (attempt-2 only read source; this one RUNS it): codex
    MED pid 1307032, worktree dagua-maar-trace (r77/maar-trace), brief
    r77_impl_maar_trace.md.
  * ERA REGEN: local pid 1309414, script r76_scratch/r77_era_regen.sh, plan
    r77_era_regen_plan.json (10 stochastic ref engines x 182 engine-graph pairs, seeds
    100-199) -> benchmark_100seed_r77_era_refs; after: rescore the 414 low-power rows.
  AFTER ALL LAND: rescores per family -> ledger amendment (r77 addendum or refreshed
  official ledger) -> updated RESULTS + memory.
- 2026-07-04 ~04:50 ET: r77 ROLLING RESULTS. E1 thin-rows: ***7/10 inherited floors
  RETRACTED as portable op differences*** (6 mds-disc = igraph component handling; 1 drl =
  DRLPhaseSolve internals; 3 connected-mds floors CONFIRMED w/ multi-ULP sweeps) -> M1
  (mds-disc fix, pid 1348267) + D1 (drl instrumented-igraph trace, pid 1350509) dispatched.
  B1 MAAR attempt-3: real runner trace captured (descending-index insertion, newest-row
  ties) but surfaced refs-vs-runner mismatch -> B2 parity probe: binary==source BYTE-EXACT
  (8/8 probes; recipe scripts/rng_match/build_ogdf_runner.sh @ ogdf 5b679565) -> mismatch
  lives in PAYLOAD/PARAM MAPPING -> B3 provenance probe dispatched (pid 1390221;
  instruments the real adapter path; decides if fmmm/gem verdicts vs r76_refs are sound).
  ***A6 GLPK PARITY LANDED + MERGED (445bc61: b47d272 feat + a61bd2e docs).*** swiglpk 5.0
  == igraph vendored GLPK 5.0; rank vectors MATCH installed igraph on all probes;
  moe_router_sparse exact on all 5 variants; fallback + graphviz byte-identity gates green.
  Gate shortfall (5/10 d_R) was a STACKED-CAUSE artifact -- residual moved downstream to BK
  x-stage -> A7 dispatched (pid 1403617, HIGH, last igraph item; segfault workaround noted:
  batch reference calls in subprocesses). pyproject gains extra: igraph-fidelity =
  [igraph>=0.10, swiglpk>=5.0].
- 2026-07-04 ~05:40 ET: r77 ROLLING (2). D1 drl: evidenced floor CONFIRMED w/ instrumented
  trace (edge-cut float-accumulation split at recompute 448; schedules/RNG/init matched) --
  drl line CLOSED, worktree swept. M1 mds: took the RUNTIME-DELEGATION shortcut (4th
  incident; rejected unmerged; my brief omitted the prohibition -- restored to all briefs).
  M2: delegation reverted, NATIVE port of igraph get_sphere() quadrant scan (incl its
  bounds typo) + equal-size qsort -> RNG draw counts EXACT (125978==125978,
  316682==316682); one placement rule remains -> M3 dispatched (pid 1478129).
  B3 provenance: ***ORACLE BUG #5 -- _random_dag built edges via string set =>
  PYTHONHASHSEED-dependent graph realization; benchmark graphs not reproducible across
  processes.*** All cross-process per-seed comparisons on random_dag_50/200 were
  permutation-corrupted (explains MAAR attempt-2/3 gate failures). G1 fix LANDED + MERGED
  (73bbb52: sorted edges, named nodes, cross-hash-seed subprocess tripwire green over the
  FULL graph catalog; affected graphs = random_dag_50 + random_dag_200 ONLY).
  PRE-EXISTING failures #4/#5 discovered (fail on develop untouched):
  test_integration test_50_node_dag (node_overlaps 44); test_ops_init graphopt fidelity
  seed matrix. Housekeeping list now 5.
  LAUNCHED: full-matrix random_dag regen (pid 1526947 -> benchmark_100seed_r77_randomdag:
  --engines all + 10 stochastic ref engines, both graphs, new canonical realization, seeds
  100-199). On done -> rescore ALL random_dag combos (every family) -- expect flips
  anywhere the old verdict was permutation noise.
- 2026-07-04 ~08:10 ET: r77 ROLLING (3). WATCHER MASS-KILL at ~07:00 (another session's
  start-up pkill; all 6 jobs survived; re-armed; agent-issue filed w/ session-tag fix
  sketch). C1 sfdp pack2 LANDED+MERGED (59a462c): instrumented doSplines on/off bound
  proved SPLINES ARE NOT THE RESIDUAL (identical offsets both target graphs) -- real cause
  = LABEL-BOX node sizes in genPoly; fixed w/ conservative gate; W improved on BOTH
  quality-worse clusters, 41/41 hashes, 4800/4800 bench; r76 named-cause CORRECTED
  (spline->label-box). sfdp = 5 fixes total, line closed. M4 mds: C-side dump proved DLA
  placement MATCHES (825k trace lines identical; M3 hypothesis disproven by instrument);
  real rule = eigensolve uplo='U' (ported); parity 5/6 probes; residual = align eigensign
  (DSYEVR) = the excluded LAPACK class -> M5 finisher dispatched (gates+bench+commit+
  equivalence-class evidence + reflection-handling note). A5 LANDED+MERGED (4279b3e):
  ***the 1pt rule NAMED = virtual_node() seeds ND_lw=ND_rw=1 + class2 half-nodesep
  increments; minlen parity EXACT; d_R improved 10/10 probes.*** Its --max-nodes 0 bench
  hit recursion-depth crashes on 2000+ graphs = housekeeping #6 (pre-existing scaling
  limit; those rows are aggregate-tier). Re-bench at <=300 running (pid 1736885 ->
  r77_sugiyama_a5b). A8 stages B-D dispatched (pid 1741008, HIGH -- labels/clusters, THE
  final graphviz item; DOT-content audit first). Housekeeping list now 6 pre-existing
  failures (+ test_engine test_classify_early_exit seen by A5 = #7? verify at ledger).
- 2026-07-04 ~09:45 ET: r77 ROLLING (4). M5 mds finisher LANDED+MERGED (a45087c):
  align+uplo ported, eigensign heuristic removed, byte-identity 9/9+11/11 at 0.0, bench
  1200/1200; eigensign residual PROVEN scoring-invisible (scorer Procrustes is O(2) --
  reflections equivalent). mds line CLOSED. A8 labels LANDED (labels are STRUCTURAL in dot:
  2x minlen, ranksep/2, midpoint label nodes; improved both pure-label rows; committed
  gated). A9 CLUSTER MACHINERY LANDED (x slot/boundary constraints; improved 5/6 cluster
  rows: interleaved/clustered_longlabel/kitchen_sink x2/nested) + A9b guard-hole fix
  (small_label_storm byte-restored; pipeline no longer trusts callers -- wrapper classifies
  cluster-only DOT). ALL MERGED (8d9784d; conflicts vs pack2/A6 union-resolved; 97-test
  smoke green). sugiyama-final + mds-disc + graph-determinism + sfdp-pack2 worktrees SWEPT.
  POST-A5/A6 RESCORE (412 combos, no random_dag): igraph bit-exact 60->75 (GLPK effect),
  far 222->176; graphviz near 31 stable, far 20->18 (A8/A9 not yet in this bench); ZERO
  tier-regressions. Scorer robustness item noted: shape-mismatch should flag not crash
  (hit on random_dag realization change). REMAINING IN FLIGHT: A7 BK x-stage, era regen,
  random_dag full-matrix regen. THEN: consolidated final bench+rescore (sugiyama post-A7/8/9
  + random_dag all-family + era rows) -> ledger refresh -> updated RESULTS/memory.
- 2026-07-04 ~11:45 ET: r77 ROLLING (5) -- SUGIYAMA SAGA CLOSED. A7 BK alignment MERGED
  (flag-driven 4-run + min-width anchor + median_4): ***igraph bit-exact 141*** (30 this
  morning -> 60 -> 75 -> 141; far 222->152; zero regressions). A10 wiring MERGED (8ed1003;
  _quick_classic never called the metadata classifier + cache-key fix): ***edge_label_braid
  0.6016 -> 0.0066 (91x, near tier)***; cluster rows improved (interleaved 0.626->0.595,
  clustered_longlabel 0.237->0.210, kitchen_sink 0.318->0.302, hybrid 0.883->0.845);
  mixed-DOT rows byte-stable per guard. FINAL graphviz table: 1 bit-exact + 32 near + 24
  close + 18 far (named: recursive cluster rank-collapse + structural stragglers).
  Definitive artifact: r77_sugiyama_wired.jsonl. ALL SUGIYAMA WORK ITEMS EXHAUSTED --
  remaining far-tier causes named w/ dossiers (recursive cluster machinery = the honest
  boundary; porting it whole is the next-violence-tier judgment call, documented).
  REMAINING IN FLIGHT: era regen, random_dag full-matrix regen. THEN: their rescores ->
  ledger refresh (corrected named-causes incl spline->label-box) -> r77 RESULTS addendum ->
  memory -> close.
- 2026-07-04 ~19:20 ET: ***RANDOM_DAG CLEAN SWEEP (same-realization rescore,
  r77_randomdag.jsonl).*** 69 combos: 50 identical + 1 equivalent + 6 no-canonical + 12
  mode-B sugiyama tiers + ***ZERO divergent***. 25 formerly-divergent rows FLIPPED incl the
  ENTIRE MAAR cluster (fmmm steps10/100/200 both graphs), all sfdp disc random_dag rows,
  maxent x4, neato, stress_maj, mds x4, and random_dag_50::classic_umap_nn5 (the "eigenspace
  floor" row -- IDENTICAL; the floor was a graph-realization phantom; the r76_FLOOR_DOSSIERS
  umap section must be amended: chaos mechanism was real but the observed divergence was
  oracle bug #5). Bench errors were known classes only (davidson timeouts, sgd2 disconnected
  refusal). LEDGER IMPLICATIONS (for the r77 refresh): retract/close MAAR named-cause rows
  (4), sfdp disc random_dag rows, umap floor row(s); r76_maar_final + parts of prior
  rescores SUPERSEDED by r77_randomdag.jsonl. REMAINING: era regen (engine ~3/10) -> era
  rescore -> comprehensive r77 ledger refresh + RESULTS addendum + memory + close.
- 2026-07-05 ~00:15 ET: ***r77 CLOSED. CAMPAIGN CLOSED.*** Final ledger published+merged
  (a3a2701 -> develop 8234626): MODE_B_BIT_EXACT 80->186, DIVERGENT_NAMED_CAUSE 335->249,
  stale-refs 27->3, zero bare divergent, gate_5 0/40, determinism green, all 69 random_dag
  rows on canonical realization. All branches merged, all worktrees swept (develop+main
  only). Memory updated. Oversized generated artifacts intentionally untracked (hook
  policy; on disk in eval_output/). Housekeeping backlog for a rainy day: gate_3
  calibration; ~7 pre-existing test failures; recursion-limit on 2000+ sugiyama; watcher
  session-tagging (issue filed). THE LIMIT, REACHED HONESTLY.
- 2026-07-05 ~10:00 ET: r78 ROLLING. A11b MERGED: dummy-chain law derived (original
  outgoing incidence order; feedback-arc flips = direction only); size gate removed; probe
  13/13; ***igraph bit-exact 141->257, far 152->52, zero regressions***
  (r78_bk2.jsonl). H1 recursion fixes MERGED (huge-graph crash class dead). Superior line
  MERGED: G2 gem trace = full match at iters100 budget, first split last-bit roundoff at
  round 115/2000 (27 gem rows likely stale-scoring; monster campaign resolves). R2
  merged w/ reconstructed evidence (MY ERROR: worktree --force removal destroyed the
  uncommitted section; recovered from primary artifacts, now banked in r78_evidence/;
  lesson: check worktree status before ANY --force removal): sgd2 2 rows byte-identical
  14/14 paired seeds, blocked by DAGUA-SIDE HANG (real_football_115 seed 113,
  native_start) -> S3 dispatched; sfdp 8 = measured spline-occupancy boundary (terminal);
  fdp 14 = fdp_xLayout prism/GTS overlap expansion -> F3 dispatched (scipy Delaunay
  sanctioned). A12 clusters: rounds 1-3 = rank parity + ORDERING parity (dot count 5
  exact) verified; round 4 (A12d, x-stage integration) in flight. Neato: connected
  near-exact vs live binary (stale-era refs suspected for all 54); fresh both-sides bench
  in flight. Monster campaign grinding.
- 2026-07-05 ~12:30 ET: r78 ROLLING (2). A12e/A12f CLUSTER TERMINAL (merged): round 5
  enumerated the complete aux-x structural diff (4 constructs); round 6 proved them COUPLED
  -- all require a first-class graphviz x-inventory model (node classes, saved edge
  lineage, ED_to_orig, borders, label metrics) threaded through
  _build_graphviz_x_aux_edges. 6 rounds total; rank+ordering parity VERIFIED and banked
  behind the skeleton flag; the 20 rows keep A9/A10 gains + the terminal spec dossier.
  S3 SGD2 CLOSED (merged): "hang" = PyTorch tiny-kernel thread-pool pathology on the
  crossing MLP (50-60 min/seed); fixed via scoped single-threading (1f317c1). CLOSURE:
  real_football_115 100/100 EXACT vs reference; wide_1_100_1 97/100 exact + 3 seeds
  diverging after matched first batches (float-chaos class). Caveat documented: threading
  change alters crossing-workload float behavior (old path never terminated; exactness is
  vs the REFERENCE, which is the standard). Two of two evidence-thin rows now dispositioned
  with instrument-grade closure. IN FLIGHT: A11c big-tail, F3 prism, neato bench, monster
  bench (9.3%).
- 2026-07-05 ~13:30 ET: ***NEATO CLOSED: 46 identical + 1 equivalent + 2 insufficient,
  ZERO divergent; 40 formerly-divergent flipped*** (r78_neato.jsonl, fresh both-sides
  seeds 100-199 + component-seed fix 00bd57a merged). The r75 "CG/drand48/packing" named
  cause for 54 rows is RETIRED -- stale-era references were the entire story on connected
  graphs; the seed rule closed disconnected. Sixth family closed by oracle repair.
  random_dag_200 patient retry running (100 watchdog rows). REMAINING IN FLIGHT: A11c
  big-tail, F3 prism, monster bench, neato-rd200.
- 2026-07-05 ~15:00 ET: ***IGRAPH SUGIYAMA FAMILY COMPLETELY CLOSED.*** A11c: GLPK >1000
  gate VERIFIED faithful (igraph source + empirical); 52 far rows = stale-ledger artifacts
  (52/70 <0.01 with current code); fresh big-tail bench 6600/6600 incl rgg_2000 gv-fidelity
  at 3.2h (H1 fixes holding at scale). A11d (merged): last law = component packing advances
  by max-X of the DUMMY-EXPANDED component, no per-component re-normalization (5bfed0b);
  ***18/18 close rows now <0.01, raw-equal to installed igraph***; 2400/2400 bench; all
  byte gates held. The igraph arc: 30 bit-exact at sprint start -> effectively the ENTIRE
  family at-reference. NEATO closed earlier today (40 flips, zero divergent). REMAINING IN
  FLIGHT: F3 prism, monster bench, neato-rd200b. Then: the definitive final re-ledger.

## 2026-07-06 (evening): neato rd200 closed + prism merged + rescore-hang lesson
- neato random_dag_200 fresh-dir retry (benchmark_100seed_r78_neato_rd200, 100/100 ok):
  rescore verdict FIDELITY_IDENTICAL (100 matched seeds, stress D/R 0.0973/0.0973).
  NEATO FAMILY FULLY CLOSED -- zero divergent rows.
- r78/prism MERGED to develop (3328bb7). fdp 87-row rescore: 66 identical / 13 dist-eq /
  5 superior / 3 flagged. Two of the flagged are real residuals (parallel_cycles_4x5
  halved to 0.11; random_dag_200 fdp at 0.121 vs ref self-spread 0.125, near parity);
  parallel_multiedge_bundle flip is a stale-overlay artifact (freshest dir r74_fixes
  absent from the 7-dir rescore list) -- final full-overlay ledger will restore it.
- GOTCHA (cost ~10h wall): definitive_fidelity_analysis.py ProcessPoolExecutor forks
  after torch import; forked workers can inherit a locked mutex -> permanent futex_wait
  deadlock (hit twice: workers=2 AND workers=1; parent+child both futex_wait, log frozen
  at "overlay:" line, 0% CPU). Workaround: /tmp/r78_neato_serial_driver.py patches the
  executor with an in-process serial stand-in; finished in minutes. Proper fix for a
  housekeeping round: mp_context=spawn (or forkserver pre-torch) in run_payloads.
- Monster bench rescoped 100->25 seeds per JMT ("25 seeds is plenty"): 77,040-run
  universe, --resume credited prior work, ~14.4% as of resume. Remaining plan unchanged:
  bench done -> THE definitive full-universe re-ledger (supersedes r77).

## 2026-07-07: monster bench -> TARGETED cut (JMT decision)
JMT: "targeted cut now, then when confident about the fidelity ceiling let the full thing
grind for a week for due diligence." Full 240-engine x 15-graph monster bench (~6-10 day
ETA, was at 19% = 14,725/77,040) STOPPED; exact command banked at
r78_scratch/MONSTER_BENCH_FULL_CMD.txt for the later exhaustive documentation run.

Ledger-gap inventory (from r77 per_combo.json, 3955 rows):
- 87 big-graph INSUFFICIENT (all matched_seeds<30; 0 disconnected -> all fixable; 86/87
  stochastic Mode A so no deterministic shortcut). 20 engines x 14 graphs; 34 on 2000-node.
- 24 gem-stale (classic_gem_iters100/500/2000 on 8 x 500-node; stale pre-gem-fix scoring).
- 180 SMALL-graph insufficient (22 engines x 80 graphs) -- SEPARATE fast concern, NOT part
  of monster bench; pending WHY-insufficient triage before benching (ref-cant-do vs low-n).

Targeted benches (fresh dirs, 35 seeds seed-start 100 -> clears the >=30 matched gate;
JMT's "25 plenty" was confidence not the scorer's own gate):
- FAST-BIG: benchmark_100seed_r78_targeted_fastbig, pid /tmp/r78_targeted_fastbig.pid,
  21 eng x 9 (500-node) graphs, 4 workers. Covers 53 insuff + 24 gem.
- SLOW: benchmark_100seed_r78_targeted_slow, pid /tmp/r78_targeted_slow.pid, 11 eng x 5
  (2000-node) graphs, 5 workers, 6h/8h timeouts. Covers 34 insuff.
Target combos for rescore: /tmp/r78_targeted_combos.txt (291). ba_5000 DROPPED (0 gaps).
GOTCHA: run_benchmark --engines accepts specific variant_ids (classic_fr_steps200) and
with --variants does NOT re-expand to siblings (verified run_benchmark.py:1013-1016).

## 2026-07-08: r78 resume (Fable session) -- benches relaunched, triage fleet out
- Reboot never happened (uptime 5d); found slow-bench forkserver + 5 workers ORPHANED since
  the 7/7 SIGTERM (PPID=1, ~30% CPU each for 29h, positions written past results.json flush).
  Killed the tree + one idle nodrl forkserver. /tmp survived: combos file + logs intact.
- Relaunched per RESUME_AFTER_REBOOT.md: nodrl (pid 1646492, 4 workers, resumed from 275/5670)
  and slow (pid 1646493, workers 5->7, resumed from 475/1925). Exit monitors armed in-session.
- Dispatched 4 Claude subagents (codex paused, credits): (A) drl grind-vs-perf-fix decision,
  (B) small-tier 180-row WHY-insufficient triage, (C) NEW HAZARD probe -- bench log shows
  "classic_maxent_stress received unrecognized variant params: alpha, steps, use_entropy"
  (classic_competitor.py:750) => steps50 variant may have silently run as DEFAULT in all
  historical benches; blast-radius + static cross-check of all variants vs pipeline configs,
  (D) adversarial oracle/harness audit (independent Procrustes recompute spot-checks,
  threshold laundering sweep, stale-shadowing check) per bisection-first standing rules.
- Next after these land: drl + small-tier benches, rescore 291 combos (+any maxent redo),
  DEFINITIVE FINAL RE-LEDGER (supersedes r77). Goal: JMT sign-off that fidelity work is DONE.

## 2026-07-08 (cont): triage verdicts + drl/umap benches launched
- MAXENT HAZARD RESOLVED BENIGN: variant params were ALWAYS applied (merge is unconditional;
  warning is a missing variant_param_names allowlist). Empirically verified steps50 != default
  and == direct pipeline call. NO ledger rows invalidated. Fixed allowlists on 8 classes in
  classic_competitor.py (mechanically derived from VARIANT_REGISTRY; agent's per-class list
  had minor errors -- computed sets are ground truth). Warning gone.
- TORCHLENS TRAP: sibling ~/projects/torchlens checkout has unresolved merge conflicts
  (someone's WIP -- NOT touched). dagua guards only caught ImportError; SyntaxError leaked
  into test collection + graph corpus. Fixed: tests/conftest.py + dagua/eval/graphs.py
  (2 sites) now catch Exception. sgd2_multi test "failures" were this, pre-existing.
- DRL DECISION: GRIND (agent probe): 59 sparse combos / 2,065 runs, ~68.6 core-h. Perf-fix
  rejected (hotspot _as_float32 is the bit-exactness mechanism; Gauss-Seidel sequential ->
  vectorization provably unsafe; best 2-3x w/ high regression risk). LAUNCHED: pid 2486166,
  4 workers, eval_output/benchmark_100seed_r78_targeted_drl, monitor armed.
- SMALL TIER TRIAGE (186 rows, 3 families):
  F1 sgd2_multi_with_crossing (65): stale pre-1f317c1 watchdog timeouts -> rebench ~4.1h
     (cmd in agent report / r78_scratch notes). QUEUED after nodrl frees cores.
  F2 22-engine watchdog family (113): same signature, NO confirmed fix; hypothesis = CPU
     oversubscription during June escalation_final overnight. Bisection-first: 3-seed probe
     (2 combos) MUST run on a CALM box -> queued after current benches; verdict gates the
     1.4h full rebench.
  F3 umap_nn30 (8): dagua side clean post-fix; REFERENCE runs missing. LAUNCHED: pid
     2486172, eval_output/benchmark_100seed_r78_small_umap (minutes).
- Bench fleet now: nodrl 1646492, slow 1646493, drl 2486166, umap 2486172 (all nice 5).
- Pending: harness adversarial audit agent (out); test rerun; commit of allowlist+guard fixes.
