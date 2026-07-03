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
- 2026-07-03 ~10:00: ***USAGE PAUSE (JMT out of Claude credits).*** Monitors DETACHED; in-flight
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
