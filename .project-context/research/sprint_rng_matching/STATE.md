---
run: sprint_rng_matching
created: 2026-06-02
state: FINAL_ALLGRAPHS_RUNNING_v2
# PHASE 1 (closing wave) + PHASE 2 (verify+commit efe6290) DONE. PHASE 3 launched:
# all-graphs relaunch wrapper PID=3453115, log=/tmp/rng_final_allgraphs.log, watcher=bg0s0apvi
# (bg-watch label rng-final2, re-armed --max-runtime-min 300 after a benign 2h watcher-cap TIMEOUT;
# run was 65.8% at 02:16 and progressing fine). --resume reuses ~58,973 valid rows. On its DONE -> PHASE 4:
# NOTE: drl on big graphs (ba_500 etc.) hits per-combo layout timeouts -> recorded as ERROR/skip, benign.
#
# 2026-06-03 ~07:00 UPDATE -- GIANT-GRAPH TAIL BOTTLENECK:
# The all-graphs run slowed hard in its tail: 65.8% (02:16) -> 71.6% (06:59), ~1.2%/hr. Remaining
# ~28% is concentrated in 3-4 GIANT graphs (ba_5000, small_world_2000, grid_50x50: ~194/777 rows;
# the 2000-node graphs are nearly done). At this rate the tail is ~20h -- NOT "overnight", and it is
# the LEAST informative part (every engine cascades on 2000-5000 node graphs via chaotic FP; that's
# the EXPECTED result, already known). The informative classification (105 graphs up to 2000 nodes)
# is essentially complete.
# ACTION TAKEN: launched a PARTIAL classification (does not disturb the live run): consolidate current
# .pt -> positions_partial.h5 + report -> eval_output/fidelity_report_partial (watcher bihxeyqu0).
# Main benchmark left RUNNING (watcher b2z07vfdf). DECISION FOR JMT (morning): (a) let giant tail
# finish ~20h, (b) kill giant graphs + finalize on the 105 done graphs now, or (c) cap graph size.
# Do NOT unilaterally kill the run (valid data) -- present the partial classification + ask.
#
# 2026-06-03 ~07:06 -- PARTIAL CLASSIFICATION DELIVERED + TEXTED JMT. Wrote ALLGRAPHS_SUMMARY.md
# (3-tier result: ~28 SCALE-ROBUST machine-eps incl 2000-node / TIER-2 small-exact-cascade-at-scale /
# TIER-3 walls). report at eval_output/fidelity_report_partial/report.md. Main run STILL RUNNING
# (watcher b2z07vfdf; giant-graph tail ~20h). AWAITING JMT DECISION (kill tail / let finish / cap size).
# WAKE-UP ROUTING from here:
#   - JMT says kill/done -> kill wrapper 3453115 process-group, finalize ALLGRAPHS_SUMMARY as final
#     (drop "partial" caveat), optionally file-for-review, state=DONE.
#   - JMT says let finish / cap -> act accordingly; on b2z07vfdf DONE, run final report, refresh summary.
#   - b2z07vfdf fires DONE before JMT replies -> run final consolidate+report (script does this itself),
#     refresh ALLGRAPHS_SUMMARY with full-data numbers, text JMT, state=DONE.
#   - Also still TODO whenever convenient: regenerate small-graph STATUS.md cleanly (was clobbered;
#     restored to HEAD pre-commit) so it shows 76 -- low priority, SUMMARY.md is authoritative.
#
# ============================================================================================
# FULL 4-TIER PIPELINE (JMT directive 2026-06-03 ~mid-morning -- the home stretch)
# JMT: "keep cooking the 5 seeds; when finished do the 100 seed for all (graph/algo) combos that
#       are NEITHER a perfect match NOR a timeout, then fidelity analysis with TOST -> four-tier
#       categorization of all algos."
# This is PHASE 5-8, triggered when the 5-seed sweep (watcher b2z07vfdf) completes. The 5-seed run
# writes eval_output/benchmark_5seed_final + eval_output/fidelity_report_final/report.md.
#
# THE FOUR TIERS (per (graph,algo) combo; aggregate to per-algo after):
#   Tier 1 BIT_IDENTICAL          -- 5-seed max per-seed RMSD < 1e-3 (really <1e-7; bit-exact). DONE, no 100-seed.
#   Tier 2 TIMEOUT                -- combo dominated by timeout/error (no ok pairs to verdict). Excluded.
#   Tier 3 STATISTICALLY_EQUIVALENT -- not bit-identical, ran, STOCHASTIC, 100-seed TOST PASSES (equivalent).
#   Tier 4 STATISTICALLY_DIFFERENT  -- not bit-identical, ran, and EITHER 100-seed TOST FAILS (stochastic)
#                                      OR the engine is DETERMINISTIC (no seed distribution -> TOST N/A ->
#                                      a deterministic difference is simply tier 4; do NOT waste 100 seeds
#                                      on deterministic engines -- classical_mds/sugiyama/spectral/pivot_mds/rt).
#
# *** SCOPING RULE (HARD-LEARNED -- the P3 over-escalation incident, JMT was furious) ***
#   The 100-seed runs ONLY on the specific (engine, graph) combos that are non-bit-identical AND
#   non-timeout AND stochastic. NOT whole engines on all graphs. NOT all combos. Per-engine, restricted
#   to THAT engine's failing graphs. scripts/r69_p3b_targeted.py already does exactly this (reads
#   /tmp/r69_failing_map.json = {engine:{"ref":refname,"graphs":[failing...]}}). USE p3b, NOT
#   r69_p3_100seed_tost.sh (that one runs escalation engines on all graphs = the bug).
#
# PHASE 5 = TRIAGE (when b2z07vfdf DONE): adapt scripts/r69_triage.py to read the _final dirs
#   (benchmark_5seed_final/results.json + fidelity_report_final/per_variant.json + variant registry
#   is_stochastic+reference). Output: per-(graph,algo) tier1/tier2/escalate/tier4-deterministic, AND
#   the failing-map JSON for p3b ({engine:{ref,graphs}} for the stochastic-non-bitexact-non-timeout set).
#   SANITY-GATE before launching 100-seed: print the escalation combo COUNT + the engine x #graphs
#   breakdown; if it looks like "all graphs" or thousands of combos, STOP and recheck the scoping.
# PHASE 6 = TARGETED 100-SEED: scripts/r69_p3b_targeted.py against the failing map (escalation output
#   dir e.g. eval_output/benchmark_100seed_escalation_final). MATCHED PARAMS + MATCHED SEEDS + NO
#   DELEGATION still apply (see [[feedback_always_parameter_match_comparisons]],
#   [[feedback_no_runtime_delegation_to_reference]]). These complete fast (they finish, just differ by basin).
#   Run via bg-watch (NOT pgrep -f). Watch for completion.
# PHASE 7 = TOST: fidelity_analysis.py / r68_tost_followup.py on the 100-seed data -> per-combo TOST
#   verdict (equivalent vs different) at matched params. -> tier 3 vs tier 4.
# PHASE 8 = ASSEMBLE 4-TIER: combine tier1 (5-seed bit-exact) + tier2 (timeouts) + tier3 (TOST-equiv) +
#   tier4 (TOST-diff + deterministic-diff) into the final categorization. Write FOUR_TIER_CATEGORIZATION.md
#   (per-algo + per-(graph,algo)), refresh ALLGRAPHS_SUMMARY, file-for-review if human-worthy, TEXT JMT
#   the four-tier result. state=DONE.
#   REPORT DIMENSION (JMT 2026-06-03): tag each GRAPH directed/undirected (+ DAG/tree vs cyclic, i.e.
#   "has a natural hierarchy") and each ALGORITHM hierarchy-requiring (sugiyama, reingold_tilford) vs
#   undirected-native (force/stress/spectral/MDS). For every DIVERGENT (algo,graph) combo, annotate
#   whether it is a DOMAIN MISMATCH -- a hierarchy-requiring algo run on a no-natural-hierarchy
#   (undirected/cyclic/dense) graph, where the algorithm must INVENT an arbitrary layering -> expected,
#   benign divergence (e.g. sugiyama on petersen/K5/wheel; sugiyama is bit-exact 1e-16 on DAGs/trees).
#   This separates "expected out-of-domain divergence" from "genuine divergence" in the final tables, so
#   a reader sees WHY a combo diverges, not just that it does.
# Anti-flail: if triage scoping looks wrong, FIX scoping before running -- never re-run the over-escalation.
#
# ============================================================================================
# PARALLEL WORKSTREAM (JMT 2026-06-03): EQUIVALENCE METRICS for the Tier-4 deterministic group.
# JMT chose "full trio": automorphism-aligned Procrustes + stress-equivalence + spectrum/distance
# diagnostic -- to SHOW practical equivalence where coordinate-RMSD over-penalizes deterministic/
# symmetric holdouts (sugiyama, classical_mds, pivot_mds, spectral_random_walk).
# Dispatched codex pid=3672824 (watcher bzphktmpf, log /tmp/equiv_metrics.log), spec at
# ./SPEC_equivalence_metrics.md. Builds dagua/eval/equivalence_metrics.py + scripts/equivalence_report.py
# + tests. NEW files only -- does NOT touch run_benchmark/competitors/variants (the live 5-seed run).
# ON CODEX DONE: review (key result = sugiyama/classical_mds plain-rmsd vs aut-rmsd / dist-corr before-
# after), verify numbers + anti-cheat (igraph for automorphism-analysis is OK; no LAYOUT delegation),
# ruff/mypy/pytest, COMMIT (no AI attribution). This metric becomes the equivalence verdict for the
# Tier-4-DETERMINISTIC engines in PHASE 8 (they skip 100-seed; equivalence shown via this trio instead).
# Idea credit: JMT proposed label-permutation -> I refined to AUTOMORPHISM-group-restricted (free perm
# = NP-hard QAP + over-permissive false-equivalences).
#
# EXTENSION QUEUED (JMT 2026-06-03): after the trio codex (bzphktmpf) completes + I commit it, dispatch
# a SECOND codex (sequential -- SAME module, NO concurrency) to add the final two invariances:
#   (4) per-connected-component rigid placement, (5) per-axis anisotropic scaling OPT-IN for free-aspect
#   engines (allowlist default {classic_sugiyama}). Spec = the "FOLLOW-UP ADDITIONS" section of
#   ./SPEC_equivalence_metrics.md. These COMPLETE the invariance criteria (principled ceiling:
#   rigid + automorphism + degenerate-eigenspace + per-component + per-axis-optin; anything further
#   launders real differences). Then commit. THIS is the final equivalence toolkit for PHASE 8 Tier-4.
#
# 2026-06-03 -- EQUIVALENCE TOOLKIT COMPLETE. Trio f9d18e1; extension (per-component + per-axis opt-in)
# committed [this commit]. 6/6 tests, ruff/mypy/anti-cheat clean. All five invariances live in
# dagua/eval/equivalence_metrics.py + scripts/equivalence_report.py.
# KEY FINDING (for PHASE 8): sugiyama/petersen does NOT collapse under ANY invariance (plain 0.845,
# automorphism 0.600, anisotropic 0.667, rotation floor ~0.53) -> genuinely different valid layerings,
# NOT an invariance artifact. This is the case the QUALITY axis (stress/crossings = equally-good drawing)
# is for. So PHASE 8: Tier-4 deterministic engines get an equivalence verdict from EITHER axis --
# invariance-equivalent (pivot_mds: collapses) OR quality-equivalent (sugiyama: equal stress).
# CAVEAT: benchmark_5seed_final sugiyama positions are STALE pre-closing-wave (--resume reused them) ->
# re-run equivalence_report on FRESH post-wave sugiyama positions before the final Tier-4 verdict.
# verify report.md non-empty, classify (graph,algo), write ALLGRAPHS_SUMMARY, text JMT, state=DONE.
# Closing-wave outcome: 74->76 bit-exact (spectral_unnormalized + rt_horizontal refs); sugiyama
# 0.93->0.37; classical_mds/drl/davidson ceilings confirmed; fmmm+sgd2 reverted (no gain).
goal: RNG-stream-match dagua's fidelity ports to their references so SMALL graphs are
      bit-identical (per-seed Procrustes RMSD < 1e-7) at MATCHED seeds, for as many of the
      24 algorithms as physically possible. Stop at <1e-7-all OR documented can't-go-further.
      Then TEXT JMT to talk. Do NOT run the full all-graphs sweep (that's a later, separate
      spot-check JMT will direct).
---

## OVERNIGHT CHAIN (JMT 2026-06-02 ~23:27 "do this autonomously overnight so I can sleep")
JMT directive: cancel the all-graphs sweep (DONE -- killed pid 3291157 clean), run ONE more
targeted "close what's closable" wave, then RE-LAUNCH the all-graphs sweep, all autonomously.
TEXT JMT only at meaningful boundaries (closing wave done + relaunch; all-graphs done). Let him sleep.

### Phase ladder (this is the authoritative wake-up routing for THIS run)
PHASE 1 = CLOSING_WAVE_RUNNING  <-- current
  6 codexes (medium effort), each owns DISTINCT files, dispatched 23:28, watchers armed:
    sgd2multi 3358698 (sgd2_multi.py -- real epoch-shuffle RNG, the one REAL closable gap)
    clmds     3358821 (classical_mds.py -- cheap scipy dsyevr try, else document metric-artifact)
    sugiyama  3359112 (sugiyama.py -- deterministic tie-break ordering)
    fmmm      3359317 (fmmm.py -- deterministic OGDF integer-packing)
    anneal    3359529 (_igraph_rng.py + drl.py + davidson_harel.py -- trace diverging cases)
    refs      3359733 (eval/competitors/* + variants.py -- add missing spectral/rt references)
  Logs: /tmp/rngc_<name>.log. Specs: ./C_<name>.md.
  WAKE-UP ROUTING while in PHASE 1:
    - CODEX_DONE for one codex -> ack mentally, do NOT act until ALL 6 terminal. Just note rc/commit.
    - CODEX_FAILED/TIMEOUT -> note it; that engine keeps its prior status (anti-flail: do NOT redispatch
      more than ONCE, and only if the failure is a trivial/recoverable error, not a documented wall).
    - When ALL 6 are terminal -> go to PHASE 2.

PHASE 2 = RE-VERIFY + COMMIT (do this yourself, inline -- it's mechanical)
  a. Re-run the small-graph bit-exact harness to measure what actually closed:
     export LD_LIBRARY_PATH=/home/jtaylor/anaconda3/envs/py311/lib:$LD_LIBRARY_PATH
     python scripts/rng_match/check_engine.py <engine>  for each touched engine
     (sgd2_multi_default/batch128/lr001, classical_mds_default, sugiyama_default,
      fmmm_steps100, drl_default, davidson_harel_rounds100, spectral_random_walk, ...).
     OR the full harness: python scripts/rng_match/bitexact_harness.py (regenerates STATUS.md).
  b. ANTI-CHEAT grep each touched pipeline diff for delegation (import igraph / subprocess dot /
     fa2util / eval.competitors). Reject + note any that delegate (do NOT commit a cheat).
  c. Update STATUS.md + SUMMARY.md headline counts with the NEW bit-exact total.
  d. Commit (one commit, conventional, NO AI attribution). Use SKIP=detect-secrets only if the
     ONLY diff is generated_at timestamp churn (verified benign). status.json/binaries gitignored.
  e. TEXT JMT: "closing wave done: was 74 bit-exact, now N. Closed: <list>. Walls confirmed: <list>.
     Relaunching all-graphs now." via ~/.claude/scripts/send-to-jmt.sh
  f. -> go to PHASE 3.

PHASE 3 = RELAUNCH ALL-GRAPHS (the final classification sweep, now with closed gaps)
  bash scripts/rng_match/final_allgraphs_5seed.sh   (run via bg-watch.sh, NOT pgrep -f)
    -> wrapper writes /tmp/rng_final_allgraphs.log ; --resume reuses completed benchmark rows.
    capture wrapper PID, arm bg-watch.sh <PID> /tmp/rng_final_allgraphs.log --label rng-final2.
  state -> FINAL_ALLGRAPHS_RUNNING_v2.
  The script itself sends a completion text (line 48) + writes eval_output/fidelity_report_final/report.md.

PHASE 4 = ALL-GRAPHS DONE (wake on rng-final2 DONE)
  a. Verify it truly finished (eval_output/fidelity_report_final/report.md exists + non-empty).
     If watcher fired on a kill/crash not completion, check log tail; redispatch with --resume once.
  b. Classify every (graph,algo): confirm hypothesis -- matched-seed bit-exact engines stay
     bit-exact except chaotic-FP-cascade on big graphs; the 36 walls diverge regardless.
  c. Write the all-graphs summary (which graphs/algos diverge + why) into SUMMARY.md / a new
     ALLGRAPHS_SUMMARY.md. File the report for review if human-worthy.
  d. TEXT JMT the final result. state -> DONE.

### Anti-flail / fallback (overnight)
- A codex failing its engine is FINE -- that engine keeps prior status, document the wall, move on.
  Do NOT chase the documented ceilings (sfdp libm-chaos, gem_iters2000) -- they were NOT dispatched.
- Codex quota exhaustion mid-wave -> the already-running codexes finish; for any not-yet-run work,
  pivot to an Agent (general-purpose, model opus) OR just proceed with what closed. Don't block the chain.
- 3 rounds same un-closeable -> accept residual (we already documented these walls once).
- Session pause/restart -> re-read THIS block, check `ps -C codex` + git log + the logs, resume at the
  right phase. Watchers: bzkz0s7lo bznmp6kzm bvadr6qdq bwia1efqo bnb826zlm b22likc8y (phase-1 codexes).

# RNG-Stream-Matching Sprint -- State (single source of truth lives in STATUS.md)

## Why this sprint (JMT directive 2026-06-02)
Prior "bit-exact everything" was an over-claim: only ~6 families (fa2/tsnet/lgl/graphopt/
linlog/rt) were truly bit-exact; the graphviz/igraph/ogdf force layouts were only
TOST-equivalent and the "BIT-EXACT vs instrumented graphviz" results were ~1e-5 on 4 tiny
graphs vs a /tmp build that's now gone. Full reconciliation: ../sprint_algo_fidelity/WHERE_WE_STAND.md.
JMT wants a REAL attempt at RNG-stream matching, with ACCURATE RECORDS this time.

## Scoping decisions (CC, can be revised when JMT and CC meet)
1. REFERENCE = a single PINNED graphviz 7.0.5 build, rebuilt permanently + logging-only
   instrumented, PROVEN identical to stock 7.0.5 output (veridical). The benchmark/harness
   compares against THIS SAME binary (trace-build == reference-build) to kill build-level
   tie-break drift. Location: ~/tools/graphviz-7.0.5-instr/ (NOT /tmp). Build script committed.
2. Other refs matched via their seeded RNG read from source: igraph (its RNG), OGDF (srand/
   C++ RNG), fa2 (python), sklearn (numpy RandomState), networkx (numpy), umap-learn.
3. BAR (JMT refined 2026-06-02): SUCCESS/ACCEPTANCE = per-seed Procrustes RMSD < 1e-7 on
   SMALL graphs at MATCHED seeds {1,2,3}. But the TARGET is the float64 NOISE FLOOR
   (~1e-13 to 1e-15) -- "squeeze all the juice": do NOT stop a float64 engine at ~1e-7 if it
   can be driven lower; a perfect RNG+arithmetic-order match in float64 should reach ~1e-15.
   1e-7 is the floor only for genuinely float32 paths (float32 eps ~1.2e-7). Per-engine, push
   to its true noise floor and RECORD the actual magnitude achieved + dtype + reference output
   precision (so "stopped at the noise floor" is always backed by the number).
   Reference run at seed=N (graphviz: -Gseed=N -Gstart=N, adapter supports it; use the
   instrumented %.17g build for full-precision reference coords), dagua replicates seed=N's
   exact RNG generator.
4. SMALL graphs = curated tiny fixtures (path, star, cycle, grid, complete, small tree,
   small random/bipartite; ~6-20 nodes; ~12-15 fixtures). Small enough that chaotic FP
   cascade is negligible, so a clean RNG+arithmetic match should hit <1e-7.
5. STOP: all algos < 1e-7 OR documented irreducible reason per engine. No full-suite run.

## ANTI-CHEAT (enforced on every P1 port -- JMT explicit 2026-06-02)
Two forbidden cheats:
1. RUNTIME DELEGATION: no import/call of the reference package or binary from
   dagua/layout/ops/ (no `import igraph`, no subprocess to dot/neato/sfdp, no fa2util, no
   calling dagua/eval/competitors/*). Common Python libs (numpy/scipy) OK. See
   [[feedback_no_runtime_delegation_to_reference]].
2. RNG-REPLAY (subtle, sprint-specific): the port must reimplement the RNG GENERATOR
   (e.g. graphviz drand48, igraph RNG) from SOURCE so it produces the same stream for ANY
   seed -- it must NOT read/replay the reference's actual emitted random values from the
   instrumented trace. The instrumented build is DEBUG-ONLY (locate divergence), never feeds
   dagua's output.
ENFORCEMENT: (a) every port spec states both rules; (b) CC greps each port diff for
delegation signatures before accepting; (c) multi-seed {1,2,3} verification is the built-in
detector -- a generator reimpl matches all 3 seeds, a replay cheat matches only its captured
seed. STATUS.md records, per engine, WHICH generator was reimplemented from WHICH source.

## INVARIANT -- MATCHED PARAMS (JMT emphatic 2026-06-02, see [[feedback_always_parameter_match_comparisons]])
EVERY analysis going forward MUST compare at matched parameters: the REFERENCE runs the SAME
config (iters/rounds/steps/perplexity/etc.) as the dagua variant, NOT reference defaults.
Also matched: seed (both seed=N), dtype (float64), graphviz build (instrumented==stock).
GUARANTEE MECHANISM:
- variants.py original_params mirrors reimpl_params (param-matching codex 2186380 fixing this).
- adapters pass params through to the reference call (graphviz -G / igraph kwargs / OGDF runner / etc.).
- HARNESS GUARDRAIL (must add to scripts/rng_match/bitexact_harness.py): record BOTH sides'
  effective params per row; FLAG/refuse any pair where reference ran defaults or params don't
  correspond -- never emit a "clean" RMSD for a mis-parameterized comparison. (Add this in the
  post-param-matching verification step; dispatch a tiny harness-guardrail codex if needed.)
- NEVER conclude an engine diverges without confirming matched params + seed first.
gem proved it: looked "divergent 1.15", was bit-exact 3.86e-13 at matched rounds.

## Phases
- P0 SETUP (parallel): (A) permanent instrumented graphviz 7.0.5 + prove logging-only/veridical
  + reproducible build script; (B) small-graph matched-seed bit-exact harness + fixtures +
  STATUS.md table (the single source of truth). CC verifies B on a known-bit-exact engine
  (fa2 -> ~0) and a known-divergent one (sfdp default -> high) before trusting it.
- P1 PORTS (parallel, many codex): per-engine RNG-stream match. Each codex: (1) read the
  reference's RNG + algorithm from source, (2) replicate exact RNG seq + arithmetic order in
  the dagua fidelity path, (3) verify <1e-7 via the harness on small graphs at seeds {1,2,3},
  (4) if stuck, use the instrumented build to trace the first diverging step. Update STATUS.md.
- P2 ITERATE: re-dispatch the engines not yet <1e-7 (one retry each, with trace evidence).
  Anti-flail: 3 rounds same engine no progress -> document irreducible reason, accept.
- DONE: STATUS.md shows every engine BIT_EXACT(<1e-7) or documented-stuck. Text JMT.

## Wake-up case routing
| Observable | Action |
|---|---|
| codex alive | yield, watcher will fire |
| codex DONE + STATUS.md engine now <1e-7 | mark done in STATUS; dispatch next engine or P2 |
| codex DONE but engine still >1e-7 | read trace; one targeted retry; else mark stuck |
| codex quota | (JMT upgraded -- should be fine) fall back to Opus Agent if hit |
| all engines BIT_EXACT or stuck | run DONE procedure |

## Records discipline (JMT: "keep accurate records so we don't get confused again")
- STATUS.md = THE per-engine table, updated after EVERY engine result. Columns:
  engine | reference | RNG source matched | best max-RMSD (small graphs, seeds 1-3) |
  verdict (BIT_EXACT<1e-7 / CLOSE / STUCK) | evidence file | notes.
- Every claim cites the harness output file. NO claim without a harness number. NO /tmp.
- Per JMT rule [[feedback_verify_against_reference_or_dont_claim]]: per-seed Procrustes only;
  no aggregate/TOST hand-waving; re-verify, don't trust commit messages.

## DONE procedure
1. Finalize STATUS.md (every engine classified, with numbers + evidence paths).
2. Write SUMMARY.md (what hit <1e-7, what's stuck + why).
3. Text JMT: "RNG-matching sprint done. N/24 algos bit-exact <1e-7 on small graphs; M stuck
   (reasons). Ready to talk + then spot-check before the full sweep."
4. Do NOT launch the full all-graphs sweep -- JMT directs that next.

## Log
| Round | Phase | engines dispatched | result |
|---|---|---|---|
| 0 | P0a | instrumented graphviz | DONE: ~/tools/graphviz-7.0.5-instr/ built; VERIDICAL PROOF PASS (54/54 bit-for-bit == stock, max_rmsd=0 -> logging-only confirmed); GV_TRACE %.17g works. Artifacts in scripts/rng_match/ (build script + patch + README) -- UNCOMMITTED (commit after P0b). Caveats: dot_builtins wrapped as bin/dot (no libltdl); fdp/neato/sfdp work. |
| 0 | P0b | bit-exact harness | DONE + CC-VALIDATED. Baseline: 52 BIT_EXACT, 44 DIVERGENT, 1 CLOSE, 10 no-ref, 8 unavail, 6 err. Foundation committed 2b3efd0. Harness discriminates correctly (tsnet/linlog/kk exact ~1e-16; fa2/graphopt/lgl ~3e-8 BIT_EXACT but SQUEEZABLE to 1e-15; divergent are real). |
| 1 | P1 wave1 + param-match + OGDF | (done) | COMMITTED f60944e. Matched-params authoritative baseline: 60 BIT_EXACT / 36 DIVERGENT / 6 neulay-ERROR / 8 sgd2-UNAVAILABLE / 10 no-ref / 1 CLOSE. Newly bit-exact: gem, pivot_mds, stress_maj, umap. Anti-cheat clean. |
| 2 | P1 wave2 (ports+iterate+depfix) | neato(2440040) sfdp(2440141) fmmm(2440413) maxent(2440612) classical_mds(2440803) drl(2440989) sgd2multi(2441175) neulay(2441364) | RUNNING. Specs PORT2_*/FIX_*.md. Each owns distinct file (drl owns _igraph_rng.py; classical_mds deterministic-no-RNG). On each DONE: anti-cheat grep + check_engine + update STATUS + commit clean. |

## Wave-2 results (partial -- batch-assess when all 8 done; anti-cheat clean so far)
- drl: igraph RNG MOSTLY matched -> 35/42 fixtures bit-exact, 7 diverge (max 1.0). Likely
  chaotic-anneal minority OR a residual RNG-draw-order case. Candidate for 1 targeted retry then accept.
- classical_mds: 1.09 -> 0.77. Bit-exact on non-degenerate fixtures (numerical floor); STUCK on
  igraph vendored-LAPACK dsyevr eigenvector basis for REPEATED eigenvalues (degenerate-eigenvector
  convention) -- genuine numerical wall, NOT RNG. Strong "documented can't-go-further" candidate.
- sgd2multi, neato, sfdp, fmmm, maxent, neulay: DONE (narrative; clean re-verify pid 2481689 running):
  * neato BIT_EXACT 6.07e-16 (42/42) -- convergence-stop fix WORKED.
  * maxent_stress BIT_EXACT 4.78e-16 (42/42).
  * neulay BIT_EXACT 6.63e-16 (42/42) + dep-fix made it RUN (old-code dim/budget bug, not missing PyG).
  * sgd2_multi: dep-fix -> now RUNS (was UNAVAILABLE) -> 0.141 DIVERGENT (needs a PORT next, wave 3).
  * fmmm 1.39->2.08e-2 (35/42 exact; residual = 1-unit integer-coord drift in OGDF final packing).
  * sfdp 0.81->0.44 (partial, multilevel hard). drl 35/42 (partial). classical_mds STUCK (LAPACK degenerate basis).
  * NOTE: status.json got CORRUPTED by 8 concurrent codex check_engine writes -> the single-writer
    re-verify (2481689) regenerates it clean. Anti-cheat grep clean across all wave-2 changes.
- ENDGAME: when all done -> clean full re-verify + anti-cheat grep ALL + commit + classify each
  engine (BIT_EXACT / STUCK-with-documented-reason / dep-fixed). Then wave 3 (davidson_harel +
  sugiyama using drl's _igraph_rng work; squeeze fa2/graphopt/lgl/gem to floor). Then SUMMARY + text JMT.
- DRIVE-OR-WALL: per JMT, each engine either <1e-7 OR a precise documented irreducible reason
  (LAPACK degenerate basis, chaotic FP cascade, non-reproducible source). No flailing, no faking.

## Wave 3 DISPATCHED 2026-06-02 18:03 (committed wave-2 = 33d4f5b, 68 bit-exact)
6 codexes: igraph(2842735: drl+davidson+sugiyama, owns _igraph_rng.py) | sgd2multi(2842848) |
ogdf_finish(2843116: fmmm+maxent) | sfdp(2843316) | fr_fa2(2843528) | classical_mds(2843735 final).
Specs W3_*.md. ACCEPTED WALLS (no codex): gem_iters2000 (chaotic FP at 2000 rounds; bit-exact at
100/500), fcose (no python port -> flag JMT), NO_REFERENCE chains (no ref to compare).
ON ALL 6 DONE: clean full re-verify -> commit -> classify each -> SUMMARY.md -> TEXT JMT (sprint done).
Standing: 68 BIT_EXACT / 2 CLOSE / 41 DIVERGENT / 10 no-ref. Targets after wave3: close the
finishable (fr_steps100 1.86e-7, sgd2_multi_batch128 9.86e-6, fmmm ~0.01, fr_steps200/fa2_linlog
~1e-3) + crack the ports (igraph family, sgd2_multi, sfdp) OR document each wall precisely.

## FINAL ALL-GRAPHS 5-SEED SWEEP (JMT go 2026-06-02 21:40) -- THE comprehensive run
Small-graph sprint DONE: 74/121 bit-exact (SUMMARY.md). Now: 5 seeds x ALL graphs x ALL engines,
current matched-params+RNG-ported+OGDF code -> classify every (graph,algo) combo bit-exact-or-not.
Script scripts/rng_match/final_allgraphs_5seed.sh -> eval_output/benchmark_5seed_final +
eval_output/fidelity_report_final. Wrapper pid 3291157 (NOT 3291150 = transient setsid parent that
gave a false-DONE), watcher btslceav1 (360min, re-arm as needed -- multi-hour run). bg PID gotcha:
setsid spawns a transient parent that exits instantly; the REAL wrapper is the log's PID=$$ -- watch THAT.
ON DONE (fidelity_report_final/report.md): classify per (graph,algo); CONFIRM hypothesis (the 74
small-bit-exact engines stay bit-exact on most graphs, diverge only via chaotic FP cascade on big/
chaotic graphs; the 36 walls diverge regardless). Write final all-graphs summary + TEXT JMT.

## FINALIZATION (small-graph sprint -- wave-3 committed 51d7ebf, SUMMARY.md done, 74/121 bit-exact)
ON re-verify3 DONE (writes STATUS.md): read final verdict counts -> write SUMMARY.md -> TEXT JMT.
NOTE: commits skip detect-secrets via SKIP= (verified TIMESTAMP-ONLY churn, NO real secret; all
other hooks run; NOT --no-verify). status.json gitignored; ogdf_runner binary gitignored.

### Documented IRREDUCIBLE WALLS (for SUMMARY -- these are "genuinely can't get further", precise):
- classical_mds: igraph vendored LAPACK 3.4.2 dsyevr eigenvector basis for DEGENERATE eigenvalues
  (multiplicity>2) is implementation-dependent; SciPy/torch can't reproduce without porting LAPACK
  tridiag-reduction + inverse-iteration. Bit-exact on NON-degenerate fixtures.
- sfdp: compiler/libm-level FP drift in transcendentals, amplified by chaotic multilevel iterations.
- sugiyama: igraph layered tie-breaking on SYMMETRIC graphs (complete5/petersen/wheel/two_triangles).
- drl: igraph RNG mostly matched (35/42 fixtures bit-exact); diverges on a few symmetric/chaotic cases.
- davidson_harel: igraph anneal RNG; diverges on specific fixtures (path8/grid3x3/complete5 seeds).
- fmmm: force-arithmetic order on SYMMETRIC cases before OGDF integer export (35/42; ~0.01-0.02).
- gem_iters2000: chaotic FP cascade at 2000 rounds (bit-exact at 100/500 rounds).
- fcose: NO python port exists (cytoscape fCoSE) -- needs from-scratch port, flag to JMT.
- NO_REFERENCE (10): fr_kk/kk_fr chains, spectral_random_walk/unnormalized, rt_horizontal -- no
  paired reference to compare against (not a fidelity failure, just nothing to match).
### THEME (the honest finding): remaining walls cluster on SYMMETRIC/DEGENERATE small graphs where
the reference's implementation-specific choices (LAPACK degenerate basis, igraph tie-breaking, libm
rounding) are not reproducible in pure python/torch. This is the genuine floor, precisely characterized.

## Held (now wave-3) -- superseded by dispatch above
- davidson_harel, sugiyama (igraph RNG -- wait for drl to settle _igraph_rng.py).
- fr_steps200/500 iterate (fr_steps100 already CLOSE 1.86e-7), fa2_linlog iterate (2.1e-3).
- SQUEEZE bit-exact-but-not-floor: fa2/graphopt/lgl (~3e-8), gem (~8e-8 -> pin exact OGDF commit for ~1e-13).
- fcose (NO python port -- flag to JMT, not a quick fix). NO_REFERENCE chains (fr_kk etc -- can't compare).
- FINAL: clean full re-verify (OGDF rows were mid-rebuild in the f60944e baseline -> refresh), then SUMMARY + text JMT.

## Wave-1 results (as they land)
- **gem DONE 2026-06-02**: NO code change needed. dagua GEM ALREADY bit-matches OGDF at MATCHED
  rounds: 100rounds=3.86e-13, 500rounds=3.93e-08 (both <1e-7!). The baseline "1.15" was a
  HARNESS PARAMETERIZATION artifact: harness runs ogdf_gem ref at DEFAULT 30000 rounds (OGDF
  runner ignores rounds param) vs dagua variant's 100/500/2000. 2000rounds genuinely chaotic
  (1.65e-1, late FP drift). ACTION (CC, central): make ogdf_gem ref honor rounds (runner/variant
  change) so iters100/500 show bit-exact; document iters2000 as chaotic-floor. NO delegation.
  >> LESSON: check reference PARAMETERIZATION -- some "divergences" = ref run at wrong params.

## Wave-1 ALL DONE (PRELIMINARY -- re-verify after param-matching) + anti-cheat CLEAN (no delegation in any)
- gem: already bit-exact at matched rounds (param issue, see above).
- neato: 1.23 -> 7.3e-3. Ported graphviz srand48/drand48 init. Residual = convergence-termination
  (graphviz stops ~152 iters on path8, dagua runs 200). Changed neato default to "graphviz". WAVE-2 iterate.
- davidson_harel: still ~0.36 (rounds50/100/200, worst grid3x3 seed3). NO improvement -- BUT igraph
  ref may be running default params (maxiter) -> RE-VERIFY after param-matching before judging.
- drl: still ~1.0 (34/42 exact). NO improvement -- same caveat (igraph param-matching).
- sfdp: made code changes (no clean number extracted) -- re-verify.
- umap: made code changes -- re-verify.
>> NEXT (when param-matching codex 2186380 DONE): re-run full harness with MATCHED PARAMS ->
   authoritative per-engine numbers. THEN: commit clean ports (grep diff for delegation first),
   classify done/iterate/stuck, dispatch wave 2. HOLD all commits until then.
>> IMPORTANT: davidson_harel/drl "no improvement" is likely the igraph adapter not passing
   maxiter/rounds (gem-style param mismatch). Don't conclude port-failure pre-param-matching.

## OGDF BUILD DONE (2026-06-02 14:54) -- gem FIXED
- OGDF installed ~/tools/ogdf (official tag foxglove-202510, static libs). Runner rebuilt
  (scripts/ogdf_runner) -- now honors --gem-rounds/--fmmm-fixed-iterations/--iterations/
  --number-of-pivots. Build script scripts/rng_match/build_ogdf_runner.sh.
- gem: NOW BIT_EXACT (iters100=7.97e-8, iters500=8.25e-8). param-matching + rebuild fixed it.
  SQUEEZE later: ~8e-8 not ~1e-13 because used official OGDF tag, not the exact 2026 ref commit.
- fmmm: STILL DIVERGENT 1.39 at matched iterations -> genuine PORT needed (wave 2).
- pivot_mds / maxent_stress / stress_maj: runner now honors params -> RE-VERIFY (post-rebuild).
- CAVEAT: harness re-verify (pid 2371127) started 14:47, runner rebuilt 14:54 MID-RUN -> its
  OGDF rows are inconsistent (binary swapped mid-run). MUST re-run OGDF engines cleanly after.
- UNCOMMITTED so far (commit after clean re-verify): param-matching (variants.py + graphviz/
  fcose/ogdf competitors + ogdf_runner.cpp), OGDF build scripts, + wave-1 port pipeline edits.

## DEP/RUNTIME FIX WAVE (JMT: "fix the dependency and runtime errors too" 2026-06-02)
Make blocked engines actually evaluable on small graphs -- FIX, don't just document:
- sgd2_multi (8 variants UNAVAILABLE): s_gd2 IS installed and `sgd2` ref works, but
  `sgd2_multi_ref.available()` returns False for another reason -> diagnose its available()
  gate + fix so it runs on small graphs.
- neulay (6 variants ERROR): ref is available() + imports fine -> errors are RUNTIME (likely
  GNN-train timeout or exception on some graphs) -> diagnose actual error + fix.
- Plus any GENUINE ERROR remaining after the clean post-param-matching re-verify (sfdp showed
  ERROR in the racy mid-flight STATUS -- likely transient; confirm on clean run).
DISPATCH AFTER param-matching lands (it edits adapters incl. sgd2 -> conflict if parallel).
fcose (cytoscape, 2): genuinely NO python port -> that's a missing-port, not a runtime error;
separate (would need a from-scratch fcose port -- flag for JMT, not part of dep-fix).

## Work list for later waves (after wave 1 validates the loop)
DIVERGENT engines not yet dispatched: sugiyama (igraph/graphviz_dot), classical_mds (igraph_mds),
fmmm/fdp (ogdf+graphviz), maxent_stress + stress_maj (ogdf_stress), pivot_mds (if divergent),
+ SQUEEZE the BIT_EXACT-but-3e-8 ones (fa2, graphopt, lgl) toward 1e-15.
NO_REFERENCE (10) + UNAVAILABLE (8: sgd2_multi ref too slow) + ERROR (6: neulay timeout) =
likely cannot be matched on small graphs; document in final SUMMARY.
