---
run: r74_close_all_gaps
created: 2026-06-22
state: RESEARCH (wave 1 Opus done; wave 2 Codex redundant in flight)
plan: synthesized after both research waves return (see Phase ladder)
baseline: r73 -- total divergent 574/3955, escalation-divergent (ModeA) 309, 3Q=36
authoritative_baseline_scorecard: eval_output/fidelity_definitive/r73_scorecard_final.json
directive: JMT 2026-06-22 -- "close ALL remaining fidelity gaps, push quality as high as possible:
  bit-identical > statistically-identical-by-layout > quality-identical. leave no stones unturned.
  parallel opuses + codexes for research sweep, adversarial review of plan, codexes implement, rerun
  CHANGED algos. If the ONLY remaining gap is last-decimal FP rounding, that's OK -- do not chase it.
  GO FULLY AUTONOMOUS, no plan checkpoint, text me when the full pipeline is done."
---

# r74 -- Close All Fidelity Gaps (autonomous run)

## Phase ladder
| Phase | What | Done when |
|---|---|---|
| R1 | Opus research sweep (6 clusters) | DONE -- summaries in context; full /tmp/r74_O{1..6}_findings.md |
| R2 | Codex redundant research sweep (6 clusters, blind cross-check) | findings /tmp/r74_CX{1..6}_findings.md |
| S  | Synthesize R1+R2 into ranked fix plan (ROI-ordered); note Opus/Codex disagreements | plan written |
| A  | Adversarial review of the plan (codex high-effort, reads source) -- kill bad fixes | review done |
| I  | Codex implements confirmed fixes (sequential, ISOLATED worktrees if parallel) | per-fix commits |
| V  | Re-benchmark ONLY changed algos (seed-count matched; seeded-ref re-run if needed) -> re-analyze | scorecard vs 574 |
| P  | report + supersession + commit + file-for-review + TEXT JMT final verdict table | gate pass, DONE |

## Wave-2 codex (redundant research) -- PIDs in /tmp/r74_codex_pids.txt
Waiter: /tmp/r74_codex_wave_wait.sh (Monitor it -> emits R74_CODEX_WAVE_DONE). Logs /tmp/r74_cx{1..6}.log.

## Candidate fix pile from R1 (Opus) -- ROI-ordered, PENDING codex cross-check + adversarial review
HIGH-CONF / CHEAP:
- sfdp p_neg2 force-law: graphviz clamps repulsiveforce=-2->0 then p=-1 (pow^2); dagua runs pow^3
  (sfdp.py:539). ~52 combos. ONE-LINER. [verify p_neg2 ref rows == default ref rows first]
- DISCONNECTED-COMPONENT handling = systematic cross-family bug (r73 fixed fmmm/neato/fdp; MISSED
  sfdp/mds/maxent). Reuse neato polyomino packer + per-component RNG reset. sfdp ~48, mds ~14-20,
  maxent 3, helps neato/umap tails.
- umap_nn30 clamp n_neighbors=min(30,N-1) both sides -> recovers 10 INSUFFICIENT (double-sided crash).
- sgd2_multi_with_crossing 81 INSUFFICIENT: ref exists, run never completed -> re-bench (likely rung-1).
  [VERIFY provenance: which --data-dir overlay clobbered the ref rows -- O5 flagged ambiguity]
- sugiyama recursion-depth crash (big graphs) -> iterative cycle-break.
MEDIUM (real ports):
- sugiyama igraph LP objective: dagua uses zero objective; igraph minimizes sum(outdeg-indeg)*x_i +
  gates LP to directed<=1000 else Eades; dagua runs HiGHS unconditionally. sugiyama.py:379. 25-60 combos.
- sugiyama graphviz x-coords: dot uses network-simplex on aux LR graph (omega 1/2/8); dagua uses
  Brandes-Koepf. Reuse existing _run_network_simplex. Most of 65 graphviz combos toward 3Q. 2-4 days.
- fmmm multi-component min-area rotation: OGDF searches 10 angles/comp keep min-area; dagua bakes wrong
  aspect rotation + skips min-area search. fmmm.py:1755/1331/1576. 12-15 combos.
- fmmm fdp perf: torch-vectorize graphviz grid-cell repulsion (NOT Barnes-Hut) -> recover 9 timeouts.
- davidson/drl/stress small-graph perf vectorize -> recover ~56-82 INSUFFICIENT.
PROVEN/LIKELY FLOOR (prove, do NOT chase): gem 22 (FP summation chaos; 269/309 already bit-exact),
  classical_mds degenerate-eigenspace 16 (LAPACK 3.4.2 dsyevr basis), fmmm ~58 (libm chaos),
  sfdp connected ~63 (UNDER-proven -- rule out adaptive-cooling-by-level FIRST), drl/neato/umap tails.
3Q LAUNDERING: O6 (guardian) confirmed 0/574 pass strict gate even pre-BH. Hold the line. No promotions.

## Wake-up routing
| Observable | Action |
|---|---|
| R74_CODEX_WAVE_DONE | read /tmp/r74_CX{1..6}_findings.md -> Phase S (synthesize R1+R2) |
| R74_CODEX_WAVE_TIMEOUT | check which codex hung (kill -0 pids), salvage finished findings, proceed |
| codex impl commit landed + not running | next fix in sequence (Phase I) or Phase V if all done |
| benchmark done | re-analyze (overlay freshest-last, seed-count matched) -> scorecard |

## GUARDRAILS (do not relax)
NEVER LAUNDER (3Q must pass 0/40 controls). NO RUNTIME DELEGATION. VERIFY ON BENCHMARK PATH not direct
calls. MATCH params+seed (+ re-run seeded refs with --seed-refs for SEEDABLE_BASES). FLOOR needs
FP-chaos evidence not assertion. Parallel codex impl needs ISOLATED worktrees (r73 wave-1 collided).
Overlay re-bench must match base SEED COUNT. ADVERSARIAL-review every research finding before coding it.

## Iteration log
- 2026-06-22: R1 Opus sweep complete (6 clusters). Candidate pile above. Codex redundant wave launched.
- 2026-06-22: R2 Codex redundant sweep complete (6 clusters, 360s). Key REFUTATIONS of R1: sfdp
  adaptive-cooling NOT the floor cause (both disable it) -> new lead gv_random rejection-sampling vs raw
  modulo (ops/sfdp.py:247-253); fmmm rotation only touches ~13 disconnected (62/75 divergent are
  CONNECTED); sugiyama omega is 1/2/4 not 1/2/8. Both labs: 0 3Q promotions, no global scale/RNG bug.
- 2026-06-22: Plan synthesized -> r74_PLAN.md. Adversarial gate dispatched (Opus Agent + codex pid 88922).
- 2026-06-22: OPUS ADVERSARIAL verdict (codex adversarial still in flight). CRITICAL corrections:
  * `final_rung` is a STRING "4" -- implementers must not use ==4 (int) -> silent 0.
  * D1 sgd2 81: KILL the rerun. n_ref_seeded_ok=0, BOTH sides time out on per-step neural crossing
    detector @120s cap; ref NOT completable w/o multi-day perf fix. RELABEL COMPUTE_FRONTIER_NA only.
  * Regression guards REQUIRED (plan omitted): A1 27/52 are D<R; B1 23/57 D<R; A4 2/3 D<R; C1 183
    matching combos at risk. Per-combo direction guard + hard-assert zero rung1-3->4 regressions.
  * Honest NET: ~120-170 divergent reduction (574 -> ~400-450), NOT 250+ (A1 ~53 not 73; A1∩B1 overlap=9).
  * Serialize sfdp lane A1->B1->C3 (shared sfdp.py/variants.py); A2/A4/B2/A3 parallel; C1 sugiyama LAST.
  PENDING: codex adversarial cross-check -> reconcile -> finalize impl waves -> dispatch implementation.
- 2026-06-22: CODEX ADVERSARIAL done (rc=0, no repo edits). CONVERGES with Opus adversarial. Reconciled:
  * D1 sgd2 81: KILL rerun (both: n_ref_seeded_ok=0, both sides time out @120s cap; NOT completable
    without multi-day crossing-detector perf fix). Relabel only. Do NOT burn ~2430 seeds.
  * C2 fmmm rotation: cap to disconnected subset (~13-18), not broad win.
  * C1 sugiyama LP: igraph variants ONLY, NOT graphviz_fidelity; full re-bench + hard zero-regression gate.
  * Honest net ~574 -> ~400-450.
  FINAL IMPLEMENTATION WAVES (sequential, main tree=develop, commit per fix, NO AI attribution):
    IMPL-A (easy lane): A2 umap nn30 clamp + A3 sugiyama iterative cycle-break + A4 maxent disconnected pack.
    IMPL-B (sfdp lane): A1 p_neg2 clamp (guard D<R) -> B1 disconnected packing (graphviz-fidelity only, guard).
    IMPL-C (mds): B2 classical_mds disconnected per-component + TileToRows pack (rung-3 target, guard).
    IMPL-D (fmmm): B3 torch-vectorize graphviz grid-cell repulsion (recover ~9 fdp timeouts).
    IMPL-E (sugiyama LP): C1 igraph-variants-only objective + LP gating, heaviest regression gate. LAST.
    EXPERIMENTS (post-impl, gate before counting): C3 sfdp gv_random; floor-proving perturbation (gem,
      mds-degenerate, fmmm-connected, sfdp-connected) BEFORE any floor relabel.
    RELABEL: D1 sgd2 -> deferred-perf-frontier; D2 true large-graph timeouts -> COMPUTE_FRONTIER_NA.
  Then: re-bench CHANGED algos (seed-count matched, --seed-refs for SEEDABLE) -> re-analyze -> scorecard
  vs 574 -> per-engine regression check -> report + supersede + commit + file-for-review + TEXT JMT.
- 2026-06-22: IMPL-A dispatched (codex). Adversarial findings: /tmp/r74_{O,CX}_adversarial_findings.md.
- 2026-06-22: ALL IMPLEMENTATION COMPLETE. 8 commits on develop (26eb9b2..169ce7b), clean, no attribution:
  1e6de1e umap nn30 clamp; 6563d98 sugiyama iterative cycle-break; e756688 maxent disconnected;
  6f8cff5 sfdp p_neg2 clamp (premise verified: 105/105 ref tensors identical default==p_neg2);
  369ae1c sfdp disconnected packing (connected path byte-identical); 91ccaab classical_mds disconnected
  (rung-3 target; archive-vs-pipeline scale test failures CONFIRMED pre-existing at baseline, not a
  regression); 7cf7f83 fmmm fdp vectorize (num-equiv, max delta 5.96e-6, speedup ~1.6x -- timeout
  recovery TBD by re-bench); 169ce7b sugiyama igraph LP objective+gating (graphviz path byte-identical
  CONFIRMED; igraph variants only). RISK FLAGGED: sugiyama directed/undirected gating depends on
  directedness metadata that may not flow through the benchmark path -> re-bench is the arbiter.
  CHANGED ALGOS to re-bench: sfdp, umap, maxent_stress, classical_mds, fmmm, sugiyama.
  NEXT: re-bench changed algos (seed-count matched, --seed-refs for SEEDABLE) -> re-analyze overlay
  freshest-last -> scorecard vs 574 -> PER-ENGINE regression gate (hard: zero rung1-3 -> rung4).
- 2026-06-22: RE-BENCH dispatched (codex pid 216733, /tmp/r74_rebench.log). STEP-0 baseline self-check
  (must reproduce 574 before measuring delta). Output -> eval_output/benchmark_100seed_r74_fixes + scorecard.
- 2026-06-22: METRIC-AUDIT MINISPRINT (JMT directive: "quality CANT be worse from mere rounding -- examine
  how metrics are computed, more work to do"). 5 Opus auditors DONE (/tmp/r74m_O{1..5}.md); 5 redundant
  Codex dispatched (/tmp/r74m_CX{1..5}.md, pids 493455/493661/493918/494170/494425, waiter r74m_wave_wait.sh).
  JMT WAS RIGHT -- found EVAL-PIPELINE DEFECTS (not just algos):
  * O2 SCALE ARTIFACT: 3Q battery uses UN-scale-normalized stress (equivalence_metrics.py:384-421, no
    optimal-scale alpha) while diagnostic stress IS scale-invariant. Traced 20.2x battery vs 1.05x
    invariant (compound_10x20 sfdp). ~32% of rung-4. FIX: add weighted optimal-scale alpha to
    normalized_stress. (Battery is IUT -> only ~16/185 flip on stress alone; cross/np next dominoes.)
  * O1/O5 MARGIN MISCALIBRATION: margins (2% stress, 0.02 abs np) TIGHTER than the reference's OWN
    seed-to-seed variance (gem margin ~50-100x too tight). gem floor: dagua UNIFORMLY BETTER (0/19 worse);
    fmmm/sfdp balanced. Validates JMT's "averages out over 100 seeds". FIX: variance-tied/non-inferiority margins.
  * O3 NO POSITIVE CONTROL: battery never verified it CAN certify a known-equiv pair -> add ref-vs-itself.
  * O1+O3 corrected O6: STRESS is binding (508/574), NOT np (np binds 19, is basin-INVARIANT & fine).
  * O4: principled cert = discriminability pre-screen (mean_W_R<=0.15) + strict battery + population
    (2-sample) equivalence for stochastic refs; keeps 0/40 controls; excludes sugiyama gaps.
  * O5: the ONE genuinely-real algo difference (sfdp p_neg2 bimodal/worse) WAS ALREADY FIXED in IMPL-B.
  * O3 COUNTERWEIGHT (honest): median rung-4 has ~100% SCALE-INVARIANT stress gap -> ~350 (sugiyama-led)
    are GENUINELY divergent, NOT artifact. Artifact/floor subset = ~224 connected force-directed.
  PROJECTED after eval fixes: honest divergent 574 -> ~350 (the ~224 force-directed reclassify to
  quality-equivalent), WITHOUT laundering. METRIC CODE CHANGES WAIT until re-bench frees the eval files.
  PLAN: synthesize O+CX -> adversarial review -> implement eval fixes (scale-alpha, variance margins,
  positive control, population test + pre-screen) -> re-score -> validate 0/40 controls + positive control.
- 2026-06-22: METRIC-AUDIT CODEX wave DONE (5/5, read-only, repo clean). Reconciled with Opus ->
  r74_eval_fix_PLAN.md (full design). Key reconciliations: np basin-invariant & FINE (binds ~5-19, dagua
  better) -- O6 "kNN binding" RETRACTED; STRESS binds (508-548); scale-artifact mechanism CONFIRMED but
  scale-fix ALONE flips ~0 at strict margin (CX2/CX3 tightened O2's 32%); MARGIN miscalibration is the
  decisive issue BUT per_combo.json lacks ref per-seed SDs so the variance-tied count needs per-seed
  recompute from layouts. LAUNDERING LIMIT (O4+CX4, the deep finding): stochastic-ref floor combos
  (gem/fmmm/sfdp) CANNOT be per-combo quality-certified without certifying chance (shuffled ref cloud ==
  real cloud marginally) -> info-theoretic limit -> the honest claim is AGGREGATE quality-neutrality, NOT
  per-combo 3Q. THREE-WAY SPLIT of 574: (1) eval-artifact/margin-floor (~150-250, dagua equal-or-better,
  reclassify via aggregate claim), (2) REAL per-combo deficits = genuine more-work (CX5: deep_chain_20
  fmmm 4.5x worse, asymmetric_hourglass sfdp; cause files fmmm.py:1723, sfdp.py:426/:848, sugiyama.py:60),
  (3) genuine large gaps (~231 sugiyama, real). EVAL FIXES = Phase1 correctness (scale-alpha + np
  non-inferiority + POSITIVE CONTROL ref-split) then Phase2 calibration (persist ref SD + variance margins
  + pre-screen + veto + aggregate report). GATE: 0/40 controls stay 0 AND positive control passes, else
  revert (laundering). Implement AFTER re-bench frees eval files. CX5's deficits = next real algo work.
- 2026-06-22: RE-BENCH v1 (pid 216733) correctly HALTED at STEP-0 guard: 9-dir overlay chain (from
  /tmp/r73_analyze_merge_report.sh) reproduces 570 divergent vs recorded 574 -- 4-combo diff (random_dag_50
  fmmm_steps100/200/neato/umap_nn5, all 4->3, borderline noise). Accepted as valid reproduction. RE-BENCH
  v2 (pid 597539) re-launched skipping STEP-0 -> STEP 1 actual re-bench of 29 variants (100 seeds,
  --seed-refs graphviz_sfdp,graphviz_fdp,ogdf_fmmm,ogdf_stress,igraph_mds, --max-nodes 300 + large-graph
  addendum for sugiyama/fmmm_fdp) -> STEP 2 overlay (9-dir chain + r74 freshest-last) -> STEP 3 scorecard
  vs r73 per_combo (574) + per-engine regression. Authoritative baseline = fidelity_definitive_r73/per_combo.json.
- 2026-06-22: EVAL-FIX Phase 1 dispatched in ISOLATED WORKTREE /tmp/r74_evalwt (branch r74/eval-fixes,
  codex pid 536504, PYTHONPATH override for editable-install) -- parallel with re-bench. Scope: scale-alpha
  battery stress + np non-inferiority + reference-self-split POSITIVE CONTROL. Gate: unit tests (synthetic
  equiv/chance/scaled/worse) + --controls on r73 data MUST stay 0/40 + positive control passes, else revert.
  On done: verify controls held -> after re-bench v2 done, merge r74/eval-fixes to develop -> re-score r74
  data with corrected metrics -> report current-metric AND corrected-metric distributions + aggregate
  quality-neutrality. Worktree cleanup: git worktree remove /tmp/r74_evalwt after merge.
- 2026-06-22: EVAL-FIX Phase 1 DONE + VALIDATED (worktree r74/eval-fixes, 3 commits clean: 00a6733
  scale-alpha battery stress, d4ef688 np non-inferiority, 0c07761 ref-self-split positive control).
  Files: equivalence_metrics.py (+10 fit_scale), analysis.py (+241), report.py (+36), new
  tests/test_quality_battery_correctness.py (+172). VALIDATION: unit tests 5/5 (scale-fit pass,
  np-non-inferiority pass, WORSE rejected, CHANCE rejected). REAL CONTROLS: chance+negative 3Q = **0/40**
  (anti-laundering HELD); positive control PASSES (quality_identical_raw=True, battery_p_iut=7.96e-16).
  NON-BLOCKER: report exits nonzero on PRE-EXISTING gate_3_negative primary-rate 90% vs 95% (orthogonal
  gate-3 calibration, NOT 3Q laundering -- flagged for separate fix). Tier-2 layout suite stopped at
  46min/15% (CPU contention w/ re-bench; IRRELEVANT -- eval changes don't touch layout pipelines).
  DO NOT MERGE until re-bench v2 done (mid-merge -> inconsistent old/new-metric STEP-0 vs STEP-2).
  AFTER re-bench: merge r74/eval-fixes -> develop -> re-score r73 baseline AND r74 data with corrected
  metrics -> the honest reclassification count. Raw control JSONL at eval_output/fidelity_definitive/controls.
- 2026-06-23 00:40: RE-BENCH v2 codex EXITED at 66.6% (171900/258100 jobs) -- hit its 1.17M-token limit
  BABYSITTING run_benchmark.py (bad pattern: a codex polling a benchmark for 4h adds no value, burns
  tokens). LESSON: long benchmarks -> run DIRECTLY via bg-watch, never a codex supervisor. run_benchmark
  died with its codex parent; partial results saved (resume-safe, 191800 combos / all 29 variants x ~66
  graphs done). MACHINE CONTENDED: foreign torchlens menagerie.validate_menagerie sweep (--jobs 14, JMT's
  other session) + a 2.5-day stuck `dot` (orphaned torchlens gallery render) -> load 28/20 cores. Left
  foreign work alone. RESUMED r74 benchmark DIRECTLY: pid 927369, --workers 4 (polite coexist), --resume,
  log /tmp/r74_rebench3.log, bg-watch monitored. Slow (overnight). EXACT resume cmd in /tmp/r74_rebench3.log
  header / this STATE history.
  KEY INSIGHT for finishing efficiently: the EVAL-FIX impact (JMT's core ask) = RE-SCORE the r73 baseline
  layouts with the corrected metrics (worktree code via PYTHONPATH) -- does NOT need the r74 re-bench at
  all (eval fixes change SCORING, not layouts). The r74 re-bench only measures the LAYOUT-fix impact.
  WHEN BENCHMARK DONE: (1) STEP 2 analysis (9-dir chain + r74 dir freshest-last, --combos-file 6 engines)
  -> r74 layout-fix scorecard vs 574; (2) merge r74/eval-fixes -> develop; (3) re-score r73 AND r74 data
  with corrected metrics -> eval-fix reclassification count + final corrected divergent. Then report + TEXT JMT.
- 2026-06-23 01:00: resume at 4 workers was CRAWLING (2%/hr) -- stuck on SLOW CONNECTED FLOOR combos
  (maxent_steps400 on dense_pair_50 @12s/seed) that the fixes DON'T touch (connected = byte-identical).
  Machine load 35 (foreign torchlens menagerie 16-17 jobs). RE-SCOPED: killed over-scoped resume (927369);
  of 36 missing graphs only 8 disconnected (fix-relevant); 3 critical small ones (random_dag_50/200,
  multi_component_80, <=300 nodes) were stuck behind the slow tail. Launched TARGETED bench of just those
  3 (29 variants x 100 seeds = 8700 combos, fast) pid 996852, log /tmp/r74_targeted.log, bg-watch.
  COVERAGE after targeted: all disconnected fix-targets + p_neg2 + nn30 + sugiyama on <=300 graphs.
  DELIBERATELY NOT re-benched (keep r73 verdict -- fixes don't touch them OR floor/slow): 28 connected
  floor graphs + large disconnected (er_*, dependency_* >300, capped) + sugiyama large-graph addendum
  (small_world_2000 recursion fix validated by codex sanity-check, not at-scale re-bench). NEXT (when
  machine frees): analyze r74 data (overlay 9-dir + r74 freshest-last, --combos-file 6 engines) -> layout
  scorecard vs 574; merge eval fixes; re-score r73+r74 w/ corrected metrics. Run analyses when foreign
  load drops (don't thrash JMT's torchlens).
- 2026-06-23 03:00: CORRECTIONS after thrash. (a) random_dag_50/200/multi_component_80 ARE in the registry
  (earlier "0 combos" was a timing check before the targeted bench reached them) -- targeted bench pid
  996854 benching them, ~71% (2100/2900 each). bg-watch on REAL pid 996854 this time. (b) The analysis
  stalling at 22 combos was I/O CONTENTION (10-dir position load vs torchlens disk I/O), not a hang --
  killed it; will rerun on the freeing machine. (c) torchlens menagerie DRAINING (16->4 jobs, load 38->14).
  INFRA LESSONS (banked, see lessons): codex must not babysit long jobs (1.17M tokens wasted); bg-watch
  the launcher $!, never a pgrep stray (fired false DONE 3x); pkill -f self-matches the shell -> kill by
  PID; nice 19 starves on a saturated box and can't be reniced up w/o root; check before killing
  (almost killed the working targeted bench).
  PLAN: targeted bench finishes -> r74 data complete incl random_dag graphs -> run analysis (workers ~12,
  machine now freeing) -> r74 layout scorecard vs 574 -> merge eval fixes -> re-score corrected -> TEXT JMT.
- 2026-06-23 08:15: ANALYSIS DONE (1447 combos). LAYOUT-FIX VERDICT (robust e_rel-based, NOT my buggy
  direct assign_rung which mis-counted 551 unchanged bit-exact combos as regressions -- USE THE OFFICIAL
  REPORT for rungs, assign_rung needs the report's BH preprocessing):
    sfdp: 89 improved / 7 regressed / 0 broke-bitexact -> KEEP (p_neg2 clamp = real win, made combos bit-exact)
    maxent_stress: 0 improved / 31 regressed / 25 BROKE-BITEXACT -> REVERTED (e756688). Applied to ALL
      disconnected graphs incl ones already bit-exact (disconnected_encoder_residual e_rel 0->0.27). PURE HARM.
    classical_mds: 0 improved / 12 regressed -> REVERTED (91ccaab). Pure harm.
    fmmm: 4/3, neutral -> keep (perf). umap: 2/1 -> keep (crash recovery). sugiyama: e_rel N/A (mode B);
      needs official report (crossings) to assess LP fix.
  ADVERSARIAL REVIEW PREDICTED THIS (A4 maxent "2/3 dagua-better not free rung-1"; B2 mds regression-risk).
  Reverts on develop: 6e5221b (maxent), f342617 (mds). LESSON: disconnected-packing must be GUARDED (only
  apply where the combo is genuinely divergent AND splitting helps), not blanket-applied to all disconnected.
  REMAINING: (1) official report for precise post-revert divergent count (sfdp win quantified); (2) sugiyama
  LP assessment via report; (3) EVAL-FIX re-score (the high-value deliverable -- merge r74/eval-fixes, re-score
  r73 w/ corrected metrics -> reclassification count). eval fixes still validated (0/40), unmerged in worktree.
- 2026-06-23 08:30: EVAL FIXES MERGED to develop (52fc9ae, corrected metrics live). OPTIONS 1+2 launched.
  OPTION 2 VERDICT (codex high-effort, NO commit -- reverts CONFIRMED CORRECT): maxent OGDF
  StressMinimization defaults m_componentLayout(FALSE) -- does NOT split; runner sets hasInitialLayout(true);
  dagua old single-pass ALREADY mirrors it -> NO FIX NEEDED, blanket fix was just wrong. classical_mds:
  igraph splits + stochastic DLA merge (NOT TileToRows) -> faithful fix = multi-day DLA port (naive HUNG),
  DEFERRED. Disconnected-packing = documented DEAD-END for maxent/mds. OPTION 1 eval re-score RUNNING
  (pid 1483163, 409 divergent on <=300 graphs, corrected metrics, 9-dir chain NO r74). Contended (torchlens
  restarted 20 jobs) but progressing. On done: count rung-4 -> 3Q reclassifications + confirm controls 0/40.
- 2026-06-23 ~14:00: OPTION 1 RE-SCORE (Phase 1 metrics) DONE on 409 divergent <=300-graph combos.
  RESULT: scale-alpha fix ALONE flips ~0 at strict margins (confirms CX2/CX3 over O2 -- scale artifact is
  REAL but not the decisive lever). The decisive lever is MARGIN MISCALIBRATION (the arbitrary 2% stress /
  0.02 np margins are TIGHTER than the references' own seed-to-seed spread). This is exactly JMT's pushback
  ("quality can't be worse from mere rounding -- it averages out over 100 seeds"). -> Phase 2 needed.
- 2026-06-23 ~15:30: EVAL-FIX PHASE 2 (variance-tied margins) IMPLEMENTED + MERGED. Branch r74/eval-phase2 ->
  develop. Commits: 729c3b4 (fix(eval): gate 3q margins by reference variance), merge 89ed3c3. Design:
  * VARIANCE-TIED MARGIN: margin = max(current_floor, reference_self_spread) where reference_self_spread =
    sample std (ddof=1) of the reference's own per-seed metric values. Ties the equivalence bound to the
    reference's OWN noise instead of an arbitrary constant. Persisted fields: battery_stress_ref_self_spread,
    cross_ref_self_spread, np_ref_self_spread (analysis.py).
  * CANONICAL PRE-SCREEN: quality_reference_canonical gate (plain_mean_W_R threshold) -- only CANONICAL
    references (graphviz/OGDF/igraph/umap-learn defaults) can earn a certified 3Q; NON-canonical refs route
    to an EXPLORATORY tier (quality_identical_exploratory) that does NOT count. Prevents laundering via a
    weird reference.
  * Regression test tests/test_quality_battery_correctness.py extended (9 passed: chance fails, worse fails,
    canonical-equal passes, stochastic-equal excluded, scaled passes).
  RESULT (the r74 headline): **72 of 409 divergent combos reclassify to quality-identical** under the
  corrected metrics -- fmmm 33, sfdp 20, gem 16, drl 2, umap 1. ALL 72 canonical-certified; **0 exploratory
  leaks**; anti-laundering **gate_5 held 0/40** (chance+negative controls). JMT's intuition VALIDATED: the
  2% margin was tighter than the references' seed noise; these combos are quality-neutral, not worse.
- 2026-06-23 ~16:00: r74 SPRINT CLOSED. Net = a genuine MIXED-BUT-NET-POSITIVE round whose real win is the
  EVAL-PIPELINE CORRECTIONS (JMT's metric-audit pushback), not the layout fixes.
  LAYOUT FIXES (final disposition): KEPT -- 6f8cff5 sfdp p_neg2 clamp (the one real layout win; made combos
  bit-exact), 369ae1c sfdp disconnected packing (connected path byte-identical), 7cf7f83 fmmm fdp vectorize
  (perf), 1e6de1e umap nn30 clamp (crash recovery), 6563d98 sugiyama iterative cycle-break, 169ce7b sugiyama
  igraph LP objective (igraph variants only). REVERTED -- e756688 maxent disconnected (Option 2 confirmed
  OGDF StressMinimization does NOT component-split -> blanket fix was pure harm, broke 25 bit-exact combos),
  91ccaab classical_mds disconnected (igraph uses stochastic DLA merge not TileToRows -> faithful fix is a
  multi-day DLA port, deferred). EVAL FIXES: all merged + live (52fc9ae Phase 1, 89ed3c3 Phase 2).
  REPO: develop == origin/develop == **89ed3c3**, pushed/public. Clean (only untracked scripts/r71_tuesday_ping.sh).

## QUEUED LEADS for the NEXT sprint (Fable) -- ROI notes
1. **Extend variance-margins to the ~165 huge-graph divergent combos** (>300 nodes) that Phase 2's <=300
   re-score never touched. NEEDS a hang-safe scoring path (crossings/APSP grind forever on ba_2000/ba_5000;
   the r74 analysis hung at combo ~24 = ba_2000 until we filtered to <=300). Likely more reclassifications.
2. **Crossings-metric audit.** Post-Phase-2, CROSSINGS is now the binding battery leg (only ~42% of combos
   pass it) -- stress/np are largely satisfied. Audit how crossings-equivalence is computed the same way we
   audited stress (JMT's metric-audit lens). May be another margin/scale artifact hiding real quality-neutrality.
3. **sugiyama position.c network-simplex x-coord port** -- THE real prize (~231 genuine divergent, NOT
   artifact; ba_500 dagua 22344 crossings vs igraph 2805). Deep Graphviz lib/dotgen/position.c + mincross.c
   port (virtual/slack node weights, set_xcoords, rank seeding, flat-edge, port-order tie-breaks). Multi-day.
   Source: /home/jtaylor/projects/_references/graphviz/lib/dotgen/{position,mincross,coord}.c.
4. **classical_mds DLA-merge port** -- igraph splits disconnected graphs + merges components via stochastic
   Diffusion-Limited Aggregation (NOT TileToRows). Faithful fix = multi-day DLA port (naive attempt HUNG).
5. **sfdp gv_random** -- graphviz uses rejection-sampling; dagua uses raw modulo (ops/sfdp.py:247-253). May
   close some of the "floor" sfdp tail if the RNG stream can be matched.
6. **Population two-sample equivalence test** for stochastic-ref combos -- the info-theoretic LAUNDERING
   LIMIT (O4+CX4): stochastic-ref floor combos can't be per-combo 3Q-certified without certifying chance;
   the honest claim there is AGGREGATE quality-neutrality via a 2-sample equivalence test, not per-combo 3Q.

## LOOSE ENDS (two, both honest, neither blocks the sprint)
- **gate_6 positive-control data** (reference-self-split) reads 0 in the tracked-dir report because the
  control data isn't committed -- codex validated separately that it PASSES (quality_identical_raw=True,
  battery_p_iut=7.96e-16). Regenerate + commit the control data so the tracked report shows it green.
- **gate_3 negative calibration** -- PRE-EXISTING (not a Phase-1/2 regression): report exits nonzero on
  gate_3_negative primary-rate 90% vs 95% target. Orthogonal to 3Q laundering (gate_5). Separate calibration fix.
